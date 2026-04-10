//! Vulkan 1.2+ renderer.
//!
//! Shader source: src/backends/render/dvui.{vert,frag}
//! Build compiles them to SPIR-V via glslc; they are embedded as `vert_spv` / `frag_spv` modules.
//!
//! Usage per-frame:
//!   After vkCmdBeginRenderPass, call `renderer.setFrame(cmd, extent)`,
//!   then `win.begin()` / dvui widgets / `win.end()`.
//!   dvui owns no command pool; it records into the command buffer you provide.

const std = @import("std");
const Allocator = std.mem.Allocator;
const dvui = @import("dvui");
const vk = @import("vk");
const Instance = vk.InstanceProxy;
const Device = vk.DeviceProxy;
const log = std.log.scoped(.dvui_vulkan);

pub const kind: dvui.enums.RenderBackend = .vulkan;

const img_format: vk.Format = .r8g8b8a8_unorm;

const color_subresource = vk.ImageSubresourceRange{
    .aspect_mask = .{ .color_bit = true },
    .base_mip_level = 0,
    .level_count = 1,
    .base_array_layer = 0,
    .layer_count = 1,
};

/// Heap-allocated GPU resources per texture.  dvui.Texture.ptr points here.
const GpuTexture = struct {
    image: vk.Image,
    memory: vk.DeviceMemory,
    view: vk.ImageView,
    sampler: vk.Sampler,
    descriptor: vk.DescriptorSet,
};

const Self = @This();

allocator: std.mem.Allocator,
instance: *const Instance,
device: *const Device,
physical_device: vk.PhysicalDevice,
graphics_queue: vk.Queue,
/// Transient pool used for one-time texture-upload commands.
cmd_pool: vk.CommandPool,

pipeline: vk.Pipeline,
pipeline_layout: vk.PipelineLayout,
dsl: vk.DescriptorSetLayout,
desc_pool: vk.DescriptorPool,

/// Host-visible+coherent vertex buffer, persistently mapped.
vtx_buf: vk.Buffer,
vtx_mem: vk.DeviceMemory,
vtx_mapped: [*]dvui.Vertex,
vtx_cap: u32,

/// Host-visible+coherent index buffer, persistently mapped.
idx_buf: vk.Buffer,
idx_mem: vk.DeviceMemory,
idx_mapped: [*]dvui.Vertex.Index,
idx_cap: u32,

/// 1×1 white texture bound when dvui draws with no texture (sampling returns (1,1,1,1)).
null_tex: *GpuTexture,

/// Set by the app each frame before win.begin(); see setFrame().
frame_cmd: vk.CommandBuffer = .null_handle,
frame_extent: vk.Extent2D = .{ .width = 0, .height = 0 },

/// Write heads into vtx/idx buffers; reset to 0 each begin().
vtx_head: u32 = 0,
idx_head: u32 = 0,

pub const InitOptions = struct {
    instance: *const Instance,
    device: *const Device,
    physical_device: vk.PhysicalDevice,
    graphics_queue: vk.Queue,
    graphics_queue_family: u32,
    /// Render pass dvui will record into.  Must have ≥1 color attachment.
    render_pass: vk.RenderPass,
    /// Pre-allocated vertex slots.  Grow this if you see overflow log messages.
    vertex_capacity: u32 = 65536,
    /// Pre-allocated index slots.
    index_capacity: u32 = 131072,
};

pub fn init(allocator: std.mem.Allocator, opts: InitOptions) !Self {
    var self: Self = undefined;
    const dev = opts.device;

    // --- Descriptor set layout: binding 0 = combined image sampler ---
    const dsl_binding = vk.DescriptorSetLayoutBinding{
        .binding = 0,
        .descriptor_type = .combined_image_sampler,
        .descriptor_count = 1,
        .stage_flags = .{ .fragment_bit = true },
        .p_immutable_samplers = null,
    };
    const dsl = try dev.createDescriptorSetLayout(&.{
        .binding_count = 1,
        .p_bindings = @ptrCast(&dsl_binding),
    }, null);
    errdefer dev.destroyDescriptorSetLayout(dsl, null);

    // --- Pipeline layout: set 0 + push constants (mat4 = 64 bytes) ---
    const push_range = vk.PushConstantRange{
        .stage_flags = .{ .vertex_bit = true },
        .offset = 0,
        .size = 64,
    };
    const pipeline_layout = try dev.createPipelineLayout(&.{
        .set_layout_count = 1,
        .p_set_layouts = @ptrCast(&dsl),
        .push_constant_range_count = 1,
        .p_push_constant_ranges = @ptrCast(&push_range),
    }, null);
    errdefer dev.destroyPipelineLayout(pipeline_layout, null);

    // --- Shader modules (SPIR-V compiled from dvui.vert / dvui.frag by build.zig) ---
    // `data` is `*const [N:0]u8` — a pointer to the embedded bytes.
    // Use it directly (not `&data`, which would be a pointer-to-pointer).
    const vert_spv = @import("vert_spv").data;
    const frag_spv = @import("frag_spv").data;

    const vert_mod = try dev.createShaderModule(&.{
        .code_size = vert_spv.len,
        .p_code = @ptrCast(@alignCast(vert_spv)),
    }, null);
    defer dev.destroyShaderModule(vert_mod, null);

    const frag_mod = try dev.createShaderModule(&.{
        .code_size = frag_spv.len,
        .p_code = @ptrCast(@alignCast(frag_spv)),
    }, null);
    defer dev.destroyShaderModule(frag_mod, null);

    // --- Graphics pipeline ---
    const shader_stages = [_]vk.PipelineShaderStageCreateInfo{
        .{ .stage = .{ .vertex_bit = true }, .module = vert_mod, .p_name = "main" },
        .{ .stage = .{ .fragment_bit = true }, .module = frag_mod, .p_name = "main" },
    };

    // dvui.Vertex layout: pos(f32x2) col(u8x4) uv(f32x2)  [20 bytes]
    const vert_bindings = [_]vk.VertexInputBindingDescription{.{
        .binding = 0,
        .stride = @sizeOf(dvui.Vertex),
        .input_rate = .vertex,
    }};
    const vert_attribs = [_]vk.VertexInputAttributeDescription{
        .{ .location = 0, .binding = 0, .format = .r32g32_sfloat, .offset = @offsetOf(dvui.Vertex, "pos") },
        .{ .location = 1, .binding = 0, .format = .r8g8b8a8_unorm, .offset = @offsetOf(dvui.Vertex, "col") },
        .{ .location = 2, .binding = 0, .format = .r32g32_sfloat, .offset = @offsetOf(dvui.Vertex, "uv") },
    };

    const dynamic_states = [_]vk.DynamicState{ .viewport, .scissor };

    const blend_attachment = vk.PipelineColorBlendAttachmentState{
        .blend_enable = .true,
        .src_color_blend_factor = .one,
        .dst_color_blend_factor = .one_minus_src_alpha,
        .color_blend_op = .add,
        .src_alpha_blend_factor = .one,
        .dst_alpha_blend_factor = .one_minus_src_alpha,
        .alpha_blend_op = .add,
        .color_write_mask = .{ .r_bit = true, .g_bit = true, .b_bit = true, .a_bit = true },
    };

    const pipeline_ci = [_]vk.GraphicsPipelineCreateInfo{
        .{
            .stage_count = shader_stages.len,
            .p_stages = &shader_stages,
            .p_vertex_input_state = &.{
                .vertex_binding_description_count = vert_bindings.len,
                .p_vertex_binding_descriptions = &vert_bindings,
                .vertex_attribute_description_count = vert_attribs.len,
                .p_vertex_attribute_descriptions = &vert_attribs,
            },
            .p_input_assembly_state = &.{
                .topology = .triangle_list,
                .primitive_restart_enable = .false,
            },
            .p_viewport_state = &.{ .viewport_count = 1, .scissor_count = 1 },
            .p_rasterization_state = &.{
                .depth_clamp_enable = .false,
                .rasterizer_discard_enable = .false,
                .polygon_mode = .fill,
                .cull_mode = .{},
                .front_face = .counter_clockwise,
                .depth_bias_enable = .false,
                .depth_bias_constant_factor = 0,
                .depth_bias_clamp = 0,
                .depth_bias_slope_factor = 0,
                .line_width = 1.0,
            },
            .p_multisample_state = &.{
                .rasterization_samples = .{ .@"1_bit" = true },
                .sample_shading_enable = .false,
                .min_sample_shading = 0,
                .alpha_to_coverage_enable = .false,
                .alpha_to_one_enable = .false,
            },
            .p_color_blend_state = &.{
                .logic_op_enable = .false,
                .logic_op = .copy,
                .attachment_count = 1,
                .p_attachments = @ptrCast(&blend_attachment),
                .blend_constants = .{ 0, 0, 0, 0 },
            },
            .p_dynamic_state = &.{
                .dynamic_state_count = dynamic_states.len,
                .p_dynamic_states = &dynamic_states,
            },
            .layout = pipeline_layout,
            .render_pass = opts.render_pass,
            .subpass = 0,
            // Must be -1 when VK_PIPELINE_CREATE_DERIVATIVE_BIT is not set (VUID-07984).
            .base_pipeline_index = -1,
        }
    };

    var pipeline: vk.Pipeline = .null_handle;
    _ = try dev.createGraphicsPipelines(.null_handle, @intCast(pipeline_ci.len), &pipeline_ci, null, @ptrCast(&pipeline));
    errdefer dev.destroyPipeline(pipeline, null);

    // --- Descriptor pool (up to 1024 textures) ---
    const pool_size = vk.DescriptorPoolSize{
        .type = .combined_image_sampler,
        .descriptor_count = 1024,
    };
    const desc_pool = try dev.createDescriptorPool(&.{
        .flags = .{ .free_descriptor_set_bit = true },
        .max_sets = 1024,
        .pool_size_count = 1,
        .p_pool_sizes = @ptrCast(&pool_size),
    }, null);
    errdefer dev.destroyDescriptorPool(desc_pool, null);

    // --- Command pool for transient texture-upload commands ---
    const cmd_pool = try dev.createCommandPool(&.{
        .flags = .{ .transient_bit = true },
        .queue_family_index = opts.graphics_queue_family,
    }, null);
    errdefer dev.destroyCommandPool(cmd_pool, null);

    // --- Host-visible geometry buffers, persistently mapped ---
    const host_flags = vk.MemoryPropertyFlags{ .host_visible_bit = true, .host_coherent_bit = true };

    const vtx_result = try createBuffer(
        opts.instance,
        dev, opts.physical_device,
        @sizeOf(dvui.Vertex) * opts.vertex_capacity,
        .{ .vertex_buffer_bit = true },
        host_flags,
    );
    errdefer {
        dev.destroyBuffer(vtx_result.buf, null);
        dev.freeMemory(vtx_result.mem, null);
    }

    const idx_result = try createBuffer(
        opts.instance, dev, opts.physical_device,
        @sizeOf(dvui.Vertex.Index) * opts.index_capacity,
        .{ .index_buffer_bit = true },
        host_flags,
    );
    errdefer {
        dev.destroyBuffer(idx_result.buf, null);
        dev.freeMemory(idx_result.mem, null);
    }

    const vtx_ptr = try dev.mapMemory(vtx_result.mem, 0, vk.WHOLE_SIZE, .{});
    const idx_ptr = try dev.mapMemory(idx_result.mem, 0, vk.WHOLE_SIZE, .{});

    self = .{
        .allocator = allocator,
        .instance = opts.instance,
        .device = dev,
        .physical_device = opts.physical_device,
        .graphics_queue = opts.graphics_queue,
        .cmd_pool = cmd_pool,
        .pipeline = pipeline,
        .pipeline_layout = pipeline_layout,
        .dsl = dsl,
        .desc_pool = desc_pool,
        .vtx_buf = vtx_result.buf,
        .vtx_mem = vtx_result.mem,
        .vtx_mapped = @ptrCast(@alignCast(vtx_ptr)),
        .vtx_cap = opts.vertex_capacity,
        .idx_buf = idx_result.buf,
        .idx_mem = idx_result.mem,
        .idx_mapped = @ptrCast(@alignCast(idx_ptr)),
        .idx_cap = opts.index_capacity,
        .null_tex = undefined,
    };

    // 1×1 white texture: sampling it returns (1,1,1,1) → vertex color passes through unchanged
    const white = [_]u8{ 0xff, 0xff, 0xff, 0xff };
    self.null_tex = try self.createGpuTexture(&white, 1, 1, .linear);

    return self;
}

pub fn deinit(self: *Self) void {
    self.device.queueWaitIdle(self.graphics_queue) catch {};
    self.destroyGpuTexture(self.null_tex);
    self.device.unmapMemory(self.idx_mem);
    self.device.unmapMemory(self.vtx_mem);
    self.device.destroyBuffer(self.idx_buf, null);
    self.device.freeMemory(self.idx_mem, null);
    self.device.destroyBuffer(self.vtx_buf, null);
    self.device.freeMemory(self.vtx_mem, null);
    self.device.destroyCommandPool(self.cmd_pool, null);
    self.device.destroyDescriptorPool(self.desc_pool, null);
    self.device.destroyPipeline(self.pipeline, null);
    self.device.destroyPipelineLayout(self.pipeline_layout, null);
    self.device.destroyDescriptorSetLayout(self.dsl, null);
}

/// Call once per frame, after vkCmdBeginRenderPass, before win.begin().
pub fn setFrame(self: *Self, cmd: vk.CommandBuffer, extent: vk.Extent2D) void {
    self.frame_cmd = cmd;
    self.frame_extent = extent;
}

// ---- dvui renderer interface ----

pub fn begin(self: *Self, _: std.mem.Allocator) !void {
    self.vtx_head = 0;
    self.idx_head = 0;

    const cmd = self.frame_cmd;
    const w: f32 = @floatFromInt(self.frame_extent.width);
    const h: f32 = @floatFromInt(self.frame_extent.height);

    self.device.cmdBindPipeline(cmd, .graphics, self.pipeline);

    const vtx_offsets = [_]vk.DeviceSize{0};
    self.device.cmdBindVertexBuffers(cmd, 0, 1, @ptrCast(&self.vtx_buf), &vtx_offsets);

    const idx_type: vk.IndexType = switch (dvui.Vertex.Index) {
        u16 => .uint16,
        u32 => .uint32,
        else => @compileError("unsupported vertex index type"),
    };
    self.device.cmdBindIndexBuffer(cmd, self.idx_buf, 0, idx_type);

    self.device.cmdSetViewport(cmd, 0, 1, @ptrCast(&vk.Viewport{
        .x = 0, .y = 0, .width = w, .height = h, .min_depth = 0, .max_depth = 1,
    }));

    // Orthographic projection: screen [0,w]×[0,h] → Vulkan NDC [-1,1]×[-1,1].
    // Stored column-major (GLSL mat4 convention).
    // Col 0: [2/w, 0, 0, 0]   Col 1: [0, 2/h, 0, 0]
    // Col 2: [0, 0, 0, 0]     Col 3: [-1, -1, 0, 1]
    const proj = [16]f32{
        2.0 / w, 0,       0, 0,
        0,       2.0 / h, 0, 0,
        0,       0,       0, 0,
        -1,      -1,      0, 1,
    };
    self.device.cmdPushConstants(cmd, self.pipeline_layout, .{ .vertex_bit = true }, 0, 64, &proj);
}

pub fn end(_: *Self) !void {}

pub fn drawClippedTriangles(
    self: *Self,
    _: dvui.Size.Physical,
    maybe_texture: ?dvui.Texture,
    vtx: []const dvui.Vertex,
    idx: []const dvui.Vertex.Index,
    maybe_clipr: ?dvui.Rect.Physical,
) !void {
    if (vtx.len == 0 or idx.len == 0) return;

    if (self.vtx_head + @as(u32, @intCast(vtx.len)) > self.vtx_cap) {
        log.err("vertex buffer overflow ({}/{}), skipping draw — increase vertex_capacity", .{ self.vtx_head + vtx.len, self.vtx_cap });
        return;
    }
    if (self.idx_head + @as(u32, @intCast(idx.len)) > self.idx_cap) {
        log.err("index buffer overflow ({}/{}), skipping draw — increase index_capacity", .{ self.idx_head + idx.len, self.idx_cap });
        return;
    }

    @memcpy(self.vtx_mapped[self.vtx_head..][0..vtx.len], vtx);
    @memcpy(self.idx_mapped[self.idx_head..][0..idx.len], idx);

    const cmd = self.frame_cmd;

    const desc = if (maybe_texture) |t| blk: {
        const tex: *GpuTexture = @ptrCast(@alignCast(t.ptr));
        break :blk tex.descriptor;
    } else self.null_tex.descriptor;

    self.device.cmdBindDescriptorSets(cmd, .graphics, self.pipeline_layout, 0, 1, @ptrCast(&desc), 0, null);

    const scissor: vk.Rect2D = if (maybe_clipr) |r| .{
        .offset = .{ .x = @intFromFloat(@max(0, r.x)), .y = @intFromFloat(@max(0, r.y)) },
        .extent = .{ .width = @intFromFloat(@max(0, r.w)), .height = @intFromFloat(@max(0, r.h)) },
    } else .{
        .offset = .{ .x = 0, .y = 0 },
        .extent = self.frame_extent,
    };
    self.device.cmdSetScissor(cmd, 0, 1, @ptrCast(&scissor));

    self.device.cmdDrawIndexed(cmd, @intCast(idx.len), 1, self.idx_head, @intCast(self.vtx_head), 0);

    self.vtx_head += @intCast(vtx.len);
    self.idx_head += @intCast(idx.len);
}

pub fn textureCreate(self: *Self, pixels: [*]const u8, width: u32, height: u32, interpolation: dvui.enums.TextureInterpolation, format: dvui.enums.TexturePixelFormat) dvui.Backend.TextureError!dvui.Texture {
    if (format != .rgba_32) {
        log.err("unsupported texture format: {}", .{format});
        return error.TextureCreate;
    }
    const tex = self.createGpuTexture(pixels, width, height, interpolation) catch |err| {
        log.err("textureCreate: {}", .{err});
        return error.TextureCreate;
    };
    return .{ .ptr = tex, .width = width, .height = height, .format = format };
}

pub fn textureUpdate(self: *Self, texture: dvui.Texture, pixels: [*]const u8) dvui.Backend.TextureError!void {
    const size: vk.DeviceSize = texture.width * texture.height * 4;
    const tex: *GpuTexture = @ptrCast(@alignCast(texture.ptr));

    const stg = createBuffer(self.instance, self.device, self.physical_device, size,
        .{ .transfer_src_bit = true },
        .{ .host_visible_bit = true, .host_coherent_bit = true },
    ) catch return error.TextureUpdate;
    defer {
        self.device.destroyBuffer(stg.buf, null);
        self.device.freeMemory(stg.mem, null);
    }

    const stg_ptr = self.device.mapMemory(stg.mem, 0, vk.WHOLE_SIZE, .{}) catch return error.TextureUpdate;
    @memcpy(@as([*]u8, @ptrCast(@alignCast(stg_ptr)))[0..size], pixels[0..size]);
    self.device.unmapMemory(stg.mem);

    const cmd = self.oneTimeSubmitBegin() catch return error.TextureUpdate;
    cmdTransitionImage(self.device, cmd, tex.image,
        .shader_read_only_optimal, .transfer_dst_optimal,
        .{ .shader_read_bit = true }, .{ .transfer_write_bit = true },
        .{ .fragment_shader_bit = true }, .{ .transfer_bit = true });
    self.device.cmdCopyBufferToImage(cmd, stg.buf, tex.image, .transfer_dst_optimal, 1, @ptrCast(&vk.BufferImageCopy{
        .buffer_offset = 0, .buffer_row_length = 0, .buffer_image_height = 0,
        .image_subresource = .{ .aspect_mask = .{ .color_bit = true }, .mip_level = 0, .base_array_layer = 0, .layer_count = 1 },
        .image_offset = .{ .x = 0, .y = 0, .z = 0 },
        .image_extent = .{ .width = texture.width, .height = texture.height, .depth = 1 },
    }));
    cmdTransitionImage(self.device, cmd, tex.image,
        .transfer_dst_optimal, .shader_read_only_optimal,
        .{ .transfer_write_bit = true }, .{ .shader_read_bit = true },
        .{ .transfer_bit = true }, .{ .fragment_shader_bit = true });
    self.oneTimeSubmitEnd(cmd) catch return error.TextureUpdate;
}

pub fn textureDestroy(self: *Self, texture: dvui.Texture) void {
    self.destroyGpuTexture(@ptrCast(@alignCast(texture.ptr)));
}

// ---- Render-to-texture (not yet implemented) ----

pub fn textureCreateTarget(_: *Self, _: u32, _: u32, _: dvui.enums.TextureInterpolation, _: dvui.enums.TexturePixelFormat) dvui.Backend.TextureError!dvui.TextureTarget {
    return error.NotImplemented;
}

pub fn textureReadTarget(_: *Self, _: dvui.TextureTarget, _: [*]u8) dvui.Backend.TextureError!void {
    return error.NotImplemented;
}

pub fn textureClearTarget(_: *Self, _: dvui.Texture.Target) void {}

pub fn textureDestroyTarget(_: *Self, _: dvui.Texture.Target) void {}

pub fn textureFromTarget(_: *Self, target: dvui.TextureTarget) dvui.Backend.TextureError!dvui.Texture {
    return .{ .ptr = target.ptr, .width = target.width, .height = target.height, .format = target.format };
}

pub fn textureFromTargetTemp(self: *Self, target: dvui.TextureTarget) dvui.Backend.TextureError!dvui.Texture {
    return self.textureFromTarget(target);
}

pub fn renderTarget(_: *Self, _: ?dvui.TextureTarget) dvui.Backend.GenericError!void {
    return error.BackendError;
}

// ---- Private helpers ----

fn createGpuTexture(self: *Self, pixels: [*]const u8, width: u32, height: u32, interpolation: dvui.enums.TextureInterpolation) !*GpuTexture {
    const dev = self.device;
    const size: vk.DeviceSize = width * height * 4;

    const image = try dev.createImage(&.{
        .image_type = .@"2d",
        .format = img_format,
        .extent = .{ .width = width, .height = height, .depth = 1 },
        .mip_levels = 1, .array_layers = 1,
        .samples = .{ .@"1_bit" = true },
        .tiling = .optimal,
        .usage = .{ .transfer_dst_bit = true, .sampled_bit = true },
        .sharing_mode = .exclusive,
        .initial_layout = .undefined,
    }, null);
    errdefer dev.destroyImage(image, null);

    const img_reqs = dev.getImageMemoryRequirements(image);
    const img_mem = try dev.allocateMemory(&.{
        .allocation_size = img_reqs.size,
        .memory_type_index = try findMemoryType(self.instance,self.physical_device, img_reqs.memory_type_bits, .{ .device_local_bit = true }),
    }, null);
    errdefer dev.freeMemory(img_mem, null);
    try dev.bindImageMemory(image, img_mem, 0);

    // Upload via staging buffer
    const stg = try createBuffer(self.instance, dev, self.physical_device, size,
        .{ .transfer_src_bit = true },
        .{ .host_visible_bit = true, .host_coherent_bit = true },
    );
    defer {
        dev.destroyBuffer(stg.buf, null);
        dev.freeMemory(stg.mem, null);
    }
    const stg_ptr = try dev.mapMemory(stg.mem, 0, vk.WHOLE_SIZE, .{});
    @memcpy(@as([*]u8, @ptrCast(@alignCast(stg_ptr)))[0..size], pixels[0..size]);
    dev.unmapMemory(stg.mem);

    const cmd = try self.oneTimeSubmitBegin();
    cmdTransitionImage(dev, cmd, image,
        .undefined, .transfer_dst_optimal,
        .{}, .{ .transfer_write_bit = true },
        .{ .top_of_pipe_bit = true }, .{ .transfer_bit = true });
    dev.cmdCopyBufferToImage(cmd, stg.buf, image, .transfer_dst_optimal, 1, @ptrCast(&vk.BufferImageCopy{
        .buffer_offset = 0, .buffer_row_length = 0, .buffer_image_height = 0,
        .image_subresource = .{ .aspect_mask = .{ .color_bit = true }, .mip_level = 0, .base_array_layer = 0, .layer_count = 1 },
        .image_offset = .{ .x = 0, .y = 0, .z = 0 },
        .image_extent = .{ .width = width, .height = height, .depth = 1 },
    }));
    cmdTransitionImage(dev, cmd, image,
        .transfer_dst_optimal, .shader_read_only_optimal,
        .{ .transfer_write_bit = true }, .{ .shader_read_bit = true },
        .{ .transfer_bit = true }, .{ .fragment_shader_bit = true });
    try self.oneTimeSubmitEnd(cmd);

    const view = try dev.createImageView(&.{
        .image = image,
        .view_type = .@"2d",
        .format = img_format,
        .components = .{ .r = .identity, .g = .identity, .b = .identity, .a = .identity },
        .subresource_range = color_subresource,
    }, null);
    errdefer dev.destroyImageView(view, null);

    const filter: vk.Filter = switch (interpolation) {
        .linear => .linear,
        .nearest => .nearest,
    };
    const sampler = try dev.createSampler(&.{
        .mag_filter = filter, .min_filter = filter,
        .mipmap_mode = .nearest,
        .address_mode_u = .clamp_to_edge,
        .address_mode_v = .clamp_to_edge,
        .address_mode_w = .clamp_to_edge,
        .mip_lod_bias = 0,
        .anisotropy_enable = .false, .max_anisotropy = 0,
        .compare_enable = .false, .compare_op = .always,
        .min_lod = 0, .max_lod = 0,
        .border_color = .float_transparent_black,
        .unnormalized_coordinates = .false,
    }, null);
    errdefer dev.destroySampler(sampler, null);

    var descriptor: vk.DescriptorSet = undefined;
    try dev.allocateDescriptorSets(&.{
        .descriptor_pool = self.desc_pool,
        .descriptor_set_count = 1,
        .p_set_layouts = @ptrCast(&self.dsl),
    }, @ptrCast(&descriptor));

    dev.updateDescriptorSets(1, @ptrCast(&vk.WriteDescriptorSet{
        .dst_set = descriptor,
        .dst_binding = 0,
        .dst_array_element = 0,
        .descriptor_count = 1,
        .descriptor_type = .combined_image_sampler,
        .p_image_info = @ptrCast(&vk.DescriptorImageInfo{
            .sampler = sampler,
            .image_view = view,
            .image_layout = .shader_read_only_optimal,
        }),
        .p_buffer_info = undefined,
        .p_texel_buffer_view = undefined,
    }), 0, null);

    const gpu_tex = try self.allocator.create(GpuTexture);
    gpu_tex.* = .{ .image = image, .memory = img_mem, .view = view, .sampler = sampler, .descriptor = descriptor };
    return gpu_tex;
}

fn destroyGpuTexture(self: *Self, tex: *GpuTexture) void {
    _ = self.device.freeDescriptorSets(self.desc_pool, 1, @ptrCast(&tex.descriptor)) catch {};
    self.device.destroySampler(tex.sampler, null);
    self.device.destroyImageView(tex.view, null);
    self.device.destroyImage(tex.image, null);
    self.device.freeMemory(tex.memory, null);
    self.allocator.destroy(tex);
}

fn oneTimeSubmitBegin(self: *const Self) !vk.CommandBuffer {
    var cmd: vk.CommandBuffer = undefined;
    try self.device.allocateCommandBuffers(&.{
        .command_pool = self.cmd_pool,
        .level = .primary,
        .command_buffer_count = 1,
    }, @ptrCast(&cmd));
    errdefer self.device.freeCommandBuffers(self.cmd_pool, 1, @ptrCast(&cmd));
    try self.device.beginCommandBuffer(cmd, &.{ .flags = .{ .one_time_submit_bit = true } });
    return cmd;
}

fn oneTimeSubmitEnd(self: *const Self, cmd: vk.CommandBuffer) !void {
    try self.device.endCommandBuffer(cmd);
    const submit = vk.SubmitInfo{
        .command_buffer_count = 1,
        .p_command_buffers = @ptrCast(&cmd),
    };
    try self.device.queueSubmit(self.graphics_queue, 1, @ptrCast(&submit), .null_handle);
    try self.device.queueWaitIdle(self.graphics_queue);
    self.device.freeCommandBuffers(self.cmd_pool, 1, @ptrCast(&cmd));
}

const BufferResult = struct { buf: vk.Buffer, mem: vk.DeviceMemory };

fn createBuffer(
    instance: *const Instance,
    dev: *const Device,
    phys: vk.PhysicalDevice,
    size: vk.DeviceSize,
    usage: vk.BufferUsageFlags,
    flags: vk.MemoryPropertyFlags,
) !BufferResult {
    const buf = try dev.createBuffer(&.{
        .size = size, .usage = usage, .sharing_mode = .exclusive,
    }, null);
    errdefer dev.destroyBuffer(buf, null);

    const reqs = dev.getBufferMemoryRequirements(buf);
    const mem = try dev.allocateMemory(&.{
        .allocation_size = reqs.size,
        .memory_type_index = try findMemoryType(instance, phys, reqs.memory_type_bits, flags),
    }, null);
    errdefer dev.freeMemory(mem, null);

    try dev.bindBufferMemory(buf, mem, 0);
    return .{ .buf = buf, .mem = mem };
}

fn findMemoryType(instance: *const Instance, phys: vk.PhysicalDevice, type_bits: u32, flags: vk.MemoryPropertyFlags) !u32 {
    const props = instance.getPhysicalDeviceMemoryProperties(phys);
    for (props.memory_types[0..props.memory_type_count], 0..) |mt, i| {
        if (type_bits & (@as(u32, 1) << @intCast(i)) != 0 and mt.property_flags.contains(flags))
            return @intCast(i);
    }
    return error.NoSuitableMemoryType;
}

fn cmdTransitionImage(
    dev: *const Device,
    cmd: vk.CommandBuffer,
    image: vk.Image,
    old_layout: vk.ImageLayout,
    new_layout: vk.ImageLayout,
    src_access: vk.AccessFlags,
    dst_access: vk.AccessFlags,
    src_stage: vk.PipelineStageFlags,
    dst_stage: vk.PipelineStageFlags,
) void {
    dev.cmdPipelineBarrier(cmd, src_stage, dst_stage, .{}, 0, null, 0, null, 1, @ptrCast(&vk.ImageMemoryBarrier{
        .src_access_mask = src_access,
        .dst_access_mask = dst_access,
        .old_layout = old_layout,
        .new_layout = new_layout,
        .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresource_range = color_subresource,
    }));
}
