//! Vulkan 1.2+ renderer.
//!
//! Shader source: src/backends/render/dvui.{vert,frag}
//! Build compiles them to SPIR-V via glslc; they are embedded as `vert_spv` / `frag_spv` modules.
//!
//! Usage per-frame:
//!   Call `renderer.setFrame(cmd, framebuffer, extent)` before `win.begin()`.
//!   dvui will call vkCmdBeginRenderPass / vkCmdEndRenderPass itself around the frame.
//!   dvui owns no command pool (other than the transient texture-upload pool).
//!
//! Use `renderer.renderPass()` to retrieve the VkRenderPass needed to create
//! the app's swapchain framebuffers.

const std = @import("std");
const Allocator = std.mem.Allocator;
const dvui = @import("dvui");
const vk = @import("vk");
const Instance = vk.InstanceProxy;
const Device = vk.DeviceProxy;
const log = std.log.scoped(.dvui_vulkan);

pub const kind: dvui.enums.RenderBackend = .vulkan;

/// Pixel format used for all offscreen render targets.
const offscreen_format: vk.Format = .r8g8b8a8_unorm;

const color_subresource = vk.ImageSubresourceRange{
    .aspect_mask = .{ .color_bit = true },
    .base_mip_level = 0,
    .level_count = 1,
    .base_array_layer = 0,
    .layer_count = 1,
};

/// Heap-allocated GPU resources per sampled texture.
const GpuTexture = struct {
    image: vk.Image,
    memory: vk.DeviceMemory,
    view: vk.ImageView,
    sampler: vk.Sampler,
    descriptor: vk.DescriptorSet,
};

/// Heap-allocated GPU resources per render-target texture.
/// `TextureTarget.ptr` points here; `Texture.ptr` points to the embedded `gpu_tex`.
const GpuRenderTarget = struct {
    gpu_tex: *GpuTexture,
    framebuffer: vk.Framebuffer,
    width: u32,
    height: u32,
};

const Self = @This();

allocator: Allocator,
instance: *const Instance,
device: *const Device,
physical_device: vk.PhysicalDevice,
graphics_queue: vk.Queue,
/// Transient pool for one-time texture-upload commands.
cmd_pool: vk.CommandPool,

/// Main-screen render pass — initial, uses the configured load_op.
main_rp: vk.RenderPass,
/// Main-screen render pass — used when re-entering main after offscreen (load_op=load).
main_rp_load: vk.RenderPass,
/// Offscreen render pass — r8g8b8a8_unorm, preserves content (load_op=load).
offscreen_rp: vk.RenderPass,

/// Pipeline compiled against main_rp; also valid for main_rp_load (compatible passes).
main_pipeline: vk.Pipeline,
/// Pipeline compiled against offscreen_rp.
offscreen_pipeline: vk.Pipeline,
pipeline_layout: vk.PipelineLayout,
dsl: vk.DescriptorSetLayout,
desc_pool: vk.DescriptorPool,

clear_value: vk.ClearColorValue,

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

/// 1×1 white texture — sampling returns (1,1,1,1) so vertex colour passes through.
null_tex: *GpuTexture,

/// Set each frame by `setFrame` before `win.begin()`.
frame_cmd: vk.CommandBuffer = .null_handle,
frame_framebuffer: vk.Framebuffer = .null_handle,
frame_extent: vk.Extent2D = .{ .width = 0, .height = 0 },

/// Write heads into vtx/idx buffers; reset to 0 each `begin()`.
vtx_head: u32 = 0,
idx_head: u32 = 0,

/// null  → currently rendering to the main screen.
/// non-null → currently rendering to this offscreen target.
active_target: ?*GpuRenderTarget = null,

/// Number of frame slots the app cycles through (see `InitOptions.frames_in_flight`).
frames_in_flight: u32 = 2,
/// Incremented once per `setFrame` call.
frame_counter: u64 = 0,
/// Render-target resources retired mid-frame (via `textureDestroyTarget` /
/// `textureFromTarget`) can still be referenced by the current frame's
/// not-yet-submitted command buffer (render-target rendering is recorded
/// directly into `frame_cmd`). Destroying them immediately is invalid, so
/// they're queued here and actually destroyed once enough frames have
/// cycled that the command buffer which referenced them is guaranteed to
/// have finished executing.
pending_destroys: std.ArrayList(PendingDestroy) = .empty,

const PendingDestroy = struct {
    framebuffer: vk.Framebuffer,
    /// Set when the whole texture (not just the framebuffer) should be destroyed too.
    gpu_tex: ?*GpuTexture,
    retire_frame: u64,
};

// ---- Public init / deinit ----

pub const InitOptions = struct {
    instance: *const Instance,
    device: *const Device,
    physical_device: vk.PhysicalDevice,
    graphics_queue: vk.Queue,
    graphics_queue_family: u32,
    /// Format of the swapchain / main colour attachment.
    color_format: vk.Format,
    /// Load operation for the main render pass colour attachment.
    /// Use `.clear` if dvui clears the background, `.load` if the app rendered its
    /// scene first and dvui is overlaid on top.
    load_op: vk.AttachmentLoadOp = .clear,
    /// Clear colour when `load_op == .clear`.
    clear_value: vk.ClearColorValue = .{ .float_32 = .{ 0, 0, 0, 1 } },
    /// Pre-allocated vertex slots.
    vertex_capacity: u32 = 65536,
    /// Pre-allocated index slots.
    index_capacity: u32 = 131072,
    /// Number of frame slots (command buffers) the app cycles through.
    /// Used to know how many `setFrame` calls must pass before it's safe to
    /// destroy a render-target resource retired mid-frame. Must match the
    /// app's actual frames-in-flight count.
    frames_in_flight: u32 = 2,
};

pub fn init(allocator: Allocator, opts: InitOptions) !Self {
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

    // --- Shader modules ---
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

    // --- Render passes ---
    const initial_layout: vk.ImageLayout = if (opts.load_op == .clear) .undefined else .color_attachment_optimal;
    const main_rp = try createColorRenderPass(dev, opts.color_format, opts.load_op, initial_layout, .present_src_khr);
    errdefer dev.destroyRenderPass(main_rp, null);

    // Resume pass: called when returning to main after offscreen rendering.
    // At that point the swapchain image is in present_src_khr (final_layout of main_rp).
    const main_rp_load = try createColorRenderPass(dev, opts.color_format, .load, .present_src_khr, .present_src_khr);
    errdefer dev.destroyRenderPass(main_rp_load, null);

    // Offscreen pass: preserves image content, keeps attachment in color_attachment_optimal
    // (we manually transition to/from shader_read_only_optimal around it).
    const offscreen_rp = try createColorRenderPass(dev, offscreen_format, .load, .color_attachment_optimal, .color_attachment_optimal);
    errdefer dev.destroyRenderPass(offscreen_rp, null);

    // --- Pipelines ---
    // main_rp and main_rp_load are compatible (same format, same sample count),
    // so main_pipeline works for both.
    const main_pipeline = try createGraphicsPipeline(dev, main_rp, pipeline_layout, vert_mod, frag_mod);
    errdefer dev.destroyPipeline(main_pipeline, null);

    const offscreen_pipeline = try createGraphicsPipeline(dev, offscreen_rp, pipeline_layout, vert_mod, frag_mod);
    errdefer dev.destroyPipeline(offscreen_pipeline, null);

    // --- Descriptor pool (up to 1024 textures) ---
    const pool_size = vk.DescriptorPoolSize{ .type = .combined_image_sampler, .descriptor_count = 1024 };
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

    // --- Host-visible geometry buffers ---
    const host_flags = vk.MemoryPropertyFlags{ .host_visible_bit = true, .host_coherent_bit = true };

    const vtx_result = try createBuffer(opts.instance, dev, opts.physical_device,
        @sizeOf(dvui.Vertex) * opts.vertex_capacity,
        .{ .vertex_buffer_bit = true }, host_flags);
    errdefer { dev.destroyBuffer(vtx_result.buf, null); dev.freeMemory(vtx_result.mem, null); }

    const idx_result = try createBuffer(opts.instance, dev, opts.physical_device,
        @sizeOf(dvui.Vertex.Index) * opts.index_capacity,
        .{ .index_buffer_bit = true }, host_flags);
    errdefer { dev.destroyBuffer(idx_result.buf, null); dev.freeMemory(idx_result.mem, null); }

    const vtx_ptr = try dev.mapMemory(vtx_result.mem, 0, vk.WHOLE_SIZE, .{});
    const idx_ptr = try dev.mapMemory(idx_result.mem, 0, vk.WHOLE_SIZE, .{});

    self = .{
        .allocator = allocator,
        .instance = opts.instance,
        .device = dev,
        .physical_device = opts.physical_device,
        .graphics_queue = opts.graphics_queue,
        .cmd_pool = cmd_pool,
        .main_rp = main_rp,
        .main_rp_load = main_rp_load,
        .offscreen_rp = offscreen_rp,
        .main_pipeline = main_pipeline,
        .offscreen_pipeline = offscreen_pipeline,
        .pipeline_layout = pipeline_layout,
        .dsl = dsl,
        .desc_pool = desc_pool,
        .clear_value = opts.clear_value,
        .vtx_buf = vtx_result.buf,
        .vtx_mem = vtx_result.mem,
        .vtx_mapped = @ptrCast(@alignCast(vtx_ptr)),
        .vtx_cap = opts.vertex_capacity,
        .idx_buf = idx_result.buf,
        .idx_mem = idx_result.mem,
        .idx_mapped = @ptrCast(@alignCast(idx_ptr)),
        .idx_cap = opts.index_capacity,
        .null_tex = undefined,
        .frames_in_flight = opts.frames_in_flight,
    };

    const white = [_]u8{ 0xff, 0xff, 0xff, 0xff };
    self.null_tex = try self.createSampledTexture(&white, 1, 1, .linear, .clamp, .clamp);

    return self;
}

pub fn deinit(self: *Self) void {
    self.device.queueWaitIdle(self.graphics_queue) catch {};
    for (self.pending_destroys.items) |pd| {
        self.device.destroyFramebuffer(pd.framebuffer, null);
        if (pd.gpu_tex) |gt| self.destroyGpuTexture(gt);
    }
    self.pending_destroys.deinit(self.allocator);
    self.destroyGpuTexture(self.null_tex);
    self.device.unmapMemory(self.idx_mem);
    self.device.unmapMemory(self.vtx_mem);
    self.device.destroyBuffer(self.idx_buf, null);
    self.device.freeMemory(self.idx_mem, null);
    self.device.destroyBuffer(self.vtx_buf, null);
    self.device.freeMemory(self.vtx_mem, null);
    self.device.destroyCommandPool(self.cmd_pool, null);
    self.device.destroyDescriptorPool(self.desc_pool, null);
    self.device.destroyPipeline(self.offscreen_pipeline, null);
    self.device.destroyPipeline(self.main_pipeline, null);
    self.device.destroyRenderPass(self.offscreen_rp, null);
    self.device.destroyRenderPass(self.main_rp_load, null);
    self.device.destroyRenderPass(self.main_rp, null);
    self.device.destroyPipelineLayout(self.pipeline_layout, null);
    self.device.destroyDescriptorSetLayout(self.dsl, null);
}

/// Returns the VkRenderPass that the app's swapchain framebuffers must be
/// compatible with.  Create framebuffers using this handle.
pub fn renderPass(self: *const Self) vk.RenderPass {
    return self.main_rp;
}

/// Call once per frame, after `vkBeginCommandBuffer`, before `win.begin()`.
/// dvui will begin/end the render pass in `begin()`/`end()`.
pub fn setFrame(self: *Self, cmd: vk.CommandBuffer, framebuffer: vk.Framebuffer, extent: vk.Extent2D) void {
    self.flushPendingDestroys();
    self.frame_counter += 1;

    self.frame_cmd = cmd;
    self.frame_framebuffer = framebuffer;
    self.frame_extent = extent;
}

/// Destroy render-target resources retired mid-frame once enough frames
/// have cycled that their last-referencing command buffer is guaranteed to
/// have finished executing (see `pending_destroys`).
fn flushPendingDestroys(self: *Self) void {
    var i: usize = 0;
    while (i < self.pending_destroys.items.len) {
        const pd = self.pending_destroys.items[i];
        if (pd.retire_frame <= self.frame_counter) {
            self.device.destroyFramebuffer(pd.framebuffer, null);
            if (pd.gpu_tex) |gt| self.destroyGpuTexture(gt);
            _ = self.pending_destroys.swapRemove(i);
        } else {
            i += 1;
        }
    }
}

// ---- dvui renderer interface ----

pub fn begin(self: *Self, _: Allocator) !void {
    self.vtx_head = 0;
    self.idx_head = 0;
    self.active_target = null;

    const cmd = self.frame_cmd;
    const clear = vk.ClearValue{ .color = self.clear_value };
    self.device.cmdBeginRenderPass(cmd, &.{
        .render_pass = self.main_rp,
        .framebuffer = self.frame_framebuffer,
        .render_area = .{ .offset = .{ .x = 0, .y = 0 }, .extent = self.frame_extent },
        .clear_value_count = 1,
        .p_clear_values = @ptrCast(&clear),
    }, .@"inline");

    self.bindStateForExtent(self.main_pipeline, self.frame_extent);
}

pub fn end(self: *Self) !void {
    self.device.cmdEndRenderPass(self.frame_cmd);
}

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
        log.err("vertex buffer overflow ({}/{}), skipping draw", .{ self.vtx_head + vtx.len, self.vtx_cap });
        return;
    }
    if (self.idx_head + @as(u32, @intCast(idx.len)) > self.idx_cap) {
        log.err("index buffer overflow ({}/{}), skipping draw", .{ self.idx_head + idx.len, self.idx_cap });
        return;
    }

    @memcpy(self.vtx_mapped[self.vtx_head..][0..vtx.len], vtx);
    @memcpy(self.idx_mapped[self.idx_head..][0..idx.len], idx);

    const cmd = self.frame_cmd;

    const desc = if (maybe_texture) |t| blk: {
        const tex: *GpuTexture = @ptrCast(@alignCast(t.ptr));
        break :blk tex.descriptor;
    } else self.null_tex.descriptor;

    self.device.cmdBindDescriptorSets(cmd, .graphics, self.pipeline_layout, 0, &[_]vk.DescriptorSet{desc}, null);

    const active_extent = if (self.active_target) |rt|
        vk.Extent2D{ .width = rt.width, .height = rt.height }
    else
        self.frame_extent;

    const scissor: vk.Rect2D = if (maybe_clipr) |r| .{
        .offset = .{ .x = @intFromFloat(@max(0, r.x)), .y = @intFromFloat(@max(0, r.y)) },
        .extent = .{ .width = @intFromFloat(@max(0, r.w)), .height = @intFromFloat(@max(0, r.h)) },
    } else .{
        .offset = .{ .x = 0, .y = 0 },
        .extent = active_extent,
    };
    self.device.cmdSetScissor(cmd, 0, &[_]vk.Rect2D{scissor});

    self.device.cmdDrawIndexed(cmd, @intCast(idx.len), 1, self.idx_head, @intCast(self.vtx_head), 0);

    self.vtx_head += @intCast(vtx.len);
    self.idx_head += @intCast(idx.len);
}

// ---- Texture create / update / destroy ----

pub fn textureCreate(self: *Self, pixels: [*]const u8, options: dvui.Texture.CreateOptions) dvui.Backend.TextureError!dvui.Texture {
    if (options.format != .rgba_32) {
        log.err("unsupported texture format: {}", .{options.format});
        return error.TextureCreate;
    }
    const tex = self.createSampledTexture(pixels, options.width, options.height, options.interpolation, options.wrap_u, options.wrap_v) catch |err| {
        log.err("textureCreate: {}", .{err});
        return error.TextureCreate;
    };
    return .{ .ptr = tex, .width = options.width, .height = options.height, .format = options.format, .interpolation = options.interpolation, .wrap_u = options.wrap_u, .wrap_v = options.wrap_v };
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
    self.device.cmdCopyBufferToImage(cmd, stg.buf, tex.image, .transfer_dst_optimal, &[_]vk.BufferImageCopy{.{
        .buffer_offset = 0, .buffer_row_length = 0, .buffer_image_height = 0,
        .image_subresource = .{ .aspect_mask = .{ .color_bit = true }, .mip_level = 0, .base_array_layer = 0, .layer_count = 1 },
        .image_offset = .{ .x = 0, .y = 0, .z = 0 },
        .image_extent = .{ .width = texture.width, .height = texture.height, .depth = 1 },
    }});
    cmdTransitionImage(self.device, cmd, tex.image,
        .transfer_dst_optimal, .shader_read_only_optimal,
        .{ .transfer_write_bit = true }, .{ .shader_read_bit = true },
        .{ .transfer_bit = true }, .{ .fragment_shader_bit = true });
    self.oneTimeSubmitEnd(cmd) catch return error.TextureUpdate;
}

pub fn textureDestroy(self: *Self, texture: dvui.Texture) void {
    self.destroyGpuTexture(@ptrCast(@alignCast(texture.ptr)));
}

// ---- Render-target API ----

pub fn textureCreateTarget(
    self: *Self,
    options: dvui.Texture.CreateOptions,
) dvui.Backend.TextureError!dvui.TextureTarget {
    if (options.format != .rgba_32) return error.TextureCreate;

    const width = options.width;
    const height = options.height;

    const gpu_tex = self.createOffscreenTexture(width, height, options.interpolation, options.wrap_u, options.wrap_v) catch |err| {
        log.err("textureCreateTarget: {}", .{err});
        return error.TextureCreate;
    };
    errdefer self.destroyGpuTexture(gpu_tex);

    const fb = self.device.createFramebuffer(&.{
        .render_pass = self.offscreen_rp,
        .attachment_count = 1,
        .p_attachments = @ptrCast(&gpu_tex.view),
        .width = width,
        .height = height,
        .layers = 1,
    }, null) catch |err| {
        log.err("textureCreateTarget framebuffer: {}", .{err});
        return error.TextureCreate;
    };

    const rt = self.allocator.create(GpuRenderTarget) catch return error.TextureCreate;
    rt.* = .{ .gpu_tex = gpu_tex, .framebuffer = fb, .width = width, .height = height };
    return .{ .ptr = rt, .width = width, .height = height, .format = options.format, .interpolation = options.interpolation, .wrap_u = options.wrap_u, .wrap_v = options.wrap_v };
}

pub fn textureDestroyTarget(self: *Self, texture: dvui.Texture.Target) void {
    const rt: *GpuRenderTarget = @ptrCast(@alignCast(texture.ptr));
    self.pending_destroys.append(self.allocator, .{
        .framebuffer = rt.framebuffer,
        .gpu_tex = rt.gpu_tex,
        .retire_frame = self.frame_counter + self.frames_in_flight,
    }) catch {
        // OOM: fall back to an immediate (technically unsafe, but leak-free) destroy.
        self.device.queueWaitIdle(self.graphics_queue) catch {};
        self.device.destroyFramebuffer(rt.framebuffer, null);
        self.destroyGpuTexture(rt.gpu_tex);
    };
    self.allocator.destroy(rt);
}

pub fn textureClearTarget(self: *Self, texture: dvui.Texture.Target) void {
    const rt: *GpuRenderTarget = @ptrCast(@alignCast(texture.ptr));
    const cmd = self.frame_cmd;

    // End whatever render pass is currently active so we can issue a transfer command.
    self.device.cmdEndRenderPass(cmd);

    // The target image is in shader_read_only_optimal when it's not the current render target
    // (that's where createOffscreenTexture and renderTarget(null) leave it).
    // Transition it to transfer_dst for the clear.
    cmdTransitionImage(self.device, cmd, rt.gpu_tex.image,
        .shader_read_only_optimal, .transfer_dst_optimal,
        .{ .shader_read_bit = true }, .{ .transfer_write_bit = true },
        .{ .fragment_shader_bit = true }, .{ .transfer_bit = true });

    const clear_color = vk.ClearColorValue{ .float_32 = .{ 0, 0, 0, 0 } };
    self.device.cmdClearColorImage(cmd, rt.gpu_tex.image, .transfer_dst_optimal, &clear_color,
        &[_]vk.ImageSubresourceRange{color_subresource});

    cmdTransitionImage(self.device, cmd, rt.gpu_tex.image,
        .transfer_dst_optimal, .shader_read_only_optimal,
        .{ .transfer_write_bit = true }, .{ .shader_read_bit = true },
        .{ .transfer_bit = true }, .{ .fragment_shader_bit = true });

    // Re-begin whichever render pass was active before.
    self.resumeCurrentRenderPass();
}

pub fn textureReadTarget(self: *Self, texture: dvui.TextureTarget, pixels_out: [*]u8) dvui.Backend.TextureError!void {
    const rt: *GpuRenderTarget = @ptrCast(@alignCast(texture.ptr));
    const size: vk.DeviceSize = rt.width * rt.height * 4;

    const stg = createBuffer(self.instance, self.device, self.physical_device, size,
        .{ .transfer_dst_bit = true },
        .{ .host_visible_bit = true, .host_coherent_bit = true },
    ) catch return error.TextureRead;
    defer {
        self.device.destroyBuffer(stg.buf, null);
        self.device.freeMemory(stg.mem, null);
    }

    const cmd = self.oneTimeSubmitBegin() catch return error.TextureRead;
    cmdTransitionImage(self.device, cmd, rt.gpu_tex.image,
        .shader_read_only_optimal, .transfer_src_optimal,
        .{ .shader_read_bit = true }, .{ .transfer_read_bit = true },
        .{ .fragment_shader_bit = true }, .{ .transfer_bit = true });
    self.device.cmdCopyImageToBuffer(cmd, rt.gpu_tex.image, .transfer_src_optimal, stg.buf, &[_]vk.BufferImageCopy{.{
        .buffer_offset = 0, .buffer_row_length = 0, .buffer_image_height = 0,
        .image_subresource = .{ .aspect_mask = .{ .color_bit = true }, .mip_level = 0, .base_array_layer = 0, .layer_count = 1 },
        .image_offset = .{ .x = 0, .y = 0, .z = 0 },
        .image_extent = .{ .width = rt.width, .height = rt.height, .depth = 1 },
    }});
    cmdTransitionImage(self.device, cmd, rt.gpu_tex.image,
        .transfer_src_optimal, .shader_read_only_optimal,
        .{ .transfer_read_bit = true }, .{ .shader_read_bit = true },
        .{ .transfer_bit = true }, .{ .fragment_shader_bit = true });
    self.oneTimeSubmitEnd(cmd) catch return error.TextureRead;

    const mapped = self.device.mapMemory(stg.mem, 0, vk.WHOLE_SIZE, .{}) catch return error.TextureRead;
    @memcpy(pixels_out[0..size], @as([*]const u8, @ptrCast(@alignCast(mapped)))[0..size]);
    self.device.unmapMemory(stg.mem);
}

pub fn textureFromTarget(self: *Self, target: dvui.TextureTarget) dvui.Backend.TextureError!dvui.Texture {
    // Transfer ownership: destroy the framebuffer + GpuRenderTarget wrapper,
    // but keep the GpuTexture alive (now owned by the returned Texture).
    const rt: *GpuRenderTarget = @ptrCast(@alignCast(target.ptr));
    const gpu_tex = rt.gpu_tex;
    // See textureDestroyTarget: the framebuffer may still be referenced by
    // the current frame's not-yet-submitted command buffer, so its
    // destruction must be deferred. gpu_tex is kept alive (transferred to
    // the returned Texture), so only the framebuffer is queued here.
    self.pending_destroys.append(self.allocator, .{
        .framebuffer = rt.framebuffer,
        .gpu_tex = null,
        .retire_frame = self.frame_counter + self.frames_in_flight,
    }) catch {
        self.device.queueWaitIdle(self.graphics_queue) catch {};
        self.device.destroyFramebuffer(rt.framebuffer, null);
    };
    self.allocator.destroy(rt);
    return .{ .ptr = gpu_tex, .width = target.width, .height = target.height, .format = target.format, .interpolation = target.interpolation, .wrap_u = target.wrap_u, .wrap_v = target.wrap_v };
}

pub fn textureFromTargetTemp(_: *Self, target: dvui.TextureTarget) dvui.Backend.TextureError!dvui.Texture {
    const rt: *GpuRenderTarget = @ptrCast(@alignCast(target.ptr));
    return .{ .ptr = rt.gpu_tex, .width = target.width, .height = target.height, .format = target.format, .interpolation = target.interpolation, .wrap_u = target.wrap_u, .wrap_v = target.wrap_v };
}

pub fn renderTarget(self: *Self, maybe_texture: ?dvui.TextureTarget) dvui.Backend.GenericError!void {
    const cmd = self.frame_cmd;

    // End the currently active render pass.
    self.device.cmdEndRenderPass(cmd);

    if (self.active_target) |prev| {
        // Transition previous offscreen target back to shader_read_only so it
        // can be sampled (e.g. in the case of nested targets).
        cmdTransitionImage(self.device, cmd, prev.gpu_tex.image,
            .color_attachment_optimal, .shader_read_only_optimal,
            .{ .color_attachment_write_bit = true }, .{ .shader_read_bit = true },
            .{ .color_attachment_output_bit = true }, .{ .fragment_shader_bit = true });
    }

    if (maybe_texture) |texture| {
        const rt: *GpuRenderTarget = @ptrCast(@alignCast(texture.ptr));
        self.active_target = rt;

        // Transition target image to color_attachment_optimal for rendering.
        cmdTransitionImage(self.device, cmd, rt.gpu_tex.image,
            .shader_read_only_optimal, .color_attachment_optimal,
            .{ .shader_read_bit = true }, .{ .color_attachment_write_bit = true },
            .{ .fragment_shader_bit = true }, .{ .color_attachment_output_bit = true });

        // Begin offscreen render pass (load_op=load, preserves content).
        self.device.cmdBeginRenderPass(cmd, &.{
            .render_pass = self.offscreen_rp,
            .framebuffer = rt.framebuffer,
            .render_area = .{ .offset = .{ .x = 0, .y = 0 }, .extent = .{ .width = rt.width, .height = rt.height } },
            .clear_value_count = 0,
            .p_clear_values = null,
        }, .@"inline");

        self.bindStateForExtent(self.offscreen_pipeline, .{ .width = rt.width, .height = rt.height });
    } else {
        // Restore main screen.
        self.active_target = null;

        // At this point the swapchain image is in present_src_khr (set by main_rp's
        // final_layout when we ended it above), so use main_rp_load.
        self.device.cmdBeginRenderPass(cmd, &.{
            .render_pass = self.main_rp_load,
            .framebuffer = self.frame_framebuffer,
            .render_area = .{ .offset = .{ .x = 0, .y = 0 }, .extent = self.frame_extent },
            .clear_value_count = 0,
            .p_clear_values = null,
        }, .@"inline");

        self.bindStateForExtent(self.main_pipeline, self.frame_extent);
    }
}

// ---- Private helpers ----

/// Rebind pipeline, vertex/index buffers, viewport, and push orthographic projection.
/// Does NOT reset vtx_head / idx_head.
fn bindStateForExtent(self: *Self, pipeline: vk.Pipeline, extent: vk.Extent2D) void {
    const cmd = self.frame_cmd;
    const w: f32 = @floatFromInt(extent.width);
    const h: f32 = @floatFromInt(extent.height);

    self.device.cmdBindPipeline(cmd, .graphics, pipeline);

    const vtx_offsets = [_]vk.DeviceSize{0};
    self.device.cmdBindVertexBuffers(cmd, 0, &[_]vk.Buffer{self.vtx_buf}, &vtx_offsets);

    const idx_type: vk.IndexType = switch (dvui.Vertex.Index) {
        u16 => .uint16,
        u32 => .uint32,
        else => @compileError("unsupported vertex index type"),
    };
    self.device.cmdBindIndexBuffer(cmd, self.idx_buf, 0, idx_type);

    self.device.cmdSetViewport(cmd, 0, &[_]vk.Viewport{.{
        .x = 0, .y = 0, .width = w, .height = h, .min_depth = 0, .max_depth = 1,
    }});

    // Orthographic projection mapping screen [0,w]×[0,h] to NDC [-1,1]×[-1,1].
    // Column-major (GLSL mat4 convention).
    const proj = [16]f32{
        2.0 / w, 0,       0, 0,
        0,       2.0 / h, 0, 0,
        0,       0,       0, 0,
        -1,      -1,      0, 1,
    };
    self.device.cmdPushConstants(cmd, self.pipeline_layout, .{ .vertex_bit = true }, 0, 64, &proj);
}

/// Re-begin whichever render pass was active before a temporary interruption
/// (used by textureClearTarget).
fn resumeCurrentRenderPass(self: *Self) void {
    const cmd = self.frame_cmd;
    if (self.active_target) |rt| {
        // Transition back to color_attachment_optimal before re-entering offscreen pass.
        cmdTransitionImage(self.device, cmd, rt.gpu_tex.image,
            .shader_read_only_optimal, .color_attachment_optimal,
            .{ .shader_read_bit = true }, .{ .color_attachment_write_bit = true },
            .{ .fragment_shader_bit = true }, .{ .color_attachment_output_bit = true });

        self.device.cmdBeginRenderPass(cmd, &.{
            .render_pass = self.offscreen_rp,
            .framebuffer = rt.framebuffer,
            .render_area = .{ .offset = .{ .x = 0, .y = 0 }, .extent = .{ .width = rt.width, .height = rt.height } },
            .clear_value_count = 0,
            .p_clear_values = null,
        }, .@"inline");
        self.bindStateForExtent(self.offscreen_pipeline, .{ .width = rt.width, .height = rt.height });
    } else {
        self.device.cmdBeginRenderPass(cmd, &.{
            .render_pass = self.main_rp_load,
            .framebuffer = self.frame_framebuffer,
            .render_area = .{ .offset = .{ .x = 0, .y = 0 }, .extent = self.frame_extent },
            .clear_value_count = 0,
            .p_clear_values = null,
        }, .@"inline");
        self.bindStateForExtent(self.main_pipeline, self.frame_extent);
    }
}

/// Create a sampled-only texture (upload via staging buffer).
fn createSampledTexture(self: *Self, pixels: [*]const u8, width: u32, height: u32, interpolation: dvui.enums.TextureInterpolation, wrap_u: dvui.enums.TextureWrap, wrap_v: dvui.enums.TextureWrap) !*GpuTexture {
    const dev = self.device;
    const size: vk.DeviceSize = width * height * 4;

    const image = try dev.createImage(&.{
        .image_type = .@"2d",
        .format = offscreen_format,
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
        .memory_type_index = try findMemoryType(self.instance, self.physical_device,
            img_reqs.memory_type_bits, .{ .device_local_bit = true }),
    }, null);
    errdefer dev.freeMemory(img_mem, null);
    try dev.bindImageMemory(image, img_mem, 0);

    const stg = try createBuffer(self.instance, dev, self.physical_device, size,
        .{ .transfer_src_bit = true },
        .{ .host_visible_bit = true, .host_coherent_bit = true });
    defer { dev.destroyBuffer(stg.buf, null); dev.freeMemory(stg.mem, null); }

    const stg_ptr = try dev.mapMemory(stg.mem, 0, vk.WHOLE_SIZE, .{});
    @memcpy(@as([*]u8, @ptrCast(@alignCast(stg_ptr)))[0..size], pixels[0..size]);
    dev.unmapMemory(stg.mem);

    const cmd = try self.oneTimeSubmitBegin();
    cmdTransitionImage(dev, cmd, image,
        .undefined, .transfer_dst_optimal,
        .{}, .{ .transfer_write_bit = true },
        .{ .top_of_pipe_bit = true }, .{ .transfer_bit = true });
    dev.cmdCopyBufferToImage(cmd, stg.buf, image, .transfer_dst_optimal, &[_]vk.BufferImageCopy{.{
        .buffer_offset = 0, .buffer_row_length = 0, .buffer_image_height = 0,
        .image_subresource = .{ .aspect_mask = .{ .color_bit = true }, .mip_level = 0, .base_array_layer = 0, .layer_count = 1 },
        .image_offset = .{ .x = 0, .y = 0, .z = 0 },
        .image_extent = .{ .width = width, .height = height, .depth = 1 },
    }});
    cmdTransitionImage(dev, cmd, image,
        .transfer_dst_optimal, .shader_read_only_optimal,
        .{ .transfer_write_bit = true }, .{ .shader_read_bit = true },
        .{ .transfer_bit = true }, .{ .fragment_shader_bit = true });
    try self.oneTimeSubmitEnd(cmd);

    return self.finishGpuTexture(image, img_mem, width, height, interpolation, wrap_u, wrap_v);
}

/// Create a render-target texture (colour_attachment_bit | sampled_bit), cleared to transparent.
fn createOffscreenTexture(self: *Self, width: u32, height: u32, interpolation: dvui.enums.TextureInterpolation, wrap_u: dvui.enums.TextureWrap, wrap_v: dvui.enums.TextureWrap) !*GpuTexture {
    const dev = self.device;

    const image = try dev.createImage(&.{
        .image_type = .@"2d",
        .format = offscreen_format,
        .extent = .{ .width = width, .height = height, .depth = 1 },
        .mip_levels = 1, .array_layers = 1,
        .samples = .{ .@"1_bit" = true },
        .tiling = .optimal,
        .usage = .{ .color_attachment_bit = true, .sampled_bit = true, .transfer_dst_bit = true, .transfer_src_bit = true },
        .sharing_mode = .exclusive,
        .initial_layout = .undefined,
    }, null);
    errdefer dev.destroyImage(image, null);

    const img_reqs = dev.getImageMemoryRequirements(image);
    const img_mem = try dev.allocateMemory(&.{
        .allocation_size = img_reqs.size,
        .memory_type_index = try findMemoryType(self.instance, self.physical_device,
            img_reqs.memory_type_bits, .{ .device_local_bit = true }),
    }, null);
    errdefer dev.freeMemory(img_mem, null);
    try dev.bindImageMemory(image, img_mem, 0);

    // Clear image to transparent and leave in shader_read_only_optimal.
    const cmd = try self.oneTimeSubmitBegin();
    cmdTransitionImage(dev, cmd, image,
        .undefined, .transfer_dst_optimal,
        .{}, .{ .transfer_write_bit = true },
        .{ .top_of_pipe_bit = true }, .{ .transfer_bit = true });
    const clear_color = vk.ClearColorValue{ .float_32 = .{ 0, 0, 0, 0 } };
    dev.cmdClearColorImage(cmd, image, .transfer_dst_optimal, &clear_color,
        &[_]vk.ImageSubresourceRange{color_subresource});
    cmdTransitionImage(dev, cmd, image,
        .transfer_dst_optimal, .shader_read_only_optimal,
        .{ .transfer_write_bit = true }, .{ .shader_read_bit = true },
        .{ .transfer_bit = true }, .{ .fragment_shader_bit = true });
    try self.oneTimeSubmitEnd(cmd);

    return self.finishGpuTexture(image, img_mem, width, height, interpolation, wrap_u, wrap_v);
}

fn addressModeFromWrap(wrap: dvui.enums.TextureWrap) vk.SamplerAddressMode {
    return switch (wrap) {
        .clamp => .clamp_to_edge,
        .repeat => .repeat,
    };
}

/// Create the ImageView, Sampler, and DescriptorSet for a GpuTexture.
fn finishGpuTexture(self: *Self, image: vk.Image, img_mem: vk.DeviceMemory, width: u32, height: u32, interpolation: dvui.enums.TextureInterpolation, wrap_u: dvui.enums.TextureWrap, wrap_v: dvui.enums.TextureWrap) !*GpuTexture {
    _ = width;
    _ = height;
    const dev = self.device;

    const view = try dev.createImageView(&.{
        .image = image,
        .view_type = .@"2d",
        .format = offscreen_format,
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
        .address_mode_u = addressModeFromWrap(wrap_u),
        .address_mode_v = addressModeFromWrap(wrap_v),
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

    dev.updateDescriptorSets(&[_]vk.WriteDescriptorSet{.{
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
    }}, null);

    const gpu_tex = try self.allocator.create(GpuTexture);
    gpu_tex.* = .{ .image = image, .memory = img_mem, .view = view, .sampler = sampler, .descriptor = descriptor };
    return gpu_tex;
}

fn destroyGpuTexture(self: *Self, tex: *GpuTexture) void {
    _ = self.device.freeDescriptorSets(self.desc_pool, &[_]vk.DescriptorSet{tex.descriptor}) catch {};
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
    errdefer self.device.freeCommandBuffers(self.cmd_pool, &[_]vk.CommandBuffer{cmd});
    try self.device.beginCommandBuffer(cmd, &.{ .flags = .{ .one_time_submit_bit = true } });
    return cmd;
}

fn oneTimeSubmitEnd(self: *const Self, cmd: vk.CommandBuffer) !void {
    try self.device.endCommandBuffer(cmd);
    const submit = vk.SubmitInfo{ .command_buffer_count = 1, .p_command_buffers = @ptrCast(&cmd) };
    try self.device.queueSubmit(self.graphics_queue, &[_]vk.SubmitInfo{submit}, .null_handle);
    try self.device.queueWaitIdle(self.graphics_queue);
    self.device.freeCommandBuffers(self.cmd_pool, &[_]vk.CommandBuffer{cmd});
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
    const buf = try dev.createBuffer(&.{ .size = size, .usage = usage, .sharing_mode = .exclusive }, null);
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
    dev.cmdPipelineBarrier(cmd, src_stage, dst_stage, .{}, null, null,
        &[_]vk.ImageMemoryBarrier{.{
            .src_access_mask = src_access,
            .dst_access_mask = dst_access,
            .old_layout = old_layout,
            .new_layout = new_layout,
            .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresource_range = color_subresource,
        }});
}

/// Create a render pass with a single colour attachment.
fn createColorRenderPass(
    dev: *const Device,
    format: vk.Format,
    load_op: vk.AttachmentLoadOp,
    initial_layout: vk.ImageLayout,
    final_layout: vk.ImageLayout,
) !vk.RenderPass {
    const attachment = vk.AttachmentDescription{
        .format = format,
        .samples = .{ .@"1_bit" = true },
        .load_op = load_op,
        .store_op = .store,
        .stencil_load_op = .dont_care,
        .stencil_store_op = .dont_care,
        .initial_layout = initial_layout,
        .final_layout = final_layout,
    };
    const color_ref = vk.AttachmentReference{ .attachment = 0, .layout = .color_attachment_optimal };
    const subpass = vk.SubpassDescription{
        .pipeline_bind_point = .graphics,
        .color_attachment_count = 1,
        .p_color_attachments = @ptrCast(&color_ref),
    };
    const dependency = vk.SubpassDependency{
        .src_subpass = vk.SUBPASS_EXTERNAL,
        .dst_subpass = 0,
        .src_stage_mask = .{ .color_attachment_output_bit = true },
        .src_access_mask = .{ .color_attachment_write_bit = true },
        .dst_stage_mask = .{ .color_attachment_output_bit = true },
        .dst_access_mask = .{ .color_attachment_read_bit = true, .color_attachment_write_bit = true },
    };
    return dev.createRenderPass(&.{
        .attachment_count = 1,
        .p_attachments = @ptrCast(&attachment),
        .subpass_count = 1,
        .p_subpasses = @ptrCast(&subpass),
        .dependency_count = 1,
        .p_dependencies = @ptrCast(&dependency),
    }, null);
}

/// Create the dvui graphics pipeline against the given render pass.
fn createGraphicsPipeline(
    dev: *const Device,
    render_pass: vk.RenderPass,
    layout: vk.PipelineLayout,
    vert_mod: vk.ShaderModule,
    frag_mod: vk.ShaderModule,
) !vk.Pipeline {
    const shader_stages = [_]vk.PipelineShaderStageCreateInfo{
        .{ .stage = .{ .vertex_bit = true }, .module = vert_mod, .p_name = "main" },
        .{ .stage = .{ .fragment_bit = true }, .module = frag_mod, .p_name = "main" },
    };

    // dvui.Vertex: pos(f32×2)  col(u8×4)  uv(f32×2) — 20 bytes
    const vert_bindings = [_]vk.VertexInputBindingDescription{.{
        .binding = 0, .stride = @sizeOf(dvui.Vertex), .input_rate = .vertex,
    }};
    const vert_attribs = [_]vk.VertexInputAttributeDescription{
        .{ .location = 0, .binding = 0, .format = .r32g32_sfloat,   .offset = @offsetOf(dvui.Vertex, "pos") },
        .{ .location = 1, .binding = 0, .format = .r8g8b8a8_unorm,  .offset = @offsetOf(dvui.Vertex, "col") },
        .{ .location = 2, .binding = 0, .format = .r32g32_sfloat,   .offset = @offsetOf(dvui.Vertex, "uv") },
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

    const pipeline_ci = [_]vk.GraphicsPipelineCreateInfo{.{
        .stage_count = shader_stages.len,
        .p_stages = &shader_stages,
        .p_vertex_input_state = &.{
            .vertex_binding_description_count = vert_bindings.len,
            .p_vertex_binding_descriptions = &vert_bindings,
            .vertex_attribute_description_count = vert_attribs.len,
            .p_vertex_attribute_descriptions = &vert_attribs,
        },
        .p_input_assembly_state = &.{ .topology = .triangle_list, .primitive_restart_enable = .false },
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
        .layout = layout,
        .render_pass = render_pass,
        .subpass = 0,
        .base_pipeline_index = -1,
    }};

    var pipelines: [1]vk.Pipeline = .{.null_handle};
    _ = try dev.createGraphicsPipelines(.null_handle, &pipeline_ci, null, &pipelines);
    return pipelines[0];
}
