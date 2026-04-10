//! dvui + Vulkan integration example / template.
//!
//! This shows exactly where dvui slots into an existing Vulkan render loop.
//! Everything marked "YOUR APP" is boilerplate you already have in your renderer.
//!
//! Build:  zig build -Dbackend=vulkan
//! Env:    VULKAN_SDK must be set (or pass -Dvk_registry=... and -Dglslc=...)

const std = @import("std");
const ArrayList = std.ArrayList;
const Allocator = std.mem.Allocator;
const dvui = @import("dvui");
const Backend = @import("glfw-backend");
const vk = @import("vulkan");
const zglfw = Backend.zglfw;

const BaseWrapper = vk.BaseWrapper;
const InstanceWrapper = vk.InstanceWrapper;
const DeviceWrapper = vk.DeviceWrapper;
const Instance = vk.InstanceProxy;
const Device = vk.DeviceProxy;

var gpa_instance = std.heap.GeneralPurposeAllocator(.{}){};
const gpa = gpa_instance.allocator();

// Function-pointer dispatch tables (one per layer).
var vkb: BaseWrapper = undefined;
var vki: InstanceWrapper = undefined;
var vkd: DeviceWrapper = undefined;

const MAX_FRAMES_IN_FLIGHT: usize = 2;

const QueueFamilyIndices = struct {
    graphics: u32,
    present: u32,
};

const SwapchainData = struct {
    handle: vk.SwapchainKHR,
    format: vk.Format,
    extent: vk.Extent2D,
    images: []vk.Image,
    views: []vk.ImageView,

    fn destroy(self: *SwapchainData, device: *const Device, allocator: Allocator) void {
        for (self.views) |view| device.destroyImageView(view, null);
        allocator.free(self.views);
        allocator.free(self.images);
        device.destroySwapchainKHR(self.handle, null);
    }
};

pub fn main() !void {
    if (@import("builtin").os.tag == .windows) { // optional
        // on windows graphical apps have no console, so output goes to nowhere - attach it manually. related: https://github.com/ziglang/zig/issues/4196
        try dvui.Backend.Common.windowsAttachConsole();
    }

    defer _ = gpa_instance.deinit();
    dvui.Examples.show_demo_window = true;

    // ------------------------------------------------------------------ //
    // YOUR APP: GLFW + Vulkan init
    // ------------------------------------------------------------------ //
    try zglfw.init();
    defer zglfw.terminate();

    std.debug.print("Making window...\n", .{});

    zglfw.windowHint(.client_api, .no_api); // Vulkan — no OpenGL context
    const window = try zglfw.Window.create(1280, 720, "My App + dvui", null);
    defer window.destroy();

    std.debug.print("Making instance...\n", .{});

    var instance = try createInstance(gpa);
    defer instance.destroyInstance(null);

    std.debug.print("Making surface...\n", .{});

    const surface = try createSurface(&instance, window);
    defer instance.destroySurfaceKHR(surface, null);

    const phys_dev = try selectPhysicalDevice(gpa, &instance, surface);
    const queue_families = try findQueueFamilies(gpa, &instance, phys_dev, surface);

    var device = try createDevice(gpa, &instance, phys_dev, queue_families);
    defer device.destroyDevice(null);

    const graphics_queue = device.getDeviceQueue(queue_families.graphics, 0);
    const present_queue = device.getDeviceQueue(queue_families.present, 0);

    var sc = try createSwapchain(gpa, &instance, &device, phys_dev, surface, queue_families, window);
    errdefer sc.destroy(&device, gpa);

    // Render pass: one color attachment, clear on load, present_src_khr final layout.
    // dvui's pipeline is compiled against this pass — keep them in sync.
    const render_pass = try createRenderPass(&device, sc.format);
    defer device.destroyRenderPass(render_pass, null);

    var framebuffers = try createFramebuffers(gpa, &device, render_pass, sc.views, sc.extent);
    errdefer {
        for (framebuffers) |fb| device.destroyFramebuffer(fb, null);
        gpa.free(framebuffers);
    }

    // Command pool with reset-per-buffer flag so we can re-record each frame.
    const cmd_pool = try device.createCommandPool(&.{
        .flags = .{ .reset_command_buffer_bit = true },
        .queue_family_index = queue_families.graphics,
    }, null);
    defer device.destroyCommandPool(cmd_pool, null);

    var cmd_bufs: [MAX_FRAMES_IN_FLIGHT]vk.CommandBuffer = undefined;
    try device.allocateCommandBuffers(&.{
        .command_pool = cmd_pool,
        .level = .primary,
        .command_buffer_count = MAX_FRAMES_IN_FLIGHT,
    }, &cmd_bufs);

    // Per-frame sync objects.
    var image_available_sems: [MAX_FRAMES_IN_FLIGHT]vk.Semaphore = undefined;
    var render_finished_sems: [MAX_FRAMES_IN_FLIGHT]vk.Semaphore = undefined;
    var in_flight_fences: [MAX_FRAMES_IN_FLIGHT]vk.Fence = undefined;
    for (0..MAX_FRAMES_IN_FLIGHT) |i| {
        image_available_sems[i] = try device.createSemaphore(&.{}, null);
        render_finished_sems[i] = try device.createSemaphore(&.{}, null);
        // Start signaled so the first frame doesn't wait forever.
        in_flight_fences[i] = try device.createFence(&.{ .flags = .{ .signaled_bit = true } }, null);
    }
    defer for (0..MAX_FRAMES_IN_FLIGHT) |i| {
        device.destroySemaphore(image_available_sems[i], null);
        device.destroySemaphore(render_finished_sems[i], null);
        device.destroyFence(in_flight_fences[i], null);
    };

    // ------------------------------------------------------------------ //
    // DVUI RENDERER INIT
    // dvui creates its own pipeline (compatible with render_pass),
    // descriptor pool, and host-visible vertex/index buffers.
    // It does NOT create an instance, device, swapchain, or framebuffers.
    // ------------------------------------------------------------------ //
    var renderer = try dvui.render_backend.init(gpa, dvui.render_backend.InitOptions{
        .instance = &instance,
        .device = &device,
        .physical_device = phys_dev,
        .graphics_queue = graphics_queue,
        .graphics_queue_family = queue_families.graphics,
        .render_pass = render_pass,
    });
    defer renderer.deinit();

    // ------------------------------------------------------------------ //
    // DVUI WINDOW/INPUT BACKEND (handles GLFW events — no Vulkan here)
    // ------------------------------------------------------------------ //
    var impl = Backend.init(gpa, window);
    defer impl.deinit();

    const backend = dvui.Backend.init(&impl, &renderer);
    var win = try dvui.Window.init(@src(), gpa, backend, .{});
    defer win.deinit();

    // ------------------------------------------------------------------ //
    // MAIN LOOP
    // ------------------------------------------------------------------ //
    var current_frame: usize = 0;

    std.debug.print("got to main loop...\n", .{});

    outer: while (!window.shouldClose()) {
        // Skip rendering while minimized (extent becomes 0×0).
        if (sc.extent.width == 0 or sc.extent.height == 0) {
            zglfw.pollEvents();
            continue;
        }

        // Wait for the GPU to finish with this frame's resources.
        _ = try device.waitForFences(1, @ptrCast(&in_flight_fences[current_frame]), .true, std.math.maxInt(u64));

        // Acquire the next swapchain image.
        const acquire = device.acquireNextImageKHR(
            sc.handle,
            std.math.maxInt(u64),
            image_available_sems[current_frame],
            .null_handle,
        ) catch |err| {
            if (err == error.OutOfDateKHR) {
                try recreateSwapchain(gpa, &instance, &device, phys_dev, surface, queue_families, window, render_pass, &sc, &framebuffers);
                continue :outer;
            }
            return err;
        };
        if (acquire.result == .suboptimal_khr) {
            // Render this frame but recreate before next.
            try recreateSwapchain(gpa, &instance, &device, phys_dev, surface, queue_families, window, render_pass, &sc, &framebuffers);
            continue :outer;
        }
        const img_index = acquire.image_index;

        // Reset fence only after a successful acquire.
        try device.resetFences(1, @ptrCast(&in_flight_fences[current_frame]));

        // Record commands.
        const cmd = cmd_bufs[current_frame];
        try device.resetCommandBuffer(cmd, .{});
        try device.beginCommandBuffer(cmd, &.{ .flags = .{ .one_time_submit_bit = true } });

        const clear_value = vk.ClearValue{ .color = .{ .float_32 = .{ 0.1, 0.1, 0.1, 1.0 } } };
        device.cmdBeginRenderPass(cmd, &.{
            .render_pass = render_pass,
            .framebuffer = framebuffers[img_index],
            .render_area = .{ .offset = .{ .x = 0, .y = 0 }, .extent = sc.extent },
            .clear_value_count = 1,
            .p_clear_values = @ptrCast(&clear_value),
        }, .@"inline");

        // ---- DVUI INTEGRATION POINT ----
        // After vkCmdBeginRenderPass and before vkCmdEndRenderPass:
        renderer.setFrame(cmd, sc.extent);
        impl.addAllEvents(&win);
        try win.begin(std.time.nanoTimestamp());

        // Your dvui widgets here:
        dvui.Examples.demo(.full);

        const endtime = try win.end(.{});

        device.cmdEndRenderPass(cmd);
        try device.endCommandBuffer(cmd);

        // Submit.
        const wait_stage = vk.PipelineStageFlags{ .color_attachment_output_bit = true };
        const submit_info = vk.SubmitInfo{
            .wait_semaphore_count = 1,
            .p_wait_semaphores = @ptrCast(&image_available_sems[current_frame]),
            .p_wait_dst_stage_mask = @ptrCast(&wait_stage),
            .command_buffer_count = 1,
            .p_command_buffers = @ptrCast(&cmd),
            .signal_semaphore_count = 1,
            .p_signal_semaphores = @ptrCast(&render_finished_sems[current_frame]),
        };
        try device.queueSubmit(graphics_queue, 1, @ptrCast(&submit_info), in_flight_fences[current_frame]);

        // Present.
        const present_info = vk.PresentInfoKHR{
            .wait_semaphore_count = 1,
            .p_wait_semaphores = @ptrCast(&render_finished_sems[current_frame]),
            .swapchain_count = 1,
            .p_swapchains = @ptrCast(&sc.handle),
            .p_image_indices = @ptrCast(&img_index),
        };
        if (device.queuePresentKHR(present_queue, &present_info)) |_| {} else |err| {
            if (err != error.OutOfDateKHR) return err;
            // Out-of-date on present: recreate at the top of the next loop iteration.
        }

        current_frame = (current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
        impl.pollEventsTimeout(&win, endtime);
    }

    // Wait for all in-flight work before cleanup.
    try device.deviceWaitIdle();
    for (framebuffers) |fb| device.destroyFramebuffer(fb, null);
    gpa.free(framebuffers);
    sc.destroy(&device, gpa);
}

// ---- Vulkan helper functions ----

fn createSurface(instance: *const Instance, window: *zglfw.Window) !vk.SurfaceKHR {
    var surface: vk.SurfaceKHR = undefined;
    // zglfw.createWindowSurface takes a raw ?*const anyopaque for the VkInstance handle.
    const raw_instance: ?*const anyopaque = @ptrFromInt(@intFromEnum(instance.handle));
    try zglfw.createWindowSurface(raw_instance, window, null, &surface);
    return surface;
}

fn findQueueFamilies(allocator: Allocator, instance: *const Instance, phys_dev: vk.PhysicalDevice, surface: vk.SurfaceKHR) !QueueFamilyIndices {
    const props = try instance.getPhysicalDeviceQueueFamilyPropertiesAlloc(phys_dev, allocator);
    defer allocator.free(props);

    var graphics: ?u32 = null;
    var present_family: ?u32 = null;

    for (props, 0..) |prop, i| {
        const idx: u32 = @intCast(i);
        if (prop.queue_flags.graphics_bit) graphics = idx;
        if (try instance.getPhysicalDeviceSurfaceSupportKHR(phys_dev, idx, surface) == .true) {
            present_family = idx;
        }
        if (graphics != null and present_family != null) break;
    }

    return .{
        .graphics = graphics orelse return error.NoGraphicsQueue,
        .present = present_family orelse return error.NoPresentQueue,
    };
}

fn chooseSwapSurfaceFormat(formats: []const vk.SurfaceFormatKHR) vk.SurfaceFormatKHR {
    // Prefer B8G8R8A8_UNORM: the Vulkan spec guarantees it supports
    // VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT (required for dvui blending).
    // B8G8R8A8_SRGB does NOT carry that guarantee and fails on some hardware.
    for (formats) |fmt| {
        if (fmt.format == .b8g8r8a8_unorm and fmt.color_space == .srgb_nonlinear_khr) return fmt;
    }
    // Also accept R8G8B8A8_UNORM — same blending guarantee, different channel order.
    for (formats) |fmt| {
        if (fmt.format == .r8g8b8a8_unorm and fmt.color_space == .srgb_nonlinear_khr) return fmt;
    }
    return formats[0];
}

fn chooseSwapPresentMode(modes: []const vk.PresentModeKHR) vk.PresentModeKHR {
    for (modes) |mode| {
        if (mode == .mailbox_khr) return mode;
        if (mode == .immediate_khr) return mode;
    }
    return .fifo_khr; // Guaranteed to be available.
}

fn chooseSwapExtent(caps: vk.SurfaceCapabilitiesKHR, window: *zglfw.Window) vk.Extent2D {
    // If current_extent is not the sentinel value, use it directly.
    if (caps.current_extent.width != std.math.maxInt(u32)) return caps.current_extent;
    const fw, const fh = window.getFramebufferSize();
    return .{
        .width = std.math.clamp(@as(u32, @intCast(@max(0, fw))), caps.min_image_extent.width, caps.max_image_extent.width),
        .height = std.math.clamp(@as(u32, @intCast(@max(0, fh))), caps.min_image_extent.height, caps.max_image_extent.height),
    };
}

fn createSwapchain(
    allocator: Allocator,
    instance: *const Instance,
    device: *const Device,
    phys_dev: vk.PhysicalDevice,
    surface: vk.SurfaceKHR,
    queue_families: QueueFamilyIndices,
    window: *zglfw.Window,
) !SwapchainData {
    const caps = try instance.getPhysicalDeviceSurfaceCapabilitiesKHR(phys_dev, surface);
    const formats = try instance.getPhysicalDeviceSurfaceFormatsAllocKHR(phys_dev, surface, allocator);
    defer allocator.free(formats);
    const present_modes = try instance.getPhysicalDeviceSurfacePresentModesAllocKHR(phys_dev, surface, allocator);
    defer allocator.free(present_modes);

    const format = chooseSwapSurfaceFormat(formats);
    const present_mode = chooseSwapPresentMode(present_modes);
    const extent = chooseSwapExtent(caps, window);

    // Request one more image than the minimum so the CPU is less likely to stall.
    var image_count = caps.min_image_count + 1;
    if (caps.max_image_count > 0) image_count = @min(image_count, caps.max_image_count);

    const sharing_mode: vk.SharingMode = if (queue_families.graphics != queue_families.present) .concurrent else .exclusive;
    // const qf_indices = [_]u32{ queue_families.graphics, queue_families.present };

    // Find the transformation of the surface
    const pre_transform: vk.SurfaceTransformFlagsKHR = init: {
        if (caps.supported_transforms.identity_bit_khr) {
            break :init .{ .identity_bit_khr = true };
        } else {
            break :init caps.current_transform;
        }
    };

    // Find a supported composite alpha format (not all devices support alpha opaque)
    var composite_alpha: vk.CompositeAlphaFlagsKHR = .{ .opaque_bit_khr = true };
    const composite_alpha_flags = [_]vk.CompositeAlphaFlagsKHR{
        .{ .opaque_bit_khr = true },
        .{ .pre_multiplied_bit_khr = true },
        .{ .post_multiplied_bit_khr = true },
        .{ .inherit_bit_khr = true },
    };

    for (composite_alpha_flags) |composite_alpha_flag| {
        if (caps.supported_composite_alpha.contains(composite_alpha_flag)) {
            composite_alpha = composite_alpha_flag;
            break;
        }
    }

    const handle = try device.createSwapchainKHR(&.{
        .surface = surface,
        .min_image_count = image_count,
        .image_format = format.format,
        .image_color_space = format.color_space,
        .image_extent = extent,
        .image_array_layers = 1,
        .image_usage = caps.supported_usage_flags,
        .image_sharing_mode = sharing_mode,
        .queue_family_index_count = 0,
        .p_queue_family_indices = null,
        .pre_transform = pre_transform,
        .composite_alpha = composite_alpha,
        .present_mode = present_mode,
        .clipped = .true,
        .old_swapchain = .null_handle,
    }, null);
    errdefer device.destroySwapchainKHR(handle, null);

    const images = try device.getSwapchainImagesAllocKHR(handle, allocator);
    errdefer allocator.free(images);

    const views = try allocator.alloc(vk.ImageView, images.len);
    errdefer allocator.free(views);

    var views_created: usize = 0;
    errdefer for (views[0..views_created]) |view| device.destroyImageView(view, null);

    for (images, views) |image, *view| {
        view.* = try device.createImageView(&.{
            .image = image,
            .view_type = .@"2d",
            .format = format.format,
            .components = .{ .r = .identity, .g = .identity, .b = .identity, .a = .identity },
            .subresource_range = .{
                .aspect_mask = .{ .color_bit = true },
                .base_mip_level = 0,
                .level_count = 1,
                .base_array_layer = 0,
                .layer_count = 1,
            },
        }, null);
        views_created += 1;
    }

    return .{
        .handle = handle,
        .format = format.format,
        .extent = extent,
        .images = images,
        .views = views,
    };
}

fn createRenderPass(device: *const Device, format: vk.Format) !vk.RenderPass {
    const color_attachment = vk.AttachmentDescription{
        .format = format,
        .samples = .{ .@"1_bit" = true },
        .load_op = .clear,
        .store_op = .store,
        .stencil_load_op = .dont_care,
        .stencil_store_op = .dont_care,
        .initial_layout = .undefined,
        .final_layout = .present_src_khr,
    };
    const color_ref = vk.AttachmentReference{
        .attachment = 0,
        .layout = .color_attachment_optimal,
    };
    const subpass = vk.SubpassDescription{
        .pipeline_bind_point = .graphics,
        .color_attachment_count = 1,
        .p_color_attachments = @ptrCast(&color_ref),
    };
    // Wait for color-attachment-output stage before writing.
    const dependency = vk.SubpassDependency{
        .src_subpass = vk.SUBPASS_EXTERNAL,
        .dst_subpass = 0,
        .src_stage_mask = .{ .color_attachment_output_bit = true },
        .src_access_mask = .{},
        .dst_stage_mask = .{ .color_attachment_output_bit = true },
        .dst_access_mask = .{ .color_attachment_write_bit = true },
    };
    return device.createRenderPass(&.{
        .attachment_count = 1,
        .p_attachments = @ptrCast(&color_attachment),
        .subpass_count = 1,
        .p_subpasses = @ptrCast(&subpass),
        .dependency_count = 1,
        .p_dependencies = @ptrCast(&dependency),
    }, null);
}

fn createFramebuffers(
    allocator: Allocator,
    device: *const Device,
    render_pass: vk.RenderPass,
    views: []const vk.ImageView,
    extent: vk.Extent2D,
) ![]vk.Framebuffer {
    const fbs = try allocator.alloc(vk.Framebuffer, views.len);
    errdefer allocator.free(fbs);
    var created: usize = 0;
    errdefer for (fbs[0..created]) |fb| device.destroyFramebuffer(fb, null);

    for (views, fbs) |view, *fb| {
        fb.* = try device.createFramebuffer(&.{
            .render_pass = render_pass,
            .attachment_count = 1,
            .p_attachments = @ptrCast(&view),
            .width = extent.width,
            .height = extent.height,
            .layers = 1,
        }, null);
        created += 1;
    }
    return fbs;
}

fn recreateSwapchain(
    allocator: Allocator,
    instance: *const Instance,
    device: *const Device,
    phys_dev: vk.PhysicalDevice,
    surface: vk.SurfaceKHR,
    queue_families: QueueFamilyIndices,
    window: *zglfw.Window,
    render_pass: vk.RenderPass,
    sc: *SwapchainData,
    framebuffers: *[]vk.Framebuffer,
) !void {
    try device.deviceWaitIdle();
    for (framebuffers.*) |fb| device.destroyFramebuffer(fb, null);
    allocator.free(framebuffers.*);
    sc.destroy(device, allocator);
    sc.* = try createSwapchain(allocator, instance, device, phys_dev, surface, queue_families, window);
    framebuffers.* = try createFramebuffers(allocator, device, render_pass, sc.views, sc.extent);
}

// ---- Instance / device setup ----

fn createInstance(allocator: Allocator) !Instance {
    const vk_get_instance_proc_addr: vk.PfnGetInstanceProcAddr = @ptrCast(&zglfw.getInstanceProcAddress);
    vkb = vk.BaseWrapper.load(vk_get_instance_proc_addr);

    var instance_extensions = try getRequiredInstanceExtensions(allocator);
    defer instance_extensions.deinit(allocator);

    const app_info = vk.ApplicationInfo{
        .p_application_name = "glfw-vulkan-ontop",
        .application_version = @bitCast(vk.makeApiVersion(1, 0, 0, 0)),
        .p_engine_name = "vulkan_backend",
        .engine_version = @bitCast(vk.makeApiVersion(1, 0, 0, 0)),
        .api_version = @bitCast(vk.makeApiVersion(0, 1, 4, 0)),
    };

    var instance_create_info: vk.InstanceCreateInfo = .{
        .flags = .{ .enumerate_portability_bit_khr = true },
        .p_application_info = &app_info,
    };

    if (instance_extensions.items.len > 0) {
        instance_create_info.enabled_extension_count = @intCast(instance_extensions.items.len);
        instance_create_info.pp_enabled_extension_names = instance_extensions.items.ptr;
    }

    const instance_initial = try vkb.createInstance(&instance_create_info, null);
    vki = InstanceWrapper.load(instance_initial, vkb.dispatch.vkGetInstanceProcAddr.?);
    return Instance.init(instance_initial, &vki);
}

fn getRequiredInstanceExtensions(allocator: Allocator) !std.ArrayList([*:0]const u8) {
    var exts: ArrayList([*:0]const u8) = .empty;
    const glfw_exts = try zglfw.getRequiredInstanceExtensions();
    try exts.appendSlice(allocator, @ptrCast(glfw_exts));
    return exts;
}

fn selectPhysicalDevice(allocator: Allocator, instance: *const Instance, surface: vk.SurfaceKHR) !vk.PhysicalDevice {
    const devices = try instance.enumeratePhysicalDevicesAlloc(allocator);
    defer allocator.free(devices);
    if (devices.len == 0) return error.NoGPUsSupportVulkan;
    for (devices) |dev| {
        if (try isDeviceSuitable(allocator, instance, dev, surface)) return dev;
    }
    return error.NoSuitableDevice;
}

fn isDeviceSuitable(allocator: Allocator, instance: *const Instance, dev: vk.PhysicalDevice, surface: vk.SurfaceKHR) !bool {
    if (!try checkDeviceExtensionSupport(allocator, instance, dev)) return false;
    // Swapchain adequacy: at least one surface format and one present mode.
    const formats = try instance.getPhysicalDeviceSurfaceFormatsAllocKHR(dev, surface, allocator);
    defer allocator.free(formats);
    const modes = try instance.getPhysicalDeviceSurfacePresentModesAllocKHR(dev, surface, allocator);
    defer allocator.free(modes);
    return formats.len > 0 and modes.len > 0;
}

fn checkDeviceExtensionSupport(allocator: Allocator, instance: *const Instance, dev: vk.PhysicalDevice) !bool {
    const available = try instance.enumerateDeviceExtensionPropertiesAlloc(dev, null, allocator);
    defer allocator.free(available);

    const required = [_][*:0]const u8{vk.extensions.khr_swapchain.name};
    for (required) |req| {
        for (available) |avail| {
            const len = std.mem.indexOfScalar(u8, &avail.extension_name, 0).?;
            if (std.mem.eql(u8, std.mem.span(req), avail.extension_name[0..len])) break;
        } else return false;
    }
    return true;
}

fn createDevice(allocator: Allocator, instance: *const Instance, phys_dev: vk.PhysicalDevice, queue_families: QueueFamilyIndices) !Device {
    const priority = [_]f32{1.0};
    var queue_cis: ArrayList(vk.DeviceQueueCreateInfo) = .empty;
    defer queue_cis.deinit(allocator);

    // Add one queue per unique family index.
    for ([_]u32{ queue_families.graphics, queue_families.present }) |family| {
        const already_added = blk: {
            for (queue_cis.items) |ci| {
                if (ci.queue_family_index == family) break :blk true;
            }
            break :blk false;
        };
        if (already_added) continue;
        try queue_cis.append(allocator, .{
            .queue_family_index = family,
            .queue_count = 1,
            .p_queue_priorities = &priority,
        });
    }

    const dev_extensions = [_][*:0]const u8{vk.extensions.khr_swapchain.name};

    const device_initial = try instance.createDevice(phys_dev, &.{
        .queue_create_info_count = @intCast(queue_cis.items.len),
        .p_queue_create_infos = queue_cis.items.ptr,
        .enabled_extension_count = dev_extensions.len,
        .pp_enabled_extension_names = &dev_extensions,
    }, null);

    vkd = DeviceWrapper.load(device_initial, instance.wrapper.dispatch.vkGetDeviceProcAddr.?);
    return Device.init(device_initial, &vkd);
}
