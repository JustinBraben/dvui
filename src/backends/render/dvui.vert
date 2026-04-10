#version 450

layout(location = 0) in vec2 v_pos;
layout(location = 1) in vec4 v_col; // VK_FORMAT_R8G8B8A8_UNORM: auto-normalized u8 -> [0,1]
layout(location = 2) in vec2 v_uv;

layout(location = 0) out vec4 f_col;
layout(location = 1) out vec2 f_uv;

layout(push_constant) uniform PC {
    mat4 proj;
} pc;

void main() {
    gl_Position = pc.proj * vec4(v_pos, 0.0, 1.0);
    f_col = v_col;
    f_uv = v_uv;
}
