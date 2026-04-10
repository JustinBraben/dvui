#version 450

layout(location = 0) in vec4 f_col;
layout(location = 1) in vec2 f_uv;

layout(location = 0) out vec4 out_color;

// null_tex (1x1 white) is bound when no texture; sampling it returns (1,1,1,1)
// so out_color = f_col, identical to the use_sampler=false branch in the OpenGL renderer.
layout(set = 0, binding = 0) uniform sampler2D tex;

void main() {
    out_color = texture(tex, f_uv) * f_col;
}
