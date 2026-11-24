#version 330 core

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_prev_paint;
uniform vec2 u_p0;             // previous cursor UV
uniform vec2 u_p1;             // current cursor UV
uniform vec2 u_resolution;
uniform float u_drawing;       // 1.0 if mouse button held in Brush mode
uniform float u_brush_radius;  // normalized radius
uniform vec3 u_brush_color;    // HDR brush color
uniform float u_paint_decay;   // 1.0 = permanent until cleared, <1.0 = fades over time

float distToSegment(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a;
    vec2 ba = b - a;
    float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
    return length(pa - ba * h);
}

void main() {
    vec3 prev = texture(u_prev_paint, v_uv).rgb * u_paint_decay;
    if (length(prev) < 0.0005) prev = vec3(0.0);

    vec3 stroke = vec3(0.0);
    if (u_drawing > 0.5) {
        vec2 aspect = vec2(u_resolution.x / max(u_resolution.y, 1.0), 1.0);
        float d = distToSegment(v_uv * aspect, u_p0 * aspect, u_p1 * aspect);
        float core = smoothstep(u_brush_radius, u_brush_radius * 0.15, d);
        float halo = exp(-d / max(u_brush_radius * 0.7, 0.002)) * 0.45;
        stroke = u_brush_color * (core * 2.2 + halo);
    }

    vec3 res = max(prev, stroke);
    fragColor = vec4(min(res, vec3(12.0)), 1.0);
}
