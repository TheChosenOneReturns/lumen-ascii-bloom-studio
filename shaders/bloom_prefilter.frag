#version 330 core

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_scene;
uniform float u_threshold;
uniform float u_soft_knee;
uniform float u_clamp_max;
uniform vec3 u_bloom_tint;

void main() {
    vec3 col = texture(u_scene, v_uv).rgb;
    col = min(col, vec3(u_clamp_max));

    float br = max(col.r, max(col.g, col.b));
    float knee = max(u_threshold * u_soft_knee, 0.0001);
    float soft = br - u_threshold + knee;
    soft = clamp(soft, 0.0, 2.0 * knee);
    soft = (soft * soft) / (4.0 * knee + 1e-5);

    float contrib = max(soft, br - u_threshold) / max(br, 1e-5);
    vec3 bright = col * max(contrib, 0.0) * u_bloom_tint;

    fragColor = vec4(max(bright, vec3(0.0)), 1.0);
}
