#version 330 core

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_src;
uniform vec2 u_texel_size;   // 1.0 / src_resolution
uniform int u_use_karis;     // 1 on first downsample mip to eliminate ASCII fireflies

float karisWeight(vec3 col) {
    float luma = dot(col, vec3(0.2126, 0.7152, 0.0722));
    return 1.0 / (1.0 + luma);
}

void main() {
    float x = u_texel_size.x;
    float y = u_texel_size.y;

    // Jorge Jimenez (CoD: Advanced Warfare) 13-tap Downsample Filter
    vec3 a = texture(u_src, vec2(v_uv.x - 2.0 * x, v_uv.y + 2.0 * y)).rgb;
    vec3 b = texture(u_src, vec2(v_uv.x,           v_uv.y + 2.0 * y)).rgb;
    vec3 c = texture(u_src, vec2(v_uv.x + 2.0 * x, v_uv.y + 2.0 * y)).rgb;

    vec3 d = texture(u_src, vec2(v_uv.x - 2.0 * x, v_uv.y)).rgb;
    vec3 e = texture(u_src, vec2(v_uv.x,           v_uv.y)).rgb;
    vec3 f = texture(u_src, vec2(v_uv.x + 2.0 * x, v_uv.y)).rgb;

    vec3 g = texture(u_src, vec2(v_uv.x - 2.0 * x, v_uv.y - 2.0 * y)).rgb;
    vec3 h = texture(u_src, vec2(v_uv.x,           v_uv.y - 2.0 * y)).rgb;
    vec3 i = texture(u_src, vec2(v_uv.x + 2.0 * x, v_uv.y - 2.0 * y)).rgb;

    vec3 j = texture(u_src, vec2(v_uv.x - x, v_uv.y + y)).rgb;
    vec3 k = texture(u_src, vec2(v_uv.x + x, v_uv.y + y)).rgb;
    vec3 l = texture(u_src, vec2(v_uv.x - x, v_uv.y - y)).rgb;
    vec3 m = texture(u_src, vec2(v_uv.x + x, v_uv.y - y)).rgb;

    vec3 result;
    if (u_use_karis == 1) {
        vec3 g0 = (a + b + d + e) * 0.25;
        vec3 g1 = (b + c + e + f) * 0.25;
        vec3 g2 = (d + e + g + h) * 0.25;
        vec3 g3 = (e + f + h + i) * 0.25;
        vec3 g4 = (j + k + l + m) * 0.25;

        float w0 = karisWeight(g0) * 0.125;
        float w1 = karisWeight(g1) * 0.125;
        float w2 = karisWeight(g2) * 0.125;
        float w3 = karisWeight(g3) * 0.125;
        float w4 = karisWeight(g4) * 0.5;

        float wSum = max(w0 + w1 + w2 + w3 + w4, 1e-5);
        result = (g0 * w0 + g1 * w1 + g2 * w2 + g3 * w3 + g4 * w4) / wSum;
    } else {
        result = e * 0.125;
        result += (a + c + g + i) * 0.03125;
        result += (b + d + f + h) * 0.0625;
        result += (j + k + l + m) * 0.125;
    }

    fragColor = vec4(max(result, vec3(0.0)), 1.0);
}
