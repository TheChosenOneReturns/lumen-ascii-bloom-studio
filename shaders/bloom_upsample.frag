#version 330 core

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_lower_mip;   // Smaller, blurrier mip level
uniform sampler2D u_higher_mip;  // Current target resolution mip level
uniform vec2 u_texel_size;       // 1.0 / lower_mip resolution
uniform float u_filter_radius;   // Tent filter spread (default ~1.0)
uniform float u_scatter;         // Blend scatter weight (default ~0.72)
uniform float u_anamorphic;      // Horizontal stretch factor (1.0 = isotropic, 3.0 = cinema lens streak)

void main() {
    float x = u_texel_size.x * u_filter_radius * max(u_anamorphic, 1.0);
    float y = u_texel_size.y * u_filter_radius;

    // 3x3 Tent Filter Upsampling
    vec3 a = texture(u_lower_mip, vec2(v_uv.x - x, v_uv.y + y)).rgb;
    vec3 b = texture(u_lower_mip, vec2(v_uv.x,     v_uv.y + y)).rgb;
    vec3 c = texture(u_lower_mip, vec2(v_uv.x + x, v_uv.y + y)).rgb;

    vec3 d = texture(u_lower_mip, vec2(v_uv.x - x, v_uv.y)).rgb;
    vec3 e = texture(u_lower_mip, vec2(v_uv.x,     v_uv.y)).rgb;
    vec3 f = texture(u_lower_mip, vec2(v_uv.x + x, v_uv.y)).rgb;

    vec3 g = texture(u_lower_mip, vec2(v_uv.x - x, v_uv.y - y)).rgb;
    vec3 h = texture(u_lower_mip, vec2(v_uv.x,     v_uv.y - y)).rgb;
    vec3 i = texture(u_lower_mip, vec2(v_uv.x + x, v_uv.y - y)).rgb;

    vec3 upsampled = e * 4.0;
    upsampled += (b + d + f + h) * 2.0;
    upsampled += (a + c + g + i);
    upsampled *= (1.0 / 16.0);

    vec3 highCol = texture(u_higher_mip, v_uv).rgb;
    vec3 merged = highCol + upsampled * u_scatter;

    fragColor = vec4(max(merged, vec3(0.0)), 1.0);
}
