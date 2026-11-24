#version 330 core

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_current;
uniform sampler2D u_prev_trail;
uniform sampler2D u_paint;
uniform float u_trail_decay;
uniform float u_trail_drift;       // radial expansion drift
uniform vec2 u_trail_drift_vec;    // directional wind drift (e.g. left-blowing dissolve like SUBJECT A-34!)
uniform float u_trail_dissolve;    // stochastic wind-particle breakup on trailing echo
uniform float u_time;

float hash21(vec2 p) {
    p = fract(p * vec2(127.1, 311.7));
    p += dot(p, p + 34.5);
    return fract(p.x * p.y);
}

void main() {
    vec3 cur = texture(u_current, v_uv).rgb;
    vec3 paint = texture(u_paint, v_uv).rgb;

    vec2 prevUV = v_uv;
    if (abs(u_trail_drift) > 0.0001) {
        prevUV = (prevUV - 0.5) * (1.0 - u_trail_drift * 0.004) + 0.5;
    }
    prevUV -= u_trail_drift_vec * 0.0065;

    vec3 prev = vec3(0.0);
    if (prevUV.x >= 0.0 && prevUV.x <= 1.0 && prevUV.y >= 0.0 && prevUV.y <= 1.0) {
        // Strictly bounded decay (< 0.94) so bright pixels NEVER latch solid white or freeze!
        float safeDecay = clamp(u_trail_decay, 0.0, 0.935);
        prev = texture(u_prev_trail, prevUV).rgb * safeDecay;
        if (u_trail_dissolve > 0.01) {
            float grain = hash21(floor(gl_FragCoord.xy * 0.5) + fract(u_time) * 97.0);
            float keep = smoothstep(u_trail_dissolve * 0.48, u_trail_dissolve * 0.48 + 0.18, grain);
            prev *= mix(1.0, keep, u_trail_dissolve);
        }
    }

    // Energy-conserving phosphor persistence (gain strictly <= 1.0 on prev -> zero pixel freeze!)
    vec3 combined = max(cur, prev) + cur * (0.08 * clamp(u_trail_decay, 0.0, 0.935));
    combined += paint;

    fragColor = vec4(min(combined, vec3(24.0)), 1.0);
}
