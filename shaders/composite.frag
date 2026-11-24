#version 330 core

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_scene_hdr;
uniform sampler2D u_bloom;
uniform sampler2D u_source;
uniform sampler2D u_edges;

uniform vec2 u_resolution;
uniform float u_time;

// Bloom & Tone Mapping
uniform float u_bloom_intensity;
uniform float u_exposure;
uniform int u_tonemap_mode;     // 0: ACES Filmic, 1: AgX-ish Soft, 2: Reinhard

// Post-FX
uniform float u_chromatic_aberration;
uniform float u_scanlines;
uniform float u_curvature;
uniform float u_vignette;
uniform float u_grain;

// Audio modulation & glitch accent
uniform vec4 u_audio;           // (kick, bass, vocals, energy)
uniform float u_audio_reactivity;
uniform float u_flash_accent;
uniform float u_glitch_tear;
uniform float u_prism_split;    // Extra directional RGB prism dispersion (Prism/Kaleido displacement!)

// Interactive Viewport Zoom / Pan & Split-Screen Curtain
uniform float u_zoom;
uniform vec2 u_pan;
uniform int u_split_enabled;
uniform float u_split_pos;      // 0..1 screen X
uniform int u_split_mode;       // 0: Original Video, 1: Sobel/DoG Edge Map, 2: Isolated Bloom

vec3 acesTonemap(vec3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

vec3 softFilmicTonemap(vec3 x) {
    // High-saturation filmic curve that preserves neon hues nicely
    vec3 m = x / (1.0 + x);
    return pow(clamp(m * 1.18, 0.0, 1.0), vec3(0.95));
}

float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

vec2 applyCurvature(vec2 uv, float amount) {
    if (amount <= 0.001) return uv;
    vec2 centered = uv * 2.0 - 1.0;
    vec2 offset = abs(centered.yx) / vec2(6.5 - amount * 3.5);
    centered = centered + centered * offset * offset * amount * 1.8;
    return centered * 0.5 + 0.5;
}

void main() {
    // Apply interactive zoom & pan centered at (0.5, 0.5)
    vec2 uv = (v_uv - 0.5) / max(u_zoom, 0.25) + 0.5 + u_pan;
    uv = applyCurvature(uv, u_curvature);

    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        fragColor = vec4(0.01, 0.01, 0.015, 1.0);
        return;
    }

    float audioLow = u_audio.x * u_audio_reactivity;
    float audioHigh = u_audio.z * u_audio_reactivity;

    // Chromatic aberration + directional Prism RGB split radiating/rotating with music
    float caAmount = (u_chromatic_aberration + audioHigh * 0.006) * length(uv - 0.5);
    vec2 caDir = normalize(uv - 0.5 + vec2(1e-5)) * caAmount;
    vec2 prismDir = vec2(cos(u_time * 1.4), sin(u_time * 1.4)) * (u_prism_split * 0.018);

    vec3 sceneHDR;
    vec3 bloomHDR;
    if (caAmount > 0.0001 || u_prism_split > 0.001) {
        vec2 offR = caDir + prismDir;
        vec2 offB = -caDir - prismDir;
        vec2 offG = vec2(-prismDir.y, prismDir.x) * 0.45;
        sceneHDR = vec3(
            texture(u_scene_hdr, clamp(uv + offR, 0.0, 1.0)).r,
            texture(u_scene_hdr, clamp(uv + offG, 0.0, 1.0)).g,
            texture(u_scene_hdr, clamp(uv + offB, 0.0, 1.0)).b
        );
        bloomHDR = vec3(
            texture(u_bloom, clamp(uv + offR * 1.5, 0.0, 1.0)).r,
            texture(u_bloom, clamp(uv + offG * 1.3, 0.0, 1.0)).g,
            texture(u_bloom, clamp(uv + offB * 1.5, 0.0, 1.0)).b
        );
    } else {
        sceneHDR = texture(u_scene_hdr, uv).rgb;
        bloomHDR = texture(u_bloom, uv).rgb;
    }

    float bloomMult = u_bloom_intensity * (1.0 + audioLow * 0.85);
    vec3 hdr = sceneHDR + bloomHDR * bloomMult;
    hdr *= u_exposure;

    vec3 col;
    if (u_tonemap_mode == 0) {
        col = acesTonemap(hdr);
    } else if (u_tonemap_mode == 1) {
        col = softFilmicTonemap(hdr);
    } else {
        col = hdr / (1.0 + hdr);
    }

    // CRT Scanlines
    if (u_scanlines > 0.001) {
        float scan = 0.5 + 0.5 * sin(uv.y * u_resolution.y * 3.14159);
        col *= mix(1.0, 0.72 + 0.28 * scan, u_scanlines);
    }

    // Vignette
    if (u_vignette > 0.001) {
        float dist = length((uv - 0.5) * vec2(1.15, 1.0));
        float vig = smoothstep(0.95, 0.25, dist);
        col *= mix(1.0, vig, u_vignette);
    }

    // Film grain
    if (u_grain > 0.001) {
        float n = (hash12(gl_FragCoord.xy + fract(u_time) * 100.0) - 0.5) * u_grain * 0.16;
        col = clamp(col + n, 0.0, 1.0);
    }

    // Crisp musical beat-1 / drop flash accent
    if (u_flash_accent > 0.001) {
        col = clamp(col + vec3(u_flash_accent * 0.55), 0.0, 1.0);
    }

    // --- Interactive Split-Screen Curtain ---
    if (u_split_enabled == 1) {
        float lineDist = abs(v_uv.x - u_split_pos) * u_resolution.x;
        if (v_uv.x < u_split_pos) {
            if (u_split_mode == 0) {
                col = texture(u_source, uv).rgb;
            } else if (u_split_mode == 1) {
                vec4 e = texture(u_edges, uv);
                float mag = clamp(length(e.xy) * 2.5, 0.0, 1.0);
                float ang = 0.5 * atan(e.y, e.x) / 3.14159 + 0.5;
                col = vec3(
                    0.5 + 0.5 * cos(6.28318 * (ang + 0.0)),
                    0.5 + 0.5 * cos(6.28318 * (ang + 0.33)),
                    0.5 + 0.5 * cos(6.28318 * (ang + 0.67))
                ) * mag;
            } else {
                col = acesTonemap(bloomHDR * bloomMult * u_exposure);
            }
        }
        // Sleek glowing split divider line
        if (lineDist < 2.0) {
            col = mix(col, vec3(0.45, 0.95, 1.0), 0.92);
        } else if (lineDist < 10.0) {
            col += vec3(0.15, 0.65, 0.9) * exp(-lineDist * 0.45) * 0.45;
        }
    }

    fragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
