#version 330 core

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_source;
uniform sampler2D u_edges;
uniform sampler2D u_atlas;
uniform sampler2D u_fft_tex;     // 64x1 live perceptual Mel FFT texture

uniform vec2 u_resolution;
uniform vec2 u_cell_size;       // (cell_w, cell_h) in pixels
uniform int u_num_ramp;         // Number of luminance-sorted ramp glyphs
uniform int u_total_glyphs;     // u_num_ramp + 4 directional edge glyphs (|, -, /, \)

// Render mode: 0 = ASCII + DoG/Sobel, 1 = Stipple/Point-Cloud CRT Dither, 2 = Hybrid Stipple + ASCII Edges
uniform int u_render_mode;

// Musical Intelligence shader choreography uniforms
uniform vec4 u_audio_stems;     // (snare, hihat, beat_phase, drop_intensity)
uniform float u_hue_shift;      // 12-semitone chord/note hue rotation (-0.5..0.5)
uniform float u_spatial_eq;     // Spatial 64-band FFT glow modulation across canvas
uniform float u_glitch_tear;    // Snare/Drop horizontal glitch-slice offset

// Tone, Cutoff & Kinetic Scramble
uniform float u_brightness;
uniform float u_contrast;
uniform float u_gamma;
uniform float u_saturation;
uniform int u_invert;
uniform float u_dither;
uniform float u_bg_opacity;
uniform float u_glyph_sharpness;
uniform float u_black_cutoff;     // Cleanly zeroes dark backgrounds (Creation Hands / Kinetic Sphere)
uniform float u_scramble_speed;   // Hz rate of live kinetic glyph cycling
uniform float u_scramble_amount;  // 0 = pure luminance ramp, 1 = full kinetic polyglot scramble!
uniform float u_weight_contrast;  // Bold/bright vs thin/dim glyph contrast (MP4 sphere look)
uniform float u_stipple_scale;    // Pixel size of stipple points (1.5 .. 6.0)
uniform float u_stipple_density;  // Stipple threshold bias

// Edge-aware ASCII (Acerola DoG + Sobel)
uniform float u_edge_strength;
uniform float u_edge_threshold;
uniform float u_edge_boost;

// Color & HDR emission
// 0: Original, 1: Cyber, 2: Phosphor, 3: Amber, 4: Solar, 5: Thermal, 6: Synthwave,
// 7: Duotone, 8: Specimen Violet, 9: Subject Ice-Cyan, 10: Monochrome Noir Ink
uniform int u_color_mode;
uniform vec3 u_color_dark;
uniform vec3 u_color_bright;
uniform float u_hdr_boost;
uniform float u_hdr_power;
uniform float u_time;
uniform float u_matrix_rain;

// Interactive canvas physics
uniform vec2 u_mouse;
uniform float u_mouse_active;
uniform int u_mouse_mode;
uniform float u_interact_radius;
uniform float u_interact_strength;
uniform vec4 u_ripples[16];
uniform int u_ripple_count;

// Audio FFT reactivity
uniform vec4 u_audio;            // (kick, bass, vocals, energy)
uniform float u_audio_reactivity;

// GPU Displacement & Kinetic Motion Engine (11 real-time spatial animation modes!)
uniform int u_disp_mode;         // 0=Off .. 11=Pixel Sort Cascade
uniform float u_disp_strength;   // Effective strength (base + audio modulation)
uniform float u_disp_speed;      // Effective animation speed
uniform float u_disp_scale;      // Spatial frequency / wavelength
uniform vec4 u_disp_audio;       // (pulse, glitch, swirl, beat_phase)

float bayer4x4(vec2 cellCoord) {
    int x = int(mod(cellCoord.x, 4.0));
    int y = int(mod(cellCoord.y, 4.0));
    int idx = y * 4 + x;
    const float m[16] = float[16](
         0.0,  8.0,  2.0, 10.0,
        12.0,  4.0, 14.0,  6.0,
         3.0, 11.0,  1.0,  9.0,
        15.0,  7.0, 13.0,  5.0
    );
    return (m[idx] / 16.0) - 0.5;
}

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

// Interleaved Gradient Noise — crisp blue-noise-like stipple distribution!
float ignNoise(vec2 p) {
    vec3 magic = vec3(0.06711056, 0.00583715, 52.9829189);
    return fract(magic.z * fract(dot(p, magic.xy)));
}

// Smooth 2D value noise + divergence-free Curl Noise (solenoidal fluid field!)
float vnoise2D(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    float a = hash21(i);
    float b = hash21(i + vec2(1.0, 0.0));
    float c = hash21(i + vec2(0.0, 1.0));
    float d = hash21(i + vec2(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

float fbm2D(vec2 p) {
    float v = 0.0;
    float amp = 0.52;
    mat2 rot = mat2(0.80, -0.60, 0.60, 0.80);
    for (int i = 0; i < 3; i++) {
        v += amp * vnoise2D(p);
        p = rot * p * 2.04 + vec2(1.7, 9.2);
        amp *= 0.5;
    }
    return v;
}

vec2 curl2D(vec2 p) {
    const float eps = 0.025;
    float n1 = fbm2D(p + vec2(0.0, eps));
    float n2 = fbm2D(p - vec2(0.0, eps));
    float n3 = fbm2D(p + vec2(eps, 0.0));
    float n4 = fbm2D(p - vec2(eps, 0.0));
    return vec2((n1 - n2), -(n3 - n4)) / (2.0 * eps);
}

// 11-Mode Audio-Reactive GPU Spatial Displacement Engine!
vec2 computeDisplacementUV(vec2 uv, vec2 aspect) {
    if (u_disp_mode <= 0 || u_disp_strength <= 0.001) {
        return uv;
    }
    float str = u_disp_strength;
    float spd = max(u_disp_speed, 0.05);
    float sc = max(u_disp_scale, 0.25);
    float t = u_time * spd;
    float pulse = u_disp_audio.x;
    float glitch = u_disp_audio.y;
    float swirlAmt = u_disp_audio.z;
    float beatPh = u_disp_audio.w;

    vec2 centered = (uv - 0.5) * aspect;
    float dist = length(centered);
    vec2 dir = dist > 0.0001 ? (centered / dist) : vec2(0.0);
    vec2 outUV = uv;

    if (u_disp_mode == 1) {
        // 1: LIQUID CURL FLOW (True divergence-free solenoidal fluid motion!)
        vec2 c = curl2D(uv * (2.6 * sc) + vec2(t * 0.28, -t * 0.21));
        outUV += (c / aspect) * (0.028 * str * (1.0 + swirlAmt * 0.55));
    } else if (u_disp_mode == 2) {
        // 2: ACOUSTIC SHOCKWAVE (Expanding bass-synced concentric pressure rings!)
        float wave = sin(dist * (18.0 * sc) - t * 5.2 - beatPh * 6.28318);
        float env = exp(-dist * 1.15) * (0.65 + pulse * 0.95);
        outUV -= (dir / aspect) * wave * env * 0.042 * str;
    } else if (u_disp_mode == 3) {
        // 3: SILK WAVE DRIFT (Dual-axis woven harmonic ocean-silk ripples)
        float wx = sin(uv.y * (10.0 * sc) + t * 2.1) * cos(uv.x * (4.0 * sc) - t * 1.3);
        float wy = cos(uv.x * (9.0 * sc) - t * 1.9) * sin(uv.y * (5.0 * sc) + t * 1.5);
        outUV += vec2(wx, wy) * (0.032 * str * (1.0 + swirlAmt * 0.45));
    } else if (u_disp_mode == 4) {
        // 4: SLIT-SCAN ECHO (TouchDesigner cinematic time-ribbon horizontal shear)
        float band = sin(uv.y * (14.0 * sc) - t * 2.6);
        float fine = sin(uv.y * (36.0 * sc) + t * 4.4) * 0.35;
        float shear = (band + fine) * (0.045 * str * (0.8 + pulse * 0.65));
        outUV.x += shear;
        outUV.y += cos(uv.x * (8.0 * sc) + t * 1.8) * 0.012 * str;
    } else if (u_disp_mode == 5) {
        // 5: 3D DEPTH PARALLAX (Luminance-height pseudo-3D Lissajous orbital camera!)
        vec3 baseSample = texture(u_source, uv).rgb;
        float height = dot(baseSample, vec3(0.2126, 0.7152, 0.0722));
        vec2 camOrbit = vec2(sin(t * 0.95), cos(t * 0.72) * 0.75) * (0.055 * str * (1.0 + pulse * 0.55));
        outUV -= camOrbit * (height - 0.35);
    } else if (u_disp_mode == 6) {
        // 6: CYBER GLITCH BLOCKS (Quantized datamosh horizontal block shifts on snares!)
        float row = floor(uv.y * (16.0 * sc));
        float col = floor(uv.x * (6.0 * sc));
        float tick = floor(t * 14.0);
        float r1 = hash21(vec2(row, tick));
        float r2 = hash21(vec2(row + col * 3.1, tick + 7.7));
        float gate = step(0.68 - min(0.45, glitch * 0.35), r1);
        float shiftX = (r2 - 0.5) * 0.16 * str * (0.65 + glitch * 0.95) * gate;
        float shiftY = (hash21(vec2(col, tick)) - 0.5) * 0.04 * str * gate;
        outUV += vec2(shiftX, shiftY);
    } else if (u_disp_mode == 7) {
        // 7: VORTEX SINGULARITY (Hypnotic radial gravity swirl breathing with music)
        float ang = exp(-dist * (1.45 / max(sc * 0.7, 0.35))) * (1.15 * str * (0.85 + swirlAmt * 0.65));
        ang *= sin(t * 0.85) * 0.65 + 0.55;
        float cs = cos(ang);
        float sn = sin(ang);
        vec2 rotC = mat2(cs, -sn, sn, cs) * centered;
        outUV = (rotC / aspect) + 0.5;
    } else if (u_disp_mode == 8) {
        // 8: PRISM RGB SPLIT (Subtle wavy prism lens warp paired with composite chromatic dispersion)
        vec2 w = vec2(sin(uv.y * 8.0 * sc + t * 2.4), cos(uv.x * 8.0 * sc - t * 2.1));
        outUV += w * (0.022 * str * (1.0 + pulse * 0.65));
    } else if (u_disp_mode == 9) {
        // 9: KALEIDO MIRROR DRIFT (Rotating VJ polar mirror fold symmetry!)
        float folds = max(3.0, floor(4.0 + sc * 2.0));
        float a = atan(centered.y, centered.x) + t * 0.38;
        float seg = 6.2831853 / folds;
        a = mod(a, seg);
        a = abs(a - seg * 0.5);
        vec2 kPos = vec2(cos(a), sin(a)) * dist;
        vec2 kUV = (kPos / aspect) + 0.5;
        outUV = mix(uv, kUV, clamp(str * 0.78, 0.0, 1.0));
    } else if (u_disp_mode == 10) {
        // 10: ELASTIC LENS BOUNCE (Spring-physics barrel/pincushion lens warp + handheld float)
        float bounce = sin(t * 2.8) * 0.25 + pulse * 0.85;
        float warp = 1.0 - (dist * dist) * 0.38 * str * bounce;
        vec2 sway = vec2(sin(t * 1.15), cos(t * 0.90)) * (0.020 * str);
        outUV = ((centered * warp) / aspect) + 0.5 + sway;
    } else {
        // 11: PIXEL SORT CASCADE (Luminance-threshold vertical waterfall melt!)
        float colBin = floor(uv.x * (64.0 * sc));
        float colRnd = hash21(vec2(colBin, 3.7));
        vec3 s0 = texture(u_source, uv).rgb;
        float lum0 = dot(s0, vec3(0.2126, 0.7152, 0.0722));
        float flow = smoothstep(0.22, 0.88, lum0) * (0.4 + 0.6 * colRnd);
        float drop = fract(t * (0.22 + colRnd * 0.35)) * 0.14 * str * (0.85 + pulse * 0.65);
        outUV.y += flow * drop;
        outUV.x += sin(uv.y * 12.0 * sc + t * 2.0) * 0.008 * str * flow;
    }

    // Mirror-wrap edges cleanly so strong displacements NEVER leave ugly stretched border streaks!
    vec2 m = mod(outUV, 2.0);
    return mix(m, 2.0 - m, step(1.0, m));
}

vec3 adjustSaturation(vec3 col, float sat) {
    float l = dot(col, vec3(0.2126, 0.7152, 0.0722));
    return mix(vec3(l), col, sat);
}

// Rodrigues color rotation around luminance axis (1,1,1) for musical pitch-to-hue shifts!
vec3 rotateHue(vec3 col, float turns) {
    if (abs(turns) < 0.001) return col;
    float angle = turns * 6.2831853;
    float c = cos(angle);
    float s = sin(angle);
    vec3 k = vec3(0.57735027);
    return max(vec3(0.0), col * c + cross(k, col) * s + k * dot(k, col) * (1.0 - c));
}

float sampleSpatialEQ(vec2 uv, vec2 aspect) {
    if (u_spatial_eq <= 0.001) return 0.0;
    // Map radial distance (low bass in center, shimmering treble on outer rings) + horizontal sweep
    float radialU = clamp(length((uv - 0.5) * aspect) * 1.15, 0.02, 0.98);
    float eqBin = texture(u_fft_tex, vec2(radialU, 0.5)).r;
    return eqBin * u_spatial_eq;
}

vec3 mapPalette(vec3 origCol, float lum, int mode) {
    lum = clamp(lum, 0.0, 1.0);
    if (mode == 0) {
        return adjustSaturation(origCol, u_saturation);
    } else if (mode == 1) {
        // Cyberpunk Neon
        vec3 c0 = vec3(0.04, 0.02, 0.16);
        vec3 c1 = vec3(0.05, 0.78, 1.00);
        vec3 c2 = vec3(1.00, 0.18, 0.72);
        vec3 c3 = vec3(1.00, 0.96, 1.00);
        if (lum < 0.35) return mix(c0, c1, lum / 0.35);
        if (lum < 0.75) return mix(c1, c2, (lum - 0.35) / 0.40);
        return mix(c2, c3, (lum - 0.75) / 0.25);
    } else if (mode == 2) {
        // CRT Phosphor Green (P31)
        vec3 c0 = vec3(0.00, 0.06, 0.02);
        vec3 c1 = vec3(0.08, 0.85, 0.32);
        vec3 c2 = vec3(0.72, 1.00, 0.82);
        return lum < 0.65 ? mix(c0, c1, lum / 0.65) : mix(c1, c2, (lum - 0.65) / 0.35);
    } else if (mode == 3) {
        // Vintage Amber Terminal (P3)
        vec3 c0 = vec3(0.08, 0.02, 0.00);
        vec3 c1 = vec3(1.00, 0.56, 0.06);
        vec3 c2 = vec3(1.00, 0.94, 0.70);
        return lum < 0.68 ? mix(c0, c1, lum / 0.68) : mix(c1, c2, (lum - 0.68) / 0.32);
    } else if (mode == 4) {
        // Solar Flare HDR
        vec3 c0 = vec3(0.06, 0.01, 0.03);
        vec3 c1 = vec3(0.88, 0.10, 0.12);
        vec3 c2 = vec3(1.00, 0.58, 0.08);
        vec3 c3 = vec3(1.00, 0.98, 0.82);
        if (lum < 0.35) return mix(c0, c1, lum / 0.35);
        if (lum < 0.72) return mix(c1, c2, (lum - 0.35) / 0.37);
        return mix(c2, c3, (lum - 0.72) / 0.28);
    } else if (mode == 5) {
        // Thermal Vision
        vec3 c0 = vec3(0.03, 0.03, 0.25);
        vec3 c1 = vec3(0.65, 0.06, 0.72);
        vec3 c2 = vec3(1.00, 0.28, 0.05);
        vec3 c3 = vec3(1.00, 0.96, 0.38);
        if (lum < 0.33) return mix(c0, c1, lum / 0.33);
        if (lum < 0.66) return mix(c1, c2, (lum - 0.33) / 0.33);
        return mix(c2, c3, (lum - 0.66) / 0.34);
    } else if (mode == 6) {
        // Synthwave Sunset
        vec3 c0 = vec3(0.08, 0.02, 0.22);
        vec3 c1 = vec3(0.95, 0.18, 0.55);
        vec3 c2 = vec3(1.00, 0.62, 0.24);
        vec3 c3 = vec3(0.30, 0.96, 1.00);
        if (lum < 0.40) return mix(c0, c1, lum / 0.40);
        if (lum < 0.75) return mix(c1, c2, (lum - 0.40) / 0.35);
        return mix(c2, c3, (lum - 0.75) / 0.25);
    } else if (mode == 7) {
        // Custom Duotone
        return mix(u_color_dark, u_color_bright, lum);
    } else if (mode == 8) {
        // SPECIMEN VIOLET (matches SPECIMEN N001 purple/lavender neon glow)
        vec3 c0 = vec3(0.06, 0.03, 0.14);
        vec3 c1 = vec3(0.55, 0.38, 0.96);
        vec3 c2 = vec3(0.86, 0.76, 1.00);
        vec3 c3 = vec3(1.00, 0.98, 1.00);
        if (lum < 0.45) return mix(c0, c1, lum / 0.45);
        if (lum < 0.80) return mix(c1, c2, (lum - 0.45) / 0.35);
        return mix(c2, c3, (lum - 0.80) / 0.20);
    } else if (mode == 9) {
        // SUBJECT ICE-CYAN (matches SUBJECT A-34 phosphor ice-blue CRT)
        vec3 c0 = vec3(0.02, 0.05, 0.10);
        vec3 c1 = vec3(0.28, 0.66, 0.98);
        vec3 c2 = vec3(0.74, 0.93, 1.00);
        vec3 c3 = vec3(0.96, 0.99, 1.00);
        if (lum < 0.45) return mix(c0, c1, lum / 0.45);
        if (lum < 0.80) return mix(c1, c2, (lum - 0.45) / 0.35);
        return mix(c2, c3, (lum - 0.80) / 0.20);
    } else {
        // MONOCHROME NOIR INK (matches Creation Hands & Kinetic Polyglot Sphere)
        return vec3(mix(0.32, 1.0, lum));
    }
}

void computeInteraction(vec2 uv, vec2 aspect, out vec2 uvOffset, out float interactEnergy, out vec3 interactTint) {
    uvOffset = vec2(0.0);
    interactEnergy = 0.0;
    interactTint = vec3(0.0);

    for (int i = 0; i < 16; i++) {
        if (i >= u_ripple_count) break;
        vec4 rp = u_ripples[i];
        vec2 delta = (uv - rp.xy) * aspect;
        float dist = length(delta);
        float waveRadius = rp.z * 0.72;
        float band = exp(-pow((dist - waveRadius) * 14.0, 2.0));
        float fade = exp(-rp.z * 1.45) * rp.w;
        float wave = sin((dist - waveRadius) * 38.0) * band * fade;

        vec2 dir = dist > 0.0001 ? delta / dist : vec2(0.0);
        uvOffset -= (dir / aspect) * wave * 0.045 * u_interact_strength;
        interactEnergy += band * fade * 1.35 * u_interact_strength;
        interactTint += vec3(0.35, 0.78, 1.0) * band * fade * 0.55;
    }

    if (u_mouse_active > 0.5 && u_mouse_mode > 0) {
        vec2 delta = (uv - u_mouse) * aspect;
        float dist = length(delta);
        float r = max(u_interact_radius, 0.04);
        float falloff = smoothstep(r, 0.0, dist);
        vec2 dir = dist > 0.0001 ? delta / dist : vec2(0.0);

        if (u_mouse_mode == 2) {
            interactEnergy += falloff * falloff * 1.6 * u_interact_strength;
            interactTint += vec3(0.55, 0.72, 1.0) * falloff * 0.45;
        } else if (u_mouse_mode == 3) {
            uvOffset -= (dir / aspect) * (falloff * falloff * 0.08 * u_interact_strength);
            interactEnergy += falloff * 0.9 * u_interact_strength;
            interactTint += vec3(1.0, 0.35, 0.85) * falloff * 0.5;
        } else if (u_mouse_mode == 4) {
            vec2 swirl = vec2(-dir.y, dir.x) * 0.45;
            uvOffset += ((dir + swirl) / aspect) * (falloff * falloff * 0.075 * u_interact_strength);
            interactEnergy += falloff * 1.1 * u_interact_strength;
            interactTint += vec3(0.35, 0.95, 0.85) * falloff * 0.45;
        }
    }
}

void main() {
    vec2 aspect = vec2(u_resolution.x / max(u_resolution.y, 1.0), 1.0);
    float audioKick = u_audio.x * u_audio_reactivity;
    float audioBass = u_audio.y * u_audio_reactivity;
    float audioVocal = u_audio.z * u_audio_reactivity;
    float audioSnare = u_audio_stems.x * u_audio_reactivity;
    float audioHihat = u_audio_stems.y * u_audio_reactivity;
    float audioLow = max(audioKick, audioBass * 0.8);
    float audioMid = max(audioVocal, audioSnare * 0.85);
    float audioHigh = audioHihat;

    // Sharp horizontal CRT Snare/Drop glitch slice offset
    float glitchShiftX = 0.0;
    if (u_glitch_tear > 0.01) {
        float bandY = floor(v_uv.y * 26.0);
        float sliceTick = floor(u_time * 24.0);
        float sliceRnd = hash21(vec2(bandY, sliceTick));
        if (sliceRnd > (0.86 - min(0.45, u_glitch_tear * 0.35))) {
            glitchShiftX = (hash21(vec2(bandY + 7.1, sliceTick)) - 0.5) * 0.095 * u_glitch_tear;
        }
    }

    // ==================== MODE 1: STIPPLE / POINT-CLOUD CRT DITHER ====================
    if (u_render_mode == 1) {
        float pScale = max(u_stipple_scale, 1.25);
        // Center grid symmetrically around (0.5, 0.5) so size changes & beat-breathe never jump/lock to bottom-left!
        vec2 centeredPx = (v_uv - 0.5) * u_resolution;
        vec2 pixGrid = floor(centeredPx / pScale);
        vec2 pixLocal = fract(centeredPx / pScale) - 0.5;
        vec2 pixUV = ((pixGrid + 0.5) * pScale) / u_resolution + 0.5;

        vec2 uvOff;
        float iEnergy;
        vec3 iTint;
        computeInteraction(pixUV, aspect, uvOff, iEnergy, iTint);
        vec2 dispUV = computeDisplacementUV(pixUV, aspect);
        vec2 sampleUV = clamp(dispUV + uvOff + vec2(glitchShiftX, 0.0), vec2(0.001), vec2(0.999));
        float spatialEQ = sampleSpatialEQ(pixUV, aspect);

        vec3 srcCol = texture(u_source, sampleUV).rgb;
        vec4 edgeVal = texture(u_edges, sampleUV);
        float rawLum = dot(srcCol, vec3(0.2126, 0.7152, 0.0722));
        float lum = (rawLum - 0.5) * u_contrast + 0.5 + u_brightness + iEnergy * 0.30 + audioLow * 0.08 + spatialEQ * 0.16;
        lum = clamp(lum, 0.0, 1.0);
        lum = pow(max(lum, 0.0001), 1.0 / max(u_gamma, 0.1));
        if (u_invert == 1) lum = 1.0 - lum;

        // Clean black cutout
        if (lum < u_black_cutoff) {
            fragColor = vec4(0.0, 0.0, 0.0, 1.0);
            return;
        }

        // Edge highlight boost inside stipple cloud (makes eyes, faces & contours pop!)
        float eMag = length(edgeVal.xy) * u_edge_strength * 0.65;
        float prob = clamp((lum + eMag * 0.35) * u_stipple_density, 0.0, 1.0);

        // Combine blue-noise-like IGN + Bayer ordered pattern for crisp organic stipple dots
        float n = mix(ignNoise(pixGrid), bayer4x4(pixGrid) + 0.5, clamp(u_dither * 0.45, 0.0, 0.6));
        if (u_scramble_speed > 0.1 && u_scramble_amount > 0.01) {
            float tick = floor(u_time * u_scramble_speed);
            float animN = hash21(pixGrid + vec2(tick * 7.13, tick * 3.71));
            n = mix(n, animN, u_scramble_amount * 0.35);
        }

        float dotOn = step(n, prob);
        float dotShape = smoothstep(0.58, 0.18, length(pixLocal));
        float mask = dotOn * mix(1.0, dotShape, clamp(u_glyph_sharpness, 0.0, 1.0));

        vec3 baseCol = rotateHue(mapPalette(srcCol, lum, u_color_mode), u_hue_shift);
        baseCol = mix(baseCol, baseCol + iTint, 0.6);

        float hdrScale = 0.35 + pow(clamp(lum, 0.0, 1.5), max(u_hdr_power, 0.2)) * u_hdr_boost;
        hdrScale += eMag * u_edge_boost * 0.5 + iEnergy * 1.2 + spatialEQ * 0.85;
        hdrScale *= (1.0 + audioLow * 0.45 + audioSnare * 0.25);

        vec3 bgCol = baseCol * (u_bg_opacity * 0.25);
        vec3 outHDR = mix(bgCol, baseCol * hdrScale, mask);
        fragColor = vec4(max(outHDR, vec3(0.0)), 1.0);
        return;
    }

    // ==================== MODE 0 (ASCII) & MODE 2 (HYBRID STIPPLE + ASCII EDGES) ====================
    vec2 cellSize = max(u_cell_size, vec2(4.0));
    vec2 gridCount = u_resolution / cellSize;

    // Symmetrically centered cell grid -> smooth center-anchored scaling & beat breathing with ZERO edge lock!
    vec2 centeredGrid = (v_uv - 0.5) * gridCount;
    vec2 cellIdx = floor(centeredGrid);
    vec2 cellLocal = fract(centeredGrid);
    vec2 cellCenterUV = (cellIdx + 0.5) / gridCount + 0.5;

    vec2 uvOff;
    float interactEnergy;
    vec3 interactTint;
    computeInteraction(cellCenterUV, aspect, uvOff, interactEnergy, interactTint);
    vec2 dispUV = computeDisplacementUV(cellCenterUV, aspect);
    vec2 sampleUV = clamp(dispUV + uvOff + vec2(glitchShiftX, 0.0), vec2(0.001), vec2(0.999));
    float spatialEQ = sampleSpatialEQ(cellCenterUV, aspect);

    // 2x2 Sub-cell Supersampling for crisp color + structure tensor edges
    vec2 subStep = (0.28 / gridCount);
    vec3 srcCol = (
        texture(u_source, clamp(sampleUV + vec2(-subStep.x, -subStep.y), 0.0, 1.0)).rgb +
        texture(u_source, clamp(sampleUV + vec2( subStep.x, -subStep.y), 0.0, 1.0)).rgb +
        texture(u_source, clamp(sampleUV + vec2(-subStep.x,  subStep.y), 0.0, 1.0)).rgb +
        texture(u_source, clamp(sampleUV + vec2( subStep.x,  subStep.y), 0.0, 1.0)).rgb
    ) * 0.25;

    vec4 edgeAvg = (
        texture(u_edges, clamp(sampleUV + vec2(-subStep.x, -subStep.y), 0.0, 1.0)) +
        texture(u_edges, clamp(sampleUV + vec2( subStep.x, -subStep.y), 0.0, 1.0)) +
        texture(u_edges, clamp(sampleUV + vec2(-subStep.x,  subStep.y), 0.0, 1.0)) +
        texture(u_edges, clamp(sampleUV + vec2( subStep.x,  subStep.y), 0.0, 1.0))
    ) * 0.25;

    float rawLum = dot(srcCol, vec3(0.2126, 0.7152, 0.0722));
    float lum = (rawLum - 0.5) * u_contrast + 0.5 + u_brightness + interactEnergy * 0.28 + audioLow * 0.08;
    lum = clamp(lum, 0.0, 1.0);
    lum = pow(max(lum, 0.0001), 1.0 / max(u_gamma, 0.1));
    if (u_invert == 1) {
        lum = 1.0 - lum;
    }

    // Black cutoff: cleanly knocks dark backgrounds to void #000000 (Creation Hands & Kinetic Sphere!)
    if (lum < u_black_cutoff && interactEnergy < 0.12) {
        fragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    // Remap luminance smoothly above cutoff so full ramp range is preserved
    float activeLum = clamp((lum - u_black_cutoff) / max(1.0 - u_black_cutoff, 0.01), 0.0, 1.0);

    float ditheredLum = clamp(activeLum + bayer4x4(cellIdx) * (u_dither / max(float(u_num_ramp), 4.0)), 0.0, 1.0);

    // Optional Matrix Rain energy pulse per column
    float rainBoost = 0.0;
    if (u_matrix_rain > 0.001) {
        float colSeed = hash21(vec2(cellIdx.x, 17.3));
        float speed = 0.35 + colSeed * 0.95;
        float dropPos = fract(colSeed * 13.7 - u_time * speed * 0.35);
        float distDrop = fract(cellCenterUV.y - dropPos);
        float trail = pow(1.0 - distDrop, 6.0);
        float head = smoothstep(0.05, 0.0, distDrop);
        rainBoost = (trail * 0.7 + head * 2.2) * u_matrix_rain;
    }

    // Edge direction detection
    float edgeMag = length(edgeAvg.xy) * (1.0 + audioMid * 0.6);
    bool isEdge = (u_edge_strength > 0.01) && (edgeMag > u_edge_threshold);

    // Base luminance-mapped glyph index
    int baseIdx = int(floor(ditheredLum * float(u_num_ramp - 1) + 0.5));
    baseIdx = clamp(baseIdx, 1, max(u_num_ramp - 1, 1));

    // Kinetic Glyph Scramble (mutates characters rapidly across Polyglot/ASCII atlas like the uploaded MP4!)
    int glyphIdx = baseIdx;
    if (u_scramble_amount > 0.01 && u_num_ramp > 3) {
        float rate = max(u_scramble_speed, 0.5) * (1.0 + audioHigh * 1.4);
        float cellOffset = hash21(cellIdx) * 3.17;
        float stepTick = floor(u_time * rate + cellOffset);
        float rnd = hash21(cellIdx * 1.37 + vec2(stepTick * 11.73, stepTick * 5.29));
        int randIdx = 1 + int(floor(rnd * float(u_num_ramp - 1)));
        // Blend between pure luminance ramp and full kinetic random glyphs
        if (u_scramble_amount >= 0.98) {
            glyphIdx = clamp(randIdx, 1, u_num_ramp - 1);
        } else {
            float jitter = (rnd - 0.5) * 2.0 * u_scramble_amount * float(u_num_ramp - 1);
            glyphIdx = clamp(baseIdx + int(round(jitter)), 1, u_num_ramp - 1);
        }
    }

    float edgeFactor = 0.0;
    if (isEdge) {
        float theta = 0.5 * atan(edgeAvg.y, edgeAvg.x);
        float absTheta = abs(theta);
        int edgeSlot = 0;
        if (absTheta < 0.3927) {
            edgeSlot = 0; // '|'
        } else if (absTheta > 1.1781) {
            edgeSlot = 1; // '-'
        } else if (theta > 0.0) {
            edgeSlot = 3; // '\'
        } else {
            edgeSlot = 2; // '/'
        }
        float eWeight = smoothstep(u_edge_threshold, u_edge_threshold + 0.18, edgeMag) * u_edge_strength;
        if (eWeight > 0.38) {
            glyphIdx = u_num_ramp + edgeSlot;
            edgeFactor = eWeight;
        }
    }

    // Hybrid Mode 2: Render stochastic stipple dots in flat areas, directional ASCII glyphs on strong edges!
    float glyphMask;
    if (u_render_mode == 2 && edgeFactor < 0.38) {
        float pScale = max(u_stipple_scale, 1.25);
        vec2 centeredPx = (v_uv - 0.5) * u_resolution;
        vec2 pixGrid = floor(centeredPx / pScale);
        vec2 pixLocal = fract(centeredPx / pScale) - 0.5;
        float n = ignNoise(pixGrid);
        float dotOn = step(n, clamp(activeLum * u_stipple_density, 0.0, 1.0));
        glyphMask = dotOn * smoothstep(0.55, 0.15, length(pixLocal));
    } else {
        float glyphU = (float(glyphIdx) + clamp(cellLocal.x, 0.02, 0.98)) / float(max(u_total_glyphs, 1));
        float glyphV = 1.0 - clamp(cellLocal.y, 0.02, 0.98);
        float rawMask = texture(u_atlas, vec2(glyphU, glyphV)).r;

        // Variable stroke weight contrast (brighter regions render bolder, dimmer regions thinner!)
        float weightShift = (activeLum - 0.45) * u_weight_contrast * 0.34;
        float threshold = clamp(0.50 - weightShift, 0.18, 0.78);
        float sharp = mix(0.42, 0.07, clamp(u_glyph_sharpness, 0.0, 1.0));
        glyphMask = smoothstep(threshold - sharp, threshold + sharp, rawMask);
    }

    vec3 baseCol = rotateHue(mapPalette(srcCol, activeLum, u_color_mode), u_hue_shift);
    baseCol = mix(baseCol, baseCol + interactTint, 0.65);

    // Weight-driven luminance boost (recreates crisp bright-white vs medium-gray kinetic letters!)
    float weightBright = mix(1.0, 0.32 + 1.15 * smoothstep(0.28, 0.88, activeLum), clamp(u_weight_contrast, 0.0, 1.0));
    float hdrScale = (0.22 + pow(clamp(activeLum + rainBoost * 0.3 + spatialEQ * 0.22, 0.0, 1.5), max(u_hdr_power, 0.2)) * u_hdr_boost) * weightBright;
    hdrScale += edgeFactor * u_edge_boost;
    hdrScale += interactEnergy * 1.15 + spatialEQ * 0.90;
    hdrScale += rainBoost * 1.4;
    hdrScale *= (1.0 + audioLow * 0.42 + audioSnare * 0.22);

    vec3 glyphColor = baseCol * hdrScale;
    vec3 bgColor = baseCol * (u_bg_opacity * 0.35);

    vec3 outHDR = mix(bgColor, glyphColor, glyphMask);
    fragColor = vec4(max(outHDR, vec3(0.0)), 1.0);
}
