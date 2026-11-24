#version 330 core

in vec2 v_uv;
out vec4 fragColor;

uniform float u_time;
uniform vec2 u_resolution;
uniform int u_scene;       // 0: Kinetic Sphere, 1: Wireframe Heart, 2: Audio Mandala EQ, 3: Singularity 3D, 4: Plasma, 5: Tunnel, 6: Aurora
uniform vec2 u_mouse;      // normalized [0, 1]
uniform vec4 u_audio;      // (kick, bass, vocals, energy)
uniform vec4 u_audio_stems;// (snare, hihat, beat_phase, drop_intensity)
uniform sampler2D u_fft_tex; // 64x1 live perceptual Mel FFT texture

mat2 rot2D(float a) {
    float c = cos(a), s = sin(a);
    return mat2(c, -s, s, c);
}

vec3 palette(float t, vec3 a, vec3 b, vec3 c, vec3 d) {
    return a + b * cos(6.28318 * (c * t + d));
}

float sampleFFT(float u) {
    return texture(u_fft_tex, vec2(clamp(u, 0.01, 0.99), 0.5)).r;
}

// --- Scene 0: Kinetic Glyph Sphere (matches uploaded MP4 rotating shaded orb on pure void black!) ---
vec3 sceneKineticSphere(vec2 uv, float t) {
    float kick = u_audio.x;
    float bass = u_audio.y;
    float vocal = u_audio.z;
    float snare = u_audio_stems.x;
    float r = 0.70 + kick * 0.10 + bass * 0.05;
    float d2 = dot(uv, uv);
    if (d2 > r * r) {
        return vec3(0.0);
    }
    float z = sqrt(max(r * r - d2, 0.0));
    vec3 n = normalize(vec3(uv, z));

    // Sample live FFT ring along sphere latitude
    float latU = clamp(n.y * 0.5 + 0.5, 0.02, 0.98);
    float eqVal = sampleFFT(latU);

    vec3 rn = n;
    rn.xz *= rot2D(t * 0.65 + (u_mouse.x - 0.5) * 3.14 + bass * 0.35);
    rn.yz *= rot2D(t * 0.38 + (u_mouse.y - 0.5) * 2.2);

    vec3 lightDir = normalize(vec3(sin(t * 0.8), cos(t * 0.6) * 0.6, 0.85));
    float diff = max(dot(n, lightDir), 0.0);

    float band1 = 0.5 + 0.5 * sin(rn.x * 5.2 + rn.y * 3.4 + t * 1.4);
    float band2 = 0.5 + 0.5 * cos(rn.z * 6.0 - rn.x * 4.1 - t * 1.1);
    float cluster = smoothstep(0.30, 0.85, band1 * 0.55 + band2 * 0.35 + diff * 0.30 + eqVal * 0.45 + vocal * 0.25);

    float rim = pow(1.0 - max(n.z, 0.0), 2.2) * (0.35 + snare * 0.55);
    float intensity = clamp(0.22 + cluster * 0.85 + rim + kick * 0.32, 0.14, 1.45);

    vec3 col = mix(vec3(0.55, 0.62, 0.78), vec3(1.0, 0.98, 1.0), cluster) * intensity;
    return col;
}

// --- Scene 1: 3D Wireframe Heart (Specimen 001 signature sculpture beating to Kick + Vocals!) ---
float sdHeart3D(vec3 p) {
    vec3 h = p * 1.05;
    h.z *= 1.25;
    float a = h.x * h.x + 1.85 * h.z * h.z + h.y * h.y - 1.0;
    float f = a * a * a - h.x * h.x * h.y * h.y * h.y - 0.1125 * h.z * h.z * h.y * h.y * h.y;
    return f * 0.16;
}

vec3 sceneWireframeHeart(vec2 uv, float t) {
    float kick = u_audio.x;
    float vocal = u_audio.z;
    float snare = u_audio_stems.x;
    float scale = 1.10 + kick * 0.22 + u_audio.y * 0.08;
    vec3 ro = vec3(0.0, 0.04, 3.0);
    vec3 rd = normalize(vec3(uv, -1.75));
    float yaw = sin(t * 0.55) * 0.58 + (u_mouse.x - 0.5) * 2.2;
    float pitch = 0.18 + cos(t * 0.42) * 0.14 + (u_mouse.y - 0.5) * 1.2;

    float dist = 0.0;
    bool hit = false;
    vec3 p = ro;
    float minD = 99.0;

    for (int i = 0; i < 64; i++) {
        p = ro + rd * dist;
        vec3 rp = p / scale;
        rp.xz *= rot2D(yaw);
        rp.yz *= rot2D(pitch);
        float d = sdHeart3D(rp);
        minD = min(minD, abs(d));
        if (d < 0.0025) {
            hit = true;
            break;
        }
        if (dist > 6.5) break;
        dist += clamp(abs(d) * 0.55, 0.015, 0.25);
    }

    vec3 col = vec3(0.0);
    col += vec3(0.45, 0.30, 0.95) * (0.018 / (minD + 0.03)) * (0.45 + kick * 0.65 + vocal * 0.35);

    if (hit) {
        vec3 rp = p / scale;
        rp.xz *= rot2D(yaw);
        rp.yz *= rot2D(pitch);

        vec2 e = vec2(0.004, 0.0);
        vec3 n = normalize(vec3(
            sdHeart3D(rp + e.xyy) - sdHeart3D(rp - e.xyy),
            sdHeart3D(rp + e.yxy) - sdHeart3D(rp - e.yxy),
            sdHeart3D(rp + e.yyx) - sdHeart3D(rp - e.yyx)
        ));

        float eqLine = sampleFFT(clamp(rp.y * 0.5 + 0.5, 0.02, 0.98));
        float lat = abs(fract(rp.y * 8.5 + 0.5) - 0.5);
        float lon = abs(fract(atan(rp.z, rp.x) * 4.5 / 3.14159 + 0.5) - 0.5);
        float gridX = smoothstep(0.085 + eqLine * 0.04, 0.015, lat);
        float gridY = smoothstep(0.085 + snare * 0.04, 0.015, lon);
        float wire = max(gridX, gridY);

        float fresnel = pow(1.0 - max(dot(n, normalize(vec3(0.0, 0.0, 1.0))), 0.0), 2.0);
        float shade = 0.25 + 0.75 * max(dot(n, normalize(vec3(0.6, 0.8, 1.0))), 0.0);

        vec3 baseViolet = mix(vec3(0.50, 0.36, 1.0), vec3(0.95, 0.92, 1.0), wire);
        col = baseViolet * (wire * (1.2 + eqLine * 0.5) + fresnel * (0.75 + vocal * 0.5) + shade * 0.28) * (1.0 + kick * 0.55);
    }
    return clamp(col, 0.0, 1.6);
}

// --- Scene 2: Audio Mandala EQ (100% driven by live 64-band Mel FFT + Kick/Snare/HiHat/Beat Phase!) ---
vec3 sceneAudioMandala(vec2 uv, float t) {
    float kick = u_audio.x;
    float bass = u_audio.y;
    float vocal = u_audio.z;
    float snare = u_audio_stems.x;
    float hihat = u_audio_stems.y;
    float beatPhase = u_audio_stems.z;
    float drop = u_audio_stems.w;

    vec2 p = uv - (u_mouse - 0.5) * 0.35;
    float r = length(p);
    float ang = atan(p.y, p.x);

    // Mirror angle across 6-fold kaleidoscope symmetry for musical geometry!
    float symAng = abs(mod(ang + 3.14159 + t * 0.25, 1.0472) - 0.5236) / 0.5236;
    float eqAng = sampleFFT(clamp(symAng * 0.85 + 0.05, 0.02, 0.98));
    float eqRad = sampleFFT(clamp(r * 0.72, 0.02, 0.98));

    // 1. Live 64-band circular EQ crown
    float baseRadius = 0.34 + kick * 0.12 + bass * 0.06;
    float eqRadius = baseRadius + eqAng * 0.42;
    float ringDist = abs(r - eqRadius);
    float crownGlow = 0.016 / (ringDist + 0.012);

    // 2. Concentric beat shockwave ring expanding synced to beat_phase (0 -> 1)!
    float beatR = fract(beatPhase) * 1.15 + 0.12;
    float beatRing = exp(-pow((r - beatR) * 18.0, 2.0)) * (0.45 + kick * 1.1 + drop * 0.8);

    // 3. Inner pulsating kick core + harmonic vocal lotus petals
    float petals = 0.5 + 0.5 * cos(ang * 6.0 - t * 1.6 + bass * 2.0);
    float coreR = 0.14 + kick * 0.09 + petals * (0.04 + vocal * 0.07);
    float core = smoothstep(coreR + 0.02, coreR - 0.03, r) + (0.022 / (abs(r - coreR) + 0.018));

    // 4. Snare & Hi-Hat outer radial rays
    float rays = pow(abs(cos(ang * 12.0 + t * (1.2 + snare * 2.0))), 16.0);
    float rayMask = smoothstep(0.25, 0.95, r) * (0.18 + snare * 1.15 + hihat * 0.85);

    // 5. Outer FFT frequency echo rings
    float eqWaves = smoothstep(0.55, 0.98, sin(r * 24.0 - t * 3.5 - kick * 3.0)) * eqRad * 0.75;

    vec3 c1 = palette(
        r * 0.55 + t * 0.12 + eqAng * 0.35,
        vec3(0.45, 0.38, 0.62),
        vec3(0.55, 0.52, 0.45),
        vec3(1.0, 0.85, 0.72),
        vec3(0.05, 0.32, 0.68)
    );
    vec3 c2 = palette(
        symAng * 0.7 - t * 0.15,
        vec3(0.50, 0.45, 0.55),
        vec3(0.50, 0.50, 0.45),
        vec3(1.0, 0.90, 0.80),
        vec3(0.20, 0.48, 0.78)
    );

    vec3 col = vec3(0.004, 0.005, 0.012);
    col += c1 * crownGlow * (0.42 + eqAng * 0.85);
    col += c2 * core * (0.55 + kick * 0.75);
    col += vec3(0.35, 0.88, 1.0) * beatRing * 0.95;
    col += vec3(1.0, 0.42, 0.88) * rays * rayMask;
    col += c1 * eqWaves * (0.45 + drop * 0.65);

    return clamp(col, 0.0, 1.8);
}

// --- Scene 3: 3D Raymarched Morphing Crystal & Orbital Rings ---
float sdTorus(vec3 p, vec2 t) {
    vec2 q = vec2(length(p.xz) - t.x, p.y);
    return length(q) - t.y;
}

float sdOctahedron(vec3 p, float s) {
    p = abs(p);
    return (p.x + p.y + p.z - s) * 0.57735027;
}

float mapSingularity(vec3 p, float t, float kick) {
    vec3 q = p;
    q.xz *= rot2D(t * 0.55 + (u_mouse.x - 0.5) * 3.0);
    q.xy *= rot2D(t * 0.35 + (u_mouse.y - 0.5) * 2.0);

    float core = mix(
        length(q) - (0.72 + kick * 0.22),
        sdOctahedron(q, 1.15 + kick * 0.24),
        0.5 + 0.5 * sin(t * 0.7)
    );
    core += 0.06 * sin(q.x * 7.0 + t * 2.0) * sin(q.y * 7.0 - t * 1.7) * sin(q.z * 7.0);

    vec3 r1 = p;
    r1.xy *= rot2D(t * 0.9 + u_audio_stems.x * 0.4);
    r1.yz *= rot2D(0.7 + sin(t * 0.4) * 0.3);
    float ring1 = sdTorus(r1, vec2(1.45 + kick * 0.16, 0.055));

    vec3 r2 = p;
    r2.xz *= rot2D(-t * 0.75);
    r2.xy *= rot2D(-0.85);
    float ring2 = sdTorus(r2, vec2(1.85 + u_audio.y * 0.18, 0.04));

    return min(core, min(ring1, ring2));
}

vec3 calcNormal(vec3 p, float t, float kick) {
    vec2 e = vec2(0.002, 0.0);
    return normalize(vec3(
        mapSingularity(p + e.xyy, t, kick) - mapSingularity(p - e.xyy, t, kick),
        mapSingularity(p + e.yxy, t, kick) - mapSingularity(p - e.yxy, t, kick),
        mapSingularity(p + e.yyx, t, kick) - mapSingularity(p - e.yyx, t, kick)
    ));
}

vec3 sceneSingularity(vec2 uv, float t) {
    float kick = u_audio.x;
    float hihat = u_audio_stems.y;
    vec3 ro = vec3(0.0, 0.0, 3.8 - kick * 0.32);
    vec3 rd = normalize(vec3(uv, -1.6));

    float dist = 0.0;
    float glow = 0.0;
    bool hit = false;
    vec3 p = ro;

    for (int i = 0; i < 64; i++) {
        p = ro + rd * dist;
        float d = mapSingularity(p, t, kick);
        glow += 0.014 / (0.045 + abs(d));
        if (d < 0.002) {
            hit = true;
            break;
        }
        if (dist > 10.0) break;
        dist += d * 0.85;
    }

    vec3 col = vec3(0.005, 0.006, 0.014);
    vec3 glowCol = palette(
        length(uv) * 0.4 + t * 0.12,
        vec3(0.5, 0.5, 0.5),
        vec3(0.5, 0.5, 0.5),
        vec3(1.0, 1.0, 1.0),
        vec3(0.0, 0.33, 0.67)
    );
    col += glowCol * glow * (0.055 + kick * 0.05);

    if (hit) {
        vec3 n = calcNormal(p, t, kick);
        vec3 l1 = normalize(vec3(1.8, 2.2, 2.5));
        vec3 l2 = normalize(vec3(-2.0, -1.5, 1.5));
        float diff1 = max(dot(n, l1), 0.0);
        float diff2 = max(dot(n, l2), 0.0);
        float fresnel = pow(1.0 - max(dot(n, -rd), 0.0), 3.0);
        vec3 h1 = normalize(l1 - rd);
        float spec = pow(max(dot(n, h1), 0.0), 32.0);

        vec3 baseCol = palette(
            dot(n, vec3(0.3, 0.6, 0.2)) + t * 0.15,
            vec3(0.45, 0.35, 0.55),
            vec3(0.55, 0.55, 0.45),
            vec3(1.0, 0.8, 0.6),
            vec3(0.15, 0.35, 0.65)
        );
        col = baseCol * (0.18 + diff1 * 1.15 + diff2 * 0.65);
        col += vec3(0.2, 0.9, 1.0) * fresnel * (1.2 + hihat * 1.1);
        col += vec3(1.0, 0.85, 0.95) * spec * 1.6;
    }

    return clamp(col, 0.0, 1.5);
}

// --- Scene 4: Cyber Fluid Plasma ---
vec3 scenePlasma(vec2 uv, float t) {
    float kick = u_audio.x;
    float bass = u_audio.y;
    vec2 p = uv * 2.2;
    p += (u_mouse - 0.5) * 1.2;
    for (int i = 1; i <= 5; i++) {
        float fi = float(i);
        p.x += (0.55 / fi) * sin(fi * 1.8 * p.y + t * (0.8 + 0.12 * fi) + kick * 1.3);
        p.y += (0.55 / fi) * cos(fi * 1.7 * p.x - t * (0.7 + 0.1 * fi) + bass * 0.7);
    }
    float v = 0.5 + 0.5 * sin(p.x * 2.2 + p.y * 2.0);
    float ridge = pow(1.0 - abs(sin(p.x * 2.5 - p.y * 1.8)), 5.0);

    vec3 col = palette(
        v * 0.8 + t * 0.08,
        vec3(0.15, 0.12, 0.25),
        vec3(0.45, 0.40, 0.50),
        vec3(1.0, 0.9, 0.7),
        vec3(0.05, 0.32, 0.65)
    );
    col = col * (0.35 + v * 0.85) + vec3(0.25, 0.95, 1.0) * ridge * (0.9 + kick * 1.1);
    return clamp(col, 0.0, 1.5);
}

// --- Scene 5: Hyperspace Neon Tunnel ---
vec3 sceneTunnel(vec2 uv, float t) {
    float kick = u_audio.x;
    float snare = u_audio_stems.x;
    vec2 p = uv - (u_mouse - 0.5) * 0.4;
    float r = max(length(p), 0.001);
    float a = atan(p.y, p.x);

    float z = 1.2 / r + t * (2.2 + kick * 1.5);
    float rings = smoothstep(0.86, 0.99, abs(sin(z * 2.2)));
    float spokes = smoothstep(0.90, 0.99, abs(sin(a * 8.0 + sin(z * 0.4) * 1.5)));
    float hex = smoothstep(0.75, 0.98, cos(a * 6.0 + t * 0.6) * sin(z * 1.5));

    vec3 c1 = palette(
        z * 0.08 - t * 0.1,
        vec3(0.5, 0.4, 0.6),
        vec3(0.5, 0.5, 0.4),
        vec3(1.0, 0.8, 0.7),
        vec3(0.0, 0.28, 0.62)
    );
    float depthFade = smoothstep(0.02, 0.45, r);
    vec3 col = c1 * (rings * (1.15 + snare * 0.6) + spokes * 0.75 + hex * 0.55) * depthFade;
    col += vec3(0.4, 0.95, 1.0) * (0.08 / (r + 0.06)) * (0.7 + kick * 1.0);
    return clamp(col, 0.0, 1.5);
}

// --- Scene 6: Aurora Harmonic Waves ---
vec3 sceneAurora(vec2 uv, float t) {
    float kick = u_audio.x;
    float vocal = u_audio.z;
    vec3 col = vec3(0.006, 0.008, 0.018);
    for (int i = 0; i < 6; i++) {
        float fi = float(i);
        float eqBin = sampleFFT((fi + 0.5) / 6.0);
        float freq = 2.0 + fi * 0.75;
        float amp = 0.20 + 0.05 * sin(t + fi) + eqBin * 0.18 + kick * 0.10;
        float y = sin(uv.x * freq + t * (1.1 + fi * 0.22) + (u_mouse.x - 0.5) * 3.0) * amp
                + cos(uv.x * (freq * 0.5) - t * 0.8 + fi * 1.4) * (0.14 + vocal * 0.10)
                + (fi - 2.5) * 0.13;
        float d = abs(uv.y - y);
        float beam = 0.018 / (d + 0.014);
        vec3 waveCol = palette(
            fi * 0.18 + t * 0.1 + uv.x * 0.25,
            vec3(0.4, 0.45, 0.55),
            vec3(0.5, 0.45, 0.45),
            vec3(1.0, 0.85, 0.75),
            vec3(0.1, 0.45, 0.72)
        );
        col += waveCol * beam * (0.24 + eqBin * 0.22);
    }
    return clamp(col, 0.0, 1.5);
}

void main() {
    vec2 aspect = vec2(u_resolution.x / max(u_resolution.y, 1.0), 1.0);
    vec2 uv = (v_uv * 2.0 - 1.0) * aspect;

    vec3 col;
    if (u_scene == 0) {
        col = sceneKineticSphere(uv, u_time);
    } else if (u_scene == 1) {
        col = sceneWireframeHeart(uv, u_time);
    } else if (u_scene == 2) {
        col = sceneAudioMandala(uv, u_time);
    } else if (u_scene == 3) {
        col = sceneSingularity(uv, u_time);
    } else if (u_scene == 4) {
        col = scenePlasma(uv, u_time);
    } else if (u_scene == 5) {
        col = sceneTunnel(uv, u_time);
    } else {
        col = sceneAurora(uv, u_time);
    }
    fragColor = vec4(col, 1.0);
}
