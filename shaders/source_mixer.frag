#version 330 core

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_deck_a;
uniform sampler2D u_deck_b;

uniform vec2 u_resolution;
uniform float u_time;

// A/B Mixer & Blend Mode
// 0=Crossfade, 1=Additive, 2=Screen, 3=Multiply, 4=Difference,
// 5=LumaKey B->A, 6=LumaKey A->B, 7=Lighten, 8=Exclusion, 9=Beat Strobe Cut
uniform float u_crossfade;        // 0.0 = Pure Deck A, 1.0 = Full Deck B / Mix
uniform int u_blend_mode;
uniform float u_luma_thresh;      // Luma key threshold (0.0 .. 0.85)
uniform float u_luma_soft;        // Luma key feather softness (0.02 .. 0.5)
uniform float u_beat_phase;       // PLL beat phase 0..1 for Beat Strobe Cut

// Deck A Pro Video Transforms & Framing
uniform float u_a_scale;          // 0.5 .. 2.5
uniform float u_a_rot;            // Radians (-pi .. pi)
uniform vec2 u_a_flip;            // (flip_x, flip_y) in {0, 1}
uniform int u_a_mirror;           // 0=Off, 1=Mirror H, 2=Mirror V, 3=Quad Mirror
uniform float u_a_aspect;         // Deck A natural aspect (w/h)
uniform int u_a_fit_mode;         // 0=Fill, 1=Contain, 2=Cover

// Deck A Pre-Grade (applied before Edge/ASCII/Bloom fusion)
uniform float u_pre_hue;          // -0.5 .. 0.5
uniform float u_pre_contrast;     // 0.4 .. 2.2
uniform float u_pre_sat;          // 0.0 .. 2.2

vec3 rotateHue(vec3 col, float turns) {
    if (abs(turns) < 0.001) return col;
    float angle = turns * 6.2831853;
    float c = cos(angle);
    float s = sin(angle);
    vec3 k = vec3(0.57735027);
    return max(vec3(0.0), col * c + cross(k, col) * s + k * dot(k, col) * (1.0 - c));
}

vec3 preGrade(vec3 col) {
    col = (col - 0.5) * u_pre_contrast + 0.5;
    float l = dot(col, vec3(0.2126, 0.7152, 0.0722));
    col = mix(vec3(l), col, u_pre_sat);
    col = rotateHue(clamp(col, 0.0, 1.5), u_pre_hue);
    return clamp(col, 0.0, 1.0);
}

vec2 transformDeckA(vec2 uv, out bool inBounds) {
    inBounds = true;
    vec2 p = uv - 0.5;

    // Aspect ratio framing (Contain / Cover / Fill)
    float outAspect = u_resolution.x / max(u_resolution.y, 1.0);
    float srcAspect = max(u_a_aspect, 0.1);
    if (u_a_fit_mode == 1) {
        // Contain (letterbox cleanly)
        if (outAspect > srcAspect) {
            p.x *= outAspect / srcAspect;
        } else {
            p.y *= srcAspect / outAspect;
        }
    } else if (u_a_fit_mode == 2) {
        // Cover (fill edge-to-edge without stretching)
        if (outAspect > srcAspect) {
            p.y *= srcAspect / outAspect;
        } else {
            p.x *= outAspect / srcAspect;
        }
    }

    // Scale & rotation around center
    float invScale = 1.0 / max(u_a_scale, 0.2);
    p *= invScale;
    if (abs(u_a_rot) > 0.0005) {
        vec2 aspectFix = vec2(outAspect, 1.0);
        vec2 pa = p * aspectFix;
        float cs = cos(-u_a_rot);
        float sn = sin(-u_a_rot);
        pa = mat2(cs, -sn, sn, cs) * pa;
        p = pa / aspectFix;
    }

    vec2 res = p + 0.5;
    if (u_a_flip.x > 0.5) res.x = 1.0 - res.x;
    if (u_a_flip.y > 0.5) res.y = 1.0 - res.y;

    // Pre-mirror symmetry modes (great for VJ visual architecture!)
    if (u_a_mirror == 1) {
        res.x = res.x < 0.5 ? res.x * 2.0 : (1.0 - res.x) * 2.0;
    } else if (u_a_mirror == 2) {
        res.y = res.y < 0.5 ? res.y * 2.0 : (1.0 - res.y) * 2.0;
    } else if (u_a_mirror == 3) {
        res.x = res.x < 0.5 ? res.x * 2.0 : (1.0 - res.x) * 2.0;
        res.y = res.y < 0.5 ? res.y * 2.0 : (1.0 - res.y) * 2.0;
    }

    if (res.x < 0.0 || res.x > 1.0 || res.y < 0.0 || res.y > 1.0) {
        inBounds = false;
    }
    return clamp(res, 0.0, 1.0);
}

void main() {
    bool inBoundsA;
    vec2 uvA = transformDeckA(v_uv, inBoundsA);
    vec3 colA = inBoundsA ? preGrade(texture(u_deck_a, uvA).rgb) : vec3(0.0);

    float mixAmt = clamp(u_crossfade, 0.0, 1.0);
    if (mixAmt <= 0.001) {
        fragColor = vec4(colA, 1.0);
        return;
    }

    vec3 colB = texture(u_deck_b, v_uv).rgb;
    float lumA = dot(colA, vec3(0.2126, 0.7152, 0.0722));
    float lumB = dot(colB, vec3(0.2126, 0.7152, 0.0722));
    float soft = max(u_luma_soft, 0.02);

    vec3 outCol = colA;
    if (u_blend_mode == 0) {
        // 0: Normal Crossfade
        outCol = mix(colA, colB, mixAmt);
    } else if (u_blend_mode == 1) {
        // 1: Additive Light Fusion
        outCol = clamp(colA + colB * mixAmt, 0.0, 1.0);
    } else if (u_blend_mode == 2) {
        // 2: Screen (Cinema Double Exposure)
        vec3 scr = 1.0 - (1.0 - colA) * (1.0 - colB);
        outCol = mix(colA, scr, mixAmt);
    } else if (u_blend_mode == 3) {
        // 3: Multiply (Art textured INSIDE bright video silhouettes!)
        vec3 mul = colA * mix(vec3(1.0), colB * 1.45, mixAmt);
        outCol = clamp(mul, 0.0, 1.0);
    } else if (u_blend_mode == 4) {
        // 4: Difference (Electric interference)
        vec3 diff = abs(colA - colB);
        outCol = mix(colA, diff, mixAmt);
    } else if (u_blend_mode == 5) {
        // 5: Luma Key B -> A (Strips black from Deck B so 3D Art floats cleanly over Video A!)
        float keyB = smoothstep(u_luma_thresh, u_luma_thresh + soft, lumB) * mixAmt;
        outCol = mix(colA, colB, keyB);
    } else if (u_blend_mode == 6) {
        // 6: Luma Key A -> B (Strips dark bg from Video A so subject floats over 3D Art B!)
        float keyA = smoothstep(u_luma_thresh, u_luma_thresh + soft, lumA);
        vec3 keyed = mix(colB, colA, keyA);
        outCol = mix(colA, keyed, mixAmt);
    } else if (u_blend_mode == 7) {
        // 7: Lighten / Max
        outCol = mix(colA, max(colA, colB), mixAmt);
    } else if (u_blend_mode == 8) {
        // 8: Exclusion
        vec3 excl = colA + colB - 2.0 * colA * colB;
        outCol = mix(colA, excl, mixAmt);
    } else {
        // 9: Beat Strobe Cut (Rhythmic A/B cut locked to musical beat phase!)
        float cut = step(0.5, fract(u_beat_phase * 2.0));
        outCol = mix(colA, mix(colA, colB, cut), mixAmt);
    }

    fragColor = vec4(clamp(outCol, 0.0, 1.0), 1.0);
}
