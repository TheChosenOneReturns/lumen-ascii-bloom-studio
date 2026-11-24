#version 330 core

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_source;
uniform vec2 u_texel_size;   // 1.0 / source_resolution
uniform float u_dog_sigma;   // Controls DoG kernel spread (default ~1.0)

float getLuma(vec2 uv) {
    vec3 c = texture(u_source, clamp(uv, 0.0, 1.0)).rgb;
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

// Fast 3x3 Gaussian blur around uv at a given stride
float gauss3x3(vec2 uv, vec2 stepSize) {
    float c = getLuma(uv) * 4.0;
    float e = (
        getLuma(uv + vec2( stepSize.x, 0.0)) +
        getLuma(uv + vec2(-stepSize.x, 0.0)) +
        getLuma(uv + vec2(0.0,  stepSize.y)) +
        getLuma(uv + vec2(0.0, -stepSize.y))
    ) * 2.0;
    float k = (
        getLuma(uv + vec2( stepSize.x,  stepSize.y)) +
        getLuma(uv + vec2(-stepSize.x,  stepSize.y)) +
        getLuma(uv + vec2( stepSize.x, -stepSize.y)) +
        getLuma(uv + vec2(-stepSize.x, -stepSize.y))
    );
    return (c + e + k) / 16.0;
}

void main() {
    vec2 ts = u_texel_size * max(u_dog_sigma, 0.5);

    // Difference of Gaussians (DoG) for crisp, noise-resistant edge emphasis
    float g1 = gauss3x3(v_uv, ts * 0.85);
    float g2 = gauss3x3(v_uv, ts * 2.15);
    float dog = abs(g1 - 0.96 * g2) * 3.5;

    // 3x3 Sobel gradient on luminance
    float tl = getLuma(v_uv + vec2(-ts.x,  ts.y));
    float tc = getLuma(v_uv + vec2(  0.0,  ts.y));
    float tr = getLuma(v_uv + vec2( ts.x,  ts.y));
    float ml = getLuma(v_uv + vec2(-ts.x,   0.0));
    float mc = getLuma(v_uv);
    float mr = getLuma(v_uv + vec2( ts.x,   0.0));
    float bl = getLuma(v_uv + vec2(-ts.x, -ts.y));
    float bc = getLuma(v_uv + vec2(  0.0, -ts.y));
    float br = getLuma(v_uv + vec2( ts.x, -ts.y));

    float gx = (tr + 2.0 * mr + br) - (tl + 2.0 * ml + bl);
    float gy = (tl + 2.0 * tc + tr) - (bl + 2.0 * bc + br);

    float mag = length(vec2(gx, gy));
    // Combine Sobel magnitude with DoG crispness
    float combinedMag = mag * 0.72 + dog * 0.55;

    // Encode doubled-angle orientation vector (cos 2*theta, sin 2*theta) * mag
    // Allows clean spatial averaging inside ASCII cells without 0/pi wrap artifacts!
    float theta = atan(gy, gx);
    vec2 orientVec = vec2(cos(2.0 * theta), sin(2.0 * theta)) * combinedMag;

    fragColor = vec4(orientVec.x, orientVec.y, mc, combinedMag);
}
