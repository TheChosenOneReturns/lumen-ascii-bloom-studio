from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any


RENDER_MODE_NAMES = [
    "ASCII + Bordes (DoG/Sobel)",
    "Stipple / Point-Cloud CRT",
    "Híbrido Stipple + Bordes ASCII",
]

COLOR_MODE_NAMES = [
    "Color Original (Video)",
    "Cyberpunk Neon",
    "CRT Fósforo Verde",
    "Terminal Ámbar P3",
    "Solar Flare HDR",
    "Visión Térmica",
    "Synthwave Sunset",
    "Duotono Personalizado",
    "Specimen Violet (Foto 2)",
    "Subject Ice-Cyan (Foto 3)",
    "Monochrome Noir Ink (Foto 1)",
]

TONEMAP_NAMES = [
    "ACES Filmic",
    "Soft Neon Filmic",
    "Reinhard",
]

PROCEDURAL_SCENE_NAMES = [
    "Kinetic Glyph Sphere (Video Ref)",
    "3D Wireframe Heart (Specimen)",
    "Audio Mandala EQ (64-Band Live)",
    "Singularity 3D (Raymarched)",
    "Cyber Fluid Plasma",
    "Hyperspace Tunnel",
    "Aurora Harmonic Waves",
]

DISPLACEMENT_MODE_NAMES = [
    "Off (Sin Desplazamiento)",
    "Liquid Curl Flow (Fluido Orgánico)",
    "Acoustic Shockwave (Ondas de Beat)",
    "Silk Wave Drift (Ondas de Seda)",
    "Slit-Scan Echo (Cinta Temporal)",
    "3D Depth Parallax (Profundidad 2.5D)",
    "Cyber Glitch Blocks (Datamosh)",
    "Vortex Singularity (Remolino)",
    "Prism RGB Split (Prisma Cromático)",
    "Kaleido Mirror Drift (Simetría VJ)",
    "Elastic Lens Bounce (Rebote Spring)",
    "Pixel Sort Cascade (Cascada Melt)",
]


@dataclass
class ShaderParams:
    # --- Render Engine Mode ---
    render_mode: int = 0  # 0=ASCII, 1=Stipple CRT, 2=Hybrid

    # --- ASCII Grid & Glyph Atlas ---
    cell_w: float = 8.0
    cell_h: float = 13.0
    lock_cell_aspect: bool = True
    charset_id: str = "ascii_detailed"
    custom_charset: str = " .:-=+*#%@"
    font_path: str = ""
    brightness: float = 0.0
    contrast: float = 1.18
    gamma: float = 1.0
    saturation: float = 1.25
    invert: bool = False
    dither: float = 0.25
    bg_opacity: float = 0.06
    glyph_sharpness: float = 0.68
    black_cutoff: float = 0.04
    matrix_rain: float = 0.0

    # --- Kinetic Glyph Scramble (MP4 Sphere) & Stipple Point-Cloud (Specimen/Subject) ---
    scramble_speed: float = 0.0
    scramble_amount: float = 0.0
    weight_contrast: float = 0.0
    stipple_scale: float = 2.2
    stipple_density: float = 1.15

    # --- Acerola Edge Detection (DoG + Sobel) ---
    edge_strength: float = 0.78
    edge_threshold: float = 0.20
    edge_boost: float = 0.85
    dog_sigma: float = 1.0

    # --- Colorization & Linear HDR Emission ---
    color_mode: int = 0
    color_dark: list[float] = field(default_factory=lambda: [0.05, 0.02, 0.14])
    color_bright: list[float] = field(default_factory=lambda: [0.68, 0.56, 1.00])
    hdr_boost: float = 2.4
    hdr_power: float = 1.45

    # --- Phosphor Persistence & Wind Dissolve Trails ---
    trail_decay: float = 0.35
    trail_drift: float = 0.0
    trail_drift_x: float = 0.0
    trail_drift_y: float = 0.0
    trail_dissolve: float = 0.0

    # --- Physically-Based Mip-Chain Bloom (CoD:AW) ---
    bloom_intensity: float = 0.95
    bloom_threshold: float = 0.55
    bloom_soft_knee: float = 0.56
    bloom_radius: float = 1.18
    bloom_scatter: float = 0.72
    bloom_anamorphic: float = 1.0
    bloom_tint: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])

    # --- Post-Processing & Tonemap ---
    exposure: float = 1.06
    tonemap_mode: int = 0
    chromatic_aberration: float = 0.003
    scanlines: float = 0.12
    curvature: float = 0.0
    vignette: float = 0.25
    grain: float = 0.06

    # --- GPU Displacement & Kinetic Motion Engine ---
    disp_mode: int = 0              # 0=Off .. 11=Pixel Sort Cascade
    disp_strength: float = 0.45     # Amplitude of spatial warp (0..2.0)
    disp_speed: float = 1.0         # Animation tempo multiplier (0.1..3.0)
    disp_scale: float = 1.25        # Spatial wavelength / frequency (0.3..3.5)
    disp_audio_sync: float = 0.85   # Audio spring-physics coupling (0..2.0)

    # --- Audio Reactivity ---
    audio_reactivity: float = 0.85

    def clone(self) -> ShaderParams:
        return copy.deepcopy(self)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ShaderParams:
        base = cls()
        valid_fields = set(asdict(base).keys())
        filtered = {k: copy.deepcopy(v) for k, v in data.items() if k in valid_fields}
        for k, v in filtered.items():
            setattr(base, k, v)
        return base


_FLOAT_MORPH_FIELDS = [
    "cell_w",
    "cell_h",
    "brightness",
    "contrast",
    "gamma",
    "saturation",
    "dither",
    "bg_opacity",
    "glyph_sharpness",
    "black_cutoff",
    "matrix_rain",
    "scramble_speed",
    "scramble_amount",
    "weight_contrast",
    "stipple_scale",
    "stipple_density",
    "edge_strength",
    "edge_threshold",
    "edge_boost",
    "dog_sigma",
    "hdr_boost",
    "hdr_power",
    "trail_decay",
    "trail_drift",
    "trail_drift_x",
    "trail_drift_y",
    "trail_dissolve",
    "bloom_intensity",
    "bloom_threshold",
    "bloom_soft_knee",
    "bloom_radius",
    "bloom_scatter",
    "bloom_anamorphic",
    "exposure",
    "chromatic_aberration",
    "scanlines",
    "curvature",
    "vignette",
    "grain",
    "disp_strength",
    "disp_speed",
    "disp_scale",
    "disp_audio_sync",
    "audio_reactivity",
]

_VEC3_MORPH_FIELDS = [
    "color_dark",
    "color_bright",
    "bloom_tint",
]


class PresetMorpher:
    """Smooth cubic ease-in-out morphing between two ShaderParams configurations."""

    def __init__(self) -> None:
        self.active: bool = False
        self.elapsed: float = 0.0
        self.duration: float = 0.75
        self._start: ShaderParams | None = None
        self._target: ShaderParams | None = None
        self.target_name: str = ""

    def start(self, current: ShaderParams, target: ShaderParams, name: str = "", duration: float = 0.75) -> bool:
        """Starts morphing toward target. Returns True if glyph atlas needs immediate rebuild."""
        self._start = current.clone()
        self._target = target.clone()
        self.elapsed = 0.0
        self.duration = max(duration, 0.01)
        self.active = True
        self.target_name = name

        atlas_changed = (
            current.charset_id != target.charset_id
            or current.custom_charset != target.custom_charset
            or current.font_path != target.font_path
        )
        current.render_mode = target.render_mode
        current.color_mode = target.color_mode
        current.charset_id = target.charset_id
        current.custom_charset = target.custom_charset
        current.font_path = target.font_path
        current.invert = target.invert
        current.tonemap_mode = target.tonemap_mode
        current.lock_cell_aspect = target.lock_cell_aspect
        current.disp_mode = target.disp_mode
        return atlas_changed

    def cancel(self) -> None:
        """Immediately stops active morphing so manual user slider edits never snap back or feel locked."""
        self.active = False

    def update(self, dt: float, current: ShaderParams) -> None:
        if not self.active or self._start is None or self._target is None:
            return

        self.elapsed += dt
        t = min(self.elapsed / self.duration, 1.0)
        s = t * t * (3.0 - 2.0 * t)

        for name in _FLOAT_MORPH_FIELDS:
            a = float(getattr(self._start, name))
            b = float(getattr(self._target, name))
            setattr(current, name, a + (b - a) * s)

        for name in _VEC3_MORPH_FIELDS:
            va = getattr(self._start, name)
            vb = getattr(self._target, name)
            setattr(current, name, [va[i] + (vb[i] - va[i]) * s for i in range(3)])

        if t >= 1.0:
            self.active = False
