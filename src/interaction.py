from __future__ import annotations

from dataclasses import dataclass, field


INTERACTION_MODES = [
    "Desactivado (Solo Visualizar)",
    "Ondas / Ripples (Clic + Arrastrar)",
    "Linterna / Aura Bloom",
    "Repulsión Magnética",
    "Vórtice / Atracción",
    "Pincel Emisivo HDR",
]

SPLIT_MODES = [
    "Video / Fuente Original",
    "Mapa de Bordes (DoG + Sobel)",
    "Solo Bloom HDR Aislado",
]


@dataclass
class Ripple:
    x: float
    y: float
    age: float = 0.0
    strength: float = 1.0


@dataclass
class InteractionState:
    mode: int = 1  # Ripple by default for instant tactile delight
    radius: float = 0.22
    strength: float = 1.0

    # Mouse UV in [0, 1] (bottom-left OpenGL origin)
    mouse_uv: list[float] = field(default_factory=lambda: [0.5, 0.5])
    prev_mouse_uv: list[float] = field(default_factory=lambda: [0.5, 0.5])
    mouse_active: bool = False
    mouse_down: bool = False

    # Ripples ring buffer (max 16 simultaneous waves)
    ripples: list[Ripple] = field(default_factory=list)
    _last_drag_ripple_time: float = 0.0
    beat_ripples: bool = False

    # Emissive Brush parameters
    brush_radius: float = 0.032
    brush_color: list[float] = field(default_factory=lambda: [0.20, 0.95, 1.00])
    paint_decay: float = 0.992  # 1.0 = permanent, <1.0 = smooth glowing fade
    clear_paint_requested: bool = False

    # Split-Screen Comparison Curtain
    split_enabled: bool = False
    split_pos: float = 0.5
    split_mode: int = 0
    dragging_split: bool = False

    # Viewport Zoom & Pan
    zoom: float = 1.0
    pan_x: float = 0.0
    pan_y: float = 0.0

    def spawn_ripple(self, x: float, y: float, strength: float = 1.0) -> None:
        self.ripples.append(Ripple(x=x, y=y, age=0.0, strength=strength))
        if len(self.ripples) > 16:
            self.ripples = self.ripples[-16:]

    def update(self, dt: float, now: float, beat_trigger: bool = False) -> None:
        # Advance and expire ripples
        alive: list[Ripple] = []
        for r in self.ripples:
            r.age += dt
            if r.age < 2.6:
                alive.append(r)
        self.ripples = alive

        # Continuous ripples while dragging in Ripple mode
        if self.mode == 1 and self.mouse_active and self.mouse_down and not self.dragging_split:
            if now - self._last_drag_ripple_time > 0.075:
                self.spawn_ripple(self.mouse_uv[0], self.mouse_uv[1], strength=0.85)
                self._last_drag_ripple_time = now

        # Optional audio-beat center ripple
        if self.beat_ripples and beat_trigger:
            self.spawn_ripple(0.5, 0.5, strength=1.1)

    def reset_view(self) -> None:
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0

    def packed_ripples(self) -> tuple[list[tuple[float, float, float, float]], int]:
        out: list[tuple[float, float, float, float]] = []
        for r in self.ripples[:16]:
            out.append((r.x, r.y, r.age, r.strength))
        count = len(out)
        while len(out) < 16:
            out.append((0.0, 0.0, 99.0, 0.0))
        return out, count
