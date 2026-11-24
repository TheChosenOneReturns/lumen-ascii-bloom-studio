from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    from PIL import ImageGrab

    _SCREEN_GRAB_AVAILABLE = True
except Exception:
    ImageGrab = None
    _SCREEN_GRAB_AVAILABLE = False


BLEND_MODE_NAMES: List[str] = [
    "Crossfade (Suave A <-> B)",
    "Additive (Suma de Luz)",
    "Screen (Doble Exposición)",
    "Multiply (Arte dentro de Video)",
    "Difference (Interferencia)",
    "Luma Key: Arte B sobre Video A",
    "Luma Key: Sujeto A sobre Arte B",
    "Lighten (Brillo Máximo)",
    "Exclusion (Cine Experimental)",
    "Beat Strobe Cut (Corte al Beat)",
]

MIRROR_MODE_NAMES: List[str] = [
    "Off",
    "Espejo Horizontal",
    "Espejo Vertical",
    "Cuádruple (4x Simetría)",
]

FIT_MODE_NAMES: List[str] = [
    "Llenar Directo (Fill)",
    "Ajustar (Contain)",
    "Cubrir Recortando (Cover)",
]

SCREEN_CROP_NAMES: List[str] = [
    "Pantalla Completa",
    "Mitad Izquierda",
    "Mitad Derecha",
    "Centro 16:9",
]

ART_OVERLAY_STYLES: List[str] = [
    "Off (Sin Capa Tipográfica)",
    "Specimen Editorial Stamp",
    "Kinetic Center Poster",
    "Cinema Ticker Ribbon",
    "Gallery Frame + Timecode",
]


@dataclass
class VideoMixerState:
    """Pro VJ Dual-Deck A/B Mixer + Video Transform + Art Typography Fusion parameters."""

    # Deck B source: 0..6 = Procedural 3D Scene index, -1 = Second loaded clip image/video!
    deck_b_mode: int = 0            # 0 = 3D Generative Scene, 1 = Imagen/Video Deck B
    deck_b_scene: int = 2           # Default Deck B: Audio Mandala EQ (fuses great with any video!)
    crossfade: float = 0.0          # 0.0 = Pure Deck A, 1.0 = Full Deck B blend
    blend_mode: int = 5             # Default: Luma Key B -> A (clean 3D art over video!)
    luma_thresh: float = 0.14
    luma_soft: float = 0.22
    crossfade_audio_mod: float = 0.0  # Kick/Drop pushes Deck B into Deck A dynamically!

    # Deck A Pro Video Transforms & Color Pre-Grade
    a_scale: float = 1.0
    a_rot_deg: float = 0.0
    a_flip_x: bool = False
    a_flip_y: bool = False
    a_mirror: int = 0
    a_fit_mode: int = 0
    pre_hue: float = 0.0
    pre_contrast: float = 1.0
    pre_sat: float = 1.0

    # Beat-Synced Video Performance Automation
    beat_speed_boost: float = 0.0   # Kicks temporarily surge video playback speed!
    auto_clip_on_drop: bool = False # Auto-triggers next clip in Clip Bank on musical Drops!
    auto_clip_bars: int = 0         # 0=Off, else switch clip every N bars (4, 8, 16)

    # Live Screen / Window Capture into Deck A
    live_screen_active: bool = False
    screen_crop_mode: int = 0

    # ART-VIDEO-MUSIC Editorial Typography Overlay
    art_overlay_style: int = 0      # 0..4
    art_use_spotify: bool = True    # Auto-pull live Spotify Track + Artist
    art_custom_title: str = "SPECIMEN // LIVE SET"
    art_custom_sub: str = "AUDIOVISUAL PERFORMANCE"
    art_overlay_opacity: float = 0.88


@dataclass
class ClipSlot:
    path: str
    name: str
    kind: str  # "video" | "image"


class ClipBank:
    """Resolume-inspired 8-slot Media Clip Bank for instant video/image triggering and Drop auto-switching."""

    MAX_SLOTS = 8

    def __init__(self) -> None:
        self.slots: List[Optional[ClipSlot]] = [None] * self.MAX_SLOTS
        self.active_slot_a: int = -1
        self.active_slot_b: int = -1

    def add_clip(self, filepath: str) -> int:
        p = Path(filepath)
        if not p.is_file():
            return -1
        # Avoid duplicates; return existing slot index
        for i, sl in enumerate(self.slots):
            if sl is not None and sl.path == str(p):
                return i
        ext = p.suffix.lower()
        kind = "image" if ext in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"} else "video"
        slot = ClipSlot(path=str(p), name=p.name, kind=kind)
        for i in range(self.MAX_SLOTS):
            if self.slots[i] is None:
                self.slots[i] = slot
                return i
        # Replace oldest/last slot if bank is full
        self.slots[-1] = slot
        return self.MAX_SLOTS - 1

    def remove_clip(self, idx: int) -> None:
        if 0 <= idx < self.MAX_SLOTS:
            self.slots[idx] = None
            if self.active_slot_a == idx:
                self.active_slot_a = -1
            if self.active_slot_b == idx:
                self.active_slot_b = -1

    def filled_indices(self) -> List[int]:
        return [i for i, s in enumerate(self.slots) if s is not None]

    def next_clip_index(self) -> int:
        filled = self.filled_indices()
        if not filled:
            return -1
        if self.active_slot_a in filled:
            pos = filled.index(self.active_slot_a)
            return filled[(pos + 1) % len(filled)]
        return filled[0]


class LiveScreenCapture:
    """
    Non-blocking background desktop / window capture via Pillow ImageGrab.
    Streams live frames (YouTube, Spotify Canvas, games, browsers, 3D apps) at ~25 FPS into Deck A!
    """

    def __init__(self, target_w: int = 960, target_h: int = 540) -> None:
        self.target_w = target_w
        self.target_h = target_h
        self.crop_mode: int = 0
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest_rgb: Optional[np.ndarray] = None
        self._dirty = False
        self.status_message: str = "Captura de pantalla inactiva"

    @staticmethod
    def is_available() -> bool:
        return _SCREEN_GRAB_AVAILABLE

    def start(self) -> bool:
        if not _SCREEN_GRAB_AVAILABLE:
            self.status_message = "ImageGrab no disponible"
            return False
        if self._running:
            return True
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self.status_message = "LIVE SCREEN CAPTURE ACTIVO"
        return True

    def stop(self) -> None:
        self._running = False
        self.status_message = "Captura de pantalla detenida"

    def _loop(self) -> None:
        while self._running:
            t0 = time.perf_counter()
            try:
                img = ImageGrab.grab()
                w, h = img.size
                cm = self.crop_mode
                if cm == 1:
                    img = img.crop((0, 0, w // 2, h))
                elif cm == 2:
                    img = img.crop((w // 2, 0, w, h))
                elif cm == 3:
                    cw = int(w * 0.65)
                    ch = int(cw * 9 / 16)
                    x0 = max(0, (w - cw) // 2)
                    y0 = max(0, (h - ch) // 2)
                    img = img.crop((x0, y0, min(w, x0 + cw), min(h, y0 + ch)))
                img = img.convert("RGB").resize((self.target_w, self.target_h), Image.Resampling.BILINEAR)
                arr = np.ascontiguousarray(np.asarray(img, dtype=np.uint8))
                with self._lock:
                    self._latest_rgb = arr
                    self._dirty = True
            except Exception as exc:
                self.status_message = f"Err captura: {str(exc)[:30]}"
                time.sleep(0.25)
            elapsed = time.perf_counter() - t0
            sleep_t = max(0.008, (1.0 / 26.0) - elapsed)
            time.sleep(sleep_t)

    def consume_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            if not self._dirty or self._latest_rgb is None:
                return None
            self._dirty = False
            return self._latest_rgb


@dataclass
class PerformanceMacro:
    """Complete ART + VIDEO + MUSIC live performance scene snapshot (Resolume / Ableton Cue Pad)."""

    name: str
    preset_name: str
    disp_mode: int
    disp_strength: float
    blend_mode: int
    crossfade: float
    deck_b_scene: int
    blob_enabled: bool
    blob_style: int
    choreography_idx: int
    theme_idx: int
    art_overlay_style: int


def default_performance_macros() -> List[PerformanceMacro]:
    return [
        PerformanceMacro(
            name="S1: SPECIMEN LIVE",
            preset_name="SPECIMEN 001 (Violet)",
            disp_mode=0,
            disp_strength=0.35,
            blend_mode=5,
            crossfade=0.0,
            deck_b_scene=1,
            blob_enabled=True,
            blob_style=0,
            choreography_idx=0,
            theme_idx=0,
            art_overlay_style=1,
        ),
        PerformanceMacro(
            name="S2: LIQUID FUSION",
            preset_name="LIQUID MERCURY CURL",
            disp_mode=1,
            disp_strength=0.82,
            blend_mode=5,
            crossfade=0.55,
            deck_b_scene=2,
            blob_enabled=True,
            blob_style=2,
            choreography_idx=8,
            theme_idx=1,
            art_overlay_style=3,
        ),
        PerformanceMacro(
            name="S3: KALEIDO RAVE",
            preset_name="KALEIDO RAVE PRISM",
            disp_mode=9,
            disp_strength=0.85,
            blend_mode=1,
            crossfade=0.45,
            deck_b_scene=4,
            blob_enabled=True,
            blob_style=3,
            choreography_idx=9,
            theme_idx=0,
            art_overlay_style=2,
        ),
        PerformanceMacro(
            name="S4: CYBER GLITCH",
            preset_name="DATAMOSH BREAKBEAT",
            disp_mode=6,
            disp_strength=0.88,
            blend_mode=4,
            crossfade=0.38,
            deck_b_scene=5,
            blob_enabled=True,
            blob_style=4,
            choreography_idx=4,
            theme_idx=2,
            art_overlay_style=4,
        ),
        PerformanceMacro(
            name="S5: SOLAR CINEMA",
            preset_name="SOLAR ANAMORPHIC",
            disp_mode=8,
            disp_strength=0.55,
            blend_mode=2,
            crossfade=0.42,
            deck_b_scene=6,
            blob_enabled=True,
            blob_style=1,
            choreography_idx=2,
            theme_idx=0,
            art_overlay_style=1,
        ),
        PerformanceMacro(
            name="S6: CLEAN NOIR",
            preset_name="CREATION NOIR (Hands)",
            disp_mode=3,
            disp_strength=0.25,
            blend_mode=0,
            crossfade=0.0,
            deck_b_scene=0,
            blob_enabled=False,
            blob_style=0,
            choreography_idx=5,
            theme_idx=2,
            art_overlay_style=4,
        ),
    ]
