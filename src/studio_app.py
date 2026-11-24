from __future__ import annotations

import math
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import moderngl
import numpy as np
from imgui_bundle import (
    glfw_utils,
    hello_imgui,
    imgui,
    immapp,
    portable_file_dialogs as pfd,
)

from .audio_engine import AUDIO_SOURCE_NAMES, AudioEngine
from .blob_tracker import BLOB_DETECT_MODES, BLOB_VISUAL_STYLES, IntelligentBlobTracker
from .config import (
    COLOR_MODE_NAMES,
    DISPLACEMENT_MODE_NAMES,
    PROCEDURAL_SCENE_NAMES,
    RENDER_MODE_NAMES,
    TONEMAP_NAMES,
    PresetMorpher,
    ShaderParams,
)
from .exporter import (
    EXPORT_FORMATS,
    ExportSettings,
    SteppedVideoExporter,
    save_png_snapshot,
)
from .glyph_atlas import CHARSET_LABELS, CHARSET_ORDER, find_monospace_font
from .gpu_pipeline import AsciiBloomPipeline
from .interaction import INTERACTION_MODES, SPLIT_MODES, InteractionState
from .media_engine import MediaEngine
from .music_intelligence import CHOREOGRAPHY_PROFILES, NOTE_NAMES
from .presets import PRESET_CATEGORY_ORDER, PresetManager
from .specimen_ui import (
    SPECIMEN_THEMES,
    SpecimenTheme,
    apply_specimen_imgui_theme,
    draw_art_typography_overlay,
    draw_audio_scope_panel,
    draw_beat_sequencer,
    draw_chroma_bars,
    draw_drop_alert_banner,
    draw_glow_node,
    draw_glow_stars,
    draw_hud_scanbeam,
    draw_intelligent_blobs,
    draw_specimen_box,
    draw_wireframe_3d_scope,
    marquee_text,
)
from .spotify_bridge import SpotifyWindowsBridge
from .video_studio import (
    ART_OVERLAY_STYLES,
    BLEND_MODE_NAMES,
    FIT_MODE_NAMES,
    MIRROR_MODE_NAMES,
    SCREEN_CROP_NAMES,
    ClipBank,
    LiveScreenCapture,
    PerformanceMacro,
    VideoMixerState,
    default_performance_macros,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHADERS_DIR = PROJECT_ROOT / "shaders"
PRESETS_DIR = PROJECT_ROOT / "presets"
EXPORTS_DIR = PROJECT_ROOT / "exports"


def _fmt_time(sec: float) -> str:
    sec = max(0.0, float(sec))
    mins = int(sec // 60)
    rem = sec - mins * 60
    return f"{mins:02d}:{rem:05.2f}"


class LumenAsciiStudioApp:
    def __init__(self) -> None:
        EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
        PRESETS_DIR.mkdir(parents=True, exist_ok=True)

        self.ctx: Optional[moderngl.Context] = None
        self.pipeline: Optional[AsciiBloomPipeline] = None

        # Theme (0=SPECIMEN VIOLET, 1=SUBJECT CYAN, 2=MONOLITH NOIR)
        self.theme_idx: int = 0
        self.theme: SpecimenTheme = SPECIMEN_THEMES[0]
        self.mono_font = None

        # Presets & Morphing
        self.preset_manager = PresetManager(PRESETS_DIR)
        self.params = self.preset_manager.presets["SPECIMEN 001 (Violet)"].clone()
        self.active_preset_name: str = "SPECIMEN 001 (Violet)"
        self.morpher = PresetMorpher()
        self.morph_duration: float = 0.72

        self.interaction = InteractionState()
        self.audio = AudioEngine()
        self.spotify = SpotifyWindowsBridge()
        self.media = MediaEngine(on_audio_ready=self._on_video_audio_ready)
        self.media.procedural_scene = 1  # Start with glowing 3D Wireframe Heart!
        self.media_b = MediaEngine()     # Independent Deck B player for Dual-Deck Art+Video Fusion!
        self.mixer = VideoMixerState()
        self.clip_bank = ClipBank()
        self.screen_cap = LiveScreenCapture()
        self.macros: List[PerformanceMacro] = default_performance_macros()
        self.active_macro_idx: int = -1
        self._bar_counter: int = 0
        self._last_beat_idx: int = 0
        self.exporter = SteppedVideoExporter()
        self.export_settings = ExportSettings()

        # Specimen HUD interactive & intelligent CV blob tracking
        self.blob_tracker = IntelligentBlobTracker()
        self.scope_mode: int = 0
        self.mini_3d_shape: int = 0
        self._fft_peak_hold = np.zeros(64, dtype=np.float32)

        # Animated HUD transitions, Preset Categories, In-App Guide & Immersive Fullscreen Cinema!
        self.show_inspector: bool = True
        self._hud_slide: float = 1.0          # Smoothly animates 1.0 (open) <-> 0.0 (collapsed)
        self.immersive_mode: bool = False     # True = 100% edge-to-edge monitor immersion + auto-hiding hover dock
        self.show_immersive_pill: bool = True # Minimal corner Spotify/BPM pill in Cinema mode (H key)
        self._hover_dock_anim: float = 0.0    # Smooth slide-down animation for top hover dock in Cinema mode
        self.show_guide_drawer: bool = False  # Animated slide-over Interactive Guide modal ([F1] or ? GUIA)
        self._guide_slide: float = 0.0
        self._active_tab_id: str = "ENGINE"
        self._tab_glide_y: float = 0.0        # Smooth vertical glide-in when switching inspector tabs!
        self.preset_cat_idx: int = 0          # Active preset category filter chip
        self._is_fullscreen: bool = False
        self._windowed_rect = (90, 60, 1380, 820)

        # Runtime state
        self._start_time = time.perf_counter()
        self._last_frame_time = self._start_time
        self._fps_smooth: float = 60.0
        self._tex_final_glo: int = 0

        self.inspector_width: float = 435.0
        self.auto_hot_reload: bool = True
        self.fit_viewport_fill: bool = False
        self.render_res_idx: int = 0
        self.save_preset_name: str = "SPECIMEN CUSTOM"
        self.toast_message: str = "SPOTIFY/PC AUDIO WASAPI LISTO // [F11] FULLSCREEN // [F] CINE"
        self.toast_until: float = time.perf_counter() + 6.5

        self._dropped_files: List[str] = []
        self._open_media_dialog = None
        self._open_clips_dialog = None
        self._open_font_dialog = None
        self._save_png_dialog = None
        self._save_video_dialog = None
        self._drop_callback_ref = None

    def _notify(self, msg: str, duration: float = 3.5) -> None:
        self.toast_message = msg.upper()
        self.toast_until = time.perf_counter() + duration

    def _set_theme(self, idx: int) -> None:
        self.theme_idx = idx % len(SPECIMEN_THEMES)
        self.theme = SPECIMEN_THEMES[self.theme_idx]
        apply_specimen_imgui_theme(self.theme)
        if self.theme_idx == 0:
            self.scope_mode = 0
        elif self.theme_idx == 1:
            self.scope_mode = 1
        self._notify(f"HUD THEME // {self.theme.specimen_code}")

    def _on_video_audio_ready(self, pcm: Optional[np.ndarray], sr: int) -> None:
        self.audio.set_video_audio(pcm, sr)

    def load_additional_fonts(self) -> None:
        font_path, _ = find_monospace_font()
        if font_path:
            try:
                flp = hello_imgui.FontLoadingParams()
                flp.inside_assets = False
                self.mono_font = hello_imgui.load_font(font_path, 14.5, flp)
                return
            except Exception:
                pass
        hello_imgui.imgui_default_settings.load_default_font_with_font_awesome_icons()

    def setup_imgui_style(self) -> None:
        apply_specimen_imgui_theme(self.theme)

    def post_init(self) -> None:
        apply_specimen_imgui_theme(self.theme)
        self.ctx = moderngl.create_context()
        self.pipeline = AsciiBloomPipeline(self.ctx, SHADERS_DIR, width=1280, height=720)
        self._rebuild_atlas_from_params()
        self._install_glfw_drop_callback()

    def _install_glfw_drop_callback(self) -> None:
        try:
            win = glfw_utils.glfw_window_hello_imgui()
            glfw_mod = glfw_utils.glfw

            def _on_drop(window, paths):
                if isinstance(paths, (list, tuple)):
                    for item in paths:
                        self._dropped_files.append(str(item))
                elif isinstance(paths, str):
                    self._dropped_files.append(paths)

            self._drop_callback_ref = _on_drop
            glfw_mod.set_drop_callback(win, _on_drop)
        except Exception:
            pass

    def _find_current_monitor(self, glfw_mod, win):
        """Finds whichever monitor currently contains the center of the window (multi-monitor safe!)."""
        try:
            wx, wy = glfw_mod.get_window_pos(win)
            ww, wh = glfw_mod.get_window_size(win)
            cx, cy = wx + ww * 0.5, wy + wh * 0.5
            monitors = glfw_mod.get_monitors() or []
            for m in monitors:
                mx, my = glfw_mod.get_monitor_pos(m)
                mode = glfw_mod.get_video_mode(m)
                if mx <= cx <= mx + mode.size.width and my <= cy <= my + mode.size.height:
                    return m
        except Exception:
            pass
        return glfw_mod.get_primary_monitor()

    def _set_fullscreen(self, enable: bool) -> None:
        if enable == self._is_fullscreen:
            return
        try:
            win = glfw_utils.glfw_window_hello_imgui()
            glfw_mod = glfw_utils.glfw
            if enable:
                x, y = glfw_mod.get_window_pos(win)
                w, h = glfw_mod.get_window_size(win)
                self._windowed_rect = (x, y, max(640, w), max(400, h))
                monitor = self._find_current_monitor(glfw_mod, win)
                mode = glfw_mod.get_video_mode(monitor)
                glfw_mod.set_window_monitor(
                    win, monitor, 0, 0, mode.size.width, mode.size.height, mode.refresh_rate
                )
                self._is_fullscreen = True
            else:
                x, y, w, h = self._windowed_rect
                glfw_mod.set_window_monitor(win, None, x, y, w, h, 0)
                self._is_fullscreen = False
        except Exception as exc:
            self._notify(f"FULLSCREEN ERR: {exc}")

    def _toggle_fullscreen(self) -> None:
        self._set_fullscreen(not self._is_fullscreen)
        if not self._is_fullscreen and self.immersive_mode:
            self.immersive_mode = False
        self._notify("PANTALLA COMPLETA // ON [F11 / ESC]" if self._is_fullscreen else "MODO VENTANA")

    def _toggle_immersive_cinema(self) -> None:
        """True 100% edge-to-edge Fullscreen Cinema Immersion with auto-hiding floating Specimen dock!"""
        self.immersive_mode = not self.immersive_mode
        if self.immersive_mode:
            self._set_fullscreen(True)
            self._notify("MODO CINE INMERSIVO // MUEVE EL MOUSE ARRIBA PARA CONTROLES // ESC PARA SALIR", duration=4.5)
        else:
            self._set_fullscreen(False)
            self._notify("MODO ESTUDIO RESTAURADO")

    def _cycle_preset(self, delta: int) -> None:
        pnames = list(self.preset_manager.presets.keys())
        if not pnames:
            return
        cur_i = pnames.index(self.active_preset_name) if self.active_preset_name in pnames else 0
        self._trigger_preset(pnames[(cur_i + delta) % len(pnames)])

    def _rebuild_atlas_from_params(self) -> None:
        if self.pipeline is None:
            return
        self.pipeline.rebuild_atlas(
            charset_id=self.params.charset_id,
            custom_charset=self.params.custom_charset,
            font_path=self.params.font_path,
        )

    def _apply_render_resolution(self) -> None:
        if self.pipeline is None:
            return
        if self.render_res_idx == 0:
            w, h = 1280, 720
        elif self.render_res_idx == 1:
            w, h = 1600, 900
        elif self.render_res_idx == 2:
            w, h = 1920, 1080
        else:
            w = max(320, min(self.media.info.width, 2560))
            h = max(180, min(self.media.info.height, 1440))
            w -= w % 2
            h -= h % 2
        self.pipeline.resize(w, h)

    def _match_aspect_for_media(self) -> None:
        """
        Matches the GPU pipeline aspect ratio to the loaded media while ALWAYS preserving HD density
        (scales UP low-res clips to 720p/900p/1080p target so ASCII cells & Stipple dots never look chunky!).
        """
        if self.pipeline is None:
            return
        mw = max(1, self.media.info.width)
        mh = max(1, self.media.info.height)
        target_max_w = 1280 if self.render_res_idx == 0 else (1600 if self.render_res_idx == 1 else 1920)
        target_max_h = 720 if self.render_res_idx == 0 else (900 if self.render_res_idx == 1 else 1080)
        scale = min(target_max_w / mw, target_max_h / mh)
        w = max(640, int(round(mw * scale)))
        h = max(360, int(round(mh * scale)))
        w -= w % 2
        h -= h % 2
        self.pipeline.resize(w, h)

    def _cancel_morph_if(self, changed: bool) -> bool:
        """Cancels any active preset morph immediately when the user edits a slider so controls NEVER lock!"""
        if changed and self.morpher.active:
            self.morpher.cancel()
        return changed

    def _handle_file_input(self, filepath: str) -> None:
        p = Path(filepath)
        ext = p.suffix.lower()
        if ext in (".ttf", ".otf", ".ttc"):
            self.params.font_path = str(p)
            self._rebuild_atlas_from_params()
            self._notify(f"FUENTE CARGADA // {p.name}")
        elif ext == ".json":
            try:
                import json

                data = json.loads(p.read_text(encoding="utf-8"))
                loaded = ShaderParams.from_dict(data.get("params", data))
                atlas_changed = self.morpher.start(self.params, loaded, name=p.stem, duration=self.morph_duration)
                if atlas_changed:
                    self._rebuild_atlas_from_params()
                self.active_preset_name = p.stem
                self._notify(f"PRESET CARGADO // {p.name}")
            except Exception as exc:
                self._notify(f"ERROR JSON: {exc}")
        else:
            if self.mixer.live_screen_active:
                self.screen_cap.stop()
                self.mixer.live_screen_active = False
            if self.media.load_file(str(p)):
                slot_i = self.clip_bank.add_clip(str(p))
                self.clip_bank.active_slot_a = slot_i
                self._match_aspect_for_media()
                self.blob_tracker.reset()
                if self.pipeline is not None:
                    self.pipeline.clear_trails_and_paint(clear_trails=True, clear_paint=False)
                self._notify(self.media.status_message)
            else:
                self._notify(self.media.status_message)

    def _load_clip_to_deck_a(self, slot_idx: int) -> None:
        if not (0 <= slot_idx < len(self.clip_bank.slots)):
            return
        slot = self.clip_bank.slots[slot_idx]
        if slot is None:
            return
        if self.mixer.live_screen_active:
            self.screen_cap.stop()
            self.mixer.live_screen_active = False
        if self.media.load_file(slot.path):
            self.clip_bank.active_slot_a = slot_idx
            self._match_aspect_for_media()
            self.blob_tracker.reset()
            self._notify(f"DECK A [0{slot_idx+1}] // {slot.name[:24]}")

    def _load_clip_to_deck_b(self, slot_idx: int) -> None:
        if not (0 <= slot_idx < len(self.clip_bank.slots)):
            return
        slot = self.clip_bank.slots[slot_idx]
        if slot is None:
            return
        if self.media_b.load_file(slot.path):
            self.clip_bank.active_slot_b = slot_idx
            self.mixer.deck_b_mode = 1
            if self.mixer.crossfade < 0.15:
                self.mixer.crossfade = 0.50
            self._notify(f"DECK B [0{slot_idx+1}] // {slot.name[:24]}")

    def _toggle_live_screen(self) -> None:
        if self.mixer.live_screen_active:
            self.screen_cap.stop()
            self.mixer.live_screen_active = False
            self._notify("LIVE SCREEN CAPTURE // OFF")
        else:
            self.screen_cap.crop_mode = self.mixer.screen_crop_mode
            if self.screen_cap.start():
                self.mixer.live_screen_active = True
                self._notify("LIVE SCREEN CAPTURE // ON (DECK A)")
            else:
                self._notify(self.screen_cap.status_message)

    def _trigger_macro(self, idx: int) -> None:
        if not (0 <= idx < len(self.macros)):
            return
        mc = self.macros[idx]
        self.active_macro_idx = idx
        if mc.preset_name in self.preset_manager.presets:
            self._trigger_preset(mc.preset_name)
        self.morpher.cancel()
        if mc.preset_name in self.preset_manager.presets:
            target_p = self.preset_manager.presets[mc.preset_name].clone()
            target_p.disp_mode = mc.disp_mode
            target_p.disp_strength = mc.disp_strength
            atlas_ch = self.morpher.start(self.params, target_p, name=mc.preset_name, duration=self.morph_duration)
            if atlas_ch:
                self._rebuild_atlas_from_params()
        else:
            self.params.disp_mode = mc.disp_mode
            self.params.disp_strength = mc.disp_strength
        self.mixer.blend_mode = mc.blend_mode
        self.mixer.crossfade = mc.crossfade
        self.mixer.deck_b_scene = mc.deck_b_scene
        self.mixer.art_overlay_style = mc.art_overlay_style
        self.blob_tracker.settings.enabled = mc.blob_enabled
        self.blob_tracker.settings.visual_style = mc.blob_style
        self.audio.routing.profile_idx = mc.choreography_idx % len(CHOREOGRAPHY_PROFILES)
        self._set_theme(mc.theme_idx)
        self._notify(f"SCENE MACRO // {mc.name}", duration=2.8)

    def _save_current_to_macro(self, idx: int) -> None:
        if not (0 <= idx < len(self.macros)):
            return
        self.macros[idx] = PerformanceMacro(
            name=f"S{idx+1}: {self.active_preset_name[:12]}",
            preset_name=self.active_preset_name,
            disp_mode=int(self.params.disp_mode),
            disp_strength=float(self.params.disp_strength),
            blend_mode=int(self.mixer.blend_mode),
            crossfade=float(self.mixer.crossfade),
            deck_b_scene=int(self.mixer.deck_b_scene),
            blob_enabled=bool(self.blob_tracker.settings.enabled),
            blob_style=int(self.blob_tracker.settings.visual_style),
            choreography_idx=int(self.audio.routing.profile_idx),
            theme_idx=int(self.theme_idx),
            art_overlay_style=int(self.mixer.art_overlay_style),
        )
        self.active_macro_idx = idx
        self._notify(f"MACRO GUARDADO // S{idx+1}")

    def _poll_dialogs_and_drops(self) -> None:
        while self._dropped_files:
            fp = self._dropped_files.pop(0)
            self._handle_file_input(fp)

        if self._open_media_dialog is not None and self._open_media_dialog.ready():
            res = self._open_media_dialog.result()
            self._open_media_dialog = None
            if res:
                self._handle_file_input(res[0])

        if self._open_clips_dialog is not None and self._open_clips_dialog.ready():
            res = self._open_clips_dialog.result()
            self._open_clips_dialog = None
            if res:
                added = 0
                first_slot = -1
                for fp in res:
                    si = self.clip_bank.add_clip(fp)
                    if si >= 0:
                        added += 1
                        if first_slot < 0:
                            first_slot = si
                if added > 0:
                    self._notify(f"CLIP BANK // +{added} CLIPS CARGADOS")
                    if self.media.info.source_type == "procedural" and first_slot >= 0:
                        self._load_clip_to_deck_a(first_slot)

        if self._open_font_dialog is not None and self._open_font_dialog.ready():
            res = self._open_font_dialog.result()
            self._open_font_dialog = None
            if res:
                self._handle_file_input(res[0])

        if self._save_png_dialog is not None and self._save_png_dialog.ready():
            res = self._save_png_dialog.result()
            self._save_png_dialog = None
            if res and self.pipeline is not None:
                out = Path(res)
                if out.suffix.lower() != ".png":
                    out = out.with_suffix(".png")
                try:
                    save_png_snapshot(self.pipeline, out)
                    self._notify(f"PNG GUARDADO // {out.name}")
                except Exception as exc:
                    self._notify(f"ERROR PNG: {exc}")

        if self._save_video_dialog is not None and self._save_video_dialog.ready():
            res = self._save_video_dialog.result()
            self._save_video_dialog = None
            if res and self.pipeline is not None:
                ext_map = {0: ".mp4", 1: ".gif", 2: ".webm"}
                out = Path(res)
                wanted_ext = ext_map.get(self.export_settings.format_idx, ".mp4")
                if out.suffix.lower() != wanted_ext:
                    out = out.with_suffix(wanted_ext)
                ok = self.exporter.start(out, self.export_settings, self.media, self.pipeline)
                self._notify(self.exporter.status_message if ok else self.exporter.error_message)

    def _handle_shortcuts(self) -> None:
        io = imgui.get_io()
        if io.want_text_input:
            return

        if imgui.is_key_pressed(imgui.Key.escape):
            if self.show_guide_drawer:
                self.show_guide_drawer = False
            elif self.immersive_mode or self._is_fullscreen:
                self.immersive_mode = False
                self._set_fullscreen(False)
                self._notify("MODO VENTANA")
        if imgui.is_key_pressed(imgui.Key.f1):
            self.show_guide_drawer = not self.show_guide_drawer
        if imgui.is_key_pressed(imgui.Key.tab):
            if self.immersive_mode:
                self.immersive_mode = False
                self.show_inspector = True
            else:
                self.show_inspector = not self.show_inspector
        if imgui.is_key_pressed(imgui.Key.f11):
            self._toggle_fullscreen()
        if imgui.is_key_pressed(imgui.Key.f):
            self._toggle_immersive_cinema()
        if imgui.is_key_pressed(imgui.Key.h):
            self.show_immersive_pill = not self.show_immersive_pill
        if imgui.is_key_pressed(imgui.Key.space) and not self.exporter.active:
            self.media.playing = not self.media.playing
        if imgui.is_key_pressed(imgui.Key.s):
            self.interaction.split_enabled = not self.interaction.split_enabled
        if imgui.is_key_pressed(imgui.Key.x):
            self.blob_tracker.settings.enabled = not self.blob_tracker.settings.enabled
            self._notify(f"BLOB TRACKING // {'ON' if self.blob_tracker.settings.enabled else 'OFF'}")
        if imgui.is_key_pressed(imgui.Key.b):
            self.blob_tracker.settings.visual_style = (self.blob_tracker.settings.visual_style + 1) % len(BLOB_VISUAL_STYLES)
            self._notify(f"BLOB STYLE // {BLOB_VISUAL_STYLES[self.blob_tracker.settings.visual_style]}")
        if imgui.is_key_pressed(imgui.Key.m):
            self.morpher.cancel()
            self.params.disp_mode = (int(self.params.disp_mode) + 1) % len(DISPLACEMENT_MODE_NAMES)
            if self.params.disp_mode > 0 and self.params.disp_strength < 0.25:
                self.params.disp_strength = 0.65
            self._notify(f"DESPLAZAMIENTO // {DISPLACEMENT_MODE_NAMES[self.params.disp_mode]}")
        if imgui.is_key_pressed(imgui.Key.p):
            self._cycle_preset(1)
        if imgui.is_key_pressed(imgui.Key.v):
            self.mixer.blend_mode = (int(self.mixer.blend_mode) + 1) % len(BLEND_MODE_NAMES)
            if self.mixer.crossfade < 0.15:
                self.mixer.crossfade = 0.50
            self._notify(f"BLEND // {BLEND_MODE_NAMES[self.mixer.blend_mode]}")
        if imgui.is_key_pressed(imgui.Key.t):
            self.mixer.art_overlay_style = (int(self.mixer.art_overlay_style) + 1) % len(ART_OVERLAY_STYLES)
            self._notify(f"ART OVERLAY // {ART_OVERLAY_STYLES[self.mixer.art_overlay_style]}")
        if imgui.is_key_pressed(imgui.Key.l):
            self._toggle_live_screen()
        if imgui.is_key_pressed(imgui.Key.r):
            self.interaction.reset_view()
            self._notify("VISTA RESTABLECIDA // ZOOM 1.0X")
        if imgui.is_key_pressed(imgui.Key.c):
            self.interaction.clear_paint_requested = True
            if self.pipeline is not None:
                self.pipeline.clear_trails_and_paint(clear_trails=True, clear_paint=True)
            self._notify("TRAILS & PINTURA LIMPIADOS")
        if imgui.is_key_pressed(imgui.Key.d):
            self.audio.trigger_manual_drop()
        if imgui.is_key_pressed(imgui.Key.f5) and self.pipeline is not None:
            self.pipeline.check_hot_reload(force=True)
            self._notify(self.pipeline.last_reload_status)

        if self.media.info.source_type == "video" and not self.exporter.active:
            if imgui.is_key_pressed(imgui.Key.left_arrow):
                self.media.step_frame(-1)
            if imgui.is_key_pressed(imgui.Key.right_arrow):
                self.media.step_frame(1)

        mode_keys = [
            (imgui.Key._1, 1),
            (imgui.Key._2, 2),
            (imgui.Key._3, 3),
            (imgui.Key._4, 4),
            (imgui.Key._5, 5),
            (imgui.Key._6, 0),
        ]
        for key, mode_val in mode_keys:
            if imgui.is_key_pressed(key):
                self.interaction.mode = mode_val
                self._notify(f"MODO CURSOR // {INTERACTION_MODES[mode_val]}")

    def _trigger_preset(self, name: str) -> None:
        target = self.preset_manager.presets.get(name)
        if target is None:
            return
        self.active_preset_name = name
        atlas_changed = self.morpher.start(self.params, target, name=name, duration=self.morph_duration)
        if atlas_changed:
            self._rebuild_atlas_from_params()

        if name == "SPECIMEN 001 (Violet)":
            self._set_theme(0)
            if self.media.info.source_type == "procedural":
                self.media.procedural_scene = 1
        elif name == "SUBJECT A-34 (Cyan)":
            self._set_theme(1)
            if self.media.info.source_type == "procedural":
                self.media.procedural_scene = 3
        elif name == "KINETIC SPHERE (MP4)":
            self._set_theme(2)
            if self.media.info.source_type == "procedural":
                self.media.procedural_scene = 0
        elif name == "CREATION NOIR (Hands)":
            self._set_theme(2)

        self._notify(f"PRESET // {name}", duration=2.4)

    def _quick_png_snapshot(self) -> None:
        if self.pipeline is None:
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = EXPORTS_DIR / f"specimen_ascii_{stamp}.png"
        try:
            save_png_snapshot(self.pipeline, out)
            self._notify(f"CAPTURA GUARDADA // exports/{out.name}")
        except Exception as exc:
            self._notify(f"ERROR PNG: {exc}")

    def _open_media_picker(self) -> None:
        self._open_media_dialog = pfd.open_file(
            "Abrir Video o Imagen",
            str(Path.home()),
            [
                "Videos e Imágenes",
                "*.mp4 *.mov *.mkv *.webm *.avi *.gif *.m4v *.png *.jpg *.jpeg *.webp *.bmp",
                "Todos los archivos",
                "*",
            ],
        )

    def _open_clips_picker(self) -> None:
        self._open_clips_dialog = pfd.open_file(
            "Añadir Videos o Imágenes al Banco de Clips (hasta 8)",
            str(Path.home()),
            [
                "Videos e Imágenes",
                "*.mp4 *.mov *.mkv *.webm *.avi *.gif *.m4v *.png *.jpg *.jpeg *.webp *.bmp",
                "Todos los archivos",
                "*",
            ],
            pfd.opt.multiselect,
        )

    def _open_font_picker(self) -> None:
        self._open_font_dialog = pfd.open_file(
            "Seleccionar Fuente Monoespaciada (.ttf / .otf)",
            r"C:\Windows\Fonts",
            ["Fuentes TrueType/OpenType", "*.ttf *.otf *.ttc", "Todos", "*"],
        )

    def _open_video_export_dialog(self) -> None:
        ext_info = {
            0: ("Video MP4 (*.mp4)", "*.mp4", "specimen_ascii_bloom.mp4"),
            1: ("GIF Animado (*.gif)", "*.gif", "specimen_ascii_bloom.gif"),
            2: ("Video WebM (*.webm)", "*.webm", "specimen_ascii_bloom.webm"),
        }
        label, pattern, default_name = ext_info.get(self.export_settings.format_idx, ext_info[0])
        self._save_video_dialog = pfd.save_file(
            "Exportar Animación ASCII + Bloom",
            str(EXPORTS_DIR / default_name),
            [label, pattern, "Todos", "*"],
        )

    def _step_simulation_and_gpu(self) -> None:
        now = time.perf_counter()
        dt = min(max(now - self._last_frame_time, 1e-4), 0.1)
        self._last_frame_time = now
        self._fps_smooth = self._fps_smooth * 0.92 + (1.0 / dt) * 0.08
        app_time = now - self._start_time

        # Smooth HUD, Guide Drawer & Tab glide displacement animations (exponential ease-out)
        target_slide = 1.0 if (self.show_inspector and not self.immersive_mode) else 0.0
        self._hud_slide += (target_slide - self._hud_slide) * min(1.0, dt * 11.5)
        if abs(self._hud_slide - target_slide) < 0.005:
            self._hud_slide = target_slide

        target_guide = 1.0 if self.show_guide_drawer else 0.0
        self._guide_slide += (target_guide - self._guide_slide) * min(1.0, dt * 12.0)
        if abs(self._guide_slide - target_guide) < 0.004:
            self._guide_slide = target_guide

        self._tab_glide_y *= math.exp(-dt * 16.0)
        if self._tab_glide_y < 0.25:
            self._tab_glide_y = 0.0

        if self.pipeline is None:
            return

        if self.auto_hot_reload:
            if self.pipeline.check_hot_reload(dt=dt):
                self._notify(self.pipeline.last_reload_status, duration=3.0)

        if self.exporter.active:
            self.exporter.step(
                pipeline=self.pipeline,
                params=self.params,
                interaction=self.interaction,
                media=self.media,
                audio=self.audio,
                frames_per_tick=2,
                mixer=self.mixer,
                use_media_b=(self.media_b.info.source_type in ("video", "image")),
            )
            self._tex_final_glo = int(self.pipeline.tex_final.glo)
            return

        self.morpher.update(dt, self.params)

        # Beat-synced video speed boost on Kicks (temporarily accelerates playback on bass punches!)
        saved_speed = self.media.speed
        if self.mixer.beat_speed_boost > 0.01 and self.media.info.source_type == "video":
            self.media.speed = min(3.5, saved_speed * (1.0 + self.audio.intel.spring_kick * self.mixer.beat_speed_boost * 0.85))
        self.media.update(dt)
        self.media.speed = saved_speed

        # Live Screen / Window Capture takes over Deck A seamlessly when enabled!
        if self.mixer.live_screen_active:
            self.screen_cap.crop_mode = self.mixer.screen_crop_mode
            scr_frame = self.screen_cap.consume_frame()
            if scr_frame is not None:
                self.pipeline.upload_media_frame(scr_frame)
        else:
            frame = self.media.consume_frame()
            if frame is not None:
                self.pipeline.upload_media_frame(frame)

        # Independent Deck B video/image playback for Dual-Deck Art + Video Fusion!
        if self.mixer.deck_b_mode == 1 and self.media_b.info.source_type in ("video", "image"):
            self.media_b.update(dt)
            frame_b = self.media_b.consume_frame()
            if frame_b is not None:
                self.pipeline.upload_media_b_frame(frame_b)

        is_video = (not self.mixer.live_screen_active) and (self.media.info.source_type == "video")
        self.audio.sync_video_playback(self.media.current_time, self.media.playing and is_video, self.media.speed)
        self.audio.update(dt, self.media.current_time, self.media.playing and is_video)

        # Track 4/4 bars for musical auto-clip switching
        cur_beat = self.audio.intel.beat_index
        if cur_beat == 0 and self._last_beat_idx != 0 and self.audio.intel.section_id > 0:
            self._bar_counter += 1
            if self.mixer.auto_clip_bars > 0 and (self._bar_counter % self.mixer.auto_clip_bars == 0):
                nxt = self.clip_bank.next_clip_index()
                if nxt >= 0 and nxt != self.clip_bank.active_slot_a:
                    self._load_clip_to_deck_a(nxt)
        self._last_beat_idx = cur_beat

        if self.mixer.auto_clip_on_drop and self.audio.intel.drop_trigger:
            nxt = self.clip_bank.next_clip_index()
            if nxt >= 0 and nxt != self.clip_bank.active_slot_a:
                self._load_clip_to_deck_a(nxt)

        # Peak-hold laser cap decay for 64-band Mel FFT scope
        self._fft_peak_hold = np.maximum(self.audio.spectrum, self._fft_peak_hold - dt * 0.55)

        if self.audio.mod.spawn_ripple_strength > 0.01:
            self.interaction.spawn_ripple(0.5, 0.5, strength=self.audio.mod.spawn_ripple_strength)
        if self.audio.mod.request_auto_morph:
            self._cycle_preset(1)
            self._notify(f"DROP AUTO-MORPH // {self.active_preset_name}", duration=3.0)

        self.interaction.update(dt, now, beat_trigger=self.audio.beat_triggered)

        use_media = self.mixer.live_screen_active or (self.media.info.source_type in ("video", "image"))
        use_media_b = self.media_b.info.source_type in ("video", "image")
        render_t = self.media.current_time if is_video else (app_time if self.media.playing else self.media.current_time)
        if not is_video and self.media.playing:
            self.media.current_time = app_time

        self._tex_final_glo = self.pipeline.render(
            params=self.params,
            interaction=self.interaction,
            time_sec=render_t,
            audio_vec4=self.audio.as_vec4(),
            use_media_source=use_media,
            procedural_scene=self.media.procedural_scene,
            export_mode=False,
            audio_stems_vec4=self.audio.stems_vec4(),
            fft_spectrum=self.audio.spectrum,
            mod=self.audio.mod,
            mixer=self.mixer,
            use_media_b=use_media_b,
        )

        # Feed lightweight 160x90 GPU analysis texture into our intelligent CV Blob Tracker (<1ms)!
        if self.blob_tracker.settings.enabled:
            analysis_rgb = self.pipeline.read_analysis_rgb() if self.blob_tracker._analyze_accum >= 0.028 or not self.blob_tracker.tracks else None
            self.blob_tracker.update(analysis_rgb, dt)

    def _draw_top_specimen_bar(self) -> None:
        app_time = time.perf_counter() - self._start_time
        intel = self.audio.intel
        imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(10.0, 7.0))
        imgui.begin_child("SpecimenTopBar", imgui.ImVec2(0, 44), imgui.ChildFlags_.none, imgui.WindowFlags_.no_scrollbar)
        win_min = imgui.get_window_pos()
        win_size = imgui.get_window_size()
        dl = imgui.get_window_draw_list()

        imgui.push_font(None, 16.0)
        imgui.text_colored(imgui.ImVec4(*self.theme.node_core), self.theme.specimen_code)
        imgui.pop_font()
        imgui.same_line(0, 10)
        imgui.text_colored(imgui.ImVec4(*self.theme.line_core), "///")
        imgui.same_line(0, 10)

        if imgui.button("+ MEDIA"):
            self._open_media_picker()
        imgui.same_line(0, 5)

        imgui.set_next_item_width(172)
        preview_src = (
            f"3D: {PROCEDURAL_SCENE_NAMES[self.media.procedural_scene][:18]}"
            if self.media.info.source_type == "procedural"
            else f"FILE: {self.media.info.name[:17]}"
        )
        if imgui.begin_combo("##QuickSource", preview_src):
            for idx, sname in enumerate(PROCEDURAL_SCENE_NAMES):
                selected = self.media.info.source_type == "procedural" and self.media.procedural_scene == idx
                if imgui.selectable(f"3D // {sname}", selected)[0]:
                    self.media.set_procedural(idx)
                    self._apply_render_resolution()
                    if self.pipeline is not None:
                        self.pipeline.clear_trails_and_paint(clear_trails=True, clear_paint=False)
            imgui.end_combo()

        imgui.same_line(0, 10)
        quick_modes = [(1, "ONDAS"), (2, "AURA"), (3, "REPUL"), (4, "VORTEX"), (5, "PINCEL")]
        for m_id, label in quick_modes:
            is_active = self.interaction.mode == m_id
            if is_active:
                imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(*self.theme.frame_active))
                imgui.push_style_color(imgui.Col_.border, imgui.ImVec4(*self.theme.node_core))
            if imgui.button(label):
                self.interaction.mode = 0 if is_active else m_id
            if is_active:
                imgui.pop_style_color(2)
            imgui.same_line(0, 3)

        imgui.same_line(0, 6)
        split_on = self.interaction.split_enabled
        if split_on:
            imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(*self.theme.frame_active))
        if imgui.button("SPLIT"):
            self.interaction.split_enabled = not self.interaction.split_enabled
        if split_on:
            imgui.pop_style_color()
        imgui.same_line(0, 3)

        blobs_on = self.blob_tracker.settings.enabled
        if blobs_on:
            imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(*self.theme.frame_active))
        if imgui.button("BLOBS"):
            self.blob_tracker.settings.enabled = not self.blob_tracker.settings.enabled
        if blobs_on:
            imgui.pop_style_color()

        imgui.same_line(0, 8)
        if imgui.button("PNG"):
            self._quick_png_snapshot()
        imgui.same_line(0, 3)
        if imgui.button("REC"):
            self._open_video_export_dialog()

        # Right-aligned Theme Switcher + Fullscreen / Cinema + Guide + FPS + HUD
        avail_right = imgui.get_content_region_avail().x
        right_block_w = 552.0
        if avail_right > right_block_w:
            imgui.same_line(imgui.get_cursor_pos_x() + avail_right - right_block_w)
        else:
            imgui.same_line(0, 6)

        for idx, th in enumerate(SPECIMEN_THEMES):
            active_t = idx == self.theme_idx
            if active_t:
                imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(*th.frame_active))
                imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*th.node_core))
            if imgui.button(th.label):
                self._set_theme(idx)
            if active_t:
                imgui.pop_style_color(2)
            imgui.same_line(0, 3)

        imgui.same_line(0, 6)
        fs_on = self._is_fullscreen
        if fs_on:
            imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(*self.theme.frame_active))
        if imgui.button("FULL[F11]"):
            self._toggle_fullscreen()
        if fs_on:
            imgui.pop_style_color()
        imgui.same_line(0, 3)

        imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(*self.theme.frame_hover))
        if imgui.button("CINE[F]"):
            self._toggle_immersive_cinema()
        imgui.pop_style_color()
        imgui.same_line(0, 3)

        guide_on = self.show_guide_drawer
        if guide_on:
            imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(*self.theme.frame_active))
        if imgui.button("?GUIA"):
            self.show_guide_drawer = not self.show_guide_drawer
        if guide_on:
            imgui.pop_style_color()

        imgui.same_line(0, 6)
        imgui.text_colored(imgui.ImVec4(*self.theme.line_core), f"{self._fps_smooth:3.0f}FPS")
        imgui.same_line(0, 5)
        if imgui.button("HUD"):
            self.show_inspector = not self.show_inspector

        draw_specimen_box(
            dl,
            imgui.ImVec2(win_min.x + 1, win_min.y + 1),
            imgui.ImVec2(win_min.x + win_size.x - 1, win_min.y + win_size.y - 1),
            self.theme,
            with_nodes=True,
            intensity=0.95,
            anim_time=app_time,
            kick_pulse=intel.kick,
            drop_glow=intel.drop_intensity,
            num_pulses=2,
        )
        imgui.end_child()
        imgui.pop_style_var()

    def _draw_immersive_floating_overlays(
        self,
        dl: imgui.ImDrawList,
        img_pos: imgui.ImVec2,
        disp_w: float,
        disp_h: float,
        app_time: float,
    ) -> None:
        """
        Ultra-immersive Cinema Fullscreen overlays:
          1. Bottom-left minimal glowing Spotify + BPM + Beat LEDs + Section pill (toggle with H)
          2. Auto-hiding top hover dock when cursor approaches top 58px of screen!
        """
        io = imgui.get_io()
        intel = self.audio.intel
        now_playing = self.spotify.get_info()

        # 1. Bottom-left minimal floating Music Telemetry Pill (auto-sized to exact text width!)
        if self.show_immersive_pill:
            title = marquee_text((now_playing.title or self.media.info.name).upper(), 30, app_time)
            meta = f"{intel.bpm:5.1f}BPM | {intel.dominant_note:<2s} | [{intel.section_name}] | {self.active_preset_name[:14]}"
            meta_w = imgui.calc_text_size(meta).x + 96.0
            title_w = imgui.calc_text_size(title).x + 24.0
            pw = max(340.0, min(disp_w - 40.0, max(meta_w, title_w)))
            ph = 58.0
            px0 = img_pos.x + 20.0
            py0 = img_pos.y + disp_h - ph - 20.0
            p0 = imgui.ImVec2(px0, py0)
            p1 = imgui.ImVec2(px0 + pw, py0 + ph)
            dl.add_rect_filled(p0, p1, imgui.get_color_u32(imgui.ImVec4(*self.theme.bg[:3], 0.78)))
            draw_specimen_box(
                dl,
                p0,
                p1,
                self.theme,
                with_nodes=True,
                intensity=0.88,
                anim_time=app_time * 1.2,
                kick_pulse=intel.kick,
                drop_glow=intel.drop_intensity,
                num_pulses=1,
            )
            dl.add_text(imgui.ImVec2(px0 + 10, py0 + 8), imgui.get_color_u32(imgui.ImVec4(*self.theme.node_core)), title)
            draw_beat_sequencer(dl, imgui.ImVec2(px0 + 10, py0 + 32), intel.beat_index, intel.beat_phase, intel.kick, self.theme)
            dl.add_text(imgui.ImVec2(px0 + 82, py0 + 32), imgui.get_color_u32(imgui.ImVec4(*self.theme.line_core)), meta)

        # 2. Smoothly auto-hiding Top Hover Dock (slides down when mouse moves near top 68px!)
        mouse_near_top = io.mouse_pos.y < (img_pos.y + 68.0)
        target_hover = 1.0 if mouse_near_top else 0.0
        dt = max(io.delta_time, 1e-4)
        self._hover_dock_anim += (target_hover - self._hover_dock_anim) * min(1.0, dt * 10.0)

        if self._hover_dock_anim > 0.02:
            dock_w = min(disp_w - 24.0, 1220.0)
            dock_h = 40.0
            slide_y = -dock_h + self._hover_dock_anim * (dock_h + 12.0)
            imgui.set_next_window_pos(imgui.ImVec2(img_pos.x + (disp_w - dock_w) * 0.5, img_pos.y + slide_y))
            imgui.set_next_window_size(imgui.ImVec2(dock_w, dock_h))
            imgui.set_next_window_bg_alpha(0.86 * self._hover_dock_anim)
            flags = (
                imgui.WindowFlags_.no_decoration
                | imgui.WindowFlags_.no_move
                | imgui.WindowFlags_.no_saved_settings
                | imgui.WindowFlags_.no_scrollbar
            )
            imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(8.0, 6.0))
            if imgui.begin("##CinemaHoverDock", None, flags):
                w0 = imgui.get_window_pos()
                w1 = imgui.ImVec2(w0.x + dock_w, w0.y + dock_h)
                draw_specimen_box(imgui.get_window_draw_list(), w0, w1, self.theme, with_nodes=True, intensity=0.95)

                if imgui.button("SALIR CINE [ESC]"):
                    self.immersive_mode = False
                    self._set_fullscreen(False)
                imgui.same_line(0, 8)
                if imgui.button("< PRESET"):
                    self._cycle_preset(-1)
                imgui.same_line(0, 3)
                if imgui.button("PRESET >"):
                    self._cycle_preset(1)
                imgui.same_line(0, 8)
                if imgui.button("AI MODE >"):
                    self.audio.routing.profile_idx = (self.audio.routing.profile_idx + 1) % len(CHOREOGRAPHY_PROFILES)
                    self._notify(CHOREOGRAPHY_PROFILES[self.audio.routing.profile_idx])
                imgui.same_line(0, 8)
                if imgui.button("|<"):
                    self.spotify.prev_track()
                imgui.same_line(0, 3)
                if imgui.button("||" if now_playing.is_playing else ">"):
                    self.spotify.toggle_play_pause()
                imgui.same_line(0, 3)
                if imgui.button(">|"):
                    self.spotify.next_track()
                imgui.same_line(0, 8)
                if imgui.button("DROP![D]"):
                    self.audio.trigger_manual_drop()
                imgui.same_line(0, 8)
                if imgui.button("BLOBS[X]"):
                    self.blob_tracker.settings.enabled = not self.blob_tracker.settings.enabled
                imgui.same_line(0, 3)
                if imgui.button("STYLE[B]"):
                    self.blob_tracker.settings.visual_style = (self.blob_tracker.settings.visual_style + 1) % len(BLOB_VISUAL_STYLES)
                    self._notify(BLOB_VISUAL_STYLES[self.blob_tracker.settings.visual_style])
                imgui.same_line(0, 3)
                if imgui.button("MOTION[M]"):
                    self.morpher.cancel()
                    self.params.disp_mode = (int(self.params.disp_mode) + 1) % len(DISPLACEMENT_MODE_NAMES)
                    if self.params.disp_mode > 0 and self.params.disp_strength < 0.25:
                        self.params.disp_strength = 0.65
                    self._notify(DISPLACEMENT_MODE_NAMES[self.params.disp_mode])
                imgui.same_line(0, 8)
                for idx, th in enumerate(SPECIMEN_THEMES):
                    if imgui.button(th.label):
                        self._set_theme(idx)
                    imgui.same_line(0, 3)
                imgui.same_line(0, 6)
                if imgui.button("HUD [TAB]"):
                    self.immersive_mode = False
                    self.show_inspector = True
            imgui.end()
            imgui.pop_style_var()

    def _draw_viewport_and_timeline(self, avail_w: float, avail_h: float) -> None:
        app_time = time.perf_counter() - self._start_time
        intel = self.audio.intel

        if self.immersive_mode:
            vp_h = avail_h
            timeline_h = 0.0
        else:
            timeline_h = 68.0 if self.media.info.source_type == "video" else 36.0
            vp_h = max(140.0, avail_h - timeline_h - 8.0)

        imgui.begin_child(
            "ViewportPane",
            imgui.ImVec2(avail_w, vp_h),
            imgui.ChildFlags_.none,
            imgui.WindowFlags_.no_scrollbar | imgui.WindowFlags_.no_scroll_with_mouse,
        )
        win_min = imgui.get_window_pos()
        win_size = imgui.get_window_size()

        pane_avail = imgui.get_content_region_avail()
        pw = max(1.0, pane_avail.x)
        ph = max(1.0, pane_avail.y)
        tex_w = float(self.pipeline.width if self.pipeline else 1280)
        tex_h = float(self.pipeline.height if self.pipeline else 720)

        # In Immersive Cinema mode, automatically fill the entire monitor edge-to-edge (or fit cleanly)!
        if self.fit_viewport_fill or self.immersive_mode:
            scale = max(pw / tex_w, ph / tex_h) if self.immersive_mode else 1.0
            if self.immersive_mode:
                disp_w, disp_h = pw, ph
            else:
                disp_w, disp_h = pw, ph
        else:
            scale = min(pw / tex_w, ph / tex_h)
            disp_w = tex_w * scale
            disp_h = tex_h * scale

        offset_x = max(0.0, (pw - disp_w) * 0.5)
        offset_y = max(0.0, (ph - disp_h) * 0.5)
        imgui.set_cursor_pos(imgui.ImVec2(imgui.get_cursor_pos_x() + offset_x, imgui.get_cursor_pos_y() + offset_y))

        img_pos = imgui.get_cursor_screen_pos()

        if self._tex_final_glo > 0:
            imgui.image(
                imgui.ImTextureRef(self._tex_final_glo),
                imgui.ImVec2(disp_w, disp_h),
                imgui.ImVec2(0.0, 1.0),
                imgui.ImVec2(1.0, 0.0),
            )

        imgui.set_cursor_screen_pos(img_pos)
        imgui.invisible_button("##CanvasInteract", imgui.ImVec2(disp_w, disp_h))
        hovered = imgui.is_item_hovered()

        io = imgui.get_io()
        mx, my = io.mouse_pos.x, io.mouse_pos.y
        screen_u = (mx - img_pos.x) / max(disp_w, 1.0)
        screen_v = 1.0 - (my - img_pos.y) / max(disp_h, 1.0)

        canvas_u = (screen_u - 0.5) / max(self.interaction.zoom, 0.25) + 0.5 + self.interaction.pan_x
        canvas_v = (screen_v - 0.5) / max(self.interaction.zoom, 0.25) + 0.5 + self.interaction.pan_y

        self.interaction.prev_mouse_uv = list(self.interaction.mouse_uv)
        self.interaction.mouse_active = hovered and not self.exporter.active

        if hovered and not self.exporter.active:
            self.interaction.mouse_uv = [float(np.clip(canvas_u, 0.0, 1.0)), float(np.clip(canvas_v, 0.0, 1.0))]
            left_down = imgui.is_mouse_down(imgui.MouseButton_.left)
            left_clicked = imgui.is_mouse_clicked(imgui.MouseButton_.left)
            self.interaction.mouse_down = left_down

            if self.interaction.split_enabled:
                split_px_dist = abs((screen_u - self.interaction.split_pos) * disp_w)
                if split_px_dist < 14.0 or self.interaction.dragging_split:
                    imgui.set_mouse_cursor(imgui.MouseCursor_.resize_ew)
                    if left_clicked:
                        self.interaction.dragging_split = True

            if self.interaction.dragging_split:
                if left_down:
                    self.interaction.split_pos = float(np.clip(screen_u, 0.02, 0.98))
                else:
                    self.interaction.dragging_split = False
            else:
                if left_clicked and self.interaction.mode == 1:
                    self.interaction.spawn_ripple(self.interaction.mouse_uv[0], self.interaction.mouse_uv[1], strength=1.15)
                if imgui.is_mouse_double_clicked(imgui.MouseButton_.right) or (
                    imgui.is_mouse_double_clicked(imgui.MouseButton_.middle)
                ):
                    self.interaction.reset_view()

            wheel = io.mouse_wheel
            if abs(wheel) > 0.001:
                self.interaction.zoom = float(np.clip(self.interaction.zoom * (1.12 ** wheel), 0.5, 6.0))

            if imgui.is_mouse_dragging(imgui.MouseButton_.right) or imgui.is_mouse_dragging(imgui.MouseButton_.middle):
                dx = io.mouse_delta.x / max(disp_w * self.interaction.zoom, 1.0)
                dy = io.mouse_delta.y / max(disp_h * self.interaction.zoom, 1.0)
                self.interaction.pan_x = float(np.clip(self.interaction.pan_x - dx, -0.6, 0.6))
                self.interaction.pan_y = float(np.clip(self.interaction.pan_y + dy, -0.6, 0.6))
        else:
            if not imgui.is_mouse_down(imgui.MouseButton_.left):
                self.interaction.dragging_split = False
                self.interaction.mouse_down = False

        dl = imgui.get_window_draw_list()

        # Intelligent TouchDesigner-style Computer Vision Blobs & Constellation Plexus!
        if self.blob_tracker.settings.enabled and not self.exporter.active:
            draw_intelligent_blobs(
                dl=dl,
                img_pos=img_pos,
                disp_w=disp_w,
                disp_h=disp_h,
                theme=self.theme,
                tracks=self.blob_tracker.tracks,
                plexus_edges=self.blob_tracker.plexus_edges,
                settings=self.blob_tracker.settings,
                time_sec=app_time,
                audio_kick=intel.kick,
                audio_snare=intel.snare,
                zoom=self.interaction.zoom,
                pan_x=self.interaction.pan_x,
                pan_y=self.interaction.pan_y,
            )

        # ART-VIDEO-MUSIC Editorial Typography Overlay Layer (Spotify Auto or Custom Artist Title!)
        if not self.exporter.active and self.mixer.art_overlay_style > 0:
            np_info = self.spotify.get_info()
            art_title = (
                (np_info.title or self.mixer.art_custom_title)
                if self.mixer.art_use_spotify
                else self.mixer.art_custom_title
            )
            art_sub = (
                (np_info.artist or self.mixer.art_custom_sub)
                if self.mixer.art_use_spotify
                else self.mixer.art_custom_sub
            )
            draw_art_typography_overlay(
                dl=dl,
                img_pos=img_pos,
                disp_w=disp_w,
                disp_h=disp_h,
                theme=self.theme,
                style=self.mixer.art_overlay_style,
                title=art_title,
                subtitle=art_sub,
                bpm=intel.bpm,
                note=intel.dominant_note,
                section_name=intel.section_name,
                time_sec=app_time,
                kick_pulse=intel.spring_kick,
                opacity=self.mixer.art_overlay_opacity,
            )

        # Animated Holographic // DROP DETECTED // Alert Banner
        if not self.exporter.active and intel.drop_intensity > 0.10:
            draw_drop_alert_banner(
                dl,
                img_pos,
                disp_w,
                disp_h,
                self.theme,
                intel.drop_intensity,
                intel.bpm,
                intel.dominant_note,
            )

        # Split-Screen badges
        if self.interaction.split_enabled:
            sx = img_pos.x + self.interaction.split_pos * disp_w
            left_label = SPLIT_MODES[self.interaction.split_mode].upper()
            ls = imgui.calc_text_size(left_label)
            p0 = imgui.ImVec2(img_pos.x + 12, img_pos.y + 12)
            p1 = imgui.ImVec2(img_pos.x + 12 + ls.x + 18, img_pos.y + 34)
            dl.add_rect_filled(p0, p1, imgui.get_color_u32(imgui.ImVec4(*self.theme.bg[:3], 0.85)))
            draw_specimen_box(dl, p0, p1, self.theme, with_nodes=False, intensity=0.8)
            dl.add_text(imgui.ImVec2(p0.x + 9, p0.y + 4), imgui.get_color_u32(imgui.ImVec4(*self.theme.node_core)), left_label)

            right_label = "ASCII // HDR BLOOM"
            rs = imgui.calc_text_size(right_label)
            q0 = imgui.ImVec2(img_pos.x + disp_w - rs.x - 30, img_pos.y + 12)
            q1 = imgui.ImVec2(img_pos.x + disp_w - 12, img_pos.y + 34)
            dl.add_rect_filled(q0, q1, imgui.get_color_u32(imgui.ImVec4(*self.theme.bg[:3], 0.85)))
            draw_specimen_box(dl, q0, q1, self.theme, with_nodes=False, intensity=0.8)
            dl.add_text(imgui.ImVec2(q0.x + 9, q0.y + 4), imgui.get_color_u32(imgui.ImVec4(*self.theme.line_core)), right_label)

            draw_glow_node(dl, imgui.ImVec2(sx, img_pos.y + disp_h * 0.5), self.theme, 5.0, intel.kick * 0.5)

        # Toast notification
        now = time.perf_counter()
        if now < self.toast_until and self.toast_message:
            tsize = imgui.calc_text_size(self.toast_message)
            tw = min(disp_w - 28.0, tsize.x + 24.0)
            th = tsize.y + 10.0
            tx = img_pos.x + (disp_w - tw) * 0.5
            ty = img_pos.y + disp_h - th - (82.0 if (self.immersive_mode and self.show_immersive_pill) else 14.0)
            t0 = imgui.ImVec2(tx, ty)
            t1 = imgui.ImVec2(tx + tw, ty + th)
            dl.add_rect_filled(t0, t1, imgui.get_color_u32(imgui.ImVec4(*self.theme.bg[:3], 0.90)))
            draw_specimen_box(dl, t0, t1, self.theme, with_nodes=True, intensity=0.85)
            dl.add_text(imgui.ImVec2(tx + 12.0, ty + 5.0), imgui.get_color_u32(imgui.ImVec4(*self.theme.node_core)), self.toast_message)

        if self.pipeline and self.pipeline.last_reload_error:
            err_msg = "GLSL ERR // " + " | ".join(self.pipeline.last_reload_error.splitlines()[:2])[:140]
            e0 = imgui.ImVec2(img_pos.x + 12, img_pos.y + 42)
            e1 = imgui.ImVec2(img_pos.x + disp_w - 12, img_pos.y + 68)
            dl.add_rect_filled(e0, e1, imgui.get_color_u32(imgui.ImVec4(0.35, 0.04, 0.08, 0.92)))
            dl.add_text(imgui.ImVec2(e0.x + 8, e0.y + 5), imgui.get_color_u32(imgui.ImVec4(1.0, 0.72, 0.72, 1.0)), err_msg)

        if self.exporter.active:
            bw = min(460.0, disp_w - 40.0)
            bx = img_pos.x + (disp_w - bw) * 0.5
            by = img_pos.y + 18.0
            b0 = imgui.ImVec2(bx, by)
            b1 = imgui.ImVec2(bx + bw, by + 44.0)
            dl.add_rect_filled(b0, b1, imgui.get_color_u32(imgui.ImVec4(*self.theme.bg[:3], 0.94)))
            draw_specimen_box(dl, b0, b1, self.theme, with_nodes=True)
            pct = self.exporter.progress
            dl.add_rect_filled(
                imgui.ImVec2(bx + 8, by + 28),
                imgui.ImVec2(bx + 8 + (bw - 16) * pct, by + 36),
                imgui.get_color_u32(imgui.ImVec4(*self.theme.line_core)),
            )
            dl.add_text(
                imgui.ImVec2(bx + 12, by + 6),
                imgui.get_color_u32(imgui.ImVec4(*self.theme.node_core)),
                f"FFMPEG RENDER // {int(pct * 100)}% ({self.exporter.current_frame}/{self.exporter.total_frames})",
            )

        # Animated Glowing Specimen Outer Frame around Viewport (subtle thin halo in Cinema mode!)
        draw_specimen_box(
            dl,
            imgui.ImVec2(win_min.x + 1, win_min.y + 1),
            imgui.ImVec2(win_min.x + win_size.x - 1, win_min.y + win_size.y - 1),
            self.theme,
            with_nodes=not self.immersive_mode,
            intensity=0.55 if self.immersive_mode else 1.0,
            anim_time=app_time,
            kick_pulse=intel.kick,
            drop_glow=intel.drop_intensity,
            num_pulses=2,
        )

        # Immersive Fullscreen Floating Pill + Auto-Hiding Hover Dock
        if self.immersive_mode:
            self._draw_immersive_floating_overlays(dl, img_pos, disp_w, disp_h, app_time)

        imgui.end_child()

        if self.immersive_mode:
            return

        # --- Bottom Specimen Timeline / Status Bar ---
        imgui.begin_child("TimelineBar", imgui.ImVec2(avail_w, timeline_h), imgui.ChildFlags_.none, imgui.WindowFlags_.no_scrollbar)
        t_min = imgui.get_window_pos()
        t_size = imgui.get_window_size()
        t_dl = imgui.get_window_draw_list()

        if self.exporter.active:
            imgui.progress_bar(self.exporter.progress, imgui.ImVec2(max(120.0, avail_w - 150.0), 20.0))
            imgui.same_line()
            if imgui.button("CANCELAR REC"):
                self.exporter.cancel()
        elif self.media.info.source_type == "video":
            if imgui.button("PAUSA" if self.media.playing else "PLAY ", imgui.ImVec2(58, 0)):
                self.media.playing = not self.media.playing
            imgui.same_line(0, 4)
            if imgui.button("<", imgui.ImVec2(24, 0)):
                self.media.step_frame(-1)
            imgui.same_line(0, 2)
            if imgui.button(">", imgui.ImVec2(24, 0)):
                self.media.step_frame(1)
            imgui.same_line(0, 10)
            imgui.text_colored(imgui.ImVec4(*self.theme.node_core), f"{_fmt_time(self.media.current_time)} / {_fmt_time(self.media.info.duration)}")
            imgui.same_line(0, 10)

            slider_w = max(120.0, imgui.get_content_region_avail().x - 10.0)
            imgui.set_next_item_width(slider_w)
            changed, new_t = imgui.slider_float(
                "##TimelineScrub",
                float(self.media.current_time),
                0.0,
                max(float(self.media.info.duration), 0.1),
                "",
            )
            if changed:
                self.media.seek(new_t)

            imgui.set_next_item_width(90)
            _, self.media.speed = imgui.slider_float("VEL", float(self.media.speed), 0.1, 3.0, "%.2fx")
            imgui.same_line(0, 10)
            _, self.media.loop = imgui.checkbox("LOOP", self.media.loop)
            imgui.same_line(0, 10)
            if imgui.button("SET A"):
                self.media.loop_a = self.media.current_time
                if self.media.loop_b <= self.media.loop_a:
                    self.media.loop_b = self.media.info.duration
            imgui.same_line(0, 4)
            if imgui.button("SET B"):
                self.media.loop_b = max(self.media.current_time, self.media.loop_a + 0.1)
                self.media.ab_loop_enabled = True
            imgui.same_line(0, 6)
            _, self.media.ab_loop_enabled = imgui.checkbox(
                f"A-B [{_fmt_time(self.media.loop_a)}-{_fmt_time(self.media.loop_b)}]",
                self.media.ab_loop_enabled,
            )
            imgui.same_line(0, 12)
            _, self.audio.muted = imgui.checkbox("MUTE", self.audio.muted)
        else:
            if imgui.button("PAUSA" if self.media.playing else "PLAY ", imgui.ImVec2(58, 0)):
                self.media.playing = not self.media.playing
            imgui.same_line(0, 12)
            grid_cols = int((self.pipeline.width if self.pipeline else 1280) / max(self.params.cell_w, 4.0))
            grid_rows = int((self.pipeline.height if self.pipeline else 720) / max(self.params.cell_h, 4.0))
            mode_tag = RENDER_MODE_NAMES[self.params.render_mode].split()[0].upper()
            imgui.text_colored(
                imgui.ImVec4(*self.theme.text_main),
                f"//{mode_tag}_ {self.media.info.name.upper()[:22]} | {grid_cols}x{grid_rows} | {intel.bpm:.0f}BPM [{intel.section_name}] | [F] CINE [F11] FULL",
            )

        draw_specimen_box(
            t_dl,
            imgui.ImVec2(t_min.x + 1, t_min.y + 1),
            imgui.ImVec2(t_min.x + t_size.x - 1, t_min.y + t_size.y - 1),
            self.theme,
            with_nodes=True,
            intensity=0.85,
            kick_pulse=intel.kick * 0.5,
        )
        imgui.end_child()

    def _apply_tab_glide(self, tab_id: str) -> None:
        """Triggers a silky 14px vertical glide-in displacement whenever switching between HUD tabs!"""
        if self._active_tab_id != tab_id:
            self._active_tab_id = tab_id
            self._tab_glide_y = 14.0
        if self._tab_glide_y > 0.3:
            imgui.dummy(imgui.ImVec2(0.0, self._tab_glide_y))

    def _draw_guide_content(self, is_modal: bool = False) -> None:
        """Interactive in-app Spanish manual with live 1-click demo triggers!"""
        c_hi = imgui.ImVec4(*self.theme.node_core)
        c_acc = imgui.ImVec4(*self.theme.line_core)
        c_txt = imgui.ImVec4(*self.theme.text_main)

        imgui.push_text_wrap_pos(0.0)
        imgui.text_colored(c_hi, "SPECIMEN N001 // GUIA RAPIDA INTERACTIVA")
        imgui.text_colored(c_txt, "Estudio visual GPU HDR en tiempo real con ASCII, Stipple CRT, 11 animaciones de desplazamiento, Blobs CV inteligentes e Inteligencia Musical.")
        imgui.spacing()

        if imgui.collapsing_header("0. QUE ES ESTA PLATAFORMA? (CLASE VJ / INSTRUMENTO AUDIOVISUAL)", imgui.TreeNodeFlags_.default_open):
            imgui.text_colored(c_txt, "• Pertenece a la clase de Software VJ / Sintetizador Visual Audiovisual en Tiempo Real (en el linaje de Resolume, TouchDesigner, Synesthesia, VDMX y Cathodemer).")
            imgui.text_colored(c_txt, "• Fusiona ARTE GENERATIVO 3D + VIDEO EN VIVO (Mezclador Dual A/B + Banco de 8 Clips + Captura de Pantalla) + MUSICA (Spotify WASAPI + Física Spring + HPSS) en una sola plataforma.")
            if imgui.button("PROBAR MACRO LIQUID FUSION"):
                self._trigger_macro(1)
            imgui.same_line(0, 6)
            if imgui.button("PROBAR CAPA TIPOGRAFICA [T]"):
                self.mixer.art_overlay_style = (self.mixer.art_overlay_style % 4) + 1
                self._notify(f"ART OVERLAY // {ART_OVERLAY_STYLES[self.mixer.art_overlay_style]}")

        if imgui.collapsing_header("1. STUDIO A/B: MEZCLADOR VIDEO + ARTE, CLIPS Y PANTALLA LIVE"):
            imgui.text_colored(c_txt, "• Pestaña MIX: mezcla Deck A (tu video, imagen o Captura de Pantalla Live [L]) con Deck B (Arte 3D Generativo o un 2.º video) usando 10 Blend Modes (Additive, Screen, Multiply, Luma Key, Beat Strobe Cut [V]).")
            imgui.text_colored(c_txt, "• Banco de 8 Clips: arrastra varios videos o pulsa [+ AÑADIR CLIPS] y activa AUTO-CLIP EN DROPS para crear un director de videoclips autónomo.")
            imgui.text_colored(c_txt, "• Transformaciones Pro: Rotación, Escala, Flip, Simetría 4x, Encuadre (Contain/Cover), Pre-Grade y Macros S1–S6 que guardan todo tu setup en 1 clic.")

        if imgui.collapsing_header("2. INICIO RAPIDO (VIDEOS, FOTOS Y 3D)"):
            imgui.text_colored(c_txt, "• Arrastra cualquier video (.mp4, .mov, .webm, .gif), imagen o fuente (.ttf/.otf) directo a la ventana, o pulsa [+ MEDIA].")
            imgui.text_colored(c_txt, "• Usa el selector superior para probar las 7 escenas 3D GPU integradas (incluyendo Audio Mandala EQ y 3D Heart).")
            imgui.text_colored(c_txt, "• El tamaño de píxeles/celdas se normaliza automáticamente a HD: nunca cambia brusco al cargar clips pequeños o cambiar 720p/1080p.")
            if imgui.button("PROBAR 3D AUDIO MANDALA"):
                self.media.set_procedural(2)
                self._notify("3D // AUDIO MANDALA EQ")
            imgui.same_line(0, 6)
            if imgui.button("MODO CINE 100% [F]"):
                self.show_guide_drawer = False
                self._toggle_immersive_cinema()

        if imgui.collapsing_header("2. AUDIO SPOTIFY / PC & FISICA SPRING", imgui.TreeNodeFlags_.default_open):
            imgui.text_colored(c_txt, "• Conecta directo a Spotify o cualquier sonido del PC vía Windows WASAPI Loopback (sin cables virtuales).")
            imgui.text_colored(c_txt, "• Si Spotify suena pero no ves barras, abre SPOTIFY/AUDIO y pulsa [AUTO-DETECTAR SALIDA].")
            imgui.text_colored(c_txt, "• Fisica Spring (2.º orden): los golpes de Kick, Bass y Drops mueven un sistema masa-resorte orgánico sin tirones robóticos.")
            imgui.text_colored(c_txt, "• HPSS en vivo: separa impactos percusivos (glitch/flashes) de energía armónica/vocal (desplazamiento líquido y brillo).")
            imgui.text_colored(c_txt, "• 10 Coreografías AI: AUTO-DIRECTOR AI cambia la reacción automáticamente entre AMBIENT, GROOVE, BUILD-UP y DROP!!.")
            if imgui.button("CONECTAR SPOTIFY / PC"):
                self.audio.set_mode(0)
                self.audio.auto_detect_active_loopback()
                self._notify("CONECTANDO WASAPI SPOTIFY / PC...")
            imgui.same_line(0, 6)
            if imgui.button("PROBAR DROP! [D]"):
                self.audio.trigger_manual_drop()

        if imgui.collapsing_header("3. ANIMACIONES DE DESPLAZAMIENTO GPU (11 MODOS)", imgui.TreeNodeFlags_.default_open):
            imgui.text_colored(c_txt, "• En la pestaña MOTION (o pulsando la tecla [M]) eliges entre 11 deformaciones espaciales GPU sincronizadas al beat:")
            imgui.text_colored(c_acc, "  1. Liquid Curl Flow   2. Acoustic Shockwave   3. Silk Wave Drift")
            imgui.text_colored(c_acc, "  4. Slit-Scan Echo     5. 3D Depth Parallax    6. Cyber Glitch Blocks")
            imgui.text_colored(c_acc, "  7. Vortex Singularity 8. Prism RGB Split      9. Kaleido Mirror")
            imgui.text_colored(c_acc, " 10. Elastic Bounce    11. Pixel Sort Cascade")
            if imgui.button("PROBAR LIQUID CURL"):
                self.morpher.cancel()
                self.params.disp_mode = 1
                self.params.disp_strength = 0.85
                self._notify("DESPLAZAMIENTO // LIQUID CURL FLOW")
            imgui.same_line(0, 6)
            if imgui.button("PROBAR KALEIDO PRISM"):
                self.morpher.cancel()
                self.params.disp_mode = 9
                self.params.disp_strength = 0.85
                self._notify("DESPLAZAMIENTO // KALEIDO MIRROR")
            imgui.same_line(0, 6)
            if imgui.button("APAGAR DESPLAZ."):
                self.morpher.cancel()
                self.params.disp_mode = 0
                self._notify("DESPLAZAMIENTO // OFF")

        if imgui.collapsing_header("4. BLOBS INTELIGENTES CV (5 ESTILOS)", imgui.TreeNodeFlags_.default_open):
            imgui.text_colored(c_txt, "• Motor real de Computer Vision con Kalman + Algoritmo Húngaro + partición de picos Voronoi y red constelación Plexus.")
            imgui.text_colored(c_txt, "• Pulsa [X] para activar/ocultar y [B] para alternar entre TouchDesigner Plexus, Swiss Minimal, Organic Contour, Constellation Nodes y Cyber Pro.")
            if imgui.button("CAMBIAR ESTILO BLOB [B]"):
                self.blob_tracker.settings.enabled = True
                self.blob_tracker.settings.visual_style = (self.blob_tracker.settings.visual_style + 1) % len(BLOB_VISUAL_STYLES)
                self._notify(f"BLOB STYLE // {BLOB_VISUAL_STYLES[self.blob_tracker.settings.visual_style]}")

        if imgui.collapsing_header("5. PRESETS, PIXELES, INTERACCION & EXPORTACION"):
            imgui.text_colored(c_txt, "• 24 Presets curados filtrables por categoría (SPECIMEN, DESPLAZ., CYBER, RAVE, RETRO, ARTE) con transición Morph suave.")
            imgui.text_colored(c_txt, "• Al mover cualquier slider de tamaño de píxel o celda, el Morph se cancela al instante para respuesta 100% inmediata.")
            imgui.text_colored(c_txt, "• Canvas interactivo: teclas [1]-[5] activan Ondas, Aura, Repulsión, Vórtice o Pincel Neón HDR. Rueda = Zoom, Clic Der. = Pan.")
            imgui.text_colored(c_txt, "• Exportación profesional: captura PNG al instante o renderiza MP4 (con audio original), GIF HD o WebM vía FFmpeg.")

        if imgui.collapsing_header("6. ATAJOS DE TECLADO COMPLETOS", imgui.TreeNodeFlags_.default_open):
            shortcuts = [
                ("F1", "Abrir / Cerrar esta Guía Interactiva"),
                ("F / F11 / ESC", "Modo Cine 100% / Pantalla Completa / Salir"),
                ("M / V / T / L", "Desplazamiento / Blend A-B / Capa Arte / Pantalla Live"),
                ("P", "Siguiente Preset con transición Morph"),
                ("X / B", "Blobs Inteligentes ON-OFF / Cambiar Estilo (5)"),
                ("D", "Disparar explosión manual DROP!!"),
                ("TAB / H", "Deslizar Dossier HUD / Mini-HUD Spotify en Cine"),
                ("ESPACIO", "Play / Pausa de video o animación 3D"),
                ("S / R / C", "Split-Screen / Reset Vista / Limpiar Estelas y Pintura"),
                ("1 - 6", "Modo cursor (Ondas, Aura, Repulsión, Vórtice, Pincel, Off)"),
                ("F5", "Hot-Reload en vivo de shaders GLSL"),
            ]
            for key_lbl, desc in shortcuts:
                imgui.text_colored(c_hi, f"  [{key_lbl:<13s}]")
                imgui.same_line(0, 6)
                imgui.text_colored(c_txt, desc)

        if is_modal:
            imgui.spacing()
            if imgui.button("CERRAR GUIA [ESC / F1]", imgui.ImVec2(-1, 28)):
                self.show_guide_drawer = False
        imgui.pop_text_wrap_pos()

    def _draw_guide_drawer_overlay(self) -> None:
        """Smoothly sliding animated Specimen Guide overlay triggered via [F1] or ?GUIA."""
        if self._guide_slide <= 0.02:
            return
        vp = imgui.get_main_viewport()
        gw = min(640.0, vp.size.x - 40.0)
        gh = min(640.0, vp.size.y - 60.0)
        # Smooth ease-out slide displacement from right edge!
        ease = 1.0 - (1.0 - self._guide_slide) ** 3
        target_x = vp.pos.x + (vp.size.x - gw) * 0.5
        start_x = vp.pos.x + vp.size.x + 20.0
        cur_x = start_x + (target_x - start_x) * ease
        cur_y = vp.pos.y + (vp.size.y - gh) * 0.5

        imgui.set_next_window_pos(imgui.ImVec2(cur_x, cur_y))
        imgui.set_next_window_size(imgui.ImVec2(gw, gh))
        imgui.set_next_window_bg_alpha(0.94 * ease)
        flags = (
            imgui.WindowFlags_.no_decoration
            | imgui.WindowFlags_.no_move
            | imgui.WindowFlags_.no_saved_settings
        )
        imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(16.0, 14.0))
        if imgui.begin("##SpecimenGuideDrawer", None, flags):
            w0 = imgui.get_window_pos()
            w1 = imgui.ImVec2(w0.x + gw, w0.y + gh)
            dl = imgui.get_window_draw_list()
            app_time = time.perf_counter() - self._start_time
            draw_hud_scanbeam(dl, w0, w1, self.theme, app_time, speed=0.18, alpha=0.16)
            draw_specimen_box(dl, w0, w1, self.theme, with_nodes=True, intensity=1.1, anim_time=app_time, num_pulses=2)

            # Pinned header bar with instant Close button so user never needs to scroll to dismiss!
            imgui.push_font(None, 16.0)
            imgui.text_colored(imgui.ImVec4(*self.theme.node_core), "SPECIMEN N001 // GUIA INTERACTIVA [F1]")
            imgui.pop_font()
            imgui.same_line(max(240.0, gw - 140.0))
            if imgui.button("CERRAR [ESC]", imgui.ImVec2(114, 22)):
                self.show_guide_drawer = False
            imgui.separator()
            imgui.begin_child("##GuideScrollInner", imgui.ImVec2(-1, -1), imgui.ChildFlags_.none)
            self._draw_guide_content(is_modal=False)
            imgui.end_child()
        imgui.end()
        imgui.pop_style_var()

    def _draw_specimen_dossier_inspector(self, panel_w: float, avail_h: float) -> None:
        imgui.begin_child("SpecimenDossier", imgui.ImVec2(panel_w, avail_h), imgui.ChildFlags_.none, imgui.WindowFlags_.no_scrollbar)
        d_min = imgui.get_window_pos()
        d_size = imgui.get_window_size()
        dl = imgui.get_window_draw_list()
        app_time = time.perf_counter() - self._start_time
        now_playing = self.spotify.get_info()
        intel = self.audio.intel

        # Subtle animated CRT scan-beam gliding across the entire Dossier!
        draw_hud_scanbeam(dl, d_min, imgui.ImVec2(d_min.x + d_size.x, d_min.y + d_size.y), self.theme, app_time, speed=0.14, alpha=0.18)

        inner_w = max(120.0, imgui.get_content_region_avail().x)
        scope_panel_h = 106.0

        # ==================== TOP TWIN SPECIMEN MODULE (3D WIREFRAME + SPOTIFY LIVE DECK) ====================
        twin_h = 188.0
        left_w = 136.0
        right_w = max(120.0, inner_w - left_w - 6.0)

        p_scope_min = imgui.get_cursor_screen_pos()
        p_scope_max = imgui.ImVec2(p_scope_min.x + left_w, p_scope_min.y + twin_h)
        imgui.invisible_button("##Mini3DScopeBtn", imgui.ImVec2(left_w, twin_h))
        if imgui.is_item_clicked():
            self.mini_3d_shape = (self.mini_3d_shape + 1) % 4
        draw_wireframe_3d_scope(
            dl,
            p_scope_min,
            p_scope_max,
            self.theme,
            app_time,
            intel.kick,
            intel.vocals,
            self.mini_3d_shape,
        )
        draw_specimen_box(dl, p_scope_min, p_scope_max, self.theme, with_nodes=True, intensity=0.92, kick_pulse=intel.kick * 0.6)
        dl.add_text(
            imgui.ImVec2(p_scope_min.x + 8, p_scope_min.y + 5),
            imgui.get_color_u32(imgui.ImVec4(*self.theme.text_dim)),
            ["3D:HEART", "3D:ORB", "3D:TORUS", "3D:DNA"][self.mini_3d_shape],
        )

        # Right cell: Live Spotify / Windows Audio Deck + Animated Marquee + Beat LEDs + TAP + Stars!
        imgui.same_line(0, 6)
        imgui.push_style_var(imgui.StyleVar_.window_padding, imgui.ImVec2(8.0, 7.0))
        imgui.begin_child("SpotifyDeckCell", imgui.ImVec2(right_w, twin_h), imgui.ChildFlags_.always_use_window_padding, imgui.WindowFlags_.no_scrollbar)
        c_min = imgui.get_window_pos()
        c_size = imgui.get_window_size()
        c_dl = imgui.get_window_draw_list()

        src_tag = "SPOTIFY LIVE" if now_playing.is_spotify else (now_playing.app_name if now_playing.connected else "PC WASAPI")
        blink = "●" if (intel.section_id > 0 and (int(app_time * 3.0) % 2 == 0 or intel.kick > 0.3)) else "○"
        imgui.text_colored(imgui.ImVec4(*self.theme.line_core), f"//AUDIO_ {blink} {src_tag}")

        # Smooth animated marquee for long Spotify track titles!
        imgui.push_font(None, 17.0)
        raw_track = (now_playing.title or "AUDIO DEL SISTEMA").upper()
        track_disp = marquee_text(raw_track, 20, app_time, speed=3.5)
        imgui.text_colored(imgui.ImVec4(*self.theme.node_core), track_disp)
        imgui.pop_font()

        raw_artist = (now_playing.artist or "REPRODUCE SPOTIFY / PC").upper()
        artist_disp = marquee_text(raw_artist, 25, app_time + 1.5, speed=3.0)
        imgui.text_colored(imgui.ImVec4(*self.theme.text_dim), f"//ARTIST_ {artist_disp}")

        if imgui.button("|<", imgui.ImVec2(32, 22)):
            self.spotify.prev_track()
        imgui.same_line(0, 4)
        play_lbl = "||" if now_playing.is_playing else " >"
        if imgui.button(play_lbl, imgui.ImVec2(36, 22)):
            self.spotify.toggle_play_pause()
        imgui.same_line(0, 4)
        if imgui.button(">|", imgui.ImVec2(32, 22)):
            self.spotify.next_track()
        imgui.same_line(0, 8)
        wasapi_on = self.audio.mode == 0
        if wasapi_on:
            imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(*self.theme.frame_active))
            imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*self.theme.node_core))
        if imgui.button("PC/SPOTIFY" if wasapi_on else "CONECTAR PC", imgui.ImVec2(-1, 22)):
            self.audio.set_mode(0)
            self.audio.auto_detect_active_loopback()
            self._notify("BUSCANDO SALIDA ACTIVA DE SPOTIFY / PC...")
        if wasapi_on:
            imgui.pop_style_color(2)

        # Glowing 5-Star Audio Energy Row + Live Musical Section Badge
        imgui.spacing()
        imgui.text_colored(imgui.ImVec4(*self.theme.line_core), "//ENERGY_")
        imgui.same_line(0, 8)
        star_pos = imgui.get_cursor_screen_pos()
        lit_stars = min(5.0, max(0.8 if intel.section_id > 0 else 0.0, self.audio.energy * 5.6))
        draw_glow_stars(c_dl, imgui.ImVec2(star_pos.x, star_pos.y + 1), 5, lit_stars, self.theme, radius=5.6, spacing=15.5, time_sec=app_time)
        imgui.same_line(0, 86)
        sec_col = (
            imgui.ImVec4(1.0, 0.42, 0.78, 1.0)
            if intel.section_id == 4
            else (imgui.ImVec4(1.0, 0.85, 0.35, 1.0) if intel.section_id == 3 else imgui.ImVec4(*self.theme.node_core))
        )
        imgui.text_colored(sec_col, f"[{intel.section_name}]")

        # BPM + 4-Beat Sequencer LEDs + Note + TAP button
        imgui.text_colored(imgui.ImVec4(*self.theme.node_core), f"{intel.bpm:5.1f}BPM")
        imgui.same_line(0, 6)
        seq_pos = imgui.get_cursor_screen_pos()
        draw_beat_sequencer(c_dl, seq_pos, intel.beat_index, intel.beat_phase, intel.kick, self.theme)
        imgui.dummy(imgui.ImVec2(66, 14))
        imgui.same_line(0, 4)
        imgui.text_colored(imgui.ImVec4(*self.theme.line_core), f"N:{intel.dominant_note:<2s}")
        imgui.same_line(0, 4)
        if imgui.button("TAP", imgui.ImVec2(-1, 18)):
            self.audio.tap_tempo()

        draw_specimen_box(
            c_dl,
            imgui.ImVec2(c_min.x + 1, c_min.y + 1),
            imgui.ImVec2(c_min.x + c_size.x - 1, c_min.y + c_size.y - 1),
            self.theme,
            with_nodes=True,
            intensity=0.92,
            kick_pulse=intel.kick * 0.6,
            drop_glow=intel.drop_intensity,
        )
        imgui.end_child()
        imgui.pop_style_var()

        imgui.spacing()

        # ==================== SCROLLABLE MIDDLE CONTROLS REGION ====================
        scroll_h = max(140.0, imgui.get_content_region_avail().y - scope_panel_h - 8.0)
        imgui.begin_child("SpecimenControlsScroll", imgui.ImVec2(inner_w, scroll_h), imgui.ChildFlags_.none)
        imgui.push_item_width(max(110.0, inner_w * 0.52))

        total_presets = len(self.preset_manager.presets)
        if imgui.collapsing_header(f"//PRESETS_ ({total_presets} ESTILOS + MORPH)", imgui.TreeNodeFlags_.default_open):
            # Category filter chips for instant browsing across 24+ presets!
            for c_idx, cat in enumerate(PRESET_CATEGORY_ORDER):
                if c_idx > 0 and c_idx != 4:
                    imgui.same_line(0, 3)
                is_cat = c_idx == self.preset_cat_idx
                if is_cat:
                    imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(*self.theme.frame_active))
                    imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*self.theme.node_core))
                if imgui.button(cat):
                    self.preset_cat_idx = c_idx
                if is_cat:
                    imgui.pop_style_color(2)

            active_cat = PRESET_CATEGORY_ORDER[self.preset_cat_idx % len(PRESET_CATEGORY_ORDER)]
            preset_names = self.preset_manager.names_for_category(active_cat)

            # Compact scrollable 2-column preset grid so tabs stay immediately accessible
            grid_h = min(148.0, max(56.0, ((len(preset_names) + 1) // 2) * 26.0 + 8.0))
            imgui.begin_child("PresetGridScroll", imgui.ImVec2(-1, grid_h), imgui.ChildFlags_.none)
            btn_w = (imgui.get_content_region_avail().x - 6.0) * 0.5
            for idx, pname in enumerate(preset_names):
                if idx % 2 == 1:
                    imgui.same_line(0, 6)
                is_cur = pname == self.active_preset_name
                if is_cur:
                    imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(*self.theme.frame_active))
                    imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*self.theme.node_core))
                if imgui.button(pname, imgui.ImVec2(btn_w, 22)):
                    self._trigger_preset(pname)
                if is_cur:
                    imgui.pop_style_color(2)
            imgui.end_child()

            if self.morpher.active:
                prog = min(1.0, self.morpher.elapsed / max(self.morpher.duration, 0.01))
                imgui.progress_bar(prog, imgui.ImVec2(-1, 5), "")

            if imgui.button("< PREV"):
                self._cycle_preset(-1)
            imgui.same_line(0, 3)
            if imgui.button("NEXT >"):
                self._cycle_preset(1)
            imgui.same_line(0, 3)
            if imgui.button("RND"):
                all_p = list(self.preset_manager.presets.keys())
                if all_p:
                    pick = all_p[int(time.perf_counter() * 1000) % len(all_p)]
                    self._trigger_preset(pick)
            imgui.same_line(0, 6)
            imgui.set_next_item_width(82)
            _, self.morph_duration = imgui.slider_float("##MorphDur", float(self.morph_duration), 0.05, 2.5, "%.2fs")
            imgui.same_line(0, 4)
            imgui.set_next_item_width(max(65.0, imgui.get_content_region_avail().x - 66.0))
            _, self.save_preset_name = imgui.input_text("##PresetName", self.save_preset_name)
            imgui.same_line(0, 3)
            if imgui.button("SAVE"):
                out = self.preset_manager.save_preset(self.save_preset_name, self.params)
                self.active_preset_name = self.save_preset_name.strip() or out.stem
                self._notify(f"PRESET GUARDADO // {out.name}")

        imgui.spacing()

        if imgui.begin_tab_bar("SpecimenTabs"):
            # ==================== TAB 1: ENGINE ====================
            if imgui.begin_tab_item_simple("ENG"):
                self._apply_tab_glide("ENGINE")
                imgui.spacing()
                rm_ch, self.params.render_mode = imgui.combo("//RENDER_", int(self.params.render_mode), RENDER_MODE_NAMES)
                self._cancel_morph_if(rm_ch)

                if self.params.render_mode in (1, 2):
                    imgui.separator_text("//STIPPLE_ POINT-CLOUD CRT")
                    ch1, self.params.stipple_scale = imgui.slider_float("TAMAÑO PUNTO", float(self.params.stipple_scale), 1.25, 8.0, "%.2fpx")
                    ch2, self.params.stipple_density = imgui.slider_float("DENSIDAD STIPPLE", float(self.params.stipple_density), 0.4, 2.2, "%.2f")
                    self._cancel_morph_if(ch1 or ch2)

                if self.params.render_mode in (0, 2):
                    imgui.separator_text("//GLYPHS_ ATLAS & GRILLA")
                    cur_cs_idx = CHARSET_ORDER.index(self.params.charset_id) if self.params.charset_id in CHARSET_ORDER else 0
                    cs_labels = [CHARSET_LABELS[k] for k in CHARSET_ORDER]
                    changed, new_cs_idx = imgui.combo("SET GLIFOS", cur_cs_idx, cs_labels)
                    if changed:
                        self.morpher.cancel()
                        self.params.charset_id = CHARSET_ORDER[new_cs_idx]
                        self._rebuild_atlas_from_params()

                    if self.params.charset_id == "custom":
                        c_ch, self.params.custom_charset = imgui.input_text("CHARS", self.params.custom_charset)
                        if c_ch:
                            self._rebuild_atlas_from_params()

                    font_name = self.pipeline.atlas_data.font_used if (self.pipeline and self.pipeline.atlas_data) else "-"
                    num_g = self.pipeline.atlas_data.num_ramp if (self.pipeline and self.pipeline.atlas_data) else 0
                    imgui.text_disabled(f"//FONT_ {font_name} ({num_g} + 4 EDGES)")
                    if imgui.button("CARGAR .TTF / .OTF"):
                        self._open_font_picker()
                    if self.params.font_path:
                        imgui.same_line()
                        if imgui.button("RESET FONT"):
                            self.params.font_path = ""
                            self._rebuild_atlas_from_params()

                    # Exact 1.5x matched bounds (4..36 x 6..54) + instant morph cancel -> ZERO locked pixel sliders!
                    cw_ch, new_cw = imgui.slider_float("ANCHO CELDA", float(self.params.cell_w), 4.0, 36.0, "%.1fpx")
                    if cw_ch:
                        self.morpher.cancel()
                        self.params.cell_w = float(np.clip(new_cw, 4.0, 36.0))
                        if self.params.lock_cell_aspect:
                            self.params.cell_h = float(np.clip(self.params.cell_w * 1.5, 6.0, 54.0))
                    ch_ch, new_ch = imgui.slider_float("ALTO CELDA", float(self.params.cell_h), 6.0, 54.0, "%.1fpx")
                    if ch_ch:
                        self.morpher.cancel()
                        self.params.cell_h = float(np.clip(new_ch, 6.0, 54.0))
                        if self.params.lock_cell_aspect:
                            self.params.cell_w = float(np.clip(self.params.cell_h / 1.5, 4.0, 36.0))
                    _, self.params.lock_cell_aspect = imgui.checkbox("BLOQUEAR ASPECTO 2:3", self.params.lock_cell_aspect)
                    imgui.same_line(0, 8)
                    if imgui.button("RESET PIXELES"):
                        self.morpher.cancel()
                        self.params.cell_w = 8.0
                        self.params.cell_h = 12.0
                        self.params.stipple_scale = 2.2
                        self._notify("TAMAÑO DE PIXEL RESTABLECIDO")

                imgui.separator_text("//KINETIC_ POLYGLOT & CUTOFF")
                c1, self.params.black_cutoff = imgui.slider_float("RECORTE NEGRO", float(self.params.black_cutoff), 0.0, 0.45, "%.2f")
                c2, self.params.scramble_amount = imgui.slider_float("MUTACION GLIFOS", float(self.params.scramble_amount), 0.0, 1.0, "%.2f")
                c3, self.params.scramble_speed = imgui.slider_float("VEL. MUTACION", float(self.params.scramble_speed), 0.0, 30.0, "%.1fHz")
                c4, self.params.weight_contrast = imgui.slider_float("PESO BOLD/FINO", float(self.params.weight_contrast), 0.0, 1.0, "%.2f")
                self._cancel_morph_if(c1 or c2 or c3 or c4)

                imgui.separator_text("//EDGES_ ACEROLA DoG + SOBEL")
                e1, self.params.edge_strength = imgui.slider_float("FUERZA BORDES", float(self.params.edge_strength), 0.0, 1.0, "%.2f")
                e2, self.params.edge_threshold = imgui.slider_float("UMBRAL SOBEL", float(self.params.edge_threshold), 0.04, 0.70, "%.2f")
                e3, self.params.edge_boost = imgui.slider_float("HDR BORDES", float(self.params.edge_boost), 0.0, 3.0, "%.2f")
                e4, self.params.dog_sigma = imgui.slider_float("ESCALA DoG", float(self.params.dog_sigma), 0.5, 2.5, "%.2f")
                self._cancel_morph_if(e1 or e2 or e3 or e4)

                imgui.separator_text("//TONE_ AJUSTES")
                t1, self.params.contrast = imgui.slider_float("CONTRASTE", float(self.params.contrast), 0.4, 2.5, "%.2f")
                t2, self.params.brightness = imgui.slider_float("BRILLO", float(self.params.brightness), -0.4, 0.4, "%.2f")
                t3, self.params.gamma = imgui.slider_float("GAMMA", float(self.params.gamma), 0.4, 2.2, "%.2f")
                t4, self.params.saturation = imgui.slider_float("SATURACION", float(self.params.saturation), 0.0, 2.2, "%.2f")
                t5, self.params.dither = imgui.slider_float("DITHERING", float(self.params.dither), 0.0, 1.2, "%.2f")
                t6, self.params.glyph_sharpness = imgui.slider_float("NITIDEZ", float(self.params.glyph_sharpness), 0.0, 1.0, "%.2f")
                t7, self.params.bg_opacity = imgui.slider_float("FONDO", float(self.params.bg_opacity), 0.0, 1.0, "%.2f")
                t8, self.params.matrix_rain = imgui.slider_float("MATRIX RAIN", float(self.params.matrix_rain), 0.0, 1.5, "%.2f")
                t9, self.params.invert = imgui.checkbox("INVERTIR LUMINANCIA", self.params.invert)
                self._cancel_morph_if(t1 or t2 or t3 or t4 or t5 or t6 or t7 or t8 or t9)

                imgui.end_tab_item()

            # ==================== TAB 2: MIX (STUDIO A/B ART + VIDEO + MUSIC FUSION!) ====================
            if imgui.begin_tab_item_simple("MIX"):
                self._apply_tab_glide("MIX")
                imgui.spacing()
                mx = self.mixer

                imgui.separator_text("//MACROS_ ESCENAS ARTE+VIDEO+MUSICA")
                m_w = (imgui.get_content_region_avail().x - 6.0) * 0.5
                for mi, mc in enumerate(self.macros):
                    if mi % 2 == 1:
                        imgui.same_line(0, 6)
                    is_m = mi == self.active_macro_idx
                    if is_m:
                        imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(*self.theme.frame_active))
                        imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*self.theme.node_core))
                    if imgui.button(mc.name, imgui.ImVec2(m_w - 28.0, 22)):
                        self._trigger_macro(mi)
                    if is_m:
                        imgui.pop_style_color(2)
                    imgui.same_line(0, 2)
                    if imgui.button(f"S##SaveM{mi}", imgui.ImVec2(24.0, 22)):
                        self._save_current_to_macro(mi)

                imgui.spacing()
                imgui.separator_text("//DECK_ A/B FUSION VIDEO + ARTE [V]")
                _, mx.crossfade = imgui.slider_float("A <-> B MIX", float(mx.crossfade), 0.0, 1.0, "%.2f")
                _, mx.crossfade_audio_mod = imgui.slider_float("AUDIO -> MIX", float(mx.crossfade_audio_mod), 0.0, 1.5, "%.2f")
                _, mx.blend_mode = imgui.combo("MODO BLEND", int(mx.blend_mode), BLEND_MODE_NAMES)
                if mx.blend_mode in (5, 6):
                    _, mx.luma_thresh = imgui.slider_float("UMBRAL LUMA KEY", float(mx.luma_thresh), 0.0, 0.85, "%.2f")
                    _, mx.luma_soft = imgui.slider_float("SUAVIDAD KEY", float(mx.luma_soft), 0.02, 0.50, "%.2f")

                _, mx.deck_b_mode = imgui.combo("ORIGEN DECK B", int(mx.deck_b_mode), ["Arte 3D Generativo (GPU)", "Clip / Video / Foto (Deck B)"])
                if mx.deck_b_mode == 0:
                    _, mx.deck_b_scene = imgui.combo("ESCENA 3D B", int(mx.deck_b_scene), PROCEDURAL_SCENE_NAMES)
                else:
                    b_lbl = self.media_b.info.name if self.media_b.info.source_type != "procedural" else "(Asigna un clip con [B] abajo)"
                    imgui.text_disabled(f"//DECK_B_ {b_lbl[:34]}")

                # Live Screen / Window Capture into Deck A
                scr_on = mx.live_screen_active
                if scr_on:
                    imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(*self.theme.frame_active))
                    imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*self.theme.node_core))
                if imgui.button("DETENER PANTALLA LIVE [L]" if scr_on else "CAPTURAR PANTALLA LIVE [L]", imgui.ImVec2(-1, 24)):
                    self._toggle_live_screen()
                if scr_on:
                    imgui.pop_style_color(2)
                if mx.live_screen_active:
                    _, mx.screen_crop_mode = imgui.combo("REGION CAPTURA", int(mx.screen_crop_mode), SCREEN_CROP_NAMES)

                imgui.spacing()
                imgui.separator_text("//CLIPS_ BANCO VJ (8 SLOTS)")
                if imgui.button("+ AÑADIR CLIPS (MULTI)", imgui.ImVec2((imgui.get_content_region_avail().x - 6.0) * 0.62, 22)):
                    self._open_clips_picker()
                imgui.same_line(0, 6)
                if imgui.button("NEXT CLIP >", imgui.ImVec2(-1, 22)):
                    nxt = self.clip_bank.next_clip_index()
                    if nxt >= 0:
                        self._load_clip_to_deck_a(nxt)

                for si, slot in enumerate(self.clip_bank.slots):
                    if slot is None:
                        continue
                    is_a = si == self.clip_bank.active_slot_a
                    is_b = si == self.clip_bank.active_slot_b
                    tag = f"{'[A]' if is_a else ''}{'[B]' if is_b else ''}"
                    if is_a:
                        imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(*self.theme.frame_active))
                    if imgui.button(f"A##cA{si}", imgui.ImVec2(24, 20)):
                        self._load_clip_to_deck_a(si)
                    if is_a:
                        imgui.pop_style_color()
                    imgui.same_line(0, 3)
                    if is_b:
                        imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(*self.theme.frame_active))
                    if imgui.button(f"B##cB{si}", imgui.ImVec2(24, 20)):
                        self._load_clip_to_deck_b(si)
                    if is_b:
                        imgui.pop_style_color()
                    imgui.same_line(0, 5)
                    imgui.text_colored(imgui.ImVec4(*self.theme.node_core if (is_a or is_b) else self.theme.text_main), f"0{si+1} {slot.name[:20]} {tag}")
                    imgui.same_line(max(240.0, imgui.get_content_region_avail().x - 24.0))
                    if imgui.button(f"X##cD{si}", imgui.ImVec2(22, 20)):
                        self.clip_bank.remove_clip(si)

                if not self.clip_bank.filled_indices():
                    imgui.text_disabled("Arrastra varios videos o pulsa [+ AÑADIR CLIPS]")

                _, mx.auto_clip_on_drop = imgui.checkbox("AUTO-CLIP EN DROPS!!", mx.auto_clip_on_drop)
                bar_opts = ["Off", "Cada 4 compases", "Cada 8 compases", "Cada 16 compases"]
                bar_vals = [0, 4, 8, 16]
                cur_b_i = bar_vals.index(mx.auto_clip_bars) if mx.auto_clip_bars in bar_vals else 0
                b_ch, new_bi = imgui.combo("AUTO-CLIP COMPAS", cur_b_i, bar_opts)
                if b_ch:
                    mx.auto_clip_bars = bar_vals[new_bi]
                _, mx.beat_speed_boost = imgui.slider_float("BEAT SPEED BOOST", float(mx.beat_speed_boost), 0.0, 1.5, "%.2f")

                imgui.spacing()
                imgui.separator_text("//VIDEO_ TRANSFORM & PRE-GRADE (A)")
                _, mx.a_fit_mode = imgui.combo("ENCUADRE", int(mx.a_fit_mode), FIT_MODE_NAMES)
                _, mx.a_mirror = imgui.combo("SIMETRIA ESPEJO", int(mx.a_mirror), MIRROR_MODE_NAMES)
                _, mx.a_scale = imgui.slider_float("ESCALA VIDEO", float(mx.a_scale), 0.5, 2.5, "%.2fx")
                _, mx.a_rot_deg = imgui.slider_float("ROTACION", float(mx.a_rot_deg), -180.0, 180.0, "%.0f deg")
                _, mx.a_flip_x = imgui.checkbox("FLIP HORIZONTAL", mx.a_flip_x)
                imgui.same_line(0, 10)
                _, mx.a_flip_y = imgui.checkbox("FLIP VERTICAL", mx.a_flip_y)
                _, mx.pre_hue = imgui.slider_float("PRE-HUE", float(mx.pre_hue), -0.5, 0.5, "%.2f")
                _, mx.pre_contrast = imgui.slider_float("PRE-CONTRASTE", float(mx.pre_contrast), 0.4, 2.2, "%.2f")
                _, mx.pre_sat = imgui.slider_float("PRE-SATURACION", float(mx.pre_sat), 0.0, 2.2, "%.2f")
                if imgui.button("RESET VIDEO TRANSFORM"):
                    mx.a_scale = 1.0
                    mx.a_rot_deg = 0.0
                    mx.a_flip_x = False
                    mx.a_flip_y = False
                    mx.a_mirror = 0
                    mx.a_fit_mode = 0
                    mx.pre_hue = 0.0
                    mx.pre_contrast = 1.0
                    mx.pre_sat = 1.0

                imgui.spacing()
                imgui.separator_text("//ART_ CAPA TIPOGRAFICA EDITORIAL [T]")
                _, mx.art_overlay_style = imgui.combo("ESTILO CAPA", int(mx.art_overlay_style), ART_OVERLAY_STYLES)
                if mx.art_overlay_style > 0:
                    _, mx.art_overlay_opacity = imgui.slider_float("OPACIDAD CAPA", float(mx.art_overlay_opacity), 0.1, 1.0, "%.2f")
                    _, mx.art_use_spotify = imgui.checkbox("USAR CANCION SPOTIFY EN VIVO", mx.art_use_spotify)
                    if not mx.art_use_spotify:
                        _, mx.art_custom_title = imgui.input_text("TITULO ARTE", mx.art_custom_title)
                        _, mx.art_custom_sub = imgui.input_text("SUBTITULO", mx.art_custom_sub)

                imgui.end_tab_item()

            # ==================== TAB 3: MOT (11 GPU DISPLACEMENT ANIMATIONS!) ====================
            if imgui.begin_tab_item_simple("MOT"):
                self._apply_tab_glide("MOTION")
                imgui.spacing()
                imgui.separator_text("//DISPLACEMENT_ 11 ANIMACIONES GPU [M]")

                dm_ch, new_dm = imgui.combo("MODO ACTIVO", int(self.params.disp_mode), DISPLACEMENT_MODE_NAMES)
                if dm_ch:
                    self.morpher.cancel()
                    self.params.disp_mode = new_dm
                    if new_dm > 0 and self.params.disp_strength < 0.25:
                        self.params.disp_strength = 0.70

                # Quick-select 2-column matrix of all 12 displacement options!
                short_disp = [
                    "00 OFF",
                    "01 LIQUID CURL",
                    "02 SHOCKWAVE",
                    "03 SILK WAVE",
                    "04 SLIT-SCAN",
                    "05 3D PARALLAX",
                    "06 GLITCH BLK",
                    "07 VORTEX",
                    "08 PRISM RGB",
                    "09 KALEIDO",
                    "10 ELASTIC",
                    "11 PIXEL MELT",
                ]
                m_btn_w = (imgui.get_content_region_avail().x - 6.0) * 0.5
                for midx, mlbl in enumerate(short_disp):
                    if midx % 2 == 1:
                        imgui.same_line(0, 6)
                    active_m = int(self.params.disp_mode) == midx
                    if active_m:
                        imgui.push_style_color(imgui.Col_.button, imgui.ImVec4(*self.theme.frame_active))
                        imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*self.theme.node_core))
                    if imgui.button(mlbl, imgui.ImVec2(m_btn_w, 22)):
                        self.morpher.cancel()
                        self.params.disp_mode = midx
                        if midx > 0 and self.params.disp_strength < 0.25:
                            self.params.disp_strength = 0.72
                        self._notify(f"MOTION // {DISPLACEMENT_MODE_NAMES[midx]}")
                    if active_m:
                        imgui.pop_style_color(2)

                imgui.spacing()
                imgui.separator_text("//PARAMETROS_ DE DESPLAZAMIENTO")
                d1, self.params.disp_strength = imgui.slider_float("FUERZA", float(self.params.disp_strength), 0.0, 2.0, "%.2f")
                d2, self.params.disp_speed = imgui.slider_float("VELOCIDAD", float(self.params.disp_speed), 0.1, 3.0, "%.2fx")
                d3, self.params.disp_scale = imgui.slider_float("ESCALA / ONDA", float(self.params.disp_scale), 0.3, 3.5, "%.2f")
                d4, self.params.disp_audio_sync = imgui.slider_float("SYNC AUDIO SPRING", float(self.params.disp_audio_sync), 0.0, 2.0, "%.2f")
                self._cancel_morph_if(d1 or d2 or d3 or d4)

                imgui.spacing()
                imgui.separator_text("//LIVE_ FISICA AUDIO -> MOTION")
                imgui.text_colored(imgui.ImVec4(*self.theme.text_main), "SPRING KICK ")
                imgui.same_line(0, 6)
                imgui.progress_bar(min(1.0, float(intel.spring_kick)), imgui.ImVec2(-1, 10), f"{intel.spring_kick:.2f}")
                imgui.text_colored(imgui.ImVec4(*self.theme.text_main), "HARMONIC FLW")
                imgui.same_line(0, 6)
                imgui.progress_bar(min(1.0, float(intel.harmonic_energy)), imgui.ImVec2(-1, 10), f"{intel.harmonic_energy:.2f}")
                imgui.text_colored(imgui.ImVec4(*self.theme.text_main), "PERCUSS FLUX")
                imgui.same_line(0, 6)
                imgui.progress_bar(min(1.0, float(intel.percussive_energy)), imgui.ImVec2(-1, 10), f"{intel.percussive_energy:.2f}")

                imgui.end_tab_item()

            # ==================== TAB 4: BLOOM & DISSOLVE ====================
            if imgui.begin_tab_item_simple("BLM"):
                self._apply_tab_glide("BLOOM")
                imgui.spacing()
                imgui.separator_text("//COLOR_ PALETA & HDR")
                b0, self.params.color_mode = imgui.combo("PALETA", int(self.params.color_mode), COLOR_MODE_NAMES)
                self._cancel_morph_if(b0)
                if self.params.color_mode == 7:
                    bc1, c_dark = imgui.color_edit3("SOMBRAS", self.params.color_dark)
                    self.params.color_dark = list(c_dark)
                    bc2, c_bright = imgui.color_edit3("LUCES", self.params.color_bright)
                    self.params.color_bright = list(c_bright)
                    self._cancel_morph_if(bc1 or bc2)
                b1, self.params.hdr_boost = imgui.slider_float("GANANCIA HDR", float(self.params.hdr_boost), 0.5, 8.0, "%.2f")
                b2, self.params.hdr_power = imgui.slider_float("CURVA HDR", float(self.params.hdr_power), 0.5, 3.0, "%.2f")

                imgui.separator_text("//BLOOM_ MIP-CHAIN (CoD:AW)")
                b3, self.params.bloom_intensity = imgui.slider_float("INTENSIDAD", float(self.params.bloom_intensity), 0.0, 3.5, "%.2f")
                b4, self.params.bloom_threshold = imgui.slider_float("UMBRAL HDR", float(self.params.bloom_threshold), 0.0, 2.5, "%.2f")
                b5, self.params.bloom_soft_knee = imgui.slider_float("SOFT KNEE", float(self.params.bloom_soft_knee), 0.05, 1.0, "%.2f")
                b6, self.params.bloom_radius = imgui.slider_float("RADIO TENT", float(self.params.bloom_radius), 0.4, 2.5, "%.2f")
                b7, self.params.bloom_scatter = imgui.slider_float("SCATTER", float(self.params.bloom_scatter), 0.2, 0.95, "%.2f")
                b8, self.params.bloom_anamorphic = imgui.slider_float("ANAMORFICO", float(self.params.bloom_anamorphic), 1.0, 4.5, "%.2fx")
                b9, bt = imgui.color_edit3("TINTE BLOOM", self.params.bloom_tint)
                self.params.bloom_tint = list(bt)

                imgui.separator_text("//TRAILS_ FOSFORO & WIND DISSOLVE")
                b10, self.params.trail_decay = imgui.slider_float("PERSISTENCIA", float(self.params.trail_decay), 0.0, 0.93, "%.2f")
                b11, self.params.trail_drift_x = imgui.slider_float("VIENTO X (DISSOLVE)", float(self.params.trail_drift_x), -3.0, 3.0, "%.2f")
                b12, self.params.trail_drift_y = imgui.slider_float("VIENTO Y", float(self.params.trail_drift_y), -3.0, 3.0, "%.2f")
                b13, self.params.trail_dissolve = imgui.slider_float("DISOLUCION PARTICULAS", float(self.params.trail_dissolve), 0.0, 1.0, "%.2f")
                b14, self.params.trail_drift = imgui.slider_float("DERIVA RADIAL", float(self.params.trail_drift), -2.0, 2.0, "%.2f")
                self._cancel_morph_if(any((b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14)))
                if imgui.button("LIMPIAR ESTELAS"):
                    if self.pipeline is not None:
                        self.pipeline.clear_trails_and_paint(clear_trails=True, clear_paint=False)

                imgui.separator_text("//CRT_ POST-PROCESADO")
                p1, self.params.exposure = imgui.slider_float("EXPOSICION", float(self.params.exposure), 0.3, 2.5, "%.2f")
                p2, self.params.tonemap_mode = imgui.combo("TONEMAP", int(self.params.tonemap_mode), TONEMAP_NAMES)
                p3, self.params.chromatic_aberration = imgui.slider_float("ABERRACION", float(self.params.chromatic_aberration), 0.0, 0.025, "%.4f")
                p4, self.params.scanlines = imgui.slider_float("SCANLINES", float(self.params.scanlines), 0.0, 1.0, "%.2f")
                p5, self.params.curvature = imgui.slider_float("CURVATURA", float(self.params.curvature), 0.0, 0.45, "%.2f")
                p6, self.params.vignette = imgui.slider_float("VIÑETA", float(self.params.vignette), 0.0, 1.0, "%.2f")
                p7, self.params.grain = imgui.slider_float("GRANO CRT", float(self.params.grain), 0.0, 0.5, "%.2f")
                self._cancel_morph_if(any((p1, p2, p3, p4, p5, p6, p7)))

                imgui.end_tab_item()

            # ==================== TAB 5: SPOTIFY / AUDIO / MUSICAL INTELLIGENCE ====================
            if imgui.begin_tab_item_simple("AUD"):
                self._apply_tab_glide("AUDIO")
                imgui.spacing()
                rt = self.audio.routing

                changed_am, new_am = imgui.combo("//FUENTE_", int(self.audio.mode), AUDIO_SOURCE_NAMES)
                if changed_am:
                    self.audio.set_mode(new_am)
                imgui.text_disabled(f"//DSP_ {self.audio.mir.dsp_backend} | AGC:{intel.agc_gain:.1f}x | {intel.rms_db:.0f}dB")

                if self.audio.mode == 0:
                    if self.audio.loopback_devices:
                        dev_labels = [name.replace("[Loopback]", "").strip()[:34] for _, name in self.audio.loopback_devices]
                        ch_dev, new_pos = imgui.combo("SALIDA PC", int(self.audio.selected_loopback_pos), dev_labels)
                        if ch_dev:
                            self.audio.select_loopback_device(new_pos)
                    if imgui.button("AUTO-DETECTAR SALIDA"):
                        self.audio.auto_detect_active_loopback()
                        self._notify("ESCANEANDO SALIDAS WASAPI ACTIVAS...")
                    imgui.same_line(0, 6)
                    if imgui.button("TAP BPM"):
                        self.audio.tap_tempo()
                    imgui.same_line(0, 6)
                    if imgui.button("DROP! [D]"):
                        self.audio.trigger_manual_drop()
                    _, self.audio.auto_switch_wasapi = imgui.checkbox("AUTO-CAMBIAR SI SALIDA ESTA MUDA", self.audio.auto_switch_wasapi)

                imgui.separator_text("//AI_ COREOGRAFIA & FISICA SPRING")
                _, rt.profile_idx = imgui.combo("COREOGRAFIA", int(rt.profile_idx), CHOREOGRAPHY_PROFILES)
                _, rt.master_reactivity = imgui.slider_float("INTENSIDAD MASTER", float(rt.master_reactivity), 0.0, 2.5, "%.2f")
                _, rt.spring_bounce = imgui.slider_float("FISICA SPRING", float(rt.spring_bounce), 0.0, 1.5, "%.2f")
                _, rt.disp_react = imgui.slider_float("AUDIO -> MOTION", float(rt.disp_react), 0.0, 2.0, "%.2f")
                _, rt.drop_sensitivity = imgui.slider_float("SENSIBILIDAD DROPS", float(rt.drop_sensitivity), 0.5, 2.2, "%.2fx")
                _, self.audio.gain = imgui.slider_float("GANANCIA ENTRADA", float(self.audio.gain), 0.25, 4.0, "%.2f")
                _, self.audio.smoothing = imgui.slider_float("SUAVIZADO", float(self.audio.smoothing), 0.15, 0.92, "%.2f")

                imgui.separator_text("//STEMS_ PARSEVAL + HPSS EN VIVO")
                stem_rows = [
                    ("KICK  (SPRING)  ", intel.spring_kick),
                    ("BASS  (GROOVE)  ", intel.bass),
                    ("SNARE (PERCUS.) ", intel.snare),
                    ("VOCAL (ARMONICO)", intel.vocals),
                    ("HIHAT (5.2-16k) ", intel.hihat),
                    ("HPSS  PERCUSIVO ", intel.percussive_energy),
                    ("HPSS  ARMONICO  ", intel.harmonic_energy),
                ]
                for lbl, val in stem_rows:
                    imgui.text_colored(imgui.ImVec4(*self.theme.text_main), f"{lbl}")
                    imgui.same_line(0, 6)
                    imgui.progress_bar(min(1.0, float(val)), imgui.ImVec2(-1, 10), f"{val:.2f}")

                imgui.separator_text(f"//CHROMA_ 12 NOTAS (KEY: {intel.dominant_note} | {intel.section_name})")
                ch_min = imgui.get_cursor_screen_pos()
                ch_w = imgui.get_content_region_avail().x
                ch_h = 56.0
                ch_max = imgui.ImVec2(ch_min.x + ch_w, ch_min.y + ch_h)
                imgui.dummy(imgui.ImVec2(ch_w, ch_h))
                draw_chroma_bars(imgui.get_window_draw_list(), ch_min, ch_max, intel.chroma, NOTE_NAMES, self.theme)

                imgui.separator_text("//MATRIX_ RUTEO DE STEMS -> VISUALES")
                _, rt.kick_punch = imgui.slider_float("KICK -> BLOOM/ZOOM", float(rt.kick_punch), 0.0, 2.0, "%.2f")
                _, rt.snare_glitch = imgui.slider_float("SNARE -> GLITCH TEAR", float(rt.snare_glitch), 0.0, 2.0, "%.2f")
                _, rt.hihat_spark = imgui.slider_float("HIHAT -> SCRAMBLE/CA", float(rt.hihat_spark), 0.0, 2.0, "%.2f")
                _, rt.vocal_glow = imgui.slider_float("VOCAL -> BORDES/GLOW", float(rt.vocal_glow), 0.0, 2.0, "%.2f")
                _, rt.pitch_color = imgui.slider_float("ACORDES -> COLOR HUE", float(rt.pitch_color), 0.0, 1.5, "%.2f")
                _, rt.spatial_eq = imgui.slider_float("EQ ESPACIAL 64-BAND", float(rt.spatial_eq), 0.0, 1.5, "%.2f")
                _, rt.grid_breathe = imgui.slider_float("KICK -> TAMAÑO GRILLA", float(rt.grid_breathe), 0.0, 1.5, "%.2f")
                _, rt.wind_audio = imgui.slider_float("AUDIO -> VIENTO/POLVO", float(rt.wind_audio), 0.0, 2.0, "%.2f")

                _, rt.shockwave_on_drop = imgui.checkbox("ONDA SHOCKWAVE EN DROPS", rt.shockwave_on_drop)
                _, rt.shockwave_on_kick = imgui.checkbox("ONDA SHOCKWAVE EN KICKS", rt.shockwave_on_kick)
                _, rt.auto_morph_on_drop = imgui.checkbox("AUTO-MORPH PRESET EN DROPS!!", rt.auto_morph_on_drop)

                imgui.spacing()
                scope_names = ["ONDA ENVOLVENTE (FOTO 2)", "OSCILOSCOPIO + HEX (FOTO 3)", "ESPECTRO 64 MEL + PEAK"]
                _, self.scope_mode = imgui.combo("VISOR INFERIOR", int(self.scope_mode), scope_names)

                imgui.end_tab_item()

            # ==================== TAB 6: PHYSICS & BLOBS ====================
            if imgui.begin_tab_item_simple("BLB"):
                self._apply_tab_glide("PHYSICS")
                imgui.spacing()
                imgui.separator_text("//CURSOR_ FISICA EN CANVAS")
                _, self.interaction.mode = imgui.combo("MODO", int(self.interaction.mode), INTERACTION_MODES)
                _, self.interaction.radius = imgui.slider_float("RADIO", float(self.interaction.radius), 0.06, 0.65, "%.2f")
                _, self.interaction.strength = imgui.slider_float("FUERZA", float(self.interaction.strength), 0.2, 2.5, "%.2f")
                if imgui.button("LANZAR ONDA CENTRAL"):
                    self.interaction.spawn_ripple(0.5, 0.5, strength=1.35)

                imgui.separator_text("//BRUSH_ PINCEL EMISIVO HDR")
                _, bc = imgui.color_edit3("COLOR NEON", self.interaction.brush_color)
                self.interaction.brush_color = list(bc)
                _, self.interaction.brush_radius = imgui.slider_float("GROSOR", float(self.interaction.brush_radius), 0.008, 0.12, "%.3f")
                _, self.interaction.paint_decay = imgui.slider_float("PERSISTENCIA", float(self.interaction.paint_decay), 0.90, 1.00, "%.4f")
                if imgui.button("LIMPIAR PINTURA [C]"):
                    self.interaction.clear_paint_requested = True

                imgui.separator_text(f"//BLOBS_ TRACKING INTELIGENTE ({len(self.blob_tracker.tracks)} ACTIVOS)")
                bs = self.blob_tracker.settings
                _, bs.enabled = imgui.checkbox("ACTIVAR BLOB TRACKING [X]", bs.enabled)
                _, bs.visual_style = imgui.combo("ESTILO VISUAL [B]", int(bs.visual_style), BLOB_VISUAL_STYLES)
                _, bs.detect_mode = imgui.combo("MOTOR DETECCION", int(bs.detect_mode), BLOB_DETECT_MODES)
                _, bs.max_blobs = imgui.slider_int("MAX BLOBS", int(bs.max_blobs), 2, 20)
                _, bs.sensitivity = imgui.slider_float("SENSIBILIDAD CV", float(bs.sensitivity), 0.15, 0.95, "%.2f")
                _, bs.smoothing = imgui.slider_float("SUAVIZADO KALMAN", float(bs.smoothing), 0.20, 0.94, "%.2f")
                _, bs.plexus_enabled = imgui.checkbox("RED CONSTELACION (PLEXUS)", bs.plexus_enabled)
                if bs.plexus_enabled:
                    _, bs.plexus_distance = imgui.slider_float("DISTANCIA PLEXUS", float(bs.plexus_distance), 0.10, 0.65, "%.2f")
                    _, bs.plexus_curved = imgui.checkbox("LINEAS CURVAS BEZIER", bs.plexus_curved)
                _, bs.show_trails = imgui.checkbox("ESTELAS DE TRAYECTORIA", bs.show_trails)
                _, bs.show_labels = imgui.checkbox("MICRO-ETIQUETAS ID", bs.show_labels)
                _, bs.audio_reactive = imgui.checkbox("BLOBS REACTIVOS AL BEAT", bs.audio_reactive)

                imgui.separator_text("//HUD_ INMERSION & SPLIT-SCREEN")
                if imgui.button("PANTALLA COMPLETA [F11]", imgui.ImVec2((inner_w - 16) * 0.5, 26)):
                    self._toggle_fullscreen()
                imgui.same_line(0, 6)
                if imgui.button("MODO CINE 100% [F]", imgui.ImVec2(-1, 26)):
                    self._toggle_immersive_cinema()
                _, self.show_immersive_pill = imgui.checkbox("MINI-HUD SPOTIFY EN CINE [H]", self.show_immersive_pill)
                _, self.interaction.split_enabled = imgui.checkbox("CORTINA SPLIT-SCREEN [S]", self.interaction.split_enabled)
                _, self.interaction.split_mode = imgui.combo("LADO IZQUIERDO", int(self.interaction.split_mode), SPLIT_MODES)
                _, self.interaction.split_pos = imgui.slider_float("POSICION SPLIT", float(self.interaction.split_pos), 0.02, 0.98, "%.2f")

                imgui.separator_text("//CAMARA_")
                _, self.interaction.zoom = imgui.slider_float("ZOOM", float(self.interaction.zoom), 0.5, 6.0, "%.2fx")
                _, self.fit_viewport_fill = imgui.checkbox("ESTIRAR AL VIEWPORT", self.fit_viewport_fill)
                if imgui.button("RESET VISTA [R]"):
                    self.interaction.reset_view()

                imgui.end_tab_item()

            # ==================== TAB 6: EXPORT & SHADERS ====================
            if imgui.begin_tab_item_simple("EXP"):
                self._apply_tab_glide("EXPORT")
                imgui.spacing()
                imgui.separator_text("//GPU_ RESOLUCION HDR")
                res_opts = ["1280x720 (HD)", "1600x900 (HD+)", "1920x1080 (FHD)", "IGUALAR FUENTE"]
                r_ch, new_r = imgui.combo("RESOLUCION", int(self.render_res_idx), res_opts)
                if r_ch:
                    self.render_res_idx = new_r
                    self._apply_render_resolution()

                imgui.separator_text("//SNAPSHOT_ PNG")
                if imgui.button("CAPTURA RAPIDA PNG (exports/)", imgui.ImVec2(-1, 26)):
                    self._quick_png_snapshot()
                if imgui.button("GUARDAR PNG COMO...", imgui.ImVec2(-1, 24)):
                    self._save_png_dialog = pfd.save_file(
                        "Guardar Captura PNG",
                        str(EXPORTS_DIR / "specimen_ascii.png"),
                        ["Imagen PNG", "*.png"],
                    )

                imgui.separator_text("//FFMPEG_ VIDEO / GIF")
                _, self.export_settings.format_idx = imgui.combo("FORMATO", int(self.export_settings.format_idx), EXPORT_FORMATS)
                if self.export_settings.format_idx in (0, 2):
                    _, self.export_settings.crf = imgui.slider_int("CRF (CALIDAD)", int(self.export_settings.crf), 10, 28)
                if self.media.info.source_type != "video":
                    _, self.export_settings.fps = imgui.slider_int("FPS", int(self.export_settings.fps), 15, 60)
                    _, self.export_settings.duration_sec = imgui.slider_float("DURACION", float(self.export_settings.duration_sec), 2.0, 30.0, "%.1fs")
                else:
                    _, self.export_settings.keep_audio = imgui.checkbox("CONSERVAR AUDIO ORIGINAL", self.export_settings.keep_audio)
                    _, self.export_settings.use_ab_range = imgui.checkbox("SOLO RANGO A-B", self.export_settings.use_ab_range)

                if not self.exporter.active:
                    if imgui.button("EXPORTAR CON FFMPEG...", imgui.ImVec2(-1, 30)):
                        self._open_video_export_dialog()
                else:
                    if imgui.button("CANCELAR EXPORTACION", imgui.ImVec2(-1, 30)):
                        self.exporter.cancel()

                if self.exporter.error_message:
                    imgui.text_colored(imgui.ImVec4(1.0, 0.45, 0.45, 1.0), self.exporter.error_message)
                elif self.exporter.status_message:
                    imgui.text_colored(imgui.ImVec4(*self.theme.line_core), self.exporter.status_message)

                imgui.separator_text("//GLSL_ HOT-RELOAD")
                _, self.auto_hot_reload = imgui.checkbox("AUTO HOT-RELOAD shaders/*.frag", self.auto_hot_reload)
                if imgui.button("RECARGAR SHADERS [F5]", imgui.ImVec2(-1, 24)) and self.pipeline is not None:
                    self.pipeline.check_hot_reload(force=True)
                    self._notify(self.pipeline.last_reload_status)
                if imgui.button("ABRIR shaders/", imgui.ImVec2((inner_w - 16) * 0.5, 24)):
                    try:
                        os.startfile(str(SHADERS_DIR))
                    except Exception:
                        subprocess.Popen(["explorer", str(SHADERS_DIR)])
                imgui.same_line(0, 6)
                if imgui.button("ABRIR exports/", imgui.ImVec2(-1, 24)):
                    try:
                        os.startfile(str(EXPORTS_DIR))
                    except Exception:
                        subprocess.Popen(["explorer", str(EXPORTS_DIR)])

                imgui.end_tab_item()

            # ==================== TAB 7: GUIA INTERACTIVA ====================
            if imgui.begin_tab_item_simple("GUIA"):
                self._apply_tab_glide("GUIA")
                imgui.spacing()
                self._draw_guide_content(is_modal=False)
                imgui.end_tab_item()

            imgui.end_tab_bar()

        imgui.pop_item_width()
        imgui.end_child()

        # ==================== DOCKED BOTTOM GLOWING AUDIO SCOPE ====================
        imgui.spacing()
        sp_min = imgui.get_cursor_screen_pos()
        sp_max = imgui.ImVec2(sp_min.x + inner_w, sp_min.y + scope_panel_h)
        imgui.invisible_button("##BottomAudioScopeBtn", imgui.ImVec2(inner_w, scope_panel_h))
        if imgui.is_item_clicked():
            self.scope_mode = (self.scope_mode + 1) % 3
        draw_audio_scope_panel(
            dl,
            sp_min,
            sp_max,
            self.theme,
            self.audio.waveform,
            self.audio.envelope_history,
            self.audio.spectrum,
            self._fft_peak_hold,
            intel.kick,
            intel.vocals,
            intel.hihat,
            app_time,
            self.scope_mode,
        )

        draw_specimen_box(
            dl,
            imgui.ImVec2(d_min.x + 1, d_min.y + 1),
            imgui.ImVec2(d_min.x + d_size.x - 1, d_min.y + d_size.y - 1),
            self.theme,
            with_nodes=True,
            intensity=0.95,
            anim_time=app_time * 0.85,
            kick_pulse=intel.kick,
            drop_glow=intel.drop_intensity,
            num_pulses=2,
        )
        imgui.end_child()

    def show_gui(self) -> None:
        self._poll_dialogs_and_drops()
        self._handle_shortcuts()
        self._step_simulation_and_gpu()

        if self.immersive_mode:
            # Zero padding edge-to-edge full-monitor Cinema Immersion!
            avail = imgui.get_content_region_avail()
            self._draw_viewport_and_timeline(max(320.0, avail.x), max(220.0, avail.y))
            self._draw_guide_drawer_overlay()
            return

        self._draw_top_specimen_bar()
        imgui.spacing()

        avail = imgui.get_content_region_avail()
        avail_h = max(220.0, avail.y)
        full_inspector_w = float(np.clip(avail.x * 0.30, 395.0, 470.0))
        self.inspector_width = full_inspector_w

        # Smooth animated slide width for the Specimen Dossier!
        cur_panel_w = full_inspector_w * self._hud_slide
        if cur_panel_w > 18.0:
            vp_w = max(320.0, avail.x - cur_panel_w - 8.0)
        else:
            vp_w = max(320.0, avail.x)

        imgui.begin_group()
        self._draw_viewport_and_timeline(vp_w, avail_h)
        imgui.end_group()

        if cur_panel_w > 18.0:
            imgui.same_line(0, 8)
            self._draw_specimen_dossier_inspector(cur_panel_w, avail_h)

        self._draw_guide_drawer_overlay()

    def before_exit(self) -> None:
        self.exporter.cancel()
        self.screen_cap.stop()
        self.media.close_media()
        self.media_b.close_media()
        self.audio.close()
        self.spotify.close()
        if self.pipeline is not None:
            self.pipeline.release()


def launch_studio() -> None:
    app = LumenAsciiStudioApp()

    runner_params = hello_imgui.RunnerParams()
    runner_params.app_window_params.window_title = "SPECIMEN N001 // LUMEN ASCII BLOOM STUDIO"
    runner_params.app_window_params.window_geometry.size = (1380, 820)
    runner_params.app_window_params.restore_previous_geometry = False

    runner_params.imgui_window_params.default_imgui_window_type = (
        hello_imgui.DefaultImGuiWindowType.provide_full_screen_window
    )
    runner_params.imgui_window_params.show_menu_bar = False
    runner_params.imgui_window_params.show_status_bar = False

    runner_params.fps_idling.enable_idling = False

    runner_params.callbacks.load_additional_fonts = app.load_additional_fonts
    runner_params.callbacks.setup_imgui_style = app.setup_imgui_style
    runner_params.callbacks.post_init = app.post_init
    runner_params.callbacks.show_gui = app.show_gui
    runner_params.callbacks.before_exit = app.before_exit

    addons = immapp.AddOnsParams()
    addons.with_implot = True

    immapp.run(runner_params, addons)
