from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import moderngl
import numpy as np

from .config import ShaderParams
from .glyph_atlas import GlyphAtlasData, build_glyph_atlas
from .interaction import InteractionState
from .music_intelligence import VisualModulation
from .video_studio import VideoMixerState


@dataclass
class _ShaderSlot:
    name: str
    vert_path: Path
    frag_path: Path
    program: moderngl.Program
    vao: moderngl.VertexArray
    vert_mtime: float
    frag_mtime: float


def _set_u(prog: moderngl.Program, name: str, val) -> None:
    if name in prog:
        prog[name].value = val


class AsciiBloomPipeline:
    """
    6-Pass HDR RGBA16F ModernGL Pipeline with Musical Intelligence 64-Band FFT Texture & Choreography Modulation.
    """

    BLOOM_MIP_COUNT = 6

    def __init__(self, ctx: moderngl.Context, shaders_dir: Path, width: int = 1280, height: int = 720) -> None:
        self.ctx = ctx
        self.shaders_dir = shaders_dir
        self.width = max(64, int(width))
        self.height = max(64, int(height))

        self.slots: Dict[str, _ShaderSlot] = {}
        self.last_reload_status: str = "Shaders GLSL compilados OK"
        self.last_reload_error: str = ""
        self._reload_timer: float = 0.0

        self.atlas_data: Optional[GlyphAtlasData] = None
        self.tex_atlas: Optional[moderngl.Texture] = None

        self.tex_media: Optional[moderngl.Texture] = None
        self._media_size: Tuple[int, int] = (0, 0)

        self.tex_media_b: Optional[moderngl.Texture] = None
        self._media_b_size: Tuple[int, int] = (0, 0)

        # Live 64x1 R32F GPU FFT Spectrum Texture for spatial audio reactivity & Audio Mandala EQ
        self.tex_fft = self.ctx.texture((64, 1), 1, data=np.zeros(64, dtype=np.float32).tobytes(), dtype="f4")
        self.tex_fft.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.tex_fft.repeat_x = False
        self.tex_fft.repeat_y = False

        # Tiny 112x64 GPU analysis target for ultra-fast (~2ms) real-time CV blob tracking on ANY video/image/3D scene!
        self._analysis_size = (112, 64)
        self.tex_analysis = self.ctx.texture(self._analysis_size, 3, dtype="f1")
        self.tex_analysis.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.fbo_analysis = self.ctx.framebuffer(color_attachments=[self.tex_analysis])
        self._analysis_prog = self.ctx.program(
            vertex_shader="""
                #version 330 core
                out vec2 v_uv;
                void main() {
                    vec2 pos = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
                    v_uv = pos;
                    gl_Position = vec4(pos * 2.0 - 1.0, 0.0, 1.0);
                }
            """,
            fragment_shader="""
                #version 330 core
                in vec2 v_uv;
                out vec4 fragColor;
                uniform sampler2D u_src;
                void main() {
                    fragColor = vec4(texture(u_src, v_uv).rgb, 1.0);
                }
            """,
        )
        self._analysis_vao = self.ctx.vertex_array(self._analysis_prog, [])
        self._last_src_tex: Optional[moderngl.Texture] = None

        self._allocated_textures: List[moderngl.Texture] = []
        self._allocated_fbos: List[moderngl.Framebuffer] = []

        self.tex_procedural: Optional[moderngl.Texture] = None
        self.fbo_procedural: Optional[moderngl.Framebuffer] = None

        self.tex_proc_b: Optional[moderngl.Texture] = None
        self.fbo_proc_b: Optional[moderngl.Framebuffer] = None

        self.tex_mixed: Optional[moderngl.Texture] = None
        self.fbo_mixed: Optional[moderngl.Framebuffer] = None

        self.tex_edges: Optional[moderngl.Texture] = None
        self.fbo_edges: Optional[moderngl.Framebuffer] = None

        self.tex_ascii: Optional[moderngl.Texture] = None
        self.fbo_ascii: Optional[moderngl.Framebuffer] = None

        self.tex_paint: List[moderngl.Texture] = []
        self.fbo_paint: List[moderngl.Framebuffer] = []
        self._paint_idx: int = 0

        self.tex_trail: List[moderngl.Texture] = []
        self.fbo_trail: List[moderngl.Framebuffer] = []
        self._trail_idx: int = 0

        self.bloom_down_tex: List[moderngl.Texture] = []
        self.bloom_down_fbo: List[moderngl.Framebuffer] = []
        self.bloom_up_tex: List[moderngl.Texture] = []
        self.bloom_up_fbo: List[moderngl.Framebuffer] = []

        self.tex_final: Optional[moderngl.Texture] = None
        self.fbo_final: Optional[moderngl.Framebuffer] = None

        self._compile_all_shaders()
        self._allocate_targets(self.width, self.height)
        self.rebuild_atlas("ascii_detailed")

    def _compile_slot(self, name: str, frag_filename: str) -> None:
        vert_path = self.shaders_dir / "fullscreen.vert"
        frag_path = self.shaders_dir / frag_filename
        vert_src = vert_path.read_text(encoding="utf-8")
        frag_src = frag_path.read_text(encoding="utf-8")
        prog = self.ctx.program(vertex_shader=vert_src, fragment_shader=frag_src)
        vao = self.ctx.vertex_array(prog, [])
        old = self.slots.get(name)
        self.slots[name] = _ShaderSlot(
            name=name,
            vert_path=vert_path,
            frag_path=frag_path,
            program=prog,
            vao=vao,
            vert_mtime=vert_path.stat().st_mtime,
            frag_mtime=frag_path.stat().st_mtime,
        )
        if old is not None:
            try:
                old.vao.release()
                old.program.release()
            except Exception:
                pass

    def _compile_all_shaders(self) -> None:
        shader_map = {
            "procedural": "procedural_demo.frag",
            "mixer": "source_mixer.frag",
            "edge": "edge_detect.frag",
            "ascii": "ascii_hdr.frag",
            "paint": "paint_brush.frag",
            "trail": "trail_feedback.frag",
            "bloom_prefilter": "bloom_prefilter.frag",
            "bloom_down": "bloom_downsample.frag",
            "bloom_up": "bloom_upsample.frag",
            "composite": "composite.frag",
        }
        for key, fname in shader_map.items():
            self._compile_slot(key, fname)

    def check_hot_reload(self, dt: float = 0.0, force: bool = False) -> bool:
        if not force:
            self._reload_timer += dt
            if self._reload_timer < 0.35:
                return False
            self._reload_timer = 0.0

        reloaded: List[str] = []
        errors: List[str] = []
        for name, slot in list(self.slots.items()):
            try:
                vm = slot.vert_path.stat().st_mtime
                fm = slot.frag_path.stat().st_mtime
            except OSError:
                continue
            if force or vm != slot.vert_mtime or fm != slot.frag_mtime:
                try:
                    self._compile_slot(name, slot.frag_path.name)
                    reloaded.append(slot.frag_path.name)
                except Exception as exc:
                    slot.vert_mtime = vm
                    slot.frag_mtime = fm
                    errors.append(f"{slot.frag_path.name}: {exc}")

        if errors:
            self.last_reload_error = "\n".join(errors)
            self.last_reload_status = "Error en hot-reload (shader previo mantenido)"
            return False
        if reloaded:
            self.last_reload_error = ""
            self.last_reload_status = f"Hot-reload OK: {', '.join(reloaded)}"
            return True
        return False

    def _make_target(self, w: int, h: int, dtype: str = "f2") -> Tuple[moderngl.Texture, moderngl.Framebuffer]:
        tex = self.ctx.texture((max(2, w), max(2, h)), 4, dtype=dtype)
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        tex.repeat_x = False
        tex.repeat_y = False
        fbo = self.ctx.framebuffer(color_attachments=[tex])
        fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self._allocated_textures.append(tex)
        self._allocated_fbos.append(fbo)
        return tex, fbo

    def _release_targets(self) -> None:
        for fbo in self._allocated_fbos:
            try:
                fbo.release()
            except Exception:
                pass
        for tex in self._allocated_textures:
            try:
                tex.release()
            except Exception:
                pass
        self._allocated_fbos.clear()
        self._allocated_textures.clear()
        self.tex_paint.clear()
        self.fbo_paint.clear()
        self.tex_trail.clear()
        self.fbo_trail.clear()
        self.bloom_down_tex.clear()
        self.bloom_down_fbo.clear()
        self.bloom_up_tex.clear()
        self.bloom_up_fbo.clear()

    def _allocate_targets(self, width: int, height: int) -> None:
        self._release_targets()
        self.width = max(64, int(width))
        self.height = max(64, int(height))

        self.tex_procedural, self.fbo_procedural = self._make_target(self.width, self.height, "f2")
        self.tex_proc_b, self.fbo_proc_b = self._make_target(self.width, self.height, "f2")
        self.tex_mixed, self.fbo_mixed = self._make_target(self.width, self.height, "f1")
        self.tex_edges, self.fbo_edges = self._make_target(self.width, self.height, "f2")
        self.tex_ascii, self.fbo_ascii = self._make_target(self.width, self.height, "f2")

        for _ in range(2):
            tp, fp = self._make_target(self.width, self.height, "f2")
            self.tex_paint.append(tp)
            self.fbo_paint.append(fp)

            tt, ft = self._make_target(self.width, self.height, "f2")
            self.tex_trail.append(tt)
            self.fbo_trail.append(ft)

        mw = max(4, self.width // 2)
        mh = max(4, self.height // 2)
        for lvl in range(self.BLOOM_MIP_COUNT):
            dt_tex, dt_fbo = self._make_target(mw, mh, "f2")
            self.bloom_down_tex.append(dt_tex)
            self.bloom_down_fbo.append(dt_fbo)
            if lvl < self.BLOOM_MIP_COUNT - 1:
                ut_tex, ut_fbo = self._make_target(mw, mh, "f2")
                self.bloom_up_tex.append(ut_tex)
                self.bloom_up_fbo.append(ut_fbo)
            mw = max(4, mw // 2)
            mh = max(4, mh // 2)

        self.tex_final, self.fbo_final = self._make_target(self.width, self.height, "f1")
        if self.ctx.screen is not None:
            self.ctx.screen.use()

    def resize(self, width: int, height: int) -> None:
        w = max(64, int(width))
        h = max(64, int(height))
        if w == self.width and h == self.height:
            return
        self._allocate_targets(w, h)

    def clear_trails_and_paint(self, clear_trails: bool = True, clear_paint: bool = True) -> None:
        if clear_trails:
            for fbo in self.fbo_trail:
                fbo.use()
                self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        if clear_paint:
            for fbo in self.fbo_paint:
                fbo.use()
                self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        if self.ctx.screen is not None:
            self.ctx.screen.use()

    def rebuild_atlas(self, charset_id: str, custom_charset: str = " .:-=+*#%@", font_path: str = "") -> GlyphAtlasData:
        atlas = build_glyph_atlas(
            charset_id=charset_id,
            custom_charset=custom_charset,
            font_path=font_path,
            tile_w=32,
            tile_h=48,
        )
        if self.tex_atlas is not None:
            try:
                self.tex_atlas.release()
            except Exception:
                pass
        tex = self.ctx.texture((atlas.width, atlas.height), 1, data=atlas.pixels.tobytes(), dtype="f1")
        tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        tex.build_mipmaps()
        tex.repeat_x = False
        tex.repeat_y = False
        self.tex_atlas = tex
        self.atlas_data = atlas
        return atlas

    def upload_media_frame(self, rgb_uint8: np.ndarray) -> None:
        h, w = rgb_uint8.shape[:2]
        flipped = np.ascontiguousarray(rgb_uint8[::-1])
        if self.tex_media is None or self._media_size != (w, h):
            if self.tex_media is not None:
                try:
                    self.tex_media.release()
                except Exception:
                    pass
            self.tex_media = self.ctx.texture((w, h), 3, data=flipped.tobytes(), dtype="f1")
            self.tex_media.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.tex_media.repeat_x = False
            self.tex_media.repeat_y = False
            self._media_size = (w, h)
        else:
            self.tex_media.write(flipped.tobytes())

    def upload_media_b_frame(self, rgb_uint8: np.ndarray) -> None:
        h, w = rgb_uint8.shape[:2]
        flipped = np.ascontiguousarray(rgb_uint8[::-1])
        if self.tex_media_b is None or self._media_b_size != (w, h):
            if self.tex_media_b is not None:
                try:
                    self.tex_media_b.release()
                except Exception:
                    pass
            self.tex_media_b = self.ctx.texture((w, h), 3, data=flipped.tobytes(), dtype="f1")
            self.tex_media_b.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.tex_media_b.repeat_x = False
            self.tex_media_b.repeat_y = False
            self._media_b_size = (w, h)
        else:
            self.tex_media_b.write(flipped.tobytes())

    def upload_fft_spectrum(self, spectrum_64: Optional[np.ndarray]) -> None:
        if spectrum_64 is None:
            return
        arr = np.ascontiguousarray(spectrum_64[:64], dtype=np.float32)
        if len(arr) < 64:
            padded = np.zeros(64, dtype=np.float32)
            padded[: len(arr)] = arr
            arr = padded
        self.tex_fft.write(arr.tobytes())

    def _draw_quad(self, slot_name: str, fbo: moderngl.Framebuffer) -> None:
        fbo.use()
        self.ctx.viewport = (0, 0, fbo.width, fbo.height)
        self.slots[slot_name].vao.render(moderngl.TRIANGLES, vertices=3)

    def render(
        self,
        params: ShaderParams,
        interaction: InteractionState,
        time_sec: float,
        audio_vec4: Tuple[float, float, float, float],
        use_media_source: bool,
        procedural_scene: int = 0,
        export_mode: bool = False,
        audio_stems_vec4: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
        fft_spectrum: Optional[np.ndarray] = None,
        mod: Optional[VisualModulation] = None,
        mixer: Optional[VideoMixerState] = None,
        use_media_b: bool = False,
    ) -> int:
        """Executes the full 7-pass HDR pipeline with Dual-Deck A/B Art-Video Mixer & Musical Choreography."""
        self.ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE | moderngl.BLEND)

        if mod is None:
            mod = VisualModulation()
        if fft_spectrum is not None:
            self.upload_fft_spectrum(fft_spectrum)

        if interaction.clear_paint_requested:
            self.clear_trails_and_paint(clear_trails=False, clear_paint=True)
            interaction.clear_paint_requested = False

        # --- Pass 1a: Deck A Source Texture (Media / Live Screen / Procedural GPU) ---
        if use_media_source and self.tex_media is not None:
            src_tex_a = self.tex_media
            src_w, src_h = self._media_size
        else:
            p_slot = self.slots["procedural"]
            prog = p_slot.program
            self.tex_fft.use(0)
            _set_u(prog, "u_fft_tex", 0)
            _set_u(prog, "u_time", float(time_sec))
            _set_u(prog, "u_resolution", (float(self.width), float(self.height)))
            _set_u(prog, "u_scene", int(procedural_scene))
            _set_u(prog, "u_mouse", (float(interaction.mouse_uv[0]), float(interaction.mouse_uv[1])))
            _set_u(prog, "u_audio", audio_vec4)
            _set_u(prog, "u_audio_stems", audio_stems_vec4)
            self._draw_quad("procedural", self.fbo_procedural)
            src_tex_a = self.tex_procedural
            src_w, src_h = self.width, self.height

        # --- Pass 1b: Dual-Deck A/B Mixer + Deck A Pro Video Transforms & Luma Keying ---
        src_tex = src_tex_a
        if mixer is not None:
            eff_xfade = float(
                np.clip(
                    mixer.crossfade + mod.disp_pulse * mixer.crossfade_audio_mod * 0.48,
                    0.0,
                    1.0,
                )
            )
            needs_transform = (
                abs(mixer.a_scale - 1.0) > 0.002
                or abs(mixer.a_rot_deg) > 0.1
                or mixer.a_flip_x
                or mixer.a_flip_y
                or mixer.a_mirror != 0
                or mixer.a_fit_mode != 0
                or abs(mixer.pre_hue) > 0.002
                or abs(mixer.pre_contrast - 1.0) > 0.002
                or abs(mixer.pre_sat - 1.0) > 0.002
            )
            if eff_xfade > 0.001 or needs_transform:
                src_tex_b = src_tex_a
                if eff_xfade > 0.001:
                    if mixer.deck_b_mode == 1 and use_media_b and self.tex_media_b is not None:
                        src_tex_b = self.tex_media_b
                    else:
                        p_prog = self.slots["procedural"].program
                        self.tex_fft.use(0)
                        _set_u(p_prog, "u_fft_tex", 0)
                        _set_u(p_prog, "u_time", float(time_sec))
                        _set_u(p_prog, "u_resolution", (float(self.width), float(self.height)))
                        _set_u(p_prog, "u_scene", int(mixer.deck_b_scene))
                        _set_u(p_prog, "u_mouse", (float(interaction.mouse_uv[0]), float(interaction.mouse_uv[1])))
                        _set_u(p_prog, "u_audio", audio_vec4)
                        _set_u(p_prog, "u_audio_stems", audio_stems_vec4)
                        self._draw_quad("procedural", self.fbo_proc_b)
                        src_tex_b = self.tex_proc_b

                m_prog = self.slots["mixer"].program
                src_tex_a.use(0)
                src_tex_b.use(1)
                _set_u(m_prog, "u_deck_a", 0)
                _set_u(m_prog, "u_deck_b", 1)
                _set_u(m_prog, "u_resolution", (float(self.width), float(self.height)))
                _set_u(m_prog, "u_time", float(time_sec))
                _set_u(m_prog, "u_crossfade", eff_xfade)
                _set_u(m_prog, "u_blend_mode", int(mixer.blend_mode))
                _set_u(m_prog, "u_luma_thresh", float(mixer.luma_thresh))
                _set_u(m_prog, "u_luma_soft", float(mixer.luma_soft))
                _set_u(m_prog, "u_beat_phase", float(audio_stems_vec4[2]))
                _set_u(m_prog, "u_a_scale", float(mixer.a_scale))
                _set_u(m_prog, "u_a_rot", math.radians(float(mixer.a_rot_deg)))
                _set_u(m_prog, "u_a_flip", (1.0 if mixer.a_flip_x else 0.0, 1.0 if mixer.a_flip_y else 0.0))
                _set_u(m_prog, "u_a_mirror", int(mixer.a_mirror))
                _set_u(m_prog, "u_a_aspect", float(src_w) / max(float(src_h), 1.0))
                _set_u(m_prog, "u_a_fit_mode", int(mixer.a_fit_mode))
                _set_u(m_prog, "u_pre_hue", float(mixer.pre_hue))
                _set_u(m_prog, "u_pre_contrast", float(mixer.pre_contrast))
                _set_u(m_prog, "u_pre_sat", float(mixer.pre_sat))
                self._draw_quad("mixer", self.fbo_mixed)
                src_tex = self.tex_mixed
                src_w, src_h = self.width, self.height

        self._last_src_tex = src_tex

        # --- Pass 2: Acerola DoG + Sobel Structure Tensor Edge Map ---
        e_prog = self.slots["edge"].program
        src_tex.use(0)
        _set_u(e_prog, "u_source", 0)
        _set_u(e_prog, "u_texel_size", (1.0 / max(src_w, 1), 1.0 / max(src_h, 1)))
        _set_u(e_prog, "u_dog_sigma", float(params.dog_sigma))
        self._draw_quad("edge", self.fbo_edges)

        # --- Pass 3: Edge-Aware HDR ASCII / Stipple + Kinetic Scramble + Audio Choreography ---
        a_prog = self.slots["ascii"].program
        src_tex.use(0)
        self.tex_edges.use(1)
        self.tex_atlas.use(2)
        self.tex_fft.use(3)
        _set_u(a_prog, "u_source", 0)
        _set_u(a_prog, "u_edges", 1)
        _set_u(a_prog, "u_atlas", 2)
        _set_u(a_prog, "u_fft_tex", 3)
        _set_u(a_prog, "u_resolution", (float(self.width), float(self.height)))

        # Resolution-normalized cell & stipple size (reference 720p) -> switching 720p/900p/1080p NEVER jumps pixel size!
        res_scale = max(0.5, float(self.height) / 720.0)
        cell_mult = float(np.clip(mod.cell_scale_mult, 0.78, 1.25))
        eff_cell_w = max(4.0, float(params.cell_w) * res_scale * cell_mult)
        eff_cell_h = max(6.0, float(params.cell_h) * res_scale * cell_mult)
        eff_stipple = max(1.25, float(params.stipple_scale) * res_scale * (1.0 + (cell_mult - 1.0) * 0.55))
        _set_u(a_prog, "u_cell_size", (eff_cell_w, eff_cell_h))
        _set_u(a_prog, "u_num_ramp", int(self.atlas_data.num_ramp))
        _set_u(a_prog, "u_total_glyphs", int(self.atlas_data.total_glyphs))
        _set_u(a_prog, "u_render_mode", int(params.render_mode))

        _set_u(a_prog, "u_brightness", float(params.brightness))
        _set_u(a_prog, "u_contrast", float(params.contrast))
        _set_u(a_prog, "u_gamma", float(params.gamma))
        _set_u(a_prog, "u_saturation", float(params.saturation))
        _set_u(a_prog, "u_invert", 1 if params.invert else 0)
        _set_u(a_prog, "u_dither", float(params.dither))
        _set_u(a_prog, "u_bg_opacity", float(params.bg_opacity))
        _set_u(a_prog, "u_glyph_sharpness", float(params.glyph_sharpness))
        _set_u(a_prog, "u_black_cutoff", float(params.black_cutoff))
        _set_u(a_prog, "u_scramble_speed", float(params.scramble_speed + mod.scramble_speed_add))
        _set_u(a_prog, "u_scramble_amount", float(min(1.0, params.scramble_amount + mod.scramble_add)))
        _set_u(a_prog, "u_weight_contrast", float(params.weight_contrast))
        _set_u(a_prog, "u_stipple_scale", eff_stipple)
        _set_u(a_prog, "u_stipple_density", float(params.stipple_density))

        _set_u(a_prog, "u_edge_strength", float(params.edge_strength))
        _set_u(a_prog, "u_edge_threshold", float(params.edge_threshold))
        _set_u(a_prog, "u_edge_boost", float(params.edge_boost + mod.edge_boost_add))

        _set_u(a_prog, "u_color_mode", int(params.color_mode))
        _set_u(a_prog, "u_color_dark", tuple(float(c) for c in params.color_dark))
        _set_u(a_prog, "u_color_bright", tuple(float(c) for c in params.color_bright))
        _set_u(a_prog, "u_hdr_boost", float(params.hdr_boost))
        _set_u(a_prog, "u_hdr_power", float(params.hdr_power))
        _set_u(a_prog, "u_time", float(time_sec))
        _set_u(a_prog, "u_matrix_rain", float(params.matrix_rain + mod.matrix_rain_add))

        _set_u(a_prog, "u_mouse", (float(interaction.mouse_uv[0]), float(interaction.mouse_uv[1])))
        _set_u(a_prog, "u_mouse_active", 0.0 if export_mode else (1.0 if interaction.mouse_active else 0.0))
        _set_u(a_prog, "u_mouse_mode", 0 if export_mode else int(interaction.mode))
        _set_u(a_prog, "u_interact_radius", float(interaction.radius))
        _set_u(a_prog, "u_interact_strength", float(interaction.strength))
        ripples_arr, ripple_cnt = interaction.packed_ripples()
        _set_u(a_prog, "u_ripples", ripples_arr)
        _set_u(a_prog, "u_ripple_count", int(ripple_cnt))  # Allow audio shockwave ripples in export too!

        _set_u(a_prog, "u_audio", audio_vec4)
        _set_u(a_prog, "u_audio_stems", audio_stems_vec4)
        _set_u(a_prog, "u_audio_reactivity", float(params.audio_reactivity))
        _set_u(a_prog, "u_hue_shift", float(mod.hue_shift))
        _set_u(a_prog, "u_spatial_eq", float(mod.spatial_eq_amount))
        _set_u(a_prog, "u_glitch_tear", float(mod.glitch_tear))

        # 11-Mode Audio-Synced GPU Displacement Engine uniforms
        d_sync = float(params.disp_audio_sync) * float(params.audio_reactivity)
        eff_disp_str = float(params.disp_strength) * (1.0 + mod.disp_intensity_add * d_sync * 0.55)
        eff_disp_spd = float(params.disp_speed) + mod.disp_speed_add * d_sync * 0.45
        _set_u(a_prog, "u_disp_mode", int(params.disp_mode))
        _set_u(a_prog, "u_disp_strength", float(min(3.5, eff_disp_str)))
        _set_u(a_prog, "u_disp_speed", float(min(5.0, eff_disp_spd)))
        _set_u(a_prog, "u_disp_scale", float(params.disp_scale))
        _set_u(
            a_prog,
            "u_disp_audio",
            (
                float(mod.disp_pulse * d_sync),
                float(mod.disp_glitch * d_sync),
                float(mod.disp_swirl * d_sync),
                float(audio_stems_vec4[2]),
            ),
        )

        self._draw_quad("ascii", self.fbo_ascii)

        # --- Pass 4a: Interactive Emissive Brush Stroke Buffer (Ping-Pong) ---
        prev_paint_idx = self._paint_idx
        next_paint_idx = 1 - prev_paint_idx
        paint_prog = self.slots["paint"].program
        self.tex_paint[prev_paint_idx].use(0)
        _set_u(paint_prog, "u_prev_paint", 0)
        _set_u(paint_prog, "u_p0", (float(interaction.prev_mouse_uv[0]), float(interaction.prev_mouse_uv[1])))
        _set_u(paint_prog, "u_p1", (float(interaction.mouse_uv[0]), float(interaction.mouse_uv[1])))
        _set_u(paint_prog, "u_resolution", (float(self.width), float(self.height)))
        is_brushing = (
            (not export_mode)
            and interaction.mode == 5
            and interaction.mouse_active
            and interaction.mouse_down
            and (not interaction.dragging_split)
        )
        _set_u(paint_prog, "u_drawing", 1.0 if is_brushing else 0.0)
        _set_u(paint_prog, "u_brush_radius", float(interaction.brush_radius))
        _set_u(paint_prog, "u_brush_color", tuple(float(c) for c in interaction.brush_color))
        _set_u(paint_prog, "u_paint_decay", float(interaction.paint_decay))
        self._draw_quad("paint", self.fbo_paint[next_paint_idx])
        self._paint_idx = next_paint_idx

        # --- Pass 4b: Temporal Phosphor Trails + Audio Wind Dissolve Feedback (Ping-Pong) ---
        prev_trail_idx = self._trail_idx
        next_trail_idx = 1 - prev_trail_idx
        t_prog = self.slots["trail"].program
        self.tex_ascii.use(0)
        self.tex_trail[prev_trail_idx].use(1)
        self.tex_paint[self._paint_idx].use(2)
        _set_u(t_prog, "u_current", 0)
        _set_u(t_prog, "u_prev_trail", 1)
        _set_u(t_prog, "u_paint", 2)
        _set_u(t_prog, "u_trail_decay", float(min(0.935, params.trail_decay + mod.trail_add)))
        _set_u(t_prog, "u_trail_drift", float(params.trail_drift))
        _set_u(t_prog, "u_trail_drift_vec", (float(params.trail_drift_x + mod.wind_x_add), float(params.trail_drift_y + mod.wind_y_add)))
        _set_u(t_prog, "u_trail_dissolve", float(min(1.0, params.trail_dissolve + mod.dissolve_add)))
        _set_u(t_prog, "u_time", float(time_sec))
        self._draw_quad("trail", self.fbo_trail[next_trail_idx])
        self._trail_idx = next_trail_idx
        hdr_scene_tex = self.tex_trail[self._trail_idx]

        # --- Pass 5a: Bloom Prefilter ---
        pf_prog = self.slots["bloom_prefilter"].program
        hdr_scene_tex.use(0)
        _set_u(pf_prog, "u_scene", 0)
        _set_u(pf_prog, "u_threshold", float(params.bloom_threshold))
        _set_u(pf_prog, "u_soft_knee", float(params.bloom_soft_knee))
        _set_u(pf_prog, "u_clamp_max", 18.0)
        _set_u(pf_prog, "u_bloom_tint", tuple(float(c) for c in params.bloom_tint))
        self._draw_quad("bloom_prefilter", self.bloom_down_fbo[0])

        # --- Pass 5b: Bloom Downsample Chain (13-tap CoD:AW + Karis) ---
        dn_prog = self.slots["bloom_down"].program
        for lvl in range(1, self.BLOOM_MIP_COUNT):
            prev_tex = self.bloom_down_tex[lvl - 1]
            prev_tex.use(0)
            _set_u(dn_prog, "u_src", 0)
            _set_u(dn_prog, "u_texel_size", (1.0 / prev_tex.width, 1.0 / prev_tex.height))
            _set_u(dn_prog, "u_use_karis", 1 if lvl == 1 else 0)
            self._draw_quad("bloom_down", self.bloom_down_fbo[lvl])

        # --- Pass 5c: Bloom Upsample Chain (Tent 3x3 + Audio Anamorphic Flare) ---
        up_prog = self.slots["bloom_up"].program
        lower_tex = self.bloom_down_tex[self.BLOOM_MIP_COUNT - 1]
        eff_anamorphic = min(4.8, float(params.bloom_anamorphic + mod.anamorphic_add))
        for lvl in range(self.BLOOM_MIP_COUNT - 2, -1, -1):
            higher_tex = self.bloom_down_tex[lvl]
            lower_tex.use(0)
            higher_tex.use(1)
            _set_u(up_prog, "u_lower_mip", 0)
            _set_u(up_prog, "u_higher_mip", 1)
            _set_u(up_prog, "u_texel_size", (1.0 / lower_tex.width, 1.0 / lower_tex.height))
            _set_u(up_prog, "u_filter_radius", float(params.bloom_radius))
            _set_u(up_prog, "u_scatter", float(params.bloom_scatter))
            _set_u(up_prog, "u_anamorphic", eff_anamorphic)
            self._draw_quad("bloom_up", self.bloom_up_fbo[lvl])
            lower_tex = self.bloom_up_tex[lvl]
        bloom_final_tex = self.bloom_up_tex[0]

        # --- Pass 6: Final ACES Filmic Composite + Kick Zoom Punch + Flash Accent + Split-Screen ---
        c_prog = self.slots["composite"].program
        hdr_scene_tex.use(0)
        bloom_final_tex.use(1)
        src_tex.use(2)
        self.tex_edges.use(3)
        _set_u(c_prog, "u_scene_hdr", 0)
        _set_u(c_prog, "u_bloom", 1)
        _set_u(c_prog, "u_source", 2)
        _set_u(c_prog, "u_edges", 3)
        _set_u(c_prog, "u_resolution", (float(self.width), float(self.height)))
        _set_u(c_prog, "u_time", float(time_sec))

        _set_u(c_prog, "u_bloom_intensity", float(params.bloom_intensity * mod.bloom_mult))
        _set_u(c_prog, "u_exposure", float(params.exposure + mod.exposure_add))
        _set_u(c_prog, "u_tonemap_mode", int(params.tonemap_mode))

        _set_u(c_prog, "u_chromatic_aberration", float(params.chromatic_aberration + mod.ca_add))
        _set_u(c_prog, "u_scanlines", float(params.scanlines))
        _set_u(c_prog, "u_curvature", float(params.curvature))
        _set_u(c_prog, "u_vignette", float(params.vignette))
        _set_u(c_prog, "u_grain", float(params.grain))

        _set_u(c_prog, "u_audio", audio_vec4)
        _set_u(c_prog, "u_audio_reactivity", float(params.audio_reactivity))
        _set_u(c_prog, "u_flash_accent", float(mod.flash_accent))
        _set_u(c_prog, "u_glitch_tear", float(mod.glitch_tear))
        prism_amt = 0.0
        if params.disp_mode == 8:
            prism_amt = float(params.disp_strength) * (0.75 + mod.disp_pulse * 0.65)
        elif params.disp_mode == 9:
            prism_amt = float(params.disp_strength) * (0.35 + mod.disp_pulse * 0.45)
        _set_u(c_prog, "u_prism_split", float(min(2.5, prism_amt)))

        eff_zoom = (1.0 if export_mode else float(interaction.zoom)) * (1.0 + mod.zoom_punch)
        _set_u(c_prog, "u_zoom", eff_zoom)
        _set_u(c_prog, "u_pan", (0.0, 0.0) if export_mode else (float(interaction.pan_x), float(interaction.pan_y)))
        _set_u(c_prog, "u_split_enabled", 0 if export_mode else (1 if interaction.split_enabled else 0))
        _set_u(c_prog, "u_split_pos", float(interaction.split_pos))
        _set_u(c_prog, "u_split_mode", int(interaction.split_mode))

        self._draw_quad("composite", self.fbo_final)

        if self.ctx.screen is not None:
            self.ctx.screen.use()
        return int(self.tex_final.glo)

    def read_analysis_rgb(self) -> Optional[np.ndarray]:
        """Downsamples the active source texture to 112x64 on GPU and reads back in <0.1ms for CV blob tracking."""
        if self._last_src_tex is None:
            return None
        self.ctx.disable(moderngl.DEPTH_TEST | moderngl.CULL_FACE | moderngl.BLEND)
        self.fbo_analysis.use()
        self.ctx.viewport = (0, 0, self._analysis_size[0], self._analysis_size[1])
        self._last_src_tex.use(0)
        self._analysis_prog["u_src"].value = 0
        self._analysis_vao.render(moderngl.TRIANGLES, vertices=3)
        raw = self.fbo_analysis.read(components=3, alignment=1)
        if self.ctx.screen is not None:
            self.ctx.screen.use()
        return np.frombuffer(raw, dtype=np.uint8).reshape((self._analysis_size[1], self._analysis_size[0], 3))

    def read_final_rgb(self) -> np.ndarray:
        raw = self.fbo_final.read(components=3, alignment=1)
        arr = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3))
        return np.ascontiguousarray(arr[::-1])

    def release(self) -> None:
        self._release_targets()
        for t in (self.tex_atlas, self.tex_media, self.tex_media_b, self.tex_fft, self.tex_analysis, self.fbo_analysis, self._analysis_vao, self._analysis_prog):
            if t is not None:
                try:
                    t.release()
                except Exception:
                    pass
        for slot in self.slots.values():
            try:
                slot.vao.release()
                slot.program.release()
            except Exception:
                pass
        self.slots.clear()
