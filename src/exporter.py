from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import av
import numpy as np
from PIL import Image

from .audio_engine import AudioEngine
from .config import ShaderParams
from .gpu_pipeline import AsciiBloomPipeline
from .interaction import InteractionState
from .media_engine import MediaEngine


EXPORT_FORMATS = ["MP4 (H.264 Alta Calidad)", "GIF (Animado HD)", "WebM (VP9)"]


def save_png_snapshot(pipeline: AsciiBloomPipeline, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = pipeline.read_final_rgb()
    Image.fromarray(rgb, mode="RGB").save(output_path, format="PNG")
    return output_path


@dataclass
class ExportSettings:
    format_idx: int = 0          # 0=MP4, 1=GIF, 2=WebM
    fps: int = 30
    duration_sec: float = 6.0    # Used for Procedural or Image sources
    crf: int = 16                # MP4/WebM quality (lower = sharper)
    keep_audio: bool = True
    use_ab_range: bool = False


class SteppedVideoExporter:
    """
    Main-thread OpenGL-safe deterministic FFmpeg video/GIF exporter.
    Steps 1-2 frames per GUI tick so the UI remains 100% responsive with live preview & progress bar!
    """

    def __init__(self) -> None:
        self.active: bool = False
        self.output_path: Optional[Path] = None
        self.total_frames: int = 0
        self.current_frame: int = 0
        self.start_wall_time: float = 0.0
        self.status_message: str = ""
        self.error_message: str = ""

        self._proc: Optional[subprocess.Popen] = None
        self._container: Optional[av.container.InputContainer] = None
        self._frame_iter = None
        self._vstream = None
        self._source_type: str = "procedural"
        self._fps: float = 30.0
        self._start_time: float = 0.0
        self._end_time: float = 0.0
        self._width: int = 1280
        self._height: int = 720

    @staticmethod
    def ffmpeg_available() -> bool:
        return shutil.which("ffmpeg") is not None

    def start(
        self,
        output_path: Path,
        settings: ExportSettings,
        media: MediaEngine,
        pipeline: AsciiBloomPipeline,
    ) -> bool:
        if not self.ffmpeg_available():
            self.error_message = "ffmpeg no encontrado en el PATH"
            return False

        self.cancel()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure even dimensions for H.264/yuv420p compatibility
        w = pipeline.width - (pipeline.width % 2)
        h = pipeline.height - (pipeline.height % 2)
        if (w, h) != (pipeline.width, pipeline.height):
            pipeline.resize(w, h)
        self._width, self._height = w, h

        self._source_type = media.info.source_type
        self.output_path = output_path
        self.error_message = ""
        self.current_frame = 0
        self.start_wall_time = time.perf_counter()

        source_video_path: Optional[str] = None

        if self._source_type == "video" and media.info.path:
            source_video_path = media.info.path
            self._fps = max(1.0, min(media.info.fps, 120.0))
            if settings.use_ab_range and media.ab_loop_enabled and media.loop_b > media.loop_a + 0.1:
                self._start_time = media.loop_a
                self._end_time = media.loop_b
            else:
                self._start_time = 0.0
                self._end_time = max(media.info.duration, 0.1)
            self.total_frames = max(1, int(round((self._end_time - self._start_time) * self._fps)))

            try:
                self._container = av.open(source_video_path)
                self._vstream = self._container.streams.video[0]
                self._vstream.thread_type = "AUTO"
                if self._start_time > 0.05 and self._vstream.time_base:
                    ts = int(self._start_time / float(self._vstream.time_base))
                    self._container.seek(ts, stream=self._vstream, backward=True)
                self._frame_iter = self._container.decode(self._vstream)
            except Exception as exc:
                self.error_message = f"No se pudo abrir el video fuente: {exc}"
                return False
        else:
            self._fps = float(max(10, min(settings.fps, 60)))
            self._start_time = 0.0
            self._end_time = max(0.5, float(settings.duration_sec))
            self.total_frames = max(1, int(round(self._end_time * self._fps)))

        # Build FFmpeg command
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{w}x{h}",
            "-r",
            f"{self._fps:.4f}",
            "-i",
            "-",
        ]

        mux_audio = (
            settings.keep_audio
            and settings.format_idx in (0, 2)
            and source_video_path is not None
            and media.info.has_audio
        )
        if mux_audio:
            if self._start_time > 0.01:
                cmd += ["-ss", f"{self._start_time:.4f}"]
            cmd += ["-i", source_video_path]

        if settings.format_idx == 0:
            # MP4 H.264
            cmd += [
                "-map",
                "0:v:0",
            ]
            if mux_audio:
                cmd += ["-map", "1:a:0?", "-c:a", "aac", "-b:a", "192k", "-shortest"]
            cmd += [
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                str(int(settings.crf)),
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output_path),
            ]
        elif settings.format_idx == 1:
            # High-quality dithered GIF using palettegen + paletteuse
            gif_fps = min(self._fps, 24.0)
            gif_w = min(w, 960)
            vf = (
                f"fps={gif_fps:.2f},scale={gif_w}:-1:flags=lanczos,"
                "split[s0][s1];[s0]palettegen=max_colors=256:stats_mode=diff[p];"
                "[s1][p]paletteuse=dither=sierra2_4a"
            )
            cmd += ["-vf", vf, str(output_path)]
        else:
            # WebM VP9
            cmd += ["-map", "0:v:0"]
            if mux_audio:
                cmd += ["-map", "1:a:0?", "-c:a", "libopus", "-b:a", "128k", "-shortest"]
            cmd += [
                "-c:v",
                "libvpx-vp9",
                "-crf",
                str(int(settings.crf + 8)),
                "-b:v",
                "0",
                "-pix_fmt",
                "yuv420p",
                str(output_path),
            ]

        try:
            creationflags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=creationflags,
            )
        except Exception as exc:
            self._cleanup()
            self.error_message = f"Error iniciando FFmpeg: {exc}"
            return False

        pipeline.clear_trails_and_paint(clear_trails=True, clear_paint=False)
        self.active = True
        self.status_message = f"Exportando {output_path.name}..."
        return True

    def step(
        self,
        pipeline: AsciiBloomPipeline,
        params: ShaderParams,
        interaction: InteractionState,
        media: MediaEngine,
        audio: AudioEngine,
        frames_per_tick: int = 2,
        mixer=None,
        use_media_b: bool = False,
    ) -> None:
        if not self.active or self._proc is None or self._proc.stdin is None:
            return

        for _ in range(frames_per_tick):
            if self.current_frame >= self.total_frames:
                self.finish()
                return

            t = self._start_time + (self.current_frame / self._fps)
            use_media = self._source_type in ("video", "image")

            if self._source_type == "video":
                if self._frame_iter is None:
                    self.finish()
                    return
                frame_rgb = None
                while True:
                    try:
                        frame = next(self._frame_iter)
                    except StopIteration:
                        self.finish()
                        return
                    pts = float(frame.pts * self._vstream.time_base) if (frame.pts is not None and self._vstream.time_base) else t
                    if pts + (0.5 / self._fps) < self._start_time:
                        continue
                    if pts > self._end_time + (0.5 / self._fps):
                        self.finish()
                        return
                    frame_rgb = frame.to_ndarray(format="rgb24")
                    t = pts
                    break
                if frame_rgb is not None:
                    pipeline.upload_media_frame(frame_rgb)
                audio_vec = audio.analyze_video_at(t, dt=1.0 / max(self._fps, 1.0))
                stems_vec = audio.stems_vec4()
                fft_spec = audio.spectrum
                vis_mod = audio.mod
            else:
                audio_vec = (0.0, 0.0, 0.0, 0.0)
                stems_vec = (0.0, 0.0, 0.0, 0.0)
                fft_spec = None
                vis_mod = None

            pipeline.render(
                params=params,
                interaction=interaction,
                time_sec=t,
                audio_vec4=audio_vec,
                use_media_source=use_media,
                procedural_scene=media.procedural_scene,
                export_mode=True,
                audio_stems_vec4=stems_vec,
                fft_spectrum=fft_spec,
                mod=vis_mod,
                mixer=mixer,
                use_media_b=use_media_b,
            )
            rgb = pipeline.read_final_rgb()
            try:
                self._proc.stdin.write(rgb.tobytes())
            except BrokenPipeError:
                err = self._proc.stderr.read().decode("utf-8", errors="ignore") if self._proc.stderr else ""
                self.error_message = f"FFmpeg cerró el pipe: {err.strip()}"
                self.cancel()
                return

            self.current_frame += 1

    @property
    def progress(self) -> float:
        if self.total_frames <= 0:
            return 0.0
        return min(1.0, self.current_frame / float(self.total_frames))

    def finish(self) -> None:
        if not self.active:
            return
        err_text = ""
        ret = 0
        if self._proc is not None:
            try:
                if self._proc.stdin:
                    self._proc.stdin.close()
                _, stderr_bytes = self._proc.communicate(timeout=30)
                ret = self._proc.returncode
                if stderr_bytes:
                    err_text = stderr_bytes.decode("utf-8", errors="ignore").strip()
            except Exception as exc:
                err_text = str(exc)
                ret = 1
        self._cleanup()
        if ret == 0 and self.output_path is not None:
            elapsed = time.perf_counter() - self.start_wall_time
            self.status_message = f"Exportado OK ({elapsed:.1f}s): {self.output_path.name}"
            self.error_message = ""
        else:
            self.error_message = f"Error FFmpeg (code {ret}): {err_text}"
            self.status_message = "Error en exportación"

    def cancel(self) -> None:
        if self._proc is not None:
            try:
                if self._proc.stdin:
                    self._proc.stdin.close()
                self._proc.terminate()
                self._proc.wait(timeout=3)
            except Exception:
                pass
        self._cleanup()
        self.status_message = "Exportación cancelada"

    def _cleanup(self) -> None:
        self.active = False
        self._proc = None
        if self._container is not None:
            try:
                self._container.close()
            except Exception:
                pass
        self._container = None
        self._frame_iter = None
        self._vstream = None
