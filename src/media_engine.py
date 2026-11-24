from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import av
import numpy as np
from PIL import Image


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".gif", ".flv", ".wmv"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass
class MediaInfo:
    source_type: str = "procedural"  # "procedural" | "image" | "video"
    path: str = ""
    name: str = "Demo Procedural GPU"
    width: int = 1280
    height: int = 720
    fps: float = 60.0
    duration: float = 0.0
    has_audio: bool = False


class MediaEngine:
    """
    Handles loading and decoding Videos (PyAV), Images (Pillow), and built-in Procedural GPU mode.
    Provides smooth playback, accurate scrubbing, speed control, A-B looping, and background audio extraction.
    """

    def __init__(self, on_audio_ready: Optional[Callable[[Optional[np.ndarray], int], None]] = None) -> None:
        self.info = MediaInfo()
        self.procedural_scene: int = 0

        self.playing: bool = True
        self.speed: float = 1.0
        self.loop: bool = True
        self.ab_loop_enabled: bool = False
        self.loop_a: float = 0.0
        self.loop_b: float = 0.0
        self.current_time: float = 0.0

        self._on_audio_ready = on_audio_ready
        self._lock = threading.Lock()
        self._container: Optional[av.container.InputContainer] = None
        self._vstream: Optional[av.video.stream.VideoStream] = None
        self._frame_iter = None

        self._current_frame_rgb: Optional[np.ndarray] = None
        self._next_frame_rgb: Optional[np.ndarray] = None
        self._next_frame_pts: float = -1.0
        self._frame_dirty: bool = False
        self._audio_thread: Optional[threading.Thread] = None
        self.status_message: str = "Modo Procedural GPU activo — arrastra un video o imagen aquí"

    def set_procedural(self, scene_idx: int = 0) -> None:
        self.close_media()
        self.procedural_scene = scene_idx
        self.info = MediaInfo(
            source_type="procedural",
            name="Demo Procedural GPU",
            width=1280,
            height=720,
            fps=60.0,
            duration=0.0,
            has_audio=False,
        )
        self.current_time = 0.0
        self.playing = True
        if self._on_audio_ready:
            self._on_audio_ready(None, 44100)
        self.status_message = "Demo Procedural GPU activa"

    def load_file(self, filepath: str) -> bool:
        p = Path(filepath)
        if not p.is_file():
            self.status_message = f"Archivo no encontrado: {filepath}"
            return False

        ext = p.suffix.lower()
        if ext in IMAGE_EXTENSIONS:
            return self._load_image(p)
        elif ext in VIDEO_EXTENSIONS:
            return self._load_video(p)
        else:
            # Try opening as video via PyAV anyway
            if self._load_video(p):
                return True
            return self._load_image(p)

    def _load_image(self, p: Path) -> bool:
        try:
            img = Image.open(p).convert("RGB")
            # Cap ultra-massive images to 2560px max dimension to stay silky fast
            max_dim = 2560
            if max(img.width, img.height) > max_dim:
                img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
            arr = np.ascontiguousarray(np.asarray(img, dtype=np.uint8))

            self.close_media()
            with self._lock:
                self._current_frame_rgb = arr
                self._frame_dirty = True
            self.info = MediaInfo(
                source_type="image",
                path=str(p),
                name=p.name,
                width=int(arr.shape[1]),
                height=int(arr.shape[0]),
                fps=60.0,
                duration=0.0,
                has_audio=False,
            )
            self.current_time = 0.0
            self.playing = True
            if self._on_audio_ready:
                self._on_audio_ready(None, 44100)
            self.status_message = f"Imagen cargada: {p.name} ({arr.shape[1]}x{arr.shape[0]})"
            return True
        except Exception as exc:
            self.status_message = f"Error al cargar imagen: {exc}"
            return False

    def _load_video(self, p: Path) -> bool:
        try:
            container = av.open(str(p))
            if not container.streams.video:
                container.close()
                self.status_message = "El archivo no contiene pista de video"
                return False

            vstream = container.streams.video[0]
            vstream.thread_type = "AUTO"
            fps = float(vstream.average_rate) if vstream.average_rate else 30.0
            if fps <= 0 or fps > 240:
                fps = 30.0

            if vstream.duration and vstream.time_base:
                duration = float(vstream.duration * vstream.time_base)
            elif container.duration:
                duration = float(container.duration) / 1_000_000.0
            else:
                duration = 0.0

            has_audio = len(container.streams.audio) > 0
            width = int(vstream.codec_context.width)
            height = int(vstream.codec_context.height)

            self.close_media()
            with self._lock:
                self._container = container
                self._vstream = vstream
                self._frame_iter = container.decode(vstream)
                self._next_frame_rgb = None
                self._next_frame_pts = -1.0

            self.info = MediaInfo(
                source_type="video",
                path=str(p),
                name=p.name,
                width=width,
                height=height,
                fps=fps,
                duration=max(duration, 0.1),
                has_audio=has_audio,
            )
            self.current_time = 0.0
            self.loop_a = 0.0
            self.loop_b = self.info.duration
            self.ab_loop_enabled = False
            self.playing = True

            # Decode first frame immediately
            self._decode_first_frame()

            # Extract audio PCM asynchronously for playback + FFT
            if has_audio and self._on_audio_ready:
                self._extract_audio_async(str(p))
            elif self._on_audio_ready:
                self._on_audio_ready(None, 44100)

            self.status_message = f"Video cargado: {p.name} ({width}x{height} @ {fps:.1f} fps)"
            return True
        except Exception as exc:
            self.status_message = f"Error al abrir video: {exc}"
            return False

    def _extract_audio_async(self, filepath: str) -> None:
        def _worker():
            try:
                ac = av.open(filepath)
                if not ac.streams.audio:
                    ac.close()
                    return
                astream = ac.streams.audio[0]
                target_sr = 44100
                resampler = av.AudioResampler(format="fltp", layout="stereo", rate=target_sr)
                chunks: list[np.ndarray] = []
                # Cap at 15 minutes of stereo float32 (~300 MB) to keep RAM safe
                max_samples = target_sr * 60 * 15
                total_samples = 0
                for frame in ac.decode(astream):
                    out_frames = resampler.resample(frame)
                    for of in out_frames:
                        arr = of.to_ndarray()  # shape (channels, samples)
                        if arr.ndim == 2:
                            arr = arr.T  # (samples, channels)
                        else:
                            arr = np.stack([arr, arr], axis=-1)
                        chunks.append(arr.astype(np.float32, copy=False))
                        total_samples += len(arr)
                    if total_samples >= max_samples:
                        break
                ac.close()
                if chunks and self.info.path == filepath and self._on_audio_ready:
                    pcm = np.concatenate(chunks, axis=0)
                    self._on_audio_ready(pcm, target_sr)
            except Exception:
                if self._on_audio_ready:
                    self._on_audio_ready(None, 44100)

        self._audio_thread = threading.Thread(target=_worker, daemon=True)
        self._audio_thread.start()

    def _frame_to_rgb(self, frame: av.VideoFrame) -> tuple[np.ndarray, float]:
        rgb = frame.to_ndarray(format="rgb24")
        if frame.pts is not None and self._vstream is not None and self._vstream.time_base is not None:
            pts = float(frame.pts * self._vstream.time_base)
        else:
            pts = self.current_time
        return np.ascontiguousarray(rgb, dtype=np.uint8), pts

    def _decode_first_frame(self) -> None:
        with self._lock:
            if self._frame_iter is None:
                return
            try:
                frame = next(self._frame_iter)
                rgb, pts = self._frame_to_rgb(frame)
                self._current_frame_rgb = rgb
                self._frame_dirty = True
                self.current_time = max(0.0, pts)
                # Update actual dimensions in case container metadata differed
                self.info.width = int(rgb.shape[1])
                self.info.height = int(rgb.shape[0])
            except StopIteration:
                pass

    def seek(self, time_sec: float) -> None:
        if self.info.source_type != "video":
            return
        target = max(0.0, min(time_sec, max(self.info.duration - 0.01, 0.0)))
        with self._lock:
            if self._container is None or self._vstream is None:
                return
            try:
                tb = self._vstream.time_base
                ts = int(target / float(tb)) if tb else int(target * 1_000_000)
                self._container.seek(ts, stream=self._vstream, backward=True, any_frame=False)
                self._frame_iter = self._container.decode(self._vstream)
                self._next_frame_rgb = None
                self._next_frame_pts = -1.0

                last_rgb = None
                last_pts = target
                # Advance to nearest frame at or right before target (capped at 90 frames for speed)
                for _ in range(90):
                    try:
                        frame = next(self._frame_iter)
                    except StopIteration:
                        break
                    rgb, pts = self._frame_to_rgb(frame)
                    last_rgb = rgb
                    last_pts = pts
                    if pts >= target - (0.5 / max(self.info.fps, 1.0)):
                        break
                if last_rgb is not None:
                    self._current_frame_rgb = last_rgb
                    self._frame_dirty = True
                    self.current_time = max(0.0, last_pts)
                else:
                    self.current_time = target
            except Exception:
                self.current_time = target

    def step_frame(self, delta: int) -> None:
        if self.info.source_type != "video":
            return
        self.playing = False
        frame_dt = 1.0 / max(self.info.fps, 1.0)
        if delta > 0:
            with self._lock:
                if self._frame_iter is not None:
                    try:
                        frame = next(self._frame_iter)
                        rgb, pts = self._frame_to_rgb(frame)
                        self._current_frame_rgb = rgb
                        self._frame_dirty = True
                        self.current_time = pts
                        return
                    except StopIteration:
                        pass
            self.seek(0.0)
        else:
            self.seek(max(0.0, self.current_time + delta * frame_dt))

    def update(self, dt: float) -> None:
        if self.info.source_type != "video" or not self.playing:
            return

        self.current_time += dt * self.speed

        # Handle A-B loop or normal loop
        end_bound = self.loop_b if (self.ab_loop_enabled and self.loop_b > self.loop_a + 0.05) else self.info.duration
        start_bound = self.loop_a if self.ab_loop_enabled else 0.0

        if self.current_time >= end_bound:
            if self.loop or self.ab_loop_enabled:
                self.seek(start_bound)
                return
            else:
                self.current_time = end_bound
                self.playing = False
                return

        hit_eof = False
        with self._lock:
            if self._frame_iter is None:
                return
            # Catch up video decoder to current_time (drop at most 6 frames per tick)
            steps = 0
            while steps < 6:
                if self._next_frame_rgb is not None and self._next_frame_pts <= self.current_time:
                    self._current_frame_rgb = self._next_frame_rgb
                    self._frame_dirty = True
                    self._next_frame_rgb = None
                    self._next_frame_pts = -1.0
                elif self._next_frame_rgb is not None and self._next_frame_pts > self.current_time:
                    break

                try:
                    frame = next(self._frame_iter)
                    rgb, pts = self._frame_to_rgb(frame)
                    if pts <= self.current_time + (0.25 / max(self.info.fps, 1.0)):
                        self._current_frame_rgb = rgb
                        self._frame_dirty = True
                    else:
                        self._next_frame_rgb = rgb
                        self._next_frame_pts = pts
                        break
                except StopIteration:
                    self._next_frame_rgb = None
                    self._next_frame_pts = -1.0
                    hit_eof = True
                    if not self.loop:
                        self.playing = False
                    break
                steps += 1

        # Safe rewind check immediately on real EOF or when duration reached
        if self.info.source_type == "video" and self.playing:
            if (hit_eof or self.current_time >= self.info.duration - 0.02) and (self.loop or self.ab_loop_enabled):
                self.seek(start_bound)

    def consume_frame(self) -> Optional[np.ndarray]:
        """Returns latest RGB uint8 frame if updated since last upload, else None."""
        with self._lock:
            if not self._frame_dirty or self._current_frame_rgb is None:
                return None
            self._frame_dirty = False
            return self._current_frame_rgb

    def current_frame_snapshot(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._current_frame_rgb

    def close_media(self) -> None:
        with self._lock:
            if self._container is not None:
                try:
                    self._container.close()
                except Exception:
                    pass
            self._container = None
            self._vstream = None
            self._frame_iter = None
            self._current_frame_rgb = None
            self._next_frame_rgb = None
            self._frame_dirty = False
