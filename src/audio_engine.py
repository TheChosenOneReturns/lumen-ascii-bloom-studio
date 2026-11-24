from __future__ import annotations

import threading
from typing import List, Optional, Tuple

import numpy as np

from .music_intelligence import (
    AudioIntelligenceFrame,
    AudioRoutingMatrix,
    MusicalIntelligenceEngine,
    VisualModulation,
    compute_visual_modulation,
)

try:
    import sounddevice as sd
except Exception:
    sd = None

try:
    import pyaudiowpatch as pyaudio
except Exception:
    pyaudio = None


AUDIO_SOURCE_NAMES = [
    "Spotify / Audio del PC (WASAPI)",
    "Audio del Video",
    "Micrófono (En Vivo)",
    "Desactivado",
]


class AudioEngine:
    """
    Precision Real-Time Audio Capture + Musical Intelligence Engine (SciPy + Librosa):
      - Mode 0: Windows System / Spotify Audio via WASAPI Loopback (PyAudioWPatch)
      - Mode 1: Loaded Video Audio Track (synchronized PCM playback + MIR DSP)
      - Mode 2: Live Microphone input (sounddevice)
      - Mode 3: Off
    """

    def __init__(self, sample_rate: int = 44100, fft_size: int = 2048, num_bins: int = 64) -> None:
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.num_bins = num_bins

        self.mode: int = 0  # Default to Spotify / System WASAPI Loopback
        self.volume: float = 0.85
        self.muted: bool = False
        self.gain: float = 1.45
        self.smoothing: float = 0.65

        # Musical Intelligence & Choreography Routing
        self.mir = MusicalIntelligenceEngine(sample_rate=sample_rate, fft_size=fft_size, num_mel_bins=num_bins)
        self.intel: AudioIntelligenceFrame = self.mir.frame
        self.routing = AudioRoutingMatrix()
        self.mod = VisualModulation()
        self._auto_morph_cd = [0.0]

        # Classic outputs (now powered by AGC-normalized surgical stems!)
        self.low: float = 0.0
        self.mid: float = 0.0
        self.high: float = 0.0
        self.energy: float = 0.0
        self.rms: float = 0.0
        self.beat_triggered: bool = False
        self.spectrum: np.ndarray = np.zeros(num_bins, dtype=np.float32)
        self.waveform: np.ndarray = np.zeros(256, dtype=np.float32)
        self.envelope_history: np.ndarray = np.zeros(96, dtype=np.float32)
        self._env_accum: float = 0.0

        self._lock = threading.Lock()
        self._video_pcm_mono: Optional[np.ndarray] = None
        self._video_pcm_stereo: Optional[np.ndarray] = None
        self._video_sr: int = sample_rate
        self._play_sample_pos: float = 0.0
        self._is_video_playing: bool = False
        self._video_speed: float = 1.0

        # Live capture ring buffer (4x fft_size for clean 2048-sample overlap)
        self._live_ring = np.zeros(fft_size * 4, dtype=np.float32)
        self._live_write_idx = 0
        self._live_sr: int = sample_rate

        self._out_stream = None
        self._in_stream = None
        self._pa_instance = None
        self._pa_stream = None

        self.loopback_devices: List[Tuple[int, str]] = []
        self.selected_loopback_pos: int = 0
        self.auto_switch_wasapi: bool = True
        self._silent_watchdog: float = 0.0
        self._auto_probe_running: bool = False
        self.status_message: str = "Inicializando motor DSP..."
        self.refresh_loopback_devices()

    def auto_detect_active_loopback(self) -> None:
        """Non-blocking background scan of all WASAPI loopback outputs to auto-lock onto the one playing audio!"""
        if pyaudio is None or self._auto_probe_running or not self.loopback_devices:
            return
        self._auto_probe_running = True

        def _probe():
            best_pos = -1
            best_rms = 0.0006
            p = None
            try:
                p = pyaudio.PyAudio()
                for pos, (dev_idx, _) in enumerate(list(self.loopback_devices)):
                    chunks = []

                    def _cb(in_data, frame_count, time_info, status_flags):
                        chunks.append(np.frombuffer(in_data, dtype=np.float32))
                        return (None, pyaudio.paContinue)

                    try:
                        info = p.get_device_info_by_index(dev_idx)
                        ch = max(1, int(info.get("maxInputChannels", 2)))
                        sr = int(info.get("defaultSampleRate", 48000))
                        st = p.open(
                            format=pyaudio.paFloat32,
                            channels=ch,
                            rate=sr,
                            input=True,
                            input_device_index=dev_idx,
                            frames_per_buffer=512,
                            stream_callback=_cb,
                        )
                        st.start_stream()
                        import time as _t

                        _t.sleep(0.18)
                        st.stop_stream()
                        st.close()
                        if chunks:
                            arr = np.concatenate(chunks)
                            rms = float(np.sqrt(np.mean(arr * arr) + 1e-12))
                            if rms > best_rms:
                                best_rms = rms
                                best_pos = pos
                    except Exception:
                        continue
            except Exception:
                pass
            finally:
                if p is not None:
                    try:
                        p.terminate()
                    except Exception:
                        pass
                self._auto_probe_running = False

            if best_pos >= 0 and best_pos != self.selected_loopback_pos and self.mode == 0:
                self.select_loopback_device(best_pos)

        threading.Thread(target=_probe, daemon=True).start()

    def refresh_loopback_devices(self) -> None:
        self.loopback_devices = []
        self.selected_loopback_pos = 0
        if pyaudio is None:
            self.status_message = "PyAudioWPatch no disponible"
            return

        p = None
        try:
            p = pyaudio.PyAudio()
            wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
            default_out_idx = wasapi_info.get("defaultOutputDevice", -1)
            default_out_name = ""
            if default_out_idx >= 0:
                default_out_name = p.get_device_info_by_index(default_out_idx).get("name", "")

            for dev in p.get_loopback_device_info_generator():
                idx = int(dev["index"])
                name = str(dev["name"])
                self.loopback_devices.append((idx, name))
                if default_out_name and default_out_name in name:
                    self.selected_loopback_pos = len(self.loopback_devices) - 1
        except Exception as exc:
            self.status_message = f"WASAPI no disponible: {exc}"
        finally:
            if p is not None:
                try:
                    p.terminate()
                except Exception:
                    pass

    def select_loopback_device(self, pos: int) -> None:
        if 0 <= pos < len(self.loopback_devices):
            self.selected_loopback_pos = pos
            if self.mode == 0:
                self._stop_streams()
                self._ensure_streams()

    def tap_tempo(self) -> None:
        self.mir.tap_tempo()

    def trigger_manual_drop(self) -> None:
        self.mir.trigger_manual_drop()

    def set_video_audio(self, pcm_stereo: Optional[np.ndarray], sample_rate: int) -> None:
        with self._lock:
            if pcm_stereo is None or len(pcm_stereo) == 0:
                self._video_pcm_stereo = None
                self._video_pcm_mono = None
                self._video_sr = self.sample_rate
                self._play_sample_pos = 0.0
                return
            if pcm_stereo.ndim == 1:
                stereo = np.stack([pcm_stereo, pcm_stereo], axis=-1).astype(np.float32, copy=False)
                mono = pcm_stereo.astype(np.float32, copy=False)
            else:
                stereo = pcm_stereo[:, :2].astype(np.float32, copy=False)
                mono = stereo.mean(axis=1).astype(np.float32, copy=False)
            self._video_pcm_stereo = np.ascontiguousarray(stereo)
            self._video_pcm_mono = np.ascontiguousarray(mono)
            self._video_sr = int(sample_rate)
            self._play_sample_pos = 0.0
        self._ensure_streams()

    def sync_video_playback(self, time_sec: float, playing: bool, speed: float = 1.0) -> None:
        with self._lock:
            self._is_video_playing = playing
            self._video_speed = max(0.1, min(speed, 4.0))
            if self._video_pcm_stereo is not None:
                target_pos = time_sec * self._video_sr
                if abs(self._play_sample_pos - target_pos) > self._video_sr * 0.08 or not playing:
                    self._play_sample_pos = max(0.0, min(target_pos, float(len(self._video_pcm_stereo) - 1)))

    def set_mode(self, mode: int) -> None:
        if mode == self.mode:
            return
        self.mode = mode
        self._stop_streams()
        with self._lock:
            self._live_ring.fill(0.0)
            self._live_write_idx = 0
        self._ensure_streams()

    def _push_live_mono(self, mono: np.ndarray) -> None:
        n = len(mono)
        if n == 0:
            return
        ring_len = len(self._live_ring)
        with self._lock:
            if n >= ring_len:
                self._live_ring[:] = mono[-ring_len:]
                self._live_write_idx = 0
            else:
                end = self._live_write_idx + n
                if end <= ring_len:
                    self._live_ring[self._live_write_idx : end] = mono
                else:
                    first = ring_len - self._live_write_idx
                    self._live_ring[self._live_write_idx :] = mono[:first]
                    self._live_ring[: n - first] = mono[first:]
                self._live_write_idx = end % ring_len

    def _ensure_streams(self) -> None:
        if self.mode == 1 and sd is not None and self._video_pcm_stereo is not None and self._out_stream is None:
            try:
                sr = self._video_sr

                def _out_cb(outdata, frames, time_info, status):
                    outdata.fill(0.0)
                    with self._lock:
                        buf = self._video_pcm_stereo
                        if buf is None or not self._is_video_playing or self.muted:
                            return
                        n = len(buf)
                        if n == 0:
                            return
                        pos = self._play_sample_pos
                        speed = self._video_speed
                        if abs(speed - 1.0) < 0.01:
                            start = int(pos)
                            end = min(start + frames, n)
                            count = max(0, end - start)
                            if count > 0:
                                outdata[:count] = buf[start:end] * self.volume
                            self._play_sample_pos = float(end)
                        else:
                            idxs = pos + np.arange(frames, dtype=np.float64) * speed
                            valid = idxs < (n - 1)
                            if np.any(valid):
                                vi = idxs[valid].astype(np.int64)
                                outdata[valid] = buf[vi] * self.volume
                            self._play_sample_pos = float(min(pos + frames * speed, n - 1))

                self._out_stream = sd.OutputStream(
                    samplerate=sr,
                    channels=2,
                    dtype="float32",
                    blocksize=1024,
                    callback=_out_cb,
                )
                self._out_stream.start()
                self.status_message = f"VIDEO PCM ({sr} Hz) // {self.mir.dsp_backend}"
            except Exception as exc:
                self._out_stream = None
                self.status_message = f"SIN SALIDA VIDEO: {exc}"

        if self.mode == 0:
            if self._pa_stream is not None:
                return
            if pyaudio is None:
                self.status_message = "INSTALA PyAudioWPatch"
                return
            if not self.loopback_devices:
                self.refresh_loopback_devices()
            if not self.loopback_devices:
                self.status_message = "SIN DISPOSITIVO WASAPI LOOPBACK"
                return
            try:
                self._pa_instance = pyaudio.PyAudio()
                dev_idx, dev_name = self.loopback_devices[self.selected_loopback_pos]
                dev_info = self._pa_instance.get_device_info_by_index(dev_idx)
                channels = max(1, int(dev_info.get("maxInputChannels", 2)))
                rate = int(dev_info.get("defaultSampleRate", 48000))
                self._live_sr = rate

                def _pa_cb(in_data, frame_count, time_info, status_flags):
                    arr = np.frombuffer(in_data, dtype=np.float32)
                    if channels > 1 and len(arr) >= channels:
                        arr = arr.reshape(-1, channels).mean(axis=1)
                    self._push_live_mono(arr)
                    return (None, pyaudio.paContinue)

                self._pa_stream = self._pa_instance.open(
                    format=pyaudio.paFloat32,
                    channels=channels,
                    rate=rate,
                    input=True,
                    input_device_index=dev_idx,
                    frames_per_buffer=512,
                    stream_callback=_pa_cb,
                )
                self._pa_stream.start_stream()
                short_name = dev_name.replace("[Loopback]", "").strip()[:24]
                self.status_message = f"WASAPI LIVE // {short_name}"
            except Exception as exc:
                self._stop_streams()
                self.status_message = f"ERR WASAPI: {str(exc)[:34]}"

        elif self.mode == 2:
            if self._in_stream is not None or sd is None:
                return
            try:
                self._live_sr = self.sample_rate

                def _in_cb(indata, frames, time_info, status):
                    mono = indata[:, 0] if indata.ndim > 1 else indata
                    self._push_live_mono(mono)

                self._in_stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype="float32",
                    blocksize=512,
                    callback=_in_cb,
                )
                self._in_stream.start()
                self.status_message = f"MICROFONO LIVE // {self.mir.dsp_backend}"
            except Exception as exc:
                self._in_stream = None
                self.status_message = f"ERR MIC: {str(exc)[:34]}"
        elif self.mode == 3:
            self.status_message = "AUDIO DESACTIVADO"

    def _stop_streams(self) -> None:
        for attr in ("_out_stream", "_in_stream"):
            stream = getattr(self, attr)
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass
                setattr(self, attr, None)

        if self._pa_stream is not None:
            try:
                self._pa_stream.stop_stream()
                self._pa_stream.close()
            except Exception:
                pass
            self._pa_stream = None

        if self._pa_instance is not None:
            try:
                self._pa_instance.terminate()
            except Exception:
                pass
            self._pa_instance = None

    def _read_live_chunk(self) -> np.ndarray:
        with self._lock:
            w = self._live_write_idx
            ring = self._live_ring
            if w >= self.fft_size:
                return ring[w - self.fft_size : w].copy()
            return np.concatenate([ring[-(self.fft_size - w) :], ring[:w]])

    def analyze_video_at(self, time_sec: float, dt: float = 1.0 / 30.0) -> tuple[float, float, float, float]:
        """Deterministic offline MIR DSP step for FFmpeg video export."""
        with self._lock:
            buf = self._video_pcm_mono
            sr = self._video_sr
        if buf is None or len(buf) == 0:
            self.mod = VisualModulation()
            return (0.0, 0.0, 0.0, 0.0)
        idx = int(time_sec * sr)
        start = max(0, idx - self.fft_size // 2)
        end = min(len(buf), start + self.fft_size)
        if end <= start:
            return (0.0, 0.0, 0.0, 0.0)
        mel, wf, intel = self.mir.process(buf[start:end], sr, dt, user_gain=self.gain, smoothing=self.smoothing)
        self.spectrum = mel
        self.waveform = wf
        self.intel = intel
        self.mod = compute_visual_modulation(intel, self.routing, dt, self._auto_morph_cd)
        return self.as_vec4()

    def update(self, dt: float, video_time_sec: float, video_playing: bool) -> None:
        chunk = np.zeros(self.fft_size, dtype=np.float32)
        sr = self._live_sr

        if self.mode == 0:
            self._ensure_streams()
            chunk = self._read_live_chunk()
            sr = self._live_sr
        elif self.mode == 1:
            self._ensure_streams()
            with self._lock:
                buf = self._video_pcm_mono
                sr = self._video_sr
            if buf is not None and len(buf) > 0 and video_playing:
                idx = int(video_time_sec * sr)
                start = max(0, idx - self.fft_size // 2)
                end = min(len(buf), start + self.fft_size)
                if end > start:
                    chunk = buf[start:end]
        elif self.mode == 2:
            self._ensure_streams()
            chunk = self._read_live_chunk()
            sr = self._live_sr

        if self.mode == 3:
            self.spectrum *= 0.85
            self.waveform *= 0.80
            self.low = self.mid = self.high = self.energy = 0.0
            self.beat_triggered = False
            self.mod = VisualModulation()
            return

        mel, wf, intel = self.mir.process(
            chunk,
            sr,
            dt,
            user_gain=self.gain,
            smoothing=self.smoothing,
            drop_sensitivity=self.routing.drop_sensitivity,
            spring_bounce=self.routing.spring_bounce,
        )
        self.intel = intel

        # Auto-WASAPI Watchdog: if current loopback output stays dead-silent (>3.0s), scan for active audio output!
        if self.mode == 0 and self.auto_switch_wasapi and len(self.loopback_devices) > 1:
            if intel.rms_db < -65.0:
                self._silent_watchdog += dt
                if self._silent_watchdog >= 3.0:
                    self._silent_watchdog = 0.0
                    self.auto_detect_active_loopback()
            else:
                self._silent_watchdog = 0.0
        self.spectrum = self.spectrum * 0.25 + mel * 0.75
        self.waveform = self.waveform * 0.30 + wf * 0.70

        self.low = float(np.clip(intel.kick * 0.65 + intel.bass * 0.45, 0.0, 1.5))
        self.mid = float(np.clip(intel.vocals * 0.60 + intel.snare * 0.55, 0.0, 1.5))
        self.high = float(np.clip(intel.hihat * 0.90 + intel.spectral_centroid * 0.20, 0.0, 1.5))
        self.energy = intel.energy
        self.rms = float(np.sqrt(np.mean(wf * wf)))
        self.beat_triggered = intel.kick_trigger or intel.drop_trigger

        # Update live visual choreography modulation
        self.mod = compute_visual_modulation(intel, self.routing, dt, self._auto_morph_cd)

        # Rolling symmetric audio envelope history (~30 Hz scroll)
        self._env_accum += dt
        if self._env_accum >= 0.032:
            self._env_accum = 0.0
            self.envelope_history = np.roll(self.envelope_history, -1)
            env_val = max(intel.energy * 1.05, intel.kick * 0.95)
            self.envelope_history[-1] = float(np.clip(env_val, 0.04, 1.0))

    def has_video_audio(self) -> bool:
        return self._video_pcm_mono is not None and len(self._video_pcm_mono) > 0

    def as_vec4(self) -> tuple[float, float, float, float]:
        """(kick, bass, vocals, energy)"""
        return (
            float(self.intel.kick),
            float(self.intel.bass),
            float(self.intel.vocals),
            float(self.intel.energy),
        )

    def stems_vec4(self) -> tuple[float, float, float, float]:
        """(snare, hihat, beat_phase, drop_intensity)"""
        return (
            float(self.intel.snare),
            float(self.intel.hihat),
            float(self.intel.beat_phase),
            float(self.intel.drop_intensity),
        )

    def physics_vec4(self) -> tuple[float, float, float, float]:
        """(spring_kick, percussive_energy, harmonic_energy, spectral_rolloff)"""
        return (
            float(self.intel.spring_kick),
            float(self.intel.percussive_energy),
            float(self.intel.harmonic_energy),
            float(self.intel.spectral_rolloff),
        )

    def close(self) -> None:
        self._stop_streams()
