from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

try:
    from scipy import signal as sp_signal

    _SCIPY_AVAILABLE = True
except Exception:
    sp_signal = None
    _SCIPY_AVAILABLE = False

try:
    import librosa

    _LIBROSA_AVAILABLE = True
except Exception:
    librosa = None
    _LIBROSA_AVAILABLE = False


NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

SECTION_NAMES = ["SILENCE", "AMBIENT", "GROOVE", "BUILD-UP", "DROP!!"]

CHOREOGRAPHY_PROFILES = [
    "AUTO-DIRECTOR AI (Inteligente)",
    "CLUB KINETIC (Kick/Snare/HiHat)",
    "CHROMATIC HARMONY (Acordes->Color)",
    "KINETIC GLYPH STORM (Glifos+Grid)",
    "CRT GLITCH BREAKBEAT (Snare Tear)",
    "AMBIENT LIQUID BLOOM (Vocal/Suave)",
    "STROBE RAVE / MATRIX (Drops Max)",
    "WIND DISSOLVE SPECTRE (Viento)",
    "LIQUID DISPLACEMENT FLOW (Curl+Beat)",
    "KALEIDO PRISM HYPNOTIC (Simetria+RGB)",
]


class SpringDamper:
    """
    2nd-order Spring-Mass-Damper physics envelope (TouchDesigner / VJ inertia style):
    Converts snappy audio transients into organic, weighted visual motion with controlled overshoot!
    """

    def __init__(self, stiffness: float = 160.0, damping: float = 16.0) -> None:
        self.pos: float = 0.0
        self.vel: float = 0.0
        self.stiffness: float = stiffness
        self.damping: float = damping

    def impulse(self, strength: float) -> None:
        self.vel += float(strength)

    def step(self, target: float, dt: float, bounce: float = 0.75) -> float:
        # Higher bounce = livelier spring with organic elastic overshoot; lower bounce = critically damped smooth glide
        k = self.stiffness * (0.65 + bounce * 0.65)
        zeta = max(0.32, 1.05 - bounce * 0.42)
        c = 2.0 * zeta * math.sqrt(k)
        # 2 substeps for unconditional numerical stability even on 30 FPS spikes
        sub_dt = min(dt, 0.05) * 0.5
        for _ in range(2):
            acc = k * (target - self.pos) - c * self.vel
            self.vel += acc * sub_dt
            self.pos += self.vel * sub_dt
        self.pos = float(max(0.0, min(2.2, self.pos)))
        return self.pos


@dataclass
class AudioIntelligenceFrame:
    # Surgical per-stem envelopes (Parseval broadband + AGC normalized 0..1.5)
    kick: float = 0.0
    bass: float = 0.0
    snare: float = 0.0
    vocals: float = 0.0
    hihat: float = 0.0

    # Spring-Mass-Damper physical envelopes (organic weight + recoil)
    spring_kick: float = 0.0
    spring_bass: float = 0.0
    spring_drop: float = 0.0

    # Streaming HPSS & Psychoacoustic features
    energy: float = 0.0
    rms_db: float = -60.0
    agc_gain: float = 1.0
    spectral_centroid: float = 0.35
    spectral_flatness: float = 0.2
    spectral_rolloff: float = 0.4
    spectral_flux: float = 0.0
    harmonic_ratio: float = 0.8
    percussive_energy: float = 0.0
    harmonic_energy: float = 0.0

    # Rhythm, Tempo & Phase-Locked Beat Clock
    bpm: float = 120.0
    bpm_confidence: float = 0.0
    beat_phase: float = 0.0
    bar_phase: float = 0.0
    beat_lfo: float = 0.0
    beat_index: int = 0
    kick_trigger: bool = False
    snare_trigger: bool = False
    hihat_trigger: bool = False
    drop_trigger: bool = False

    # Musical Structure & 12-Semitone Pitch Chroma
    section_id: int = 0          # 0=SILENCE, 1=AMBIENT, 2=GROOVE, 3=BUILD-UP, 4=DROP!!
    section_name: str = "SILENCE"
    buildup_tension: float = 0.0
    drop_intensity: float = 0.0
    dominant_note: str = "--"
    pitch_hue: float = 0.0
    chroma: np.ndarray = field(default_factory=lambda: np.zeros(12, dtype=np.float32))


@dataclass
class AudioRoutingMatrix:
    """Manual fine-tuning weights layered cleanly on top of any Choreography Profile."""

    profile_idx: int = 0
    master_reactivity: float = 1.0
    drop_sensitivity: float = 1.25     # Adjustable Drop detection sensitivity (0.5 .. 2.0)
    spring_bounce: float = 0.78        # 2nd-order Spring-Damper bounce physics (0.0=smooth, 1.5=elastic)
    disp_react: float = 1.0            # Audio-to-Displacement modulation strength (0.0 .. 2.0)
    kick_punch: float = 1.0
    snare_glitch: float = 0.90
    hihat_spark: float = 0.85
    vocal_glow: float = 0.95
    pitch_color: float = 0.45
    spatial_eq: float = 0.45
    grid_breathe: float = 0.35
    wind_audio: float = 0.40
    shockwave_on_kick: bool = False
    shockwave_on_drop: bool = True
    auto_morph_on_drop: bool = False


@dataclass
class VisualModulation:
    bloom_mult: float = 1.0
    anamorphic_add: float = 0.0
    exposure_add: float = 0.0
    zoom_punch: float = 0.0
    cell_scale_mult: float = 1.0
    edge_boost_add: float = 0.0
    scramble_add: float = 0.0
    scramble_speed_add: float = 0.0
    glitch_tear: float = 0.0
    ca_add: float = 0.0
    hue_shift: float = 0.0
    spatial_eq_amount: float = 0.0
    wind_x_add: float = 0.0
    wind_y_add: float = 0.0
    dissolve_add: float = 0.0
    trail_add: float = 0.0
    flash_accent: float = 0.0
    matrix_rain_add: float = 0.0
    # Audio-driven GPU Displacement choreography outputs!
    disp_intensity_add: float = 0.0
    disp_speed_add: float = 0.0
    disp_pulse: float = 0.0
    disp_glitch: float = 0.0
    disp_swirl: float = 0.0
    spawn_ripple_strength: float = 0.0
    request_auto_morph: bool = False


class MusicalIntelligenceEngine:
    """
    Real-World Calibrated Audio MIR & DSP Engine (SciPy + Librosa):
      - True Parseval-calibrated broadband per-band RMS power (works identically on real Spotify songs & synths!)
      - Ultra-low real silence floor (-60 dB sustained > 0.45s -> ZERO false [SILENCE] during soft verses!)
      - Fast-adapting percentile AGC normalizer
      - 4th-order SciPy SOS Butterworth filterbank (Kick 28-115Hz, Snare 200-3kHz, Hi-Hat >6kHz)
      - Multi-scale real-music Drop Detector (Sub-Bass +5.5dB contrast vs lagged 2.5s baseline + Build-Up release!)
      - 64-band Librosa perceptual Mel spectrum + 12-semitone Chroma pitch tracker + Autocorrelation BPM PLL
    """

    def __init__(self, sample_rate: int = 44100, fft_size: int = 2048, num_mel_bins: int = 64) -> None:
        self.sample_rate = max(8000, int(sample_rate))
        self.fft_size = fft_size
        self.num_mel_bins = num_mel_bins

        self.frame = AudioIntelligenceFrame()
        self.dsp_backend: str = self._detect_backend_label()

        self._window = np.hanning(fft_size).astype(np.float32)
        self._win_power = float(np.sum(self._window * self._window))
        self._freqs = np.fft.rfftfreq(fft_size, 1.0 / self.sample_rate).astype(np.float32)
        self._prev_log_power = np.zeros(fft_size // 2 + 1, dtype=np.float32)

        self._build_filterbanks(self.sample_rate)

        # Rolling AGC loudness history (warm-starts dynamically on first active audio!)
        self._rms_hist: List[float] = []
        self._agc_smooth = 1.5
        self._active_time = 0.0
        self._silence_hold = 1.0  # Seconds continuously below -60 dB

        # Rolling per-band novelty histories for adaptive median peak picking (~0.45s)
        self._flux_win = 26
        self._kick_nov_hist = np.zeros(self._flux_win, dtype=np.float32)
        self._snare_nov_hist = np.zeros(self._flux_win, dtype=np.float32)
        self._hihat_nov_hist = np.zeros(self._flux_win, dtype=np.float32)
        self._flux_ptr = 0

        # Cooldowns
        self._kick_cd = 0.0
        self._snare_cd = 0.0
        self._hihat_cd = 0.0
        self._drop_cd = 0.0

        # Multi-scale loudness & sub-bass ring buffers at ~60 Hz (for rock-solid real-music DROP detection!)
        # Stores last 4.0 seconds (240 frames) so we compare NOW (last 0.2s) vs PRE-DROP (0.5s..3.2s ago)!
        self._hist_fps = 60.0
        self._sub_hist = np.zeros(240, dtype=np.float32)
        self._loud_hist = np.zeros(240, dtype=np.float32)
        self._cent_hist = np.zeros(240, dtype=np.float32)
        self._hist_primed = False
        self._hist_accum = 0.0
        self._manual_drop_requested = False

        # BPM Autocorrelation Onset Strength Envelope (7 seconds @ 60 Hz)
        self._ose_buf = np.zeros(420, dtype=np.float32)
        self._bpm_eval_timer = 0.0
        self._tap_times: List[float] = []
        self._manual_bpm_until = 0.0

        self._onset_rate_ema = 0.0
        self._hue_vec = np.array([1.0, 0.0], dtype=np.float32)

        # 2nd-Order Spring-Damper integrators for silky weighted musical physics
        self._spring_kick = SpringDamper(stiffness=190.0, damping=16.0)
        self._spring_bass = SpringDamper(stiffness=75.0, damping=12.0)
        self._spring_drop = SpringDamper(stiffness=130.0, damping=11.0)

    @staticmethod
    def _detect_backend_label() -> str:
        parts = []
        if _SCIPY_AVAILABLE:
            parts.append("SciPy-SOS")
        if _LIBROSA_AVAILABLE:
            parts.append("Librosa-MIR")
        return " + ".join(parts) if parts else "NumPy-FFT"

    def _build_filterbanks(self, sr: int) -> None:
        self.sample_rate = max(8000, int(sr))
        self._freqs = np.fft.rfftfreq(self.fft_size, 1.0 / self.sample_rate).astype(np.float32)
        n_fft_bins = self.fft_size // 2 + 1

        self._mask_kick = (self._freqs >= 28.0) & (self._freqs <= 120.0)
        self._mask_bass = (self._freqs >= 40.0) & (self._freqs <= 220.0)
        self._mask_snare = (self._freqs >= 190.0) & (self._freqs <= 3000.0)
        self._mask_vocal = (self._freqs >= 280.0) & (self._freqs <= 3600.0)
        self._mask_hihat = (self._freqs >= 5200.0) & (self._freqs <= min(16500.0, self.sample_rate * 0.46))

        if _LIBROSA_AVAILABLE:
            try:
                self._mel_fb = librosa.filters.mel(
                    sr=self.sample_rate,
                    n_fft=self.fft_size,
                    n_mels=self.num_mel_bins,
                    fmin=28.0,
                    fmax=min(16500.0, self.sample_rate * 0.46),
                ).astype(np.float32)
            except Exception:
                self._mel_fb = None
            try:
                self._chroma_fb = librosa.filters.chroma(
                    sr=self.sample_rate,
                    n_fft=self.fft_size,
                    n_chroma=12,
                ).astype(np.float32)
            except Exception:
                self._chroma_fb = None
        else:
            self._mel_fb = None
            self._chroma_fb = None

        # 4th-order SOS Butterworth time-domain filterbanks via SciPy
        self._sos_kick = None
        self._sos_snare = None
        self._sos_hihat = None
        if _SCIPY_AVAILABLE:
            try:
                nyq = self.sample_rate * 0.5
                self._sos_kick = sp_signal.butter(
                    4, [max(24.0 / nyq, 0.0005), min(120.0 / nyq, 0.45)], btype="bandpass", output="sos"
                )
                self._sos_snare = sp_signal.butter(
                    4, [max(220.0 / nyq, 0.002), min(2800.0 / nyq, 0.45)], btype="bandpass", output="sos"
                )
                self._sos_hihat = sp_signal.butter(
                    4, min(5800.0 / nyq, 0.45), btype="highpass", output="sos"
                )
            except Exception:
                self._sos_kick = self._sos_snare = self._sos_hihat = None

        edges = np.geomspace(1, n_fft_bins - 1, self.num_mel_bins + 1)
        self._fallback_edges = np.clip(np.round(edges).astype(np.int32), 1, n_fft_bins - 1)

    def tap_tempo(self) -> None:
        now = time.perf_counter()
        self._tap_times = [t for t in self._tap_times if now - t < 3.0]
        self._tap_times.append(now)
        if len(self._tap_times) >= 2:
            diffs = np.diff(self._tap_times[-6:])
            avg_dt = float(np.median(diffs))
            if 0.28 <= avg_dt <= 1.2:
                self.frame.bpm = float(np.clip(60.0 / avg_dt, 60.0, 190.0))
                self.frame.bpm_confidence = 1.0
                self.frame.beat_phase = 0.0
                self._manual_bpm_until = now + 15.0

    def trigger_manual_drop(self) -> None:
        """Lets the user fire a manual DROP burst live from the HUD."""
        self._manual_drop_requested = True

    def _band_rms_from_power(self, power_spec: np.ndarray, mask: np.ndarray) -> float:
        """Exact Parseval band RMS regardless of FFT size or broadband vs narrowband signal!"""
        if not np.any(mask):
            return 0.0
        # Factor of 2 for one-sided real FFT spectrum
        band_energy = float(np.sum(power_spec[mask])) * 2.0 / (self.fft_size * self._win_power)
        return math.sqrt(max(band_energy, 0.0))

    def process(
        self,
        chunk: np.ndarray,
        sample_rate: int,
        dt: float,
        user_gain: float = 1.4,
        smoothing: float = 0.62,
        drop_sensitivity: float = 1.25,
        spring_bounce: float = 0.78,
    ) -> Tuple[np.ndarray, np.ndarray, AudioIntelligenceFrame]:
        if abs(sample_rate - self.sample_rate) > 100 and sample_rate >= 8000:
            self._build_filterbanks(sample_rate)

        dt = min(max(dt, 1e-4), 0.1)
        self._kick_cd = max(0.0, self._kick_cd - dt)
        self._snare_cd = max(0.0, self._snare_cd - dt)
        self._hihat_cd = max(0.0, self._hihat_cd - dt)
        self._drop_cd = max(0.0, self._drop_cd - dt)

        if len(chunk) < self.fft_size:
            sig = np.zeros(self.fft_size, dtype=np.float32)
            sig[-len(chunk) :] = chunk
        else:
            sig = chunk[-self.fft_size :].astype(np.float32, copy=False)

        # --- 1. True Real-World Silence Floor & Fast-Adapting AGC ---
        raw_rms = float(np.sqrt(np.mean(sig * sig) + 1e-12))
        rms_db = float(20.0 * math.log10(max(raw_rms, 1e-6)))
        self.frame.rms_db = rms_db

        # Real audio floor: -62 dB (raw_rms < 0.0008). Must stay silent for >0.45s before declaring [SILENCE]!
        frame_is_quiet = raw_rms < 0.0008
        if frame_is_quiet:
            self._silence_hold = min(5.0, self._silence_hold + dt)
            self._active_time = max(0.0, self._active_time - dt * 0.5)
        else:
            self._silence_hold = 0.0
            self._active_time += dt

        is_truly_silent = self._silence_hold >= 0.45

        if not frame_is_quiet:
            self._rms_hist.append(raw_rms)
            if len(self._rms_hist) > 360:
                self._rms_hist.pop(0)
            ref_level = float(np.percentile(self._rms_hist, 82))
            # Normalize real music nicely around 0.24 target RMS (works from -45dB quiet Spotify to 0dB loud!)
            target_agc = float(np.clip(0.24 / max(ref_level, 0.0015), 0.5, 16.0))
        else:
            target_agc = 1.0

        agc_rate = 8.0 if target_agc < self._agc_smooth else 1.2
        self._agc_smooth += (target_agc - self._agc_smooth) * min(1.0, dt * agc_rate)
        effective_gain = self._agc_smooth * float(np.clip(user_gain, 0.25, 4.5))
        self.frame.agc_gain = self._agc_smooth

        norm_sig = np.clip(sig * effective_gain, -3.5, 3.5)
        wf = np.clip(norm_sig[-512::2] * 0.72, -1.0, 1.0).astype(np.float32)

        # --- 2. Parseval-Calibrated Power Spectrum & Log Spectral Flux ---
        fft_c = np.fft.rfft(norm_sig * self._window)
        power_spec = (fft_c.real * fft_c.real + fft_c.imag * fft_c.imag).astype(np.float32)
        mag_spec = np.sqrt(power_spec + 1e-12).astype(np.float32)

        # Perceptual log power scaled properly for broadband music
        norm_power = power_spec * (2.0 / (self.fft_size * self._win_power))
        log_power = np.log1p(norm_power * 4500.0)
        flux_vec = np.maximum(0.0, log_power - self._prev_log_power)
        self._prev_log_power = log_power

        # --- 3. 64-Band Perceptual Mel Spectrum (Parseval scaled -> lush active bars on real songs!) ---
        if self._mel_fb is not None:
            mel_energy = self._mel_fb @ norm_power
            mel_spec = np.clip(np.log1p(mel_energy * 260.0) / 3.4, 0.0, 1.5).astype(np.float32)
        else:
            mel_spec = np.zeros(self.num_mel_bins, dtype=np.float32)
            for i in range(self.num_mel_bins):
                a = int(self._fallback_edges[i])
                b = max(a + 1, int(self._fallback_edges[i + 1]))
                mel_spec[i] = float(np.clip(math.log1p(float(np.sum(norm_power[a:b])) * 180.0) / 3.4, 0.0, 1.5))

        # --- 4. Psychoacoustic Spectral Centroid, Flatness, Rolloff (85%) & Streaming HPSS ---
        mag_sum = float(np.sum(mag_spec) + 1e-6)
        centroid_hz = float(np.sum(self._freqs * mag_spec) / mag_sum)
        centroid_norm = float(np.clip(math.log2(max(centroid_hz, 80.0) / 80.0) / 7.0, 0.0, 1.0))

        # 85% Spectral Rolloff frequency (bright synth sweeps / shimmering highs push rolloff toward 1.0)
        cum_p = np.cumsum(norm_power)
        total_p = float(cum_p[-1]) if len(cum_p) > 0 else 0.0
        if total_p > 1e-10:
            roll_idx = int(np.searchsorted(cum_p, total_p * 0.85))
            roll_hz = float(self._freqs[min(roll_idx, len(self._freqs) - 1)])
            rolloff_norm = float(np.clip(math.log2(max(roll_hz, 120.0) / 120.0) / 7.2, 0.0, 1.0))
        else:
            rolloff_norm = 0.25

        mid_p = norm_power[self._mask_snare] + 1e-12
        geo_m = float(np.exp(np.mean(np.log(mid_p))))
        arith_m = float(np.mean(mid_p))
        flatness = float(np.clip(math.sqrt(geo_m / max(arith_m, 1e-12)), 0.0, 1.0))
        harmonic_ratio = 1.0 - flatness

        # Half-wave rectified broadband spectral flux -> real-time Percussive vs Harmonic separation!
        broad_flux = float(np.clip(np.mean(flux_vec) * 2.4, 0.0, 1.6)) if not is_truly_silent else 0.0
        self.frame.spectral_flux = self.frame.spectral_flux * 0.55 + broad_flux * 0.45
        self.frame.spectral_centroid = self.frame.spectral_centroid * 0.78 + centroid_norm * 0.22
        self.frame.spectral_rolloff = self.frame.spectral_rolloff * 0.78 + rolloff_norm * 0.22
        self.frame.spectral_flatness = self.frame.spectral_flatness * 0.72 + flatness * 0.28
        self.frame.harmonic_ratio = 1.0 - self.frame.spectral_flatness

        # --- 5. True Parseval 5-Stem Extraction (Calibrated for REAL Spotify Broadband Audio!) ---
        # Instantaneous band RMS levels (directly in normalized amplitude ~0.0 .. 1.0!)
        sub_rms = self._band_rms_from_power(power_spec, self._mask_kick)
        bass_rms = self._band_rms_from_power(power_spec, self._mask_bass)
        vocal_rms = self._band_rms_from_power(power_spec, self._mask_vocal)
        hihat_rms = self._band_rms_from_power(power_spec, self._mask_hihat)

        # Time-domain SciPy SOS Butterworth impact signals (last 512 samples ~10.6ms -> crisp transients!)
        tail = norm_sig[-512:]
        if self._sos_kick is not None and not is_truly_silent:
            k_td = float(np.sqrt(np.mean(sp_signal.sosfilt(self._sos_kick, tail) ** 2) + 1e-9))
            s_td = float(np.sqrt(np.mean(sp_signal.sosfilt(self._sos_snare, tail) ** 2) + 1e-9))
            h_td = float(np.sqrt(np.mean(sp_signal.sosfilt(self._sos_hihat, tail) ** 2) + 1e-9))
        else:
            k_td, s_td, h_td = sub_rms, vocal_rms, hihat_rms

        # Per-band spectral flux (top-quartile flux catches sharp hits without wide-band dilution!)
        kick_flux = float(np.percentile(flux_vec[self._mask_kick], 80)) if np.any(self._mask_kick) else 0.0
        snare_flux = float(np.percentile(flux_vec[self._mask_snare], 85)) if np.any(self._mask_snare) else 0.0
        hihat_flux = float(np.percentile(flux_vec[self._mask_hihat], 90)) if np.any(self._mask_hihat) else 0.0

        kick_nov = kick_flux * 0.55 + k_td * 2.2
        snare_nov = (snare_flux * 0.50 + s_td * 1.4) * (0.45 + 1.85 * flatness)
        hihat_nov = hihat_flux * 0.55 + h_td * 2.8

        # Push into adaptive median peak-picking ring buffers
        self._kick_nov_hist[self._flux_ptr] = kick_nov
        self._snare_nov_hist[self._flux_ptr] = snare_nov
        self._hihat_nov_hist[self._flux_ptr] = hihat_nov
        self._flux_ptr = (self._flux_ptr + 1) % self._flux_win

        k_med = float(np.median(self._kick_nov_hist))
        s_med = float(np.median(self._snare_nov_hist))
        h_med = float(np.median(self._hihat_nov_hist))

        k_thresh = k_med * 1.38 + 0.14
        s_thresh = s_med * 1.45 + 0.12
        h_thresh = h_med * 1.40 + 0.08

        kick_trig = (not is_truly_silent) and (kick_nov > k_thresh) and (self._kick_cd <= 0.0)
        snare_trig = (not is_truly_silent) and (snare_nov > s_thresh) and (self._snare_cd <= 0.0)
        hihat_trig = (not is_truly_silent) and (hihat_nov > h_thresh) and (self._hihat_cd <= 0.0)

        if kick_trig:
            self._kick_cd = 0.16
            k_amp = float(np.clip(0.55 + (kick_nov - k_med) * 1.15, 0.55, 1.45))
            self.frame.kick = max(self.frame.kick, k_amp)
        if snare_trig:
            self._snare_cd = 0.15
            s_amp = float(np.clip(0.50 + (snare_nov - s_med) * 1.25, 0.50, 1.40))
            self.frame.snare = max(self.frame.snare, s_amp)
        if hihat_trig:
            self._hihat_cd = 0.065
            h_amp = float(np.clip(0.45 + (hihat_nov - h_med) * 1.35, 0.45, 1.35))
            self.frame.hihat = max(self.frame.hihat, h_amp)

        # Also let sustained hihat/shimmer stay gently alive between crisp ticks
        decay_scale = 0.55 + 0.75 * (1.0 - float(np.clip(smoothing, 0.15, 0.94)))
        self.frame.kick = max(0.0, self.frame.kick * math.exp(-dt * 8.5 * decay_scale))
        self.frame.snare = max(0.0, self.frame.snare * math.exp(-dt * 10.5 * decay_scale))
        hihat_floor = float(np.clip(hihat_rms * 2.4, 0.0, 0.65)) if not is_truly_silent else 0.0
        self.frame.hihat = max(hihat_floor, self.frame.hihat * math.exp(-dt * 13.0 * decay_scale))

        # Parseval-calibrated sustained Bass & Vocals/Melody (NEVER stuck at 0.01!)
        s_alpha = 1.0 - float(np.clip(smoothing, 0.15, 0.92))
        if is_truly_silent:
            bass_target = 0.0
            vocal_target = 0.0
        else:
            bass_target = float(np.clip(bass_rms * 2.55, 0.0, 1.45))
            # Vocals/Melody uses Parseval vocal band RMS boosted by harmonic tonality
            vocal_target = float(np.clip(vocal_rms * 2.85 * (0.55 + 0.65 * harmonic_ratio), 0.0, 1.45))

        self.frame.bass = float(np.clip(self.frame.bass + (bass_target - self.frame.bass) * s_alpha, 0.0, 1.45))
        self.frame.vocals = float(np.clip(self.frame.vocals + (vocal_target - self.frame.vocals) * s_alpha, 0.0, 1.45))

        self.frame.kick_trigger = kick_trig
        self.frame.snare_trigger = snare_trig
        self.frame.hihat_trigger = hihat_trig

        # Global normalized energy combining all 5 stems cleanly
        if is_truly_silent:
            self.frame.energy = max(0.0, self.frame.energy * math.exp(-dt * 6.0))
        else:
            raw_energy = (
                self.frame.kick * 0.28
                + self.frame.bass * 0.26
                + self.frame.vocals * 0.22
                + self.frame.snare * 0.14
                + self.frame.hihat * 0.10
            )
            self.frame.energy = float(np.clip(self.frame.energy * 0.35 + raw_energy * 0.65, 0.0, 1.5))

        # --- 6. 12-Semitone Musical Chroma & Dominant Note (Librosa) ---
        if self._chroma_fb is not None and not is_truly_silent:
            chroma_raw = self._chroma_fb @ norm_power
            c_max = float(np.max(chroma_raw))
            chroma_norm = (chroma_raw / c_max).astype(np.float32) if c_max > 1e-9 else np.zeros(12, dtype=np.float32)
            self.frame.chroma = self.frame.chroma * 0.78 + chroma_norm * 0.22
            dom_idx = int(np.argmax(self.frame.chroma))
            self.frame.dominant_note = NOTE_NAMES[dom_idx]
            fifths_pos = (dom_idx * 7) % 12
            target_ang = (fifths_pos / 12.0) * 2.0 * math.pi
            self._hue_vec[0] = self._hue_vec[0] * 0.92 + math.cos(target_ang) * 0.08
            self._hue_vec[1] = self._hue_vec[1] * 0.92 + math.sin(target_ang) * 0.08
            self.frame.pitch_hue = (math.atan2(float(self._hue_vec[1]), float(self._hue_vec[0])) / (2.0 * math.pi)) % 1.0
        elif is_truly_silent:
            self.frame.chroma *= 0.85
            self.frame.dominant_note = "--"

        # --- 7. Real-Time Autocorrelation BPM Tracker + Phase-Locked Loop Beat Clock ---
        self._update_bpm_and_clock(dt, kick_nov + snare_nov * 0.40, kick_trig, is_truly_silent)

        # --- 8. Real-Music Multi-Scale Song Structure & DROP Detector ---
        self._update_song_structure(
            dt=dt,
            is_truly_silent=is_truly_silent,
            raw_rms=raw_rms,
            raw_sub_rms=sub_rms / max(effective_gain, 0.1),  # Raw un-AGC'd sub-bass for true dB contrast!
            inst_bass=bass_target,
            kick_trig=kick_trig,
            snare_trig=snare_trig,
            hihat_trig=hihat_trig,
            drop_sensitivity=drop_sensitivity,
        )

        # --- 9. 2nd-Order Spring-Damper Physics & Streaming HPSS Envelopes ---
        bounce = float(np.clip(spring_bounce, 0.0, 1.5))
        if kick_trig:
            self._spring_kick.impulse(4.8 * (0.45 + bounce * 0.55))
        if self.frame.drop_trigger:
            self._spring_drop.impulse(9.5 * (0.55 + bounce * 0.45))

        self.frame.spring_kick = self._spring_kick.step(self.frame.kick, dt, bounce=bounce)
        self.frame.spring_bass = self._spring_bass.step(self.frame.bass, dt, bounce=bounce * 0.65)
        self.frame.spring_drop = self._spring_drop.step(self.frame.drop_intensity, dt, bounce=bounce * 0.9)

        # Streaming HPSS: Percussive (transients + flux) vs Harmonic (tonal vocals + bass sustain)
        perc_raw = (
            self.frame.kick * 0.42
            + self.frame.snare * 0.35
            + self.frame.hihat * 0.22
            + self.frame.spectral_flux * 0.45
        )
        harm_raw = (self.frame.vocals * 0.60 + self.frame.bass * 0.45) * (0.40 + 0.75 * self.frame.harmonic_ratio)
        if is_truly_silent:
            self.frame.percussive_energy *= math.exp(-dt * 7.0)
            self.frame.harmonic_energy *= math.exp(-dt * 4.0)
        else:
            self.frame.percussive_energy = float(np.clip(self.frame.percussive_energy * 0.35 + perc_raw * 0.65, 0.0, 1.6))
            self.frame.harmonic_energy = float(np.clip(self.frame.harmonic_energy * 0.62 + harm_raw * 0.38, 0.0, 1.6))

        # Beat-locked smooth sinusoidal LFO synced strictly to the PLL BPM clock!
        self.frame.beat_lfo = float(math.sin(self.frame.bar_phase * 2.0 * math.pi))

        return mel_spec, wf, self.frame

    def _update_bpm_and_clock(self, dt: float, onset_strength: float, kick_trig: bool, is_silent: bool) -> None:
        self._hist_accum += dt
        step = 1.0 / self._hist_fps
        # Advance OSE buffer at 60 Hz
        n_steps = int(self._hist_accum / step)
        if n_steps > 0:
            self._hist_accum -= n_steps * step
            n_steps = min(n_steps, 6)
            self._ose_buf[:-n_steps] = self._ose_buf[n_steps:]
            self._ose_buf[-n_steps:] = 0.0 if is_silent else float(onset_strength)

        self._bpm_eval_timer += dt
        if self._bpm_eval_timer >= 0.45 and time.perf_counter() > self._manual_bpm_until:
            self._bpm_eval_timer = 0.0
            if not is_silent and float(np.std(self._ose_buf)) > 0.03:
                sig = self._ose_buf - float(np.mean(self._ose_buf))
                n = len(sig)
                f_sig = np.fft.rfft(sig, n=n * 2)
                ac = np.fft.irfft(f_sig * np.conj(f_sig))[:n]
                ac /= float(ac[0]) + 1e-6

                min_lag = max(1, int(round(60.0 * self._hist_fps / 175.0)))
                max_lag = min(n - 1, int(round(60.0 * self._hist_fps / 68.0)))
                if max_lag > min_lag + 2:
                    lags = np.arange(min_lag, max_lag, dtype=np.float32)
                    bpms = (60.0 * self._hist_fps) / lags
                    prior = np.exp(-0.5 * ((np.log2(bpms / 120.0)) / 0.52) ** 2)
                    scored = ac[min_lag:max_lag] * prior
                    best_i = int(np.argmax(scored))
                    best_lag = min_lag + best_i
                    conf = float(np.clip(ac[best_lag] * 2.2, 0.0, 1.0))

                    if conf > 0.20:
                        est_bpm = float(60.0 * self._hist_fps / best_lag)
                        if abs(est_bpm * 0.5 - self.frame.bpm) < 6.0:
                            est_bpm *= 0.5
                        elif abs(est_bpm * 2.0 - self.frame.bpm) < 6.0:
                            est_bpm *= 2.0
                        alpha = 0.16 if abs(est_bpm - self.frame.bpm) < 12.0 else 0.07
                        self.frame.bpm = float(self.frame.bpm * (1.0 - alpha) + est_bpm * alpha)
                    self.frame.bpm_confidence = float(self.frame.bpm_confidence * 0.7 + conf * 0.3)

        beats_per_sec = max(0.8, self.frame.bpm / 60.0)
        phase = self.frame.beat_phase + dt * beats_per_sec
        if kick_trig:
            err = phase - round(phase)
            if abs(err) < 0.28:
                phase -= err * 0.34
        if phase >= 1.0:
            wraps = int(math.floor(phase))
            phase -= wraps
            self.frame.beat_index = (self.frame.beat_index + wraps) % 4

        self.frame.beat_phase = float(phase % 1.0)
        self.frame.bar_phase = float((self.frame.beat_index + self.frame.beat_phase) / 4.0)

    def _update_song_structure(
        self,
        dt: float,
        is_truly_silent: bool,
        raw_rms: float,
        raw_sub_rms: float,
        inst_bass: float,
        kick_trig: bool,
        snare_trig: bool,
        hihat_trig: bool,
        drop_sensitivity: float,
    ) -> None:
        # Prime 4s history on first live non-silent frame to prevent false startup drops!
        if (not is_truly_silent) and (not self._hist_primed) and raw_rms > 0.001:
            self._sub_hist[:] = float(raw_sub_rms)
            self._loud_hist[:] = float(raw_rms)
            self._cent_hist[:] = float(self.frame.spectral_centroid)
            self._hist_primed = True

        # Push raw (pre-AGC!) sub-bass & loudness into 4-second sliding history (immune to AGC flattening drops!)
        self._sub_hist[:-1] = self._sub_hist[1:]
        self._sub_hist[-1] = 0.0 if is_truly_silent else float(raw_sub_rms)

        self._loud_hist[:-1] = self._loud_hist[1:]
        self._loud_hist[-1] = 0.0 if is_truly_silent else float(raw_rms)

        self._cent_hist[:-1] = self._cent_hist[1:]
        self._cent_hist[-1] = float(self.frame.spectral_centroid)

        hit_pulse = (1.0 if kick_trig else 0.0) + (0.85 if snare_trig else 0.0) + (0.45 if hihat_trig else 0.0)
        self._onset_rate_ema = self._onset_rate_ema * math.exp(-dt * 1.1) + hit_pulse * 0.42

        # Compare NOW peak envelope (last ~0.22s = 14 frames) vs LAGGED PRE-DROP BAR ENVELOPE (0.45s .. 2.2s ago)!
        # CRITICAL: Use mean/p82 (beat envelope), NEVER median (median picks silent gaps between kicks in steady grooves!)
        now_sub = float(np.percentile(self._sub_hist[-14:], 85))
        prev_sub = float(np.percentile(self._sub_hist[100:212], 82))
        now_loud = float(np.percentile(self._loud_hist[-14:], 85))
        prev_loud = float(np.percentile(self._loud_hist[100:212], 82))
        now_cent = float(np.mean(self._cent_hist[-20:]))
        prev_cent = float(np.mean(self._cent_hist[60:180]))

        # Sub-bass & loudness dB jump relative to previous bar's peak envelope (steady groove = ~0 dB!)
        sub_db_jump = 20.0 * math.log10((now_sub + 1e-4) / (prev_sub + 1e-4))
        loud_db_jump = 20.0 * math.log10((now_loud + 1e-4) / (prev_loud + 1e-4))

        # Build-up tension: rising brightness (riser synth) + dense snares/hihats while sub-bass is withheld!
        cent_rise = max(0.0, now_cent - prev_cent)
        sub_withheld = 1.0 if (now_sub < max(prev_sub * 0.55, 0.015) and now_loud > 0.004) else 0.0
        raw_tension = cent_rise * 3.0 + max(0.0, self._onset_rate_ema - 0.65) * 0.42 + sub_withheld * 0.35
        self.frame.buildup_tension = float(
            np.clip(self.frame.buildup_tension * math.exp(-dt * 0.85) + raw_tension * dt * 1.5, 0.0, 1.0)
        )

        # If song enters a quiet breakdown (sub-bass withheld), re-arm drop cooldown quickly so the upcoming DROP never misses!
        if sub_withheld > 0.5 and self._drop_cd > 0.4:
            self._drop_cd = max(0.35, self._drop_cd - dt * 2.5)

        # Real-Music DROP DETECTOR (calibrated via drop_sensitivity 0.5..2.2):
        sens = float(np.clip(drop_sensitivity, 0.5, 2.2))
        sub_thresh_db = 7.2 / sens      # Default ~5.8 dB sub-bass beat-envelope jump (steady groove is ~0-1 dB!)
        loud_thresh_db = 5.0 / sens     # Default ~4.0 dB full-band loudness beat-envelope jump

        drop_trig = False
        if self._manual_drop_requested:
            self._manual_drop_requested = False
            drop_trig = True
        elif (not is_truly_silent) and self._active_time > 1.5 and self._drop_cd <= 0.0:
            has_impact = kick_trig or (self.frame.kick > 0.62)
            # Signature 1: Classic EDM/Trap/Pop sub-bass explosion after vocal/riser/ambient break!
            sub_drop = has_impact and (sub_db_jump >= sub_thresh_db) and (inst_bass > 0.30)
            # Signature 2: Full-band chorus/drop surge (+loudness & strong sub jump)!
            chorus_drop = has_impact and (loud_db_jump >= loud_thresh_db) and (sub_db_jump >= sub_thresh_db * 0.60)
            # Signature 3: Build-up tension release into heavy kick + bass!
            buildup_release = (self.frame.buildup_tension > 0.48) and kick_trig and (inst_bass > 0.40) and (sub_db_jump > 3.2)
            if sub_drop or chorus_drop or buildup_release:
                drop_trig = True

        if drop_trig:
            self._drop_cd = 4.2
            self.frame.drop_intensity = 1.0
            self.frame.buildup_tension = 0.0

        self.frame.drop_trigger = drop_trig
        self.frame.drop_intensity = float(max(0.0, self.frame.drop_intensity * math.exp(-dt * 0.55)))

        # Classify current musical section (NEVER false [SILENCE] while music is actually playing!)
        if is_truly_silent:
            sid = 0  # SILENCE (only when raw RMS < -62 dB for > 0.45s!)
        elif self.frame.drop_intensity > 0.22:
            sid = 4  # DROP!!
        elif self.frame.buildup_tension > 0.44:
            sid = 3  # BUILD-UP
        elif self.frame.kick < 0.25 and self._onset_rate_ema < 0.30:
            sid = 1  # AMBIENT (soft vocal/pad passage without heavy percussion!)
        else:
            sid = 2  # GROOVE

        self.frame.section_id = sid
        self.frame.section_name = SECTION_NAMES[sid]


def compute_visual_modulation(
    intel: AudioIntelligenceFrame,
    routing: AudioRoutingMatrix,
    dt: float,
    auto_morph_cooldown_ref: list[float],
) -> VisualModulation:
    """
    Translates musical intelligence (stems, beat phase, pitch chroma, song sections)
    into rich, non-destructive multi-dimensional visual choreography!
    """
    mod = VisualModulation()
    auto_morph_cooldown_ref[0] = max(0.0, auto_morph_cooldown_ref[0] - dt)

    m = float(np.clip(routing.master_reactivity, 0.0, 2.5))
    if m <= 0.001 or intel.section_id == 0:
        return mod

    k_raw = intel.kick * routing.kick_punch * m
    # Spring-mass-damper weighted kick & bass envelopes for lush, organic non-twitchy visuals!
    k = (intel.spring_kick * 0.72 + intel.kick * 0.28) * routing.kick_punch * m
    b = (intel.spring_bass * 0.65 + intel.bass * 0.35) * m
    s = intel.snare * routing.snare_glitch * m
    v = intel.vocals * routing.vocal_glow * m
    h = intel.hihat * routing.hihat_spark * m
    perc = intel.percussive_energy * m
    harm = intel.harmonic_energy * m
    drop = max(intel.drop_intensity, intel.spring_drop * 0.85) * m
    tension = intel.buildup_tension * m
    beat_pulse = math.exp(-intel.beat_phase * 4.5) * m * min(1.0, intel.energy * 1.4)
    dr = float(np.clip(routing.disp_react, 0.0, 2.0))

    prof = routing.profile_idx

    if prof == 0:
        sid = intel.section_id
        if sid == 1:  # AMBIENT
            mod.bloom_mult = 1.0 + v * 0.48 + b * 0.25
            mod.edge_boost_add = v * 0.72
            mod.trail_add = min(0.32, v * 0.24 + b * 0.14)
            mod.hue_shift = (intel.pitch_hue - 0.5) * 0.45 * routing.pitch_color
            mod.spatial_eq_amount = routing.spatial_eq * 0.40 * m
        elif sid == 3:  # BUILD-UP
            mod.bloom_mult = 1.0 + tension * 0.65 + s * 0.35
            mod.zoom_punch = tension * 0.08 + k * 0.03
            mod.scramble_add = min(0.85, tension * 0.55 + h * 0.35)
            mod.scramble_speed_add = tension * 14.0 + h * 10.0
            mod.ca_add = tension * 0.004 + h * 0.004
            mod.glitch_tear = s * 0.45
            mod.edge_boost_add = tension * 0.8 + s * 0.5
            mod.spatial_eq_amount = routing.spatial_eq * 0.65 * m
        elif sid == 4:  # DROP!!
            mod.bloom_mult = 1.0 + k * 0.95 + drop * 0.90
            mod.anamorphic_add = drop * 2.1 + k * 0.95
            mod.zoom_punch = k * 0.11 + drop * 0.08
            mod.cell_scale_mult = 1.0 - min(0.28, k * 0.16 * routing.grid_breathe)
            mod.glitch_tear = min(1.25, s * 0.75 + drop * 0.48)
            mod.ca_add = s * 0.007 + h * 0.006 + drop * 0.006
            mod.scramble_add = min(0.95, h * 0.45 + drop * 0.35)
            mod.scramble_speed_add = h * 14.0 + drop * 8.0
            mod.edge_boost_add = v * 0.75 + s * 1.1 + drop * 0.75
            mod.hue_shift = (intel.pitch_hue - 0.5) * 0.65 * routing.pitch_color
            mod.spatial_eq_amount = routing.spatial_eq * 1.0 * m
            mod.wind_x_add = -k * 1.2 * routing.wind_audio
            mod.dissolve_add = s * 0.45 * routing.wind_audio
            mod.flash_accent = 0.28 if (intel.drop_trigger or (intel.kick_trigger and intel.beat_index == 0)) else 0.0
        else:  # GROOVE
            mod.bloom_mult = 1.0 + k * 0.68 + v * 0.30 + beat_pulse * 0.16
            mod.anamorphic_add = k * 0.65
            mod.zoom_punch = k * 0.065
            mod.cell_scale_mult = 1.0 - min(0.20, k * 0.11 * routing.grid_breathe)
            mod.glitch_tear = s * 0.50
            mod.ca_add = h * 0.0055 + s * 0.0035
            mod.scramble_add = h * 0.38
            mod.scramble_speed_add = h * 10.0
            mod.edge_boost_add = v * 0.78 + s * 0.65
            mod.hue_shift = (intel.pitch_hue - 0.5) * 0.50 * routing.pitch_color
            mod.spatial_eq_amount = routing.spatial_eq * 0.75 * m
            mod.wind_x_add = -k * 0.65 * routing.wind_audio

    elif prof == 1:
        mod.bloom_mult = 1.0 + k * 0.85 + s * 0.35 + beat_pulse * 0.22
        mod.anamorphic_add = k * 0.85
        mod.zoom_punch = k * 0.085
        mod.cell_scale_mult = 1.0 - min(0.22, k * 0.12 * routing.grid_breathe)
        mod.glitch_tear = s * 0.65
        mod.ca_add = h * 0.0065 + s * 0.004
        mod.scramble_add = h * 0.42 + s * 0.25
        mod.scramble_speed_add = h * 12.0
        mod.edge_boost_add = v * 0.85 + s * 0.75
        mod.spatial_eq_amount = routing.spatial_eq * 0.85 * m
        mod.hue_shift = (intel.pitch_hue - 0.5) * 0.30 * routing.pitch_color

    elif prof == 2:
        mod.bloom_mult = 1.0 + v * 0.55 + b * 0.45 + k * 0.30
        mod.hue_shift = (intel.pitch_hue - 0.5) * max(0.85, routing.pitch_color * 1.4)
        mod.edge_boost_add = v * 1.15
        mod.trail_add = min(0.35, v * 0.25 + b * 0.15)
        mod.zoom_punch = beat_pulse * 0.03
        mod.spatial_eq_amount = routing.spatial_eq * 0.65 * m
        mod.ca_add = h * 0.003

    elif prof == 3:
        mod.bloom_mult = 1.0 + k * 0.60 + h * 0.35
        mod.scramble_add = min(1.0, 0.22 + h * 0.65 + s * 0.45 + tension * 0.35)
        mod.scramble_speed_add = 6.0 + h * 18.0 + s * 10.0
        mod.cell_scale_mult = 1.0 + (s * 0.26 - k * 0.24) * max(0.5, routing.grid_breathe)
        mod.edge_boost_add = s * 1.1 + v * 0.5
        mod.ca_add = h * 0.006
        mod.spatial_eq_amount = routing.spatial_eq * 0.90 * m

    elif prof == 4:
        mod.bloom_mult = 1.0 + k * 0.72 + s * 0.42
        mod.anamorphic_add = k * 2.2 + drop * 1.5
        mod.glitch_tear = min(1.4, s * 1.15 + h * 0.25 + drop * 0.5)
        mod.ca_add = s * 0.011 + h * 0.006
        mod.zoom_punch = k * 0.07
        mod.edge_boost_add = s * 1.25
        mod.wind_x_add = -s * 1.4 * max(0.4, routing.wind_audio)
        mod.spatial_eq_amount = routing.spatial_eq * 0.55 * m

    elif prof == 5:
        mod.bloom_mult = 1.0 + b * 0.52 + v * 0.48
        mod.exposure_add = v * 0.14 + b * 0.08
        mod.trail_add = min(0.40, b * 0.25 + v * 0.22)
        mod.edge_boost_add = v * 0.80
        mod.hue_shift = (intel.pitch_hue - 0.5) * 0.45 * routing.pitch_color
        mod.zoom_punch = b * 0.025
        mod.spatial_eq_amount = routing.spatial_eq * 0.45 * m

    elif prof == 6:
        mod.bloom_mult = 1.0 + k * 1.15 + drop * 0.95 + s * 0.45
        mod.anamorphic_add = k * 1.6 + drop * 2.0
        mod.zoom_punch = k * 0.12 + drop * 0.08
        mod.glitch_tear = s * 0.85 + drop * 0.5
        mod.ca_add = h * 0.009 + s * 0.006
        mod.scramble_add = h * 0.55 + drop * 0.4
        mod.scramble_speed_add = h * 16.0
        mod.matrix_rain_add = min(1.2, h * 0.55 + drop * 0.65)
        mod.flash_accent = 0.35 if (intel.drop_trigger or (intel.kick_trigger and intel.beat_index == 0 and intel.kick > 0.85)) else 0.0
        mod.spatial_eq_amount = routing.spatial_eq * 1.1 * m

    elif prof == 7:
        mod.bloom_mult = 1.0 + k * 0.65 + v * 0.40
        mod.wind_x_add = -(0.55 + k * 2.2 + b * 0.9) * max(0.6, routing.wind_audio)
        mod.wind_y_add = (intel.spectral_centroid - 0.45) * 1.2 * max(0.5, routing.wind_audio)
        mod.dissolve_add = min(0.95, s * 0.75 + h * 0.45 + drop * 0.5)
        mod.trail_add = min(0.36, 0.14 + b * 0.22)
        mod.edge_boost_add = v * 0.95 + s * 0.6
        mod.ca_add = h * 0.005

    elif prof == 8:
        # LIQUID DISPLACEMENT FLOW (Harmonic Curl + Spring-Physics Bass Waves)
        mod.bloom_mult = 1.0 + harm * 0.55 + k * 0.55 + drop * 0.5
        mod.zoom_punch = k * 0.055 + b * 0.03
        mod.hue_shift = (intel.pitch_hue - 0.5) * 0.55 * routing.pitch_color
        mod.edge_boost_add = harm * 0.90 + s * 0.55
        mod.trail_add = min(0.32, harm * 0.20 + b * 0.12)
        mod.disp_intensity_add = (0.35 + harm * 0.65 + k * 0.55 + drop * 0.75) * dr
        mod.disp_speed_add = (0.25 + b * 0.60 + perc * 0.45) * dr
        mod.disp_pulse = (k * 0.95 + drop * 0.85) * dr
        mod.disp_swirl = (harm * 0.85 + intel.beat_lfo * 0.35) * dr
        mod.spatial_eq_amount = routing.spatial_eq * 0.70 * m

    else:
        # KALEIDO PRISM HYPNOTIC (Beat-Locked Symmetry + RGB Prism Split + Drops)
        mod.bloom_mult = 1.0 + k * 0.85 + s * 0.40 + drop * 0.85
        mod.anamorphic_add = k * 1.3 + drop * 1.8
        mod.zoom_punch = k * 0.09 + drop * 0.07
        mod.hue_shift = (intel.pitch_hue - 0.5) * 0.75 * max(0.5, routing.pitch_color)
        mod.ca_add = (0.003 + perc * 0.009 + drop * 0.008)
        mod.glitch_tear = s * 0.65 + drop * 0.45
        mod.disp_intensity_add = (0.45 + perc * 0.70 + drop * 0.95) * dr
        mod.disp_speed_add = (0.35 + k * 0.85 + intel.spectral_rolloff * 0.60) * dr
        mod.disp_pulse = (k * 1.15 + drop * 1.1) * dr
        mod.disp_glitch = (s * 0.90 + intel.spectral_flatness * perc * 0.65) * dr
        mod.disp_swirl = (harm * 0.75 + beat_pulse * 0.45) * dr
        mod.spatial_eq_amount = routing.spatial_eq * 0.95 * m

    # Universal baseline audio-to-displacement coupling across ALL profiles so displacement always breathes with music!
    if prof < 8 and dr > 0.01:
        mod.disp_intensity_add = max(mod.disp_intensity_add, (k * 0.55 + harm * 0.35 + drop * 0.80) * dr)
        mod.disp_speed_add = max(mod.disp_speed_add, (b * 0.45 + perc * 0.35) * dr)
        mod.disp_pulse = max(mod.disp_pulse, (k * 0.85 + drop * 0.90) * dr)
        mod.disp_glitch = max(mod.disp_glitch, (s * 0.75 + intel.spectral_flatness * perc * 0.55) * dr)
        mod.disp_swirl = max(mod.disp_swirl, (harm * 0.60 + b * 0.35) * dr)

    if routing.shockwave_on_drop and intel.drop_trigger:
        mod.spawn_ripple_strength = 1.55
    elif routing.shockwave_on_kick and intel.kick_trigger and k_raw > 0.62:
        mod.spawn_ripple_strength = float(np.clip(intel.kick * 0.95, 0.55, 1.25))

    if routing.auto_morph_on_drop and intel.drop_trigger and auto_morph_cooldown_ref[0] <= 0.0:
        mod.request_auto_morph = True
        auto_morph_cooldown_ref[0] = 10.0

    return mod
