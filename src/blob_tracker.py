from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull


BLOB_DETECT_MODES = [
    "Híbrido (Luz + Movimiento)",
    "Zonas Luminosas (Highlights)",
    "Movimiento Real (Motion Diff)",
    "Bordes / Alto Contraste",
]

BLOB_VISUAL_STYLES = [
    "TouchDesigner Plexus",
    "Swiss Minimal Hairline",
    "Organic Contour / Aura",
    "Constellation Nodes",
    "Cyber Targeting Pro",
]


@dataclass
class BlobSettings:
    enabled: bool = True
    detect_mode: int = 0          # 0=Hybrid, 1=Luminance, 2=Motion, 3=Edges
    visual_style: int = 0         # 0=TD Plexus, 1=Swiss, 2=Organic Contour, 3=Constellation, 4=Cyber Pro
    max_blobs: int = 10
    sensitivity: float = 0.65     # Higher = catches more subtle highlights/motion (0.15 .. 0.95)
    min_area: float = 0.0012      # Minimum normalized blob area (catches crisp mid-small highlights)
    max_area: float = 0.075       # Oversized regions auto-split around local peaks instead of 1 giant box!
    smoothing: float = 0.72       # Kalman / EMA fluidity (0.2 = snappy, 0.92 = ultra-silky)
    plexus_enabled: bool = True
    plexus_distance: float = 0.34 # Max normalized distance to connect two blobs
    plexus_curved: bool = True    # Elegant quadratic Bezier arcs vs crisp straight lines
    show_trails: bool = True      # Silky motion trajectory ribbons
    show_labels: bool = True      # Minimal Swiss ID / coordinate micro-tags
    audio_reactive: bool = True   # Blobs & plexus glow/pulse to kicks & beats


@dataclass
class _Detection:
    cx: float
    cy: float
    w: float
    h: float
    area: float
    intensity: float
    hull: List[Tuple[float, float]]  # Normalized (u, v) in [0, 1] bottom-up OpenGL coords


@dataclass
class TrackedBlob:
    track_id: int
    # Kalman-smoothed normalized state (u, v in [0, 1], OpenGL bottom-up origin!)
    cx: float
    cy: float
    vx: float = 0.0
    vy: float = 0.0
    w: float = 0.08
    h: float = 0.08
    area: float = 0.006
    intensity: float = 0.8
    alpha: float = 0.0            # Smooth fade-in / fade-out [0..1]
    age: float = 0.0
    missed: int = 0
    trail: List[Tuple[float, float]] = field(default_factory=list)
    hull: List[Tuple[float, float]] = field(default_factory=list)


def _chaikin_smooth_closed(pts: List[Tuple[float, float]], iterations: int = 2) -> List[Tuple[float, float]]:
    """Corner-cutting Chaikin subdivision for silky organic contour loops."""
    if len(pts) < 3:
        return pts
    cur = list(pts)
    for _ in range(iterations):
        nxt: List[Tuple[float, float]] = []
        n = len(cur)
        for i in range(n):
            p0 = cur[i]
            p1 = cur[(i + 1) % n]
            nxt.append((p0[0] * 0.75 + p1[0] * 0.25, p0[1] * 0.75 + p1[1] * 0.25))
            nxt.append((p0[0] * 0.25 + p1[0] * 0.75, p0[1] * 0.25 + p1[1] * 0.75))
        cur = nxt
    return cur


class IntelligentBlobTracker:
    """
    TouchDesigner-inspired Real-Time Computer Vision Multi-Blob Tracker:
      - Runs on a tiny 160x90 grayscale/RGB frame (< 1.2ms per frame!)
      - Supports Luminance Highlights, Optical Motion Diff, Sobel Edges, or Smart Hybrid
      - Adaptive percentile thresholding + morphological cleanup + connected components (scipy.ndimage)
      - Persistent ID matching via Hungarian Algorithm (scipy.optimize.linear_sum_assignment)
      - Constant-velocity Kalman prediction + smooth fade-in/out + organic convex-hull contour extraction
      - Computes proximity constellation plexus edges with smooth distance weights!
    """

    def __init__(self) -> None:
        self.settings = BlobSettings()
        self.tracks: List[TrackedBlob] = []
        self.plexus_edges: List[Tuple[TrackedBlob, TrackedBlob, float]] = []
        self._next_id: int = 1
        self._prev_luma: np.ndarray | None = None
        self._bg_luma: np.ndarray | None = None
        self._analyze_accum: float = 0.0

    def reset(self) -> None:
        self.tracks.clear()
        self.plexus_edges.clear()
        self._prev_luma = None
        self._bg_luma = None

    def _build_detection(
        self,
        sub_mask: np.ndarray,
        sub_feat: np.ndarray,
        y0: int,
        x0: int,
        h: int,
        w: int,
        total_px: float,
    ) -> _Detection | None:
        area_px = float(np.count_nonzero(sub_mask))
        if area_px < 4.0:
            return None
        weights = np.where(sub_mask, np.maximum(sub_feat, 0.01), 0.0)
        w_sum = float(np.sum(weights))
        if w_sum <= 1e-6:
            return None

        sh, sw = sub_mask.shape
        yy, xx = np.mgrid[0:sh, 0:sw]
        cx = float(np.sum((xx + x0) * weights) / w_sum) / float(w)
        cy = float(np.sum((yy + y0) * weights) / w_sum) / float(h)

        pts_y, pts_x = np.nonzero(sub_mask)
        min_x, max_x = int(pts_x.min()), int(pts_x.max()) + 1
        min_y, max_y = int(pts_y.min()), int(pts_y.max()) + 1
        bw = max(0.032, float(max_x - min_x) / float(w))
        bh = max(0.038, float(max_y - min_y) / float(h))
        intensity = float(np.max(sub_feat[sub_mask]))

        # Smooth organic convex hull contour for Organic Contour style
        hull_pts: List[Tuple[float, float]] = []
        if self.settings.visual_style == 2 and len(pts_x) >= 6:
            px_s, py_s = pts_x, pts_y
            if len(px_s) > 96:
                step = max(1, len(px_s) // 96)
                px_s = px_s[::step]
                py_s = py_s[::step]
            coords = np.column_stack(((px_s + x0 + 0.5) / float(w), (py_s + y0 + 0.5) / float(h)))
            try:
                ch = ConvexHull(coords)
                raw_hull = [(float(coords[v, 0]), float(coords[v, 1])) for v in ch.vertices]
                hull_pts = _chaikin_smooth_closed(raw_hull, iterations=2)
            except Exception:
                hull_pts = []

        return _Detection(
            cx=cx,
            cy=cy,
            w=bw,
            h=bh,
            area=area_px / total_px,
            intensity=intensity,
            hull=hull_pts,
        )

    def _extract_detections(self, rgb_small: np.ndarray) -> List[_Detection]:
        """
        rgb_small: (H, W, 3) uint8 in OpenGL bottom-up orientation (y=0 is bottom, y=H-1 is top).
        """
        h, w = rgb_small.shape[:2]
        if h < 8 or w < 8:
            return []

        luma = (
            rgb_small[:, :, 0].astype(np.float32) * 0.2126
            + rgb_small[:, :, 1].astype(np.float32) * 0.7152
            + rgb_small[:, :, 2].astype(np.float32) * 0.0722
        ) / 255.0

        # Resolution-aware Gaussian smoothing to fuse grainy stipple / sensor noise into coherent regions
        sig = max(0.85, min(1.4, w / 100.0))
        smooth = ndimage.gaussian_filter(luma, sigma=sig)

        if self._prev_luma is None or self._prev_luma.shape != smooth.shape:
            self._prev_luma = smooth.copy()
            self._bg_luma = smooth.copy()

        # Frame-to-frame motion + slow background delta
        motion_fast = np.abs(smooth - self._prev_luma)
        motion_bg = np.abs(smooth - self._bg_luma)
        motion = np.clip(motion_fast * 2.6 + motion_bg * 0.9, 0.0, 1.0)

        self._prev_luma = smooth.copy()
        self._bg_luma = self._bg_luma * 0.92 + smooth * 0.08

        mode = self.settings.detect_mode
        if mode == 1:
            feature_map = smooth
        elif mode == 2:
            feature_map = ndimage.gaussian_filter(motion, sigma=sig * 0.9)
        elif mode == 3:
            gx = ndimage.sobel(smooth, axis=1)
            gy = ndimage.sobel(smooth, axis=0)
            feature_map = ndimage.gaussian_filter(np.hypot(gx, gy), sigma=sig * 0.9)
        else:
            motion_energy = float(np.percentile(motion, 96))
            m_weight = 0.65 if motion_energy > 0.08 else 0.15
            feature_map = smooth * (1.0 - m_weight * 0.35) + ndimage.gaussian_filter(motion, sigma=sig * 0.85) * (m_weight * 1.4)

        peak = float(np.max(feature_map))
        if peak < 0.06:
            return []

        sens = float(np.clip(self.settings.sensitivity, 0.10, 0.96))
        pct = 95.0 - sens * 24.0
        p_thresh = float(np.percentile(feature_map, pct))
        abs_thresh = max(0.09, peak * (0.64 - sens * 0.42))
        thresh = max(p_thresh, abs_thresh)

        mask = feature_map >= thresh
        # Gaussian smoothing already suppressed 1px sensor speckle; only close tiny hairline gaps
        mask = ndimage.binary_closing(mask, iterations=1)

        labeled, num_features = ndimage.label(mask)
        if num_features == 0:
            return []

        slices = ndimage.find_objects(labeled)
        total_px = float(w * h)
        min_px = max(3.5, self.settings.min_area * total_px)
        max_px = max(min_px * 6.0, self.settings.max_area * total_px)

        candidates: List[_Detection] = []
        for label_id, sl in enumerate(slices, start=1):
            if sl is None:
                continue
            ys, xs = sl
            sub_mask = labeled[ys, xs] == label_id
            area_px = float(np.count_nonzero(sub_mask))
            if area_px < min_px:
                continue

            sub_feat = feature_map[ys, xs]
            box_w_norm = float(xs.stop - xs.start) / float(w)
            box_h_norm = float(ys.stop - ys.start) / float(h)
            oversized = (area_px > max_px) or (box_w_norm > 0.28) or (box_h_norm > 0.28)

            if not oversized:
                det = self._build_detection(sub_mask, sub_feat, ys.start, xs.start, h, w, total_px)
                if det is not None:
                    candidates.append(det)
            else:
                # Smart Peak-Splitting for oversized bright clouds/washes:
                # Finds distinct local maxima inside the large region and partitions them into tight, clean blobs!
                win_sz = max(5, min(11, int(min(h, w) * 0.14)) | 1)
                local_max = (sub_feat == ndimage.maximum_filter(sub_feat, size=win_sz)) & sub_mask
                py_idx, px_idx = np.nonzero(local_max)
                if len(py_idx) == 0:
                    continue
                vals = sub_feat[py_idx, px_idx]
                order = np.argsort(vals)[::-1]
                # Enforce clean minimum peak separation so split blobs never stack awkwardly
                min_sep = max(5.0, min(w, h) * 0.14)
                chosen: List[Tuple[int, int]] = []
                for idx in order:
                    py_i, px_i = int(py_idx[idx]), int(px_idx[idx])
                    if all(math.hypot(py_i - cy_i, px_i - cx_i) >= min_sep for cy_i, cx_i in chosen):
                        chosen.append((py_i, px_i))
                    if len(chosen) >= 4:
                        break
                if not chosen:
                    continue

                sh, sw = sub_mask.shape
                yy, xx = np.mgrid[0:sh, 0:sw]
                # Voronoi assignment to nearest strong peak, clamped to a refined luminous radius
                dists = np.stack([(yy - cy_i) ** 2 + (xx - cx_i) ** 2 for cy_i, cx_i in chosen], axis=0)
                nearest = np.argmin(dists, axis=0)
                max_rad_sq = (max(6.5, min(w, h) * 0.16)) ** 2
                for k, (cy_i, cx_i) in enumerate(chosen):
                    pk_val = float(sub_feat[cy_i, cx_i])
                    cell_mask = (
                        sub_mask
                        & (nearest == k)
                        & (dists[k] <= max_rad_sq)
                        & (sub_feat >= pk_val * 0.65)
                    )
                    if float(np.count_nonzero(cell_mask)) >= min_px:
                        det = self._build_detection(cell_mask, sub_feat, ys.start, xs.start, h, w, total_px)
                        if det is not None:
                            candidates.append(det)

        # Rank blobs intelligently by visual salience (intensity * sqrt(area)) and keep top N
        candidates.sort(key=lambda d: d.intensity * math.sqrt(d.area), reverse=True)
        return candidates[: max(1, int(self.settings.max_blobs))]

    def update(self, rgb_small: np.ndarray | None, dt: float) -> List[TrackedBlob]:
        """
        Runs CV blob detection (~30 Hz for sub-1ms budget) and smoothly updates Kalman
        positions, fade-in/out alpha, trails, and plexus constellation links at full 60-144 FPS!
        """
        dt = min(max(dt, 1e-4), 0.1)
        if not self.settings.enabled:
            self.tracks.clear()
            self.plexus_edges.clear()
            return self.tracks

        self._analyze_accum += dt
        run_cv = (rgb_small is not None) and (self._analyze_accum >= 0.030 or not self.tracks)

        if run_cv and rgb_small is not None:
            self._analyze_accum = 0.0
            detections = self._extract_detections(rgb_small)
            self._associate_and_update(detections, 0.033)
        else:
            # Between CV ticks, smoothly glide tracks using Kalman velocity for buttery 120+ FPS motion!
            for tr in self.tracks:
                tr.cx = float(np.clip(tr.cx + tr.vx * dt * 0.55, 0.01, 0.99))
                tr.cy = float(np.clip(tr.cy + tr.vy * dt * 0.55, 0.01, 0.99))
                tr.age += dt
                if tr.missed == 0:
                    tr.alpha = min(1.0, tr.alpha + dt * 5.2)
                else:
                    tr.alpha = max(0.0, tr.alpha - dt * 3.8)

        # Build proximity Constellation / Plexus network between visible blobs
        self._compute_plexus()
        return self.tracks

    def _associate_and_update(self, detections: List[_Detection], step_dt: float) -> None:
        max_missed = 10
        dist_gate = 0.24
        smooth = float(np.clip(self.settings.smoothing, 0.15, 0.94))
        pos_alpha = 1.0 - smooth * 0.86
        size_alpha = 1.0 - smooth * 0.90

        # 1. Kalman predict step for all existing tracks
        preds: List[Tuple[float, float]] = []
        for tr in self.tracks:
            px = float(np.clip(tr.cx + tr.vx * step_dt, 0.0, 1.0))
            py = float(np.clip(tr.cy + tr.vy * step_dt, 0.0, 1.0))
            preds.append((px, py))

        assigned_tracks = set()
        assigned_dets = set()

        if self.tracks and detections:
            cost = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)
            for i, (tr, (px, py)) in enumerate(zip(self.tracks, preds)):
                for j, det in enumerate(detections):
                    d_pos = math.hypot(px - det.cx, py - det.cy)
                    d_sz = abs(tr.w - det.w) + abs(tr.h - det.h)
                    cost[i, j] = d_pos + d_sz * 0.25

            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] > dist_gate:
                    continue
                assigned_tracks.add(r)
                assigned_dets.add(c)
                tr = self.tracks[r]
                det = detections[c]
                px, py = preds[r]

                # Kalman innovation update (smooth position + velocity + adaptive size)
                new_cx = px + (det.cx - px) * pos_alpha
                new_cy = py + (det.cy - py) * pos_alpha
                inst_vx = (new_cx - tr.cx) / max(step_dt, 1e-3)
                inst_vy = (new_cy - tr.cy) / max(step_dt, 1e-3)
                tr.vx = tr.vx * 0.65 + float(np.clip(inst_vx, -2.5, 2.5)) * 0.35
                tr.vy = tr.vy * 0.65 + float(np.clip(inst_vy, -2.5, 2.5)) * 0.35
                tr.cx = new_cx
                tr.cy = new_cy
                tr.w = tr.w + (det.w - tr.w) * size_alpha
                tr.h = tr.h + (det.h - tr.h) * size_alpha
                tr.area = tr.area * 0.75 + det.area * 0.25
                tr.intensity = tr.intensity * 0.70 + det.intensity * 0.30
                tr.hull = det.hull if det.hull else tr.hull
                tr.missed = 0
                tr.age += step_dt
                tr.alpha = min(1.0, tr.alpha + step_dt * 5.2)

                tr.trail.append((tr.cx, tr.cy))
                if len(tr.trail) > 22:
                    tr.trail.pop(0)

        # 2. Occlusion dead-reckoning & smooth fade-out for temporarily missed tracks
        alive: List[TrackedBlob] = []
        for i, tr in enumerate(self.tracks):
            if i not in assigned_tracks:
                tr.missed += 1
                tr.vx *= 0.82
                tr.vy *= 0.82
                tr.cx = float(np.clip(tr.cx + tr.vx * step_dt, 0.01, 0.99))
                tr.cy = float(np.clip(tr.cy + tr.vy * step_dt, 0.01, 0.99))
                tr.alpha = max(0.0, tr.alpha - step_dt * 3.8)
            if tr.missed <= max_missed and (tr.alpha > 0.02 or tr.missed == 0):
                alive.append(tr)
        self.tracks = alive

        # 3. Spawn new tracks for unmatched detections (capped cleanly by max_blobs)
        max_b = max(1, int(self.settings.max_blobs))
        for j, det in enumerate(detections):
            if j in assigned_dets or len(self.tracks) >= max_b:
                continue
            tb = TrackedBlob(
                track_id=self._next_id,
                cx=det.cx,
                cy=det.cy,
                w=det.w,
                h=det.h,
                area=det.area,
                intensity=det.intensity,
                alpha=0.18,
                trail=[(det.cx, det.cy)],
                hull=det.hull,
            )
            self._next_id = (self._next_id % 99) + 1
            self.tracks.append(tb)

    def _compute_plexus(self) -> None:
        self.plexus_edges = []
        if not self.settings.plexus_enabled or len(self.tracks) < 2:
            return
        max_d = max(0.08, float(self.settings.plexus_distance))
        # Connect each blob to up to 3 closest neighbors within max_d (clean, non-cluttered constellation!)
        seen_pairs = set()
        n = len(self.tracks)
        for i in range(n):
            a = self.tracks[i]
            if a.alpha < 0.08:
                continue
            neighbors: List[Tuple[float, int]] = []
            for j in range(n):
                if i == j:
                    continue
                b = self.tracks[j]
                if b.alpha < 0.08:
                    continue
                d = math.hypot(a.cx - b.cx, a.cy - b.cy)
                if d <= max_d:
                    neighbors.append((d, j))
            neighbors.sort(key=lambda item: item[0])
            for d, j in neighbors[:3]:
                pair = (min(i, j), max(i, j))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                b = self.tracks[j]
                weight = (1.0 - d / max_d) * min(a.alpha, b.alpha)
                self.plexus_edges.append((a, b, float(weight)))
