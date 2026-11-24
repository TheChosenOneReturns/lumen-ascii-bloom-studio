from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from imgui_bundle import imgui

from .blob_tracker import BlobSettings, TrackedBlob


@dataclass(frozen=True)
class SpecimenTheme:
    id: str
    label: str
    specimen_code: str
    bg: Tuple[float, float, float, float]
    panel_bg: Tuple[float, float, float, float]
    frame_bg: Tuple[float, float, float, float]
    frame_hover: Tuple[float, float, float, float]
    frame_active: Tuple[float, float, float, float]
    line_core: Tuple[float, float, float, float]
    line_glow: Tuple[float, float, float, float]
    node_core: Tuple[float, float, float, float]
    accent_alt: Tuple[float, float, float, float]
    text_main: Tuple[float, float, float, float]
    text_dim: Tuple[float, float, float, float]
    grid_sub: Tuple[float, float, float, float]


SPECIMEN_THEMES: List[SpecimenTheme] = [
    SpecimenTheme(
        id="violet",
        label="VIOLET",
        specimen_code="SPECIMEN N001",
        bg=(0.043, 0.028, 0.078, 1.0),
        panel_bg=(0.062, 0.040, 0.112, 1.0),
        frame_bg=(0.095, 0.064, 0.170, 1.0),
        frame_hover=(0.155, 0.105, 0.275, 1.0),
        frame_active=(0.240, 0.165, 0.420, 1.0),
        line_core=(0.74, 0.66, 1.00, 0.95),
        line_glow=(0.50, 0.32, 1.00, 0.38),
        node_core=(0.98, 0.96, 1.00, 1.00),
        accent_alt=(0.92, 0.46, 1.00, 1.00),
        text_main=(0.94, 0.91, 1.00, 1.00),
        text_dim=(0.58, 0.52, 0.74, 1.00),
        grid_sub=(0.48, 0.36, 0.80, 0.22),
    ),
    SpecimenTheme(
        id="cyan",
        label="CYAN",
        specimen_code="SUBJECT A-34",
        bg=(0.020, 0.032, 0.052, 1.0),
        panel_bg=(0.030, 0.048, 0.078, 1.0),
        frame_bg=(0.050, 0.088, 0.140, 1.0),
        frame_hover=(0.082, 0.150, 0.235, 1.0),
        frame_active=(0.125, 0.245, 0.380, 1.0),
        line_core=(0.48, 0.82, 1.00, 0.95),
        line_glow=(0.18, 0.58, 1.00, 0.38),
        node_core=(0.94, 0.99, 1.00, 1.00),
        accent_alt=(0.68, 0.92, 1.00, 1.00),
        text_main=(0.86, 0.95, 1.00, 1.00),
        text_dim=(0.44, 0.64, 0.78, 1.00),
        grid_sub=(0.28, 0.64, 0.92, 0.22),
    ),
    SpecimenTheme(
        id="noir",
        label="NOIR",
        specimen_code="MONOLITH // 00",
        bg=(0.018, 0.018, 0.022, 1.0),
        panel_bg=(0.032, 0.032, 0.038, 1.0),
        frame_bg=(0.075, 0.075, 0.086, 1.0),
        frame_hover=(0.135, 0.138, 0.155, 1.0),
        frame_active=(0.220, 0.225, 0.250, 1.0),
        line_core=(0.90, 0.92, 0.96, 0.92),
        line_glow=(0.65, 0.72, 0.85, 0.28),
        node_core=(1.00, 1.00, 1.00, 1.00),
        accent_alt=(0.82, 0.86, 0.94, 1.00),
        text_main=(0.96, 0.96, 0.98, 1.00),
        text_dim=(0.54, 0.56, 0.60, 1.00),
        grid_sub=(0.55, 0.58, 0.64, 0.20),
    ),
]


def _col(rgba: Tuple[float, float, float, float], alpha_scale: float = 1.0) -> int:
    return imgui.get_color_u32(imgui.ImVec4(rgba[0], rgba[1], rgba[2], min(1.0, max(0.0, rgba[3] * alpha_scale))))


def apply_specimen_imgui_theme(theme: SpecimenTheme) -> None:
    """Configures sharp architectural sci-fi terminal geometry matching SPECIMEN N001 / SUBJECT A-34."""
    style = imgui.get_style()
    style.window_rounding = 0.0
    style.child_rounding = 0.0
    style.frame_rounding = 0.0
    style.popup_rounding = 0.0
    style.grab_rounding = 0.0
    style.tab_rounding = 0.0
    style.scrollbar_rounding = 0.0
    style.window_border_size = 0.0
    style.child_border_size = 0.0
    style.frame_border_size = 1.0
    style.item_spacing = imgui.ImVec2(7.0, 5.0)
    style.frame_padding = imgui.ImVec2(7.0, 4.0)
    style.window_padding = imgui.ImVec2(8.0, 8.0)

    col = imgui.Col_
    lc = theme.line_core
    lg = theme.line_glow
    bg = theme.bg
    pb = theme.panel_bg
    fb = theme.frame_bg
    fh = theme.frame_hover
    fa = theme.frame_active

    entries = [
        (col.text, imgui.ImVec4(*theme.text_main)),
        (col.text_disabled, imgui.ImVec4(*theme.text_dim)),
        (col.window_bg, imgui.ImVec4(*bg)),
        (col.child_bg, imgui.ImVec4(*pb)),
        (col.popup_bg, imgui.ImVec4(pb[0], pb[1], pb[2], 0.97)),
        (col.border, imgui.ImVec4(lc[0], lc[1], lc[2], 0.42)),
        (col.border_shadow, imgui.ImVec4(lg[0], lg[1], lg[2], 0.20)),
        (col.frame_bg, imgui.ImVec4(*fb)),
        (col.frame_bg_hovered, imgui.ImVec4(*fh)),
        (col.frame_bg_active, imgui.ImVec4(*fa)),
        (col.title_bg, imgui.ImVec4(*bg)),
        (col.title_bg_active, imgui.ImVec4(*pb)),
        (col.header, imgui.ImVec4(fh[0], fh[1], fh[2], 0.85)),
        (col.header_hovered, imgui.ImVec4(fa[0], fa[1], fa[2], 0.95)),
        (col.header_active, imgui.ImVec4(lc[0], lc[1], lc[2], 0.65)),
        (col.button, imgui.ImVec4(fb[0], fb[1], fb[2], 0.92)),
        (col.button_hovered, imgui.ImVec4(*fh)),
        (col.button_active, imgui.ImVec4(*fa)),
        (col.slider_grab, imgui.ImVec4(lc[0], lc[1], lc[2], 0.95)),
        (col.slider_grab_active, imgui.ImVec4(*theme.node_core)),
        (col.check_mark, imgui.ImVec4(*theme.node_core)),
        (col.tab, imgui.ImVec4(fb[0], fb[1], fb[2], 0.90)),
        (col.tab_hovered, imgui.ImVec4(*fh)),
        (col.tab_selected, imgui.ImVec4(*fa)),
        (col.tab_selected_overline, imgui.ImVec4(*theme.node_core)),
        (col.tab_dimmed, imgui.ImVec4(pb[0], pb[1], pb[2], 0.95)),
        (col.tab_dimmed_selected, imgui.ImVec4(*fh)),
        (col.tab_dimmed_selected_overline, imgui.ImVec4(*lc)),
        (col.plot_histogram, imgui.ImVec4(*lc)),
        (col.plot_histogram_hovered, imgui.ImVec4(*theme.node_core)),
        (col.separator, imgui.ImVec4(lc[0], lc[1], lc[2], 0.48)),
        (col.scrollbar_bg, imgui.ImVec4(bg[0], bg[1], bg[2], 0.9)),
        (col.scrollbar_grab, imgui.ImVec4(lc[0], lc[1], lc[2], 0.35)),
        (col.scrollbar_grab_hovered, imgui.ImVec4(lc[0], lc[1], lc[2], 0.65)),
        (col.scrollbar_grab_active, imgui.ImVec4(*lc)),
    ]
    for cid, rgba in entries:
        style.set_color_(cid.value, rgba)


def marquee_text(text: str, max_chars: int, time_sec: float, speed: float = 4.2) -> str:
    """Smooth scrolling cyber marquee for long Spotify track/artist names."""
    if len(text) <= max_chars:
        return text
    padded = text + "   ///   "
    n = len(padded)
    shift = int(time_sec * speed) % n
    doubled = padded + padded
    return doubled[shift : shift + max_chars]


def draw_glow_line(
    dl: imgui.ImDrawList,
    p1: imgui.ImVec2,
    p2: imgui.ImVec2,
    theme: SpecimenTheme,
    intensity: float = 1.0,
    kick_pulse: float = 0.0,
) -> None:
    boost = intensity * (1.0 + kick_pulse * 0.35)
    dl.add_line(p1, p2, _col(theme.line_glow, 0.38 * boost), 5.0 + kick_pulse * 2.0)
    dl.add_line(p1, p2, _col(theme.line_glow, 0.78 * boost), 2.4)
    dl.add_line(p1, p2, _col(theme.line_core, 0.96 * boost), 1.0)


def draw_glow_node(
    dl: imgui.ImDrawList,
    pos: imgui.ImVec2,
    theme: SpecimenTheme,
    radius: float = 2.6,
    kick_pulse: float = 0.0,
) -> None:
    r = radius * (1.0 + kick_pulse * 0.48)
    if kick_pulse > 0.20:
        ring_r = radius * (2.6 + kick_pulse * 3.4)
        dl.add_circle(pos, ring_r, _col(theme.line_core, min(0.85, kick_pulse * 0.65)), 16, 1.0)
    dl.add_circle_filled(pos, r * 2.5, _col(theme.line_glow, 0.34 + kick_pulse * 0.22), 12)
    dl.add_circle_filled(pos, r * 1.45, _col(theme.line_core, 0.76), 12)
    dl.add_circle_filled(pos, r * 0.80, _col(theme.node_core, 1.00), 10)


def _perimeter_point(p_min: imgui.ImVec2, p_max: imgui.ImVec2, u: float) -> imgui.ImVec2:
    """Maps u in [0, 1) around the 4 edges of a rectangle cleanly."""
    w = max(1.0, p_max.x - p_min.x)
    h = max(1.0, p_max.y - p_min.y)
    perim = 2.0 * (w + h)
    d = (u % 1.0) * perim
    if d < w:
        return imgui.ImVec2(p_min.x + d, p_min.y)
    d -= w
    if d < h:
        return imgui.ImVec2(p_max.x, p_min.y + d)
    d -= h
    if d < w:
        return imgui.ImVec2(p_max.x - d, p_max.y)
    d -= w
    return imgui.ImVec2(p_min.x, p_max.y - d)


def draw_specimen_box(
    dl: imgui.ImDrawList,
    p_min: imgui.ImVec2,
    p_max: imgui.ImVec2,
    theme: SpecimenTheme,
    with_nodes: bool = True,
    intensity: float = 1.0,
    anim_time: float = -1.0,
    kick_pulse: float = 0.0,
    drop_glow: float = 0.0,
    num_pulses: int = 2,
) -> None:
    """
    Draws the iconic multi-layer glowing 1px hairline border + breathing vertex nodes
    + animated traveling laser packets racing along the perimeter synced to music!
    """
    boost = intensity * (1.0 + kick_pulse * 0.35 + drop_glow * 0.55)
    dl.add_rect(p_min, p_max, _col(theme.line_glow, 0.28 * boost), rounding=0.0, thickness=5.8 + drop_glow * 3.0)
    dl.add_rect(p_min, p_max, _col(theme.line_glow, 0.62 * boost), rounding=0.0, thickness=2.4)
    dl.add_rect(p_min, p_max, _col(theme.line_core, 0.94 * boost), rounding=0.0, thickness=1.0)

    # Animated traveling neon comet light-packets along the 1px hairline frame!
    if anim_time >= 0.0 and num_pulses > 0:
        speed = 0.10 + kick_pulse * 0.06 + drop_glow * 0.10
        tail_steps = 7
        for p_idx in range(num_pulses):
            head_u = (anim_time * speed + p_idx / float(num_pulses)) % 1.0
            for s in range(tail_steps):
                u0 = (head_u - s * 0.0085) % 1.0
                u1 = (head_u - (s + 1) * 0.0085) % 1.0
                pt0 = _perimeter_point(p_min, p_max, u0)
                pt1 = _perimeter_point(p_min, p_max, u1)
                # Avoid cutting diagonally across corners if segment jumps
                if abs(pt0.x - pt1.x) < 1.0 or abs(pt0.y - pt1.y) < 1.0:
                    fade = (1.0 - s / float(tail_steps)) ** 1.6
                    dl.add_line(pt0, pt1, _col(theme.line_glow, 0.72 * fade), 4.2)
                    dl.add_line(pt0, pt1, _col(theme.node_core, 0.95 * fade), 1.6)
            head_pt = _perimeter_point(p_min, p_max, head_u)
            dl.add_circle_filled(head_pt, 2.2 + kick_pulse * 1.4, _col(theme.node_core, 0.95), 8)

    if with_nodes:
        draw_glow_node(dl, imgui.ImVec2(p_min.x, p_min.y), theme, 2.4, kick_pulse * 0.75)
        draw_glow_node(dl, imgui.ImVec2(p_max.x, p_min.y), theme, 2.4, kick_pulse * 0.75)
        draw_glow_node(dl, imgui.ImVec2(p_min.x, p_max.y), theme, 2.4, kick_pulse * 0.75)
        draw_glow_node(dl, imgui.ImVec2(p_max.x, p_max.y), theme, 2.4, kick_pulse * 0.75)


def draw_hud_scanbeam(
    dl: imgui.ImDrawList,
    p_min: imgui.ImVec2,
    p_max: imgui.ImVec2,
    theme: SpecimenTheme,
    time_sec: float,
    speed: float = 0.22,
    alpha: float = 0.24,
) -> None:
    """Subtle gliding holographic horizontal scanline sweeping down a Specimen panel."""
    h = max(4.0, p_max.y - p_min.y)
    y = p_min.y + ((time_sec * speed) % 1.0) * h
    dl.add_rect_filled(
        imgui.ImVec2(p_min.x + 1, max(p_min.y + 1, y - 10.0)),
        imgui.ImVec2(p_max.x - 1, min(p_max.y - 1, y)),
        _col(theme.line_glow, alpha * 0.35),
    )
    dl.add_line(
        imgui.ImVec2(p_min.x + 1, y),
        imgui.ImVec2(p_max.x - 1, y),
        _col(theme.line_core, alpha),
        1.0,
    )


def draw_corner_brackets(
    dl: imgui.ImDrawList,
    p_min: imgui.ImVec2,
    p_max: imgui.ImVec2,
    theme: SpecimenTheme,
    arm: float = 11.0,
    kick_pulse: float = 0.0,
) -> None:
    """Glowing sci-fi viewfinder L-brackets that subtly breathe/lock in sync with kicks."""
    inset = kick_pulse * 2.4
    x0, y0 = p_min.x + inset, p_min.y + inset
    x1, y1 = p_max.x - inset, p_max.y - inset
    segs = [
        (imgui.ImVec2(x0, y0), imgui.ImVec2(x0 + arm, y0)),
        (imgui.ImVec2(x0, y0), imgui.ImVec2(x0, y0 + arm)),
        (imgui.ImVec2(x1, y0), imgui.ImVec2(x1 - arm, y0)),
        (imgui.ImVec2(x1, y0), imgui.ImVec2(x1, y0 + arm)),
        (imgui.ImVec2(x0, y1), imgui.ImVec2(x0 + arm, y1)),
        (imgui.ImVec2(x0, y1), imgui.ImVec2(x0, y1 - arm)),
        (imgui.ImVec2(x1, y1), imgui.ImVec2(x1 - arm, y1)),
        (imgui.ImVec2(x1, y1), imgui.ImVec2(x1, y1 - arm)),
    ]
    for a, b in segs:
        draw_glow_line(dl, a, b, theme, 0.95 + kick_pulse * 0.35)


def draw_glow_stars(
    dl: imgui.ImDrawList,
    origin: imgui.ImVec2,
    total: int,
    filled_count: float,
    theme: SpecimenTheme,
    radius: float = 6.5,
    spacing: float = 17.5,
    time_sec: float = 0.0,
) -> None:
    """Luminous animated 5-point vector stars (★★★★★) matching SPECIMEN N001 & SUBJECT A-34."""
    for i in range(total):
        cx = origin.x + i * spacing + radius
        cy = origin.y + radius
        is_lit = (i + 0.5) <= filled_count
        shimmer = 1.0 + 0.10 * math.sin(time_sec * 5.0 + i * 1.1) if is_lit else 1.0
        rad = radius * shimmer

        pts: List[imgui.ImVec2] = []
        for k in range(10):
            ang = -math.pi * 0.5 + k * (math.pi / 5.0)
            r = rad if (k % 2 == 0) else rad * 0.44
            pts.append(imgui.ImVec2(cx + math.cos(ang) * r, cy + math.sin(ang) * r))

        if is_lit:
            dl.add_circle_filled(imgui.ImVec2(cx, cy), rad * 1.75, _col(theme.line_glow, 0.48), 14)
            center = imgui.ImVec2(cx, cy)
            fill_c = _col(theme.node_core, 0.98)
            for k in range(10):
                dl.add_triangle_filled(center, pts[k], pts[(k + 1) % 10], fill_c)
            dl.add_polyline(pts, _col(theme.line_core, 0.95), thickness=1.1, flags=imgui.ImDrawFlags_.closed)
        else:
            dl.add_polyline(pts, _col(theme.line_glow, 0.45), thickness=2.2, flags=imgui.ImDrawFlags_.closed)
            dl.add_polyline(pts, _col(theme.line_core, 0.80), thickness=1.0, flags=imgui.ImDrawFlags_.closed)


def draw_specimen_divider(dl: imgui.ImDrawList, x0: float, x1: float, y: float, theme: SpecimenTheme, nodes: bool = True) -> None:
    draw_glow_line(dl, imgui.ImVec2(x0, y), imgui.ImVec2(x1, y), theme, 0.85)
    if nodes:
        draw_glow_node(dl, imgui.ImVec2(x0, y), theme, 2.0)
        draw_glow_node(dl, imgui.ImVec2(x1, y), theme, 2.0)


def _project_3d(x: float, y: float, z: float, yaw: float, pitch: float, cx: float, cy: float, scale: float) -> Tuple[float, float, float]:
    cyaw, syaw = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    x1 = x * cyaw + z * syaw
    z1 = -x * syaw + z * cyaw
    y2 = y * cp - z1 * sp
    z2 = y * sp + z1 * cp
    persp = 3.2 / (3.2 - z2)
    return cx + x1 * scale * persp, cy - y2 * scale * persp, z2


def draw_wireframe_3d_scope(
    dl: imgui.ImDrawList,
    p_min: imgui.ImVec2,
    p_max: imgui.ImVec2,
    theme: SpecimenTheme,
    time_sec: float,
    audio_kick: float,
    audio_mid: float,
    shape_idx: int,
) -> None:
    """
    Animated 3D neon wireframe sculpture inside the HUD:
      0 = Heart (SPECIMEN N001), 1 = Geodesic Orb, 2 = Torus, 3 = Audio DNA Double Helix!
    """
    w = p_max.x - p_min.x
    h = p_max.y - p_min.y
    cx = p_min.x + w * 0.5
    cy = p_min.y + h * 0.48
    scale = min(w, h) * (0.31 + audio_kick * 0.055)

    draw_hud_scanbeam(dl, p_min, p_max, theme, time_sec, speed=0.28, alpha=0.28)
    dl.add_circle_filled(imgui.ImVec2(cx, cy), scale * 1.18, _col(theme.line_glow, 0.24 + audio_kick * 0.18), 24)

    yaw = time_sec * (0.78 + audio_mid * 0.90)
    pitch = 0.26 + math.sin(time_sec * 0.65) * 0.18
    mode = shape_idx % 4

    glow_c = _col(theme.line_glow, 0.42 + audio_kick * 0.20)
    core_c = _col(theme.line_core, 0.88)
    bright_c = _col(theme.node_core, 0.94)

    if mode == 0:
        n_u, n_v = 18, 11
        grid: List[List[Tuple[float, float, float]]] = []
        for iv in range(n_v):
            v = -1.0 + 2.0 * (iv / float(n_v - 1))
            thickness = math.sqrt(max(0.0, 1.0 - v * v))
            row: List[Tuple[float, float, float]] = []
            for iu in range(n_u):
                u = (iu / float(n_u)) * 2.0 * math.pi
                hx = 16.0 * (math.sin(u) ** 3) / 16.5
                hy = (13.0 * math.cos(u) - 5.0 * math.cos(2.0 * u) - 2.0 * math.cos(3.0 * u) - math.cos(4.0 * u)) / 16.5
                px = hx * (0.28 + 0.72 * thickness)
                py = (hy + 0.12) * (0.28 + 0.72 * thickness)
                pz = v * 0.42 * (0.55 + 0.45 * math.cos(u * 0.5))
                row.append(_project_3d(px, py, pz, yaw, pitch, cx, cy, scale))
            grid.append(row)
        for iv in range(n_v):
            for iu in range(n_u):
                ax, ay, az = grid[iv][iu]
                bx, by, bz = grid[iv][(iu + 1) % n_u]
                c = bright_c if (az + bz) > 0.1 else core_c
                dl.add_line(imgui.ImVec2(ax, ay), imgui.ImVec2(bx, by), glow_c, 2.4)
                dl.add_line(imgui.ImVec2(ax, ay), imgui.ImVec2(bx, by), c, 1.0)
                if iv < n_v - 1 and iu % 2 == 0:
                    cx2, cy2, _ = grid[iv + 1][iu]
                    dl.add_line(imgui.ImVec2(ax, ay), imgui.ImVec2(cx2, cy2), core_c, 0.9)
    elif mode == 1:
        n_lat, n_lon = 8, 14
        grid = []
        for il in range(n_lat):
            phi = -math.pi * 0.46 + (il / float(n_lat - 1)) * math.pi * 0.92
            row = []
            for io in range(n_lon):
                theta = (io / float(n_lon)) * 2.0 * math.pi
                px = math.cos(phi) * math.cos(theta)
                py = math.sin(phi)
                pz = math.cos(phi) * math.sin(theta)
                row.append(_project_3d(px, py, pz, yaw, pitch, cx, cy, scale))
            grid.append(row)
        for il in range(n_lat):
            for io in range(n_lon):
                ax, ay, _ = grid[il][io]
                bx, by, _ = grid[il][(io + 1) % n_lon]
                dl.add_line(imgui.ImVec2(ax, ay), imgui.ImVec2(bx, by), glow_c, 2.2)
                dl.add_line(imgui.ImVec2(ax, ay), imgui.ImVec2(bx, by), core_c, 1.0)
                if il < n_lat - 1:
                    cx2, cy2, _ = grid[il + 1][io]
                    dl.add_line(imgui.ImVec2(ax, ay), imgui.ImVec2(cx2, cy2), core_c, 0.9)
    elif mode == 2:
        n_u, n_v = 16, 10
        R, r = 0.78, 0.32
        grid = []
        for iu in range(n_u):
            u = (iu / float(n_u)) * 2.0 * math.pi
            row = []
            for iv in range(n_v):
                v = (iv / float(n_v)) * 2.0 * math.pi
                px = (R + r * math.cos(v)) * math.cos(u)
                py = r * math.sin(v)
                pz = (R + r * math.cos(v)) * math.sin(u)
                row.append(_project_3d(px, py, pz, yaw, pitch + 0.35, cx, cy, scale))
            grid.append(row)
        for iu in range(n_u):
            for iv in range(n_v):
                ax, ay, _ = grid[iu][iv]
                bx, by, _ = grid[iu][(iv + 1) % n_v]
                cx2, cy2, _ = grid[(iu + 1) % n_u][iv]
                dl.add_line(imgui.ImVec2(ax, ay), imgui.ImVec2(bx, by), glow_c, 2.0)
                dl.add_line(imgui.ImVec2(ax, ay), imgui.ImVec2(bx, by), core_c, 1.0)
                dl.add_line(imgui.ImVec2(ax, ay), imgui.ImVec2(cx2, cy2), core_c, 0.9)
    else:
        # 3D Animated Glowing Audio DNA Double Helix!
        steps = 22
        strand_a = []
        strand_b = []
        rad = 0.52 + audio_kick * 0.14
        for i in range(steps):
            frac = i / float(steps - 1)
            y = (frac - 0.5) * 1.95
            ang = frac * 2.8 * math.pi + time_sec * 2.0
            pa = _project_3d(math.cos(ang) * rad, y, math.sin(ang) * rad, yaw * 0.5, 0.15, cx, cy, scale)
            pb = _project_3d(math.cos(ang + math.pi) * rad, y, math.sin(ang + math.pi) * rad, yaw * 0.5, 0.15, cx, cy, scale)
            strand_a.append(pa)
            strand_b.append(pb)
            if i % 2 == 0:
                dl.add_line(imgui.ImVec2(pa[0], pa[1]), imgui.ImVec2(pb[0], pb[1]), core_c, 1.0)
                dl.add_circle_filled(imgui.ImVec2(pa[0], pa[1]), 2.2, bright_c, 8)
                dl.add_circle_filled(imgui.ImVec2(pb[0], pb[1]), 2.2, bright_c, 8)
        for i in range(steps - 1):
            dl.add_line(imgui.ImVec2(strand_a[i][0], strand_a[i][1]), imgui.ImVec2(strand_a[i + 1][0], strand_a[i + 1][1]), glow_c, 3.0)
            dl.add_line(imgui.ImVec2(strand_a[i][0], strand_a[i][1]), imgui.ImVec2(strand_a[i + 1][0], strand_a[i + 1][1]), bright_c, 1.2)
            dl.add_line(imgui.ImVec2(strand_b[i][0], strand_b[i][1]), imgui.ImVec2(strand_b[i + 1][0], strand_b[i + 1][1]), glow_c, 3.0)
            dl.add_line(imgui.ImVec2(strand_b[i][0], strand_b[i][1]), imgui.ImVec2(strand_b[i + 1][0], strand_b[i + 1][1]), core_c, 1.2)

    # Animated mini-ruler ticks at bottom
    ry = p_max.y - 10.0
    dl.add_line(imgui.ImVec2(p_min.x + 8, ry), imgui.ImVec2(p_max.x - 8, ry), _col(theme.line_core, 0.65), 1.0)
    marker_x = p_min.x + 8.0 + (w - 16.0) * ((math.sin(time_sec * 1.4) * 0.5) + 0.5)
    dl.add_circle_filled(imgui.ImVec2(marker_x, ry), 2.3, bright_c, 8)
    for k in range(5):
        tx = p_min.x + 8.0 + (w - 16.0) * (k / 4.0)
        dl.add_line(imgui.ImVec2(tx, ry), imgui.ImVec2(tx, ry - 5.0), _col(theme.line_core, 0.75), 1.0)

    draw_corner_brackets(
        dl,
        imgui.ImVec2(p_min.x + 4, p_min.y + 4),
        imgui.ImVec2(p_max.x - 4, p_max.y - 4),
        theme,
        arm=9.0,
        kick_pulse=audio_kick * 0.65,
    )


def draw_audio_scope_panel(
    dl: imgui.ImDrawList,
    p_min: imgui.ImVec2,
    p_max: imgui.ImVec2,
    theme: SpecimenTheme,
    waveform: np.ndarray,
    envelope: np.ndarray,
    spectrum: np.ndarray,
    peak_hold: np.ndarray,
    audio_low: float,
    audio_mid: float,
    audio_high: float,
    time_sec: float,
    scope_mode: int,
) -> None:
    """Animated glowing audio visualizer with beat-pulsing grid & Peak-Hold laser caps."""
    dl.add_rect_filled(p_min, p_max, _col(theme.bg, 0.95))
    x0, y0 = p_min.x + 6.0, p_min.y + 6.0
    x1, y1 = p_max.x - 6.0, p_max.y - 6.0
    w = max(10.0, x1 - x0)

    hex_h = 30.0 if scope_mode == 1 else 0.0
    wave_y1 = y1 - hex_h
    wave_h = max(16.0, wave_y1 - y0)
    mid_y = y0 + wave_h * 0.5

    # Beat-illuminated calibration grid
    grid_c = _col(theme.grid_sub, 1.0 + audio_low * 0.65)
    cols = 18
    rows = 4
    for i in range(cols + 1):
        gx = x0 + w * (i / float(cols))
        dl.add_line(imgui.ImVec2(gx, y0), imgui.ImVec2(gx, wave_y1), grid_c, 1.0)
    for j in range(rows + 1):
        gy = y0 + wave_h * (j / float(rows))
        dl.add_line(imgui.ImVec2(x0, gy), imgui.ImVec2(x1, gy), grid_c, 1.0)

    if scope_mode == 0:
        n = len(envelope)
        step = w / float(max(n, 1))
        for i in range(n):
            bx = x0 + (i + 0.5) * step
            idle = 0.08 + 0.06 * math.sin(i * 0.24 - time_sec * 3.0)
            amp = max(float(envelope[i]), idle)
            half = min(wave_h * 0.46, amp * wave_h * 0.44)
            top = imgui.ImVec2(bx, mid_y - half)
            bot = imgui.ImVec2(bx, mid_y + half)
            dl.add_line(top, bot, _col(theme.line_glow, 0.40), max(2.6, step * 0.95))
            dl.add_line(top, bot, _col(theme.line_core, 0.94), max(1.0, step * 0.44))
        dl.add_line(imgui.ImVec2(x0, mid_y), imgui.ImVec2(x1, mid_y), _col(theme.node_core, 0.78), 1.0)

    elif scope_mode == 1:
        pts1: List[imgui.ImVec2] = []
        pts2: List[imgui.ImVec2] = []
        n_pts = 96
        wf_len = len(waveform)
        amp_env = 0.22 + min(0.75, audio_low * 0.65 + audio_mid * 0.35)
        for i in range(n_pts):
            t = i / float(n_pts - 1)
            px = x0 + t * w
            wf_val = float(waveform[min(wf_len - 1, int(t * (wf_len - 1)))])
            s1 = math.sin(t * 22.0 - time_sec * 4.5) * amp_env * 0.65 + wf_val * 0.55
            s2 = math.sin(t * 14.0 + time_sec * 3.2 + 1.2) * (0.18 + audio_mid * 0.50) - wf_val * 0.25
            y_a = mid_y - max(-0.46, min(0.46, s1 * 0.45)) * wave_h
            y_b = mid_y - max(-0.46, min(0.46, s2 * 0.45)) * wave_h
            pts1.append(imgui.ImVec2(px, y_a))
            pts2.append(imgui.ImVec2(px, y_b))

        dl.add_polyline(pts2, _col(theme.line_glow, 0.45), thickness=3.0, flags=0)
        dl.add_polyline(pts2, _col(theme.accent_alt, 0.78), thickness=1.1, flags=0)
        dl.add_polyline(pts1, _col(theme.line_glow, 0.68), thickness=4.5, flags=0)
        dl.add_polyline(pts1, _col(theme.node_core, 0.98), thickness=1.4, flags=0)

        tick = int(time_sec * 11.0)
        hex_chunks_1 = []
        hex_chunks_2 = []
        spec_len = len(spectrum)
        for k in range(10):
            v1 = int(min(255, spectrum[(k * 3 + tick) % spec_len] * 220.0 + ((k * 37 + tick * 13) % 48)))
            v2 = int(min(255, spectrum[(k * 5 + tick * 2) % spec_len] * 220.0 + ((k * 53 + tick * 7) % 64)))
            hex_chunks_1.append(f"{v1:02X}")
            hex_chunks_2.append(f"{v2:02X}")
        row1 = " ".join(hex_chunks_1) + f"  K:{int(audio_low*99):02d} V:{int(audio_mid*99):02d} H:{int(audio_high*99):02d}"
        row2 = " ".join(hex_chunks_2) + "  SIG//MIR-PARSEVAL"
        dl.add_text(imgui.ImVec2(x0 + 2, wave_y1 + 3), _col(theme.line_core, 0.85), row1)
        dl.add_text(imgui.ImVec2(x0 + 2, wave_y1 + 16), _col(theme.text_dim, 0.90), row2)

    else:
        # 64-Band Mel FFT Spectrum with glowing Peak-Hold laser caps!
        n = len(spectrum)
        step = w / float(max(n, 1))
        for i in range(n):
            bx = x0 + i * step
            val = min(1.0, float(spectrum[i]) * 0.85)
            pk = min(1.0, float(peak_hold[i]) * 0.85) if i < len(peak_hold) else val
            bh = max(2.0, val * (wave_h - 4.0))
            pkh = max(bh, pk * (wave_h - 4.0))
            p_a = imgui.ImVec2(bx + 1.0, wave_y1 - bh)
            p_b = imgui.ImVec2(bx + max(2.0, step - 1.2), wave_y1 - 1.0)
            dl.add_rect_filled(p_a, p_b, _col(theme.line_glow, 0.48))
            dl.add_rect(p_a, p_b, _col(theme.line_core, 0.82), rounding=0.0, thickness=1.0)
            # Floating white-hot peak-hold laser cap
            cap_y = wave_y1 - pkh
            dl.add_line(
                imgui.ImVec2(p_a.x, cap_y),
                imgui.ImVec2(p_b.x, cap_y),
                _col(theme.node_core, 0.96),
                2.0,
            )

    draw_specimen_box(dl, p_min, p_max, theme, with_nodes=True, intensity=0.92, anim_time=time_sec * 1.15, kick_pulse=audio_low * 0.5, num_pulses=1)


def _blob_uv_to_screen(
    u: float,
    v: float,
    img_pos: imgui.ImVec2,
    disp_w: float,
    disp_h: float,
    zoom: float,
    pan_x: float,
    pan_y: float,
) -> Tuple[float, float]:
    su = (u - 0.5 - pan_x) * zoom + 0.5
    sv = (v - 0.5 - pan_y) * zoom + 0.5
    return img_pos.x + su * disp_w, img_pos.y + (1.0 - sv) * disp_h


def _quad_bezier_pts(
    p0: Tuple[float, float],
    ctrl: Tuple[float, float],
    p1: Tuple[float, float],
    steps: int = 14,
) -> List[imgui.ImVec2]:
    out: List[imgui.ImVec2] = []
    for k in range(steps + 1):
        t = k / float(steps)
        mt = 1.0 - t
        x = mt * mt * p0[0] + 2.0 * mt * t * ctrl[0] + t * t * p1[0]
        y = mt * mt * p0[1] + 2.0 * mt * t * ctrl[1] + t * t * p1[1]
        out.append(imgui.ImVec2(x, y))
    return out


def _ink_col(alpha: float) -> int:
    """Deep dark ink backing stroke so thin neon vector lines stay razor-sharp even over pure-white HDR bloom!"""
    return imgui.get_color_u32(imgui.ImVec4(0.015, 0.025, 0.048, max(0.0, min(1.0, alpha))))


def _draw_micro_pill(
    dl: imgui.ImDrawList,
    pos: imgui.ImVec2,
    text: str,
    theme: SpecimenTheme,
    alpha: float,
    accent_bar: bool = True,
    bounds: Tuple[float, float, float, float] | None = None,
) -> None:
    """Ultra-clean dark glass Swiss micro-tag readable on both pure black and bright white bloom."""
    ts = imgui.calc_text_size(text)
    pad_x = 5.0
    pad_y = 1.5
    pill_w = ts.x + pad_x * 2.0 + (3.0 if accent_bar else 0.0)
    pill_h = ts.y + pad_y * 2.0
    x0 = pos.x
    y0 = pos.y
    if bounds is not None:
        bx0, by0, bx1, by1 = bounds
        x0 = max(bx0 + 6.0, min(bx1 - pill_w - 6.0, x0))
        y0 = max(by0 + 26.0, min(by1 - pill_h - 6.0, y0))
    x1 = x0 + pill_w
    y1 = y0 + pill_h
    dl.add_rect_filled(imgui.ImVec2(x0, y0), imgui.ImVec2(x1, y1), _ink_col(0.76 * alpha), 2.0)
    dl.add_rect(imgui.ImVec2(x0, y0), imgui.ImVec2(x1, y1), _col(theme.line_core, 0.42 * alpha), rounding=2.0, thickness=1.0)
    if accent_bar:
        dl.add_rect_filled(imgui.ImVec2(x0, y0), imgui.ImVec2(x0 + 2.5, y1), _col(theme.line_core, 0.95 * alpha))
    dl.add_text(imgui.ImVec2(x0 + pad_x + (3.0 if accent_bar else 0.0), y0 + pad_y - 0.5), _col(theme.node_core, 0.96 * alpha), text)


def draw_intelligent_blobs(
    dl: imgui.ImDrawList,
    img_pos: imgui.ImVec2,
    disp_w: float,
    disp_h: float,
    theme: SpecimenTheme,
    tracks: List[TrackedBlob],
    plexus_edges: List[Tuple[TrackedBlob, TrackedBlob, float]],
    settings: BlobSettings,
    time_sec: float,
    audio_kick: float,
    audio_snare: float,
    zoom: float = 1.0,
    pan_x: float = 0.0,
    pan_y: float = 0.0,
) -> None:
    """
    Renders 5 refined TouchDesigner / Swiss-Cyber visual styles over real CV-tracked blobs:
      0 = TouchDesigner Plexus (ultra-clean adaptive corner ticks + Bezier constellation + micro-pills)
      1 = Swiss Minimal Hairline (1px gallery rectangles + distance rulers)
      2 = Organic Contour / Aura (Chaikin-smoothed glowing splines hugging real blob silhouettes)
      3 = Constellation Nodes (zero boxes — luminous orbital star-map nodes + traveling beat sparks)
      4 = Cyber Targeting Pro (sleek double-corner brackets + velocity direction vectors + center diamond)
    All strokes include a subtle dark ink contrast backing so lines remain razor-sharp over intense HDR bloom!
    """
    if not settings.enabled or not tracks:
        return

    dl.push_clip_rect(img_pos, imgui.ImVec2(img_pos.x + disp_w, img_pos.y + disp_h), True)
    vp_bounds = (img_pos.x, img_pos.y, img_pos.x + disp_w, img_pos.y + disp_h)

    kick = audio_kick if settings.audio_reactive else 0.0
    snare = audio_snare if settings.audio_reactive else 0.0
    style = settings.visual_style

    # --- Layer 1: Constellation Plexus Connection Threads (Curved Bezier or Crisp Straight) ---
    if settings.plexus_enabled and plexus_edges:
        for idx, (ba, bb, weight) in enumerate(plexus_edges):
            ax, ay = _blob_uv_to_screen(ba.cx, ba.cy, img_pos, disp_w, disp_h, zoom, pan_x, pan_y)
            bx, by = _blob_uv_to_screen(bb.cx, bb.cy, img_pos, disp_w, disp_h, zoom, pan_x, pan_y)
            w_alpha = min(1.0, weight * (0.85 + kick * 0.55))
            if w_alpha < 0.04:
                continue

            if settings.plexus_curved:
                mx, my = (ax + bx) * 0.5, (ay + by) * 0.5
                dx, dy = bx - ax, by - ay
                dist = math.hypot(dx, dy) + 1e-4
                bend = min(44.0, dist * 0.16) * (1.0 if ((ba.track_id + bb.track_id) % 2 == 0) else -1.0)
                bend *= 1.0 + 0.25 * math.sin(time_sec * 1.8 + idx)
                cx = mx + (-dy / dist) * bend
                cy = my + (dx / dist) * bend
                pts = _quad_bezier_pts((ax, ay), (cx, cy), (bx, by), steps=14)
                # Dark ink contrast backing (invisible on black, razor-sharp on bright bloom!)
                dl.add_polyline(pts, _ink_col(0.62 * w_alpha), thickness=3.6, flags=0)
                dl.add_polyline(pts, _col(theme.line_glow, 0.42 * w_alpha), thickness=4.4 + kick * 1.8, flags=0)
                dl.add_polyline(pts, _col(theme.line_core, 0.88 * w_alpha), thickness=1.35, flags=0)

                # Traveling luminous data-spark racing between connected blobs!
                spark_t = (time_sec * (0.45 + kick * 0.55 + snare * 0.45) + idx * 0.37) % 1.0
                mt = 1.0 - spark_t
                sx = mt * mt * ax + 2.0 * mt * spark_t * cx + spark_t * spark_t * bx
                sy = mt * mt * ay + 2.0 * mt * spark_t * cy + spark_t * spark_t * by
                dl.add_circle_filled(imgui.ImVec2(sx, sy), 4.2, _ink_col(0.55 * w_alpha), 10)
                dl.add_circle_filled(imgui.ImVec2(sx, sy), 3.8, _col(theme.line_glow, 0.55 * w_alpha), 10)
                dl.add_circle_filled(imgui.ImVec2(sx, sy), 1.8, _col(theme.node_core, 0.98 * w_alpha), 8)
            else:
                pa, pb = imgui.ImVec2(ax, ay), imgui.ImVec2(bx, by)
                dl.add_line(pa, pb, _ink_col(0.62 * w_alpha), 3.2)
                dl.add_line(pa, pb, _col(theme.line_glow, 0.38 * w_alpha), 3.8 + kick * 1.4)
                dl.add_line(pa, pb, _col(theme.line_core, 0.84 * w_alpha), 1.3)
                if style == 1 and w_alpha > 0.35:
                    dist_norm = math.hypot(ba.cx - bb.cx, ba.cy - bb.cy)
                    _draw_micro_pill(
                        dl,
                        imgui.ImVec2((ax + bx) * 0.5 + 4, (ay + by) * 0.5 - 9),
                        f"{dist_norm:.2f}",
                        theme,
                        w_alpha * 0.85,
                        accent_bar=False,
                    )

    # --- Layer 2: Silky Fading Motion Trajectory Ribbons ---
    if settings.show_trails:
        for tr in tracks:
            if len(tr.trail) < 2 or tr.alpha < 0.06:
                continue
            n_t = len(tr.trail)
            prev_pt = _blob_uv_to_screen(tr.trail[0][0], tr.trail[0][1], img_pos, disp_w, disp_h, zoom, pan_x, pan_y)
            for k in range(1, n_t):
                cur_pt = _blob_uv_to_screen(tr.trail[k][0], tr.trail[k][1], img_pos, disp_w, disp_h, zoom, pan_x, pan_y)
                frac = k / float(n_t - 1)
                a = (frac ** 1.5) * tr.alpha * 0.72
                p_prev = imgui.ImVec2(prev_pt[0], prev_pt[1])
                p_cur = imgui.ImVec2(cur_pt[0], cur_pt[1])
                dl.add_line(p_prev, p_cur, _ink_col(a * 0.55), 2.2 + frac * 1.6)
                dl.add_line(p_prev, p_cur, _col(theme.line_glow, a * 0.55), 1.4 + frac * 2.6)
                dl.add_line(p_prev, p_cur, _col(theme.node_core, a * 0.92), 1.0 + frac * 0.7)
                prev_pt = cur_pt

    # --- Layer 3: Individual Tracked Blobs in Selected Aesthetic Style ---
    for tr in tracks:
        a = tr.alpha
        if a < 0.04:
            continue

        cx, cy = _blob_uv_to_screen(tr.cx, tr.cy, img_pos, disp_w, disp_h, zoom, pan_x, pan_y)
        pulse = 1.0 + kick * 0.09
        bw = max(18.0, tr.w * disp_w * zoom * pulse)
        bh = max(18.0, tr.h * disp_h * zoom * pulse)
        x0, y0 = cx - bw * 0.5, cy - bh * 0.5
        x1, y1 = cx + bw * 0.5, cy + bh * 0.5
        p0, p1 = imgui.ImVec2(x0, y0), imgui.ImVec2(x1, y1)

        if style == 0:
            # === STYLE 0: TOUCHDESIGNER PLEXUS (Crisp Adaptive Corner Ticks + Micro-Crosshair + Swiss Pill) ===
            arm = max(6.0, min(18.0, min(bw, bh) * 0.26))
            corners = [
                (imgui.ImVec2(x0, y0), imgui.ImVec2(x0 + arm, y0)),
                (imgui.ImVec2(x0, y0), imgui.ImVec2(x0, y0 + arm)),
                (imgui.ImVec2(x1, y0), imgui.ImVec2(x1 - arm, y0)),
                (imgui.ImVec2(x1, y0), imgui.ImVec2(x1, y0 + arm)),
                (imgui.ImVec2(x0, y1), imgui.ImVec2(x0 + arm, y1)),
                (imgui.ImVec2(x0, y1), imgui.ImVec2(x0, y1 - arm)),
                (imgui.ImVec2(x1, y1), imgui.ImVec2(x1 - arm, y1)),
                (imgui.ImVec2(x1, y1), imgui.ImVec2(x1, y1 - arm)),
            ]
            # Subtle dark+luminous perimeter whisper
            dl.add_rect(p0, p1, _ink_col(0.32 * a), rounding=0.0, thickness=2.0)
            dl.add_rect(p0, p1, _col(theme.line_core, 0.28 * a), rounding=0.0, thickness=1.0)
            for ca, cb in corners:
                dl.add_line(ca, cb, _ink_col(0.75 * a), 3.6)
                dl.add_line(ca, cb, _col(theme.line_glow, 0.62 * a), 3.4)
                dl.add_line(ca, cb, _col(theme.node_core, 0.98 * a), 1.45)

            # Delicate center micro-crosshair (+) & luminous pinpoint with contrast halo
            ch = 5.0 + kick * 2.2
            dl.add_line(imgui.ImVec2(cx - ch, cy), imgui.ImVec2(cx + ch, cy), _ink_col(0.68 * a), 2.8)
            dl.add_line(imgui.ImVec2(cx, cy - ch), imgui.ImVec2(cx, cy + ch), _ink_col(0.68 * a), 2.8)
            dl.add_line(imgui.ImVec2(cx - ch, cy), imgui.ImVec2(cx + ch, cy), _col(theme.line_core, 0.92 * a), 1.2)
            dl.add_line(imgui.ImVec2(cx, cy - ch), imgui.ImVec2(cx, cy + ch), _col(theme.line_core, 0.92 * a), 1.2)
            dl.add_circle_filled(imgui.ImVec2(cx, cy), 4.6 + kick * 2.0, _ink_col(0.45 * a), 12)
            dl.add_circle_filled(imgui.ImVec2(cx, cy), 4.2 + kick * 2.0, _col(theme.line_glow, 0.42 * a), 12)
            dl.add_circle_filled(imgui.ImVec2(cx, cy), 2.0, _col(theme.node_core, 1.0 * a), 10)

            if settings.show_labels:
                id_txt = f"{tr.track_id:02d}"
                coord_txt = f"{int(tr.cx * 99):02d}.{int(tr.cy * 99):02d}"
                badge_w = 22.0
                badge_h = 15.0
                c_sz = imgui.calc_text_size(coord_txt)
                tot_w = badge_w + c_sz.x + 10.0
                lx = max(vp_bounds[0] + 6.0, min(vp_bounds[2] - tot_w - 6.0, x0))
                ly = max(vp_bounds[1] + 26.0, min(vp_bounds[3] - badge_h - 6.0, y0 - badge_h - 3.0))
                # Unified dark glass coordinate pill attached cleanly to luminous ID badge
                dl.add_rect_filled(
                    imgui.ImVec2(lx, ly),
                    imgui.ImVec2(lx + tot_w, ly + badge_h),
                    _ink_col(0.78 * a),
                    2.0,
                )
                dl.add_rect_filled(
                    imgui.ImVec2(lx, ly),
                    imgui.ImVec2(lx + badge_w, ly + badge_h),
                    _col(theme.line_core, 0.95 * a),
                    2.0,
                )
                dl.add_rect(
                    imgui.ImVec2(lx, ly),
                    imgui.ImVec2(lx + tot_w, ly + badge_h),
                    _col(theme.line_core, 0.52 * a),
                    rounding=2.0,
                    thickness=1.0,
                )
                dl.add_text(
                    imgui.ImVec2(lx + 4.0, ly + 0.5),
                    _ink_col(0.98 * a),
                    id_txt,
                )
                dl.add_text(
                    imgui.ImVec2(lx + badge_w + 5.0, ly + 0.5),
                    _col(theme.node_core, 0.95 * a),
                    coord_txt,
                )

        elif style == 1:
            # === STYLE 1: SWISS MINIMAL HAIRLINE (Architectural Gallery Precision) ===
            dl.add_rect_filled(p0, p1, _col(theme.line_glow, 0.05 * a))
            dl.add_rect(p0, p1, _ink_col(0.68 * a), rounding=0.0, thickness=3.0)
            dl.add_rect(p0, p1, _col(theme.line_glow, 0.34 * a), rounding=0.0, thickness=2.8)
            dl.add_rect(p0, p1, _col(theme.node_core, 0.92 * a), rounding=0.0, thickness=1.2)
            # Crisp 4px anchor squares at top-left & bottom-right
            dl.add_rect_filled(imgui.ImVec2(x0 - 2.5, y0 - 2.5), imgui.ImVec2(x0 + 3.5, y0 + 3.5), _ink_col(0.82 * a))
            dl.add_rect_filled(imgui.ImVec2(x0 - 1.5, y0 - 1.5), imgui.ImVec2(x0 + 2.5, y0 + 2.5), _col(theme.node_core, a))
            dl.add_rect_filled(imgui.ImVec2(x1 - 3.5, y1 - 3.5), imgui.ImVec2(x1 + 2.5, y1 + 2.5), _ink_col(0.82 * a))
            dl.add_rect_filled(imgui.ImVec2(x1 - 2.5, y1 - 2.5), imgui.ImVec2(x1 + 1.5, y1 + 1.5), _col(theme.node_core, a))
            dl.add_circle_filled(imgui.ImVec2(cx, cy), 3.4, _ink_col(0.75 * a), 8)
            dl.add_circle_filled(imgui.ImVec2(cx, cy), 2.0, _col(theme.node_core, 0.96 * a), 8)
            if settings.show_labels:
                _draw_micro_pill(dl, imgui.ImVec2(x0, y0 - 19.0), f"//{tr.track_id:02d}  {int(tr.area * 1000):03d}", theme, a, bounds=vp_bounds)

        elif style == 2:
            # === STYLE 2: ORGANIC CONTOUR / AURA (Chaikin-Smoothed Real Silhouette Hull!) ===
            if len(tr.hull) >= 4:
                spts = [
                    imgui.ImVec2(*_blob_uv_to_screen(hu, hv, img_pos, disp_w, disp_h, zoom, pan_x, pan_y))
                    for hu, hv in tr.hull
                ]
                dl.add_polyline(spts, _ink_col(0.68 * a), thickness=4.2, flags=imgui.ImDrawFlags_.closed)
                dl.add_polyline(spts, _col(theme.line_glow, 0.42 * a), thickness=6.2 + kick * 2.5, flags=imgui.ImDrawFlags_.closed)
                dl.add_polyline(spts, _col(theme.line_core, 0.90 * a), thickness=2.0, flags=imgui.ImDrawFlags_.closed)
                dl.add_polyline(spts, _col(theme.node_core, 0.96 * a), thickness=1.15, flags=imgui.ImDrawFlags_.closed)
            else:
                rad = max(12.0, math.sqrt(bw * bh) * 0.46)
                dl.add_circle(imgui.ImVec2(cx, cy), rad, _ink_col(0.68 * a), 28, 3.8)
                dl.add_circle(imgui.ImVec2(cx, cy), rad, _col(theme.line_glow, 0.48 * a), 28, 4.4)
                dl.add_circle(imgui.ImVec2(cx, cy), rad, _col(theme.node_core, 0.94 * a), 28, 1.35)

            # Soft glowing core + orbiting satellite micro-dot
            dl.add_circle_filled(imgui.ImVec2(cx, cy), 5.5 + kick * 2.5, _ink_col(0.55 * a), 14)
            dl.add_circle_filled(imgui.ImVec2(cx, cy), 5.5 + kick * 3.0, _col(theme.line_glow, 0.45 * a), 14)
            dl.add_circle_filled(imgui.ImVec2(cx, cy), 2.2, _col(theme.node_core, 0.98 * a), 10)
            orb_r = max(13.0, min(bw, bh) * 0.56)
            orb_ang = time_sec * (1.8 + (tr.track_id % 3) * 0.5) + tr.track_id
            ox = cx + math.cos(orb_ang) * orb_r
            oy = cy + math.sin(orb_ang) * orb_r
            dl.add_circle_filled(imgui.ImVec2(ox, oy), 3.8, _ink_col(0.75 * a), 8)
            dl.add_circle_filled(imgui.ImVec2(ox, oy), 2.4, _col(theme.node_core, 0.96 * a), 8)

            if settings.show_labels:
                _draw_micro_pill(dl, imgui.ImVec2(cx + 8, cy - 18), f"B{tr.track_id:02d}", theme, a, accent_bar=True, bounds=vp_bounds)

        elif style == 3:
            # === STYLE 3: CONSTELLATION NODES (Pure Star-Map Rings — Zero Boxes!) ===
            mass_r = max(10.0, min(42.0, math.sqrt(bw * bh) * 0.34)) * (1.0 + kick * 0.14)
            dl.add_circle_filled(imgui.ImVec2(cx, cy), mass_r * 0.85, _col(theme.line_glow, 0.16 * a), 24)
            dl.add_circle(imgui.ImVec2(cx, cy), mass_r, _ink_col(0.68 * a), 28, 3.6)
            dl.add_circle(imgui.ImVec2(cx, cy), mass_r, _col(theme.line_glow, 0.48 * a), 28, 3.6)
            dl.add_circle(imgui.ImVec2(cx, cy), mass_r, _col(theme.line_core, 0.94 * a), 28, 1.35)
            # Rotating inner arc
            arc_r = mass_r * 0.62
            a0 = time_sec * 1.6 * (1.0 if tr.track_id % 2 == 0 else -1.0) + tr.track_id
            arc_pts = [
                imgui.ImVec2(cx + math.cos(a0 + 2.2 * (k / 10.0)) * arc_r, cy + math.sin(a0 + 2.2 * (k / 10.0)) * arc_r)
                for k in range(11)
            ]
            dl.add_polyline(arc_pts, _ink_col(0.65 * a), thickness=3.0, flags=0)
            dl.add_polyline(arc_pts, _col(theme.node_core, 0.96 * a), thickness=1.45, flags=0)
            dl.add_circle_filled(imgui.ImVec2(cx, cy), 4.0 + kick * 1.2, _ink_col(0.65 * a), 10)
            dl.add_circle_filled(imgui.ImVec2(cx, cy), 2.6 + kick * 1.2, _col(theme.node_core, a), 10)

            if settings.show_labels:
                _draw_micro_pill(dl, imgui.ImVec2(cx + mass_r + 6, cy - 9), f"N{tr.track_id:02d}", theme, a, accent_bar=True, bounds=vp_bounds)

        else:
            # === STYLE 4: CYBER TARGETING PRO (Sleek Adaptive L-Brackets + Velocity Vector + Diamond) ===
            arm = max(6.0, min(18.0, min(bw, bh) * 0.28))
            # Dark ink backing behind L-brackets so they remain hyper-sharp over bright bloom
            ink_corners = [
                (imgui.ImVec2(x0, y0), imgui.ImVec2(x0 + arm, y0)),
                (imgui.ImVec2(x0, y0), imgui.ImVec2(x0, y0 + arm)),
                (imgui.ImVec2(x1, y0), imgui.ImVec2(x1 - arm, y0)),
                (imgui.ImVec2(x1, y0), imgui.ImVec2(x1, y0 + arm)),
                (imgui.ImVec2(x0, y1), imgui.ImVec2(x0 + arm, y1)),
                (imgui.ImVec2(x0, y1), imgui.ImVec2(x0, y1 - arm)),
                (imgui.ImVec2(x1, y1), imgui.ImVec2(x1 - arm, y1)),
                (imgui.ImVec2(x1, y1), imgui.ImVec2(x1, y1 - arm)),
            ]
            for ca, cb in ink_corners:
                dl.add_line(ca, cb, _ink_col(0.76 * a), 3.8)
            draw_corner_brackets(dl, p0, p1, theme, arm=arm, kick_pulse=kick * 0.5)
            # Centered micro diamond ◇
            dr = 4.5 + kick * 1.8
            diamond = [
                imgui.ImVec2(cx, cy - dr),
                imgui.ImVec2(cx + dr, cy),
                imgui.ImVec2(cx, cy + dr),
                imgui.ImVec2(cx - dr, cy),
            ]
            dl.add_polyline(diamond, _ink_col(0.75 * a), thickness=3.0, flags=imgui.ImDrawFlags_.closed)
            dl.add_polyline(diamond, _col(theme.node_core, 0.96 * a), thickness=1.35, flags=imgui.ImDrawFlags_.closed)

            # Velocity vector line showing real physical direction of blob motion!
            spd = math.hypot(tr.vx, tr.vy)
            if spd > 0.04:
                vx_px = tr.vx * disp_w * zoom * 0.14
                vy_px = -tr.vy * disp_h * zoom * 0.14
                dl.add_line(imgui.ImVec2(cx, cy), imgui.ImVec2(cx + vx_px, cy + vy_px), _ink_col(0.72 * a), 3.2)
                dl.add_line(imgui.ImVec2(cx, cy), imgui.ImVec2(cx + vx_px, cy + vy_px), _col(theme.accent_alt, 0.95 * a), 1.55)
                dl.add_circle_filled(imgui.ImVec2(cx + vx_px, cy + vy_px), 3.4, _ink_col(0.78 * a), 8)
                dl.add_circle_filled(imgui.ImVec2(cx + vx_px, cy + vy_px), 2.1, _col(theme.node_core, 0.98 * a), 8)

            if settings.show_labels:
                _draw_micro_pill(dl, imgui.ImVec2(x0, y0 - 19.0), f"#{tr.track_id:02d} v:{spd:.1f}", theme, a, accent_bar=True, bounds=vp_bounds)

    dl.pop_clip_rect()


def draw_drop_alert_banner(
    dl: imgui.ImDrawList,
    img_pos: imgui.ImVec2,
    disp_w: float,
    disp_h: float,
    theme: SpecimenTheme,
    drop_intensity: float,
    bpm: float,
    note: str,
) -> None:
    """Animated expanding holographic // DROP DETECTED // alert banner on major drops!"""
    if drop_intensity <= 0.08:
        return
    alpha = min(1.0, drop_intensity * 1.35)
    expand = 1.0 - (1.0 - min(1.0, drop_intensity)) ** 2
    cx = img_pos.x + disp_w * 0.5
    cy = img_pos.y + 42.0
    half_w = 135.0 + expand * 55.0
    half_h = 16.0

    p0 = imgui.ImVec2(cx - half_w, cy - half_h)
    p1 = imgui.ImVec2(cx + half_w, cy + half_h)
    dl.add_rect_filled(p0, p1, _col(theme.bg, 0.84 * alpha))

    # Expanding side laser wings
    wing = 65.0 * expand
    draw_glow_line(dl, imgui.ImVec2(p0.x - wing, cy), imgui.ImVec2(p0.x, cy), theme, alpha)
    draw_glow_line(dl, imgui.ImVec2(p1.x, cy), imgui.ImVec2(p1.x + wing, cy), theme, alpha)
    draw_specimen_box(dl, p0, p1, theme, with_nodes=True, intensity=1.25 * alpha, kick_pulse=drop_intensity)

    msg = f"// DROP DETECTED // {bpm:03.0f}BPM // KEY:{note}"
    ts = imgui.calc_text_size(msg)
    dl.add_text(imgui.ImVec2(cx - ts.x * 0.5, cy - ts.y * 0.5), _col(theme.node_core, alpha), msg)


def draw_beat_sequencer(
    dl: imgui.ImDrawList,
    origin: imgui.ImVec2,
    beat_index: int,
    beat_phase: float,
    kick_env: float,
    theme: SpecimenTheme,
) -> None:
    """4 luminous quarter-note beat LEDs [● ○ ○ ○] synchronized to the PLL beat clock."""
    pulse = max(0.25, math.exp(-beat_phase * 3.8))
    for i in range(4):
        cx = origin.x + i * 16.0 + 6.0
        cy = origin.y + 7.0
        is_cur = i == (beat_index % 4)
        if is_cur:
            r = 4.2 + kick_env * 1.5
            dl.add_circle_filled(imgui.ImVec2(cx, cy), r * 2.2, _col(theme.line_glow, 0.50 * pulse), 12)
            dl.add_circle_filled(imgui.ImVec2(cx, cy), r, _col(theme.node_core, 0.98), 12)
        else:
            dl.add_circle(imgui.ImVec2(cx, cy), 3.6, _col(theme.line_core, 0.55), 12, 1.0)


def draw_chroma_bars(
    dl: imgui.ImDrawList,
    p_min: imgui.ImVec2,
    p_max: imgui.ImVec2,
    chroma: np.ndarray,
    note_names: List[str],
    theme: SpecimenTheme,
) -> None:
    """Glowing 12-semitone musical Chroma bar strip (C .. B)."""
    dl.add_rect_filled(p_min, p_max, _col(theme.bg, 0.95))
    x0, y0 = p_min.x + 5.0, p_min.y + 4.0
    x1, y1 = p_max.x - 5.0, p_max.y - 16.0
    w = max(24.0, x1 - x0)
    h = max(14.0, y1 - y0)
    step = w / 12.0
    dom_idx = int(np.argmax(chroma)) if len(chroma) == 12 and float(np.max(chroma)) > 0.05 else -1

    for i in range(min(12, len(chroma))):
        val = float(np.clip(chroma[i], 0.0, 1.0))
        bx0 = x0 + i * step + 1.5
        bx1 = x0 + (i + 1) * step - 1.5
        bh = max(2.0, val * h)
        by0 = y1 - bh
        is_dom = i == dom_idx
        fill_c = _col(theme.node_core if is_dom else theme.line_core, 0.88 if is_dom else 0.52)
        if is_dom:
            dl.add_rect_filled(imgui.ImVec2(bx0 - 1, by0 - 1), imgui.ImVec2(bx1 + 1, y1), _col(theme.line_glow, 0.58))
        dl.add_rect_filled(imgui.ImVec2(bx0, by0), imgui.ImVec2(bx1, y1), fill_c)
        lbl = note_names[i] if i < len(note_names) else ""
        dl.add_text(
            imgui.ImVec2(bx0 + 1.0, y1 + 2.0),
            _col(theme.node_core if is_dom else theme.text_dim, 0.95),
            lbl,
        )

    draw_specimen_box(dl, p_min, p_max, theme, with_nodes=False, intensity=0.75)


def draw_art_typography_overlay(
    dl: imgui.ImDrawList,
    img_pos: imgui.ImVec2,
    disp_w: float,
    disp_h: float,
    theme: SpecimenTheme,
    style: int,
    title: str,
    subtitle: str,
    bpm: float,
    note: str,
    section_name: str,
    time_sec: float,
    kick_pulse: float,
    opacity: float = 0.88,
) -> None:
    """
    ART-VIDEO-MUSIC Editorial Typography Overlay rendered directly onto the visual canvas:
      1 = Specimen Editorial Stamp (Swiss glowing top-left archival stamp + live BPM/Key)
      2 = Kinetic Center Poster (Giant bold audio-pulsing poster lettering + corner rules)
      3 = Cinema Ticker Ribbon (Sleek bottom editorial ticker ribbon with live track & audio telemetry)
      4 = Gallery Frame + Timecode (Minimalist exhibition passe-partout frame + timecode/signature)
    """
    if style <= 0 or opacity <= 0.02:
        return

    a = max(0.0, min(1.0, opacity))
    title_up = (title or "SPECIMEN // LIVE").upper()[:36]
    sub_up = (subtitle or "AUDIOVISUAL PERFORMANCE").upper()[:44]

    dl.push_clip_rect(img_pos, imgui.ImVec2(img_pos.x + disp_w, img_pos.y + disp_h), True)

    if style == 1:
        # === 1: SPECIMEN EDITORIAL STAMP (Top-left architectural Swiss tag) ===
        x0 = img_pos.x + 22.0
        y0 = img_pos.y + 22.0
        imgui.push_font(None, 22.0)
        ts_t = imgui.calc_text_size(title_up)
        imgui.pop_font()
        meta = f"{sub_up}  //  {bpm:03.0f}BPM  //  KEY:{note}  [{section_name}]"
        ts_m = imgui.calc_text_size(meta)
        box_w = min(disp_w - 44.0, max(ts_t.x, ts_m.x) + 28.0)
        box_h = 56.0
        p0, p1 = imgui.ImVec2(x0, y0), imgui.ImVec2(x0 + box_w, y0 + box_h)
        dl.add_rect_filled(p0, p1, _ink_col(0.70 * a), 2.0)
        dl.add_rect_filled(p0, imgui.ImVec2(x0 + 3.5, p1.y), _col(theme.line_core, 0.96 * a))
        draw_specimen_box(dl, p0, p1, theme, with_nodes=True, intensity=0.82 * a, kick_pulse=kick_pulse * 0.6)
        imgui.push_font(None, 22.0)
        dl.add_text(imgui.ImVec2(x0 + 13.0, y0 + 7.0), _col(theme.node_core, 0.98 * a), title_up)
        imgui.pop_font()
        dl.add_text(imgui.ImVec2(x0 + 13.0, y0 + 35.0), _col(theme.line_core, 0.92 * a), meta)

    elif style == 2:
        # === 2: KINETIC CENTER POSTER (Brutalist beat-pulsing exhibition typography!) ===
        cx = img_pos.x + disp_w * 0.5
        cy = img_pos.y + disp_h * 0.5
        font_sz = min(56.0, max(28.0, disp_w * 0.044)) + kick_pulse * 6.0
        imgui.push_font(None, font_sz)
        ts_t = imgui.calc_text_size(title_up)
        tx = cx - ts_t.x * 0.5
        ty = cy - ts_t.y * 0.5 - 8.0
        # Crisp dark contrast outline shadow so title punches cleanly over white HDR bloom
        for ox, oy in ((-2, -2), (2, -2), (-2, 2), (2, 2), (0, 3)):
            dl.add_text(imgui.ImVec2(tx + ox, ty + oy), _ink_col(0.75 * a), title_up)
        dl.add_text(imgui.ImVec2(tx, ty), _col(theme.node_core, 0.96 * a), title_up)
        imgui.pop_font()

        sub_line = f"// {sub_up} // {bpm:03.0f} BPM // {note} //"
        ts_s = imgui.calc_text_size(sub_line)
        sx = cx - ts_s.x * 0.5
        sy = ty + ts_t.y + 8.0
        dl.add_rect_filled(imgui.ImVec2(sx - 10, sy - 3), imgui.ImVec2(sx + ts_s.x + 10, sy + ts_s.y + 4), _ink_col(0.74 * a), 2.0)
        dl.add_rect(imgui.ImVec2(sx - 10, sy - 3), imgui.ImVec2(sx + ts_s.x + 10, sy + ts_s.y + 4), _col(theme.line_core, 0.65 * a), rounding=2.0, thickness=1.0)
        dl.add_text(imgui.ImVec2(sx, sy), _col(theme.line_core, 0.96 * a), sub_line)

    elif style == 3:
        # === 3: CINEMA TICKER RIBBON (Animated scrolling editorial footer strip) ===
        bar_h = 28.0
        y0 = img_pos.y + disp_h - bar_h - 14.0
        x0 = img_pos.x + 18.0
        x1 = img_pos.x + disp_w - 18.0
        p0, p1 = imgui.ImVec2(x0, y0), imgui.ImVec2(x1, y0 + bar_h)
        dl.add_rect_filled(p0, p1, _ink_col(0.76 * a))
        draw_glow_line(dl, p0, imgui.ImVec2(x1, y0), theme, 0.88 * a, kick_pulse=kick_pulse * 0.5)
        draw_glow_line(dl, imgui.ImVec2(x0, p1.y), p1, theme, 0.65 * a)
        ticker_raw = f"  ///  {title_up}  —  {sub_up}  //  {bpm:03.0f} BPM  //  KEY:{note}  //  SECTION:{section_name}  "
        max_chars = max(36, int((x1 - x0 - 24.0) / 8.5))
        tick_str = marquee_text(ticker_raw * 2, max_chars, time_sec, speed=6.0)
        dl.add_text(imgui.ImVec2(x0 + 12.0, y0 + 6.0), _col(theme.node_core, 0.96 * a), tick_str)

    else:
        # === 4: GALLERY FRAME + TIMECODE (Exhibition passe-partout + live metadata) ===
        pad = 18.0
        f0 = imgui.ImVec2(img_pos.x + pad, img_pos.y + pad)
        f1 = imgui.ImVec2(img_pos.x + disp_w - pad, img_pos.y + disp_h - pad)
        dl.add_rect(f0, f1, _ink_col(0.68 * a), rounding=0.0, thickness=3.0)
        dl.add_rect(f0, f1, _col(theme.line_core, 0.72 * a), rounding=0.0, thickness=1.0)
        draw_corner_brackets(dl, f0, f1, theme, arm=18.0, kick_pulse=kick_pulse * 0.5)

        mins = int(time_sec // 60)
        secs = int(time_sec % 60)
        frms = int((time_sec % 1.0) * 60)
        tc = f"REC {mins:02d}:{secs:02d}:{frms:02d} // {bpm:03.0f}BPM // {note}"
        _draw_micro_pill(dl, imgui.ImVec2(f0.x + 12, f0.y + 10), f"{title_up}  |  {sub_up}", theme, a, accent_bar=True)
        tc_w = imgui.calc_text_size(tc).x + 18.0
        _draw_micro_pill(dl, imgui.ImVec2(f1.x - tc_w - 10, f0.y + 10), tc, theme, a, accent_bar=False)

    dl.pop_clip_rect()
