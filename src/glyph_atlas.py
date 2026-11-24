from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


CHARSET_LABELS: Dict[str, str] = {
    "ascii_detailed": "ASCII Detallado (68 glifos)",
    "mono_fine": "Monocromo Fino (Creation Noir)",
    "polyglot": "Polyglot Cinético (Cirílico/Griego/Latín)",
    "ascii_classic": "ASCII Clásico (12 glifos)",
    "blocks": "Bloques Unicode (░▒▓█)",
    "braille": "Braille Ultra-Detalle (32 niveles)",
    "katakana": "Katakana / Cyber Matrix",
    "geometric": "Geométrico / HUD Técnico",
    "custom": "Personalizado",
}

CHARSET_ORDER = list(CHARSET_LABELS.keys())

_PRESET_STRINGS: Dict[str, str] = {
    "ascii_detailed": " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$",
    "mono_fine": " .,:;-~=+*!?#%@&MWB8Q0OZmwqpdbkhao",
    "polyglot": " .·:°~^;+=*!?cnoaeuxzvwrgpsdfhklmtyAEIOUYZBCDFGHJKLMNPQRSTVWXÆØßÑÞÐÇÜÖÄÉÈÊËÎÏÔÛÝαβγδεζηθικλμνξπρστυφχψωΓΔΘΛΞΠΣΦΨΩабвгдеёжзийклмнопрстуфхцчшщъыьэюяБГДЁЖЗИЙЛПФЦЧШЩЪЫЭЮЯ",
    "ascii_classic": " .:-=+*#%@MW",
    "katakana": " .·:ｰ-=+ｦｱｳｴｵｶｷｹｺｻｼｽｾｿﾀﾂﾃﾅﾆﾇﾈﾊﾋﾎﾏﾐﾑﾒﾓﾔﾕﾗﾘﾜ0189@#",
    "geometric": " .·:°+×*▫▪◇◈◆■▓█",
}

EDGE_GLYPHS = ("|", "-", "/", "\\")


@dataclass
class GlyphAtlasData:
    pixels: np.ndarray  # uint8 (H, W)
    width: int
    height: int
    num_ramp: int
    total_glyphs: int
    font_used: str


def _windows_fonts_dir() -> Path:
    windir = os.environ.get("WINDIR", r"C:\Windows")
    return Path(windir) / "Fonts"


def find_monospace_font(custom_path: str = "", prefer_cjk: bool = False) -> Tuple[str, str]:
    """Returns (font_file_path_or_empty, friendly_name)."""
    if custom_path and Path(custom_path).is_file():
        return custom_path, Path(custom_path).name

    fdir = _windows_fonts_dir()
    if prefer_cjk:
        candidates = [
            "msgothic.ttc",
            "YuGothM.ttc",
            "YuGothB.ttc",
            "meiryo.ttc",
            "malgun.ttf",
            "CascadiaMono.ttf",
            "consola.ttf",
        ]
    else:
        candidates = [
            "CascadiaMono.ttf",
            "CascadiaCode.ttf",
            "consola.ttf",
            "consolab.ttf",
            "DejaVuSansMono.ttf",
            "lucon.ttf",
            "cour.ttf",
            "seguisym.ttf",
        ]

    for fname in candidates:
        p = fdir / fname
        if p.is_file():
            return str(p), fname
    return "", "Default Bitmap"


def _render_char_tile(
    ch: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    tile_w: int,
    tile_h: int,
) -> np.ndarray:
    img = Image.new("L", (tile_w, tile_h), 0)
    if ch == " ":
        return np.zeros((tile_h, tile_w), dtype=np.uint8)
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), ch, font=font)
    gw = bbox[2] - bbox[0]
    gh = bbox[3] - bbox[1]
    x = (tile_w - gw) / 2.0 - bbox[0]
    y = (tile_h - gh) / 2.0 - bbox[1]
    draw.text((x, y), ch, fill=255, font=font)
    return np.asarray(img, dtype=np.uint8)


def _render_braille_tiles(tile_w: int, tile_h: int) -> List[np.ndarray]:
    """Procedurally renders 32 density-ordered crisp 2x4 Braille dot patterns."""
    dot_order = [
        (0, 3), (1, 3), (0, 2), (1, 2),
        (0, 1), (1, 1), (0, 0), (1, 0),
    ]
    tiles: List[np.ndarray] = [np.zeros((tile_h, tile_w), dtype=np.uint8)]
    cx = [tile_w * 0.30, tile_w * 0.70]
    cy = [tile_h * 0.14, tile_h * 0.38, tile_h * 0.62, tile_h * 0.86]

    masks_and_radii = [
        (0b00000001, 0.14), (0b00000100, 0.16), (0b00000011, 0.16), (0b00010001, 0.16),
        (0b00010010, 0.17), (0b00001100, 0.17), (0b00010101, 0.17), (0b00100110, 0.17),
        (0b00011011, 0.17), (0b01010101, 0.17), (0b00110011, 0.18), (0b01011010, 0.18),
        (0b00111100, 0.18), (0b01101101, 0.18), (0b01110110, 0.18), (0b01011111, 0.18),
        (0b10111101, 0.18), (0b01111110, 0.18), (0b11011110, 0.19), (0b11101111, 0.19),
        (0b11110111, 0.19), (0b11111101, 0.19), (0b11111111, 0.17), (0b11111111, 0.19),
        (0b11111111, 0.21), (0b11111111, 0.23), (0b11111111, 0.25), (0b11111111, 0.27),
        (0b11111111, 0.29), (0b11111111, 0.31), (0b11111111, 0.34),
    ]

    yy, xx = np.mgrid[0:tile_h, 0:tile_w].astype(np.float32)
    for mask, rad_frac in masks_and_radii:
        canvas = np.zeros((tile_h, tile_w), dtype=np.float32)
        rad = tile_w * rad_frac
        for bit, (dc, dr) in enumerate(dot_order):
            if mask & (1 << bit):
                dist = np.sqrt((xx - cx[dc]) ** 2 + (yy - cy[dr]) ** 2)
                dot = np.clip((rad + 1.2 - dist) / 2.2, 0.0, 1.0)
                canvas = np.maximum(canvas, dot)
        tiles.append((canvas * 255.0).astype(np.uint8))
    return tiles


def _render_block_tiles(tile_w: int, tile_h: int) -> List[np.ndarray]:
    """Procedurally renders crisp retro Unicode shade/block tiles."""
    tiles: List[np.ndarray] = [np.zeros((tile_h, tile_w), dtype=np.uint8)]
    yy, xx = np.mgrid[0:tile_h, 0:tile_w]

    t1 = np.zeros((tile_h, tile_w), dtype=np.uint8)
    t1[((yy // 4) % 2 == 0) & ((xx // 4) % 2 == 0)] = 190
    tiles.append(t1)

    t2 = np.zeros((tile_h, tile_w), dtype=np.uint8)
    t2[((yy // 3) + (xx // 3)) % 3 == 0] = 215
    tiles.append(t2)

    t3 = np.zeros((tile_h, tile_w), dtype=np.uint8)
    t3[((yy // 3) + (xx // 3)) % 2 == 0] = 230
    tiles.append(t3)

    t4 = np.zeros((tile_h, tile_w), dtype=np.uint8)
    t4[(yy % 4) < 2] = 240
    tiles.append(t4)

    t5 = np.full((tile_h, tile_w), 245, dtype=np.uint8)
    t5[((yy // 3) % 2 == 0) & ((xx // 3) % 2 == 0)] = 40
    tiles.append(t5)

    for margin_frac in (0.30, 0.24, 0.18, 0.12, 0.07, 0.03):
        t = np.zeros((tile_h, tile_w), dtype=np.uint8)
        mx = max(1, int(tile_w * margin_frac))
        my = max(1, int(tile_h * margin_frac))
        t[my : tile_h - my, mx : tile_w - mx] = 255
        tiles.append(t)

    return tiles


def _render_edge_tiles(tile_w: int, tile_h: int) -> List[np.ndarray]:
    """Renders crisp directional edge strokes: |, -, /, \\."""
    tiles: List[np.ndarray] = []
    stroke = max(2, int(round(tile_w * 0.14)))
    for sym in EDGE_GLYPHS:
        img = Image.new("L", (tile_w, tile_h), 0)
        draw = ImageDraw.Draw(img)
        pad_x = int(tile_w * 0.14)
        pad_y = int(tile_h * 0.10)
        if sym == "|":
            cx = tile_w // 2
            draw.line([(cx, pad_y), (cx, tile_h - pad_y)], fill=255, width=stroke)
        elif sym == "-":
            cy = tile_h // 2
            draw.line([(pad_x, cy), (tile_w - pad_x, cy)], fill=255, width=stroke)
        elif sym == "/":
            draw.line([(pad_x, tile_h - pad_y), (tile_w - pad_x, pad_y)], fill=255, width=stroke)
        else:
            draw.line([(pad_x, pad_y), (tile_w - pad_x, tile_h - pad_y)], fill=255, width=stroke)
        tiles.append(np.asarray(img, dtype=np.uint8))
    return tiles


def build_glyph_atlas(
    charset_id: str,
    custom_charset: str = " .:-=+*#%@",
    font_path: str = "",
    tile_w: int = 32,
    tile_h: int = 48,
) -> GlyphAtlasData:
    """Builds a density-calibrated horizontal R8 glyph atlas strip + 4 directional edge glyphs."""
    prefer_cjk = charset_id == "katakana"
    resolved_font_path, friendly_name = find_monospace_font(font_path, prefer_cjk=prefer_cjk)

    font_size = int(tile_h * 0.78)
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont
    try:
        if resolved_font_path:
            font = ImageFont.truetype(resolved_font_path, size=font_size)
        else:
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()
        friendly_name = "Default Bitmap"

    if charset_id == "braille":
        ramp_tiles = _render_braille_tiles(tile_w, tile_h)
        friendly_name = "Procedural Braille 2x4"
    elif charset_id == "blocks":
        ramp_tiles = _render_block_tiles(tile_w, tile_h)
        friendly_name = "Procedural Unicode Blocks"
    else:
        if charset_id == "custom":
            chars = custom_charset if len(custom_charset.strip()) >= 2 else " .:-=+*#%@"
            if not chars.startswith(" "):
                chars = " " + chars
        else:
            chars = _PRESET_STRINGS.get(charset_id, _PRESET_STRINGS["ascii_detailed"])

        unique_chars: List[str] = [" "]
        seen = {" "}
        for ch in chars:
            if ch not in seen and ch not in ("\n", "\r", "\t"):
                seen.add(ch)
                unique_chars.append(ch)

        rendered: List[Tuple[float, np.ndarray]] = []
        # Detect tofu/missing glyph signature to filter out unsupported characters cleanly
        missing_ref = _render_char_tile("\uffff", font, tile_w, tile_h)
        for idx, ch in enumerate(unique_chars):
            tile = _render_char_tile(ch, font, tile_w, tile_h)
            density = float(tile.mean()) / 255.0
            if idx > 0 and density < 0.002:
                continue
            if idx > 0 and np.array_equal(tile, missing_ref):
                continue
            rendered.append((density, tile))

        # Keep strip width well within GPU max texture width (16384px -> <= 200 glyphs @ 32px)
        if len(rendered) > 160:
            idxs = np.linspace(0, len(rendered) - 1, 160).astype(int)
            rendered = [rendered[i] for i in idxs]

        if len(rendered) < 2:
            rendered = [(0.0, np.zeros((tile_h, tile_w), dtype=np.uint8))] + [
                (0.5, _render_char_tile(c, font, tile_w, tile_h)) for c in ".:-=+*#%@"
            ]

        rendered.sort(key=lambda item: item[0])
        ramp_tiles = [item[1] for item in rendered]

    edge_tiles = _render_edge_tiles(tile_w, tile_h)
    all_tiles = ramp_tiles + edge_tiles
    strip = np.concatenate(all_tiles, axis=1)
    strip = np.ascontiguousarray(strip, dtype=np.uint8)

    return GlyphAtlasData(
        pixels=strip,
        width=int(strip.shape[1]),
        height=int(strip.shape[0]),
        num_ramp=len(ramp_tiles),
        total_glyphs=len(all_tiles),
        font_used=friendly_name,
    )
