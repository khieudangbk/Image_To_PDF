import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from .fonts import FontSet, fit_font_px, get_fontset
from .model import BBox, Figure, Line, Page, TextBlock, Word, center_in, union
from .ocr import OLine, OPar, ocr_paragraphs
from .preprocess import Prepared
from .settings import Settings
from .tables import TableGrid, detect_rules, detect_tables
from .vn_correct import correct_tokens, split_merged

_WORDLIKE = re.compile(r"\w", re.UNICODE)
DARK_INK = 110  # HSV value below which a pixel is printed black ink rather than seal ink
Segment = tuple[OLine, list]  # a (possibly partial) OCR line and its words


def _is_good_word(w, strict: bool) -> bool:
    has_alnum = bool(_WORDLIKE.search(w.text))
    if strict:
        return w.conf >= 55 or (w.conf >= 35 and has_alnum and len(w.text) >= 2)
    if w.conf < 15 or (w.conf < 40 and len(w.text) < 3):
        return False
    return has_alnum or w.conf >= 60


# ---------------------------------------------------------------- bold detection

def _stroke_width(ink: np.ndarray, box: BBox) -> float:
    """Median horizontal ink run length, i.e. the width of vertical stems."""
    x0, y0, x1, y1 = box
    crop = (ink[max(0, y0):y1, max(0, x0):x1] > 0).astype(np.int8)
    if crop.sum() < 10:
        return 0.0
    edges = np.diff(np.pad(crop, ((0, 0), (1, 1))), axis=1)
    runs = np.nonzero(edges == -1)[1] - np.nonzero(edges == 1)[1]
    return float(np.median(runs)) if len(runs) else 0.0


def _mark_bold(lines: list[Line], ink: np.ndarray, fs: FontSet):
    """Bold = stroke width clearly above the page's typical stroke width for that font size."""
    ratios: dict[int, float] = {}
    samples = []
    for ln in lines:
        ink_px = sum(w.bbox[2] - w.bbox[0] for w in ln.words)
        em = sum(fs.width(w.text) for w in ln.words)
        em_px = ink_px / em if em > 0 else ln.x_size
        for w in ln.words:
            r = _stroke_width(ink, w.bbox) / max(em_px, 1.0)
            ratios[id(w)] = r
            if len(w.text) >= 3 and r > 0:
                samples.append(r)
    if len(samples) < 5:
        return
    ref = float(np.percentile(samples, 30))
    for ln in lines:
        for w in ln.words:
            w.bold = ratios[id(w)] > ref * 1.25
        long_words = [w for w in ln.words if len(w.text) >= 2]
        if not long_words:  # single glyphs (digits, bullets) give unreliable stroke estimates
            for w in ln.words:
                w.bold = ratios[id(w)] > ref * 1.6
            continue
        share = sum(w.bold for w in long_words) / len(long_words)
        if share >= 0.6 or share <= 0.2:
            for w in ln.words:
                w.bold = share >= 0.6


# ---------------------------------------------------------------- paragraph splitting

def _split_segments(ol: OLine, words: list) -> list[Segment]:
    words = sorted(words, key=lambda w: w.bbox[0])
    segs, cur = [], [words[0]]
    for prev, w in zip(words, words[1:]):
        if w.bbox[0] - prev.bbox[2] > 2.5 * ol.x_size:
            segs.append((ol, cur))
            cur = []
        cur.append(w)
    segs.append((ol, cur))
    return segs


def _seg_box(seg: Segment) -> BBox:
    return union(w.bbox for w in seg[1])


def split_paragraph(entries: list[Segment]) -> list[list[Segment]]:
    """Splits a Tesseract paragraph into visually separate blocks (columns, large vertical gaps)."""
    groups: list[list[Segment]] = []
    for ol, words in entries:
        for seg in _split_segments(ol, words):
            x0, _, x1, _ = _seg_box(seg)
            best, best_ov = None, 0
            for g in groups:
                gx0 = min(_seg_box(s)[0] for s in g)
                gx1 = max(_seg_box(s)[2] for s in g)
                ov = min(x1, gx1) - max(x0, gx0)
                if ov > best_ov and g[-1][0] is not ol:
                    best, best_ov = g, ov
            if best is None:
                groups.append([seg])
            else:
                best.append(seg)
    out = []
    for g in groups:
        g.sort(key=lambda s: s[0].baseline)
        pitches = [b[0].baseline - a[0].baseline for a, b in zip(g, g[1:])]
        normal = min(pitches) if pitches else 0
        cur = [g[0]]
        for prev, seg, pitch in zip(g, g[1:], pitches):
            xs = max(prev[0].x_size, seg[0].x_size)
            size_jump = xs > 1.3 * min(prev[0].x_size, seg[0].x_size)
            if pitch > 2.2 * xs or pitch > max(1.7 * xs, 1.45 * normal) or size_jump:
                out.append(cur)
                cur = []
            cur.append(seg)
        out.append(cur)
    out.sort(key=lambda b: (round(_seg_box(b[0])[1] / max(b[0][0].x_size, 1.0)), _seg_box(b[0])[0]))
    return out


# ---------------------------------------------------------------- figures

def _blue_mask(color: np.ndarray) -> np.ndarray:
    """Ballpoint-pen blue; light bluish watermarks are excluded by the value limit."""
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    return ((hue >= 95) & (hue <= 135) & (sat > 70) & (val < 210)).astype(np.uint8) * 255


def _pen_word(prep: Prepared, wd) -> bool:
    """Word whose ink is mostly blue pen: handwriting, which OCR reads as garbage."""
    x0, y0, x1, y1 = wd.bbox
    ink = prep.binary[y0:y1, x0:x1] > 0
    if not ink.any():
        return False
    return float((_blue_mask(prep.color[y0:y1, x0:x1]) > 0)[ink].mean()) > 0.4


def _red_mask(color: np.ndarray, strict: bool = False) -> np.ndarray:
    """Red ink (seals/stamps). `strict` keeps only vivid red that cannot be black text lying on
    a coloured background, so it is safe to erase before OCR."""
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    red = (hue < 10) | (hue > 160)
    if strict:
        return (red & (sat > 110) & (val > 120) & (val < 250)).astype(np.uint8) * 255
    return (red & (sat > 70) & (val < 245)).astype(np.uint8) * 255


def _detect_figures(prep: Prepared, good_boxes: list[BBox], rule_mask: np.ndarray,
                    tables: list[TableGrid]) -> list[BBox]:
    h, w = prep.binary.shape
    text_mask = np.zeros((h, w), np.uint8)
    for x0, y0, x1, y1 in good_boxes:
        pad = max(2, (y1 - y0) // 4)
        cv2.rectangle(text_mask, (x0 - pad, y0 - pad), (x1 + pad, y1 + pad), 255, -1)
    for t in tables:
        for x0, y0, x1, y1 in t.cells:
            cv2.rectangle(text_mask, (x0, y0), (x1, y1), 255, -1)
    residual = cv2.bitwise_and(prep.binary, cv2.bitwise_not(cv2.bitwise_or(text_mask, rule_mask)))
    residual = cv2.morphologyEx(residual, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    # Red (seals) and blue (pen: signatures, handwritten dates) ink count even where OCR found
    # words, except well-recognised words themselves.
    colored = cv2.bitwise_or(_red_mask(prep.color), _blue_mask(prep.color))
    colored = cv2.bitwise_and(colored, cv2.bitwise_not(text_mask))
    candidates = cv2.bitwise_or(residual, colored)

    k = max(9, int(min(h, w) * 0.012))
    merged = cv2.dilate(candidates, np.ones((k, k), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
    min_side = max(40, int(min(h, w) * 0.025))
    boxes = []
    for i in range(1, n):
        x, y, cw, ch, _ = stats[i]
        if cw < min_side or ch < min_side:
            continue
        sub = candidates[y:y + ch, x:x + cw] & (lab[y:y + ch, x:x + cw] == i).astype(np.uint8) * 255
        ys, xs = np.nonzero(sub)
        if len(xs) < min_side * 8:
            continue
        x0, x1, y0, y1 = x + xs.min(), x + xs.max() + 1, y + ys.min(), y + ys.max() + 1
        if x1 - x0 < min_side or y1 - y0 < min_side or len(xs) < 0.02 * (x1 - x0) * (y1 - y0):
            continue
        pad = 4
        boxes.append((max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)))
    area = lambda b: (b[2] - b[0]) * (b[3] - b[1])  # noqa: E731
    return [b for b in boxes if not any(o is not b and area(o) > area(b) and _overlap_frac(b, o) > 0.8
                                        for o in boxes)]


def _overlap_frac(inner: BBox, outer: BBox) -> float:
    ix = max(0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    iy = max(0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    area = max(1, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return ix * iy / area


# ---------------------------------------------------------------- alignment

def _alignment(lines: list[Line], c0: float, c1: float, font_px: float) -> str:
    width = max(c1 - c0, 1)
    tol = max(0.6 * font_px, 0.01 * width)
    if len(lines) == 1:
        x0, _, x1, _ = lines[0].bbox
        lg, rg = x0 - c0, c1 - x1
        if abs(lg - rg) < max(tol * 2, 0.04 * width) and lg > 0.08 * width:
            return "center"
        if rg < tol * 2 and lg > 0.3 * width:
            return "right"
        return "left"
    boxes = [ln.bbox for ln in lines]
    lefts = [b[0] for b in boxes[1:]]
    rights = [b[2] for b in boxes[:-1]]
    centers = [(b[0] + b[2]) / 2 for b in boxes]
    left_ok = max(lefts) - min(lefts) < tol
    right_ok = max(rights) - min(rights) < tol
    center_ok = max(centers) - min(centers) < tol * 1.5
    if center_ok and not left_ok:
        return "center"
    if left_ok and right_ok:
        return "justify"
    if right_ok and not left_ok:
        return "right"
    return "left"


# ---------------------------------------------------------------- tables

def _ocr_cells(prep: Prepared, rule_mask: np.ndarray, cells: list[BBox], lang: str,
               gray: np.ndarray | None = None) -> dict[int, list[Segment]]:
    """OCRs all cells of a table in one Tesseract call by stacking the cell crops vertically."""
    gray = prep.gray if gray is None else gray
    gap, pad = 90, 40
    crops, y_cursor, max_w = [], gap, 0
    for idx, (x0, y0, x1, y1) in enumerate(cells):
        cx0, cy0, cx1, cy1 = x0 + 4, y0 + 4, x1 - 4, y1 - 4
        if cx1 - cx0 < 5 or cy1 - cy0 < 5:
            continue
        if np.count_nonzero(prep.binary[cy0:cy1, cx0:cx1] & ~rule_mask[cy0:cy1, cx0:cx1]) < 15:
            continue
        crop = gray[cy0:cy1, cx0:cx1].copy()
        crop[rule_mask[cy0:cy1, cx0:cx1] > 0] = 255
        crops.append((idx, crop, cx0, cy0, y_cursor))
        y_cursor += crop.shape[0] + gap
        max_w = max(max_w, crop.shape[1])
    if not crops:
        return {}
    canvas = np.full((y_cursor, max_w + 2 * pad), 255, np.uint8)
    for _, crop, _, _, yo in crops:
        canvas[yo:yo + crop.shape[0], pad:pad + crop.shape[1]] = crop
    result: dict[int, list[Segment]] = {}
    for par in ocr_paragraphs(canvas, lang, psm=6):
        for ol in par.lines:
            groups: dict[int, list] = {}
            for wd in ol.words:
                cy = (wd.bbox[1] + wd.bbox[3]) / 2
                for k, (_, crop, _, _, yo) in enumerate(crops):
                    if yo - gap / 2 <= cy <= yo + crop.shape[0] + gap / 2:
                        groups.setdefault(k, []).append(wd)
                        break
            for k, words in groups.items():
                idx, _, cx0, cy0, yo = crops[k]
                dx, dy = cx0 - pad, cy0 - yo
                for wd in words:
                    x0, y0, x1, y1 = wd.bbox
                    wd.bbox = (x0 + dx, y0 + dy, x1 + dx, y1 + dy)
                result.setdefault(idx, []).append((OLine(ol.bbox, ol.baseline + dy, ol.x_size, words), words))
    return result


def _dedupe_tables(tables: list[TableGrid], w: int, h: int) -> list[TableGrid]:
    out, claimed = [], []
    for t in sorted(tables, key=lambda t: (t.bbox[2] - t.bbox[0]) * (t.bbox[3] - t.bbox[1])):
        area = (t.bbox[2] - t.bbox[0]) * (t.bbox[3] - t.bbox[1])
        if area > 0.85 * w * h:
            continue
        cells = [c for c in t.cells if not any(_overlap_frac(c, o) > 0.5 for o in claimed)]
        # A frame whose "cells" are mostly other tables is not a table itself.
        if len(cells) >= 2:
            claimed += cells
            out.append(TableGrid(t.bbox, cells))
    return out


def _recover_missed_text(prep: Prepared, rule_mask: np.ndarray, covered: list[BBox], x_size: float,
                         lang: str, tentative: list[BBox], only_in: list[BBox] | None = None,
                         gray: np.ndarray | None = None) -> tuple[list[TextBlock], set[int]]:
    """Second pass: OCR text-shaped ink that the page-level layout analysis skipped.

    `tentative` are figure candidates shaped like a line of text; the indices of those that
    turn out to be readable text are returned so they can be dropped as figures.
    `only_in` limits the search to these regions and `gray` replaces the OCR source image
    (used to read black text under seals from an image with the red ink removed).
    """
    h, w = prep.binary.shape
    ink = cv2.bitwise_and(prep.binary, cv2.bitwise_not(rule_mask))
    if only_in is not None:
        region = np.zeros_like(ink)
        for x0, y0, x1, y1 in only_in:
            region[y0:y1, x0:x1] = 255
        ink = cv2.bitwise_and(ink, region)
    if gray is not None:
        ink[gray >= 250] = 0
    for x0, y0, x1, y1 in covered:
        ink[max(0, y0 - 3):y1 + 3, max(0, x0 - 3):x1 + 3] = 0
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    kx, ky = max(3, int(0.9 * x_size)), max(1, int(0.2 * x_size))
    lines = cv2.dilate(ink, np.ones((ky, kx), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(lines, connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, bw, bh, _ = stats[i]
        x0, y0, x1, y1 = x + kx // 2, y + ky // 2, x + bw - kx // 2, y + bh - ky // 2
        bw, bh = x1 - x0, y1 - y0
        if not (0.5 * x_size <= bh <= 3 * x_size) or bw < 1.5 * bh:
            continue
        if np.count_nonzero(ink[y0:y1, x0:x1]) < 0.08 * bw * bh:
            continue
        p = int(0.3 * x_size)
        boxes.append((max(0, x0 - p), max(0, y0 - p), min(w, x1 + p), min(h, y1 + p)))
    n_regular = len(boxes)
    boxes += tentative
    if not boxes:
        return [], set()
    found = _ocr_cells(prep, rule_mask, boxes, lang, gray)
    out, converted = [], set()
    for idx, segs in found.items():
        segs = [(ol, [wd for wd in ws if wd.conf >= 60 and _WORDLIKE.search(wd.text)]) for ol, ws in segs]
        segs = sorted((s for s in segs if s[1]), key=lambda s: s[0].baseline)
        if not segs:
            continue
        if idx >= n_regular:
            bx = boxes[idx]
            text_area = sum((wd.bbox[2] - wd.bbox[0]) * (wd.bbox[3] - wd.bbox[1]) for _, ws in segs for wd in ws)
            if text_area < 0.35 * (bx[2] - bx[0]) * (bx[3] - bx[1]):
                continue  # mostly not text (e.g. a signature with a stray letter)
            converted.add(idx - n_regular)
        out.append(TextBlock([_to_line(s) for s in segs]))
    return out, converted


def _merge_into_line(blocks: list[TextBlock], extra: Line, x_size: float) -> bool:
    """Puts recovered words into the existing line they belong to (same baseline, adjacent),
    re-joining a word that OCR split in two ("b" + "ăng"). Returns True when merged."""
    ex0, _, ex1, _ = extra.bbox
    for b in blocks:
        for ln in b.lines:
            if abs(ln.baseline - extra.baseline) > 0.5 * x_size:
                continue
            lx0, _, lx1, _ = ln.bbox
            if ex0 > lx1 + 4 * x_size or ex1 < lx0 - 4 * x_size:
                continue
            words = sorted(ln.words + extra.words, key=lambda w: w.bbox[0])
            joined = [words[0]]
            for wd in words[1:]:
                prev = joined[-1]
                if wd.bbox[0] - prev.bbox[2] < 0.12 * x_size and prev.text[-1:].isalpha() and wd.text[:1].isalpha():
                    joined[-1] = Word(prev.text + wd.text, union([prev.bbox, wd.bbox]),
                                      min(prev.conf, wd.conf), prev.bold)
                else:
                    joined.append(wd)
            ln.words = joined
            return True
    return False


def _text_shaped(box: BBox, x_size: float) -> bool:
    """A wide, line-high region (possibly coloured text such as a red title) rather than a
    round seal or a photo; worth an OCR attempt before it is kept as a picture."""
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    return (bh <= 2.6 * x_size and bw >= 1.5 * bh) or (bh <= 6 * x_size and bw >= 2.5 * bh)


def _rule_is_ink(prep: Prepared, r) -> bool:
    """Only dark, neutral strokes are drawn as lines; paper edges, shadows and coloured
    decorative frames are not."""
    n = max(2, int(max(abs(r.x1 - r.x0), abs(r.y1 - r.y0)) // 4))
    xs = np.linspace(r.x0, r.x1, n).astype(int).clip(0, prep.gray.shape[1] - 1)
    ys = np.linspace(r.y0, r.y1, n).astype(int).clip(0, prep.gray.shape[0] - 1)
    hsv = cv2.cvtColor(prep.color[ys, xs][None], cv2.COLOR_BGR2HSV)[0]
    return float(np.median(prep.gray[ys, xs])) < 150 and float(np.median(hsv[:, 1])) < 90


def _block_color(prep: Prepared, block: TextBlock) -> tuple[float, float, float] | None:
    samples = []
    for ln in block.lines:
        for wd in ln.words:
            x0, y0, x1, y1 = wd.bbox
            ink = prep.binary[y0:y1, x0:x1] > 0
            if ink.any():
                samples.append(prep.color[y0:y1, x0:x1][ink])
    if not samples:
        return None
    px = np.concatenate(samples)
    gray = px.mean(1)
    px = px[gray <= np.percentile(gray, 50)]  # stroke centres, not anti-aliased edges
    b, g, r = np.median(px, axis=0)
    hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0, 0]
    if hsv[1] < 60 or hsv[2] < 80:
        return None
    return (r / 255, g / 255, b / 255)


def _figure_crop(prep: Prepared, box: BBox) -> np.ndarray:
    x0, y0, x1, y1 = box
    crop = prep.color[y0:y1, x0:x1].copy()
    # Photos have large smooth areas that are not paper (skin, hair, backdrop); seals and
    # signatures are thin strokes on paper.
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(cv2.cvtColor(prep.raw[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY), cv2.CV_32F)
    if np.mean((np.abs(lap) < 8) & (gray < 225)) > 0.28:
        return prep.raw[y0:y1, x0:x1].copy()
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    ink = (cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) < 170) | ((hsv[:, :, 1] > 80) & (hsv[:, :, 2] < 235))
    crop[~ink] = 255  # drop paper, guilloche and watermark patterns
    return crop


# ---------------------------------------------------------------- text cleanup

_NOISE = {"|", "¦", "®", "©"}
_ONE_LIKE = {"l", "I", "|", "]", "["}


_VN_MARKS = {"̀", "́", "̃", "̉", "̣", "̂", "̆", "̛"}
_PUNCT = set(".,;:!?-–—/\\()[]\"'“”‘’%&+*=@#°<>_…")


def _plausible_chars(text: str) -> bool:
    """Only Latin letters with Vietnamese diacritics, digits and ordinary punctuation."""
    for ch in unicodedata.normalize("NFD", text):
        if not (("a" <= ch.lower() <= "z") or ch.isdigit() or ch in "đĐ" or ch in _VN_MARKS or ch in _PUNCT):
            return False
    return True


def _clean_text(block: TextBlock, fix_diacritics: bool):
    words = [w for ln in block.lines for w in ln.words]
    if block.kind == "cell" and len(words) == 1 and words[0].text in _ONE_LIKE:
        words[0].text = "1"  # a lone vertical stroke in a table cell is almost always the digit
        return
    for ln in block.lines:
        ln.words = [w for w in ln.words
                    if w.text not in _NOISE and (w.conf >= 85 or _plausible_chars(w.text))]
        # a lone dot/colon opening a line is a speck of background pattern, not punctuation
        while len(ln.words) > 1 and ln.words[0].text in {".", ",", ":", ";", "'", "`", "¡"} \
                and ln.words[0].conf < 95:
            ln.words.pop(0)
    block.lines = [ln for ln in block.lines if ln.words]
    words = [w for ln in block.lines for w in ln.words]
    alnum = sum(ch.isalnum() for w in words for ch in w.text)
    if block.kind != "cell" and words and (
            alnum == 0 or (alnum <= 2 and np.mean([w.conf for w in words]) < 85)):
        block.lines = []  # isolated specks next to seals, photos and handwriting
        return
    if fix_diacritics:
        tokens = [w.text for ln in block.lines for w in ln.words]
        groups = iter(split_merged(tokens))
        for ln in block.lines:
            new_words = []
            for w, parts in zip(ln.words, groups):
                new_words += _split_word(w, parts)
            ln.words = new_words
        words = [w for ln in block.lines for w in ln.words]
        for w, t in zip(words, correct_tokens([w.text for w in words])):
            w.text = t


def _split_word(w: Word, parts: list[str]) -> list[Word]:
    """Divides a word box among its parts in proportion to their length, leaving a space-sized gap."""
    if len(parts) == 1:
        return [w]
    x0, y0, x1, y1 = w.bbox
    total = sum(len(p) for p in parts) + 0.6 * (len(parts) - 1)
    unit = (x1 - x0) / total
    out, x = [], float(x0)
    for p in parts:
        width = len(p) * unit
        out.append(Word(p, (int(x), y0, int(x + width), y1), w.conf, w.bold))
        x += width + 0.6 * unit
    return out


# ---------------------------------------------------------------- page assembly

def _to_line(seg: Segment) -> Line:
    ol, words = seg
    words = sorted(words, key=lambda w: w.bbox[0])
    return Line([Word(w.text, w.bbox, w.conf) for w in words], ol.baseline, ol.x_size)


def build_page(source: str, prep: Prepared, settings: Settings) -> Page:
    h, w = prep.binary.shape
    horiz, vert, rules = detect_rules(prep.binary)
    rules = [r for r in rules if _rule_is_ink(prep, r)]
    rule_mask = cv2.dilate(cv2.bitwise_or(horiz, vert), np.ones((3, 3), np.uint8))
    tables = _dedupe_tables(detect_tables(horiz, vert), w, h) if settings.detect_tables else []

    ocr_img = prep.gray.copy()
    ocr_img[rule_mask > 0] = 255
    with ThreadPoolExecutor(max_workers=2) as ex:
        page_future = ex.submit(ocr_paragraphs, ocr_img, settings.lang, 3)
        cell_futures = [ex.submit(_ocr_cells, prep, rule_mask, t.cells, settings.lang) for t in tables]
        pars: list[OPar] = page_future.result()
        cell_results = [f.result() for f in cell_futures]

    all_cells = [c for t in tables for c in t.cells]
    par_entries: list[list[Segment]] = []
    for par in pars:
        entries = []
        for ol in par.lines:
            words = [wd for wd in ol.words if not any(center_in(wd.bbox, c) for c in all_cells)]
            if words:
                entries.append((ol, words))
        if entries:
            par_entries.append(entries)

    # Handwriting filled into a printed line (e.g. the date): OCR is unsure of it and it is
    # taller than the print around it. It is kept as a picture rather than read wrongly.
    handwritten, hand_boxes = set(), []
    for entries in par_entries:
        for ol, ws in entries:
            med_h = float(np.median([x.bbox[3] - x.bbox[1] for x in ol.words]))
            for wd in ws:
                digits = any(ch.isdigit() for ch in wd.text)
                unsure = (digits and wd.conf < 90) or (not _plausible_chars(wd.text) and wd.conf < 75)
                if unsure and wd.bbox[3] - wd.bbox[1] >= 1.12 * med_h:
                    handwritten.add(id(wd))
                    p = max(4, int(0.15 * med_h))
                    x0, y0, x1, y1 = wd.bbox
                    hand_boxes.append((max(0, x0 - p), max(0, y0 - p), min(w, x1 + p), min(h, y1 + p)))

    figure_boxes: list[BBox] = []
    strict_good = [wd for entries in par_entries for _, ws in entries for wd in ws
                   if _is_good_word(wd, True) and id(wd) not in handwritten]
    if settings.keep_figures:
        figure_boxes = _detect_figures(prep, [wd.bbox for wd in strict_good], rule_mask, tables)
        figure_boxes += [b for b in hand_boxes if not any(_overlap_frac(b, f) > 0.5 for f in figure_boxes)]
    strict_ids = {id(wd) for wd in strict_good}

    def keep(wd, ol) -> bool:
        if id(wd) in handwritten or (wd.conf < 90 and _pen_word(prep, wd)):
            return False
        line_conf = float(np.median([x.conf for x in ol.words]))
        in_figure = any(center_in(wd.bbox, f) for f in figure_boxes)
        if in_figure and id(wd) not in strict_ids:
            # text crossing a seal stays text when the rest of its line reads well
            return line_conf >= 80 and wd.conf >= 40 and bool(_WORDLIKE.search(wd.text))
        if _is_good_word(wd, False):
            return True
        # A low-confidence word inside a well-read line is usually a real word with a doubtful
        # accent (the dictionary pass repairs it), not noise.
        return line_conf >= 80 and wd.conf >= 3 and wd.text.isalpha() and len(wd.text) >= 2

    blocks: list[TextBlock] = []
    for entries in par_entries:
        entries = [(ol, [wd for wd in ws if keep(wd, ol)]) for ol, ws in entries]
        entries = [e for e in entries if e[1]]
        if not entries:
            continue
        for group in split_paragraph(entries):
            blocks.append(TextBlock([_to_line(s) for s in group]))

    cell_blocks: list[tuple[int, TextBlock]] = []
    for ti, (t, res) in enumerate(zip(tables, cell_results)):
        for idx, cell in enumerate(t.cells):
            segs = [(ol, [wd for wd in ws if _is_good_word(wd, False)]) for ol, ws in res.get(idx, [])]
            segs = sorted((s for s in segs if s[1]), key=lambda s: s[0].baseline)
            if segs:
                cell_blocks.append((ti, TextBlock([_to_line(s) for s in segs], kind="cell", container=cell)))

    kept_lines = [ln for b in blocks + [b for _, b in cell_blocks] for ln in b.lines]
    if kept_lines:
        x_size = float(np.median([ln.x_size for ln in kept_lines]))
        tentative_idx = [i for i, f in enumerate(figure_boxes) if _text_shaped(f, x_size) and f not in hand_boxes]
        solid = [f for i, f in enumerate(figure_boxes) if i not in tentative_idx]
        covered = [wd.bbox for ln in kept_lines for wd in ln.words] + figure_boxes + all_cells
        recovered, converted = _recover_missed_text(prep, rule_mask, covered, x_size, settings.lang,
                                                    [figure_boxes[i] for i in tentative_idx])
        figure_boxes = solid + [figure_boxes[i] for k, i in enumerate(tentative_idx) if k not in converted]

        # Black text crossing a red seal: read it again from the dark pixels only. Printed black
        # ink is far darker than seal ink, so a brightness cut separates them cleanly.
        red = _red_mask(prep.color)
        seals = [f for f in figure_boxes
                 if np.count_nonzero(red[f[1]:f[3], f[0]:f[2]]) > 0.03 * (f[2] - f[0]) * (f[3] - f[1])]
        if seals:
            no_red = prep.gray.copy()
            value = cv2.cvtColor(prep.color, cv2.COLOR_BGR2HSV)[:, :, 2]
            for x0, y0, x1, y1 in seals:
                sub = no_red[y0:y1, x0:x1]
                sub[value[y0:y1, x0:x1] >= DARK_INK] = 255
            done = [wd.bbox for ln in kept_lines for wd in ln.words] + \
                [wd.bbox for b in recovered for ln in b.lines for wd in ln.words] + all_cells
            under, _ = _recover_missed_text(prep, rule_mask, done, x_size, settings.lang, [],
                                            only_in=seals, gray=no_red)
            recovered += under

        existing = [wd.bbox for ln in kept_lines for wd in ln.words]
        for rb in recovered:
            for ln in rb.lines:
                ln.words = [wd for wd in ln.words if not any(_overlap_frac(wd.bbox, e) > 0.3 for e in existing)]
            rb.lines = [ln for ln in rb.lines if ln.words and not _merge_into_line(blocks, ln, x_size)]
            if not rb.lines:
                continue
            pos = next((i for i, b in enumerate(blocks) if b.bbox[1] > rb.bbox[1]), len(blocks))
            blocks.insert(pos, rb)

    fs = get_fontset("Times New Roman")
    all_blocks = blocks + [b for _, b in cell_blocks]
    for b in all_blocks:
        _clean_text(b, settings.fix_diacritics and "vie" in settings.lang)
    all_blocks = [b for b in all_blocks if b.lines]
    blocks = [b for b in blocks if b.lines]
    cell_blocks = [(i, b) for i, b in cell_blocks if b.lines]
    _mark_bold([ln for b in all_blocks for ln in b.lines],
               cv2.bitwise_and(prep.binary, cv2.bitwise_not(rule_mask)), fs)

    if blocks:
        c0 = float(np.percentile([b.bbox[0] for b in blocks], 10))
        c1 = float(np.percentile([b.bbox[2] for b in blocks], 90))
    else:
        c0, c1 = 0.0, float(w)
    font_px, sizes = {}, []
    for b in all_blocks:
        font_px[id(b)] = f = fit_font_px(b, fs)
        b.color = _block_color(prep, b)
        sizes += [f] * len(b.text)
        if b.kind == "cell":
            b.align = _alignment(b.lines, b.container[0], b.container[2], f)
        else:
            b.align = _alignment(b.lines, min(c0, b.bbox[0]), max(c1, b.bbox[2]), f)
    body = float(np.median(sizes)) if sizes else 0
    for b in blocks:
        short = len(b.lines) <= 3 and len(b.text) <= 160
        if short and font_px[id(b)] >= 1.25 * body:
            b.kind, b.heading_level = "heading", 0
        elif short and b.bold and b.align == "center" and len(b.text) >= 3:
            b.kind, b.heading_level = "heading", 1

    figures = []
    kept_words = [(wd, b.color is not None) for b in all_blocks for ln in b.lines for wd in ln.words]
    for box in figure_boxes:
        x0, y0, x1, y1 = box
        crop = _figure_crop(prep, box)
        for wd, colored in kept_words:  # text drawn as real text must not also appear in the picture
            if box not in hand_boxes and _overlap_frac(wd.bbox, box) > 0:
                a0, b0, a1, b1 = wd.bbox
                sub = crop[max(0, b0 - y0 - 2):max(0, b1 - y0 + 2), max(0, a0 - x0 - 2):max(0, a1 - x0 + 2)]
                if sub.size:
                    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
                    if colored:  # red/blue printed text: remove all of its ink
                        text_ink = hsv[:, :, 2] < 235
                    else:  # black text: remove only dark strokes, a seal underneath stays
                        text_ink = hsv[:, :, 2] < DARK_INK
                    sub[cv2.dilate(text_ink.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0] = 255
        figures.append(Figure(box, crop))
    rules = [r for r in rules if not any(
        center_in((int(r.x0), int(r.y0), int(r.x1), int(r.y1)), f) for f in figure_boxes)]

    ordered = _reading_order(blocks, cell_blocks, tables)
    page = Page(source, prep.color, ordered, rules, figures, [t.bbox for t in tables], list(prep.notes))
    page.raw = prep.raw
    page.text_mask = _text_mask(prep, kept_words)
    for x0, y0, x1, y1 in hand_boxes:  # handwriting is never erased, even under a printed word's box
        page.text_mask[y0:y1, x0:x1] = 0
    page.decorative = _is_decorative(prep, page.text_mask, figure_boxes)
    return page


def _is_decorative(prep: Prepared, text_mask: np.ndarray, figures: list[BBox]) -> bool:
    """Paper with printed colour or patterns (frames, guilloche, tinted forms), measured
    outside the text and the pictures."""
    hsv = cv2.cvtColor(prep.color, cv2.COLOR_BGR2HSV)
    tinted = (hsv[:, :, 1] > 40) & (hsv[:, :, 2] < 252)
    keep = text_mask == 0
    for x0, y0, x1, y1 in figures:
        keep[y0:y1, x0:x1] = False
    return float(tinted[keep].mean()) > 0.06 if keep.any() else False


def _text_mask(prep: Prepared, words) -> np.ndarray:
    """Ink of every recognised word, slightly grown, so it can be erased from the photo."""
    mask = np.zeros(prep.binary.shape, np.uint8)
    value = cv2.cvtColor(prep.color, cv2.COLOR_BGR2HSV)[:, :, 2]
    for wd, colored in words:
        x0, y0, x1, y1 = wd.bbox
        x0, y0 = max(0, x0 - 2), max(0, y0 - 2)
        ink = prep.binary[y0:y1 + 2, x0:x1 + 2] > 0
        if not colored:  # black text over a seal: leave the seal's red ink alone
            ink &= value[y0:y1 + 2, x0:x1 + 2] < DARK_INK + 40
        mask[y0:y1 + 2, x0:x1 + 2][ink] = 255
    return cv2.dilate(mask, np.ones((5, 5), np.uint8))


def _reading_order(blocks: list[TextBlock], cells: list[tuple[int, TextBlock]],
                   tables: list[TableGrid]) -> list[TextBlock]:
    tbls = []
    for ti, t in enumerate(tables):
        tcells = sorted((b for i, b in cells if i == ti), key=lambda c: (c.container[1] // 10, c.container[0]))
        if tcells:
            tbls.append((t.bbox[1], tcells))
    tbls.sort(key=lambda x: x[0])
    out: list[TextBlock] = []
    for b in blocks:
        while tbls and tbls[0][0] <= b.bbox[1]:
            out += tbls.pop(0)[1]
        out.append(b)
    for _, tcells in tbls:
        out += tcells
    return out
