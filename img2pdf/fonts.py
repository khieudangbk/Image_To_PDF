from functools import lru_cache

import numpy as np
from reportlab.lib.fonts import addMapping
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from .model import Line, TextBlock
from .paths import find_font

FAMILIES = {
    "Times New Roman": (["times.ttf", "LiberationSerif-Regular.ttf", "DejaVuSerif.ttf"],
                        ["timesbd.ttf", "LiberationSerif-Bold.ttf", "DejaVuSerif-Bold.ttf"]),
    "Arial": (["arial.ttf", "LiberationSans-Regular.ttf", "DejaVuSans.ttf"],
              ["arialbd.ttf", "LiberationSans-Bold.ttf", "DejaVuSans-Bold.ttf"]),
    "Tahoma": (["tahoma.ttf", "DejaVuSans.ttf"], ["tahomabd.ttf", "DejaVuSans-Bold.ttf"]),
    "Cambria": (["cambria.ttc", "LiberationSerif-Regular.ttf"], ["cambriab.ttf", "LiberationSerif-Bold.ttf"]),
    "Courier New": (["cour.ttf", "LiberationMono-Regular.ttf"], ["courbd.ttf", "LiberationMono-Bold.ttf"]),
}

# Tesseract's x_size (row height) relative to the font's em size.
XSIZE_PER_EM = 1.08


class FontSet:
    def __init__(self, family: str):
        reg_files, bold_files = FAMILIES.get(family, FAMILIES["Times New Roman"])
        reg, bold = find_font(reg_files), find_font(bold_files)
        if reg is None:
            raise RuntimeError(f"Không tìm thấy phông chữ '{family}' trên máy")
        bold = bold or reg
        key = "".join(ch for ch in family if ch.isalnum())
        self.regular, self.bold = f"{key}-Regular", f"{key}-Bold"
        if self.regular not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(self.regular, str(reg)))
            pdfmetrics.registerFont(TTFont(self.bold, str(bold)))
            addMapping(self.regular, 0, 0, self.regular)
            addMapping(self.regular, 1, 0, self.bold)
            addMapping(self.regular, 0, 1, self.regular)
            addMapping(self.regular, 1, 1, self.bold)

    def name(self, bold: bool) -> str:
        return self.bold if bold else self.regular

    def width(self, text: str, bold: bool = False, size: float = 1.0) -> float:
        return pdfmetrics.stringWidth(text, self.name(bold), size)


@lru_cache(maxsize=None)
def get_fontset(family: str) -> FontSet:
    return FontSet(family)


def _line_estimate(ln: Line, fs: FontSet) -> tuple[float, float, int]:
    """Font size (em in px) from width fit, from height, and number of glyphs."""
    h_est = ln.x_size / XSIZE_PER_EM
    ink_px = sum(w.bbox[2] - w.bbox[0] for w in ln.words)
    em = sum(fs.width(w.text, w.bold) for w in ln.words)
    n = sum(len(w.text) for w in ln.words)
    w_est = ink_px / em if em > 0 else 0.0
    return w_est, h_est, n


def fit_font_px(block: TextBlock, fs: FontSet) -> float:
    if block.font_px_override:
        return block.font_px_override
    ests, weights, heights = [], [], []
    for ln in block.lines:
        w_est, h_est, n = _line_estimate(ln, fs)
        heights.append(h_est)
        if n >= 4 and w_est > 0:
            ests.append(float(np.clip(w_est, 0.75 * h_est, 1.3 * h_est)))
            weights.append(n)
    if not ests:
        return float(np.median(heights)) if heights else 12.0
    order = np.argsort(ests)
    cum = np.cumsum(np.array(weights)[order])
    return float(np.array(ests)[order][np.searchsorted(cum, cum[-1] / 2)])
