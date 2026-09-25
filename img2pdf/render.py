import io
from xml.sax.saxutils import escape

import cv2
import numpy as np
from PIL import Image
from reportlab.lib.colors import Color
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.platypus import Paragraph

from . import APP_NAME, __version__
from .fonts import FontSet, fit_font_px, get_fontset
from .model import Page, TextBlock
from .restore import restore_colors
from .settings import PAPER_FIT_IMAGE, Settings

A4 = (595.2756, 841.8898)
LETTER = (612.0, 792.0)
_ALIGN = {"left": TA_LEFT, "center": TA_CENTER, "right": TA_RIGHT, "justify": TA_JUSTIFY}


class Geometry:
    """Maps page-image pixels to PDF points. `fill` stretches the page image over the whole
    sheet (small aspect differences only) so a kept background has clean, square edges."""

    def __init__(self, page: Page, paper: str, fill: bool = False):
        w, h = page.width, page.height
        if paper == PAPER_FIT_IMAGE:
            self.pw = A4[0] if w <= h else A4[1]
            self.sx = self.sy = self.pw / w
            self.ph = h * self.sy
            self.ox = self.oy = 0.0
        else:
            pw, ph = LETTER if paper == "Letter" else A4
            if w > h:
                pw, ph = ph, pw
            self.pw, self.ph = pw, ph
            t = page.trim if fill else 0
            if fill and abs((pw / ph) / ((w - 2 * t) / (h - 2 * t)) - 1) <= 0.12:
                # the sheet inside its trimmed edge covers the page; the edge falls outside it
                self.sx, self.sy = pw / (w - 2 * t), ph / (h - 2 * t)
                self.ox, self.oy = -t * self.sx, -t * self.sy
            else:
                self.sx = self.sy = min(pw / w, ph / h)
                self.ox, self.oy = (pw - w * self.sx) / 2, (ph - h * self.sy) / 2
        self.s = (self.sx + self.sy) / 2  # for font sizes and line widths

    def x(self, px: float) -> float:
        return self.ox + px * self.sx

    def y(self, py: float) -> float:
        return self.ph - (self.oy + py * self.sy)


def uses_background(page: Page, settings: Settings) -> bool:
    return page.raw is not None and (
        settings.background == "on" or (settings.background == "auto" and page.decorative))


def page_geometry(page: Page, settings: Settings) -> Geometry:
    return Geometry(page, settings.paper, fill=uses_background(page, settings))


def _draw_exact(c, block: TextBlock, fs: FontSet, g: Geometry, size_px: float, on_background: bool = False):
    size_pt = size_px * g.s
    text = c.beginText()
    for ln in block.lines:
        words = ln.words
        yb = g.y(ln.baseline)
        for i, w in enumerate(words):
            # squeeze only a word that would run into the next one (e.g. a misread "HỘI" -> "HỌ I");
            # the rest of the line keeps its natural width
            need = fs.width(w.text, w.bold) * size_px
            if i + 1 < len(words):
                avail = words[i + 1].bbox[0] - w.bbox[0] - 0.15 * size_px
            else:
                avail = (w.bbox[2] - w.bbox[0]) * 1.1 + 0.1 * size_px
            hscale = max(70.0, min(100.0, 100.0 * avail / need)) if avail > 0 and need > 0 else 100.0
            text.setTextOrigin(g.x(w.bbox[0]), yb)
            text.setFont(fs.name(w.bold), size_pt)
            text.setHorizScale(hscale)
            # 3 = invisible: the original pixels stay visible, the text is still searchable.
            # 2 = fill + a hairline outline in the same colour: small serif type on screen is
            # much thinner than printed ink, this gives it the weight of the original print
            text.setTextRenderMode(3 if on_background and _show_original(block, w) else 2)
            text.textOut(w.text + (" " if i + 1 < len(words) else ""))
    # The PDF text render mode outlives the text object; isolate it so an invisible word
    # at the end of one block does not make the next block invisible too.
    c.saveState()
    c.setStrokeColorRGB(*(block.color or (0, 0, 0)))
    c.setLineWidth(INK_SPREAD * size_pt)
    c.drawText(text)
    c.restoreState()


def _line_pitch(block: TextBlock, size_px: float) -> float:
    bases = sorted(ln.baseline for ln in block.lines)
    if len(bases) >= 2:
        diffs = np.diff(bases)
        diffs = diffs[diffs > 0.5 * size_px]
        if len(diffs):
            return float(np.median(diffs))
    return size_px * 1.2


def _draw_flow(c, block: TextBlock, fs: FontSet, g: Geometry, size_px: float):
    box = block.container if block.container else block.bbox
    if block.container:
        box = (box[0] + 4, box[1] + 4, box[2] - 4, box[3] - 4)
    x0, y0, x1, _ = box
    top = block.bbox[1] if block.lines else y0
    width = max((x1 - x0) * g.sx, 10)
    size_pt = size_px * g.s
    if not block.container and len(block.lines) <= 1 and "\n" not in block.text:
        # A one-line block whose edited text got longer grows sideways instead of wrapping.
        need = fs.width(block.text, block.bold, size_pt) * 1.02
        page_w = g.pw - 2 * 36
        if need > width:
            grow = min(need, page_w) - width
            shift = {"center": grow / 2, "right": grow}.get(block.align, 0.0)
            left_pt = min(max(g.x(x0) - shift, 36), g.pw - 36 - min(need, page_w))
            x0 = (left_pt - g.ox) / g.sx
            width = min(need, page_w)
    style = ParagraphStyle("b", fontName=fs.regular, fontSize=size_pt,
                           textColor=Color(*(block.color or (0, 0, 0))),
                           leading=_line_pitch(block, size_px) * g.s, alignment=_ALIGN.get(block.align, TA_LEFT))
    markup = "<br/>".join(escape(line) for line in block.text.split("\n"))
    if block.bold:
        markup = f"<b>{markup}</b>"
    p = Paragraph(markup, style)
    _, ph = p.wrap(width, 10_000)
    ascent = size_pt * 0.9
    p.drawOn(c, g.x(x0), g.y(top) - ph + (style.leading - ascent) * 0.5)


INK_SPREAD = 0.028  # outline width as a fraction of the font size (≈0.3 pt at 11 pt)
LOW_CONF = 30  # with the original background, words OCR can barely read keep their original pixels


def _show_original(block: TextBlock, w) -> bool:
    # letters were checked against the dictionary; digits and symbols were not
    return block.override_text is None and w.conf < LOW_CONF and not w.text.strip(".,;:!?").isalpha()


def background(page: Page) -> np.ndarray:
    """The page with its printed text painted out (computed once per page), in print-like
    restored colours; the real text is drawn straight on top of it."""
    if page.background is None:
        raw, mask = restore_colors(page.raw), page.text_mask.copy()
        for b in page.blocks:
            for ln in b.lines:
                for w in ln.words:
                    if _show_original(b, w):
                        x0, y0, x1, y1 = w.bbox
                        mask[y0:y1, x0:x1] = 0
        k = min(1.0, 2600 / raw.shape[1])  # ~300 dpi for A4; inpainting cost grows fast with size
        if k < 1:
            small = cv2.resize(raw, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
            small_mask = cv2.resize(mask, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            small, small_mask = raw, mask
        page.background = cv2.inpaint(small, small_mask, 4, cv2.INPAINT_TELEA)
    return page.background


def _jpeg(bgr: np.ndarray, width_pt: float, dpi: int = 220, quality: int = 88) -> io.BytesIO:
    max_w = max(1, int(width_pt / 72 * dpi))
    if bgr.shape[1] > max_w:
        bgr = cv2.resize(bgr, (max_w, max(1, int(bgr.shape[0] * max_w / bgr.shape[1]))),
                         interpolation=cv2.INTER_AREA)
    buf = io.BytesIO()
    Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).save(buf, "JPEG", quality=quality, optimize=True)
    buf.seek(0)
    return buf


def draw_page(c, page: Page, settings: Settings, page_no: int):
    g = page_geometry(page, settings)
    fs = get_fontset(settings.font)
    c.setPageSize((g.pw, g.ph))
    c.setFont(fs.regular, 12)  # otherwise ReportLab references its default Helvetica

    on_background = uses_background(page, settings)
    if on_background:
        # The original page with its printed text erased already holds the colours, frames,
        # seals, photos and lines; only the real text is drawn on top.
        c.drawImage(ImageReader(_jpeg(background(page), page.width * g.sx, dpi=300, quality=92)),
                    g.x(0), g.y(page.height), width=page.width * g.sx, height=page.height * g.sy)
    else:
        for fig in page.figures:
            x0, y0, x1, y1 = fig.bbox
            c.drawImage(ImageReader(_jpeg(fig.image, (x1 - x0) * g.sx)), g.x(x0), g.y(y1),
                        width=(x1 - x0) * g.sx, height=(y1 - y0) * g.sy)
        c.setStrokeColorRGB(0, 0, 0)
        for r in page.rules:
            c.setLineWidth(max(0.4, r.thickness * g.s))
            c.line(g.x(r.x0), g.y(r.y0), g.x(r.x1), g.y(r.y1))

    for i, block in enumerate(page.blocks):
        if not block.text.strip():
            continue
        c.setFillColorRGB(*(block.color or (0, 0, 0)))
        size_px = fit_font_px(block, fs)
        if block.override_text is not None:
            _draw_flow(c, block, fs, g, size_px)
        else:
            _draw_exact(c, block, fs, g, size_px, on_background)
        if block.kind == "heading":
            key = f"p{page_no}b{i}"
            c.bookmarkPage(key, fit="XYZ", top=g.y(block.bbox[1]) + 4, left=0)
            title = " ".join(block.text.split())[:120]
            c.addOutlineEntry(title, key, level=min(block.heading_level, 1), closed=False)
    c.showPage()


def _new_canvas(target, title: str, settings: Settings):
    c = rl_canvas.Canvas(target, pageCompression=1, initialFontName=get_fontset(settings.font).regular,
                         initialFontSize=12)
    c.setTitle(title)
    c.setCreator(f"{APP_NAME} {__version__}")
    c.setProducer(f"{APP_NAME} {__version__}")
    return c


def export_pdf(pages: list[Page], out_path: str, settings: Settings, title: str = ""):
    c = _new_canvas(out_path, title or "Tài liệu", settings)
    # Outline levels must not jump (0 -> 1 without a 0 first), so normalise per document.
    seen_top = False
    for p in pages:
        for b in p.blocks:
            if b.kind == "heading":
                if b.heading_level == 0:
                    seen_top = True
                elif not seen_top:
                    b.heading_level = 0
    for i, p in enumerate(pages):
        draw_page(c, p, settings, i)
    c.save()


def render_preview(page: Page, settings: Settings, zoom: float = 1.5) -> bytes:
    import pymupdf as fitz

    buf = io.BytesIO()
    c = _new_canvas(buf, "preview", settings)
    levels = [b.heading_level for b in page.blocks]
    for b in page.blocks:
        b.heading_level = 0
    try:
        draw_page(c, page, settings, 0)
        c.save()
    finally:
        for b, lv in zip(page.blocks, levels):
            b.heading_level = lv
    with fitz.open(stream=buf.getvalue(), filetype="pdf") as doc:
        pix = doc[0].get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        return pix.tobytes("png")

