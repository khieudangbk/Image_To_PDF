"""Measures how readable the text of an exported page is: for every text line, the contrast
between the text and the background right around it (WCAG-style luminance contrast) and how
busy that background is, on the original photo and on the PDF. Writes a crop to look at."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from img2pdf.pipeline import process_image  # noqa: E402
from img2pdf.render import export_pdf, page_geometry  # noqa: E402
from img2pdf.settings import Settings  # noqa: E402

OUT = Path(__file__).parent / "out"
MIN_CONTRAST = 7.0  # WCAG AAA for normal text


def luminance(bgr: np.ndarray) -> np.ndarray:
    c = bgr[..., ::-1].astype(np.float64) / 255
    c = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return 0.2126 * c[..., 0] + 0.7152 * c[..., 1] + 0.0722 * c[..., 2]


def line_stats(img: np.ndarray, boxes) -> tuple[list[float], list[float]]:
    """Per line: text/background contrast, and background clutter (spread of its lightness in %)."""
    lum = luminance(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
    contrast, clutter = [], []
    for x0, y0, x1, y1 in boxes:
        region = lum[max(0, y0):y1, max(0, x0):x1]
        g = gray[max(0, y0):y1, max(0, x0):x1]
        if region.size == 0:
            continue
        text, bg = np.percentile(region, 5), np.percentile(region, 60)
        contrast.append((bg + 0.05) / (text + 0.05))
        clutter.append(float(g[g > np.percentile(g, 45)].std()) / 2.55)  # strokes left out
    return contrast, clutter


def main(path: str) -> int:
    OUT.mkdir(exist_ok=True)
    s = Settings()
    page = process_image(path, s)
    g = page_geometry(page, s)
    zoom = 2.0
    boxes = [ln.bbox for b in page.blocks if b.color is None for ln in b.lines]

    def to_pdf_px(box):
        x0, y0, x1, y1 = box
        return (int(g.x(x0) * zoom), int((g.ph - g.y(y0)) * zoom), int(g.x(x1) * zoom), int((g.ph - g.y(y1)) * zoom))

    c0, k0 = line_stats(page.raw, boxes)
    print(f"original photo: contrast {np.median(c0):.1f}:1, background clutter {np.median(k0):.1f}%")
    out = OUT / "readability.pdf"
    export_pdf([page], str(out), s)
    with pymupdf.open(out) as doc:
        pix = doc[0].get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
        img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3][:, :, ::-1]
    c1, k1 = line_stats(np.ascontiguousarray(img), [to_pdf_px(b) for b in boxes])
    ok = np.median(c1) >= MIN_CONTRAST and min(c1) >= 4.5
    print(f"pdf: contrast {np.median(c1):.1f}:1 (worst {min(c1):.1f}), background clutter "
          f"{np.median(k1):.1f}% (worst {max(k1):.1f})")
    cv2.imwrite(str(OUT / "readability.jpg"), img[int(img.shape[0] * 0.44):int(img.shape[0] * 0.6),
                                                  int(img.shape[1] * 0.25):])
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) < 2:
        sys.exit("usage: python tests/test_readability.py <image>")
    sys.exit(main(sys.argv[1]))
