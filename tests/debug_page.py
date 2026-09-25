"""Writes an overlay of detected regions and a PDF preview for one image."""
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from img2pdf.pipeline import process_image  # noqa: E402
from img2pdf.render import render_preview  # noqa: E402
from img2pdf.settings import Settings  # noqa: E402

COLORS = {"paragraph": (255, 120, 0), "heading": (0, 140, 255), "cell": (0, 170, 0)}


def main(path: str):
    s = Settings()
    page = process_image(path, s)
    img = page.image.copy()
    for t in page.tables:
        cv2.rectangle(img, t[:2], t[2:], (200, 0, 200), 6)
    for f in page.figures:
        cv2.rectangle(img, f.bbox[:2], f.bbox[2:], (0, 0, 255), 6)
    for b in page.blocks:
        x0, y0, x1, y1 = b.bbox
        cv2.rectangle(img, (x0, y0), (x1, y1), COLORS[b.kind], 3)
        cv2.putText(img, f"{b.align[0]}{'B' if b.bold else ''}", (x0, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    COLORS[b.kind], 2)
    out = Path(__file__).parent / "out"
    out.mkdir(exist_ok=True)
    stem = Path(path).stem
    cv2.imwrite(str(out / f"{stem}_overlay.jpg"), cv2.resize(img, None, fx=0.5, fy=0.5))
    (out / f"{stem}_preview.png").write_bytes(render_preview(page, s, zoom=1.2))
    for b in page.blocks:
        print(b.kind, b.align, "B" if b.bold else "-", b.bbox, repr(b.text[:70]))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1])
