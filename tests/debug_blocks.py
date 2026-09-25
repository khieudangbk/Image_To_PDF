"""Dumps the blocks (with word boxes and confidence) that contain given snippets."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from img2pdf.layout import _ocr_cells, build_page  # noqa: E402
from img2pdf.preprocess import preprocess  # noqa: E402
from img2pdf.settings import Settings  # noqa: E402


def main(path: str, *snippets: str):
    prep = preprocess(path)
    page = build_page("x", prep, Settings())
    for i, b in enumerate(page.blocks):
        if any(s in b.text for s in snippets):
            print(f"block {i} kind={b.kind} bbox={b.bbox}")
            for ln in b.lines:
                print("   ", round(ln.baseline), [(w.text, int(w.conf), w.bbox) for w in ln.words])
    for f in page.figures:
        x0, y0, x1, y1 = f.bbox
        if y1 - y0 < 120:  # text-sized figure: what does OCR read there?
            res = _ocr_cells(prep, prep.binary * 0, [f.bbox], "vie")
            print("figure", f.bbox, [[(w.text, int(w.conf)) for w in ws] for _, ws in res.get(0, [])])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(*sys.argv[1:])
