"""Places a document photo on a table at various angles/perspectives and checks that the app
straightens and crops it (text lines level, key phrases readable)."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from img2pdf.ocr import ocr_paragraphs  # noqa: E402
from img2pdf.preprocess import estimate_skew, load_image, preprocess  # noqa: E402

OUT = Path(__file__).parent / "out" / "skewed"


def place(doc: np.ndarray, angle: float, persp: float, bg=(70, 95, 120)) -> np.ndarray:
    """Doc rotated by `angle` degrees and tilted by `persp` (0..0.3) on a textured table."""
    h, w = doc.shape[:2]
    W, H = int(w * 1.8), int(h * 1.6)
    rng = np.random.default_rng(1)
    table = np.full((H, W, 3), bg, np.uint8)
    table = np.clip(table + rng.normal(0, 8, table.shape) +
                    np.linspace(-25, 25, W)[None, :, None], 0, 255).astype(np.uint8)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    c = np.float32([W / 2, H / 2])
    corners = src - np.float32([w / 2, h / 2])
    corners[0, 0] += persp * w * 0.5
    corners[1, 0] -= persp * w * 0.5  # top edge narrower: camera tilted
    a = np.deg2rad(angle)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], np.float32)
    dst = (corners @ rot.T + c).astype(np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(doc, M, (W, H))
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M, (W, H))
    table[mask > 0] = warped[mask > 0]
    return table


KEYS = ["CHỨNG", "Luật", "Nghiệp", "chuyên", "Vũng", "GIÁM"]


def main(src: str) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    doc = load_image(src)
    ok = True
    for angle, persp in [(0, 0), (8, 0), (-15, 0), (25, 0), (6, 0.18), (-12, 0.25), (90, 0), (180, 0.1)]:
        name = f"a{angle}_p{int(persp * 100)}"
        path = OUT / f"{name}.jpg"
        cv2.imwrite(str(path), place(doc, angle, persp), [cv2.IMWRITE_JPEG_QUALITY, 90])
        prep = preprocess(str(path))
        residual = estimate_skew(prep.binary)
        words = {w.text.strip(".,;:") for p in ocr_paragraphs(prep.gray, "vie") for ln in p.lines for w in ln.words}
        found = [k for k in KEYS if k in words]
        h, w = prep.gray.shape
        portrait = h > w
        cropped = any("cắt" in n for n in prep.notes)  # every case lies on a table
        good = abs(residual) < 0.6 and len(found) >= 5 and portrait and cropped
        ok &= good
        print(f"{'ok ' if good else 'BAD'} {name:10s} residual skew {residual:+.2f}°  keys {len(found)}/{len(KEYS)}  "
              f"size {w}x{h}  {prep.notes}")
        cv2.imwrite(str(OUT / f"{name}_out.jpg"), cv2.resize(prep.raw, None, fx=0.25, fy=0.25))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) < 2:
        sys.exit("usage: python tests/test_skewed.py <image>")
    sys.exit(main(sys.argv[1]))
