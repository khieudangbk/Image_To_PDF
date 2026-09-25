"""Checks automatic orientation fixing: a rotated page must come back upright."""
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from img2pdf.ocr import ocr_paragraphs  # noqa: E402
from img2pdf.preprocess import preprocess  # noqa: E402

SAMPLES = Path(__file__).parent / "samples"
OUT = Path(__file__).parent / "out"


def main():
    OUT.mkdir(exist_ok=True)
    img = cv2.imread(str(SAMPLES / "01_clean.png"))
    ok = True
    for name, code in [("90", cv2.ROTATE_90_CLOCKWISE), ("180", cv2.ROTATE_180),
                       ("270", cv2.ROTATE_90_COUNTERCLOCKWISE)]:
        path = OUT / f"rot_{name}.png"
        cv2.imwrite(str(path), cv2.rotate(img, code))
        prep = preprocess(str(path))
        words = [w.text for p in ocr_paragraphs(prep.gray, "vie") for ln in p.lines for w in ln.words]
        upright = "Quyết" in words and prep.gray.shape[0] > prep.gray.shape[1]
        ok &= upright
        print(f"rotated {name}: notes={prep.notes} upright={upright}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
