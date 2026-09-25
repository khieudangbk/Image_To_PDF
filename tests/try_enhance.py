"""Experiments with restoring a clean, digital-looking colour background from a photo."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from img2pdf.preprocess import preprocess  # noqa: E402

OUT = Path(__file__).parent / "out"


def flat_field(bgr: np.ndarray) -> np.ndarray:
    """Divide by a smooth estimate of the paper brightness (large scale only, so printed
    colour areas survive), then map paper to pure white."""
    h, w = bgr.shape[:2]
    k = 400 / max(h, w)
    small = cv2.resize(bgr, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    # brightest level in each neighbourhood = paper, not ink or pattern
    paper = cv2.dilate(small, np.ones((15, 15), np.uint8))
    paper = cv2.GaussianBlur(paper, (0, 0), 12)
    paper = cv2.resize(paper, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    gray_paper = paper.mean(axis=2, keepdims=True)  # correct brightness, keep hue of the print
    out = bgr.astype(np.float32) / np.maximum(gray_paper, 1) * 255
    return np.clip(out, 0, 255).astype(np.uint8)


def white_balance(bgr: np.ndarray) -> np.ndarray:
    """Make the paper neutral: scale channels so the bright paper pixels become grey-equal."""
    f = bgr.astype(np.float32)
    lum = f.mean(axis=2)
    paper = f[lum > np.percentile(lum, 80)]
    ref = paper.mean(axis=0)
    return np.clip(f * (ref.mean() / np.maximum(ref, 1)), 0, 255).astype(np.uint8)


def levels(bgr: np.ndarray, white: int = 225, black: int = 25) -> np.ndarray:
    f = (bgr.astype(np.float32) - black) / (white - black) * 255
    return np.clip(f, 0, 255).astype(np.uint8)


def saturate(bgr: np.ndarray, factor: float = 1.25) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def main(path: str):
    OUT.mkdir(exist_ok=True)
    prep = preprocess(path)
    raw = prep.raw
    x = flat_field(raw)
    x = white_balance(x)
    x = cv2.fastNlMeansDenoisingColored(x, None, 5, 5, 7, 21)
    x = levels(x)
    x = saturate(x)
    blur = cv2.GaussianBlur(x, (0, 0), 1.2)
    x = cv2.addWeighted(x, 1.5, blur, -0.5, 0)
    cv2.imwrite(str(OUT / "enh.jpg"), cv2.resize(x, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA))
    cv2.imwrite(str(OUT / "enh_raw.jpg"), cv2.resize(raw, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA))


if __name__ == "__main__":
    main(sys.argv[1])
