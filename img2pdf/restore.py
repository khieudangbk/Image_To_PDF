"""Turns a photographed page into a clean, print-like colour image: even lighting, white
paper, faithful print colours, crisp fine detail."""
import cv2
import numpy as np


def flat_field(bgr: np.ndarray, paper_level: float = 245) -> np.ndarray:
    """Divides out uneven lighting at a large scale only, so printed colour areas (frames,
    photos) keep their colour while the paper becomes an even `paper_level`."""
    h, w = bgr.shape[:2]
    k = 400 / max(h, w)
    small = cv2.resize(bgr, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    paper = cv2.GaussianBlur(cv2.dilate(small, np.ones((15, 15), np.uint8)), (0, 0), 12)
    paper = cv2.resize(paper, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    lum = paper.mean(axis=2, keepdims=True)  # brightness only: keep the hue of the print
    return np.clip(bgr.astype(np.float32) / np.maximum(lum, 1) * paper_level, 0, 255).astype(np.uint8)


def _white_balance(bgr: np.ndarray) -> np.ndarray:
    f = bgr.astype(np.float32)
    lum = f.mean(axis=2)
    ref = f[lum > np.percentile(lum, 80)].mean(axis=0)  # the paper
    return np.clip(f * (ref.mean() / np.maximum(ref, 1)), 0, 255).astype(np.uint8)


def _tone(bgr: np.ndarray) -> np.ndarray:
    """Paper to white, mid-tones (the print) a little deeper so they keep their depth."""
    x = np.arange(256, dtype=np.float32)
    lut = (np.clip((x - 12) / (244 - 12), 0, 1) ** 1.22 * 255).astype(np.uint8)
    return cv2.LUT(bgr, lut)


def _sharpen(bgr: np.ndarray) -> np.ndarray:
    # two scales: fine lines of guilloche patterns and the broader ornament shapes
    x = cv2.addWeighted(bgr, 1.7, cv2.GaussianBlur(bgr, (0, 0), 0.8), -0.7, 0)
    return cv2.addWeighted(x, 1.35, cv2.GaussianBlur(x, (0, 0), 2.5), -0.35, 0)


def _clean_border(bgr: np.ndarray) -> np.ndarray:
    """Greyish shadow left right at the paper edge becomes paper white."""
    h, w = bgr.shape[:2]
    band = max(4, int(0.025 * min(h, w)))
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    shadow = (hsv[:, :, 1] < 70) & (hsv[:, :, 2] > 140)
    edge = np.zeros((h, w), bool)
    edge[:band, :] = edge[-band:, :] = edge[:, :band] = edge[:, -band:] = True
    bgr[edge & shadow] = 255
    return bgr


def restore_colors(bgr: np.ndarray) -> np.ndarray:
    x = _white_balance(flat_field(bgr, paper_level=241))  # not blown out: keeps the print's depth
    x = cv2.fastNlMeansDenoisingColored(x, None, 3, 3, 5, 15)
    return _clean_border(_sharpen(_tone(x)))
