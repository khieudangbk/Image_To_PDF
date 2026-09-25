"""Turns a photographed page into a clean, print-like colour image: even lighting, pure white
paper, vivid print colours, no sensor noise."""
import cv2
import numpy as np


def _flat_field(bgr: np.ndarray) -> np.ndarray:
    # Paper brightness estimated at a large scale only, so printed colour areas survive.
    h, w = bgr.shape[:2]
    k = 400 / max(h, w)
    small = cv2.resize(bgr, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    paper = cv2.GaussianBlur(cv2.dilate(small, np.ones((15, 15), np.uint8)), (0, 0), 12)
    paper = cv2.resize(paper, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    lum = paper.mean(axis=2, keepdims=True)  # brightness only: keep the hue of the print
    return np.clip(bgr.astype(np.float32) / np.maximum(lum, 1) * 255, 0, 255).astype(np.uint8)


def _white_balance(bgr: np.ndarray) -> np.ndarray:
    f = bgr.astype(np.float32)
    lum = f.mean(axis=2)
    ref = f[lum > np.percentile(lum, 80)].mean(axis=0)  # the paper
    return np.clip(f * (ref.mean() / np.maximum(ref, 1)), 0, 255).astype(np.uint8)


def _levels(bgr: np.ndarray, black: int = 25, white: int = 225) -> np.ndarray:
    f = (bgr.astype(np.float32) - black) / (white - black) * 255
    return np.clip(f, 0, 255).astype(np.uint8)


def _saturate(bgr: np.ndarray, factor: float) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _clean_border(bgr: np.ndarray) -> np.ndarray:
    """Leftover shadow or table right at the paper edge becomes paper."""
    h, w = bgr.shape[:2]
    band = max(4, int(0.02 * min(h, w)))
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    neutral = hsv[:, :, 1] < 45
    edge = np.zeros((h, w), bool)
    edge[:band, :] = edge[-band:, :] = edge[:, :band] = edge[:, -band:] = True
    bgr[edge & neutral] = 255
    return bgr


def restore_colors(bgr: np.ndarray) -> np.ndarray:
    x = _white_balance(_flat_field(bgr))
    x = cv2.fastNlMeansDenoisingColored(x, None, 5, 5, 7, 21)
    x = _saturate(_levels(x), 1.25)
    x = cv2.addWeighted(x, 1.5, cv2.GaussianBlur(x, (0, 0), 1.2), -0.5, 0)  # sharpen
    return _clean_border(x)
