from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageOps

from .ocr import detect_orientation

TARGET_TEXT_HEIGHT = 30  # px; Tesseract is most accurate around this cap height
MAX_SIDE = 7000


@dataclass
class Prepared:
    color: np.ndarray   # background-normalized BGR
    gray: np.ndarray    # background-normalized grayscale
    binary: np.ndarray  # ink = 255
    notes: list[str] = field(default_factory=list)


def load_image(path: str) -> np.ndarray:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            im = Image.alpha_composite(bg, im)
        im = im.convert("RGB")
        arr = np.asarray(im)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _order_quad(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2).astype(np.float32)
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.array([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]], np.float32)


def find_document_quad(bgr: np.ndarray) -> np.ndarray | None:
    h, w = bgr.shape[:2]
    k = 900 / max(h, w)
    small = cv2.resize(bgr, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.dilate(cv2.Canny(gray, 40, 120), np.ones((3, 3), np.uint8), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_img = small.shape[0] * small.shape[1]
    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        ratio = cv2.contourArea(approx) / area_img
        if not 0.35 < ratio < 0.97:
            continue
        mask = np.zeros(gray.shape, np.uint8)
        cv2.fillConvexPoly(mask, approx.reshape(4, 2), 255)
        inside, outside = gray[mask > 0].mean(), gray[mask == 0].mean()
        if inside - outside < 25:  # paper must stand out from its surroundings
            continue
        return _order_quad(approx) / k
    return None


def warp_quad(bgr: np.ndarray, quad: np.ndarray) -> np.ndarray:
    tl, tr, br, bl = quad
    w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    return cv2.warpPerspective(bgr, cv2.getPerspectiveTransform(quad, dst), (w, h),
                               flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255))


def normalize_background(bgr: np.ndarray) -> np.ndarray:
    """Divide out uneven lighting/shadows so paper becomes pure white."""
    h, w = bgr.shape[:2]
    k = max(15, (min(h, w) // 60) | 1)
    out = np.empty_like(bgr)
    for i in range(3):
        ch = bgr[:, :, i]
        bg = cv2.medianBlur(cv2.dilate(ch, np.ones((7, 7), np.uint8)), k)
        out[:, :, i] = np.clip(ch.astype(np.float32) / np.maximum(bg, 1) * 255, 0, 255).astype(np.uint8)
    return out


def binarize(gray: np.ndarray) -> np.ndarray:
    _, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return b


def estimate_skew(binary: np.ndarray) -> float:
    h, w = binary.shape
    k = 1000 / max(h, w)
    small = cv2.resize(binary, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
    center = (small.shape[1] / 2, small.shape[0] / 2)

    def score(angle):
        m = cv2.getRotationMatrix2D(center, angle, 1.0)
        rot = cv2.warpAffine(small, m, (small.shape[1], small.shape[0]), flags=cv2.INTER_NEAREST)
        prof = rot.sum(1, dtype=np.float64)
        return float(np.sum(np.diff(prof) ** 2))

    best = max(np.arange(-10, 10.01, 0.5), key=score)
    best = max(np.arange(best - 0.5, best + 0.51, 0.05), key=score)
    return float(best)


def rotate(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    border = (255, 255, 255) if img.ndim == 3 else 255
    return cv2.warpAffine(img, m, (nw, nh), flags=cv2.INTER_CUBIC, borderValue=border)


def estimate_text_height(binary: np.ndarray) -> float:
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    hs = stats[1:, cv2.CC_STAT_HEIGHT]
    ws = stats[1:, cv2.CC_STAT_WIDTH]
    keep = (hs > 4) & (hs < binary.shape[0] / 10) & (ws < binary.shape[1] / 5) & (ws > 1)
    return float(np.median(hs[keep])) if keep.sum() > 20 else 0.0


def _trim_margins(bgr: np.ndarray) -> np.ndarray:
    """Remove dark scanner/photo borders touching the image edge."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    dark = (gray < 90).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    h, w = gray.shape
    band = np.zeros_like(dark, dtype=bool)
    bh, bw = max(3, h // 25), max(3, w // 25)
    band[:bh, :] = band[-bh:, :] = band[:, :bw] = band[:, -bw:] = True
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        touches = x == 0 or y == 0 or x + cw >= w or y + ch >= h
        if touches and area > 0.002 * h * w and (cw > 0.3 * w or ch > 0.3 * h):
            comp = lab == i
            if band[comp].mean() > 0.6:
                bgr[comp] = 255
    return bgr


_ROTATE_CW = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def preprocess(path: str, auto_crop: bool = True, rotation: int = 0) -> Prepared:
    """`rotation` (clockwise degrees) is a manual override; 0 means detect automatically."""
    notes = []
    bgr = load_image(path)
    if max(bgr.shape[:2]) > MAX_SIDE:
        k = MAX_SIDE / max(bgr.shape[:2])
        bgr = cv2.resize(bgr, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)

    if auto_crop:
        quad = find_document_quad(bgr)
        if quad is not None:
            bgr = warp_quad(bgr, quad)
            notes.append("Đã cắt và nắn phối cảnh trang giấy")

    if rotation % 360:
        bgr = cv2.rotate(bgr, _ROTATE_CW[rotation % 360])
    else:
        small_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        k = min(1.0, 2000 / max(small_gray.shape))
        if k < 1:
            small_gray = cv2.resize(small_gray, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        rot, conf = detect_orientation(small_gray)
        if rot in _ROTATE_CW and conf >= 2.0:
            bgr = cv2.rotate(bgr, _ROTATE_CW[rot])
            notes.append(f"Đã tự xoay trang {rot}°")

    color = normalize_background(bgr)
    if auto_crop:
        color = _trim_margins(color)
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    binary = binarize(gray)

    angle = estimate_skew(binary)
    if abs(angle) >= 0.1:
        color = rotate(color, angle)
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        binary = binarize(gray)
        notes.append(f"Đã chỉnh nghiêng {angle:+.2f}°")

    th = estimate_text_height(binary)
    if th > 0:
        scale = TARGET_TEXT_HEIGHT / th
        scale = min(scale, 3.0, MAX_SIDE / max(gray.shape))
        if scale > 1.25 or scale < 0.6:
            interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
            color = cv2.resize(color, None, fx=scale, fy=scale, interpolation=interp)
            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            binary = binarize(gray)
    return Prepared(color, gray, binary, notes)
