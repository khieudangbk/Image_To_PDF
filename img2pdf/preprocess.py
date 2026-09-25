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
    raw: np.ndarray     # same geometry as `color` but original colours (for photos)
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


def _intersect(l1, l2) -> np.ndarray:
    (p, d), (q, e) = l1, l2
    t = np.linalg.solve(np.array([d, -e]).T, q - p)[0]
    return p + t * d


def find_document_by_edges(bgr: np.ndarray) -> np.ndarray | None:
    """Paper that nearly fills the photo or touches its border: find the outermost straight
    edge on each side with a clear colour change across it; missing sides use the image border."""
    h, w = bgr.shape[:2]
    k = 1000 / max(h, w)
    small = cv2.resize(bgr, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    lab = cv2.cvtColor(cv2.GaussianBlur(small, (9, 9), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    lines = cv2.HoughLinesP(cv2.Canny(gray, 30, 90), 1, np.pi / 360, threshold=60,
                            minLineLength=int(0.3 * min(sh, sw)), maxLineGap=15)
    if lines is None:
        return None

    def strip(p0, p1, normal, offset):
        ts = np.linspace(0.05, 0.95, 40)[:, None]
        pts = p0 + (p1 - p0) * ts + normal * offset
        xs = np.clip(pts[:, 0].astype(int), 0, sw - 1)
        ys = np.clip(pts[:, 1].astype(int), 0, sh - 1)
        return lab[ys, xs].mean(0)

    # Paper colour: the bright pixels of the central region.
    centre = lab[sh // 4:3 * sh // 4, sw // 4:3 * sw // 4].reshape(-1, 3)
    paper = np.median(centre[centre[:, 0] >= np.percentile(centre[:, 0], 60)], axis=0)

    def outside_is_background(side: str, p0, p1) -> bool:
        """Everything between the line and the image border must look unlike paper."""
        a, b = (p0, p1) if (p0[1] < p1[1] if side in ("left", "right") else p0[0] < p1[0]) else (p1, p0)
        if side in ("left", "right"):
            d = (b - a) / max(b[1] - a[1], 1e-3)
            top, bot = a - d * a[1], a + d * (sh - 1 - a[1])
            edge = 0 if side == "left" else sw - 1
            poly = [top, bot, (edge, sh - 1), (edge, 0)]
        else:
            d = (b - a) / max(b[0] - a[0], 1e-3)
            lft, rgt = a - d * a[0], a + d * (sw - 1 - a[0])
            edge = 0 if side == "top" else sh - 1
            poly = [lft, rgt, (sw - 1, edge), (0, edge)]
        mask = np.zeros((sh, sw), np.uint8)
        cv2.fillPoly(mask, [np.array(poly, np.int32)], 255)
        px = lab[mask > 0]
        if len(px) < 30:
            return False
        like_paper = np.linalg.norm(px - paper, axis=1) < 12
        return like_paper.mean() < 0.3

    best: dict[str, tuple[float, tuple]] = {}
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4).astype(np.float32):
        p0, p1 = np.array([x1, y1], np.float32), np.array([x2, y2], np.float32)
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        if dx < 0.08 * dy:
            xm = (x1 + x2) / 2
            side, dist, normal = ("left", xm, (-1, 0)) if xm < sw / 2 else ("right", sw - xm, (1, 0))
            limit = 0.15 * sw
        elif dy < 0.08 * dx:
            ym = (y1 + y2) / 2
            side, dist, normal = ("top", ym, (0, -1)) if ym < sh / 2 else ("bottom", sh - ym, (0, 1))
            limit = 0.15 * sh
        else:
            continue
        if dist > limit or dist < 3:
            continue
        normal = np.array(normal, np.float32)
        contrast = np.linalg.norm(strip(p0, p1, normal, 6) - strip(p0, p1, normal, -6))
        if contrast < 14 or not outside_is_background(side, p0, p1):
            continue
        if side not in best or dist < best[side][0]:
            best[side] = (dist, (p0, (p1 - p0) / np.linalg.norm(p1 - p0)))
    if not best:
        return None
    border = {"left": (np.array([0, 0], np.float32), np.array([0, 1], np.float32)),
              "right": (np.array([sw - 1, 0], np.float32), np.array([0, 1], np.float32)),
              "top": (np.array([0, 0], np.float32), np.array([1, 0], np.float32)),
              "bottom": (np.array([0, sh - 1], np.float32), np.array([1, 0], np.float32))}
    L = {s: best[s][1] if s in best else border[s] for s in border}
    quad = np.array([_intersect(L["top"], L["left"]), _intersect(L["top"], L["right"]),
                     _intersect(L["bottom"], L["right"]), _intersect(L["bottom"], L["left"])], np.float32)
    quad[:, 0] = np.clip(quad[:, 0], 0, sw - 1)
    quad[:, 1] = np.clip(quad[:, 1], 0, sh - 1)
    ratio = cv2.contourArea(quad) / (sw * sh)
    if not 0.4 < ratio < 0.985:
        return None
    return quad / k


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


def _trim_margins(bgr: np.ndarray, raw: np.ndarray):
    """Remove dark scanner/photo borders touching the image edge (in both images)."""
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
                raw[comp] = 255


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
        if quad is None:
            quad = find_document_by_edges(bgr)
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

    raw = bgr.copy()
    color = normalize_background(bgr)
    if auto_crop:
        _trim_margins(color, raw)
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    binary = binarize(gray)

    angle = estimate_skew(binary)
    if abs(angle) >= 0.1:
        color = rotate(color, angle)
        raw = rotate(raw, angle)
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
            raw = cv2.resize(raw, (color.shape[1], color.shape[0]), interpolation=interp)
            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            binary = binarize(gray)
    return Prepared(color, gray, binary, raw, notes)
