from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageOps

from .ocr import available_languages, detect_orientation, ocr_paragraphs

TARGET_TEXT_HEIGHT = 30  # px; Tesseract is most accurate around this cap height
MAX_SIDE = 7000


@dataclass
class Prepared:
    color: np.ndarray   # background-normalized BGR
    gray: np.ndarray    # background-normalized grayscale
    binary: np.ndarray  # ink = 255
    raw: np.ndarray     # same geometry as `color` but original colours (for photos)
    notes: list[str] = field(default_factory=list)
    trim: int = 0       # px to cut from each edge of a photographed sheet in the output


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
        # corners are often chipped or bumped by the table; simplify the hull until 4 remain
        hull = cv2.convexHull(c)
        approx = None
        for eps in (0.02, 0.03, 0.04, 0.05, 0.06):
            a = cv2.approxPolyDP(hull, eps * cv2.arcLength(hull, True), True)
            if len(a) == 4:
                approx = a
                break
        if approx is None or not cv2.isContourConvex(approx):
            continue
        if cv2.contourArea(approx) < 0.85 * cv2.contourArea(hull):
            continue
        ratio = cv2.contourArea(approx) / area_img
        if not 0.12 < ratio < 0.97:  # a sheet photographed from afar can be small in the frame
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


def find_frame_quad(bgr: np.ndarray) -> np.ndarray | None:
    """The main area of a framed page: the paper enclosed by a printed border (a coloured
    ornamental band or a dark border rule). Returns its corners TL, TR, BR, BL, or None when
    the page has no closed frame (plain letters), in which case nothing should be reshaped."""
    h, w = bgr.shape[:2]
    k = 1200 / max(h, w)
    small = cv2.resize(bgr, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    hsv = cv2.cvtColor(cv2.GaussianBlur(small, (5, 5), 0), cv2.COLOR_BGR2HSV)
    # the border: saturated print (ornamental band) or dark ink (border rules)
    border = ((hsv[:, :, 1] > 60) | (hsv[:, :, 2] < 110)).astype(np.uint8) * 255
    border = cv2.morphologyEx(border, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    inside = cv2.bitwise_not(border)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(inside, connectivity=4)
    best = None
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if x == 0 or y == 0 or x + bw >= sw or y + bh >= sh:
            continue  # touches the image edge: not enclosed by a frame
        if bw * bh < 0.4 * sw * sh:
            continue
        if best is None or bw * bh > stats[best][2] * stats[best][3]:
            best = i
    if best is None:
        return None
    region = (lab == best).astype(np.uint8) * 255
    # text and pictures inside the area leave holes; the outline is what matters
    contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    quad = None
    for eps in (0.01, 0.015, 0.02, 0.03, 0.04):
        a = cv2.approxPolyDP(hull, eps * cv2.arcLength(hull, True), True)
        if len(a) == 4:
            quad = _order_quad(a)
            break
    if quad is None or cv2.contourArea(quad) < 0.9 * cv2.contourArea(hull):
        return None
    for i in range(4):
        a, b, c = quad[i - 1], quad[i], quad[(i + 1) % 4]
        v1, v2 = a - b, c - b
        cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        if abs(cos) > 0.2:  # corner angle outside ~78°..102°
            return None
    return quad / k


def band_box(bgr: np.ndarray, inner: tuple[float, float, float, float]) -> tuple[int, int, int, int] | None:
    """After squaring: the outer edge of a coloured border band as (x0, y0, x1, y1), where the
    page is cut so leftovers of the photo (shadow, table) disappear. `inner` is the frame's
    inner rectangle. None when the border has no coloured band all round."""
    h, w = bgr.shape[:2]
    band = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2HSV)[:, :, 1] > 60
    l, t, r, b = (int(v) for v in inner)
    rows, cols = slice(t, b), slice(l, r)

    def outer(profile: np.ndarray) -> int:
        """Index where the band ends when walking outward along `profile` (fractions)."""
        run = 0
        for i, frac in enumerate(profile):
            run = run + 1 if frac < 0.12 else 0
            if run >= max(4, len(profile) // 60):
                return i - run + 1
        return len(profile)

    left = l - outer(band[rows, :l][:, ::-1].mean(axis=0))
    right = r + outer(band[rows, r:].mean(axis=0))
    top = t - outer(band[:t, cols][::-1, :].mean(axis=1))
    bottom = b + outer(band[b:, cols].mean(axis=1))
    widths = [l - left, right - r, t - top, bottom - b]
    if min(widths) < 0.01 * min(h, w):
        return None
    return max(0, left), max(0, top), min(w, right), min(h, bottom)


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


def _readable_words(gray: np.ndarray) -> int:
    lang = "vie" if "vie" in available_languages() else "eng"
    try:
        pars = ocr_paragraphs(gray, lang, psm=3)
    except RuntimeError:
        return 0
    return sum(1 for p in pars for ln in p.lines for w in ln.words if w.conf >= 85 and len(w.text) >= 2)


def _confirm_orientation(gray: np.ndarray, rot: int) -> int:
    """OSD was unsure: read the page both ways and keep the orientation with more clear words."""
    h, w = gray.shape
    crop = gray[h // 4:3 * h // 4, w // 8:7 * w // 8]  # the centre is enough and much faster
    turned = cv2.rotate(crop, _ROTATE_CW[rot])
    return rot if _readable_words(turned) > _readable_words(crop) else 0


def _rotation_matrix(rot: int, w: int, h: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Exact 90/180/270° clockwise rotation as a 3x3 transform, with the new (w, h)."""
    if rot == 90:
        return np.array([[0, -1, h - 1], [1, 0, 0], [0, 0, 1]], np.float64), (h, w)
    if rot == 180:
        return np.array([[-1, 0, w - 1], [0, -1, h - 1], [0, 0, 1]], np.float64), (w, h)
    return np.array([[0, 1, 0], [-1, 0, w - 1], [0, 0, 1]], np.float64), (h, w)


class _Geometry:
    """Accumulates every geometric correction as one transform from the original photo, so
    the final page is resampled exactly once (each resampling blurs small print)."""

    def __init__(self, src: np.ndarray):
        self.src = src
        self.m = np.eye(3)
        self.size = (src.shape[1], src.shape[0])  # (w, h)
        self.changed = False

    def then(self, m: np.ndarray, size: tuple[int, int]):
        self.m = np.asarray(m, np.float64) @ self.m
        self.size = (int(size[0]), int(size[1]))
        self.changed = True

    def render(self, interp=cv2.INTER_LINEAR) -> np.ndarray:
        if not self.changed:
            return self.src.copy()
        return cv2.warpPerspective(self.src, self.m, self.size, flags=interp,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))


def preprocess(path: str, auto_crop: bool = True, rotation: int = 0) -> Prepared:
    """`rotation` (clockwise degrees) is a manual override; 0 means detect automatically."""
    notes = []
    cropped = False
    src = load_image(path)
    if max(src.shape[:2]) > MAX_SIDE:
        k = MAX_SIDE / max(src.shape[:2])
        src = cv2.resize(src, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    geo = _Geometry(src)
    bgr = src
    squared = False

    if auto_crop:
        quad = find_document_quad(bgr)
        if quad is None:
            quad = find_document_by_edges(bgr)
        if quad is not None:
            tl, tr, br, bl = quad
            w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
            h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
            dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
            geo.then(cv2.getPerspectiveTransform(quad, dst), (w, h))
            bgr = geo.render()
            cropped = True
            notes.append("Đã cắt và nắn phối cảnh trang giấy")

    rot = rotation % 360
    if not rot:
        small_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        k = min(1.0, 2000 / max(small_gray.shape))
        if k < 1:
            small_gray = cv2.resize(small_gray, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        rot, conf = detect_orientation(small_gray)
        if rot in _ROTATE_CW and conf < 2.0:
            rot = _confirm_orientation(small_gray, rot)
        if rot in _ROTATE_CW:
            notes.append(f"Đã tự xoay trang {rot}°")
    if rot in _ROTATE_CW:
        m, size = _rotation_matrix(rot, *geo.size)
        geo.then(m, size)
        bgr = cv2.rotate(bgr, _ROTATE_CW[rot])

    if auto_crop:
        # a framed document (certificate…) whose frame is a parallelogram/trapezoid in the
        # photo: make the frame an exact rectangle, then cut cleanly around it
        frame = find_frame_quad(bgr)
        if frame is not None:
            w0, h0 = geo.size
            tl, tr, br, bl = frame
            left, top = (tl[0] + bl[0]) / 2, (tl[1] + tr[1]) / 2
            right, bottom = (tr[0] + br[0]) / 2, (bl[1] + br[1]) / 2
            target = np.array([[left, top], [right, top], [right, bottom], [left, bottom]], np.float32)
            geo.then(cv2.getPerspectiveTransform(frame.astype(np.float32), target), (w0, h0))
            bgr = geo.render()
            box = band_box(bgr, (left, top, right, bottom))
            if box is not None:
                # the page is the document itself: cut exactly at the outer edge of its border
                x0, y0, x1, y1 = box
                geo.then(np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], np.float64), (x1 - x0, y1 - y0))
                bgr = geo.render()
            squared = True
            cropped = False  # edges are clean now; no extra trim needed
            notes.append("Đã nắn khung giấy tờ thành hình chữ nhật")

    # measure skew and print size on the working image, then fold both into the transform
    color = normalize_background(bgr)
    binary = binarize(cv2.cvtColor(color, cv2.COLOR_BGR2GRAY))
    angle = 0.0 if squared else estimate_skew(binary)  # a squared frame already defines level
    if abs(angle) >= 0.1:
        w0, h0 = geo.size
        m = cv2.getRotationMatrix2D((w0 / 2, h0 / 2), angle, 1.0)
        cos, sin = abs(m[0, 0]), abs(m[0, 1])
        nw, nh = int(h0 * sin + w0 * cos), int(h0 * cos + w0 * sin)
        m[0, 2] += nw / 2 - w0 / 2
        m[1, 2] += nh / 2 - h0 / 2
        geo.then(np.vstack([m, [0, 0, 1]]), (nw, nh))
        notes.append(f"Đã chỉnh nghiêng {angle:+.2f}°")
    th = estimate_text_height(binary)
    if th > 0:
        scale = min(TARGET_TEXT_HEIGHT / th, 3.0, MAX_SIDE / max(geo.size))
        if scale > 1.25 or scale < 0.6:
            geo.then(np.diag([scale, scale, 1.0]), (int(geo.size[0] * scale), int(geo.size[1] * scale)))

    # the one real resampling of the photo
    bgr = geo.render(cv2.INTER_LANCZOS4 if geo.changed else cv2.INTER_LINEAR)
    raw = bgr.copy()
    color = normalize_background(bgr)
    if auto_crop:
        _trim_margins(color, raw)
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    binary = binarize(gray)
    # A photographed sheet keeps a sliver of shadow/table along its edges; the page output
    # steps inside it (OCR still sees the full crop, which reads better).
    trim = int(0.008 * min(color.shape[:2])) if cropped else 0
    return Prepared(color, gray, binary, raw, notes, trim)
