"""Generates synthetic Vietnamese document images with known ground truth."""
import json
import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).parent / "samples"
FONT_DIR = Path(r"C:\Windows\Fonts")
DPI = 300
W, H = int(8.27 * DPI), int(11.69 * DPI)
PX = DPI / 72


def font(size_pt, bold=False):
    return ImageFont.truetype(str(FONT_DIR / ("timesbd.ttf" if bold else "times.ttf")), int(size_pt * PX))


class Doc:
    def __init__(self):
        self.img = Image.new("RGB", (W, H), "white")
        self.d = ImageDraw.Draw(self.img)
        self.truth = []
        self.bold_truth = []
        self.y = int(0.8 * DPI)
        self.left, self.right = int(1.2 * DPI), W - int(0.8 * DPI)

    def text_at(self, x, y, s, f, bold=False, record=True):
        self.d.text((x, y), s, font=f, fill="black")
        if record:
            self.truth.append(s)
            if bold:
                self.bold_truth.append(s)

    def centered(self, s, f, x0, x1, bold=False):
        w = self.d.textlength(s, font=f)
        self.text_at((x0 + x1 - w) / 2, self.y, s, f, bold)
        return w

    def paragraph(self, s, f, indent=0.5 * DPI, spacing=1.35):
        words, lines, cur = s.split(), [], []
        width = self.right - self.left
        for wd in words:
            trial = " ".join(cur + [wd])
            first = not lines
            avail = width - (indent if first else 0)
            if self.d.textlength(trial, font=f) > avail and cur:
                lines.append(cur)
                cur = [wd]
            else:
                cur.append(wd)
        lines.append(cur)
        size = f.size
        for i, ln in enumerate(lines):
            x = self.left + (indent if i == 0 else 0)
            avail = width - (indent if i == 0 else 0)
            if i < len(lines) - 1 and len(ln) > 1:
                total = sum(self.d.textlength(w, font=f) for w in ln)
                gap = (avail - total) / (len(ln) - 1)
                for w in ln:
                    self.d.text((x, self.y), w, font=f, fill="black")
                    x += self.d.textlength(w, font=f) + gap
            else:
                self.d.text((x, self.y), " ".join(ln), font=f, fill="black")
            self.y += int(size * spacing)
        self.truth.append(" ".join(words))
        self.y += int(size * 0.3)


def build():
    d = Doc()
    f13, f13b, f14b, f12 = font(13), font(13, True), font(14, True), font(12)
    colL = (d.left - int(0.4 * DPI), d.left + int(2.6 * DPI))
    colR = (d.left + int(2.6 * DPI), d.right + int(0.2 * DPI))
    y0 = d.y
    d.centered("UBND TỈNH NAM ĐỊNH", f13, *colL)
    d.y += int(13 * PX * 1.3)
    w = d.centered("SỞ GIÁO DỤC VÀ ĐÀO TẠO", f13b, *colL, bold=True)
    d.y = y0
    d.centered("CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM", f13b, *colR, bold=True)
    d.y += int(13 * PX * 1.3)
    w2 = d.centered("Độc lập - Tự do - Hạnh phúc", f14b, *colR, bold=True)
    d.y += int(14 * PX * 1.35)
    cx = (colR[0] + colR[1]) / 2
    d.d.line([(cx - w2 / 2, d.y), (cx + w2 / 2, d.y)], fill="black", width=4)
    cxl = (colL[0] + colL[1]) / 2
    d.d.line([(cxl - w / 4, d.y), (cxl + w / 4, d.y)], fill="black", width=4)
    d.y += int(0.35 * DPI)
    d.text_at(colL[0] + int(0.5 * DPI), d.y, "Số: 1234/QĐ-SGDĐT", f13)
    d.text_at(colR[0] + int(1.0 * DPI), d.y, "Nam Định, ngày 15 tháng 9 năm 2026", font(13), record=True)
    d.y += int(0.6 * DPI)

    d.centered("QUYẾT ĐỊNH", font(15, True), d.left, d.right, bold=True)
    d.y += int(15 * PX * 1.4)
    d.centered("Về việc ban hành Quy chế tổ chức kỳ thi chọn học sinh giỏi cấp tỉnh", f14b, d.left, d.right, bold=True)
    d.y += int(0.5 * DPI)
    d.centered("GIÁM ĐỐC SỞ GIÁO DỤC VÀ ĐÀO TẠO", f14b, d.left, d.right, bold=True)
    d.y += int(0.45 * DPI)

    d.paragraph("Căn cứ Luật Giáo dục ngày 14 tháng 6 năm 2019; Căn cứ Nghị định số 127/2018/NĐ-CP ngày 21 "
                "tháng 9 năm 2018 của Chính phủ quy định trách nhiệm quản lý nhà nước về giáo dục; theo đề nghị "
                "của Trưởng phòng Giáo dục trung học, Chánh Văn phòng Sở và các đơn vị liên quan.", f13)
    d.paragraph("Điều 1. Ban hành kèm theo Quyết định này Quy chế tổ chức kỳ thi chọn học sinh giỏi cấp tỉnh "
                "năm học 2026-2027 đối với các trường trung học phổ thông trên địa bàn tỉnh. Quyết định có hiệu "
                "lực kể từ ngày ký; các quy định trước đây trái với Quyết định này đều bãi bỏ.", f13)
    d.y += int(0.15 * DPI)

    # Table
    cols = [0.7, 2.6, 1.3, 1.6]
    rows = [["STT", "Môn thi", "Thời gian", "Ngày thi"],
            ["1", "Ngữ văn", "180 phút", "12/12/2026"],
            ["2", "Toán học", "180 phút", "12/12/2026"],
            ["3", "Tiếng Anh", "150 phút", "13/12/2026"],
            ["4", "Lịch sử và Địa lý", "150 phút", "13/12/2026"]]
    tx = d.left + int(0.2 * DPI)
    rh = int(13 * PX * 1.9)
    xs = [tx]
    for c in cols:
        xs.append(xs[-1] + int(c * DPI))
    ys = [d.y + i * rh for i in range(len(rows) + 1)]
    for x in xs:
        d.d.line([(x, ys[0]), (x, ys[-1])], fill="black", width=3)
    for y in ys:
        d.d.line([(xs[0], y), (xs[-1], y)], fill="black", width=3)
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            f = f13b if r == 0 else f13
            tw = d.d.textlength(cell, font=f)
            d.text_at((xs[c] + xs[c + 1] - tw) / 2, ys[r] + rh * 0.22, cell, f, bold=(r == 0))
    d.y = ys[-1] + int(0.35 * DPI)

    d.paragraph("Điều 2. Chánh Văn phòng, Trưởng các phòng thuộc Sở, Hiệu trưởng các trường trung học phổ "
                "thông chịu trách nhiệm thi hành Quyết định này.", f13)
    d.y += int(0.2 * DPI)
    sig_x0 = d.left + int(3.4 * DPI)
    sig_x1 = d.right
    d.centered("GIÁM ĐỐC", f13b, sig_x0, sig_x1, bold=True)
    sy = d.y + int(0.2 * DPI)
    # Red stamp + blue signature (figures, not text)
    scx, scy, r = int((sig_x0 + sig_x1) / 2 - 0.5 * DPI), sy + int(0.55 * DPI), int(0.55 * DPI)
    d.d.ellipse([scx - r, scy - r, scx + r, scy + r], outline=(210, 30, 40), width=10)
    d.d.ellipse([scx - r + 40, scy - r + 40, scx + r - 40, scy + r - 40], outline=(210, 30, 40), width=5)
    d.d.regular_polygon((scx, scy, 45), 5, fill=(210, 30, 40))
    pts = [(int((sig_x0 + sig_x1) / 2 - 0.3 * DPI + t * 3), int(sy + 0.6 * DPI + 60 * math.sin(t / 12)))
           for t in range(200)]
    d.d.line(pts, fill=(20, 40, 160), width=7)
    d.y = sy + int(1.2 * DPI)
    d.centered("Nguyễn Văn An", f13b, sig_x0, sig_x1, bold=True)
    d.y += int(0.5 * DPI)
    d.text_at(d.left, d.y, "Nơi nhận:", font(12, True), bold=True)
    d.y += int(12 * PX * 1.3)
    d.text_at(d.left, d.y, "- Như Điều 2;", f12)
    d.y += int(12 * PX * 1.3)
    d.text_at(d.left, d.y, "- Lưu: VT, GDTrH.", f12)
    return d


def build_columns():
    """Two-column article with a centered title, bullet list and mixed bold text."""
    d = Doc()
    f12, f12b = font(12), font(12, True)
    d.centered("BẢN TIN KHOA HỌC VÀ CÔNG NGHỆ", font(18, True), d.left, d.right, bold=True)
    d.y += int(18 * PX * 1.6)
    d.centered("Số 42 - Tháng 9 năm 2026", font(12), d.left, d.right)
    d.y += int(0.45 * DPI)
    top = d.y
    gutter = int(0.35 * DPI)
    mid = (d.left + d.right) // 2
    columns = [(d.left, mid - gutter // 2), (mid + gutter // 2, d.right)]
    texts = [
        ["Trí tuệ nhân tạo đang thay đổi cách chúng ta làm việc và học tập. Các mô hình ngôn ngữ lớn có thể "
         "đọc hiểu văn bản, dịch thuật và tóm tắt tài liệu dài chỉ trong vài giây.",
         "Tại Việt Nam, nhiều trường đại học đã đưa môn học về dữ liệu và thuật toán vào chương trình đào tạo "
         "chính thức cho sinh viên năm thứ nhất."],
        ["Chuyển đổi số là nhiệm vụ trọng tâm của các cơ quan nhà nước trong giai đoạn tới. Mục tiêu đến năm "
         "2030 là hoàn thành việc số hóa hồ sơ hành chính.",
         "Những lợi ích chính bao gồm:"],
    ]
    bullets = ["- Tiết kiệm thời gian xử lý hồ sơ;", "- Giảm chi phí in ấn và lưu trữ;",
               "- Minh bạch thông tin với người dân."]
    for (x0, x1), paras in zip(columns, texts):
        d.left, d.right, d.y = x0, x1, top
        for p in paras:
            d.paragraph(p, f12, indent=int(0.3 * DPI))
        if paras is texts[1]:
            for b in bullets:
                d.text_at(x0 + int(0.2 * DPI), d.y, b, f12)
                d.y += int(12 * PX * 1.35)
    d.left, d.right = int(1.2 * DPI), W - int(0.8 * DPI)
    d.y = max(d.y, top) + int(0.6 * DPI)
    d.text_at(d.left, d.y, "Ghi chú:", f12b, bold=True)
    d.text_at(d.left + int(0.9 * DPI), d.y, "Bản tin được phát hành miễn phí hàng tháng.", f12)
    return d


def degrade_scan(img: np.ndarray, rng) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), 1.3, 1.0)
    img = cv2.warpAffine(img, m, (w, h), borderValue=(255, 255, 255))
    img = cv2.GaussianBlur(img, (3, 3), 0.8)
    noise = rng.normal(0, 9, img.shape)
    img = np.clip(img.astype(np.float32) * 0.95 + 8 + noise, 0, 255).astype(np.uint8)
    img = cv2.resize(img, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)  # ~210 dpi
    return img


def degrade_photo(img: np.ndarray, rng) -> np.ndarray:
    h, w = img.shape[:2]
    small = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    yy, xx = np.mgrid[0:sh, 0:sw]
    shade = 0.72 + 0.28 * (xx / sw) * (1 - 0.3 * yy / sh)
    small = np.clip(small * shade[..., None], 0, 255).astype(np.uint8)
    CW, CH = int(sw * 1.35), int(sh * 1.25)
    bg = np.full((CH, CW, 3), (60, 70, 85), np.uint8)
    src = np.float32([[0, 0], [sw, 0], [sw, sh], [0, sh]])
    dst = np.float32([[CW * 0.14, CH * 0.07], [CW * 0.86, CH * 0.10], [CW * 0.90, CH * 0.94], [CW * 0.10, CH * 0.91]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(small, M, (CW, CH))
    mask = cv2.warpPerspective(np.full((sh, sw), 255, np.uint8), M, (CW, CH))
    bg[mask > 0] = warped[mask > 0]
    bg = cv2.GaussianBlur(bg, (3, 3), 0.7)
    noise = rng.normal(0, 6, bg.shape)
    return np.clip(bg + noise, 0, 255).astype(np.uint8)


def main():
    OUT.mkdir(exist_ok=True)
    d = build()
    clean = cv2.cvtColor(np.asarray(d.img), cv2.COLOR_RGB2BGR)
    rng = np.random.default_rng(7)
    cv2.imwrite(str(OUT / "01_clean.png"), clean)
    cv2.imwrite(str(OUT / "02_scan.jpg"), degrade_scan(clean, rng), [cv2.IMWRITE_JPEG_QUALITY, 80])
    cv2.imwrite(str(OUT / "03_photo.jpg"), degrade_photo(clean, rng), [cv2.IMWRITE_JPEG_QUALITY, 85])
    truth = {"text": d.truth, "bold": d.bold_truth}
    for stem in ("01_clean", "02_scan", "03_photo"):
        (OUT / f"{stem}.json").write_text(json.dumps(truth, ensure_ascii=False, indent=1), encoding="utf-8")

    c = build_columns()
    cols = cv2.cvtColor(np.asarray(c.img), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(OUT / "04_columns.jpg"), degrade_scan(cols, rng), [cv2.IMWRITE_JPEG_QUALITY, 85])
    (OUT / "04_columns.json").write_text(json.dumps({"text": c.truth, "bold": c.bold_truth}, ensure_ascii=False,
                                                    indent=1), encoding="utf-8")
    print("samples written to", OUT)


if __name__ == "__main__":
    main()
