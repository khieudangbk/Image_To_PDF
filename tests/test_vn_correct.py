"""Regression cases for the Vietnamese diacritic corrector (fixes and must-not-touch cases)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from img2pdf.vn_correct import correct_tokens  # noqa: E402

CASES = [
    # OCR output -> expected
    ("GIÁM ĐÓC SỞ GIÁO DỤC", "GIÁM ĐỐC SỞ GIÁO DỤC"),
    ("trường trung học phô thông trên địa", "trường trung học phổ thông trên địa"),
    ("lực kề từ ngày ký;", "lực kể từ ngày ký;"),
    ("đều bãi bó.", "đều bãi bỏ."),
    ("SỚ GIÁO DỤC VÀ ĐÀO TẠO", "SỞ GIÁO DỤC VÀ ĐÀO TẠO"),
    ("CỘNG HÒA XÃ HỘI CHÚ NGHĨA", "CỘNG HÒA XÃ HỘI CHỦ NGHĨA"),
    ("UBND TĨNH NAM ĐỊNH", "UBND TỈNH NAM ĐỊNH"),
    ("học sinh giói cấp", "học sinh giỏi cấp"),
    ("Chuyên đôi số là nhiệm vụ", "Chuyển đổi số là nhiệm vụ"),
    ("CẤP CHỨNG CHÍ HÀNH NGH KHÁM BỆNH", "CẤP CHỨNG CHỈ HÀNH NGHỀ KHÁM BỆNH"),
    ("CỌNG HÒA XÃ HỘI", "CỘNG HÒA XÃ HỘI"),
    # correct text that must stay untouched
    ("Họ và tên: ĐẶNG THỊ THỦY.", None),
    ("học sinh giỏi cấp tỉnh năm học", None),
    ("bệnh cấp tính", None),
    ("Môn thi", None),
    ("các phòng thuộc Sở, Hiệu trưởng", None),
    ("Nguyễn Văn An", None),
    ("Độc lập - Tự do - Hạnh phúc", None),
    ("sức khỏe tốt, thủy lợi, hòa bình", None),
    ("Căn cứ Luật Giáo dục ngày 14 tháng 6 năm 2019;", None),
    ("chuyên đề", None),
    ("đôi khi", None),
]


def main() -> int:
    failed = 0
    for src, want in CASES:
        want = want or src
        got = " ".join(correct_tokens(src.split()))
        mark = "ok " if got == want else "BAD"
        failed += got != want
        print(f"{mark} {src!r} -> {got!r}" + ("" if got == want else f"  (want {want!r})"))
    print("PASS" if not failed else f"FAIL ({failed})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
