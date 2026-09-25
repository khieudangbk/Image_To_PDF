"""Regression cases for the Vietnamese diacritic corrector (fixes and must-not-touch cases)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from img2pdf.vn_correct import (correct_tokens, fix_abbreviations, fix_confusions,  # noqa: E402
                                fix_place_names, join_split, repair_split)

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
    ("Họ và tên: ĐẶNG THỊ MAI.", None),
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
    ("mảnh đất đó rất rộng", None),
    ("Đó là quy định mới", None),
    ("xã Phước I.ong Thọ,", "xã Phước Long Thọ,"),
    ("huyện Đất Đó, tỉnh Bà Rịa-Vũng Tàu.", "huyện Đất Đỏ, tỉnh Bà Rịa-Vũng Tàu."),
    ("Số: 004835/RRVT - CCHIN", "Số: 004835/BRVT - CCHN"),
    ("QĐ-SGDĐT", None),
    ("Địa chỉ cư trú: Áp Phước Trung,", "Địa chỉ cư trú: Ấp Phước Trung,"),
    ("áp dụng từ ngày", None),
    ("Khám hữa bệnh chuyên khoa", "Khám chữa bệnh chuyên khoa"),
    ("Hà Nội", None),
]


JOINS = [
    ("CỘNG HÒA XÃ HỌ I CHỦ NGHĨA", "CỘNG HÒA XÃ HỘI CHỦ NGHĨA"),
    ("Nguyễn Văn A", None),
    ("cà phê", None),
    ("xã hội chủ nghĩa", None),
    ("Văn bằng chuyên môi n: Bác sĩ", "Văn bằng chuyên môn: Bác sĩ"),
    ("số 5 n", None),
]


def main() -> int:
    failed = 0
    for src, want in JOINS:
        want = want or src
        toks = src.split()
        for i in reversed(join_split(toks)):
            toks[i:i + 2] = [toks[i] + toks[i + 1]]
        for i, word in reversed(repair_split(toks)):
            toks[i:i + 2] = [word]
        got = " ".join(correct_tokens(toks))
        failed += got != want
        print(f"{'ok ' if got == want else 'BAD'} join {src!r} -> {got!r}")
    for src, want in CASES:
        want = want or src
        toks = fix_abbreviations(fix_confusions(src.split()))
        got = " ".join(correct_tokens(fix_place_names(toks)))
        mark = "ok " if got == want else "BAD"
        failed += got != want
        print(f"{mark} {src!r} -> {got!r}" + ("" if got == want else f"  (want {want!r})"))
    print("PASS" if not failed else f"FAIL ({failed})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
