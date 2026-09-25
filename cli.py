import argparse
import sys
import time

from img2pdf.fonts import FAMILIES
from img2pdf.pipeline import collect_images, convert
from img2pdf.settings import PAPERS, Settings


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Chuyển ảnh tài liệu thành PDF văn bản thật (OCR tiếng Việt).")
    ap.add_argument("inputs", nargs="+", help="Ảnh hoặc thư mục ảnh (theo thứ tự trang)")
    ap.add_argument("-o", "--output", required=True, help="File PDF đầu ra")
    ap.add_argument("--lang", default="vie", help="Ngôn ngữ Tesseract, vd: vie, vie+eng")
    ap.add_argument("--font", default="Times New Roman", choices=list(FAMILIES))
    ap.add_argument("--paper", default="A4", choices=PAPERS)
    ap.add_argument("--no-crop", action="store_true", help="Không tự cắt/nắn phối cảnh")
    ap.add_argument("--no-tables", action="store_true", help="Không nhận dạng bảng")
    ap.add_argument("--no-figures", action="store_true", help="Không giữ hình (logo, con dấu, chữ ký)")
    a = ap.parse_args(argv)

    files = collect_images(a.inputs)
    if not files:
        print("Không tìm thấy ảnh hợp lệ.", file=sys.stderr)
        return 2
    s = Settings(lang=a.lang, font=a.font, paper=a.paper, auto_crop=not a.no_crop,
                 detect_tables=not a.no_tables, keep_figures=not a.no_figures)
    t0 = time.time()

    def report(i, page, err):
        name = files[i]
        print(f"[{i + 1}/{len(files)}] {'LỖI: ' + err if err else 'xong'} - {name}", flush=True)

    convert(files, a.output, s, report)
    print(f"Đã tạo {a.output} ({len(files)} trang, {time.time() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
