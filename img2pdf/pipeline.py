import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from .layout import build_page
from .model import Page
from .ocr import available_languages
from .preprocess import preprocess
from .render import export_pdf
from .settings import Settings

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif", ".jfif"}


def natural_key(path: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(path))]


def collect_images(inputs: list[str]) -> list[str]:
    files = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            files += sorted((str(f) for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS), key=natural_key)
        elif p.suffix.lower() in IMAGE_EXTS and p.exists():
            files.append(str(p))
    return files


def check_language(lang: str):
    missing = [l for l in lang.split("+") if l not in available_languages()]
    if missing:
        raise RuntimeError(f"Thiếu dữ liệu ngôn ngữ OCR: {', '.join(missing)} (models/tessdata)")


def process_image(path: str, settings: Settings, rotation: int = 0) -> Page:
    prep = preprocess(path, settings.auto_crop, rotation)
    return build_page(path, prep, settings)


def default_workers() -> int:
    return max(1, min(4, (os.cpu_count() or 2) // 2))


def process_many(paths: list[str], settings: Settings,
                 on_page: Callable[[int, Page | None, str | None], None] | None = None,
                 workers: int | None = None) -> list[Page | None]:
    check_language(settings.lang)
    results: list[Page | None] = [None] * len(paths)
    with ThreadPoolExecutor(max_workers=workers or default_workers()) as ex:
        futures = {ex.submit(process_image, p, settings): i for i, p in enumerate(paths)}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                results[i] = fut.result()
                if on_page:
                    on_page(i, results[i], None)
            except Exception as e:  # one bad image must not abort the whole batch
                if on_page:
                    on_page(i, None, str(e))
    return results


def convert(paths: list[str], out_pdf: str, settings: Settings,
            on_page: Callable[[int, Page | None, str | None], None] | None = None) -> list[Page]:
    pages = [p for p in process_many(paths, settings, on_page) if p is not None]
    if not pages:
        raise RuntimeError("Không xử lý được ảnh nào")
    export_pdf(pages, out_pdf, settings, title=Path(out_pdf).stem)
    return pages
