import shutil
import sys
from pathlib import Path


def app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent.parent


def tessdata_dir() -> Path:
    return app_root() / "models" / "tessdata"


def tesseract_exe() -> str:
    bundled = app_root() / "tesseract" / ("tesseract.exe" if sys.platform == "win32" else "tesseract")
    if bundled.exists():
        return str(bundled)
    found = shutil.which("tesseract")
    if found:
        return found
    for p in (r"C:\Program Files\Tesseract-OCR\tesseract.exe",
              r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"):
        if Path(p).exists():
            return p
    raise RuntimeError("Không tìm thấy Tesseract OCR. Hãy cài đặt Tesseract 5 "
                       "(https://github.com/UB-Mannheim/tesseract/wiki).")


def font_dirs() -> list[Path]:
    dirs = [app_root() / "fonts"]
    if sys.platform == "win32":
        dirs.append(Path(r"C:\Windows\Fonts"))
        dirs.append(Path.home() / "AppData/Local/Microsoft/Windows/Fonts")
    else:
        dirs += [Path("/usr/share/fonts"), Path("/usr/local/share/fonts"),
                 Path.home() / ".fonts", Path("/Library/Fonts"), Path("/System/Library/Fonts")]
    return [d for d in dirs if d.exists()]


def find_font(filenames: list[str]) -> Path | None:
    wanted = {f.lower() for f in filenames}
    for d in font_dirs():
        for f in filenames:
            p = d / f
            if p.exists():
                return p
        for p in d.rglob("*.tt[fc]"):
            if p.name.lower() in wanted:
                return p
    return None
