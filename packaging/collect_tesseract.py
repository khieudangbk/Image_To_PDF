"""Copies tesseract.exe and only the DLLs it (transitively) needs into build/tesseract."""
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = Path(sys.argv[1] if len(sys.argv) > 1 else r"C:\Program Files\Tesseract-OCR")
DST = ROOT / "build" / "tesseract"
OBJDUMP = next((p for p in (r"C:\msys64\ucrt64\bin\objdump.exe", r"C:\msys64\usr\bin\objdump.exe",
                            shutil.which("objdump") or "") if p and Path(p).exists()), None)


def deps(binary: Path) -> set[str]:
    if OBJDUMP is None:
        return {p.name for p in SRC.glob("*.dll")}  # no objdump: copy every DLL
    out = subprocess.run([OBJDUMP, "-p", str(binary)], capture_output=True, text=True).stdout
    return set(re.findall(r"DLL Name:\s*(\S+)", out))


def main():
    if DST.exists():
        shutil.rmtree(DST)
    DST.mkdir(parents=True)
    local = {p.name.lower(): p for p in SRC.glob("*.dll")}
    todo, seen = [SRC / "tesseract.exe"], set()
    while todo:
        b = todo.pop()
        shutil.copy2(b, DST / b.name)
        for d in deps(b):
            key = d.lower()
            if key in local and key not in seen:
                seen.add(key)
                todo.append(local[key])
    size = sum(p.stat().st_size for p in DST.iterdir()) / 1e6
    print(f"copied tesseract.exe + {len(seen)} DLLs ({size:.0f} MB) to {DST}")


if __name__ == "__main__":
    main()
