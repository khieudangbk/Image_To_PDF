import os
import re
import subprocess
import sys
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import cv2
import numpy as np

from .paths import tessdata_dir, tesseract_exe

_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0
_LINE_CLASSES = {"ocr_line", "ocr_header", "ocr_textfloat", "ocr_caption"}


@dataclass
class OWord:
    text: str
    bbox: tuple[int, int, int, int]
    conf: float


@dataclass
class OLine:
    bbox: tuple[int, int, int, int]
    baseline: float
    x_size: float
    words: list[OWord]


@dataclass
class OPar:
    bbox: tuple[int, int, int, int]
    lines: list[OLine]


def _run(img: np.ndarray, args: list[str], timeout: int = 600) -> str:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("Không mã hoá được ảnh để OCR")
    env = os.environ.copy()
    env["TESSDATA_PREFIX"] = str(tessdata_dir())
    env["OMP_THREAD_LIMIT"] = "1"
    proc = subprocess.run([tesseract_exe(), "stdin", "stdout", *args], input=buf.tobytes(),
                          capture_output=True, env=env, timeout=timeout, creationflags=_NO_WINDOW)
    if proc.returncode != 0:
        raise RuntimeError("Tesseract lỗi: " + proc.stderr.decode("utf-8", "replace").strip())
    return proc.stdout.decode("utf-8", "replace")


def available_languages() -> set[str]:
    return {p.stem for p in tessdata_dir().glob("*.traineddata")}


def detect_orientation(gray: np.ndarray) -> tuple[int, float]:
    """Returns (degrees to rotate clockwise to make the page upright, confidence)."""
    try:
        out = _run(gray, ["--psm", "0", "-l", "osd"], timeout=60)
    except (RuntimeError, subprocess.TimeoutExpired):
        return 0, 0.0
    rot = re.search(r"Rotate:\s*(\d+)", out)
    conf = re.search(r"Orientation confidence:\s*([\d.]+)", out)
    return (int(rot.group(1)) if rot else 0), (float(conf.group(1)) if conf else 0.0)


def ocr_paragraphs(img: np.ndarray, lang: str, psm: int = 3, offset=(0, 0)) -> list[OPar]:
    out = _run(img, ["-l", lang, "--oem", "1", "--psm", str(psm), "--dpi", "300",
                    "-c", "tessedit_create_hocr=1", "-c", "hocr_font_info=0"])
    return parse_hocr(out, offset)


def _title(el) -> dict[str, list[str]]:
    d = {}
    for part in (el.get("title") or "").split(";"):
        part = part.strip()
        if part:
            k, _, v = part.partition(" ")
            d[k] = v.split()
    return d


def _bbox(t, offset):
    x0, y0, x1, y1 = (int(v) for v in t["bbox"][:4])
    return (x0 + offset[0], y0 + offset[1], x1 + offset[0], y1 + offset[1])


def parse_hocr(text: str, offset=(0, 0)) -> list[OPar]:
    root = ET.fromstring(text.encode("utf-8"))
    pars = []
    for par in root.iter():
        if par.get("class") != "ocr_par":
            continue
        lines = []
        for ln in par.iter():
            if ln.get("class") not in _LINE_CLASSES:
                continue
            t = _title(ln)
            bbox = _bbox(t, offset)
            words = []
            for w in ln.iter():
                if w.get("class") != "ocrx_word":
                    continue
                wt = unicodedata.normalize("NFC", "".join(w.itertext())).strip()
                if not wt:
                    continue
                tw = _title(w)
                words.append(OWord(wt, _bbox(tw, offset), float(tw.get("x_wconf", ["0"])[0])))
            if not words:
                continue
            baseline = bbox[3]
            if "baseline" in t:
                slope, off = float(t["baseline"][0]), float(t["baseline"][1])
                baseline = bbox[3] + off + slope * (bbox[2] - bbox[0]) / 2
            x_size = float(t.get("x_size", [bbox[3] - bbox[1]])[0])
            lines.append(OLine(bbox, baseline, x_size, words))
        if lines:
            pars.append(OPar(_bbox(_title(par), offset), lines))
    return pars
