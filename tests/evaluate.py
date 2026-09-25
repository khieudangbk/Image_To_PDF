"""Converts the synthetic samples and measures text accuracy of the produced PDFs."""
import json
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from img2pdf.pipeline import convert  # noqa: E402
from img2pdf.settings import Settings  # noqa: E402

SAMPLES = Path(__file__).parent / "samples"
OUTDIR = Path(__file__).parent / "out"
MIN_RECALL = 0.95


def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def words(s: str) -> list[str]:
    return unicodedata.normalize("NFC", s).split()


def main() -> int:
    OUTDIR.mkdir(exist_ok=True)
    ok = True
    for img in sorted(p for p in SAMPLES.glob("0*") if p.suffix != ".json"):
        truth = json.loads(img.with_suffix(".json").read_text(encoding="utf-8"))
        gt_words = [w for line in truth["text"] for w in words(line)]
        gt_counter = Counter(gt_words)
        out = OUTDIR / (img.stem + ".pdf")
        t0 = time.time()
        pages = convert([str(img)], str(out), Settings())
        dt = time.time() - t0
        with pymupdf.open(out) as doc:
            text = "".join(p.get_text() for p in doc)
            n_images = sum(len(p.get_images()) for p in doc)
        got = words(text)
        common = sum((Counter(got) & gt_counter).values())
        recall, precision = common / len(gt_words), common / max(len(got), 1)
        gt_joined = " ".join(gt_words)
        order_cer = levenshtein(" ".join(got), gt_joined) / len(gt_joined)
        page = pages[0]
        ok &= recall >= MIN_RECALL
        print(f"{img.name}: {dt:4.1f}s  words recall {recall:.1%} precision {precision:.1%}  "
              f"reading-order CER {order_cer:.1%}  blocks {len(page.blocks)} tables {len(page.tables)} "
              f"figures {len(page.figures)} (pdf images {n_images})  {page.notes}")
        print("   missing :", list((gt_counter - Counter(got)).elements())[:30])
        print("   extra   :", list((Counter(got) - gt_counter).elements())[:30])
        print("   headings:", [b.text.replace("\n", " / ")[:50] for b in page.blocks if b.kind == "heading"])
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
