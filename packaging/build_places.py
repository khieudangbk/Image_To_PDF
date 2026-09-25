"""Builds models/dict/places.txt (Vietnamese administrative place names) from the open
dvhcvn dataset: https://github.com/daohoangson/dvhcvn (data/dvhcvn.json)."""
import json
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PREFIXES = ("thành phố", "tỉnh", "quận", "huyện", "thị xã", "thị trấn", "phường", "xã")


def main(src: str):
    data = json.loads(Path(src).read_text(encoding="utf-8"))["data"]
    names: set[str] = set()

    def add(full: str):
        full = " ".join(unicodedata.normalize("NFC", full).lower().split())
        names.add(full)
        for p in PREFIXES:
            if full.startswith(p + " "):
                bare = full[len(p) + 1:]
                if not bare.isdigit():
                    names.add(bare)
                break

    for l1 in data:
        add(l1["name"])
        for l2 in l1.get("level2s", []):
            add(l2["name"])
            for l3 in l2.get("level3s", []):
                add(l3["name"])
    names = {n for n in names if len(n.split()) >= 2}  # single syllables add nothing as phrases
    out = ROOT / "models" / "dict" / "places.txt"
    out.write_text("# Địa danh hành chính Việt Nam (nguồn: dvhcvn)\n" + "\n".join(sorted(names)) + "\n",
                   encoding="utf-8")
    print(f"{len(names)} place names -> {out}")


if __name__ == "__main__":
    main(sys.argv[1])
