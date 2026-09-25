"""Context-based repair of Vietnamese diacritics that OCR commonly confuses (ố/ó, ổ/ô, ỉ/ĩ...).

A syllable is replaced by a diacritic variant only when the original forms no known
compound word with its neighbours and a single variant has the strongest evidence.
"""
import re
import unicodedata
from functools import lru_cache

from .paths import app_root

_TONES = {"̀", "́", "̃", "̉", "̣"}
_MODS = {"̂", "̆", "̛"}
_SPLIT = re.compile(r"^(\W*)(.*?)(\W*)$", re.UNICODE)
# Old/new tone placement are both correct ("hoà" = "hòa"); compare in one canonical form.
_PLACEMENT = {"òa": "oà", "óa": "oá", "ỏa": "oả", "õa": "oã", "ọa": "oạ",
              "òe": "oè", "óe": "oé", "ỏe": "oẻ", "õe": "oẽ", "ọe": "oẹ",
              "ùy": "uỳ", "úy": "uý", "ủy": "uỷ", "ũy": "uỹ", "ụy": "uỵ"}


def skeleton(s: str) -> str:
    nfd = unicodedata.normalize("NFD", s.lower())
    return "".join(ch for ch in nfd if ch not in _TONES and ch not in _MODS).replace("đ", "d")


def canonical(syl: str) -> str:
    syl = unicodedata.normalize("NFC", syl.lower())
    tail = syl[-2:]
    return syl[:-2] + _PLACEMENT[tail] if tail in _PLACEMENT else syl


_SURNAMES = {canonical(s) for s in (
    "nguyễn trần lê phạm hoàng huỳnh phan vũ võ đặng bùi đỗ hồ ngô dương lý đinh đào "
    "lương trịnh mai tô tạ châu cao lâm quách thái hà kiều vương doãn triệu la phùng").split()}
MAX_ORIG_COUNT = 60   # never "correct" a very common syllable
MIN_RATIO = 3.0       # the variant must be this much more common than the original


class Lexicon:
    def __init__(self):
        self.phrases: set[str] = set()
        self.domain: set[str] = set()
        d = app_root() / "models" / "dict"
        for name in ("Viet74K.txt", "extra_phrases.txt", "user_phrases.txt"):
            p = d / name
            if not p.exists():
                continue
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    ph = " ".join(canonical(s) for s in line.split())
                    self.phrases.add(ph)
                    if name != "Viet74K.txt":
                        self.domain.add(ph)
        self.index: dict[str, set[str]] = {}
        self.count: dict[str, int] = {}
        for ph in self.phrases:
            for syl in ph.split():
                if syl.isalpha():
                    self.index.setdefault(skeleton(syl), set()).add(syl)
                    self.count[syl] = self.count.get(syl, 0) + 1
        # skeleton prefix -> longer skeletons (up to 2 extra letters), for truncated syllables
        self.prefixes: dict[str, list[str]] = {}
        for key in self.index:
            for cut in (1, 2):
                if len(key) - cut >= 2:
                    self.prefixes.setdefault(key[:-cut], []).append(key)

    def has(self, *syllables: str) -> bool:
        return " ".join(syllables) in self.phrases

    def in_domain(self, *syllables: str) -> bool:
        return " ".join(syllables) in self.domain


@lru_cache(maxsize=1)
def lexicon() -> Lexicon:
    return Lexicon()


def _match_case(src: str, repl: str) -> str:
    if src.isupper() and len(src) > 1:
        return repl.upper()
    if src[:1].isupper():
        return repl[:1].upper() + repl[1:]
    return repl


def _diff(a: str, b: str) -> int:
    na, nb = unicodedata.normalize("NFD", a), unicodedata.normalize("NFD", b)
    return len(set(na) ^ set(nb))


def correct_tokens(tokens: list[str]) -> list[str]:
    lex = lexicon()
    if not lex.phrases:
        return tokens
    parts = [_SPLIT.match(t).groups() for t in tokens]
    low = [canonical(p[1]) for p in parts]
    n = len(low)

    def evidence(i: int, word: str) -> tuple[int, bool]:
        """(length of the longest known phrase containing `word` at i, found in the domain list)."""
        best, domain = 0, False
        for size in (2, 3):
            for start in range(i - size + 1, i + 1):
                if start < 0 or start + size > n:
                    continue
                seq = low[start:i] + [word] + low[i + 1:start + size]
                if lex.has(*seq):
                    best = max(best, size)
                    domain = domain or lex.in_domain(*seq)
        return best, domain

    def suspicious(i: int) -> bool:
        core = low[i]
        if not core.isalpha() or len(core) > 7 or core in _SURNAMES or evidence(i, core)[0]:
            return False
        orig = parts[i][1]
        sentence_start = i == 0 or tokens[i - 1][-1:] in ".:;!?"
        # Capitalised mid-sentence word: probably a proper noun, unless it is not a real syllable.
        return not (orig[:1].isupper() and not orig.isupper() and not sentence_start
                    and lex.count.get(core, 0) > 2)

    def plausible(orig: str, cand: str) -> bool:
        n_orig = lex.count.get(orig, 0)
        return n_orig <= MAX_ORIG_COUNT and lex.count.get(cand, 0) >= MIN_RATIO * max(n_orig, 1)

    def variants(i: int) -> set[str]:
        skel = skeleton(low[i])
        out = set(lex.index.get(skel, set()))
        if lex.count.get(low[i], 0) == 0 and len(skel) >= 2:
            # not a real syllable: OCR may also have dropped a final letter ("ngh" -> "nghề")
            for key in lex.prefixes.get(skel, ()):
                out |= lex.index[key]
        return out - {low[i]}

    out = list(tokens)

    def replace(i: int, word: str):
        pre, orig, post = parts[i]
        out[i] = pre + _match_case(orig, word) + post
        low[i] = word

    for i in range(n):
        if not suspicious(i):
            continue
        scored = []
        for c in variants(i):
            size, domain = evidence(i, c)
            if size and (domain or plausible(low[i], c)):
                scored.append((size, domain, -_diff(low[i], c), c))
        scored.sort(reverse=True)
        if scored and not (len(scored) > 1 and scored[0][:3] == scored[1][:3]):
            replace(i, scored[0][3])

    # Two neighbouring syllables both misread (e.g. "chuyên đôi" -> "chuyển đổi").
    for i in range(n - 1):
        if not (suspicious(i) and suspicious(i + 1)):
            continue
        if any(len(parts[j][1]) > 1 and parts[j][1].isupper() for j in (i, i + 1)):
            continue  # all-caps pairs are mostly names ("ĐẶNG THỊ"); too risky to rewrite both
        pairs = []
        for a in variants(i):
            for b in variants(i + 1):
                if not lex.has(a, b):
                    continue
                domain = lex.in_domain(a, b)
                # Both syllables being orphans while a variant pair is a known word is strong
                # evidence on its own; only reject variants that are much rarer.
                if domain or all(lex.count.get(x, 0) >= 0.5 * lex.count.get(o, 0)
                                 for x, o in ((a, low[i]), (b, low[i + 1]))):
                    pairs.append((domain, -(_diff(low[i], a) + _diff(low[i + 1], b)), a, b))
        pairs.sort(reverse=True)
        if pairs and not (len(pairs) > 1 and pairs[0][:2] == pairs[1][:2]):
            replace(i, pairs[0][2])
            replace(i + 1, pairs[0][3])
    return out
