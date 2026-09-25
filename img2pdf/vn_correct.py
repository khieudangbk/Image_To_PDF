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
# Everyday function words: frequent in text but rare in a compound-word list, so the
# dictionary would wrongly treat them as suspicious. They are never rewritten.
_PROTECTED = {canonical(s) for s in (
    "đó đây đấy kia này nọ là và của có không được bị cho với các những một mọi mỗi người đã "
    "sẽ đang vẫn còn thì mà như khi nếu vì nên hay hoặc cũng rất lắm quá chỉ đều cùng tại từ "
    "đến tới trong ngoài trên dưới sau trước giữa ra vào lên xuống về theo bởi do để nào ai gì "
    "sao đâu bao nhiêu ấy họ ta tôi chúng nó thế vậy nhé ạ ơi "
    "à ừ nhỉ chứ nhưng tuy song dù hễ thật").split()}
MAX_ORIG_COUNT = 60   # never "correct" a very common syllable
MIN_RATIO = 3.0       # the variant must be this much more common than the original


class Lexicon:
    def __init__(self):
        self.phrases: set[str] = set()
        self.domain: set[str] = set()
        d = app_root() / "models" / "dict"
        self.places: set[str] = set()
        p = d / "places.txt"
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line and not line.startswith("#"):
                    name = " ".join(canonical(s) for s in line.split())
                    self.places.add(name)
                    if " - " in name:  # also as printed without spaces: "Bà Rịa-Vũng Tàu"
                        self.places.add(name.replace(" - ", "-"))
        self.abbreviations: set[str] = set()
        p = d / "abbreviations.txt"
        if p.exists():
            self.abbreviations = {unicodedata.normalize("NFC", line.strip()) for line in
                                  p.read_text(encoding="utf-8").splitlines()
                                  if line.strip() and not line.startswith("#")}
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
        for ph in self.places:  # place syllables are valid forms, but must not skew frequencies
            for syl in ph.split():
                if syl.isalpha():
                    self.index.setdefault(skeleton(syl), set()).add(syl)
        # skeleton prefix -> longer skeletons (up to 2 extra letters), for truncated syllables
        self.prefixes: dict[str, list[str]] = {}
        self.suffixes: dict[str, list[str]] = {}  # skeleton minus its first letter -> skeletons
        for key in self.index:
            for cut in (1, 2):
                if len(key) - cut >= 2:
                    self.prefixes.setdefault(key[:-cut], []).append(key)
            if len(key) >= 3:
                self.suffixes.setdefault(key[1:], []).append(key)

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
        if not core.isalpha() or len(core) > 7 or core in _SURNAMES or core in _PROTECTED \
                or evidence(i, core)[0]:
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
            # or the first one ("hữa" -> "chữa")
            for key in lex.prefixes.get(skel, []) + lex.suffixes.get(skel, []):
                out |= lex.index[key]
        return out - {low[i]}

    out = list(tokens)

    def replace(i: int, word: str):
        pre, orig, post = parts[i]
        out[i] = pre + _match_case(orig, word) + post
        low[i] = word

    # Two rounds: fixing one syllable can supply the context another one needs ("SƠ Y TẺ").
    for _ in range(2):
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
            continue  # all-caps pairs are mostly names; too risky to rewrite both
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


_CONFUSIONS = [("I.", "L"), ("l.", "L"), ("1.", "L"), ("|.", "L"), ("I", "L"), ("l", "L"), ("0", "O"),
               ("rn", "m"), ("cl", "d"), ("ii", "ü")]


def is_syllable(word: str) -> bool:
    return _is_syllable(word)


def _is_syllable(word: str) -> bool:
    lex = lexicon()
    return lex.count.get(canonical(word), 0) > 0 or canonical(word) in lex.index.get(skeleton(word), ())


def fix_confusions(tokens: list[str]) -> list[str]:
    """Character shapes OCR mixes up inside a word ("I.ong" -> "Long"): applied only when the
    token is not a syllable and exactly one substitution makes it a real one."""
    out = list(tokens)
    for i, t in enumerate(tokens):
        pre, core, post = _SPLIT.match(t).groups()
        if not core or (core.isalpha() and _is_syllable(core)) or any(ch.isdigit() for ch in core[1:]):
            continue
        fixes = set()
        for a, b in _CONFUSIONS:
            if a in core:
                cand = core.replace(a, b, 1)
                if cand.isalpha() and _is_syllable(cand):
                    fixes.add(cand)
        if len(fixes) == 1:
            out[i] = pre + fixes.pop() + post
    return out


_ADMIN_SKELETONS = {"tinh", "huyen", "xa", "phuong", "quan", "thi"}
_SURNAME_BY_SKELETON: dict[str, list[str]] = {}
for _s in _SURNAMES:
    _SURNAME_BY_SKELETON.setdefault(skeleton(_s), []).append(_s)


def fix_surnames(tokens: list[str]) -> list[str]:
    """A surname that lost its accents ("Pham Minh An", "Nguyen Van A"): only plain unaccented
    capitalised words followed by another capitalised word, matching exactly one surname."""
    out = list(tokens)
    for i in range(len(tokens) - 1):
        pre, core, post = _SPLIT.match(tokens[i]).groups()
        nxt = _SPLIT.match(tokens[i + 1]).group(2)
        if not (core[:1].isupper() and core.isascii() and core.isalpha() and nxt[:1].isupper()):
            continue
        names = _SURNAME_BY_SKELETON.get(skeleton(core), [])
        if len(names) == 1 and names[0] != core.lower():
            out[i] = pre + _match_case(core, names[0]) + post
    return out


def fix_place_names(tokens: list[str]) -> list[str]:
    """Capitalised runs that are a place name except for one accent ("Đất Đó" -> "Đất Đỏ",
    "Phước Long Thọ"). Only proper-noun capitalisation counts, so ordinary phrases such as
    "mảnh đất đó" are never touched."""
    lex = lexicon()
    if not lex.places:
        return tokens
    out = list(tokens)
    parts = [_SPLIT.match(t).groups() for t in tokens]
    # "Ấp" (hamlet) opening an address is below the place list's level; "Áp"/"Ap" before a
    # capitalised name is always it
    for i in range(len(tokens) - 1):
        core = parts[i][1]
        nxt = parts[i + 1][1]
        if core in ("Áp", "ÁP", "Ap", "AP", "Âp") and nxt[:1].isupper() and nxt.isalpha():
            out[i] = parts[i][0] + ("ẤP" if core.isupper() and len(core) > 1 else "Ấp") + parts[i][2]

    def capital(i):
        core = parts[i][1]
        return core[:1].isupper() and core.replace("-", "").isalpha()

    def admin_word(i):  # "tỉnh", "huyện"… may open a place name in lower case (or be misread)
        return parts[i][1].isalpha() and skeleton(parts[i][1]) in _ADMIN_SKELETONS

    for size in (4, 3, 2):
        for s in range(len(tokens) - size + 1):
            idx = list(range(s, s + size))
            if not all(capital(i) or (k == 0 and admin_word(i)) for k, i in enumerate(idx)):
                continue
            cores = [canonical(_SPLIT.match(out[i]).group(2)) for i in idx]
            if " ".join(cores) in lex.places:
                continue
            # change exactly one syllable to a same-skeleton variant that completes a place name
            hits = []
            for j, i in enumerate(idx):
                for v in lex.index.get(skeleton(cores[j]), ()):
                    if v != cores[j] and " ".join(cores[:j] + [v] + cores[j + 1:]) in lex.places:
                        hits.append((i, v))
            if len(hits) == 1:
                i, v = hits[0]
                pre, orig, post = _SPLIT.match(out[i]).groups()
                out[i] = pre + _match_case(orig, v) + post
    return out


def _edit1(a: str, b: str) -> bool:
    """True when a and b differ by exactly one insertion, deletion or substitution."""
    if a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    s, l = (a, b) if len(a) < len(b) else (b, a)
    return any(l[:k] + l[k + 1:] == s for k in range(len(l)))


def fix_abbreviations(tokens: list[str]) -> list[str]:
    """Upper-case codes read one letter off ("CCHIN" -> "CCHN", "004835/RRVT" -> ".../BRVT")."""
    lex = lexicon()
    if not lex.abbreviations:
        return tokens
    out = []
    for t in tokens:
        pieces = re.split(r"([/\-–.,;:()])", t)
        fixed = []
        for p in pieces:
            if len(p) >= 4 and p.isalpha() and p.isupper() and p not in lex.abbreviations:
                near = [a for a in lex.abbreviations if len(a) >= 4 and _edit1(p, a)]
                if len(near) == 1:
                    p = near[0]
            fixed.append(p)
        out.append("".join(fixed))
    return out


def repair_split(tokens: list[str]) -> list[tuple[int, str]]:
    """A syllable OCR broke in two *and* padded with a stray stroke ("môn:" read "môi n:"):
    returns (i, word) where tokens[i] + tokens[i+1] should become `word` (punctuation of the
    second token kept). Only when the second half is no syllable, and the repaired word is the
    one form that fits a known phrase with the previous word."""
    lex = lexicon()
    out = []
    for i in range(1, len(tokens) - 1):
        _, a, _ = _SPLIT.match(tokens[i]).groups()
        _, b, post = _SPLIT.match(tokens[i + 1]).groups()
        lone_consonant = len(b) == 1 and skeleton(b) not in "aeiouy"
        if not (a.isalpha() and b.isalpha() and len(b) <= 2 and (lone_consonant or not _is_syllable(b))):
            continue
        prev = canonical(_SPLIT.match(tokens[i - 1]).group(2))
        joined = a + b
        cands = set()
        for k in range(len(joined)):
            for f in lex.index.get(skeleton(joined[:k] + joined[k + 1:]), ()):
                if lex.has(prev, f):
                    cands.add(f)
        if len(cands) == 1:
            out.append((i, _match_case(a, cands.pop()) + post))
    return out


def join_split(tokens: list[str]) -> list[int]:
    """Indices i where tokens[i] and tokens[i+1] are halves of one syllable that OCR broke
    apart ("HỌ I" -> "HỘI"): the halves have no support, the joined form fits a known phrase."""
    lex = lexicon()
    if not lex.phrases:
        return []
    cores = [canonical(_SPLIT.match(t).group(2)) for t in tokens]
    joins, i = [], 0
    while i < len(cores) - 1:
        a, b = cores[i], cores[i + 1]
        if a.isalpha() and b.isalpha() and len(a) + len(b) <= 7 and min(len(a), len(b)) <= 2:
            prev = cores[i - 1] if i > 0 else None
            nxt = cores[i + 2] if i + 2 < len(cores) else None

            def fits(x, y):
                return x is not None and y is not None and lex.has(x, y)

            if not (fits(prev, a) or fits(a, b) or fits(b, nxt)):
                forms = lex.index.get(skeleton(a + b), set())
                if any(fits(prev, f) or fits(f, nxt) for f in forms):
                    joins.append(i)
                    i += 2
                    continue
        i += 1
    return joins


_CAMEL = re.compile(r"(?<=[a-zà-ỹđ])(?=[A-ZÀ-ỸĐ])")


def split_merged(tokens: list[str]) -> list[list[str]]:
    """Splits tokens where OCR glued two syllables together ("SƠY" -> "SƠ Y", "MinhAn" ->
    "Minh An"). A token is split only if it is not itself a syllable and exactly one split
    point yields two syllables that form a known phrase with each other or a neighbour."""
    lex = lexicon()
    if not lex.phrases:
        return [[t] for t in tokens]
    cores = [_SPLIT.match(t).groups() for t in tokens]

    def forms(word: str) -> set[str]:
        return lex.index.get(skeleton(word), set()) | {canonical(word)}

    out = []
    for i, (pre, core, post) in enumerate(cores):
        if not core.isalpha() or len(core) < 3 or lex.count.get(canonical(core), 0) > 0 \
                or canonical(core) in _SURNAMES or skeleton(core) in lex.index:
            # a real syllable, or one with only a wrong accent ("hiều"): not glued syllables
            out.append([tokens[i]])
            continue
        camel = [p for p in _CAMEL.split(core) if p]
        if len(camel) == 2 and all(skeleton(p) in lex.index for p in camel):
            out.append([pre + camel[0], camel[1] + post])
            continue
        prev = forms(cores[i - 1][1]) if i > 0 and cores[i - 1][1].isalpha() else set()
        nxt = forms(cores[i + 1][1]) if i + 1 < len(cores) and cores[i + 1][1].isalpha() else set()
        splits = []
        for k in range(1, len(core)):
            a, b = core[:k], core[k:]
            if skeleton(a) not in lex.index or skeleton(b) not in lex.index:
                continue
            va, vb = lex.index[skeleton(a)], lex.index[skeleton(b)]
            if any(lex.has(x, y) for x in va for y in vb) or \
                    any(lex.has(y, z) for y in vb for z in nxt) or \
                    any(lex.has(z, x) for z in prev for x in va):
                splits.append(k)
        if len(splits) == 1:
            k = splits[0]
            out.append([pre + core[:k], core[k:] + post])
        else:
            out.append([tokens[i]])
    return out
