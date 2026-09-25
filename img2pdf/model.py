from dataclasses import dataclass, field

import numpy as np

BBox = tuple[int, int, int, int]  # x0, y0, x1, y1 in processed-image pixels


def union(boxes) -> BBox:
    boxes = list(boxes)
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def center_in(box: BBox, container: BBox) -> bool:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return container[0] <= cx <= container[2] and container[1] <= cy <= container[3]


@dataclass
class Word:
    text: str
    bbox: BBox
    conf: float
    bold: bool = False


@dataclass
class Line:
    words: list[Word]
    baseline: float
    x_size: float

    @property
    def bbox(self) -> BBox:
        return union(w.bbox for w in self.words)

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)


@dataclass
class TextBlock:
    lines: list[Line]
    kind: str = "paragraph"          # paragraph | heading | cell
    align: str = "left"              # left | center | right | justify
    heading_level: int = 0
    container: BBox | None = None    # cell box for table cells
    override_text: str | None = None
    font_px_override: float | None = None
    color: tuple[float, float, float] | None = None  # RGB 0..1; None = black

    @property
    def bbox(self) -> BBox:
        if not self.lines:
            return self.container or (0, 0, 0, 0)
        return union(ln.bbox for ln in self.lines)

    @property
    def original_text(self) -> str:
        return "\n".join(ln.text for ln in self.lines)

    @property
    def text(self) -> str:
        return self.override_text if self.override_text is not None else self.original_text

    @property
    def bold(self) -> bool:
        words = [w for ln in self.lines for w in ln.words]
        if not words:
            return False
        n_bold = sum(len(w.text) for w in words if w.bold)
        return n_bold >= 0.6 * sum(len(w.text) for w in words)

    def set_bold(self, value: bool):
        for ln in self.lines:
            for w in ln.words:
                w.bold = value


@dataclass
class Rule:
    x0: float
    y0: float
    x1: float
    y1: float
    thickness: float


@dataclass
class Figure:
    bbox: BBox
    image: np.ndarray  # BGR crop


@dataclass
class Page:
    source: str
    image: np.ndarray            # processed (cropped, deskewed, normalized) BGR image
    blocks: list[TextBlock] = field(default_factory=list)
    rules: list[Rule] = field(default_factory=list)
    figures: list[Figure] = field(default_factory=list)
    tables: list[BBox] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    raw: np.ndarray | None = None        # original colours, same geometry as `image`
    text_mask: np.ndarray | None = None  # pixels of recognised text (to erase from the background)
    background: np.ndarray | None = None  # cached `raw` with the text erased
    decorative: bool = False             # coloured/patterned paper worth keeping as background

    @property
    def width(self) -> int:
        return self.image.shape[1]

    @property
    def height(self) -> int:
        return self.image.shape[0]
