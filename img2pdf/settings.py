from dataclasses import dataclass

LANGUAGES = {
    "Tiếng Việt": "vie",
    "Tiếng Việt + Tiếng Anh": "vie+eng",
    "Tiếng Anh": "eng",
}
PAPER_FIT_IMAGE = "Theo ảnh"
PAPERS = ["A4", "Letter", PAPER_FIT_IMAGE]


@dataclass
class Settings:
    lang: str = "vie"
    font: str = "Times New Roman"
    paper: str = "A4"
    auto_crop: bool = True
    detect_tables: bool = True
    keep_figures: bool = True
    fix_diacritics: bool = True
    keep_background: bool = False  # draw the original page (text erased) behind the real text

    def ocr_key(self) -> tuple:
        return (self.lang, self.auto_crop, self.detect_tables, self.keep_figures, self.fix_diacritics)
