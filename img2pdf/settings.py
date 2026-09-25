from dataclasses import dataclass

LANGUAGES = {
    "Tiếng Việt": "vie",
    "Tiếng Việt + Tiếng Anh": "vie+eng",
    "Tiếng Anh": "eng",
}
BACKGROUNDS = {"Tự động (giữ nền nếu có hoa văn/màu)": "auto", "Giữ nền và màu gốc": "on",
               "Nền trắng": "off"}
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
    # Original page (text erased) behind the real text: "auto" = only for decorated pages
    # (certificates, coloured forms), "on" = always, "off" = clean white page.
    background: str = "auto"

    def ocr_key(self) -> tuple:
        return (self.lang, self.auto_crop, self.detect_tables, self.keep_figures, self.fix_diacritics)
