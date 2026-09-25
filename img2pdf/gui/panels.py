from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout,
                             QLabel, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget)

from ..fonts import FAMILIES, fit_font_px, get_fontset
from ..model import Page, TextBlock
from ..render import Geometry
from ..settings import LANGUAGES, PAPERS, Settings

KIND_NAMES = {"paragraph": "Đoạn văn", "heading": "Tiêu đề", "cell": "Ô bảng", "figure": "Hình ảnh"}
ALIGNS = [("left", "Trái"), ("center", "Giữa"), ("right", "Phải"), ("justify", "Đều hai bên")]


class EditorPanel(QWidget):
    """Edits the selected text block (or deletes the selected figure)."""

    changed = pyqtSignal()   # content changed, preview must be refreshed
    deleted = pyqtSignal()   # selection was removed

    def __init__(self, parent=None):
        super().__init__(parent)
        self.page: Page | None = None
        self.key = None
        self.settings: Settings | None = None

        self.title = QLabel("Chọn một vùng trên ảnh hoặc bản xem trước để chỉnh sửa.")
        self.title.setWordWrap(True)
        f = self.title.font()
        f.setBold(True)
        self.title.setFont(f)

        self.text = QPlainTextEdit()
        self.text.setFont(QFont("Times New Roman", 12))
        self.text.setPlaceholderText("Nội dung văn bản…")

        self.size = QDoubleSpinBox()
        self.size.setRange(3, 96)
        self.size.setDecimals(1)
        self.size.setSingleStep(0.5)
        self.size.setSuffix(" pt")
        self.auto_size = QCheckBox("Tự động")
        size_row = QHBoxLayout()
        size_row.addWidget(self.size, 1)
        size_row.addWidget(self.auto_size)
        self.auto_size.toggled.connect(lambda on: self.size.setEnabled(not on))

        self.bold = QCheckBox("In đậm")
        self.align = QComboBox()
        for key, name in ALIGNS:
            self.align.addItem(name, key)
        self.kind = QComboBox()
        for key in ("paragraph", "heading"):
            self.kind.addItem(KIND_NAMES[key], key)

        form = QFormLayout()
        form.addRow("Cỡ chữ:", size_row)
        form.addRow("Căn lề:", self.align)
        form.addRow("Loại:", self.kind)
        form.addRow("", self.bold)

        self.apply_btn = QPushButton("Áp dụng")
        self.apply_btn.setDefault(True)
        self.reset_btn = QPushButton("Khôi phục")
        self.reset_btn.setToolTip("Trả lại văn bản và cỡ chữ đã nhận dạng")
        self.delete_btn = QPushButton("Xoá vùng")
        btns = QHBoxLayout()
        btns.addWidget(self.apply_btn)
        btns.addWidget(self.reset_btn)
        btns.addStretch(1)
        btns.addWidget(self.delete_btn)

        self.form_box = QWidget()
        fl = QVBoxLayout(self.form_box)
        fl.setContentsMargins(0, 0, 0, 0)
        fl.addWidget(self.text, 1)
        fl.addLayout(form)

        lay = QVBoxLayout(self)
        lay.addWidget(self.title)
        lay.addWidget(self.form_box, 1)
        lay.addLayout(btns)

        self.apply_btn.clicked.connect(self.apply)
        self.reset_btn.clicked.connect(self.reset)
        self.delete_btn.clicked.connect(self.delete)
        self.set_target(None, None, None)

    # ------------------------------------------------------------------
    def _block(self) -> TextBlock | None:
        if self.page is None or self.key is None or self.key[0] != "block":
            return None
        return self.page.blocks[self.key[1]] if self.key[1] < len(self.page.blocks) else None

    def _scale(self) -> float:
        return Geometry(self.page, self.settings.paper).s

    def set_target(self, page: Page | None, key, settings: Settings | None):
        self.page, self.key, self.settings = page, key, settings
        has = page is not None and key is not None
        self.delete_btn.setEnabled(has)
        block = self._block()
        self.form_box.setVisible(block is not None)
        self.apply_btn.setVisible(block is not None)
        self.reset_btn.setVisible(block is not None)
        if not has:
            self.title.setText("Chọn một vùng trên ảnh hoặc bản xem trước để chỉnh sửa.")
            return
        if block is None:
            self.title.setText("Hình ảnh (logo, con dấu, chữ ký) — được giữ nguyên dạng ảnh trong PDF.")
            return
        self.title.setText(f"{KIND_NAMES.get(block.kind, block.kind)} #{key[1] + 1}")
        self.text.setPlainText(block.text)
        fs = get_fontset(settings.font)
        self.size.setValue(round(fit_font_px(block, fs) * self._scale() * 2) / 2)
        self.auto_size.setChecked(block.font_px_override is None)
        self.bold.setChecked(block.bold)
        self.align.setCurrentIndex(max(0, self.align.findData(block.align)))
        self.kind.setEnabled(block.kind != "cell")
        self.kind.setCurrentIndex(max(0, self.kind.findData(block.kind if block.kind != "cell" else "paragraph")))

    def apply(self):
        block = self._block()
        if block is None:
            return
        text = self.text.toPlainText().rstrip("\n")
        align = self.align.currentData()
        if text != block.original_text or align != block.align:
            block.override_text = text  # edited text is re-flowed inside the original box
        block.align = align
        block.font_px_override = None if self.auto_size.isChecked() else self.size.value() / self._scale()
        if self.bold.isChecked() != block.bold:
            block.set_bold(self.bold.isChecked())
        if block.kind != "cell":
            block.kind = self.kind.currentData()
        self.changed.emit()
        self.set_target(self.page, self.key, self.settings)

    def reset(self):
        block = self._block()
        if block is None:
            return
        block.override_text = None
        block.font_px_override = None
        self.changed.emit()
        self.set_target(self.page, self.key, self.settings)

    def delete(self):
        if self.page is None or self.key is None:
            return
        kind, idx = self.key
        if kind == "block":
            del self.page.blocks[idx]
        else:
            del self.page.figures[idx]
        self.set_target(None, None, None)
        self.deleted.emit()


class SettingsPanel(QWidget):
    ocrChanged = pyqtSignal()     # requires re-recognition
    renderChanged = pyqtSignal()  # only the PDF rendering changes

    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self.settings = settings

        self.lang = QComboBox()
        for name, code in LANGUAGES.items():
            self.lang.addItem(name, code)
        self.font = QComboBox()
        self.font.addItems(list(FAMILIES))
        self.paper = QComboBox()
        self.paper.addItems(PAPERS)
        self.auto_crop = QCheckBox("Tự cắt mép và nắn phẳng ảnh chụp")
        self.tables = QCheckBox("Nhận dạng bảng có đường kẻ")
        self.figures = QCheckBox("Giữ hình ảnh (logo, con dấu, chữ ký)")
        self.diacritics = QCheckBox("Sửa dấu tiếng Việt theo từ điển")

        ocr_box = QGroupBox("Nhận dạng (OCR)")
        f1 = QFormLayout(ocr_box)
        f1.addRow("Ngôn ngữ:", self.lang)
        f1.addRow(self.auto_crop)
        f1.addRow(self.tables)
        f1.addRow(self.figures)
        f1.addRow(self.diacritics)
        pdf_box = QGroupBox("Xuất PDF")
        f2 = QFormLayout(pdf_box)
        f2.addRow("Phông chữ:", self.font)
        f2.addRow("Khổ giấy:", self.paper)

        lay = QVBoxLayout(self)
        lay.addWidget(ocr_box)
        lay.addWidget(pdf_box)
        lay.addStretch(1)
        self.load()

        for w in (self.lang,):
            w.currentIndexChanged.connect(self._ocr)
        for w in (self.auto_crop, self.tables, self.figures, self.diacritics):
            w.toggled.connect(self._ocr)
        self.font.currentIndexChanged.connect(self._render)
        self.paper.currentIndexChanged.connect(self._render)

    def load(self):
        s = self.settings
        self.lang.setCurrentIndex(max(0, self.lang.findData(s.lang)))
        self.font.setCurrentIndex(max(0, self.font.findText(s.font)))
        self.paper.setCurrentIndex(max(0, self.paper.findText(s.paper)))
        self.auto_crop.setChecked(s.auto_crop)
        self.tables.setChecked(s.detect_tables)
        self.figures.setChecked(s.keep_figures)
        self.diacritics.setChecked(s.fix_diacritics)

    def _ocr(self):
        s = self.settings
        s.lang = self.lang.currentData()
        s.auto_crop = self.auto_crop.isChecked()
        s.detect_tables = self.tables.isChecked()
        s.keep_figures = self.figures.isChecked()
        s.fix_diacritics = self.diacritics.isChecked()
        self.ocrChanged.emit()

    def _render(self):
        self.settings.font = self.font.currentText()
        self.settings.paper = self.paper.currentText()
        self.renderChanged.emit()
