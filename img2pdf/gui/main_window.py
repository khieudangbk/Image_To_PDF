import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from PyQt6.QtCore import QObject, QPointF, QSettings, QSize, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QIcon, QImage, QKeySequence, QPixmap, QTransform
from PyQt6.QtWidgets import (QAbstractItemView, QApplication, QDockWidget, QFileDialog, QLabel, QListWidget,
                             QListWidgetItem, QMainWindow, QMessageBox, QProgressBar, QSplitter, QStyle,
                             QTabWidget, QToolBar, QVBoxLayout, QWidget)

from .. import APP_NAME, __version__
from ..model import Page
from ..pipeline import IMAGE_EXTS, check_language, collect_images, default_workers, process_image
from ..render import Geometry, export_pdf, render_preview
from ..settings import Settings
from .panels import EditorPanel, SettingsPanel
from .views import PreviewView, SourceView, load_qimage

PREVIEW_ZOOM = 2.0
CANCELLED = "Đã huỷ"


class Entry:
    def __init__(self, path: str):
        self.path = path
        self.rotation = 0
        self.page: Page | None = None
        self.error: str | None = None
        self.ocr_key = None
        self.busy = False
        self.thumb = load_qimage(path, 200)

    @property
    def name(self) -> str:
        return Path(self.path).name

    def stale(self, settings: Settings) -> bool:
        return self.page is None or self.ocr_key != (settings.ocr_key(), self.rotation)


class RecognizeWorker(QObject):
    pageDone = pyqtSignal(object, object, str)
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(bool)

    def __init__(self, entries: list[Entry], settings: Settings):
        super().__init__()
        self.entries = entries
        self.settings = Settings(**asdict(settings))
        self._cancel = threading.Event()

    def cancel(self):
        self._cancel.set()

    def run(self):
        total, done = len(self.entries), 0
        self.progress.emit(0, total)
        with ThreadPoolExecutor(max_workers=default_workers()) as ex:
            futures = {ex.submit(self._one, e): e for e in self.entries}
            for fut in as_completed(futures):
                e = futures[fut]
                page, err = fut.result()
                done += 1
                self.pageDone.emit(e, page, err or "")
                self.progress.emit(done, total)
        self.finished.emit(self._cancel.is_set())

    def _one(self, e: Entry):
        if self._cancel.is_set():
            return None, CANCELLED
        try:
            return process_image(e.path, self.settings, e.rotation), None
        except Exception as ex:  # reported per page in the UI
            return None, str(ex)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} {__version__} — Chuyển ảnh thành PDF văn bản")
        self.resize(1500, 950)
        self.setAcceptDrops(True)
        self.qs = QSettings("ImageToPDFVN", "ImageToPDFVN")
        self.settings = self._load_settings()
        self.thread: QThread | None = None
        self.worker: RecognizeWorker | None = None
        self.export_after: str | None = None
        self.selected_key = None

        # page list
        self.pages = QListWidget()
        self.pages.setIconSize(QSize(72, 100))
        self.pages.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.pages.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.pages.setSpacing(2)
        self.pages.currentItemChanged.connect(lambda *_: self.show_current())
        self.pages.model().rowsMoved.connect(lambda *_: self.refresh_labels())
        left = QDockWidget("Trang", self)
        left.setObjectName("pages")
        left.setWidget(self.pages)
        left.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, left)

        # centre: source + preview
        self.source = SourceView()
        self.preview = PreviewView()
        self.source.elementClicked.connect(self.select_element)
        self.preview.pointClicked.connect(self.preview_clicked)
        split = QSplitter()
        split.addWidget(self._titled("Ảnh gốc — vùng nhận dạng (bấm để chọn)", self.source))
        split.addWidget(self._titled("Bản PDF xem trước", self.preview))
        split.setSizes([700, 700])
        self.setCentralWidget(split)

        # right: editor + settings
        self.editor = EditorPanel()
        self.editor.changed.connect(self.content_changed)
        self.editor.deleted.connect(self.content_changed)
        self.settings_panel = SettingsPanel(self.settings)
        self.settings_panel.ocrChanged.connect(self.ocr_settings_changed)
        self.settings_panel.renderChanged.connect(self.render_settings_changed)
        self.tabs = tabs = QTabWidget()
        tabs.addTab(self.editor, "Chỉnh sửa")
        tabs.addTab(self.settings_panel, "Cài đặt")
        right = QDockWidget("Thuộc tính", self)
        right.setObjectName("props")
        right.setWidget(tabs)
        right.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable)
        right.setMinimumWidth(340)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, right)

        # status
        self.status_label = QLabel()
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(260)
        self.progress.setVisible(False)
        self.statusBar().addWidget(self.status_label, 1)
        self.statusBar().addPermanentWidget(self.progress)

        self.preview_timer = QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.setInterval(250)
        self.preview_timer.timeout.connect(self.update_preview)
        self._build_actions()
        self.update_actions()
        self.show_current()
        self.status("Kéo thả ảnh vào cửa sổ hoặc bấm “Thêm ảnh” để bắt đầu.")

    # ------------------------------------------------------------------ setup
    def _titled(self, title: str, w: QWidget) -> QWidget:
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(2, 2, 2, 2)
        lab = QLabel(title)
        lab.setStyleSheet("font-weight:600; padding:3px;")
        lay.addWidget(lab)
        lay.addWidget(w, 1)
        return box

    def _act(self, text, slot, shortcut=None, icon=None, tip=None) -> QAction:
        a = QAction(text, self)
        if icon is not None:
            a.setIcon(self.style().standardIcon(icon))
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        if tip:
            a.setToolTip(tip)
            a.setStatusTip(tip)
        a.triggered.connect(slot)
        return a

    def _build_actions(self):
        SP = QStyle.StandardPixmap
        self.a_add = self._act("Thêm ảnh", self.add_files, "Ctrl+O", SP.SP_DialogOpenButton, "Thêm ảnh tài liệu")
        self.a_add_dir = self._act("Thêm thư mục", self.add_folder, "Ctrl+Shift+O", SP.SP_DirOpenIcon)
        self.a_remove = self._act("Xoá trang", self.remove_pages, "Del", SP.SP_TrashIcon)
        self.a_up = self._act("Lên", lambda: self.move_page(-1), "Ctrl+Up", SP.SP_ArrowUp, "Đưa trang lên")
        self.a_down = self._act("Xuống", lambda: self.move_page(1), "Ctrl+Down", SP.SP_ArrowDown, "Đưa trang xuống")
        self.a_rot = self._act("Xoay 90°", self.rotate_pages, "Ctrl+R", SP.SP_BrowserReload,
                               "Xoay ảnh gốc 90° theo chiều kim đồng hồ (khi tự nhận hướng bị sai)")
        self.a_ocr = self._act("Nhận dạng", self.recognize_needed, "F5", SP.SP_MediaPlay,
                               "Nhận dạng các trang chưa xử lý")
        self.a_ocr_sel = self._act("Nhận dạng lại trang chọn", self.recognize_selected, "Shift+F5")
        self.a_stop = self._act("Dừng", self.stop, "Esc", SP.SP_MediaStop)
        self.a_export = self._act("Xuất PDF", self.export, "Ctrl+E", SP.SP_DialogSaveButton,
                                  "Xuất toàn bộ trang ra một file PDF")
        self.a_txt = self._act("Xuất văn bản (.txt)", self.export_txt, "Ctrl+Shift+E")
        self.a_regions = self._act("Hiện khung vùng", self.toggle_regions, "Ctrl+H")
        self.a_regions.setCheckable(True)
        self.a_regions.setChecked(True)
        self.a_zin = self._act("Phóng to", lambda: self._zoom(1.25), "Ctrl++")
        self.a_zout = self._act("Thu nhỏ", lambda: self._zoom(0.8), "Ctrl+-")
        self.a_zfit = self._act("Vừa khung", self._zoom_fit, "Ctrl+0")

        tb = QToolBar("Công cụ")
        tb.setObjectName("tools")
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        tb.setIconSize(QSize(20, 20))
        for a in (self.a_add, self.a_add_dir, None, self.a_up, self.a_down, self.a_rot, self.a_remove, None,
                  self.a_ocr, self.a_stop, None, self.a_export):
            tb.addSeparator() if a is None else tb.addAction(a)
        self.addToolBar(tb)

        m = self.menuBar().addMenu("&Tệp")
        for a in (self.a_add, self.a_add_dir, None, self.a_export, self.a_txt, None):
            m.addSeparator() if a is None else m.addAction(a)
        m.addAction(self._act("Thoát", self.close, "Ctrl+Q"))
        m = self.menuBar().addMenu("T&rang")
        for a in (self.a_up, self.a_down, self.a_rot, self.a_remove, None, self.a_ocr, self.a_ocr_sel, self.a_stop):
            m.addSeparator() if a is None else m.addAction(a)
        m = self.menuBar().addMenu("&Xem")
        for a in (self.a_regions, None, self.a_zin, self.a_zout, self.a_zfit):
            m.addSeparator() if a is None else m.addAction(a)
        m = self.menuBar().addMenu("Trợ &giúp")
        m.addAction(self._act("Hướng dẫn nhanh", self.show_help, "F1"))
        m.addAction(self._act("Giới thiệu", self.show_about))

    # ------------------------------------------------------------------ settings persistence
    def _load_settings(self) -> Settings:
        s = Settings()
        for field, default in asdict(s).items():
            v = self.qs.value(f"settings/{field}", default)
            if isinstance(default, bool):
                v = v in (True, "true", "1", 1)
            setattr(s, field, type(default)(v))
        return s

    def _save_settings(self):
        for field, v in asdict(self.settings).items():
            self.qs.setValue(f"settings/{field}", v)

    # ------------------------------------------------------------------ helpers
    def status(self, text: str):
        self.status_label.setText(text)

    def entries(self) -> list[Entry]:
        return [self.pages.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.pages.count())]

    def current(self) -> Entry | None:
        item = self.pages.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def selected_entries(self) -> list[Entry]:
        items = sorted(self.pages.selectedItems(), key=self.pages.row)
        return [i.data(Qt.ItemDataRole.UserRole) for i in items] or ([self.current()] if self.current() else [])

    def busy(self) -> bool:
        return self.thread is not None

    def refresh_labels(self):
        for i in range(self.pages.count()):
            item = self.pages.item(i)
            e: Entry = item.data(Qt.ItemDataRole.UserRole)
            if e.busy:
                state = "đang xử lý…"
            elif e.error:
                state = "lỗi"
            elif e.page is None:
                state = "chưa nhận dạng"
            elif e.stale(self.settings):
                state = "cần nhận dạng lại"
            else:
                state = f"xong · {len(e.page.blocks)} khối"
            item.setText(f"Trang {i + 1}\n{e.name}\n{state}")
            item.setToolTip(e.error or e.path)
            thumb = e.thumb
            if e.rotation and not thumb.isNull():
                thumb = thumb.transformed(QTransform().rotate(e.rotation))
            item.setIcon(QIcon(QPixmap.fromImage(thumb)))
        self.update_actions()

    def update_actions(self):
        has = self.pages.count() > 0
        busy = self.busy()
        for a in (self.a_remove, self.a_up, self.a_down, self.a_rot):
            a.setEnabled(has and not busy)
        self.a_ocr.setEnabled(has and not busy)
        self.a_ocr_sel.setEnabled(has and not busy)
        self.a_export.setEnabled(has and not busy)
        self.a_txt.setEnabled(has and not busy)
        self.a_stop.setEnabled(busy)
        self.a_add.setEnabled(not busy)
        self.a_add_dir.setEnabled(not busy)

    # ------------------------------------------------------------------ page management
    def add_paths(self, paths: list[str]):
        paths = collect_images(paths)
        if not paths:
            return
        known = {e.path for e in self.entries()}
        for p in paths:
            if p in known:
                continue
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, Entry(p))
            self.pages.addItem(item)
        if self.pages.currentRow() < 0:
            self.pages.setCurrentRow(0)
        self.refresh_labels()
        self.status(f"Đã thêm {len(paths)} ảnh. Bấm “Nhận dạng” (F5) để xử lý.")

    def add_files(self):
        start = self.qs.value("last_dir", str(Path.home()))
        pattern = " ".join(f"*{e}" for e in sorted(IMAGE_EXTS))
        files, _ = QFileDialog.getOpenFileNames(self, "Chọn ảnh tài liệu", start,
                                                f"Ảnh ({pattern});;Tất cả (*.*)")
        if files:
            self.qs.setValue("last_dir", str(Path(files[0]).parent))
            self.add_paths(files)

    def add_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Chọn thư mục ảnh", self.qs.value("last_dir", str(Path.home())))
        if d:
            self.qs.setValue("last_dir", d)
            self.add_paths([d])

    def remove_pages(self):
        for item in self.pages.selectedItems() or ([self.pages.currentItem()] if self.pages.currentItem() else []):
            self.pages.takeItem(self.pages.row(item))
        self.refresh_labels()
        self.show_current()

    def move_page(self, delta: int):
        row = self.pages.currentRow()
        new = row + delta
        if row < 0 or not 0 <= new < self.pages.count():
            return
        item = self.pages.takeItem(row)
        self.pages.insertItem(new, item)
        self.pages.setCurrentRow(new)
        self.refresh_labels()

    def rotate_pages(self):
        for e in self.selected_entries():
            e.rotation = (e.rotation + 90) % 360
        self.refresh_labels()
        self.show_current()
        self.status("Đã xoay. Bấm “Nhận dạng” để xử lý lại trang đã xoay.")

    # ------------------------------------------------------------------ recognition
    def recognize_needed(self, export_to: str | None = None):
        todo = [e for e in self.entries() if e.stale(self.settings) or e.error]
        if not todo:
            self.status("Tất cả các trang đã được nhận dạng.")
            return False
        self.start_recognition(todo, export_to)
        return True

    def recognize_selected(self):
        todo = self.selected_entries()
        if todo:
            self.start_recognition(todo, None)

    def start_recognition(self, todo: list[Entry], export_to: str | None):
        try:
            check_language(self.settings.lang)
        except RuntimeError as ex:
            QMessageBox.critical(self, APP_NAME, str(ex))
            return
        for e in todo:
            e.busy, e.error = True, None
        self.export_after = export_to
        self.thread = QThread(self)
        self.worker = RecognizeWorker(todo, self.settings)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.pageDone.connect(self.page_done)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished.connect(self.recognition_finished)
        self.progress.setVisible(True)
        self.progress.setRange(0, len(todo))
        self.progress.setValue(0)
        self.refresh_labels()
        self.thread.start()

    def on_progress(self, done: int, total: int):
        self.progress.setValue(done)
        self.status(f"Đang nhận dạng… {done}/{total} trang")

    def page_done(self, e: Entry, page, err: str):
        e.busy = False
        if page is not None:
            e.page, e.error = page, None
            e.ocr_key = (self.worker.settings.ocr_key(), e.rotation)
        elif err != CANCELLED:
            e.error = err or "Lỗi không xác định"
        self.refresh_labels()
        if e is self.current():
            self.show_current(keep_zoom=True)

    def recognition_finished(self, cancelled: bool):
        self.thread.quit()
        self.thread.wait()
        self.thread = self.worker = None
        for e in self.entries():
            e.busy = False
        self.progress.setVisible(False)
        self.refresh_labels()
        errors = [e for e in self.entries() if e.error]
        if cancelled:
            self.status("Đã dừng nhận dạng.")
        elif errors:
            self.status(f"Hoàn tất, {len(errors)} trang lỗi.")
            QMessageBox.warning(self, APP_NAME, "Một số trang không xử lý được:\n\n" +
                                "\n".join(f"• {e.name}: {e.error}" for e in errors[:10]))
        else:
            self.status("Nhận dạng xong. Kiểm tra bản xem trước rồi bấm “Xuất PDF”.")
        target, self.export_after = self.export_after, None
        if target and not cancelled:
            self._write_pdf(target)

    def stop(self):
        if self.worker:
            self.worker.cancel()
            self.status("Đang dừng… (chờ các trang đang xử lý xong)")

    # ------------------------------------------------------------------ display
    def show_current(self, keep_zoom=False):
        e = self.current()
        self.selected_key = None
        self.editor.set_target(None, None, None)
        if e is None:
            self.source.set_image(None)
            self.preview.set_image(None)
            return
        if e.page is None:
            img = load_qimage(e.path, 2400)
            if e.rotation and not img.isNull():
                img = img.transformed(QTransform().rotate(e.rotation))
            self.source.show_page(None, img, keep_zoom)
            self.preview.set_image(None)
            if e.error:
                self.status(f"Lỗi: {e.error}")
            return
        self.source.show_page(e.page, None, keep_zoom)
        self.update_preview(keep_zoom=keep_zoom)
        if e.page.notes:
            self.status("; ".join(e.page.notes))

    def update_preview(self, keep_zoom=True):
        e = self.current()
        if e is None or e.page is None:
            return
        try:
            png = render_preview(e.page, self.settings, PREVIEW_ZOOM)
        except Exception as ex:
            self.status(f"Lỗi hiển thị: {ex}")
            return
        self.preview.set_image(QImage.fromData(png), keep_zoom=keep_zoom)

    def select_element(self, key):
        e = self.current()
        if e is None or e.page is None:
            return
        self.selected_key = key
        self.source.highlight(key)
        self.editor.set_target(e.page, key, self.settings)
        self.tabs.setCurrentWidget(self.editor)

    def preview_clicked(self, pos: QPointF):
        e = self.current()
        if e is None or e.page is None:
            return
        g = Geometry(e.page, self.settings.paper)
        px = (pos.x() / PREVIEW_ZOOM - g.ox) / g.s
        py = (pos.y() / PREVIEW_ZOOM - g.oy) / g.s
        hits = []
        for i, b in enumerate(e.page.blocks):
            x0, y0, x1, y1 = b.container or b.bbox
            if x0 - 4 <= px <= x1 + 4 and y0 - 4 <= py <= y1 + 4:
                hits.append(((x1 - x0) * (y1 - y0), ("block", i)))
        for i, f in enumerate(e.page.figures):
            x0, y0, x1, y1 = f.bbox
            if x0 <= px <= x1 and y0 <= py <= y1:
                hits.append(((x1 - x0) * (y1 - y0), ("figure", i)))
        if hits:
            self.select_element(min(hits)[1])

    def content_changed(self):
        e = self.current()
        if e is None or e.page is None:
            return
        self.source.show_page(e.page, None, keep_zoom=True)
        if self.selected_key and self.editor.key:
            self.source.highlight(self.selected_key)
        self.preview_timer.start()
        self.refresh_labels()

    def ocr_settings_changed(self):
        self._save_settings()
        self.refresh_labels()
        if any(e.page is not None for e in self.entries()):
            self.status("Cài đặt nhận dạng đã đổi — bấm “Nhận dạng” (F5) để xử lý lại.")

    def render_settings_changed(self):
        self._save_settings()
        self.preview_timer.start()

    def toggle_regions(self):
        self.source.show_regions = self.a_regions.isChecked()
        self.show_current(keep_zoom=True)

    def _zoom(self, f):
        self.source.zoom(f)
        self.preview.zoom(f)

    def _zoom_fit(self):
        self.source.reset_zoom()
        self.preview.reset_zoom()

    # ------------------------------------------------------------------ export
    def _ask_target(self, suffix: str, filt: str) -> str | None:
        entries = self.entries()
        default = Path(self.qs.value("last_out_dir", str(Path(entries[0].path).parent))) / \
            (Path(entries[0].path).stem + suffix)
        path, _ = QFileDialog.getSaveFileName(self, "Lưu thành", str(default), filt)
        if path:
            self.qs.setValue("last_out_dir", str(Path(path).parent))
        return path or None

    def export(self):
        if not self.entries():
            return
        target = self._ask_target(".pdf", "PDF (*.pdf)")
        if not target:
            return
        if not target.lower().endswith(".pdf"):
            target += ".pdf"
        if any(e.stale(self.settings) or e.error for e in self.entries()):
            self.recognize_needed(export_to=target)
        else:
            self._write_pdf(target)

    def _write_pdf(self, target: str):
        pages = [e.page for e in self.entries() if e.page is not None]
        skipped = self.pages.count() - len(pages)
        if not pages:
            QMessageBox.warning(self, APP_NAME, "Chưa có trang nào được nhận dạng thành công.")
            return
        try:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            export_pdf(pages, target, self.settings, title=Path(target).stem)
        except PermissionError:
            QMessageBox.critical(self, APP_NAME, "Không ghi được file. File có thể đang mở trong trình đọc PDF.")
            return
        except Exception as ex:
            QMessageBox.critical(self, APP_NAME, f"Xuất PDF thất bại:\n{ex}")
            return
        finally:
            QApplication.restoreOverrideCursor()
        msg = f"Đã tạo {Path(target).name} ({len(pages)} trang)."
        if skipped:
            msg += f"\nBỏ qua {skipped} trang lỗi."
        self.status(msg.replace("\n", " "))
        box = QMessageBox(QMessageBox.Icon.Information, APP_NAME, msg, parent=self)
        open_btn = box.addButton("Mở file", QMessageBox.ButtonRole.AcceptRole)
        folder_btn = box.addButton("Mở thư mục", QMessageBox.ButtonRole.ActionRole)
        box.addButton("Đóng", QMessageBox.ButtonRole.RejectRole)
        box.exec()
        if box.clickedButton() is open_btn:
            _open_path(target)
        elif box.clickedButton() is folder_btn:
            _open_path(str(Path(target).parent))

    def export_txt(self):
        if any(e.page is None for e in self.entries()):
            QMessageBox.information(self, APP_NAME, "Hãy nhận dạng tất cả các trang trước (F5).")
            return
        target = self._ask_target(".txt", "Văn bản (*.txt)")
        if not target:
            return
        parts = []
        for i, e in enumerate(self.entries()):
            parts.append("\n\n".join(b.text for b in e.page.blocks))
        Path(target).write_text("\n\n\f".join(parts), encoding="utf-8")
        self.status(f"Đã lưu {Path(target).name}")

    # ------------------------------------------------------------------ misc
    def show_help(self):
        QMessageBox.information(self, "Hướng dẫn nhanh", (
            "1. Kéo thả ảnh (JPG, PNG, TIFF…) hoặc bấm “Thêm ảnh”. Thứ tự trong danh sách là thứ tự trang; "
            "kéo để sắp xếp lại.\n"
            "2. Bấm “Nhận dạng” (F5). Chương trình tự cắt mép, nắn phẳng, chỉnh nghiêng, xoay đúng chiều, "
            "nhận dạng chữ tiếng Việt, bảng, chữ đậm và giữ lại con dấu/chữ ký dạng hình.\n"
            "3. Bấm vào một vùng trên ảnh hoặc bản xem trước để sửa chữ, cỡ chữ, in đậm, căn lề.\n"
            "4. Bấm “Xuất PDF” (Ctrl+E). PDF chứa văn bản thật (chọn, sao chép, tìm kiếm được) với "
            "phông chữ nhúng, như xuất từ Word.\n\n"
            "Mẹo: Ctrl + lăn chuột để phóng to; ảnh chụp càng rõ, thẳng và đủ sáng thì kết quả càng tốt."))

    def show_about(self):
        QMessageBox.about(self, "Giới thiệu", (
            f"<b>{APP_NAME} {__version__}</b><br>Chuyển ảnh tài liệu thành PDF văn bản thật.<br><br>"
            "OCR: Tesseract (tessdata_best, tiếng Việt)<br>Từ điển: Viet74K<br>"
            "PDF: ReportLab · Giao diện: Qt 6"))

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        if self.busy():
            return
        self.add_paths([u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()])

    def closeEvent(self, e):
        if self.busy():
            self.worker.cancel()
            self.thread.quit()
            self.thread.wait(3000)
        self._save_settings()
        super().closeEvent(e)


def _open_path(path: str):
    if sys.platform == "win32":
        os.startfile(path)  # noqa: S606
    else:
        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", path])
