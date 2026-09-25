"""End-to-end GUI test: add pages, recognise, edit a block, export, and screenshot the window."""
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pymupdf  # noqa: E402
from PyQt6.QtCore import QEventLoop, QTimer  # noqa: E402
from PyQt6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from img2pdf.gui.main_window import MainWindow  # noqa: E402

SAMPLES = ROOT / "tests" / "samples"
OUT = ROOT / "tests" / "out"


def wait_until(cond, timeout_ms=300_000):
    loop = QEventLoop()
    timer = QTimer()
    timer.timeout.connect(lambda: cond() and loop.quit())
    timer.start(100)
    QTimer.singleShot(timeout_ms, loop.quit)
    loop.exec()
    timer.stop()
    return cond()


def main() -> int:
    OUT.mkdir(exist_ok=True)
    app = QApplication(sys.argv)
    QMessageBox.exec = lambda self: 0  # never block on dialogs
    QMessageBox.warning = QMessageBox.information = QMessageBox.critical = staticmethod(lambda *a, **k: 0)
    w = MainWindow()
    w.resize(1600, 1000)
    w.show()
    w.add_paths([str(SAMPLES / "03_photo.jpg"), str(SAMPLES / "02_scan.jpg")])
    assert w.pages.count() == 2

    w.recognize_needed()
    assert wait_until(lambda: not w.busy()), "recognition did not finish"
    entries = w.entries()
    assert all(e.page is not None for e in entries), [e.error for e in entries]
    print("recognised:", [len(e.page.blocks) for e in entries])

    # Select the "QUYẾT ĐỊNH" heading via the preview and edit it.
    w.pages.setCurrentRow(0)
    page = entries[0].page
    idx = next(i for i, b in enumerate(page.blocks) if "QUYẾT" in b.text)
    w.select_element(("block", idx))
    assert w.editor.text.toPlainText().startswith("QUYẾT")
    w.editor.text.setPlainText("QUYẾT ĐỊNH (ĐÃ SỬA)")
    w.editor.apply()
    app.processEvents()
    wait_until(lambda: not w.preview_timer.isActive(), 3000)
    app.processEvents()

    # Reorder pages, export, and verify the PDF.
    w.pages.setCurrentRow(1)
    w.move_page(-1)
    target = OUT / "gui_export.pdf"
    w._write_pdf(str(target))
    with pymupdf.open(target) as doc:
        texts = [" ".join(p.get_text().split()) for p in doc]
        toc = doc.get_toc()
    assert len(texts) == 2
    assert "QUYẾT ĐỊNH (ĐÃ SỬA)" in texts[1], texts[1][:300]
    assert "Nghị định số" in texts[0]
    print("export OK, toc:", toc)

    w.pages.setCurrentRow(1)
    app.processEvents()
    w.select_element(("block", idx))
    app.processEvents()
    w.grab().save(str(OUT / "gui_screenshot.png"))
    print("PASS")
    w.close()
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
