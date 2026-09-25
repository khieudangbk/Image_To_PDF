import sys
import traceback

from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication, QMessageBox

from img2pdf import APP_NAME
from img2pdf.gui.main_window import MainWindow
from img2pdf.paths import app_root


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        import cli
        return cli.main(sys.argv[2:])
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("ImageToPDFVN")
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setStyle("Fusion")
    icon = app_root() / "assets" / "app.ico"
    if icon.exists():
        app.setWindowIcon(QIcon(str(icon)))

    def excepthook(etype, value, tb):
        text = "".join(traceback.format_exception(etype, value, tb))
        sys.__stderr__ and sys.__stderr__.write(text)
        QMessageBox.critical(None, APP_NAME, "Đã xảy ra lỗi không mong muốn:\n\n" + text[-3000:])

    sys.excepthook = excepthook
    w = MainWindow()
    w.show()
    if len(sys.argv) > 1:
        w.add_paths(sys.argv[1:])
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
