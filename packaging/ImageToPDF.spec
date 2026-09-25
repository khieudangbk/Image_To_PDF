# PyInstaller spec — build with packaging\build.ps1
from pathlib import Path

ROOT = Path(SPECPATH).parent

datas = [
    (str(ROOT / "models" / "tessdata"), "models/tessdata"),
    (str(ROOT / "models" / "dict"), "models/dict"),
    (str(ROOT / "assets" / "app.ico"), "assets"),
]

a = Analysis(
    [str(ROOT / "app.py")],
    pathex=[str(ROOT)],
    datas=datas,
    hiddenimports=["cli"],
    excludes=["tkinter", "matplotlib", "scipy", "pandas", "IPython", "PyQt6.QtWebEngineCore",
              "PyQt6.QtQml", "PyQt6.QtQuick", "PyQt6.QtMultimedia", "PyQt6.Qt3DCore", "PyQt6.QtBluetooth"],
    noarchive=False,
)

# Not needed at runtime: video I/O, software OpenGL, AVIF codec.
UNUSED = ("opencv_videoio_ffmpeg", "opengl32sw", "_avif", "Qt6Pdf", "Qt6Quick", "Qt6Qml")
a.binaries = [b for b in a.binaries if not any(u.lower() in b[0].lower() for u in UNUSED)]

pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="ImageToPDF",
    icon=str(ROOT / "assets" / "app.ico"),
    console=False,
)
# Tesseract is added as a plain tree so PyInstaller does not duplicate its DLLs at the top level.
tesseract = Tree(str(ROOT / "build" / "tesseract"), prefix="tesseract")
coll = COLLECT(exe, a.binaries, a.datas, tesseract, name="ImageToPDF")
