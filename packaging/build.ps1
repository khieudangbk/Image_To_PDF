# Builds dist\ImageToPDF\ImageToPDF.exe (self-contained: Python, Qt, Tesseract, Vietnamese models).
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

python -m pip install -r requirements.txt pyinstaller
python packaging\collect_tesseract.py
python -m PyInstaller --noconfirm --clean --distpath dist --workpath build\pyinstaller packaging\ImageToPDF.spec
python -c "import shutil; shutil.make_archive('dist/ImageToPDF-win64', 'zip', 'dist', 'ImageToPDF')"
Write-Host "Done: dist\ImageToPDF\ImageToPDF.exe and dist\ImageToPDF-win64.zip"
