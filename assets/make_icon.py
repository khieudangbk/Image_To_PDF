"""Draws the application icon (assets/app.ico)."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

S = 256
img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
d = ImageDraw.Draw(img)
d.rounded_rectangle([40, 14, 216, 242], 18, fill=(255, 255, 255), outline=(60, 70, 90), width=8)
d.polygon([(166, 14), (216, 64), (166, 64)], fill=(200, 208, 220))
for i, y in enumerate(range(70, 150, 22)):
    d.rounded_rectangle([66, y, 190 - (40 if i == 3 else 0), y + 9], 4, fill=(120, 130, 150))
d.rounded_rectangle([28, 160, 228, 232], 14, fill=(214, 40, 48))
font = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 58)
w = d.textlength("PDF", font=font)
d.text(((S - w) / 2, 163), "PDF", font=font, fill="white")
out = Path(__file__).parent / "app.ico"
img.save(out, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
img.save(out.with_suffix(".png"))
print("wrote", out)
