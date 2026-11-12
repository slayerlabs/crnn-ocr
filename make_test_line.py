"""Deterministyczna linia testowa z polskimi znakami -> PNG (tryb "L").

Uruchomienie:
    python make_test_line.py                            # pangram -> test_line.png
    python make_test_line.py --text "Ala ma kota." --out linia.png

Następnie: python predict.py --image test_line.png
"""
import argparse
import json
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.stdout.reconfigure(encoding="utf-8")    # polskie znaki w konsoli Windows

DEFAULT_TEXT = "Zażółć gęślą jaźń — próbka testowa OCR"
def font_candidates():
    """Kandydaci na TTF z pełnymi polskimi glifami, wg systemu.

    Windows: katalog fontów liczony z WINDIR (litera dysku bywa różna,
    nie zakładamy C:/ na sztywno); fonty instalowane per-user (Win 10/11)
    lądują w LOCALAPPDATA. Linux: typowe ścieżki DejaVuSans.
    """
    windir = os.environ.get("WINDIR")
    if windir:
        yield Path(windir) / "Fonts" / "arial.ttf"
        local = os.environ.get("LOCALAPPDATA")
        if local:
            yield Path(local) / "Microsoft" / "Windows" / "Fonts" / "arial.ttf"
    yield "arial.ttf"                                     # nazwa: CWD/PIL
    yield "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"     # Debian/Ubuntu
    yield "/usr/share/fonts/TTF/DejaVuSans.ttf"                # Arch


MARGIN_X, CANVAS_H, OFFSET_Y = 20, 64, 18   # stałe — deterministyczny render


def find_font():
    for path in font_candidates():
        if Path(path).is_file():
            return path
    sys.exit("[ERROR] Brak fontu TTF z polskimi glifami — podaj --font ścieżkę")


def check_charset(text):
    """Ostrzega, gdy tekst wychodzi poza zamrożony alfabet (model nie ma tych klas)."""
    cs_file = Path("configs/charset.json")
    if not cs_file.is_file():
        return
    allowed = set(json.loads(cs_file.read_text(encoding="utf-8")))
    outside = sorted(set(text) - allowed)
    if outside:
        print(f"[UWAGA] Znaki poza alfabetem modelu: {outside} — predict.py "
              "nigdy ich nie odtworzy")


def main():
    ap = argparse.ArgumentParser(description="Linia testowa OCR -> PNG")
    ap.add_argument("--text", default=DEFAULT_TEXT)
    ap.add_argument("--out", default="test_line.png")
    ap.add_argument("--font", default=None, help="ścieżka TTF (domyślnie: auto")
    ap.add_argument("--size", type=int, default=28)
    args = ap.parse_args()

    font_path = args.font or find_font()
    font = ImageFont.truetype(font_path, args.size)
    w = int(font.getlength(args.text)) + 2 * MARGIN_X
    img = Image.new("L", (w, CANVAS_H), 255)             # "L" jak w LineDataset
    ImageDraw.Draw(img).text((MARGIN_X, OFFSET_Y), args.text, font=font, fill=0)
    img.save(args.out)

    check_charset(args.text)
    print(f"[OK] {args.out}  ({w}x{CANVAS_H}px, font: {Path(font_path).name})")
    print(f"     tekst: {args.text!r}")
    print(f"     predykcja: python predict.py --image {args.out}")


if __name__ == "__main__":
    main()
