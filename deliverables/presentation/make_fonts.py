#!/usr/bin/env python3
"""Шрифты для PPTX: статические TTF из вариативных TTF google/fonts.

PowerPoint и LibreOffice надёжно понимают только пару «семейство + жирный» (b="1").
Поэтому из каждого вариативного шрифта делаем статические начертания:
  400 — «Onest», 700 — «Onest» с флагом Bold (обычная пара Regular/Bold);
  500, 600, 800 — отдельные семейства «Onest Medium», «Onest SemiBold», «Onest ExtraBold»
  (классические имена без typographic family: так их находит и PowerPoint, и LibreOffice).
То же для «JetBrains Mono» и «Tektur». Список начертаний пишется в fonts-ttf/fonts.json —
по нему to_pptx.mjs выбирает гарнитуру для каждой толщины.

    python make_fonts.py              скачать и собрать fonts-ttf/*.ttf
    python make_fonts.py --install    плюс установить в ~/Library/Fonts (macOS) или ~/.local/share/fonts

Нужен fontTools (есть в .venv проекта). Источник — github.com/google/fonts, лицензия SIL OFL 1.1.
"""
from __future__ import annotations

import argparse
import io
import json
import platform
import shutil
import sys
import urllib.parse
import urllib.request
from pathlib import Path

from fontTools.ttLib import TTFont
from fontTools.varLib import instancer

HERE = Path(__file__).resolve().parent
OUT = HERE / "fonts-ttf"
RAW = "https://raw.githubusercontent.com/google/fonts/main/ofl/"

# семейство в PPTX → (каталог google/fonts, вариативный файл, фиксированные оси, префикс файлов)
FAMILIES = {
    "Onest": ("onest", "Onest[wght].ttf", {}, "Onest"),
    "JetBrains Mono": ("jetbrainsmono", "JetBrainsMono[wght].ttf", {}, "JetBrainsMono"),
    "Tektur": ("tektur", "Tektur[wdth,wght].ttf", {"wdth": 100}, "Tektur"),
}
# толщина → имя начертания; 400 и 700 — одно семейство (Regular/Bold), остальные — отдельные семейства
WEIGHTS = {400: "Regular", 500: "Medium", 600: "SemiBold", 700: "Bold", 800: "ExtraBold"}


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as resp:
        return resp.read()


def legacy_names(family: str, weight: int) -> tuple[str, str]:
    """(семейство, начертание) в классической схеме RIBBI."""
    if weight == 400:
        return family, "Regular"
    if weight == 700:
        return family, "Bold"
    return f"{family} {WEIGHTS[weight]}", "Regular"


def set_names(font: TTFont, family: str, weight: int) -> None:
    fam, style = legacy_names(family, weight)
    ps = family.replace(" ", "") + "-" + WEIGHTS[weight]
    full = f"{family} {WEIGHTS[weight]}"
    table = font["name"]
    for name_id in (1, 2, 3, 4, 6, 16, 17, 21, 22, 25):
        table.removeNames(nameID=name_id)
    for name_id, value in {1: fam, 2: style, 3: ps + ";static", 4: full, 6: ps}.items():
        table.setName(value, name_id, 3, 1, 0x409)  # Windows, Unicode BMP, en-US
        table.setName(value, name_id, 1, 0, 0)      # Macintosh Roman
    bold = style == "Bold"
    os2 = font["OS/2"]
    os2.usWeightClass = weight
    sel = os2.fsSelection & ~(0x01 | 0x20 | 0x40)   # сброс ITALIC, BOLD, REGULAR
    os2.fsSelection = sel | (0x20 if bold else 0x40)
    head = font["head"]
    head.macStyle = (head.macStyle & ~0x03) | (0x01 if bold else 0)


def build() -> list[Path]:
    OUT.mkdir(exist_ok=True)
    made: list[Path] = []
    manifest: dict[str, dict[str, dict]] = {}
    for family, (folder, var_file, fixed, prefix) in FAMILIES.items():
        url = RAW + folder + "/" + urllib.parse.quote(var_file)
        print("скачиваю", url)
        data = fetch(url)
        (OUT / f"OFL-{prefix}.txt").write_bytes(fetch(RAW + folder + "/OFL.txt"))
        manifest[family] = {}
        for weight, style_name in WEIGHTS.items():
            vf = TTFont(io.BytesIO(data))
            axes = {a.axisTag for a in vf["fvar"].axes}
            loc = {"wght": weight, **{k: v for k, v in fixed.items() if k in axes}}
            font = instancer.instantiateVariableFont(vf, loc, updateFontNames=False)
            for tag in ("STAT", "MVAR", "HVAR", "avar", "fvar", "gvar", "cvar"):
                if tag in font:
                    del font[tag]
            set_names(font, family, weight)
            path = OUT / f"{prefix}-{style_name}.ttf"
            font.save(path)
            made.append(path)
            fam, style = legacy_names(family, weight)
            manifest[family][str(weight)] = {"typeface": fam, "bold": style == "Bold", "file": path.name}
            print("  ", path.name, f"wght={weight}", "→", fam, "/", style, path.stat().st_size, "байт")
    (OUT / "fonts.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return made


def install(paths: list[Path]) -> None:
    system = platform.system()
    if system == "Darwin":
        target = Path.home() / "Library" / "Fonts"
    elif system == "Linux":
        target = Path.home() / ".local" / "share" / "fonts"
    else:
        print("Windows: выделите fonts-ttf/*.ttf → правая кнопка → «Установить»", file=sys.stderr)
        return
    target.mkdir(parents=True, exist_ok=True)
    for p in paths:
        shutil.copy2(p, target / p.name)
    print("установлено в", target, "—", len(paths), "файлов")


def main() -> None:
    ap = argparse.ArgumentParser(description="Статические TTF для PPTX из вариативных шрифтов google/fonts")
    ap.add_argument("--install", action="store_true", help="установить шрифты в пользовательский каталог")
    args = ap.parse_args()
    paths = build()
    if args.install:
        install(paths)


if __name__ == "__main__":
    main()
