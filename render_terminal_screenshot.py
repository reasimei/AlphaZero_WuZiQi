from __future__ import annotations

import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def load_text(log_path: Path, max_lines: int) -> list[str]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    if max_lines > 0:
        lines = lines[-max_lines:]
    return lines


def load_font(font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/Consolas.ttf",
        "C:/Windows/Fonts/lucon.ttf",
    ]
    for candidate in candidates:
        font_path = Path(candidate)
        if font_path.exists():
            return ImageFont.truetype(str(font_path), font_size)
    return ImageFont.load_default()


def render(lines: list[str], output_path: Path, width: int, font_size: int) -> None:
    font = load_font(font_size)
    line_height = font_size + 6
    margin = 24
    content_height = max(1, len(lines)) * line_height
    height = max(300, margin * 2 + content_height)
    image = Image.new("RGB", (width, height), color=(17, 24, 39))
    draw = ImageDraw.Draw(image)

    y = margin
    for line in lines:
        draw.text((margin, y), line, font=font, fill=(229, 231, 235))
        y += line_height

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a terminal log to PNG.")
    parser.add_argument("log_file")
    parser.add_argument("output_file")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--font-size", type=int, default=22)
    parser.add_argument("--max-lines", type=int, default=40)
    args = parser.parse_args()

    lines = load_text(Path(args.log_file), args.max_lines)
    render(lines, Path(args.output_file), args.width, args.font_size)


if __name__ == "__main__":
    main()
