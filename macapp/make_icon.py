"""
Generates AppIcon.icns for PersonaBot.app: a blue squircle with a simple
white chat-bubble glyph, matching the app's own iMessage-blue accent and
bubble shape. Requires Pillow (`pip install pillow`) and macOS's iconutil.

Usage: python3 make_icon.py [out_dir]   (default: ./build)
"""
import os
import subprocess
import sys
import tempfile

from PIL import Image, ImageDraw

SIZE = 1024
ACCENT = (0, 122, 255, 255)  # matches --accent in frontend/style.css
WHITE = (255, 255, 255, 255)
ICONSET_SIZES = [16, 32, 128, 256, 512]  # each also gets an @2x


def build_master_image():
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    corner = int(SIZE * 0.225)
    d.rounded_rectangle([0, 0, SIZE - 1, SIZE - 1], radius=corner, fill=ACCENT)

    bw, bh = int(SIZE * 0.62), int(SIZE * 0.44)
    bx, by = (SIZE - bw) // 2, int(SIZE * 0.28)
    d.rounded_rectangle([bx, by, bx + bw, by + bh], radius=int(bh * 0.42), fill=WHITE)

    tail_w, tail_h = int(SIZE * 0.09), int(SIZE * 0.11)
    tx, ty = bx + int(bw * 0.16), by + bh - int(bh * 0.12)
    d.polygon([(tx, ty), (tx + tail_w, ty), (tx + int(tail_w * 0.15), ty + tail_h)], fill=WHITE)

    dot_r = int(bh * 0.09)
    cy = by + bh // 2
    spacing = int(bw * 0.16)
    cx0 = SIZE // 2 - spacing
    for i in range(3):
        cx = cx0 + i * spacing
        d.ellipse([cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r], fill=ACCENT)

    return img


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "build"
    os.makedirs(out_dir, exist_ok=True)
    img = build_master_image()

    with tempfile.TemporaryDirectory() as tmp:
        iconset = os.path.join(tmp, "AppIcon.iconset")
        os.makedirs(iconset)
        for s in ICONSET_SIZES:
            img.resize((s, s), Image.LANCZOS).save(f"{iconset}/icon_{s}x{s}.png")
            img.resize((s * 2, s * 2), Image.LANCZOS).save(f"{iconset}/icon_{s}x{s}@2x.png")

        out_path = os.path.join(out_dir, "AppIcon.icns")
        subprocess.run(["iconutil", "-c", "icns", iconset, "-o", out_path], check=True)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
