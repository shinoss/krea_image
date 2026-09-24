"""Render the KreaImage app icon -> AppIcon.png / AppIcon.icns.

A macOS-style rounded tile with a warm orange-to-magenta gradient and a white "K" built from three
round-capped strokes, with a small sparkle. Drawn at 4x and downsampled for anti-aliased edges.
"""
import math
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = 1024
SS = 4  # supersampling factor
S = OUT * SS


def gradient(size):
    stops = np.array([[255, 138, 60], [232, 72, 60], [176, 38, 118]], dtype=np.float32)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / (size - 1)
    t = np.clip(0.55 * yy + 0.45 * xx, 0, 1)[..., None]
    lo = np.where(t < 0.5, stops[0], stops[1])
    hi = np.where(t < 0.5, stops[1], stops[2])
    u = np.where(t < 0.5, t * 2, (t - 0.5) * 2)
    return Image.fromarray((lo + (hi - lo) * u).astype(np.uint8), "RGB")


def capsule(draw, p0, p1, width, fill):
    """A line segment with round caps (PIL lines have flat caps)."""
    (x0, y0), (x1, y1) = p0, p1
    dx, dy = x1 - x0, y1 - y0
    n = math.hypot(dx, dy)
    ox, oy = -dy / n * width / 2, dx / n * width / 2
    draw.polygon([(x0 + ox, y0 + oy), (x1 + ox, y1 + oy), (x1 - ox, y1 - oy), (x0 - ox, y0 - oy)], fill=fill)
    r = width / 2
    for x, y in (p0, p1):
        draw.ellipse([x - r, y - r, x + r, y + r], fill=fill)


def sparkle(draw, x, y, r, w, fill):
    pts = [(x, y - r), (x + w, y - w), (x + r, y), (x + w, y + w), (x, y + r), (x - w, y + w), (x - r, y), (x - w, y - w)]
    draw.polygon(pts, fill=fill)


def render():
    k = SS
    margin, radius = 100 * k, 185 * k
    tile = Image.new("L", (S, S), 0)
    ImageDraw.Draw(tile).rounded_rectangle([margin, margin, S - margin, S - margin], radius=radius, fill=255)

    icon = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    shadow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    shadow.putalpha(tile.filter(ImageFilter.GaussianBlur(18 * k)).point(lambda v: v * 110 // 255))
    icon.alpha_composite(shadow, (0, 12 * k))
    body = gradient(S).convert("RGBA")
    body.putalpha(tile)
    icon.alpha_composite(body)

    # "K": stem + two arms meeting left of centre
    g = Image.new("L", (S, S), 0)
    d = ImageDraw.Draw(g)
    w = 118 * k
    x0, top, bot = 345 * k, 300 * k, 724 * k
    joint = (x0 + 30 * k, 520 * k)
    capsule(d, (x0, top), (x0, bot), w, 255)
    capsule(d, joint, (675 * k, top), w, 255)
    # the lower arm starts on the upper arm's centre line, so its round cap stays inside that stroke
    a = 0.38
    start = (joint[0] + a * (675 * k - joint[0]), joint[1] + a * (top - joint[1]))
    capsule(d, start, (690 * k, bot), w, 255)
    sparkle(d, 790 * k, 250 * k, 62 * k, 13 * k, 235)
    white = Image.new("RGBA", (S, S), (255, 255, 255, 0))
    white.putalpha(g.point(lambda v: v * 245 // 255))
    icon.alpha_composite(white)
    return icon.resize((OUT, OUT), Image.LANCZOS)


def main():
    icon = render()
    iconset = os.path.join(HERE, "AppIcon.iconset")
    os.makedirs(iconset, exist_ok=True)
    for size in (16, 32, 128, 256, 512):
        icon.resize((size, size), Image.LANCZOS).save(os.path.join(iconset, f"icon_{size}x{size}.png"))
        icon.resize((size * 2, size * 2), Image.LANCZOS).save(os.path.join(iconset, f"icon_{size}x{size}@2x.png"))
    icon.save(os.path.join(HERE, "AppIcon.png"))
    subprocess.run(["iconutil", "-c", "icns", iconset, "-o", os.path.join(HERE, "AppIcon.icns")], check=True)
    print("wrote", os.path.join(HERE, "AppIcon.icns"))


if __name__ == "__main__":
    sys.exit(main())
