"""Fit the latent -> RGB projection used by the live preview.

Generates a handful of images with the engine (the project's own prompts, never the user's), keeps each final
packed latent and the VAE-decoded image, unpacks the latent to its 16 channels at 1/8 resolution, box-averages
the image to that resolution (8x8) and solves rgb = latent @ A + b by least squares.
Writes weights/engine/latent_rgb.json.

  python tools/fit_latent_rgb.py [--size 1024] [--keep DIR]
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ui"))
from krea import Engine  # noqa: E402

# Varied palettes / content so the fit covers the colour space (not tied to any user data).
PROMPTS = [
    "A bowl of ripe oranges, lemons and limes on a turquoise tablecloth, bright studio light",
    "Dense green rainforest with a red macaw on a branch, morning mist",
    "Snowy mountain range at blue hour, deep blue shadows and pink peaks",
    "A crowded night market with neon signs in magenta, cyan and yellow",
    "Close-up portrait of an elderly fisherman, warm golden sunset light",
    "Abstract oil painting with thick strokes of cobalt blue, crimson and ochre",
    "A white ceramic teapot on a black slate surface, minimalist product photo",
    "Lavender fields under a stormy purple sky with a single yellow house",
]


def unpack(lat, H16, W16):
    """[H16*W16, 64] packed tokens (c, py, px) -> [2*H16 * 2*W16, 16]."""
    x = lat.reshape(H16, W16, 16, 2, 2).transpose(0, 3, 1, 4, 2)
    return x.reshape(4 * H16 * W16, 16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--keep", default="", help="optional directory to save the fitting images")
    a = ap.parse_args()
    eng = Engine()
    X, Y = [], []
    for i, prompt in enumerate(PROMPTS):
        img, st = eng.generate(prompt, a.size, a.size, 8, seed=1000 + i, return_latent=True)
        h16 = w16 = a.size // 16
        lat = unpack(np.frombuffer(st["latent"], dtype=np.float32), h16, w16)
        rgb = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        small = rgb.reshape(2 * h16, 8, 2 * w16, 8, 3).mean(axis=(1, 3)).reshape(-1, 3)
        X.append(lat)
        Y.append(small)
        if a.keep:
            os.makedirs(a.keep, exist_ok=True)
            img.save(os.path.join(a.keep, f"fit_{i}.png"))
        print(f"[{i + 1}/{len(PROMPTS)}] {prompt[:60]}", flush=True)
    X = np.concatenate(X)
    Y = np.concatenate(Y)
    Xb = np.concatenate([X, np.ones((len(X), 1), np.float32)], axis=1)
    sol, *_ = np.linalg.lstsq(Xb, Y, rcond=None)
    A, b = sol[:16], sol[16]
    pred = np.clip(Xb @ sol, 0, 1)
    rmse = float(np.sqrt(((pred - Y) ** 2).mean()))
    r2 = 1 - ((pred - Y) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()
    out = {"weights": A.tolist(), "bias": b.tolist(), "rmse": rmse, "r2": float(r2), "samples": int(len(X)),
           "note": "normalized DiT latent (16 ch at 1/8 resolution) -> RGB in [0,1]; fitted against the Qwen-Image VAE"}
    path = os.path.join(ROOT, "weights", "engine", "latent_rgb.json")
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"wrote {path}: {len(X)} samples, RMSE {rmse:.4f}, R^2 {r2:.3f}")


if __name__ == "__main__":
    main()
