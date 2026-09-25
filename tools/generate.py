"""Command-line text-to-image with the Krea 2 Turbo Metal engine.

  .venv/bin/python tools/generate.py "a red fox asleep in fresh snow at dawn" -o fox.png
  .venv/bin/python tools/generate.py "..." --preset fast --size 1280 --seed 7
  .venv/bin/python tools/generate.py "..." --width 1536 --height 1024 --gpu-only

Presets: quality = the official 8-step Turbo schedule; fast = the 4-step distillation LoRA (merged weights).
The prompt goes through the same content-filter hook as the web UI (ui/content_filter.py).
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ui"))
import content_filter  # noqa: E402
from krea import PRESETS, Engine  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt")
    ap.add_argument("-o", "--out", default="out.png")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--preset", choices=list(PRESETS), default="quality")
    ap.add_argument("--steps", type=int, help="override the Quality preset's 8 steps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu-only", action="store_true", help="disable the Neural Engine assist")
    ap.add_argument("--json", action="store_true", help="also write <out>.json with the settings and timings")
    a = ap.parse_args()
    ok, reason = content_filter.check_prompt(a.prompt)
    if not ok:
        sys.exit(reason)
    steps, fast = PRESETS[a.preset]
    if a.steps and not fast:
        steps = a.steps
    t = time.time()
    eng = Engine(preset=a.preset)  # loads only this preset's weights
    print(f"engine loaded in {time.time() - t:.1f}s")

    def progress(stage, step, total, ms, preview=None):
        if stage == "denoise":
            print(f"\r  {stage} {step}/{total}  {ms / 1000:.1f}s", end="", flush=True)
        else:
            print(f"\n  {stage}  {ms / 1000:.1f}s", end="", flush=True)

    img, st = eng.generate(a.prompt, a.width or a.size, a.height or a.size, steps, a.seed, fast=fast,
                           ane=not a.gpu_only, progress=progress)
    print()
    ok, reason = content_filter.check_image(img, a.prompt)
    if not ok:
        sys.exit(reason)
    img.save(a.out)
    if a.json:
        with open(os.path.splitext(a.out)[0] + ".json", "w") as f:
            json.dump({"prompt": a.prompt, "preset": a.preset, "seed": a.seed, **st}, f, indent=1)
    print(f"saved {a.out}  total {st['total_ms'] / 1000:.1f}s | encode {st['encode_ms']:.0f}ms"
          f"{' (cached prompt)' if st['cached_prompt'] else ''} | {st['steps']} steps, {st['step_ms'] / 1000:.2f}s/step"
          f" | decode {st['decode_ms'] / 1000:.2f}s")


if __name__ == "__main__":
    main()
