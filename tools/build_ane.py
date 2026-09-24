"""Build the Neural Engine half of the hybrid Krea 2 DiT (Core ML programs, one set per layer).

Every DiT layer's linear work is split tensor-parallel between the GPU and the ANE:
  qkvg_XX  columns [c0, 15360) of the fused q | k | v | gate projection (the GPU computes [0, c0))
  mlp_XX   SwiGLU hidden units [H1, 16384) with their down-projection (the GPU computes [0, H1)); the GPU
           adds both partial down-projections to the residual
  o_XX     the full attention output projection (6144 -> 6144), used for pipelined chunks while the GPU
           runs the next chunk's attention (the GPU keeps a copy for the chunks it projects itself)
The GPU weight file with the matching slices is written by tools/convert.py dit_split.

Programs are 1x1 convolutions on one fixed shape, input "x" fp16 [1, 6144, 1, CHUNK] (channel-major, one
chunk of CHUNK token rows), output "out" fp16 [1, N, 1, CHUNK]. Weights are per-output-channel int8
(constexpr_affine_dequantize) by default; --mode lut6 builds 6-bit per-tensor palettes with per-channel scales
(macOS 14's opset has no grouped palettes) for the speed/quality experiment. SiLU is written as
g / (1 + exp(-g)) (the ANE's own silu is a coarse approximation).

fp16 range: the ANE multiplies raw int8 weights by fp16 activations before the per-channel scale, so
activations are pre-scaled: W_up by 1/OUT_SCALE (the MLP's hidden units and output shrink by that factor;
the GPU merge multiplies back), and the O-projection input by 1/O_SCALE (the GPU hand-off divides, the
merge multiplies back). meta.json records both.

  TMPDIR=<scratch> python tools/build_ane.py --gpu-units 3072 --qkvg-c0 3072 [--chunk 1024] [--fast] [--layers N]
"""
import argparse
import gc
import json
import os
import shutil
import sys
import time

import numpy as np
import torch
from safetensors import safe_open

import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert import DIFFUSERS, DIT_FILE, LORA_FILE  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D, FF, LAYERS, QKVG = 6144, 16384, 28, 15360
OUT_SCALE = 16.0
O_SCALE = 16.0


class Source:
    def __init__(self, lora):
        self.f = safe_open(DIT_FILE, "pt")
        self.lora = safe_open(LORA_FILE, "pt") if lora else None

    def get(self, name):
        """fp32 numpy weight [out, in], with the 4-step LoRA merged in fp32 when building the fast set."""
        w = self.f.get_tensor(name).float()
        if self.lora is not None:
            base = DIFFUSERS.get(name[: -len(".weight")])
            if base is not None and base + ".lora_A.weight" in self.lora.keys():
                A = self.lora.get_tensor(base + ".lora_A.weight").float()
                B = self.lora.get_tensor(base + ".lora_B.weight").float()
                w = w + float(self.lora.get_tensor(base + ".alpha")) / A.shape[0] * (B @ A)
        return w.numpy()


def weight_op(w, mode):
    """[out, in] linear weight -> 1x1 conv weight [out, in, 1, 1]."""
    if mode == "fp16":
        return mb.const(val=w.astype(np.float16).reshape(*w.shape, 1, 1))
    if mode == "int8":
        s = np.abs(w).max(axis=1, keepdims=True).clip(min=1e-12) / 127.0
        q = np.clip(np.round(w / s), -127, 127).astype(np.int8).reshape(*w.shape, 1, 1)
        return mb.constexpr_affine_dequantize(quantized_data=q, zero_point=np.int8(0),
                                              scale=s.reshape(-1).astype(np.float16), axis=0)
    raise ValueError(mode)


def kmeans_lut(w, nbits, iters=300):
    """Per-tensor k-means palette (weighted Lloyd on the fp16 value histogram) -> (lut fp16 [2^nbits],
    indices uint8 shaped like w). From the Qwen engine's ANE study (dev/engine/opt/ane_misc)."""
    k = 1 << nbits
    w16 = np.ascontiguousarray(w, dtype=np.float16)
    bits = w16.view(np.uint16).ravel()
    cnt = np.bincount(bits, minlength=65536).astype(np.float64)
    vals = np.arange(65536, dtype=np.uint32).astype(np.uint16).view(np.float16).astype(np.float64)
    ok = (cnt > 0) & np.isfinite(vals)
    u, c = vals[ok], cnt[ok]
    order = np.argsort(u)
    u, c, codes = u[order], c[order], np.nonzero(ok)[0][order]
    cdf = np.cumsum(c) / c.sum()
    cent = np.interp((np.arange(k) + 0.5) / k, cdf, u)
    for _ in range(iters):
        a = np.searchsorted((cent[1:] + cent[:-1]) / 2, u)
        s_ = np.bincount(a, weights=u * c, minlength=k)
        n = np.bincount(a, weights=c, minlength=k)
        new = np.where(n > 0, s_ / np.maximum(n, 1e-30), cent)
        new.sort()
        if np.allclose(new, cent, rtol=0, atol=1e-9):
            break
        cent = new
    a = np.searchsorted((cent[1:] + cent[:-1]) / 2, u)
    table = np.zeros(65536, np.uint8)
    table[codes] = a
    return cent.astype(np.float16), table[bits].reshape(w.shape)


def lut6_conv(x, w):
    """6-bit per-tensor palette on row-normalized weights with the per-channel scale applied as a mul after
    the conv (emulates per-channel palettization; macOS 14's opset only has per-tensor palettes)."""
    from coremltools.optimize._utils import pack_elements_into_bits

    s = np.abs(w).max(axis=1).clip(min=1e-12)
    lut, idx = kmeans_lut(w / s[:, None], 6)
    shape = np.array([w.shape[0], w.shape[1], 1, 1], dtype=np.uint32)
    wd = mb.constexpr_lut_to_dense(indices=pack_elements_into_bits(idx.ravel(), 6), lut=lut, shape=shape)
    return mb.mul(x=mb.conv(x=x, weight=wd), y=s.astype(np.float16).reshape(1, -1, 1, 1))


def conv(x, w, mode, name=None):
    y = lut6_conv(x, w) if mode == "lut6" else mb.conv(x=x, weight=weight_op(w, mode))
    return mb.identity(x=y, name=name) if name else y


def silu(g):
    """g / (1 + exp(-g)): the ANE's native silu (and g * sigmoid(g), which the compiler fuses into it) is a
    coarse piecewise approximation, ~2% relative error on small gate values; this form is ~45x more accurate
    (Qwen engine ANE study)."""
    return mb.real_div(x=g, y=mb.add(x=mb.exp(x=mb.mul(x=g, y=np.float16(-1.0))), y=np.float16(1.0)))


def convert(prog):
    return ct.convert(prog, compute_units=ct.ComputeUnit.CPU_AND_NE, minimum_deployment_target=ct.target.macOS14,
                      compute_precision=ct.precision.FLOAT16)


def build_linear(w, mode, chunk):
    @mb.program(input_specs=[mb.TensorSpec(shape=(1, D, 1, chunk), dtype=types.fp16)], opset_version=ct.target.macOS14)
    def prog(x):
        return conv(x, w, mode, "out")

    return convert(prog)


def build_mlp(wg, wu, wd, mode, chunk):
    @mb.program(input_specs=[mb.TensorSpec(shape=(1, D, 1, chunk), dtype=types.fp16)], opset_version=ct.target.macOS14)
    def prog(x):
        g = conv(x, wg, mode)
        u = conv(x, wu, mode)
        h = mb.mul(x=silu(g), y=u)
        return conv(h, wd, mode, "out")

    return convert(prog)


def save(model, out_dir, name):
    pkg = os.path.join(out_dir, name + ".mlpackage")
    model.save(pkg)
    compiled = ct.models.utils.compile_model(pkg)
    shutil.move(compiled, os.path.join(out_dir, name + ".mlmodelc"))
    shutil.rmtree(pkg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-units", type=int, default=3072, help="MLP hidden units computed on the GPU (rest -> ANE)")
    ap.add_argument("--qkvg-c0", type=int, default=3072, help="fused q|k|v|gate columns computed on the GPU")
    ap.add_argument("--chunk", type=int, default=1024, help="token rows per ANE call")
    ap.add_argument("--mode", default="int8", choices=["int8", "lut6", "fp16"])
    ap.add_argument("--no-o-proj", action="store_true", help="skip the o_XX programs")
    ap.add_argument("--fast", action="store_true", help="merge the 4-step LoRA (fp32) before quantizing")
    ap.add_argument("--layers", type=int, default=LAYERS)
    ap.add_argument("--out", default=None, help="output directory under weights/ (default ane or ane_fast)")
    a = ap.parse_args()
    if a.gpu_units % 32 or a.qkvg_c0 % 64:
        raise SystemExit("--gpu-units must be a multiple of 32 and --qkvg-c0 of 64")
    out_dir = os.path.join(ROOT, "weights", a.out or ("ane_fast" if a.fast else "ane"))
    os.makedirs(out_dir, exist_ok=True)
    src = Source(a.fast)
    done = lambda name: os.path.exists(os.path.join(out_dir, name + ".mlmodelc"))
    H1, c0 = a.gpu_units, a.qkvg_c0
    for l in range(a.layers):
        t = time.time()
        p = f"blocks.{l}."
        if not done(f"qkvg_{l:02d}"):
            w = np.concatenate([src.get(p + f"attn.{n}.weight") for n in ("wq", "wk", "wv", "gate")])[c0:]
            save(build_linear(w, a.mode, a.chunk), out_dir, f"qkvg_{l:02d}")
        if not done(f"mlp_{l:02d}"):
            wg = src.get(p + "mlp.gate.weight")[H1:]
            wu = src.get(p + "mlp.up.weight")[H1:] / OUT_SCALE
            wd = src.get(p + "mlp.down.weight")[:, H1:]
            save(build_mlp(wg, wu, wd, a.mode, a.chunk), out_dir, f"mlp_{l:02d}")
        if not a.no_o_proj and not done(f"o_{l:02d}"):
            save(build_linear(src.get(p + "attn.wo.weight"), a.mode, a.chunk), out_dir, f"o_{l:02d}")
        gc.collect()
        print(f"layer {l:2d} built in {time.time() - t:.1f}s", flush=True)
    meta = {"gpu_units": H1, "ane_units": FF - H1, "qkvg_c0": c0, "chunk": a.chunk, "mode": a.mode,
            "out_scale": OUT_SCALE, "o_proj": not a.no_o_proj, "o_scale": O_SCALE, "layers": a.layers,
            "lora": os.path.basename(LORA_FILE) if a.fast else None,
            "layout": "conv [1, 6144, 1, chunk] fp16 -> [1, N, 1, chunk]"}
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    print("done:", out_dir)


if __name__ == "__main__":
    sys.exit(main())
