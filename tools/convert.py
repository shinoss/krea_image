"""Convert the Krea 2 Turbo checkpoints into the engine's precompiled weight layout (weights/engine/).

  te.qw        Qwen3-VL-4B text model, layers 0-34 + embedding (bf16; norms f32)
  txt.qw       TextFusion (bf16 linears) + txtmlp (f32)
  dit.qw       28 DiT blocks (bf16) + first / last (f32); dit_q8.qw: blocks in int8 (group 128)
  dit_fast.qw  the same with the 4-step LoRA merged (W + B A in fp32, rounded once)
  cond.qw      f32 tmlp / tproj (for custom schedules, CPU only) + t(sigma), tvec(sigma) of the preset
               schedules (8 steps; 4 steps with the LoRA)
  vae.qw       Qwen-Image VAE decoder + encoder (bf16 convs, f32 norms / biases)

Format: "QWTS" | u32 version | u64 header_len | JSON header | zero pad | tensor data. Tensors are 16 KiB
aligned (the page size), grouped by name prefix (the engine wraps each prefix group, e.g. one DiT layer
"L12", in its own Metal buffer). Linear weights are stored transposed ([in, out], the GEMM's layout), Q/K/V
(/gate) projections are fused, and SwiGLU gate/up rows are interleaved per 64-column GEMM tile.
Norm weights of the (1 + w) form are stored as 1 + w in f32.

  python tools/convert.py te txt dit cond vae      # dit_q8, dit_fast, dit_fast_q8 on request
  python tools/convert.py verify                  # round-trip check of every tensor against the sources
"""
import argparse
import json
import math
import os
import struct
import sys
import time

import numpy as np
import torch
from safetensors import safe_open

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "weights", "src")
DST = os.path.join(ROOT, "weights", "engine")
DIT_FILE = os.path.join(SRC, "diffusion_models", "krea2_turbo_bf16.safetensors")
TE_FILE = os.path.join(SRC, "qwen3vl", "lm_layers0-34.safetensors")
VAE_DIR = os.path.join(SRC, "qwen-image", "vae")
LORA_FILE = os.path.join(SRC, "lora", "krea2_turbo_4step_rank_64_lora.safetensors")
ALIGN = 16384
GEMM_BN = 64   # must match BN in engine/kernels/gemm.metal
Q_GROUP = 128  # must match kQGroup in engine/kernels/gemm.metal
D, FF, LAYERS, TD, TFF, TE_LAYERS = 6144, 16384, 28, 2560, 6912, 35

torch.set_grad_enabled(False)

_DT = {"bf16": (torch.bfloat16, 2), "f32": (torch.float32, 4), "f16": (torch.float16, 2), "i8": (torch.int8, 1)}


class StreamWriter:
    """Writes tensors in a fixed order at offsets planned from their specs, so a 24 GB file never has to
    be held in memory: plan (name, dtype, shape) for every tensor first, then write() them in order."""

    def __init__(self, path, specs, meta=None):
        self.path, self.tmp = path, path + ".tmp"
        header = {"tensors": {}, "meta": meta or {}}
        off = 0
        self.order = []
        for name, dtype, shape in specs:
            nbytes = int(np.prod(shape)) * _DT[dtype][1]
            header["tensors"][name] = {"dtype": dtype, "shape": list(shape), "offset": off, "nbytes": nbytes}
            self.order.append(name)
            off = (off + nbytes + ALIGN - 1) // ALIGN * ALIGN
        hj = json.dumps(header).encode()
        self.data_start = (16 + len(hj) + ALIGN - 1) // ALIGN * ALIGN
        self.header = header
        self.f = open(self.tmp, "wb")
        self.f.write(b"QWTS" + struct.pack("<IQ", 1, len(hj)) + hj)
        self.f.truncate(self.data_start + off)
        self.next = 0

    def write(self, name, t):
        if self.order[self.next] != name:
            raise RuntimeError(f"write order: expected {self.order[self.next]}, got {name}")
        info = self.header["tensors"][name]
        dt = _DT[info["dtype"]][0]
        t = t.detach().to(dt).contiguous()
        if list(t.shape) != info["shape"]:
            raise RuntimeError(f"{name}: shape {list(t.shape)} != planned {info['shape']}")
        raw = t.view(torch.int16) if dt == torch.bfloat16 else t
        self.f.seek(self.data_start + info["offset"])
        self.f.write(raw.numpy().tobytes())
        self.next += 1

    def close(self):
        if self.next != len(self.order):
            raise RuntimeError(f"{self.path}: {len(self.order) - self.next} tensors not written")
        self.f.close()
        os.replace(self.tmp, self.path)
        print(f"wrote {self.path}: {len(self.order)} tensors, {os.path.getsize(self.path) / 1e9:.2f} GB", flush=True)


def q8(w_t):
    """[K, N] -> int8 [K, N] + f32 scales [K/128, N] (symmetric per output column and 128-row group)."""
    w = w_t.float()
    K, N = w.shape
    g = w.view(K // Q_GROUP, Q_GROUP, N)
    scale = g.abs().amax(dim=1).clamp(min=1e-12) / 127.0
    q = torch.round(g / scale[:, None, :]).clamp(-127, 127).view(K, N)
    return q.to(torch.int8), scale


def interleave_gate_up(gate, up):
    """gate/up [F, D] -> [2F, D] with rows grouped per GEMM tile: [gate(32) | up(32)] x (F/32)."""
    F_, Dm = gate.shape
    half = GEMM_BN // 2
    return torch.cat([gate.view(F_ // half, half, Dm), up.view(F_ // half, half, Dm)], dim=1).reshape(2 * F_, Dm)


def deinterleave_gate_up(gu):
    """inverse of interleave_gate_up: [2F, D] -> (gate [F, D], up [F, D])."""
    F2, Dm = gu.shape
    half = GEMM_BN // 2
    v = gu.view(F2 // GEMM_BN, GEMM_BN, Dm)
    return v[:, :half].reshape(-1, Dm), v[:, half:].reshape(-1, Dm)


def linear_specs(name, K, N, quant):
    if quant:
        return [(name, "i8", (K, N)), (name + ".s", "f32", (K // Q_GROUP, N))]
    return [(name, "bf16", (K, N))]


def put_linear(w, name, w_t, quant):
    """w_t: [K, N] (already transposed); bf16 (rounded once from f32 if merged) or int8."""
    if quant:
        q, s = q8(w_t)
        w.write(name, q)
        w.write(name + ".s", s)
    else:
        w.write(name, w_t)


# ---------------------------------------------------------------------------------------------
class DitSource:
    """Original checkpoint tensors, with the 4-step LoRA merged in fp32 (W + (alpha/r) B A) when asked."""

    def __init__(self, lora=False):
        self.f = safe_open(DIT_FILE, "pt")
        self.lora = safe_open(LORA_FILE, "pt") if lora else None
        self.lkeys = set(self.lora.keys()) if lora else set()
        self.merged = 0

    def raw(self, name):
        return self.f.get_tensor(name)

    def get(self, name):
        """fp32 if the LoRA touches it (then merged), else the stored dtype."""
        w = self.f.get_tensor(name)
        base = DIFFUSERS.get(name[: -len(".weight")]) if name.endswith(".weight") else None
        if self.lora is not None and base is not None and base + ".lora_A.weight" in self.lkeys:
            A = self.lora.get_tensor(base + ".lora_A.weight").float()
            B = self.lora.get_tensor(base + ".lora_B.weight").float()
            alpha = float(self.lora.get_tensor(base + ".alpha"))
            w = w.float() + (alpha / A.shape[0]) * (B @ A)
            self.merged += 1
        return w


def _diffusers_names():
    m = {"first": "img_in", "tmlp.0": "time_embed.linear_1", "tmlp.2": "time_embed.linear_2",
         "tproj.1": "time_mod_proj", "txtmlp.1": "txt_in.linear_1", "txtmlp.3": "txt_in.linear_2",
         "last.linear": "final_layer.linear"}
    for i in range(LAYERS):
        for a, b in {"wq": "to_q", "wk": "to_k", "wv": "to_v", "gate": "to_gate", "wo": "to_out.0"}.items():
            m[f"blocks.{i}.attn.{a}"] = f"transformer_blocks.{i}.attn.{b}"
        for a in ("gate", "up", "down"):
            m[f"blocks.{i}.mlp.{a}"] = f"transformer_blocks.{i}.ff.{a}"
    return m


DIFFUSERS = _diffusers_names()


def dit_specs(quant, H1=FF, split=False):
    specs = []
    for i in range(LAYERS):
        p = f"L{i}."
        specs += linear_specs(p + "kv", D, 2 * 1536, quant) if split else linear_specs(p + "qkvg", D, 2 * D + 2 * 1536, quant)
        specs += linear_specs(p + "o", D, D, quant)
        specs += linear_specs(p + "gu", D, 2 * H1, quant)
        specs += linear_specs(p + "down", H1, D, quant)
        specs += [(p + "qn", "f32", (128,)), (p + "kn", "f32", (128,)), (p + "n1", "f32", (D,)),
                  (p + "n2", "f32", (D,)), (p + "mod", "f32", (6, D))]
    specs += [("first.w", "f32", (64, D)), ("first.b", "f32", (D,)),
              ("last.w", "f32", (D, 64)), ("last.b", "f32", (64,)), ("last.n", "f32", (D,)), ("last.mod", "f32", (2, D))]
    return specs


def convert_dit(quant=False, fast=False, H1=FF, split=False):
    """split: the GPU's slices of the GPU/ANE split, dit_h<H1>_kv.qw: the k|v projection (L*.kv, 3072 columns),
    MLP units [0, H1) and the full O-projection; tools/build_ane.py builds the rest (q|gate, MLP units
    [H1, 16384), O) with the same H1."""
    s = DitSource(lora=fast)
    name = "dit" + (f"_h{H1}_kv" if split else "") + ("_fast" if fast else "") + ("_q8" if quant else "") + ".qw"
    w = StreamWriter(os.path.join(DST, name), dit_specs(quant, H1, split),
                     {"model": "krea2-turbo-dit", "layers": LAYERS, "q8": quant, "gpu_units": H1,
                      "attn_split": "kv" if split else None, "lora": os.path.basename(LORA_FILE) if fast else None})
    t0 = time.time()
    cast = (lambda t: t) if quant else (lambda t: t.to(torch.bfloat16))
    for i in range(LAYERS):
        p, b = f"L{i}.", f"blocks.{i}."
        if split:
            kv = torch.cat([s.get(b + f"attn.{n}.weight").float() for n in ("wk", "wv")])  # [3072, 6144]
            put_linear(w, p + "kv", cast(kv.t()), quant)
        else:
            qkvg = torch.cat([s.get(b + f"attn.{n}.weight").float() for n in ("wq", "wk", "wv", "gate")])  # [15360, 6144]
            put_linear(w, p + "qkvg", cast(qkvg.t()), quant)
        put_linear(w, p + "o", cast(s.get(b + "attn.wo.weight").float().t()), quant)
        gu = interleave_gate_up(s.get(b + "mlp.gate.weight").float()[:H1], s.get(b + "mlp.up.weight").float()[:H1])
        put_linear(w, p + "gu", cast(gu.t()), quant)
        put_linear(w, p + "down", cast(s.get(b + "mlp.down.weight").float()[:, :H1].t()), quant)
        w.write(p + "qn", s.raw(b + "attn.qknorm.qnorm.scale").float() + 1)
        w.write(p + "kn", s.raw(b + "attn.qknorm.knorm.scale").float() + 1)
        w.write(p + "n1", s.raw(b + "prenorm.scale").float() + 1)
        w.write(p + "n2", s.raw(b + "postnorm.scale").float() + 1)
        w.write(p + "mod", s.raw(b + "mod.lin").float().view(6, D))
        print(f"  block {i:2d}  {time.time() - t0:6.1f} s", flush=True)
    w.write("first.w", s.get("first.weight").float().t())
    w.write("first.b", s.raw("first.bias").float())
    w.write("last.w", s.get("last.linear.weight").float().t())
    w.write("last.b", s.raw("last.linear.bias").float())
    w.write("last.n", s.raw("last.norm.scale").float() + 1)
    w.write("last.mod", s.raw("last.modulation.lin").float())
    w.close()
    if fast:
        print(f"  merged {s.merged} LoRA modules into the blocks / first / last")


def convert_txt():
    s = DitSource()
    specs = []
    for b in range(4):
        p = f"F{b}."
        specs += [(p + "n1", "f32", (TD,)), (p + "qkvg", "bf16", (TD, 4 * TD)), (p + "qn", "f32", (128,)),
                  (p + "kn", "f32", (128,)), (p + "o", "bf16", (TD, TD)), (p + "n2", "f32", (TD,)),
                  (p + "gu", "bf16", (TD, 2 * TFF)), (p + "down", "bf16", (TFF, TD))]
    specs += [("proj", "f32", (12,)), ("M.n", "f32", (TD,)), ("M.w1", "f32", (TD, D)), ("M.b1", "f32", (D,)),
              ("M.w2", "f32", (D, D)), ("M.b2", "f32", (D,))]
    w = StreamWriter(os.path.join(DST, "txt.qw"), specs, {"model": "krea2-textfusion+txtmlp"})
    for b in range(4):
        p = f"F{b}."
        q = f"txtfusion.{'layerwise_blocks' if b < 2 else 'refiner_blocks'}.{b % 2}."
        w.write(p + "n1", s.raw(q + "prenorm.scale").float() + 1)
        qkvg = torch.cat([s.raw(q + f"attn.{n}.weight") for n in ("wq", "wk", "wv", "gate")])
        w.write(p + "qkvg", qkvg.t())
        w.write(p + "qn", s.raw(q + "attn.qknorm.qnorm.scale").float() + 1)
        w.write(p + "kn", s.raw(q + "attn.qknorm.knorm.scale").float() + 1)
        w.write(p + "o", s.raw(q + "attn.wo.weight").t())
        w.write(p + "n2", s.raw(q + "postnorm.scale").float() + 1)
        w.write(p + "gu", interleave_gate_up(s.raw(q + "mlp.gate.weight"), s.raw(q + "mlp.up.weight")).t())
        w.write(p + "down", s.raw(q + "mlp.down.weight").t())
    w.write("proj", s.raw("txtfusion.projector.weight").float().view(12))
    w.write("M.n", s.raw("txtmlp.0.scale").float() + 1)
    w.write("M.w1", s.raw("txtmlp.1.weight").float().t())
    w.write("M.b1", s.raw("txtmlp.1.bias").float())
    w.write("M.w2", s.raw("txtmlp.3.weight").float().t())
    w.write("M.b2", s.raw("txtmlp.3.bias").float())
    w.close()


def sigmas(steps, mu=1.15):
    """krea_sigmas() of engine/src/model.mm, bit for bit (double math, rounded to f32)."""
    e = math.exp(mu)
    out = []
    for i in range(steps + 1):
        t = 1.0 - i / steps
        out.append(0.0 if t <= 0 else float(np.float32(e / (e + (1.0 / t - 1.0)))))
    return out


def bf16(x):
    return float(torch.tensor(x, dtype=torch.float32).to(torch.bfloat16).float())


def conditioning(s, sig):
    """t [6144], tvec [36864] for sigma (the model sees bf16(sigma), as in sampling.py); fp32."""
    half = 128
    freqs = torch.exp(-math.log(1e4) * torch.arange(half, dtype=torch.float32) / half)
    args = (bf16(sig) * 1e3) * freqs
    e = torch.cat([torch.cos(args), torch.sin(args)])
    h = torch.nn.functional.gelu(e @ s.get("tmlp.0.weight").float().T + s.raw("tmlp.0.bias").float(), approximate="tanh")
    t = h @ s.get("tmlp.2.weight").float().T + s.raw("tmlp.2.bias").float()
    tvec = torch.nn.functional.gelu(t, approximate="tanh") @ s.get("tproj.1.weight").float().T + s.raw("tproj.1.bias").float()
    return t, tvec


def convert_cond():
    base, fast = DitSource(), DitSource(lora=True)
    s8, s4 = sigmas(8)[:-1], sigmas(4)[:-1]
    specs = [("tmlp1.w", "f32", (256, D)), ("tmlp1.b", "f32", (D,)), ("tmlp2.w", "f32", (D, D)), ("tmlp2.b", "f32", (D,)),
             ("tproj.w", "f32", (D, 6 * D)), ("tproj.b", "f32", (6 * D,)),
             ("base.sigma", "f32", (len(s8),)), ("base.t", "f32", (len(s8), D)), ("base.tvec", "f32", (len(s8), 6 * D)),
             ("fast.sigma", "f32", (len(s4),)), ("fast.t", "f32", (len(s4), D)), ("fast.tvec", "f32", (len(s4), 6 * D))]
    w = StreamWriter(os.path.join(DST, "cond.qw"), specs, {"model": "krea2-conditioning", "mu": 1.15,
                                                           "note": "t and tvec use bf16(sigma), as sampling.py"})
    w.write("tmlp1.w", base.raw("tmlp.0.weight").float().t())
    w.write("tmlp1.b", base.raw("tmlp.0.bias").float())
    w.write("tmlp2.w", base.raw("tmlp.2.weight").float().t())
    w.write("tmlp2.b", base.raw("tmlp.2.bias").float())
    w.write("tproj.w", base.raw("tproj.1.weight").float().t())
    w.write("tproj.b", base.raw("tproj.1.bias").float())
    for tag, src, sig in (("base", base, s8), ("fast", fast, s4)):
        ts, tvs = zip(*(conditioning(src, x) for x in sig))
        w.write(tag + ".sigma", torch.tensor(sig, dtype=torch.float32))
        w.write(tag + ".t", torch.stack(ts))
        w.write(tag + ".tvec", torch.stack(tvs))
        print(f"  {tag}: sigmas {[round(x, 5) for x in sig]}")
    w.close()


def convert_te(quant=False):
    f = safe_open(TE_FILE, "pt")
    pre = "model.language_model."
    specs = [("embed", "bf16", (151936, TD))]
    for i in range(TE_LAYERS):
        p = f"L{i}."
        specs += [(p + "ln1", "f32", (TD,))] + linear_specs(p + "qkv", TD, 6144, quant) + \
                 [(p + "qn", "f32", (128,)), (p + "kn", "f32", (128,))] + linear_specs(p + "o", 4096, TD, quant) + \
                 [(p + "ln2", "f32", (TD,))] + linear_specs(p + "gu", TD, 2 * 9728, quant) + linear_specs(p + "down", 9728, TD, quant)
    w = StreamWriter(os.path.join(DST, "te_q8.qw" if quant else "te.qw"), specs,
                     {"model": "qwen3-vl-4b-text", "layers": TE_LAYERS, "q8": quant})
    w.write("embed", f.get_tensor(pre + "embed_tokens.weight"))
    for i in range(TE_LAYERS):
        p, q = f"L{i}.", f"{pre}layers.{i}."
        w.write(p + "ln1", f.get_tensor(q + "input_layernorm.weight").float())
        qkv = torch.cat([f.get_tensor(q + f"self_attn.{n}_proj.weight") for n in "qkv"])
        put_linear(w, p + "qkv", qkv.t(), quant)
        w.write(p + "qn", f.get_tensor(q + "self_attn.q_norm.weight").float())
        w.write(p + "kn", f.get_tensor(q + "self_attn.k_norm.weight").float())
        put_linear(w, p + "o", f.get_tensor(q + "self_attn.o_proj.weight").t(), quant)
        w.write(p + "ln2", f.get_tensor(q + "post_attention_layernorm.weight").float())
        put_linear(w, p + "gu", interleave_gate_up(f.get_tensor(q + "mlp.gate_proj.weight"),
                                                   f.get_tensor(q + "mlp.up_proj.weight")).t(), quant)
        put_linear(w, p + "down", f.get_tensor(q + "mlp.down_proj.weight").t(), quant)
    w.close()


def conv_w(t):
    """Conv weight [Cout, Cin, (T,) kh, kw] -> implicit-GEMM layout [(kh*kw*Cin), Cout]; a causal 3-D conv
    keeps its last temporal tap (a single frame sees the others only through zero padding)."""
    if t.dim() == 5:
        t = t[:, :, -1]
    co, ci, kh, kw = t.shape
    return t.permute(2, 3, 1, 0).reshape(kh * kw * ci, co)


def convert_vae():
    """Qwen-Image (Wan 2.1) VAE, single frame: decoder (+ post_quant_conv) for generation, encoder (+ quant_conv)
    for editing. Causal 3-D convs keep their last temporal tap; the downsamplers' / upsamplers' time_conv only
    runs from the second frame on and is dropped."""
    f = safe_open(os.path.join(VAE_DIR, "diffusion_pytorch_model.safetensors"), "pt")
    cfg = json.load(open(os.path.join(VAE_DIR, "config.json")))
    parts = ("decoder.", "post_quant_conv.", "encoder.", "quant_conv.")
    keys = sorted(k for k in f.keys() if k.startswith(parts) and "time_conv" not in k)
    tensors = []
    for k in keys:
        t = f.get_tensor(k).float()
        if k in ("post_quant_conv.weight", "quant_conv.weight"):
            tensors.append((k, "f32", t[:, :, 0, 0, 0].t().contiguous()))  # [in, out]
        elif k == "decoder.conv_out.weight":  # 3 -> 4 output channels (alpha = bias 1)
            w4 = torch.zeros(4, *t.shape[1:])
            w4[:3] = t
            tensors.append((k, "bf16", conv_w(w4)))
        elif k == "decoder.conv_out.bias":
            tensors.append((k, "f32", torch.cat([t, torch.ones(1)])))
        elif k == "encoder.conv_in.weight":  # RGB input padded to 16 channels (the conv kernel's K step)
            w16 = torch.zeros(t.shape[0], 16, *t.shape[2:])
            w16[:, :3] = t
            tensors.append((k, "bf16", conv_w(w16)))
        elif k == "encoder.conv_out.weight":  # 32 -> 48 output channels (the conv kernel's N tile)
            w48 = torch.zeros(48, *t.shape[1:])
            w48[:32] = t
            tensors.append((k, "bf16", conv_w(w48)))
        elif k == "encoder.conv_out.bias":
            tensors.append((k, "f32", torch.cat([t, torch.zeros(16)])))
        elif k.endswith(".weight") and t.dim() >= 4:
            tensors.append((k, "bf16", conv_w(t)))
        elif k.endswith(".gamma"):
            tensors.append((k, "f32", t.reshape(-1)))
        else:
            tensors.append((k, "f32", t))
    tensors.append(("latents_mean", "f32", torch.tensor(cfg["latents_mean"])))
    tensors.append(("latents_std", "f32", torch.tensor(cfg["latents_std"])))
    # group by segment prefix (up to the first '.'): "decoder", "encoder", "post_quant_conv", ...
    tensors.sort(key=lambda x: x[0].split(".")[0])
    w = StreamWriter(os.path.join(DST, "vae.qw"), [(n, d, tuple(t.shape)) for n, d, t in tensors],
                     {"model": "qwen-image-vae (wan2.1): decoder + encoder"})
    for n, d, t in tensors:
        w.write(n, t)
    w.close()


def copy_tokenizer():
    import shutil
    shutil.copy(os.path.join(SRC, "krea2", "tokenizer", "tokenizer.json"), os.path.join(DST, "tokenizer.json"))
    print("copied tokenizer.json")


# ---------------------------------------------------------------------------------------------
def verify():
    """Every converted tensor, inverted, against the source bytes (exact for bf16 / f32 layouts)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from qw import QwFile

    bad = 0

    def eq(name, a, b):
        nonlocal bad
        ok = a.shape == b.shape and torch.equal(a, b)
        if not ok:
            bad += 1
            print(f"  MISMATCH {name}: {tuple(a.shape)} vs {tuple(b.shape)}")
        return ok

    if os.path.exists(os.path.join(DST, "dit.qw")):
        q, s = QwFile(os.path.join(DST, "dit.qw")), DitSource()
        for i in range(LAYERS):
            p, b = f"L{i}.", f"blocks.{i}."
            qkvg = q.get(p + "qkvg").t()
            for n, part in zip(("wq", "wk", "wv", "gate"), qkvg.split([D, 1536, 1536, D])):
                eq(p + n, part, s.raw(b + f"attn.{n}.weight"))
            eq(p + "o", q.get(p + "o").t(), s.raw(b + "attn.wo.weight"))
            g, u = deinterleave_gate_up(q.get(p + "gu").t())
            eq(p + "gate", g, s.raw(b + "mlp.gate.weight"))
            eq(p + "up", u, s.raw(b + "mlp.up.weight"))
            eq(p + "down", q.get(p + "down").t(), s.raw(b + "mlp.down.weight"))
            eq(p + "mod", q.get(p + "mod").reshape(-1), s.raw(b + "mod.lin"))
            eq(p + "n1", q.get(p + "n1") - 1, s.raw(b + "prenorm.scale"))
        eq("first.w", q.get("first.w").t(), s.raw("first.weight"))
        eq("last.w", q.get("last.w").t(), s.raw("last.linear.weight"))
        print(f"dit.qw checked ({bad} mismatches so far)")
    if os.path.exists(os.path.join(DST, "txt.qw")):
        q, s = QwFile(os.path.join(DST, "txt.qw")), DitSource()
        for b in range(4):
            p, src = f"F{b}.", f"txtfusion.{'layerwise_blocks' if b < 2 else 'refiner_blocks'}.{b % 2}."
            for n, part in zip(("wq", "wk", "wv", "gate"), q.get(p + "qkvg").t().split(TD)):
                eq(p + n, part, s.raw(src + f"attn.{n}.weight"))
            g, u = deinterleave_gate_up(q.get(p + "gu").t())
            eq(p + "gate", g, s.raw(src + "mlp.gate.weight"))
            eq(p + "up", u, s.raw(src + "mlp.up.weight"))
            eq(p + "down", q.get(p + "down").t(), s.raw(src + "mlp.down.weight"))
        eq("M.w2", q.get("M.w2").t(), s.raw("txtmlp.3.weight"))
        print(f"txt.qw checked ({bad} mismatches so far)")
    if os.path.exists(os.path.join(DST, "te.qw")):
        q, f = QwFile(os.path.join(DST, "te.qw")), safe_open(TE_FILE, "pt")
        pre = "model.language_model."
        eq("embed", q.get("embed"), f.get_tensor(pre + "embed_tokens.weight"))
        for i in range(TE_LAYERS):
            p, s_ = f"L{i}.", f"{pre}layers.{i}."
            for n, part in zip("qkv", q.get(p + "qkv").t().split([4096, 1024, 1024])):
                eq(p + n, part, f.get_tensor(s_ + f"self_attn.{n}_proj.weight"))
            g, u = deinterleave_gate_up(q.get(p + "gu").t())
            eq(p + "gate", g, f.get_tensor(s_ + "mlp.gate_proj.weight"))
            eq(p + "up", u, f.get_tensor(s_ + "mlp.up_proj.weight"))
            eq(p + "down", q.get(p + "down").t(), f.get_tensor(s_ + "mlp.down_proj.weight"))
            eq(p + "ln1", q.get(p + "ln1").to(torch.bfloat16), f.get_tensor(s_ + "input_layernorm.weight"))
        print(f"te.qw checked ({bad} mismatches so far)")
    print("verify:", "all tensors match" if not bad else f"{bad} MISMATCHES")
    return bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("parts", nargs="+", help="te te_q8 txt dit dit_q8 dit_fast dit_fast_q8 dit_split dit_split_fast "
                                             "cond vae tokenizer verify")
    ap.add_argument("--gpu-units", type=int, default=3072, help="dit_split: MLP units kept on the GPU")
    a = ap.parse_args()
    os.makedirs(DST, exist_ok=True)
    jobs = {"te": convert_te, "te_q8": lambda: convert_te(True), "txt": convert_txt, "dit": convert_dit,
            "dit_q8": lambda: convert_dit(True), "dit_fast": lambda: convert_dit(False, True),
            "dit_fast_q8": lambda: convert_dit(True, True),
            "dit_split": lambda: convert_dit(False, False, a.gpu_units, True),
            "dit_split_fast": lambda: convert_dit(False, True, a.gpu_units, True),
            "cond": convert_cond, "vae": convert_vae,
            "tokenizer": copy_tokenizer, "verify": verify}
    for part in a.parts:
        t = time.time()
        print(f"[{part}]", flush=True)
        r = jobs[part]()
        print(f"[{part}] {time.time() - t:.1f} s", flush=True)
        if part == "verify" and r:
            sys.exit(1)
