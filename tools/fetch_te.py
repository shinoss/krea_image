"""Fetch only the language-model tensors that Krea 2 needs from Qwen/Qwen3-VL-4B-Instruct.

Krea 2 taps the text encoder's hidden_states[2, 5, ..., 35], i.e. the outputs of decoder layers
1, 4, ..., 34, so layers 0-34 and the embedding table are all it ever uses: the vision tower, layer
35 and the final norm are skipped. The tensors are read with HTTP range requests straight from the
two checkpoint shards (never storing the shards) into one local safetensors file with the original
names and bytes:

  weights/src/qwen3vl/lm_layers0-34.safetensors   (~7.84 GB)

  python tools/fetch_te.py [--layers 35] [--workers 8]
"""
import argparse
import json
import os
import struct
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = "Qwen/Qwen3-VL-4B-Instruct"
SRC = os.path.join(ROOT, "weights", "src", "qwen3vl")
OUT = os.path.join(SRC, "lm_layers0-34.safetensors")
PIECE = 32 << 20  # bytes per range request

_session = threading.local()


def session():
    if not hasattr(_session, "s"):
        _session.s = requests.Session()
    return _session.s


def url(shard):
    return f"https://huggingface.co/{REPO}/resolve/main/{shard}"


def get_range(shard, lo, hi, tries=6):
    """Bytes [lo, hi) of a shard."""
    for attempt in range(tries):
        try:
            r = session().get(url(shard), headers={"Range": f"bytes={lo}-{hi - 1}"}, timeout=120)
            if r.status_code != 206 or len(r.content) != hi - lo:
                raise IOError(f"HTTP {r.status_code}, {len(r.content)} of {hi - lo} bytes")
            return r.content
        except Exception as e:  # transient network errors: back off and retry
            if attempt == tries - 1:
                raise
            print(f"  retry {shard} [{lo}, {hi}): {e}", flush=True)
            time.sleep(2 * (attempt + 1))


def header(shard):
    hlen = struct.unpack("<Q", get_range(shard, 0, 8))[0]
    return hlen, json.loads(get_range(shard, 8, 8 + hlen))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=35, help="decoder layers to keep (0 .. layers-1)")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    wm = json.load(open(os.path.join(SRC, "model.safetensors.index.json")))["weight_map"]
    pre = "model.language_model."
    want = [k for k in wm if k == pre + "embed_tokens.weight" or
            (k.startswith(pre + "layers.") and int(k[len(pre) + 7:].split(".")[0]) < a.layers)]
    shards = sorted({wm[k] for k in want})
    heads = {s: header(s) for s in shards}
    # output layout: shard order, then source offset order (sequential reads, contiguous writes)
    items = []
    for s in shards:
        hlen, h = heads[s]
        for k in sorted((k for k in want if wm[k] == s), key=lambda k: h[k]["data_offsets"][0]):
            lo, hi = h[k]["data_offsets"]
            items.append((k, s, 8 + hlen + lo, 8 + hlen + hi, h[k]["dtype"], h[k]["shape"]))
    out_h, off = {}, 0
    for k, s, lo, hi, dt, shape in items:
        out_h[k] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + hi - lo]}
        off += hi - lo
    out_h["__metadata__"] = {"source": REPO, "note": f"language-model layers 0-{a.layers - 1} + embed_tokens only"}
    hj = json.dumps(out_h, separators=(",", ":")).encode()
    hj += b" " * ((8 - len(hj) % 8) % 8)
    base = 8 + len(hj)
    total = off
    print(f"{len(items)} tensors, {total / 1e9:.3f} GB from {len(shards)} shards -> {OUT}", flush=True)

    tmp = OUT + ".part"
    fd = os.open(tmp, os.O_RDWR | os.O_CREAT, 0o644)
    os.ftruncate(fd, base + total)
    os.pwrite(fd, struct.pack("<Q", len(hj)) + hj, 0)
    # contiguous source spans -> pieces of PIECE bytes
    jobs, dst = [], base
    for k, s, lo, hi, *_ in items:
        p = lo
        while p < hi:
            q = min(hi, p + PIECE)
            jobs.append((s, p, q, dst + (p - lo)))
            p = q
        dst += hi - lo
    done = [0]
    lock = threading.Lock()
    t0 = time.time()

    def run(job):
        s, lo, hi, at = job
        data = get_range(s, lo, hi)
        os.pwrite(fd, data, at)
        with lock:
            done[0] += hi - lo
            if done[0] // (256 << 20) != (done[0] - (hi - lo)) // (256 << 20):
                el = time.time() - t0
                print(f"  {done[0] / 1e9:6.2f} / {total / 1e9:.2f} GB  {done[0] / el / 1e6:5.1f} MB/s", flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(run, jobs))
    os.fsync(fd)
    os.close(fd)
    os.replace(tmp, OUT)
    print(f"done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
