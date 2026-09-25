"""Python binding for the Krea 2 Turbo Metal engine (engine/build/libkrea.dylib).

Handles prompt templating + tokenization (HF `tokenizers`, identical ids to the official encoder.py:
system prefix + prompt truncated to 541 tokens, then the 5 suffix tokens; the official padding sits between
the two, is masked and takes no position, so it is simply left out) and calls the native engine through
ctypes.

KREA_BUILD=<dir> loads engine/<dir>/libkrea.dylib (and the engine then reads that build's metallib)
instead of engine/build, so a development build can run next to the installed one.
"""
import ctypes
import os
import threading

from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREFIX = ("<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, "
          "spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n")
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
MAX_PREFIX_PROMPT = 512 + 34 - 5  # encoder.py: max_length for prefix + prompt
PRESETS = {"quality": (8, False), "fast": (4, True)}  # steps, 4-step LoRA


def build_dir(root=ROOT):
    """engine/<KREA_BUILD> (default engine/build; an absolute path is used as is), as in the engine."""
    return os.path.join(root, "engine", os.environ.get("KREA_BUILD") or "build")


class Preview(ctypes.Structure):
    _fields_ = [("width", ctypes.c_int), ("height", ctypes.c_int), ("channels", ctypes.c_int), ("step", ctypes.c_int),
                ("hq", ctypes.c_int), ("x0", ctypes.c_void_p), ("noisy", ctypes.c_void_p)]


PROGRESS_FN = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_double,
                               ctypes.POINTER(Preview))


class Params(ctypes.Structure):
    _fields_ = [("width", ctypes.c_int), ("height", ctypes.c_int), ("steps", ctypes.c_int),
                ("seed", ctypes.c_uint64), ("cfg_scale", ctypes.c_float), ("ane", ctypes.c_int),
                ("preview", ctypes.c_int), ("hq_preview_every", ctypes.c_int), ("latent_out", ctypes.c_void_p),
                ("fast", ctypes.c_int), ("init_rgba", ctypes.c_void_p), ("mask", ctypes.c_void_p),
                ("start_step", ctypes.c_int)]


class Stats(ctypes.Structure):
    _fields_ = [("encode_ms", ctypes.c_double), ("denoise_ms", ctypes.c_double), ("decode_ms", ctypes.c_double),
                ("total_ms", ctypes.c_double), ("step_ms", ctypes.c_double), ("text_tokens", ctypes.c_int),
                ("cached_prompt", ctypes.c_int)]


class Tokenizer2:
    def __init__(self, root=ROOT):
        self.tok = Tokenizer.from_file(os.path.join(root, "weights", "engine", "tokenizer.json"))
        self.suffix = self.tok.encode(SUFFIX, add_special_tokens=False).ids

    def __call__(self, prompt):
        ids = self.tok.encode(PREFIX + prompt, add_special_tokens=False).ids[:MAX_PREFIX_PROMPT]
        return ids + self.suffix


class Engine:
    def __init__(self, root=ROOT):
        self.lib = ctypes.CDLL(os.path.join(build_dir(root), "libkrea.dylib"))
        L = self.lib
        L.krea_create.restype = ctypes.c_void_p
        L.krea_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
        for fn in ("krea_ane_available", "krea_fast_available", "krea_gpu_only_available"):
            getattr(L, fn).restype = ctypes.c_int
            getattr(L, fn).argtypes = [ctypes.c_void_p]
        L.krea_generate.restype = ctypes.c_int
        L.krea_generate.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int,
            ctypes.POINTER(Params), ctypes.c_char_p, ctypes.POINTER(Stats), PROGRESS_FN, ctypes.c_void_p,
            ctypes.c_char_p, ctypes.c_int]
        L.krea_prepare_prompt.restype = ctypes.c_int
        L.krea_prepare_prompt.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_char_p,
                                          ctypes.c_int]
        self.tokenize = Tokenizer2(root)
        err = ctypes.create_string_buffer(1024)
        self.h = L.krea_create(root.encode(), err, 1024)
        if not self.h:
            raise RuntimeError(err.value.decode())
        self.lock = threading.Lock()
        self.ane_available = bool(L.krea_ane_available(self.h))
        self.fast_available = bool(L.krea_fast_available(self.h))
        self.gpu_only_available = bool(L.krea_gpu_only_available(self.h))
        L.krea_destroy.argtypes = [ctypes.c_void_p]

    def close(self):
        """Free the engine (GPU buffers, Neural Engine programs) now instead of at process exit."""
        with self.lock:
            if self.h:
                self.lib.krea_destroy(self.h)
                self.h = None

    def prepare(self, prompt):
        """Encode a prompt ahead of generate() (text encoder + text fusion, kept in the prompt cache).
        Returns "prepared" or "cached"."""
        ids = self.tokenize(prompt)
        ia = (ctypes.c_int * len(ids))(*ids)
        err = ctypes.create_string_buffer(1024)
        with self.lock:
            rc = self.lib.krea_prepare_prompt(self.h, ia, len(ids), err, 1024)
        if rc < 0:
            raise RuntimeError(err.value.decode())
        return "cached" if rc == 1 else "prepared"

    def generate(self, prompt, width=1024, height=1024, steps=8, seed=0, fast=False, cfg_scale=0.0, negative_prompt="",
                 ane=True, progress=None, preview=False, hq_preview_every=0, return_latent=False,
                 init_image=None, mask=None, start_step=0):
        """Returns (PIL.Image RGB, stats dict). progress(stage, step, total, elapsed_ms, preview) -> truthy
        cancels; preview is None or a dict {width, height, channels, step, hq, x0: bytes, noisy: bytes|None}.
        With return_latent, stats["latent"] holds the final packed latent (float32 bytes [H/16 * W/16, 64]).

        Editing: init_image (PIL) is the source, resized to width x height; start_step (1 .. steps-1) is the first
        denoising step run, so a later start keeps more of the source; mask (PIL, white = regenerate, same size)
        limits the change to a region and keeps the source pixels elsewhere. Progress then counts the steps run."""
        from PIL import Image

        ids = self.tokenize(prompt)
        neg = self.tokenize(negative_prompt) if cfg_scale > 0 else []
        w, h = width // 16 * 16, height // 16 * 16
        init_buf = mask_buf = None
        if init_image is not None:
            src = init_image.convert("RGB")
            if src.size != (w, h):
                src = src.resize((w, h), Image.LANCZOS)
            init_buf = ctypes.create_string_buffer(src.convert("RGBA").tobytes(), w * h * 4)
            if mask is not None:
                mk = mask.convert("L")
                if mk.size != (w, h):
                    mk = mk.resize((w, h), Image.BILINEAR)
                mask_buf = ctypes.create_string_buffer(mk.tobytes(), w * h)
        out = ctypes.create_string_buffer(w * h * 4)
        st = Stats()
        err = ctypes.create_string_buffer(1024)

        def cb(_user, stage, step, total, ms, pv):
            if progress is None:
                return 0
            try:
                prev = None
                if pv:
                    v = pv.contents
                    n = v.width * v.height * v.channels
                    prev = {"width": v.width, "height": v.height, "channels": v.channels, "step": v.step,
                            "hq": bool(v.hq), "x0": ctypes.string_at(v.x0, n),
                            "noisy": ctypes.string_at(v.noisy, n) if v.noisy else None}
                return 1 if progress(stage.decode(), step, total, ms, prev) else 0
            except Exception:
                import traceback
                traceback.print_exc()
                return 0

        cfn = PROGRESS_FN(cb)
        lat = (ctypes.c_float * ((h // 16) * (w // 16) * 64))() if return_latent else None
        steps = 4 if fast else steps
        p = Params(w, h, steps, seed & (2**64 - 1), cfg_scale, 1 if ane else 0, 1 if preview else 0,
                   max(0, int(hq_preview_every)), ctypes.cast(lat, ctypes.c_void_p) if lat is not None else None,
                   1 if fast else 0, ctypes.cast(init_buf, ctypes.c_void_p) if init_buf is not None else None,
                   ctypes.cast(mask_buf, ctypes.c_void_p) if mask_buf is not None else None,
                   int(start_step) if init_buf is not None else 0)
        ia = (ctypes.c_int * len(ids))(*ids)
        na = (ctypes.c_int * max(1, len(neg)))(*neg) if neg else None
        with self.lock:
            rc = self.lib.krea_generate(self.h, ia, len(ids), na, len(neg), ctypes.byref(p), out, ctypes.byref(st),
                                        cfn, None, err, 1024)
        if rc < 0:
            raise RuntimeError(err.value.decode())
        if rc == 1:
            return None, {"cancelled": True}
        img = Image.frombuffer("RGBA", (w, h), out.raw, "raw", "RGBA", 0, 1).convert("RGB")
        stats = {k: getattr(st, k) for k, _ in Stats._fields_}
        stats.update(width=w, height=h, steps=steps, seed=seed, fast=bool(fast), cfg_scale=cfg_scale)
        if init_buf is not None:
            stats.update(start_step=int(start_step), masked=mask_buf is not None)
        if lat is not None:
            stats["latent"] = bytes(lat)
        return img, stats
