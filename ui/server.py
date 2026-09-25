"""Local web UI for the Krea 2 Turbo Metal engine.

  python ui/server.py            # then open http://127.0.0.1:7861

One generation runs at a time (the engine owns the whole GPU/ANE); further requests queue. Images and
their settings are saved to outputs/ (PNG + JSON sidecar).

Editing (image-to-image / inpainting; Krea 2 has no instruction-editing mode): /api/generate with an "edit"
object {"file": <an image in the outputs folder> | "image": <data URL>, "mask": <data URL, white = change> | null,
"start_step": k} VAE-encodes the source, noises it to sigma[k] and denoises from there (a later start keeps more
of it); with a mask only that area changes and the source pixels are kept elsewhere.

Presets: one process holds one preset's DiT weights (~16-18 GB of memory that can't be paged out; two would
pass macOS's wired-memory limit). A generation with the other preset restarts the server in place: the engine
is freed, the server waits until macOS reports the memory released, then re-executes itself (same pid and port)
with --preset and the pending jobs, which resume under the same ids (the UI keeps polling them).

While the user types, the UI posts the prompt to /api/prepare: when no generation is running or
queued, the worker encodes it ahead (text encoder + text fusion, cached in the engine), so Generate
starts denoising right away. A newer prepare replaces a pending one. /api/status reports "busy" while
either runs (so other GPU work can wait for it), "generating" for a generation only.

Every request passes the content-filter hook (ui/content_filter.py; required by the Krea 2 Turbo model card):
the prompt before any work, the image before it is saved or shown (edit sources included: a blocked prompt never
reaches the engine, and the edited image is checked like any other).

Everything the process prints (including the native engine's stderr) is also written to
logs/server.log. logs/server.state.json records {pid, port} while the server runs so a launcher
(the macOS app) can reuse it; POST /api/shutdown stops it. With --parent-pid the server exits on
its own when that process disappears, so it can never be left orphaned.
"""
import argparse
import gc
import io
import json
import os
import queue
import random
import re
import signal
import sys
import threading
import time
import traceback
import uuid
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "outputs")
LOGS = os.path.join(ROOT, "logs")
STATE_FILE = os.path.join(LOGS, "server.state.json")
SERVER_NAME = "krea-image-studio"
sys.path.insert(0, HERE)

import content_filter  # noqa: E402

SAFE_IMAGE_NAME = re.compile(r"^[\w.-]+\.png$")  # plain file names inside OUT only
PRESETS = {"quality": {"steps": 8, "fast": False}, "fast": {"steps": 4, "fast": True}}


class State:
    def __init__(self):
        self.engine = None
        self.engine_state = "loading"
        self.engine_error = ""
        self.load_seconds = 0.0
        self.jobs = {}
        self.order = []
        self.q = queue.Queue()
        self.lock = threading.Lock()
        self.cv = threading.Condition()  # wakes the worker (new job / prepare) and /api/prepare waiters
        self.prep = None      # pending prepare request (at most one: newer requests replace it)
        self.prep_cur = None  # prepare request being encoded
        self.preps = {}       # recent prepare requests by id
        self.ane_available = False
        self.fast_available = False
        self.gpu_only_available = False
        self.preset = "quality"  # the preset whose weights this process loads
        self.switching = False   # a restart into the other preset is under way
        self.argv = []           # command line for the restart (without --preset / --resume)
        self.orig_fds = None     # stdout / stderr before setup_logging (restored before exec)


S = State()

PREP_FINAL = ("prepared", "cached", "stopped", "superseded", "error", "unsupported", "blocked")


def load_engine():
    t = time.time()
    try:
        from krea import Engine

        # up to 60 s for memory: after a preset switch the previous engine's memory may still be draining
        S.engine = Engine(ROOT, preset=S.preset, memory_timeout=60.0)
        S.ane_available = S.engine.ane_available
        S.fast_available = S.engine.fast_available
        S.gpu_only_available = S.engine.gpu_only_available
        S.engine_state = "ready"
    except Exception as e:  # surfaced in the UI
        S.engine_state = "error"
        S.engine_error = str(e)
        traceback.print_exc()
    S.load_seconds = time.time() - t
    print(f"[server] engine {S.engine_state} ({S.preset} preset) in {S.load_seconds:.1f}s; content filter: "
          f"{content_filter.describe()}", flush=True)


def next_work():
    """Block until there is work: the next queued generation, else the pending prepare (low priority:
    it never starts while a generation is queued)."""
    with S.cv:
        while True:
            if not S.q.empty():
                return S.q.get_nowait(), None
            if S.prep is not None:
                req, S.prep = S.prep, None
                req["state"] = "running"
                S.prep_cur = req
                return None, req
            S.cv.wait()


def worker():
    while True:
        job, prep = next_work()
        if prep is not None:
            run_prepare(prep)
        else:
            run_job(job)


def run_prepare(req):
    t0 = time.time()
    try:
        r = {"state": S.engine.prepare(req["prompt"])}
    except Exception as e:
        traceback.print_exc()
        r = {"state": "error", "error": str(e)}
    r["ms"] = (time.time() - t0) * 1e3
    with S.cv:
        req.update(r)
        S.prep_cur = None
        S.cv.notify_all()
    if r["state"] == "prepared":
        print(f"[server] prompt encoded ahead in {r['ms']:.0f} ms", flush=True)


def prepare_available():
    return S.engine_state == "ready"


def schedule_prepare(prompt):
    """Queue a prepare for this prompt, replacing the pending one; a request for the prompt already
    pending or being encoded shares that request."""
    with S.cv:
        if S.prep is not None and S.prep["prompt"] == prompt:
            return S.prep
        if S.prep is not None:
            S.prep["state"] = "superseded"
            S.prep = None
            S.cv.notify_all()
        if S.prep_cur is not None and S.prep_cur["prompt"] == prompt:
            return S.prep_cur
        req = {"id": uuid.uuid4().hex[:10], "prompt": prompt, "state": "queued", "created": time.time()}
        S.prep = req
        S.preps[req["id"]] = req
        while len(S.preps) > 32:
            S.preps.pop(next(iter(S.preps)))
        S.cv.notify_all()
        return req


def prepare_reply(req, wait=0):
    """A prepare request's state for the client, once it settled or after `wait` seconds (max 30)."""
    try:
        deadline = time.time() + max(0.0, min(30.0, float(wait or 0)))
    except (TypeError, ValueError):
        deadline = time.time()
    with S.cv:
        while req["state"] not in PREP_FINAL and time.time() < deadline:
            S.cv.wait(deadline - time.time())
        out = {k: req[k] for k in ("id", "state", "ms", "error") if k in req}
    out["ready"] = out["state"] in ("prepared", "cached")
    return out


def run_job(job):
    if job["cancel"]:
        job["state"] = "cancelled"
        return
    while S.engine_state == "loading":
        time.sleep(0.2)
    if S.engine_state != "ready":
        job.update(state="error", error=S.engine_error or "engine not available")
        return
    if job["params"]["preset"] != S.preset:
        switch_preset(job)  # does not return: the process restarts with the job's preset
    job["state"] = "running"
    job["started"] = time.time()
    p = job["params"]
    src, mask = job.pop("_edit", (None, None))  # the edit source and mask (PIL), dropped with this call

    def progress(stage, step, total, ms, preview=None):
        job.update(stage=stage, step=step, total=total, elapsed_ms=ms)
        if preview is not None:  # keep only the latest (raw pixels; PNG-encoded on request)
            preview["seq"] = time.monotonic_ns()
            if preview["hq"]:
                job["_hq"] = preview
                job["preview_hq_step"] = preview["step"]
            else:
                job["_pv"] = preview
                job["preview_step"] = preview["step"]
                job["preview_size"] = [preview["width"], preview["height"]]
        if stage == "denoise" and step > 0:
            job["step_ms"] = (ms - job.setdefault("denoise_t0", ms)) / max(step - job.setdefault("denoise_s0", step), 1)
        if stage == "denoise" and "denoise_t0" not in job:
            job["denoise_t0"], job["denoise_s0"] = ms, step
        return job["cancel"]

    try:
        ok, reason = content_filter.check_prompt(p["prompt"])
        if not ok:
            print("[server] prompt blocked by the content filter", flush=True)
            job.update(state="error", error=reason)
            return
        img, stats = S.engine.generate(p["prompt"], p["width"], p["height"], p["steps"], p["seed"], fast=p["fast"],
                                       ane=p.get("ane", True), progress=progress, preview=p.get("preview", True),
                                       hq_preview_every=p.get("hq_preview_every", 0), init_image=src, mask=mask,
                                       start_step=p["edit"]["start_step"] if src is not None else 0)
        if img is None:
            job["state"] = "cancelled"
            return
        ok, reason = content_filter.check_image(img, p["prompt"])
        if not ok:
            print("[server] image blocked by the content filter (not saved)", flush=True)
            job.update(state="error", error=reason)
            for k in ("_pv", "_hq"):
                job.pop(k, None)
            return
        os.makedirs(OUT, exist_ok=True)
        name = time.strftime("%Y%m%d-%H%M%S") + f"-{p['seed']}" + ("-edit" if src is not None else "")
        img.save(os.path.join(OUT, name + ".png"))
        meta = {"prompt": p["prompt"], **{k: v for k, v in p.items() if k != "prompt"}, "stats": stats,
                "file": name + ".png", "created": time.time(), "model": "Krea 2 Turbo"}
        with open(os.path.join(OUT, name + ".json"), "w") as f:
            json.dump(meta, f, indent=1)
        job.update(state="done", result=meta)
    except Exception as e:
        traceback.print_exc()
        job.update(state="error", error=str(e))


def switch_preset(job):
    """Load the job's preset by restarting this process (never two weight sets in memory): free the engine, wait
    until macOS has released its memory, save the pending jobs, exec this server again with --preset/--resume."""
    from krea import wired_bytes

    new = job["params"]["preset"]
    job.update(state="running", stage="loading", step=0, total=1)
    print(f"[server] switching preset {S.preset} -> {new}: restarting the engine", flush=True)
    with S.cv:
        S.switching = True
        pending = [job]
        while not S.q.empty():
            j = S.q.get_nowait()
            if j["state"] == "queued" and not j["cancel"]:
                pending.append(j)
        S.prep = None
    w0, t0 = wired_bytes(), time.time()
    eng, S.engine, S.engine_state = S.engine, None, "loading"
    try:
        eng.close()
    except Exception:
        traceback.print_exc()
    del eng
    gc.collect()
    last = w0
    while time.time() - t0 < 30:  # until it dropped by most of a weight set, or stopped dropping
        time.sleep(0.5)
        w = wired_bytes()
        if w <= w0 - 8e9 or (time.time() - t0 > 3 and w >= last - 0.05e9):
            break
        last = w
    print(f"[server] engine freed: wired {w0 / 1e9:.1f} -> {wired_bytes() / 1e9:.1f} GB after {time.time() - t0:.1f}s",
          flush=True)
    rdir = os.path.join(LOGS, "resume")
    os.makedirs(rdir, exist_ok=True)
    items = []
    for j in pending:
        it = {k: j[k] for k in ("id", "params", "created")}
        src, mask = j.get("_edit", (None, None))
        if src is not None:
            it["src"] = os.path.join(rdir, j["id"] + "-src.png")
            src.save(it["src"])
            if mask is not None:
                it["mask"] = os.path.join(rdir, j["id"] + "-mask.png")
                mask.save(it["mask"])
        items.append(it)
    path = os.path.join(rdir, "jobs.json")
    with open(path, "w") as f:
        json.dump({"preset": new, "jobs": items}, f)
    restart(["--preset", new, "--resume", path])


def restart(extra):
    """Re-execute this server in place (same pid, so a launcher such as the macOS app keeps supervising it)."""
    print(f"[server] restarting: {' '.join(extra)}", flush=True)
    time.sleep(0.3)  # let the log pump drain
    if S.orig_fds:
        os.dup2(S.orig_fds[0], 1)
        os.dup2(S.orig_fds[1], 2)
    os.execv(sys.executable, [sys.executable, "-u"] + S.argv + extra)


def resume_jobs(path):
    """Re-queue the jobs a preset switch carried over (same ids: the UI keeps polling them)."""
    from PIL import Image

    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return
    for it in data.get("jobs", []):
        p = it["params"]
        job = {"id": it["id"], "state": "queued", "params": p, "created": it["created"], "cancel": False,
               "stage": "loading", "step": 0, "total": p["steps"] - p.get("edit", {}).get("start_step", 0),
               "elapsed_ms": 0}
        if it.get("src"):
            src = Image.open(it["src"]).convert("RGB")
            mask = Image.open(it["mask"]).convert("L") if it.get("mask") else None
            job["_edit"] = (src, mask)
        with S.lock:
            S.jobs[job["id"]] = job
            S.order.append(job["id"])
        S.q.put(job)
    for fn in os.listdir(os.path.dirname(path)):
        os.remove(os.path.join(os.path.dirname(path), fn))
    print(f"[server] resumed {len(data.get('jobs', []))} job(s) after the preset switch", flush=True)


def history(limit=48):
    if not os.path.isdir(OUT):
        return []
    items = []
    for fn in sorted(os.listdir(OUT), reverse=True):
        if fn.endswith(".json"):
            try:
                with open(os.path.join(OUT, fn)) as f:
                    items.append(json.load(f))
            except Exception:
                pass
        if len(items) >= limit:
            break
    return items


def decode_data_url(url, what):
    """PIL image from a data: URL (PNG / JPEG / WebP ...)."""
    import base64

    from PIL import Image

    if not isinstance(url, str) or not url.startswith("data:") or "," not in url:
        raise ValueError(f"the {what} must be a data: URL")
    try:
        img = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
        img.load()
    except Exception:
        raise ValueError(f"couldn't read the {what}")
    return img


def parse_edit(e, steps):
    """Validate an edit request -> {"image": PIL RGB, "mask": PIL L | None, "width", "height", "info"}.
    The size follows the source: scaled down to fit 2048 px and rounded down to multiples of 16."""
    from PIL import Image

    if e.get("file"):
        fn = str(e["file"])
        if fn != os.path.basename(fn) or not SAFE_IMAGE_NAME.match(fn) or not os.path.isfile(os.path.join(OUT, fn)):
            raise ValueError("no such image in the outputs folder")
        src, source = Image.open(os.path.join(OUT, fn)), fn
        src.load()
    else:
        src, source = decode_data_url(e.get("image"), "source image"), "upload"
    src = src.convert("RGB")
    w, h = src.size
    scale = min(1.0, 2048 / max(w, h))
    w, h = int(w * scale) // 16 * 16, int(h * scale) // 16 * 16
    if min(w, h) < 256:
        raise ValueError("the image is too small to edit (at least 256 px per side)")
    if (w, h) != src.size:
        src = src.resize((w, h), Image.LANCZOS)
    mask = None
    if e.get("mask"):
        mask = decode_data_url(e["mask"], "mask")
        mask = (mask.getchannel("A") if mask.mode in ("RGBA", "LA") else mask.convert("L")).resize((w, h), Image.BILINEAR)
        if mask.point(lambda v: 255 if v >= 128 else 0).getbbox() is None:
            mask = None  # nothing painted: edit the whole image
    k = clamp_int(e.get("start_step"), 0, steps - 1, max(1, steps // 2))
    if mask is None and k == 0:
        k = 1  # without a mask, step 0 would ignore the source entirely
    return {"image": src, "mask": mask, "width": w, "height": h,
            "info": {"source": source, "start_step": k, "masked": mask is not None}}


def clamp_int(v, lo, hi, default):
    try:
        return max(lo, min(hi, int(v)))
    except (TypeError, ValueError):
        return default


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=os.path.join(HERE, "static"), **kw)

    def log_message(self, fmt, *args):
        pass

    def end_headers(self):
        # The UI is a single evolving page; never let the browser serve a stale copy.
        if not self.path.startswith("/outputs/"):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            print(f"[server] UI page served ({self.headers.get('User-Agent', '?')[:60]})", flush=True)
        if path == "/api/status":
            running = [j for j in S.jobs.values() if j["state"] == "running"]
            return self.send_json({"server": SERVER_NAME, "pid": os.getpid(), "engine": S.engine_state,
                                   "error": S.engine_error, "load_seconds": S.load_seconds,
                                   "busy": bool(running) or S.prep_cur is not None, "generating": bool(running),
                                   "queued": S.q.qsize(), "ane_available": S.ane_available,
                                   "fast_available": S.fast_available, "gpu_only_available": S.gpu_only_available,
                                   "preset": S.preset, "switching": S.switching,
                                   "prepare_available": prepare_available(), "preparing": S.prep_cur is not None,
                                   "preview_available": os.path.exists(os.path.join(ROOT, "weights", "engine", "latent_rgb.json")),
                                   "content_filter": content_filter.describe()})
        if path.startswith("/api/prepare/"):
            req = S.preps.get(path.rsplit("/", 1)[-1])
            if not req:
                return self.send_json({"error": "no such prepare request"}, 404)
            wait = parse_qs(urlparse(self.path).query).get("wait", ["0"])[0]
            return self.send_json(prepare_reply(req, wait))
        if path.startswith("/api/job/"):
            job = S.jobs.get(path.rsplit("/", 1)[-1])
            if not job:
                return self.send_json({"error": "no such job"}, 404)
            pos = sum(1 for jid in S.order if S.jobs[jid]["state"] == "queued" and S.jobs[jid]["created"] < job["created"])
            return self.send_json({k: v for k, v in job.items() if k != "cancel" and not k.startswith("_")}
                                  | {"queue_position": pos})
        if path.startswith("/api/preview/"):
            # latest live preview of a job as PNG: ?kind=x0 (predicted image) | noisy | hq (VAE decode)
            job = S.jobs.get(path.rsplit("/", 1)[-1])
            kind = parse_qs(urlparse(self.path).query).get("kind", ["x0"])[0]
            pv = job and (job.get("_hq") if kind == "hq" else job.get("_pv"))
            if not pv:
                return self.send_error(HTTPStatus.NOT_FOUND)
            data = preview_png(pv, "noisy" if kind == "noisy" else "x0")
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if path == "/api/history":
            return self.send_json(history())
        if path.startswith("/outputs/"):
            fn = os.path.basename(path)
            fp = os.path.join(OUT, fn)
            if not fn.endswith(".png") or not os.path.isfile(fp):
                return self.send_error(HTTPStatus.NOT_FOUND)
            with open(fp, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        return super().do_GET()

    def do_POST(self):
        path = urlparse(self.path).path
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            return self.send_json({"error": "bad json"}, 400)
        if path == "/api/generate":
            if S.switching:
                return self.send_json({"error": "the engine is switching presets; try again in a moment"}, 503)
            prompt = str(body.get("prompt", "")).strip()
            if not prompt:
                return self.send_json({"error": "prompt is empty"}, 400)
            ok, reason = content_filter.check_prompt(prompt)
            if not ok:
                return self.send_json({"error": reason}, 400)
            preset = body.get("preset", "quality")
            if preset not in PRESETS:
                preset = "quality"
            if PRESETS[preset]["fast"] and not S.fast_available:
                return self.send_json({"error": "the Fast preset needs the 4-step LoRA weights (see README)"}, 400)
            seed = body.get("seed")
            seed = random.randint(0, 2**31 - 1) if seed in (None, "", -1) else clamp_int(seed, 0, 2**63 - 1, 0)
            steps = PRESETS[preset]["steps"]
            if preset == "quality" and body.get("steps"):
                steps = clamp_int(body.get("steps"), 1, 32, 8)
            edit = None
            if body.get("edit"):
                try:
                    edit = parse_edit(body["edit"], steps)
                except ValueError as e:
                    return self.send_json({"error": str(e)}, 400)
            params = {
                "prompt": prompt,
                "width": edit["width"] if edit else clamp_int(body.get("width"), 256, 2048, 1024) // 16 * 16,
                "height": edit["height"] if edit else clamp_int(body.get("height"), 256, 2048, 1024) // 16 * 16,
                "preset": preset,
                "steps": steps,
                "fast": PRESETS[preset]["fast"],
                "seed": seed,
                "ane": S.ane_available and (bool(body.get("ane", True)) or not S.gpu_only_available),
                "preview": bool(body.get("preview", True)),
                "hq_preview_every": clamp_int(body.get("hq_preview_every"), 0, 32, 0),
            }
            if edit:
                params["edit"] = edit["info"]
            jid = uuid.uuid4().hex[:10]
            job = {"id": jid, "state": "queued", "params": params, "created": time.time(), "cancel": False,
                   "stage": "queued", "step": 0, "total": params["steps"] - (edit["info"]["start_step"] if edit else 0),
                   "elapsed_ms": 0}
            if edit:
                job["_edit"] = (edit["image"], edit["mask"])
            with S.lock:
                S.jobs[jid] = job
                S.order.append(jid)
            with S.cv:
                S.q.put(job)
                S.cv.notify_all()
            return self.send_json({"id": jid, "params": params})
        if path == "/api/prepare":
            # Encode the prompt ahead of Generate (see the module docstring); "wait": N holds the reply
            # until the request settles (at most N s).
            if not prepare_available():
                return self.send_json({"state": "unsupported", "ready": False})
            prompt = str(body.get("prompt", "")).strip()
            if not prompt:
                return self.send_json({"state": "empty", "ready": False})
            if not content_filter.check_prompt(prompt)[0]:
                return self.send_json({"state": "blocked", "ready": False})
            return self.send_json(prepare_reply(schedule_prepare(prompt), body.get("wait")))
        if path == "/api/delete":
            # Permanently delete a generated image and its JSON sidecar from the outputs folder.
            fn = str(body.get("file", ""))
            if fn != os.path.basename(fn) or not SAFE_IMAGE_NAME.match(fn):
                return self.send_json({"error": "invalid file name"}, 400)
            stem = fn[:-4]
            removed = []
            for name in (stem + ".png", stem + ".json"):
                fp = os.path.join(OUT, name)
                if os.path.isfile(fp):
                    os.remove(fp)
                    removed.append(name)
            if not removed:
                return self.send_json({"error": "no such image"}, 404)
            print(f"[server] deleted {', '.join(removed)}", flush=True)
            return self.send_json({"ok": True, "deleted": removed})
        if path == "/api/shutdown":
            self.send_json({"ok": True})
            threading.Thread(target=shutdown, args=("shutdown requested",), daemon=True).start()
            return
        if path.startswith("/api/cancel/"):
            job = S.jobs.get(path.rsplit("/", 1)[-1])
            if job:
                job["cancel"] = True
                if job["state"] == "queued":
                    job["state"] = "cancelled"
            return self.send_json({"ok": True})
        return self.send_json({"error": "not found"}, 404)


_png_cache = {}


def preview_png(pv, which):
    """Encode a raw preview (RGB8 / RGBA8) as PNG, caching the latest encoding per buffer."""
    from PIL import Image

    raw = pv[which] if pv.get(which) is not None else pv["x0"]
    key = (pv["seq"], which)
    if key not in _png_cache:
        _png_cache.clear()
        img = Image.frombytes("RGBA" if pv["channels"] == 4 else "RGB", (pv["width"], pv["height"]), raw)
        buf = io.BytesIO()
        img.save(buf, "PNG", compress_level=1)
        _png_cache[key] = buf.getvalue()
    return _png_cache[key]


def setup_logging(path, quiet):
    """Send fds 1 and 2 (Python and the native engine) through a pipe that is timestamped into `path`
    and, unless quiet, echoed to the original stdout."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and os.path.getsize(path) > 8 << 20:
        os.replace(path, path + ".1")
    logf = open(path, "ab", buffering=0)
    echo = None if quiet else os.dup(1)
    S.orig_fds = (os.dup(1), os.dup(2))  # restored before a restart (the pipe's reader does not survive exec)
    r, w = os.pipe()
    os.dup2(w, 1)
    os.dup2(w, 2)
    os.close(w)
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    def pump():
        pending = b""
        while True:
            chunk = os.read(r, 65536)
            if not chunk:
                break
            if echo is not None:
                os.write(echo, chunk)
            pending += chunk
            *lines, pending = pending.split(b"\n")
            stamp = time.strftime("[%Y-%m-%d %H:%M:%S] ").encode()
            logf.write(b"".join(stamp + ln + b"\n" for ln in lines))

    threading.Thread(target=pump, daemon=True).start()


def shutdown(reason, code=0):
    """Stop immediately: cancel work, drop the state file, flush logs, exit (frees all engine memory)."""
    print(f"[server] shutting down: {reason}", flush=True)
    for job in list(S.jobs.values()):
        job["cancel"] = True
    try:
        with open(STATE_FILE) as f:
            if json.load(f).get("pid") == os.getpid():
                os.remove(STATE_FILE)
    except (OSError, ValueError):
        pass
    time.sleep(0.2)  # let the log pump drain
    os._exit(code)


def watch_parent(pid):
    """Exit when the launching process disappears (e.g. the app was force-quit)."""
    while True:
        time.sleep(1.0)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            shutdown(f"parent process {pid} exited")
        except PermissionError:
            pass


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--log", default=os.path.join(LOGS, "server.log"), help="log file ('' to disable)")
    ap.add_argument("--quiet", action="store_true", help="log to the file only (no stdout echo)")
    ap.add_argument("--parent-pid", type=int, default=0, help="exit when this process exits")
    ap.add_argument("--outputs", default=OUT, help="where generated images are saved (and listed from)")
    ap.add_argument("--no-engine", action="store_true", help="UI/gallery only, don't load the model (development)")
    ap.add_argument("--preset", choices=list(PRESETS), default=os.environ.get("KREA_PRESET") or "quality",
                    help="weights to load (a generation with the other preset restarts the server with it)")
    ap.add_argument("--resume", default="", help=argparse.SUPPRESS)  # jobs carried over by a preset switch
    a = ap.parse_args()
    S.preset = a.preset
    argv, skip = [], False
    for x in sys.argv:  # the command line minus --preset / --resume, for restarts
        if skip:
            skip = False
        elif x in ("--preset", "--resume"):
            skip = True
        elif not x.startswith(("--preset=", "--resume=")):
            argv.append(x)
    S.argv = argv
    OUT = os.path.abspath(a.outputs)
    if a.log:
        setup_logging(a.log, a.quiet)
    signal.signal(signal.SIGTERM, lambda *_: shutdown("SIGTERM"))
    signal.signal(signal.SIGINT, lambda *_: shutdown("SIGINT"))
    try:
        srv = ThreadingHTTPServer((a.host, a.port), Handler)
    except OSError as e:
        print(f"[server] cannot bind {a.host}:{a.port}: {e}", flush=True)
        shutdown("bind failed", 2)
    # Dev servers (no engine, or a scratch outputs folder) don't advertise themselves for reuse.
    if not a.no_engine and OUT == os.path.join(ROOT, "outputs"):
        os.makedirs(LOGS, exist_ok=True)
        with open(STATE_FILE, "w") as f:
            json.dump({"pid": os.getpid(), "port": srv.server_address[1], "host": a.host, "started": time.time()}, f)
    if a.parent_pid:
        threading.Thread(target=watch_parent, args=(a.parent_pid,), daemon=True).start()
    if a.no_engine:
        S.engine_state, S.engine_error = "error", "engine disabled (--no-engine)"
    else:
        threading.Thread(target=load_engine, daemon=True).start()
    if a.resume:
        resume_jobs(a.resume)
    threading.Thread(target=worker, daemon=True).start()
    print(f"[server] http://{a.host}:{srv.server_address[1]}  pid {os.getpid()}  outputs {OUT}  ({a.preset} preset, "
          "engine loading)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
