// End-to-end text-to-image pipeline + C API.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#import <Foundation/Foundation.h>
#include <dlfcn.h>
#include <sys/stat.h>
#include <unistd.h>

#include "ane.h"
#include "krea.h"
#include "model.h"

namespace krea {

static double now_ms() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Deterministic N(0,1) noise: splitmix64 counter stream + Box-Muller.
static void gaussian_noise(uint64_t seed, float* out, size_t n) {
  auto mix = [](uint64_t z) {
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  };
  for (size_t i = 0; i < n; i += 2) {
    const uint64_t a = mix(seed * 0x9E3779B97F4A7C15ULL + i), b = mix(seed * 0x9E3779B97F4A7C15ULL + i + 1);
    const double u1 = ((a >> 11) + 1.0) / 9007199254740993.0, u2 = (b >> 11) / 9007199254740992.0;
    const double r = std::sqrt(-2.0 * std::log(u1));
    out[i] = (float)(r * std::cos(2 * M_PI * u2));
    if (i + 1 < n) out[i + 1] = (float)(r * std::sin(2 * M_PI * u2));
  }
}

// Build directory of the metallib: engine/<KREA_BUILD> (default engine/build; an absolute path is used as
// is), the same rule as ui/krea.py, so a development build can run next to the installed one.
static std::string build_dir(const std::string& root) {
  const char* b = getenv("KREA_BUILD");
  if (!b || !*b) return root + "/engine/build";
  return b[0] == '/' ? std::string(b) : root + "/engine/" + b;
}

// FNV hash of files (metallib, this binary, weight headers): part of every prompt-cache key, so a rebuilt
// engine or reconverted weights never reuse text features computed by other code or weights.
static uint64_t fnv_files(const std::vector<std::string>& paths, bool whole) {
  uint64_t h = 1469598103934665603ULL;
  for (const std::string& path : paths) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) continue;
    std::vector<unsigned char> buf(1 << 16);
    size_t total = 0;
    for (size_t n; (n = fread(buf.data(), 1, buf.size(), f)) > 0 && (whole || total < (1 << 20));) {
      for (size_t i = 0; i < n; i++) h = (h ^ buf[i]) * 1099511628211ULL;
      total += n;
    }
    fclose(f);
  }
  return h;
}

static std::string self_path() {
  Dl_info info;
  if (dladdr((const void*)&fnv_files, &info) && info.dli_fname) return info.dli_fname;
  return "";
}

static bool exists(const std::string& p) { return access(p.c_str(), R_OK) == 0; }

// ------------------------------------------------------------------------------------------
class Engine {
 public:
  explicit Engine(const std::string& root) : root_(root), m_(build_dir(root) + "/krea.metallib") {
    const std::string e = root + "/weights/engine/";
    vae_ = std::make_unique<VAEDecoder>(m_, e + "vae.qw");
    tf_ = std::make_unique<TextFusion>(m_, e + "txt.qw");
    cond_ = std::make_unique<Conditioning>(m_, e + "cond.qw");
    load_latent_rgb();
    // Default to the GPU+ANE hybrid when its weights exist; otherwise GPU-only.
    load_dit(false, ane_available(false) && !getenv("KREA_GPU_ONLY"));
    kv_build_ = fnv_files({build_dir(root) + "/krea.metallib", self_path()}, true) ^
                fnv_files({e + "te.qw", e + "txt.qw"}, false);
    dit_->weights().prefetch();
  }

  bool ane_available(bool fast = false) const { return !split_path(fast).empty(); }
  bool gpu_only_available() const { return !dit_path(false).empty(); }
  bool fast_available() const { return !dit_path(true).empty() || ane_available(true); }

  int generate(const std::vector<int>& ids, const std::vector<int>& neg, const krea_params& p, uint8_t* out,
               krea_stats* st, krea_progress_fn cb, void* user) {
    const double t0 = now_ms();
    krea_stats s{};
    auto report = [&](const char* stage, int i, int n, const krea_preview* pv = nullptr) {
      return cb ? cb(user, stage, i, n, now_ms() - t0, pv) : 0;
    };
    const bool fast = p.fast != 0;
    if (fast && !fast_available()) throw std::runtime_error("the 4-step LoRA weights are not installed");
    // GPU-only needs the full DiT; without it the hybrid is the only mode.
    const bool want_ane = ane_available(fast) && (p.ane || dit_path(fast).empty());
    if (fast != fast_ || want_ane != (ane_ != nullptr)) {
      if (report("loading", 0, 1)) return 1;
      load_dit(fast, want_ane);
      dit_->weights().prefetch();
    }
    const int H16 = p.height / 16, W16 = p.width / 16, M = H16 * W16;
    if (H16 < 2 || W16 < 2) throw std::runtime_error("image too small");
    const int steps = std::max(1, p.steps);
    const bool cfg = p.cfg_scale > 0 && !neg.empty();

    // ---- 1. text path (prompt cache: text encoder + text fusion + txtmlp output) ----
    if (report("encode", 0, 1)) return 1;
    double t1 = now_ms();
    int T = 0, Tn = 0;
    bool hit = false;
    Tensor txt = text_features(ids, &T, &hit), ntxt;
    if (cfg) ntxt = text_features(neg, &Tn, nullptr);
    s.encode_ms = now_ms() - t1;
    s.text_tokens = T;
    s.cached_prompt = hit;
    if (report("encode", 1, 1)) return 1;

    // ---- 2. schedule and per-step conditioning ----
    const std::vector<float> sig = krea_sigmas(fast ? 4 : steps);
    const int n_steps = (int)sig.size() - 1;
    std::vector<StepCond> conds(n_steps);
    {
      std::vector<float> tv, tvec;
      for (int i = 0; i < n_steps; i++) {
        cond_->get(fast, sig[i], tv, tvec);
        dit_->make_cond(tv.data(), tvec.data(), conds[i]);
      }
    }

    // ---- 3. denoising loop ----
    Tensor z = m_.alloc((size_t)M * kLatC * 4), vel = m_.alloc((size_t)M * kLatC * 4), vel2;
    if (cfg) vel2 = m_.alloc((size_t)M * kLatC * 4);
    gaussian_noise(p.seed, z.ptr<float>(), (size_t)M * kLatC);
    const bool fastpv = p.preview != 0;
    const int hq_every = std::max(0, p.hq_preview_every);
    std::vector<float> zs, vs;
    std::vector<uint8_t> px0, pnz, phq;
    Tensor hqz;
    if (fastpv) {
      zs.resize((size_t)M * kLatC);
      vs.resize((size_t)M * kLatC);
      px0.resize((size_t)M * 4 * 3);
      pnz.resize((size_t)M * 4 * 3);
    }
    if (hq_every) {
      hqz = m_.alloc((size_t)M * kLatC * 4);
      phq.resize((size_t)M * 256 * 4);
    }
    int pending = -1;
    auto emit = [&](int done) {
      project_preview(zs.data(), vs.data(), sig[done], H16, W16, px0.data(), pnz.data());
      const krea_preview pv{2 * W16, 2 * H16, 3, done, 0, px0.data(), pnz.data()};
      return report("denoise", done, n_steps, &pv);
    };
    int cancel = report("denoise", 0, n_steps);
    t1 = now_ms();
    for (int i = 0; i < n_steps && !cancel; i++) {
      m_.begin();
      if (cfg) {  // the text length differs between the two passes: RoPE tables per pass
        dit_->set_grid(Tn, H16, W16);
        dit_->forward(ntxt, z, conds[i], vel2);
        m_.end_wait();
        m_.begin();
      }
      dit_->set_grid(T, H16, W16);
      dit_->forward(txt, z, conds[i], vel);
      if (cfg) {
        struct { int n; float scale; } cp{M * kLatC, p.cfg_scale};
        m_.dispatch_threads("cfg_combine", MTLSizeMake((size_t)M * kLatC, 1, 1), MTLSizeMake(256, 1, 1), {vel, vel2},
                            &cp, sizeof(cp), 2);
      }
      m_.euler_step(z, vel, M * kLatC, sig[i + 1] - sig[i]);
      m_.commit();
      // GPU is now running step i; report step i-1 (with its preview) meanwhile.
      if (pending >= 0) {
        cancel = emit(pending);
        pending = -1;
      } else if (i > 0) {
        cancel = report("denoise", i, n_steps);
      }
      m_.wait();
      if (fastpv) {
        memcpy(zs.data(), z.ptr<float>(), zs.size() * 4);
        memcpy(vs.data(), vel.ptr<float>(), vs.size() * 4);
        pending = i + 1;
      }
      if (hq_every && (i + 1) % hq_every == 0 && i + 1 < n_steps && !cancel) {
        const float* zp = z.ptr<float>();
        const float* vp = vel.ptr<float>();
        float* xp = hqz.ptr<float>();
        for (size_t k = 0; k < (size_t)M * kLatC; k++) xp[k] = zp[k] - sig[i + 1] * vp[k];
        vae_->decode(hqz, H16, W16, phq.data());
        const krea_preview pv{16 * W16, 16 * H16, 4, i + 1, 1, phq.data(), nullptr};
        cancel = report("denoise", i + 1, n_steps, &pv);
      }
    }
    if (!cancel) cancel = pending >= 0 ? emit(pending) : report("denoise", n_steps, n_steps);
    if (cancel) return 1;
    s.denoise_ms = now_ms() - t1;
    s.step_ms = s.denoise_ms / n_steps;
    if (p.latent_out) memcpy(p.latent_out, z.ptr<float>(), (size_t)M * kLatC * 4);

    // ---- 4. VAE decode ----
    t1 = now_ms();
    if (report("decode", 0, 1)) return 1;
    vae_->weights().prefetch();
    vae_->decode(z, H16, W16, out);
    s.decode_ms = now_ms() - t1;
    s.total_ms = now_ms() - t0;
    report("done", 1, 1);
    if (st) *st = s;
    return 0;
  }

  int prepare(const std::vector<int>& ids) {
    int T = 0;
    bool hit = false;
    text_features(ids, &T, &hit);
    return hit ? 1 : 0;
  }

 private:
  // ---- text path + prompt cache ----
  // The text path (Qwen3-VL taps -> TextFusion -> txtmlp) depends only on the token ids: its output
  // f32 [T, 6144] (24 KB per token) is cached per prompt in memory and in <root>/cache/prompts.
  std::string prompt_key(const std::vector<int>& ids) const {
    uint64_t h = 1469598103934665603ULL;
    auto mix = [&](uint64_t v) {
      for (int i = 0; i < 8; i++) {
        h ^= (v >> (8 * i)) & 0xff;
        h *= 1099511628211ULL;
      }
    };
    for (const char* c = "ktxt1"; *c; c++) mix((uint64_t)*c);
    mix(kv_build_);
    if (const char* e = getenv("KREA_TE_LAYERS")) mix(0x4c41594552ULL + (uint64_t)atoi(e));
    for (int id : ids) mix((uint64_t)(uint32_t)id);
    char buf[17];
    snprintf(buf, sizeof(buf), "%016llx", (unsigned long long)h);
    return buf;
  }
  std::string cache_dir() const {
    const char* d = getenv("KREA_CACHE_DIR");
    return d && *d ? std::string(d) : root_ + "/cache/prompts";
  }
  Tensor text_features(const std::vector<int>& ids, int* T, bool* hit) {
    if ((int)ids.size() <= kDrop) throw std::runtime_error("prompt too short");
    const std::string key = prompt_key(ids);
    auto it = mem_.find(key);
    if (it != mem_.end()) {
      it->second.used = ++clock_;
      *T = it->second.T;
      if (hit) *hit = true;
      return it->second.txt;
    }
    const std::string path = cache_dir() + "/" + key + ".txt";
    if (FILE* f = fopen(path.c_str(), "rb")) {
      uint32_t hdr[2] = {0, 0};
      Tensor t;
      if (fread(hdr, 4, 2, f) == 2 && hdr[0] == 0x5458544b && hdr[1] > 0 && hdr[1] <= 1024) {
        t = m_.alloc((size_t)hdr[1] * kD * 4);
        if (fread(t.ptr<void>(), 1, t.bytes, f) != t.bytes) t = Tensor();
      }
      fclose(f);
      if (t) {
        *T = (int)hdr[1];
        remember(key, t, *T);
        if (hit) *hit = true;
        return t;
      }
    }
    if (hit) *hit = false;
    const int Tn = (int)ids.size() - kDrop;
    Tensor taps = m_.alloc((size_t)Tn * kTaps * kTD * 4), txt = m_.alloc((size_t)Tn * kD * 4);
    if (!te_) te_ = std::make_unique<TextEncoder>(m_, root_ + "/weights/engine/te.qw");
    m_.begin();
    te_->encode(ids, kDrop, taps);
    tf_->run(taps, Tn, txt);
    m_.end_wait();
    *T = Tn;
    remember(key, txt, Tn);
    // disk copy (small: 24 KB per token); keep the directory under 256 MB, oldest entries first
    @autoreleasepool {
      NSFileManager* fm = [NSFileManager defaultManager];
      [fm createDirectoryAtPath:@(cache_dir().c_str()) withIntermediateDirectories:YES attributes:nil error:nil];
      const std::string tmp = path + "." + std::to_string(getpid()) + ".tmp";
      if (FILE* f = fopen(tmp.c_str(), "wb")) {
        const uint32_t hdr[2] = {0x5458544b, (uint32_t)Tn};
        bool ok = fwrite(hdr, 4, 2, f) == 2 && fwrite(txt.ptr<void>(), 1, txt.bytes, f) == txt.bytes;
        ok = fclose(f) == 0 && ok;
        if (ok) rename(tmp.c_str(), path.c_str());
        else unlink(tmp.c_str());
      }
      NSArray* names = [fm contentsOfDirectoryAtPath:@(cache_dir().c_str()) error:nil];
      std::vector<std::pair<double, std::pair<std::string, uint64_t>>> files;
      uint64_t total = 0;
      for (NSString* n in names) {
        if (![n hasSuffix:@".txt"]) continue;
        const std::string fp = cache_dir() + "/" + n.UTF8String;
        struct stat sb;
        if (stat(fp.c_str(), &sb) != 0) continue;
        total += sb.st_size;
        files.push_back({(double)sb.st_mtime, {fp, (uint64_t)sb.st_size}});
      }
      std::sort(files.begin(), files.end());
      for (size_t i = 0; i < files.size() && total > (256ull << 20); i++) {
        unlink(files[i].second.first.c_str());
        total -= files[i].second.second;
      }
    }
    release_text_encoder();
    return txt;
  }
  void remember(const std::string& key, const Tensor& t, int T) {
    if (mem_.size() >= 16) {
      auto lru = mem_.begin();
      for (auto it = mem_.begin(); it != mem_.end(); ++it)
        if (it->second.used < lru->second.used) lru = it;
      mem_.erase(lru);
    }
    mem_[key] = Entry{t, T, ++clock_};
  }
  // The streamed text encoder holds only its activations and the prefix K/V between prompts.
  void release_text_encoder() {
    if (getenv("KREA_RELEASE_TE")) te_.reset();
  }

  // ---- DiT weights ----
  std::string dit_path(bool fast) const {
    const std::string e = root_ + "/weights/engine/";
    if (const char* f = getenv(fast ? "KREA_DIT_FAST" : "KREA_DIT")) return exists(e + f) ? e + f : "";
    for (const char* n : fast ? std::vector<const char*>{"dit_fast.qw", "dit_fast_q8.qw"}
                              : std::vector<const char*>{"dit.qw", "dit_q8.qw"})
      if (exists(e + n)) return e + n;
    return "";
  }
  // Hybrid mode: weights/ane[_fast]/meta.json records the GPU/ANE split; dit_h<H1>_c<c0>[_fast].qw holds the
  // GPU's slices (MLP units [0, H1), q|k|v|gate columns [0, c0)); the ANE programs hold the rest.
  std::string ane_dir(bool fast) const { return root_ + (fast ? "/weights/ane_fast" : "/weights/ane"); }
  std::string split_path(bool fast) const {
    NSData* d = [NSData dataWithContentsOfFile:@((ane_dir(fast) + "/meta.json").c_str())];
    NSDictionary* j = d ? [NSJSONSerialization JSONObjectWithData:d options:0 error:nil] : nil;
    if (!j) return "";
    const std::string path = root_ + "/weights/engine/dit_h" + std::to_string([j[@"gpu_units"] intValue]) + "_c" +
                             std::to_string([j[@"qkvg_c0"] intValue]) + (fast ? "_fast" : "") + ".qw";
    return exists(path) ? path : "";
  }
  void load_dit(bool fast, bool hybrid) {
    dit_.reset();
    ane_.reset();  // free the other set's programs before loading
    if (hybrid) {
      dit_ = std::make_unique<DiT>(m_, split_path(fast));
      ane_ = std::make_unique<ANEOffload>(m_, ane_dir(fast));
      dit_->set_ane(ane_.get());
    } else {
      const std::string path = dit_path(fast);
      if (path.empty()) throw std::runtime_error(std::string("DiT weights not found (") + (fast ? "fast" : "base") + ")");
      dit_ = std::make_unique<DiT>(m_, path);
    }
    fast_ = fast;
  }

  // ---- live preview: 16 latent channels -> RGB per latent pixel (H/8 x W/8), fitted by
  // tools/fit_latent_rgb.py against this VAE (normalized latents, RGB in [0, 1]) ----
  float lat_a_[16][3] = {}, lat_b_[3] = {0.5f, 0.5f, 0.5f};
  void load_latent_rgb() {
    NSString* path = [NSString stringWithFormat:@"%s/weights/engine/latent_rgb.json", root_.c_str()];
    NSData* d = [NSData dataWithContentsOfFile:path];
    NSDictionary* j = d ? [NSJSONSerialization JSONObjectWithData:d options:0 error:nil] : nil;
    if (!j) {  // crude fallback until the projection is fitted
      for (int k = 0; k < 3; k++) lat_a_[k][k] = 0.15f;
      return;
    }
    NSArray* w = j[@"weights"];
    NSArray* b = j[@"bias"];
    for (int c = 0; c < 16; c++)
      for (int k = 0; k < 3; k++) lat_a_[c][k] = [w[c][k] floatValue];
    for (int k = 0; k < 3; k++) lat_b_[k] = [b[k] floatValue];
  }
  // x0 = z - sigma v and z itself, projected to RGB8 at latent resolution [2*H16, 2*W16, 3].
  void project_preview(const float* z, const float* v, float sigma, int H16, int W16, uint8_t* x0_rgb,
                       uint8_t* z_rgb) const {
    const int W8 = 2 * W16;
    for (int t = 0; t < H16 * W16; t++) {
      const int ti = t / W16, tj = t % W16;
      for (int sp = 0; sp < 4; sp++) {
        float a[3] = {lat_b_[0], lat_b_[1], lat_b_[2]}, n[3] = {lat_b_[0], lat_b_[1], lat_b_[2]};
        for (int c = 0; c < 16; c++) {
          const size_t k = (size_t)t * kLatC + c * 4 + sp;
          const float x0 = z[k] - sigma * v[k];
          for (int q = 0; q < 3; q++) {
            a[q] += x0 * lat_a_[c][q];
            n[q] += z[k] * lat_a_[c][q];
          }
        }
        const size_t px = (size_t)(2 * ti + (sp >> 1)) * W8 + 2 * tj + (sp & 1);
        for (int q = 0; q < 3; q++) {
          x0_rgb[px * 3 + q] = (uint8_t)std::lround(std::min(std::max(a[q], 0.f), 1.f) * 255.f);
          z_rgb[px * 3 + q] = (uint8_t)std::lround(std::min(std::max(n[q], 0.f), 1.f) * 255.f);
        }
      }
    }
  }

  std::string root_;
  Metal m_;
  std::unique_ptr<TextEncoder> te_;
  std::unique_ptr<TextFusion> tf_;
  std::unique_ptr<DiT> dit_;
  std::unique_ptr<ANEOffload> ane_;
  std::unique_ptr<VAEDecoder> vae_;
  std::unique_ptr<Conditioning> cond_;
  bool fast_ = false;
  struct Entry {
    Tensor txt;
    int T = 0;
    uint64_t used = 0;
  };
  std::map<std::string, Entry> mem_;
  uint64_t clock_ = 0, kv_build_ = 0;
};

}  // namespace krea

struct krea_engine {
  std::unique_ptr<krea::Engine> e;
};

static void set_err(char* err, int n, const char* msg) {
  if (err && n > 0) snprintf(err, n, "%s", msg);
}

extern "C" krea_engine* krea_create(const char* root, char* err, int errlen) {
  @autoreleasepool {
    try {
      auto* h = new krea_engine;
      h->e = std::make_unique<krea::Engine>(root);
      return h;
    } catch (const std::exception& ex) {
      set_err(err, errlen, ex.what());
      return nullptr;
    }
  }
}

extern "C" void krea_destroy(krea_engine* e) { delete e; }
extern "C" int krea_ane_available(krea_engine* e) { return e->e->ane_available() ? 1 : 0; }
extern "C" int krea_fast_available(krea_engine* e) { return e->e->fast_available() ? 1 : 0; }
extern "C" int krea_gpu_only_available(krea_engine* e) { return e->e->gpu_only_available() ? 1 : 0; }

extern "C" int krea_generate(krea_engine* e, const int* ids, int n_ids, const int* neg_ids, int n_neg,
                             const krea_params* p, uint8_t* out_rgba, krea_stats* stats, krea_progress_fn cb,
                             void* user, char* err, int errlen) {
  @autoreleasepool {
    try {
      std::vector<int> a(ids, ids + n_ids), b;
      if (neg_ids && n_neg > 0) b.assign(neg_ids, neg_ids + n_neg);
      return e->e->generate(a, b, *p, out_rgba, stats, cb, user);
    } catch (const std::exception& ex) {
      set_err(err, errlen, ex.what());
      return -1;
    }
  }
}

extern "C" int krea_prepare_prompt(krea_engine* e, const int* ids, int n_ids, char* err, int errlen) {
  @autoreleasepool {
    try {
      return e->e->prepare(std::vector<int>(ids, ids + n_ids));
    } catch (const std::exception& ex) {
      set_err(err, errlen, ex.what());
      return -1;
    }
  }
}
