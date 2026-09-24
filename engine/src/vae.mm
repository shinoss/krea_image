// Qwen-Image VAE decoder (the Wan 2.1 decoder, single frame), as used by Krea 2:
//   unpack + de-normalize + post_quant_conv -> conv_in 16->384 -> mid (res, attention, res) ->
//   up0: 3 res @384, up 384->192 | up1: res 192->384, 2 res @384, up 384->192 | up2: 3 res @192, up 192->96
//   | up3: 3 res @96 -> RMS norm + SiLU -> conv_out 96->3 (+ alpha 1) -> RGBA8.
// The causal 3-D convs reduce to 2-D convs with their last temporal tap (tools/convert.py keeps only
// that tap); the upsamplers' time_conv only runs from the second frame on and is dropped.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "model.h"

namespace krea {

struct VConvParams { int H, W, Cin, Cout, has_res, Cres, ft, repeats; };
struct NormParams { int P, C, silu; };
struct TransParams { int P, C, ld, off, ldo; };
struct SoftmaxParams { int n, ld; float scale; };
struct OutParams { int H, W, Cin; };
struct WinoParams { int H, W, C, Cout, has_res, ty0, th; };

static constexpr size_t BF = 2;

VAEDecoder::VAEDecoder(Metal& m, const std::string& path) : m_(m), w_(m, path) {
  if (const char* e = getenv("KREA_VAE_WINO")) wino_ = atoi(e);
  // Winograd keeps V = B^T d B in fp16. A resblock conv reads an RMS-norm (+ SiLU) output, bounded by
  // sqrt(C) * max|gamma|, so |V| <= 4x that; require a wide margin (else direct convs).
  for (int b = -1; b < 4 && wino_; b++)
    for (int r = 0; r < (b < 0 ? 2 : 3); r++)
      for (int n = 1; n <= 2; n++) {
        const std::string g = (b < 0 ? std::string("decoder.mid_block.") : "decoder.up_blocks." + std::to_string(b) + ".") +
                              "resnets." + std::to_string(r) + ".norm" + std::to_string(n) + ".gamma";
        const Tensor t = w_.get(g);
        const size_t C = t.bytes / 4;
        float mx = 0;
        for (size_t i = 0; i < C; i++) mx = std::max(mx, std::fabs(t.ptr<float>()[i]));
        if (4 * std::sqrt((float)C) * mx > 8192) wino_ = 0;
      }
}

// Scratch buffers are recycled within a decode (a fresh multi-100-MB buffer costs the driver ~15 ms per
// 300 MB to map); release() hands one back once the command buffer that last reads it has completed.
Tensor VAEDecoder::scratch(size_t bytes) {
  bytes = std::max<size_t>(bytes, 16);
  int best = -1;
  for (int i = 0; i < (int)free_.size(); i++) {
    const size_t cap = free_[i].buf.length;
    if (cap >= bytes && cap <= bytes + bytes / 4 && (best < 0 || cap < free_[best].buf.length)) best = i;
  }
  if (best >= 0) {
    Tensor t{free_[best].buf, 0, bytes};
    free_.erase(free_.begin() + best);
    return t;
  }
  free_.erase(std::remove_if(free_.begin(), free_.end(), [&](const Tensor& t) { return t.buf.length < bytes; }),
              free_.end());
  return m_.alloc(bytes);
}

void VAEDecoder::release(const Tensor& t) {
  if (t.buf) pending_.push_back({t, inflight_});
}

void VAEDecoder::trim() {
  free_.clear();
  pending_.clear();
}

static void add_busy(id<MTLCommandBuffer> cb, double& ms, double& end) {
  const double s = std::max(cb.GPUStartTime, end);
  if (cb.GPUEndTime > s) ms += (cb.GPUEndTime - s) * 1e3;
  end = std::max(end, cb.GPUEndTime);
}

// Every op gets its own command buffer; the CPU waits for the previous op's buffer only after committing
// the next one, so the GPU never idles and at most two ops' intermediates are alive.
void VAEDecoder::op_done(const char* op, const Act& y) {
  id<MTLCommandBuffer> cb = m_.command_buffer();
  if (m_.profiling()) {
    char label[64];
    snprintf(label, sizeof(label), "vae %s %dx%d", op, y.W, y.C);
    m_.mark(label);
  } else {
    m_.split();
  }
  retire();
  inflight_ = cb;
}

void VAEDecoder::retire() {
  if (!inflight_) return;
  id<MTLCommandBuffer> cb = inflight_;
  inflight_ = nil;
  [cb waitUntilCompleted];
  if (cb.status == MTLCommandBufferStatusError)
    throw std::runtime_error(std::string("GPU error: ") + (cb.error ? cb.error.localizedDescription.UTF8String : "?"));
  add_busy(cb, gpu_ms_, gpu_end_);
  for (size_t i = 0; i < pending_.size();)
    if (!pending_[i].second || pending_[i].second.status == MTLCommandBufferStatusCompleted) {
      free_.push_back(pending_[i].first);
      pending_.erase(pending_.begin() + i);
    } else {
      i++;
    }
}

// Implicit-GEMM conv (kernels/vae.metal vconv): mode 1 = 1x1, 3 = 3x3, 2 = nearest-2x + 3x3 as four 2x2
// sub-pixel convs (weights from vae_subpix_weights).
void VAEDecoder::conv(int mode, const Act& x, const Tensor& w, const Tensor& b, const Tensor* res, const Tensor& y,
                      int cout) {
  const int bn = cout % 64 == 0 ? 64 : 48;
  if (cout % bn || x.C % 16) throw std::runtime_error("conv: unsupported channels");
  const int N = mode == 2 ? 4 * cout : cout;
  const MTLSize grid = MTLSizeMake(N / bn, (x.H * x.W + 63) / 64, 1);
  const char* kind = mode == 3 ? "3x3" : mode == 1 ? "1x1" : "_up2";
  VConvParams p{x.H, x.W, x.C, cout, res ? 1 : 0, 0, 0, 1};
  m_.dispatch(std::string("vconv") + kind + "_bn" + std::to_string(bn), grid, MTLSizeMake(128, 1, 1),
              {x.t, w, b, res ? *res : y, y}, &p, sizeof(p), 5);
}

VAEDecoder::Act VAEDecoder::conv3x3(const Act& x, const std::string& name, int cout, const Tensor* residual) {
  Act y{scratch((size_t)x.H * x.W * cout * BF), x.H, x.W, cout};
  conv(3, x, w_.get(name + ".weight"), w_.get(name + ".bias"), residual, y.t, cout);
  op_done("conv3x3", y);
  return y;
}

VAEDecoder::Act VAEDecoder::conv1x1(const Act& x, const std::string& name, int cout) {
  Act y{scratch((size_t)x.H * x.W * cout * BF), x.H, x.W, cout};
  conv(1, x, w_.get(name + ".weight"), w_.get(name + ".bias"), nullptr, y.t, cout);
  op_done("conv1x1", y);
  return y;
}

// nearest-2x upsample + 3x3 conv `name` (x.C -> cout) as sub-pixel convs on the low-res input.
VAEDecoder::Act VAEDecoder::upsample(const Act& x, const std::string& name, int cout) {
  Tensor w2 = scratch((size_t)16 * x.C * cout * BF);
  const int wp[2] = {x.C, cout};
  m_.dispatch_threads("vae_subpix_weights", MTLSizeMake(4 * cout / 8, 4 * x.C, 1), MTLSizeMake(32, 8, 1),
                      {w_.get(name + ".weight"), w2}, wp, sizeof(wp), 2);
  Act y{scratch((size_t)4 * x.H * x.W * cout * BF), 2 * x.H, 2 * x.W, cout};
  conv(2, x, w2, w_.get(name + ".bias"), nullptr, y.t, cout);
  op_done("up2 subpixel", y);
  release(w2);
  return y;
}

VAEDecoder::Act VAEDecoder::norm_silu(const Act& x, const std::string& gamma, bool silu) {
  Act y{scratch(x.t.bytes), x.H, x.W, x.C};
  NormParams p{x.H * x.W, x.C, silu ? 1 : 0};
  // C / 72 lanes per pixel (the Qwen engine's tuning): 1 at 96 channels, 2 at 192, 4 at 384
  int L = 1;
  while (L < 16 && 2 * L * 72 <= x.C) L *= 2;
  const size_t sgs = ((size_t)p.P * L + 31) / 32;
  m_.dispatch_threads("vae_norm_silu_l" + std::to_string(L), MTLSizeMake(sgs * 32, 1, 1), MTLSizeMake(256, 1, 1),
                      {x.t, w_.get(gamma), y.t}, &p, sizeof(p), 3);
  op_done("norm", y);
  return y;
}

// Releases its intermediates; the caller owns x.
VAEDecoder::Act VAEDecoder::resblock(const Act& x, const std::string& p, int cout) {
  Act h = x;
  if (x.C != cout) h = conv1x1(x, p + "conv_shortcut", cout);
  Act t1 = norm_silu(x, p + "norm1.gamma", true);
  Act t2 = wino(t1, cout) ? conv3x3_wino(t1, p + "conv1", cout, nullptr) : conv3x3(t1, p + "conv1", cout, nullptr);
  release(t1.t);
  Act t3 = norm_silu(t2, p + "norm2.gamma", true);
  release(t2.t);
  Act y = wino(t3, cout) ? conv3x3_wino(t3, p + "conv2", cout, &h.t) : conv3x3(t3, p + "conv2", cout, &h.t);
  release(t3.t);
  if (h.t.buf != x.t.buf) release(h.t);
  return y;
}

// Winograd F(2x2, 3x3) where it pays: the convs with >= 288 channels (here the 384-channel stages).
bool VAEDecoder::wino(const Act& x, int cout) const {
  return wino_ && (wino_ > 1 || std::min(x.C, cout) >= 288) && x.H % 2 == 0 && x.W % 2 == 0;
}

// Tile rows per band, so that a band's V (fp16) and M (fp32) take at most 4x the conv input, within
// [128, 640] MB.
int VAEDecoder::wino_rows(const Act& x, int cout) const {
  const int TH = x.H / 2;
  const size_t row = (size_t)16 * (x.W / 2) * (x.C * 2 + cout * 4),
               budget = std::clamp<size_t>(4 * (size_t)x.H * x.W * x.C * BF, (size_t)128 << 20, (size_t)640 << 20);
  const int bands = (int)std::min<size_t>(TH, (row * TH + budget - 1) / budget);
  return (TH + bands - 1) / bands;
}

void VAEDecoder::prewire(std::initializer_list<size_t> sizes) {
  std::vector<Tensor> ts;
  for (size_t b : sizes) {
    ts.push_back(scratch(b));
    m_.dispatch_threads("vae_touch", MTLSizeMake(1, 1, 1), MTLSizeMake(1, 1, 1), {ts.back()});
  }
  op_done("prewire", Act{Tensor(), 0, 0, 0});
  for (const Tensor& t : ts) release(t);
}

VAEDecoder::Act VAEDecoder::conv3x3_wino(const Act& x, const std::string& name, int cout, const Tensor* residual) {
  const int TH = x.H / 2, TW = x.W / 2, th = wino_rows(x, cout);
  const int wp[2] = {x.C, cout};
  Tensor U = scratch((size_t)16 * x.C * cout * 2);
  m_.dispatch_threads("vae_wino_weights", MTLSizeMake(cout / 8, x.C, 1), MTLSizeMake(32, 8, 1),
                      {w_.get(name + ".weight"), U}, wp, sizeof(wp), 2);
  Tensor V = scratch((size_t)16 * th * TW * x.C * 2), Mt = scratch((size_t)16 * th * TW * cout * 4);
  Act y{scratch((size_t)x.H * x.W * cout * BF), x.H, x.W, cout};
  const int bn = cout % 64 == 0 ? 64 : 48;
  for (int ty0 = 0; ty0 < TH; ty0 += th) {
    WinoParams p{x.H, x.W, x.C, cout, residual ? 1 : 0, ty0, std::min(th, TH - ty0)};
    const int T = p.th * TW;
    m_.dispatch_threads("vae_wino_in", MTLSizeMake(x.C / 8, T, 1), MTLSizeMake(32, 8, 1), {x.t, V}, &p, sizeof(p), 2);
    m_.dispatch("vae_wino_gemm_bn" + std::to_string(bn), MTLSizeMake(cout / bn, (T + 63) / 64, 16),
                MTLSizeMake(128, 1, 1), {V, U, Mt}, &p, sizeof(p), 3);
    m_.dispatch_threads("vae_wino_out", MTLSizeMake(cout / 8, T, 1), MTLSizeMake(32, 8, 1),
                        {Mt, w_.get(name + ".bias"), residual ? *residual : y.t, y.t}, &p, sizeof(p), 4);
  }
  op_done("wino3x3", y);
  release(U);
  release(V);
  release(Mt);
  return y;
}

// Single-head self-attention over all pixels (C = 384; 16384 pixels at 1024^2), explicit S = Q K^T in fp32
// and a row softmax, in chunks of query rows so that S stays at 128 MB. The key axis is padded to a
// multiple of 64 (GEMM tile); padded keys are excluded from the softmax and their P / V entries are zero.
VAEDecoder::Act VAEDecoder::attention(const Act& x, const std::string& p) {
  const int P = x.H * x.W, C = x.C;
  const int Pp = (P + 63) / 64 * 64;
  Act n = norm_silu(x, p + "norm.gamma", false);
  Act qkv{scratch((size_t)Pp * 3 * C * BF), x.H, x.W, 3 * C};
  memset(qkv.t.ptr<void>(), 0, qkv.t.bytes);
  conv(1, n, w_.get(p + "to_qkv.weight"), w_.get(p + "to_qkv.bias"), nullptr, qkv.t, 3 * C);
  Tensor kt = scratch((size_t)C * Pp * BF);
  memset(kt.ptr<void>(), 0, kt.bytes);
  TransParams tp{P, C, 3 * C, C, Pp};
  m_.dispatch_threads("vae_transpose", MTLSizeMake(P, C, 1), MTLSizeMake(64, 4, 1), {qkv.t, kt}, &tp, sizeof(tp), 2);
  const int Qc = std::min(P, 2048);
  Tensor S = scratch((size_t)Qc * Pp * 4);
  Tensor Pm = scratch((size_t)Qc * Pp * BF);
  memset(Pm.ptr<void>(), 0, Pm.bytes);
  Act o{scratch((size_t)P * C * BF), x.H, x.W, C};
  for (int q0 = 0; q0 < P; q0 += Qc) {
    const int rows = std::min(Qc, P - q0);
    m_.gemm(Epi::F32, qkv.t.at((size_t)q0 * 3 * C * BF), kt, S, rows, Pp, C, 3 * C, Pp, Pp);
    SoftmaxParams sp{P, Pp, (float)(1.0 / std::sqrt((double)C))};
    m_.dispatch("softmax_rows", MTLSizeMake(rows, 1, 1), MTLSizeMake(256, 1, 1), {S, Pm}, &sp, sizeof(sp), 2);
    m_.gemm(Epi::BF16, Pm, qkv.t.at((size_t)2 * C * BF), o.t.at((size_t)q0 * C * BF), rows, C, Pp, Pp, 3 * C, C);
  }
  Act y{scratch(x.t.bytes), x.H, x.W, C};
  conv(1, o, w_.get(p + "proj.weight"), w_.get(p + "proj.bias"), &x.t, y.t, C);
  op_done("attention", y);
  for (const Tensor* t : {&n.t, &qkv.t, &kt, &S, &Pm, &o.t}) release(*t);
  return y;
}

void VAEDecoder::decode(const Tensor& z, int H16, int W16, uint8_t* out_rgba) {
  gpu_ms_ = gpu_end_ = 0;
  try {
    run(z, H16, W16, out_rgba);
  } catch (...) {
    try {
      m_.end_wait();
    } catch (...) {
    }
    if (inflight_) [inflight_ waitUntilCompleted];
    inflight_ = nil;
    trim();
    throw;
  }
}

void VAEDecoder::run(const Tensor& z, int H16, int W16, uint8_t* out_rgba) {
  const int H = 2 * H16, W = 2 * W16, P = H * W;
  trim();
  m_.begin();
  // unpack + de-normalize + post_quant_conv (16 -> 16, 1x1) in one pass: pixels x 16 channels
  Act x{scratch((size_t)P * 16 * BF), H, W, 16};
  m_.dispatch_threads("vae_unpack_pq", MTLSizeMake((size_t)P, 1, 1), MTLSizeMake(256, 1, 1),
                      {z, w_.get("latents_mean"), w_.get("latents_std"), w_.get("post_quant_conv.weight"),
                       w_.get("post_quant_conv.bias"), x.t},
                      &W16, sizeof(W16), 6);
  op_done("unpack", x);
  auto next = [&](const Act& y) {
    release(x.t);
    x = y;
  };
  next(conv3x3(x, "decoder.conv_in", 384, nullptr));
  next(resblock(x, "decoder.mid_block.resnets.0.", 384));
  next(attention(x, "decoder.mid_block.attentions.0."));
  next(resblock(x, "decoder.mid_block.resnets.1.", 384));
  const int outs[4] = {384, 384, 192, 96};
  for (int b = 0; b < 4; b++) {
    const std::string p = "decoder.up_blocks." + std::to_string(b) + ".";
    const Act stage{Tensor(), x.H, x.W, outs[b]};
    if (wino(stage, outs[b])) {  // map the stage's Winograd buffers while the GPU runs the previous op
      const size_t T = (size_t)wino_rows(stage, outs[b]) * (x.W / 2);
      prewire({(size_t)16 * outs[b] * outs[b] * 2, 16 * T * outs[b] * 2, 16 * T * outs[b] * 4});
    }
    for (int r = 0; r < 3; r++) next(resblock(x, p + "resnets." + std::to_string(r) + ".", outs[b]));
    if (b < 3) next(upsample(x, p + "upsamplers.0.resample.1", outs[b] / 2));
  }
  next(norm_silu(x, "decoder.norm_out.gamma", true));
  Tensor rgba = scratch((size_t)x.H * x.W * 4);
  OutParams op{x.H, x.W, x.C};
  m_.dispatch_threads("vae_conv_out_rgba", MTLSizeMake((size_t)x.H * x.W, 1, 1), MTLSizeMake(256, 1, 1),
                      {x.t, w_.get("decoder.conv_out.weight"), w_.get("decoder.conv_out.bias"), rgba}, &op, sizeof(op), 4);
  id<MTLCommandBuffer> last = m_.command_buffer();
  m_.end_wait();
  retire();
  add_busy(last, gpu_ms_, gpu_end_);
  memcpy(out_rgba, rgba.ptr<uint8_t>(), (size_t)x.H * x.W * 4);
  trim();
}

}  // namespace krea
