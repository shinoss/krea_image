#include "model.h"

#include "ane.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <thread>

#import <Accelerate/Accelerate.h>

namespace krea {

static constexpr size_t BF = 2, F4 = 4;

float bf16_round(float x) {
  uint32_t u;
  memcpy(&u, &x, 4);
  u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000u;
  float r;
  memcpy(&r, &u, 4);
  return r;
}

static double now_ms() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ==========================================================================================
// Text encoder (Qwen3-VL-4B text model, layers 0..34)
// ==========================================================================================
TextEncoder::TextEncoder(Metal& m, const std::string& path) : m_(m), w_(m, path) {
  if (const char* e = getenv("KREA_TE_RESIDENT")) resident_ = atoi(e) != 0;
}

void TextEncoder::ensure(int rows) {
  if (rows <= cap_) return;
  cap_ = std::max(rows, 256);
  x_ = m_.alloc((size_t)cap_ * kTD * F4);
  h_ = m_.alloc((size_t)cap_ * kTD * BF);
  qkv_ = m_.alloc((size_t)cap_ * kTeQKV * BF);
  ao_ = m_.alloc((size_t)cap_ * kTeHeads * kHd * BF);
  mlp_ = m_.alloc((size_t)cap_ * kTeFF * BF);
  cos_ = m_.alloc((size_t)cap_ * 64 * F4);
  sin_ = m_.alloc((size_t)cap_ * 64 * F4);
}

void TextEncoder::encode(const std::vector<int>& ids, int drop, const Tensor& taps) {
  const int L = (int)ids.size();
  if (L <= drop) throw std::runtime_error("prompt too short");
  ensure(L);
  // Qwen3 rotate-half RoPE, theta 5e6. (Qwen3-VL's interleaved M-RoPE reduces to 1-D RoPE for text-only
  // input because all three position axes are equal.)
  float* c = cos_.ptr<float>();
  float* s = sin_.ptr<float>();
  for (int p = 0; p < L; p++)
    for (int i = 0; i < 64; i++) {
      const double a = (double)p / std::pow(5000000.0, (2.0 * i) / 128.0);
      c[p * 64 + i] = (float)std::cos(a);
      s[p * 64 + i] = (float)std::sin(a);
    }
  // The system prefix's K/V never depend on the prompt (causal attention): reuse them when the first
  // `drop` ids match the cached prefix (a prompt starting with a newline merges into the prefix's last
  // token, so the check is on the ids, not the text).
  static const bool no_prefix = getenv("KREA_TE_NO_PREFIX") != nullptr;
  const bool use_prefix = !no_prefix && prefix_kv_ && (int)prefix_ids_.size() == drop &&
                          std::equal(prefix_ids_.begin(), prefix_ids_.end(), ids.begin());
  const int L0 = use_prefix ? drop : 0;
  // Embedding rows are gathered on the CPU straight from the mapped file: a GPU command touching the
  // table's buffer would make the whole 0.78 GB segment resident.
  {
    const TensorInfo& ei = w_.info("embed");
    const uint16_t* table = (const uint16_t*)w_.host_ptr(ei.offset);
    float* xp = x_.ptr<float>();
    for (int i = L0; i < L; i++) {
      const uint16_t* row = table + (size_t)ids[i] * kTD;
      float* dst = xp + (size_t)(i - L0) * kTD;
      for (int k = 0; k < kTD; k++) {
        const uint32_t u = (uint32_t)row[k] << 16;
        memcpy(&dst[k], &u, 4);
      }
    }
  }
  const bool store = !no_prefix && !use_prefix;
  if (store && !prefix_kv_) prefix_kv_ = m_.alloc((size_t)kTeLayers * drop * 2 * kTeKv * kHd * BF);
  if (store && prefix_kv_.bytes < (size_t)kTeLayers * drop * 2 * kTeKv * kHd * BF)
    prefix_kv_ = m_.alloc((size_t)kTeLayers * drop * 2 * kTeKv * kHd * BF);
  layers(L0, L, taps, store);
  if (store) prefix_ids_.assign(ids.begin(), ids.begin() + drop);
}

// Layers 0..34 over rows [L0, L) (rows [0, L0) are the cached prefix). With store, the prefix rows'
// K|V (rows [0, drop) of a full run) are saved into prefix_kv_ for later prompts.
void TextEncoder::layers(int L0, int L, const Tensor& taps, bool store) {
  const int Lr = L - L0, drop = kDrop, kvw = 2 * kTeKv * kHd;
  static const int n_layers = getenv("KREA_TE_LAYERS") ? atoi(getenv("KREA_TE_LAYERS")) : kTeLayers;
  // Weights: streamed (default) or read through the mapping (KREA_TE_RESIDENT=1). Streaming reads groups
  // of kGroup layers with F_NOCACHE (~4 GB/s) into two staging buffers, a reader thread filling one while
  // the GPU runs the other: a new prompt costs about one sequential read of the file and never evicts
  // the DiT's pages.
  constexpr int kGroup = 2;
  auto group_span = [&](int g) {
    size_t lo = SIZE_MAX, hi = 0;
    for (int l = g; l < std::min(g + kGroup, n_layers); l++) {
      const auto [off, len] = w_.span("L" + std::to_string(l) + ".");
      lo = std::min(lo, off);
      hi = std::max(hi, off + len);
    }
    return std::make_pair(lo, hi - lo);
  };
  Tensor stage[2];
  size_t stage_off[2] = {0, 0};
  int cur = 0;
  std::thread reader;
  if (!resident_) {
    size_t stage_bytes = 0;
    for (int g = 0; g < n_layers; g += kGroup) stage_bytes = std::max(stage_bytes, group_span(g).second);
    stage[0] = m_.alloc(stage_bytes);
    stage[1] = m_.alloc(stage_bytes);
  }
  auto load = [&](int g, int b) {
    const auto [off, len] = group_span(g);
    w_.read_range(off, len, stage[b].ptr<void>());
    stage_off[b] = off;
  };
  auto T = [&](const std::string& name) {
    if (resident_) return w_.get(name);
    const TensorInfo& ti = w_.info(name);
    return stage[cur].at(ti.offset - stage_off[cur]);
  };
  auto Wt = [&](const std::string& name) {
    return w_.has(name + ".s") ? Weight{T(name), T(name + ".s"), true} : Weight{T(name), Tensor(), false};
  };
  if (!resident_) load(0, 0);
  const Tensor qkv_r = qkv_.at((size_t)L0 * kTeQKV * BF);
  for (int l = 0; l < n_layers; l++) {
    if (!resident_ && l % kGroup == 0) {
      if (l > 0) {  // run group l/kGroup - 1 (in stage[cur]) while group l/kGroup finishes loading
        m_.commit();
        m_.wait();
        reader.join();
        cur ^= 1;
        m_.begin();
      }
      if (l + kGroup < n_layers) reader = std::thread(load, l + kGroup, cur ^ 1);
    }
    const std::string p = "L" + std::to_string(l) + ".";
    m_.rms_norm(x_, T(p + "ln1"), Tensor(), h_, Lr, kTD, kTD, kTD, kTeEps);
    m_.gemm(Epi::BF16, h_, Wt(p + "qkv"), qkv_r, Lr, kTeQKV, kTD, kTD, kTeQKV, kTeQKV);
    m_.qk_norm_rope(qkv_r, T(p + "qn"), T(p + "kn"), cos_.at((size_t)L0 * 64 * F4), sin_.at((size_t)L0 * 64 * F4), Lr,
                    kTeQKV, 0, kTeHeads, kTeHeads * kHd, kTeKv, kTeEps, kRopeHalf);
    const Tensor kv_cache = prefix_kv_ ? prefix_kv_.at((size_t)l * drop * kvw * BF) : Tensor();
    if (L0 > 0) m_.copy_rows(kv_cache, qkv_.at(kTeHeads * kHd * BF), drop, kvw, kvw, kTeQKV);
    if (store) m_.copy_rows(qkv_.at(kTeHeads * kHd * BF), kv_cache, drop, kvw, kTeQKV, kvw);
    m_.attention(qkv_r, qkv_.at(kTeHeads * kHd * BF), qkv_.at((kTeHeads + kTeKv) * kHd * BF), ao_, Lr, L, kTeHeads,
                 kTeQKV, kTeQKV, kTeQKV, kTeHeads * kHd, kTeHeads / kTeKv, true, L0);
    m_.gemm(Epi::Resid, ao_, Wt(p + "o"), x_, Lr, kTD, kTeHeads * kHd, kTeHeads * kHd, kTD, kTD);
    m_.rms_norm(x_, T(p + "ln2"), Tensor(), h_, Lr, kTD, kTD, kTD, kTeEps);
    m_.gemm(Epi::SwiGLU, h_, Wt(p + "gu"), mlp_, Lr, 2 * kTeFF, kTD, kTD, 2 * kTeFF, kTeFF);
    m_.gemm(Epi::Resid, mlp_, Wt(p + "down"), x_, Lr, kTD, kTeFF, kTeFF, kTD, kTD);
    if ((l + 1) % 3 == 2) {  // hidden_states[l + 1] is tapped: rows after the prefix -> taps[:, j]
      const int j = (l + 1 - 2) / 3;
      m_.copy_rows_f32(x_.at((size_t)(drop - L0) * kTD * F4), taps.at((size_t)j * kTD * F4), L - drop, kTD, kTD,
                       kTaps * kTD);
    }
  }
  if (reader.joinable()) reader.join();
  // (command buffers retain the staging buffers until they complete; then they are freed)
}

// ==========================================================================================
// TextFusion + txtmlp
// ==========================================================================================
TextFusion::TextFusion(Metal& m, const std::string& path) : m_(m), w_(m, path) {}

void TextFusion::ensure(int T) {
  if (T <= cap_) return;
  cap_ = std::max(T, 64);
  const size_t rows = (size_t)cap_ * kTaps;
  h_ = m_.alloc(rows * kTD * BF);
  qkvg_ = m_.alloc(rows * kTfQKVG * BF);
  ao_ = m_.alloc(rows * kTD * BF);
  mid_ = m_.alloc(rows * kTfFF * BF);
  fused_ = m_.alloc((size_t)cap_ * kTD * F4);
  hm_ = m_.alloc((size_t)cap_ * kTD * F4);
  m1_ = m_.alloc((size_t)cap_ * kD * F4);
}

void TextFusion::block(int b, const Tensor& x, int rows, bool layerwise) {
  const std::string p = "F" + std::to_string(b) + ".";
  m_.rms_norm(x, w_.get(p + "n1"), Tensor(), h_, rows, kTD, kTD, kTD, kEps);
  m_.gemm(Epi::BF16, h_, w_.linear(p + "qkvg"), qkvg_, rows, kTfQKVG, kTD, kTD, kTfQKVG, kTfQKVG);
  m_.qk_norm_rope(qkvg_, w_.get(p + "qn"), w_.get(p + "kn"), Tensor(), Tensor(), rows, kTfQKVG, 0, kTfHeads, kTD,
                  kTfHeads, kEps, kRopeNone);
  if (layerwise)
    m_.layerwise_attn(qkvg_, ao_, rows, kTfQKVG, kTD, kTfHeads, 0, kTD, 2 * kTD, 3 * kTD);
  else
    m_.attention(qkvg_, qkvg_.at(kTD * BF), qkvg_.at(2 * kTD * BF), ao_, rows, rows, kTfHeads, kTfQKVG, kTfQKVG, kTfQKVG,
                 kTD, 1, false, 0, qkvg_.at(3 * kTD * BF), kTfQKVG);
  m_.gemm(Epi::Resid, ao_, w_.linear(p + "o"), x, rows, kTD, kTD, kTD, kTD, kTD);
  m_.rms_norm(x, w_.get(p + "n2"), Tensor(), h_, rows, kTD, kTD, kTD, kEps);
  m_.gemm(Epi::SwiGLU, h_, w_.linear(p + "gu"), mid_, rows, 2 * kTfFF, kTD, kTD, 2 * kTfFF, kTfFF);
  m_.gemm(Epi::Resid, mid_, w_.linear(p + "down"), x, rows, kTD, kTfFF, kTfFF, kTD, kTD);
}

void TextFusion::run(const Tensor& taps, int T, const Tensor& out) {
  ensure(T);
  for (int b = 0; b < 2; b++) block(b, taps, T * kTaps, true);
  m_.tap_project(taps, w_.get("proj"), fused_, T, kTD);
  for (int b = 2; b < 4; b++) block(b, fused_, T, false);
  m_.rms_norm(fused_, w_.get("M.n"), Tensor(), hm_, T, kTD, kTD, kTD, kEps, true);
  m_.gemm_f32(hm_, w_.get("M.w1"), m1_, T, kD, kTD, kTD, kD, kD, kF32Bias | kF32Gelu, w_.get("M.b1"));
  m_.gemm_f32(m1_, w_.get("M.w2"), out, T, kD, kD, kD, kD, kD, kF32Bias, w_.get("M.b2"));
}

// ==========================================================================================
// DiT
// ==========================================================================================
DiT::DiT(Metal& m, const std::string& path) : m_(m), w_(m, path) {
  if (const char* e = getenv("KREA_DIT_LAYERS")) layers_ = std::min(kLayers, std::max(1, atoi(e)));
  const TensorInfo& gu = w_.info("L0.gu");
  file_units_ = (int)(gu.shape[1] / 2);
  qkvg_cols_ = (int)w_.info("L0.qkvg").shape[1];
  // q.k gain product per head dimension: block 0 has one dimension at 52.8 x 52.8 (every other block stays
  // below ~32); such a dimension dominates the logits (~3e4) and needs fp32 (see kernels/norm.metal).
  static const float kBigGain = getenv("KREA_BIG_GAIN") ? (float)atof(getenv("KREA_BIG_GAIN")) : 256.0f;
  big_dim_.assign(kLayers, -1);
  for (int l = 0; l < kLayers; l++) {
    const float* q = (const float*)w_.host_ptr(w_.info("L" + std::to_string(l) + ".qn").offset);
    const float* k = (const float*)w_.host_ptr(w_.info("L" + std::to_string(l) + ".kn").offset);
    float best = 0;
    for (int d = 0; d < kHd; d++)
      if (std::fabs(q[d] * k[d]) > kBigGain && std::fabs(q[d] * k[d]) > best) {
        best = std::fabs(q[d] * k[d]);
        big_dim_[l] = d;
      }
  }
}

void DiT::set_ane(ANEOffload* ane) {
  if (ane && (ane->gpu_units() != file_units_ || ane->qkvg_c0() != qkvg_cols_))
    throw std::runtime_error("the Neural Engine programs were built for a different GPU/ANE split than the DiT weight file");
  ane_ = ane;
  if (const char* e = getenv("KREA_GPU_O_LAST")) o_last_ = std::max(0, atoi(e));
  serial_ = getenv("KREA_SERIAL") != nullptr;
  cap_rows_ = 0;  // re-reserve: the row buffers are padded to whole chunks
}

void DiT::reserve(int T, int M) {
  size_t R = (size_t)T + M;
  if (ane_) R = (R + ane_->chunk() - 1) / ane_->chunk() * ane_->chunk();  // whole chunks
  if (R <= cap_rows_) return;
  cap_rows_ = R;
  x_ = m_.alloc(R * kD * F4);
  h_ = m_.alloc(R * kD * BF);
  qkvg_ = m_.alloc(R * kQKVG * BF);
  ao_ = m_.alloc(R * kD * BF);
  mid_ = m_.alloc(R * file_units_ * BF);
  hf_ = m_.alloc((size_t)M * kD * F4);
  cos_ = m_.alloc(R * 64 * F4);
  sin_ = m_.alloc(R * 64 * F4);
  st_ = m_.alloc(R * F4);
  qbig_ = m_.alloc(R * kHeads * F4);
  kbig_ = m_.alloc(R * kKvHeads * F4);
}

// QK-norm + RoPE of `rows` fused rows starting at row0 (fp32 big-dimension split for layers that have one).
void DiT::qk_rope(int l, const Tensor& qkvg_rows, int rows, int row0) {
  const std::string p = "L" + std::to_string(l) + ".";
  const int big = big_dim_[l];
  m_.qk_norm_rope(qkvg_rows, w_.get(p + "qn"), w_.get(p + "kn"), cos_.at((size_t)row0 * 64 * F4),
                  sin_.at((size_t)row0 * 64 * F4), rows, kQKVG, 0, kHeads, kD, kKvHeads, kEps, kRopePairs, big,
                  big >= 0 ? qbig_.at((size_t)row0 * kHeads * F4) : Tensor(),
                  big >= 0 ? kbig_.at((size_t)row0 * kKvHeads * F4) : Tensor());
}

// Gated attention of `rows` query rows starting at row0 against all R keys.
void DiT::attend(int l, const Tensor& q_rows, int rows, int row0, const Tensor& o_rows, int R) {
  const int kcol = kD, vcol = kD + kKvHeads * kHd, gcol = kD + 2 * kKvHeads * kHd;
  const bool big = big_dim_[l] >= 0;
  m_.attention(q_rows, qkvg_.at(kcol * BF), qkvg_.at(vcol * BF), o_rows, rows, R, kHeads, kQKVG, kQKVG, kQKVG, kD,
               kHeads / kKvHeads, false, 0, q_rows.at(gcol * BF), kQKVG,
               big ? qbig_.at((size_t)row0 * kHeads * F4) : Tensor(), big ? kbig_ : Tensor());
}

void DiT::set_grid(int T, int H16, int W16) {
  reserve(T, H16 * W16);
  T_ = T;
  H16_ = H16;
  W16_ = W16;
  // 3-axis RoPE (32/48/48 head dims, theta 1000, interleaved pairs): pair p < 16 rotates by the frame
  // coordinate (always 0 here), p in [16, 40) by the row, p in [40, 64) by the column. Text rows sit at
  // (0, 0, 0), image token (i, j) at (0, i, j) (sampling.py prepare()).
  double om[64];
  for (int p = 0; p < 64; p++) {
    const int ax = p < 16 ? 0 : p < 40 ? 1 : 2, k = p - (ax == 0 ? 0 : ax == 1 ? 16 : 40), d = ax == 0 ? 32 : 48;
    om[p] = 1.0 / std::pow(1000.0, (2.0 * k) / d);
  }
  float* c = cos_.ptr<float>();
  float* s = sin_.ptr<float>();
  const int R = T + H16 * W16;
  for (int r = 0; r < R; r++) {
    const int k = r - T;
    const double pi = r < T ? 0 : k / W16, pj = r < T ? 0 : k % W16;
    for (int p = 0; p < 64; p++) {
      const double a = (p < 16 ? 0.0 : p < 40 ? pi : pj) * om[p];
      c[(size_t)r * 64 + p] = (float)std::cos(a);
      s[(size_t)r * 64 + p] = (float)std::sin(a);
    }
  }
}

void DiT::make_cond(const float* t, const float* tvec, StepCond& out) {
  const size_t per = (size_t)6 * kD;
  if (!out.mods) out.mods = m_.alloc(kLayers * per * F4);
  if (!out.fin) out.fin = m_.alloc(2 * kD * F4);
  auto host = [&](const std::string& name) { return (const float*)w_.host_ptr(w_.info(name).offset); };
  float* m = out.mods.ptr<float>();
  for (int l = 0; l < kLayers; l++) {
    const std::string p = "L" + std::to_string(l) + ".";
    const float* tab = host(p + "mod");
    const float* n1 = host(p + "n1");
    const float* n2 = host(p + "n2");
    float* o = m + l * per;
    for (int k = 0; k < kD; k++) {
      float v[6];
      for (int j = 0; j < 6; j++) v[j] = tvec[j * kD + k] + tab[j * kD + k];
      o[k] = n1[k] * (1.0f + v[0]);
      o[kD + k] = v[1];
      o[2 * kD + k] = v[2];
      o[3 * kD + k] = n2[k] * (1.0f + v[3]);
      o[4 * kD + k] = v[4];
      o[5 * kD + k] = v[5];
    }
  }
  const float* lin = host("last.mod");
  const float* nf = host("last.n");
  float* f = out.fin.ptr<float>();
  for (int k = 0; k < kD; k++) {
    f[k] = nf[k] * (1.0f + t[k] + lin[k]);
    f[kD + k] = t[k] + lin[kD + k];
  }
}

void DiT::lin_qkvg(int l, const Tensor& h, const Tensor& dst, int rows) {
  m_.gemm(Epi::BF16, h, w_.linear("L" + std::to_string(l) + ".qkvg"), dst, rows, qkvg_cols_, kD, kD, qkvg_cols_, kQKVG);
}

void DiT::lin_o(int l, const Tensor& ao, const Tensor& x, int rows, const Tensor& gate) {
  m_.gemm(Epi::ResidGate, ao, w_.linear("L" + std::to_string(l) + ".o"), x, rows, kD, kD, kD, kD, kD, gate);
}

void DiT::lin_mlp(int l, const Tensor& h, const Tensor& mid, const Tensor& x, int rows, const Tensor& gate) {
  const std::string p = "L" + std::to_string(l) + ".";
  const int H1 = file_units_;
  m_.gemm(Epi::SwiGLU, h, w_.linear(p + "gu"), mid, rows, 2 * H1, kD, kD, 2 * H1, H1);
  m_.gemm(Epi::ResidGate, mid, w_.linear(p + "down"), x, rows, kD, H1, H1, kD, kD, gate);
}

// GPU-only layer (the whole weight file on the GPU).
void DiT::layer(int l, const StepCond& c) {
  const std::string p = "L" + std::to_string(l) + ".";
  const int R = T_ + H16_ * W16_;
  const Tensor mo = c.mods.at((size_t)l * 6 * kD * F4);
  auto mv = [&](int j) { return mo.at((size_t)j * kD * F4); };
  m_.rms_norm(x_, mv(0), mv(1), h_, R, kD, kD, kD, kEps);
  lin_qkvg(l, h_, qkvg_, R);
  qk_rope(l, qkvg_, R, 0);
  m_.mark("qkvg+norm+rope");
  attend(l, qkvg_, R, 0, ao_, R);
  m_.mark("attention");
  lin_o(l, ao_, x_, R, mv(2));
  m_.mark("o_proj");
  m_.rms_norm(x_, mv(3), mv(4), h_, R, kD, kD, kD, kEps);
  lin_mlp(l, h_, mid_, x_, R, mv(5));
  m_.mark("mlp");
}

// Hybrid layers, pipelined over chunks of C = ane_->chunk() rows of [text ; image]. Everything in a layer
// except attention is row-wise, so a chunk can move on as soon as attention has produced it:
//
//   GPU: attention(c) -> hand off chunk c's attention output ... attention(c+1) ... add ANE O(c),
//        RMSNorm2(c) hand-off ... [O-proj of the last chunk] ... per chunk: GPU MLP half(c), add ANE MLP(c),
//        next layer's RMSNorm1(c) hand-off + GPU q|k|v|gate columns(c) ... add ANE columns + QK-norm/RoPE
//   ANE: O(0) O(1) MLP(0) O(2) MLP(1) ... MLP(n-1) QKVG(0) ... QKVG(n-1)
//
// so the ANE's O-projection and MLP work overlaps the GPU's attention, and the next layer's QKVG(c) starts
// as soon as chunk c's MLP is merged, overlapping the GPU's MLP half of the later chunks (the Qwen engine's
// "sched 1"). The last chunk does its O-projection on the GPU (its ANE turn would come after earlier chunks'
// MLP and stall the GPU), before the wait for the previous chunk's ANE O-projection. With one chunk
// nothing overlaps and the O-projection goes to the ANE. Text rows are part of chunk 0: every chunk holds
// ANE-shaped rows, the last one zero-padded.
void DiT::forward_hybrid(int l0, int l1, const StepCond& cond) {
  using A = ANEOffload;
  const int R = T_ + H16_ * W16_, C = ane_->chunk(), n = (R + C - 1) / C;
  if (n > A::kMaxChunks) throw std::runtime_error("image too large for the Neural Engine buffers");
  const int c0 = qkvg_cols_, nq = kQKVG - c0;
  const int o_last = o_last_ >= 0 ? o_last_ : (n >= 2 ? 1 : 0);
  const int n_ane_o = ane_->has(A::OPROJ) ? std::max(0, n - o_last) : 0;
  auto rows = [&](int c) { return std::min(C, R - c * C); };
  auto xr = [&](int c) { return x_.at((size_t)c * C * kD * F4); };
  auto hr = [&](int c) { return h_.at((size_t)c * C * kD * BF); };
  auto qr = [&](int c) { return qkvg_.at((size_t)c * C * kQKVG * BF); };
  auto ar = [&](int c) { return ao_.at((size_t)c * C * kD * BF); };
  auto mr = [&](int c) { return mid_.at((size_t)c * C * file_units_ * BF); };
  auto in = [&](A::Kind k, int c) { return ane_->input(k).at(c * ane_->chunk_bytes(kD)); };
  auto out = [&](A::Kind k, int c) { return ane_->output(k).at(c * ane_->chunk_bytes(ane_->out_channels(k))); };
  auto mv = [&](int l, int j) { return cond.mods.at(((size_t)l * 6 + j) * kD * F4); };
  std::vector<uint64_t> tq(n), to(n), tm(n);
  std::vector<char> qdone(n, 0);

  // RMSNorm(x) * mul + add of chunk c -> bf16 rows in h_ (GPU half) and the ANE input chunk of kind k.
  auto handoff = [&](const Tensor& mul, const Tensor& add, A::Kind k, int c) {
    m_.rms_stats(xr(c), st_.at((size_t)c * C * F4), rows(c), kD, kEps);
    m_.rms_dual(xr(c), st_.at((size_t)c * C * F4), mul, add, hr(c), in(k, c), rows(c), C, kD, kD, C, 1.0f);
  };
  auto qkv_start = [&](int l, int c) {
    handoff(mv(l, 0), mv(l, 1), A::QKVG, c);
    m_.mark("ln1+handoff");
    tq[c] = ane_->launch(A::QKVG, l, c, c + 1);
    qdone[c] = 0;
    lin_qkvg(l, hr(c), qr(c), rows(c));
    m_.mark("qkvg.gpu");
  };
  auto qkv_finish = [&](int l, int c) {  // no-op if chunk c's q|k|v|gate of this layer are already final
    if (qdone[c]) return;
    qdone[c] = 1;
    ane_->gpu_wait(tq[c]);
    m_.ane_cols_scatter(out(A::QKVG, c), qr(c), rows(c), nq, C, kQKVG, c0);
    qk_rope(l, qr(c), rows(c), c * C);
    m_.mark("qkvg.wait+scatter+rope");
  };
  auto mlp_start = [&](int l, int c) {  // chunk c's O-projection is in x
    handoff(mv(l, 3), mv(l, 4), A::MLP, c);
    m_.mark("ln2+handoff");
    tm[c] = ane_->launch(A::MLP, l, c, c + 1);
  };

  if (serial_) {  // unpipelined reference schedule: whole-layer hand-offs, the GPU waits at each merge
    for (int l = l0; l < l1; l++) {
      for (int c = 0; c < n; c++) qkv_start(l, c);
      for (int c = 0; c < n; c++) qkv_finish(l, c);
      attend(l, qkvg_, R, 0, ao_, R);
      lin_o(l, ao_, x_, R, mv(l, 2));
      for (int c = 0; c < n; c++) mlp_start(l, c);
      for (int c = 0; c < n; c++) lin_mlp(l, hr(c), mr(c), xr(c), rows(c), mv(l, 5));
      for (int c = 0; c < n; c++) {
        ane_->gpu_wait(tm[c]);
        m_.resid_gate_add_t16(xr(c), out(A::MLP, c), mv(l, 5), rows(c), kD, C, ane_->out_scale());
      }
    }
    return;
  }

  for (int c = 0; c < n; c++) {
    qkv_start(l0, c);
    if (c) qkv_finish(l0, c - 1);
  }
  qkv_finish(l0, n - 1);
  for (int l = l0; l < l1; l++) {
    const Tensor g1 = mv(l, 2), g2 = mv(l, 5);
    auto o_finish = [&](int c) {
      ane_->gpu_wait(to[c]);
      m_.resid_gate_add_t16(xr(c), out(A::OPROJ, c), g1, rows(c), kD, C, ane_->o_scale());
      m_.mark("o.wait+resid");
      mlp_start(l, c);
    };
    for (int c = 0; c < n; c++) qkv_finish(l, c);  // (all done in the previous layer)
    for (int c = 0; c < n; c++) {
      attend(l, qr(c), rows(c), c * C, ar(c), R);
      m_.mark("attention");
      if (c < n_ane_o) {
        m_.to_ane_bf16(ar(c), in(A::OPROJ, c), rows(c), C, kD, kD, C, 1.0f / ane_->o_scale());
        m_.mark("o.handoff");
        to[c] = ane_->launch(A::OPROJ, l, c, c + 1);
      }
      if (c >= 1 && c - 1 < n_ane_o) {
        // the first GPU-projected chunk does its O-projection before the wait for the ANE's last
        // O-projection (queued behind MLP work), which then has time to finish
        if (c == n_ane_o) {
          lin_o(l, ar(c), xr(c), rows(c), g1);
          m_.mark("o_proj");
        }
        o_finish(c - 1);
        if (c == n_ane_o) mlp_start(l, c);
      }
    }
    if (n_ane_o == n) o_finish(n - 1);
    for (int c = n_ane_o + (n_ane_o >= 1); c < n; c++) {
      lin_o(l, ar(c), xr(c), rows(c), g1);
      m_.mark("o_proj");
      mlp_start(l, c);
    }
    // Per chunk: GPU MLP half, ANE merge, next layer's RMSNorm1 hand-off + GPU q|k|v|gate columns, so the
    // ANE's QKVG(c) is queued right behind its MLP work. The waits for the ANE's columns come last.
    for (int c = 0; c < n; c++) {
      lin_mlp(l, hr(c), mr(c), xr(c), rows(c), g2);
      m_.mark("mlp.gpu");
      ane_->gpu_wait(tm[c]);
      m_.resid_gate_add_t16(xr(c), out(A::MLP, c), g2, rows(c), kD, C, ane_->out_scale());
      m_.mark("mlp.wait+resid");
      if (l + 1 < l1) qkv_start(l + 1, c);
    }
    if (l + 1 < l1)
      for (int c = 0; c < n; c++) qkv_finish(l + 1, c);
  }
}

void DiT::embed(const Tensor& txt, const Tensor& z) {
  const int T = T_, M = H16_ * W16_;
  if (!M) throw std::runtime_error("DiT: set_grid first");
  m_.copy_rows_f32(txt, x_, T, kD, kD, kD);
  m_.gemm_f32(z, w_.get("first.w"), x_.at((size_t)T * kD * F4), M, kD, kLatC, kLatC, kD, kD, kF32Bias, w_.get("first.b"));
}

void DiT::run_layers(int l0, int l1, const StepCond& c) {
  l1 = std::min(l1, layers_);
  if (l0 >= l1) return;
  if (ane_) return forward_hybrid(l0, l1, c);
  if (needs_ane()) throw std::runtime_error("this DiT weight file needs the Neural Engine half (weights/ane)");
  for (int l = l0; l < l1; l++) layer(l, c);
}

void DiT::head(const StepCond& c, const Tensor& vel) {
  const int T = T_, M = H16_ * W16_;
  const Tensor xi = x_.at((size_t)T * kD * F4);
  m_.rms_norm(xi, c.fin, c.fin.at(kD * F4), hf_, M, kD, kD, kD, kEps, true);
  m_.gemm_f32(hf_, w_.get("last.w"), vel, M, kLatC, kD, kD, kLatC, kLatC, kF32Bias, w_.get("last.b"));
}

void DiT::forward(const Tensor& txt, const Tensor& z, const StepCond& c, const Tensor& vel) {
  embed(txt, z);
  run_layers(0, layers_, c);
  head(c, vel);
}

// ==========================================================================================
// Conditioning
// ==========================================================================================
Conditioning::Conditioning(Metal& m, const std::string& path) : w_(m, path) {}

void Conditioning::get(bool fast, float sigma, std::vector<float>& t, std::vector<float>& tvec) {
  t.resize(kD);
  tvec.resize(6 * kD);
  const std::string pre = fast ? "fast." : "base.";
  auto host = [&](const std::string& n) { return (const float*)w_.host_ptr(w_.info(n).offset); };
  if (w_.has(pre + "sigma")) {
    const TensorInfo& si = w_.info(pre + "sigma");
    const float* s = host(pre + "sigma");
    for (int i = 0; i < (int)si.shape[0]; i++)
      if (s[i] == sigma) {
        memcpy(t.data(), host(pre + "t") + (size_t)i * kD, kD * 4);
        memcpy(tvec.data(), host(pre + "tvec") + (size_t)i * 6 * kD, 6 * kD * 4);
        return;
      }
  }
  if (fast) throw std::runtime_error("the 4-step LoRA only supports its 4 trained sigmas (use 4 steps)");
  for (auto& [s, v] : memo_)
    if (s == sigma) {
      t = v.first;
      tvec = v.second;
      return;
    }
  compute(sigma, t, tvec);
  memo_.push_back({sigma, {t, tvec}});
}

void Conditioning::compute(float sigma, std::vector<float>& t, std::vector<float>& tvec) {
  auto host = [&](const std::string& n) { return (const float*)w_.host_ptr(w_.info(n).offset); };
  auto gelu = [](float x) { return 0.5f * x * (1.0f + std::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x))); };
  // temb (cos first, t * 1000, period 1e4) of bf16(sigma), then tmlp and tproj, all f32
  const float ts = bf16_round(sigma);
  std::vector<float> e(256), h(kD), g(kD);
  for (int i = 0; i < 128; i++) {
    const float f = std::exp(-std::log(10000.0f) * (float)i / 128.0f);
    e[i] = std::cos(ts * 1000.0f * f);
    e[128 + i] = std::sin(ts * 1000.0f * f);
  }
  auto linear = [&](const std::string& n, const float* x, float* y, int in, int out) {
    memcpy(y, host(n + ".b"), out * 4);
    cblas_sgemv(CblasRowMajor, CblasTrans, in, out, 1.0f, host(n + ".w"), out, x, 1, 1.0f, y, 1);
  };
  linear("tmlp1", e.data(), h.data(), 256, kD);
  for (float& v : h) v = gelu(v);
  linear("tmlp2", h.data(), t.data(), kD, kD);
  for (int i = 0; i < kD; i++) g[i] = gelu(t[i]);
  linear("tproj", g.data(), tvec.data(), kD, 6 * kD);
}

// ==========================================================================================
std::vector<float> krea_sigmas(int steps, float mu) {
  std::vector<float> s(steps + 1);
  const double e = std::exp((double)mu);
  for (int i = 0; i <= steps; i++) {
    const double t = 1.0 - (double)i / steps;
    s[i] = t <= 0 ? 0.0f : (float)(e / (e + (1.0 / t - 1.0)));
  }
  return s;
}

}  // namespace krea
