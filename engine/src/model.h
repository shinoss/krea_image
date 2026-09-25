// Krea 2 Turbo model components on Metal.
#pragma once
#include <memory>
#include <string>
#include <vector>

#include "metal.h"
#include "weights.h"

namespace krea {

// Model shapes of Krea 2 Turbo. Everything below is specialized for them.
constexpr int kD = 6144;            // DiT width
constexpr int kHeads = 48, kKvHeads = 12, kHd = 128;
constexpr int kQKVG = kD + 2 * kKvHeads * kHd + kD;  // fused q | k | v | gate = 15360
constexpr int kKV = 2 * kKvHeads * kHd;              // k | v columns (3072), at column kD of the fused rows
constexpr int kGateCol = kD + kKV;                   // gate columns start (9216)
constexpr int kFF = 16384;          // DiT SwiGLU units
constexpr int kLayers = 28;
constexpr int kLatC = 64;           // packed latent channels (16 x 2 x 2)
constexpr int kTD = 2560;           // text width (encoder and fusion)
constexpr int kTaps = 12;           // tapped encoder layers
constexpr int kTeLayers = 35;       // encoder layers used (taps are the outputs of layers 1, 4, ..., 34)
constexpr int kTeHeads = 32, kTeKv = 8, kTeFF = 9728;
constexpr int kTeQKV = (kTeHeads + 2 * kTeKv) * kHd;  // 6144
constexpr int kTfHeads = 20, kTfFF = 6912;
constexpr int kTfQKVG = 4 * kTD;    // 10240
constexpr int kDrop = 34;           // system-prefix tokens dropped from the encoder output
constexpr float kEps = 1e-5f;       // Krea RMSNorm eps
constexpr float kTeEps = 1e-6f;     // Qwen3 RMSNorm eps

// Round-trip a float through bfloat16 (round-to-nearest-even), matching torch's .to(bfloat16).
float bf16_round(float x);

// ------------------------------------------------------------------------------------------
// Qwen3-VL-4B text model, text only, layers 0..34. Returns the 12 taps (outputs of layers 1, 4, ..., 34)
// of the rows after the system prefix. The prefix is the same for every prompt, so its per-layer K/V are
// computed once and reused: later prompts run only their own rows (exact: attention is causal).
class TextEncoder {
 public:
  TextEncoder(Metal& m, const std::string& path);
  // taps: f32 [(L - drop) * 12, 2560], row t * 12 + j = tap j of token drop + t. Encodes into the current
  // command buffer (may commit and wait internally while streaming weights); valid after the caller's
  // commit + wait.
  void encode(const std::vector<int>& ids, int drop, const Tensor& taps);
  WeightFile& weights() { return w_; }

 private:
  void ensure(int rows);
  void layers(int L0, int L, const Tensor& taps, bool store_prefix);
  Metal& m_;
  WeightFile w_;
  int cap_ = 0;
  Tensor x_, h_, qkv_, ao_, mlp_, cos_, sin_;
  // cached system prefix: token ids and per-layer post-norm/RoPE K|V rows (bf16 [35][drop][2048])
  std::vector<int> prefix_ids_;
  Tensor prefix_kv_;
  bool resident_ = false;  // KREA_TE_RESIDENT=1: use the mapped weights instead of streaming them
};

// ------------------------------------------------------------------------------------------
// TextFusion (2 layerwise blocks across the 12 taps of each token, 12 -> 1 projector, 2 refiner blocks
// across tokens) followed by txtmlp. Depends only on the prompt: its output is what the prompt cache
// keeps (f32 [T, 6144]).
class TextFusion {
 public:
  TextFusion(Metal& m, const std::string& path);
  // taps f32 [T * 12, 2560] (overwritten) -> out f32 [T, 6144]. Encodes into the current command buffer.
  void run(const Tensor& taps, int T, const Tensor& out);
  WeightFile& weights() { return w_; }
  Tensor fused() const { return fused_; }  // f32 [T, 2560] TextFusion output (valid after run)

 private:
  void block(int b, const Tensor& x, int rows, bool layerwise);
  void ensure(int T);
  Metal& m_;
  WeightFile w_;
  int cap_ = 0;
  Tensor h_, qkvg_, ao_, mid_, fused_, hm_, m1_;
};

// ------------------------------------------------------------------------------------------
// Per-step conditioning. For each step: per-layer modulation vectors, f32 [28][6][6144]
//   0: m1 = (1 + w_pre) * (1 + prescale)   1: b1 = preshift    2: g1 = pregate
//   3: m2 = (1 + w_post) * (1 + postscale) 4: b2 = postshift   5: g2 = postgate
// and the final layer's f32 [2][6144]: mf = (1 + w_last) * (1 + scale_f), bf = shift_f.
// Built on the CPU from t(sigma) [6144] and tvec(sigma) [36864] (precomputed by tools/convert.py for the
// preset schedules) and the DiT's per-layer tables and norm weights.
struct StepCond {
  Tensor mods;  // f32 [28][6][6144]
  Tensor fin;   // f32 [2][6144]
};

// Conditioning source: t(sigma) [6144] and tvec(sigma) [36864] (the model sees t = bf16(sigma), as in
// sampling.py), looked up in the tables tools/convert.py precomputed for the preset schedules (cond.qw),
// else computed on the CPU from the f32 tmlp / tproj weights in the same file (1.06 GB, touched only for a
// custom schedule; the 4-step LoRA has only its 4 trained sigmas).
class Conditioning {
 public:
  Conditioning(Metal& m, const std::string& path);
  void get(bool fast, float sigma, std::vector<float>& t, std::vector<float>& tvec);

 private:
  void compute(float sigma, std::vector<float>& t, std::vector<float>& tvec);
  WeightFile w_;
  std::vector<std::pair<float, std::pair<std::vector<float>, std::vector<float>>>> memo_;
};

// ------------------------------------------------------------------------------------------
// Krea 2 single-stream DiT. Every step runs [text T ; image M] rows through all 28 blocks (text is
// t-modulated and attends both ways, so nothing per layer can be cached across steps).
class DiT {
 public:
  DiT(Metal& m, const std::string& path);
  void reserve(int T, int M);
  // RoPE tables for [T text rows at (0,0,0) ; image grid H16 x W16 at (0, i, j)] (host-written f32 [R, 64]).
  void set_grid(int T, int H16, int W16);
  // CPU: conditioning for one step from t(sigma) and tvec(sigma).
  void make_cond(const float* t, const float* tvec, StepCond& out);
  // velocity f32 [M, 64] for the latent z f32 [M, 64]; txt f32 [T, 6144] (txtmlp output).
  void forward(const Tensor& txt, const Tensor& z, const StepCond& c, const Tensor& vel);
  // The same in pieces (validation): residual = [txt ; first(z)], blocks [l0, l1), final layer.
  void embed(const Tensor& txt, const Tensor& z);
  void run_layers(int l0, int l1, const StepCond& c);
  void head(const StepCond& c, const Tensor& vel);
  Tensor residual() const { return x_; }  // f32 [T + M, 6144]
  WeightFile& weights() { return w_; }
  int layers() const { return layers_; }
  // Attach the Neural Engine half of the linear layers. Required when the weight file holds only the
  // GPU's slices (dit_split: the k | v projection, MLP units [0, H1) and the O-projection); see ane.h.
  void set_ane(class ANEOffload* ane);
  bool needs_ane() const { return split_; }
  int gpu_units() const { return file_units_; }

 private:
  void layer(int l, const StepCond& c);
  // GPU linear layers: the fused q|k|v|gate projection (GPU-only file) or, with the ANE attached, only its
  // k | v columns, into the fused rows `dst` (row stride 15360); the gated O-projection; MLP units [0, H1).
  void lin_qkvg(int l, const Tensor& h, const Tensor& dst, int rows);
  void lin_kv(int l, const Tensor& h, const Tensor& dst, int rows);
  void lin_o(int l, const Tensor& ao, const Tensor& x, int rows, const Tensor& gate);
  void lin_mlp(int l, const Tensor& h, const Tensor& mid, const Tensor& x, int rows, const Tensor& gate);
  // Hybrid layers [l0, l1), pipelined over chunks of the ANE's row count (see model.mm).
  void forward_hybrid(int l0, int l1, const StepCond& c);
  Metal& m_;
  WeightFile w_;
  class ANEOffload* ane_ = nullptr;
  int file_units_ = kFF;
  bool split_ = false;    // the file holds only the GPU's slices (L*.kv instead of L*.qkvg)
  int o_last_ = -1;       // trailing chunks whose O-projection stays on the GPU (KREA_GPU_O_LAST; -1 = auto)
  bool serial_ = false;   // KREA_SERIAL: unpipelined hybrid schedule (for comparison)
  int layers_ = kLayers;  // debug: KREA_DIT_LAYERS runs only the first N blocks
  int T_ = 0, H16_ = 0, W16_ = 0;
  size_t cap_rows_ = 0;
  Tensor x_, h_, qkvg_, ao_, mid_, hf_, cos_, sin_, st_;
  // Head dimension whose QK-norm gain product is extreme (block 0: dim 6, 52.8^2), computed in fp32 through
  // the attention's rank-1 score term; -1 = none. Detected from the norm weights at load.
  std::vector<int> big_dim_;
  Tensor qbig_, kbig_;  // f32 [R, 48] / [R, 12]
  void qk_rope(int l, const Tensor& qkvg_rows, int rows, int row0, bool q = true, bool k = true);
  void attend(int l, const Tensor& q_rows, int rows, int row0, const Tensor& o_rows, int R);
};

// ------------------------------------------------------------------------------------------
// Qwen-Image VAE (Wan 2.1 architecture, single frame): 16-channel latents, 8x up/downsampling. The decoder
// renders every image; the encoder turns an existing image into a latent for editing.
class VAEDecoder {
 public:
  VAEDecoder(Metal& m, const std::string& path);
  // z: f32 [H16 * W16, 64] packed DiT latent (token-major). Writes RGBA8 [16*H16 x 16*W16 x 4] to out.
  void decode(const Tensor& z, int H16, int W16, uint8_t* out_rgba);
  // RGBA8 [H x W x 4] (H, W multiples of 16) -> normalized packed latent f32 [H/16 * W/16, 64] in z (the
  // encoder's mean). Encodes, commits and waits.
  void encode(const uint8_t* rgba, int H, int W, const Tensor& z);
  double gpu_ms() const { return gpu_ms_; }
  WeightFile& weights() { return w_; }

 private:
  double gpu_ms_ = 0;
  double gpu_end_ = 0;
  void run(const Tensor& z, int H16, int W16, uint8_t* out_rgba);
  void run_encode(const uint8_t* rgba, int H, int W, const Tensor& z);
  struct Act {
    Tensor t;
    int H, W, C;
  };
  Act conv3x3(const Act& x, const std::string& name, int cout, const Tensor* residual);
  Act conv1x1(const Act& x, const std::string& name, int cout);
  void conv(int mode, const Act& x, const Tensor& w, const Tensor& b, const Tensor* res, const Tensor& y, int cout);
  Act upsample(const Act& x, const std::string& name, int cout);
  Act downsample(const Act& x, const std::string& name, int cout);
  Act norm_silu(const Act& x, const std::string& gamma, bool silu);
  Act conv3x3_wino(const Act& x, const std::string& name, int cout, const Tensor* residual);
  bool wino(const Act& x, int cout) const;
  int wino_rows(const Act& x, int cout) const;
  int wino_ = 1;  // KREA_VAE_WINO: 0 = direct convs, 1 = convs with >= 288 channels, 2 = all resblock convs
  Act resblock(const Act& x, const std::string& p, int cout);
  Act attention(const Act& x, const std::string& p);
  Tensor scratch(size_t bytes);
  void release(const Tensor& t);
  void trim();
  void prewire(std::initializer_list<size_t> sizes);
  std::vector<Tensor> free_;
  std::vector<std::pair<Tensor, id<MTLCommandBuffer>>> pending_;
  void op_done(const char* op, const Act& y);
  void retire();
  id<MTLCommandBuffer> inflight_ = nil;
  Metal& m_;
  WeightFile w_;
};

// ------------------------------------------------------------------------------------------
// Flow-matching schedule of Krea 2 (sampling.py timesteps with a pinned mu): sigma_i =
// e^mu / (e^mu + 1/t_i - 1) on t = linspace(1, 0, steps + 1); the last entry is 0.
std::vector<float> krea_sigmas(int steps, float mu = 1.15f);

}  // namespace krea
