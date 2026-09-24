// Neural Engine co-processor for the Krea DiT: tensor-parallel GPU/ANE split of every layer's linear
// layers (Core ML programs built by tools/build_ane.py).
//
//  * QKVG: the fused q | k | v | gate projection is separable over output columns. The GPU computes
//    columns [0, c0) (weights in the split DiT file); the ANE computes [c0, 15360) and its output is
//    scattered into the GPU's fused buffer before attention (which runs on the GPU for all heads).
//  * MLP: the SwiGLU MLP is separable over its 16384 hidden units. The GPU computes units [0, H1); the ANE
//    computes [H1, 16384). Their partial down-projections are summed into the residual on the GPU.
//  * OPROJ (optional): the full attention output projection for a chunk of rows. It lets the ANE work
//    while the GPU runs attention for the following chunks (DiT::forward_hybrid pipelines by chunk).
//
// ANE programs have one fixed shape: chunks of chunk() token rows in the 1x1-conv layout [1, C, 1, chunk]
// (fp16, channel-major). Rows are [text ; image], so chunk 0 starts with the text rows; the last chunk is
// zero-padded.
//
// Hand-off is asynchronous through two MTLSharedEvents: the GPU signals `go` with a ticket once a job's
// inputs are written (see Metal::split for why the signal ends its command buffer); a serial ANE thread
// runs jobs in ticket order and publishes each finished ticket on `done`, which the GPU waits on before
// reading the job's outputs. Several jobs can be in flight.
//
//   GPU: RMSNorm -> fp16 chunks --signal(go, t)--| more GPU work --wait(done, t)--> merge ANE part
//   ANE thread:              (on go >= t) Core ML predict per chunk --signal(done, t)-->
#pragma once
#include <memory>
#include <string>
#include <vector>

#include "metal.h"

namespace krea {

class ANEOffload {
 public:
  enum Kind { QKVG = 0, MLP = 1, OPROJ = 2 };
  static constexpr int kKinds = 3;
  static constexpr int kMaxChunks = 20;  // up to 20 * chunk rows (2048^2 + 512 text rows = 16896)

  // dir: weights/<ane dir>/ with qkvg_XX, mlp_XX (+ o_XX) .mlmodelc for 28 layers and meta.json.
  ANEOffload(Metal& m, const std::string& dir);
  ~ANEOffload();

  int chunk() const { return chunk_; }
  int gpu_units() const { return gpu_units_; }  // MLP hidden units on the GPU
  int qkvg_c0() const { return qkvg_c0_; }      // fused q|k|v|gate columns on the GPU
  bool has(Kind k) const { return k != OPROJ || has_oproj_; }
  float out_scale() const { return out_scale_; }  // multiply the ANE MLP partial output by this
  float o_scale() const { return o_scale_; }      // O-proj input is pre-scaled by 1/o_scale
  const std::string& mode() const { return mode_; }

  // Per-kind chunked channel-major inputs ([chunk][6144][C], written by the GPU hand-off kernels) and
  // outputs ([chunk][channels][C]; channels = 6144 for MLP and OPROJ, 15360 - c0 for QKVG).
  const Tensor& input(Kind k) const { return in_[k]; }
  const Tensor& output(Kind k) const { return out_[k]; }
  int out_channels(Kind k) const;
  size_t chunk_bytes(int channels) const { return (size_t)channels * chunk_ * 2; }

  // Encode the hand-off for program `k` of `layer` over chunks [c0, c1) at the current point of the
  // GPU stream. Returns the ticket the GPU must wait on before reading those output chunks.
  uint64_t launch(Kind k, int layer, int c0, int c1);
  void gpu_wait(uint64_t ticket);
  double busy_ms() const { return busy_ms_; }
  void reset_stats() { busy_ms_ = 0; }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  Metal& m_;
  int chunk_ = 1024, gpu_units_ = 0, qkvg_c0_ = 0;
  bool has_oproj_ = false;
  float out_scale_ = 1.0f, o_scale_ = 1.0f;
  std::string mode_;
  Tensor in_[kKinds], out_[kKinds];
  double busy_ms_ = 0;
};

}  // namespace krea
