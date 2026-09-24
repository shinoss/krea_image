// Neural Engine co-processor for the Krea DiT: tensor-parallel GPU/ANE split of every layer's linear
// layers (Core ML programs built by tools/build_ane.py).
//
//  * QG: the fused q | k | v | gate projection is separable over output columns. The GPU computes the
//    k | v columns (weights in the split DiT file); the ANE computes q | gate, which the GPU scatters into
//    its fused buffer before attention (which runs on the GPU for all heads). Attention of a chunk needs
//    every chunk's K/V but only its own Q, so K/V come from the GPU and the ANE's Q can arrive per chunk.
//  * MLP: the SwiGLU MLP is separable over its 16384 hidden units. The GPU computes units [0, H1); the ANE
//    computes [H1, 16384). Their partial down-projections are summed into the residual on the GPU.
//  * OPROJ (optional): the full attention output projection for a chunk of rows. It lets the ANE work
//    while the GPU runs attention for the following chunks (DiT::forward_hybrid pipelines by chunk).
//
// ANE programs have one fixed shape: chunks of chunk() token rows in the 1x1-conv layout [1, C, 1, chunk]
// (fp16, channel-major). Rows are [text ; image], so chunk 0 starts with the text rows; the last chunk is
// zero-padded. Every chunk's input and output is its own IOSurface, wrapped both as a Metal buffer (no copy)
// and as a pixel-buffer-backed MLMultiArray, so Core ML hands it to the ANE without copying (with plain
// data-pointer arrays it copies inputs and outputs on every call, which also competes with the GPU for
// memory bandwidth).
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
  enum Kind { QG = 0, MLP = 1, OPROJ = 2 };
  static constexpr int kKinds = 3;
  static constexpr int kMaxChunks = 20;  // up to 20 * chunk rows (2048^2 + 512 text rows = 16896)

  // dir: weights/<ane dir>/ with qg_XX, mlp_XX (+ o_XX) .mlmodelc for 28 layers and meta.json.
  ANEOffload(Metal& m, const std::string& dir);
  ~ANEOffload();

  int chunk() const { return chunk_; }
  int gpu_units() const { return gpu_units_; }  // MLP hidden units on the GPU
  bool has(Kind k) const { return k != OPROJ || has_oproj_; }
  float out_scale() const { return out_scale_; }  // multiply the ANE MLP partial output by this
  float o_scale() const { return o_scale_; }      // O-proj input is pre-scaled by 1/o_scale
  const std::string& mode() const { return mode_; }

  // Chunk c's channel-major input ([6144][C], written by the GPU hand-off kernels) and output
  // ([channels][C]; channels = 6144 for MLP and OPROJ, 12288 = q | gate for QG) of program kind k.
  // ensure_chunks(n) allocates chunks [0, n) (call before encoding work on them).
  void ensure_chunks(int n);
  Tensor input(Kind k, int c) const;
  Tensor output(Kind k, int c) const;
  int out_channels(Kind k) const;
  size_t chunk_bytes(int channels) const { return (size_t)channels * chunk_ * 2; }

  // Encode the hand-off for program `k` of `layer` over chunks [c0, c1) at the current point of the
  // GPU stream. Returns the ticket the GPU must wait on before reading those output chunks.
  uint64_t launch(Kind k, int layer, int c0, int c1);
  void gpu_wait(uint64_t ticket);
  double busy_ms() const { return busy_ms_; }
  void reset_stats() { busy_ms_ = 0; }
  // KREA_ANE_STATS=1: the ANE thread scans every chunk's fp16 input and output (max |x|, non-finite count)
  // per program kind and layer; report() prints the table. Debug only (a CPU pass over every buffer).
  void report() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  Metal& m_;
  int chunk_ = 1024, gpu_units_ = 0;
  bool has_oproj_ = false;
  float out_scale_ = 1.0f, o_scale_ = 1.0f;
  std::string mode_;
  double busy_ms_ = 0;
};

}  // namespace krea
