// Shared definitions for the Krea 2 Turbo Metal kernels.
#pragma once
#include <metal_stdlib>
using namespace metal;

typedef bfloat bf16;

// Model constants (Krea 2 DiT / text fusion / Qwen3-VL-4B text encoder). Row widths of the norm and
// glue kernels are parameters; these are the shapes the hot paths are specialized for.
constant constexpr int kDit = 6144;     // DiT hidden size
constant constexpr int kText = 2560;    // text encoder / text fusion hidden size
constant constexpr int kHeadDim = 128;

// Lane coordinate inside an 8x8 simdgroup matrix: lane holds (row fm, cols fn and fn+1).
// Lanes l, l^1, l^8, l^9 hold the same row.
inline short2 frag_coord(ushort lane) {
  const short qid = lane / 4;
  const short fm = (qid & 4) + ((lane / 2) % 4);
  const short fn = (qid & 2) * 2 + (lane % 2) * 2;
  return short2(fn, fm);
}

inline float silu(float x) { return x / (1.0f + exp(-x)); }
inline float sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }

inline float gelu_tanh(float x) {
  const float k0 = 0.7978845608028654f, k1 = 0.044715f;
  return 0.5f * x * (1.0f + precise::tanh(k0 * (x + k1 * x * x * x)));
}
