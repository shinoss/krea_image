// Normalization, RoPE, text-fusion attention and small utility kernels (Krea 2 Turbo).
#include "common.h"

inline float tg_sum(float v, threadgroup float* scratch, ushort sgid, ushort lane) {
  v = simd_sum(v);
  if (lane == 0) scratch[sgid] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float t = lane < 8 ? scratch[lane] : 0.f;
  t = simd_sum(t);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return t;
}

// ---------------------------------------------------------------------------------------
// Row RMSNorm with a modulation: y = x * rrms * mul[c] (+ add[c]), rrms = 1/sqrt(mean(x^2) + eps).
// Everything in fp32, one rounding at the store (bf16 or f32 output). Covers:
//   Krea DiT pre-norms:  mul = (1 + w) * (1 + scale), add = shift   (per layer and step)
//   Krea final layer:    mul = (1 + w) * (1 + scale_f), add = shift_f (f32 output)
//   text fusion / txtmlp: mul = 1 + w                                 (no add)
//   Qwen3 text encoder:   mul = w                                     (no add)
// x: f32 rows (stride ld_in), width % 4 == 0. One 256-thread threadgroup per row.
struct RmsParams {
  int width, ld_in, ld_out;
  float eps;
  int has_add;
};

template <typename TO>
[[kernel]] void rms_norm(device const float* x [[buffer(0)]],
                         device const float* mul [[buffer(1)]],
                         device const float* add [[buffer(2)]],
                         device TO* y [[buffer(3)]],
                         constant RmsParams& p [[buffer(4)]],
                         uint row [[threadgroup_position_in_grid]],
                         ushort tid [[thread_index_in_threadgroup]],
                         ushort sgid [[simdgroup_index_in_threadgroup]],
                         ushort lane [[thread_index_in_simdgroup]]) {
  threadgroup float scratch[8];
  device const float4* xr = (device const float4*)(x + (size_t)row * p.ld_in);
  const int n4 = p.width / 4;
  float q = 0.f;
  for (int i = tid; i < n4; i += 256) {
    const float4 v = xr[i];
    q += dot(v, v);
  }
  const float rrms = rsqrt(tg_sum(q, scratch, sgid, lane) / p.width + p.eps);
  device const float4* mv = (device const float4*)mul;
  device const float4* av = (device const float4*)add;
  device vec<TO, 4>* yr = (device vec<TO, 4>*)(y + (size_t)row * p.ld_out);
  for (int i = tid; i < n4; i += 256) {
    float4 o = xr[i] * rrms * mv[i];
    if (p.has_add) o += av[i];
    yr[i] = vec<TO, 4>(o);
  }
}
template [[host_name("rms_norm_bf16")]] [[kernel]] void rms_norm<bf16>(
    device const float*, device const float*, device const float*, device bf16*, constant RmsParams&, uint, ushort, ushort, ushort);
template [[host_name("rms_norm_f32")]] [[kernel]] void rms_norm<float>(
    device const float*, device const float*, device const float*, device float*, constant RmsParams&, uint, ushort, ushort, ushort);

// ---------------------------------------------------------------------------------------
// Per-head RMSNorm (f32 weight [128]) + rotary embedding, in place on a fused projection buffer.
// One simdgroup per (row, head); lane holds 4 consecutive channels.
//   rope_mode 0: interleaved pairs (2i, 2i+1) share frequency i      (Krea DiT)
//   rope_mode 1: rotate-half, channel i pairs with i + 64            (Qwen3 text encoder)
//   rope_mode 2: no rotation                                         (Krea text fusion)
// cos/sin: f32 [rows, 64] per-row tables (frequency i of that row's position).
// big_dim >= 0 (Krea block 0): that head dimension carries a QK-norm gain of ~52.8 on both q and k, so its
// q.k term dominates the logits (up to ~3e4) and bf16 rounding of it flips the near-hard softmax. Its
// post-norm/RoPE value goes to qbig [rows, n_q] / kbig [rows, n_k] in fp32 and is zeroed in the bf16 buffer;
// the attention kernel adds the rank-1 fp32 term qbig kbig^T to the scores.
struct QKNormParams {
  int rows, ld;
  int q_off, n_q;  // column offset / head count of the queries
  int k_off, n_k;  // column offset / head count of the keys
  float eps;
  int rope_mode;
  int big_dim;
};

kernel void qk_norm_rope(device bf16* qkv [[buffer(0)]],
                         device const float* wq [[buffer(1)]],
                         device const float* wk [[buffer(2)]],
                         device const float* cosb [[buffer(3)]],
                         device const float* sinb [[buffer(4)]],
                         constant QKNormParams& p [[buffer(5)]],
                         device float* qbig [[buffer(6)]],
                         device float* kbig [[buffer(7)]],
                         uint gid [[thread_position_in_grid]]) {
  const uint lane = gid & 31;
  const uint unit = gid >> 5;
  const int heads = p.n_q + p.n_k;
  const int row = unit / heads;
  const int hh = unit % heads;
  if (row >= p.rows) return;
  const bool is_q = hh < p.n_q;
  const int col = is_q ? p.q_off + hh * kHeadDim : p.k_off + (hh - p.n_q) * kHeadDim;
  device vec<bf16, 4>* ptr = (device vec<bf16, 4>*)(qkv + (size_t)row * p.ld + col) + lane;
  device const float4* w = (device const float4*)(is_q ? wq : wk);

  float4 x = float4(*ptr);
  const float rrms = rsqrt(simd_sum(dot(x, x)) * (1.0f / kHeadDim) + p.eps);
  x = x * rrms * w[lane];
  float4 o = x;
  if (p.rope_mode == 0) {
    device const float* cr = cosb + (size_t)row * 64;
    device const float* sr = sinb + (size_t)row * 64;
    const int f = lane * 2;
    const float c0 = cr[f], s0 = sr[f], c1 = cr[f + 1], s1 = sr[f + 1];
    o.x = x.x * c0 - x.y * s0;
    o.y = x.x * s0 + x.y * c0;
    o.z = x.z * c1 - x.w * s1;
    o.w = x.z * s1 + x.w * c1;
  } else if (p.rope_mode == 1) {
    device const float* cr = cosb + (size_t)row * 64;
    device const float* sr = sinb + (size_t)row * 64;
    float4 other;  // partner half lives in lane ^ 16
    other.x = simd_shuffle_xor(x.x, 16);
    other.y = simd_shuffle_xor(x.y, 16);
    other.z = simd_shuffle_xor(x.z, 16);
    other.w = simd_shuffle_xor(x.w, 16);
    const int f = (lane & 15) * 4;
    const float4 c = float4(cr[f], cr[f + 1], cr[f + 2], cr[f + 3]);
    const float4 s = float4(sr[f], sr[f + 1], sr[f + 2], sr[f + 3]);
    o = lane < 16 ? x * c - other * s : x * c + other * s;
  }
  if (p.big_dim >= 0 && (int)lane == p.big_dim / 4) {
    const int e = p.big_dim % 4;
    if (is_q) qbig[(size_t)row * p.n_q + hh] = o[e];
    else kbig[(size_t)row * p.n_k + (hh - p.n_q)] = o[e];
    o[e] = 0.0f;
  }
  *ptr = vec<bf16, 4>(o);
}

// ---------------------------------------------------------------------------------------
// Text fusion, layerwise blocks: every token attends across its own 12 tapped states. Rows are
// token-major (row = t * 12 + j) in a fused [q | k | v | gate] buffer (ld columns, heads of 128 at
// q_off, k_off, v_off, g_off). out[row, h*128 + d] = softmax_j(q.k_j / sqrt(128)) v_j * sigmoid(gate).
// One simdgroup per (row, head); lane holds 4 of the 128 channels. Scores and softmax in fp32.
struct LayerAttnParams {
  int rows, ld, ldo, heads;
  int q_off, k_off, v_off, g_off;
  float scale;
};
constant constexpr int kTaps = 12;

kernel void layerwise_attn(device const bf16* qkvg [[buffer(0)]],
                           device bf16* out [[buffer(1)]],
                           constant LayerAttnParams& p [[buffer(2)]],
                           uint gid [[thread_position_in_grid]]) {
  const uint lane = gid & 31;
  const uint unit = gid >> 5;
  const int row = unit / p.heads, h = unit % p.heads;
  if (row >= p.rows) return;
  const int base = row - row % kTaps;
  device const bf16* qr = qkvg + (size_t)row * p.ld + p.q_off + h * kHeadDim;
  const float4 q = float4(*((device const vec<bf16, 4>*)qr + lane));
  float s[kTaps];
  float mx = -INFINITY;
  for (int j = 0; j < kTaps; j++) {
    device const bf16* kr = qkvg + (size_t)(base + j) * p.ld + p.k_off + h * kHeadDim;
    s[j] = simd_sum(dot(q, float4(*((device const vec<bf16, 4>*)kr + lane)))) * p.scale;
    mx = max(mx, s[j]);
  }
  float l = 0.f;
  float4 o = 0.f;
  for (int j = 0; j < kTaps; j++) {
    const float e = exp(s[j] - mx);
    l += e;
    device const bf16* vr = qkvg + (size_t)(base + j) * p.ld + p.v_off + h * kHeadDim;
    o += e * float4(*((device const vec<bf16, 4>*)vr + lane));
  }
  device const bf16* gr = qkvg + (size_t)row * p.ld + p.g_off + h * kHeadDim;
  const float4 g = float4(*((device const vec<bf16, 4>*)gr + lane));
  o = o / l * float4(sigmoid(g.x), sigmoid(g.y), sigmoid(g.z), sigmoid(g.w));
  *((device vec<bf16, 4>*)(out + (size_t)row * p.ldo + h * kHeadDim) + lane) = vec<bf16, 4>(o);
}

// Text fusion projector: y[t, c] = sum_j w[j] * x[t * 12 + j, c]  (f32). Thread per 4 channels.
kernel void tap_project(device const float* x [[buffer(0)]],
                        device const float* w [[buffer(1)]],
                        device float* y [[buffer(2)]],
                        constant int2& p [[buffer(3)]],  // (rows T, width)
                        uint2 gid [[thread_position_in_grid]]) {
  const int t = gid.y, c = gid.x * 4;
  if (t >= p.x || c >= p.y) return;
  float4 acc = 0.f;
  for (int j = 0; j < kTaps; j++) acc += w[j] * *(device const float4*)(x + (size_t)(t * kTaps + j) * p.y + c);
  *(device float4*)(y + (size_t)t * p.y + c) = acc;
}

// Strided f32 row copy: dst[r * ld_dst + c] = src[r * ld_src + c], c < cols (cols % 4 == 0).
struct CopyParams {
  int rows, cols, ld_src, ld_dst;
};
kernel void copy_rows_f32(device const float* src [[buffer(0)]],
                          device float* dst [[buffer(1)]],
                          constant CopyParams& p [[buffer(2)]],
                          uint2 gid [[thread_position_in_grid]]) {
  const int r = gid.y, c = gid.x * 4;
  if (r >= p.rows || c >= p.cols) return;
  *(device float4*)(dst + (size_t)r * p.ld_dst + c) = *(device const float4*)(src + (size_t)r * p.ld_src + c);
}

// Copy a [rows, cols] bf16 block between strided buffers (cols multiple of 8).
kernel void copy_rows(device const bf16* src [[buffer(0)]],
                      device bf16* dst [[buffer(1)]],
                      constant CopyParams& p [[buffer(2)]],
                      uint2 gid [[thread_position_in_grid]]) {
  const int r = gid.y, c = gid.x * 8;
  if (r >= p.rows || c >= p.cols) return;
  *(device uint4*)(dst + (size_t)r * p.ld_dst + c) = *(device const uint4*)(src + (size_t)r * p.ld_src + c);
}

// ---------------------------------------------------------------------------------------
// Sampler and latent helpers (f32 latents in token layout [M, 64]).
//   z <- cz z + c0 x0 + c1 x0_prev with x0 = z - sigma v; then x0_prev <- x0.
// Euler (the Krea sampler): cz = s'/s, c0 = 1 - s'/s, c1 = 0 (host computes the coefficients).
struct FlowStepParams {
  int n;
  float sigma, cz, c0, c1;
};
kernel void flow_step(device float* z [[buffer(0)]], device const float* v [[buffer(1)]],
                      device float* x0p [[buffer(2)]], constant FlowStepParams& p [[buffer(3)]],
                      uint gid [[thread_position_in_grid]]) {
  if ((int)gid >= p.n) return;
  const float x0 = z[gid] - p.sigma * v[gid];
  z[gid] = p.cz * z[gid] + p.c0 * x0 + p.c1 * x0p[gid];
  x0p[gid] = x0;
}

// Plain Euler step: z += dt * v (exact flow-matching Euler, as sampling.py).
kernel void euler_step(device float* z [[buffer(0)]], device const float* v [[buffer(1)]],
                       constant float2& p [[buffer(2)]],  // (n, dt)
                       uint gid [[thread_position_in_grid]]) {
  if ((int)gid < (int)p.x) z[gid] += p.y * v[gid];
}

// f32 helpers over [n] (n % 4 == 0, float4 per thread).
kernel void f32_copy(device const float4* x [[buffer(0)]], device float4* y [[buffer(1)]],
                     uint gid [[thread_position_in_grid]]) {
  y[gid] = x[gid];
}
kernel void f32_axpy(device const float4* x [[buffer(0)]], device float4* y [[buffer(1)]],
                     constant float& a [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  y[gid] += a * x[gid];
}
kernel void f32_to_bf16(device const float* x [[buffer(0)]], device bf16* y [[buffer(1)]],
                        uint gid [[thread_position_in_grid]]) {
  y[gid] = bf16(x[gid]);
}

// True classifier-free guidance, Krea convention: v <- v_cond + scale * (v_cond - v_neg).
struct CfgParams {
  int n;
  float scale;
};
kernel void cfg_combine(device float* vc [[buffer(0)]], device const float* vn [[buffer(1)]],
                        constant CfgParams& p [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if ((int)gid < p.n) vc[gid] = vc[gid] + p.scale * (vc[gid] - vn[gid]);
}
