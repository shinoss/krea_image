// Shape-specialized bf16 GEMM with fused epilogues.
//
//   acc[M,N] = A[M,K] (bf16, row stride lda) x Wt[K,N] (bf16, row stride ldw; weights stored
//              transposed = "input-major", laid out offline by tools/convert.py)
//
// Two tile families share one kernel template (see T32 below): the 64x64x16 kernels keep the
// original names and dispatch (drop-in), the 32x32x16 "_t32" kernels are opt-in for the host.
// Each simdgroup computes 32x32 outputs (4x4 fragments) with interleaved 8x8 fragment ownership;
// threadgroup tiles are walked in a grouped (swizzled) order. Tuned on M3 Pro
// (engine/opt/gemm/RESULTS.md): 4.65-4.75 TFLOPS on the DiT projections (96-98% of the
// simdgroup-MMA peak) vs 4.40-4.47 for the previous version, with bit-identical results.
// What made the difference, all measured:
//  - Global->threadgroup tiles are moved with direct 16-byte struct copies into a double-buffered
//    threadgroup tile (next tile copied before the current one is multiplied, one barrier per
//    K-step) instead of staging the next tile in registers across the MMAs.
//  - The K loop is unrolled by 2 so both threadgroup buffer indices are compile-time constants,
//    and the per-lane fragment offsets are computed once.
//  - A/B fragments are read per lane (2 elements each) and widened to float; accumulators live
//    in plain float2 registers, and each MMA builds its 8x8 operands from them (the MLX steel
//    pattern). Each B fragment load is interleaved with the 4 MMAs that consume it.
//  - int8 weights (QW = 1): A is copied directly, the int8 W tile is loaded into registers before
//    the MMAs and dequantized into threadgroup memory after them; scales are reloaded only when
//    the 128-row quantization group changes.
#include "common.h"

#define BK 16
#define SWZ 8

enum Epilogue : int {
  EPI_BF16 = 0,        // C(bf16) = acc
  EPI_F32 = 1,         // C(f32) = acc
  EPI_RESID_GATE = 2,  // X(f32) += gate[n] * acc          (DiT: gated residual, gate = tanh(g))
  EPI_RESID = 3,       // X(f32) += acc                    (text encoder residual)
  EPI_SWIGLU = 4,      // C(bf16)[M, N/2] = silu(g) * u    (gate/up tile-interleaved weights)
  EPI_GELU = 5,        // C(bf16) = gelu_tanh(acc)
  EPI_BIAS_BF16 = 6,   // C(bf16) = acc + bias[n]
  EPI_BIAS_F32 = 7,    // C(f32) = acc + bias[n]
};

struct GemmParams {
  int M, N, K;
  int lda;  // A row stride (elements)
  int ldw;  // Wt row stride (elements)
  int ldc;  // output row stride (elements)
  // Optional column remap for EPI_BF16: output column n goes to (n / seg) * seg_stride + n % seg.
  // Used for the GPU's slice of a head-split Q/K/V projection (seg = 128 * gpu_heads, stride 4096).
  int seg, seg_stride;
  // Optional second product accumulated into the same tile before the epilogue (LR = 1):
  // acc += A2[M, K2] x W2[K2, N] (bf16). Used for exact runtime LoRA: W x + B (A x) with
  // A2 = x A^T (rank columns) and W2 = B^T, so fused epilogues (SwiGLU, gated residual) see
  // the full result.
  int K2, lda2, ldw2;
};

// Weight formats: QW = 0 -> bf16 Wt[K, N]; QW = 1 -> int8 Wq[K, N] with f32 scales
// S[K / kQGroup, N] (symmetric, per output column and group of kQGroup input rows), dequantized
// to bf16 on the way into threadgroup memory.
constant constexpr int kQGroup = 128;

// 16-byte chunk moved global -> threadgroup by one struct copy (lowered to a memcpy).
struct alignas(16) Chunk16 { uint8_t v[16]; };

// Two tile configurations (template parameter T32):
//   T32 = 0: 64x64 tile, 4 simdgroups (2x2), 128 threads. Host: grid (N/64, ceil(M/64)).
//            Kernel names gemm_<epi>[_q8] (unchanged).
//   T32 = 1: 32x32 tile, 1 simdgroup, 32 threads.         Host: grid (N/32, ceil(M/32)).
//            Kernel names gemm_<epi>_t32, bf16 weights only. At K = 4096 it is 0.3-1% faster
//            than T32 = 0 for M = 4096, 1-2% for M = 1024 or 4096 + T and up to 35% for M < 128
//            (more threadgroups), but it moves twice the global->threadgroup traffic: ~10% slower
//            at K = 12288, and slower with int8 weights. Suggested host rule: t32 iff bf16 weights
//            and K <= 4096.
// Larger per-simdgroup tiles (32x64, 64x32) spill registers on this GPU and are much slower.
template <int EPI, int QW, int T32, int LR>
[[kernel, max_total_threads_per_threadgroup(T32 ? 32 : 128)]] void gemm_tiled(
    device const bf16* A [[buffer(0)]],
    device const void* Wv [[buffer(1)]],
    device void* Cv [[buffer(2)]],
    constant GemmParams& p [[buffer(3)]],
    device const float* evec [[buffer(4)]],    // gate / bias (float)
    device const float* wscale [[buffer(5)]],  // int8 weight scales (QW = 1)
    device const bf16* A2 [[buffer(6)]],       // LR = 1: second product (bf16)
    device const bf16* W2 [[buffer(7)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tgn [[threadgroups_per_grid]],
    ushort sgid [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort tid [[thread_index_in_threadgroup]]) {
  constexpr int BM = T32 ? 32 : 64, BN = BM;  // threadgroup tile
  constexpr int WM = T32 ? 1 : 2, WN = WM;     // simdgroup grid
  constexpr int NT = 32 * WM * WN;             // threads
  constexpr int TM = BM / (8 * WM), TN = BN / (8 * WN);  // fragments per simdgroup (4 x 4)
  constexpr int LDA_S = BK, LDB_S = BN;        // threadgroup tiles A[BM][BK], B[BK][BN]
  constexpr int CH = BM * BK / 8 / NT;         // 16-byte chunks of A (and of W) per thread per K-step
  static_assert(CH * NT * 8 == BM * BK && CH * NT * 8 == BK * BN, "chunk split");
  constexpr int A_RS = NT / (BK / 8), B_RS = NT / (BN / 8);  // rows between a thread's chunks
  threadgroup bf16 As[2][BM * LDA_S];
  threadgroup bf16 Bs[2][BK * LDB_S];

  // Grouped tile order: SWZ tile-rows are walked column-major so resident threadgroups
  // share both A row-panels and W column-panels in the GPU caches.
  const uint lin = tgid.y * tgn.x + tgid.x;
  const uint group_size = SWZ * tgn.x;
  const uint first_m = (lin / group_size) * SWZ;
  const uint gm = min((uint)SWZ, tgn.y - first_m);
  const uint in_g = lin % group_size;
  const int m0 = (first_m + in_g % gm) * BM;
  const int n0 = (in_g / gm) * BN;

  int K = p.K, ldw = p.ldw;
  const int lda = p.lda;
  bool qw = QW;  // the LoRA segment is always bf16
  const int sm = sgid / WN, sn = sgid % WN;

  float2 acc[TM][TN];
  for (int i = 0; i < TM; i++)
    for (int j = 0; j < TN; j++) acc[i][j] = float2(0.f);

  // Each thread moves CH 16-byte chunks of A and of W per K-step (rows r, r + *_RS, ...).
  const int a_r = tid / (BK / 8), a_c = (tid % (BK / 8)) * 8;
  const int b_r = tid / (BN / 8), b_c = (tid % (BN / 8)) * 8;
  device const bf16* a_src[CH];
  for (int c = 0; c < CH; c++)  // clamp: rows >= M are computed but never stored
    a_src[c] = A + (size_t)min(m0 + a_r + c * A_RS, p.M - 1) * lda + a_c;
  // Weight column of this thread's chunks. SwiGLU with 32-column tiles: tile n0 / 32 = 2g + h
  // takes gate columns [64g + 16h, +16) and the matching up columns [64g + 32 + 16h, +16) of
  // the tile-interleaved weights, so each simdgroup still holds gate/up pairs (frag j, j + 2).
  int w_col = n0 + b_c;
  if (EPI == EPI_SWIGLU && T32) w_col = (n0 / 64) * 64 + (b_c / 16) * 32 + ((n0 / 32) & 1) * 16 + (b_c & 8);
  device const bf16* b_src = (device const bf16*)Wv + (size_t)b_r * ldw + w_col;
  device const char* q_src = (device const char*)Wv + (size_t)b_r * ldw + w_col;
  device const float* s_src = wscale + w_col;
  threadgroup bf16* a_dst0 = &As[0][a_r * LDA_S + a_c];
  threadgroup bf16* a_dst1 = &As[1][a_r * LDA_S + a_c];
  threadgroup bf16* b_dst0 = &Bs[0][b_r * LDB_S + b_c];
  threadgroup bf16* b_dst1 = &Bs[1][b_r * LDB_S + b_c];

  uint2 rq[CH];      // int8 W chunks (QW = 1)
  float4 s0, s1;     // their scales (all of a thread's chunks share one quantization group)

  // PRE(k, dst): start moving K-step k into buffer dst (bf16: complete copy; int8: A copy +
  // W load into registers). POST(dst): finish it (int8: dequantize W into dst).
#define PRE(k, a_dst, b_dst)                                                                 \
  for (int c = 0; c < CH; c++)                                                               \
    *(threadgroup Chunk16*)((a_dst) + c * A_RS * LDA_S) = *(device const Chunk16*)(a_src[c] + (k)); \
  if (QW && qw) {                                                                                  \
    for (int c = 0; c < CH; c++)                                                             \
      rq[c] = *(device const uint2*)(q_src + (size_t)((k) + c * B_RS) * ldw);                \
    if ((k) % kQGroup == 0) {                                                                \
      device const float4* sp = (device const float4*)(s_src + (size_t)((k) / kQGroup) * ldw); \
      s0 = sp[0];                                                                            \
      s1 = sp[1];                                                                            \
    }                                                                                        \
  } else {                                                                                   \
    for (int c = 0; c < CH; c++)                                                             \
      *(threadgroup Chunk16*)((b_dst) + c * B_RS * LDB_S) =                                  \
          *(device const Chunk16*)(b_src + (size_t)((k) + c * B_RS) * ldw);                  \
  }
#define POST(b_dst)                                                                          \
  if (QW && qw) {                                                                                  \
    for (int c = 0; c < CH; c++) {                                                           \
      const char4 q0 = as_type<char4>(rq[c].x), q1 = as_type<char4>(rq[c].y);               \
      vec<bf16, 8> d;                                                                        \
      d[0] = bf16(float(q0.x) * s0.x); d[1] = bf16(float(q0.y) * s0.y);                      \
      d[2] = bf16(float(q0.z) * s0.z); d[3] = bf16(float(q0.w) * s0.w);                      \
      d[4] = bf16(float(q1.x) * s1.x); d[5] = bf16(float(q1.y) * s1.y);                      \
      d[6] = bf16(float(q1.z) * s1.z); d[7] = bf16(float(q1.w) * s1.w);                      \
      *(threadgroup vec<bf16, 8>*)((b_dst) + c * B_RS * LDB_S) = d;                          \
    }                                                                                        \
  }

  // Lane (fc.y, fc.x..fc.x+1) of every 8x8 fragment. Simdgroup (sm, sn) owns fragment rows
  // sm*8 + i*8*WM and fragment columns sn*8 + j*8*WN (interleaved ownership).
  const short2 fc = frag_coord(lane);
  const int a_off = (sm * 8 + fc.y) * LDA_S + fc.x;
  const int b_off = fc.y * LDB_S + sn * 8 + fc.x;

  // One BK-step of MMAs on buffer `bf` (a literal): load the 4 A fragments, then for each B
  // fragment load it and issue its 4 MMAs.
#define COMPUTE(bf)                                                                          \
  _Pragma("unroll") for (int kk = 0; kk < BK; kk += 8) {                                     \
    float2 a[TM], b[TN];                                                                     \
    simdgroup_barrier(mem_flags::mem_none);                                                  \
    _Pragma("unroll") for (int i = 0; i < TM; i++)                                           \
      a[i] = float2(*(threadgroup const vec<bf16, 2>*)&As[bf][a_off + i * 8 * WM * LDA_S + kk]); \
    _Pragma("unroll") for (int j = 0; j < TN; j++) {                                         \
      b[j] = float2(*(threadgroup const vec<bf16, 2>*)&Bs[bf][b_off + kk * LDB_S + j * 8 * WN]); \
      simdgroup_barrier(mem_flags::mem_none);                                                \
      _Pragma("unroll") for (int i = 0; i < TM; i++) {                                       \
        simdgroup_matrix<float, 8, 8> ma, mb, mc;                                            \
        ma.thread_elements()[0] = a[i][0]; ma.thread_elements()[1] = a[i][1];                \
        mb.thread_elements()[0] = b[j][0]; mb.thread_elements()[1] = b[j][1];                \
        mc.thread_elements()[0] = acc[i][j][0]; mc.thread_elements()[1] = acc[i][j][1];      \
        simdgroup_multiply_accumulate(mc, ma, mb, mc);                                       \
        acc[i][j] = float2(mc.thread_elements()[0], mc.thread_elements()[1]);                \
      }                                                                                      \
    }                                                                                        \
  }

  // Double-buffered threadgroup tiles, one barrier per K-step: step s+1 is moved into the other
  // buffer while step s is multiplied. Unrolled by 2 so buffer indices are literals.
  for (int segment = 0; segment <= LR; segment++) {
  if (segment == 1) {  // switch to the LoRA product; the previous pass ended with a barrier
    K = p.K2;
    ldw = p.ldw2;
    qw = false;
    for (int c = 0; c < CH; c++) a_src[c] = A2 + (size_t)min(m0 + a_r + c * A_RS, p.M - 1) * p.lda2 + a_c;
    b_src = W2 + (size_t)b_r * ldw + w_col;
  }
  PRE(0, a_dst0, b_dst0);
  POST(b_dst0);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  int k0 = 0;
  for (; k0 + 2 * BK <= K; k0 += 2 * BK) {
    PRE(k0 + BK, a_dst1, b_dst1);
    COMPUTE(0);
    POST(b_dst1);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bool has_next = k0 + 2 * BK < K;
    if (has_next) { PRE(k0 + 2 * BK, a_dst0, b_dst0); }
    COMPUTE(1);
    if (has_next) { POST(b_dst0); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (k0 < K) { COMPUTE(0); }  // odd number of K-steps: the last one is in buffer 0
  if (LR) threadgroup_barrier(mem_flags::mem_threadgroup);
  }  // segment
#undef PRE
#undef POST
#undef COMPUTE

  // ---- epilogue ----
  const int ldc = p.ldc;
  if (EPI == EPI_SWIGLU) {
    // Weight columns of 64-column tile t: [gate(t*32 .. t*32+31) | up(t*32 .. t*32+31)]. 64x64:
    // with interleaved ownership, simdgroup sn holds 8-col blocks {sn, sn+2, sn+4, sn+6}, so frag
    // j pairs with j+2. 32x32: the tile holds 16 gate then 16 up columns (see w_col): j with j+2.
    device bf16* C = (device bf16*)Cv;
    const int c0 = (n0 / 64) * 32 + (T32 ? ((n0 / 32) & 1) * 16 : 0);
    for (int i = 0; i < TM; i++) {
      const int row = m0 + sm * 8 + i * 8 * WM + fc.y;
      if (row >= p.M) continue;
      for (int j = 0; j < 2; j++) {
        const float2 g = acc[i][j], u = acc[i][j + 2];
        const int col = c0 + (T32 ? j * 8 : (sn + 2 * j) * 8) + fc.x;
        *(device vec<bf16, 2>*)(C + (size_t)row * ldc + col) =
            vec<bf16, 2>(bf16(silu(g[0]) * u[0]), bf16(silu(g[1]) * u[1]));
      }
    }
    return;
  }
  for (int i = 0; i < TM; i++) {
    const int row = m0 + sm * 8 + i * 8 * WM + fc.y;
    if (row >= p.M) continue;
    for (int j = 0; j < TN; j++) {
      const int col = n0 + sn * 8 + j * 8 * WN + fc.x;
      float2 v = acc[i][j];
      const size_t off = (size_t)row * ldc + col;
      if (EPI == EPI_BF16) {
        const size_t o = p.seg ? (size_t)row * ldc + (col / p.seg) * p.seg_stride + col % p.seg : off;
        *(device vec<bf16, 2>*)((device bf16*)Cv + o) = vec<bf16, 2>(bf16(v.x), bf16(v.y));
      } else if (EPI == EPI_F32) {
        *(device float2*)((device float*)Cv + off) = v;
      } else if (EPI == EPI_RESID_GATE) {
        device float2* X = (device float2*)((device float*)Cv + off);
        *X += v * float2(evec[col], evec[col + 1]);
      } else if (EPI == EPI_RESID) {
        device float2* X = (device float2*)((device float*)Cv + off);
        *X += v;
      } else if (EPI == EPI_GELU) {
        *(device vec<bf16, 2>*)((device bf16*)Cv + off) = vec<bf16, 2>(bf16(gelu_tanh(v.x)), bf16(gelu_tanh(v.y)));
      } else if (EPI == EPI_BIAS_BF16) {
        v += float2(evec[col], evec[col + 1]);
        *(device vec<bf16, 2>*)((device bf16*)Cv + off) = vec<bf16, 2>(bf16(v.x), bf16(v.y));
      } else if (EPI == EPI_BIAS_F32) {
        v += float2(evec[col], evec[col + 1]);
        *(device float2*)((device float*)Cv + off) = v;
      }
    }
  }
}

#define INST_GEMM_LR(name, E, Q, T, L)                                                        \
  template [[host_name(name)]] [[kernel]] void gemm_tiled<E, Q, T, L>(                        \
      device const bf16*, device const void*, device void*, constant GemmParams&,             \
      device const float*, device const float*, device const bf16*, device const bf16*,       \
      uint2, uint2, ushort, ushort, ushort);
#define INST_GEMM(name, E, Q, T) INST_GEMM_LR(name, E, Q, T, 0)

INST_GEMM("gemm_bf16", EPI_BF16, 0, 0)
INST_GEMM("gemm_f32", EPI_F32, 0, 0)
INST_GEMM("gemm_resid_gate", EPI_RESID_GATE, 0, 0)
INST_GEMM("gemm_resid", EPI_RESID, 0, 0)
INST_GEMM("gemm_swiglu", EPI_SWIGLU, 0, 0)
INST_GEMM("gemm_gelu", EPI_GELU, 0, 0)
INST_GEMM("gemm_bias_bf16", EPI_BIAS_BF16, 0, 0)
INST_GEMM("gemm_bias_f32", EPI_BIAS_F32, 0, 0)
INST_GEMM("gemm_bf16_q8", EPI_BF16, 1, 0)
INST_GEMM("gemm_f32_q8", EPI_F32, 1, 0)
INST_GEMM("gemm_resid_gate_q8", EPI_RESID_GATE, 1, 0)
INST_GEMM("gemm_resid_q8", EPI_RESID, 1, 0)
INST_GEMM("gemm_swiglu_q8", EPI_SWIGLU, 1, 0)
INST_GEMM("gemm_gelu_q8", EPI_GELU, 1, 0)
INST_GEMM("gemm_bf16_t32", EPI_BF16, 0, 1)
INST_GEMM("gemm_f32_t32", EPI_F32, 0, 1)
INST_GEMM("gemm_resid_gate_t32", EPI_RESID_GATE, 0, 1)
INST_GEMM("gemm_resid_t32", EPI_RESID, 0, 1)
INST_GEMM("gemm_swiglu_t32", EPI_SWIGLU, 0, 1)
INST_GEMM("gemm_gelu_t32", EPI_GELU, 0, 1)
INST_GEMM("gemm_bias_bf16_t32", EPI_BIAS_BF16, 0, 1)
INST_GEMM("gemm_bias_f32_t32", EPI_BIAS_F32, 0, 1)
// runtime-LoRA variants (bf16 weights, 64x64 tiles)
INST_GEMM_LR("gemm_bf16_lr", EPI_BF16, 0, 0, 1)
INST_GEMM_LR("gemm_resid_gate_lr", EPI_RESID_GATE, 0, 0, 1)
INST_GEMM_LR("gemm_swiglu_lr", EPI_SWIGLU, 0, 0, 1)
INST_GEMM_LR("gemm_f32_lr", EPI_F32, 0, 0, 1)

// ---------------------------------------------------------------------------------------
// GEMV for the per-step conditioning vectors (M = 1..4 rows): y[r, n] = sum_k x[r, k] Wt[k, n].
// One thread per output column; the K loop reads Wt rows coalesced across the simdgroup.
struct GemvParams {
  int rows, N, K, ldw;
  int act;  // bit 0: silu applied to x on load; bit 1: accumulate into y (y += x W)
};

kernel void gemv_bf16(device const float* x [[buffer(0)]],
                      device const bf16* W [[buffer(1)]],
                      device float* y [[buffer(2)]],
                      constant GemvParams& p [[buffer(3)]],
                      uint gid [[thread_position_in_grid]]) {
  const int n = gid;
  if (n >= p.N) return;
  float acc[4] = {0, 0, 0, 0};
  for (int k = 0; k < p.K; k++) {
    const float w = float(W[(size_t)k * p.ldw + n]);
    for (int r = 0; r < p.rows; r++) {
      float xv = x[r * p.K + k];
      if (p.act & 1) xv = silu(xv);
      acc[r] += xv * w;
    }
  }
  for (int r = 0; r < p.rows; r++) y[r * p.N + n] = (p.act & 2) ? y[r * p.N + n] + acc[r] : acc[r];
}
