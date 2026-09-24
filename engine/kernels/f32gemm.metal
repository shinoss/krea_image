// fp32 GEMM for the layers Krea 2 ships in F32 (first, last, txtmlp): all operands f32, fp32 simdgroup
// MMA, fp32 accumulation. These layers are small (M.N.K up to a few GFLOP), so this is a plain
// 64x64x16 tile with 2x2 simdgroups (32x32 each), double-buffered threadgroup tiles.
//
//   C[M, N] = A[M, K] (row stride lda) x W[K, N] (row stride ldw; weights stored [in, out])
//   mode bit 0: + bias[n];  bit 1: GELU(tanh) after the bias;  bit 2: accumulate (C += result)
// Host: grid (N / 64, ceil(M / 64)), 128 threads; N % 64 == 0, K % 16 == 0.
#include "common.h"

struct F32GemmParams {
  int M, N, K, lda, ldw, ldc, mode;
};

[[kernel, max_total_threads_per_threadgroup(128)]] void gemm_f32w(
    device const float* A [[buffer(0)]],
    device const float* W [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant F32GemmParams& p [[buffer(3)]],
    device const float* bias [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    ushort sgid [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort tid [[thread_index_in_threadgroup]]) {
  constexpr int BM = 64, BN = 64, BK = 16;
  threadgroup float As[2][BM * BK];
  threadgroup float Bs[2][BK * BN];
  const int m0 = tgid.y * BM, n0 = tgid.x * BN;
  const int sm = sgid / 2, sn = sgid % 2;

  // loaders: A 64x16 = 256 float4 (2 per thread), W 16x64 = 256 float4 (2 per thread)
  const int ar = tid / 4, ac = (tid % 4) * 4;
  const int br = tid / 16, bc = (tid % 16) * 4;
  device const float* a0 = A + (size_t)min(m0 + ar, p.M - 1) * p.lda + ac;
  device const float* a1 = A + (size_t)min(m0 + ar + 32, p.M - 1) * p.lda + ac;
  device const float* b0 = W + (size_t)br * p.ldw + n0 + bc;

  simdgroup_float8x8 acc[4][4];
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.f);

#define LOAD(k0, buf)                                                                                       \
  {                                                                                                         \
    *(threadgroup float4*)&As[buf][ar * BK + ac] = *(device const float4*)(a0 + (k0));                      \
    *(threadgroup float4*)&As[buf][(ar + 32) * BK + ac] = *(device const float4*)(a1 + (k0));               \
    *(threadgroup float4*)&Bs[buf][br * BN + bc] = *(device const float4*)(b0 + (size_t)(k0) * p.ldw);      \
    *(threadgroup float4*)&Bs[buf][(br + 8) * BN + bc] = *(device const float4*)(b0 + (size_t)((k0) + 8) * p.ldw); \
  }
  LOAD(0, 0);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  int buf = 0;
  for (int k0 = 0; k0 < p.K; k0 += BK) {
    if (k0 + BK < p.K) LOAD(k0 + BK, buf ^ 1);
    for (int kk = 0; kk < BK; kk += 8) {
      simdgroup_float8x8 a[4], b[4];
      for (int i = 0; i < 4; i++) simdgroup_load(a[i], &As[buf][(sm * 32 + i * 8) * BK + kk], BK);
      for (int j = 0; j < 4; j++) simdgroup_load(b[j], &Bs[buf][kk * BN + sn * 32 + j * 8], BN);
      for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    buf ^= 1;
  }
#undef LOAD

  const short2 fc = frag_coord(lane);
  for (int i = 0; i < 4; i++) {
    const int row = m0 + sm * 32 + i * 8 + fc.y;
    if (row >= p.M) continue;
    for (int j = 0; j < 4; j++) {
      const int col = n0 + sn * 32 + j * 8 + fc.x;
      thread auto& e = acc[i][j].thread_elements();
      float2 v = float2(e[0], e[1]);
      if (p.mode & 1) v += float2(bias[col], bias[col + 1]);
      if (p.mode & 2) v = float2(gelu_tanh(v.x), gelu_tanh(v.y));
      device float2* o = (device float2*)(C + (size_t)row * p.ldc + col);
      if (p.mode & 4) *o += v;
      else *o = v;
    }
  }
}
