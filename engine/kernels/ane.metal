// GPU <-> Neural Engine hand-off kernels (Krea DiT hybrid). The Core ML programs use the 1x1-conv layout
// [1, C, 1, R] (channel-major, one chunk of R = ld_t token rows); the GPU keeps token-major rows. A chunked
// buffer is [chunk][channels][ld_t]: row r lives in chunk r / ld_t, column r % ld_t. All kernels move
// 32-row x 64-column tiles through threadgroup memory with 16-byte loads and stores.
#include "common.h"

// ---------------------------------------------------------------------------------------
// Fused RMSNorm hand-off: rrms per row first, then one pass that writes y = x * rrms * mul (+ add) both as
// bf16 rows (GPU GEMM input, row stride ld_y) and as fp16 channel-major ANE chunks (times t_scale); rows in
// [rows, rows_pad) of the ANE layout are zero-filled.
struct RowStatsParams {
  int rows, width;
  float eps;
};
// st[r] = rrms of row r; one simdgroup per row (the row stays in cache for the apply pass).
kernel void rms_row_stats(device const float* x [[buffer(0)]],
                          device float* st [[buffer(1)]],
                          constant RowStatsParams& p [[buffer(2)]],
                          uint tg [[threadgroup_position_in_grid]],
                          ushort sgid [[simdgroup_index_in_threadgroup]],
                          ushort lane [[thread_index_in_simdgroup]]) {
  const int r = tg * 8 + sgid;
  if (r >= p.rows) return;
  device const float4* xr = (device const float4*)(x + (size_t)r * p.width);
  float q = 0.f;
  for (int i = lane; i < p.width / 4; i += 32) {
    const float4 v = xr[i];
    q += dot(v, v);
  }
  const float rrms = rsqrt(simd_sum(q) / p.width + p.eps);
  if (lane == 0) st[r] = rrms;
}

struct DualParams {
  int rows, width, ld_y, ld_t, has_add;
  float t_scale;
};
// grid (width / 64, rows_pad / 32), 256 threads: a 32-row x 64-column tile per threadgroup.
kernel void rms_apply_dual(device const float* x [[buffer(0)]],
                           device const float* st [[buffer(1)]],
                           device const float* mul [[buffer(2)]],
                           device const float* add [[buffer(3)]],
                           device bf16* y [[buffer(4)]],
                           device half* yt [[buffer(5)]],
                           constant DualParams& p [[buffer(6)]],
                           uint2 tg [[threadgroup_position_in_grid]],
                           ushort tid [[thread_index_in_threadgroup]]) {
  threadgroup half tile[64][40];  // [column][row]
  const int c0 = tg.x * 64, r0 = tg.y * 32;
  const int rr = tid / 8, cc = (tid % 8) * 8, r = r0 + rr;
  if (r < p.rows) {
    const float s = st[r];
    device const float4* xr = (device const float4*)(x + (size_t)r * p.width + c0 + cc);
    device const float4* mv = (device const float4*)(mul + c0 + cc);
    float4 a = xr[0] * s * mv[0];
    float4 b = xr[1] * s * mv[1];
    if (p.has_add) {
      device const float4* av = (device const float4*)(add + c0 + cc);
      a += av[0];
      b += av[1];
    }
    device vec<bf16, 4>* yr = (device vec<bf16, 4>*)(y + (size_t)r * p.ld_y + c0 + cc);
    yr[0] = vec<bf16, 4>(a);
    yr[1] = vec<bf16, 4>(b);
    for (int j = 0; j < 4; j++) {
      tile[cc + j][rr] = half(a[j] * p.t_scale);
      tile[cc + 4 + j][rr] = half(b[j] * p.t_scale);
    }
  } else {
    for (int j = 0; j < 8; j++) tile[cc + j][rr] = half(0);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int ch = tid / 4, rs = (tid % 4) * 8;
  device half* o = yt + (size_t)(r0 / p.ld_t) * p.width * p.ld_t + (size_t)(c0 + ch) * p.ld_t + (r0 % p.ld_t) + rs;
  *(device half4*)o = half4(tile[ch][rs], tile[ch][rs + 1], tile[ch][rs + 2], tile[ch][rs + 3]);
  *(device half4*)(o + 4) = half4(tile[ch][rs + 4], tile[ch][rs + 5], tile[ch][rs + 6], tile[ch][rs + 7]);
}

// bf16 rows [rows, width] (row stride ld_in) -> fp16 channel-major ANE chunks ([chunk][width][ld_t]) times
// `scale`; rows in [rows, rows_pad) are zero-filled. grid (width / 64, rows_pad / 32), 256 threads.
struct ToAneParams {
  int rows, width, ld_in, ld_t;
  float scale;
};
kernel void to_ane_bf16(device const bf16* in [[buffer(0)]],
                        device half* out [[buffer(1)]],
                        constant ToAneParams& p [[buffer(2)]],
                        uint2 tg [[threadgroup_position_in_grid]],
                        ushort tid [[thread_index_in_threadgroup]]) {
  threadgroup half tile[64][40];  // [column][row]
  const int c0 = tg.x * 64, r0 = tg.y * 32;
  const int rr = tid / 8, cc = (tid % 8) * 8, r = r0 + rr;
  if (r < p.rows) {
    device const vec<bf16, 4>* ip = (device const vec<bf16, 4>*)(in + (size_t)r * p.ld_in + c0 + cc);
    const float4 a = float4(ip[0]), b = float4(ip[1]);
    for (int j = 0; j < 4; j++) {
      tile[cc + j][rr] = half(a[j] * p.scale);
      tile[cc + 4 + j][rr] = half(b[j] * p.scale);
    }
  } else {
    for (int j = 0; j < 8; j++) tile[cc + j][rr] = half(0);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int ch = tid / 4, rs = (tid % 4) * 8;
  device half* o = out + (size_t)(r0 / p.ld_t) * p.width * p.ld_t + (size_t)(c0 + ch) * p.ld_t + (r0 % p.ld_t) + rs;
  *(device half4*)o = half4(tile[ch][rs], tile[ch][rs + 1], tile[ch][rs + 2], tile[ch][rs + 3]);
  *(device half4*)(o + 4) = half4(tile[ch][rs + 4], tile[ch][rs + 5], tile[ch][rs + 6], tile[ch][rs + 7]);
}

// x(f32)[r, c] += gate[c] * scale * y(f16)[c, r] (y: one channel-major chunk, row stride ld_in) for NY
// inputs y, y2 in turn (explicit fma: one rounding per input, never reassociated).
// grid (width / 64, ceil(rows / 32)), 256 threads.
struct ResidT16Params {
  int rows, width, ld_in;
  float scale;
};
template <int NY>
[[kernel]] void resid_gate_add_t16(device float* x [[buffer(0)]],
                                   device const half* y [[buffer(1)]],
                                   device const float* gate [[buffer(2)]],
                                   constant ResidT16Params& p [[buffer(3)]],
                                   device const half* y2 [[buffer(4)]],
                                   uint2 tg [[threadgroup_position_in_grid]],
                                   ushort tid [[thread_index_in_threadgroup]]) {
  threadgroup float tile[NY][64][33];  // [input][column][row], scale applied
  const int c0 = tg.x * 64, r0 = tg.y * 32;
  const int ch = tid / 4, rs = (tid % 4) * 8;
  for (int k = 0; k < NY; k++) {
    device const half4* yp = (device const half4*)((k ? y2 : y) + (size_t)(c0 + ch) * p.ld_in + r0 + rs);
    const half4 a = yp[0], b = yp[1];
    for (int j = 0; j < 4; j++) {
      tile[k][ch][rs + j] = p.scale * float(a[j]);
      tile[k][ch][rs + 4 + j] = p.scale * float(b[j]);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int rr = tid / 8, cc = (tid % 8) * 8, r = r0 + rr;
  if (r >= p.rows) return;
  device float4* xr = (device float4*)(x + (size_t)r * p.width + c0 + cc);
  device const float4* g = (device const float4*)(gate + c0 + cc);
  float4 u0 = xr[0], u1 = xr[1];
  const float4 g0 = g[0], g1 = g[1];
  for (int k = 0; k < NY; k++) {
    u0 = fma(g0, float4(tile[k][cc][rr], tile[k][cc + 1][rr], tile[k][cc + 2][rr], tile[k][cc + 3][rr]), u0);
    u1 = fma(g1, float4(tile[k][cc + 4][rr], tile[k][cc + 5][rr], tile[k][cc + 6][rr], tile[k][cc + 7][rr]), u1);
  }
  xr[0] = u0;
  xr[1] = u1;
}
template [[host_name("resid_gate_add_t16")]] [[kernel]] void resid_gate_add_t16<1>(
    device float*, device const half*, device const float*, constant ResidT16Params&, device const half*, uint2, ushort);
template [[host_name("resid_gate_add_t16x2")]] [[kernel]] void resid_gate_add_t16<2>(
    device float*, device const half*, device const float*, constant ResidT16Params&, device const half*, uint2, ushort);

// ANE column block -> bf16 rows: in fp16 channel-major [N][ld_in] (one chunk), out[r, col0 + j] = in[j, r].
// grid (N / 64, ceil(rows / 32)), 256 threads (N % 64 == 0).
struct ColsScatterParams {
  int rows, ld_in, ld_out, col0;
};
kernel void ane_cols_scatter(device const half* in [[buffer(0)]],
                             device bf16* out [[buffer(1)]],
                             constant ColsScatterParams& p [[buffer(2)]],
                             uint2 tg [[threadgroup_position_in_grid]],
                             ushort tid [[thread_index_in_threadgroup]]) {
  threadgroup float tile[64][33];  // [channel][row]
  const int j0 = tg.x * 64, r0 = tg.y * 32;
  {
    const int ch = tid / 4, rs = (tid % 4) * 8;
    device const half4* ip = (device const half4*)(in + (size_t)(j0 + ch) * p.ld_in + r0 + rs);
    const half4 a = ip[0], b = ip[1];
    for (int j = 0; j < 4; j++) {
      tile[ch][rs + j] = float(a[j]);
      tile[ch][rs + 4 + j] = float(b[j]);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int rr = tid / 8, cc = (tid % 8) * 8, r = r0 + rr;
  if (r >= p.rows) return;
  device vec<bf16, 4>* o = (device vec<bf16, 4>*)(out + (size_t)r * p.ld_out + p.col0 + j0 + cc);
  o[0] = vec<bf16, 4>(float4(tile[cc][rr], tile[cc + 1][rr], tile[cc + 2][rr], tile[cc + 3][rr]));
  o[1] = vec<bf16, 4>(float4(tile[cc + 4][rr], tile[cc + 5][rr], tile[cc + 6][rr], tile[cc + 7][rr]));
}
