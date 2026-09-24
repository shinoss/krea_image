// VAE decoder kernels (NHWC bf16 activations, fp32 accumulation).
#include "common.h"

// ---------------------------------------------------------------------------------------
// Implicit-GEMM convolution, KS x KS kernel (KS = 1 or 3, stride 1, zero padding KS/2):
//   out[p, n] = bias[n] + sum_{tap, ci} in[src(p, tap), ci] * Wt[tap * Cin + ci, n] (+ res[p, n])
// M = output pixels, N = Cout (multiple of BN), K = KS*KS*Cin (Cin multiple of 16).
// UP = 1 fuses a nearest-neighbour 2x upsample of the input (the input is H/2 x W/2).
// Tile BM=64 x BN x BK=16 with the same 2x2-simdgroup interleaved MMA core as gemm.metal.
struct ConvParams {
  int H, W;      // output spatial size
  int Cin, Cout;
  int up;        // input is (H/2, W/2) and nearest-upsampled on the fly
  int has_res;
};

template <int KS>
inline uint4 conv_load_a(device const bf16* in, int k0, int Cin, int a_c, int py, int px, int H, int W,
                         int Win, int up) {
  const int tap = k0 / Cin, ci = k0 % Cin + a_c;
  int sy = py, sx = px;
  if (KS == 3) {
    sy += tap / 3 - 1;
    sx += tap % 3 - 1;
    if (sy < 0 || sy >= H || sx < 0 || sx >= W) return uint4(0);
  }
  if (up) {
    sy >>= 1;
    sx >>= 1;
  }
  return *(device const uint4*)(in + ((size_t)sy * Win + sx) * Cin + ci);
}

template <int KS, int BN>
[[kernel]] void conv_igemm(device const bf16* in [[buffer(0)]],
                           device const bf16* Wt [[buffer(1)]],
                           device const float* bias [[buffer(2)]],
                           device const bf16* res [[buffer(3)]],
                           device bf16* out [[buffer(4)]],
                           constant ConvParams& p [[buffer(5)]],
                           uint2 tgid [[threadgroup_position_in_grid]],
                           ushort sgid [[simdgroup_index_in_threadgroup]],
                           ushort lane [[thread_index_in_simdgroup]],
                           ushort tid [[thread_index_in_threadgroup]]) {
  constexpr int BM = 64, BK = 16, WM = 2, WN = 2;
  constexpr int TM = BM / (8 * WM), TN = BN / (8 * WN);
  constexpr int B_CHUNKS = BK * BN / 8;
  threadgroup bf16 As[2][BM * BK];
  threadgroup bf16 Bs[2][BK * BN];

  const int M = p.H * p.W;
  const int K = KS * KS * p.Cin;
  const int m0 = tgid.y * BM, n0 = tgid.x * BN;
  const int sm = sgid / WN, sn = sgid % WN;
  const int Hin = p.up ? p.H / 2 : p.H, Win = p.up ? p.W / 2 : p.W;

  // A loader: thread -> (row a_r, 8-channel chunk a_c) of the BM x BK tile
  const int a_r = tid / 2, a_c = (tid % 2) * 8;
  const int pix = min(m0 + a_r, M - 1);
  const int py = pix / p.W, px = pix % p.W;
  const int b_r = tid / (BN / 8), b_c = (tid % (BN / 8)) * 8;
  const bool b_active = tid < B_CHUNKS;

  simdgroup_matrix<float, 8, 8> acc[TM][TN];
  for (int i = 0; i < TM; i++)
    for (int j = 0; j < TN; j++) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.f);

  (void)Hin;
#define LOAD_A(k0) conv_load_a<KS>(in, (k0), p.Cin, a_c, py, px, p.H, p.W, Win, p.up)
  uint4 ra = LOAD_A(0);
  uint4 rb = b_active ? *(device const uint4*)(Wt + (size_t)b_r * p.Cout + n0 + b_c) : uint4(0);
  *(threadgroup uint4*)(&As[0][a_r * BK + a_c]) = ra;
  if (b_active) *(threadgroup uint4*)(&Bs[0][b_r * BN + b_c]) = rb;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  int buf = 0;
  for (int k0 = 0; k0 < K; k0 += BK) {
    const bool has_next = k0 + BK < K;
    if (has_next) {
      ra = LOAD_A(k0 + BK);
      if (b_active) rb = *(device const uint4*)(Wt + (size_t)(k0 + BK + b_r) * p.Cout + n0 + b_c);
    }
#pragma unroll
    for (int kk = 0; kk < BK; kk += 8) {
      simdgroup_matrix<bf16, 8, 8> a[TM], b[TN];
      simdgroup_barrier(mem_flags::mem_none);
#pragma unroll
      for (int i = 0; i < TM; i++) simdgroup_load(a[i], &As[buf][(sm * 8 + i * 8 * WM) * BK + kk], BK);
      simdgroup_barrier(mem_flags::mem_none);
#pragma unroll
      for (int j = 0; j < TN; j++) simdgroup_load(b[j], &Bs[buf][kk * BN + sn * 8 + j * 8 * WN], BN);
      simdgroup_barrier(mem_flags::mem_none);
#pragma unroll
      for (int i = 0; i < TM; i++)
#pragma unroll
        for (int j = 0; j < TN; j++) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    }
    if (has_next) {
      *(threadgroup uint4*)(&As[buf ^ 1][a_r * BK + a_c]) = ra;
      if (b_active) *(threadgroup uint4*)(&Bs[buf ^ 1][b_r * BN + b_c]) = rb;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    buf ^= 1;
  }

  const short2 fc = frag_coord(lane);
  for (int i = 0; i < TM; i++) {
    const int row = m0 + sm * 8 + i * 8 * WM + fc.y;
    if (row >= M) continue;
    for (int j = 0; j < TN; j++) {
      const int col = n0 + sn * 8 + j * 8 * WN + fc.x;
      thread auto& e = acc[i][j].thread_elements();
      float2 v = float2(e[0], e[1]) + float2(bias[col], bias[col + 1]);
      const size_t off = (size_t)row * p.Cout + col;
      if (p.has_res) v += float2(*(device const vec<bf16, 2>*)(res + off));
      *(device vec<bf16, 2>*)(out + off) = vec<bf16, 2>(bf16(v.x), bf16(v.y));
    }
  }
}

#define INST_CONV(name, KS, BN)                                                                   \
  template [[host_name(name)]] [[kernel]] void conv_igemm<KS, BN>(                                \
      device const bf16*, device const bf16*, device const float*, device const bf16*, device bf16*, \
      constant ConvParams&, uint2, ushort, ushort, ushort);
INST_CONV("conv3x3_bn64", 3, 64)
INST_CONV("conv3x3_bn48", 3, 48)
INST_CONV("conv1x1_bn64", 1, 64)
INST_CONV("conv1x1_bn48", 1, 48)

// ---------------------------------------------------------------------------------------
// Upsampler (nearest 2x, then 3x3 conv) in sub-pixel form: output pixel (2i + py, 2j + px) only
// sees the low-resolution pixels i + py - 1 + r, j + px - 1 + s (r, s in {0, 1}), so each of the
// four output phases is a 2x2 conv over the low-res input whose taps are sums of 3x3 taps
// (vae_subpix_weights). Zero padding of the upsampled image is zero padding of the low-res input,
// so borders are exact; 4 instead of 9 taps per output pixel.

// W2[(t*Cin + ci), phase*Cout + co] = sum of the 3x3 taps (ky, kx) of W[(ky*3 + kx)*Cin + ci, co]
// that fall on low-res offset (r, s) = (t >> 1, t & 1) for phase (py, px) = (phase >> 1, phase & 1):
// rows ky in [0, py] for r = 0 and [1 + py, 2] for r = 1 (columns likewise). Summed in fp32, rounded
// once to bf16. One thread per 8 output columns.
kernel void vae_subpix_weights(device const bf16* Wt [[buffer(0)]],
                               device bf16* W2 [[buffer(1)]],
                               constant int2& p [[buffer(2)]],  // (Cin, Cout)
                               uint2 gid [[thread_position_in_grid]]) {
  const int Cin = p.x, Cout = p.y;
  const int col = gid.x * 8, row = gid.y;
  if (col >= 4 * Cout || row >= 4 * Cin) return;
  const int t = row / Cin, ci = row % Cin, ph = col / Cout, co = col % Cout;
  const int r = t >> 1, s = t & 1, py = ph >> 1, px = ph & 1;
  const int y0 = r ? 1 + py : 0, y1 = r ? 2 : py, x0 = s ? 1 + px : 0, x1 = s ? 2 : px;
  float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int ky = y0; ky <= y1; ky++)
    for (int kx = x0; kx <= x1; kx++) {
      const vec<bf16, 8> w = *(device const vec<bf16, 8>*)(Wt + ((size_t)(ky * 3 + kx) * Cin + ci) * Cout + co);
      for (int k = 0; k < 8; k++) acc[k] += float(w[k]);
    }
  vec<bf16, 8> o;
  for (int k = 0; k < 8; k++) o[k] = bf16(acc[k]);
  *(device vec<bf16, 8>*)(W2 + (size_t)row * 4 * Cout + col) = o;
}

// ---------------------------------------------------------------------------------------
// Implicit-GEMM convolutions with the tuned core of kernels/gemm.metal (same 64 x BN x 16 tile,
// 2x2 simdgroups and K order as conv_igemm, so the products and their accumulation order are the
// same): the next K-step is copied global -> threadgroup (double buffer, one barrier per step, K
// loop unrolled by 2) before the current one is multiplied, fragments are read per lane and
// widened to float2 accumulators, and tiles are walked in grouped order. The K index walks
// (tap, channel) incrementally, so a thread recomputes its A source pixel only once per tap.
//   MODE 1: 1x1 conv, MODE 3: 3x3 conv (zero padding 1); M = H*W pixels, N = Cout, K = taps*Cin,
//           has_res adds res[p, n] (same layout as the output).
//   MODE 2: sub-pixel upsampler on the H x W low-res input: N = 4*Cout (column = phase*Cout + co,
//           phase = 2*py + px), K = 4*Cin with W2 from vae_subpix_weights; each BN tile lies in one
//           phase and scatters its rows into the 2H x 2W output. has_res fuses the DupUp3D shortcut
//           out[2i + py, 2j + px, co] += res[i, j, (((co*ft + ft-1)*2 + py)*2 + px) / repeats].
struct VConvParams {
  int H, W;  // input spatial size (MODE 2: low-res; the output is 2H x 2W)
  int Cin, Cout;
  int has_res, Cres, ft, repeats;
};

template <int MODE, int BN>
[[kernel, max_total_threads_per_threadgroup(128)]] void vconv(device const bf16* in [[buffer(0)]],
                                                              device const bf16* Wt [[buffer(1)]],
                                                              device const float* bias [[buffer(2)]],
                                                              device const bf16* res [[buffer(3)]],
                                                              device bf16* out [[buffer(4)]],
                                                              constant VConvParams& p [[buffer(5)]],
                                                              uint2 tgid [[threadgroup_position_in_grid]],
                                                              uint2 tgn [[threadgroups_per_grid]],
                                                              ushort sgid [[simdgroup_index_in_threadgroup]],
                                                              ushort lane [[thread_index_in_simdgroup]],
                                                              ushort tid [[thread_index_in_threadgroup]]) {
  constexpr int BM = 64, BK = 16, WM = 2, WN = 2, SWZ = 8;
  constexpr int TM = BM / (8 * WM), TN = BN / (8 * WN);
  constexpr int TAPS = MODE == 3 ? 9 : MODE == 2 ? 4 : 1;
  constexpr int B_CHUNKS = BK * BN / 8;
  threadgroup bf16 As[2][BM * BK];
  threadgroup bf16 Bs[2][BK * BN];

  const int M = p.H * p.W, N = MODE == 2 ? 4 * p.Cout : p.Cout, K = TAPS * p.Cin;
  // Grouped tile order: SWZ tile-rows are walked column-major so resident threadgroups share
  // both input rows and weight column panels in the GPU caches.
  const uint lin = tgid.y * tgn.x + tgid.x;
  const uint group_size = SWZ * tgn.x;
  const uint first_m = (lin / group_size) * SWZ;
  const uint gm = min((uint)SWZ, tgn.y - first_m);
  const uint in_g = lin % group_size;
  const int m0 = (first_m + in_g % gm) * BM;
  const int n0 = (in_g / gm) * BN;
  const int phase = MODE == 2 ? n0 / p.Cout : 0, py = phase >> 1, px = phase & 1;
  const int sm = sgid / WN, sn = sgid % WN;

  // A chunk of this thread: tile row a_r (pixel (y, x)), channels [a_c, a_c + 8) of each K-step
  const int a_r = tid / 2, a_c = (tid % 2) * 8;
  const int pix = min(m0 + a_r, M - 1);  // rows >= M are computed but never stored
  const int y = pix / p.W, x = pix % p.W;
  const int b_r = tid / (BN / 8), b_c = (tid % (BN / 8)) * 8;
  const bool b_active = tid < B_CHUNKS;
  device const bf16* b_src = Wt + (size_t)b_r * N + n0 + b_c;
  threadgroup bf16* a_dst0 = &As[0][a_r * BK + a_c];
  threadgroup bf16* a_dst1 = &As[1][a_r * BK + a_c];
  threadgroup bf16* b_dst0 = &Bs[0][b_r * BN + b_c];
  threadgroup bf16* b_dst1 = &Bs[1][b_r * BN + b_c];

  int tap = 0, ci = 0;
  bool a_ok = true;
  device const bf16* a_src = in + (size_t)pix * p.Cin + a_c;
#define SET_TAP()                                                              \
  {                                                                            \
    int sy = y, sx = x;                                                        \
    if (MODE == 3) {                                                           \
      sy += tap / 3 - 1;                                                       \
      sx += tap % 3 - 1;                                                       \
    } else if (MODE == 2) {                                                    \
      sy += py - 1 + (tap >> 1);                                               \
      sx += px - 1 + (tap & 1);                                                \
    }                                                                          \
    a_ok = sy >= 0 && sy < p.H && sx >= 0 && sx < p.W;                         \
    a_src = in + (size_t)(a_ok ? sy * p.W + sx : 0) * p.Cin + a_c;             \
  }
  if (MODE != 1) SET_TAP();

  // PRE(k, a_dst, b_dst): copy K-step k (the next one in K order) into a threadgroup buffer.
#define PRE(k, a_dst, b_dst)                                                                   \
  {                                                                                            \
    *(threadgroup uint4*)(a_dst) = a_ok ? *(device const uint4*)(a_src + ci) : uint4(0);       \
    if (b_active) *(threadgroup uint4*)(b_dst) = *(device const uint4*)(b_src + (size_t)(k) * N); \
    ci += BK;                                                                                  \
    if (MODE != 1 && ci == p.Cin) {                                                            \
      ci = 0;                                                                                  \
      tap++;                                                                                   \
      SET_TAP();                                                                               \
    }                                                                                          \
  }

  float2 acc[TM][TN];
  for (int i = 0; i < TM; i++)
    for (int j = 0; j < TN; j++) acc[i][j] = float2(0.f);
  const short2 fc = frag_coord(lane);
  const int a_off = (sm * 8 + fc.y) * BK + fc.x;
  const int b_off = fc.y * BN + sn * 8 + fc.x;

#define COMPUTE(bf)                                                                               \
  _Pragma("unroll") for (int kk = 0; kk < BK; kk += 8) {                                          \
    float2 a[TM], b[TN];                                                                          \
    simdgroup_barrier(mem_flags::mem_none);                                                       \
    _Pragma("unroll") for (int i = 0; i < TM; i++)                                                \
      a[i] = float2(*(threadgroup const vec<bf16, 2>*)&As[bf][a_off + i * 8 * WM * BK + kk]);     \
    _Pragma("unroll") for (int j = 0; j < TN; j++) {                                              \
      b[j] = float2(*(threadgroup const vec<bf16, 2>*)&Bs[bf][b_off + kk * BN + j * 8 * WN]);     \
      simdgroup_barrier(mem_flags::mem_none);                                                     \
      _Pragma("unroll") for (int i = 0; i < TM; i++) {                                            \
        simdgroup_matrix<float, 8, 8> ma, mb, mc;                                                 \
        ma.thread_elements()[0] = a[i][0]; ma.thread_elements()[1] = a[i][1];                     \
        mb.thread_elements()[0] = b[j][0]; mb.thread_elements()[1] = b[j][1];                     \
        mc.thread_elements()[0] = acc[i][j][0]; mc.thread_elements()[1] = acc[i][j][1];           \
        simdgroup_multiply_accumulate(mc, ma, mb, mc);                                            \
        acc[i][j] = float2(mc.thread_elements()[0], mc.thread_elements()[1]);                     \
      }                                                                                           \
    }                                                                                             \
  }

  PRE(0, a_dst0, b_dst0);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  int k0 = 0;
  for (; k0 + 2 * BK <= K; k0 += 2 * BK) {
    PRE(k0 + BK, a_dst1, b_dst1);
    COMPUTE(0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (k0 + 2 * BK < K) PRE(k0 + 2 * BK, a_dst0, b_dst0);
    COMPUTE(1);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (k0 < K) COMPUTE(0);  // odd number of K-steps: the last one is in buffer 0
#undef SET_TAP
#undef PRE
#undef COMPUTE

  if (MODE == 2) {
    const int W2 = 2 * p.W;
    for (int i = 0; i < TM; i++) {
      const int row = m0 + sm * 8 + i * 8 * WM + fc.y;
      if (row >= M) continue;
      const int ry = row / p.W, rx = row % p.W;
      device bf16* o = out + ((size_t)(2 * ry + py) * W2 + 2 * rx + px) * p.Cout;
      for (int j = 0; j < TN; j++) {
        const int co = n0 - phase * p.Cout + sn * 8 + j * 8 * WN + fc.x;
        float2 v = acc[i][j] + float2(bias[co], bias[co + 1]);
        if (p.has_res) {
          device const bf16* rr = res + (size_t)row * p.Cres;
          const int s0 = (((co * p.ft + p.ft - 1) * 2 + py) * 2 + px) / p.repeats;
          const int s1 = ((((co + 1) * p.ft + p.ft - 1) * 2 + py) * 2 + px) / p.repeats;
          v += float2(rr[s0], rr[s1]);
        }
        *(device vec<bf16, 2>*)(o + co) = vec<bf16, 2>(bf16(v.x), bf16(v.y));
      }
    }
    return;
  }
  for (int i = 0; i < TM; i++) {
    const int row = m0 + sm * 8 + i * 8 * WM + fc.y;
    if (row >= M) continue;
    for (int j = 0; j < TN; j++) {
      const int col = n0 + sn * 8 + j * 8 * WN + fc.x;
      float2 v = acc[i][j] + float2(bias[col], bias[col + 1]);
      const size_t off = (size_t)row * p.Cout + col;
      if (p.has_res) v += float2(*(device const vec<bf16, 2>*)(res + off));
      *(device vec<bf16, 2>*)(out + off) = vec<bf16, 2>(bf16(v.x), bf16(v.y));
    }
  }
}

#define INST_VCONV(name, MODE, BN)                                                                \
  template [[host_name(name)]] [[kernel]] void vconv<MODE, BN>(                                   \
      device const bf16*, device const bf16*, device const float*, device const bf16*, device bf16*, \
      constant VConvParams&, uint2, uint2, ushort, ushort, ushort);
INST_VCONV("vconv3x3_bn64", 3, 64)
INST_VCONV("vconv3x3_bn48", 3, 48)
INST_VCONV("vconv1x1_bn64", 1, 64)
INST_VCONV("vconv1x1_bn48", 1, 48)
INST_VCONV("vconv_up2_bn64", 2, 64)
INST_VCONV("vconv_up2_bn48", 2, 48)

// ---------------------------------------------------------------------------------------
// Winograd F(2x2, 3x3) for the resblock 3x3 convs: out = A^T [(G g G^T) . (B^T d B)] A per 2x2
// output tile (d: the 4x4 input patch at (2ty - 1, 2tx - 1), zero padded), 16 products per tile
// instead of 36. Three passes: vae_wino_in (V = B^T d B), 16 GEMMs M[e] = V[e] U[e] (vae_wino_gemm,
// e = 4*xi + nu), vae_wino_out (A^T M A + bias + residual).
// V and U are fp16 (11-bit mantissa; a bf16 V/U would add ~2x the conv's own rounding error): the
// convs' inputs are RMS-norm (+ SiLU) outputs, |y_c| <= sqrt(C) |gamma_c| <= 228 for this model, so
// |V| <= 4 * 228, far inside fp16's range. M is fp32 (the output transform cancels terms).
//   B^T = [1 0 -1 0; 0 1 1 0; 0 -1 1 0; 0 1 0 -1], G = [1 0 0; .5 .5 .5; .5 -.5 .5; 0 0 1],
//   A^T = [1 1 1 0; 0 1 -1 -1].
struct WinoParams {
  int H, W, C;  // conv input size / channels (H, W even)
  int Cout, has_res;
  int ty0, th;  // band of tile rows [ty0, ty0 + th): T = th * W/2 tiles (V and M hold one band)
};

// U[e][ci][co] = (G g G^T)[xi][nu] with g[ky][kx] = Wt[(ky*3 + kx)*Cin + ci, co]. Thread per
// (8 output columns, ci).
kernel void vae_wino_weights(device const bf16* Wt [[buffer(0)]],
                             device half* U [[buffer(1)]],
                             constant int2& p [[buffer(2)]],  // (Cin, Cout)
                             uint2 gid [[thread_position_in_grid]]) {
  const int Cin = p.x, Cout = p.y, co = gid.x * 8, ci = gid.y;
  if (co >= Cout || ci >= Cin) return;
  float g[3][3][8];
  for (int t = 0; t < 9; t++) {
    const vec<bf16, 8> w = *(device const vec<bf16, 8>*)(Wt + ((size_t)t * Cin + ci) * Cout + co);
    for (int k = 0; k < 8; k++) g[t / 3][t % 3][k] = float(w[k]);
  }
  // rows: Gg[xi][j] = sum_i G[xi][i] g[i][j]
  float Gg[4][3][8];
  for (int j = 0; j < 3; j++)
    for (int k = 0; k < 8; k++) {
      Gg[0][j][k] = g[0][j][k];
      Gg[1][j][k] = 0.5f * (g[0][j][k] + g[1][j][k] + g[2][j][k]);
      Gg[2][j][k] = 0.5f * (g[0][j][k] - g[1][j][k] + g[2][j][k]);
      Gg[3][j][k] = g[2][j][k];
    }
  for (int xi = 0; xi < 4; xi++) {
    vec<half, 8> u[4];
    for (int k = 0; k < 8; k++) {
      const float a = Gg[xi][0][k], b = Gg[xi][1][k], c = Gg[xi][2][k];
      u[0][k] = half(a);
      u[1][k] = half(0.5f * (a + b + c));
      u[2][k] = half(0.5f * (a - b + c));
      u[3][k] = half(c);
    }
    for (int nu = 0; nu < 4; nu++)
      *(device vec<half, 8>*)(U + ((size_t)(xi * 4 + nu) * Cin + ci) * Cout + co) = u[nu];
  }
}

// V[e][t][c] for tile t = ty * (W/2) + tx. Thread per (8 channels, tile).
kernel void vae_wino_in(device const bf16* in [[buffer(0)]],
                        device half* V [[buffer(1)]],
                        constant WinoParams& p [[buffer(2)]],
                        uint2 gid [[thread_position_in_grid]]) {
  const int c = gid.x * 8, t = gid.y, TW = p.W / 2, T = p.th * TW;
  if (c >= p.C || t >= T) return;
  const int y0 = (p.ty0 + t / TW) * 2 - 1, x0 = (t % TW) * 2 - 1;
  float d[4][4][8];
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++) {
      const int y = y0 + i, x = x0 + j;
      const bool ok = y >= 0 && y < p.H && x >= 0 && x < p.W;
      const vec<bf16, 8> v = ok ? *(device const vec<bf16, 8>*)(in + ((size_t)y * p.W + x) * p.C + c) : vec<bf16, 8>(0);
      for (int k = 0; k < 8; k++) d[i][j][k] = float(v[k]);
    }
  // columns: e[i][nu] = (d B)[i][nu]
  for (int i = 0; i < 4; i++)
    for (int k = 0; k < 8; k++) {
      const float a = d[i][0][k], b = d[i][1][k], cc = d[i][2][k], e = d[i][3][k];
      d[i][0][k] = a - cc;
      d[i][1][k] = b + cc;
      d[i][2][k] = cc - b;
      d[i][3][k] = b - e;
    }
  const size_t es = (size_t)T * p.C;  // stride between the 16 matrices
  device half* o = V + (size_t)t * p.C + c;
  for (int nu = 0; nu < 4; nu++) {
    vec<half, 8> r0, r1, r2, r3;
    for (int k = 0; k < 8; k++) {
      const float a = d[0][nu][k], b = d[1][nu][k], cc = d[2][nu][k], e = d[3][nu][k];
      r0[k] = half(a - cc);
      r1[k] = half(b + cc);
      r2[k] = half(cc - b);
      r3[k] = half(b - e);
    }
    *(device vec<half, 8>*)(o + (0 * 4 + nu) * es) = r0;
    *(device vec<half, 8>*)(o + (1 * 4 + nu) * es) = r1;
    *(device vec<half, 8>*)(o + (2 * 4 + nu) * es) = r2;
    *(device vec<half, 8>*)(o + (3 * 4 + nu) * es) = r3;
  }
}

// M[e] = V[e] [T, C] x U[e] [C, Cout] (fp32 out), e = tgid.z; same core as vconv (MODE 1).
template <int BN>
[[kernel, max_total_threads_per_threadgroup(128)]] void vae_wino_gemm(device const half* V [[buffer(0)]],
                                                                      device const half* U [[buffer(1)]],
                                                                      device float* Mo [[buffer(2)]],
                                                                      constant WinoParams& p [[buffer(3)]],
                                                                      uint3 tgid [[threadgroup_position_in_grid]],
                                                                      uint3 tgn [[threadgroups_per_grid]],
                                                                      ushort sgid [[simdgroup_index_in_threadgroup]],
                                                                      ushort lane [[thread_index_in_simdgroup]],
                                                                      ushort tid [[thread_index_in_threadgroup]]) {
  constexpr int BM = 64, BK = 16, WM = 2, WN = 2, SWZ = 8;
  constexpr int TM = BM / (8 * WM), TN = BN / (8 * WN);
  constexpr int B_CHUNKS = BK * BN / 8;
  threadgroup half As[2][BM * BK];
  threadgroup half Bs[2][BK * BN];

  const int M = p.th * (p.W / 2), N = p.Cout, K = p.C;
  const uint lin = tgid.y * tgn.x + tgid.x;
  const uint group_size = SWZ * tgn.x;
  const uint first_m = (lin / group_size) * SWZ;
  const uint gm = min((uint)SWZ, tgn.y - first_m);
  const uint in_g = lin % group_size;
  const int m0 = (first_m + in_g % gm) * BM;
  const int n0 = (in_g / gm) * BN;
  const int sm = sgid / WN, sn = sgid % WN;
  V += (size_t)tgid.z * M * K;
  U += (size_t)tgid.z * K * N;
  Mo += (size_t)tgid.z * M * N;

  const int a_r = tid / 2, a_c = (tid % 2) * 8;
  const int b_r = tid / (BN / 8), b_c = (tid % (BN / 8)) * 8;
  const bool b_active = tid < B_CHUNKS;
  device const half* a_src = V + (size_t)min(m0 + a_r, M - 1) * K + a_c;
  device const half* b_src = U + (size_t)b_r * N + n0 + b_c;
  threadgroup half* a_dst0 = &As[0][a_r * BK + a_c];
  threadgroup half* a_dst1 = &As[1][a_r * BK + a_c];
  threadgroup half* b_dst0 = &Bs[0][b_r * BN + b_c];
  threadgroup half* b_dst1 = &Bs[1][b_r * BN + b_c];
#define PRE(k, a_dst, b_dst)                                                                   \
  {                                                                                            \
    *(threadgroup uint4*)(a_dst) = *(device const uint4*)(a_src + (k));                       \
    if (b_active) *(threadgroup uint4*)(b_dst) = *(device const uint4*)(b_src + (size_t)(k) * N); \
  }

  float2 acc[TM][TN];
  for (int i = 0; i < TM; i++)
    for (int j = 0; j < TN; j++) acc[i][j] = float2(0.f);
  const short2 fc = frag_coord(lane);
  const int a_off = (sm * 8 + fc.y) * BK + fc.x;
  const int b_off = fc.y * BN + sn * 8 + fc.x;
#define COMPUTE(bf)                                                                               \
  _Pragma("unroll") for (int kk = 0; kk < BK; kk += 8) {                                          \
    float2 a[TM], b[TN];                                                                          \
    simdgroup_barrier(mem_flags::mem_none);                                                       \
    _Pragma("unroll") for (int i = 0; i < TM; i++)                                                \
      a[i] = float2(*(threadgroup const vec<half, 2>*)&As[bf][a_off + i * 8 * WM * BK + kk]);     \
    _Pragma("unroll") for (int j = 0; j < TN; j++) {                                              \
      b[j] = float2(*(threadgroup const vec<half, 2>*)&Bs[bf][b_off + kk * BN + j * 8 * WN]);     \
      simdgroup_barrier(mem_flags::mem_none);                                                     \
      _Pragma("unroll") for (int i = 0; i < TM; i++) {                                            \
        simdgroup_matrix<float, 8, 8> ma, mb, mc;                                                 \
        ma.thread_elements()[0] = a[i][0]; ma.thread_elements()[1] = a[i][1];                     \
        mb.thread_elements()[0] = b[j][0]; mb.thread_elements()[1] = b[j][1];                     \
        mc.thread_elements()[0] = acc[i][j][0]; mc.thread_elements()[1] = acc[i][j][1];           \
        simdgroup_multiply_accumulate(mc, ma, mb, mc);                                            \
        acc[i][j] = float2(mc.thread_elements()[0], mc.thread_elements()[1]);                     \
      }                                                                                           \
    }                                                                                             \
  }
  PRE(0, a_dst0, b_dst0);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  int k0 = 0;
  for (; k0 + 2 * BK <= K; k0 += 2 * BK) {
    PRE(k0 + BK, a_dst1, b_dst1);
    COMPUTE(0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (k0 + 2 * BK < K) PRE(k0 + 2 * BK, a_dst0, b_dst0);
    COMPUTE(1);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (k0 < K) COMPUTE(0);
#undef PRE
#undef COMPUTE
  for (int i = 0; i < TM; i++) {
    const int row = m0 + sm * 8 + i * 8 * WM + fc.y;
    if (row >= M) continue;
    for (int j = 0; j < TN; j++) {
      const int col = n0 + sn * 8 + j * 8 * WN + fc.x;
      *(device float2*)(Mo + (size_t)row * N + col) = acc[i][j];
    }
  }
}
template [[host_name("vae_wino_gemm_bn64")]] [[kernel]] void vae_wino_gemm<64>(
    device const half*, device const half*, device float*, constant WinoParams&, uint3, uint3, ushort, ushort, ushort);
template [[host_name("vae_wino_gemm_bn48")]] [[kernel]] void vae_wino_gemm<48>(
    device const half*, device const half*, device float*, constant WinoParams&, uint3, uint3, ushort, ushort, ushort);

// out[2ty + a, 2tx + b, co] = (A^T M A)[a][b] + bias[co] (+ res, same layout as out). Thread per
// (8 output channels, tile).
kernel void vae_wino_out(device const float* Mi [[buffer(0)]],
                         device const float* bias [[buffer(1)]],
                         device const bf16* res [[buffer(2)]],
                         device bf16* out [[buffer(3)]],
                         constant WinoParams& p [[buffer(4)]],
                         uint2 gid [[thread_position_in_grid]]) {
  const int co = gid.x * 8, t = gid.y, TW = p.W / 2, T = p.th * TW;
  if (co >= p.Cout || t >= T) return;
  const size_t es = (size_t)T * p.Cout;
  device const float* mi = Mi + (size_t)t * p.Cout + co;
  // rows: s[a][nu] = sum_xi A^T[a][xi] M[xi][nu]
  float s[2][4][8];
  for (int nu = 0; nu < 4; nu++) {
    float m[4][8];
    for (int xi = 0; xi < 4; xi++) {
      const float4 lo = *(device const float4*)(mi + (xi * 4 + nu) * es);
      const float4 hi = *(device const float4*)(mi + (xi * 4 + nu) * es + 4);
      m[xi][0] = lo.x; m[xi][1] = lo.y; m[xi][2] = lo.z; m[xi][3] = lo.w;
      m[xi][4] = hi.x; m[xi][5] = hi.y; m[xi][6] = hi.z; m[xi][7] = hi.w;
    }
    for (int k = 0; k < 8; k++) {
      s[0][nu][k] = m[0][k] + m[1][k] + m[2][k];
      s[1][nu][k] = m[1][k] - m[2][k] - m[3][k];
    }
  }
  const float4 b0 = *(device const float4*)(bias + co), b1 = *(device const float4*)(bias + co + 4);
  const float bb[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
  const int y = (p.ty0 + t / TW) * 2, x = (t % TW) * 2;
  for (int a = 0; a < 2; a++)
    for (int b = 0; b < 2; b++) {
      const size_t off = ((size_t)(y + a) * p.W + x + b) * p.Cout + co;
      vec<bf16, 8> r = p.has_res ? *(device const vec<bf16, 8>*)(res + off) : vec<bf16, 8>(0);
      vec<bf16, 8> o;
      for (int k = 0; k < 8; k++) {
        const float v = b == 0 ? s[a][0][k] + s[a][1][k] + s[a][2][k] : s[a][1][k] - s[a][2][k] - s[a][3][k];
        o[k] = bf16(v + bb[k] + float(r[k]));
      }
      *(device vec<bf16, 8>*)(out + off) = o;
    }
}

// ---------------------------------------------------------------------------------------
// Channel RMS norm (F.normalize over C * sqrt(C) * gamma) with optional SiLU. One simdgroup
// per pixel; C is a multiple of 8.
struct NormParams {
  int P, C, silu;
};
kernel void vae_norm_silu(device const bf16* x [[buffer(0)]],
                          device const float* gamma [[buffer(1)]],
                          device bf16* y [[buffer(2)]],
                          constant NormParams& p [[buffer(3)]],
                          uint gid [[thread_position_in_grid]]) {
  const uint lane = gid & 31, pix = gid >> 5;
  if ((int)pix >= p.P) return;
  device const bf16* xr = x + (size_t)pix * p.C;
  device bf16* yr = y + (size_t)pix * p.C;
  const int chunks = p.C / 8;
  float ss = 0.f;
  for (int c = lane; c < chunks; c += 32) {
    const vec<bf16, 8> v = *(device const vec<bf16, 8>*)(xr + c * 8);
    for (int k = 0; k < 8; k++) ss += float(v[k]) * float(v[k]);
  }
  ss = simd_sum(ss);
  const float scale = sqrt((float)p.C) / max(sqrt(ss), 1e-12f);
  for (int c = lane; c < chunks; c += 32) {
    const vec<bf16, 8> v = *(device const vec<bf16, 8>*)(xr + c * 8);
    vec<bf16, 8> o;
    for (int k = 0; k < 8; k++) {
      float t = float(v[k]) * scale * gamma[c * 8 + k];
      if (p.silu) t = silu(t);
      o[k] = bf16(t);
    }
    *(device vec<bf16, 8>*)(yr + c * 8) = o;
  }
}

// Same op with L lanes per pixel (32 / L pixels per simdgroup): lane q of a pixel handles chunks
// q, q + L, ...; the partial sums are combined with shuffles within the L-lane group.
template <int L>
[[kernel]] void vae_norm_silu_g(device const bf16* x [[buffer(0)]],
                                device const float* gamma [[buffer(1)]],
                                device bf16* y [[buffer(2)]],
                                constant NormParams& p [[buffer(3)]],
                                uint gid [[thread_position_in_grid]]) {
  const uint lane = gid & 31, q = lane % L;
  const uint pix0 = (gid >> 5) * (32 / L) + lane / L;
  const bool valid = (int)pix0 < p.P;
  const uint pix = valid ? pix0 : p.P - 1;
  device const bf16* xr = x + (size_t)pix * p.C;
  device bf16* yr = y + (size_t)pix * p.C;
  const int chunks = p.C / 8;
  float ss = 0.f;
  for (int c = q; c < chunks; c += L) {
    const vec<bf16, 8> v = *(device const vec<bf16, 8>*)(xr + c * 8);
    for (int k = 0; k < 8; k++) ss += float(v[k]) * float(v[k]);
  }
  for (ushort o = L / 2; o > 0; o /= 2) ss += simd_shuffle_xor(ss, o);
  const float scale = sqrt((float)p.C) / max(sqrt(ss), 1e-12f);
  if (!valid) return;
  for (int c = q; c < chunks; c += L) {
    const vec<bf16, 8> v = *(device const vec<bf16, 8>*)(xr + c * 8);
    const float4 g0 = *(device const float4*)(gamma + c * 8), g1 = *(device const float4*)(gamma + c * 8 + 4);
    const float g[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
    vec<bf16, 8> o;
    for (int k = 0; k < 8; k++) {
      float t = float(v[k]) * scale * g[k];
      if (p.silu) t = silu(t);
      o[k] = bf16(t);
    }
    *(device vec<bf16, 8>*)(yr + c * 8) = o;
  }
}
template [[host_name("vae_norm_silu_l1")]] [[kernel]] void vae_norm_silu_g<1>(device const bf16*, device const float*,
                                                                              device bf16*, constant NormParams&, uint);
template [[host_name("vae_norm_silu_l2")]] [[kernel]] void vae_norm_silu_g<2>(device const bf16*, device const float*,
                                                                              device bf16*, constant NormParams&, uint);
template [[host_name("vae_norm_silu_l4")]] [[kernel]] void vae_norm_silu_g<4>(device const bf16*, device const float*,
                                                                              device bf16*, constant NormParams&, uint);
template [[host_name("vae_norm_silu_l8")]] [[kernel]] void vae_norm_silu_g<8>(device const bf16*, device const float*,
                                                                              device bf16*, constant NormParams&, uint);
template [[host_name("vae_norm_silu_l16")]] [[kernel]] void vae_norm_silu_g<16>(device const bf16*, device const float*,
                                                                                device bf16*, constant NormParams&, uint);

// ---------------------------------------------------------------------------------------
// DupUp3D shortcut of the residual up-blocks (single frame): out[Y, X, c] +=
//   in[Y/2, X/2, (((c*ft + ft-1)*2 + (Y&1))*2 + (X&1)) / repeats]
struct DupParams {
  int H, W, Cin, Cout, ft, repeats;
};
kernel void vae_dup_up_add(device const bf16* in [[buffer(0)]],
                           device bf16* out [[buffer(1)]],
                           constant DupParams& p [[buffer(2)]],
                           uint2 gid [[thread_position_in_grid]]) {
  const int pix = gid.y, c = gid.x;
  if (c >= p.Cout || pix >= p.H * p.W) return;
  const int Y = pix / p.W, X = pix % p.W;
  const int j = ((c * p.ft + p.ft - 1) * 2 + (Y & 1)) * 2 + (X & 1);
  const int src = j / p.repeats;
  const size_t si = ((size_t)(Y >> 1) * (p.W / 2) + (X >> 1)) * p.Cin + src;
  const size_t di = (size_t)pix * p.Cout + c;
  out[di] = bf16(float(out[di]) + float(in[si]));
}

// Unpatchify + de-normalize + post_quant_conv of the DiT latent into the VAE input (NHWC bf16
// [2*H16, 2*W16, 16]): token (i, j) of the [H16 * W16, 64] f32 latent holds features (c, py, px) =
// c*4 + py*2 + px, i.e. the 2x2 block of latent pixels (2i + py, 2j + px). Per pixel: u = z * std + mean,
// then y = W u + b (post_quant_conv, 16x16 f32 [in][out]). Thread per pixel.
kernel void vae_unpack_pq(device const float* z [[buffer(0)]],
                          device const float* mean [[buffer(1)]],
                          device const float* stdv [[buffer(2)]],
                          device const float* w [[buffer(3)]],
                          device const float* b [[buffer(4)]],
                          device bf16* y [[buffer(5)]],
                          constant int& W16 [[buffer(6)]],
                          uint pix [[thread_position_in_grid]]) {
  const int W8 = 2 * W16, yy = pix / W8, xx = pix % W8;
  device const float* zt = z + (size_t)((yy >> 1) * W16 + (xx >> 1)) * 64 + (yy & 1) * 2 + (xx & 1);
  float u[16];
  for (int c = 0; c < 16; c++) u[c] = zt[c * 4] * stdv[c] + mean[c];
  device bf16* yo = y + (size_t)pix * 16;
  for (int o = 0; o < 16; o++) {
    float acc = b[o];
    for (int c = 0; c < 16; c++) acc += w[c * 16 + o] * u[c];
    yo[o] = bf16(acc);
  }
}

// K^T for the mid-block attention: out[c, p] = qkv[p, off + c]
struct TransParams {
  int P, C, ld, off, ldo;
};
kernel void vae_transpose(device const bf16* qkv [[buffer(0)]],
                          device bf16* out [[buffer(1)]],
                          constant TransParams& p [[buffer(2)]],
                          uint2 gid [[thread_position_in_grid]]) {
  const int pix = gid.x, c = gid.y;
  if (pix >= p.P || c >= p.C) return;
  out[(size_t)c * p.ldo + pix] = qkv[(size_t)pix * p.ld + p.off + c];
}

// Row softmax: S f32 [rows, n] * scale -> P bf16. One threadgroup (256) per row.
struct SoftmaxParams {
  int n, ld;  // valid columns, row stride
  float scale;
};
kernel void softmax_rows(device const float* S [[buffer(0)]],
                         device bf16* P [[buffer(1)]],
                         constant SoftmaxParams& p [[buffer(2)]],
                         uint row [[threadgroup_position_in_grid]],
                         ushort tid [[thread_index_in_threadgroup]],
                         ushort sgid [[simdgroup_index_in_threadgroup]],
                         ushort lane [[thread_index_in_simdgroup]]) {
  threadgroup float scratch[8];
  device const float* s = S + (size_t)row * p.ld;
  float mx = -INFINITY;
  for (int i = tid; i < p.n; i += 256) mx = max(mx, s[i] * p.scale);
  mx = simd_max(mx);
  if (lane == 0) scratch[sgid] = mx;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  mx = simd_max(lane < 8 ? scratch[lane] : -INFINITY);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float sum = 0.f;
  for (int i = tid; i < p.n; i += 256) sum += exp(s[i] * p.scale - mx);
  sum = simd_sum(sum);
  if (lane == 0) scratch[sgid] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  sum = simd_sum(lane < 8 ? scratch[lane] : 0.f);
  const float inv = 1.f / sum;
  device bf16* o = P + (size_t)row * p.ld;
  for (int i = tid; i < p.n; i += 256) o[i] = bf16(exp(s[i] * p.scale - mx) * inv);
}

// Final 3x3 conv (Cin -> 4, RGBA) + clamp(-1, 1) + [0, 255] quantization. One thread per pixel.
struct OutParams {
  int H, W, Cin;
};
kernel void vae_conv_out_rgba(device const bf16* in [[buffer(0)]],
                              device const bf16* Wt [[buffer(1)]],  // [9*Cin, 4]
                              device const float* bias [[buffer(2)]],
                              device uchar4* out [[buffer(3)]],
                              constant OutParams& p [[buffer(4)]],
                              uint gid [[thread_position_in_grid]]) {
  const int pix = gid;
  if (pix >= p.H * p.W) return;
  const int y = pix / p.W, x = pix % p.W;
  float4 acc = float4(bias[0], bias[1], bias[2], bias[3]);
  for (int tap = 0; tap < 9; tap++) {
    const int sy = y + tap / 3 - 1, sx = x + tap % 3 - 1;
    if (sy < 0 || sy >= p.H || sx < 0 || sx >= p.W) continue;
    device const bf16* ip = in + ((size_t)sy * p.W + sx) * p.Cin;
    device const vec<bf16, 4>* wp = (device const vec<bf16, 4>*)(Wt + (size_t)tap * p.Cin * 4);
    for (int c = 0; c < p.Cin; c += 8) {
      const vec<bf16, 8> v = *(device const vec<bf16, 8>*)(ip + c);
      for (int k = 0; k < 8; k++) acc += float(v[k]) * float4(wp[c + k]);
    }
  }
  acc = clamp(acc, -1.f, 1.f);
  const float4 u = rint(clamp(acc * 0.5f + 0.5f, 0.f, 1.f) * 255.f);
  out[pix] = uchar4(u);
}

// Touch a scratch buffer (VAEDecoder::prewire): makes the driver map it ahead of its first use.
kernel void vae_touch(device uint* p [[buffer(0)]], uint gid [[thread_position_in_grid]]) { p[gid] = 0; }
