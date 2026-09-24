// Flash attention forward, head_dim = 128, bf16 in/out, fp32 accumulation and softmax.
//
// Q: [Lq rows, row stride ldq] per head at column h*128; K/V: [Lk rows, stride ldk/ldv] at
// column kvh*128 (GQA: kvh = h / group). One threadgroup = WQ simdgroups x 8 query rows of one
// head. Dispatch: (ceil(Lq / (8*WQ)), heads) threadgroups of 32*WQ threads (WQ = 4 by default,
// i.e. 32 query rows / 128 threads per threadgroup).
//
// Used for: the Krea DiT (non-causal, GQA 48/12, [text ; image] rows), the text-fusion refiner blocks
// (non-causal, 20 heads) and the Qwen3 text encoder (causal after a cached system prefix, GQA 32/8).
//
// Output gate (Krea): when p.ldg > 0 the normalized output is multiplied by sigmoid(G[row, h*128 + d])
// before the bf16 store (G: bf16, row stride ldg), i.e. attn(q, k, v) * sigmoid(gate) fused into the store.
//
// Split-K (attention_d128_first / _mid / _last, non-causal only): the same loop over one key block
// (K/V pointers and Lk describe the block), with the loop state carried between launches in fp32:
// So [Lq][heads*128] unnormalized O, Sm [Lq][heads] running max (log2 units) and Sl [Lq][heads][8]
// the per-lane partial row sums. _first starts from the empty state and stores it, _mid loads and
// stores, _last loads and writes the normalized bf16 O like attention_d128. When every block but the
// last is a multiple of BK keys, the result is bit-identical to one attention_d128 over all keys.
//
// Design (engine/opt/attn/RESULTS.md has the measurements behind each choice):
//  * Every fragment lives in registers as the lane's vec<T,2>; simdgroup_matrix objects exist only
//    around each MMA (arrays of simdgroup_matrix were the main cost of the previous kernel).
//  * S^T = K Q^T instead of S = Q K^T, so K is a row-major A operand and nothing is transposed
//    when staging: K/V blocks are plain 16-byte copies into threadgroup memory (unpadded rows,
//    16-byte chunks XOR-swizzled by row to avoid bank conflicts).
//  * The head dimension is permuted inside each 32-column group (fragment j, fragment column c ->
//    d = (j/4)*32 + (c/2)*8 + (j%4)*2 + c%2) so one 16-byte read gives a lane its elements of four
//    consecutive fragments, for K, for V and for the output store. The contraction over d does not
//    care about the order, and O uses the same permutation, undone by the store.
//  * P^T (keys x queries, from S^T) becomes the A operand P through a 256-byte per-simdgroup
//    threadgroup scratch (two 16-bit stores + one 32-bit load per fragment).
//  * Lazy online softmax: the running max only moves when a score exceeds it by more than LAZY_TH
//    (log2 units; P <= 2^LAZY_TH, far inside fp32/bf16 range), so the steady state has no
//    shuffles and no O rescale; row sums are per-lane partials reduced once at the end.
//    LAZY_TH=0 gives the classic exact-max update (P_max == 1 exactly) at ~1-2% more time.
//  * K/V staging: the next block is fetched into registers during the current block and stored
//    into the other half of a double buffer, one threadgroup barrier per 16-key block.
#include "common.h"

#ifndef WQ
#define WQ 4       // simdgroups per threadgroup (8 query rows each); host: kAttnWQ must match
#endif
#ifndef BK
#define BK 16      // keys per block
#endif
#ifndef LAZY_TH
#define LAZY_TH 8.0f
#endif
#define UNROLL _Pragma("clang loop unroll(full)")

#define D 128
#define BQ (8 * WQ)
#define NT (32 * WQ)
#define DF 16                       // 8-column fragments along the head dim
#define KF (BK / 8)                 // key fragments per block
#define BLK (BK * D)                // one staged K or V block (unpadded, swizzled)
#define CH ((BK * 16 + NT - 1) / NT)  // 16-byte chunks per thread per K (or V) block
#define NEG_BIG (-3.0e38f)          // masked score
#define M_INIT (-1.5e38f)           // initial running max: below any real score, above masked ones

typedef vec<bf16, 2> b2;
typedef vec<bf16, 8> b8;

struct AttnParams {
  int Lq, Lk;
  int ldq, ldk, ldv, ldo;
  int group;        // q heads per kv head
  int causal;       // 1: key j visible to query i iff j <= i + q_pos0
  int q_pos0;       // absolute position of query row 0 (causal offset)
  float scale_log2; // softmax scale * log2(e)
  int ldg;          // > 0: output gate G (bf16, row stride ldg), O *= sigmoid(G) at the store
  int big;          // 1: add the fp32 rank-1 term QB[q, h] * KB[k, h / group] to the scores (see qk_norm_rope)
  int ldqb, ldkb;   // row strides of QB (query heads) and KB (kv heads)
};

METAL_FUNC void mma(thread float2& c, thread const b2& a, thread const b2& b) {
  simdgroup_matrix<bf16, 8, 8> A, B;
  simdgroup_matrix<float, 8, 8> C, Dm;
  reinterpret_cast<thread b2&>(A.thread_elements()) = a;
  reinterpret_cast<thread b2&>(B.thread_elements()) = b;
  reinterpret_cast<thread float2&>(C.thread_elements()) = c;
  simdgroup_multiply_accumulate(Dm, A, B, C);
  c = reinterpret_cast<thread float2&>(Dm.thread_elements());
}

// kArow[lane] = a lane whose S^T columns (queries fn, fn+1) include query row fm of `lane`, i.e.
// the inverse of frag_coord at (row 0, column pair fm & 6). Tabulated on purpose: the closed-form
// bit arithmetic for such lane maps gets folded into llvm.bitreverse.i3, which this Metal compiler
// (Xcode 15, metal 32023.155) miscompiles.
constant ushort kArow[32] = {0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1,
                             8, 8, 8, 8, 9, 9, 9, 9, 8, 8, 8, 8, 9, 9, 9, 9};

// Load this thread's 16-byte chunks of K and V block rows [kb, kb+BK) (rows tid/16 + i*NT/16,
// columns (tid%16)*8 ..). kp/vp point at the thread's first row of the block; rows past Lk are
// clamped (only in the last block).
METAL_FUNC void fetch(thread uint4* rk, thread uint4* rv, device const bf16* kp, device const bf16* vp,
                      int ldk, int ldv, int kb, int Lk, ushort tid) {
  if (kb + BK <= Lk) {
    UNROLL for (int i = 0; i < CH; i++) {
      if (BK * 16 % NT != 0 && i * NT + tid >= BK * 16) break;
      rk[i] = *(device const uint4*)(kp + (size_t)(i * (NT / 16)) * ldk);
      rv[i] = *(device const uint4*)(vp + (size_t)(i * (NT / 16)) * ldv);
    }
  } else {
    const int r0 = tid >> 4;
    UNROLL for (int i = 0; i < CH; i++) {
      if (BK * 16 % NT != 0 && i * NT + tid >= BK * 16) break;
      const int dr = min(kb + r0 + i * (NT / 16), Lk - 1) - (kb + r0);
      rk[i] = *(device const uint4*)(kp + (long)dr * ldk);
      rv[i] = *(device const uint4*)(vp + (long)dr * ldv);
    }
  }
}

// Store the fetched chunks: row r, chunk c (16 bytes) goes to chunk slot c ^ (r & 7).
METAL_FUNC void put(threadgroup bf16* Kd, threadgroup bf16* Vd, thread const uint4* rk, thread const uint4* rv,
                    ushort tid) {
  UNROLL for (int i = 0; i < CH; i++) {
    const int c = i * NT + tid;
    if (BK * 16 % NT != 0 && c >= BK * 16) break;
    const int r = c >> 4, cc = ((c & 15) ^ (r & 7)) * 8;
    *(threadgroup uint4*)(Kd + r * D + cc) = rk[i];
    *(threadgroup uint4*)(Vd + r * D + cc) = rv[i];
  }
}

// The kernel body. LOAD: start from the split-K state (So, Sm, Sl) instead of the empty state;
// STORE: store the state instead of the normalized output. KVs/Pscr: the kernel's threadgroup
// buffers (2 * 2 * BLK and WQ * KF * 64 bf16).
template <bool LOAD, bool STORE>
METAL_FUNC void attn_body(device const bf16* Q, device const bf16* K, device const bf16* V, device bf16* O,
                          device const bf16* G, device const float* QB, device const float* KB,
                          device float* So, device float* Sm, device float* Sl, constant AttnParams& p,
                          uint2 tg, uint heads, ushort sg, ushort lane, ushort tid, threadgroup bf16* KVs,
                          threadgroup bf16* Pscr) {
  const int h = tg.y;
  const int kvh = h / p.group;
  const int qblk = tg.x * BQ;
  const int q0 = qblk + sg * 8;           // first query row of this simdgroup
  const short2 fc = frag_coord(lane);
  const ushort sn = fc.x, sm = fc.y;      // lane's fragment column pair / row

  // Q^T fragments (B operand of S^T = K Q^T): lane holds Q[q0 + sn + e][d(j, sm)]; rows past Lq
  // are clamped (their outputs are not stored)
  b2 qt[DF];
  {
    const int r0 = min(q0 + sn, p.Lq - 1), r1 = min(q0 + sn + 1, p.Lq - 1);
    device const bf16* qa = Q + (size_t)r0 * p.ldq + h * D + (sm >> 1) * 8 + (sm & 1);
    device const bf16* qb = Q + (size_t)r1 * p.ldq + h * D + (sm >> 1) * 8 + (sm & 1);
    UNROLL for (int j = 0; j < DF; j++) {
      const int off = (j >> 2) * 32 + (j & 3) * 2;
      qt[j] = b2(qa[off], qb[off]);
    }
  }

  // fp32 rank-1 score term (big head dimension): this lane's two query columns
  float2 qb = float2(0.f);
  if (p.big) qb = float2(QB[(size_t)min(q0 + sn, p.Lq - 1) * p.ldqb + h], QB[(size_t)min(q0 + sn + 1, p.Lq - 1) * p.ldqb + h]);

  float2 of[DF];                          // O: lane holds (query q0 + sm, columns d(j, sn + e))
  float2 m_i = float2(M_INIT);            // running max (log2 units) of queries q0 + sn + e
  float2 l_i = float2(0.f);               // this lane's partial row sums for the same queries
  if (LOAD) {
    // clamped rows (past Lq) resume from the state of row Lq - 1, like their clamped Q rows
    device const float* so = So + (size_t)min(q0 + sm, p.Lq - 1) * heads * D + h * D + (sn >> 1) * 8;
    UNROLL for (int a = 0; a < 4; a++) {
      const float4 u0 = *(device const float4*)(so + a * 32), u1 = *(device const float4*)(so + a * 32 + 4);
      of[4 * a] = u0.xy;
      of[4 * a + 1] = u0.zw;
      of[4 * a + 2] = u1.xy;
      of[4 * a + 3] = u1.zw;
    }
    const int ra = min(q0 + sn, p.Lq - 1), rb = min(q0 + sn + 1, p.Lq - 1);
    m_i = float2(Sm[ra * heads + h], Sm[rb * heads + h]);
    l_i = float2(Sl[(ra * heads + h) * 8 + sm], Sl[(rb * heads + h) * 8 + sm]);
  } else {
    UNROLL for (int j = 0; j < DF; j++) of[j] = float2(0.f);
  }

  // key range for this threadgroup (causal: up to the last query of the block)
  int kend = p.Lk;
  if (p.causal) kend = min(p.Lk, qblk + BQ + p.q_pos0);

  device const bf16* kp = K + kvh * D + (size_t)(tid >> 4) * p.ldk + (tid & 15) * 8;
  device const bf16* vp = V + kvh * D + (size_t)(tid >> 4) * p.ldv + (tid & 15) * 8;
  const int kstep = BK * p.ldk, vstep = BK * p.ldv;
  const ushort arow_src = kArow[lane];
  const int frag_off = sm * D;
  // this lane's 16-byte read of block row n*8 + sm, 32-column group a (swizzled chunk)
#define FOFF(n, a) (frag_off + (n) * 8 * D + ((((a) * 4 + (sn >> 1)) ^ (sm & 7)) * 8))
  threadgroup bf16* ps_w = Pscr + sg * (KF * 64) + sn * 8 + sm;
  threadgroup const bf16* ps_r = Pscr + sg * (KF * 64) + sm * 8 + sn;

  // prologue: block 0 into buffer 0, block 1 into registers
  uint4 rk[CH], rv[CH];
  fetch(rk, rv, kp, vp, p.ldk, p.ldv, 0, p.Lk, tid);
  kp += kstep; vp += vstep;
  put(KVs, KVs + BLK, rk, rv, tid);
  if (BK < kend) {
    fetch(rk, rv, kp, vp, p.ldk, p.ldv, BK, p.Lk, tid);
    kp += kstep; vp += vstep;
  }

  int cur = 0;
  for (int kb = 0; kb < kend; kb += BK) {
    // one barrier per block: block kb is complete in buffer `cur`, and everyone is done reading
    // the other buffer (block kb - BK), which now receives block kb + BK from registers
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup const bf16* Ks = KVs + cur * 2 * BLK;
    threadgroup const bf16* Vs = Ks + BLK;
    if (kb + BK < kend) {
      threadgroup bf16* nk = KVs + (cur ^ 1) * 2 * BLK;
      put(nk, nk + BLK, rk, rv, tid);
      if (kb + 2 * BK < kend) {
        fetch(rk, rv, kp, vp, p.ldk, p.ldv, kb + 2 * BK, p.Lk, tid);
        kp += kstep; vp += vstep;
      }
    }
    cur ^= 1;

    // S^T = K Q^T: st[n] lane holds (key kb + n*8 + sm, queries q0 + sn + e), unscaled
    float2 st[KF];
    UNROLL for (int n = 0; n < KF; n++) st[n] = float2(0.f);
    UNROLL for (int a = 0; a < 4; a++) {
      b8 k8[KF];
      UNROLL for (int n = 0; n < KF; n++) k8[n] = *(threadgroup const b8*)(Ks + FOFF(n, a));
      UNROLL for (int b = 0; b < 4; b++)
        UNROLL for (int n = 0; n < KF; n++) mma(st[n], b2(k8[n][2 * b], k8[n][2 * b + 1]), qt[4 * a + b]);
    }

    if (p.big) {
      UNROLL for (int n = 0; n < KF; n++) st[n] += KB[(size_t)min(kb + n * 8 + sm, p.Lk - 1) * p.ldkb + kvh] * qb;
    }

    // lazy online softmax over keys (column-wise in S^T; a query's keys live in lanes l^2, l^4, l^16)
    const bool need_mask = (kb + BK > p.Lk) || p.causal;
    float2 lmax = float2(NEG_BIG);
    UNROLL for (int n = 0; n < KF; n++) {
      float2 v = st[n];
      if (need_mask) {
        const int key = kb + n * 8 + sm;
        const int row = q0 + sn;
        v.x = (key < p.Lk && (!p.causal || key <= row + p.q_pos0)) ? v.x : NEG_BIG;
        v.y = (key < p.Lk && (!p.causal || key <= row + 1 + p.q_pos0)) ? v.y : NEG_BIG;
      }
      st[n] = v;
      lmax = max(lmax, v);
    }
    lmax *= p.scale_log2;
    if (simd_any(lmax.x > m_i.x + LAZY_TH || lmax.y > m_i.y + LAZY_TH)) {
      float2 bmax = lmax;
      bmax = max(bmax, simd_shuffle_xor(bmax, 2));
      bmax = max(bmax, simd_shuffle_xor(bmax, 4));
      bmax = max(bmax, simd_shuffle_xor(bmax, 16));
      float2 m_new;
      m_new.x = bmax.x > m_i.x + LAZY_TH ? bmax.x : m_i.x;
      m_new.y = bmax.y > m_i.y + LAZY_TH ? bmax.y : m_i.y;
      const float2 alpha = fast::exp2(m_i - m_new);
      m_i = m_new;
      l_i *= alpha;
      const float2 ar = simd_shuffle(alpha, arow_src);  // alpha of this lane's O row (query q0 + sm)
      const float a = (sm & 1) ? ar.y : ar.x;
      UNROLL for (int j = 0; j < DF; j++) of[j] *= a;
    }

    // P^T -> scratch as [query][key] -> P fragments (A operand)
    b2 pf[KF];
    UNROLL for (int n = 0; n < KF; n++) {
      const float2 pv = fast::exp2(fma(st[n], float2(p.scale_log2), -m_i));
      l_i += pv;
      ps_w[n * 64] = bf16(pv.x);
      ps_w[n * 64 + 8] = bf16(pv.y);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    UNROLL for (int n = 0; n < KF; n++) pf[n] = *(threadgroup const b2*)(ps_r + n * 64);

    // O += P V
    UNROLL for (int a = 0; a < 4; a++) {
      b8 v8[KF];
      UNROLL for (int n = 0; n < KF; n++) v8[n] = *(threadgroup const b8*)(Vs + FOFF(n, a));
      UNROLL for (int b = 0; b < 4; b++)
        UNROLL for (int n = 0; n < KF; n++) mma(of[4 * a + b], pf[n], b2(v8[n][2 * b], v8[n][2 * b + 1]));
    }
  }
#undef FOFF

  if (STORE) {  // unreduced state: O rows q0 + sm, (m, partial l) of queries q0 + sn + e
    if (q0 + sm < p.Lq) {
      device float* so = So + (size_t)(q0 + sm) * heads * D + h * D + (sn >> 1) * 8;
      UNROLL for (int a = 0; a < 4; a++) {
        *(device float4*)(so + a * 32) = float4(of[4 * a], of[4 * a + 1]);
        *(device float4*)(so + a * 32 + 4) = float4(of[4 * a + 2], of[4 * a + 3]);
      }
    }
    const int ra = q0 + sn, rb = q0 + sn + 1;
    if (ra < p.Lq) {
      if (sm == 0) Sm[ra * heads + h] = m_i.x;
      Sl[(ra * heads + h) * 8 + sm] = l_i.x;
    }
    if (rb < p.Lq) {
      if (sm == 0) Sm[rb * heads + h] = m_i.y;
      Sl[(rb * heads + h) * 8 + sm] = l_i.y;
    }
    return;
  }

  // full row sums (8 lanes per query column), fetched for this lane's O row
  l_i += simd_shuffle_xor(l_i, 2);
  l_i += simd_shuffle_xor(l_i, 4);
  l_i += simd_shuffle_xor(l_i, 16);
  const float2 lr = simd_shuffle(l_i, arow_src);
  const float l = (sm & 1) ? lr.y : lr.x;
  const int row = q0 + sm;
  if (row < p.Lq) {
    const float inv = l > 0.f ? 1.f / l : 0.f;
    device bf16* op = O + (size_t)row * p.ldo + h * D + (sn >> 1) * 8;
    device const bf16* gp = G + (size_t)row * p.ldg + h * D + (sn >> 1) * 8;
    UNROLL for (int a = 0; a < 4; a++) {
      b8 o8;
      float g[8] = {1, 1, 1, 1, 1, 1, 1, 1};
      if (p.ldg > 0) {
        const b8 g8 = *(device const b8*)(gp + a * 32);
        UNROLL for (int e = 0; e < 8; e++) g[e] = sigmoid(float(g8[e]));
      }
      UNROLL for (int b = 0; b < 4; b++) {
        o8[2 * b] = bf16(of[4 * a + b].x * inv * g[2 * b]);
        o8[2 * b + 1] = bf16(of[4 * a + b].y * inv * g[2 * b + 1]);
      }
      *(device b8*)(op + a * 32) = o8;
    }
  }
}

kernel void attention_d128(device const bf16* Q [[buffer(0)]],
                           device const bf16* K [[buffer(1)]],
                           device const bf16* V [[buffer(2)]],
                           device bf16* O [[buffer(3)]],
                           constant AttnParams& p [[buffer(4)]],
                           device const bf16* G [[buffer(8)]],
                           device const float* QB [[buffer(9)]],
                           device const float* KB [[buffer(10)]],
                           uint2 tg [[threadgroup_position_in_grid]],
                           uint2 ntg [[threadgroups_per_grid]],
                           ushort sg [[simdgroup_index_in_threadgroup]],
                           ushort lane [[thread_index_in_simdgroup]],
                           ushort tid [[thread_index_in_threadgroup]]) {
  threadgroup bf16 KVs[2 * 2 * BLK];      // double buffer of (K block, V block): 16 KB
  threadgroup bf16 Pscr[WQ * KF * 64];    // per simdgroup P fragments, [query][key]
  attn_body<false, false>(Q, K, V, O, G, QB, KB, nullptr, nullptr, nullptr, p, tg, ntg.y, sg, lane, tid, KVs, Pscr);
}

// Split-K entry points (see the header): buffers 5..7 hold the fp32 state So, Sm, Sl.
#define ATTN_PART(NAME, LOAD, STORE)                                                                      \
  kernel void NAME(device const bf16* Q [[buffer(0)]], device const bf16* K [[buffer(1)]],                \
                   device const bf16* V [[buffer(2)]], device bf16* O [[buffer(3)]],                      \
                   constant AttnParams& p [[buffer(4)]], device float* So [[buffer(5)]],                   \
                   device float* Sm [[buffer(6)]], device float* Sl [[buffer(7)]],                         \
                   device const bf16* G [[buffer(8)]], device const float* QB [[buffer(9)]],               \
                   device const float* KB [[buffer(10)]],                                                  \
                   uint2 tg [[threadgroup_position_in_grid]], uint2 ntg [[threadgroups_per_grid]],        \
                   ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]], \
                   ushort tid [[thread_index_in_threadgroup]]) {                                           \
    threadgroup bf16 KVs[2 * 2 * BLK];                                                                    \
    threadgroup bf16 Pscr[WQ * KF * 64];                                                                  \
    attn_body<LOAD, STORE>(Q, K, V, O, G, QB, KB, So, Sm, Sl, p, tg, ntg.y, sg, lane, tid, KVs, Pscr);    \
  }
ATTN_PART(attention_d128_first, false, true)
ATTN_PART(attention_d128_mid, true, true)
ATTN_PART(attention_d128_last, true, false)
