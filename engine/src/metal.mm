#include "metal.h"

#include <mach/mach_time.h>

#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace krea {

// Parameter blocks: layouts mirror the structs in kernels/*.metal.
struct GemmParams { int M, N, K, lda, ldw, ldc, seg, seg_stride, K2, lda2, ldw2; };
struct F32GemmParams { int M, N, K, lda, ldw, ldc, mode; };
struct RmsParams { int width, ld_in, ld_out; float eps; int has_add; };
struct QKNormParams { int rows, ld, q_off, n_q, k_off, n_k; float eps; int rope_mode; int big_dim; };
struct AttnParams { int Lq, Lk, ldq, ldk, ldv, ldo, group, causal, q_pos0; float scale_log2; int ldg, big, ldqb, ldkb; };
struct LayerAttnParams { int rows, ld, ldo, heads, q_off, k_off, v_off, g_off; float scale; };
struct CopyParams { int rows, cols, ld_src, ld_dst; };

static constexpr int kGemmBM = 64, kGemmBN = 64;
static constexpr int kAttnWQ = 4;  // must match WQ in kernels/attention.metal

Metal::Metal(const std::string& metallib_path) {
  // MTLCreateSystemDefaultDevice() can return nil in non-GUI sessions; enumerate instead.
  NSArray<id<MTLDevice>>* devs = MTLCopyAllDevices();
  dev = devs.count ? devs[0] : MTLCreateSystemDefaultDevice();
  if (!dev) throw std::runtime_error("no Metal device");
  queue = [dev newCommandQueue];
  profile_ = getenv("KREA_PROFILE") != nullptr;
  NSError* err = nil;
  lib_ = [dev newLibraryWithURL:[NSURL fileURLWithPath:@(metallib_path.c_str())] error:&err];
  if (!lib_)
    throw std::runtime_error("cannot load metallib " + metallib_path + ": " +
                             (err ? err.localizedDescription.UTF8String : "?"));
}

Tensor Metal::alloc(size_t bytes) {
  bytes = std::max<size_t>(bytes, 16);
  id<MTLBuffer> b = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  if (!b) throw std::runtime_error("allocation failed: " + std::to_string(bytes) + " bytes");
  return Tensor{b, 0, bytes};
}

id<MTLComputePipelineState> Metal::pso(const std::string& name) {
  auto it = psos_.find(name);
  if (it != psos_.end()) return it->second;
  id<MTLFunction> fn = [lib_ newFunctionWithName:@(name.c_str())];
  if (!fn) throw std::runtime_error("kernel not found: " + name);
  NSError* err = nil;
  id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:fn error:&err];
  if (!p) throw std::runtime_error("pipeline failed: " + name);
  psos_[name] = p;
  return p;
}

void Metal::begin() {
  cb_ = [queue commandBuffer];
  enc_ = [cb_ computeCommandEncoder];
}

double Metal::host_now() {
  static mach_timebase_info_data_t tb = [] {
    mach_timebase_info_data_t t;
    mach_timebase_info(&t);
    return t;
  }();
  return (double)mach_absolute_time() * tb.numer / tb.denom / 1e9;
}

void Metal::record_host(const std::string& label, double start, double end) {
  std::lock_guard<std::mutex> g(prof_mu_);
  prof_.push_back({label, start, end, false});
}

std::vector<Metal::ProfEvent> Metal::take_profile() {
  std::lock_guard<std::mutex> g(prof_mu_);
  std::vector<ProfEvent> out;
  out.swap(prof_);
  return out;
}

void Metal::commit() {
  if (!cb_) return;
  if (profile_) {
    NSString* label = cb_.label ?: @"other";
    [cb_ addCompletedHandler:^(id<MTLCommandBuffer> b) {
      std::lock_guard<std::mutex> g(prof_mu_);
      prof_.push_back({label.UTF8String, b.GPUStartTime, b.GPUEndTime, true});
    }];
  }
  [enc_ endEncoding];
  [cb_ commit];
  last_cb_ = cb_;
  cb_ = nil;
  enc_ = nil;
}

void Metal::wait() {
  if (!last_cb_) return;
  [last_cb_ waitUntilCompleted];
  if (last_cb_.status == MTLCommandBufferStatusError) {
    NSError* e = last_cb_.error;
    throw std::runtime_error(std::string("GPU error: ") + (e ? e.localizedDescription.UTF8String : "?"));
  }
  last_gpu_ms_ = (last_cb_.GPUEndTime - last_cb_.GPUStartTime) * 1e3;
  last_cb_ = nil;
}

void Metal::signal_event(id<MTLSharedEvent> ev, uint64_t value) {
  [enc_ endEncoding];
  [cb_ encodeSignalEvent:ev value:value];
  enc_ = [cb_ computeCommandEncoder];
}

void Metal::wait_event(id<MTLSharedEvent> ev, uint64_t value) {
  [enc_ endEncoding];
  [cb_ encodeWaitForEvent:ev value:value];
  enc_ = [cb_ computeCommandEncoder];
}

void Metal::bind(std::initializer_list<Tensor> bufs, const void* params, size_t psize, int param_index) {
  int i = 0;
  for (const Tensor& t : bufs) {
    [enc_ setBuffer:t.buf offset:(t.buf ? t.off : 0) atIndex:i];
    i++;
  }
  if (params) [enc_ setBytes:params length:psize atIndex:(param_index >= 0 ? param_index : i)];
}

void Metal::dispatch(const std::string& kernel, MTLSize grid_tg, MTLSize tg, std::initializer_list<Tensor> bufs,
                     const void* params, size_t psize, int param_index) {
  [enc_ setComputePipelineState:pso(kernel)];
  bind(bufs, params, psize, param_index);
  [enc_ dispatchThreadgroups:grid_tg threadsPerThreadgroup:tg];
}

void Metal::dispatch_threads(const std::string& kernel, MTLSize threads, MTLSize tg,
                             std::initializer_list<Tensor> bufs, const void* params, size_t psize,
                             int param_index) {
  [enc_ setComputePipelineState:pso(kernel)];
  bind(bufs, params, psize, param_index);
  [enc_ dispatchThreads:threads threadsPerThreadgroup:tg];
}

// ------------------------------------------------------------------------------------------
void Metal::gemm(Epi epi, const Tensor& A, const Weight& W, const Tensor& C, int M, int N, int K, int lda, int ldw,
                 int ldc, const Tensor& vec, const LoraTerm* lr) {
  static const char* names[] = {"gemm_bf16",   "gemm_f32",  "gemm_resid_gate", "gemm_resid",
                                "gemm_swiglu", "gemm_gelu", "gemm_bias_bf16",  "gemm_bias_f32"};
  if (N % kGemmBN || K % 16 || M <= 0) throw std::runtime_error("gemm: unsupported shape");
  if (W.q8 && K % 128) throw std::runtime_error("gemm: int8 weights need K % 128 == 0");
  if (lr && (W.q8 || lr->K % 16)) throw std::runtime_error("gemm: bad LoRA term");
  GemmParams p{M, N, K, lda, ldw, ldc, 0, 0, lr ? lr->K : 0, lr ? lr->lda : 0, lr ? lr->ldw : 0};
  MTLSize grid = MTLSizeMake(N / kGemmBN, (M + kGemmBM - 1) / kGemmBM, 1);
  // buffers 4 and 5 must be bound even when unused
  const Tensor v = vec.buf ? vec : C;
  const Tensor sc = W.q8 ? W.s : C;
  std::string name = names[(int)epi];
  if (W.q8) name += "_q8";
  if (lr) name += "_lr";
  dispatch(name, grid, MTLSizeMake(128, 1, 1), {A, W.w, C, Tensor(), v, sc, lr ? lr->A : C, lr ? lr->W : C}, &p,
           sizeof(p), 3);
}

void Metal::gemm_f32(const Tensor& A, const Tensor& W, const Tensor& C, int M, int N, int K, int lda, int ldw, int ldc,
                     int mode, const Tensor& bias) {
  if (N % 64 || K % 16 || M <= 0) throw std::runtime_error("gemm_f32: unsupported shape");
  if ((mode & kF32Bias) && !bias) throw std::runtime_error("gemm_f32: missing bias");
  F32GemmParams p{M, N, K, lda, ldw, ldc, mode};
  dispatch("gemm_f32w", MTLSizeMake(N / 64, (M + 63) / 64, 1), MTLSizeMake(128, 1, 1), {A, W, C, Tensor(), bias ? bias : C},
           &p, sizeof(p), 3);
}

void Metal::rms_norm(const Tensor& x, const Tensor& mul, const Tensor& add, const Tensor& y, int rows, int width,
                     int ld_in, int ld_out, float eps, bool f32_out) {
  if (width % 4 || ld_in % 4 || ld_out % 4) throw std::runtime_error("rms_norm: bad shape");
  RmsParams p{width, ld_in, ld_out, eps, add ? 1 : 0};
  dispatch(f32_out ? "rms_norm_f32" : "rms_norm_bf16", MTLSizeMake(rows, 1, 1), MTLSizeMake(256, 1, 1),
           {x, mul, add ? add : mul, y}, &p, sizeof(p), 4);
}

void Metal::qk_norm_rope(const Tensor& qkv, const Tensor& wq, const Tensor& wk, const Tensor& cos, const Tensor& sin,
                         int rows, int ld, int q_off, int n_q, int k_off, int n_k, float eps, RopeMode mode,
                         int big_dim, const Tensor& qbig, const Tensor& kbig) {
  if (big_dim >= 0 && (!qbig || !kbig)) throw std::runtime_error("qk_norm_rope: big_dim needs qbig / kbig");
  QKNormParams p{rows, ld, q_off, n_q, k_off, n_k, eps, (int)mode, big_dim};
  const size_t threads = (size_t)rows * (n_q + n_k) * 32;
  const Tensor c = cos ? cos : wq, s = sin ? sin : wq;
  dispatch_threads("qk_norm_rope", MTLSizeMake(threads, 1, 1), MTLSizeMake(256, 1, 1),
                   {qkv, wq, wk, c, s, Tensor(), qbig ? qbig : wq, kbig ? kbig : wq}, &p, sizeof(p), 5);
}

void Metal::attention(const Tensor& Q, const Tensor& K, const Tensor& V, const Tensor& O, int Lq, int Lk, int heads,
                      int ldq, int ldk, int ldv, int ldo, int group, bool causal, int q_pos0, const Tensor& G, int ldg,
                      const Tensor& QB, const Tensor& KB) {
  const bool big = QB && KB;
  AttnParams p{Lq, Lk, ldq, ldk, ldv, ldo, group, causal ? 1 : 0, q_pos0,
               (float)(1.0 / std::sqrt(128.0) * 1.4426950408889634), G ? ldg : 0, big ? 1 : 0, heads, heads / group};
  const int bq = 8 * kAttnWQ;
  dispatch("attention_d128", MTLSizeMake((Lq + bq - 1) / bq, heads, 1), MTLSizeMake(32 * kAttnWQ, 1, 1),
           {Q, K, V, O, Tensor(), Tensor(), Tensor(), Tensor(), G ? G : Q, big ? QB : Q, big ? KB : Q}, &p, sizeof(p), 4);
}

void Metal::layerwise_attn(const Tensor& qkvg, const Tensor& out, int rows, int ld, int ldo, int heads, int q_off,
                           int k_off, int v_off, int g_off) {
  LayerAttnParams p{rows, ld, ldo, heads, q_off, k_off, v_off, g_off, (float)(1.0 / std::sqrt(128.0))};
  dispatch_threads("layerwise_attn", MTLSizeMake((size_t)rows * heads * 32, 1, 1), MTLSizeMake(256, 1, 1), {qkvg, out},
                   &p, sizeof(p), 2);
}

void Metal::tap_project(const Tensor& x, const Tensor& w, const Tensor& y, int T, int width) {
  const int p[2] = {T, width};
  dispatch_threads("tap_project", MTLSizeMake(width / 4, T, 1), MTLSizeMake(64, 4, 1), {x, w, y}, p, sizeof(p), 3);
}

void Metal::copy_rows(const Tensor& src, const Tensor& dst, int rows, int cols, int ld_src, int ld_dst) {
  CopyParams p{rows, cols, ld_src, ld_dst};
  dispatch_threads("copy_rows", MTLSizeMake(cols / 8, rows, 1), MTLSizeMake(64, 4, 1), {src, dst}, &p, sizeof(p), 2);
}

void Metal::copy_rows_f32(const Tensor& src, const Tensor& dst, int rows, int cols, int ld_src, int ld_dst) {
  CopyParams p{rows, cols, ld_src, ld_dst};
  dispatch_threads("copy_rows_f32", MTLSizeMake(cols / 4, rows, 1), MTLSizeMake(64, 4, 1), {src, dst}, &p, sizeof(p), 2);
}

void Metal::flow_step(const Tensor& z, const Tensor& v, const Tensor& x0_prev, int n, float sigma, float cz, float c0,
                      float c1) {
  struct { int n; float sigma, cz, c0, c1; } p{n, sigma, cz, c0, c1};
  dispatch_threads("flow_step", MTLSizeMake(n, 1, 1), MTLSizeMake(256, 1, 1), {z, v, x0_prev}, &p, sizeof(p), 3);
}

void Metal::euler_step(const Tensor& z, const Tensor& v, int n, float dt) {
  const float p[2] = {(float)n, dt};
  dispatch_threads("euler_step", MTLSizeMake(n, 1, 1), MTLSizeMake(256, 1, 1), {z, v}, p, sizeof(p), 2);
}

void Metal::f32_copy(const Tensor& x, const Tensor& y, size_t n) {
  dispatch_threads("f32_copy", MTLSizeMake(n / 4, 1, 1), MTLSizeMake(256, 1, 1), {x, y});
}

void Metal::f32_axpy(const Tensor& x, const Tensor& y, float a, size_t n) {
  dispatch_threads("f32_axpy", MTLSizeMake(n / 4, 1, 1), MTLSizeMake(256, 1, 1), {x, y}, &a, sizeof(a), 2);
}

void Metal::f32_to_bf16(const Tensor& x, const Tensor& y, int n) {
  dispatch_threads("f32_to_bf16", MTLSizeMake(n, 1, 1), MTLSizeMake(256, 1, 1), {x, y});
}

void Metal::rms_stats(const Tensor& x, const Tensor& st, int rows, int width, float eps) {
  struct { int rows, width; float eps; } p{rows, width, eps};
  dispatch("rms_row_stats", MTLSizeMake((rows + 7) / 8, 1, 1), MTLSizeMake(256, 1, 1), {x, st}, &p, sizeof(p), 2);
}

void Metal::rms_dual(const Tensor& x, const Tensor& st, const Tensor& mul, const Tensor& add, const Tensor& y,
                     const Tensor& yt, int rows, int rows_pad, int width, int ld_y, int ld_t, float t_scale) {
  if (rows_pad % 32 || ld_t % 32 || rows > rows_pad || width % 64) throw std::runtime_error("rms_dual: bad shape");
  struct { int rows, width, ld_y, ld_t, has_add; float t_scale; } p{rows, width, ld_y, ld_t, add ? 1 : 0, t_scale};
  dispatch("rms_apply_dual", MTLSizeMake(width / 64, rows_pad / 32, 1), MTLSizeMake(256, 1, 1),
           {x, st, mul, add ? add : mul, y, yt}, &p, sizeof(p), 6);
}

void Metal::to_ane_bf16(const Tensor& in, const Tensor& out, int rows, int rows_pad, int width, int ld_in, int ld_t,
                        float scale) {
  if (rows_pad % 32 || ld_t % 32 || rows > rows_pad || width % 64) throw std::runtime_error("to_ane_bf16: bad shape");
  struct { int rows, width, ld_in, ld_t; float scale; } p{rows, width, ld_in, ld_t, scale};
  dispatch("to_ane_bf16", MTLSizeMake(width / 64, rows_pad / 32, 1), MTLSizeMake(256, 1, 1), {in, out}, &p, sizeof(p), 2);
}

void Metal::resid_gate_add_t16(const Tensor& x, const Tensor& y, const Tensor& gate, int rows, int width, int ld_y,
                               float scale, const Tensor& y2) {
  if (ld_y % 32 || rows > ld_y || width % 64) throw std::runtime_error("resid_gate_add_t16: bad shape");
  struct { int rows, width, ld_in; float scale; } p{rows, width, ld_y, scale};
  dispatch(y2 ? "resid_gate_add_t16x2" : "resid_gate_add_t16", MTLSizeMake(width / 64, (rows + 31) / 32, 1),
           MTLSizeMake(256, 1, 1), {x, y, gate, Tensor(), y2 ? y2 : y}, &p, sizeof(p), 3);
}

void Metal::ane_cols_scatter(const Tensor& in, const Tensor& out, int rows, int n, int ld_in, int ld_out, int col0) {
  if (n % 64 || ld_in % 32 || rows > ld_in) throw std::runtime_error("ane_cols_scatter: bad shape");
  struct { int rows, ld_in, ld_out, col0; } p{rows, ld_in, ld_out, col0};
  dispatch("ane_cols_scatter", MTLSizeMake(n / 64, (rows + 31) / 32, 1), MTLSizeMake(256, 1, 1), {in, out}, &p,
           sizeof(p), 2);
}

}  // namespace krea
