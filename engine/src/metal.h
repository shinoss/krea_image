// Minimal Metal runtime: device, precompiled pipelines, buffers and op encoders.
#pragma once
#import <Metal/Metal.h>

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace krea {

// A byte range inside a Metal buffer.
struct Tensor {
  id<MTLBuffer> buf = nil;
  size_t off = 0;
  size_t bytes = 0;

  Tensor at(size_t byte_off) const { return Tensor{buf, off + byte_off, bytes - byte_off}; }
  template <class T>
  T* ptr() const { return (T*)((char*)buf.contents + off); }
  explicit operator bool() const { return buf != nil; }
};

// A GEMM weight: bf16 Wt[K, N], or int8 Wq[K, N] with f32 group scales s[K/128, N].
struct Weight {
  Tensor w, s;
  bool q8 = false;
};

// Second product accumulated by a GEMM before its epilogue: acc += A[M, K] x W[K, N] (bf16).
struct LoraTerm {
  Tensor A, W;
  int K, lda, ldw;
};

// Epilogues supported by the GEMM kernels (must match kernels/gemm.metal).
enum class Epi { BF16, F32, ResidGate, Resid, SwiGLU, GELU, BiasBF16, BiasF32 };

// f32-weight GEMM modes (kernels/f32gemm.metal): bias, GELU(tanh), accumulate
enum F32Mode { kF32Bias = 1, kF32Gelu = 2, kF32Acc = 4 };

// RoPE modes of qk_norm_rope (kernels/norm.metal)
enum RopeMode { kRopePairs = 0, kRopeHalf = 1, kRopeNone = 2 };

class Metal {
 public:
  explicit Metal(const std::string& metallib_path);

  id<MTLDevice> dev = nil;
  id<MTLCommandQueue> queue = nil;

  Tensor alloc(size_t bytes);
  id<MTLComputePipelineState> pso(const std::string& name);

  // Command stream: all ops are encoded into the current compute encoder.
  void begin();
  void commit();       // commit without waiting
  void wait();         // wait for the last committed command buffer
  void end_wait() { commit(); wait(); }
  double last_gpu_ms() const { return last_gpu_ms_; }

  // ---- ops ----
  // acc[M,N] = A[M,K] x Wt[K,N]; see kernels/gemm.metal for epilogues. For SwiGLU the output has N/2
  // columns. `vec` is the gate or bias vector (float) for the epilogues that use one.
  void gemm(Epi epi, const Tensor& A, const Weight& W, const Tensor& C, int M, int N, int K, int lda, int ldw,
            int ldc, const Tensor& vec = Tensor(), const LoraTerm* lr = nullptr);
  void gemm(Epi epi, const Tensor& A, const Tensor& W, const Tensor& C, int M, int N, int K, int lda, int ldw,
            int ldc, const Tensor& vec = Tensor()) {
    gemm(epi, A, Weight{W, Tensor(), false}, C, M, N, K, lda, ldw, ldc, vec);
  }
  // f32 x f32 GEMM (weights stored [K, N]) with F32Mode flags.
  void gemm_f32(const Tensor& A, const Tensor& W, const Tensor& C, int M, int N, int K, int lda, int ldw, int ldc,
                int mode, const Tensor& bias = Tensor());
  // y = x * rrms(x) * mul (+ add), per row of `width` f32 values; bf16 or f32 output.
  void rms_norm(const Tensor& x, const Tensor& mul, const Tensor& add, const Tensor& y, int rows, int width, int ld_in,
                int ld_out, float eps, bool f32_out = false);
  // big_dim >= 0: that head dimension goes to qbig [rows, n_q] / kbig [rows, n_k] in fp32 (zeroed in qkv),
  // for the attention's fp32 rank-1 score term (see kernels/norm.metal).
  void qk_norm_rope(const Tensor& qkv, const Tensor& wq, const Tensor& wk, const Tensor& cos, const Tensor& sin,
                    int rows, int ld, int q_off, int n_q, int k_off, int n_k, float eps, RopeMode mode,
                    int big_dim = -1, const Tensor& qbig = Tensor(), const Tensor& kbig = Tensor());
  // Flash attention (d = 128). group = q heads per kv head; G (optional): output gate columns
  // (bf16, row stride ldg), O = attn * sigmoid(G); QB/KB (optional): fp32 rank-1 score term
  // QB[q, h] * KB[k, h / group] (row strides = heads and heads / group).
  void attention(const Tensor& Q, const Tensor& K, const Tensor& V, const Tensor& O, int Lq, int Lk, int heads,
                 int ldq, int ldk, int ldv, int ldo, int group, bool causal, int q_pos0, const Tensor& G = Tensor(),
                 int ldg = 0, const Tensor& QB = Tensor(), const Tensor& KB = Tensor());
  // Text fusion: per-token attention across the 12 tapped states (rows = tokens * 12).
  void layerwise_attn(const Tensor& qkvg, const Tensor& out, int rows, int ld, int ldo, int heads, int q_off,
                      int k_off, int v_off, int g_off);
  void tap_project(const Tensor& x, const Tensor& w, const Tensor& y, int T, int width);
  void copy_rows(const Tensor& src, const Tensor& dst, int rows, int cols, int ld_src, int ld_dst);  // bf16
  void copy_rows_f32(const Tensor& src, const Tensor& dst, int rows, int cols, int ld_src, int ld_dst);
  // z <- cz z + c0 x0 + c1 x0_prev with x0 = z - sigma v; then x0_prev <- x0 (see kernels/norm.metal)
  void flow_step(const Tensor& z, const Tensor& v, const Tensor& x0_prev, int n, float sigma, float cz, float c0,
                 float c1);
  void euler_step(const Tensor& z, const Tensor& v, int n, float dt);
  void f32_copy(const Tensor& x, const Tensor& y, size_t n);
  void f32_axpy(const Tensor& x, const Tensor& y, float a, size_t n);
  void f32_to_bf16(const Tensor& x, const Tensor& y, int n);

  // ---- GPU <-> ANE hand-off (kernels/ane.metal); chunked ANE buffers are [chunk][channels][ld_t] ----
  // st[r] = rrms of row r (f32 rows of `width`)
  void rms_stats(const Tensor& x, const Tensor& st, int rows, int width, float eps);
  // y = x * st * mul (+ add) -> bf16 rows (ld_y) and fp16 ANE chunks times t_scale (rows_pad % 32 == 0)
  void rms_dual(const Tensor& x, const Tensor& st, const Tensor& mul, const Tensor& add, const Tensor& y,
                const Tensor& yt, int rows, int rows_pad, int width, int ld_y, int ld_t, float t_scale);
  void to_ane_bf16(const Tensor& in, const Tensor& out, int rows, int rows_pad, int width, int ld_in, int ld_t,
                   float scale);
  // x[r, c] += gate[c] * scale * y[c, r] (one ANE chunk, row stride ld_y); with y2: both inputs in one pass
  void resid_gate_add_t16(const Tensor& x, const Tensor& y, const Tensor& gate, int rows, int width, int ld_y,
                          float scale, const Tensor& y2 = Tensor());
  // out[r, col0 + j] = in[j, r] for j < n (n % 64 == 0; one ANE chunk, row stride ld_in)
  void ane_cols_scatter(const Tensor& in, const Tensor& out, int rows, int n, int ld_in, int ld_out, int col0);

  // Cross-device synchronization inside the current command buffer (closes and reopens the
  // compute encoder around the event operation).
  void signal_event(id<MTLSharedEvent> ev, uint64_t value);
  void wait_event(id<MTLSharedEvent> ev, uint64_t value);
  // Commit the current command buffer and continue in a fresh one (no CPU wait). An event signal
  // inside a command buffer is only delivered once the GPU finishes the segment up to the next
  // wait, so a hand-off signal must end its command buffer to fire as soon as its inputs are ready.
  void split() { commit(); begin(); }

  // Profiling (env KREA_PROFILE=1). mark() ends the current command buffer under `label`;
  // committed buffers record their GPU start/end (host time base, seconds) so time can be
  // attributed per op, and host-side work (ANE calls) is recorded with record_host().
  struct ProfEvent {
    std::string label;
    double start, end;
    bool gpu;
  };
  bool profiling() const { return profile_; }
  void mark(const char* label) {
    if (!profile_) return;
    cb_.label = @(label);
    split();
  }
  void record_host(const std::string& label, double start, double end);
  std::vector<ProfEvent> take_profile();
  static double host_now();  // same time base as MTLCommandBuffer.GPUStartTime

  // encoder access for model-specific kernels
  id<MTLComputeCommandEncoder> encoder() { return enc_; }
  id<MTLCommandBuffer> command_buffer() { return cb_; }  // the one being encoded (nil outside begin/commit)
  void dispatch(const std::string& kernel, MTLSize grid_tg, MTLSize tg, std::initializer_list<Tensor> bufs,
                const void* params = nullptr, size_t psize = 0, int param_index = -1);
  void dispatch_threads(const std::string& kernel, MTLSize threads, MTLSize tg, std::initializer_list<Tensor> bufs,
                        const void* params = nullptr, size_t psize = 0, int param_index = -1);

 private:
  id<MTLLibrary> lib_ = nil;
  std::unordered_map<std::string, id<MTLComputePipelineState>> psos_;
  id<MTLCommandBuffer> cb_ = nil;
  id<MTLCommandBuffer> last_cb_ = nil;
  id<MTLComputeCommandEncoder> enc_ = nil;
  double last_gpu_ms_ = 0;
  bool profile_ = false;
  std::mutex prof_mu_;
  std::vector<ProfEvent> prof_;
  void bind(std::initializer_list<Tensor> bufs, const void* params, size_t psize, int param_index);
};

}  // namespace krea
