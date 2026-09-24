#include "ane.h"

#import <CoreML/CoreML.h>
#import <CoreVideo/CoreVideo.h>
#import <IOSurface/IOSurface.h>
#include <unistd.h>

#include <array>
#include <chrono>
#include <cstdio>
#include <deque>
#include <mutex>
#include <stdexcept>

namespace krea {

static constexpr int D = 6144, QG_N = 2 * 6144;
static const char* kNames[ANEOffload::kKinds] = {"qg", "mlp", "o"};

// fp16 scan for KREA_ANE_STATS: largest finite magnitude (as fp16 bits) and count of inf / nan.
static void scan_f16(const void* p, size_t n, uint16_t* max_bits, size_t* bad) {
  const uint16_t* h = (const uint16_t*)p;
  uint16_t mx = *max_bits;
  size_t b = 0;
  for (size_t i = 0; i < n; i++) {
    const uint16_t a = h[i] & 0x7fff;
    if (a >= 0x7c00) b++;
    else if (a > mx) mx = a;
  }
  *max_bits = mx;
  *bad += b;
}

static float f16_to_float(uint16_t b) {
  __fp16 h;
  memcpy(&h, &b, 2);
  return (float)h;
}

// One chunk buffer shared by the GPU and the ANE: an IOSurface-backed pixel buffer (width = chunk rows,
// height = channels, fp16) seen by Metal as a no-copy buffer and by Core ML as an MLMultiArray.
struct Surf {
  CVPixelBufferRef pb = nullptr;
  void* host = nullptr;
  Tensor t;
  MLMultiArray* arr = nil;
};

static Surf make_surface(Metal& m, int channels, int chunk) {
  Surf s;
  NSDictionary* attrs = @{(id)kCVPixelBufferIOSurfacePropertiesKey : @{}};
  if (CVPixelBufferCreate(kCFAllocatorDefault, chunk, channels, kCVPixelFormatType_OneComponent16Half,
                          (__bridge CFDictionaryRef)attrs, &s.pb) != kCVReturnSuccess)
    throw std::runtime_error("ANE: cannot create an IOSurface chunk buffer");
  IOSurfaceRef io = CVPixelBufferGetIOSurface(s.pb);
  if (!io || IOSurfaceGetBytesPerRow(io) != (size_t)chunk * 2)
    throw std::runtime_error("ANE: IOSurface rows are padded; the chunk size must be a multiple of 32");
  IOSurfaceLock(io, 0, nullptr);
  s.host = IOSurfaceGetBaseAddress(io);
  IOSurfaceUnlock(io, 0, nullptr);
  id<MTLBuffer> b = [m.dev newBufferWithBytesNoCopy:s.host
                                             length:IOSurfaceGetAllocSize(io)
                                            options:MTLResourceStorageModeShared
                                        deallocator:nil];
  if (!b) throw std::runtime_error("ANE: cannot wrap an IOSurface as a Metal buffer");
  s.t = Tensor{b, 0, (size_t)channels * chunk * 2};
  s.arr = [[MLMultiArray alloc] initWithPixelBuffer:s.pb shape:@[ @1, @(channels), @1, @(chunk) ]];
  return s;
}

static void free_surface(Surf& s) {
  s.arr = nil;
  s.t = Tensor();  // the Metal buffer must go before the memory it wraps
  if (s.pb) CVPixelBufferRelease(s.pb);
  s = Surf();
}

struct ANEOffload::Impl {
  struct Job {
    int kind, layer, c0, c1;
    uint64_t ticket;
  };
  std::vector<MLModel*> models[kKinds];  // [kind][layer]
  std::vector<Surf> in[kKinds], out[kKinds];  // [kind][chunk], kMaxChunks entries, filled by ensure_chunks
  ~Impl() {
    for (int k = 0; k < kKinds; k++) {
      for (Surf& s : in[k]) free_surface(s);
      for (Surf& s : out[k]) free_surface(s);
    }
  }
  id<MTLSharedEvent> go = nil, done = nil;
  MTLSharedEventListener* listener = nil;
  dispatch_queue_t queue = nil;
  uint64_t issued = 0;
  std::mutex mu;         // guards jobs and error
  std::deque<Job> jobs;  // launched, not yet run (ticket order)
  std::string error;
  bool stats = false;
  struct Stat {
    uint16_t in_max = 0, out_max = 0;
    size_t bad = 0, calls = 0;
  };
  std::vector<Stat> stat[kKinds];  // [kind][layer]
};

static double since_ms(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// One Core ML call on chunk buffers; returns false (and sets err) on failure.
static bool predict(MLModel* model, const Surf& in, const Surf& out, std::string* err_out) {
  NSError* err = nil;
  MLDictionaryFeatureProvider* fp = [[MLDictionaryFeatureProvider alloc]
      initWithDictionary:@{@"x" : [MLFeatureValue featureValueWithMultiArray:in.arr]} error:&err];
  MLPredictionOptions* opts = [MLPredictionOptions new];
  opts.outputBackings = @{@"out" : out.arr};
  id<MLFeatureProvider> res = [model predictionFromFeatures:fp options:opts error:&err];
  if (!res) {
    if (err_out && err_out->empty()) *err_out = err ? err.localizedDescription.UTF8String : "prediction failed";
    return false;
  }
  // Core ML may return its own buffer instead of the backing; copy if so (same contiguous layout).
  MLMultiArray* r = [res featureValueForName:@"out"].multiArrayValue;
  if (r && r != out.arr && r.pixelBuffer != out.pb) {
    void* dst = out.host;
    const size_t cap = out.t.bytes;
    [r getBytesWithHandler:^(const void* bytes, NSInteger size) { memcpy(dst, bytes, std::min(cap, (size_t)size)); }];
  }
  return true;
}

int ANEOffload::out_channels(Kind k) const { return k == QG ? QG_N : D; }

void ANEOffload::ensure_chunks(int n) {
  if (n > kMaxChunks) throw std::runtime_error("ANE: too many chunks");
  for (int k = 0; k < kKinds; k++) {
    if (!has((Kind)k)) continue;
    for (int c = 0; c < n; c++) {
      if (!impl_->in[k][c].pb) impl_->in[k][c] = make_surface(m_, D, chunk_);
      if (!impl_->out[k][c].pb) impl_->out[k][c] = make_surface(m_, out_channels((Kind)k), chunk_);
    }
  }
}

Tensor ANEOffload::input(Kind k, int c) const {
  if (!impl_->in[k][c].pb) throw std::runtime_error("ANE: chunk buffer not allocated");
  return impl_->in[k][c].t;
}

Tensor ANEOffload::output(Kind k, int c) const {
  if (!impl_->out[k][c].pb) throw std::runtime_error("ANE: chunk buffer not allocated");
  return impl_->out[k][c].t;
}

ANEOffload::ANEOffload(Metal& m, const std::string& dir) : impl_(new Impl), m_(m) {
  NSError* err = nil;
  NSData* meta = [NSData dataWithContentsOfFile:[NSString stringWithFormat:@"%s/meta.json", dir.c_str()]];
  NSDictionary* md = meta ? [NSJSONSerialization JSONObjectWithData:meta options:0 error:&err] : nil;
  if (!md) throw std::runtime_error("ANE: missing or unreadable " + dir + "/meta.json");
  chunk_ = [md[@"chunk"] intValue];
  gpu_units_ = [md[@"gpu_units"] intValue];
  if (!md[@"attn_split"] || ![md[@"attn_split"] isEqualToString:@"kv"])
    throw std::runtime_error("ANE: " + dir + " was built for an older GPU/ANE split; rebuild it with tools/build_ane.py");
  out_scale_ = md[@"out_scale"] ? [md[@"out_scale"] floatValue] : 1.0f;
  has_oproj_ = [md[@"o_proj"] boolValue] && getenv("KREA_ANE_NO_O") == nullptr;
  o_scale_ = md[@"o_scale"] ? [md[@"o_scale"] floatValue] : 1.0f;
  mode_ = md[@"mode"] ? [md[@"mode"] UTF8String] : "int8";
  if (chunk_ % 32 || chunk_ <= 0) throw std::runtime_error("ANE: chunk must be a positive multiple of 32");

  for (int k = 0; k < kKinds; k++) {
    impl_->in[k].resize(kMaxChunks);
    impl_->out[k].resize(kMaxChunks);
  }
  ensure_chunks(1);

  // Load + warm up: the first prediction compiles each program for the ANE (then cached by the OS). Doing
  // it here keeps compilation off the GPU timeline, where a command buffer stalled on a compiling ANE would
  // trip the GPU watchdog.
  const auto t0 = std::chrono::steady_clock::now();
  MLModelConfiguration* cfg = [MLModelConfiguration new];
  cfg.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
  const int layers = getenv("KREA_DIT_LAYERS") ? std::min(28, std::max(1, atoi(getenv("KREA_DIT_LAYERS")))) : 28;
  for (int kind = 0; kind < kKinds; kind++) {
    if (!has((Kind)kind)) continue;
    for (int l = 0; l < layers; l++) {
      NSString* path = [NSString stringWithFormat:@"%s/%s_%02d.mlmodelc", dir.c_str(), kNames[kind], l];
      MLModel* model = [MLModel modelWithContentsOfURL:[NSURL fileURLWithPath:path] configuration:cfg error:&err];
      if (!model)
        throw std::runtime_error(std::string("ANE: cannot load ") + path.UTF8String + ": " +
                                 (err ? err.localizedDescription.UTF8String : "?"));
      std::string perr;
      if (!predict(model, impl_->in[kind][0], impl_->out[kind][0], &perr)) throw std::runtime_error("ANE warm-up: " + perr);
      impl_->models[kind].push_back(model);
    }
  }
  fprintf(stderr, "[ane] programs ready (%s, chunk %d): MLP %d GPU / %d ANE units, k|v on the GPU, q|gate on the ANE%s (%.1f s)\n",
          mode_.c_str(), chunk_, gpu_units_, 16384 - gpu_units_, has_oproj_ ? ", O-proj" : "", since_ms(t0) / 1e3);
  impl_->stats = getenv("KREA_ANE_STATS") != nullptr;
  for (int k = 0; k < kKinds; k++) impl_->stat[k].resize(28);

  impl_->go = [m_.dev newSharedEvent];
  impl_->done = [m_.dev newSharedEvent];
  impl_->queue = dispatch_queue_create("krea.ane", DISPATCH_QUEUE_SERIAL);
  impl_->listener = [[MTLSharedEventListener alloc] initWithDispatchQueue:impl_->queue];
}

ANEOffload::~ANEOffload() { report(); }

uint64_t ANEOffload::launch(Kind kind, int layer, int c0, int c1) {
  Impl* I = impl_.get();
  if (!has(kind)) throw std::runtime_error("ANE: program kind not loaded");
  if (c1 > kMaxChunks || c0 >= c1) throw std::runtime_error("ANE: bad chunk range");
  const uint64_t ticket = ++I->issued;
  {
    std::lock_guard<std::mutex> g(I->mu);
    I->jobs.push_back({(int)kind, layer, c0, c1, ticket});
  }
  double* busy = &busy_ms_;
  Metal* mtl = &m_;
  const size_t in_elems = (size_t)D * chunk_;
  std::array<size_t, kKinds> out_elems;
  for (int k = 0; k < kKinds; k++) out_elems[k] = (size_t)out_channels((Kind)k) * chunk_;
  // The block runs every queued job up to its ticket, in order: robust to notifications that are
  // coalesced or delivered out of order, and `done` only ever increases.
  [I->go notifyListener:I->listener
                atValue:ticket
                  block:^(id<MTLSharedEvent>, uint64_t) {
                    for (;;) {
                      Impl::Job j;
                      {
                        std::lock_guard<std::mutex> g(I->mu);
                        if (I->jobs.empty() || I->jobs.front().ticket > ticket) break;
                        j = I->jobs.front();
                        I->jobs.pop_front();
                      }
                      const auto t0 = std::chrono::steady_clock::now();
                      MLModel* model = I->models[j.kind][j.layer];
                      std::string err;
                      for (int c = j.c0; c < j.c1; c++) {
                        @autoreleasepool {
                          predict(model, I->in[j.kind][c], I->out[j.kind][c], &err);
                        }
                        if (I->stats) {
                          Impl::Stat& st = I->stat[j.kind][j.layer];
                          size_t none = 0;
                          scan_f16(I->in[j.kind][c].host, in_elems, &st.in_max, &none);
                          scan_f16(I->out[j.kind][c].host, out_elems[j.kind], &st.out_max, &st.bad);
                          st.bad += none;
                          st.calls++;
                        }
                      }
                      if (!err.empty()) {
                        std::lock_guard<std::mutex> g(I->mu);
                        if (I->error.empty()) I->error = err;
                      }
                      const double ms = since_ms(t0);
                      *busy += ms;
                      if (mtl->profiling()) {
                        const double now = Metal::host_now();
                        mtl->record_host(std::string("ane.") + kNames[j.kind], now - ms / 1e3, now);
                      }
                      I->done.signaledValue = j.ticket;  // always signal, even on error, so the GPU never hangs
                    }
                  }];
  m_.signal_event(I->go, ticket);
  m_.split();
  return ticket;
}

void ANEOffload::report() const {
  if (!impl_->stats) return;
  fprintf(stderr, "[ane stats] per layer: max |input| / max |output| (fp16 values as seen by the ANE), non-finite\n");
  for (int k = 0; k < kKinds; k++) {
    float in_all = 0, out_all = 0;
    size_t bad = 0;
    for (int l = 0; l < 28; l++) {
      const Impl::Stat& st = impl_->stat[k][l];
      if (!st.calls) continue;
      fprintf(stderr, "  %-3s L%02d  in %8.2f  out %9.2f  %s\n", kNames[k], l, f16_to_float(st.in_max),
              f16_to_float(st.out_max), st.bad ? "NON-FINITE" : "");
      in_all = std::max(in_all, f16_to_float(st.in_max));
      out_all = std::max(out_all, f16_to_float(st.out_max));
      bad += st.bad;
    }
    fprintf(stderr, "  %-3s all  in %8.2f  out %9.2f  non-finite %zu\n", kNames[k], in_all, out_all, bad);
  }
}

void ANEOffload::gpu_wait(uint64_t ticket) {
  {
    std::lock_guard<std::mutex> g(impl_->mu);
    if (!impl_->error.empty()) throw std::runtime_error("ANE: " + impl_->error);
  }
  m_.wait_event(impl_->done, ticket);
}

}  // namespace krea
