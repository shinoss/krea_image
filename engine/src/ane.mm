#include "ane.h"

#import <CoreML/CoreML.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <deque>
#include <mutex>
#include <stdexcept>

namespace krea {

static constexpr int D = 6144, QKVG_N = 15360;
static const char* kNames[ANEOffload::kKinds] = {"qkvg", "mlp", "o"};

struct ANEOffload::Impl {
  struct Job {
    int kind, layer, c0, c1;
    uint64_t ticket;
  };
  std::vector<MLModel*> models[kKinds];                // [kind][layer]
  std::vector<MLMultiArray*> in[kKinds], out[kKinds];  // per-chunk views into the shared buffers
  id<MTLSharedEvent> go = nil, done = nil;
  MTLSharedEventListener* listener = nil;
  dispatch_queue_t queue = nil;
  uint64_t issued = 0;
  std::mutex mu;         // guards jobs and error
  std::deque<Job> jobs;  // launched, not yet run (ticket order)
  std::string error;
};

static double since_ms(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// One Core ML call on chunk views; returns false (and sets err) on failure.
static bool predict(MLModel* model, MLMultiArray* in, MLMultiArray* out, std::string* err_out) {
  NSError* err = nil;
  MLDictionaryFeatureProvider* fp = [[MLDictionaryFeatureProvider alloc]
      initWithDictionary:@{@"x" : [MLFeatureValue featureValueWithMultiArray:in]} error:&err];
  MLPredictionOptions* opts = [MLPredictionOptions new];
  opts.outputBackings = @{@"out" : out};
  id<MLFeatureProvider> res = [model predictionFromFeatures:fp options:opts error:&err];
  if (!res) {
    if (err_out && err_out->empty()) *err_out = err ? err.localizedDescription.UTF8String : "prediction failed";
    return false;
  }
  // Core ML may return its own buffer instead of the backing; copy if so (same contiguous layout).
  MLMultiArray* r = [res featureValueForName:@"out"].multiArrayValue;
  if (r && r.dataPointer != out.dataPointer) {
    [r getBytesWithHandler:^(const void* bytes, NSInteger size) { memcpy(out.dataPointer, bytes, (size_t)size); }];
  }
  return true;
}

static std::vector<MLMultiArray*> chunk_views(const Tensor& t, int channels, int chunk) {
  std::vector<MLMultiArray*> v;
  const size_t bytes = (size_t)channels * chunk * 2;
  NSArray* shape = @[ @1, @(channels), @1, @(chunk) ];
  NSArray* strides = @[ @((long)channels * chunk), @(chunk), @(chunk), @1 ];
  for (int c = 0; c < ANEOffload::kMaxChunks; c++) {
    NSError* err = nil;
    MLMultiArray* a = [[MLMultiArray alloc] initWithDataPointer:(char*)t.ptr<void>() + c * bytes
                                                          shape:shape
                                                       dataType:MLMultiArrayDataTypeFloat16
                                                        strides:strides
                                                    deallocator:nil
                                                          error:&err];
    if (!a) throw std::runtime_error("ANE: cannot wrap buffers");
    v.push_back(a);
  }
  return v;
}

int ANEOffload::out_channels(Kind k) const { return k == QKVG ? QKVG_N - qkvg_c0_ : D; }

ANEOffload::ANEOffload(Metal& m, const std::string& dir) : impl_(new Impl), m_(m) {
  NSError* err = nil;
  NSData* meta = [NSData dataWithContentsOfFile:[NSString stringWithFormat:@"%s/meta.json", dir.c_str()]];
  NSDictionary* md = meta ? [NSJSONSerialization JSONObjectWithData:meta options:0 error:&err] : nil;
  if (!md) throw std::runtime_error("ANE: missing or unreadable " + dir + "/meta.json");
  chunk_ = [md[@"chunk"] intValue];
  gpu_units_ = [md[@"gpu_units"] intValue];
  qkvg_c0_ = [md[@"qkvg_c0"] intValue];
  out_scale_ = md[@"out_scale"] ? [md[@"out_scale"] floatValue] : 1.0f;
  has_oproj_ = [md[@"o_proj"] boolValue] && getenv("KREA_ANE_NO_O") == nullptr;
  o_scale_ = md[@"o_scale"] ? [md[@"o_scale"] floatValue] : 1.0f;
  mode_ = md[@"mode"] ? [md[@"mode"] UTF8String] : "int8";
  if (chunk_ % 32 || chunk_ <= 0) throw std::runtime_error("ANE: chunk must be a positive multiple of 32");

  // Buffers are sized for kMaxChunks but only the chunks actually used are ever touched.
  for (int k = 0; k < kKinds; k++) {
    if (!has((Kind)k)) continue;
    in_[k] = m_.alloc(chunk_bytes(D) * kMaxChunks);
    out_[k] = m_.alloc(chunk_bytes(out_channels((Kind)k)) * kMaxChunks);
    impl_->in[k] = chunk_views(in_[k], D, chunk_);
    impl_->out[k] = chunk_views(out_[k], out_channels((Kind)k), chunk_);
  }

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
  fprintf(stderr, "[ane] programs ready (%s, chunk %d): MLP %d GPU / %d ANE units, q|k|v|gate columns %d GPU / %d ANE%s (%.1f s)\n",
          mode_.c_str(), chunk_, gpu_units_, 16384 - gpu_units_, qkvg_c0_, QKVG_N - qkvg_c0_, has_oproj_ ? ", O-proj" : "",
          since_ms(t0) / 1e3);

  impl_->go = [m_.dev newSharedEvent];
  impl_->done = [m_.dev newSharedEvent];
  impl_->queue = dispatch_queue_create("krea.ane", DISPATCH_QUEUE_SERIAL);
  impl_->listener = [[MTLSharedEventListener alloc] initWithDispatchQueue:impl_->queue];
}

ANEOffload::~ANEOffload() = default;

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

void ANEOffload::gpu_wait(uint64_t ticket) {
  {
    std::lock_guard<std::mutex> g(impl_->mu);
    if (!impl_->error.empty()) throw std::runtime_error("ANE: " + impl_->error);
  }
  m_.wait_event(impl_->done, ticket);
}

}  // namespace krea
