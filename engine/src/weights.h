// Zero-copy loader for the engine's .qw weight files (see tools/convert.py): the file is mmapped and
// every "segment" (all tensors sharing a name prefix up to the first '.', e.g. one DiT layer "L12")
// is wrapped in its own no-copy Metal buffer. Segments keep each buffer under Metal's maximum buffer
// length (21.7 GB on this machine; the bf16 DiT is 24.3 GB) and mean a GPU command only ever pins the
// pages of the segments it uses, never the whole file. Tensors of one segment are contiguous (the
// converter writes them grouped) and 16 KiB aligned, so every segment is page aligned.
#pragma once
#include <string>
#include <unordered_map>
#include <vector>

#include "metal.h"

namespace krea {

struct TensorInfo {
  std::string dtype;
  std::vector<int64_t> shape;
  size_t offset = 0, nbytes = 0;
  int segment = -1;
};

class WeightFile {
 public:
  WeightFile(Metal& mtl, const std::string& path);
  ~WeightFile();
  WeightFile(const WeightFile&) = delete;

  Tensor get(const std::string& name) const;
  // GEMM weight: int8 + scales if the file stores `name` quantized (companion `name.s`).
  Weight linear(const std::string& name) const {
    if (has(name + ".s")) return Weight{get(name), get(name + ".s"), true};
    return Weight{get(name), Tensor(), false};
  }
  const TensorInfo& info(const std::string& name) const;
  bool has(const std::string& name) const { return tensors_.count(name) > 0; }
  size_t size() const { return size_; }
  const std::string& path() const { return path_; }
  // JSON "meta" object of the header (as a string: small, parsed by the caller when needed)
  const std::string& meta_json() const { return meta_; }
  // Touch every page so the first denoising step does not pay for page faults.
  void prefetch() const;
  // Byte range [off, off + len) of the data section covering every tensor whose name starts with
  // `prefix`, and range-wise read-ahead / release (the pages stay file-backed: released ones are
  // simply re-read on next use).
  std::pair<size_t, size_t> span(const std::string& prefix) const;
  void prefetch_range(size_t off, size_t len) const;
  void discard_range(size_t off, size_t len) const;
  // Read data-section bytes [off, off + len) straight from the file into dst with the file cache
  // bypassed (F_NOCACHE, several threads): ~4 GB/s, and nothing else gets evicted.
  void read_range(size_t off, size_t len, void* dst) const;
  // CPU pointer to data-section byte `off` (through the mapping; touches only the pages read).
  const void* host_ptr(size_t off) const { return (const char*)base_ + data_start_ + off; }

 private:
  std::string path_, meta_;
  void* base_ = nullptr;
  size_t size_ = 0, map_size_ = 0, data_start_ = 0;
  struct Segment {
    size_t lo, hi;  // data-section byte range
    id<MTLBuffer> buf;
  };
  std::vector<Segment> segs_;
  std::unordered_map<std::string, TensorInfo> tensors_;
};

}  // namespace krea
