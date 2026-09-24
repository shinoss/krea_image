#include "weights.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <map>
#include <stdexcept>
#include <thread>

namespace krea {

WeightFile::WeightFile(Metal& mtl, const std::string& path) : path_(path) {
  int fd = open(path.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("cannot open weights: " + path);
  struct stat st;
  fstat(fd, &st);
  size_ = st.st_size;
  const size_t page = getpagesize();
  map_size_ = (size_ + page - 1) / page * page;
  base_ = mmap(nullptr, map_size_, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (base_ == MAP_FAILED) throw std::runtime_error("mmap failed: " + path);

  const char* p = (const char*)base_;
  if (memcmp(p, "QWTS", 4) != 0) throw std::runtime_error("bad weight file magic: " + path);
  uint64_t hlen;
  memcpy(&hlen, p + 8, 8);
  NSData* hdata = [NSData dataWithBytesNoCopy:(void*)(p + 16) length:hlen freeWhenDone:NO];
  NSError* err = nil;
  NSDictionary* hdr = [NSJSONSerialization JSONObjectWithData:hdata options:0 error:&err];
  if (!hdr) throw std::runtime_error("bad weight header: " + path);
  data_start_ = (16 + hlen + 16383) / 16384 * 16384;
  if (NSDictionary* meta = hdr[@"meta"]) {
    NSData* mj = [NSJSONSerialization dataWithJSONObject:meta options:0 error:nil];
    if (mj) meta_.assign((const char*)mj.bytes, mj.length);
  }
  NSDictionary* ts = hdr[@"tensors"];
  // segment = name prefix up to the first '.'; its byte range spans all of its tensors
  std::map<std::string, std::pair<size_t, size_t>> ranges;
  for (NSString* name in ts) {
    NSDictionary* d = ts[name];
    TensorInfo ti;
    ti.dtype = [d[@"dtype"] UTF8String];
    for (NSNumber* n in d[@"shape"]) ti.shape.push_back(n.longLongValue);
    ti.offset = [d[@"offset"] unsignedLongLongValue];
    ti.nbytes = [d[@"nbytes"] unsignedLongLongValue];
    const std::string n = name.UTF8String;
    const std::string seg = n.substr(0, n.find('.'));
    auto it = ranges.find(seg);
    if (it == ranges.end()) ranges[seg] = {ti.offset, ti.offset + ti.nbytes};
    else it->second = {std::min(it->second.first, ti.offset), std::max(it->second.second, ti.offset + ti.nbytes)};
    tensors_[n] = ti;
  }
  std::map<std::string, int> seg_index;
  for (auto& [name, r] : ranges) {
    const size_t lo = r.first / page * page, hi = std::min(map_size_ - data_start_, (r.second + page - 1) / page * page);
    id<MTLBuffer> b = [mtl.dev newBufferWithBytesNoCopy:(char*)base_ + data_start_ + lo
                                                  length:std::max(hi - lo, page)
                                                 options:MTLResourceStorageModeShared
                                             deallocator:nil];
    if (!b) throw std::runtime_error("cannot wrap weight segment " + name + " of " + path + " in a Metal buffer");
    seg_index[name] = (int)segs_.size();
    segs_.push_back({lo, hi, b});
  }
  for (auto& [n, ti] : tensors_) {
    ti.segment = seg_index[n.substr(0, n.find('.'))];
    const Segment& s = segs_[ti.segment];
    if (ti.offset < s.lo || ti.offset + ti.nbytes > s.hi) throw std::runtime_error("segments of " + path + " overlap");
  }
  // tensors of different segments must not interleave (a segment's buffer would cover the other's)
  std::vector<std::pair<size_t, size_t>> spans;
  for (const Segment& s : segs_) spans.push_back({s.lo, s.hi});
  std::sort(spans.begin(), spans.end());
  for (size_t i = 1; i < spans.size(); i++)
    if (spans[i].first < spans[i - 1].second)
      throw std::runtime_error("weight file " + path + ": segments are not contiguous (reconvert)");
}

WeightFile::~WeightFile() {
  segs_.clear();
  if (base_) munmap(base_, map_size_);
}

const TensorInfo& WeightFile::info(const std::string& name) const {
  auto it = tensors_.find(name);
  if (it == tensors_.end()) throw std::runtime_error("missing tensor: " + name + " in " + path_);
  return it->second;
}

Tensor WeightFile::get(const std::string& name) const {
  const TensorInfo& ti = info(name);
  const Segment& s = segs_[ti.segment];
  return Tensor{s.buf, ti.offset - s.lo, ti.nbytes};
}

void WeightFile::prefetch() const { prefetch_range(0, map_size_ - data_start_); }

void WeightFile::make_resident(Metal& m) const {
  id<MTLBuffer> sink = [m.dev newBufferWithLength:4 * std::max<size_t>(1, segs_.size()) options:MTLResourceStorageModeShared];
  id<MTLCommandBuffer> cb = [m.queue commandBuffer];
  id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
  for (size_t i = 0; i < segs_.size(); i++)
    [bl copyFromBuffer:segs_[i].buf sourceOffset:0 toBuffer:sink destinationOffset:4 * i size:4];
  [bl endEncoding];
  [cb commit];
  [cb waitUntilCompleted];
}

void WeightFile::prefetch_range(size_t off, size_t len) const {
  // Touch one byte per page from 8 threads over contiguous ranges: resident pages cost ~nothing,
  // evicted ones are read back at SSD speed (much faster than GPU-side page faults).
  const size_t page = getpagesize();
  const size_t lo0 = (data_start_ + off) / page * page;
  const size_t hi0 = std::min(map_size_, data_start_ + off + len);
  if (hi0 <= lo0) return;
  madvise((char*)base_ + lo0, hi0 - lo0, MADV_WILLNEED);
  const int nt = 8;
  const size_t chunk = ((hi0 - lo0) / nt + page - 1) / page * page;
  std::vector<std::thread> th;
  for (int t = 0; t < nt; t++) {
    th.emplace_back([=] {
      volatile char sink = 0;
      const char* p = (const char*)base_;
      const size_t lo = lo0 + t * chunk, hi = std::min(hi0, lo + chunk);
      for (size_t o = lo; o < hi; o += page) sink += p[o];
      (void)sink;
    });
  }
  for (auto& x : th) x.join();
}

void WeightFile::read_range(size_t off, size_t len, void* dst) const {
  const int nt = 4;
  const size_t chunk = (len / nt + 16383) / 16384 * 16384;
  std::vector<std::thread> th;
  bool ok = true;
  for (int t = 0; t < nt; t++) {
    const size_t lo = t * chunk, hi = std::min(len, lo + chunk);
    if (lo >= hi) break;
    th.emplace_back([&, lo, hi] {
      const int fd = open(path_.c_str(), O_RDONLY);
      if (fd < 0) { ok = false; return; }
      fcntl(fd, F_NOCACHE, 1);
      size_t o = lo;
      while (o < hi) {
        const ssize_t n = pread(fd, (char*)dst + o, std::min<size_t>(hi - o, 64 << 20), (off_t)(data_start_ + off + o));
        if (n <= 0) { ok = false; break; }
        o += (size_t)n;
      }
      close(fd);
    });
  }
  for (auto& x : th) x.join();
  if (!ok) throw std::runtime_error("read failed: " + path_);
}

void WeightFile::discard_range(size_t off, size_t len) const {
  const size_t page = getpagesize();
  const size_t lo = (data_start_ + off + page - 1) / page * page, hi = (data_start_ + off + len) / page * page;
  if (hi > lo) madvise((char*)base_ + lo, hi - lo, MADV_DONTNEED);
}

std::pair<size_t, size_t> WeightFile::span(const std::string& prefix) const {
  size_t lo = SIZE_MAX, hi = 0;
  for (const auto& [name, ti] : tensors_)
    if (name.compare(0, prefix.size(), prefix) == 0) {
      lo = std::min(lo, ti.offset);
      hi = std::max(hi, ti.offset + ti.nbytes);
    }
  return lo < hi ? std::make_pair(lo, hi - lo) : std::make_pair((size_t)0, (size_t)0);
}

}  // namespace krea
