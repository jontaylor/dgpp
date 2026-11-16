#pragma once
// Bounded cache for immutable, separately stored payload/scale rows. Storage
// alignment is 4 KiB even on hosts with 64 KiB virtual-memory pages.
#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fcntl.h>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace dgpp {
inline bool direct_engram_enabled() {
  const char* value = std::getenv("DGPP_ENGRAM_DIRECT");
  return value && std::atoi(value) != 0;
}
class DirectRowCache {
 public:
  static constexpr size_t kWorkers = 16, kBlock = 4096, kWays = 4, kSlots = 1 << 18;
  static constexpr size_t kSets = kSlots / kWays;
  struct Stats { uint64_t reads, bytes, hits, misses; };
  static size_t reserved_bytes(size_t row_bytes) {
    return kSlots * (row_bytes + sizeof(uint32_t)) + kSets + kWorkers * 2 * kBlock;
  }
  // Per-call indices: one row id + one destination offset per requested row.
  static size_t index_bytes(size_t rows) { return rows * sizeof(Request) + kWorkers * sizeof(std::vector<Request>); }
  DirectRowCache(const std::string& path, uint64_t payload, uint64_t scales,
                 uint64_t rows, size_t payload_bytes, size_t scale_bytes)
      : payload_(payload), scales_(scales), rows_(rows), pb_(payload_bytes), sb_(scale_bytes),
        data_(kSlots * (pb_+sb_)), tags_(kSlots, UINT32_MAX), victims_(kSets) {
    if (!rows || rows > INT32_MAX || !pb_ || !sb_) throw std::invalid_argument("direct rows: invalid geometry");
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
    if (fd < 0) throw std::runtime_error("direct rows: cannot open " + path);
    struct Close { int fd; ~Close() { if (fd >= 0) ::close(fd); } } guard{fd};
    struct stat st{};
    if (::fstat(fd, &st) || st.st_size < 0 || payload > uint64_t(st.st_size) || scales > uint64_t(st.st_size) ||
        rows > (uint64_t(st.st_size)-payload)/pb_ || rows > (uint64_t(st.st_size)-scales)/sb_)
      throw std::runtime_error("direct rows: table exceeds file");
    for (auto& worker : workers_) for (auto& block : worker.blocks) {
      void* p = nullptr;
      if (::posix_memalign(&p, kBlock, kBlock)) throw std::bad_alloc();
      block.data.reset(static_cast<uint8_t*>(p));
    }
    fd_ = fd;
    const char* parallel = std::getenv("DGPP_ENGRAM_PARALLEL_SMALL");
    if (parallel && std::atoi(parallel) != 0) {
      for (size_t w = 0; w < kWorkers; ++w) {
        threads_.emplace_back([this, w](std::stop_token stop) {
          uint64_t seen = 0;
          std::unique_lock lock(pool_mutex_);
          while (pool_ready_.wait(lock, stop, [&] { return generation_ != seen; })) {
            seen = generation_;
            lock.unlock();
            pool_job_(w);
            lock.lock();
            if (--pending_ == 0) pool_done_.notify_one();
          }
        });
      }
    }
    guard.fd = -1;
  }
  ~DirectRowCache() {
    for (auto& thread : threads_) thread.request_stop();
    pool_ready_.notify_all();
    threads_.clear();
    if (fd_ >= 0) ::close(fd_);
  }
  DirectRowCache(const DirectRowCache&) = delete;
  DirectRowCache& operator=(const DirectRowCache&) = delete;
  Stats stats() const { return {reads_.load(), bytes_.load(), hits_.load(), misses_.load()}; }
  void gather(const int32_t* ids, int64_t stride, const int32_t* selected, int heads, int n, uint8_t* dst) {
    if (n <= 0 || heads <= 0) return;
    // The model's current and look-ahead channels may call the same table.
    // One owner protects cache tags and the bounded per-worker block buffers.
    std::lock_guard lock(mutex_);
    std::array<std::vector<Request>, kWorkers> requests;
    const size_t total = static_cast<size_t>(n)*heads;
    for (size_t i = 0; i < total; ++i) {
      const int32_t row = ids[(i/heads)*stride + selected[i%heads]];
      if (row < 0 || uint64_t(row) >= rows_) throw std::out_of_range("direct rows: row index");
      requests[set(row)%kWorkers].push_back({static_cast<uint32_t>(row), i});
    }
    std::array<std::exception_ptr, kWorkers> errors{};
    const auto work = [&](size_t w) {
      try {
        auto& jobs = requests[w];
        // Preserve output order through explicit offsets, while improving
        // block locality and making repeated ids hit the compact row cache.
        if (total > 256) std::sort(jobs.begin(), jobs.end(), [](auto a, auto b) { return a.row < b.row; });
        for (const auto& job : jobs) {
          const auto* row = get(w, job.row);
          std::memcpy(dst + job.output*(pb_+sb_), row, pb_+sb_);
        }
      } catch (...) { errors[w] = std::current_exception(); }
    };
    size_t missing = 0;
    if (total <= 256 && !threads_.empty()) {
      for (const auto& jobs : requests) {
        for (const auto& job : jobs) {
          const size_t first = set(job.row)*kWays;
          if (std::find(tags_.begin()+first, tags_.begin()+first+kWays, job.row) == tags_.begin()+first+kWays &&
              ++missing == 2) break;
        }
        if (missing == 2) break;
      }
    }
    if (total <= 256 && missing >= 2) {
      std::unique_lock pool_lock(pool_mutex_);
      pool_job_ = work;
      pending_ = kWorkers;
      ++generation_;
      pool_ready_.notify_all();
      pool_done_.wait(pool_lock, [&] { return pending_ == 0; });
      pool_job_ = {};
    } else if (total <= 256) {
      for (size_t w = 0; w < kWorkers; ++w) work(w);
    } else {
      // jthread also joins already-created workers if thread creation fails.
      std::vector<std::jthread> threads;
      for (size_t w = 0; w < kWorkers; ++w)
        if (!requests[w].empty()) threads.emplace_back(work, w);
    }
    for (auto error : errors) if (error) std::rethrow_exception(error);
  }
 private:
  struct Request { uint32_t row; size_t output; };
  struct Block {
    std::unique_ptr<uint8_t, decltype(&std::free)> data{nullptr, &std::free};
    uint64_t offset = UINT64_MAX;
    size_t valid = 0;
  };
  struct Worker { std::array<Block, 2> blocks; };
  static size_t set(uint32_t row) {
    uint32_t h = row;
    h ^= h >> 16; h *= 0x7feb352dU; h ^= h >> 15; h *= 0x846ca68bU; h ^= h >> 16;
    return h % kSets;
  }
  void copy(Block& block, uint64_t offset, size_t bytes, uint8_t* dst) {
    while (bytes) {
      const uint64_t begin = offset / kBlock * kBlock;
      const size_t within = offset - begin, take = std::min(bytes, kBlock-within);
      if (block.offset != begin) {
        block.offset = UINT64_MAX;
        ssize_t got;
        do { got = ::pread(fd_, block.data.get(), kBlock, static_cast<off_t>(begin)); } while (got < 0 && errno == EINTR);
        if (got <= 0) throw std::runtime_error("direct rows: failed block read");
        block.valid = static_cast<size_t>(got); block.offset = begin;
        ++reads_; bytes_ += got;
      }
      if (within+take > block.valid) throw std::runtime_error("direct rows: truncated row");
      std::memcpy(dst, block.data.get()+within, take);
      offset += take; dst += take; bytes -= take;
    }
  }
  const uint8_t* get(size_t worker, uint32_t row) {
    const size_t group = set(row), first = group*kWays;
    for (size_t i = 0; i < kWays; ++i) if (tags_[first+i] == row) {
      ++hits_; return data_.data()+(first+i)*(pb_+sb_);
    }
    ++misses_;
    const size_t slot = first + victims_[group];
    victims_[group] = (victims_[group]+1)%kWays;
    tags_[slot] = UINT32_MAX;
    auto* dst = data_.data()+slot*(pb_+sb_);
    copy(workers_[worker].blocks[0], payload_+uint64_t(row)*pb_, pb_, dst);
    copy(workers_[worker].blocks[1], scales_+uint64_t(row)*sb_, sb_, dst+pb_);
    tags_[slot] = row;  // publish only after both immutable planes are complete
    return dst;
  }
  int fd_ = -1;
  uint64_t payload_, scales_, rows_;
  size_t pb_, sb_;
  std::vector<uint8_t> data_;
  std::vector<uint32_t> tags_;
  std::vector<uint8_t> victims_;
  std::array<Worker, kWorkers> workers_;
  std::mutex mutex_;
  std::atomic<uint64_t> reads_{0}, bytes_{0}, hits_{0}, misses_{0};
  std::mutex pool_mutex_;
  std::condition_variable_any pool_ready_, pool_done_;
  std::function<void(size_t)> pool_job_;
  uint64_t generation_ = 0;
  size_t pending_ = 0;
  std::vector<std::jthread> threads_;
};
}  // namespace dgpp
