#pragma once
// File-worker I/O only. CUDA transfers and completion-to-cache publication
// remain the engine thread's job. All staging is reserved before workers start.
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <span>
#include <vector>

#include "loaders/expert_store.hpp"

namespace dgpp::expert {
struct ReadPool;

class ReadLease {
 public:
  ~ReadLease();
  ReadLease(const ReadLease&) = delete;
  ReadLease& operator=(const ReadLease&) = delete;
  std::span<const uint8_t> bytes() const;

 private:
  friend struct ReadQueueImpl;
  ReadLease(std::shared_ptr<ReadPool> pool, size_t slot, size_t size);
  std::shared_ptr<ReadPool> pool_;
  size_t slot_, size_;
  bool active_ = false;
};

class ReadTicket {
 public:
  bool ready() const;
  bool wait_for(std::chrono::milliseconds timeout) const;
  // Throws on I/O failure or cancellation. The same immutable lease is shared
  // by all coalesced consumers. Keep it through the last staging-buffer reader.
  // Release both the ticket and its leases afterward to return staging capacity.
  std::shared_ptr<const ReadLease> wait() const;

 private:
  friend struct ReadQueueImpl;
  ReadTicket();
  std::promise<std::shared_ptr<const ReadLease>> completion_;
  std::shared_future<std::shared_ptr<const ReadLease>> result_;
};

enum class ReadPriority { kDemand, kPrefetch };
struct ReadQueueConfig {
  bool operator==(const ReadQueueConfig&) const = default;
  size_t buffers = 2, workers = 1, max_pending = 8;
  uint64_t staging_budget_bytes = 0;
  uint64_t prefetch_bytes = 0;  // queued + active speculative record bytes
};
struct ReadQueueStats {
  uint64_t accepted = 0, coalesced = 0, promoted = 0, refused = 0;
  uint64_t completed = 0, failed = 0, cancelled = 0;
  uint64_t pending = 0, active = 0, speculative_bytes = 0;
  uint64_t staging_reserved = 0, staging_in_use = 0, staging_peak = 0;
};

class ExpertReadQueue {
 public:
  ExpertReadQueue(std::shared_ptr<const ExpertStore> store, ReadQueueConfig config);
  ~ExpertReadQueue();
  ExpertReadQueue(const ExpertReadQueue&) = delete;
  ExpertReadQueue& operator=(const ExpertReadQueue&) = delete;
  // Returns nullptr for bounded backpressure. Coalescing is checked before
  // capacity, so a full queue still accepts another waiter for the same read.
  // Demand takes priority and can displace a queued speculative read.
  std::shared_ptr<ReadTicket> request(size_t record, ReadPriority priority = ReadPriority::kDemand);
  // Wait for any selected ticket, including failures/cancellation. Returns its
  // index; callers consume the result with wait(). Tickets must belong here.
  size_t wait_any(std::span<const std::shared_ptr<ReadTicket>> tickets) const;
  ReadQueueStats stats() const;
  // Stops new work, cancels queued/blocked work and joins active file reads.
  // Completed leases remain valid afterward, without freeing their staging.
  void shutdown();
  // Stable aligned spans, for an engine's optional startup registration.
  // The caller must drain leases and unregister BEFORE releasing this queue;
  // the file worker never registers, unregisters or transfers CUDA memory.
  std::vector<std::span<uint8_t>> staging_spans() const;

 private:
  std::unique_ptr<struct ReadQueueImpl> impl_;
};
}  // namespace dgpp::expert
