#include "loaders/expert_read_queue.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>
#if defined(__aarch64__) && defined(__linux__)
#include <sys/auxv.h>
#endif

namespace dgpp::expert {
namespace {
bool timed_reader_wait_available() {
#if defined(__aarch64__) && defined(__linux__)
  constexpr unsigned long wfxt = 1UL << 31;
  const char* mode = std::getenv("DGPP_EXPERT_READY_WORKER");
  return mode && std::atoi(mode) == 2 && (getauxval(AT_HWCAP2) & wfxt);
#else
  return false;
#endif
}

// CPU publication invalidates the waiter's exclusive monitor. A bounded wait
// also guarantees progress if an event is missed; callers always recheck state.
void reader_wait_change(uint32_t* sequence, uint32_t previous) {
#if defined(__aarch64__) && defined(__linux__)
  uint64_t now, frequency;
  __asm__ __volatile__("isb\n\tmrs %0, cntvct_el0" : "=r"(now) :: "memory");
  __asm__ __volatile__("mrs %0, cntfrq_el0" : "=r"(frequency));
  const uint64_t deadline = now + std::max<uint64_t>(1, frequency / 1000);
  uint32_t changed;
  __asm__ __volatile__("mov x9, %[deadline]\n\t"
                       "sevl\n\twfe\n\t"
                       "ldxr %w[changed], %[sequence]\n\t"
                       "eor %w[changed], %w[changed], %w[previous]\n\t"
                       "cbnz %w[changed], 1f\n\t"
                       ".inst 0xd5031009\n\t"  // wfet x9; runtime HWCAP gated
                       "1:\n\tclrex"
                       : [changed] "=&r"(changed), [sequence] "+Q"(*sequence)
                       : [previous] "r"(previous), [deadline] "r"(deadline)
                       : "x9", "memory");
  (void)changed;
#else
  (void)sequence;
  (void)previous;
#endif
}

void reader_pause() {
#if defined(__i386__) || defined(__x86_64__)
  __asm__ __volatile__("pause");
#elif defined(__aarch64__) || defined(__arm__)
  __asm__ __volatile__("yield");
#else
  std::this_thread::yield();
#endif
}

bool compactible_scales(std::span<const uint8_t> data, const ChunkLayout& layout) {
  for (const auto* matrix : {&layout.gate, &layout.up, &layout.down}) {
    if (matrix->scales_offset > data.size() || matrix->scales_bytes > data.size() - matrix->scales_offset)
      throw std::logic_error("expert read queue: scale span outside record");
    const auto scales = data.subspan(matrix->scales_offset, matrix->scales_bytes);
    uint64_t invalid = 0;
    size_t i = 0;
    for (; i + sizeof(uint64_t) <= scales.size(); i += sizeof(uint64_t)) {
      uint64_t word;
      std::memcpy(&word, scales.data() + i, sizeof(word));
      invalid |= word;
    }
    if (invalid & UINT64_C(0x8787878787878787)) return false;
    for (; i < scales.size(); ++i) if (scales[i] & 0x87) return false;
  }
  return true;
}
}  // namespace
struct ReadPool {
  std::mutex mutex;
  std::condition_variable available;
  std::vector<std::unique_ptr<void, decltype(&std::free)>> buffers;
  std::vector<size_t> free;
  size_t slot_bytes = 0;
  uint64_t used = 0, peak = 0;
};

ReadLease::ReadLease(std::shared_ptr<ReadPool> pool, size_t slot, size_t size)
    : pool_(std::move(pool)), slot_(slot), size_(size) {}
ReadLease::~ReadLease() {
  if (!active_) return;
  {
    std::lock_guard lock(pool_->mutex);
    pool_->free.push_back(slot_);  // capacity reserved at startup
    pool_->used -= pool_->slot_bytes;
  }
  pool_->available.notify_one();
}
std::span<const uint8_t> ReadLease::bytes() const {
  return {static_cast<const uint8_t*>(pool_->buffers[slot_].get()), size_};
}
ReadTicket::ReadTicket() : result_(completion_.get_future().share()) {}
bool ReadTicket::ready() const { return wait_for(std::chrono::milliseconds{0}); }
bool ReadTicket::wait_for(std::chrono::milliseconds timeout) const {
  return result_.wait_for(timeout) == std::future_status::ready;
}
std::shared_ptr<const ReadLease> ReadTicket::wait() const { return result_.get(); }

struct ReadQueueImpl {
  struct Job {
    size_t record;
    ReadPriority priority;
    bool started = false;
    std::shared_ptr<ReadTicket> ticket;
  };
  std::shared_ptr<const ExpertStore> store;
  ReadQueueConfig config;
  std::shared_ptr<ReadPool> pool = std::make_shared<ReadPool>();
  mutable std::mutex mutex;
  std::mutex shutdown_mutex;
  std::condition_variable work, completed;
  std::atomic<bool> stopping{false};
  std::atomic<bool> demand_window_active{false};
  // Writers hold mutex; the proactive reader polls without taking it.
  std::atomic<size_t> queued{0};
  // Keep unrelated counters off the monitored cache line. Only GNU atomic
  // operations and the exclusive-load wait above access this word.
  alignas(128) uint32_t wake_sequence = 0;
  alignas(128) const bool timed_reader_wait = timed_reader_wait_available();
  std::deque<std::shared_ptr<Job>> demand, prefetch;
  std::unordered_map<size_t, std::weak_ptr<Job>> jobs;
  std::unordered_map<size_t, std::weak_ptr<ReadTicket>> tickets;
  ReadQueueStats counters;
  std::vector<std::thread> workers;

  ReadQueueImpl(std::shared_ptr<const ExpertStore> source, ReadQueueConfig cfg)
      : store(std::move(source)), config(cfg) {
    if (!store || !cfg.buffers || !cfg.workers || !cfg.max_pending ||
        cfg.workers > cfg.max_pending || cfg.workers > cfg.buffers)
      throw std::invalid_argument("expert read queue: invalid worker/buffer capacity");
    for (const auto& r : store->records()) {
      if (r.layout.record_bytes > std::numeric_limits<size_t>::max())
        throw std::overflow_error("expert read queue: record exceeds host address space");
      pool->slot_bytes = std::max(pool->slot_bytes, static_cast<size_t>(r.layout.record_bytes));
    }
    if (!pool->slot_bytes || cfg.buffers > cfg.staging_budget_bytes / pool->slot_bytes)
      throw std::invalid_argument("expert read queue: staging reservation exceeds budget");
    counters.staging_reserved = cfg.buffers * pool->slot_bytes;
    pool->buffers.reserve(cfg.buffers);
    pool->free.reserve(cfg.buffers);
    for (size_t b = 0; b < cfg.buffers; ++b) {
      void* data = nullptr;
      if (::posix_memalign(&data, 4096, pool->slot_bytes) != 0)
        throw std::bad_alloc();
      pool->buffers.emplace_back(data, &std::free);
      pool->free.push_back(b);
    }
    workers.reserve(cfg.workers);
    try {
      for (size_t i = 0; i < cfg.workers; ++i) workers.emplace_back([this, i] { run(i); });
    } catch (...) {
      shutdown();
      throw;
    }
  }
  ~ReadQueueImpl() { shutdown(); }

  static std::exception_ptr cancelled() {
    return std::make_exception_ptr(std::runtime_error("expert read queue: cancelled"));
  }
  uint64_t bytes(const Job& job) const { return store->records()[job.record].layout.record_bytes; }

  void retire_locked(const std::shared_ptr<Job>& job) {
    --counters.pending;
    if (job->started) --counters.active;
    if (job->priority == ReadPriority::kPrefetch) counters.speculative_bytes -= bytes(*job);
    jobs.erase(job->record);
  }
  void cancel_queued_locked(const std::shared_ptr<Job>& job) {
    --queued;
    retire_locked(job);
    tickets.erase(job->record);
    ++counters.cancelled;
    job->ticket->completion_.set_exception(cancelled());
    completed.notify_all();
  }
  std::shared_ptr<ReadTicket> request(size_t record, ReadPriority priority) {
    if (record >= store->records().size() ||
        (priority != ReadPriority::kDemand && priority != ReadPriority::kPrefetch))
      throw std::invalid_argument("expert read queue: invalid record or priority");
    std::lock_guard lock(mutex);
    if (stopping) throw std::runtime_error("expert read queue: shut down");
    if (auto ticket = tickets[record].lock()) {
      if (auto job = jobs[record].lock(); job && priority == ReadPriority::kDemand && job->priority == ReadPriority::kPrefetch) {
        if (!job->started) {
          const auto it = std::find(prefetch.begin(), prefetch.end(), job);
          if (it == prefetch.end()) throw std::logic_error("expert read queue: lost speculative job");
          demand.push_back(job);  // allocate before erasing the old owner
          prefetch.erase(it);
        }
        counters.speculative_bytes -= bytes(*job);
        job->priority = ReadPriority::kDemand;
        ++counters.promoted;
      }
      ++counters.coalesced;
      return ticket;
    }
    const uint64_t size = store->records()[record].layout.record_bytes;
    if (priority == ReadPriority::kPrefetch &&
        size > config.prefetch_bytes - counters.speculative_bytes) {
      ++counters.refused;
      return nullptr;
    }
    if (counters.pending >= config.max_pending) {
      if (priority == ReadPriority::kDemand && !prefetch.empty()) {
        const auto victim = prefetch.back();
        prefetch.pop_back();
        cancel_queued_locked(victim);
      } else {
        ++counters.refused;
        return nullptr;
      }
    }
    auto ticket = std::shared_ptr<ReadTicket>(new ReadTicket());
    auto job = std::make_shared<Job>(Job{record, priority, false, ticket});
    // Queue/map insertions may allocate; roll back partial insertion on error.
    auto& queue = priority == ReadPriority::kDemand ? demand : prefetch;
    try {
      tickets[record] = ticket;
      jobs[record] = job;
      queue.push_back(job);
    } catch (...) {
      tickets.erase(record);
      jobs.erase(record);
      throw;
    }
    ++counters.accepted;
    ++queued;
    __atomic_add_fetch(&wake_sequence, 1u, __ATOMIC_RELEASE);
    ++counters.pending;
    if (priority == ReadPriority::kPrefetch) counters.speculative_bytes += size;
    work.notify_one();
    return ticket;
  }

  std::shared_ptr<ReadLease> acquire(size_t record) {
    std::unique_lock lock(pool->mutex);
    pool->available.wait(lock, [&] { return stopping || !pool->free.empty(); });
    if (stopping) std::rethrow_exception(cancelled());
    const auto slot = pool->free.back();
    // Allocate the lease before removing its slot: failure leaves the pool intact.
    auto lease = std::shared_ptr<ReadLease>(new ReadLease(pool, slot, store->records()[record].layout.record_bytes));
    pool->free.pop_back();
    pool->used += pool->slot_bytes;
    pool->peak = std::max(pool->peak, pool->used);
    lease->active_ = true;
    return lease;
  }
  void set_demand_window(bool active) {
    {
      std::lock_guard lock(mutex);
      demand_window_active.store(active);
      __atomic_add_fetch(&wake_sequence, 1u, __ATOMIC_RELEASE);
    }
    if (active) work.notify_all();
  }
  void run(size_t index) {
    for (;;) {
      if (index == 0) {
        while (demand_window_active.load() && !stopping.load() && queued.load() == 0) {
          if (timed_reader_wait) {
            const uint32_t previous = __atomic_load_n(&wake_sequence, __ATOMIC_ACQUIRE);
            if (demand_window_active.load() && !stopping.load() && queued.load() == 0)
              reader_wait_change(&wake_sequence, previous);
          } else {
            reader_pause();
          }
        }
      }
      if (stopping.load()) return;
      std::shared_ptr<Job> job;
      {
        std::unique_lock lock(mutex);
        work.wait(lock, [&] {
          return stopping || !demand.empty() || !prefetch.empty() ||
                 (index == 0 && demand_window_active.load());
        });
        if (stopping) return;
        // Arming an empty queue wakes worker 0 into the lock-free poll.
        // Another worker may also have taken the work observed by the poll.
        if (demand.empty() && prefetch.empty()) continue;
        auto& queue = demand.empty() ? prefetch : demand;
        job = std::move(queue.front());
        queue.pop_front();
        --queued;
        job->started = true;
        ++counters.active;
      }
      std::shared_ptr<ReadLease> lease;
      std::exception_ptr failure;
      bool abandoned = job->ticket.use_count() == 1;
      try {
        if (abandoned) std::rethrow_exception(cancelled());
        lease = acquire(job->record);
        abandoned = job->ticket.use_count() == 1;
        if (abandoned) std::rethrow_exception(cancelled());
        auto buffer = std::span<uint8_t>(static_cast<uint8_t*>(pool->buffers[lease->slot_].get()), lease->size_);
        store->read(job->record, buffer);
        if (config.scan_nvfp4_scales && store->identity().format == Format::kNvfp4)
          lease->scale_compactibility_ = compactible_scales(buffer, store->records()[job->record].layout)
              ? ScaleCompactibility::kExact : ScaleCompactibility::kUnsupported;
      } catch (...) { failure = std::current_exception(); }
      {
        std::lock_guard lock(mutex);
        retire_locked(job);
        if (stopping || abandoned) {
          failure = cancelled();
          ++counters.cancelled;
        } else if (failure) {
          ++counters.failed;
        } else {
          ++counters.completed;
        }
        // On failure, don't make future callers coalesce into a permanently
        // failed ticket. Existing consumers still receive this same failure.
        if (failure) tickets.erase(job->record);
        if (failure) job->ticket->completion_.set_exception(failure);
        else job->ticket->completion_.set_value(lease);
        completed.notify_all();
      }
    }
  }
  void shutdown() {
    std::lock_guard shutdown_lock(shutdown_mutex);
    {
      std::lock_guard lock(mutex);
      if (!stopping.exchange(true)) {
        for (const auto& job : demand) cancel_queued_locked(job);
        for (const auto& job : prefetch) cancel_queued_locked(job);
        demand.clear();
        prefetch.clear();
        queued.store(0);
        __atomic_add_fetch(&wake_sequence, 1u, __ATOMIC_RELEASE);
      }
    }
    work.notify_all();
    pool->available.notify_all();
    for (auto& worker : workers) if (worker.joinable()) worker.join();
  }
  ReadQueueStats stats() const {
    std::lock_guard lock(mutex);
    auto result = counters;
    std::lock_guard pool_lock(pool->mutex);
    result.staging_in_use = pool->used;
    result.staging_peak = pool->peak;
    return result;
  }
};

ExpertReadQueue::ExpertReadQueue(std::shared_ptr<const ExpertStore> store, ReadQueueConfig cfg)
    : impl_(std::make_unique<ReadQueueImpl>(std::move(store), cfg)) {}
ExpertReadQueue::~ExpertReadQueue() = default;
std::shared_ptr<ReadTicket> ExpertReadQueue::request(size_t record, ReadPriority priority) { return impl_->request(record, priority); }
size_t ExpertReadQueue::wait_any(std::span<const std::shared_ptr<ReadTicket>> tickets) const {
  if (tickets.empty() || std::any_of(tickets.begin(), tickets.end(), [](const auto& t) { return !t; }))
    throw std::invalid_argument("expert read queue: wait_any requires nonempty valid tickets");
  std::unique_lock lock(impl_->mutex);
  size_t ready = tickets.size();
  impl_->completed.wait(lock, [&] {
    for (size_t i = 0; i < tickets.size(); ++i) if (tickets[i]->ready()) { ready = i; return true; }
    return false;
  });
  return ready;
}
ReadQueueStats ExpertReadQueue::stats() const { return impl_->stats(); }
void ExpertReadQueue::set_demand_window(bool active) { impl_->set_demand_window(active); }
void ExpertReadQueue::shutdown() { impl_->shutdown(); }
std::vector<std::span<uint8_t>> ExpertReadQueue::staging_spans() const {
  std::vector<std::span<uint8_t>> result;
  for (const auto& b : impl_->pool->buffers)
    result.emplace_back(static_cast<uint8_t*>(b.get()), impl_->pool->slot_bytes);
  return result;
}
}  // namespace dgpp::expert
