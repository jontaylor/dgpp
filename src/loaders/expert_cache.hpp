#pragma once
// Engine-thread cache ownership. File workers never call the storage backend.
// Both native cache bindings and reconstructed slots use existing full-expert
// kernels. Lease retirement protects payload readers through backend ordering.
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "loaders/expert_read_queue.hpp"
#include "loaders/expert_pressure.hpp"

namespace dgpp::expert {
struct CacheConfig {
  bool operator==(const CacheConfig&) const = default;
  PlannerConfig planner;
  ReadQueueConfig reads;
  size_t execution_slots = 1;
  // Internal execution layout, selected before memory preflight. The second
  // half of each execution slot is charged conversion space for miss waves.
  bool packed_nvfp4 = false;
};
struct PagingOptions {
  bool operator==(const PagingOptions&) const = default;
  std::string image_directory;
  CacheConfig cache;
  uint64_t chunk_intermediate = 64;
  uint64_t adapt_every_routes = 1024;
  IoMode io_mode = IoMode::kDirect;
};
struct CacheMemory {
  uint64_t budget_bytes = 0, cache_slot_bytes = 0, cache_slots = 0;
  uint64_t execution_slot_bytes = 0, execution_slots = 0;
  uint64_t staging_bytes = 0, unallocated_bytes = 0;
  uint64_t allocated_bytes() const;
  bool operator==(const CacheMemory&) const = default;
};
// Shared preflight formula, before allocating device or staging slabs.
CacheMemory expert_cache_memory(const StoreIdentity&, const CacheConfig&);

struct CachedExpert {
  uint32_t layer = 0, expert = 0;
  bool draft = false;
  std::vector<size_t> records;  // ordered intermediate chunks, complete TP slice
};

// The backend must allocate exactly the reported slabs at startup, must not
// allocate payload/staging scratch during calls, and must fence all submitted
// reads/copies/kernels in finish(). The caller owns its execution stream.
class ExpertCacheStorage {
 public:
  virtual ~ExpertCacheStorage() = default;
  virtual const CacheMemory& memory() const = 0;
  virtual void retain(size_t slot, std::span<const uint8_t> bytes) = 0;
  // Every successful reconstruction assembles all chunks before publication.
  // Backends may leave scratch untouched when their consumers read only the
  // fully overwritten tensor ranges, rather than allocation padding.
  virtual void begin_execution(size_t slot) = 0;
  virtual uint64_t execution_zero_bytes_per_begin() const { return memory().execution_slot_bytes; }
  virtual void assemble(size_t execution_slot, const StoreRecord& record,
      std::optional<size_t> cache_slot, std::span<const uint8_t> host_bytes) = 0;
  virtual const uint8_t* execution_data(size_t slot) const = 0;
  virtual const uint8_t* packed_execution_data(size_t) {
    throw std::logic_error("packed NVFP4 execution unsupported");
  }
  // Optional native-layout reuse within the SAME cache slab. These operations
  // use a caller-owned execution slot as bounded packing/unpacking scratch.
  // Returning null disables native reuse for reference backends.
  virtual const uint8_t* cache_data(size_t) const { return nullptr; }
  virtual void move_cache_slot(size_t, size_t) { throw std::logic_error("native cache unsupported"); }
  virtual void native_to_execution(size_t, size_t) { throw std::logic_error("native cache unsupported"); }
  virtual void execution_to_native(size_t, size_t) { throw std::logic_error("native cache unsupported"); }
  virtual void unpack_native_chunk(size_t, const StoreRecord&, size_t) { throw std::logic_error("native cache unsupported"); }
  virtual bool accepts_execution_context(const void*) const { return false; }
  virtual uint64_t completion_wait_ns() const { return 0; }
  // Caller must quiesce all payload readers before switching. A paused GPU
  // demand gate is quiescent: it reads metadata only, never expert payloads.
  virtual void set_execution_context(void*) { throw std::logic_error("cache stream switching unsupported"); }
  // Retire native readers. A backend with one ordered execution/transfer
  // stream may release host leases without draining it: all future writes
  // are ordered after these readers. Other backends retain the full fence.
  virtual void retire_native() { finish(); }
  // The same ordering rule applies to reconstruction scratch. Staging leases
  // have already been drained before acquire() returns; only device readers
  // remain. A backend may reuse the slot through ordered GPU submissions.
  virtual void retire_execution() { finish(); }
  virtual void finish() = 0;
};
using CacheStorageFactory = std::function<std::unique_ptr<ExpertCacheStorage>(
    const CacheMemory&, std::span<const std::span<uint8_t>>)>;

struct ExpertCacheState;
class ExpertExecutionLease {
 public:
  ExpertExecutionLease() = default;
  ~ExpertExecutionLease();
  ExpertExecutionLease(ExpertExecutionLease&&) noexcept;
  ExpertExecutionLease& operator=(ExpertExecutionLease&&) noexcept;
  ExpertExecutionLease(const ExpertExecutionLease&) = delete;
  ExpertExecutionLease& operator=(const ExpertExecutionLease&) = delete;
  const uint8_t* data() const;
  // Prepared native hits already have this layout; reconstructed misses are
  // converted into their separately charged execution-slot region.
  const uint8_t* packed_data() const;
  // Retire readers before reuse, by stream ordering or a backend fence.
  // Explicit release reports failures; destructor failures poison the cache.
  void release();
  explicit operator bool() const { return state_ != nullptr; }
 private:
  friend class ExpertCache;
  ExpertExecutionLease(std::shared_ptr<ExpertCacheState>, size_t);
  ExpertExecutionLease(std::shared_ptr<ExpertCacheState>, size_t, const uint8_t*);
  void release_noexcept() noexcept;
  std::shared_ptr<ExpertCacheState> state_;
  size_t slot_ = 0;
  const uint8_t* native_data_ = nullptr;
};
struct CacheStats {
  uint64_t requests = 0, full_hits = 0, partial_hits = 0, chunk_lookups = 0, chunk_hits = 0;
  uint64_t demand_bytes = 0, retained_bytes = 0;
  uint64_t read_wait_ns = 0;  // engine wait on file tickets; not device I/O time
  // Successful reconstruction only. Wall time excludes the file-ticket waits
  // above, but includes engine CPU work and backend completion fences.
  uint64_t reconstruction_wall_ns = 0;
  uint64_t cached_copy_bytes = 0, staged_copy_bytes = 0, execution_zero_bytes = 0;
  uint64_t native_hits = 0, native_packs = 0, native_unpacks = 0;
  uint64_t direct_native_hits = 0, canonical_unpacks = 0, canonical_unpack_bytes = 0;
  uint64_t canonical_unpack_enqueue_ns = 0, completion_wait_ns = 0;
  uint64_t native_layout_wall_ns = 0, native_layout_copy_bytes = 0;
  uint64_t active_native_leases = 0;
  uint64_t promotions = 0, demotions = 0, evictions = 0;
  uint64_t promotion_bytes = 0, demotion_bytes = 0;
  uint64_t demand_admissions = 0, demand_admission_bytes = 0;
  uint64_t active_execution_slots = 0, peak_execution_slots = 0;
  uint64_t backpressure = 0;
  uint64_t adaptations_deferred = 0, prefetches_deferred = 0;
  bool initialized = false, poisoned = false;
};
struct CacheTelemetry {
  Policy policy = Policy::kWholeExpert;
  IoMode io_mode = IoMode::kBuffered;
  CacheStats cache;
  ReadQueueStats queue;
  ReadStats file;
  CacheMemory memory;
  uint64_t resident_bytes = 0, resident_chunks = 0, total_chunks = 0;
  ResourcePressure pressure;
};

class ExpertCache {
 public:
  // Validates the complete population/floor/budget before invoking factory.
  // Cache slots charge the largest chunk allocation even for smaller tails.
  ExpertCache(std::shared_ptr<const ExpertStore>, CacheConfig, CacheStorageFactory,
              ResourcePressureSampler = {});
  ~ExpertCache();
  ExpertCache(const ExpertCache&) = delete;
  ExpertCache& operator=(const ExpertCache&) = delete;
  const CacheMemory& memory() const;
  uint64_t binding_identity() const;
  const ChunkLayout& execution_layout() const;
  const std::vector<CachedExpert>& experts() const;
  // Completes mandatory fractional population before making any expert usable.
  void initialize();
  void observe(std::span<const size_t> selected);
  // Transitions outside running kernels. A failed transition poisons the cache;
  // a partially loaded/evicted population can never be executed or committed.
  void adapt();
  const ResidencyPlan& residency() const;
  uint64_t observations() const;
  // nullptr-style empty lease means bounded execution-slot backpressure.
  // Demand reads wait for completion, using queued prefetch tickets if present.
  ExpertExecutionLease acquire(size_t expert, bool canonical = false);
  // Reconstruct bounded waves, consuming file completions independently.
  // Callback indices address selected; leases stay valid through the callback.
  // Callback only enqueues consumers on the cache stream: it must not release
  // leases, mutate/reenter the cache, switch streams, or retain their addresses.
  // Admission follows execution submission and uses the reconstructed payload.
  void execute_ready(std::span<const size_t> selected,
      const std::function<void(std::span<const size_t>, std::span<const ExpertExecutionLease>)>& execute);
  bool packed_nvfp4() const;
  // Borrowed metadata, not a lease. Only usable by a quiescent engine-owned
  // walk that stops payload readers before any cache mutation. A revision
  // change invalidates the published table; null means not prepared.
  const uint8_t* prepared_address(size_t expert) const;
  uint64_t prepared_revision() const;
  void set_execution_context(void* context);
  // Account for GPU-executed native hits after the walk has completed.
  void record_native_hits(std::span<const size_t> selected);
  // Prepare a complete routed set before pinning any members. Returns false
  // without changing residency when a member is partial or backend is compact-only.
  bool prepare_resident(std::span<const size_t> experts);
  // Retire a grouped native chain using the backend's reader ordering.
  void release_native_leases(std::span<ExpertExecutionLease> leases);
  // Keeps at most buffers-1 speculative tickets, reserving staging for demand.
  // Returns number accepted/coalesced; never calls the device backend.
  size_t prefetch(size_t expert);
  // Already-routed upcoming work, not speculative background traffic. Uses
  // at most half of the existing staging slots and never changes residency.
  size_t lookahead_demand(size_t expert);
  CacheStats stats() const;
  ReadQueueStats read_stats() const;
  CacheTelemetry telemetry() const;
  void require_execution_context(const void*) const;
  // Explicit, quiescent handoff from model construction to the serving loop.
  // The old owner releases before the new thread binds; implicit stealing is
  // refused. No leases or operations may cross the handoff.
  void release_execution_thread();
  void bind_execution_thread();
 private:
  std::shared_ptr<ExpertCacheState> state_;
};
}  // namespace dgpp::expert
