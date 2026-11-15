#include "loaders/expert_cache.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <deque>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <utility>

namespace dgpp::expert {
namespace {
uint64_t add(uint64_t a, uint64_t b) {
  if (b > UINT64_MAX - a) throw std::overflow_error("expert cache: byte count overflow");
  return a + b;
}
uint64_t mul(uint64_t a, uint64_t b) {
  if (a && b > UINT64_MAX / a) throw std::overflow_error("expert cache: byte count overflow");
  return a * b;
}
std::atomic<uint64_t> next_binding_identity{1};
constexpr size_t absent = std::numeric_limits<size_t>::max();
}
uint64_t CacheMemory::allocated_bytes() const {
  return add(add(mul(cache_slot_bytes, cache_slots), mul(execution_slot_bytes, execution_slots)), staging_bytes);
}

CacheMemory expert_cache_memory(const StoreIdentity& id, const CacheConfig& cfg) {
  if (!cfg.execution_slots || cfg.planner.reserved_bytes)
    throw std::invalid_argument("expert cache: invalid execution reservation");
  const auto chunks = partition(id.hidden, id.intermediate, id.world, id.rank, id.chunk_intermediate, id.format);
  const auto full = partition(id.hidden, id.intermediate, id.world, id.rank, id.intermediate / id.world, id.format).front();
  CacheMemory m;
  m.budget_bytes = cfg.planner.budget_bytes;
  m.execution_slot_bytes = full.allocation_bytes;
  if (cfg.packed_nvfp4) {
    if (id.format != Format::kNvfp4)
      throw std::invalid_argument("expert cache: packed execution requires NVFP4");
    m.execution_slot_bytes = mul(m.execution_slot_bytes, 2);
  }
  m.execution_slots = cfg.execution_slots;
  uint64_t record_bytes = 0;
  for (const auto& c : chunks) {
    m.cache_slot_bytes = std::max(m.cache_slot_bytes, c.allocation_bytes);
    record_bytes = std::max(record_bytes, c.record_bytes);
  }
  m.staging_bytes = mul(record_bytes, cfg.reads.buffers);
  if (!cfg.reads.buffers || !cfg.reads.workers || cfg.reads.workers > cfg.reads.buffers ||
      cfg.reads.workers > cfg.reads.max_pending || cfg.reads.max_pending < cfg.reads.buffers ||
      m.staging_bytes > cfg.reads.staging_budget_bytes)
    throw std::invalid_argument("expert cache: invalid bounded read capacity");
  const uint64_t reserved = add(m.staging_bytes, mul(m.execution_slot_bytes, m.execution_slots));
  if (reserved > m.budget_bytes) throw std::invalid_argument("expert cache: staging/execution exceed budget");
  m.cache_slots = (m.budget_bytes - reserved) / m.cache_slot_bytes;
  m.unallocated_bytes = (m.budget_bytes - reserved) % m.cache_slot_bytes;
  if (m.cache_slots > std::numeric_limits<size_t>::max() ||
      m.allocated_bytes() > std::numeric_limits<size_t>::max())
    throw std::overflow_error("expert cache: reservation exceeds address space");
  return m;
}

struct ExpertCacheState {
  const uint64_t binding_identity = next_binding_identity.fetch_add(1, std::memory_order_relaxed);
  // Destruction order matters: device backend unregisters staging before the
  // read queue frees its aligned pool. Leases retain this whole state.
  std::shared_ptr<const ExpertStore> store;
  std::unique_ptr<ExpertReadQueue> reads;
  std::shared_ptr<const ReadLease> pending_copy;  // retained across a failed device fence
  std::vector<std::shared_ptr<const ReadLease>> pending_copies;
  std::unique_ptr<ExpertCacheStorage> storage;
  std::unique_ptr<ResidencyPlanner> planner;
  CacheConfig config;
  CacheMemory memory;
  ChunkLayout full;
  std::vector<CachedExpert> experts;
  std::vector<size_t> slots, free_slots;
  std::vector<size_t> slot_records, native_start, native_holes, free_positions;
  std::vector<size_t> record_owner;
  std::vector<uint64_t> expert_record_bytes;
  std::vector<bool> execution_busy;
  struct QueuedRead {
    std::shared_ptr<ReadTicket> ticket;
    std::chrono::steady_clock::time_point demanded_at{};
  };
  std::map<size_t, QueuedRead> prefetched;
  CacheStats counters;
  uint64_t prepared_revision = 1;
  bool observed_for_adaptation = false;
  bool all_observed_full = true;
  ResourcePressureSampler resource_sampler;
  ResourcePressure pressure;
  mutable std::mutex owner_mutex;
  std::thread::id owner = std::this_thread::get_id();

  void check_thread() const {
    std::lock_guard lock(owner_mutex);
    if (owner != std::this_thread::get_id())
      throw std::logic_error("expert cache: calls and lease release belong to the engine thread");
  }
  void check(bool initialized = true) const {
    check_thread();
    if (counters.poisoned) throw std::runtime_error("expert cache: poisoned by an earlier failure");
    if (initialized && !counters.initialized) throw std::logic_error("expert cache: not initialized");
  }
  void expert_index(size_t e) const {
    if (e >= experts.size()) throw std::out_of_range("expert cache: expert index");
  }
  bool background_allowed() {
    if (resource_sampler) pressure = resource_sampler();
    return pressure.background_allowed();
  }
  // Only the engine accesses prefetched tickets. Retained completed reads use
  // at most buffers-1 leases; one full staging slot is always left for demand.
  std::shared_ptr<const ReadLease> read(size_t record,
      std::shared_ptr<ReadTicket> queued_ticket = {},
      std::chrono::steady_clock::time_point queued = {}) {
    const auto started = std::chrono::steady_clock::now();
    const bool ahead = static_cast<bool>(queued_ticket);
    if (!ahead) queued = {};
    bool speculative = false;
    std::shared_ptr<ReadTicket> ticket = std::move(queued_ticket);
    if (auto it = prefetched.find(record); it != prefetched.end()) {
      speculative = it->second.demanded_at == std::chrono::steady_clock::time_point{};
      if (!speculative) queued = it->second.demanded_at;
      if (!ticket) ticket = std::move(it->second.ticket);
      prefetched.erase(it);
      // A look-ahead already promoted this ticket. Preserve its original
      // result, including errors, rather than retrying a failed queued read.
      if (!ahead) (void)reads->request(record, ReadPriority::kDemand);
    } else if (!ticket) ticket = reads->request(record, ReadPriority::kDemand);
    if (!ticket) {
      // Pending speculation is ours. Releasing it lets workers cancel queued
      // jobs, but request refusal is still explicit bounded backpressure.
      prefetched.clear();
      ++counters.backpressure;
      throw std::runtime_error("expert cache: read queue capacity exhausted");
    }
    auto lease = ticket->wait();
    const auto completed = std::chrono::steady_clock::now();
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(completed - started).count();
    counters.read_wait_ns += static_cast<uint64_t>(std::max<int64_t>(elapsed, 0));
    if (!speculative) {
      const size_t expert = record_owner[record];
      // A completed look-ahead must not look like zero-cost storage just
      // because its I/O overlapped the previous chunk's device transfer.
      const auto load_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(completed -
          (queued == std::chrono::steady_clock::time_point{} ? started : queued)).count();
      const double full_ms = static_cast<double>(std::max<int64_t>(load_ns, 1)) / 1e6 *
          static_cast<double>(expert_record_bytes[expert]) / store->records()[record].layout.record_bytes;
      planner->observe_load(expert, full_ms);
    }
    return lease;
  }
  void add_free_slot(size_t slot) {
    if (slot_records[slot] != absent || free_positions[slot] != absent)
      throw std::logic_error("expert cache: duplicate or occupied free slot");
    free_positions[slot] = free_slots.size();
    free_slots.push_back(slot);
  }
  void remove_free_slot(size_t slot) {
    const size_t at = free_positions[slot];
    if (at == absent || at >= free_slots.size() || free_slots[at] != slot)
      throw std::logic_error("expert cache: missing free slot index");
    const size_t last = free_slots.back();
    free_slots[at] = last;
    free_positions[last] = at;
    free_slots.pop_back();
    free_positions[slot] = absent;
  }
  void retain(size_t record) {
    if (free_slots.empty()) throw std::logic_error("expert cache: no free slot in validated transition");
    pending_copy = read(record);
    const auto size = store->records()[record].layout.allocation_bytes;
    const auto slot = free_slots.back();
    storage->retain(slot, pending_copy->bytes().first(size));
    storage->finish();  // staging cannot be reused until the copy is complete
    pending_copy.reset();
    remove_free_slot(slot);
    slots[record] = slot;
    slot_records[slot] = record;
    counters.retained_bytes += size;
  }
  void unpack_native(size_t expert, size_t scratch) {
    const auto started = std::chrono::steady_clock::now();
    storage->native_to_execution(scratch, native_start[expert]);
    for (const auto r : experts[expert].records)
      storage->unpack_native_chunk(scratch, store->records()[r], slots[r]);
    storage->finish();
    native_start[expert] = absent;
    ++prepared_revision;
    ++counters.native_unpacks;
    counters.native_layout_copy_bytes += full.allocation_bytes;
    for (const auto r : experts[expert].records)
      counters.native_layout_copy_bytes += store->records()[r].layout.useful_bytes;
    counters.native_layout_wall_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - started).count();
  }
  // One-time layout conversion inside the already allocated cache slab. The
  // demanded expert's compact chunks become holes after gathering to scratch.
  // Foreign occupants of the chosen run move into holes outside that run.
  // No resident payload is duplicated outside the existing execution scratch.
  void pack_native(size_t expert, size_t scratch) {
    const auto started = std::chrono::steady_clock::now();
    const auto nested_before = counters.native_layout_wall_ns;
    const auto& records = experts[expert].records;
    // All experts in an image have the same TP geometry. Aligned runs keep
    // existing native groups stable: packing another full resident cannot
    // repeatedly unpack and rebuild its hot neighbors.
    // A complete expert has more chunks than the incomplete slab tail can
    // hold. At least one of its own chunks therefore belongs to a full run.
    // Native runs occupy complete aligned groups, so that owned run cannot
    // contain a foreign native binding. A maximum-ownership run is among
    // these candidates; zero-ownership runs cannot improve it.
    native_holes.clear();
    for (const auto r : records) {
      if (slots[r] == absent) throw std::logic_error("expert cache: native source chunk absent");
      const size_t begin = slots[r] / records.size() * records.size();
      if (begin + records.size() <= slot_records.size()) native_holes.push_back(begin);
    }
    std::sort(native_holes.begin(), native_holes.end());
    native_holes.erase(std::unique(native_holes.begin(), native_holes.end()), native_holes.end());
    size_t first = absent, best = 0;
    for (const size_t begin : native_holes) {
      size_t own = 0;
      bool blocked = false;
      for (size_t slot = begin; slot < begin + records.size(); ++slot) {
        const auto r = slot_records[slot];
        if (r == absent) continue;
        const auto other = record_owner[r];
        if (native_start[other] != absent) { blocked = true; break; }
        own += other == expert;
      }
      if (!blocked && (first == absent || own > best)) { first = begin; best = own; }
      if (best == records.size()) break;
    }
    if (first == absent) throw std::logic_error("expert cache: no native run for complete resident expert");
    const size_t end = first + records.size();
    for (size_t slot = first; slot < end; ++slot) {
      if (slot_records[slot] == absent) continue;
      const auto other = record_owner[slot_records[slot]];
      if (native_start[other] != absent) unpack_native(other, scratch);
    }
    storage->begin_execution(scratch);
    for (const auto r : records)
      storage->assemble(scratch, store->records()[r], slots[r], {});
    storage->finish();
    native_holes.clear();
    for (const auto r : records) {
      const size_t slot = slots[r];
      slot_records[slot] = absent;
      if (slot < first || slot >= end) native_holes.push_back(slot);
    }
    // Vacating this expert provides enough outside holes for every foreign
    // occupant of the target run, even if the cache was completely full.
    for (size_t slot = first; slot < end; ++slot) {
      const auto r = slot_records[slot];
      if (r == absent) continue;
      if (native_holes.empty()) throw std::logic_error("expert cache: native layout has no relocation hole");
      const auto dest = native_holes.back();
      native_holes.pop_back();
      storage->move_cache_slot(dest, slot);
      counters.native_layout_copy_bytes += memory.cache_slot_bytes;
      slots[r] = dest;
      slot_records[dest] = r;
      slot_records[slot] = absent;
    }
    storage->execution_to_native(scratch, first);
    storage->finish();
    for (size_t c = 0; c < records.size(); ++c) {
      if (free_positions[first + c] != absent) remove_free_slot(first + c);
      slots[records[c]] = first + c;
      slot_records[first + c] = records[c];
    }
    for (const size_t slot : native_holes) add_free_slot(slot);
    native_start[expert] = first;
    ++prepared_revision;
    ++counters.native_packs;
    counters.native_layout_copy_bytes += full.allocation_bytes;
    for (const auto r : records) counters.native_layout_copy_bytes += store->records()[r].layout.useful_bytes;
    const auto wall = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - started).count());
    const auto nested = counters.native_layout_wall_ns - nested_before;
    counters.native_layout_wall_ns += wall > nested ? wall - nested : 0;
  }
  void evict(const ResidencyPlan& plan, size_t scratch = 0) {
    ++prepared_revision;
    const auto& previous = planner->current().chunks;
    for (size_t e = 0; e < experts.size(); ++e) {
      if (plan.chunks[e] < previous[e] && native_start[e] != absent) {
        if (plan.chunks[e]) unpack_native(e, scratch);
        else native_start[e] = absent;
      }
      for (size_t c = plan.chunks[e]; c < previous[e]; ++c) {
        const auto r = experts[e].records[c];
        if (slots[r] == absent) throw std::logic_error("expert cache: missing committed chunk");
        slot_records[slots[r]] = absent;
        add_free_slot(slots[r]);
        slots[r] = absent;
      }
    }
  }
  void count_transition(const ResidencyPlan& plan, const std::vector<size_t>& previous) {
    for (size_t e = 0; e < previous.size(); ++e) {
      if (plan.chunks[e] > previous[e]) ++counters.promotions;
      if (plan.chunks[e] < previous[e]) ++counters.demotions;
      if (previous[e] && !plan.chunks[e]) ++counters.evictions;
    }
    counters.promotion_bytes += plan.load_bytes;
    counters.demotion_bytes += plan.release_bytes;
  }
  void release_execution(size_t slot) {
    check_thread();
    // Backends either fence readers or guarantee that all subsequent writes
    // to this scratch slot are ordered after them on the execution stream.
    try { storage->retire_execution(); }
    catch (...) { counters.poisoned = true; throw; }
    execution_busy[slot] = false;
    --counters.active_execution_slots;
  }
  void release_native() {
    check_thread();
    try { storage->retire_native(); }
    catch (...) { counters.poisoned = true; throw; }
    --counters.active_native_leases;
  }
};

ExpertCache::ExpertCache(std::shared_ptr<const ExpertStore> source, CacheConfig cfg, CacheStorageFactory factory,
                        ResourcePressureSampler resource_sampler)
    : state_(std::make_shared<ExpertCacheState>()) {
  auto& s = *state_;
  if (!source || !factory || !cfg.execution_slots || cfg.planner.reserved_bytes)
    throw std::invalid_argument("expert cache: invalid source, factory or execution reservation");
  s.store = std::move(source);
  s.config = cfg;
  s.resource_sampler = std::move(resource_sampler);
  const auto& id = s.store->identity();
  const auto chunks = partition(id.hidden, id.intermediate, id.world, id.rank, id.chunk_intermediate, id.format);
  s.full = partition(id.hidden, id.intermediate, id.world, id.rank, id.intermediate / id.world, id.format).front();
  s.memory = expert_cache_memory(id, cfg);
  const auto& m = s.memory;
  const uint64_t reserved = add(m.staging_bytes, mul(m.execution_slot_bytes, m.execution_slots));
  std::map<std::tuple<bool, uint32_t, uint32_t>, size_t> groups;
  s.slots.assign(s.store->records().size(), absent);
  s.record_owner.resize(s.store->records().size());
  for (size_t r = 0; r < s.store->records().size(); ++r) {
    const auto& key = s.store->records()[r].key;
    auto [it, inserted] = groups.emplace(std::tuple{key.draft, key.layer, key.expert}, s.experts.size());
    if (inserted) s.experts.push_back({key.layer, key.expert, key.draft, std::vector<size_t>(chunks.size(), absent)});
    auto& records = s.experts[it->second].records;
    if (key.chunk >= records.size() || records[key.chunk] != absent)
      throw std::invalid_argument("expert cache: invalid chunk population");
    records[key.chunk] = r;
    s.record_owner[r] = it->second;
  }
  std::vector<ExpertSpec> specs;
  for (const auto& e : s.experts) {
    if (std::find(e.records.begin(), e.records.end(), absent) != e.records.end())
      throw std::invalid_argument("expert cache: incomplete TP expert population");
    specs.push_back({std::vector<uint64_t>(chunks.size(), m.cache_slot_bytes), 1.0});
    uint64_t bytes = 0;
    for (const auto r : e.records) bytes = add(bytes, s.store->records()[r].layout.record_bytes);
    s.expert_record_bytes.push_back(bytes);
  }
  cfg.planner.reserved_bytes = add(reserved, m.unallocated_bytes);
  s.planner = std::make_unique<ResidencyPlanner>(cfg.planner, std::move(specs));
  s.free_slots.reserve(static_cast<size_t>(m.cache_slots));
  s.slot_records.assign(static_cast<size_t>(m.cache_slots), absent);
  s.free_positions.assign(static_cast<size_t>(m.cache_slots), absent);
  s.native_holes.reserve(chunks.size());
  s.native_start.assign(s.experts.size(), absent);
  for (size_t slot = 0; slot < m.cache_slots; ++slot) s.add_free_slot(slot);
  s.execution_busy.resize(cfg.execution_slots, false);
  s.pending_copies.reserve(cfg.reads.buffers);
  s.reads = std::make_unique<ExpertReadQueue>(s.store, cfg.reads);
  auto spans = s.reads->staging_spans();
  s.storage = factory(m, spans);
  if (!s.storage || s.storage->memory() != m)
    throw std::invalid_argument("expert cache: backend reservation disagrees with the memory plan");
}
ExpertCache::~ExpertCache() = default;
const CacheMemory& ExpertCache::memory() const { return state_->memory; }
uint64_t ExpertCache::binding_identity() const { return state_->binding_identity; }
const ChunkLayout& ExpertCache::execution_layout() const { return state_->full; }
const std::vector<CachedExpert>& ExpertCache::experts() const { return state_->experts; }
void ExpertCache::initialize() {
  auto& s = *state_;
  s.check(false);
  if (s.counters.initialized) return;
  try {
    const auto& counts = s.planner->current().chunks;
    for (size_t e = 0; e < s.experts.size(); ++e)
      for (size_t c = 0; c < counts[e]; ++c) s.retain(s.experts[e].records[c]);
    s.counters.initialized = true;
  } catch (...) { s.counters.poisoned = true; throw; }
}
void ExpertCache::observe(std::span<const size_t> selected) {
  auto& s = *state_;
  s.check();
  s.planner->observe(selected);
  s.observed_for_adaptation |= !selected.empty();
  for (const auto e : selected)
    s.all_observed_full &= s.planner->current().chunks[e] == s.experts[e].records.size();
}
const ResidencyPlan& ExpertCache::residency() const { state_->check(); return state_->planner->current(); }
uint64_t ExpertCache::observations() const { state_->check(); return state_->planner->observations(); }
void ExpertCache::adapt() {
  auto& s = *state_;
  s.check();
  // Execution slots contain their own complete weights, but drain kernels
  // before making any backend transfer (the fabric's safe boundary contract).
  if (s.counters.active_execution_slots || s.counters.active_native_leases)
    throw std::logic_error("expert cache: release execution leases before adapting");
  if (!s.background_allowed()) { ++s.counters.adaptations_deferred; return; }
  // Full coverage of every observed route already has zero missing-weight
  // cost. Keep collecting frequency/recency, but defer population-wide
  // rescoring until a partial expert is used. Explicit idle adaptation (no
  // observations) retains its existing behavior.
  if (s.observed_for_adaptation && s.all_observed_full) {
    s.observed_for_adaptation = false;
    return;
  }
  const auto plan = s.planner->propose();
  const auto previous = s.planner->current().chunks;
  try {
    s.storage->finish();
    s.evict(plan);
    for (size_t e = 0; e < s.experts.size(); ++e)
      for (size_t c = previous[e]; c < plan.chunks[e]; ++c) s.retain(s.experts[e].records[c]);
    s.planner->commit(plan);  // publication only after all transfers succeed
    s.count_transition(plan, previous);
    s.observed_for_adaptation = false;
    s.all_observed_full = true;
  } catch (...) { s.counters.poisoned = true; throw; }
}
bool ExpertCache::packed_nvfp4() const { return state_->config.packed_nvfp4; }
const uint8_t* ExpertCache::prepared_address(size_t expert) const {
  const auto& s = *state_;
  s.check(); s.expert_index(expert);
  return s.native_start[expert] == absent ? nullptr : s.storage->cache_data(s.native_start[expert]);
}
uint64_t ExpertCache::prepared_revision() const { state_->check(); return state_->prepared_revision; }
void ExpertCache::set_execution_context(void* context) {
  auto& s = *state_; s.check();
  if (s.counters.active_execution_slots || s.counters.active_native_leases)
    throw std::logic_error("expert cache: stream handoff with live leases");
  s.storage->set_execution_context(context);
}
void ExpertCache::record_native_hits(std::span<const size_t> selected) {
  auto& s = *state_; s.check();
  observe(selected);
  std::vector<size_t> unique(selected.begin(), selected.end());
  std::sort(unique.begin(), unique.end());
  unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
  for (auto e : unique) {
    s.expert_index(e);
    ++s.counters.requests; ++s.counters.full_hits; ++s.counters.native_hits;
    ++s.counters.direct_native_hits;
    s.counters.chunk_lookups += s.experts[e].records.size();
    s.counters.chunk_hits += s.experts[e].records.size();
  }
}
ExpertExecutionLease ExpertCache::acquire(size_t expert, bool canonical) {
  auto& s = *state_;
  s.check();
  s.expert_index(expert);
  const auto& e = s.experts[expert];
  if (s.native_start[expert] != absent) {
    ++s.counters.requests;
    ++s.counters.full_hits;
    s.counters.chunk_lookups += e.records.size();
    s.counters.chunk_hits += e.records.size();
    ++s.counters.native_hits;
    if (canonical && s.config.packed_nvfp4) {
      const auto it = std::find(s.execution_busy.begin(), s.execution_busy.end(), false);
      if (it == s.execution_busy.end()) { ++s.counters.backpressure; return {}; }
      const size_t slot = static_cast<size_t>(it - s.execution_busy.begin());
      const auto started = std::chrono::steady_clock::now();
      try { s.storage->native_to_execution(slot, s.native_start[expert]); }
      catch (...) { s.counters.poisoned = true; throw; }
      ++s.counters.canonical_unpacks;
      s.counters.canonical_unpack_bytes += s.full.useful_bytes;
      s.counters.canonical_unpack_enqueue_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - started).count();
      s.execution_busy[slot] = true;
      ++s.counters.active_execution_slots;
      s.counters.peak_execution_slots = std::max(s.counters.peak_execution_slots, s.counters.active_execution_slots);
      return ExpertExecutionLease(state_, slot);
    }
    ++s.counters.direct_native_hits;
    ++s.counters.active_native_leases;
    return ExpertExecutionLease(state_, expert, s.storage->cache_data(s.native_start[expert]));
  }
  const auto it = std::find(s.execution_busy.begin(), s.execution_busy.end(), false);
  if (it == s.execution_busy.end()) { ++s.counters.backpressure; return {}; }
  const size_t slot = static_cast<size_t>(it - s.execution_busy.begin());
  size_t hits = 0;
  for (const auto r : e.records) hits += s.slots[r] != absent;
  if (hits < e.records.size()) s.all_observed_full = false;
  ++s.counters.requests;
  s.counters.chunk_lookups += e.records.size();
  s.counters.chunk_hits += hits;
  if (hits == e.records.size()) ++s.counters.full_hits;
  else if (hits) ++s.counters.partial_hits;
  const auto reconstruction_started = std::chrono::steady_clock::now();
  const auto wait_before = s.counters.read_wait_ns;
  uint64_t cached_bytes = 0, staged_bytes = 0;
  try {
    if (hits == e.records.size() && !s.counters.active_native_leases && s.storage->cache_data(0) &&
        !(canonical && s.config.packed_nvfp4)) {
      s.pack_native(expert, slot);
      ++s.counters.native_hits;
      ++s.counters.direct_native_hits;
      ++s.counters.active_native_leases;
      return ExpertExecutionLease(state_, expert, s.storage->cache_data(s.native_start[expert]));
    }
    std::optional<ResidencyPlan> admission;
    std::vector<size_t> previous;
    if (hits < e.records.size() && !s.counters.active_native_leases) {
      auto plan = s.planner->propose_admission(expert);
      if (plan.chunks[expert] > s.planner->current().chunks[expert]) {
        previous = s.planner->current().chunks;
        admission = std::move(plan);
        // Earlier executions own independent reconstruction slots. Drain any
        // pending copies before reassigning persistent cache slots.
        s.storage->finish();
        s.evict(*admission, slot);
      }
    }
    s.storage->begin_execution(slot);
    // Keep a bounded window of missing chunks in flight, including the first
    // read. Ready chunks can share one device fence; their leases remain live
    // until that fence succeeds. Drain before waiting on an unfinished read
    // or refilling a window whose buffers are all held by device readers.
    struct PendingRead {
      size_t chunk;
      std::shared_ptr<ReadTicket> ticket;
      std::chrono::steady_clock::time_point started;
    };
    std::deque<PendingRead> pending_reads;
    size_t next_chunk = 0;
    const auto drain_copies = [&] {
      s.storage->finish();
      s.pending_copies.clear();
    };
    const auto fill_window = [&] {
      const size_t reserved = s.prefetched.size() + s.pending_copies.size();
      const size_t ahead = s.config.reads.buffers > reserved ? s.config.reads.buffers - reserved : 0;
      size_t fresh = 0;
      for (const auto& pending : pending_reads)
        fresh += !s.prefetched.contains(e.records[pending.chunk]);
      while (next_chunk < e.records.size()) {
        if (s.slots[e.records[next_chunk]] != absent) { ++next_chunk; continue; }
        // Coalescing a speculative ticket needs no additional staging.
        const bool shared = s.prefetched.contains(e.records[next_chunk]);
        if (!shared && fresh >= ahead) break;
        const auto started = std::chrono::steady_clock::now();
        auto ticket = s.reads->request(e.records[next_chunk], ReadPriority::kDemand);
        if (!ticket) break;
        pending_reads.push_back({next_chunk++, std::move(ticket), started});
        fresh += !shared;
      }
    };
    for (size_t chunk = 0; chunk < e.records.size(); ++chunk) {
      const size_t r = e.records[chunk];
      const auto& record = s.store->records()[r];
      if (s.slots[r] != absent) {
        s.storage->assemble(slot, record, s.slots[r], {});
        cached_bytes += record.layout.useful_bytes;
      } else {
        fill_window();
        if (pending_reads.empty() && !s.pending_copies.empty()) {
          drain_copies();
          fill_window();
        }
        std::shared_ptr<ReadTicket> queued;
        std::chrono::steady_clock::time_point queued_at;
        if (!pending_reads.empty() && pending_reads.front().chunk == chunk) {
          queued = std::move(pending_reads.front().ticket);
          queued_at = pending_reads.front().started;
          pending_reads.pop_front();
        }
        if ((!queued || !queued->ready()) && !s.pending_copies.empty()) drain_copies();
        s.pending_copy = s.read(r, std::move(queued), queued_at);
        s.pending_copies.push_back(std::move(s.pending_copy));
        next_chunk = std::max(next_chunk, chunk + 1);
        const auto bytes = s.pending_copies.back()->bytes().first(record.layout.allocation_bytes);
        s.storage->assemble(slot, record, std::nullopt, bytes);
        const bool retain_demand = admission && record.key.chunk < admission->chunks[expert];
        if (retain_demand) {
          if (s.free_slots.empty()) throw std::logic_error("expert cache: demand admission has no validated slot");
          const auto slot = s.free_slots.back();
          s.storage->retain(slot, bytes);
          // Reserve each destination immediately, but do not publish the
          // planner's new residency until every copy has completed below.
          s.remove_free_slot(slot);
          s.slots[r] = slot;
          s.slot_records[slot] = r;
          s.counters.retained_bytes += record.layout.allocation_bytes;
        }
        s.counters.demand_bytes += record.layout.record_bytes;
        staged_bytes += record.layout.useful_bytes;
      }
    }
    drain_copies();
    if (admission) {
      // Publish the admitted prefix only after all demand copies are fenced,
      // without a second storage read. Whole policy remains all-or-nothing.
      s.planner->commit_admission(expert, *admission);
      s.count_transition(*admission, previous);
      ++s.counters.demand_admissions;
      s.counters.demand_admission_bytes += admission->load_bytes;
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - reconstruction_started).count();
    const auto waited = s.counters.read_wait_ns - wait_before;
    const auto wall_ns = static_cast<uint64_t>(std::max<int64_t>(elapsed, 0));
    s.counters.reconstruction_wall_ns += wall_ns > waited ? wall_ns - waited : 0;
    s.counters.cached_copy_bytes += cached_bytes;
    s.counters.staged_copy_bytes += staged_bytes;
    s.counters.execution_zero_bytes += s.storage->execution_zero_bytes_per_begin();
    s.execution_busy[slot] = true;
    ++s.counters.active_execution_slots;
    s.counters.peak_execution_slots = std::max(s.counters.peak_execution_slots, s.counters.active_execution_slots);
    return ExpertExecutionLease(state_, slot);
  } catch (...) { s.counters.poisoned = true; throw; }
}
void ExpertCache::execute_ready(std::span<const size_t> selected,
    const std::function<void(std::span<const size_t>, std::span<const ExpertExecutionLease>)>& execute) {
  auto& s = *state_;
  s.check();
  if (s.counters.active_execution_slots || s.counters.active_native_leases)
    throw std::logic_error("expert cache: ready batch requires quiescent leases");
  for (auto e : selected) s.expert_index(e);
  // A partial eviction may unpack a native donor. Give it independent scratch
  // so it cannot overwrite an expert which is awaiting admission below.
  const bool fractional = s.config.planner.policy == Policy::kAdaptiveFractional;
  const size_t capacity = s.memory.execution_slots - (fractional ? 1 : 0);
  if (!capacity) {
    for (size_t i = 0; i < selected.size(); ++i) {
      auto lease = acquire(selected[i], true);
      execute(std::span<const size_t>(&i, 1), std::span<const ExpertExecutionLease>(&lease, 1));
    }
    return;
  }
  struct Job { size_t input, expert, slot, remaining = 0; bool launched = false; };
  struct Read { size_t job, record; std::shared_ptr<ReadTicket> ticket;
    std::chrono::steady_clock::time_point started; };
  const auto started = std::chrono::steady_clock::now();
  const auto waited_before = s.counters.read_wait_ns;
  try {
    for (size_t base = 0; base < selected.size(); base += capacity) {
      const size_t count = std::min(capacity, selected.size() - base);
      std::vector<Job> jobs;
      std::vector<ExpertExecutionLease> leases;
      std::vector<std::pair<size_t, size_t>> missing;
      jobs.reserve(count); leases.reserve(count);
      // Snapshot every resident source before any admission can evict it.
      for (size_t j = 0; j < count; ++j) {
        const size_t e = selected[base+j];
        jobs.push_back({base+j, e, j});
        s.storage->begin_execution(j);
        s.counters.execution_zero_bytes += s.storage->execution_zero_bytes_per_begin();
        size_t hits = 0;
        if (s.native_start[e] != absent) {
          s.storage->native_to_execution(j, s.native_start[e]);
          hits = s.experts[e].records.size();
          ++s.counters.native_hits;
          ++s.counters.canonical_unpacks;
          s.counters.canonical_unpack_bytes += s.full.useful_bytes;
        } else for (const auto r : s.experts[e].records) {
          if (s.slots[r] != absent) {
            s.storage->assemble(j, s.store->records()[r], s.slots[r], {});
            s.counters.cached_copy_bytes += s.store->records()[r].layout.useful_bytes;
            ++hits;
          } else {
            missing.emplace_back(j, r);
            ++jobs.back().remaining;
          }
        }
        ++s.counters.requests;
        s.counters.chunk_lookups += s.experts[e].records.size();
        s.counters.chunk_hits += hits;
        if (hits == s.experts[e].records.size()) ++s.counters.full_hits;
        else { s.all_observed_full = false; if (hits) ++s.counters.partial_hits; }
        s.execution_busy[j] = true;
        ++s.counters.active_execution_slots;
        s.counters.peak_execution_slots = std::max(s.counters.peak_execution_slots, s.counters.active_execution_slots);
        leases.push_back(ExpertExecutionLease(state_, j));
      }
      // Preserve useful speculative results, including errors. Unrelated
      // speculation must not occupy staging needed by these demanded records.
      for (auto it = s.prefetched.begin(); it != s.prefetched.end();) {
        if (std::none_of(missing.begin(), missing.end(), [&](const auto& m) { return m.second == it->first; }))
          it = s.prefetched.erase(it);
        else ++it;
      }
      std::vector<Read> pending;
      size_t next = 0, launched = 0;
      const auto drain = [&] { s.storage->finish(); s.pending_copies.clear(); };
      const auto launch_ready = [&] {
        // Contiguous completed runs need no additional lease ownership.
        for (size_t j = 0; j < jobs.size();) {
          if (jobs[j].remaining || jobs[j].launched) { ++j; continue; }
          const size_t first = j;
          std::vector<size_t> inputs;
          while (j < jobs.size() && !jobs[j].remaining && !jobs[j].launched) {
            inputs.push_back(jobs[j].input); jobs[j++].launched = true; ++launched;
          }
          execute(inputs, std::span<const ExpertExecutionLease>(leases).subspan(first, j-first));
        }
      };
      while (launched < jobs.size()) {
        while (next < missing.size()) {
          const auto [j, r] = missing[next];
          const auto pref = s.prefetched.find(r);
          if (pref == s.prefetched.end() &&
              pending.size() + s.pending_copies.size() + s.prefetched.size() >= s.config.reads.buffers) break;
          const auto queued = std::chrono::steady_clock::now();
          auto ticket = s.reads->request(r, ReadPriority::kDemand);
          if (pref != s.prefetched.end()) { ticket = std::move(pref->second.ticket); s.prefetched.erase(pref); }
          if (!ticket) break;
          pending.push_back({j, r, std::move(ticket), queued}); ++next;
        }
        bool consumed = false;
        for (size_t i = 0; i < pending.size();) {
          if (!pending[i].ticket->ready()) { ++i; continue; }
          auto read = std::move(pending[i]);
          pending.erase(pending.begin()+i);
          const auto& record = s.store->records()[read.record];
          s.pending_copy = s.read(read.record, std::move(read.ticket), read.started);
          s.pending_copies.push_back(std::move(s.pending_copy));
          s.storage->assemble(jobs[read.job].slot, record, std::nullopt,
              s.pending_copies.back()->bytes().first(record.layout.allocation_bytes));
          s.counters.demand_bytes += record.layout.record_bytes;
          s.counters.staged_copy_bytes += record.layout.useful_bytes;
          --jobs[read.job].remaining;
          consumed = true;
        }
        launch_ready();  // same-stream consumers follow assembly, without a fence
        if (launched == jobs.size()) break;
        if (consumed) continue;
        if (!s.pending_copies.empty()) { drain(); continue; }
        if (pending.empty()) throw std::runtime_error("expert cache: ready batch read backpressure");
        std::vector<std::shared_ptr<ReadTicket>> tickets;
        for (const auto& r : pending) tickets.push_back(r.ticket);
        const auto wait = std::chrono::steady_clock::now();
        (void)s.reads->wait_any(tickets);
        s.counters.read_wait_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - wait).count();
      }
      drain();
      // Execution has already been submitted. Admission reads its complete
      // canonical scratch, not a second file read or retained staging window.
      for (const auto& job : jobs) {
        auto plan = s.planner->propose_admission(job.expert);
        const auto previous = s.planner->current().chunks;
        if (plan.chunks[job.expert] <= previous[job.expert]) continue;
        s.evict(plan, fractional ? capacity : job.slot);
        for (size_t c = previous[job.expert]; c < plan.chunks[job.expert]; ++c) {
          const auto r = s.experts[job.expert].records[c];
          if (s.free_slots.empty()) throw std::logic_error("expert cache: ready admission slot missing");
          const size_t slot = s.free_slots.back();
          s.storage->unpack_native_chunk(job.slot, s.store->records()[r], slot);
          s.remove_free_slot(slot); s.slots[r] = slot; s.slot_records[slot] = r;
          s.counters.retained_bytes += s.store->records()[r].layout.allocation_bytes;
        }
        s.storage->finish();
        s.planner->commit_admission(job.expert, plan);
        s.count_transition(plan, previous);
        ++s.counters.demand_admissions;
        s.counters.demand_admission_bytes += plan.load_bytes;
      }
    }
    const auto wall = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - started).count());
    const auto waited = s.counters.read_wait_ns - waited_before;
    s.counters.reconstruction_wall_ns += wall > waited ? wall - waited : 0;
  } catch (...) { s.counters.poisoned = true; throw; }
}
bool ExpertCache::prepare_resident(std::span<const size_t> experts) {
  auto& s = *state_;
  s.check();
  if (s.counters.active_execution_slots || s.counters.active_native_leases)
    throw std::logic_error("expert cache: native preparation requires quiescent readers");
  for (const auto e : experts) {
    s.expert_index(e);
    if (s.planner->current().chunks[e] != s.experts[e].records.size()) return false;
  }
  if (experts.empty() || !s.storage->cache_data(0)) return false;
  try {
    for (const auto e : experts) if (s.native_start[e] == absent) s.pack_native(e, 0);
  } catch (...) { s.counters.poisoned = true; throw; }
  return true;
}
void ExpertCache::release_native_leases(std::span<ExpertExecutionLease> leases) {
  auto& s = *state_;
  s.check_thread();
  for (const auto& lease : leases)
    if (lease.state_.get() != state_.get() || !lease.native_data_)
      throw std::invalid_argument("expert cache: grouped release requires this cache's native leases");
  if (leases.empty()) return;
  try { s.storage->retire_native(); }
  catch (...) { s.counters.poisoned = true; throw; }
  for (auto& lease : leases) { lease.state_.reset(); lease.native_data_ = nullptr; }
  s.counters.active_native_leases -= leases.size();
}
size_t ExpertCache::prefetch(size_t expert) {
  auto& s = *state_;
  s.check();
  s.expert_index(expert);
  if (!s.background_allowed()) { ++s.counters.prefetches_deferred; return 0; }
  size_t accepted = 0;
  for (auto record : s.experts[expert].records) {
    if (s.slots[record] != absent || s.prefetched.contains(record)) continue;
    if (s.prefetched.size() >= s.config.reads.buffers - 1) break;
    auto ticket = s.reads->request(record, ReadPriority::kPrefetch);
    if (!ticket) break;
    s.prefetched.emplace(record, ExpertCacheState::QueuedRead{std::move(ticket), {}});
    ++accepted;
  }
  return accepted;
}
size_t ExpertCache::lookahead_demand(size_t expert) {
  auto& s = *state_;
  s.check();
  s.expert_index(expert);
  const size_t limit = s.config.reads.buffers / 2;
  size_t accepted = 0;
  for (auto record : s.experts[expert].records) {
    if (s.slots[record] != absent || s.prefetched.contains(record)) continue;
    if (s.prefetched.size() + s.pending_copies.size() >= limit) break;
    const auto started = std::chrono::steady_clock::now();
    auto ticket = s.reads->request(record, ReadPriority::kDemand);
    if (!ticket) break;
    s.prefetched.emplace(record, ExpertCacheState::QueuedRead{std::move(ticket), started});
    ++accepted;
  }
  return accepted;
}
CacheStats ExpertCache::stats() const {
  state_->check_thread();
  auto result = state_->counters;
  result.completion_wait_ns = state_->storage->completion_wait_ns();
  return result;
}
ReadQueueStats ExpertCache::read_stats() const { return state_->reads->stats(); }
CacheTelemetry ExpertCache::telemetry() const {
  state_->check_thread();
  const auto& s = *state_;
  CacheTelemetry result;
  result.policy = s.config.planner.policy;
  result.io_mode = s.store->io_mode();
  result.cache = s.counters;
  result.cache.completion_wait_ns = s.storage->completion_wait_ns();
  result.queue = s.reads->stats();
  result.file = s.store->stats();
  result.memory = s.memory;
  result.pressure = s.pressure;
  for (const auto slot : s.slots) if (slot != absent) {
    ++result.resident_chunks;
    result.resident_bytes += s.memory.cache_slot_bytes;
  }
  for (const auto& e : s.experts) result.total_chunks += e.records.size();
  return result;
}
void ExpertCache::require_execution_context(const void* context) const {
  state_->check();
  if (!state_->storage->accepts_execution_context(context))
    throw std::invalid_argument("expert cache: executor context differs from storage completion context");
}
void ExpertCache::release_execution_thread() {
  auto& s = *state_;
  s.check_thread();
  std::lock_guard lock(s.owner_mutex);
  if (s.counters.active_execution_slots || s.counters.active_native_leases || s.pending_copy)
    throw std::logic_error("expert cache: cannot hand off live execution leases or copies");
  try { s.storage->finish(); }
  catch (...) { s.counters.poisoned = true; throw; }
  s.owner = {};
}
void ExpertCache::bind_execution_thread() {
  auto& s = *state_;
  std::lock_guard lock(s.owner_mutex);
  const auto current = std::this_thread::get_id();
  if (s.owner != std::thread::id{} && s.owner != current)
    throw std::logic_error("expert cache: previous execution thread must release ownership first");
  s.owner = current;
}
ExpertExecutionLease::ExpertExecutionLease(std::shared_ptr<ExpertCacheState> state, size_t slot)
    : state_(std::move(state)), slot_(slot) {}
ExpertExecutionLease::ExpertExecutionLease(std::shared_ptr<ExpertCacheState> state, size_t expert, const uint8_t* data)
    : state_(std::move(state)), slot_(expert), native_data_(data) {}
ExpertExecutionLease::~ExpertExecutionLease() { release_noexcept(); }
ExpertExecutionLease::ExpertExecutionLease(ExpertExecutionLease&& other) noexcept
    : state_(std::move(other.state_)), slot_(other.slot_), native_data_(other.native_data_) {}
ExpertExecutionLease& ExpertExecutionLease::operator=(ExpertExecutionLease&& other) noexcept {
  if (this != &other) { release_noexcept(); state_ = std::move(other.state_); slot_ = other.slot_; native_data_ = other.native_data_; }
  return *this;
}
const uint8_t* ExpertExecutionLease::data() const {
  if (!state_) throw std::logic_error("expert cache: empty execution lease");
  state_->check();
  return native_data_ ? native_data_ : state_->storage->execution_data(slot_);
}
const uint8_t* ExpertExecutionLease::packed_data() const {
  if (!state_) throw std::logic_error("expert cache: empty execution lease");
  state_->check();
  if (!state_->config.packed_nvfp4) throw std::logic_error("expert cache: packed layout is disabled");
  if (native_data_) return native_data_;
  try { return state_->storage->packed_execution_data(slot_); }
  catch (...) { state_->counters.poisoned = true; throw; }
}
void ExpertExecutionLease::release() {
  if (!state_) return;
  auto state = state_;  // retain storage through the completion fence
  if (native_data_) state->release_native();
  else state->release_execution(slot_);
  state_.reset();
}
void ExpertExecutionLease::release_noexcept() noexcept {
  if (!state_) return;
  try { release(); }
  catch (...) { state_->counters.poisoned = true; state_.reset(); }
}
}  // namespace dgpp::expert
