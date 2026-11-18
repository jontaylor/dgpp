#include "loaders/expert_cache.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <utility>
#include <unistd.h>

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
  return add(add(mul(cache_slot_bytes, cache_slots), mul(execution_slot_bytes, execution_slots)),
             add(staging_bytes, chunk_scratch_bytes));
}

CacheMemory expert_cache_memory(const StoreIdentity& id, const CacheConfig& cfg) {
  if (!cfg.execution_slots || cfg.planner.reserved_bytes)
    throw std::invalid_argument("expert cache: invalid execution reservation");
  const auto chunks = partition(id.hidden, id.intermediate, id.world, id.rank, id.chunk_intermediate, id.format);
  const auto full = partition(id.hidden, id.intermediate, id.world, id.rank, id.intermediate / id.world, id.format).front();
  CacheMemory m;
  m.budget_bytes = cfg.planner.budget_bytes;
  m.execution_slot_bytes = full.allocation_bytes;
  if (cfg.packed_fp4) {
    m.execution_slot_bytes = mul(m.execution_slot_bytes, 2);
  }
  m.execution_slots = cfg.execution_slots;
  uint64_t record_bytes = 0;
  for (const auto& c : chunks) {
    m.cache_slot_bytes = std::max(m.cache_slot_bytes, c.allocation_bytes);
    record_bytes = std::max(record_bytes, c.record_bytes);
  }
  if (cfg.reads.require_mx_nibbles) {
    if (!cfg.packed_fp4 || id.format != Format::kMxfp4)
      throw std::invalid_argument("MX scale packing requires native MXFP4 expert execution");
    const auto payload = chunks.front().gate.payload_bytes;
    const auto scales = payload / 16;
    for (const auto& c : chunks) {
      uint64_t offset = 0;
      for (const auto* matrix : {&c.gate, &c.up, &c.down}) {
        if (matrix->payload_offset != offset || matrix->payload_bytes != payload ||
            matrix->scales_offset != offset + payload || matrix->scales_bytes != scales)
          throw std::invalid_argument("MX scale packing requires uniform contiguous chunks");
        offset += payload + scales;
      }
      if (payload % 512 || c.allocation_bytes != offset)
        throw std::invalid_argument("MX scale packing requires aligned chunks without padding");
    }
    m.chunk_scratch_bytes = m.cache_slot_bytes;
    m.cache_slot_bytes -= 3 * scales / 2;
  }
  m.staging_bytes = mul(record_bytes, cfg.reads.buffers);
  if (!cfg.reads.buffers || !cfg.reads.workers || cfg.reads.workers > cfg.reads.buffers ||
      cfg.reads.workers > cfg.reads.max_pending || cfg.reads.max_pending < cfg.reads.buffers ||
      m.staging_bytes > cfg.reads.staging_budget_bytes)
    throw std::invalid_argument("expert cache: invalid bounded read capacity");
  const uint64_t reserved = add(add(m.staging_bytes, m.chunk_scratch_bytes), mul(m.execution_slot_bytes, m.execution_slots));
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
  // Allocated only when the opt-in read scan is enabled. The store is
  // immutable and these results survive eviction, but every later read
  // replaces the result for its record before a caller can use it.
  std::vector<ScaleCompactibility> record_scale_compactibility;
  std::vector<uint64_t> expert_record_bytes;
  std::vector<bool> execution_busy, execution_packed;
  size_t next_execution_slot = 0;
  std::vector<std::vector<std::shared_ptr<const ReadLease>>> direct_reads;
  size_t direct_staging_count() const {
    size_t count = 0;
    for (const auto& reads : direct_reads) count += reads.size();
    return count;
  }
  struct QueuedRead {
    std::shared_ptr<ReadTicket> ticket;
    std::chrono::steady_clock::time_point demanded_at{};
  };
  std::map<size_t, QueuedRead> prefetched;
  CacheStats counters;
  uint64_t prepared_revision = 1;
  std::map<uint32_t, uint64_t> prepared_layer_revisions;
  void set_native_start(size_t expert, size_t start) {
    native_start[expert] = start;
    prepared_layer_revisions.at(experts[expert].layer) = ++prepared_revision;
  }
  // Optional diagnostics. The engine thread owns this stream; it is never
  // touched by I/O workers or allocated in the normal serving configuration.
  std::unique_ptr<std::ofstream> demand_trace;
  bool demand_trace_header = false;
  std::unique_ptr<std::ofstream> residency_trace;
  std::vector<size_t> traced_residency;
  uint64_t demand_sequence = 0;
  // Diagnostic only: rank absent upcoming experts without submitting I/O or
  // changing admission. Histories contain only observations already seen.
  std::unique_ptr<std::ofstream> prefetch_shadow;
  std::vector<std::vector<size_t>> shadow_layers;
  std::vector<uint64_t> shadow_all, shadow_decode;
  uint64_t shadow_sequence = 0;
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
  void trace_demand(std::span<const size_t> selected, bool prefill) {
    if (!demand_trace) return;
    auto& out = *demand_trace;
    auto word = [&](uint64_t value) {
      out.write(reinterpret_cast<const char*>(&value), sizeof(value));
    };
    if (!demand_trace_header) {
      // Little-endian 64-bit words: magic, version, capacity in cache slots,
      // expert count, then (draft, layer, expert, chunk count, resident chunks,
      // record bytes) per expert. Slots have a uniform charge in this cache.
      word(0x31444e414d454744ULL); // DGEMAND1 in little endian
      word(1); word(memory.cache_slots); word(experts.size());
      const auto& resident = planner->current().chunks;
      for (size_t e = 0; e < experts.size(); ++e) {
        word(experts[e].draft); word(experts[e].layer); word(experts[e].expert);
        word(experts[e].records.size()); word(resident[e]); word(expert_record_bytes[e]);
      }
      demand_trace_header = true;
    }
    // Each event is one layer/batch observation, including cache hits and
    // repeated routed selections. The reader groups IDs for fetch accounting.
    const auto timestamp = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
    word(prefill); word(selected.size()); word(timestamp);
    for (const auto e : selected) word(e);
    out.flush();
    if (!out) throw std::runtime_error("expert cache: failed to write demand trace");
    if (residency_trace) {
      auto& trace = *residency_trace;
      auto value = [&](uint64_t v) { trace.write(reinterpret_cast<const char*>(&v), sizeof(v)); };
      const auto& resident = planner->current().chunks;
      if (traced_residency.empty()) {
        trace.write("DGRSIDE1", 8);
        value(1); value(experts.size()); value(memory.cache_slots);
        for (auto chunks : resident) value(chunks);
        traced_residency.assign(resident.begin(), resident.end());
      }
      size_t changes = 0;
      for (size_t e = 0; e < resident.size(); ++e) changes += resident[e] != traced_residency[e];
      value(demand_sequence); value(timestamp); value(changes);
      for (size_t e = 0; e < resident.size(); ++e) {
        if (resident[e] == traced_residency[e]) continue;
        value(e); value(resident[e]); traced_residency[e] = resident[e];
      }
      trace.flush();
      if (!trace) throw std::runtime_error("expert cache: failed to write residency trace");
    }
    ++demand_sequence;
  }
  void trace_prefetch_shadow(std::span<const size_t> selected, bool prefill) {
    if (!prefetch_shadow) return;
    ++shadow_sequence;
    if (selected.empty()) return;
    const auto layer = experts[selected.front()].layer;
    for (auto e : selected) {
      if (experts[e].draft || experts[e].layer != layer) return;
    }
    for (auto e : selected) {
      ++shadow_all[e];
      if (!prefill) ++shadow_decode[e];
    }
    const auto timestamp = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
    const auto& resident = planner->current().chunks;
    for (const unsigned distance : {1u, 2u, 4u}) {
      const auto target = static_cast<size_t>(layer) + distance;
      if (target >= shadow_layers.size()) continue;
      for (unsigned mode = 0; mode < 2; ++mode) {
        const auto& history = mode ? shadow_decode : shadow_all;
        std::array<size_t, 3> best{absent, absent, absent};
        auto better = [&](size_t a, size_t b) {
          return b == absent || history[a] > history[b] ||
              (history[a] == history[b] && experts[a].expert < experts[b].expert);
        };
        for (const auto e : shadow_layers[target]) {
          if (!history[e] || resident[e] == experts[e].records.size()) continue;
          // Already queued reads are not new speculative opportunities.
          if (std::any_of(experts[e].records.begin(), experts[e].records.end(),
                          [&](size_t r) { return prefetched.contains(r); })) continue;
          for (size_t rank = 0; rank < best.size(); ++rank) {
            if (!better(e, best[rank])) continue;
            for (size_t j = best.size() - 1; j > rank; --j) best[j] = best[j-1];
            best[rank] = e;
            break;
          }
        }
        for (size_t rank = 0; rank < best.size() && best[rank] != absent; ++rank) {
          const auto e = best[rank];
          // Ten little-endian u64 fields, one record per offered expert.
          const uint64_t record[]{timestamp, shadow_sequence, layer, target, distance,
                                  mode, rank + 1, experts[e].expert, history[e], uint64_t(prefill)};
          prefetch_shadow->write(reinterpret_cast<const char*>(record), sizeof(record));
        }
      }
    }
    prefetch_shadow->flush();
    if (!*prefetch_shadow) throw std::runtime_error("expert cache: failed to write prefetch shadow trace");
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
    if (!record_scale_compactibility.empty())
      record_scale_compactibility[record] = lease->scale_compactibility();
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
  size_t free_execution_slot() {
    for (size_t i = 0; i < execution_busy.size(); ++i) {
      const size_t slot = (next_execution_slot + i) % execution_busy.size();
      if (!execution_busy[slot]) {
        next_execution_slot = (slot + 1) % execution_busy.size();
        return slot;
      }
    }
    return absent;
  }
  void reap_copies(bool wait_one = false) {
    try {
      // Retire a completed prefix once. Chunk-direct waves can keep hundreds
      // of sources live: repeatedly scanning and erasing that whole vector
      // for every incoming chunk turns bounded bookkeeping into quadratic work.
      // Waiting for the oldest consumer is safe; it never depends on a future
      // acquisition, and later completed sources may remain owned meanwhile.
      size_t completed = 0;
      while (completed < pending_copies.size() &&
             storage->staging_complete(pending_copies[completed]->bytes(), wait_one && completed == 0))
        ++completed;
      pending_copies.erase(pending_copies.begin(), pending_copies.begin() + completed);
    } catch (...) { counters.poisoned = true; throw; }
  }
  void retain(size_t record) {
    if (free_slots.empty()) throw std::logic_error("expert cache: no free slot in validated transition");
    reap_copies(true);
    pending_copy = read(record);
    const auto size = store->records()[record].layout.allocation_bytes;
    const auto slot = free_slots.back();
    storage->retain(slot, pending_copy->bytes().first(size));
    storage->staging_complete(pending_copy->bytes(), true);
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
    storage->order_mutations();
    set_native_start(expert, absent);
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
    storage->order_mutations();
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
    storage->order_mutations();
    for (size_t c = 0; c < records.size(); ++c) {
      if (free_positions[first + c] != absent) remove_free_slot(first + c);
      slots[records[c]] = first + c;
      slot_records[first + c] = records[c];
    }
    for (const size_t slot : native_holes) add_free_slot(slot);
    set_native_start(expert, first);
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
        else set_native_start(e, absent);
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
    try {
      storage->retire_execution(slot);
      for (auto& read : direct_reads[slot]) {
        storage->retire_staging_reader(read->bytes());
        pending_copies.push_back(std::move(read));
      }
      direct_reads[slot].clear();
    }
    catch (...) { counters.poisoned = true; throw; }
    execution_busy[slot] = false;
    --counters.active_execution_slots;
  }
  void release_native(size_t expert) {
    check_thread();
    try { storage->retire_native_extent(native_start[expert], experts[expert].records.size()); }
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
  const uint64_t reserved = add(add(m.staging_bytes, m.chunk_scratch_bytes), mul(m.execution_slot_bytes, m.execution_slots));
  std::map<std::tuple<bool, uint32_t, uint32_t>, size_t> groups;
  s.slots.assign(s.store->records().size(), absent);
  if (cfg.reads.scan_nvfp4_scales)
    s.record_scale_compactibility.assign(s.store->records().size(), ScaleCompactibility::kUnknown);
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
    specs.push_back({std::vector<uint64_t>(chunks.size(), m.cache_slot_bytes), 1.0,
                     uint64_t{e.layer} * 2 + static_cast<uint64_t>(e.draft)});
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
  for (const auto& expert : s.experts) s.prepared_layer_revisions.try_emplace(expert.layer, 0);
  for (size_t slot = 0; slot < m.cache_slots; ++slot) s.add_free_slot(slot);
  s.execution_busy.resize(cfg.execution_slots, false);
  s.execution_packed.resize(cfg.execution_slots, false);
  s.direct_reads.resize(cfg.execution_slots);
  for (auto& reads : s.direct_reads) reads.reserve(chunks.size());
  s.pending_copies.reserve(cfg.reads.buffers);
  s.reads = std::make_unique<ExpertReadQueue>(s.store, cfg.reads);
  auto spans = s.reads->staging_spans();
  s.storage = factory(m, spans);
  if (!s.storage || s.storage->memory() != m)
    throw std::invalid_argument("expert cache: backend reservation disagrees with the memory plan");
  if (const char* prefix = std::getenv("DGPP_EXPERT_DEMAND_TRACE"); prefix && *prefix) {
    const auto path = std::string(prefix) + ".r" + std::to_string(id.rank) +
        ".p" + std::to_string(::getpid()) + ".c" + std::to_string(s.binding_identity) + ".bin";
    s.demand_trace = std::make_unique<std::ofstream>(path, std::ios::binary | std::ios::trunc);
    if (!*s.demand_trace) throw std::runtime_error("expert cache: cannot open demand trace");
  }
  if (const char* prefix = std::getenv("DGPP_EXPERT_RESIDENCY_TRACE"); prefix && *prefix) {
    if (!s.demand_trace)
      throw std::invalid_argument("expert cache: residency trace requires demand trace");
    const auto path = std::string(prefix) + ".r" + std::to_string(id.rank) +
        ".p" + std::to_string(::getpid()) + ".c" + std::to_string(s.binding_identity) + ".bin";
    s.residency_trace = std::make_unique<std::ofstream>(path, std::ios::binary | std::ios::trunc);
    if (!*s.residency_trace) throw std::runtime_error("expert cache: cannot open residency trace");
  }
  if (const char* prefix = std::getenv("DGPP_EXPERT_PREFETCH_SHADOW"); prefix && *prefix) {
    const auto path = std::string(prefix) + ".r" + std::to_string(id.rank) +
        ".p" + std::to_string(::getpid()) + ".bin";
    s.prefetch_shadow = std::make_unique<std::ofstream>(path, std::ios::binary | std::ios::trunc);
    if (!*s.prefetch_shadow) throw std::runtime_error("expert cache: cannot open prefetch shadow trace");
    s.shadow_all.resize(s.experts.size());
    s.shadow_decode.resize(s.experts.size());
    for (size_t e = 0; e < s.experts.size(); ++e) {
      if (s.experts[e].draft) continue;
      if (s.shadow_layers.size() <= s.experts[e].layer) s.shadow_layers.resize(s.experts[e].layer + 1);
      s.shadow_layers[s.experts[e].layer].push_back(e);
    }
  }
}
ExpertCache::~ExpertCache() = default;
const CacheMemory& ExpertCache::memory() const { return state_->memory; }
uint64_t ExpertCache::binding_identity() const { return state_->binding_identity; }
const ChunkLayout& ExpertCache::execution_layout() const { return state_->full; }
const std::vector<CachedExpert>& ExpertCache::experts() const { return state_->experts; }
ScaleCompactibility ExpertCache::scale_compactibility(size_t expert) const {
  const auto& s = *state_;
  s.check(); s.expert_index(expert);
  if (s.record_scale_compactibility.empty()) return ScaleCompactibility::kUnknown;
  auto result = ScaleCompactibility::kExact;
  for (const auto record : s.experts[expert].records) {
    const auto value = s.record_scale_compactibility[record];
    if (value == ScaleCompactibility::kUnsupported) return value;
    if (value == ScaleCompactibility::kUnknown) result = value;
  }
  return result;
}
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
void ExpertCache::observe(std::span<const size_t> selected, bool prefill) {
  auto& s = *state_;
  s.check();
  s.planner->observe(selected, prefill);
  s.trace_demand(selected, prefill);
  s.trace_prefetch_shadow(selected, prefill);
  s.observed_for_adaptation |= !selected.empty();
  for (const auto e : selected)
    s.all_observed_full &= s.planner->current().chunks[e] == s.experts[e].records.size();
}
const ResidencyPlan& ExpertCache::residency() const { state_->check(); return state_->planner->current(); }
uint64_t ExpertCache::observations() const { state_->check(); return state_->planner->observations(); }
void ExpertCache::adapt() {
  auto& s = *state_;
  s.check();
  // Host leases must be retired before adaptation; the backend orders slot
  // writes against the corresponding device readers.
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
    s.storage->order_mutations();
    s.evict(plan);
    for (size_t e = 0; e < s.experts.size(); ++e)
      for (size_t c = previous[e]; c < plan.chunks[e]; ++c) s.retain(s.experts[e].records[c]);
    s.planner->commit(plan);  // publication only after all transfers succeed
    s.count_transition(plan, previous);
    s.observed_for_adaptation = false;
    s.all_observed_full = true;
  } catch (...) { s.counters.poisoned = true; throw; }
}
bool ExpertCache::packed_fp4() const { return state_->config.packed_fp4; }
const uint8_t* ExpertCache::prepared_address(size_t expert) const {
  const auto& s = *state_;
  s.check(); s.expert_index(expert);
  return s.native_start[expert] == absent ? nullptr : s.storage->cache_data(s.native_start[expert]);
}
uint64_t ExpertCache::prepared_revision() const { state_->check(); return state_->prepared_revision; }
uint64_t ExpertCache::prepared_layer_revision(uint32_t layer) const {
  state_->check();
  return state_->prepared_layer_revisions.at(layer);
}
void ExpertCache::prepared_addresses(std::span<const size_t> experts,
                                    std::span<const uint8_t*> addresses) const {
  const auto& s = *state_;
  s.check();
  if (experts.size() != addresses.size())
    throw std::invalid_argument("expert cache: prepared address span sizes differ");
  for (const auto e : experts) s.expert_index(e);
  for (size_t i = 0; i < experts.size(); ++i) {
    const auto slot = s.native_start[experts[i]];
    addresses[i] = slot == absent ? nullptr : s.storage->cache_data(slot);
  }
}
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
const ChunkLayout& ExpertCache::chunk_layout() const { return state_->store->records().front().layout; }
ExpertExecutionLease ExpertCache::acquire(size_t expert, bool canonical, bool chunk_direct) {
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
    if (canonical && s.config.packed_fp4) {
      const size_t slot = s.free_execution_slot();
      if (slot == absent) { ++s.counters.backpressure; return {}; }
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
  const size_t slot = s.free_execution_slot();
  if (slot == absent) { ++s.counters.backpressure; return {}; }
  size_t hits = 0;
  for (const auto r : e.records) hits += s.slots[r] != absent;
  // Initial support covers complete misses. Partial cache views continue to
  // use reconstruction so later admissions cannot invalidate their sources.
  const auto& chunk = chunk_layout();
  chunk_direct &= hits == 0 && s.storage->supports_chunk_direct() &&
      chunk.intermediate_size % 64 == 0 &&
      (chunk.intermediate_size & (chunk.intermediate_size-1)) == 0 && s.full.intermediate_size % chunk.intermediate_size == 0 &&
      e.records.size() <= s.config.reads.buffers / 2;
  std::vector<const uint8_t*> chunk_sources;
  if (chunk_direct) chunk_sources.reserve(e.records.size());
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
        !(canonical && s.config.packed_fp4)) {
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
        // Earlier executions own independent reconstruction slots. The backend
        // orders writes after readers of the persistent slots being reused.
        s.storage->order_mutations();
        s.evict(*admission, slot);
      }
    }
    s.storage->begin_execution(slot);
    // Keep a bounded window of missing chunks in flight. Staging leases remain
    // live until their individual copy fences complete, even across acquire().
    // Wait for one buffer only when the read window cannot make progress.
    struct PendingRead {
      size_t chunk;
      std::shared_ptr<ReadTicket> ticket;
      std::chrono::steady_clock::time_point started;
    };
    std::deque<PendingRead> pending_reads;
    size_t next_chunk = 0;
    const auto drain_copies = [&] {
      s.reap_copies(true);
    };
    const auto fill_window = [&] {
      s.reap_copies();
      const size_t reserved = s.prefetched.size() + s.pending_copies.size() + s.direct_staging_count();
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
        s.reap_copies();
        s.pending_copy = s.read(r, std::move(queued), queued_at);
        auto& readers = chunk_direct ? s.direct_reads[slot] : s.pending_copies;
        readers.push_back(std::move(s.pending_copy));
        next_chunk = std::max(next_chunk, chunk + 1);
        const auto bytes = readers.back()->bytes().first(record.layout.allocation_bytes);
        if (chunk_direct) chunk_sources.push_back(s.storage->staging_data(bytes));
        else s.storage->assemble(slot, record, std::nullopt, bytes);
        const bool retain_demand = admission && record.key.chunk < admission->chunks[expert];
        if (retain_demand) {
          if (s.free_slots.empty()) throw std::logic_error("expert cache: demand admission has no validated slot");
          const auto slot = s.free_slots.back();
          s.storage->retain(slot, bytes);
          // Reserve each destination immediately, but do not publish the
          // planner's new residency until every copy has been ordered below.
          s.remove_free_slot(slot);
          s.slots[r] = slot;
          s.slot_records[slot] = r;
          s.counters.retained_bytes += record.layout.allocation_bytes;
        }
        s.counters.demand_bytes += record.layout.record_bytes;
        staged_bytes += record.layout.useful_bytes;
      }
    }
    if (chunk_direct) s.storage->bind_chunk_sources(slot, chunk_sources);
    s.reap_copies();
    if (admission) {
      // Publish the admitted prefix after all demand copies are ordered; the
      // backend fences consumers before they can read these addresses,
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
    if (chunk_direct) {
      ++s.counters.chunk_direct_executions;
      s.counters.chunk_direct_bytes += staged_bytes;
    } else s.counters.staged_copy_bytes += staged_bytes;
    s.counters.execution_zero_bytes += s.storage->execution_zero_bytes_per_begin();
    s.execution_busy[slot] = true;
    ++s.counters.active_execution_slots;
    s.counters.peak_execution_slots = std::max(s.counters.peak_execution_slots, s.counters.active_execution_slots);
    return ExpertExecutionLease(state_, slot, chunk_direct);
  } catch (...) { s.counters.poisoned = true; throw; }
}
void ExpertCache::execute_ready(std::span<const size_t> selected,
    const std::function<void(std::span<const size_t>, std::span<const ExpertExecutionLease>)>& execute,
    ProjectionReads* projections, const ExecuteProjection& execute_projection) {
  auto& s = *state_;
  s.check();
  if (s.counters.active_execution_slots || s.counters.active_native_leases)
    throw std::logic_error("expert cache: ready batch requires quiescent leases");
  for (auto e : selected) s.expert_index(e);
  // A partial eviction may unpack a native donor. Give it independent scratch
  // so it cannot overwrite an expert which is awaiting admission below.
  const bool fractional = s.config.planner.policy == Policy::kAdaptiveFractional;
  const size_t capacity = s.memory.execution_slots - (fractional ? 1 : 0);
  if (projections && (!execute_projection || selected.size() > capacity ||
      !std::equal(selected.begin(), selected.end(), projections->experts.begin(), projections->experts.end())))
    throw std::logic_error("expert cache: projection reservation no longer matches the missing experts");
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
      if (!projections && group_reads_enabled()) (void)prime_group(selected.subspan(base, count));
      // Snapshot every resident source before any admission can evict it.
      for (size_t j = 0; j < count; ++j) {
        const size_t e = selected[base+j];
        jobs.push_back({base+j, e, j});
        s.execution_packed[j] = false;
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
        if (std::none_of(missing.begin(), missing.end(), [&](const auto& m) { return m.second == it->first; }) &&
            !(group_reads_enabled() && std::any_of(selected.begin()+base+count, selected.end(), [&](size_t e) {
              return std::find(s.experts[e].records.begin(), s.experts[e].records.end(), it->first) != s.experts[e].records.end();
            })))
          it = s.prefetched.erase(it);
        else ++it;
      }
      std::vector<Read> pending;
      std::vector<ExpertCacheStorage::Assembly> ready_assemblies;
      ready_assemblies.reserve(s.config.reads.buffers);
      size_t next = 0, launched = 0;
      const auto drain = [&] { s.reap_copies(true); };
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
      if (projections) {
        if (projections->tickets.size() != 2 * missing.size())
          throw std::logic_error("expert cache: projection reservation size changed");
        std::vector<const uint8_t*> addresses(count);
        for (auto part : {ExpertProjection::kGateUp, ExpertProjection::kDown}) {
          const bool gate = part == ExpertProjection::kGateUp;
          size_t idx = 0;
          for (auto& job : jobs) {
            ready_assemblies.clear();
            for (const size_t r : s.experts[job.expert].records) {
              const auto& record = s.store->records()[r];
              auto ticket = std::move(projections->tickets[(gate ? 0 : missing.size()) + idx++]);
              if (gate) {
                const auto wait = std::chrono::steady_clock::now();
                s.pending_copies.push_back(ticket->wait());
                s.counters.read_wait_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - wait).count();
              } else {
                s.pending_copies.push_back(s.read(r, std::move(ticket), projections->queued));
                s.counters.demand_bytes += record.layout.record_bytes;
                s.counters.staged_copy_bytes += record.layout.useful_bytes;
              }
              ready_assemblies.push_back({job.slot, &record, std::nullopt,
                  s.pending_copies.back()->bytes().first(record.layout.allocation_bytes)});
            }
            addresses[job.slot] = s.storage->prepare_projection(job.slot, ready_assemblies, part);
            if (!gate) s.execution_packed[job.slot] = true;
          }
          if (gate) {
            const auto down = std::span(projections->tickets).subspan(missing.size());
            if (projections->delay_gate)
              for (const auto& ticket : down) (void)ticket->wait();
            projections->down_pending_at_gate = std::any_of(down.begin(), down.end(),
                [](const auto& t) { return !t->ready(); });
          }
          execute_projection(addresses, part);
        }
        launched = count;
      }
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
        ready_assemblies.clear();
        for (size_t i = 0; i < pending.size();) {
          if (!pending[i].ticket->ready()) { ++i; continue; }
          auto read = std::move(pending[i]);
          pending.erase(pending.begin()+i);
          const auto& record = s.store->records()[read.record];
          s.pending_copy = s.read(read.record, std::move(read.ticket), read.started);
          s.pending_copies.push_back(std::move(s.pending_copy));
          ready_assemblies.push_back({jobs[read.job].slot, &record, std::nullopt,
              s.pending_copies.back()->bytes().first(record.layout.allocation_bytes)});
          s.counters.demand_bytes += record.layout.record_bytes;
          s.counters.staged_copy_bytes += record.layout.useful_bytes;
          --jobs[read.job].remaining;
          consumed = true;
        }
        s.storage->assemble_many(ready_assemblies);
        if (group_reads_enabled() && next == missing.size() && base+count < selected.size()) {
          const auto following = selected.subspan(base+count, std::min(capacity, selected.size()-base-count));
          size_t fresh = 0;
          for (const auto e : following) for (const auto r : s.experts[e].records)
            fresh += s.slots[r] == absent && !s.prefetched.contains(r);
          // Current rolling reads may be queued without destinations. Leave
          // credits for all of them, not just buffers already held physically.
          const size_t promised = pending.size() + s.pending_copies.size() +
              s.direct_staging_count() + s.prefetched.size();
          if (promised <= s.config.reads.buffers && fresh <= s.config.reads.buffers-promised)
            (void)prime_group(following);
        }
        launch_ready();  // lease publication orders consumers after assembly
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
      s.reap_copies();
      // Execution has already been submitted. Admission reads its complete
      // canonical scratch, not a second file read or retained staging window.
      for (const auto& job : jobs) {
        auto plan = s.planner->propose_admission(job.expert);
        const auto previous = s.planner->current().chunks;
        if (plan.chunks[job.expert] <= previous[job.expert]) continue;
        s.evict(plan, fractional ? capacity : job.slot);
        // A wholly missing expert can be retained directly in the prepared
        // layout when eviction freed an aligned whole-expert extent. Its
        // canonical reconstruction is still live in this execution slot.
        // Fall back to independent chunk retention for fragmented space and
        // for all partial-residency admissions.
        static const bool direct_native_admission = [] {
          const char* value = std::getenv("DGPP_EXPERT_DIRECT_NATIVE_ADMISSION");
          return value && std::atoi(value) != 0;
        }();
        const auto& records = s.experts[job.expert].records;
        size_t native_first = absent;
        if (direct_native_admission && s.config.packed_fp4 && !fractional &&
            previous[job.expert] == 0 && plan.chunks[job.expert] == records.size()) {
          for (const size_t first : s.free_slots) {
            if (first % records.size() || first + records.size() > s.slot_records.size()) continue;
            bool free = true;
            for (size_t c = 0; c < records.size(); ++c)
              free &= s.free_positions[first + c] != absent;
            if (free) { native_first = first; break; }
          }
        }
        if (native_first != absent) {
          if (!s.execution_packed[job.slot] ||
              !s.storage->packed_execution_to_native(job.slot, native_first)) {
            s.storage->execution_to_native(job.slot, native_first);
          } else {
            ++s.counters.direct_native_packed_copies;
          }
          s.storage->order_mutations();
          for (size_t c = 0; c < records.size(); ++c) {
            const auto r = records[c];
            const size_t slot = native_first + c;
            s.remove_free_slot(slot); s.slots[r] = slot; s.slot_records[slot] = r;
            s.counters.retained_bytes += s.store->records()[r].layout.allocation_bytes;
          }
          s.set_native_start(job.expert, native_first);
          ++s.counters.native_packs;
          s.counters.native_layout_copy_bytes += s.full.allocation_bytes;
        } else {
          for (size_t c = previous[job.expert]; c < plan.chunks[job.expert]; ++c) {
            const auto r = records[c];
            if (s.free_slots.empty()) throw std::logic_error("expert cache: ready admission slot missing");
            const size_t slot = s.free_slots.back();
            s.storage->unpack_native_chunk(job.slot, s.store->records()[r], slot);
            s.remove_free_slot(slot); s.slots[r] = slot; s.slot_records[slot] = r;
            s.counters.retained_bytes += s.store->records()[r].layout.allocation_bytes;
          }
          s.storage->order_mutations();
        }
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
  try {
    std::vector<std::pair<size_t, size_t>> extents;
    extents.reserve(leases.size());
    for (const auto& lease : leases)
      extents.emplace_back(s.native_start[lease.slot_], s.experts[lease.slot_].records.size());
    s.storage->retire_native_extents(extents);
  }
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
bool early_demand_enabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("DGPP_EXPERT_EARLY_DEMAND");
    return !value || std::atoi(value) != 0;
  }();
  return enabled;
}
bool ready_worker_enabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("DGPP_EXPERT_READY_WORKER");
    return value && std::atoi(value) != 0;
  }();
  return enabled;
}
void ExpertCache::set_demand_window(bool active) noexcept {
  state_->reads->set_demand_window(active);
}
bool group_reads_enabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("DGPP_EXPERT_GROUP_READS");
    return !value || std::atoi(value) != 0;
  }();
  return enabled;
}
size_t ExpertCache::demand_group_capacity() const {
  const auto& s = *state_;
  const size_t chunks = std::max<size_t>(1, s.experts.front().records.size());
  return std::max<size_t>(1, std::min(s.memory.execution_slots / 2,
                                    s.config.reads.buffers / (2*chunks)));
}
ProjectionReads ExpertCache::prime_projections(std::span<const int32_t> ids,
    std::span<const size_t> mapping) {
  auto& s = *state_;
  s.check();
  if (s.config.planner.policy != Policy::kWholeExpert || !s.storage->supports_projection_reads()) return {};
  std::array<bool, 384> selected{};
  if (mapping.size() > selected.size()) return {};
  for (const int id : ids) {
    if (id < 0 || static_cast<size_t>(id) >= mapping.size())
      throw std::out_of_range("expert cache: projection route ID");
    selected[id] = true;
  }
  ProjectionReads result;
  std::vector<size_t> records;
  for (size_t id = 0; id < mapping.size(); ++id) {
    if (!selected[id]) continue;
    const auto e = mapping[id];
    s.expert_index(e);
    const auto& chunks = s.experts[e].records;
    const size_t resident = s.planner->current().chunks[e];
    if (resident == chunks.size()) continue;
    if (resident || result.experts.size() == s.memory.execution_slots ||
        records.size() + chunks.size() > s.config.reads.buffers / 2) return {};
    for (const auto r : chunks)
      if (s.slots[r] != absent || s.prefetched.contains(r)) return {};
    result.experts.push_back(e);
    records.insert(records.end(), chunks.begin(), chunks.end());
  }
  if (records.empty()) return {};
  s.reap_copies();
  result.queued = std::chrono::steady_clock::now();
  result.tickets = s.reads->reserve_projections(records);
  if (result.tickets.empty()) return {};
  return result;
}
size_t ExpertCache::prime_group(std::span<const size_t> selected) {
  auto& s = *state_;
  s.check();
  s.reap_copies();
  std::vector<size_t> records;
  for (size_t i = 0; i < selected.size(); ++i) {
    const auto e = selected[i];
    s.expert_index(e);
    if (std::find(selected.begin(), selected.begin()+i, e) != selected.begin()+i) continue;
    for (const auto r : s.experts[e].records)
      if (s.slots[r] == absent) records.push_back(r);
  }
  if (records.empty()) return 0;
  auto tickets = s.reads->reserve_demand(records);
  if (tickets.empty()) return 0;
  const auto started = std::chrono::steady_clock::now();
  size_t fresh = 0;
  for (size_t j = 0; j < records.size(); ++j) {
    const auto [it, added] = s.prefetched.try_emplace(records[j],
        ExpertCacheState::QueuedRead{std::move(tickets[j]), started});
    fresh += added;
    if (!added && it->second.demanded_at == std::chrono::steady_clock::time_point{})
      it->second.demanded_at = started;
  }
  return fresh;
}
size_t ExpertCache::prime_demand(std::span<const int32_t> ids, std::span<const size_t> mapping) {
  auto& s = *state_;
  s.check();
  std::array<bool,384> selected{};
  if (mapping.size() > selected.size()) return 0;
  for (const auto id : ids) {
    if (id < 0 || static_cast<size_t>(id) >= mapping.size())
      throw std::out_of_range("expert cache: early route ID");
    selected[id] = true;
  }
  if (group_reads_enabled()) {
    std::array<size_t,384> group{};
    size_t count = 0;
    const size_t capacity = demand_group_capacity();
    for (size_t id = 0; id < mapping.size() && count < capacity; ++id) {
      if (!selected[id]) continue;
      const auto e = mapping[id];
      s.expert_index(e);
      if (s.planner->current().chunks[e] != s.experts[e].records.size()) group[count++] = e;
    }
    return prime_group(std::span<const size_t>(group).first(count));
  }
  size_t accepted = 0, issued_experts = 0;
  for (size_t id = 0; id < mapping.size() && issued_experts < 2; ++id) {
    if (!selected[id]) continue;
    const auto e = mapping[id];
    s.expert_index(e);
    if (s.planner->current().chunks[e] == s.experts[e].records.size()) continue;
    const size_t count = lookahead_demand(e);
    accepted += count;
    if (count) ++issued_experts;
    if (s.prefetched.size() + s.pending_copies.size() + s.direct_staging_count() >= s.config.reads.buffers / 2)
      break;
  }
  return accepted;
}
bool ExpertCache::demand_ready(size_t expert) const {
  const auto& s = *state_;
  s.check();
  s.expert_index(expert);
  if (s.native_start[expert] != absent) return true;
  for (auto record : s.experts[expert].records) {
    if (s.slots[record] != absent) continue;
    const auto it = s.prefetched.find(record);
    if (it == s.prefetched.end() || !it->second.ticket->ready()) return false;
  }
  return true;
}
size_t ExpertCache::lookahead_demand(size_t expert) {
  auto& s = *state_;
  s.check();
  s.expert_index(expert);
  s.reap_copies();
  const size_t limit = s.config.reads.buffers / 2;
  size_t accepted = 0;
  for (auto record : s.experts[expert].records) {
    if (s.slots[record] != absent || s.prefetched.contains(record)) continue;
    if (s.prefetched.size() + s.pending_copies.size() + s.direct_staging_count() >= limit) break;
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
  s.pending_copies.clear();
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
ExpertExecutionLease::ExpertExecutionLease(std::shared_ptr<ExpertCacheState> state, size_t slot, bool chunked)
    : state_(std::move(state)), slot_(slot), chunked_(chunked) {}
ExpertExecutionLease::ExpertExecutionLease(std::shared_ptr<ExpertCacheState> state, size_t expert, const uint8_t* data)
    : state_(std::move(state)), slot_(expert), native_data_(data) {}
ExpertExecutionLease::~ExpertExecutionLease() { release_noexcept(); }
ExpertExecutionLease::ExpertExecutionLease(ExpertExecutionLease&& other) noexcept
    : state_(std::move(other.state_)), slot_(other.slot_), native_data_(other.native_data_), chunked_(other.chunked_) {}
ExpertExecutionLease& ExpertExecutionLease::operator=(ExpertExecutionLease&& other) noexcept {
  if (this != &other) { release_noexcept(); state_ = std::move(other.state_); slot_ = other.slot_; native_data_ = other.native_data_; chunked_ = other.chunked_; }
  return *this;
}
const uint8_t* const* ExpertExecutionLease::chunk_sources() const {
  if (!state_ || !chunked_) throw std::logic_error("expert cache: not a chunk-direct lease");
  state_->check();
  return reinterpret_cast<const uint8_t* const*>(state_->storage->execution_data(slot_));
}
const uint8_t* ExpertExecutionLease::data() const {
  if (chunked_) throw std::logic_error("expert cache: chunk-direct lease has no contiguous payload");
  if (!state_) throw std::logic_error("expert cache: empty execution lease");
  state_->check();
  return native_data_ ? native_data_ : state_->storage->execution_data(slot_);
}
const uint8_t* ExpertExecutionLease::packed_data() const {
  if (chunked_) throw std::logic_error("expert cache: chunk-direct lease has no packed payload");
  if (!state_) throw std::logic_error("expert cache: empty execution lease");
  state_->check();
  if (!state_->config.packed_fp4) throw std::logic_error("expert cache: packed layout is disabled");
  if (native_data_) return native_data_;
  try {
    const auto* data = state_->storage->packed_execution_data(slot_);
    state_->execution_packed[slot_] = true;
    return data;
  }
  catch (...) { state_->counters.poisoned = true; throw; }
}
void ExpertExecutionLease::release() {
  if (!state_) return;
  auto state = state_;  // retain storage through the completion fence
  if (native_data_) state->release_native(slot_);
  else state->release_execution(slot_);
  state_.reset();
}
void ExpertExecutionLease::release_noexcept() noexcept {
  if (!state_) return;
  try { release(); }
  catch (...) { state_->counters.poisoned = true; state_.reset(); }
}
}  // namespace dgpp::expert
