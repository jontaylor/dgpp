#include "loaders/expert_cache.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <chrono>
#include <functional>
#include <exception>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#include "common/test.hpp"

namespace {
using namespace dgpp::expert;
void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
template <typename F> void refuses(F f) {
  bool rejected = false;
  try { f(); } catch (const std::exception&) { rejected = true; }
  require(rejected, "invalid/failed cache operation must be refused");
}
struct Fixture {
  std::filesystem::path dir;
  std::string path;
  StoreIdentity identity;
  std::shared_ptr<ExpertStore> store;
  std::array<std::vector<uint8_t>, 3> codes, scales;
  std::array<MatrixSource, 3> sources;
  bool distinct;
  Fixture(Format format = Format::kMxfp4, bool distinct_experts = false) : distinct(distinct_experts) {
    std::string pattern = "/tmp/dgpp-expert-cache-XXXXXX";
    require(::mkdtemp(pattern.data()) != nullptr, "mkdtemp");
    dir = pattern;
    path = (dir / "experts.img").string();
    identity.checkpoint_sha256.fill(43);
    identity.format = format;
    identity.rank = 1;
    identity.world = 2;
    identity.hidden = 64;
    identity.intermediate = 256;
    identity.chunk_intermediate = 32;
    for (size_t i = 0; i < 3; ++i) {
      const uint64_t rows = i < 2 ? identity.intermediate : identity.hidden;
      const uint64_t cols = i < 2 ? identity.hidden : identity.intermediate;
      codes[i].resize(rows * cols / 2);
      scales[i].resize(rows * cols / (format == Format::kMxfp4 ? 32 : 16));
      for (size_t b = 0; b < codes[i].size(); ++b) codes[i][b] = static_cast<uint8_t>(b * 41 + i);
      for (size_t b = 0; b < scales[i].size(); ++b) scales[i][b] = static_cast<uint8_t>(b * 31 + i);
      sources[i] = {rows, cols, format, codes[i], scales[i], std::nullopt};
      if (format == Format::kNvfp4) sources[i].global = 1.5f + static_cast<float>(i);
    }
    std::vector<RecordKey> keys;
    for (uint32_t e = 0; e < 4; ++e)
      for (uint32_t c = 0; c < 4; ++c) keys.push_back({e / 2, e % 2, c, e >= 2});
    ExpertStoreBuilder builder(path, identity, keys);
    for (size_t i = 0; i < keys.size(); ++i)
      builder.write(i, pack_expert(keys[i].layer * 2 + keys[i].expert, 128 + keys[i].chunk * 32, 32));
    builder.finish();
    store = std::make_shared<ExpertStore>(path, identity, keys.size(), IoMode::kDirect);
  }
  ~Fixture() { std::error_code ec; std::filesystem::remove_all(dir, ec); }
  CacheConfig config(Policy policy) const {
    const auto chunk = partition(64, 256, 2, 1, 32, identity.format).front();
    const auto full = pack_chunk(sources, 128, 128);
    CacheConfig cfg;
    cfg.execution_slots = 2;
    cfg.reads.buffers = 3;
    cfg.reads.workers = 1;
    cfg.reads.max_pending = 4;
    cfg.reads.staging_budget_bytes = 3 * chunk.record_bytes;
    cfg.reads.prefetch_bytes = 2 * chunk.record_bytes;
    cfg.planner.policy = policy;
    cfg.planner.budget_bytes = 8 * chunk.allocation_bytes + 2 * full.layout.allocation_bytes + cfg.reads.staging_budget_bytes + 17;
    cfg.planner.minimum_fraction = 0.25;
    cfg.planner.decay_routes = 4;
    cfg.planner.horizon_routes = 8192;
    cfg.planner.promotion_margin = 0;
    cfg.planner.cooldown_routes = 0;
    return cfg;
  }
  PackedChunk pack_expert(size_t expert, uint64_t begin, uint64_t width) const {
    if (!distinct) return pack_chunk(sources, begin, width);
    auto payloads = codes;
    auto unique = sources;
    for (size_t m = 0; m < 3; ++m) {
      for (auto& byte : payloads[m]) byte ^= static_cast<uint8_t>(expert * 17);
      unique[m].payload = payloads[m];
      if (unique[m].global) *unique[m].global += static_cast<float>(expert);
    }
    return pack_chunk(unique, begin, width);
  }
  std::vector<uint8_t> expected(size_t expert = 0) const { return pack_expert(expert, 128, 128).bytes; }
};
struct Control {
  size_t finishes = 0, retains = 0, staged_assemblies = 0;
  bool fail_copy = false, fail_finish = false, native = false;
  std::function<void()> before_finish;
};
// Host byte reference, with precisely the same bounded slabs as the CUDA
// backend. This validates the real cache/store/queue lifecycle, not GPU kernels.
class HostStorage final : public ExpertCacheStorage {
 public:
  HostStorage(CacheMemory memory, StoreIdentity identity, std::shared_ptr<Control> control)
      : memory_(memory), id_(identity), control_(std::move(control)),
        cache_(memory.cache_slots * memory.cache_slot_bytes),
        execution_(memory.execution_slots * memory.execution_slot_bytes) {}
  const CacheMemory& memory() const override { return memory_; }
  void retain(size_t slot, std::span<const uint8_t> bytes) override {
    ++control_->retains;
    if (control_->fail_copy) throw std::runtime_error("injected device copy failure");
    require(slot < memory_.cache_slots && bytes.size() <= memory_.cache_slot_bytes, "cache slab bound");
    std::memcpy(cache_.data() + slot * memory_.cache_slot_bytes, bytes.data(), bytes.size());
  }
  void begin_execution(size_t slot) override {
    auto span = execution(slot);
    std::fill(span.begin(), span.end(), 0);
  }
  void assemble(size_t slot, const StoreRecord& r, std::optional<size_t> cache_slot,
      std::span<const uint8_t> host_bytes) override {
    if (!cache_slot) ++control_->staged_assemblies;
    if (cache_slot) {
      require(*cache_slot < memory_.cache_slots, "source cache slab bound");
      host_bytes = std::span<const uint8_t>(cache_).subspan(*cache_slot * memory_.cache_slot_bytes, r.layout.allocation_bytes);
    }
    scatter_chunk(r.layout, host_bytes, id_.hidden, id_.intermediate / id_.world, id_.format,
                  execution(slot), id_.rank * (id_.intermediate / id_.world));
  }
  const uint8_t* execution_data(size_t slot) const override {
    require(slot < memory_.execution_slots, "execution pointer bound");
    return execution_.data() + slot * memory_.execution_slot_bytes;
  }
  const uint8_t* cache_data(size_t slot) const override {
    if (!control_->native) return nullptr;
    require(slot < memory_.cache_slots, "native cache pointer bound");
    return cache_.data() + slot * memory_.cache_slot_bytes;
  }
  void move_cache_slot(size_t dest, size_t source) override {
    require(dest < memory_.cache_slots && source < memory_.cache_slots, "native relocation bounds");
    std::memcpy(cache_.data() + dest * memory_.cache_slot_bytes,
                cache_.data() + source * memory_.cache_slot_bytes, memory_.cache_slot_bytes);
  }
  void native_to_execution(size_t slot, size_t first) override {
    require(first * memory_.cache_slot_bytes + memory_.execution_slot_bytes <= cache_.size(), "native read extent");
    std::memcpy(execution(slot).data(), cache_.data() + first * memory_.cache_slot_bytes, memory_.execution_slot_bytes);
  }
  void execution_to_native(size_t slot, size_t first) override {
    require(first * memory_.cache_slot_bytes + memory_.execution_slot_bytes <= cache_.size(), "native write extent");
    if (control_->fail_copy) throw std::runtime_error("injected native pack failure");
    std::memcpy(cache_.data() + first * memory_.cache_slot_bytes, execution(slot).data(), memory_.execution_slot_bytes);
  }
  void unpack_native_chunk(size_t slot, const StoreRecord& r, size_t cache_slot) override {
    require(cache_slot < memory_.cache_slots, "native unpack bound");
    auto* dest = cache_.data() + cache_slot * memory_.cache_slot_bytes;
    std::memset(dest, 0, memory_.cache_slot_bytes);
    const auto source = execution(slot);
    for (const auto& c : scatter_copies(r.layout, id_.hidden, id_.intermediate / id_.world,
                                      id_.format, id_.rank * (id_.intermediate / id_.world)))
      for (uint64_t row = 0; row < c.rows; ++row)
        std::memcpy(dest + c.source_offset + row * c.source_pitch,
                    source.data() + c.destination_offset + row * c.destination_pitch, c.row_bytes);
  }
  void finish() override {
    ++control_->finishes;
    if (control_->before_finish) control_->before_finish();
    if (control_->fail_finish) throw std::runtime_error("injected device fence failure");
  }
 private:
  std::span<uint8_t> execution(size_t slot) {
    require(slot < memory_.execution_slots, "execution slab bound");
    return std::span<uint8_t>(execution_).subspan(slot * memory_.execution_slot_bytes, memory_.execution_slot_bytes);
  }
  CacheMemory memory_;
  StoreIdentity id_;
  std::shared_ptr<Control> control_;
  std::vector<uint8_t> cache_, execution_;
};
CacheStorageFactory host(const Fixture& f, std::shared_ptr<Control> control = std::make_shared<Control>()) {
  return [id = f.identity, control = std::move(control)](const CacheMemory& m, std::span<const std::span<uint8_t>> spans) {
    uint64_t staging = 0;
    for (const auto span : spans) staging += span.size();
    require(staging == m.staging_bytes, "complete staging reservation passed to backend");
    return std::make_unique<HostStorage>(m, id, control);
  };
}
void check_bytes(const ExpertExecutionLease& lease, const Fixture& f, size_t expert = 0) {
  const auto expected = f.expected(expert);
  require(lease && std::memcmp(lease.data(), expected.data(), expected.size()) == 0,
          "complete TP-local reconstruction exactly matches checkpoint bytes including padding/globals");
}
}

DGPP_TEST(expert_cache_fractional_floor_initialization_exact_reconstruction_and_budget) {
  for (auto format : {Format::kMxfp4, Format::kNvfp4}) {
    Fixture f(format);
    ExpertCache cache(f.store, f.config(Policy::kAdaptiveFractional), host(f));
    refuses([&] { (void)cache.acquire(0); });
    cache.initialize();
    require(cache.experts().size() == 4 && cache.experts()[2].draft, "global main/draft population");
    const auto& memory = cache.memory();
    require(memory.allocated_bytes() + memory.unallocated_bytes == memory.budget_bytes && memory.unallocated_bytes == 17,
            "staging, execution, chunk padding and unallocated tail reconcile to budget");
    const auto calls = f.store->stats().calls;
    require(calls == 8, "universal 50 percent coverage populated at startup");
    for (size_t e = 0; e < 4; ++e) {
      auto lease = cache.acquire(e);
      check_bytes(lease, f);
    }
    require(cache.stats().partial_hits == 4 && cache.stats().chunk_hits == 8, "first lookup hit denominator before reads");
    require(f.store->stats().calls == calls + 8, "only missing half read for executions");
    require(cache.stats().active_execution_slots == 0, "destructors fence and return slots");
    const auto stats = cache.stats();
    // NVFP4 globals are copied once per chunk, including repeated values.
    const auto useful_chunk = f.store->records().front().layout.useful_bytes;
    require(stats.cached_copy_bytes == 8 * useful_chunk &&
            stats.staged_copy_bytes == stats.cached_copy_bytes &&
            stats.execution_zero_bytes == 4 * memory.execution_slot_bytes,
            "reconstruction copy and zero traffic stay distinct from aligned file reads");
  }
}
DGPP_TEST(expert_cache_fractional_demand_retains_without_rereads_and_reaches_hot_set) {
  for (auto format : {Format::kMxfp4, Format::kNvfp4}) {
    Fixture f(format);
    auto cfg = f.config(Policy::kAdaptiveFractional);
    cfg.planner.minimum_fraction = 0;
    cfg.planner.adaptation_bytes = 0;
    ExpertCache cache(f.store, cfg, host(f));
    cache.initialize();
    for (size_t e : {size_t{0}, size_t{1}}) {
      cache.observe(std::array<size_t, 1>{e});
      const auto missing = 4 - cache.residency().chunks[e];
      const auto before = f.store->stats().calls;
      auto lease = cache.acquire(e);
      check_bytes(lease, f);
      lease.release();
      require(f.store->stats().calls == before + missing, "admission reuses demanded bytes without another file read");
    }
    require(cache.residency().chunks == std::vector<size_t>({4, 4, 0, 0}), "two hot experts fit exactly with zero cold floor");
    const auto reads = f.store->stats().calls;
    for (int repeat = 0; repeat < 4; ++repeat)
      for (size_t e : {size_t{0}, size_t{1}}) {
        cache.observe(std::array<size_t, 1>{e});
        auto lease = cache.acquire(e);
        check_bytes(lease, f);
      }
    require(f.store->stats().calls == reads && cache.stats().full_hits == 8 &&
                cache.stats().demand_admissions == 2, "hot repeats have no I/O without periodic adaptation");
    require(cache.memory().allocated_bytes() <= cfg.planner.budget_bytes, "shared reservation remains bounded");
  }
}

DGPP_TEST(expert_cache_native_hits_bind_same_slab_without_reconstruction_or_reads) {
  for (auto format : {Format::kMxfp4, Format::kNvfp4}) {
    for (const auto policy : {Policy::kWholeExpert, Policy::kAdaptiveFractional}) {
      Fixture f(format);
      auto control = std::make_shared<Control>();
      control->native = true;
      auto cfg = f.config(policy);
      cfg.planner.minimum_fraction = 0;
      ExpertCache cache(f.store, cfg, host(f, control));
      cache.initialize();
      cache.observe(std::array<size_t, 1>{0});
      { auto cold = cache.acquire(0); check_bytes(cold, f); }
      { auto packed = cache.acquire(0); check_bytes(packed, f); }
      const auto before = cache.stats();
      const auto reads = f.store->stats().calls;
      auto first = cache.acquire(0);
      auto second = cache.acquire(0);
      check_bytes(first, f);
      check_bytes(second, f);
      require(first.data() == second.data() && cache.stats().active_native_leases == 2,
              "native leases bind the same resident payload without consuming execution scratch");
      const auto after = cache.stats();
      require(after.native_packs == 1 && after.native_hits == before.native_hits + 2 &&
                  after.direct_native_hits == before.direct_native_hits + 2 &&
                  after.canonical_unpacks == before.canonical_unpacks &&
                  after.cached_copy_bytes == before.cached_copy_bytes &&
                  after.staged_copy_bytes == before.staged_copy_bytes &&
                  after.execution_zero_bytes == before.execution_zero_bytes &&
                  after.reconstruction_wall_ns == before.reconstruction_wall_ns &&
                  f.store->stats().calls == reads, "native hits perform no reconstruction, zeroing or storage reads");
      refuses([&] { cache.adapt(); });
      refuses([&] { cache.release_execution_thread(); });
      auto moved = std::move(first);
      moved.release(); second.release();
      require(cache.stats().active_native_leases == 0, "moved native leases release exactly once");
    }
  }
}

DGPP_TEST(expert_cache_native_fragmentation_demotion_and_pinned_readers_preserve_bytes) {
  for (auto format : {Format::kMxfp4, Format::kNvfp4}) {
    Fixture f(format, true);
    auto control = std::make_shared<Control>();
    control->native = true;
    ExpertCache cache(f.store, f.config(Policy::kAdaptiveFractional), host(f, control));
    cache.initialize();
    cache.observe(std::vector<size_t>(128, 0));
    { auto promoted = cache.acquire(0); check_bytes(promoted, f); }
    auto native = cache.acquire(0); check_bytes(native, f);
    cache.observe(std::vector<size_t>(256, 3));
    const auto old = cache.residency().chunks;
    { auto demand = cache.acquire(3); check_bytes(demand, f, 3); }
    require(cache.residency().chunks == old, "a native reader prevents all relocation and admission eviction");
    check_bytes(native, f);
    native.release();
    { auto demand = cache.acquire(3); check_bytes(demand, f, 3); }
    require(cache.stats().native_unpacks >= 1 && cache.residency().chunks[0] >= 1,
            "native-to-partial demotion restores independently owned chunks before eviction");
    for (size_t e = 0; e < 4; ++e) { auto lease = cache.acquire(e); check_bytes(lease, f, e); }
    require(cache.memory().allocated_bytes() <= cache.memory().budget_bytes, "layout conversion never adds a payload slab");
  }
}

DGPP_TEST(expert_cache_native_layouts_survive_repeated_random_admission_and_adaptation) {
  for (auto format : {Format::kMxfp4, Format::kNvfp4}) {
    for (const auto policy : {Policy::kWholeExpert, Policy::kAdaptiveFractional}) {
      for (const double floor : {0., .25}) {
        for (const size_t tail : {size_t{0}, size_t{3}}) {
          Fixture f(format, true);
          auto control = std::make_shared<Control>(); control->native = true;
          auto cfg = f.config(policy); cfg.planner.minimum_fraction = floor;
          // Four chunks per expert: exercise both an exact slab and a final
          // incomplete aligned run that must never host a native expert.
          cfg.planner.budget_bytes += tail * f.store->records().front().layout.allocation_bytes;
          ExpertCache cache(f.store, cfg, host(f, control));
          cache.initialize();
          std::mt19937 rng(1017);
          for (int step = 0; step < 128; ++step) {
            const size_t e = rng() % 4;
            cache.observe(std::array<size_t, 1>{e});
            { auto lease = cache.acquire(e); check_bytes(lease, f, e); }
            { auto lease = cache.acquire(e); check_bytes(lease, f, e); }
            if (step % 17 == 0) cache.adapt();
            const auto& p = cache.residency();
            require(p.cache_bytes + p.reserved_bytes <= cfg.planner.budget_bytes, "native layout churn respects hard budget");
            for (size_t count : p.chunks)
              require(policy == Policy::kWholeExpert ? count == 0 || count == 4 : count >= (floor ? 1u : 0u),
                      "native layout churn preserves indivisibility or fractional floor");
          }
          require(cache.stats().native_packs > 0 && cache.stats().active_native_leases == 0,
                  "random transitions exercise native execution and retire all readers");
        }
      }
    }
  }
}

DGPP_TEST(expert_cache_grouped_native_preparation_is_stable_and_retires_with_one_fence) {
  Fixture f(Format::kNvfp4, true);
  auto control = std::make_shared<Control>(); control->native = true;
  auto cfg = f.config(Policy::kAdaptiveFractional); cfg.planner.minimum_fraction = 0;
  ExpertCache cache(f.store, cfg, host(f, control));
  cache.initialize();
  const std::array<size_t, 2> routed{0, 1};
  const auto initial_revision = cache.prepared_revision();
  require(!cache.prepared_address(0), "unprepared experts have no GPU binding");
  require(!cache.prepare_resident(routed) && cache.stats().native_packs == 0,
          "partial routed set does not convert layouts speculatively");
  for (const auto e : routed) {
    cache.observe(std::array<size_t, 1>{e});
    auto lease = cache.acquire(e); check_bytes(lease, f, e);
  }
  require(cache.prepare_resident(routed) && cache.stats().native_packs == 2, "all hot experts pack inside the shared slab");
  const auto revision = cache.prepared_revision();
  require(revision > initial_revision && cache.prepared_address(0) && cache.prepared_address(1),
          "native packing publishes addresses with a new revision");
  const auto packed = cache.stats();
  for (int repeat = 0; repeat < 4; ++repeat) {
    require(cache.prepare_resident(routed), "stable routed set remains prepared");
    std::vector<ExpertExecutionLease> leases;
    for (const auto e : routed) { leases.push_back(cache.acquire(e)); check_bytes(leases.back(), f, e); }
    const auto fences = control->finishes;
    cache.release_native_leases(leases);
    require(control->finishes == fences + 1 && !leases[0] && !leases[1] &&
                cache.stats().active_native_leases == 0, "one fence retires every grouped reader");
  }
  const auto after = cache.stats();
  require(cache.prepared_revision() == revision, "native hit execution leaves binding revision stable");
  const auto observations = cache.observations();
  cache.record_native_hits(std::array<size_t, 3>{0, 1, 0});
  require(cache.stats().requests == after.requests + 2 && cache.stats().native_hits == after.native_hits + 2 &&
          cache.observations() == observations + 3,
          "GPU hit accounting preserves unique executions and repeated route observations");
  require(after.native_packs == packed.native_packs && after.native_unpacks == packed.native_unpacks &&
              after.native_layout_copy_bytes == packed.native_layout_copy_bytes,
          "neighboring hot experts do not repeatedly unpack or relocate one another");
}

DGPP_TEST(expert_cache_failed_native_layout_conversion_poisons_before_publication) {
  Fixture f;
  auto control = std::make_shared<Control>(); control->native = true;
  ExpertCache cache(f.store, f.config(Policy::kWholeExpert), host(f, control));
  cache.initialize();
  { auto demand = cache.acquire(0); check_bytes(demand, f); }
  control->fail_copy = true;
  refuses([&] { (void)cache.acquire(0); });
  require(cache.stats().poisoned && cache.stats().native_packs == 0 && cache.stats().active_native_leases == 0,
          "a failed native pack cannot expose an executable native binding");
}

DGPP_TEST(expert_cache_whole_adaptation_hits_and_execution_slot_backpressure) {
  Fixture f;
  auto control = std::make_shared<Control>();
  ExpertCache cache(f.store, f.config(Policy::kWholeExpert), host(f, control));
  cache.initialize();
  require(f.store->stats().calls == 0, "whole cache starts cold");
  const std::array<size_t, 4> routes{0, 1, 0, 1};
  cache.observe(routes);
  cache.adapt();
  require(cache.residency().chunks == std::vector<size_t>({4, 4, 0, 0}), "whole experts only, no partial publication");
  const auto calls = f.store->stats().calls;
  auto first = cache.acquire(0);
  auto second = cache.acquire(1);
  check_bytes(first, f);
  check_bytes(second, f);
  require(!cache.acquire(2), "full execution pool returns backpressure without I/O");
  require(f.store->stats().calls == calls, "resident experts perform zero file reads");
  refuses([&] { cache.adapt(); });
  const auto fences = control->finishes;
  first.release();
  require(control->finishes == fences + 1, "release fences kernel readers before scratch reuse");
  auto missing = cache.acquire(2);
  check_bytes(missing, f);
  require(f.store->stats().calls == calls + 4, "nonresident whole expert reconstructed completely before execution");
  require(cache.stats().full_hits == 2 && cache.stats().peak_execution_slots == 2, "hits and peak leases measured");
}
DGPP_TEST(expert_cache_full_hit_workloads_skip_population_rescoring_until_locality_changes) {
  Fixture f;
  for (const auto policy : {Policy::kWholeExpert, Policy::kAdaptiveFractional}) {
    auto control = std::make_shared<Control>();
    ExpertCache cache(f.store, f.config(policy), host(f, control));
    cache.initialize();
    cache.observe(std::vector<size_t>(64, 0));
    cache.adapt();
    require(cache.residency().chunks[0] == 4, "hot expert is complete before hit-only adaptation");
    const auto generation = cache.residency().generation;
    const auto fences = control->finishes;
    cache.observe(std::vector<size_t>(128, 0));
    cache.adapt();
    require(cache.residency().generation == generation && control->finishes == fences,
            "complete observed coverage avoids both planning commit and storage fences");
    require(cache.observations() == 192, "hit-only fast path retains frequency observations");
    cache.observe(std::vector<size_t>(256, 3));
    cache.adapt();
    require(cache.residency().chunks[3] == 4, "a locality change re-enables promotion");
  }
}

DGPP_TEST(expert_cache_locality_shift_preserves_floor_and_only_loads_promotions) {
  Fixture f;
  ExpertCache cache(f.store, f.config(Policy::kAdaptiveFractional), host(f));
  cache.initialize();
  cache.observe(std::vector<size_t>(64, 0));
  cache.adapt();
  require(cache.residency().chunks[0] == 4, "hot expert promoted to full residency");
  auto counts = cache.residency().chunks;
  require(std::all_of(counts.begin(), counts.end(), [](size_t c) { return c >= 1; }), "cold experts preserve minimum coverage");
  auto lease = cache.acquire(0);
  check_bytes(lease, f);
  lease.release();
  cache.observe(std::vector<size_t>(128, 3));
  const auto before = f.store->stats().calls;
  cache.adapt();
  counts = cache.residency().chunks;
  require(counts[3] == 4 && counts[0] < 4, "allocation follows changed locality");
  require(std::all_of(counts.begin(), counts.end(), [](size_t c) { return c >= 1; }), "demotion never violates floor");
  require(f.store->stats().calls > before && cache.stats().promotions && cache.stats().demotions,
          "real store reads and transition counters track adaptation");
  for (size_t e = 0; e < 4; ++e) { auto execution = cache.acquire(e); check_bytes(execution, f); }
}
DGPP_TEST(expert_cache_whole_demand_reads_fill_cache_without_rereading_or_adaptation) {
  for (auto format : {Format::kMxfp4, Format::kNvfp4}) {
    Fixture f(format);
    auto cfg = f.config(Policy::kWholeExpert);
    cfg.planner.adaptation_bytes = 0;
    ExpertCache cache(f.store, cfg, host(f));
    cache.initialize();
    for (size_t e : {size_t{0}, size_t{1}}) {
      cache.observe(std::vector<size_t>{e});
      auto lease = cache.acquire(e);
      check_bytes(lease, f);
    }
    require(cache.residency().chunks == std::vector<size_t>({4, 4, 0, 0}), "first demand fills both complete-expert grants");
    const auto calls = f.store->stats().calls;
    require(calls == 8 && cache.stats().demand_admissions == 2, "exactly one read per demanded chunk also populates cache");
    for (size_t e : {size_t{0}, size_t{1}}) { auto lease = cache.acquire(e); check_bytes(lease, f); }
    require(f.store->stats().calls == calls && cache.stats().full_hits == 2, "immediate repeated demand is a full hit");
    cache.observe(std::vector<size_t>(128, 3));
    auto replacement = cache.acquire(3);
    check_bytes(replacement, f);
    require(f.store->stats().calls == calls + 4 && cache.stats().evictions == 1, "replacement reuses demand bytes and evicts one whole expert");
    require(cache.stats().demand_admissions == 3 && cache.stats().demand_admission_bytes == 12 * cache.memory().cache_slot_bytes,
            "demand admission bytes include aligned slots and are distinct from storage traffic");
    require(cache.telemetry().resident_bytes <= cache.memory().cache_slots * cache.memory().cache_slot_bytes,
            "demand replacement stays inside the common preallocated budget");
  }
}

DGPP_TEST(expert_cache_failed_demand_admission_is_poisoned_without_partial_publication) {
  for (const auto policy : {Policy::kWholeExpert, Policy::kAdaptiveFractional}) {
  Fixture f;
  auto control = std::make_shared<Control>();
  ExpertCache cache(f.store, f.config(policy), host(f, control));
  cache.initialize();
  cache.observe(std::array<size_t, 1>{0});
  control->fail_copy = true;
  refuses([&] { (void)cache.acquire(0); });
  require(cache.stats().poisoned && cache.stats().demand_admissions == 0 && cache.stats().promotions == 0,
          "failed cache retention never publishes a complete demand admission");
  const auto held = cache.read_stats().staging_in_use;
  require(held >= cache.memory().staging_bytes / 3 && held <= cache.memory().staging_bytes,
          "failed demand-copy staging survives alongside bounded read-ahead until backend teardown");
  refuses([&] { (void)cache.acquire(0); });
  control->fail_copy = false;
  }
}
DGPP_TEST(expert_cache_prefetch_reserves_demand_capacity_and_does_not_fake_hits) {
  Fixture f;
  ExpertCache cache(f.store, f.config(Policy::kWholeExpert), host(f));
  cache.initialize();
  require(cache.prefetch(0) == 2, "speculation bounded below staging count");
  require(cache.prefetch(1) == 0, "speculation cannot consume demand reserve");
  auto unrelated = cache.acquire(1);
  check_bytes(unrelated, f);
  unrelated.release();
  auto prefetched = cache.acquire(0);
  check_bytes(prefetched, f);
  require(cache.stats().chunk_hits == 0 && cache.stats().full_hits == 0, "prefetch read tickets are not device cache hits");
  require(f.store->stats().calls == 8, "two prefetched chunks consumed once, no rereads");
  require(cache.read_stats().staging_peak <= cache.memory().staging_bytes, "bounded staging despite unconsumed speculation");
}
DGPP_TEST(expert_cache_failure_never_publishes_partial_transition_or_reuses_pending_staging) {
  Fixture f;
  auto control = std::make_shared<Control>();
  ExpertCache cache(f.store, f.config(Policy::kWholeExpert), host(f, control));
  cache.initialize();
  cache.observe(std::vector<size_t>(32, 0));
  control->fail_finish = true;
  refuses([&] { cache.adapt(); });
  require(cache.stats().poisoned, "failed fence poisons cache");
  refuses([&] { (void)cache.acquire(0); });
  refuses([&] { cache.adapt(); });
  control->fail_finish = false;
}
DGPP_TEST(expert_cache_copy_failure_holds_staging_until_backend_teardown) {
  Fixture f;
  auto control = std::make_shared<Control>();
  ExpertCache cache(f.store, f.config(Policy::kAdaptiveFractional), host(f, control));
  control->fail_copy = true;
  refuses([&] { cache.initialize(); });
  require(cache.stats().poisoned && !cache.stats().initialized, "initial population must complete before use");
  require(cache.read_stats().staging_in_use == cache.memory().staging_bytes / 3,
          "failed device copy cannot return its source to an I/O worker");
  refuses([&] { (void)cache.acquire(0); });
}
DGPP_TEST(expert_cache_execution_lease_outlives_owner_and_move_returns_exactly_one_slot) {
  Fixture f;
  auto control = std::make_shared<Control>();
  ExpertExecutionLease lease;
  {
    ExpertCache cache(f.store, f.config(Policy::kWholeExpert), host(f, control));
    cache.initialize();
    lease = cache.acquire(0);
  }
  check_bytes(lease, f);
  const auto before = control->finishes;
  auto moved = std::move(lease);
  require(!lease, "moved lease no longer owns slot");
  moved.release();
  require(control->finishes == before + 1 && !moved, "last lease retains storage and fences once");
}
DGPP_TEST(expert_cache_invalid_budget_refused_before_payload_allocation) {
  Fixture f;
  bool called = false;
  auto cfg = f.config(Policy::kAdaptiveFractional);
  cfg.planner.budget_bytes = 1;
  CacheStorageFactory factory = [&](const CacheMemory&, std::span<const std::span<uint8_t>>) -> std::unique_ptr<ExpertCacheStorage> {
    called = true;
    throw std::runtime_error("unexpected allocation");
  };
  refuses([&] { ExpertCache cache(f.store, cfg, factory); });
  require(!called, "budget check precedes backend allocation");
  cfg = f.config(Policy::kAdaptiveFractional);
  cfg.planner.minimum_fraction = 1;
  refuses([&] { ExpertCache cache(f.store, cfg, factory); });
  require(!called, "universal minimum validated before any allocation");
}
DGPP_TEST(expert_cache_thread_handoff_requires_quiescent_explicit_release_and_bind) {
  Fixture f;
  ExpertCache cache(f.store, f.config(Policy::kAdaptiveFractional), host(f));
  cache.initialize();
  std::exception_ptr failure;
  std::thread unauthorized([&] {
    try {
      refuses([&] { cache.bind_execution_thread(); });
      refuses([&] { (void)cache.stats(); });
    } catch (...) { failure = std::current_exception(); }
  });
  unauthorized.join();
  if (failure) std::rethrow_exception(failure);
  auto lease = cache.acquire(0);
  refuses([&] { cache.release_execution_thread(); });
  lease.release();
  cache.release_execution_thread();
  refuses([&] { (void)cache.telemetry(); });
  std::thread engine([&] {
    try {
      cache.bind_execution_thread();
      auto work = cache.acquire(1);
      check_bytes(work, f);
      work.release();
      require(cache.telemetry().cache.active_execution_slots == 0, "new owner publishes completed metrics");
      cache.release_execution_thread();
    } catch (...) { failure = std::current_exception(); }
  });
  engine.join();
  if (failure) std::rethrow_exception(failure);
  cache.bind_execution_thread();
  require(cache.stats().requests == 2, "startup owner resumes after the worker joins");
}
DGPP_TEST(expert_resource_monitor_hysteresis_and_sampling_are_independent_of_wall_clock) {
  Fixture f;
  const auto mem = f.dir / "meminfo", block = f.dir / "stat";
  auto write = [&](uint64_t gib, uint64_t busy) {
    std::ofstream(mem) << "MemTotal: 131828224 kB\nMemAvailable: " << gib * 1024 * 1024 << " kB\n";
    std::ofstream(block) << "0 0 0 0 0 0 0 0 0 " << busy << " 0\n";
  };
  ResourcePressureMonitor monitor(mem.string(), block.string());
  write(3, 100);
  auto s = monitor.sample(1000);
  require(s.memory_pressure && !s.storage_busy_fraction && s.sample == 1, "first sample has no storage interval");
  write(5, 1050);
  require(monitor.sample(1500).sample == 1, "sampling capped at one hertz without filesystem rereads");
  s = monitor.sample(2000);
  require(s.memory_pressure && s.storage_pressure && *s.storage_busy_fraction == .95, "pressure thresholds and device interval");
  write(5, 1850);
  s = monitor.sample(3000);
  require(s.memory_pressure && s.storage_pressure, "both pressure signals retain hysteresis in their middle bands");
  write(7, 2450);
  s = monitor.sample(4000);
  require(s.background_allowed() && *s.available_bytes == 7ull * 1024 * 1024 * 1024 && *s.storage_busy_fraction == .6,
          "both gates resume below their release thresholds");
}
DGPP_TEST(expert_resource_monitor_unknown_reset_and_malformed_signals_do_not_invent_utilization) {
  Fixture f;
  const auto mem = f.dir / "meminfo", block = f.dir / "stat";
  ResourcePressureMonitor monitor(mem.string(), block.string());
  auto s = monitor.sample(1000);
  require(s.background_allowed() && !s.available_bytes && !s.storage_busy_fraction, "missing signals are unknown");
  std::ofstream(mem) << "MemAvailable: 0 kB\n";
  std::ofstream(block) << "0 0 0 0 0 0 0 0 0 1000 0\n";
  s = monitor.sample(2000);
  require(s.available_bytes == 0 && s.memory_pressure && !s.storage_busy_fraction, "zero memory is a valid pressure signal");
  std::ofstream(mem) << "MemAvailable: 18446744073709551615 kB\n";
  std::ofstream(block) << "0 0 0 0 0 0 0 0 0 50 0\n";
  s = monitor.sample(3000);
  require(!s.available_bytes && !s.storage_busy_fraction && s.background_allowed(), "overflow and reset counters become unknown");
  for (const auto text : {"MemAvailable: -1 kB\n", "MemAvailable: 10 MB\n", "MemAvailable: 10 kB extra\n"}) {
    std::ofstream(mem) << text;
    s = monitor.sample(4000 + s.sample * 1000);
    require(!s.available_bytes, "malformed memory source is unknown");
  }
  ResourcePressureLimits bad;
  bad.storage_resume_fraction = bad.storage_pause_fraction;
  refuses([&] { ResourcePressureMonitor invalid(mem.string(), block.string(), bad); });
}
DGPP_TEST(expert_cache_resource_pressure_defers_only_background_work) {
  for (const auto policy : {Policy::kWholeExpert, Policy::kAdaptiveFractional}) {
    Fixture f;
    ResourcePressure pressure;
    pressure.memory_pressure = true;
    ExpertCache cache(f.store, f.config(policy), host(f), [&] { return pressure; });
    cache.initialize();
    const auto before = cache.residency().chunks;
    const auto reads = f.store->stats().calls;
    for (int i = 0; i < 32; ++i) cache.observe(std::array<size_t, 1>{0});
    cache.adapt();
    require(cache.prefetch(0) == 0 && cache.residency().chunks == before && f.store->stats().calls == reads,
            "no background transition or prefetch under pressure");
    auto lease = cache.acquire(0);
    check_bytes(lease, f);
    lease.release();
    require(f.store->stats().calls > reads, "demand still loads all missing bytes under pressure");
    if (policy == Policy::kWholeExpert) require(cache.stats().demand_admissions == 1, "demand admission still reuses required reads");
    pressure.memory_pressure = false;
    pressure.storage_pressure = true;
    cache.adapt();
    require(cache.stats().adaptations_deferred == 2 && cache.stats().prefetches_deferred == 1 && cache.telemetry().pressure.storage_pressure,
            "storage pressure has the same bounded background gate and visible counters");
    for (int i = 0; i < 32; ++i) cache.observe(std::array<size_t, 1>{1});
    pressure.storage_pressure = false;
    const auto demand_reads = f.store->stats().calls;
    cache.adapt();
    require(f.store->stats().calls > demand_reads && !cache.telemetry().pressure.memory_pressure && !cache.telemetry().pressure.storage_pressure,
            "background promotions resume after pressure clears");
  }
}
int main() { return dgpp::test::run_all(); }

DGPP_TEST(expert_cache_demand_io_overlaps_current_chunk_completion_with_two_buffers) {
  for (const auto format : {Format::kMxfp4, Format::kNvfp4}) {
    for (const auto policy : {Policy::kWholeExpert, Policy::kAdaptiveFractional}) {
      Fixture f(format, true);
      auto cfg = f.config(policy);
      const auto record_bytes = f.store->records().front().layout.record_bytes;
      cfg.reads.buffers = 2;
      cfg.reads.staging_budget_bytes = 2 * record_bytes;
      cfg.planner.budget_bytes -= record_bytes;
      auto control = std::make_shared<Control>();
      ExpertCache cache(f.store, cfg, host(f, control));
      cache.initialize();
      const auto initial_chunks = cache.residency().chunks[0];
      require(initial_chunks <= 2, "at least two demand chunks exercise overlap");
      const auto calls = f.store->stats().calls;
      bool overlapped = false;
      control->before_finish = [&] {
        if (!control->staged_assemblies || overlapped) return;
        // Simulate a backend copy still reading the first staging buffer.
        // A serial loader cannot complete a second file read before this
        // callback returns. The other staging buffer must make progress.
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (f.store->stats().calls < calls + 2 && std::chrono::steady_clock::now() < deadline)
          std::this_thread::yield();
        require(f.store->stats().calls >= calls + 2,
                "next demand read completes before the current GPU copy fence");
        overlapped = true;
      };
      auto lease = cache.acquire(0);
      control->before_finish = {};
      require(overlapped, "bounded demand pipeline was exercised");
      check_bytes(lease, f, 0);
      require(f.store->stats().calls == calls + 4 - initial_chunks,
              "look-ahead coalesces without duplicate file reads");
      require(cache.read_stats().staging_peak <= cache.memory().staging_bytes,
              "overlap stays inside the preallocated staging budget");
      lease.release();
    }
  }
}

DGPP_TEST(expert_cache_parallel_demand_window_preserves_bytes_and_staging_reserve) {
  for (const auto format : {Format::kMxfp4, Format::kNvfp4}) {
    for (const auto policy : {Policy::kWholeExpert, Policy::kAdaptiveFractional}) {
      Fixture f(format, true);
      auto cfg = f.config(policy);
      const auto record_bytes = cfg.reads.staging_budget_bytes / cfg.reads.buffers;
      cfg.planner.budget_bytes += 5 * record_bytes;
      cfg.reads.buffers = 8;
      cfg.reads.workers = 4;
      cfg.reads.max_pending = 16;
      cfg.reads.staging_budget_bytes = 8 * record_bytes;
      auto control = std::make_shared<Control>();
      ExpertCache cache(f.store, cfg, host(f, control));
      cache.initialize();
      const auto calls = f.store->stats().calls;
      const auto missing = 4 - cache.residency().chunks[0];
      bool pipelined = false;
      control->before_finish = [&] {
        if (!control->staged_assemblies || pipelined) return;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (f.store->stats().calls < calls + missing && std::chrono::steady_clock::now() < deadline)
          std::this_thread::yield();
        require(f.store->stats().calls == calls + missing,
                "all missing chunks read while the first device copy is still consuming staging");
        pipelined = true;
      };
      { auto lease = cache.acquire(0); check_bytes(lease, f, 0); }
      control->before_finish = {};
      require(pipelined, "parallel demand window was exercised");
      (void)cache.prefetch(1);  // independent tickets must keep their staging reserve
      for (size_t expert : {size_t{0}, size_t{2}, size_t{3}, size_t{0}}) {
        cache.observe(std::array<size_t, 1>{expert});
        auto lease = cache.acquire(expert);
        check_bytes(lease, f, expert);
        require(cache.read_stats().staging_in_use <= cache.memory().staging_bytes,
                "parallel read window stays inside preallocated staging");
      }
      require(cache.memory().allocated_bytes() <= cfg.planner.budget_bytes,
              "parallel reads remain charged to the common expert budget");
    }
  }
}

DGPP_TEST(expert_cache_ready_chunks_share_a_fence_and_survive_a_failed_fence) {
  for (const auto format : {Format::kMxfp4, Format::kNvfp4}) {
    for (const auto policy : {Policy::kWholeExpert, Policy::kAdaptiveFractional}) {
      for (const bool fail : {false, true}) {
        Fixture f(format, true);
        auto control = std::make_shared<Control>();
        ExpertCache cache(f.store, f.config(policy), host(f, control));
        cache.initialize();
        const auto completed = cache.read_stats().completed;
        require(cache.prefetch(0) == 2, "two ready staging sources exercise batched copies");
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (cache.read_stats().completed < completed + 2 && std::chrono::steady_clock::now() < deadline)
          std::this_thread::yield();
        require(cache.read_stats().completed == completed + 2, "prefetched reads completed");
        bool batched = false;
        control->before_finish = [&] {
          if (!control->staged_assemblies || batched) return;
          require(control->staged_assemblies >= 2, "ready chunks share the first copy fence");
          require(cache.read_stats().staging_in_use >= 2 * cache.memory().staging_bytes / 3,
                  "every pending copy retains its staging source through the fence");
          batched = true;
          control->fail_finish = fail;
        };
        if (fail) {
          refuses([&] { (void)cache.acquire(0); });
          require(cache.stats().poisoned && cache.read_stats().staging_in_use >= 2 * cache.memory().staging_bytes / 3,
                  "failed batch keeps every source alive and poisons the cache");
          refuses([&] { (void)cache.acquire(0); });
        } else {
          auto lease = cache.acquire(0);
          check_bytes(lease, f, 0);
          require(cache.read_stats().staging_in_use == 0, "successful batch releases its sources");
        }
        require(batched, "batched copy fence was exercised");
        control->before_finish = {};
        control->fail_finish = false;
      }
    }
  }
}
