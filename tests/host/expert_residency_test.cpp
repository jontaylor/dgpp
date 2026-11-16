#include "loaders/expert_residency.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/test.hpp"

namespace {
using namespace dgpp::expert;
void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
template <typename F> void refuses(F f) {
  bool failed = false;
  try { f(); } catch (const std::exception&) { failed = true; }
  require(failed, "invalid request must be refused");
}
std::vector<ExpertSpec> equal_experts(size_t n, size_t chunks, uint64_t bytes = 100) {
  return std::vector<ExpertSpec>(n, ExpertSpec{std::vector<uint64_t>(chunks, bytes), 1.0});
}
PlannerConfig config(Policy policy, uint64_t cache) {
  PlannerConfig c;
  c.policy = policy;
  c.reserved_bytes = 400;
  c.budget_bytes = cache + c.reserved_bytes;
  c.minimum_fraction = 0.25;
  c.decay_routes = 32;
  c.horizon_routes = 4096;
  c.cooldown_routes = 0;
  return c;
}
void hot(ResidencyPlanner& p, size_t e, size_t n = 128) {
  const std::vector<size_t> routes(n, e);
  p.observe(routes);
}
void bounded(const ResidencyPlanner& p, const PlannerConfig& c) {
  require(p.current().cache_bytes + p.current().reserved_bytes <= c.budget_bytes, "hard byte budget");
}
}  // namespace

DGPP_TEST(expert_partition_matches_dsv41_tp_slices_and_format_bytes) {
  for (uint64_t world : {1, 2, 4}) {
    uint64_t total = 0;
    for (uint64_t rank = 0; rank < world; ++rank) {
      const auto chunks = partition(5120, 2304, world, rank, 64, Format::kMxfp4);
      require(chunks.size() == 36 / world, "all intermediate dimensions partitioned once");
      uint64_t next = rank * (2304 / world);
      for (const auto& c : chunks) {
        require(c.intermediate_begin == next, "rank-local intervals have no gap/overlap");
        next += c.intermediate_size;
        require(c.gate.rows == 64 && c.gate.cols == 5120 && c.up.rows == 64,
                "gate and up are row slices");
        require(c.down.rows == 5120 && c.down.cols == 64, "down is a matching column slice");
        require(c.useful_bytes == 522240, "MXFP4 payload plus E8M0 scale bytes");
        require(c.record_bytes == 524288 && c.allocation_bytes == 522240, "I/O padding is explicit");
        require(!c.gate.global_offset && !c.down.global_offset, "MXFP4 has no global scales");
        total += c.useful_bytes;
      }
    }
    require(total == 3ull * 2304 * 5120 / 2 + 3ull * 2304 * 5120 / 32, "TP byte conservation");
  }
  const auto nv = partition(5120, 2304, 4, 0, 64, Format::kNvfp4).front();
  require(nv.useful_bytes == 552972 && nv.gate.global_offset && nv.down.global_offset,
          "NVFP4 includes E4M3 per-16 scales and three source globals");
  require(nv.allocation_bytes == 553728 && nv.record_bytes == 557056, "globals and alignment are charged");
}

DGPP_TEST(expert_partition_rejects_invalid_geometry_and_overflow_and_keeps_aligned_tail) {
  refuses([] { (void)partition(5120, 2304, 0, 0, 64, Format::kMxfp4); });
  refuses([] { (void)partition(5120, 2304, 4, 4, 64, Format::kMxfp4); });
  refuses([] { (void)partition(5120, 2304, 5, 0, 64, Format::kMxfp4); });
  refuses([] { (void)partition(5121, 2304, 4, 0, 64, Format::kMxfp4); });
  refuses([] { (void)partition(5120, 2304, 4, 0, 48, Format::kNvfp4); });
  refuses([] { (void)partition(5120, 2304, 4, 0, 64, static_cast<Format>(99)); });
  refuses([] { (void)partition(UINT64_MAX - 31, 256, 1, 0, 256, Format::kMxfp4); });
  const auto tail = partition(5120, 2304, 4, 3, 128, Format::kMxfp4);
  require(tail.size() == 5 && tail.back().intermediate_size == 64 &&
              tail.back().intermediate_begin == 2240, "aligned tail is represented, never truncated");
}

DGPP_TEST(expert_chunk_packing_preserves_codes_scales_globals_and_tp_reconstruction) {
  std::mt19937 rng(1017);
  for (auto format : {Format::kMxfp4, Format::kNvfp4}) {
    const uint64_t group = format == Format::kMxfp4 ? 32 : 16;
    for (uint64_t hidden : {32, 64, 128}) {
      constexpr uint64_t inter = 192;
      std::array<std::vector<uint8_t>, 3> codes, scales;
      std::array<MatrixSource, 3> source;
      for (size_t i = 0; i < 3; ++i) {
        const uint64_t rows = i < 2 ? inter : hidden, cols = i < 2 ? hidden : inter;
        codes[i].resize(rows * cols / 2);
        scales[i].resize(rows * cols / group);
        for (auto& v : codes[i]) v = static_cast<uint8_t>(rng());
        // All scale codes, including the MXFP4 NaN sentinel, survive unchanged.
        for (auto& v : scales[i]) v = static_cast<uint8_t>(rng());
        source[i] = {rows, cols, format, codes[i], scales[i], std::nullopt};
        if (format == Format::kNvfp4) source[i].global = 1.25f + static_cast<float>(i);
      }
      for (uint64_t rank = 0; rank < 2; ++rank) {
        const uint64_t base = rank * (inter / 2), local = inter / 2;
        const auto full = pack_chunk(source, base, local);
        std::vector<uint8_t> reconstructed(full.bytes.size());
        const auto pieces = partition(hidden, inter, 2, rank, 64, format);
        require(pieces.size() == 2 && pieces.back().intermediate_size == 32, "rank-local aligned tail");
        for (auto it = pieces.rbegin(); it != pieces.rend(); ++it) {
          const auto chunk = pack_chunk(source, it->intermediate_begin, it->intermediate_size);
          const auto& c = chunk.layout;
          for (uint64_t row = 0; row < hidden; ++row) {
            const size_t dst_payload = c.down.payload_offset + row * c.intermediate_size / 2;
            const size_t src_payload = row * inter / 2 + c.intermediate_begin / 2;
            require(std::equal(chunk.bytes.begin() + dst_payload,
                               chunk.bytes.begin() + dst_payload + c.intermediate_size / 2,
                               codes[2].begin() + src_payload), "down columns match original strided bytes");
            const size_t dst_scale = c.down.scales_offset + row * c.intermediate_size / group;
            const size_t src_scale = row * inter / group + c.intermediate_begin / group;
            require(std::equal(chunk.bytes.begin() + dst_scale,
                               chunk.bytes.begin() + dst_scale + c.intermediate_size / group,
                               scales[2].begin() + src_scale), "down scale groups follow the same columns");
          }
          require(std::equal(chunk.bytes.begin() + c.gate.payload_offset,
                             chunk.bytes.begin() + c.gate.payload_offset + c.gate.payload_bytes,
                             codes[0].begin() + c.intermediate_begin * hidden / 2), "gate rows match checkpoint");
          if (format == Format::kNvfp4) {
            float value = 0;
            std::memcpy(&value, chunk.bytes.data() + *c.down.global_offset, sizeof(value));
            require(value == *source[2].global, "global scale is copied, never recomputed for the chunk");
          }
          scatter_chunk(chunk, hidden, local, format, reconstructed, base);
        }
        require(reconstructed == full.bytes, "out-of-order chunk assembly is byte-exact for the TP expert");
      }
    }
  }
}

DGPP_TEST(expert_chunk_packing_refuses_invalid_scales_ranges_and_forged_layouts) {
  std::vector<uint8_t> bytes(32 * 64 / 2), scales(32 * 64 / 32);
  const std::array<MatrixSource, 3> valid{{{64, 32, Format::kMxfp4, bytes, scales, {}},
                                         {64, 32, Format::kMxfp4, bytes, scales, {}},
                                         {32, 64, Format::kMxfp4, bytes, scales, {}}}};
  auto bad = valid;
  bad[2].scales = {};
  refuses([&] { (void)pack_chunk(bad, 0, 32); });
  bad = valid;
  bad[1].global = 1;
  refuses([&] { (void)pack_chunk(bad, 0, 32); });
  refuses([&] { (void)pack_chunk(valid, 16, 32); });
  refuses([&] { (void)pack_chunk(valid, 32, 64); });
  auto chunk = pack_chunk(valid, 0, 32);
  auto full = pack_chunk(valid, 0, 64);
  chunk.layout.down.scales_offset = UINT64_MAX;
  refuses([&] { scatter_chunk(chunk, 32, 64, Format::kMxfp4, full.bytes); });
  chunk = pack_chunk(valid, 0, 32);
  refuses([&] { scatter_chunk(chunk, 32, 32, Format::kMxfp4, chunk.bytes); });
  refuses([&] { scatter_chunk(chunk, 32, 64, Format::kMxfp4, full.bytes, 32); });
}

DGPP_TEST(expert_planner_fractional_initialization_uses_bytes_and_rounded_coverage) {
  auto c = config(Policy::kAdaptiveFractional, 800);
  ResidencyPlanner p(c, equal_experts(4, 4));
  require(p.current().chunks == std::vector<size_t>({2, 2, 2, 2}), "universal half residency");
  require(p.current().load_bytes == 800, "cold baseline population is a real loading cost");
  bounded(p, c);
  auto uneven = equal_experts(3, 4);
  uneven[1].chunk_bytes = {100, 200, 300, 400};
  uneven[2].chunk_bytes = {75, 75, 75, 75, 75, 75, 75, 75};
  ResidencyPlanner q(c, uneven);
  bounded(q, c);
  for (size_t e = 0; e < uneven.size(); ++e)
    require(q.current().chunks[e] >= std::ceil(uneven[e].chunk_bytes.size() * c.minimum_fraction),
            "minimum is per expert, including unequal chunk counts");
  c.initial_fraction = 0.5;
  c.budget_bytes = 799 + c.reserved_bytes;
  refuses([&] { ResidencyPlanner no_fit(c, equal_experts(4, 4)); });
  c.budget_bytes = 800 + c.reserved_bytes;
  ResidencyPlanner exact(c, equal_experts(4, 4));
  require(exact.current().chunks == std::vector<size_t>({2, 2, 2, 2}), "explicit fraction retained");
}

DGPP_TEST(expert_planner_rejects_unusable_budgets_and_invalid_configuration) {
  auto c = config(Policy::kAdaptiveFractional, 399);
  refuses([&] { ResidencyPlanner p(c, equal_experts(4, 4)); });
  c.policy = Policy::kWholeExpert;
  refuses([&] { ResidencyPlanner p(c, equal_experts(4, 4)); });
  c = config(Policy::kAdaptiveFractional, 800);
  c.minimum_fraction = -0.01;
  refuses([&] { ResidencyPlanner p(c, equal_experts(4, 4)); });
  c.minimum_fraction = std::numeric_limits<double>::quiet_NaN();
  refuses([&] { ResidencyPlanner p(c, equal_experts(4, 4)); });
  c = config(Policy::kAdaptiveFractional, 800);
  c.initial_fraction = 0.1;
  refuses([&] { ResidencyPlanner p(c, equal_experts(4, 4)); });
  c.initial_fraction.reset();
  refuses([&] { ResidencyPlanner p(c, {{std::vector<uint64_t>{UINT64_MAX, 1}, 1}}); });
  c.policy = Policy::kWholeExpert;
  c.initial_fraction = 0.5;
  refuses([&] { ResidencyPlanner p(c, equal_experts(4, 4)); });
}

DGPP_TEST(expert_planner_fractional_demand_promotes_hot_set_without_speculative_allowance) {
  auto c = config(Policy::kAdaptiveFractional, 800);
  c.minimum_fraction = 0;
  c.adaptation_bytes = 0;
  auto too_small = c;
  too_small.budget_bytes = too_small.reserved_bytes + 399;
  refuses([&] { ResidencyPlanner missing_initial_coverage(too_small, equal_experts(4, 4)); });
  ResidencyPlanner p(c, equal_experts(4, 4));
  require(p.current().chunks == std::vector<size_t>({2, 2, 2, 2}), "zero floor still starts with universal coverage");
  for (size_t e : {size_t{0}, size_t{1}}) {
    hot(p, e);
    auto plan = p.propose_admission(e);
    require(plan.chunks[e] == 4 && plan.load_bytes > 0, "already demanded chunks bypass periodic loading cap");
    refuses([&] { p.commit(plan); });
    p.commit_admission(e, plan);
    bounded(p, c);
  }
  require(p.current().chunks == std::vector<size_t>({4, 4, 0, 0}), "feasible repeated hot set becomes fully resident");
  c.minimum_fraction = .25;
  ResidencyPlanner floor(c, equal_experts(4, 4));
  for (size_t e : {size_t{0}, size_t{1}}) {
    hot(floor, e);
    floor.commit_admission(e, floor.propose_admission(e));
  }
  require(floor.current().chunks[2] >= 1 && floor.current().chunks[3] >= 1 &&
              floor.current().chunks[0] + floor.current().chunks[1] <= 6,
          "H plus cold minimum exceeds budget: cannot promise all hot experts fit");
}

DGPP_TEST(expert_planner_fractional_demand_preserves_hotter_residents_and_cooldown) {
  auto c = config(Policy::kAdaptiveFractional, 800);
  c.minimum_fraction = 0;
  c.cooldown_routes = 128;
  ResidencyPlanner p(c, equal_experts(4, 4));
  hot(p, 0);
  p.commit_admission(0, p.propose_admission(0));
  p.observe(std::array<size_t, 1>{3});
  const auto plan = p.propose_admission(3);
  require(plan.chunks[0] == 4, "demand never displaces a hotter resident or a protected promotion");
  p.commit_admission(3, plan);
  bounded(p, c);
  refuses([&] { (void)p.propose_admission(4); });
}

DGPP_TEST(expert_planner_whole_experts_are_indivisible_and_replace_stale_hot_set) {
  auto c = config(Policy::kWholeExpert, 800);
  ResidencyPlanner p(c, equal_experts(4, 4));
  require(p.current().cache_bytes == 0, "whole cache starts empty");
  hot(p, 0);
  p.commit(p.propose());
  require(p.current().chunks[0] == 4, "hot expert promoted completely");
  hot(p, 1);
  p.commit(p.propose());
  hot(p, 3, 1024);
  const auto next = p.propose();
  require(next.chunks[3] == 4, "changed hot set admitted");
  require(next.chunks[0] == 0, "decayed oldest hot expert evicted");
  for (auto n : next.chunks) require(n == 0 || n == 4, "no partially resident whole experts");
  p.commit(next);
  bounded(p, c);
}

DGPP_TEST(expert_planner_fractional_promotes_demotes_and_preserves_universal_floor) {
  auto c = config(Policy::kAdaptiveFractional, 800);
  ResidencyPlanner p(c, equal_experts(4, 4));
  hot(p, 0);
  auto next = p.propose();
  require(next.chunks[0] == 4, "hot expert reaches full residency");
  require(next.release_bytes > 0 && next.load_bytes > 0, "promotion funded by actual demotion bytes");
  p.commit(next);
  hot(p, 3, 1024);
  next = p.propose();
  require(next.chunks[3] == 4 && next.chunks[0] < 4, "old promotion demoted after workload changes");
  for (auto n : next.chunks) require(n >= 1, "cold experts retain chunks");
  p.commit(next);
  bounded(p, c);
}

DGPP_TEST(expert_planner_demand_admission_fills_then_replaces_by_reuse_cost) {
  auto c = config(Policy::kWholeExpert, 800);
  c.adaptation_bytes = 0;  // No speculative loads; demand bytes are already read.
  ResidencyPlanner p(c, equal_experts(4, 4));
  p.observe(std::vector<size_t>{0});
  auto first = p.propose_admission(0);
  refuses([&] { p.commit(first); });
  p.commit_admission(0, first);
  p.observe(std::vector<size_t>{1});
  p.commit_admission(1, p.propose_admission(1));
  require(p.current().cache_bytes == 800, "demand fills available capacity without periodic adaptation");
  // A single scan miss must not displace frequently reused cached weights.
  hot(p, 0);
  hot(p, 1);
  p.observe(std::vector<size_t>{2});
  auto scan = p.propose_admission(2);
  require(scan.chunks[2] == 0 && scan.release_bytes == 0, "weak demand bypasses a valuable hot population");
  hot(p, 2);
  auto next = p.propose_admission(2);
  require(next.chunks[2] == 4 && next.release_bytes == 400, "new hot demand replaces the least valuable whole expert");
  p.commit_admission(2, next);
  refuses([&] { p.commit_admission(2, next); });
  bounded(p, c);

  auto unequal = equal_experts(3, 4);
  unequal[2].chunk_bytes = {200, 200, 200, 200};
  ResidencyPlanner q(c, unequal);
  q.commit_admission(0, q.propose_admission(0));
  q.commit_admission(1, q.propose_admission(1));
  hot(q, 2);
  next = q.propose_admission(2);
  require(next.release_bytes == 800 && next.load_bytes == 800, "large demand can replace several smaller residents");
  q.commit_admission(2, next);
  require(q.current().chunks == std::vector<size_t>({0, 0, 4}), "unequal experts remain indivisible");
}

DGPP_TEST(expert_planner_demand_admission_preserves_cooldown_and_refuses_wrong_commit) {
  auto c = config(Policy::kWholeExpert, 400);
  c.cooldown_routes = 128;
  ResidencyPlanner p(c, equal_experts(3, 4));
  p.commit_admission(0, p.propose_admission(0));
  hot(p, 1, 64);
  require(p.propose_admission(1).chunks[1] == 0, "recent admission remains protected through cooldown");
  hot(p, 1, 128);
  auto next = p.propose_admission(1);
  require(next.chunks[1] == 4, "demand can replace after cooldown");
  refuses([&] { p.commit_admission(2, next); });
  p.commit_admission(1, next);
  refuses([&] { (void)p.propose_admission(3); });
  ResidencyPlanner fractional(config(Policy::kAdaptiveFractional, 800), equal_experts(4, 4));
  hot(fractional, 1);
  const auto other = fractional.propose_admission(1);
  refuses([&] { fractional.commit_admission(0, other); });
}

DGPP_TEST(expert_planner_leases_and_cooldown_prevent_unsafe_or_oscillating_demotions) {
  auto c = config(Policy::kAdaptiveFractional, 800);
  c.cooldown_routes = 256;
  ResidencyPlanner p(c, equal_experts(4, 4));
  hot(p, 0);
  p.commit(p.propose());
  hot(p, 3, 128);
  auto next = p.propose();
  require(next.chunks[0] == 4, "minimum residence interval holds old hot expert");
  hot(p, 3, 1024);
  const std::vector<size_t> leased{4, 0, 0, 0};
  next = p.propose(leased);
  require(next.chunks[0] == 4, "active GPU lease holds contents despite new workload");
  next = p.propose();
  require(next.chunks[0] < 4 && next.chunks[3] == 4, "released contents can be demoted");
  refuses([&] { (void)p.propose(std::vector<size_t>{5, 0, 0, 0}); });
  refuses([&] { (void)p.propose(std::vector<size_t>{0}); });
}

DGPP_TEST(expert_planner_limits_newly_retained_bytes_and_rejects_stale_or_forged_plans) {
  auto c = config(Policy::kAdaptiveFractional, 800);
  c.adaptation_bytes = 100;
  ResidencyPlanner p(c, equal_experts(4, 4));
  hot(p, 0);
  auto next = p.propose();
  require(next.chunks[0] == 3 && next.load_bytes <= 100, "adaptation-byte cap limits promotion");
  auto malformed = next;
  ++malformed.cache_bytes;
  refuses([&] { p.commit(malformed); });
  hot(p, 0, 1);
  refuses([&] { p.commit(next); });
  next = p.propose();
  p.commit(next);
  refuses([&] { p.commit(next); });
  const auto previous = p.observations();
  refuses([&] { p.observe(std::vector<size_t>{0, 999}); });
  require(p.observations() == previous, "invalid route batch is atomic");
  refuses([&] { p.observe_load(0, -1); });
}

DGPP_TEST(expert_planner_cost_aware_admission_distinguishes_equal_frequency_experts) {
  auto c = config(Policy::kWholeExpert, 400);
  auto specs = equal_experts(2, 4);
  specs[1].estimated_load_ms = 10;
  ResidencyPlanner p(c, specs);
  for (int n = 0; n < 256; ++n) p.observe(std::vector<size_t>{0, 1});
  require(p.propose().chunks[1] == 4, "expensive equally frequent miss is retained");
}

DGPP_TEST(expert_planner_fixed_cost_preserves_frequency_ranking_despite_load_samples) {
  for (bool learn : {false, true}) {
    auto c = config(Policy::kWholeExpert, 400);
    c.learn_load_cost = learn;
    ResidencyPlanner p(c, equal_experts(2, 4));
    p.observe(std::vector<size_t>{0, 0, 0, 1});
    p.observe_load(1, 100);
    const auto plan = p.propose();
    require(plan.chunks[learn ? 1 : 0] == 4,
            "fixed costs preserve popularity; learned costs may prefer the expensive expert");
    refuses([&] { p.observe_load(2, 1); });
    refuses([&] { p.observe_load(0, 0); });
  }
}

DGPP_TEST(expert_planner_random_workload_shift_maintains_budget_and_policy_invariants) {
  std::mt19937 rng(41);
  for (auto policy : {Policy::kWholeExpert, Policy::kAdaptiveFractional}) {
    auto c = config(policy, 2800);
    auto specs = equal_experts(12, 4);
    for (auto& s : specs)
      for (auto& bytes : s.chunk_bytes) bytes = 50 + rng() % 100;
    ResidencyPlanner p(c, specs);
    for (int window = 0; window < 500; ++window) {
      std::vector<size_t> routes;
      const size_t hot_id = (window / 50) % specs.size();
      for (int row = 0; row < 64; ++row)
        routes.push_back(rng() % 4 == 0 ? rng() % specs.size() : hot_id);
      p.observe(routes);
      const auto target = p.propose();
      p.commit(target);
      bounded(p, c);
      for (auto count : target.chunks)
        require(policy == Policy::kWholeExpert ? count == 0 || count == 4 : count >= 1 && count <= 4,
                "coverage invariant through locality changes");
    }
  }
}

DGPP_TEST(expert_planner_layer_history_preserves_popularity_across_layer_order) {
  auto c = config(Policy::kWholeExpert, 800);
  c.layer_local_history = true;
  auto specs = equal_experts(4, 4);
  specs[2].history_group = specs[3].history_group = 999;
  ResidencyPlanner forward(c, specs), reverse(c, specs), split(c, specs);
  std::vector<size_t> a, b;
  for (size_t i = 0; i < 128; ++i) {
    a.push_back(i % 4 ? 0 : 1);
    b.push_back(i % 4 ? 2 : 3);
  }
  forward.observe(a); forward.observe(b);
  reverse.observe(b); reverse.observe(a);
  for (size_t i = 0; i < a.size(); ++i) {
    split.observe(std::span<const size_t>(&a[i], 1));
    split.observe(std::span<const size_t>(&b[i], 1));
  }
  const auto plan = forward.propose();
  require(plan.chunks == reverse.propose().chunks && plan.chunks == split.propose().chunks,
          "cross-layer order and observe batching do not change local popularity");
  require(plan.chunks == std::vector<size_t>({4, 0, 4, 0}), "popular expert from each layer retained");
  refuses([&] { forward.observe(std::vector<size_t>{0, 999}); });
  require(forward.observations() == 256 && forward.propose().chunks == plan.chunks,
          "invalid observations leave both clocks unchanged");
  hot(forward, 1, 256); hot(forward, 3, 256);
  require(forward.propose().chunks == std::vector<size_t>({0, 4, 0, 4}), "local histories adapt to a workload switch");
}

DGPP_TEST(expert_planner_single_group_local_history_matches_global) {
  auto c = config(Policy::kWholeExpert, 800);
  auto specs = equal_experts(4, 4);
  ResidencyPlanner global(c, specs);
  c.layer_local_history = true;
  ResidencyPlanner local(c, specs);
  std::mt19937 rng(912);
  for (int step = 0; step < 200; ++step) {
    const std::vector<size_t> routes{rng() % 4, rng() % 4, rng() % 4};
    global.observe(routes); local.observe(routes);
    const auto a = global.propose(), b = local.propose();
    require(a.chunks == b.chunks, "single-group estimator identical under churn");
    global.commit(a); local.commit(b);
  }
}

DGPP_TEST(expert_planner_prefill_weight_preserves_decode_history_and_popularity) {
  auto c = config(Policy::kWholeExpert, 400);
  c.prefill_route_weight = 1.0 / 16;
  ResidencyPlanner weighted(c, equal_experts(3, 4));
  c.prefill_route_weight = 1;
  ResidencyPlanner control(c, equal_experts(3, 4));
  hot(weighted, 0, 128); hot(control, 0, 128);
  std::vector<size_t> scan;
  for (int i = 0; i < 64; ++i) scan.push_back(i % 4 ? 1 : 2);
  weighted.observe(scan, true); control.observe(scan, true);
  require(weighted.propose().chunks[0] == 4, "weighted scan preserves recent decode popularity");
  require(control.propose().chunks[1] == 4, "unweighted scan supplants decode history");
  require(weighted.observations() == control.observations(), "weight does not alter admission or adaptation clock");
  for (int i = 0; i < 32; ++i) weighted.observe(scan, true);
  require(weighted.propose().chunks[1] == 4, "repeated prefill learns relative popularity");
  const auto before = weighted.propose().chunks;
  refuses([&] { weighted.observe(std::vector<size_t>{0, 999}, true); });
  require(weighted.propose().chunks == before, "invalid weighted batch is atomic");
}

DGPP_TEST(expert_planner_weight_one_preserves_phase_independent_behavior) {
  auto c = config(Policy::kWholeExpert, 800);
  ResidencyPlanner decode(c, equal_experts(4, 4)), prefill(c, equal_experts(4, 4));
  std::mt19937 rng(173);
  for (int step = 0; step < 200; ++step) {
    const std::vector<size_t> routes{rng() % 4, rng() % 4, rng() % 4};
    decode.observe(routes); prefill.observe(routes, true);
    auto a = decode.propose(), b = prefill.propose();
    require(a.chunks == b.chunks, "default weighting is phase independent");
    decode.commit(a); prefill.commit(b);
  }
  for (double w : {0., 2., std::numeric_limits<double>::quiet_NaN()}) {
    c.prefill_route_weight = w;
    refuses([&] { ResidencyPlanner invalid(c, equal_experts(4, 4)); });
  }
}

DGPP_TEST(expert_planner_equal_size_donor_matches_general_heap_under_churn) {
  auto c = config(Policy::kWholeExpert, 2000);
  c.cooldown_routes = 17;
  auto specs = equal_experts(16, 4);
  for (size_t i = 0; i < specs.size(); ++i) specs[i].estimated_load_ms = 1.0 + (i % 5) * 0.2;
  auto unequal = specs;
  unequal.push_back(ExpertSpec{std::vector<uint64_t>(4, 200), 1.0});
  ResidencyPlanner optimized(c, specs), reference(c, unequal);
  std::mt19937 rng(7249);
  for (int step = 0; step < 1500; ++step) {
    std::vector<size_t> routes;
    for (int j = 0; j < 1 + static_cast<int>(rng() % 16); ++j)
      routes.push_back((rng() % 4) ? (step / 100) % 16 : rng() % 16);
    optimized.observe(routes); reference.observe(routes);
    const auto e = routes.back();
    const auto a = optimized.propose_admission(e), b = reference.propose_admission(e);
    require(std::equal(a.chunks.begin(), a.chunks.end(), b.chunks.begin()) && b.chunks.back() == 0 &&
                a.cache_bytes == b.cache_bytes && a.load_bytes == b.load_bytes && a.release_bytes == b.release_bytes,
            "equal-size donor selection matches general heap including ties, cooldown and workload shifts");
    optimized.commit_admission(e, a); reference.commit_admission(e, b);
  }
}

int main() { return dgpp::test::run_all(); }
