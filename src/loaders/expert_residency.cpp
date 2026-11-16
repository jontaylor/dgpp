#include "loaders/expert_residency.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <queue>
#include <stdexcept>
#include <utility>

namespace dgpp::expert {
namespace {
uint64_t add(uint64_t a, uint64_t b) {
  if (b > std::numeric_limits<uint64_t>::max() - a)
    throw std::overflow_error("expert residency: byte count overflow");
  return a + b;
}
uint64_t mul(uint64_t a, uint64_t b) {
  if (a && b > std::numeric_limits<uint64_t>::max() / a)
    throw std::overflow_error("expert residency: byte count overflow");
  return a * b;
}
uint64_t align(uint64_t bytes, uint64_t alignment) {
  const auto tail = bytes % alignment;
  return tail ? add(bytes, alignment - tail) : bytes;
}
bool fraction(double f) { return std::isfinite(f) && f > 0 && f <= 1; }
size_t fraction_chunks(double f, size_t count) {
  if (f == 0) return 0;
  // Round UP: a configured minimum is a guarantee, not a target to undershoot.
  return std::min(count, std::max(size_t{1}, static_cast<size_t>(std::ceil(f * count))));
}
MatrixLayout matrix(uint64_t rows, uint64_t cols, Format format, uint64_t& cursor,
                    uint64_t& useful) {
  MatrixLayout m;
  m.rows = rows;
  m.cols = cols;
  const uint64_t group = format == Format::kMxfp4 ? 32 : 16;
  m.payload_bytes = mul(rows, cols / 2);
  m.scales_bytes = mul(rows, cols / group);
  m.payload_offset = cursor;
  cursor = add(cursor, align(m.payload_bytes, 256));
  m.scales_offset = cursor;
  cursor = add(cursor, align(m.scales_bytes, 256));
  useful = add(useful, add(m.payload_bytes, m.scales_bytes));
  if (format == Format::kNvfp4) {
    m.global_offset = cursor;
    cursor = add(cursor, 256);
    useful = add(useful, sizeof(float));
  }
  return m;
}
ChunkLayout layout(uint64_t hidden, uint64_t begin, uint64_t width, Format format) {
  ChunkLayout c;
  c.intermediate_begin = begin;
  c.intermediate_size = width;
  uint64_t cursor = 0;
  c.gate = matrix(width, hidden, format, cursor, c.useful_bytes);
  c.up = matrix(width, hidden, format, cursor, c.useful_bytes);
  c.down = matrix(hidden, width, format, cursor, c.useful_bytes);
  c.allocation_bytes = cursor;
  c.record_bytes = align(cursor, 4096);
  return c;
}
bool same_matrix(const MatrixLayout& a, const MatrixLayout& b) {
  return a.rows == b.rows && a.cols == b.cols && a.payload_offset == b.payload_offset &&
         a.payload_bytes == b.payload_bytes && a.scales_offset == b.scales_offset &&
         a.scales_bytes == b.scales_bytes && a.global_offset == b.global_offset;
}
bool same_layout(const ChunkLayout& a, const ChunkLayout& b) {
  return a.intermediate_begin == b.intermediate_begin && a.intermediate_size == b.intermediate_size &&
         a.allocation_bytes == b.allocation_bytes && a.record_bytes == b.record_bytes &&
         a.useful_bytes == b.useful_bytes && same_matrix(a.gate, b.gate) &&
         same_matrix(a.up, b.up) && same_matrix(a.down, b.down);
}
void check_geometry(uint64_t hidden, uint64_t inter, Format format) {
  if ((format != Format::kMxfp4 && format != Format::kNvfp4) || !hidden || hidden % 32 || !inter || inter % 32)
    throw std::invalid_argument("expert residency: invalid FP4 expert geometry");
}
void check_source(const MatrixSource& s) {
  check_geometry(s.cols, s.rows, s.format);
  const uint64_t group = s.format == Format::kMxfp4 ? 32 : 16;
  if (s.payload.size() != mul(s.rows, s.cols / 2) || s.scales.size() != mul(s.rows, s.cols / group) ||
      s.global.has_value() != (s.format == Format::kNvfp4))
    throw std::invalid_argument("expert residency: source bytes/scales do not match matrix geometry");
  if (s.global && (!std::isfinite(*s.global) || *s.global <= 0))
    throw std::invalid_argument("expert residency: invalid NVFP4 source global");
}
}  // namespace

std::vector<ChunkLayout> partition(uint64_t hidden, uint64_t intermediate, uint64_t world,
                                   uint64_t rank, uint64_t width, Format format) {
  if (format != Format::kMxfp4 && format != Format::kNvfp4)
    throw std::invalid_argument("expert residency: unknown FP4 format");
  if (!hidden || hidden % 32 || !world || rank >= world || !intermediate || intermediate % world)
    throw std::invalid_argument("expert residency: invalid hidden or TP geometry");
  const uint64_t local = intermediate / world;
  // 32 also preserves NVFP4's 16-value scales and packed-byte boundaries.
  if (!width || width % 32 || local % 32)
    throw std::invalid_argument("expert residency: intermediate slices must align to 32 values");
  const uint64_t count = local / width + (local % width != 0);
  std::vector<ChunkLayout> result;
  if (count > result.max_size()) throw std::overflow_error("expert residency: too many chunks");
  result.reserve(static_cast<size_t>(count));
  uint64_t begin = 0;
  while (begin < local) {
    ChunkLayout c = layout(hidden, add(mul(rank, local), begin), std::min(width, local - begin), format);
    result.push_back(c);
    begin = add(begin, c.intermediate_size);
  }
  return result;
}

PackedChunk pack_chunk(const std::array<MatrixSource, 3>& source, uint64_t begin, uint64_t width) {
  for (const auto& s : source) check_source(s);
  const uint64_t hidden = source[0].cols, intermediate = source[0].rows;
  const Format format = source[0].format;
  if (source[1].rows != intermediate || source[1].cols != hidden || source[2].rows != hidden ||
      source[2].cols != intermediate || source[1].format != format || source[2].format != format ||
      !width || width % 32 || begin % 32 || begin >= intermediate || width > intermediate - begin)
    throw std::invalid_argument("expert residency: inconsistent SwiGLU triple or chunk interval");
  PackedChunk packed;
  packed.layout = layout(hidden, begin, width, format);
  if (packed.layout.allocation_bytes > packed.bytes.max_size())
    throw std::overflow_error("expert residency: packed chunk too large");
  packed.bytes.resize(static_cast<size_t>(packed.layout.allocation_bytes));
  const std::array<MatrixLayout, 3> dest{packed.layout.gate, packed.layout.up, packed.layout.down};
  const uint64_t group = format == Format::kMxfp4 ? 32 : 16;
  for (size_t i = 0; i < source.size(); ++i) {
    const auto& s = source[i];
    const auto& d = dest[i];
    if (i < 2) {
      std::memcpy(packed.bytes.data() + d.payload_offset, s.payload.data() + mul(begin, hidden / 2), d.payload_bytes);
      std::memcpy(packed.bytes.data() + d.scales_offset, s.scales.data() + mul(begin, hidden / group), d.scales_bytes);
    } else {
      for (uint64_t r = 0; r < hidden; ++r) {
        std::memcpy(packed.bytes.data() + d.payload_offset + mul(r, width / 2),
                    s.payload.data() + mul(r, intermediate / 2) + begin / 2, width / 2);
        std::memcpy(packed.bytes.data() + d.scales_offset + mul(r, width / group),
                    s.scales.data() + mul(r, intermediate / group) + begin / group, width / group);
      }
    }
    if (d.global_offset)
      std::memcpy(packed.bytes.data() + *d.global_offset, &*s.global, sizeof(float));
  }
  return packed;
}

std::vector<ScatterCopy> scatter_copies(const ChunkLayout& c, uint64_t hidden,
    uint64_t intermediate, Format format, uint64_t destination_begin) {
  check_geometry(hidden, intermediate, format);
  const uint64_t absolute_begin = c.intermediate_begin, width = c.intermediate_size;
  if (destination_begin % 32 || absolute_begin < destination_begin)
    throw std::invalid_argument("expert residency: chunk precedes reconstruction slice");
  const uint64_t begin = absolute_begin - destination_begin;
  if (!width || width % 32 || begin % 32 || begin >= intermediate || width > intermediate - begin)
    throw std::invalid_argument("expert residency: invalid reconstruction interval");
  if (!same_layout(c, layout(hidden, absolute_begin, width, format)))
    throw std::invalid_argument("expert residency: invalid packed layout");
  const auto full = layout(hidden, 0, intermediate, format);
  const std::array<MatrixLayout, 3> sources{c.gate, c.up, c.down}, dest{full.gate, full.up, full.down};
  const uint64_t group = format == Format::kMxfp4 ? 32 : 16;
  std::vector<ScatterCopy> copies;
  copies.reserve(9);
  for (size_t i = 0; i < sources.size(); ++i) {
    const auto& a = sources[i];
    const auto& b = dest[i];
    if (i < 2) {
      copies.push_back({a.payload_offset, b.payload_offset + mul(begin, hidden / 2),
                        a.payload_bytes, 1, a.payload_bytes, a.payload_bytes});
      copies.push_back({a.scales_offset, b.scales_offset + mul(begin, hidden / group),
                        a.scales_bytes, 1, a.scales_bytes, a.scales_bytes});
    } else {
      copies.push_back({a.payload_offset, b.payload_offset + begin / 2,
                        width / 2, hidden, width / 2, intermediate / 2});
      copies.push_back({a.scales_offset, b.scales_offset + begin / group,
                        width / group, hidden, width / group, intermediate / group});
    }
    if (b.global_offset)
      copies.push_back({*a.global_offset, *b.global_offset, sizeof(float), 1, sizeof(float), sizeof(float)});
  }
  return copies;
}

void scatter_chunk(const ChunkLayout& c, std::span<const uint8_t> bytes, uint64_t hidden,
    uint64_t intermediate, Format format, std::span<uint8_t> destination, uint64_t destination_begin) {
  const auto copies = scatter_copies(c, hidden, intermediate, format, destination_begin);
  const auto full = layout(hidden, 0, intermediate, format);
  if (bytes.size() != c.allocation_bytes || destination.size() != full.allocation_bytes)
    throw std::invalid_argument("expert residency: invalid packed/reconstruction extent");
  const auto src_address = reinterpret_cast<uintptr_t>(bytes.data());
  const auto dst_address = reinterpret_cast<uintptr_t>(destination.data());
  if (src_address <= dst_address ? dst_address - src_address < bytes.size()
                                 : src_address - dst_address < destination.size())
    throw std::invalid_argument("expert residency: reconstruction source aliases destination");
  for (const auto& copy : copies)
    for (uint64_t row = 0; row < copy.rows; ++row)
      std::memcpy(destination.data() + copy.destination_offset + row * copy.destination_pitch,
                  bytes.data() + copy.source_offset + row * copy.source_pitch, copy.row_bytes);
}

void scatter_chunk(const PackedChunk& chunk, uint64_t hidden, uint64_t intermediate,
    Format format, std::span<uint8_t> destination, uint64_t destination_begin) {
  scatter_chunk(chunk.layout, chunk.bytes, hidden, intermediate, format, destination, destination_begin);
}

ResidencyPlanner::ResidencyPlanner(PlannerConfig cfg, std::vector<ExpertSpec> experts)
    : config_(cfg), experts_(std::move(experts)), history_(experts_.size()),
      decode_history_(cfg.decode_admission_boost > 0 ? experts_.size() : 0) {
  if (cfg.policy != Policy::kWholeExpert && cfg.policy != Policy::kAdaptiveFractional)
    throw std::invalid_argument("expert residency: unknown policy");
  if (cfg.policy == Policy::kWholeExpert && cfg.initial_fraction)
    throw std::invalid_argument("expert residency: whole-expert policy has no initial fractional residency");
  if (experts_.empty() || !cfg.budget_bytes || cfg.reserved_bytes > cfg.budget_bytes ||
      !std::isfinite(cfg.minimum_fraction) || cfg.minimum_fraction < 0 || cfg.minimum_fraction > 1 ||
      (cfg.initial_fraction && (!fraction(*cfg.initial_fraction) ||
                               *cfg.initial_fraction < cfg.minimum_fraction)) ||
      !std::isfinite(cfg.decay_routes) || cfg.decay_routes <= 0 ||
      !std::isfinite(cfg.horizon_routes) || cfg.horizon_routes <= 0 ||
      !std::isfinite(cfg.prefill_route_weight) || cfg.prefill_route_weight < 1.0 / 1024 || cfg.prefill_route_weight > 1 ||
      !std::isfinite(cfg.decode_admission_boost) || cfg.decode_admission_boost < 0 || cfg.decode_admission_boost > 256 ||
      !std::isfinite(cfg.decode_history_multiplier) || cfg.decode_history_multiplier < 1 || cfg.decode_history_multiplier > 256 ||
      !std::isfinite(cfg.promotion_margin) || cfg.promotion_margin < 0)
    throw std::invalid_argument("expert residency: invalid budget, fraction or adaptation setting");
  if (cfg.decode_admission_boost > 0)
    decode_decay_ = std::exp2(-1.0 / (cfg.decay_routes * cfg.decode_history_multiplier));
  std::map<uint64_t, size_t> groups;
  for (const auto& e : experts_) {
    const auto [it, inserted] = groups.emplace(e.history_group, groups.size());
    history_groups_.push_back(it->second);
  }
  group_epochs_.resize(groups.size());
  prefix_.reserve(experts_.size());
  uint64_t largest = 0;
  for (const auto& e : experts_) {
    if (e.chunk_bytes.empty() || !std::isfinite(e.estimated_load_ms) || e.estimated_load_ms <= 0)
      throw std::invalid_argument("expert residency: empty expert or invalid load estimate");
    std::vector<uint64_t> p{0};
    for (uint64_t bytes : e.chunk_bytes) {
      if (!bytes) throw std::invalid_argument("expert residency: empty chunk");
      p.push_back(add(p.back(), bytes));
    }
    largest = std::max(largest, p.back());
    prefix_.push_back(std::move(p));
  }
  const uint64_t capacity = cfg.budget_bytes - cfg.reserved_bytes;
  // Whole policy must fit at least one complete expert even when cold.
  if (cfg.policy == Policy::kWholeExpert && largest > capacity)
    throw std::invalid_argument("expert residency: cache cannot hold the largest complete expert");
  std::vector<size_t> counts(experts_.size(), 0);
  if (cfg.policy == Policy::kAdaptiveFractional) {
    uint64_t used = 0;
    for (size_t e = 0; e < counts.size(); ++e) {
      counts[e] = cfg.initial_fraction ? fraction_chunks(*cfg.initial_fraction, experts_[e].chunk_bytes.size())
                                       : std::max(size_t{1}, minimum_chunks(e));
      used = add(used, prefix_bytes(e, counts[e]));
    }
    if (used > capacity)
      throw std::invalid_argument("expert residency: budget cannot provide the rounded universal fraction");
    // Water-fill in increasing actual fraction order. Unequal experts are
    // charged by bytes, not a count times an assumed common expert size.
    if (!cfg.initial_fraction) {
      auto less_covered = [&](size_t a, size_t b) {
        const long double fa = static_cast<long double>(counts[a]) / experts_[a].chunk_bytes.size();
        const long double fb = static_cast<long double>(counts[b]) / experts_[b].chunk_bytes.size();
        return fa == fb ? a > b : fa > fb;
      };
      std::priority_queue<size_t, std::vector<size_t>, decltype(less_covered)> queue(less_covered);
      for (size_t e = 0; e < counts.size(); ++e)
        if (counts[e] < experts_[e].chunk_bytes.size()) queue.push(e);
      while (!queue.empty()) {
        const size_t e = queue.top();
        queue.pop();
        const uint64_t bytes = experts_[e].chunk_bytes[counts[e]];
        if (bytes > capacity - used) continue;
        used += bytes;
        ++counts[e];
        if (counts[e] < experts_[e].chunk_bytes.size()) queue.push(e);
      }
    }
  }
  current_.chunks.assign(experts_.size(), 0);
  auto initial = summarize(std::move(counts));
  validate(initial);
  current_ = std::move(initial);
}

uint64_t ResidencyPlanner::prefix_bytes(size_t expert, size_t chunks) const {
  return prefix_.at(expert).at(chunks);
}
size_t ResidencyPlanner::minimum_chunks(size_t expert) const {
  return config_.policy == Policy::kAdaptiveFractional
             ? fraction_chunks(config_.minimum_fraction, experts_[expert].chunk_bytes.size()) : 0;
}
double ResidencyPlanner::frequency(size_t expert) const {
  const History& h = history_[expert];
  const auto clock = config_.layer_local_history ? group_epochs_[history_groups_[expert]] : history_epoch_;
  const double scale = config_.layer_local_history ? static_cast<double>(group_epochs_.size()) : 1.0;
  return h.accesses * std::exp2(-static_cast<double>(clock - h.updated) * scale / config_.decay_routes);
}

double ResidencyPlanner::admission_frequency(size_t expert) const {
  const double base = frequency(expert);
  if (config_.decode_admission_boost == 0 || last_prefill_) return base;
  const auto& h = decode_history_[expert];
  return base + config_.decode_admission_boost * h.accesses * decode_scale_;
}

void ResidencyPlanner::observe(std::span<const size_t> selected, bool prefill) {
  const double weight = prefill ? config_.prefill_route_weight : 1.0;
  if (selected.size() > std::numeric_limits<uint64_t>::max() - epoch_)
    throw std::overflow_error("expert residency: observation clock overflow");
  for (size_t e : selected)
    if (e >= experts_.size()) throw std::out_of_range("expert residency: invalid observed expert");
  if (!selected.empty()) last_prefill_ = prefill;
  for (size_t e : selected) {
    ++epoch_;
    history_epoch_ += weight;
    if (config_.layer_local_history) group_epochs_[history_groups_[e]] += weight;
    auto& h = history_[e];
    h.accesses = frequency(e) + weight;
    h.updated = config_.layer_local_history ? group_epochs_[history_groups_[e]] : history_epoch_;
    if (!prefill && config_.decode_admission_boost > 0) {
      // Keep all decode counts in units of one common scale. This removes an
      // exponential per resident during each admission scan. Rebase rarely
      // to keep the reciprocal bounded; an underflowed decay correctly
      // drops all earlier counts before adding the current observation.
      decode_scale_ *= decode_decay_;
      if (decode_scale_ < 1e-12) {
        for (auto& d : decode_history_) d.accesses *= decode_scale_;
        decode_scale_ = 1;
      }
      auto& d = decode_history_[e];
      d.accesses += 1 / decode_scale_;
    }
  }
}
void ResidencyPlanner::observe_load(size_t expert, double ms) {
  if (expert >= experts_.size() || !std::isfinite(ms) || ms <= 0)
    throw std::invalid_argument("expert residency: invalid measured load cost");
  if (!config_.learn_load_cost) return;
  experts_[expert].estimated_load_ms = 0.75 * experts_[expert].estimated_load_ms + 0.25 * ms;
}

ResidencyPlan ResidencyPlanner::summarize(std::vector<size_t> counts) const {
  ResidencyPlan p;
  p.chunks = std::move(counts);
  p.observations = epoch_;
  p.generation = generation_;
  p.reserved_bytes = config_.reserved_bytes;
  for (size_t e = 0; e < experts_.size(); ++e) {
    const uint64_t bytes = prefix_bytes(e, p.chunks[e]);
    const uint64_t previous = prefix_bytes(e, current_.chunks[e]);
    p.cache_bytes = add(p.cache_bytes, bytes);
    if (bytes > previous) p.load_bytes = add(p.load_bytes, bytes - previous);
    else p.release_bytes = add(p.release_bytes, previous - bytes);
  }
  return p;
}

ResidencyPlan ResidencyPlanner::propose(std::span<const size_t> leased) const {
  if (!leased.empty() && leased.size() != experts_.size())
    throw std::invalid_argument("expert residency: wrong lease population");
  std::vector<size_t> counts(experts_.size());
  uint64_t used = 0, newly_loaded = 0;
  for (size_t e = 0; e < counts.size(); ++e) {
    const size_t pinned = leased.empty() ? 0 : leased[e];
    if (pinned > current_.chunks[e])
      throw std::invalid_argument("expert residency: lease exceeds committed residency");
    counts[e] = std::max(minimum_chunks(e), pinned);
    const auto& h = history_[e];
    if (h.has_changed && epoch_ - h.changed < config_.cooldown_routes)
      counts[e] = std::max(counts[e], current_.chunks[e]);
    if (config_.policy == Policy::kWholeExpert && counts[e]) counts[e] = experts_[e].chunk_bytes.size();
    used = add(used, prefix_bytes(e, counts[e]));
  }
  const uint64_t capacity = config_.budget_bytes - config_.reserved_bytes;
  if (used > capacity) throw std::logic_error("expert residency: pinned population exceeds budget");
  struct Candidate { double score; size_t expert; };
  auto lower = [](const Candidate& a, const Candidate& b) {
    return a.score == b.score ? a.expert > b.expert : a.score < b.score;
  };
  std::priority_queue<Candidate, std::vector<Candidate>, decltype(lower)> queue(lower);
  auto candidate = [&](size_t e) {
    const size_t next = config_.policy == Policy::kWholeExpert ? experts_[e].chunk_bytes.size() : counts[e] + 1;
    const uint64_t bytes = prefix_bytes(e, next) - prefix_bytes(e, counts[e]);
    const bool retained = next <= current_.chunks[e];
    const double cost = experts_[e].estimated_load_ms *
                        (static_cast<double>(bytes) / prefix_[e].back());
    // Normalize the decayed count to predicted accesses over the horizon.
    const double benefit = frequency(e) * std::log(2.0) / config_.decay_routes * config_.horizon_routes * cost;
    const double net = benefit - (retained ? 0 : cost * (1 + config_.promotion_margin));
    if (!std::isfinite(net)) throw std::overflow_error("expert residency: adaptation score overflow");
    return Candidate{net / static_cast<double>(bytes), e};
  };
  for (size_t e = 0; e < counts.size(); ++e)
    if (counts[e] < experts_[e].chunk_bytes.size()) queue.push(candidate(e));
  while (!queue.empty()) {
    const auto c = queue.top();
    queue.pop();
    const size_t e = c.expert;
    const size_t next = config_.policy == Policy::kWholeExpert ? experts_[e].chunk_bytes.size() : counts[e] + 1;
    const uint64_t bytes = prefix_bytes(e, next) - prefix_bytes(e, counts[e]);
    const uint64_t extra = next > current_.chunks[e]
        ? prefix_bytes(e, next) - prefix_bytes(e, std::max(counts[e], current_.chunks[e])) : 0;
    // Retained zero-benefit chunks may occupy otherwise unused capacity;
    // negative-benefit speculative loads are never started by the planner.
    if (c.score < 0 || bytes > capacity - used || extra > config_.adaptation_bytes - newly_loaded) continue;
    used += bytes;
    newly_loaded += extra;
    counts[e] = next;
    if (next < experts_[e].chunk_bytes.size()) queue.push(candidate(e));
  }
  auto result = summarize(std::move(counts));
  validate(result);
  return result;
}

void ResidencyPlanner::validate(const ResidencyPlan& p) const {
  if (p.chunks.size() != experts_.size() || p.observations != epoch_ || p.generation != generation_ ||
      p.reserved_bytes != config_.reserved_bytes)
    throw std::invalid_argument("expert residency: stale or malformed plan");
  for (size_t e = 0; e < p.chunks.size(); ++e) {
    if (p.chunks[e] > experts_[e].chunk_bytes.size() || p.chunks[e] < minimum_chunks(e) ||
        (config_.policy == Policy::kWholeExpert && p.chunks[e] && p.chunks[e] != experts_[e].chunk_bytes.size()))
      throw std::invalid_argument("expert residency: plan violates policy coverage");
  }
  const auto checked = summarize(p.chunks);
  if (checked.cache_bytes != p.cache_bytes || checked.load_bytes != p.load_bytes ||
      checked.release_bytes != p.release_bytes || p.cache_bytes > config_.budget_bytes - config_.reserved_bytes)
    throw std::invalid_argument("expert residency: inconsistent or over-budget plan");
}
ResidencyPlan ResidencyPlanner::propose_admission(size_t expert) const {
  if (expert >= experts_.size())
    throw std::invalid_argument("expert residency: demand admission requires a valid expert");
  auto counts = current_.chunks;
  const auto unchanged = [&] { return summarize(current_.chunks); };
  if (config_.policy == Policy::kAdaptiveFractional) {
    if (counts[expert] == experts_[expert].chunk_bytes.size()) return unchanged();
    // Demand bytes are already needed for execution, so admission has no
    // speculative read cost or periodic adaptation-byte limit. Compare the
    // expected saved load cost per byte against colder resident tails.
    const auto density = [&](size_t e) {
      return admission_frequency(e) * experts_[e].estimated_load_ms / static_cast<double>(prefix_[e].back());
    };
    const double benefit = density(expert);
    struct Victim { size_t expert; double density; };
    std::vector<Victim> victims;
    uint64_t available = config_.budget_bytes - config_.reserved_bytes - current_.cache_bytes;
    for (size_t e = 0; e < counts.size(); ++e) {
      const auto& h = history_[e];
      if (e == expert || counts[e] <= minimum_chunks(e) ||
          (h.has_changed && epoch_ - h.changed < config_.cooldown_routes)) continue;
      const double cost = density(e);
      if (!std::isfinite(benefit) || !std::isfinite(cost) || benefit <= cost * (1 + config_.promotion_margin)) continue;
      victims.push_back({e, cost});
      available = add(available, prefix_bytes(e, counts[e]) - prefix_bytes(e, minimum_chunks(e)));
    }
    const size_t previous = counts[expert];
    while (counts[expert] < experts_[expert].chunk_bytes.size()) {
      const auto bytes = experts_[expert].chunk_bytes[counts[expert]];
      if (bytes > available) break;
      available -= bytes;
      ++counts[expert];
    }
    if (counts[expert] == previous) return unchanged();
    // A demand usually needs only one donor. Build a minimum heap instead of
    // sorting the whole cache, and evaluate each decayed density only once.
    // The heap returns exactly the same density/expert-ID order as the sort.
    const auto more_valuable = [](const Victim& a, const Victim& b) {
      return a.density == b.density ? a.expert > b.expert : a.density > b.density;
    };
    std::make_heap(victims.begin(), victims.end(), more_valuable);
    const uint64_t required = prefix_bytes(expert, counts[expert]) - prefix_bytes(expert, previous);
    uint64_t free = config_.budget_bytes - config_.reserved_bytes - current_.cache_bytes;
    while (free < required && !victims.empty()) {
      const size_t e = victims.front().expert;
      std::pop_heap(victims.begin(), victims.end(), more_valuable);
      victims.pop_back();
      while (free < required && counts[e] > minimum_chunks(e)) {
        --counts[e];
        free = add(free, experts_[e].chunk_bytes[counts[e]]);
      }
    }
    auto result = summarize(std::move(counts));
    validate(result);
    return result;
  }
  if (counts[expert]) return unchanged();
  const uint64_t capacity = config_.budget_bytes - config_.reserved_bytes;
  const uint64_t required = prefix_[expert].back();
  uint64_t available = capacity - current_.cache_bytes;
  if (available < required) {
    struct Victim { size_t expert; double value, density; };
    std::vector<Victim> victims;
    // Expected avoided load cost over the common horizon, per byte. Demand
    // admission incurs no extra storage read: its staging bytes are already
    // needed for execution. Replacement still sacrifices resident reuse.
    const auto value = [&](size_t e) {
      return admission_frequency(e) * experts_[e].estimated_load_ms;
    };
    for (size_t e = 0; e < counts.size(); ++e) {
      const auto& h = history_[e];
      if (counts[e] && !(h.has_changed && epoch_ - h.changed < config_.cooldown_routes)) {
        const auto cost = value(e);
        victims.push_back({e, cost, cost / static_cast<double>(prefix_[e].back())});
      }
    }
    const auto more_valuable = [](const Victim& a, const Victim& b) {
      return a.density == b.density ? a.expert > b.expert : a.density > b.density;
    };
    std::make_heap(victims.begin(), victims.end(), more_valuable);
    double lost = 0;
    while (available < required && !victims.empty()) {
      const auto victim = victims.front();
      std::pop_heap(victims.begin(), victims.end(), more_valuable);
      victims.pop_back();
      counts[victim.expert] = 0;
      available = add(available, prefix_[victim.expert].back());
      lost += victim.value;
    }
    const double benefit = value(expert);
    if (available < required || !std::isfinite(benefit) || !std::isfinite(lost) ||
        benefit <= lost * (1 + config_.promotion_margin)) return unchanged();
  }
  counts[expert] = experts_[expert].chunk_bytes.size();
  auto result = summarize(std::move(counts));
  validate(result);
  return result;
}
void ResidencyPlanner::commit_admission(size_t expert, const ResidencyPlan& plan) {
  if (expert >= experts_.size())
    throw std::invalid_argument("expert residency: invalid demand admission commit");
  validate(plan);
  const bool changed = plan.chunks[expert] != current_.chunks[expert];
  if (plan.chunks[expert] < current_.chunks[expert] ||
      (changed && config_.policy == Policy::kWholeExpert &&
       (current_.chunks[expert] || plan.chunks[expert] != experts_[expert].chunk_bytes.size())))
    throw std::invalid_argument("expert residency: admission must grow its demanded expert within policy coverage");
  for (size_t e = 0; e < experts_.size(); ++e) {
    if (e == expert) continue;
    if (plan.chunks[e] > current_.chunks[e] || (!changed && plan.chunks[e] != current_.chunks[e]))
      throw std::invalid_argument("expert residency: admission may load only its demanded expert");
  }
  commit_validated(plan);
}
void ResidencyPlanner::commit(const ResidencyPlan& plan) {
  validate(plan);
  if (plan.load_bytes > config_.adaptation_bytes)
    throw std::invalid_argument("expert residency: transition exceeds adaptation-byte allowance");
  commit_validated(plan);
}
void ResidencyPlanner::commit_validated(const ResidencyPlan& plan) {
  if (generation_ == UINT64_MAX) throw std::overflow_error("expert residency: generation overflow");
  ResidencyPlan next = plan;  // allocation failure must not change history
  for (size_t e = 0; e < experts_.size(); ++e)
    if (plan.chunks[e] != current_.chunks[e]) {
      history_[e].changed = epoch_;
      history_[e].has_changed = true;
    }
  current_ = std::move(next);
  ++generation_;
}

}  // namespace dgpp::expert
