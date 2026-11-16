#pragma once
// CUDA-free geometry and desired-residency planning. This does not own weight
// buffers, issue I/O, or make an uncompleted load available to an executor.
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace dgpp::expert {

enum class Format { kMxfp4, kNvfp4 };
enum class Policy { kWholeExpert, kAdaptiveFractional };

struct MatrixLayout {
  uint64_t rows = 0, cols = 0;
  uint64_t payload_offset = 0, payload_bytes = 0;
  uint64_t scales_offset = 0, scales_bytes = 0;
  std::optional<uint64_t> global_offset;
};

struct ChunkLayout {
  // The global intermediate coordinate, including this rank's TP offset.
  uint64_t intermediate_begin = 0, intermediate_size = 0;
  MatrixLayout gate, up, down;
  uint64_t useful_bytes = 0;
  uint64_t allocation_bytes = 0;  // 256-byte aligned grants, including globals
  uint64_t record_bytes = 0;      // allocation rounded to 4 KiB for direct I/O
};

// Partition gate/up ROWS and down COLUMNS together. Source strided-column
// packing is the store builder's job. No weights are converted or requantized.
std::vector<ChunkLayout> partition(uint64_t hidden, uint64_t intermediate,
                                   uint64_t world, uint64_t rank,
                                   uint64_t chunk_intermediate, Format format);

struct MatrixSource {
  uint64_t rows = 0, cols = 0;
  Format format = Format::kMxfp4;
  std::span<const uint8_t> payload, scales;
  std::optional<float> global;
};
struct PackedChunk {
  ChunkLayout layout;
  std::vector<uint8_t> bytes;  // allocation_bytes; disk padding is separate
};
struct ScatterCopy {
  uint64_t source_offset = 0, destination_offset = 0;
  uint64_t row_bytes = 0, rows = 0, source_pitch = 0, destination_pitch = 0;
};
// Validated pitched copies, shared by the host reference and CUDA transfers.
// An NVFP4 triple adds one global-scale copy per matrix.
std::vector<ScatterCopy> scatter_copies(const ChunkLayout& chunk, uint64_t hidden,
    uint64_t intermediate, Format format, uint64_t destination_intermediate_begin = 0);
// Sources are gate, up, down in checkpoint row-major order. Validates their
// complete geometry before allocating. Copies codes/scales/globals unchanged.
PackedChunk pack_chunk(const std::array<MatrixSource, 3>& source,
                        uint64_t intermediate_begin, uint64_t intermediate_size);

// Copies one packed chunk into a full intermediate-width triple. The caller
// owns the bounded execution slot and its lifetime. The layout is validated
// independently; complete coverage and chunk completion are the caller's job.
void scatter_chunk(const PackedChunk& chunk, uint64_t hidden, uint64_t intermediate,
                    Format format, std::span<uint8_t> destination,
                    uint64_t destination_intermediate_begin = 0);
void scatter_chunk(const ChunkLayout& layout, std::span<const uint8_t> bytes,
                    uint64_t hidden, uint64_t intermediate, Format format,
                    std::span<uint8_t> destination, uint64_t destination_intermediate_begin = 0);

struct ExpertSpec {
  // Sizes charged to the persistent cache; contain allocation padding. The
  // runtime resolves the vector index to its (model, layer, expert, rank) key.
  std::vector<uint64_t> chunk_bytes;
  double estimated_load_ms = 1.0;  // full expert, updated from measured reads
  uint64_t history_group = 0;  // shared clock, normally one model/layer
};

struct PlannerConfig {
  bool operator==(const PlannerConfig&) const = default;
  Policy policy = Policy::kWholeExpert;
  uint64_t budget_bytes = 0;
  // Complete preallocated staging + execution reservation, counted once.
  uint64_t reserved_bytes = 0;
  double minimum_fraction = 0.05;  // zero permits cold experts to lose all coverage
  std::optional<double> initial_fraction;
  double decay_routes = 1024;  // exponential half-life in routed selections
  double horizon_routes = 8192;
  double prefill_route_weight = 1.0;  // weight in both popularity and ageing clock
  bool layer_local_history = false;  // experimental; scale local clocks by group count
  bool learn_load_cost = true;  // false retains each expert's initial cost estimate
  double promotion_margin = 0.25;
  uint64_t cooldown_routes = 128;
  uint64_t adaptation_bytes = UINT64_MAX;  // newly retained bytes per plan
};

struct ResidencyPlan {
  // Prefix chunk count per expert. Whole policy returns either 0 or all.
  std::vector<size_t> chunks;
  uint64_t cache_bytes = 0;
  uint64_t reserved_bytes = 0;
  uint64_t load_bytes = 0;  // newly retained bytes, not a claim of actual I/O
  uint64_t release_bytes = 0;
  uint64_t observations = 0;
  uint64_t generation = 0;  // identifies the committed allocation being changed
};

class ResidencyPlanner {
 public:
  ResidencyPlanner(PlannerConfig config, std::vector<ExpertSpec> experts);
  const ResidencyPlan& current() const { return current_; }
  uint64_t observations() const { return epoch_; }
  // Each element is a selected expert index, including repeated selections
  // across concurrent rows. Invalid input leaves the estimator unchanged.
  void observe(std::span<const size_t> selected, bool prefill = false);
  void observe_load(size_t expert, double full_expert_ms);
  // leased is a per-expert prefix lower bound supplied by the cache manager.
  // In-flight/GPU readers cannot be demoted by a desired-residency change.
  ResidencyPlan propose(std::span<const size_t> leased = {}) const;
  // Demand admission reuses bytes already needed for execution. Whole policy
  // admits complete experts; fractional policy can grow a resident prefix by
  // evicting lower-value tails, preserving configured floors and cooldown.
  // Neither policy consumes the periodic speculative-loading byte allowance.
  ResidencyPlan propose_admission(size_t expert) const;
  void commit_admission(size_t expert, const ResidencyPlan& plan);
  // The manager calls commit only AFTER its transition is complete. A plan
  // is not an I/O completion receipt. Stale or malformed plans are refused.
  void commit(const ResidencyPlan& plan);

 private:
  struct History {
    double accesses = 0;
    double updated = 0;
    uint64_t changed = 0;
    bool has_changed = false;
  };
  uint64_t prefix_bytes(size_t expert, size_t chunks) const;
  size_t minimum_chunks(size_t expert) const;
  double frequency(size_t expert) const;
  ResidencyPlan summarize(std::vector<size_t> chunks) const;
  void validate(const ResidencyPlan& plan) const;
  void commit_validated(const ResidencyPlan& plan);

  PlannerConfig config_;
  std::vector<ExpertSpec> experts_;
  std::vector<std::vector<uint64_t>> prefix_;
  std::vector<History> history_;
  std::vector<size_t> history_groups_;
  std::vector<double> group_epochs_;
  double history_epoch_ = 0;
  ResidencyPlan current_;
  uint64_t epoch_ = 0;
  uint64_t generation_ = 0;
};

}  // namespace dgpp::expert
