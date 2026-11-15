// Controlled storage/reconstruction and optional existing-expert-kernel
// diagnostic. No routing, attention or HTTP: not model token throughput.
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/resource.h>

#include "loaders/expert_cache_cuda.hpp"
#include "common/cuda_check.hpp"
#include "kernels/fp4_gemv.hpp"
#include "kernels/glm_moe_launch.hpp"
#include "serve/json_out.hpp"

namespace {
using namespace dgpp::expert;
using Clock = std::chrono::steady_clock;
uint64_t ns(Clock::time_point start) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
}
uint64_t process_read_bytes() {
  std::ifstream in("/proc/self/io");
  std::string key;
  uint64_t value;
  while (in >> key >> value) if (key == "read_bytes:") return value;
  throw std::runtime_error("cannot read process physical I/O accounting");
}
uint64_t peak_rss_bytes() {
  rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage)) throw std::runtime_error("getrusage failed");
  return static_cast<uint64_t>(usage.ru_maxrss) * 1024;
}
void number(std::string& out, const char* name, uint64_t value) {
  out += ",\""; out += name; out += "\":";
  dgpp::serve::append_json_uint(&out, value);
}
struct ComputeStats {
  uint64_t kernel_wall_ns = 0, readback_validation_ns = 0;
  uint64_t output_hash = 14695981039346656037ULL;
  std::vector<uint64_t> selection_ns;
};
struct DeviceFree {
  void operator()(void* p) const { if (p) (void)cudaFree(p); }
};
class ExpertCompute {
 public:
  ExpertCompute(int rows, const StoreIdentity& identity)
      : rows_(rows), hidden_(identity.hidden), intermediate_(identity.intermediate),
        full_(partition(identity.hidden, identity.intermediate, 1, 0,
                        identity.intermediate, identity.format).front()), format_(identity.format) {
    const size_t h = static_cast<size_t>(rows_) * hidden_;
    const size_t inter = static_cast<size_t>(rows_) * intermediate_;
    // Activation/output storage is separate from the common expert budget.
    bytes_ = (h + 3 * inter) * sizeof(uint16_t) + h * sizeof(float);
    void* ptr = nullptr;
    DGPP_CUDA_OK(cudaMalloc(&ptr, bytes_));
    storage_.reset(ptr);
    input_ = static_cast<uint16_t*>(ptr);
    gate_ = input_ + h; up_ = gate_ + inter; act_ = up_ + inter;
    output_ = reinterpret_cast<float*>(act_ + inter);
    host_.resize(h);
    std::vector<uint16_t> input(h);
    for (size_t i = 0; i < h; ++i)
      input[i] = static_cast<uint16_t>(0x3c00 + (i % 4) * 0x80 + ((i % 7 == 0) ? 0x8000 : 0));
    DGPP_CUDA_OK(cudaMemcpy(input_, input.data(), h * sizeof(uint16_t), cudaMemcpyHostToDevice));
  }
  uint64_t bytes() const { return bytes_; }
  void run(const ExpertExecutionLease& lease, ComputeStats& stats) {
    const auto matrices = execution_matrices(lease, full_, format_);
    for (const auto& matrix : matrices)
      if (!dgpp::fp4_gemv_accepts(matrix)) throw std::invalid_argument("expert shape unsupported by GEMV");
    auto tick = Clock::now();
    dgpp::launch_fp4_gemv_bf16(input_, hidden_, matrices[0], gate_, rows_, intermediate_, hidden_, nullptr);
    dgpp::launch_fp4_gemv_bf16(input_, hidden_, matrices[1], up_, rows_, intermediate_, hidden_, nullptr);
    dgpp::launch_moe_swiglu_clamp(gate_, up_, act_, static_cast<int64_t>(rows_) * intermediate_, 7.0f, nullptr);
    dgpp::launch_fp4_gemv_f32(act_, intermediate_, matrices[2], output_, rows_, hidden_, intermediate_, nullptr);
    DGPP_CUDA_OK(cudaStreamSynchronize(nullptr));
    stats.kernel_wall_ns += ns(tick);
    tick = Clock::now();
    DGPP_CUDA_OK(cudaMemcpy(host_.data(), output_, host_.size() * sizeof(float), cudaMemcpyDeviceToHost));
    for (float value : host_) {
      if (!std::isfinite(value)) throw std::runtime_error("nonfinite expert output");
      // Hash every output bit in a fixed byte order, across every selection.
      const auto bits = std::bit_cast<uint32_t>(value);
      for (unsigned shift = 0; shift < 32; shift += 8) {
        stats.output_hash ^= (bits >> shift) & 255;
        stats.output_hash *= 1099511628211ULL;
      }
    }
    stats.readback_validation_ns += ns(tick);
  }
 private:
  int rows_, hidden_, intermediate_;
  ChunkLayout full_;
  Format format_;
  uint64_t bytes_;
  std::unique_ptr<void, DeviceFree> storage_;
  uint16_t *input_, *gate_, *up_, *act_;
  float* output_;
  std::vector<float> host_;
};
void report(const std::string& phase, const CacheTelemetry& before,
            const CacheTelemetry& after, uint64_t wall_ns, uint64_t device_reads,
            std::vector<uint64_t> latency, const std::string& compute,
            uint64_t rows, uint64_t activation_bytes, ComputeStats stats = {}) {
  std::string out = "{\"phase\":";
  dgpp::serve::append_json_string(&out, phase);
  number(out, "wall_ns", wall_ns);
  number(out, "operations", latency.size());
  out += ",\"compute\":";
  dgpp::serve::append_json_string(&out, compute);
  number(out, "batch_rows", rows);
  number(out, "activation_output_device_bytes", activation_bytes);
  number(out, "kernel_launch_sync_wall_ns", stats.kernel_wall_ns);
  number(out, "output_readback_validation_ns", stats.readback_validation_ns);
  number(out, "output_fnv1a64", stats.output_hash);
  std::sort(stats.selection_ns.begin(), stats.selection_ns.end());
  std::sort(latency.begin(), latency.end());
  for (const auto pct : {50, 95, 99}) {
    const auto value = latency.empty() ? 0 : latency[(latency.size() * pct + 99) / 100 - 1];
    const auto name = "acquire_adapt_p" + std::to_string(pct) + "_ns";
    number(out, name.c_str(), value);
    if (!stats.selection_ns.empty()) {
      const auto selection_name = "selection_p" + std::to_string(pct) + "_ns";
      number(out, selection_name.c_str(), stats.selection_ns[(stats.selection_ns.size() * pct + 99) / 100 - 1]);
    }
  }
  // /proc/self/io includes all reads charged to this process and its threads,
  // including any driver activity. Never label it expert-only NVMe bytes.
  number(out, "process_physical_read_bytes", device_reads);
  number(out, "peak_process_rss_bytes", peak_rss_bytes());
  number(out, "expert_reserved_bytes", after.memory.allocated_bytes());
  number(out, "resident_bytes", after.resident_bytes);
#define DELTA(field) number(out, #field, after.cache.field - before.cache.field)
  DELTA(requests); DELTA(full_hits); DELTA(partial_hits);
  DELTA(chunk_lookups); DELTA(chunk_hits); DELTA(demand_bytes);
  DELTA(read_wait_ns); DELTA(reconstruction_wall_ns);
  DELTA(cached_copy_bytes); DELTA(staged_copy_bytes); DELTA(execution_zero_bytes);
  DELTA(native_hits); DELTA(native_packs); DELTA(native_unpacks);
  DELTA(direct_native_hits); DELTA(canonical_unpacks); DELTA(canonical_unpack_bytes);
  DELTA(canonical_unpack_enqueue_ns); DELTA(completion_wait_ns);
  DELTA(native_layout_wall_ns); DELTA(native_layout_copy_bytes);
  DELTA(promotions); DELTA(demotions); DELTA(evictions);
  DELTA(promotion_bytes); DELTA(demotion_bytes);
#undef DELTA
  number(out, "file_calls", after.file.calls - before.file.calls);
  number(out, "file_returned_bytes", after.file.returned_bytes - before.file.returned_bytes);
  out += '}';
  std::cout << out << std::endl;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    std::map<std::string, std::string> args;
    for (int i = 1; i < argc; i += 2) {
      if (i + 1 == argc || !args.emplace(argv[i], argv[i + 1]).second)
        throw std::invalid_argument("expected unique --option value pairs");
    }
    const auto take = [&](const char* key) {
      const auto it = args.find(key);
      if (it == args.end()) throw std::invalid_argument(std::string("missing ") + key);
      auto value = it->second; args.erase(it); return value;
    };
    const auto integer = [&](const char* key) {
      auto value = take(key);
      if (value.empty() || value.front() == '-') throw std::invalid_argument("expected positive integer");
      size_t used = 0;
      const uint64_t result = std::stoull(value, &used);
      if (used != value.size() || !result) throw std::invalid_argument("expected positive integer");
      return result;
    };
    StoreIdentity identity;
    identity.checkpoint_sha256.fill(0xB1);  // Fixed synthetic source recipe v1.
    identity.hidden = integer("--hidden");
    identity.intermediate = integer("--intermediate");
    identity.chunk_intermediate = integer("--chunk");
    const auto experts = integer("--experts");
    if (experts > 65536) throw std::invalid_argument("synthetic expert count exceeds 65536");
    const auto format = take("--format");
    if (format != "mxfp4" && format != "nvfp4") throw std::invalid_argument("unknown format");
    identity.format = format == "mxfp4" ? Format::kMxfp4 : Format::kNvfp4;
    CacheConfig config;
    config.planner.budget_bytes = integer("--budget-bytes");
    if (args.contains("--minimum-fraction")) {
      const auto value = take("--minimum-fraction");
      size_t used = 0;
      config.planner.minimum_fraction = std::stod(value, &used);
      if (used != value.size() || !std::isfinite(config.planner.minimum_fraction) ||
          config.planner.minimum_fraction < 0 || config.planner.minimum_fraction > 1)
        throw std::invalid_argument("minimum fraction must be in [0,1]");
    }
    config.planner.cooldown_routes = 128;
    config.planner.adaptation_bytes = 64 * 1024 * 1024;
    const auto adapt_every = integer("--adapt-every");
    const auto policy = take("--policy");
    if (policy != "whole_expert_cache" && policy != "adaptive_fractional_cache")
      throw std::invalid_argument("unknown policy");
    config.planner.policy = policy == "whole_expert_cache" ? Policy::kWholeExpert : Policy::kAdaptiveFractional;
    const auto image = take("--image"), trace = take("--trace");
    const auto compute = args.contains("--compute") ? take("--compute") : "none";
    const auto rows = args.contains("--rows") ? integer("--rows") : 1;
    if (compute != "none" && compute != "gemv") throw std::invalid_argument("compute must be none or gemv");
    if (rows > 64 || (compute == "none" && rows != 1)) throw std::invalid_argument("rows require GEMV and must be in 1..64");
    if (identity.hidden > 65536 || identity.intermediate > 65536)
      throw std::invalid_argument("diagnostic dimensions exceed 65536");
    if (!args.empty()) throw std::invalid_argument("unknown option: " + args.begin()->first);
    const auto layouts = partition(identity.hidden, identity.intermediate, 1, 0,
                                    identity.chunk_intermediate, identity.format);
    for (const auto& layout : layouts)
      config.reads.staging_budget_bytes = std::max(config.reads.staging_budget_bytes,
          layout.record_bytes * config.reads.buffers);
    std::vector<RecordKey> keys;
    for (uint32_t e = 0; e < experts; ++e)
      for (uint32_t chunk = 0; chunk < layouts.size(); ++chunk) keys.push_back({0, e, chunk, false});
    // Persist the entire route sequence outside the binary. It is reused
    // unchanged across policies and can contain abrupt population shifts.
    std::ifstream input(trace);
    if (!input) throw std::runtime_error("cannot open trace");
    std::vector<std::pair<std::string, size_t>> selections;
    std::string line;
    while (std::getline(input, line)) {
      std::istringstream row(line);
      std::string phase, extra;
      size_t expert;
      if (!(row >> phase >> expert) || (row >> extra) || expert >= experts)
        throw std::invalid_argument("trace rows require PHASE EXPERT_INDEX");
      selections.emplace_back(phase, expert);
    }
    if (selections.empty()) throw std::invalid_argument("empty trace");
    if (!std::filesystem::exists(image)) {
      // Bound host payload to one synthetic expert; generate before timing.
      const auto full = partition(identity.hidden, identity.intermediate, 1, 0,
                                  identity.intermediate, identity.format).front();
      std::array<std::vector<uint8_t>, 3> payload, scales;
      std::array<MatrixSource, 3> source;
      const std::array<MatrixLayout, 3> matrices{full.gate, full.up, full.down};
      for (size_t m = 0; m < 3; ++m) {
        const auto& matrix = matrices[m];
        payload[m].resize(matrix.payload_bytes);
        scales[m].assign(matrix.scales_bytes, format == "mxfp4" ? 127 : 56);
        source[m] = {matrix.rows, matrix.cols, identity.format, payload[m], scales[m], std::nullopt};
        if (identity.format == Format::kNvfp4) source[m].global = 1.0f;
      }
      ExpertStoreBuilder builder(image, identity, keys);
      uint32_t rng = 43;
      for (uint32_t e = 0; e < experts; ++e) {
        for (auto& matrix : payload) for (auto& byte : matrix) {
          rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
          byte = static_cast<uint8_t>(rng);
        }
        for (size_t chunk = 0; chunk < layouts.size(); ++chunk)
          builder.write(e * layouts.size() + chunk,
              pack_chunk(source, layouts[chunk].intermediate_begin, layouts[chunk].intermediate_size));
      }
      builder.finish();
    }
    auto store = std::make_shared<ExpertStore>(image, identity, keys.size(), IoMode::kDirect);
    std::unique_ptr<ExpertCompute> arithmetic;
    if (compute == "gemv") arithmetic = std::make_unique<ExpertCompute>(static_cast<int>(rows), identity);
    const auto activation_bytes = arithmetic ? arithmetic->bytes() : 0;
    ExpertCache cache(store, config, cuda_cache_storage(identity, nullptr));
    auto before = cache.telemetry();
    auto reads = process_read_bytes();
    auto start = Clock::now();
    cache.initialize();
    report("initialization", before, cache.telemetry(), ns(start), process_read_bytes() - reads, {},
           compute, rows, activation_bytes);
    for (size_t begin = 0; begin < selections.size();) {
      size_t end = begin + 1;
      while (end < selections.size() && selections[end].first == selections[begin].first) ++end;
      before = cache.telemetry(); reads = process_read_bytes(); start = Clock::now();
      std::vector<uint64_t> latency;
      ComputeStats stats;
      latency.reserve(end - begin);
      for (size_t i = begin; i < end; ++i) {
        const auto tick = Clock::now();
        const auto expert = selections[i].second;
        cache.observe(std::span<const size_t>(&expert, 1));
        auto lease = cache.acquire(expert);
        if (!lease) throw std::runtime_error("unexpected execution backpressure");
        const auto acquire_ns = ns(tick);
        if (arithmetic) arithmetic->run(lease, stats);
        const auto release_tick = Clock::now();
        lease.release();
        if ((i + 1) % adapt_every == 0) cache.adapt();
        latency.push_back(acquire_ns + ns(release_tick));
        if (arithmetic) stats.selection_ns.push_back(ns(tick));
      }
      report(selections[begin].first, before, cache.telemetry(), ns(start), process_read_bytes() - reads,
             std::move(latency), compute, rows, activation_bytes, std::move(stats));
      begin = end;
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "expert replay: " << e.what() << '\n';
    return 1;
  }
}
