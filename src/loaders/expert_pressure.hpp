#pragma once
// Optional Linux resource signals for background expert reads. No CUDA calls,
// memory reclamation, demand throttling or physical-bandwidth inference.
#include <cstdint>
#include <functional>
#include <optional>
#include <string>

namespace dgpp::expert {
struct ResourcePressure {
  uint64_t sample = 0;
  std::optional<uint64_t> available_bytes;
  std::optional<double> storage_busy_fraction;
  bool memory_pressure = false, storage_pressure = false;
  bool background_allowed() const { return !memory_pressure && !storage_pressure; }
};
struct ResourcePressureLimits {
  uint64_t memory_pause_bytes = 4ull << 30, memory_resume_bytes = 6ull << 30;
  double storage_pause_fraction = 0.9, storage_resume_fraction = 0.7;
  uint64_t sample_interval_ms = 1000;
};
using ResourcePressureSampler = std::function<ResourcePressure()>;
// Exposed paths/clock permit deterministic host tests without sleeps or GPU use.
class ResourcePressureMonitor {
 public:
  ResourcePressureMonitor(std::string meminfo, std::string block_stat,
                          ResourcePressureLimits limits = {});
  ResourcePressure sample(uint64_t monotonic_ms);
 private:
  std::string meminfo_, block_stat_;
  ResourcePressureLimits limits_;
  ResourcePressure current_;
  std::optional<uint64_t> previous_ms_, previous_busy_ms_;
};
// Resolve the image filesystem's block device. Network filesystems leave the
// device signal unknown; they are never reported as idle NVMe. Signals are
// node/device-wide, include unrelated consumers, and are sampled at most 1 Hz.
ResourcePressureSampler linux_expert_resource_sampler(const std::string& image_directory);
}  // namespace dgpp::expert
