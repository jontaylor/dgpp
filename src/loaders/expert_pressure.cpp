#include "loaders/expert_pressure.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <utility>

namespace dgpp::expert {
namespace {
std::optional<uint64_t> memory_available(const std::string& path) {
  std::ifstream input(path);
  std::string line;
  while (std::getline(input, line)) {
    std::istringstream row(line);
    std::string key, value, unit, extra;
    if (!(row >> key) || key != "MemAvailable:") continue;
    if (!(row >> value >> unit) || unit != "kB" || (row >> extra) || value.empty() || value[0] == '-') return {};
    size_t used = 0;
    try {
      const auto kib = std::stoull(value, &used);
      if (used == value.size() && kib <= std::numeric_limits<uint64_t>::max() / 1024) return kib * 1024;
    } catch (const std::exception&) {}
    return {};
  }
  return {};
}
std::optional<uint64_t> busy_ms(const std::string& path) {
  if (path.empty()) return {};
  std::ifstream input(path);
  // Linux block stat field 10 is milliseconds with at least one I/O active.
  std::string field;
  for (int i = 0; i < 10; ++i) if (!(input >> field)) return {};
  if (field.empty() || field[0] == '-') return {};
  size_t used = 0;
  try {
    const auto value = std::stoull(field, &used);
    if (used == field.size()) return value;
  } catch (const std::exception&) {}
  return {};
}
}
ResourcePressureMonitor::ResourcePressureMonitor(std::string meminfo, std::string block_stat,
                                                ResourcePressureLimits limits)
    : meminfo_(std::move(meminfo)), block_stat_(std::move(block_stat)), limits_(limits) {
  if (limits.memory_pause_bytes >= limits.memory_resume_bytes || !limits.sample_interval_ms ||
      !std::isfinite(limits.storage_resume_fraction) || !std::isfinite(limits.storage_pause_fraction) ||
      limits.storage_resume_fraction < 0 || limits.storage_pause_fraction > 1 ||
      limits.storage_resume_fraction >= limits.storage_pause_fraction)
    throw std::invalid_argument("expert resource monitor: invalid hysteresis or sample interval");
}
ResourcePressure ResourcePressureMonitor::sample(uint64_t now) {
  if (previous_ms_ && now >= *previous_ms_ && now - *previous_ms_ < limits_.sample_interval_ms) return current_;
  const auto busy = busy_ms(block_stat_);
  const auto available = memory_available(meminfo_);
  std::optional<double> utilization;
  if (previous_ms_ && previous_busy_ms_ && busy && now > *previous_ms_ && *busy >= *previous_busy_ms_)
    utilization = std::min(1.0, static_cast<double>(*busy - *previous_busy_ms_) / static_cast<double>(now - *previous_ms_));
  previous_ms_ = now;
  previous_busy_ms_ = busy;
  current_.sample++;
  current_.available_bytes = available;
  current_.storage_busy_fraction = utilization;
  // Lost/reset signals are explicitly unknown and cannot freeze background
  // activity indefinitely. The next valid sample re-establishes pressure.
  if (!available) current_.memory_pressure = false;
  else if (*available <= limits_.memory_pause_bytes) current_.memory_pressure = true;
  else if (*available >= limits_.memory_resume_bytes) current_.memory_pressure = false;
  if (!utilization) current_.storage_pressure = false;
  else if (*utilization >= limits_.storage_pause_fraction) current_.storage_pressure = true;
  else if (*utilization <= limits_.storage_resume_fraction) current_.storage_pressure = false;
  return current_;
}
ResourcePressureSampler linux_expert_resource_sampler(const std::string& directory) {
  struct stat st {};
  std::string block;
  if (::stat(directory.c_str(), &st) == 0)
    block = "/sys/dev/block/" + std::to_string(major(st.st_dev)) + ":" + std::to_string(minor(st.st_dev)) + "/stat";
  auto monitor = std::make_shared<ResourcePressureMonitor>("/proc/meminfo", std::move(block));
  return [monitor] {
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    return monitor->sample(static_cast<uint64_t>(ms));
  };
}
}  // namespace dgpp::expert
