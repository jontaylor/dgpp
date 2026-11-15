#include "serve/expert_paging.hpp"

#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>

#include "serve/json_out.hpp"

namespace dgpp::serve {
namespace {
[[noreturn]] void bad(const std::string& what, const std::string& key, const char* why) {
  throw std::runtime_error(what + ".expert_streaming." + key + ": " + why);
}
uint64_t integer(const minijson::Value& v, const std::string& what, const std::string& key,
                  uint64_t lo = 0, uint64_t hi = (uint64_t{1} << 53) - 1) {
  if (!v.is_number() || !std::isfinite(v.as_double()) || v.as_double() < static_cast<double>(lo) ||
      v.as_double() > static_cast<double>(hi) || std::trunc(v.as_double()) != v.as_double())
    bad(what, key, "expected a bounded nonnegative integer");
  return static_cast<uint64_t>(v.as_double());
}
double number(const minijson::Value& v, const std::string& what, const std::string& key,
                double lo, double hi) {
  if (!v.is_number() || !std::isfinite(v.as_double()) || v.as_double() < lo || v.as_double() > hi)
    bad(what, key, "expected a finite value in range");
  return v.as_double();
}
std::string string(const minijson::Value& v, const std::string& what, const std::string& key) {
  if (!v.is_string() || v.as_string().empty()) bad(what, key, "expected a nonempty string");
  return std::string(v.as_string());
}
}
expert::PagingOptions parse_expert_paging(const minijson::Value& v, const std::string& what) {
  using namespace expert;
  if (!v.is_object()) bad(what, "", "expected an object");
  PagingOptions result;
  bool policy = false, budget = false, directory = false;
  auto& p = result.cache.planner;
  auto& r = result.cache.reads;
  for (const auto& m : v.members()) {
    const auto& key = m.key;
    const auto& x = m.value;
    if (key == "policy") {
      const auto name = string(x, what, key);
      if (name == "whole_expert_cache") p.policy = Policy::kWholeExpert;
      else if (name == "adaptive_fractional_cache") p.policy = Policy::kAdaptiveFractional;
      else bad(what, key, "expected whole_expert_cache or adaptive_fractional_cache");
      policy = true;
    } else if (key == "budget_bytes") { p.budget_bytes = integer(x, what, key, 1); budget = true; }
    else if (key == "image_directory") { result.image_directory = string(x, what, key); directory = true; }
    else if (key == "chunk_intermediate") result.chunk_intermediate = integer(x, what, key, 32, 1 << 20);
    else if (key == "adapt_every_routes") result.adapt_every_routes = integer(x, what, key, 1);
    else if (key == "execution_slots") result.cache.execution_slots = integer(x, what, key, 1, 4096);
    else if (key == "io_buffers") r.buffers = integer(x, what, key, 1, 4096);
    else if (key == "io_workers") r.workers = integer(x, what, key, 1, 4096);
    else if (key == "queue_depth") r.max_pending = integer(x, what, key, 1, 1 << 20);
    else if (key == "prefetch_bytes") r.prefetch_bytes = integer(x, what, key);
    else if (key == "io_mode") {
      const auto mode = string(x, what, key);
      if (mode == "direct") result.io_mode = IoMode::kDirect;
      else if (mode == "buffered") result.io_mode = IoMode::kBuffered;
      else bad(what, key, "expected direct or buffered");
    } else if (key == "minimum_fraction") p.minimum_fraction = number(x, what, key, 0, 1);
    else if (key == "initial_fraction") p.initial_fraction = number(x, what, key, std::numeric_limits<double>::min(), 1);
    else if (key == "decay_routes") p.decay_routes = number(x, what, key, 1, 1e15);
    else if (key == "horizon_routes") p.horizon_routes = number(x, what, key, 1, 1e15);
    else if (key == "prefill_route_weight") p.prefill_route_weight = number(x, what, key, 1.0 / 1024, 1);
    else if (key == "layer_local_history") {
      if (!x.is_bool()) bad(what, key, "expected true or false");
      p.layer_local_history = x.as_bool();
    }
    else if (key == "learn_load_cost") {
      if (!x.is_bool()) bad(what, key, "expected true or false");
      p.learn_load_cost = x.as_bool();
    }
    else if (key == "promotion_margin") p.promotion_margin = number(x, what, key, 0, 1e6);
    else if (key == "cooldown_routes") p.cooldown_routes = integer(x, what, key);
    else if (key == "adaptation_bytes") p.adaptation_bytes = integer(x, what, key);
    else bad(what, key, "unknown setting");
  }
  if (!policy || !budget || !directory) bad(what, "", "policy, budget_bytes and image_directory are required");
  if (result.chunk_intermediate % 32) bad(what, "chunk_intermediate", "must align to 32 intermediate values");
  if (r.workers > r.buffers || r.max_pending < r.buffers) bad(what, "io_buffers", "require workers <= buffers <= queue_depth");
  if (p.policy == Policy::kWholeExpert && p.initial_fraction) bad(what, "initial_fraction", "only valid for adaptive_fractional_cache");
  if (p.initial_fraction && *p.initial_fraction < p.minimum_fraction) bad(what, "initial_fraction", "must be at least minimum_fraction");
  // The shape-specific pool size is checked against this limit and the common
  // total budget in preflight; it is never an extra out-of-budget allowance.
  r.staging_budget_bytes = p.budget_bytes;
  return result;
}
std::string expert_paging_json(const expert::PagingOptions& o) {
  using namespace expert;
  const auto& p = o.cache.planner;
  const auto& r = o.cache.reads;
  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << std::setprecision(std::numeric_limits<double>::max_digits10);
  out << "{\"policy\":" << json_string(p.policy == Policy::kWholeExpert ? "whole_expert_cache" : "adaptive_fractional_cache")
      << ",\"budget_bytes\":" << p.budget_bytes << ",\"image_directory\":" << json_string(o.image_directory)
      << ",\"chunk_intermediate\":" << o.chunk_intermediate << ",\"adapt_every_routes\":" << o.adapt_every_routes
      << ",\"execution_slots\":" << o.cache.execution_slots << ",\"io_buffers\":" << r.buffers
      << ",\"io_workers\":" << r.workers << ",\"queue_depth\":" << r.max_pending
      << ",\"prefetch_bytes\":" << r.prefetch_bytes << ",\"io_mode\":" << json_string(o.io_mode == IoMode::kDirect ? "direct" : "buffered")
      << ",\"minimum_fraction\":" << p.minimum_fraction << ",\"decay_routes\":" << p.decay_routes
      << ",\"horizon_routes\":" << p.horizon_routes << ",\"promotion_margin\":" << p.promotion_margin
      << ",\"prefill_route_weight\":" << p.prefill_route_weight
      << ",\"layer_local_history\":" << (p.layer_local_history ? "true" : "false")
      << ",\"learn_load_cost\":" << (p.learn_load_cost ? "true" : "false")
      << ",\"cooldown_routes\":" << p.cooldown_routes;
  if (p.initial_fraction) out << ",\"initial_fraction\":" << *p.initial_fraction;
  if (p.adaptation_bytes != UINT64_MAX) out << ",\"adaptation_bytes\":" << p.adaptation_bytes;
  out << '}';
  return out.str();
}
}  // namespace dgpp::serve
