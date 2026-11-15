#include "serve/expert_paging.hpp"

#include <stdexcept>
#include <string>

#include "common/test.hpp"

namespace {
using namespace dgpp::serve;
void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
template <typename F> void refuses(F f) {
  bool rejected = false;
  try { f(); } catch (const std::exception&) { rejected = true; }
  require(rejected, "invalid expert streaming configuration must be refused");
}
dgpp::expert::PagingOptions parse(const std::string& json) {
  const auto document = dgpp::minijson::parse(json);
  return parse_expert_paging(document.root, "test.engine");
}
}
DGPP_TEST(expert_paging_configuration_roundtrips_every_effective_setting) {
  for (const auto* policy : {"whole_expert_cache", "adaptive_fractional_cache"}) {
    const std::string base = std::string("{\"policy\":\"") + policy + R"(","budget_bytes":1073741824,"image_directory":"/ssd/expert \"cache\""})";
    const auto first = parse(base);
    require(first.cache.planner.learn_load_cost, "load-cost learning remains the default");
    require(first.cache.reads.staging_budget_bytes == 1073741824, "staging is charged inside the shared budget");
    const auto encoded = expert_paging_json(first);
    require(parse(encoded) == first && expert_paging_json(parse(encoded)) == encoded,
            "canonical full settings stable across rank journal roundtrip");
  }
  const auto configured = parse(R"({"policy":"adaptive_fractional_cache","budget_bytes":1073741824,
    "image_directory":"/ssd/experts","io_mode":"buffered","chunk_intermediate":128,
    "execution_slots":3,"io_buffers":4,"io_workers":2,"queue_depth":7,"prefetch_bytes":4096,
    "minimum_fraction":0.125,"initial_fraction":0.375,"decay_routes":32,"horizon_routes":64,"learn_load_cost":false,
    "promotion_margin":0.5,"cooldown_routes":17,"adaptation_bytes":1024,"adapt_every_routes":8})");
  require(parse(expert_paging_json(configured)) == configured, "nondefault settings preserved exactly");
  require(!configured.cache.planner.learn_load_cost, "fixed-cost mode parsed explicitly");
  auto different = configured;
  ++different.adapt_every_routes;
  require(expert_paging_json(different) != expert_paging_json(configured), "adaptation interval included in effective digest input");
}
DGPP_TEST(expert_paging_zero_floor_round_trips_without_disabling_initial_coverage) {
  const auto cfg = parse(R"({"policy":"adaptive_fractional_cache","budget_bytes":100000,"image_directory":"x","minimum_fraction":0})");
  require(cfg.cache.planner.minimum_fraction == 0 && !cfg.cache.planner.initial_fraction,
          "zero cold floor with automatic universal initial population");
  require(parse(expert_paging_json(cfg)).cache.planner.minimum_fraction == 0, "zero floor survives configuration serialization");
}
DGPP_TEST(expert_paging_configuration_refuses_unknown_inert_and_unbounded_settings) {
  const std::string prefix = R"({"policy":"whole_expert_cache","budget_bytes":1073741824,"image_directory":"/ssd/experts")";
  for (const auto* suffix : {R"(,"unknown":1})", R"(,"budget_bytes":0})", R"(,"initial_fraction":0.5})",
       R"(,"io_mode":"automatic"})", R"(,"chunk_intermediate":17})", R"(,"minimum_fraction":-0.1})",
       R"(,"io_buffers":1,"io_workers":2})", R"(,"io_buffers":9,"queue_depth":3})",
       R"(,"execution_slots":0})", R"(,"adapt_every_routes":0})", R"(,"budget_bytes":1.5})", R"(,"learn_load_cost":1})"})
    refuses([&] { (void)parse(prefix + suffix); });
  refuses([&] { (void)parse("{}"); });
  refuses([&] { (void)parse(R"({"policy":"fractional","budget_bytes":1,"image_directory":"x"})"); });
  refuses([&] { (void)parse(R"({"policy":"adaptive_fractional_cache","budget_bytes":1000,"image_directory":"x","minimum_fraction":0.8,"initial_fraction":0.2})"); });
}
int main() { return dgpp::test::run_all(); }
