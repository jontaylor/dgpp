#include <stdexcept>
#include <string>
#include <vector>

#include "common/test.hpp"
#include "models/mimo/dflash_boundary.hpp"

namespace {
struct Reducer : dgpp::BoundaryReducer {
  std::vector<std::string>& events;
  uint16_t* destination;
  bool ordered;
  int reductions = 0;
  Reducer(std::vector<std::string>& events, uint16_t* destination, bool ordered)
      : events(events), destination(destination), ordered(ordered) {}
  uint16_t* stage(int, int) override {
    events.push_back("stage");
    return destination;
  }
  bool stream_ordered() const override { return ordered; }
  void reduce(uint16_t* value, int, int) override {
    events.push_back("reduce");
    ++reductions;
    *value += 10;
  }
};
void require(bool condition) {
  if (!condition) throw std::runtime_error("DFlash reducer ordering/selection");
}
void call(dgpp::BoundaryReducer* reducer, uint16_t* fallback, uint16_t* expected, bool capture,
          std::vector<std::string>& events, int world = 2) {
  dgpp::mimo_dflash_fold(
      reducer, world, fallback, 1, 4096, capture,
      [&](uint16_t* p) {
        require(p == expected);
        events.push_back("project");
        *p = 1;
      },
      [&] { events.push_back("sync"); },
      [&](uint16_t* p) {
        require(p == expected);
        events.push_back("add");
        *p += 100;
      });
}
}  // namespace
DGPP_TEST(mimo_dflash_boundary_swap_uses_current_call_and_restores_eager) {
  std::vector<std::string> events;
  uint16_t fallback = 0, staged = 0;
  Reducer eager(events, nullptr, false);
  call(&eager, &fallback, &fallback, false, events);
  require(events == std::vector<std::string>{"stage", "project", "sync", "reduce", "add"});
  require(fallback == 111 && eager.reductions == 1);
  events.clear();
  {
    Reducer recorder(events, &staged, false);
    call(&recorder, &fallback, &staged, true, events);
    require(events == std::vector<std::string>{"stage", "project", "reduce", "add"});
    require(staged == 111 && recorder.reductions == 1 && eager.reductions == 1);
  }  // recorder is destroyed, so later eager use must not retain its pointer.
  events.clear();
  call(&eager, &fallback, &fallback, false, events);
  require(events == std::vector<std::string>{"stage", "project", "sync", "reduce", "add"});
  require(eager.reductions == 2);
}
DGPP_TEST(mimo_dflash_stream_ordered_eager_falls_back_without_host_sync) {
  std::vector<std::string> events;
  uint16_t fallback = 0;
  Reducer eager(events, nullptr, true);
  call(&eager, &fallback, &fallback, false, events);
  require(events == std::vector<std::string>{"stage", "project", "reduce", "add"});
  require(fallback == 111);
}
DGPP_TEST(mimo_dflash_capture_refuses_unstaged_or_missing_tp_reducer) {
  std::vector<std::string> events;
  uint16_t fallback = 0;
  Reducer eager(events, nullptr, true);
  bool threw = false;
  try {
    call(&eager, &fallback, &fallback, true, events);
  } catch (const std::logic_error&) {
    threw = true;
  }
  require(threw && events == std::vector<std::string>{"stage"} && fallback == 0);
  events.clear();
  threw = false;
  try {
    call(nullptr, &fallback, &fallback, true, events);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  require(threw && events.empty() && fallback == 0);
}
DGPP_TEST(mimo_dflash_world_one_has_no_collective_or_host_sync) {
  std::vector<std::string> events;
  uint16_t fallback = 0;
  call(nullptr, &fallback, &fallback, true, events, 1);
  require(events == std::vector<std::string>{"project", "add"} && fallback == 101);
}
