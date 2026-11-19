#pragma once

#include <cstdint>

namespace dgpp::sample {

struct Candidate {
  int32_t id = -1;
  float logit = 0.0f;
};

}  // namespace dgpp::sample
