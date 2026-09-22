#pragma once
#include <cstdint>
namespace dgpp {
// Per-layer K then V. Finite inputs only contribute to changed/error/norm.
// Counts are executed append elements, not distinct tokens or committed cache.
struct MimoFp8AuditStats {
  uint64_t conversions = 0, clipped = 0, nonfinite = 0, changed = 0;
  double error_squared = 0, reference_squared = 0;
};
static_assert(2 * sizeof(MimoFp8AuditStats) <= 256);  // one aligned model allocation per layer
}  // namespace dgpp
