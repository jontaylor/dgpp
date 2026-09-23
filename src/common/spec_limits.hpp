#pragma once

namespace dgpp {
// One verification/pick for the pending token plus one pick per draft.
// Shared by engine row bounds, kernel proposals/outcomes and DevicePicker's
// slot-major allocations. Keep these coupled when adding a wider drafter.
inline constexpr int kSpeculationMaxDrafts = 7;
inline constexpr int kSpeculationRows = 1 + kSpeculationMaxDrafts;
}  // namespace dgpp
