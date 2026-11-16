#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <stdexcept>

namespace dgpp::dsv41_detail {
// Packed W13 begins at the native allocation's base. Its address therefore
// identifies all six binding pointers for a fixed model geometry. Payload
// changes at the same address do not require republishing the pointers.
// The caller must finish the previous upload before modifying this host table.
template <class MakeBinding>
bool update_paged_binding_table(std::span<uint64_t> table,
    std::span<const uint8_t* const> addresses, MakeBinding make_binding, bool force = false) {
  const size_t experts = addresses.size();
  if (table.size() != experts * 6) throw std::invalid_argument("paged binding table geometry");
  bool changed = force;
  for (size_t e = 0; e < experts; ++e) {
    if (!force && table[e * 3] == reinterpret_cast<uintptr_t>(addresses[e])) continue;
    const auto binding = addresses[e] ? make_binding(addresses[e]) : std::array<uint64_t, 6>{};
    for (size_t j = 0; j < 3; ++j) {
      table[e * 3 + j] = binding[j];
      table[experts * 3 + e * 3 + j] = binding[3 + j];
    }
    changed = true;
  }
  return changed;
}
}  // namespace dgpp::dsv41_detail
