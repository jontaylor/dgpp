#include <array>
#include <iostream>
#include <stdexcept>
#include "models/dsv41/paged_binding_table.hpp"

int main() {
  using dgpp::dsv41_detail::update_paged_binding_table;
  auto require = [](bool condition) { if (!condition) throw std::runtime_error("paged binding regression"); };
  std::array<uint64_t, 18> table{};
  std::array<uint8_t, 64> first{}, second{};
  std::array<const uint8_t*, 3> addresses{};
  int builds = 0;
  auto binding = [&](const uint8_t* address) {
    ++builds;
    std::array<uint64_t, 6> result;
    for (size_t i = 0; i < result.size(); ++i) result[i] = reinterpret_cast<uintptr_t>(address + i);
    return result;
  };
  require(update_paged_binding_table(table, addresses, binding, true)); // initializes empty device table
  require(!update_paged_binding_table(table, addresses, binding));
  require(builds == 0);
  addresses[1] = first.data();
  require(update_paged_binding_table(table, addresses, binding));
  require(builds == 1);
  for (int j = 0; j < 3; ++j) {
    require(table[3 + j] == reinterpret_cast<uintptr_t>(first.data() + j));
    require(table[12 + j] == reinterpret_cast<uintptr_t>(first.data() + 3 + j));
  }
  require(!update_paged_binding_table(table, addresses, binding));
  require(builds == 1);
  addresses[1] = second.data(); // relocation refreshes all six pointers
  require(update_paged_binding_table(table, addresses, binding));
  require(table[3] == reinterpret_cast<uintptr_t>(second.data()));
  addresses[1] = nullptr; // eviction must clear stale pointers in both projection halves
  require(update_paged_binding_table(table, addresses, binding));
  for (const auto word : table) require(word == 0);
  addresses[0] = first.data();
  addresses[2] = second.data();
  require(update_paged_binding_table(table, addresses, binding));
  require(table[0] == reinterpret_cast<uintptr_t>(first.data()));
  require(table[6] == reinterpret_cast<uintptr_t>(second.data()));
  require(table[15] == reinterpret_cast<uintptr_t>(second.data() + 3));
  bool rejected = false;
  try { update_paged_binding_table(std::span<uint64_t>(table).first(17), addresses, binding); }
  catch (const std::invalid_argument&) { rejected = true; }
  require(rejected);
  std::cout << "paged binding updates passed\n";
}
