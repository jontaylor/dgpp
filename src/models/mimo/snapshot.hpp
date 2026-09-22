#pragma once

#include <cstdint>
#include <unordered_map>

namespace dgpp {
// Host-only provenance for the immutable global prefix already in a destination.
// A released destination or a reset/restored request must never inherit it.
class MimoSnapshotHistory {
 public:
  // Only arena-owned allocations opt in. Raw session_snapshot callers retain
  // full-copy behavior, including when an external allocator reuses an address.
  void register_destination(const void* dst) { entries_[dst] = {}; }
  void unregister_destination(const void* dst) { entries_.erase(dst); }
  int64_t begin(int req, const void* dst, int64_t position) {
    const auto it = entries_.find(dst);
    if (it == entries_.end()) return 0;
    const int64_t previous = it->second.req == req && it->second.position <= position
                                 ? it->second.position
                                 : 0;
    // Publish only after every layer's copies have been enqueued successfully.
    it->second = {};
    return previous;
  }
  void commit(int req, const void* dst, int64_t position) {
    const auto it = entries_.find(dst);
    if (it != entries_.end()) it->second = {req, position};
  }
  void release(const void* dst) {
    const auto it = entries_.find(dst);
    if (it != entries_.end()) it->second = {};
  }
  void rewind(int req, int64_t position = 0) {
    for (auto& [dst, entry] : entries_)
      if (entry.req == req && entry.position > position) entry = {};
  }

 private:
  struct Entry {
    int req = -1;
    int64_t position = 0;
  };
  std::unordered_map<const void*, Entry> entries_;
};
}  // namespace dgpp
