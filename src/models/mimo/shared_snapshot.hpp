#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace dgpp {
// Immutable snapshot blocks only; live attention remains contiguous. Weak
// request lineage references never keep an evicted snapshot's storage alive.
// The allocator and copier are injected so ownership is testable without CUDA.
class MimoSharedSnapshots {
 public:
  using Storage = std::shared_ptr<uint8_t>;
  using Allocate = std::function<Storage(size_t)>;
  using Write = std::function<void(uint8_t*, int64_t, int64_t)>;
  using Read = std::function<void(const uint8_t*, int64_t, int64_t)>;
  static constexpr int64_t block_tokens = 256;
  MimoSharedSnapshots(size_t bytes_per_token, size_t private_bytes, Allocate allocate)
      : token_bytes_(bytes_per_token), private_bytes_(private_bytes),
        allocate_([allocate = std::move(allocate), owned = owned_](size_t bytes) {
          auto storage = allocate(bytes);
          auto* pointer = storage.get();
          owned->fetch_add(bytes, std::memory_order_relaxed);
          // The wrapper counts unique allocations, not snapshot references.
          // Its counter can outlive the store if a copy is unwinding.
          return Storage(pointer, [storage = std::move(storage), owned, bytes](uint8_t*) mutable {
            storage.reset();
            owned->fetch_sub(bytes, std::memory_order_relaxed);
          });
        }) {}
  void register_slot(const void* slot) { slots_.try_emplace(slot); }
  bool contains(const void* slot) const { return slots_.contains(slot); }
  void release(const void* slot) { if (contains(slot)) slots_.at(slot) = {}; }
  void unregister_slot(const void* slot) { slots_.erase(slot); }
  void rewind(int req, int64_t position = 0) {
    auto it = lineage_.find(req);
    if (it == lineage_.end()) return;
    // A block straddling the rollback boundary can be overwritten next.
    it->second.resize(std::min(it->second.size(), size_t(position / block_tokens)));
  }
  size_t write(int req, const void* slot, int64_t position,
               const Write& write_global, const Write& write_private) {
    if (position <= 0 || !contains(slot)) throw std::invalid_argument("shared snapshot destination/position");
    auto& history = lineage_[req];
    Snapshot next;
    // Retain reusable full blocks before dropping the replaced snapshot. This
    // bounds transient allocation to private state and the unfinished block.
    for (int64_t first = 0; first < position; first += block_tokens) {
      const int64_t count = std::min(block_tokens, position - first);
      const size_t index = size_t(first / block_tokens);
      auto block = index < history.size() ? history[index].lock() : nullptr;
      if (count != block_tokens) block.reset();
      next.blocks.push_back({std::move(block), count});
    }
    slots_.at(slot) = {};
    size_t copied = 0;
    for (size_t i = 0; i < next.blocks.size(); ++i) {
      auto& block = next.blocks[i];
      if (!block.data) {
        block.data = allocate_(size_t(block.tokens) * token_bytes_);
        write_global(block.data.get(), int64_t(i) * block_tokens, block.tokens);
        copied += size_t(block.tokens) * token_bytes_;
      }
    }
    next.private_data = allocate_(private_bytes_);
    write_private(next.private_data.get(), 0, position);
    copied += private_bytes_;
    next.position = position;
    // Publish ownership and provenance only after every copy is enqueued.
    slots_.at(slot) = std::move(next);
    seed(req, slots_.at(slot));
    return copied;
  }
  void read(int req, const void* slot, int64_t position,
            const Read& read_global, const Read& read_private) {
    const auto& snap = slots_.at(slot);
    if (snap.position != position) throw std::invalid_argument("shared snapshot position mismatch");
    for (size_t i = 0; i < snap.blocks.size(); ++i)
      read_global(snap.blocks[i].data.get(), int64_t(i) * block_tokens, snap.blocks[i].tokens);
    read_private(snap.private_data.get(), 0, position);
    seed(req, snap);
  }
  // HTTP metrics may run concurrently with the model owner. Never traverse
  // mutable snapshot maps from that thread.
  size_t unique_bytes() const { return owned_->load(std::memory_order_relaxed); }

 private:
  struct Block { Storage data; int64_t tokens; };
  struct Snapshot {
    int64_t position = 0;
    std::vector<Block> blocks;
    Storage private_data;
  };
  void seed(int req, const Snapshot& snap) {
    auto& history = lineage_[req];
    history.clear();
    for (const auto& block : snap.blocks)
      if (block.tokens == block_tokens) history.push_back(block.data);
  }
  std::shared_ptr<std::atomic<size_t>> owned_ = std::make_shared<std::atomic<size_t>>(0);
  size_t token_bytes_, private_bytes_;
  Allocate allocate_;
  std::unordered_map<const void*, Snapshot> slots_;
  std::unordered_map<int, std::vector<std::weak_ptr<uint8_t>>> lineage_;
};
}  // namespace dgpp
