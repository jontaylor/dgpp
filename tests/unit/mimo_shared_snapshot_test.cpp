#include "common/test.hpp"
#include "models/mimo/shared_snapshot.hpp"
#include <cstring>
#include <thread>

namespace {
void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
}
DGPP_TEST(mimo_shared_snapshot_restore_branch_rollback_eviction) {
  size_t owned = 0;
  dgpp::MimoSharedSnapshots store(1, 3, [&](size_t n) {
    owned += n;
    return dgpp::MimoSharedSnapshots::Storage(new uint8_t[n], [&, n](uint8_t* p) { delete[] p; owned -= n; });
  });
  int a, b, c;
  for (const auto* slot : {&a, &b, &c}) store.register_slot(slot);
  std::vector<uint8_t> live(1024);
  for (size_t i = 0; i < live.size(); ++i) live[i] = uint8_t(i * 17 + 1);
  auto write = [&](int req, void* slot, int64_t position) {
    return store.write(req, slot, position,
        [&](uint8_t* dst, int64_t first, int64_t count) { std::memcpy(dst, live.data() + first, count); },
        [&](uint8_t* dst, int64_t, int64_t pos) { std::memcpy(dst, live.data() + pos - 3, 3); });
  };
  auto read = [&](int req, void* slot, int64_t position) {
    std::vector<uint8_t> out(position), tail(3);
    store.read(req, slot, position,
        [&](const uint8_t* src, int64_t first, int64_t count) { std::memcpy(out.data() + first, src, count); },
        [&](const uint8_t* src, int64_t, int64_t) { std::memcpy(tail.data(), src, 3); });
    require(std::equal(tail.begin(), tail.end(), out.end() - 3), "private state corrupted");
    return out;
  };
  require(write(0, &a, 300) == 303, "initial copy size");
  const auto original = live;
  require(write(0, &b, 600) == 347, "full block not reused");
  require(owned == 650 && owned == store.unique_bytes(), "shared storage accounting");
  require(read(1, &a, 300) == std::vector<uint8_t>(original.begin(), original.begin()+300), "initial restore");
  std::fill(live.begin()+300, live.end(), 91);
  write(1, &c, 550);  // branch shares first complete block; private partial diverges
  require(read(2, &b, 600) == std::vector<uint8_t>(original.begin(), original.begin()+600), "branch mutated sibling");
  store.rewind(1, 270);
  std::fill(live.begin()+270, live.end(), 42);
  write(1, &c, 550);
  require(read(1, &c, 550) == std::vector<uint8_t>(live.begin(), live.begin()+550), "rollback reused overwritten block");
  store.rewind(1);
  std::fill(live.begin(), live.end(), 7);
  write(1, &c, 550);
  require(read(1, &c, 550) == std::vector<uint8_t>(550,7), "request reset reused prior lineage");
  store.release(&a);
  store.release(&b);
  store.release(&c);
  require(owned == 0 && store.unique_bytes() == 0, "weak lineage pinned evicted memory");
  for (auto* slot : {&a, &b, &c}) store.unregister_slot(slot);
}
DGPP_TEST(mimo_shared_snapshot_replace_and_copy_failure) {
  size_t owned = 0;
  dgpp::MimoSharedSnapshots store(1, 1, [&](size_t n) {
    owned += n;
    return dgpp::MimoSharedSnapshots::Storage(new uint8_t[n], [&, n](uint8_t* p) { delete[] p; owned -= n; });
  });
  int slot;
  store.register_slot(&slot);
  auto fill = [](uint8_t* dst, int64_t, int64_t n) { std::memset(dst, 3, n); };
  auto one = [](uint8_t* dst, int64_t, int64_t) { *dst = 8; };
  require(store.write(0, &slot, 256, fill, one) == 257, "first full block");
  require(store.write(0, &slot, 512, fill, one) == 257, "replacement failed to retain full blocks");
  require(owned == 513, "replacement retained obsolete allocation");
  bool failed = false;
  try {
    store.write(0, &slot, 700, [](uint8_t*, int64_t, int64_t) { throw std::runtime_error("copy failure"); }, one);
  } catch (const std::runtime_error&) { failed = true; }
  require(failed && owned == 0, "failed replacement leaked storage");
  require(store.write(0, &slot, 700, fill, one) == 701, "failure published stale provenance");
  store.unregister_slot(&slot);
  require(owned == 0, "unregister leaked storage");
}

DGPP_TEST(mimo_shared_snapshot_metrics_concurrent_read) {
  dgpp::MimoSharedSnapshots store(1, 4, [](size_t n) {
    return dgpp::MimoSharedSnapshots::Storage(new uint8_t[n], std::default_delete<uint8_t[]>());
  });
  int slot; store.register_slot(&slot);
  std::atomic<bool> stop{false}, invalid{false};
  std::thread reader([&] {
    while (!stop.load(std::memory_order_relaxed))
      if (store.unique_bytes() > 1028) invalid.store(true, std::memory_order_relaxed);
  });
  for (int i = 0; i < 1000; ++i) {
    store.rewind(0);
    store.write(0, &slot, 512, [](uint8_t*, int64_t, int64_t) {}, [](uint8_t*, int64_t, int64_t) {});
    store.release(&slot);
  }
  stop.store(true, std::memory_order_relaxed); reader.join();
  require(!invalid.load() && store.unique_bytes() == 0, "concurrent metric accounting");
}

DGPP_TEST(mimo_shared_snapshot_slot_alignment) {
  using Layout = dgpp::MimoSharedSnapshots;
  require(Layout::header_bytes % 16 == 0, "draft payload offset alignment");
  for (size_t suffix = 0; suffix < 1024; ++suffix) {
    const size_t stride = Layout::slot_storage_bytes(suffix);
    require(stride % 16 == 0, "arena slot stride alignment");
    require(stride >= Layout::header_bytes + suffix, "arena slot truncates payload");
    require(stride - Layout::header_bytes - suffix < 16, "excess arena padding");
  }
}
