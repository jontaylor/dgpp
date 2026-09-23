// No checkpoint or fabric required. Exercise production copy/restore helpers and
// PrefixArena lifetime hooks; all comparisons are bit-exact, including tails.
#include <cuda_runtime.h>

#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "engine/prefix_arena.hpp"
#include "models/mimo/snapshot.hpp"
#include "models/mimo/shared_snapshot.hpp"
#include "models/mimo/snapshot_copy.hpp"

namespace {
void check(bool ok, const char* why) {
  if (!ok) throw std::runtime_error(why);
}
struct Buffer {
  uint8_t* p = nullptr;
  explicit Buffer(size_t n) { DGPP_CUDA_OK(cudaMalloc(&p, n)); }
  ~Buffer() { cudaFree(p); }
};
template<class CacheT>
struct Model {
  struct SessionSnapshotMeta {
    int64_t position = 0;
  };
  struct SnapshotRequest {
    int64_t position = 0;
    void* dst = nullptr;
    SessionSnapshotMeta* meta = nullptr;
    bool taken = false;
  };
  // Capacity deliberately not a multiple of the window, to exercise both spans.
  std::array<dgpp::MimoAttentionShape, 2> shapes{{{1, 2, 1, 1024, 0, sizeof(CacheT)==1}, {1, 2, 1, 137, 128, sizeof(CacheT)==1}}};
  dgpp::MimoSnapshotHistory history;
  Buffer live{(1024 + 137) * 320 * sizeof(CacheT)};
  int64_t pos = 0;
  uint64_t copied = 0;
  cudaStream_t stream() const { return nullptr; }
  size_t session_snapshot_bytes() const { return (1024 + 128) * 320 * sizeof(CacheT); }
  int64_t session_position(int) const { return pos; }
  void register_state_snapshot(const void* dst) { history.register_destination(dst); }
  void unregister_state_snapshot(const void* dst) { history.unregister_destination(dst); }
  void invalidate_state_snapshot(const void* dst) { history.release(dst); }
  void session_release_snapshot(const SessionSnapshotMeta&) {}
  void reset() {
    history.rewind(0);
    pos = 0;
  }
  void rollback(int64_t p) {
    history.rewind(0, p);
    pos = p;
  }
  void fill(int64_t first, int64_t end, int salt) {
    size_t offset = 0;
    for (const auto& s : shapes) {
      std::vector<CacheT> values(size_t(s.capacity) * 320);
      DGPP_CUDA_OK(cudaMemcpy(values.data(), live.p + offset, values.size() * sizeof(CacheT), cudaMemcpyDeviceToHost));
      for (int64_t p = first; p < end; ++p) {
        const int64_t slot = s.window ? p % s.capacity : p;
        for (int j = 0; j < s.k_width(); ++j)
          values[slot * s.k_width() + j] = CacheT(p * 71 + j + salt);
        for (int j = 0; j < s.v_width(); ++j)
          values[size_t(s.capacity) * s.k_width() + slot * s.v_width() + j] = CacheT(p * 37 + j + salt);
      }
      DGPP_CUDA_OK(cudaMemcpy(live.p + offset, values.data(), values.size() * sizeof(CacheT), cudaMemcpyHostToDevice));
      offset += values.size() * sizeof(CacheT);
    }
    pos = end;
  }
  SessionSnapshotMeta snapshot(int req, void* dst, int64_t p) {
    const int64_t previous = history.begin(req, dst, p);
    auto* d = static_cast<uint8_t*>(dst);
    size_t offset = 0;
    for (const auto& s : shapes) {
      auto* k = reinterpret_cast<CacheT*>(live.p + offset);
      copied += dgpp::mimo_write_snapshot_layer(s, p, previous, k, k + s.capacity * s.k_width(), d, stream());
      d += size_t(s.window ? s.window : s.capacity) * 320 * sizeof(CacheT);
      offset += size_t(s.capacity) * 320 * sizeof(CacheT);
    }
    history.commit(req, dst, p);
    return {p};
  }
  SessionSnapshotMeta session_snapshot(int req, void* dst) { return snapshot(req, dst, pos); }
  SessionSnapshotMeta session_snapshot_post_row0(int req, void* dst, int, int after) {
    return snapshot(req, dst, pos - after);
  }
  void session_attach(int req, const void* src, SessionSnapshotMeta meta) {
    history.rewind(req);
    const auto* d = static_cast<const uint8_t*>(src);
    size_t offset = 0;
    for (const auto& s : shapes) {
      auto* k = reinterpret_cast<CacheT*>(live.p + offset);
      dgpp::mimo_read_snapshot_layer(s, meta.position, d, k, k + s.capacity * s.k_width(), stream());
      offset += size_t(s.capacity) * 320 * sizeof(CacheT);
      d += size_t(s.window ? s.window : s.capacity) * 320 * sizeof(CacheT);
    }
    pos = meta.position;
  }
  // Independent host oracle constructs the expected packed snapshot directly
  // from physical cache indices; compare only populated history, not capacity.
  void verify(const void* src, int64_t p) {
    size_t live_offset = 0, snap_offset = 0;
    for (const auto& s : shapes) {
      std::vector<CacheT> cache(size_t(s.capacity) * 320), snap(size_t(s.window ? s.window : s.capacity) * 320);
      DGPP_CUDA_OK(cudaMemcpy(cache.data(), live.p + live_offset, cache.size() * sizeof(CacheT), cudaMemcpyDeviceToHost));
      DGPP_CUDA_OK(cudaMemcpy(snap.data(), static_cast<const uint8_t*>(src) + snap_offset, snap.size() * sizeof(CacheT), cudaMemcpyDeviceToHost));
      const int64_t count = s.window ? std::min<int64_t>(p, s.window) : p;
      for (int64_t i = 0; i < count; ++i) {
        const int64_t index = s.window ? (p - count + i) % s.capacity : i;
        for (int j = 0; j < s.k_width(); ++j)
          check(snap[i * s.k_width() + j] == cache[index * s.k_width() + j], "K differs from full-copy oracle");
        for (int j = 0; j < s.v_width(); ++j)
          check(snap[size_t(s.window ? s.window : s.capacity) * s.k_width() + i * s.v_width() + j] ==
                    cache[size_t(s.capacity) * s.k_width() + index * s.v_width() + j], "V differs from full-copy oracle");
      }
      live_offset += cache.size() * sizeof(CacheT);
      snap_offset += snap.size() * sizeof(CacheT);
    }
  }
};
template<class CacheT>
void run() {
  Model<CacheT> m;
  dgpp::PrefixArena<Model<CacheT>> arena(&m, 4);
  m.fill(0, 120, 1);
  for (int s = 0; s < 4; ++s) arena.snapshot(0, s, m.pos);
  for (int end : {127, 128, 129, 137, 138, 260, 400}) {
    m.fill(m.pos, end, 1);
    for (int s = 0; s < 4; ++s) {
      const auto before = m.copied;
      const auto old = arena.position(s);
      arena.snapshot(0, s, m.pos);
      check(m.copied - before == uint64_t(end - old + std::min(end, 128)) * (320 * sizeof(CacheT)), "incremental byte budget");
      m.verify(arena.slot_data(s), end);
    }
  }
  // Speculative tail stays physically resident. A hop captures an earlier
  // committed position, then rejection rewinds the live mirror.
  m.fill(400, 404, 2);
  arena.snapshot_post_row0(0, 0, 401, 0, 3);
  m.verify(arena.slot_data(0), 401);
  arena.snapshot(0, 1, 404);  // deliberately snapshot the soon-to-be-rejected tail
  m.rollback(401);
  m.fill(401, 407, 3);
  const auto before = m.copied;
  arena.snapshot(0, 1, 407);
  check(m.copied - before == uint64_t(407 + 128) * (320 * sizeof(CacheT)), "rollback did not force full copy");
  m.verify(arena.slot_data(1), 407);
  // Restore must exactly replace populated entries while preserving all tails.
  Buffer expected((1024 + 137) * 320 * sizeof(CacheT));
  DGPP_CUDA_OK(cudaMemcpy(expected.p, m.live.p, (1024 + 137) * 320 * sizeof(CacheT), cudaMemcpyDeviceToDevice));
  m.reset();
  DGPP_CUDA_OK(cudaMemset(m.live.p, 0xa5, (1024 + 137) * 320 * sizeof(CacheT)));
  arena.attach(0, 1);
  m.verify(arena.slot_data(1), 407);
  // Restored live data must match the pre-reset cache wherever attention reads.
  std::vector<CacheT> old((1024 + 137) * 320), restored(old.size());
  DGPP_CUDA_OK(cudaMemcpy(old.data(), expected.p, old.size() * sizeof(CacheT), cudaMemcpyDeviceToHost));
  DGPP_CUDA_OK(cudaMemcpy(restored.data(), m.live.p, restored.size() * sizeof(CacheT), cudaMemcpyDeviceToHost));
  size_t offset = 0;
  for (const auto& s : m.shapes) {
    for (int slot = 0; slot < s.capacity; ++slot) {
      bool populated = !s.window ? slot < 407 : false;
      if (s.window) for (int p = 407 - 128; p < 407; ++p) populated |= p % s.capacity == slot;
      for (int width : {s.k_width(), s.v_width()}) {
        const size_t base = offset + (width == s.k_width() ? 0 : s.capacity * s.k_width());
        for (int j = 0; j < width; ++j)
          check(restored[base + slot * width + j] == (populated ? old[base + slot * width + j] : CacheT(0xa5a5)), "restore altered tail or populated bits");
      }
    }
    offset += size_t(s.capacity) * 320;
  }
  // Cancellation/close then slot reuse at a longer position with different data.
  m.reset();
  m.fill(0, 500, 51);
  for (int s = 0; s < 4; ++s) {
    arena.snapshot(0, s, 500);
    m.verify(arena.slot_data(s), 500);
  }
  // Explicit release, destination reassignment and prefill request each invalidate.
  arena.release(0);
  auto b = m.copied;
  arena.snapshot(0, 0, 500);
  check(m.copied - b == uint64_t(500 + 128) * (320 * sizeof(CacheT)), "release provenance");
  b = m.copied;
  arena.snapshot(1, 0, 500);
  check(m.copied - b == uint64_t(500 + 128) * (320 * sizeof(CacheT)), "reassignment provenance");
  auto request = arena.request(0, 500);
  b = m.copied;
  *request.meta = m.session_snapshot(1, request.dst);
  request.taken = true;
  arena.commit(0, request);
  check(m.copied - b == uint64_t(500 + 128) * (320 * sizeof(CacheT)), "prefill request provenance");
  m.verify(arena.slot_data(0), 500);
  std::cout << "PASS: exact rolling/hop snapshots, wraps, rollback, restore, four-slot reuse; copied=" << m.copied << " bytes\n";
}
void shared_gpu_storage() {
  cudaStream_t stream;
  DGPP_CUDA_OK(cudaStreamCreate(&stream));
  {
    Buffer live(1024), restored(1024);
    dgpp::MimoSharedSnapshots store(1, 4, [stream](size_t n) {
      uint8_t* ptr = nullptr;
      DGPP_CUDA_OK(cudaMallocAsync(&ptr, n, stream));
      return dgpp::MimoSharedSnapshots::Storage(ptr, [stream](uint8_t* p) { cudaFreeAsync(p, stream); });
    });
    int a, b;
    store.register_slot(&a); store.register_slot(&b);
    DGPP_CUDA_OK(cudaMemsetAsync(live.p, 17, 1024, stream));
    auto write = [&](void* dst, int64_t pos) {
      return store.write(0, dst, pos,
          [&](uint8_t* out, int64_t first, int64_t count) {
            DGPP_CUDA_OK(cudaMemcpyAsync(out, live.p + first, count, cudaMemcpyDeviceToDevice, stream));
          }, [&](uint8_t* out, int64_t, int64_t) {
            DGPP_CUDA_OK(cudaMemcpyAsync(out, live.p, 4, cudaMemcpyDeviceToDevice, stream));
          });
    };
    check(write(&a, 300) == 304, "shared GPU first copy");
    check(write(&b, 600) == 348, "shared GPU block reuse");
    check(store.unique_bytes() == 652, "shared GPU owned storage");
    DGPP_CUDA_OK(cudaMemsetAsync(live.p, 29, 1024, stream));
    store.rewind(0);
    write(&a, 300);  // old A must not corrupt B's shared block
    store.read(1, &b, 600,
        [&](const uint8_t* in, int64_t first, int64_t count) {
          DGPP_CUDA_OK(cudaMemcpyAsync(restored.p + first, in, count, cudaMemcpyDeviceToDevice, stream));
        }, [&](const uint8_t* in, int64_t, int64_t) {
          DGPP_CUDA_OK(cudaMemcpyAsync(restored.p + 600, in, 4, cudaMemcpyDeviceToDevice, stream));
        });
    store.release(&b);  // stream-ordered free must follow restore copies
    DGPP_CUDA_OK(cudaStreamSynchronize(stream));
    std::vector<uint8_t> got(604);
    DGPP_CUDA_OK(cudaMemcpy(got.data(), restored.p, got.size(), cudaMemcpyDeviceToHost));
    check(std::all_of(got.begin(), got.end(), [](uint8_t x) { return x == 17; }), "shared GPU restore changed bytes");
    store.unregister_slot(&a); store.unregister_slot(&b);
    check(store.unique_bytes() == 0, "shared GPU eviction leaked ownership");
    DGPP_CUDA_OK(cudaStreamSynchronize(stream));
  }
  DGPP_CUDA_OK(cudaStreamDestroy(stream));
  std::cout << "PASS: shared GPU snapshots, async allocation/free ordering, branch and eviction\n";
}
}  // namespace
int main() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || !count) return 2;
  try {
    run<uint16_t>();
    run<uint8_t>();
    shared_gpu_storage();
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
