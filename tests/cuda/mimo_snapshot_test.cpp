// No checkpoint or fabric required. Exercise production copy/restore helpers and
// PrefixArena lifetime hooks; all comparisons are bit-exact, including tails.
#include <cuda_runtime.h>

#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "engine/prefix_arena.hpp"
#include "engine/paged_blocks.hpp"
#include "engine/session_model.hpp"
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
// A checkpoint-free family supplies only its state copies; snapshot metadata,
// pointer advancement, hidden-row preservation and attach are SessionModel's.
struct SharedMtpModel : dgpp::SessionModel<SharedMtpModel> {
  using Layout = dgpp::MimoSharedSnapshots;
  static constexpr size_t draft_bytes = 96;
  static constexpr int width = 9;  // deliberately exercises slot tail padding
  bool old_layout;
  Buffer live{1024 + 16}, draft{draft_bytes};
  std::unique_ptr<Layout> store;
  struct UnusedPool : dgpp::PagedBlockTable {
    void copy_block_contents(int32_t, int32_t, cudaStream_t) {
      throw std::logic_error("test has no paged pool");
    }
  } unused;
  explicit SharedMtpModel(bool old = false) : old_layout(old) {
    dgpp::SessionParams p;
    p.max_tokens = 16; p.max_cache_tokens = 1024; p.max_position_embeddings = 1024;
    p.max_requests = 2; p.mtp = true; p.hidden = width; p.draft_width = width;
    p.vocab_size = 16; p.lm_vocab_count = 16;
    init_session(p);
    store = std::make_unique<Layout>(1, 16, [this](size_t n) {
      uint8_t* ptr = nullptr;
      DGPP_CUDA_OK(cudaMallocAsync(&ptr, n, stream_));
      return Layout::Storage(ptr, [stream = stream_](uint8_t* v) { cudaFreeAsync(v, stream); });
    });
  }
  ~SharedMtpModel() {
    store.reset();
    cudaStreamSynchronize(stream_);
  }
  static constexpr int prefill_chunk_tokens() { return 16; }
  bool has_pool() const { return false; }
  UnusedPool& pool() { return unused; }
  size_t snapshot_state_bytes() const { return 1040; }
  size_t draft_state_bytes() const { return draft_bytes; }
  size_t prefix_arena_storage_bytes() const {
    const auto suffix = session_snapshot_bytes() - snapshot_state_bytes();
    return old_layout ? suffix + 8 : Layout::slot_storage_bytes(suffix);
  }
  size_t snapshot_state_storage_bytes(const void* dst) const {
    check(store->contains(dst), "unregistered shared slot");
    return old_layout ? 8 : Layout::header_bytes;
  }
  void register_state_snapshot(const void* d) { store->register_slot(d); }
  void unregister_state_snapshot(const void* d) { store->unregister_slot(d); }
  void invalidate_state_snapshot(const void* d) { store->release(d); }
  void reset_slot_state(int req) { store->rewind(req); }
  void write_state_snapshot(int req, uint8_t* dst, int row) {
    const auto pos = session_pos_[req] - (row < 0 ? 0 : rows_after_for_snapshot_);
    store->write(req, dst, pos,
      [&](uint8_t* out, int64_t first, int64_t n) {
        DGPP_CUDA_OK(cudaMemcpyAsync(out, live.p + first, n, cudaMemcpyDeviceToDevice, stream_));
      }, [&](uint8_t* out, int64_t, int64_t) {
        dgpp::glm_device_copy(out, live.p + 1024, 16, stream_);
      });
  }
  void read_state_snapshot(int req, const uint8_t* src, int64_t pos) {
    store->read(req, src, pos,
      [&](const uint8_t* in, int64_t first, int64_t n) {
        DGPP_CUDA_OK(cudaMemcpyAsync(live.p + first, in, n, cudaMemcpyDeviceToDevice, stream_));
      }, [&](const uint8_t* in, int64_t, int64_t) {
        dgpp::glm_device_copy(live.p + 1024, in, 16, stream_);
      });
  }
  void write_draft_snapshot(int, uint8_t* dst, bool, int64_t) {
    // Multiple aligned segments model native heads' independent state planes.
    for (size_t offset = 0; offset < draft_bytes; offset += 32)
      dgpp::glm_device_copy(dst + offset, draft.p + offset, 32, stream_);
  }
  void read_draft_snapshot(int, const uint8_t* src) {
    for (size_t offset = 0; offset < draft_bytes; offset += 32)
      dgpp::glm_device_copy(draft.p + offset, src + offset, 32, stream_);
  }
  void seed(int64_t pos, int salt, bool hop = false) {
    reset_slot_state(0);
    session_pos_[0] = pos + (hop ? 1 : 0);
    mtp_pos_[0] = hop ? pos - 1 : pos;
    DGPP_CUDA_OK(cudaMemsetAsync(live.p, salt, 1040, stream_));
    std::array<uint8_t, draft_bytes> values;
    for (size_t i = 0; i < values.size(); ++i) values[i] = uint8_t(salt + 7 * i);
    DGPP_CUDA_OK(cudaMemcpyAsync(draft.p, values.data(), values.size(), cudaMemcpyHostToDevice, stream_));
    DGPP_CUDA_OK(cudaMemsetAsync(mtp_window(0), salt + 1, max_decode_rows_ * width * 2, stream_));
    DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
  }
  void poison() {
    DGPP_CUDA_OK(cudaMemsetAsync(live.p, 0, 1040, stream_));
    DGPP_CUDA_OK(cudaMemsetAsync(draft.p, 0, draft_bytes, stream_));
    DGPP_CUDA_OK(cudaMemsetAsync(mtp_window(1), 0, max_decode_rows_ * width * 2, stream_));
  }
  void verify(int64_t pos, int salt, bool hop) {
    DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
    check(session_position(1) == pos && mtp_pos_[1] == pos - (hop ? 1 : 0), "session/draft position restore");
    std::vector<uint8_t> cache(1040), got(draft_bytes), hidden(width * 2);
    DGPP_CUDA_OK(cudaMemcpy(cache.data(), live.p, cache.size(), cudaMemcpyDeviceToHost));
    DGPP_CUDA_OK(cudaMemcpy(got.data(), draft.p, got.size(), cudaMemcpyDeviceToHost));
    DGPP_CUDA_OK(cudaMemcpy(hidden.data(), mtp_window(1) + ((pos - 1) % max_decode_rows_) * width,
                           hidden.size(), cudaMemcpyDeviceToHost));
    for (int64_t i = 0; i < pos; ++i) check(cache[i] == salt, "shared global restore");
    for (int i = 1024; i < 1040; ++i) check(cache[i] == salt, "shared private restore");
    for (size_t i = 0; i < got.size(); ++i) check(got[i] == uint8_t(salt + 7 * i), "MTP payload restore");
    for (auto v : hidden) check(v == salt + 1, "MTP hidden row restore");
  }
};
void shared_session_mtp() {
  {
    SharedMtpModel broken(true);
    dgpp::PrefixArena<SharedMtpModel> arena(&broken, 4);
    broken.seed(300, 17);
    bool rejected = false;
    try { arena.snapshot(0, 0, 300); }
    catch (const std::invalid_argument& e) {
      rejected = std::string(e.what()) == "glm_device_copy: alignment/size";
    }
    check(rejected, "negative control did not catch the old +8 payload layout");
  }
  SharedMtpModel m;
  dgpp::PrefixArena<SharedMtpModel> arena(&m, 4);
  for (int pass = 0; pass < 2; ++pass) {
    for (int slot = 0; slot < 4; ++slot) {
      const int64_t pos = 300 + 7 * slot + 100 * pass;
      const int salt = 17 + slot + 31 * pass;
      const bool hop = slot % 2;
      check(reinterpret_cast<uintptr_t>(arena.slot_data(slot)) % 16 == 0, "arena slot alignment");
      m.seed(pos, salt, hop);
      if (hop) arena.snapshot_post_row0(0, slot, pos, 0, 1);
      else arena.snapshot(0, slot, pos);
    }
    for (int slot = 0; slot < 4; ++slot) {
      m.session_close(1);
      m.poison();
      arena.attach(1, slot);
      m.verify(300 + 7 * slot + 100 * pass, 17 + slot + 31 * pass, slot % 2);
    }
  }
  for (int slot = 0; slot < 4; ++slot) arena.release(slot);
  check(m.store->unique_bytes() == 0, "session shared storage leaked after release");
  std::cout << "PASS: real SessionModel MTP snapshots/attach, four slots, hop/reuse, +8 negative control\n";
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
    shared_session_mtp();
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
