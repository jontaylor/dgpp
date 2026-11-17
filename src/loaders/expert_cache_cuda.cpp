#include "loaders/expert_cache_cuda.hpp"

#include <array>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <limits>
#include <stdexcept>
#include <utility>

#include "common/cuda_check.hpp"
#include "kernels/moe_fp4_pack.hpp"

namespace dgpp::expert {
namespace {
class CudaCacheStorage final : public ExpertCacheStorage {
 public:
  CudaCacheStorage(CacheMemory memory, std::span<const std::span<uint8_t>> staging,
                   StoreIdentity identity, cudaStream_t stream, bool packed_fp4)
      : memory_(memory), identity_(identity), stream_(stream), packed_fp4_(packed_fp4) {
    full_ = partition(identity.hidden, identity.intermediate, identity.world, identity.rank,
                      identity.intermediate / identity.world, identity.format).front();
    if (packed_fp4_ && (memory_.execution_slot_bytes != 2 * full_.allocation_bytes ||
        moe_fp4_packed_bytes(identity.hidden, identity.intermediate / identity.world, scale_group()) > full_.allocation_bytes))
      throw std::invalid_argument("expert CUDA cache: invalid packed FP4 reservation");
    // Precompute the pitched-copy lists at startup: no hot-path metadata or
    // payload allocations. Store validation guarantees authoritative layouts.
    layouts_ = partition(identity.hidden, identity.intermediate, identity.world,
                         identity.rank, identity.chunk_intermediate, identity.format);
    copies_.reserve(layouts_.size());
    for (const auto& c : layouts_)
      copies_.push_back(scatter_copies(c, identity.hidden, identity.intermediate / identity.world,
                                      identity.format, identity.rank * (identity.intermediate / identity.world)));
    const char* scatter = std::getenv("DGPP_EXPERT_GPU_SCATTER");
    if (scatter) gpu_scatter_ = scatter[0] == '1';
    else {
      int device = 0, integrated = 0;
      DGPP_CUDA_OK(cudaGetDevice(&device));
      DGPP_CUDA_OK(cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, device));
      gpu_scatter_ = integrated != 0;
    }
    if (gpu_scatter_) {
      for (const auto& copies : copies_) {
        scatter_plans_.push_back(expert_chunk_copy_plan(copies));
        auto reverse = copies;
        for (auto& c : reverse) {
          std::swap(c.source_offset, c.destination_offset);
          std::swap(c.source_pitch, c.destination_pitch);
        }
        gather_plans_.push_back(expert_chunk_copy_plan(reverse));
      }
    }
    const char* batch = std::getenv("DGPP_EXPERT_BATCH_SCATTER");
    batch_scatter_ = batch && std::atoi(batch) != 0;
    const char* direct = std::getenv("DGPP_EXPERT_CHUNK_DIRECT");
    chunk_direct_ = direct && std::atoi(direct) != 0;
    registered_.reserve(staging.size());
    mapped_.reserve(staging.size());
    try {
      DGPP_CUDA_OK(cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking));
      create_event(ready_);
      create_event(native_readers_);
      execution_readers_.resize(memory_.execution_slots);
      native_extent_readers_.resize((memory_.cache_slots + layouts_.size() - 1) / layouts_.size());
      staging_done_.resize(staging.size());
      staging_readers_.resize(staging.size());
      for (auto& event : execution_readers_) create_event(event);
      for (auto& event : native_extent_readers_) create_event(event);
      for (auto& event : staging_done_) create_event(event);
      for (auto& event : staging_readers_) create_event(event);
      if (chunk_direct_) {
        metadata_done_.resize(memory_.execution_slots);
        for (auto& event : metadata_done_) create_event(event);
        DGPP_CUDA_OK(cudaMallocHost(reinterpret_cast<void**>(&chunk_sources_),
            memory_.execution_slots * layouts_.size() * sizeof(const uint8_t*)));
      }
      if (memory_.cache_slots)
        DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&cache_), memory_.cache_slots * memory_.cache_slot_bytes));
      DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&execution_), memory_.execution_slots * memory_.execution_slot_bytes));
      for (const auto span : staging) {
        DGPP_CUDA_OK(cudaHostRegister(span.data(), span.size(), cudaHostRegisterDefault));
        registered_.push_back(span.data());
        if (gpu_scatter_ || chunk_direct_) {
          void* device = nullptr;
          DGPP_CUDA_OK(cudaHostGetDevicePointer(&device, span.data(), 0));
          mapped_.push_back(static_cast<const uint8_t*>(device));
        }
      }
    } catch (...) { cleanup(); throw; }
  }
  ~CudaCacheStorage() override { cleanup(); }
  const CacheMemory& memory() const override { return memory_; }
  void retain(size_t slot, std::span<const uint8_t> bytes) override {
    if (slot >= memory_.cache_slots || bytes.size() > memory_.cache_slot_bytes)
      throw std::invalid_argument("expert CUDA cache: invalid retention span");
    wait_cache(slot);
    Transfer transfer(*this, bytes.data());
    DGPP_CUDA_OK(cudaMemcpyAsync(cache_ + slot * memory_.cache_slot_bytes, bytes.data(), bytes.size(),
                                 cudaMemcpyHostToDevice, copy_stream_));
  }
  void begin_execution(size_t slot) override {
    check_execution(slot);
    // assemble() overwrites every payload, scale and global of the complete
    // expert. Packing likewise overwrites its companion's tensor ranges.
    // Neither consumer reads allocation padding. Clearing both reservations
    // first adds a full memory sweep without supplying any consumed values.
  }
  uint64_t execution_zero_bytes_per_begin() const override { return 0; }
  void assemble(size_t execution_slot, const StoreRecord& record,
      std::optional<size_t> cache_slot, std::span<const uint8_t> host_bytes) override {
    check_execution(execution_slot);
    wait_execution(execution_slot);
    Transfer transfer(*this, host_bytes.empty() ? nullptr : host_bytes.data());
    if (record.key.chunk >= copies_.size()) throw std::invalid_argument("expert CUDA cache: chunk index");
    const uint8_t* source = nullptr;
    auto kind = cudaMemcpyHostToDevice;
    if (cache_slot) {
      if (*cache_slot >= memory_.cache_slots || !host_bytes.empty())
        throw std::invalid_argument("expert CUDA cache: invalid cache source");
      source = cache_ + *cache_slot * memory_.cache_slot_bytes;
      kind = cudaMemcpyDeviceToDevice;
    } else {
      if (host_bytes.size() != layouts_[record.key.chunk].allocation_bytes)
        throw std::invalid_argument("expert CUDA cache: invalid staging source");
      source = host_bytes.data();
    }
    const auto& expected = layouts_[record.key.chunk];
    if (record.layout.intermediate_begin != expected.intermediate_begin ||
        record.layout.intermediate_size != expected.intermediate_size ||
        record.layout.allocation_bytes != expected.allocation_bytes)
      throw std::invalid_argument("expert CUDA cache: record geometry disagrees with TP layout");
    auto* destination = execution_ + execution_slot * memory_.execution_slot_bytes;
    if (gpu_scatter_) {
      if (!cache_slot) {
        source = mapped_[staging_index(source)];
      }
      launch_expert_chunk_scatter(destination, source, scatter_plans_[record.key.chunk], copy_stream_);
      return;
    }
    for (const auto& c : copies_[record.key.chunk])
      DGPP_CUDA_OK(cudaMemcpy2DAsync(destination + c.destination_offset, c.destination_pitch,
                                     source + c.source_offset, c.source_pitch, c.row_bytes, c.rows, kind, copy_stream_));
  }
  void assemble_many(std::span<const Assembly> ready) override {
    if (ready.empty()) return;
    // Keep the existing path for unsupported layouts and tiny completions.
    // Resident-source assembly continues to use its original ordering.
    if (!batch_scatter_ || !gpu_scatter_ || ready.size() == 1 ||
        std::any_of(ready.begin(), ready.end(), [](const auto& a) { return a.cache_slot.has_value(); })) {
      ExpertCacheStorage::assemble_many(ready);
      return;
    }
    Transfer transfer(*this);
    std::array<ExpertChunkScatter, kExpertScatterBatch> chunks;
    for (size_t base = 0; base < ready.size(); base += chunks.size()) {
      const size_t count = std::min(chunks.size(), ready.size() - base);
      for (size_t i = 0; i < count; ++i) {
        const auto& a = ready[base + i];
        check_execution(a.execution_slot);
        if (!a.record || a.record->key.chunk >= layouts_.size())
          throw std::invalid_argument("expert CUDA cache: batch chunk index");
        const auto chunk = a.record->key.chunk;
        const auto& expected = layouts_[chunk];
        if (a.host_bytes.size() != expected.allocation_bytes ||
            a.record->layout.intermediate_begin != expected.intermediate_begin ||
            a.record->layout.intermediate_size != expected.intermediate_size ||
            a.record->layout.allocation_bytes != expected.allocation_bytes)
          throw std::invalid_argument("expert CUDA cache: batch record geometry");
        wait_execution(a.execution_slot);
        chunks[i] = {execution_ + a.execution_slot * memory_.execution_slot_bytes,
                     mapped_.at(staging_index(a.host_bytes.data())), scatter_plans_[chunk]};
      }
      launch_expert_chunk_scatter_batch(std::span<const ExpertChunkScatter>(chunks).first(count), copy_stream_);
      // Do not release any source before the kernel containing its last read.
      // Ready records already have owners in ExpertCache::pending_copies.
      for (size_t i = 0; i < count; ++i)
        DGPP_CUDA_OK(cudaEventRecord(staging_done_.at(staging_index(ready[base + i].host_bytes.data())), copy_stream_));
    }
  }
  const uint8_t* execution_data(size_t slot) const override {
    check_execution(slot);
    publish();
    return execution_ + slot * memory_.execution_slot_bytes;
  }
  const uint8_t* packed_execution_data(size_t slot) override {
    check_execution(slot);
    if (!packed_fp4_) throw std::logic_error("expert CUDA cache: packed layout disabled");
    auto* source = execution_ + slot * memory_.execution_slot_bytes;
    auto* destination = source + full_.allocation_bytes;
    wait_execution(slot);
    { Transfer transfer(*this); pack(destination, source); }
    publish();
    return destination;
  }
  const uint8_t* cache_data(size_t slot) const override {
    check_cache(slot);
    publish();
    return cache_ + slot * memory_.cache_slot_bytes;
  }
  void move_cache_slot(size_t dest, size_t source) override {
    check_cache(dest);
    check_cache(source);
    wait_cache(dest);
    Transfer transfer(*this);
    if (gpu_scatter_) {
      const auto bytes = memory_.cache_slot_bytes;
      const ScatterCopy copy{0, 0, bytes, 1, bytes, bytes};
      const auto plan = expert_chunk_copy_plan(std::span(&copy, 1));
      launch_expert_chunk_scatter(cache_ + dest * bytes, cache_ + source * bytes, plan, copy_stream_);
    } else {
      DGPP_CUDA_OK(cudaMemcpyAsync(cache_ + dest * memory_.cache_slot_bytes,
          cache_ + source * memory_.cache_slot_bytes, memory_.cache_slot_bytes,
          cudaMemcpyDeviceToDevice, copy_stream_));
    }
  }
  void native_to_execution(size_t execution_slot, size_t first) override {
    check_native(execution_slot, first);
    wait_execution(execution_slot);
    Transfer transfer(*this);
    if (packed_fp4_) {
      auto* destination = execution_ + execution_slot * memory_.execution_slot_bytes;
      auto* source = cache_ + first * memory_.cache_slot_bytes;
      Fp4ExpertDestination out;
      const std::array<MatrixLayout, 3> matrices{full_.gate, full_.up, full_.down};
      for (size_t m = 0; m < matrices.size(); ++m) {
        out.payload[m] = destination + matrices[m].payload_offset;
        out.scales[m] = destination + matrices[m].scales_offset;
        out.globals[m] = matrices[m].global_offset ? reinterpret_cast<float*>(destination + *matrices[m].global_offset) : nullptr;
      }
      launch_moe_fp4_unpack(out, moe_fp4_packed_view(source, identity_.hidden,
          identity_.intermediate / identity_.world, scale_group()), identity_.hidden,
          identity_.intermediate / identity_.world, copy_stream_, scale_group());
      return;
    }
    DGPP_CUDA_OK(cudaMemcpyAsync(execution_ + execution_slot * memory_.execution_slot_bytes,
        cache_ + first * memory_.cache_slot_bytes, memory_.execution_slot_bytes,
        cudaMemcpyDeviceToDevice, copy_stream_));
  }
  void execution_to_native(size_t execution_slot, size_t first) override {
    check_native(execution_slot, first);
    for (size_t c = 0; c < layouts_.size(); ++c) wait_cache(first+c);
    Transfer transfer(*this);
    if (packed_fp4_) {
      pack(cache_ + first * memory_.cache_slot_bytes,
           execution_ + execution_slot * memory_.execution_slot_bytes);
      return;
    }
    DGPP_CUDA_OK(cudaMemcpyAsync(cache_ + first * memory_.cache_slot_bytes,
        execution_ + execution_slot * memory_.execution_slot_bytes, memory_.execution_slot_bytes,
        cudaMemcpyDeviceToDevice, copy_stream_));
  }
  bool packed_execution_to_native(size_t execution_slot, size_t first) override {
    if (!packed_fp4_) return false;
    check_native(execution_slot, first);
    for (size_t c = 0; c < layouts_.size(); ++c) wait_cache(first+c);
    Transfer transfer(*this);
    const size_t bytes = moe_fp4_packed_bytes(identity_.hidden, identity_.intermediate / identity_.world, scale_group());
    DGPP_CUDA_OK(cudaMemcpyAsync(cache_ + first * memory_.cache_slot_bytes,
        execution_ + execution_slot * memory_.execution_slot_bytes + full_.allocation_bytes,
        bytes, cudaMemcpyDeviceToDevice, copy_stream_));
    return true;
  }
  void unpack_native_chunk(size_t execution_slot, const StoreRecord& record, size_t cache_slot) override {
    check_execution(execution_slot);
    check_cache(cache_slot);
    wait_cache(cache_slot);
    Transfer transfer(*this);
    if (record.key.chunk >= copies_.size()) throw std::invalid_argument("expert CUDA cache: native chunk index");
    auto* dest = cache_ + cache_slot * memory_.cache_slot_bytes;
    const auto* source = execution_ + execution_slot * memory_.execution_slot_bytes;
    // The gather overwrites every tensor byte; allocation padding is never
    // consumed by a matrix kernel or included in a reconstructed tensor.
    if (gpu_scatter_) {
      launch_expert_chunk_scatter(dest, source, gather_plans_[record.key.chunk], copy_stream_);
      return;
    }
    for (const auto& c : copies_[record.key.chunk])
      DGPP_CUDA_OK(cudaMemcpy2DAsync(dest + c.source_offset, c.source_pitch,
          source + c.destination_offset, c.destination_pitch, c.row_bytes, c.rows,
          cudaMemcpyDeviceToDevice, copy_stream_));
  }
  // CPU lease retirement records device readers; subsequent writes wait only
  // for the storage they reuse. Transfers to other scratch slots can overlap.
  void retire_native() override { DGPP_CUDA_OK(cudaEventRecord(native_readers_, stream_)); }
  void retire_native_extent(size_t first, size_t count) override {
    for (size_t group = first / layouts_.size(); group <= (first+count-1) / layouts_.size(); ++group)
      DGPP_CUDA_OK(cudaEventRecord(native_extent_readers_.at(group), stream_));
  }
  void retire_native_extents(std::span<const std::pair<size_t, size_t>> extents) override {
    for (auto [first, count] : extents) retire_native_extent(first, count);
  }
  void retire_execution(size_t slot) override {
    DGPP_CUDA_OK(cudaEventRecord(execution_readers_.at(slot), stream_));
  }
  bool supports_chunk_direct() const override { return chunk_direct_; }
  const uint8_t* staging_data(std::span<const uint8_t> bytes) const override {
    if (!chunk_direct_) throw std::logic_error("expert CUDA cache: chunk-direct disabled");
    return mapped_.at(staging_index(bytes.data()));
  }
  void bind_chunk_sources(size_t slot, std::span<const uint8_t* const> sources) override {
    check_execution(slot);
    if (!chunk_direct_ || sources.size() != layouts_.size() || sources.size_bytes() > memory_.execution_slot_bytes)
      throw std::invalid_argument("expert CUDA cache: invalid chunk binding");
    // Pinned pointer metadata is bounded independently of payload staging.
    DGPP_CUDA_OK(cudaEventSynchronize(metadata_done_[slot]));
    auto* host = chunk_sources_ + slot * layouts_.size();
    std::copy(sources.begin(), sources.end(), host);
    wait_execution(slot);
    Transfer transfer(*this);
    DGPP_CUDA_OK(cudaMemcpyAsync(execution_ + slot * memory_.execution_slot_bytes,
        host, sources.size_bytes(), cudaMemcpyHostToDevice, copy_stream_));
    DGPP_CUDA_OK(cudaEventRecord(metadata_done_[slot], copy_stream_));
  }
  void retire_staging_reader(std::span<const uint8_t> bytes) override {
    DGPP_CUDA_OK(cudaEventRecord(staging_readers_.at(staging_index(bytes.data())), stream_));
  }
  void order_mutations() override {} // all storage operations share copy_stream_
  bool staging_complete(std::span<const uint8_t> bytes, bool wait) override {
    const size_t index = staging_index(bytes.data());
    // Coalesced reads can have both transfer-stream and compute-stream
    // consumers. A later transfer must not erase an earlier kernel's fence.
    const std::array<cudaEvent_t,2> events{staging_done_[index], staging_readers_[index]};
    if (!wait) {
      for (auto event : events) {
        const auto status = cudaEventQuery(event);
        if (status == cudaErrorNotReady) return false;
        DGPP_CUDA_OK(status);
      }
      return true;
    }
    const auto started = std::chrono::steady_clock::now();
    for (auto event : events) DGPP_CUDA_OK(cudaEventSynchronize(event));
    completion_wait_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - started).count();
    return true;
  }

  void finish() override {
    const auto started = std::chrono::steady_clock::now();
    DGPP_CUDA_OK(cudaStreamSynchronize(copy_stream_));
    DGPP_CUDA_OK(cudaStreamSynchronize(stream_));
    completion_wait_ns_ += std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - started).count();
  }
  uint64_t completion_wait_ns() const override { return completion_wait_ns_; }
  bool accepts_execution_context(const void* context) const override { return context == stream_; }
  void set_execution_context(void* context) override {
    if (!context) throw std::invalid_argument("expert cache: null execution stream");
    stream_ = static_cast<cudaStream_t>(context);
    published_generation_ = 0;
  }
 private:
  static void create_event(cudaEvent_t& event) {
    DGPP_CUDA_OK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
  }
  // Publication follows the last transfer, including both uses of a staging
  // buffer when a demanded chunk is assembled and retained.
  struct Transfer {
    CudaCacheStorage& owner;
    const uint8_t* staging;
    Transfer(CudaCacheStorage& o, const uint8_t* p = nullptr) : owner(o), staging(p) {}
    ~Transfer() noexcept(false) {
      if (std::uncaught_exceptions()) return;
      if (staging) DGPP_CUDA_OK(cudaEventRecord(owner.staging_done_.at(owner.staging_index(staging)), owner.copy_stream_));
      DGPP_CUDA_OK(cudaEventRecord(owner.ready_, owner.copy_stream_));
      ++owner.generation_;
    }
  };
  void publish() const {
    if (published_generation_ == generation_) return;
    DGPP_CUDA_OK(cudaStreamWaitEvent(stream_, ready_, 0));
    published_generation_ = generation_;
  }
  void wait_execution(size_t slot) {
    DGPP_CUDA_OK(cudaStreamWaitEvent(copy_stream_, execution_readers_.at(slot), 0));
  }
  void wait_cache(size_t slot) {
    DGPP_CUDA_OK(cudaStreamWaitEvent(copy_stream_, native_readers_, 0));
    DGPP_CUDA_OK(cudaStreamWaitEvent(copy_stream_, native_extent_readers_.at(slot / layouts_.size()), 0));
  }
  uint64_t completion_wait_ns_ = 0;
  size_t staging_index(const uint8_t* source) const {
    size_t index = 0;
    while (index < registered_.size() && registered_[index] != source) ++index;
    if (index == registered_.size()) throw std::invalid_argument("expert CUDA cache: unregistered staging");
    return index;
  }
  int scale_group() const { return identity_.format == Format::kNvfp4 ? 16 : 32; }
  void pack(uint8_t* destination, const uint8_t* source) {
    Fp4ExpertSource in;
    const std::array<MatrixLayout, 3> matrices{full_.gate, full_.up, full_.down};
    for (size_t m = 0; m < matrices.size(); ++m) {
      in.payload[m] = source + matrices[m].payload_offset;
      in.scales[m] = source + matrices[m].scales_offset;
      in.globals[m] = matrices[m].global_offset ? reinterpret_cast<const float*>(source + *matrices[m].global_offset) : nullptr;
    }
    launch_moe_fp4_pack(moe_fp4_packed_view(destination, identity_.hidden,
        identity_.intermediate / identity_.world, scale_group()), in, identity_.hidden,
        identity_.intermediate / identity_.world, copy_stream_, scale_group());
  }
  void check_cache(size_t slot) const {
    if (slot >= memory_.cache_slots) throw std::invalid_argument("expert CUDA cache: cache slot");
  }
  void check_native(size_t slot, size_t first) const {
    check_execution(slot);
    check_cache(first);
    if (full_.allocation_bytes > (memory_.cache_slots - first) * memory_.cache_slot_bytes)
      throw std::invalid_argument("expert CUDA cache: native extent exceeds slab");
  }
  void check_execution(size_t slot) const {
    if (slot >= memory_.execution_slots) throw std::out_of_range("expert CUDA cache: execution slot");
  }
  void cleanup() noexcept {
    // Drain before unregister/free. On device/context failure the error is
    // already terminal to the owner; destructors cannot propagate it.
    if (copy_stream_) (void)cudaStreamSynchronize(copy_stream_);
    (void)cudaStreamSynchronize(stream_);
    for (auto event : metadata_done_) if (event) (void)cudaEventDestroy(event);
    if (chunk_sources_) (void)cudaFreeHost(chunk_sources_);
    for (auto event : staging_readers_) if (event) (void)cudaEventDestroy(event);
    for (auto event : staging_done_) if (event) (void)cudaEventDestroy(event);
    for (auto event : execution_readers_) if (event) (void)cudaEventDestroy(event);
    for (auto event : native_extent_readers_) if (event) (void)cudaEventDestroy(event);
    if (ready_) (void)cudaEventDestroy(ready_);
    if (native_readers_) (void)cudaEventDestroy(native_readers_);
    if (copy_stream_) (void)cudaStreamDestroy(copy_stream_);
    for (auto* ptr : registered_) (void)cudaHostUnregister(ptr);
    registered_.clear();
    if (execution_) (void)cudaFree(std::exchange(execution_, nullptr));
    if (cache_) (void)cudaFree(std::exchange(cache_, nullptr));
  }
  CacheMemory memory_;
  StoreIdentity identity_;
  cudaStream_t stream_;
  cudaStream_t copy_stream_ = nullptr;
  cudaEvent_t ready_ = nullptr, native_readers_ = nullptr;
  std::vector<cudaEvent_t> staging_done_, staging_readers_, execution_readers_, native_extent_readers_;
  uint64_t generation_ = 0;
  mutable uint64_t published_generation_ = 0;
  bool chunk_direct_ = false;
  const uint8_t** chunk_sources_ = nullptr;
  std::vector<cudaEvent_t> metadata_done_;
  bool packed_fp4_ = false;
  bool gpu_scatter_ = false;
  bool batch_scatter_ = false;
  ChunkLayout full_;
  uint8_t* cache_ = nullptr;
  uint8_t* execution_ = nullptr;
  std::vector<void*> registered_;
  std::vector<const uint8_t*> mapped_;
  std::vector<ExpertChunkCopyPlan> scatter_plans_, gather_plans_;
  std::vector<ChunkLayout> layouts_;
  std::vector<std::vector<ScatterCopy>> copies_;
};
}
CacheStorageFactory cuda_cache_storage(const StoreIdentity& identity, cudaStream_t stream, bool packed_fp4) {
  return [identity, stream, packed_fp4](const CacheMemory& memory, std::span<const std::span<uint8_t>> staging) {
    return std::make_unique<CudaCacheStorage>(memory, staging, identity, stream, packed_fp4);
  };
}
std::array<GlmFp4Matrix, 3> execution_matrices(const ExpertExecutionLease& lease,
    const ChunkLayout& full, Format format) {
  if (format != Format::kMxfp4 && format != Format::kNvfp4)
    throw std::invalid_argument("expert execution: unknown FP4 format");
  const auto* data = lease.data();
  const std::array<MatrixLayout, 3> layouts{full.gate, full.up, full.down};
  std::array<GlmFp4Matrix, 3> result;
  for (size_t i = 0; i < result.size(); ++i) {
    const auto& l = layouts[i];
    if (l.rows > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
        l.cols > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
        l.global_offset.has_value() != (format == Format::kNvfp4))
      throw std::invalid_argument("expert execution: invalid matrix geometry/format");
    auto& m = result[i];
    m.payload = data + l.payload_offset;
    m.scales = data + l.scales_offset;
    m.global_scale = l.global_offset ? reinterpret_cast<const float*>(data + *l.global_offset) : nullptr;
    m.rows = static_cast<int64_t>(l.rows);
    m.cols = static_cast<int64_t>(l.cols);
    m.scale_group = format == Format::kMxfp4 ? kMxfp4Group : kFp4Group;
  }
  return result;
}
}  // namespace dgpp::expert
