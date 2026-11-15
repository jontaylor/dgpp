#include "loaders/expert_cache_cuda.hpp"

#include <array>
#include <cstdlib>
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
                   StoreIdentity identity, cudaStream_t stream, bool packed_nvfp4)
      : memory_(memory), identity_(identity), stream_(stream), packed_nvfp4_(packed_nvfp4) {
    full_ = partition(identity.hidden, identity.intermediate, identity.world, identity.rank,
                      identity.intermediate / identity.world, identity.format).front();
    if (packed_nvfp4_ && (identity.format != Format::kNvfp4 ||
        memory_.execution_slot_bytes != 2 * full_.allocation_bytes ||
        moe_fp4_packed_bytes(identity.hidden, identity.intermediate / identity.world) > full_.allocation_bytes))
      throw std::invalid_argument("expert CUDA cache: invalid packed NVFP4 reservation");
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
    registered_.reserve(staging.size());
    mapped_.reserve(staging.size());
    try {
      if (memory_.cache_slots)
        DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&cache_), memory_.cache_slots * memory_.cache_slot_bytes));
      DGPP_CUDA_OK(cudaMalloc(reinterpret_cast<void**>(&execution_), memory_.execution_slots * memory_.execution_slot_bytes));
      for (const auto span : staging) {
        DGPP_CUDA_OK(cudaHostRegister(span.data(), span.size(), cudaHostRegisterDefault));
        registered_.push_back(span.data());
        if (gpu_scatter_) {
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
    DGPP_CUDA_OK(cudaMemcpyAsync(cache_ + slot * memory_.cache_slot_bytes, bytes.data(), bytes.size(),
                                 cudaMemcpyHostToDevice, stream_));
  }
  void begin_execution(size_t slot) override {
    check_execution(slot);
    DGPP_CUDA_OK(cudaMemsetAsync(execution_ + slot * memory_.execution_slot_bytes, 0,
                                 memory_.execution_slot_bytes, stream_));
  }
  void assemble(size_t execution_slot, const StoreRecord& record,
      std::optional<size_t> cache_slot, std::span<const uint8_t> host_bytes) override {
    check_execution(execution_slot);
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
      launch_expert_chunk_scatter(destination, source, scatter_plans_[record.key.chunk], stream_);
      return;
    }
    for (const auto& c : copies_[record.key.chunk])
      DGPP_CUDA_OK(cudaMemcpy2DAsync(destination + c.destination_offset, c.destination_pitch,
                                     source + c.source_offset, c.source_pitch, c.row_bytes, c.rows, kind, stream_));
  }
  const uint8_t* execution_data(size_t slot) const override {
    check_execution(slot);
    return execution_ + slot * memory_.execution_slot_bytes;
  }
  const uint8_t* packed_execution_data(size_t slot) override {
    check_execution(slot);
    if (!packed_nvfp4_) throw std::logic_error("expert CUDA cache: packed layout disabled");
    auto* source = execution_ + slot * memory_.execution_slot_bytes;
    auto* destination = source + full_.allocation_bytes;
    pack(destination, source);
    return destination;
  }
  const uint8_t* cache_data(size_t slot) const override {
    check_cache(slot);
    return cache_ + slot * memory_.cache_slot_bytes;
  }
  void move_cache_slot(size_t dest, size_t source) override {
    check_cache(dest);
    check_cache(source);
    DGPP_CUDA_OK(cudaMemcpyAsync(cache_ + dest * memory_.cache_slot_bytes,
        cache_ + source * memory_.cache_slot_bytes, memory_.cache_slot_bytes,
        cudaMemcpyDeviceToDevice, stream_));
  }
  void native_to_execution(size_t execution_slot, size_t first) override {
    check_native(execution_slot, first);
    if (packed_nvfp4_) {
      auto* destination = execution_ + execution_slot * memory_.execution_slot_bytes;
      auto* source = cache_ + first * memory_.cache_slot_bytes;
      Fp4ExpertDestination out;
      const std::array<MatrixLayout, 3> matrices{full_.gate, full_.up, full_.down};
      for (size_t m = 0; m < matrices.size(); ++m) {
        out.payload[m] = destination + matrices[m].payload_offset;
        out.scales[m] = destination + matrices[m].scales_offset;
        out.globals[m] = reinterpret_cast<float*>(destination + *matrices[m].global_offset);
      }
      launch_moe_fp4_unpack(out, moe_fp4_packed_view(source, identity_.hidden,
          identity_.intermediate / identity_.world), identity_.hidden,
          identity_.intermediate / identity_.world, stream_);
      return;
    }
    DGPP_CUDA_OK(cudaMemcpyAsync(execution_ + execution_slot * memory_.execution_slot_bytes,
        cache_ + first * memory_.cache_slot_bytes, memory_.execution_slot_bytes,
        cudaMemcpyDeviceToDevice, stream_));
  }
  void execution_to_native(size_t execution_slot, size_t first) override {
    check_native(execution_slot, first);
    if (packed_nvfp4_) {
      pack(cache_ + first * memory_.cache_slot_bytes,
           execution_ + execution_slot * memory_.execution_slot_bytes);
      return;
    }
    DGPP_CUDA_OK(cudaMemcpyAsync(cache_ + first * memory_.cache_slot_bytes,
        execution_ + execution_slot * memory_.execution_slot_bytes, memory_.execution_slot_bytes,
        cudaMemcpyDeviceToDevice, stream_));
  }
  void unpack_native_chunk(size_t execution_slot, const StoreRecord& record, size_t cache_slot) override {
    check_execution(execution_slot);
    check_cache(cache_slot);
    if (record.key.chunk >= copies_.size()) throw std::invalid_argument("expert CUDA cache: native chunk index");
    auto* dest = cache_ + cache_slot * memory_.cache_slot_bytes;
    const auto* source = execution_ + execution_slot * memory_.execution_slot_bytes;
    DGPP_CUDA_OK(cudaMemsetAsync(dest, 0, memory_.cache_slot_bytes, stream_));
    if (gpu_scatter_) {
      launch_expert_chunk_scatter(dest, source, gather_plans_[record.key.chunk], stream_);
      return;
    }
    for (const auto& c : copies_[record.key.chunk])
      DGPP_CUDA_OK(cudaMemcpy2DAsync(dest + c.source_offset, c.source_pitch,
          source + c.destination_offset, c.destination_pitch, c.row_bytes, c.rows,
          cudaMemcpyDeviceToDevice, stream_));
  }
  // require_execution_context binds kernels to stream_. Every cache write,
  // relocation and unpack is submitted to that same stream, so none can
  // overwrite weights before earlier kernel readers complete. Host leases
  // need not force GPU completion; staging and destruction still drain.
  void retire_native() override {}
  void finish() override { DGPP_CUDA_OK(cudaStreamSynchronize(stream_)); }
  bool accepts_execution_context(const void* context) const override { return context == stream_; }
 private:
  size_t staging_index(const uint8_t* source) const {
    size_t index = 0;
    while (index < registered_.size() && registered_[index] != source) ++index;
    if (index == registered_.size()) throw std::invalid_argument("expert CUDA cache: unregistered staging");
    return index;
  }
  void pack(uint8_t* destination, const uint8_t* source) {
    Fp4ExpertSource in;
    const std::array<MatrixLayout, 3> matrices{full_.gate, full_.up, full_.down};
    for (size_t m = 0; m < matrices.size(); ++m) {
      in.payload[m] = source + matrices[m].payload_offset;
      in.scales[m] = source + matrices[m].scales_offset;
      in.globals[m] = reinterpret_cast<const float*>(source + *matrices[m].global_offset);
    }
    launch_moe_fp4_pack(moe_fp4_packed_view(destination, identity_.hidden,
        identity_.intermediate / identity_.world), in, identity_.hidden,
        identity_.intermediate / identity_.world, stream_);
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
    (void)cudaStreamSynchronize(stream_);
    for (auto* ptr : registered_) (void)cudaHostUnregister(ptr);
    registered_.clear();
    if (execution_) (void)cudaFree(std::exchange(execution_, nullptr));
    if (cache_) (void)cudaFree(std::exchange(cache_, nullptr));
  }
  CacheMemory memory_;
  StoreIdentity identity_;
  cudaStream_t stream_;
  bool packed_nvfp4_ = false;
  bool gpu_scatter_ = false;
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
CacheStorageFactory cuda_cache_storage(const StoreIdentity& identity, cudaStream_t stream, bool packed_nvfp4) {
  return [identity, stream, packed_nvfp4](const CacheMemory& memory, std::span<const std::span<uint8_t>> staging) {
    return std::make_unique<CudaCacheStorage>(memory, staging, identity, stream, packed_nvfp4);
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
