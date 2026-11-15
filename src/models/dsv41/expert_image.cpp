#include "models/dsv41/expert_image.hpp"

#include <array>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "adapters/sha256.hpp"
#include "common/log.hpp"
#include "models/dsv41/binding.hpp"
#include "models/dsv41/nvfp4_scales.hpp"

namespace dgpp {
namespace {
void integer(adapters::Sha256& hash, uint64_t value) {
  std::array<uint8_t, 8> bytes;
  for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = static_cast<uint8_t>(value >> (i * 8));
  hash.update(bytes.data(), bytes.size());
}
void text(adapters::Sha256& hash, std::string_view value) {
  integer(hash, value.size());
  hash.update(value);
}
struct SourceTriple {
  std::vector<const TensorInfo*> tensors;  // source payload/scales and any ModelOpt scalars
  std::array<expert::MatrixSource, 3> matrices;
};
SourceTriple source(const Dsv41TextConfig& cfg, const Dsv41ExpertSources& tensors, int layer, int e,
                    bool read_scalars = true) {
  SourceTriple result;
  const bool nvfp4 = cfg.experts_nvfp4 && !cfg.is_draft(layer);
  result.tensors.reserve(nvfp4 ? 12 : 6);
  const auto prefix = dsv41_layer_prefix(cfg, layer) + "ffn.experts." + std::to_string(e) + ".";
  const std::array<const char*, 3> names{"w1", "w3", "w2"};
  for (size_t m = 0; m < 3; ++m) {
    const int64_t rows = m == 2 ? cfg.hidden_size : cfg.moe_intermediate_size;
    const int64_t cols = m == 2 ? cfg.moe_intermediate_size : cfg.hidden_size;
    for (size_t scale = 0; scale < 2; ++scale) {
      const auto name = prefix + names[m] + (scale ? (nvfp4 ? ".weight_scale" : ".scale") : ".weight");
      const auto it = tensors.find(name);
      if (it == tensors.end() || !it->second || !it->second->data)
        throw std::invalid_argument("dsv41 expert image: missing source " + name);
      const auto* t = it->second;
      const auto dtype = scale ? (nvfp4 ? DType::F8_E4M3 : DType::F8_E8M0) : (nvfp4 ? DType::U8 : DType::I8);
      const int64_t source_cols = cols / (scale ? (nvfp4 ? 16 : 32) : 2);
      if (t->name != name || t->dtype != dtype || t->shape != std::vector<int64_t>{rows, source_cols} ||
          t->nbytes() != t->numel())
        throw std::invalid_argument("dsv41 expert image: source geometry/format mismatch " + name);
      result.tensors.push_back(t);
    }
    const auto* codes = result.tensors[result.tensors.size() - 2];
    const auto* scales = result.tensors.back();
    std::optional<float> global;
    if (nvfp4) {
      std::array<float, 2> values;
      for (size_t s = 0; s < values.size(); ++s) {
        const auto name = prefix + names[m] + (s ? ".input_scale" : ".weight_scale_2");
        const auto it = tensors.find(name);
        if (it == tensors.end() || !it->second || !it->second->data || it->second->name != name ||
            it->second->dtype != DType::F32 || !it->second->shape.empty() || it->second->nbytes() != 4)
          throw std::invalid_argument("dsv41 expert image: missing or malformed NVFP4 scalar " + name);
        result.tensors.push_back(it->second);
        if (read_scalars) std::memcpy(&values[s], it->second->data, 4);
      }
      if (read_scalars) global = dsv41_nvfp4_inverse_global(values[0], values[1]);
    }
    result.matrices[m] = {static_cast<uint64_t>(rows), static_cast<uint64_t>(cols),
        nvfp4 ? expert::Format::kNvfp4 : expert::Format::kMxfp4,
        {static_cast<const uint8_t*>(codes->data), codes->nbytes()},
        {static_cast<const uint8_t*>(scales->data), scales->nbytes()}, global};
  }
  return result;
}
void prefetch(const SourceTriple& s) {
  // Bound read-ahead to one expert. Especially on NFS, faulting every source
  // page separately needlessly serializes verification of a reusable image.
  // This is only an I/O hint: every byte is still hashed and validated.
  for (const auto* t : s.tensors) if (t->owner) t->owner->prefetch(*t);
}
void discard(const SourceTriple& s) {
  for (const auto* t : s.tensors) if (t->owner) t->owner->discard(*t);
}
void stat_fields(adapters::Sha256& hash, const struct stat& s) {
  for (const auto value : {static_cast<uint64_t>(s.st_dev), static_cast<uint64_t>(s.st_ino),
       static_cast<uint64_t>(s.st_size), static_cast<uint64_t>(s.st_mtim.tv_sec),
       static_cast<uint64_t>(s.st_mtim.tv_nsec), static_cast<uint64_t>(s.st_ctim.tv_sec),
       static_cast<uint64_t>(s.st_ctim.tv_nsec)}) integer(hash, value);
}
std::string image_stamp(const std::string& path) {
  struct stat s{};
  if (::stat(path.c_str(), &s) || !S_ISREG(s.st_mode))
    throw std::runtime_error("dsv41 expert image: cannot stat image " + path);
  adapters::Sha256 hash;
  stat_fields(hash, s);
  return hash.finish();
}
std::optional<std::string> source_stamp(const Dsv41TextConfig& cfg, std::string_view config,
    const Dsv41ExpertSources& tensors, int layers) {
  adapters::Sha256 hash;
  text(hash, "dgpp-dsv41-mapped-source-stamp-v1");
  text(hash, config);
  integer(hash, layers);
  std::unordered_map<const SafetensorsFile*, std::string> shards;
  for (int l = 0; l < layers; ++l) {
    const int experts = cfg.is_draft(l) ? cfg.dspark_n_routed_experts : cfg.n_routed_experts;
    integer(hash, experts);
    for (int e = 0; e < experts; ++e) {
      // Inspect descriptors only. In particular, do not fault NVFP4 scalars in
      // when reusing an image whose scalar values were validated at creation.
      for (const auto* t : source(cfg, tensors, l, e, false).tensors) {
        if (!t->owner) return std::nullopt;  // borrowed/in-memory sources must be hashed
        if (t->data_end > t->owner->map_size() ||
            t->data != t->owner->map_base() + t->data_begin)
          throw std::invalid_argument("dsv41 expert image: source is not its owner's mapped view");
        auto it = shards.find(t->owner);
        if (it == shards.end()) {
          const auto s = t->owner->file_stat();
          if (!S_ISREG(s.st_mode) || static_cast<uint64_t>(s.st_size) != t->owner->map_size())
            throw std::invalid_argument("dsv41 expert image: mapped source size changed");
          adapters::Sha256 shard;
          text(shard, t->owner->path());
          stat_fields(shard, s);
          integer(shard, t->owner->header_fold());
          it = shards.emplace(t->owner, shard.finish()).first;
        }
        text(hash, it->second);
        text(hash, t->name);
        text(hash, dtype_name(t->dtype));
        integer(hash, t->shape.size());
        for (auto d : t->shape) integer(hash, static_cast<uint64_t>(d));
        integer(hash, t->data_begin);
        integer(hash, t->data_end);
      }
    }
  }
  return hash.finish();
}
std::array<uint8_t, 32> digest_bytes(std::string_view digest) {
  if (digest.size() != 64 || digest.find_first_not_of("0123456789abcdef") != std::string_view::npos)
    throw std::invalid_argument("dsv41 expert image: malformed source digest");
  auto nibble = [](char c) -> uint8_t { return c <= '9' ? c - '0' : c - 'a' + 10; };
  std::array<uint8_t, 32> result{};
  for (size_t i = 0; i < result.size(); ++i)
    result[i] = static_cast<uint8_t>((nibble(digest[i * 2]) << 4) | nibble(digest[i * 2 + 1]));
  return result;
}
std::optional<std::string> cached_digest(const std::string& path, const std::string& stamp) {
  std::ifstream input(path + ".source-stamp", std::ios::binary);
  std::array<char, 512> bytes{};
  input.read(bytes.data(), bytes.size());
  if (!input.eof() || input.gcount() == 0) return std::nullopt;
  std::istringstream lines(std::string(bytes.data(), static_cast<size_t>(input.gcount())));
  std::string version, saved_stamp, image, digest, checksum, extra;
  if (!std::getline(lines, version) || !std::getline(lines, saved_stamp) || !std::getline(lines, image) ||
      !std::getline(lines, digest) || !std::getline(lines, checksum) || std::getline(lines, extra) ||
      version != "dgpp-expert-source-stamp-v1" || saved_stamp != stamp || digest.size() != 64 ||
      digest.find_first_not_of("0123456789abcdef") != std::string::npos) return std::nullopt;
  const auto body = version + "\n" + saved_stamp + "\n" + image + "\n" + digest + "\n";
  if (adapters::sha256(body) != checksum || image != image_stamp(path)) return std::nullopt;
  (void)digest_bytes(digest);
  return digest;
}
void save_digest(const std::string& path, const std::string& stamp, const std::string& digest) {
  // This is disposable local cache metadata. Failure to save it must not
  // prevent using an otherwise valid image (for example in a read-only cache).
  std::string temporary;
  int fd = -1;
  try {
    const auto body = std::string("dgpp-expert-source-stamp-v1\n") + stamp + "\n" + image_stamp(path) + "\n" + digest + "\n";
    const auto data = body + adapters::sha256(body) + "\n";
    temporary = path + ".source-stamp.tmp-XXXXXX";
    fd = ::mkstemp(temporary.data());
    if (fd < 0) return;
    size_t at = 0;
    while (at < data.size()) {
      const auto n = ::write(fd, data.data() + at, data.size() - at);
      if (n < 0 && errno == EINTR) continue;
      if (n <= 0) throw std::runtime_error("cannot write expert source stamp");
      at += static_cast<size_t>(n);
    }
    if (::fsync(fd)) throw std::runtime_error("cannot sync expert source stamp");
    ::close(fd); fd = -1;
    if (::rename(temporary.c_str(), (path + ".source-stamp").c_str()))
      throw std::runtime_error("cannot publish expert source stamp");
  } catch (const std::exception&) {
    if (fd >= 0) ::close(fd);
    if (!temporary.empty()) ::unlink(temporary.c_str());
  }
}
}
std::shared_ptr<expert::ExpertStore> dsv41_prepare_expert_image(
    const std::string& path, const Dsv41TextConfig& cfg, std::string_view config_json,
    const Dsv41ExpertSources& tensors, int rank, int world, uint64_t chunk_intermediate,
    bool include_draft, expert::IoMode mode) {
  using namespace expert;
  if (path.empty() || config_json.empty() || cfg.weights_exl3 || cfg.fp4_block_size != 32 ||
      cfg.num_hidden_layers <= 0 || cfg.num_nextn_predict_layers < 0 || cfg.n_routed_experts <= 0 ||
      (include_draft && cfg.num_nextn_predict_layers && (cfg.dspark_n_routed_experts <= 0 || cfg.experts_nvfp4)) ||
      rank < 0 || world <= 0 || cfg.num_nextn_predict_layers > INT32_MAX - cfg.num_hidden_layers)
    throw std::invalid_argument("dsv41 expert image: unsupported source/configuration (mixed NVFP4/MXFP4 draft paging is not implemented)");
  const auto format = cfg.experts_nvfp4 ? Format::kNvfp4 : Format::kMxfp4;
  const auto chunks = partition(cfg.hidden_size, cfg.moe_intermediate_size, world, rank,
                                 chunk_intermediate, format);
  StoreIdentity identity;
  identity.format = format;
  identity.rank = static_cast<uint32_t>(rank);
  identity.world = static_cast<uint32_t>(world);
  identity.hidden = cfg.hidden_size;
  identity.intermediate = cfg.moe_intermediate_size;
  identity.chunk_intermediate = chunk_intermediate;
  const int layers = include_draft ? cfg.max_layer() : cfg.num_hidden_layers;
  std::vector<RecordKey> keys;
  for (int l = 0; l < layers; ++l) {
    const int experts = cfg.is_draft(l) ? cfg.dspark_n_routed_experts : cfg.n_routed_experts;
    for (int e = 0; e < experts; ++e)
      for (size_t c = 0; c < chunks.size(); ++c) {
        if (c > UINT32_MAX) throw std::overflow_error("dsv41 expert image: chunk key overflow");
        keys.push_back({static_cast<uint32_t>(l), static_cast<uint32_t>(e), static_cast<uint32_t>(c), cfg.is_draft(l)});
      }
  }
  const char* verify = std::getenv("DGPP_EXPERT_VERIFY_RECORDS");
  const bool verify_records = verify && verify[0] == '1';
  auto open_store = [&] {
    auto store = std::make_shared<ExpertStore>(path, identity, keys.size(), mode, verify_records);
    for (size_t r = 0; r < keys.size(); ++r)
      if (store->records()[r].key != keys[r])
        throw std::invalid_argument("dsv41 expert image: image record population differs from model");
    DGPP_LOG_INFO("dsv41 expert image: per-read payload checksums {} (DGPP_EXPERT_VERIFY_RECORDS); image identity and index checked",
                  verify_records ? "on" : "off");
    return store;
  };
  const auto stamp = source_stamp(cfg, config_json, tensors, layers);
  if (stamp && std::filesystem::exists(path)) {
    if (const auto saved = cached_digest(path, *stamp)) {
      identity.checkpoint_sha256 = digest_bytes(*saved);
      return open_store();
    }
  }
  adapters::Sha256 hash;
  text(hash, "dgpp-dsv41-expert-source-set-v1");
  text(hash, config_json);
  integer(hash, layers);
  // Reads no more than one borrowed source triple at once. Advise its mapped
  // pages away after hashing so preparation cannot pin the complete source.
  for (int l = 0; l < layers; ++l) {
    const int experts = cfg.is_draft(l) ? cfg.dspark_n_routed_experts : cfg.n_routed_experts;
    integer(hash, static_cast<uint64_t>(l));
    integer(hash, static_cast<uint64_t>(experts));
    for (int e = 0; e < experts; ++e) {
      const auto src = source(cfg, tensors, l, e);
      prefetch(src);
      for (const auto* tensor : src.tensors) {
        text(hash, tensor->name);
        text(hash, dtype_name(tensor->dtype));
        integer(hash, tensor->shape.size());
        for (const auto d : tensor->shape) integer(hash, static_cast<uint64_t>(d));
        integer(hash, tensor->nbytes());
        constexpr size_t block = 1024 * 1024;
        const auto* bytes = static_cast<const uint8_t*>(tensor->data);
        for (size_t at = 0; at < tensor->nbytes();) {
          const auto count = std::min(block, tensor->nbytes() - at);
          hash.update(bytes + at, count);
          at += count;
        }
      }
      discard(src);
    }
  }
  const auto digest = hash.finish();
  identity.checkpoint_sha256 = digest_bytes(digest);
  if (!std::filesystem::exists(path)) {
    ExpertStoreBuilder builder(path, identity, keys);
    size_t record = 0;
    for (int l = 0; l < layers; ++l) {
      const int experts = cfg.is_draft(l) ? cfg.dspark_n_routed_experts : cfg.n_routed_experts;
      for (int e = 0; e < experts; ++e) {
        const auto src = source(cfg, tensors, l, e);
        prefetch(src);
        for (const auto& chunk : chunks)
          builder.write(record++, pack_chunk(src.matrices, chunk.intermediate_begin, chunk.intermediate_size));
        discard(src);
      }
    }
    builder.finish();
  }
  auto store = open_store();
  if (stamp) {
    if (source_stamp(cfg, config_json, tensors, layers) != stamp)
      throw std::runtime_error("dsv41 expert image: source changed during preparation");
    save_digest(path, *stamp, digest);
  }
  return store;
}
}  // namespace dgpp
