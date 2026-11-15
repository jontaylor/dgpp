#include "models/dsv41/expert_image.hpp"

#include <array>
#include <filesystem>
#include <cmath>
#include <cstring>
#include <limits>
#include <fstream>
#include <sstream>

#include "models/dsv41/nvfp4_scales.hpp"
#include <stdexcept>
#include <string>
#include <vector>

#include "common/test.hpp"
#include "models/dsv41/binding.hpp"

namespace {
using namespace dgpp;
using namespace dgpp::expert;
void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
template <typename F> void refuses(F f) {
  bool rejected = false;
  try { f(); } catch (const std::exception&) { rejected = true; }
  require(rejected, "invalid/stale expert source or image must be refused");
}
struct Fixture {
  Dsv41TextConfig cfg;
  std::filesystem::path dir;
  std::string path;
  std::vector<TensorInfo> info;
  std::vector<std::vector<uint8_t>> bytes;
  Dsv41ExpertSources tensors;
  Fixture(bool nvfp4 = false) {
    cfg.experts_nvfp4 = nvfp4;
    cfg.hidden_size = 32;
    cfg.moe_intermediate_size = 128;
    cfg.num_hidden_layers = 1;
    cfg.num_nextn_predict_layers = 1;
    cfg.n_routed_experts = 2;
    cfg.dspark_n_routed_experts = 1;
    std::string pattern = "/tmp/dgpp-dsv41-expert-image-XXXXXX";
    require(::mkdtemp(pattern.data()) != nullptr, "mkdtemp");
    dir = pattern;
    path = (dir / "experts.img").string();
    info.reserve(36);
    bytes.reserve(36);
    for (int l = 0; l < 2; ++l) {
      const int experts = l ? 1 : 2;
      for (int e = 0; e < experts; ++e)
        for (const auto* matrix : {"w1", "w3", "w2"})
          for (int scale = 0; scale < ((nvfp4 && l == 0) ? 4 : 2); ++scale) {
            const bool nv = nvfp4 && l == 0;
            const bool down = std::string(matrix) == "w2";
            const int64_t rows = down ? 32 : 128, cols = down ? 128 : 32;
            TensorInfo t;
            t.name = dsv41_layer_prefix(cfg, l) + "ffn.experts." + std::to_string(e) + "." + matrix +
                     (scale == 0 ? ".weight" : scale == 1 ? (nv ? ".weight_scale" : ".scale") :
                      scale == 2 ? ".weight_scale_2" : ".input_scale");
            t.dtype = scale >= 2 ? DType::F32 : scale == 1 ? (nv ? DType::F8_E4M3 : DType::F8_E8M0) : (nv ? DType::U8 : DType::I8);
            if (scale < 2) t.shape = {rows, cols / (scale ? (nv ? 16 : 32) : 2)};
            t.data_end = t.numel() * dtype_size(t.dtype);
            bytes.emplace_back(t.nbytes());
            for (size_t b = 0; b < t.nbytes(); ++b)
              bytes.back()[b] = static_cast<uint8_t>(b * 37 + info.size() * 23);
            if (scale >= 2) {
              const float value = scale == 2 ? 0x1p-13f : 1.25f;
              std::memcpy(bytes.back().data(), &value, 4);
            }
            t.data = bytes.back().data();
            info.push_back(std::move(t));
            tensors.emplace(info.back().name, &info.back());
          }
    }
  }
  ~Fixture() { std::error_code ec; std::filesystem::remove_all(dir, ec); }
  std::shared_ptr<ExpertStore> prepare(bool draft = true, int rank = 1, std::string_view config = "fixture config bytes v1") {
    return dsv41_prepare_expert_image(path, cfg, config, tensors, rank, 2, 32, draft, IoMode::kBuffered);
  }
};
}
DGPP_TEST(dsv41_expert_image_builds_exact_tp_records_and_reuses_verified_image) {
  Fixture f;
  auto store = f.prepare();
  require(store->records().size() == 6 && store->records()[4].key.draft, "main and DSpark TP chunks");
  for (size_t record = 0; record < store->records().size(); ++record) {
    const auto& r = store->records()[record];
    const auto prefix = dsv41_layer_prefix(f.cfg, r.key.layer) + "ffn.experts." + std::to_string(r.key.expert) + ".";
    std::array<MatrixSource, 3> src;
    const std::array<const char*, 3> names{"w1", "w3", "w2"};
    for (size_t m = 0; m < 3; ++m) {
      const auto* w = f.tensors.at(prefix + names[m] + ".weight");
      const auto* s = f.tensors.at(prefix + names[m] + ".scale");
      src[m] = {static_cast<uint64_t>(w->shape[0]), static_cast<uint64_t>(w->shape[1] * 2), Format::kMxfp4,
                {static_cast<const uint8_t*>(w->data), w->nbytes()},
                {static_cast<const uint8_t*>(s->data), s->nbytes()}, std::nullopt};
    }
    const auto expected = pack_chunk(src, 64 + r.key.chunk * 32, 32);
    std::vector<uint8_t> read(r.layout.record_bytes);
    store->read(record, read);
    require(std::equal(expected.bytes.begin(), expected.bytes.end(), read.begin()), "source payloads/scales and strided down slices preserved");
  }
  const auto before = std::filesystem::last_write_time(f.path);
  auto again = f.prepare();
  require(again->identity() == store->identity() && std::filesystem::last_write_time(f.path) == before,
          "verified image reused without rewriting");
  require(store->identity().checkpoint_sha256 != std::array<uint8_t, 32>{}, "real source SHA256 populated");
}
DGPP_TEST(dsv41_expert_image_source_digest_rejects_changed_weights_config_and_population) {
  Fixture f;
  (void)f.prepare();
  f.bytes[0][31] ^= 1;
  refuses([&] { (void)f.prepare(); });
  f.bytes[0][31] ^= 1;
  refuses([&] { (void)f.prepare(true, 1, "different config"); });
  refuses([&] { (void)f.prepare(false); });
  refuses([&] { (void)f.prepare(true, 0); });
  (void)f.prepare();  // stale attempts preserved original image
}
DGPP_TEST(dsv41_expert_image_reuses_mapped_sources_without_reading_payload_and_invalidates_edits) {
  for (bool nvfp4 : {false, true}) {
    Fixture f(nvfp4);
    const auto shard_path = (f.dir / "source.safetensors").string();
    std::ostringstream header;
    header << '{';
    uint64_t offset = 0;
    for (size_t i = 0; i < f.info.size(); ++i) {
      const auto& t = f.info[i];
      if (i) header << ',';
      header << '"' << t.name << "\":{\"dtype\":\"" << dtype_name(t.dtype) << "\",\"shape\":[";
      for (size_t d = 0; d < t.shape.size(); ++d) {
        if (d) header << ',';
        header << t.shape[d];
      }
      header << "],\"data_offsets\":[" << offset << ',' << offset + t.nbytes() << "]}";
      offset += t.nbytes();
    }
    header << '}';
    const auto json = header.str();
    const uint64_t length = json.size();
    {
      std::ofstream file(shard_path, std::ios::binary);
      file.write(reinterpret_cast<const char*>(&length), sizeof(length));
      file << json;
      for (const auto& bytes : f.bytes)
        file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
      require(bool(file), "write mapped fixture");
    }
    auto shard = SafetensorsFile::open(shard_path);
    f.tensors.clear();
    for (const auto& t : f.info) f.tensors.emplace(t.name, &shard->at(t.name));
    const auto store = f.prepare(false);
    require(std::filesystem::exists(f.path + ".source-stamp"), "mapped source receipt created");
    // A cache hit must work even when every source payload page is unreadable.
    // This also covers NVFP4 scalar bytes, which are validated on the first pass.
    auto* mapping = const_cast<uint8_t*>(shard->map_base());
    require(::mprotect(mapping, shard->map_size(), PROT_NONE) == 0, "protect source payload");
    try {
      auto again = f.prepare(false);
      require(again->identity() == store->identity(), "cached source digest reused");
    } catch (...) {
      ::mprotect(mapping, shard->map_size(), PROT_READ);
      throw;
    }
    require(::mprotect(mapping, shard->map_size(), PROT_READ) == 0, "restore source payload");
    // An in-place edit is caught even when size and mtime are preserved.
    const auto previous = shard->file_stat();
    const int fd = ::open(shard_path.c_str(), O_RDWR | O_CLOEXEC);
    require(fd >= 0, "open source for edit");
    const uint8_t changed = f.bytes[0][31] ^ 1;
    const auto position = static_cast<off_t>(shard->at(f.info[0].name).data_begin + 31);
    require(::pwrite(fd, &changed, 1, position) == 1, "edit source weight");
    const std::array<timespec, 2> times{previous.st_atim, previous.st_mtim};
    require(::futimens(fd, times.data()) == 0, "restore original source mtime");
    ::close(fd);
    refuses([&] { (void)f.prepare(false); });
    refuses([&] { (void)f.prepare(false, 1, "changed config"); });
  }
}
DGPP_TEST(dsv41_expert_image_validates_full_source_before_publication) {
  Fixture f;
  f.info[0].dtype = DType::U8;
  refuses([&] { (void)f.prepare(); });
  require(!std::filesystem::exists(f.path), "wrong dtype refused before creating image");
  f.info[0].dtype = DType::I8;
  f.info[1].shape[1] += 1;
  refuses([&] { (void)f.prepare(); });
  f.info[1].shape[1] -= 1;
  f.cfg.weights_exl3 = true;
  refuses([&] { (void)f.prepare(); });
  f.cfg.weights_exl3 = false;
  f.tensors.erase(f.info.back().name);
  refuses([&] { (void)f.prepare(); });
  require(!std::filesystem::exists(f.path), "missing late draft tensor cannot publish partial main image");
}
DGPP_TEST(dsv41_nvfp4_scale_convention_preserves_exact_source_factors) {
  for (int e = -126; e <= 126; ++e) {
    const float source = std::ldexp(1.0f, e);
    const float inverse = dsv41_nvfp4_inverse_global(source, 1.25f);
    require(inverse == std::ldexp(1.0f, -e) && source * inverse == 1.0f,
            "source multiplication factor becomes an exact descriptor divisor");
  }
  for (float bad : {0.0f, -1.0f, 1.5f, std::numeric_limits<float>::infinity(),
                   std::numeric_limits<float>::quiet_NaN(), 0x1p-127f, 0x1p127f})
    refuses([&] { (void)dsv41_nvfp4_inverse_global(bad, 1.0f); });
  for (float bad : {0.0f, -1.0f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()})
    refuses([&] { (void)dsv41_nvfp4_inverse_global(1.0f, bad); });
}

DGPP_TEST(dsv41_nvfp4_binding_distinguishes_main_experts_from_mxfp4_drafts) {
  Fixture fixture(true);
  for (int layer = 0; layer < 2; ++layer) {
    size_t matched = 0;
    const auto expected = dsv41_expected_layer_tensors(fixture.cfg, layer);
    for (const auto& tensor : expected) {
      if (tensor.cls != Dsv41WeightClass::RoutedExpert) continue;
      const auto found = fixture.tensors.find(tensor.name);
      require(found != fixture.tensors.end(), "binding emitted a name absent from independent checkpoint fixture");
      require(tensor.dtype == found->second->dtype, "binding dtype differs from native checkpoint");
      require(tensor.shape == found->second->shape, "binding shape differs from native checkpoint");
      require(tensor.nbytes() == found->second->nbytes(), "binding native byte count differs");
      ++matched;
    }
    require(matched == static_cast<size_t>(layer == 0 ? 24 : 6), "main/draft tensor population differs");
  }
  auto invalid = fixture.cfg;
  invalid.hidden_size = 30;
  refuses([&] { (void)dsv41_expected_layer_tensors(invalid, 0); });
}

DGPP_TEST(dsv41_nvfp4_expert_image_preserves_codes_scales_and_exact_global_factor) {
  Fixture f(true);
  auto store = f.prepare(false);
  require(store->identity().format == Format::kNvfp4 && store->records().size() == 4,
          "NVFP4 main experts use the native common image format");
  for (size_t i = 0; i < store->records().size(); ++i) {
    const auto& r = store->records()[i];
    const auto prefix = dsv41_layer_prefix(f.cfg, r.key.layer) + "ffn.experts." + std::to_string(r.key.expert) + ".";
    std::array<MatrixSource, 3> src;
    size_t m = 0;
    for (const char* projection : {"w1", "w3", "w2"}) {
      const auto* w = f.tensors.at(prefix + projection + ".weight");
      const auto* s = f.tensors.at(prefix + projection + ".weight_scale");
      src[m++] = {static_cast<uint64_t>(w->shape[0]), static_cast<uint64_t>(w->shape[1] * 2), Format::kNvfp4,
          {static_cast<const uint8_t*>(w->data), w->nbytes()}, {static_cast<const uint8_t*>(s->data), s->nbytes()}, 8192.0f};
    }
    const auto expected = pack_chunk(src, 64 + r.key.chunk * 32, 32);
    std::vector<uint8_t> data(r.layout.record_bytes);
    store->read(i, data);
    require(std::equal(expected.bytes.begin(), expected.bytes.end(), data.begin()),
            "native NVFP4 codes, E4M3 scales, down slices and canonical globals preserved");
  }
  const auto identity = store->identity();
  require(f.prepare(false)->identity() == identity, "verified NVFP4 image reused");
  // Calibrated activation metadata is not used by the BF16 activation path,
  // but remains part of source identity and must be validated and hashed.
  const auto* input = f.tensors.at("layers.0.ffn.experts.0.w1.input_scale");
  const float changed = 1.5f;
  std::memcpy(const_cast<void*>(input->data), &changed, 4);
  refuses([&] { (void)f.prepare(false); });
}

DGPP_TEST(dsv41_nvfp4_expert_image_refuses_bad_scales_and_mixed_draft_before_publication) {
  Fixture f(true);
  refuses([&] { (void)f.prepare(true); });
  require(!std::filesystem::exists(f.path), "mixed-format draft paging refused before publication");
  auto* t = f.tensors.at("layers.0.ffn.experts.0.w1.weight_scale_2");
  const float invalid = 1.5f;
  std::memcpy(const_cast<void*>(t->data), &invalid, 4);
  refuses([&] { (void)f.prepare(false); });
  require(!std::filesystem::exists(f.path), "inexact source-global translation cannot publish an image");
}
int main() { return dgpp::test::run_all(); }
