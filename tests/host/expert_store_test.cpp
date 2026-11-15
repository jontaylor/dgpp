#include "loaders/expert_store.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <fcntl.h>
#include <unistd.h>

#include "common/test.hpp"

namespace {
using namespace dgpp::expert;
void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
template <typename F> void refuses(F f) {
  bool rejected = false;
  try { f(); } catch (const std::exception&) { rejected = true; }
  require(rejected, "invalid image or operation must be refused");
}
struct Fixture {
  std::filesystem::path dir;
  std::string path;
  StoreIdentity identity;
  std::vector<RecordKey> keys{{0, 1, 0, false}, {0, 1, 1, false}, {40, 2, 0, true}, {40, 2, 1, true}};
  std::array<std::vector<uint8_t>, 3> codes, scales;
  std::array<MatrixSource, 3> sources;
  Fixture(Format format = Format::kMxfp4) {
    std::string pattern = "/tmp/dgpp-expert-store-XXXXXX";
    require(::mkdtemp(pattern.data()) != nullptr, "mkdtemp");
    dir = pattern;
    path = (dir / "experts.img").string();
    identity.checkpoint_sha256.fill(41);
    identity.format = format;
    identity.rank = 1;
    identity.world = 2;
    identity.hidden = 32;
    identity.intermediate = 128;
    identity.chunk_intermediate = 32;
    for (size_t i = 0; i < 3; ++i) {
      const uint64_t rows = i < 2 ? 128 : 32, cols = i < 2 ? 32 : 128;
      codes[i].resize(rows * cols / 2);
      scales[i].resize(rows * cols / (format == Format::kMxfp4 ? 32 : 16));
      for (size_t b = 0; b < codes[i].size(); ++b) codes[i][b] = static_cast<uint8_t>(b * 37 + i);
      for (size_t b = 0; b < scales[i].size(); ++b) scales[i][b] = static_cast<uint8_t>(b * 13 + i);
      sources[i] = {rows, cols, format, codes[i], scales[i], std::nullopt};
      if (format == Format::kNvfp4) sources[i].global = 1.5f + static_cast<float>(i);
    }
  }
  ~Fixture() { std::error_code ec; std::filesystem::remove_all(dir, ec); }
  PackedChunk chunk(size_t index) const { return pack_chunk(sources, 64 + keys[index].chunk * 32, 32); }
  void build() const {
    ExpertStoreBuilder writer(path, identity, keys);
    for (size_t i = keys.size(); i > 0; --i) writer.write(i - 1, chunk(i - 1));
    writer.finish();
  }
};
void corrupt(const std::string& path, uint64_t offset) {
  std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
  f.seekg(static_cast<std::streamoff>(offset));
  char c = 0;
  f.get(c);
  require(static_cast<bool>(f), "read corruption target");
  f.seekp(static_cast<std::streamoff>(offset));
  f.put(c ^ 1);
  f.flush();
  require(static_cast<bool>(f), "write corruption target");
}
}  // namespace

DGPP_TEST(expert_store_bounded_record_roundtrip_reconstructs_tp_experts_for_both_formats) {
  for (auto format : {Format::kMxfp4, Format::kNvfp4}) {
    Fixture f(format);
    f.build();
    ExpertStore store(f.path, f.identity, f.keys.size(), IoMode::kBuffered);
    uint64_t bytes = 0;
    const auto expected = pack_chunk(f.sources, 64, 64);
    std::vector<uint8_t> reconstructed(expected.bytes.size());
    for (size_t i = 0; i < store.records().size(); ++i) {
      const auto& r = store.records()[i];
      require(r.key == f.keys[i], "main/draft expert keys survive serialization");
      PackedChunk chunk{r.layout, std::vector<uint8_t>(r.layout.record_bytes)};
      store.read(i, chunk.bytes);
      bytes += chunk.bytes.size();
      require(std::all_of(chunk.bytes.begin() + r.layout.allocation_bytes, chunk.bytes.end(),
                          [](uint8_t b) { return b == 0; }), "disk padding is initialized and checksummed");
      chunk.bytes.resize(r.layout.allocation_bytes);
      require(chunk.bytes == f.chunk(i).bytes, "read record matches packed checkpoint bytes");
      if (i < 2) scatter_chunk(chunk, 32, 64, format, reconstructed, 64);
    }
    require(reconstructed == expected.bytes, "storage plus assembly is byte-exact");
    const auto stats = store.stats();
    require(stats.returned_bytes == bytes && stats.calls == f.keys.size() && stats.failures == 0,
            "actual returned file bytes and syscalls reconcile");
  }
}

DGPP_TEST(expert_store_incomplete_builder_preserves_existing_image_and_cleans_its_temporary) {
  Fixture f;
  std::ofstream(f.path) << "existing image";
  {
    ExpertStoreBuilder writer(f.path, f.identity, f.keys);
    writer.write(0, f.chunk(0));
    refuses([&] { writer.finish(); });
    refuses([&] { writer.write(0, f.chunk(0)); });
    refuses([&] { writer.write(1, f.chunk(0)); });
  }
  std::ifstream input(f.path);
  std::string original;
  std::getline(input, original);
  require(original == "existing image", "unpublished partial build cannot replace old image");
  size_t files = 0;
  for (const auto& e : std::filesystem::directory_iterator(f.dir)) { (void)e; ++files; }
  require(files == 1, "only the original image remains");
  f.build();
  ExpertStore reader(f.path, f.identity, f.keys.size(), IoMode::kBuffered);
  require(reader.records().size() == f.keys.size(), "complete image replaces atomically");
}

DGPP_TEST(expert_store_refuses_stale_identity_duplicate_keys_and_wrong_population) {
  Fixture f;
  auto keys = f.keys;
  keys[1] = keys[0];
  refuses([&] { ExpertStoreBuilder w(f.path, f.identity, keys); });
  auto identity = f.identity;
  identity.checkpoint_sha256.fill(0);
  refuses([&] { ExpertStoreBuilder w(f.path, identity, f.keys); });
  f.build();
  identity = f.identity;
  identity.checkpoint_sha256[7] ^= 1;
  refuses([&] { ExpertStore s(f.path, identity, f.keys.size(), IoMode::kBuffered); });
  identity = f.identity;
  identity.rank = 0;
  refuses([&] { ExpertStore s(f.path, identity, f.keys.size(), IoMode::kBuffered); });
  refuses([&] { ExpertStore s(f.path, f.identity, f.keys.size() + 1, IoMode::kBuffered); });
  refuses([&] { ExpertStore s(f.path, f.identity, SIZE_MAX, IoMode::kBuffered); });
}

DGPP_TEST(expert_store_corrupt_header_index_body_and_short_reads_fail_closed) {
  Fixture f;
  f.build();
  corrupt(f.path, 0);
  for (const bool verify : {false, true})
    refuses([&] { ExpertStore s(f.path, f.identity, f.keys.size(), IoMode::kBuffered, verify); });
  f.build();
  corrupt(f.path, 4096 + 32);
  refuses([&] { ExpertStore s(f.path, f.identity, f.keys.size(), IoMode::kBuffered); });
  f.build();
  ExpertStore reader(f.path, f.identity, f.keys.size(), IoMode::kBuffered);
  const auto& r = reader.records()[0];
  corrupt(f.path, r.file_offset + 9);
  std::vector<uint8_t> buffer(r.layout.record_bytes);
  refuses([&] { reader.read(0, buffer); });
  require(reader.stats().failures == 1 && reader.stats().returned_bytes == buffer.size(),
          "checksum failure still accounts for bytes actually read");
  ExpertStore fast(f.path, f.identity, f.keys.size(), IoMode::kBuffered, false);
  fast.read(0, buffer);
  require(fast.stats().failures == 0 && fast.stats().returned_bytes == buffer.size(),
          "serving reader can omit the serial payload checksum");
  require(::truncate(f.path.c_str(), static_cast<off_t>(r.file_offset + 16)) == 0, "truncate fixture");
  refuses([&] { reader.read(0, buffer); });
  require(reader.stats().failures == 2, "short body is never marked successful");
  refuses([&] { fast.read(0, buffer); });
  require(fast.stats().failures == 1, "complete reads remain mandatory without payload hashing");
}

DGPP_TEST(expert_store_direct_reads_require_aligned_budgeted_staging_without_fallback) {
  Fixture f;
  f.build();
  ExpertStore reference(f.path, f.identity, f.keys.size(), IoMode::kBuffered);
  const auto& record = reference.records()[0];
  void* allocation = nullptr;
  require(::posix_memalign(&allocation, 4096, record.layout.record_bytes) == 0, "allocate aligned staging");
  std::unique_ptr<void, decltype(&std::free)> owned(allocation, &std::free);
  std::span<uint8_t> buffer(static_cast<uint8_t*>(allocation), record.layout.record_bytes);
  const int probe = ::open(f.path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
  if (probe < 0) {
    if (errno == EINVAL || errno == EOPNOTSUPP || errno == ENOSYS)
      dgpp::test::skip("fixture filesystem does not support direct I/O");
    throw std::runtime_error("unexpected direct-I/O probe open failure");
  }
  const auto got = ::pread(probe, buffer.data(), buffer.size(), static_cast<off_t>(record.file_offset));
  const int saved_errno = errno;
  ::close(probe);
  if (got < 0 && (saved_errno == EINVAL || saved_errno == EOPNOTSUPP || saved_errno == ENOSYS))
    dgpp::test::skip("fixture filesystem rejects aligned direct reads");
  require(got == static_cast<ssize_t>(buffer.size()), "direct probe must read a complete valid record");
  ExpertStore reader(f.path, f.identity, f.keys.size(), IoMode::kDirect);
  reader.read(0, buffer);
  require(reader.io_mode() == IoMode::kDirect && reader.stats().returned_bytes == buffer.size(), "explicit direct mode");
  refuses([&] { reader.read(0, buffer.subspan(1)); });
  refuses([&] { reader.read(999, buffer); });
}

int main() { return dgpp::test::run_all(); }
