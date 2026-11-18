#include "loaders/expert_store.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <limits>
#include <set>
#include <stdexcept>
#include <sys/stat.h>
#include <tuple>
#include <unistd.h>

namespace dgpp::expert {
namespace {
constexpr uint64_t kPage = 4096, kEntry = 64, kSeed = 14695981039346656037ull;
constexpr std::array<uint8_t, 8> kMagic{'D', 'G', 'P', 'P', 'X', 'P', 'R', 'T'};
constexpr uint32_t kVersion = 1;
std::atomic<uint64_t> temporary_id{0};
[[noreturn]] void error(const std::string& what) { throw std::runtime_error("expert store: " + what); }
uint64_t add(uint64_t a, uint64_t b) {
  if (b > static_cast<uint64_t>(std::numeric_limits<off_t>::max()) ||
      a > static_cast<uint64_t>(std::numeric_limits<off_t>::max()) - b) error("file extent overflow");
  return a + b;
}
uint64_t aligned(uint64_t n) { return n % kPage ? add(n, kPage - n % kPage) : n; }
uint64_t hash(std::span<const uint8_t> data, uint64_t state = kSeed) {
  for (auto b : data) state = (state ^ b) * 1099511628211ull;
  return state;
}
void put(std::span<uint8_t> out, size_t at, uint64_t value, size_t bytes = 8) {
  for (size_t i = 0; i < bytes; ++i) out[at + i] = static_cast<uint8_t>(value >> (i * 8));
}
uint64_t get(std::span<const uint8_t> in, size_t at, size_t bytes = 8) {
  uint64_t value = 0;
  for (size_t i = 0; i < bytes; ++i) value |= static_cast<uint64_t>(in[at + i]) << (i * 8);
  return value;
}
void write_all(int fd, std::span<const uint8_t> data, uint64_t offset) {
  while (!data.empty()) {
    const size_t count = std::min<size_t>(data.size(), 1ull << 30);
    const auto n = ::pwrite(fd, data.data(), count, static_cast<off_t>(offset));
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) error("write failed: " + std::string(std::strerror(errno)));
    offset = add(offset, static_cast<uint64_t>(n));
    data = data.subspan(static_cast<size_t>(n));
  }
}
void read_all(int fd, std::span<uint8_t> data, uint64_t offset) {
  while (!data.empty()) {
    const auto n = ::pread(fd, data.data(), std::min<size_t>(data.size(), 1ull << 30), static_cast<off_t>(offset));
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) error("short or failed metadata read");
    offset = add(offset, static_cast<uint64_t>(n));
    data = data.subspan(static_cast<size_t>(n));
  }
}
std::vector<ChunkLayout> layouts(const StoreIdentity& id) {
  if (std::all_of(id.checkpoint_sha256.begin(), id.checkpoint_sha256.end(), [](uint8_t v) { return v == 0; }))
    error("a complete source identity is required");
  return partition(id.hidden, id.intermediate, id.world, id.rank, id.chunk_intermediate, id.format);
}
std::vector<StoreRecord> plan(const StoreIdentity& id, const std::vector<RecordKey>& keys, uint64_t* file_bytes) {
  if (keys.empty() || keys.size() > (static_cast<uint64_t>(std::numeric_limits<off_t>::max()) - kPage) / kEntry)
    error("invalid record count");
  const auto chunks = layouts(id);
  uint64_t offset = aligned(add(kPage, keys.size() * kEntry));
  std::set<std::tuple<uint32_t, uint32_t, uint32_t, bool>> seen;
  std::vector<StoreRecord> records;
  records.reserve(keys.size());
  for (const auto& key : keys) {
    if (key.chunk >= chunks.size() || !seen.emplace(key.layer, key.expert, key.chunk, key.draft).second)
      error("duplicate or out-of-range chunk key");
    records.push_back({key, chunks[key.chunk], offset, 0});
    offset = add(offset, chunks[key.chunk].record_bytes);
  }
  *file_bytes = offset;
  return records;
}
std::vector<uint8_t> index_bytes(const std::vector<StoreRecord>& records) {
  std::vector<uint8_t> bytes(records.size() * kEntry);
  for (size_t i = 0; i < records.size(); ++i) {
    auto entry = std::span<uint8_t>(bytes).subspan(i * kEntry, kEntry);
    const auto& r = records[i];
    put(entry, 0, r.key.layer, 4);
    put(entry, 4, r.key.expert, 4);
    put(entry, 8, r.key.chunk, 4);
    put(entry, 12, r.key.draft ? 1 : 0, 4);
    put(entry, 16, r.layout.intermediate_begin);
    put(entry, 24, r.layout.intermediate_size);
    put(entry, 32, r.file_offset);
    put(entry, 40, r.layout.record_bytes);
    put(entry, 48, r.checksum);
    put(entry, 56, r.layout.allocation_bytes);
  }
  return bytes;
}
std::array<uint8_t, kPage> header(const StoreIdentity& id, size_t records, uint64_t file_bytes, uint64_t index_hash) {
  std::array<uint8_t, kPage> h{};
  std::copy(kMagic.begin(), kMagic.end(), h.begin());
  put(h, 8, kVersion, 4);
  put(h, 12, id.format == Format::kMxfp4 ? 0 : 1, 4);
  put(h, 16, id.rank, 4);
  put(h, 20, id.world, 4);
  put(h, 24, id.hidden);
  put(h, 32, id.intermediate);
  put(h, 40, id.chunk_intermediate);
  put(h, 48, records);
  put(h, 56, file_bytes);
  std::copy(id.checkpoint_sha256.begin(), id.checkpoint_sha256.end(), h.begin() + 64);
  put(h, 96, index_hash);
  put(h, 104, hash(std::span<const uint8_t>(h).first(104)));
  return h;
}
std::filesystem::path parent(const std::string& path) {
  const auto p = std::filesystem::path(path).parent_path();
  return p.empty() ? std::filesystem::path(".") : p;
}
}  // namespace

ExpertStoreBuilder::ExpertStoreBuilder(std::string path, StoreIdentity identity, std::vector<RecordKey> keys)
    : path_(std::move(path)), identity_(identity), written_(keys.size(), false) {
  records_ = plan(identity, keys, &file_bytes_);
  temporary_ = path_ + ".tmp." + std::to_string(::getpid()) + "." + std::to_string(temporary_id++);
  // Anonymous images disappear when the last descriptor closes, including
  // SIGKILL during preparation. Unsupported filesystems retain named staging.
#ifdef O_TMPFILE
  fd_ = ::open(parent(path_).c_str(), O_TMPFILE | O_RDWR | O_CLOEXEC, 0600);
  anonymous_ = fd_ >= 0;
#endif
  if (fd_ < 0) fd_ = ::open(temporary_.c_str(), O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
  if (fd_ < 0) error("create temporary image: " + std::string(std::strerror(errno)));
  temporary_linked_ = !anonymous_;
  if (::ftruncate(fd_, static_cast<off_t>(file_bytes_)) != 0) {
    const auto why = std::string(std::strerror(errno));
    ::close(fd_);
    fd_ = -1;
    if (temporary_linked_) ::unlink(temporary_.c_str());
    error("size temporary image: " + why);
  }
  // Reserve the complete image before packing, rather than discovering a
  // full disk after minutes of reads and writes to a sparse file.
  const int allocated = ::posix_fallocate(fd_, 0, static_cast<off_t>(file_bytes_));
  if (allocated && allocated != EOPNOTSUPP && allocated != ENOSYS && allocated != EINVAL) {
    ::close(fd_);
    fd_ = -1;
    if (temporary_linked_) ::unlink(temporary_.c_str());
    error("reserve temporary image: " + std::string(std::strerror(allocated)));
  }
}
ExpertStoreBuilder::~ExpertStoreBuilder() {
  if (fd_ >= 0) ::close(fd_);
  if (!published_ && temporary_linked_) ::unlink(temporary_.c_str());
}
void ExpertStoreBuilder::write(size_t index, const PackedChunk& chunk) {
  if (published_ || index >= records_.size() || written_[index]) error("invalid or duplicate record write");
  auto& record = records_[index];
  // Compare the complete layout, not only its total size, before reading bytes.
  const auto& expected = record.layout;
  if (chunk.layout.intermediate_begin != expected.intermediate_begin ||
      chunk.layout.intermediate_size != expected.intermediate_size || chunk.bytes.size() != expected.allocation_bytes)
    error("chunk does not match planned record");
  // The authoritative view is the identity-derived layout, not the caller's
  // mutable pointer offsets. Validate without allocating a full expert.
  const auto views_equal = [](const MatrixLayout& a, const MatrixLayout& b) {
    return a.rows == b.rows && a.cols == b.cols && a.payload_offset == b.payload_offset &&
           a.payload_bytes == b.payload_bytes && a.scales_offset == b.scales_offset &&
           a.scales_bytes == b.scales_bytes && a.global_offset == b.global_offset;
  };
  if (!views_equal(chunk.layout.gate, expected.gate) || !views_equal(chunk.layout.up, expected.up) ||
      !views_equal(chunk.layout.down, expected.down) || chunk.layout.useful_bytes != expected.useful_bytes ||
      chunk.layout.record_bytes != expected.record_bytes || chunk.layout.allocation_bytes != expected.allocation_bytes)
    error("invalid packed chunk layout");
  write_all(fd_, chunk.bytes, record.file_offset);
  const std::array<uint8_t, kPage> zeros{};
  const auto padding = std::span<const uint8_t>(zeros).first(expected.record_bytes - expected.allocation_bytes);
  write_all(fd_, padding, add(record.file_offset, expected.allocation_bytes));
  record.checksum = hash(padding, hash(chunk.bytes));
  written_[index] = true;
}
void ExpertStoreBuilder::finish() {
  if (published_ || std::find(written_.begin(), written_.end(), false) != written_.end())
    error("cannot publish an incomplete or already published image");
  const auto index = index_bytes(records_);
  write_all(fd_, index, kPage);
  const auto h = header(identity_, records_.size(), file_bytes_, hash(index));
  write_all(fd_, h, 0);
  if (::fdatasync(fd_) != 0) error("sync temporary image failed");
  if (anonymous_) {
    const auto descriptor = "/proc/self/fd/" + std::to_string(fd_);
    if (::linkat(AT_FDCWD, descriptor.c_str(), AT_FDCWD, temporary_.c_str(), AT_SYMLINK_FOLLOW) != 0)
      error("link completed image: " + std::string(std::strerror(errno)));
    temporary_linked_ = true;
  }
  if (::rename(temporary_.c_str(), path_.c_str()) != 0) error("publish image failed");
  published_ = true;
  const int dir = ::open(parent(path_).c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (dir < 0) error("open publication directory failed (image already renamed)");
  const int synced = ::fsync(dir);
  ::close(dir);
  if (synced != 0) error("sync publication directory failed (image already renamed)");
}

ExpertStore::ExpertStore(const std::string& path, const StoreIdentity& expected, size_t expected_records,
                         IoMode mode, bool verify_record_checksums)
    : identity_(expected), mode_(mode), verify_record_checksums_(verify_record_checksums) {
  if (mode != IoMode::kBuffered && mode != IoMode::kDirect) error("unknown I/O mode");
  const int metadata = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (metadata < 0) error("open image failed");
  int body = -1;
  try {
    struct stat st{};
    if (::fstat(metadata, &st) != 0 || !S_ISREG(st.st_mode) || st.st_size < static_cast<off_t>(kPage))
      error("invalid image file");
    std::array<uint8_t, kPage> h{};
    read_all(metadata, h, 0);
    if (!std::equal(kMagic.begin(), kMagic.end(), h.begin()) || get(h, 8, 4) != kVersion ||
        get(h, 104) != hash(std::span<const uint8_t>(h).first(104)) ||
        get(h, 48) != expected_records || get(h, 56) != static_cast<uint64_t>(st.st_size) ||
        !expected_records || expected_records > (static_cast<uint64_t>(st.st_size) - kPage) / kEntry)
      error("invalid header or record population");
    const auto expected_header = header(expected, expected_records, static_cast<uint64_t>(st.st_size), get(h, 96));
    if (h != expected_header) error("image identity or layout version mismatch");
    std::vector<uint8_t> index(expected_records * kEntry);
    read_all(metadata, index, kPage);
    if (hash(index) != get(h, 96)) error("index checksum mismatch");
    std::vector<RecordKey> keys;
    keys.reserve(expected_records);
    for (size_t i = 0; i < expected_records; ++i) {
      const auto e = std::span<const uint8_t>(index).subspan(i * kEntry, kEntry);
      if (get(e, 12, 4) > 1) error("invalid record flags");
      keys.push_back({static_cast<uint32_t>(get(e, 0, 4)), static_cast<uint32_t>(get(e, 4, 4)),
                      static_cast<uint32_t>(get(e, 8, 4)), get(e, 12, 4) != 0});
    }
    uint64_t planned_bytes = 0;
    records_ = plan(expected, keys, &planned_bytes);
    if (planned_bytes != static_cast<uint64_t>(st.st_size)) error("invalid body extents");
    for (size_t i = 0; i < records_.size(); ++i) records_[i].checksum = get(index, i * kEntry + 48);
    if (index_bytes(records_) != index) error("record extent/geometry mismatch");
    if (mode == IoMode::kDirect) {
      body = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
      if (body < 0) error("direct I/O unavailable (buffered fallback was not requested)");
      struct stat other{};
      if (::fstat(body, &other) != 0 || other.st_dev != st.st_dev || other.st_ino != st.st_ino || other.st_size != st.st_size)
        error("image changed during open");
      fd_ = body;
      body = -1;
      ::close(metadata);
    } else {
      fd_ = metadata;
    }
  } catch (...) {
    ::close(metadata);
    if (body >= 0) ::close(body);
    throw;
  }
  // Optional diagnostic only: fixed-size completion records, no model payload.
  // One append per read preserves worker concurrency without a tracing mutex.
  try {
    if (const char* prefix = std::getenv("DGPP_EXPERT_READ_TRACE"); prefix && *prefix) {
      const auto trace = std::string(prefix) + ".r" + std::to_string(identity_.rank) +
          ".p" + std::to_string(::getpid()) + ".bin";
      trace_fd_ = ::open(trace.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
      if (trace_fd_ < 0) error("cannot open diagnostic read trace");
    }
  } catch (...) { ::close(fd_); fd_ = -1; throw; }
}
ExpertStore::~ExpertStore() {
  if (trace_fd_ >= 0) ::close(trace_fd_);
  if (fd_ >= 0) ::close(fd_);
}
bool ExpertStore::can_split(size_t index) const {
  const auto& layout = records_.at(index).layout;
  return identity_.format == Format::kMxfp4 && !verify_record_checksums_ &&
      layout.down.payload_offset > 0 && layout.down.payload_offset % kPage == 0;
}
void ExpertStore::read(size_t index, std::span<uint8_t> destination, ExpertProjection projection) const {
  if (index >= records_.size()) error("invalid read index");
  const auto& record = records_[index];
  if (destination.size() != record.layout.record_bytes ||
      (mode_ == IoMode::kDirect && reinterpret_cast<uintptr_t>(destination.data()) % kPage))
    error("read staging has invalid size/alignment");
  const auto timestamp = [] { return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::system_clock::now().time_since_epoch()).count()); };
  const uint64_t trace_start = trace_fd_ >= 0 ? timestamp() : 0;
  auto remaining = destination;
  uint64_t offset = record.file_offset;
  if (projection != ExpertProjection::kAll) {
    if (!can_split(index)) error("unsupported projection read");
    const size_t split = record.layout.down.payload_offset;
    if (projection == ExpertProjection::kGateUp) remaining = remaining.first(split);
    else { remaining = remaining.subspan(split); offset += split; }
  }
  const auto read_size = remaining.size();
  const auto read_offset = offset;
  try {
    while (!remaining.empty()) {
      calls_.fetch_add(1, std::memory_order_relaxed);
      const auto n = ::pread(fd_, remaining.data(), std::min<size_t>(remaining.size(), 1ull << 30), static_cast<off_t>(offset));
      if (n < 0 && errno == EINTR) continue;
      if (n <= 0) error("short or failed record read");
      bytes_.fetch_add(static_cast<uint64_t>(n), std::memory_order_relaxed);
      offset = add(offset, static_cast<uint64_t>(n));
      remaining = remaining.subspan(static_cast<size_t>(n));
    }
    if (verify_record_checksums_ && hash(destination) != record.checksum) error("record checksum mismatch");
    if (trace_fd_ >= 0) {
      // Native uint64 fields: start/end Unix ns, layer, expert, chunk, draft,
      // record bytes, file offset. Consumers must reject incomplete records.
      const uint64_t event[]{trace_start, timestamp(), record.key.layer, record.key.expert,
          record.key.chunk, record.key.draft, read_size, read_offset};
      ssize_t written;
      do { written = ::write(trace_fd_, event, sizeof(event)); } while (written < 0 && errno == EINTR);
      if (written != static_cast<ssize_t>(sizeof(event))) error("incomplete diagnostic read trace");
    }
  } catch (...) {
    failures_.fetch_add(1, std::memory_order_relaxed);
    throw;
  }
}
ReadStats ExpertStore::stats() const {
  return {calls_.load(std::memory_order_relaxed), bytes_.load(std::memory_order_relaxed), failures_.load(std::memory_order_relaxed)};
}
}  // namespace dgpp::expert
