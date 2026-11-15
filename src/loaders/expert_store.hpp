#pragma once
// Immutable rank-local expert-chunk records. The builder accepts one chunk
// at a time; the reader fills caller-owned, budgeted staging without a copy.
#include <array>
#include <atomic>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "loaders/expert_residency.hpp"

namespace dgpp::expert {

struct StoreIdentity {
  std::array<uint8_t, 32> checkpoint_sha256{};
  Format format = Format::kMxfp4;
  uint32_t rank = 0, world = 1;
  uint64_t hidden = 0, intermediate = 0, chunk_intermediate = 0;
  bool operator==(const StoreIdentity&) const = default;
};
struct RecordKey {
  uint32_t layer = 0, expert = 0, chunk = 0;
  bool draft = false;
  bool operator==(const RecordKey&) const = default;
};
struct StoreRecord {
  RecordKey key;
  ChunkLayout layout;
  uint64_t file_offset = 0, checksum = 0;
};
enum class IoMode { kBuffered, kDirect };
struct ReadStats {
  uint64_t calls = 0, returned_bytes = 0, failures = 0;
};

class ExpertStoreBuilder {
 public:
  ExpertStoreBuilder(std::string path, StoreIdentity identity, std::vector<RecordKey> keys);
  ~ExpertStoreBuilder();
  ExpertStoreBuilder(const ExpertStoreBuilder&) = delete;
  ExpertStoreBuilder& operator=(const ExpertStoreBuilder&) = delete;
  const std::vector<StoreRecord>& records() const { return records_; }
  // Records may be supplied out of order, but exactly once. No publication
  // occurs until finish verifies that every planned record was written.
  void write(size_t record, const PackedChunk& chunk);
  void finish();

 private:
  std::string path_, temporary_;
  StoreIdentity identity_;
  std::vector<StoreRecord> records_;
  std::vector<bool> written_;
  uint64_t file_bytes_ = 0;
  int fd_ = -1;
  bool published_ = false;
};

class ExpertStore {
 public:
  // The model supplies expected_records from its binding plan so malformed
  // files cannot choose an arbitrarily large metadata allocation.
  ExpertStore(const std::string& path, const StoreIdentity& expected,
               size_t expected_records, IoMode mode, bool verify_record_checksums = true);
  ~ExpertStore();
  ExpertStore(const ExpertStore&) = delete;
  ExpertStore& operator=(const ExpertStore&) = delete;
  const std::vector<StoreRecord>& records() const { return records_; }
  IoMode io_mode() const { return mode_; }
  const StoreIdentity& identity() const { return identity_; }
  // destination.size must equal record_bytes, including direct-I/O padding.
  // Direct mode also requires 4 KiB pointer alignment. Checksums include
  // padding. Payload checksumming is optional for the serving reader; image
  // identity, metadata, extent and complete-read checks always apply.
  void read(size_t record, std::span<uint8_t> destination) const;
  ReadStats stats() const;

 private:
  std::vector<StoreRecord> records_;
  StoreIdentity identity_;
  IoMode mode_;
  bool verify_record_checksums_;
  int fd_ = -1, trace_fd_ = -1;
  mutable std::atomic<uint64_t> calls_{0}, bytes_{0}, failures_{0};
};

}  // namespace dgpp::expert
