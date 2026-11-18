#include "loaders/expert_read_queue.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "common/test.hpp"

namespace {
using namespace dgpp::expert;
using namespace std::chrono_literals;
void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
template <typename F> void refuses(F f) {
  bool rejected = false;
  try { f(); } catch (const std::exception&) { rejected = true; }
  require(rejected, "invalid/failed work must be refused");
}
template <typename F> void eventually(F predicate) {
  const auto deadline = std::chrono::steady_clock::now() + 3s;
  while (!predicate()) {
    if (std::chrono::steady_clock::now() >= deadline) throw std::runtime_error("queue did not make expected progress");
    std::this_thread::sleep_for(1ms);
  }
}
std::shared_ptr<const ReadLease> ready(const std::shared_ptr<ReadTicket>& t) {
  require(t != nullptr && t->wait_for(3s), "ticket completes within test deadline");
  return t->wait();
}
struct Fixture {
  std::filesystem::path dir;
  std::string path;
  std::shared_ptr<ExpertStore> store;
  explicit Fixture(bool incompatible_scale = false) {
    std::string pattern = "/tmp/dgpp-expert-queue-XXXXXX";
    require(::mkdtemp(pattern.data()) != nullptr, "mkdtemp");
    dir = pattern;
    path = (dir / "experts.img").string();
    StoreIdentity id;
    id.checkpoint_sha256.fill(17);
    id.hidden = 32;
    id.intermediate = 128;
    id.chunk_intermediate = 32;
    std::vector<RecordKey> keys;
    for (uint32_t e = 0; e < 2; ++e)
      for (uint32_t c = 0; c < 4; ++c) keys.push_back({0, e, c, false});
    std::vector<uint8_t> codes(32 * 128 / 2, 77), scales(32 * 128 / 32, 127);
    std::array<MatrixSource, 3> source{{{128, 32, Format::kMxfp4, codes, scales, {}},
                                       {128, 32, Format::kMxfp4, codes, scales, {}},
                                       {32, 128, Format::kMxfp4, codes, scales, {}}}};
    ExpertStoreBuilder writer(path, id, keys);
    for (size_t i = 0; i < keys.size(); ++i) {
      auto chunk = pack_chunk(source, keys[i].chunk * 32, 32);
      if (incompatible_scale && i == 1) chunk.bytes[chunk.layout.down.scales_offset + 7] = 128;
      writer.write(i, chunk);
    }
    writer.finish();
    store = std::make_shared<ExpertStore>(path, id, keys.size(), IoMode::kBuffered);
  }
  ~Fixture() { std::error_code ec; std::filesystem::remove_all(dir, ec); }
  ReadQueueConfig config(size_t pending = 4) const {
    ReadQueueConfig cfg;
    cfg.buffers = 1;
    cfg.workers = 1;
    cfg.max_pending = pending;
    cfg.staging_budget_bytes = store->records()[0].layout.record_bytes;
    cfg.prefetch_bytes = 3 * cfg.staging_budget_bytes;
    return cfg;
  }
};
}  // namespace

DGPP_TEST(expert_read_queue_refuses_unrepresentable_mx_scale_before_publication) {
  Fixture f(true);
  auto cfg = f.config();
  cfg.require_mx_nibbles = true;
  ExpertReadQueue queue(f.store, cfg);
  require(ready(queue.request(0)) != nullptr, "representable record accepted");
  refuses([&] { ready(queue.request(1)); });
  require(ready(queue.request(2)) != nullptr, "later valid record still readable");
}
DGPP_TEST(expert_read_queue_vectors_reserved_records_and_checks_each_lease) {
  Fixture f(true);
  auto store = std::make_shared<ExpertStore>(f.path, f.store->identity(), 8, IoMode::kBuffered, false);
  auto cfg = f.config();
  cfg.buffers = cfg.adjacent_records = 4;
  cfg.staging_budget_bytes *= cfg.buffers;
  cfg.require_mx_nibbles = true;
  ExpertReadQueue queue(store, cfg);
  const std::array<size_t,4> records{0, 1, 2, 3};
  auto tickets = queue.reserve_demand(records);
  require(tickets.size() == 4, "all four records have staging");
  auto first = ready(tickets[0]);
  refuses([&] { (void)ready(tickets[1]); });
  auto third = ready(tickets[2]);
  auto fourth = ready(tickets[3]);
  require(store->stats().calls == 1 && queue.stats().failed == 1, "one vectored read, one bad scale");
  require(first->bytes().data() != third->bytes().data() && third->bytes().data() != fourth->bytes().data(),
          "each successful ticket owns a distinct buffer");
  tickets.clear();
  third.reset();
  fourth.reset();
  eventually([&] { return queue.stats().staging_in_use == first->bytes().size(); });
}
DGPP_TEST(expert_read_queue_split_joins_original_record) {
  Fixture f;
  const auto path = (f.dir / "projection.img").string();
  StoreIdentity id; id.checkpoint_sha256.fill(29);
  id.hidden = 1024; id.intermediate = id.chunk_intermediate = 64;
  const std::vector<RecordKey> keys{{0,0,0,false}};
  std::vector<uint8_t> codes(1024*64/2), scales(1024*64/32, 120);
  for (size_t i = 0; i < codes.size(); ++i) codes[i] = static_cast<uint8_t>(i*37+i/23);
  std::array<MatrixSource,3> source{{{64,1024,Format::kMxfp4,codes,scales,{}},
      {64,1024,Format::kMxfp4,codes,scales,{}}, {1024,64,Format::kMxfp4,codes,scales,{}}}};
  ExpertStoreBuilder writer(path, id, keys);
  const auto chunk = pack_chunk(source, 0, 64);
  writer.write(0, chunk); writer.finish();
  auto store = std::make_shared<ExpertStore>(path, id, 1, IoMode::kDirect, false);
  const size_t n = chunk.layout.record_bytes, cut = chunk.layout.down.payload_offset;
  ReadQueueConfig cfg;
  cfg.buffers = cfg.max_pending = 2; cfg.staging_budget_bytes = 2*n; cfg.require_mx_nibbles = true;
  const std::array<size_t,1> record{0};
  ExpertReadQueue queue(store, cfg);
  const auto tickets = queue.reserve_projections(record);
  require(tickets.size() == 2, "one reservation for both halves");
  auto first = ready(tickets[0]), last = ready(tickets[1]);
  require(std::equal(chunk.bytes.begin(), chunk.bytes.begin()+cut, first->bytes().begin()) &&
          std::equal(chunk.bytes.begin()+cut, chunk.bytes.end(), last->bytes().begin()+cut),
          "projection ranges reproduce canonical chunk");
  require(store->stats().returned_bytes == n && store->stats().calls == 2, "two reads total one record");
  ExpertStore checked(path, id, 1, IoMode::kDirect, true);
  require(!checked.can_split(0), "whole-record verification requires full reads");
}
DGPP_TEST(expert_read_queue_classifies_nvfp4_scales_without_changing_record_bytes) {
  Fixture f;
  const auto path = (f.dir / "nvfp4.img").string();
  StoreIdentity id; id.format = Format::kNvfp4;
  id.checkpoint_sha256.fill(23);
  id.hidden = 64; id.intermediate = 64; id.chunk_intermediate = 64;
  const std::vector<RecordKey> keys{{0,0,0,false}, {0,1,0,false}, {0,2,0,false}};
  std::vector<uint8_t> codes(64*64/2, 77), scales(64*64/16, 72);
  std::array<MatrixSource,3> source{{{64,64,Format::kNvfp4,codes,scales,1.f},
                                     {64,64,Format::kNvfp4,codes,scales,1.f},
                                     {64,64,Format::kNvfp4,codes,scales,1.f}}};
  ExpertStoreBuilder writer(path,id,keys);
  auto exact = pack_chunk(source,0,64);
  writer.write(0,exact);
  auto low = exact; low.bytes[low.layout.up.scales_offset+3] |= 1;
  writer.write(1,low);
  auto high = exact; high.bytes[high.layout.down.scales_offset+7] |= 128;
  writer.write(2,high);
  writer.finish();
  auto store = std::make_shared<ExpertStore>(path,id,keys.size(),IoMode::kBuffered);
  ReadQueueConfig cfg; cfg.buffers=1; cfg.workers=1; cfg.max_pending=2;
  cfg.staging_budget_bytes=store->records()[0].layout.record_bytes;
  cfg.scan_nvfp4_scales=true;
  ExpertReadQueue queue(store,cfg);
  for (size_t i=0;i<keys.size();++i) {
    auto ticket=queue.request(i);
    const auto lease=ready(ticket);
    require(lease->scale_compactibility() == (i ? ScaleCompactibility::kUnsupported : ScaleCompactibility::kExact),
            "unsupported scale bits cannot enter a compact representation");
    const auto& expected = i==0 ? exact : i==1 ? low : high;
    require(std::equal(expected.bytes.begin(),expected.bytes.end(),lease->bytes().begin()),
            "scale classification preserves the original staged record");
  }
  cfg.scan_nvfp4_scales=false;
  ExpertReadQueue disabled(store,cfg);
  require(ready(disabled.request(0))->scale_compactibility()==ScaleCompactibility::kUnknown,
          "disabled scan never implies compactibility");
  cfg.scan_nvfp4_scales=true;
  ExpertReadQueue other_format(f.store,cfg);
  require(ready(other_format.request(0))->scale_compactibility()==ScaleCompactibility::kUnknown,
          "MXFP4 is never classified as NVFP4");
}

DGPP_TEST(expert_read_queue_coalesces_concurrent_consumers_without_duplicate_reads) {
  Fixture f;
  auto cfg = f.config();
  cfg.buffers = cfg.workers = 2;
  cfg.staging_budget_bytes *= 2;
  ExpertReadQueue queue(f.store, cfg);
  std::vector<std::shared_ptr<ReadTicket>> tickets(32);
  std::vector<std::thread> requesters;
  for (size_t i = 0; i < tickets.size(); ++i) requesters.emplace_back([&, i] { tickets[i] = queue.request(0); });
  for (auto& thread : requesters) thread.join();
  const auto first = ready(tickets[0]);
  for (const auto& t : tickets) require(t == tickets[0] && ready(t) == first, "shared ticket and immutable lease");
  const auto stats = queue.stats();
  require(stats.accepted == 1 && stats.coalesced == 31 && stats.completed == 1, "one completed underlying read");
  require(f.store->stats().calls == 1, "one actual file read for concurrent waiters");
  require(stats.staging_peak <= stats.staging_reserved && stats.staging_reserved == cfg.staging_budget_bytes,
          "staging allocation obeys exact reservation");
  require(queue.staging_spans().size() == 2, "engine can register preallocated aligned spans");
}

DGPP_TEST(expert_read_queue_demand_window_reads_coalesce_and_disarm) {
  Fixture f;
  auto cfg = f.config();
  cfg.buffers = cfg.workers = 2;
  cfg.staging_budget_bytes *= 2;
  ExpertReadQueue queue(f.store, cfg);
  // Exercise wakeup from idle, repeated arming, and normal CV reads after disarm.
  for (size_t record = 0; record < 8; ++record) {
    const auto start = std::chrono::steady_clock::now();
    queue.set_demand_window(true);
    queue.set_demand_window(true);
    require(std::chrono::steady_clock::now() - start < 1s, "arming is bounded");
    std::this_thread::sleep_for(2ms);
    auto ticket = queue.request(record);
    auto joined = queue.request(record);
    require(ticket == joined, "armed reader preserves coalescing");
    const auto lease = ready(ticket);
    require(!lease->bytes().empty() && lease->bytes()[0] == 77, "armed reader returns record bytes");
    queue.set_demand_window(false);
    queue.set_demand_window(false);
  }
  eventually([&] { return queue.stats().staging_in_use == 0; });
  (void)ready(queue.request(0));
  require(queue.stats().completed == 9 && queue.stats().coalesced == 8 &&
              queue.stats().pending == 0 && f.store->stats().calls == 9,
          "armed and disarmed reads retain normal accounting");
}

DGPP_TEST(expert_read_queue_demand_window_empty_shutdown_is_bounded) {
  Fixture f;
  ExpertReadQueue queue(f.store, f.config());
  // Leave the window armed and empty so shutdown must interrupt the poll.
  queue.set_demand_window(true);
  std::this_thread::sleep_for(10ms);
  require(f.store->stats().calls == 0, "arming alone does not issue reads");
  const auto start = std::chrono::steady_clock::now();
  queue.shutdown();
  require(std::chrono::steady_clock::now() - start < 1s, "shutdown interrupts an armed empty reader");
  queue.shutdown();
  require(queue.stats().pending == 0 && queue.stats().active == 0, "empty shutdown stays drained");
}

DGPP_TEST(expert_read_queue_lease_backpressure_never_reuses_live_staging) {
  Fixture f;
  ExpertReadQueue queue(f.store, f.config(2));
  auto first = queue.request(0);
  auto lease = ready(first);
  const auto original = std::vector<uint8_t>(lease->bytes().begin(), lease->bytes().end());
  auto second = queue.request(1);
  eventually([&] { return queue.stats().active == 1; });
  auto third = queue.request(2);
  require(third != nullptr && queue.request(3) == nullptr, "bounded pending population applies backpressure");
  auto joined = queue.request(2);
  require(joined == third, "a full queue still coalesces an existing miss");
  require(!second->ready() && !third->ready(), "live lease prevents staging reuse");
  require(std::equal(original.begin(), original.end(), lease->bytes().begin()), "leased bytes remain unchanged");
  first.reset();
  lease.reset();
  (void)ready(second);
  second.reset();
  (void)ready(third);
  third.reset();
  joined.reset();
  eventually([&] { return queue.stats().staging_in_use == 0; });
  require(queue.stats().completed == 3 && queue.stats().staging_peak == f.config().staging_budget_bytes,
          "all work progresses through one bounded staging slot");
}

DGPP_TEST(expert_read_queue_demand_promotes_and_displaces_queued_prefetch) {
  Fixture f;
  ExpertReadQueue queue(f.store, f.config(3));
  queue.set_demand_window(true);
  auto first = queue.request(0);
  auto held = ready(first);
  auto active = queue.request(1, ReadPriority::kPrefetch);
  eventually([&] { return queue.stats().active == 1; });
  auto promoted = queue.request(2, ReadPriority::kPrefetch);
  auto joined = queue.request(2, ReadPriority::kDemand);
  require(joined == promoted && queue.stats().promoted == 1, "demand promotes the existing speculative read");
  auto displaced = queue.request(3, ReadPriority::kPrefetch);
  auto urgent = queue.request(4, ReadPriority::kDemand);
  require(urgent != nullptr && displaced->wait_for(3s), "demand displaces a queued speculative job");
  refuses([&] { (void)displaced->wait(); });
  first.reset();
  held.reset();
  (void)ready(active);
  active.reset();
  (void)ready(promoted);
  promoted.reset();
  joined.reset();
  (void)ready(urgent);
  urgent.reset();
  eventually([&] { return queue.stats().staging_in_use == 0; });
  const auto stats = queue.stats();
  require(stats.completed == 4 && stats.cancelled == 1 && stats.speculative_bytes == 0 && stats.pending == 0,
          "prefetch accounting reconciles across promotion and displacement");
  require(f.store->stats().calls == 4, "displaced prefetch performs no read");
}

DGPP_TEST(expert_read_queue_shutdown_cancels_waiting_work_without_invalidating_completed_leases) {
  Fixture f;
  std::shared_ptr<const ReadLease> survivor;
  std::shared_ptr<ReadTicket> blocked, queued;
  {
    ExpertReadQueue queue(f.store, f.config());
    queue.set_demand_window(true);
    auto first = queue.request(0);
    survivor = ready(first);
    blocked = queue.request(1);
    eventually([&] { return queue.stats().active == 1; });
    queued = queue.request(2);
    queue.shutdown();
    queue.shutdown();
    refuses([&] { (void)blocked->wait(); });
    refuses([&] { (void)queued->wait(); });
    refuses([&] { (void)queue.request(3); });
    require(queue.stats().pending == 0 && queue.stats().active == 0 && queue.stats().cancelled == 2,
            "shutdown drains queued and buffer-blocked work exactly once");
  }
  require(!survivor->bytes().empty() && survivor->bytes()[0] == 77, "lease remains valid after worker/pool owner shutdown");
  require(f.store->stats().calls == 1, "shutdown did not issue blocked reads");
}

DGPP_TEST(expert_read_queue_coalesced_failure_wakes_every_waiter_and_does_not_retry_automatically) {
  Fixture f;
  ExpertReadQueue queue(f.store, f.config());
  auto first = queue.request(0);
  auto held = ready(first);
  const auto offset = f.store->records()[1].file_offset;
  {
    std::fstream file(f.path, std::ios::in | std::ios::out | std::ios::binary);
    file.seekp(static_cast<std::streamoff>(offset));
    file.put(0);
    file.flush();
    require(static_cast<bool>(file), "corrupt fixture record");
  }
  std::vector<std::shared_ptr<ReadTicket>> waiters;
  for (size_t i = 0; i < 16; ++i) waiters.push_back(queue.request(1));
  first.reset();
  held.reset();
  for (const auto& t : waiters) {
    require(t == waiters[0] && t->wait_for(3s), "every waiter shares one completed failure");
    refuses([&] { (void)t->wait(); });
  }
  require(queue.stats().failed == 1 && f.store->stats().calls == 2, "one failing read, no blind retry");
}

DGPP_TEST(expert_read_queue_released_waiters_cancel_before_blocked_read_and_budget_is_strict) {
  Fixture f;
  auto cfg = f.config();
  --cfg.staging_budget_bytes;
  refuses([&] { ExpertReadQueue queue(f.store, cfg); });
  cfg = f.config();
  cfg.workers = 2;
  refuses([&] { ExpertReadQueue queue(f.store, cfg); });
  cfg = f.config();
  cfg.prefetch_bytes = 0;
  ExpertReadQueue queue(f.store, cfg);
  require(queue.request(0, ReadPriority::kPrefetch) == nullptr, "zero prefetch allowance is enforced");
  auto first = queue.request(0);
  auto held = ready(first);
  auto abandoned = queue.request(1);
  eventually([&] { return queue.stats().active == 1; });
  abandoned.reset();
  first.reset();
  held.reset();
  eventually([&] { return queue.stats().cancelled == 1 && queue.stats().pending == 0; });
  require(f.store->stats().calls == 1, "abandoned buffer-blocked job does not perform I/O");
  auto next = queue.request(2);
  (void)ready(next);
  require(f.store->stats().calls == 2, "remaining requests still progress");
  refuses([&] { (void)queue.request(999); });
}

int main() { return dgpp::test::run_all(); }

DGPP_TEST(expert_read_queue_wait_any_skips_blocked_first_ticket) {
  Fixture f;
  ExpertReadQueue queue(f.store, f.config());
  auto complete = queue.request(0);
  auto held = ready(complete);
  auto blocked = queue.request(1);
  std::vector<std::shared_ptr<ReadTicket>> tickets{blocked, complete};
  require(queue.wait_any(tickets) == 1, "completed later ticket bypasses blocked first read");
  tickets.clear(); complete.reset(); held.reset();
  tickets.push_back(blocked);
  require(queue.wait_any(tickets) == 0, "completion wakes waiter");
  (void)blocked->wait();
  refuses([&] { queue.wait_any({}); });
}

DGPP_TEST(expert_read_queue_reserved_batch_is_atomic_and_reuses_destinations) {
  Fixture f;
  auto cfg = f.config(8); cfg.buffers = 4; cfg.staging_budget_bytes *= 4;
  ExpertReadQueue queue(f.store,cfg);
  const std::array<size_t,4> records{0,1,2,3};
  auto batch = queue.reserve_demand(records);
  require(batch.size()==4,"complete group reserves all destinations");
  require(queue.reserve_demand(std::array<size_t,1>{4}).empty(),"held destinations cause nonblocking refusal");
  const auto before = queue.stats().accepted;
  auto joined = queue.reserve_demand(records);
  require(joined==batch && queue.stats().accepted==before,"reserved group coalesces without more space or reads");
  refuses([&] { queue.reserve_demand(std::array<size_t,2>{4,4}); });
  for (const auto& t : batch) require(ready(t)->bytes()[0]==77,"reserved bytes are correct");
  require(queue.stats().staging_peak==cfg.staging_budget_bytes,"destinations count against actual staging budget");
  joined.clear(); batch.clear();
  eventually([&] { return queue.stats().staging_in_use==0; });
  auto next = queue.reserve_demand(std::array<size_t,2>{4,5});
  require(next.size()==2,"released group credits immediately reusable");
  for (const auto& t : next) (void)ready(t);
  require(f.store->stats().calls==6,"no duplicate reads for reservations");
}

DGPP_TEST(expert_read_queue_reserved_work_cannot_be_starved_by_buffer_waiters) {
  Fixture f;
  auto cfg=f.config(8); cfg.buffers=2; cfg.staging_budget_bytes*=2;
  ExpertReadQueue queue(f.store,cfg);
  for (size_t attempt=0; attempt<32; ++attempt) {
    auto held=queue.request(0); (void)ready(held);
    auto held2=queue.request(1); (void)ready(held2);
    auto blocked=queue.request(2);
    eventually([&] { return queue.stats().active==1; });
    const auto accepted=queue.stats().accepted;
    require(queue.reserve_demand(std::array<size_t,1>{3}).empty(),"no destination means no submission");
    require(queue.stats().accepted==accepted,"failed reservation leaves queue unchanged");
    held.reset();
    auto urgent=queue.reserve_demand(std::array<size_t,1>{3});
    if (!urgent.empty()) (void)ready(urgent[0]); // blocked worker must yield to reserved job
    urgent.clear(); held2.reset();
    (void)ready(blocked); blocked.reset();
    eventually([&] { return queue.stats().staging_in_use==0; });
  }
}

DGPP_TEST(expert_read_queue_reserved_shutdown_releases_unconsumed_destinations) {
  Fixture f;
  auto cfg=f.config(8); cfg.buffers=4; cfg.staging_budget_bytes*=4;
  ExpertReadQueue queue(f.store,cfg);
  auto batch=queue.reserve_demand(std::array<size_t,4>{0,1,2,3});
  queue.shutdown();
  for (const auto& t:batch) require(t->wait_for(3s),"shutdown resolves all reserved tickets");
  batch.clear();
  require(queue.stats().pending==0 && queue.stats().active==0 && queue.stats().staging_in_use==0,
          "shutdown and ticket retirement return every destination");
}
