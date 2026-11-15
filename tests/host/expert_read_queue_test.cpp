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
  Fixture() {
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
    for (size_t i = 0; i < keys.size(); ++i) writer.write(i, pack_chunk(source, keys[i].chunk * 32, 32));
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
