#include "serve/rank_metrics.hpp"

#include <arpa/inet.h>
#include <netdb.h>

#include <memory>
#include <stdexcept>
#include <string>

namespace dgpp::serve {

namespace {

// Cluster nodes may be hostnames, while HttpServer binds numeric IPv4
// addresses. Resolve only at listener construction, never on a scrape.
std::string metrics_bind_address(const std::string& host) {
  addrinfo hints{};
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  addrinfo* list = nullptr;
  const int rc = ::getaddrinfo(host.c_str(), nullptr, &hints, &list);
  if (rc != 0 || list == nullptr)
    throw std::runtime_error("rank metrics: cannot resolve IPv4 bind host '" + host + "': " +
                             (rc == 0 ? "no addresses" : ::gai_strerror(rc)));
  const std::unique_ptr<addrinfo, decltype(&::freeaddrinfo)> addresses(list, ::freeaddrinfo);
  const auto* addr = reinterpret_cast<const sockaddr_in*>(list->ai_addr);
  char text[INET_ADDRSTRLEN];
  if (::inet_ntop(AF_INET, &addr->sin_addr, text, sizeof(text)) == nullptr)
    throw std::runtime_error("rank metrics: cannot format IPv4 bind host '" + host + "'");
  return text;
}

}  // namespace

void write_rank_metrics(prom::Writer& w, const RankIdentity& id,
                        const dgpp::sched::Scheduler::Meters& m, double snapshot_age_s,
                        int64_t ticks, const std::atomic<uint64_t>* collectives) {
  const std::string rank = "rank=\"" + std::to_string(id.rank) + "\"";
  w.gauge("dgpp_rank_info", "This rank's build and world; the value is always 1.", int64_t{1},
          rank + ",world_size=\"" + std::to_string(id.world) + "\",version=\"" + prom::escape_label(id.version) +
              "\",git_sha=\"" + prom::escape_label(id.git_sha) + "\"");
  w.gauge("dgpp_rank_snapshot_age_seconds",
          "Since this rank last published its meters (a peer publishes after every tick it applies).",
          snapshot_age_s, rank);
  if (ticks >= 0) w.counter("dgpp_rank_ticks_total", "Journal records this rank applied.", ticks, rank);
  w.counter("dgpp_rank_decode_steps_total", "Decode passes this rank ran.", m.decode_steps, rank);
  w.counter("dgpp_rank_prefill_chunk_steps_total", "Budgeted prefill advances on this rank; groups count once.", m.prefill_chunk_steps, rank);
  w.counter("dgpp_rank_prefill_chunk_budget_tokens_total", "Reader budgets offered to this rank's prefill advances.", m.prefill_chunk_budget_tokens, rank);
  w.counter("dgpp_rank_prefill_chunk_available_tokens_total", "Reader budgets capped by remaining prompt tokens, before structural cuts.", m.prefill_chunk_available_tokens, rank);
  w.counter("dgpp_rank_prefill_chunk_computed_tokens_total", "Reader tokens computed in this rank's budgeted advances.", m.prefill_chunk_computed_tokens, rank);
  w.gauge("dgpp_rank_prefill_adaptive_limit_tokens", "Last leader-selected prefill ceiling; 0 means fixed budgeting.",
          m.prefill_adaptive_limit_tokens, rank);
  w.counter("dgpp_rank_mixed_steps_total", "This rank's decode passes sharing a walk with prefill.", m.mixed_steps, rank);
  w.counter("dgpp_rank_mixed_seconds_total", "Shared time included in both prefill and decode seconds on this rank.",
            m.mixed_ms / 1000.0, rank);
  w.counter("dgpp_rank_decode_step_seconds_total",
            "This rank's wall time inside decode steps, collective waits included.", m.step_ms / 1000.0, rank);
  w.counter("dgpp_rank_prefill_seconds_total",
            "This rank's wall time inside prefill calls, collective waits included.", m.prefill_ms / 1000.0, rank);
  w.counter("dgpp_rank_generation_tokens_total", "Tokens this rank's scheduler generated.", m.tokens_generated,
            rank);
  w.gauge("dgpp_rank_num_requests_running", "Requests holding a slot on this rank.", int64_t{m.active}, rank);
  w.gauge("dgpp_rank_kv_pool_blocks_total", "This rank's K/V pool blocks.", m.pool_blocks_total, rank);
  w.gauge("dgpp_rank_kv_pool_blocks_in_use", "This rank's K/V pool blocks in use.", m.pool_blocks_in_use, rank);
  if (m.expert_streaming) {
    const auto& x = *m.expert_streaming;
    const auto& c = x.cache;
    const auto& q = x.queue;
    const std::string labels = rank + ",policy=\"" +
        (x.policy == expert::Policy::kWholeExpert ? "whole_expert_cache" : "adaptive_fractional_cache") +
        "\",io_mode=\"" + (x.io_mode == expert::IoMode::kDirect ? "direct" : "buffered") + "\"";
    w.counter("dgpp_rank_expert_requests_total", "Unique expert executions after route segmentation; cache state at first acquisition.", c.requests, labels);
    w.counter("dgpp_rank_expert_full_hits_total", "Executions whose entire expert was resident at first acquisition.", c.full_hits, labels);
    w.counter("dgpp_rank_expert_partial_hits_total", "Executions whose expert was partially resident at first acquisition.", c.partial_hits, labels);
    w.counter("dgpp_rank_expert_chunk_lookups_total", "Chunk lookups before demand reads or reconstruction.", c.chunk_lookups, labels);
    w.counter("dgpp_rank_expert_chunk_hits_total", "Chunks resident on device at first acquisition; prefetched file buffers do not count.", c.chunk_hits, labels);
    w.counter("dgpp_rank_expert_file_read_bytes_total", "Returned expert-image file bytes including padding; NOT physical NVMe bytes.", x.file.returned_bytes, labels);
    w.counter("dgpp_rank_expert_file_read_calls_total", "Expert-image read syscalls, including short reads and failures.", x.file.calls, labels);
    w.counter("dgpp_rank_expert_file_read_failures_total", "Failed expert-image reads or checksum checks.", x.file.failures, labels);
    w.counter("dgpp_rank_expert_read_wait_seconds_total", "Engine wait on expert read tickets; includes queueing, not a device-I/O clock.", static_cast<double>(c.read_wait_ns) / 1e9, labels);
    w.counter("dgpp_rank_expert_reconstruction_wall_seconds_total", "Successful reconstruction wall time excluding file-ticket waits; includes CPU work and completion fences, not GPU-only time.", static_cast<double>(c.reconstruction_wall_ns) / 1e9, labels);
    w.counter("dgpp_rank_expert_cached_copy_bytes_total", "Useful bytes copied from resident chunks into completed execution slots; excludes zeroing and allocation padding.", c.cached_copy_bytes, labels);
    w.counter("dgpp_rank_expert_chunk_direct_executions_total", "Experts executed directly from staged chunks.", c.chunk_direct_executions, labels);
    w.counter("dgpp_rank_expert_chunk_direct_bytes_total", "Useful chunk bytes consumed without reconstruction.", c.chunk_direct_bytes, labels);
    w.counter("dgpp_rank_expert_staged_copy_bytes_total", "Useful bytes copied from file staging into completed execution slots; logical copy volume, not physical NVMe traffic.", c.staged_copy_bytes, labels);
    w.counter("dgpp_rank_expert_execution_zero_bytes_total", "Execution-slot bytes zeroed for successful reconstructions; separate from useful copy volume.", c.execution_zero_bytes, labels);
    w.counter("dgpp_rank_expert_native_hits_total", "Native-layout acquisitions, including direct execution and canonical unpacking; NOT a copy-free execution count.", c.native_hits, labels);
    w.counter("dgpp_rank_expert_direct_native_hits_total", "Executions bound directly to native cache storage; first preparation is counted separately as a native pack.", c.direct_native_hits, labels);
    w.counter("dgpp_rank_expert_canonical_unpacks_total", "Native-resident acquisitions unpacked into canonical execution scratch.", c.canonical_unpacks, labels);
    w.counter("dgpp_rank_expert_canonical_unpack_bytes_total", "Useful canonical output bytes produced by native-hit unpacking; NOT physical DRAM traffic.", c.canonical_unpack_bytes, labels);
    w.counter("dgpp_rank_expert_canonical_unpack_enqueue_seconds_total", "CPU time enqueueing canonical unpack operations; NOT GPU unpack duration.", static_cast<double>(c.canonical_unpack_enqueue_ns) / 1e9, labels);
    w.counter("dgpp_rank_expert_completion_wait_seconds_total", "Host wait in cache backend completion fences, including pending copies and expert execution; overlaps reconstruction/layout wall timers.", static_cast<double>(c.completion_wait_ns) / 1e9, labels);
    w.counter("dgpp_rank_expert_native_packs_total", "Completed native layout conversions within the existing cache slab.", c.native_packs, labels);
    w.counter("dgpp_rank_expert_native_unpacks_total", "Native layouts restored to compact chunks before partial demotion or relocation.", c.native_unpacks, labels);
    w.counter("dgpp_rank_expert_native_layout_seconds_total", "Nonoverlapping packing/unpacking wall time; separate from reconstruction, nested within engine time.", static_cast<double>(c.native_layout_wall_ns) / 1e9, labels);
    w.counter("dgpp_rank_expert_native_layout_copy_bytes_total", "Native conversion and relocation copy volume; not physical memory bandwidth.", c.native_layout_copy_bytes, labels);
    w.gauge("dgpp_rank_expert_active_native_leases", "Direct native bindings that prevent cache eviction and relocation.", static_cast<double>(c.active_native_leases), labels);
    w.counter("dgpp_rank_expert_coalesced_reads_total", "Read requests sharing an existing file ticket.", q.coalesced, labels);
    w.counter("dgpp_rank_expert_promotions_total", "Experts with an increased committed resident prefix.", c.promotions, labels);
    w.counter("dgpp_rank_expert_demotions_total", "Experts with a decreased committed resident prefix.", c.demotions, labels);
    w.counter("dgpp_rank_expert_evictions_total", "Experts demoted from nonzero to zero residency.", c.evictions, labels);
    w.counter("dgpp_rank_expert_promotion_bytes_total", "Aligned cache slot bytes newly committed by adaptation or demand admission.", c.promotion_bytes, labels);
    w.counter("dgpp_rank_expert_demand_admissions_total", "Executions that grew their expert's persistent residency directly from demand reads; a subset of promotions.", c.demand_admissions, labels);
    w.counter("dgpp_rank_expert_demand_admission_bytes_total", "Aligned slot bytes committed from demand staging without rereading storage; included in promotion bytes.", c.demand_admission_bytes, labels);
    w.counter("dgpp_rank_expert_demotion_bytes_total", "Aligned cache slot bytes released by adaptation.", c.demotion_bytes, labels);
    w.counter("dgpp_rank_expert_backpressure_total", "Execution-slot or demand-queue capacity refusals.", c.backpressure, labels);
    w.counter("dgpp_rank_expert_adaptations_deferred_total", "Background adaptation calls deferred by sampled memory or block-device pressure.", c.adaptations_deferred, labels);
    w.counter("dgpp_rank_expert_prefetches_deferred_total", "New prefetch calls deferred by sampled memory or block-device pressure; demand remains enabled.", c.prefetches_deferred, labels);
    w.counter("dgpp_rank_expert_resource_samples_total", "Resource monitor sample sequence; zero when no sampler is installed.", x.pressure.sample, labels);
    w.gauge("dgpp_rank_expert_memory_pressure", "Sampled node MemAvailable hysteresis gate; does not imply slab reclamation.", int64_t{x.pressure.memory_pressure ? 1 : 0}, labels);
    w.gauge("dgpp_rank_expert_storage_pressure", "Sampled image block-device busy-time hysteresis gate; includes unrelated I/O.", int64_t{x.pressure.storage_pressure ? 1 : 0}, labels);
    w.gauge("dgpp_rank_expert_memory_signal_known", "Whether the latest sampled node MemAvailable is known.", int64_t{x.pressure.available_bytes ? 1 : 0}, labels);
    w.gauge("dgpp_rank_expert_storage_signal_known", "Whether the latest device busy-time fraction is known; network filesystems may be unknown.", int64_t{x.pressure.storage_busy_fraction ? 1 : 0}, labels);
    if (x.pressure.available_bytes)
      w.gauge("dgpp_rank_expert_node_available_bytes", "Latest sampled node MemAvailable, including all consumers.", static_cast<double>(*x.pressure.available_bytes), labels);
    if (x.pressure.storage_busy_fraction)
      w.gauge("dgpp_rank_expert_storage_busy_fraction", "Latest sampled block-device I/O active milliseconds divided by elapsed milliseconds; not bandwidth or expert-only attribution.", *x.pressure.storage_busy_fraction, labels);
    w.gauge("dgpp_rank_expert_budget_bytes", "Complete per-rank expert-weight budget, staging and execution included.", static_cast<double>(x.memory.budget_bytes), labels);
    w.gauge("dgpp_rank_expert_allocated_bytes", "Preallocated expert cache, execution and aligned pinned staging slabs.", static_cast<double>(x.memory.allocated_bytes()), labels);
    w.gauge("dgpp_rank_expert_resident_bytes", "Aligned slot bytes of committed persistent weights.", static_cast<double>(x.resident_bytes), labels);
    w.gauge("dgpp_rank_expert_resident_chunks", "Chunks in the committed resident population.", static_cast<double>(x.resident_chunks), labels);
    w.gauge("dgpp_rank_expert_chunks_total", "Complete main and enabled draft expert chunk population.", static_cast<double>(x.total_chunks), labels);
    w.gauge("dgpp_rank_expert_staging_in_use_bytes", "Aligned staging bytes held by workers or read consumers.", static_cast<double>(q.staging_in_use), labels);
    w.gauge("dgpp_rank_expert_staging_peak_bytes", "Maximum aligned staging bytes held concurrently since startup.", static_cast<double>(q.staging_peak), labels);
    w.gauge("dgpp_rank_expert_pending_reads", "Queued and active expert-image reads.", static_cast<double>(q.pending), labels);
    w.gauge("dgpp_rank_expert_poisoned", "Cache terminal failure; incomplete weights are never executable.", int64_t{c.poisoned ? 1 : 0}, labels);
  }
  if (collectives != nullptr)
    w.counter("dgpp_rank_collectives_total", "Multi-rank collectives this rank completed.",
              collectives->load(std::memory_order_relaxed), rank);
}

RankMetricsServer::RankMetricsServer(uint16_t port, const std::string& bind_host, RankIdentity id,
                                     const std::atomic<uint64_t>* collectives)
    : id_(std::move(id)), collectives_(collectives),
      http_(port, this, /*max_connections=*/8, metrics_bind_address(bind_host)) {
  loop_ = std::thread([this] { http_.serve(); });
}

RankMetricsServer::~RankMetricsServer() {
  http_.stop();
  if (loop_.joinable()) loop_.join();
}

void RankMetricsServer::publish(const dgpp::sched::Scheduler::Meters& m, int64_t ticks) {
  std::lock_guard<std::mutex> lock(mutex_);
  meters_ = m;
  ticks_ = ticks;
  published_ = std::chrono::steady_clock::now();
}

void RankMetricsServer::handle(const HttpRequest& req, HttpResponseWriter& w) {
  if (req.path == "/health") {
    w.respond(200, "application/json", "{\"status\":\"ok\",\"rank\":" + std::to_string(id_.rank) + "}");
    return;
  }
  if (req.path != "/metrics/prometheus" && req.path != "/metrics") {
    w.respond(404, "text/plain; charset=utf-8", "rank metrics: GET /metrics/prometheus\n");
    return;
  }
  if (req.method != "GET") {
    w.respond(405, "text/plain; charset=utf-8", "use GET for metrics\n");
    return;
  }
  dgpp::sched::Scheduler::Meters m;
  int64_t ticks = 0;
  double age_s = 0;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    m = meters_;
    ticks = ticks_;
    age_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - published_).count();
  }
  prom::Writer out("");
  write_rank_metrics(out, id_, m, age_s, ticks, collectives_);
  w.respond(200, "text/plain; version=0.0.4; charset=utf-8", out.take());
}

}  // namespace dgpp::serve
