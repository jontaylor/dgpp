// dsv41_forward_check: the engine's per-layer residual streams and collapse
// coefficients over a prompt on the REAL DeepSeek-V4.1-Flash checkpoint
// (default world 1, the streaming loader: each layer read from the NVMe as the walk
// reaches it), dumped for the independent torch reference
// (tools/dsv41_torch_reference.py: the release's own model.py layer code)
// — docs/deepseek_v41_flash_plan.md G4/G5, the cross-check on the real
// weights.
//
//   dsv41_forward_check --model ORG/NAME | --checkpoint-dir DIR --out FILE
//                       (--ids 1,2,3 | --text FILE [--no-bos] | --tokens N [--seed S])
//                       [--layers N] [--bounded]
//                       [--expert-streaming-json FILE] [--logits-out FILE]
//                       [--world 2|4 --rank R --peer RANK0_IP --fabric-port P]
//                       [--rendezvous-timeout-ms MS]
// The optional file contains the engine.expert_streaming object. Without it
// this retains the whole-layer streaming reference, which fits on one GB10.
// TP worlds use the same boundary reducer for the reference and both policies;
// every rank writes its own state and vocabulary-slice logits files.
//
// The dump: "DSV41ST3", int32 L T H4, bf16 states [L][T][H4] (every walked
// layer's output streams, [4, H] per row), fp32 pre [L][T][4] (the collapse
// coefficients the NEXT layer's attention site uses — the layer's ffn_pre),
// int64 ids [T], then int32 K and the routed expert ids [L][T][K] (ascending),
// int32 Li and ms and the index sources' selections [Li][T][ms] (-1 padded;
// the sources among the walked layers). With every layer walked the last
// row's top-8 logits print.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "common/log.hpp"
#include "engine/tp_bus.hpp"
#include "loaders/architecture.hpp"
#include "loaders/hf_cache.hpp"
#include "models/dsv41/config.hpp"
#include "models/dsv41/model.hpp"
#include "serve/expert_paging.hpp"
#include "text/tokenizer.hpp"

int main(int argc, char** argv) {
  std::string model_id, ckpt, out, ids_text, text_path, paging_path, logits_out, peer;
  int tokens = 0, layers = 0;
  int rank = 0, world = 1, fabric_port = 29992, rendezvous_timeout_ms = 3600000;
  uint64_t seed = 7;
  bool bos = true, bounded = false;
  const auto next = [&](int& i) {
    if (i + 1 >= argc) throw std::runtime_error("missing value after " + std::string(argv[i]));
    return argv[++i];
  };
  try {
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "--model") model_id = next(i);
      else if (a == "--checkpoint-dir") ckpt = next(i);
      else if (a == "--out") out = next(i);
      else if (a == "--expert-streaming-json") paging_path = next(i);
      else if (a == "--logits-out") logits_out = next(i);
      else if (a == "--ids") ids_text = next(i);
      else if (a == "--text") text_path = next(i);
      else if (a == "--tokens") tokens = std::stoi(next(i));
      else if (a == "--seed") seed = std::stoull(next(i));
      else if (a == "--layers") layers = std::stoi(next(i));
      else if (a == "--rank") rank = std::stoi(next(i));
      else if (a == "--world") world = std::stoi(next(i));
      else if (a == "--peer") peer = next(i);
      else if (a == "--fabric-port") fabric_port = std::stoi(next(i));
      else if (a == "--rendezvous-timeout-ms") rendezvous_timeout_ms = std::stoi(next(i));
      else if (a == "--no-bos") bos = false;
      else if (a == "--bounded") bounded = true;
      else throw std::runtime_error("unknown argument " + a);
    }
    if ((world != 1 && world != 2 && world != 4) || rank < 0 || rank >= world ||
        fabric_port < 1 || fabric_port > 65535 || rendezvous_timeout_ms < 1 ||
        (rank != 0 && peer.empty()))
      throw std::runtime_error("invalid TP rank/world, peer, fabric port or rendezvous timeout");
    if (ckpt.empty()) {
      if (model_id.empty()) throw std::runtime_error("--model or --checkpoint-dir is required");
      std::string err;
      ckpt = dgpp::hf::model_dir(model_id, &err);
      if (ckpt.empty()) throw std::runtime_error("cannot resolve " + model_id + ": " + err);
    }
    if (out.empty()) throw std::runtime_error("--out is required");
    const std::string cfg_path = (std::filesystem::path(ckpt) / "config.json").string();
    if (dgpp::detect_architecture_file(cfg_path) != dgpp::ModelArchitecture::DeepseekV41)
      throw std::runtime_error("not a DeepseekV41 checkpoint: " + ckpt);
    const dgpp::Dsv41TextConfig cfg = dgpp::Dsv41TextConfig::from_json_file(cfg_path);
    std::optional<dgpp::Dsv41ExpertPaging> paging;
    if (!paging_path.empty()) {
      std::ifstream input(paging_path);
      if (!input) throw std::runtime_error("cannot open " + paging_path);
      std::stringstream contents;
      contents << input.rdbuf();
      const auto text = contents.str();  // minijson borrows its string storage
      const auto json = dgpp::minijson::parse(text);
      paging = dgpp::serve::parse_expert_paging(json.root, paging_path);
    }
    // The prompt: explicit ids, a text through the checkpoint's tokenizer
    // (BOS first unless --no-bos), or random ids.
    std::vector<int64_t> ids;
    if (!ids_text.empty()) {
      std::stringstream ss(ids_text);
      std::string item;
      while (std::getline(ss, item, ',')) ids.push_back(std::stoll(item));
    } else if (!text_path.empty()) {
      std::ifstream f(text_path);
      if (!f) throw std::runtime_error("cannot open " + text_path);
      std::stringstream buf;
      buf << f.rdbuf();
      const dgpp::text::Tokenizer tok =
          dgpp::text::Tokenizer::load((std::filesystem::path(ckpt) / "tokenizer.json").string());
      if (bos) {
        int64_t bos_id = -1;
        for (const auto& added : tok.added_tokens())
          if (added.content == "<｜begin▁of▁sentence｜>") bos_id = added.id;
        if (bos_id < 0) throw std::runtime_error("the tokenizer has no BOS token");
        ids.push_back(bos_id);
      }
      const std::vector<int64_t> body = tok.encode(buf.str());
      ids.insert(ids.end(), body.begin(), body.end());
      if (tokens > 0 && static_cast<int>(ids.size()) > tokens) ids.resize(static_cast<size_t>(tokens));
    } else {
      if (tokens <= 0) throw std::runtime_error("--ids, --text or --tokens is required");
      std::mt19937_64 rng(seed);
      for (int i = 0; i < tokens; ++i) ids.push_back(static_cast<int64_t>(rng() % static_cast<uint64_t>(cfg.vocab_size)));
    }
    const int T = static_cast<int>(ids.size());
    if (T < 1) throw std::runtime_error("an empty prompt");
    DGPP_LOG_INFO("dsv41_forward_check: {} tokens{}, {} layers, {} prefill", T, bounded ? "" : "",
                  layers > 0 ? layers : cfg.num_hidden_layers, bounded ? "bounded" : "exact");
    const auto t0 = std::chrono::steady_clock::now();
    // Form the same canonical-rank boundary reducer for the streaming control
    // and both paging policies. Each rank writes its own vocab slice; compare
    // matching slices, then require complete nonoverlapping vocab coverage.
    std::unique_ptr<dgpp::net::CollectiveBus> bus;
    std::unique_ptr<dgpp::BusBoundaryReducer> reducer;
    if (world > 1) {
      dgpp::net::BusOptions options;
      options.world_size = world;
      options.my_rank = rank;
      options.rendezvous_host = rank == 0 ? "" : peer;
      options.rendezvous_port = static_cast<uint16_t>(fabric_port);
      options.rendezvous_timeout_ms = rendezvous_timeout_ms;
      options.lat_slots = 8;
      options.lat_slot_bytes = static_cast<size_t>(16) * (cfg.hc_mult + 1) * cfg.hidden_size * 2;
      options.bulk_slots = 8;
      options.bulk_slot_bytes = 262144;
      options.qp_depth = 1024;
      options.completion_timeout_ms = rendezvous_timeout_ms;
      options.consumer_deadline_s = rendezvous_timeout_ms / 1000.0;
      options.launch_consumers = false;
      bus = std::make_unique<dgpp::net::CollectiveBus>(options);
      std::string error;
      if (!bus->start(&error)) throw std::runtime_error("TP bus start: " + error);
      reducer = std::make_unique<dgpp::BusBoundaryReducer>(*bus, rendezvous_timeout_ms);
    }
    dgpp::Dsv41Model model(cfg, ckpt, /*max_tokens=*/std::max({T, cfg.sliding_window, 16}), /*max_cache_tokens=*/std::max(T + 256, 1024),
                           paging ? dgpp::Dsv41Residency::Resident : dgpp::Dsv41Residency::Streaming,
                           reducer.get(), rank, world, 1, false, 0, false, paging ? &*paging : nullptr);
    model.set_prefill_bounded(bounded);
    if (layers > 0) model.set_debug_layer_limit(layers);
    const dgpp::Dsv41Model::Outputs o = model.forward(ids, /*capture_layers=*/true);
    const std::vector<std::vector<float>> pre = model.debug_layer_pre();
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const int L = static_cast<int>(o.layer_states.size());
    const int H4 = cfg.hc_mult * cfg.hidden_size;
    if (L < 1 || static_cast<int>(pre.size()) != L) throw std::runtime_error("no layer captures");
    std::ofstream f(out, std::ios::binary);
    if (!f) throw std::runtime_error("cannot write " + out);
    f.write("DSV41ST3", 8);
    const int32_t hdr[3] = {L, T, H4};
    f.write(reinterpret_cast<const char*>(hdr), 12);
    for (const auto& st : o.layer_states) {
      if (st.size() != static_cast<size_t>(T) * H4) throw std::runtime_error("a layer capture with the wrong rows (bounded?)");
      f.write(reinterpret_cast<const char*>(st.data()), static_cast<std::streamsize>(st.size() * 2));
    }
    for (const auto& p : pre) {
      if (p.size() != static_cast<size_t>(T) * 4) throw std::runtime_error("a pre capture with the wrong rows");
      f.write(reinterpret_cast<const char*>(p.data()), static_cast<std::streamsize>(p.size() * 4));
    }
    f.write(reinterpret_cast<const char*>(ids.data()), static_cast<std::streamsize>(ids.size() * 8));
    const int32_t K = static_cast<int32_t>(cfg.num_experts_per_tok);
    f.write(reinterpret_cast<const char*>(&K), 4);
    if (static_cast<int>(o.route_ids.size()) < L) throw std::runtime_error("route captures");
    for (int l = 0; l < L; ++l) {
      if (o.route_ids[static_cast<size_t>(l)].size() != static_cast<size_t>(T) * K) throw std::runtime_error("route rows");
      f.write(reinterpret_cast<const char*>(o.route_ids[static_cast<size_t>(l)].data()), static_cast<std::streamsize>(T) * K * 4);
    }
    const int32_t Li = static_cast<int32_t>(o.dsa_selections.size());
    const int32_t ms = static_cast<int32_t>(cfg.index_topk);
    f.write(reinterpret_cast<const char*>(&Li), 4);
    f.write(reinterpret_cast<const char*>(&ms), 4);
    for (const auto& sel : o.dsa_selections) {
      if (sel.size() != static_cast<size_t>(T) * ms) throw std::runtime_error("selection rows");
      f.write(reinterpret_cast<const char*>(sel.data()), static_cast<std::streamsize>(sel.size() * 4));
    }
    f.close();
    if (!f) throw std::runtime_error("failed writing " + out);
    if (!logits_out.empty()) {
      const int V = o.lm_vocab_count;
      if (V <= 0 || o.logits.empty() || o.logits.size() % static_cast<size_t>(V))
        throw std::runtime_error("logits output requires a complete forward with head rows");
      std::ofstream logits(logits_out, std::ios::binary);
      if (!logits) throw std::runtime_error("cannot write " + logits_out);
      logits.write("DSV41LG1", 8);
      const int32_t shape[3] = {static_cast<int32_t>(o.logits.size() / V),
                                o.lm_vocab_begin, o.lm_vocab_count};
      logits.write(reinterpret_cast<const char*>(shape), sizeof(shape));
      logits.write(reinterpret_cast<const char*>(o.logits.data()),
                   static_cast<std::streamsize>(o.logits.size() * sizeof(float)));
      logits.close();
      if (!logits) throw std::runtime_error("failed writing " + logits_out);
    }
    DGPP_LOG_INFO("dsv41_forward_check: {} layers x {} rows x {} (+ routes, {} index sources) written to {} in {:.1f} s", L, T, H4,
                  Li, out, secs);
    if (layers <= 0 && !o.logits.empty()) {
      const int V = o.lm_vocab_count;
      const float* last = o.logits.data() + o.logits.size() - static_cast<size_t>(V);
      std::vector<int> order(static_cast<size_t>(V));
      for (int i = 0; i < V; ++i) order[static_cast<size_t>(i)] = i;
      std::partial_sort(order.begin(), order.begin() + 8, order.end(), [&](int a, int b) { return last[a] > last[b]; });
      std::string s;
      for (int i = 0; i < 8; ++i) s += std::format("{}{}:{:.3f}", i ? " " : "", o.lm_vocab_begin + order[static_cast<size_t>(i)], last[order[static_cast<size_t>(i)]]);
      DGPP_LOG_INFO("dsv41_forward_check: the last row's top-8 logits (id:value): {}", s);
    }
    return 0;
  } catch (const std::exception& e) {
    DGPP_LOG_ERROR("dsv41_forward_check: {}", e.what());
    return 1;
  }
}
