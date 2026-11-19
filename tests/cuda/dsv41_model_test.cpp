// The DeepSeek-V4.1 model on the loader fixture at world 1 (plan G5's
// first gates): the cold forward runs every layer (the residual streams,
// the index sources' selections captured); a session prefill's last row
// is bitwise the forward's; decode steps and a verify batch with a
// rollback match a re-forward over the same tokens within the decode
// budget (the GEMV chain against the tile kernels).
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_check.hpp"
#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "dsv41_fixture.hpp"
#include "engine/eager_engine.hpp"
#include "kda_test_helpers.hpp"
#include "models/dsv41/model.hpp"

namespace {
namespace fs = std::filesystem;
using dgpp::bf16_bits_to_float;

void require(bool cond, const std::string& what) {
  if (!cond) throw std::runtime_error(what);
}

struct Fixture {
  dgpp::Dsv41TextConfig cfg;
  std::string dir;
};
Fixture make_fixture(bool nvfp4 = false) {
  Fixture fx;
  fx.cfg = dsv41fx::tiny_config();
  fx.cfg.experts_nvfp4 = nvfp4;
  fx.cfg.engram_mxfp4 = nvfp4;
  fx.dir = (fs::current_path() / (nvfp4 ? "dsv41_model_nvfp4_fixture" : "dsv41_model_fixture")).string();
  dsv41fx::write_fixture(fx.cfg, fx.dir);
  (void)dsv41fx::fixture_sidecar(fx.cfg, fx.dir);
  return fx;
}
std::vector<int64_t> tokens_of(uint64_t seed, int n, int vocab) {
  std::vector<int64_t> t(static_cast<size_t>(n));
  uint64_t s = seed | 1;
  for (int i = 0; i < n; ++i) {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    t[static_cast<size_t>(i)] = static_cast<int64_t>(s % static_cast<uint64_t>(vocab));
  }
  return t;
}
// Relative l2 between two logits rows and the argmax agreement.
double rel_l2(const float* a, const float* b, size_t n) {
  double d = 0, m = 0;
  for (size_t i = 0; i < n; ++i) { d += double(a[i] - b[i]) * (a[i] - b[i]); m += double(b[i]) * b[i]; }
  return std::sqrt(d) / std::max(std::sqrt(m), 1e-30);
}
int argmax(const float* a, size_t n) {
  int best = 0;
  for (size_t i = 1; i < n; ++i) if (a[i] > a[size_t(best)]) best = int(i);
  return best;
}
// Whether two forwards of the same prompt selected different compressed
// entries for row `row` on any index source: a certified selection flip
// (the two forms' rounding tipped a near-tie), past which the row's logits
// are compared as the decode audit compares them — the move is the flip's.
bool selection_flipped(const dgpp::Dsv41Model::Outputs& a, int row_a, const dgpp::Dsv41Model::Outputs& b, int row_b,
                       int topk, std::string* where) {
  bool flipped = false;
  if (a.dsa_selections.size() != b.dsa_selections.size()) throw std::runtime_error("selection captures differ in sources");
  for (size_t s = 0; s < a.dsa_selections.size(); ++s) {
    const size_t off_a = size_t(row_a) * size_t(topk), off_b = size_t(row_b) * size_t(topk);
    if (a.dsa_selections[s].size() < off_a + size_t(topk) || b.dsa_selections[s].size() < off_b + size_t(topk))
      throw std::runtime_error("selection captures do not cover the compared rows");
    if (std::memcmp(a.dsa_selections[s].data() + off_a, b.dsa_selections[s].data() + off_b, size_t(topk) * 4) != 0) {
      flipped = true;
      if (where) *where += (where->empty() ? "source " : ", source ") + std::to_string(s);
    }
  }
  return flipped;
}
}  // namespace

// The group prefill (session_prefill_group): three cold prompts as the
// spans of one walk against each prefilled alone — the last rows' logits
// bitwise (the dense sites and the MoE tile kernel are row-invariant, the
// attention and the publication run per span, the Engram takes the
// spans), and the first decode step of each request bitwise the alone
// model's (the draft state, positions and rings per request). The exact
// prefill mode: the span limit is the walk's rows.
DGPP_TEST(dsv41_model_group_prefill_is_bitwise_the_prefills_alone) {
  const Fixture fx = make_fixture();
  const int V = fx.cfg.vocab_size;
  const std::vector<std::vector<int64_t>> prompts = {tokens_of(0xA1, 23, V), tokens_of(0xA2, 17, V), tokens_of(0xA3, 11, V)};
  std::vector<std::vector<float>> alone_logits, alone_step;
  {
    dgpp::Dsv41Model m(fx.cfg, fx.dir, 96, 512, dgpp::Dsv41Residency::Streaming, nullptr, 0, 1, 3, /*mtp=*/true, 8);
    for (int r = 0; r < 3; ++r) {
      const auto o = m.session_prefill(r, prompts[static_cast<size_t>(r)]);
      alone_logits.push_back(o.logits);
      const int32_t first = argmax(o.logits.data(), size_t(V));
      const auto st = m.session_step(r, first);
      alone_step.push_back(st.logits);
    }
  }
  {
    dgpp::Dsv41Model m(fx.cfg, fx.dir, 96, 512, dgpp::Dsv41Residency::Streaming, nullptr, 0, 1, 3, /*mtp=*/true, 8);
    require(m.prefill_group_span_limit() >= 23, "the exact prefill's span limit is the walk's rows");
    const std::vector<const std::vector<int64_t>*> pp = {&prompts[0], &prompts[1], &prompts[2]};
    const auto outs = m.session_prefill_group({2, 0, 1}, {pp[2], pp[0], pp[1]});  // a permuted slot order
    require(outs.size() == 3, "one output per request");
    const int order[3] = {2, 0, 1};
    for (int i = 0; i < 3; ++i) {
      const int r = order[i];
      require(outs[static_cast<size_t>(i)].logits.size() == size_t(V), "a last-row logits vector per request");
      require(outs[static_cast<size_t>(i)].logits == alone_logits[static_cast<size_t>(r)],
              "group prefill logits bitwise the prefill alone (request " + std::to_string(r) + ")");
    }
    for (int i = 0; i < 3; ++i) {
      const int r = order[i];
      const int32_t first = argmax(alone_logits[static_cast<size_t>(r)].data(), size_t(V));
      const auto st = m.session_step(r, first);
      require(st.logits == alone_step[static_cast<size_t>(r)],
              "the first step after a group prefill bitwise the alone model's (request " + std::to_string(r) + ")");
    }
    std::printf("[ OK ] group prefill of 23 + 17 + 11 rows: last rows and first steps bitwise the prefills alone\n");
  }
}

// The same under the bounded prefill with spans wider than the window
// (16 in the fixture): each span's decoder segment is its last window
// rows, packed; the last rows' logits and the first steps bitwise the
// bounded prefills alone.
DGPP_TEST(dsv41_model_group_prefill_bounded_wide_spans_is_bitwise_the_prefills_alone) {
  const Fixture fx = make_fixture();
  const int V = fx.cfg.vocab_size;
  require(fx.cfg.sliding_window < 23, "the fixture's window is narrower than the widest prompt");
  const std::vector<std::vector<int64_t>> prompts = {tokens_of(0xB1, 23, V), tokens_of(0xB2, 9, V), tokens_of(0xB3, 40, V)};
  std::vector<std::vector<float>> alone_logits, alone_step;
  {
    dgpp::Dsv41Model m(fx.cfg, fx.dir, 96, 512, dgpp::Dsv41Residency::Streaming, nullptr, 0, 1, 3, /*mtp=*/true, 8);
    m.set_prefill_bounded(true);
    for (int r = 0; r < 3; ++r) {
      const auto o = m.session_prefill(r, prompts[static_cast<size_t>(r)]);
      alone_logits.push_back(o.logits);
      const int32_t first = argmax(o.logits.data(), size_t(V));
      alone_step.push_back(m.session_step(r, first).logits);
    }
  }
  {
    dgpp::Dsv41Model m(fx.cfg, fx.dir, 96, 512, dgpp::Dsv41Residency::Streaming, nullptr, 0, 1, 3, /*mtp=*/true, 8);
    m.set_prefill_bounded(true);
    const std::vector<const std::vector<int64_t>*> pp = {&prompts[0], &prompts[1], &prompts[2]};
    const auto outs = m.session_prefill_group({0, 1, 2}, pp);
    for (int r = 0; r < 3; ++r)
      require(outs[static_cast<size_t>(r)].logits == alone_logits[static_cast<size_t>(r)],
              "bounded group prefill logits bitwise the prefill alone (request " + std::to_string(r) + ")");
    for (int r = 0; r < 3; ++r) {
      const int32_t first = argmax(alone_logits[static_cast<size_t>(r)].data(), size_t(V));
      require(m.session_step(r, first).logits == alone_step[static_cast<size_t>(r)],
              "the first step after a bounded group prefill bitwise the alone model's (request " + std::to_string(r) + ")");
    }
    std::printf("[ OK ] bounded group prefill of 23 + 9 + 40 rows (window %d): bitwise the prefills alone\n", fx.cfg.sliding_window);
  }
}

DGPP_TEST(dsv41_model_forward_prefill_decode_and_rollback_at_world_1) {
  const Fixture fx = make_fixture();
  const int V = fx.cfg.vocab_size;
  dgpp::Dsv41Model model(fx.cfg, fx.dir, /*max_tokens=*/96, /*max_cache_tokens=*/512, dgpp::Dsv41Residency::Streaming,
                         nullptr, 0, 1, /*max_requests=*/2, /*mtp=*/false, /*decode_rows=*/8);
  // This gate audits the decode path against the prefill walk's chain, so
  // the model under test prefills on the decode rows' dense form (the
  // streaming form); the prefill walk's default tile form is gated against
  // it in 1b (engine.prefill_dense, 2026-10-18). On this random fixture the
  // tile-built index keys tip a near-tie selection against the streaming
  // decode query (a certified flip, dsv41_decode_test's regime, ~0.37 in the
  // logits with the argmax unchanged), which this gate's budget predates.
  model.set_prefill_dense_tile(false);
  // Likewise the window attention: the decode window runs the scalar
  // partial kernel, so the audited walk does too; the default flash form
  // is gated against it in 1c (engine.prefill_window, 2026-10-18).
  model.set_prefill_window_flash(false);
  // 1) the cold forward over 70 tokens (past the 16-entry select horizon at
  // ratio 2 and the candidate pool at ratio 1).
  const auto prompt = tokens_of(0xD5, 70, V);
  auto fwd = model.forward(prompt, /*capture_layers=*/true);
  const std::vector<dgpp::Dsv41Model::IndexLogits> fwd_index = model.debug_index_logits();  // before the next walk clears it
  {
    // 1b) the prefill walk's dense form (engine.prefill_dense, 2026-10-18):
    // the default tile GEMM (tile_model) against the streaming decode form
    // (model, above) over the same prompt. The two place the same bf16(e4m3 x scale) weights and
    // accumulate in fp32 in different orders: every layer within rounding
    // (the window-split budget's order), the last row's argmax equal, its
    // logits within a certified flip's move when a near-tie selection tips.
    dgpp::Dsv41Model tile_model(fx.cfg, fx.dir, 96, 512, dgpp::Dsv41Residency::Streaming, nullptr, 0, 1, 2, false, 8);
    tile_model.set_prefill_dense_tile(true);
    const auto sfwd = tile_model.forward(prompt, /*capture_layers=*/true);  // the tile form; fwd is the streaming form
    require(sfwd.layer_states.size() == fwd.layer_states.size() && !fwd.layer_states.empty(), "both forms capture every layer");
    const int rows_cmp = static_cast<int>(prompt.size());
    const int H4 = 4 * fx.cfg.hidden_size;
    for (size_t l = 0; l < fwd.layer_states.size(); ++l) {
      double d = 0, m = 0;
      for (size_t i = 0; i < size_t(rows_cmp) * H4; ++i) {
        const double a = bf16_bits_to_float(fwd.layer_states[l][i]), b = bf16_bits_to_float(sfwd.layer_states[l][i]);
        d += (a - b) * (a - b); m += b * b;
      }
      const double l2 = std::sqrt(d) / std::max(std::sqrt(m), 1e-30);
      std::printf("[INFO] prefill tile vs streaming form, layer %zu: rel l2 %.3e\n", l, l2);  // fwd streaming, sfwd tile
      require(l2 < 0.12, "prefill tile form: layer " + std::to_string(l) + " within a flip's move of the streaming form");
    }
    const float* a = fwd.logits.data() + size_t(rows_cmp - 1) * V;
    const float* b = sfwd.logits.data() + size_t(rows_cmp - 1) * V;
    const double l2 = rel_l2(a, b, size_t(V));
    std::string where;
    const bool flipped = selection_flipped(fwd, rows_cmp - 1, sfwd, rows_cmp - 1, fx.cfg.index_topk, &where);
    std::printf("[INFO] prefill tile vs streaming form, last row logits: rel l2 %.2e, argmax %d / %d, selection %s\n", l2,
                argmax(a, size_t(V)), argmax(b, size_t(V)), flipped ? ("FLIPPED (" + where + ")").c_str() : "equal");
    // Without a selection flip the two forms are rounding apart: the argmax
    // equal, the logits within the budget. With one (the EXL3 fixture's
    // random trellis weights tip a near-tie the fp8 fixture's do not) the
    // row attends to other entries and its logits are the flip's move, as
    // the decode audit below treats a certified flip.
    if (!flipped) {
      require(argmax(a, size_t(V)) == argmax(b, size_t(V)), "prefill tile form: the last row's argmax equals the streaming form's");
      require(l2 < 0.12, "prefill tile form: the last row's logits within a flip's move of the streaming form");
    }
  }
  {
    // 1c) the prefill walk's window attention (engine.prefill_window,
    // 2026-10-18): the default flash listed kernel (win_model) against the
    // scalar partial kernel the audited walk and the decode window run
    // (model, above), both on the streaming dense form. The two sum each
    // row's window scores in another fp32 order: every layer within a
    // flip's move, the last row's argmax equal.
    dgpp::Dsv41Model win_model(fx.cfg, fx.dir, 96, 512, dgpp::Dsv41Residency::Streaming, nullptr, 0, 1, 2, false, 8);
    win_model.set_prefill_dense_tile(false);
    win_model.set_prefill_window_flash(true);
    const auto wfwd = win_model.forward(prompt, /*capture_layers=*/true);
    require(wfwd.layer_states.size() == fwd.layer_states.size() && !fwd.layer_states.empty(), "both window forms capture every layer");
    const int rows_cmp = static_cast<int>(prompt.size());
    const int H4 = 4 * fx.cfg.hidden_size;
    for (size_t l = 0; l < fwd.layer_states.size(); ++l) {
      double d = 0, m = 0;
      for (size_t i = 0; i < size_t(rows_cmp) * H4; ++i) {
        const double a = bf16_bits_to_float(fwd.layer_states[l][i]), b = bf16_bits_to_float(wfwd.layer_states[l][i]);
        d += (a - b) * (a - b); m += b * b;
      }
      const double l2 = std::sqrt(d) / std::max(std::sqrt(m), 1e-30);
      std::printf("[INFO] prefill flash vs scalar window, layer %zu: rel l2 %.3e\n", l, l2);
      require(l2 < 0.12, "prefill flash window: layer " + std::to_string(l) + " within a flip's move of the scalar form");
    }
    const float* a = fwd.logits.data() + size_t(rows_cmp - 1) * V;
    const float* b = wfwd.logits.data() + size_t(rows_cmp - 1) * V;
    const double l2 = rel_l2(a, b, size_t(V));
    std::string where;
    const bool flipped = selection_flipped(fwd, rows_cmp - 1, wfwd, rows_cmp - 1, fx.cfg.index_topk, &where);
    std::printf("[INFO] prefill flash vs scalar window, last row logits: rel l2 %.2e, argmax %d / %d, selection %s\n", l2,
                argmax(a, size_t(V)), argmax(b, size_t(V)), flipped ? ("FLIPPED (" + where + ")").c_str() : "equal");
    if (!flipped) {
      require(argmax(a, size_t(V)) == argmax(b, size_t(V)), "prefill flash window: the last row's argmax equals the scalar form's");
      require(l2 < 0.12, "prefill flash window: the last row's logits within a flip's move of the scalar form");
    }
  }
  require(fwd.logits.size() == size_t(70) * V, "forward logits rows");
  for (const float v : fwd.logits) require(std::isfinite(v), "forward logits finite");
  require(fwd.layer_states.size() == 8, "eight layer states");
  require(fwd.dsa_selections.size() == 3, "three index sources' selections (layers 2, 4, 6)");
  for (const auto& st : fwd.layer_states)
    for (const uint16_t b : st) require(std::isfinite(bf16_bits_to_float(b)), "stream finite");
  // Different rows produce different logits (the attention and the caches
  // actually feed the head).
  require(rel_l2(fwd.logits.data(), fwd.logits.data() + size_t(69) * V, size_t(V)) > 1e-3, "rows differ");

  // 2) prefill == forward bitwise on the last row.
  auto pre = model.session_prefill(0, prompt);
  require(pre.logits.size() == size_t(V), "prefill returns the last row");
  for (int i = 0; i < V; ++i)
    require(pre.logits[size_t(i)] == fwd.logits[size_t(69) * V + i], "prefill's last row bitwise the forward's");

  // The decode path against the prefill path row by row: a 69-token
  // prefill, then the prompt's own last token as a decode step, its layer
  // rows against the forward's row 69. The window-only layer and the
  // Engram layer are bitwise (the same kernels on both paths); the
  // compressed layers differ by the GEMV chain vs the tile kernels only.
  {
    dgpp::Dsv41Model m2(fx.cfg, fx.dir, 96, 512, dgpp::Dsv41Residency::Streaming, nullptr, 0, 1, 1, false, 8);
    m2.set_prefill_window_flash(false);  // the state its step reads comes from the audited window form
    const std::vector<int64_t> p69(prompt.begin(), prompt.begin() + 69);
    (void)m2.session_prefill(0, p69);
    setenv("DGPP_DSV41_CAPTURE_DECODE", "1", 1);
    const auto st = m2.session_step(0, prompt[69]);
    unsetenv("DGPP_DSV41_CAPTURE_DECODE");
    const std::vector<dgpp::Dsv41Model::IndexLogits> dec_index = m2.debug_index_logits();
    const int H4 = 4 * fx.cfg.hidden_size;
    require(st.layer_states.size() == 8, "the decode walk captured every layer");
    // The index sources whose coded query (e4m3 codes + row scales) differs
    // between the decode row and the forward's row 69: from the first such
    // source's layer on, the two paths select from different logits (the
    // coding turns sub-ulp noise into code steps of 1/8 or more), so a
    // selection flip and the layer budget's excess are the coding's, not a
    // decode-path defect — the layers before it must stay within budget.
    require(fwd_index.size() == 3 && dec_index.size() == 3 && st.dsa_selections.size() == 3, "index captures on both walks");
    int first_coded_layer = 99;
    for (size_t s = 0; s < 3; ++s) {
      const auto& a = fwd_index[s]; const auto& b = dec_index[s];
      require(a.q_rows == 70 && b.q_rows == 1 && a.q_heads == b.q_heads && a.q_heads > 0, "coded index queries captured");
      const size_t w = size_t(a.q_heads) * 128;
      int diffs = 0;
      for (size_t i = 0; i < w; ++i) diffs += a.q_codes[size_t(69) * w + i] != b.q_codes[i];
      for (int h = 0; h < a.q_heads; ++h) diffs += a.q_scales[size_t(69) * a.q_heads + h] != b.q_scales[size_t(h)];
      const bool flipped = std::memcmp(st.dsa_selections[s].data(), fwd.dsa_selections[s].data() + size_t(69) * fx.cfg.index_topk,
                                       size_t(fx.cfg.index_topk) * 4) != 0;
      std::printf("[INFO] index source %zu (layer %d): coded query differs in %d of %zu codes/scales, selection %s\n", s,
                  fx.cfg.index_source_layer_ids[s], diffs, w + size_t(a.q_heads), flipped ? "FLIPPED" : "equal");
      if (flipped) require(diffs > 0, "a selection flip between the decode and the prefill row without a coded-query difference");
      if (diffs > 0 && flipped) first_coded_layer = std::min(first_coded_layer, fx.cfg.index_source_layer_ids[s]);
    }
    for (size_t l = 0; l < st.layer_states.size(); ++l) {
      double d = 0, m = 0;
      for (int i = 0; i < H4; ++i) {
        const double a = bf16_bits_to_float(st.layer_states[l][size_t(i)]);
        const double b = bf16_bits_to_float(fwd.layer_states[l][size_t(69) * H4 + i]);
        d += (a - b) * (a - b); m += b * b;
      }
      const double l2 = std::sqrt(d) / std::max(std::sqrt(m), 1e-30);
      std::printf("[INFO] decode vs prefill, layer %zu: rel l2 %.3e\n", l, l2);
      // Layers 0-1 are window-only (no compressed source), so their decode
      // output used to be BITWISE the forward's. Since 2026-09-14 the decode
      // window attention splits its 128 keys across the SMs (csa2_layer:
      // kWinDecodeSplit) — a single unsplit block was latency-bound and left
      // the GPU idle. The split changes the online-softmax combine's fp32
      // order, so the bf16-rounded attention output moves by about one ULP
      // (measured ~3.8e-3 relative). It is no longer bitwise, but a
      // decode-path bug would be orders of magnitude larger; the tight budget
      // still catches that. The decode-vs-re-forward audit (dsv41_decode_test,
      // certified selection flips) is the behavioural gate.
      if (l < 2) require(l2 < 8e-3, "layer " + std::to_string(l) + " (window-only) within the window-split rounding");
      else if (static_cast<int>(l) < first_coded_layer) require(l2 < 0.01, "layer " + std::to_string(l) + " within the decode budget");
      else require(l2 < 0.12, "layer " + std::to_string(l) + " past a coded selection flip within a flip's move");
    }
    // The logits: within the budget before any coded flip; past one, the
    // argmax (the generated token) is the assertion, as in the step loop below.
    if (first_coded_layer > 7)
      require(rel_l2(st.logits.data(), fwd.logits.data() + size_t(69) * V, size_t(V)) < 0.01, "decode logits within the budget");
    else
      require(argmax(st.logits.data(), size_t(V)) == argmax(fwd.logits.data() + size_t(69) * V, size_t(V)),
              "decode argmax matches the prefill's past a coded selection flip");
    // The sublayer captures of layer 0: every stage bitwise (the MoE decode
    // slot chain on the 32 x 32 shared expert grid included — the bug of
    // 2026-09-13 lived there).
    {
      dgpp::Dsv41Model m4(fx.cfg, fx.dir, 96, 512, dgpp::Dsv41Residency::Streaming, nullptr, 0, 1, 1, false, 8);
      m4.set_prefill_window_flash(false);  // the decode window's kernel, as the audited walk
      (void)m4.forward(prompt, true);
      const auto& pf = m4.debug_sites();
      const auto& dec = m2.debug_sites();
      require(!pf.empty() && !dec.empty(), "site captures");
      const int H = fx.cfg.hidden_size;
      auto same = [&](const std::vector<uint16_t>& a, const std::vector<uint16_t>& b, size_t row_elems) {
        for (size_t i = 0; i < row_elems; ++i)
          if (a[i] != b[size_t(69) * row_elems + i]) return false;
        return true;
      };
      // The attention INPUT is before the window attention, so it stays
      // bitwise. From the attention output on, the decode window split (above)
      // moves the bf16 result by ~1 ULP, which propagates into the MoE input
      // and output; those three hold a tight rounding budget instead of
      // bitwise (a real slot-chain regression — the 2026-09-13 bug — is orders
      // of magnitude larger and still trips it).
      auto close = [&](const std::vector<uint16_t>& a, const std::vector<uint16_t>& b, size_t row_elems) {
        double d = 0, m = 0;
        for (size_t i = 0; i < row_elems; ++i) {
          const double x = bf16_bits_to_float(a[i]);
          const double y = bf16_bits_to_float(b[size_t(69) * row_elems + i]);
          d += (x - y) * (x - y);
          m += y * y;
        }
        return std::sqrt(d) / std::max(std::sqrt(m), 1e-30) < 8e-3;
      };
      require(same(dec[0].x_attn, pf[0].x_attn, size_t(H)), "layer 0 attention input bitwise");
      require(close(dec[0].attn_out, pf[0].attn_out, size_t(H)), "layer 0 attention output within the window-split rounding");
      require(close(dec[0].x_ffn, pf[0].x_ffn, size_t(H)), "layer 0 MoE input within the window-split rounding");
      require(close(dec[0].ffn_out, pf[0].ffn_out, size_t(H)), "layer 0 MoE output within the window-split rounding (the decode slot chain)");
    }
    m2.session_close(0);
  }
  // 3) decode steps: each step's logits against a re-forward over the
  // prompt plus the fed tokens (the GEMV chain vs the tile kernels).
  std::vector<int64_t> history = prompt;
  const auto feed = tokens_of(0xD6, 6, V);
  std::vector<std::string> step_failures;
  for (int k = 0; k < 3; ++k) {
    setenv("DGPP_DSV41_CAPTURE_DECODE", "1", 1);  // the step's selections and coded queries (the flip certificate)
    const auto step = model.session_step(0, feed[size_t(k)]);
    unsetenv("DGPP_DSV41_CAPTURE_DECODE");
    const std::vector<dgpp::Dsv41Model::IndexLogits> step_index = model.debug_index_logits();
    history.push_back(feed[size_t(k)]);
    dgpp::Dsv41Model ref(fx.cfg, fx.dir, 96, 512, dgpp::Dsv41Residency::Streaming, nullptr, 0, 1, 1, false, 8);
    // The re-forward on the decode rows' dense form (engine.prefill_dense
    // streaming): the gate compares the decode path against the same chain.
    // The prefill walk's tile form is gated below against this form.
    ref.set_prefill_dense_tile(false);
    ref.set_prefill_window_flash(false);
    const auto again = ref.forward(history, true);
    const std::vector<dgpp::Dsv41Model::IndexLogits> again_index = ref.debug_index_logits();
    const int last = static_cast<int>(history.size()) - 1;
    const float* want = again.logits.data() + size_t(last) * V;
    const double l2 = rel_l2(step.logits.data(), want, size_t(V));
    // A selection flip between the step and the re-forward's last row is
    // certified as in 2): the coded index query differs at a flipped source.
    std::string where;
    const bool flipped = selection_flipped(step, 0, again, last, fx.cfg.index_topk, &where);
    int coded_diffs = 0;
    if (flipped && step_index.size() == again_index.size()) {
      for (size_t s = 0; s < step_index.size(); ++s) {
        const auto& a = step_index[s]; const auto& b = again_index[s];
        if (a.q_rows < 1 || b.q_rows <= last || a.q_heads != b.q_heads) continue;
        const size_t w = size_t(a.q_heads) * 128;
        for (size_t i = 0; i < w; ++i) coded_diffs += a.q_codes[i] != b.q_codes[size_t(last) * w + i];
        for (int h = 0; h < a.q_heads; ++h) coded_diffs += a.q_scales[size_t(h)] != b.q_scales[size_t(last) * a.q_heads + h];
      }
    }
    // The last layer's streams of the step against the re-forward's row:
    // the head's input (the logits' move is the head's when these agree).
    double last_l2 = -1;
    std::string per_layer;
    for (size_t l = 0; l < step.layer_states.size() && l < again.layer_states.size(); ++l) {
      const size_t H4 = step.layer_states[l].size();
      const auto& a = step.layer_states[l];
      const auto& b = again.layer_states[l];
      if (b.size() < size_t(last + 1) * H4) break;
      double d = 0, m = 0;
      for (size_t i = 0; i < H4; ++i) {
        const double x = bf16_bits_to_float(a[i]), y = bf16_bits_to_float(b[size_t(last) * H4 + i]);
        d += (x - y) * (x - y); m += y * y;
      }
      last_l2 = std::sqrt(d) / std::max(std::sqrt(m), 1e-30);
      char buf[64];
      std::snprintf(buf, sizeof(buf), " L%zu:%.3g", l, last_l2);
      per_layer += buf;
    }
    if (!per_layer.empty()) std::printf("[INFO] decode step %d: streams vs the re-forward's row per layer%s\n", k, per_layer.c_str());
    std::printf("[INFO] decode step %d: rel l2 %.2e vs the re-forward (last layer streams %.2e), argmax %d / %d, selection %s\n",
                k, l2, last_l2, argmax(step.logits.data(), size_t(V)), argmax(want, size_t(V)),
                flipped ? ("FLIPPED (" + where + ", coded query differs in " + std::to_string(coded_diffs) + ")").c_str()
                        : "equal");
    // The generated token is the assertion that matters: decode's argmax
    // equals the re-forward's. The logit-vector l2 rides the decode window
    // split's rounding (above): when it tips a near-tie selection the row's
    // logits move like a certified flip, the regime dsv41_decode_test audits
    // with its flip certification — on the EXL3 fixture a flip's move
    // reaches ~0.3 and takes the argmax of a near-tied top pair with it, so
    // a certified flip (the coded query differs) exempts the row as in 2).
    // A real decode-path defect breaks the argmax without a flip, or blows
    // well past this budget.
    // Every step's diagnostics print before the verdict (the trail of a
    // failing step's successors is the evidence).
    if (flipped && coded_diffs <= 0) step_failures.push_back("decode step " + std::to_string(k) + ": a selection flip without a coded-query difference");
    if (!flipped && argmax(step.logits.data(), size_t(V)) != argmax(want, size_t(V)))
      step_failures.push_back("decode step " + std::to_string(k) + " argmax matches the re-forward");
    if (!flipped && !(l2 < 0.12)) step_failures.push_back("decode step " + std::to_string(k) + " within the decode budget");
  }
  require(step_failures.empty(), step_failures.empty() ? "" : step_failures.front());
  // 4) a verify batch of six rows (the DSpark shape: past the GEMV's
  // four-row chunk), one accepted, then the next batch.
  const auto more = tokens_of(0xD9, 3, V);
  const std::vector<int64_t> drafts = {feed[3], feed[4], feed[5], more[0], more[1], more[2]};
  setenv("DGPP_DSV41_CAPTURE_DECODE", "1", 1);
  const auto ver = model.session_verify(0, drafts);
  unsetenv("DGPP_DSV41_CAPTURE_DECODE");
  const std::vector<dgpp::Dsv41Model::IndexLogits> ver_index = model.debug_index_logits();
  require(ver.logits.size() == size_t(6) * V, "verify returns every row");
  // A decode-path row against the re-forward's: the argmax is the
  // assertion; the l2 rides the near-tie rounding (dsv41_decode_test audits
  // the flips) within a wider budget — unless the row's coded index query
  // differs from the re-forward's at a source whose selection differs
  // (the coding's discontinuity: the row then reads a different attention
  // set, and its logits move like a flipped row's, ~0.35 seen). A defect
  // breaks the argmax, or the budget without that evidence.
  const auto check_row = [&](const char* what, int r, const float* got, const float* want,
                             const std::vector<dgpp::Dsv41Model::IndexLogits>& got_index,
                             const std::vector<std::vector<int32_t>>& got_sels, size_t fwd_row,
                             const std::vector<dgpp::Dsv41Model::IndexLogits>& fwd_idx,
                             const std::vector<std::vector<int32_t>>& fwd_sels) {
    const double l2 = rel_l2(got, want, size_t(V));
    int coded_flip = 0;
    for (size_t s = 0; s < fwd_idx.size() && s < got_index.size(); ++s) {
      const auto& a = fwd_idx[s]; const auto& b = got_index[s];
      if (a.q_heads == 0 || a.q_heads != b.q_heads || b.q_rows <= r) continue;
      const size_t w = size_t(a.q_heads) * 128;
      int diffs = 0;
      for (size_t i = 0; i < w; ++i) diffs += a.q_codes[fwd_row * w + i] != b.q_codes[size_t(r) * w + i];
      const size_t ms = size_t(fx.cfg.index_topk);
      const bool flipped = std::memcmp(got_sels[s].data() + size_t(r) * ms, fwd_sels[s].data() + fwd_row * ms, ms * 4) != 0;
      if (flipped && diffs > 0) ++coded_flip;
    }
    std::printf("[INFO] %s %d: rel l2 %.2e vs the re-forward%s, argmax %d / %d\n", what, r, l2,
                coded_flip ? " (a coded selection flip)" : "", argmax(got, size_t(V)), argmax(want, size_t(V)));
    // A coded selection flip moves the row's logits by the flip (up to ~0.4
    // on these fixtures) and, on the EXL3 fixture's near-tied top pairs,
    // the argmax with it: the certified rows are exempt as in 3); without
    // a flip the argmax must match and the logits sit within the budget.
    if (!coded_flip) {
      require(argmax(got, size_t(V)) == argmax(want, size_t(V)), std::string(what) + " " + std::to_string(r) + " argmax matches the re-forward");
      require(l2 < 0.12, std::string(what) + " " + std::to_string(r) + " within the decode budget");
    }
  };
  {
    std::vector<int64_t> h = history;
    h.insert(h.end(), drafts.begin(), drafts.end());
    dgpp::Dsv41Model ref(fx.cfg, fx.dir, 96, 512, dgpp::Dsv41Residency::Streaming, nullptr, 0, 1, 1, false, 8);
    ref.set_prefill_dense_tile(false);
    ref.set_prefill_window_flash(false);
    const auto again = ref.forward(h, true);
    const std::vector<dgpp::Dsv41Model::IndexLogits> again_index = ref.debug_index_logits();
    for (int r = 0; r < 6; ++r)
      check_row("verify row", r, ver.logits.data() + size_t(r) * V, again.logits.data() + (history.size() + r) * V, ver_index,
                ver.dsa_selections, history.size() + r, again_index, again.dsa_selections);
  }
  model.session_rollback(0, 1);
  history.push_back(drafts[0]);
  {
    const auto tok2 = tokens_of(0xD7, 2, V);
    setenv("DGPP_DSV41_CAPTURE_DECODE", "1", 1);
    const auto ver2 = model.session_verify(0, tok2);
    unsetenv("DGPP_DSV41_CAPTURE_DECODE");
    const std::vector<dgpp::Dsv41Model::IndexLogits> ver2_index = model.debug_index_logits();
    std::vector<int64_t> h = history;
    h.insert(h.end(), tok2.begin(), tok2.end());
    dgpp::Dsv41Model ref(fx.cfg, fx.dir, 96, 512, dgpp::Dsv41Residency::Streaming, nullptr, 0, 1, 1, false, 8);
    ref.set_prefill_dense_tile(false);
    ref.set_prefill_window_flash(false);
    const auto again = ref.forward(h, true);
    const std::vector<dgpp::Dsv41Model::IndexLogits> again_index = ref.debug_index_logits();
    for (int r = 0; r < 2; ++r)
      check_row("post-rollback row", r, ver2.logits.data() + size_t(r) * V, again.logits.data() + (history.size() + r) * V,
                ver2_index, ver2.dsa_selections, history.size() + r, again_index, again.dsa_selections);
  }
  model.session_close(0);
}

// The DSpark draft through the eager session API (plan D8): after a
// prefill, the first draft call (the accepted row's main hidden into the
// draft rings, the block over [next, noise x 4], the Markov-biased row 0)
// and four chain rows; then a verify over [next, drafts], a rollback and
// the next draft — every row finite, the block's rows distinct, the
// confidence logits finite; the transcript-level gates are the engine
// test's and the reference parity the forward test's.
DGPP_TEST(dsv41_model_dspark_draft_rows_and_chain_at_world_1) {
  const Fixture fx = make_fixture();
  const int V = fx.cfg.vocab_size;
  const int B = fx.cfg.dspark_block_size;
  require(B == 5, "the fixture's block");
  dgpp::Dsv41Model model(fx.cfg, fx.dir, 96, 512, dgpp::Dsv41Residency::Resident, nullptr, 0, 1, /*max_requests=*/1,
                         /*mtp=*/true, /*decode_rows=*/6);
  const auto prompt = tokens_of(0xD8, 40, V);
  const auto pre = model.session_prefill(0, prompt);
  const int next = argmax(pre.logits.data(), size_t(V));
  std::vector<int32_t> ids{static_cast<int32_t>(next)};
  std::vector<std::vector<float>> rows;
  const auto d0 = model.session_draft(0, {next});
  require(d0.logits.size() == size_t(V), "the draft returns one row");
  rows.push_back(d0.logits);
  for (int i = 0; i < B - 1; ++i) {
    const int tok = argmax(rows.back().data(), size_t(V));
    ids.push_back(tok);
    require(model.session_draft_chain_fits(0, i), "the chain fits the context");
    const auto dc = model.session_draft_chain(0, tok, i, i == 0, i == B - 2);
    require(dc.logits.size() == size_t(V), "a chain row returns one row");
    rows.push_back(dc.logits);
  }
  ids.push_back(argmax(rows.back().data(), size_t(V)));
  for (const auto& r : rows)
    for (const float v : r) require(std::isfinite(v), "draft logits finite");
  for (int i = 1; i < B; ++i) require(rel_l2(rows[size_t(i)].data(), rows[0].data(), size_t(V)) > 1e-4, "block rows differ");
  std::vector<float> conf(static_cast<size_t>(B));
  DGPP_CUDA_OK(cudaMemcpy(conf.data(), model.debug_confidence(), size_t(B) * 4, cudaMemcpyDeviceToHost));
  for (const float c : conf) require(std::isfinite(c), "confidence finite");
  std::printf("[INFO] dspark ids: next %d drafts", next);
  for (size_t i = 1; i < ids.size(); ++i) std::printf(" %d", ids[i]);
  std::printf("; confidence %.3f %.3f %.3f %.3f %.3f\n", conf[0], conf[1], conf[2], conf[3], conf[4]);
  // The verify over [next, d1 .. d5], greedy acceptance, the rollback and
  // the next draft (the accepted rows' main hidden appended to the rings).
  std::vector<int64_t> fed;
  for (const int32_t t : ids) fed.push_back(t);
  const auto ver = model.session_verify(0, fed);
  require(ver.logits.size() == size_t(B + 1) * V, "the verify's rows");
  int accepted = 1;
  while (accepted < B + 1 && argmax(ver.logits.data() + size_t(accepted - 1) * V, size_t(V)) == ids[size_t(accepted)]) ++accepted;
  if (accepted < B + 1) model.session_rollback(0, accepted);
  std::printf("[INFO] verify accepted %d of %d rows\n", accepted, B + 1);
  std::vector<int64_t> acc(fed.begin(), fed.begin() + accepted);
  // The draft consumes the accepted rows' tokens: row i's token is the
  // token AFTER row i (the winner); the last accepted row's is the new next.
  std::vector<int64_t> draft_tokens;
  for (int i = 1; i < accepted; ++i) draft_tokens.push_back(fed[size_t(i)]);
  draft_tokens.push_back(argmax(ver.logits.data() + size_t(accepted - 1) * V, size_t(V)));
  const auto d1 = model.session_draft(0, draft_tokens);
  for (const float v : d1.logits) require(std::isfinite(v), "second draft finite");
  const auto dc1 = model.session_draft_chain(0, argmax(d1.logits.data(), size_t(V)), 0, true, false);
  for (const float v : dc1.logits) require(std::isfinite(v), "second chain finite");
  model.session_close(0);

  // After verifying the same proposed tokens and accepting the same prefix,
  // the next neural draft must see exactly the same DSpark rings.
  (void)model.session_prefill(0, prompt);
  model.session_advance_draft(0, {next});
  const auto copied = model.session_verify(0, fed);
  require(copied.logits == ver.logits, "history proposals preserve target logits");
  if (accepted < B + 1) model.session_rollback(0, accepted);
  const auto resumed = model.session_draft(0, draft_tokens);
  require(resumed.logits == d1.logits, "resumed neural draft sees the same accepted history");
  model.session_close(0);
}

int main() { return dgpp::test::run_all(); }

namespace {
dgpp::Dsv41ExpertPaging fixture_paging(const Fixture& fx, dgpp::expert::Policy policy, bool mtp = false) {
  using namespace dgpp::expert;
  const auto format = fx.cfg.experts_nvfp4 ? Format::kNvfp4 : Format::kMxfp4;
  const auto chunk = partition(fx.cfg.hidden_size, fx.cfg.moe_intermediate_size, 1, 0, 64, format).front();
  const auto full = partition(fx.cfg.hidden_size, fx.cfg.moe_intermediate_size, 1, 0,
                              fx.cfg.moe_intermediate_size, format).front();
  const auto chunks_per_expert = static_cast<size_t>(fx.cfg.moe_intermediate_size) / 64;
  const auto population = static_cast<size_t>(fx.cfg.num_hidden_layers) * fx.cfg.n_routed_experts +
      (mtp ? static_cast<size_t>(fx.cfg.num_nextn_predict_layers) * fx.cfg.dspark_n_routed_experts : 0);
  dgpp::Dsv41ExpertPaging paging;
  paging.image_directory = (fs::path(fx.dir) / "expert-images").string();
  paging.chunk_intermediate = 64;
  paging.adapt_every_routes = 1;
  paging.cache.execution_slots = 1;
  paging.cache.reads.buffers = 2;
  paging.cache.reads.workers = 1;
  paging.cache.reads.max_pending = 4;
  paging.cache.reads.staging_budget_bytes = 2 * chunk.record_bytes;
  paging.cache.planner.policy = policy;
  paging.cache.planner.minimum_fraction = 0.25;
  paging.cache.planner.cooldown_routes = 0;
  paging.cache.planner.budget_bytes = population * chunks_per_expert / 2 * chunk.allocation_bytes +
    full.allocation_bytes + paging.cache.reads.staging_budget_bytes;
  return paging;
}
}

namespace {
void check_paged_model_matches_resident(bool nvfp4) {
  using namespace dgpp::expert;
  const Fixture fx = make_fixture(nvfp4);
  const auto prompt = tokens_of(0xCA43, 23, fx.cfg.vocab_size);
  std::vector<std::vector<float>> expected;
  {
    dgpp::Dsv41Model model(fx.cfg, fx.dir, 96, 512, dgpp::Dsv41Residency::Resident,
                           nullptr, 0, 1, 1, false, 8);
    expected.push_back(model.session_prefill(0, prompt).logits);
    for (int i = 0; i < 3; ++i)
      expected.push_back(model.session_step(0, static_cast<int64_t>(17 + i)).logits);
  }
  for (const auto policy : {Policy::kWholeExpert, Policy::kAdaptiveFractional}) {
    auto paging = fixture_paging(fx, policy);
    const auto plan = dgpp::Dsv41Model::plan_memory(fx.cfg, 96, 512, 0, 1,
        dgpp::Dsv41Residency::Resident, 1, false, 8, false, &paging);
    const auto resident_plan = dgpp::Dsv41Model::plan_memory(fx.cfg, 96, 512, 0, 1,
        dgpp::Dsv41Residency::Resident, 1, false, 8);
    require(plan.total_bytes() < resident_plan.total_bytes(), "plan replaces full expert population with bounded reservation");
    dgpp::Dsv41Model model(fx.cfg, fx.dir, 96, 512, dgpp::Dsv41Residency::Resident,
                           nullptr, 0, 1, 1, false, 8, false, &paging);
    require(model.expert_cache() && !model.prefill_fold_overlap(), "model owns paging cache and uses eager walk");
    model.release_execution_thread();
    std::exception_ptr failure;
    std::thread serving([&] {
      try {
        model.bind_execution_thread();
        require(model.session_prefill(0, prompt).logits == expected[0], "paged prefill logits bitwise the resident model");
        for (int i = 0; i < 3; ++i)
          require(model.session_step(0, static_cast<int64_t>(17 + i)).logits == expected[static_cast<size_t>(i + 1)],
                  "paged decode logits bitwise the resident model after adaptation");
        model.release_execution_thread();
      } catch (...) { failure = std::current_exception(); }
    });
    serving.join();
    if (failure) std::rethrow_exception(failure);
    model.bind_execution_thread();
    const auto* cache = model.expert_cache();
    require(cache->stats().requests && cache->stats().active_execution_slots == 0 &&
            cache->memory().allocated_bytes() <= paging.cache.planner.budget_bytes,
            "real model routes run through bounded execution leases");
  }
}
}  // namespace

DGPP_TEST(dsv41_paged_model_matches_resident_prefill_and_decode_within_the_same_byte_budget) {
  check_paged_model_matches_resident(false);
}

DGPP_TEST(dsv41_paged_nvfp4_model_matches_native_resident_prefill_and_decode) {
  check_paged_model_matches_resident(true);
}

DGPP_TEST(dsv41_paged_model_charges_resident_draft_experts_to_the_budget) {
  for (bool nvfp4 : {false, true}) {
    const Fixture fx = make_fixture(nvfp4);
    auto paging = fixture_paging(fx, dgpp::expert::Policy::kWholeExpert, true);
    auto descriptors = fx.cfg;
    descriptors.runtime_page_experts = true;
    uint64_t draft_bytes = 0;
    for (int layer = fx.cfg.num_hidden_layers; layer < fx.cfg.max_layer(); ++layer)
      draft_bytes += dgpp::Dsv41LayerStream::layer_bytes(fx.cfg, layer) -
          dgpp::Dsv41LayerStream::layer_bytes(descriptors, layer);
    {
      dgpp::Dsv41Model model(fx.cfg, fx.dir, 96, 512, dgpp::Dsv41Residency::Resident,
                            nullptr, 0, 1, 1, true, 8, false, &paging);
      require(model.expert_cache()->memory().budget_bytes + draft_bytes == paging.cache.planner.budget_bytes,
              "resident draft experts and main cache share the configured byte budget");
      for (const auto& expert : model.expert_cache()->experts())
        require(!expert.draft, "both formats reuse the main-only image");
    }
    paging.cache.planner.budget_bytes = draft_bytes;
    bool refused = false;
    try {
      (void)dgpp::Dsv41Model::plan_memory(fx.cfg, 96, 512, 0, 1,
          dgpp::Dsv41Residency::Resident, 1, true, 8, false, &paging);
    } catch (const std::invalid_argument& e) {
      refused = std::string(e.what()).find("budget cannot cover resident DSpark") != std::string::npos;
    }
    require(refused, "insufficient draft budget fails before allocation");
  }
}

DGPP_TEST(dsv41_paged_eager_adapter_batches_plain_decode_and_preserves_slot_state) {
  using namespace dgpp::expert;
  const Fixture fx = make_fixture();
  const std::vector<std::vector<int64_t>> prompts{
      tokens_of(43, 23, fx.cfg.vocab_size), tokens_of(47, 17, fx.cfg.vocab_size)};
  const auto pick = [](const dgpp::DecodeOutputs& out) -> int32_t {
    require(!out.logits.empty(), "batch pick has host logits");
    for (const auto value : out.logits) require(std::isfinite(value), "finite batch logits");
    return out.lm_vocab_begin + static_cast<int32_t>(
        std::max_element(out.logits.begin(), out.logits.end()) - out.logits.begin());
  };
  std::vector<int32_t> opening(2);
  std::vector<std::vector<std::vector<int32_t>>> expected;
  std::vector<std::vector<int>> order;
  int32_t reopened = -1, scalar = -1;
  {
    dgpp::Dsv41Model reference(fx.cfg, fx.dir, 96, 1024, dgpp::Dsv41Residency::Resident,
                               nullptr, 0, 1, 2, false, 8);
    require(reference.eager_plain_decode_batch_capacity() == 1, "resident eager default is preserved");
    for (int slot = 0; slot < 2; ++slot) opening[slot] = pick(reference.session_prefill(slot, prompts[slot]));
    auto pending = opening;
    for (int step = 0; step < 6; ++step) {
      order.push_back(step % 2 ? std::vector<int>{1, 0} : std::vector<int>{0, 1});
      const auto& slots = order.back();
      std::vector<std::vector<int64_t>> feeds{{pending[slots[0]]}, {pending[slots[1]]}};
      std::vector<int> offsets;
      const auto outputs = reference.session_verify_batch(slots, feeds, &offsets);
      std::vector<std::vector<int32_t>> committed;
      for (size_t i = 0; i < slots.size(); ++i) {
        reference.session_rollback(slots[i], 1, 1, offsets[i]);
        pending[slots[i]] = pick(outputs[i]);
        committed.push_back({pending[slots[i]]});
      }
      expected.push_back(std::move(committed));
    }
    reference.session_close(0); reference.session_close(1);
    reopened = pick(reference.session_prefill(0, prompts[1]));
    scalar = pick(reference.session_step(0, reopened));
  }
  for (const auto policy : {Policy::kWholeExpert, Policy::kAdaptiveFractional}) {
    auto paging = fixture_paging(fx, policy);
    dgpp::Dsv41Model model(fx.cfg, fx.dir, 96, 1024, dgpp::Dsv41Residency::Resident,
                           nullptr, 0, 1, 2, false, 8, false, &paging);
    dgpp::EagerEngineAdapter<dgpp::Dsv41Model> engine(&model, 2, pick);
    engine.set_prefill_group(true);
    require(engine.prefill_group_span_limit() == 96, "paging exposes grouped prefill workspace");
    require(engine.decode_batch_capacity() == 2, "paging exposes the native two-row decode walk");
    for (int slot = 0; slot < 2; ++slot)
      require(engine.prefill(slot, prompts[slot]) == opening[slot], "opening token matches resident reference");
    const auto requests = model.expert_cache()->stats().requests;
    for (const auto& invalid : {std::vector<int>{0, 0}, std::vector<int>{0, 2}, std::vector<int>{0, 1, 0}}) {
      bool rejected = false;
      try { (void)engine.step_batch(invalid); }
      catch (const std::invalid_argument&) { rejected = true; }
      require(rejected, "invalid batches are refused before model work");
    }
    require(model.expert_cache()->stats().requests == requests, "invalid batches do not advance cache/model state");
    for (size_t step = 0; step < order.size(); ++step) {
      require(engine.step_batch(order[step]) == expected[step], "batched greedy picks match resident batch oracle");
      for (int slot = 0; slot < 2; ++slot)
        require(model.session_position(slot) == static_cast<int64_t>(prompts[slot].size() + step + 1),
                "each slot advances exactly one position per batch");
    }
    engine.close(0); engine.close(1);
    require(engine.prefill(0, prompts[1]) == reopened, "closed slot can be reused");
    require(engine.step_batch({0}) == std::vector<std::vector<int32_t>>{{scalar}},
            "single-slot batch retains the scalar kernel path");
    engine.close(0);
    require(engine.prefill_group({0, 1}, {&prompts[0], &prompts[1]}) == opening,
            "warm grouped paging retains each request's opening token");
    require(engine.step_batch(order[0]) == expected[0], "warm group retains batched decode state");
    engine.close(0); engine.close(1);
    require(model.expert_cache()->stats().active_execution_slots == 0, "batched execution retires all leases");
  }
}

// Identical operation sequences isolate paging from row-shape and state effects.
DGPP_TEST(dsv41_paged_model_preserves_group_mixed_rollback_prefix_and_slot_reuse) {
  using namespace dgpp::expert;
  const Fixture fx = make_fixture();
  const auto a = tokens_of(0xB17, 23, fx.cfg.vocab_size);
  const auto b = tokens_of(0xB18, 9, fx.cfg.vocab_size);
  const auto long_prompt = tokens_of(0xB19, 256, fx.cfg.vocab_size);
  for (const bool bounded : {false, true}) {
    const auto run = [&](const dgpp::Dsv41ExpertPaging* paging) {
      dgpp::Dsv41Model model(fx.cfg, fx.dir, 256, 2048, dgpp::Dsv41Residency::Resident,
                            nullptr, 0, 1, 3, false, 8, false, paging);
      model.set_prefill_bounded(bounded);
      std::vector<std::vector<float>> transcript;
      const auto save = [&](const auto& outputs) {
        for (const auto& output : outputs) transcript.push_back(output.logits);
      };
      save(model.session_prefill_group({0, 1}, {&a, &b}));
      std::vector<int> offsets;
      save(model.session_verify_batch({0, 1}, {{3, 4, 5}, {6, 7}}, &offsets));
      model.session_rollback(0, 1, 3, offsets[0]);
      model.session_rollback(1, 1, 2, offsets[1]);
      transcript.push_back(model.session_step(0, 11).logits);
      transcript.push_back(model.session_step(1, 12).logits);
      model.session_close(1);
      auto cursor = model.session_prefill_begin(1, long_prompt, 280, 128);
      while (cursor.next != cursor.end) {
        save(model.session_prefill_verify(cursor, 128, {0}, {{13}}));
        model.session_rollback(0, 1, 1);
      }
      transcript.push_back(cursor.output.logits);
      const size_t bytes = model.session_snapshot_bytes();
      void* snapshot = nullptr;
      DGPP_CUDA_OK(cudaMalloc(&snapshot, bytes));
      struct Free { void* p; ~Free() { (void)cudaFree(p); } } cleanup{snapshot};
      const auto meta = model.session_snapshot(1, snapshot);
      model.session_attach(2, snapshot, meta);
      const auto original = model.session_step(1, 17).logits;
      const auto attached = model.session_step(2, 17).logits;
      require(original == attached, "prefix attach preserves the paged model state bitwise");
      transcript.push_back(original);
      transcript.push_back(attached);
      model.session_release_snapshot(meta);
      model.session_close(2);
      transcript.push_back(model.session_prefill(2, b).logits);
      if (paging) require(model.expert_cache()->stats().active_execution_slots == 0,
                          "mixed and prefix operations retire all weight leases");
      return transcript;
    };
    const auto expected = run(nullptr);
    for (const auto policy : {Policy::kWholeExpert, Policy::kAdaptiveFractional}) {
      auto paging = fixture_paging(fx, policy);
      require(run(&paging) == expected,
              "paging is bitwise resident across grouped/mixed prefill, rollback, prefix attach and slot reuse");
    }
  }
}

// Model-level DSpark qualification; serving MTP still requires an engine gate.
void check_paged_dspark(bool nvfp4) {
  using namespace dgpp::expert;
  const Fixture fx = make_fixture(nvfp4);
  const auto prompt = tokens_of(0xD8, 40, fx.cfg.vocab_size);
  const int B = fx.cfg.dspark_block_size, V = fx.cfg.vocab_size;
  const auto run = [&](const dgpp::Dsv41ExpertPaging* paging) {
    dgpp::Dsv41Model model(fx.cfg, fx.dir, 96, 512, dgpp::Dsv41Residency::Resident,
                          nullptr, 0, 1, 1, true, B + 1, false, paging);
    std::vector<std::vector<float>> transcript;
    const auto pre = model.session_prefill(0, prompt);
    transcript.push_back(pre.logits);
    std::vector<int64_t> fed{argmax(pre.logits.data(), static_cast<size_t>(V))};
    auto draft = model.session_draft(0, fed);
    transcript.push_back(draft.logits);
    for (int i = 0; i < B - 1; ++i) {
      const auto token = argmax(draft.logits.data(), static_cast<size_t>(V));
      fed.push_back(token);
      draft = model.session_draft_chain(0, token, i, i == 0, i == B - 2);
      transcript.push_back(draft.logits);
    }
    fed.push_back(argmax(draft.logits.data(), static_cast<size_t>(V)));
    const auto verified = model.session_verify(0, fed);
    transcript.push_back(verified.logits);
    // A fixed acceptance cut exercises rollback independently of random argmax ties.
    model.session_rollback(0, 1, B + 1);
    const auto next = argmax(verified.logits.data(), static_cast<size_t>(V));
    draft = model.session_draft(0, {next});
    transcript.push_back(draft.logits);
    transcript.push_back(model.session_draft_chain(0,
        argmax(draft.logits.data(), static_cast<size_t>(V)), 0, true, false).logits);
    if (paging) {
      bool draft_population = false;
      for (const auto& expert : model.expert_cache()->experts()) draft_population |= expert.draft;
      require(!draft_population && model.expert_cache()->stats().active_execution_slots == 0,
              "both formats keep drafts resident and retire target execution leases");
    }
    model.session_close(0);
    return transcript;
  };
  const auto expected = run(nullptr);
  for (const auto policy : {Policy::kWholeExpert, Policy::kAdaptiveFractional}) {
    auto paging = fixture_paging(fx, policy, true);
    require(run(&paging) == expected, "DSpark main/draft/chain/verify/rollback logits are bitwise resident");
  }
}

DGPP_TEST(dsv41_paged_model_dspark_block_chain_verify_and_rollback_are_bitwise_resident) {
  check_paged_dspark(false);
}
DGPP_TEST(dsv41_paged_nvfp4_model_dspark_block_chain_verify_and_rollback_are_bitwise_resident) {
  check_paged_dspark(true);
}

DGPP_TEST(dsv41_paged_eager_dspark_preserves_tool_and_json_constraints) {
  using namespace dgpp::text;
  const Fixture fx = make_fixture(true);
  const int vocab_size = fx.cfg.vocab_size;
  constexpr int think_open = 480, think_close = 481, dsml = 482, eos = 511;
  std::vector<std::string> texts(static_cast<size_t>(vocab_size), "x");
  for (int b = 0; b < 256; ++b) texts[b] = std::string(1, static_cast<char>(b));
  texts[think_open] = "<think>";
  texts[think_close] = "</think>";
  texts[dsml] = "｜DSML｜";
  texts[eos] = "";
  ChatMarkers markers;
  markers.think_open = {think_open, "<think>"};
  markers.think_close = {think_close, "</think>"};
  markers.dsml = {dsml, "｜DSML｜"};
  GrammarVocab vocab(std::move(texts), markers, {eos}, vocab_size, eos);
  std::vector<GrammarSpec> constraints(2);
  constraints[0].mode = GrammarSpec::Mode::kJson;
  constraints[0].json_schema = R"({"const":{"path":"/home/jon/dgpp","ok":true}})";
  constraints[1].mode = GrammarSpec::Mode::kAuto;
  GrammarTool tool;
  tool.name = "list_directory";
  constraints[1].tools.push_back(tool);
  const std::vector<std::vector<int64_t>> prompts{{65, 66, 67}, {68, 69, 70}};
  auto paging = fixture_paging(fx, dgpp::expert::Policy::kWholeExpert, true);
  dgpp::Dsv41Model model(fx.cfg, fx.dir, 96, 1024, dgpp::Dsv41Residency::Resident,
                        nullptr, 0, 1, 2, true, 16, false, &paging);
  dgpp::EagerEngineAdapter<dgpp::Dsv41Model> engine(&model, 2,
      dgpp::make_w1_pick(vocab_size), dgpp::make_w1_sample(vocab_size), &vocab);
  std::vector<std::vector<int32_t>> expected(2), actual(2);
  for (int req = 0; req < 2; ++req) {
    engine.configure_constraint(req, constraints[req]);
    expected[req].push_back(engine.prefill(req, prompts[req]));
    engine.reserve(req, 128);
    while (expected[req].size() < 80 && expected[req].back() != eos)
      expected[req].push_back(engine.step(req).at(0));
    engine.close(req);
  }
  bool force_first = true;
  engine.configure_eager_mtp(3, [&](const std::vector<dgpp::sample::Candidate>& rows) {
    std::vector<int32_t> picks;
    for (const auto& row : rows) picks.push_back(row.id);
    if (force_first) {
      require(rows.size() == 2, "first draft stacks both constrained requests");
      picks = {think_close, expected[1].at(1)};
      force_first = false;
    }
    return picks;
  });
  for (int req = 0; req < 2; ++req) {
    engine.configure_constraint(req, constraints[req]);
    actual[req].push_back(engine.prefill(req, prompts[req]));
    engine.reserve(req, 128);
  }
  const auto first = engine.step_batch({0, 1});
  require(first[0].size() == 1, "forbidden JSON proposal is rejected without poisoning the live grammar");
  require(first[1].size() >= 2, "tool-enabled prose accepts predictions");
  for (int req = 0; req < 2; ++req)
    actual[req].insert(actual[req].end(), first[req].begin(), first[req].end());
  for (int iteration = 0;; ++iteration) {
    std::vector<int> reqs;
    for (int req : {0, 1}) if (actual[req].size() < expected[req].size()) reqs.push_back(req);
    if (reqs.empty()) break;
    if (iteration % 3 == 0 && reqs.size() > 1) reqs.resize(1);
    const auto fresh = engine.step_batch(reqs);
    for (size_t i = 0; i < reqs.size(); ++i)
      actual[reqs[i]].insert(actual[reqs[i]].end(), fresh[i].begin(), fresh[i].end());
  }
  for (int req = 0; req < 2; ++req) {
    actual[req].resize(expected[req].size());
    require(actual[req] == expected[req], "constrained speculation matches ordinary masked greedy tokens");
    GrammarState check(&vocab, constraints[req], false);
    for (auto token : actual[req]) {
      require(check.allows(token), "every committed token satisfies its live grammar");
      check.advance(token);
    }
    const auto acceptance = static_cast<dgpp::sched::SchedulerEngine&>(engine).mtp_acceptance(req);
    require(acceptance.attempts[0] > 0 && acceptance.accepts[0] > 0,
            "both constrained requests actually speculate and accept drafts");
    require(engine.prefix_snapshot_ready(req), "speculation keeps draft state ready for prefix snapshots");
    engine.close(req);
  }
}

DGPP_TEST(dsv41_compact_neural_draft_candidates_match_full_logits) {
  const Fixture fx = make_fixture(false);
  const auto prompt = tokens_of(912, 23, fx.cfg.vocab_size);
  struct Enable {
    const char* key;
    std::optional<std::string> saved;
    explicit Enable(const char* name) : key(name) {
      if (const char* old = std::getenv(key)) saved = old;
      setenv(key, "1", 1);
    }
    ~Enable() {
      if (saved) setenv(key, saved->c_str(), 1);
      else unsetenv(key);
    }
  } candidates("DGPP_DSV41_DRAFT_CANDIDATES"), sparse("DGPP_DSV41_SPARSE_GRAMMAR");
  dgpp::Dsv41Model model(fx.cfg, fx.dir, 96, 512, dgpp::Dsv41Residency::Resident,
                         nullptr, 0, 1, 1, true, 8);
  model.session_prefill(0, prompt);
  const auto first = model.session_draft(0, {23});
  const auto chain = model.session_draft_chain(0, 15, 0, true, true);
  model.session_close(0);
  model.session_prefill(0, prompt);
  {
    auto selection = model.draft_candidate_scope();
    const auto check = [&](const auto& compact, const auto& full) {
      require(compact.logits.empty() && compact.local_candidates.size() == 1, "one draft winner");
      const auto best = dgpp::sample::local_max(full.logits.data(), full.lm_vocab_count, full.lm_vocab_begin);
      require(compact.local_candidates[0].id == best.id && compact.local_candidates[0].logit == best.logit,
              "compact draft preserves the token and logit");
      require(model.verification_logits_row(0) == full.logits, "an illegal draft winner can be fully masked");
      const int vocab = full.lm_vocab_count + 1;
      for (const auto& ids : std::vector<std::vector<int>>{{1}, {3, 4, 5}, {0, 9, 17}, {vocab - 1}}) {
        std::vector<uint32_t> words((vocab + 31) / 32);
        for (int token : ids) words[token / 32] |= uint32_t{1} << (token % 32);
        const auto got = model.verification_sparse_max(0, words.data(), vocab, ids.size());
        const auto want = dgpp::sample::local_masked_max(full.logits.data(), full.lm_vocab_count, 0, words.data(), vocab);
        require(got && got->id == want.id && got->logit == want.logit, "sparse grammar agrees with full-row masking");
      }
    };
    check(model.session_draft(0, {23}), first);
    check(model.session_draft_chain(0, 15, 0, true, true), chain);
  }
  model.session_close(0);
}

// The target has seen every proposed row when the draft input projection
// starts. Only the accepted prefix may be appended to the draft rings.
DGPP_TEST(dsv41_dspark_early_input_matches_accepted_rows) {
  const Fixture fx = make_fixture(false);
  auto paging = fixture_paging(fx, dgpp::expert::Policy::kWholeExpert, true);
  dgpp::Dsv41Model model(fx.cfg, fx.dir, 96, 512, dgpp::Dsv41Residency::Resident,
                         nullptr, 0, 1, 1, true, 8, false, &paging);
  const auto prompt = tokens_of(730, 23, fx.cfg.vocab_size);
  const std::vector<int64_t> feed{10, 20, 30, 40};
  const auto run = [&](bool candidates) {
    model.session_prefill(0, prompt);
    model.session_advance_draft(0, {feed[0]});
    std::vector<std::vector<float>> logits;
    for (int accepted : {1, 3, 4, 2}) {
      if (candidates) (void)model.session_verify_candidates(0, feed);
      else (void)model.session_verify(0, feed);
      model.session_rollback(0, accepted, 4);
      std::vector<int64_t> picks(feed.begin(), feed.begin() + accepted);
      logits.push_back(model.session_draft(0, picks).logits);
      logits.push_back(model.session_draft_chain(0, 11, 0, true, true).logits);
    }
    model.session_close(0);
    return logits;
  };
  const auto reference = run(false);
  require(run(true) == reference, "prepared draft and chain logits match for every accepted width");
}

DGPP_TEST(dsv41_eager_dspark_interior_snapshot_logits_match_live_cut) {
  const Fixture fx = make_fixture(false);
  const auto prompt = tokens_of(751, 125, fx.cfg.vocab_size);
  auto paging = fixture_paging(fx, dgpp::expert::Policy::kWholeExpert, true);
  // Both executions verify the same six rows. One rolls back to 128 and
  // saves a live snapshot. The other commits past 128 and recovers row 2.
  const auto resume = [&](bool interior) {
    dgpp::Dsv41Model model(fx.cfg, fx.dir, 256, 1024, dgpp::Dsv41Residency::Resident,
                           nullptr, 0, 1, 2, true, 16, false, &paging);
    const std::vector<int64_t> fed{10, 20, 30, 40, 50, 60};
    model.session_prefill(0, prompt);
    model.session_advance_draft(0, {fed.front()});
    model.session_verify(0, fed);
    const int accepted = interior ? 5 : 3;
    model.session_rollback(0, accepted, 6);
    model.session_advance_draft(0, {fed.begin() + 1, fed.begin() + 1 + accepted});
    void* snapshot = nullptr;
    DGPP_CUDA_OK(cudaMalloc(&snapshot, model.session_snapshot_bytes()));
    struct Free { void* p; ~Free() { (void)cudaFree(p); } } cleanup{snapshot};
    const auto meta = interior ? model.session_snapshot_post_row0(0, snapshot, 2, 2)
                               : model.session_snapshot(0, snapshot);
    model.session_close(0);
    model.session_attach(1, snapshot, meta);
    std::vector<std::vector<float>> logits;
    logits.push_back(model.session_prefill_resume(1, {41, 64, 74, 84}, {}).logits);
    logits.push_back(model.session_draft(1, {94}).logits);
    logits.push_back(model.session_verify(1, {94, 104}).logits);
    model.session_release_snapshot(meta);
    model.session_close(1);
    return logits;
  };
  const auto live = resume(false), interior = resume(true);
  require(interior[0] == live[0], "all resumed prefill logits match the live aligned snapshot");
  require(interior[1] == live[1], "all draft logits match after attaching the changed suffix");
  require(interior[2] == live[2], "all following target verification logits match");
}

DGPP_TEST(dsv41_eager_dspark_restores_interior_prefix_for_each_request) {
  using Engine = dgpp::sched::SchedulerEngine;
  const Fixture fx = make_fixture(false);
  const auto pick = dgpp::make_w1_pick(fx.cfg.vocab_size);
  std::vector<std::vector<int64_t>> prompts{
      tokens_of(43, 125, fx.cfg.vocab_size), tokens_of(47, 126, fx.cfg.vocab_size)};
  std::vector<std::vector<int32_t>> expected(2);
  {
    dgpp::Dsv41Model model(fx.cfg, fx.dir, 256, 1024, dgpp::Dsv41Residency::Resident,
                           nullptr, 0, 1, 2, false, 16);
    for (int req : {0, 1}) {
      auto& ids = expected[req];
      ids.push_back(pick(model.session_prefill(req, prompts[req])));
      for (int t = 0; t < 16; ++t) ids.push_back(pick(model.session_step(req, ids.back())));
      model.session_close(req);
    }
  }
  auto paging = fixture_paging(fx, dgpp::expert::Policy::kWholeExpert, true);
  dgpp::Dsv41Model model(fx.cfg, fx.dir, 256, 1024, dgpp::Dsv41Residency::Resident,
                         nullptr, 0, 1, 2, true, 16, false, &paging);
  dgpp::EagerEngineAdapter<dgpp::Dsv41Model> engine(&model, 2, pick, nullptr, nullptr, 2);
  int draft = 0;
  engine.configure_eager_mtp(5, [&](const std::vector<dgpp::sample::Candidate>& rows) {
    if (draft < 5 && rows.size() == 2) {
      ++draft;
      return std::vector<int32_t>{expected[0][draft], expected[1][draft]};
    }
    std::vector<int32_t> ids;
    for (const auto& row : rows) ids.push_back(row.id);
    return ids;
  });
  for (int req : {0, 1}) {
    require(engine.prefill(req, prompts[req]) == expected[req][0], "prefill agrees");
    engine.reserve(req, 160);
    engine.prefix_arm_hop(req, req, 128);
  }
  const auto batches = engine.step_batch({0, 1});
  for (int req : {0, 1}) {
    require(batches[req].size() == 6 && std::equal(batches[req].begin(), batches[req].end(),
            expected[req].begin() + 1), "full verification crosses the cut without shortening the pass");
    const auto saved = engine.take_prefix_hop_result(req);
    require(saved.status == Engine::PrefixHopStatus::kStored && saved.position == 128 &&
            saved.slot == req && engine.prefix_position(req) == 128, "saved the interior row of this slot");
    engine.close(req);
  }
  for (int req : {0, 1}) {
    // Resume with a suffix that differs from the verified rows past the cut.
    // A fresh full prefill supplies the reference picks and draft state.
    auto next = prompts[req];
    next.insert(next.end(), expected[req].begin(), expected[req].begin() + (128 - next.size()));
    next.insert(next.end(), {7, 11, 13, 17});
    const auto fresh = engine.prefill(req, next);
    engine.reserve(req, 160);
    const auto following = engine.step(req);
    engine.close(req);
    const std::vector<int64_t> boundaries{128};
    Engine::PrefixPrefill plan;
    plan.boundaries = &boundaries;
    plan.attach_slot = req;
    plan.attach_position = 128;
    require(engine.prefill_cached(req, next, &plan) == fresh, "reattached suffix agrees with a fresh prefill");
    engine.reserve(req, 160);
    require(engine.step(req) == following, "next verified and drafted tokens agree after the attach");
    engine.close(req);
  }
}

DGPP_TEST(dsv41_paged_eager_stale_draft_skips_optional_snapshot) {
  const Fixture fx = make_fixture(true);
  const auto prompt = tokens_of(0x5A9, 253, fx.cfg.vocab_size);
  const auto pick = dgpp::make_w1_pick(fx.cfg.vocab_size);
  std::vector<int32_t> expected;
  {
    dgpp::Dsv41Model reference(fx.cfg, fx.dir, 256, 1024, dgpp::Dsv41Residency::Resident,
                              nullptr, 0, 1, 1, false, 8);
    expected.push_back(pick(reference.session_prefill(0, prompt)));
    for (int i = 0; i < 7; ++i)
      expected.push_back(pick(reference.session_step(0, expected.back())));
  }
  auto paging = fixture_paging(fx, dgpp::expert::Policy::kWholeExpert, true);
  dgpp::Dsv41Model model(fx.cfg, fx.dir, 256, 1024, dgpp::Dsv41Residency::Resident,
                        nullptr, 0, 1, 1, true, 8, false, &paging);
  dgpp::EagerEngineAdapter<dgpp::Dsv41Model> engine(&model, 1, pick, nullptr, nullptr, 2);
  const std::vector<int64_t> boundaries{129};
  dgpp::sched::SchedulerEngine::PrefixPrefill plan;
  plan.boundaries = &boundaries;
  plan.snap_slot = 0;
  plan.snap_position = 128;
  require(engine.prefill_cached(0, prompt, &plan) == expected[0] && plan.snap_taken,
          "MTP prefill retains a valid prompt snapshot");
  require(engine.prefix_snapshot_ready(0), "prefill has current draft state");
  engine.reserve(0, prompt.size() + 8);
  bool crossed = false;
  for (int i = 0; i < 7; ++i) {
    require(engine.step(0) == std::vector<int32_t>{expected[i + 1]},
            "ordinary decoding with MTP enabled preserves target output");
    require(!engine.prefix_snapshot_ready(0), "unused draft state makes live snapshots unavailable");
    if (model.session_position(0) % model.session_snapshot_align() == 0) {
      crossed = true;
      require(model.session_draft_position(0) < model.session_position(0) - 1,
              "snapshot boundary reproduces the stale draft condition");
    }
  }
  require(crossed && engine.prefix_position(0) == 128,
          "crossing alignment preserves the valid prompt snapshot");
  engine.close(0);
  dgpp::sched::SchedulerEngine::PrefixPrefill reuse;
  reuse.boundaries = &boundaries;
  reuse.attach_slot = 0;
  reuse.attach_position = 128;
  require(engine.prefill_cached(0, prompt, &reuse) == expected[0] && engine.prefix_snapshot_ready(0),
          "cached follow-up restores valid target and draft state");
  engine.close(0);
}

DGPP_TEST(dsv41_paged_eager_dspark_batches_and_matches_plain_greedy) {
  using namespace dgpp::expert;
  struct PendingBoundary : dgpp::BoundaryReducer {
    bool pending = false;
    void reduce(uint16_t*, int, int) override { pending = true; }
    bool stream_ordered() const override { return true; }
    void settle() override { pending = false; }
  };
  for (bool nvfp4 : {false, true}) {
    const Fixture fx = make_fixture(nvfp4);
    const std::vector<std::vector<int64_t>> prompts{
        tokens_of(43, 23, fx.cfg.vocab_size), tokens_of(47, 17, fx.cfg.vocab_size)};
    const auto pick = dgpp::make_w1_pick(fx.cfg.vocab_size);
    std::vector<std::vector<int32_t>> expected(2);
    {
      dgpp::Dsv41Model reference(fx.cfg, fx.dir, 96, 512, dgpp::Dsv41Residency::Resident,
                                nullptr, 0, 1, 2, false, 16);
      for (int req = 0; req < 2; ++req) {
        auto& ids = expected[req];
        ids.push_back(pick(reference.session_prefill(req, prompts[req])));
        for (int t = 1; t < 30; ++t) ids.push_back(pick(reference.session_step(req, ids.back())));
        reference.session_close(req);
      }
    }
    for (auto policy : {Policy::kWholeExpert, Policy::kAdaptiveFractional}) {
      auto paging = fixture_paging(fx, policy, true);
      PendingBoundary boundary;
      dgpp::Dsv41Model model(fx.cfg, fx.dir, 96, 512, dgpp::Dsv41Residency::Resident,
                            nullptr, 0, 1, 2, true, 16, false, &paging);
      model.set_boundary(&boundary);
      dgpp::EagerEngineAdapter<dgpp::Dsv41Model> engine(&model, 2, pick);
      // Force only the first draft blocks to the known target tokens. The
      // verify still uses actual target logits, exercising all-accept commits
      // before naturally generated drafts exercise rejection and rollback.
      std::vector<int32_t> forced;
      for (int start : {1, 7}) for (int c = 0; c < 5; ++c) for (int req : {0, 1})
        forced.push_back(expected[req][start + c]);
      size_t forced_at = 0;
      engine.configure_eager_mtp(5, [&](const std::vector<dgpp::sample::Candidate>& rows) {
        require(!boundary.pending, "draft collectives settle before host token picks");
        if (rows.size() <= 2 && forced_at < forced.size()) {
          std::vector<int32_t> ids(forced.begin() + forced_at, forced.begin() + forced_at + rows.size());
          forced_at += rows.size();
          return ids;
        }
        std::vector<int32_t> ids;
        for (const auto& row : rows) ids.push_back(row.id);
        return ids;
      });
      require(engine.max_tokens_per_step() == 6 && engine.decode_batch_capacity() == 2,
              "DSpark advertises block reservations and batched capacity");
      std::vector<std::vector<int32_t>> actual(2);
      for (int req = 0; req < 2; ++req) {
        actual[req].push_back(engine.prefill(req, prompts[req]));
        engine.reserve(req, prompts[req].size() + 40);
      }
      bool rejected = false;
      try { (void)engine.step_batch({0, 0}); } catch (const std::invalid_argument&) { rejected = true; }
      require(rejected && forced_at == 0, "invalid batches do not draft or mutate requests");
      const auto first = engine.step_batch({0, 1});
      require(first[0].size() == 6 && first[1].size() == 6, "both forced draft blocks are fully accepted");
      for (int req = 0; req < 2; ++req) actual[req].insert(actual[req].end(), first[req].begin(), first[req].end());
      engine.configure_eager_mtp(5, [](const std::vector<dgpp::sample::Candidate>& rows) {
        std::vector<int32_t> ids;
        for (const auto& row : rows) ids.push_back(row.id);
        return ids;
      }, true, 8, 28, 100, 1, false);
      int iteration = 0;
      while (actual[0].size() < 24 || actual[1].size() < 24) {
        std::vector<int> reqs;
        for (int req : (iteration++ % 2 ? std::vector<int>{0, 1} : std::vector<int>{1, 0}))
          if (actual[req].size() < 24) reqs.push_back(req);
        const auto fresh = engine.step_batch(reqs);
        if (iteration > 1) for (const auto& ids : fresh)
          require(ids.size() <= 2, "confidence scheduling shortens verification to one draft");
        for (size_t i = 0; i < reqs.size(); ++i)
          actual[reqs[i]].insert(actual[reqs[i]].end(), fresh[i].begin(), fresh[i].end());
      }
      for (int req = 0; req < 2; ++req) {
        actual[req].resize(24);
        require(std::equal(actual[req].begin(), actual[req].end(), expected[req].begin()),
                "batched DSpark transcript equals plain greedy across accepts and rejects");
        const auto acceptance = static_cast<dgpp::sched::SchedulerEngine&>(engine).mtp_acceptance(req);
        require(acceptance.depth == 5 && acceptance.accepts[4] >= 1, "per-request acceptance includes full draft block");
        engine.close(req);
      }
      require(static_cast<dgpp::sched::SchedulerEngine&>(engine).mtp_acceptance().accepts[4] >= 2,
              "global acceptance survives slot retirement");
      require(engine.prefill(0, prompts[0]) == expected[0][0], "DSpark slot can be reused");
      const auto fresh = engine.step(0);
      require(std::equal(fresh.begin(), fresh.end(), expected[0].begin() + 1), "reused slot drafts from its new prompt");
      engine.close(0);
      int scheduled_picks = 0;
      engine.configure_eager_mtp(5, [&](const std::vector<dgpp::sample::Candidate>& rows) {
        if (rows.size() == 1) ++scheduled_picks;
        std::vector<int32_t> ids;
        for (const auto& row : rows) ids.push_back(row.id);
        return ids;
      }, true, 8, 28, 1e-30f, 1, false);
      require(engine.prefill(0, prompts[0]) == expected[0][0], "scheduled slot reopens");
      DGPP_CUDA_OK(cudaMemsetAsync(const_cast<float*>(model.device_confidence()), 0xff,
                                  model.confidence_rows() * sizeof(float), model.stream()));
      const auto scheduled = engine.step(0);
      require(scheduled_picks == 10,
              "scheduling reads freshly produced chain confidences, ignoring poisoned future rows");
      require(std::equal(scheduled.begin(), scheduled.end(), expected[0].begin() + 1),
              "scheduled full-depth output equals ordinary greedy");
      engine.close(0);
    }
  }
}

DGPP_TEST(dsv41_paged_prefill_overlap_matches_eager_and_preserves_decode) {
  const char* old = std::getenv("DGPP_EXPERT_PREFILL_OVERLAP");
  const std::optional<std::string> saved = old ? std::optional<std::string>(old) : std::nullopt;
  struct Restore {
    std::optional<std::string> value;
    ~Restore() { if (value) setenv("DGPP_EXPERT_PREFILL_OVERLAP", value->c_str(), 1);
                 else unsetenv("DGPP_EXPERT_PREFILL_OVERLAP"); }
  } restore{saved};
  for (bool nvfp4 : {false, true}) {
    const Fixture fx = make_fixture(nvfp4);
    const auto prompt = tokens_of(0xA134, 256, fx.cfg.vocab_size);
    for (auto policy : {dgpp::expert::Policy::kWholeExpert, dgpp::expert::Policy::kAdaptiveFractional}) {
      auto paging = fixture_paging(fx, policy);
      std::vector<std::vector<float>> expected;
      for (bool overlap : {false, true}) {
        setenv("DGPP_EXPERT_PREFILL_OVERLAP", overlap ? "1" : "0", 1);
        dgpp::Dsv41Model model(fx.cfg, fx.dir, 256, 512, dgpp::Dsv41Residency::Resident,
                              nullptr, 0, 1, 1, false, 8, false, &paging);
        if (overlap) model.set_prefill_fold_overlap(true, 256, true);
        std::vector<std::vector<float>> outputs{model.session_prefill(0, prompt).logits};
        for (int i=0; i<3; ++i) outputs.push_back(model.session_step(0, 17+i).logits);
        if (!overlap) expected = std::move(outputs);
        else require(outputs == expected, "paged row-block overlap preserves prefill and subsequent decode logits bitwise");
        require(model.expert_cache()->stats().active_execution_slots == 0, "overlap retires all paged scratch");
      }
    }
  }
}
