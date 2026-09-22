#include <algorithm>
#include <cmath>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>

#include "common/dtypes.hpp"
#include "common/test.hpp"
#include "loaders/minijson.hpp"
#include "models/mimo/attention.hpp"

namespace {
void check(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error(what);
}
template <class F>
void rejects(F fn, const std::string& part) {
  try {
    fn();
  } catch (const std::invalid_argument& e) {
    check(std::string(e.what()).find(part) != std::string::npos, e.what());
    return;
  }
  throw std::runtime_error("accepted invalid " + part);
}
void compare(const uint16_t* got, const dgpp::minijson::Value& expected, const std::string& label) {
  double diff = 0, norm = 0;
  int far = 0;
  for (size_t i = 0; i < expected.items().size(); ++i) {
    const double a = dgpp::bf16_bits_to_float(got[i]);
    const double b = dgpp::bf16_bits_to_float(expected.items()[i].as_int());
    const double d = std::abs(a - b);
    diff += d * d;
    norm += b * b;
    if (d > 1e-5 && d > std::max(std::abs(a), std::abs(b)) / 128) ++far;
  }
  check(far == 0 && std::sqrt(diff / std::max(norm, 1e-30)) <= 0.002,
        label + ": differs from PyTorch beyond one BF16 ulp / 0.2% relative L2");
}
}  // namespace

DGPP_TEST(mimo_attention_matches_cpu_pytorch_eager_goldens) {
  std::ifstream f(std::string(DGPP_SOURCE_DIR) + "/tests/data/mimo/attention_goldens.json");
  check(bool(f), "missing attention goldens");
  const std::string text{std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
  const auto doc = dgpp::minijson::parse(text);
  int checked = 0;
  for (int window : {0, 128}) {
    const dgpp::MimoAttentionShape s{1, 2, 1, window ? 128 : 257, window};
    std::vector<uint16_t> fused(s.fused_width()), q(s.q_width()), k(s.capacity * s.k_width()),
        v(s.capacity * s.v_width()), out(s.q_heads * 128);
    const uint16_t sinks[] = {dgpp::float_to_bf16_bits(0.25f), dgpp::float_to_bf16_bits(-0.5f)};
    const auto freq = dgpp::mimo_ref::inv_freq(window ? 1e4 : 1e7);
    for (int64_t pos = 0; pos <= 256; ++pos) {
      for (int i = 0; i < s.fused_width(); ++i)
        fused[i] = dgpp::float_to_bf16_bits(float((i * 7 + pos * 13) % 101 - 50) / 64);
      dgpp::mimo_ref::qkv_append(s, fused.data(), freq.data(), &pos, q.data(), k.data(), v.data());
      for (const auto& golden : doc.root.at("cases").items()) {
        if (golden.at("window").as_int() != window || golden.at("position").as_int() != pos)
          continue;
        const std::string label =
            "window=" + std::to_string(window) + " position=" + std::to_string(pos);
        compare(q.data(), golden.at("q"), "Q " + label);
        compare(k.data() + (pos % s.capacity) * s.k_width(), golden.at("k"), "K " + label);
        compare(v.data() + (pos % s.capacity) * s.v_width(), golden.at("v"), "V " + label);
        dgpp::mimo_ref::attention(s, q.data(), k.data(), v.data(), &pos, window ? sinks : nullptr,
                                  out.data());
        compare(out.data(), golden.at("output"), "attention " + label);
        ++checked;
      }
    }
  }
  check(checked == 16, "all PyTorch boundary cases checked");
  for (const auto& golden : doc.root.at("rotary_cases").items()) {
    const bool swa = golden.at("window").as_int() != 0;
    // Only inspect rotary processing here; a full 1M-token attention cache
    // is neither allocated nor implied by these high-position goldens.
    const dgpp::MimoAttentionShape shape{1, 2, 1, 128, 128};
    const int64_t pos = golden.at("position").as_int();
    const auto freq = dgpp::mimo_ref::inv_freq(swa ? 1e4 : 1e7);
    std::vector<uint16_t> fused(shape.fused_width()), q(shape.q_width()), k(128 * shape.k_width()),
        v(128 * shape.v_width());
    for (int i = 0; i < shape.fused_width(); ++i)
      fused[i] = dgpp::float_to_bf16_bits(float((i * 7 + pos * 13) % 101 - 50) / 64);
    dgpp::mimo_ref::qkv_append(shape, fused.data(), freq.data(), &pos, q.data(), k.data(),
                               v.data());
    compare(q.data(), golden.at("q"), "high-position Q " + std::to_string(pos));
    compare(k.data() + (pos % 128) * shape.k_width(), golden.at("k"),
            "high-position K " + std::to_string(pos));
  }
}

DGPP_TEST(mimo_sliding_ring_evicts_exactly_the_oldest_token) {
  const dgpp::MimoAttentionShape s{1, 2, 1, 128, 128};
  std::vector<uint16_t> fused(s.fused_width(), 0), q(s.q_width()), k(s.capacity * s.k_width()),
      v(s.capacity * s.v_width()), out(s.q_heads * 128);
  const auto freq = dgpp::mimo_ref::inv_freq(1e4);
  for (int64_t pos = 0; pos <= 256; ++pos) {
    std::fill(fused.begin() + s.q_width() + s.k_width(), fused.end(),
              dgpp::float_to_bf16_bits(pos == 0 ? 64.f : 1.f));
    dgpp::mimo_ref::qkv_append(s, fused.data(), freq.data(), &pos, q.data(), k.data(), v.data());
    if (pos != 127 && pos != 128 && pos != 256) continue;
    dgpp::mimo_ref::attention(s, q.data(), k.data(), v.data(), &pos, nullptr, out.data());
    if (pos == 127)
      check(dgpp::bf16_bits_to_float(out[0]) > 1, "oldest still visible at window end");
    else
      for (auto value : out)
        check(value == dgpp::float_to_bf16_bits(0.707f), "expired token leaked across wrap");
  }
  // A new request in the same slot starts at zero; old ring entries must
  // remain invisible without clearing the entire cache allocation.
  const int64_t reset = 0;
  std::fill(fused.begin() + s.q_width() + s.k_width(), fused.end(), dgpp::float_to_bf16_bits(2.f));
  dgpp::mimo_ref::qkv_append(s, fused.data(), freq.data(), &reset, q.data(), k.data(), v.data());
  dgpp::mimo_ref::attention(s, q.data(), k.data(), v.data(), &reset, nullptr, out.data());
  for (auto value : out)
    check(value == dgpp::float_to_bf16_bits(2.f * 0.707f), "reset read stale ring entries");
}

DGPP_TEST(mimo_attention_sink_is_zero_value_softmax_mass) {
  const dgpp::MimoAttentionShape s{1, 2, 1, 1, 0};
  std::vector<uint16_t> q(s.q_width()), k(s.k_width()), v(s.v_width(), dgpp::float_to_bf16_bits(1)),
      out(256);
  const int64_t pos = 0;
  dgpp::mimo_ref::attention(s, q.data(), k.data(), v.data(), &pos, nullptr, out.data());
  check(out[0] == dgpp::float_to_bf16_bits(1), "one key without sink");
  const uint16_t sinks[] = {0, dgpp::float_to_bf16_bits(128)};
  dgpp::mimo_ref::attention(s, q.data(), k.data(), v.data(), &pos, sinks, out.data());
  check(out[0] == dgpp::float_to_bf16_bits(0.5f), "equal sink halves output");
  check(out[128] == 0, "large sink is stable and suppresses output");
}

DGPP_TEST(mimo_attention_rows_pause_reset_and_reject_invalid_positions) {
  const dgpp::MimoAttentionShape s{17, 2, 1, 128, 128};
  const auto freq = dgpp::mimo_ref::inv_freq(1e4);
  std::vector<uint16_t> fused(s.requests * s.fused_width()), q(s.requests * s.q_width()),
      k(s.requests * s.capacity * s.k_width(), 0x1234),
      v(s.requests * s.capacity * s.v_width(), 0x1234), out(s.requests * 256);
  std::vector<int64_t> pos(s.requests, 0);
  for (int r = 0; r < s.requests; ++r) {
    if (r % 3 == 0) pos[r] = -1;
    std::fill(fused.begin() + r * s.fused_width() + s.q_width() + s.k_width(),
              fused.begin() + (r + 1) * s.fused_width(),
              dgpp::float_to_bf16_bits(float(r + 1) / 16));
  }
  dgpp::mimo_ref::qkv_append(s, fused.data(), freq.data(), pos.data(), q.data(), k.data(),
                             v.data());
  dgpp::mimo_ref::attention(s, q.data(), k.data(), v.data(), pos.data(), nullptr, out.data());
  for (int r = 0; r < s.requests; ++r) {
    check(out[r * 256] == (pos[r] < 0 ? 0 : dgpp::float_to_bf16_bits(float(r + 1) / 16 * 0.707f)),
          "row isolation");
    if (pos[r] < 0)
      check(k[r * s.capacity * s.k_width()] == 0x1234 && v[r * s.capacity * s.v_width()] == 0x1234,
            "padding writes cache");
  }
  const auto before = k;
  pos.back() = 1048576;
  rejects(
      [&] {
        dgpp::mimo_ref::qkv_append(s, fused.data(), freq.data(), pos.data(), q.data(), k.data(),
                                   v.data());
      },
      "position");
  check(k == before, "invalid append partially mutated cache");
  rejects([&] { dgpp::MimoAttentionShape{1, 3, 2, 128, 128}.validate(); }, "heads");
  rejects([&] { dgpp::MimoAttentionShape{1, 2, 1, 127, 128}.validate(); }, "window");
  rejects([&] { dgpp::MimoAttentionShape{0, 2, 1, 128, 128}.validate(); }, "requests");
}
