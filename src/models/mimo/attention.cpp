#include "models/mimo/attention.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "common/dtypes.hpp"

namespace dgpp {
namespace mimo_ref {
namespace {
float rb(float x) {
  return bf16_bits_to_float(float_to_bf16_bits(x));
}
bool active(const MimoAttentionShape& s, int64_t p) {
  if (p >= 1048576 || (s.window == 0 && p >= s.capacity))
    throw std::invalid_argument("MiMo attention position: outside cache/context range");
  return p >= 0;
}
}  // namespace

std::vector<float> inv_freq(double theta) {
  if (!std::isfinite(theta) || theta <= 0)
    throw std::invalid_argument("MiMo RoPE theta: must be positive and finite");
  std::vector<float> out(32);
  for (int i = 0; i < 32; ++i)
    out[i] = 1.0f / std::pow(static_cast<float>(theta), static_cast<float>(2 * i) / 64.0f);
  return out;
}

void qkv_append(const MimoAttentionShape& s, const uint16_t* fused, const float* freq,
                const int64_t* positions, uint16_t* q, uint16_t* kc, uint16_t* vc) {
  s.validate();
  if (!fused || !freq || !positions || !q || !kc || !vc)
    throw std::invalid_argument("MiMo append: null pointer");
  // Validate all positions before mutating any cache row.
  for (int r = 0; r < s.requests; ++r) (void)active(s, positions[r]);
  for (int r = 0; r < s.requests; ++r) {
    const int64_t p = positions[r];
    if (p < 0) continue;
    const auto* input = fused + int64_t(r) * s.fused_width();
    const int64_t slot = int64_t(r) * s.capacity + p % s.capacity;
    for (int kind = 0; kind < 2; ++kind) {
      const int heads = kind == 0 ? s.q_heads : s.kv_heads;
      const auto* source = input + (kind == 0 ? 0 : s.q_width());
      auto* dest = kind == 0 ? q + int64_t(r) * s.q_width() : kc + slot * s.k_width();
      for (int h = 0; h < heads; ++h) {
        for (int d = 0; d < 192; ++d) {
          const float x = bf16_bits_to_float(source[h * 192 + d]);
          float y = x;
          if (d < 64) {
            const int i = d % 32;
            const float angle = static_cast<float>(p) * freq[i];
            const float cos = rb(std::cos(angle)), sin = rb(std::sin(angle));
            const float other = bf16_bits_to_float(source[h * 192 + (d < 32 ? d + 32 : d - 32)]);
            y = rb(rb(x * cos) + rb((d < 32 ? -other : other) * sin));
          }
          dest[h * 192 + d] = float_to_bf16_bits(y);
        }
      }
    }
    for (int d = 0; d < s.v_width(); ++d)
      vc[slot * s.v_width() + d] =
          float_to_bf16_bits(bf16_bits_to_float(input[s.q_width() + s.k_width() + d]) * 0.707f);
  }
}

void attention(const MimoAttentionShape& s, const uint16_t* q, const uint16_t* kc,
               const uint16_t* vc, const int64_t* positions, const uint16_t* sinks, uint16_t* out) {
  s.validate();
  if (!q || !kc || !vc || !positions || !out)
    throw std::invalid_argument("MiMo attention: null pointer");
  for (int r = 0; r < s.requests; ++r) (void)active(s, positions[r]);
  const float scale = static_cast<float>(1.0 / std::sqrt(192.0));
  for (int r = 0; r < s.requests; ++r) {
    const int64_t pos = positions[r];
    if (pos < 0) {
      std::fill_n(out + int64_t(r) * s.q_heads * 128, s.q_heads * 128, uint16_t(0));
      continue;
    }
    const int64_t begin = s.window ? std::max<int64_t>(0, pos - s.window + 1) : 0;
    const int n = static_cast<int>(pos - begin + 1);
    std::vector<float> scores(n + (sinks ? 1 : 0));
    for (int h = 0; h < s.q_heads; ++h) {
      const int kh = h / (s.q_heads / s.kv_heads);
      for (int t = 0; t < n; ++t) {
        const int64_t slot = int64_t(r) * s.capacity + (begin + t) % s.capacity;
        float dot = 0;
        for (int d = 0; d < 192; ++d)
          dot = std::fma(bf16_bits_to_float(q[(int64_t(r) * s.q_heads + h) * 192 + d]),
                         bf16_bits_to_float(kc[slot * s.k_width() + kh * 192 + d]), dot);
        scores[t] = rb(rb(dot) * scale);
      }
      if (sinks) scores[n] = bf16_bits_to_float(sinks[h]);
      const float max = *std::max_element(scores.begin(), scores.end());
      float denom = 0;
      for (auto& score : scores) {
        score = std::exp(rb(score - max));
        denom += score;
      }
      for (int t = 0; t < n; ++t) scores[t] = rb(scores[t] / denom);
      for (int d = 0; d < 128; ++d) {
        float sum = 0;
        for (int t = 0; t < n; ++t) {
          const int64_t slot = int64_t(r) * s.capacity + (begin + t) % s.capacity;
          sum = std::fma(scores[t], bf16_bits_to_float(vc[slot * s.v_width() + kh * 128 + d]), sum);
        }
        out[(int64_t(r) * s.q_heads + h) * 128 + d] = float_to_bf16_bits(sum);
      }
    }
  }
}
}  // namespace mimo_ref
}  // namespace dgpp
