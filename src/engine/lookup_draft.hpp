#pragma once
// Context-lookup drafting (HyperQwen's `dflash2-lookup-drafting` idea,
// adapted to dgpp's MTP path): propose the continuation of the most recent
// earlier occurrence of the suffix just generated, from the request's own
// token history. Drafts are point masses, so the greedy verify stays exact:
// a draft commits iff it equals the target's argmax at that position, and an
// unverified or rejected tail is decoded plainly next step.
//
// v1 scope: same-depth fusion (lookup replaces MTP drafts within the
// configured depth) plus engine.lookup_tail: extra verify rows past the MTP
// block (a pinned mode; the device chain spans the full width and a strong
// lookup match fuses over it). Adaptive (triggered) tail sizing is
// follow-up work.
#include <algorithm>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace dgpp {

// What the history scan found.
struct LookupDraft {
  std::vector<int32_t> drafts;  // continuation tokens after the match (up to max_drafts)
  int match_len = 0;            // length of the matched suffix (0: no usable match)
  int64_t match_end = -1;       // history index where the matched occurrence ends
  bool strong = false;          // match_len >= nstrong
};

// Propose a continuation from token history.
//
// history[0..n): prompt ids + committed + pending tokens (the full sequence
//   the request has consumed so far, in order).
// max_drafts: continuation tokens to propose (clamped by what follows the match).
// nmin: minimum suffix-match length to propose anything (a coincidental short
//   match costs acceptance on prose).
// max_suffix: longest suffix considered (the scan compares at most this many
//   back from the end).
// nstrong: threshold marking a match takable on its own (see lookup_fuse).
//
// Picks the most recent (largest match_end < n - 1) occurrence of the longest
// suffix; ties go to the most recent. Overlapping occurrences are allowed, so
// a repeating period proposes from its own period. O(n * max_suffix) worst
// case with an early-out on first mismatch — fine for C1-C8 host-side use at
// moderate contexts; a 256K-context production path wants an index, not this.
inline LookupDraft lookup_propose(const int64_t* history, int64_t n, int max_drafts, int nmin = 6,
                                  int max_suffix = 64, int nstrong = 8) {
  LookupDraft out;
  if (history == nullptr || n < 2 || max_drafts <= 0 || nmin <= 0) return out;
  if (max_suffix > n - 1) max_suffix = static_cast<int>(n - 1);
  const int64_t last = history[n - 1];
  int best_len = 0;
  int64_t best_end = -1;
  // Most-recent-first: strictly-greater comparison keeps the first (most
  // recent) end on ties.
  for (int64_t e = n - 2; e >= 0; --e) {
    if (history[e] != last) continue;
    // Max length this end could reach: bounded by history start, the suffix
    // cap, and the suffix itself (the occurrence must end before the current
    // end... it may overlap, so only by the buffer start).
    int64_t cap = e + 1;  // occurrence start >= 0
    if (cap > max_suffix) cap = max_suffix;
    if (cap > n - 1) cap = n - 1;
    if (cap <= best_len) continue;  // cannot beat the incumbent
    int len = 1;
    while (len < cap && history[e - len] == history[n - 1 - len]) ++len;
    if (len > best_len) {
      best_len = len;
      best_end = e;
    }
  }
  if (best_len < nmin || best_end < 0) {
    // Report what the scan found even when it proposes nothing (below the
    // floor): the counter and the log distinguish "no repeat" from a weak
    // match the fusion rule skips.
    out.match_len = best_len;
    out.match_end = best_end;
    out.strong = best_len >= nstrong;
    return out;
  }
  out.match_len = best_len;
  out.match_end = best_end;
  out.strong = best_len >= nstrong;
  for (int i = 0; i < max_drafts && best_end + 1 + i < n; ++i)
    out.drafts.push_back(static_cast<int32_t>(history[best_end + 1 + i]));
  if (out.drafts.empty()) {
    out.match_len = 0;
    out.match_end = -1;
    out.strong = false;
  }
  return out;
}

// Incremental index for append-only eager slot histories. Every candidate is
// checked against the original tokens, including hash collisions. Ties use
// the latest occurrence, as in lookup_propose. Only accepted tokens enter it.
class LookupHistoryIndex {
 public:
  LookupDraft propose(std::span<const int32_t> history, int width, int nmin, int nstrong) {
    LookupDraft result;
    const int size = static_cast<int>(history.size());
    if (width <= 0 || nmin <= 0 || size <= nmin) return result;
    const int key_tokens = std::min(nmin, 4);
    if (key_tokens != key_tokens_ || previous_.size() > history.size()) {
      heads_.assign(65536, -1);
      previous_.clear();
      key_tokens_ = key_tokens;
    }
    while (previous_.size() < history.size()) {
      const int end = static_cast<int>(previous_.size());
      if (end + 1 < key_tokens) {
        previous_.push_back(-1);
        continue;
      }
      uint32_t hash = 2166136261u;
      for (int j = end + 1 - key_tokens; j <= end; ++j)
        hash = (hash ^ static_cast<uint32_t>(history[j])) * 16777619u;
      auto& head = heads_[(hash ^ (hash >> 16)) & 65535u];
      previous_.push_back(head);
      head = end;
    }
    const int cap = std::min(64, size - 1);
    for (int end = previous_.back(); end >= nmin - 1; end = previous_[end]) {
      int len = 0;
      while (len <= end && len < cap && history[end - len] == history[size - 1 - len]) ++len;
      if (len > result.match_len && len >= nmin) {
        result.match_len = len;
        result.match_end = end;
        if (len == cap) break;
      }
    }
    if (result.match_end >= 0) {
      const int start = static_cast<int>(result.match_end + 1);
      result.drafts.assign(history.begin() + start, history.begin() + start + std::min(width, size - start));
      result.strong = result.match_len >= nstrong;
    }
    return result;
  }

  // Called after propose() on this same history. Only consider older strong
  // occurrences that agree with every neural token and have at least one
  // further known token. Bound the retry cost even for repetitive prompts.
  std::span<const int32_t> alternate_tail(std::span<const int32_t> history,
      std::span<const int32_t> prefix, int width, int strong) const {
    if (history.empty() || previous_.size() != history.size() || prefix.empty() ||
        width <= static_cast<int>(prefix.size()) || strong <= 0) return {};
    const int last = static_cast<int>(history.size()) - 1;
    const int cap = std::min(64, last);
    int best = strong - 1;
    std::span<const int32_t> tail;
    int remaining = 32;
    for (int end = previous_.back(); end >= best && remaining > 0; end = previous_[end], --remaining) {
      const auto continuation = history.subspan(end + 1);
      if (continuation.size() <= prefix.size() ||
          !std::equal(prefix.begin(), prefix.end(), continuation.begin())) continue;
      int length = 0;
      while (length <= end && length < cap && history[end - length] == history[last - length]) ++length;
      if (length <= best) continue;
      best = length;
      tail = continuation.first(std::min<size_t>(width, continuation.size())).subspan(prefix.size());
      if (best == cap) break;
    }
    return tail;
  }

 private:
  int key_tokens_ = 0;
  std::vector<int32_t> heads_, previous_;
};

// Eager DSpark already computed the prefix. Spend the extra verification rows
// only on a strong continuation that agrees with every one of those tokens.
inline void lookup_extend(std::vector<int32_t>& drafts, const LookupDraft& copy) {
  if (copy.strong && copy.drafts.size() > drafts.size() &&
      std::equal(drafts.begin(), drafts.end(), copy.drafts.begin()))
    drafts.insert(drafts.end(), copy.drafts.begin() + drafts.size(), copy.drafts.end());
}

// Fuse MTP drafts with a lookup proposal (same depth, no extra rows).
//
// A strong match is taken on its own. A weak one only if the drafter
// independently proposed the same first `agree` tokens — two sources
// agreeing (hidden state + text) is the cheap confidence signal that stops a
// coincidental short match from costing acceptance on prose. Otherwise the
// MTP drafts stand. Lengths: the fused block has mtp.size() entries (lookup
// tail truncated or zero-padded with the MTP tail when short).
inline LookupDraft lookup_fuse(const std::vector<int32_t>& mtp, const LookupDraft& lu, int nstrong = 8,
                               int agree = 2) {
  LookupDraft out;
  out.match_len = lu.match_len;
  out.match_end = lu.match_end;
  out.drafts = mtp;
  if (lu.drafts.empty() || lu.match_len < 0) return out;
  bool take = lu.match_len >= nstrong;
  if (!take && lu.match_len > 0 && agree > 0) {
    take = true;
    for (int i = 0; i < agree; ++i) {
      if (i >= static_cast<int>(mtp.size()) || i >= static_cast<int>(lu.drafts.size()) ||
          mtp[static_cast<size_t>(i)] != lu.drafts[static_cast<size_t>(i)]) {
        take = false;
        break;
      }
    }
  }
  if (!take) return out;
  out.strong = lu.strong;
  for (size_t i = 0; i < out.drafts.size() && i < lu.drafts.size(); ++i)
    out.drafts[i] = lu.drafts[i];
  return out;
}

// Schedules the long verify block only while a copy is running: each extra
// verify row costs attention over the full context, so the long block is
// worth it only when steps saturate. A single saturated step happens inside
// ordinary prose; two in a row is a copy (HyperQwen's rule).
class LookupTrigger {
 public:
  explicit LookupTrigger(int need = 2) : need_(need < 1 ? 1 : need) {}
  // verified: rows fed this step (drafts); accepted: drafts accepted.
  // Returns true when the next step may use the long block.
  bool update(int verified, int accepted) {
    if (verified > 0 && accepted >= verified)
      ++streak_;
    else
      streak_ = 0;
    return streak_ >= need_;
  }
  void reset() { streak_ = 0; }
  int streak() const { return streak_; }

 private:
  int need_ = 2;
  int streak_ = 0;
};

}  // namespace dgpp
