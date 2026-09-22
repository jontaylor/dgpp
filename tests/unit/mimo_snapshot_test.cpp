#include <stdexcept>

#include "common/test.hpp"
#include "models/mimo/snapshot.hpp"

DGPP_TEST(mimo_snapshot_provenance_release_reassign_reset_rollback) {
  dgpp::MimoSnapshotHistory h;
  int slots[4];
  auto check = [](bool ok) { if (!ok) throw std::runtime_error("snapshot provenance"); };
  for (int i = 0; i < 4; ++i) {
    h.register_destination(&slots[i]);
    check(h.begin(0, &slots[i], 100) == 0);
    h.commit(0, &slots[i], 100);
  }
  check(h.begin(0, &slots[0], 110) == 100);
  // Failed enqueue (no commit) cannot be reused.
  check(h.begin(0, &slots[0], 111) == 0);
  h.commit(0, &slots[0], 111);
  check(h.begin(1, &slots[1], 120) == 0);
  h.commit(1, &slots[1], 120);
  h.release(&slots[2]);
  check(h.begin(0, &slots[2], 130) == 0);
  h.commit(0, &slots[2], 130);
  h.rewind(0, 110);
  check(h.begin(0, &slots[0], 140) == 0);
  check(h.begin(0, &slots[2], 140) == 0);
  check(h.begin(0, &slots[3], 140) == 100);
  h.commit(0, &slots[3], 140);
  check(h.begin(1, &slots[1], 140) == 120);
  h.commit(1, &slots[1], 140);
  h.rewind(0);  // close, cancel or attach/reset
  check(h.begin(0, &slots[3], 150) == 0);
  check(h.begin(1, &slots[1], 150) == 140);
  h.commit(1, &slots[1], 150);
  check(h.begin(1, &slots[1], 149) == 0);  // defensive backwards refresh
  h.unregister_destination(&slots[1]);
  h.commit(1, &slots[1], 150);
  check(h.begin(1, &slots[1], 151) == 0);  // raw/reallocated destination
  h.register_destination(&slots[1]);
  h.commit(1, &slots[1], 150);
  check(h.begin(1, &slots[1], 151) == 150);
}
