#pragma once
#include <stdexcept>

#include "engine/boundary_reducer.hpp"

namespace dgpp {
// The reducer belongs to the CURRENT call, never to drafter weights/state.
// Graph capture temporarily installs GraphRecordReducer on SessionModel; graph
// replay executes its recorded nodes, while later eager calls use the restored
// live reducer. Keeping this adapter stateless prevents retaining that temporary.
// Callbacks preserve the device ordering and make it testable without a GPU.
template <class Project, class Synchronize, class Add>
void mimo_dflash_fold(BoundaryReducer* boundary, int world, uint16_t* fallback, int rows,
                      int hidden, bool capture, Project&& project, Synchronize&& synchronize,
                      Add&& add) {
  if ((world > 1) != (boundary != nullptr))
    throw std::invalid_argument("DFlash proposal requires the current TP reducer");
  uint16_t* partial = boundary ? boundary->stage(rows, hidden) : nullptr;
  if (capture && boundary && !partial)
    throw std::logic_error("DFlash capture requires boundary staging");
  if (!partial) partial = fallback;
  project(partial);
  if (boundary) {
    if (!capture && !boundary->stream_ordered()) synchronize();
    boundary->reduce(partial, rows, hidden);
  }
  add(partial);
}
}  // namespace dgpp
