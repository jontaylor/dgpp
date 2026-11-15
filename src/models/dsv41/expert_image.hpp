#pragma once
// Startup preparation only, before releasing source mappings. Images contain
// the rank's exact MXFP4/NVFP4 slices; EXL3 is explicitly unsupported. Source tensors
// must remain immutable for the entire hash/pack/open operation.
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>

#include "loaders/expert_store.hpp"
#include "loaders/safetensors.hpp"
#include "models/dsv41/config.hpp"

namespace dgpp {
using Dsv41ExpertSources = std::unordered_map<std::string, const TensorInfo*>;
// Hashes config bytes plus a canonical, length-delimited stream of all enabled
// routed expert tensor names, shapes, dtypes and complete source bytes. This is
// an expert-source-set SHA256, not a hash of unrelated checkpoint payloads.
// The ABI's checkpoint_sha256 field holds this source-set identity. Chunk/TP
// geometry and the enabled main/draft record population are checked separately.
// An existing invalid/stale image is refused; only an absent image is built.
// Prepared mapped-file images save a .source-stamp sidecar. Subsequent opens
// reuse the source digest when source descriptors/inode metadata and image
// inode metadata match, avoiding another payload scan. Missing/stale metadata
// falls back to the full source hash; anonymous borrowed tensors always hash.
// ModelOpt NVFP4 main experts preserve codes/block scales and translate the
// exactly invertible weight_scale_2 into the existing kernel's divisor
// convention. Original weight/input scalar bytes remain in source identity.
// Mixed NVFP4 main / MXFP4 draft images are refused.
std::shared_ptr<expert::ExpertStore> dsv41_prepare_expert_image(
    const std::string& path, const Dsv41TextConfig& cfg, std::string_view config_json,
    const Dsv41ExpertSources& tensors, int rank, int world, uint64_t chunk_intermediate,
    bool include_draft, expert::IoMode mode);
}  // namespace dgpp
