// Host-only checkpoint inspection. Never reads tensor payloads or allocates
// GPU memory; --index-only also works before the weight download completes.
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <stdexcept>
#include <unordered_set>

#include "loaders/safetensors.hpp"
#include "models/mimo/binding.hpp"
#include "models/mimo/weights.hpp"

namespace {
std::string read(const std::filesystem::path& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open " + path.string());
  std::string text{std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
  if (f.bad()) throw std::runtime_error("cannot read " + path.string());
  return text;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2 || argc > 3 || (argc == 3 && std::string(argv[2]) != "--index-only"))
      throw std::runtime_error("usage: mimo_checkpoint_check CHECKPOINT_DIR [--index-only]");
    const bool index_only = argc == 3;
    const std::filesystem::path dir(argv[1]);
    const auto cfg = dgpp::MimoTextConfig::from_json_file((dir / "config.json").string());
    const auto index_text = read(dir / "model.safetensors.index.json");
    const auto index = dgpp::minijson::parse(index_text);
    const auto& metadata = index.root.at("metadata");
    const auto& tp = metadata.at("tp_size");
    if (!tp.is_number() || tp.as_double() != 4 || metadata.at("save_format").as_string() != "mxfp4")
      throw std::runtime_error("MiMo index metadata: expected tp_size=4 and save_format=mxfp4");
    std::map<std::string, std::vector<std::string>> shards;
    std::unordered_set<std::string> names;
    const auto& weights = index.root.at("weight_map");
    if (!weights.is_object()) throw std::runtime_error("MiMo index weight_map: expected object");
    for (const auto& m : weights.members()) {
      if (!m.value.is_string()) throw std::runtime_error("MiMo index: shard name must be string");
      const std::filesystem::path shard(m.value.as_string());
      if (shard.empty() || shard != shard.filename() || shard.extension() != ".safetensors")
        throw std::runtime_error("MiMo index: invalid shard path");
      if (!names.emplace(m.key).second)
        throw std::runtime_error("MiMo index: duplicate tensor name");
      shards[shard.string()].emplace_back(m.key);
    }
    std::unordered_map<std::string, dgpp::MimoTensorDesc> tensors;
    if (index_only) {
      // Use the expected description solely to check name coverage. No
      // claim about stored dtype, shape or payload follows from this mode.
      for (const auto& t : dgpp::mimo_expected_text_tensors(cfg)) {
        if (!names.contains(t.name)) throw std::runtime_error("MiMo index: missing " + t.name);
        tensors.emplace(t.name, t);
      }
      for (const auto& name : names) tensors.try_emplace(name);
    } else {
      for (const auto& [shard, entries] : shards) {
        const auto file = dgpp::SafetensorsFile::open((dir / shard).string());
        if (file->tensor_count() != entries.size())
          throw std::runtime_error("MiMo index/header tensor count mismatch: " + shard);
        for (const auto& name : entries) {
          const auto& t = file->at(name);
          if (t.data_end < t.data_begin || t.nbytes() != t.numel() * dgpp::dtype_size(t.dtype))
            throw std::runtime_error("MiMo header: payload byte count mismatch " + name);
          tensors.emplace(name, dgpp::MimoTensorDesc{t.dtype, t.shape});
        }
      }
    }
    const auto report = dgpp::mimo_validate_text_binding(cfg, tensors);
    std::cout << (index_only ? "Index names checked (shapes and payloads NOT checked)"
                             : "Tensor headers checked (payloads NOT checked)")
              << "\ntext tensors: " << report.text << "\nunvalidated MTP tensors: " << report.mtp
              << "\nunvalidated multimodal tensors: " << report.multimodal
              << "\nshards: " << shards.size()
              << "\nBF16 KV bytes per rank at 32K context, TP2: " << cfg.kv_bytes(32768, 2)
              << "\nPlanned initial weight bytes per rank, TP2 (not measured residency): "
              << dgpp::mimo_weight_budget(cfg, 2).total()
              << "\nNo MiMo execution backend is implemented.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
