#pragma once
#include <string>

#include "loaders/expert_cache.hpp"
#include "loaders/minijson.hpp"

namespace dgpp::serve {
expert::PagingOptions parse_expert_paging(const minijson::Value&, const std::string& what);
// Canonical full effective settings, also used by rank journal/digest.
std::string expert_paging_json(const expert::PagingOptions&);
}  // namespace dgpp::serve
