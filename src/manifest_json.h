#pragma once

#include <filesystem>

#include <jsoncons/json.hpp>

namespace ttl_internal {

// Parses JSON, validates the canonical version-1 schema, then checks the
// cross-field relationships and semantic invariants that JSON Schema cannot
// express.
jsoncons::json parse_and_validate_manifest(const std::filesystem::path &path);
jsoncons::json parse_and_validate_launch(const std::filesystem::path &path);
jsoncons::json parse_and_validate_program(const std::filesystem::path &path);
jsoncons::json parse_and_validate_fixture(const std::filesystem::path &path);

}  // namespace ttl_internal
