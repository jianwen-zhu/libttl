#include "manifest_json.h"

#include "ttl_kernel_module_schema.h"
#include "ttl_kernel_launch_schema.h"
#include "ttl_program_schema.h"
#include "ttl_fixture_schema.h"

#include <jsoncons_ext/jsonschema/jsonschema.hpp>

#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>

namespace ttl_internal {
namespace {

using json = jsoncons::json;

const jsoncons::jsonschema::json_schema<json> &module_schema() {
    static const auto compiled = [] {
        const std::string text(
            reinterpret_cast<const char *>(kernel_module_schema),
            kernel_module_schema_size);
        return jsoncons::jsonschema::make_json_schema(json::parse(text));
    }();
    return compiled;
}

const jsoncons::jsonschema::json_schema<json> &launch_schema() {
    static const auto compiled = [] {
        const std::string text(
            reinterpret_cast<const char *>(kernel_launch_schema),
            kernel_launch_schema_size);
        return jsoncons::jsonschema::make_json_schema(json::parse(text));
    }();
    return compiled;
}

const jsoncons::jsonschema::json_schema<json> &program_schema() {
    static const auto compiled = [] {
        const std::string text(
            reinterpret_cast<const char *>(program_schema_data),
            program_schema_data_size);
        return jsoncons::jsonschema::make_json_schema(json::parse(text));
    }();
    return compiled;
}

const jsoncons::jsonschema::json_schema<json> &fixture_schema() {
    static const auto compiled = [] {
        const std::string text(
            reinterpret_cast<const char *>(fixture_schema_data),
            fixture_schema_data_size);
        return jsoncons::jsonschema::make_json_schema(json::parse(text));
    }();
    return compiled;
}

json parse_and_validate(
    const std::filesystem::path &path,
    const jsoncons::jsonschema::json_schema<json> &schema,
    const char *description) {
    std::ifstream input(path);
    if (!input) {
        throw std::invalid_argument(
            std::string("cannot open TTL ") + description + ": " +
            path.string());
    }
    std::ostringstream contents;
    contents << input.rdbuf();
    json root = json::parse(contents.str());
    schema.validate(root);
    return root;
}

void validate_relationships(const json &root) {
    std::map<std::string, bool> tensors;
    for (const auto &tensor : root.at("tensors").array_range()) {
        const std::string name = tensor.at("name").as<std::string>();
        if (!tensors.emplace(name, true).second) {
            throw std::invalid_argument("module tensor names must be unique");
        }
    }

    std::map<std::string, bool> scalars;
    for (const auto &scalar : root.at("scalars").array_range()) {
        const std::string name = scalar.at("name").as<std::string>();
        if (tensors.count(name) != 0 ||
            !scalars.emplace(name, true).second) {
            throw std::invalid_argument(
                "module tensor and scalar names must be unique");
        }
    }

    std::map<std::string, bool> parameters;
    for (const auto &argument : root.at("arguments").array_range()) {
        const std::string parameter =
            argument.at("parameter").as<std::string>();
        if (!parameters.emplace(parameter, true).second) {
            throw std::invalid_argument(
                "launch argument parameter names must be unique");
        }
        const std::string kind = argument.at("kind").as<std::string>();
        if (kind == "literal") continue;
        const std::string name = argument.at("name").as<std::string>();
        const bool known = kind == "tensor"
            ? tensors.count(name) != 0 : scalars.count(name) != 0;
        if (!known) {
            throw std::invalid_argument(
                kind == "tensor" ? "tensor argument name is unknown"
                                  : "scalar argument name is unknown");
        }
    }
}

void validate_program_relationships(const json &root) {
    std::map<std::string, bool> defined;
    for (const auto &input : root.at("inputs").array_range()) {
        const std::string name = input.at("name").as<std::string>();
        if (!defined.emplace(name, true).second) {
            throw std::invalid_argument("program input names must be unique");
        }
    }
    std::map<std::string, bool> outputs;
    for (const auto &output : root.at("outputs").array_range()) {
        const std::string name = output.at("name").as<std::string>();
        if (defined.count(name) != 0 || !outputs.emplace(name, true).second) {
            throw std::invalid_argument(
                "program output names must be unique and distinct from inputs");
        }
    }
    for (const auto &call : root.at("calls").array_range()) {
        for (const auto &argument : call.at("arguments").array_range()) {
            const std::string name = argument.as<std::string>();
            if (defined.count(name) == 0) {
                throw std::invalid_argument(
                    "program call uses a value before its definition: " + name);
            }
        }
        for (const auto &result : call.at("results").array_range()) {
            const std::string name = result.as<std::string>();
            if (!defined.emplace(name, true).second) {
                throw std::invalid_argument(
                    "program call result is not a fresh value: " + name);
            }
        }
    }
    for (const auto &[name, present] : outputs) {
        (void)present;
        if (defined.count(name) == 0) {
            throw std::invalid_argument(
                "program output is not produced: " + name);
        }
    }

    std::set<std::string> needed;
    for (const auto &[name, present] : outputs) {
        (void)present;
        needed.insert(name);
    }
    const auto &calls = root.at("calls");
    for (size_t index = calls.size(); index != 0; --index) {
        const auto &call = calls.at(index - 1);
        bool reachable = false;
        for (const auto &result : call.at("results").array_range()) {
            reachable |= needed.erase(result.as<std::string>()) != 0;
        }
        if (!reachable) {
            throw std::invalid_argument(
                "program contains a call whose results do not reach an output");
        }
        for (const auto &argument : call.at("arguments").array_range()) {
            needed.insert(argument.as<std::string>());
        }
    }
    for (const auto &input : root.at("inputs").array_range()) {
        const std::string name = input.at("name").as<std::string>();
        if (needed.erase(name) == 0) {
            throw std::invalid_argument("program input is unused: " + name);
        }
    }
    if (!needed.empty()) {
        throw std::invalid_argument(
            "program output dependency is not a declared input");
    }
}

}  // namespace

json parse_and_validate_manifest(const std::filesystem::path &path) {
    try {
        json root = parse_and_validate(
            path, module_schema(), "module manifest");
        validate_relationships(root);
        return root;
    } catch (const std::invalid_argument &) {
        throw;
    } catch (const std::exception &error) {
        throw std::invalid_argument(
            "invalid TTL module manifest " + path.string() + ": " +
            error.what());
    }
}

json parse_and_validate_launch(const std::filesystem::path &path) {
    try {
        return parse_and_validate(path, launch_schema(), "private launch description");
    } catch (const std::invalid_argument &) {
        throw;
    } catch (const std::exception &error) {
        throw std::invalid_argument(
            "invalid TTL private launch description " + path.string() +
            ": " + error.what());
    }
}

json parse_and_validate_program(const std::filesystem::path &path) {
    try {
        json root = parse_and_validate(
            path, program_schema(), "program manifest");
        validate_program_relationships(root);
        return root;
    } catch (const std::invalid_argument &) {
        throw;
    } catch (const std::exception &error) {
        throw std::invalid_argument(
            "invalid TTL program manifest " + path.string() + ": " +
            error.what());
    }
}

json parse_and_validate_fixture(const std::filesystem::path &path) {
    try {
        return parse_and_validate(path, fixture_schema(), "test fixture");
    } catch (const std::invalid_argument &) {
        throw;
    } catch (const std::exception &error) {
        throw std::invalid_argument(
            "invalid TTL test fixture " + path.string() + ": " +
            error.what());
    }
}

}  // namespace ttl_internal
