#include "manifest_json.h"

#include <exception>
#include <iostream>
#include <string>

int main(int argc, char **argv) {
    const bool emit_json = argc == 3 &&
        std::string(argv[1]) == "--emit-json";
    const bool launch = argc == 3 && std::string(argv[1]) == "--launch";
    const bool program = argc == 3 && std::string(argv[1]) == "--program";
    const bool fixture = argc == 3 && std::string(argv[1]) == "--fixture";
    if ((!emit_json && !launch && !program && !fixture && argc != 2) ||
        ((emit_json || launch || program || fixture) && argc != 3)) {
        std::cerr <<
            "usage: ttl-contract-check [--emit-json] MODULE_JSON\n"
            "       ttl-contract-check --launch LAUNCH_JSON\n"
            "       ttl-contract-check --program PROGRAM_JSON\n"
            "       ttl-contract-check --fixture FIXTURE_JSON\n";
        return 2;
    }
    try {
        const auto manifest = fixture
            ? ttl_internal::parse_and_validate_fixture(argv[2])
            : program
            ? ttl_internal::parse_and_validate_program(argv[2])
            : launch ? ttl_internal::parse_and_validate_launch(argv[2])
                     : ttl_internal::parse_and_validate_manifest(
                           argv[emit_json ? 2 : 1]);
        if (emit_json) {
            manifest.dump(std::cout);
            std::cout << '\n';
        }
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "ttl-contract-check: " << error.what() << '\n';
        return 1;
    }
}
