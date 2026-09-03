#!/usr/bin/env python3
"""Embed the canonical TTL JSON Schema in a generated C++ header."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--symbol", default="kernel_module_schema")
    arguments = parser.parse_args()

    data = arguments.input.read_bytes()
    rows = []
    for offset in range(0, len(data), 16):
        rows.append("    " + ", ".join(
            f"0x{byte:02x}" for byte in data[offset:offset + 16]) + ",")
    text = "\n".join([
        f"// Generated from {arguments.input.name}.",
        "#pragma once",
        "#include <cstddef>",
        "namespace ttl_internal {",
        f"inline constexpr unsigned char {arguments.symbol}[] = {{",
        *rows,
        "};",
        (f"inline constexpr std::size_t {arguments.symbol}_size = "
         f"sizeof({arguments.symbol});"),
        "}  // namespace ttl_internal",
        "",
    ])
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(text)


if __name__ == "__main__":
    main()
