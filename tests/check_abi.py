#!/usr/bin/env python3
"""Reject accidental changes to libttl's exported C symbol set."""

from __future__ import annotations

import subprocess
import sys
import platform
from pathlib import Path


def main() -> None:
    library = Path(sys.argv[1])
    expected_path = Path(__file__).with_name("expected_abi_symbols.txt")
    expected = set(expected_path.read_text().split())
    exports_path = Path(__file__).parents[1] / "src" / "libttl.exports"
    declared = set(exports_path.read_text().split())
    if declared != expected:
        missing = sorted(expected - declared)
        extra = sorted(declared - expected)
        raise SystemExit(
            f"TTL export control mismatch: missing={missing}, extra={extra}")
    command = (["nm", "-gU", str(library)] if platform.system() == "Darwin"
               else ["nm", "-D", "--defined-only", str(library)])
    completed = subprocess.run(
        command,
        check=True, text=True, capture_output=True)
    symbols = [
        line.split()[-1].split("@@", 1)[0].removeprefix("_")
        for line in completed.stdout.splitlines() if line.split()
    ]
    actual = set(symbols)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise SystemExit(f"TTL ABI mismatch: missing={missing}, extra={extra}")
    print(f"[abi] {len(actual)} public TTL symbols match")


if __name__ == "__main__":
    main()
