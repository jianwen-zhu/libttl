#!/usr/bin/env python3
"""Package and inspect a real shared-memory TileLang Metal kernel."""

from __future__ import annotations

import argparse
import os
import runpy
import subprocess
import sys
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tool"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    os.environ["TILELANG_DEFAULT_TARGET"] = "metal"
    os.environ["TILELANG_DISABLE_CACHE"] = "1"

    import tilelang

    namespace = runpy.run_path(args.source)
    entrypoint = namespace["TTL_ENTRYPOINT"]
    specializations = namespace["TTL_SPECIALIZATIONS"]
    kernel = namespace[entrypoint]

    compiled = kernel.compile(**specializations)
    expected_source = compiled.get_kernel_source()

    captured_sources: list[str] = []
    original_lower = tilelang.lower

    def capture_lower(*lower_args, **lower_kwargs):
        if lower_kwargs.get("enable_host_codegen") is not False or \
                lower_kwargs.get("enable_device_compile") is not True:
            raise AssertionError(
                "ttl-tilelang must use corrected lazy Metal device codegen")
        result = original_lower(*lower_args, **lower_kwargs)
        captured_sources.append(result.kernel_source)
        return result

    real_run = subprocess.run
    tool_argv = [
        str(ROOT / "tool" / "ttl-tilelang"),
        "--source", str(args.source),
        "--manifest", str(args.manifest),
        "--output", str(args.output),
    ]
    with mock.patch.object(tilelang, "lower", side_effect=capture_lower), \
            mock.patch("subprocess.run", wraps=real_run) as run, \
            mock.patch.object(sys, "argv", tool_argv), \
            mock.patch.dict(os.environ, {"TTL_BACKEND": "libtorch_mps"}):
        runpy.run_path(
            str(ROOT / "tool" / "ttl-tilelang"), run_name="__main__")

    if captured_sources != [expected_source]:
        raise AssertionError(
            "ttl-tilelang Metal source differs from normal TileLang codegen")

    xcrun_commands = [
        call.args[0] for call in run.call_args_list
        if isinstance(call.args[0], list) and call.args[0] and
        Path(call.args[0][0]).name == "xcrun"
    ]
    compile_commands = [
        command for command in xcrun_commands if "-sdk" in command
    ]
    if len(xcrun_commands) != 4 or len(compile_commands) != 2:
        raise AssertionError(
            "Metal packaging must perform two tool lookups and exactly one "
            "metal/metallib compilation pair; observed: " +
            repr(xcrun_commands))
    if not any("metal" in command for command in compile_commands) or \
            not any("metallib" in command for command in compile_commands):
        raise AssertionError(
            "Metal packaging did not invoke both metal and metallib")

    print("TILELANG METAL SOURCE PARITY, ONE METAL COMPILE, AND EXECUTION "
          "PACKAGE PASSED")


if __name__ == "__main__":
    main()
