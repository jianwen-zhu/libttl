"""Shared TTL kernel-module emission used by compiler front ends."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import argparse
import tempfile
from pathlib import Path
from typing import Any


COMPILER_INTERFACE_VERSION = 1
PROJECT_VERSION = (Path(__file__).resolve().parents[1] / "VERSION").read_text().strip()
DEFAULT_XCODE_DEVELOPER_DIR = Path(
    "/Applications/Xcode.app/Contents/Developer")


def add_common_arguments(parser: argparse.ArgumentParser, program: str) -> None:
    parser.add_argument(
        "--version", action="version",
        version=(f"{program} {PROJECT_VERSION} "
                 f"(TTL compiler interface {COMPILER_INTERFACE_VERSION})"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arch", default="sm_75")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def xcode_environment() -> dict[str, str]:
    """Find full Xcode without changing the host's selected developer tools."""
    environment = os.environ.copy()
    if environment.get("DEVELOPER_DIR"):
        return environment

    selected = subprocess.run(
        ["xcode-select", "-p"], text=True, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL)
    candidates = []
    if selected.returncode == 0:
        candidates.append(Path(selected.stdout.strip()))
    candidates.append(DEFAULT_XCODE_DEVELOPER_DIR)
    for candidate in candidates:
        if (candidate / "usr/bin/xcodebuild").is_file() and \
                candidate.name != "CommandLineTools":
            environment["DEVELOPER_DIR"] = str(candidate)
            break
    return environment


def contract_checker() -> str:
    checker = os.environ.get("TTL_CONTRACT_CHECKER")
    if checker is None:
        build_root = os.environ.get("TTL_BUILD_DIR")
        if build_root is None:
            state_root = ("/content/.libttl" if Path("/content").is_dir()
                          else str(Path.home() / ".cache/libttl"))
            build_root = str(Path(state_root) / "build")
        checker = str(Path(build_root) / "ttl-contract-check")
    return checker


def run_checker(arguments: list[str], description: str) -> str:
    checker = contract_checker()
    try:
        completed = subprocess.run(
            [checker, *arguments], check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as error:
        raise RuntimeError(
            "ttl-contract-check is missing; run make contract-check") \
            from error
    except subprocess.CalledProcessError as error:
        raise ValueError(error.stderr.strip()) from error
    return completed.stdout


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(run_checker(["--emit-json", str(path)], "manifest"))


def kernel_parameter_names(source: str, name: str) -> tuple[str, list[str]]:
    patterns = [
        rf'(?:extern\s+"C"\s+)?__global__\s+void\s+'
        rf'(?:__launch_bounds__\([^)]*\)\s+)?{re.escape(name)}\s*\(([^)]*)\)',
        rf'(?:extern\s+"C"\s+)?__global__\s+'
        rf'(?:__launch_bounds__\([^)]*\)\s+)?void\s+'
        rf'{re.escape(name)}\s*\(([^)]*)\)',
    ]
    match = next((candidate for pattern in patterns
                  if (candidate := re.search(pattern, source, re.DOTALL))), None)
    if not match:
        raise ValueError(f"cannot find CUDA kernel {name!r}")
    signature = " ".join(match.group(0).split())
    parameters = match.group(1).strip()
    if not parameters:
        return signature, []
    names = []
    for parameter in parameters.split(","):
        name_match = re.search(r"([A-Za-z_]\w*)\s*$", parameter.strip())
        if not name_match:
            raise ValueError(f"cannot parse CUDA parameter: {parameter}")
        names.append(name_match.group(1))
    return signature, names


def validate_arguments(
    manifest: dict[str, Any], parameter_names: list[str]
) -> None:
    bindings = manifest["arguments"]
    authored_parameters = [binding["parameter"] for binding in bindings]
    require(authored_parameters == parameter_names,
            "manifest arguments must match native kernel parameter order exactly")


def normalize_launch_configuration(launch: dict[str, Any]) -> dict[str, Any]:
    require(isinstance(launch, dict), "ttl-launch value must be an object")
    require(set(launch) <= {"grid", "block", "shared_bytes"},
            "ttl-launch contains an unknown field")
    for field in ("grid", "block"):
        geometry = launch.get(field)
        require(isinstance(geometry, list) and 1 <= len(geometry) <= 3,
                f"ttl-launch {field} must contain one to three dimensions")
        require(all(isinstance(value, int) and 1 <= value <= 0xFFFFFFFF
                    for value in geometry),
                f"ttl-launch {field} dimensions must be positive uint32 values")
    shared_bytes = launch.get("shared_bytes", 0)
    require(isinstance(shared_bytes, int) and 0 <= shared_bytes <= 0xFFFFFFFF,
            "ttl-launch shared_bytes must be a uint32 value")
    return {"grid": launch["grid"], "block": launch["block"],
            "shared_bytes": shared_bytes}


def source_launch_configuration(source: str) -> dict[str, Any]:
    matches = re.findall(
        r"^\s*//\s*ttl-launch:\s*(\{.*\})\s*$", source, re.MULTILINE)
    require(len(matches) == 1,
            "native source requires exactly one // ttl-launch: {...} line")
    try:
        launch = json.loads(matches[0])
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid ttl-launch JSON: {error}") from error
    return normalize_launch_configuration(launch)


def metal_parameter_names(source: str, name: str) -> tuple[str, list[str]]:
    patterns = [
        rf"\bkernel\s+void\s+{re.escape(name)}\s*\(",
        rf"\[\[\s*kernel(?:\s*,[^]]+)?\]\]\s+void\s+"
        rf"{re.escape(name)}\s*\(",
    ]
    match = next((candidate for pattern in patterns
                  if (candidate := re.search(pattern, source))), None)
    if not match:
        raise ValueError(f"cannot find Metal kernel {name!r}")
    opening = match.end() - 1
    depth = 1
    closing = opening + 1
    while closing < len(source) and depth != 0:
        if source[closing] == "(":
            depth += 1
        elif source[closing] == ")":
            depth -= 1
        closing += 1
    require(depth == 0, f"unterminated Metal kernel signature for {name!r}")
    signature = " ".join(source[match.start():closing].split())
    parameters = source[opening + 1:closing - 1].strip()
    if not parameters:
        return signature, []
    names: list[str] = []
    expected_index = 0
    for parameter in parameters.split(","):
        index_match = re.search(r"\[\[\s*buffer\((\d+)\)\s*\]\]", parameter)
        if index_match is None:
            require("[[" in parameter and "]]" in parameter,
                    "Metal non-buffer parameters require a built-in attribute")
            continue
        require(int(index_match.group(1)) == expected_index,
                "Metal buffer indices must be dense and match parameter order")
        declaration = re.sub(r"\[\[.*?\]\]", "", parameter).strip()
        name_match = re.search(r"([A-Za-z_]\w*)\s*$", declaration)
        if not name_match:
            raise ValueError(f"cannot parse Metal parameter: {parameter}")
        names.append(name_match.group(1))
        expected_index += 1
    return signature, names


def emit_private_launch(
    *, output: Path, backend: str, artifact_format: str,
    artifact_file: str, kernel_name: str, launch: dict[str, Any],
) -> None:
    private = {
        "schema_version": 1,
        "backend": backend,
        "artifact": {"format": artifact_format, "file": artifact_file},
        "kernel": {"name": kernel_name, **launch},
    }
    path = output / "launch.json"
    path.write_text(json.dumps(private, indent=2) + "\n")
    run_checker(["--launch", str(path)], "private launch description")


def copy_manifest(manifest_path: Path, output: Path) -> None:
    destination_manifest = output / "module.json"
    if manifest_path.resolve() != destination_manifest.resolve():
        shutil.copyfile(manifest_path, destination_manifest)


def emit_cuda_module(
    *, source: Path, manifest_path: Path, output: Path, arch: str,
    compiler_name: str, includes: list[Path] | None = None,
    nvcc_args: list[str] | None = None,
    launch: dict[str, Any] | None = None,
) -> None:
    manifest = load_manifest(manifest_path)
    kernel = manifest["kernel"]
    name = kernel["name"]
    source_text = source.read_text()
    launch = (normalize_launch_configuration(launch) if launch is not None
              else source_launch_configuration(source_text))
    artifact_file = "kernel.cubin"
    signature, parameters = kernel_parameter_names(source_text, name)
    validate_arguments(manifest, parameters)

    output.mkdir(parents=True, exist_ok=True)
    artifact = output / artifact_file
    command = [
        shutil.which("nvcc") or "nvcc", "-std=c++17", "-O2", "--cubin",
        "-gencode", f"arch=compute_{arch.removeprefix('sm_')},code={arch}",
    ]
    for include in includes or []:
        command.extend(["-I", str(include)])
    command.extend(nvcc_args or [])
    command.extend([str(source), "-o", str(artifact)])
    subprocess.run(command, check=True)
    copy_manifest(manifest_path, output)
    emit_private_launch(
        output=output, backend="libtorch_cuda", artifact_format="cubin",
        artifact_file=artifact_file, kernel_name=name, launch=launch)
    print(f"[{compiler_name}] kernel: {signature}")
    print(f"[{compiler_name}] module: {output}")


def emit_metal_module(
    *, source: Path, manifest_path: Path, output: Path,
    compiler_name: str = "ttl-metal",
    launch: dict[str, Any] | None = None,
) -> None:
    manifest = load_manifest(manifest_path)
    name = manifest["kernel"]["name"]
    source_text = source.read_text()
    launch = (normalize_launch_configuration(launch) if launch is not None
              else source_launch_configuration(source_text))
    signature, parameters = metal_parameter_names(source_text, name)
    validate_arguments(manifest, parameters)

    xcrun = shutil.which("xcrun")
    require(xcrun is not None,
            "xcrun is missing; install Xcode and its Metal toolchain")
    environment = xcode_environment()
    for tool in ("metal", "metallib"):
        located = subprocess.run(
            [xcrun, "--find", tool], text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment)
        require(located.returncode == 0,
                f"{tool} is missing; rerun the libtorch_mps bootstrap")

    output.mkdir(parents=True, exist_ok=True)
    artifact_file = "kernel.metallib"
    artifact = output / artifact_file
    with tempfile.TemporaryDirectory(prefix="ttl-metal-") as temporary:
        air = Path(temporary) / "kernel.air"
        module_cache = Path(temporary) / "module-cache"
        module_cache.mkdir()
        subprocess.run(
            [xcrun, "-sdk", "macosx", "metal", "-c", str(source),
             f"-fmodules-cache-path={module_cache}",
             "-o", str(air)], check=True, env=environment)
        subprocess.run(
            [xcrun, "-sdk", "macosx", "metallib", str(air),
             "-o", str(artifact)], check=True, env=environment)
    copy_manifest(manifest_path, output)
    emit_private_launch(
        output=output, backend="libtorch_mps", artifact_format="metallib",
        artifact_file=artifact_file, kernel_name=name, launch=launch)
    print(f"[{compiler_name}] kernel: {signature}")
    print(f"[{compiler_name}] module: {output}")
