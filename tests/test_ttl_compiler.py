import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).parents[1] / "tool"))

from ttl_compiler import (  # noqa: E402
    emit_cuda_module, kernel_parameter_names, load_manifest,
    metal_parameter_names, normalize_launch_configuration,
    source_launch_configuration, validate_arguments,
)


ROOT = Path(__file__).parents[1]


class TtcCompilerTest(unittest.TestCase):
    def manifest(self):
        return {
            "tensors": [
                {"name": "output", "dtype": "float32", "access": "write",
                 "shape": [64]},
                {"name": "input", "dtype": "float32", "access": "read",
                 "shape": [64]},
            ],
            "scalars": [{"name": "factor", "dtype": "float32"}],
            "arguments": [
                {"parameter": "output", "kind": "tensor", "name": "output"},
                {"parameter": "input", "kind": "tensor", "name": "input"},
                {"parameter": "factor", "kind": "scalar_value", "name": "factor"},
            ],
        }

    def test_authored_order_matches_native_signature(self):
        source = """// ttl-launch: {"grid":[1],"block":[64]}
extern "C" __global__ void scale_kernel(
    float *output, const float *input, float factor) {}
"""
        _, parameters = kernel_parameter_names(source, "scale_kernel")
        validate_arguments(self.manifest(), parameters)

    def test_device_scalar_is_valid(self):
        manifest = self.manifest()
        manifest["arguments"][-1]["kind"] = "scalar_device"
        validate_arguments(manifest, ["output", "input", "factor"])

    def test_wrong_parameter_order_is_rejected(self):
        manifest = self.manifest()
        manifest["arguments"][0], manifest["arguments"][1] = (
            manifest["arguments"][1], manifest["arguments"][0])
        with self.assertRaisesRegex(ValueError, "parameter order"):
            validate_arguments(manifest, ["output", "input", "factor"])

    def test_all_checked_in_manifests_are_valid(self):
        manifests = sorted((ROOT / "tests" / "fixtures").glob("*/*/module.json"))
        self.assertGreaterEqual(len(manifests), 2)
        def accepted(command, **_kwargs):
            path = Path(command[-1])
            return subprocess.CompletedProcess(
                command, 0, stdout=path.read_text(), stderr="")

        with mock.patch("ttl_compiler.subprocess.run", side_effect=accepted):
            for path in manifests:
                with self.subTest(path=path):
                    load_manifest(path)

    def test_schema_document_identifies_version_one(self):
        schema = json.loads(
            (ROOT / "schema" / "ttl-kernel-module-v1.schema.json").read_text())
        self.assertEqual(schema["$schema"],
                         "https://json-schema.org/draft/2020-12/schema")
        self.assertEqual(schema["properties"]["schema_version"]["const"], 1)

    def test_tensor_manifests_do_not_impose_an_artificial_rank_limit(self):
        module_schema = json.loads(
            (ROOT / "schema" / "ttl-kernel-module-v1.schema.json").read_text())
        program_schema = json.loads(
            (ROOT / "schema" / "ttl-program-v1.schema.json").read_text())
        self.assertNotIn(
            "maxItems", module_schema["$defs"]["tensor"]["properties"]["shape"])
        self.assertNotIn(
            "maxItems", program_schema["$defs"]["tensor"]["properties"]["shape"])

    def test_metal_signature_ignores_builtin_parameters(self):
        source = """// ttl-launch: {"grid":[1],"block":[64]}
kernel void scale_kernel(
    device float *output [[buffer(0)]],
    device const float *input [[buffer(1)]],
    constant float &factor [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {}
"""
        _, parameters = metal_parameter_names(source, "scale_kernel")
        self.assertEqual(parameters, ["output", "input", "factor"])
        validate_arguments(self.manifest(), parameters)
        self.assertEqual(source_launch_configuration(source)["block"], [64])

    def test_tilelang_metal_signature_is_accepted(self):
        source = """[[kernel, max_total_threads_per_threadgroup(64)]]
void scale_kernel(
    device float *output [[buffer(0)]],
    device const float *input [[buffer(1)]],
    constant float &factor [[buffer(2)]],
    uint3 blockIdx [[threadgroup_position_in_grid]]) {}
"""
        signature, parameters = metal_parameter_names(source, "scale_kernel")
        self.assertIn("max_total_threads_per_threadgroup", signature)
        self.assertEqual(parameters, ["output", "input", "factor"])
        validate_arguments(self.manifest(), parameters)

    def test_compiler_launch_is_normalized(self):
        launch = normalize_launch_configuration({"grid": [4], "block": [64]})
        self.assertEqual(launch["shared_bytes"], 0)

    def test_checked_in_native_signatures_match_shared_manifests(self):
        for suffix, parser in (("*.cu", kernel_parameter_names),
                               ("*.metal", metal_parameter_names)):
            for source in sorted(
                    (ROOT / "tests" / "fixtures").glob(f"*/02_device/{suffix}")):
                with self.subTest(source=source):
                    manifest = json.loads(
                        source.with_name("module.json").read_text())
                    text = source.read_text()
                    source_launch_configuration(text)
                    _, parameters = parser(text, manifest["kernel"]["name"])
                    validate_arguments(manifest, parameters)

    def test_manifest_is_copied_byte_for_byte(self):
        manifest = {
            "schema_version": 1,
            "name": "test.copy",
            "kernel": {"name": "noop"},
            "tensors": [], "scalars": [], "arguments": [],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "noop.cu"
            authored = root / "authored.json"
            output = root / "out"
            source.write_text(
                '// ttl-launch: {"grid":[1],"block":[1]}\n'
                'extern "C" __global__ void noop() {}\n')
            authored.write_text(json.dumps(manifest, indent=3) + "\n")
            with mock.patch("ttl_compiler.load_manifest",
                            return_value=manifest), \
                    mock.patch("ttl_compiler.subprocess.run"):
                emit_cuda_module(
                    source=source, manifest_path=authored, output=output,
                    arch="sm_75", compiler_name="test")
            self.assertEqual(authored.read_bytes(),
                             (output / "module.json").read_bytes())
            launch = json.loads((output / "launch.json").read_text())
            self.assertEqual(launch["backend"], "libtorch_cuda")
            self.assertEqual(launch["artifact"]["file"], "kernel.cubin")

    def test_shared_checker_rejection_is_reported(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "module.json"
            path.write_text("{}\n")
            error = subprocess.CalledProcessError(
                1, ["ttl-contract-check", str(path)],
                stderr="manifest does not match schema")
            with mock.patch("ttl_compiler.subprocess.run", side_effect=error):
                with self.assertRaisesRegex(ValueError, "does not match schema"):
                    load_manifest(path)


class CompilerCliTest(unittest.TestCase):
    def run_tool(self, tool: str, flag: str) -> str:
        return subprocess.run(
            [sys.executable, str(ROOT / "tool" / tool), flag],
            check=True, text=True, capture_output=True).stdout

    def test_compilers_publish_common_interface_and_version(self):
        for tool in ("ttl-cuda", "ttl-metal", "ttl-tilelang"):
            with self.subTest(tool=tool):
                help_text = self.run_tool(tool, "--help")
                for flag in ("--source", "--manifest", "--output", "--arch"):
                    self.assertIn(flag, help_text)
                version = self.run_tool(tool, "--version")
                self.assertIn("0.1.0", version)
                self.assertIn("compiler interface 1", version)


if __name__ == "__main__":
    unittest.main()
