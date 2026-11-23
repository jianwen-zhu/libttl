"""TileLang launch metadata must come from the selected lowered kernel."""

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "tool"))

from tilelang_metadata import cuda_launch_from_device_ir  # noqa: E402


class FakeGlobalVar:
    def __init__(self, name_hint):
        self.name_hint = name_hint


class FakeFunction:
    def __init__(self, symbol=None, dynamic_shared=None):
        self.attrs = {}
        if symbol is not None:
            self.attrs["global_symbol"] = symbol
        if dynamic_shared is not None:
            self.attrs["dyn_shared_memory_buf"] = dynamic_shared


class FakeIntImm:
    def __init__(self, value):
        self.value = value


class FakeDeviceModule:
    def __init__(self, *functions):
        self.functions = {
            FakeGlobalVar(f"function_{index}"): function
            for index, function in enumerate(functions)
        }


class TileLangMetadataTest(unittest.TestCase):
    def extract(self, module, launch=None):
        return cuda_launch_from_device_ir(
            module,
            entrypoint="node_6",
            manifest_symbol="node_6_kernel",
            authored_launch=(
                {"grid": [1], "block": [32], "shared_bytes": 0}
                if launch is None else launch),
        )

    def test_zero_placeholder_is_replaced_by_lowered_allocation(self):
        authored = {"grid": [1], "block": [32], "shared_bytes": 0}
        result = self.extract(FakeDeviceModule(
            FakeFunction("node_6_kernel", FakeIntImm(768))), authored)
        self.assertEqual(result["shared_bytes"], 768)
        self.assertEqual(authored["shared_bytes"], 0)

    def test_missing_attribute_means_no_dynamic_allocation(self):
        result = self.extract(FakeDeviceModule(
            FakeFunction("node_6_kernel")))
        self.assertEqual(result["shared_bytes"], 0)

    def test_omitted_placeholder_is_filled_from_lowered_allocation(self):
        result = self.extract(
            FakeDeviceModule(
                FakeFunction("node_6_kernel", FakeIntImm(768))),
            {"grid": [1], "block": [32]})
        self.assertEqual(result["shared_bytes"], 768)

    def test_matching_explicit_allocation_is_accepted(self):
        result = self.extract(
            FakeDeviceModule(FakeFunction("node_6_kernel", FakeIntImm(768))),
            {"grid": [1], "block": [32], "shared_bytes": 768})
        self.assertEqual(result["shared_bytes"], 768)

    def test_stale_explicit_allocation_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "disagrees with TileLang"):
            self.extract(
                FakeDeviceModule(
                    FakeFunction("node_6_kernel", FakeIntImm(768))),
                {"grid": [1], "block": [32], "shared_bytes": 512})

    def test_manifest_symbol_selects_the_lowered_function(self):
        result = self.extract(FakeDeviceModule(
            FakeFunction("helper_kernel", FakeIntImm(128)),
            FakeFunction("node_6_kernel", FakeIntImm(768))))
        self.assertEqual(result["shared_bytes"], 768)

    def test_missing_or_ambiguous_symbol_is_rejected(self):
        modules = [
            FakeDeviceModule(FakeFunction("other_kernel", FakeIntImm(64))),
            FakeDeviceModule(
                FakeFunction("node_6_kernel", FakeIntImm(64)),
                FakeFunction("node_6_kernel", FakeIntImm(64))),
            FakeDeviceModule(FakeFunction(dynamic_shared=FakeIntImm(64))),
        ]
        for module in modules:
            with self.subTest(module=module):
                with self.assertRaisesRegex(ValueError, "exactly one"):
                    self.extract(module)

    def test_symbolic_negative_and_oversized_allocations_are_rejected(self):
        invalid = [object(), FakeIntImm(-1), FakeIntImm(0x100000000)]
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                        ValueError, "constant integer|nonnegative uint32"):
                    self.extract(FakeDeviceModule(
                        FakeFunction("node_6_kernel", value)))


if __name__ == "__main__":
    unittest.main()
