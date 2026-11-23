"""TileLang assertion compatibility must preserve checks and fail closed."""

import json
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tool"))

from metal_assertions import lower_assertions  # noqa: E402


SOURCE = r'''#include <metal_stdlib>
using namespace metal;
[[kernel, max_total_threads_per_threadgroup(1)]] void checked_kernel(
    const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    uint3 blockIdx [[threadgroup_position_in_grid]]) {
  if (!((input[0] != 0.0f))) {
    const char* __tvm_assert_parts[1] = {"numerical domain failure"};
    TVMFFIErrorSetRaisedFromCStrParts("RuntimeError", __tvm_assert_parts, 1);
    return -1;
  }
  output[0] = 1.0f / input[0];
}'''


class MetalAssertionsTest(unittest.TestCase):
    def test_known_tilelang_form_uses_a_trailing_error_buffer(self):
        result, index = lower_assertions(SOURCE, "checked_kernel", 2)
        self.assertEqual(index, 2)
        self.assertIn("if (!((input[0] != 0.0f)))", result)
        self.assertIn(
            'TORCH_REPORT_ERROR(__ttl_errors, "numerical domain failure")',
            result)
        self.assertIn("__ttl_errors [[buffer(2)]]", result)
        self.assertIn("return;", result)
        self.assertNotIn("return -1", result)
        self.assertNotIn("TVMFFI", result)

    def test_multiple_known_assertions_are_all_preserved(self):
        second = SOURCE.replace(
            "  output[0] = 1.0f / input[0];",
            """  if (!((input[0] < 4.0f))) {
    const char* __tvm_assert_parts[1] = {"gather index out of bounds"};
    TVMFFIErrorSetRaisedFromCStrParts("RuntimeError", __tvm_assert_parts, 1);
    return -1;
  }
  output[0] = 1.0f / input[0];""")
        result, _ = lower_assertions(second, "checked_kernel", 2)
        self.assertEqual(result.count("TORCH_REPORT_ERROR"), 2)
        self.assertNotIn("__tvm_assert_parts", result)

    def test_unchecked_source_is_byte_identical(self):
        source = "kernel void plain() {}\n"
        self.assertEqual(
            lower_assertions(source, "plain", 0), (source, None))

    def test_escaped_literal_is_preserved(self):
        escaped = json.dumps('bad "value"')[1:-1]
        result, _ = lower_assertions(
            SOURCE.replace("numerical domain failure", escaped),
            "checked_kernel", 2)
        self.assertIn('TORCH_REPORT_ERROR(__ttl_errors, "bad \\"value\\"")',
                      result)

    def test_unsupported_shapes_fail_closed(self):
        unsupported = (
            SOURCE.replace("[1]", "[2]"),
            SOURCE.replace(", 1);", ", 2);"),
            SOURCE.replace('"RuntimeError"', '"ValueError"'),
            SOURCE.replace("return -1;", "return;"),
            SOURCE + "\nthreadgroup_barrier(mem_flags::mem_threadgroup);",
            SOURCE + "\nsimdgroup_barrier(mem_flags::mem_none);",
            SOURCE + "\nassert(false);",
            "kernel void checked_kernel() { assert(false); }",
            SOURCE.replace("__tvm_assert_parts", "__ttl_errors"),
        )
        for source in unsupported:
            with self.subTest(source=source[-100:]):
                with self.assertRaises(ValueError):
                    lower_assertions(source, "checked_kernel", 2)

    def test_unavailable_buffer_index_fails_closed(self):
        for count in (-1, 31):
            with self.subTest(count=count):
                with self.assertRaisesRegex(ValueError, "binding is unavailable"):
                    lower_assertions(SOURCE, "checked_kernel", count)


if __name__ == "__main__":
    unittest.main()
