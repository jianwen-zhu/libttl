"""TileLang CUDA assertion compatibility preserves checks fail-closed."""

import json
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tool"))

from cuda_assertions import lower_assertions  # noqa: E402


SOURCE = r'''extern "C" __global__ void checked_kernel(
    const float* input, float* output) {
  if (!((input[0] != 0.0f))) {
    const char* __tvm_assert_parts[1] = {"division by zero"};
    TVMFFIErrorSetRaisedFromCStrParts(
        "RuntimeError", __tvm_assert_parts, 1);
    return -1;
  }
  output[0] = 1.0f / input[0];
}'''


class CudaAssertionsTest(unittest.TestCase):
    def test_known_tilelang_form_becomes_native_device_assertion(self):
        result, count = lower_assertions(SOURCE)
        self.assertEqual(count, 1)
        self.assertTrue(result.startswith("#include <assert.h>\n"))
        self.assertIn('assert(0 && "division by zero");', result)
        self.assertIn("return;", result)
        self.assertNotIn("return -1", result)
        self.assertNotIn("TVMFFI", result)

    def test_multiple_known_assertions_are_all_preserved(self):
        second = SOURCE.replace(
            "  output[0] = 1.0f / input[0];",
            SOURCE[SOURCE.index("  if (!"):SOURCE.index("  output[0]")]
            .replace("division by zero", "gather index out of bounds")
            + "  output[0] = 1.0f / input[0];")
        result, count = lower_assertions(second)
        self.assertEqual(count, 2)
        self.assertEqual(result.count("assert(0 &&"), 2)

    def test_unchecked_source_is_byte_identical(self):
        source = 'extern "C" __global__ void plain() {}\n'
        self.assertEqual(lower_assertions(source), (source, 0))

    def test_escaped_literal_is_preserved(self):
        escaped = json.dumps('bad "value"')[1:-1]
        result, _ = lower_assertions(
            SOURCE.replace("division by zero", escaped))
        self.assertIn('assert(0 && "bad \\"value\\"");', result)

    def test_unsupported_shapes_fail_closed(self):
        unsupported = (
            SOURCE.replace("[1]", "[2]"),
            SOURCE.replace(", 1);", ", 2);"),
            SOURCE.replace('"RuntimeError"', '"ValueError"'),
            SOURCE.replace("return -1;", "return;"),
            SOURCE.replace("__tvm_assert_parts", "other_parts"),
            "#define NDEBUG\n" + SOURCE,
        )
        for source in unsupported:
            with self.subTest(source=source[-100:]):
                with self.assertRaises(ValueError):
                    lower_assertions(source)


if __name__ == "__main__":
    unittest.main()
