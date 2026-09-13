import re
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
FUNCTION = re.compile(r"\b(ttl_[a-z_]+)\s*\(")


class ApiSurfaceTest(unittest.TestCase):
    def test_every_public_function_is_exercised_by_an_api_runner(self):
        declared = set(FUNCTION.findall(
            (ROOT / "include" / "libttl.h").read_text()))
        exercised = set()
        for source in ("api_host_tests.c", "api_device_tests.c"):
            exercised.update(FUNCTION.findall(
                (ROOT / "tests" / source).read_text()))
        self.assertEqual(declared, exercised)
        self.assertEqual(len(declared), 42)


if __name__ == "__main__":
    unittest.main()
