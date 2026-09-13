import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]


class BackendBoundaryTest(unittest.TestCase):
    def test_public_and_shared_sources_do_not_depend_on_libtorch(self):
        shared_sources = [
            ROOT / "include" / "libttl.h",
            ROOT / "src" / "manifest_json.h",
            ROOT / "src" / "manifest_json.cc",
            ROOT / "src" / "manifest_check.cc",
        ]
        private_names = ("ATen", "c10::", "torch/", "libtorch")
        for source in shared_sources:
            text = source.read_text()
            for private_name in private_names:
                self.assertNotIn(private_name, text, source)

    def test_libtorch_build_requirements_are_backend_scoped(self):
        common_makefile = (ROOT / "src" / "Makefile").read_text()
        libtorch_makefile = (ROOT / "src" / "backend_libtorch.mk").read_text()
        self.assertNotIn("backend_libtorch_impl.inc", common_makefile)
        self.assertNotIn("torch_flags.py", common_makefile)
        self.assertIn("backend_libtorch_impl.inc", libtorch_makefile)
        self.assertIn("torch_flags.py", libtorch_makefile)


if __name__ == "__main__":
    unittest.main()
