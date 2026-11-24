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
        self.assertIn("BACKEND_CXX_STANDARD ?= c++17", common_makefile)
        self.assertIn("-std=$(BACKEND_CXX_STANDARD)", common_makefile)
        self.assertIn("BACKEND_CXX_STANDARD := c++20", libtorch_makefile)

    def test_contract_checker_retains_backend_independent_standard(self):
        common_makefile = (ROOT / "src" / "Makefile").read_text()
        self.assertIn("$(CXX) -std=c++17 -O2 -Wall -Wextra", common_makefile)

    def test_cuda_programs_share_hardlinked_native_modules(self):
        implementation = (
            ROOT / "src" / "backend_libtorch_impl.inc").read_text()
        self.assertIn(
            "std::shared_ptr<cuda_module_handle> module", implementation)
        self.assertIn("std::filesystem::equivalent(", implementation)
        self.assertIn("native_module = loaded_module", implementation)
        self.assertIn("cuModuleUnload(value)", implementation)

    def test_tilelang_source_builder_is_backend_specific(self):
        packager = (ROOT / "tool" / "ttl-tilelang").read_text()
        self.assertIn(
            'enable_device_compile=(backend == "libtorch_mps")', packager)


if __name__ == "__main__":
    unittest.main()
