"""Print compiler or linker flags for the installed LibTorch wheel."""

from __future__ import annotations

import shlex
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="Failed to initialize NumPy")
import torch


def main() -> None:
    if (len(sys.argv) != 3 or sys.argv[1] not in {"compile", "link"} or
            sys.argv[2] not in {"libtorch_cuda", "libtorch_mps"}):
        raise SystemExit(
            "usage: torch_flags.py {compile|link} "
            "{libtorch_cuda|libtorch_mps}")
    backend = sys.argv[2]

    torch_root = Path(torch.__file__).resolve().parent
    include = torch_root / "include"
    library = torch_root / "lib"
    if sys.argv[1] == "compile":
        flags = ["-isystem", str(include), "-isystem",
                 str(include / "torch/csrc/api/include")]
        if backend == "libtorch_cuda":
            flags.append(
                f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}")
        cuda_include_candidates = [
            Path("/usr/local/cuda/include"),
            torch_root.parent / "nvidia/cuda_runtime/include",
        ]
        if backend == "libtorch_cuda":
            for cuda_include in cuda_include_candidates:
                if (cuda_include / "cuda_runtime.h").is_file():
                    flags.extend(["-isystem", str(cuda_include)])
                    break
            else:
                raise SystemExit("could not locate cuda_runtime.h")
    else:
        flags = [f"-L{library}", f"-Wl,-rpath,{library}", "-ltorch",
                 "-ltorch_cpu", "-lc10"]
        if backend == "libtorch_cuda":
            flags[2:2] = ["-Wl,--no-as-needed"]
            flags.extend(["-ltorch_cuda", "-lc10_cuda", "-Wl,--as-needed",
                          "-lcuda"])
        else:
            flags.extend(["-framework", "Metal", "-framework", "Foundation"])
    print(shlex.join(flags))


if __name__ == "__main__":
    main()
