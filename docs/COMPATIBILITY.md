# Backend compatibility

libttl separates a stable C interface from a selected implementation. The
implemented backend selections are `libtorch_cuda` and `libtorch_mps`.
LibTorch, CUDA, and MPS types remain private.

## Build requirements

Both backends require Python 3.10 or newer with a compatible PyTorch
installation. The CUDA backend also requires CUDA development headers and the
driver library. The MPS backend requires macOS, Xcode, and a PyTorch build with
MPS support. The TileLang packager additionally requires TileLang.

jsoncons 1.9.0 is the only additional C++ dependency. `make bootstrap`
installs its checksum-pinned headers outside the checkout.

## Validated release profiles

The following records successful release-gate executions. It is evidence of
tested configurations, not a restriction on other compatible systems.

| Date | Backend | Hardware | Software profile | Result |
| --- | --- | --- | --- | --- |
| 2026-09-13 | `libtorch_cuda` | NVIDIA T4 on CloudMake | CloudMake CUDA/PyTorch image | Complete backend release checks passed; provenance `d0bc345cc5bc46c087017f7d2923c3ac` |
| 2026-09-13 | `libtorch_mps` | Apple M3 Pro | macOS 26.0.1, Xcode 26.3, Python 3.13.7, PyTorch 2.13.0 | Complete clean-source and backend release checks passed |

Future releases should update this table with their actual validation profiles
instead of treating these versions as permanent minimums or maximums.

## Runtime capabilities

| Capability | `libtorch_cuda` | `libtorch_mps` |
| --- | --- | --- |
| `DYNAMIC_WORKGROUP_MEMORY` | yes | no |
| `INDEPENDENT_STREAMS` | yes | no |
| `NATIVE_GRAPH_EXECUTION` | yes | no |

These values describe the current implementations, not inherent limits of
CUDA or Metal. A program without native graph execution remains valid: the
backend launches its final functions in topological order.

Modules declare their requirements independently. Applications can compare
those requirements with device capabilities before preparation, and libttl
repeats the check at the API boundary.

## Build and installation

Select one backend when building:

```sh
make build TTL_BACKEND=libtorch_cuda PYTHON=/path/to/python
make build TTL_BACKEND=libtorch_mps PYTHON=/path/to/python
```

A backend implementation follows the filename pattern
`src/backend_<framework>_<device-family>.cc`, with `.mm` used when
Objective-C++ is required. Consumers may link from `TTL_BUILD_DIR`, or stage
an installation:

```sh
make -C src install BACKEND=libtorch_cuda PYTHON=/path/to/python \
    DESTDIR=/tmp/libttl-package PREFIX=/usr/local
```

The staged layout is:

```text
usr/local/
  include/libttl.h
  lib/libttl.so
  lib/pkgconfig/libttl.pc
  share/libttl/schema/*.schema.json
  share/doc/libttl/LICENSE
  share/doc/libttl/THIRD_PARTY_NOTICES.md
```

The macOS library uses `@rpath/libttl.so` as its install name and records the C
API major version as its compatibility floor. A consuming executable or
library must supply an RPATH for its chosen installation layout. The installed
library dynamically depends on the LibTorch installation used to build it; the
package does not bundle PyTorch or accelerator runtimes. The installed
`libttl.pc` publishes only the stable C include and link flags; backend-private
compiler flags are not part of the consumer interface.
