# Compatibility profile

libttl separates a stable C surface from a selected implementation. The
implemented selections are `TTL_BACKEND=libtorch_cuda` and
`TTL_BACKEND=libtorch_mps`. LibTorch, CUDA, and MPS types remain private.

The build requires Python 3.10 or newer with a compatible PyTorch installation.
The TileLang packager additionally requires the TileLang package. jsoncons
1.9.0 is the sole added C++ dependency; bootstrap installs its checksum-pinned
headers into external state.

## Capabilities

| Capability | `libtorch_cuda` | `libtorch_mps` |
| --- | --- | --- |
| `DYNAMIC_WORKGROUP_MEMORY` | yes | no |
| `INDEPENDENT_STREAMS` | yes | no |
| `NATIVE_GRAPH_EXECUTION` | yes | no |

These values describe the current backend implementations, not an inherent
limit of CUDA or Metal. Programs without native graph execution remain valid:
the backend launches their final functions in topological order.

Modules separately declare their requirements. Applications can compare those
requirements with device capabilities before preparation, and libttl repeats
the check at the API boundary.

## Selection and installation

```sh
make build TTL_BACKEND=libtorch_cuda PYTHON=/path/to/python
make build TTL_BACKEND=libtorch_mps PYTHON=/path/to/python
```

A backend implementation follows
`src/backend_<framework>_<device-family>.cc` (or `.mm`). Consumers may link
directly from `TTL_BUILD_DIR`, or stage an installation with:

```sh
make -C src install BACKEND=libtorch_cuda PYTHON=/path/to/python \
    DESTDIR=/tmp/libttl-package PREFIX=/usr/local
```
