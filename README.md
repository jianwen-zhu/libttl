# libttl

`libttl` is the Tiny Tensor Library: a small, opaque C interface for creating
tensors, selecting devices, submitting work to streams, and executing compiled
TTL programs. Backend-specific LibTorch, CUDA, and Metal types remain private.

The installed public surface is deliberately flat:

```c
#include <libttl.h>
```

The library owns two portable JSON contracts:

- `schema/ttl-program-v1.schema.json` describes executable TTL programs.
- `schema/ttl-fixture-v1.schema.json` describes concrete inputs and expected
  results used to check those programs.

Low-level final functions use the module and private-launch schemas in the
same directory. `ttl-cuda`, `ttl-metal`, and `ttl-tilelang` package native
functions for those programs.

## Build

The selected Python must provide the corresponding LibTorch installation:

```sh
make bootstrap
make build TTL_BACKEND=libtorch_mps PYTHON=/path/to/python
make tests TTL_BACKEND=libtorch_mps PYTHON=/path/to/python
```

CUDA uses `TTL_BACKEND=libtorch_cuda`. Build products and the pinned jsoncons
headers are stored outside the checkout under `~/.cache/libttl` by default.

See [`docs/TTL_API.md`](docs/TTL_API.md),
[`docs/TTL_PROGRAM.md`](docs/TTL_PROGRAM.md), and
[`docs/KERNEL_MODULE.md`](docs/KERNEL_MODULE.md) for the contracts.

