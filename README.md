# libttl

libttl is the Tiny Tensor Library: a small, opaque C interface for creating
tensors, selecting accelerator devices, ordering work on explicit streams, and
executing compiled TTL programs. LibTorch, CUDA, and Metal implementation types
remain private.

The installed public header is deliberately flat:

```c
#include <libttl.h>
```

libttl currently provides LibTorch-backed CUDA and MPS implementations. It also
provides `ttl-cuda`, `ttl-metal`, and `ttl-tilelang`, which package native
final functions for the runtime. The TTL language and its compiler are separate
from this repository. The public interface, JSON contracts, and shared contract
validator are backend-independent; each implementation supplies its own private
build requirements and runtime types.

## Build

The selected Python must be Python 3.10 or newer and provide a compatible
PyTorch installation. Bootstrap downloads checksum-pinned jsoncons 1.9.0
headers into external state:

```sh
make bootstrap PYTHON=/path/to/python
make build TTL_BACKEND=libtorch_mps PYTHON=/path/to/python
make api-check TTL_BACKEND=libtorch_mps PYTHON=/path/to/python
```

Use `TTL_BACKEND=libtorch_cuda` for CUDA. Build products and dependencies are
stored under `~/.cache/libttl` by default; set `TTL_STATE_ROOT` or
`TTL_BUILD_DIR` to select another location.

Run the standalone host example against the selected build:

```sh
make examples-check TTL_BACKEND=libtorch_mps PYTHON=/path/to/python
```

It creates two tensors through the public C API and compares them without
using backend-private types. Installations also provide `libttl.pc`, so C
consumers can obtain compile and link flags with `pkg-config`.

Checks that do not require an accelerator are available separately:

```sh
make static-check PYTHON=/path/to/python
make tools-check PYTHON=/path/to/python
make package-check TTL_BACKEND=libtorch_mps PYTHON=/path/to/python
```

`package-check` stages an installation, verifies the complete exported C
symbol set, and links and runs a consumer using only the staged header and
library.

`api-check` exercises every public function. It builds small native modules
under `TTL_BUILD_DIR`, then covers host tensors, device transfers, streams,
timers, final-function launches, reusable program execution, and documented
error paths. The CUDA profile also stresses repeated device-scalar launches
across independent streams. An accelerator must be available for this target.

Maintainers use `make archive-check` on a clean worktree to validate exactly
the files that `git archive` will publish. `make release-check` combines that
source-archive gate with the tool, example, package, ABI, and live API checks
for the selected backend. Snapshot-based remote runners that do not preserve
`.git`, including CloudMake, run `make backend-check` after the archive gate has
passed in the local Git worktree.

## Contracts and documentation

libttl owns four versioned JSON contracts under `schema/`:

- `ttl-program-v1.schema.json` describes executable programs.
- `ttl-fixture-v1.schema.json` describes concrete inputs and expected results.
- `ttl-kernel-module-v1.schema.json` describes a final function's public
  signature.
- `ttl-kernel-launch-v1.schema.json` describes its private native launch data.

The corresponding documentation is:

- [Public C API](docs/TTL_API.md)
- [Program format](docs/TTL_PROGRAM.md)
- [Fixture format](docs/TTL_FIXTURE.md)
- [Kernel module format](docs/KERNEL_MODULE.md)
- [Backend compatibility](docs/COMPATIBILITY.md)
- [Release gates](docs/RELEASING.md)

Non-normative proposals live under `docs/design/` and are not part of the
implemented API or contracts.

Release history is recorded in [CHANGELOG.md](CHANGELOG.md). libttl is licensed
under the [MIT License](LICENSE). Dependency notices are in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
