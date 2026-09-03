# TTL program format, version 1

A TTL program describes a pure, static directed acyclic graph of final device
functions. A human-authored teaching program is normally small enough to
review directly. A compiler-emitted program may instead contain an entire
flattened model, including statically unrolled repeated structure. Program
size is not a semantic distinction. A compiler has already made fusion,
tiling, specialization, and scheduling decisions before emitting its final
functions; a TTL backend does not rewrite the program DAG.

```text
program-directory/
  program.json
  functions/
    FUNCTION/
      module.json
      launch.json
      kernel.cubin | kernel.metallib
```

The function directories are low-level TTL kernel packages. `module.json`
describes one final function signature, while `launch.json` and the native
artifact are backend-specific compiler products. Host applications load only
the program directory and do not enumerate those packages.

A program is analogous to a C function and may be used at either boundary. The
current handwritten EinyGPT example loads one transformer-layer program and
repeats it from C because that organization makes the layer structure easy to
teach. A compiler consuming a flattened TTL export may emit the complete
unrolled DAG as one program without changing this format.

The normative structure is
[`schema/ttl-program-v1.schema.json`](../schema/ttl-program-v1.schema.json).
`ttl-contract-check --program PROGRAM_JSON` and `libttl` use the same embedded
schema, jsoncons parser, and relationship checks.

## Semantic contract

`program.json` has exactly five concepts:

- `schema_version`: `1`;
- `name`: a diagnostic program name;
- `inputs`: named tensors with dtype and static shape;
- `outputs`: named tensors with dtype and static shape;
- `calls`: final function name, ordered argument values, and fresh result
  values.

Calls are written in a valid topological order. Every result name is fresh,
arguments must already be defined, and dependencies are inferred from those
value names. Every call and input must contribute to a declared output. There
are no mutable tensors, explicit edges, control flow, backend names, launch
choices, or optimization hints in the format.

Each referenced function is already a non-fusible launch unit for this
program. Its tensor parameters must be purely `read` or `write`; `read_write`
parameters and runtime scalar arguments are outside version 1. Compile-time
typed literals may remain in the low-level function package. Function tensor
order and shape must agree with the corresponding call.

This is deliberately close to CUDA Graph semantics without requiring CUDA.
Independent calls have no ordering relationship merely because one appears
first in the JSON array. The array order is a correct sequential fallback.

## Public lifecycle

Include the program API with:

```c
#include "libttl.h"
```

The application loads a program, binds every external input and output by
name, and prepares it for one opened device:

```c
ttl_program_t *program = NULL;
ttl_program_execution_t *execution = NULL;
ttl_program_binding_t bindings[] = {
    {"input", input_tensor},
    {"output", output_tensor},
};

TTL_CHECK(ttl_program_load("compiled-program", &program));
TTL_CHECK(ttl_program_prepare(
    program, device, bindings, 2, &execution));
TTL_CHECK(ttl_program_launch(execution, stream));
TTL_CHECK(ttl_stream_synchronize(stream));
```

Preparation validates all external tensors, loads and prepares the referenced
functions, and allocates intermediate tensors. The binding array is consumed
during the call; its names need not remain alive afterward. Bound tensors,
however, must remain alive through every launch that uses the execution.
One loaded program may be prepared multiple times with different bindings;
this is the normal representation of repeated layers.

On a backend with `TTL_CAPABILITY_NATIVE_GRAPH_EXECUTION`, preparation may
instantiate the final DAG as a native graph and launch it as one unit. A
backend without that capability launches the same final functions in the
listed topological order. This capability changes overhead, not program
meaning or compatibility.

Synchronize dependent work before destroying the execution or any bound
tensor. Destroying the execution releases prepared functions and intermediate
storage; destroying the program handle releases the parsed description.

## Composition and compiler boundary

Version 1 intentionally contains no loop or control-flow syntax. An
application may compose reusable programs through ordinary control flow, while
a compiler may statically unroll the same composition into a larger DAG. Both
forms have identical program semantics. The choice belongs to the producer,
not the backend.

The checked-in TileLang reference programs use the same directories that a
future `ttl-ttl` compiler must emit: pure DAGs plus final device functions.
Different backends may package different native functions for the same source
program, while the C harness and program API remain unchanged.
