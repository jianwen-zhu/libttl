# TTL public C API

libttl (the Tiny Tensor Library) provides a small C interface for a fixed host
harness to allocate tensors, move data, order device work, launch a prepared
kernel, and check its result. Kernel producers may use handwritten device code,
TileLang, or a future compiler while the harness continues to use this API.

The public interface deliberately does not expose LibTorch, CUDA, Metal,
DLPack, device pointers, or native stream handles. Current implementations use
LibTorch with CUDA or MPS internally; those are implementation choices, not
public TTL types.

## Header and linking

The complete public API is available through one flat header:

```c
#include "libttl.h"
```

The header has C linkage guards and can be included from C or C++. All TTL
objects are opaque and are created and destroyed through TTL functions.

The header publishes `TTL_API_VERSION_MAJOR`, `MINOR`, and `PATCH`, their
encoded `TTL_API_VERSION`, and `ttl_api_version()`. Major versions are ABI
incompatible, minor versions add backward-compatible API, and patch versions
do not change the ABI. An application therefore accepts a library with the
same major version and at least the minor version it was compiled to require:

```c
const uint32_t loaded = ttl_api_version();
const uint32_t loaded_major = loaded >> 24;
const uint32_t loaded_minor = (loaded >> 16) & UINT32_C(0xff);
if (loaded_major != TTL_API_VERSION_MAJOR ||
    loaded_minor < TTL_API_VERSION_MINOR) {
    /* The loaded library does not provide the required API. */
}
```

## Error handling

Most operations return `ttl_status_t`:

| Status | Meaning |
| --- | --- |
| `TTL_STATUS_OK` | Operation completed successfully. |
| `TTL_STATUS_INVALID_ARGUMENT` | A pointer, shape, dtype, tolerance, or argument contract is invalid. |
| `TTL_STATUS_OUT_OF_RANGE` | A device position or tensor dimension is out of range. |
| `TTL_STATUS_UNSUPPORTED` | The requested operation is outside a backend's supported profile. |
| `TTL_STATUS_BACKEND_ERROR` | The implementation or loaded module reported a failure. |

After a non-OK result, call `ttl_last_error()` immediately for a diagnostic.
The returned string is owned by TTL and remains valid only until the next TTL
call on the same thread.

```c
#define TTL_CHECK(expression)                                                   \
    do {                                                                        \
        ttl_status_t status = (expression);                                     \
        if (status != TTL_STATUS_OK) {                                          \
            fprintf(stderr, "%s failed: %s\n", #expression, ttl_last_error()); \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)
```

Destroy functions return `void`. Passing `NULL` to a destroy function is safe.

## Dtypes

`ttl_dtype_t` describes the logical element encoding of a tensor. Bit width,
rather than C `sizeof`, is fundamental because TTL includes packed types:

| TTL dtype | Encoding | Bits |
| --- | --- | ---: |
| `TTL_DTYPE_BOOL` | Boolean byte | 8 |
| `TTL_DTYPE_INT8` | Signed integer | 8 |
| `TTL_DTYPE_INT32` | Signed integer | 32 |
| `TTL_DTYPE_FLOAT16` | IEEE 754 binary16 | 16 |
| `TTL_DTYPE_FLOAT32` | IEEE 754 binary32 | 32 |
| `TTL_DTYPE_FLOAT4_E2M1` | Finite E2M1, two logical elements per byte | 4 |
| `TTL_DTYPE_FLOAT8_E4M3` | Finite E4M3; suitable for NVFP4 block scales | 8 |
| `TTL_DTYPE_FLOAT8_E8M0` | Unsigned E8M0; suitable for MXFP4 block scales | 8 |

```c
const char *ttl_dtype_name(ttl_dtype_t dtype);
size_t ttl_dtype_bit_width(ttl_dtype_t dtype);
ttl_status_t ttl_dtype_storage_bytes(
    ttl_dtype_t dtype, size_t element_count, size_t *bytes);
```

`ttl_dtype_name` returns a static name such as `"float32"`; an invalid enum
value produces `"invalid"`. `ttl_dtype_bit_width` returns zero for an invalid
value. `ttl_dtype_storage_bytes` calculates the checked packed storage size:
four-bit storage uses `ceil(element_count / 2)` bytes.

TTL does not define public C arithmetic types for float16, float4, or float8.
Host reads and writes transport their encodings as raw bytes.

### FP4 packing and scaling

`TTL_DTYPE_FLOAT4_E2M1` uses the OCP E2M1 value encoding. In contiguous linear
order, element 0 occupies the low nibble of byte 0, element 1 occupies the high
nibble, element 2 occupies the low nibble of byte 1, and so on. For an odd
element count, the unused high nibble of the final byte must be zero.

The dtype represents E2M1 values, not an entire quantization recipe. Scaling is
explicit in the module signature:

- MXFP4 passes E2M1 data and an E8M0 block-scale tensor, normally with one
  scale per block of 32 values.
- NVFP4 passes E2M1 data, an E4M3 block-scale tensor, and any tensor-wide
  float32 scale required by its recipe, normally with one block scale per 16
  values.

Keeping these as separate tensor arguments supports both recipes without
embedding one vendor's block geometry or scale layout into `ttl_tensor_t`.
The producer's module contract specifies scale shapes, padding, and layout.
The encodings follow the
[OCP Microscaling Formats specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf).
NVIDIA's
[NVFP4 documentation](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/nvfp4/nvfp4.html)
describes the E2M1, E4M3 block-scale, and float32 global-scale combination.

## Devices

A `ttl_device_t` identifies an opened accelerator. Host storage is represented
by a host tensor and has no device handle.

```c
ttl_status_t ttl_device_count(size_t *count);
ttl_status_t ttl_device_open(size_t position, ttl_device_t **device);
ttl_status_t ttl_device_capabilities(
    const ttl_device_t *device, ttl_capability_set_t *capabilities);
void ttl_device_destroy(ttl_device_t *device);
```

Device positions are local accelerator positions in the range
`[0, device_count)`. A position is not a distributed rank. The returned device
owns no tensor or stream; destroying it does not destroy objects created from
it. Keep the device alive while it is still needed to create those objects.

### Capability discovery

Capabilities are an extensible `uint64_t` bitset. Query the opened device
rather than assuming that every implementation has identical launch or
scheduling behavior:

| Capability | Contract |
| --- | --- |
| `TTL_CAPABILITY_DYNAMIC_WORKGROUP_MEMORY` | A module may request a runtime-sized region shared by one workgroup (a CUDA block or Metal threadgroup). |
| `TTL_CAPABILITY_INDEPENDENT_STREAMS` | Distinct TTL streams can use independently scheduled native queues. Without it, their ordering contract still holds, but they may serialize. |
| `TTL_CAPABILITY_NATIVE_GRAPH_EXECUTION` | A prepared TTL program can be instantiated and submitted as one native graph. Without it, the backend launches the same final functions in topological order. |

`TTL_CAPABILITY_KNOWN_MASK` is the union of capability bits understood by the
current header. New bits may be added in later minor API versions. Portable
applications must ignore unknown available bits and test only features they
need.

Capability discovery is deliberately separate from backend naming. It lets a
harness make a compatibility decision without knowing whether the device is
implemented with CUDA, Metal, or another accelerator API.

## Tensors

### Allocation and lifetime

```c
ttl_status_t ttl_tensor_empty_host(
    ttl_dtype_t dtype, size_t rank, const int64_t *shape,
    ttl_tensor_t **tensor);

ttl_status_t ttl_tensor_empty(
    const ttl_device_t *device, ttl_dtype_t dtype, size_t rank,
    const int64_t *shape, ttl_tensor_t **tensor);

void ttl_tensor_destroy(ttl_tensor_t *tensor);
```

Both allocation functions create uninitialized, contiguous storage. `rank == 0`
creates a scalar and permits `shape == NULL`. For a non-scalar tensor, `shape`
must point to `rank` non-negative extents. A zero extent is valid.

`ttl_tensor_empty_host` creates CPU-accessible storage.
`ttl_tensor_empty` creates storage on the opened accelerator. TTL intentionally
provides no public tensor arithmetic; values are produced by host writes,
copies, or launched kernels.

The caller owns the returned tensor. Do not destroy a tensor while work that
uses its storage is still in flight.

### Host-boundary transfers

```c
ttl_status_t ttl_tensor_write(
    ttl_tensor_t *tensor, const void *source, size_t bytes);

ttl_status_t ttl_tensor_read(
    const ttl_tensor_t *tensor, void *destination, size_t bytes);
```

These operations are synchronous at the host boundary: when they return, the
source host buffer may be reused after a write and the destination host buffer
contains the tensor values after a read. The tensor must be contiguous and
`bytes` must equal its complete byte size. Partial reads and writes are not
supported.

The host buffer contains raw elements in the tensor dtype and contiguous
row-major order. For example, a `float32` tensor is written from a `float[]`,
while E2M1 is transferred as nibble-packed bytes using the rule above.

### Stream-ordered copies

```c
ttl_status_t ttl_tensor_copy(
    ttl_tensor_t *destination, const ttl_tensor_t *source,
    ttl_stream_t *stream);
```

This enqueues a copy on `stream`. Source and destination must have equal shapes
and dtypes. The operation may be asynchronous; synchronize the stream or
establish a stream dependency before observing the destination or releasing
either tensor.

### Metadata

```c
ttl_status_t ttl_tensor_rank(const ttl_tensor_t *tensor, size_t *rank);
ttl_status_t ttl_tensor_extent(
    const ttl_tensor_t *tensor, size_t dimension, int64_t *extent);
ttl_status_t ttl_tensor_stride(
    const ttl_tensor_t *tensor, size_t dimension, int64_t *stride);
ttl_status_t ttl_tensor_dtype(
    const ttl_tensor_t *tensor, ttl_dtype_t *dtype);
ttl_status_t ttl_tensor_byte_size(
    const ttl_tensor_t *tensor, size_t *bytes);
ttl_status_t ttl_tensor_is_host(
    const ttl_tensor_t *tensor, bool *is_host);
ttl_status_t ttl_tensor_is_contiguous(
    const ttl_tensor_t *tensor, bool *is_contiguous);
```

Dimensions are zero-based. A stride is measured in elements, not bytes.
`ttl_tensor_byte_size` reports physical storage bytes. For byte-aligned dtypes
this is the number of logical elements multiplied by bytes per element; for
E2M1 it is `ceil(logical_elements / 2)`. `ttl_tensor_is_host` distinguishes
host storage from accelerator storage without exposing a backend-specific
placement type.

The public API currently creates only contiguous tensors, but the metadata and
module contracts make layout requirements explicit.

## Correctness comparison

```c
typedef struct ttl_close_result {
    bool is_close;
    size_t mismatch_count;
    size_t worst_linear_index;
    double max_absolute_error;
    double max_relative_error;
} ttl_close_result_t;

ttl_status_t ttl_close(
    const ttl_tensor_t *actual,
    const ttl_tensor_t *reference,
    double relative_tolerance,
    double absolute_tolerance,
    ttl_close_result_t *result);
```

`ttl_close` is a verification utility, not a tensor kernel. It copies both
tensors to the CPU, makes them contiguous, and compares corresponding values.
Shapes must match, but dtypes may differ within the TTL dtype profile. Both
tolerances must be non-negative.

For E2M1, E4M3, and E8M0, comparison decodes the individual value encoding. It
does not implicitly apply a companion scale tensor; a scaled-format CPU
reference must apply the recipe explicitly.

An element passes when:

```text
abs(actual - reference) <= absolute_tolerance
                           + relative_tolerance * abs(reference)
```

`mismatch_count` counts elements that fail that test. Errors and
`worst_linear_index` use contiguous row-major linear indexing. If all elements
are exactly equal, both maximum errors are zero and `worst_linear_index` is
`SIZE_MAX`. NaNs are not close; equal infinities compare equal.

Because this function synchronizes transfers and performs scalar CPU work, use
it for correctness gates rather than performance measurement.

## Streams and timers

```c
ttl_status_t ttl_stream_create(
    const ttl_device_t *device, ttl_stream_t **stream);

ttl_status_t ttl_stream_wait(
    ttl_stream_t *waiting, const ttl_stream_t *dependency);

ttl_status_t ttl_stream_synchronize(ttl_stream_t *stream);
void ttl_stream_destroy(ttl_stream_t *stream);
```

`ttl_stream_create` returns an owned stream that must be destroyed. TTL has no
ambient current stream and therefore no get-current or set-current operation.
Every stream-ordered TTL function receives its stream explicitly. This makes
ordering visible in C source and avoids backend-specific thread-local state.

`ttl_stream_wait(waiting, dependency)` records the dependency so that future
work on `waiting` begins after previously enqueued work on `dependency`. It does
not block the host. `ttl_stream_synchronize` blocks the host until all work
submitted to that stream before the synchronization call has completed. That
includes work pulled in through earlier `ttl_stream_wait` dependencies. It does
not wait for unrelated streams, and it does not cover work submitted later by
another host thread.

A backend may conservatively map multiple TTL handles to one ordered native
queue. The current LibTorch/MPS implementation does so because LibTorch exposes
one serial MPS stream. The explicit TTL calls and their ordering guarantees do
not change; independent overlap is simply unavailable on that backend.

Destroying a wrapper is not a substitute for synchronization. The caller must
synchronize dependent streams before destroying tensors, prepared kernels, or
other resources still used by queued work.

TTL also provides an opaque timer for accelerator measurements:

```c
ttl_status_t ttl_timer_create(
    const ttl_device_t *device, ttl_timer_t **timer);
ttl_status_t ttl_timer_begin(ttl_timer_t *timer, ttl_stream_t *stream);
ttl_status_t ttl_timer_end(
    ttl_timer_t *timer, ttl_stream_t *stream, double *milliseconds);
void ttl_timer_destroy(ttl_timer_t *timer);
```

`ttl_timer_begin` establishes the start of an interval after work previously
submitted to that stream. Submit the measured work to the same stream, then
call `ttl_timer_end`. The end call blocks until the interval completes and
returns its duration in milliseconds. A timer measures only one active
interval at a time and may be reused after a successful end. The timer and
stream must belong to the same accelerator.

Native timing types remain private. CUDA uses timing-enabled device events.
The current LibTorch/MPS backend, which does not expose an equivalent portable
event through its C++ surface, synchronizes the MPS stream and measures the
interval with a monotonic host clock. Thus both implementations measure a
completed stream interval, while CUDA excludes more host dispatch overhead.
Use warm-up iterations and multiple measured submissions for stable results.

## Modules and kernels

A module is a directory containing a backend-neutral human-authored manifest,
a compiler-generated private launch description, and a native device artifact.
A prepared kernel binds that module to one accelerator and creates persistent
launch state before the launch path.

```c
ttl_status_t ttl_module_load(const char *path, ttl_module_t **module);
ttl_status_t ttl_module_required_capabilities(
    const ttl_module_t *module, ttl_capability_set_t *capabilities);
void ttl_module_destroy(ttl_module_t *module);

ttl_status_t ttl_kernel_prepare(
    const ttl_module_t *module,
    const ttl_device_t *device,
    ttl_kernel_t **kernel);
void ttl_kernel_destroy(ttl_kernel_t *kernel);
```

`ttl_module_load` opens the module directory at `path`, parses `module.json`
and private `launch.json`, and validates both. The runtime derives the module's
requirements from that validated description; callers do not parse the private
file or duplicate a hand-authored capability list.

Before preparation, a backend-neutral harness can preflight the pair:

```c
ttl_capability_set_t available = 0;
ttl_capability_set_t required = 0;
TTL_CHECK(ttl_device_capabilities(device, &available));
TTL_CHECK(ttl_module_required_capabilities(module, &required));
if ((required & ~available) != 0) {
    /* Select a compatible module or report the unsupported requirements. */
}
```

`ttl_kernel_prepare` repeats this check as a safety gate and returns
`TTL_STATUS_UNSUPPORTED` before loading the native artifact when capabilities
are missing. It otherwise loads the artifact and resolves its device symbol
for `device`; compilation, symbol lookup, and persistent allocation belong in
module production or prepare, not launch.

A prepared kernel retains the loaded module internally, so destroying the
public module handle before the prepared kernel is permitted. Destruction does
not wait for device work; synchronize first.

### Scalars

```c
typedef struct ttl_scalar {
    ttl_dtype_t dtype;
    union {
        int32_t int32;
        float float32;
    } value;
} ttl_scalar_t;
```

The version-1 scalar profile supports only `TTL_DTYPE_INT32` and
`TTL_DTYPE_FLOAT32`. Initialize the union member selected by `dtype`.

### Fixed launch envelope

```c
ttl_status_t ttl_kernel_launch(
    ttl_kernel_t *kernel,
    ttl_tensor_t *const *tensors,
    size_t tensor_count,
    const ttl_scalar_t *scalars,
    size_t scalar_count,
    ttl_stream_t *stream);
```

Launch validates all arguments against the module manifest and then enqueues
the kernel on `stream`. The stream and every tensor must use the accelerator on
which the kernel was prepared. Tensor and scalar counts, ordering, dtypes,
ranks, fixed extents, and required contiguous layouts must match the manifest.
The arrays may be `NULL` only when their corresponding count is zero.

The call reports enqueue-time errors but does not synchronize. Call
`ttl_stream_synchronize` before checking results or destroying resources.
Launch itself performs no compilation, dynamic symbol lookup, or persistent
allocation.

The public harness does not inspect the private manifest. In a lesson, the
module's documented contract defines the array order. The AXPY modules use:

```c
ttl_tensor_t *tensors[] = {output, x, y};
const ttl_scalar_t scalars[] = {
    {.dtype = TTL_DTYPE_FLOAT32, .value.float32 = alpha},
};

TTL_CHECK(ttl_kernel_launch(kernel, tensors, 3, scalars, 1, stream));
TTL_CHECK(ttl_stream_synchronize(stream));
```

The module authoring contract is separate from this public host API; see
[KERNEL_MODULE.md](KERNEL_MODULE.md).

## Programs

A program is a pure, static DAG of final kernel functions. It may be a small
human-authored reusable routine or an entire compiler-emitted flattened model:

```c
typedef struct ttl_program_binding {
    const char *name;
    ttl_tensor_t *tensor;
} ttl_program_binding_t;

ttl_status_t ttl_program_load(
    const char *path, ttl_program_t **program);
void ttl_program_destroy(ttl_program_t *program);

ttl_status_t ttl_program_prepare(
    const ttl_program_t *program,
    const ttl_device_t *device,
    const ttl_program_binding_t *bindings,
    size_t binding_count,
    ttl_program_execution_t **execution);
void ttl_program_execution_destroy(ttl_program_execution_t *execution);

ttl_status_t ttl_program_launch(
    ttl_program_execution_t *execution, ttl_stream_t *stream);
```

Preparation binds every declared input and output by name, validates exact
dtypes and static shapes, prepares referenced functions, and allocates private
intermediates. Launch is stream-ordered and asynchronous. CUDA currently uses
a native CUDA Graph; the MPS implementation executes the same final-function
DAG in its valid topological order. The public semantics are identical.
One program may be prepared repeatedly with different tensor bindings, as in
the twelve executions of the EinyGPT transformer-layer program.

The complete format and lifecycle are specified in
[TTL_PROGRAM.md](TTL_PROGRAM.md). A program never exposes native graph handles,
device pointers, backend names, or mutable tensor state.

## Complete lifecycle example

The reusable harness, instantiated with the AXPY case, demonstrates the
intended order:

```text
run the host reference
if --device was supplied:
    open device
    allocate and initialize tensors
    create stream
    load module
    compare module requirements with device capabilities
    prepare kernel for device
    launch on stream
    synchronize stream
    compare declared outputs with the reference using ttl_close
    destroy kernel, module, stream, tensors, and device
```

The `ece467-labs` repository supplies small C exercises and an
application-scale llm.c harness that demonstrate this lifecycle without
exposing its function graph to host code.

## Portability boundary

Portable harness code may depend on `include/libttl.h` and documented API
contracts only. It must not depend on object layouts, private module descriptor
types, native storage addresses, native streams, or either current LibTorch
implementation. A different accelerator backend can implement the same TTL
surface and consume its appropriate native module artifact.

SPMD collective communication is intentionally a separate additive layer. Its
proposed device-bound process-group contract is isolated as a non-normative
design note in [design/TTL_GROUP_API.md](design/TTL_GROUP_API.md); it does not
add rank or communication state to the local tensor, device, stream, or module
interfaces.
