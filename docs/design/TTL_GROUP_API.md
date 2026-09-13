# Proposed TTL group API

Status: non-normative design note; not part of the implemented public API.

This document proposes an additive SPMD collective layer for TTL. It would
extend `libttl.h` without changing tensor ownership or the fixed local kernel
launch envelope.

The central abstraction is:

> A `ttl_group_t` is this rank's local, device-bound endpoint in one shared
> logical process group.

There is no group daemon. Every participating rank owns one local group handle.
The handles share group size, collective ordering, and a bootstrap identity,
but each contains rank-local device and backend communicator state.

## Layering

```text
launcher / topology policy / bootstrap transport
    world rank, node rank, TP/DP/EP/CP membership, ID exchange
                              |
                              v
libttl.h
    device-bound group rank, collectives, asynchronous completion
                              |
                              v
private backend
    NCCL, HCCL, CNCL, MCCL, or another process-group implementation
```

TTL does not assign semantic names such as tensor parallel or data parallel to
groups. The application can create any number of groups from any rank sets.

## Proposed header

The eventual interface belongs in the flat `include/libttl.h` header:

```c
#ifndef TTL_TTL_GROUP_H
#define TTL_TTL_GROUP_H

#include "libttl.h"

#ifdef __cplusplus
extern "C" {
#endif

#define TTL_GROUP_OPTIONS_VERSION 1u

typedef struct ttl_group ttl_group_t;
typedef struct ttl_work ttl_work_t;

typedef enum ttl_reduce_op {
    TTL_REDUCE_SUM = 0,
    TTL_REDUCE_PRODUCT = 1,
    TTL_REDUCE_MINIMUM = 2,
    TTL_REDUCE_MAXIMUM = 3
} ttl_reduce_op_t;

typedef struct ttl_group_options {
    size_t struct_size;
    uint32_t version;
    int32_t rank;
    int32_t size;
    const void *id;
    size_t id_bytes;
    int64_t timeout_milliseconds;
} ttl_group_options_t;

void ttl_group_options_init(ttl_group_options_t *options);

ttl_status_t ttl_group_id_size(
    const ttl_device_t *device, size_t *bytes);

ttl_status_t ttl_group_id_create(
    const ttl_device_t *device, void *id, size_t bytes);

ttl_status_t ttl_group_create(
    const ttl_device_t *device,
    const ttl_group_options_t *options,
    ttl_group_t **group);

ttl_status_t ttl_group_rank(
    const ttl_group_t *group, int32_t *rank);

ttl_status_t ttl_group_size(
    const ttl_group_t *group, int32_t *size);

void ttl_group_destroy(ttl_group_t *group);

ttl_status_t ttl_group_all_reduce(
    ttl_group_t *group,
    ttl_tensor_t *tensor,
    ttl_reduce_op_t operation,
    ttl_stream_t *stream,
    ttl_work_t **work);

ttl_status_t ttl_group_all_gather(
    ttl_group_t *group,
    ttl_tensor_t *output,
    const ttl_tensor_t *input,
    ttl_stream_t *stream,
    ttl_work_t **work);

ttl_status_t ttl_group_reduce_scatter(
    ttl_group_t *group,
    ttl_tensor_t *output,
    const ttl_tensor_t *input,
    ttl_reduce_op_t operation,
    ttl_stream_t *stream,
    ttl_work_t **work);

ttl_status_t ttl_group_all_to_all(
    ttl_group_t *group,
    ttl_tensor_t *output,
    const ttl_tensor_t *input,
    ttl_stream_t *stream,
    ttl_work_t **work);

ttl_status_t ttl_work_is_complete(
    const ttl_work_t *work, bool *is_complete);

ttl_status_t ttl_work_wait_stream(
    ttl_work_t *work, ttl_stream_t *stream);

ttl_status_t ttl_work_synchronize(ttl_work_t *work);
void ttl_work_destroy(ttl_work_t *work);

#ifdef __cplusplus
}
#endif

#endif
```

This is a C surface. NCCL IDs, HCCL root information, Torch process groups,
native streams, and native work handles remain private.

## Bootstrap and membership

Group creation deliberately does not contain a rendezvous daemon, TCP store,
MPI implementation, node identity, or parallelism policy.

The bootstrap sequence is:

1. Every participant calls `ttl_group_id_size` for the selected backend.
2. One designated participant allocates that many bytes and calls
   `ttl_group_id_create`.
3. The application distributes the opaque byte sequence to the intended group
   members using its existing control plane, such as MPI broadcast, a TCP
   store, an application coordinator, or a job launcher.
4. Each member initializes `ttl_group_options_t`, supplies its group-local rank
   and group size, and calls `ttl_group_create` with the same ID bytes.
5. The ID buffer may be released after `ttl_group_create` returns.

Each independently created logical group uses a distinct ID. Only group
members receive that ID.

`rank` is group-local and must be in `[0, size)`. It is not a device position,
node rank, process ID, or necessarily the application's global world rank. The
topology layer derives it from the group's ordered global membership.

`ttl_group_options_init` initializes `struct_size`, `version`, and a backend
default timeout of `-1`. The size-and-version prefix permits compatible fields
to be added later without changing the creation function.

`ttl_group_create` may block while all ranks initialize the backend
communicator. Initialization failure or timeout is returned through the normal
TTL status and `ttl_last_error` mechanism.

## Application startup example

The application begins with four values supplied by its launch environment:

- the local device position;
- this endpoint's group-local rank;
- the group size; and
- the same opaque group ID bytes supplied to every group member.

How those values are produced or transported is outside the TTL API. Given
them, every rank runs the same startup code:

```c
typedef struct application_group_startup {
    size_t device_position;
    int32_t rank;
    int32_t size;
    const void *group_id;
    size_t group_id_bytes;
} application_group_startup_t;

static void run_application_rank(
    const application_group_startup_t *startup) {
    ttl_device_t *device = NULL;
    ttl_group_t *group = NULL;
    ttl_stream_t *stream = NULL;
    ttl_tensor_t *value = NULL;

    TTL_CHECK(ttl_device_open(startup->device_position, &device));

    ttl_group_options_t options;
    ttl_group_options_init(&options);
    options.rank = startup->rank;
    options.size = startup->size;
    options.id = startup->group_id;
    options.id_bytes = startup->group_id_bytes;
    TTL_CHECK(ttl_group_create(device, &options, &group));

    const int64_t shape[] = {1};
    const float local_value = (float)(startup->rank + 1);
    TTL_CHECK(ttl_tensor_empty(
        device, TTL_DTYPE_FLOAT32, 1, shape, &value));
    TTL_CHECK(ttl_tensor_write(value, &local_value, sizeof(local_value)));
    TTL_CHECK(ttl_stream_create(device, &stream));

    TTL_CHECK(ttl_group_all_reduce(
        group, value, TTL_REDUCE_SUM, stream, NULL));
    TTL_CHECK(ttl_stream_synchronize(stream));

    float sum = 0.0f;
    TTL_CHECK(ttl_tensor_read(value, &sum, sizeof(sum)));
    printf("rank %d of %d: sum=%g\n",
           startup->rank, startup->size, (double)sum);

    ttl_tensor_destroy(value);
    ttl_stream_destroy(stream);
    ttl_group_destroy(group);
    ttl_device_destroy(device);
}
```

For a group of size four, the ranks contribute `1`, `2`, `3`, and `4`; every
rank prints `sum=10`. The same function works for one accelerator, multiple
accelerators in one node, or accelerators across nodes because none of those
deployment details alter the group-local TTL calls.

## Device and tensor contract

A group is permanently bound to the `ttl_device_t` passed to
`ttl_group_create`.

Every collective tensor must be:

- an accelerator tensor, not a host tensor;
- resident on the group's local device;
- contiguous;
- alive until the collective has completed; and
- compatible in dtype and shape with the corresponding arguments on every
  other rank.

The supplied stream must also belong to the group's local device. A group call
never reads or writes a tensor on another local accelerator. Cross-node
transport remains invisible: a backend may use direct device networking or
private staging without changing the public tensor contract.

Host tensors remain outside the group API. Use `ttl_tensor_write`,
`ttl_tensor_read`, and `ttl_close` at the host boundary.

## Collective shapes

All output tensors are allocated by the caller. A collective performs no
publicly observable tensor allocation.

### All-reduce

`ttl_group_all_reduce` is in-place. Every rank supplies the same logical shape
and dtype. The selected operation combines corresponding elements, and the
result is written to every rank's tensor.

### All-gather

For an input of shape `[d0, d1, ...]`, output has shape
`[group_size, d0, d1, ...]`. Slice `output[r]` receives the input from group
rank `r`.

This rank-leading form is the primitive operation. A higher-level lesson or
framework may reshape or concatenate along another dimension after completion.

### Reduce-scatter

This is the inverse shape of all-gather. For output shape `[d0, d1, ...]`,
input has shape `[group_size, d0, d1, ...]`. Corresponding input values are
reduced across ranks, and group rank `r` receives reduced slice `r`.

### All-to-all

Input and output have the same shape and dtype. Their first extent is divisible
by `group_size`. Each rank splits the input evenly along dimension zero, sends
chunk `r` to destination rank `r`, and receives source chunks concatenated in
source-rank order.

Unequal split sizes, point-to-point send/receive, broadcast, and other
collectives can be added without changing these primitives.

## Stream ordering

A collective call is logically enqueued on its explicit `stream`:

1. It observes all work previously enqueued on that stream.
2. Communication may execute on that stream or a private communication stream.
3. The implementation preserves the same observable order in either case.
4. Work submitted later to the same stream observes the collective result.

This rule lets the simple case use no additional synchronization object:

```c
TTL_CHECK(ttl_group_all_reduce(
    group, tensor, TTL_REDUCE_SUM, compute_stream, NULL));
TTL_CHECK(ttl_kernel_launch(
    kernel, tensors, tensor_count, scalars, scalar_count, compute_stream));
TTL_CHECK(ttl_stream_synchronize(compute_stream));
```

The kernel launch is ordered after the all-reduce because it is submitted later
to the same stream.

## Asynchronous work

The final `work` output is optional. Passing `NULL` retains ordinary explicit
stream ordering. Passing a non-NULL output returns an owned `ttl_work_t` for
completion observation or cross-stream dependencies.

`ttl_work_wait_stream(work, consumer)` makes future work on `consumer` wait for
the collective without blocking the host:

```c
ttl_work_t *work = NULL;
TTL_CHECK(ttl_group_all_reduce(
    group, tensor, TTL_REDUCE_SUM, communication_stream, &work));
TTL_CHECK(ttl_work_wait_stream(work, compute_stream));
TTL_CHECK(ttl_kernel_launch(
    kernel, tensors, tensor_count, scalars, scalar_count, compute_stream));
ttl_work_destroy(work);
```

Both streams must use the group's local device.

`ttl_work_synchronize` blocks the host until the collective completes and
reports asynchronous backend failure. `ttl_work_is_complete` is a nonblocking
query.

Destroying a work handle does not cancel or synchronize the operation. The
group and participating tensors must remain alive until either the associated
stream, a consumer stream that waits on the work, or the work itself has been
synchronized. The implementation retains internal operation state independently
of the optional public work handle.

`ttl_stream_synchronize` must report asynchronous collective failures for work
ordered on that stream, just as it reports asynchronous kernel failures.

## Ordering across ranks

All ranks in a group must issue collective calls in the same sequence. For
example, rank 0 cannot issue all-reduce then all-gather while rank 1 issues
all-gather then all-reduce.

Calls on one group must be serialized by the host program. Version 1 does not
permit concurrent calls on the same group from multiple host threads. Different
groups may progress independently.

These are SPMD ordering rules, not topology rules. A TP group and a DP group can
have different sequences because they are distinct group objects.

## Dtypes and packed storage

Collective dtype support is operation- and backend-dependent. Unsupported
combinations return `TTL_STATUS_UNSUPPORTED`.

- Reductions normally support backend numeric dtypes such as float16, float32,
  and selected integer types.
- E2M1 FP4 reduction is not implied by admitting E2M1 tensor storage. Its scale
  recipe is explicit and a backend may reject direct reduction.
- Movement collectives such as all-gather and all-to-all may transport packed
  FP4 as raw storage when every communicated shard starts and ends on a byte
  boundary.
- For four-bit data, each per-rank or per-destination shard therefore contains
  an even number of logical elements unless the backend implements explicit
  repacking.
- Scale tensors are ordinary explicit E4M3, E8M0, or float32 collective
  arguments when the algorithm requires them.

The group API never silently changes dtype, applies a quantization scale, or
inserts padding.

## Lifecycle

The intended ownership order is:

```text
open local device
obtain and exchange group ID
create local group endpoint
allocate local tensors and streams
enqueue collectives and kernels
synchronize completion
destroy work handles
destroy tensors and streams
destroy group
destroy device handle
```

`ttl_group_destroy(NULL)` and `ttl_work_destroy(NULL)` are safe. Destroying a
group with in-flight collective work is invalid; the caller synchronizes first.

## Non-goals for version 1

- No central TTL service or daemon
- No global rank stored in `ttl_device_t`
- No DP, TP, EP, CP, node, or host concepts in `ttl_group_t`
- No host-tensor collectives
- No public NCCL, HCCL, Torch, MPI, or native stream types
- No hidden tensor allocation, dtype conversion, scaling, or padding
- No requirement that different backends use the same bootstrap-byte encoding

These exclusions keep distributed mechanics independent from both local kernel
programming and application topology policy.
