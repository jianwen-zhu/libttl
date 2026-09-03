# TTL test fixture format, version 1.0

A TTL test fixture is a self-contained, deterministic numerical example used
to check the meaning of an executable TTL program. It carries concrete tensor
values, expected observations, comparison policies, and enough provenance to
reproduce the oracle.

A fixture is host-side test data. It is not a TTL program, kernel module,
compiler input, launch description, or scheduling format. Compilers do not
read or emit fixtures. A fixed runner uses one after compilation:

```text
fixture.json --------------------> inputs, parameters, expected values
                                        |                 ^
compiled program --> libttl runner -----+--> execution ---+ comparison
```

The normative JSON structure is
[`schema/ttl-fixture-v1.schema.json`](../schema/ttl-fixture-v1.schema.json).
This document defines the field meanings and cross-field invariants. The
schema and semantics together form the version-1.0 fixture contract.

## Bundle sections

| Field | Meaning |
| --- | --- |
| `$schema` | Optional schema location. Consumers must not require network access to use it. |
| `schema_version` | The string `"1.0"`. Fixture versions are independent of the integer versions used by `program.json` and `module.json`. |
| `id` | Stable diagnostic name for this concrete fixture. |
| `model` | Stable name of the model or tensor program being exercised. |
| `profile` | Language profile used by the producer: `TTL-Edu` or `TTL-Prod`. |
| `axes` | Concrete positive extent for every named axis used by a tensor in the bundle. |
| `references` | Nonempty list of pinned implementations used to obtain the oracle. |
| `generator` | Reproducible adapter path, command, and optional seed and notes. |
| `inputs` | Invocation tensors supplied to the exported computation. |
| `parameters` | Model or program tensors, such as weights, kept distinct from invocation inputs. |
| `checkpoints` | Optional named internal observations used to localize disagreement. |
| `expected` | Named exported results. A numerical conformance fixture has at least one. |

`inputs` and `parameters` are semantically distinct provenance groups, but
both supply concrete tensors to an evaluator or runner. `checkpoints` are
diagnostic and do not enlarge the program's public result ABI. `expected`
names the externally checked results.

## Axes and concrete shapes

An `axes` entry binds an axis identity to one concrete extent for this test.
A dynamic TTL axis retains its `$` prefix but is still materialized to a
positive extent in each fixture:

```json
"axes": {
  "batch": 2,
  "$token": 5,
  "channel": 8
}
```

Thus `$token` records that the source dimension is dynamic while `5` selects
the particular case being tested. Different fixtures may bind it differently.

Every tensor record contains:

- `dtype`: its TTL nominal element type, such as `bool`, `i32`, `f16`, or
  `f32`;
- `axes`: its ordered axis identities, from outermost to innermost;
- `shape`: concrete nonnegative extents in the same order;
- `data`: nested row-major logical values with exactly that shape.

Fixture dtype names follow the TTL language, not the spelling of the libttl C
enum. For example, the fixed runner maps TTL `f32` to
`TTL_DTYPE_FLOAT32`. A target profile defines any additional nominal dtype
mapping.

Tensor records obey these semantic invariants:

1. `axes` and `shape` have the same length.
2. Every tensor axis has a binding in the top-level `axes` object.
3. Each tensor extent equals the corresponding top-level binding.
4. The nested `data` shape equals the declared `shape`.
5. Rank-zero tensor data is a scalar, not a one-element array.
6. Boolean data contains only JSON Boolean values; `i32` data contains only
   signed 32-bit integers; numeric dtype data contains finite JSON numbers.

The JSON values represent logical tensor elements. Backend-specific packed
storage, byte order, device placement, and physical layout do not appear in a
fixture.

## Observations and comparison

Each `checkpoints` or `expected` entry contains a `tensor`, a `comparison`,
and an optional `stage` description. Stable dotted observation names such as
`block.0.attention.probability` are encouraged. `stage` may identify the
corresponding location in the pinned reference, but it has no execution
semantics.

There are two comparison policies:

```json
{"mode": "exact"}
```

```json
{
  "mode": "tolerance",
  "absolute_tolerance": 1e-6,
  "relative_tolerance": 1e-5
}
```

Exact comparison requires identical values. Tolerance comparison accepts each
actual value `a` and expected value `e` when:

```text
abs(a - e) <= absolute_tolerance + relative_tolerance * abs(e)
```

Shapes and dtypes must agree before values are compared. Boolean and integer
observations use exact comparison. Floating observations normally use an
explicit tolerance because parallel reductions may reassociate operations;
exact floating comparison remains valid for intentionally bit-identical
references.

## Provenance and generation

Each reference records at least `name`, repository URI, hexadecimal revision,
and license. Optional `source` and `role` fields identify the exact entry point
and its contribution. The revision must be pinned; a branch name is not a
reproducible reference.

The generator is an offline adapter. It may invoke an upstream environment and
convert layouts or serialize values, but it must obtain expected results by
executing or importing the pinned reference implementation. Reimplementing the
same equations in the adapter would create a second implementation, not an
independent oracle. Synthetic inputs and weights should come from a documented
seed rather than redistributed model checkpoints.

A committed fixture should be small enough to inspect and run on a CPU. Its
recorded command should regenerate the bundle deterministically. Regeneration
belongs to the fixture producer; ordinary consumers and student compilers need
no upstream framework or generator dependencies.

## Small example

This complete example checks a two-element identity program. Real model
fixtures use an independently executed upstream reference and generally carry
more parameters and observations.

```json
{
  "$schema": "https://ttl-lang.dev/schema/fixture-v1.json",
  "schema_version": "1.0",
  "id": "identity-two-f32",
  "model": "identity",
  "profile": "TTL-Edu",
  "axes": {"element": 2},
  "references": [
    {
      "name": "identity-reference",
      "repository": "https://github.com/jianwen-zhu/libttl",
      "revision": "ed71a850463632a1c6c8125a7c9d75ab1515a3e0",
      "license": "MIT",
      "source": "tests/programs/identity/program.json",
      "role": "structural contract example"
    }
  ],
  "generator": {
    "path": "tests/fixtures/minimal.fixture.json",
    "command": "authored structural fixture",
    "seed": 7
  },
  "inputs": {
    "input": {
      "dtype": "f32",
      "axes": ["element"],
      "shape": [2],
      "data": [1.25, -2.5]
    }
  },
  "parameters": {},
  "checkpoints": {},
  "expected": {
    "output": {
      "tensor": {
        "dtype": "f32",
        "axes": ["element"],
        "shape": [2],
        "data": [1.25, -2.5]
      },
      "comparison": {"mode": "exact"}
    }
  }
}
```

The checked-in
[`tests/fixtures/minimal.fixture.json`](../tests/fixtures/minimal.fixture.json)
is deliberately only a structural validator smoke test; it is not a numerical
oracle for a model.

## Validation and ownership

Build the shared validator and check a fixture with:

```sh
TTL_BUILD_DIR="${TTL_BUILD_DIR:-$HOME/.cache/libttl/build}"
make contract-check TTL_BUILD_DIR="$TTL_BUILD_DIR"
"$TTL_BUILD_DIR/ttl-contract-check" --fixture path/to/fixture.json
```

`ttl-contract-check` uses jsoncons and the same schema embedded in libttl. JSON
Schema validates the portable structure. Cross-field invariants involving
nested data shape, axis bindings, dtype value domains, and comparison policy
must also be enforced by a fixture producer and by the runner before execution.

The ownership boundary is intentional:

- fixture producers own provenance, deterministic generation, and expected
  values;
- compilers own `program.json` and final function packages;
- the fixed runner owns fixture decoding, TTL-to-libttl dtype mapping, tensor
  construction, execution, and comparison;
- students implementing a compiler do not parse, generate, or optimize around
  fixture contents.
