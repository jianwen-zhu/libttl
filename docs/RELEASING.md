# Release and compatibility gates

The repository version is stored in `VERSION`. The independently versioned
public contracts are:

- `TTL_API_VERSION` in `include/libttl.h` for the C ABI;
- compiler interface version 1 for the three function packagers;
- the version-1 module, program, and fixture JSON schemas under `schema/`.

Before publishing a release:

1. Run `make static-check`.
2. Run `make tests TTL_BACKEND=libtorch_cuda` on the supported CUDA profile.
3. Run `make tests TTL_BACKEND=libtorch_mps` on the supported Metal profile.
4. Run `make package-check` for both backends.
5. Qualify the release through the pinned `ece467-labs` regression suite.

`tests/expected_abi_symbols.txt` is the ABI allowlist. Change it only with a
deliberate public API version decision. Never change the meaning of a published
schema version; add a new schema version instead.
