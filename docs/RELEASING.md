# Release gates

The repository version is stored in `VERSION`. Four public surfaces are
versioned independently:

- `ttl_library_version()` identifies the compiled libttl release. Installed
  `share/libttl/VERSION` and `libttl.pc` publish the same value.
- `TTL_API_VERSION` in `include/libttl.h` versions the C ABI.
- Compiler interface version 1 covers the three native function packagers.
- The module, launch, program, and fixture schemas carry their own versions.

Before publishing a release:

1. Update `VERSION`, finalize the matching `CHANGELOG.md` entry, and commit the
   candidate so the worktree is clean.
2. Run `make archive-check` in the clean local Git worktree.
3. Run `make backend-check TTL_BACKEND=libtorch_cuda` on the validated CUDA
   profile.
4. Run `make backend-check TTL_BACKEND=libtorch_mps` on the validated MPS
   profile.
5. Record the actual profiles and run provenance in `docs/COMPATIBILITY.md`.
6. Qualify the candidate through the pinned downstream integration suite.
7. Build the public source archive from the tested commit and inspect its file
   list before creating the release tag.

Publish an immutable tag and the source archive SHA-256 together. Downstream
build caches should include that source checksum, release version, backend,
toolchain, and target profile in their key. Before execution, a runner should
call `ttl_library_version()` in the same process and require the tested release.
When provenance must identify the exact backend binary as well, resolve the
loaded library path and compare its SHA-256 with the build-specific digest
recorded by the packager. Binary digests are platform-specific and do not
replace the canonical source archive checksum.

`release-check` combines `archive-check` and `backend-check` when both can run
in the same Git worktree. `archive-check` refuses a dirty or non-Git worktree,
extracts `git archive HEAD` into a temporary directory, and validates that
committed source independently of the checkout. Snapshot-based remote runners
such as CloudMake use `backend-check`, because their synchronized source does
not include `.git`. CI runs the non-GPU archive stage on Linux and macOS; live
CUDA and MPS execution remains a release gate on real accelerator hardware.

`package-check` stages a fresh installation, links and runs a consumer
against that staged package, and verifies that the library exports exactly the
symbols listed in `tests/expected_abi_symbols.txt`. `src/libttl.exports` is
the matching linker export list. Change either list only as part of a
deliberate public API version decision.

Never change the meaning of a published schema version. Add a new schema
version when a structural or semantic rule must change.
