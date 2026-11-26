# Changelog

This file records user-visible changes to libttl. The C API and each JSON
contract carry their own compatibility versions.

## Unreleased

## 0.1.2 - 2026-11-25

- Build LibTorch-backed implementations as C++20 for compatibility with
  PyTorch 2.14 headers while keeping the shared contract checker independent.
- Preserve supported TileLang CUDA assertions before native compilation and
  reject inconsistent dynamic shared-memory launch metadata.
- Derive TileLang CUDA dynamic shared-memory bytes from the selected lowered
  device function while retaining authored launch metadata for raw CUDA.
- Reuse a hardlinked combined cubin across functions prepared as one CUDA
  program while retaining independent per-function package contracts.
- Preserve TileLang 0.1.13 Metal shared-memory address-space finalization while
  retaining one authoritative native Metal compilation in the packager.
- Publish the compiled library release through `ttl_library_version()`, the
  installed `share/libttl/VERSION` file, and the existing pkg-config metadata.
  This backward-compatible API addition advances the C API to 1.4.0.

## 0.1.1 - 2026-11-23

- Preserved TileLang 0.1.13 bounds and domain assertions in Metal packages,
  with fail-closed lowering and asynchronous error reporting at the existing
  stream synchronization boundary. The public C API is unchanged.

## 0.1.0 - 2026-09-13

- Introduced the opaque version 1.3 C API for tensors, devices, streams,
  timers, native final functions, and reusable program execution.
- Added LibTorch-backed CUDA and MPS implementations.
- Added the `ttl-cuda`, `ttl-metal`, and `ttl-tilelang` final-function
  packagers.
- Added version 1 module, launch, program, and fixture JSON contracts with a
  shared structural and semantic validator.
- Added strict ABI, package-consumer, host, CUDA, and MPS API test coverage.
