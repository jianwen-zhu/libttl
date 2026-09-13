# Changelog

This file records user-visible changes to libttl. The C API and each JSON
contract carry their own compatibility versions.

## 0.1.0 - 2026-09-13

- Introduced the opaque version 1.3 C API for tensors, devices, streams,
  timers, native final functions, and reusable program execution.
- Added LibTorch-backed CUDA and MPS implementations.
- Added the `ttl-cuda`, `ttl-metal`, and `ttl-tilelang` final-function
  packagers.
- Added version 1 module, launch, program, and fixture JSON contracts with a
  shared structural and semantic validator.
- Added strict ABI, package-consumer, host, CUDA, and MPS API test coverage.
