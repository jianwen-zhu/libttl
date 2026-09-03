SHELL := /bin/bash
.DEFAULT_GOAL := help

LIBTTL_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
TTL_BACKEND ?= libtorch_cuda
PYTHON ?= python3

ifeq ($(wildcard /content),/content)
DEFAULT_STATE_ROOT := /content/.libttl
else
DEFAULT_STATE_ROOT := $(HOME)/.cache/libttl
endif

TTL_STATE_ROOT ?= $(DEFAULT_STATE_ROOT)
TTL_BUILD_DIR ?= $(TTL_STATE_ROOT)/build
JSONCONS_ROOT ?= $(TTL_STATE_ROOT)/deps/jsoncons-1.9.0
JSONCONS_INCLUDE_DIR ?= $(JSONCONS_ROOT)/include

export LIBTTL_ROOT TTL_BACKEND TTL_STATE_ROOT TTL_BUILD_DIR
export JSONCONS_ROOT JSONCONS_INCLUDE_DIR

.PHONY: help bootstrap build contract-check tools-check tests package-check \
	static-check

help:
	@printf '%s\n' \
		'libttl — the Tiny Tensor Library' \
		'' \
		'  bootstrap       Install the pinned jsoncons headers in external state' \
		'  build           Build libttl for TTL_BACKEND' \
		'  contract-check  Build the shared JSON contract validator' \
		'  tools-check     Check the native and TileLang packagers' \
		'  tests           Run backend runtime, contract, and ABI tests' \
		'  package-check   Stage and inspect a library installation' \
		'  static-check    Run checks that do not require an accelerator'

bootstrap:
	@mkdir -p '$(TTL_STATE_ROOT)'
	@$(PYTHON) scripts/install_jsoncons.py \
		--lock third_party/jsoncons.lock.json --destination '$(JSONCONS_ROOT)'

build: bootstrap
	@$(MAKE) -C src BACKEND='$(TTL_BACKEND)' PYTHON='$(PYTHON)'

contract-check: bootstrap
	@$(MAKE) -C src contract-checker PYTHON='$(PYTHON)'

tools-check: contract-check
	@$(MAKE) -C tool PYTHON='$(PYTHON)' check

tests: build
	@$(MAKE) -C tests PYTHON='$(PYTHON)'

package-check: build
	@$(MAKE) -C src install BACKEND='$(TTL_BACKEND)' PYTHON='$(PYTHON)' \
		DESTDIR='$(TTL_BUILD_DIR)/package-root' PREFIX=/usr/local
	@test -x '$(TTL_BUILD_DIR)/package-root/usr/local/lib/libttl.so'
	@test -f '$(TTL_BUILD_DIR)/package-root/usr/local/include/libttl.h'
	@test -f '$(TTL_BUILD_DIR)/package-root/usr/local/share/libttl/schema/ttl-program-v1.schema.json'
	@test -f '$(TTL_BUILD_DIR)/package-root/usr/local/share/libttl/schema/ttl-fixture-v1.schema.json'
	@echo '[package] staged libttl, its public header, and JSON contracts'

static-check:
	@for document in schema/*.json $$(find tests -name '*.json' -type f); do \
		$(PYTHON) -m json.tool "$$document" >/dev/null || exit; \
	done
	@PYTHONPYCACHEPREFIX=/tmp/libttl-pycache $(PYTHON) -m py_compile \
		tool/ttl_compiler.py tool/ttl-cuda tool/ttl-metal tool/ttl-tilelang \
		tool/torch_flags.py tool/embed_schema.py scripts/install_jsoncons.py
	@PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m unittest discover -s tests

