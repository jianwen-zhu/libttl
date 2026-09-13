SHELL := /bin/bash
.DEFAULT_GOAL := help

LIBTTL_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
TTL_BACKEND ?= libtorch_cuda
PYTHON ?= python3
CC ?= cc
PKG_CONFIG ?= pkg-config
PLATFORM := $(shell uname -s)

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

ifeq ($(PLATFORM),Darwin)
PACKAGE_RPATH := @loader_path/../lib
else
PACKAGE_RPATH := $$ORIGIN/../lib
endif

.PHONY: help bootstrap build contract-check tools-check api-check tests \
	examples-check package-check static-check archive-check backend-check \
	release-check

help:
	@printf '%s\n' \
		'libttl — the Tiny Tensor Library' \
		'' \
		'  bootstrap       Install the pinned jsoncons headers in external state' \
		'  build           Build libttl for TTL_BACKEND' \
		'  contract-check  Build the shared JSON contract validator' \
		'  tools-check     Check the native and TileLang packagers' \
		'  api-check       Exercise the complete public API on TTL_BACKEND' \
		'  examples-check  Build and run the standalone host example' \
		'  tests           Run backend runtime, contract, and ABI tests' \
		'  package-check   Stage, link, and inspect a library installation' \
		'  static-check    Run checks that do not require an accelerator' \
		'  archive-check   Validate a clean source archive without a GPU' \
		'  backend-check   Run release gates that need the selected backend' \
		'  release-check   Run every release gate for TTL_BACKEND'

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

api-check: tests

tests: build
	@$(MAKE) -C tests PYTHON='$(PYTHON)'

examples-check: build
	@$(MAKE) -C examples check CC='$(CC)' \
		TTL_BACKEND='$(TTL_BACKEND)' TTL_BUILD_DIR='$(TTL_BUILD_DIR)'

package-check: build
	@set -euo pipefail; \
	package_root="$$(mktemp -d "$${TMPDIR:-/tmp}/libttl-package.XXXXXX")"; \
	trap 'rm -rf "$$package_root"' EXIT; \
	$(MAKE) -C src install BACKEND='$(TTL_BACKEND)' PYTHON='$(PYTHON)' \
		DESTDIR="$$package_root" PREFIX=/usr/local; \
	test -x "$$package_root/usr/local/lib/libttl.so"; \
	test -f "$$package_root/usr/local/include/libttl.h"; \
	for schema in ttl-kernel-module-v1 ttl-kernel-launch-v1 ttl-program-v1 \
		ttl-fixture-v1; do \
		test -f "$$package_root/usr/local/share/libttl/schema/$$schema.schema.json"; \
	done; \
	test -f "$$package_root/usr/local/share/doc/libttl/LICENSE"; \
	test -f "$$package_root/usr/local/share/doc/libttl/THIRD_PARTY_NOTICES.md"; \
	test -f "$$package_root/usr/local/lib/pkgconfig/libttl.pc"; \
	test "$$(PKG_CONFIG_PATH="$$package_root/usr/local/lib/pkgconfig" \
		$(PKG_CONFIG) --modversion libttl)" = "$$(cat VERSION)"; \
	install -d "$$package_root/usr/local/bin"; \
	$(CC) -std=c11 -Wall -Wextra -Wpedantic \
		-I"$$package_root/usr/local/include" tests/api_host_tests.c \
		-L"$$package_root/usr/local/lib" -lttl \
		'-Wl,-rpath,$(PACKAGE_RPATH)' \
		-lm \
		-o "$$package_root/usr/local/bin/libttl-package-smoke"; \
	"$$package_root/usr/local/bin/libttl-package-smoke"; \
	pkg_flags="$$(PKG_CONFIG_PATH="$$package_root/usr/local/lib/pkgconfig" \
		PKG_CONFIG_SYSROOT_DIR="$$package_root" \
		$(PKG_CONFIG) --cflags --libs libttl)"; \
	$(CC) -std=c11 -Wall -Wextra -Wpedantic examples/host_compare.c \
		$$pkg_flags '-Wl,-rpath,$(PACKAGE_RPATH)' -lm \
		-o "$$package_root/usr/local/bin/libttl-host-compare"; \
	"$$package_root/usr/local/bin/libttl-host-compare"; \
	$(PYTHON) tests/check_abi.py "$$package_root/usr/local/lib/libttl.so"; \
	echo '[package] staged and linked libttl using only installed files'

static-check:
	@for document in schema/*.json $$(find tests -name '*.json' -type f); do \
		$(PYTHON) -m json.tool "$$document" >/dev/null || exit; \
	done
	@PYTHONPYCACHEPREFIX=/tmp/libttl-pycache $(PYTHON) -m py_compile \
		tool/ttl_compiler.py tool/ttl-cuda tool/ttl-metal tool/ttl-tilelang \
		tool/torch_flags.py tool/embed_schema.py scripts/install_jsoncons.py
	@$(CC) -std=c11 -Wall -Wextra -Wpedantic -Werror \
		-Iinclude -fsyntax-only examples/host_compare.c
	@PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m unittest discover -s tests

archive-check:
	@git rev-parse --is-inside-work-tree >/dev/null 2>&1 || { \
		echo 'archive-check requires a Git worktree' >&2; exit 2; \
	}
	@test -z "$$(git status --porcelain --untracked-files=all)" || { \
		echo 'archive-check requires a clean worktree' >&2; exit 2; \
	}
	@set -euo pipefail; \
	archive_root="$$(mktemp -d "$${TMPDIR:-/tmp}/libttl-archive.XXXXXX")"; \
	trap 'rm -rf "$$archive_root"' EXIT; \
	mkdir -p "$$archive_root/source" "$$archive_root/state"; \
	git archive --format=tar HEAD | tar -xf - -C "$$archive_root/source"; \
	$(MAKE) -C "$$archive_root/source" static-check contract-check \
		PYTHON='$(PYTHON)' CC='$(CC)' \
		TTL_STATE_ROOT="$$archive_root/state" \
		TTL_BUILD_DIR="$$archive_root/state/build" \
		JSONCONS_ROOT='$(JSONCONS_ROOT)'; \
	echo '[archive] committed source is self-contained and passes static checks'

backend-check: tools-check examples-check package-check api-check

release-check: archive-check backend-check
