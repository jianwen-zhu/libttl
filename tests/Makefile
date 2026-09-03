TTL_BUILD_DIR ?= /content/.libttl/build
TTL_BACKEND ?= libtorch_cuda
TTL_LIBRARY_DIR := $(TTL_BUILD_DIR)/$(TTL_BACKEND)
PYTHON ?= python3
PROGRAM := $(TTL_LIBRARY_DIR)/tests/runtime_module_tests
CONTRACT_CHECKER := $(TTL_BUILD_DIR)/ttl-contract-check
VALID_MANIFESTS := $(wildcard fixtures/*/02_device/module.json) \
	$(wildcard fixtures/*/03_tilelang/module.json)
CAPABILITY_MODULE := capability_modules/$(TTL_BACKEND)
ifeq ($(TTL_BACKEND),libtorch_cuda)
BACKEND_DEFINE := -DTTL_TEST_CUDA=1
else ifeq ($(TTL_BACKEND),libtorch_mps)
BACKEND_DEFINE := -DTTL_TEST_MPS=1
else
$(error unsupported TTL_BACKEND '$(TTL_BACKEND)')
endif

.PHONY: all schema run abi

all: schema run abi

$(CONTRACT_CHECKER):
	@$(MAKE) -C ../src contract-checker TTL_BUILD_DIR='$(TTL_BUILD_DIR)' \
		PYTHON='$(PYTHON)'

schema: $(CONTRACT_CHECKER)
	@for manifest in $(VALID_MANIFESTS); do \
		$(CONTRACT_CHECKER) "$$manifest" || exit; \
	done
	@for manifest in malformed_modules/bad_schema_version/module.json \
		malformed_modules/duplicate_tensor/module.json \
		malformed_modules/unknown_binding/module.json; do \
		if $(CONTRACT_CHECKER) "$$manifest" >/dev/null 2>&1; then \
			echo "shared validator unexpectedly accepted $$manifest" >&2; \
			exit 1; \
		fi; \
	done
	@$(CONTRACT_CHECKER) --program programs/identity/program.json
	@$(CONTRACT_CHECKER) --fixture fixtures/minimal.fixture.json
	@for program in malformed_programs/*/program.json; do \
		if $(CONTRACT_CHECKER) --program "$$program" >/dev/null 2>&1; then \
			echo "shared validator unexpectedly accepted $$program" >&2; \
			exit 1; \
		fi; \
	done
	@echo 'TTL JSON CONTRACT VALIDATION PASSED'

$(PROGRAM): runtime_module_tests.c $(TTL_LIBRARY_DIR)/libttl.so
	@mkdir -p '$(dir $(PROGRAM))'
	@$(CC) -std=c11 -O2 -Wall -Wextra -Wpedantic -I../include \
		$(BACKEND_DEFINE) runtime_module_tests.c -L$(TTL_LIBRARY_DIR) -lttl \
		'-Wl,-rpath,$$ORIGIN/..' -o $(PROGRAM)

run: $(PROGRAM)
	@$(PROGRAM) malformed_modules $(CAPABILITY_MODULE)

abi: $(TTL_LIBRARY_DIR)/libttl.so
	@$(PYTHON) check_abi.py $(TTL_LIBRARY_DIR)/libttl.so

