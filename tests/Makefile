TTL_BUILD_DIR ?= /content/.libttl/build
TTL_BACKEND ?= libtorch_cuda
TTL_LIBRARY_DIR := $(TTL_BUILD_DIR)/$(TTL_BACKEND)
PYTHON ?= python3
PLATFORM := $(shell uname -s)
HOST_TEST := $(TTL_LIBRARY_DIR)/tests/api_host_tests
DEVICE_TEST := $(TTL_LIBRARY_DIR)/tests/api_device_tests
CONTRACT_CHECKER := $(TTL_BUILD_DIR)/ttl-contract-check
VALID_MANIFESTS := $(wildcard fixtures/*/02_device/module.json) \
	$(wildcard fixtures/*/03_tilelang/module.json) \
	$(wildcard smoke/*/module.json)
VALID_FIXTURES := fixtures/minimal.fixture.json fixtures/semantic.fixture.json \
	fixtures/identity.fixture.json
MALFORMED_FIXTURES := $(wildcard malformed_fixtures/*.fixture.json)
VALID_LAUNCHES := $(wildcard launches/*.launch.json)
MALFORMED_LAUNCHES := $(wildcard malformed_launches/*.launch.json)
ifeq ($(TTL_BACKEND),libtorch_cuda)
BACKEND_DEFINE := -DTTL_TEST_CUDA=1
PACKAGER := ../tool/ttl-cuda
NATIVE_SUFFIX := cu
NATIVE_ARTIFACT := kernel.cubin
else ifeq ($(TTL_BACKEND),libtorch_mps)
BACKEND_DEFINE := -DTTL_TEST_MPS=1
PACKAGER := ../tool/ttl-metal
NATIVE_SUFFIX := metal
NATIVE_ARTIFACT := kernel.metallib
else
$(error unsupported TTL_BACKEND '$(TTL_BACKEND)')
endif
ifeq ($(PLATFORM),Darwin)
TEST_RPATH := @loader_path/..
else
TEST_RPATH := $$ORIGIN/..
endif

SMOKE_BUILD_ROOT := $(TTL_LIBRARY_DIR)/tests/smoke
BINDING_MODULE := $(SMOKE_BUILD_ROOT)/binding
IDENTITY_MODULE := $(SMOKE_BUILD_ROOT)/identity
DYNAMIC_MODULE := $(SMOKE_BUILD_ROOT)/dynamic
SCALAR_STRESS_MODULE := $(SMOKE_BUILD_ROOT)/scalar_stress
ASSERT_DOMAIN_MODULE := $(SMOKE_BUILD_ROOT)/assert_domain
ASSERT_BOUNDS_MODULE := $(SMOKE_BUILD_ROOT)/assert_bounds
TILELANG_SHARED_MODULE := $(SMOKE_BUILD_ROOT)/tilelang_shared
SMOKE_PROGRAM := $(SMOKE_BUILD_ROOT)/program
BINDING_STAMP := $(BINDING_MODULE)/.built
IDENTITY_STAMP := $(IDENTITY_MODULE)/.built
DYNAMIC_STAMP := $(DYNAMIC_MODULE)/.built
PROGRAM_STAMP := $(SMOKE_PROGRAM)/.built
ASSERT_DOMAIN_STAMP := $(ASSERT_DOMAIN_MODULE)/.built
ASSERT_BOUNDS_STAMP := $(ASSERT_BOUNDS_MODULE)/.built
TILELANG_SHARED_STAMP := $(TILELANG_SHARED_MODULE)/.built
ifeq ($(TTL_BACKEND),libtorch_cuda)
SCALAR_STRESS_STAMP := $(SCALAR_STRESS_MODULE)/.built
SCALAR_STRESS_ARGUMENT := $(SCALAR_STRESS_MODULE)
ASSERTION_STAMPS :=
TILELANG_SHARED_STAMPS :=
ASSERT_DOMAIN_ARGUMENT := -
ASSERT_BOUNDS_ARGUMENT := -
TILELANG_SHARED_ARGUMENT := -
else
SCALAR_STRESS_STAMP :=
SCALAR_STRESS_ARGUMENT := -
ASSERTION_STAMPS := $(ASSERT_DOMAIN_STAMP) $(ASSERT_BOUNDS_STAMP)
TILELANG_SHARED_STAMPS := $(TILELANG_SHARED_STAMP)
ASSERT_DOMAIN_ARGUMENT := $(ASSERT_DOMAIN_MODULE)
ASSERT_BOUNDS_ARGUMENT := $(ASSERT_BOUNDS_MODULE)
TILELANG_SHARED_ARGUMENT := $(TILELANG_SHARED_MODULE)
endif
SMOKE_STAMPS := $(BINDING_STAMP) $(IDENTITY_STAMP) $(DYNAMIC_STAMP) \
	$(PROGRAM_STAMP) $(SCALAR_STRESS_STAMP) $(ASSERTION_STAMPS) \
	$(TILELANG_SHARED_STAMPS)

.PHONY: all schema host run abi

all: schema host run abi

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
	@$(CONTRACT_CHECKER) --program smoke/program/program.json
	@for fixture in $(VALID_FIXTURES); do \
		$(CONTRACT_CHECKER) --fixture "$$fixture" || exit; \
	done
	@for fixture in $(MALFORMED_FIXTURES); do \
		if $(CONTRACT_CHECKER) --fixture "$$fixture" >/dev/null 2>&1; then \
			echo "shared validator unexpectedly accepted $$fixture" >&2; \
			exit 1; \
		fi; \
	done
	@for program in malformed_programs/*/program.json; do \
		if $(CONTRACT_CHECKER) --program "$$program" >/dev/null 2>&1; then \
			echo "shared validator unexpectedly accepted $$program" >&2; \
			exit 1; \
		fi; \
	done
	@for launch in $(VALID_LAUNCHES); do \
		$(CONTRACT_CHECKER) --launch "$$launch" || exit; \
	done
	@for launch in $(MALFORMED_LAUNCHES); do \
		if $(CONTRACT_CHECKER) --launch "$$launch" >/dev/null 2>&1; then \
			echo "shared validator unexpectedly accepted $$launch" >&2; \
			exit 1; \
		fi; \
	done
	@echo 'TTL JSON CONTRACT VALIDATION PASSED'

$(HOST_TEST): api_host_tests.c $(TTL_LIBRARY_DIR)/libttl.so
	@mkdir -p '$(dir $(HOST_TEST))'
	@$(CC) -std=c11 -O2 -Wall -Wextra -Wpedantic -I../include \
		api_host_tests.c -L$(TTL_LIBRARY_DIR) -lttl \
		'-Wl,-rpath,$(TEST_RPATH)' -lm -o $(HOST_TEST)

$(DEVICE_TEST): api_device_tests.c $(TTL_LIBRARY_DIR)/libttl.so
	@mkdir -p '$(dir $(DEVICE_TEST))'
	@$(CC) -std=c11 -O2 -Wall -Wextra -Wpedantic -I../include \
		$(BACKEND_DEFINE) api_device_tests.c -L$(TTL_LIBRARY_DIR) -lttl \
		'-Wl,-rpath,$(TEST_RPATH)' -lm -o $(DEVICE_TEST)

$(BINDING_STAMP): smoke/binding/module.json smoke/binding/kernel.$(NATIVE_SUFFIX) \
		$(CONTRACT_CHECKER)
	@TTL_BACKEND='$(TTL_BACKEND)' TTL_CONTRACT_CHECKER='$(CONTRACT_CHECKER)' \
		PYTHONDONTWRITEBYTECODE=1 $(PYTHON) $(PACKAGER) \
		--source smoke/binding/kernel.$(NATIVE_SUFFIX) \
		--manifest smoke/binding/module.json --output '$(BINDING_MODULE)'
	@touch '$@'

$(IDENTITY_STAMP): smoke/identity/module.json smoke/identity/kernel.$(NATIVE_SUFFIX) \
		$(CONTRACT_CHECKER)
	@TTL_BACKEND='$(TTL_BACKEND)' TTL_CONTRACT_CHECKER='$(CONTRACT_CHECKER)' \
		PYTHONDONTWRITEBYTECODE=1 $(PYTHON) $(PACKAGER) \
		--source smoke/identity/kernel.$(NATIVE_SUFFIX) \
		--manifest smoke/identity/module.json --output '$(IDENTITY_MODULE)'
	@touch '$@'

$(DYNAMIC_STAMP): smoke/dynamic/module.json smoke/dynamic/kernel.$(NATIVE_SUFFIX) \
		$(CONTRACT_CHECKER)
	@TTL_BACKEND='$(TTL_BACKEND)' TTL_CONTRACT_CHECKER='$(CONTRACT_CHECKER)' \
		PYTHONDONTWRITEBYTECODE=1 $(PYTHON) $(PACKAGER) \
		--source smoke/dynamic/kernel.$(NATIVE_SUFFIX) \
		--manifest smoke/dynamic/module.json --output '$(DYNAMIC_MODULE)'
	@touch '$@'

$(PROGRAM_STAMP): smoke/program/program.json $(IDENTITY_STAMP)
	@install -d '$(SMOKE_PROGRAM)/functions/identity'
	@install -m 644 smoke/program/program.json '$(SMOKE_PROGRAM)/program.json'
	@install -m 644 '$(IDENTITY_MODULE)/module.json' \
		'$(IDENTITY_MODULE)/launch.json' '$(IDENTITY_MODULE)/$(NATIVE_ARTIFACT)' \
		'$(SMOKE_PROGRAM)/functions/identity/'
	@touch '$@'

ifeq ($(TTL_BACKEND),libtorch_cuda)
$(SCALAR_STRESS_STAMP): smoke/scalar_stress/module.json \
		smoke/scalar_stress/kernel.cu $(CONTRACT_CHECKER)
	@TTL_BACKEND='$(TTL_BACKEND)' TTL_CONTRACT_CHECKER='$(CONTRACT_CHECKER)' \
		PYTHONDONTWRITEBYTECODE=1 $(PYTHON) $(PACKAGER) \
		--source smoke/scalar_stress/kernel.cu \
		--manifest smoke/scalar_stress/module.json \
		--output '$(SCALAR_STRESS_MODULE)'
	@touch '$@'
endif

ifeq ($(TTL_BACKEND),libtorch_mps)
$(ASSERT_DOMAIN_STAMP): smoke/assert_domain/module.json \
		smoke/assert_domain/kernel.metal package_metal_assertion.py \
		../tool/metal_assertions.py ../tool/ttl_compiler.py $(CONTRACT_CHECKER)
	@TTL_CONTRACT_CHECKER='$(CONTRACT_CHECKER)' \
		PYTHONDONTWRITEBYTECODE=1 $(PYTHON) package_metal_assertion.py \
		--source smoke/assert_domain/kernel.metal \
		--manifest smoke/assert_domain/module.json \
		--output '$(ASSERT_DOMAIN_MODULE)'
	@touch '$@'

$(ASSERT_BOUNDS_STAMP): smoke/assert_bounds/module.json \
		smoke/assert_bounds/kernel.metal package_metal_assertion.py \
		../tool/metal_assertions.py ../tool/ttl_compiler.py $(CONTRACT_CHECKER)
	@TTL_CONTRACT_CHECKER='$(CONTRACT_CHECKER)' \
		PYTHONDONTWRITEBYTECODE=1 $(PYTHON) package_metal_assertion.py \
		--source smoke/assert_bounds/kernel.metal \
		--manifest smoke/assert_bounds/module.json \
		--output '$(ASSERT_BOUNDS_MODULE)'
	@touch '$@'

$(TILELANG_SHARED_STAMP): smoke/tilelang_shared/module.json \
		smoke/tilelang_shared/kernel.py package_tilelang_metal_shared.py \
		../tool/ttl_compiler.py ../tool/metal_assertions.py $(CONTRACT_CHECKER)
	@TTL_CONTRACT_CHECKER='$(CONTRACT_CHECKER)' \
		PYTHONDONTWRITEBYTECODE=1 $(PYTHON) package_tilelang_metal_shared.py \
		--source smoke/tilelang_shared/kernel.py \
		--manifest smoke/tilelang_shared/module.json \
		--output '$(TILELANG_SHARED_MODULE)'
	@touch '$@'
endif

host: $(HOST_TEST)
	@$(HOST_TEST)

run: $(DEVICE_TEST) $(SMOKE_STAMPS)
	@$(DEVICE_TEST) malformed_modules malformed_programs \
		'$(BINDING_MODULE)' '$(DYNAMIC_MODULE)' '$(SMOKE_PROGRAM)' \
		'$(SCALAR_STRESS_ARGUMENT)' '$(ASSERT_DOMAIN_ARGUMENT)' \
		'$(ASSERT_BOUNDS_ARGUMENT)' '$(TILELANG_SHARED_ARGUMENT)'

abi: $(TTL_LIBRARY_DIR)/libttl.so
	@$(PYTHON) check_abi.py $(TTL_LIBRARY_DIR)/libttl.so
