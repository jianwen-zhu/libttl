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
SMOKE_PROGRAM := $(SMOKE_BUILD_ROOT)/program
BINDING_STAMP := $(BINDING_MODULE)/.built
IDENTITY_STAMP := $(IDENTITY_MODULE)/.built
DYNAMIC_STAMP := $(DYNAMIC_MODULE)/.built
PROGRAM_STAMP := $(SMOKE_PROGRAM)/.built
ifeq ($(TTL_BACKEND),libtorch_cuda)
SCALAR_STRESS_STAMP := $(SCALAR_STRESS_MODULE)/.built
SCALAR_STRESS_ARGUMENT := $(SCALAR_STRESS_MODULE)
else
SCALAR_STRESS_STAMP :=
SCALAR_STRESS_ARGUMENT := -
endif
SMOKE_STAMPS := $(BINDING_STAMP) $(IDENTITY_STAMP) $(DYNAMIC_STAMP) \
	$(PROGRAM_STAMP) $(SCALAR_STRESS_STAMP)

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

host: $(HOST_TEST)
	@$(HOST_TEST)

run: $(DEVICE_TEST) $(SMOKE_STAMPS)
	@$(DEVICE_TEST) malformed_modules malformed_programs \
		'$(BINDING_MODULE)' '$(DYNAMIC_MODULE)' '$(SMOKE_PROGRAM)' \
		'$(SCALAR_STRESS_ARGUMENT)'

abi: $(TTL_LIBRARY_DIR)/libttl.so
	@$(PYTHON) check_abi.py $(TTL_LIBRARY_DIR)/libttl.so
