CXX ?= c++
ifeq ($(wildcard /content),/content)
TTL_STATE_ROOT ?= /content/.libttl
else
TTL_STATE_ROOT ?= $(HOME)/.cache/libttl
endif
PYTHON ?= python3
TTL_BUILD_DIR ?= $(TTL_STATE_ROOT)/build
BACKEND ?= libtorch_cuda
OUTPUT := $(TTL_BUILD_DIR)/$(BACKEND)/libttl.so
SOURCE := $(firstword $(wildcard backend_$(BACKEND).cc backend_$(BACKEND).mm))
MANIFEST_SOURCE := manifest_json.cc
CONTRACT_CHECKER := $(TTL_BUILD_DIR)/ttl-contract-check
MODULE_SCHEMA := ../schema/ttl-kernel-module-v1.schema.json
MODULE_SCHEMA_HEADER := $(TTL_BUILD_DIR)/generated/ttl_kernel_module_schema.h
LAUNCH_SCHEMA := ../schema/ttl-kernel-launch-v1.schema.json
LAUNCH_SCHEMA_HEADER := $(TTL_BUILD_DIR)/generated/ttl_kernel_launch_schema.h
PROGRAM_SCHEMA := ../schema/ttl-program-v1.schema.json
PROGRAM_SCHEMA_HEADER := $(TTL_BUILD_DIR)/generated/ttl_program_schema.h
FIXTURE_SCHEMA := ../schema/ttl-fixture-v1.schema.json
FIXTURE_SCHEMA_HEADER := $(TTL_BUILD_DIR)/generated/ttl_fixture_schema.h
JSONCONS_INCLUDE_DIR ?= $(TTL_STATE_ROOT)/deps/jsoncons-1.9.0/include
PREFIX ?= /usr/local
LIBDIR ?= $(PREFIX)/lib
INCLUDEDIR ?= $(PREFIX)/include
DATADIR ?= $(PREFIX)/share

.PHONY: all validate-backend contract-checker install clean

all: validate-backend $(CONTRACT_CHECKER) $(OUTPUT)

validate-backend:
	@test -n '$(SOURCE)' || { \
		echo 'unknown TTL backend: $(BACKEND)' >&2; exit 2; \
	}

$(MODULE_SCHEMA_HEADER): $(MODULE_SCHEMA) ../tool/embed_schema.py
	@mkdir -p '$(dir $(MODULE_SCHEMA_HEADER))'
	@$(PYTHON) ../tool/embed_schema.py \
		--input '$(MODULE_SCHEMA)' --output '$(MODULE_SCHEMA_HEADER)'

$(LAUNCH_SCHEMA_HEADER): $(LAUNCH_SCHEMA) ../tool/embed_schema.py
	@mkdir -p '$(dir $(LAUNCH_SCHEMA_HEADER))'
	@$(PYTHON) ../tool/embed_schema.py \
		--input '$(LAUNCH_SCHEMA)' --output '$(LAUNCH_SCHEMA_HEADER)' \
		--symbol kernel_launch_schema

$(PROGRAM_SCHEMA_HEADER): $(PROGRAM_SCHEMA) ../tool/embed_schema.py
	@mkdir -p '$(dir $(PROGRAM_SCHEMA_HEADER))'
	@$(PYTHON) ../tool/embed_schema.py \
		--input '$(PROGRAM_SCHEMA)' --output '$(PROGRAM_SCHEMA_HEADER)' \
		--symbol program_schema_data

$(FIXTURE_SCHEMA_HEADER): $(FIXTURE_SCHEMA) ../tool/embed_schema.py
	@mkdir -p '$(dir $(FIXTURE_SCHEMA_HEADER))'
	@$(PYTHON) ../tool/embed_schema.py \
		--input '$(FIXTURE_SCHEMA)' --output '$(FIXTURE_SCHEMA_HEADER)' \
		--symbol fixture_schema_data

$(CONTRACT_CHECKER): manifest_check.cc $(MANIFEST_SOURCE) manifest_json.h \
		$(MODULE_SCHEMA_HEADER) $(LAUNCH_SCHEMA_HEADER) \
		$(PROGRAM_SCHEMA_HEADER) $(FIXTURE_SCHEMA_HEADER)
	@test -f '$(JSONCONS_INCLUDE_DIR)/jsoncons/json.hpp' || { \
		echo 'jsoncons headers are missing; run make bootstrap' >&2; exit 2; \
	}
	@mkdir -p '$(TTL_BUILD_DIR)'
	@$(CXX) -std=c++17 -O2 -Wall -Wextra \
		-I'$(JSONCONS_INCLUDE_DIR)' -I'$(dir $(MODULE_SCHEMA_HEADER))' \
		manifest_check.cc $(MANIFEST_SOURCE) -o '$(CONTRACT_CHECKER)'
	@echo '[ttl-contract-check] ready: $(CONTRACT_CHECKER)'

contract-checker: $(CONTRACT_CHECKER)

$(OUTPUT): $(SOURCE) backend_libtorch_impl.inc $(MANIFEST_SOURCE) manifest_json.h \
		../include/libttl.h ../tool/torch_flags.py \
		$(MODULE_SCHEMA_HEADER) $(LAUNCH_SCHEMA_HEADER) \
		$(PROGRAM_SCHEMA_HEADER) $(FIXTURE_SCHEMA_HEADER)
	@mkdir -p '$(dir $(OUTPUT))'
	@compile_flags="$$( $(PYTHON) ../tool/torch_flags.py compile '$(BACKEND)' )"; \
	link_flags="$$( $(PYTHON) ../tool/torch_flags.py link '$(BACKEND)' )"; \
	$(CXX) -std=c++17 -O2 -fPIC -Wall -Wextra \
		-I../include -I'$(JSONCONS_INCLUDE_DIR)' \
		-I'$(dir $(MODULE_SCHEMA_HEADER))' $$compile_flags -shared \
		$(SOURCE) $(MANIFEST_SOURCE) -o $(OUTPUT) \
		$$link_flags
	@echo '[libttl] ready: $(OUTPUT)'

install: all
	@install -d '$(DESTDIR)$(LIBDIR)' '$(DESTDIR)$(INCLUDEDIR)' \
		'$(DESTDIR)$(DATADIR)/libttl/schema'
	@install -m 755 '$(OUTPUT)' '$(DESTDIR)$(LIBDIR)/libttl.so'
	@install -m 644 ../include/libttl.h '$(DESTDIR)$(INCLUDEDIR)/libttl.h'
	@install -m 644 ../schema/*.schema.json \
		'$(DESTDIR)$(DATADIR)/libttl/schema/'
	@echo '[libttl] installed under $(DESTDIR)$(PREFIX)'

clean:
	@echo 'Build state is external; remove $(OUTPUT) and $(CONTRACT_CHECKER) explicitly if needed.'
