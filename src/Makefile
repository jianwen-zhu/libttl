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
PLATFORM := $(shell uname -s)
BACKEND_FAMILY := $(firstword $(subst _, ,$(BACKEND)))
BACKEND_CONFIG := $(firstword $(wildcard \
	backend_$(BACKEND).mk backend_$(BACKEND_FAMILY).mk))
BACKEND_SUPPORT_FILES :=
BACKEND_COMPILE_FLAGS_COMMAND := :
BACKEND_LINK_FLAGS_COMMAND := :
-include $(BACKEND_CONFIG)
API_VERSION_MAJOR := $(shell awk \
	'$$2 == "TTL_API_VERSION_MAJOR" { sub(/u$$/, "", $$3); print $$3 }' \
	../include/libttl.h)
API_VERSION_MINOR := $(shell awk \
	'$$2 == "TTL_API_VERSION_MINOR" { sub(/u$$/, "", $$3); print $$3 }' \
	../include/libttl.h)
API_VERSION_PATCH := $(shell awk \
	'$$2 == "TTL_API_VERSION_PATCH" { sub(/u$$/, "", $$3); print $$3 }' \
	../include/libttl.h)
API_VERSION := $(API_VERSION_MAJOR).$(API_VERSION_MINOR).$(API_VERSION_PATCH)
API_COMPATIBILITY_VERSION := $(API_VERSION_MAJOR).0.0
SOURCE := $(firstword $(wildcard backend_$(BACKEND).cc backend_$(BACKEND).mm))
MANIFEST_SOURCE := manifest_json.cc
CONTRACT_CHECKER := $(TTL_BUILD_DIR)/ttl-contract-check
PUBLIC_EXPORTS := libttl.exports
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
DOCDIR ?= $(DATADIR)/doc/libttl
PKGCONFIGDIR ?= $(LIBDIR)/pkgconfig
PACKAGE_VERSION := $(strip $(shell cat ../VERSION))

ifeq ($(PLATFORM),Darwin)
EXPORT_CONTROL := $(TTL_BUILD_DIR)/generated/libttl.exported-symbols
SHARED_LINK_FLAGS := -Wl,-exported_symbols_list,$(EXPORT_CONTROL) \
	-Wl,-install_name,@rpath/libttl.so \
	-Wl,-current_version,$(API_VERSION) \
	-Wl,-compatibility_version,$(API_COMPATIBILITY_VERSION)
else
EXPORT_CONTROL := $(TTL_BUILD_DIR)/generated/libttl.version-script
SHARED_LINK_FLAGS := -Wl,--version-script,$(EXPORT_CONTROL)
endif

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

ifeq ($(PLATFORM),Darwin)
$(EXPORT_CONTROL): $(PUBLIC_EXPORTS)
	@mkdir -p '$(dir $(EXPORT_CONTROL))'
	@awk 'NF { print "_" $$0 }' '$(PUBLIC_EXPORTS)' > '$@.tmp'
	@mv '$@.tmp' '$@'
else
$(EXPORT_CONTROL): $(PUBLIC_EXPORTS)
	@mkdir -p '$(dir $(EXPORT_CONTROL))'
	@awk 'BEGIN { print "{"; print "  global:" } \
		NF { print "    " $$0 ";" } \
		END { print "  local:"; print "    *;"; print "};" }' \
		'$(PUBLIC_EXPORTS)' > '$@.tmp'
	@mv '$@.tmp' '$@'
endif

$(OUTPUT): Makefile $(BACKEND_CONFIG) $(SOURCE) $(BACKEND_SUPPORT_FILES) \
		$(MANIFEST_SOURCE) manifest_json.h ../include/libttl.h $(EXPORT_CONTROL) \
		$(MODULE_SCHEMA_HEADER) $(LAUNCH_SCHEMA_HEADER) \
		$(PROGRAM_SCHEMA_HEADER) $(FIXTURE_SCHEMA_HEADER)
	@mkdir -p '$(dir $(OUTPUT))'
	@set -eu; \
	compile_flags="$$( $(BACKEND_COMPILE_FLAGS_COMMAND) )"; \
	link_flags="$$( $(BACKEND_LINK_FLAGS_COMMAND) )"; \
	$(CXX) -std=c++17 -O2 -fPIC -Wall -Wextra \
		-I../include -I'$(JSONCONS_INCLUDE_DIR)' \
		-I'$(dir $(MODULE_SCHEMA_HEADER))' $$compile_flags -shared \
		$(SOURCE) $(MANIFEST_SOURCE) -o $(OUTPUT) \
		$$link_flags $(SHARED_LINK_FLAGS)
	@echo '[libttl] ready: $(OUTPUT)'

install: all
	@install -d '$(DESTDIR)$(LIBDIR)' '$(DESTDIR)$(INCLUDEDIR)' \
		'$(DESTDIR)$(DATADIR)/libttl/schema' '$(DESTDIR)$(DOCDIR)' \
		'$(DESTDIR)$(PKGCONFIGDIR)'
	@install -m 755 '$(OUTPUT)' '$(DESTDIR)$(LIBDIR)/libttl.so'
	@install -m 644 ../include/libttl.h '$(DESTDIR)$(INCLUDEDIR)/libttl.h'
	@install -m 644 ../schema/*.schema.json \
		'$(DESTDIR)$(DATADIR)/libttl/schema/'
	@install -m 644 ../LICENSE ../THIRD_PARTY_NOTICES.md \
		'$(DESTDIR)$(DOCDIR)/'
	@sed -e 's|@PREFIX@|$(PREFIX)|g' \
		-e 's|@LIBDIR@|$(LIBDIR)|g' \
		-e 's|@INCLUDEDIR@|$(INCLUDEDIR)|g' \
		-e 's|@VERSION@|$(PACKAGE_VERSION)|g' \
		libttl.pc.in > '$(DESTDIR)$(PKGCONFIGDIR)/libttl.pc'
	@echo '[libttl] installed under $(DESTDIR)$(PREFIX)'

clean:
	@echo 'Build state is external; remove $(OUTPUT) and $(CONTRACT_CHECKER) explicitly if needed.'
