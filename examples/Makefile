CC ?= cc
TTL_BACKEND ?= libtorch_cuda
TTL_BUILD_DIR ?= $(HOME)/.cache/libttl/build
LIBTTL_INCLUDE_DIR ?= ../include
LIBTTL_LIBRARY_DIR ?= $(TTL_BUILD_DIR)/$(TTL_BACKEND)
OUTPUT_DIR ?= $(TTL_BUILD_DIR)/$(TTL_BACKEND)/examples
TARGET := $(OUTPUT_DIR)/host_compare

ifeq ($(shell uname -s),Darwin)
EXAMPLE_RPATH := @loader_path/..
else
EXAMPLE_RPATH := $$ORIGIN/..
endif

.PHONY: all check

all: $(TARGET)

$(TARGET): host_compare.c $(LIBTTL_INCLUDE_DIR)/libttl.h \
		$(LIBTTL_LIBRARY_DIR)/libttl.so
	@mkdir -p '$(OUTPUT_DIR)'
	@$(CC) -std=c11 -O2 -Wall -Wextra -Wpedantic \
		-I'$(LIBTTL_INCLUDE_DIR)' host_compare.c \
		-L'$(LIBTTL_LIBRARY_DIR)' -lttl \
		'-Wl,-rpath,$(EXAMPLE_RPATH)' -lm -o '$(TARGET)'

check: $(TARGET)
	@'$(TARGET)'
