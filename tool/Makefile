PYTHON ?= python3

.PHONY: all check

all: check

check:
	PYTHONPYCACHEPREFIX=/tmp/libttl-pycache $(PYTHON) -m py_compile ttl_compiler.py ttl-cuda ttl-metal ttl-tilelang torch_flags.py
	@$(PYTHON) ./ttl-cuda --version
	@$(PYTHON) ./ttl-metal --version
	@$(PYTHON) ./ttl-tilelang --version
