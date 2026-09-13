PYTHON ?= python3

.PHONY: all check

all: check

check:
	PYTHONPYCACHEPREFIX=/tmp/libttl-pycache $(PYTHON) -m py_compile ttl_compiler.py ttl-cuda ttl-metal ttl-tilelang torch_flags.py
	@PYTHONDONTWRITEBYTECODE=1 $(PYTHON) ./ttl-cuda --version
	@PYTHONDONTWRITEBYTECODE=1 $(PYTHON) ./ttl-metal --version
	@PYTHONDONTWRITEBYTECODE=1 $(PYTHON) ./ttl-tilelang --version
