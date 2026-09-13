# Private build requirements shared by the LibTorch-backed implementations.
BACKEND_SUPPORT_FILES += backend_libtorch_impl.inc ../tool/torch_flags.py
BACKEND_COMPILE_FLAGS_COMMAND = \
	$(PYTHON) ../tool/torch_flags.py compile '$(BACKEND)'
BACKEND_LINK_FLAGS_COMMAND = \
	$(PYTHON) ../tool/torch_flags.py link '$(BACKEND)'
