"""Lower TileLang 0.1.13 host-style assertions to Metal error reporting.

TileLang 0.1.13 emits its C-host AssertStmt body in Metal source.  Recognize
only that known, single-literal form and bind LibTorch's existing Metal error
buffer.  Any unfamiliar checked source is rejected rather than compiled with
missing checks.
"""

from __future__ import annotations

import re


_LITERAL = r'"(?:[^"\\]|\\.)*"'
_ASSERTION = re.compile(
    r"(?P<indent>^[ \t]*)"
    r"const\s+char\s*\*\s*__tvm_assert_parts\s*\[\s*1\s*\]\s*=\s*"
    r"\{\s*(?P<message>" + _LITERAL + r")\s*\}\s*;\s*"
    r"TVMFFIErrorSetRaisedFromCStrParts\s*\(\s*\"RuntimeError\"\s*,\s*"
    r"__tvm_assert_parts\s*,\s*1\s*\)\s*;\s*"
    r"return\s+-1\s*;",
    re.MULTILINE,
)
_ASSERTION_MARKERS = (
    "__tvm_assert_parts",
    "TVMFFIErrorSetRaisedFromCStrParts",
)


def _matching(source: str, opening: int, left: str, right: str) -> int:
    depth = 1
    cursor = opening + 1
    while cursor < len(source) and depth:
        if source[cursor] == left:
            depth += 1
        elif source[cursor] == right:
            depth -= 1
        cursor += 1
    if depth:
        raise ValueError("unterminated checked Metal kernel")
    return cursor - 1


def _kernel_bounds(source: str, kernel: str) -> tuple[int, int, int, int]:
    patterns = (
        rf"\bkernel\s+void\s+{re.escape(kernel)}\s*\(",
        rf"\[\[\s*kernel(?:\s*,[^]]+)?\]\]\s+void\s+"
        rf"{re.escape(kernel)}\s*\(",
    )
    match = next(
        (candidate for pattern in patterns
         if (candidate := re.search(pattern, source))),
        None,
    )
    if match is None:
        raise ValueError("cannot locate checked Metal kernel")
    signature_open = match.end() - 1
    signature_close = _matching(source, signature_open, "(", ")")
    body_open = source.find("{", signature_close + 1)
    if body_open < 0 or ";" in source[signature_close + 1:body_open]:
        raise ValueError("cannot locate checked Metal kernel body")
    body_close = _matching(source, body_open, "{", "}")
    return signature_open, signature_close, body_open, body_close


def lower_assertions(
    source: str, kernel: str, parameter_count: int
) -> tuple[str, int | None]:
    """Return checked Metal source and its private error-buffer index."""
    has_host_assertion = any(marker in source for marker in _ASSERTION_MARKERS)
    if not has_host_assertion:
        if re.search(r"\bassert\s*\(", source):
            raise ValueError(
                "unsupported Metal assertion form; checks cannot be disabled")
        return source, None

    if not isinstance(parameter_count, int) or not 0 <= parameter_count <= 30:
        raise ValueError("Metal error buffer binding is unavailable")
    if "__ttl_errors" in source:
        raise ValueError("Metal error buffer name is already in use")
    if re.search(r"\b(?:threadgroup_barrier|simdgroup_barrier)\s*\(", source):
        raise ValueError(
            "checked Metal kernels with barriers require collective error lowering")
    if re.search(r"\bassert\s*\(", source):
        raise ValueError(
            "unsupported Metal assertion form; checks cannot be disabled")

    signature_open, signature_close, body_open, body_close = _kernel_bounds(
        source, kernel)
    outside = source[:body_open + 1] + source[body_close:]
    if any(marker in outside for marker in _ASSERTION_MARKERS):
        raise ValueError("TileLang assertion appears outside the selected kernel")

    body = source[body_open + 1:body_close]

    def replacement(match: re.Match[str]) -> str:
        indent = match.group("indent")
        return (
            f"{indent}TORCH_REPORT_ERROR(__ttl_errors, "
            f"{match.group('message')});\n{indent}return;")

    lowered_body, count = _ASSERTION.subn(replacement, body)
    if count == 0 or any(marker in lowered_body for marker in _ASSERTION_MARKERS):
        raise ValueError(
            "unrecognized TileLang Metal assertion; refusing unchecked compilation")

    source = source[:body_open + 1] + lowered_body + source[body_close:]
    separator = ", " if source[signature_open + 1:signature_close].strip() else ""
    parameter = (
        separator + "device c10::metal::ErrorMessages* __ttl_errors "
        f"[[buffer({parameter_count})]]")
    source = source[:signature_close] + parameter + source[signature_close:]
    includes = "#include <metal_stdlib>\n#include <c10/metal/error.h>\n"
    return includes + source, parameter_count
