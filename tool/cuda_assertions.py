"""Lower TileLang 0.1.13 host-style assertions to CUDA assertions.

TileLang 0.1.13 can emit its C-host AssertStmt body into CUDA device source.
Recognize only the known, single-literal RuntimeError form and preserve it as a
native CUDA device assertion.  Unfamiliar host-error shapes are rejected.
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


def lower_assertions(source: str) -> tuple[str, int]:
    """Return CUDA source with known host assertions lowered fail-closed."""
    has_host_assertion = any(marker in source for marker in _ASSERTION_MARKERS)
    if not has_host_assertion:
        return source, 0
    if re.search(r"^\s*#\s*define\s+NDEBUG\b", source, re.MULTILINE):
        raise ValueError("checked CUDA source cannot disable device assertions")

    def replacement(match: re.Match[str]) -> str:
        indent = match.group("indent")
        return (
            f"{indent}assert(0 && {match.group('message')});\n"
            f"{indent}return;")

    lowered, count = _ASSERTION.subn(replacement, source)
    if count == 0 or any(marker in lowered for marker in _ASSERTION_MARKERS):
        raise ValueError(
            "unrecognized TileLang CUDA assertion; refusing unchecked compilation")
    return "#include <assert.h>\n" + lowered, count
