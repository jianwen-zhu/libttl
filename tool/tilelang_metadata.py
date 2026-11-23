"""Extract bounded launch metadata from lowered TileLang device IR."""

from __future__ import annotations

from typing import Any


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _constant_uint32(value: object, description: str) -> int:
    if type(value) is int:
        result = value
    else:
        # TileLang represents a constant attribute as a TIR IntImm. Read its
        # value instead of calling int() so symbolic PrimExprs fail closed.
        result = getattr(value, "value", None)
        _require(type(result) is int,
                 f"{description} must be a constant integer")
    _require(0 <= result <= 0xFFFFFFFF,
             f"{description} must be a nonnegative uint32 value")
    return result


def cuda_launch_from_device_ir(
    device_mod: object,
    *,
    entrypoint: str,
    manifest_symbol: str,
    authored_launch: dict[str, Any],
) -> dict[str, Any]:
    """Return a CUDA launch with TileLang's dynamic allocation attached.

    The manifest symbol, rather than a guessed ``_kernel`` suffix, selects the
    lowered device function. An authored zero remains a compatible placeholder;
    a nonzero value must agree with TileLang's computed allocation.
    """

    functions = getattr(device_mod, "functions", None)
    _require(functions is not None,
             "TileLang lowering did not return a device function map")

    matches: list[object] = []
    available: list[str] = []
    for global_var, function in functions.items():
        attrs = getattr(function, "attrs", None)
        if attrs is None:
            continue
        global_symbol = attrs.get("global_symbol")
        if global_symbol is None:
            name_hint = getattr(global_var, "name_hint", "<unnamed>")
            available.append(f"{name_hint} (missing global_symbol)")
            continue
        symbol = str(global_symbol)
        available.append(symbol)
        if symbol == manifest_symbol:
            matches.append(function)

    discovered = ", ".join(available) if available else "none"
    _require(
        len(matches) == 1,
        f"TileLang entrypoint {entrypoint!r} must lower exactly one device "
        f"function matching manifest kernel {manifest_symbol!r}; found "
        f"{len(matches)} (available: {discovered})",
    )

    attrs = matches[0].attrs
    dynamic_shared = attrs.get("dyn_shared_memory_buf")
    derived_shared = (
        0 if dynamic_shared is None else
        _constant_uint32(
            dynamic_shared,
            f"TileLang kernel {manifest_symbol!r} dyn_shared_memory_buf")
    )

    authored_shared = _constant_uint32(
        authored_launch.get("shared_bytes", 0),
        "TTL_LAUNCH shared_bytes")
    _require(
        authored_shared in (0, derived_shared),
        f"TTL_LAUNCH shared_bytes {authored_shared} disagrees with TileLang "
        f"kernel {manifest_symbol!r} dyn_shared_memory_buf {derived_shared}",
    )

    launch = dict(authored_launch)
    launch["shared_bytes"] = derived_shared
    return launch
