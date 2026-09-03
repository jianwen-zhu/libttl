#!/usr/bin/env python3
"""Install pinned jsoncons headers into external project state."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath


def install(lock_path: Path, destination: Path) -> bool:
    lock_bytes = lock_path.read_bytes()
    lock = json.loads(lock_bytes)
    stamp_value = hashlib.sha256(lock_bytes).hexdigest()
    stamp = destination / ".jsoncons-lock-sha256"
    legacy_stamp = destination / ".ttc-lock-sha256"
    accepted_stamps = (stamp, legacy_stamp)
    if destination.is_dir() and any(
            candidate.is_file() and
            candidate.read_text().strip() == stamp_value
            for candidate in accepted_stamps):
        return False
    if destination.exists():
        raise RuntimeError(
            f"incomplete or mismatched dependency at {destination}; "
            "move it aside and retry")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.", dir=destination.parent))
    archive = temporary / "source.tar.gz"
    extracted = temporary / "package"
    try:
        with urllib.request.urlopen(lock["url"]) as response:
            archive.write_bytes(response.read())
        actual = hashlib.sha256(archive.read_bytes()).hexdigest()
        if actual != lock["sha256"]:
            raise RuntimeError(
                f"jsoncons archive checksum mismatch: {actual}")

        with tarfile.open(archive, "r:gz") as source:
            members = source.getmembers()
            roots = {PurePosixPath(member.name).parts[0] for member in members
                     if member.name}
            if len(roots) != 1:
                raise RuntimeError("jsoncons archive has an unexpected layout")
            root = next(iter(roots))
            prefix = PurePosixPath(root, "include")
            for member in members:
                path = PurePosixPath(member.name)
                if path == prefix or prefix not in path.parents:
                    continue
                relative = path.relative_to(PurePosixPath(root))
                target = extracted.joinpath(*relative.parts)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    file_object = source.extractfile(member)
                    if file_object is None:
                        raise RuntimeError(f"cannot extract {member.name}")
                    with file_object, target.open("wb") as output:
                        shutil.copyfileobj(file_object, output)
                else:
                    raise RuntimeError(
                        f"unsupported archive entry: {member.name}")
        if not (extracted / "include/jsoncons/json.hpp").is_file():
            raise RuntimeError("jsoncons archive lacks include/jsoncons/json.hpp")
        (extracted / ".jsoncons-lock-sha256").write_text(stamp_value + "\n")
        extracted.rename(destination)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    arguments = parser.parse_args()
    changed = install(arguments.lock, arguments.destination)
    action = "installed" if changed else "reusing"
    print(f"[bootstrap] {action} jsoncons at {arguments.destination}")


if __name__ == "__main__":
    main()
