"""Package a captured TileLang 0.1.13 Metal assertion fixture."""

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tool"))

from ttl_compiler import emit_metal_module  # noqa: E402


parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

emit_metal_module(
    source=args.source,
    manifest_path=args.manifest,
    output=args.output,
    compiler_name="ttl-tilelang-assertion-test",
    tilelang_assertions=True,
)
