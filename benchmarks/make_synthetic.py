"""Generate independent FP32 unit-vector data for local CAGRA stress testing."""

import argparse
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path)
parser.add_argument("--rows", type=int, default=250000)
parser.add_argument("--dimension", type=int, default=256)
parser.add_argument("--queries", type=int, default=100)
parser.add_argument("--seed", type=int, default=20261031)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(args.seed)
for name, rows in (("base", args.rows), ("query", args.queries)):
    values = rng.standard_normal((rows, args.dimension), dtype=np.float32)
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    values.tofile(args.output / f"{name}.f32")
manifest = {
    "distribution": "independent standard Gaussian, normalized to unit L2 length",
    "rows": args.rows, "dimension": args.dimension, "queries": args.queries, "seed": args.seed,
    "purpose": "development stress data; not competition data or an official score",
}
(args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest))
