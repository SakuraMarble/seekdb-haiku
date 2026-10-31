"""Build isolated parameter variants and compare the real GPU bridge.

Run in the assigned AMD development container. Each measurement gets its own
process and output files; variants run consecutively to avoid GPU contention.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dimension", type=int, required=True)
    args = parser.parse_args()
    source = args.root / "candidate"
    original = (source / "seekdb_cuvs_bridge.c").read_text()
    variants = [
        ("nn100-auto512", "NN_DESCENT", 128, 256, 100, 512),
        ("nn32-g64-auto2048", "NN_DESCENT", 64, 128, 32, 2048),
        ("nn32-g64-auto4096", "NN_DESCENT", 64, 128, 32, 4096),
        ("ivfpq-g128-auto2048", "IVF_PQ", 128, 256, 100, 2048),
        ("ivfpq-g64-auto2048", "IVF_PQ", 64, 128, 100, 2048),
        ("ivfpq-g64-auto4096", "IVF_PQ", 64, 128, 100, 4096),
    ]
    for name, algorithm, degree, intermediate, iterations, candidates in variants:
        target = args.root / name
        target.mkdir(exist_ok=True)
        for filename in ["build.sh", "seekdb_cuvs_bridge.h", "gpu_worker.c", "gpu_worker.h"]:
            shutil.copy2(source / filename, target / filename)
        text = original.replace("build_algo = NN_DESCENT", f"build_algo = {algorithm}")
        text = text.replace("graph_degree = 128;", f"graph_degree = {degree};")
        text = text.replace("intermediate_graph_degree = 256;", f"intermediate_graph_degree = {intermediate};")
        text = text.replace("nn_descent_niter = 24;", f"nn_descent_niter = {iterations};")
        text = text.replace("sp->algo = SINGLE_CTA;", "sp->algo = AUTO;")
        text = text.replace("if (required_itopk < 512) required_itopk = 512;",
                            f"if (required_itopk < {candidates}) required_itopk = {candidates};")
        (target / "seekdb_cuvs_bridge.c").write_text(text)
        library = target / "libseekdb_cuvs_bridge.so"
        build = subprocess.run(["bash", str(target / "build.sh"), str(library)], capture_output=True, text=True)
        (target / "compile.log").write_text(build.stdout + build.stderr)
        print(json.dumps({"variant": name, "compile_exit": build.returncode}), flush=True)
        if build.returncode:
            continue
        result = subprocess.run([
            sys.executable, str(args.root / "cagra_bridge.py"), str(library),
            "--data", str(args.data), "--dimension", str(args.dimension), "--transient-threads",
        ], capture_output=True, text=True, env=os.environ.copy())
        (target / "benchmark.json").write_text(result.stdout)
        (target / "benchmark.stderr").write_text(result.stderr)
        print(json.dumps({"variant": name, "exit": result.returncode,
                          "output": result.stdout, "error": result.stderr[-2000:]}), flush=True)


if __name__ == "__main__":
    main()
