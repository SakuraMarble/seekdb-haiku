"""Compare graph builders and FP32 IVF-PQ distances on both development corpora."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


root = Path(sys.argv[1])
reference = root / "ivfpq-g64-auto4096"
original = (reference / "seekdb_cuvs_bridge.c").read_text()
variants = [("ivfpq-f32-r2", 2, 4096), ("ivfpq-f32-r4", 4, 4096),
            ("nn32-g32-4096", 0, 4096), ("nn32-g32-8192", 0, 8192)]
if len(sys.argv) > 2:
    variants = [variant for variant in variants if variant[0].startswith(sys.argv[2])]

for name, refinement, candidates in variants:
    target = root / name
    target.mkdir(exist_ok=True)
    for filename in ["build.sh", "seekdb_cuvs_bridge.h", "gpu_worker.c", "gpu_worker.h"]:
        shutil.copy2(reference / filename, target / filename)
    text = original
    if refinement:
        text = text.replace("#include <cuvs/neighbors/cagra.h>",
                            "#include <cuvs/neighbors/cagra.h>\n#include <cuvs/neighbors/ivf_pq.h>")
        before = "  if (cuvsCagraBuild(h->res, h->ip, &dset, h->idx) != CUVS_SUCCESS) goto fail_idx;"
        after = """  cuvsIvfPqIndexParams_t pq_build = NULL;
  cuvsIvfPqSearchParams_t pq_search = NULL;
  struct cuvsIvfPqParams graph_params = {0};
  if (cuvsIvfPqIndexParamsCreate(&pq_build) != CUVS_SUCCESS) goto fail_idx;
  if (cuvsIvfPqSearchParamsCreate(&pq_search) != CUVS_SUCCESS) {
    cuvsIvfPqIndexParamsDestroy(pq_build); goto fail_idx;
  }
  pq_build->n_lists = (uint32_t)((n + 1023) / 1024);
  pq_search->n_probes = pq_build->n_lists < 32 ? pq_build->n_lists : 32;
  pq_search->lut_dtype = HIP_R_32F;
  pq_search->internal_distance_dtype = HIP_R_32F;
  pq_search->coarse_search_dtype = HIP_R_32F;
  graph_params.ivf_pq_build_params = pq_build;
  graph_params.ivf_pq_search_params = pq_search;
  graph_params.refinement_rate = REFINEMENT;
  h->ip->graph_build_params = &graph_params;
  cuvsError_t build_status = cuvsCagraBuild(h->res, h->ip, &dset, h->idx);
  h->ip->graph_build_params = NULL;
  cuvsIvfPqSearchParamsDestroy(pq_search);
  cuvsIvfPqIndexParamsDestroy(pq_build);
  if (build_status != CUVS_SUCCESS) goto fail_idx;""".replace("REFINEMENT", str(refinement) + ".0f")
        assert text.count(before) == 1
        text = text.replace(before, after)
    else:
        text = text.replace("build_algo = IVF_PQ", "build_algo = NN_DESCENT")
        text = text.replace("h->ip->graph_degree = 64;", "h->ip->graph_degree = 32;")
        text = text.replace("h->ip->intermediate_graph_degree = 128;", "h->ip->intermediate_graph_degree = 64;")
        text = text.replace("nn_descent_niter = 100;", "nn_descent_niter = 32;")
        text = text.replace("if (required_itopk < 4096) required_itopk = 4096;",
                            f"if (required_itopk < {candidates}) required_itopk = {candidates};")
    (target / "seekdb_cuvs_bridge.c").write_text(text)
    library = target / "libseekdb_cuvs_bridge.so"
    build = subprocess.run(["bash", str(target / "build.sh"), str(library)], capture_output=True, text=True)
    (target / "compile.log").write_text(build.stdout + build.stderr)
    print(json.dumps({"variant": name, "compile_exit": build.returncode,
                      "error": build.stderr[-2000:]}), flush=True)
    if build.returncode:
        continue
    for dataset, dimension in [("sift250k", 128), ("synthetic250k-d256", 256)]:
        result = subprocess.run([
            sys.executable, str(root / "cagra_bridge.py"), str(library),
            "--data", str(root / dataset), "--dimension", str(dimension), "--transient-threads",
        ], capture_output=True, text=True, env=os.environ.copy())
        (target / (dataset + ".json")).write_text(result.stdout)
        (target / (dataset + ".stderr")).write_text(result.stderr)
        print(json.dumps({"variant": name, "dataset": dataset, "exit": result.returncode,
                          "output": result.stdout, "error": result.stderr[-2000:]}), flush=True)
