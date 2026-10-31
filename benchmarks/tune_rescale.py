"""Build a graph after uniform rescaling, then retain original FP32 search data.

A uniform scale preserves Euclidean neighbour order. Keeping original vectors
in the final CAGRA index also preserves query distances and snapshot format.
"""

from pathlib import Path
import json
import os
import shutil
import subprocess
import sys


root = Path(sys.argv[1])
reference = root / "ivfpq-g64-auto4096"
target = root / "ivfpq-rescale"
target.mkdir(exist_ok=True)
for filename in ["build.sh", "seekdb_cuvs_bridge.h", "gpu_worker.c", "gpu_worker.h"]:
    shutil.copy2(reference / filename, target / filename)
text = (reference / "seekdb_cuvs_bridge.c").read_text()
text = text.replace("#include <stdio.h>", "#include <stdio.h>\n#include <math.h>")
before = "  DLManagedTensor dset; long ds[2] = {n, dim};"
after = """  float *scaled = NULL;
  float maximum = 1.0f;
  const size_t elements = (size_t)n * dim;
  for (size_t i = 0; i < elements; ++i) {
    const float value = fabsf(base[i]);
    if (value > maximum) maximum = value;
  }
  if (maximum > 1.0f) {
    int exponent = 0;
    frexpf(maximum, &exponent);
    const float scale = ldexpf(1.0f, -exponent);
    scaled = (float*)malloc(elements * sizeof(float));
    if (!scaled) goto fail_res;
    for (size_t i = 0; i < elements; ++i) scaled[i] = base[i] * scale;
  }
  DLManagedTensor dset; long ds[2] = {n, dim};"""
assert text.count(before) == 1
text = text.replace(before, after).replace("dset.dl_tensor.data = (void*)base;",
                                          "dset.dl_tensor.data = (void*)(scaled ? scaled : base);")
before = "  h->n = n; h->dim = dim;"
after = """  if (scaled) {
    DLManagedTensor dataset = {0};
    if (cuvsCagraIndexGetDataset(h->idx, &dataset) != CUVS_SUCCESS) goto fail_idx;
    cudaStream_t stream = 0;
    int failed = cuvsStreamGet(h->res, &stream) != CUVS_SUCCESS;
    const size_t row_bytes = (size_t)dim * sizeof(float);
    const int64_t row_stride = dataset.dl_tensor.strides ? dataset.dl_tensor.strides[0] : dim;
    char *device = (char*)dataset.dl_tensor.data + dataset.dl_tensor.byte_offset;
    if (!failed && row_stride == dim) {
      failed = cudaMemcpyAsync(device, base, elements * sizeof(float),
                               cudaMemcpyHostToDevice, stream) != cudaSuccess;
    } else if (!failed) {
      for (long row = 0; row < n && !failed; ++row) {
        failed = cudaMemcpyAsync(device + row * row_stride * sizeof(float),
                                 base + row * dim, row_bytes,
                                 cudaMemcpyHostToDevice, stream) != cudaSuccess;
      }
    }
    if (cuvsStreamSync(h->res) != CUVS_SUCCESS) failed = 1;
    if (dataset.deleter) dataset.deleter(&dataset);
    if (failed) goto fail_idx;
  }
  free(scaled); scaled = NULL;
  h->n = n; h->dim = dim;"""
assert text.count(before) == 1
text = text.replace(before, after)
text = text.replace("fail_res: cuvsResourcesDestroy(h->res);",
                    "fail_res: free(scaled); cuvsResourcesDestroy(h->res);")
(target / "seekdb_cuvs_bridge.c").write_text(text)
library = target / "libseekdb_cuvs_bridge.so"
build = subprocess.run(["bash", str(target / "build.sh"), str(library)], capture_output=True, text=True)
(target / "compile.log").write_text(build.stdout + build.stderr)
print("compile", build.returncode, build.stdout, build.stderr, flush=True)
if build.returncode:
    raise SystemExit(build.returncode)
for dataset, dimension in [("sift250k", 128), ("synthetic250k-d256", 256)]:
    result = subprocess.run([
        sys.executable, str(root / "cagra_bridge.py"), str(library),
        "--data", str(root / dataset), "--dimension", str(dimension), "--transient-threads",
    ], capture_output=True, text=True, env=os.environ.copy())
    (target / (dataset + ".json")).write_text(result.stdout)
    (target / (dataset + ".stderr")).write_text(result.stderr)
    print(json.dumps({"dataset": dataset, "exit": result.returncode,
                      "output": result.stdout, "error": result.stderr[-2500:]}), flush=True)
