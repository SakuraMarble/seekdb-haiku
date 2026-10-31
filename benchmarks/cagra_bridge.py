"""Measure real bridge construction, scalar queries, recall, and snapshot recovery.

This isolates the bridge; SQL DDL timing still requires a complete seekdb build.
"""

import argparse
import ctypes as c
import hashlib
import json
from pathlib import Path
import resource
import time
import threading

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", type=Path)
    parser.add_argument("--data", type=Path, default=Path("/seekdb_workspace/datasets"))
    parser.add_argument("--dimension", type=int, default=128)
    parser.add_argument("--rows", type=int)
    parser.add_argument("--transient-threads", action="store_true", help="Match the adaptor's new thread per query")
    args = parser.parse_args()
    threading.stack_size(32 * 1024 * 1024)
    soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
    resource.setrlimit(resource.RLIMIT_STACK, (hard, hard))
    base = np.fromfile(args.data / "base.f32", dtype=np.float32).reshape(-1, args.dimension)
    if args.rows:
        base = np.ascontiguousarray(base[:args.rows])
    queries = np.fromfile(args.data / "query.f32", dtype=np.float32).reshape(-1, args.dimension)
    k_max = min(100, len(base))
    exact = []
    base64 = base.astype(np.float64)
    squared = np.sum(base64 * base64, axis=1)
    query64 = queries.astype(np.float64)
    all_distances = squared[None, :] + np.sum(query64 * query64, axis=1)[:, None] - 2 * (query64 @ base64.T)
    for distances in all_distances:
        ids = np.argpartition(distances, k_max - 1)[:k_max]
        ids = ids[np.lexsort((ids, distances[ids]))]
        exact.append(ids)
    del base64
    del all_distances
    library = c.CDLL(str(args.library.resolve()))
    float_p, uint_p = c.POINTER(c.c_float), c.POINTER(c.c_uint)
    library.seekdb_cuvs_build.argtypes = [float_p, c.c_long, c.c_long]
    library.seekdb_cuvs_build.restype = c.c_void_p
    library.seekdb_cuvs_search.argtypes = [c.c_void_p, float_p, c.c_long, c.c_long, uint_p, float_p]
    library.seekdb_cuvs_search.restype = c.c_int
    library.seekdb_cuvs_free.argtypes = [c.c_void_p]
    library.seekdb_cuvs_serialize.argtypes = [c.c_void_p, c.c_char_p]
    library.seekdb_cuvs_deserialize.argtypes = [c.c_char_p, c.c_long, c.c_long]
    library.seekdb_cuvs_deserialize.restype = c.c_void_p

    def execute_search(handle, query, k, check_distance=False):
        ids = np.empty(k, dtype=np.uint32)
        distances = np.empty(k, dtype=np.float32)
        rc = library.seekdb_cuvs_search(
            handle, query.ctypes.data_as(float_p), 1, k,
            ids.ctypes.data_as(uint_p), distances.ctypes.data_as(float_p),
        )
        if rc:
            raise RuntimeError(f"GPU search failed: {rc}")
        if len(set(map(int, ids))) != k or np.any(ids >= len(base)) or not np.isfinite(distances).all():
            raise RuntimeError("invalid GPU result identities or distances")
        if check_distance:
            expected = np.sum((base[ids].astype(np.float64) - query.astype(np.float64)) ** 2, axis=1)
            if not np.allclose(distances, expected, rtol=1e-4, atol=2e-5):
                raise RuntimeError("GPU distances differ from original FP32 squared L2")
        return ids

    def search(handle, query, k):
        if not args.transient_threads:
            return execute_search(handle, query, k)
        result, errors = [], []

        def invoke():
            try:
                result.append(execute_search(handle, query, k))
            except Exception as error:
                errors.append(error)

        worker = threading.Thread(target=invoke)
        worker.start()
        worker.join()
        if errors:
            raise errors[0]
        return result[0]

    start = time.perf_counter()
    handle = library.seekdb_cuvs_build(base.ctypes.data_as(float_p), len(base), args.dimension)
    if not handle:
        raise RuntimeError("GPU construction failed")
    report = {
        "library": str(args.library), "rows": len(base), "dimension": args.dimension,
        "transient_threads": args.transient_threads,
    }
    snapshot = args.library.with_suffix(".snapshot")
    recovered_snapshot = args.library.with_suffix(".recovered.snapshot")
    try:
        search(handle, queries[0], min(10, len(base)))
        report["build_and_first_query_seconds"] = time.perf_counter() - start
        for k in sorted({min(10, len(base)), k_max}):
            latency, recall = [], []
            for i, query in enumerate(queries):
                begin = time.perf_counter()
                ids = search(handle, query, k)
                latency.append((time.perf_counter() - begin) * 1000)
                recall.append(len(set(map(int, ids)) & set(map(int, exact[i][:k]))) / k)
            report[f"p50_query_ms_at_{k}"] = float(np.median(latency))
            report[f"recall_at_{k}"] = float(np.mean(recall))
        before = search(handle, queries[0], k_max)
        repeated = search(handle, queries[0], k_max)
        report["repeat_id_overlap"] = len(set(map(int, before)) & set(map(int, repeated))) / k_max
        for query in queries[:5]:
            execute_search(handle, query, k_max, check_distance=True)
        report["original_l2_distances_verified"] = True
        if library.seekdb_cuvs_serialize(handle, str(snapshot).encode()):
            raise RuntimeError("snapshot write failed")
    finally:
        library.seekdb_cuvs_free(handle)
    handle = library.seekdb_cuvs_deserialize(str(snapshot).encode(), len(base), args.dimension)
    if not handle:
        raise RuntimeError("snapshot load failed")
    try:
        after = search(handle, queries[0], k_max)
        report["snapshot_same_ids"] = bool(np.array_equal(before, after))
        report["snapshot_id_overlap"] = len(set(map(int, before)) & set(map(int, after))) / k_max
        if library.seekdb_cuvs_serialize(handle, str(recovered_snapshot).encode()):
            raise RuntimeError("recovered snapshot write failed")
        def digest(path):
            with path.open("rb") as stream:
                return hashlib.file_digest(stream, "sha256").hexdigest()
        report["snapshot_bytes_identical"] = digest(snapshot) == digest(recovered_snapshot)
        for query in queries[:5]:
            execute_search(handle, query, k_max, check_distance=True)
    finally:
        library.seekdb_cuvs_free(handle)
        recovered_snapshot.unlink(missing_ok=True)
    print(json.dumps(report, sort_keys=True), flush=True)
    # CAGRA traversal can vary between calls. Verify the graph and dataset are
    # preserved byte-for-byte rather than requiring a deterministic traversal.
    if not report["snapshot_bytes_identical"]:
        raise RuntimeError("snapshot changed graph or dataset bytes")


if __name__ == "__main__":
    main()
