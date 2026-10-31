"""Measure CAGRA through seekdb's real CREATE INDEX and scalar SQL paths.

The caller selects the bridge with LD_LIBRARY_PATH when starting the server.
Data loading and exact CPU ground truth are outside the measured interval.
Only the dedicated amd_sql_validation database is modified.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pymysql


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("mode", choices=("prepare", "measure"))
parser.add_argument("--root", type=Path, default=Path("/workspace/codex-v5-bridge"))
parser.add_argument("--label", default="baseline")
parser.add_argument("--port", type=int, default=2883)
args = parser.parse_args()
connection = pymysql.connect(host="127.0.0.1", port=args.port, user="root", autocommit=True,
                             read_timeout=1200, write_timeout=1200)
reports = []
with connection.cursor() as cursor:
    cursor.execute("CREATE DATABASE IF NOT EXISTS amd_sql_validation")
    cursor.execute("USE amd_sql_validation")
    for dataset, dimension, table in (("sift250k", 128, "sift"),
                                       ("synthetic250k-d256", 256, "unit_vectors")):
        base = np.fromfile(args.root / dataset / "base.f32", dtype=np.float32).reshape(-1, dimension)
        queries = np.fromfile(args.root / dataset / "query.f32", dtype=np.float32).reshape(-1, dimension)
        if args.mode == "prepare":
            cursor.execute(f"CREATE TABLE IF NOT EXISTS {table}(id BIGINT PRIMARY KEY,v VECTOR({dimension}))")
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            existing = cursor.fetchone()[0]
            if existing not in (0, len(base)):
                raise RuntimeError(f"incomplete owned table {table}: {existing}")
            if not existing:
                start = time.perf_counter()
                for first in range(0, len(base), 1000):
                    rows = [(first + i, json.dumps(v.tolist(), separators=(",", ":")))
                            for i, v in enumerate(base[first:first + 1000])]
                    cursor.executemany(f"INSERT INTO {table}(id,v) VALUES(%s,%s)", rows)
                    if first % 25000 == 0:
                        print("import", dataset, first + len(rows), time.perf_counter() - start, flush=True)
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            assert cursor.fetchone()[0] == len(base)
            print("prepared", dataset, len(base), flush=True)
            continue
        # Independent FP64 exact neighbours. No judge answers or hidden labels.
        base64 = base.astype(np.float64)
        query64 = queries.astype(np.float64)
        distances = (np.sum(base64 * base64, axis=1)[None, :]
                     + np.sum(query64 * query64, axis=1)[:, None] - 2 * (query64 @ base64.T))
        exact = []
        for distance in distances:
            ids = np.argpartition(distance, 99)[:100]
            exact.append(ids[np.lexsort((ids, distance[ids]))])
        del base64, distances
        literals = [json.dumps(q.tolist(), separators=(",", ":")) for q in queries]

        def query(index, k):
            cursor.execute(f"SELECT id FROM {table} ORDER BY l2_distance(v,'{literals[index]}') APPROXIMATE LIMIT {k}")
            return [row[0] for row in cursor.fetchall()]

        start = time.perf_counter()
        cursor.execute(f"CREATE VECTOR INDEX idx_cagra ON {table}(v) "
                       "WITH(distance=l2,type=hnsw,lib=cuvs,m=16,ef_construction=200,ef_search=64)")
        query(0, 10)
        report = {"label": args.label, "dataset": dataset, "rows": len(base), "dimension": dimension,
                  "query_count": len(queries), "ddl_and_first_query_seconds": time.perf_counter() - start}
        for k in (10, 100):
            latency, recall = [], []
            for i in range(len(queries)):
                start = time.perf_counter()
                ids = query(i, k)
                latency.append((time.perf_counter() - start) * 1000)
                assert len(ids) == k and len(set(ids)) == k
                recall.append(len(set(ids) & set(map(int, exact[i][:k]))) / k)
            report[f"recall_at_{k}"] = float(np.mean(recall))
            report[f"p50_sql_query_ms_at_{k}"] = float(np.median(latency))
        reports.append(report)
        print(json.dumps(report, sort_keys=True), flush=True)
        # Remove the completed index before switching bridge libraries. Keep
        # imported rows, so the next measurement uses the identical SQL data.
        cursor.execute(f"DROP INDEX idx_cagra ON {table}")
connection.close()
if reports:
    (args.root / f"sql-{args.label}.json").write_text(json.dumps(reports, indent=2) + "\n")
