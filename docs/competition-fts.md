# Full-text retrieval optimization

Based on contest commit `5bce9e06d1281347ad61344e4b01fab9158c58a2` and the existing AI-function answer.

## Changes

- Document-local word-frequency maps and IK arbitration maps do not lock their buckets. The initialized global stopword set is immutable. Equal-collation tokens retain the baseline last-spelling representative and full frequency.
- IK processors reuse the already decoded current character. Built-in main and quantifier dictionaries are published lazily as immutable readers. Each reader owns copied packed DAT bytes; KV-cache handles are released before publication, so cache eviction and global shutdown cannot invalidate the reader. Document cursors and segmentation modes remain private.
- A natural-language MATCH used only as a Boolean predicate can consume postings without calculating BM25. All non-WHERE expressions and additional conditions are checked for score consumers. Numeric thresholds, ranked/projection consumers, Boolean search mode, boosts and vector-filter plans retain their scoring path. The existing positive IDF floor makes real natural-language postings nonzero.
- Unscored execution supports DaaT and large-query TaaT, normal and domain-mapping table lookup, and scalar projection. The storage skip vector is allocated independently of scoring; no statistics iterator is accessed when unused.
- Index coverage excludes MATCH columns used solely to identify the index. Actual original-text projection, filtering, aggregation and ordering still require their data columns. Dynamic query text and boosts remain real dependencies. This coverage pass is limited to a single natural-language MATCH in SELECT statements without subqueries or windows.

No query results, documents, token outputs or benchmark answers are cached. Data generation, SQL, timing and scoring assets are unchanged.

## Native regressions

With an isolated native server and `pymysql` installed:

```sh
python unittest/sql/engine/expr/test_tokenize_hot_path.py
EXPECT_BOOLEAN_ELISION=1 EXPECT_FTS_COVERING=1 \
  python unittest/sql/engine/expr/test_fts_boolean_retrieval.py
```

The first test covers concurrent first use of smart/max_word IK, independent document state, collation-equal word frequencies and stopwords. The second covers empty/NULL/no-hit queries, 300 distinct query terms, Boolean mode, live insert/update/delete and rollback, scores/projections/ranking/thresholds, original-text consumers and a JOIN. Both run on a real SQL engine. The Boolean test creates and removes only its uniquely named temporary database.

An independent development harness compared 445 complete TOKENIZE outputs and four complete ordered MATCH-ID sets against the original AI-answer binary. All were identical; 445 additional parses on eight independent concurrent connections also matched. Native shutdown through SIGUSR1 completed with exit code 0. SIGTERM deliberately maps to SIGKILL in this engine and does not exercise teardown.

## Development measurements

The original official generator and SQL were executed via PyMySQL with the official iteration/sample counts. These are paired development measurements on one host, with visible run-to-run load variation, and do not establish the contest score. In particular the latest tokenizer timings regressed slightly; the result is reported unchanged.

| Metric | Original AI binary | This version |
|---|---:|---:|
| build_ik_all_sec | 3.8444 | 3.7470 |
| build_ik_content_sec | 2.9179 | 2.7191 |
| build_beng_en_sec | 1.6139 | 1.5143 |
| tokenize_ik_avg_ms | 0.0796 | 0.0833 |
| tokenize_beng_avg_ms | 0.0419 | 0.0442 |
| query_cn_avg_ms | 1.3400 | 0.9516 |
| query_beng_avg_ms | 1.8700 | 1.5393 |
| query_mixed_avg_ms | 1.5690 | 1.2085 |
| query_limit_avg_ms | 1.7655 | 1.7025 |

The official reproducible workload remains `tools/benchmark/fts_large_bench.sh` and its unchanged score script, with ROWS=20000, BATCH=500, ROUNDS=3000, QUERY_ROUNDS=200, SAMPLES=3, WARMUP=30 and SKIP_LOAD=0. All four expected hit counts remain 8001/11000/7332/20.

CPU sampling found repeated UTF-8/collation work during index sorting and per-row merge/projection overhead in queries. Parser and coverage changes address only part of those costs; further improvements remain possible.
