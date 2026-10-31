"""Check real postings, score consumers and transactional visibility together.

Run against a native seekdb with pymysql installed. The temporary database is
owned by this test and removed on completion. EXPECT_BOOLEAN_ELISION=1 also
checks the optimized plan; the behavioral assertions run on the baseline.
"""

import os
import unittest
import uuid

from test_tokenize_hot_path import connect


class BooleanRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.connection = connect()
        cls.database = "fts_score_" + uuid.uuid4().hex[:12]
        with cls.connection.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE {cls.database}")
            cursor.execute(f"USE {cls.database}")
            cursor.execute("CREATE TABLE docs(id INT PRIMARY KEY, body TEXT)")
            cursor.executemany("INSERT INTO docs VALUES (%s,%s)", [
                (1, "common database database audit"),
                (2, "common database performance"),
                (3, "common audit"),
                (4, "common unrelated"),
                (5, None), (6, ""), (7, "the and a"),
            ])
            cursor.execute("ALTER TABLE docs ADD FULLTEXT INDEX ft(body) WITH PARSER beng")

    @classmethod
    def tearDownClass(cls):
        with cls.connection.cursor() as cursor:
            cursor.execute(f"DROP DATABASE {cls.database}")
        cls.connection.close()

    def rows(self, sql, parameters=None):
        with self.connection.cursor() as cursor:
            cursor.execute(sql, parameters)
            return cursor.fetchall()

    def test_postings_match_numeric_predicate_and_empty_queries(self):
        for query in ("database", "database audit", "common", "absent", "", "the and", None):
            with self.subTest(query=query):
                boolean = self.rows("SELECT id FROM docs WHERE MATCH(body) AGAINST(%s) ORDER BY id", (query,))
                numeric = self.rows("SELECT id FROM docs WHERE MATCH(body) AGAINST(%s)>0.0 ORDER BY id", (query,))
                self.assertEqual(boolean, numeric)
        self.assertEqual(self.rows("SELECT id FROM docs WHERE MATCH(body) AGAINST('common') ORDER BY id"),
                         ((1,), (2,), (3,), (4,)))
        self.assertEqual(self.rows("SELECT id FROM docs WHERE MATCH(body) AGAINST('database') ORDER BY id LIMIT 1 OFFSET 1"),
                         ((2,),))

    def test_large_term_set_and_boolean_mode(self):
        query = "database " + " ".join(f"absentterm{i}" for i in range(300))
        self.assertEqual(self.rows("SELECT id FROM docs WHERE MATCH(body) AGAINST(%s) ORDER BY id", (query,)),
                         ((1,), (2,)))
        self.assertEqual(self.rows(
            "SELECT id FROM docs WHERE MATCH(body) AGAINST('+database -audit' IN BOOLEAN MODE) ORDER BY id"),
            ((2,),))

    def test_projected_ranked_threshold_and_nested_score_consumers(self):
        select = "SELECT id,MATCH(body) AGAINST('database') score FROM docs "
        boolean = self.rows(select + "WHERE MATCH(body) AGAINST('database') ORDER BY id")
        numeric = self.rows(select + "WHERE MATCH(body) AGAINST('database')>0.0 ORDER BY id")
        self.assertEqual(boolean, numeric)
        self.assertEqual([row[0] for row in boolean], [1, 2])
        self.assertTrue(all(row[1] > 0.0 for row in boolean))
        ranked = self.rows(select + "WHERE MATCH(body) AGAINST('database') ORDER BY score DESC,id LIMIT 2")
        self.assertEqual(ranked, tuple(sorted(boolean, key=lambda row: (-row[1], row[0]))))
        threshold = sum(row[1] for row in boolean) / 2
        self.assertEqual(self.rows(select + "WHERE MATCH(body) AGAINST('database')>%s ORDER BY id", (threshold,)),
                         tuple(row for row in boolean if row[1] > threshold))
        nested = self.rows("SELECT id,ROUND(MATCH(body) AGAINST('database')*100,8) FROM docs "
                           "WHERE MATCH(body) AGAINST('database') ORDER BY id")
        self.assertEqual(nested, tuple((row[0], round(row[1] * 100, 8)) for row in boolean))

    def test_original_text_projection_filter_and_sort(self):
        bodies = ((1, "common database database audit"), (2, "common database performance"))
        self.assertEqual(self.rows("SELECT id,body FROM docs WHERE MATCH(body) AGAINST('database') ORDER BY id"),
                         bodies)
        self.assertEqual(self.rows("SELECT id FROM docs WHERE MATCH(body) AGAINST('database') "
                                   "AND body LIKE '%audit%' ORDER BY id"), ((1,),))
        self.assertEqual(self.rows("SELECT SUM(LENGTH(body)) FROM docs WHERE MATCH(body) AGAINST('database')"),
                         ((sum(len(body) for _, body in bodies),),))
        self.assertEqual(self.rows("SELECT id FROM docs WHERE MATCH(body) AGAINST('database') "
                                   "ORDER BY LENGTH(body),id LIMIT 1"), ((2,),))
        self.assertEqual(self.rows("SELECT COUNT(*) FROM docs d JOIN docs x ON x.id=d.id "
                                   "WHERE MATCH(d.body) AGAINST('database') AND x.body LIKE '%audit%'"), ((1,),))

    def test_live_update_delete_insert_and_rollback(self):
        try:
            self.rows("UPDATE docs SET body='common database' WHERE id=4")
            self.rows("DELETE FROM docs WHERE id=2")
            self.rows("INSERT INTO docs VALUES(8,'common database')")
            self.assertEqual(self.rows("SELECT id FROM docs WHERE MATCH(body) AGAINST('database') ORDER BY id"),
                             ((1,), (4,), (8,)))
            self.rows("START TRANSACTION")
            self.rows("UPDATE docs SET body='unrelated' WHERE id=1")
            self.rows("ROLLBACK")
            self.assertEqual(self.rows("SELECT id FROM docs WHERE MATCH(body) AGAINST('database') ORDER BY id"),
                             ((1,), (4,), (8,)))
        finally:
            self.rows("ROLLBACK")
            self.rows("UPDATE docs SET body='common unrelated' WHERE id=4")
            self.rows("INSERT INTO docs VALUES(2,'common database performance')")
            self.rows("DELETE FROM docs WHERE id=8")

    def test_plans_keep_numeric_score_consumers(self):
        if os.environ.get("EXPECT_BOOLEAN_ELISION") != "1":
            return
        plain = " ".join(str(value) for row in self.rows(
            "EXPLAIN SELECT COUNT(*) FROM docs WHERE MATCH(body) AGAINST('database')") for value in row)
        ranked = " ".join(str(value) for row in self.rows(
            "EXPLAIN SELECT id,MATCH(body) AGAINST('database') score FROM docs "
            "WHERE MATCH(body) AGAINST('database') ORDER BY score DESC") for value in row)
        self.assertIn("calc_relevance=false", plain)
        self.assertIn("calc_relevance=true", ranked)
        if os.environ.get("EXPECT_FTS_COVERING") == "1":
            self.assertIn("index_back=false", plain)
            projected = " ".join(str(value) for row in self.rows(
                "EXPLAIN SELECT body FROM docs WHERE MATCH(body) AGAINST('database')") for value in row)
            self.assertIn("index_back=true", projected)


if __name__ == "__main__":
    unittest.main()
