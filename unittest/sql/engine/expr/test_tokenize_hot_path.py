"""Native TOKENIZE regression for local maps and immutable stopword lookup.

Run with a seekdb instance built from this checkout:
    SEEKDB_PORT=2881 python unittest/sql/engine/expr/test_tokenize_hot_path.py
Requires pymysql. These checks issue SELECT statements only.
"""

from concurrent.futures import ThreadPoolExecutor
import json
import os
import unittest

import pymysql


def connect():
    return pymysql.connect(
        host=os.environ.get("SEEKDB_HOST", "127.0.0.1"),
        port=int(os.environ.get("SEEKDB_PORT", "2881")),
        user=os.environ.get("SEEKDB_USER", "root"),
        password=os.environ.get("SEEKDB_PASSWORD", ""),
        charset="utf8mb4",
        autocommit=True,
    )


def tokenize(connection, text, parser, expression="%s", options=None):
    with connection.cursor() as cursor:
        cursor.execute(
            f"SELECT tokenize({expression},%s,%s) AS tokens",
            (text, parser, options or '[{"output":"all"}]'),
        )
        return json.loads(cursor.fetchone()[0])


class TokenizeHotPath(unittest.TestCase):
    def test_a_concurrent_first_ik_use_preserves_independent_modes(self):
        text = "中华人民共和国人民大会堂。三百二十个人正在学习数据库。"

        def options(mode):
            return json.dumps([{"output": "all"}, {"additional_args": [{"ik_mode": mode}]}])

        def check(worker):
            mode = "smart" if worker % 2 == 0 else "max_word"
            with connect() as connection:
                return mode, tokenize(connection, text, "ik", options=options(mode))

        with ThreadPoolExecutor(max_workers=8) as pool:
            parallel = list(pool.map(check, range(8)))
        with connect() as connection:
            expected = {mode: tokenize(connection, text, "ik", options=options(mode))
                        for mode in ("smart", "max_word")}
        self.assertNotEqual(expected["smart"], expected["max_word"])
        for mode, result in parallel:
            self.assertEqual(result, expected[mode])

    def test_collation_equal_words_keep_the_last_spelling_and_full_frequency(self):
        with connect() as connection:
            for parser in ("beng", "space"):
                for text, spelling in (
                    ("THE Café cafe", "cafe"),
                    ("THE cafe CAFÉ", "café"),
                ):
                    with self.subTest(parser=parser, text=text):
                        self.assertEqual(
                            tokenize(connection, text, parser),
                            {"doc_len": 2, "tokens": [{spelling: 2}]},
                        )

    def test_stopword_lookup_across_supported_collations(self):
        with connect() as connection:
            for collation in ("utf8mb4_general_ci", "utf8mb4_bin"):
                with self.subTest(collation=collation):
                    value = tokenize(
                        connection,
                        "THE coffee coffee",
                        "space",
                        f"%s COLLATE {collation}",
                    )
                    self.assertEqual(value, {"doc_len": 2, "tokens": [{"coffee": 2}]})

    def test_concurrent_parsers_do_not_share_document_state(self):
        def check(worker):
            with connect() as connection:
                for index in range(20):
                    word = f"tenant{worker}value{index}"
                    self.assertEqual(
                        tokenize(connection, f"THE {word} {word}", "beng"),
                        {"doc_len": 2, "tokens": [{word: 2}]},
                    )

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(check, range(8)))


if __name__ == "__main__":
    unittest.main()
