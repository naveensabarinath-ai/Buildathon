import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import chat_usage


class ChatUsageTests(unittest.TestCase):
    def test_migrates_existing_usage_table_and_closes_connections(self):
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "chat_usage.db"
            db = sqlite3.connect(database_path)
            try:
                db.execute(
                    """CREATE TABLE chat_requests (
                    id INTEGER PRIMARY KEY, created_utc TEXT NOT NULL,
                    model TEXT NOT NULL, prompt_tokens INTEGER NOT NULL,
                    obsolete_tokens INTEGER NOT NULL,
                    completion_tokens INTEGER NOT NULL, total_tokens INTEGER NOT NULL,
                    estimated_cost_usd REAL NOT NULL, scheme_id TEXT,
                    source_count INTEGER NOT NULL)"""
                )
                db.execute(
                    """INSERT INTO chat_requests VALUES
                    (1, '2025-01-01', 'old-model', 10, 3, 4, 17, 0.1, '1', 2)"""
                )
                db.commit()
            finally:
                db.close()

            with patch.object(chat_usage, "DB", database_path):
                self.assertEqual(chat_usage.totals()["requests"], 1)
                self.assertEqual(chat_usage.totals()["prompt_tokens"], 10)
                recorded = chat_usage.record(
                    model="new-model",
                    prompt_tokens=2,
                    completion_tokens=3,
                    scheme_id=None,
                    source_count=1,
                )
                self.assertEqual(recorded["total_tokens"], 5)
                self.assertEqual(chat_usage.totals()["requests"], 2)


if __name__ == "__main__":
    unittest.main()
