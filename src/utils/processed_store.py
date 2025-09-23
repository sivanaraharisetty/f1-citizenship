import os
import sqlite3
from typing import Iterable


class ProcessedKeyStore:
    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS keys (key TEXT PRIMARY KEY)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_keys_key ON keys(key)"
            )

    def exists(self, key: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT 1 FROM keys WHERE key = ? LIMIT 1", (key,))
            return cur.fetchone() is not None

    def add_many(self, keys: Iterable[str]) -> int:
        keys = list(set(keys))
        if not keys:
            return 0
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany("INSERT OR IGNORE INTO keys(key) VALUES (?)", [(k,) for k in keys])
            return conn.total_changes


