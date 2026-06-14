"""
SurakshaNet - Violation Logging Module
Persists every PPE breach (Head/Person without Helmet) into a local SQLite DB
with timestamp, class, confidence and bounding box coordinates.
"""

import sqlite3
import threading
from datetime import datetime
from contextlib import contextmanager


DB_PATH = "surakshanet_logs.db"


class ViolationLogger:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS violations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    class_name TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    x1 INTEGER, y1 INTEGER, x2 INTEGER, y2 INTEGER
                )
            """)
            conn.commit()

    def log_violation(self, class_name: str, confidence: float, bbox: list):
        """Insert a single violation record. Thread-safe."""
        x1, y1, x2, y2 = bbox
        timestamp = datetime.now().isoformat(timespec="seconds")

        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """INSERT INTO violations
                       (timestamp, class_name, confidence, x1, y1, x2, y2)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (timestamp, class_name, confidence, x1, y1, x2, y2)
                )
                conn.commit()

    def log_batch(self, results: list):
        """Log every result with status == 'BREACH' from classify_compliance()."""
        for r in results:
            if r.get("status") == "BREACH":
                self.log_violation(r["class"], r["confidence"], r["bbox"])

    def get_recent(self, limit: int = 50):
        """Return the most recent violation records (for dashboard display)."""
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT timestamp, class_name, confidence, x1, y1, x2, y2 "
                "FROM violations ORDER BY id DESC LIMIT ?",
                (limit,)
            )
            return cur.fetchall()

    def get_summary(self):
        """Return total count and breakdown by class (for dashboard stats)."""
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM violations").fetchone()[0]
            by_class = conn.execute(
                "SELECT class_name, COUNT(*) FROM violations GROUP BY class_name"
            ).fetchall()
            return {"total": total, "by_class": dict(by_class)}
