import json
import os
import sqlite3
import time
from typing import Any, Optional


def _default_db_path() -> str:
    # api/app/store.py -> api/
    api_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(api_root, "data", "jobs.db")


DATABASE_URL = os.environ.get("DATABASE_URL", _default_db_path())


def _connect() -> sqlite3.Connection:
    # Needed for async web servers (multiple requests).
    return sqlite3.connect(DATABASE_URL, check_same_thread=False)


def init_db() -> None:
    db_dir = os.path.dirname(os.path.abspath(DATABASE_URL))
    os.makedirs(db_dir, exist_ok=True)

    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
              id TEXT PRIMARY KEY,
              status TEXT NOT NULL,
              source TEXT NOT NULL,
              url TEXT NOT NULL,
              created_at REAL NOT NULL,
              updated_at REAL NOT NULL,
              result_json TEXT,
              error TEXT,
              progress TEXT
            );
            """
        )
        # Add progress column if it doesn't exist (for migration)
        try:
            conn.execute("ALTER TABLE jobs ADD COLUMN progress TEXT;")
        except sqlite3.OperationalError:
            pass  # Column already exists
        conn.commit()


def create_job(job_id: str, *, url: str, source: str) -> None:
    now = time.time()
    with _connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO jobs
              (id, status, source, url, created_at, updated_at, result_json, error, progress)
            VALUES
              (?, ?, ?, ?, ?, ?, NULL, NULL, NULL)
            """,
            (job_id, "queued", source, url, now, now),
        )
        conn.commit()


def set_job_status(job_id: str, status: str) -> None:
    now = time.time()
    with _connect() as conn:
        conn.execute(
            "UPDATE jobs SET status = ?, updated_at = ? WHERE id = ?",
            (status, now, job_id),
        )
        conn.commit()


def set_job_succeeded(job_id: str, result_json: str) -> None:
    now = time.time()
    with _connect() as conn:
        conn.execute(
            """
            UPDATE jobs
            SET status = 'succeeded', updated_at = ?, result_json = ?, error = NULL
            WHERE id = ?
            """,
            (now, result_json, job_id),
        )
        conn.commit()


def set_job_failed(job_id: str, error: str) -> None:
    now = time.time()
    with _connect() as conn:
        conn.execute(
            """
            UPDATE jobs
            SET status = 'failed', updated_at = ?, result_json = NULL, error = ?
            WHERE id = ?
            """,
            (now, error, job_id),
        )
        conn.commit()


def set_job_result(job_id: str, result_json: str) -> None:
    now = time.time()
    with _connect() as conn:
        conn.execute(
            "UPDATE jobs SET result_json = ?, updated_at = ? WHERE id = ?",
            (result_json, now, job_id),
        )
        conn.commit()


def set_job_progress(job_id: str, progress: str) -> None:
    now = time.time()
    with _connect() as conn:
        conn.execute(
            "UPDATE jobs SET progress = ?, updated_at = ? WHERE id = ?",
            (progress, now, job_id),
        )
        conn.commit()

def get_job(job_id: str) -> Optional[dict[str, Any]]:
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            "SELECT id, status, url, result_json, error, progress FROM jobs WHERE id = ?",
            (job_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return dict(row)

