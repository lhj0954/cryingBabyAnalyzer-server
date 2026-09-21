import sqlite3
from pathlib import Path
from typing import Optional, List, Dict, Any

DB_PATH = Path("cry_records.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_column(conn: sqlite3.Connection, column_name: str, column_sql: str) -> None:
    columns = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(cry_records)").fetchall()
    }
    if column_name not in columns:
        conn.execute(f"ALTER TABLE cry_records ADD COLUMN {column_name} {column_sql}")


def init_db() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cry_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                filename TEXT NOT NULL,
                saved_filename TEXT NOT NULL,
                audio_path TEXT NOT NULL,
                label TEXT,
                confidence REAL,
                duration_sec REAL,
                message TEXT,
                feedback_correct INTEGER,
                actual_reason TEXT,
                caregiver_action TEXT,
                feedback_created_at TEXT
            )
            """
        )

        # 기존 DB도 그대로 사용할 수 있도록 누락 컬럼만 추가
        _ensure_column(conn, "feedback_correct", "INTEGER")
        _ensure_column(conn, "actual_reason", "TEXT")
        _ensure_column(conn, "caregiver_action", "TEXT")
        _ensure_column(conn, "feedback_created_at", "TEXT")

        conn.commit()


def insert_record(
    *,
    created_at: str,
    filename: str,
    saved_filename: str,
    audio_path: str,
    label: Optional[str],
    confidence: Optional[float],
    duration_sec: Optional[float],
    message: Optional[str],
) -> int:
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO cry_records (
                created_at,
                filename,
                saved_filename,
                audio_path,
                label,
                confidence,
                duration_sec,
                message
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                filename,
                saved_filename,
                audio_path,
                label,
                confidence,
                duration_sec,
                message,
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)


def update_feedback(
    record_id: int,
    *,
    correct: bool,
    actual_reason: Optional[str],
    caregiver_action: Optional[str],
    feedback_created_at: str,
) -> bool:
    with get_connection() as conn:
        cursor = conn.execute(
            """
            UPDATE cry_records
            SET feedback_correct = ?,
                actual_reason = ?,
                caregiver_action = ?,
                feedback_created_at = ?
            WHERE id = ?
            """,
            (
                1 if correct else 0,
                actual_reason,
                caregiver_action,
                feedback_created_at,
                record_id,
            ),
        )
        conn.commit()
        return cursor.rowcount > 0


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    return dict(row)


def list_records(
    *,
    date: Optional[str] = None,
    label: Optional[str] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    query = "SELECT * FROM cry_records WHERE 1=1"
    params: list[Any] = []

    if date:
        query += " AND substr(created_at, 1, 10) = ?"
        params.append(date)

    if label:
        query += " AND label = ?"
        params.append(label)

    query += " ORDER BY created_at DESC, id DESC LIMIT ?"
    params.append(max(1, min(limit, 500)))

    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()
        return [_row_to_dict(row) for row in rows]


def get_record(record_id: int) -> Optional[Dict[str, Any]]:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM cry_records WHERE id = ?",
            (record_id,),
        ).fetchone()
        return _row_to_dict(row) if row else None


def get_stats(date: Optional[str] = None) -> Dict[str, Any]:
    where = "WHERE 1=1"
    params: list[Any] = []

    if date:
        where += " AND substr(created_at, 1, 10) = ?"
        params.append(date)

    with get_connection() as conn:
        total_row = conn.execute(
            f"SELECT COUNT(*) AS total FROM cry_records {where}",
            params,
        ).fetchone()

        label_rows = conn.execute(
            f"""
            SELECT label, COUNT(*) AS count
            FROM cry_records
            {where}
            GROUP BY label
            ORDER BY count DESC, label ASC
            """,
            params,
        ).fetchall()

        hour_rows = conn.execute(
            f"""
            SELECT substr(created_at, 12, 2) AS hour, COUNT(*) AS count
            FROM cry_records
            {where}
            GROUP BY hour
            ORDER BY count DESC, hour ASC
            """,
            params,
        ).fetchall()

    return {
        "total": int(total_row["total"] if total_row else 0),
        "by_label": [_row_to_dict(row) for row in label_rows],
        "by_hour": [_row_to_dict(row) for row in hour_rows],
    }
