"""
database.py
-----------
SQLite database handler for the Daily Brain Journal system.
Uses only the sqlite3 standard library module.
Designed to easily add user_id column later.
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "brain_journal.db")


def get_connection():
    """Return a new SQLite connection with row_factory for dict-like access."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """
    Create all tables if they don't exist.
    To add user_id later: just uncomment the user_id line in each table.
    """
    conn = get_connection()
    try:
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS focus_sessions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                -- user_id  INTEGER DEFAULT 1,
                timestamp   TEXT    NOT NULL,
                duration_s  REAL    NOT NULL DEFAULT 0,
                focus_score REAL    NOT NULL DEFAULT 0,
                focus_label TEXT    NOT NULL DEFAULT '',
                notes       TEXT    DEFAULT '',
                created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS emotion_sessions (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                -- user_id       INTEGER DEFAULT 1,
                timestamp        TEXT    NOT NULL,
                duration_s       REAL    NOT NULL DEFAULT 0,
                dominant_emotion TEXT    NOT NULL DEFAULT '',
                emotion_scores   TEXT    NOT NULL DEFAULT '{}',
                notes            TEXT    DEFAULT '',
                created_at       TEXT    NOT NULL DEFAULT (datetime('now'))
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS seizure_sessions (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                -- user_id   INTEGER DEFAULT 1,
                timestamp    TEXT    NOT NULL,
                duration_s   REAL    NOT NULL DEFAULT 0,
                detected     INTEGER NOT NULL DEFAULT 0,
                confidence   REAL    NOT NULL DEFAULT 0,
                seizure_type TEXT    DEFAULT '',
                notes        TEXT    DEFAULT '',
                created_at   TEXT    NOT NULL DEFAULT (datetime('now'))
            )
        """)

        conn.commit()
        print("[DB] Tables initialized successfully.")
    except Exception as e:
        conn.rollback()
        print(f"[DB ERROR] init_db: {e}")
        raise
    finally:
        conn.close()


# ─────────────────────────────────────────────
#  INSERT FUNCTIONS (auto-called by modules)
# ─────────────────────────────────────────────

def insert_focus_session(duration_s: float, focus_score: float,
                          focus_label: str, notes: str = "") -> int:
    conn = get_connection()
    try:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cur = conn.execute("""
            INSERT INTO focus_sessions (timestamp, duration_s, focus_score, focus_label, notes)
            VALUES (?, ?, ?, ?, ?)
        """, (ts, duration_s, focus_score, focus_label, notes))
        conn.commit()
        return cur.lastrowid
    except Exception as e:
        conn.rollback()
        raise
    finally:
        conn.close()


def insert_emotion_session(duration_s: float, dominant_emotion: str,
                            emotion_scores: str, notes: str = "") -> int:
    conn = get_connection()
    try:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cur = conn.execute("""
            INSERT INTO emotion_sessions (timestamp, duration_s, dominant_emotion, emotion_scores, notes)
            VALUES (?, ?, ?, ?, ?)
        """, (ts, duration_s, dominant_emotion, emotion_scores, notes))
        conn.commit()
        print("Inserting emotion row:", dominant_emotion)
        return cur.lastrowid
    except Exception as e:
        conn.rollback()
        raise
    finally:
        conn.close()

    


def insert_seizure_session(duration_s: float, detected: bool,
                            confidence: float, seizure_type: str = "",
                            notes: str = "") -> int:
    conn = get_connection()
    try:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cur = conn.execute("""
            INSERT INTO seizure_sessions (timestamp, duration_s, detected, confidence, seizure_type, notes)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (ts, duration_s, 1 if detected else 0, confidence, seizure_type, notes))
        conn.commit()
        return cur.lastrowid
    except Exception as e:
        conn.rollback()
        raise
    finally:
        conn.close()


# ─────────────────────────────────────────────
#  QUERY / SUMMARY FUNCTIONS
# ─────────────────────────────────────────────

def _date_filter_clause(period: str, date_from: str = None, date_to: str = None):
    if period == "daily":
        return "WHERE date(timestamp) = date('now')", ()
    elif period == "weekly":
        return "WHERE timestamp >= datetime('now', '-7 days')", ()
    elif period == "monthly":
        return "WHERE timestamp >= datetime('now', '-30 days')", ()
    elif period == "custom" and date_from and date_to:
        return "WHERE date(timestamp) BETWEEN ? AND ?", (date_from, date_to)
    else:
        return "", ()


def get_focus_summary(period="daily", date_from=None, date_to=None):
    where, params = _date_filter_clause(period, date_from, date_to)
    conn = get_connection()
    try:
        stats = conn.execute(f"""
            SELECT
                COUNT(*)                         AS total_sessions,
                ROUND(AVG(focus_score)*100, 1)   AS avg_score_pct,
                ROUND(MAX(focus_score)*100, 1)   AS peak_score_pct,
                ROUND(SUM(duration_s)/60, 1)     AS total_minutes
            FROM focus_sessions {where}
        """, params).fetchone()

        timeline = conn.execute(f"""
            SELECT timestamp,
                   ROUND(focus_score*100,1) AS score_pct,
                   focus_label, duration_s
            FROM focus_sessions {where}
            ORDER BY timestamp ASC
        """, params).fetchall()

        labels = conn.execute(f"""
            SELECT focus_label, COUNT(*) AS count
            FROM focus_sessions {where}
            GROUP BY focus_label
        """, params).fetchall()

        return {
            "stats": dict(stats) if stats else {},
            "timeline": [dict(r) for r in timeline],
            "label_distribution": [dict(r) for r in labels],
            "has_data": (stats["total_sessions"] or 0) > 0
        }
    finally:
        conn.close()


def get_emotion_summary(period="daily", date_from=None, date_to=None):
    where, params = _date_filter_clause(period, date_from, date_to)
    conn = get_connection()
    try:
        stats = conn.execute(f"""
            SELECT COUNT(*) AS total_sessions,
                   ROUND(SUM(duration_s)/60,1) AS total_minutes
            FROM emotion_sessions {where}
        """, params).fetchone()

        dist = conn.execute(f"""
            SELECT dominant_emotion, COUNT(*) AS count
            FROM emotion_sessions {where}
            GROUP BY dominant_emotion ORDER BY count DESC
        """, params).fetchall()

        timeline = conn.execute(f"""
            SELECT timestamp, dominant_emotion, emotion_scores, duration_s
            FROM emotion_sessions {where}
            ORDER BY timestamp ASC
        """, params).fetchall()

        return {
            "stats": dict(stats) if stats else {},
            "distribution": [dict(r) for r in dist],
            "timeline": [dict(r) for r in timeline],
            "has_data": (stats["total_sessions"] or 0) > 0
        }
    finally:
        conn.close()


def get_seizure_summary(period="daily", date_from=None, date_to=None):
    where, params = _date_filter_clause(period, date_from, date_to)
    conn = get_connection()
    try:
        stats = conn.execute(f"""
            SELECT COUNT(*) AS total_scans,
                   SUM(detected) AS total_detected,
                   COUNT(*) - SUM(detected) AS total_clear,
                   ROUND(SUM(duration_s)/60,1) AS total_minutes,
                   ROUND(AVG(CASE WHEN detected=1 THEN confidence END)*100,1) AS avg_conf_pct
            FROM seizure_sessions {where}
        """, params).fetchone()

        timeline = conn.execute(f"""
            SELECT timestamp, detected, confidence, seizure_type, duration_s
            FROM seizure_sessions {where}
            ORDER BY timestamp ASC
        """, params).fetchall()

        return {
            "stats": dict(stats) if stats else {},
            "timeline": [dict(r) for r in timeline],
            "has_data": (stats["total_scans"] or 0) > 0
        }
    finally:
        conn.close()


def get_combined_summary(period="daily", date_from=None, date_to=None):
    return {
        "focus":   get_focus_summary(period, date_from, date_to),
        "emotion": get_emotion_summary(period, date_from, date_to),
        "seizure": get_seizure_summary(period, date_from, date_to),
        "period":  period
    }


# ─────────────────────────────────────────────
#  DB VIEWER QUERIES (for the DB Viewer UI)
# ─────────────────────────────────────────────

def get_table_rows(table: str, limit: int = 100, offset: int = 0):
    """Return raw rows from any table for the DB viewer."""
    allowed = {"focus_sessions", "emotion_sessions", "seizure_sessions"}
    if table not in allowed:
        raise ValueError(f"Unknown table: {table}")
    conn = get_connection()
    try:
        rows = conn.execute(
            f"SELECT * FROM {table} ORDER BY id DESC LIMIT ? OFFSET ?",
            (limit, offset)
        ).fetchall()
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        return {"rows": [dict(r) for r in rows], "total": count}
    finally:
        conn.close()


def delete_row(table: str, row_id: int):
    """Delete a specific row by id (for DB viewer)."""
    allowed = {"focus_sessions", "emotion_sessions", "seizure_sessions"}
    if table not in allowed:
        raise ValueError(f"Unknown table: {table}")
    conn = get_connection()
    try:
        conn.execute(f"DELETE FROM {table} WHERE id = ?", (row_id,))
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    init_db()
    print(f"[DB] Database ready at: {DB_PATH}")