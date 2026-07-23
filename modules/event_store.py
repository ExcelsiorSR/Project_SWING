# =============================================
#              MODULE IMPORTS
# =============================================

import sqlite3
import json
import time
from pathlib import Path
from contextlib import contextmanager

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "event_log.db"

# =============================================
#              FNCTIONAL MODULE
# =============================================

# Shared SQLite event timeline.

# This is a plain utility module - it defines no dependency between the Digital Twin and the AI Architecture. 
# Each service imports THIS file only (never each other) and writes its own domain events into one shared timeline, 
# so an operator (or a future dashboard) can see everything that happened - physics-side and AI-side - in one place, ordered by time.

def _get_db_path(db_path: str = None) -> Path:
    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

@contextmanager
def _connect(db_path: str = None):
    conn = sqlite3.connect(str(_get_db_path(db_path)), timeout=10)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def init_db(db_path: str = None):
    with _connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                source TEXT NOT NULL,       -- 'digital_twin' or 'ai_agent'
                grid_id TEXT NOT NULL,
                event_type TEXT NOT NULL,   -- e.g. 'LOAD_CHANGE', 'PROPOSAL', 'EXECUTION', 'FAULT_INJECTED'
                payload TEXT NOT NULL       -- JSON-encoded details
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_events_grid ON events(grid_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)")


def log_event(source: str, grid_id: str, event_type: str, payload: dict, db_path: str = None):
    '''
    Append one event to the timeline. Never raises on failure - an event
    store outage should never take down the Digital Twin or the AI service;
    it just prints a warning and moves on.
    '''
    try:
        init_db(db_path)
        with _connect(db_path) as conn:
            conn.execute(
                "INSERT INTO events (timestamp, source, grid_id, event_type, payload) VALUES (?, ?, ?, ?, ?)",
                (time.time(), source, grid_id, event_type, json.dumps(payload, default=str))
            )
    except Exception as e:
        print(f"[EVENT STORE] Failed to log event ({source}/{event_type}): {e}")


def get_events(grid_id: str = None, source: str = None, limit: int = 100, db_path: str = None) -> list:
    '''Returns the most recent events, newest first, optionally filtered by grid_id and/or source.'''
    init_db(db_path)
    query = "SELECT timestamp, source, grid_id, event_type, payload FROM events WHERE 1=1"
    params = []

    if grid_id:
        query += " AND grid_id = ?"
        params.append(grid_id)
    if source:
        query += " AND source = ?"
        params.append(source)

    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    with _connect(db_path) as conn:
        rows = conn.execute(query, params).fetchall()

    return [
        {
            "timestamp": row[0],
            "source": row[1],
            "grid_id": row[2],
            "event_type": row[3],
            "payload": json.loads(row[4]),
        }
        for row in rows
    ]