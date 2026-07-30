"""D16: Real User Profile.

Serialize embeddings as BLOBs using numpy.tobytes().
"""

import json
import sqlite3

import numpy as np


def get_profile(conn: sqlite3.Connection, user_id: str = "default") -> dict:
    row = conn.execute("SELECT * FROM user_profile WHERE user_id = ?", (user_id,)).fetchone()
    if row is None:
        conn.execute("INSERT OR IGNORE INTO user_profile(user_id) VALUES (?)", (user_id,))
        conn.commit()
        row = conn.execute("SELECT * FROM user_profile WHERE user_id = ?", (user_id,)).fetchone()
    return dict(row) if row else {}


def update_goals(conn: sqlite3.Connection, user_id: str, goals: list[str]) -> None:
    goals_json = json.dumps(goals)
    conn.execute(
        """INSERT INTO user_profile(user_id, goals_json)
           VALUES (?, ?)
           ON CONFLICT(user_id) DO UPDATE SET goals_json=excluded.goals_json""",
        (user_id, goals_json),
    )


def update_values(conn: sqlite3.Connection, user_id: str, values: list[str]) -> None:
    values_json = json.dumps(values)
    conn.execute(
        """INSERT INTO user_profile(user_id, values_json)
           VALUES (?, ?)
           ON CONFLICT(user_id) DO UPDATE SET values_json=excluded.values_json""",
        (user_id, values_json),
    )


def update_goal_embeddings(conn: sqlite3.Connection, user_id: str, embeddings: np.ndarray) -> None:
    blob = embeddings.astype(np.float32).tobytes()
    conn.execute(
        """INSERT INTO user_profile(user_id, goal_embeddings)
           VALUES (?, ?)
           ON CONFLICT(user_id) DO UPDATE SET goal_embeddings=excluded.goal_embeddings""",
        (user_id, blob),
    )


def update_value_embeddings(conn: sqlite3.Connection, user_id: str, embeddings: np.ndarray) -> None:
    blob = embeddings.astype(np.float32).tobytes()
    conn.execute(
        """INSERT INTO user_profile(user_id, value_embeddings)
           VALUES (?, ?)
           ON CONFLICT(user_id) DO UPDATE SET value_embeddings=excluded.value_embeddings""",
        (user_id, blob),
    )


def get_goal_embeddings(conn: sqlite3.Connection, user_id: str = "default") -> np.ndarray | None:
    row = conn.execute(
        "SELECT goal_embeddings FROM user_profile WHERE user_id = ?", (user_id,)
    ).fetchone()
    if row and row[0]:
        return np.frombuffer(row[0], dtype=np.float32)
    return None
