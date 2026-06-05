"""Database utilities for feedback and active learning."""

import sqlite3
from pathlib import Path
from typing import Any
import json
from datetime import datetime, timezone

DB_PATH = Path("data/feedback.sqlite")

def init_db() -> None:
    """Initialize the SQLite database schema."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        
        # Feedback table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                request_id TEXT NOT NULL,
                was_correct BOOLEAN NOT NULL,
                true_label TEXT,
                notes TEXT,
                processed BOOLEAN DEFAULT 0
            )
        ''')
        
        # Candidate patterns table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS candidate_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                text TEXT NOT NULL,
                embedding_category TEXT,
                similarity FLOAT,
                status TEXT DEFAULT 'pending'
            )
        ''')
        conn.commit()

def save_feedback(request_id: str, was_correct: bool, true_label: str | None, notes: str | None) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            '''INSERT INTO feedback (timestamp, request_id, was_correct, true_label, notes)
               VALUES (?, ?, ?, ?, ?)''',
            (datetime.now(timezone.utc).isoformat(), request_id, was_correct, true_label, notes)
        )
        conn.commit()

def save_candidate_pattern(text: str, category: str, similarity: float) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            '''INSERT INTO candidate_patterns (timestamp, text, embedding_category, similarity)
               VALUES (?, ?, ?, ?)''',
            (datetime.now(timezone.utc).isoformat(), text, category, similarity)
        )
        conn.commit()
