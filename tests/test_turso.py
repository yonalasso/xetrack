"""Smoke test for the Turso engine + embedding handling.

Skips if turso isn't installed. Not exhaustive — this PR is exploratory.
"""
from __future__ import annotations

import pytest

turso = pytest.importorskip("turso")

import orjson  # noqa: E402

from xetrack import Tracker  # noqa: E402


def test_turso_log_scalar_and_embedding(tmp_path):
    """Scalars behave like SQLite; list[float] round-trips as a native vector."""
    db = str(tmp_path / "turso.db")
    tracker = Tracker(db=db, engine="turso")
    tracker.log({"acc": 0.9, "epoch": 1})
    tracker.log({"acc": 0.95, "epoch": 2, "emb": [0.1, 0.2, 0.3]})

    assert tracker.latest["acc"] == 0.95
    assert tracker.latest["emb"] is not None

    row = tracker.conn.execute(
        'SELECT vector_extract(emb) FROM "default" WHERE emb IS NOT NULL'
    ).fetchone()
    assert orjson.loads(row[0]) == pytest.approx([0.1, 0.2, 0.3], rel=1e-6)

    assert tracker.engine.count_records(tracker.track_id) == 2


def test_turso_mixed_and_int_lists(tmp_path):
    """Mixed int/float lists are embeddings; pure-int lists stay TEXT."""
    db = str(tmp_path / "turso.db")
    tracker = Tracker(db=db, engine="turso")
    tracker.log({"emb": [0.5, 1, 2.0], "ids": [1, 2, 3]})

    row = tracker.conn.execute(
        'SELECT vector_extract(emb), ids FROM "default" WHERE emb IS NOT NULL'
    ).fetchone()
    assert orjson.loads(row[0]) == pytest.approx([0.5, 1.0, 2.0], rel=1e-6)
    assert row[1] == "[1, 2, 3]"
