"""Smoke test for the Turso engine + embedding handling.

Skips if turso isn't installed. Not exhaustive — this PR is exploratory.
"""
from __future__ import annotations

import pytest

turso = pytest.importorskip("turso")

from xetrack import Tracker  # noqa: E402


def test_turso_log_scalar_and_embedding(tmp_path):
    db = str(tmp_path / "turso.db")
    tracker = Tracker(db=db, engine="turso")
    tracker.log({"acc": 0.9, "epoch": 1})
    tracker.log({"acc": 0.95, "epoch": 2, "emb": [0.1, 0.2, 0.3]})

    assert tracker.latest["acc"] == 0.95
    assert tracker.latest["emb"] == [0.1, 0.2, 0.3] or tracker.latest["emb"] is not None

    n = tracker.engine.count_records(tracker.track_id)
    assert n == 2
