"""Turso engine — SQLite-wire-compatible with native vector columns.

Turso (https://github.com/tursodatabase/turso) is a Rust rewrite of SQLite.
This engine inherits everything from ``SqliteEngine`` and only overrides:

* connection init (``turso.connect`` instead of ``sqlite3.connect``)
* ``to_sql_type`` — numeric lists become native ``F32_BLOB(N)`` / ``F64_BLOB(N)``
* ``_sanitize_value`` — embedding values are wrapped in ``vector32()`` / ``vector64()``

Everything else (schema, dtypes, execute, insert transaction, assets, cache)
is unchanged.

Embedding constraint: all values logged to the same embedding column must
share the dimension of the first value logged (the column is created as
``F32_BLOB(N)``). Mismatched dimensions raise at insert time.

Turso is beta. Not for production. See README for the full case.
"""
from __future__ import annotations

import logging
import os
from typing import Any, List, Literal

import turso  # pyturso on PyPI

from xetrack.engine import SqliteEngine
from xetrack.config import SCHEMA_PARAMS, DEFAULTS

logger = logging.getLogger(__name__)

EmbeddingDtype = Literal["f32", "f64"]


def _is_embedding(value: Any) -> bool:
    """A non-empty numeric list with at least one float counts as an embedding.

    Pure-int lists stay TEXT (consistent with the SQLite engine); bools are
    excluded because ``bool`` subclasses ``int``.
    """
    return (
        isinstance(value, list)
        and len(value) > 0
        and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in value)
        and any(isinstance(x, float) for x in value)
    )


class TursoEngine(SqliteEngine):
    """Turso backend. Opt-in via ``Tracker(..., engine='turso')``."""

    def __init__(
        self,
        db: str = DEFAULTS.DB,
        compress: bool = False,
        table_name: str = SCHEMA_PARAMS.DEFAULT_TABLE,
        embedding_dtype: EmbeddingDtype = "f32",
    ):
        """
        Args:
            db: Path to the database file (or ":memory:").
            compress: Whether asset compression is enabled.
            table_name: The events table name.
            embedding_dtype: Vector storage precision, "f32" (default) or "f64".
        """
        self.embedding_dtype = embedding_dtype
        super().__init__(db=db, compress=compress, table_name=table_name)

    def _init_connection(self):  # type: ignore[override]
        if self.db != ":memory:":
            db_dir = os.path.dirname(self.db)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)

        # ponytail: turso's Python driver is sqlite3-DB-API compatible; no row_factory yet.
        return turso.connect(self.db)

    def _vector_sql_type(self, value: List[float]) -> str:
        """SQL column type for an embedding value, e.g. ``F32_BLOB(4)``."""
        dim = len(value)
        return f"F{'32' if self.embedding_dtype == 'f32' else '64'}_BLOB({dim})"

    def _vector_fn(self) -> str:
        """The Turso SQL function that parses an embedding literal."""
        return "vector32" if self.embedding_dtype == "f32" else "vector64"

    def to_sql_type(self, value: Any) -> str:  # type: ignore[override]
        if _is_embedding(value):
            return self._vector_sql_type(value)
        return SqliteEngine.to_sql_type(value)

    def _sanitize_value(self, key: str, value: Any) -> tuple[str, Any]:
        """Wrap embeddings in vector32()/vector64(); defer everything else."""
        if _is_embedding(value):
            return f"{self._vector_fn()}(?)", str([float(x) for x in value])
        return super()._sanitize_value(key, value)
