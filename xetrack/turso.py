"""Turso engine — SQLite-wire-compatible with native vector columns.

Turso (https://github.com/tursodatabase/turso) is a Rust rewrite of SQLite.
This engine inherits everything from ``SqliteEngine`` and only overrides:

* connection init (``turso.connect`` instead of ``sqlite3.connect``)
* ``list[float]`` → native ``F32_BLOB(N)`` / ``F64_BLOB(N)`` vector columns
* insert path — wrap embedding values in ``vector32()`` / ``vector64()``

Everything else (schema, dtypes, execute, assets, cache) is unchanged.

Turso is beta. Not for production. See README for the full case.
"""
from __future__ import annotations

import logging
import os
from typing import Any, List, Optional, Literal

from xetrack.engine import SqliteEngine
from xetrack.config import SCHEMA_PARAMS, DEFAULTS

logger = logging.getLogger(__name__)

EmbeddingDtype = Literal["f32", "f64"]


def _is_embedding(value: Any) -> bool:
    """A list of floats (and non-empty) counts as an embedding."""
    return (
        isinstance(value, list)
        and len(value) > 0
        and all(isinstance(x, float) for x in value)
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
        self.embedding_dtype = embedding_dtype
        super().__init__(db=db, compress=compress, table_name=table_name)

    def _init_connection(self):  # type: ignore[override]
        import turso  # pytursodatabase on PyPI

        if self.db != ":memory:":
            db_dir = os.path.dirname(self.db)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)

        # ponytail: turso's Python driver is sqlite3-DB-API compatible; no row_factory yet.
        return turso.connect(self.db)

    def _vector_sql_type(self, value: List[float]) -> str:
        dim = len(value)
        return f"F{'32' if self.embedding_dtype == 'f32' else '64'}_BLOB({dim})"

    def _vector_fn(self) -> str:
        return "vector32" if self.embedding_dtype == "f32" else "vector64"

    def to_sql_type(self, value: Any) -> str:  # type: ignore[override]
        if _is_embedding(value):
            return self._vector_sql_type(value)
        return SqliteEngine.to_sql_type(value)

    def _insert_raw(self, data):  # type: ignore[override]
        """Same as SqliteEngine._insert_raw but wraps embeddings in vector32/64()."""
        try:
            self.conn.execute("BEGIN")
            vec_fn = self._vector_fn()
            for keys, values, size in data:
                placeholders: list[str] = []
                sanitized: list[Any] = []
                for value in values:
                    if _is_embedding(value):
                        placeholders.append(f"{vec_fn}(?)")
                        sanitized.append(str(value))  # '[0.1, 0.2, ...]'
                    elif isinstance(value, (dict, list, tuple)):
                        placeholders.append("?")
                        sanitized.append(str(value))
                    else:
                        placeholders.append("?")
                        sanitized.append(value)
                quoted_table_name = self._quote_table_name(self.table_name)
                query = (
                    f"INSERT INTO {quoted_table_name} ({', '.join(keys)}) "
                    f"VALUES ({', '.join(placeholders)})"
                )
                self.conn.execute(query, sanitized)
            self.conn.commit()
        except Exception as e:
            try:
                self.conn.rollback()
            except Exception:
                pass
            logger.error(f"Turso insert failed: {e}")
            raise
