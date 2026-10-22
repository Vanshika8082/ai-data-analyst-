
"""Safe CSV loading and read-only SQLite query execution."""

from pathlib import Path
import sqlite3
import time

import pandas as pd
import sqlglot
from sqlglot import exp


TABLE_NAME = "dataset"
MAX_ROWS = 500
QUERY_TIMEOUT_SECONDS = 3


def load_csv(file_path: str | Path) -> pd.DataFrame:
    """Load a CSV and validate its basic structure."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("The dataset contains no data rows.")

    df.columns = [str(column).strip() for column in df.columns]

    if any(not column for column in df.columns):
        raise ValueError("Column names cannot be empty.")

    if df.columns.duplicated().any():
        raise ValueError("Column names must be unique.")

    return df


def create_connection(df: pd.DataFrame) -> sqlite3.Connection:
    """Create a temporary SQLite database with restricted access."""
    conn = sqlite3.connect(":memory:")

    df.to_sql(TABLE_NAME, conn, index=False)

    # SQLite-level protection against data modifications.
    conn.execute("PRAGMA query_only = ON")

    # Limit individual SQLite operations where supported.
    if hasattr(conn, "setlimit"):
        conn.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, 10_000)
        conn.setlimit(sqlite3.SQLITE_LIMIT_COLUMN, 100)
        conn.setlimit(sqlite3.SQLITE_LIMIT_EXPR_DEPTH, 100)

    forbidden_functions = {
        "load_extension",
        "readfile",
        "writefile",
        "eval",
        "shell",
    }

    def authorize(action, arg1, arg2, database, trigger):
        # Queries can read only the uploaded dataset.
        if action == sqlite3.SQLITE_READ:
            return (
                sqlite3.SQLITE_OK
                if arg1 == TABLE_NAME and database == "main"
                else sqlite3.SQLITE_DENY
            )

        # SELECT operations are allowed.
        if action == sqlite3.SQLITE_SELECT:
            return sqlite3.SQLITE_OK

        # Allow normal SQL functions, but explicitly reject risky ones.
        if action == sqlite3.SQLITE_FUNCTION:
            function_name = (arg2 or arg1 or "").lower()
            return (
                sqlite3.SQLITE_DENY
                if function_name in forbidden_functions
                else sqlite3.SQLITE_OK
            )

        # Deny writes, schema changes, PRAGMAs, ATTACH,
        # DETACH, and other unapproved SQLite operations.
        return sqlite3.SQLITE_DENY

    conn.set_authorizer(authorize)
    return conn


def validate_sql(query: str) -> str:
    """Allow one SELECT query that references only dataset."""
    if not query or not query.strip():
        raise ValueError("SQL query cannot be empty.")

    if len(query) > 10_000:
        raise ValueError("SQL query is too long.")

    try:
        statements = sqlglot.parse(query, read="sqlite")
    except sqlglot.errors.ParseError as exc:
        raise ValueError("Invalid SQL syntax.") from exc

    statements = [statement for statement in statements
                  if statement is not None]

    if len(statements) != 1:
        raise ValueError("Only one SQL statement is allowed.")

    tree = statements[0]

    if not isinstance(tree, exp.Select):
        raise ValueError("Only SELECT queries are allowed.")

    # Keep the initial query policy deliberately simple.
    if tree.find(exp.With) is not None:
        raise ValueError("CTEs are not supported.")

    if tree.find(exp.Subquery) is not None:
        raise ValueError("Nested subqueries are not supported yet.")

    if tree.args.get("into") is not None:
        raise ValueError("SELECT INTO is not allowed.")

    tables = list(tree.find_all(exp.Table))

    if not tables:
        raise ValueError(
            f"The query must read from the '{TABLE_NAME}' table."
        )

    for table in tables:
        if table.name.lower() != TABLE_NAME:
            raise ValueError(
                f"Only the '{TABLE_NAME}' table can be queried."
            )

        if table.db or table.catalog:
            raise ValueError("Qualified table names are not allowed.")

    # Add a row limit or lower an excessive numeric limit.
    limit = tree.args.get("limit")

    if limit is None:
        tree = tree.limit(MAX_ROWS)
    else:
        limit_value = limit.expression

        if (
            isinstance(limit_value, exp.Literal)
            and limit_value.is_int
            and int(limit_value.this) > MAX_ROWS
        ):
            tree = tree.limit(MAX_ROWS)

    return tree.sql(dialect="sqlite")


def run_sql(
    conn: sqlite3.Connection,
    query: str,
) -> pd.DataFrame:
    """Validate and execute SQL with a time limit."""
    safe_query = validate_sql(query)
    deadline = time.monotonic() + QUERY_TIMEOUT_SECONDS

    def check_deadline():
        return int(time.monotonic() >= deadline)

    # Check execution progress every 1,000 SQLite virtual-machine steps.
    conn.set_progress_handler(check_deadline, 1000)

    try:
        return pd.read_sql_query(safe_query, conn)
    except (sqlite3.Error, ValueError) as exc:
        raise ValueError(
            f"Query execution failed or exceeded its time budget: {exc}"
        ) from exc
    finally:
        conn.set_progress_handler(None, 0)


def dataframe_profile(df: pd.DataFrame) -> dict:
    """Summarize the uploaded dataset."""
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "missing_cells": int(df.isna().sum().sum()),
        "column_names": list(df.columns),
        "dtypes": {
            str(column): str(dtype)
            for column, dtype in df.dtypes.items()
        },
    }
