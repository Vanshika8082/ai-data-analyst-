
from pathlib import Path
import sqlite3

import pandas as pd
import sqlglot
from sqlglot import exp

TABLE_NAME = "dataset"
MAX_ROWS = 500


def load_csv(file_path: str | Path) -> pd.DataFrame:
    """Load and validate a CSV dataset."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("The dataset contains no rows.")

    if len(df.columns) == 0:
        raise ValueError("The dataset has no columns.")

    df.columns = [str(c).strip() for c in df.columns]

    if any(not c for c in df.columns):
        raise ValueError("Column names cannot be empty.")

    if df.columns.duplicated().any():
        raise ValueError("Column names must be unique.")

    return df


def create_connection(df: pd.DataFrame) -> sqlite3.Connection:
    """Create an in-memory, read-only SQLite database."""
    conn = sqlite3.connect(":memory:")

    df.to_sql(TABLE_NAME, conn, index=False)

    # Prevent INSERT, UPDATE, DELETE, and other writes.
    conn.execute("PRAGMA query_only = ON")

    return conn


def validate_sql(query: str) -> str:
    """Allow one SELECT statement on the dataset table only."""
    if not query or not query.strip():
        raise ValueError("SQL query cannot be empty.")

    if len(query) > 10_000:
        raise ValueError("SQL query is too long.")

    try:
        statements = sqlglot.parse(query, read="sqlite")
    except sqlglot.errors.ParseError as exc:
        raise ValueError("Invalid SQL syntax.") from exc

    statements = [s for s in statements if s is not None]

    if len(statements) != 1:
        raise ValueError("Only one SQL statement is allowed.")

    tree = statements[0]

    if not isinstance(tree, exp.Select):
        raise ValueError("Only SELECT queries are allowed.")

    for table in tree.find_all(exp.Table):
        if table.name.lower() != TABLE_NAME:
            raise ValueError(
                f"Only the '{TABLE_NAME}' table can be queried."
            )

        if table.db or table.catalog:
            raise ValueError("Qualified table names are not allowed.")

    # Add a maximum result size if one wasn't specified.
    if tree.args.get("limit") is None:
        tree = tree.limit(MAX_ROWS)

    return tree.sql(dialect="sqlite")


def run_sql(
    conn: sqlite3.Connection,
    query: str
) -> pd.DataFrame:
    """Validate and execute a SQL query."""
    safe_query = validate_sql(query)

    try:
        return pd.read_sql_query(safe_query, conn)
    except (sqlite3.Error, ValueError) as exc:
        raise ValueError(f"Query execution failed: {exc}") from exc


def dataframe_profile(df: pd.DataFrame) -> dict:
    """Summarize the dataset for the assistant."""
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_cells": int(df.isna().sum().sum()),
        "column_names": list(df.columns),
        "dtypes": {
            str(column): str(dtype)
            for column, dtype in df.dtypes.items()
        },
    }
