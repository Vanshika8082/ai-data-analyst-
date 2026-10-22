
import pandas as pd
import pytest

from database import (
    create_connection,
    run_sql,
    validate_sql,
)


@pytest.fixture
def connection():
    df = pd.DataFrame({
        "region": ["North", "South", "East"],
        "sales": [100, 200, 150],
    })
    return create_connection(df)


def test_select_query(connection):
    result = run_sql(
        connection,
        "SELECT region, sales FROM dataset ORDER BY sales",
    )

    assert result["sales"].tolist() == [100, 150, 200]


def test_aggregation(connection):
    result = run_sql(
        connection,
        "SELECT SUM(sales) AS total_sales FROM dataset",
    )

    assert result.iloc[0]["total_sales"] == 450


def test_group_by(connection):
    result = run_sql(
        connection,
        "SELECT region, COUNT(*) AS total "
        "FROM dataset GROUP BY region",
    )

    assert len(result) == 3


def test_delete_is_rejected():
    with pytest.raises(ValueError):
        validate_sql("DELETE FROM dataset")


def test_other_table_is_rejected():
    with pytest.raises(ValueError):
        validate_sql("SELECT * FROM users")


def test_multiple_statements_are_rejected():
    with pytest.raises(ValueError):
        validate_sql(
            "SELECT * FROM dataset; DROP TABLE dataset",
        )


def test_empty_sql_is_rejected():
    with pytest.raises(ValueError):
        validate_sql("")


def test_cte_is_rejected():
    with pytest.raises(ValueError):
        validate_sql(
            "WITH summary AS "
            "(SELECT * FROM dataset) "
            "SELECT * FROM summary",
        )


def test_nested_subquery_is_rejected():
    with pytest.raises(ValueError):
        validate_sql(
            "SELECT * FROM dataset "
            "WHERE sales > (SELECT AVG(sales) FROM dataset)",
        )


def test_result_limit_is_added():
    query = validate_sql("SELECT * FROM dataset")

    assert "LIMIT 500" in query.upper()


def test_excessive_result_limit_is_reduced():
    query = validate_sql("SELECT * FROM dataset LIMIT 100000")

    assert "LIMIT 500" in query.upper()


def test_database_rejects_writes(connection):
    with pytest.raises(Exception):
        connection.execute(
            "DELETE FROM dataset WHERE region = 'North'"
        )


def test_unknown_table_is_rejected_at_execution(connection):
    # Even if a caller bypasses validate_sql, SQLite authorization
    # should still reject reads from other tables.
    with pytest.raises(Exception):
        connection.execute("SELECT * FROM sqlite_master").fetchall()
