
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from database import (
    create_connection,
    dataframe_profile,
    load_csv,
    run_sql,
)

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = BASE_DIR / "data" / "test.csv"

st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="wide",
)

st.title("📊 AI Data Analyst")
st.caption(
    "Ask questions about your data, inspect SQL, "
    "and visualize evidence."
)


def generate_sql(question: str, df: pd.DataFrame) -> str:
    """Ask Groq to generate SQL for the current dataset."""
    api_key = os.getenv("GROQ_API_KEY")
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Configure it in your .env file."
        )

    schema = {
        str(column): str(dtype)
        for column, dtype in df.dtypes.items()
    }

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": """
You are a careful data analyst who generates SQLite SQL.

Rules:
- The only available table is dataset.
- Use only columns in the supplied schema.
- Return exactly one SELECT query, with no markdown.
- Never generate INSERT, UPDATE, DELETE, DROP, or other
  data-changing statements.
- Do not access any other tables.
- For row-level results, use LIMIT 100.
- Never invent data or column names.
- If the question cannot be answered from the schema,
  return: SELECT * FROM dataset LIMIT 5
""",
            },
            {
                "role": "user",
                "content": (
                    f"Dataset schema: {schema}\n"
                    f"Question: {question}"
                ),
            },
        ],
    )

    sql = response.choices[0].message.content

    if not sql:
        raise RuntimeError("Groq returned an empty response.")

    # SQL validation happens separately before execution.
    return sql.strip()


def explain_results(
    question: str,
    sql: str,
    result: pd.DataFrame,
) -> str:
    """Explain only the evidence returned by the SQL query."""
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return "AI explanation unavailable: configure GROQ_API_KEY."

    client = Groq(api_key=api_key)

    # Keep the evidence sent to the model bounded.
    evidence = result.head(30).to_dict(orient="records")

    response = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a data analyst. Explain the result using "
                    "only the supplied query result. Distinguish facts "
                    "from possible explanations. Mention limitations. "
                    "Do not claim causation without supporting evidence. "
                    "If the result is truncated or insufficient, say so."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n"
                    f"SQL: {sql}\n"
                    f"Result row count: {len(result)}\n"
                    f"Columns: {list(result.columns)}\n"
                    f"Result sample: {evidence}"
                ),
            },
        ],
    )

    return response.choices[0].message.content or (
        "No explanation was returned."
    )


# Load the uploaded CSV or the default sample.
uploaded_file = st.sidebar.file_uploader(
    "Upload a CSV file",
    type=["csv"],
)

try:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = load_csv(DEFAULT_CSV)

    if df.empty or len(df.columns) == 0:
        st.error("The dataset contains no usable data.")
        st.stop()

    df.columns = [str(c).strip() for c in df.columns]

    if (
        any(not c for c in df.columns)
        or df.columns.duplicated().any()
    ):
        st.error("Column names must be non-empty and unique.")
        st.stop()

except Exception as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.stop()


# Create the database for this dataset.
conn = create_connection(df)
profile = dataframe_profile(df)

# Dataset overview.
col1, col2, col3 = st.columns(3)

col1.metric("Rows", f"{profile['rows']:,}")
col2.metric("Columns", f"{profile['columns']:,}")
col3.metric("Missing cells", f"{profile['missing_cells']:,}")

with st.expander("Preview dataset", expanded=True):
    st.dataframe(df.head(100), use_container_width=True)

with st.expander("Schema and data types"):
    st.json(profile["dtypes"])

st.subheader("Ask your data a question")

st.write(
    "Examples: Which region has the highest sales? "
    "What is the average sales value? Show total sales by product."
)

question = st.text_input(
    "Your question",
    placeholder="Which region has the highest sales?",
)

if st.button("Analyze", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    try:
        # Step 1: Generate a query.
        sql = generate_sql(question, df)

        # Step 2: Show the proposed query.
        st.subheader("Generated SQL")
        st.code(sql, language="sql")

        # Step 3: Validate before execution.
        # run_sql calls validate_sql internally.
        result = run_sql(conn, sql)

        # Step 4: Display actual query evidence.
        st.subheader("Query results")
        st.dataframe(result, use_container_width=True)

        st.caption(
            f"Returned {len(result)} rows from a source dataset "
            f"containing {profile['rows']} rows."
        )

        # Step 5: Explain the result using the query evidence.
        with st.spinner("Preparing evidence-based explanation..."):
            explanation = explain_results(
                question,
                sql,
                result,
            )

        st.subheader("Analyst's explanation")
        st.write(explanation)

        # Step 6: Let the user choose a chart.
        if not result.empty and len(result.columns) >= 2:
            st.subheader("Visualize the results")

            numeric_columns = result.select_dtypes(
                include="number"
            ).columns.tolist()

            if numeric_columns:
                x_column = st.selectbox(
                    "X-axis",
                    options=list(result.columns),
                )

                y_column = st.selectbox(
                    "Y-axis",
                    options=numeric_columns,
                )

                chart_type = st.selectbox(
                    "Chart type",
                    ["Bar chart", "Line chart", "Scatter plot"],
                )

                if chart_type == "Bar chart":
                    fig = px.bar(
                        result,
                        x=x_column,
                        y=y_column,
                        title=f"{y_column} by {x_column}",
                    )
                elif chart_type == "Line chart":
                    fig = px.line(
                        result,
                        x=x_column,
                        y=y_column,
                        title=f"{y_column} by {x_column}",
                    )
                else:
                    fig = px.scatter(
                        result,
                        x=x_column,
                        y=y_column,
                        title=f"{y_column} vs {x_column}",
                    )

                st.plotly_chart(fig, use_container_width=True)

            else:
                st.info(
                    "This result has no numeric columns to plot. "
                    "Try an aggregation such as SUM or AVG."
                )

        # Record the evidence shown to the user.
        with st.expander("Evidence and methodology"):
            st.write("**Question:**", question)
            st.write("**SQL executed:**")
            st.code(sql, language="sql")
            st.write("**Result columns:**", list(result.columns))
            st.write("**Rows returned:**", len(result))
            st.write(
                "The explanation is based on the returned query results. "
                "A pattern in these results does not by itself establish "
                "causation."
            )

    except Exception as exc:
        st.error(f"Analysis failed: {exc}")
        st.info(
            "Check your API key, model name, SQL query, "
            "and dataset schema, then try again."
        )

st.divider()
st.caption(
    "AI Data Analyst | Python · SQLite · Groq · Pandas · Plotly"
)
