# MetricMind — AI-Powered Data Analyst

**Turn your data into insights using natural language.**

MetricMind is an AI-powered data analytics application that allows users to upload CSV datasets, ask business questions in plain English, generate SQL queries using Groq, and visualize results through interactive charts. It focuses on transparent, evidence-based analysis by showing the SQL queries and data behind its conclusions.

## Features

- **Dynamic CSV Upload:** Upload your own CSV dataset for analysis.
- **AI-Powered Analysis:** Ask questions about your data using natural language.
- **SQL Generation:** Convert business questions into SQL queries using Groq.
- **Read-Only Query Execution:** Execute validated analytical queries against a SQLite database.
- **Data Overview:** View dataset dimensions, column information, and missing-value statistics.
- **Interactive Data Explorer:** Search, inspect, and download filtered dataset records.
- **Evidence-Based Findings:** View query results and explanations supporting analytical conclusions.
- **Interactive Visualizations:** Explore results using charts such as bar, line, and scatter plots.
- **SQL Transparency:** Inspect generated SQL before relying on its results.
- **Automated Testing:** Use pytest to test core application functionality.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core application logic |
| Streamlit | Interactive web interface |
| Groq API | Large language model integration |
| SQLite | SQL query execution |
| Pandas | Data processing and analysis |
| SQLGlot | SQL parsing and query transformation |
| Plotly | Interactive data visualizations |
| pytest | Automated testing |
| python-dotenv | Environment variable management |

## Project Structure

```text
MetricMind/
├── .venv/                 # Python virtual environment
├── data/
│   └── test.csv           # Optional sample dataset
├── tests/
│   └── test_database.py   # Database and SQL tests
├── database.py            # CSV-to-SQLite and query utilities
├── main.py                # Streamlit application
├── requirements.txt       # Project dependencies
├── .env                   # API credentials (not committed)
├── .gitignore              # Files excluded from Git
└── README.md               # Project documentation
```

## Getting Started

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd MetricMind
```

Replace `<YOUR_REPOSITORY_URL>` with your GitHub repository URL.

### 2. Create a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If the virtual environment is already created, activate it instead of creating it again.

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Configure Groq

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=your_supported_groq_model
```

Obtain your API key from the [Groq Console](https://console.groq.com/).

Set `GROQ_MODEL` to a model currently available to your Groq account. Never commit your API key to GitHub.

### 5. Run the application

```powershell
python -m streamlit run main.py
```

Streamlit will display a local URL, typically:

```text
http://localhost:8501
```

Open the URL in your browser and upload a CSV file to begin.

## How to Use MetricMind

1. **Upload a dataset:** Select a CSV file from the sidebar.
2. **Explore the data:** Review the number of rows, columns, missing cells, and available fields.
3. **Ask a question:** Open AI Analyst and enter a natural-language business question.
4. **Inspect the SQL:** Review the generated query and verify that it matches your question.
5. **Review the evidence:** Inspect the returned records and the analyst's explanation.
6. **Visualize the results:** Select chart dimensions and a suitable chart type.
7. **Export results:** Download query results or filtered CSV data.

## Example Questions

MetricMind can be used to explore questions such as:

- Which region has the fewest orders?
- What is the total sales value by region?
- Which products generate the highest revenue?
- What are the average sales by customer segment?
- How do sales change over time?
- Which categories have the highest order counts?

The questions that can be answered depend on the columns and data available in the uploaded CSV.

## Running Tests

Run the project's automated tests with:

```powershell
python -m pytest -q
```

Tests help verify that database utilities and SQL restrictions behave as expected.

## Security and Responsible Use

MetricMind is designed to use validated, read-only analytical SQL. However, SQL validation is not a guarantee of complete security.

- Keep Groq credentials in environment variables.
- Do not commit `.env` files or sensitive datasets.
- Treat AI-generated SQL and explanations as untrusted until validated.
- Verify important findings against the returned data.
- Avoid uploading confidential or personally identifiable information to external AI services unless authorized.
- Test SQL restrictions, query execution limits, and error handling before public deployment.

## Current Limitations

- Analysis quality depends on the dataset, the user's question, and the language model's SQL generation.
- Ambiguous questions may require clearer wording.
- Automatic identification of the correct metric and comparison dimension may need further refinement.
- Charts depend on the columns and results returned by the query.
- Production deployment requires additional security, performance, and reliability testing.

## Future Improvements

- Conversation history and follow-up questions.
- Automated chart recommendations.
- More comprehensive SQL safety and regression testing.
- Dataset quality reports and anomaly detection.
- Exportable analytical reports.
- Improved handling of large datasets.
- Deployment with stronger authentication, monitoring, and error reporting.

## Learning Outcomes

This project demonstrates practical experience with:

- Python application development
- Natural-language-to-SQL workflows
- LLM API integration using Groq
- Data cleaning and exploratory analysis
- SQL query validation and execution
- Interactive dashboard development
- Data visualization
- Testing and debugging
- Evidence-based analytical reasoning

## License

Choose a license before distributing this project publicly. If you intend to use an open-source license, add the corresponding `LICENSE` file to the repository.

---

**MetricMind** — Ask better questions. Discover meaningful patterns. Make data-driven decisions.
