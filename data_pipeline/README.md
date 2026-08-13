# Module 1 — Data Pipeline

## Overview

This module implements an end-to-end catalog data pipeline:

1. Scrape book data from Books to Scrape.
2. Clean and validate the scraped fields.
3. Convert GBP prices to INR using the required fixed project rate.
4. Load the cleaned data into a normalized SQLite database.
5. Execute SQL queries against the database.
6. Validate SQL results using pandas.
7. Reproduce the SQL JOIN using `pandas.merge()`.

The implementation is fully reproducible from the Python scripts and does not require any paid service or API key.

---

## 1. Data Source

The source used for this project is:

**Books to Scrape**

http://books.toscrape.com/

This is a public scraping-practice website and does not require authentication or an API key.

The scraper uses:

- `requests` for HTTP requests
- `BeautifulSoup` from `bs4` for HTML parsing

The final cleaned dataset contains:

- **100 books**
- **29 categories**
- **0 missing values**

This exceeds the assignment requirement of at least 60 books across at least 3 categories.

---

## 2. Project Structure

```text
data_pipeline/
│
├── README.md
├── scrape_books.py
├── database.py
├── queries.py
├── pandas_validation.py
├── requirements.txt
│
└── output/
    ├── raw_books.csv
    ├── cleaned_books.csv
    ├── books.db
    │
    ├── query_outputs/
    │   ├── query_01_where.csv
    │   ├── query_02_order_limit.csv
    │   ├── query_03_distinct.csv
    │   ├── query_04_between.csv
    │   ├── query_05_in.csv
    │   ├── query_06_join.csv
    │   └── query_results.txt
    │
    └── pandas_validation/
        ├── pd_read_sql_query_01.csv
        ├── pd_read_sql_join.csv
        ├── pd_merge_join.csv
        └── sql_vs_pandas_merge.txt---

## Module 1 Verification

The complete pipeline was executed successfully using:

```powershell
python run_pipeline.py---

## Reproducibility Check

The pipeline is designed to run end to end without manual copy-pasting.

From the `data_pipeline` directory, the complete workflow can be executed with:

```powershell
python run_pipeline.py
output/

Save with:

```text
