from pathlib import Path
import sqlite3
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
DATABASE_FILE = OUTPUT_DIR / "books.db"
QUERY_OUTPUT_DIR = OUTPUT_DIR / "query_outputs"


QUERIES = {
    "query_01_where": """
        SELECT title, price_gbp, rating, in_stock
        FROM books
        WHERE rating >= 4
        ORDER BY rating DESC
    """,

    "query_02_order_limit": """
        SELECT title, price_gbp, price_inr
        FROM books
        ORDER BY price_gbp DESC
        LIMIT 10
    """,

    "query_03_distinct": """
        SELECT DISTINCT category_name
        FROM categories
        ORDER BY category_name
    """,

    "query_04_between": """
        SELECT title, price_gbp, rating
        FROM books
        WHERE price_gbp BETWEEN 20 AND 40
        ORDER BY price_gbp
    """,

    "query_05_in": """
        SELECT
            title,
            price_gbp,
            rating,
            category_id
        FROM books
        WHERE rating IN (4, 5)
        ORDER BY rating DESC, price_gbp DESC
    """,

    "query_06_join": """
        SELECT
            b.book_id,
            b.title,
            b.price_gbp,
            b.price_inr,
            b.rating,
            b.in_stock,
            c.category_name
        FROM books AS b
        JOIN categories AS c
            ON b.category_id = c.category_id
        ORDER BY b.rating DESC, b.price_gbp DESC
        LIMIT 10
    """,
}


def run_queries():
    """Execute all required SQL queries and save their outputs."""

    if not DATABASE_FILE.exists():
        raise FileNotFoundError(
            f"Database not found: {DATABASE_FILE}. "
            "Run database.py first."
        )

    QUERY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DATABASE_FILE)

    results = {}

    try:
        results_file = QUERY_OUTPUT_DIR / "query_results.txt"

        with open(results_file, "w", encoding="utf-8") as output_file:

            output_file.write("DATA PIPELINE SQL QUERY RESULTS\n")
            output_file.write("=" * 80 + "\n\n")

            for query_name, query in QUERIES.items():

                print("\n" + "=" * 80)
                print(query_name.upper())
                print("=" * 80)

                print("SQL:")
                print(query.strip())

                dataframe = pd.read_sql_query(
                    query,
                    connection,
                )

                results[query_name] = dataframe

                csv_file = (
                    QUERY_OUTPUT_DIR
                    / f"{query_name}.csv"
                )

                dataframe.to_csv(
                    csv_file,
                    index=False,
                )

                print("\nOUTPUT:")
                print(dataframe.to_string(index=False))

                output_file.write(
                    f"{query_name.upper()}\n"
                )
                output_file.write("-" * 80 + "\n")
                output_file.write("SQL:\n")
                output_file.write(query.strip() + "\n\n")
                output_file.write("OUTPUT:\n")
                output_file.write(
                    dataframe.to_string(index=False)
                )
                output_file.write("\n\n")

    finally:
        connection.close()

    print("\n" + "=" * 80)
    print("SQL QUERY EXECUTION COMPLETE")
    print("=" * 80)
    print(f"Saved query outputs to: {QUERY_OUTPUT_DIR}")
    print(f"Saved query report to: {QUERY_OUTPUT_DIR / 'query_results.txt'}")

    return results


if __name__ == "__main__":
    run_queries()