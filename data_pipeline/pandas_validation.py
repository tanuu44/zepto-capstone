from pathlib import Path
import sqlite3

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
DATABASE_FILE = OUTPUT_DIR / "books.db"
VALIDATION_DIR = OUTPUT_DIR / "pandas_validation"


def run_pandas_validation():
    """Validate SQL results using pandas read_sql and merge."""

    if not DATABASE_FILE.exists():
        raise FileNotFoundError(
            f"Database not found: {DATABASE_FILE}. "
            "Run database.py first."
        )

    VALIDATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = sqlite3.connect(DATABASE_FILE)

    try:
        # 1. Read SQL Query 1 result using pd.read_sql()
        query_1 = """
            SELECT
                title,
                price_gbp,
                rating,
                in_stock
            FROM books
            WHERE rating >= 4
            ORDER BY rating DESC
        """

        df_sql_1 = pd.read_sql(
            query_1,
            connection,
        )

        # 2. Read SQL JOIN result using pd.read_sql()
        join_query = """
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
        """

        df_sql_join = pd.read_sql(
            join_query,
            connection,
        )

        # 3. Load source tables into pandas
        df_books = pd.read_sql(
            """
            SELECT
                book_id,
                title,
                price_gbp,
                price_inr,
                rating,
                in_stock,
                category_id
            FROM books
            """,
            connection,
        )

        df_categories = pd.read_sql(
            """
            SELECT
                category_id,
                category_name
            FROM categories
            """,
            connection,
        )

    finally:
        connection.close()

    # 4. Reproduce JOIN using pd.merge()
    df_merged = pd.merge(
        df_books,
        df_categories,
        on="category_id",
        how="inner",
    )

    # Apply the same ordering and LIMIT as SQL JOIN query
    df_merge_join = (
        df_merged
        .sort_values(
            by=["rating", "price_gbp"],
            ascending=[False, False],
        )
        .head(10)
        [
            [
                "book_id",
                "title",
                "price_gbp",
                "price_inr",
                "rating",
                "in_stock",
                "category_name",
            ]
        ]
        .reset_index(drop=True)
    )

    df_sql_join_compare = df_sql_join.reset_index(drop=True)

    # 5. Compare SQL JOIN and pandas merge results
    equivalent = df_sql_join_compare.equals(
        df_merge_join
    )

    # 6. Save outputs
    df_sql_1.to_csv(
        VALIDATION_DIR / "pd_read_sql_query_01.csv",
        index=False,
    )

    df_sql_join.to_csv(
        VALIDATION_DIR / "pd_read_sql_join.csv",
        index=False,
    )

    df_merge_join.to_csv(
        VALIDATION_DIR / "pd_merge_join.csv",
        index=False,
    )

    comparison_file = (
        VALIDATION_DIR / "sql_vs_pandas_merge.txt"
    )

    with open(
        comparison_file,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "PANDAS VALIDATION REPORT\n"
        )
        file.write("=" * 80 + "\n\n")

        file.write(
            "1. pd.read_sql() - Query 1\n"
        )
        file.write("-" * 80 + "\n")
        file.write(
            df_sql_1.to_string(index=False)
        )
        file.write("\n\n")

        file.write(
            "2. pd.read_sql() - SQL JOIN\n"
        )
        file.write("-" * 80 + "\n")
        file.write(
            df_sql_join.to_string(index=False)
        )
        file.write("\n\n")

        file.write(
            "3. pd.merge() - Reproduced JOIN\n"
        )
        file.write("-" * 80 + "\n")
        file.write(
            df_merge_join.to_string(index=False)
        )
        file.write("\n\n")

        file.write(
            "4. SQL JOIN vs pandas.merge()\n"
        )
        file.write("-" * 80 + "\n")
        file.write(
            f"Equivalent: {equivalent}\n"
        )

    # 7. Print validation results
    print("\n" + "=" * 80)
    print("PANDAS VALIDATION")
    print("=" * 80)

    print("\n1. pd.read_sql() - Query 1:")
    print(df_sql_1.to_string(index=False))

    print("\n2. pd.read_sql() - SQL JOIN:")
    print(df_sql_join.to_string(index=False))

    print("\n3. pd.merge() - Reproduced JOIN:")
    print(df_merge_join.to_string(index=False))

    print("\n4. SQL JOIN vs pandas.merge():")
    print(f"Equivalent: {equivalent}")

    if equivalent:
        print(
            "\nPASS: SQL JOIN and pandas.merge() "
            "produce equivalent results."
        )
    else:
        print(
            "\nFAIL: SQL JOIN and pandas.merge() "
            "do not match."
        )

    print(
        f"\nValidation files saved to: {VALIDATION_DIR}"
    )


if __name__ == "__main__":
    run_pandas_validation()