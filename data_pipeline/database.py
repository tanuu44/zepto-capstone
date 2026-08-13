from pathlib import Path
import sqlite3

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

CLEANED_CSV = OUTPUT_DIR / "cleaned_books.csv"
DATABASE_FILE = OUTPUT_DIR / "books.db"


def get_connection():
    """Create a SQLite connection with foreign-key enforcement enabled."""
    connection = sqlite3.connect(DATABASE_FILE)

    connection.execute("PRAGMA foreign_keys = ON")

    return connection


def create_tables(connection):
    """Create the normalized categories and books tables."""
    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS categories (
            category_id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT NOT NULL UNIQUE
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS books (
            book_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            price_gbp REAL NOT NULL,
            price_inr REAL NOT NULL,
            rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
            in_stock INTEGER NOT NULL CHECK (in_stock IN (0, 1)),
            category_id INTEGER NOT NULL,
            source_url TEXT,
            FOREIGN KEY (category_id)
                REFERENCES categories(category_id)
        )
        """
    )

    connection.commit()


def clear_existing_data(connection):
    """
    Clear previous pipeline output so the database can be
    regenerated from scratch without duplicate rows.
    """
    cursor = connection.cursor()

    cursor.execute("DELETE FROM books")
    cursor.execute("DELETE FROM categories")

    connection.commit()


def load_categories(connection, dataframe):
    """Insert unique categories and return category-name to ID mapping."""
    categories = sorted(
        dataframe["category"].dropna().unique().tolist()
    )

    cursor = connection.cursor()

    cursor.executemany(
        """
        INSERT INTO categories (category_name)
        VALUES (?)
        """,
        [(category,) for category in categories],
    )

    connection.commit()

    rows = cursor.execute(
        """
        SELECT category_id, category_name
        FROM categories
        ORDER BY category_id
        """
    ).fetchall()

    return {
        category_name: category_id
        for category_id, category_name in rows
    }


def load_books(connection, dataframe, category_map):
    """Insert cleaned books using category foreign keys."""
    records = []

    for _, row in dataframe.iterrows():
        category_id = category_map[row["category"]]

        records.append(
            (
                row["title"],
                float(row["price_gbp"]),
                float(row["price_inr"]),
                int(row["rating"]),
                int(bool(row["in_stock"])),
                int(category_id),
                row["source_url"],
            )
        )

    cursor = connection.cursor()

    cursor.executemany(
        """
        INSERT INTO books (
            title,
            price_gbp,
            price_inr,
            rating,
            in_stock,
            category_id,
            source_url
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        records,
    )

    connection.commit()


def validate_database(connection):
    """Print basic database validation results."""
    cursor = connection.cursor()

    category_count = cursor.execute(
        "SELECT COUNT(*) FROM categories"
    ).fetchone()[0]

    book_count = cursor.execute(
        "SELECT COUNT(*) FROM books"
    ).fetchone()[0]

    foreign_key_violations = cursor.execute(
        "PRAGMA foreign_key_check"
    ).fetchall()

    print("\n" + "=" * 60)
    print("DATABASE VALIDATION")
    print("=" * 60)

    print(f"Categories loaded: {category_count}")
    print(f"Books loaded: {book_count}")

    if foreign_key_violations:
        print("Foreign-key check: FAILED")
        print(foreign_key_violations)
    else:
        print("Foreign-key check: PASSED")

    print("\nSample books:")

    sample = cursor.execute(
        """
        SELECT
            b.book_id,
            b.title,
            b.price_gbp,
            b.price_inr,
            b.rating,
            b.in_stock,
            c.category_name
        FROM books b
        JOIN categories c
            ON b.category_id = c.category_id
        ORDER BY b.book_id
        LIMIT 5
        """
    ).fetchall()

    for row in sample:
        print(row)


def load_database():
    """Build the SQLite database from cleaned CSV data."""
    if not CLEANED_CSV.exists():
        raise FileNotFoundError(
            f"Cleaned CSV not found: {CLEANED_CSV}\n"
            "Run scrape_books.py first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataframe = pd.read_csv(CLEANED_CSV)

    required_columns = {
        "title",
        "price_gbp",
        "price_inr",
        "rating",
        "in_stock",
        "category",
        "source_url",
    }

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    # Remove an old database so the pipeline is reproducible.
    if DATABASE_FILE.exists():
        DATABASE_FILE.unlink()

    connection = get_connection()

    try:
        create_tables(connection)
        clear_existing_data(connection)

        category_map = load_categories(
            connection,
            dataframe,
        )

        load_books(
            connection,
            dataframe,
            category_map,
        )

        validate_database(connection)

    finally:
        connection.close()

    print(f"\nDatabase created successfully:")
    print(DATABASE_FILE)

    return DATABASE_FILE


if __name__ == "__main__":
    load_database()