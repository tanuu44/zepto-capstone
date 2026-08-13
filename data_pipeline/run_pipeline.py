"""
End-to-end runner for Module 1 Data Pipeline.

Runs:
1. Scraping and cleaning
2. SQLite database creation
3. SQL queries
4. Pandas validation
"""

import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def run_script(script_name):
    script_path = BASE_DIR / script_name

    print("\n" + "=" * 60)
    print(f"RUNNING: {script_name}")
    print("=" * 60)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=BASE_DIR,
        check=False
    )

    if result.returncode != 0:
        print(f"\nFAILED: {script_name}")
        sys.exit(result.returncode)

    print(f"\nCOMPLETED: {script_name}")


def main():
    scripts = [
        "scrape_books.py",
        "database.py",
        "queries.py",
        "pandas_validation.py",
    ]

    for script in scripts:
        run_script(script)

    print("\n" + "=" * 60)
    print("MODULE 1 PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()