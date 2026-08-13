from pathlib import Path
import re
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


BASE_URL = "https://books.toscrape.com"
GBP_TO_INR = 105.50
PAGES_TO_SCRAPE = 5

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
RAW_OUTPUT = OUTPUT_DIR / "raw_books.csv"
CLEAN_OUTPUT = OUTPUT_DIR / "cleaned_books.csv"


RATING_MAP = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5,
}


def create_session():
    """Create a requests session with retries."""
    session = requests.Session()

    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )

    adapter = HTTPAdapter(max_retries=retry)

    session.mount("https://", adapter)
    session.mount("http://", adapter)

    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/142.0 Safari/537.36"
            )
        }
    )

    return session


def get_soup(session, url):
    """Download a page and return BeautifulSoup."""
    response = session.get(url, timeout=20)
    response.raise_for_status()

    return BeautifulSoup(response.text, "html.parser")


def parse_price(price_text):
    """Convert a GBP price such as £51.77 to float."""
    if not price_text:
        return None

    match = re.search(r"([\d.]+)", price_text.replace(",", ""))

    if not match:
        return None

    try:
        return float(match.group(1))
    except ValueError:
        return None


def parse_rating(rating_element):
    """Convert textual star rating to integer 1-5."""
    if rating_element is None:
        return None

    classes = rating_element.get("class", [])

    for class_name in classes:
        if class_name in RATING_MAP:
            return RATING_MAP[class_name]

    return None


def parse_stock(availability_text):
    """Convert availability text to boolean."""
    if not availability_text:
        return None

    text = availability_text.strip().lower()

    if "in stock" in text:
        return True

    if "out of stock" in text:
        return False

    return None


def parse_category(detail_soup):
    """Extract the category from the product breadcrumb."""
    breadcrumb = detail_soup.select("ul.breadcrumb li a")

    if breadcrumb:
        # Breadcrumb is Home > Books > Category
        return breadcrumb[-1].get_text(strip=True)

    return None


def scrape_book_detail(session, url, listing_data):
    """Scrape category from an individual book page."""
    try:
        soup = get_soup(session, url)

        category = parse_category(soup)

        return {
            **listing_data,
            "category": category,
            "source_url": url,
        }

    except requests.RequestException as exc:
        print(f"WARNING: Could not fetch detail page: {url}")
        print(f"Reason: {exc}")

        return {
            **listing_data,
            "category": None,
            "source_url": url,
        }


def scrape_listing_page(session, page_number):
    """Scrape one catalogue page."""
    if page_number == 1:
        url = f"{BASE_URL}/catalogue/page-1.html"
    else:
        url = f"{BASE_URL}/catalogue/page-{page_number}.html"

    print(f"Scraping listing page {page_number}: {url}")

    soup = get_soup(session, url)

    books = soup.select("article.product_pod")

    page_records = []

    for book in books:
        title_element = book.select_one("h3 a")
        price_element = book.select_one("p.price_color")
        rating_element = book.select_one("p.star-rating")
        availability_element = book.select_one("p.availability")

        title = (
            title_element.get("title", "").strip()
            if title_element
            else None
        )

        price_text = (
            price_element.get_text(" ", strip=True)
            if price_element
            else None
        )

        rating_text = None

        if rating_element:
            classes = rating_element.get("class", [])

            for class_name in classes:
                if class_name in RATING_MAP:
                    rating_text = class_name
                    break

        availability_text = (
            availability_element.get_text(" ", strip=True)
            if availability_element
            else None
        )

        href = title_element.get("href") if title_element else None

        if not href:
            continue

        detail_url = requests.compat.urljoin(url, href)

        page_records.append(
            {
                "title": title,
                "price": price_text,
                "star_rating": rating_text,
                "availability": availability_text,
                "detail_url": detail_url,
            }
        )

    return page_records


def scrape_books():
    """Scrape the first five catalogue pages."""
    session = create_session()

    all_records = []

    for page_number in range(1, PAGES_TO_SCRAPE + 1):
        try:
            records = scrape_listing_page(session, page_number)

            for record in records:
                detail_record = scrape_book_detail(
                    session,
                    record["detail_url"],
                    record,
                )

                all_records.append(detail_record)

                # Be polite to the public practice website.
                time.sleep(0.05)

        except requests.RequestException as exc:
            print(
                f"WARNING: Failed to scrape page {page_number}: {exc}"
            )

    return pd.DataFrame(all_records)


def clean_books(df):
    """Clean raw scraped data and calculate INR prices."""
    cleaned = df.copy()

    # Convert GBP price to numeric.
    cleaned["price_gbp"] = cleaned["price"].apply(parse_price)

    # Convert textual rating to integer.
    cleaned["rating"] = cleaned["star_rating"].map(RATING_MAP)

    # Convert availability to boolean.
    cleaned["in_stock"] = cleaned["availability"].apply(parse_stock)

    # Numeric fields use median imputation as required.
    price_median = cleaned["price_gbp"].median()
    rating_median = cleaned["rating"].median()

    if pd.isna(price_median):
        raise ValueError("Unable to calculate price median.")

    if pd.isna(rating_median):
        raise ValueError("Unable to calculate rating median.")

    cleaned["price_gbp"] = cleaned["price_gbp"].fillna(price_median)

    cleaned["rating"] = (
        cleaned["rating"]
        .fillna(round(rating_median))
        .astype(int)
    )

    # If essential categorical/boolean fields fail to parse,
    # drop those rows rather than inventing values.
    cleaned = cleaned.dropna(
        subset=["title", "category", "in_stock"]
    ).copy()

    cleaned["in_stock"] = cleaned["in_stock"].astype(bool)

    # Required project-defined fixed conversion.
    cleaned["price_inr"] = (
        cleaned["price_gbp"] * GBP_TO_INR
    ).round(2)

    cleaned["rating"] = cleaned["rating"].clip(1, 5)

    # Keep only the fields needed downstream.
    cleaned = cleaned[
        [
            "title",
            "price_gbp",
            "price_inr",
            "rating",
            "in_stock",
            "category",
            "source_url",
        ]
    ].reset_index(drop=True)

    return cleaned


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("BOOKS TO SCRAPE - DATA COLLECTION")
    print("=" * 60)

    raw_df = scrape_books()

    if raw_df.empty:
        raise RuntimeError("No books were scraped.")

    raw_df.to_csv(RAW_OUTPUT, index=False)

    print(f"\nRaw rows scraped: {len(raw_df)}")

    cleaned_df = clean_books(raw_df)

    if len(cleaned_df) < 60:
        raise RuntimeError(
            f"Only {len(cleaned_df)} cleaned rows available. "
            "At least 60 are required."
        )

    category_count = cleaned_df["category"].nunique()

    if category_count < 3:
        raise RuntimeError(
            f"Only {category_count} categories found. "
            "At least 3 are required."
        )

    cleaned_df.to_csv(CLEAN_OUTPUT, index=False)

    print("\nCleaning complete.")
    print(f"Clean rows: {len(cleaned_df)}")
    print(f"Categories: {category_count}")
    print(f"GBP -> INR rate: {GBP_TO_INR}")

    print("\nData types:")
    print(cleaned_df.dtypes)

    print("\nCategory counts:")
    print(cleaned_df["category"].value_counts())

    print(f"\nSaved raw data to: {RAW_OUTPUT}")
    print(f"Saved cleaned data to: {CLEAN_OUTPUT}")

    return cleaned_df


if __name__ == "__main__":
    main()