import csv
import json
import logging
import os
import re
from urllib.parse import urljoin

from bs4 import BeautifulSoup
import pandas as pd
import requests
import sqlite3


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

BASE_URL = "https://books.toscrape.com/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

RATING_CLASSES = ["One", "Two", "Three", "Four", "Five"]

# Project's fixed baseline conversion rate (1 GBP = 105.50 INR)
GBP_TO_INR_RATE = 105.50


def get_categories(session: requests.Session) -> list[dict]:
    """Extract category names and their URLs from the main sidebar navigation."""
    logger.info(f"Fetching home page to discover categories: {BASE_URL}")
    response = session.get(BASE_URL, headers=HEADERS, timeout=15)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, "html.parser")
    cat_container = soup.find("ul", class_="nav-list")
    if not cat_container:
        raise ValueError("Could not find category navigation element on page.")

    # The first 'ul' child inside 'nav-list' contains the individual categories
    sub_ul = cat_container.find("ul")
    if not sub_ul:
        raise ValueError("Could not find categories list.")

    categories = []
    for li in sub_ul.find_all("li", recursive=False):
        a_tag = li.find("a")
        if a_tag and "href" in a_tag.attrs:
            cat_name = a_tag.get_text(strip=True)
            cat_url = urljoin(BASE_URL, a_tag["href"])
            categories.append({"name": cat_name, "url": cat_url})

    logger.info(f"Discovered {len(categories)} categories.")
    return categories


def parse_books_from_page(html_content: str | bytes, category_name: str) -> tuple[list[dict], str | None]:
    """Parse book information and return the list of books along with next page URL if any."""
    soup = BeautifulSoup(html_content, "html.parser")
    books = []

    product_pods = soup.find_all("article", class_="product_pod")
    for pod in product_pods:
        # Title
        h3 = pod.find("h3")
        a_title = h3.find("a") if h3 else None
        title = a_title.get("title", "").strip() if a_title and a_title.get("title") else (a_title.get_text(strip=True) if a_title else "")

        # Price
        price_elem = pod.find("p", class_="price_color")
        price = price_elem.get_text(strip=True) if price_elem else ""

        # Star Rating
        star_elem = pod.find("p", class_="star-rating")
        star_rating = ""
        if star_elem:
            classes = star_elem.get("class", [])
            for c in classes:
                if c in RATING_CLASSES:
                    star_rating = c
                    break

        # Availability
        avail_elem = pod.find("p", class_="instock availability")
        availability = re.sub(r"\s+", " ", avail_elem.get_text()).strip() if avail_elem else ""

        books.append({
            "title": title,
            "price": price,
            "star_rating": star_rating,
            "availability": availability,
            "category": category_name
        })

    # Check for next page
    next_btn = soup.find("li", class_="next")
    next_relative_url = None
    if next_btn and next_btn.find("a"):
        next_relative_url = next_btn.find("a")["href"]

    return books, next_relative_url


def scrape_category(session: requests.Session, category: dict, max_books: int | None = None) -> list[dict]:
    """Scrapes all books within a given category, traversing pagination if needed."""
    cat_name = category["name"]
    current_url = category["url"]
    books_collected = []

    logger.info(f"Scraping category: '{cat_name}' from {current_url}")

    while current_url:
        resp = session.get(current_url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        resp.encoding = "utf-8"

        page_books, next_rel_url = parse_books_from_page(resp.text, cat_name)
        books_collected.extend(page_books)
        logger.info(f"  Scraped {len(page_books)} books from {current_url}. Total in category: {len(books_collected)}")

        if max_books and len(books_collected) >= max_books:
            books_collected = books_collected[:max_books]
            break

        if next_rel_url:
            current_url = urljoin(current_url, next_rel_url)
        else:
            current_url = None

    return books_collected


def scrape_books_to_dataframe(min_books: int = 60, min_categories: int = 3) -> pd.DataFrame:
    """Scrapes books from books.toscrape.com and loads raw records directly into a pandas DataFrame."""
    with requests.Session() as session:
        categories = get_categories(session)
        if not categories:
            logger.error("No categories found.")
            return pd.DataFrame()

        all_books = []
        categories_scraped = 0

        for cat in categories:
            books = scrape_category(session, cat)
            if books:
                all_books.extend(books)
                categories_scraped += 1

            if categories_scraped >= min_categories and len(all_books) >= min_books:
                break

        logger.info(
            f"Scraping completed! Total books scraped: {len(all_books)} "
            f"across {categories_scraped} categories."
        )

        # Transfer raw records directly into a pandas DataFrame (no cleaning/validation)
        df = pd.DataFrame(all_books)
        return df


def main():
    target_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(target_dir, "scraped_books.csv")

    df = scrape_books_to_dataframe(min_books=60, min_categories=3)

    logger.info(f"\nDataFrame Info:\n{df.info()}")
    logger.info(f"\nFirst 5 rows of raw DataFrame:\n{df.head()}")

    # Persist DataFrame to CSV
    df.to_csv(csv_file, index=False, encoding="utf-8")
    logger.info(f"Saved raw DataFrame to CSV: {csv_file}")
    return df


def save_to_sqlite(df: pd.DataFrame, db_path: str):
    """
    Inserts cleaned DataFrame records into the normalized SQLite schema
    (categories and books tables).
    """
    logger.info(f"Saving normalized records to SQLite database: {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Enable Foreign Key support
        cursor.execute("PRAGMA foreign_keys = ON;")

        # 1. Insert unique categories (avoid duplicates)
        unique_categories = df['category'].drop_duplicates().tolist()
        cursor.executemany(
            "INSERT OR IGNORE INTO categories (category_name) VALUES (?);",
            [(cat,) for cat in unique_categories]
        )
        conn.commit()

        # 2. Query category_id mapping from database
        cursor.execute("SELECT category_name, category_id FROM categories;")
        category_map = dict(cursor.fetchall())

        # Map category_id onto DataFrame
        df_to_save = df.copy()
        df_to_save['category_id'] = df_to_save['category'].map(category_map)

        # 3. Insert Books
        books_records = [
            (
                str(row['title']),
                float(row['price_gbp']),
                float(row['price_inr']),
                int(row['star_rating']),
                int(row['in_stock']),
                int(row['category_id'])
            )
            for _, row in df_to_save.iterrows()
        ]

        cursor.executemany("""
        INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(title, category_id) DO UPDATE SET
            price_gbp = excluded.price_gbp,
            price_inr = excluded.price_inr,
            rating    = excluded.rating,
            in_stock  = excluded.in_stock;
        """, books_records)

        conn.commit()
        logger.info(f"Successfully upserted {len(books_records)} books across {len(unique_categories)} categories into SQLite!")
    finally:
        conn.close()


if __name__ == "__main__":
    df = main()
    # Stripping the currency symbol, if the currency symbol is other than £, it will be converted to NaN and handled by the 'coerce' parameter
    df['price'] = pd.to_numeric(
        df['price'].astype(str).str.replace('£','', regex = False),
        errors = 'coerce'
    )
    #Rename price column to price_gbp
    df.rename(columns = {'price': 'price_gbp'},inplace=True)
    median_price = df['price_gbp'].median()
    #Fill the null values with median price, because median is not sensitive to outliers.
    df['price_gbp'] = df['price_gbp'].fillna(median_price)
    # Map rating words to integers
    rating_map = {
        "One": 1,
        "Two": 2,
        "Three": 3,
        "Four": 4,
        "Five": 5
    }
    df['star_rating'] = df['star_rating'].str.title().map(rating_map)
    #Impute with the mode since this is a discrete variable.
    if not df['star_rating'].mode().empty:
        mode_rating = df['star_rating'].mode()[0]
        df['star_rating'] = df['star_rating'].fillna(mode_rating)
    df['star_rating'] = df['star_rating'].astype(int)
    df['in_stock'] = df['availability'].str.contains("in stock", case=False, na=False)
    # Convert price_gbp to price_inr using project baseline fixed conversion rate: 1 GBP = 105.50 INR
    df['price_inr'] = (df['price_gbp'] * GBP_TO_INR_RATE).round(2)

    # Insert into SQLite Database
    target_dir = os.path.dirname(os.path.abspath(__file__))
    db_file = os.path.join(target_dir, "books_database.db")
    save_to_sqlite(df, db_file)