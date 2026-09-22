from pathlib import Path
import sqlite3
import pandas as pd

# Using pathlib to build paths
DB_PATH = Path(__file__).resolve().parent / "books_database.db"
OUTPUT_PATH = Path(__file__).resolve().parent / "query_outputs.md"


def df_to_markdown_table(df: pd.DataFrame) -> str:
    """Converts a DataFrame to a clean Markdown table string without external dependencies."""
    if df.empty:
        return "_No records found._"

    headers = [str(col) for col in df.columns]
    rows = [[str(val) for val in row] for row in df.values]

    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(val))

    header_line = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    separator_line = "| " + " | ".join("-" * col_widths[i] for i in range(len(headers))) + " |"
    row_lines = ["| " + " | ".join(val.ljust(col_widths[i]) for i, val in enumerate(row)) + " |" for row in rows]

    return "\n".join([header_line, separator_line] + row_lines)


def run_query(title: str, description: str, query: str, conn: sqlite3.Connection) -> tuple[pd.DataFrame, str]:
    """Helper function to execute, display, and return formatted Markdown for SQL query results."""
    print("=" * 80)
    print(f"QUERY: {title}")
    print(f"DESCRIPTION: {description}")
    print(f"SQL:\n{query.strip()}")
    print("-" * 80)
    df = pd.read_sql_query(query, conn)
    print(df.to_string(index=False))
    print(f"Row count: {len(df)}\n")

    # Format into clean Markdown section
    table_md = df_to_markdown_table(df)
    md_section = f"""## {title}

**Description:** {description}

```sql
{query.strip()}
```

**Results ({len(df)} rows):**

{table_md}

---
"""
    return df, md_section


def execute_queries(db_path: Path | str = DB_PATH, output_file: Path | str = OUTPUT_PATH):
    db_path = Path(db_path)
    output_file = Path(output_file)

    if not db_path.exists():
        raise FileNotFoundError(f"Database not found at {db_path}. Please run scraper.py or init_db.py first.")

    conn = sqlite3.connect(db_path)
    md_output_chunks = ["# Books Database SQL Query Outputs\n\n"]

    try:
        # Query 1: SELECT & WHERE - Filter in-stock books with top ratings (rating >= 4)
        q1_title = "High Rated In-Stock Books"
        q1_desc = "Demonstrates SELECT with WHERE condition filtering books having rating >= 4 and in_stock = 1."
        q1_sql = """
SELECT 
    book_id,
    title,
    price_gbp,
    price_inr,
    rating,
    in_stock
FROM books
WHERE rating >= 4 AND in_stock = 1;
        """
        _, md1 = run_query(q1_title, q1_desc, q1_sql, conn)
        md_output_chunks.append(md1)

        # Query 2: DISTINCT - List unique ratings available across the catalogue
        q2_title = "Distinct Star Ratings in Database"
        q2_desc = "Demonstrates DISTINCT to find unique rating scores stored in the books table."
        q2_sql = """
SELECT DISTINCT 
    rating
FROM books
ORDER BY rating ASC;
        """
        _, md2 = run_query(q2_title, q2_desc, q2_sql, conn)
        md_output_chunks.append(md2)

        # Query 3: ORDER BY & LIMIT - Top 5 most expensive books in INR
        q3_title = "Top 5 Most Expensive Books"
        q3_desc = "Demonstrates ORDER BY and LIMIT to find the 5 highest priced books."
        q3_sql = """
SELECT 
    book_id,
    title,
    price_gbp,
    price_inr,
    rating
FROM books
ORDER BY price_inr DESC
LIMIT 5;
        """
        _, md3 = run_query(q3_title, q3_desc, q3_sql, conn)
        md_output_chunks.append(md3)

        # Query 4: INNER JOIN - Books enriched with Category Names (JOIN + WHERE + ORDER BY + LIMIT)
        q4_title = "Books with Category Names (INNER JOIN)"
        q4_desc = "Demonstrates INNER JOIN between books and categories tables, combined with WHERE and ORDER BY."
        q4_sql = """
SELECT 
    b.book_id,
    b.title,
    c.category_name,
    b.price_gbp,
    b.price_inr,
    b.rating
FROM books b
INNER JOIN categories c ON b.category_id = c.category_id
WHERE c.category_name = 'Mystery'
ORDER BY b.price_inr ASC
LIMIT 5;
        """
        _, md4 = run_query(q4_title, q4_desc, q4_sql, conn)
        md_output_chunks.append(md4)

        # Query 5: AGGREGATE JOIN with GROUP BY - Category-level price & book metrics
        q5_title = "Category-Level Aggregations (JOIN + GROUP BY + ORDER BY)"
        q5_desc = "Demonstrates JOIN with GROUP BY to compute total books, average price in INR, and avg rating per category."
        q5_sql = """
SELECT 
    c.category_id,
    c.category_name,
    COUNT(b.book_id) AS total_books,
    ROUND(AVG(b.price_inr), 2) AS avg_price_inr,
    ROUND(AVG(b.price_gbp), 2) AS avg_price_gbp,
    ROUND(AVG(b.rating), 2) AS avg_rating,
    MIN(b.price_inr) AS min_price_inr,
    MAX(b.price_inr) AS max_price_inr
FROM categories c
INNER JOIN books b ON c.category_id = b.category_id
GROUP BY c.category_id, c.category_name
ORDER BY total_books DESC;
        """
        _, md5 = run_query(q5_title, q5_desc, q5_sql, conn)
        md_output_chunks.append(md5)

        # Write all queries and outputs to the Markdown file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(md_output_chunks))
        print(f"All query strings and outputs successfully saved to: {output_file}")

    finally:
        conn.close()


def demonstrate_pandas_equivalence(db_path: Path | str = DB_PATH):
    """
    Demonstrates equivalence between:
    1. SQL executed directly via pd.read_sql
    2. Pure in-memory pandas operations using pd.merge on loaded tables
    """
    print("\n" + "=" * 80)
    print("DEMONSTRATING SQL vs PANDAS DATAFRAME EQUIVALENCE")
    print("=" * 80)

    db_path = Path(db_path)
    conn = sqlite3.connect(db_path)

    try:
        # 1. Read two raw tables into in-memory DataFrames
        df_books_raw = pd.read_sql("SELECT * FROM books;", conn)
        df_categories_raw = pd.read_sql("SELECT * FROM categories;", conn)

        # -------------------------------------------------------------------------
        # Demonstration 1: Filtered Join Query (Query 4)
        # -------------------------------------------------------------------------
        sql_q4 = """
        SELECT 
            b.book_id,
            b.title,
            c.category_name,
            b.price_gbp,
            b.price_inr,
            b.rating
        FROM books b
        INNER JOIN categories c ON b.category_id = c.category_id
        WHERE c.category_name = 'Mystery'
        ORDER BY b.price_inr ASC
        LIMIT 5;
        """
        df_sql_q4 = pd.read_sql(sql_q4, conn)

        # Reproduce purely in pandas using pd.merge (no SQL):
        df_pandas_q4 = (
            pd.merge(
                df_books_raw,
                df_categories_raw,
                on="category_id",
                how="inner"
            )
            .query("category_name == 'Mystery'")
            .sort_values(by="price_inr", ascending=True)
            .head(5)
            [["book_id", "title", "category_name", "price_gbp", "price_inr", "rating"]]
            .reset_index(drop=True)
        )

        print("\n--- [Query 4: Filtered INNER JOIN] ---")
        print("SQL Output (via pd.read_sql):")
        print(df_sql_q4.to_string(index=False))
        print("\nPandas in-memory Output (via pd.merge):")
        print(df_pandas_q4.to_string(index=False))

        is_q4_equal = df_sql_q4.equals(df_pandas_q4)
        print(f"\nAre Query 4 SQL and pd.merge results identical? -> {is_q4_equal}")
        assert is_q4_equal, "Query 4 results do not match!"

        # -------------------------------------------------------------------------
        # Demonstration 2: Aggregation Join Query (Query 5)
        # -------------------------------------------------------------------------
        sql_q5 = """
        SELECT 
            c.category_id,
            c.category_name,
            COUNT(b.book_id) AS total_books,
            ROUND(AVG(b.price_inr), 2) AS avg_price_inr,
            ROUND(AVG(b.price_gbp), 2) AS avg_price_gbp,
            ROUND(AVG(b.rating), 2) AS avg_rating,
            MIN(b.price_inr) AS min_price_inr,
            MAX(b.price_inr) AS max_price_inr
        FROM categories c
        INNER JOIN books b ON c.category_id = b.category_id
        GROUP BY c.category_id, c.category_name
        ORDER BY total_books DESC;
        """
        df_sql_q5 = pd.read_sql(sql_q5, conn)

        # Reproduce purely in pandas using pd.merge + groupby (no SQL):
        merged_raw = pd.merge(df_categories_raw, df_books_raw, on="category_id", how="inner")
        df_pandas_q5 = (
            merged_raw
            .groupby(["category_id", "category_name"], as_index=False)
            .agg(
                total_books=("book_id", "count"),
                avg_price_inr=("price_inr", lambda x: round(x.mean(), 2)),
                avg_price_gbp=("price_gbp", lambda x: round(x.mean(), 2)),
                avg_rating=("rating", lambda x: round(x.mean(), 2)),
                min_price_inr=("price_inr", "min"),
                max_price_inr=("price_inr", "max")
            )
            .sort_values(by="total_books", ascending=False)
            .reset_index(drop=True)
        )

        print("\n--- [Query 5: Category Aggregations JOIN] ---")
        print("SQL Output (via pd.read_sql):")
        print(df_sql_q5.to_string(index=False))
        print("\nPandas in-memory Output (via pd.merge + groupby):")
        print(df_pandas_q5.to_string(index=False))

        is_q5_equal = df_sql_q5.equals(df_pandas_q5)
        print(f"\nAre Query 5 SQL and pd.merge results identical? -> {is_q5_equal}")
        assert is_q5_equal, "Query 5 results do not match!"

        print("\n" + "=" * 80)
        print("SUCCESS: Both SQL and in-memory Pandas methods produce 100% EQUIVALENT results!")
        print("=" * 80 + "\n")

    finally:
        conn.close()


if __name__ == "__main__":
    execute_queries()
    demonstrate_pandas_equivalence()


