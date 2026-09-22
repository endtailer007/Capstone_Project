# Data Pipeline Documentation

### Step 1: Initialize Database
Run `init_db.py` once to initialize the database:
```powershell
python init_db.py
```

### Step 2: Scrape and Ingest Data
Run `scraper.py` to scrape the data from the website and populate the database:
```powershell
python scraper.py
```

> **Note on Data Cleaning & Imputation:**  
> When I was scraping the website I did not find any star rating/price which were missing and all books were available, however I have included conditions in the code to handle, if the program encounters any null values, it will impute them with median for price since median is robust to outliers, and mode for star rating, since it is a discrete value. `in_stock` column has been handled by converting the string to boolean, and if it is not available it will be considered as `False`.

### Step 3: Run SQL Queries & Verification
Run `query_books.py` to run SQL queries on the database created in Step 2:
```powershell
python query_books.py
```
This program will generate `query_outputs.md` file, in which there will be a description for all the queries, the query itself and the output of the query, I have included `demonstrate_pandas_equivalence()` function to demonstrate equivalence between SQL query results and pandas merge results.

---

### Misc / Reference Files
- **Scraped Data**: The scraped books are stored in `scraped_books.csv`.
- **Database Schema**: Database schema is stored in `schema.sql`.
- **Database File**: Database where `categories` and `books` tables are stored in `books_database.db`.
- **scraper.py**: While inserting data into `books` table, I have used 'ON CONFLICT(title, category_id) DO UPDATE SET' to avoid duplicate entries for the same title, so if we run the scraper.py multiple times, it will not insert duplicate entries, but keeps prices, ratings and stock status upto date.
- **schema.sql**: While creating tables, I have used 'IF NOT EXISTS' clause to avoid errors if the tables already exist.