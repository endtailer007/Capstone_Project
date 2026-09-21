import os
import sqlite3
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "books_database.db")
SCHEMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.sql")


def init_db(db_path: str = DB_PATH, schema_path: str = SCHEMA_PATH):
    """Executes schema.sql to initialize the SQLite database tables."""
    logger.info(f"Initializing database at: {db_path}")
    with open(schema_path, "r", encoding="utf-8") as f:
        schema_sql = f.read()

    conn = sqlite3.connect(db_path)
    try:
        with conn:
            conn.executescript(schema_sql)
        logger.info("Database initialized successfully with tables: categories, books.")
    finally:
        conn.close()


if __name__ == "__main__":
    init_db()
