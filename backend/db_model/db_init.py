import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "wildfire.db"


def initialize_database() -> Path:
    """Create the SQLite database file if it does not exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("PRAGMA journal_mode=WAL;")
    return DB_PATH


# Create the database automatically on import.
initialize_database()


if __name__ == "__main__":
    print(f"SQLite database ready at {DB_PATH}")
