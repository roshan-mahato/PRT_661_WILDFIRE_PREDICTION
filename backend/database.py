import os
import threading
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

# WILDFIRE_DB points everything at another file (the tests use a temporary one).
DATABASE_URL = Path(os.environ.get(
    "WILDFIRE_DB", Path(__file__).resolve().parent.parent / "wildfire_db.db"
))

engine = create_engine(
    f"sqlite:///{DATABASE_URL}",
    connect_args={"check_same_thread": False, "timeout": 60},  # SQLite specific
)


@event.listens_for(engine, "connect")
def _sqlite_pragmas(dbapi_conn, _record):
    """
    WAL lets the API read while a fetch is writing, synchronous=NORMAL is safe
    under WAL and much faster for bulk inserts, and SQLite only enforces
    foreign keys when asked to, per connection.
    """
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA synchronous=NORMAL")
    cur.execute("PRAGMA foreign_keys=ON")
    cur.close()


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# SQLite allows concurrent readers but only one writer. The fetch scripts
# write from a thread pool, so every write in backend/api goes through this.
write_lock = threading.RLock()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
