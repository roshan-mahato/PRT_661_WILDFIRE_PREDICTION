from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from pathlib import Path

DATABASE_URL = Path(__file__).resolve().parent.parent / "wildfire_db.db"

engine = create_engine(
    f"sqlite:///{DATABASE_URL}",
    connect_args={"check_same_thread": False}  # SQLite specific
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()