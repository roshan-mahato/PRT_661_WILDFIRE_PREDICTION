import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import bcrypt
from backend.database import engine, SessionLocal
from backend.db_model.base import Base
from backend.db_model.nasa_firms import NASAFirms

def init_db():
    print("Initializing database...")
    Base.metadata.create_all(bind=engine)
    print("Database initialized.")

    db = SessionLocal()

if __name__ == "__main__":
    init_db()