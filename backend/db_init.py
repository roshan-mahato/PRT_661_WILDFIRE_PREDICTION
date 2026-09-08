import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import bcrypt

from backend.database import SessionLocal, engine
from backend.db_model.base import Base
from backend.db_model.nasa_firms import NASAFirms
from backend.db_model.openmeteo import OpenMeteo


def init_db():
    print("Initializing database...")
    Base.metadata.create_all(bind=engine)
    print("Database initialized.")

    db = SessionLocal()


if __name__ == "__main__":
    init_db()
