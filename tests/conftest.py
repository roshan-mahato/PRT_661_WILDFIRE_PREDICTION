import os
import tempfile

# Always a fresh temporary database -- never setdefault: the DB tests drop and
# recreate every table, so an inherited WILDFIRE_DB (e.g. the real
# wildfire_db.db) must never be used. Set before anything imports
# backend.database.
os.environ["WILDFIRE_DB"] = os.path.join(tempfile.mkdtemp(prefix="wildfire_test_"), "test.db")
