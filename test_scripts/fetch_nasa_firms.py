import os
from datetime import datetime
from pathlib import Path
import pandas as pd
import requests
from dotenv import load_dotenv

OUTPUT_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv()

map_key = os.getenv("NASA_API_KEY")
nasa_uri = os.getenv("NASA_FIRMS_URI", "https://firms.modaps.eosdis.nasa.gov/api")

COUNTRY_CODE = 'AUS'
QUERY_METHOD = country
