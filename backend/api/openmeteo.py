"""
API for querying OpenMeteo weather data from the database and returning as pandas DataFrames.
"""

import pandas as pd
from datetime import datetime
from typing import Optional, Tuple

from backend.database import SessionLocal
from backend.db_model.openmeteo import OpenMeteo


def get_openmeteo_all() -> pd.DataFrame:
    """
    Retrieve all OpenMeteo weather data from the database.
    
    Returns:
        pd.DataFrame: DataFrame containing all weather observations with columns:
                     [id, latitude, longitude, datetime_utc, fetched_at, temperature_2m,
                      relative_humidity_2m, wind_speed_10m, wind_direction_10m, 
                      wind_gusts_10m, precipitation, soil_moisture_0_to_7cm, cape,
                      vapour_pressure_deficit, et0_fao_evapotranspiration]
    """
    db = SessionLocal()
    try:
        query = db.query(OpenMeteo).all()
        if not query:
            return pd.DataFrame()
        
        data = []
        for record in query:
            data.append({
                'id': record.id,
                'latitude': record.latitude,
                'longitude': record.longitude,
                'datetime_utc': record.datetime_utc,
                'fetched_at': record.fetched_at,
                'temperature_2m': record.temperature_2m,
                'relative_humidity_2m': record.relative_humidity_2m,
                'wind_speed_10m': record.wind_speed_10m,
                'wind_direction_10m': record.wind_direction_10m,
                'wind_gusts_10m': record.wind_gusts_10m,
                'precipitation': record.precipitation,
                'soil_moisture_0_to_7cm': record.soil_moisture_0_to_7cm,
                'cape': record.cape,
                'vapour_pressure_deficit': record.vapour_pressure_deficit,
                'et0_fao_evapotranspiration': record.et0_fao_evapotranspiration,
            })
        
        df = pd.DataFrame(data)
        return df
    finally:
        db.close()


def get_openmeteo_by_location(latitude: float, longitude: float) -> pd.DataFrame:
    """
    Retrieve OpenMeteo weather data for a specific location.
    
    Args:
        latitude (float): Latitude coordinate (WGS84)
        longitude (float): Longitude coordinate (WGS84)
    
    Returns:
        pd.DataFrame: DataFrame containing weather observations for the specified location
    """
    db = SessionLocal()
    try:
        query = db.query(OpenMeteo).filter(
            OpenMeteo.latitude == latitude,
            OpenMeteo.longitude == longitude
        ).all()
        
        if not query:
            return pd.DataFrame()
        
        data = []
        for record in query:
            data.append({
                'id': record.id,
                'latitude': record.latitude,
                'longitude': record.longitude,
                'datetime_utc': record.datetime_utc,
                'fetched_at': record.fetched_at,
                'temperature_2m': record.temperature_2m,
                'relative_humidity_2m': record.relative_humidity_2m,
                'wind_speed_10m': record.wind_speed_10m,
                'wind_direction_10m': record.wind_direction_10m,
                'wind_gusts_10m': record.wind_gusts_10m,
                'precipitation': record.precipitation,
                'soil_moisture_0_to_7cm': record.soil_moisture_0_to_7cm,
                'cape': record.cape,
                'vapour_pressure_deficit': record.vapour_pressure_deficit,
                'et0_fao_evapotranspiration': record.et0_fao_evapotranspiration,
            })
        
        df = pd.DataFrame(data)
        return df.sort_values('datetime_utc')
    finally:
        db.close()


def get_openmeteo_by_date_range(
    start_date: datetime,
    end_date: datetime
) -> pd.DataFrame:
    """
    Retrieve OpenMeteo weather data within a date range.
    
    Args:
        start_date (datetime): Start date (inclusive)
        end_date (datetime): End date (inclusive)
    
    Returns:
        pd.DataFrame: DataFrame containing weather observations within the date range
    """
    db = SessionLocal()
    try:
        query = db.query(OpenMeteo).filter(
            OpenMeteo.datetime_utc >= start_date,
            OpenMeteo.datetime_utc <= end_date
        ).all()
        
        if not query:
            return pd.DataFrame()
        
        data = []
        for record in query:
            data.append({
                'id': record.id,
                'latitude': record.latitude,
                'longitude': record.longitude,
                'datetime_utc': record.datetime_utc,
                'fetched_at': record.fetched_at,
                'temperature_2m': record.temperature_2m,
                'relative_humidity_2m': record.relative_humidity_2m,
                'wind_speed_10m': record.wind_speed_10m,
                'wind_direction_10m': record.wind_direction_10m,
                'wind_gusts_10m': record.wind_gusts_10m,
                'precipitation': record.precipitation,
                'soil_moisture_0_to_7cm': record.soil_moisture_0_to_7cm,
                'cape': record.cape,
                'vapour_pressure_deficit': record.vapour_pressure_deficit,
                'et0_fao_evapotranspiration': record.et0_fao_evapotranspiration,
            })
        
        df = pd.DataFrame(data)
        return df.sort_values('datetime_utc')
    finally:
        db.close()


def get_openmeteo_by_location_and_date(
    latitude: float,
    longitude: float,
    start_date: datetime,
    end_date: datetime
) -> pd.DataFrame:
    """
    Retrieve OpenMeteo weather data for a specific location within a date range.
    
    Args:
        latitude (float): Latitude coordinate (WGS84)
        longitude (float): Longitude coordinate (WGS84)
        start_date (datetime): Start date (inclusive)
        end_date (datetime): End date (inclusive)
    
    Returns:
        pd.DataFrame: DataFrame containing weather observations for the location and date range
    """
    db = SessionLocal()
    try:
        query = db.query(OpenMeteo).filter(
            OpenMeteo.latitude == latitude,
            OpenMeteo.longitude == longitude,
            OpenMeteo.datetime_utc >= start_date,
            OpenMeteo.datetime_utc <= end_date
        ).all()
        
        if not query:
            return pd.DataFrame()
        
        data = []
        for record in query:
            data.append({
                'id': record.id,
                'latitude': record.latitude,
                'longitude': record.longitude,
                'datetime_utc': record.datetime_utc,
                'fetched_at': record.fetched_at,
                'temperature_2m': record.temperature_2m,
                'relative_humidity_2m': record.relative_humidity_2m,
                'wind_speed_10m': record.wind_speed_10m,
                'wind_direction_10m': record.wind_direction_10m,
                'wind_gusts_10m': record.wind_gusts_10m,
                'precipitation': record.precipitation,
                'soil_moisture_0_to_7cm': record.soil_moisture_0_to_7cm,
                'cape': record.cape,
                'vapour_pressure_deficit': record.vapour_pressure_deficit,
                'et0_fao_evapotranspiration': record.et0_fao_evapotranspiration,
            })
        
        df = pd.DataFrame(data)
        return df.sort_values('datetime_utc')
    finally:
        db.close()


def get_openmeteo_latest_by_location(latitude: float, longitude: float) -> pd.DataFrame:
    """
    Retrieve the latest OpenMeteo weather record for a specific location.
    
    Args:
        latitude (float): Latitude coordinate (WGS84)
        longitude (float): Longitude coordinate (WGS84)
    
    Returns:
        pd.DataFrame: DataFrame containing the most recent weather observation for the location
    """
    db = SessionLocal()
    try:
        record = db.query(OpenMeteo).filter(
            OpenMeteo.latitude == latitude,
            OpenMeteo.longitude == longitude
        ).order_by(OpenMeteo.datetime_utc.desc()).first()
        
        if not record:
            return pd.DataFrame()
        
        data = [{
            'id': record.id,
            'latitude': record.latitude,
            'longitude': record.longitude,
            'datetime_utc': record.datetime_utc,
            'fetched_at': record.fetched_at,
            'temperature_2m': record.temperature_2m,
            'relative_humidity_2m': record.relative_humidity_2m,
            'wind_speed_10m': record.wind_speed_10m,
            'wind_direction_10m': record.wind_direction_10m,
            'wind_gusts_10m': record.wind_gusts_10m,
            'precipitation': record.precipitation,
            'soil_moisture_0_to_7cm': record.soil_moisture_0_to_7cm,
            'cape': record.cape,
            'vapour_pressure_deficit': record.vapour_pressure_deficit,
            'et0_fao_evapotranspiration': record.et0_fao_evapotranspiration,
        }]
        
        return pd.DataFrame(data)
    finally:
        db.close()
