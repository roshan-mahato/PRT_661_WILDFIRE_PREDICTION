from sqlalchemy import Boolean, Column, Float, Integer, UniqueConstraint

from backend.db_model.base import Base


class GridCell(Base):
    """
    One row per 0.5-degree grid cell. Every other table points here by
    cell_id instead of repeating float lat/lon, so joins and indexes are on a
    small integer and never depend on float equality.

    is_land is NULL for a cell that has not been probed yet; the land mask
    (data/full_grid_land_mask.csv) and every weather save fill it in.
    """

    __tablename__ = "grid_cell"

    cell_id = Column(Integer, primary_key=True)
    lat_round = Column(Float, nullable=False)
    lon_round = Column(Float, nullable=False)
    is_land = Column(Boolean, nullable=True)

    __table_args__ = (
        UniqueConstraint("lat_round", "lon_round", name="uq_grid_cell_lat_lon"),
    )
