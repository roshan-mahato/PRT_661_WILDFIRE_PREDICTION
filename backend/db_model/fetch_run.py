from sqlalchemy import TIMESTAMP, CheckConstraint, Column, Date, Integer, String

from backend.db_model.base import Base


class FetchRun(Base):
    """
    One row per weather fetch. Every weather_daily row points at the run that
    fetched it, so a forecast issued on Monday and one issued on Tuesday for
    the same day are both kept, and a bad run can be found and removed.

    source:
        archive  -- observed history (fetch_weather_history.py). Archive rows
                    form ONE continuous series per cell; the KBDI recursion
                    runs along it.
        forecast -- one live forecast (fetch_weather_live.py). Its features
                    continue from the latest archive feature row, so a new
                    forecast never changes an older one.
    """

    __tablename__ = "fetch_run"

    run_id = Column(Integer, primary_key=True)
    source = Column(String, nullable=False)
    issued_at = Column(TIMESTAMP, nullable=False)      # when the fetch started (UTC)
    finished_at = Column(TIMESTAMP, nullable=True)
    status = Column(String, nullable=False, default="running")   # running | ok | partial | failed
    window_start = Column(Date, nullable=True)
    window_end = Column(Date, nullable=True)
    rows_written = Column(Integer, nullable=False, default=0)
    note = Column(String, nullable=True)

    __table_args__ = (
        CheckConstraint("source IN ('archive', 'forecast')", name="ck_fetch_run_source"),
    )
