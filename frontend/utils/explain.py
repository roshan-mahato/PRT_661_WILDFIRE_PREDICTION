"""
Plain-English reasons behind a grid cell's fire risk.

Turns the conditions returned with each prediction (temperature, humidity,
wind, drought indices) into short phrases like "Very dry air (14%)", so a
user can see *why* a place is rated the way it is.

These are rules of thumb about fire weather, not the model's own reasoning:
the model also weighs location and season, which no weather reading shows.
When the risk is high but the weather is unremarkable, explain_cell() says
so plainly rather than inventing a weather cause.

The API reports daily MEANS for temperature, humidity and wind (not the
afternoon peak), so the thresholds below are set for daily averages -- a
30 C daily mean is a very hot day.
"""

from typing import NamedTuple

from utils.regions import ffdi_band


class Driver(NamedTuple):
    strength: int  # 1 = mild, 2 = notable, 3 = severe
    icon: str      # key into utils.styles icon set
    text: str


# (minimum value, strength, label) -- checked from the most severe down.
_TEMP = [(30, 3, "Very hot"), (25, 2, "Hot"), (21, 1, "Warm")]
_WIND = [(35, 3, "Strong wind"), (22, 2, "Fresh wind"), (15, 1, "Breezy")]
_DROUGHT_FACTOR = [(9, 3, "Fuel extremely dry"), (7, 2, "Dry fuel"), (5, 1, "Drying fuel")]
_KBDI = [(150, 3, "Severe soil drought"), (100, 2, "Dry soil"), (50, 1, "Some soil drying")]
# Humidity: lower is worse, so (maximum value, strength, label).
_HUMIDITY = [(20, 3, "Very dry air"), (30, 2, "Dry air"), (40, 1, "Fairly dry air")]


def _above(value, scale):
    for threshold, strength, label in scale:
        if value >= threshold:
            return strength, label
    return None


def _below(value, scale):
    for threshold, strength, label in scale:
        if value <= threshold:
            return strength, label
    return None


def fire_drivers(row) -> list:
    """Conditions that raise fire danger, strongest first."""
    found = []

    def add(match, icon_name, detail):
        if match:
            strength, label = match
            found.append(Driver(strength, icon_name, f"{label} ({detail})"))

    add(_above(row["temperature_2m"], _TEMP), "thermometer", f"{row['temperature_2m']:.0f}°C average")
    add(_below(row["relative_humidity_2m"], _HUMIDITY), "droplet", f"{row['relative_humidity_2m']:.0f}% humidity")
    add(_above(row["wind_speed_10m"], _WIND), "wind", f"{row['wind_speed_10m']:.0f} km/h average")
    add(_above(row["drought_factor"], _DROUGHT_FACTOR), "flame", f"drought factor {row['drought_factor']:.1f}/10")
    add(_above(row["kbdi"], _KBDI), "sun", f"KBDI {row['kbdi']:.0f}")
    if row["ffdi"] >= 25:
        add((3 if row["ffdi"] >= 50 else 2, f"{ffdi_band(row['ffdi'])} fire danger"), "gauge",
            f"FFDI {row['ffdi']:.0f}")
    return sorted(found, key=lambda d: d.strength, reverse=True)


def calming_factors(row) -> list:
    """Conditions working against fire, so a low rating is explained too."""
    found = []
    if row["relative_humidity_2m"] >= 70:
        found.append(f"humid air ({row['relative_humidity_2m']:.0f}%)")
    if row["temperature_2m"] < 15:
        found.append(f"cool ({row['temperature_2m']:.0f}°C average)")
    if row["drought_factor"] < 3:
        found.append(f"damp fuel (drought factor {row['drought_factor']:.1f}/10)")
    if row["wind_speed_10m"] < 8:
        found.append(f"light wind ({row['wind_speed_10m']:.0f} km/h)")
    return found


def explain_cell(row, limit: int = 4) -> dict:
    """
    Everything the UI needs to explain one cell:
      title    -- heading that fits the risk level
      drivers  -- up to `limit` Driver tuples, strongest first
      calming  -- conditions working against fire
      note     -- extra context (e.g. risk high but weather mild), or ""
    """
    level = str(row["risk_level"])
    drivers = fire_drivers(row)[:limit]
    calming = calming_factors(row)
    elevated = level in ("High", "Extreme")

    note = ""
    if elevated and not any(d.strength >= 2 for d in drivers):
        note = ("The weather here is not extreme. This rating mostly reflects the "
                "area's history of fires and the time of year, which the model also weighs.")
    elif not elevated and not drivers:
        note = "No fire-weather warning signs in the forecast for this place."
    if bool(row.get("kbdi_spinup_flag", False)):
        note = (note + " " if note else "") + (
            "Drought values here are early estimates (under 30 days of history).")

    return {
        "title": "Why is the risk high here?" if elevated else "What's affecting the risk here?",
        "drivers": drivers,
        "calming": calming,
        "note": note,
    }


def short_reason(row) -> str:
    """One line for tight spaces such as map popups: 'Very dry air, Hot, Dry fuel'."""
    drivers = fire_drivers(row)[:3]
    if drivers:
        return ", ".join(d.text.split(" (")[0] for d in drivers)
    calming = calming_factors(row)
    return ("Mild: " + ", ".join(c.split(" (")[0] for c in calming[:2])) if calming else "No notable fire weather"
