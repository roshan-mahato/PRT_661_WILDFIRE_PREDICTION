"""
Risk levels and fire-danger (FFDI) bands.

Two scales appear on the dashboard and are easy to confuse:

- RISK LEVEL (Low / Moderate / High / Extreme) comes from the model's fire
  probability, banded around its alert threshold by the backend.
- FFDI BAND is the McArthur Forest Fire Danger Index rating, computed from
  the weather. A cell can sit in a high FFDI band while the model gives it a
  low fire probability, and vice versa.
"""

RISK_ORDER = ["Low", "Moderate", "High", "Extreme"]

# McArthur FFDI bands: (lower bound, name).
FFDI_BANDS = [
    (0, "Low–Moderate"),
    (12, "High"),
    (25, "Very High"),
    (50, "Severe"),
    (75, "Extreme"),
    (100, "Catastrophic"),
]


def ffdi_band(ffdi: float) -> str:
    """
    FFDI band name as shown in the cards, banner and map popups.

    Same boundaries as FFDI_BANDS. The lowest band is written with a hyphen
    ("Low-Moderate") here and with an en dash in the charts (ffdi_band_name);
    both spellings are kept so the dashboard's text is unchanged.
    """
    if ffdi < 12:
        return "Low-Moderate"
    if ffdi < 25:
        return "High"
    if ffdi < 50:
        return "Very High"
    if ffdi < 75:
        return "Severe"
    if ffdi < 100:
        return "Extreme"
    return "Catastrophic"


def ffdi_band_name(ffdi: float) -> str:
    """FFDI band name as shown in the charts (en dash: "Low–Moderate")."""
    name = FFDI_BANDS[0][1]
    for lower, band in FFDI_BANDS:
        if ffdi >= lower:
            name = band
    return name
