"""
Plain-English definitions for the technical terms on the dashboard, shown as
tooltips: term("ffdi") renders "FFDI" with a dotted underline, and hovering,
tapping or tabbing to it shows the definition.
"""

from html import escape

GLOSSARY = {
    "ffdi": (
        "FFDI",
        "Forest Fire Danger Index. Combines temperature, humidity, wind and fuel "
        "dryness into one number for how hard a fire would be to control. "
        "Under 12 Low–Moderate, 12–24 High, 25–49 Very High, 50–74 Severe, "
        "75–99 Extreme, 100+ Catastrophic.",
    ),
    "kbdi": (
        "KBDI",
        "Keetch–Byram Drought Index. How dried-out the soil and deeper vegetation "
        "are, from 0 (saturated) to 203 (completely dry). It rises on hot dry "
        "days and drops after rain.",
    ),
    "drought_factor": (
        "Drought factor",
        "How much of the fine fuel (grass, leaves, twigs) is dry enough to burn, "
        "from 0 to 10, based on recent rain and the drought index. 10 is driest.",
    ),
    "grid_cell": (
        "grid cell",
        "The forecast splits Australia into squares 0.5° across (about 55 km). "
        "Each square gets its own prediction, so one value covers the whole square.",
    ),
    "probability": (
        "fire probability",
        "The model's estimated chance that a satellite detects fire in a grid cell "
        "on that day. It is an estimate from weather, drought, location and season.",
    ),
    "risk_level": (
        "Risk level",
        "Fire probability grouped relative to the point where the model raises an "
        "alert. High and Extreme: the model expects fire activity. Moderate: below "
        "the alert point but at least half of it. Low: well below it.",
    ),
    "at_risk": (
        "Cells at risk",
        "Grid cells whose fire probability is above the model's alert threshold, "
        "i.e. cells rated High or Extreme.",
    ),
}


def term(key: str, label: str = "") -> str:
    """HTML for a term with its definition as a tooltip."""
    default_label, definition = GLOSSARY[key]
    return (
        f'<span class="term" tabindex="0" role="note" aria-label="{escape(definition)}" '
        f'data-tip="{escape(definition)}">{label or default_label}</span>'
    )
