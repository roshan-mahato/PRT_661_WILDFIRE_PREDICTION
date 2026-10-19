"""
Status banner at the top of the page: the one-glance answer to "how bad is it,
where, and when?", followed by what that means and where official warnings
live.
"""

import pandas as pd

from utils.glossary import term
from core.danger import RISK_ORDER, ffdi_band
from core.geography import STATE_NAMES, is_national
from core.places import describe_location, format_coords
from utils.styles import html, icon
from utils.theme import COLORS

# Lighter tints of the severity colours: the banner is dark, and the regular
# severity colours are too dim to read against it.
SEVERITY_ON_DARK = {
    "Extreme": ("#FF8F80", "rgba(224, 64, 48, 0.55)"),
    "High": ("#FFB27A", "rgba(232, 118, 44, 0.50)"),
    "Moderate": ("#F4CF63", "rgba(214, 164, 30, 0.42)"),
    "Low": ("#86D6A0", "rgba(62, 136, 88, 0.45)"),
}

ADVICE = {
    "Extreme": "Fire activity is very likely in parts of this region. Stay across "
    "official warnings, avoid activities that could start a fire, and have your "
    "bushfire plan ready.",
    "High": "Fire activity is likely in parts of this region. Check official "
    "warnings before any outdoor burning or travel through bushland.",
    "Moderate": "There is some chance of fire activity. Keep an eye on conditions "
    "and official updates.",
    "Low": "Fire activity is unlikely across this region on this day.",
}

# State fire services -- where official warnings for each region are issued.
FIRE_AUTHORITIES = {
    "New South Wales": ("NSW Rural Fire Service", "https://www.rfs.nsw.gov.au"),
    "Victoria": ("Country Fire Authority", "https://www.cfa.vic.gov.au"),
    "Queensland": ("Queensland Fire Department", "https://www.fire.qld.gov.au"),
    "Western Australia": ("DFES Western Australia", "https://www.dfes.wa.gov.au"),
    "South Australia": ("SA Country Fire Service", "https://www.cfs.sa.gov.au"),
    "Tasmania": ("Tasmania Fire Service", "https://www.fire.tas.gov.au"),
    "Northern Territory": ("Bushfires NT", "https://bushfires.nt.gov.au"),
    "Australian Capital Territory": ("ACT Emergency Services Agency", "https://esa.act.gov.au"),
}
NATIONAL_AUTHORITY = ("Australian Warning System", "https://www.australianwarningsystem.com.au")


def _scale(level: str) -> str:
    """
    The four risk levels as equal steps, with a marker in the middle of the
    current one.

    The steps are levels, not a probability axis: the level boundaries are
    scaled around the model's alert threshold (Scripts/prediction_engine.py),
    so they are not evenly spaced in probability and the API does not return
    them. Placing the marker by level keeps it correct whatever the bands are.
    """
    segments = "".join(
        f'<span style="background:{COLORS[name.lower()]}"></span>' for name in RISK_ORDER
    )
    labels = "".join(f"<span>{name}</span>" for name in RISK_ORDER)
    position = RISK_ORDER.index(level) if level in RISK_ORDER else 0
    left = (position + 0.5) / len(RISK_ORDER) * 100
    return (
        '<div class="scale">'
        f'<div class="scale-track">{segments}</div>'
        f'<div class="scale-marker" style="left:{left:.1f}%"></div>'
        f'<div class="scale-labels">{labels}</div>'
        "</div>"
    )


def render_hero(summary: dict, region: str, source_label: str) -> None:
    """
    Args:
        summary: output of core.summaries.summarise_region() for the selected day.
        region: selected region name.
        source_label: where the numbers came from ("from the last saved prediction run").
    """
    place = "Australia" if is_national(region) else region
    authority, url = NATIONAL_AUTHORITY if is_national(region) else FIRE_AUTHORITIES.get(
        region, NATIONAL_AUTHORITY
    )

    if not summary.get("has_data"):
        light, glow = "#B8ADA3", "rgba(184, 173, 163, 0.25)"
        html(
            f"""
            <div class="hero empty" style="--sev-light:{light};--sev-glow:{glow};">
              <div>
                <div class="hero-eyebrow"><span>{icon("pin", 13)} {place}</span></div>
                <div class="hero-title">No forecast for this region yet</div>
                <div class="hero-text">There are no predictions for {place} in the
                  current data. Try another region, switch to <b>Live</b>, or press
                  <b>Refresh</b> to generate a new forecast.</div>
              </div>
            </div>
            """
        )
        return

    level = summary["max_risk_level"]
    light, glow = SEVERITY_ON_DARK[level]
    day = pd.Timestamp(summary["forecast_date"])
    cells, at_risk = summary["cell_count"], summary["at_risk_count"]
    share = at_risk / cells if cells else 0
    lat, lon = summary["hottest_cell"]

    html(
        f"""
        <div class="hero" style="--sev-light:{light};--sev-glow:{glow};">
          <div>
            <div class="hero-eyebrow">
              <span>{icon("pin", 13)} {place}</span>
              <span>{icon("calendar", 13)} {day.strftime("%A %d %B %Y")}</span>
            </div>
            <div class="hero-title"><span class="level">{level}</span> fire risk</div>
            <div class="hero-text">
              <b>{at_risk} of {cells}</b> monitored {term("grid_cell", "grid cells")} ({share:.0%})
              are flagged for likely fire activity. The highest risk is
              <b>{describe_location(lat, lon)}</b> ({format_coords(lat, lon)}).
            </div>
            <div class="hero-advice">
              {icon("alert", 18)}
              <div>{ADVICE[level]} Official warnings:
                <a href="{url}" target="_blank" rel="noopener">{authority}</a>.</div>
            </div>
            <div class="hero-meta">{icon("database", 13)} Showing {cells} grid cell(s)
              for {region} — forecast day {summary["forecast_date"]}, {source_label}.</div>
          </div>
          <div class="hero-gauge">
            <div class="hero-gauge-label">Peak {term("probability")}</div>
            <div class="hero-gauge-value">{summary["max_probability"] * 100:.0f}<small>%</small></div>
            {_scale(level)}
            <div class="hero-mini">
              <div>Regional average<b>{summary["mean_probability"]:.0%}</b></div>
              <div>Peak {term("ffdi")}<b>{summary["max_ffdi"]:.1f} · {ffdi_band(summary["max_ffdi"])}</b></div>
            </div>
          </div>
        </div>
        """
    )


def _empty_hero(eyebrow: str, title: str, text: str) -> None:
    html(
        f"""
        <div class="hero empty" style="--sev-light:#B8ADA3;--sev-glow:rgba(184, 173, 163, 0.25);">
          <div>
            <div class="hero-eyebrow"><span>{icon("pin", 13)} {eyebrow}</span></div>
            <div class="hero-title">{title}</div>
            <div class="hero-text">{text}</div>
          </div>
        </div>
        """
    )


def render_town_hero(focus, town_row, area_summary: dict, source_label: str) -> None:
    """
    Banner for town focus: the searched town's own risk for the selected day,
    how much of the area around it is at risk, and the state fire service.

    Args:
        focus: core.focus.Focus for the searched town.
        town_row: the town's forecast row for the selected day (None if missing).
        area_summary: core.summaries.summarise_region() of the area within the radius.
        source_label: where the numbers came from.
    """
    if town_row is None or not area_summary.get("has_data"):
        _empty_hero(
            focus.town,
            f"No forecast near {focus.name}",
            f"There is no forecast area within reach of {focus.town} in the current data. "
            "Try another town, clear the search, or press <b>Refresh</b>.",
        )
        return

    level = str(town_row["risk_level"])
    light, glow = SEVERITY_ON_DARK[level]
    day = pd.Timestamp(area_summary["forecast_date"])
    authority, url = FIRE_AUTHORITIES.get(STATE_NAMES.get(focus.state, ""), NATIONAL_AUTHORITY)
    cells, at_risk = area_summary["cell_count"], area_summary["at_risk_count"]
    share = at_risk / cells if cells else 0
    ffdi = float(town_row["ffdi"])
    fallback = (
        f" {focus.name}'s own area has no forecast, so this uses the nearest forecast "
        f"area, about {int(round(focus.offset_km / 5) * 5)} km {focus.offset_direction}."
        if not focus.own_area else ""
    )

    html(
        f"""
        <div class="hero" style="--sev-light:{light};--sev-glow:{glow};">
          <div>
            <div class="hero-eyebrow">
              <span>{icon("pin", 13)} {focus.town} · within {focus.radius_km} km</span>
              <span>{icon("calendar", 13)} {day.strftime("%A %d %B %Y")}</span>
            </div>
            <div class="hero-title"><span class="level">{level}</span> fire risk in {focus.name}</div>
            <div class="hero-text">
              The chance of fire in {focus.name}'s area is <b>{float(town_row["fire_probability"]):.0%}</b>.
              Within {focus.radius_km} km, <b>{at_risk} of {cells}</b>
              {term("grid_cell", "areas")} ({share:.0%}) are at High or Extreme risk.{fallback}
            </div>
            <div class="hero-advice">
              {icon("alert", 18)}
              <div>{ADVICE[level].replace("this region", "this area")} Official warnings:
                <a href="{url}" target="_blank" rel="noopener">{authority}</a>.</div>
            </div>
            <div class="hero-meta">{icon("database", 13)} Showing {cells} grid cell(s)
              within {focus.radius_km} km of {focus.town} — forecast day {area_summary["forecast_date"]},
              {source_label}.</div>
          </div>
          <div class="hero-gauge">
            <div class="hero-gauge-label">{term("probability", "Fire probability")} in {focus.name}</div>
            <div class="hero-gauge-value">{float(town_row["fire_probability"]) * 100:.0f}<small>%</small></div>
            {_scale(level)}
            <div class="hero-mini">
              <div>Area average<b>{area_summary["mean_probability"]:.0%}</b></div>
              <div>{term("ffdi")} in {focus.name}<b>{ffdi:.1f} · {ffdi_band(ffdi)}</b></div>
            </div>
          </div>
        </div>
        """
    )
