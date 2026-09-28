"""
Status banner at the top of the page: the one-glance answer to "how bad is it,
where, and when?", followed by what that means and where official warnings
live.
"""

import pandas as pd

from utils.regions import RISK_ORDER, ffdi_band, is_national
from utils.styles import format_coords, html, icon
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


def _scale(probability: float) -> str:
    """Four equal bands (the model's risk levels are 25% wide) with a marker
    at the peak probability."""
    segments = "".join(
        f'<span style="background:{COLORS[level.lower()]}"></span>' for level in RISK_ORDER
    )
    labels = "".join(f"<span>{level}</span>" for level in RISK_ORDER)
    left = min(max(probability, 0.0), 1.0) * 100
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
        summary: output of utils.regions.summarise_region() for the selected day.
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
              <b>{at_risk} of {cells}</b> monitored grid cells ({share:.0%}) are flagged
              for likely fire activity. The highest-risk cell is near
              <b>{format_coords(lat, lon)}</b>.
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
            <div class="hero-gauge-label">Peak fire probability</div>
            <div class="hero-gauge-value">{summary["max_probability"] * 100:.0f}<small>%</small></div>
            {_scale(summary["max_probability"])}
            <div class="hero-mini">
              <div>Regional average<b>{summary["mean_probability"]:.0%}</b></div>
              <div>Peak FFDI<b>{summary["max_ffdi"]:.1f} · {ffdi_band(summary["max_ffdi"])}</b></div>
            </div>
          </div>
        </div>
        """
    )
