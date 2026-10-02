"""
Page-wide stylesheet and small HTML helpers.

Every custom-HTML component uses the class names defined in GLOBAL_CSS rather
than inline styles, so the look is controlled from one place. Streamlit
containers created with `key="card-..."` pick up the card style through the
`st-key-<key>` class Streamlit adds to keyed containers.
"""

import textwrap

import streamlit as st

from utils.theme import COLORS as C
from utils.theme import FONT_BODY, FONT_DISPLAY

# Lucide icon paths (https://lucide.dev, ISC licence), drawn with currentColor
# so an icon takes the colour of the text around it.
_ICON_PATHS = {
    "flame": '<path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.072-2.143-.224-4.054 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.153.433-2.294 1-3a2.5 2.5 0 0 0 2.5 2.5z"/>',
    "thermometer": '<path d="M14 4v10.54a4 4 0 1 1-4 0V4a2 2 0 0 1 4 0Z"/>',
    "droplet": '<path d="M12 22a7 7 0 0 0 7-7c0-2-1-3.9-3-5.5s-3.5-4-4-6.5c-.5 2.5-2 4.9-4 6.5C6 11.1 5 13 5 15a7 7 0 0 0 7 7z"/>',
    "wind": '<path d="M17.7 7.7a2.5 2.5 0 1 1 1.8 4.3H2"/><path d="M9.6 4.6A2 2 0 1 1 11 8H2"/><path d="M12.6 19.4A2 2 0 1 0 14 16H2"/>',
    "pin": '<path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z"/><circle cx="12" cy="10" r="3"/>',
    "alert": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
    "gauge": '<path d="m12 14 4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/>',
    "grid": '<rect width="18" height="18" x="3" y="3" rx="2"/><path d="M3 9h18"/><path d="M3 15h18"/><path d="M9 3v18"/><path d="M15 3v18"/>',
    "info": '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>',
    "calendar": '<rect width="18" height="18" x="3" y="4" rx="2"/><path d="M16 2v4"/><path d="M8 2v4"/><path d="M3 10h18"/>',
    "external": '<path d="M15 3h6v6"/><path d="M10 14 21 3"/><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>',
    "sun": '<circle cx="12" cy="12" r="4"/><path d="M12 2v2"/><path d="M12 20v2"/><path d="m4.93 4.93 1.41 1.41"/><path d="m17.66 17.66 1.41 1.41"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="m6.34 17.66-1.41 1.41"/><path d="m19.07 4.93-1.41 1.41"/>',
    "trend_up": '<polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/>',
    "trend_down": '<polyline points="22 17 13.5 8.5 8.5 13.5 2 7"/><polyline points="16 17 22 17 22 11"/>',
    "minus": '<path d="M5 12h14"/>',
    "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5V19A9 3 0 0 0 21 19V5"/><path d="M3 12A9 3 0 0 0 21 12"/>',
}


def icon(name: str, size: int = 16) -> str:
    """Inline SVG icon markup."""
    return (
        f'<svg class="icon" width="{size}" height="{size}" viewBox="0 0 24 24" '
        'fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" '
        f'stroke-linejoin="round" aria-hidden="true">{_ICON_PATHS[name]}</svg>'
    )


def html(markup: str) -> None:
    """
    Renders an HTML snippet through st.markdown.

    Lines are stripped and blank lines dropped first: Markdown treats a line
    indented by four spaces as a code block and a blank line as the end of an
    HTML block, so indented f-strings would otherwise leak raw tags onto the
    page.
    """
    lines = (line.strip() for line in textwrap.dedent(markup).splitlines())
    # Joined with a space, not "", so words split across source lines stay
    # separate ("flagged" / "for" -> "flagged for").
    st.markdown(" ".join(line for line in lines if line), unsafe_allow_html=True)


def section_header(eyebrow: str, title: str, subtitle: str = "") -> None:
    sub = f'<p class="section-sub">{subtitle}</p>' if subtitle else ""
    html(
        f"""
        <div class="section-head">
          <div class="eyebrow">{eyebrow}</div>
          <h2 class="section-title">{title}</h2>
          {sub}
        </div>
        """
    )


def format_coords(lat: float, lon: float) -> str:
    """-33.5, 151.0 -> '33.5°S, 151.0°E'."""
    ns = "S" if lat < 0 else "N"
    ew = "W" if lon < 0 else "E"
    return f"{abs(lat):.1f}°{ns}, {abs(lon):.1f}°{ew}"


GLOBAL_CSS = f"""
<style>
/* ---------- page ---------- */
.stApp {{
  background:
    radial-gradient(1200px 400px at 85% -120px, rgba(217, 72, 15, 0.07), transparent 70%),
    {C["bg"]};
}}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stMainBlockContainer"] {{
  max-width: 1380px;
  padding-top: 1.6rem;
  padding-bottom: 3rem;
}}
[data-testid="stHeaderActionElements"] {{ display: none; }}
[data-testid="stWidgetLabel"] p {{
  font-size: 0.7rem;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: {C["text_dim"]};
}}
.icon {{ display: inline-block; vertical-align: -0.18em; flex-shrink: 0; }}

/* ---------- cards: any container keyed "card-..." ---------- */
[class*="st-key-card"] {{
  background: {C["surface"]};
  border: 1px solid {C["border"]};
  border-radius: 16px;
  padding: 18px 20px 12px;
  box-shadow: 0 1px 2px rgba(28, 25, 23, 0.04), 0 8px 24px -12px rgba(28, 25, 23, 0.10);
}}

/* ---------- top bar ---------- */
.st-key-topbar {{ margin-bottom: 0.4rem; }}
.brand {{ display: flex; align-items: center; gap: 14px; }}
.brand-mark {{
  width: 46px; height: 46px; border-radius: 13px;
  display: grid; place-items: center; color: #fff;
  background: linear-gradient(145deg, #F76707 0%, {C["ember"]} 45%, {C["extreme"]} 100%);
  box-shadow: 0 8px 18px -8px rgba(217, 72, 15, 0.7);
}}
.brand-title {{
  font-family: {FONT_DISPLAY};
  font-size: 1.55rem !important; font-weight: 700 !important; line-height: 1.15 !important;
  color: {C["text"]}; letter-spacing: -0.02em; white-space: nowrap;
}}
.brand-sub {{ font-size: 0.82rem; color: {C["text_muted"]}; margin-top: 3px; }}

/* ---------- hero status banner ---------- */
.hero {{
  position: relative;
  border-radius: 22px;
  padding: 28px 32px;
  color: {C["ink_text"]};
  background:
    radial-gradient(90% 140% at 100% 0%, var(--sev-glow), transparent 60%),
    linear-gradient(135deg, {C["ink"]} 0%, {C["ink_2"]} 100%);
  box-shadow: 0 18px 40px -22px rgba(27, 23, 20, 0.65);
  display: grid; grid-template-columns: minmax(0, 1.6fr) minmax(260px, 1fr);
  gap: 32px; align-items: center;
  margin: 0.4rem 0 1.4rem;
}}
.hero-eyebrow {{
  display: flex; flex-wrap: wrap; gap: 8px 14px; align-items: center;
  font-size: 0.72rem; font-weight: 600; letter-spacing: 0.08em;
  text-transform: uppercase; color: {C["ink_muted"]};
}}
.hero-eyebrow span {{ display: inline-flex; align-items: center; gap: 6px; }}
.hero-title {{
  font-family: {FONT_DISPLAY};
  font-size: 2.35rem; font-weight: 700; letter-spacing: -0.025em;
  line-height: 1.1; margin: 12px 0 10px; color: #fff;
}}
.hero-title .level {{ color: var(--sev-light); }}
.hero-text {{ font-size: 0.98rem; line-height: 1.6; color: {C["ink_text"]}; max-width: 62ch; }}
.hero-text b {{ color: #fff; }}
.hero-advice {{
  display: flex; gap: 10px; align-items: flex-start;
  margin-top: 16px; padding: 12px 14px;
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.06);
  border: 1px solid rgba(255, 255, 255, 0.10);
  font-size: 0.88rem; line-height: 1.5; color: {C["ink_text"]};
}}
.hero-advice .icon {{ color: var(--sev-light); margin-top: 2px; }}
.hero-advice a {{ color: #fff; font-weight: 600; text-decoration: underline; text-underline-offset: 3px; }}
.hero-meta {{
  display: flex; gap: 6px; align-items: center;
  margin-top: 14px; font-size: 0.76rem; color: {C["ink_muted"]};
}}
.hero-gauge {{
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.10);
  border-radius: 18px; padding: 20px 22px;
}}
.hero-gauge-label {{ font-size: 0.75rem; color: {C["ink_muted"]}; text-transform: uppercase; letter-spacing: 0.07em; font-weight: 600; }}
.hero-gauge-value {{
  font-family: {FONT_DISPLAY};
  font-size: 3.4rem; font-weight: 700; line-height: 1; margin: 8px 0 4px;
  color: #fff; letter-spacing: -0.03em;
}}
.hero-gauge-value small {{ font-size: 1.4rem; color: {C["ink_muted"]}; margin-left: 2px; }}
.scale {{ position: relative; margin: 18px 0 6px; }}
.scale-track {{ display: flex; gap: 3px; height: 10px; }}
.scale-track span {{ flex: 1; border-radius: 3px; opacity: 0.9; }}
.scale-marker {{
  position: absolute; top: -5px; width: 4px; height: 20px; border-radius: 2px;
  background: #fff; box-shadow: 0 0 0 3px rgba(27, 23, 20, 0.85);
  transform: translateX(-50%);
}}
.scale-labels {{ display: flex; font-size: 0.68rem; color: {C["ink_muted"]}; margin-top: 8px; }}
.scale-labels span {{ flex: 1; text-align: center; }}
.hero-mini {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 16px; }}
.hero-mini div {{ font-size: 0.75rem; color: {C["ink_muted"]}; }}
.hero-mini b {{ display: block; font-size: 1.05rem; color: #fff; font-weight: 600; margin-top: 2px; }}
.hero.empty {{ grid-template-columns: 1fr; }}

/* ---------- pills ---------- */
.pill {{
  display: inline-flex; align-items: center; gap: 6px;
  padding: 3px 10px; border-radius: 999px;
  font-size: 0.74rem; font-weight: 600; white-space: nowrap;
}}
.pill .dot {{ width: 7px; height: 7px; border-radius: 50%; background: currentColor; }}

/* ---------- day strip ---------- */
.day-strip-head {{
  display: flex; justify-content: space-between; align-items: baseline;
  margin: 0 0 8px;
}}
.day-strip-head .title {{ font-weight: 600; font-size: 0.95rem; color: {C["text"]}; }}
.day-strip-head .hint {{ font-size: 0.8rem; color: {C["text_dim"]}; }}
.st-key-daystrip [data-testid="stColumn"] {{ min-width: 0; }}
.st-key-daystrip button {{
  border-radius: 12px 12px 0 0 !important;
  border-bottom: none !important;
  min-height: 2.1rem; font-weight: 600;
}}
.day-tile {{
  border: 1px solid {C["border"]}; border-top: 3px solid var(--sev);
  border-radius: 0 0 12px 12px;
  background: {C["surface"]};
  padding: 10px 8px 12px; text-align: center;
  transition: box-shadow .15s ease, transform .15s ease;
}}
.day-tile.selected {{
  background: var(--sev-bg);
  box-shadow: 0 10px 22px -14px var(--sev);
}}
.day-tile .date {{ font-size: 0.72rem; color: {C["text_dim"]}; }}
.day-tile .prob {{
  font-family: {FONT_DISPLAY}; font-size: 1.55rem; font-weight: 700;
  color: var(--sev); line-height: 1.15; margin: 2px 0;
}}
.day-tile .lvl {{ font-size: 0.74rem; font-weight: 600; color: var(--sev); }}
.day-tile .meta {{ font-size: 0.7rem; color: {C["text_muted"]}; margin-top: 6px; }}
.day-placeholder {{
  border: 1px dashed {C["border"]}; border-radius: 12px;
  height: 100%; min-height: 150px;
  display: grid; place-items: center; text-align: center;
  font-size: 0.72rem; color: {C["text_dim"]}; padding: 8px;
  background: rgba(255, 255, 255, 0.4);
}}

/* ---------- stat tiles ---------- */
.stat-tile {{
  background: {C["surface"]};
  border: 1px solid {C["border"]};
  border-radius: 16px;
  padding: 16px 18px;
  height: 100%;
  box-shadow: 0 1px 2px rgba(28, 25, 23, 0.04);
}}
.stat-head {{ display: flex; align-items: center; gap: 10px; }}
.stat-icon {{
  width: 32px; height: 32px; border-radius: 9px;
  display: grid; place-items: center;
  color: var(--accent); background: var(--accent-bg);
}}
.stat-label {{ font-size: 0.8rem; font-weight: 600; color: {C["text_muted"]}; }}
.stat-value {{
  font-family: {FONT_DISPLAY};
  font-size: 2rem; font-weight: 700; letter-spacing: -0.02em;
  color: {C["text"]}; margin-top: 12px; line-height: 1.1;
}}
.stat-value small {{ font-size: 1rem; color: {C["text_dim"]}; font-weight: 500; }}
.stat-context {{ font-size: 0.8rem; color: {C["text_muted"]}; margin-top: 6px; line-height: 1.45; }}
.bar {{ height: 6px; border-radius: 999px; background: {C["bg"]}; margin-top: 12px; overflow: hidden; }}
.bar span {{ display: block; height: 100%; border-radius: 999px; background: var(--accent); }}
.conditions {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-top: 12px; }}
.conditions div {{ font-size: 0.72rem; color: {C["text_dim"]}; }}
.conditions b {{
  display: block; font-family: {FONT_DISPLAY};
  font-size: 1.3rem; color: {C["text"]}; font-weight: 700; margin-top: 2px;
}}
.st-key-stats {{ margin-top: 1.1rem; }}
.st-key-stats [data-testid="stColumn"] > div {{ height: 100%; }}

/* ---------- section headings ---------- */
.section-head {{ margin: 2rem 0 0.9rem; }}
.eyebrow {{
  font-size: 0.72rem; font-weight: 700; letter-spacing: 0.1em;
  text-transform: uppercase; color: {C["ember"]};
}}
.section-title {{
  font-family: {FONT_DISPLAY};
  font-size: 1.55rem !important; font-weight: 700 !important;
  letter-spacing: -0.02em; margin: 4px 0 0 !important; padding: 0 !important;
  color: {C["text"]};
}}
.section-sub {{ font-size: 0.9rem; color: {C["text_muted"]}; margin: 4px 0 0; }}

/* ---------- chart cards ---------- */
.chart-title {{ font-weight: 600; font-size: 1rem; color: {C["text"]}; }}
.chart-sub {{ font-size: 0.8rem; color: {C["text_dim"]}; margin: 2px 0 6px; }}
.donut-facts {{ display: grid; gap: 10px; margin-top: 24px; }}
.donut-facts div {{ font-size: 0.75rem; color: {C["text_dim"]}; }}
.donut-facts b {{ display: block; font-size: 1.05rem; color: {C["text"]}; }}

/* ---------- map card ---------- */
.st-key-card-map iframe {{ border-radius: 12px; }}
.legend {{
  display: flex; flex-wrap: wrap; align-items: center; gap: 8px 18px;
  font-size: 0.78rem; color: {C["text_muted"]};
  padding: 10px 2px 4px;
}}
.legend .sw {{ width: 12px; height: 12px; border-radius: 50%; display: inline-block; margin-right: 6px; vertical-align: -1px; }}
.legend .note {{ margin-left: auto; color: {C["text_dim"]}; }}

/* ---------- threat panel ---------- */
.panel-title {{ font-weight: 600; font-size: 1rem; color: {C["text"]}; margin-bottom: 12px; }}
.panel-row {{
  display: flex; justify-content: space-between; align-items: baseline; gap: 10px;
  padding: 8px 0; border-bottom: 1px solid {C["border"]};
  font-size: 0.82rem; color: {C["text_muted"]};
}}
.panel-row b {{ color: {C["text"]}; font-weight: 600; text-align: right; }}
.panel-row small {{ color: {C["text_dim"]}; font-weight: 400; }}
.mix-label {{ font-size: 0.78rem; font-weight: 600; color: {C["text_muted"]}; margin: 14px 0 6px; }}
.mix-bar {{ display: flex; height: 10px; border-radius: 999px; overflow: hidden; gap: 2px; background: {C["bg"]}; }}
.mix-key {{ display: grid; grid-template-columns: 1fr 1fr; gap: 4px 12px; margin-top: 8px; font-size: 0.74rem; color: {C["text_muted"]}; }}
.mix-key span b {{ color: {C["text"]}; }}
.cell-list {{ margin-top: 6px; }}
.cell-row {{
  display: flex; align-items: center; justify-content: space-between; gap: 8px;
  padding: 7px 0; font-size: 0.8rem; color: {C["text_muted"]};
  border-bottom: 1px dashed {C["border"]};
}}
.cell-row:last-child {{ border-bottom: none; }}
.cell-row .rank {{
  width: 20px; height: 20px; border-radius: 6px; display: inline-grid; place-items: center;
  font-size: 0.68rem; font-weight: 700; background: {C["bg"]}; color: {C["text_muted"]};
  margin-right: 8px;
}}
.cell-row .prob {{ font-weight: 700; color: {C["text"]}; }}
.callout {{
  display: flex; gap: 8px; align-items: flex-start;
  margin-top: 12px; padding: 10px 12px; border-radius: 10px;
  font-size: 0.76rem; line-height: 1.45;
  background: {C["moderate_bg"]}; color: {C["text"]};
}}
.callout .icon {{ color: {C["moderate"]}; margin-top: 1px; }}
.empty-note {{ font-size: 0.85rem; color: {C["text_muted"]}; line-height: 1.5; }}

/* ---------- glossary tooltips: utils.glossary.term() ---------- */
.term {{
  position: relative; cursor: help;
  text-decoration: underline dotted; text-underline-offset: 3px;
  text-decoration-color: currentColor; text-decoration-thickness: 1px;
}}
.term:focus {{ outline: 2px solid {C["ember"]}; outline-offset: 2px; border-radius: 3px; }}
.term::after {{
  content: attr(data-tip);
  position: absolute; left: 50%; bottom: calc(100% + 8px);
  transform: translateX(-50%);
  width: max-content; max-width: 260px;
  padding: 9px 11px; border-radius: 9px;
  background: {C["ink"]}; color: {C["ink_text"]};
  font-size: 0.74rem; font-weight: 400; line-height: 1.45;
  letter-spacing: 0; text-transform: none; text-align: left; white-space: normal;
  box-shadow: 0 10px 24px -10px rgba(0, 0, 0, 0.5);
  opacity: 0; visibility: hidden; pointer-events: none;
  transition: opacity .12s ease; z-index: 1000;
}}
.term:hover::after, .term:focus::after {{ opacity: 1; visibility: visible; }}

/* ---------- best / worst day line ---------- */
.week-line {{
  display: flex; flex-wrap: wrap; gap: 8px 22px;
  margin: 0 0 10px; font-size: 0.85rem; color: {C["text_muted"]};
}}
.week-line span {{ display: inline-flex; align-items: center; gap: 7px; }}
.week-line b {{ color: {C["text"]}; }}
.week-line .up {{ color: {C["extreme"]}; }}
.week-line .down {{ color: {C["low"]}; }}

/* ---------- "why is the risk high here" ---------- */
.why-list {{ display: grid; gap: 7px; margin-top: 4px; }}
.why-item {{ display: flex; gap: 8px; align-items: flex-start; font-size: 0.8rem; color: {C["text"]}; line-height: 1.4; }}
.why-item .icon {{ margin-top: 1px; color: var(--c); }}
.why-calm {{ font-size: 0.76rem; color: {C["text_muted"]}; margin-top: 8px; line-height: 1.45; }}
.why-note {{ font-size: 0.76rem; color: {C["text_muted"]}; margin-top: 8px; line-height: 1.45; font-style: italic; }}
.place {{ display: block; font-size: 0.72rem; color: {C["text_dim"]}; font-weight: 400; }}

/* ---------- medium screens: wrap rows before they get cramped ---------- */
.st-key-topbar button p {{ white-space: nowrap; }}
@media (max-width: 1100px) {{
  .st-key-topbar [data-testid="stHorizontalBlock"],
  .st-key-stats [data-testid="stHorizontalBlock"],
  .st-key-maprow [data-testid="stHorizontalBlock"] {{ flex-wrap: wrap; }}
  .st-key-topbar [data-testid="stColumn"]:first-child,
  .st-key-maprow [data-testid="stColumn"] {{
    flex: 1 1 100% !important; min-width: 100% !important;
  }}
  .st-key-stats [data-testid="stColumn"] {{
    flex: 1 1 calc(50% - 1rem) !important; min-width: calc(50% - 1rem) !important;
  }}
}}

/* ---------- small screens ---------- */
@media (max-width: 900px) {{
  .hero {{ grid-template-columns: 1fr; padding: 22px; }}
  .hero-title {{ font-size: 1.8rem; }}
}}
</style>
"""


def inject_global_css() -> None:
    """Adds the page stylesheet. Call once, near the top of main.py."""
    st.html(GLOBAL_CSS)
