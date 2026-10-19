"""
Page stylesheet loader and small HTML helpers.

The CSS itself lives in frontend/assets/style.css. Components use its class
names rather than inline styles, so the look is controlled from one place;
the colours and fonts it uses are CSS variables generated from utils.theme.
"""

import textwrap
from functools import lru_cache
from pathlib import Path

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


STYLESHEET = Path(__file__).resolve().parents[1] / "assets" / "style.css"


def theme_variables() -> str:
    """The theme as CSS custom properties: COLORS["text_dim"] -> --c-text-dim."""
    lines = [f"  --c-{name.replace('_', '-')}: {value};" for name, value in C.items()]
    lines += [f"  --font-body: {FONT_BODY};", f"  --font-display: {FONT_DISPLAY};"]
    return ":root {\n" + "\n".join(lines) + "\n}\n"


@lru_cache(maxsize=1)
def global_css() -> str:
    """The full <style> block: theme variables, then assets/style.css."""
    return f"<style>\n{theme_variables()}{STYLESHEET.read_text(encoding='utf-8')}</style>"


def inject_global_css() -> None:
    """Adds the page stylesheet. Call once, near the top of main.py."""
    st.html(global_css())
