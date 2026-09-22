"""
Styling and theming utilities for Streamlit UI.

Provides consistent styling across all UI components: one place that decides what
"critical" looks like, how finding citations are drawn, and the page's spacing.
"""

import re

import streamlit as st

_SEVERITY_COLOR = {"critical": "red", "warning": "orange", "info": "blue"}
_SEVERITY_ICON = {"critical": "🔴", "warning": "🟠", "info": "🔵"}

# Small, conservative tweaks; nothing here depends on Streamlit internals beyond stable test ids.
_CSS = """
<style>
.block-container { padding-top: 2.2rem; }
[data-testid="stMetricValue"] { font-size: 1.5rem; }
[data-testid="stMetricLabel"] p { font-size: 0.8rem; opacity: 0.75; }
</style>
"""


def apply_theme():
    """Apply the app's page spacing and typography tweaks. Call once per run, after ``st.set_page_config``."""
    st.markdown(_CSS, unsafe_allow_html=True)


def severity_color(severity: str) -> str:
    """
    Get the Streamlit color name for a finding severity level.

    Args:
        severity: Severity level ("info", "warning", "critical")

    Returns:
        A color usable by ``st.badge`` and ``:color[text]`` markdown; unknown severities are gray.
    """
    return _SEVERITY_COLOR.get(severity, "gray")


def severity_icon(severity: str) -> str:
    return _SEVERITY_ICON.get(severity, "⚪")


def severity_badge(severity: str, label: str | None = None):
    """Draw a colored severity badge."""
    st.badge(label or severity, color=severity_color(severity))


def demote_headings(text: str, levels: int = 2) -> str:
    """Push Markdown headings down (## becomes ####) so a report's sections sit below its title."""
    return re.sub(r"^(#{1,6})(?= )", lambda m: "#" * min(len(m.group(1)) + levels, 6), text, flags=re.MULTILINE)


def format_markdown(content: str) -> str:
    """
    Make Manager narrative easier to read in Streamlit.

    Finding citations like ``[#12]`` are drawn as small gray tags so they stay
    visible for verification without cluttering the sentences.
    """
    return re.sub(r"\[#(\d+)\]", lambda m: f":gray[`#{m.group(1)}`]", content)
