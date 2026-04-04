from __future__ import annotations

import json
import math
from html import escape
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from src.config import ProjectPaths

ROOT = Path(__file__).resolve().parent
PATHS = ProjectPaths(ROOT)

TOTAL_HOUSE_SEATS = 435
MAJORITY_SEATS = 218

st.set_page_config(page_title="2026 House Forecast", page_icon="\U0001f5f3\ufe0f", layout="wide")

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
DARK = st.session_state.dark_mode

if DARK:
    DEM_COLOR = "#4A90D9"
    GOP_COLOR = "#E05C4F"
    BG = "#0C0C0F"
    INK = "#DDDDE0"
    MUTED = "#888890"
    SUBTLE = "#555560"
    RULE = "#222228"
    AMBER = "#FFD166"
    TOOLTIP_BG = "#1A1A1F"
    TOOLTIP_BORDER = "#333338"
else:
    DEM_COLOR = "#2B5F91"
    GOP_COLOR = "#BF3B33"
    BG = "#F7F6F3"
    INK = "#1A1A18"
    MUTED = "#6B6860"
    SUBTLE = "#8A8780"
    RULE = "#DDD9D0"
    AMBER = "#C4940A"
    TOOLTIP_BG = "#FFFFFF"
    TOOLTIP_BORDER = "#D8D4CC"

CHART_GRID = "rgba(255,255,255,0.04)" if DARK else "rgba(0,0,0,0.06)"
st.markdown(
    '<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">',
    unsafe_allow_html=True,
)
st.markdown(
    f"""
    <style>
    /* ── theme foundation ── */
    .stApp, .stApp > div, [data-testid="stAppViewContainer"],
    [data-testid="stBottom"], [data-testid="stBottomBlockContainer"],
    .stMainBlockContainer, [data-testid="stMainBlockContainer"] {{
        background: {BG} !important;
        color: {INK} !important;
    }}
    [data-testid="stSidebar"] {{ display: none !important; }}

    /* ── theme button ── */
    button[data-testid="stBaseButton-secondary"] {{
        background: {"rgba(255,255,255,0.06)" if DARK else "rgba(0,0,0,0.04)"} !important;
        border: 1px solid {RULE} !important;
        border-radius: 4px !important;
        color: {INK} !important;
        font-size: 0.62rem !important;
        font-weight: 600 !important;
        padding: 0.25rem 0.6rem !important;
        min-height: auto !important;
        height: auto !important;
    }}
    /* ── theme toggle button ── */
    button[data-testid="stBaseButton-secondary"] {{
        background: {"rgba(255,255,255,0.06)" if DARK else "rgba(0,0,0,0.04)"} !important;
        border: 1px solid {RULE} !important;
        border-radius: 4px !important;
        color: {MUTED} !important;
        font-size: 0.68rem !important;
        font-weight: 600 !important;
        padding: 0.25rem 0.7rem !important;
        min-height: auto !important;
        height: auto !important;
        letter-spacing: 0.04em !important;
    }}
    button[data-testid="stBaseButton-secondary"]:hover {{
        background: {"rgba(255,255,255,0.12)" if DARK else "rgba(0,0,0,0.08)"} !important;
        color: {INK} !important;
    }}
    [data-testid="stHeader"] {{
        background: {"rgba(12,12,15,0.94)" if DARK else "rgba(255,255,255,0.94)"} !important;
        backdrop-filter: blur(8px);
        border-bottom: 1px solid {RULE};
    }}
    [data-testid="stToolbar"] {{ display: none !important; }}
    [data-testid="stDecoration"] {{ display: none !important; }}
    [data-testid="stStatusWidget"] {{ display: none !important; }}
    .block-container {{
        max-width: 1060px;
        margin: 0 auto;
        padding: 3.2rem 1.5rem 2rem;
    }}
    [data-testid="stVerticalBlock"] {{ gap: 0.15rem !important; }}

    /* ── typography ── */
    *, h1, h2, h3, h4, p, li, label, span,
    div[data-testid="stCaptionContainer"],
    div[data-testid="stExpander"] summary span {{
        font-family: 'Space Grotesk', -apple-system, sans-serif !important;
    }}
    h1, h2, h3, h4 {{ color: {INK} !important; }}
    p, li, label, div[data-testid="stCaptionContainer"] {{
        color: {MUTED} !important;
    }}
    a {{ color: {DEM_COLOR} !important; }}

    /* ── kill Streamlit chrome ── */
    iframe {{
        display: block;
        margin: 0 auto;
        border: none !important;
        background: {BG} !important;
    }}
    .stPlotlyChart > div {{
        background: transparent !important;
    }}
    [data-testid="stMetricValue"],
    [data-testid="stCaptionContainer"] p {{
        color: {MUTED} !important;
    }}
    .stRadio > label {{ display: none !important; }}
    div[data-testid="stRadio"] > div > div {{
        background: transparent !important;
    }}

    .intro {{
        border-bottom: 1px solid {RULE};
        padding: 0.4rem 0 0.9rem;
        margin-bottom: 0.3rem;
    }}
    .intro-kicker {{
        text-transform: uppercase;
        letter-spacing: 0.18em;
        font-size: 0.55rem;
        font-weight: 700;
        color: {MUTED} !important;
        margin: 0 0 0.35rem;
    }}
    .intro-title {{
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 1.35rem;
        font-weight: 700;
        line-height: 1.15;
        color: {INK} !important;
        margin: 0 0 0.7rem;
        letter-spacing: -0.02em;
    }}
    .intro-title em {{
        font-style: normal;
    }}
    .intro-stats {{
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 0;
    }}
    .intro-stat {{
        display: flex;
        flex-direction: column;
        padding: 0.4rem 0;
        border-left: 1px solid {RULE};
        padding-left: 0.7rem;
    }}
    .intro-stat:first-child {{
        border-left: none;
        padding-left: 0;
    }}
    .intro-stat-val {{
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.1rem;
        font-weight: 700;
        line-height: 1.1;
    }}
    .intro-stat-label {{
        font-size: 0.55rem;
        font-weight: 500;
        color: {MUTED} !important;
        margin-top: 0.1rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }}
    .intro-updated {{
        font-size: 0.58rem;
        color: {SUBTLE} !important;
        margin-top: 0.6rem;
        padding-top: 0.8rem;
        border-top: 1px solid {RULE};
    }}

    .rule {{
        height: 1px;
        background: {RULE};
        margin: 1.2rem 0;
    }}
    .sec-label {{
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-size: 0.58rem;
        font-weight: 700;
        color: {SUBTLE};
        margin: 0 0 0.4rem;
    }}
    .sec-note {{
        font-size: 0.72rem;
        color: {SUBTLE};
        margin: -0.15rem 0 0.4rem;
        font-weight: 400;
    }}

    /* ── seat strip ── */
    .strip-wrap {{
        position: relative;
        margin: 0.4rem 0 0.6rem;
    }}
    .strip {{
        display: flex;
        height: 20px;
        border-radius: 3px;
        overflow: hidden;
        box-shadow: 0 0 12px rgba(74,144,217,0.08), 0 0 12px rgba(224,92,79,0.08);
    }}
    .strip > div {{ flex: 1; }}

    /* ── segmented seat bar ── */
    .segbar-wrap {{
        position: relative;
        margin: 0.4rem 0 0;
    }}
    .segbar {{
        display: flex;
        height: 32px;
        border-radius: 4px;
        overflow: hidden;
    }}
    .segbar-seg {{
        display: flex;
        align-items: center;
        justify-content: center;
        position: relative;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.68rem;
        font-weight: 700;
        color: white;
        min-width: 0;
    }}
    .segbar-seg + .segbar-seg {{
        border-left: 2px solid {BG};
    }}
    .segbar-labels {{
        display: flex;
        margin-top: 0.3rem;
    }}
    .segbar-lbl {{
        display: flex;
        flex-direction: column;
        align-items: center;
        font-family: 'Space Grotesk', sans-serif !important;
        min-width: 0;
        overflow: hidden;
    }}
    .segbar-lbl-name {{
        font-size: 0.52rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: {MUTED} !important;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: 100%;
    }}
    .segbar-marker {{
        position: absolute;
        top: -6px;
        bottom: -28px;
        width: 2px;
        background: {AMBER};
        z-index: 2;
    }}
    .segbar-marker::before {{
        content: '';
        position: absolute;
        top: 0; left: -3px;
        width: 8px; height: 8px;
        background: {AMBER};
        border-radius: 50%;
    }}
    .segbar-marker::after {{
        content: '218';
        position: absolute;
        bottom: -14px; left: 50%;
        transform: translateX(-50%);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.56rem;
        font-weight: 700;
        color: {AMBER};
        white-space: nowrap;
    }}
    .segbar-totals {{
        display: flex;
        justify-content: space-between;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.68rem;
        font-weight: 600;
        margin-top: 1.5rem;
    }}
    .segbar-bracket {{
        display: flex;
        align-items: center;
        gap: 0.3rem;
    }}
    .segbar-bracket-line {{
        height: 1px;
        flex: 1;
        background: {RULE};
    }}

    /* ── radio toggle ── */
    div[data-testid="stRadio"] {{
        margin-bottom: 0.6rem !important;
    }}
    div[data-testid="stRadio"] > div {{
        flex-direction: row !important;
        gap: 3px !important;
        flex-wrap: wrap !important;
    }}
    div[data-testid="stRadio"] > div > label {{
        background: transparent !important;
        border: 1px solid {TOOLTIP_BORDER} !important;
        border-radius: 4px !important;
        padding: 0.28rem 0.6rem !important;
        font-size: 0.62rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.03em !important;
        text-transform: uppercase !important;
        color: {INK} !important;
        cursor: pointer !important;
        transition: all 120ms ease !important;
        margin: 0 !important;
        white-space: nowrap !important;
    }}
    div[data-testid="stRadio"] > div > label:hover {{
        border-color: {INK} !important;
        background: {"rgba(255,255,255,0.06)" if DARK else "rgba(0,0,0,0.04)"} !important;
    }}
    div[data-testid="stRadio"] > div > label[data-checked="true"],
    div[data-testid="stRadio"] > div > label:has(input:checked) {{
        background: {"rgba(255,255,255,0.08)" if DARK else "rgba(0,0,0,0.07)"} !important;
        color: {INK} !important;
        border-color: {INK} !important;
    }}
    div[data-testid="stRadio"] > div > label p,
    div[data-testid="stRadio"] > div > label span {{
        color: {INK} !important;
    }}
    div[data-testid="stRadio"] input[type="radio"] {{
        display: none !important;
    }}
    div[data-testid="stRadio"] > label {{
        display: none !important;
    }}

    /* ── text input ── */
    div[data-testid="stTextInput"] > label {{ display: none !important; }}
    div[data-testid="stTextInput"] input {{
        background: {"#18181B" if DARK else "#FFF"} !important;
        border: 1px solid {"#333" if DARK else "#CCC"} !important;
        border-radius: 5px !important;
        color: {INK} !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 0.75rem !important;
        font-weight: 500 !important;
        padding: 0.45rem 0.8rem !important;
    }}
    div[data-testid="stTextInput"] input:focus {{
        border-color: {"#666" if DARK else "#888"} !important;
        outline: none !important;
    }}
    div[data-testid="stTextInput"] input::placeholder {{
        color: {MUTED} !important;
    }}

    /* ── selectbox — nuclear override ── */
    div[data-testid="stSelectbox"] {{ overflow: visible !important; position: relative; z-index: 10; }}
    div[data-testid="stSelectbox"] > label {{ display: none !important; }}
    div[data-testid="stSelectbox"] div[data-baseweb="select"],
    div[data-testid="stSelectbox"] div[data-baseweb="select"] div,
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div > div {{
        background-color: {"#18181B" if DARK else "#FFF"} !important;
        background: {"#18181B" if DARK else "#FFF"} !important;
        border-color: {"#333" if DARK else "#CCC"} !important;
        color: {INK} !important;
    }}
    div[data-testid="stSelectbox"] * {{
        color: {INK} !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 0.75rem !important;
        border-color: {"#333" if DARK else "#CCC"} !important;
    }}
    div[data-testid="stSelectbox"] svg {{
        fill: {MUTED} !important;
    }}
    /* popover dropdown */
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] > div,
    div[data-baseweb="popover"] ul,
    ul[role="listbox"] {{
        background-color: {"#1A1A1F" if DARK else "#FFF"} !important;
        background: {"#1A1A1F" if DARK else "#FFF"} !important;
        border-color: {"#333" if DARK else "#DDD"} !important;
    }}
    li[role="option"] {{
        color: {INK} !important;
        background-color: transparent !important;
        background: transparent !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-size: 0.72rem !important;
    }}
    li[role="option"]:hover,
    li[role="option"][aria-selected="true"] {{
        background-color: {"#262630" if DARK else "#F0F0F0"} !important;
        background: {"#262630" if DARK else "#F0F0F0"} !important;
    }}

    /* ── expander ── */
    div[data-testid="stExpander"] {{
        border: 1px solid {"#333" if DARK else "#DDD"} !important;
        border-radius: 5px !important;
        background: {"#18181B" if DARK else "#FAFAFA"} !important;
    }}
    div[data-testid="stExpander"] summary {{
        padding: 0.5rem 0.8rem !important;
        font-size: 0.68rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
        color: {INK} !important;
    }}
    div[data-testid="stExpander"] summary svg {{
        fill: {MUTED} !important;
    }}
    div[data-testid="stExpander"] [data-testid="stExpanderDetails"] {{
        border-top: 1px solid {"#333" if DARK else "#DDD"} !important;
        padding: 0.4rem 0.8rem 0.6rem !important;
    }}

    .foot {{
        font-size: 0.58rem;
        color: {SUBTLE} !important;
        font-weight: 400;
        margin-top: 0.3rem;
        padding-top: 0.5rem;
        border-top: 1px solid {RULE};
    }}

    /* ── responsive ── */

    /* ── race table ── */
    .race-row {{
        display: grid;
        grid-template-columns: 46px 1fr 60px 130px;
        gap: 0.5rem;
        align-items: center;
        padding: 0.45rem 0;
        border-bottom: 1px solid {RULE};
    }}

    /* ── responsive ── */
    @media (max-width: 1024px) {{
        .block-container {{ padding: 2.5rem 1.25rem 1.5rem; }}
    }}
    @media (max-width: 768px) {{
        .block-container {{ padding: 2.5rem 1rem 1.5rem; }}
        .intro-title {{ font-size: 1.15rem; }}
        .intro-stat-val {{ font-size: 1rem; }}
        .intro-stats {{ grid-template-columns: repeat(3, 1fr); }}
        .intro-stat:nth-child(4) {{ border-left: none; padding-left: 0; }}
        .race-row {{ grid-template-columns: 40px 1fr 50px 100px; gap: 0.35rem; }}
    }}
    @media (max-width: 480px) {{
        .block-container {{ padding: 2rem 0.75rem 1.5rem; }}
        .intro-title {{ font-size: 1.05rem; }}
        .intro-kicker {{ font-size: 0.48rem; }}
        .intro-stat-val {{ font-size: 0.9rem; }}
        .intro-stat-label {{ font-size: 0.48rem; }}
        .intro-stats {{ grid-template-columns: repeat(2, 1fr); gap: 0.4rem 0; }}
        .intro-stat {{ padding: 0.3rem 0; }}
        .intro-stat:nth-child(n+3) {{ border-left: none; padding-left: 0; }}
        .intro-stat:nth-child(even) {{ border-left: 1px solid {RULE}; padding-left: 0.7rem; }}
        .intro-stat:nth-child(5) {{ grid-column: 1 / -1; border-left: none; padding-left: 0; }}
        .intro-updated {{ font-size: 0.5rem; }}
        .sec-label {{ font-size: 0.52rem; }}
        .sec-note {{ font-size: 0.62rem; }}
        .segbar-seg {{ font-size: 0.55rem; }}
        .segbar-totals {{ font-size: 0.58rem; }}
        .foot {{ font-size: 0.5rem; }}
        .race-row {{ grid-template-columns: 40px 1fr 100px; }}
        .race-row > div:nth-child(3) {{ display: none; }}
        div[data-testid="stRadio"] > div {{ flex-wrap: wrap !important; }}
        div[data-testid="stRadio"] > div > label {{ font-size: 0.58rem !important; padding: 0.25rem 0.55rem !important; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def _prob_text(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _margin_text(value: float) -> str:
    if pd.isna(value):
        return "\u2014"
    return f"{'D' if float(value) >= 0 else 'R'}+{abs(float(value)):.1f}"


TOSSUP_COLOR = "#9B7FBF" if DARK else "#8B6DAF"


def _margin_fill(margin: float) -> str:
    m = abs(float(margin))
    if m < 1.5:
        return TOSSUP_COLOR
    if DARK:
        if float(margin) >= 0:
            if m > 10: return "#1D5CB4"
            if m > 3: return "#4A90D9"
            return "#82B8E8"
        if m > 10: return "#C12A1C"
        if m > 3: return "#E05C4F"
        return "#E89990"
    else:
        if float(margin) >= 0:
            if m > 10: return "#1B4F72"
            if m > 3: return "#2B5F91"
            return "#8BB8D4"
        if m > 10: return "#7B241C"
        if m > 3: return "#BF3B33"
        return "#E0A49F"


def _prep_history(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["as_of_date"] = pd.to_datetime(out["as_of_date"], errors="coerce")
    out = out.dropna(subset=["as_of_date"]).sort_values("as_of_date").reset_index(drop=True)
    return out


@st.cache_data(show_spinner=False)
def load_bundle(root: str) -> dict[str, pd.DataFrame | dict]:
    paths = ProjectPaths(Path(root))
    latest = paths.latest_dir
    bundle: dict[str, pd.DataFrame | dict] = {}
    file_map = {
        "district_forecast": latest / "district_forecast.csv",
        "history": latest / "forecast_curve.csv" if (latest / "forecast_curve.csv").exists() else paths.forecast_history_csv,
        "run_history": paths.run_history_csv,
    }
    for name, fp in file_map.items():
        bundle[name] = pd.read_csv(fp) if fp.exists() else pd.DataFrame()
    sp = latest / "summary.json"
    bundle["summary"] = json.loads(sp.read_text()) if sp.exists() else {}
    return bundle


@st.cache_data(show_spinner=False)
def load_district_geojson(root: str) -> dict:
    gp = Path(root) / "data/processed/cd119_districts.geojson"
    return json.loads(gp.read_text()) if gp.exists() else {}


def _transform_geometry_coords(coords, transform):
    if not coords:
        return coords
    first = coords[0]
    if isinstance(first, (float, int)):
        lon, lat = coords
        return transform(float(lon), float(lat))
    return [_transform_geometry_coords(part, transform) for part in coords]


def _reposition_geometry(geometry: dict, *, center_lon: float, center_lat: float, target_lon: float, target_lat: float, scale: float) -> dict:
    def transform(lon: float, lat: float):
        return [target_lon + scale * (lon - center_lon), target_lat + scale * (lat - center_lat)]
    return {"type": geometry.get("type"), "coordinates": _transform_geometry_coords(geometry.get("coordinates"), transform)}


def _iter_coords(coords):
    if not coords:
        return
    first = coords[0]
    if isinstance(first, (float, int)):
        yield float(coords[0]), float(coords[1])
        return
    for part in coords:
        yield from _iter_coords(part)


def _iter_rings(geometry: dict):
    if not geometry:
        return
    if geometry.get("type") == "Polygon":
        for ring in geometry.get("coordinates", []):
            yield ring
    elif geometry.get("type") == "MultiPolygon":
        for polygon in geometry.get("coordinates", []):
            for ring in polygon:
                yield ring


# ── hemicycle ──────────────────────────────────────────────────────────
def _hemicycle_html(districts: pd.DataFrame, summary: dict) -> str:
    hc = districts.copy()
    hc["projected_winner"] = hc["mean_margin_sim"].map(lambda x: "Democrat" if float(x) >= 0 else "Republican")
    hc["projected_margin"] = hc["mean_margin_sim"].map(_margin_text)
    hc["gop_prob_t"] = hc["gop_win_prob"].map(_prob_text)
    hc["dem_prob_t"] = hc["dem_win_prob"].map(_prob_text)
    hc["open_t"] = hc["open_seat"].map(lambda x: "Open seat" if bool(x) else "Incumbent-held")
    hc["rating_t"] = hc["rating"].fillna("No rating")
    hc["result_t"] = hc["projected_winner"] + " " + hc["projected_margin"]
    hc["fill"] = hc["mean_margin_sim"].map(_margin_fill)
    hc["accent"] = hc["mean_margin_sim"].map(lambda x: DEM_COLOR if float(x) >= 0 else GOP_COLOR)
    dn_col = "district_name" if "district_name" in hc.columns else "district_code"
    hc = hc.sort_values("mean_margin_sim", ascending=False).reset_index(drop=True)

    cx0, cy0 = 400, 430
    inner_r, sp, nr = 130, 28, 9
    radii = [inner_r + i * sp for i in range(nr)]
    tot_r = sum(radii)
    row_seats = [round(TOTAL_HOUSE_SEATS * r / tot_r) for r in radii]
    diff = TOTAL_HOUSE_SEATS - sum(row_seats)
    for i in range(abs(diff)):
        row_seats[nr - 1 - i] += 1 if diff > 0 else -1

    seats = []
    for ri in range(nr):
        cnt = row_seats[ri]
        r = radii[ri]
        for j in range(cnt):
            pos = j / max(cnt - 1, 1)
            ang = math.pi * (0.95 - 0.9 * pos)
            seats.append((pos, cx0 + r * math.cos(ang), cy0 - r * math.sin(ang)))
    seats.sort(key=lambda s: s[0])

    circles = []
    nd = len(hc)
    for idx, (_, cx, cy) in enumerate(seats):
        if idx < nd:
            row = hc.iloc[idx]
            fill = row["fill"]
            acc = row["accent"]
            name = str(row.get(dn_col, "")) or str(row["district_code"])
            tip = (
                f"<div class='th' style='border-left:3px solid {acc}'>"
                f"<div class='tc'>{escape(str(row['district_code']))}</div>"
                f"<div class='tn'>{escape(name)}</div></div>"
                f"<div class='tr' style='color:{acc}'>{escape(str(row['result_t']))}</div>"
                f"<div class='tg'>"
                f"<span>Dem</span><strong style='color:{DEM_COLOR}'>{escape(str(row['dem_prob_t']))}</strong>"
                f"<span>GOP</span><strong style='color:{GOP_COLOR}'>{escape(str(row['gop_prob_t']))}</strong>"
                f"<span>Rating</span><strong>{escape(str(row['rating_t']))}</strong>"
                f"<span>Status</span><strong>{escape(str(row['open_t']))}</strong>"
                f"</div>"
            )
            circles.append(
                f'<circle class="hd" cx="{cx:.1f}" cy="{cy:.1f}" r="4.5" '
                f'fill="{fill}" data-tip="{escape(tip, quote=True)}" '
                f'style="animation-delay:{idx * 3}ms"/>'
            )
        else:
            circles.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4.5" fill="#1A1A1F"/>')

    gp = float(summary["gop_control_prob"])
    dp = float(summary["dem_control_prob"])
    fav = gp > dp
    lead = round(max(gp, dp) * 100)
    lc = GOP_COLOR if fav else DEM_COLOR
    lp = "Republicans" if fav else "Democrats"
    la = "keep" if fav else "win"

    return f"""
    <div class="hw" id="hcf">
      <svg viewBox="-15 -10 830 475" preserveAspectRatio="xMidYMid meet">
        {''.join(circles)}
        <text x="400" y="350" text-anchor="middle" dominant-baseline="central" class="hb" fill="{lc}">{lead}</text>
        <text x="400" y="393" text-anchor="middle" dominant-baseline="central" class="hu">in 100</text>
        <text x="400" y="415" text-anchor="middle" dominant-baseline="central" class="hl">{lp} {la} the House</text>
      </svg>
      <div id="htt" class="tt hid"></div>
    </div>
    <style>
      @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');
      *{{margin:0;padding:0;box-sizing:border-box}}
      body{{overflow:visible;background:{BG};margin:0}}
      .hw{{position:relative;padding:0 8px}}
      .hw svg{{display:block;width:100%;height:auto}}
      @keyframes seatIn{{from{{opacity:0;r:0}}to{{opacity:1;r:4.5}}}}
      .hd{{
        cursor:pointer;opacity:0;
        animation:seatIn 200ms ease forwards;
        transition:filter 100ms ease;
      }}
      .hd:hover{{
        r:7;
        filter:drop-shadow(0 0 8px currentColor) brightness(1.3);
      }}
      .hb{{
        font-family:'JetBrains Mono',monospace;
        font-size:72px;font-weight:700;letter-spacing:-0.04em;
      }}
      .hu{{
        font-family:'Space Grotesk',sans-serif;
        font-size:14px;font-weight:600;fill:{MUTED};
        text-transform:uppercase;letter-spacing:0.18em;
      }}
      .hl{{
        font-family:'Space Grotesk',sans-serif;
        font-size:13px;font-weight:500;fill:#AAABB0;
      }}
      .tt{{
        position:absolute;pointer-events:none;width:235px;
        background:{TOOLTIP_BG};border:1px solid {TOOLTIP_BORDER};
        border-radius:6px;box-shadow:0 8px 28px rgba(0,0,0,0.6);
        padding:11px 13px 9px;z-index:10;
        opacity:0;transition:opacity 80ms ease;
      }}
      .tt.vis{{opacity:1}}.tt.hid{{display:none}}
      .th{{padding-left:9px;margin-bottom:6px}}
      .tc{{font-family:'Space Grotesk',sans-serif;font-size:14px;font-weight:700;color:{INK}}}
      .tn{{font-family:'Space Grotesk',sans-serif;font-size:10.5px;color:{MUTED};margin-top:1px}}
      .tr{{
        font-family:'Space Grotesk',sans-serif;
        font-size:13px;font-weight:600;margin-bottom:7px;
        padding-bottom:6px;border-bottom:1px solid {TOOLTIP_BORDER};
      }}
      .tg{{
        display:grid;grid-template-columns:auto auto;
        gap:1px 10px;font-size:11px;line-height:1.65;
        font-family:'Space Grotesk',sans-serif;
      }}
      .tg span{{color:{MUTED}}}.tg strong{{text-align:right;font-weight:600;color:{INK}}}
    </style>
    <script>
      (function(){{
        var f=document.getElementById('hcf'),t=document.getElementById('htt');
        if(!f||!t)return;
        f.querySelectorAll('.hd').forEach(function(n){{
          n.addEventListener('mousemove',function(e){{
            t.innerHTML=n.dataset.tip;t.classList.remove('hid');t.classList.add('vis');
            var r=f.getBoundingClientRect(),l=e.clientX-r.left+14,tp=e.clientY-r.top-30;
            if(l+245>r.width)l=e.clientX-r.left-250;if(tp<0)tp=10;
            t.style.left=l+'px';t.style.top=tp+'px';
          }});
          n.addEventListener('mouseleave',function(){{t.classList.remove('vis');t.classList.add('hid')}});
        }});
      }})();
      (function(){{function rz(){{try{{if(window.frameElement)window.frameElement.style.height=document.body.scrollHeight+'px'}}catch(e){{}}}}rz();window.addEventListener('load',rz);new ResizeObserver(rz).observe(document.body)}})();
    </script>
    """


# ── district map ───────────────────────────────────────────────────────
def _district_map_html(districts: pd.DataFrame, district_geojson: dict, color_mode: str = "Forecast") -> str:
    if not district_geojson or "features" not in district_geojson:
        return "<p>District geography unavailable.</p>"

    m = districts.copy()
    m["projected_winner"] = m["mean_margin_sim"].map(lambda x: "Democrat" if float(x) >= 0 else "Republican")
    m["projected_margin"] = m["mean_margin_sim"].map(_margin_text)
    m["gop_win_prob_text"] = m["gop_win_prob"].map(_prob_text)
    m["dem_win_prob_text"] = m["dem_win_prob"].map(_prob_text)
    m["open_seat_text"] = m["open_seat"].map(lambda x: "Open seat" if bool(x) else "Incumbent-held")
    m["rating_text"] = m["rating"].fillna("No rating")
    m["estimated_result"] = m["projected_winner"] + " " + m["projected_margin"]
    lookup = m.set_index("district_code").to_dict("index")

    enriched = {"type": "FeatureCollection", "features": []}
    for feature in district_geojson.get("features", []):
        props = dict(feature.get("properties", {}))
        dc = props.get("district_code")
        row = lookup.get(dc)
        if row is None:
            continue
        geometry = feature.get("geometry")
        margin = float(row["mean_margin_sim"])
        accent = DEM_COLOR if margin >= 0 else GOP_COLOR
        _proj_winner = "DEM" if margin >= 0 else "REP"
        _prev_winner = str(row.get("winner_party_2024", ""))
        _is_flip = pd.notna(row.get("winner_party_2024")) and _prev_winner != _proj_winner
        if color_mode == "Forecast":
            fill = _margin_fill(margin)
        elif color_mode == "Shift vs 2024":
            _h24 = row.get("house_margin_2024", None)
            if pd.notna(_h24):
                _shift = margin - float(_h24)  # positive = shifted toward Dem
                _sa = abs(_shift)
                if _shift >= 0:  # shifted toward Dem (blue)
                    fill = "#1D5CB4" if _sa > 10 else ("#4A90D9" if _sa > 5 else "#82B8E8")
                else:  # shifted toward GOP (red)
                    fill = "#C12A1C" if _sa > 10 else ("#E05C4F" if _sa > 5 else "#E89990")
            else:
                fill = TOSSUP_COLOR  # no 2024 baseline
        else:  # Margin
            _m = abs(margin)
            if margin >= 0:
                fill = "#0E3F5E" if _m > 15 else (DEM_COLOR if _m > 8 else ("#6AAAD4" if _m > 3 else "#A8CBE3"))
            else:
                fill = "#6E1E18" if _m > 15 else (GOP_COLOR if _m > 8 else ("#D8877E" if _m > 3 else "#E8B8B2"))
        props.update({
            "district_name": row.get("district_name") or dc,
            "estimated_result": row.get("estimated_result"),
            "dem_odds": row.get("dem_win_prob_text"),
            "gop_odds": row.get("gop_win_prob_text"),
            "rating": row.get("rating_text"),
            "seat_status": row.get("open_seat_text"),
            "fill_color": fill, "accent": accent, "flipped": _is_flip,
            "region_group": "alaska" if dc.startswith("AK-") else "hawaii" if dc.startswith("HI-") else "main",
        })
        enriched["features"].append({"type": "Feature", "properties": props, "geometry": geometry})

    width, height = 1840, 1400
    lat0 = math.radians(37.0)
    boxes = {
        "main": {"x": 120, "y": 10, "w": 1600, "h": 810, "pad": 16},
        "alaska": {"x": 40, "y": 850, "w": 800, "h": 510, "pad": 18},
        "hawaii": {"x": 880, "y": 950, "w": 420, "h": 300, "pad": 18},
    }
    group_points: dict[str, list[tuple[float, float]]] = {"main": [], "alaska": [], "hawaii": []}
    for feature in enriched["features"]:
        grp = feature["properties"]["region_group"]
        for lon, lat in _iter_coords(feature.get("geometry", {}).get("coordinates", [])):
            if grp == "alaska" and lon > 0:
                lon = lon - 360  # wrap Aleutians across date line
            group_points[grp].append((lon * math.cos(lat0), lat))

    def make_projector(gn):
        pts = group_points[gn]
        if not pts:
            return lambda lon, lat: (0.0, 0.0)
        mnx, mxx = min(x for x, _ in pts), max(x for x, _ in pts)
        mny, mxy = min(y for _, y in pts), max(y for _, y in pts)
        bx = boxes[gn]
        sc = min((bx["w"] - 2 * bx["pad"]) / max(mxx - mnx, 1e-9), (bx["h"] - 2 * bx["pad"]) / max(mxy - mny, 1e-9))
        cw, ch = (mxx - mnx) * sc, (mxy - mny) * sc
        def project(lon, lat):
            x = (lon * math.cos(lat0) - mnx) * sc + bx["x"] + (bx["w"] - cw) / 2
            y = bx["y"] + bx["h"] - ((lat - mny) * sc + (bx["h"] - ch) / 2)
            return x, y
        return project

    projectors = {n: make_projector(n) for n in boxes}

    def ring_to_path(ring, gn):
        def _wrap(lo, la):
            lo = float(lo)
            if gn == "alaska" and lo > 0:
                lo = lo - 360
            return projectors[gn](lo, float(la))
        pts = [_wrap(lo, la) for lo, la in ring]
        return ("M " + " L ".join(f"{x:.2f} {y:.2f}" for x, y in pts) + " Z") if pts else ""

    paths = []
    for feature in enriched["features"]:
        geo = feature.get("geometry", {})
        gn = feature["properties"]["region_group"]
        rp = [ring_to_path(ring, gn) for ring in _iter_rings(geo)]
        pd_str = " ".join(p for p in rp if p)
        pr = feature["properties"]
        acc = pr["accent"]
        _flip_tag = "<span style='color:#FFD166;font-size:10px;font-weight:700'> FLIP</span>" if pr.get("flipped") else ""
        tip = (
            f"<div class='th' style='border-left:3px solid {acc}'>"
            f"<div class='tc'>{escape(pr['district_code'])}{_flip_tag}</div>"
            f"<div class='tn'>{escape(str(pr['district_name']))}</div></div>"
            f"<div class='tr' style='color:{acc}'>{escape(str(pr['estimated_result']))}</div>"
            f"<div class='tg'>"
            f"<span>Dem</span><strong style='color:{DEM_COLOR}'>{escape(str(pr['dem_odds']))}</strong>"
            f"<span>GOP</span><strong style='color:{GOP_COLOR}'>{escape(str(pr['gop_odds']))}</strong>"
            f"<span>Rating</span><strong>{escape(str(pr['rating']))}</strong>"
            f"<span>Status</span><strong>{escape(str(pr['seat_status']))}</strong>"
            f"</div>"
        )
        _fill_val = pr["fill_color"]
        if pr.get("flipped"):
            _fpid = f"hf_{pr['district_code'].replace('-','_')}"
            paths.append(
                f'<pattern id="{_fpid}" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(135)">'
                f'<rect width="5" height="5" fill="{pr["fill_color"]}"/>'
                f'<rect width="2.5" height="5" fill="rgba(255,255,255,0.2)"/></pattern>'
            )
            _fill_val = f"url(#{_fpid})"
        paths.append(f'<path class="d" d="{pd_str}" fill="{_fill_val}" data-tip="{escape(tip, quote=True)}"/>')

    _h_defs = [p for p in paths if p.startswith("<pattern")]
    _h_paths = [p for p in paths if not p.startswith("<pattern")]
    ak, hi = boxes["alaska"], boxes["hawaii"]
    return f"""
    <div class="mw" id="dmf">
      <svg viewBox="0 0 {width} {height}">
        <defs>{''.join(_h_defs)}</defs>
        <rect x="0" y="0" width="{width}" height="{height}" fill="{BG}" rx="8" ry="8"/>
        <line x1="{ak['x']}" y1="{ak['y']}" x2="{ak['x']+ak['w']}" y2="{ak['y']}" stroke="{RULE}" stroke-width="0.5"/>
        <line x1="{hi['x']}" y1="{hi['y']}" x2="{hi['x']+hi['w']}" y2="{hi['y']}" stroke="{RULE}" stroke-width="0.5"/>
        <g>{''.join(_h_paths)}</g>
        <text x="{ak['x']+14}" y="{ak['y']+18}" class="il">ALASKA</text>
        <text x="{hi['x']+14}" y="{hi['y']+18}" class="il">HAWAII</text>
      </svg>
      <div class="lg">
        <div class="li"><span style="background:#1D5CB4"></span>Safe D</div>
        <div class="li"><span style="background:#4A90D9"></span>Likely D</div>
        <div class="li"><span style="background:#82B8E8"></span>Lean D</div>
        <div class="li"><span style="background:#9B7FBF"></span>Tossup</div>
        <div class="li"><span style="background:#E89990"></span>Lean R</div>
        <div class="li"><span style="background:#E05C4F"></span>Likely R</div>
        <div class="li"><span style="background:#C12A1C"></span>Safe R</div>
      </div>
      <div id="mtt" class="tt hid"></div>
    </div>
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');
      *{{font-family:'Space Grotesk',sans-serif;margin:0;padding:0;box-sizing:border-box}}
      body{{overflow:visible;background:{BG};margin:0}}
      .mw{{position:relative}}
      .mw svg{{display:block;width:100%;height:auto}}
      .d{{stroke:{BG};stroke-width:0.7;cursor:pointer;transition:stroke 80ms ease,stroke-width 80ms ease,filter 80ms ease}}
      .d:hover{{stroke:{INK};stroke-width:1.6;filter:brightness(1.15)}}
      .il{{font-size:11px;font-weight:700;letter-spacing:0.14em;fill:#555560}}
      .tt{{
        position:absolute;pointer-events:none;width:235px;
        background:{TOOLTIP_BG};border:1px solid {TOOLTIP_BORDER};
        border-radius:6px;box-shadow:0 8px 28px rgba(0,0,0,0.6);
        padding:11px 13px 9px;z-index:10;opacity:0;transition:opacity 80ms ease;
      }}
      .tt.vis{{opacity:1}}.tt.hid{{display:none}}
      .th{{padding-left:9px;margin-bottom:6px}}
      .tc{{font-size:14px;font-weight:700;color:{INK}}}
      .tn{{font-size:10.5px;color:{MUTED};margin-top:1px}}
      .tr{{font-size:13px;font-weight:600;margin-bottom:7px;padding-bottom:6px;border-bottom:1px solid {TOOLTIP_BORDER}}}
      .tg{{display:grid;grid-template-columns:auto auto;gap:1px 10px;font-size:11px;line-height:1.65}}
      .tg span{{color:{MUTED}}}.tg strong{{text-align:right;font-weight:600;color:{INK}}}
      .lg{{display:flex;justify-content:center;gap:14px;padding:8px 0 0}}
      .li{{display:flex;align-items:center;gap:4px;font-size:10.5px;font-weight:600;color:{MUTED}}}
      .li span{{display:block;width:11px;height:11px;border-radius:2px}}
      @media(max-width:480px){{.tt{{width:180px;font-size:10px}}.tc{{font-size:12px}}.tg{{font-size:9px;gap:1px 6px}}}}
    </style>
    <script>
      (function(){{
        var f=document.getElementById('dmf'),t=document.getElementById('mtt');
        if(!f||!t)return;
        f.querySelectorAll('.d').forEach(function(n){{
          n.addEventListener('mousemove',function(e){{
            t.innerHTML=n.dataset.tip;t.classList.remove('hid');t.classList.add('vis');
            var r=f.getBoundingClientRect(),l=e.clientX-r.left+14,tp=e.clientY-r.top-30;
            if(l+245>r.width)l=e.clientX-r.left-250;if(tp<0)tp=10;
            t.style.left=l+'px';t.style.top=tp+'px';
          }});
          n.addEventListener('mouseleave',function(){{t.classList.remove('vis');t.classList.add('hid')}});
        }});
      }})();
      (function(){{function rz(){{try{{if(window.frameElement)window.frameElement.style.height=document.body.scrollHeight+'px'}}catch(e){{}}}}rz();window.addEventListener('load',rz);new ResizeObserver(rz).observe(document.body)}})();
    </script>
    """


# ── seat distribution histogram ────────────────────────────────────────
def _seat_distribution_chart(seat_dist: pd.DataFrame, summary: dict) -> go.Figure:
    df = seat_dist[(seat_dist["frequency"] > 0)].copy()
    df["dem_seats"] = TOTAL_HOUSE_SEATS - df["gop_seats"]
    # Color gradient: deeper for bars closer to the expected value, lighter at tails
    expected = float(summary["expected_gop_seats"])
    def _dist_color(gop_s):
        if gop_s < MAJORITY_SEATS:
            return DEM_COLOR
        return GOP_COLOR
    df["color"] = df["gop_seats"].map(_dist_color)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["gop_seats"], y=df["probability"],
        marker=dict(color=df["color"], line=dict(width=0)),
        customdata=list(zip(df["dem_seats"])),
        hovertemplate="<b>GOP %{x} / Dem %{customdata[0]}</b><br>Probability: %{y:.2%}<extra></extra>",
    ))
    # Quantile markers from the actual model output
    q05 = float(summary.get("gop_seat_q05", 0))
    q25 = float(summary.get("gop_seat_q25", 0))
    median = float(summary.get("median_gop_seats", 0))
    q75 = float(summary.get("gop_seat_q75", 0))
    q95 = float(summary.get("gop_seat_q95", 0))
    fig.add_vline(x=MAJORITY_SEATS, line_width=2, line_color=AMBER,
                  annotation_text="218 majority", annotation_position="top",
                  annotation_font=dict(size=10, color=AMBER, family="JetBrains Mono, monospace"))
    fig.add_vline(x=median, line_width=1.5, line_dash="solid", line_color=INK,
                  annotation_text=f"Median: {median:.0f}", annotation_position="top right",
                  annotation_font=dict(size=9, color=INK, family="JetBrains Mono, monospace"))
    # 90% CI shading
    fig.add_vrect(x0=q05, x1=q95, fillcolor=f"rgba(255,255,255,0.03)" if DARK else "rgba(0,0,0,0.03)",
                  line_width=0, annotation_text=f"90% CI: {q05:.0f}\u2013{q95:.0f}",
                  annotation_position="bottom right",
                  annotation_font=dict(size=8, color=MUTED, family="JetBrains Mono, monospace"))
    fig.update_layout(
        height=260, showlegend=False, bargap=0,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=44, r=12, t=28, b=32),
        hovermode="x",
        font=dict(family="Space Grotesk, sans-serif", color=INK, size=12),
        hoverlabel=dict(bgcolor=TOOLTIP_BG, bordercolor=TOOLTIP_BORDER, namelength=-1,
                        font=dict(color=INK, family="Space Grotesk, sans-serif", size=13)),
    )
    fig.update_xaxes(showgrid=False, linecolor=RULE, tickfont=dict(size=10, color=SUBTLE),
                     title=dict(text="GOP seats", font=dict(size=11, color=SUBTLE)))
    fig.update_yaxes(showgrid=True, gridcolor=CHART_GRID, zeroline=False,
                     linecolor=RULE, tickformat=".1%", tickfont=dict(size=10, color=SUBTLE))
    return fig


# ── seat aggregate over time ────────────────────────────────────────────
def _seat_aggregate_chart(history: pd.DataFrame) -> go.Figure:
    h = history.copy()
    h["expected_dem_seats"] = TOTAL_HOUSE_SEATS - h["expected_gop_seats"]

    fig = go.Figure()

    has_ci = "gop_seat_q05" in h.columns and "gop_seat_q95" in h.columns
    if has_ci:
        h["dem_q95"] = TOTAL_HOUSE_SEATS - h["gop_seat_q05"]
        h["dem_q05"] = TOTAL_HOUSE_SEATS - h["gop_seat_q95"]
        fig.add_trace(go.Scatter(
            x=h["as_of_date"], y=h["gop_seat_q95"],
            mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=h["as_of_date"], y=h["gop_seat_q05"],
            mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
            fill="tonexty", fillcolor="rgba(224,92,79,0.08)",
        ))
        fig.add_trace(go.Scatter(
            x=h["as_of_date"], y=h["dem_q95"],
            mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=h["as_of_date"], y=h["dem_q05"],
            mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
            fill="tonexty", fillcolor="rgba(74,144,217,0.08)",
        ))

    fig.add_trace(go.Scatter(
        x=h["as_of_date"], y=h["expected_gop_seats"],
        mode="lines", name="GOP seats",
        line=dict(color=GOP_COLOR, width=2.5),
        hovertemplate="%{x|%b %d, %Y}<br>GOP: %{y:.1f} seats<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=h["as_of_date"], y=h["expected_dem_seats"],
        mode="lines", name="Dem seats",
        line=dict(color=DEM_COLOR, width=2.5),
        hovertemplate="%{x|%b %d, %Y}<br>Dem: %{y:.1f} seats<extra></extra>",
    ))
    fig.add_hline(y=MAJORITY_SEATS, line_width=1, line_dash="dot", line_color=AMBER,
                  annotation_text="218", annotation_position="right",
                  annotation_font=dict(size=10, color=AMBER, family="JetBrains Mono, monospace"))
    fig.update_layout(
        height=280,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=44, r=12, t=10, b=32),
        hovermode="x unified",
        font=dict(family="Space Grotesk, sans-serif", color=INK, size=12),
        legend=dict(orientation="h", y=1.1, x=0, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=11, color=MUTED)),
        hoverlabel=dict(bgcolor=TOOLTIP_BG, bordercolor=TOOLTIP_BORDER, namelength=-1,
                        font=dict(color=INK, family="Space Grotesk, sans-serif", size=13)),
    )
    fig.update_xaxes(showgrid=False, linecolor=RULE, tickfont=dict(size=10, color=SUBTLE), tickformat="%b '%y")
    fig.update_yaxes(showgrid=True, gridcolor=CHART_GRID, zeroline=False,
                     linecolor=RULE, title=dict(text="Seats", font=dict(size=11, color=SUBTLE)),
                     tickfont=dict(size=10, color=SUBTLE))
    return fig


# ── odds chart ─────────────────────────────────────────────────────────
def _odds_chart(history: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=history["as_of_date"], y=history["gop_control_prob"],
        mode="lines", name="Republicans",
        line=dict(color=GOP_COLOR, width=2.5),
        hovertemplate="%{x|%b %d, %Y}<br>GOP: %{y:.1%}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=history["as_of_date"], y=history["dem_control_prob"],
        mode="lines", name="Democrats",
        line=dict(color=DEM_COLOR, width=2.5),
        hovertemplate="%{x|%b %d, %Y}<br>Dem: %{y:.1%}<extra></extra>",
    ))
    fig.add_hline(y=0.50, line_dash="dot", line_color="#333338", line_width=1)
    fig.update_layout(
        height=260,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=44, r=12, t=10, b=32),
        hovermode="x unified",
        font=dict(family="Space Grotesk, sans-serif", color=INK, size=12),
        legend=dict(orientation="h", y=1.1, x=0, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=11, color=MUTED)),
        hoverlabel=dict(bgcolor=TOOLTIP_BG, bordercolor=TOOLTIP_BORDER, namelength=-1,
                        font=dict(color=INK, family="Space Grotesk, sans-serif", size=13)),
    )
    fig.update_xaxes(showgrid=False, linecolor=RULE, tickfont=dict(size=10, color=SUBTLE), tickformat="%b '%y")
    fig.update_yaxes(showgrid=True, gridcolor=CHART_GRID, zeroline=False,
                     linecolor=RULE, tickformat=".0%", range=[0, 1],
                     tickvals=[0, 0.25, 0.5, 0.75, 1.0], tickfont=dict(size=10, color=SUBTLE))
    return fig


# ── generic ballot chart ────────────────────────────────────────────────
def _generic_ballot_chart(history: pd.DataFrame, polls: pd.DataFrame) -> go.Figure:
    h = history.copy()
    h["dem_pct_est"] = 50.0 + h["generic_ballot_margin_dem"] / 2.0
    h["gop_pct_est"] = 50.0 - h["generic_ballot_margin_dem"] / 2.0
    h["margin"] = h["generic_ballot_margin_dem"]

    ci_half = h.get("filtered_generic_sd")
    if ci_half is not None:
        h["dem_hi"] = h["dem_pct_est"] + 1.645 * ci_half / 2
        h["dem_lo"] = h["dem_pct_est"] - 1.645 * ci_half / 2
        h["gop_hi"] = h["gop_pct_est"] + 1.645 * ci_half / 2
        h["gop_lo"] = h["gop_pct_est"] - 1.645 * ci_half / 2

    fig = go.Figure()

    if "dem_hi" in h.columns:
        fig.add_trace(go.Scatter(x=h["as_of_date"], y=h["dem_hi"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=h["as_of_date"], y=h["dem_lo"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip", fill="tonexty", fillcolor="rgba(74,144,217,0.08)"))
        fig.add_trace(go.Scatter(x=h["as_of_date"], y=h["gop_hi"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=h["as_of_date"], y=h["gop_lo"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip", fill="tonexty", fillcolor="rgba(224,92,79,0.08)"))

    fig.add_trace(go.Scatter(
        x=h["as_of_date"], y=h["dem_pct_est"],
        mode="lines", name="Democrats",
        line=dict(color=DEM_COLOR, width=2.5),
        customdata=list(zip(h["gop_pct_est"], h["margin"])),
        hovertemplate="<b style='font-size:12px'>%{x|%b %d, %Y}</b><br><span style='color:" + DEM_COLOR + "'>D %{y:.1f}%</span> / <span style='color:" + GOP_COLOR + "'>R %{customdata[0]:.1f}%</span><br>Margin: D%{customdata[1]:+.1f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=h["as_of_date"], y=h["gop_pct_est"],
        mode="lines", name="Republicans",
        line=dict(color=GOP_COLOR, width=2.5),
        hoverinfo="skip",
    ))

    if not polls.empty and "obs_date" in polls.columns:
        p = polls.copy()
        p["obs_date"] = pd.to_datetime(p["obs_date"])
        fig.add_trace(go.Scatter(
            x=p["obs_date"], y=p["dem_pct"],
            mode="markers", name="Dem poll",
            marker=dict(color=DEM_COLOR, size=5, opacity=0.35, line=dict(width=0)),
            customdata=list(zip(p["pollster"], p["rep_pct"], p["margin_a"], p["sample_size"].fillna(0).astype(int), p["population"].fillna(""))),
            hovertemplate="<b style='font-size:11px'>%{customdata[0]}</b><br><span style='font-size:10px'>%{x|%b %d} &middot; n=%{customdata[3]:,} %{customdata[4]}</span><br><span style='color:" + DEM_COLOR + "'>D %{y:.0f}%</span> / <span style='color:" + GOP_COLOR + "'>R %{customdata[1]:.0f}%</span> &middot; D%{customdata[2]:+.1f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=p["obs_date"], y=p["rep_pct"],
            mode="markers", name="GOP poll",
            marker=dict(color=GOP_COLOR, size=5, opacity=0.35, line=dict(width=0)),
            hoverinfo="skip",
        ))

    fig.add_hline(y=50, line_width=1, line_dash="dot", line_color="#333338")

    fig.update_layout(
        height=240,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=44, r=12, t=8, b=28),
        hovermode="closest",
        font=dict(family="Space Grotesk, sans-serif", color=INK, size=12),
        legend=dict(orientation="h", y=1.1, x=0, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=10, color=MUTED)),
        hoverlabel=dict(bgcolor=TOOLTIP_BG, bordercolor=TOOLTIP_BORDER, namelength=-1,
                        font=dict(color=INK, family="Space Grotesk, sans-serif", size=12)),
    )
    fig.update_xaxes(showgrid=False, linecolor=RULE, tickfont=dict(size=10, color=SUBTLE), tickformat="%b '%y")
    fig.update_yaxes(showgrid=True, gridcolor=CHART_GRID, zeroline=False,
                     linecolor=RULE, ticksuffix="%", tickfont=dict(size=10, color=SUBTLE))
    return fig


# ── trump approval chart ────────────────────────────────────────────────
APPROVE_COLOR = "#5BAA6A" if DARK else "#2D8A3E"
DISAPPROVE_COLOR = "#C9604E" if DARK else "#B5423A"


def _approval_chart(approval: pd.DataFrame, polls: pd.DataFrame) -> go.Figure:
    a = approval.copy()
    a["as_of_date"] = pd.to_datetime(a["as_of_date"])
    a["net"] = a["approve_avg"] - a["disapprove_avg"]

    fig = go.Figure()

    if "approve_low_90" in a.columns:
        fig.add_trace(go.Scatter(
            x=a["as_of_date"], y=a["approve_high_90"],
            mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=a["as_of_date"], y=a["approve_low_90"],
            mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
            fill="tonexty", fillcolor="rgba(91,170,106,0.08)",
        ))
        fig.add_trace(go.Scatter(
            x=a["as_of_date"], y=a["disapprove_high_90"],
            mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=a["as_of_date"], y=a["disapprove_low_90"],
            mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
            fill="tonexty", fillcolor="rgba(201,96,78,0.08)",
        ))

    fig.add_trace(go.Scatter(
        x=a["as_of_date"], y=a["approve_avg"],
        mode="lines", name="Approve",
        line=dict(color=APPROVE_COLOR, width=2.5),
        customdata=list(zip(a["disapprove_avg"], a["net"])),
        hovertemplate=(
            "<b>%{x|%b %d, %Y}</b><br>"
            "<span style='color:#5BAA6A'>Approve: %{y:.1f}%</span><br>"
            "<span style='color:#C9604E'>Disapprove: %{customdata[0]:.1f}%</span><br>"
            "Net: %{customdata[1]:+.1f}<extra></extra>"
        ),
    ))
    fig.add_trace(go.Scatter(
        x=a["as_of_date"], y=a["disapprove_avg"],
        mode="lines", name="Disapprove",
        line=dict(color=DISAPPROVE_COLOR, width=2.5),
        hoverinfo="skip",
    ))

    if not polls.empty and "obs_date" in polls.columns:
        p = polls.copy()
        p["obs_date"] = pd.to_datetime(p["obs_date"])
        p["net"] = p["pct_a"] - p["pct_b"]
        fig.add_trace(go.Scatter(
            x=p["obs_date"], y=p["pct_a"],
            mode="markers", name="Poll (approve)",
            marker=dict(color=APPROVE_COLOR, size=5, opacity=0.4, line=dict(width=0)),
            customdata=list(zip(p["pollster"], p["pct_b"], p["net"], p["sample_size"], p["population"])),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "%{x|%b %d, %Y} &middot; n=%{customdata[3]:,} %{customdata[4]}<br>"
                "<span style='color:#5BAA6A'>Approve: %{y:.0f}%</span><br>"
                "<span style='color:#C9604E'>Disapprove: %{customdata[1]:.0f}%</span><br>"
                "Net: %{customdata[2]:+.0f}<extra></extra>"
            ),
        ))
        fig.add_trace(go.Scatter(
            x=p["obs_date"], y=p["pct_b"],
            mode="markers", name="Poll (disapprove)",
            marker=dict(color=DISAPPROVE_COLOR, size=5, opacity=0.4, line=dict(width=0)),
            hoverinfo="skip",
        ))

    fig.update_layout(
        height=240,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=44, r=12, t=8, b=28),
        hovermode="x",
        font=dict(family="Space Grotesk, sans-serif", color=INK, size=12),
        legend=dict(orientation="h", y=1.1, x=0, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=10, color=MUTED)),
        hoverlabel=dict(
            bgcolor=TOOLTIP_BG, bordercolor=TOOLTIP_BORDER,
            font=dict(color=INK, family="Space Grotesk, sans-serif", size=12),
        ),
    )
    fig.update_xaxes(showgrid=False, linecolor=RULE,
                     tickfont=dict(size=10, color=SUBTLE), tickformat="%b '%y")
    fig.update_yaxes(showgrid=True, gridcolor=CHART_GRID, zeroline=False,
                     linecolor=RULE, ticksuffix="%", tickfont=dict(size=10, color=SUBTLE))
    return fig


# ── data ───────────────────────────────────────────────────────────────
bundle = load_bundle(str(ROOT))
summary = bundle["summary"] or {}
districts = bundle["district_forecast"] if isinstance(bundle["district_forecast"], pd.DataFrame) else pd.DataFrame()
history = _prep_history(bundle["history"] if isinstance(bundle["history"], pd.DataFrame) else pd.DataFrame())
district_geojson = load_district_geojson(str(ROOT))

_latest = ProjectPaths(ROOT).latest_dir
seat_dist_path = _latest / "seat_distribution.csv"
seat_dist = pd.read_csv(seat_dist_path) if seat_dist_path.exists() else pd.DataFrame()

# District-level daily history from suite
SUITE_ROOT = ROOT / "suite"
_dist_hist_path = SUITE_ROOT / "house/data/history/district_forecast_history.csv"
dist_history = pd.read_csv(_dist_hist_path) if _dist_hist_path.exists() else pd.DataFrame()
if not dist_history.empty:
    dist_history["as_of_date"] = pd.to_datetime(dist_history["as_of_date"], errors="coerce")

approval_path = _latest / "trump_approval_curve.csv"
approval_data = pd.read_csv(approval_path) if approval_path.exists() else pd.DataFrame()

approval_polls_path = _latest / "trump_approval_polls.csv"
approval_polls = pd.read_csv(approval_polls_path) if approval_polls_path.exists() else pd.DataFrame()

gb_polls_path = _latest / "generic_ballot_polls.csv"
gb_polls = pd.read_csv(gb_polls_path) if gb_polls_path.exists() else pd.DataFrame()

# ── suite data (upgraded source) ───────────────────────────────────────
SUITE_ROOT = ROOT / "suite"
SENATE_ROOT = SUITE_ROOT / "senate"
_sen_latest = SENATE_ROOT / "data/runtime/latest"
_sen_summary_path = _sen_latest / "summary.json"
sen_summary = json.loads(_sen_summary_path.read_text()) if _sen_summary_path.exists() else {}
_sen_races_path = _sen_latest / "race_forecast.csv"
sen_races = pd.read_csv(_sen_races_path) if _sen_races_path.exists() else pd.DataFrame()
_sen_dist_path = _sen_latest / "seat_distribution.csv"
sen_seat_dist = pd.read_csv(_sen_dist_path) if _sen_dist_path.exists() else pd.DataFrame()
_sen_gpolls_path = _sen_latest / "general_polls.csv"
_sen_gpolls = pd.read_csv(_sen_gpolls_path) if _sen_gpolls_path.exists() else pd.DataFrame()
_sen_poll_counts = _sen_gpolls.groupby("state_abbr").size().to_dict() if not _sen_gpolls.empty and "state_abbr" in _sen_gpolls.columns else {}

# Suite extras: race-level time series
_sen_race_hist_path = SENATE_ROOT / "data/history/state_forecast_history.csv"
sen_race_history = pd.read_csv(_sen_race_hist_path) if _sen_race_hist_path.exists() else pd.DataFrame()
if not sen_race_history.empty:
    sen_race_history["as_of_date"] = pd.to_datetime(sen_race_history["as_of_date"], errors="coerce")

# Combined chamber history (both chambers in one file)
_combined_hist_path = SUITE_ROOT / "data/combined/chamber_history.csv"
combined_chamber_hist = pd.read_csv(_combined_hist_path) if _combined_hist_path.exists() else pd.DataFrame()
if not combined_chamber_hist.empty:
    combined_chamber_hist["as_of_date"] = pd.to_datetime(combined_chamber_hist["as_of_date"], errors="coerce")

if not summary or districts.empty or history.empty:
    st.error("Forecast outputs not available. Run the main app refresh first.")
    st.stop()

gop_seats = float(summary["expected_gop_seats"])
dem_seats = TOTAL_HOUSE_SEATS - gop_seats
gop_prob = float(summary["gop_control_prob"])
dem_prob = float(summary["dem_control_prob"])
gop_chances = round(gop_prob * 100)
dem_chances = round(dem_prob * 100)
gop_favored = gop_prob > dem_prob
lead_party = "Republicans" if gop_favored else "Democrats"
lead_action = "keep" if gop_favored else "win"
lead_color = GOP_COLOR if gop_favored else DEM_COLOR

# ── seat strip data ────────────────────────────────────────────────────
_sorted = districts.sort_values("mean_margin_sim", ascending=False)
_strip = "".join(f'<div style="background:{_margin_fill(r["mean_margin_sim"])}"></div>' for _, r in _sorted.iterrows())

# count seats per category
def _categorize(margin: float) -> str:
    m = abs(float(margin))
    if m < 1.5:
        return "tossup"
    if float(margin) >= 0:
        if m > 10: return "safe_d"
        if m > 3: return "likely_d"
        return "lean_d"
    if m > 10: return "safe_r"
    if m > 3: return "likely_r"
    return "lean_r"

_cats = districts["mean_margin_sim"].map(_categorize)
_cat_counts = {
    "safe_d": int((_cats == "safe_d").sum()),
    "likely_d": int((_cats == "likely_d").sum()),
    "lean_d": int((_cats == "lean_d").sum()),
    "tossup": int((_cats == "tossup").sum()),
    "lean_r": int((_cats == "lean_r").sum()),
    "likely_r": int((_cats == "likely_r").sum()),
    "safe_r": int((_cats == "safe_r").sum()),
}
_seg_defs = [
    ("Safe D", "#1D5CB4", _cat_counts["safe_d"]),
    ("Likely D", "#4A90D9", _cat_counts["likely_d"]),
    ("Lean D", "#82B8E8", _cat_counts["lean_d"]),
    ("Tossup", TOSSUP_COLOR, _cat_counts["tossup"]),
    ("Lean R", "#E89990", _cat_counts["lean_r"]),
    ("Likely R", "#E05C4F", _cat_counts["likely_r"]),
    ("Safe R", "#C12A1C", _cat_counts["safe_r"]),
]

# ── senate tile map ─────────────────────────────────────────────────────
TILE_POS = {
    "AK":(0,0),"ME":(10,0),
    "WI":(5,1),"VT":(9,1),"NH":(10,1),
    "WA":(0,2),"ID":(1,2),"MT":(2,2),"ND":(3,2),"MN":(4,2),"IL":(5,2),"MI":(6,2),"NY":(8,2),"MA":(9,2),"CT":(10,2),
    "OR":(0,3),"NV":(1,3),"WY":(2,3),"SD":(3,3),"IA":(4,3),"IN":(5,3),"OH":(6,3),"PA":(7,3),"NJ":(8,3),"RI":(9,3),
    "CA":(0,4),"UT":(1,4),"CO":(2,4),"NE":(3,4),"MO":(4,4),"KY":(5,4),"WV":(6,4),"VA":(7,4),"MD":(8,4),"DE":(9,4),
    "AZ":(1,5),"NM":(2,5),"KS":(3,5),"AR":(4,5),"TN":(5,5),"NC":(6,5),"SC":(7,5),
    "OK":(3,6),"LA":(4,6),"MS":(5,6),"AL":(6,6),"GA":(7,6),
    "HI":(0,7),"TX":(3,7),"FL":(7,7),
}


def _senate_tile_map_html(races: pd.DataFrame, sen_sum: dict) -> str:
    lookup = races.set_index("state_abbr").to_dict("index") if not races.empty else {}
    cell = 52
    gap = 3
    step = cell + gap
    max_col = max(c for c, _ in TILE_POS.values()) + 1
    max_row = max(r for _, r in TILE_POS.values()) + 1
    w = max_col * step + gap
    h = max_row * step + gap

    tiles = []
    for st_abbr, (col, row) in TILE_POS.items():
        x = gap + col * step
        y = gap + row * step
        race = lookup.get(st_abbr)
        if race:
            dp = float(race["dem_win_prob"])
            fill = _margin_fill(float(race["expected_dem_margin"]))
            cands = f"{race.get('top_dem_candidate','?')} vs {race.get('top_rep_candidate','?')}"
            tip = (
                f"<div class='tc'>{st_abbr} &mdash; {race.get('state_name','')}</div>"
                f"<div class='tr' style='color:{DEM_COLOR if dp>0.5 else GOP_COLOR}'>"
                f"{'Dem' if dp>0.5 else 'GOP'} {max(dp,1-dp):.0%}</div>"
                f"<div class='tn'>{escape(cands)}</div>"
                f"<div class='tg'>"
                f"<span>Dem</span><strong style='color:{DEM_COLOR}'>{dp:.1%}</strong>"
                f"<span>GOP</span><strong style='color:{GOP_COLOR}'>{1-dp:.1%}</strong>"
                f"<span>Margin</span><strong>{_margin_text(float(race['expected_dem_margin']))}</strong>"
                f"<span>Rating</span><strong>{race.get('rating_category','').replace('_',' ').title()}</strong>"
                f"</div>"
            )
            opacity = "1"
        else:
            fill = "#1A1A1F"
            tip = f"<div class='tc'>{st_abbr}</div><div class='tn'>Not up in 2026</div>"
            opacity = "0.5"
        tiles.append(
            f'<rect class="st" x="{x}" y="{y}" width="{cell}" height="{cell}" rx="4" '
            f'fill="{fill}" opacity="{opacity}" data-tip="{escape(tip, quote=True)}"/>'
            f'<text x="{x+cell//2}" y="{y+cell//2+1}" text-anchor="middle" dominant-baseline="central" '
            f'class="sl" {"style=opacity:0.45" if not race else ""}>{st_abbr}</text>'
        )

    return f"""
    <div class="sm" id="smf">
      <svg viewBox="0 0 {w} {h}" preserveAspectRatio="xMidYMid meet">
        {''.join(tiles)}
      </svg>
      <div id="stt" class="tt hid"></div>
    </div>
    <style>
      @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
      *{{margin:0;padding:0;box-sizing:border-box;font-family:'Space Grotesk',sans-serif}}
      body{{overflow:visible;background:{BG};margin:0}}
      .sm{{position:relative}}
      .sm svg{{display:block;width:100%;height:auto}}
      .st{{cursor:pointer;transition:opacity 100ms ease,filter 100ms ease}}
      .st:hover{{filter:brightness(1.25)}}
      .sl{{font-family:'JetBrains Mono',monospace;font-size:11px;font-weight:700;fill:white;pointer-events:none}}
      .tt{{
        position:absolute;pointer-events:none;width:230px;
        background:{TOOLTIP_BG};border:1px solid {TOOLTIP_BORDER};
        border-radius:6px;box-shadow:0 8px 28px rgba(0,0,0,0.6);
        padding:10px 12px 8px;z-index:10;opacity:0;transition:opacity 80ms ease;
      }}
      .tt.vis{{opacity:1}}.tt.hid{{display:none}}
      .tc{{font-size:14px;font-weight:700;color:{INK};margin-bottom:2px}}
      .tn{{font-size:10.5px;color:{MUTED};margin-bottom:6px}}
      .tr{{font-size:13px;font-weight:600;margin-bottom:6px;padding-bottom:5px;border-bottom:1px solid {TOOLTIP_BORDER}}}
      .tg{{display:grid;grid-template-columns:auto auto;gap:1px 10px;font-size:11px;line-height:1.65}}
      .tg span{{color:{MUTED}}}.tg strong{{text-align:right;font-weight:600;color:{INK}}}
    </style>
    <script>
      (function(){{
        var f=document.getElementById('smf'),t=document.getElementById('stt');
        if(!f||!t)return;
        f.querySelectorAll('.st').forEach(function(n){{
          n.addEventListener('mousemove',function(e){{
            t.innerHTML=n.dataset.tip;t.classList.remove('hid');t.classList.add('vis');
            var r=f.getBoundingClientRect(),l=e.clientX-r.left+14,tp=e.clientY-r.top-30;
            if(l+240>r.width)l=e.clientX-r.left-245;if(tp<0)tp=10;
            t.style.left=l+'px';t.style.top=tp+'px';
          }});
          n.addEventListener('mouseleave',function(){{t.classList.remove('vis');t.classList.add('hid')}});
        }});
      }})();
    </script>
    """


def _senate_seat_dist_dots_html(dist: pd.DataFrame, sen_sum: dict) -> str:
    """200 circles arranged in stacked columns like 538's dot distribution."""
    df = dist[dist["probability"] > 0].copy().sort_values("gop_seats")
    total_dots = 200
    # Allocate dots per column proportional to probability
    df["raw_dots"] = df["probability"] * total_dots
    df["dots"] = df["raw_dots"].apply(lambda x: max(round(x), 1) if x > 0.3 else 0)
    # Adjust to hit exactly 200
    diff = total_dots - df["dots"].sum()
    if diff != 0:
        idx_sorted = df.sort_values("raw_dots", ascending=False).index
        for i in range(abs(int(diff))):
            df.loc[idx_sorted[i % len(idx_sorted)], "dots"] += 1 if diff > 0 else -1
    df = df[df["dots"] > 0]

    r = 5  # circle radius
    gap = 2
    step = r * 2 + gap
    col_width = step
    n_cols = len(df)
    max_dots_col = int(df["dots"].max())

    # SVG dimensions
    svg_w = n_cols * col_width + 40  # padding
    svg_h = max_dots_col * step + 60  # room for axis
    base_y = svg_h - 40  # bottom of dot area

    # Center the columns
    total_bar_w = n_cols * col_width
    x_offset = (svg_w - total_bar_w) / 2

    circles = []
    labels = []
    median = float(sen_sum.get("median_gop_seats", 51))
    majority_x = None

    for ci, (_, row) in enumerate(df.iterrows()):
        gop_s = int(row["gop_seats"])
        n_dots = int(row["dots"])
        cx = x_offset + ci * col_width + col_width / 2
        color = GOP_COLOR if gop_s >= 50 else DEM_COLOR

        for di in range(n_dots):
            cy = base_y - di * step
            circles.append(
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{color}" opacity="0.85"/>'
            )

        # Axis label for key values
        if gop_s % 2 == 0 or gop_s in (50, int(median)):
            labels.append(
                f'<text x="{cx:.1f}" y="{base_y + 18}" text-anchor="middle" '
                f'class="al">{gop_s}</text>'
            )

        # Mark the 50-seat line
        if gop_s == 50:
            majority_x = cx

    # Majority marker
    marker_html = ""
    if majority_x is not None:
        marker_html = (
            f'<line x1="{majority_x}" y1="8" x2="{majority_x}" y2="{base_y + 5}" '
            f'stroke="{AMBER}" stroke-width="1.5" stroke-dasharray="4,3"/>'
            f'<text x="{majority_x}" y="6" text-anchor="middle" '
            f'style="font-family:JetBrains Mono,monospace;font-size:9px;font-weight:700;fill:{AMBER}">50</text>'
        )

    return f"""
    <div class="sd" id="sdf">
      <svg viewBox="0 0 {svg_w} {svg_h}" preserveAspectRatio="xMidYMid meet">
        {marker_html}
        {''.join(circles)}
        {''.join(labels)}
        <text x="{x_offset - 4}" y="{base_y + 30}" text-anchor="start"
              style="font-family:JetBrains Mono,monospace;font-size:9px;font-weight:600;fill:{DEM_COLOR}">
          &larr; Dem control</text>
        <text x="{x_offset + total_bar_w + 4}" y="{base_y + 30}" text-anchor="end"
              style="font-family:JetBrains Mono,monospace;font-size:9px;font-weight:600;fill:{GOP_COLOR}">
          GOP control &rarr;</text>
      </svg>
    </div>
    <style>
      @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
      *{{margin:0;padding:0;box-sizing:border-box}}
      body{{overflow:visible;background:{BG};margin:0}}
      .sd{{position:relative}}
      .sd svg{{display:block;width:100%;height:auto;max-width:700px;margin:0 auto}}
      .al{{font-family:'JetBrains Mono',monospace;font-size:9px;font-weight:600;fill:{MUTED}}}
    </style>
    """


def _senate_seat_dist_chart(dist, sen_sum):
    """Kept as fallback but primary is the dot HTML version."""
    df = dist[dist["probability"] > 0].copy()
    df["color"] = df["gop_seats"].map(lambda x: GOP_COLOR if x >= 50 else DEM_COLOR)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["gop_seats"], y=df["probability"], marker=dict(color=df["color"], line=dict(width=0))))
    fig.update_layout(height=240, showlegend=False, bargap=0.1, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=44, r=12, t=28, b=32), font=dict(family="Space Grotesk, sans-serif", color=INK, size=12))
    fig.update_xaxes(showgrid=False, linecolor=RULE, tickfont=dict(size=10, color=SUBTLE))
    fig.update_yaxes(showgrid=True, gridcolor=CHART_GRID, zeroline=False, linecolor=RULE, tickformat=".0%", tickfont=dict(size=10, color=SUBTLE))
    return fig
    return fig


# ── layout ─────────────────────────────────────────────────────────────


# Chamber + theme controls
_ctl_cols = st.columns([3, 1])
with _ctl_cols[0]:
    chamber = st.radio("Chamber", ["House", "Senate"], horizontal=True, label_visibility="collapsed", key="chamber_toggle")
with _ctl_cols[1]:
    if st.button("Light mode" if DARK else "Dark mode", key="theme_toggle"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

if chamber == "House":
    _net_app = summary.get("trump_net_approval", 0)
    _net_app_c = DISAPPROVE_COLOR if _net_app < 0 else APPROVE_COLOR
    st.markdown(
        f"""<div class="intro">
        <p class="intro-kicker">2026 Midterm &mdash; U.S. House Forecast</p>
        <h1 class="intro-title"><em style="color:{lead_color}">{lead_party}</em> are favored to {lead_action} the House</h1>
        <div class="intro-stats">
          <div class="intro-stat"><span class="intro-stat-val" style="color:{lead_color}">{max(gop_chances, dem_chances)}</span><span class="intro-stat-label">in 100 chances</span></div>
          <div class="intro-stat"><span class="intro-stat-val" style="color:{DEM_COLOR}">{dem_seats:.0f}</span><span class="intro-stat-label">expected Dem seats</span></div>
          <div class="intro-stat"><span class="intro-stat-val" style="color:{GOP_COLOR}">{gop_seats:.0f}</span><span class="intro-stat-label">expected GOP seats</span></div>
          <div class="intro-stat"><span class="intro-stat-val" style="color:{INK}">{_margin_text(summary["generic_ballot_margin_dem"])}</span><span class="intro-stat-label">generic ballot</span></div>
          <div class="intro-stat"><span class="intro-stat-val" style="color:{_net_app_c}">{_net_app:+.1f}</span><span class="intro-stat-label">Trump net approval</span></div>
        </div>
        <p class="intro-updated">Updated {summary["as_of_date"]} &middot; {int(summary.get("simulations", 50000)):,} simulations &middot; {int(summary.get("generic_poll_archive_rows", 0))} generic ballot + {int(summary.get("approval_recent_poll_rows", 0))} approval polls</p>
        </div>""",
        unsafe_allow_html=True,
    )
    components.html(_hemicycle_html(districts, summary), height=540, scrolling=False)
    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
    seat_view = st.radio("Seat projection view", ["Expected Seats", "Simulation Distribution"], horizontal=True, label_visibility="collapsed", key="seat_toggle")
    if seat_view == "Expected Seats":
        majority_pct = MAJORITY_SEATS / TOTAL_HOUSE_SEATS * 100
        _seg_html = ""
        for _name, _color, _count in _seg_defs:
            if _count > 0:
                _label = str(_count) if _count / TOTAL_HOUSE_SEATS * 100 > 4 else ""
                _seg_html += f'<div class="segbar-seg" style="flex:{_count};background:{_color}">{_label}</div>'
        _lbl_html = ""
        for _name, _color, _count in _seg_defs:
            if _count > 0:
                _lbl_html += f'<div class="segbar-lbl" style="flex:{_count}"><span class="segbar-lbl-name">{_name}</span></div>'
        st.markdown(
            f'<p class="sec-label">Seat Projection by Rating</p>'
            f'<div class="segbar-wrap"><div class="segbar">{_seg_html}</div>'
            f'<div class="segbar-labels">{_lbl_html}</div>'
            f'<div class="segbar-marker" style="left:{majority_pct:.2f}%"></div></div>'
            f'<div class="segbar-totals"><span style="color:{DEM_COLOR} !important">DEM {dem_seats:.0f}</span>'
            f'<span style="color:{AMBER} !important">218 TO WIN</span>'
            f'<span style="color:{GOP_COLOR} !important">GOP {gop_seats:.0f}</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<p class="sec-label">Seat Distribution &mdash; 50,000 simulations</p>', unsafe_allow_html=True)
        if not seat_dist.empty:
            st.plotly_chart(_seat_distribution_chart(seat_dist, summary), use_container_width=True, key="seat_dist_chart", config={"displayModeBar": False})
        else:
            st.caption("Seat distribution data not available.")
    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
    _house_color_mode = st.radio("Map color", ["Forecast", "Shift vs 2024", "Margin"], horizontal=True, label_visibility="collapsed", key="house_color_mode")

    # Timeline as quarter radio
    import datetime as _dt
    _h_periods = [
        ("Jan '25", _dt.date(2025, 1, 23)),
        ("Apr '25", _dt.date(2025, 4, 1)),
        ("Jul '25", _dt.date(2025, 7, 1)),
        ("Oct '25", _dt.date(2025, 10, 1)),
        ("Jan '26", _dt.date(2026, 1, 1)),
        ("Now", _dt.date(2026, 4, 1)),
    ]
    _h_period_names = [p[0] for p in _h_periods]
    _h_sel_period = st.radio("Timeline", _h_period_names, index=len(_h_periods) - 1, horizontal=True, label_visibility="collapsed", key="house_timeline")
    _h_sel_date = dict(_h_periods)[_h_sel_period]
    _h_dates_all = sorted(dist_history["as_of_date"].dt.date.unique()) if not dist_history.empty else [_dt.date(2026, 4, 1)]
    _h_date = max([d for d in _h_dates_all if d <= _h_sel_date], default=_h_dates_all[-1])
    _h_max = _h_dates_all[-1]

    if _h_date < _h_max:
        _h_date_str = _h_date.strftime("%Y-%m-%d")
        _h_snap = dist_history[dist_history["as_of_date"].dt.strftime("%Y-%m-%d") == _h_date_str] if not dist_history.empty else pd.DataFrame()
        if not _h_snap.empty:
            _h_dem_wins = int((_h_snap["dem_win_prob"] > 0.5).sum())
            _h_gop_wins = TOTAL_HOUSE_SEATS - _h_dem_wins
            st.markdown(
                f'<div style="padding:0.4rem 0.6rem;border-left:3px solid {AMBER};margin:0.4rem 0 0.5rem;font-size:0.65rem">'
                f'<span style="color:{AMBER};font-weight:700">{_h_date.strftime("%b %d, %Y")}</span>'
                f'<span style="color:{MUTED};margin:0 0.4rem">&mdash;</span>'
                f'<span style="font-family:JetBrains Mono,monospace;font-weight:700">'
                f'<span style="color:{DEM_COLOR}">D {_h_dem_wins}</span>'
                f' <span style="color:{MUTED}">/</span> '
                f'<span style="color:{GOP_COLOR}">R {_h_gop_wins}</span></span></div>',
                unsafe_allow_html=True,
            )

    # Determine which district data to use (current or historical)
    _h_use_districts = districts
    if not dist_history.empty and _h_date < _h_max:
        _h_date_str = _h_date.strftime("%Y-%m-%d")
        _h_snap_full = dist_history[dist_history["as_of_date"].dt.strftime("%Y-%m-%d") == _h_date_str]
        if not _h_snap_full.empty:
            # Map historical columns to what _district_map_html expects
            _h_map_df = _h_snap_full.copy()
            _h_map_df["mean_margin_sim"] = _h_map_df["expected_dem_margin"]
            _h_map_df["dem_win_prob"] = _h_map_df["dem_win_prob"]
            _h_map_df["gop_win_prob"] = _h_map_df["gop_win_prob"]
            # Fill in columns from current districts that history doesn't have
            _curr_cols = ["district_name", "open_seat", "rating", "winner_party_2024", "pres24_dem_margin", "house_margin_2024", "dem_candidate", "rep_candidate"]
            _curr_lookup = districts.set_index("district_code")[_curr_cols].to_dict("index")
            for _col in _curr_cols:
                if _col not in _h_map_df.columns:
                    _h_map_df[_col] = _h_map_df["district_code"].map(lambda dc, c=_col: _curr_lookup.get(dc, {}).get(c))
            _h_use_districts = _h_map_df

    st.markdown(
        f'<p class="sec-label">District Map{" — " + _h_date.strftime("%b %d, %Y") if _h_date < _h_max else ""}</p>'
        f'<p class="sec-note" style="margin-bottom:1rem">Hover any district for forecast details.</p>',
        unsafe_allow_html=True,
    )
    components.html(_district_map_html(_h_use_districts, district_geojson, color_mode=_house_color_mode), height=900, scrolling=False)
    _gb_margin = summary.get("generic_ballot_margin_dem", 0)
    _gb_dem = 50.0 + _gb_margin / 2.0
    _gb_gop = 50.0 - _gb_margin / 2.0
    st.markdown(
        f'<div class="rule"></div><p class="sec-label">Generic Ballot</p><p class="sec-note">'
        f'<span style="color:{DEM_COLOR} !important">Dem {_gb_dem:.1f}%</span> &nbsp;/&nbsp; '
        f'<span style="color:{GOP_COLOR} !important">GOP {_gb_gop:.1f}%</span>'
        f' &nbsp;&middot;&nbsp; Margin: D{_gb_margin:+.1f}'
        f' &nbsp;&middot;&nbsp; {int(summary.get("generic_poll_archive_rows", 0))} polls</p>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(_generic_ballot_chart(history, gb_polls), use_container_width=True, key="gb_chart", config={"displayModeBar": False})
    if not approval_data.empty:
        _net = summary.get("trump_net_approval", 0)
        _app = summary.get("trump_approve_pct", 0)
        _dis = summary.get("trump_disapprove_pct", 0)
        st.markdown(
            f'<div class="rule"></div><p class="sec-label">Trump Job Approval</p><p class="sec-note">'
            f'<span style="color:{APPROVE_COLOR} !important">{_app:.1f}% approve</span> &nbsp;/&nbsp; '
            f'<span style="color:{DISAPPROVE_COLOR} !important">{_dis:.1f}% disapprove</span>'
            f' &nbsp;&middot;&nbsp; Net: <span style="color:{DISAPPROVE_COLOR if _net < 0 else APPROVE_COLOR} !important">{_net:+.1f}</span>'
            f' &nbsp;&middot;&nbsp; {int(summary.get("approval_recent_poll_rows", 0))} approval polls</p>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(_approval_chart(approval_data, approval_polls), use_container_width=True, key="approval_chart", config={"displayModeBar": False})
    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
    time_view = st.radio("Over time view", ["Control Odds", "Seat Aggregate"], horizontal=True, label_visibility="collapsed", key="time_toggle")
    if time_view == "Control Odds":
        st.markdown('<p class="sec-label">Control Probability Over Time</p>', unsafe_allow_html=True)
        st.plotly_chart(_odds_chart(history), use_container_width=True, key="odds_chart", config={"displayModeBar": False})
    else:
        st.markdown('<p class="sec-label">Expected Seats Over Time</p>', unsafe_allow_html=True)
        st.plotly_chart(_seat_aggregate_chart(history), use_container_width=True, key="seat_agg_chart", config={"displayModeBar": False})
    # District detail explorer
    if not dist_history.empty:
        st.markdown('<div class="rule"></div><p class="sec-label">District Explorer</p>', unsafe_allow_html=True)
        _dist_codes = sorted(districts["district_code"].unique().tolist())
        _col_sel, _col_tog = st.columns([1, 1])
        with _col_sel:
            _sel_dist = st.selectbox("District", ["Select a district..."] + _dist_codes, index=0, label_visibility="collapsed", key="house_dist_detail")
        if _sel_dist != "Select a district...":
            _dh = dist_history[dist_history["district_code"] == _sel_dist].sort_values("as_of_date")
            _dinfo = districts[districts["district_code"] == _sel_dist]
            if not _dh.empty and not _dinfo.empty:
                _di = _dinfo.iloc[0]
                _dname = str(_di.get("district_name", _sel_dist))
                _dm = float(_di["mean_margin_sim"])
                _dfc = _margin_fill(_dm)
                _dem_cand_raw = _di.get("dem_candidate", "")
                _dem_cand = str(_dem_cand_raw) if pd.notna(_dem_cand_raw) and str(_dem_cand_raw).strip() else "TBD"
                _gop_cand_raw = _di.get("rep_candidate", "")
                _gop_cand = str(_gop_cand_raw) if pd.notna(_gop_cand_raw) and str(_gop_cand_raw).strip() else "TBD"
                _d_rating = str(_di.get("rating", "")) or ""
                _d_dem_p = float(_di["dem_win_prob"])
                _d_gop_p = float(_di["gop_win_prob"])
                st.markdown(
                    f'<div style="margin:0.6rem 0 0.8rem;padding:0.8rem;border:1px solid {RULE};border-radius:6px">'
                    f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem">'
                    f'<div><span style="font-family:JetBrains Mono,monospace;font-size:1.1rem;font-weight:800;color:{_dfc}">{_sel_dist}</span>'
                    f'<span style="font-size:0.72rem;color:{MUTED};margin-left:0.5rem">{_dname}</span></div>'
                    f'<span style="font-family:JetBrains Mono,monospace;font-size:0.8rem;font-weight:700;color:{_dfc}">{_margin_text(_dm)}</span></div>'
                    f'<div style="display:flex;gap:0.3rem;font-size:0.7rem;margin-bottom:0.6rem">'
                    f'<span style="color:{DEM_COLOR};font-weight:600">{escape(_dem_cand)}</span>'
                    f'<span style="color:{MUTED}">vs</span>'
                    f'<span style="color:{GOP_COLOR};font-weight:600">{escape(_gop_cand)}</span>'
                    f'{"<span style=font-size:0.6rem;color:" + MUTED + ">" + _d_rating + "</span>" if _d_rating else ""}'
                    f'</div>'
                    f'<div style="display:flex;height:20px;border-radius:3px;overflow:hidden;gap:1px">'
                    f'<div style="flex:{_d_dem_p*100:.1f};background:{DEM_COLOR};font-size:9px;display:flex;align-items:center;justify-content:center;color:white;font-weight:700">D {_d_dem_p:.0%}</div>'
                    f'<div style="flex:{_d_gop_p*100:.1f};background:{GOP_COLOR};font-size:9px;display:flex;align-items:center;justify-content:center;color:white;font-weight:700">R {_d_gop_p:.0%}</div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )
                _dv = st.radio("View", ["Win Probability", "Expected Margin"], horizontal=True, label_visibility="collapsed", key="dist_detail_view")
                _dfig = go.Figure()
                if _dv == "Win Probability":
                    _dfig.add_trace(go.Scatter(x=_dh["as_of_date"], y=_dh["dem_win_prob"], mode="lines", name="Dem", line=dict(color=DEM_COLOR, width=2.5), hovertemplate="%{x|%b %d, %Y}<br>Dem: %{y:.1%}<extra></extra>"))
                    _dfig.add_trace(go.Scatter(x=_dh["as_of_date"], y=_dh["gop_win_prob"], mode="lines", name="GOP", line=dict(color=GOP_COLOR, width=2.5), hovertemplate="%{x|%b %d, %Y}<br>GOP: %{y:.1%}<extra></extra>"))
                    _dfig.add_hline(y=0.5, line_dash="dot", line_color="#333338", line_width=1)
                    _dfig.update_yaxes(tickformat=".0%", range=[0, 1], tickvals=[0, 0.25, 0.5, 0.75, 1.0])
                else:
                    _dh["dem_pct_est"] = 50.0 + _dh["expected_dem_margin"] / 2.0
                    _dh["gop_pct_est"] = 50.0 - _dh["expected_dem_margin"] / 2.0
                    _dfig.add_trace(go.Scatter(x=_dh["as_of_date"], y=_dh["dem_pct_est"], mode="lines", name="Dem %", line=dict(color=DEM_COLOR, width=2.5), hovertemplate="%{x|%b %d, %Y}<br>Dem: %{y:.1f}%<extra></extra>"))
                    _dfig.add_trace(go.Scatter(x=_dh["as_of_date"], y=_dh["gop_pct_est"], mode="lines", name="GOP %", line=dict(color=GOP_COLOR, width=2.5), hovertemplate="%{x|%b %d, %Y}<br>GOP: %{y:.1f}%<extra></extra>"))
                    _dfig.add_hline(y=50, line_dash="dot", line_color="#333338", line_width=1)
                    _dfig.update_yaxes(ticksuffix="%")
                _dfig.update_layout(height=250, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=44, r=12, t=10, b=32), hovermode="x unified", font=dict(family="Space Grotesk, sans-serif", color=INK, size=12), legend=dict(orientation="h", y=1.1, x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=11, color=MUTED)), hoverlabel=dict(bgcolor=TOOLTIP_BG, bordercolor=TOOLTIP_BORDER, namelength=-1, font=dict(color=INK, family="Space Grotesk, sans-serif", size=13)))
                _dfig.update_xaxes(showgrid=False, linecolor=RULE, tickfont=dict(size=10, color=SUBTLE), tickformat="%b '%y")
                _dfig.update_yaxes(showgrid=True, gridcolor=CHART_GRID, zeroline=False, linecolor=RULE, tickfont=dict(size=10, color=SUBTLE))
                st.plotly_chart(_dfig, use_container_width=True, key="dist_detail_chart", config={"displayModeBar": False})

    st.markdown(
        f'<p class="foot">Generic-ballot polls: {int(summary.get("generic_poll_archive_rows", 0))}'
        f' &middot; Approval polls: {int(summary.get("approval_recent_poll_rows", 0))}'
        f' &middot; Simulations: {int(summary.get("simulations", 50000)):,}</p>',
        unsafe_allow_html=True,
    )

else:
    # ── SENATE VIEW ──────────────────────────────────────────────────────
    _sgop = float(sen_summary.get("expected_gop_seats", 51))
    _sdem = float(sen_summary.get("expected_dem_seats", 49))
    _sgop_prob = float(sen_summary.get("gop_control_prob", 0.5))
    _sdem_prob = float(sen_summary.get("dem_control_prob", 0.5))
    _stie = float(sen_summary.get("tied_chamber_prob", 0))
    _sgop_ch = round(_sgop_prob * 100)
    _sdem_ch = round(_sdem_prob * 100)
    _sfav = _sgop_prob > _sdem_prob
    _slead = "Republicans" if _sfav else "Democrats"
    _slead_c = GOP_COLOR if _sfav else DEM_COLOR

    st.markdown(
        f"""<div class="intro">
        <p class="intro-kicker">2026 Midterm &mdash; U.S. Senate Forecast</p>
        <h1 class="intro-title"><em style="color:{_slead_c}">{_slead}</em> are favored to control the Senate</h1>
        <div class="intro-stats">
          <div class="intro-stat"><span class="intro-stat-val" style="color:{_slead_c}">{max(_sgop_ch, _sdem_ch)}</span><span class="intro-stat-label">in 100 chances</span></div>
          <div class="intro-stat"><span class="intro-stat-val" style="color:{DEM_COLOR}">{round(_sdem)}</span><span class="intro-stat-label">expected Dem seats</span></div>
          <div class="intro-stat"><span class="intro-stat-val" style="color:{GOP_COLOR}">{round(_sgop)}</span><span class="intro-stat-label">expected GOP seats</span></div>
          <div class="intro-stat"><span class="intro-stat-val" style="color:{AMBER}">{_stie:.0%}</span><span class="intro-stat-label">50-50 tie chance</span></div>
          <div class="intro-stat"><span class="intro-stat-val" style="color:{INK}">{int(sen_summary.get("race_count", 35))}</span><span class="intro-stat-label">races in 2026</span></div>
        </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # Color mode + timeline
    _sen_color_mode = st.radio("Color mode", ["Forecast", "Shift vs 2024", "Margin"], horizontal=True, label_visibility="collapsed", key="sen_color_mode")
    import datetime as _dt
    _s_periods = [
        ("Jan '25", _dt.date(2025, 1, 23)),
        ("Apr '25", _dt.date(2025, 4, 1)),
        ("Jul '25", _dt.date(2025, 7, 1)),
        ("Oct '25", _dt.date(2025, 10, 1)),
        ("Jan '26", _dt.date(2026, 1, 1)),
        ("Now", _dt.date(2026, 4, 1)),
    ]
    _s_period_names = [p[0] for p in _s_periods]
    _s_sel_period = st.radio("Timeline", _s_period_names, index=len(_s_periods) - 1, horizontal=True, label_visibility="collapsed", key="sen_timeline")
    _s_sel_date = dict(_s_periods)[_s_sel_period]
    _s_dates_all = sorted(sen_race_history["as_of_date"].dt.date.unique()) if not sen_race_history.empty else [_dt.date(2026, 4, 1)]
    _s_date = max([d for d in _s_dates_all if d <= _s_sel_date], default=_s_dates_all[-1])
    _s_max = _s_dates_all[-1]

    if _s_date < _s_max:
        _s_date_str = _s_date.strftime("%Y-%m-%d")
        _s_snap = sen_race_history[sen_race_history["as_of_date"].dt.strftime("%Y-%m-%d") == _s_date_str] if not sen_race_history.empty else pd.DataFrame()
        if not _s_snap.empty:
            _s_dem_wins = int((_s_snap["dem_win_prob"] > 0.5).sum())
            _s_not_up_dem = int(sen_summary.get("starting_dem_not_up", 34))
            _s_total_dem = _s_not_up_dem + _s_dem_wins
            _s_total_gop = 100 - _s_total_dem
            st.markdown(
                f'<div style="padding:0.4rem 0.6rem;border-left:3px solid {AMBER};margin:0.4rem 0 0.5rem;font-size:0.65rem">'
                f'<span style="color:{AMBER};font-weight:700">{_s_date.strftime("%b %d, %Y")}</span>'
                f'<span style="color:{MUTED};margin:0 0.4rem">&mdash;</span>'
                f'<span style="font-family:JetBrains Mono,monospace;font-weight:700">'
                f'<span style="color:{DEM_COLOR}">D {_s_total_dem}</span>'
                f' <span style="color:{MUTED}">/</span> '
                f'<span style="color:{GOP_COLOR}">R {_s_total_gop}</span></span></div>',
                unsafe_allow_html=True,
            )

    def _sen_seat_color(row, mode):
        margin = float(row["expected_dem_margin"])
        if mode == "Forecast":
            return _margin_fill(margin)
        elif mode == "Shift vs 2024":
            pres24 = float(row.get("state_pres_dem_margin_2024", 0) or 0)
            shift = margin - pres24  # positive = shifted toward Dem
            sa = abs(shift)
            if shift >= 0:  # shifted toward Dem (blue)
                if sa > 10: return "#1D5CB4"
                if sa > 5: return "#4A90D9"
                return "#82B8E8"
            else:  # shifted toward GOP (red)
                if sa > 10: return "#C12A1C"
                if sa > 5: return "#E05C4F"
                return "#E89990"
        else:
            m = abs(margin)
            if margin >= 0:
                if m > 15: return "#0E3F5E"
                if m > 8: return DEM_COLOR
                if m > 3: return "#6AAAD4"
                return "#A8CBE3"
            else:
                if m > 15: return "#6E1E18"
                if m > 8: return GOP_COLOR
                if m > 3: return "#D8877E"
                return "#E8B8B2"

    # Use historical data if timeline is set to past
    _sen_use_races = sen_races
    if not sen_race_history.empty and _s_date < _s_max:
        _s_date_str = _s_date.strftime("%Y-%m-%d")
        _s_snap_full = sen_race_history[sen_race_history["as_of_date"].dt.strftime("%Y-%m-%d") == _s_date_str]
        if not _s_snap_full.empty:
            _sen_use_races = _s_snap_full.copy()

    # Seat bar with flip indicators
    _sen_sorted = _sen_use_races.sort_values("expected_dem_margin", ascending=False)
    _not_up_dem = int(sen_summary.get("starting_dem_not_up", 34))
    _not_up_gop = int(sen_summary.get("starting_gop_not_up", 31))
    _stripe = f"repeating-linear-gradient(135deg,transparent,transparent 2px,rgba(255,255,255,0.25) 2px,rgba(255,255,255,0.25) 4px)"
    _bar_segs = f'<div style="flex:{_not_up_dem};background:{DEM_COLOR};opacity:0.2" title="{_not_up_dem} safe Dem"></div>'
    for _, _sr in _sen_sorted.iterrows():
        _sc = _sen_seat_color(_sr, _sen_color_mode)
        _flipped = str(_sr.get("current_party", "")) != str(_sr.get("projected_winner", ""))
        _bg = f"background:{_sc}" if not _flipped else f"background:{_stripe},{_sc}"
        _bar_segs += f'<div style="flex:1;{_bg}" title="{_sr["state_abbr"]}{"*" if _flipped else ""}"></div>'
    _bar_segs += f'<div style="flex:{_not_up_gop};background:{GOP_COLOR};opacity:0.2" title="{_not_up_gop} safe GOP"></div>'
    _flip_count = int((sen_races["current_party"] != sen_races["projected_winner"]).sum())
    st.markdown(
        f"""<div style="position:relative;margin:0.6rem 0 2.2rem">
        <div style="display:flex;height:26px;border-radius:4px;overflow:hidden;gap:1px">{_bar_segs}</div>
        <div style="position:absolute;top:-5px;bottom:-20px;width:2px;background:{AMBER};left:50%"></div>
        <div style="position:absolute;bottom:-16px;left:50%;transform:translateX(-50%);font-family:'JetBrains Mono',monospace;font-size:0.52rem;font-weight:700;color:{AMBER}">50</div>
        <div style="display:flex;justify-content:space-between;align-items:center;font-family:'JetBrains Mono',monospace;font-size:0.6rem;font-weight:600;margin-top:1.2rem">
          <span style="color:{DEM_COLOR}">DEM {round(_sdem)}</span>
          <span style="font-size:0.5rem;color:{MUTED};font-weight:500"><span style="display:inline-block;width:10px;height:10px;background:{_stripe},{MUTED};border-radius:2px;vertical-align:middle;margin-right:3px"></span>= projected flip ({_flip_count})</span>
          <span style="color:{GOP_COLOR}">GOP {round(_sgop)}</span>
        </div></div>""",
        unsafe_allow_html=True,
    )

    # Senate state map — uses same color mode
    _sen_geo_path = ROOT / "data/processed/us_states.geojson"
    _sen_geo = json.loads(_sen_geo_path.read_text()) if _sen_geo_path.exists() else {}
    if _sen_geo and not sen_races.empty:
        _sen_lookup = _sen_use_races.set_index("state_abbr").to_dict("index")
        _swidth, _sheight = 960, 600
        _slat0 = math.radians(37.0)
        _all_pts = []
        for _feat in _sen_geo.get("features", []):
            for _lon, _lat in _iter_coords(_feat.get("geometry", {}).get("coordinates", [])):
                if -130 < _lon < -60:
                    _all_pts.append((_lon * math.cos(_slat0), _lat))
        if _all_pts:
            _mnx, _mxx = min(x for x, _ in _all_pts), max(x for x, _ in _all_pts)
            _mny, _mxy = min(y for _, y in _all_pts), max(y for _, y in _all_pts)
            _sscale = min((_swidth - 40) / max(_mxx - _mnx, 1e-9), (_sheight - 40) / max(_mxy - _mny, 1e-9))
            _scw, _sch = (_mxx - _mnx) * _sscale, (_mxy - _mny) * _sscale
            _sox, _soy = (_swidth - _scw) / 2, (_sheight - _sch) / 2

            def _sproj(_lon, _lat):
                return (_lon * math.cos(_slat0) - _mnx) * _sscale + _sox, _sheight - ((_lat - _mny) * _sscale + _soy)

            _spaths = []
            for _feat in _sen_geo.get("features", []):
                _props = _feat.get("properties", {})
                _abbr = _props.get("abbr", "")
                _sname = _props.get("name", "")
                if _abbr in ("AK", "HI", "DC", "PR", ""):
                    continue
                _race = _sen_lookup.get(_abbr)
                if _race:
                    _sfill = _sen_seat_color(_race, _sen_color_mode)
                    _dp = float(_race["dem_win_prob"])
                    _sflipped = str(_race.get("current_party", "")) != str(_race.get("projected_winner", ""))
                    _flip_tag = "<span style='color:#FFD166;font-size:10px;font-weight:700'> FLIP</span>" if _sflipped else ""
                    _stip = (
                        f"<div class='tc'>{_abbr} &mdash; {_sname}{_flip_tag}</div>"
                        f"<div class='tr' style='color:{DEM_COLOR if _dp > 0.5 else GOP_COLOR}'>"
                        f"{'Dem' if _dp > 0.5 else 'GOP'} {max(_dp, 1-_dp):.0%}</div>"
                        f"<div class='tn'>{escape(str(_race.get('top_dem_candidate', '?')))} vs {escape(str(_race.get('top_rep_candidate', '?')))}</div>"
                        f"<div class='tg'>"
                        f"<span>Dem</span><strong style='color:{DEM_COLOR}'>{_dp:.1%}</strong>"
                        f"<span>GOP</span><strong style='color:{GOP_COLOR}'>{1-_dp:.1%}</strong>"
                        f"<span>Margin</span><strong>{_margin_text(float(_race['expected_dem_margin']))}</strong>"
                        f"<span>Rating</span><strong>{str(_race.get('rating_category','')).replace('_',' ').title()}</strong>"
                        f"</div>"
                    )
                else:
                    _sfill = "#1A1A1F"
                    _stip = f"<div class='tc'>{_abbr} &mdash; {_sname}</div><div class='tn'>Not up in 2026</div>"

                _geo = _feat.get("geometry", {})
                _sring_paths = []
                for _ring in _iter_rings(_geo):
                    _pts = [_sproj(float(_lo), float(_la)) for _lo, _la in _ring if -130 < float(_lo) < -60]
                    if _pts:
                        _sring_paths.append("M " + " L ".join(f"{_x:.1f} {_y:.1f}" for _x, _y in _pts) + " Z")
                _sd = " ".join(_sring_paths)
                if _sd:
                    _opacity = "1" if _race else "0.35"
                    _use_fill = _sfill
                    if _race and _sflipped:
                        _pid = f"stripe_{_abbr}"
                        _spaths.append(
                            f'<pattern id="{_pid}" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(135)">'
                            f'<rect width="6" height="6" fill="{_sfill}"/>'
                            f'<rect width="3" height="6" fill="rgba(255,255,255,0.2)"/></pattern>'
                        )
                        _use_fill = f"url(#{_pid})"
                    _spaths.append(
                        f'<path class="ss" d="{_sd}" fill="{_use_fill}" opacity="{_opacity}" '
                        f'data-tip="{escape(_stip, quote=True)}"/>'
                    )

            _defs = [p for p in _spaths if p.startswith("<pattern")]
            _paths_only = [p for p in _spaths if not p.startswith("<pattern")]
            _smap_html = f"""
            <div class="sm" id="smf">
              <svg viewBox="0 0 {_swidth} {_sheight}">
                <defs>{''.join(_defs)}</defs>
                <rect width="{_swidth}" height="{_sheight}" fill="{BG}" rx="8"/>
                {''.join(_paths_only)}
              </svg>
              <div id="stt" class="tt hid"></div>
            </div>
            <style>
              @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
              *{{margin:0;padding:0;box-sizing:border-box;font-family:'Space Grotesk',sans-serif}}
              body{{overflow:visible;background:{BG};margin:0}}
              .sm{{position:relative;padding-bottom:30px}}.sm svg{{display:block;width:100%;height:auto}}
              .ss{{stroke:{BG};stroke-width:0.8;cursor:pointer;transition:filter 80ms ease,stroke 80ms ease}}
              .ss:hover{{filter:brightness(1.2);stroke:{INK};stroke-width:1.2}}
              .tt{{position:absolute;pointer-events:none;width:230px;background:{TOOLTIP_BG};border:1px solid {TOOLTIP_BORDER};border-radius:6px;box-shadow:0 8px 28px rgba(0,0,0,0.6);padding:10px 12px 8px;z-index:10;opacity:0;transition:opacity 80ms ease}}
              .tt.vis{{opacity:1}}.tt.hid{{display:none}}
              .tc{{font-size:14px;font-weight:700;color:{INK};margin-bottom:2px}}
              .tn{{font-size:10.5px;color:{MUTED};margin-bottom:6px}}
              .tr{{font-size:13px;font-weight:600;margin-bottom:6px;padding-bottom:5px;border-bottom:1px solid {TOOLTIP_BORDER}}}
              .tg{{display:grid;grid-template-columns:auto auto;gap:1px 10px;font-size:11px;line-height:1.65}}
              .tg span{{color:{MUTED}}}.tg strong{{text-align:right;font-weight:600;color:{INK}}}
            </style>
            <script>
              (function(){{var f=document.getElementById('smf'),t=document.getElementById('stt');if(!f||!t)return;f.querySelectorAll('.ss').forEach(function(n){{n.addEventListener('mousemove',function(e){{t.innerHTML=n.dataset.tip;t.classList.remove('hid');t.classList.add('vis');var r=f.getBoundingClientRect(),l=e.clientX-r.left+14,tp=e.clientY-r.top-30;if(l+240>r.width)l=e.clientX-r.left-245;if(tp<0)tp=10;t.style.left=l+'px';t.style.top=tp+'px'}});n.addEventListener('mouseleave',function(){{t.classList.remove('vis');t.classList.add('hid')}})}})}})();(function(){{function rz(){{try{{if(window.frameElement)window.frameElement.style.height=document.body.scrollHeight+'px'}}catch(e){{}}}}rz();window.addEventListener('load',rz);new ResizeObserver(rz).observe(document.body)}})()</script>
            """
            _s_map_title = f'Senate Race Map{" — " + _s_date.strftime("%b %d, %Y") if _s_date < _s_max else ""}'
            st.markdown(f'<p class="sec-label">{_s_map_title}</p><p class="sec-note" style="margin-bottom:1.2rem">Hover any state for the race forecast. Dimmed states have no race in 2026.</p>', unsafe_allow_html=True)
            components.html(_smap_html, height=660, scrolling=False)

    # Race table helper
    def _race_row_html(_rr):
        _dp = float(_rr["dem_win_prob"])
        _rp = 1 - _dp
        _dm = float(_rr["expected_dem_margin"])
        _fc = _margin_fill(_dm)
        _dpct = _dp * 100
        _abbr = _rr["state_abbr"]
        _sname = str(_rr.get("state_name", ""))
        _rating = str(_rr.get("rating_category", "")).replace("_", " ").title()
        _dcand = str(_rr.get("top_dem_candidate", "TBD"))
        _rcand = str(_rr.get("top_rep_candidate", "TBD"))
        _holder = str(_rr.get("current_party", ""))
        _open = bool(_rr.get("open_seat", False))
        _special = bool(_rr.get("special", False))
        if _special:
            _tag = f'<span style="font-size:0.52rem;background:{AMBER};color:{BG};padding:1px 4px;border-radius:2px;font-weight:700;margin-left:0.3rem">SPECIAL</span>'
        elif _open:
            _tag = f'<span style="font-size:0.52rem;background:{RULE};color:{MUTED};padding:1px 4px;border-radius:2px;font-weight:700;margin-left:0.3rem">OPEN</span>'
        else:
            _ic = GOP_COLOR if _holder == "REP" else DEM_COLOR
            _tag = f'<span style="font-size:0.52rem;background:{"rgba(255,255,255,0.06)" if DARK else "rgba(0,0,0,0.04)"};color:{_ic};padding:1px 4px;border-radius:2px;font-weight:700;margin-left:0.3rem">{_holder} INC</span>'
        _polls = _sen_poll_counts.get(_abbr, 0)
        _poll_str = f'{_polls} poll{"s" if _polls != 1 else ""}' if _polls > 0 else "No polls"
        return (
            f'<div class="race-row">'
            f'<div style="text-align:center"><div style="font-family:JetBrains Mono,monospace;font-size:0.8rem;font-weight:800;color:{_fc};line-height:1.1">{_abbr}</div>'
            f'<div style="font-size:0.48rem;color:{MUTED};font-weight:600;text-transform:uppercase;line-height:1.3">{_rating}</div></div>'
            f'<div><span style="font-size:0.7rem;color:{DEM_COLOR};font-weight:600">{escape(_dcand)}</span>'
            f'<span style="font-size:0.58rem;color:{MUTED}"> v </span>'
            f'<span style="font-size:0.7rem;color:{GOP_COLOR};font-weight:600">{escape(_rcand)}</span>'
            f'{_tag}<br><span style="font-size:0.58rem;color:{MUTED}">{_sname}</span>'
            f'<span style="font-size:0.52rem;color:{SUBTLE};margin-left:0.3rem">{_poll_str}</span></div>'
            f'<div style="text-align:right;font-family:JetBrains Mono,monospace;font-size:0.78rem;font-weight:700;color:{_fc}">{_margin_text(_dm)}</div>'
            f'<div style="display:flex;height:18px;border-radius:3px;overflow:hidden;gap:1px">'
            f'<div style="flex:{_dpct:.1f};background:{DEM_COLOR};font-size:8px;display:flex;align-items:center;justify-content:center;color:white;font-weight:700">{_dp:.0%}</div>'
            f'<div style="flex:{100-_dpct:.1f};background:{GOP_COLOR};font-size:8px;display:flex;align-items:center;justify-content:center;color:white;font-weight:700">{_rp:.0%}</div>'
            f'</div></div>'
        )

    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)

    if not sen_races.empty:
        _all_races = sen_races.copy()
        _all_races["_comp"] = (_all_races["dem_win_prob"] - 0.5).abs()
        _all_races = _all_races.sort_values("_comp")

        st.markdown(f'<p class="sec-label">All {len(_all_races)} Races</p>', unsafe_allow_html=True)
        _race_search = st.text_input("Filter", placeholder="State, candidate, or rating...", label_visibility="collapsed", key="race_search")
        _filtered = _all_races
        if _race_search:
            _q = _race_search.lower()
            _filtered = _all_races[
                _all_races.apply(lambda r: _q in str(r.get("state_abbr","")).lower()
                    or _q in str(r.get("state_name","")).lower()
                    or _q in str(r.get("top_dem_candidate","")).lower()
                    or _q in str(r.get("top_rep_candidate","")).lower()
                    or _q in str(r.get("rating_category","")).lower().replace("_"," "), axis=1)
            ]
        _tbl = ""
        for _, _rr in _filtered.iterrows():
            _tbl += _race_row_html(_rr)
        if _tbl:
            st.markdown(_tbl, unsafe_allow_html=True)
        else:
            st.markdown(f'<p style="font-size:0.72rem;color:{MUTED};padding:0.5rem 0">No races match.</p>', unsafe_allow_html=True)

    # Race detail: odds over time for a selected state
    if not sen_race_history.empty:
        st.markdown('<div class="rule"></div><p class="sec-label">Race Explorer</p>', unsafe_allow_html=True)
        _race_options = ["Select a race..."] + sorted(sen_races["state_abbr"].unique().tolist(), key=lambda x: float(sen_races[sen_races["state_abbr"] == x]["dem_win_prob"].iloc[0] - 0.5).__abs__())
        _scol_sel, _scol_tog = st.columns([1, 1])
        with _scol_sel:
            _sel_race = st.selectbox("Race", _race_options, index=0, label_visibility="collapsed", key="sen_race_detail")
        if _sel_race != "Select a race...":
            _rh = sen_race_history[sen_race_history["state_abbr"] == _sel_race].sort_values("as_of_date")
            _rinfo = sen_races[sen_races["state_abbr"] == _sel_race].iloc[0] if not sen_races[sen_races["state_abbr"] == _sel_race].empty else None
            if not _rh.empty and _rinfo is not None:
                _rdm = float(_rinfo["expected_dem_margin"])
                _rfc = _margin_fill(_rdm)
                _rdp = float(_rinfo["dem_win_prob"])
                _rrp = 1 - _rdp
                _rcand_d = str(_rinfo.get("top_dem_candidate", "?"))
                _rcand_r = str(_rinfo.get("top_rep_candidate", "?"))
                _rrating = str(_rinfo.get("rating_category", "")).replace("_", " ").title()
                _rstate = str(_rinfo.get("state_name", ""))
                st.markdown(
                    f'<div style="margin:0.6rem 0 0.8rem;padding:0.8rem;border:1px solid {RULE};border-radius:6px">'
                    f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.5rem">'
                    f'<div><span style="font-family:JetBrains Mono,monospace;font-size:1.1rem;font-weight:800;color:{_rfc}">{_sel_race}</span>'
                    f'<span style="font-size:0.72rem;color:{MUTED};margin-left:0.5rem">{_rstate}</span>'
                    f'<span style="font-size:0.58rem;color:{SUBTLE};margin-left:0.4rem">{_rrating}</span></div>'
                    f'<span style="font-family:JetBrains Mono,monospace;font-size:0.8rem;font-weight:700;color:{_rfc}">{_margin_text(_rdm)}</span></div>'
                    f'<div style="display:flex;gap:0.3rem;font-size:0.7rem;margin-bottom:0.6rem">'
                    f'<span style="color:{DEM_COLOR};font-weight:600">{escape(_rcand_d)}</span>'
                    f'<span style="color:{MUTED}">vs</span>'
                    f'<span style="color:{GOP_COLOR};font-weight:600">{escape(_rcand_r)}</span></div>'
                    f'<div style="display:flex;height:20px;border-radius:3px;overflow:hidden;gap:1px">'
                    f'<div style="flex:{_rdp*100:.1f};background:{DEM_COLOR};font-size:9px;display:flex;align-items:center;justify-content:center;color:white;font-weight:700">D {_rdp:.0%}</div>'
                    f'<div style="flex:{_rrp*100:.1f};background:{GOP_COLOR};font-size:9px;display:flex;align-items:center;justify-content:center;color:white;font-weight:700">R {_rrp:.0%}</div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )
                _rv = st.radio("View", ["Win Probability", "Expected Margin"], horizontal=True, label_visibility="collapsed", key="race_detail_view")
                _rfig = go.Figure()
                if _rv == "Win Probability":
                    _rfig.add_trace(go.Scatter(x=_rh["as_of_date"], y=_rh["dem_win_prob"], mode="lines", name="Dem win prob", line=dict(color=DEM_COLOR, width=2.5), hovertemplate="%{x|%b %d, %Y}<br>Dem: %{y:.1%}<extra></extra>"))
                    _rfig.add_trace(go.Scatter(x=_rh["as_of_date"], y=_rh["rep_win_prob"], mode="lines", name="GOP win prob", line=dict(color=GOP_COLOR, width=2.5), hovertemplate="%{x|%b %d, %Y}<br>GOP: %{y:.1%}<extra></extra>"))
                    _rfig.add_hline(y=0.5, line_dash="dot", line_color="#333338", line_width=1)
                    _rfig.update_yaxes(tickformat=".0%", range=[0, 1], tickvals=[0, 0.25, 0.5, 0.75, 1.0])
                else:
                    _rh["dem_pct_est"] = 50.0 + _rh["expected_dem_margin"] / 2.0
                    _rh["gop_pct_est"] = 50.0 - _rh["expected_dem_margin"] / 2.0
                    _rfig.add_trace(go.Scatter(x=_rh["as_of_date"], y=_rh["dem_pct_est"], mode="lines", name="Dem %", line=dict(color=DEM_COLOR, width=2.5), hovertemplate="%{x|%b %d, %Y}<br>Dem: %{y:.1f}%<extra></extra>"))
                    _rfig.add_trace(go.Scatter(x=_rh["as_of_date"], y=_rh["gop_pct_est"], mode="lines", name="GOP %", line=dict(color=GOP_COLOR, width=2.5), hovertemplate="%{x|%b %d, %Y}<br>GOP: %{y:.1f}%<extra></extra>"))
                    _rfig.add_hline(y=50, line_dash="dot", line_color="#333338", line_width=1)
                    _rfig.update_yaxes(ticksuffix="%")
                _rfig.update_layout(height=250, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=44, r=12, t=10, b=32), hovermode="x unified", font=dict(family="Space Grotesk, sans-serif", color=INK, size=12), legend=dict(orientation="h", y=1.1, x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=11, color=MUTED)), hoverlabel=dict(bgcolor=TOOLTIP_BG, bordercolor=TOOLTIP_BORDER, namelength=-1, font=dict(color=INK, family="Space Grotesk, sans-serif", size=13)))
                _rfig.update_xaxes(showgrid=False, linecolor=RULE, tickfont=dict(size=10, color=SUBTLE), tickformat="%b '%y")
                _rfig.update_yaxes(showgrid=True, gridcolor=CHART_GRID, zeroline=False, linecolor=RULE, tickfont=dict(size=10, color=SUBTLE))
                st.plotly_chart(_rfig, use_container_width=True, key="race_detail_chart", config={"displayModeBar": False})

    # Senate seat distribution
    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
    _sen_view = st.radio("Senate charts", ["Seat Distribution", "Control Odds", "Seat Aggregate"], horizontal=True, label_visibility="collapsed", key="sen_chart_toggle")
    _sen_hist_path = SENATE_ROOT / "data/history/forecast_history.csv"
    _sen_hist = pd.DataFrame()
    if _sen_hist_path.exists():
        _sen_hist = pd.read_csv(_sen_hist_path)
        _sen_hist["as_of_date"] = pd.to_datetime(_sen_hist["as_of_date"], errors="coerce")
        _sen_hist = _sen_hist.dropna(subset=["as_of_date"]).sort_values("as_of_date")

    if _sen_view == "Seat Distribution":
        st.markdown(f'<p class="sec-label">Senate Seat Distribution &mdash; {int(sen_summary.get("simulation_current", 120000)):,} simulations</p>', unsafe_allow_html=True)
        if not sen_seat_dist.empty:
            st.plotly_chart(_senate_seat_dist_chart(sen_seat_dist, sen_summary), use_container_width=True, key="sen_seat_dist_chart", config={"displayModeBar": False})
    elif _sen_view == "Control Odds" and not _sen_hist.empty:
        st.markdown('<p class="sec-label">Senate Control Probability Over Time</p>', unsafe_allow_html=True)
        _sfig = go.Figure()
        _sfig.add_trace(go.Scatter(x=_sen_hist["as_of_date"], y=_sen_hist["gop_control_prob"], mode="lines", name="Republicans", line=dict(color=GOP_COLOR, width=2.5), hovertemplate="%{x|%b %d, %Y}<br>GOP: %{y:.1%}<extra></extra>"))
        _sfig.add_trace(go.Scatter(x=_sen_hist["as_of_date"], y=_sen_hist["dem_control_prob"], mode="lines", name="Democrats", line=dict(color=DEM_COLOR, width=2.5), hovertemplate="%{x|%b %d, %Y}<br>Dem: %{y:.1%}<extra></extra>"))
        if "tied_chamber_prob" in _sen_hist.columns:
            _sfig.add_trace(go.Scatter(x=_sen_hist["as_of_date"], y=_sen_hist["tied_chamber_prob"], mode="lines", name="50-50 tie", line=dict(color=AMBER, width=1.5, dash="dot"), hovertemplate="%{x|%b %d, %Y}<br>Tie: %{y:.1%}<extra></extra>"))
        _sfig.add_hline(y=0.50, line_dash="dot", line_color="#333338", line_width=1)
        _sfig.update_layout(height=280, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=44, r=12, t=10, b=32), hovermode="x unified", font=dict(family="Space Grotesk, sans-serif", color=INK, size=12), legend=dict(orientation="h", y=1.1, x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=11, color=MUTED)), hoverlabel=dict(bgcolor=TOOLTIP_BG, bordercolor=TOOLTIP_BORDER, namelength=-1, font=dict(color=INK, family="Space Grotesk, sans-serif", size=13)))
        _sfig.update_xaxes(showgrid=False, linecolor=RULE, tickfont=dict(size=10, color=SUBTLE), tickformat="%b '%y")
        _sfig.update_yaxes(showgrid=True, gridcolor=CHART_GRID, zeroline=False, linecolor=RULE, tickformat=".0%", range=[0, 1], tickvals=[0, 0.25, 0.5, 0.75, 1.0], tickfont=dict(size=10, color=SUBTLE))
        st.plotly_chart(_sfig, use_container_width=True, key="sen_odds_chart", config={"displayModeBar": False})
    elif _sen_view == "Seat Aggregate" and not _sen_hist.empty:
        st.markdown('<p class="sec-label">Expected Senate Seats Over Time</p>', unsafe_allow_html=True)
        _sfig2 = go.Figure()
        _sen_hist["expected_dem_seats_h"] = 100 - _sen_hist["expected_gop_seats"]
        if "gop_seat_q05" in _sen_hist.columns:
            _sen_hist["dem_q95_h"] = 100 - _sen_hist["gop_seat_q05"]
            _sen_hist["dem_q05_h"] = 100 - _sen_hist["gop_seat_q95"]
            _sfig2.add_trace(go.Scatter(x=_sen_hist["as_of_date"], y=_sen_hist["gop_seat_q95"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
            _sfig2.add_trace(go.Scatter(x=_sen_hist["as_of_date"], y=_sen_hist["gop_seat_q05"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip", fill="tonexty", fillcolor="rgba(224,92,79,0.06)"))
            _sfig2.add_trace(go.Scatter(x=_sen_hist["as_of_date"], y=_sen_hist["dem_q95_h"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
            _sfig2.add_trace(go.Scatter(x=_sen_hist["as_of_date"], y=_sen_hist["dem_q05_h"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip", fill="tonexty", fillcolor="rgba(74,144,217,0.06)"))
        _sfig2.add_trace(go.Scatter(x=_sen_hist["as_of_date"], y=_sen_hist["expected_gop_seats"], mode="lines", name="GOP seats", line=dict(color=GOP_COLOR, width=2.5), hovertemplate="%{x|%b %d, %Y}<br>GOP: %{y:.1f}<extra></extra>"))
        _sfig2.add_trace(go.Scatter(x=_sen_hist["as_of_date"], y=_sen_hist["expected_dem_seats_h"], mode="lines", name="Dem seats", line=dict(color=DEM_COLOR, width=2.5), hovertemplate="%{x|%b %d, %Y}<br>Dem: %{y:.1f}<extra></extra>"))
        _sfig2.add_hline(y=50, line_width=1, line_dash="dot", line_color=AMBER, annotation_text="50 (VP tiebreak)", annotation_position="right", annotation_font=dict(size=10, color=AMBER, family="JetBrains Mono, monospace"))
        _sfig2.update_layout(height=280, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=44, r=12, t=10, b=32), hovermode="x unified", font=dict(family="Space Grotesk, sans-serif", color=INK, size=12), legend=dict(orientation="h", y=1.1, x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=11, color=MUTED)), hoverlabel=dict(bgcolor=TOOLTIP_BG, bordercolor=TOOLTIP_BORDER, namelength=-1, font=dict(color=INK, family="Space Grotesk, sans-serif", size=13)))
        _sfig2.update_xaxes(showgrid=False, linecolor=RULE, tickfont=dict(size=10, color=SUBTLE), tickformat="%b '%y")
        _sfig2.update_yaxes(showgrid=True, gridcolor=CHART_GRID, zeroline=False, linecolor=RULE, title=dict(text="Seats", font=dict(size=11, color=SUBTLE)), tickfont=dict(size=10, color=SUBTLE))
        st.plotly_chart(_sfig2, use_container_width=True, key="sen_seats_chart", config={"displayModeBar": False})

    st.markdown(
        f'<p class="foot">{sen_summary.get("tie_rule", "")}</p>',
        unsafe_allow_html=True,
    )
