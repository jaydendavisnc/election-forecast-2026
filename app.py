from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import DEFAULT_CONFIG, ProjectPaths
from src.pipeline import ForecastPipeline

ROOT = Path(__file__).resolve().parent
PATHS = ProjectPaths(ROOT)

st.set_page_config(page_title="2026 House forecast", page_icon="📊", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.1rem; padding-bottom: 2rem;}
    .summary-band {
        padding: 1rem 1.1rem;
        border: 1px solid rgba(250,250,250,0.10);
        border-radius: 14px;
        margin-bottom: 1rem;
        background: linear-gradient(180deg, rgba(250,250,250,0.03), rgba(250,250,250,0.01));
    }
    .small-note {color: rgba(250,250,250,0.74); font-size: 0.95rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def _prob_text(x: float) -> str:
    return f"{100.0 * float(x):.1f}%"


def _pct_text(x: Optional[float]) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{float(x):.1f}%"


def _margin_text(x: Optional[float]) -> str:
    if x is None or pd.isna(x):
        return "—"
    party = "D" if float(x) >= 0 else "R"
    return f"{party}+{abs(float(x)):.1f}"


def _signed_text(x: Optional[float], suffix: str = "") -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{float(x):+.1f}{suffix}"


def _delta_from_history(history: pd.DataFrame, column: str, days_back: int = 7) -> Optional[float]:
    if history.empty or column not in history.columns or "as_of_date" not in history.columns:
        return None
    hist = history.sort_values("as_of_date").copy()
    target = hist["as_of_date"].max() - pd.Timedelta(days=days_back)
    earlier = hist.loc[hist["as_of_date"] <= target]
    if earlier.empty:
        return None
    return float(hist.iloc[-1][column] - earlier.iloc[-1][column])


def _delta_label(value: Optional[float], kind: str = "number") -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    if kind == "prob":
        return f"{100.0 * value:+.1f} pts vs 7d"
    if kind == "margin":
        return f"{value:+.1f} pts vs 7d"
    if kind == "pct":
        return f"{value:+.1f} pts vs 7d"
    return f"{value:+.1f} vs 7d"


def _prep_history(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["as_of_date"] = pd.to_datetime(out["as_of_date"], errors="coerce")
    out = out.dropna(subset=["as_of_date"]).sort_values("as_of_date").reset_index(drop=True)
    return out


def _prep_poll_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for col in ["start_date", "end_date", "published_date", "obs_date"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    for col in ["sample_size", "dem_pct", "rep_pct", "pct_a", "pct_b", "margin_a"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in ["date_exact", "sample_exact", "population_exact", "partisan_flag", "use_official_override"]:
        if col in out.columns:
            lowered = out[col].astype(str).str.lower()
            mapped = lowered.map({"true": True, "false": False, "1": True, "0": False}).fillna(False)
            out[col] = mapped.astype(bool)
    sort_cols = [c for c in ["end_date", "published_date", "pollster"] if c in out.columns]
    return out.sort_values(sort_cols).reset_index(drop=True) if sort_cols else out


@st.cache_data(show_spinner=False)
def load_bundle(root: str) -> dict[str, pd.DataFrame | dict]:
    paths = ProjectPaths(Path(root))
    latest = paths.latest_dir
    bundle: dict[str, pd.DataFrame | dict] = {}

    file_map = {
        "district_forecast": latest / "district_forecast.csv",
        "district_priors": latest / "district_priors.csv",
        "district_master": latest / "district_master.csv",
        "seat_distribution": latest / "seat_distribution.csv",
        "history": latest / "forecast_curve.csv" if (latest / "forecast_curve.csv").exists() else paths.forecast_history_csv,
        "run_history": paths.run_history_csv,
        "district_polls": latest / "district_polls.csv",
        "generic_polls": latest / "generic_ballot_polls.csv" if (latest / "generic_ballot_polls.csv").exists() else (paths.seed_dir / "generic_ballot_polls_master.csv"),
        "approval_polls": latest / "trump_approval_polls.csv" if (latest / "trump_approval_polls.csv").exists() else (paths.seed_dir / "trump_approval_recent_polls.csv"),
        "approval_curve": latest / "trump_approval_curve.csv",
    }
    for name, file_path in file_map.items():
        bundle[name] = pd.read_csv(file_path) if file_path.exists() else pd.DataFrame()

    summary_path = latest / "summary.json"
    audit_path = paths.run_audit_json
    bundle["summary"] = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    bundle["audit"] = json.loads(audit_path.read_text()) if audit_path.exists() else {}
    return bundle


def run_pipeline_interactive(use_live: bool, include_fec: bool, include_district_polls: bool, fec_api_key: str | None) -> None:
    pipeline = ForecastPipeline(ROOT)
    with st.spinner("Refreshing the forecast, rebuilding the daily curve, and updating the national polling views..."):
        pipeline.run(
            use_live=use_live,
            include_fec=include_fec,
            include_district_polls=include_district_polls,
            fec_api_key=fec_api_key or None,
            as_of_date=date.today(),
        )
    st.cache_data.clear()


def _trend_range_selector() -> dict:
    return {
        "buttons": [
            {"count": 3, "label": "3m", "step": "month", "stepmode": "backward"},
            {"count": 6, "label": "6m", "step": "month", "stepmode": "backward"},
            {"count": 1, "label": "1y", "step": "year", "stepmode": "backward"},
            {"step": "all", "label": "All"},
        ]
    }


def _filter_window(history: pd.DataFrame, choice: str) -> pd.DataFrame:
    if history.empty or choice == "All":
        return history
    days = int(choice)
    cutoff = history["as_of_date"].max() - pd.Timedelta(days=days - 1)
    return history.loc[history["as_of_date"] >= cutoff].copy()


def _probability_chart(history: pd.DataFrame, run_history: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history["as_of_date"],
            y=history["gop_control_prob"],
            mode="lines",
            name="GOP hold probability",
            hovertemplate="%{x|%b %d, %Y}<br>GOP hold: %{y:.1%}<extra></extra>",
        )
    )
    if not run_history.empty and len(run_history) > 1:
        run_history = run_history.copy()
        run_history["as_of_date"] = pd.to_datetime(run_history["as_of_date"], errors="coerce")
        fig.add_trace(
            go.Scatter(
                x=run_history["as_of_date"],
                y=run_history["gop_control_prob"],
                mode="markers",
                name="Saved run snapshot",
                hovertemplate="Saved run<br>%{x|%b %d, %Y}<br>GOP hold: %{y:.1%}<extra></extra>",
            )
        )
    fig.add_hline(y=0.50, line_dash="dash", annotation_text="50%", annotation_position="top left")
    fig.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=24, b=20),
        yaxis=dict(title="Probability", tickformat=".0%", range=[0, 1]),
        xaxis=dict(title="Date", rangeselector=_trend_range_selector()),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.08),
    )
    return fig


def _seat_trend_chart(history: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if {"gop_seat_q05", "gop_seat_q95"}.issubset(history.columns):
        fig.add_trace(go.Scatter(x=history["as_of_date"], y=history["gop_seat_q95"], mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(
            go.Scatter(
                x=history["as_of_date"],
                y=history["gop_seat_q05"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                name="90% interval",
                hovertemplate="%{x|%b %d, %Y}<br>90%% interval floor: %{y:.0f}<extra></extra>",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=history["as_of_date"],
            y=history["expected_gop_seats"],
            mode="lines",
            name="Expected GOP seats",
            hovertemplate="%{x|%b %d, %Y}<br>Expected GOP seats: %{y:.1f}<extra></extra>",
        )
    )
    fig.add_hline(y=218, line_dash="dash", annotation_text="218 seats", annotation_position="top left")
    fig.update_layout(
        height=340,
        margin=dict(l=20, r=20, t=24, b=20),
        yaxis=dict(title="GOP seats"),
        xaxis=dict(title="Date", rangeselector=_trend_range_selector()),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.08),
    )
    return fig


def _seat_distribution_chart(seat_distribution: pd.DataFrame, expected_seats: float) -> go.Figure:
    fig = px.bar(
        seat_distribution,
        x="gop_seats",
        y="probability",
        labels={"gop_seats": "GOP seats", "probability": "Probability"},
    )
    fig.add_vline(x=218, line_dash="dash")
    fig.add_vline(x=expected_seats, line_dash="dot")
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=24, b=20))
    return fig


def _generic_chart(history: pd.DataFrame, generic_polls: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if {"filtered_generic_sd", "generic_ballot_margin_dem"}.issubset(history.columns):
        z90 = 1.645
        upper = history["generic_ballot_margin_dem"] + z90 * history["filtered_generic_sd"]
        lower = history["generic_ballot_margin_dem"] - z90 * history["filtered_generic_sd"]
        fig.add_trace(go.Scatter(x=history["as_of_date"], y=upper, mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=history["as_of_date"], y=lower, mode="lines", line=dict(width=0), fill="tonexty", name="90% band", hoverinfo="skip"))
    fig.add_trace(
        go.Scatter(
            x=history["as_of_date"],
            y=history["generic_ballot_margin_dem"],
            mode="lines",
            name="Filtered generic ballot",
            hovertemplate="%{x|%b %d, %Y}<br>Dem margin: %{y:+.1f}<extra></extra>",
        )
    )
    if "national_mean_margin_dem" in history.columns:
        fig.add_trace(
            go.Scatter(
                x=history["as_of_date"],
                y=history["national_mean_margin_dem"],
                mode="lines",
                name="Model national environment",
                hovertemplate="%{x|%b %d, %Y}<br>Model env: %{y:+.1f}<extra></extra>",
            )
        )
    if not generic_polls.empty:
        exact = generic_polls.loc[generic_polls.get("date_exact", False) == True].copy() if "date_exact" in generic_polls.columns else pd.DataFrame()
        inferred = generic_polls.loc[generic_polls.get("date_exact", False) != True].copy() if "date_exact" in generic_polls.columns else generic_polls.copy()

        def add_points(df: pd.DataFrame, name: str) -> None:
            if df.empty:
                return
            custom_cols = []
            for col in ["pollster", "population", "sample_size", "published_date", "metadata_source"]:
                custom_cols.append(df[col].astype(str) if col in df.columns else pd.Series(["—"] * len(df)))
            customdata = np.column_stack(custom_cols)
            fig.add_trace(
                go.Scatter(
                    x=df["end_date"].fillna(df.get("published_date")),
                    y=df["margin_a"],
                    mode="markers",
                    name=name,
                    customdata=customdata,
                    hovertemplate=(
                        "%{x|%b %d, %Y}<br>%{customdata[0]}<br>%{customdata[1]} · n=%{customdata[2]}"
                        "<br>published %{customdata[3]}<br>%{customdata[4]}"
                        "<br>Dem margin: %{y:+.1f}<extra></extra>"
                    ),
                )
            )
        add_points(exact, "Exact field-date rows")
        add_points(inferred, "Published-date fallback rows")
    fig.add_hline(y=0, line_dash="dash")
    fig.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=24, b=20),
        yaxis=dict(title="Dem generic-ballot margin"),
        xaxis=dict(title="Date", rangeselector=_trend_range_selector()),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.10),
    )
    return fig


def _approval_chart(approval_curve: pd.DataFrame, approval_polls: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if approval_curve.empty:
        fig.update_layout(height=360, margin=dict(l=20, r=20, t=24, b=20), xaxis=dict(title="Date"), yaxis=dict(title="Percent"))
        return fig

    if {"approve_low_90", "approve_high_90"}.issubset(approval_curve.columns):
        fig.add_trace(go.Scatter(x=approval_curve["as_of_date"], y=approval_curve["approve_high_90"], mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=approval_curve["as_of_date"], y=approval_curve["approve_low_90"], mode="lines", line=dict(width=0), fill="tonexty", name="Approve 90% band", hoverinfo="skip"))
    if {"disapprove_low_90", "disapprove_high_90"}.issubset(approval_curve.columns):
        fig.add_trace(go.Scatter(x=approval_curve["as_of_date"], y=approval_curve["disapprove_high_90"], mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=approval_curve["as_of_date"], y=approval_curve["disapprove_low_90"], mode="lines", line=dict(width=0), fill="tonexty", name="Disapprove 90% band", hoverinfo="skip"))

    fig.add_trace(
        go.Scatter(
            x=approval_curve["as_of_date"],
            y=approval_curve["approve_avg"],
            mode="lines",
            name="Approve",
            hovertemplate="%{x|%b %d, %Y}<br>Approve: %{y:.1f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=approval_curve["as_of_date"],
            y=approval_curve["disapprove_avg"],
            mode="lines",
            name="Disapprove",
            hovertemplate="%{x|%b %d, %Y}<br>Disapprove: %{y:.1f}%<extra></extra>",
        )
    )

    if not approval_polls.empty:
        fig.add_trace(
            go.Scatter(
                x=approval_polls["end_date"].fillna(approval_polls.get("published_date")),
                y=approval_polls["pct_a"],
                mode="markers",
                name="Approve polls",
                customdata=np.column_stack([
                    approval_polls.get("pollster", pd.Series(["—"] * len(approval_polls))).astype(str),
                    approval_polls.get("population", pd.Series(["—"] * len(approval_polls))).astype(str),
                    approval_polls.get("sample_size", pd.Series(["—"] * len(approval_polls))).astype(str),
                ]),
                hovertemplate="%{x|%b %d, %Y}<br>%{customdata[0]}<br>%{customdata[1]} · n=%{customdata[2]}<br>Approve: %{y:.1f}%<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=approval_polls["end_date"].fillna(approval_polls.get("published_date")),
                y=approval_polls["pct_b"],
                mode="markers",
                name="Disapprove polls",
                customdata=np.column_stack([
                    approval_polls.get("pollster", pd.Series(["—"] * len(approval_polls))).astype(str),
                    approval_polls.get("population", pd.Series(["—"] * len(approval_polls))).astype(str),
                    approval_polls.get("sample_size", pd.Series(["—"] * len(approval_polls))).astype(str),
                ]),
                hovertemplate="%{x|%b %d, %Y}<br>%{customdata[0]}<br>%{customdata[1]} · n=%{customdata[2]}<br>Disapprove: %{y:.1f}%<extra></extra>",
            )
        )

    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=24, b=20),
        yaxis=dict(title="Percent"),
        xaxis=dict(title="Date", rangeselector=_trend_range_selector()),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.10),
    )
    return fig


def _render_source_status(source_status: dict) -> None:
    if not source_status:
        return
    cols = st.columns(6)
    items = [
        ("National archive", source_status.get("national_context_live", False)),
        ("Pres by district", source_status.get("presidential_by_district_live", False)),
        ("Open seats", source_status.get("open_seats_live", False)),
        ("Consensus ratings", source_status.get("consensus_ratings_live", False)),
        ("District polls", source_status.get("district_polls_live", False)),
        ("FEC", source_status.get("fec_live", False)),
    ]
    for col, (label, ok) in zip(cols, items):
        col.metric(label, "live" if ok else "seed / fallback")


st.title("2026 House midterm forecast")
st.caption("District-level House model with an archived generic-ballot filter, a weak Trump-approval cross-check, structural district priors, and Monte Carlo seat simulations.")

with st.sidebar:
    st.header("Controls")
    mode = st.radio("Data mode", ["Live", "Seed"], index=0)
    use_live = mode == "Live"
    include_fec = st.toggle("Use OpenFEC finance data", value=True)
    include_district_polls = st.toggle("Scan district poll pages", value=True)
    fec_api_key = st.text_input("FEC API key", value=os.getenv("FEC_API_KEY", ""), type="password")
    trend_window = st.selectbox("Trend window", ["90", "180", "365", "All"], index=2)
    if st.button("Run / refresh forecast", type="primary", use_container_width=True):
        run_pipeline_interactive(use_live, include_fec, include_district_polls, fec_api_key)
    st.divider()
    st.markdown(
        "The odds line now keeps today’s district and campaign inputs fixed through the reconstructed daily path, so the endpoint is comparable instead of jumping because the model definition changed."
    )

if not (PATHS.latest_dir / "summary.json").exists():
    pipeline = ForecastPipeline(ROOT)
    pipeline.run(use_live=False, include_fec=False, include_district_polls=False, as_of_date=date(2026, 4, 1))

bundle = load_bundle(str(ROOT))
summary = bundle["summary"] or {}
audit = bundle["audit"] or {}
districts = bundle["district_forecast"] if isinstance(bundle["district_forecast"], pd.DataFrame) else pd.DataFrame()
seat_distribution = bundle["seat_distribution"] if isinstance(bundle["seat_distribution"], pd.DataFrame) else pd.DataFrame()
history = _prep_history(bundle["history"] if isinstance(bundle["history"], pd.DataFrame) else pd.DataFrame())
run_history = _prep_history(bundle["run_history"] if isinstance(bundle["run_history"], pd.DataFrame) else pd.DataFrame())
generic_polls = _prep_poll_frame(bundle["generic_polls"] if isinstance(bundle["generic_polls"], pd.DataFrame) else pd.DataFrame())
approval_polls = _prep_poll_frame(bundle["approval_polls"] if isinstance(bundle["approval_polls"], pd.DataFrame) else pd.DataFrame())
approval_curve = _prep_history(bundle["approval_curve"] if isinstance(bundle["approval_curve"], pd.DataFrame) else pd.DataFrame())
district_polls = bundle["district_polls"] if isinstance(bundle["district_polls"], pd.DataFrame) else pd.DataFrame()

if not summary:
    st.error("No forecast outputs found.")
    st.stop()

hist_view = _filter_window(history, trend_window)
approval_view = _filter_window(approval_curve, trend_window) if not approval_curve.empty else approval_curve
prob_delta = _delta_from_history(history, "gop_control_prob")
seat_delta = _delta_from_history(history, "expected_gop_seats")
generic_delta = _delta_from_history(history, "generic_ballot_margin_dem")
model_env_delta = _delta_from_history(history, "national_mean_margin_dem")
approve_delta = _delta_from_history(approval_curve, "approve_avg")
disapprove_delta = _delta_from_history(approval_curve, "disapprove_avg")
archive_rows = int(summary.get("generic_poll_archive_rows", summary.get("visible_national_poll_rows", 0)))
exact_date_rows = int(summary.get("generic_poll_exact_date_rows", 0))
inferred_date_rows = int(summary.get("generic_poll_inferred_date_rows", 0))
approval_rows = int(summary.get("approval_recent_poll_rows", 0))

_render_source_status(summary.get("source_status", {}))

st.markdown(
    f"""
    <div class="summary-band">
      <div><strong>Current read:</strong> Republicans are at <strong>{_prob_text(summary['gop_control_prob'])}</strong> to hold the House,
      with <strong>{summary['expected_gop_seats']:.1f}</strong> expected seats. The raw filtered generic ballot is
      <strong>{_margin_text(summary['generic_ballot_margin_dem'])}</strong>; after the model’s weak Trump-approval cross-check,
      the national House environment used in simulation is <strong>{_margin_text(summary.get('national_mean_margin_dem'))}</strong>.</div>
      <div class="small-note">Trend line methodology: today’s district and campaign inputs are frozen through the daily reconstruction for comparability, while archived saved runs remain separate for audit. Trump approval view: { _pct_text(summary.get('trump_approve_pct')) } approve / { _pct_text(summary.get('trump_disapprove_pct')) } disapprove, from {approval_rows} recent parsed poll rows plus the current DDHQ aggregate endpoint.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

top_metrics = st.columns(6)
top_metrics[0].metric("GOP hold odds", _prob_text(summary["gop_control_prob"]), _delta_label(prob_delta, "prob"))
top_metrics[1].metric("Expected GOP seats", f"{summary['expected_gop_seats']:.1f}", _delta_label(seat_delta))
top_metrics[2].metric("Model national env", _margin_text(summary.get("national_mean_margin_dem")), _delta_label(model_env_delta, "margin"))
top_metrics[3].metric("Filtered generic ballot", _margin_text(summary["generic_ballot_margin_dem"]), _delta_label(generic_delta, "margin"))
top_metrics[4].metric("Trump approve", _pct_text(summary.get("trump_approve_pct")), _delta_label(approve_delta, "pct"))
top_metrics[5].metric("Trump disapprove", _pct_text(summary.get("trump_disapprove_pct")), _delta_label(disapprove_delta, "pct"))

sub_metrics = st.columns(6)
sub_metrics[0].metric("Median GOP seats", f"{summary['median_gop_seats']:.0f}")
sub_metrics[1].metric("90% seat range", f"{summary['gop_seat_q05']:.0f}–{summary['gop_seat_q95']:.0f}")
sub_metrics[2].metric("Trump net approval", _signed_text(summary.get("trump_net_approval"), " pts"))
sub_metrics[3].metric("Approval adjustment", _signed_text(summary.get("approval_adjustment_to_national_margin"), " pts"))
sub_metrics[4].metric("Archive rows", f"{archive_rows}")
sub_metrics[5].metric("As of", summary["as_of_date"])

tab_overview, tab_trend, tab_polls, tab_districts, tab_audit, tab_method = st.tabs(
    ["Overview", "Trend", "National polling", "Districts", "Audit", "Methodology"]
)

with tab_overview:
    left, right = st.columns([1.12, 0.88])
    with left:
        st.subheader("Odds over time")
        st.plotly_chart(_probability_chart(hist_view, run_history), use_container_width=True)
        st.subheader("National generic-ballot aggregate")
        st.plotly_chart(_generic_chart(hist_view, generic_polls), use_container_width=True)
    with right:
        st.subheader("Expected GOP seats")
        st.plotly_chart(_seat_trend_chart(hist_view), use_container_width=True)
        if not seat_distribution.empty:
            st.subheader("Seat distribution")
            st.plotly_chart(_seat_distribution_chart(seat_distribution, summary["expected_gop_seats"]), use_container_width=True)
        if not districts.empty:
            st.subheader("Closest seats")
            close = districts.copy()
            close["abs_margin"] = pd.to_numeric(close["mean_margin_sim"], errors="coerce").abs()
            table = close.sort_values("abs_margin").head(16)[["district_code", "mean_margin_sim", "dem_win_prob", "gop_win_prob", "open_seat", "rating", "poll_count"]].copy()
            table["mean_margin_sim"] = table["mean_margin_sim"].map(_margin_text)
            table["dem_win_prob"] = table["dem_win_prob"].map(_prob_text)
            table["gop_win_prob"] = table["gop_win_prob"].map(_prob_text)
            st.dataframe(table, use_container_width=True, hide_index=True)
    st.subheader("Trump approval")
    st.plotly_chart(_approval_chart(approval_view, approval_polls), use_container_width=True)

with tab_trend:
    st.subheader("Daily control and seat path")
    trend_caption = "The line is a comparable daily reconstruction: current campaign structure is frozen through time, and the raw national generic-ballot state moves day by day."
    endpoint_gap_prob = summary.get("history_current_endpoint_gap_prob")
    endpoint_gap_seats = summary.get("history_current_endpoint_gap_seats")
    if endpoint_gap_prob is not None and not pd.isna(endpoint_gap_prob):
        trend_caption += f" Current official snapshot minus comparable endpoint: {_signed_text(100.0 * float(endpoint_gap_prob), ' pts')} in hold odds and {_signed_text(endpoint_gap_seats)} seats."
    st.caption(trend_caption)

    trend_cols = st.columns(4)
    if not hist_view.empty:
        start_row = hist_view.iloc[0]
        end_row = hist_view.iloc[-1]
        trend_cols[0].metric("Start GOP hold odds", _prob_text(float(start_row["gop_control_prob"])))
        trend_cols[1].metric("Now GOP hold odds", _prob_text(float(end_row["gop_control_prob"])))
        trend_cols[2].metric("Start expected seats", f"{float(start_row['expected_gop_seats']):.1f}")
        trend_cols[3].metric("Now expected seats", f"{float(end_row['expected_gop_seats']):.1f}")

        st.plotly_chart(_probability_chart(hist_view, run_history), use_container_width=True)
        st.plotly_chart(_seat_trend_chart(hist_view), use_container_width=True)

        with st.expander("Daily history table", expanded=False):
            show = hist_view.copy()
            for col in ["gop_control_prob", "dem_control_prob"]:
                if col in show.columns:
                    show[col] = show[col].map(_prob_text)
            for col in ["generic_ballot_margin_dem", "national_mean_margin_dem", "approval_implied_generic_margin_dem"]:
                if col in show.columns:
                    show[col] = show[col].map(_margin_text)
            for col in ["trump_approve_pct", "trump_disapprove_pct"]:
                if col in show.columns:
                    show[col] = show[col].map(_pct_text)
            st.dataframe(show.sort_values("as_of_date", ascending=False), use_container_width=True, hide_index=True)

with tab_polls:
    st.subheader("National polling inputs")
    poll_metrics = st.columns(6)
    poll_metrics[0].metric("Generic archive rows", f"{archive_rows}")
    poll_metrics[1].metric("Exact generic dates", f"{exact_date_rows}")
    poll_metrics[2].metric("Fallback generic dates", f"{inferred_date_rows}")
    poll_metrics[3].metric("RCP current average", _margin_text(summary.get("rcp_current_average_margin_dem", np.nan)))
    poll_metrics[4].metric("DDHQ approve", _pct_text(summary.get("trump_approve_pct")))
    poll_metrics[5].metric("DDHQ disapprove", _pct_text(summary.get("trump_disapprove_pct")))

    st.subheader("Generic congressional vote")
    st.plotly_chart(_generic_chart(hist_view, generic_polls), use_container_width=True)

    st.subheader("Trump approval")
    st.plotly_chart(_approval_chart(approval_view, approval_polls), use_container_width=True)

    if not generic_polls.empty:
        with st.expander("Generic-ballot poll table", expanded=False):
            show_cols = [
                "published_date", "end_date", "pollster", "dem_pct", "rep_pct", "margin_a", "sample_size", "population",
                "date_exact", "sample_exact", "partisan_flag", "topline_source", "metadata_source", "notes"
            ]
            show_cols = [c for c in show_cols if c in generic_polls.columns]
            show = generic_polls.copy()
            if "published_date" in show.columns:
                show = show.sort_values(["published_date", "end_date"], ascending=False)
            st.dataframe(show[show_cols], use_container_width=True, hide_index=True)

    if not approval_polls.empty:
        with st.expander("Trump-approval poll table", expanded=False):
            show_cols = [
                "end_date", "pollster", "pct_a", "pct_b", "margin_a", "sample_size", "population",
                "date_exact", "sample_exact", "partisan_flag", "topline_source", "metadata_source", "notes"
            ]
            show_cols = [c for c in show_cols if c in approval_polls.columns]
            show = approval_polls.copy().sort_values("end_date", ascending=False)
            st.dataframe(show[show_cols], use_container_width=True, hide_index=True)

with tab_districts:
    if districts.empty:
        st.warning("No district outputs found.")
    else:
        district_codes = districts["district_code"].tolist()
        selected = st.selectbox("District", options=district_codes, index=0)
        row = districts.loc[districts["district_code"] == selected].iloc[0]

        cols = st.columns(5)
        cols[0].metric("Projected margin", _margin_text(row["mean_margin_sim"]))
        cols[1].metric("GOP win probability", _prob_text(row["gop_win_prob"]))
        cols[2].metric("2024 House margin", _margin_text(row["house_margin_2024"]))
        cols[3].metric("Rating", row["rating"] if pd.notna(row["rating"]) else "—")
        cols[4].metric("Open seat", "Yes" if bool(row["open_seat"]) else "No")

        expl = pd.DataFrame(
            {
                "Field": [
                    "Posterior intercept mean",
                    "Posterior intercept SD",
                    "District polls parsed",
                    "Finance effect",
                    "Dem candidate",
                    "Rep candidate",
                    "Presidential margin 2024",
                    "Presidential margin 2020",
                ],
                "Value": [
                    f"{float(row['intercept_mean']):+.2f}",
                    f"{float(row['intercept_sd']):.2f}",
                    int(row["poll_count"]),
                    f"{float(row['finance_effect']):+.2f}",
                    row.get("dem_candidate") or "—",
                    row.get("rep_candidate") or "—",
                    _margin_text(row.get("pres24_dem_margin")),
                    _margin_text(row.get("pres20_dem_margin")),
                ],
            }
        )
        st.dataframe(expl, use_container_width=True, hide_index=True)

        close = districts.copy()
        close["abs_margin"] = pd.to_numeric(close["mean_margin_sim"], errors="coerce").abs()
        close = close.sort_values("abs_margin")
        show = close[["district_code", "mean_margin_sim", "dem_win_prob", "gop_win_prob", "house_margin_2024", "rating", "open_seat", "poll_count"]].copy()
        show["mean_margin_sim"] = show["mean_margin_sim"].map(_margin_text)
        show["dem_win_prob"] = show["dem_win_prob"].map(_prob_text)
        show["gop_win_prob"] = show["gop_win_prob"].map(_prob_text)
        show["house_margin_2024"] = show["house_margin_2024"].map(_margin_text)
        st.markdown("**All district forecasts**")
        st.dataframe(show, use_container_width=True, hide_index=True)

        if not district_polls.empty and "district_code" in district_polls.columns:
            district_view = district_polls.loc[district_polls["district_code"] == selected].copy()
            if not district_view.empty:
                st.markdown("**Parsed district polls for this seat**")
                st.dataframe(district_view, use_container_width=True, hide_index=True)

with tab_audit:
    st.subheader("Run audit")
    audit_cols = st.columns(6)
    audit_cols[0].metric("Main simulations", f"{int(summary.get('simulations', DEFAULT_CONFIG.simulations)):,}")
    audit_cols[1].metric("History simulations", f"{int(summary.get('history_simulations', DEFAULT_CONFIG.history_simulations)):,}")
    audit_cols[2].metric("Approval adjustment", _signed_text(summary.get("approval_adjustment_to_national_margin"), " pts"))
    audit_cols[3].metric("Approval-implied gap", _signed_text(summary.get("approval_generic_gap_dem"), " pts"))
    audit_cols[4].metric("Endpoint odds gap", _signed_text(None if summary.get("history_current_endpoint_gap_prob") is None else 100.0 * float(summary.get("history_current_endpoint_gap_prob")), " pts"))
    audit_cols[5].metric("Endpoint seat gap", _signed_text(summary.get("history_current_endpoint_gap_seats")))

    left, right = st.columns(2)
    with left:
        st.markdown("**Source status**")
        source_rows = [{"Source field": k, "Status": v} for k, v in summary.get("source_status", {}).items()]
        st.dataframe(pd.DataFrame(source_rows), use_container_width=True, hide_index=True)
        st.markdown("**Audit manifest**")
        st.json(audit)
    with right:
        params = pd.DataFrame(
            {
                "Parameter": [
                    "Filtered generic ballot",
                    "Model national environment",
                    "Trump approve",
                    "Trump disapprove",
                    "Trump net approval",
                    "Approval-implied generic margin",
                    "Approval prior SD",
                    "Approval-to-generic slope",
                    "History mode",
                    "History campaign inputs mode",
                    "History approval mode",
                ],
                "Value": [
                    _margin_text(summary.get("generic_ballot_margin_dem", np.nan)),
                    _margin_text(summary.get("national_mean_margin_dem", np.nan)),
                    _pct_text(summary.get("trump_approve_pct", np.nan)),
                    _pct_text(summary.get("trump_disapprove_pct", np.nan)),
                    _signed_text(summary.get("trump_net_approval", np.nan), " pts"),
                    _margin_text(summary.get("approval_implied_generic_margin_dem", np.nan)),
                    f"{summary.get('approval_prior_sd', np.nan):.2f}",
                    f"{summary.get('approval_to_generic_slope', np.nan):.2f}",
                    summary.get("history_mode", "—"),
                    summary.get("history_campaign_inputs_mode", "—"),
                    summary.get("history_approval_mode", "—"),
                ],
            }
        )
        st.dataframe(params, use_container_width=True, hide_index=True)

with tab_method:
    st.subheader("Method")
    st.markdown(
        """
The core national signal is still the archived generic-ballot poll table. Each generic-ballot row is treated as a noisy observation of a latent national House environment, with larger observation variance when metadata are thin, dates are inferred, sample sizes are missing, or the poll is partisan/internal.

Trump approval is now explicitly in the model, but only as a weak cross-check. The model converts Trump net approval into an approval-implied House environment using a shallow slope and then combines that with the filtered generic ballot as a weak prior. That design matters: approval is informative about the broader political climate, but it is highly collinear with the generic ballot, so letting it dominate would double-count the same national mood.

The displayed generic-ballot chart now separates the raw filtered generic-ballot line from the actual national environment used in the seat simulation. Their difference is the approval adjustment, which is deliberately small unless generic polling is sparse or unusually noisy.

The old endpoint spike came from a model-definition break: the historical line used stripped-down history inputs while the current point used the full district model. The trend line now holds today’s district and campaign inputs fixed all the way through the reconstructed history so the odds path is internally comparable. Saved run snapshots are still preserved separately for audit.

Trump approval is also broken out visually as two lines — approve and disapprove — with 90% filter bands built from the recent parsed approval rows. That chart is a polling readout and audit surface, not a second forecasting engine. The seat forecast is still district-first and Monte Carlo based.

For each district, the forecast starts from 2024 House results, presidential vote by district when available, open-seat status, consensus ratings, finance, and any parsed district polls. The chamber forecast then runs Monte Carlo draws with a national House environment, correlated state error, and district-specific uncertainty. Control odds are the share of simulations where Republicans win at least 218 seats.
        """
    )
