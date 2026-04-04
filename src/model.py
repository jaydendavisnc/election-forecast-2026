from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from .config import (
    ELECTION_DATE,
    NATIONAL_HOUSE_MARGIN_2024,
    NATIONAL_PRES_MARGIN_2020,
    NATIONAL_PRES_MARGIN_2024,
    ForecastConfig,
    ProjectPaths,
)
from .data_sources import DDHQMetric
from .utils import (
    clip,
    combine_normal_estimates,
    district_sort_key,
    ensure_json_serializable,
    parse_month_day_range,
    sample_to_margin_sd,
    weighted_mean,
    weighted_std,
)


@dataclass(slots=True)
class NationalEnvironment:
    mean_margin_dem: float
    current_sd: float
    election_day_sd: float
    generic_ballot_margin_dem: float
    trump_net_approval: float
    right_track_net: float
    polls_included: Optional[int]
    days_to_election: int


@dataclass(slots=True)
class SimulationSummary:
    gop_control_prob: float
    expected_gop_seats: float
    median_gop_seats: float
    gop_seat_q05: float
    gop_seat_q25: float
    gop_seat_q75: float
    gop_seat_q95: float
    dem_control_prob: float


@dataclass(slots=True)
class SimulationOutputs:
    district_results: pd.DataFrame
    seat_distribution: pd.DataFrame
    summary: SimulationSummary
    seat_draws: np.ndarray


@dataclass(slots=True)
class SummaryRandomDraws:
    simulations: int
    z_national: np.ndarray
    z_state: dict[str, np.ndarray]
    z_district: np.ndarray


# ---------------------------------------------------------------------------
# National environment helpers
# ---------------------------------------------------------------------------


def _prepare_recent_polls_frame(recent_polls: pd.DataFrame, default_year: int = 2026) -> pd.DataFrame:
    if recent_polls is None:
        return pd.DataFrame()
    df = recent_polls.copy()
    if df.empty:
        return df
    if "sample_size" in df.columns:
        df["sample_size"] = pd.to_numeric(df["sample_size"], errors="coerce")
    if "margin_a" in df.columns:
        df["margin_a"] = pd.to_numeric(df["margin_a"], errors="coerce")
    if "end_date" not in df.columns and "field_dates" in df.columns:
        df["end_date"] = df["field_dates"].map(lambda x: parse_month_day_range(str(x), default_year))
    df["end_date"] = pd.to_datetime(df.get("end_date"), errors="coerce")
    return df


def _compute_poll_weights(df: pd.DataFrame, config: ForecastConfig, as_of_date: date) -> pd.DataFrame:
    out = df.copy()
    out["age_days"] = (
        pd.Timestamp(as_of_date) - out["end_date"].fillna(pd.Timestamp(as_of_date))
    ).dt.days.clip(lower=0)
    out["population_weight"] = out["population"].map(
        lambda x: config.national_population_weights.get(str(x).upper(), 0.86)
    )
    out["recency_weight"] = 0.5 ** (
        out["age_days"] / max(config.district_poll_recency_half_life_days, 1)
    )
    out["sample_weight"] = np.sqrt(out["sample_size"].fillna(800).clip(lower=100))
    out["weight"] = out["population_weight"] * out["recency_weight"] * out["sample_weight"]
    return out


def metric_snapshot_from_recent_polls(
    metric: DDHQMetric,
    as_of_date: date,
    config: ForecastConfig,
) -> Optional[DDHQMetric]:
    recent = _prepare_recent_polls_frame(metric.recent_polls)
    if recent.empty:
        return None
    recent = recent.loc[recent["end_date"].dt.date <= as_of_date].copy()
    recent = recent.loc[recent["margin_a"].notna()].copy()
    if recent.empty:
        return None
    recent = _compute_poll_weights(recent, config, as_of_date)
    recent = recent.loc[recent["weight"] > 0].copy()
    if recent.empty:
        return None

    mean_margin = weighted_mean(recent["margin_a"], recent["weight"])
    if np.isnan(mean_margin):
        return None

    pct_a = pd.to_numeric(recent.get("pct_a"), errors="coerce") if "pct_a" in recent.columns else None
    pct_b = pd.to_numeric(recent.get("pct_b"), errors="coerce") if "pct_b" in recent.columns else None
    if pct_a is not None and pct_a.notna().any():
        pct_a_val = float(weighted_mean(pct_a.fillna(pct_a.mean()), recent["weight"]))
    else:
        pct_a_val = float(metric.pct_a)
    if pct_b is not None and pct_b.notna().any():
        pct_b_val = float(weighted_mean(pct_b.fillna(pct_b.mean()), recent["weight"]))
    else:
        pct_b_val = float(metric.pct_b)

    return DDHQMetric(
        metric=metric.metric,
        label_a=metric.label_a,
        label_b=metric.label_b,
        pct_a=pct_a_val,
        pct_b=pct_b_val,
        margin_a=float(mean_margin),
        polls_included=int(len(recent)),
        recent_polls=recent.drop(columns=[c for c in ["age_days", "population_weight", "recency_weight", "sample_weight", "weight"] if c in recent.columns]),
        source_url=metric.source_url,
        is_live=metric.is_live,
    )


# ---------------------------------------------------------------------------
# National environment
# ---------------------------------------------------------------------------


def estimate_national_environment(
    generic: DDHQMetric,
    approval: DDHQMetric,
    right_track: DDHQMetric,
    config: ForecastConfig,
    today: Optional[date] = None,
) -> NationalEnvironment:
    today = today or date.today()
    days = max((ELECTION_DATE - today).days, 0)

    current_mean = generic.margin_a
    current_sd = config.national_poll_floor_sd
    recent = _prepare_recent_polls_frame(generic.recent_polls)
    recent = recent.loc[recent["margin_a"].notna()].copy()
    if not recent.empty:
        recent = _compute_poll_weights(recent, config, today)
        if recent["weight"].sum() > 0:
            recent_mean = weighted_mean(recent["margin_a"], recent["weight"])
            recent_sd = weighted_std(recent["margin_a"], recent["weight"])
            if not np.isnan(recent_mean):
                current_mean = 0.85 * generic.margin_a + 0.15 * recent_mean
            if not np.isnan(recent_sd):
                current_sd = max(
                    config.national_poll_floor_sd,
                    0.45 * recent_sd,
                    5.5 / math.sqrt(max(len(recent), 1)),
                )
    election_day_sd = math.sqrt(
        current_sd * current_sd
        + (config.national_daily_random_walk_sd ** 2) * days
        + config.national_election_day_shock_sd ** 2
    )
    return NationalEnvironment(
        mean_margin_dem=float(current_mean),
        current_sd=float(current_sd),
        election_day_sd=float(election_day_sd),
        generic_ballot_margin_dem=float(generic.margin_a),
        trump_net_approval=float(approval.margin_a),
        right_track_net=float(right_track.margin_a),
        polls_included=generic.polls_included,
        days_to_election=int(days),
    )


def _history_progress(day: date, history_start: date, history_end: date) -> float:
    span = max((history_end - history_start).days, 1)
    return float(clip((day - history_start).days / span, 0.0, 1.0))


def _history_neutral_weight(progress: float, config: ForecastConfig) -> float:
    return float(
        config.history_neutral_prior_max_weight * ((1.0 - clip(progress, 0.0, 1.0)) ** config.history_neutral_prior_alpha)
    )


def _history_campaign_weight(progress: float, config: ForecastConfig) -> float:
    return float(clip(progress, 0.0, 1.0) ** config.history_campaign_signal_alpha)


def _neutral_generic_metric(template: DDHQMetric, config: ForecastConfig, mean_margin: Optional[float] = None) -> DDHQMetric:
    mean_margin = float(config.history_neutral_prior_mean_dem_margin if mean_margin is None else mean_margin)
    pct_a = 50.0 + mean_margin / 2.0
    pct_b = 50.0 - mean_margin / 2.0
    return DDHQMetric(
        metric=template.metric,
        label_a=template.label_a,
        label_b=template.label_b,
        pct_a=pct_a,
        pct_b=pct_b,
        margin_a=mean_margin,
        polls_included=0,
        recent_polls=pd.DataFrame(columns=["field_dates", "end_date", "sample_size", "population", "pollster", "pct_a", "pct_b", "margin_a"]),
        source_url=template.source_url,
        is_live=template.is_live,
    )


def estimate_history_national_environment(
    generic: DDHQMetric,
    approval: DDHQMetric,
    right_track: DDHQMetric,
    config: ForecastConfig,
    day: date,
    history_start: date,
    history_end: date,
    neutral_prior_mean_dem_margin: Optional[float] = None,
) -> tuple[NationalEnvironment, float, int]:
    progress = _history_progress(day, history_start, history_end)
    neutral_weight = _history_neutral_weight(progress, config)
    neutral_mean = float(config.history_neutral_prior_mean_dem_margin if neutral_prior_mean_dem_margin is None else neutral_prior_mean_dem_margin)

    approval_day = metric_snapshot_from_recent_polls(approval, day, config) or approval
    right_track_day = metric_snapshot_from_recent_polls(right_track, day, config) or right_track
    generic_day = metric_snapshot_from_recent_polls(generic, day, config)

    if generic_day is None:
        base_env = estimate_national_environment(
            _neutral_generic_metric(generic, config, mean_margin=neutral_mean),
            approval_day,
            right_track_day,
            config,
            today=day,
        )
        current_sd = max(base_env.current_sd, config.history_neutral_current_sd)
        days = max((ELECTION_DATE - day).days, 0)
        election_day_sd = math.sqrt(
            current_sd * current_sd
            + (config.national_daily_random_walk_sd ** 2) * days
            + config.national_election_day_shock_sd ** 2
        )
        return (
            NationalEnvironment(
                mean_margin_dem=float(neutral_mean),
                current_sd=float(current_sd),
                election_day_sd=float(election_day_sd),
                generic_ballot_margin_dem=float(neutral_mean),
                trump_net_approval=float(base_env.trump_net_approval),
                right_track_net=float(base_env.right_track_net),
                polls_included=0,
                days_to_election=int(days),
            ),
            1.0,
            0,
        )

    base_env = estimate_national_environment(generic_day, approval_day, right_track_day, config, today=day)
    mean_margin = (
        (1.0 - neutral_weight) * base_env.mean_margin_dem
        + neutral_weight * neutral_mean
    )
    generic_margin = (
        (1.0 - neutral_weight) * base_env.generic_ballot_margin_dem
        + neutral_weight * neutral_mean
    )
    current_sd = math.sqrt(base_env.current_sd ** 2 + (neutral_weight * config.history_neutral_current_sd) ** 2)
    days = max((ELECTION_DATE - day).days, 0)
    election_day_sd = math.sqrt(
        current_sd * current_sd
        + (config.national_daily_random_walk_sd ** 2) * days
        + config.national_election_day_shock_sd ** 2
    )
    return (
        NationalEnvironment(
            mean_margin_dem=float(mean_margin),
            current_sd=float(current_sd),
            election_day_sd=float(election_day_sd),
            generic_ballot_margin_dem=float(generic_margin),
            trump_net_approval=float(base_env.trump_net_approval),
            right_track_net=float(base_env.right_track_net),
            polls_included=generic_day.polls_included,
            days_to_election=int(days),
        ),
        float(neutral_weight),
        int(generic_day.polls_included or 0),
    )


# ---------------------------------------------------------------------------
# District master frame and priors
# ---------------------------------------------------------------------------


def _summarize_candidate_field(candidates_df: pd.DataFrame) -> pd.DataFrame:
    if candidates_df.empty:
        return pd.DataFrame(columns=["district_code"])
    df = candidates_df.copy()
    df["party_bucket"] = np.where(
        df["party"].astype(str).str.startswith("DEM"),
        "DEM",
        np.where(df["party"].astype(str).str.startswith("REP"), "REP", "OTHER"),
    )
    rows: list[dict[str, Any]] = []
    for district_code, g in df.groupby("district_code"):
        incumbent = g[g["incumbent_challenge"].astype(str).str.upper() == "I"]
        open_cands = g[g["incumbent_challenge"].astype(str).str.upper() == "O"]
        rows.append(
            {
                "district_code": district_code,
                "incumbent_running": not incumbent.empty,
                "incumbent_party": incumbent["party_bucket"].iloc[0] if not incumbent.empty else None,
                "open_seat_fec": (not open_cands.empty) and incumbent.empty,
                "dem_candidate_count": int((g["party_bucket"] == "DEM").sum()),
                "rep_candidate_count": int((g["party_bucket"] == "REP").sum()),
                "any_major_party_candidate": bool(((g["party_bucket"] == "DEM") | (g["party_bucket"] == "REP")).any()),
            }
        )
    return pd.DataFrame(rows)


def _apply_finance_effect(master: pd.DataFrame, config: ForecastConfig) -> pd.DataFrame:
    df = master.copy()
    for col in [
        "dem_receipts",
        "rep_receipts",
        "dem_itemized",
        "rep_itemized",
        "dem_cash_on_hand",
        "rep_cash_on_hand",
    ]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    dem_strength = (
        0.50 * np.log1p(df["dem_cash_on_hand"].fillna(0.0))
        + 0.35 * np.log1p(df["dem_receipts"].fillna(0.0))
        + 0.15 * np.log1p(df["dem_itemized"].fillna(0.0))
    )
    rep_strength = (
        0.50 * np.log1p(df["rep_cash_on_hand"].fillna(0.0))
        + 0.35 * np.log1p(df["rep_receipts"].fillna(0.0))
        + 0.15 * np.log1p(df["rep_itemized"].fillna(0.0))
    )
    raw_gap = dem_strength - rep_strength
    valid = np.isfinite(raw_gap)
    if valid.sum() >= 8:
        mean_gap = float(np.nanmean(raw_gap[valid]))
        std_gap = float(np.nanstd(raw_gap[valid])) or 1.0
        z = (raw_gap - mean_gap) / std_gap
        df["finance_gap_z"] = z
        df["finance_effect"] = np.tanh(z / 1.5) * config.finance_effect_beta
        df["finance_effect"] = df["finance_effect"].clip(
            lower=-config.finance_effect_cap, upper=config.finance_effect_cap
        )
    else:
        df["finance_gap_z"] = np.nan
        df["finance_effect"] = 0.0

    if {"dem_candidate", "rep_candidate"}.issubset(df.columns):
        both_sides = df[["dem_candidate", "rep_candidate"]].notna().all(axis=1)
        df.loc[~both_sides, "finance_effect"] = 0.0
    return df


def build_master_frame(
    baseline_df: pd.DataFrame,
    pres_df: pd.DataFrame,
    open_seats_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    candidate_overview: pd.DataFrame,
    district_finance: pd.DataFrame,
    config: ForecastConfig,
) -> pd.DataFrame:
    master = baseline_df.copy()

    candidate_field = _summarize_candidate_field(candidate_overview)
    for merge_df in [pres_df, open_seats_df, ratings_df, candidate_field, district_finance]:
        if merge_df is None or merge_df.empty:
            continue
        merge_df = merge_df.copy()
        if "district_code" not in merge_df.columns:
            continue
        merge_df = merge_df[merge_df["district_code"].notna()].drop_duplicates(subset=["district_code"])
        master = master.merge(merge_df, on="district_code", how="left")

    if "open_seat_y" in master.columns:
        master["open_seat"] = master["open_seat_y"].astype("boolean").fillna(False).astype(bool)
    elif "open_seat" in master.columns:
        master["open_seat"] = master["open_seat"].astype("boolean").fillna(False).astype(bool)
    else:
        master["open_seat"] = False
    if "open_seat_fec" in master.columns:
        master["open_seat"] = master["open_seat"].astype("boolean").fillna(False).astype(bool) | master["open_seat_fec"].fillna(False).astype(bool)
    if "incumbent_running" in master.columns:
        master.loc[master["incumbent_running"].astype("boolean").fillna(False).astype(bool), "open_seat"] = False
    else:
        master["incumbent_running"] = False

    master["incumbent_running"] = master["incumbent_running"].astype("boolean").fillna(False).astype(bool)
    if "rating" not in master.columns:
        master["rating"] = np.nan

    master = _apply_finance_effect(master, config)

    master["house_margin_2024_clipped"] = master["two_party_dem_margin_2024"].clip(-40, 40)
    master["has_pres_data"] = (
        master[["pres24_dem_margin", "pres20_dem_margin"]].notna().all(axis=1)
        if {"pres24_dem_margin", "pres20_dem_margin"}.issubset(master.columns)
        else False
    )
    return master.sort_values(["state_abbr", "district_code"]).reset_index(drop=True)


def aggregate_district_polls(
    district_polls: pd.DataFrame,
    config: ForecastConfig,
    as_of_date: Optional[date] = None,
) -> pd.DataFrame:
    if district_polls is None or district_polls.empty:
        return pd.DataFrame(columns=["district_code", "poll_margin_mean", "poll_margin_sd", "poll_count"])
    as_of_date = as_of_date or date.today()
    df = district_polls.copy()
    if "end_date" in df.columns:
        df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
        df = df.loc[df["end_date"].isna() | (df["end_date"].dt.date <= as_of_date)].copy()
        df["age_days"] = (
            pd.Timestamp(as_of_date) - df["end_date"].fillna(pd.Timestamp(as_of_date))
        ).dt.days.clip(lower=0)
    else:
        df["age_days"] = 0
    df["sample_size"] = pd.to_numeric(df["sample_size"], errors="coerce").fillna(600)
    df["margin_dem"] = pd.to_numeric(df["margin_dem"], errors="coerce")
    df = df[df["margin_dem"].notna()].copy()
    if df.empty:
        return pd.DataFrame(columns=["district_code", "poll_margin_mean", "poll_margin_sd", "poll_count"])

    df["recency_weight"] = 0.5 ** (df["age_days"] / max(config.district_poll_recency_half_life_days, 1))
    df["sample_weight"] = np.sqrt(df["sample_size"].clip(lower=100))
    df["weight"] = df["recency_weight"] * df["sample_weight"]
    df["poll_sd_each"] = df["sample_size"].map(lambda n: sample_to_margin_sd(int(n), config.district_poll_min_sd))

    rows: list[dict[str, Any]] = []
    for district_code, g in df.groupby("district_code"):
        weights = g["weight"].to_numpy(dtype=float)
        margins = g["margin_dem"].to_numpy(dtype=float)
        mean_margin = float(np.average(margins, weights=weights))
        dispersion = weighted_std(margins, weights)
        precision_sum = float(np.sum(1.0 / np.square(g["poll_sd_each"].to_numpy(dtype=float))))
        sampling_sd = math.sqrt(1.0 / precision_sum) if precision_sum > 0 else config.district_poll_min_sd
        poll_sd = max(config.district_poll_min_sd, sampling_sd)
        if not np.isnan(dispersion):
            poll_sd = max(poll_sd, 0.55 * dispersion)
        rows.append(
            {
                "district_code": district_code,
                "poll_margin_mean": mean_margin,
                "poll_margin_sd": poll_sd,
                "poll_count": int(len(g)),
            }
        )
    return pd.DataFrame(rows)


def prepare_district_priors(
    master: pd.DataFrame,
    national: NationalEnvironment,
    district_poll_agg: pd.DataFrame,
    config: ForecastConfig,
    signal_weights: Optional[dict[str, float]] = None,
) -> pd.DataFrame:
    signal_weights = signal_weights or {}
    open_seat_weight = float(clip(signal_weights.get("open_seat", 1.0), 0.0, 1.0))
    rating_weight = float(clip(signal_weights.get("rating", 1.0), 0.0, 1.0))
    finance_weight = float(clip(signal_weights.get("finance", 1.0), 0.0, 1.0))
    district_poll_weight = float(clip(signal_weights.get("district_polls", 1.0), 0.0, 1.0))

    df = master.copy()
    if district_poll_agg is not None and not district_poll_agg.empty:
        df = df.merge(district_poll_agg, on="district_code", how="left")
    else:
        df["poll_margin_mean"] = np.nan
        df["poll_margin_sd"] = np.nan
        df["poll_count"] = 0

    default_non_open_sd = (config.base_prior_sd_incumbent + config.base_prior_sd_open) / 2.0
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        has_pres = bool(row.get("has_pres_data", False))
        same_incumbent = bool(row.get("incumbent_running", False))
        open_seat = bool(row.get("open_seat", False))

        if has_pres:
            lean24 = float(row["pres24_dem_margin"]) - NATIONAL_PRES_MARGIN_2024
            lean20 = float(row["pres20_dem_margin"]) - NATIONAL_PRES_MARGIN_2020
            lean = config.pres_weight_2024 * lean24 + (1.0 - config.pres_weight_2024) * lean20
            expected_house24 = NATIONAL_HOUSE_MARGIN_2024 + lean24
            overperf = clip(
                float(row["house_margin_2024_clipped"]) - expected_house24,
                -config.house_overperf_clip,
                config.house_overperf_clip,
            )
            if same_incumbent:
                base_sd = config.base_prior_sd_incumbent
            elif open_seat:
                base_sd = (
                    (1.0 - open_seat_weight) * default_non_open_sd
                    + open_seat_weight * config.base_prior_sd_open
                )
            else:
                base_sd = default_non_open_sd
        else:
            lean = 0.85 * (float(row["house_margin_2024_clipped"]) - NATIONAL_HOUSE_MARGIN_2024)
            overperf = 0.0
            base_sd = config.base_prior_sd_missing_pres

        if same_incumbent:
            rho = config.carryover_same_incumbent
        elif open_seat:
            rho = (
                (1.0 - open_seat_weight) * config.carryover_non_open_nonincumbent
                + open_seat_weight * config.carryover_open_seat
            )
        else:
            rho = config.carryover_non_open_nonincumbent

        applied_finance_effect = finance_weight * float(row.get("finance_effect", 0.0) or 0.0)
        intercept_mean = lean + rho * overperf + applied_finance_effect
        intercept_sd = math.sqrt(base_sd ** 2 + config.extra_future_district_sd ** 2)

        rating = row.get("rating")
        if rating_weight > 0 and pd.notna(rating) and str(rating) in config.rating_margin_map:
            rating_margin = config.rating_margin_map[str(rating)]
            rating_sd = config.rating_sd_map[str(rating)] / math.sqrt(max(rating_weight, 1e-9))
            rating_intercept = rating_margin - national.mean_margin_dem
            intercept_mean, intercept_sd = combine_normal_estimates(
                intercept_mean, intercept_sd, rating_intercept, rating_sd
            )

        if district_poll_weight > 0 and pd.notna(row.get("poll_margin_mean")) and pd.notna(row.get("poll_margin_sd")):
            poll_intercept = float(row["poll_margin_mean"]) - national.mean_margin_dem
            poll_sd = float(row["poll_margin_sd"]) / math.sqrt(max(district_poll_weight, 1e-9))
            intercept_mean, intercept_sd = combine_normal_estimates(
                intercept_mean,
                intercept_sd,
                poll_intercept,
                poll_sd,
            )

        mean_margin = national.mean_margin_dem + intercept_mean
        approx_total_sd = math.sqrt(
            intercept_sd ** 2 + national.election_day_sd ** 2 + config.state_correlation_sd ** 2
        )
        dem_win_prob = 1.0 - norm.cdf(0.0, loc=mean_margin, scale=approx_total_sd)
        gop_win_prob = 1.0 - dem_win_prob
        rows.append(
            {
                "district_code": row["district_code"],
                "state_abbr": row["state_abbr"],
                "district_name": row.get("district_name"),
                "latitude": row.get("latitude"),
                "longitude": row.get("longitude"),
                "winner_party_2024": row.get("winner_party_2024"),
                "winner_2024": row.get("winner_2024"),
                "open_seat": open_seat,
                "incumbent_running": same_incumbent,
                "rating": row.get("rating"),
                "intercept_mean": intercept_mean,
                "intercept_sd": intercept_sd,
                "mean_margin_dem": mean_margin,
                "approx_total_sd": approx_total_sd,
                "dem_win_prob_analytic": dem_win_prob,
                "gop_win_prob_analytic": gop_win_prob,
                "finance_effect": applied_finance_effect,
                "raw_finance_effect": float(row.get("finance_effect", 0.0) or 0.0),
                "poll_count": int(row.get("poll_count", 0) or 0),
                "poll_margin_mean": row.get("poll_margin_mean"),
                "dem_candidate": row.get("dem_candidate"),
                "rep_candidate": row.get("rep_candidate"),
                "dem_cash_on_hand": row.get("dem_cash_on_hand"),
                "rep_cash_on_hand": row.get("rep_cash_on_hand"),
                "dem_receipts": row.get("dem_receipts"),
                "rep_receipts": row.get("rep_receipts"),
                "house_margin_2024": row.get("two_party_dem_margin_2024"),
                "pres24_dem_margin": row.get("pres24_dem_margin"),
                "pres20_dem_margin": row.get("pres20_dem_margin"),
                "applied_open_seat_weight": open_seat_weight if open_seat else 0.0,
                "applied_rating_weight": rating_weight if pd.notna(rating) else 0.0,
                "applied_finance_weight": finance_weight if pd.notna(row.get("finance_effect")) else 0.0,
                "applied_district_poll_weight": district_poll_weight if int(row.get("poll_count", 0) or 0) > 0 else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(by=["state_abbr", "district_code"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Simulation engine
# ---------------------------------------------------------------------------


def choose_poll_scan_districts(
    priors: pd.DataFrame,
    open_seats: pd.DataFrame,
    ratings: pd.DataFrame,
    config: ForecastConfig,
) -> list[str]:
    selected: set[str] = set()
    if open_seats is not None and not open_seats.empty:
        selected.update(open_seats.loc[open_seats["open_seat"] == True, "district_code"].dropna().tolist())
    if ratings is not None and not ratings.empty:
        selected.update(ratings["district_code"].dropna().tolist())
    if priors is not None and not priors.empty:
        close = priors.loc[
            priors["mean_margin_dem"].abs() <= config.competitive_poll_margin_threshold,
            ["district_code", "mean_margin_dem"],
        ].sort_values("mean_margin_dem", key=lambda s: s.abs())
        selected.update(close["district_code"].tolist())
        if len(selected) < config.competitive_poll_limit:
            extra = priors.sort_values("mean_margin_dem", key=lambda s: s.abs())["district_code"].tolist()
            for code in extra:
                selected.add(code)
                if len(selected) >= config.competitive_poll_limit:
                    break
    return sorted(selected, key=district_sort_key)


def _summary_from_seat_draws(seat_draws_gop: np.ndarray) -> SimulationSummary:
    return SimulationSummary(
        gop_control_prob=float(np.mean(seat_draws_gop >= 218)),
        expected_gop_seats=float(np.mean(seat_draws_gop)),
        median_gop_seats=float(np.median(seat_draws_gop)),
        gop_seat_q05=float(np.quantile(seat_draws_gop, 0.05)),
        gop_seat_q25=float(np.quantile(seat_draws_gop, 0.25)),
        gop_seat_q75=float(np.quantile(seat_draws_gop, 0.75)),
        gop_seat_q95=float(np.quantile(seat_draws_gop, 0.95)),
        dem_control_prob=float(np.mean(seat_draws_gop <= 217)),
    )


def run_simulation(
    priors: pd.DataFrame,
    national: NationalEnvironment,
    config: ForecastConfig,
    seed: int = 20260401,
) -> SimulationOutputs:
    rng = np.random.default_rng(seed)
    sims = config.simulations
    national_draws = rng.normal(loc=national.mean_margin_dem, scale=national.election_day_sd, size=sims)
    state_codes = sorted([x for x in priors["state_abbr"].dropna().unique().tolist() if isinstance(x, str)])
    state_error = {st: rng.normal(loc=0.0, scale=config.state_correlation_sd, size=sims) for st in state_codes}

    seat_draws_gop = np.zeros(sims, dtype=np.int16)
    district_rows: list[dict[str, Any]] = []

    for _, row in priors.iterrows():
        intercept_draws = rng.normal(loc=float(row["intercept_mean"]), scale=float(row["intercept_sd"]), size=sims)
        margins = national_draws + state_error[row["state_abbr"]] + intercept_draws
        dem_win = margins > 0.0
        gop_win = ~dem_win
        seat_draws_gop += gop_win.astype(np.int16)
        district_rows.append(
            {
                **row.to_dict(),
                "mean_margin_sim": float(np.mean(margins)),
                "median_margin_sim": float(np.median(margins)),
                "p05_margin_sim": float(np.quantile(margins, 0.05)),
                "p95_margin_sim": float(np.quantile(margins, 0.95)),
                "dem_win_prob": float(np.mean(dem_win)),
                "gop_win_prob": float(np.mean(gop_win)),
            }
        )

    district_results = pd.DataFrame(district_rows)
    bins = np.bincount(seat_draws_gop, minlength=436)
    seat_distribution = pd.DataFrame(
        {
            "gop_seats": np.arange(len(bins)),
            "frequency": bins,
            "probability": bins / bins.sum(),
        }
    )
    summary = _summary_from_seat_draws(seat_draws_gop)
    return SimulationOutputs(
        district_results=district_results,
        seat_distribution=seat_distribution,
        summary=summary,
        seat_draws=seat_draws_gop,
    )


def prepare_common_summary_draws(
    priors: pd.DataFrame,
    config: ForecastConfig,
    seed: int,
    simulations: Optional[int] = None,
) -> SummaryRandomDraws:
    sims = simulations or config.history_simulations
    rng = np.random.default_rng(seed)
    state_codes = sorted([x for x in priors["state_abbr"].dropna().unique().tolist() if isinstance(x, str)])
    z_state = {st: rng.standard_normal(sims) for st in state_codes}
    z_district = rng.standard_normal((len(priors), sims))
    return SummaryRandomDraws(
        simulations=sims,
        z_national=rng.standard_normal(sims),
        z_state=z_state,
        z_district=z_district,
    )


def run_simulation_summary_only(
    priors: pd.DataFrame,
    national: NationalEnvironment,
    config: ForecastConfig,
    draws: Optional[SummaryRandomDraws] = None,
    seed: int = 20260401,
    simulations: Optional[int] = None,
) -> tuple[SimulationSummary, np.ndarray]:
    draws = draws or prepare_common_summary_draws(priors, config, seed=seed, simulations=simulations)
    sims = draws.simulations
    national_draws = national.mean_margin_dem + national.election_day_sd * draws.z_national

    seat_draws_gop = np.zeros(sims, dtype=np.int16)
    intercept_mean = priors["intercept_mean"].to_numpy(dtype=float)
    intercept_sd = priors["intercept_sd"].to_numpy(dtype=float)
    states = priors["state_abbr"].astype(str).tolist()

    for idx in range(len(priors)):
        margins = (
            national_draws
            + config.state_correlation_sd * draws.z_state[states[idx]]
            + intercept_mean[idx]
            + intercept_sd[idx] * draws.z_district[idx]
        )
        seat_draws_gop += (margins <= 0.0).astype(np.int16)

    return _summary_from_seat_draws(seat_draws_gop), seat_draws_gop


# ---------------------------------------------------------------------------
# Output persistence and trend reconstruction
# ---------------------------------------------------------------------------


def solve_control_neutral_national_margin(
    master: pd.DataFrame,
    approval: DDHQMetric,
    right_track: DDHQMetric,
    config: ForecastConfig,
    history_start: date,
    seed: int = 20260401,
) -> float:
    neutral_env = estimate_national_environment(
        DDHQMetric(
            metric="generic_ballot",
            label_a="Democrat",
            label_b="Republican",
            pct_a=50.0,
            pct_b=50.0,
            margin_a=0.0,
            polls_included=0,
            recent_polls=pd.DataFrame(columns=["margin_a"]),
            source_url="",
            is_live=False,
        ),
        approval,
        right_track,
        config,
        today=history_start,
    )
    neutral_current_sd = max(neutral_env.current_sd, config.history_neutral_current_sd)
    neutral_election_day_sd = math.sqrt(
        neutral_current_sd * neutral_current_sd
        + (config.national_daily_random_walk_sd ** 2) * neutral_env.days_to_election
        + config.national_election_day_shock_sd ** 2
    )
    neutral_env = NationalEnvironment(
        mean_margin_dem=0.0,
        current_sd=neutral_current_sd,
        election_day_sd=neutral_election_day_sd,
        generic_ballot_margin_dem=0.0,
        trump_net_approval=neutral_env.trump_net_approval,
        right_track_net=neutral_env.right_track_net,
        polls_included=0,
        days_to_election=neutral_env.days_to_election,
    )
    priors = prepare_district_priors(
        master,
        neutral_env,
        pd.DataFrame(),
        config,
        signal_weights={"open_seat": 0.0, "rating": 0.0, "finance": 0.0, "district_polls": 0.0},
    )
    draws = prepare_common_summary_draws(priors, config, seed=seed + 911, simulations=config.history_simulations)
    low, high = -4.0, 4.0
    for _ in range(14):
        mid = (low + high) / 2.0
        env_mid = NationalEnvironment(
            mean_margin_dem=mid,
            current_sd=neutral_env.current_sd,
            election_day_sd=neutral_env.election_day_sd,
            generic_ballot_margin_dem=mid,
            trump_net_approval=neutral_env.trump_net_approval,
            right_track_net=neutral_env.right_track_net,
            polls_included=0,
            days_to_election=neutral_env.days_to_election,
        )
        summary_mid, _ = run_simulation_summary_only(priors, env_mid, config, draws=draws)
        if summary_mid.gop_control_prob > 0.5:
            low = mid
        else:
            high = mid
    return float((low + high) / 2.0)


def append_run_history(paths: ProjectPaths, snapshot: dict[str, Any]) -> pd.DataFrame:
    history_path = paths.run_history_csv
    if history_path.exists():
        hist = pd.read_csv(history_path)
    else:
        hist = pd.DataFrame()
    new_row = pd.DataFrame([snapshot])
    if not hist.empty and "as_of_date" in hist.columns:
        hist = hist[hist["as_of_date"] != snapshot["as_of_date"]].copy()
    hist = pd.concat([hist, new_row], ignore_index=True)
    hist = hist.sort_values("as_of_date").reset_index(drop=True)
    hist.to_csv(history_path, index=False)
    return hist


def _history_row_from_summary(
    as_of_date: date,
    summary: SimulationSummary,
    national: NationalEnvironment,
    priors: pd.DataFrame,
    polls_used: Optional[int],
    is_observed_run: bool,
) -> dict[str, Any]:
    return {
        "as_of_date": as_of_date.isoformat(),
        "gop_control_prob": summary.gop_control_prob,
        "expected_gop_seats": summary.expected_gop_seats,
        "median_gop_seats": summary.median_gop_seats,
        "gop_seat_q05": summary.gop_seat_q05,
        "gop_seat_q25": summary.gop_seat_q25,
        "gop_seat_q75": summary.gop_seat_q75,
        "gop_seat_q95": summary.gop_seat_q95,
        "dem_control_prob": summary.dem_control_prob,
        "generic_ballot_margin_dem": national.generic_ballot_margin_dem,
        "national_mean_margin_dem": national.mean_margin_dem,
        "national_election_day_sd": national.election_day_sd,
        "days_to_election": national.days_to_election,
        "districts_polled": int((priors["poll_count"] > 0).sum()) if not priors.empty else 0,
        "districts_open": int(priors["open_seat"].sum()) if not priors.empty else 0,
        "visible_national_poll_rows": int(polls_used or 0),
        "is_observed_run": bool(is_observed_run),
    }


def build_history_curve(
    paths: ProjectPaths,
    master: pd.DataFrame,
    generic: DDHQMetric,
    approval: DDHQMetric,
    right_track: DDHQMetric,
    district_polls_raw: pd.DataFrame,
    config: ForecastConfig,
    as_of_date: date,
    current_summary: SimulationSummary,
    current_priors: pd.DataFrame,
    current_national: NationalEnvironment,
    seed: int = 20260401,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    run_rows = pd.DataFrame()

    history_start = as_of_date - timedelta(days=max(config.history_max_days - 1, 0)) if config.history_max_days > 0 else as_of_date
    full_dates = list(pd.date_range(start=history_start, end=as_of_date, freq="D").date)

    recent = _prepare_recent_polls_frame(generic.recent_polls)
    poll_dates: set[date] = set()
    if not recent.empty and "end_date" in recent.columns:
        poll_dates = set(recent["end_date"].dropna().dt.date.tolist())
    neutral_control_margin = solve_control_neutral_national_margin(
        master,
        approval,
        right_track,
        config,
        history_start,
        seed=seed,
    )

    anchor_dates: list[date] = []
    for day in full_dates:
        days_back = (as_of_date - day).days
        include = False
        if day in {history_start, as_of_date}:
            include = True
        elif day in poll_dates:
            include = True
        elif days_back <= 45:
            include = True
        elif days_back <= 120 and (days_back % 7 == 0):
            include = True
        elif days_back % 14 == 0:
            include = True
        if include:
            anchor_dates.append(day)
    anchor_dates = sorted(set(anchor_dates))

    common_draws: Optional[SummaryRandomDraws] = None
    for day in anchor_dates:
        progress = _history_progress(day, history_start, as_of_date)
        campaign_weight = _history_campaign_weight(progress, config)
        national_day, neutral_weight, polls_used = estimate_history_national_environment(
            generic,
            approval,
            right_track,
            config,
            day,
            history_start,
            as_of_date,
            neutral_prior_mean_dem_margin=neutral_control_margin,
        )
        district_poll_day = aggregate_district_polls(district_polls_raw, config, as_of_date=day)
        priors_day = prepare_district_priors(
            master,
            national_day,
            district_poll_day,
            config,
            signal_weights={
                "open_seat": campaign_weight,
                "rating": campaign_weight,
                "finance": campaign_weight,
                "district_polls": 1.0,
            },
        )
        if common_draws is None:
            common_draws = prepare_common_summary_draws(
                priors_day,
                config,
                seed=seed + 911,
                simulations=config.history_simulations,
            )
        summary_day, _ = run_simulation_summary_only(
            priors_day,
            national_day,
            config,
            draws=common_draws,
        )
        row = _history_row_from_summary(
            day,
            summary_day,
            national_day,
            priors_day,
            polls_used,
            is_observed_run=False,
        )
        row["neutral_prior_weight"] = neutral_weight
        row["campaign_signal_weight"] = campaign_weight
        rows.append(row)

    current_row = _history_row_from_summary(
        as_of_date,
        current_summary,
        current_national,
        current_priors,
        current_national.polls_included,
        is_observed_run=True,
    )
    current_row["neutral_prior_weight"] = 0.0
    current_row["campaign_signal_weight"] = 1.0

    anchor_df = pd.DataFrame(rows)
    if anchor_df.empty:
        curve = pd.DataFrame([current_row])
    else:
        anchor_df = anchor_df[anchor_df["as_of_date"] != current_row["as_of_date"]].copy()
        anchor_df = pd.concat([anchor_df, pd.DataFrame([current_row])], ignore_index=True)
        anchor_df["as_of_date"] = pd.to_datetime(anchor_df["as_of_date"], errors="coerce")
        anchor_df = anchor_df.sort_values("as_of_date").set_index("as_of_date")

        curve = anchor_df.reindex(pd.to_datetime(full_dates))
        numeric_cols = [
            c for c in [
                "gop_control_prob",
                "expected_gop_seats",
                "median_gop_seats",
                "gop_seat_q05",
                "gop_seat_q25",
                "gop_seat_q75",
                "gop_seat_q95",
                "dem_control_prob",
                "generic_ballot_margin_dem",
                "national_mean_margin_dem",
                "national_election_day_sd",
                "days_to_election",
                "districts_polled",
                "districts_open",
                "visible_national_poll_rows",
                "neutral_prior_weight",
                "campaign_signal_weight",
            ]
            if c in curve.columns
        ]
        for col in numeric_cols:
            curve[col] = pd.to_numeric(curve[col], errors="coerce").interpolate(method="linear", limit_direction="both")

        for col in ["days_to_election", "districts_polled", "districts_open", "visible_national_poll_rows"]:
            if col in curve.columns:
                curve[col] = curve[col].round().astype(int)

        curve["is_observed_run"] = False
        curve.loc[pd.Timestamp(as_of_date), "is_observed_run"] = True
        curve.index.name = "as_of_date"
        curve = curve.reset_index()
        curve["as_of_date"] = curve["as_of_date"].dt.date.astype(str)

    if paths.run_history_csv.exists():
        try:
            run_rows = pd.read_csv(paths.run_history_csv)
        except Exception:
            run_rows = pd.DataFrame()
    paths.forecast_history_csv.parent.mkdir(parents=True, exist_ok=True)
    curve.to_csv(paths.forecast_history_csv, index=False)
    curve.to_csv(paths.forecast_curve_csv, index=False)
    return curve, run_rows

def write_outputs(
    paths: ProjectPaths,
    national: NationalEnvironment,
    generic: DDHQMetric,
    approval: DDHQMetric,
    right_track: DDHQMetric,
    master: pd.DataFrame,
    priors: pd.DataFrame,
    outputs: SimulationOutputs,
    source_status: dict[str, Any],
    as_of_date: date,
    config: ForecastConfig,
    district_polls_raw: Optional[pd.DataFrame] = None,
    seed: int = 20260401,
) -> dict[str, Any]:
    paths.latest_dir.mkdir(parents=True, exist_ok=True)
    outputs.district_results.to_csv(paths.latest_dir / "district_forecast.csv", index=False)
    outputs.seat_distribution.to_csv(paths.latest_dir / "seat_distribution.csv", index=False)
    priors.to_csv(paths.latest_dir / "district_priors.csv", index=False)
    master.to_csv(paths.latest_dir / "district_master.csv", index=False)

    recent = _prepare_recent_polls_frame(generic.recent_polls)
    poll_start = None
    poll_end = None
    if not recent.empty and recent["end_date"].notna().any():
        poll_start = recent["end_date"].dropna().min().date().isoformat()
        poll_end = recent["end_date"].dropna().max().date().isoformat()

    run_snapshot = {
        "as_of_date": as_of_date.isoformat(),
        "gop_control_prob": outputs.summary.gop_control_prob,
        "expected_gop_seats": outputs.summary.expected_gop_seats,
        "median_gop_seats": outputs.summary.median_gop_seats,
        "gop_seat_q05": outputs.summary.gop_seat_q05,
        "gop_seat_q25": outputs.summary.gop_seat_q25,
        "gop_seat_q75": outputs.summary.gop_seat_q75,
        "gop_seat_q95": outputs.summary.gop_seat_q95,
        "dem_control_prob": outputs.summary.dem_control_prob,
        "generic_ballot_margin_dem": national.generic_ballot_margin_dem,
        "trump_net_approval": national.trump_net_approval,
        "right_track_net": national.right_track_net,
        "districts_polled": int((priors["poll_count"] > 0).sum()),
        "districts_open": int(priors["open_seat"].sum()),
        "national_mean_margin_dem": national.mean_margin_dem,
        "national_election_day_sd": national.election_day_sd,
        "visible_national_poll_rows": int(len(recent)) if not recent.empty else 0,
    }
    run_history = append_run_history(paths, run_snapshot)

    history_curve, _ = build_history_curve(
        paths,
        master,
        generic,
        approval,
        right_track,
        district_polls_raw if district_polls_raw is not None else pd.DataFrame(),
        config,
        as_of_date,
        outputs.summary,
        priors,
        national,
        seed=seed,
    )

    summary_dict = {
        "as_of_date": as_of_date.isoformat(),
        "gop_control_prob": outputs.summary.gop_control_prob,
        "expected_gop_seats": outputs.summary.expected_gop_seats,
        "median_gop_seats": outputs.summary.median_gop_seats,
        "gop_seat_q05": outputs.summary.gop_seat_q05,
        "gop_seat_q25": outputs.summary.gop_seat_q25,
        "gop_seat_q75": outputs.summary.gop_seat_q75,
        "gop_seat_q95": outputs.summary.gop_seat_q95,
        "dem_control_prob": outputs.summary.dem_control_prob,
        "national_mean_margin_dem": national.mean_margin_dem,
        "national_election_day_sd": national.election_day_sd,
        "generic_ballot_margin_dem": national.generic_ballot_margin_dem,
        "trump_net_approval": national.trump_net_approval,
        "right_track_net": national.right_track_net,
        "source_status": source_status,
        "simulations": config.simulations,
        "history_simulations": config.history_simulations,
        "visible_national_poll_rows": int(len(recent)) if not recent.empty else 0,
        "visible_national_poll_start": poll_start,
        "visible_national_poll_end": poll_end,
        "history_mode": "reconstructed_backcast_with_cycle_neutral_prior",
        "history_points": int(len(history_curve)),
        "history_start_date": history_curve["as_of_date"].min() if not history_curve.empty else as_of_date.isoformat(),
        "history_end_date": history_curve["as_of_date"].max() if not history_curve.empty else as_of_date.isoformat(),
        "history_neutral_prior_start_weight": float(history_curve.iloc[0]["neutral_prior_weight"]) if (not history_curve.empty and "neutral_prior_weight" in history_curve.columns) else None,
        "history_campaign_signal_start_weight": float(history_curve.iloc[0]["campaign_signal_weight"]) if (not history_curve.empty and "campaign_signal_weight" in history_curve.columns) else None,
        "history_control_neutral_dem_margin": float(history_curve.iloc[0]["generic_ballot_margin_dem"]) if not history_curve.empty else None,
        "run_history_points": int(len(run_history)),
    }
    with (paths.latest_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(ensure_json_serializable(summary_dict), f, indent=2)

    audit = {
        "as_of_date": as_of_date.isoformat(),
        "simulations": config.simulations,
        "history_simulations": config.history_simulations,
        "source_status": source_status,
        "visible_national_poll_rows": int(len(recent)) if not recent.empty else 0,
        "visible_national_poll_start": poll_start,
        "visible_national_poll_end": poll_end,
        "history_mode": "reconstructed_backcast_with_cycle_neutral_prior",
        "history_max_days": int(config.history_max_days),
        "history_neutral_prior_max_weight": float(config.history_neutral_prior_max_weight),
        "history_neutral_prior_alpha": float(config.history_neutral_prior_alpha),
        "history_campaign_signal_alpha": float(config.history_campaign_signal_alpha),
        "history_control_neutral_dem_margin": float(history_curve.iloc[0]["generic_ballot_margin_dem"]) if not history_curve.empty else None,
        "districts_total": int(len(priors)),
        "districts_open": int(priors["open_seat"].sum()),
        "districts_with_polls": int((priors["poll_count"] > 0).sum()),
        "districts_with_ratings": int(priors["rating"].notna().sum()) if "rating" in priors.columns else 0,
        "history_points": int(len(history_curve)),
        "run_history_points": int(len(run_history)),
        "seed": int(seed),
    }
    with paths.run_audit_json.open("w", encoding="utf-8") as f:
        json.dump(ensure_json_serializable(audit), f, indent=2)

    return {
        "summary": summary_dict,
        "history": history_curve,
        "run_history": run_history,
        "audit": audit,
    }



# ---------------------------------------------------------------------------
# Comprehensive generic-ballot archive / state-space overrides
# ---------------------------------------------------------------------------


def _prepare_national_poll_archive_frame(recent_polls: pd.DataFrame, default_year: int = 2026) -> pd.DataFrame:
    if recent_polls is None or recent_polls.empty:
        return pd.DataFrame()
    df = recent_polls.copy()
    for col in [
        "sample_size",
        "margin_a",
        "dem_pct",
        "rep_pct",
        "rcp_dem_pct",
        "rcp_rep_pct",
        "official_dem_pct",
        "official_rep_pct",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "margin_a" not in df.columns and {"dem_pct", "rep_pct"}.issubset(df.columns):
        df["margin_a"] = df["dem_pct"] - df["rep_pct"]
    if "end_date" not in df.columns and "field_dates" in df.columns:
        df["end_date"] = df["field_dates"].map(lambda x: parse_month_day_range(str(x), default_year))
    if "published_date" not in df.columns:
        df["published_date"] = df.get("end_date")
    if "start_date" not in df.columns:
        df["start_date"] = df.get("end_date")
    df["end_date"] = pd.to_datetime(df.get("end_date"), errors="coerce")
    df["published_date"] = pd.to_datetime(df.get("published_date"), errors="coerce")
    df["start_date"] = pd.to_datetime(df.get("start_date"), errors="coerce")
    if "population" in df.columns:
        df["population"] = (
            df["population"]
            .astype(str)
            .str.upper()
            .replace({"A": "ADULTS", "ADULT": "ADULTS", "NAN": np.nan})
        )
    for col in ["date_exact", "sample_exact", "population_exact", "partisan_flag"]:
        if col not in df.columns:
            if col == "partisan_flag":
                df[col] = df.get("pollster", "").astype(str).str.contains(r"\*\*")
            else:
                df[col] = False
        df[col] = df[col].astype(bool)
    df["date_inferred"] = ~df["date_exact"]
    df["sample_inferred"] = ~df["sample_exact"]
    df["population_inferred"] = ~df["population_exact"]
    if "pollster_key" not in df.columns:
        df["pollster_key"] = (
            df.get("pollster", pd.Series([""] * len(df)))
            .astype(str)
            .str.replace("**", "", regex=False)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
            .str.lower()
        )
    df["obs_date"] = df["end_date"].fillna(df["published_date"])
    df = df.loc[df["obs_date"].notna() & df["margin_a"].notna()].copy()
    return df.sort_values(["obs_date", "published_date", "pollster_key"]).reset_index(drop=True)


def _national_poll_population_penalty(population: Any) -> float:
    pop = str(population).upper()
    if pop == "LV":
        return 0.0
    if pop == "RV":
        return 0.25
    if pop in {"A", "ADULTS"}:
        return 0.55
    return 0.35


def _national_poll_observation_sd(row: pd.Series, config: ForecastConfig) -> float:
    effective_n = row.get("sample_size")
    if pd.isna(effective_n) or not effective_n or effective_n <= 0:
        effective_n = getattr(config, "national_poll_default_sample_size", 1000)
    sampling_sd = 100.0 / math.sqrt(float(effective_n))
    house_effect_sd = float(getattr(config, "national_poll_house_effect_sd", 1.35))
    penalties_sq = 0.0
    penalties_sq += _national_poll_population_penalty(row.get("population")) ** 2
    if bool(row.get("date_inferred", False)):
        penalties_sq += float(getattr(config, "national_poll_missing_date_penalty_sd", 0.85)) ** 2
    if bool(row.get("sample_inferred", False)):
        penalties_sq += float(getattr(config, "national_poll_missing_sample_penalty_sd", 0.45)) ** 2
    if bool(row.get("population_inferred", False)):
        penalties_sq += float(getattr(config, "national_poll_unknown_population_penalty_sd", 0.30)) ** 2
    if bool(row.get("partisan_flag", False)):
        penalties_sq += float(getattr(config, "national_poll_partisan_penalty_sd", 0.65)) ** 2
    return float(math.sqrt(sampling_sd * sampling_sd + house_effect_sd * house_effect_sd + penalties_sq))


def _run_national_poll_filter(
    polls: pd.DataFrame,
    start_date: date,
    end_date: date,
    prior_mean: float,
    prior_sd: float,
    process_sd: float,
) -> pd.DataFrame:
    if polls is None or polls.empty:
        days = pd.date_range(start=start_date, end=end_date, freq="D")
        return pd.DataFrame(
            {
                "as_of_date": days,
                "filtered_mean_margin_dem": float(prior_mean),
                "filtered_sd": float(prior_sd),
                "poll_rows_today": 0,
                "poll_rows_cumulative": 0,
                "exact_date_rows_cumulative": 0,
                "inferred_date_rows_cumulative": 0,
            }
        )

    df = polls.copy()
    df["obs_date"] = pd.to_datetime(df["obs_date"], errors="coerce")
    df = df.loc[df["obs_date"].notna()].copy()
    df = df.loc[df["obs_date"].dt.date <= end_date].copy()
    if df.empty:
        days = pd.date_range(start=start_date, end=end_date, freq="D")
        return pd.DataFrame(
            {
                "as_of_date": days,
                "filtered_mean_margin_dem": float(prior_mean),
                "filtered_sd": float(prior_sd),
                "poll_rows_today": 0,
                "poll_rows_cumulative": 0,
                "exact_date_rows_cumulative": 0,
                "inferred_date_rows_cumulative": 0,
            }
        )

    if "obs_sd" not in df.columns:
        df["obs_sd"] = 2.5
    else:
        df["obs_sd"] = pd.to_numeric(df["obs_sd"], errors="coerce").fillna(2.5)
    mean = float(prior_mean)
    var = float(prior_sd) ** 2
    cumulative_rows = 0
    cumulative_exact = 0
    cumulative_inferred = 0
    records: list[dict[str, Any]] = []

    for day_ts in pd.date_range(start=start_date, end=end_date, freq="D"):
        if records:
            var += float(process_sd) ** 2

        day_rows = df.loc[df["obs_date"].dt.normalize() == day_ts.normalize()].copy()
        day_count = 0
        if not day_rows.empty:
            day_rows = day_rows.sort_values(["published_date", "pollster_key", "margin_a"]).reset_index(drop=True)
            for _, row in day_rows.iterrows():
                obs = float(row["margin_a"])
                obs_sd = float(row["obs_sd"])
                if not np.isfinite(obs) or not np.isfinite(obs_sd) or obs_sd <= 0:
                    continue
                k = var / (var + obs_sd * obs_sd)
                mean = mean + k * (obs - mean)
                var = max((1.0 - k) * var, 1e-9)
                day_count += 1
            cumulative_rows += int(day_count)
            cumulative_exact += int(day_rows["date_exact"].sum())
            cumulative_inferred += int((~day_rows["date_exact"]).sum())

        records.append(
            {
                "as_of_date": day_ts,
                "filtered_mean_margin_dem": float(mean),
                "filtered_sd": float(math.sqrt(max(var, 1e-9))),
                "poll_rows_today": int(day_count),
                "poll_rows_cumulative": int(cumulative_rows),
                "exact_date_rows_cumulative": int(cumulative_exact),
                "inferred_date_rows_cumulative": int(cumulative_inferred),
            }
        )

    return pd.DataFrame(records)


def _recent_precision_weighted_average(polls: pd.DataFrame, as_of_date: date, half_life_days: int = 21) -> Optional[float]:
    if polls is None or polls.empty:
        return None
    df = polls.copy()
    df["obs_date"] = pd.to_datetime(df["obs_date"], errors="coerce")
    df = df.loc[df["obs_date"].notna() & (df["obs_date"].dt.date <= as_of_date)].copy()
    if df.empty:
        return None
    df["age_days"] = (pd.Timestamp(as_of_date) - df["obs_date"]).dt.days.clip(lower=0)
    weights = (0.5 ** (df["age_days"] / max(int(half_life_days), 1))) / np.square(df["obs_sd"].astype(float))
    if float(weights.sum()) <= 0:
        return None
    return float(np.average(df["margin_a"].astype(float), weights=weights))


def _compute_control_neutral_margin(
    history_master: pd.DataFrame,
    approval: DDHQMetric,
    right_track: DDHQMetric,
    config: ForecastConfig,
    start_date: date,
    seed: int,
) -> float:
    days_to_election = max((ELECTION_DATE - start_date).days, 0)
    current_sd = float(getattr(config, "national_poll_floor_sd", 1.0))
    election_day_sd = math.sqrt(
        current_sd * current_sd
        + (config.national_daily_random_walk_sd ** 2) * days_to_election
        + config.national_election_day_shock_sd ** 2
    )
    base_env = NationalEnvironment(
        mean_margin_dem=0.0,
        current_sd=current_sd,
        election_day_sd=election_day_sd,
        generic_ballot_margin_dem=0.0,
        trump_net_approval=float(approval.margin_a),
        right_track_net=float(right_track.margin_a),
        polls_included=0,
        days_to_election=days_to_election,
    )
    provisional_priors = prepare_district_priors(history_master, base_env, pd.DataFrame(), config)
    common_draws = prepare_common_summary_draws(
        provisional_priors,
        config,
        seed=seed + 911,
        simulations=config.history_simulations,
    )

    def control_prob(margin: float) -> float:
        env = NationalEnvironment(
            mean_margin_dem=float(margin),
            current_sd=current_sd,
            election_day_sd=election_day_sd,
            generic_ballot_margin_dem=float(margin),
            trump_net_approval=float(approval.margin_a),
            right_track_net=float(right_track.margin_a),
            polls_included=0,
            days_to_election=days_to_election,
        )
        return run_simulation_summary_only(
            provisional_priors,
            env,
            config,
            draws=common_draws,
        )[0].gop_control_prob

    lo, hi = -8.0, 8.0
    plo = control_prob(lo)
    phi = control_prob(hi)
    if not (plo >= 0.5 and phi <= 0.5):
        return float(getattr(config, "history_neutral_margin_dem", 0.0))

    for _ in range(18):
        mid = 0.5 * (lo + hi)
        pmid = control_prob(mid)
        if pmid > 0.5:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def metric_snapshot_from_recent_polls(
    metric: DDHQMetric,
    as_of_date: date,
    config: ForecastConfig,
) -> Optional[DDHQMetric]:
    archive = _prepare_national_poll_archive_frame(metric.recent_polls)
    if archive.empty:
        return None
    archive["obs_sd"] = archive.apply(lambda row: _national_poll_observation_sd(row, config), axis=1)
    subset = archive.loc[archive["obs_date"].dt.date <= as_of_date].copy()
    if subset.empty:
        return None

    prior_mean = float(getattr(config, "history_neutral_margin_dem", 0.0))
    prior_sd = float(getattr(config, "national_state_prior_sd", 4.5))
    filter_hist = _run_national_poll_filter(
        subset,
        start_date=subset["obs_date"].min().date(),
        end_date=as_of_date,
        prior_mean=prior_mean,
        prior_sd=prior_sd,
        process_sd=float(getattr(config, "national_daily_random_walk_sd", 0.17)),
    )
    last = filter_hist.iloc[-1]
    recent_avg = _recent_precision_weighted_average(subset, as_of_date, half_life_days=21)
    pct_a = float(metric.pct_a)
    pct_b = float(metric.pct_b)
    if recent_avg is not None:
        recent_two_party_mid = 0.5 * (pct_a + pct_b)
        pct_a = recent_two_party_mid + recent_avg / 2.0
        pct_b = recent_two_party_mid - recent_avg / 2.0

    return DDHQMetric(
        metric=metric.metric,
        label_a=metric.label_a,
        label_b=metric.label_b,
        pct_a=float(pct_a),
        pct_b=float(pct_b),
        margin_a=float(last["filtered_mean_margin_dem"]),
        polls_included=int(len(subset)),
        recent_polls=subset.drop(columns=[c for c in ["obs_date", "obs_sd"] if c in subset.columns]),
        source_url=metric.source_url,
        is_live=metric.is_live,
    )


def estimate_national_environment(
    generic: DDHQMetric,
    approval: DDHQMetric,
    right_track: DDHQMetric,
    config: ForecastConfig,
    today: Optional[date] = None,
) -> NationalEnvironment:
    today = today or date.today()
    days = max((ELECTION_DATE - today).days, 0)

    archive = _prepare_national_poll_archive_frame(generic.recent_polls)
    if archive.empty:
        current_mean = float(generic.margin_a)
        current_sd = float(getattr(config, "national_state_prior_sd", 4.5))
    else:
        archive["obs_sd"] = archive.apply(lambda row: _national_poll_observation_sd(row, config), axis=1)
        subset = archive.loc[archive["obs_date"].dt.date <= today].copy()
        if subset.empty:
            current_mean = float(generic.margin_a)
            current_sd = float(getattr(config, "national_state_prior_sd", 4.5))
        else:
            prior_mean = float(getattr(config, "history_neutral_margin_dem", 0.0))
            prior_sd = float(getattr(config, "national_state_prior_sd", 4.5))
            filter_hist = _run_national_poll_filter(
                subset,
                start_date=subset["obs_date"].min().date(),
                end_date=today,
                prior_mean=prior_mean,
                prior_sd=prior_sd,
                process_sd=float(getattr(config, "national_daily_random_walk_sd", 0.17)),
            )
            last = filter_hist.iloc[-1]
            current_mean = float(last["filtered_mean_margin_dem"])
            current_sd = float(last["filtered_sd"])

    election_day_sd = math.sqrt(
        current_sd * current_sd
        + (config.national_daily_random_walk_sd ** 2) * days
        + config.national_election_day_shock_sd ** 2
    )
    return NationalEnvironment(
        mean_margin_dem=float(current_mean),
        current_sd=float(current_sd),
        election_day_sd=float(election_day_sd),
        generic_ballot_margin_dem=float(current_mean),
        trump_net_approval=float(approval.margin_a),
        right_track_net=float(right_track.margin_a),
        polls_included=int(len(archive.loc[archive["obs_date"].dt.date <= today])) if not archive.empty else generic.polls_included,
        days_to_election=int(days),
    )


def build_history_curve(
    paths: ProjectPaths,
    master: pd.DataFrame,
    generic: DDHQMetric,
    approval: DDHQMetric,
    right_track: DDHQMetric,
    district_polls_raw: pd.DataFrame,
    config: ForecastConfig,
    as_of_date: date,
    current_summary: SimulationSummary,
    current_priors: pd.DataFrame,
    current_national: NationalEnvironment,
    seed: int = 20260401,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    archive = _prepare_national_poll_archive_frame(generic.recent_polls)
    default_start = as_of_date - timedelta(days=max(int(config.history_max_days), 1) - 1)
    archive_start = archive["obs_date"].min().date() if not archive.empty else default_start
    start_date = min(default_start, archive_start)

    history_master = master.copy()
    if "rating" in history_master.columns:
        history_master["rating"] = np.nan
    if "finance_effect" in history_master.columns:
        history_master["finance_effect"] = 0.0
    if "open_seat" in history_master.columns:
        history_master["open_seat"] = False
    if "incumbent_running" in history_master.columns:
        history_master["incumbent_running"] = False
    for col in [
        "dem_cash_on_hand",
        "rep_cash_on_hand",
        "dem_receipts",
        "rep_receipts",
        "dem_candidate",
        "rep_candidate",
    ]:
        if col in history_master.columns:
            history_master[col] = np.nan

    control_neutral_margin = _compute_control_neutral_margin(
        history_master,
        approval,
        right_track,
        config,
        start_date=start_date,
        seed=seed,
    )

    if archive.empty:
        filter_history = _run_national_poll_filter(
            pd.DataFrame(),
            start_date=start_date,
            end_date=as_of_date,
            prior_mean=control_neutral_margin,
            prior_sd=float(getattr(config, "national_state_prior_sd", 4.5)),
            process_sd=float(getattr(config, "national_daily_random_walk_sd", 0.17)),
        )
    else:
        archive["obs_sd"] = archive.apply(lambda row: _national_poll_observation_sd(row, config), axis=1)
        filter_history = _run_national_poll_filter(
            archive,
            start_date=start_date,
            end_date=as_of_date,
            prior_mean=control_neutral_margin,
            prior_sd=float(getattr(config, "national_state_prior_sd", 4.5)),
            process_sd=float(getattr(config, "national_daily_random_walk_sd", 0.17)),
        )

    base_priors_env = NationalEnvironment(
        mean_margin_dem=0.0,
        current_sd=float(getattr(config, "national_poll_floor_sd", 1.0)),
        election_day_sd=float(getattr(config, "national_poll_floor_sd", 1.0)),
        generic_ballot_margin_dem=0.0,
        trump_net_approval=float(approval.margin_a),
        right_track_net=float(right_track.margin_a),
        polls_included=0,
        days_to_election=max((ELECTION_DATE - start_date).days, 0),
    )
    provisional_priors = prepare_district_priors(history_master, base_priors_env, pd.DataFrame(), config)
    common_draws = prepare_common_summary_draws(
        provisional_priors,
        config,
        seed=seed + 1234,
        simulations=config.history_simulations,
    )

    rows: list[dict[str, Any]] = []
    for _, state_row in filter_history.iterrows():
        day = pd.Timestamp(state_row["as_of_date"]).date()
        days_to_election = max((ELECTION_DATE - day).days, 0)
        national_day = NationalEnvironment(
            mean_margin_dem=float(state_row["filtered_mean_margin_dem"]),
            current_sd=float(state_row["filtered_sd"]),
            election_day_sd=float(
                math.sqrt(
                    float(state_row["filtered_sd"]) ** 2
                    + (config.national_daily_random_walk_sd ** 2) * days_to_election
                    + config.national_election_day_shock_sd ** 2
                )
            ),
            generic_ballot_margin_dem=float(state_row["filtered_mean_margin_dem"]),
            trump_net_approval=float(approval.margin_a),
            right_track_net=float(right_track.margin_a),
            polls_included=int(state_row["poll_rows_cumulative"]),
            days_to_election=days_to_election,
        )
        district_poll_day = aggregate_district_polls(district_polls_raw, config, as_of_date=day)
        priors_day = prepare_district_priors(history_master, national_day, district_poll_day, config)
        summary_day, _ = run_simulation_summary_only(
            priors_day,
            national_day,
            config,
            draws=common_draws,
        )
        history_stage = "observed_poll_filter" if int(state_row["poll_rows_cumulative"]) > 0 else "prior_only"
        row = _history_row_from_summary(
            day,
            summary_day,
            national_day,
            priors_day,
            int(state_row["poll_rows_cumulative"]),
            is_observed_run=False,
            history_stage=history_stage,
        )
        row["filtered_generic_sd"] = float(state_row["filtered_sd"])
        row["exact_date_rows_cumulative"] = int(state_row["exact_date_rows_cumulative"])
        row["inferred_date_rows_cumulative"] = int(state_row["inferred_date_rows_cumulative"])
        row["poll_rows_today"] = int(state_row["poll_rows_today"])
        rows.append(row)

    current_row = _history_row_from_summary(
        as_of_date,
        current_summary,
        current_national,
        current_priors,
        current_national.polls_included,
        is_observed_run=True,
        history_stage="observed_run",
    )
    current_row["filtered_generic_sd"] = float(current_national.current_sd)
    current_row["exact_date_rows_cumulative"] = (
        int(archive["date_exact"].sum()) if not archive.empty else 0
    )
    current_row["inferred_date_rows_cumulative"] = (
        int((~archive["date_exact"]).sum()) if not archive.empty else 0
    )
    current_row["poll_rows_today"] = int(
        archive.loc[archive["obs_date"].dt.date == as_of_date].shape[0]
    ) if not archive.empty else 0

    curve = pd.DataFrame(rows)
    if curve.empty:
        curve = pd.DataFrame([current_row])
    else:
        curve = curve.loc[curve["as_of_date"] != current_row["as_of_date"]].copy()
        curve = pd.concat([curve, pd.DataFrame([current_row])], ignore_index=True)
        curve = curve.sort_values("as_of_date").reset_index(drop=True)

    run_rows = pd.DataFrame()
    if paths.run_history_csv.exists():
        try:
            run_rows = pd.read_csv(paths.run_history_csv)
        except Exception:
            run_rows = pd.DataFrame()

    history_meta = {
        "history_neutral_margin_dem": float(control_neutral_margin),
        "history_mode": "state_space_poll_filter",
        "history_campaign_inputs_mode": "structural_only_for_history",
        "history_archive_rows": int(len(archive)),
        "history_archive_start": archive_start.isoformat() if archive_start else None,
    }
    curve.attrs["history_meta"] = history_meta
    paths.forecast_history_csv.parent.mkdir(parents=True, exist_ok=True)
    curve.to_csv(paths.forecast_history_csv, index=False)
    curve.to_csv(paths.forecast_curve_csv, index=False)
    return curve, run_rows


def write_outputs(
    paths: ProjectPaths,
    national: NationalEnvironment,
    generic: DDHQMetric,
    approval: DDHQMetric,
    right_track: DDHQMetric,
    master: pd.DataFrame,
    priors: pd.DataFrame,
    outputs: SimulationOutputs,
    source_status: dict[str, Any],
    as_of_date: date,
    config: ForecastConfig,
    district_polls_raw: Optional[pd.DataFrame] = None,
    seed: int = 20260401,
) -> dict[str, Any]:
    paths.latest_dir.mkdir(parents=True, exist_ok=True)
    outputs.district_results.to_csv(paths.latest_dir / "district_forecast.csv", index=False)
    outputs.seat_distribution.to_csv(paths.latest_dir / "seat_distribution.csv", index=False)
    priors.to_csv(paths.latest_dir / "district_priors.csv", index=False)
    master.to_csv(paths.latest_dir / "district_master.csv", index=False)

    archive = _prepare_national_poll_archive_frame(generic.recent_polls)
    if not archive.empty:
        archive.to_csv(paths.latest_dir / "generic_ballot_polls.csv", index=False)
    poll_start = archive["obs_date"].min().date().isoformat() if not archive.empty else None
    poll_end = archive["obs_date"].max().date().isoformat() if not archive.empty else None

    run_snapshot = {
        "as_of_date": as_of_date.isoformat(),
        "gop_control_prob": outputs.summary.gop_control_prob,
        "expected_gop_seats": outputs.summary.expected_gop_seats,
        "median_gop_seats": outputs.summary.median_gop_seats,
        "gop_seat_q05": outputs.summary.gop_seat_q05,
        "gop_seat_q25": outputs.summary.gop_seat_q25,
        "gop_seat_q75": outputs.summary.gop_seat_q75,
        "gop_seat_q95": outputs.summary.gop_seat_q95,
        "dem_control_prob": outputs.summary.dem_control_prob,
        "generic_ballot_margin_dem": national.generic_ballot_margin_dem,
        "trump_net_approval": national.trump_net_approval,
        "right_track_net": national.right_track_net,
        "districts_polled": int((priors["poll_count"] > 0).sum()),
        "districts_open": int(priors["open_seat"].sum()),
        "national_mean_margin_dem": national.mean_margin_dem,
        "national_election_day_sd": national.election_day_sd,
        "visible_national_poll_rows": int(len(archive)) if not archive.empty else 0,
    }
    run_history = append_run_history(paths, run_snapshot)

    history_curve, _ = build_history_curve(
        paths,
        master,
        generic,
        approval,
        right_track,
        district_polls_raw if district_polls_raw is not None else pd.DataFrame(),
        config,
        as_of_date,
        outputs.summary,
        priors,
        national,
        seed=seed,
    )
    history_meta = history_curve.attrs.get("history_meta", {}) if hasattr(history_curve, "attrs") else {}

    exact_date_rows = int(archive["date_exact"].sum()) if not archive.empty else 0
    inferred_date_rows = int((~archive["date_exact"]).sum()) if not archive.empty else 0
    exact_sample_rows = int(archive["sample_exact"].sum()) if not archive.empty else 0
    inferred_sample_rows = int((~archive["sample_exact"]).sum()) if not archive.empty else 0
    recent_weighted_margin = None
    if not archive.empty:
        archive = archive.copy()
        archive["obs_sd"] = archive.apply(lambda row: _national_poll_observation_sd(row, config), axis=1)
        recent_weighted_margin = _recent_precision_weighted_average(archive, as_of_date, half_life_days=21)

    summary_dict = {
        "as_of_date": as_of_date.isoformat(),
        "gop_control_prob": outputs.summary.gop_control_prob,
        "expected_gop_seats": outputs.summary.expected_gop_seats,
        "median_gop_seats": outputs.summary.median_gop_seats,
        "gop_seat_q05": outputs.summary.gop_seat_q05,
        "gop_seat_q25": outputs.summary.gop_seat_q25,
        "gop_seat_q75": outputs.summary.gop_seat_q75,
        "gop_seat_q95": outputs.summary.gop_seat_q95,
        "dem_control_prob": outputs.summary.dem_control_prob,
        "national_mean_margin_dem": national.mean_margin_dem,
        "national_current_sd": national.current_sd,
        "national_election_day_sd": national.election_day_sd,
        "generic_ballot_margin_dem": national.generic_ballot_margin_dem,
        "rcp_current_average_margin_dem": float(generic.pct_a - generic.pct_b),
        "recent_precision_weighted_margin_dem": recent_weighted_margin,
        "trump_net_approval": national.trump_net_approval,
        "right_track_net": national.right_track_net,
        "source_status": source_status,
        "simulations": config.simulations,
        "history_simulations": config.history_simulations,
        "visible_national_poll_rows": int(len(archive)) if not archive.empty else 0,
        "visible_national_poll_start": poll_start,
        "visible_national_poll_end": poll_end,
        "generic_poll_archive_rows": int(len(archive)) if not archive.empty else 0,
        "generic_poll_exact_date_rows": exact_date_rows,
        "generic_poll_inferred_date_rows": inferred_date_rows,
        "generic_poll_exact_sample_rows": exact_sample_rows,
        "generic_poll_inferred_sample_rows": inferred_sample_rows,
        "history_points": int(len(history_curve)),
        "history_start_date": history_curve["as_of_date"].min() if not history_curve.empty else as_of_date.isoformat(),
        "history_end_date": history_curve["as_of_date"].max() if not history_curve.empty else as_of_date.isoformat(),
        "history_backcast_days": int((history_curve.get("history_stage") == "prior_only").sum()) if not history_curve.empty and "history_stage" in history_curve.columns else 0,
        "history_observed_days": int((history_curve.get("history_stage") == "observed_poll_filter").sum()) if not history_curve.empty and "history_stage" in history_curve.columns else 0,
        "history_initial_gop_control_prob": float(history_curve.iloc[0]["gop_control_prob"]) if not history_curve.empty else outputs.summary.gop_control_prob,
        "history_initial_expected_gop_seats": float(history_curve.iloc[0]["expected_gop_seats"]) if not history_curve.empty else outputs.summary.expected_gop_seats,
        "history_neutral_margin_dem": history_meta.get("history_neutral_margin_dem"),
        "history_mode": history_meta.get("history_mode"),
        "history_campaign_inputs_mode": history_meta.get("history_campaign_inputs_mode"),
        "history_archive_rows": history_meta.get("history_archive_rows"),
        "history_archive_start": history_meta.get("history_archive_start"),
        "run_history_points": int(len(run_history)),
    }
    with (paths.latest_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(ensure_json_serializable(summary_dict), f, indent=2)

    audit = {
        "as_of_date": as_of_date.isoformat(),
        "simulations": config.simulations,
        "history_simulations": config.history_simulations,
        "source_status": source_status,
        "generic_poll_archive_rows": int(len(archive)) if not archive.empty else 0,
        "generic_poll_archive_start": poll_start,
        "generic_poll_archive_end": poll_end,
        "generic_poll_exact_date_rows": exact_date_rows,
        "generic_poll_inferred_date_rows": inferred_date_rows,
        "generic_poll_exact_sample_rows": exact_sample_rows,
        "generic_poll_inferred_sample_rows": inferred_sample_rows,
        "rcp_current_average_margin_dem": float(generic.pct_a - generic.pct_b),
        "recent_precision_weighted_margin_dem": recent_weighted_margin,
        "districts_total": int(len(priors)),
        "districts_open": int(priors["open_seat"].sum()),
        "districts_with_polls": int((priors["poll_count"] > 0).sum()),
        "districts_with_ratings": int(priors["rating"].notna().sum()) if "rating" in priors.columns else 0,
        "history_points": int(len(history_curve)),
        "history_backcast_days": int((history_curve.get("history_stage") == "prior_only").sum()) if not history_curve.empty and "history_stage" in history_curve.columns else 0,
        "history_observed_days": int((history_curve.get("history_stage") == "observed_poll_filter").sum()) if not history_curve.empty and "history_stage" in history_curve.columns else 0,
        "history_neutral_margin_dem": history_meta.get("history_neutral_margin_dem"),
        "history_mode": history_meta.get("history_mode"),
        "history_campaign_inputs_mode": history_meta.get("history_campaign_inputs_mode"),
        "run_history_points": int(len(run_history)),
        "seed": int(seed),
    }
    with paths.run_audit_json.open("w", encoding="utf-8") as f:
        json.dump(ensure_json_serializable(audit), f, indent=2)

    return {
        "summary": summary_dict,
        "history": history_curve,
        "run_history": run_history,
        "audit": audit,
    }



def _history_row_from_summary(
    as_of_date: date,
    summary: SimulationSummary,
    national: NationalEnvironment,
    priors: pd.DataFrame,
    polls_used: Optional[int],
    is_observed_run: bool,
    history_stage: Optional[str] = None,
) -> dict[str, Any]:
    return {
        "as_of_date": as_of_date.isoformat(),
        "gop_control_prob": summary.gop_control_prob,
        "expected_gop_seats": summary.expected_gop_seats,
        "median_gop_seats": summary.median_gop_seats,
        "gop_seat_q05": summary.gop_seat_q05,
        "gop_seat_q25": summary.gop_seat_q25,
        "gop_seat_q75": summary.gop_seat_q75,
        "gop_seat_q95": summary.gop_seat_q95,
        "dem_control_prob": summary.dem_control_prob,
        "generic_ballot_margin_dem": national.generic_ballot_margin_dem,
        "national_mean_margin_dem": national.mean_margin_dem,
        "national_election_day_sd": national.election_day_sd,
        "days_to_election": national.days_to_election,
        "districts_polled": int((priors["poll_count"] > 0).sum()) if (not priors.empty and "poll_count" in priors.columns) else 0,
        "districts_open": int(priors["open_seat"].sum()) if (not priors.empty and "open_seat" in priors.columns) else 0,
        "visible_national_poll_rows": int(polls_used or 0),
        "is_observed_run": bool(is_observed_run),
        "history_stage": history_stage,
    }


# ---------------------------------------------------------------------------
# v5 audit + approval-aware overrides
# ---------------------------------------------------------------------------

def _coerce_bool_series(values: pd.Series, default: bool = False) -> pd.Series:
    if values is None:
        return pd.Series(dtype=bool)
    if values.dtype == bool:
        return values.fillna(default)
    lowered = values.astype(str).str.strip().str.lower()
    mapped = lowered.map({"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False})
    mapped = mapped.where(~lowered.isin(["", "nan", "none"]), other=np.nan)
    return mapped.fillna(default).astype(bool)


def _prepare_national_poll_archive_frame(recent_polls: pd.DataFrame, default_year: int = 2026) -> pd.DataFrame:
    if recent_polls is None or recent_polls.empty:
        return pd.DataFrame()

    df = recent_polls.copy()
    for col in [
        "sample_size",
        "margin_a",
        "pct_a",
        "pct_b",
        "dem_pct",
        "rep_pct",
        "rcp_dem_pct",
        "rcp_rep_pct",
        "official_dem_pct",
        "official_rep_pct",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "margin_a" not in df.columns:
        if {"pct_a", "pct_b"}.issubset(df.columns):
            df["margin_a"] = df["pct_a"] - df["pct_b"]
        elif {"dem_pct", "rep_pct"}.issubset(df.columns):
            df["margin_a"] = df["dem_pct"] - df["rep_pct"]

    if "end_date" not in df.columns and "field_dates" in df.columns:
        df["end_date"] = df["field_dates"].map(lambda x: parse_month_day_range(str(x), default_year))
    if "published_date" not in df.columns:
        df["published_date"] = df.get("end_date")
    if "start_date" not in df.columns:
        df["start_date"] = df.get("end_date")

    df["end_date"] = pd.to_datetime(df.get("end_date"), errors="coerce")
    df["published_date"] = pd.to_datetime(df.get("published_date"), errors="coerce")
    df["start_date"] = pd.to_datetime(df.get("start_date"), errors="coerce")

    if "population" in df.columns:
        df["population"] = (
            df["population"].astype(str).str.upper().replace({"A": "ADULTS", "ADULT": "ADULTS", "NAN": np.nan})
        )
    else:
        df["population"] = np.nan

    inferred_date_default = df["end_date"].notna() & df.get("field_dates", pd.Series([None] * len(df))).astype(str).str.strip().ne("")
    if "date_exact" in df.columns:
        df["date_exact"] = _coerce_bool_series(df["date_exact"], default=False)
    else:
        df["date_exact"] = inferred_date_default.astype(bool)

    if "sample_exact" in df.columns:
        df["sample_exact"] = _coerce_bool_series(df["sample_exact"], default=False)
    else:
        df["sample_exact"] = df["sample_size"].notna().astype(bool) if "sample_size" in df.columns else False

    if "population_exact" in df.columns:
        df["population_exact"] = _coerce_bool_series(df["population_exact"], default=False)
    else:
        df["population_exact"] = df["population"].notna().astype(bool)

    if "partisan_flag" in df.columns:
        df["partisan_flag"] = _coerce_bool_series(df["partisan_flag"], default=False)
    else:
        pollster_series = df.get("pollster", pd.Series([""] * len(df))).astype(str)
        notes_series = df.get("notes", pd.Series([""] * len(df))).astype(str)
        df["partisan_flag"] = (
            pollster_series.str.contains(r"\*\*", regex=True)
            | notes_series.str.contains(r"internal|partisan", case=False, regex=True)
        )

    df["date_inferred"] = ~df["date_exact"]
    df["sample_inferred"] = ~df["sample_exact"]
    df["population_inferred"] = ~df["population_exact"]

    if "pollster_key" not in df.columns:
        df["pollster_key"] = (
            df.get("pollster", pd.Series([""] * len(df)))
            .astype(str)
            .str.replace("**", "", regex=False)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
            .str.lower()
        )

    df["obs_date"] = df["end_date"].fillna(df["published_date"])
    df = df.loc[df["obs_date"].notna() & pd.to_numeric(df["margin_a"], errors="coerce").notna()].copy()
    return df.sort_values(["obs_date", "published_date", "pollster_key"]).reset_index(drop=True)


def metric_snapshot_from_recent_polls(
    metric: DDHQMetric,
    as_of_date: date,
    config: ForecastConfig,
) -> Optional[DDHQMetric]:
    archive = _prepare_national_poll_archive_frame(metric.recent_polls)
    if archive.empty:
        return None

    archive["obs_sd"] = archive.apply(lambda row: _national_poll_observation_sd(row, config), axis=1)
    subset = archive.loc[archive["obs_date"].dt.date <= as_of_date].copy()
    if subset.empty:
        return None

    prior_mean = float(getattr(config, "national_state_prior_sd", 4.5))
    prior_mean = 0.0 if not np.isfinite(prior_mean) else 0.0
    filter_hist = _run_national_poll_filter(
        subset,
        start_date=subset["obs_date"].min().date(),
        end_date=as_of_date,
        prior_mean=prior_mean,
        prior_sd=float(getattr(config, "national_state_prior_sd", 4.5)),
        process_sd=float(getattr(config, "national_daily_random_walk_sd", 0.17)),
    )
    last = filter_hist.iloc[-1]

    pct_a = float(metric.pct_a)
    pct_b = float(metric.pct_b)
    if "pct_a" in subset.columns and subset["pct_a"].notna().any() and "pct_b" in subset.columns and subset["pct_b"].notna().any():
        recent_mid = 0.5 * (
            float(pd.to_numeric(subset["pct_a"], errors="coerce").dropna().iloc[-1])
            + float(pd.to_numeric(subset["pct_b"], errors="coerce").dropna().iloc[-1])
        )
        pct_a = recent_mid + float(last["filtered_mean_margin_dem"]) / 2.0
        pct_b = recent_mid - float(last["filtered_mean_margin_dem"]) / 2.0

    cleaned_recent = subset.drop(columns=[c for c in ["obs_date", "obs_sd"] if c in subset.columns])
    return DDHQMetric(
        metric=metric.metric,
        label_a=metric.label_a,
        label_b=metric.label_b,
        pct_a=float(pct_a),
        pct_b=float(pct_b),
        margin_a=float(last["filtered_mean_margin_dem"]),
        polls_included=int(len(subset)),
        recent_polls=cleaned_recent,
        source_url=metric.source_url,
        is_live=metric.is_live,
    )


def _blend_generic_with_approval(
    generic_mean: float,
    generic_sd: float,
    approval_metric: Optional[DDHQMetric],
    config: ForecastConfig,
) -> tuple[float, float, Optional[float], Optional[float], float]:
    if approval_metric is None or approval_metric.margin_a is None or not np.isfinite(float(approval_metric.margin_a)):
        return float(generic_mean), float(generic_sd), None, None, 0.0

    approval_implied = float(config.approval_to_generic_slope) * float(approval_metric.margin_a)
    approval_prior_sd = float(config.approval_prior_sd)
    blended_mean, blended_sd = combine_normal_estimates(
        float(generic_mean),
        float(generic_sd),
        approval_implied,
        approval_prior_sd,
    )
    disagreement = abs(float(generic_mean) - approval_implied)
    extra_sd = max(disagreement - float(config.approval_disagreement_tolerance), 0.0) * float(config.approval_disagreement_inflation)
    if extra_sd > 0:
        blended_sd = math.sqrt(blended_sd * blended_sd + extra_sd * extra_sd)
    return float(blended_mean), float(blended_sd), approval_implied, approval_prior_sd, float(disagreement)


def _build_metric_curve_from_recent_polls(
    metric: DDHQMetric,
    config: ForecastConfig,
    as_of_date: date,
) -> pd.DataFrame:
    archive = _prepare_national_poll_archive_frame(metric.recent_polls)
    if archive.empty or not {"pct_a", "pct_b"}.issubset(archive.columns):
        return pd.DataFrame()

    archive["obs_sd"] = archive.apply(lambda row: _national_poll_observation_sd(row, config), axis=1)
    start_date = archive["obs_date"].min().date()

    def _scalar_filter(value_col: str, prior_mean: float, prior_sd: float) -> pd.DataFrame:
        subset = archive.copy()
        subset["margin_a"] = pd.to_numeric(subset[value_col], errors="coerce")
        subset = subset.loc[subset["margin_a"].notna()].copy()
        return _run_national_poll_filter(
            subset,
            start_date=start_date,
            end_date=as_of_date,
            prior_mean=float(prior_mean),
            prior_sd=float(prior_sd),
            process_sd=float(getattr(config, "national_daily_random_walk_sd", 0.17)),
        )

    first_a = pd.to_numeric(archive["pct_a"], errors="coerce").dropna()
    first_b = pd.to_numeric(archive["pct_b"], errors="coerce").dropna()
    first_margin = pd.to_numeric(archive["margin_a"], errors="coerce").dropna()
    if first_a.empty or first_b.empty or first_margin.empty:
        return pd.DataFrame()

    approve_hist = _scalar_filter("pct_a", prior_mean=float(first_a.iloc[0]), prior_sd=4.0)
    disapprove_hist = _scalar_filter("pct_b", prior_mean=float(first_b.iloc[0]), prior_sd=4.0)
    net_hist = _scalar_filter("margin_a", prior_mean=float(first_margin.iloc[0]), prior_sd=6.0)

    curve = approve_hist[["as_of_date", "filtered_mean_margin_dem", "filtered_sd", "poll_rows_cumulative"]].rename(
        columns={"filtered_mean_margin_dem": "approve_avg", "filtered_sd": "approve_sd"}
    )
    curve = curve.merge(
        disapprove_hist[["as_of_date", "filtered_mean_margin_dem", "filtered_sd"]].rename(
            columns={"filtered_mean_margin_dem": "disapprove_avg", "filtered_sd": "disapprove_sd"}
        ),
        on="as_of_date",
        how="outer",
    )
    curve = curve.merge(
        net_hist[["as_of_date", "filtered_mean_margin_dem", "filtered_sd"]].rename(
            columns={"filtered_mean_margin_dem": "net_approval", "filtered_sd": "net_approval_sd"}
        ),
        on="as_of_date",
        how="outer",
    )
    curve = curve.sort_values("as_of_date").reset_index(drop=True)
    if not curve.empty:
        # Put the current aggregate on the endpoint for display; the uncertainty still comes from the recent-row filter.
        curve.loc[curve.index[-1], "approve_avg"] = float(metric.pct_a)
        curve.loc[curve.index[-1], "disapprove_avg"] = float(metric.pct_b)
        curve.loc[curve.index[-1], "net_approval"] = float(metric.margin_a)
        z90 = 1.645
        curve["approve_low_90"] = curve["approve_avg"] - z90 * curve["approve_sd"]
        curve["approve_high_90"] = curve["approve_avg"] + z90 * curve["approve_sd"]
        curve["disapprove_low_90"] = curve["disapprove_avg"] - z90 * curve["disapprove_sd"]
        curve["disapprove_high_90"] = curve["disapprove_avg"] + z90 * curve["disapprove_sd"]
    return curve


def estimate_national_environment(
    generic: DDHQMetric,
    approval: DDHQMetric,
    right_track: DDHQMetric,
    config: ForecastConfig,
    today: Optional[date] = None,
) -> NationalEnvironment:
    today = today or date.today()
    days = max((ELECTION_DATE - today).days, 0)

    archive = _prepare_national_poll_archive_frame(generic.recent_polls)
    if archive.empty:
        raw_generic_mean = float(generic.margin_a)
        raw_generic_sd = float(getattr(config, "national_state_prior_sd", 4.5))
    else:
        archive["obs_sd"] = archive.apply(lambda row: _national_poll_observation_sd(row, config), axis=1)
        subset = archive.loc[archive["obs_date"].dt.date <= today].copy()
        if subset.empty:
            raw_generic_mean = float(generic.margin_a)
            raw_generic_sd = float(getattr(config, "national_state_prior_sd", 4.5))
        else:
            filter_hist = _run_national_poll_filter(
                subset,
                start_date=subset["obs_date"].min().date(),
                end_date=today,
                prior_mean=0.0,
                prior_sd=float(getattr(config, "national_state_prior_sd", 4.5)),
                process_sd=float(getattr(config, "national_daily_random_walk_sd", 0.17)),
            )
            last = filter_hist.iloc[-1]
            raw_generic_mean = float(last["filtered_mean_margin_dem"])
            raw_generic_sd = max(float(last["filtered_sd"]), float(getattr(config, "national_poll_floor_sd", 1.0)))

    blended_mean, blended_current_sd, _, _, _ = _blend_generic_with_approval(
        raw_generic_mean,
        raw_generic_sd,
        approval,
        config,
    )
    election_day_sd = math.sqrt(
        blended_current_sd * blended_current_sd
        + (config.national_daily_random_walk_sd ** 2) * days
        + config.national_election_day_shock_sd ** 2
    )
    return NationalEnvironment(
        mean_margin_dem=float(blended_mean),
        current_sd=float(blended_current_sd),
        election_day_sd=float(election_day_sd),
        generic_ballot_margin_dem=float(raw_generic_mean),
        trump_net_approval=float(approval.margin_a),
        right_track_net=float(right_track.margin_a),
        polls_included=int(len(archive.loc[archive["obs_date"].dt.date <= today])) if not archive.empty else generic.polls_included,
        days_to_election=int(days),
    )


def _history_row_from_summary(
    as_of_date: date,
    summary: SimulationSummary,
    national: NationalEnvironment,
    priors: pd.DataFrame,
    polls_used: Optional[int],
    is_observed_run: bool,
    history_stage: Optional[str] = None,
    approval_metric: Optional[DDHQMetric] = None,
    approval_implied_generic_margin_dem: Optional[float] = None,
) -> dict[str, Any]:
    return {
        "as_of_date": as_of_date.isoformat(),
        "gop_control_prob": summary.gop_control_prob,
        "expected_gop_seats": summary.expected_gop_seats,
        "median_gop_seats": summary.median_gop_seats,
        "gop_seat_q05": summary.gop_seat_q05,
        "gop_seat_q25": summary.gop_seat_q25,
        "gop_seat_q75": summary.gop_seat_q75,
        "gop_seat_q95": summary.gop_seat_q95,
        "dem_control_prob": summary.dem_control_prob,
        "generic_ballot_margin_dem": national.generic_ballot_margin_dem,
        "national_mean_margin_dem": national.mean_margin_dem,
        "approval_adjustment_to_national_margin": float(national.mean_margin_dem - national.generic_ballot_margin_dem),
        "approval_implied_generic_margin_dem": approval_implied_generic_margin_dem,
        "trump_net_approval": float(approval_metric.margin_a) if approval_metric is not None else np.nan,
        "trump_approve_pct": float(approval_metric.pct_a) if approval_metric is not None else np.nan,
        "trump_disapprove_pct": float(approval_metric.pct_b) if approval_metric is not None else np.nan,
        "national_election_day_sd": national.election_day_sd,
        "days_to_election": national.days_to_election,
        "districts_polled": int((priors["poll_count"] > 0).sum()) if (not priors.empty and "poll_count" in priors.columns) else 0,
        "districts_open": int(priors["open_seat"].sum()) if (not priors.empty and "open_seat" in priors.columns) else 0,
        "visible_national_poll_rows": int(polls_used or 0),
        "is_observed_run": bool(is_observed_run),
        "history_stage": history_stage,
    }


def build_history_curve(
    paths: ProjectPaths,
    master: pd.DataFrame,
    generic: DDHQMetric,
    approval: DDHQMetric,
    right_track: DDHQMetric,
    district_polls_raw: pd.DataFrame,
    config: ForecastConfig,
    as_of_date: date,
    current_summary: SimulationSummary,
    current_priors: pd.DataFrame,
    current_national: NationalEnvironment,
    seed: int = 20260401,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    archive = _prepare_national_poll_archive_frame(generic.recent_polls)
    default_start = as_of_date - timedelta(days=max(int(config.history_max_days), 1) - 1)
    archive_start = archive["obs_date"].min().date() if not archive.empty else default_start
    start_date = min(default_start, archive_start)

    # Hold current campaign inputs fixed through the reconstructed daily curve so the endpoint
    # stays definitionally comparable to the current forecast.
    history_master = master.copy()

    control_neutral_margin = _compute_control_neutral_margin(
        history_master,
        approval,
        right_track,
        config,
        start_date=start_date,
        seed=seed,
    )

    if archive.empty:
        filter_history = _run_national_poll_filter(
            pd.DataFrame(),
            start_date=start_date,
            end_date=as_of_date,
            prior_mean=control_neutral_margin,
            prior_sd=float(getattr(config, "national_state_prior_sd", 4.5)),
            process_sd=float(getattr(config, "national_daily_random_walk_sd", 0.17)),
        )
    else:
        archive = archive.copy()
        archive["obs_sd"] = archive.apply(lambda row: _national_poll_observation_sd(row, config), axis=1)
        filter_history = _run_national_poll_filter(
            archive,
            start_date=start_date,
            end_date=as_of_date,
            prior_mean=control_neutral_margin,
            prior_sd=float(getattr(config, "national_state_prior_sd", 4.5)),
            process_sd=float(getattr(config, "national_daily_random_walk_sd", 0.17)),
        )

    base_priors_env = NationalEnvironment(
        mean_margin_dem=float(current_national.mean_margin_dem),
        current_sd=float(current_national.current_sd),
        election_day_sd=float(current_national.current_sd),
        generic_ballot_margin_dem=float(current_national.generic_ballot_margin_dem),
        trump_net_approval=float(approval.margin_a),
        right_track_net=float(right_track.margin_a),
        polls_included=0,
        days_to_election=max((ELECTION_DATE - start_date).days, 0),
    )
    provisional_priors = prepare_district_priors(history_master, base_priors_env, pd.DataFrame(), config)
    common_draws = prepare_common_summary_draws(
        provisional_priors,
        config,
        seed=seed + 1234,
        simulations=config.history_simulations,
    )

    rows: list[dict[str, Any]] = []
    comparable_today_summary: Optional[SimulationSummary] = None
    for _, state_row in filter_history.iterrows():
        day = pd.Timestamp(state_row["as_of_date"]).date()
        days_to_election = max((ELECTION_DATE - day).days, 0)

        raw_generic_mean = float(state_row["filtered_mean_margin_dem"])
        raw_generic_sd = max(float(state_row["filtered_sd"]), float(getattr(config, "national_poll_floor_sd", 1.0)))
        blended_mean, blended_current_sd, approval_implied, _, _ = _blend_generic_with_approval(
            raw_generic_mean,
            raw_generic_sd,
            approval,
            config,
        )

        national_day = NationalEnvironment(
            mean_margin_dem=float(blended_mean),
            current_sd=float(blended_current_sd),
            election_day_sd=float(
                math.sqrt(
                    float(blended_current_sd) ** 2
                    + (config.national_daily_random_walk_sd ** 2) * days_to_election
                    + config.national_election_day_shock_sd ** 2
                )
            ),
            generic_ballot_margin_dem=float(raw_generic_mean),
            trump_net_approval=float(approval.margin_a),
            right_track_net=float(right_track.margin_a),
            polls_included=int(state_row["poll_rows_cumulative"]),
            days_to_election=days_to_election,
        )
        district_poll_day = aggregate_district_polls(district_polls_raw, config, as_of_date=day)
        priors_day = prepare_district_priors(history_master, national_day, district_poll_day, config)
        summary_day, _ = run_simulation_summary_only(
            priors_day,
            national_day,
            config,
            draws=common_draws,
        )
        history_stage = "observed_poll_filter" if int(state_row["poll_rows_cumulative"]) > 0 else "prior_only"
        row = _history_row_from_summary(
            day,
            summary_day,
            national_day,
            priors_day,
            int(state_row["poll_rows_cumulative"]),
            is_observed_run=(day == as_of_date),
            history_stage=history_stage,
            approval_metric=approval,
            approval_implied_generic_margin_dem=approval_implied,
        )
        row["filtered_generic_sd"] = float(state_row["filtered_sd"])
        row["filtered_national_sd"] = float(blended_current_sd)
        row["exact_date_rows_cumulative"] = int(state_row["exact_date_rows_cumulative"])
        row["inferred_date_rows_cumulative"] = int(state_row["inferred_date_rows_cumulative"])
        row["poll_rows_today"] = int(state_row["poll_rows_today"])
        rows.append(row)
        if day == as_of_date:
            comparable_today_summary = summary_day

    curve = pd.DataFrame(rows)
    if curve.empty:
        curve = pd.DataFrame([
            _history_row_from_summary(
                as_of_date,
                current_summary,
                current_national,
                current_priors,
                current_national.polls_included,
                is_observed_run=True,
                history_stage="observed_run",
                approval_metric=approval,
                approval_implied_generic_margin_dem=float(config.approval_to_generic_slope) * float(approval.margin_a),
            )
        ])
        curve["filtered_generic_sd"] = float(current_national.current_sd)
        curve["filtered_national_sd"] = float(current_national.current_sd)
        curve["exact_date_rows_cumulative"] = int(archive["date_exact"].sum()) if not archive.empty else 0
        curve["inferred_date_rows_cumulative"] = int((~archive["date_exact"]).sum()) if not archive.empty else 0
        curve["poll_rows_today"] = int(archive.loc[archive["obs_date"].dt.date == as_of_date].shape[0]) if not archive.empty else 0

    run_rows = pd.DataFrame()
    if paths.run_history_csv.exists():
        try:
            run_rows = pd.read_csv(paths.run_history_csv)
        except Exception:
            run_rows = pd.DataFrame()

    prob_gap = None
    seat_gap = None
    if comparable_today_summary is not None:
        prob_gap = float(current_summary.gop_control_prob - comparable_today_summary.gop_control_prob)
        seat_gap = float(current_summary.expected_gop_seats - comparable_today_summary.expected_gop_seats)

    history_meta = {
        "history_neutral_margin_dem": float(control_neutral_margin),
        "history_mode": "state_space_poll_filter",
        "history_campaign_inputs_mode": "current_inputs_frozen_for_comparability",
        "history_approval_mode": "current_approval_crosscheck_frozen_for_comparability",
        "history_archive_rows": int(len(archive)),
        "history_archive_start": archive_start.isoformat() if archive_start else None,
        "history_current_endpoint_gap_prob": prob_gap,
        "history_current_endpoint_gap_seats": seat_gap,
    }
    curve.attrs["history_meta"] = history_meta
    paths.forecast_history_csv.parent.mkdir(parents=True, exist_ok=True)
    curve.to_csv(paths.forecast_history_csv, index=False)
    curve.to_csv(paths.forecast_curve_csv, index=False)
    return curve, run_rows


def write_outputs(
    paths: ProjectPaths,
    national: NationalEnvironment,
    generic: DDHQMetric,
    approval: DDHQMetric,
    right_track: DDHQMetric,
    master: pd.DataFrame,
    priors: pd.DataFrame,
    outputs: SimulationOutputs,
    source_status: dict[str, Any],
    as_of_date: date,
    config: ForecastConfig,
    district_polls_raw: Optional[pd.DataFrame] = None,
    seed: int = 20260401,
) -> dict[str, Any]:
    paths.latest_dir.mkdir(parents=True, exist_ok=True)
    outputs.district_results.to_csv(paths.latest_dir / "district_forecast.csv", index=False)
    outputs.seat_distribution.to_csv(paths.latest_dir / "seat_distribution.csv", index=False)
    priors.to_csv(paths.latest_dir / "district_priors.csv", index=False)
    master.to_csv(paths.latest_dir / "district_master.csv", index=False)

    generic_archive = _prepare_national_poll_archive_frame(generic.recent_polls)
    if not generic_archive.empty:
        generic_archive.to_csv(paths.latest_dir / "generic_ballot_polls.csv", index=False)
    poll_start = generic_archive["obs_date"].min().date().isoformat() if not generic_archive.empty else None
    poll_end = generic_archive["obs_date"].max().date().isoformat() if not generic_archive.empty else None

    approval_archive = _prepare_national_poll_archive_frame(approval.recent_polls)
    if not approval_archive.empty:
        approval_archive.to_csv(paths.latest_dir / "trump_approval_polls.csv", index=False)
    approval_curve = _build_metric_curve_from_recent_polls(approval, config, as_of_date)
    if not approval_curve.empty:
        approval_curve.to_csv(paths.latest_dir / "trump_approval_curve.csv", index=False)
    approval_start = approval_archive["obs_date"].min().date().isoformat() if not approval_archive.empty else None
    approval_end = approval_archive["obs_date"].max().date().isoformat() if not approval_archive.empty else None

    approval_implied_generic_margin_dem = float(config.approval_to_generic_slope) * float(approval.margin_a)
    approval_adjustment = float(national.mean_margin_dem - national.generic_ballot_margin_dem)
    approval_generic_gap = float(approval_implied_generic_margin_dem - national.generic_ballot_margin_dem)

    run_snapshot = {
        "as_of_date": as_of_date.isoformat(),
        "gop_control_prob": outputs.summary.gop_control_prob,
        "expected_gop_seats": outputs.summary.expected_gop_seats,
        "median_gop_seats": outputs.summary.median_gop_seats,
        "gop_seat_q05": outputs.summary.gop_seat_q05,
        "gop_seat_q25": outputs.summary.gop_seat_q25,
        "gop_seat_q75": outputs.summary.gop_seat_q75,
        "gop_seat_q95": outputs.summary.gop_seat_q95,
        "dem_control_prob": outputs.summary.dem_control_prob,
        "generic_ballot_margin_dem": national.generic_ballot_margin_dem,
        "national_mean_margin_dem": national.mean_margin_dem,
        "approval_adjustment_to_national_margin": approval_adjustment,
        "trump_net_approval": national.trump_net_approval,
        "trump_approve_pct": float(approval.pct_a),
        "trump_disapprove_pct": float(approval.pct_b),
        "right_track_net": national.right_track_net,
        "districts_polled": int((priors["poll_count"] > 0).sum()),
        "districts_open": int(priors["open_seat"].sum()),
        "national_election_day_sd": national.election_day_sd,
        "visible_national_poll_rows": int(len(generic_archive)) if not generic_archive.empty else 0,
    }
    run_history = append_run_history(paths, run_snapshot)

    history_curve, _ = build_history_curve(
        paths,
        master,
        generic,
        approval,
        right_track,
        district_polls_raw if district_polls_raw is not None else pd.DataFrame(),
        config,
        as_of_date,
        outputs.summary,
        priors,
        national,
        seed=seed,
    )
    history_meta = history_curve.attrs.get("history_meta", {}) if hasattr(history_curve, "attrs") else {}

    exact_date_rows = int(generic_archive["date_exact"].sum()) if not generic_archive.empty else 0
    inferred_date_rows = int((~generic_archive["date_exact"]).sum()) if not generic_archive.empty else 0
    exact_sample_rows = int(generic_archive["sample_exact"].sum()) if not generic_archive.empty else 0
    inferred_sample_rows = int((~generic_archive["sample_exact"]).sum()) if not generic_archive.empty else 0
    recent_weighted_margin = None
    if not generic_archive.empty:
        generic_archive = generic_archive.copy()
        generic_archive["obs_sd"] = generic_archive.apply(lambda row: _national_poll_observation_sd(row, config), axis=1)
        recent_weighted_margin = _recent_precision_weighted_average(generic_archive, as_of_date, half_life_days=21)

    summary_dict = {
        "as_of_date": as_of_date.isoformat(),
        "gop_control_prob": outputs.summary.gop_control_prob,
        "expected_gop_seats": outputs.summary.expected_gop_seats,
        "median_gop_seats": outputs.summary.median_gop_seats,
        "gop_seat_q05": outputs.summary.gop_seat_q05,
        "gop_seat_q25": outputs.summary.gop_seat_q25,
        "gop_seat_q75": outputs.summary.gop_seat_q75,
        "gop_seat_q95": outputs.summary.gop_seat_q95,
        "dem_control_prob": outputs.summary.dem_control_prob,
        "national_mean_margin_dem": national.mean_margin_dem,
        "national_current_sd": national.current_sd,
        "national_election_day_sd": national.election_day_sd,
        "generic_ballot_margin_dem": national.generic_ballot_margin_dem,
        "approval_adjustment_to_national_margin": approval_adjustment,
        "approval_implied_generic_margin_dem": approval_implied_generic_margin_dem,
        "approval_generic_gap_dem": approval_generic_gap,
        "approval_prior_sd": float(config.approval_prior_sd),
        "approval_to_generic_slope": float(config.approval_to_generic_slope),
        "rcp_current_average_margin_dem": float(generic.pct_a - generic.pct_b),
        "recent_precision_weighted_margin_dem": recent_weighted_margin,
        "trump_approve_pct": float(approval.pct_a),
        "trump_disapprove_pct": float(approval.pct_b),
        "trump_net_approval": national.trump_net_approval,
        "approval_recent_poll_rows": int(len(approval_archive)) if not approval_archive.empty else 0,
        "approval_recent_poll_start": approval_start,
        "approval_recent_poll_end": approval_end,
        "approval_curve_points": int(len(approval_curve)),
        "right_track_net": national.right_track_net,
        "source_status": source_status,
        "simulations": config.simulations,
        "history_simulations": config.history_simulations,
        "visible_national_poll_rows": int(len(generic_archive)) if not generic_archive.empty else 0,
        "visible_national_poll_start": poll_start,
        "visible_national_poll_end": poll_end,
        "generic_poll_archive_rows": int(len(generic_archive)) if not generic_archive.empty else 0,
        "generic_poll_exact_date_rows": exact_date_rows,
        "generic_poll_inferred_date_rows": inferred_date_rows,
        "generic_poll_exact_sample_rows": exact_sample_rows,
        "generic_poll_inferred_sample_rows": inferred_sample_rows,
        "history_points": int(len(history_curve)),
        "history_start_date": history_curve["as_of_date"].min() if not history_curve.empty else as_of_date.isoformat(),
        "history_end_date": history_curve["as_of_date"].max() if not history_curve.empty else as_of_date.isoformat(),
        "history_backcast_days": int((history_curve.get("history_stage") == "prior_only").sum()) if not history_curve.empty and "history_stage" in history_curve.columns else 0,
        "history_observed_days": int((history_curve.get("history_stage") == "observed_poll_filter").sum()) if not history_curve.empty and "history_stage" in history_curve.columns else 0,
        "history_initial_gop_control_prob": float(history_curve.iloc[0]["gop_control_prob"]) if not history_curve.empty else outputs.summary.gop_control_prob,
        "history_initial_expected_gop_seats": float(history_curve.iloc[0]["expected_gop_seats"]) if not history_curve.empty else outputs.summary.expected_gop_seats,
        "history_neutral_margin_dem": history_meta.get("history_neutral_margin_dem"),
        "history_mode": history_meta.get("history_mode"),
        "history_campaign_inputs_mode": history_meta.get("history_campaign_inputs_mode"),
        "history_approval_mode": history_meta.get("history_approval_mode"),
        "history_archive_rows": history_meta.get("history_archive_rows"),
        "history_archive_start": history_meta.get("history_archive_start"),
        "history_current_endpoint_gap_prob": history_meta.get("history_current_endpoint_gap_prob"),
        "history_current_endpoint_gap_seats": history_meta.get("history_current_endpoint_gap_seats"),
        "run_history_points": int(len(run_history)),
    }
    with (paths.latest_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(ensure_json_serializable(summary_dict), f, indent=2)

    audit = {
        "as_of_date": as_of_date.isoformat(),
        "simulations": config.simulations,
        "history_simulations": config.history_simulations,
        "source_status": source_status,
        "generic_poll_archive_rows": int(len(generic_archive)) if not generic_archive.empty else 0,
        "generic_poll_archive_start": poll_start,
        "generic_poll_archive_end": poll_end,
        "generic_poll_exact_date_rows": exact_date_rows,
        "generic_poll_inferred_date_rows": inferred_date_rows,
        "generic_poll_exact_sample_rows": exact_sample_rows,
        "generic_poll_inferred_sample_rows": inferred_sample_rows,
        "approval_recent_poll_rows": int(len(approval_archive)) if not approval_archive.empty else 0,
        "approval_recent_poll_start": approval_start,
        "approval_recent_poll_end": approval_end,
        "approval_curve_points": int(len(approval_curve)),
        "approval_to_generic_slope": float(config.approval_to_generic_slope),
        "approval_prior_sd": float(config.approval_prior_sd),
        "approval_implied_generic_margin_dem": approval_implied_generic_margin_dem,
        "approval_adjustment_to_national_margin": approval_adjustment,
        "approval_generic_gap_dem": approval_generic_gap,
        "rcp_current_average_margin_dem": float(generic.pct_a - generic.pct_b),
        "recent_precision_weighted_margin_dem": recent_weighted_margin,
        "filtered_generic_ballot_margin_dem": national.generic_ballot_margin_dem,
        "national_mean_margin_dem": national.mean_margin_dem,
        "trump_approve_pct": float(approval.pct_a),
        "trump_disapprove_pct": float(approval.pct_b),
        "trump_net_approval": national.trump_net_approval,
        "right_track_net": national.right_track_net,
        "districts_total": int(len(priors)),
        "districts_open": int(priors["open_seat"].sum()),
        "districts_with_polls": int((priors["poll_count"] > 0).sum()),
        "districts_with_ratings": int(priors["rating"].notna().sum()) if "rating" in priors.columns else 0,
        "history_points": int(len(history_curve)),
        "history_backcast_days": int((history_curve.get("history_stage") == "prior_only").sum()) if not history_curve.empty and "history_stage" in history_curve.columns else 0,
        "history_observed_days": int((history_curve.get("history_stage") == "observed_poll_filter").sum()) if not history_curve.empty and "history_stage" in history_curve.columns else 0,
        "history_neutral_margin_dem": history_meta.get("history_neutral_margin_dem"),
        "history_mode": history_meta.get("history_mode"),
        "history_campaign_inputs_mode": history_meta.get("history_campaign_inputs_mode"),
        "history_approval_mode": history_meta.get("history_approval_mode"),
        "history_current_endpoint_gap_prob": history_meta.get("history_current_endpoint_gap_prob"),
        "history_current_endpoint_gap_seats": history_meta.get("history_current_endpoint_gap_seats"),
        "run_history_points": int(len(run_history)),
        "seed": int(seed),
    }
    with paths.run_audit_json.open("w", encoding="utf-8") as f:
        json.dump(ensure_json_serializable(audit), f, indent=2)

    return {
        "summary": summary_dict,
        "history": history_curve,
        "run_history": run_history,
        "audit": audit,
    }
