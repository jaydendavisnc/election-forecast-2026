from __future__ import annotations

import io
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from .config import (
    DDHQ_GENERIC_URL,
    DDHQ_RIGHT_TRACK_URL,
    DDHQ_TRUMP_APPROVAL_URL,
    DOWNBALLOT_OPEN_SEATS_CSV,
    DOWNBALLOT_PRES_CSV,
    OPENFEC_API_BASE,
    STATE_ABBR_TO_NAME,
    URL_270TOWIN_CONSENSUS,
    URL_270TOWIN_POLL_ROOT,
    ForecastConfig,
    ProjectPaths,
    VOTING_STATE_ABBRS,
)
from .utils import (
    DISTRICT_RE,
    FetchError,
    clip,
    fetch_json,
    fetch_text,
    infer_sample_size,
    normalize_district_code,
    parse_float,
    parse_month_day_range,
    read_csv_from_text,
    sample_to_margin_sd,
    safe_to_datetime,
    weighted_mean,
    weighted_std,
)


@dataclass(slots=True)
class DDHQMetric:
    metric: str
    label_a: str
    label_b: str
    pct_a: float
    pct_b: float
    margin_a: float
    polls_included: Optional[int]
    recent_polls: pd.DataFrame
    source_url: str
    is_live: bool = False


def load_house_baseline(paths: ProjectPaths) -> pd.DataFrame:
    df = pd.read_csv(paths.baseline_csv)
    df["district_code"] = df["district_code"].map(normalize_district_code)
    df = df[df["state_abbr"].isin(VOTING_STATE_ABBRS)].copy()
    df["two_party_dem_margin_2024"] = pd.to_numeric(
        df["two_party_dem_margin_2024"], errors="coerce"
    )
    df["votes_dem_2024"] = pd.to_numeric(df["votes_dem_2024"], errors="coerce")
    df["votes_gop_2024"] = pd.to_numeric(df["votes_gop_2024"], errors="coerce")

    df["winner_party_2024"] = np.select(
        [
            df["votes_dem_2024"] > df["votes_gop_2024"],
            df["votes_gop_2024"] > df["votes_dem_2024"],
        ],
        ["DEM", "REP"],
        default=None,
    )

    # A few districts in the baseline file have no reported two-party vote totals because the
    # major-party race was effectively uncontested. Keep them as safe seats instead of allowing
    # structural priors to go missing entirely.
    uncontested_party_overrides = {
        "FL-20": "DEM",
        "OK-03": "REP",
    }
    missing_party = df["winner_party_2024"].isna()
    df.loc[missing_party, "winner_party_2024"] = (
        df.loc[missing_party, "district_code"].map(uncontested_party_overrides)
    )

    missing_margin = df["two_party_dem_margin_2024"].isna()
    df.loc[missing_margin & (df["winner_party_2024"] == "DEM"), "two_party_dem_margin_2024"] = 40.0
    df.loc[missing_margin & (df["winner_party_2024"] == "REP"), "two_party_dem_margin_2024"] = -40.0

    df["votes_dem_2024"] = df["votes_dem_2024"].fillna(0.0)
    df["votes_gop_2024"] = df["votes_gop_2024"].fillna(0.0)
    return df.sort_values(["state_abbr", "district_code"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# DDHQ national poll pages
# ---------------------------------------------------------------------------


def _extract_first_percent(text: str, label: str) -> Optional[float]:
    m = re.search(rf"\b{re.escape(label)}\b\s*([0-9]+(?:\.[0-9]+)?)%", text, flags=re.I)
    if not m:
        return None
    return float(m.group(1))


def _parse_ddhq_recent_polls(text: str, label_a: str, label_b: str) -> pd.DataFrame:
    if "polls included in this average" not in text:
        return pd.DataFrame()
    tail = text.split("polls included in this average", 1)[-1]
    if "Show More" in tail:
        tail = tail.split("Show More", 1)[0]
    if "Read our" in tail:
        tail = tail.split("Read our", 1)[0]

    date_chunk_re = re.compile(
        r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d+\s*[\u2013\-]\s*\d+.*?)(?=(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d+\s*[\u2013\-]\s*\d+|$)",
        flags=re.S,
    )
    sample_re = re.compile(
        r"([\d,]+)\s+(LV|RV|Adults|Men|Women|Democrats|Republicans|Independents|White|Hispanic|African American)",
        flags=re.I,
    )

    rows: list[dict[str, Any]] = []
    for chunk in date_chunk_re.findall(tail):
        date_m = re.search(
            r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d+\s*[\u2013\-]\s*\d+)",
            chunk,
        )
        sample_m = sample_re.search(chunk)
        if not date_m or not sample_m:
            continue
        pct_vals = re.findall(r"(\d+(?:\.\d+)?)%", chunk)
        if len(pct_vals) < 2:
            continue
        pct_a = float(pct_vals[0])
        pct_b = float(pct_vals[1])
        pollster_start = sample_m.end()
        label_pos = chunk.find(label_a, pollster_start)
        if label_pos == -1:
            label_pos = chunk.find(label_b, pollster_start)
        pollster = chunk[pollster_start:label_pos if label_pos != -1 else None].strip()
        pollster = re.sub(r"\s+", " ", pollster)
        rows.append(
            {
                "field_dates": date_m.group(1),
                "end_date": parse_month_day_range(date_m.group(1), 2026),
                "sample_size": int(sample_m.group(1).replace(",", "")),
                "population": sample_m.group(2).upper(),
                "pollster": pollster,
                "pct_a": pct_a,
                "pct_b": pct_b,
                "margin_a": pct_a - pct_b,
                "label_a": label_a,
                "label_b": label_b,
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["field_dates", "sample_size", "population", "pollster"]).copy()
    return df


def fetch_ddhq_metric(
    url: str,
    label_a: str,
    label_b: str,
    metric_name: str,
    paths: ProjectPaths,
    config: ForecastConfig,
    use_cache: bool = True,
) -> DDHQMetric:
    html = fetch_text(
        url,
        cache_dir=paths.cache_dir,
        ttl_hours=config.cache_hours_short if use_cache else 0,
        timeout=config.request_timeout_seconds,
        force_refresh=not use_cache,
    )
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text("\n", strip=True)

    pct_a = _extract_first_percent(text, label_a)
    pct_b = _extract_first_percent(text, label_b)
    if pct_a is None or pct_b is None:
        raise FetchError(f"Could not parse DDHQ values for {metric_name}")

    polls_included = None
    m = re.search(r"(\d+)\s+polls included in this average", text, flags=re.I)
    if m:
        polls_included = int(m.group(1))

    recent = _parse_ddhq_recent_polls(text, label_a, label_b)
    return DDHQMetric(
        metric=metric_name,
        label_a=label_a,
        label_b=label_b,
        pct_a=pct_a,
        pct_b=pct_b,
        margin_a=pct_a - pct_b,
        polls_included=polls_included,
        recent_polls=recent,
        source_url=url,
        is_live=True,
    )


def load_seed_context(paths: ProjectPaths) -> tuple[DDHQMetric, DDHQMetric, DDHQMetric]:
    import json

    context_path = paths.seed_dir / "current_context.json"
    with context_path.open("r", encoding="utf-8") as f:
        ctx = json.load(f)
    recent_polls = pd.read_csv(paths.seed_dir / "recent_generic_ballot_polls.csv")
    for col in ["sample_size", "pct_a", "pct_b", "margin_a"]:
        recent_polls[col] = pd.to_numeric(recent_polls[col], errors="coerce")
    if "field_dates" in recent_polls.columns and "end_date" not in recent_polls.columns:
        recent_polls["end_date"] = recent_polls["field_dates"].map(lambda x: parse_month_day_range(str(x), 2026))
    generic = DDHQMetric(
        metric="generic_ballot",
        label_a="Democrat",
        label_b="Republican",
        pct_a=ctx["generic_ballot"]["dem_pct"],
        pct_b=ctx["generic_ballot"]["rep_pct"],
        margin_a=ctx["generic_ballot"]["dem_pct"] - ctx["generic_ballot"]["rep_pct"],
        polls_included=ctx["generic_ballot"]["polls_included"],
        recent_polls=recent_polls,
        source_url=DDHQ_GENERIC_URL,
        is_live=False,
    )
    approval = DDHQMetric(
        metric="trump_approval",
        label_a="Approve",
        label_b="Disapprove",
        pct_a=ctx["trump_approval"]["approve_pct"],
        pct_b=ctx["trump_approval"]["disapprove_pct"],
        margin_a=ctx["trump_approval"]["approve_pct"] - ctx["trump_approval"]["disapprove_pct"],
        polls_included=ctx["trump_approval"]["polls_included"],
        recent_polls=pd.DataFrame(),
        source_url=DDHQ_TRUMP_APPROVAL_URL,
        is_live=False,
    )
    right_track = DDHQMetric(
        metric="right_track",
        label_a="Right Track",
        label_b="Wrong Track",
        pct_a=ctx["right_track"]["right_track_pct"],
        pct_b=ctx["right_track"]["wrong_track_pct"],
        margin_a=ctx["right_track"]["right_track_pct"] - ctx["right_track"]["wrong_track_pct"],
        polls_included=ctx["right_track"]["polls_included"],
        recent_polls=pd.DataFrame(),
        source_url=DDHQ_RIGHT_TRACK_URL,
        is_live=False,
    )
    return generic, approval, right_track


def fetch_national_context(
    paths: ProjectPaths,
    config: ForecastConfig,
    use_live: bool = True,
) -> tuple[DDHQMetric, DDHQMetric, DDHQMetric]:
    if not use_live:
        return load_seed_context(paths)
    try:
        generic = fetch_ddhq_metric(
            DDHQ_GENERIC_URL,
            label_a="Democrat",
            label_b="Republican",
            metric_name="generic_ballot",
            paths=paths,
            config=config,
            use_cache=True,
        )
        approval = fetch_ddhq_metric(
            DDHQ_TRUMP_APPROVAL_URL,
            label_a="Approve",
            label_b="Disapprove",
            metric_name="trump_approval",
            paths=paths,
            config=config,
            use_cache=True,
        )
        right_track = fetch_ddhq_metric(
            DDHQ_RIGHT_TRACK_URL,
            label_a="Right Track",
            label_b="Wrong Track",
            metric_name="right_track",
            paths=paths,
            config=config,
            use_cache=True,
        )
        return generic, approval, right_track
    except Exception:
        return load_seed_context(paths)


# ---------------------------------------------------------------------------
# The Downballot Google Sheets exports
# ---------------------------------------------------------------------------


def _combine_two_header_rows(header_a: pd.Series, header_b: pd.Series) -> list[str]:
    cols: list[str] = []
    for a, b in zip(header_a.fillna(""), header_b.fillna("")):
        aa = str(a).strip()
        bb = str(b).strip()
        name = " ".join(x for x in [aa, bb] if x and x.lower() != "nan")
        name = re.sub(r"\s+", " ", name).strip()
        cols.append(name if name else "unnamed")
    return cols


def parse_presidential_by_district_csv(text: str) -> pd.DataFrame:
    raw = pd.read_csv(io.StringIO(text), header=None)
    header_idx = None
    for idx, row in raw.iterrows():
        row_vals = row.fillna("").astype(str).tolist()
        if any(v.strip().lower() == "district" for v in row_vals):
            header_idx = idx
            break
    if header_idx is None:
        raise ValueError("Could not locate header row in presidential-by-district export")

    cols = _combine_two_header_rows(raw.loc[header_idx], raw.loc[header_idx + 1])
    df = raw.loc[header_idx + 2 :].copy()
    df.columns = cols
    district_col = next(c for c in df.columns if c.lower().startswith("district"))
    df[district_col] = df[district_col].map(normalize_district_code)
    df = df[df[district_col].notna()].copy()

    def find_col(patterns: Iterable[str]) -> Optional[str]:
        for pat in patterns:
            for col in df.columns:
                if pat.lower() in col.lower():
                    return col
        return None

    harris_col = find_col(["2024 harris"])
    trump24_col = find_col(["2024 trump"])
    margin24_col = find_col(["2024 margin"])
    biden_col = find_col(["2020 biden"])
    trump20_col = find_col(["2020 trump"])
    margin20_col = find_col(["2020 margin"])
    incumbent_col = find_col(["incumbent"])
    party_col = find_col(["party"])

    out = pd.DataFrame(
        {
            "district_code": df[district_col].map(normalize_district_code),
            "incumbent_label": df[incumbent_col].astype(str).str.strip() if incumbent_col else None,
            "party_label": df[party_col].astype(str).str.strip() if party_col else None,
            "pres24_dem_pct": df[harris_col].map(parse_float) if harris_col else None,
            "pres24_gop_pct": df[trump24_col].map(parse_float) if trump24_col else None,
            "pres24_dem_margin": df[margin24_col].map(parse_float) if margin24_col else None,
            "pres20_dem_pct": df[biden_col].map(parse_float) if biden_col else None,
            "pres20_gop_pct": df[trump20_col].map(parse_float) if trump20_col else None,
            "pres20_dem_margin": df[margin20_col].map(parse_float) if margin20_col else None,
        }
    )
    return out.reset_index(drop=True)


def fetch_presidential_by_district(
    paths: ProjectPaths,
    config: ForecastConfig,
    use_live: bool = True,
) -> pd.DataFrame:
    if not use_live:
        seed = paths.seed_dir / "presidential_by_district_seed.csv"
        return pd.read_csv(seed) if seed.exists() else pd.DataFrame()
    try:
        text = fetch_text(
            DOWNBALLOT_PRES_CSV,
            cache_dir=paths.cache_dir,
            ttl_hours=config.cache_hours_long,
            timeout=config.request_timeout_seconds,
        )
        df = parse_presidential_by_district_csv(text)
        df.attrs["is_live"] = True
        df.to_csv(paths.latest_dir / "presidential_by_district.csv", index=False)
        return df
    except Exception:
        seed = paths.seed_dir / "presidential_by_district_seed.csv"
        df = pd.read_csv(seed) if seed.exists() else pd.DataFrame()
        df.attrs["is_live"] = False
        return df


_SECTION_MAP = {
    "SEATS HELD BY REPUBLICANS": "REP_OPEN",
    "SEATS HELD BY DEMOCRATS": "DEM_OPEN",
    "NONCONTRIBUTING DEPARTURES": "NONCONTRIB",
}


def parse_open_seat_tracker_csv(text: str) -> pd.DataFrame:
    raw = pd.read_csv(io.StringIO(text), header=None).fillna("")
    section = None
    records: list[dict[str, Any]] = []
    for _, row in raw.iterrows():
        first = str(row.iloc[0]).strip()
        if not first:
            continue
        for prefix, mapped in _SECTION_MAP.items():
            if first.startswith(prefix):
                section = mapped
                first = ""
                break
        if not section or not first:
            continue
        district_code = normalize_district_code(first)
        if not district_code:
            continue
        records.append(
            {
                "district_code": district_code,
                "section": section,
                "holding_party": "REP" if section == "REP_OPEN" else "DEM" if section == "DEM_OPEN" else None,
                "open_seat": section in {"REP_OPEN", "DEM_OPEN"},
                "noncontributing": section == "NONCONTRIB",
                "incumbent": str(row.iloc[1]).strip() if len(row) > 1 else None,
                "party_label": str(row.iloc[2]).strip() if len(row) > 2 else None,
                "announced": str(row.iloc[3]).strip() if len(row) > 3 else None,
                "reason": str(row.iloc[4]).strip() if len(row) > 4 else None,
                "harris_pct": parse_float(row.iloc[5]) if len(row) > 5 else None,
                "trump24_pct": parse_float(row.iloc[6]) if len(row) > 6 else None,
                "biden_pct": parse_float(row.iloc[7]) if len(row) > 7 else None,
                "trump20_pct": parse_float(row.iloc[8]) if len(row) > 8 else None,
                "preferred_successor": str(row.iloc[9]).strip() if len(row) > 9 else None,
            }
        )
    return pd.DataFrame(records)


def fetch_open_seats(
    paths: ProjectPaths,
    config: ForecastConfig,
    use_live: bool = True,
) -> pd.DataFrame:
    seed = paths.seed_dir / "open_seats_seed.csv"
    if not use_live:
        df = pd.read_csv(seed) if seed.exists() else pd.DataFrame()
        df.attrs["is_live"] = False
        return df
    try:
        text = fetch_text(
            DOWNBALLOT_OPEN_SEATS_CSV,
            cache_dir=paths.cache_dir,
            ttl_hours=config.cache_hours_long,
            timeout=config.request_timeout_seconds,
        )
        df = parse_open_seat_tracker_csv(text)
        df.attrs["is_live"] = True
        df.to_csv(paths.latest_dir / "open_seats.csv", index=False)
        return df
    except Exception:
        df = pd.read_csv(seed) if seed.exists() else pd.DataFrame()
        df.attrs["is_live"] = False
        return df


# ---------------------------------------------------------------------------
# 270toWin consensus ratings and district polls
# ---------------------------------------------------------------------------


def fetch_consensus_ratings(
    paths: ProjectPaths,
    config: ForecastConfig,
    use_live: bool = True,
) -> pd.DataFrame:
    seed = paths.seed_dir / "consensus_ratings_seed.csv"
    if not use_live:
        df = pd.read_csv(seed) if seed.exists() else pd.DataFrame()
        df.attrs["is_live"] = False
        return df
    try:
        html = fetch_text(
            URL_270TOWIN_CONSENSUS,
            cache_dir=paths.cache_dir,
            ttl_hours=config.cache_hours_short,
            timeout=config.request_timeout_seconds,
        )
        text = BeautifulSoup(html, "lxml").get_text("\n", strip=True)
        labels = [
            ("Likely Dem", "likely_dem"),
            ("Leans Dem", "lean_dem"),
            ("Toss-up", "tossup"),
            ("Leans Rep", "lean_rep"),
            ("Likely Rep", "likely_rep"),
        ]
        blocks: list[tuple[str, str]] = []
        for i, (human, key) in enumerate(labels):
            start = text.find(human)
            if start == -1:
                continue
            end = text.find(labels[i + 1][0], start + len(human)) if i + 1 < len(labels) else text.find("Map Updated", start)
            if end == -1:
                end = len(text)
            blocks.append((key, text[start:end]))
        rows: list[dict[str, str]] = []
        for key, block in blocks:
            district_codes = re.findall(r"\b[A-Z]{2}-(?:AL|\d{1,2})\b", block)
            for code in district_codes:
                rows.append({"district_code": normalize_district_code(code), "rating": key})
        df = pd.DataFrame(rows).drop_duplicates(subset=["district_code"]).reset_index(drop=True)
        if not df.empty:
            df.attrs["is_live"] = True
            df.to_csv(paths.latest_dir / "consensus_ratings.csv", index=False)
            return df
        raise FetchError("No consensus ratings parsed")
    except Exception:
        df = pd.read_csv(seed) if seed.exists() else pd.DataFrame()
        df.attrs["is_live"] = False
        return df


def _district_to_270towin_urls(district_code: str) -> list[str]:
    state, district = district_code.split("-")
    state_slug = STATE_ABBR_TO_NAME[state].lower().replace(" ", "-")
    urls: list[str] = []
    if district == "AL":
        urls.extend(
            [
                f"{URL_270TOWIN_POLL_ROOT}/{state_slug}/at-large",
                f"{URL_270TOWIN_POLL_ROOT}/{state_slug}/district-0",
                f"{URL_270TOWIN_POLL_ROOT}/{state_slug}/district-00",
            ]
        )
    else:
        n = int(district)
        urls.extend(
            [
                f"{URL_270TOWIN_POLL_ROOT}/{state_slug}/district-{n}",
                f"{URL_270TOWIN_POLL_ROOT}/{state_slug}/district-{district}",
            ]
        )
    return urls


def _parse_270towin_poll_tables(html: str, district_code: str, today_year: int = 2026) -> pd.DataFrame:
    tables: list[pd.DataFrame] = []
    try:
        tables = pd.read_html(io.StringIO(html))
    except ValueError:
        return pd.DataFrame()

    all_rows: list[dict[str, Any]] = []
    for table in tables:
        cols = [str(c).strip().lower() for c in table.columns]
        if not any(c in cols for c in ["poll", "date", "field dates", "sample", "result", "spread"]):
            continue
        t = table.copy()
        t.columns = [str(c).strip() for c in table.columns]
        t = t.dropna(how="all")
        for _, row in t.iterrows():
            row_text = " | ".join(str(v) for v in row.tolist() if pd.notna(v))
            margin = None
            result_text = row_text
            m = re.search(r"(?:Democrat|D)\s*\+\s*(\d+(?:\.\d+)?)", row_text, flags=re.I)
            if m:
                margin = float(m.group(1))
            m = re.search(r"(?:Republican|Rep|R)\s*\+\s*(\d+(?:\.\d+)?)", row_text, flags=re.I)
            if m:
                margin = -float(m.group(1))
            if margin is None:
                # Try explicit numeric columns for Dem/Rep if present.
                dem_col = next((c for c in t.columns if c.lower() in {"democrat", "dem", "d"}), None)
                rep_col = next((c for c in t.columns if c.lower() in {"republican", "rep", "r"}), None)
                if dem_col and rep_col:
                    dem_pct = parse_float(row.get(dem_col))
                    rep_pct = parse_float(row.get(rep_col))
                    if dem_pct is not None and rep_pct is not None:
                        margin = dem_pct - rep_pct
            if margin is None:
                continue
            date_value = None
            for candidate_col in ["Field Dates", "Date", "Dates"]:
                if candidate_col in t.columns:
                    date_value = row.get(candidate_col)
                    break
            if date_value is None:
                mdate = re.search(r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d+\s*[\u2013\-]\s*\d+)", row_text)
                date_value = mdate.group(1) if mdate else None
            sample_value = None
            for candidate_col in ["Sample", "Samples"]:
                if candidate_col in t.columns:
                    sample_value = row.get(candidate_col)
                    break
            sample_size = infer_sample_size(sample_value or row_text)
            all_rows.append(
                {
                    "district_code": district_code,
                    "field_dates": str(date_value) if date_value is not None else None,
                    "end_date": parse_month_day_range(str(date_value), today_year) if date_value is not None else None,
                    "sample_size": sample_size,
                    "margin_dem": margin,
                    "raw_row": result_text,
                }
            )
    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset=["district_code", "field_dates", "sample_size", "margin_dem"])
    return df


def _fetch_single_district_poll_page(
    district_code: str,
    paths: ProjectPaths,
    config: ForecastConfig,
) -> pd.DataFrame:
    for url in _district_to_270towin_urls(district_code):
        try:
            html = fetch_text(
                url,
                cache_dir=paths.cache_dir,
                ttl_hours=config.cache_hours_short,
                timeout=config.request_timeout_seconds,
            )
            parsed = _parse_270towin_poll_tables(html, district_code)
            if not parsed.empty:
                parsed["source_url"] = url
                return parsed
        except Exception:
            continue
    return pd.DataFrame()


def fetch_district_polls_270towin(
    district_codes: list[str],
    paths: ProjectPaths,
    config: ForecastConfig,
    max_workers: int = 8,
) -> pd.DataFrame:
    if not district_codes:
        df = pd.DataFrame()
        df.attrs["is_live"] = False
        return df
    rows: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_fetch_single_district_poll_page, code, paths, config): code
            for code in district_codes
        }
        for fut in as_completed(futures):
            try:
                df = fut.result()
                if not df.empty:
                    rows.append(df)
            except Exception:
                pass
    if not rows:
        df = pd.DataFrame()
        df.attrs["is_live"] = False
        return df
    out = pd.concat(rows, ignore_index=True)
    out.attrs["is_live"] = True
    out.to_csv(paths.latest_dir / "district_polls.csv", index=False)
    return out


# ---------------------------------------------------------------------------
# OpenFEC candidate and finance data
# ---------------------------------------------------------------------------


def fetch_fec_house_candidates(
    api_key: str,
    paths: ProjectPaths,
    config: ForecastConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    page = 1
    while True:
        payload = fetch_json(
            f"{OPENFEC_API_BASE}/candidates/",
            params={
                "api_key": api_key,
                "office": "H",
                "cycle": 2026,
                "election_year": 2026,
                "is_active_candidate": True,
                "has_raised_funds": True,
                "per_page": 100,
                "page": page,
                "sort": "name",
            },
            cache_dir=paths.cache_dir,
            ttl_hours=config.cache_hours_short,
            timeout=config.request_timeout_seconds,
        )
        results = payload.get("results", [])
        if not results:
            break
        for r in results:
            district = normalize_district_code(f"{r.get('state')}-{r.get('district')}")
            if not district:
                continue
            rows.append(
                {
                    "candidate_id": r.get("candidate_id"),
                    "name": r.get("name"),
                    "party": r.get("party"),
                    "state": r.get("state"),
                    "district_code": district,
                    "office": r.get("office"),
                    "incumbent_challenge": r.get("incumbent_challenge"),
                    "incumbent_challenge_full": r.get("incumbent_challenge_full"),
                    "candidate_status": r.get("candidate_status"),
                    "last_file_date": r.get("last_file_date"),
                }
            )
        pagination = payload.get("pagination", {})
        if page >= int(pagination.get("pages", page)):
            break
        page += 1
    df = pd.DataFrame(rows).drop_duplicates(subset=["candidate_id"])
    df.attrs["is_live"] = True
    if not df.empty:
        df.to_csv(paths.latest_dir / "fec_candidates.csv", index=False)
    return df


def fetch_fec_candidate_totals(
    candidate_id: str,
    api_key: str,
    paths: ProjectPaths,
    config: ForecastConfig,
) -> dict[str, Any]:
    payload = fetch_json(
        f"{OPENFEC_API_BASE}/candidate/{candidate_id}/totals/",
        params={
            "api_key": api_key,
            "cycle": 2026,
            "election_full": False,
            "per_page": 20,
            "sort": "-cycle",
        },
        cache_dir=paths.cache_dir,
        ttl_hours=config.cache_hours_short,
        timeout=config.request_timeout_seconds,
    )
    totals = payload.get("results", [])
    if not totals:
        return {
            "candidate_id": candidate_id,
            "receipts": np.nan,
            "individual_itemized": np.nan,
            "cash_on_hand": np.nan,
            "net_contributions": np.nan,
            "coverage_end_date": None,
        }
    receipts = 0.0
    individual_itemized = 0.0
    cash_on_hand = 0.0
    net_contributions = 0.0
    coverage_end_date = None
    for t in totals:
        receipts += float(t.get("receipts") or 0.0)
        individual_itemized += float(t.get("individual_itemized_contributions") or 0.0)
        cash_on_hand = max(cash_on_hand, float(t.get("last_cash_on_hand_end_period") or 0.0))
        net_contributions += float(t.get("net_contributions") or 0.0)
        cov = t.get("coverage_end_date")
        if cov and (coverage_end_date is None or cov > coverage_end_date):
            coverage_end_date = cov
    return {
        "candidate_id": candidate_id,
        "receipts": receipts,
        "individual_itemized": individual_itemized,
        "cash_on_hand": cash_on_hand,
        "net_contributions": net_contributions,
        "coverage_end_date": coverage_end_date,
    }


def fetch_fec_finance_by_district(
    api_key: str,
    candidate_overview: pd.DataFrame,
    paths: ProjectPaths,
    config: ForecastConfig,
    district_filter: Optional[Iterable[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if candidate_overview.empty:
        df1, df2 = pd.DataFrame(), pd.DataFrame()
        df1.attrs["is_live"] = False
        df2.attrs["is_live"] = False
        return df1, df2
    if district_filter is not None:
        district_set = set(district_filter)
        candidates = candidate_overview[candidate_overview["district_code"].isin(district_set)].copy()
    else:
        candidates = candidate_overview.copy()

    totals_rows: list[dict[str, Any]] = []
    max_workers = 8
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(fetch_fec_candidate_totals, cid, api_key, paths, config): cid
            for cid in candidates["candidate_id"].dropna().unique().tolist()
        }
        for fut in as_completed(futures):
            try:
                totals_rows.append(fut.result())
            except Exception:
                pass
    totals_df = pd.DataFrame(totals_rows)
    if totals_df.empty:
        empty = pd.DataFrame()
        empty.attrs["is_live"] = False
        candidates.attrs["is_live"] = True
        return candidates, empty

    merged = candidates.merge(totals_df, on="candidate_id", how="left")
    merged["party"] = merged["party"].replace({"DFL": "DEM"})
    merged["party_bucket"] = np.where(merged["party"].str.startswith("DEM"), "DEM", np.where(merged["party"].str.startswith("REP"), "REP", "OTHER"))
    merged = merged[merged["party_bucket"].isin(["DEM", "REP"])].copy()

    def choose_top(group: pd.DataFrame) -> pd.Series:
        sort_cols = ["cash_on_hand", "receipts", "individual_itemized", "name"]
        ascending = [False, False, False, True]
        g = group.sort_values(sort_cols, ascending=ascending)
        return g.iloc[0]

    selected = (
        merged.groupby(["district_code", "party_bucket"], dropna=False)
        .apply(choose_top)
        .reset_index(drop=True)
    )

    rows: list[dict[str, Any]] = []
    for district_code, g in selected.groupby("district_code"):
        dem = g[g["party_bucket"] == "DEM"]
        rep = g[g["party_bucket"] == "REP"]
        dem_row = dem.iloc[0] if not dem.empty else None
        rep_row = rep.iloc[0] if not rep.empty else None
        rows.append(
            {
                "district_code": district_code,
                "dem_candidate": dem_row["name"] if dem_row is not None else None,
                "rep_candidate": rep_row["name"] if rep_row is not None else None,
                "dem_receipts": float(dem_row["receipts"]) if dem_row is not None and pd.notna(dem_row["receipts"]) else np.nan,
                "rep_receipts": float(rep_row["receipts"]) if rep_row is not None and pd.notna(rep_row["receipts"]) else np.nan,
                "dem_itemized": float(dem_row["individual_itemized"]) if dem_row is not None and pd.notna(dem_row["individual_itemized"]) else np.nan,
                "rep_itemized": float(rep_row["individual_itemized"]) if rep_row is not None and pd.notna(rep_row["individual_itemized"]) else np.nan,
                "dem_cash_on_hand": float(dem_row["cash_on_hand"]) if dem_row is not None and pd.notna(dem_row["cash_on_hand"]) else np.nan,
                "rep_cash_on_hand": float(rep_row["cash_on_hand"]) if rep_row is not None and pd.notna(rep_row["cash_on_hand"]) else np.nan,
            }
        )
    district_finance = pd.DataFrame(rows)
    merged.attrs["is_live"] = True
    district_finance.attrs["is_live"] = True
    if not district_finance.empty:
        district_finance.to_csv(paths.latest_dir / "district_finance.csv", index=False)
    return merged, district_finance



# ---------------------------------------------------------------------------
# RCP-backed national polling archive overrides
# ---------------------------------------------------------------------------

RCP_GENERIC_CURRENT_URL = "https://www.realclearpolling.com/polls/state-of-the-union/generic-congressional-vote"
RCP_LATEST_HOUSE_URL = "https://www.realclearpolling.com/latest-polls/house"


def _normalize_pollster_key(raw: Any) -> str:
    return re.sub(r"\s+", " ", str(raw).replace("**", "")).strip().lower()


def _load_generic_poll_archive_seed(paths: ProjectPaths) -> pd.DataFrame:
    candidates = [
        paths.seed_dir / "generic_ballot_polls_master.csv",
        paths.seed_dir / "recent_generic_ballot_polls.csv",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            df.attrs["seed_path"] = str(path)
            return df
    return pd.DataFrame()


def _prepare_generic_poll_archive(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    for col in [
        "sample_size",
        "dem_pct",
        "rep_pct",
        "margin_a",
        "rcp_dem_pct",
        "rcp_rep_pct",
        "official_dem_pct",
        "official_rep_pct",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "margin_a" not in out.columns and {"dem_pct", "rep_pct"}.issubset(out.columns):
        out["margin_a"] = out["dem_pct"] - out["rep_pct"]
    if "end_date" not in out.columns and "field_dates" in out.columns:
        out["end_date"] = out["field_dates"].map(lambda x: parse_month_day_range(str(x), 2026))
    out["end_date"] = pd.to_datetime(out.get("end_date"), errors="coerce")
    if "published_date" not in out.columns:
        out["published_date"] = out["end_date"]
    out["published_date"] = pd.to_datetime(out.get("published_date"), errors="coerce")
    if "start_date" not in out.columns:
        out["start_date"] = out["end_date"]
    out["start_date"] = pd.to_datetime(out.get("start_date"), errors="coerce")
    if "population" in out.columns:
        out["population"] = (
            out["population"]
            .astype(str)
            .str.upper()
            .replace({"A": "ADULTS", "ADULT": "ADULTS", "NAN": np.nan})
        )
    for col in ["date_exact", "sample_exact", "population_exact", "partisan_flag"]:
        if col not in out.columns:
            if col == "partisan_flag":
                out[col] = out.get("pollster", "").astype(str).str.contains(r"\*\*")
            else:
                out[col] = False
        out[col] = out[col].astype(bool)
    out["date_inferred"] = ~out["date_exact"]
    out["sample_inferred"] = ~out["sample_exact"]
    out["population_inferred"] = ~out["population_exact"]
    out["pollster_key"] = out.get("pollster", pd.Series([""] * len(out))).map(_normalize_pollster_key)
    out["obs_date"] = out["end_date"].fillna(out["published_date"])
    out = out.loc[out["obs_date"].notna() & out["margin_a"].notna()].copy()
    out = out.sort_values(["obs_date", "published_date", "pollster_key"]).reset_index(drop=True)
    return out


def _parse_rcp_current_average_text(text: str) -> Optional[tuple[float, float]]:
    compact = re.sub(r"\s+", " ", text)
    m = re.search(
        r"RealClearPolitics Poll Average.*?Democrats\s+([0-9]+(?:\.[0-9]+)?)%.*?Republicans\s+([0-9]+(?:\.[0-9]+)?)%",
        compact,
        flags=re.I,
    )
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))


_WEEKDAY_RE = re.compile(
    r"^(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s+"
    r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})$"
)
_RESULTS_RE = re.compile(r"^Democrats\s+([0-9]+(?:\.[0-9]+)?)\s+Republicans\s+([0-9]+(?:\.[0-9]+)?)$")


def _parse_rcp_latest_house_rows(text: str, year: int) -> pd.DataFrame:
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    month_map = {name: idx for idx, name in enumerate(
        ["January", "February", "March", "April", "May", "June",
         "July", "August", "September", "October", "November", "December"],
        start=1,
    )}

    current_pub: Optional[pd.Timestamp] = None
    rows: list[dict[str, Any]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        date_m = _WEEKDAY_RE.match(line)
        if date_m:
            current_pub = pd.Timestamp(year=year, month=month_map[date_m.group(2)], day=int(date_m.group(3)))
            i += 1
            continue

        if line == "2026 Generic Congressional Vote" and current_pub is not None:
            pollster = None
            dem_pct = None
            rep_pct = None
            j = i + 1
            while j < len(lines) and j < i + 14:
                marker = lines[j]
                if _WEEKDAY_RE.match(marker) or marker == "2026 Generic Congressional Vote":
                    break
                if marker == "Poll":
                    k = j + 1
                    while k < len(lines) and lines[k] in {"Poll", "Results", "Spread"}:
                        k += 1
                    if k < len(lines):
                        pollster = lines[k]
                elif marker == "Results":
                    k = j + 1
                    while k < len(lines) and lines[k] in {"Poll", "Results", "Spread"}:
                        k += 1
                    if k < len(lines):
                        rm = _RESULTS_RE.match(lines[k])
                        if rm:
                            dem_pct = float(rm.group(1))
                            rep_pct = float(rm.group(2))
                j += 1
            if pollster and dem_pct is not None and rep_pct is not None:
                rows.append(
                    {
                        "pollster": pollster,
                        "field_dates": "",
                        "start_date": current_pub.date().isoformat(),
                        "end_date": current_pub.date().isoformat(),
                        "published_date": current_pub.date().isoformat(),
                        "sample_size": np.nan,
                        "population": np.nan,
                        "dem_pct": dem_pct,
                        "rep_pct": rep_pct,
                        "margin_a": dem_pct - rep_pct,
                        "label_a": "Democrat",
                        "label_b": "Republican",
                        "date_exact": False,
                        "sample_exact": False,
                        "population_exact": False,
                        "date_inferred": True,
                        "sample_inferred": True,
                        "population_inferred": True,
                        "partisan_flag": "**" in pollster,
                        "source_family": "rcp_latest_house_page_live",
                        "topline_source": "rcp_latest_page_live",
                        "metadata_source": "published_date_fallback_live",
                        "notes": "Live scrape of the RealClearPolling latest house page; publication-date fallback until field dates/sample metadata are filled.",
                    }
                )
            i = j
            continue
        i += 1

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["pollster_key"] = out["pollster"].map(_normalize_pollster_key)
    out["obs_date"] = pd.to_datetime(out["end_date"], errors="coerce")
    return out


def fetch_rcp_generic_metric(
    paths: ProjectPaths,
    config: ForecastConfig,
    use_cache: bool = True,
) -> DDHQMetric:
    archive = _prepare_generic_poll_archive(_load_generic_poll_archive_seed(paths))
    if archive.empty:
        raise FetchError("No bundled generic-ballot poll archive found")

    current_html = fetch_text(
        RCP_GENERIC_CURRENT_URL,
        cache_dir=paths.cache_dir,
        ttl_hours=config.cache_hours_short if use_cache else 0,
        timeout=config.request_timeout_seconds,
        force_refresh=not use_cache,
    )
    current_text = BeautifulSoup(current_html, "lxml").get_text("\n", strip=True)
    avg_vals = _parse_rcp_current_average_text(current_text)
    if avg_vals is None:
        raise FetchError("Could not parse current RCP generic-ballot average")
    dem_avg, rep_avg = avg_vals

    # Append any newly published rows from the latest-house page without disturbing the
    # historical archive we ship with the project. This keeps live updates daily even if
    # RCP's old epolls archive lags.
    try:
        latest_html = fetch_text(
            RCP_LATEST_HOUSE_URL,
            cache_dir=paths.cache_dir,
            ttl_hours=config.cache_hours_short if use_cache else 0,
            timeout=config.request_timeout_seconds,
            force_refresh=not use_cache,
        )
        latest_text = BeautifulSoup(latest_html, "lxml").get_text("\n", strip=True)
        latest_rows = _parse_rcp_latest_house_rows(latest_text, year=pd.Timestamp.utcnow().year)
        if not latest_rows.empty:
            existing = archive.copy()
            existing["published_date"] = pd.to_datetime(existing["published_date"], errors="coerce")
            max_pub = existing["published_date"].dropna().max() if "published_date" in existing.columns else None
            if max_pub is not None:
                latest_rows = latest_rows.loc[latest_rows["published_date"] > max_pub].copy()
            if not latest_rows.empty:
                keep_cols = sorted(set(existing.columns).union(latest_rows.columns))
                existing = existing.reindex(columns=keep_cols)
                latest_rows = latest_rows.reindex(columns=keep_cols)
                archive = pd.concat([existing, latest_rows], ignore_index=True)
                archive = _prepare_generic_poll_archive(archive)
                archive = archive.drop_duplicates(
                    subset=["pollster_key", "published_date", "margin_a"],
                    keep="last",
                ).reset_index(drop=True)
    except Exception:
        pass

    return DDHQMetric(
        metric="generic_ballot",
        label_a="Democrat",
        label_b="Republican",
        pct_a=float(dem_avg),
        pct_b=float(rep_avg),
        margin_a=float(dem_avg - rep_avg),
        polls_included=int(len(archive)),
        recent_polls=archive.drop(columns=[c for c in ["obs_date"] if c in archive.columns]),
        source_url=RCP_GENERIC_CURRENT_URL,
        is_live=True,
    )


def load_seed_context(paths: ProjectPaths) -> tuple[DDHQMetric, DDHQMetric, DDHQMetric]:
    import json

    context_path = paths.seed_dir / "current_context.json"
    with context_path.open("r", encoding="utf-8") as f:
        ctx = json.load(f)

    archive = _prepare_generic_poll_archive(_load_generic_poll_archive_seed(paths))

    generic = DDHQMetric(
        metric="generic_ballot",
        label_a="Democrat",
        label_b="Republican",
        pct_a=float(ctx["generic_ballot"]["dem_pct"]),
        pct_b=float(ctx["generic_ballot"]["rep_pct"]),
        margin_a=float(ctx["generic_ballot"]["dem_pct"] - ctx["generic_ballot"]["rep_pct"]),
        polls_included=int(ctx["generic_ballot"].get("polls_included", len(archive))),
        recent_polls=archive.drop(columns=[c for c in ["obs_date"] if c in archive.columns]),
        source_url=RCP_GENERIC_CURRENT_URL,
        is_live=False,
    )
    approval_seed = paths.seed_dir / "trump_approval_recent_polls.csv"
    approval_recent = pd.read_csv(approval_seed) if approval_seed.exists() else pd.DataFrame()

    right_track_seed = paths.seed_dir / "right_track_recent_polls.csv"
    right_track_recent = pd.read_csv(right_track_seed) if right_track_seed.exists() else pd.DataFrame()

    approval = DDHQMetric(
        metric="trump_approval",
        label_a="Approve",
        label_b="Disapprove",
        pct_a=float(ctx["trump_approval"]["approve_pct"]),
        pct_b=float(ctx["trump_approval"]["disapprove_pct"]),
        margin_a=float(ctx["trump_approval"]["approve_pct"] - ctx["trump_approval"]["disapprove_pct"]),
        polls_included=ctx["trump_approval"]["polls_included"],
        recent_polls=approval_recent,
        source_url=DDHQ_TRUMP_APPROVAL_URL,
        is_live=False,
    )
    right_track = DDHQMetric(
        metric="right_track",
        label_a="Right Track",
        label_b="Wrong Track",
        pct_a=float(ctx["right_track"]["right_track_pct"]),
        pct_b=float(ctx["right_track"]["wrong_track_pct"]),
        margin_a=float(ctx["right_track"]["right_track_pct"] - ctx["right_track"]["wrong_track_pct"]),
        polls_included=ctx["right_track"]["polls_included"],
        recent_polls=right_track_recent,
        source_url=DDHQ_RIGHT_TRACK_URL,
        is_live=False,
    )
    return generic, approval, right_track


def fetch_national_context(
    paths: ProjectPaths,
    config: ForecastConfig,
    use_live: bool = True,
) -> tuple[DDHQMetric, DDHQMetric, DDHQMetric]:
    seed_generic, seed_approval, seed_right_track = load_seed_context(paths)
    if not use_live:
        return seed_generic, seed_approval, seed_right_track

    generic = seed_generic
    approval = seed_approval
    right_track = seed_right_track

    try:
        generic = fetch_rcp_generic_metric(paths, config, use_cache=True)
    except Exception:
        generic = seed_generic

    try:
        approval = fetch_ddhq_metric(
            DDHQ_TRUMP_APPROVAL_URL,
            label_a="Approve",
            label_b="Disapprove",
            metric_name="trump_approval",
            paths=paths,
            config=config,
            use_cache=True,
        )
    except Exception:
        approval = seed_approval

    try:
        right_track = fetch_ddhq_metric(
            DDHQ_RIGHT_TRACK_URL,
            label_a="Right Track",
            label_b="Wrong Track",
            metric_name="right_track",
            paths=paths,
            config=config,
            use_cache=True,
        )
    except Exception:
        right_track = seed_right_track

    return generic, approval, right_track
