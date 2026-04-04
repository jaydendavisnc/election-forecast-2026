from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from .config import DEFAULT_CONFIG, ProjectPaths
from .data_sources import (
    fetch_consensus_ratings,
    fetch_district_polls_270towin,
    fetch_fec_finance_by_district,
    fetch_fec_house_candidates,
    fetch_national_context,
    fetch_open_seats,
    fetch_presidential_by_district,
    load_house_baseline,
)
from .model import (
    NationalEnvironment,
    aggregate_district_polls,
    build_master_frame,
    choose_poll_scan_districts,
    estimate_national_environment,
    prepare_district_priors,
    run_simulation,
    write_outputs,
)


@dataclass(slots=True)
class ForecastRunArtifacts:
    national: NationalEnvironment
    district_master: pd.DataFrame
    district_priors: pd.DataFrame
    district_results: pd.DataFrame
    seat_distribution: pd.DataFrame
    history: pd.DataFrame
    summary: dict[str, Any]
    source_status: dict[str, Any]


class ForecastPipeline:
    def __init__(self, root: str | Path, config=DEFAULT_CONFIG) -> None:
        self.paths = ProjectPaths(Path(root))
        self.paths.ensure()
        self.config = config

    def run(
        self,
        *,
        use_live: bool = True,
        include_fec: bool = True,
        include_district_polls: bool = True,
        fec_api_key: Optional[str] = None,
        as_of_date: Optional[date] = None,
        seed: int = 20260401,
    ) -> ForecastRunArtifacts:
        as_of_date = as_of_date or date.today()

        source_status: dict[str, Any] = {
            "national_context_live": False,
            "presidential_by_district_live": False,
            "open_seats_live": False,
            "consensus_ratings_live": False,
            "district_polls_live": False,
            "fec_live": False,
            "district_poll_scan_count": 0,
        }

        baseline = load_house_baseline(self.paths)
        generic, approval, right_track = fetch_national_context(self.paths, self.config, use_live=use_live)
        source_status["national_context_live"] = bool(getattr(generic, "is_live", False))
        national = estimate_national_environment(generic, approval, right_track, self.config, today=as_of_date)

        pres_df = fetch_presidential_by_district(self.paths, self.config, use_live=use_live)
        source_status["presidential_by_district_live"] = bool(pres_df.attrs.get("is_live", False))

        open_df = fetch_open_seats(self.paths, self.config, use_live=use_live)
        source_status["open_seats_live"] = bool(open_df.attrs.get("is_live", False))

        ratings_df = fetch_consensus_ratings(self.paths, self.config, use_live=use_live)
        source_status["consensus_ratings_live"] = bool(ratings_df.attrs.get("is_live", False))

        candidate_overview = pd.DataFrame()
        district_finance = pd.DataFrame()
        api_key = fec_api_key or os.getenv("FEC_API_KEY")
        if include_fec and api_key:
            try:
                candidate_overview = fetch_fec_house_candidates(api_key, self.paths, self.config)
                if not candidate_overview.empty:
                    # Limit initial finance pass to likely-in-play districts plus districts with active candidates.
                    initial_districts = set()
                    if not ratings_df.empty:
                        initial_districts.update(ratings_df["district_code"].dropna().tolist())
                    if not open_df.empty:
                        initial_districts.update(open_df.loc[open_df["open_seat"] == True, "district_code"].dropna().tolist())
                    candidate_overview, district_finance = fetch_fec_finance_by_district(
                        api_key,
                        candidate_overview,
                        self.paths,
                        self.config,
                        district_filter=initial_districts if initial_districts else None,
                    )
                    source_status["fec_live"] = bool(candidate_overview.attrs.get("is_live", False) or district_finance.attrs.get("is_live", False))
            except Exception:
                candidate_overview = pd.DataFrame()
                district_finance = pd.DataFrame()

        master = build_master_frame(
            baseline,
            pres_df=pres_df,
            open_seats_df=open_df,
            ratings_df=ratings_df,
            candidate_overview=candidate_overview,
            district_finance=district_finance,
            config=self.config,
        )

        priors_prelim = prepare_district_priors(master, national, pd.DataFrame(), self.config)

        district_polls_raw = pd.DataFrame()
        if include_district_polls:
            poll_scan_districts = choose_poll_scan_districts(
                priors_prelim,
                open_df,
                ratings_df,
                self.config,
            )
            source_status["district_poll_scan_count"] = len(poll_scan_districts)
            if poll_scan_districts:
                try:
                    district_polls_raw = fetch_district_polls_270towin(
                        poll_scan_districts,
                        self.paths,
                        self.config,
                    )
                    source_status["district_polls_live"] = bool(district_polls_raw.attrs.get("is_live", False))
                except Exception:
                    district_polls_raw = pd.DataFrame()

        district_poll_agg = aggregate_district_polls(district_polls_raw, self.config, as_of_date=as_of_date)
        priors = prepare_district_priors(master, national, district_poll_agg, self.config)
        outputs = run_simulation(priors, national, self.config, seed=seed)

        written = write_outputs(
            self.paths,
            national=national,
            generic=generic,
            approval=approval,
            right_track=right_track,
            master=master,
            priors=priors,
            outputs=outputs,
            source_status=source_status,
            as_of_date=as_of_date,
            config=self.config,
            district_polls_raw=district_polls_raw,
            seed=seed,
        )
        return ForecastRunArtifacts(
            national=national,
            district_master=master,
            district_priors=priors,
            district_results=outputs.district_results,
            seat_distribution=outputs.seat_distribution,
            history=written["history"],
            summary=written["summary"],
            source_status=source_status,
        )


def run_forecast_project(
    root: str | Path,
    *,
    use_live: bool = True,
    include_fec: bool = True,
    include_district_polls: bool = True,
    fec_api_key: Optional[str] = None,
    as_of_date: Optional[date] = None,
    seed: int = 20260401,
) -> ForecastRunArtifacts:
    pipeline = ForecastPipeline(root)
    return pipeline.run(
        use_live=use_live,
        include_fec=include_fec,
        include_district_polls=include_district_polls,
        fec_api_key=fec_api_key,
        as_of_date=as_of_date,
        seed=seed,
    )
