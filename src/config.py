from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict

PROJECT_NAME = "house_2026_forecast"
ELECTION_DATE = date(2026, 11, 3)
TODAY_FOR_SEED = date(2026, 4, 1)

# 2024 House two-party national margin (Democratic margin) derived from the
# 435 voting districts in the bundled 2024 district-level baseline file.
NATIONAL_HOUSE_MARGIN_2024 = -2.825964571232448

# National presidential two-party margins from the American Presidency Project.
NATIONAL_PRES_MARGIN_2024 = -1.4996691434200151
NATIONAL_PRES_MARGIN_2020 = 4.535500065694229

VOTING_STATE_ABBRS = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID",
    "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS",
    "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK",
    "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV",
    "WI", "WY",
}

STATE_ABBR_TO_NAME: Dict[str, str] = {
    "AL": "Alabama",
    "AK": "Alaska",
    "AZ": "Arizona",
    "AR": "Arkansas",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DE": "Delaware",
    "FL": "Florida",
    "GA": "Georgia",
    "HI": "Hawaii",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "IA": "Iowa",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "ME": "Maine",
    "MD": "Maryland",
    "MA": "Massachusetts",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MS": "Mississippi",
    "MO": "Missouri",
    "MT": "Montana",
    "NE": "Nebraska",
    "NV": "Nevada",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NY": "New York",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VT": "Vermont",
    "VA": "Virginia",
    "WA": "Washington",
    "WV": "West Virginia",
    "WI": "Wisconsin",
    "WY": "Wyoming",
}

DDHQ_GENERIC_URL = "https://polls.decisiondeskhq.com/averages/generic-ballot/national/lv-rv-adults"
DDHQ_TRUMP_APPROVAL_URL = (
    "https://polls.decisiondeskhq.com/averages/"
    "presidential-approval/donald-j.-trump-5/national/lv-rv-adults"
)
DDHQ_RIGHT_TRACK_URL = "https://polls.decisiondeskhq.com/averages/right-wrong-track/national/lv-rv-adults"

DOWNBALLOT_PRES_CSV = (
    "https://docs.google.com/spreadsheets/d/"
    "1ng1i_Dm_RMDnEvauH44pgE6JCUsapcuu8F2pCfeLWFo/export?format=csv&gid=620838163"
)
DOWNBALLOT_OPEN_SEATS_CSV = (
    "https://docs.google.com/spreadsheets/d/"
    "12RhR9oZZpyKKceyLO3C5am84abKzu2XqLWjP2LnQDgI/export?format=csv&gid=1397842864"
)
URL_270TOWIN_CONSENSUS = "https://www.270towin.com/2026-house-election/table/consensus-2026-house-forecast"
URL_270TOWIN_POLL_ROOT = "https://www.270towin.com/2026-house-polls"

OPENFEC_API_BASE = "https://api.open.fec.gov/v1"


@dataclass(slots=True)
class ProjectPaths:
    root: Path
    data_dir: Path = field(init=False)
    seed_dir: Path = field(init=False)
    processed_dir: Path = field(init=False)
    runtime_dir: Path = field(init=False)
    latest_dir: Path = field(init=False)
    history_dir: Path = field(init=False)
    cache_dir: Path = field(init=False)
    baseline_csv: Path = field(init=False)
    forecast_history_csv: Path = field(init=False)
    run_history_csv: Path = field(init=False)
    forecast_curve_csv: Path = field(init=False)
    run_audit_json: Path = field(init=False)

    def __post_init__(self) -> None:
        self.data_dir = self.root / "data"
        self.seed_dir = self.data_dir / "seed"
        self.processed_dir = self.data_dir / "processed"
        self.runtime_dir = self.data_dir / "runtime"
        self.latest_dir = self.runtime_dir / "latest"
        self.history_dir = self.data_dir / "history"
        self.cache_dir = self.runtime_dir / "cache"
        self.baseline_csv = self.processed_dir / "house_2024_baseline.csv"
        self.forecast_history_csv = self.history_dir / "forecast_history.csv"
        self.run_history_csv = self.history_dir / "run_history.csv"
        self.forecast_curve_csv = self.latest_dir / "forecast_curve.csv"
        self.run_audit_json = self.latest_dir / "run_audit.json"

    def ensure(self) -> None:
        for p in [
            self.data_dir,
            self.seed_dir,
            self.processed_dir,
            self.runtime_dir,
            self.latest_dir,
            self.history_dir,
            self.cache_dir,
        ]:
            p.mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class ForecastConfig:
    simulations: int = 50000
    history_simulations: int = 1500
    history_max_days: int = 366
    pres_weight_2024: float = 0.65
    house_overperf_clip: float = 15.0

    history_neutral_prior_mean_dem_margin: float = 0.0
    history_neutral_prior_max_weight: float = 0.92
    history_neutral_prior_alpha: float = 1.55
    history_neutral_current_sd: float = 2.75
    history_campaign_signal_alpha: float = 1.45

    carryover_same_incumbent: float = 0.60
    carryover_non_open_nonincumbent: float = 0.35
    carryover_open_seat: float = 0.15

    finance_effect_beta: float = 1.25
    finance_effect_cap: float = 2.5

    rating_margin_map: Dict[str, float] = field(
        default_factory=lambda: {
            "likely_dem": 8.0,
            "lean_dem": 4.0,
            "tossup": 0.0,
            "lean_rep": -4.0,
            "likely_rep": -8.0,
        }
    )
    rating_sd_map: Dict[str, float] = field(
        default_factory=lambda: {
            "likely_dem": 4.5,
            "lean_dem": 5.5,
            "tossup": 7.0,
            "lean_rep": 5.5,
            "likely_rep": 4.5,
        }
    )

    base_prior_sd_incumbent: float = 4.0
    base_prior_sd_open: float = 5.6
    base_prior_sd_missing_pres: float = 6.2
    extra_future_district_sd: float = 1.4

    national_poll_floor_sd: float = 1.0
    national_state_prior_sd: float = 4.5
    national_poll_default_sample_size: int = 1000
    national_poll_house_effect_sd: float = 1.35
    national_poll_missing_date_penalty_sd: float = 0.85
    national_poll_missing_sample_penalty_sd: float = 0.45
    national_poll_unknown_population_penalty_sd: float = 0.30
    national_poll_partisan_penalty_sd: float = 0.65
    national_daily_random_walk_sd: float = 0.17
    national_election_day_shock_sd: float = 0.65
    state_correlation_sd: float = 1.0

    approval_to_generic_slope: float = -0.35
    approval_prior_sd: float = 6.0
    approval_disagreement_tolerance: float = 4.0
    approval_disagreement_inflation: float = 0.10

    district_poll_min_sd: float = 2.25
    district_poll_recency_half_life_days: int = 30

    national_population_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "LV": 1.00,
            "RV": 0.93,
            "ADULTS": 0.84,
        }
    )

    competitive_poll_margin_threshold: float = 14.0
    competitive_poll_limit: int = 120
    request_timeout_seconds: int = 30
    cache_hours_short: int = 6
    cache_hours_long: int = 24


DEFAULT_CONFIG = ForecastConfig()
