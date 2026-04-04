from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import run_forecast_project


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the 2026 House forecast pipeline.")
    parser.add_argument("--seed-only", action="store_true", help="Use bundled seed inputs only.")
    parser.add_argument("--no-fec", action="store_true", help="Disable OpenFEC finance inputs.")
    parser.add_argument("--no-district-polls", action="store_true", help="Disable district poll scanning.")
    parser.add_argument("--fec-api-key", default=None, help="Optional OpenFEC API key.")
    args = parser.parse_args()

    run_forecast_project(
        ROOT,
        use_live=not args.seed_only,
        include_fec=not args.no_fec,
        include_district_polls=not args.no_district_polls,
        fec_api_key=args.fec_api_key,
    )


if __name__ == "__main__":
    main()
