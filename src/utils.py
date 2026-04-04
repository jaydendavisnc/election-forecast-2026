from __future__ import annotations

import hashlib
import io
import json
import math
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

import numpy as np
import pandas as pd
import requests

DISTRICT_RE = re.compile(r"^([A-Z]{2})-(AL|\d{1,2})$")
MONTH_RE = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"


class FetchError(RuntimeError):
    pass


def normalize_district_code(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip().upper()
    if not s or s == "NAN":
        return None
    s = s.replace(" ", "").replace("_", "-")
    s = s.replace("AT-LARGE", "AL")
    m = re.match(r"^([A-Z]{2})-?(AL|\d{1,2}|00)$", s)
    if not m:
        return None
    state, district = m.group(1), m.group(2)
    if district in {"AL", "00"}:
        return f"{state}-AL"
    return f"{state}-{int(district):02d}"


def parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        if pd.isna(value):
            return None
        return float(value)
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return None
    s = s.replace(",", "")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def combine_normal_estimates(
    mean_a: float,
    sd_a: float,
    mean_b: Optional[float],
    sd_b: Optional[float],
) -> tuple[float, float]:
    if mean_b is None or sd_b is None or sd_b <= 0:
        return mean_a, sd_a
    if sd_a <= 0:
        return mean_b, sd_b
    var_a = sd_a * sd_a
    var_b = sd_b * sd_b
    precision = 1.0 / var_a + 1.0 / var_b
    mean = (mean_a / var_a + mean_b / var_b) / precision
    sd = math.sqrt(1.0 / precision)
    return mean, sd


def now_utc() -> datetime:
    return datetime.utcnow().replace(microsecond=0)


def days_until(target: date, today: Optional[date] = None) -> int:
    today = today or date.today()
    return (target - today).days


def _cache_file_for(cache_dir: Path, key: str, suffix: str) -> Path:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    return cache_dir / f"{digest}{suffix}"


def fetch_text(
    url: str,
    cache_dir: Path,
    ttl_hours: int,
    timeout: int = 30,
    force_refresh: bool = False,
    headers: Optional[dict[str, str]] = None,
) -> str:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_file_for(cache_dir, url, ".txt")
    if path.exists() and not force_refresh:
        age_hours = (time.time() - path.stat().st_mtime) / 3600.0
        if age_hours <= ttl_hours:
            return path.read_text(encoding="utf-8")
    req_headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; HouseForecastBot/1.0; +https://example.com)"
        )
    }
    if headers:
        req_headers.update(headers)
    resp = requests.get(url, timeout=timeout, headers=req_headers)
    if resp.status_code >= 400:
        raise FetchError(f"HTTP {resp.status_code} for {url}")
    text = resp.text
    path.write_text(text, encoding="utf-8")
    return text


def fetch_json(
    url: str,
    params: dict[str, Any],
    cache_dir: Path,
    ttl_hours: int,
    timeout: int = 30,
    force_refresh: bool = False,
) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = url + "?" + "&".join(f"{k}={params[k]}" for k in sorted(params))
    path = _cache_file_for(cache_dir, cache_key, ".json")
    if path.exists() and not force_refresh:
        age_hours = (time.time() - path.stat().st_mtime) / 3600.0
        if age_hours <= ttl_hours:
            return json.loads(path.read_text(encoding="utf-8"))
    resp = requests.get(url, params=params, timeout=timeout)
    if resp.status_code >= 400:
        raise FetchError(f"HTTP {resp.status_code} for {resp.url}")
    payload = resp.json()
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def read_csv_from_text(text: str, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(text), **kwargs)


def weighted_mean(values: Iterable[float], weights: Iterable[float]) -> float:
    v = np.asarray(list(values), dtype=float)
    w = np.asarray(list(weights), dtype=float)
    if len(v) == 0:
        return float("nan")
    return float(np.average(v, weights=w))


def weighted_std(values: Iterable[float], weights: Iterable[float]) -> float:
    v = np.asarray(list(values), dtype=float)
    w = np.asarray(list(weights), dtype=float)
    if len(v) <= 1:
        return float("nan")
    mean = np.average(v, weights=w)
    var = np.average((v - mean) ** 2, weights=w)
    return float(np.sqrt(var))


def infer_sample_size(value: Any) -> Optional[int]:
    if value is None:
        return None
    s = str(value)
    m = re.search(r"([\d,]+)", s)
    if not m:
        return None
    return int(m.group(1).replace(",", ""))


def sample_to_margin_sd(sample_size: Optional[int], floor: float = 2.25) -> float:
    if not sample_size or sample_size <= 0:
        return floor
    se = 100.0 / math.sqrt(sample_size)
    return max(floor, se)


def parse_month_day_range(value: str, year: int) -> Optional[date]:
    s = str(value).strip()
    m = re.match(rf"({MONTH_RE})\s+(\d+)\s*[\u2013\-]\s*(\d+)", s)
    if not m:
        m2 = re.match(rf"({MONTH_RE})\s+(\d+)", s)
        if not m2:
            return None
        month = datetime.strptime(m2.group(1), "%b").month
        day = int(m2.group(2))
        return date(year, month, day)
    month = datetime.strptime(m.group(1), "%b").month
    end_day = int(m.group(3))
    return date(year, month, end_day)


def safe_to_datetime(value: Any) -> Optional[pd.Timestamp]:
    if value is None or value == "":
        return None
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    return ts


def district_sort_key(code: str) -> tuple[str, int]:
    state, district = code.split("-")
    if district == "AL":
        return state, 0
    return state, int(district)


def ensure_json_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: ensure_json_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [ensure_json_serializable(v) for v in value]
    if isinstance(value, tuple):
        return [ensure_json_serializable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return value
