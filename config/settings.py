"""Loads configuration from environment variables (via .env) and the allowlist YAML.

No secrets are ever hardcoded here. All values come from `.env` (see `.env.example`
for the expected keys); missing required values raise a clear error rather than
silently falling back to an empty string.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
ALLOWED_SOURCES_PATH = ROOT_DIR / "config" / "allowed_sources.yaml"

load_dotenv(ROOT_DIR / ".env")


@dataclass(frozen=True)
class RedditCredentials:
    client_id: str
    client_secret: str
    user_agent: str


@dataclass(frozen=True)
class TiingoCredentials:
    """Credentials for Tiingo's news endpoint (https://www.tiingo.com).

    Note: Tiingo's News API is a paid add-on. A free key authenticates fine for
    prices and fundamentals but returns HTTP 403 for news.
    """

    api_key: str


@dataclass(frozen=True)
class FinnhubCredentials:
    """Credentials for Finnhub's company-news endpoint (https://finnhub.io)."""

    api_key: str


def _require_env(name: str) -> str:
    """Return a required environment variable, or raise a clear error."""
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(
            f"Required environment variable {name!r} is missing or empty. "
            f"Copy .env.example to .env and fill in a real value."
        )
    return value


def get_reddit_credentials() -> RedditCredentials:
    """Read Reddit API credentials from the environment."""
    return RedditCredentials(
        client_id=_require_env("REDDIT_CLIENT_ID"),
        client_secret=_require_env("REDDIT_CLIENT_SECRET"),
        user_agent=_require_env("REDDIT_USER_AGENT"),
    )


def get_tiingo_credentials() -> TiingoCredentials:
    """Read Tiingo API credentials from the environment."""
    return TiingoCredentials(api_key=_require_env("TIINGO_API_KEY"))


def get_finnhub_credentials() -> FinnhubCredentials:
    """Read Finnhub API credentials from the environment."""
    return FinnhubCredentials(api_key=_require_env("FINNHUB_API_KEY"))


@lru_cache(maxsize=1)
def load_allowed_sources() -> dict:
    """Load config/allowed_sources.yaml as a dict.

    Cached: the allowlist is read once per process and does not change at runtime.
    """
    if not ALLOWED_SOURCES_PATH.exists():
        raise FileNotFoundError(
            f"Allowlist not found at {ALLOWED_SOURCES_PATH}. This file is required — "
            f"every external fetch reads its permitted sources from it (see SECURITY.md)."
        )
    with ALLOWED_SOURCES_PATH.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(
            f"{ALLOWED_SOURCES_PATH} must parse to a mapping, got {type(data).__name__}."
        )
    return data


SCORING_PATH = ROOT_DIR / "config" / "scoring.yaml"


@lru_cache(maxsize=1)
def load_scoring_config() -> dict:
    """Load config/scoring.yaml: model version, leg weights and flow rules."""
    if not SCORING_PATH.exists():
        raise FileNotFoundError(f"Scoring config not found at {SCORING_PATH}.")
    with SCORING_PATH.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{SCORING_PATH} must parse to a mapping, got {type(data).__name__}.")
    return data


def get_sec_user_agent() -> str:
    """User-Agent for SEC EDGAR: SEC asks for a descriptive name and a contact email.

    The variable's name is config/scoring.yaml flows.sec.user_agent_env.
    """
    return _require_env(str(load_scoring_config()["flows"]["sec"]["user_agent_env"]))


def get_rate_limits() -> dict[str, float]:
    """Return the rate-limit defaults from the allowlist, with safe fallbacks."""
    limits = load_allowed_sources().get("rate_limits") or {}
    return {
        "requests_per_minute": float(limits.get("requests_per_minute", 30)),
        "min_seconds_between_requests": float(
            limits.get("min_seconds_between_requests", 2)
        ),
    }
