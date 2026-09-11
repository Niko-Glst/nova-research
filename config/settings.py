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
from urllib.parse import urlparse

import yaml
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
ALLOWED_SOURCES_PATH = ROOT_DIR / "config" / "allowed_sources.yaml"

load_dotenv(ROOT_DIR / ".env")

# Substrings that mark a broker base_url as a paper/sandbox endpoint. A URL must
# match one of these to be accepted (see SECURITY.md rule 2 — no live execution).
_SANDBOX_URL_MARKERS = ("paper", "sandbox", "demo", "test", "localhost", "127.0.0.1")


@dataclass(frozen=True)
class RedditCredentials:
    client_id: str
    client_secret: str
    user_agent: str


@dataclass(frozen=True)
class NewsApiCredentials:
    api_key: str


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


@dataclass(frozen=True)
class BrokerSandboxCredentials:
    """Credentials for a paper/sandbox broker endpoint only.

    base_url must never point at a live-trading endpoint (see SECURITY.md).
    """

    api_key: str
    api_secret: str
    base_url: str


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


def get_news_api_credentials() -> NewsApiCredentials:
    """Read news API credentials from the environment."""
    return NewsApiCredentials(api_key=_require_env("NEWS_API_KEY"))


def get_tiingo_credentials() -> TiingoCredentials:
    """Read Tiingo API credentials from the environment."""
    return TiingoCredentials(api_key=_require_env("TIINGO_API_KEY"))


def get_finnhub_credentials() -> FinnhubCredentials:
    """Read Finnhub API credentials from the environment."""
    return FinnhubCredentials(api_key=_require_env("FINNHUB_API_KEY"))


def get_broker_sandbox_credentials() -> BrokerSandboxCredentials:
    """Read paper/sandbox broker credentials from the environment.

    Rejects any base_url that does not look like a paper/sandbox endpoint.
    """
    base_url = _require_env("BROKER_BASE_URL")
    parsed = urlparse(base_url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(
            f"BROKER_BASE_URL must be an absolute http(s) URL, got {base_url!r}."
        )
    if not any(marker in base_url.lower() for marker in _SANDBOX_URL_MARKERS):
        raise ValueError(
            f"Refusing to use BROKER_BASE_URL {base_url!r}: it does not look like a "
            f"paper/sandbox endpoint (expected one of {_SANDBOX_URL_MARKERS} in the URL). "
            f"This project is simulation-only — see SECURITY.md."
        )
    return BrokerSandboxCredentials(
        api_key=_require_env("BROKER_API_KEY"),
        api_secret=_require_env("BROKER_API_SECRET"),
        base_url=base_url,
    )


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


def get_rate_limits() -> dict[str, float]:
    """Return the rate-limit defaults from the allowlist, with safe fallbacks."""
    limits = load_allowed_sources().get("rate_limits") or {}
    return {
        "requests_per_minute": float(limits.get("requests_per_minute", 30)),
        "min_seconds_between_requests": float(
            limits.get("min_seconds_between_requests", 2)
        ),
    }


def get_log_level() -> str:
    """Return the configured log level (defaults to INFO)."""
    return os.getenv("LOG_LEVEL", "INFO").strip().upper() or "INFO"
