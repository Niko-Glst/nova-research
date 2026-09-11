"""Loads configuration from environment variables (via .env) and the allowlist YAML.

No secrets are ever hardcoded here. All values come from `.env` (see `.env.example`
for the expected keys); missing required values should raise a clear error rather
than silently falling back to an empty string.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
class NewsApiCredentials:
    api_key: str


@dataclass(frozen=True)
class BrokerSandboxCredentials:
    """Credentials for a paper/sandbox broker endpoint only.

    base_url must never point at a live-trading endpoint (see SECURITY.md).
    """

    api_key: str
    api_secret: str
    base_url: str


def get_reddit_credentials() -> RedditCredentials:
    """Read Reddit API credentials from the environment.

    TODO: implement — read REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET / REDDIT_USER_AGENT,
    raise ValueError with a clear message if any required var is missing.
    """
    raise NotImplementedError


def get_news_api_credentials() -> NewsApiCredentials:
    """Read news API credentials from the environment.

    TODO: implement — read NEWS_API_KEY, raise ValueError if missing.
    """
    raise NotImplementedError


def get_broker_sandbox_credentials() -> BrokerSandboxCredentials:
    """Read paper/sandbox broker credentials from the environment.

    TODO: implement. Must validate that base_url is a sandbox/paper endpoint,
    not a live-trading endpoint.
    """
    raise NotImplementedError


def load_allowed_sources() -> dict:
    """Load config/allowed_sources.yaml as a dict.

    TODO: implement with PyYAML (add to requirements if not already present),
    raise FileNotFoundError with a clear message if the file is missing.
    """
    raise NotImplementedError
