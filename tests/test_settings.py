"""Tests for configuration loading and the live-endpoint guard.

The broker URL check enforces SECURITY.md rule 2 (no live execution), so it is
tested for the cases that matter: a live URL must be refused even when every
credential is present and correct.
"""

from __future__ import annotations

import pytest

from config import settings


class TestRequireEnv:
    def test_present_value_is_returned(self, monkeypatch):
        monkeypatch.setenv("NOVA_TEST_VAR", "value")
        assert settings._require_env("NOVA_TEST_VAR") == "value"

    def test_missing_value_names_the_variable(self, monkeypatch):
        monkeypatch.delenv("NOVA_TEST_VAR", raising=False)
        with pytest.raises(ValueError, match="NOVA_TEST_VAR"):
            settings._require_env("NOVA_TEST_VAR")

    def test_whitespace_only_counts_as_missing(self, monkeypatch):
        monkeypatch.setenv("NOVA_TEST_VAR", "   ")
        with pytest.raises(ValueError):
            settings._require_env("NOVA_TEST_VAR")


class TestCredentialLoading:
    def test_reddit_credentials_are_assembled(self, monkeypatch):
        monkeypatch.setenv("REDDIT_CLIENT_ID", "id")
        monkeypatch.setenv("REDDIT_CLIENT_SECRET", "shh")
        monkeypatch.setenv("REDDIT_USER_AGENT", "nova-research/0.1")
        credentials = settings.get_reddit_credentials()
        assert credentials.client_id == "id"
        assert credentials.user_agent == "nova-research/0.1"

    def test_partial_reddit_credentials_are_refused(self, monkeypatch):
        monkeypatch.setenv("REDDIT_CLIENT_ID", "id")
        monkeypatch.delenv("REDDIT_CLIENT_SECRET", raising=False)
        monkeypatch.delenv("REDDIT_USER_AGENT", raising=False)
        with pytest.raises(ValueError, match="REDDIT_CLIENT_SECRET"):
            settings.get_reddit_credentials()


class TestBrokerSandboxGuard:
    @pytest.fixture(autouse=True)
    def _broker_keys(self, monkeypatch):
        monkeypatch.setenv("BROKER_API_KEY", "k")
        monkeypatch.setenv("BROKER_API_SECRET", "s")

    @pytest.mark.parametrize(
        "url",
        [
            "https://paper-api.example.com/v2",
            "https://sandbox.example.com",
            "https://demo.broker.test",
            "http://localhost:8080",
        ],
    )
    def test_sandbox_endpoints_are_accepted(self, monkeypatch, url):
        monkeypatch.setenv("BROKER_BASE_URL", url)
        assert settings.get_broker_sandbox_credentials().base_url == url

    @pytest.mark.parametrize(
        "url",
        [
            "https://api.example.com/v2",
            "https://live.broker.com",
            "https://trading.broker.com/api",
        ],
    )
    def test_live_endpoints_are_refused(self, monkeypatch, url):
        """This is the guardrail that keeps the project simulation-only."""
        monkeypatch.setenv("BROKER_BASE_URL", url)
        with pytest.raises(ValueError, match="paper/sandbox"):
            settings.get_broker_sandbox_credentials()

    def test_malformed_url_is_refused(self, monkeypatch):
        monkeypatch.setenv("BROKER_BASE_URL", "not-a-url-sandbox")
        with pytest.raises(ValueError, match="absolute http"):
            settings.get_broker_sandbox_credentials()


class TestAllowedSources:
    def test_allowlist_loads_expected_sections(self):
        config = settings.load_allowed_sources()
        for key in ("subreddits", "news_domains", "peer_groups", "rate_limits"):
            assert key in config

    def test_rate_limits_have_usable_defaults(self):
        limits = settings.get_rate_limits()
        assert limits["min_seconds_between_requests"] >= 0
        assert limits["requests_per_minute"] > 0

    def test_peer_groups_are_lists_of_tickers(self):
        for niche, peers in settings.load_allowed_sources()["peer_groups"].items():
            assert isinstance(peers, list), niche
            assert all(isinstance(p, str) and p.isupper() for p in peers), niche
