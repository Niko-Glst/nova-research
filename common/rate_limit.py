"""Process-wide rate limiting for external API calls.

SECURITY.md rule 4 requires every external call to back off between requests.
Rather than each agent rolling its own sleep, they all share the limiters here,
keyed by provider name, so two agents hitting the same provider still respect a
single combined pace.
"""

from __future__ import annotations

import threading
import time

from config.settings import get_rate_limits


class RateLimiter:
    """Blocks until at least `min_interval` seconds have passed since the last call.

    Thread-safe: the pipeline may fan out across threads, and a per-provider
    limiter is only meaningful if all callers share the same lock.
    """

    def __init__(self, min_interval: float) -> None:
        self.min_interval = max(0.0, float(min_interval))
        self._lock = threading.Lock()
        self._last_call: float | None = None

    def wait(self) -> None:
        """Sleep as long as needed so calls stay `min_interval` apart."""
        with self._lock:
            now = time.monotonic()
            if self._last_call is not None:
                elapsed = now - self._last_call
                remaining = self.min_interval - elapsed
                if remaining > 0:
                    time.sleep(remaining)
                    now = time.monotonic()
            self._last_call = now


_limiters: dict[str, RateLimiter] = {}
_limiters_lock = threading.Lock()


def get_limiter(provider: str) -> RateLimiter:
    """Return the shared limiter for a provider, creating it on first use.

    The interval comes from `rate_limits` in config/allowed_sources.yaml.
    """
    with _limiters_lock:
        limiter = _limiters.get(provider)
        if limiter is None:
            interval = get_rate_limits()["min_seconds_between_requests"]
            limiter = RateLimiter(interval)
            _limiters[provider] = limiter
        return limiter
