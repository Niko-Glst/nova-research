"""What counts as a ticker symbol, checked before one reaches a data provider."""

from __future__ import annotations

# Real symbols are letters with an optional dot or hyphen (BRK.B, RDS-A); anything
# else is rejected before it reaches a provider.
TICKER_MAX_LENGTH = 12


def valid_ticker(raw: str | None) -> str | None:
    """Return a normalized ticker, or None when the input is not one."""
    candidate = (raw or "").strip().upper()
    if not candidate or len(candidate) > TICKER_MAX_LENGTH:
        return None
    if not all(character.isalnum() or character in ".-^" for character in candidate):
        return None
    if not any(character.isalpha() for character in candidate):
        return None
    return candidate
