"""Run the signal study over a universe of symbols.

    python scripts/run_signal_study.py
    python scripts/run_signal_study.py --window 126 AAPL MSFT NVDA

Tests whether the analyst stances carry any relationship to realised returns --
see backtests/signal_study.py for what this can and cannot establish.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtests.signal_study import observations_to_frame, study_signals  # noqa: E402

DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "AMD",
    "JPM", "BAC", "V", "MA", "XOM", "CVX", "JNJ", "PFE",
    "WMT", "COST", "DIS", "NFLX", "RIVN", "MP", "RDDT", "INTC",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Study analyst signals against returns.")
    parser.add_argument("symbols", nargs="*", default=None, help="Symbols to study.")
    parser.add_argument("--window", type=int, default=63, help="Return window in days.")
    parser.add_argument("--csv", help="Write the observation table to this CSV path.")
    args = parser.parse_args()

    universe = args.symbols or DEFAULT_UNIVERSE
    study = study_signals(universe, return_window_days=args.window)
    print(study.summary())

    if args.csv:
        frame = observations_to_frame(study.observations)
        frame.to_csv(args.csv, index=False)
        print(f"\nObservations written to {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
