"""Historical backtest engine.

Takes entry/exit signals plus price history and runs a vectorized long-only
simulation. Simulation only -- no connection to any live or paper broker order
path, and nothing here places an order.

On the choice of implementation: requirements.txt originally specified vectorbt,
chosen for vectorized speed. The simulation is implemented directly in numpy
instead, for two reasons. It is a few dozen lines for a long-only, one-position
strategy, so the dependency buys little; and vectorbt pulls in numba and
llvmlite, which are frequently unavailable for the newest Python releases,
making the whole pipeline unusable on a machine where they fail to build. The
numpy version runs everywhere pandas does.

The mechanics that matter for honest results:

- **Signals act on the next bar.** A signal computed from a day's close cannot
  be traded at that same close. Entries and exits are shifted forward one bar,
  which is the single most common way a backtest flatters itself.
- **Costs are charged on both sides.** `fees_bps` is applied to each entry and
  each exit, so a strategy that trades constantly pays for it.
- **Exposure is reported.** A strategy in cash 90% of the time has a flattering
  Sharpe; `time_in_market_pct` makes that visible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Trading days per year, for annualizing.
TRADING_DAYS = 252


@dataclass(frozen=True)
class Trade:
    """One completed round trip."""

    entry_index: int
    exit_index: int
    entry_price: float
    exit_price: float
    return_pct: float
    bars_held: int


@dataclass(frozen=True)
class BacktestResult:
    symbol: str
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    num_trades: int
    # Extra diagnostics beyond the original four fields.
    buy_and_hold_return_pct: float = 0.0
    win_rate_pct: float = 0.0
    time_in_market_pct: float = 0.0
    annualized_return_pct: float = 0.0
    volatility_pct: float = 0.0
    sortino_ratio: float = 0.0
    profit_factor: float = 0.0
    equity_curve: pd.Series | None = field(default=None, repr=False)
    trades: list[Trade] = field(default_factory=list, repr=False)

    @property
    def beats_buy_and_hold(self) -> bool:
        return self.total_return_pct > self.buy_and_hold_return_pct

    def summary(self) -> str:
        """A readable summary, including the comparison that matters most."""
        verdict = (
            "beats buy-and-hold"
            if self.beats_buy_and_hold
            else "does NOT beat buy-and-hold"
        )
        return "\n".join(
            [
                f"Backtest: {self.symbol}",
                f"  strategy return:    {self.total_return_pct:>8.2f}%",
                f"  buy & hold return:  {self.buy_and_hold_return_pct:>8.2f}%  ({verdict})",
                f"  annualized:         {self.annualized_return_pct:>8.2f}%",
                f"  max drawdown:       {self.max_drawdown_pct:>8.2f}%",
                f"  volatility (ann.):  {self.volatility_pct:>8.2f}%",
                f"  Sharpe:             {self.sharpe_ratio:>8.2f}",
                f"  Sortino:            {self.sortino_ratio:>8.2f}",
                f"  trades:             {self.num_trades:>8}",
                f"  win rate:           {self.win_rate_pct:>8.1f}%",
                f"  profit factor:      {self.profit_factor:>8.2f}",
                f"  time in market:     {self.time_in_market_pct:>8.1f}%",
            ]
        )


def _positions_from_signals(entries: np.ndarray, exits: np.ndarray) -> np.ndarray:
    """Walk signals into a 0/1 position series, long-only, one position at a time.

    Entries while already long are ignored, as are exits while flat. The walk is
    sequential because position state depends on its own history -- this is the
    one part that cannot be expressed as a pure array operation.
    """
    n = len(entries)
    position = np.zeros(n, dtype=np.int8)
    holding = False

    for i in range(n):
        if holding and exits[i]:
            holding = False
        elif not holding and entries[i]:
            holding = True
        position[i] = 1 if holding else 0

    return position


def _extract_trades(
    position: np.ndarray, prices: np.ndarray, fee_rate: float
) -> list[Trade]:
    """Pair each entry with its exit, net of costs on both sides."""
    trades: list[Trade] = []
    changes = np.diff(np.concatenate([[0], position]))
    entry_indices = np.flatnonzero(changes == 1)
    exit_indices = np.flatnonzero(changes == -1)

    for k, entry_index in enumerate(entry_indices):
        # An open position at the end of the series is closed at the last bar.
        exit_index = exit_indices[k] if k < len(exit_indices) else len(position) - 1
        entry_price = prices[entry_index]
        exit_price = prices[exit_index]
        if entry_price <= 0:
            continue
        gross = exit_price / entry_price
        net = gross * (1 - fee_rate) ** 2  # charged on entry and on exit
        trades.append(
            Trade(
                entry_index=int(entry_index),
                exit_index=int(exit_index),
                entry_price=float(entry_price),
                exit_price=float(exit_price),
                return_pct=float((net - 1.0) * 100.0),
                bars_held=int(exit_index - entry_index),
            )
        )
    return trades


def run_backtest(
    price_history: pd.DataFrame,
    entries: pd.Series,
    exits: pd.Series,
    fees_bps: float = 0.0,
    symbol: str = "",
    shift_signals: bool = True,
) -> BacktestResult:
    """Run a long-only historical backtest for a single symbol.

    `shift_signals` delays every signal by one bar, so a signal derived from a
    close is acted on at the next bar. Set it to False only if the signals are
    already lagged; leaving it on is what keeps the result honest.
    """
    if price_history is None or price_history.empty:
        raise ValueError("price_history is empty, so there is nothing to backtest")
    if "Close" not in price_history.columns:
        raise ValueError("price_history must contain a 'Close' column")

    close = price_history["Close"].astype(float)
    entries = entries.reindex(close.index).fillna(False).astype(bool)
    exits = exits.reindex(close.index).fillna(False).astype(bool)

    if shift_signals:
        entries = entries.shift(1, fill_value=False)
        exits = exits.shift(1, fill_value=False)

    prices = close.to_numpy()
    position = _positions_from_signals(entries.to_numpy(), exits.to_numpy())
    fee_rate = fees_bps / 10_000.0

    # Bar-by-bar returns, earned only while positioned.
    price_returns = np.zeros_like(prices)
    price_returns[1:] = prices[1:] / prices[:-1] - 1.0
    strategy_returns = position * price_returns

    # Charge the round-trip cost on the bar the position changes.
    turnover = np.abs(np.diff(np.concatenate([[0], position])))
    strategy_returns = strategy_returns - turnover * fee_rate

    equity = np.cumprod(1.0 + strategy_returns)
    equity_curve = pd.Series(equity, index=close.index, name="equity")

    total_return = (equity[-1] - 1.0) * 100.0
    buy_and_hold = (prices[-1] / prices[0] - 1.0) * 100.0

    running_max = np.maximum.accumulate(equity)
    drawdowns = equity / running_max - 1.0
    max_drawdown = float(drawdowns.min() * 100.0)

    active = strategy_returns[position > 0]
    std = float(strategy_returns.std(ddof=1)) if len(strategy_returns) > 1 else 0.0
    mean = float(strategy_returns.mean())

    sharpe = (mean / std * np.sqrt(TRADING_DAYS)) if std > 0 else 0.0
    downside = strategy_returns[strategy_returns < 0]
    downside_std = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    sortino = (mean / downside_std * np.sqrt(TRADING_DAYS)) if downside_std > 0 else 0.0

    years = len(prices) / TRADING_DAYS
    annualized = (
        ((equity[-1] ** (1.0 / years)) - 1.0) * 100.0
        if years > 0 and equity[-1] > 0
        else 0.0
    )

    trades = _extract_trades(position, prices, fee_rate)
    wins = [t for t in trades if t.return_pct > 0]
    losses = [t for t in trades if t.return_pct <= 0]
    gross_profit = sum(t.return_pct for t in wins)
    gross_loss = abs(sum(t.return_pct for t in losses))

    return BacktestResult(
        symbol=symbol or "",
        total_return_pct=float(total_return),
        max_drawdown_pct=max_drawdown,
        sharpe_ratio=float(sharpe),
        num_trades=len(trades),
        buy_and_hold_return_pct=float(buy_and_hold),
        win_rate_pct=float(len(wins) / len(trades) * 100.0) if trades else 0.0,
        time_in_market_pct=float(position.mean() * 100.0),
        annualized_return_pct=float(annualized),
        volatility_pct=float(std * np.sqrt(TRADING_DAYS) * 100.0),
        sortino_ratio=float(sortino),
        profit_factor=float(gross_profit / gross_loss) if gross_loss > 0 else float("inf"),
        equity_curve=equity_curve,
        trades=trades,
    )
