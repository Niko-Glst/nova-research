# nova-research

An equity research pipeline that turns a ticker into a probabilistic assessment:
four independent analysts, a falsifiable investment thesis, and a distribution of
outcomes rather than a price target.

It is a **research tool, not financial advice**, and it places no orders. That
constraint is enforced in code, not just stated — see [Guardrails](#guardrails).

```powershell
python research.py NVDA --html --simulate
```

<sub>Python 3.12 · ~6,900 lines · 273 tests · 9 runtime dependencies</sub>

---

## What it does

Four analysts run independently, each producing a stance and the reasoning
behind it. Their verdicts are combined into one weighted score, and a leg that
cannot be computed contributes nothing rather than counting as neutral — absence
of data is not evidence of balance.

| Analyst | Weight | Source | Measures |
|---|---:|---|---|
| Fundamental | 1.0 | yfinance | EPS, margins, growth, leverage, **valuation against niche peers** |
| Technical | 0.8 | yfinance | Trend, Donchian breakout, volume confirmation, RSI, MACD |
| Insider | 0.6 | SEC Form 4 | Open-market buys and sells, separated from compensation |
| Sentiment | 0.4 | Finnhub, Reddit | Recency-weighted news score, coverage-volume change |

On top of that:

- **Investment thesis** — the conditions that would have to hold for a
  constructive view, each marked met / unmet / unknown with the current reading,
  plus what would break it.
- **SWOT** derived from the figures, where every line names the number it rests on.
- **Outcome probabilities** — P(return > X) and P(drawdown > X) over a one-year
  horizon, split by volatility regime.
- **Backtest and Monte Carlo** for a trading rule, with the realised result
  located within its own resampled distribution.
- **Self-contained HTML report** with inline-SVG charts. No CDN, no network
  calls at view time.

---

## Design decisions worth explaining

Most of what is interesting in this codebase is the reasoning behind a handful
of choices. Each of these was made against a plausible-looking alternative.

### A peer median is meaningless without the peers

A P/E of 36 says nothing until you know the industry trades at 18. Peer groups
are declared explicitly in [`config/allowed_sources.yaml`](config/allowed_sources.yaml)
and keyed on the provider's industry label, because no public API offers a
trustworthy peer list.

Two problems surfaced against live data and are handled:

- **Negative multiples are excluded on both sides.** A loss-making company has no
  P/E, and scoring `-8.99` as "below the median of 9.70" reads a loss as a
  discount. This was a real bug, caught on RIVN.
- **Industry labels are often wrong for comparison purposes.** The provider files
  Grindr and Duolingo alongside Salesforce and Workday. A consumer subscription
  app and a B2B seat-licence business share an industry code and nothing else, so
  those groups are curated.

### Insider selling is not the signal people think it is

Insiders sell for diversification, taxes, and scheduled 10b5-1 plans; they buy on
the open market for one reason. The asymmetry is the signal.

`insider_analyst` classifies each Form 4 filing and **excludes compensation
grants and option exercises from the score** — a "Stock Award(Grant)" at a price
of 0.00 is payroll, and counting it as a purchase makes every company look like
its executives are loading up. On a typical megacap that removes well over half
the filings — 14 of 23 on MSFT at the time of writing.

The provider leaves the transaction-type column empty and puts the real nature in
free text, so the classifier reads that instead.

### Recency and volume, not just direction

News sentiment averaged over undated headlines treats a story from three weeks
ago as evidence about today. Articles decay exponentially with a seven-day
half-life, and undated ones are discounted rather than dropped or assumed fresh.

**Coverage volume is tracked separately and never folded into the sentiment
score.** A sudden tripling of article count is a real event, but whether it is
good or bad news is exactly what the volume does not tell you. Reporting it as
bullish would be a guess dressed as a measurement.

### Distributions, not point estimates

A backtest produces one number: the path that happened. `backtests/outcome_distribution.py`
resamples the stock's own returns into thousands of alternative paths and reports
the resulting probabilities — split by volatility regime, because a single
blended distribution averages two different worlds together.

For NVDA, the probability of a drawdown beyond 50%:

| Regime | Probability |
|---|---:|
| Calm | 1% |
| Volatile | 36% |
| Blended | 10% |

The blended figure describes neither regime. The gap between the first two is the
honest measure of how much the answer depends on conditions.

Regime labels always appear next to the volatility they were split on, because
"calm" is relative: 46% annualised is ordinary for NVDA and extreme for a utility.

### Backtests that do not flatter themselves

Three choices, all of which make results look worse and all of which are correct:

1. **Signals are shifted one bar.** A signal computed from a close cannot be
   traded at that same close. This is the most common way a backtest lies.
2. **Fees are charged on both sides** of every trade.
3. **Buy-and-hold is always reported alongside.** A strategy that trails it is
   not a strategy.

Under those rules, none of the strategies implemented here beat buy-and-hold on
the symbols tested. That is recorded rather than tuned away.

### Dependencies were removed, not added

- **vectorbt** was the original backtesting choice. It installs but fails to
  import against plotly 7, and pulls numba and llvmlite, which routinely fail to
  build on current Python. A long-only vectorised simulation is a few dozen lines
  of numpy; the dependency bought little and cost portability.
- **pandas-ta** is unmaintained and breaks on numpy 2.x. RSI, MACD, SMA and
  Donchian channels are computed directly in pandas.
- **statsmodels / scipy** were not added for the regression work. OLS with
  standard errors, t and F p-values is implemented in
  [`backtests/regression.py`](backtests/regression.py) via the incomplete beta
  function, and **verified against published statistical tables to four decimal
  places** — not against a second implementation of my own, where a shared error
  would look like agreement.

---

## Validating the weights — and finding them unsupported

The analyst weights (1.0 / 0.8 / 0.6 / 0.4) were chosen by argument. 
[`backtests/signal_study.py`](backtests/signal_study.py) tests them by regressing
each stance on realised returns.

Across 24 symbols over a 63-day window:

| Signal | Correlation | p-value | Assigned weight |
|---|---:|---:|---:|
| Technical | 0.72 | 0.0001 | 0.8 |
| Fundamental | 0.45 | 0.028 | 1.0 |
| Insider | −0.33 | 0.15 | 0.6 |
| Sentiment | 0.02 | 0.94 | 0.4 |

Technical was the only signal to survive a multivariate fit. Sentiment was
indistinguishable from noise.

**The weights were not changed on this evidence**, for three reasons stated in
the module:

1. Stances are computed from *current* data and paired with *past* returns, so
   this measures association, not prediction.
2. 24 observations against 4 predictors is below the 10:1 rule of thumb;
   `is_underpowered` flags it automatically.
3. Technical correlates with recent return almost by construction — price above
   its 50-day average largely *is* a recent rise.

Retuning on that would be fitting noise. The finding is documented; the fix is
point-in-time snapshots, which the project does not yet store.

---

## Guardrails

Defined in [SECURITY.md](SECURITY.md) and enforced in code:

| Rule | Enforcement |
|---|---|
| No live order execution | `get_broker_sandbox_credentials()` rejects any URL that does not look like a paper/sandbox endpoint, even with valid credentials |
| No secrets in git | `.env` gitignored from commit 1; `detect-secrets` pre-commit hook plus a regex fallback hook |
| Allowlisted sources only | Every external fetch reads its permitted domains and subreddits from `config/allowed_sources.yaml`; an unlisted subreddit raises |
| Rate limiting | Shared per-provider limiters in `common/rate_limit.py`, thread-safe, so two agents hitting one provider respect a single pace |

The sandbox-URL guard has its own tests covering the live-endpoint cases, because
that is the rule most costly to get wrong.

---

## Engineering notes

**273 tests, ~2,000 lines of test code.** Statistical functions are checked
against published tables; indicator maths against synthetic series with known
properties. Several tests are explicit regressions for bugs that only appeared
against live data:

- A MACD histogram of +0.01 on a \$16 stock casting a full bullish vote while
  price sat below its 50-day average.
- A truncated news feed reporting a coverage "spike" for every symbol, because a
  provider cap meant there was never a baseline behind it.
- `analyze_volume` accepting a `now` argument but reading the wall clock for one
  of its calculations, making the result depend on when it ran. Two tests began
  failing four days after they were written; the bug was real.

**Graceful degradation.** Every external source is optional. Without a Finnhub
key the pipeline falls back to a shorter feed and says that volume tracking is
unavailable; without Reddit credentials that leg is reported as unavailable. The
other analysts still run, and confidence drops to reflect what is missing.

**Untrusted input is treated as untrusted.** Headline text and URLs come from an
external feed and are escaped before rendering; there are tests for both.

---

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

copy .env.example .env   # then fill in real values; .env is gitignored
pre-commit install       # enables secret scanning
```

All API keys are optional. The fundamental, technical and insider analysts need
no credentials at all.

| Variable | Needed for | Free tier |
|---|---|---|
| `FINNHUB_API_KEY` | Dated news, coverage volume | Yes — [finnhub.io](https://finnhub.io/register) |
| `REDDIT_CLIENT_ID` / `_SECRET` / `_USER_AGENT` | Reddit sentiment | Yes |
| `TIINGO_API_KEY` | Alternative news source | News is a **paid** add-on; a free key returns HTTP 403 here |

## Usage

```powershell
python research.py AAPL                      # terminal report
python research.py AAPL MSFT NVDA            # several symbols
python research.py NVDA --html               # HTML report, opens in browser
python research.py NVDA --html --simulate    # adds backtest, Monte Carlo, probabilities
python research.py TSLA --period 6mo --insider-days 90
python research.py AAPL --no-peers           # skip peer comparison (much faster)
python research.py AAPL --json > report.json

python scripts/run_signal_study.py --csv reports/signal_study.csv
```

## Layout

```
agents/
  fa_analyst.py         fundamentals + niche peer comparison
  ta_analyst.py         trend, Donchian, volume, RSI, MACD
  insider_analyst.py    Form 4 classification
  sentiment_analyst.py  finance-tuned lexicon, negation and stemming
  news_sources.py       provider abstraction with fallback chain
  news_signal.py        recency weighting, coverage-volume detection
  thesis.py             falsifiable conditions for a constructive view
  swot.py               SWOT derived from measured figures
  research_report.py    orchestration and weighting
  charts.py             inline-SVG charts
  html_report.py        self-contained HTML rendering
  strategy_agent.py     crossover, thesis-driven and Donchian breakout rules
backtests/
  backtest_engine.py    long-only vectorised simulation
  monte_carlo.py        block and iid bootstrap
  outcome_distribution.py  regime-split forward probabilities
  regression.py         OLS with t and F tests, no scipy
  signal_study.py       do the signals relate to returns at all?
common/rate_limit.py    shared per-provider limiters
config/settings.py      credentials and allowlist loading
```

## Not yet implemented

The scaffold defines several stages that remain unbuilt: `investor_board`
(archetype personas debating the inputs), `narrative_synth`, `podcast_summarizer`,
`risk_manager` and `allocation_decision`. They raise `NotImplementedError` rather
than returning plausible-looking defaults.

---

<sub>**Nikolay Gelshtein** · Research output only — not financial advice, and no
orders are placed. Figures are point-in-time reads from public data sources and
may be stale or wrong.</sub>
