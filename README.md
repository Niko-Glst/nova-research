# nova-research

An equity research pipeline that turns a ticker into a probabilistic assessment:
six independent analysts, a falsifiable investment thesis, and a distribution of
outcomes rather than a price target.

It is a **research tool, not financial advice**, and it places no orders. That
constraint is enforced in code, not just stated — see [Guardrails](#guardrails).

```powershell
python serve.py              # web UI at localhost:8000
python research.py NVDA --html --simulate    # or straight from the CLI
```

<sub>Python 3.12 · ~22,400 lines · 879 tests · 8 runtime dependencies</sub>

---

## What it does

Six analysts run independently, each producing a stance and the reasoning
behind it. Their verdicts are combined into one weighted score, and a leg that
cannot be computed contributes nothing rather than counting as neutral — absence
of data is not evidence of balance.

| Analyst | Weight | Source | Measures |
|---|---:|---|---|
| Fundamental | 1.0 | yfinance | EPS, margins, growth, leverage, **valuation against niche peers** |
| Technical | 0.8 | yfinance | Trend, Donchian breakout, volume confirmation, RSI, MACD |
| Breadth | 0.5 | yfinance | Equal- over cap-weight S&P 500 (RSP/SPY): where it sits in its 1-year range, a turn through its 50-day average in the top or bottom 20%, otherwise its 3-month trend line. Breadth sectors score the regime, mega-cap leaders its opposite; in a zone without a turn it makes no call and is left out |
| Insider | 0.6 | SEC Form 4 | Open-market buys and sells, separated from compensation |
| Sentiment | 0.4 | Finnhub, Reddit | Recency-weighted news score, coverage-volume change |
| Flows | 0.4 | yfinance, SEC EDGAR | Net share issuance (Fama-French, from 10-K share counts, buyback flag checked against SEC); S&P 500 trend and volatility (systematic funds' setting); the stock's Yang-Zhang volatility ratio. Options open interest shown, not scored |

Each report also reads the S&P 500's last move against the move VIX implied
(VIX/16, the one-day move priced at the previous close). Within 1x is noise; a
drop beyond 2x says the market's assumptions may have changed. Shown, not
scored; `history.py implied-move-study` holds the 1990–2026 evidence.

Each report also lists the **past calls on this ticker**: every earlier verdict,
settled from the first close after its run against the listing's own index
(AEX for Amsterdam, CAC 40 for Paris, SPY for the US, 17 markets in all) over
1 week, 1, 3 and 6 months. `history.py settle` gives the record across all
tickers. Shown only: it never changes a score or a weight.

Each report also runs a **risk gate and position size**: Sharpe, Sortino, drawdown
and correlation against the local index and each peer, then the size three methods
allow, at the report's score and at full conviction. No holdings are stored, so the
index stands in for the book. Shown only.

With `--simulate`, a **position-trading rule** runs next to the 50/200
crossover: a pullback in an uptrend (after ThinkMarkets' position-trading
guide) taken only when every condition holds: uptrend on the 50/200-day,
price in the 50-61.8% Fibonacci retracement of the last weekly up-leg, support
within 2%, a Bollinger lower-band touch with the close back inside, and a MACD
or RSI momentum turn. Stops sit at weekly swing lows and
trail upward. Weekly swings are used only once confirmed, so the backtest never
sees a turning point before it was visible.

Listings with no news under their ticker (most non-US ones) are searched by
company name, keeping only headlines about the company; Reddit is searched by
quoted name plus the ticker root when it is distinctive (`"BE Semiconductor
Industries" OR BESI`).

Every report and recorded run carries the scoring model's version
(`config/scoring.yaml`, now 4.0.0), so runs under different rules are never
compared as if they were the same.

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
trustworthy peer list. A company without its own group gets a question in the
terminal, and the answer is kept in [`config/custom_peers.yaml`](config/custom_peers.yaml).

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
- **statsmodels / scipy** were not added for significance tests. The t-test is computed in
  [`backtests/regression.py`](backtests/regression.py) via the incomplete beta
  function, and checked against published tables.

---

## Validating the weights — and finding them unsupported

The first four analyst weights (1.0 / 0.8 / 0.6 / 0.4) were chosen by argument. A study
regressed each stance on realised returns. It paired today's stances with past returns, so
it was replaced by the point-in-time record (`history.py`, the track record) and removed;
it stays in the git history.

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
point-in-time testing, which `history.py` now does for the two newest legs
(month-ends, prices up to that day, 10-K facts from their filing date). Neither
earns a higher weight: the flow score's IC is about zero (only the issuance side
is near significance, t ≈ 1.7), and the breadth regime's best call (3-month
trend, 3 months ahead) hits 61% with t ≈ 1.6, so the owner lowered breadth from
0.8 to 0.5. Weights are set by the owner, never fitted to these reports.

---

## Guardrails

Defined in [SECURITY.md](SECURITY.md) and enforced in code:

| Rule | Enforcement |
|---|---|
| No live order execution | There is no broker code: nothing in the project can reach one |
| No secrets in git | `.env` gitignored from commit 1; a pre-commit hook blocks anything that looks like a key (`.git-hooks/pre-commit`) |
| Allowlisted sources only | Every external fetch reads its permitted domains and subreddits from `config/allowed_sources.yaml`; an unlisted subreddit raises |
| Rate limiting | Shared per-provider limiters in `common/rate_limit.py`, thread-safe, so two agents hitting one provider respect a single pace |

---

## Engineering notes

**879 tests, ~8,000 lines of test code, none of them online.** Statistical functions are checked
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
git config core.hooksPath .git-hooks   # enables secret scanning
```

All API keys are optional. The fundamental, technical and insider analysts need
no credentials at all.

| Variable | Needed for | Free tier |
|---|---|---|
| `FINNHUB_API_KEY` | Dated news, coverage volume | Yes — [finnhub.io](https://finnhub.io/register) |
| `REDDIT_CLIENT_ID` / `_SECRET` / `_USER_AGENT` | Reddit sentiment | Yes |
| `TIINGO_API_KEY` | Alternative news source | News is a **paid** add-on; a free key returns HTTP 403 here |
| `SEC_USER_AGENT` | SEC EDGAR share counts and buyback lines (flows leg) | Free; SEC asks for a name and contact email in the User-Agent |

## Usage

### Web UI

```powershell
python serve.py
python serve.py --port 9000 --no-open
```

Type a ticker, watch each analyst report in as it finishes, and land on the
report when it completes. Progress streams over Server-Sent Events, because a
full run takes one to three minutes — almost entirely rate-limit sleeping — and
a page that sits silent for that long reads as a hang.

Built on `http.server` from the standard library rather than Flask: the whole
server is one handler with four routes, and a framework would be more dependency
than code. **It binds to 127.0.0.1 only.** That is a deliberate constraint rather
than a default — the process holds API credentials and will fetch whatever it is
asked to, so it must not be reachable from the network. There is no
authentication because there is no remote access to authenticate.

### Command line

```powershell
python research.py AAPL                      # terminal report
python research.py AAPL MSFT NVDA            # several symbols
python research.py NVDA --html               # HTML report, opens in browser
python research.py NVDA --html --simulate    # adds backtest, Monte Carlo, probabilities
python research.py TSLA --period 6mo --insider-days 90
python research.py AAPL --no-peers           # skip peer comparison (much faster, not in the track record)
python research.py A B C --no-ask-peers      # never stop to ask for peers (batch runs)
python research.py AAPL --json > report.json

python history.py rescore                    # recorded runs under the current model
python history.py backfill                   # point-in-time flow inputs (needs pyarrow)
python history.py backtest --from-sources    # flow score IC and quintile spreads
python history.py breadth-backtest           # RSP/SPY regime against what followed
python history.py implied-move-study         # S&P moves against VIX/16 since 1990
python history.py settle                     # every recorded verdict vs its local index
python scripts/migrate_snapshots_to_long.py  # one-time: CSV snapshots to parquet

python correlation.py --universe cross_asset --html   # correlation report for a universe
python correlation.py --pair AAPL MSFT --html         # one pair, with a forecast
python optimize.py --universe us_large_caps           # signal weights, tested out of sample
```

Output that is redirected to a file never fails on characters the console
cannot show (an emoji in a headline prints as `?`), and every cache and report
file is written whole or not at all.

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
  flow_analyst.py       net issuance, systematic-fund conditions, stock volatility
  volatility.py         Yang-Zhang and close-to-close estimators, ratio z-score
  sec_edgar.py          SEC companyfacts client (cached, rate-limited)
  breadth_analyst.py    RSP/SPY regime and the stock's group
  implied_move.py       S&P 500 move against VIX/16 (shown, not scored)
  position_strategy.py  position-trading rule: MA trend, Fibonacci, weekly support, momentum
  benchmarks.py         each listing's local index, by ticker suffix
  currency.py           minor units, spot conversion, amounts in the right symbol
  track_record.py       recorded verdicts settled against the local index
  flow_evidence.py      flow and breadth readings as thesis/SWOT evidence lines
  research_report.py    orchestration and weighting
  signal_snapshots.py   one row per run, CSV or the parquet store
  snapshot_store.py     long-format, append-only parquet store
  rescore.py            rescore recorded runs under the current model
  flow_history.py       flow backfill and IC backtest
  breadth_history.py    breadth regime backtest
  charts.py             inline-SVG charts
  html_report.py        self-contained HTML rendering
  report_archive.py     latest and dated copies of every report, with an index
  strategy_agent.py     crossover, thesis-driven and Donchian breakout rules
  risk_manager.py       correlation to the book, Sharpe/Sortino, stated rejections
  allocation_decision.py  position sizing (simulation only)
  risk_read.py          both of those in every report, against the local index
backtests/
  backtest_engine.py    long-only vectorised simulation
  monte_carlo.py        block and iid bootstrap
  outcome_distribution.py  regime-split forward probabilities
  regression.py         Pearson correlation with a t-test, no scipy
  market_data.py        cached OHLCV and earnings history
  signals.py, signal_weights.py, walk_forward.py   weights fitted and tested out of sample
correlations/           correlation and covariance estimation, regimes, walk-forward
common/
  rate_limit.py         shared per-provider limiters
  files.py              atomic file writes for caches and reports
  console.py            console output that survives unprintable characters
config/settings.py      credentials and allowlist loading
config/scoring.yaml     model version, leg weights, every flows and breadth threshold
serve.py                local web UI, stdlib http.server + SSE
research.py             command-line entry point
history.py              rescore, backfill and backtests of the scoring model
correlation.py          correlation reports and the covariance API
optimize.py             signal-weight fitting, walk-forward
```

---

<sub>**Nikolay Gelshtein** · Research output only — not financial advice, and no
orders are placed. Figures are point-in-time reads from public data sources and
may be stale or wrong.</sub>
