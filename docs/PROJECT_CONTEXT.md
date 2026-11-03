# nova-research: project context

A handoff document for anyone (person or agent) building a tool that sits next
to this project. It describes what nova-research does, how the pieces fit, the
conventions the code follows, and which building blocks to reuse rather than
rewrite. The next planned tool is a **portfolio allocation tool** (maximum
weight per asset given current holdings); section 7 describes the correlation
API built for it.

State as of 2026-10-25 (GitHub: `Niko-Glst/nova-research`). Everything since
the 2026-10-08 commit `3101bf3` (the correlation module, signal-weight
optimisation, scoring models 2.0.0 and 3.0.0, the hardening below) was
committed on 2026-10-25. The README is the public-facing write-up. The signal-weight
optimisation work of 2026-10-12 (`backtests/signal_weights.py`,
`backtests/walk_forward.py`, `optimize.py`) is described in the README and its
module docstrings only.

**Finalise mode (2026-10-25):** no new features. The project was audited and
hardened instead:
- every entry point survives text the console cannot print (`common/console.py`);
- caches and reports are written atomically (`common/files.py`), and a cut-short
  SEC cache is refetched, not trusted;
- the web server serialises file writes from concurrent runs and reports a
  failed write without failing the run;
- the pyarrow-dependent commands stop up front with the fix, and the migration
  rolls back a half-written store;
- the flows leg takes no stance from the market part alone (a ticker with no
  data had read POSITIVE +1.00 on the S&P 500's trend); it needs issuance or the
  stock's own volatility.

**Model 3.0.0 (owner, 2026-10-25):** breadth weight 0.8 -> 0.5 after its
backtest; a breadth regime of 0 (in a zone, no turn yet) is no call and leaves
the leg out instead of counting as neutral; plus the flows rule above. US
breadth and S&P-based flow parts still apply to non-US listings (owner's
choice). 2.0.0 runs stored every input and rescore under 3.0.0.

Live audit, 2026-10-25: AAPL (with simulation), JPM, IBRX, SU.PA, SPY and a
nonexistent ticker ran end to end, ~40-65 s each, with no errors and no NaN or
None in any output; the nonexistent ticker reads inconclusive at 0%.

Update 2026-10-25: scoring model **2.0.0** rebuilt the flows leg (net issuance from 10-K share counts with an SEC EDGAR
cross-check, Yang-Zhang stock volatility, weighted parts), a sixth leg
(**breadth**, RSP/SPY), a model version on every run, a long-format parquet
snapshot store, and `history.py` for rescoring, backfill and backtests.
Sections 2, 4 and 5 cover them.

---

## 1. What it is

Two tools that share data plumbing and conventions:

- **Research** (`research.py`, `agents/`, `backtests/`): the single-ticker
  pipeline described below.
- **Correlation** (`correlation.py`, `correlations/`): correlation and
  covariance estimation across a universe, regime and tail analysis, a
  walk-forward test of which estimator forecasts best, and the covariance API
  for the allocation tool (sections 5 and 7).

The research tool is a single-ticker equity research pipeline. Given a symbol, six independent
analysts each produce a stance and the reasons behind it; the stances are
combined into one weighted score with a confidence figure. On top of that it
builds a falsifiable investment thesis, a figure-backed SWOT, regime-split
outcome probabilities, an optional backtest with Monte Carlo, and a
self-contained HTML report. Every report is archived so runs on the same ticker
can be compared over time.

It is a **research tool only**. It places no orders, and that is enforced in
code (see section 8).

- Owner: Nikolay Gelshtein (solo project).
- Language: Python 3.12 (`requires-python >= 3.10`), Windows, venv at `.venv/`.
- Runtime dependencies: `pandas`, `numpy`, `yfinance`, `praw`, `pyyaml`,
  `python-dotenv`. **Optional:** `scipy` (fast Kendall's tau, and reference
  checks in the tests), with a built-in fallback for everything it does;
  `pyarrow` (the parquet snapshot store; without it runs are recorded in CSV and
  the store-based steps say so). No statsmodels, matplotlib, vectorbt or
  pandas-ta. This is deliberate (section 8).
- 881 tests (`pytest`), all offline and each with its own empty signal history
  (`tests/conftest.py`): all pass with pyarrow installed; without it
  2 are skipped (2026-11-03).
- Model 4.0.0, currency fix (2026-11-02, `agents/currency.py`): Yahoo divides a
  market cap in the quote currency by revenue in the reporting currency without
  converting. Price/sales is now recomputed wherever the two differ, price/book
  for local listings (for US-listed foreign shares it stays Yahoo's, with a
  note; TSMC's looks wrong). P/E is right as given: EPS is in the quote currency.
  The mega-cap test runs in dollars (Toyota's yen no longer pass $500B). Amounts
  show their own currency. Tel Aviv has a benchmark (TA-125).
- Repo audit (2026-11-02, owner's go): removed the three stub agents, the broker
  guard (no broker exists), `backtests/signal_study.py` and its script (stances
  of today against past returns, replaced by the point-in-time record), the unused
  pre-commit config (its baseline never existed; the plain hook does the scanning),
  and loose dead lines. Wired into every report: `risk_manager` and
  `allocation_decision` through `agents/risk_read.py`, peer correlations through
  `correlations.load_returns`. With `--simulate`: the `thesis_signals` and
  `donchian_breakout` rules beside the others. Now shown as well: insider trades one
  by one, simulated drawdowns, the stop of an open trade, and SWOT, backtests and
  odds in the text report. Scores are unchanged, so the model stays 4.0.0.
- Usage audit (2026-10-29, owner's go): removed 17 functions nothing called
  (and the 4 tests that only exercised them), the unused NEWS_API_KEY and
  LOG_LEVEL settings, the example.com `rss_feeds` placeholder, and the empty
  `notebooks/`, `data/raw/`, `data/processed/` folders. Kept on purpose: the
  broker sandbox guard (SECURITY.md), the Kendall reference implementation the
  tests check the fast one against, and -- pending the owner's decision -- the
  unwired `risk_manager.assess` / `allocation_decision.size_position` and the
  three placeholder agents. The position-trading rule now builds its weekly
  swing state once per confirmation (identical output, about 10x faster).
- Backtest engine fix (2026-10-28): the trade list reported each trade at the
  close one day after the one the equity curve used, so trades did not add up to
  the total. Equity was right (signal traded at its own close, earning the moves
  after it, as in the walk-forward study); the trade list now matches it.
- News search by company name (2026-10-28): when Finnhub, Tiingo and the ticker's
  Yahoo feed are empty, `news_sources.fetch_yfinance_search_news` searches by name
  and keeps headlines Yahoo tags with the ticker or that name the company. Reddit
  searches `"<name>" OR <root>` when the root has 3+ letters (config
  `name_search`). BESI.AS: 0 -> 20 articles; Reddit still finds almost nothing in
  the three allowlisted (US-centred) subreddits.

---

## 2. Running it

```powershell
.venv\Scripts\python.exe research.py AAPL                        # terminal report
.venv\Scripts\python.exe research.py AAPL --html --simulate      # HTML + backtest + probabilities
.venv\Scripts\python.exe research.py SU.PA --html --no-open      # write without opening a browser
.venv\Scripts\python.exe research.py AAPL --json > report.json   # machine-readable
.venv\Scripts\python.exe serve.py                                # web UI on 127.0.0.1:8000
.venv\Scripts\python.exe -m pytest -q
```

Scoring history (model 2.0.0 onward):

```powershell
.venv\Scripts\python.exe history.py rescore                    # recorded runs under the current model
.venv\Scripts\python.exe history.py backfill                   # point-in-time flow inputs into the store
.venv\Scripts\python.exe history.py backtest                   # flow IC from the store
.venv\Scripts\python.exe history.py backtest --from-sources    # same, recomputed in memory (no store)
.venv\Scripts\python.exe history.py breadth-backtest           # RSP/SPY regime against what followed
.venv\Scripts\python.exe history.py implied-move-study         # S&P moves vs VIX/16 since 1990
.venv\Scripts\python.exe history.py settle                     # recorded verdicts vs local index
.venv\Scripts\python.exe scripts/migrate_snapshots_to_long.py  # one-time CSV -> parquet, round-trip checked
```

Backtests write `reports/flows_backtest.{csv,html}` and
`reports/breadth_backtest.{csv,html}`. They report; they never change a weight.

Other flags: `--period 6mo` (technical window, default `1y`),
`--insider-days 90` (default 180), `--no-peers` (skip peer comparison, much
faster), `--any-news-source` (disable the news allowlist).

A full run takes 1–3 minutes, almost all of it rate-limit sleeping (2 s between
calls per provider, plus one call per peer).

Use the same flags each time for runs you intend to compare. A run with
`--no-peers` and one without differ in the fundamental stance for reasons
unrelated to the company. So a run in which no peer comparison judged the price
(`--no-peers`, no peers known, every lookup failed, or the fundamental read
failed) is not added to the track record, in `serve.py` likewise.

When a company has no peer group of its own, `research.py` asks in the terminal
which tickers to compare it with: Enter keeps the sector list, or skips when there
is none. The answer is checked against Yahoo and saved per company in
`config/custom_peers.yaml`, which goes before `peer_groups`. No LLM is involved.
`--no-ask-peers` turns the question off for batch runs, and it is never asked when
input does not come from a terminal, or in `serve.py`.

Correlation analysis:

```powershell
.venv\Scripts\python.exe correlation.py --universe cross_asset --html --no-open
.venv\Scripts\python.exe correlation.py --universe global_multi_asset --base USD
.venv\Scripts\python.exe correlation.py --universe semiconductors --winsorize   # any peer group
.venv\Scripts\python.exe correlation.py --universe cross_asset --symbols SPY TLT GLD SU.PA
.venv\Scripts\python.exe correlation.py --universe cross_asset --json > corr.json
.venv\Scripts\python.exe correlation.py --pair AAPL MSFT --html               # one pair + forecast
```

Flags: `--base EUR|USD` (default EUR), `--frequency daily|weekly|both`,
`--start/--end YYYY-MM-DD`, `--winsorize`, `--refresh` (ignore the price
cache), `--no-evaluation` (skip the walk-forward). Reports go to
`correlation_reports/` (latest, dated archive, own index), separate from
`reports/`. A 30-asset run takes about 1.5 minutes with a cold cache (mostly
rate-limited fetching) and 15 s with a warm one.

---

## 3. Data flow

```
                    research_report.analyze(symbol)
                                 │
   ┌─────────────┬───────────────┼────────────────┬──────────────────┐
   ▼             ▼               ▼                ▼                  ▼
ta_analyst   fa_analyst     sentiment_analyst  insider_analyst   (--simulate)
yfinance     yfinance +     Reddit (praw) +    yfinance          strategy_agent →
OHLCV        peer group     news_sources →     Form 4 data       backtest_engine →
             from config    news_signal                          monte_carlo,
                                                                 outcome_distribution
   │             │               │                │                  │
   └─────────────┴───────┬───────┴────────────────┘                  │
                         ▼                                           │
          weighted score + verdict + confidence                      │
                         │                                           │
                 thesis.py, swot.py ◄────────────────────────────────┘
                         │
                 ResearchReport (dataclass)
                         │
     ┌───────────────────┼──────────────────────┐
     ▼                   ▼                      ▼
format_report()   html_report.render_report()   --json
 (terminal)              │
                 report_archive.save_report()
                 reports/<SYM>.html, reports/archive/<SYM>_<stamp>.html,
                 reports/index.html
```

Two more legs run after the insider leg, in the same way: `flow_analyst`
(yfinance prices and statements, SEC EDGAR companyfacts) and `breadth_analyst`
(RSP and SPY prices; it takes the sector and market cap from the fundamental
read). Their readings also become evidence lines (`flow_evidence`) listed under
the thesis and SWOT: numbers with their source and data flag, never a change to
the conclusions.

Every analyst runs inside its own `try`. A failing leg is recorded in
`ResearchReport.errors` and drops out of the score instead of failing the run.

---

## 4. Scoring

Each analyst's stance maps to a number: bullish or undervalued is **+1**,
neutral or fair is **0**, bearish or overvalued is **−1**. A stance of
`unknown` maps to nothing: that leg is excluded, not counted as 0.

| Analyst | Weight | Stances |
|---|---:|---|
| fundamental | 1.0 | undervalued / fair / overvalued |
| technical | 0.8 | bullish / neutral / bearish |
| insider | 0.6 | bullish / neutral / bearish / unknown |
| breadth | 0.5 | bullish / bearish / unknown (no call or no group) |
| sentiment | 0.4 | bullish / neutral / bearish / unknown |
| flows | 0.4 | bullish / neutral / bearish / unknown |

Weights live in `config/scoring.yaml` (`legs`), with the model version (3.0.0).

- `score = Σ(value × weight) / Σ(weight of legs that reported)`, in −1..+1.
- `confidence = Σ(weight of legs that reported) / 3.7` (2.8 for model 1.0.0, 3.2
  for 1.1.0 with flows, 4.0 for 2.0.0). Runs under different models are not
  compared: the reports index shows a version break instead of a change, and
  `history.py rescore` rescores 2.0.0+ runs from their stored inputs.
- Verdict: score ≥ +0.25 is `positive`, ≤ −0.25 is `negative`, anything between is
  `mixed`. If nothing reported, the verdict is `inconclusive`.

Because each leg contributes −1/0/+1, the score moves in steps. One stance flip
moves it by up to ±weight/3.7 (fundamental 0.27).

**Breadth leg** (owner's rule, 2026-10-25). Position of RSP/SPY in its 252-day
range. Bottom 20% and back above its 50-day average: favour breadth (+1); top
20% and back below it: favour the leaders (−1); in a zone without that turn: no
call, and the leg is left out of the verdict (owner, model 3.0.0; before, it
counted as neutral). Between the zones, the slope of a least-squares line through ln(ratio)
over 63 days decides (±1). Breadth sectors (Healthcare, Financial Services,
Consumer Defensive, Utilities, Industrials, Real Estate) score the regime;
leaders (Technology, Communication Services, Consumer Cyclical at $500B or more)
score its opposite; other stocks get no breadth score. The report shows a
1Y / 3M / 1M chart of the ratio (CSS tabs, no script). Weight 0.5 (0.8 until the
owner lowered it): the backtest (2004–2026, 269 month-ends) found no significant
edge (best: trend calls on the ratio 3 months ahead, 61% hits, t +1.6; zone calls
only 18 months).

**Flows leg, model 2.0.0.** Weighted mean of three parts, a missing part
reweighted away (but with neither stock part available the stance is
`unknown`: the market part alone does not speak for one stock): net issuance (ln shares at FYE t−1 minus t−2; issuance side
weighted 1.0, buyback side 0.5; ±2% is full scale), the S&P 500's
trend/volatility setting, and the stock's Yang-Zhang 21-day over 252-day
volatility ratio as a continuous `-clip(z, -1, 1)`. Buyback flags:
`reported`, `confirmed_zero` (10-K has no repurchase tag), `assumed_zero`,
`unknown` (no cash-flow line; share count still scores), `data_conflict`
(left out). Tax withholding never counts as a buyback.

The weights were chosen by argument, not fitted. A study (removed 2026-11-02, in
the git history) regressed stances on returns across 24 symbols. Technical correlated (r 0.72,
p 0.0001), fundamental weakly (0.45, p 0.03), and insider and sentiment showed
no significant relationship. The weights were left unchanged because stances
are current-data snapshots paired with past returns (association, not
prediction) and the sample is underpowered. Point-in-time testing now exists
for the flows and breadth legs (`history.py`); the older four legs still lack
it, because their inputs were never recorded historically.

---

## 5. Module reference

### Analysts (`agents/`)

| Module | Entry point | Returns | Notes |
|---|---|---|---|
| `ta_analyst` | `analyze(symbol, period="1y")` | `TechnicalRead(signal, indicators, reasons)` | SMA50/200, RSI(14, Wilder), MACD(12/26/9), Donchian(20, prior bars), volume ratio. Also exposes **`fetch_price_history(symbol, period)`** (section 7). |
| `fa_analyst` | `analyze(symbol, include_peers=True)` | `FundamentalRead(signal, metrics, reasons, peer_comparison, name)` | yfinance `.info`; ratios; peer comparison against a group from config, keyed on the yfinance industry label. Negative multiples excluded on both sides. `name` = company short name. |
| `sentiment_analyst` | `analyze(symbol, strict_allowlist=True, company_name="")` | `SentimentRead(...)` incl. `reddit_posts`, `news_articles` (every scored item with link), `reddit_query` | Finance-tuned lexicon with negation and stemming. Reddit: allowlisted subs, past month, upvote-weighted. Exchange-suffixed tickers (SU.PA) are searched by company name. |
| `news_sources` | `fetch_news(symbol, allowed_domains)` | `(list[NewsItem], unavailable)` | Fallback chain: Tiingo → Finnhub → yfinance. Finnhub is the one that works on a free key. |
| `news_signal` | `build_news_signal(items, score_fn)` | `NewsSignal` | 7-day half-life recency weighting; coverage volume tracked separately and never folded into the direction. |
| `flow_analyst` | `analyze(symbol)` | `FlowRead(signal, score, components, options, quarters, ...)` | Added 2026-10-21, rebuilt for model 2.0.0 (section 4). Each `FlowComponent` has a `score` (None = missing, reweighted away) and a `flag`. Issuance from SEC 10-K share counts first, yfinance second. Shown, not scored: options open interest and at-the-money implied vol (no dealer-gamma guess), quarterly buybacks. Real-money rebalancing deliberately left out. Thresholds in `config/scoring.yaml`, by judgement, not fitted. |
| `volatility` | `volatility_ratio(ohlc, short, long)` | `VolatilityRatio(ratio, z, se, estimator, ...)` | Yang-Zhang (α 1.34) with a close-to-close fallback when over 20% of rows have bad OHLC. Simulated false-signal rate under Gaussian returns: 2.2% / 0.0% per side (CC-21d: 10.3% / 4.4%). |
| `sec_edgar` | `cik_for`, `company_facts`, `latest_annual_report(facts, as_of)` | `AnnualReport(accn, filed, shares_end, shares_prior, repurchased, issued, net_issuance)` | SEC companyfacts, User-Agent from `SEC_USER_AGENT`, under 10 req/s, cached (`data/cache/sec`), point in time by filing date, original 10-K not amendments. |
| `breadth_analyst` | `analyze(symbol, sector=None, market_cap=None)` | `BreadthRead(signal, score, group, regime, reasons, history)` | RSP/SPY regime (section 4). `regime_from_ratio` is pure and point in time; `trend_line` fits the 63-day line. |
| `implied_move` | `analyze(now=None)` | `MarketMove(session, move, implied, multiple, tier, partial, next_day_implied, recent, base)` | Added 2026-10-25, shown not scored. The S&P 500's last move against VIX/sqrt(252) at the previous close ("VIX/16"). Tiers (owner): within 1x noise, 1x-2x notable, a drop beyond 2x "assumptions may have changed" (`assumptions_changed`; up-moves never set it). VIX1D shown alongside. Base rates from cached 1990+ history. `history.py implied-move-study` holds the evidence: moves beyond 1x on ~9% of down sessions (VIX overstates the move, realised ~86% of implied); after a drop beyond 2x, another big move within 5 sessions ~17% vs ~7%, and next-month realised vol up to VIX. No change in forward direction. |
| `position_strategy` | `run(symbol, history)`, `position_signals(ohlc)`, `levels(ohlc)` | `PositionBacktest` | Added 2026-10-28 (owner, after ThinkMarkets' position-trading guide). Pullback in an uptrend, entered only when all hold: close > 200-day and 50 > 200; close in the 50-61.8% retracement of the last weekly up-leg; support (weekly swing low or 100/200-day) within 2% below; Bollinger 20/2: a close at or below the lower band in the last 10 sessions and back inside now (owner's addition 2026-10-29, not in the guide); MACD over its signal or RSI back above 50 within 10 sessions of that setup, with the close not below the 61.8% level. Exit below the stop (leg's swing low, raised to each higher confirmed weekly swing low) or the 200-day. Weekly swings count only once their 2 confirming weeks have closed. In the report with --simulate, next to the crossover, with Monte Carlo, trades and how often each condition held. Very selective: META 1 trade in 5 years. |
| `benchmarks` | `benchmark_for(symbol)` | `str \| None` | Added 2026-10-28 (owner, after comparing with TradingAgents). The listing's local price index by Yahoo suffix, from `config/scoring.yaml` `benchmarks` (17 markets; Oslo left out, Yahoo serves one row). One-letter unconfigured suffixes (BRK.B) count as US. |
| `track_record` | `for_symbol(symbol)`, `settle(rows, prices)`, `summary(calls)` | `list[SettledCall]` | Added 2026-10-28. Each recorded verdict settled from the first close dated after the run's UTC day, against the local index, over 1W/1M/3M/6M sessions. Repeat runs with the same entry count once (last run wins). Positive is right when the stock beat its index, negative when it trailed, mixed takes no side. Shown in each report ("Past calls on this ticker") and by `history.py settle`; never scored. Price indices exclude dividends (stated). |
| `flow_evidence` | `evidence_lines(flows, reference)`, `breadth_line(breadth)`, `market_move_line(mm)` | `list[str]` | Inputs, not verdicts: they never change the thesis or SWOT conclusions. The issuance rank needs the SEC backfill in the store; without it the line says so. |
| `insider_analyst` | `analyze(symbol, lookback_days=180)` | `InsiderRead` | Form 4 via yfinance. Grants and option exercises are excluded; only open-market buys and sells count. **US filers only**: non-US names return `unknown`. |
| `thesis` | `build_thesis(...)` | `Thesis(conditions, breaks_if)` | "What must be true": each condition is met, unmet or unknown, with the current reading. Never a buy/sell instruction. |
| `swot` | `build_swot(...)` | `Swot` | Every line names the figure it rests on. |
| `research_report` | `analyze(symbol, period, include_peers, insider_lookback_days, strict_allowlist, run_simulation)` | `ResearchReport` | Orchestration and weighting; `format_report()` gives the terminal text. |

### Scoring history (`agents/`, `history.py`)

| Module | Entry point | Notes |
|---|---|---|
| `signal_snapshots` | `record(report)`, `load_rows()` | One row per run with every leg's stance and, from 2.0.0, the raw flow and breadth inputs, `model_version`, `schema_version`, `asof`. Goes to the parquet store once migrated, else to `data/snapshots/signals.csv` (old rows load with new columns empty). |
| `snapshot_store` | `append(long)`, `load(source=None)`, `active()` | Long format (date, ticker, field, value, source, filed_date, model_version, schema_version, recorded_at), append-only part files under `data/snapshots/long/`. Sources: `live_run`, `sec_pit`, `price_backfill`, `rescore`. Values stored as exact strings. |
| `rescore` | `rescore_row(row)` | Same scoring functions as a live run; a current-model run reproduces its score exactly. Pre-2.0.0 rows lack the inputs and are reported, not guessed. |
| `flow_history` | `build_panel`, `information_coefficients` | Month-end Spearman IC against 1/3/6-month excess return over SPY, quintile spread, Newey-West t (lag h−1), by part. |
| `breadth_history` | `build_panel`, `summarize` | Point-in-time regime per month-end against the ln RSP/SPY change and breadth ETFs (XLV, XLF, XLP, XLU, XLI) minus XLK; signed mean, hit rate, Newey-West t, and the same net of drift. |

### Simulation and statistics (`backtests/`)

| Module | Entry point | What it answers |
|---|---|---|
| `backtest_engine` | `run_backtest(price_history, entries, exits, fees_bps, symbol, shift_signals=True)` | Would a rule have worked? Long-only, signals lagged one bar, fees on both sides, buy-and-hold always reported alongside. |
| `monte_carlo` | `run_monte_carlo(symbol, returns, num_simulations, horizon_days, method="block", ...)` | How much of a backtest result was luck? Block bootstrap preserves volatility clustering. |
| `outcome_distribution` | `build_distribution(symbol, price_history, horizon_days=252, num_simulations=5000, seed=42)` | P(return > X), P(drawdown > X) over a horizon, **split by volatility regime** (calm vs volatile, split at the symbol's own 67th percentile of 21-day realised vol). |
| `regression` | **`correlate(x, y, name)`** | Pearson r with a t-based p-value. Implemented without scipy and checked against published tables to within 0.001. `SIGNIFICANCE_LEVEL = 0.05`. |

### Portfolio-side (`agents/`)

| Module | Entry point | Notes |
|---|---|---|
| `risk_manager` | `assess(symbol, candidate_returns, portfolio_returns=None, max_correlation=0.85, min_sharpe=0.0)` | **Correlation of a candidate to an equal-weighted portfolio aggregate** plus Sharpe/Sortino. Rejects with a stated reason. Also `compute_correlation()` and `compute_risk_adjusted_metrics()`. |
| `allocation_decision` | `size_position(symbol, conviction, risk_assessment, portfolio_equity, method=...)` | Fixed-fractional, volatility-target (default) or half-Kelly sizing, clipped and reduced for correlation to the book. Simulation only. |
| `risk_read` | `analyze(symbol, score, peers=())` | Both of the above in every report, shown and never scored. The local index stands in for the book; conviction is the score when positive. Returns from `correlations.load_returns`, local currency, 5 years (`risk_read` in `config/scoring.yaml`). |
| `strategy_agent` | `moving_average_crossover`, `donchian_breakout`, `thesis_signals` | Boolean entry/exit Series for the backtester. |

### Output

| Module | Purpose |
|---|---|
| `html_report.render_report(report, generated_at)` | One self-contained HTML file: inline CSS, inline SVG charts (`charts.py`), no CDN, no network calls at view time. External text is HTML-escaped. |
| `report_archive.save_report(report, directory="reports")` | Writes `reports/<SYM>.html` (latest), `reports/archive/<SYM>_<YYYY-MM-DD_HHMM>.html` (never overwritten), and rebuilds `reports/index.html` (every run per ticker: verdict, score, change vs previous run, confidence). The index is rebuilt by parsing the archived files, so it cannot drift from disk. |
| `serve.py` | stdlib `http.server` + Server-Sent Events progress. Binds to **127.0.0.1 only**. |

### Correlation analysis (`correlations/`, entry point `correlation.py`)

| Module | Entry point | What it does |
|---|---|---|
| `data` | `load_prices`, `align_prices`, `log_returns`, `convert_to_base`, `resolve_universe` | CSV price cache (`data/cache/prices/`, `CACHE_MAX_AGE_HOURS = 18`); currency by ticker suffix (unknown suffix is an error); prices inner-joined on dates every market traded *before* returns are taken; daily and weekly log returns; robust-z outlier flags; FX spike-and-reversal removal. |
| `fred` | `fetch_fred_series` | Fed H.10 noon FX rates (e.g. `DEXUSEU`), because Yahoo's EURUSD is dated one day late after ~2010. Allowlisted under `fred`. |
| `estimators` | **`estimate_correlation(returns, method)`**, **`estimate_covariance(returns)`** | Pearson/Spearman/Kendall (pairwise p-values, BH), EWMA, rolling, Ledoit-Wolf, RMT denoised, PCA and market + group factor models. Every result is a frozen `CorrelationEstimate` with PSD check, parameters and warnings. `estimate_covariance` is the allocation entry point (section 7). |
| `stats` | `kendall_matrix`, `benjamini_hochberg`, `check_psd`, `scipy_status` | Kendall uses scipy when it loads, a built-in O(n²) BLAS version otherwise (agree to 1e-12). |
| `shrinkage`, `rmt`, `factors`, `clustering` | `ledoit_wolf`, `denoise`, `pca_factor_model`, `market_group_factor_model`, `cluster` | Hand-rolled; Ledoit-Wolf matches scikit-learn and PyPortfolioOpt to 16 digits. |
| `regimes`, `synchronicity` | `analyze_regimes`, `lagged_correlations`, `epps_flags` | Calm/volatile, VIX > 30, named stress windows, down-vs-up asymmetry (bootstrap), tail dependence vs a Gaussian benchmark; lag-1 and daily-vs-weekly checks for cross-continent pairs. |
| `evaluation` | **`run_walk_forward(returns, market, groups)`**, `min_variance(cov, long_only)` | Monthly walk-forward: each estimator forecasts from the trailing 252 days and is scored on the next month by (a) Frobenius error vs the realised matrix and (b) realised volatility of the minimum-variance portfolio built from it (long-only by an exact active-set QP, and unconstrained). Diebold-Mariano with Newey-West vs the 252-day sample, BH across estimators. |
| `anchoring` | `anchor_path`, `anchored_correlation` | The `anchored` estimator: w x (last 252 days) + (1 - w) x (all history so far), one weight per matrix fitted by least squares on earlier outcomes known at the forecast date (no look-ahead). Falls back to the plain sample until 24 outcomes are known. Prefix sums: 50 ms for 30 assets x 20 years. |
| `pair` | **`analyze_pair(a, b, base)`**, `format_pair` | `correlation.py --pair A B`: every measure for one pair on its full common history, calendar-year correlations, stress figures, and an anchored forecast for the next month and 12 months with an 80% range from its own past out-of-sample errors. HTML via `html.save_pair` (`correlation_reports/pair_A_B.html`). |
| `analysis` | `run_analysis(universe, base)`, **`load_returns(symbols, base)`** | Orchestration, one frozen `CorrelationAnalysis`; `load_returns` is the data pipeline alone. |
| `report`, `findings`, `html`, `charts` | `format_analysis`, `to_jsonable`, `build_findings`, `html.save` | Terminal, JSON (no NaN), plain-language findings, HTML report in `correlation_reports/` with its own archive and index. |

**What the walk-forward found (2026-10-16, EUR view, 2007/08 to 2026):** in
both configured universes Ledoit-Wolf with the constant-correlation target had
the lowest forecast error (-1.3% to -1.4% vs the 252-day sample, significant
after BH) and tied the sample on the long-only minimum-variance portfolio. RMT
denoising, the PCA and market + group factor models were *significantly worse*
than the plain sample at forecasting, and EWMA (effective sample ~32 days) was
worst for portfolios. On 11 assets the unconstrained portfolio slightly
preferred the plain sample (+0.8% for Ledoit-Wolf). The differences among the
good estimators are small (1-2%); the large ones are which estimators to avoid.
This is why `DEFAULT_COVARIANCE_METHOD = "ledoit_wolf"`. In USD the long-only
portfolio is ~97% SHY, so that test is uninformative there and the report says
so.

**The anchored estimator (added 2026-10-16, same test, three universes in EUR):**
best forecaster on individual stocks (`us_mega_cap`: -2.2% error vs the
sample, significant; Ledoit-Wolf -0.5%, not significant), behind Ledoit-Wolf
on the ETF universes (-0.5% and -1.2%, not significant, against -1.4% and
-1.3%), best on the unconstrained portfolio in all three (significant in two,
where Ledoit-Wolf was once significantly *worse* than the sample), tied on the
long-only portfolio everywhere. It is the only estimator never significantly
worse than the benchmark on any metric. Its fitted weight on the last year is
0.6-0.8 at a one-month horizon. The default is still Ledoit-Wolf pending the
owner's decision; caveat for either choice: the estimators were compared on
the same data they are now chosen on, and the gaps are 1-2%.

### Infrastructure

- `config/allowed_sources.yaml`: subreddits, news domains, rate limits,
  `peer_groups` (keyed on normalised yfinance industry label),
  `default_peers_by_sector`, `max_peers: 6`.
- `config/custom_peers.yaml`: peers chosen per company, written by `research.py`
  or by hand; they go before `peer_groups`.
- `config/settings.py`: `load_allowed_sources()`, credential getters (raise if
  unset).
- `common/rate_limit.py`: `get_limiter("yfinance").wait()`. One shared,
  thread-safe limiter per provider name; interval from config (2 s).

---

## 6. Data sources and their quirks

| Source | Used for | Key needed | Quirks |
|---|---|---|---|
| yfinance | prices, fundamentals, insider (Form 4), fallback news | no | `history(auto_adjust=True)`. The index is **tz-aware in the exchange's own timezone**. `.info` fields can be missing; missing ≠ 0. `marketCap` is in the listing currency, but the report prints `$` regardless (known cosmetic bug). Forward EPS for some non-US names mixes IFRS reported and adjusted figures. |
| Finnhub | dated news, coverage volume | `FINNHUB_API_KEY` | Works on the free tier for US names; returned 403 for SU.PA. |
| Tiingo | news | `TIINGO_API_KEY` | News is a paid add-on; a free key returns 403. |
| Reddit (praw) | sentiment | `REDDIT_CLIENT_ID/SECRET/USER_AGENT` | Only r/wallstreetbets, r/investing, r/stocks. These rarely mention Europe-only names (one Schneider post in a month), and English-language EU subs returned nothing either. |
| SEC Form 4 | insider | none | US filers only. Non-US names (e.g. Schneider, reporting to the French AMF) have no insider data; OpenInsider has the same limitation. |
| FRED (Fed H.10) | FX for base-currency conversion | none | Noon New York rates, published weekly a few days in arrears, so the EUR view ends up to a week before the local one. Yahoo's `EURUSD=X` was rejected: dated one day late from ~2010 and with spike-and-reversal errors in 2008. |

Ticker conventions follow Yahoo: `AAPL`, `SU.PA` (Paris), `SIE.DE` (Xetra),
`ABBN.SW` (SIX), `LR.PA`. Peer groups mix currencies safely because only ratios
and percentages are compared.

---

## 7. Building blocks to reuse

### For the allocation tool: the correlation API

```python
from correlations import load_returns, estimate_covariance

data = load_returns(["SPY", "TLT", "SU.PA", "BESI.AS"], base="EUR")  # ReturnsData
risk = estimate_covariance(data.returns)                             # CovarianceEstimate
risk.covariance    # annualised DataFrame, EUR, from the last 252 complete days
risk.volatility    # annualised Series
risk.correlation   # the CorrelationEstimate behind it (method, PSD check, warnings)
risk.warnings      # read these: a NaN covariance must stop an allocation
data.convention    # the currency convention, to print next to any figure
```

- `covariance = D C D`: C from the method, D the sample volatilities over the
  same window. That is exactly what the walk-forward evaluated, so the default
  is backed by evidence rather than argument.
- Default method `ledoit_wolf` (constant-correlation target), window 252 days
  (`COVARIANCE_WINDOW`). Other methods: `COVARIANCE_METHODS`, including
  `anchored` (pass the full history: it uses every row for the long run, the
  last 252 for the recent part and the volatilities; it needs about three
  years before its weight is fitted). Spearman and Kendall are refused: they
  are not on the Pearson scale.
- Fewer than 252 rows: uses what there is and warns. Fewer than 60: NaN with a
  reason.
- `load_returns` runs the same pipeline as `correlation.py` (cache, FRED FX,
  calendar alignment), so the tool never re-implements it. `base=None` gives
  local currencies.
- Daily returns carry a non-synchronous-close bias for cross-continent pairs
  (e.g. SU.PA vs US names is understated daily). Weekly returns avoid it:
  `load_returns(..., frequency="weekly")` then `estimate_covariance(...,
  frequency="weekly", window=104)` (two years; the minimum is 60 weeks). The
  walk-forward evidence is for daily data only.
- `evaluation.min_variance(cov, long_only=True)` is an exact long-only
  minimum-variance solver (active set, verified against scipy SLSQP and the KKT
  conditions) the allocation tool can build on.

### Building blocks in the research tool

1. **Prices:** `agents.ta_analyst.fetch_price_history(symbol, period)` is
   rate-limited, validates the ticker, flattens MultiIndex columns and drops NaN
   closes. Returns OHLCV with a DatetimeIndex. Fetch symbols one at a time
   through it; the shared limiter keeps the pace polite even across threads.
2. **Significance:** `backtests.regression.correlate(x, y, name)` returns a
   `CorrelationResult(correlation, p_value, n_observations, is_significant)`.
   It handles NaN removal, constant series and |r| = 1. Use it for every pair
   instead of a bare `np.corrcoef`, so each number carries its p-value and n.
3. **Portfolio correlation:** `agents.risk_manager.compute_correlation(candidate_returns, portfolio_returns)`
   gives correlation to an equal-weighted aggregate (inner-joined dates,
   NaN when the overlap is under 3 or a series is constant). `assess()` gates at
   |r| > 0.85.
4. **Risk metrics:** `risk_manager.compute_risk_adjusted_metrics(returns)`
   gives Sharpe, Sortino, annualised return/vol, max drawdown and observations.
5. **Regimes:** `backtests.outcome_distribution.classify_regimes(...)` labels
   calm vs volatile periods per symbol. Correlations differ sharply between
   regimes (they rise in sell-offs), so a regime split is the natural next
   step.
6. **Peers:** `config/allowed_sources.yaml → peer_groups` already defines
   curated comparable sets (semiconductors, electrification, social media, …),
   which make natural universes to correlate.
7. **Output:** follow `html_report` / `report_archive`, i.e. one self-contained
   HTML file with inline SVG (`agents/charts.py` has the drawing helpers and
   style), plus a dated archive copy so runs can be compared.
8. **Rate limiting and config:** `common.rate_limit.get_limiter(provider)` and
   `config.settings.load_allowed_sources()`. Any new external source must be
   added to the allowlist first.

### Pitfalls that this codebase has already hit or guards against

- **Correlate returns, never price levels.** Two trending prices correlate
  strongly by construction. Use daily log or simple returns.
- **Align calendars before computing anything.** US, Paris and Zürich have
  different holidays, and yfinance indexes are tz-aware per exchange.
  Normalise to a date (`index.tz_localize(None).normalize()` or `.date`), then
  inner-join. Never forward-fill a closed market's price into a return of 0,
  because that biases correlation toward zero.
- **Asynchronous closes:** a European close is ~5.5 hours before the US close,
  so same-day cross-continent correlations are understated. Consider weekly
  returns, or report a lag-1 correlation alongside.
- **Currency:** SU.PA is in EUR, ABBN.SW in CHF, AAPL in USD. Local-currency
  returns mix FX moves into the correlation. Pick one convention (local, or
  converted to a base currency) and state it in the output.
- **Sample size and significance:** report n and the p-value with every r.
  The project's minimum for stable metrics is 60 observations
  (`risk_manager.MIN_OBSERVATIONS`).
- **Multiple comparisons:** a correlation matrix of N names tests N(N−1)/2
  pairs, so at p < 0.05 some will be "significant" by chance. Say so, or correct
  for it.
- **Stability:** a single full-period r hides change. Rolling-window
  correlation, or a calm/volatile regime split, shows whether the relationship
  holds.
- **NaN over a fake number:** when a value cannot be computed, return NaN or
  `unknown` with a reason, never a plausible default (section 8).

---

## 8. Conventions to keep

- **Research only.** No module places orders and there is no broker code.
  `serve.py` binds to localhost only.
- **Secrets:** `.env` only (gitignored), never in code or config. The hook in
  `.git-hooks/pre-commit` scans every commit. `.claudeignore` mirrors `.gitignore` for sensitive files.
- **Allowlisted sources only** (`config/allowed_sources.yaml`). Unlisted
  subreddits raise. New sources are added there first (SECURITY.md rule 3).
- **Rate-limit every external call** through `common/rate_limit.py`.
- **Absence of data is not evidence.** Missing legs are excluded, not counted
  as neutral. Missing metrics are omitted, not defaulted to 0. Uncomputable
  statistics return NaN.
- **Every judgement carries its reason.** Stances come with `reasons`, the risk
  gate rejects with a `reason` string, thesis conditions show their current
  reading, and SWOT lines name their figure.
- **Thresholds are module-level named constants**, with a comment on why.
- **Frozen dataclasses** for results. Optional/new fields get defaults so
  existing callers keep working.
- **Dependencies are removed, not added.** Indicators are computed in pandas;
  statistics are hand-rolled and verified against published tables.
  vectorbt and pandas-ta were rejected for portability. Justify any new
  dependency.
- **scipy is optional (owner's decision, 2026-10-16).** It is used where it is
  clearly better (Kendall's tau, ~500x faster) and as an independent reference
  in tests (`pytest.importorskip`), never as the only implementation: the code
  falls back to the built-in version and the report says so, with the install
  command. Reason for the fallback: Windows Smart App Control on the owner's PC
  blocks unrecognised compiled files intermittently ("DLL load failed ...
  beleid voor toepassingsbeheer"); scipy 1.18.1 loads. When that error
  appears, tell the owner the command to run rather than working around it.
- **Tests on synthetic series with known properties** (e.g. a constructed pair
  with a known correlation), plus explicit regression tests for bugs found
  against live data.
- **Untrusted text** (headlines, URLs, post titles) is HTML-escaped before
  rendering.
- **Docstrings explain why**, including the rejected alternative. Code style:
  line length 100, `from __future__ import annotations`, type hints.
- Generated output (`reports/`) is gitignored.

---

## 9. Known gaps

- No point-in-time snapshots of stances, so stance-vs-forward-return studies
  are not possible yet.
- Sentiment has no minimum sample: one Reddit post can set the leg to bullish.
- Insider data is US-only; no AMF/FCA/BaFin source.
- No portfolio or holdings store: `risk_read` uses the local index as the book.
- Correlation walk-forward: daily returns only; one 252-day window for the
  structured estimators (shrinkage might gain more on shorter windows, not
  tested); turnover is reported but no transaction costs are charged; the
  realised target from ~21 days is noisy (rankings unbiased, levels inflated).
- Correlation phase 4 (DCC-GARCH, Student-t copula) is not built; it needs the
  owner's OK.
- `risk_read` reads plain daily volatility and correlation; it does not use
  `estimate_covariance`.
