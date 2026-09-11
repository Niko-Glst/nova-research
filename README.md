# nova-research

An investment **research and analysis pipeline**. It is a research tool, **not financial
advice**, and it **does not execute live trades**. Every strategy, backtest, and allocation
decision produced here is paper/simulation only — see [SECURITY.md](SECURITY.md) for the
guardrails that keep it that way.

## Setup

```powershell
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure credentials
copy .env.example .env
# then edit .env and fill in real values — never commit this file

# 4. (Optional but recommended) enable pre-commit secret scanning
pre-commit install

# Alternative to step 4 if you don't want the pre-commit framework: a plain
# fallback hook (regex-based secret scan) is included and can be enabled with:
# git config core.hooksPath .git-hooks
```

## Quick start

Analyze a ticker:

```powershell
.venv\Scripts\python.exe research.py AAPL
```

More options:

```powershell
python research.py AAPL MSFT NVDA        # several at once
python research.py TSLA --period 6mo     # different price window
python research.py TSLA --insider-days 90
python research.py AAPL --no-peers       # skip peer comparison (much faster)
python research.py AAPL --json > report.json
python research.py MP --html            # full HTML report + open in browser
python research.py MP --html --no-open  # write it, don't open
```

### HTML report

`--html` writes a self-contained page to `reports/<SYMBOL>.html` (gitignored)
and opens it. No CDN, no network calls at view time -- the charts are inline SVG.
It carries everything the terminal shows, plus:

- a **peer comparison chart**: each metric as a bar against the peer median, with
  green on the favourable side (which differs per metric -- cheap is good for a
  multiple, high is good for a margin);
- a **SWOT** derived from the figures, where every line names the number it rests
  on;
- the **investment thesis**: the conditions that would have to hold for a
  constructive view, each marked met / unmet / unknown, plus what would break it.

### On "should I buy this?"

The report deliberately does not answer that. It states what would have to be
true, and where each of those conditions currently stands -- which is the part
you can check against the next set of filings. See SECURITY.md: this is a
research tool, not advice.

A run produces one weighted verdict from four analysts:

| Analyst | Weight | Source | Needs credentials |
|---|---|---|---|
| Fundamental (+ peer comparison) | 1.0 | yfinance | no |
| Technical | 0.8 | yfinance | no |
| Insider activity | 0.6 | yfinance (Form 4) | no |
| Sentiment | 0.4 | Finnhub news, Reddit | both optional |

Only the sentiment leg needs keys, and it degrades rather than failing: without
`FINNHUB_API_KEY` it falls back to yfinance headlines (which cannot support
coverage-volume tracking), and without the Reddit keys that source is simply
reported as unavailable. The other three analysts need no credentials at all.

**A note on reading the output.** Insider *selling* is close to universal at
large caps (diversification, taxes, scheduled 10b5-1 plans), so most megacaps
read bearish on that leg. Open-market *buying* is the rare and more informative
event. Compensation grants and option exercises are excluded from the score for
the same reason — they are pay, not conviction.

### Peer comparison

`fa_analyst` benchmarks valuation against companies in the same niche, because a
P/E of 36 means nothing until you know what the rest of the industry trades at.
Peer groups are declared explicitly under `peer_groups` in
[config/allowed_sources.yaml](config/allowed_sources.yaml), keyed by yfinance's
`industry` field; unlisted industries fall back to a sector-level list and the
report says so. To add a niche, add a key there.

Negative multiples are excluded on both sides of the comparison: a loss-making
company has no P/E, and treating one as "below the median" would read a loss as
a discount.

## Backtesting and signal validation

```powershell
# Does a strategy beat buy-and-hold?
python -c "from agents.ta_analyst import fetch_price_history; from agents.strategy_agent import moving_average_crossover; from backtests.backtest_engine import run_backtest; h = fetch_price_history('AAPL', period='5y'); s = moving_average_crossover(h, 'AAPL'); print(run_backtest(h, s.entries, s.exits, fees_bps=10, symbol='AAPL').summary())"

# Do the analyst signals relate to realised returns at all?
python scripts/run_signal_study.py --csv reports/signal_study.csv
```

The backtest **shifts every signal one bar** before trading and charges fees on
both sides, and it always reports buy-and-hold next to the strategy. Those three
choices are what stop a backtest from flattering itself.

Monte Carlo resamples the strategy's own returns to show the distribution it
could plausibly have produced. The useful number is where the realised backtest
sits in that distribution: a result at the 95th percentile of its own resampling
was a favourable draw, not evidence of skill.

### What the signal study found

Over 24 symbols and a 63-day return window — **descriptive, not predictive**,
for the reasons the module documents:

| Signal | Correlation | p | Current weight |
|---|---|---|---|
| technical | 0.72 | 0.0001 | 0.8 |
| fundamental | 0.45 | 0.028 | 1.0 |
| insider | −0.33 | 0.15 | 0.6 |
| sentiment | 0.02 | 0.94 | 0.4 |

Technical is the only signal that survives a multivariate fit; sentiment is
indistinguishable from noise. **The weights have deliberately not been changed
on this basis** — 24 symbols at one moment in time, with signals computed today
and paired with past returns, is not grounds for retuning. A genuine test needs
point-in-time snapshots taken before the return period, which this project does
not yet store.

## Module flow

Data flows through the pipeline roughly in this order:

```
IMPLEMENTED                              NOT YET IMPLEMENTED
-----------                              -------------------
ta_analyst        --+
fa_analyst        --+                    investor_board --> narrative_synth
sentiment_analyst --+--> research_report                          |
insider_analyst   --+    (weighted verdict)                strategy_agent
                                                                  |
                                                                  v
                                                            risk_manager
                                                                  |
                                                                  v
                                                       backtest_engine /
                                                          monte_carlo
                                                                  |
                                                                  v
                                                      allocation_decision

podcast_summarizer --> (feeds narrative_synth, not yet implemented)
```

- **`agents/podcast_summarizer.py`** — turns audio (e.g. investing podcasts) into a
  transcript and summary.
- **`agents/sentiment_analyst.py`** — pulls sentiment from news and Reddit (allowlisted
  sources only, see `config/allowed_sources.yaml`).
- **`agents/ta_analyst.py`** / **`agents/fa_analyst.py`** — technical and fundamental
  analysis on price/financials data. `fa_analyst` also benchmarks valuation against
  niche peers (see [Peer comparison](#peer-comparison) above).
- **`agents/insider_analyst.py`** — Form 4 insider activity, separating open-market
  buys and sells from compensation grants and option exercises.
- **`agents/research_report.py`** — combines the four implemented analysts into one
  weighted verdict; `research.py` at the repo root is its CLI.
- **`agents/thesis.py`** — turns the figures into falsifiable conditions ("what must
  be true") and the list of things that would break them.
- **`agents/swot.py`** — a SWOT derived from measured figures, each entry carrying
  the metric behind it.
- **`agents/html_report.py`** — renders a report as a standalone HTML page with
  inline-SVG charts.
- **`agents/investor_board.py`** — a panel of fictional investor *archetypes* (not real
  individuals) that debate the inputs above.
- **`agents/narrative_synth.py`** — combines all upstream signals into a single investment
  thesis.
- **`agents/strategy_agent.py`** — turns the thesis into concrete entry/exit logic.
- **`agents/risk_manager.py`** — checks correlation and risk-adjusted return before
  anything is sized.
- **`backtests/backtest_engine.py`** / **`backtests/monte_carlo.py`** — historical and
  Monte Carlo simulation of the strategy, using [vectorbt](https://github.com/polakowo/vectorbt)
  (chosen over backtrader for its vectorized speed, which matters a lot for the repeated
  simulations Monte Carlo requires — see the comment in `requirements.txt`).
- **`agents/allocation_decision.py`** — final position sizing, including transaction costs.

## Project layout

See the directory tree in this repo — `agents/` (pipeline stages), `backtests/` (simulation),
`config/` (settings + source allowlist), `data/` (raw/processed, gitignored), `tests/`.
