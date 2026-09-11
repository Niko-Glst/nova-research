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
```

A run produces one weighted verdict from four analysts:

| Analyst | Weight | Source | Needs credentials |
|---|---|---|---|
| Fundamental (+ peer comparison) | 1.0 | yfinance | no |
| Technical | 0.8 | yfinance | no |
| Insider activity | 0.6 | yfinance (Form 4) | no |
| Sentiment | 0.4 | news headlines, Reddit | Reddit only |

Only the Reddit leg needs an API key. Without one it is reported as unavailable
and the other three still run — a missing optional source degrades the read
rather than breaking the pipeline.

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
