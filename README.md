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

## Module flow

Data flows through the pipeline roughly in this order:

```
podcast_summarizer  ──┐
sentiment_analyst   ──┤
ta_analyst           ─┼──► investor_board ──► narrative_synth ──► strategy_agent
fa_analyst           ──┘                                                │
                                                                         ▼
                                                                  risk_manager
                                                                         │
                                                                         ▼
                                                              backtest_engine /
                                                                monte_carlo
                                                                         │
                                                                         ▼
                                                             allocation_decision
```

- **`agents/podcast_summarizer.py`** — turns audio (e.g. investing podcasts) into a
  transcript and summary.
- **`agents/sentiment_analyst.py`** — pulls sentiment from news and Reddit (allowlisted
  sources only, see `config/allowed_sources.yaml`).
- **`agents/ta_analyst.py`** / **`agents/fa_analyst.py`** — technical and fundamental
  analysis on price/financials data.
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
