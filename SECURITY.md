# Security Policy

nova-research is a **research/analysis tool**. It is not a live trading system and must
never become one without an explicit, separate request from the project owner.

## Non-negotiable rules

1. **No secrets in git, ever.** All credentials live in a local `.env` file (see
   `.env.example` for the expected keys). `.env` is gitignored from commit 1. Never hardcode
   an API key, token, password, or credential in code, comments, config, or example files.
2. **No live order execution.** No module in this repo places orders or talks to a broker:
   there is no broker code. Everything is simulation only. If a broker integration is ever
   added, it must use a paper-trading/sandbox endpoint, never a live-trading endpoint, unless
   explicitly and separately requested.
3. **Allowlisted external sources only.** Any module that fetches external data (Reddit,
   news APIs, browser automation) must read its list of permitted domains/subreddits/feeds
   from `config/allowed_sources.yaml`. No module scrapes arbitrary URLs.
4. **Rate limiting.** All external API calls must back off / sleep between requests so we
   never exceed a provider's quota or trigger abuse detection.
5. **Pre-commit secret scanning.** A plain git hook (`.git-hooks/pre-commit`, enabled with
   `git config core.hooksPath .git-hooks`) scans every commit. A commit that looks like it
   contains a secret is blocked, not warned.

## Two ignore files, two jobs

- `.gitignore` keeps sensitive/generated files out of version control.
- `.claudeignore` keeps the same sensitive files out of Claude Code's own session context.

Both must be kept in sync when new sensitive file patterns are introduced.

## If a secret leaks anyway

1. **Rotate immediately.** Go to the provider (Reddit app console, news API dashboard)
   and revoke/regenerate the leaked key before doing anything else.
2. **Contain.** If the repo has already been pushed anywhere, treat the key as
   compromised the moment it's rotated — don't wait to clean history first.
3. **Clean git history.** Use `git filter-repo` (preferred) or the BFG Repo-Cleaner to
   remove the secret from all commits, not just HEAD. A plain new commit that deletes the
   file is not sufficient — the secret remains in history.
4. **Force-push carefully**, and only after coordinating with anyone else who has a clone,
   since rewritten history invalidates their local branches.
5. **Re-audit.** Search the cleaned history for the key (`git log -p -S`) and confirm the
   old key no longer works at the provider.
