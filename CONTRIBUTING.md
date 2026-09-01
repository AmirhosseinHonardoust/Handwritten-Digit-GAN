# Contributing

Thanks for considering a contribution to Handwritten Digit GAN!

## Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

For an exact reproduction of the environment CI uses (Linux, pinned transitive
deps), install from `requirements-lock.txt` instead:
```bash
pip install -r requirements-lock.txt
```

## Before opening a PR
Run the same quality gate CI runs:
```bash
ruff check --select E,F,I,B,SIM,UP src/ tests/
black --check src/ tests/
mypy src/
pytest -v --cov=src --cov-report=term-missing
```
`black` (without `--check`) and `ruff check --fix` will auto-fix most formatting/lint issues.

Optionally, run `pip install pre-commit && pre-commit install` once to have
ruff/black/mypy run automatically on each commit (config in
`.pre-commit-config.yaml`).

The fast suite (default `pytest`) uses synthetic data and requires no network
access; it enforces 90%+ coverage of `src/`. A separate slow suite exercises
the real MNIST download path and is skipped by default — run it explicitly
with `pytest -v -m slow` (requires network; CI runs it automatically).

## Guidelines
- Keep changes minimal and focused; avoid unrelated renames or file moves.
- Add or update tests for any behavior change in `src/` — the fast suite
  should stay at 90%+ coverage.
- Don't commit datasets, model checkpoints, or generated images — `data/` and
  `outputs/` are gitignored on purpose.
- Match the existing code style (type hints, docstrings on public functions).
