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
pytest -v
```
`black` (without `--check`) and `ruff check --fix` will auto-fix most formatting/lint issues.

## Guidelines
- Keep changes minimal and focused; avoid unrelated renames or file moves.
- Add or update tests for any behavior change in `src/`.
- Don't commit datasets, model checkpoints, or generated images — `data/` and
  `outputs/` are gitignored on purpose.
- Match the existing code style (type hints, docstrings on public functions).
