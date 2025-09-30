# Contributing

Thanks for your interest! For small fixes, please open a pull request.

## Environment
- Python 3.10+
- Install runtime deps: `pip install -r requirements.txt`
- Dev tools (tests/lint): `pip install -r requirements-dev.txt`

## Code style
- Run `ruff check .` before committing.
- Keep functions impersonal and documented in English.

## Tests
- Add smoke tests for new CLIs (`--help` runs).
- Avoid shipping large data; use tiny toy graphs for tests.

## Branching
- `main`: stable.
- `dev`: incoming changes.
