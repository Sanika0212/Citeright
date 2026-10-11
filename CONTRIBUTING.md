# Contributing

Thanks for your interest in Citeright.

## Development setup

```bash
git clone https://github.com/Sanika0212/Citeright.git
cd Citeright
uv sync --extra models --extra bench --extra viz
```

## Checks

Every pull request must pass the same checks CI runs:

```bash
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
uv run mypy
uv run coverage run -m pytest && uv run coverage report --fail-under=85
```

## Ground rules

1. **Tests never touch the network or download weights.** Use the scripted sources in
   `tests/test_resolve_check.py`, recorded responses with `httpx.MockTransport` (see
   `tests/test_sources.py`) and the word-overlap checker.
2. **Be a polite API client.** New sources go through `citeright.sources.Http` (cache,
   spacing, retries). Never send a contact address unless the user set one.
3. **Numbers come from runs.** Every number in the README must be reproducible with
   `citeright bench` and `scripts/make_figures.py`, and
   [docs/benchmarks.md](docs/benchmarks.md) must list the exact commands.
4. **Every flag explains itself.** A new check must say what is wrong and what the
   record says instead, not just lower a score.
5. Use [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`,
   `docs:`, `test:`, `chore:`, `ci:`).

## License of contributions

By contributing, you agree that your contributions are licensed under the Apache
License 2.0, the license of this project.
