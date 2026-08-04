# Contributing

Contributions must preserve the audit-first purpose of this repository.

1. Create a focused branch and add tests for scientific or behavioural changes.
2. Run `ruff check .`, `ruff format --check .`, `mypy --strict src`, and `pytest`.
3. Never add model weights, private data, fabricated benchmark values, or unverified provenance.
4. Any new model must have a registry entry with its exact checkpoint identifier, version,
   licence, parameter count, documented pretraining corpora, availability and leakage status.
5. Any quantitative README statement must be linked to a committed result artefact through the
   claim-marker convention tested by `fmva.truthfulness`.
6. Document changed assumptions in `docs/ASSUMPTIONS.md` and design choices in
   `docs/DECISIONS.md`.
