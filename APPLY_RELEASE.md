# Publish FM Value Audit v0.5.0a1

This folder contains a complete GitHub-ready pre-release.

## Repository status

No connected GitHub repository named FM Value Audit was available during packaging. Create or connect the repository first, then copy the source tree or push the full source ZIP contents.

## Recommended publication

1. Use `fm-value-audit-v0.5.0a1-portfolio-source.zip` as the repository source.
2. Keep the tag `v0.5.0a1`.
3. Mark the GitHub Release as a **pre-release**.
4. Use `RELEASE_BODY.md` as the release description.
5. Attach the wheel, source distribution, Colab bundle, portfolio summary, figures, validation record and `release-manifest.json`.
6. Do not publish a final `v0.5.0` until contextual outputs and the new prospective evaluation are complete.

## Local verification

```bash
python -m pip install -e '.[dev]'
python -m compileall -q src tests
ruff check .
ruff format --check .
pytest --cov=fmva --cov-report=term-missing --cov-fail-under=85
fmva temporal-validate results/temporal_benchmark_v0.4
python -m build
```

## GitHub automation

The included `.github/workflows/release.yml` publishes the pre-release when tag `v0.5.0a1` is pushed. Strict mypy remains a visible non-blocking alpha audit.
