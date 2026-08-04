# Validation policy — FM Value Audit v0.5.0a1

## Authoritative evidence

The current engineering state is established by executable GitHub Actions workflows:

- [`CI`](.github/workflows/ci.yml) runs on every push to `main`, every pull request and manual dispatch.
- [`Publish v0.5.0a1 pre-release`](.github/workflows/release.yml) reruns the blocking gates before creating release assets.

A manually written Markdown statement, fixed Shields badge, PNG, screenshot or historical log is **not** accepted as proof that the current commit passes.

## Blocking CI gates

1. Package installation from the checked-out source.
2. Python compilation of `src/` and `tests/`.
3. The full pytest suite on Python 3.11, 3.12 and 3.13.
4. Branch-aware coverage of at least 85%.
5. Model-registry validation through the installed `fmva` CLI.
6. Sealed v0.4 temporal-benchmark audit through the installed CLI.
7. End-to-end synthetic simulation, benchmark and report generation.
8. Ruff lint.
9. Ruff formatting check.
10. Wheel and source-distribution build.
11. Installation and CLI smoke testing of the built wheel in a clean virtual environment.

## Evidence retained by GitHub Actions

Each test-matrix job uploads:

- `pytest.xml` — JUnit test results;
- `pytest.log` — full test output;
- `coverage.json` — machine-readable observed coverage;
- `coverage.xml` — Cobertura coverage report;
- `model-registry.log` — registry audit;
- `temporal-validation.log` — sealed temporal audit;
- `validation-summary.json` — commit-bound summary generated during the run.

The package job uploads the wheel and source distribution built from the same commit.

## Strict typing

Strict mypy is a blocking CI gate. The repository repair corrected the pandas Series generics, contextual key typing, metadata narrowing and NumPy return annotations that previously prevented a strict pass.

The corrected source tree reports:

```text
Success: no issues found in 44 source files
```

## Historical release measurement

The local release-preparation execution on 4 August 2026 reported:

```text
70 tests passed
86.78% branch-aware coverage
85% required threshold
Ruff check and formatting passed
Strict mypy passed: 44 source files
```

These values describe that execution only. The latest GitHub Actions run is authoritative for the current commit.

## Scientific audit boundary

The v0.4 temporal reference bundle is expected to return:

```text
26 PASS
1 WARNING
0 FAIL
PASS_WITH_WARNINGS
```

The warning is the preserved Stage-2B 14-row accounting discrepancy. It must not be silently rewritten. The sealed downstream row counts and hashes remain mutually consistent.

The frozen primary result remains:

```text
AUPRC(geneformer_static_plus_cheap) - AUPRC(cheap_historical)
point difference: -0.00032755486429918244
95% paired gene-bootstrap interval: [-0.009484827468089183, 0.0013916328916157368]
conclusion: NOT_SUPPORTED
```

The prospective contextual v0.5 evaluation is not yet computed, and no future post-cutoff labels have been opened.
