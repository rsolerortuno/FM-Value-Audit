# Validation record — FM Value Audit v0.5.0a1

Validation date: 2026-08-04  
Reference runtime: CPython 3.13.5 on Linux x86_64

## Release classification

`v0.5.0a1` is an execution-ready **GitHub pre-release** for the contextual and prospective workflow. It is not the final v0.5 scientific evaluation.

No post-cutoff clinical labels were opened and no contextual efficacy result is claimed.

## Engineering gates

| Gate | Status | Evidence |
|---|---|---|
| Frozen prospective protocol | PASS | `protocols/FMVA_CONTEXTUAL_PROSPECTIVE_v0.5.json` |
| Contextual protocol and artefact contracts | PASS | Unit, CLI and tamper tests |
| Label-only future power gate | PASS | Rejects score, rank, probability and metric columns |
| Future score freeze | PASS | Rejects label-like columns and records hashes |
| Notebook structure | PASS | 3 valid nbformat-4 notebooks; no stored outputs |
| Ruff lint | PASS | Ruff 0.16.0; `ruff check .` |
| Ruff format | PASS | Ruff 0.16.0; `ruff format --check .` |
| Python compilation | PASS | `python -m compileall -q src tests` |
| Functional suite | PASS | **70 tests passed** |
| Branch-aware coverage | PASS | **86.78%**; threshold 85% |
| Wheel build/import/CLI smoke | PASS | `fm_value_audit-0.5.0a1-py3-none-any.whl` |
| Temporal integration audit | PASS_WITH_WARNINGS | 26 pass, 1 inherited warning, 0 fail |
| Strict mypy | NOT_PASS_ALPHA_DEBT | NumPy generic typing and inherited return annotations remain |
| Geneformer contextual execution | NOT_RUN_REQUIRES_COLAB_GPU | Notebook 01 supplied |
| scGPT contextual execution | NOT_RUN_REQUIRES_COLAB_GPU | Notebook 02 supplied |
| Prospective future evaluation | NOT_DUE | First eligible snapshot is 2027-02-04 |
| GitHub Actions | NOT_RUN_LOCALLY | Workflow supplied; requires repository push |

## Commands executed

```bash
python -m compileall -q src tests
ruff check .
ruff format --check .
pytest -q
pytest --cov=fmva --cov-report=term-missing --cov-fail-under=85
python -m build
```

Observed output:

```text
70 passed
Total branch-aware coverage: 86.78%
Required coverage threshold: 85%
Ruff check: PASS
Ruff format: PASS
Wheel smoke: PASS
```

## Temporal scientific audit

The v0.4 reference bundle was validated across:

- strict Stage-2A power-gate contracts;
- pre-cutoff exclusion counts;
- gene-disjoint holdout sealing;
- configuration freeze before label access;
- score, label and configuration hashes;
- method-set agreement;
- final metric row counts;
- recomputation of the primary AUPRC difference;
- bootstrap interval and replicate count;
- conclusion-rule consistency;
- clinical-OOD abstention.

Result:

```text
26 PASS
1 WARNING
0 FAIL
PASS_WITH_WARNINGS
```

The warning is an inherited Stage-2B aggregate accounting discrepancy of 14 rows. Development, holdout, score and label hashes remain mutually consistent. The original scientific manifest is preserved rather than silently edited.

## Scientific result boundary

The final v0.4 primary comparison was:

```text
AUPRC(geneformer_static_plus_cheap) - AUPRC(cheap_historical)
```

Result:

```text
Point difference: -0.00032755486429918244
95% paired gene-bootstrap interval:
[-0.009484827468089183, 0.0013916328916157368]
Conclusion: NOT_SUPPORTED
```

Static checkpoint token geometry must not be described as a contextual cell embedding.

## Prospective v0.5 boundary

- Information cutoff: 2026-08-04.
- Outcome window starts: 2026-08-05.
- The opened v0.4 holdout is development knowledge only.
- Primary method: Geneformer contextual plus contemporaneous cheap baseline.
- Secondary method: scGPT contextual plus contemporaneous cheap baseline.
- Negative controls: shape-matched random Geneformer and scGPT encoders.
- Snapshot selection uses label counts only, never scores or metrics.
- First eligible snapshot: 2027-02-04.
- No future labels have been accessed.

## Known alpha debt

Strict mypy is not reported as passing. The recovered Python 3.13 and NumPy typing stack exposes unparameterised `np.ndarray` signatures and inherited return-type issues. Runtime tests, scientific contract tests, coverage, Ruff, compilation and package smoke tests remain clean.

This debt blocks beta promotion but does not change the frozen scientific protocol.
