# Assumption ledger

| ID | Assumption | Risk if wrong | Status / mitigation |
|---|---|---|---|
| A01 | First clinical-development entry can be represented by one defensible year. | Temporal labels may be misordered or ambiguous. | Real ingestion requires source-specific rules and provenance; not exercised synthetically. |
| A02 | Features labelled pre-cutoff can be reconstructed without post-cutoff revisions. | Hidden temporal leakage. | Every real feature must carry an as-of date; synthetic fields are generated explicitly. |
| A03 | Genes are adequate ranking units despite isoforms, complexes and target pairs. | Misrepresents drug mechanisms. | Scope is gene-level only and documented as a limitation. |
| A04 | Genes without observed development by the evaluation end can serve as censored negatives. | False negatives depress or distort metrics. | Results describe known-label recovery, not biological truth. |
| A05 | Publication count is a useful study-intensity proxy. | Popularity control may understate non-public study bias. | Alternative proxies can be added; raw counts and source must be retained. |
| A06 | Exact identifier overlap is a meaningful lower bound on pretraining leakage. | Undetected semantic or derivative overlap. | Clean status is reserved for documented identifiers; incomplete documentation is UNRESOLVED. |
| A07 | Equal configuration counts are a useful minimum fairness constraint. | Methods differ greatly in per-configuration sophistication and compute. | Wall-clock, CPU and GPU compute are also published. |
| A08 | Gene bootstrap is informative for uncertainty. | Correlated genes make intervals too narrow. | Limitation is explicit; pathway/block bootstrap is a future extension. |
| A09 | Synthetic OOD perturbation tests software behaviour. | It may not resemble real tissue shift. | No biomedical transfer claim is permitted. |
| A10 | Logistic regression is an appropriately strong simple supervised baseline. | Nonlinear cheap methods could outperform it. | Registry and adapter design permit additional baselines; current scope is intentionally bounded. |
| A11 | Random MLP embeddings are a useful executable control. | They are not identically shaped to any unavailable foundation model. | Labelled as a substitute; true shape-matched controls remain NOT_COMPUTED. |
| A12 | Process CPU time approximates method compute in the sandbox. | Multithreading and system load reduce comparability. | Wall-clock and environment metadata are retained; no cross-hardware speed claim. |
| A13 | Three configurations constitute a matched tuning budget for the demonstration. | Budget may be too small for either method family. | Budget is a configurable protocol and the unmatched ablation exposes sensitivity. |
| A14 | The simulation prevalence is sufficient for bootstrap stability. | Too few positives cause undefined samples. | Generator validates minimum positives; bootstrap discards and counts invalid replicates. |
