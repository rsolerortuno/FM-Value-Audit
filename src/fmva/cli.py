"""Command-line interface for fm-value-audit."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fmva.adapters import ExternalNpzAdapter, RandomMLPAdapter
from fmva.adapters.base import RepresentationAdapter
from fmva.baselines import (
    ScoreOutput,
    canonical_feature_columns,
    run_random_mlp,
    run_random_ranking,
    run_scalar_baseline,
    run_supervised_baseline,
)
from fmva.benchmark import run_benchmark
from fmva.contextual import (
    ContextualAuditStatus,
    ContextualProtocol,
    validate_contextual_artifact,
    write_default_protocol,
)
from fmva.effort import EffortTracker
from fmva.evaluation import evaluate_frozen_methods
from fmva.ingest import add_network_centrality, validate_canonical
from fmva.io import read_json, read_table, write_json, write_table
from fmva.leakage import audit_overlap
from fmva.manifest import load_registry, validate_registry
from fmva.prospective import (
    GateStatus,
    validate_future_score_freeze,
    write_label_only_gate_report,
)
from fmva.reporting import write_report
from fmva.schemas import EffortRecord
from fmva.simulation import simulate_benchmark
from fmva.splits import build_random_split, build_temporal_split
from fmva.temporal import TemporalAuditStatus, validate_temporal_benchmark

app = typer.Typer(
    no_args_is_help=True,
    help="Audit decision-relevant value of representation methods for target prioritisation.",
)


@app.command()
def simulate(
    output: Annotated[Path, typer.Option("--output", help="Output CSV path.")],
    genes: Annotated[int, typer.Option("--genes", min=100)] = 1500,
    seed: Annotated[int, typer.Option("--seed")] = 20260729,
    cutoff_year: Annotated[int, typer.Option("--cutoff-year")] = 2019,
    evaluation_end_year: Annotated[int, typer.Option("--evaluation-end-year")] = 2024,
) -> None:
    """Generate the planted synthetic benchmark."""
    frame = simulate_benchmark(genes, seed, cutoff_year, evaluation_end_year)
    write_table(frame, output)
    typer.echo(f"Wrote {len(frame)} synthetic genes to {output}")


@app.command()
def ingest(
    input_csv: Annotated[Path, typer.Option("--input", help="Canonical input CSV.")],
    output: Annotated[Path, typer.Option("--output", help="Validated output CSV.")],
    network_edges: Annotated[
        Path | None,
        typer.Option("--network-edges", help="Optional two-column interaction edge CSV."),
    ] = None,
) -> None:
    """Validate a canonical table and optionally compute degree centrality."""
    frame = read_table(input_csv)
    if network_edges is not None:
        frame = add_network_centrality(frame, read_table(network_edges))
    validate_canonical(frame)
    write_table(frame, output)
    typer.echo(f"Validated {len(frame)} genes")


@app.command()
def split(
    data: Annotated[Path, typer.Option("--data")],
    output: Annotated[Path, typer.Option("--output")],
    kind: Annotated[str, typer.Option("--kind", help="temporal or random")] = "temporal",
    seed: Annotated[int, typer.Option("--seed")] = 20260729,
) -> None:
    """Materialise temporal or intentionally leaky random split assignments."""
    frame = read_table(data)
    assignments = pd.DataFrame({"gene_id": frame["gene_id"].astype(str)})
    if kind == "temporal":
        temporal = build_temporal_split(frame, seed)
        assignments["historical_train"] = 0
        assignments["historical_validation"] = 0
        assignments["prospective_evaluation"] = 0
        assignments.loc[temporal.historical_train, "historical_train"] = 1
        assignments.loc[temporal.historical_validation, "historical_validation"] = 1
        assignments.loc[temporal.evaluation, "prospective_evaluation"] = 1
    elif kind == "random":
        random = build_random_split(frame, seed)
        assignments["random_train"] = 0
        assignments["random_test"] = 0
        assignments.loc[random.train, "random_train"] = 1
        assignments.loc[random.test, "random_test"] = 1
    else:
        raise typer.BadParameter("kind must be temporal or random")
    write_table(assignments, output)


@app.command()
def leakage(
    manifest: Annotated[Path, typer.Option("--manifest")],
    model_id: Annotated[str, typer.Option("--model-id")],
    evaluation_ids: Annotated[Path, typer.Option("--evaluation-ids")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Audit exact overlap between documented pretraining and evaluation identifiers."""
    registry = load_registry(manifest)
    identifiers = [line.strip() for line in evaluation_ids.read_text(encoding="utf-8").splitlines()]
    audit = audit_overlap(registry.get(model_id), identifiers)
    write_json(audit.to_dict(), output)
    typer.echo(audit.status.value)


@app.command()
def baseline(
    data: Annotated[Path, typer.Option("--data")],
    method: Annotated[str, typer.Option("--method")],
    output: Annotated[Path, typer.Option("--output")],
    context: Annotated[str, typer.Option("--context")] = "id",
    seed: Annotated[int, typer.Option("--seed")] = 20260729,
) -> None:
    """Run one cheap baseline or random control and write scores plus effort."""
    frame = read_table(data)
    temporal = build_temporal_split(frame, seed)
    if method in {
        "popularity",
        "mean_expression",
        "differential_expression",
        "cell_type_specificity",
        "network_centrality",
        "druggable_class",
    }:
        result = run_scalar_baseline(frame, temporal, method, context, seed)
    elif method == "supervised":
        result = run_supervised_baseline(frame, temporal, context, seed)
    elif method == "random_mlp":
        result = run_random_mlp(frame, temporal, context, seed)
    elif method == "random_ranking":
        result = run_random_ranking(frame, temporal, seed)
    else:
        raise typer.BadParameter(f"Unsupported method: {method}")
    score_table = pd.DataFrame({"gene_id": frame["gene_id"], "score": result.scores})
    if result.probabilities is not None:
        score_table["probability"] = result.probabilities
    write_table(score_table, output)
    write_json(result.effort.model_dump(mode="json"), output.with_suffix(".effort.json"))


def _fit_embedding_head(
    embeddings: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, EffortRecord]:
    tracker = EffortTracker(
        method="embedding_head",
        seed=seed,
        configurations=["C=0.1", "C=1", "C=10"],
    )

    def make_model(c_value: float) -> Pipeline:
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=c_value,
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=seed,
                        solver="liblinear",
                    ),
                ),
            ]
        )

    def tune() -> float:
        best_c = 0.1
        best_value = -np.inf
        for c_value in [0.1, 1.0, 10.0]:
            model = make_model(c_value)
            model.fit(embeddings[train_indices], labels[train_indices])
            probabilities = model.predict_proba(embeddings[validation_indices])[:, 1]
            positives = labels[validation_indices] == 1
            ranks = np.argsort(-probabilities)
            value = float(np.flatnonzero(positives[ranks])[0]) * -1.0
            if value > best_value:
                best_c, best_value = c_value, value
        return best_c

    selected_c = float(tracker.run_tuning(tune))
    tracker.selected_configuration = f"C={selected_c:g}"

    def final() -> np.ndarray:
        model = make_model(selected_c)
        model.fit(embeddings, labels)
        return np.asarray(model.predict_proba(embeddings)[:, 1], dtype=np.float64)

    probabilities = np.asarray(tracker.run_final(final), dtype=float)
    return probabilities, tracker.to_record()


@app.command()
def embed(
    data: Annotated[Path, typer.Option("--data")],
    adapter: Annotated[str, typer.Option("--adapter", help="random-mlp or external-npz")],
    model_id: Annotated[str, typer.Option("--model-id")],
    output: Annotated[Path, typer.Option("--output")],
    embedding_path: Annotated[Path | None, typer.Option("--embedding-path")] = None,
    context: Annotated[str, typer.Option("--context")] = "id",
    subset: Annotated[int | None, typer.Option("--subset", min=20)] = None,
    seed: Annotated[int, typer.Option("--seed")] = 20260729,
) -> None:
    """Embed genes with a local adapter and fit a historical logistic head."""
    frame = read_table(data)
    if subset is not None:
        frame = frame.head(subset).copy()
    temporal = build_temporal_split(frame, seed)
    columns = canonical_feature_columns(context)
    representation: RepresentationAdapter
    if adapter == "random-mlp":
        representation = RandomMLPAdapter(model_id, seed, 16, 8)
    elif adapter == "external-npz":
        if embedding_path is None:
            raise typer.BadParameter("--embedding-path is required for external-npz")
        representation = ExternalNpzAdapter(model_id, embedding_path)
    else:
        raise typer.BadParameter("adapter must be random-mlp or external-npz")
    embeddings = representation.embed(frame, columns)
    probabilities, effort = _fit_embedding_head(
        embeddings,
        temporal.historical_labels,
        temporal.historical_train,
        temporal.historical_validation,
        seed,
    )
    write_table(
        pd.DataFrame(
            {
                "gene_id": frame["gene_id"].astype(str),
                "score": probabilities,
                "probability": probabilities,
            }
        ),
        output,
    )
    write_json(effort.model_dump(mode="json"), output.with_suffix(".effort.json"))


@app.command("reassemble-split")
def reassemble_split(
    manifest: Annotated[Path, typer.Option("--manifest")],
    parts: Annotated[Path, typer.Option("--parts")],
    output: Annotated[Path, typer.Option("--output")],
    audit_output: Annotated[Path | None, typer.Option("--audit-output")] = None,
) -> None:
    """Reassemble byte-split inputs and verify part and complete-file hashes."""
    from fmva.real_data import reassemble_split_file
    from fmva.real_data import write_json as write_real_json

    audit = reassemble_split_file(manifest, parts, output)
    destination = audit_output or output.with_suffix(output.suffix + ".audit.json")
    write_real_json(audit, destination)
    typer.echo(str(destination))


@app.command("real-preprocess")
def real_preprocess(
    gse115978_counts: Annotated[Path, typer.Option("--gse115978-counts")],
    gse115978_metadata: Annotated[Path, typer.Option("--gse115978-metadata")],
    gse120575_tpm: Annotated[Path, typer.Option("--gse120575-tpm")],
    gse120575_metadata: Annotated[Path, typer.Option("--gse120575-metadata")],
    geneformer_symbol_map: Annotated[Path, typer.Option("--geneformer-symbol-map")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Build sample-level melanoma features without touching target labels."""
    import pickle

    from fmva.real_data import (
        combine_real_gene_features,
        process_gse115978,
        process_gse120575,
    )
    from fmva.real_data import (
        write_csv as write_real_csv,
    )
    from fmva.real_data import (
        write_json as write_real_json,
    )

    features_115, summary_115 = process_gse115978(gse115978_counts, gse115978_metadata, output)
    allowed = set(features_115["gene_symbol"].astype(str))
    features_120, summary_120 = process_gse120575(
        gse120575_tpm, gse120575_metadata, output, allowed_genes=allowed
    )
    with geneformer_symbol_map.open("rb") as handle:
        symbol_map = pickle.load(handle)
    combined = combine_real_gene_features(features_115, features_120, symbol_map)
    write_real_csv(combined, output / "real_melanoma_gene_features.csv.gz")
    write_real_json(
        {
            "datasets": [summary_115.__dict__, summary_120.__dict__],
            "feature_universe": "GSE115978 genes; GSE120575 intersected to this universe",
            "combined_genes": len(combined),
            "mapped_ensembl": int(combined["gene_id"].notna().sum()),
            "gse115978_cells": summary_115.cells,
            "gse115978_genes": summary_115.genes,
            "gse115978_samples": summary_115.samples,
            "gse120575_cells": summary_120.cells,
            "gse120575_genes": summary_120.genes,
            "gse120575_patient_states": summary_120.samples,
            "gse120575_patients": summary_120.patients,
            "target_evaluation_status": "NOT_COMPUTED",
        },
        output / "real_data_summary.json",
    )
    typer.echo(str(output / "real_data_summary.json"))


@app.command("representation-smoke")
def representation_smoke(
    counts: Annotated[Path, typer.Option("--counts")],
    metadata: Annotated[Path, typer.Option("--metadata")],
    geneformer_model: Annotated[Path, typer.Option("--geneformer-model")],
    geneformer_symbol_map: Annotated[Path, typer.Option("--geneformer-symbol-map")],
    geneformer_tokens: Annotated[Path, typer.Option("--geneformer-tokens")],
    geneformer_medians: Annotated[Path, typer.Option("--geneformer-medians")],
    scgpt_model: Annotated[Path, typer.Option("--scgpt-model")],
    scgpt_vocab: Annotated[Path, typer.Option("--scgpt-vocab")],
    scgpt_args: Annotated[Path, typer.Option("--scgpt-args")],
    output: Annotated[Path, typer.Option("--output")],
    cells: Annotated[int, typer.Option("--cells", min=4)] = 12,
    sequence_length: Annotated[int, typer.Option("--sequence-length", min=16)] = 64,
    seed: Annotated[int, typer.Option("--seed")] = 20260729,
) -> None:
    """Run both real checkpoints and shape-matched random controls on one frozen subset."""
    from fmva.model_runtime import run_geneformer_subset, run_scgpt_subset
    from fmva.real_data import (
        choose_model_cells,
        load_selected_counts,
        read_gse115978_metadata,
    )
    from fmva.real_data import (
        write_json as write_real_json,
    )

    output.mkdir(parents=True, exist_ok=True)
    annotations = read_gse115978_metadata(metadata)
    selected = choose_model_cells(annotations, cells, seed)
    selected.to_csv(output / "model_subset_cells.csv", index=False)
    genes, selected_counts = load_selected_counts(counts, selected["cell_id"].tolist())
    pretrained_geneformer, random_geneformer, geneformer_result = run_geneformer_subset(
        geneformer_model,
        genes,
        selected_counts,
        geneformer_symbol_map,
        geneformer_tokens,
        geneformer_medians,
        sequence_length,
        seed,
    )
    np.savez_compressed(
        output / "geneformer_cpu_subset_embeddings.npz",
        cell_id=selected["cell_id"].to_numpy(),
        pretrained=pretrained_geneformer,
        random=random_geneformer,
    )
    pretrained_scgpt, random_scgpt, scgpt_result = run_scgpt_subset(
        scgpt_model,
        scgpt_vocab,
        scgpt_args,
        genes,
        selected_counts,
        sequence_length,
        seed,
    )
    np.savez_compressed(
        output / "scgpt_cpu_subset_embeddings.npz",
        cell_id=selected["cell_id"].to_numpy(),
        pretrained=pretrained_scgpt,
        random=random_scgpt,
    )
    write_real_json(
        {
            "seed": seed,
            "dataset": "GSE115978",
            "models": [geneformer_result.__dict__, scgpt_result.__dict__],
            "target_ranking_status": "NOT_COMPUTED",
            "leakage_status": "UNRESOLVED",
            "headline_permission": (
                "Representation execution only; no decision-relevant performance "
                "claim is permitted."
            ),
        },
        output / "model_cpu_subset_summary.json",
    )
    typer.echo(str(output / "model_cpu_subset_summary.json"))


@app.command("response-transfer-proxy")
def response_transfer_proxy(
    gse115978_features: Annotated[Path, typer.Option("--gse115978-features")],
    gse120575_features: Annotated[Path, typer.Option("--gse120575-features")],
    gse179994_pseudobulk: Annotated[Path, typer.Option("--gse179994-pseudobulk")],
    gse179994_gene_order: Annotated[Path, typer.Option("--gse179994-gene-order")],
    gse179994_sample_metadata: Annotated[Path, typer.Option("--gse179994-sample-metadata")],
    gse179994_supplement: Annotated[Path, typer.Option("--gse179994-supplement")],
    geneformer_model: Annotated[Path, typer.Option("--geneformer-model")],
    geneformer_symbol_map: Annotated[Path, typer.Option("--geneformer-symbol-map")],
    geneformer_tokens: Annotated[Path, typer.Option("--geneformer-tokens")],
    scgpt_model: Annotated[Path, typer.Option("--scgpt-model")],
    scgpt_vocab: Annotated[Path, typer.Option("--scgpt-vocab")],
    output: Annotated[Path, typer.Option("--output")],
    bootstrap: Annotated[int, typer.Option("--bootstrap", min=20)] = 300,
    seed: Annotated[int, typer.Option("--seed")] = 1729,
) -> None:
    """Run the melanoma-to-NSCLC response-transfer proxy, not clinical target evaluation."""
    from fmva.ood_proxy import run_response_transfer_proxy

    summary = run_response_transfer_proxy(
        gse115978_features=gse115978_features,
        gse120575_features=gse120575_features,
        gse179994_pseudobulk=gse179994_pseudobulk,
        gse179994_gene_order=gse179994_gene_order,
        gse179994_sample_metadata=gse179994_sample_metadata,
        gse179994_supplement=gse179994_supplement,
        geneformer_model=geneformer_model,
        geneformer_symbol_map=geneformer_symbol_map,
        geneformer_token_dictionary=geneformer_tokens,
        scgpt_model=scgpt_model,
        scgpt_vocab=scgpt_vocab,
        output_directory=output,
        bootstrap_iterations=bootstrap,
        seed=seed,
    )
    typer.echo(summary["status"])


@app.command()
def evaluate(
    data: Annotated[Path, typer.Option("--data")],
    scores: Annotated[Path, typer.Option("--scores")],
    model_id: Annotated[str, typer.Option("--model-id")],
    manifest: Annotated[Path, typer.Option("--manifest")],
    effort_json: Annotated[Path, typer.Option("--effort-json")],
    output: Annotated[Path, typer.Option("--output")],
    bootstrap: Annotated[int, typer.Option("--bootstrap", min=20)] = 400,
    seed: Annotated[int, typer.Option("--seed")] = 20260729,
) -> None:
    """Evaluate one precomputed score vector against the popularity control."""
    frame = read_table(data)
    score_frame = read_table(scores)
    merged = frame.merge(score_frame, on="gene_id", how="left", validate="one_to_one")
    if merged["score"].isna().any():
        raise typer.BadParameter("Scores do not cover every input gene")
    temporal = build_temporal_split(merged, seed)
    popularity = run_scalar_baseline(merged, temporal, "popularity", "id", seed)
    effort = EffortRecord.model_validate(read_json(effort_json))
    probabilities = (
        merged["probability"].to_numpy(dtype=float) if "probability" in merged.columns else None
    )
    external = ScoreOutput(
        scores=merged["score"].to_numpy(dtype=float),
        probabilities=probabilities,
        effort=effort,
        metadata={"model_id": model_id, "source": str(scores)},
    )
    registry = load_registry(manifest)
    model = registry.get(model_id)
    results, metadata = evaluate_frozen_methods(
        merged,
        temporal,
        {"popularity": popularity, model_id: external},
        bootstrap,
        seed,
        leakage_statuses={model_id: model.leakage_status},
    )
    output.mkdir(parents=True, exist_ok=True)
    write_json(
        {name: result.model_dump(mode="json") for name, result in results.items()},
        output / "evaluation.json",
    )
    write_json(metadata["claims"], output / "claims.json")


@app.command()
def benchmark(
    data: Annotated[Path, typer.Option("--data")],
    output: Annotated[Path, typer.Option("--output")],
    bootstrap: Annotated[int, typer.Option("--bootstrap", min=20)] = 400,
    seed: Annotated[int, typer.Option("--seed")] = 20260729,
    subset: Annotated[int | None, typer.Option("--subset", min=100)] = None,
) -> None:
    """Run all executable baselines, controls, metrics and ablations."""
    artifacts = run_benchmark(read_table(data), output, bootstrap, seed, subset)
    typer.echo(str(artifacts.summary_path))


@app.command()
def report(
    results: Annotated[Path, typer.Option("--results")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Render a Markdown report from a benchmark summary."""
    write_report(results, output)
    typer.echo(str(output))


@app.command("contextual-protocol-init")
def contextual_protocol_init_command(
    output: Annotated[Path, typer.Argument(help="Path for the frozen v0.5 protocol JSON.")],
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite an existing file."),
    ] = False,
) -> None:
    """Write the preregistered prospective contextual protocol."""
    if output.exists() and not force:
        raise typer.BadParameter(f"Refusing to overwrite existing protocol: {output}")
    protocol = write_default_protocol(output)
    typer.echo(protocol.protocol_id)


@app.command("contextual-validate")
def contextual_validate_command(
    root: Annotated[Path, typer.Argument(help="Contextual embedding export directory.")],
    protocol: Annotated[
        Path | None,
        typer.Option("--protocol", help="Optional frozen protocol JSON for hash linkage."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Optional JSON audit report path."),
    ] = None,
    fail_on_warnings: Annotated[
        bool,
        typer.Option("--fail-on-warnings", help="Return code 2 for warnings."),
    ] = False,
) -> None:
    """Validate a contextual pretrained/random artifact pair."""
    audit = validate_contextual_artifact(root, protocol_path=protocol)
    if output is not None:
        write_json(audit.model_dump(mode="json"), output)
    typer.echo(audit.status.value)
    if audit.status == ContextualAuditStatus.FAIL:
        raise typer.Exit(code=1)
    if fail_on_warnings and audit.status == ContextualAuditStatus.PASS_WITH_WARNINGS:
        raise typer.Exit(code=2)


@app.command("contextual-snapshot-gate")
def contextual_snapshot_gate_command(
    labels: Annotated[Path, typer.Argument(help="Label-only future snapshot table.")],
    protocol: Annotated[Path, typer.Option("--protocol", help="Frozen v0.5 protocol JSON.")],
    snapshot_date: Annotated[
        str, typer.Option("--snapshot-date", help="YYYY-MM-DD snapshot date.")
    ],
    output: Annotated[Path, typer.Option("--output", help="Gate report JSON path.")],
) -> None:
    """Evaluate the future power gate without accepting model scores."""
    report = write_label_only_gate_report(
        labels,
        protocol,
        output,
        snapshot_date=snapshot_date,
    )
    typer.echo(report.status.value)
    if report.status == GateStatus.FAIL:
        raise typer.Exit(code=1)
    if report.status == GateStatus.WAIT:
        raise typer.Exit(code=3)


@app.command("contextual-score-freeze-validate")
def contextual_score_freeze_validate_command(
    root: Annotated[Path, typer.Argument(help="Frozen future-score directory.")],
    protocol: Annotated[Path, typer.Option("--protocol", help="Frozen v0.5 protocol JSON.")],
) -> None:
    """Validate the hash-linked future score freeze without opening labels."""
    protocol_model = ContextualProtocol.model_validate_json(protocol.read_text(encoding="utf-8"))
    manifest = validate_future_score_freeze(root, protocol_model)
    typer.echo(f"PASS {manifest.rows} rows {len(manifest.methods)} methods")


@app.command("temporal-validate")
def temporal_validate_command(
    root: Annotated[Path, typer.Argument(help="Temporal benchmark or compact reference root.")],
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Optional JSON audit report path."),
    ] = None,
    fail_on_warnings: Annotated[
        bool,
        typer.Option("--fail-on-warnings", help="Return a non-zero exit code for warnings."),
    ] = False,
) -> None:
    """Validate Stage 2A-WIDE through Stage 3B manifests, seals and results."""
    audit = validate_temporal_benchmark(root)
    if output is not None:
        write_json(audit.model_dump(mode="json"), output)
    typer.echo(audit.status.value)
    if audit.status == TemporalAuditStatus.FAIL:
        raise typer.Exit(code=1)
    if fail_on_warnings and audit.status == TemporalAuditStatus.PASS_WITH_WARNINGS:
        raise typer.Exit(code=2)


@app.command("validate-manifest")
def validate_manifest_command(
    manifest: Annotated[Path, typer.Argument(help="Registry YAML path.")],
) -> None:
    """Validate model registry provenance and default-execution constraints."""
    registry = validate_registry(manifest)
    typer.echo(f"Validated {len(registry.models)} model entries")


if __name__ == "__main__":
    app()
