"""Minimal, checkpoint-faithful CPU encoders for auditable subset execution.

These implementations intentionally expose only the frozen encoder path used by
this repository. They are not replacements for the upstream training packages.
"""

from __future__ import annotations

import json
import math
import pickle
import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RuntimeResult:
    """One measured pretrained-versus-random representation run."""

    model_id: str
    cells: int
    sequence_length: int
    embedding_width: int
    parameter_count: int
    pretrained_wall_seconds: float
    random_wall_seconds: float
    peak_rss_mb: float
    finite: bool
    pretrained_variance: float
    random_variance: float
    geometry_correlation: float
    input_gene_coverage: float
    execution_status: str
    claim_scope: str


def _peak_rss_mb() -> float:
    usage = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return usage / 1024.0


def _geometry_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 3:
        return float("nan")
    left_norm = left / np.maximum(np.linalg.norm(left, axis=1, keepdims=True), 1e-12)
    right_norm = right / np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-12)
    tri = np.triu_indices(len(left), 1)
    left_similarity = (left_norm @ left_norm.T)[tri]
    right_similarity = (right_norm @ right_norm.T)[tri]
    if np.std(left_similarity) == 0.0 or np.std(right_similarity) == 0.0:
        return float("nan")
    return float(np.corrcoef(left_similarity, right_similarity)[0, 1])


def _torch() -> Any:
    import torch

    return torch


def _linear(x: Any, weight: Any, bias: Any | None = None) -> Any:
    torch = _torch()
    return torch.nn.functional.linear(x, weight, bias)


def _layer_norm(x: Any, weight: Any, bias: Any, eps: float) -> Any:
    torch = _torch()
    return torch.nn.functional.layer_norm(x, (x.shape[-1],), weight, bias, eps)


def _geneformer_hidden_states(tokens: Any, mask: Any, state: dict[str, Any]) -> Any:
    torch = _torch()
    functional = torch.nn.functional
    batch, length = tokens.shape
    positions = torch.arange(length, device=tokens.device).unsqueeze(0).expand(batch, length)
    token_types = torch.zeros_like(tokens)
    hidden = (
        state["bert.embeddings.word_embeddings.weight"][tokens]
        + state["bert.embeddings.position_embeddings.weight"][positions]
        + state["bert.embeddings.token_type_embeddings.weight"][token_types]
    )
    hidden = _layer_norm(
        hidden,
        state["bert.embeddings.LayerNorm.weight"],
        state["bert.embeddings.LayerNorm.bias"],
        1e-12,
    )
    heads = 4
    width = hidden.shape[-1]
    head_width = width // heads
    attention_mask = (~mask).unsqueeze(1).unsqueeze(2)
    for layer in range(6):
        prefix = f"bert.encoder.layer.{layer}"
        query = _linear(
            hidden,
            state[f"{prefix}.attention.self.query.weight"],
            state[f"{prefix}.attention.self.query.bias"],
        )
        key = _linear(
            hidden,
            state[f"{prefix}.attention.self.key.weight"],
            state[f"{prefix}.attention.self.key.bias"],
        )
        value = _linear(
            hidden,
            state[f"{prefix}.attention.self.value.weight"],
            state[f"{prefix}.attention.self.value.bias"],
        )
        query = query.view(batch, length, heads, head_width).transpose(1, 2)
        key = key.view(batch, length, heads, head_width).transpose(1, 2)
        value = value.view(batch, length, heads, head_width).transpose(1, 2)
        scores = query @ key.transpose(-2, -1) / math.sqrt(head_width)
        scores = scores.masked_fill(attention_mask, torch.finfo(scores.dtype).min)
        probabilities = functional.softmax(scores, dim=-1)
        context = probabilities @ value
        context = context.transpose(1, 2).contiguous().view(batch, length, width)
        attention_output = _linear(
            context,
            state[f"{prefix}.attention.output.dense.weight"],
            state[f"{prefix}.attention.output.dense.bias"],
        )
        hidden = _layer_norm(
            hidden + attention_output,
            state[f"{prefix}.attention.output.LayerNorm.weight"],
            state[f"{prefix}.attention.output.LayerNorm.bias"],
            1e-12,
        )
        intermediate = _linear(
            hidden,
            state[f"{prefix}.intermediate.dense.weight"],
            state[f"{prefix}.intermediate.dense.bias"],
        )
        intermediate = functional.relu(intermediate)
        layer_output = _linear(
            intermediate,
            state[f"{prefix}.output.dense.weight"],
            state[f"{prefix}.output.dense.bias"],
        )
        hidden = _layer_norm(
            hidden + layer_output,
            state[f"{prefix}.output.LayerNorm.weight"],
            state[f"{prefix}.output.LayerNorm.bias"],
            1e-12,
        )
    return hidden


def _geneformer_forward(tokens: Any, mask: Any, state: dict[str, Any]) -> Any:
    hidden = _geneformer_hidden_states(tokens, mask, state)
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _random_like_geneformer(state: dict[str, Any], seed: int) -> dict[str, Any]:
    torch = _torch()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    result: dict[str, Any] = {}
    for name, tensor in state.items():
        if not name.startswith("bert."):
            continue
        if name.endswith("LayerNorm.weight"):
            result[name] = torch.ones_like(tensor)
        elif name.endswith(".bias"):
            result[name] = torch.zeros_like(tensor)
        elif name.endswith("position_ids"):
            result[name] = tensor.clone()
        else:
            result[name] = torch.empty_like(tensor).normal_(0.0, 0.02, generator=generator)
    return result


def tokenize_geneformer(
    gene_symbols: list[str],
    counts: np.ndarray,
    symbol_to_ensembl: dict[str, str],
    token_dictionary: dict[str, int],
    gene_medians: dict[str, float],
    maximum_length: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Apply the V1 rank-value tokenisation to raw counts."""
    mapped: list[tuple[int, int, float]] = []
    for gene_index, symbol in enumerate(gene_symbols):
        ensembl = symbol_to_ensembl.get(symbol)
        if ensembl is None or ensembl not in token_dictionary or ensembl not in gene_medians:
            continue
        mapped.append((gene_index, token_dictionary[ensembl], float(gene_medians[ensembl])))
    if not mapped:
        raise ValueError("No genes map to the Geneformer vocabulary")
    column_indices = np.asarray([item[0] for item in mapped], dtype=int)
    token_ids = np.asarray([item[1] for item in mapped], dtype=np.int64)
    medians = np.asarray([item[2] for item in mapped], dtype=np.float64)
    output = np.zeros((len(counts), maximum_length), dtype=np.int64)
    mask = np.zeros_like(output, dtype=bool)
    for cell_index, row in enumerate(counts[:, column_indices]):
        total = float(counts[cell_index].sum())
        if total <= 0.0:
            continue
        scores = row.astype(np.float64) / total * 10_000.0 / np.maximum(medians, 1e-12)
        detected = np.flatnonzero(row > 0)
        if len(detected) == 0:
            continue
        order = detected[np.argsort(-scores[detected], kind="mergesort")[:maximum_length]]
        output[cell_index, : len(order)] = token_ids[order]
        mask[cell_index, : len(order)] = True
    coverage = len(mapped) / max(len(gene_symbols), 1)
    return output, mask, float(coverage)


def run_geneformer_subset(
    model_path: Path,
    gene_symbols: list[str],
    counts: np.ndarray,
    symbol_to_ensembl_path: Path,
    token_dictionary_path: Path,
    gene_median_path: Path,
    maximum_length: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, RuntimeResult]:
    """Execute pretrained and identically shaped random V1 encoders on CPU."""
    torch = _torch()
    from safetensors.torch import load_file

    with symbol_to_ensembl_path.open("rb") as handle:
        symbol_to_ensembl = pickle.load(handle)
    with token_dictionary_path.open("rb") as handle:
        token_dictionary = pickle.load(handle)
    with gene_median_path.open("rb") as handle:
        gene_medians = pickle.load(handle)
    token_array, mask_array, coverage = tokenize_geneformer(
        gene_symbols,
        counts,
        symbol_to_ensembl,
        token_dictionary,
        gene_medians,
        maximum_length,
    )
    state = load_file(str(model_path), device="cpu")
    random_state = _random_like_geneformer(state, seed)
    tokens = torch.from_numpy(token_array)
    mask = torch.from_numpy(mask_array)
    with torch.inference_mode():
        start = time.perf_counter()
        pretrained = _geneformer_forward(tokens, mask, state).cpu().numpy()
        pretrained_seconds = time.perf_counter() - start
        start = time.perf_counter()
        random = _geneformer_forward(tokens, mask, random_state).cpu().numpy()
        random_seconds = time.perf_counter() - start
    parameter_count = int(sum(t.numel() for name, t in state.items() if name.startswith("bert.")))
    result = RuntimeResult(
        model_id="geneformer_v1_10m",
        cells=len(counts),
        sequence_length=maximum_length,
        embedding_width=pretrained.shape[1],
        parameter_count=parameter_count,
        pretrained_wall_seconds=pretrained_seconds,
        random_wall_seconds=random_seconds,
        peak_rss_mb=_peak_rss_mb(),
        finite=bool(np.isfinite(pretrained).all() and np.isfinite(random).all()),
        pretrained_variance=float(np.var(pretrained)),
        random_variance=float(np.var(random)),
        geometry_correlation=_geometry_correlation(pretrained, random),
        input_gene_coverage=coverage,
        execution_status="EXECUTED_CPU_SUBSET",
        claim_scope="REPRESENTATION_SANITY_ONLY_NOT_TARGET_EVALUATION",
    )
    return pretrained, random, result


def _digitize_deterministic(values: np.ndarray, bins: int) -> np.ndarray:
    output = np.zeros_like(values, dtype=np.float32)
    nonzero = np.flatnonzero(values > 0)
    if len(nonzero) == 0:
        return output
    nonzero_values = values[nonzero]
    edges = np.quantile(nonzero_values, np.linspace(0, 1, bins - 1))
    output[nonzero] = np.digitize(nonzero_values, edges, right=True) + 1
    return output


def tokenize_scgpt(
    gene_symbols: list[str],
    counts: np.ndarray,
    vocab: dict[str, int],
    maximum_length: int,
    bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Create deterministic binned sequences with a leading CLS token."""
    mapped = [
        (index, vocab[symbol]) for index, symbol in enumerate(gene_symbols) if symbol in vocab
    ]
    if not mapped:
        raise ValueError("No genes map to the scGPT vocabulary")
    columns = np.asarray([item[0] for item in mapped], dtype=int)
    ids = np.asarray([item[1] for item in mapped], dtype=np.int64)
    pad_id = int(vocab["<pad>"])
    cls_id = int(vocab["<cls>"])
    token_output = np.full((len(counts), maximum_length), pad_id, dtype=np.int64)
    value_output = np.full((len(counts), maximum_length), -2.0, dtype=np.float32)
    mask = np.zeros((len(counts), maximum_length), dtype=bool)
    token_output[:, 0] = cls_id
    value_output[:, 0] = 0.0
    mask[:, 0] = True
    for cell_index, raw in enumerate(counts[:, columns]):
        total = float(raw.sum())
        if total <= 0.0:
            continue
        normalized = np.log1p(raw.astype(np.float64) / total * 10_000.0)
        binned = _digitize_deterministic(normalized, bins)
        detected = np.flatnonzero(raw > 0)
        order = detected[np.argsort(-normalized[detected], kind="mergesort")]
        order = order[: maximum_length - 1]
        token_output[cell_index, 1 : len(order) + 1] = ids[order]
        value_output[cell_index, 1 : len(order) + 1] = binned[order]
        mask[cell_index, 1 : len(order) + 1] = True
    return token_output, value_output, mask, len(mapped) / max(len(gene_symbols), 1)


def _scgpt_hidden_states(tokens: Any, values: Any, mask: Any, state: dict[str, Any]) -> Any:
    torch = _torch()
    functional = torch.nn.functional
    hidden = state["encoder.embedding.weight"][tokens]
    hidden = _layer_norm(
        hidden, state["encoder.enc_norm.weight"], state["encoder.enc_norm.bias"], 1e-5
    )
    value_hidden = _linear(
        values.unsqueeze(-1),
        state["value_encoder.linear1.weight"],
        state["value_encoder.linear1.bias"],
    )
    value_hidden = functional.relu(value_hidden)
    value_hidden = _linear(
        value_hidden, state["value_encoder.linear2.weight"], state["value_encoder.linear2.bias"]
    )
    value_hidden = _layer_norm(
        value_hidden, state["value_encoder.norm.weight"], state["value_encoder.norm.bias"], 1e-5
    )
    hidden = hidden + value_hidden
    batch, length, width = hidden.shape
    heads = 8
    head_width = width // heads
    padding_mask = (~mask).unsqueeze(1).unsqueeze(2)
    for layer in range(12):
        prefix = f"transformer_encoder.layers.{layer}"
        qkv = _linear(
            hidden,
            state[f"{prefix}.self_attn.Wqkv.weight"],
            state[f"{prefix}.self_attn.Wqkv.bias"],
        )
        query, key, value = qkv.chunk(3, dim=-1)
        query = query.view(batch, length, heads, head_width).transpose(1, 2)
        key = key.view(batch, length, heads, head_width).transpose(1, 2)
        value = value.view(batch, length, heads, head_width).transpose(1, 2)
        scores = query @ key.transpose(-2, -1) / math.sqrt(head_width)
        scores = scores.masked_fill(padding_mask, torch.finfo(scores.dtype).min)
        probabilities = functional.softmax(scores, dim=-1)
        context = probabilities @ value
        context = context.transpose(1, 2).contiguous().view(batch, length, width)
        attention_output = _linear(
            context,
            state[f"{prefix}.self_attn.out_proj.weight"],
            state[f"{prefix}.self_attn.out_proj.bias"],
        )
        hidden = _layer_norm(
            hidden + attention_output,
            state[f"{prefix}.norm1.weight"],
            state[f"{prefix}.norm1.bias"],
            1e-5,
        )
        feed_forward = _linear(
            hidden, state[f"{prefix}.linear1.weight"], state[f"{prefix}.linear1.bias"]
        )
        feed_forward = functional.relu(feed_forward)
        feed_forward = _linear(
            feed_forward, state[f"{prefix}.linear2.weight"], state[f"{prefix}.linear2.bias"]
        )
        hidden = _layer_norm(
            hidden + feed_forward,
            state[f"{prefix}.norm2.weight"],
            state[f"{prefix}.norm2.bias"],
            1e-5,
        )
    return hidden


def _scgpt_forward(tokens: Any, values: Any, mask: Any, state: dict[str, Any]) -> Any:
    return _scgpt_hidden_states(tokens, values, mask, state)[:, 0, :]


def _random_like_scgpt(state: dict[str, Any], seed: int) -> dict[str, Any]:
    torch = _torch()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    result: dict[str, Any] = {}
    prefixes = ("encoder.", "value_encoder.", "transformer_encoder.")
    for name, tensor in state.items():
        if not name.startswith(prefixes):
            continue
        if name.endswith("norm.weight") or ".norm1.weight" in name or ".norm2.weight" in name:
            result[name] = torch.ones_like(tensor)
        elif name.endswith(".bias"):
            result[name] = torch.zeros_like(tensor)
        else:
            result[name] = torch.empty_like(tensor).normal_(0.0, 0.02, generator=generator)
    return result


def run_scgpt_subset(
    model_path: Path,
    vocab_path: Path,
    args_path: Path,
    gene_symbols: list[str],
    counts: np.ndarray,
    maximum_length: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, RuntimeResult]:
    """Execute the frozen whole-human encoder and a shape-matched random control."""
    torch = _torch()
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    args = json.loads(args_path.read_text(encoding="utf-8"))
    tokens_np, values_np, mask_np, coverage = tokenize_scgpt(
        gene_symbols, counts, vocab, maximum_length, int(args["n_bins"])
    )
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    random_state = _random_like_scgpt(state, seed)
    tokens = torch.from_numpy(tokens_np)
    values = torch.from_numpy(values_np)
    mask = torch.from_numpy(mask_np)
    with torch.inference_mode():
        start = time.perf_counter()
        pretrained = _scgpt_forward(tokens, values, mask, state).cpu().numpy()
        pretrained_seconds = time.perf_counter() - start
        start = time.perf_counter()
        random = _scgpt_forward(tokens, values, mask, random_state).cpu().numpy()
        random_seconds = time.perf_counter() - start
    prefixes = ("encoder.", "value_encoder.", "transformer_encoder.")
    parameter_count = int(sum(t.numel() for name, t in state.items() if name.startswith(prefixes)))
    result = RuntimeResult(
        model_id="scgpt_whole_human",
        cells=len(counts),
        sequence_length=maximum_length,
        embedding_width=pretrained.shape[1],
        parameter_count=parameter_count,
        pretrained_wall_seconds=pretrained_seconds,
        random_wall_seconds=random_seconds,
        peak_rss_mb=_peak_rss_mb(),
        finite=bool(np.isfinite(pretrained).all() and np.isfinite(random).all()),
        pretrained_variance=float(np.var(pretrained)),
        random_variance=float(np.var(random)),
        geometry_correlation=_geometry_correlation(pretrained, random),
        input_gene_coverage=float(coverage),
        execution_status="EXECUTED_CPU_SUBSET",
        claim_scope="REPRESENTATION_SANITY_ONLY_NOT_TARGET_EVALUATION",
    )
    return pretrained, random, result


def tokenize_geneformer_contextual(
    gene_symbols: list[str],
    counts: np.ndarray,
    symbol_to_ensembl: dict[str, str],
    token_dictionary: dict[str, int],
    gene_medians: dict[str, float],
    maximum_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Tokenise raw counts and retain original gene-column indices per token position."""
    mapped: list[tuple[int, int, float]] = []
    for gene_index, symbol in enumerate(gene_symbols):
        ensembl = symbol_to_ensembl.get(symbol)
        if ensembl is None or ensembl not in token_dictionary or ensembl not in gene_medians:
            continue
        mapped.append((gene_index, token_dictionary[ensembl], float(gene_medians[ensembl])))
    if not mapped:
        raise ValueError("No genes map to the Geneformer vocabulary")
    column_indices = np.asarray([item[0] for item in mapped], dtype=int)
    token_ids = np.asarray([item[1] for item in mapped], dtype=np.int64)
    medians = np.asarray([item[2] for item in mapped], dtype=np.float64)
    token_output = np.zeros((len(counts), maximum_length), dtype=np.int64)
    gene_output = np.full((len(counts), maximum_length), -1, dtype=np.int32)
    mask = np.zeros_like(token_output, dtype=bool)
    for cell_index, row in enumerate(counts[:, column_indices]):
        total = float(counts[cell_index].sum())
        if total <= 0.0:
            continue
        scores = row.astype(np.float64) / total * 10_000.0 / np.maximum(medians, 1e-12)
        detected = np.flatnonzero(row > 0)
        if len(detected) == 0:
            continue
        order = detected[np.argsort(-scores[detected], kind="mergesort")[:maximum_length]]
        token_output[cell_index, : len(order)] = token_ids[order]
        gene_output[cell_index, : len(order)] = column_indices[order]
        mask[cell_index, : len(order)] = True
    coverage = len(mapped) / max(len(gene_symbols), 1)
    return token_output, mask, gene_output, float(coverage)


def tokenize_scgpt_contextual(
    gene_symbols: list[str],
    counts: np.ndarray,
    vocab: dict[str, int],
    maximum_length: int,
    bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Create scGPT input and retain original gene-column indices for non-CLS tokens."""
    mapped = [
        (index, vocab[symbol]) for index, symbol in enumerate(gene_symbols) if symbol in vocab
    ]
    if not mapped:
        raise ValueError("No genes map to the scGPT vocabulary")
    columns = np.asarray([item[0] for item in mapped], dtype=int)
    ids = np.asarray([item[1] for item in mapped], dtype=np.int64)
    pad_id = int(vocab["<pad>"])
    cls_id = int(vocab["<cls>"])
    token_output = np.full((len(counts), maximum_length), pad_id, dtype=np.int64)
    value_output = np.full((len(counts), maximum_length), -2.0, dtype=np.float32)
    gene_output = np.full((len(counts), maximum_length), -1, dtype=np.int32)
    mask = np.zeros((len(counts), maximum_length), dtype=bool)
    token_output[:, 0] = cls_id
    value_output[:, 0] = 0.0
    mask[:, 0] = True
    for cell_index, raw in enumerate(counts[:, columns]):
        total = float(raw.sum())
        if total <= 0.0:
            continue
        normalized = np.log1p(raw.astype(np.float64) / total * 10_000.0)
        binned = _digitize_deterministic(normalized, bins)
        detected = np.flatnonzero(raw > 0)
        order = detected[np.argsort(-normalized[detected], kind="mergesort")]
        order = order[: maximum_length - 1]
        token_output[cell_index, 1 : len(order) + 1] = ids[order]
        value_output[cell_index, 1 : len(order) + 1] = binned[order]
        gene_output[cell_index, 1 : len(order) + 1] = columns[order]
        mask[cell_index, 1 : len(order) + 1] = True
    coverage = len(mapped) / max(len(gene_symbols), 1)
    return token_output, value_output, mask, gene_output, float(coverage)


def load_geneformer_contextual_states(
    model_path: Path,
    *,
    seed: int,
    device: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load frozen Geneformer encoder weights and its shape-matched random control."""
    from safetensors.torch import load_file

    loaded = load_file(str(model_path), device="cpu")
    cpu_state = {name: tensor for name, tensor in loaded.items() if name.startswith("bert.")}
    cpu_random = _random_like_geneformer(cpu_state, seed)
    state = {name: tensor.to(device) for name, tensor in cpu_state.items()}
    random_state = {name: tensor.to(device) for name, tensor in cpu_random.items()}
    return state, random_state


def load_scgpt_contextual_states(
    model_path: Path,
    *,
    seed: int,
    device: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load frozen scGPT encoder weights and its shape-matched random control."""
    torch = _torch()
    prefixes = ("encoder.", "value_encoder.", "transformer_encoder.")
    loaded = torch.load(model_path, map_location="cpu", weights_only=True)
    cpu_state = {name: tensor for name, tensor in loaded.items() if name.startswith(prefixes)}
    cpu_random = _random_like_scgpt(cpu_state, seed)
    state = {name: tensor.to(device) for name, tensor in cpu_state.items()}
    random_state = {name: tensor.to(device) for name, tensor in cpu_random.items()}
    return state, random_state


def encode_geneformer_contextual_batch(
    tokens: np.ndarray,
    mask: np.ndarray,
    state: dict[str, Any],
    *,
    device: str,
) -> np.ndarray:
    """Return final token hidden states for one Geneformer batch."""
    torch = _torch()
    token_tensor = torch.from_numpy(tokens).to(device)
    mask_tensor = torch.from_numpy(mask).to(device)
    with torch.inference_mode():
        hidden = _geneformer_hidden_states(token_tensor, mask_tensor, state)
    return np.asarray(hidden.float().cpu().numpy(), dtype=np.float32)


def encode_scgpt_contextual_batch(
    tokens: np.ndarray,
    values: np.ndarray,
    mask: np.ndarray,
    state: dict[str, Any],
    *,
    device: str,
) -> np.ndarray:
    """Return final token hidden states for one scGPT batch."""
    torch = _torch()
    token_tensor = torch.from_numpy(tokens).to(device)
    value_tensor = torch.from_numpy(values).to(device)
    mask_tensor = torch.from_numpy(mask).to(device)
    with torch.inference_mode():
        hidden = _scgpt_hidden_states(token_tensor, value_tensor, mask_tensor, state)
    return np.asarray(hidden.float().cpu().numpy(), dtype=np.float32)
