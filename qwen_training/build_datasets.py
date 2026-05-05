#!/usr/bin/env python3
"""Build and save train/test datasets from Kyudan/MathBridge with PCA-guided stratified sampling."""

from __future__ import annotations

import argparse
import os
import re
from typing import Any

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


SYSTEM_PROMPT = (
    "You are a LaTeX equation converter. "
    "Output ONLY the raw LaTeX expression with NO explanation, NO commentary, NO markdown, NO dollar signs. "
    "Single line only. Example input: 'x squared plus y squared' → Example output: x^2 + y^2"
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build train/test datasets from Kyudan/MathBridge with PCA-based stratified sampling."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3.5-4B",
        help="Tokenizer to use when formatting training samples into ChatML.",
    )
    parser.add_argument("--dataset_name", type=str, default="Kyudan/MathBridge")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument(
        "--cached_dataset_path",
        type=str,
        default=None,
        help="Path to a dataset already enriched by build_cache.py. "
        "If provided, skips downloading and complexity/PCA computation.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./datasets",
        help="Root directory where train_dataset/ and test_dataset/ will be saved.",
    )
    parser.add_argument(
        "--dataset_fraction",
        type=float,
        default=0.005,
        help="Fraction of the pool (after test split) to use for training. Ignored when --max_train_samples is set.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Hard cap on training rows. Overrides --dataset_fraction when provided.",
    )
    parser.add_argument(
        "--max_test_samples",
        type=int,
        default=300,
        help="Number of test rows to sample (default: 300).",
    )
    parser.add_argument(
        "--train_dataset_name",
        type=str,
        default="train_dataset",
        help="Folder name for the saved train dataset (default: train_dataset).",
    )
    parser.add_argument(
        "--test_dataset_name",
        type=str,
        default="test_dataset",
        help="Folder name for the saved test dataset (default: test_dataset).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face token. Falls back to HF_TOKEN env var.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def make_chatml_text(example: dict[str, Any], tokenizer: AutoTokenizer) -> str:
    spoken_english = normalize_text(example.get("spoken_English"))
    equation = normalize_text(example.get("equation"))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": spoken_english},
        {"role": "assistant", "content": equation},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


# ---------------------------------------------------------------------------
# PCA-based stratified sampling (identical logic to original)
# ---------------------------------------------------------------------------


def _latex_complexity_features(example: dict) -> dict:
    command_pattern = re.compile(r"\\[a-zA-Z]+")
    operator_pattern = re.compile(r"[+\-*/=<>]")
    expression_pattern = re.compile(
        r"\$\$.*?\$\$|\$.*?\$|\\\(.+?\\\)|\\\[.+?\\\]", re.DOTALL
    )

    def _max_brace_nesting_depth(text: str) -> int:
        depth = max_depth = 0
        for ch in text:
            if ch == "{":
                depth += 1
                max_depth = max(depth, max_depth)
            elif ch == "}" and depth > 0:
                depth -= 1
        return max_depth

    equation = normalize_text(example.get("equation")) or ""
    latex_total_length = len(equation)
    latex_total_braces = equation.count("{") + equation.count("}")
    latex_total_commands = len(command_pattern.findall(equation))
    latex_total_subscripts = equation.count("_")
    latex_total_superscripts = equation.count("^")
    latex_total_operators = len(operator_pattern.findall(equation))
    latex_total_nesting_depth = _max_brace_nesting_depth(equation)
    latex_num_expressions = len(expression_pattern.findall(equation)) or (
        1 if equation else 0
    )

    weighted_sum = (
        latex_total_length
        + latex_total_braces
        + (2 * latex_total_commands)
        + (2 * latex_total_operators)
        + (3 * latex_total_nesting_depth)
        + (2 * latex_total_subscripts)
        + (2 * latex_total_superscripts)
    )
    latex_avg_complexity_score = float(weighted_sum) / float(
        max(1, latex_num_expressions)
    )

    return {
        "latex_avg_complexity_score": latex_avg_complexity_score,
        "latex_num_expressions": int(latex_num_expressions),
        "latex_total_braces": int(latex_total_braces),
        "latex_total_commands": int(latex_total_commands),
        "latex_total_length": int(latex_total_length),
        "latex_total_nesting_depth": int(latex_total_nesting_depth),
        "latex_total_operators": int(latex_total_operators),
        "latex_total_subscripts": int(latex_total_subscripts),
        "latex_total_superscripts": int(latex_total_superscripts),
    }


def _build_quantile_bins(scores: np.ndarray, num_bins: int = 10) -> np.ndarray:
    if scores.size <= 1:
        return np.zeros(scores.size, dtype=np.int32)
    quantiles = np.linspace(0.0, 1.0, num_bins + 1)
    edges = np.unique(np.quantile(scores, quantiles))
    if edges.size <= 2:
        return np.zeros(scores.size, dtype=np.int32)
    return np.digitize(scores, edges[1:-1], right=True).astype(np.int32)


def _stratified_sample_indices(
    indices: np.ndarray,
    bins: np.ndarray,
    target_size: int,
    rng: np.random.Generator,
) -> list[int]:
    if target_size >= len(indices):
        sampled = indices.copy()
        rng.shuffle(sampled)
        return sampled.tolist()

    unique_bins, counts = np.unique(bins, return_counts=True)
    proportions = counts / counts.sum()
    raw_targets = proportions * target_size
    bin_targets = np.floor(raw_targets).astype(int)

    remainder = target_size - int(bin_targets.sum())
    if remainder > 0:
        for idx in np.argsort(-(raw_targets - bin_targets))[:remainder]:
            bin_targets[idx] += 1

    bin_targets = np.minimum(bin_targets, counts)
    missing = target_size - int(bin_targets.sum())
    if missing > 0:
        spare = counts - bin_targets
        for idx in np.argsort(-spare):
            if missing == 0:
                break
            add = min(spare[idx], missing)
            bin_targets[idx] += add
            missing -= add

    chosen: list[int] = []
    for bin_id, n_take in zip(unique_bins, bin_targets):
        if n_take <= 0:
            continue
        bin_indices = indices[bins == bin_id].copy()
        rng.shuffle(bin_indices)
        chosen.extend(bin_indices[:n_take].tolist())

    if len(chosen) < target_size:
        remaining = np.setdiff1d(
            indices, np.asarray(chosen, dtype=np.int64), assume_unique=False
        )
        rng.shuffle(remaining)
        chosen.extend(remaining[: target_size - len(chosen)].tolist())

    rng.shuffle(chosen)
    return chosen[:target_size]


# ---------------------------------------------------------------------------
# Main dataset builder
# ---------------------------------------------------------------------------


def build_datasets(args: argparse.Namespace, tokenizer: AutoTokenizer):
    hf_token = args.hf_token or os.getenv("HF_TOKEN")

    if args.cached_dataset_path:
        print(
            f"[1/8] Loading pre-enriched dataset from cache: {args.cached_dataset_path} ..."
        )
        from datasets import load_from_disk

        dataset = load_from_disk(args.cached_dataset_path)
        print(f"[1/8] Loaded {len(dataset):,} rows from cache.")
        required_cache_cols = {"spoken_English", "equation", "pca_component_1"}
        missing_cache = required_cache_cols - set(dataset.column_names)
        if missing_cache:
            raise ValueError(
                f"Cache is missing columns {missing_cache}. "
                "Re-run build_cache.py to regenerate it."
            )
        print("[2/8] Skipping complexity computation (already in cache).")
        print("[3/8] Skipping complexity computation (already in cache).")
        print("[4/8] Skipping PCA (already in cache).")
        pca_scores = np.asarray(dataset["pca_component_1"], dtype=np.float64)
    else:
        print(
            f"[1/8] Loading dataset '{args.dataset_name}' split='{args.dataset_split}'..."
        )
        dataset = load_dataset(
            args.dataset_name, split=args.dataset_split, token=hf_token
        )
        print(f"[1/8] Loaded {len(dataset):,} rows.")

        required_columns = {"spoken_English", "equation"}
        missing = required_columns - set(dataset.column_names)
        if missing:
            raise ValueError(
                f"Dataset missing required columns: {', '.join(sorted(missing))}"
            )

        print("[2/8] Computing LaTeX complexity indicators...")
        dataset = dataset.map(
            _latex_complexity_features, desc="Computing complexity indicators"
        )
        print(f"[3/8] Complexity indicators added. Rows: {len(dataset):,}")

        complexity_columns = [
            "latex_avg_complexity_score",
            "latex_num_expressions",
            "latex_total_braces",
            "latex_total_commands",
            "latex_total_nesting_depth",
            "latex_total_operators",
            "latex_total_subscripts",
            "latex_total_superscripts",
        ]
        complexity_matrix = np.column_stack(
            [np.asarray(dataset[col], dtype=np.float64) for col in complexity_columns]
        )
        col_mins = complexity_matrix.min(axis=0)
        col_maxs = complexity_matrix.max(axis=0)
        col_ranges = np.where((col_maxs - col_mins) == 0.0, 1.0, col_maxs - col_mins)
        normalized_matrix = (complexity_matrix - col_mins) / col_ranges

        if len(dataset) > 1:
            _, _, vh = np.linalg.svd(normalized_matrix, full_matrices=False)
            pca_scores = normalized_matrix @ vh[0]
        else:
            pca_scores = np.zeros(len(dataset), dtype=np.float64)

        dataset = dataset.add_column("pca_component_1", pca_scores.tolist())
        print("[4/8] PCA (1 component) computed and added.")

    rng = np.random.default_rng(args.seed)
    all_indices = np.arange(len(dataset), dtype=np.int64)
    pca_bins = _build_quantile_bins(
        np.asarray(pca_scores, dtype=np.float64), num_bins=10
    )

    target_test_rows = args.max_test_samples
    if len(dataset) <= target_test_rows:
        raise ValueError(
            f"Dataset has {len(dataset)} rows which is not enough to build "
            f"a {target_test_rows}-row test set and a separate training set."
        )

    print(
        f"[5/8] Building test set ({target_test_rows} rows) with PCA-bin proportions..."
    )
    test_indices = _stratified_sample_indices(
        all_indices, pca_bins, target_test_rows, rng
    )
    test_dataset = dataset.select(test_indices)

    train_pool_indices = np.setdiff1d(
        all_indices, np.asarray(test_indices, dtype=np.int64), assume_unique=False
    )
    train_pool_rows = len(train_pool_indices)
    print(
        f"[6/8] Test set ready: {len(test_dataset)} rows. Remaining pool: {train_pool_rows} rows."
    )

    if args.max_train_samples is not None:
        if args.max_train_samples <= 0:
            raise ValueError("--max_train_samples must be > 0.")
        target_train_rows = min(args.max_train_samples, train_pool_rows)
        print(
            f"[7/8] Train target: {target_train_rows} rows (max_train_samples={args.max_train_samples})."
        )
    else:
        if not (0 < args.dataset_fraction <= 1.0):
            raise ValueError("--dataset_fraction must be in (0, 1].")
        target_train_rows = max(1, int(train_pool_rows * args.dataset_fraction))
        print(
            f"[7/8] Train target: {target_train_rows} rows (fraction={args.dataset_fraction})."
        )

    train_pool_bins = pca_bins[train_pool_indices]
    selected_train_indices = _stratified_sample_indices(
        train_pool_indices, train_pool_bins, target_train_rows, rng
    )
    train_dataset = dataset.select(selected_train_indices)
    print(f"[7/8] Training set before filtering: {len(train_dataset)} rows.")

    print("[8/8] Filtering and formatting training dataset...")
    train_dataset = train_dataset.filter(
        lambda x: bool(normalize_text(x.get("spoken_English")))
        and bool(normalize_text(x.get("equation"))),
        desc="Filtering empty samples",
    )
    train_dataset = train_dataset.map(
        lambda x: {"text": make_chatml_text(x, tokenizer)},
        remove_columns=train_dataset.column_names,
        desc="Formatting into ChatML",
    )
    print(f"[8/8] Final training rows: {len(train_dataset)}")

    os.makedirs(args.output_dir, exist_ok=True)
    train_save = os.path.join(
        args.output_dir, getattr(args, "train_dataset_name", "train_dataset")
    )
    test_save = os.path.join(
        args.output_dir, getattr(args, "test_dataset_name", "test_dataset")
    )
    train_dataset.save_to_disk(train_save)
    test_dataset.save_to_disk(test_save)
    print(f"\n✓ Train dataset saved to: {train_save}  ({len(train_dataset)} rows)")
    print(f"✓ Test  dataset saved to: {test_save}  ({len(test_dataset)} rows)")

    return train_dataset, test_dataset


def main():
    args = parse_args()
    hf_token = args.hf_token or os.getenv("HF_TOKEN")

    print(f"Loading tokenizer from '{args.model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True, token=hf_token
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    build_datasets(args, tokenizer)


if __name__ == "__main__":
    main()
