#!/usr/bin/env python3
"""Pipeline orchestrator: build datasets, train, or both.

Modes
-----
  --mode datasets   Build and save train/test datasets only.
  --mode train      Train on pre-existing datasets (requires --train_dataset_dir / --test_dataset_dir).
  --mode all        Build datasets then train in one go (default).

All arguments from build_datasets.py and train.py are available here.
"""

from __future__ import annotations

import argparse
import os

import torch
from transformers import AutoTokenizer, set_seed

from build_datasets import parse_args as _dataset_parse_args
from train import build_model_and_tokenizer, train, parse_args as _train_parse_args


# ---------------------------------------------------------------------------
# Unified CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run dataset building and/or QLoRA fine-tuning for Qwen/Qwen3.5-4B on MathBridge.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Mode ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["datasets", "train", "all"],
        help=(
            "'datasets': build and save datasets only. "
            "'train': train on existing datasets (needs --train_dataset_dir & --test_dataset_dir). "
            "'all': build datasets then train."
        ),
    )

    # ── Shared ────────────────────────────────────────────────────────────
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face token. Falls back to HF_TOKEN env var.",
    )

    # ── Dataset building ──────────────────────────────────────────────────
    dataset_group = parser.add_argument_group(
        "Dataset building (--mode datasets | all)"
    )
    dataset_group.add_argument("--dataset_name", type=str, default="Kyudan/MathBridge")
    dataset_group.add_argument("--dataset_split", type=str, default="train")
    dataset_group.add_argument(
        "--datasets_output_dir",
        type=str,
        default="./datasets",
        help="Directory where train_dataset/ and test_dataset/ are saved.",
    )
    dataset_group.add_argument(
        "--dataset_fraction",
        type=float,
        default=0.005,
        help="Fraction of the pool to use for training. Ignored when --max_train_samples is set.",
    )
    dataset_group.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Hard cap on training rows. Overrides --dataset_fraction.",
    )
    dataset_group.add_argument(
        "--max_test_samples",
        type=int,
        default=300,
        help="Number of test rows to build.",
    )

    # ── Training ──────────────────────────────────────────────────────────
    train_group = parser.add_argument_group("Training (--mode train | all)")
    train_group.add_argument(
        "--train_dataset_dir",
        type=str,
        default=None,
        help="Pre-built train dataset path. Auto-set in 'all' mode.",
    )
    train_group.add_argument(
        "--test_dataset_dir",
        type=str,
        default=None,
        help="Pre-built test dataset path. Auto-set in 'all' mode.",
    )
    train_group.add_argument(
        "--output_dir",
        type=str,
        default="./qwen-mathbridge-qlora",
        help="Directory for model checkpoints and final adapter.",
    )
    train_group.add_argument("--max_seq_length", type=int, default=256)
    train_group.add_argument("--per_device_train_batch_size", type=int, default=1)
    train_group.add_argument("--gradient_accumulation_steps", type=int, default=16)
    train_group.add_argument("--max_steps", type=int, default=1200)
    train_group.add_argument("--learning_rate", type=float, default=2e-4)
    train_group.add_argument("--warmup_ratio", type=float, default=0.03)
    train_group.add_argument("--logging_steps", type=int, default=10)
    train_group.add_argument("--save_steps", type=int, default=100)
    train_group.add_argument("--lora_r", type=int, default=8)
    train_group.add_argument("--lora_alpha", type=int, default=16)
    train_group.add_argument("--lora_dropout", type=float, default=0.05)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Adapters to reuse module functions without re-parsing sys.argv
# ---------------------------------------------------------------------------


class _DatasetArgs:
    """Thin namespace fed into build_datasets()."""

    def __init__(self, args: argparse.Namespace):
        self.model_name = args.model_name
        self.dataset_name = args.dataset_name
        self.dataset_split = args.dataset_split
        self.output_dir = args.datasets_output_dir
        self.dataset_fraction = args.dataset_fraction
        self.max_train_samples = args.max_train_samples
        self.max_test_samples = args.max_test_samples
        self.seed = args.seed
        self.hf_token = args.hf_token


class _TrainArgs:
    """Thin namespace fed into train()."""

    def __init__(self, args: argparse.Namespace):
        self.model_name = args.model_name
        self.train_dataset_dir = args.train_dataset_dir
        self.test_dataset_dir = args.test_dataset_dir
        self.output_dir = args.output_dir
        self.max_seq_length = args.max_seq_length
        self.per_device_train_batch_size = args.per_device_train_batch_size
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.max_steps = args.max_steps
        self.learning_rate = args.learning_rate
        self.warmup_ratio = args.warmup_ratio
        self.logging_steps = args.logging_steps
        self.save_steps = args.save_steps
        self.lora_r = args.lora_r
        self.lora_alpha = args.lora_alpha
        self.lora_dropout = args.lora_dropout
        self.seed = args.seed
        self.hf_token = args.hf_token


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    hf_token = args.hf_token or os.getenv("HF_TOKEN")
    args.hf_token = hf_token
    set_seed(args.seed)

    # ── MODE: datasets ────────────────────────────────────────────────────
    if args.mode == "datasets":
        print("=== MODE: datasets ===")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name, trust_remote_code=True, token=hf_token
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        build_datasets(_DatasetArgs(args), tokenizer)
        return

    # ── MODE: train ───────────────────────────────────────────────────────
    if args.mode == "train":
        print("=== MODE: train ===")
        if not args.train_dataset_dir or not args.test_dataset_dir:
            raise ValueError(
                "--mode train requires --train_dataset_dir and --test_dataset_dir."
            )
        if not torch.cuda.is_available():
            raise EnvironmentError("CUDA GPU is required for training.")
        train(_TrainArgs(args))
        return

    # ── MODE: all ─────────────────────────────────────────────────────────
    print("=== MODE: all (build datasets → train) ===")
    if not torch.cuda.is_available():
        raise EnvironmentError("CUDA GPU is required for training.")

    # Step 1: build datasets (reuse tokenizer after for training)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True, token=hf_token
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset_args = _DatasetArgs(args)
    build_datasets(dataset_args, tokenizer)

    # Wire the freshly built dataset paths into train args
    args.train_dataset_dir = os.path.join(args.datasets_output_dir, "train_dataset")
    args.test_dataset_dir = os.path.join(args.datasets_output_dir, "test_dataset")

    # Step 2: train
    train(_TrainArgs(args))


if __name__ == "__main__":
    main()
