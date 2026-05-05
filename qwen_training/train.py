#!/usr/bin/env python3
"""Fine-tune Qwen/Qwen3.5-4B on pre-built datasets with QLoRA (4-bit).

Designed for constrained GPUs such as RTX 3070 8GB VRAM.
Expects datasets already built by build_datasets.py (or run_pipeline.py --mode datasets).
"""

from __future__ import annotations

import argparse
import inspect
import os

import torch
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from trl import SFTConfig, SFTTrainer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning for Qwen/Qwen3.5-4B on pre-built datasets."
    )
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument(
        "--train_dataset_dir",
        type=str,
        required=True,
        help="Path to the pre-built train dataset (load_from_disk).",
    )
    parser.add_argument(
        "--test_dataset_dir",
        type=str,
        required=True,
        help="Path to the pre-built test dataset (load_from_disk).",
    )
    parser.add_argument("--output_dir", type=str, default="./qwen-mathbridge-qlora")

    # Training hyperparameters
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--max_steps", type=int, default=1200)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)

    # LoRA
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face token. Falls back to HF_TOKEN env var.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model + tokenizer
# ---------------------------------------------------------------------------


def build_model_and_tokenizer(
    model_name: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    hf_token: str | None,
):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, token=hf_token
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        token=hf_token,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace):
    if not torch.cuda.is_available():
        raise EnvironmentError("CUDA GPU is required for this script.")

    set_seed(args.seed)
    hf_token = args.hf_token or os.getenv("HF_TOKEN")

    print(f"[1/4] Loading datasets from disk...")
    train_dataset = load_from_disk(args.train_dataset_dir)
    test_dataset = load_from_disk(args.test_dataset_dir)
    print(f"       Train: {len(train_dataset)} rows | Test: {len(test_dataset)} rows")

    print(f"[2/4] Building model and tokenizer ({args.model_name})...")
    try:
        model, tokenizer = build_model_and_tokenizer(
            model_name=args.model_name,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            hf_token=hf_token,
        )
    except OSError as err:
        raise RuntimeError(
            f"Failed to load model/tokenizer '{args.model_name}'. Check model ID and token."
        ) from err

    print("[3/4] Configuring SFT trainer...")
    sft_kwargs = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "max_steps": args.max_steps,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": 2,
        "fp16": False,
        "bf16": True,
        "gradient_checkpointing": True,
        "optim": "paged_adamw_8bit",
        "lr_scheduler_type": "cosine",
        "report_to": "none",
        "remove_unused_columns": False,
        "dataloader_pin_memory": False,
        "packing": False,
    }

    sft_signature = inspect.signature(SFTConfig.__init__).parameters
    if "max_seq_length" in sft_signature:
        sft_kwargs["max_seq_length"] = args.max_seq_length
    elif "max_length" in sft_signature:
        sft_kwargs["max_length"] = args.max_seq_length
    if "gradient_checkpointing_kwargs" in sft_signature:
        sft_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    if "dataset_text_field" in sft_signature:
        sft_kwargs["dataset_text_field"] = "text"

    sft_config = SFTConfig(**sft_kwargs)

    trainer_kwargs = {
        "model": model,
        "args": sft_config,
        "train_dataset": train_dataset,
    }
    signature_params = inspect.signature(SFTTrainer.__init__).parameters
    if "dataset_text_field" in signature_params:
        trainer_kwargs["dataset_text_field"] = "text"
    if "processing_class" in signature_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = SFTTrainer(**trainer_kwargs)

    print("[4/4] Starting training...")
    torch.cuda.empty_cache()
    trainer.train()

    final_dir = os.path.join(args.output_dir, "final")
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\n✓ Training complete. Adapter saved to: {final_dir}")


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
