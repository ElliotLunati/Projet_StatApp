"""Fine-tune Qwen/Qwen3.5-4B on Kyudan/MathBridge with QLoRA (4-bit).

This script is designed for constrained GPUs such as RTX 3070 8GB VRAM.
"""

from __future__ import annotations

import argparse
import inspect
import os
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)
from trl import SFTConfig, SFTTrainer


SYSTEM_PROMPT = (
    "Tu es un assistant mathematique qui traduit de l'anglais parle vers des equations."
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments with safe defaults for 8GB VRAM."""
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning for Qwen/Qwen3.5-4B on Kyudan/MathBridge"
    )
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3.5-4B")
    parser.add_argument("--dataset_name", type=str, default="Kyudan/MathBridge")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument(
        "--dataset_fraction",
        type=float,
        default=0.005,
        help=(
            "Fraction of the split to randomly sample before filtering/formatting. "
            "Use 1.0 to keep the full split."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "Optional cap on number of rows randomly sampled before filtering/formatting. "
            "If set, overrides --dataset_fraction."
        ),
    )
    parser.add_argument("--output_dir", type=str, default="./qwen-mathbridge-qlora")

    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--max_steps", type=int, default=1200)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)

    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face token (optional). If not set, uses HF_TOKEN env var.",
    )
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    """Normalize nullable values from dataset rows."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def make_chatml_text(example: dict[str, Any], tokenizer: AutoTokenizer) -> str:
    """Format one training sample into ChatML text for Qwen."""
    spoken_english = normalize_text(example.get("spoken_English"))
    equation = normalize_text(example.get("equation"))

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": spoken_english},
        {"role": "assistant", "content": equation},
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def load_and_prepare_dataset(
    dataset_name: str,
    dataset_split: str,
    tokenizer: AutoTokenizer,
    hf_token: str | None,
    dataset_fraction: float,
    max_train_samples: int | None,
    sample_seed: int,
):
    """Load train split and convert to a single text field used by SFTTrainer."""
    dataset = load_dataset(dataset_name, split=dataset_split, token=hf_token)
    total_rows = len(dataset)

    # Step 1: randomly subsample rows before preprocessing to avoid always using
    # the same leading slice of the split.
    if max_train_samples is not None:
        if max_train_samples <= 0:
            raise ValueError("--max_train_samples must be > 0 when provided.")
        target_rows = min(max_train_samples, total_rows)
        if target_rows < total_rows:
            dataset = dataset.shuffle(seed=sample_seed).select(range(target_rows))
    else:
        if not (0 < dataset_fraction <= 1.0):
            raise ValueError("--dataset_fraction must be in (0, 1].")
        if dataset_fraction < 1.0:
            target_rows = max(1, int(total_rows * dataset_fraction))
            dataset = dataset.shuffle(seed=sample_seed).select(range(target_rows))

    print(f"Loaded split '{dataset_split}' with {len(dataset)} rows before filtering.")

    required_columns = {"spoken_English", "equation"}
    missing_columns = required_columns - set(dataset.column_names)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Dataset is missing required columns: {missing}. "
            "Expected columns: spoken_English and equation."
        )

    dataset = dataset.filter(
        lambda x: bool(normalize_text(x.get("spoken_English")))
        and bool(normalize_text(x.get("equation"))),
        desc="Filtering empty samples",
    )

    # Step 2: build one ChatML string per sample for supervised fine-tuning.
    dataset = dataset.map(
        lambda x: {"text": make_chatml_text(x, tokenizer)},
        remove_columns=dataset.column_names,
        desc="Formatting samples into ChatML",
    )
    return dataset


def build_model_and_tokenizer(
    model_name: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    hf_token: str | None,
):
    """Load quantized base model and attach a LoRA adapter."""
    # Step 3: QLoRA quantization settings for low VRAM training.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=hf_token,
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

    # Step 4: memory-saving toggles required for 8GB cards.
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        # Include attention projections (q/k/v/o) and common MLP projections.
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


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if not torch.cuda.is_available():
        raise EnvironmentError("CUDA GPU is required for this script. ")

    hf_token = args.hf_token or os.getenv("HF_TOKEN")

    try:
        # Step 5: load quantized model + LoRA adapter.
        model, tokenizer = build_model_and_tokenizer(
            model_name=args.model_name,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            hf_token=hf_token,
        )

        # Step 6: build the training text field from dataset columns.
        train_dataset = load_and_prepare_dataset(
            dataset_name=args.dataset_name,
            dataset_split=args.dataset_split,
            tokenizer=tokenizer,
            hf_token=hf_token,
            dataset_fraction=args.dataset_fraction,
            max_train_samples=args.max_train_samples,
            sample_seed=args.seed,
        )

        # Step 7: configure low-memory SFT training.
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

        # Keep compatibility between TRL versions where tokenizer arg was renamed.
        signature_params = inspect.signature(SFTTrainer.__init__).parameters
        if "dataset_text_field" in signature_params:
            trainer_kwargs["dataset_text_field"] = "text"
        if "processing_class" in signature_params:
            trainer_kwargs["processing_class"] = tokenizer
        else:
            trainer_kwargs["tokenizer"] = tokenizer

        # Step 8: train and export the final LoRA adapter.
        trainer = SFTTrainer(**trainer_kwargs)
        torch.cuda.empty_cache()
        trainer.train()

        final_dir = os.path.join(args.output_dir, "final")
        trainer.model.save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        print(f"Training complete. Adapter and tokenizer saved to: {final_dir}")

    except OSError as err:
        raise RuntimeError(
            "Failed to load model or tokenizer. Ensure model ID is valid: "
            f"{args.model_name}."
        ) from err
    except RuntimeError as err:
        if "out of memory" in str(err).lower():
            raise RuntimeError(
                "CUDA OOM detected. Try lowering --max_seq_length to 128, "
                "or reduce --gradient_accumulation_steps and --save_steps frequency."
            ) from err
        raise


if __name__ == "__main__":
    main()
