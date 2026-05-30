#!/usr/bin/env python3
"""LoRA SFT smoke trainer for Code Signals synthetic-chat-sft.jsonl (ROCm / PyTorch)."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MIN_TRANSFORMERS_VERSION = (5, 9, 0)
DEFAULT_MODEL_ID = "Qwen/Qwen3.5-0.8B"
DEFAULT_SYSTEM_PROMPT = (
    "You answer questions about this repository using grounded facts from "
    "extracted code and documentation artifacts."
)

LORA_TARGET_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass(frozen=True)
class TrainRecord:
    record_id: str
    messages: list[dict[str, str]]


def _parse_version_tuple(version: str) -> tuple[int, ...]:
    parts: list[int] = []
    for piece in version.split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def require_transformers_version(min_version: tuple[int, ...] = MIN_TRANSFORMERS_VERSION) -> None:
    import transformers

    installed = _parse_version_tuple(transformers.__version__)
    if installed < min_version:
        raise SystemExit(
            f"transformers>={'.'.join(str(x) for x in min_version)} required (Qwen3.5 + LoRA SFT) "
            f"(installed {transformers.__version__})."
        )


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            records.append(payload)
    return records


def record_to_messages(raw: dict[str, Any]) -> list[dict[str, str]]:
    messages = raw.get("messages")
    if isinstance(messages, list) and messages:
        normalized: list[dict[str, str]] = []
        for entry in messages:
            if not isinstance(entry, dict):
                raise ValueError("messages entries must be objects")
            role = str(entry.get("role") or "").strip()
            content = str(entry.get("content") or "").strip()
            if not role or not content:
                raise ValueError("each message needs non-empty role and content")
            normalized.append({"role": role, "content": content})
        if normalized[-1]["role"] != "assistant":
            raise ValueError("last message must be from assistant")
        return normalized

    prompt = str(raw.get("prompt") or "").strip()
    response = str(raw.get("response") or raw.get("completion") or "").strip()
    if not prompt or not response:
        raise ValueError("record needs messages[] or prompt+response/completion")
    return [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]


def records_from_jsonl(path: Path) -> list[TrainRecord]:
    parsed: list[TrainRecord] = []
    for index, raw in enumerate(load_jsonl_records(path)):
        record_id = str(raw.get("id") or f"row-{index}")
        messages = record_to_messages(raw)
        parsed.append(TrainRecord(record_id=record_id, messages=messages))
    if not parsed:
        raise ValueError(f"No training records found in {path}")
    return parsed


def discover_lora_target_modules(model: Any) -> list[str]:
    names: list[str] = []
    for module_name, _module in model.named_modules():
        if module_name.split(".")[-1] in LORA_TARGET_SUFFIXES:
            names.append(module_name)
    if not names:
        raise RuntimeError(
            "No LoRA target modules found; expected suffixes "
            f"{LORA_TARGET_SUFFIXES} in model.named_modules()."
        )
    return sorted(set(names))


def tokenize_sft_example(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    max_seq_len: int,
) -> dict[str, list[int]]:
    if messages[-1]["role"] != "assistant":
        raise ValueError("last message must be assistant")

    prompt_messages = messages[:-1]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if not full_ids:
        raise ValueError("tokenization produced empty input_ids")

    if len(full_ids) > max_seq_len:
        full_ids = full_ids[:max_seq_len]
        prompt_len = min(len(prompt_ids), max_seq_len)
    else:
        prompt_len = len(prompt_ids)

    labels = [-100] * len(full_ids)
    for index in range(prompt_len, len(full_ids)):
        labels[index] = full_ids[index]

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


def build_dataset_dict(records: list[TrainRecord], tokenizer: Any, *, max_seq_len: int) -> Any:
    from datasets import Dataset

    rows: list[dict[str, list[int]]] = []
    for record in records:
        rows.append(
            tokenize_sft_example(tokenizer, record.messages, max_seq_len=max_seq_len),
        )
    return Dataset.from_list(rows)


def rocm_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LoRA SFT smoke trainer for synthetic Code Signals chat JSONL.",
    )
    parser.add_argument("--data", required=True, type=Path, help="synthetic-chat-sft.jsonl path")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for adapter + metrics")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Hugging Face model id")
    parser.add_argument("--eval-data", type=Path, default=None, help="Optional eval JSONL path")
    parser.add_argument("--max-steps", type=int, default=20, help="Training step budget (smoke default)")
    parser.add_argument("--max-seq-len", type=int, default=2048, help="Truncate sequences to this length")
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--per-device-batch-size", type=int, default=1, help="Per-device train batch size")
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="AdamW learning rate")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load two records and run one forward pass (no adapter save).",
    )
    return parser.parse_args(argv)


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    require_transformers_version()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
        Trainer,
        TrainingArguments,
    )

    data_path = args.data.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_records = records_from_jsonl(data_path)
    if args.dry_run:
        train_records = train_records[:2]

    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer = processor.tokenizer
    if tokenizer is None:
        raise SystemExit(f"No tokenizer on processor for model {args.model_id}")

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if rocm_available() else torch.float32,
    )
    if rocm_available():
        model = model.to("cuda")

    target_modules = discover_lora_target_modules(model)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)

    train_dataset = build_dataset_dict(train_records, tokenizer, max_seq_len=args.max_seq_len)
    eval_dataset = None
    if args.eval_data and args.eval_data.is_file():
        eval_records = records_from_jsonl(args.eval_data.resolve())
        if args.dry_run:
            eval_records = eval_records[:2]
        eval_dataset = build_dataset_dict(eval_records, tokenizer, max_seq_len=args.max_seq_len)

    metrics: dict[str, Any] = {
        "model_id": args.model_id,
        "record_count": len(train_records),
        "rocm_available": rocm_available(),
        "dry_run": args.dry_run,
        "max_steps": args.max_steps,
        "lora_target_modules": target_modules,
    }

    if args.dry_run:
        model.train()
        batch = train_dataset[0]
        input_ids = torch.tensor([batch["input_ids"]], device=model.device)
        attention_mask = torch.tensor([batch["attention_mask"]], device=model.device)
        labels = torch.tensor([batch["labels"]], device=model.device)
        with torch.autocast(device_type=model.device.type, dtype=torch.bfloat16, enabled=rocm_available()):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        metrics["dry_run_loss"] = float(outputs.loss.detach().cpu())
        metrics_path = output_dir / "train-metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        return metrics

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        logging_steps=max(1, args.max_steps // 10),
        save_steps=args.max_steps,
        save_total_limit=1,
        bf16=rocm_available(),
        report_to=[],
        remove_unused_columns=False,
        dataloader_pin_memory=rocm_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    train_result = trainer.train()
    metrics["train_loss"] = float(train_result.training_loss) if train_result.training_loss is not None else None
    if eval_dataset is not None:
        eval_metrics = trainer.evaluate()
        metrics["eval"] = {key: float(value) for key, value in eval_metrics.items() if isinstance(value, (int, float))}

    adapter_dir = output_dir / "adapter"
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)

    metrics_path = output_dir / "train-metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.data.is_file():
        raise SystemExit(f"Training data not found: {args.data}")
    if args.eval_data is not None and not args.eval_data.is_file():
        raise SystemExit(f"Eval data not found: {args.eval_data}")

    metrics = run_training(args)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as error:  # noqa: BLE001
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
