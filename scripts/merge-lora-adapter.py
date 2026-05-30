#!/usr/bin/env python3
"""Merge a Code Signals PEFT adapter into its base Hugging Face model snapshot."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


def configure_rocm_visible_devices(device_id: str) -> str:
    device_id = device_id.strip()
    if not device_id:
        raise ValueError("rocm device id must not be empty")
    os.environ["CUDA_VISIBLE_DEVICES"] = device_id
    os.environ["HIP_VISIBLE_DEVICES"] = device_id
    os.environ["ROCR_VISIBLE_DEVICES"] = device_id
    return device_id


def resolve_model_id(lora_dir: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    metrics_path = lora_dir / "train-metrics.json"
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        model_id = metrics.get("model_id")
        if isinstance(model_id, str) and model_id.strip():
            return model_id.strip()
    raise SystemExit(
        f"Could not resolve base model id; pass --model-id or include train-metrics.json under {lora_dir}"
    )


def merge_adapter(
    *,
    lora_dir: Path,
    merged_dir: Path,
    model_id: str,
    rocm_device_id: str,
) -> dict[str, Any]:
    configure_rocm_visible_devices(rocm_device_id)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    adapter_dir = lora_dir / "adapter"
    if not adapter_dir.is_dir():
        raise SystemExit(f"Missing adapter directory: {adapter_dir}")

    merged_dir.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    if torch.cuda.is_available():
        model = model.to("cuda:0")

    model = PeftModel.from_pretrained(model, adapter_dir)
    model = model.merge_and_unload()

    model.save_pretrained(merged_dir, safe_serialization=True)
    processor.save_pretrained(merged_dir)

    return {
        "model_id": model_id,
        "adapter_dir": str(adapter_dir),
        "merged_dir": str(merged_dir),
        "rocm_device_id": rocm_device_id,
        "rocm_available": bool(torch.cuda.is_available()),
        "rocm_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge synthetic LoRA adapter into base HF weights.")
    parser.add_argument(
        "--lora-dir",
        required=True,
        type=Path,
        help="Directory containing adapter/ and train-metrics.json",
    )
    parser.add_argument("--merged-dir", required=True, type=Path, help="Output merged HF snapshot")
    parser.add_argument("--model-id", default=None, help="Override base model id")
    parser.add_argument(
        "--rocm-device-id",
        default=os.environ.get("ROCM_DEVICE_ID", "0"),
        help="Visible ROCm device index (default 0)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    lora_dir = args.lora_dir.resolve()
    if not lora_dir.is_dir():
        raise SystemExit(f"Lora dir not found: {lora_dir}")

    model_id = resolve_model_id(lora_dir, args.model_id)
    summary = merge_adapter(
        lora_dir=lora_dir,
        merged_dir=args.merged_dir.resolve(),
        model_id=model_id,
        rocm_device_id=args.rocm_device_id,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as error:  # noqa: BLE001
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
