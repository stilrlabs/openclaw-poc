#!/usr/bin/env python3
"""Unit tests for code-signals-train-lora JSONL parsing (no GPU)."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent / "code-signals-train-lora.py"
_SPEC = importlib.util.spec_from_file_location("code_signals_train_lora", _SCRIPT)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

load_jsonl_records = _MODULE.load_jsonl_records
record_to_messages = _MODULE.record_to_messages
records_from_jsonl = _MODULE.records_from_jsonl


class TestJsonlParsing(unittest.TestCase):
    def test_records_from_jsonl_rejects_invalid_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.jsonl"
            path.write_text('{"id": "a", "messages": []}\n\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                records_from_jsonl(path)

    def test_record_to_messages_chat_format(self) -> None:
        raw = {
            "id": "1",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ],
        }
        messages = record_to_messages(raw)
        self.assertEqual(len(messages), 3)
        self.assertEqual(messages[-1]["role"], "assistant")

    def test_record_to_messages_instruction_format(self) -> None:
        raw = {"id": "2", "prompt": "Where is routing?", "completion": "See src/routing."}
        messages = record_to_messages(raw)
        self.assertEqual(messages[1]["content"], "Where is routing?")
        self.assertEqual(messages[2]["content"], "See src/routing.")

    def test_records_from_jsonl_round_trip(self) -> None:
        rows = [
            {
                "id": "chat-1",
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "user"},
                    {"role": "assistant", "content": "assistant"},
                ],
            },
            {"id": "inst-1", "prompt": "p", "completion": "c"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "synthetic-chat-sft.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            records = records_from_jsonl(path)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].record_id, "chat-1")
        self.assertEqual(records[1].messages[-1]["content"], "c")


if __name__ == "__main__":
    unittest.main()
