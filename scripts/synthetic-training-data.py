#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import textwrap
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

ALLOWED_RECORD_TYPES = {
    "architecture-explanations",
    "dependency-reasoning",
    "doc-grounded-qa",
    "repo-facts",
    "repo-navigation",
}
ALLOWED_CONFIDENCE = {"high", "medium", "low"}
DEFAULT_ENDPOINT = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen3.5:2b"
DEFAULT_MIN_RECORDS = 25
DEFAULT_EVAL_SAMPLE_SIZE = 50
DEFAULT_ARTIFACT_PREFIX = "code-signals-"
DEFAULT_MAX_TOPOLOGY_OWNER_UNITS = 24
DEFAULT_MAX_TOPOLOGY_CONSUMER_UNITS = 16
DEFAULT_MAX_SEAM_FAMILIES = 16
DEFAULT_MAX_SYMBOL_PATH_UNITS = 24

_DEFAULT_POLICY_FILE = Path(__file__).resolve().parent / "data" / "synthetic-training-policy.json"


@dataclass(frozen=True)
class SyntheticTrainingPolicy:
    """Loaded from scripts/data/synthetic-training-policy.json (or --policy / env override)."""

    schema_version: int
    max_excerpt_chars: int
    max_evidence_items: int
    top_level_routing_dirs: tuple[str, ...]
    symbol_summary_max_paths: int
    symbol_summary_max_symbols: int
    symbol_allowed_languages: frozenset[str]
    symbol_allowed_kinds: frozenset[str]
    record_type_cap_ratio: dict[str, float]
    bucket_priority: dict[str, int]
    structural_unit_types: frozenset[str]
    plugin_unit_type: str
    max_plugin_unit_records_in_corpus: int
    rebalance_default_tier: int
    rebalance_tier_by_unit_type: dict[str, int]
    low_value_doc_patterns: tuple[str, ...]
    low_value_response_patterns: tuple[str, ...]
    trivial_prompt_substrings: tuple[str, ...]
    doc_page_repository_context_tokens: tuple[str, ...]
    inference_checks: tuple[tuple[str, tuple[str, ...]], ...]
    plugin_provider_tautology: re.Pattern
    plugin_path_only_max_response_chars: int


_ACTIVE_POLICY: SyntheticTrainingPolicy | None = None


def _parse_policy_payload(raw: dict[str, Any]) -> SyntheticTrainingPolicy:
    version = int(raw.get("schema_version", 1))
    if version != 1:
        raise ValueError(f"Unsupported synthetic-training-policy schema_version: {version}")

    inference_raw = raw.get("inference_checks") or []
    inference_checks: list[tuple[str, tuple[str, ...]]] = []
    for entry in inference_raw:
        if not isinstance(entry, dict):
            continue
        phrase = str(entry.get("when_response_contains") or "").strip()
        tokens = entry.get("evidence_must_contain_any") or []
        if not phrase or not isinstance(tokens, list):
            continue
        inference_checks.append((phrase, tuple(str(t).lower() for t in tokens)))

    regex_str = str(raw.get("plugin_provider_tautology_regex") or "")
    flags = re.IGNORECASE if raw.get("plugin_provider_tautology_ignore_case", True) else 0
    try:
        tautology = re.compile(regex_str, flags)
    except re.error as error:
        raise ValueError(f"Invalid plugin_provider_tautology_regex: {error}") from error

    langs = raw.get("symbol_allowed_languages") or []
    kinds = raw.get("symbol_allowed_kinds") or []
    structural = raw.get("structural_unit_types") or []

    cap_ratio = raw.get("record_type_cap_ratio") or {}
    bucket = raw.get("bucket_priority") or {}
    tiers = raw.get("rebalance_tier_by_unit_type") or {}

    return SyntheticTrainingPolicy(
        schema_version=version,
        max_excerpt_chars=int(raw["max_excerpt_chars"]),
        max_evidence_items=int(raw["max_evidence_items"]),
        top_level_routing_dirs=tuple(str(x) for x in (raw.get("top_level_routing_dirs") or [])),
        symbol_summary_max_paths=int(raw["symbol_summary_max_paths"]),
        symbol_summary_max_symbols=int(raw["symbol_summary_max_symbols"]),
        symbol_allowed_languages=frozenset(str(x) for x in langs),
        symbol_allowed_kinds=frozenset(str(x) for x in kinds),
        record_type_cap_ratio={str(k): float(v) for k, v in cap_ratio.items()},
        bucket_priority={str(k): int(v) for k, v in bucket.items()},
        structural_unit_types=frozenset(str(x) for x in structural),
        plugin_unit_type=str(raw.get("plugin_unit_type") or "plugin"),
        max_plugin_unit_records_in_corpus=int(raw["max_plugin_unit_records_in_corpus"]),
        rebalance_default_tier=int(raw.get("rebalance_default_tier", 3)),
        rebalance_tier_by_unit_type={str(k): int(v) for k, v in tiers.items()},
        low_value_doc_patterns=tuple(str(x) for x in (raw.get("low_value_doc_patterns") or [])),
        low_value_response_patterns=tuple(str(x) for x in (raw.get("low_value_response_patterns") or [])),
        trivial_prompt_substrings=tuple(str(x) for x in (raw.get("trivial_prompt_substrings") or [])),
        doc_page_repository_context_tokens=tuple(
            str(x) for x in (raw.get("doc_page_repository_context_tokens") or [])
        ),
        inference_checks=tuple(inference_checks),
        plugin_provider_tautology=tautology,
        plugin_path_only_max_response_chars=int(raw.get("plugin_path_only_max_response_chars", 110)),
    )


def _load_builtin_fallback_policy() -> SyntheticTrainingPolicy:
    """Used only if scripts/data/synthetic-training-policy.json is missing."""
    return _parse_policy_payload(
        {
            "schema_version": 1,
            "max_excerpt_chars": 220,
            "max_evidence_items": 2,
            "top_level_routing_dirs": [
                "src",
                "extensions",
                "packages",
                "ui",
                "apps",
                "docs",
                ".github",
            ],
            "symbol_summary_max_paths": 24,
            "symbol_summary_max_symbols": 120,
            "symbol_allowed_languages": ["JavaScript", "TypeScript"],
            "symbol_allowed_kinds": [
                "class",
                "constant",
                "enum",
                "function",
                "interface",
                "method",
                "module",
                "type",
                "variable",
            ],
            "record_type_cap_ratio": {
                "doc-grounded-qa": 0.12,
                "repo-facts": 0.38,
                "repo-navigation": 0.32,
                "architecture-explanations": 0.35,
                "dependency-reasoning": 0.28,
            },
            "bucket_priority": {
                "ownership": 100,
                "routing": 90,
                "boundary": 85,
                "impact": 80,
                "symbol": 60,
                "docs": 40,
            },
            "structural_unit_types": [
                "topology-owner",
                "topology-consumer",
                "seam-family",
                "repo-topology",
                "routing-root",
                "symbol-path",
            ],
            "plugin_unit_type": "plugin",
            "max_plugin_unit_records_in_corpus": 12,
            "rebalance_default_tier": 3,
            "rebalance_tier_by_unit_type": {
                "dependency-module": 1,
                "repo-summary": 2,
                "workspace-overview": 2,
                "workflow-routing": 2,
                "doc-page": 2,
                "plugin": 4,
            },
            "low_value_doc_patterns": [
                "code fence",
                "code fences",
                "heading count",
                "supported code fence",
            ],
            "low_value_response_patterns": [
                "open the file",
                "navigate to the source by opening",
            ],
            "trivial_prompt_substrings": [
                "total number of files",
                "primary programming language",
            ],
            "doc_page_repository_context_tokens": ["document", "docs", "page"],
            "inference_checks": [
                {"when_response_contains": "compiled", "evidence_must_contain_any": ["compiled"]},
                {
                    "when_response_contains": "runtime artifact",
                    "evidence_must_contain_any": ["runtime artifact"],
                },
                {
                    "when_response_contains": "not a source file",
                    "evidence_must_contain_any": ["not a source file"],
                },
                {
                    "when_response_contains": "enabled by default",
                    "evidence_must_contain_any": ["enabledbydefault"],
                },
                {
                    "when_response_contains": "disabled by default",
                    "evidence_must_contain_any": ["enabledbydefault"],
                },
            ],
            "plugin_provider_tautology_regex": (
                r"the\s+['\"]?([\w-]+)['\"]?\s+plugin\s+is\s+associated\s+with\s+the\s+['\"]?\1['\"]?\s+provider"
            ),
            "plugin_provider_tautology_ignore_case": True,
            "plugin_path_only_max_response_chars": 110,
        }
    )


if _DEFAULT_POLICY_FILE.is_file():
    DEFAULT_SYNTHETIC_POLICY = _parse_policy_payload(
        json.loads(_DEFAULT_POLICY_FILE.read_text(encoding="utf-8"))
    )
else:
    DEFAULT_SYNTHETIC_POLICY = _load_builtin_fallback_policy()


def active_policy() -> SyntheticTrainingPolicy:
    policy = _ACTIVE_POLICY
    return policy if policy is not None else DEFAULT_SYNTHETIC_POLICY


def load_synthetic_policy_from_path(path: Path) -> SyntheticTrainingPolicy:
    if not path.is_file():
        raise SystemExit(f"Synthetic training policy file not found: {path}")
    return _parse_policy_payload(json.loads(path.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class EvidenceEntry:
    artifact: str
    locator: str
    excerpt: str


@dataclass(frozen=True)
class GenerationUnit:
    unit_id: str
    unit_type: str
    signal_bucket: str
    title: str
    prompt_context: dict[str, Any]
    source_artifacts: list[str]
    evidence: list[EvidenceEntry]
    preferred_record_types: list[str]
    example_budget: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic training-data artifacts.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download-artifact", help="Download a Code Signals artifact.")
    download.add_argument("--repo", required=True, help="owner/repo")
    download.add_argument("--run-id", required=True, type=int, help="GitHub Actions run id")
    download.add_argument(
        "--artifact-prefix",
        default=DEFAULT_ARTIFACT_PREFIX,
        help=f"Artifact name prefix to match (default: {DEFAULT_ARTIFACT_PREFIX})",
    )
    download.add_argument("--out-dir", required=True, help="Directory to extract the artifact into.")
    download.add_argument(
        "--token",
        default=os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN"),
        help="GitHub token; defaults to GITHUB_TOKEN or GH_TOKEN.",
    )

    generate = subparsers.add_parser("generate", help="Generate synthetic training-data artifacts.")
    generate.add_argument("--input-dir", required=True, help="Directory containing code-signal artifacts.")
    generate.add_argument("--out-dir", required=True, help="Directory to write synthetic data artifacts.")
    generate.add_argument(
        "--helper-endpoint",
        default=os.environ.get("SYNTHETIC_HELPER_ENDPOINT", DEFAULT_ENDPOINT),
        help=f"Helper endpoint URL (default: {DEFAULT_ENDPOINT})",
    )
    generate.add_argument(
        "--helper-model",
        default=os.environ.get("SYNTHETIC_HELPER_MODEL", DEFAULT_MODEL),
        help=f"Helper model name (default: {DEFAULT_MODEL})",
    )
    generate.add_argument(
        "--helper-timeout-seconds",
        default=float(os.environ.get("SYNTHETIC_HELPER_TIMEOUT_SECONDS", "120")),
        type=float,
        help="HTTP timeout for helper calls.",
    )
    generate.add_argument(
        "--helper-temperature",
        default=float(os.environ.get("SYNTHETIC_HELPER_TEMPERATURE", "0.2")),
        type=float,
        help="Sampling temperature for the helper model.",
    )
    generate.add_argument(
        "--min-records",
        default=DEFAULT_MIN_RECORDS,
        type=int,
        help=f"Minimum record count required for success (default: {DEFAULT_MIN_RECORDS})",
    )
    generate.add_argument(
        "--eval-sample-size",
        default=DEFAULT_EVAL_SAMPLE_SIZE,
        type=int,
        help=f"Number of records to project into the eval sample (default: {DEFAULT_EVAL_SAMPLE_SIZE})",
    )
    generate.add_argument(
        "--max-plugin-units",
        default=10,
        type=int,
        help="Maximum per-plugin manifest units (default: 10; lower reduces manifest trivia).",
    )
    generate.add_argument("--max-workflow-units", default=12, type=int, help="Maximum workflow units.")
    generate.add_argument("--max-doc-units", default=24, type=int, help="Maximum docs units.")
    generate.add_argument("--max-dependency-units", default=24, type=int, help="Maximum dependency units.")
    generate.add_argument(
        "--max-topology-owner-units",
        default=DEFAULT_MAX_TOPOLOGY_OWNER_UNITS,
        type=int,
        help=f"Maximum ts-topology-owner-map units (default: {DEFAULT_MAX_TOPOLOGY_OWNER_UNITS}).",
    )
    generate.add_argument(
        "--max-topology-consumer-units",
        default=DEFAULT_MAX_TOPOLOGY_CONSUMER_UNITS,
        type=int,
        help=f"Maximum consumer-topology units (default: {DEFAULT_MAX_TOPOLOGY_CONSUMER_UNITS}).",
    )
    generate.add_argument(
        "--max-seam-families",
        default=DEFAULT_MAX_SEAM_FAMILIES,
        type=int,
        help=f"Maximum duplicated seam family units (default: {DEFAULT_MAX_SEAM_FAMILIES}).",
    )
    generate.add_argument(
        "--max-symbol-path-units",
        default=DEFAULT_MAX_SYMBOL_PATH_UNITS,
        type=int,
        help=f"Maximum symbol-path units (default: {DEFAULT_MAX_SYMBOL_PATH_UNITS}).",
    )
    generate.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate deterministic placeholder records without calling the helper endpoint.",
    )
    generate.add_argument(
        "--policy",
        default=None,
        help=(
            "JSON policy file (filters, rebalance, caps). "
            "Default: scripts/data/synthetic-training-policy.json or SYNTHETIC_TRAINING_POLICY env."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "download-artifact":
        download_artifact(
            repo=args.repo,
            run_id=args.run_id,
            artifact_prefix=args.artifact_prefix,
            out_dir=Path(args.out_dir).resolve(),
            token=args.token,
        )
        return 0

    if args.command == "generate":
        generate_training_data(args)
        return 0

    raise SystemExit(f"Unsupported command: {args.command}")


def download_artifact(*, repo: str, run_id: int, artifact_prefix: str, out_dir: Path, token: str | None) -> None:
    if not token:
        raise SystemExit("A GitHub token is required to download workflow artifacts.")

    artifacts_url = f"https://api.github.com/repos/{repo}/actions/runs/{run_id}/artifacts?per_page=100"
    payload = github_api_json(artifacts_url, token)
    artifacts = payload.get("artifacts", [])
    matching = [
        artifact
        for artifact in artifacts
        if not artifact.get("expired", False)
        and isinstance(artifact.get("name"), str)
        and artifact["name"].startswith(artifact_prefix)
    ]
    if not matching:
        raise SystemExit(
            f"No non-expired artifact with prefix '{artifact_prefix}' found for run {run_id} in {repo}."
        )

    artifact = sorted(matching, key=lambda item: item.get("created_at", ""), reverse=True)[0]
    archive_url = artifact.get("archive_download_url")
    if not archive_url:
        raise SystemExit("Artifact did not include an archive_download_url.")

    out_dir.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(suffix=".zip", delete=False) as archive_file:
        archive_path = Path(archive_file.name)
    try:
        download_to_file(archive_url, archive_path, token)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(out_dir)
    finally:
        archive_path.unlink(missing_ok=True)


def generate_training_data(args: argparse.Namespace) -> None:
    global _ACTIVE_POLICY
    input_dir = Path(args.input_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    previous_policy = _ACTIVE_POLICY
    try:
        if args.policy:
            _ACTIVE_POLICY = load_synthetic_policy_from_path(Path(args.policy).resolve())
        elif os.environ.get("SYNTHETIC_TRAINING_POLICY"):
            _ACTIVE_POLICY = load_synthetic_policy_from_path(
                Path(os.environ["SYNTHETIC_TRAINING_POLICY"]).expanduser().resolve()
            )
        else:
            _ACTIVE_POLICY = None

        bundle = load_bundle(input_dir)
        units = build_generation_units(
            bundle,
            max_plugin_units=args.max_plugin_units,
            max_workflow_units=args.max_workflow_units,
            max_doc_units=args.max_doc_units,
            max_dependency_units=args.max_dependency_units,
            max_topology_owner_units=args.max_topology_owner_units,
            max_topology_consumer_units=args.max_topology_consumer_units,
            max_seam_families=args.max_seam_families,
            max_symbol_path_units=args.max_symbol_path_units,
        )
        records: list[dict[str, Any]] = []
        stats = {
            "units_total": len(units),
            "units_succeeded": 0,
            "units_failed": 0,
            "records_generated": 0,
            "records_after_validation": 0,
            "records_kept": 0,
            "duplicates_dropped": 0,
            "contradictions_dropped": 0,
            "validation_dropped": 0,
            "rebalance_dropped": 0,
            "unit_errors": [],
        }

        seen_prompt_to_response: dict[str, str] = {}
        seen_record_hashes: set[str] = set()

        for unit in units:
            try:
                generated = (
                    generate_dry_run_examples(unit)
                    if args.dry_run
                    else call_helper_endpoint(
                        unit=unit,
                        helper_endpoint=args.helper_endpoint,
                        helper_model=args.helper_model,
                        helper_timeout_seconds=args.helper_timeout_seconds,
                        helper_temperature=args.helper_temperature,
                    )
                )
                normalized = normalize_records(
                    generated,
                    unit=unit,
                    bundle=bundle,
                    seen_prompt_to_response=seen_prompt_to_response,
                    seen_record_hashes=seen_record_hashes,
                    stats=stats,
                )
                stats["units_succeeded"] += 1
                stats["records_generated"] += len(generated)
                stats["records_after_validation"] += len(normalized)
                records.extend(normalized)
            except Exception as error:  # noqa: BLE001
                stats["units_failed"] += 1
                stats["unit_errors"].append({"unit_id": unit.unit_id, "error": str(error)})

        pre_rebalance = len(records)
        if pre_rebalance < args.min_records:
            raise SystemExit(
                f"Generated only {pre_rebalance} records, below the minimum required {args.min_records}."
            )

        records = rebalance_records(records)
        stats["rebalance_dropped"] = pre_rebalance - len(records)
        stats["records_kept"] = len(records)
        if len(records) < args.min_records:
            raise SystemExit(
                f"After rebalance only {len(records)} records remain, below the minimum required {args.min_records}."
            )
        records.sort(key=lambda record: (record["record_type"], record["prompt"], record["response"]))
        write_jsonl(out_dir / "synthetic-training-corpus.jsonl", records)
        write_jsonl(out_dir / "synthetic-chat-sft.jsonl", [to_chat_sft_record(record) for record in records])
        write_jsonl(out_dir / "synthetic-instruction.jsonl", [to_instruction_record(record) for record in records])
        write_jsonl(
            out_dir / "synthetic-eval-sample.jsonl",
            project_eval_sample(records, args.eval_sample_size),
        )
        write_report(
            out_dir / "synthetic-generation-report.md",
            bundle=bundle,
            records=records,
            units=units,
            helper_endpoint=args.helper_endpoint,
            helper_model=args.helper_model,
            dry_run=args.dry_run,
            stats=stats,
        )
    finally:
        _ACTIVE_POLICY = previous_policy


def load_bundle(input_dir: Path) -> dict[str, Any]:
    bundle: dict[str, Any] = {"input_dir": input_dir}
    bundle["summary_markdown"] = read_text(input_dir / "summary.md")
    bundle["summary"] = parse_summary(bundle["summary_markdown"])
    bundle["repo_tree_markdown"] = read_optional_text(input_dir / "repo-tree.md")
    bundle["repo_tree_summary"] = parse_repo_tree_summary(bundle["repo_tree_markdown"])
    bundle["workspace_packages"] = read_json(input_dir / "workspace-packages.json")
    bundle["plugin_manifests"] = read_json(input_dir / "plugin-manifests.json")
    bundle["workflow_inventory"] = read_json(input_dir / "workflow-inventory.json")
    bundle["docs_inventory"] = read_json(input_dir / "docs-inventory.json")
    bundle["language_summary"] = read_json(input_dir / "language-summary.json")
    bundle["ts_deps"] = read_json(input_dir / "ts-deps.json")
    bundle["ts_topology_owner_map"] = read_optional_json(input_dir / "ts-topology-owner-map.json")
    bundle["ts_topology_consumer_topology"] = read_optional_json(
        input_dir / "ts-topology-consumer-topology.json"
    )
    bundle["seam_inventory"] = read_optional_json(input_dir / "seam-inventory.json")
    bundle["symbol_summary"] = load_symbol_summary(input_dir)
    return bundle


def parse_summary(summary_markdown: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in summary_markdown.splitlines():
        if not line.startswith("- "):
            continue
        if ":" not in line:
            continue
        label, value = line[2:].split(":", 1)
        fields[label.strip().lower().replace(" ", "_")] = value.strip().strip("`")
    return fields


def parse_repo_tree_summary(repo_tree_markdown: str | None) -> dict[str, Any] | None:
    if not repo_tree_markdown:
        return None
    match = re.search(r"```text\n(.*?)\n```", repo_tree_markdown, re.DOTALL)
    if not match:
        return None
    lines = [line.rstrip() for line in match.group(1).splitlines() if line.strip()]
    top_level_entries: list[str] = []
    key_root_children: dict[str, list[str]] = {}
    current_root: str | None = None
    for line in lines:
        if line == ".":
            continue
        normalized = line.replace("│", " ").replace("├", " ").replace("└", " ").replace("──", " ")
        stripped = normalized.strip()
        if not stripped:
            continue
        indent = len(normalized) - len(normalized.lstrip(" "))
        if indent == 0:
            current_root = stripped
            top_level_entries.append(stripped)
            key_root_children.setdefault(stripped, [])
            continue
        if indent == 4 and current_root in active_policy().top_level_routing_dirs:
            children = key_root_children.setdefault(current_root, [])
            if len(children) < 8:
                children.append(stripped)
    return {
        "topLevelEntries": top_level_entries[:24],
        "keyRoots": {root: children for root, children in key_root_children.items() if children},
    }


def load_symbol_summary(input_dir: Path) -> dict[str, Any] | None:
    explicit_summary = read_optional_json(input_dir / "symbol-summary.json")
    if explicit_summary is not None:
        return explicit_summary
    tags_path = input_dir / "tags.json"
    if not tags_path.is_file():
        return None
    tags_payload = read_json(tags_path)
    tags = tags_payload.get("tags", [])
    policy = active_policy()
    filtered = []
    for tag in tags:
        if not isinstance(tag, dict):
            continue
        language = tag.get("language")
        kind = tag.get("kind")
        path = tag.get("path")
        if language not in policy.symbol_allowed_languages:
            continue
        if kind not in policy.symbol_allowed_kinds:
            continue
        if not isinstance(path, str) or not path.startswith(("src/", "extensions/", "packages/", "ui/")):
            continue
        filtered.append(
            {
                "name": tag.get("name"),
                "kind": kind,
                "language": language,
                "path": path,
                "scope": tag.get("scope"),
            }
        )
    filtered.sort(
        key=lambda tag: (
            str(tag.get("path") or ""),
            str(tag.get("kind") or ""),
            str(tag.get("name") or ""),
        )
    )
    by_path: dict[str, list[dict[str, Any]]] = {}
    for tag in filtered:
        by_path.setdefault(tag["path"], []).append(tag)
    top_paths = sorted(
        (
            {
                "path": path,
                "symbolCount": len(path_tags),
                "sampleSymbols": [
                    {
                        "name": path_tag.get("name"),
                        "kind": path_tag.get("kind"),
                        "scope": path_tag.get("scope"),
                    }
                    for path_tag in path_tags[:6]
                ],
            }
            for path, path_tags in by_path.items()
        ),
        key=lambda item: (-item["symbolCount"], item["path"]),
    )[: policy.symbol_summary_max_paths]
    top_symbols = filtered[: policy.symbol_summary_max_symbols]
    return {
        "source": "derived-from-tags",
        "importantPaths": top_paths,
        "importantSymbols": top_symbols,
    }


def build_generation_units(
    bundle: dict[str, Any],
    *,
    max_plugin_units: int,
    max_workflow_units: int,
    max_doc_units: int,
    max_dependency_units: int,
    max_topology_owner_units: int,
    max_topology_consumer_units: int,
    max_seam_families: int,
    max_symbol_path_units: int,
) -> list[GenerationUnit]:
    ownership_units = build_ownership_units(bundle, max_plugin_units, max_topology_owner_units)
    routing_units = build_routing_units(bundle, max_workflow_units)
    boundary_units = build_boundary_units(
        bundle,
        max_topology_consumer_units=max_topology_consumer_units,
        max_seam_families=max_seam_families,
    )
    impact_units = build_impact_units(bundle, max_dependency_units)
    doc_units = build_doc_routing_units(bundle, max_doc_units)
    symbol_units = build_curated_symbol_units(bundle, max_symbol_path_units)
    units = [
        *ownership_units,
        *routing_units,
        *boundary_units,
        *impact_units,
        *doc_units,
        *symbol_units,
    ]
    return sorted(
        units,
        key=lambda unit: (
            -active_policy().bucket_priority.get(unit.signal_bucket, 0),
            unit.unit_type,
            unit.unit_id,
        ),
    )


def make_unit(
    *,
    unit_id: str,
    unit_type: str,
    signal_bucket: str,
    title: str,
    prompt_context: dict[str, Any],
    source_artifacts: list[str],
    evidence: list[EvidenceEntry],
    preferred_record_types: list[str],
    example_budget: int,
) -> GenerationUnit:
    return GenerationUnit(
        unit_id=unit_id,
        unit_type=unit_type,
        signal_bucket=signal_bucket,
        title=title,
        prompt_context=prompt_context,
        source_artifacts=source_artifacts,
        evidence=evidence[: active_policy().max_evidence_items],
        preferred_record_types=preferred_record_types,
        example_budget=example_budget,
    )


def build_ownership_units(bundle: dict[str, Any], plugin_limit: int, topology_owner_limit: int) -> list[GenerationUnit]:
    packages = bundle["workspace_packages"]["packages"]
    summary_fields = bundle["summary"]
    tree_summary = bundle.get("repo_tree_summary") or {}
    units = [
        make_unit(
            unit_id="repo-summary",
            unit_type="repo-summary",
            signal_bucket="ownership",
            title="Repository summary",
            prompt_context={
                "repo": summary_fields.get("repo"),
                "commit": summary_fields.get("commit"),
                "workspacePackageCount": len(packages),
                "topLevelEntries": tree_summary.get("topLevelEntries", [])[:12],
            },
            source_artifacts=["summary.md", "workspace-packages.json", "repo-tree.md"],
            evidence=[
                evidence_from_object("summary.md", "frontmatter", summary_fields),
                evidence_from_object("workspace-packages.json", "packages", packages[:8]),
                evidence_from_object("repo-tree.md", "topLevelEntries", tree_summary.get("topLevelEntries", [])[:12]),
            ],
            preferred_record_types=["repo-facts", "repo-navigation"],
            example_budget=3,
        ),
        make_unit(
            unit_id="workspace-packages-overview",
            unit_type="workspace-overview",
            signal_bucket="ownership",
            title="Workspace packages overview",
            prompt_context={
                "packageCount": len(packages),
                "samplePackages": packages[:16],
            },
            source_artifacts=["workspace-packages.json"],
            evidence=[evidence_from_object("workspace-packages.json", "packages", packages[:16])],
            preferred_record_types=["repo-facts", "repo-navigation"],
            example_budget=3,
        ),
    ]

    plugin_units = []
    plugins = bundle["plugin_manifests"]["plugins"]
    normalized_plugins = []
    for plugin in plugins:
        manifest = plugin.get("manifest", {})
        normalized_plugins.append(
            {
                "id": manifest.get("id"),
                "name": manifest.get("name"),
                "path": plugin.get("path"),
                "enabledByDefault": manifest.get("enabledByDefault"),
                "providers": manifest.get("providers", []),
                "contracts": sorted((manifest.get("contracts") or {}).keys()),
                "configKeys": sorted((manifest.get("configSchema", {}).get("properties") or {}).keys()),
                "providerAuthChoices": [
                    choice.get("choiceId")
                    for choice in manifest.get("providerAuthChoices", [])
                    if isinstance(choice, dict) and choice.get("choiceId")
                ],
            }
        )
    selected_plugins = sorted(
        normalized_plugins,
        key=lambda plugin: (
            -len(plugin["contracts"]),
            -len(plugin["configKeys"]),
            str(plugin["id"] or ""),
        ),
    )[:plugin_limit]
    for index, plugin in enumerate(selected_plugins):
        plugin_units.append(
            make_unit(
                unit_id=f"plugin-{plugin['id'] or index}",
                unit_type="plugin",
                signal_bucket="ownership",
                title=f"Plugin {plugin['id']}",
                prompt_context=plugin,
                source_artifacts=["plugin-manifests.json"],
                evidence=[evidence_from_object("plugin-manifests.json", f"plugins[{index}]", plugin)],
                preferred_record_types=["repo-facts", "repo-navigation", "architecture-explanations"],
                example_budget=2,
            )
        )

    units.extend(plugin_units)
    units.extend(build_topology_owner_units(bundle, topology_owner_limit))
    return units


def build_routing_units(bundle: dict[str, Any], limit: int) -> list[GenerationUnit]:
    units: list[GenerationUnit] = []
    tree_summary = bundle.get("repo_tree_summary")
    if tree_summary:
        units.append(
            make_unit(
                unit_id="repo-topology-overview",
                unit_type="repo-topology",
                signal_bucket="routing",
                title="Repository topology overview",
                prompt_context=tree_summary,
                source_artifacts=["repo-tree.md"],
                evidence=[evidence_from_object("repo-tree.md", "summary", tree_summary)],
                preferred_record_types=["repo-navigation", "repo-facts"],
                example_budget=3,
            )
        )
        for root_name in active_policy().top_level_routing_dirs:
            children = tree_summary.get("keyRoots", {}).get(root_name, [])
            if not children:
                continue
            units.append(
                make_unit(
                    unit_id=f"routing-{slugify(root_name)}",
                    unit_type="routing-root",
                    signal_bucket="routing",
                    title=f"Routing root {root_name}",
                    prompt_context={"root": root_name, "children": children[:8]},
                    source_artifacts=["repo-tree.md"],
                    evidence=[evidence_from_object("repo-tree.md", f"root:{root_name}", children[:8])],
                    preferred_record_types=["repo-navigation", "repo-facts"],
                    example_budget=2,
                )
            )

    workflows = bundle["workflow_inventory"]["workflows"]
    selected = sorted(
        workflows,
        key=lambda workflow: (-int(workflow.get("jobCount", 0)), str(workflow.get("name") or "")),
    )[:limit]
    for index, workflow in enumerate(selected):
        context = {
            "name": workflow.get("name"),
            "path": workflow.get("path"),
            "triggers": workflow.get("triggers"),
            "jobCount": workflow.get("jobCount"),
            "jobIds": [job.get("id") for job in workflow.get("jobs", [])[:10]],
        }
        units.append(
            make_unit(
                unit_id=f"workflow-{slugify(workflow.get('name') or workflow.get('path') or str(index))}",
                unit_type="workflow-routing",
                signal_bucket="routing",
                title=f"Workflow {workflow.get('name')}",
                prompt_context=context,
                source_artifacts=["workflow-inventory.json"],
                evidence=[evidence_from_object("workflow-inventory.json", f"workflows[{index}]", context)],
                preferred_record_types=["repo-navigation", "repo-facts"],
                example_budget=2,
            )
        )
    return units


def build_boundary_units(
    bundle: dict[str, Any],
    *,
    max_topology_consumer_units: int,
    max_seam_families: int,
) -> list[GenerationUnit]:
    units: list[GenerationUnit] = []
    consumer_topology = bundle.get("ts_topology_consumer_topology")
    if isinstance(consumer_topology, dict):
        for index, record in enumerate((consumer_topology.get("records") or [])[:max_topology_consumer_units]):
            context = {
                "symbol": first_present(record, "canonicalKey", "publicSpecifiers", default="<unknown>"),
                "declarationPath": record.get("declarationPath"),
                "productionOwners": record.get("productionOwners", [])[:6],
                "productionConsumers": record.get("productionConsumers", [])[:6],
                "productionRefCount": record.get("productionRefCount"),
            }
            units.append(
                make_unit(
                    unit_id=f"boundary-consumer-{index}",
                    unit_type="topology-consumer",
                    signal_bucket="boundary",
                    title=f"Topology consumer {index + 1}",
                    prompt_context=context,
                    source_artifacts=["ts-topology-consumer-topology.json"],
                    evidence=[
                        evidence_from_object(
                            "ts-topology-consumer-topology.json",
                            f"records[{index}]",
                            context,
                        )
                    ],
                    preferred_record_types=["repo-navigation", "architecture-explanations"],
                    example_budget=2,
                )
            )

    seam_inventory = bundle.get("seam_inventory")
    if isinstance(seam_inventory, dict):
        duplicated = seam_inventory.get("duplicatedSeamFamilies") or {}
        for family_name, family in list(sorted(duplicated.items()))[:max_seam_families]:
            context = {
                "family": family_name,
                "count": family.get("count"),
                "files": family.get("files", [])[:6],
            }
            units.append(
                make_unit(
                    unit_id=f"seam-{slugify(family_name)}",
                    unit_type="seam-family",
                    signal_bucket="boundary",
                    title=f"Seam family {family_name}",
                    prompt_context=context,
                    source_artifacts=["seam-inventory.json"],
                    evidence=[evidence_from_object("seam-inventory.json", f"duplicatedSeamFamilies.{family_name}", context)],
                    preferred_record_types=["architecture-explanations", "repo-navigation"],
                    example_budget=2,
                )
            )
    return units


def build_doc_routing_units(bundle: dict[str, Any], limit: int) -> list[GenerationUnit]:
    pages = bundle["docs_inventory"]["pages"]
    selected = sorted(
        (
            page
            for page in pages
            if isinstance(page.get("path"), str)
            and not page["path"].startswith(("docs/.generated/", "docs/.i18n/"))
        ),
        key=lambda page: (doc_priority(page), str(page.get("path") or "")),
    )[:limit]
    units = []
    for index, page in enumerate(selected):
        context = {
            "path": page.get("path"),
            "title": page.get("title"),
            "headings": page.get("headings", [])[:8],
            "links": page.get("links", [])[:8],
        }
        units.append(
            make_unit(
                unit_id=f"doc-{slugify(page.get('path') or str(index))}",
                unit_type="doc-page",
                signal_bucket="docs",
                title=f"Doc page {page.get('path')}",
                prompt_context=context,
                source_artifacts=["docs-inventory.json"],
                evidence=[evidence_from_object("docs-inventory.json", f"pages[{index}]", context)],
                preferred_record_types=["doc-grounded-qa", "repo-navigation"],
                example_budget=1,
            )
        )
    return units


def build_impact_units(bundle: dict[str, Any], limit: int) -> list[GenerationUnit]:
    modules = bundle["ts_deps"]["cruiseResult"]["modules"]
    candidate_modules = []
    for module in modules:
        source = module.get("source")
        if not isinstance(source, str) or "/" not in source:
            continue
        dependencies = module.get("dependencies", [])
        local_dependencies = [
            dependency.get("resolved")
            for dependency in dependencies
            if isinstance(dependency, dict)
            and isinstance(dependency.get("resolved"), str)
            and not dependency.get("coreModule", False)
        ]
        dependents = [dependent for dependent in module.get("dependents", []) if isinstance(dependent, str)]
        if len(local_dependencies) == 0 and len(dependents) < 2:
            continue
        candidate_modules.append(
            {
                "source": source,
                "area": source.split("/", 1)[0],
                "dependencyCount": len(local_dependencies),
                "dependentCount": len(dependents),
                "localDependencies": local_dependencies[:6],
                "dependents": dependents[:6],
                "dependentAreas": sorted({dependent.split("/", 1)[0] for dependent in dependents if "/" in dependent}),
                "orphan": module.get("orphan"),
            }
        )
    selected = sorted(
        candidate_modules,
        key=lambda module: (
            -module["dependentCount"],
            -module["dependencyCount"],
            module["source"],
        ),
    )[:limit]
    units = []
    for index, module in enumerate(selected):
        units.append(
            make_unit(
                unit_id=f"dep-{slugify(module['source'])}",
                unit_type="dependency-module",
                signal_bucket="impact",
                title=f"Dependency module {module['source']}",
                prompt_context=module,
                source_artifacts=["ts-deps.json"],
                evidence=[evidence_from_object("ts-deps.json", f"cruiseResult.modules[{index}]", module)],
                preferred_record_types=["dependency-reasoning", "architecture-explanations", "repo-navigation"],
                example_budget=2,
            )
        )
    return units


def build_curated_symbol_units(bundle: dict[str, Any], limit: int) -> list[GenerationUnit]:
    symbol_summary = bundle.get("symbol_summary")
    if not isinstance(symbol_summary, dict):
        return []
    artifact_name = "symbol-summary.json" if symbol_summary.get("source") != "derived-from-tags" else "tags.json"
    important_paths = symbol_summary.get("importantPaths", [])[:limit]
    units = []
    for index, path_summary in enumerate(important_paths):
        context = {
            "path": path_summary.get("path"),
            "symbolCount": path_summary.get("symbolCount"),
            "sampleSymbols": path_summary.get("sampleSymbols", [])[:6],
        }
        units.append(
            make_unit(
                unit_id=f"symbol-{slugify(path_summary.get('path') or str(index))}",
                unit_type="symbol-path",
                signal_bucket="symbol",
                title=f"Symbol path {path_summary.get('path')}",
                prompt_context=context,
                source_artifacts=[artifact_name],
                evidence=[evidence_from_object(artifact_name, f"importantPaths[{index}]", context)],
                preferred_record_types=["repo-navigation", "repo-facts"],
                example_budget=1,
            )
        )
    return units


def build_topology_owner_units(bundle: dict[str, Any], limit: int) -> list[GenerationUnit]:
    owner_map = bundle.get("ts_topology_owner_map")
    if not isinstance(owner_map, dict):
        return []
    units = []
    for index, record in enumerate((owner_map.get("records") or [])[:limit]):
        context = {
            "symbol": first_present(record, "canonicalKey", "publicSpecifiers", default="<unknown>"),
            "declarationPath": record.get("declarationPath"),
            "productionOwners": record.get("productionOwners", [])[:6],
            "productionExtensions": record.get("productionExtensions", [])[:6],
            "productionPackages": record.get("productionPackages", [])[:6],
        }
        units.append(
            make_unit(
                unit_id=f"owner-map-{index}",
                unit_type="topology-owner",
                signal_bucket="ownership",
                title=f"Ownership map {index + 1}",
                prompt_context=context,
                source_artifacts=["ts-topology-owner-map.json"],
                evidence=[evidence_from_object("ts-topology-owner-map.json", f"records[{index}]", context)],
                preferred_record_types=["repo-facts", "repo-navigation", "architecture-explanations"],
                example_budget=2,
            )
        )
    return units


def first_present(record: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        value = record.get(key)
        if value:
            return value
    return default


def doc_priority(page: dict[str, Any]) -> tuple[int, int]:
    path = str(page.get("path") or "")
    title = str(page.get("title") or "")
    structural_boost = 0
    for token in ("plugins", "architecture", "gateway", "channels", "concepts", "install", "configuration"):
        if token in path or token in title.lower():
            structural_boost -= 1
    return (structural_boost, -(len(page.get("links", [])) + len(page.get("headings", []))))


def call_helper_endpoint(
    *,
    unit: GenerationUnit,
    helper_endpoint: str,
    helper_model: str,
    helper_timeout_seconds: float,
    helper_temperature: float,
) -> list[dict[str, Any]]:
    prompt = build_helper_prompt(unit)
    payload = {
        "model": helper_model,
        "messages": [{"role": "user", "content": prompt}],
        "think": False,
        "stream": False,
        "format": "json",
        "options": {"temperature": helper_temperature},
    }
    response = http_post_json(helper_endpoint, payload, timeout_seconds=helper_timeout_seconds)
    response_text = ""
    message = response.get("message")
    if isinstance(message, dict):
        response_text = str(message.get("content", "")).strip()
    if not response_text:
        response_text = str(response.get("response", "")).strip()
    if not response_text:
        response_text = str(response.get("thinking", "")).strip()
    parsed = parse_json_block(response_text)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("examples"), list):
        raise ValueError(f"Helper response for unit {unit.unit_id} did not contain an examples array.")
    return parsed["examples"]


def build_helper_prompt(unit: GenerationUnit) -> str:
    schema = {
        "examples": [
            {
                "record_type": "repo-facts",
                "prompt": "Question or instruction text",
                "response": "Grounded answer text",
                "confidence": "high",
                "metadata": {"style": "fact"},
            }
        ]
    }
    contract = helper_contract_for_unit(unit)
    compact_unit = {
        "id": unit.unit_id,
        "t": unit.unit_type,
        "b": unit.signal_bucket,
        "title": unit.title,
        "ctx": unit.prompt_context,
        "ev": [
            {
                "a": evidence.artifact,
                "l": evidence.locator,
                "x": evidence.excerpt,
            }
            for evidence in unit.evidence
        ],
    }
    return textwrap.dedent(
        f"""
        Generate grounded repository-training examples.

        Rules:
        - Use only the supplied unit.
        - No invented files, symbols, packages, plugins, workflows, or relationships.
        - Keep examples short, factual, and scoped to this unit only.
        - Return JSON only.
        - Produce at most {unit.example_budget} examples.
        - Allowed record_type: {sorted(unit.preferred_record_types)}.
        - Allowed confidence: ["high", "medium", "low"].
        - Do not include evidence or source_artifacts fields.

        Unit contract:
        {json.dumps(contract, indent=2, sort_keys=True)}

        Unit:
        {json.dumps(compact_unit, indent=2, sort_keys=True)}

        Output schema:
        {json.dumps(schema, indent=2)}
        """
    ).strip()


def helper_contract_for_unit(unit: GenerationUnit) -> dict[str, Any]:
    shared = {
        "focus": "repo competence through grounded structural facts",
        "avoid": [
            "repo-wide claims not supported by the unit",
            "code-fence trivia",
            "rephrasing the file path as the answer",
        ],
    }
    by_type = {
        "repo-summary": {
            "task": "ownership and high-level repo facts",
            "ask_for": ["repo-facts", "repo-navigation"],
        },
        "workspace-overview": {
            "task": "workspace ownership and package routing",
            "ask_for": ["repo-facts", "repo-navigation"],
        },
        "plugin": {
            "task": "plugin ownership, capability location, and boundary explanations",
            "ask_for": ["repo-facts", "repo-navigation", "architecture-explanations"],
            "avoid": [
                "answers that are only a manifest file path",
                "questions about default configuration keys when the answer is empty array or []",
                "tautologies that repeat the plugin id as the provider name without new information",
                "claims about compiled output, bundles, or runtime artifacts unless stated in evidence",
            ],
        },
        "topology-owner": {
            "task": "who owns what and where a surface belongs",
            "ask_for": ["repo-facts", "repo-navigation", "architecture-explanations"],
        },
        "repo-topology": {
            "task": "where to look first in the repo layout",
            "ask_for": ["repo-navigation", "repo-facts"],
        },
        "routing-root": {
            "task": "directory-level routing and likely lookup starting points",
            "ask_for": ["repo-navigation", "repo-facts"],
        },
        "workflow-routing": {
            "task": "workflow ownership and where CI behaviors live",
            "ask_for": ["repo-navigation", "repo-facts"],
        },
        "topology-consumer": {
            "task": "boundary usage and which owners consume a surface",
            "ask_for": ["repo-navigation", "architecture-explanations"],
        },
        "seam-family": {
            "task": "architectural boundary hotspots and shared seam explanations",
            "ask_for": ["architecture-explanations", "repo-navigation"],
        },
        "dependency-module": {
            "task": "impact reasoning, dependency paths, and likely affected areas",
            "ask_for": ["dependency-reasoning", "architecture-explanations", "repo-navigation"],
        },
        "doc-page": {
            "task": "where a topic is documented or where to look in docs",
            "ask_for": ["doc-grounded-qa", "repo-navigation"],
        },
        "symbol-path": {
            "task": "where major exported surfaces or symbols are located",
            "ask_for": ["repo-navigation", "repo-facts"],
        },
    }
    specific = by_type.get(
        unit.unit_type,
        {"task": "bounded factual synthesis", "ask_for": unit.preferred_record_types},
    )
    merged: dict[str, Any] = {**shared, **{key: value for key, value in specific.items() if key != "avoid"}}
    extra_avoid = specific.get("avoid")
    merged["avoid"] = list(shared["avoid"])
    if isinstance(extra_avoid, list):
        merged["avoid"].extend(extra_avoid)
    return merged


def generate_dry_run_examples(unit: GenerationUnit) -> list[dict[str, Any]]:
    return [
        {
            "record_type": unit.preferred_record_types[0],
            "prompt": f"What does {unit.title} tell us about the repository?",
            "response": f"{unit.title} is represented in the code-signals artifact and can be explained using the attached evidence.",
            "confidence": "medium",
            "metadata": {"style": "dry-run", "unit_type": unit.unit_type},
        }
    ]


def combined_evidence_blob(evidence: list[EvidenceEntry]) -> str:
    return " ".join(entry.excerpt.lower() for entry in evidence)


def response_has_unsupported_inference(response: str, evidence: list[EvidenceEntry]) -> bool:
    blob = combined_evidence_blob(evidence)
    lower = response.lower()
    for phrase, required_tokens in active_policy().inference_checks:
        if phrase not in lower:
            continue
        if not any(token in blob for token in required_tokens):
            return True
    return False


def is_plugin_manifest_trivia(unit: GenerationUnit, prompt: str, response: str) -> bool:
    policy = active_policy()
    if unit.unit_type != policy.plugin_unit_type:
        return False
    prompt_lower = prompt.lower()
    response_lower = response.lower()
    stripped = response.strip().strip("`").strip()
    max_path = policy.plugin_path_only_max_response_chars
    if stripped.startswith("extensions/") and stripped.endswith(".json") and len(response) < max_path:
        return True
    if "default configuration key" in prompt_lower or (
        "configuration key" in prompt_lower and "default" in prompt_lower
    ):
        if "empty array" in response_lower or "represented as []" in response_lower or "[]" in response_lower:
            return True
    if policy.plugin_provider_tautology.search(response):
        return True
    return False


def unit_rebalance_tier(unit_type: str) -> int:
    policy = active_policy()
    if unit_type in policy.structural_unit_types:
        return 0
    return policy.rebalance_tier_by_unit_type.get(unit_type, policy.rebalance_default_tier)


def normalize_records(
    raw_examples: list[dict[str, Any]],
    *,
    unit: GenerationUnit,
    bundle: dict[str, Any],
    seen_prompt_to_response: dict[str, str],
    seen_record_hashes: set[str],
    stats: dict[str, Any],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    artifact_files = {path.name for path in Path(bundle["input_dir"]).iterdir() if path.is_file()}
    for example in raw_examples:
        if not isinstance(example, dict):
            stats["validation_dropped"] += 1
            continue
        record_type = str(example.get("record_type", "")).strip()
        prompt = str(example.get("prompt", "")).strip()
        response = str(example.get("response", "")).strip()
        confidence = str(example.get("confidence", "medium")).strip().lower()
        if record_type not in ALLOWED_RECORD_TYPES:
            stats["validation_dropped"] += 1
            continue
        if record_type not in unit.preferred_record_types:
            stats["validation_dropped"] += 1
            continue
        if not prompt or not response or len(prompt) > 6000 or len(response) > 12000:
            stats["validation_dropped"] += 1
            continue
        if confidence not in ALLOWED_CONFIDENCE:
            stats["validation_dropped"] += 1
            continue
        if not unit.evidence:
            stats["validation_dropped"] += 1
            continue
        if any(evidence.artifact not in artifact_files for evidence in unit.evidence):
            stats["validation_dropped"] += 1
            continue
        if is_low_value_record(unit, prompt, response):
            stats["validation_dropped"] += 1
            continue
        if is_plugin_manifest_trivia(unit, prompt, response):
            stats["validation_dropped"] += 1
            continue
        if response_has_unsupported_inference(response, unit.evidence):
            stats["validation_dropped"] += 1
            continue

        prompt_key = normalize_text(prompt)
        response_key = normalize_text(response)
        previous_response = seen_prompt_to_response.get(prompt_key)
        if previous_response and previous_response != response_key:
            stats["contradictions_dropped"] += 1
            continue
        seen_prompt_to_response.setdefault(prompt_key, response_key)

        record_hash = stable_hash({"record_type": record_type, "prompt": prompt_key, "response": response_key})
        if record_hash in seen_record_hashes:
            stats["duplicates_dropped"] += 1
            continue
        seen_record_hashes.add(record_hash)

        record = {
            "id": f"{unit.unit_id}-{record_hash[:12]}",
            "record_type": record_type,
            "prompt": prompt,
            "response": response,
            "evidence": [
                {
                    "artifact": evidence.artifact,
                    "locator": evidence.locator,
                    "excerpt": evidence.excerpt,
                }
                for evidence in unit.evidence
            ],
            "source_artifacts": sorted(unit.source_artifacts),
            "confidence": confidence,
            "metadata": {
                "unit_id": unit.unit_id,
                "unit_type": unit.unit_type,
                "unit_title": unit.title,
                "signal_bucket": unit.signal_bucket,
                **(example.get("metadata") if isinstance(example.get("metadata"), dict) else {}),
            },
        }
        normalized.append(record)
    return normalized


def is_low_value_record(unit: GenerationUnit, prompt: str, response: str) -> bool:
    policy = active_policy()
    prompt_lower = prompt.lower()
    response_lower = response.lower()
    if unit.unit_type == "doc-page":
        if any(pattern in prompt_lower for pattern in policy.low_value_doc_patterns):
            return True
        if any(pattern in response_lower for pattern in policy.low_value_doc_patterns):
            return True
        if "repository" in prompt_lower and not any(
            token in prompt_lower for token in policy.doc_page_repository_context_tokens
        ):
            return True
    if unit.unit_type in {"dependency-module", "symbol-path"}:
        source_path = str(unit.prompt_context.get("source") or unit.prompt_context.get("path") or "").lower()
        if source_path and source_path in prompt_lower and any(
            pattern in response_lower for pattern in policy.low_value_response_patterns
        ):
            return True
    if any(sub in prompt_lower for sub in policy.trivial_prompt_substrings):
        return True
    if any(pattern in response_lower for pattern in policy.low_value_response_patterns):
        return True
    return False


def rebalance_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not records:
        return records
    policy = active_policy()
    total = len(records)
    cap_by_type = {
        record_type: max(4, math.ceil(total * ratio))
        for record_type, ratio in policy.record_type_cap_ratio.items()
    }

    def sort_key(record: dict[str, Any]) -> tuple[int, int, str, str, str]:
        meta = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        unit_type = str(meta.get("unit_type") or "")
        bucket = str(meta.get("signal_bucket") or "")
        return (
            unit_rebalance_tier(unit_type),
            -policy.bucket_priority.get(bucket, 0),
            str(record.get("record_type") or ""),
            str(record.get("prompt") or ""),
            str(record.get("id") or ""),
        )

    prioritized = sorted(records, key=sort_key)
    kept: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    plugin_rows = 0
    for record in prioritized:
        record_type = record["record_type"]
        cap = cap_by_type.get(record_type, total)
        if counts.get(record_type, 0) >= cap:
            continue
        meta = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        unit_type = str(meta.get("unit_type") or "")
        if (
            unit_type == policy.plugin_unit_type
            and plugin_rows >= policy.max_plugin_unit_records_in_corpus
        ):
            continue
        kept.append(record)
        counts[record_type] = counts.get(record_type, 0) + 1
        if unit_type == policy.plugin_unit_type:
            plugin_rows += 1
    return kept


def to_chat_sft_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record["id"],
        "messages": [
            {
                "role": "system",
                "content": (
                    "You answer questions about this repository using grounded facts from extracted code and documentation artifacts."
                ),
            },
            {"role": "user", "content": record["prompt"]},
            {"role": "assistant", "content": record["response"]},
        ],
        "metadata": {
            "record_type": record["record_type"],
            "confidence": record["confidence"],
            "source_artifacts": record["source_artifacts"],
            "evidence": record["evidence"],
            **record["metadata"],
        },
    }


def to_instruction_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record["id"],
        "prompt": record["prompt"],
        "completion": record["response"],
        "metadata": {
            "record_type": record["record_type"],
            "confidence": record["confidence"],
            "source_artifacts": record["source_artifacts"],
            **record["metadata"],
        },
    }


def project_eval_sample(records: list[dict[str, Any]], sample_size: int) -> list[dict[str, Any]]:
    if sample_size <= 0:
        return []
    by_type: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_type.setdefault(record["record_type"], []).append(record)
    selected: list[dict[str, Any]] = []
    per_type = max(1, math.ceil(sample_size / max(1, len(by_type))))
    for record_type in sorted(by_type.keys()):
        selected.extend(by_type[record_type][:per_type])
    return selected[:sample_size]


def write_report(
    path: Path,
    *,
    bundle: dict[str, Any],
    records: list[dict[str, Any]],
    units: list[GenerationUnit],
    helper_endpoint: str,
    helper_model: str,
    dry_run: bool,
    stats: dict[str, Any],
) -> None:
    by_record_type: dict[str, int] = {}
    for record in records:
        by_record_type[record["record_type"]] = by_record_type.get(record["record_type"], 0) + 1

    report = [
        "# Synthetic Training Data Report",
        "",
        f"- Source commit: `{bundle['summary'].get('commit', 'unknown')}`",
        f"- Source repo: `{bundle['summary'].get('repo', 'unknown')}`",
        f"- Helper endpoint: `{helper_endpoint}`",
        f"- Helper model: `{helper_model}`",
        f"- Dry run: `{str(dry_run).lower()}`",
        f"- Units processed: `{len(units)}`",
        f"- Units succeeded: `{stats['units_succeeded']}`",
        f"- Units failed: `{stats['units_failed']}`",
        f"- Records generated: `{stats['records_generated']}`",
        f"- Records after validation (pre-rebalance): `{stats['records_after_validation']}`",
        f"- Records kept (after rebalance): `{stats['records_kept']}`",
        f"- Rebalance dropped: `{stats['rebalance_dropped']}`",
        f"- Duplicates dropped: `{stats['duplicates_dropped']}`",
        f"- Contradictions dropped: `{stats['contradictions_dropped']}`",
        f"- Validation dropped: `{stats['validation_dropped']}`",
        "",
        "## Record Counts",
    ]
    report.extend([f"- `{record_type}`: `{count}`" for record_type, count in sorted(by_record_type.items())])
    if stats["unit_errors"]:
        report.append("")
        report.append("## Unit Errors")
        report.extend(
            [
                f"- `{error['unit_id']}`: {error['error']}"
                for error in stats["unit_errors"][:25]
            ]
        )
    path.write_text("\n".join(report) + "\n", encoding="utf-8")


def evidence_from_object(artifact: str, locator: str, payload: Any) -> EvidenceEntry:
    rendered = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    limit = active_policy().max_excerpt_chars
    excerpt = rendered[:limit]
    if len(rendered) > limit:
        excerpt += "..."
    return EvidenceEntry(artifact=artifact, locator=locator, excerpt=excerpt)


def github_api_json(url: str, token: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "openclaw-synthetic-training-data",
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def download_to_file(url: str, destination: Path, token: str) -> None:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "openclaw-synthetic-training-data",
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
        destination.write_bytes(response.read())


def http_post_json(url: str, payload: dict[str, Any], *, timeout_seconds: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:  # noqa: PERF203
        body = error.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Helper endpoint returned HTTP {error.code}: {body}") from error


def parse_json_block(response_text: str) -> Any:
    text = response_text.strip()
    if text.startswith("```"):
        fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "unit"


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).lower()


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def read_optional_json(path: Path) -> Any | None:
    if not path.is_file():
        return None
    return read_json(path)


def read_optional_text(path: Path) -> str | None:
    if not path.is_file():
        return None
    return read_text(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True))
            handle.write("\n")


if __name__ == "__main__":
    sys.exit(main())
