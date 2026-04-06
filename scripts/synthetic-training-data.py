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
DEFAULT_ENDPOINT = "http://127.0.0.1:11434/api/generate"
DEFAULT_MODEL = "synthetic-data-helper"
DEFAULT_MIN_RECORDS = 25
DEFAULT_EVAL_SAMPLE_SIZE = 50
MAX_EXCERPT_CHARS = 320
DEFAULT_ARTIFACT_PREFIX = "code-signals-"


@dataclass(frozen=True)
class EvidenceEntry:
    artifact: str
    locator: str
    excerpt: str


@dataclass(frozen=True)
class GenerationUnit:
    unit_id: str
    unit_type: str
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
    generate.add_argument("--max-plugin-units", default=24, type=int, help="Maximum plugin units.")
    generate.add_argument("--max-workflow-units", default=12, type=int, help="Maximum workflow units.")
    generate.add_argument("--max-doc-units", default=24, type=int, help="Maximum docs units.")
    generate.add_argument("--max-dependency-units", default=24, type=int, help="Maximum dependency units.")
    generate.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate deterministic placeholder records without calling the helper endpoint.",
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
    input_dir = Path(args.input_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_bundle(input_dir)
    units = build_generation_units(
        bundle,
        max_plugin_units=args.max_plugin_units,
        max_workflow_units=args.max_workflow_units,
        max_doc_units=args.max_doc_units,
        max_dependency_units=args.max_dependency_units,
    )
    records: list[dict[str, Any]] = []
    stats = {
        "units_total": len(units),
        "units_succeeded": 0,
        "units_failed": 0,
        "records_generated": 0,
        "records_kept": 0,
        "duplicates_dropped": 0,
        "contradictions_dropped": 0,
        "validation_dropped": 0,
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
            stats["records_kept"] += len(normalized)
            records.extend(normalized)
        except Exception as error:  # noqa: BLE001
            stats["units_failed"] += 1
            stats["unit_errors"].append({"unit_id": unit.unit_id, "error": str(error)})

    if len(records) < args.min_records:
        raise SystemExit(
            f"Generated only {len(records)} records, below the minimum required {args.min_records}."
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


def load_bundle(input_dir: Path) -> dict[str, Any]:
    bundle: dict[str, Any] = {"input_dir": input_dir}
    bundle["summary_markdown"] = read_text(input_dir / "summary.md")
    bundle["summary"] = parse_summary(bundle["summary_markdown"])
    bundle["workspace_packages"] = read_json(input_dir / "workspace-packages.json")
    bundle["plugin_manifests"] = read_json(input_dir / "plugin-manifests.json")
    bundle["workflow_inventory"] = read_json(input_dir / "workflow-inventory.json")
    bundle["docs_inventory"] = read_json(input_dir / "docs-inventory.json")
    bundle["language_summary"] = read_json(input_dir / "language-summary.json")
    bundle["ts_deps"] = read_json(input_dir / "ts-deps.json")
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


def build_generation_units(
    bundle: dict[str, Any],
    *,
    max_plugin_units: int,
    max_workflow_units: int,
    max_doc_units: int,
    max_dependency_units: int,
) -> list[GenerationUnit]:
    units: list[GenerationUnit] = []
    units.extend(build_repo_summary_units(bundle))
    units.extend(build_plugin_units(bundle, max_plugin_units))
    units.extend(build_workflow_units(bundle, max_workflow_units))
    units.extend(build_doc_units(bundle, max_doc_units))
    units.extend(build_dependency_units(bundle, max_dependency_units))
    return units


def build_repo_summary_units(bundle: dict[str, Any]) -> list[GenerationUnit]:
    packages = bundle["workspace_packages"]["packages"]
    languages = bundle["language_summary"]["languages"]
    summary_fields = bundle["summary"]
    repo_unit = GenerationUnit(
        unit_id="repo-summary",
        unit_type="repo-summary",
        title="Repository summary",
        prompt_context={
            "commit": summary_fields.get("commit"),
            "repo": summary_fields.get("repo"),
            "workspace_package_count": len(packages),
            "top_languages": languages[:8],
            "generated_files": summary_fields,
        },
        source_artifacts=["summary.md", "workspace-packages.json", "language-summary.json"],
        evidence=[
            evidence_from_object("summary.md", "frontmatter", summary_fields),
            evidence_from_object("workspace-packages.json", "packages", packages[:8]),
            evidence_from_object("language-summary.json", "languages", languages[:8]),
        ],
        preferred_record_types=["repo-facts", "architecture-explanations"],
        example_budget=4,
    )

    package_unit = GenerationUnit(
        unit_id="workspace-packages-overview",
        unit_type="workspace-overview",
        title="Workspace packages overview",
        prompt_context={
            "package_count": len(packages),
            "sample_packages": packages[:20],
        },
        source_artifacts=["workspace-packages.json"],
        evidence=[evidence_from_object("workspace-packages.json", "packages", packages[:20])],
        preferred_record_types=["repo-facts", "repo-navigation"],
        example_budget=3,
    )
    return [repo_unit, package_unit]


def build_plugin_units(bundle: dict[str, Any], limit: int) -> list[GenerationUnit]:
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
                "configSchemaProperties": sorted((manifest.get("configSchema", {}).get("properties") or {}).keys()),
                "providerAuthChoices": [
                    choice.get("choiceId")
                    for choice in manifest.get("providerAuthChoices", [])
                    if isinstance(choice, dict) and choice.get("choiceId")
                ],
            }
        )

    selected = sorted(
        normalized_plugins,
        key=lambda plugin: (
            -len(plugin["contracts"]),
            -len(plugin["configSchemaProperties"]),
            str(plugin["id"] or ""),
        ),
    )[:limit]

    units = []
    for index, plugin in enumerate(selected):
        units.append(
            GenerationUnit(
                unit_id=f"plugin-{plugin['id'] or index}",
                unit_type="plugin",
                title=f"Plugin {plugin['id']}",
                prompt_context=plugin,
                source_artifacts=["plugin-manifests.json"],
                evidence=[evidence_from_object("plugin-manifests.json", f"plugins[{index}]", plugin)],
                preferred_record_types=["repo-facts", "repo-navigation", "architecture-explanations"],
                example_budget=3,
            )
        )
    return units


def build_workflow_units(bundle: dict[str, Any], limit: int) -> list[GenerationUnit]:
    workflows = bundle["workflow_inventory"]["workflows"]
    selected = sorted(
        workflows,
        key=lambda workflow: (-int(workflow.get("jobCount", 0)), str(workflow.get("name") or "")),
    )[:limit]
    units = []
    for index, workflow in enumerate(selected):
        context = {
            "name": workflow.get("name"),
            "path": workflow.get("path"),
            "triggers": workflow.get("triggers"),
            "jobCount": workflow.get("jobCount"),
            "jobs": workflow.get("jobs", [])[:12],
        }
        units.append(
            GenerationUnit(
                unit_id=f"workflow-{slugify(workflow.get('name') or workflow.get('path') or str(index))}",
                unit_type="workflow",
                title=f"Workflow {workflow.get('name')}",
                prompt_context=context,
                source_artifacts=["workflow-inventory.json"],
                evidence=[evidence_from_object("workflow-inventory.json", f"workflows[{index}]", context)],
                preferred_record_types=["repo-facts", "repo-navigation", "architecture-explanations"],
                example_budget=3,
            )
        )
    return units


def build_doc_units(bundle: dict[str, Any], limit: int) -> list[GenerationUnit]:
    pages = bundle["docs_inventory"]["pages"]
    selected = sorted(
        pages,
        key=lambda page: (
            -(len(page.get("headings", [])) + len(page.get("links", []))),
            str(page.get("path") or ""),
        ),
    )[:limit]
    units = []
    for index, page in enumerate(selected):
        context = {
            "path": page.get("path"),
            "title": page.get("title"),
            "headings": page.get("headings", [])[:12],
            "links": page.get("links", [])[:12],
            "codeFenceLanguages": page.get("codeFenceLanguages", []),
        }
        units.append(
            GenerationUnit(
                unit_id=f"doc-{slugify(page.get('path') or str(index))}",
                unit_type="doc-page",
                title=f"Doc page {page.get('path')}",
                prompt_context=context,
                source_artifacts=["docs-inventory.json"],
                evidence=[evidence_from_object("docs-inventory.json", f"pages[{index}]", context)],
                preferred_record_types=["doc-grounded-qa", "repo-navigation", "architecture-explanations"],
                example_budget=2,
            )
        )
    return units


def build_dependency_units(bundle: dict[str, Any], limit: int) -> list[GenerationUnit]:
    modules = bundle["ts_deps"]["cruiseResult"]["modules"]
    candidate_modules = []
    for module in modules:
        source = module.get("source")
        if not isinstance(source, str) or "/" not in source:
            continue
        dependencies = module.get("dependencies", [])
        dependents = module.get("dependents", [])
        candidate_modules.append(
            {
                "source": source,
                "dependencyCount": len(dependencies),
                "dependentCount": len(dependents),
                "dependencies": dependencies[:8],
                "dependents": dependents[:8],
                "orphan": module.get("orphan"),
            }
        )

    selected = sorted(
        candidate_modules,
        key=lambda module: (-module["dependentCount"], -module["dependencyCount"], module["source"]),
    )[:limit]

    units = []
    for index, module in enumerate(selected):
        units.append(
            GenerationUnit(
                unit_id=f"dep-{slugify(module['source'])}",
                unit_type="dependency-module",
                title=f"Dependency module {module['source']}",
                prompt_context=module,
                source_artifacts=["ts-deps.json"],
                evidence=[evidence_from_object("ts-deps.json", f"cruiseResult.modules[{index}]", module)],
                preferred_record_types=["dependency-reasoning", "architecture-explanations", "repo-navigation"],
                example_budget=2,
            )
        )
    return units


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
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": helper_temperature},
    }
    response = http_post_json(helper_endpoint, payload, timeout_seconds=helper_timeout_seconds)
    response_text = response.get("response", "")
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
    return textwrap.dedent(
        f"""
        You are generating grounded synthetic training-data examples for repository understanding.

        Constraints:
        - Use only the supplied evidence and context.
        - Do not invent files, symbols, packages, plugins, workflows, or relationships.
        - Keep outputs concise, factual, and machine-parseable.
        - Return JSON only.
        - Produce at most {unit.example_budget} examples.
        - Allowed record_type values: {sorted(unit.preferred_record_types)}.
        - Allowed confidence values: ["high", "medium", "low"].
        - Do not include evidence or source_artifacts fields; they will be attached later.
        - Favor examples that improve a model's factual competence about this repository.

        Unit:
        {json.dumps({
            "unit_id": unit.unit_id,
            "unit_type": unit.unit_type,
            "title": unit.title,
            "preferred_record_types": unit.preferred_record_types,
            "prompt_context": unit.prompt_context,
            "evidence": [
                {
                    "artifact": evidence.artifact,
                    "locator": evidence.locator,
                    "excerpt": evidence.excerpt,
                }
                for evidence in unit.evidence
            ],
        }, indent=2, sort_keys=True)}

        Output schema:
        {json.dumps(schema, indent=2)}
        """
    ).strip()


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
                **(example.get("metadata") if isinstance(example.get("metadata"), dict) else {}),
            },
        }
        normalized.append(record)
    return normalized


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
        f"- Records kept: `{stats['records_kept']}`",
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
    excerpt = rendered[:MAX_EXCERPT_CHARS]
    if len(rendered) > MAX_EXCERPT_CHARS:
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True))
            handle.write("\n")


if __name__ == "__main__":
    sys.exit(main())
