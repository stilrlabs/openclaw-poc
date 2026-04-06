#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import yaml

REPOMIX_VERSION = "1.13.1"
DEPENDENCY_CRUISER_VERSION = "17.3.10"
DEFAULT_REPOMIX_STYLE = "markdown"
TREE_IGNORES = {".artifacts", ".git", "coverage", "dist", "node_modules"}
SYMBOL_SUMMARY_LANGUAGES = {"JavaScript", "TypeScript"}
SYMBOL_SUMMARY_KINDS = {
    "class",
    "constant",
    "enum",
    "function",
    "interface",
    "method",
    "module",
    "type",
    "variable",
}
EXTENSION_LANGUAGE_MAP = {
    ".cjs": "JavaScript",
    ".cts": "TypeScript",
    ".jsx": "JavaScript",
    ".js": "JavaScript",
    ".md": "Markdown",
    ".mdx": "MDX",
    ".mjs": "JavaScript",
    ".mts": "TypeScript",
    ".py": "Python",
    ".sh": "Shell",
    ".swift": "Swift",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".yaml": "YAML",
    ".yml": "YAML",
}


@dataclass(frozen=True)
class CommandResult:
    command: list[str]
    stdout: str
    stderr: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate raw code-signal artifacts.")
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for generated artifacts.",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root to analyze (default: current directory).",
    )
    parser.add_argument(
        "--repomix-style",
        default=DEFAULT_REPOMIX_STYLE,
        choices=["markdown", "json", "plain", "xml"],
        help="Repomix output style (default: markdown).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tracked_files = git_ls_files(repo_root)
    if not tracked_files:
        raise SystemExit("No tracked files found. Refusing to generate empty signal bundle.")

    generated_at = datetime.now(timezone.utc).isoformat()
    repo_sha = run_command(["git", "rev-parse", "HEAD"], cwd=repo_root).stdout.strip()

    write_text(
        out_dir / "repo-tree.md",
        render_repo_tree(repo_root, tracked_files),
    )

    repomix_extension = {
        "json": "json",
        "markdown": "md",
        "plain": "txt",
        "xml": "xml",
    }[args.repomix_style]
    repomix_path = out_dir / f"repomix-output.{repomix_extension}"
    run_repomix(repo_root, tracked_files, repomix_path, args.repomix_style)

    write_json(out_dir / "workspace-packages.json", build_workspace_inventory(repo_root))
    write_json(out_dir / "plugin-manifests.json", build_plugin_inventory(repo_root, tracked_files))
    write_json(out_dir / "workflow-inventory.json", build_workflow_inventory(repo_root, tracked_files))
    write_json(out_dir / "docs-inventory.json", build_docs_inventory(repo_root, tracked_files))
    write_json(out_dir / "language-summary.json", build_language_summary(repo_root, tracked_files))
    ctags_inventory = build_ctags_inventory(repo_root, tracked_files)
    write_json(out_dir / "tags.json", ctags_inventory)
    write_json(out_dir / "symbol-summary.json", summarize_ctags_inventory(ctags_inventory))
    write_json(out_dir / "ts-deps.json", build_dependency_cruiser_graph(repo_root))
    write_optional_json(
        out_dir / "ts-topology-owner-map.json",
        build_optional_json_command(
            repo_root,
            [
                "pnpm",
                "run",
                "--silent",
                "ts-topology",
                "--",
                "--scope=plugin-sdk",
                "--report=owner-map",
                "--json",
            ],
        ),
    )
    write_optional_json(
        out_dir / "ts-topology-consumer-topology.json",
        build_optional_json_command(
            repo_root,
            [
                "pnpm",
                "run",
                "--silent",
                "ts-topology",
                "--",
                "--scope=plugin-sdk",
                "--report=consumer-topology",
                "--json",
            ],
        ),
    )
    write_optional_json(
        out_dir / "seam-inventory.json",
        build_optional_json_command(repo_root, ["pnpm", "run", "--silent", "audit:seams"]),
    )
    write_text(
        out_dir / "summary.md",
        build_summary(
            repo_root=repo_root,
            out_dir=out_dir,
            repo_sha=repo_sha,
            generated_at=generated_at,
            repomix_style=args.repomix_style,
        ),
    )

    return 0


def run_command(
    command: list[str],
    *,
    cwd: Path,
    stdin: str | None = None,
    env: dict[str, str] | None = None,
    optional: bool = False,
) -> CommandResult:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            input=stdin,
            text=True,
            capture_output=True,
            env=merged_env,
            check=False,
        )
    except FileNotFoundError as error:
        if optional:
            return CommandResult(command, "", f"{error}")
        raise SystemExit(f"Missing required command '{command[0]}'.") from error
    if completed.returncode != 0 and not optional:
        raise SystemExit(
            "Command failed:\n"
            f"  {' '.join(command)}\n"
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    return CommandResult(command, completed.stdout, completed.stderr)


def git_ls_files(repo_root: Path) -> list[str]:
    result = run_command(["git", "ls-files", "-z"], cwd=repo_root)
    return sorted([path for path in result.stdout.split("\0") if path])


def render_repo_tree(repo_root: Path, tracked_files: list[str]) -> str:
    tree_command = shutil.which("tree")
    if tree_command:
        result = run_command(
            [
                tree_command,
                "-a",
                "--noreport",
                "--sort=name",
                "-I",
                "|".join(sorted(TREE_IGNORES)),
            ],
            cwd=repo_root,
            optional=True,
        )
        if result.stdout.strip():
            return "# Repository Tree\n\n```text\n" + result.stdout.rstrip() + "\n```\n"

    nested: dict[str, Any] = {}
    for path in tracked_files:
        parts = path.split("/")
        if any(part in TREE_IGNORES for part in parts):
            continue
        cursor = nested
        for part in parts:
            cursor = cursor.setdefault(part, {})

    lines = ["."]
    render_tree_lines(nested, lines, "")
    return "# Repository Tree\n\n```text\n" + "\n".join(lines) + "\n```\n"


def render_tree_lines(tree: dict[str, Any], lines: list[str], prefix: str) -> None:
    names = sorted(tree.keys())
    for index, name in enumerate(names):
        is_last = index == len(names) - 1
        branch = "└── " if is_last else "├── "
        lines.append(f"{prefix}{branch}{name}")
        child_prefix = f"{prefix}{'    ' if is_last else '│   '}"
        render_tree_lines(tree[name], lines, child_prefix)


def run_repomix(repo_root: Path, tracked_files: list[str], output_path: Path, style: str) -> None:
    run_command(
        [
            "pnpm",
            "dlx",
            f"repomix@{REPOMIX_VERSION}",
            "--stdin",
            "--style",
            style,
            "--output",
            str(output_path),
            "--no-git-sort-by-changes",
            "--quiet",
        ],
        cwd=repo_root,
        stdin="\n".join(tracked_files) + "\n",
    )


def build_workspace_inventory(repo_root: Path) -> dict[str, Any]:
    root_package = read_json(repo_root / "package.json")
    workspace_config = yaml.load(
        (repo_root / "pnpm-workspace.yaml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    raw_workspace_patterns = workspace_config.get("packages", [])
    workspace_patterns = sorted(
        pattern
        for pattern in raw_workspace_patterns
        if isinstance(pattern, str) and pattern.strip()
    )
    package_dirs: set[Path] = {repo_root}
    for pattern in workspace_patterns:
        if pattern in {".", "./"}:
            continue
        for match in sorted(repo_root.glob(pattern)):
            package_json = match / "package.json"
            if match.is_dir() and package_json.is_file():
                package_dirs.add(match)

    packages = []
    for package_dir in sorted(package_dirs):
        package_json = package_dir / "package.json"
        if not package_json.is_file():
            continue
        package = read_json(package_json)
        packages.append(
            {
                "path": str(package_dir.relative_to(repo_root)).replace("\\", "/") or ".",
                "name": package.get("name"),
                "version": package.get("version"),
                "private": bool(package.get("private", False)),
            }
        )

    return {
        "root": {
            "name": root_package.get("name"),
            "version": root_package.get("version"),
            "packageManager": root_package.get("packageManager"),
        },
        "workspacePatterns": workspace_patterns,
        "packages": packages,
    }


def build_plugin_inventory(repo_root: Path, tracked_files: list[str]) -> dict[str, Any]:
    plugin_paths = sorted(
        path for path in tracked_files if re.fullmatch(r"extensions/[^/]+/openclaw\.plugin\.json", path)
    )
    return {
        "plugins": [
            {
                "path": plugin_path,
                "manifest": read_json(repo_root / plugin_path),
            }
            for plugin_path in plugin_paths
        ]
    }


def build_workflow_inventory(repo_root: Path, tracked_files: list[str]) -> dict[str, Any]:
    workflow_paths = sorted(
        path
        for path in tracked_files
        if path.startswith(".github/workflows/") and path.endswith((".yml", ".yaml"))
    )
    workflows = []
    for workflow_path in workflow_paths:
        raw = yaml.load((repo_root / workflow_path).read_text(encoding="utf-8"), Loader=yaml.BaseLoader) or {}
        jobs = raw.get("jobs", {}) if isinstance(raw, dict) else {}
        normalized_jobs = []
        for job_id in sorted(jobs.keys()):
            job = jobs[job_id] or {}
            steps = job.get("steps", []) if isinstance(job, dict) else []
            normalized_jobs.append(
                {
                    "id": job_id,
                    "name": job.get("name", job_id),
                    "runsOn": job.get("runs-on"),
                    "needs": normalize_list(job.get("needs")),
                    "if": job.get("if"),
                    "stepNames": [
                        step.get("name")
                        for step in steps
                        if isinstance(step, dict) and step.get("name")
                    ],
                }
            )
        workflows.append(
            {
                "path": workflow_path,
                "name": raw.get("name"),
                "triggers": raw.get("on"),
                "jobCount": len(normalized_jobs),
                "jobs": normalized_jobs,
            }
        )
    return {"workflows": workflows}


def build_docs_inventory(repo_root: Path, tracked_files: list[str]) -> dict[str, Any]:
    docs_paths = sorted(path for path in tracked_files if path.startswith("docs/") and path.endswith((".md", ".mdx")))
    pages = []
    link_re = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
    fence_re = re.compile(r"^```([^\s`]+)?", re.MULTILINE)
    heading_re = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    for docs_path in docs_paths:
        text = (repo_root / docs_path).read_text(encoding="utf-8")
        headings = [
            {
                "depth": len(match.group(1)),
                "text": match.group(2).strip(),
            }
            for match in heading_re.finditer(text)
        ]
        links = [
            {
                "label": match.group(1),
                "target": match.group(2),
            }
            for match in link_re.finditer(text)
        ]
        pages.append(
            {
                "path": docs_path,
                "title": headings[0]["text"] if headings else None,
                "headings": headings,
                "links": links,
                "codeFenceLanguages": sorted(
                    {
                        match.group(1).strip()
                        for match in fence_re.finditer(text)
                        if match.group(1) and match.group(1).strip()
                    }
                ),
            }
        )
    return {"pages": pages}


def build_language_summary(repo_root: Path, tracked_files: list[str]) -> dict[str, Any]:
    cloc_command = shutil.which("cloc")
    if cloc_command:
        with NamedTemporaryFile("w", encoding="utf-8", delete=False) as file_list:
            file_list.write("\n".join(tracked_files) + "\n")
            file_list_path = Path(file_list.name)
        try:
            result = run_command(
                [cloc_command, "--json", f"--list-file={file_list_path}"],
                cwd=repo_root,
            )
        finally:
            file_list_path.unlink(missing_ok=True)
        return normalize_cloc_output(json.loads(result.stdout))

    counts: dict[str, dict[str, int]] = {}
    for relative_path in tracked_files:
        extension = Path(relative_path).suffix.lower()
        language = EXTENSION_LANGUAGE_MAP.get(extension, "Other")
        bucket = counts.setdefault(language, {"files": 0, "lines": 0})
        bucket["files"] += 1
        bucket["lines"] += sum(
            1 for _ in (repo_root / relative_path).read_text(encoding="utf-8", errors="ignore").splitlines()
        )
    languages = [
        {"language": language, **counts[language]}
        for language in sorted(counts.keys())
    ]
    return {"source": "fallback", "languages": languages}


def normalize_cloc_output(raw: dict[str, Any]) -> dict[str, Any]:
    normalized_languages = []
    for language in sorted(key for key in raw.keys() if key not in {"header", "SUM"}):
        stats = raw[language] or {}
        normalized_languages.append(
            {
                "language": language,
                "files": stats.get("nFiles"),
                "blank": stats.get("blank"),
                "comment": stats.get("comment"),
                "code": stats.get("code"),
            }
        )
    return {
        "source": "cloc",
        "header": raw.get("header"),
        "sum": raw.get("SUM"),
        "languages": normalized_languages,
    }


def build_ctags_inventory(repo_root: Path, tracked_files: list[str]) -> dict[str, Any]:
    ctags_command = shutil.which("ctags")
    if not ctags_command:
        return {"source": "missing-ctags", "tags": []}

    with NamedTemporaryFile("w", encoding="utf-8", delete=False) as file_list:
        file_list.write("\n".join(tracked_files) + "\n")
        file_list_path = Path(file_list.name)
    try:
        result = run_command(
            [
                ctags_command,
                "--extras=+p",
                "--fields=*",
                "--output-format=json",
                "-L",
                str(file_list_path),
            ],
            cwd=repo_root,
        )
    finally:
        file_list_path.unlink(missing_ok=True)

    tags = []
    pseudo_tags = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get("_type") == "ptag":
            pseudo_tags.append(entry)
        else:
            tags.append(entry)
    tags.sort(key=lambda tag: (tag.get("path", ""), tag.get("line", 0), tag.get("kind", ""), tag.get("name", "")))
    pseudo_tags.sort(key=lambda tag: (tag.get("name", ""), tag.get("path", "")))
    return {"source": "universal-ctags", "pseudoTags": pseudo_tags, "tags": tags}


def summarize_ctags_inventory(inventory: dict[str, Any]) -> dict[str, Any]:
    tags = inventory.get("tags", [])
    filtered = []
    for tag in tags:
        if not isinstance(tag, dict):
            continue
        language = tag.get("language")
        kind = tag.get("kind")
        path = tag.get("path")
        if language not in SYMBOL_SUMMARY_LANGUAGES:
            continue
        if kind not in SYMBOL_SUMMARY_KINDS:
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
    important_paths = sorted(
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
    )[:24]
    important_symbols = filtered[:120]
    return {
        "source": "curated-from-tags",
        "importantPaths": important_paths,
        "importantSymbols": important_symbols,
    }


def build_dependency_cruiser_graph(repo_root: Path) -> dict[str, Any]:
    result = run_command(
        [
            "pnpm",
            "dlx",
            f"dependency-cruiser@{DEPENDENCY_CRUISER_VERSION}",
            "--no-config",
            "--ts-config",
            "tsconfig.json",
            "--exclude",
            "(^|/)(node_modules|dist|coverage|\\.artifacts)(/|$)",
            "--do-not-follow",
            "(^|/)(node_modules|dist|coverage|\\.artifacts)(/|$)",
            "--output-type",
            "json",
            "src",
            "extensions",
            "packages",
            "ui",
            "scripts",
        ],
        cwd=repo_root,
    )
    raw = json.loads(result.stdout)
    return {
        "source": "dependency-cruiser",
        "cruiseResult": raw,
    }


def build_optional_json_command(repo_root: Path, command: list[str]) -> dict[str, Any] | None:
    result = run_command(command, cwd=repo_root, optional=True)
    if not result.stdout.strip():
        return None
    return json.loads(result.stdout)


def build_summary(
    *,
    repo_root: Path,
    out_dir: Path,
    repo_sha: str,
    generated_at: str,
    repomix_style: str,
) -> str:
    relative_out_dir = out_dir.relative_to(repo_root)
    files = sorted(path for path in out_dir.iterdir() if path.is_file())
    file_lines = [
        f"- `{relative_out_dir / path.name}` ({path.stat().st_size} bytes)"
        for path in files
    ]
    return (
        "# Code Signals Summary\n\n"
        f"- Repo: `{repo_root.name}`\n"
        f"- Commit: `{repo_sha}`\n"
        f"- Generated at: `{generated_at}`\n"
        f"- Repomix style: `{repomix_style}`\n"
        f"- Repomix version: `{REPOMIX_VERSION}`\n"
        f"- Dependency-cruiser version: `{DEPENDENCY_CRUISER_VERSION}`\n\n"
        "## Generated Files\n"
        + "\n".join(file_lines)
        + "\n"
    )


def normalize_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_optional_json(path: Path, payload: Any) -> None:
    if payload is None:
        return
    write_json(path, payload)


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
