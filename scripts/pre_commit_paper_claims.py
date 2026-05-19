#!/usr/bin/env python3
"""Fast pre-commit gate for paper claim inputs.

This hook intentionally runs only the deterministic claim compiler. It does not
invoke an LLM proposer. If the compiler changes generated paper artifacts, the
commit should stop so the author can review and stage those artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


STATIC_INPUTS = {
    "Makefile",
    "paper/claims.yaml",
    "paper/paper.typ",
    "scripts/derive_paper_claims.py",
}

INPUT_PREFIXES = (
    "paper/agent_candidates/",
)

GENERATED_OUTPUTS = (
    "paper/generated/agent_instructions.md",
    "paper/generated/claims.json",
    "paper/generated/claims.typ",
    "paper/generated/gap_report.md",
    "paper/generated/limitations.md",
    "paper/generated/paper_agent_trace.jsonl",
    "paper/generated/paper_agent_trace.md",
    "paper/generated/roadmap_links.md",
    "paper/generated/support_report.md",
)

GENERATED_PREFIXES = (
    "paper/generated/",
)


def run_git(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def run_git_bytes(args: list[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def repo_root() -> Path:
    result = run_git(["rev-parse", "--show-toplevel"])
    if result.returncode != 0:
        raise SystemExit(result.stderr.strip() or "not inside a git repository")
    return Path(result.stdout.strip())


def staged_paths() -> list[str]:
    result = run_git(["diff", "--cached", "--name-only", "--diff-filter=ACMR"])
    if result.returncode != 0:
        raise SystemExit(result.stderr.strip() or "failed to inspect staged paths")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def normalize_path(path: str) -> str:
    return path.strip().lstrip("./")


def load_registry_sources(root: Path) -> set[str]:
    registry_path = root / "paper/claims.yaml"
    if not registry_path.exists():
        return set()
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return set()

    sources: set[str] = set()

    def add_path(value: Any) -> None:
        if not isinstance(value, str):
            return
        path = normalize_path(value)
        if path and not any(path.startswith(prefix) for prefix in GENERATED_PREFIXES):
            sources.add(path)

    def walk(value: Any, key: str | None = None) -> None:
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                walk(child_value, str(child_key))
            return
        if isinstance(value, list):
            for item in value:
                walk(item, key)
            return
        if key in {"source", "secondary_sources", "artifacts"}:
            add_path(value)

    walk(registry)
    return sources


def should_run(paths: list[str], watched_sources: set[str]) -> bool:
    for path in paths:
        normalized = normalize_path(path)
        if normalized in STATIC_INPUTS or normalized in watched_sources:
            return True
        if any(normalized.startswith(prefix) for prefix in INPUT_PREFIXES):
            return True
    return False


def scrub_volatile_claim_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: scrub_volatile_claim_fields(child)
            for key, child in value.items()
            if key not in {"generated_at", "modified_utc"}
        }
    if isinstance(value, list):
        return [scrub_volatile_claim_fields(item) for item in value]
    return value


def digest_bytes(path_name: str, data: bytes) -> str:
    if Path(path_name).name == "claims.json":
        try:
            payload = scrub_volatile_claim_fields(json.loads(data.decode("utf-8")))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return hashlib.sha256(data).hexdigest()
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    return hashlib.sha256(data).hexdigest()


def index_file_digest(path: str) -> str | None:
    result = run_git_bytes(["show", f":{path}"])
    if result.returncode != 0:
        return None
    return digest_bytes(path, result.stdout)


def file_digest(path: Path) -> str | None:
    if not path.exists():
        return None
    return digest_bytes(path.name, path.read_bytes())


def output_digests(root: Path) -> dict[str, str | None]:
    return {path: file_digest(root / path) for path in GENERATED_OUTPUTS}


def changed_outputs(before: dict[str, str | None], after: dict[str, str | None]) -> list[str]:
    return [path for path in GENERATED_OUTPUTS if before.get(path) != after.get(path)]


def unstaged_generated_outputs(root: Path) -> list[str]:
    return [
        path
        for path in GENERATED_OUTPUTS
        if file_digest(root / path) != index_file_digest(path)
    ]


def run_claim_compiler(root: Path) -> int:
    return subprocess.run(
        [sys.executable, "scripts/derive_paper_claims.py"],
        cwd=root,
        check=False,
    ).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="run even when no staged input changed")
    parser.add_argument(
        "--no-generated-diff-check",
        action="store_true",
        help="do not fail if generated paper artifacts changed",
    )
    args = parser.parse_args()

    root = repo_root()
    paths = staged_paths()
    watched_sources = load_registry_sources(root)
    if not args.force and not should_run(paths, watched_sources):
        print("paper claim pre-commit: no staged paper claim inputs changed")
        return 0

    before = output_digests(root)
    code = run_claim_compiler(root)
    if code != 0:
        return code

    if args.no_generated_diff_check:
        return 0

    after = output_digests(root)
    changed = changed_outputs(before, after)
    if changed:
        print("paper claim pre-commit: generated artifacts changed; review and stage them:", file=sys.stderr)
        for path in changed:
            print(f"  {path}", file=sys.stderr)
        return 1
    unstaged = unstaged_generated_outputs(root)
    if unstaged:
        print("paper claim pre-commit: generated artifacts have unstaged changes; stage them:", file=sys.stderr)
        for path in unstaged:
            print(f"  {path}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
