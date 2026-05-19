#!/usr/bin/env python3
"""Enforce the editorial-vs-claim-change PR boundary."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


EDITORIAL_FORBIDDEN_PREFIXES = (
    ".github/workflows/",
    "examples/",
    "include/",
    "paper/generated/",
    "paper/sources/",
    "results/",
    "scripts/",
    "src/",
    "tests/",
)

EDITORIAL_FORBIDDEN_FILES = {
    "paper/claims.yaml",
    "paper/data.json",
    "paper/data.typ",
    "paper/references.bib",
    "paper/references.md",
    "paper/related-work-insights.md",
}

PROSE_SUFFIXES = {".md", ".typ", ".tex"}
NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_])[-+]?(?:\d+\.\d+|\d+)(?:\s*(?:%|pp|ms|s|rows|tasks|cells))?")
CITATION_RE = re.compile(r"@(?!(?:fig|tab|sec|eq|lst|tbl):)[A-Za-z][A-Za-z0-9_:-]+")
GENERATED_BINDING_RE = re.compile(r"\b(?:claim_|paper_claim_)[A-Za-z0-9_]*")

CLAIM_CHANGE_FIELDS = (
    "Claim id",
    "Artifact path",
    "Regenerated command",
    "Before/after scope",
    "Interpretation rationale",
)


def run_git(args: list[str], cwd: Path, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"git {' '.join(args)} failed")
    return result.stdout


def rev_exists(rev: str, cwd: Path) -> bool:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}"],
        cwd=cwd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def load_event() -> dict:
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if not event_path:
        return {}
    try:
        with open(event_path, "r", encoding="utf-8") as fp:
            return json.load(fp)
    except (OSError, json.JSONDecodeError):
        return {}


def normalize_kind(value: str | None) -> str | None:
    if not value:
        return None
    key = value.strip().lower().replace("_", "-")
    if key in {"editorial", "editorial-pr", "prose"}:
        return "editorial"
    if key in {"claim-change", "claim change", "claim", "claims"}:
        return "claim-change"
    if key in {"engineering", "engineering-research", "research", "code"}:
        return "engineering"
    return None


def checkbox_kinds(body: str) -> list[str]:
    patterns = {
        "editorial": r"(?im)^\s*-\s*\[[xX]\]\s*Editorial PR\s*$",
        "claim-change": r"(?im)^\s*-\s*\[[xX]\]\s*Claim change PR\s*$",
        "engineering": r"(?im)^\s*-\s*\[[xX]\]\s*Engineering/research PR\s*$",
    }
    return [kind for kind, pattern in patterns.items() if re.search(pattern, body)]


def infer_pr_kind(args: argparse.Namespace, event: dict) -> tuple[str | None, list[str], str]:
    body = args.body or os.environ.get("PR_BODY", "")
    title = args.title or os.environ.get("PR_TITLE", "")
    labels_text = os.environ.get("PR_LABELS", "")

    pull_request = event.get("pull_request") or {}
    if pull_request:
        body = body or pull_request.get("body") or ""
        title = title or pull_request.get("title") or ""
        labels = pull_request.get("labels") or []
        labels_text = labels_text or ",".join(str(label.get("name", "")) for label in labels)

    explicit = normalize_kind(args.pr_type or os.environ.get("PR_TYPE"))
    if explicit:
        return explicit, [], body

    checked = checkbox_kinds(body)
    if len(checked) == 1:
        return checked[0], [], body
    if len(checked) > 1:
        return None, [f"Multiple PR type checkboxes are selected: {', '.join(checked)}."], body

    title_kind = normalize_kind(title.split(":", 1)[0] if ":" in title else None)
    if title_kind:
        return title_kind, [], body

    for label in re.split(r"[,;\n]", labels_text):
        label_kind = normalize_kind(label)
        if label_kind:
            return label_kind, [], body

    return None, [], body


def is_pull_request_event(event: dict) -> bool:
    return bool(event.get("pull_request")) or os.environ.get("GITHUB_EVENT_NAME") == "pull_request"


def resolve_diff_base(args: argparse.Namespace, event: dict, cwd: Path) -> str | None:
    if args.base:
        return args.base

    pull_request = event.get("pull_request") or {}
    base = pull_request.get("base") or {}
    candidates: list[str] = []
    if base.get("sha"):
        candidates.append(base["sha"])
    if os.environ.get("GITHUB_BASE_REF"):
        candidates.append(f"origin/{os.environ['GITHUB_BASE_REF']}")
        candidates.append(os.environ["GITHUB_BASE_REF"])
    if base.get("ref"):
        candidates.append(f"origin/{base['ref']}")
        candidates.append(base["ref"])

    for candidate in candidates:
        if rev_exists(candidate, cwd):
            return candidate
    return None


def changed_files(args: argparse.Namespace, event: dict, cwd: Path) -> tuple[list[str], list[str]]:
    if args.changed_file:
        return sorted(set(args.changed_file)), []

    base = resolve_diff_base(args, event, cwd)
    if not base:
        return [], ["Could not resolve PR base commit for editorial-boundary diff."]

    try:
        output = run_git(["diff", "--name-only", f"{base}...{args.head}"], cwd)
    except RuntimeError as exc:
        return [], [str(exc)]
    return [line.strip() for line in output.splitlines() if line.strip()], []


def changed_lines(base: str, head: str, path: str, cwd: Path) -> list[tuple[str, str]]:
    output = run_git(["diff", "--unified=0", "--no-ext-diff", f"{base}...{head}", "--", path], cwd)
    lines: list[tuple[str, str]] = []
    for line in output.splitlines():
        if line.startswith(("+++", "---")):
            continue
        if line.startswith("+"):
            lines.append(("+", line[1:]))
        elif line.startswith("-"):
            lines.append(("-", line[1:]))
    return lines


def editorial_failures(files: list[str], args: argparse.Namespace, event: dict, cwd: Path) -> list[str]:
    failures: list[str] = []
    for path in files:
        if path in EDITORIAL_FORBIDDEN_FILES:
            failures.append(f"Editorial PR touches forbidden claim/citation file: {path}")
        for prefix in EDITORIAL_FORBIDDEN_PREFIXES:
            if path.startswith(prefix):
                failures.append(f"Editorial PR touches forbidden path {prefix}: {path}")
                break

    base = resolve_diff_base(args, event, cwd)
    if args.changed_file or not base:
        return failures

    for path in files:
        if Path(path).suffix not in PROSE_SUFFIXES:
            continue
        try:
            diff_lines = changed_lines(base, args.head, path, cwd)
        except RuntimeError as exc:
            failures.append(str(exc))
            continue
        for sign, text in diff_lines:
            stripped = text.strip()
            if not stripped:
                continue
            if NUMBER_RE.search(stripped):
                failures.append(f"Editorial PR changes numeric text in {path}: {sign}{stripped}")
            if CITATION_RE.search(stripped) or "#cite" in stripped:
                failures.append(f"Editorial PR changes citation-like text in {path}: {sign}{stripped}")
            if GENERATED_BINDING_RE.search(stripped):
                failures.append(f"Editorial PR changes generated claim binding in {path}: {sign}{stripped}")

    return failures


def claim_change_failures(body: str) -> list[str]:
    failures: list[str] = []
    for field in CLAIM_CHANGE_FIELDS:
        pattern = re.compile(rf"(?im)^\s*-\s*\*\*{re.escape(field)}\*\*:\s*(.+?)\s*$")
        match = pattern.search(body)
        value = match.group(1).strip() if match else ""
        if not value or value.lower() in {"n/a", "na", "none", "tbd", "todo"}:
            failures.append(f"Claim change PR is missing required field: {field}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", help="base git revision for diff")
    parser.add_argument("--head", help="head git revision for diff", default="HEAD")
    parser.add_argument("--type", dest="pr_type", help="PR type: editorial, claim-change, or engineering")
    parser.add_argument("--title", help="PR title override")
    parser.add_argument("--body", help="PR body override")
    parser.add_argument("--changed-file", action="append", help="changed file override; may be repeated")
    args = parser.parse_args()

    cwd = Path.cwd()
    event = load_event()
    kind, kind_errors, body = infer_pr_kind(args, event)

    if not is_pull_request_event(event) and not args.pr_type:
        print("editorial boundary: no PR context; skipping")
        return 0

    failures = list(kind_errors)
    if kind is None:
        failures.append(
            "PR type is unknown; select exactly one PR Type checkbox or use an "
            "editorial:/claim-change:/engineering: title prefix."
        )

    files, file_errors = changed_files(args, event, cwd)
    failures.extend(file_errors)

    if kind == "editorial":
        failures.extend(editorial_failures(files, args, event, cwd))
    elif kind == "claim-change":
        failures.extend(claim_change_failures(body))

    if failures:
        print("editorial boundary: failed")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print(f"editorial boundary: ok ({kind}; {len(files)} changed files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
