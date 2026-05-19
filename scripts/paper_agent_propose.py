#!/usr/bin/env python3
"""Prepare or run the explicit LLM paper-agent proposal pass.

By default this script writes a prompt built from rejected paper-agent rows and
accepted agent briefs. If PAPER_AGENT_LLM_CMD or --llm-command is provided, it
runs that command with the prompt on stdin and normalizes JSONL candidate rows
to the paper-agent input schema.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


SCHEMA = "crlp.paper_agent_candidate.v1"
TRACE_VERSION = 1


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text.startswith("{"):
                rows.append(json.loads(text))
    return rows


def compact_refs(row: dict[str, Any]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for ref in row.get("source_refs", []):
        if not isinstance(ref, dict):
            continue
        refs.append(
            {
                "path": ref.get("path"),
                "exists": ref.get("exists"),
                "sha256": ref.get("sha256"),
                "row_count": ref.get("row_count"),
            }
        )
    return refs


def build_prompt(rows: list[dict[str, Any]], model: str) -> str:
    rejected = [row for row in rows if row.get("verifier_status") == "rejected"]
    briefs = [
        row for row in rows
        if row.get("candidate_type") == "agent_brief" and row.get("verifier_status") == "accepted"
    ]
    supported = [
        row for row in rows
        if row.get("candidate_type") == "claim_sentence" and
        row.get("verifier_status") == "accepted" and
        row.get("verifier", {}).get("support_status") == "supported"
    ]

    payload = {
        "task": "propose_next_paper_agent_candidates",
        "schema": SCHEMA,
        "trace_version": TRACE_VERSION,
        "model": model,
        "output_contract": {
            "format": "jsonl",
            "required_fields": [
                "schema",
                "trace_version",
                "candidate_type",
                "proposer_kind",
                "proposer_model",
                "prompt_hash",
                "proposed_text",
                "claim_ids",
                "paper_surfaces",
                "source_refs",
            ],
            "candidate_type": "claim_sentence",
            "proposer_kind": "llm",
        },
        "constraints": [
            "Do not assert unsupported or contradicted claims as paper claims.",
            "Use current source_refs exactly when citing an artifact-backed claim.",
            "Prefer narrower replacement claims or limitation/roadmap language when feedback says unsupported.",
            "Return JSONL only, one JSON object per line.",
        ],
        "rejected_candidates": [
            {
                "candidate_id": row.get("candidate_id"),
                "claim_ids": row.get("claim_ids", []),
                "paper_surfaces": row.get("paper_surfaces", []),
                "proposed_text": row.get("proposed_text", ""),
                "rejection_reason": row.get("rejection_reason"),
                "residual_feedback": row.get("residual_feedback", []),
                "source_refs": compact_refs(row),
            }
            for row in rejected
        ],
        "accepted_agent_briefs": [
            {
                "candidate_id": row.get("candidate_id"),
                "claim_ids": row.get("claim_ids", []),
                "brief": row.get("proposed_text", ""),
                "source_refs": compact_refs(row),
            }
            for row in briefs
        ],
        "supported_claim_surfaces": [
            {
                "claim_ids": row.get("claim_ids", []),
                "paper_surfaces": row.get("paper_surfaces", []),
                "source_refs": compact_refs(row),
            }
            for row in supported
        ],
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def prompt_hash(prompt: str) -> str:
    return "sha256:" + hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def run_llm(command: str, prompt: str) -> str:
    argv = shlex.split(command)
    if not argv:
        raise SystemExit("empty LLM command")
    result = subprocess.run(
        argv,
        input=prompt,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        if result.stderr:
            sys.stderr.write(result.stderr)
        raise SystemExit(result.returncode)
    return result.stdout


def normalize_candidate_rows(raw: str, model: str, prompt_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(raw.splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        try:
            row = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"LLM output line {line_no} is not JSON: {exc.msg}") from exc
        if not isinstance(row, dict):
            raise SystemExit(f"LLM output line {line_no} is not a JSON object")
        row.setdefault("schema", SCHEMA)
        row.setdefault("trace_version", TRACE_VERSION)
        row.setdefault("candidate_type", "claim_sentence")
        row.setdefault("proposer_kind", "llm")
        row.setdefault("proposer_model", model)
        row.setdefault("prompt_hash", prompt_id)
        rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=Path("paper/generated/paper_agent_trace.jsonl"))
    parser.add_argument("--prompt-output", type=Path, default=Path("paper/agent_candidates/next_prompt.md"))
    parser.add_argument("--output", type=Path, default=Path("paper/agent_candidates/llm_proposals.jsonl"))
    parser.add_argument("--model", default=os.environ.get("PAPER_AGENT_MODEL", "manual-llm"))
    parser.add_argument("--llm-command", default=os.environ.get("PAPER_AGENT_LLM_CMD", ""))
    args = parser.parse_args()

    rows = read_jsonl(args.trace)
    prompt = build_prompt(rows, args.model)
    prompt_id = prompt_hash(prompt)
    args.prompt_output.parent.mkdir(parents=True, exist_ok=True)
    args.prompt_output.write_text(
        "# Paper Agent Proposal Prompt\n\n"
        f"- Prompt hash: `{prompt_id}`\n"
        f"- Source trace: `{args.trace}`\n\n"
        "```json\n"
        f"{prompt}```\n",
        encoding="utf-8",
    )

    if not args.llm_command:
        print(f"wrote {args.prompt_output}")
        print("set PAPER_AGENT_LLM_CMD to generate JSONL candidate rows")
        return 0

    raw = run_llm(args.llm_command, prompt)
    candidates = normalize_candidate_rows(raw, args.model, prompt_id)
    write_jsonl(args.output, candidates)
    print(f"wrote {args.output} rows={len(candidates)} prompt_hash={prompt_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
