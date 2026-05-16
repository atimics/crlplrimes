#!/usr/bin/env python3
"""Phased matched-budget experiments for the unified-scorer proposal.

This script is the experiment-side companion to docs/unified-scorer-proposal.md.
It assumes the engine agent has landed the `--proposer unified` selection
strategy in `prime_rule_search` and the `--unified-scorer-hidden` flag. If the
engine work has not landed yet, individual invocations will fail with the
binary's own error message; the per-phase snapshot will record the failure so
the run can be re-tried after the engine pieces merge.

Usage:
    python3 scripts/unified_scorer_experiments.py --phase 1
    python3 scripts/unified_scorer_experiments.py --phase 2
    python3 scripts/unified_scorer_experiments.py --phase 3
    python3 scripts/unified_scorer_experiments.py --phase all

Outputs land under results/unified_scorer/phase_<N>/:

    raw.jsonl        machine-readable rows from every binary invocation
    summary.csv      one row per (grammar, proposer, hidden, subsample) cell
    summary.md       comparison tables matching the proposal's reporting schema
    commands.txt     exact CLI invocations
    logs/*.log       full stdout+stderr per invocation

Conventions follow scripts/neurosymbolic_tuning.py and
scripts/capture_results.py so the unified-scorer rows are interoperable with
the existing `make results` flow once the engine work lands.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
RULE_SEARCH_SCHEMA = "crlp.rule_search.summary.v1"
DEFAULT_RANDOM_SEED = 20240511
PHASE_OUTPUT_ROOT = REPO_ROOT / "results" / "unified_scorer"

# Per-phase defaults. Override at the CLI.
PHASE_DEFAULTS: dict[int, dict[str, object]] = {
    1: {
        "hidden_sizes": ["64x32"],
        "marginal_states": [4096],
        "label": "Baseline parity",
        "purpose": "Unified scorer should land within ±1pp of the dedicated closed-loop at hidden=64x32.",
    },
    2: {
        "hidden_sizes": ["64x32", "128x64", "256x128"],
        "marginal_states": [4096],
        "label": "Capacity scaling",
        "purpose": "Sweep hidden capacity; compare unified vs closed-loop at each size.",
    },
    3: {
        "hidden_sizes": ["128x64"],
        "marginal_states": [64, 256, 4096],
        "label": "Data-sparse target",
        "purpose": "Data-sparse marginal-utility target. Unified scorer should beat greedy-on-subsample.",
    },
}

GRAMMAR_SPECS: dict[str, list[str]] = {
    "unconditional": ["--candidate-only", "--unconditional-only", "--moduli-range", "3:255"],
    "conditional": ["--gap-only"],
}

# Effective input width per proposer. The unified scorer consumes the
# multitask envelope (96 slots); the dedicated closed-loop proposer consumes
# the rule-discovery feature vector (RULE_DISCOVERY_FEATURE_COUNT in
# src/prime_gap/internal.h, currently 25). These values let the driver
# estimate parameter count without requiring the binary to emit it. Override
# at the CLI if the engine changes these widths.
DEFAULT_FEATURE_WIDTHS: dict[str, int] = {
    "unified": 96,
    "closed-loop": 25,
}

# All proposers reported per phase. The unified scorer is added per-hidden in the runner.
STATIC_PROPOSERS = (
    ("random", ["--proposer", "random", "--proposer-seed", str(DEFAULT_RANDOM_SEED)]),
    ("grammar", ["--proposer", "grammar"]),
    ("structural", ["--proposer", "structural"]),
    ("validation-oracle", ["--proposer", "validation-oracle"]),
    ("neural", ["--proposer", "neural"]),
    ("closed-loop", ["--proposer", "closed-loop"]),
)


@dataclass
class Invocation:
    invocation_id: str
    grammar: str
    proposer_label: str
    command: list[str]
    context: dict[str, object] = field(default_factory=dict)


@dataclass
class RunOutcome:
    invocation: Invocation
    returncode: int
    stdout: str
    stderr: str
    rows: list[dict[str, object]]
    log_path: Path


def shell_join(command: Iterable[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def parse_hidden_layers(spec: str) -> list[int]:
    """Parse a hidden-size spec string like '64x32' into [64, 32]."""
    if not spec:
        return []
    parts = [piece.strip() for piece in spec.replace(",", "x").split("x") if piece.strip()]
    out: list[int] = []
    for part in parts:
        try:
            value = int(part)
        except ValueError:
            continue
        if value > 0:
            out.append(value)
    return out


def estimate_capacity(input_features: int, hidden_layers: list[int], output_size: int = 1) -> dict[str, int]:
    """Compute parameter count and forward-pass MACs for a fully-connected MLP.

    Matches the layout of `nn_create_config` in src/nn.c: dense layers with
    bias per unit, scalar inputs. Returns a dict with `params_est` and
    `forward_macs_est`. Each row scored at inference time costs
    `forward_macs_est` multiply-accumulates; multiply by the softmax group
    size to estimate per-state work.
    """
    if input_features <= 0 or not hidden_layers:
        return {"params_est": 0, "forward_macs_est": 0}
    sizes = [input_features] + list(hidden_layers) + [output_size]
    params = 0
    macs = 0
    for prev, nxt in zip(sizes[:-1], sizes[1:]):
        params += prev * nxt + nxt  # weights + biases
        macs += prev * nxt
    return {"params_est": params, "forward_macs_est": macs}


def capacity_for_row(proposer: str, hidden_spec: str | None) -> dict[str, int]:
    """Best-effort capacity estimate for a given proposer + hidden spec."""
    if not hidden_spec:
        return {"params_est": 0, "forward_macs_est": 0}
    width = DEFAULT_FEATURE_WIDTHS.get(proposer)
    if width is None:
        return {"params_est": 0, "forward_macs_est": 0}
    return estimate_capacity(width, parse_hidden_layers(hidden_spec))


def format_compact_int(value: object) -> str:
    """Render an integer with a K/M suffix for table compactness."""
    if value is None or value == "":
        return ""
    try:
        n = int(float(value))
    except (TypeError, ValueError):
        return str(value)
    if n == 0:
        return "—"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--phase",
        choices=["1", "2", "3", "all"],
        default="1",
        help="Which phase to run. 'all' runs 1, 2, and 3 in sequence.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=16,
        help="Rule budget for matched-budget comparison (default: 16).",
    )
    parser.add_argument(
        "--binary",
        default=str(REPO_ROOT / "build" / "float" / "prime_rule_search"),
        help="Path to prime_rule_search binary.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PHASE_OUTPUT_ROOT,
        help="Directory under results/ where per-phase artifacts land.",
    )
    parser.add_argument(
        "--grammars",
        default="unconditional,conditional",
        help="CSV grammar names to evaluate.",
    )
    parser.add_argument(
        "--hidden-sizes",
        default=None,
        help="CSV override of unified-scorer hidden ladder (e.g. '64x32,128x64').",
    )
    parser.add_argument(
        "--marginal-states",
        default=None,
        help="CSV override of marginal-utility subsample sizes (e.g. '64,4096').",
    )
    parser.add_argument(
        "--include-baselines",
        action="store_true",
        help="Run the matched-budget baselines alongside the unified scorer. "
        "Default for phases 1 and 3; opt-in for phase 2 (which focuses on capacity).",
    )
    parser.add_argument(
        "--skip-baselines",
        action="store_true",
        help="Skip the baselines (use when comparing against a previous baseline snapshot).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    parser.add_argument(
        "--coverage-cache-dir",
        type=Path,
        default=None,
        help="Directory for content-addressed prime_rule_search coverage blobs. "
        "Defaults to phase_<N>/coverage-cache.",
    )
    parser.add_argument(
        "--no-coverage-cache",
        action="store_true",
        help="Do not pass --coverage-cache-dir to prime_rule_search.",
    )
    return parser.parse_args(argv)


def select_phases(arg: str) -> list[int]:
    if arg == "all":
        return [1, 2, 3]
    return [int(arg)]


def build_invocations(
    binary: str,
    phase: int,
    budget: int,
    grammars: list[str],
    hidden_sizes: list[str],
    marginal_states: list[int],
    run_baselines: bool,
    coverage_cache_dir: Path | None,
) -> list[Invocation]:
    invocations: list[Invocation] = []
    cache_flags = (
        ["--coverage-cache-dir", str(coverage_cache_dir)]
        if coverage_cache_dir is not None
        else []
    )
    for grammar in grammars:
        if grammar not in GRAMMAR_SPECS:
            raise ValueError(f"unknown grammar: {grammar}")
        grammar_flags = GRAMMAR_SPECS[grammar]
        for marginal in marginal_states:
            marginal_flags = ["--max-marginal-states", str(marginal)]
            if run_baselines:
                for proposer_label, proposer_flags in STATIC_PROPOSERS:
                    baseline_hidden_sizes = hidden_sizes if proposer_label == "closed-loop" else [""]
                    for baseline_hidden in baseline_hidden_sizes:
                        hidden_suffix = f"-{baseline_hidden}" if baseline_hidden else ""
                        invocation_id = (
                            f"phase{phase}_{grammar}_marginal-{marginal}_{proposer_label}{hidden_suffix}"
                        )
                        command = (
                            [binary, "--max-rules", str(budget)]
                            + grammar_flags
                            + marginal_flags
                            + cache_flags
                            + proposer_flags
                        )
                        context: dict[str, object] = {
                            "phase": phase,
                            "grammar": grammar,
                            "proposer": proposer_label,
                            "marginal_states": marginal,
                        }
                        if baseline_hidden:
                            command += ["--closed-loop-hidden", baseline_hidden]
                            context["closed_loop_hidden"] = baseline_hidden
                            context.update(
                                capacity_for_row("closed-loop", baseline_hidden)
                            )
                        invocations.append(
                            Invocation(
                                invocation_id=invocation_id,
                                grammar=grammar,
                                proposer_label=proposer_label,
                                command=command,
                                context=context,
                            )
                        )
            for hidden in hidden_sizes:
                proposer_label = "unified"
                invocation_id = (
                    f"phase{phase}_{grammar}_marginal-{marginal}_unified-{hidden}"
                )
                command = (
                    [binary, "--max-rules", str(budget)]
                    + grammar_flags
                    + marginal_flags
                    + cache_flags
                    + [
                        "--proposer",
                        "unified",
                        "--unified-scorer-hidden",
                        hidden,
                    ]
                )
                unified_context: dict[str, object] = {
                    "phase": phase,
                    "grammar": grammar,
                    "proposer": "unified",
                    "marginal_states": marginal,
                    "unified_scorer_hidden": hidden,
                }
                unified_context.update(capacity_for_row("unified", hidden))
                invocations.append(
                    Invocation(
                        invocation_id=invocation_id,
                        grammar=grammar,
                        proposer_label=f"unified ({hidden})",
                        command=command,
                        context=unified_context,
                    )
                )
    return invocations


def run_invocation(invocation: Invocation, log_dir: Path, dry_run: bool) -> RunOutcome:
    log_path = log_dir / f"{invocation.invocation_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"$ {shell_join(invocation.command)}", flush=True)
    if dry_run:
        log_path.write_text("dry-run\n", encoding="utf-8")
        return RunOutcome(invocation=invocation, returncode=0, stdout="", stderr="", rows=[], log_path=log_path)

    result = subprocess.run(
        invocation.command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    log_path.write_text(result.stdout + result.stderr, encoding="utf-8")

    rows: list[dict[str, object]] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("schema") != RULE_SEARCH_SCHEMA:
            continue
        row.update(invocation.context)
        row["invocation_id"] = invocation.invocation_id
        rows.append(row)

    return RunOutcome(
        invocation=invocation,
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        rows=rows,
        log_path=log_path,
    )


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def format_number(value: object, places: int = 2) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.{places}f}"
    except (TypeError, ValueError):
        return str(value)


def format_percent(value: object) -> str:
    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.2f}%"
    except (TypeError, ValueError):
        return str(value)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = "\n".join("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(part for part in (head, sep, body) if part)


def render_phase_summary(
    phase: int,
    label: str,
    purpose: str,
    rows: list[dict[str, object]],
    failures: list[RunOutcome],
) -> str:
    lines: list[str] = []
    lines.append(f"# Unified Scorer — Phase {phase}: {label}")
    lines.append("")
    lines.append(purpose)
    lines.append("")
    lines.append("See `docs/unified-scorer-proposal.md` for the full design.")
    lines.append("")

    if not rows:
        lines.append("_No `crlp.rule_search.summary.v1` rows captured._")
        if failures:
            lines.append("")
            lines.append("## Failures")
            lines.append("")
            for outcome in failures:
                lines.append(
                    f"- `{outcome.invocation.invocation_id}` (exit {outcome.returncode}). "
                    f"See `logs/{outcome.invocation.invocation_id}.log`."
                )
        return "\n".join(lines) + "\n"

    grammars_in_rows = sorted({str(row.get("grammar", "")) for row in rows if row.get("grammar")})
    for grammar in grammars_in_rows:
        lines.append(f"## {grammar.capitalize()} grammar")
        lines.append("")
        marginals_in_grammar = sorted(
            {int(row.get("marginal_states", 0)) for row in rows if row.get("grammar") == grammar},
        )
        for marginal in marginals_in_grammar:
            subset = [
                row
                for row in rows
                if row.get("grammar") == grammar and int(row.get("marginal_states", 0)) == marginal
            ]
            subset.sort(key=lambda row: (str(row.get("proposer", "")), str(row.get("unified_scorer_hidden", ""))))
            lines.append(f"### marginal_states = {marginal}")
            lines.append("")
            table_rows: list[list[str]] = []
            for row in subset:
                hidden_value = row.get("unified_scorer_hidden") or row.get("closed_loop_hidden") or ""
                hidden = str(hidden_value) if hidden_value else "—"
                proposer_label = str(row.get("proposer", ""))
                if proposer_label == "unified":
                    proposer_label = f"unified ({hidden})"
                elif proposer_label == "closed-loop" and hidden != "—":
                    proposer_label = f"closed-loop ({hidden})"
                # Prefer binary-emitted values when present, fall back to the
                # driver's estimate. Either way the params and forward work
                # appear in the table so capacity saturation is visible.
                params = row.get("parameters") or row.get("params_est") or ""
                forward_work = (
                    row.get("forward_flops")
                    or row.get("forward_macs")
                    or row.get("forward_macs_est")
                    or ""
                )
                cache_hit_write = (
                    f"{row.get('coverage_cache_hits', '')}/{row.get('coverage_cache_writes', '')}"
                    if "coverage_cache_hits" in row or "coverage_cache_writes" in row
                    else ""
                )
                table_rows.append([
                    proposer_label,
                    str(row.get("max_rules", "")),
                    str(row.get("softmax_group_size", "")),
                    cache_hit_write,
                    format_compact_int(params),
                    format_compact_int(forward_work),
                    format_number(row.get("expanded_exact"), 3),
                    format_percent(row.get("expanded_saved_pct")),
                    format_number(row.get("proposer_loss"), 4),
                    str(row.get("accepted_rules", "")),
                    str(row.get("false_rejections", "")),
                ])
            lines.append(markdown_table(
                [
                    "Proposer",
                    "Budget",
                    "Softmax group",
                    "Cov hit/write",
                    "Params",
                    "Forward work",
                    "Exact probes",
                    "Saved %",
                    "Proposer loss",
                    "Accepted",
                    "False rej.",
                ],
                table_rows,
            ))
            lines.append("")
            lines.append(
                "_Params and Forward work are driver estimates from the "
                "declared hidden size unless the binary emitted matching "
                "`parameters` / `forward_flops` fields, in which case those "
                "are used. The estimate uses input width = 96 for unified "
                "rows and 25 for closed-loop rows; adjust "
                "`DEFAULT_FEATURE_WIDTHS` if the engine changes them._"
            )
            lines.append("")

    if failures:
        lines.append("## Failures")
        lines.append("")
        for outcome in failures:
            lines.append(
                f"- `{outcome.invocation.invocation_id}` (exit {outcome.returncode}). "
                f"See `logs/{outcome.invocation.invocation_id}.log`."
            )
        lines.append("")
        lines.append(
            "If failures cluster on `--proposer unified` or `--unified-scorer-hidden`, the engine "
            "work in `docs/unified-scorer-proposal.md` has not landed yet. Re-run after merge."
        )
        lines.append("")

    return "\n".join(lines) + "\n"


def run_phase(phase: int, args: argparse.Namespace) -> int:
    defaults = PHASE_DEFAULTS[phase]
    hidden_csv = args.hidden_sizes or ",".join(defaults["hidden_sizes"])
    marginal_csv = args.marginal_states or ",".join(str(value) for value in defaults["marginal_states"])
    hidden_sizes = [part.strip() for part in hidden_csv.split(",") if part.strip()]
    marginal_states = [int(part.strip()) for part in marginal_csv.split(",") if part.strip()]
    grammars = [part.strip() for part in args.grammars.split(",") if part.strip()]

    run_baselines = not args.skip_baselines and (phase != 2 or args.include_baselines)

    phase_dir = args.output_root / f"phase_{phase}"
    log_dir = phase_dir / "logs"
    phase_dir.mkdir(parents=True, exist_ok=True)
    coverage_cache_dir = None
    if not args.no_coverage_cache:
        coverage_cache_dir = args.coverage_cache_dir or (phase_dir / "coverage-cache")
        coverage_cache_dir.mkdir(parents=True, exist_ok=True)

    invocations = build_invocations(
        binary=args.binary,
        phase=phase,
        budget=args.budget,
        grammars=grammars,
        hidden_sizes=hidden_sizes,
        marginal_states=marginal_states,
        run_baselines=run_baselines,
        coverage_cache_dir=coverage_cache_dir,
    )

    if not invocations:
        print(f"phase {phase}: nothing to run", file=sys.stderr)
        return 0

    rows: list[dict[str, object]] = []
    failures: list[RunOutcome] = []
    commands: list[str] = []
    for invocation in invocations:
        commands.append(shell_join(invocation.command))
        outcome = run_invocation(invocation, log_dir, args.dry_run)
        if outcome.returncode != 0:
            failures.append(outcome)
        rows.extend(outcome.rows)

    (phase_dir / "commands.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")
    write_jsonl(phase_dir / "raw.jsonl", rows)
    write_csv(phase_dir / "summary.csv", rows)
    summary_md = render_phase_summary(
        phase=phase,
        label=str(defaults["label"]),
        purpose=str(defaults["purpose"]),
        rows=rows,
        failures=failures,
    )
    (phase_dir / "summary.md").write_text(summary_md, encoding="utf-8")

    if failures:
        print(
            f"phase {phase}: {len(failures)} invocation(s) failed; see {phase_dir / 'summary.md'}",
            file=sys.stderr,
        )
        return 1

    print(f"phase {phase}: wrote {phase_dir}")
    return 0


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    phases = select_phases(args.phase)
    exit_code = 0
    for phase in phases:
        result = run_phase(phase, args)
        if result != 0:
            exit_code = result
    return exit_code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
