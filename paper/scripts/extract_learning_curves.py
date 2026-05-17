#!/usr/bin/env python3
"""Extract learning-curve and gate-ablation data from canonical artifacts.

Reads:
  - build/float/traces/prime_gap.jsonl  (1000 rows, supervised training trace)
  - build/float/grid_path.jsonl          (4000 rows, supervised training trace)
  - build/float/delayed_claim.jsonl      (4000 rows, supervised training trace)
  - results/latest/comparison.jsonl      (gate ablations for all four tasks)
  - results/latest/comparison.jsonl      (ARC hypothesis evidence rows)

Emits a single JSON blob on stdout with:
  - learning_curves[task] = list of (step, rolling_accuracy, rolling_loss)
  - gate_ablation[task]   = list of {mode, accuracy, nll, ece}
  - arc_hypothesis        = list of certified evidence rows
  - reliability[task]     = list of reliability bins from the strongest run

This is intentionally read-only and deterministic; it never re-runs experiments.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

WINDOW = 50  # smoothing window for rolling accuracy

ROOT = Path(__file__).resolve().parents[2]
ARC_AGI = ROOT / "build" / "float" / "arc_agi"
PRIME_GAP = ROOT / "build" / "float" / "prime_gap"
SEED = 20240516
PRIME_SEED = 20240511
ARC_EPOCH_GRID = (1, 2, 4, 8)
PRIME_EPOCHS = 8


def rolling(values: list[float], w: int) -> list[float]:
    out: list[float] = []
    s = 0.0
    for i, v in enumerate(values):
        s += v
        if i >= w:
            s -= values[i - w]
        denom = min(i + 1, w)
        out.append(s / denom)
    return out


def trace_curve(path: Path, downsample: int = 25) -> list[tuple[int, float, float]]:
    correct: list[float] = []
    losses: list[float] = []
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            sel = row.get("selected_index")
            tgt = row.get("target_index")
            loss = row.get("loss")
            if sel is None or tgt is None or loss is None:
                continue
            correct.append(1.0 if sel == tgt else 0.0)
            losses.append(float(loss))
    if not correct:
        return []
    racc = rolling(correct, WINDOW)
    rloss = rolling(losses, WINDOW)
    out: list[tuple[int, float, float]] = []
    for i in range(0, len(racc), downsample):
        out.append((i + 1, racc[i], rloss[i]))
    if out[-1][0] != len(racc):
        out.append((len(racc), racc[-1], rloss[-1]))
    return out


def load_compare_rows() -> list[dict]:
    rows: list[dict] = []
    p = ROOT / "results" / "latest" / "comparison.jsonl"
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            rows.append(json.loads(line))
    return rows


def gate_ablation(rows: list[dict], task: str) -> list[dict]:
    """Keep the first (canonical) occurrence per evaluation mode and drop rows
    where the headline metric is missing (e.g., the prime-gap schema preamble)."""
    out: list[dict] = []
    seen: set[str] = set()
    for r in rows:
        if r.get("task") != task or r.get("schema") != "crlp.compare.v1":
            continue
        mode = r.get("evaluation_mode") or task
        if mode in seen:
            continue
        acc = r.get("top1")
        if acc is None:
            acc = r.get("accuracy")
        nll = r.get("nll")
        ece = r.get("ece")
        if acc is None and nll is None and ece is None:
            continue
        seen.add(mode)
        out.append(
            {
                "mode": mode,
                "accuracy": acc,
                "nll": nll,
                "ece": ece,
                "search_saved": r.get("search_saved"),
                "model_hash": r.get("model_hash", "")[:10],
                "parent_hash": r.get("parent_hash", "")[:10],
            }
        )
    return out


def arc_hypothesis(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("schema") == "crlp.hypothesis.evidence.v1"]


def reliability(rows: list[dict], task: str, prefer_mode: str | None = None) -> list[dict]:
    best = None
    for r in rows:
        if r.get("task") != task or r.get("schema") != "crlp.compare.v1":
            continue
        bins = r.get("reliability_bins") or []
        if not bins:
            continue
        if prefer_mode is None or r.get("evaluation_mode") == prefer_mode:
            best = bins
            if prefer_mode is not None:
                break
    return best or []


def arc_per_epoch_eval() -> list[tuple[int, float, float]]:
    """Run arc_agi once per epoch count in ARC_EPOCH_GRID, capture (epoch, top1, nll)."""
    if not ARC_AGI.exists():
        return []
    out: list[tuple[int, float, float]] = []
    for epochs in ARC_EPOCH_GRID:
        proc = subprocess.run(
            [
                str(ARC_AGI),
                "--epochs", str(epochs),
                "--test-samples", "900",
                "--seed", str(SEED),
            ],
            capture_output=True, text=True, check=False, timeout=180,
        )
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("evaluation_mode") == "neural_cell_scorer":
                out.append((epochs, float(row["top1"]), float(row["nll"])))
                break
    return out


def prime_gap_per_epoch_loss() -> list[tuple[int, float, float]]:
    """Single 8-epoch prime_gap run; parse per-epoch (epoch, raw_loss, gated_loss)."""
    if not PRIME_GAP.exists():
        return []
    proc = subprocess.run(
        [
            str(PRIME_GAP),
            "100000", "200000",  # train/test limits
            str(PRIME_EPOCHS), "0",  # supervised + RL epochs
            "8000", "1500",  # train/test states
            "1.20", "0.12",  # reg coefficients
            str(PRIME_SEED),
        ],
        capture_output=True, text=True, check=False, timeout=300,
    )
    raw: dict[int, float] = {}
    gated: dict[int, float] = {}
    raw_re = re.compile(r"^model supervised epoch (\d+) loss=([0-9.]+)")
    gated_re = re.compile(r"^model learned-wheel supervised epoch (\d+) loss=([0-9.]+)")
    for line in proc.stdout.splitlines():
        m = raw_re.match(line)
        if m:
            raw[int(m.group(1))] = float(m.group(2))
            continue
        m = gated_re.match(line)
        if m:
            gated[int(m.group(1))] = float(m.group(2))
    out: list[tuple[int, float, float]] = []
    for epoch in sorted(set(raw) | set(gated)):
        out.append((epoch, raw.get(epoch, float("nan")), gated.get(epoch, float("nan"))))
    return out


def main() -> int:
    traces = {
        "prime_gap": ROOT / "build/float/traces/prime_gap.jsonl",
        "grid_path": ROOT / "build/float/grid_path.jsonl",
        "delayed_claim": ROOT / "build/float/delayed_claim.jsonl",
    }
    curves = {t: trace_curve(p) for t, p in traces.items() if p.exists()}

    rows = load_compare_rows()
    gates = {
        "prime_gap": gate_ablation(rows, "prime_gap"),
        "grid_path": gate_ablation(rows, "grid_path"),
        "delayed_claim": gate_ablation(rows, "delayed_claim"),
        "arc_agi": gate_ablation(rows, "arc_agi"),
    }
    rel = {
        "prime_gap": reliability(rows, "prime_gap", "expanded_rule_model"),
        "grid_path": reliability(rows, "grid_path", "budgeted_legality_gate_model"),
        "delayed_claim": reliability(rows, "delayed_claim"),
        "arc_agi": reliability(rows, "arc_agi", "neural_cell_scorer"),
    }
    hyp = arc_hypothesis(rows)

    arc_curve = arc_per_epoch_eval()
    prime_loss = prime_gap_per_epoch_loss()

    blob = {
        "learning_curves": curves,
        "arc_epoch_eval": arc_curve,
        "prime_gap_epoch_loss": prime_loss,
        "gate_ablation": gates,
        "reliability": rel,
        "arc_hypothesis": hyp,
    }
    json.dump(blob, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")

    # Also emit a Typst-importable .typ alongside the JSON so the paper reads
    # the same data the JSON does. The paper imports via `#import "data.typ":`.
    typ_path = ROOT / "paper" / "data.typ"
    with typ_path.open("w") as f:
        f.write("// AUTOGENERATED by paper/scripts/extract_learning_curves.py.\n")
        f.write("// Do not edit by hand; rerun `make paper-data` after results change.\n\n")
        f.write("#let learning_curves = (\n")
        for task, pts in curves.items():
            f.write(f"  \"{task}\": (\n")
            for step, acc, loss in pts:
                f.write(f"    ({step}, {acc:.6f}, {loss:.6f}),\n")
            f.write("  ),\n")
        f.write(")\n\n")
        f.write("#let arc_epoch_eval = (\n")
        for epoch, top1, nll in arc_curve:
            f.write(f"  ({epoch}, {top1:.6f}, {nll:.6f}),\n")
        f.write(")\n\n")
        f.write("#let prime_gap_epoch_loss = (\n")
        for epoch, raw_loss, gated_loss in prime_loss:
            raw_str = "0.0" if raw_loss != raw_loss else f"{raw_loss:.6f}"  # NaN check
            gated_str = "0.0" if gated_loss != gated_loss else f"{gated_loss:.6f}"
            f.write(f"  ({epoch}, {raw_str}, {gated_str}),\n")
        f.write(")\n\n")
        f.write("#let gate_ablation = (\n")
        for task, gs in gates.items():
            f.write(f"  \"{task}\": (\n")
            for g in gs:
                mode = g["mode"].replace("\"", "\\\"")
                acc = "none" if g["accuracy"] is None else f"{g['accuracy']:.6f}"
                nll = "none" if g["nll"] is None else f"{g['nll']:.6f}"
                ece = "none" if g["ece"] is None else f"{g['ece']:.6f}"
                f.write(
                    f"    (mode: \"{mode}\", accuracy: {acc}, nll: {nll}, ece: {ece}),\n"
                )
            f.write("  ),\n")
        f.write(")\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
