#!/usr/bin/env python3
"""Train an offline selector for ARC Holo bucket review candidates.

This script consumes `crlp.arc_holo_bucket_review_candidate.v1` rows emitted by
`arc_agi --arc-holo-residual-diagnostics`. It is intentionally offline: the goal
is to decide whether the bounded candidate rows contain enough signal to learn a
review reranker before wiring anything into the C runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Iterable


SCHEMA = "crlp.arc_holo_bucket_review_candidate.v1"
SUMMARY_SCHEMA = "crlp.arc_holo_bucket_review_selector.summary.v1"
WEIGHT_SCHEMA = "crlp.arc_holo_bucket_review_selector.weights.v1"


def stable_hash(text: str) -> int:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def expand_inputs(inputs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        if item.is_dir():
            raw_dir = item / "raw"
            if raw_dir.is_dir():
                paths.extend(sorted(raw_dir.glob("*.jsonl")))
            else:
                paths.extend(sorted(item.glob("*.jsonl")))
        elif any(ch in str(item) for ch in "*?[]"):
            paths.extend(sorted(Path().glob(str(item))))
        else:
            paths.append(item)
    unique = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def iter_rows(paths: Iterable[Path]) -> Iterable[dict]:
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
                if row.get("schema") == SCHEMA:
                    row["_source"] = str(path)
                    yield row


def number(row: dict, key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def flag(row: dict, key: str) -> float:
    return 1.0 if row.get(key) else 0.0


def add_feature(features: dict[str, float], name: str, value: float) -> None:
    if value != 0.0 and math.isfinite(value):
        features[name] = value


def row_features(row: dict, hash_bins: int) -> dict[str, float]:
    baseline_rank = max(1.0, number(row, "baseline_rank", 1.0))
    review_rank = max(1.0, number(row, "review_rank", 1.0))
    budget = max(1.0, number(row, "budget", 1.0))
    pressure_raw = number(row, "pressure_raw")
    review_priority = number(row, "review_priority")
    frontier = number(row, "review_frontier_priority")
    baseline_score = number(row, "baseline_score")
    structural_score = number(row, "structural_score")
    features: dict[str, float] = {"bias": 1.0}
    add_feature(features, "log_baseline_rank", math.log1p(baseline_rank) / 8.0)
    add_feature(features, "inv_baseline_rank", 1.0 / baseline_rank)
    add_feature(features, "log_review_rank", math.log1p(review_rank) / 8.0)
    add_feature(features, "inv_review_rank", 1.0 / review_rank)
    add_feature(features, "log_budget", math.log1p(budget) / 6.0)
    add_feature(features, "pressure_raw", pressure_raw)
    add_feature(features, "review_priority", review_priority / 8.0)
    add_feature(features, "frontier_priority", frontier / 8.0)
    add_feature(features, "priority_deficit", number(row, "priority_deficit") / 8.0)
    add_feature(features, "priority_margin", (review_priority - frontier) / 8.0)
    add_feature(features, "baseline_score", baseline_score / 160.0)
    add_feature(features, "structural_score", structural_score / 160.0)
    add_feature(features, "task_prior_delta", (baseline_score - structural_score) / 80.0)
    add_feature(features, "holo_pressure_updates", number(row, "holo_pressure_updates") / 128.0)
    add_feature(features, "positive_rows", number(row, "positive_rows") / 128.0)
    add_feature(features, "would_review_under_fixed_gate", flag(row, "would_review_under_fixed_gate"))
    add_feature(features, "gate_open", flag(row, "gate_open"))
    for key in (
        "op",
        "input_color",
        "output_color",
        "background_color",
        "selector_kind",
        "output_selector_kind",
        "reference_color",
        "render_mode",
        "step_count",
        "hypothesis_template_id",
        "hypothesis_slot_count",
    ):
        value = int(number(row, key))
        add_feature(features, f"{key}={value}", 1.0)
    add_feature(features, "dx", number(row, "dx") / 16.0)
    add_feature(features, "dy", number(row, "dy") / 16.0)
    add_feature(features, "scale", number(row, "scale") / 16.0)
    name = str(row.get("candidate_name", ""))
    if hash_bins > 0:
        for token in name.replace("-", "_").split("_"):
            if not token:
                continue
            bucket = stable_hash(token) % hash_bins
            add_feature(features, f"name_hash_{bucket}", 1.0)
    return features


def dot(weights: dict[str, float], features: dict[str, float]) -> float:
    return sum(weights.get(name, 0.0) * value for name, value in features.items())


def subtract(lhs: dict[str, float], rhs: dict[str, float]) -> dict[str, float]:
    result = dict(lhs)
    for name, value in rhs.items():
        result[name] = result.get(name, 0.0) - value
        if result[name] == 0.0:
            del result[name]
    return result


def group_key(row: dict) -> tuple[str, str, int]:
    return (
        str(row.get("_source", "")),
        str(row.get("task_instance", "")),
        int(number(row, "budget")),
    )


def split_key(row: dict) -> str:
    return str(row.get("task_instance", ""))


def assign_split(row: dict, holdout_mod: int, holdout_bucket: int) -> str:
    if holdout_mod <= 1:
        return "train"
    bucket = stable_hash(split_key(row)) % holdout_mod
    return "test" if bucket == holdout_bucket else "train"


def train_pairwise(
    groups: list[list[dict]],
    features_by_id: dict[int, dict[str, float]],
    *,
    epochs: int,
    learning_rate: float,
    l2: float,
    negative_limit: int,
    seed: int,
) -> dict[str, float]:
    rng = random.Random(seed)
    weights: dict[str, float] = {}
    train_groups = [group for group in groups if any(r.get("exact_label") for r in group)]
    for epoch in range(epochs):
        rng.shuffle(train_groups)
        rate = learning_rate / math.sqrt(epoch + 1.0)
        for group in train_groups:
            positives = [r for r in group if r.get("exact_label")]
            negatives = [r for r in group if not r.get("exact_label")]
            negatives.sort(key=lambda r: number(r, "review_priority"), reverse=True)
            if negative_limit > 0:
                negatives = negatives[:negative_limit]
            for positive in positives:
                pos_features = features_by_id[id(positive)]
                for negative in negatives:
                    diff = subtract(pos_features, features_by_id[id(negative)])
                    margin = dot(weights, diff)
                    if margin > 40.0:
                        coeff = 0.0
                    elif margin < -40.0:
                        coeff = 1.0
                    else:
                        coeff = 1.0 / (1.0 + math.exp(margin))
                    for name, value in diff.items():
                        weights[name] = weights.get(name, 0.0) * (1.0 - rate * l2)
                        weights[name] += rate * coeff * value
    return weights


def rank_of_exact(group: list[dict], score_key: str) -> list[int]:
    ranked = sorted(group, key=lambda row: row[score_key], reverse=True)
    ranks = []
    for rank, row in enumerate(ranked, 1):
        if row.get("exact_label"):
            ranks.append(rank)
    return ranks


def evaluate(
    groups: list[list[dict]],
    weights: dict[str, float],
    features_by_id: dict[int, dict[str, float]],
) -> dict:
    actual_selected_exact = 0
    raw_loaded_selected_exact = 0
    learned_loaded_selected_exact = 0
    exact_rows = 0
    original_exact_ranks: list[int] = []
    raw_loaded_exact_ranks: list[int] = []
    learned_loaded_exact_ranks: list[int] = []
    groups_with_exact = 0
    groups_with_actual_hit = 0
    groups_with_raw_loaded_hit = 0
    groups_with_learned_loaded_hit = 0
    for group in groups:
        for row in group:
            row["_learned_score"] = dot(weights, features_by_id[id(row)])
            row["_raw_score"] = number(row, "review_priority")
        positives = [row for row in group if row.get("exact_label")]
        if positives:
            groups_with_exact += 1
        review_slots = max(1, int(number(group[0], "review_slots", 1.0)))
        actual_hits = [
            row for row in positives if row.get("selected_for_review")
        ]
        raw_hits = [
            row
            for row in sorted(group, key=lambda r: r["_raw_score"], reverse=True)[:review_slots]
            if row.get("exact_label")
        ]
        learned_hits = [
            row
            for row in sorted(group, key=lambda r: r["_learned_score"], reverse=True)[
                :review_slots
            ]
            if row.get("exact_label")
        ]
        actual_selected_exact += len(actual_hits)
        raw_loaded_selected_exact += len(raw_hits)
        learned_loaded_selected_exact += len(learned_hits)
        if actual_hits:
            groups_with_actual_hit += 1
        if raw_hits:
            groups_with_raw_loaded_hit += 1
        if learned_hits:
            groups_with_learned_loaded_hit += 1
        for row in positives:
            exact_rows += 1
            original_exact_ranks.append(int(number(row, "review_rank")))
        raw_loaded_exact_ranks.extend(rank_of_exact(group, "_raw_score"))
        learned_loaded_exact_ranks.extend(rank_of_exact(group, "_learned_score"))
    return {
        "groups": len(groups),
        "groups_with_exact": groups_with_exact,
        "exact_rows": exact_rows,
        "actual_selected_exact_rows": actual_selected_exact,
        "raw_loaded_selected_exact_rows": raw_loaded_selected_exact,
        "learned_loaded_selected_exact_rows": learned_loaded_selected_exact,
        "groups_with_actual_hit": groups_with_actual_hit,
        "groups_with_raw_loaded_hit": groups_with_raw_loaded_hit,
        "groups_with_learned_loaded_hit": groups_with_learned_loaded_hit,
        "mean_original_exact_review_rank": statistics.fmean(original_exact_ranks)
        if original_exact_ranks
        else 0.0,
        "mean_raw_loaded_exact_rank": statistics.fmean(raw_loaded_exact_ranks)
        if raw_loaded_exact_ranks
        else 0.0,
        "mean_learned_loaded_exact_rank": statistics.fmean(learned_loaded_exact_ranks)
        if learned_loaded_exact_ranks
        else 0.0,
        "max_original_exact_review_rank": max(original_exact_ranks)
        if original_exact_ranks
        else 0,
        "max_learned_loaded_exact_rank": max(learned_loaded_exact_ranks)
        if learned_loaded_exact_ranks
        else 0,
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=Path("build/float/arc_holo_bucket_review_selector/summary.jsonl"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--negative-limit", type=int, default=24)
    parser.add_argument("--holdout-mod", type=int, default=5)
    parser.add_argument("--holdout-bucket", type=int, default=0)
    parser.add_argument("--hash-bins", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260614)
    args = parser.parse_args()

    paths = expand_inputs(args.inputs)
    if not paths:
        raise SystemExit("no input JSONL files found")
    rows = list(iter_rows(paths))
    if not rows:
        raise SystemExit("no review candidate rows found")
    features_by_id = {id(row): row_features(row, args.hash_bins) for row in rows}
    grouped: dict[tuple[str, str, int], list[dict]] = {}
    for row in rows:
        grouped.setdefault(group_key(row), []).append(row)
    train_groups: list[list[dict]] = []
    test_groups: list[list[dict]] = []
    for group in grouped.values():
        split = assign_split(group[0], args.holdout_mod, args.holdout_bucket)
        if split == "test":
            test_groups.append(group)
        else:
            train_groups.append(group)
    weights = train_pairwise(
        train_groups,
        features_by_id,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        negative_limit=args.negative_limit,
        seed=args.seed,
    )
    summary_rows = []
    for split_name, groups in (("train", train_groups), ("test", test_groups), ("all", list(grouped.values()))):
        metrics = evaluate(groups, weights, features_by_id)
        summary_rows.append(
            {
                "schema": SUMMARY_SCHEMA,
                "split": split_name,
                "input_files": len(paths),
                "rows": sum(len(group) for group in groups),
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "l2": args.l2,
                "negative_limit": args.negative_limit,
                "holdout_mod": args.holdout_mod,
                "holdout_bucket": args.holdout_bucket,
                **metrics,
            }
        )
    top_weights = sorted(weights.items(), key=lambda item: abs(item[1]), reverse=True)[:40]
    summary_rows.append(
        {
            "schema": WEIGHT_SCHEMA,
            "feature_count": len(weights),
            "top_weights": [
                {"feature": name, "weight": value} for name, value in top_weights
            ],
        }
    )
    write_jsonl(args.out, summary_rows)
    for row in summary_rows:
        print(json.dumps(row, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
