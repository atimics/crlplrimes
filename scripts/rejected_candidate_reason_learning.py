#!/usr/bin/env python3
"""Learn structural priors from rejected certificate rows across gate tasks.

The experiment consumes rule-candidate traces with verifier result codes. It
trains on one structural split, reranks held-out rows, and checks whether
accepted/safe rows move ahead of unsafe or unsupported rows without feeding the
held-out verifier result into the scorer.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import subprocess
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable


RESULT_LABELS = ("accepted", "safe_duplicate", "unsafe", "no_support", "other")
DEFAULT_BUDGETS = (8, 16, 32, 64)
DEFAULT_PRIME_SPLIT_GRID = (43, 53, 61, 71)
DEFAULT_MODULAR_SPLIT_GRID = (5, 7, 11, 13)
DEFAULT_DELAYED_SPLIT_GRID = (44, 66, 88)
STRATEGIES = ("model_score", "reason_prior", "reason_aware", "trained_residual")
RESIDUAL_DIVISOR_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31)
RESIDUAL_EPOCHS = 1000
RESIDUAL_LEARNING_RATE = 0.03
RESIDUAL_L2 = 1.0e-4
RankKey = Callable[[dict[str, Any]], tuple[Any, ...]]
SAFE_LABELS = ("accepted", "safe_duplicate")


@dataclass
class LearningRun:
    task: str
    title: str
    trace_path: Path
    split_key_name: str
    split_value: int
    split_grid: list[int]
    selection_seed: int | None
    rows: list[dict[str, Any]]
    train_rows: list[dict[str, Any]]
    eval_rows: list[dict[str, Any]]
    strategy_rows: list[dict[str, Any]]
    residual_model: dict[str, Any]
    sweep_rows: list[dict[str, Any]]
    reason_family_rows: list[dict[str, Any]]


def row_number(row: dict[str, Any], key: str) -> float:
    value = row.get(key, 0)
    if value is None:
        return 0.0
    return float(value)


def row_int(row: dict[str, Any], key: str) -> int:
    value = row.get(key, 0)
    if value is None:
        return 0
    return int(float(value))


def normalized_base(row: dict[str, Any], schema: str, task: str) -> dict[str, Any]:
    out = dict(row)
    out["source_schema"] = schema
    out["task"] = task
    out.setdefault("subtask", "")
    out.setdefault("selection_strategy", "")
    out.setdefault("conditional", False)
    out.setdefault("state_modulus", 1)
    out.setdefault("state_residue", 0)
    out.setdefault("filter_modulus", 1)
    out.setdefault("filter_residue", 0)
    out.setdefault("action", 0)
    out.setdefault("feature", 0)
    out.setdefault("other_action", 0)
    out.setdefault("direction", 0)
    out.setdefault("threshold", 0.0)
    out.setdefault("margin", 0.0)
    out.setdefault("structural_score", 0.0)
    out.setdefault("model_score", 0.0)
    out.setdefault("train_marginal_utility", row.get("train_rejects", 0))
    out.setdefault("validation_marginal_utility", row.get("validation_rejects", 0))
    out.setdefault("train_false_rejections", 0)
    out.setdefault("validation_false_rejections", 0)
    out.setdefault("rule_text", "")
    out.setdefault("rule_id", row.get("rule_rank", 0))
    out.setdefault("rule_kind", "unknown")
    return out


def normalize_prime_row(row: dict[str, Any]) -> dict[str, Any]:
    out = normalized_base(row, str(row["schema"]), "prime_gap")
    out["split_key"] = row_int(out, "filter_modulus")
    out["split_key_name"] = "filter_modulus"
    out["rule_id"] = row.get("rule_rank", out.get("rule_id", 0))
    return out


def normalize_modular_row(row: dict[str, Any]) -> dict[str, Any] | None:
    if row.get("selection_strategy") not in (None, "", "learned_structural"):
        return None
    subtask = str(row.get("task", "modular_gate"))
    out = normalized_base(row, str(row["schema"]), "modular_gate")
    out["subtask"] = subtask
    out["rule_text"] = str(row.get("rule_text", row.get("rule", "")))
    out["filter_modulus"] = row_int(row, "modulus")
    out["filter_residue"] = row_int(row, "residue")
    out["state_modulus"] = 1
    out["state_residue"] = row_int(row, "tuple_offset")
    out["split_key"] = max(1, row_int(out, "filter_modulus"))
    out["split_key_name"] = "modulus"
    if row_int(out, "validation_marginal_utility") == 0 and row_int(row, "validation_support") > 0:
        out["validation_marginal_utility"] = row_int(row, "validation_support")
    out["rule_id"] = "{task}:{kind}:{prime}:{power}:{modulus}:{residue}:{offset}:{sequence}".format(
        task=subtask,
        kind=str(row.get("rule_kind", "unknown")),
        prime=row_int(row, "prime"),
        power=row_int(row, "power"),
        modulus=row_int(row, "modulus"),
        residue=row_int(row, "residue"),
        offset=row_int(row, "tuple_offset"),
        sequence=str(row.get("sequence", "")),
    )
    return out


def delayed_rule_text(row: dict[str, Any]) -> str:
    kind = str(row.get("rule_kind", "unknown"))
    action = str(row.get("action_name", row.get("action", "action")))
    if kind == "utility_margin":
        return "{action} dominated by action {other} by margin {margin:.3g}".format(
            action=action,
            other=row_int(row, "other_action"),
            margin=row_number(row, "margin"),
        )
    direction = ">=" if row_int(row, "direction") > 0 else "<="
    return "reject {action} when feature {feature} {direction} {threshold:.3g}".format(
        action=action,
        feature=row_int(row, "feature"),
        direction=direction,
        threshold=row_number(row, "threshold"),
    )


def normalize_delayed_row(row: dict[str, Any]) -> dict[str, Any]:
    out = normalized_base(row, str(row["schema"]), "delayed_claim")
    rule_id = row_int(row, "rule_id")
    out["rule_id"] = rule_id
    if str(row.get("rule_kind", "")) == "threshold":
        threshold_index = int(round(row_number(row, "threshold") * 10.0))
        direction_index = 1 if row_int(row, "direction") > 0 else 0
        out["split_key"] = row_int(row, "feature") * 22 + direction_index * 11 + threshold_index
    else:
        out["split_key"] = 132 + max(0, rule_id - 528)
    out["split_key_name"] = "rule_template"
    out["filter_modulus"] = rule_id + 1
    out["filter_residue"] = row_int(row, "feature")
    out["rule_text"] = delayed_rule_text(row)
    return out


def normalize_row(row: dict[str, Any]) -> dict[str, Any] | None:
    schema = row.get("schema")
    if schema == "crlp.rule_trace.v1":
        return normalize_prime_row(row)
    if schema in {"crlp.modular_gate.rule_trace.v1", "crlp.modular_gate.rule.v1"}:
        return normalize_modular_row(row)
    if schema == "crlp.delayed_claim.rule_trace.v1":
        return normalize_delayed_row(row)
    return None


def parse_json_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line.startswith("{"):
                continue
            normalized = normalize_row(json.loads(line))
            if normalized is not None:
                rows.append(normalized)
    return rows


def run_checked(command: list[str], stdout_path: Path) -> None:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    stdout_path.write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(command)} failed; see {stdout_path}")


def run_prime_rule_search(args: argparse.Namespace, trace_path: Path, stdout_path: Path) -> None:
    command = [
        str(args.binary),
        "--candidate-only",
        "--unconditional-only",
        "--moduli-range",
        args.moduli_range,
        "--train-limit",
        str(args.train_limit),
        "--test-limit",
        str(args.test_limit),
        "--max-train-states",
        str(args.max_train_states),
        "--max-test-states",
        str(args.max_test_states),
        "--max-rules",
        str(max(args.budgets)),
        "--proposer-seed",
        str(args.proposer_seed),
        "--trace",
        str(trace_path),
    ]
    run_checked(command, stdout_path)


def run_modular_gate_search(args: argparse.Namespace, trace_path: Path, stdout_path: Path) -> None:
    command = [
        str(args.modular_binary),
        "--task",
        "all",
        "--max-states",
        str(args.modular_max_states),
        "--max-actions",
        str(args.modular_max_actions),
        "--max-rules",
        str(args.modular_max_rules),
        "--selection-seed",
        str(args.modular_selection_seed),
        "--trace",
        str(trace_path),
    ]
    run_checked(command, stdout_path)


def run_delayed_claim_rule_search(
    args: argparse.Namespace,
    trace_path: Path,
    train_stdout_path: Path,
    rule_stdout_path: Path,
    checkpoint_path: Path,
) -> None:
    train_command = [
        str(args.delayed_claim_binary),
        "--save",
        str(checkpoint_path),
    ]
    run_checked(train_command, train_stdout_path)
    rule_command = [
        str(args.delayed_claim_binary),
        "--rule-search",
        str(checkpoint_path),
        "--max-rules",
        str(args.delayed_claim_max_rules),
        "--proposer",
        args.delayed_claim_proposer,
        "--proposer-seed",
        str(args.delayed_claim_proposer_seed),
        "--rule-trace",
        str(trace_path),
    ]
    run_checked(rule_command, rule_stdout_path)


def reason_label(row: dict[str, Any]) -> str:
    result = str(row.get("verifier_result", "other"))
    if result == "accepted":
        return "accepted"
    if result == "verified_not_selected":
        return "safe_duplicate"
    if result in ("unsafe", "no_support"):
        return result
    return "other"


def structural_bucket(row: dict[str, Any]) -> tuple[Any, ...]:
    modulus = row_int(row, "filter_modulus")
    residue = row_int(row, "filter_residue")
    gcd = math.gcd(abs(modulus), abs(residue))
    threshold_bucket = int(max(0.0, min(1.0, row_number(row, "threshold"))) * 5.0)
    margin_bucket = int(row_number(row, "margin") * 20.0)
    return (
        str(row.get("task", "unknown")),
        str(row.get("subtask", "")),
        str(row.get("rule_kind", "unknown")),
        bool(row.get("conditional", False)),
        residue == 0,
        gcd > 1,
        modulus % 2 == 0,
        modulus % 3 == 0,
        modulus % 5 == 0,
        row_int(row, "action"),
        row_int(row, "feature"),
        row_int(row, "direction"),
        threshold_bucket,
        margin_bucket,
        str(row.get("sequence", "")),
    )


def coarse_structural_bucket(row: dict[str, Any]) -> tuple[Any, ...]:
    modulus = row_int(row, "filter_modulus")
    residue = row_int(row, "filter_residue")
    gcd = math.gcd(abs(modulus), abs(residue))
    return (
        "coarse",
        str(row.get("task", "unknown")),
        str(row.get("subtask", "")),
        str(row.get("rule_kind", "unknown")),
        bool(row.get("conditional", False)),
        residue == 0,
        gcd > 1,
        modulus % 2 == 0,
        modulus % 3 == 0,
        modulus % 5 == 0,
        row_int(row, "action"),
        row_int(row, "feature"),
        row_int(row, "direction"),
        str(row.get("sequence", "")),
    )


def split_rows(
    rows: list[dict[str, Any]], split_value: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train = [row for row in rows if row_int(row, "split_key") <= split_value]
    eval_rows = [row for row in rows if row_int(row, "split_key") > split_value]
    if train and eval_rows:
        return train, eval_rows
    train = [row for index, row in enumerate(rows) if index % 2 == 0]
    eval_rows = [row for index, row in enumerate(rows) if index % 2 == 1]
    return train, eval_rows


def train_reason_tables(
    rows: list[dict[str, Any]]
) -> tuple[dict[tuple[Any, ...], collections.Counter[str]], collections.Counter[str]]:
    buckets: dict[tuple[Any, ...], collections.Counter[str]] = {}
    global_counts: collections.Counter[str] = collections.Counter()
    for row in rows:
        label = reason_label(row)
        global_counts[label] += 1
        buckets.setdefault(structural_bucket(row), collections.Counter())[label] += 1
        buckets.setdefault(coarse_structural_bucket(row), collections.Counter())[label] += 1
    return buckets, global_counts


def smoothed_rates(
    row: dict[str, Any],
    buckets: dict[tuple[Any, ...], collections.Counter[str]],
    global_counts: collections.Counter[str],
) -> dict[str, float]:
    if str(row.get("task", "")) != "delayed_claim":
        counts = buckets.get(structural_bucket(row), global_counts)
    else:
        counts = collections.Counter(global_counts)
        for weight, bucket in (
            (3, coarse_structural_bucket(row)),
            (8, structural_bucket(row)),
        ):
            for label, count in buckets.get(bucket, {}).items():
                counts[label] += weight * count
    total = sum(counts.values()) + len(RESULT_LABELS)
    return {label: (counts.get(label, 0) + 1) / total for label in RESULT_LABELS}


def reason_score(
    row: dict[str, Any],
    buckets: dict[tuple[Any, ...], collections.Counter[str]],
    global_counts: collections.Counter[str],
) -> float:
    rates = smoothed_rates(row, buckets, global_counts)
    return (
        2.5 * rates["accepted"]
        + 0.4 * rates["safe_duplicate"]
        - 2.5 * rates["unsafe"]
        - 0.3 * rates["no_support"]
        - 0.8 * rates["other"]
    )


def rule_key(row: dict[str, Any]) -> str:
    return "{task}:{subtask}:{id}:{kind}:{conditional}:{state_mod}:{state_res}:{filter_mod}:{filter_res}:{action}:{feature}:{direction}:{threshold}:{margin}".format(
        task=str(row.get("task", "unknown")),
        subtask=str(row.get("subtask", "")),
        id=str(row.get("rule_id", "")),
        kind=str(row.get("rule_kind", "unknown")),
        conditional=1 if bool(row.get("conditional", False)) else 0,
        state_mod=row_int(row, "state_modulus"),
        state_res=row_int(row, "state_residue"),
        filter_mod=row_int(row, "filter_modulus"),
        filter_res=row_int(row, "filter_residue"),
        action=row_int(row, "action"),
        feature=row_int(row, "feature"),
        direction=row_int(row, "direction"),
        threshold=f"{row_number(row, 'threshold'):.6g}",
        margin=f"{row_number(row, 'margin'):.6g}",
    )


def grammar_tie_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("task", "unknown")),
        str(row.get("subtask", "")),
        str(row.get("rule_kind", "unknown")),
        bool(row.get("conditional", False)),
        row_int(row, "state_modulus"),
        row_int(row, "state_residue"),
        row_int(row, "filter_modulus"),
        row_int(row, "filter_residue"),
        row_int(row, "action"),
        row_int(row, "feature"),
        row_int(row, "direction"),
        row_number(row, "threshold"),
        row_number(row, "margin"),
        str(row.get("sequence", "")),
        str(row.get("rule_id", "")),
        str(row.get("rule_text", "")),
    )


def residual_feature_names() -> list[str]:
    return [
        "bias",
        "split_key",
        "inverse_split_key",
        "filter_modulus",
        "filter_residue",
        "filter_residue_is_zero",
        "filter_common_factor",
        "model_score",
        "structural_score",
        "action",
        "feature",
        "direction",
        "threshold",
        "margin",
        "rule_kind_is_threshold",
        "rule_kind_is_utility_margin",
        *[f"filter_modulus_divisible_by_{prime}" for prime in RESIDUAL_DIVISOR_PRIMES],
    ]


def residual_features(row: dict[str, Any], max_split_key: int, max_modulus: int) -> list[float]:
    split_key = max(1, row_int(row, "split_key"))
    modulus = max(1, row_int(row, "filter_modulus"))
    residue = row_int(row, "filter_residue")
    gcd = math.gcd(abs(modulus), abs(residue))
    max_split = max(max_split_key, 1)
    max_mod = max(max_modulus, 1)
    rule_kind = str(row.get("rule_kind", ""))
    return [
        1.0,
        float(split_key) / float(max_split),
        1.0 / float(split_key),
        float(modulus) / float(max_mod),
        float(residue) / float(modulus),
        1.0 if residue == 0 else 0.0,
        float(gcd) / float(modulus),
        row_number(row, "model_score"),
        row_number(row, "structural_score"),
        row_number(row, "action") / 4.0,
        row_number(row, "feature") / 8.0,
        row_number(row, "direction"),
        row_number(row, "threshold"),
        row_number(row, "margin"),
        1.0 if rule_kind == "threshold" else 0.0,
        1.0 if rule_kind == "utility_margin" else 0.0,
        *[1.0 if modulus % prime == 0 else 0.0 for prime in RESIDUAL_DIVISOR_PRIMES],
    ]


def residual_target(row: dict[str, Any], max_train_utility: float) -> float:
    label = reason_label(row)
    if label == "accepted":
        utility = row_number(row, "train_marginal_utility")
        scale = math.log1p(max(max_train_utility, 1.0))
        return 1.0 + 0.25 * math.log1p(max(utility, 0.0)) / scale
    if label == "safe_duplicate":
        return 0.1
    if label == "no_support":
        return -0.4
    if label == "unsafe":
        return -1.0
    return -0.2


def safety_target(row: dict[str, Any]) -> float:
    return 1.0 if reason_label(row) in SAFE_LABELS else -1.0


def standardize_features(
    feature_rows: list[list[float]],
) -> tuple[list[list[float]], list[float], list[float]]:
    if not feature_rows:
        return [], [], []
    feature_count = len(feature_rows[0])
    means = [0.0] * feature_count
    scales = [1.0] * feature_count
    for feature_index in range(1, feature_count):
        means[feature_index] = sum(row[feature_index] for row in feature_rows) / len(feature_rows)
        variance = (
            sum((row[feature_index] - means[feature_index]) ** 2 for row in feature_rows)
            / len(feature_rows)
        )
        scales[feature_index] = math.sqrt(variance) or 1.0
    standardized: list[list[float]] = []
    for row in feature_rows:
        standardized.append(
            [
                row[0],
                *[
                    (row[feature_index] - means[feature_index]) / scales[feature_index]
                    for feature_index in range(1, feature_count)
                ],
            ]
        )
    return standardized, means, scales


def fit_linear_model(
    feature_rows: list[list[float]],
    targets: list[float],
) -> tuple[list[float], list[float], list[float]]:
    standardized, means, scales = standardize_features(feature_rows)
    weights = [0.0] * (len(standardized[0]) if standardized else len(residual_feature_names()))

    if standardized:
        for _epoch in range(RESIDUAL_EPOCHS):
            gradients = [0.0] * len(weights)
            for features, target in zip(standardized, targets):
                prediction = sum(weight * value for weight, value in zip(weights, features))
                error = prediction - target
                for index, value in enumerate(features):
                    gradients[index] += error * value
            scale = 1.0 / float(len(standardized))
            for index in range(len(weights)):
                gradients[index] = gradients[index] * scale + RESIDUAL_L2 * weights[index]
                weights[index] -= RESIDUAL_LEARNING_RATE * gradients[index]

    return weights, means, scales


def train_residual_model(
    train_rows: list[dict[str, Any]],
    all_rows: list[dict[str, Any]],
    buckets: dict[tuple[Any, ...], collections.Counter[str]],
    global_counts: collections.Counter[str],
) -> dict[str, Any]:
    max_split_key = max((row_int(row, "split_key") for row in all_rows), default=1)
    max_modulus = max((row_int(row, "filter_modulus") for row in all_rows), default=1)
    scored_rows = [(reason_score(row, buckets, global_counts), row) for row in train_rows]
    max_reason = max((score for score, _row in scored_rows), default=0.0)
    residual_rows = [row for score, row in scored_rows if score > 0.0]
    if len(residual_rows) < 4:
        residual_rows = [row for score, row in scored_rows if score >= max_reason - 1.0e-12]
    if len(residual_rows) < 4:
        residual_rows = train_rows

    max_train_utility = max(
        (row_number(row, "train_marginal_utility") for row in train_rows),
        default=1.0,
    )
    utility_feature_rows = [
        residual_features(row, max_split_key=max_split_key, max_modulus=max_modulus)
        for row in residual_rows
    ]
    utility_targets = [residual_target(row, max_train_utility) for row in residual_rows]
    utility_weights, utility_means, utility_scales = fit_linear_model(
        utility_feature_rows,
        utility_targets,
    )
    safety_feature_rows = [
        residual_features(row, max_split_key=max_split_key, max_modulus=max_modulus)
        for row in train_rows
    ]
    safety_targets = [safety_target(row) for row in train_rows]
    safety_weights, safety_means, safety_scales = fit_linear_model(
        safety_feature_rows,
        safety_targets,
    )

    return {
        "feature_names": residual_feature_names(),
        "weights": utility_weights,
        "feature_means": utility_means,
        "feature_scales": utility_scales,
        "utility_weights": utility_weights,
        "utility_feature_means": utility_means,
        "utility_feature_scales": utility_scales,
        "safety_weights": safety_weights,
        "safety_feature_means": safety_means,
        "safety_feature_scales": safety_scales,
        "max_split_key": max_split_key,
        "max_modulus": max_modulus,
        "train_rows": len(residual_rows),
        "safety_train_rows": len(train_rows),
        "safety_train_labels": dict(collections.Counter(reason_label(row) for row in train_rows)),
        "train_reason_score": max_reason,
        "train_reason_score_min": min(
            (reason_score(row, buckets, global_counts) for row in residual_rows),
            default=0.0,
        ),
        "train_reason_labels": dict(collections.Counter(reason_label(row) for row in residual_rows)),
    }


def linear_model_score(
    row: dict[str, Any],
    model: dict[str, Any],
    weights_key: str,
    means_key: str,
    scales_key: str,
) -> float:
    features = residual_features(
        row,
        max_split_key=int(model["max_split_key"]),
        max_modulus=int(model["max_modulus"]),
    )
    means = model[means_key]
    scales = model[scales_key]
    standardized = [features[0]]
    standardized.extend(
        (features[index] - means[index]) / scales[index]
        for index in range(1, len(features))
    )
    return sum(
        float(weight) * value
        for weight, value in zip(model[weights_key], standardized)
    )


def residual_model_score(row: dict[str, Any], model: dict[str, Any]) -> float:
    return linear_model_score(
        row,
        model,
        "utility_weights",
        "utility_feature_means",
        "utility_feature_scales",
    )


def residual_safety_score(row: dict[str, Any], model: dict[str, Any]) -> float:
    return linear_model_score(
        row,
        model,
        "safety_weights",
        "safety_feature_means",
        "safety_feature_scales",
    )


def trained_residual_rank_key(
    task: str,
    row: dict[str, Any],
    buckets: dict[tuple[Any, ...], collections.Counter[str]],
    global_counts: collections.Counter[str],
    residual_model: dict[str, Any],
) -> tuple[Any, ...]:
    if task == "delayed_claim":
        return (
            -residual_safety_score(row, residual_model),
            -residual_model_score(row, residual_model),
            -reason_score(row, buckets, global_counts),
            *grammar_tie_key(row),
        )
    return (
        -reason_score(row, buckets, global_counts),
        -residual_model_score(row, residual_model),
        *grammar_tie_key(row),
    )


def top_rule_details(
    selected: list[dict[str, Any]],
    buckets: dict[tuple[Any, ...], collections.Counter[str]],
    global_counts: collections.Counter[str],
    residual_model: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    details = []
    for row in selected[:5]:
        detail = {
            "rule": str(row.get("rule_text", "")),
            "label": reason_label(row),
            "reason_score": reason_score(row, buckets, global_counts),
            "model_score": row_number(row, "model_score"),
        }
        if residual_model is not None:
            detail["trained_residual_score"] = residual_model_score(row, residual_model)
            detail["trained_safety_score"] = residual_safety_score(row, residual_model)
        details.append(detail)
    return details


def evaluate_strategy(
    task: str,
    rows: list[dict[str, Any]],
    budgets: list[int],
    strategy: str,
    rank_key_fn: RankKey,
    buckets: dict[tuple[Any, ...], collections.Counter[str]],
    global_counts: collections.Counter[str],
    residual_model: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=rank_key_fn)
    out: list[dict[str, Any]] = []
    for budget in budgets:
        selected = ordered[:budget]
        counts = collections.Counter(reason_label(row) for row in selected)
        out.append(
            {
                "schema": "crlp.rejected_candidate_learning.summary.v1",
                "report_version": 2,
                "date": date.today().isoformat(),
                "task": task,
                "experiment": "rejected_candidate_reason_learning",
                "strategy": strategy,
                "budget": budget,
                "selected_rules": len(selected),
                "accepted_rules": counts.get("accepted", 0),
                "safe_duplicate_rules": counts.get("safe_duplicate", 0),
                "unsafe_rules": counts.get("unsafe", 0),
                "no_support_rules": counts.get("no_support", 0),
                "other_rules": counts.get("other", 0),
                "validation_utility": sum(
                    row_number(row, "validation_marginal_utility") for row in selected
                ),
                "validation_false_rejections": sum(
                    row_int(row, "validation_false_rejections")
                    for row in selected
                ),
                "train_false_rejections": sum(
                    row_int(row, "train_false_rejections") for row in selected
                ),
                "selected_rule_keys": [rule_key(row) for row in selected],
                "top_rules": [str(row.get("rule_text", "")) for row in selected[:5]],
                "top_rule_details": top_rule_details(
                    selected,
                    buckets,
                    global_counts,
                    residual_model,
                ),
            }
        )
    return out


def attach_prior_delta_metrics(rows: list[dict[str, Any]]) -> None:
    by_strategy: dict[str, dict[int, dict[str, Any]]] = {}
    for row in rows:
        by_strategy.setdefault(str(row["strategy"]), {})[int(row["budget"])] = row

    prior_rows = by_strategy.get("reason_prior", {})
    for row in rows:
        budget = int(row["budget"])
        prior = prior_rows.get(budget)
        if prior is None:
            continue
        prior_keys = set(prior["selected_rule_keys"])
        row_keys = set(row["selected_rule_keys"])
        overlap_count = len(prior_keys & row_keys)
        denom = max(1, min(len(prior_keys), len(row_keys)))
        row["prior_vs_aware_topk_overlap_count"] = overlap_count
        row["prior_vs_aware_topk_overlap"] = overlap_count / denom
        row["delta_accepted_rules_vs_reason_prior"] = (
            int(row["accepted_rules"]) - int(prior["accepted_rules"])
        )
        row["delta_validation_utility_vs_reason_prior"] = (
            float(row["validation_utility"]) - float(prior["validation_utility"])
        )


def build_strategy_rows(
    task: str,
    eval_rows: list[dict[str, Any]],
    budgets: list[int],
    buckets: dict[tuple[Any, ...], collections.Counter[str]],
    global_counts: collections.Counter[str],
    residual_model: dict[str, Any],
) -> list[dict[str, Any]]:
    strategy_rows: list[dict[str, Any]] = []
    strategy_rows.extend(
        evaluate_strategy(
            task,
            eval_rows,
            budgets,
            "model_score",
            lambda row: (-row_number(row, "model_score"), *grammar_tie_key(row)),
            buckets,
            global_counts,
            residual_model,
        )
    )
    strategy_rows.extend(
        evaluate_strategy(
            task,
            eval_rows,
            budgets,
            "reason_prior",
            lambda row: (-reason_score(row, buckets, global_counts), *grammar_tie_key(row)),
            buckets,
            global_counts,
            residual_model,
        )
    )
    strategy_rows.extend(
        evaluate_strategy(
            task,
            eval_rows,
            budgets,
            "reason_aware",
            lambda row: (
                -reason_score(row, buckets, global_counts),
                -row_number(row, "model_score"),
                *grammar_tie_key(row),
            ),
            buckets,
            global_counts,
            residual_model,
        )
    )
    strategy_rows.extend(
        evaluate_strategy(
            task,
            eval_rows,
            budgets,
            "trained_residual",
            lambda row: trained_residual_rank_key(
                task,
                row,
                buckets,
                global_counts,
                residual_model,
            ),
            buckets,
            global_counts,
            residual_model,
        )
    )
    attach_prior_delta_metrics(strategy_rows)
    return strategy_rows


def enrich_strategy_rows(
    strategy_rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    split_key_name: str,
    split_value: int,
    residual_model: dict[str, Any],
    selection_seed: int | None,
) -> None:
    for row in strategy_rows:
        row["train_rows"] = len(train_rows)
        row["eval_rows"] = len(eval_rows)
        row["split_key_name"] = split_key_name
        row["split_value"] = split_value
        row["selection_seed"] = selection_seed
        if split_key_name in {"filter_modulus", "modulus"}:
            row["split_filter_modulus"] = split_value
        row["trained_residual_train_rows"] = residual_model["train_rows"]
        row["trained_residual_train_reason_labels"] = residual_model["train_reason_labels"]
        row["trained_safety_train_rows"] = residual_model["safety_train_rows"]
        row["trained_safety_train_reason_labels"] = residual_model["safety_train_labels"]
        row["trained_residual_feature_names"] = residual_model["feature_names"]


def run_split_evaluation(
    task: str,
    rows: list[dict[str, Any]],
    budgets: list[int],
    split_key_name: str,
    split_value: int,
    selection_seed: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    train_rows, eval_rows = split_rows(rows, split_value)
    buckets, global_counts = train_reason_tables(train_rows)
    residual_model = train_residual_model(train_rows, rows, buckets, global_counts)
    strategy_rows = build_strategy_rows(
        task,
        eval_rows,
        budgets,
        buckets,
        global_counts,
        residual_model,
    )
    enrich_strategy_rows(
        strategy_rows,
        train_rows,
        eval_rows,
        split_key_name,
        split_value,
        residual_model,
        selection_seed,
    )
    return train_rows, eval_rows, strategy_rows, residual_model


def split_sweep_rows(
    task: str,
    split_key_name: str,
    all_strategy_rows: dict[int, list[dict[str, Any]]],
    budgets: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_value in sorted(all_strategy_rows):
        strategy_rows = all_strategy_rows[split_value]
        by_strategy = {
            (str(row["strategy"]), int(row["budget"])): row for row in strategy_rows
        }
        for budget in budgets:
            prior = by_strategy[("reason_prior", budget)]
            aware = by_strategy[("reason_aware", budget)]
            trained = by_strategy[("trained_residual", budget)]
            rows.append(
                {
                    "schema": "crlp.rejected_candidate_learning.split_sweep.v1",
                    "report_version": 2,
                    "date": date.today().isoformat(),
                    "task": task,
                    "experiment": "rejected_candidate_reason_learning",
                    "split_key_name": split_key_name,
                    "split_value": split_value,
                    "budget": budget,
                    "train_rows": prior["train_rows"],
                    "eval_rows": prior["eval_rows"],
                    "reason_prior_accepted_rules": prior["accepted_rules"],
                    "reason_prior_validation_utility": prior["validation_utility"],
                    "reason_prior_validation_false_rejections": prior[
                        "validation_false_rejections"
                    ],
                    "reason_aware_delta_accepted_rules": aware[
                        "delta_accepted_rules_vs_reason_prior"
                    ],
                    "reason_aware_delta_validation_utility": aware[
                        "delta_validation_utility_vs_reason_prior"
                    ],
                    "trained_residual_accepted_rules": trained["accepted_rules"],
                    "trained_residual_validation_utility": trained["validation_utility"],
                    "trained_residual_validation_false_rejections": trained[
                        "validation_false_rejections"
                    ],
                    "trained_residual_delta_accepted_rules": trained[
                        "delta_accepted_rules_vs_reason_prior"
                    ],
                    "trained_residual_delta_validation_utility": trained[
                        "delta_validation_utility_vs_reason_prior"
                    ],
                    "trained_residual_topk_overlap": trained[
                        "prior_vs_aware_topk_overlap"
                    ],
                    "trained_residual_nonnegative_vs_reason_prior": (
                        int(trained["delta_accepted_rules_vs_reason_prior"]) >= 0
                        and float(
                            trained["delta_validation_utility_vs_reason_prior"]
                        )
                        >= 0.0
                        and int(trained["validation_false_rejections"]) == 0
                    ),
                }
            )
    return rows


def stable_residual_pass(sweep_rows: list[dict[str, Any]], budget: int) -> bool:
    budget_rows = [row for row in sweep_rows if int(row["budget"]) == budget]
    return bool(budget_rows) and all(
        bool(row["trained_residual_nonnegative_vs_reason_prior"])
        for row in budget_rows
    )


def result_table_lines(run: LearningRun, budgets: list[int]) -> list[str]:
    by_strategy: dict[str, dict[int, dict[str, Any]]] = {}
    for row in run.strategy_rows:
        by_strategy.setdefault(str(row["strategy"]), {})[int(row["budget"])] = row
    lines = [
        "| Strategy | Budget | Accepted | Safe duplicate | Unsafe | No support | Validation utility | Validation false rejections |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for budget in budgets:
        for strategy in STRATEGIES:
            row = by_strategy[strategy][budget]
            lines.append(
                "| {strategy} | {budget} | {accepted} | {duplicate} | {unsafe} | {no_support} | {utility:.0f} | {false} |".format(
                    strategy=strategy,
                    budget=budget,
                    accepted=row["accepted_rules"],
                    duplicate=row["safe_duplicate_rules"],
                    unsafe=row["unsafe_rules"],
                    no_support=row["no_support_rules"],
                    utility=float(row["validation_utility"]),
                    false=row["validation_false_rejections"],
                )
            )
    return lines


def metric_cell(row: dict[str, Any]) -> str:
    return "{accepted}/{unsafe}/{false}/{utility:.0f}".format(
        accepted=row["accepted_rules"],
        unsafe=row["unsafe_rules"],
        false=row["validation_false_rejections"],
        utility=float(row["validation_utility"]),
    )


def paper_grade_table_lines(run: LearningRun, budgets: list[int]) -> list[str]:
    by_strategy: dict[str, dict[int, dict[str, Any]]] = {}
    for row in run.strategy_rows:
        by_strategy.setdefault(str(row["strategy"]), {})[int(row["budget"])] = row
    lines = [
        "| Budget | Baseline candidate score A/U/FR/Utl | Reason prior A/U/FR/Utl | Trained residual A/U/FR/Utl |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for budget in budgets:
        lines.append(
            "| {budget} | {baseline} | {prior} | {residual} |".format(
                budget=budget,
                baseline=metric_cell(by_strategy["model_score"][budget]),
                prior=metric_cell(by_strategy["reason_prior"][budget]),
                residual=metric_cell(by_strategy["trained_residual"][budget]),
            )
        )
    return lines


def compact_metrics(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "accepted": int(row["accepted_rules"]),
        "safe_duplicate": int(row["safe_duplicate_rules"]),
        "unsafe": int(row["unsafe_rules"]),
        "no_support": int(row["no_support_rules"]),
        "false_rejections": int(row["validation_false_rejections"]),
        "utility": float(row["validation_utility"]),
    }


def paper_table_rows(runs: list[LearningRun], budgets: list[int]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for run in runs:
        by_strategy = {
            (str(row["strategy"]), int(row["budget"])): row
            for row in run.strategy_rows
        }
        for budget in budgets:
            baseline = by_strategy[("model_score", budget)]
            prior = by_strategy[("reason_prior", budget)]
            residual = by_strategy[("trained_residual", budget)]
            out.append(
                {
                    "schema": "crlp.rejected_candidate_learning.paper_table.v1",
                    "report_version": 1,
                    "date": date.today().isoformat(),
                    "task": run.task,
                    "experiment": "rejected_candidate_reason_learning",
                    "budget": budget,
                    "selection_seed": run.selection_seed,
                    "split_key_name": run.split_key_name,
                    "split_value": run.split_value,
                    "train_rows": len(run.train_rows),
                    "eval_rows": len(run.eval_rows),
                    "baseline_candidate_score": compact_metrics(baseline),
                    "reason_prior": compact_metrics(prior),
                    "trained_residual": compact_metrics(residual),
                }
            )
    return out


def reason_family_rows_for_split(
    task: str,
    eval_rows: list[dict[str, Any]],
    strategy_rows: list[dict[str, Any]],
    split_key_name: str,
    split_value: int,
    selection_seed: int | None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    eval_by_key = {rule_key(row): row for row in eval_rows}
    total_by_family = collections.Counter(reason_label(row) for row in eval_rows)
    utility_by_family = collections.Counter()
    false_by_family = collections.Counter()
    for row in eval_rows:
        family = reason_label(row)
        utility_by_family[family] += row_number(row, "validation_marginal_utility")
        false_by_family[family] += row_int(row, "validation_false_rejections")
    for strategy_row in strategy_rows:
        selected = [
            eval_by_key[key]
            for key in strategy_row["selected_rule_keys"]
            if key in eval_by_key
        ]
        selected_by_family = collections.Counter(reason_label(row) for row in selected)
        selected_utility = collections.Counter()
        selected_false = collections.Counter()
        for row in selected:
            family = reason_label(row)
            selected_utility[family] += row_number(row, "validation_marginal_utility")
            selected_false[family] += row_int(row, "validation_false_rejections")
        for family in RESULT_LABELS:
            out.append(
                {
                    "schema": "crlp.rejected_candidate_learning.reason_family_sweep.v1",
                    "report_version": 2,
                    "date": date.today().isoformat(),
                    "task": task,
                    "experiment": "rejected_candidate_reason_learning",
                    "strategy": strategy_row["strategy"],
                    "budget": strategy_row["budget"],
                    "selection_seed": selection_seed,
                    "split_key_name": split_key_name,
                    "split_value": split_value,
                    "reason_family": family,
                    "eval_rows": total_by_family.get(family, 0),
                    "eval_utility": float(utility_by_family.get(family, 0.0)),
                    "eval_false_rejections": int(false_by_family.get(family, 0)),
                    "selected_rows": selected_by_family.get(family, 0),
                    "selected_utility": float(selected_utility.get(family, 0.0)),
                    "selected_false_rejections": int(selected_false.get(family, 0)),
                }
            )
    return out


def reason_family_sweep_rows(runs: list[LearningRun]) -> list[dict[str, Any]]:
    return [row for run in runs for row in run.reason_family_rows]


def reason_family_table_lines(run: LearningRun, budget: int) -> list[str]:
    family_rows = [
        row
        for row in run.reason_family_rows
        if int(row["split_value"]) == run.split_value
    ]
    by_key = {
        (str(row["strategy"]), str(row["reason_family"])): row
        for row in family_rows
        if int(row["budget"]) == budget
    }
    lines = [
        "| Reason family | Baseline selected | Reason prior selected | Trained residual selected |",
        "| --- | ---: | ---: | ---: |",
    ]
    for family in RESULT_LABELS:
        lines.append(
            "| {family} | {baseline} | {prior} | {residual} |".format(
                family=family,
                baseline=by_key.get(("model_score", family), {}).get("selected_rows", 0),
                prior=by_key.get(("reason_prior", family), {}).get("selected_rows", 0),
                residual=by_key.get(("trained_residual", family), {}).get("selected_rows", 0),
            )
        )
    return lines


def guard_table_lines(run: LearningRun, budget: int) -> list[str]:
    budget_rows = [
        row for row in run.sweep_rows if int(row["budget"]) == budget
    ]
    budget_rows.sort(key=lambda row: int(row["split_value"]))
    lines = [
        "| Split | Reason prior accepted / utility | Trained residual accepted / utility | Delta utility | Guard |",
        "| ---: | ---: | ---: | ---: | --- |",
    ]
    for row in budget_rows:
        guard = "pass" if row["trained_residual_nonnegative_vs_reason_prior"] else "fail"
        lines.append(
            "| {split} | {prior_accepted} / {prior_utility:.0f} | {trained_accepted} / {trained_utility:.0f} | {delta:+.0f} | {guard} |".format(
                split=row["split_value"],
                prior_accepted=row["reason_prior_accepted_rules"],
                prior_utility=float(row["reason_prior_validation_utility"]),
                trained_accepted=row["trained_residual_accepted_rules"],
                trained_utility=float(row["trained_residual_validation_utility"]),
                delta=float(row["trained_residual_delta_validation_utility"]),
                guard=guard,
            )
        )
    return lines


def transfer_guard(row: dict[str, Any]) -> bool:
    return (
        int(row.get("delta_accepted_rules_vs_reason_prior", 0)) >= 0
        and float(row.get("delta_validation_utility_vs_reason_prior", 0.0)) >= 0.0
        and int(row.get("validation_false_rejections", 0)) == 0
    )


def write_markdown(
    path: Path,
    runs: list[LearningRun],
    budgets: list[int],
    family_transfer_rows: list[dict[str, Any]],
) -> None:
    first_budget = budgets[0]
    lines = [
        "# Rejected-Candidate Reason Learning",
        "",
        f"Date: {date.today().isoformat()}",
        "",
        "This experiment treats certificate result codes as reusable training signal. "
        "It now covers the original prime-gap rule search, an additional modular-gate "
        "family, and a second non-prime task using the same accepted / safe-duplicate / "
        "unsafe / no-support learning loop.",
        "",
        "## Coverage",
        "",
        "- `prime_gap`: original modular candidate filters from `crlp.rule_trace.v1`.",
        "- `modular_gate`: prime-constellation, squarefree/k-free, and Fibonacci/Lucas recurrence gates from `crlp.modular_gate.rule_trace.v1`.",
        "- `delayed_claim`: threshold and utility-margin claim gates from `crlp.delayed_claim.rule_trace.v1`.",
        "",
        "## Ranking Discipline",
        "",
        "- `model_score`: task-local neural or structural proposer score, with grammar-order ties.",
        "- `reason_prior`: learned reason-code rate only, with grammar-order ties.",
        "- `reason_aware`: learned reason-code rate first, then `model_score` as the explicit residual tiebreaker.",
        "- `trained_residual`: learned reason-code rate first, then a dedicated utility residual for prime/modular rows; on `delayed_claim`, where unsafe threshold rows are the failure mode, a trained safety head ranks predicted safe rows before the utility head.",
        "",
    ]

    for run in runs:
        train_counts = collections.Counter(reason_label(row) for row in run.train_rows)
        eval_counts = collections.Counter(reason_label(row) for row in run.eval_rows)
        stable_label = "PASS" if stable_residual_pass(run.sweep_rows, first_budget) else "FAIL"
        lines.extend(
            [
                f"## {run.title}",
                "",
                f"Raw trace: `{run.trace_path}`",
                "",
                f"- Split key: `{run.split_key_name}` <= `{run.split_value}` for train.",
                f"- Train rows: {len(run.train_rows)}; reason labels: {dict(train_counts)}",
                f"- Eval rows: {len(run.eval_rows)}; reason labels: {dict(eval_counts)}",
                "- Trained residual rows: {rows}; reason labels: {labels}".format(
                    rows=run.residual_model["train_rows"],
                    labels=run.residual_model["train_reason_labels"],
                ),
                f"- Selection seed: {run.selection_seed}",
                f"- Stable trained-residual claim at budget {first_budget}: {stable_label}",
                "",
                "### Paper-Grade Comparison",
                "",
                "Cells are accepted / unsafe / validation false rejections / validation utility.",
                "",
            ]
        )
        lines.extend(paper_grade_table_lines(run, budgets))
        lines.extend(
            [
                "",
                f"### Budget {first_budget} Reason-Family Sweep",
                "",
            ]
        )
        lines.extend(reason_family_table_lines(run, first_budget))
        lines.extend(
            [
                "",
                "### Budget Results",
                "",
            ]
        )
        lines.extend(result_table_lines(run, budgets))
        lines.extend(
            [
                "",
                "### Split Sweep Guard",
                "",
                "- Split grid: {splits}".format(
                    splits=", ".join(str(value) for value in run.split_grid)
                ),
                "",
            ]
        )
        lines.extend(guard_table_lines(run, first_budget))
        by_strategy = {
            (str(row["strategy"]), int(row["budget"])): row for row in run.strategy_rows
        }
        lines.extend(
            [
                "",
                f"### Budget {first_budget} Top Rules",
                "",
                "| Strategy | Top rules |",
                "| --- | --- |",
            ]
        )
        for strategy in STRATEGIES:
            row = by_strategy[(strategy, first_budget)]
            lines.append(
                "| {strategy} | {rules} |".format(
                    strategy=strategy,
                    rules="<br>".join(row["top_rules"]),
                )
            )
        lines.append("")

    if family_transfer_rows:
        by_key = {
            (
                str(row["heldout_family"]),
                str(row["strategy"]),
                int(row["budget"]),
            ): row
            for row in family_transfer_rows
        }
        families = sorted({str(row["heldout_family"]) for row in family_transfer_rows})
        lines.extend(
            [
                "## Modular Leave-One-Family-Out",
                "",
                "This transfer check trains on two modular families and ranks the held-out third family.",
                "",
                "| Held-out family | Reason prior accepted / utility / false | Trained residual accepted / utility / false | Delta utility | Guard |",
                "| --- | ---: | ---: | ---: | --- |",
            ]
        )
        for family in families:
            prior = by_key[(family, "reason_prior", first_budget)]
            trained = by_key[(family, "trained_residual", first_budget)]
            guard = "pass" if transfer_guard(trained) else "fail"
            lines.append(
                "| {family} | {prior_accepted} / {prior_utility:.0f} / {prior_false} | {trained_accepted} / {trained_utility:.0f} / {trained_false} | {delta:+.0f} | {guard} |".format(
                    family=family,
                    prior_accepted=prior["accepted_rules"],
                    prior_utility=float(prior["validation_utility"]),
                    prior_false=prior["validation_false_rejections"],
                    trained_accepted=trained["accepted_rules"],
                    trained_utility=float(trained["validation_utility"]),
                    trained_false=trained["validation_false_rejections"],
                    delta=float(trained["delta_validation_utility_vs_reason_prior"]),
                    guard=guard,
                )
            )
        lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "",
            "The prime-gap result remains the cleanest positive case. The added modular "
            "family checks whether the same reason-code learner survives beyond next-prime "
            "gap filters, while delayed-claim checks whether the loop transfers to a "
            "non-arithmetic decision task. A neural contribution claim should still be tied "
            "to the `trained_residual` row and the split-sweep guard, not to accidental "
            "trace-order agreement between `reason_prior` and `reason_aware`.",
            "",
            "## Trace Contract",
            "",
            "- Candidate rows carry a task, structural descriptors, `model_score`, train/eval support counters, false-rejection counters, and `verifier_result`.",
            "- Rejected rows are labels for the next ranker; held-out verifier results are not ranker inputs.",
            "- Trace artifact: `crlp.rejected_candidate_learning.summary.v1`, `crlp.rejected_candidate_learning.split_sweep.v1`, and `crlp.rejected_candidate_learning.family_transfer.v1` rows.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_learning_run(
    task: str,
    title: str,
    trace_path: Path,
    rows: list[dict[str, Any]],
    split_key_name: str,
    split_value: int,
    split_grid: list[int],
    budgets: list[int],
    selection_seed: int | None,
) -> LearningRun:
    train_rows, eval_rows, strategy_rows, residual_model = run_split_evaluation(
        task,
        rows,
        budgets,
        split_key_name,
        split_value,
        selection_seed,
    )
    grid = sorted(set(split_grid or []) | {split_value})
    all_split_strategy_rows: dict[int, list[dict[str, Any]]] = {
        split_value: strategy_rows
    }
    all_split_eval_rows: dict[int, list[dict[str, Any]]] = {
        split_value: eval_rows
    }
    for grid_split in grid:
        if grid_split == split_value:
            continue
        _train, _eval, split_strategy_rows, _residual = run_split_evaluation(
            task,
            rows,
            budgets,
            split_key_name,
            grid_split,
            selection_seed,
        )
        all_split_strategy_rows[grid_split] = split_strategy_rows
        all_split_eval_rows[grid_split] = _eval
    sweep_rows = split_sweep_rows(task, split_key_name, all_split_strategy_rows, budgets)
    reason_rows: list[dict[str, Any]] = []
    for grid_split in grid:
        reason_rows.extend(
            reason_family_rows_for_split(
                task,
                all_split_eval_rows[grid_split],
                all_split_strategy_rows[grid_split],
                split_key_name,
                grid_split,
                selection_seed,
            )
        )
    return LearningRun(
        task=task,
        title=title,
        trace_path=trace_path,
        split_key_name=split_key_name,
        split_value=split_value,
        split_grid=grid,
        selection_seed=selection_seed,
        rows=rows,
        train_rows=train_rows,
        eval_rows=eval_rows,
        strategy_rows=strategy_rows,
        residual_model=residual_model,
        sweep_rows=sweep_rows,
        reason_family_rows=reason_rows,
    )


def modular_leave_one_family_out_rows(
    rows: list[dict[str, Any]],
    budgets: list[int],
) -> list[dict[str, Any]]:
    families = sorted({str(row.get("subtask", "")) for row in rows})
    out: list[dict[str, Any]] = []
    for heldout_family in families:
        train_rows = [row for row in rows if str(row.get("subtask", "")) != heldout_family]
        eval_rows = [row for row in rows if str(row.get("subtask", "")) == heldout_family]
        if not train_rows or not eval_rows:
            continue
        buckets, global_counts = train_reason_tables(train_rows)
        residual_model = train_residual_model(train_rows, rows, buckets, global_counts)
        strategy_rows = build_strategy_rows(
            "modular_gate",
            eval_rows,
            budgets,
            buckets,
            global_counts,
            residual_model,
        )
        enrich_strategy_rows(
            strategy_rows,
            train_rows,
            eval_rows,
            "heldout_family",
            0,
            residual_model,
            None,
        )
        train_families = sorted(
            {str(row.get("subtask", "")) for row in train_rows}
        )
        for row in strategy_rows:
            row["schema"] = "crlp.rejected_candidate_learning.family_transfer.v1"
            row["transfer_mode"] = "leave_one_family_out"
            row["heldout_family"] = heldout_family
            row["train_families"] = train_families
        out.extend(strategy_rows)
    return out


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("build/float/prime_rule_search"))
    parser.add_argument("--input", type=Path, help="Existing prime crlp.rule_trace.v1 JSONL file")
    parser.add_argument("--modular-binary", type=Path, default=Path("build/float/modular_gate_search"))
    parser.add_argument("--modular-input", type=Path, help="Existing modular gate rule trace")
    parser.add_argument("--delayed-claim-binary", type=Path, default=Path("build/float/delayed_claim"))
    parser.add_argument("--delayed-claim-input", type=Path, help="Existing delayed-claim rule trace")
    parser.add_argument("--output-dir", type=Path, default=Path("build/float/rejected_candidate_learning"))
    parser.add_argument("--skip-prime", action="store_true")
    parser.add_argument("--skip-modular", action="store_true")
    parser.add_argument("--skip-delayed-claim", action="store_true")
    parser.add_argument("--moduli-range", default="3:97")
    parser.add_argument("--train-limit", type=int, default=100000)
    parser.add_argument("--test-limit", type=int, default=200000)
    parser.add_argument("--max-train-states", type=int, default=2000)
    parser.add_argument("--max-test-states", type=int, default=2000)
    parser.add_argument("--proposer-seed", type=int, default=20240511)
    parser.add_argument("--split-filter-modulus", type=int, default=61)
    parser.add_argument(
        "--split-filter-modulus-grid",
        type=int,
        nargs="*",
        default=list(DEFAULT_PRIME_SPLIT_GRID),
        help="Prime split grid; pass no values to use only the main split.",
    )
    parser.add_argument("--modular-max-states", type=int, default=512)
    parser.add_argument("--modular-max-actions", type=int, default=32)
    parser.add_argument("--modular-max-rules", type=int, default=16)
    parser.add_argument("--modular-selection-seed", type=int, default=20240511)
    parser.add_argument("--modular-split-modulus", type=int, default=11)
    parser.add_argument(
        "--modular-split-modulus-grid",
        type=int,
        nargs="*",
        default=list(DEFAULT_MODULAR_SPLIT_GRID),
        help="Modular-gate split grid; pass no values to use only the main split.",
    )
    parser.add_argument("--delayed-claim-max-rules", type=int, default=8)
    parser.add_argument("--delayed-claim-proposer", default="closed-loop")
    parser.add_argument("--delayed-claim-proposer-seed", type=int, default=20240511)
    parser.add_argument("--delayed-claim-split-rule-id", type=int, default=66)
    parser.add_argument(
        "--delayed-claim-split-rule-id-grid",
        type=int,
        nargs="*",
        default=list(DEFAULT_DELAYED_SPLIT_GRID),
        help="Delayed-claim structural template split grid; pass no values to use only the main split.",
    )
    parser.add_argument("--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs: list[LearningRun] = []
    family_transfer_rows: list[dict[str, Any]] = []

    if not args.skip_prime:
        trace_path = args.input or (args.output_dir / "prime_rule_trace.jsonl")
        if args.input is None:
            run_prime_rule_search(
                args,
                trace_path,
                args.output_dir / "prime_rule_search.stdout",
            )
        rows = parse_json_rows(trace_path)
        if not rows:
            raise RuntimeError(f"{trace_path} contains no prime rule trace rows")
        runs.append(
            build_learning_run(
                "prime_gap",
                "Prime-Gap Modular Filters",
                trace_path,
                rows,
                "filter_modulus",
                args.split_filter_modulus,
                sorted(set(args.split_filter_modulus_grid or []) | {args.split_filter_modulus}),
                args.budgets,
                args.proposer_seed,
            )
        )

    if not args.skip_modular:
        trace_path = args.modular_input or (args.output_dir / "modular_gate_trace.jsonl")
        if args.modular_input is None:
            run_modular_gate_search(
                args,
                trace_path,
                args.output_dir / "modular_gate_search.stdout",
            )
        rows = parse_json_rows(trace_path)
        if not rows:
            raise RuntimeError(f"{trace_path} contains no modular gate trace rows")
        family_transfer_rows = modular_leave_one_family_out_rows(rows, args.budgets)
        runs.append(
            build_learning_run(
                "modular_gate",
                "Additional Modular Gate Families",
                trace_path,
                rows,
                "modulus",
                args.modular_split_modulus,
                sorted(set(args.modular_split_modulus_grid or []) | {args.modular_split_modulus}),
                args.budgets,
                args.modular_selection_seed,
            )
        )

    if not args.skip_delayed_claim:
        trace_path = args.delayed_claim_input or (args.output_dir / "delayed_claim_rule_trace.jsonl")
        if args.delayed_claim_input is None:
            run_delayed_claim_rule_search(
                args,
                trace_path,
                args.output_dir / "delayed_claim_train.stdout",
                args.output_dir / "delayed_claim_rule_search.stdout",
                args.output_dir / "delayed_claim_parent.nnckpt",
            )
        rows = parse_json_rows(trace_path)
        if not rows:
            raise RuntimeError(f"{trace_path} contains no delayed-claim rule trace rows")
        runs.append(
            build_learning_run(
                "delayed_claim",
                "Delayed-Claim Threshold Gates",
                trace_path,
                rows,
                "rule_template",
                args.delayed_claim_split_rule_id,
                sorted(set(args.delayed_claim_split_rule_id_grid or []) | {args.delayed_claim_split_rule_id}),
                args.budgets,
                args.delayed_claim_proposer_seed,
            )
        )

    if not runs:
        raise RuntimeError("no rejected-candidate learning tasks were enabled")

    strategy_rows = [row for run in runs for row in run.strategy_rows]
    sweep_rows = [row for run in runs for row in run.sweep_rows]
    jsonl_path = args.output_dir / "reason_learning.jsonl"
    split_sweep_path = args.output_dir / "split_sweep.jsonl"
    family_transfer_path = args.output_dir / "modular_leave_one_family_out.jsonl"
    paper_table_path = args.output_dir / "paper_table.jsonl"
    reason_family_path = args.output_dir / "reason_family_sweep.jsonl"
    write_jsonl(jsonl_path, strategy_rows)
    write_jsonl(split_sweep_path, sweep_rows)
    write_jsonl(paper_table_path, paper_table_rows(runs, args.budgets))
    write_jsonl(reason_family_path, reason_family_sweep_rows(runs))
    if family_transfer_rows:
        write_jsonl(family_transfer_path, family_transfer_rows)
    write_markdown(
        args.output_dir / "summary.md",
        runs,
        args.budgets,
        family_transfer_rows,
    )
    print(f"wrote {jsonl_path}")
    print(f"wrote {split_sweep_path}")
    print(f"wrote {paper_table_path}")
    print(f"wrote {reason_family_path}")
    if family_transfer_rows:
        print(f"wrote {family_transfer_path}")
    print(f"wrote {args.output_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
