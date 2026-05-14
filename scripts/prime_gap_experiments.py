#!/usr/bin/env python3
"""Run repeatable prime-gap experiment sweeps and summarize policy metrics."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


DEFAULT_REGULARIZATION_GRID = "0.00:0.00,0.40:0.04,0.80:0.08,1.20:0.12"

ROW_RE = re.compile(
    r"^(?P<label>[A-Za-z0-9][A-Za-z0-9\- ]*?)\s+"
    r"(?P<temp>\d+(?:\.\d+)?)\s+"
    r"(?P<top1>\d+(?:\.\d+)?)%\s+"
    r"(?P<top10>\d+(?:\.\d+)?)%\s+"
    r"(?P<nll>\d+(?:\.\d+)?)\s+"
    r"(?P<ece>\d+(?:\.\d+)?)\s+"
    r"(?P<confidence>\d+(?:\.\d+)?)\s+"
    r"(?P<high_confidence_coverage>\d+(?:\.\d+)?)%\s+"
    r"(?P<high_confidence_accuracy>\d+(?:\.\d+)?)%$"
)
SUPERVISED_VALIDATION_RE = re.compile(
    r"^model supervised temperature=(?P<temperature>\d+(?:\.\d+)?) "
    r"validation_nll=(?P<nll>\d+(?:\.\d+)?)$"
)
REGULARIZED_VALIDATION_RE = re.compile(
    r"^model regularized-rl temperature=(?P<temperature>\d+(?:\.\d+)?) "
    r"validation_nll=(?P<nll>\d+(?:\.\d+)?)$"
)


METRIC_FIELDS = [
    "temp",
    "top1",
    "top10",
    "nll",
    "ece",
    "confidence",
    "high_confidence_coverage",
    "high_confidence_accuracy",
]
CSV_FIELDS = [
    "experiment",
    "seed",
    "train_limit",
    "test_limit",
    "supervised_epochs",
    "rl_epochs",
    "max_train_states",
    "max_test_states",
    "reference_weight",
    "entropy_weight",
    "label",
    "seeds",
    "validation_temperature",
    "validation_nll",
    *METRIC_FIELDS,
    "command",
    "log",
]


@dataclass(frozen=True)
class GridPoint:
    reference_weight: float
    entropy_weight: float


def parse_csv_ints(text: str) -> list[int]:
    values: list[int] = []
    for raw in text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        values.append(int(raw))
    if not values:
        raise ValueError("expected at least one integer")
    return values


def parse_regularization_grid(text: str) -> list[GridPoint]:
    points: list[GridPoint] = []
    for raw_point in text.split(","):
        raw_point = raw_point.strip()
        if not raw_point:
            continue
        parts = raw_point.split(":")
        if len(parts) != 2:
            raise ValueError(f"bad grid point '{raw_point}', expected reference:entropy")
        points.append(GridPoint(float(parts[0]), float(parts[1])))
    if not points:
        raise ValueError("expected at least one regularization grid point")
    return points


def parse_policy_rows(output: str) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    for line in output.splitlines():
        match = ROW_RE.match(line.strip())
        if match is None:
            continue
        data = match.groupdict()
        label = data.pop("label")
        rows[label] = {name: float(value) for name, value in data.items()}
    return rows


def parse_validation(output: str, pattern: re.Pattern[str]) -> tuple[float, float]:
    for line in output.splitlines():
        match = pattern.match(line.strip())
        if match is not None:
            return float(match.group("temperature")), float(match.group("nll"))
    return 1.0, float("nan")


def run_prime_gap(
    binary: Path,
    args: list[str],
    output_dir: Path,
    log_name: str,
) -> tuple[str, Path]:
    command = [str(binary), *args]
    print("$ " + " ".join(command), flush=True)
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    log_path = output_dir / "logs" / f"{log_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"command failed with exit code {result.returncode}: {log_path}")
    return result.stdout, log_path


def row_from_metrics(
    *,
    experiment: str,
    seed: int,
    train_limit: int,
    test_limit: int,
    supervised_epochs: int,
    rl_epochs: int,
    max_train_states: int,
    max_test_states: int,
    reference_weight: float,
    entropy_weight: float,
    label: str,
    validation_temperature: float,
    validation_nll: float,
    metrics: dict[str, float],
    command: list[str],
    log_path: Path,
) -> dict[str, str]:
    row = {
        "experiment": experiment,
        "seed": str(seed),
        "train_limit": str(train_limit),
        "test_limit": str(test_limit),
        "supervised_epochs": str(supervised_epochs),
        "rl_epochs": str(rl_epochs),
        "max_train_states": str(max_train_states),
        "max_test_states": str(max_test_states),
        "reference_weight": f"{reference_weight:.4f}",
        "entropy_weight": f"{entropy_weight:.4f}",
        "label": label,
        "validation_temperature": f"{validation_temperature:.4f}",
        "validation_nll": f"{validation_nll:.6f}",
        "command": " ".join(command),
        "log": str(log_path),
    }
    for field in METRIC_FIELDS:
        row[field] = f"{metrics[field]:.6f}"
    return row


def run_epoch_sweep(args: argparse.Namespace, output_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for seed in args.seeds:
        for epochs in args.supervised_epochs:
            run_args = [
                str(args.train_limit),
                str(args.test_limit),
                str(epochs),
                "0",
                str(args.max_train_states),
                str(args.max_test_states),
                f"{args.default_reference_weight:.4f}",
                f"{args.default_entropy_weight:.4f}",
                str(seed),
            ]
            log_name = f"epoch_seed-{seed}_epochs-{epochs}"
            output, log_path = run_prime_gap(args.binary, run_args, output_dir, log_name)
            policy_rows = parse_policy_rows(output)
            validation_temperature, validation_nll = parse_validation(
                output, SUPERVISED_VALIDATION_RE
            )
            for label in ("supervised", "supervised temp-scaled"):
                rows.append(
                    row_from_metrics(
                        experiment="epoch_sweep",
                        seed=seed,
                        train_limit=args.train_limit,
                        test_limit=args.test_limit,
                        supervised_epochs=epochs,
                        rl_epochs=0,
                        max_train_states=args.max_train_states,
                        max_test_states=args.max_test_states,
                        reference_weight=args.default_reference_weight,
                        entropy_weight=args.default_entropy_weight,
                        label=label,
                        validation_temperature=validation_temperature,
                        validation_nll=validation_nll,
                        metrics=policy_rows[label],
                        command=[str(args.binary), *run_args],
                        log_path=log_path,
                    )
                )
    return rows


def run_regularization_sweep(args: argparse.Namespace, output_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for seed in args.seeds:
        for point in args.regularization_grid:
            run_args = [
                str(args.train_limit),
                str(args.test_limit),
                str(args.selection_supervised_epochs),
                str(args.selection_rl_epochs),
                str(args.max_train_states),
                str(args.max_test_states),
                f"{point.reference_weight:.4f}",
                f"{point.entropy_weight:.4f}",
                str(seed),
            ]
            log_name = (
                f"regularization_seed-{seed}_ref-{point.reference_weight:.2f}"
                f"_ent-{point.entropy_weight:.2f}"
            )
            output, log_path = run_prime_gap(args.binary, run_args, output_dir, log_name)
            policy_rows = parse_policy_rows(output)
            validation_temperature, validation_nll = parse_validation(
                output, REGULARIZED_VALIDATION_RE
            )
            for label in ("regularized-rl", "regularized-rl temp-scaled"):
                rows.append(
                    row_from_metrics(
                        experiment="regularization_sweep",
                        seed=seed,
                        train_limit=args.train_limit,
                        test_limit=args.test_limit,
                        supervised_epochs=args.selection_supervised_epochs,
                        rl_epochs=args.selection_rl_epochs,
                        max_train_states=args.max_train_states,
                        max_test_states=args.max_test_states,
                        reference_weight=point.reference_weight,
                        entropy_weight=point.entropy_weight,
                        label=label,
                        validation_temperature=validation_temperature,
                        validation_nll=validation_nll,
                        metrics=policy_rows[label],
                        command=[str(args.binary), *run_args],
                        log_path=log_path,
                    )
                )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = CSV_FIELDS
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(
    rows: list[dict[str, str]],
    key_fields: tuple[str, ...],
    metric_fields: tuple[str, ...],
) -> list[dict[str, str]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in key_fields)].append(row)

    output: list[dict[str, str]] = []
    for key, group in sorted(grouped.items()):
        summary = {field: value for field, value in zip(key_fields, key)}
        summary["seeds"] = str(len({row["seed"] for row in group}))
        for field in metric_fields:
            values = [float(row[field]) for row in group]
            summary[field] = f"{mean(values):.6f}"
        output.append(summary)
    return output


def select_regularization(rows: list[dict[str, str]]) -> dict[str, str] | None:
    candidates = [row for row in rows if row["label"] == "regularized-rl temp-scaled"]
    if not candidates:
        return None

    aggregate = aggregate_rows(
        candidates,
        ("reference_weight", "entropy_weight"),
        (
            "validation_nll",
            "top1",
            "top10",
            "nll",
            "ece",
            "confidence",
            "high_confidence_coverage",
            "high_confidence_accuracy",
        ),
    )
    return min(aggregate, key=lambda row: float(row["validation_nll"]))


def markdown_table(rows: list[dict[str, str]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(title for title, _ in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(row.get(field, "") for _, field in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def write_summary(
    path: Path,
    epoch_rows: list[dict[str, str]],
    regularization_rows: list[dict[str, str]],
    selected: dict[str, str] | None,
) -> None:
    epoch_aggregate = aggregate_rows(
        [row for row in epoch_rows if row["label"] == "supervised temp-scaled"],
        ("supervised_epochs",),
        ("validation_nll", "top1", "top10", "nll", "ece"),
    )
    regularization_aggregate = aggregate_rows(
        [
            row
            for row in regularization_rows
            if row["label"] == "regularized-rl temp-scaled"
        ],
        ("reference_weight", "entropy_weight"),
        ("validation_nll", "top1", "top10", "nll", "ece", "confidence"),
    )

    lines = [
        "# Prime-Gap Experiment Summary",
        "",
        "Selection metric: mean validation NLL across configured seeds.",
        "",
        "## Supervised Epoch Sweep",
        "",
        markdown_table(
            epoch_aggregate,
            [
                ("Epochs", "supervised_epochs"),
                ("Seeds", "seeds"),
                ("Val NLL", "validation_nll"),
                ("Test top-1", "top1"),
                ("Test top-10", "top10"),
                ("Test NLL", "nll"),
                ("Test ECE", "ece"),
            ],
        ),
        "",
        "## Regularization Sweep",
        "",
        markdown_table(
            regularization_aggregate,
            [
                ("Ref", "reference_weight"),
                ("Entropy", "entropy_weight"),
                ("Seeds", "seeds"),
                ("Val NLL", "validation_nll"),
                ("Test top-1", "top1"),
                ("Test top-10", "top10"),
                ("Test NLL", "nll"),
                ("Test ECE", "ece"),
                ("Conf", "confidence"),
            ],
        ),
        "",
        "## Selected Regularization",
        "",
    ]

    if selected is None:
        lines.append("_No regularized RL rows were produced._")
    else:
        lines.extend(
            [
                (
                    f"`reference_weight={selected['reference_weight']}`, "
                    f"`entropy_weight={selected['entropy_weight']}`"
                ),
                "",
                markdown_table(
                    [selected],
                    [
                        ("Ref", "reference_weight"),
                        ("Entropy", "entropy_weight"),
                        ("Seeds", "seeds"),
                        ("Val NLL", "validation_nll"),
                        ("Test top-1", "top1"),
                        ("Test top-10", "top10"),
                        ("Test NLL", "nll"),
                        ("Test ECE", "ece"),
                    ],
                ),
            ]
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("build/prime_gap_experiments"))
    parser.add_argument("--train-limit", type=int, default=1_000_000)
    parser.add_argument("--test-limit", type=int, default=3_000_000)
    parser.add_argument("--max-train-states", type=int, default=15_000)
    parser.add_argument("--max-test-states", type=int, default=15_000)
    parser.add_argument("--supervised-epochs", type=parse_csv_ints, default=parse_csv_ints("1,2,4,8"))
    parser.add_argument("--selection-supervised-epochs", type=int, default=2)
    parser.add_argument("--selection-rl-epochs", type=int, default=1)
    parser.add_argument("--seeds", type=parse_csv_ints, default=parse_csv_ints("20240511"))
    parser.add_argument(
        "--regularization-grid",
        type=parse_regularization_grid,
        default=parse_regularization_grid(DEFAULT_REGULARIZATION_GRID),
    )
    parser.add_argument("--default-reference-weight", type=float, default=1.20)
    parser.add_argument("--default-entropy-weight", type=float, default=0.12)
    return parser


def main(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.binary = args.binary.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.binary.exists():
        parser.error(f"binary does not exist: {args.binary}")

    epoch_rows = run_epoch_sweep(args, output_dir)
    regularization_rows = run_regularization_sweep(args, output_dir)
    selected = select_regularization(regularization_rows)

    write_csv(output_dir / "epoch_sweep.csv", epoch_rows)
    write_csv(output_dir / "regularization_sweep.csv", regularization_rows)
    if selected is not None:
        write_csv(
            output_dir / "selected_regularization.csv",
            [selected],
            [
                "reference_weight",
                "entropy_weight",
                "seeds",
                "validation_nll",
                "top1",
                "top10",
                "nll",
                "ece",
                "confidence",
                "high_confidence_coverage",
                "high_confidence_accuracy",
            ],
        )
    write_summary(output_dir / "summary.md", epoch_rows, regularization_rows, selected)

    print(f"\nwrote {output_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
