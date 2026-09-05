#!/usr/bin/env python3
"""Connect a verified Braid release to a tiny model and Signal measurements."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "integrations/signal-demo/config.json"
CONTROLLERS = {"goal_heading_no_safety", "flight_steer_to", "neural_raw", "neural_guarded"}
METRICS = {
    "utility_delta": "utility", "neural_utility": "utility", "heuristic_utility": "utility",
    "neural_damage_rate": "fraction", "heuristic_damage_rate": "fraction",
    "neural_death_rate": "fraction", "heuristic_death_rate": "fraction",
    "train_rows": "rows", "heldout_rows": "rows", "rollout_episodes": "episodes",
    "parameter_count": "parameters",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_config(config):
    require(config["schema"] == "crlp.signal_demo.config.v1", "Unexpected demo config schema")
    require(config["outcome"] == {"primary_metric": "utility_delta", "threshold": 0,
                                  "comparison": "neural_guarded minus flight_steer_to"},
            "This version supports the fixed neural_guarded minus flight_steer_to utility rule at zero")
    require(config["training"]["feature_mode"] == "live_observable", "The model needs live-observable features")
    seeds = [config["data"][split]["seed"] for split in ("train", "heldout")]
    require(len(set(seeds + [config["rollout"]["seed"]])) == 3, "Train, held-out, and rollout seeds must differ")


def command(args, *, cwd=ROOT, log=None, capture=False, timeout=1200):
    args = [str(arg) for arg in args]
    if log:
        with Path(log).open("w") as output:
            subprocess.run(args, cwd=cwd, stdout=output, stderr=subprocess.STDOUT,
                           check=True, timeout=timeout)
        return ""
    result = subprocess.run(args, cwd=cwd, stdout=subprocess.PIPE if capture else sys.stderr,
                            stderr=sys.stderr, text=True, check=True, timeout=timeout)
    return result.stdout if capture else ""


def simulation_stack():
    # Signal's flight_trace keeps several world_t values on its main thread stack.
    # This limit belongs to this command and its children, and leaves the shell alone.
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
    requested = 64 * 1024 * 1024
    if hard != resource.RLIM_INFINITY:
        requested = min(requested, hard)
    require(requested >= 32 * 1024 * 1024, "Signal trace generation needs a 32 MiB or larger stack limit")
    if soft != resource.RLIM_INFINITY and soft < requested:
        resource.setrlimit(resource.RLIMIT_STACK, (requested, hard))


def source_state(root, expected=None):
    root = Path(root).resolve()
    head = command(["git", "rev-parse", "HEAD"], cwd=root, capture=True).strip()
    require(expected is None or head == expected, f"Source pin differs at {root}: {head}")
    require(not command(["git", "diff", "HEAD", "--name-only"], cwd=root, capture=True).strip(),
            f"Commit tracked source changes first: {root}")
    return {"root": str(root), "commit": head}


def inspect_trace(path):
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(rows, f"Empty Signal trace: {path}")
    groups = {}
    seeds = set()
    for row in rows:
        require(row["schema"] == "signal.flight_counterfactual.v1", "Unexpected trace schema")
        require(row["certificate_status"] == "certified" and
                row["proof_kind"] == "deterministic_authoritative_replay", "Uncertified trace row")
        key = (int(row["seed"]), int(row["episode"]), int(row["tick"]))
        seeds.add(key[0])
        groups.setdefault(key, []).append(int(row["action"]))
        for field in ("utility", "damage_events", "death_events", "hull_loss"):
            require(math.isfinite(float(row[field])), f"Non-finite trace field: {field}")
    require(all(sorted(actions) == list(range(9)) for actions in groups.values()),
            "Each replay group must contain all nine actions exactly once")
    return {"rows": len(rows), "groups": len(groups), "seeds": sorted(seeds), "sha256": digest(path)}


def binding(name, path):
    return {"name": name, "path": str(Path(path).resolve()), "sha256": digest(path)}


def verify_bindings(bindings):
    for item in bindings:
        require(digest(item["path"]) == item["sha256"], f"Artifact changed: {item['name']}")


def prepare(args):
    config_path = Path(args.config).resolve()
    config = read_json(config_path)
    validate_config(config)
    out = Path(args.output).resolve()
    require(not out.exists(), f"Choose a fresh output directory: {out}")
    crlp = source_state(ROOT)
    signal = source_state(args.signal_root, config["sources"]["signal"]["commit"])
    braid = source_state(args.braid_root, config["sources"]["braid"]["commit"])
    out.mkdir(parents=True)
    logs = out / "logs"
    logs.mkdir()
    node = shutil.which("node")
    require(node, "Node 22 or newer is required")
    require(int(command([node, "-p", "process.versions.node.split('.')[0]"], capture=True)) >= 22,
            "Node 22 or newer is required")
    signal_root = Path(signal["root"])
    build = signal_root / "build-connected"
    command(["cmake", "-S", signal_root, "-B", build, "-DCMAKE_BUILD_TYPE=Release",
             "-DBUILD_TESTS_ONLY=ON", "-DBUILD_TOOLS=ON"], log=logs / "signal-configure.log")
    command(["cmake", "--build", build, "--target", "flight_trace", "--parallel", "4"],
            log=logs / "signal-build.log")
    command(["make", "-j4", "NN_BLAS=none", f"SIGNAL_FLIGHT_SIGNAL_DIR={signal_root}",
             "build/float/signal_flight", "build/float/signal_flight_rollout"], log=logs / "model-build.log")
    braid_root = Path(braid["root"])
    braid_cli = braid_root / "dist/cli.js"
    pnpm = shutil.which("pnpm")
    require(pnpm, "pnpm 10.19.0 is required to build Braid")
    command([pnpm, "build"], cwd=braid_root, log=logs / "braid-tool-build.log")
    require(braid_cli.is_file(), f"Install pinned Braid dependencies with pnpm install --frozen-lockfile in {braid_root}")
    cli = [node, braid_cli]
    simulation_stack()
    traces, sources = {}, []
    for split, target_split in (("train", "train"), ("heldout", "validation")):
        raw = out / "raw" / split
        raw.mkdir(parents=True)
        path = raw / "trace.csv"
        data = config["data"]
        generation = [build / "flight_trace", "--episodes", data[split]["episodes"],
                      "--ticks", data["ticks"], "--seed", data[split]["seed"], "--format", "csv",
                      "--counterfactual", "--sample-stride", data["sample_stride"],
                      "--horizon-ticks", data["horizon_ticks"], "--explore-random", data["explore_random"],
                      "--out", path]
        if data["wall_probe"]:
            generation.append("--wall-probe")
        command(generation, log=logs / f"generate-{split}.log")
        traces[split] = inspect_trace(path)
        snapshot = command(cli + ["snapshot", raw], capture=True)
        match = re.search(r"^sha256: ([a-f0-9]{64})$", snapshot, re.MULTILINE)
        require(match, "Braid snapshot did not return a SHA-256")
        sources.append({"id": split, "kind": "directory", "path": f"raw/{split}",
                        "extensions": [".csv"], "license": "LicenseRef-Signal-Simulation",
                        "attribution": "Generated from Signal by the connected flight demonstration",
                        "domain": "flight", "split": target_split,
                        "snapshot": {"revision": signal["commit"], "sha256": match[1]}})
    require(set(traces["train"]["seeds"]).isdisjoint(traces["heldout"]["seeds"]), "Trace splits share seeds")
    spec = {"apiVersion": "braid/v1", "kind": "DatasetBuild",
            "metadata": {"name": config["id"], "version": "v1", "description": "Deterministic Signal simulation observations"},
            "spec": {"purposes": ["research"], "sources": sources,
                     "rights": {"allowedLicenses": ["LicenseRef-Signal-Simulation"],
                                "requireAttribution": True, "requirePinnedSnapshots": True},
                     "chunking": {"targetCharacters": 100000000, "overlapCharacters": 0},
                     "quality": {"minimumCharacters": 1, "maximumCharacters": 100000000,
                                 "minimumAlphabeticRatio": 0, "maximumRepeatedLineRatio": 1,
                                 "nearDuplicateHammingDistance": 0},
                     "selection": {"strategy": "ranked", "seed": config["id"], "maximumDocuments": 2},
                     "evaluation": {"minimumDocuments": 2, "minimumSourceDocuments": 2,
                                    "maximumRejectionRatio": 0, "minimumDocumentsBySplit": {"train": 1, "validation": 1}},
                     "output": {"directory": "releases", "formats": ["jsonl"]}, "publication": {"target": "none"}}}
    write_json(out / "braid.json", spec)
    command(cli + ["build", out / "braid.json", "--output", out / "releases"], log=logs / "braid-build.log")
    manifests = list((out / "releases").glob("*/release.json"))
    require(len(manifests) == 1, "Expected one Braid release")
    manifest_path = manifests[0]
    manifest = read_json(manifest_path)
    release = manifest_path.parent
    require(manifest["status"] == "RELEASED", "Braid release did not pass its checks")
    command(cli + ["verify", release, "--expected", manifest["releaseId"]], log=logs / "braid-verify.log")
    exports = out / "exports"
    exports.mkdir()
    inputs = [binding("config", config_path), binding("braid-release", manifest_path),
              binding("demo-runner", Path(__file__)), binding("trace-generator", build / "flight_trace"),
              binding("python", sys.executable), binding("node", node), binding("braid-cli", braid_cli)]
    for split, braid_split in (("train", "train"), ("heldout", "validation")):
        records = [json.loads(line) for line in (release / "data" / f"{braid_split}.jsonl").read_text().splitlines()]
        require(len(records) == 1, f"Braid must preserve one complete {split} document")
        exported = exports / f"{split}.csv"
        exported.write_text(records[0]["text"] + "\n")
        require(digest(exported) == traces[split]["sha256"], f"Braid changed the {split} CSV bytes")
        inputs.append(binding(f"{split}-csv", exported))
    for name, filename in (("trainer", "signal_flight"), ("evaluator", "signal_flight_rollout")):
        inputs.append(binding(name, ROOT / "build/float" / filename))
    outputs = [{"name": name, "path": str(out / name)} for name in
               ("checkpoint.nnckpt", "training.jsonl", "rollout.jsonl", "result.json")]
    prepared = {"schema": "crlp.signal_demo.prepared.v1", "config": config, "output": str(out),
                "sources": {"crlplrimes": crlp, "signal": signal, "braid": braid},
                "braid": {"release_id": manifest["releaseId"], "release_directory": str(release),
                          "manifest_sha256": digest(manifest_path)},
                "traces": traces, "inputs": inputs, "outputs": outputs}
    write_json(out / "prepared.json", prepared)
    print(str(out / "prepared.json"))


def json_lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def result_metrics(training, rollout, config, model_hash):
    require(training["schema"] == "crlp.signal_flight.summary.v1", "Unexpected training schema")
    require(training["training_mode"] == "counterfactual_utility", "Training supervision differs from the configuration")
    require(training["feature_mode"] == "live_observable", "Training used outcome features")
    require(training["epochs"] == config["training"]["epochs"] and
            training["seed"] == config["training"]["seed"], "Training schedule differs from the configuration")
    require(math.isclose(training["learning_rate"], config["training"]["learning_rate"], rel_tol=1e-6),
            "Learning rate differs from the configuration")
    require(training["model_hash"] == model_hash, "Training summary names a different checkpoint")
    require(training["parameter_count"] == 2113, "Unexpected model size")
    require(len(rollout) == 4 and {r["controller"] for r in rollout} == CONTROLLERS,
            "Rollout needs four distinct matched controllers")
    for row in rollout:
        require(row["schema"] == "crlp.signal_flight_rollout.summary.v1", "Unexpected rollout schema")
        require(row["checkpoint_hash"] == model_hash, "Rollout used a different checkpoint")
        for key, config_key in (("episodes", "episodes"), ("ticks_per_episode", "ticks"), ("seed", "seed")):
            require(row[key] == config["rollout"][config_key], f"Rollout {key} differs from the frozen configuration")
        require(row["wall_probes"] is True, "Rollout must include wall probes")
        for key, value in row.items():
            if isinstance(value, (int, float)):
                require(math.isfinite(value), f"Non-finite rollout metric: {key}")
    by_controller = {r["controller"]: r for r in rollout}
    neural, baseline = by_controller["neural_guarded"], by_controller["flight_steer_to"]
    return {"utility_delta": neural["avg_utility"] - baseline["avg_utility"],
            "neural_utility": neural["avg_utility"], "heuristic_utility": baseline["avg_utility"],
            "neural_damage_rate": neural["damage_episode_rate"], "heuristic_damage_rate": baseline["damage_episode_rate"],
            "neural_death_rate": neural["death_rate"], "heuristic_death_rate": baseline["death_rate"],
            "train_rows": training["train_rows"], "heldout_rows": training["test_rows"],
            "rollout_episodes": config["rollout"]["episodes"], "parameter_count": training["parameter_count"]}


def execute(args):
    prepared = read_json(args.prepared)
    validate_config(prepared["config"])
    verify_bindings(prepared["inputs"])
    for state in prepared["sources"].values():
        source_state(state["root"], state["commit"])
    out = Path(prepared["output"])
    require(all(not Path(item["path"]).exists() for item in prepared["outputs"]), "Execution requires fresh output files")
    config = prepared["config"]
    paths = {item["name"]: item["path"] for item in prepared["inputs"]}
    for split in ("train", "heldout"):
        require(inspect_trace(paths[f"{split}-csv"]) == prepared["traces"][split], "Trace metadata changed")
    train = config["training"]
    command([paths["trainer"], "--train", paths["train-csv"], "--test", paths["heldout-csv"],
             "--epochs", train["epochs"], "--learning-rate", train["learning_rate"], "--seed", train["seed"],
             "--live-features", "--save", out / "checkpoint.nnckpt", "--summary", out / "training.jsonl"],
            log=out / "logs/train.log")
    simulation_stack()
    rollout_config = config["rollout"]
    command([paths["evaluator"], "--checkpoint", out / "checkpoint.nnckpt", "--episodes", rollout_config["episodes"],
             "--ticks", rollout_config["ticks"], "--seed", rollout_config["seed"], "--summary", out / "rollout.jsonl"],
            log=out / "logs/rollout.log")
    summaries = json_lines(out / "training.jsonl")
    require(len(summaries) == 1, "Expected one training summary")
    rollout = json_lines(out / "rollout.jsonl")
    model_hash = digest(out / "checkpoint.nnckpt")
    metrics = result_metrics(summaries[0], rollout, config, model_hash)
    require(metrics["train_rows"] == prepared["traces"]["train"]["rows"] and
            metrics["heldout_rows"] == prepared["traces"]["heldout"]["rows"], "Trainer did not consume the complete Braid exports")
    verify_bindings(prepared["inputs"])
    result = {"schema": "crlp.signal_demo.result.v1", "metrics": metrics, "rollout": rollout,
              "braid_release_id": prepared["braid"]["release_id"], "checkpoint_sha256": model_hash,
              "inputs": prepared["inputs"],
              "outputs": [binding(name, out / name) for name in ("checkpoint.nnckpt", "training.jsonl", "rollout.jsonl")],
              "scope": "One deterministic local flight study with fixed seeds and four matched controllers."}
    write_json(out / "result.json", result)
    print(json.dumps({"metrics": metrics}, allow_nan=False))


def run(args):
    prepare(args)
    out = Path(args.output).resolve()
    prepared_path = out / "prepared.json"
    prepared = read_json(prepared_path)
    config = prepared["config"]
    ilxyr = source_state(args.ilxyr_root, config["sources"]["ilxyr"]["commit"])
    ilxyr_root = Path(ilxyr["root"])
    command(["cargo", "build", "--locked", "-p", "ilxyr-cli"], cwd=ilxyr_root,
            log=out / "logs/ilxyr-build.log")
    ilxyr["cli"] = str(ilxyr_root / "target/debug/ilxyr")
    native = {"id": config["id"], "title": "Braid to tiny flight model to measured Signal rollout",
              "seed": config["rollout"]["seed"],
              "source": {"repository": "https://github.com/atimics/crlplrimes.git", **prepared["sources"]["crlplrimes"]},
              "ilxyr": ilxyr,
              "runner": {"program": str(Path(sys.executable).resolve()),
                         "args": [str(Path(__file__).resolve()), "execute", "--prepared", str(prepared_path)],
                         "cwd": str(ROOT), "timeout_seconds": 300},
              "inputs": [*prepared["inputs"], binding("prepared", prepared_path)], "outputs": prepared["outputs"],
              "metrics": [{"name": name, "unit": unit, "description": name.replace("_", " ")} for name, unit in METRICS.items()],
              "outcome": {"primary_metric": "utility_delta", "threshold": 0}}
    write_json(out / "ilxyr-config.json", native)
    command([shutil.which("node"), ROOT / "scripts/signal_demo_ilxyr.mjs", "--config", out / "ilxyr-config.json",
             "--output", out / "ilxyr", "--execute"], log=out / "logs/ilxyr-execution.log", timeout=400)
    verify(argparse.Namespace(output=str(out)))


def verify(args):
    out = Path(args.output).resolve()
    prepared = read_json(out / "prepared.json")
    validate_config(prepared["config"])
    require(Path(prepared["output"]).resolve() == out, "Prepared execution belongs to a different output directory")
    result = read_json(out / "result.json")
    verify_bindings(prepared["inputs"])
    verify_bindings(result["outputs"])
    native = read_json(out / "ilxyr/frozen-config.json")["config"]
    recorded_bindings = read_json(out / "ilxyr/artifact-bindings.json")
    verify_bindings(recorded_bindings)
    braid_cli = Path(prepared["sources"]["braid"]["root"]) / "dist/cli.js"
    node = next(item["path"] for item in prepared["inputs"] if item["name"] == "node")
    command([node, braid_cli, "verify", prepared["braid"]["release_directory"],
             "--expected", prepared["braid"]["release_id"]], log=out / "logs/braid-final-verify.log")
    ledger_check = json.loads(command([native["ilxyr"]["cli"], "verify", out / "ilxyr/ledger"], capture=True))
    require(ledger_check["valid"], "ilXyr ledger verification failed")
    status = json.loads(command([native["ilxyr"]["cli"], "status", out / "ilxyr/ledger", native["id"]], capture=True))
    expected = result_metrics(json_lines(out / "training.jsonl")[0], json_lines(out / "rollout.jsonl"),
                              prepared["config"], digest(out / "checkpoint.nnckpt"))
    receipt = read_json(out / "ilxyr/receipt.json")
    require(Path(receipt["ledger"]).resolve() == out / "ilxyr/ledger", "Receipt belongs to a different ledger")
    require(result["metrics"] == expected == receipt["metrics"], "Recorded metrics differ from the measured artifacts")
    verify_native_record(status, expected, receipt["outcome"], recorded_bindings)
    require(receipt["executed"] and receipt["outcome"] in ("improved", "not_improved"), "Experiment did not finish")
    require((receipt["outcome"] == "improved") == (expected["utility_delta"] > 0), "Outcome differs from its frozen rule")
    check = {"schema": "crlp.signal_demo.verification.v1", "valid": True, "outcome": receipt["outcome"],
             "metrics": expected, "braid_release_id": prepared["braid"]["release_id"],
             "checkpoint_sha256": digest(out / "checkpoint.nnckpt"), "ledger": ledger_check,
             "result_sha256": digest(out / "result.json")}
    write_json(out / "verification.json", check)
    print(json.dumps(check, indent=2, allow_nan=False))


def verify_native_record(status, metrics, outcome, bindings):
    require(status["latest_run"]["metrics"] == metrics and
            status["latest_evidence"]["resolved_outcome"] == outcome,
            "Metrics or outcome differ from the native ilXyr record")
    artifacts = status["latest_run"]["source_attestation"]["artifacts"]
    recorded = {item["path"]: item["sha256"] for item in artifacts}
    require(len(recorded) == len(artifacts), "Native artifact names are repeated")
    expected = {item["name"]: item["sha256"] for item in bindings}
    require(recorded == expected, "Artifact bindings differ from the native ilXyr record")


def distinct_runs(first, second):
    require(Path(first).resolve() != Path(second).resolve(), "Repeatability requires two distinct run directories")


def reproduce(args):
    """Create pinned companion checkouts and run the complete study twice."""
    workspace = Path(args.workspace).resolve()
    require(not workspace.exists(), f"Choose a fresh demonstration workspace: {workspace}")
    source_state(ROOT)
    config = read_json(DEFAULT_CONFIG)
    validate_config(config)
    workspace.mkdir(parents=True)
    for name, source in config["sources"].items():
        path = workspace / name
        path.mkdir()
        command(["git", "init", "--quiet"], cwd=path)
        command(["git", "remote", "add", "origin", source["repository"]], cwd=path)
        command(["git", "fetch", "--depth", "1", "origin", source["commit"]], cwd=path)
        command(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=path)
    command(["pnpm", "install", "--frozen-lockfile"], cwd=workspace / "braid")
    for name in ("run-1", "run-2"):
        run(argparse.Namespace(config=str(DEFAULT_CONFIG), signal_root=str(workspace / "signal"),
                               braid_root=str(workspace / "braid"), ilxyr_root=str(workspace / "ilxyr"),
                               output=str(workspace / name)))
    compare(argparse.Namespace(first=str(workspace / "run-1"), second=str(workspace / "run-2"),
                               receipt=str(workspace / "repeatability.json")))


def compare(args):
    runs = [Path(path).resolve() for path in (args.first, args.second)]
    distinct_runs(*runs)
    prepared = [read_json(path / "prepared.json") for path in runs]
    results = [read_json(path / "result.json") for path in runs]
    for path, before, result in zip(runs, prepared, results):
        verify(argparse.Namespace(output=str(path)))
        verify_bindings(before["inputs"])
        verify_bindings(result["outputs"])
    checks = {"config": prepared[0]["config"] == prepared[1]["config"],
              "source_commits": {name: state["commit"] for name, state in prepared[0]["sources"].items()} ==
                                {name: state["commit"] for name, state in prepared[1]["sources"].items()},
              "braid_release_id": results[0]["braid_release_id"] == results[1]["braid_release_id"],
              "exported_csv": prepared[0]["traces"] == prepared[1]["traces"],
              "checkpoint": results[0]["checkpoint_sha256"] == results[1]["checkpoint_sha256"],
              "metrics": results[0]["metrics"] == results[1]["metrics"],
              "all_controllers": results[0]["rollout"] == results[1]["rollout"]}
    require(all(checks.values()), f"Repeated runs differ: {checks}")
    receipt = {"schema": "crlp.signal_demo.repeatability.v1", "checks": checks,
               "runs": [binding(f"run-{i + 1}", path / "result.json") for i, path in enumerate(runs)]}
    if args.receipt:
        write_json(args.receipt, receipt)
    print(json.dumps(receipt, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for name in ("prepare", "run"):
        setup = sub.add_parser(name)
        setup.add_argument("--config", default=str(DEFAULT_CONFIG))
        setup.add_argument("--signal-root", required=True)
        setup.add_argument("--braid-root", required=True)
        setup.add_argument("--output", required=True)
        if name == "run":
            setup.add_argument("--ilxyr-root", required=True)
    execution = sub.add_parser("execute", help="Invoked by the frozen ilXyr experiment")
    execution.add_argument("--prepared", required=True)
    verification = sub.add_parser("verify")
    verification.add_argument("--output", required=True)
    reproduction = sub.add_parser("reproduce", help="Fetch pinned companions and run two complete local studies")
    reproduction.add_argument("--workspace", required=True)
    repeat = sub.add_parser("compare")
    repeat.add_argument("first")
    repeat.add_argument("second")
    repeat.add_argument("--receipt")
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "execute": execute, "verify": verify,
     "reproduce": reproduce, "compare": compare}[args.mode](args)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(f"signal-demo: {error}", file=sys.stderr)
        sys.exit(1)
