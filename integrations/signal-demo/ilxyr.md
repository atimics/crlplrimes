# Local Signal experiment record

`scripts/signal_demo_ilxyr.mjs` runs the Signal demo through the official
ilXyr local executor. It creates a fresh ledger, records four automated
contributions, freezes the experiment, records fixture accounting, and admits
the experiment before starting training. The local executor resolves the
outcome and settles the fixture forecast. The adapter then verifies the
ledger and exports native ilXyr evidence.

The four contribution records describe stages of one automated adapter.
The input checker and forecast have distinct service handles to exercise
ilXyr's role checks. The forecast probabilities are an illustrative fixture.
The single accounting credit has zero monetary value. Scientific conclusions
come from the measured Signal comparison and its stated sample scope.

## Setup

Use Node.js 22 or newer. Build the official CLI from a pinned ilXyr checkout:

```sh
cargo build -p ilxyr-cli --manifest-path /absolute/ilxyr/Cargo.toml
```

The resulting CLI is `/absolute/ilxyr/target/debug/ilxyr`. The adapter integration
tests were run with ilXyr commit
`46ce1e1e038f44e87fe76939366dbc9db13ffe84`.

## Configuration

The driver writes a JSON configuration with these fields:

```json
{
  "id": "signal.flight-demo.seed4242.v1",
  "title": "Signal local flight controller comparison",
  "seed": 4242,
  "source": {
    "repository": "https://github.com/atimics/crlplrimes",
    "commit": "FULL_CRLP_COMMIT",
    "root": "/absolute/crlplrimes"
  },
  "ilxyr": {
    "root": "/absolute/ilxyr",
    "commit": "46ce1e1e038f44e87fe76939366dbc9db13ffe84",
    "cli": "/absolute/ilxyr/target/debug/ilxyr"
  },
  "runner": {
    "program": "/absolute/python3",
    "args": ["/absolute/crlplrimes/scripts/signal_demo.py", "execute", "--prepared", "/absolute/run/prepared.json"],
    "cwd": "/absolute/crlplrimes",
    "timeout_seconds": 300
  },
  "inputs": [
    {"name": "runner.py", "path": "/absolute/crlplrimes/scripts/signal_demo.py", "sha256": "SHA256"},
    {"name": "prepared.json", "path": "/absolute/run/prepared.json", "sha256": "SHA256"},
    {"name": "train.csv", "path": "/absolute/run/train.csv", "sha256": "SHA256"}
  ],
  "outputs": [
    {"name": "checkpoint.nnckpt", "path": "/absolute/run/checkpoint.nnckpt"},
    {"name": "training.jsonl", "path": "/absolute/run/training.jsonl"},
    {"name": "rollout.jsonl", "path": "/absolute/run/rollout.jsonl"},
    {"name": "result.json", "path": "/absolute/run/result.json"}
  ],
  "metrics": [
    {"name": "utility_delta", "unit": "utility", "description": "Neural guarded mean utility minus matched flight_steer_to mean utility."}
  ],
  "outcome": {"primary_metric": "utility_delta", "threshold": 0}
}
```

Replace every placeholder with an actual path, full commit, or SHA-256. Extend
`inputs` to cover the Braid manifest, both exported CSV files, prepared settings,
source scripts, repository pins, and trainer/evaluator executables. Each input
and output needs a unique simple name and absolute file path. The first runner
argument is its script and must appear in the hashed input list. Commit the
source scripts before the authoritative run.

For a compiled runner, set `runner.mode` to `"native"`, set `runner.program`
to its absolute executable path, and include that executable in `inputs`
with its SHA-256. Its `args` are ordinary command arguments. Script runners
use `runner.mode: "script"`, which is also the default, and hash-bind the
script at `args[0]`. Both modes bind the program under `runtime/runner`.

Declare every returned metric in `metrics`. The runner writes exactly one JSON
object to stdout:

```json
{"metrics":{"utility_delta":-0.2}}
```

Diagnostics use stderr or files in the run directory. The runner receives a
cleared environment, so it should use explicit executable paths. The timeout
is between 1 and 300 seconds. Source and ilXyr checkout commits, all inputs,
the adapter, Node, runner executable, and CLI are checked before execution.
Input and runtime hashes are checked again after the runner finishes.

## Run and inspect

```sh
node scripts/signal_demo_ilxyr.mjs \
  --config /absolute/run/ilxyr-config.json \
  --output /absolute/run/ilxyr \
  --execute
```

Use a fresh `--output` directory. The declared result files must also be fresh.
Leaving off `--execute` prepares and admits a fresh experiment for inspection;
an execution uses another fresh output directory.

`utility_delta > 0` resolves `improved`; zero or a negative value resolves
`not_improved`. Both are completed measurements. A process failure, timeout,
or invalid metric envelope records `execution_failure` and returns a failing
adapter exit status.

The output directory contains the frozen configuration, native contribution
and experiment files, admission, run, status, verification, evidence export,
receipt, runner logs, and the `.ilxyr` ledger under `ledger/`. The official
run's `source_attestation.artifacts` contains the verified input/runtime/output
hashes. `artifact-bindings.json` maps those logical artifact names to local
files. Output bytes remain in the driver's declared artifact paths.

Verify ledger integrity with:

```sh
/absolute/ilxyr/target/debug/ilxyr verify /absolute/run/ilxyr/ledger
```

For later artifact verification, compare the current file SHA-256 values with
the source attestation in the ledger-verified native evidence export. The
adapter checks those bytes at execution time; ledger verification checks the
recorded objects and event chain. Preserve the complete run directory with
its artifact files when copying evidence.

## Adapter tests

```sh
ILXYR_TEST_CLI=/absolute/ilxyr/target/debug/ilxyr \
ILXYR_TEST_REPO=/absolute/ilxyr \
node --test scripts/signal_demo_ilxyr.test.mjs
```

The native tests check ordering before execution, all three numerical outcome
cases, output hash recording, and input mutation handling. With the two
environment variables omitted, the native integration cases are skipped.
The compiled-runner test uses a C compiler available as `cc`.
