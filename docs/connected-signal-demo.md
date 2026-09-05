# A connected Signal flight study

This demonstration turns Signal flight observations into a verified Braid
dataset, trains a 2,113-parameter C model on the exported data, and measures
four controllers in Signal. An ilXyr experiment freezes the inputs and outcome
rule before training. Its native ledger records the measurements and hashes
of the dataset, executable files, model, and result files.

The experiment reports utility, damage, and deaths. Its outcome is `improved`
when the guarded model's mean utility exceeds `flight_steer_to`, and
`not_improved` when the difference is zero or negative. Both are completed
measurements. The fixed seeds and small scenario set define the scope of the
result. Wider game performance and live deployment need further studies.

Utility rewards reaching the goal and distance gained. It subtracts damage,
deaths, stuck episodes, and elapsed time. The evaluator computes each episode as:

```text
2 × reached + clip(progress / 3000, -1, 1)
− 0.03 × damage_amount − 10 × died − stuck − 0.05 × ticks / max_ticks
```

The primary comparison uses the mean of these episode scores. Damage and
death rates remain separate measurements so a change in utility can be read
alongside its safety cost.

## Run it

Use a clean CRLPLRIMES checkout on Linux or macOS with Git, a C11 compiler,
Make, CMake, Node 22+, pnpm 10.19.0, and Rust 1.85+ available.
Braid and CRLPLRIMES are private repositories; the caller needs read access.
The initial setup downloads the source pins and locked Braid/Rust dependencies.
The simulation and model run locally on the CPU.

From the checkout:

```sh
make build/float/signal_demo NN_BLAS=none
./build/float/signal_demo reproduce --workspace /tmp/signal-connected-study
```

Choose a new workspace directory for each reproduction. The command fetches
the exact Signal, Braid, and ilXyr commits in
[`config.json`](../integrations/signal-demo/config.json), builds the tools,
runs the complete study twice, and compares the results. The default workload
uses 32 training episodes, 12 held-out episodes, three training epochs, and
24 rollout episodes for each of four controllers. Training, held-out data,
and rollout evaluation use separate seeds.

The trace generator needs a larger main-thread stack. The driver sets a
process-local limit of up to 64 MiB before launching it. A hard limit of at
least 32 MiB is required. The flight evaluator stores its world on the heap.

An existing Braid checkout can be supplied to `reproduce` with `--braid-root`.
The driver verifies its pinned commit. Existing pinned checkouts for all three
companions can also be used directly:

```sh
./build/float/signal_demo run \
  --signal-root /path/to/signal \
  --braid-root /path/to/braid \
  --ilxyr-root /path/to/ilxyr \
  --output /tmp/signal-study-run-1
```

Install Braid's locked dependencies with `pnpm install --frozen-lockfile` in
that checkout first. Each source pin must match the configuration and tracked
source edits must be committed. The driver builds fresh executable files and
binds their hashes before execution.

## What connects the projects

1. Signal's `flight_trace` simulates all nine WASD candidates from each shared
   state. Each row carries the action outcome and replay certificate.
2. Braid builds two whole CSV documents with separate training and validation
   splits. Directory snapshots pin the source bytes. The official Braid
   verifier checks the release. The exported documents must exactly match the
   original CSV hashes, and every group must retain all nine actions.
3. CRLPLRIMES consumes those exported CSV files. `--live-features` uses the
   observations available before action selection. The evaluator accepts only
   checkpoints marked `signal-flight-live-v2`, encoder version 2.
4. ilXyr compiles the declared study, admits it, and invokes training followed
   by evaluation. The four controllers share the same rollout scenarios:
   `goal_heading_no_safety`, `flight_steer_to`, `neural_raw`, and
   `neural_guarded`.
5. The official run stores measured metrics and source artifact hashes. The
   final verifier compares local result files with that native record, checks
   the Braid release again, and verifies the ledger.

The adapter uses one clearly labeled local fixture credit and an illustrative
forecast to exercise ilXyr's admission and settlement path. They have zero
monetary value. The experiment's evidence level is deterministic replay over
the declared seeds. The dataset contains generated simulation observations;
its source attribution is recorded as `LicenseRef-Signal-Simulation`.

## Read and check the result

Each run contains:

| File | Meaning |
|---|---|
| `prepared.json` | Source commits, fixed configuration, Braid release ID, CSV counts, and frozen inputs |
| `releases/<dataset>/<version>/<release-id>/release.json` | Official Braid release manifest |
| `exports/*.csv` | Exact data consumed by the model |
| `checkpoint.nnckpt` | Trained model, including its feature contract |
| `training.jsonl` | Held-out ranking, utility, and model-size measurements |
| `rollout.jsonl` | Four matched Signal controller summaries |
| `ilxyr/ledger/` | Native experiment, execution, settlement, and evidence records |
| `result.json` | Measurements with dataset, model, and output hashes |
| `verification.json` | Fresh verification against native records |

```sh
./build/float/signal_demo verify --output /tmp/signal-connected-study/run-1
./build/float/signal_demo compare \
  /tmp/signal-connected-study/run-1 /tmp/signal-connected-study/run-2 \
  --receipt /tmp/signal-connected-study/repeatability.json
```

Repeatability requires two distinct runs. It compares source commits, full
configuration, Braid release identity, CSV bytes, model bytes, all four
controller summaries, and measured metrics. Braid's manifest includes a
creation timestamp, so each execution binds its own manifest hash while the
portable comparison uses the release content identity. Exact repeatability
is measured within one toolchain; separate platform runs each carry their
own executable and environment bindings.

Contract checks:

```sh
make signal-demo-contract-check NN_BLAS=none
ILXYR_TEST_CLI=/path/to/ilxyr/target/debug/ilxyr \
  ILXYR_TEST_REPO=/path/to/ilxyr \
  node --test scripts/signal_demo_ilxyr.test.mjs
```

## CI source access

The connected-study workflow uses the read-only `BRAID_DEMO_READ_KEY` deploy
key to check out the pinned private Braid source. The key is an encrypted
CRLPLRIMES repository secret. Checkout uses `persist-credentials: false`; the
study consumes the resulting checkout. The workflow runs two complete studies
and uploads their native evidence records to the private repository artifacts.
