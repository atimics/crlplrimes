# Multitask Transfer Research Plan

Date: 2026-05-17
Status: implementation plan with first instrumentation pass landed

## Purpose

The current shared-vs-single multitask result is asymmetric:

| Domain | Shared top-1 | Single top-1 | Delta |
| --- | ---: | ---: | ---: |
| `prime_gap` | 10.9% | 10.9% | +0.0pp |
| `grid_path` | 89.7% | 91.0% | -1.3pp |
| `delayed_claim` | 98.4% | 95.7% | +2.7pp |
| `arc_agi` | 14.2% | 42.8% | -28.6pp |

Those rows come from the one-seed `crlp.multitask.summary.v1` capture in
`results/latest/comparison.canonical.jsonl` and the imported paper table in
`paper/data.typ`. A separate three-seed matrix reports the same broad shape:
all-domain training is weak on `prime_gap`, easy on `grid_path` and
`delayed_claim`, and unsaturated on the fixed `arc_agi` cell-color row.

The goal of this plan is to decide whether this is a useful shared scorer
recipe, a capacity/training artifact, or only a verifier-gating substrate
ablation. The plan keeps scorer-transfer questions separate from symbolic
ARC-loader questions.

## Ground Rules

- The multitask ARC row is the fixed ARC-AGI-1 task `0520fde7` reduced to
  output-cell color scoring in the 96-feature shared action envelope.
- The public ARC loader (`arc_agi --task-dir`) measures transform/program
  candidates and certificate-gate coverage over ARC JSON tasks. It does not
  evaluate the `multitask_domain` scorer checkpoint.
- Exact certificate gates are scorer-independent unless a budgeted proposer or
  scorer is used to choose which candidates are evaluated.
- Multi-task conclusions should use repeated seeds; one-seed rows are useful
  for debugging and paper-table smoke, not final claims.
- Keep the shared feature envelope fixed while testing transfer. Larger
  domain-specific feature sets answer a different question.

## Current Baselines

### One-Seed Shared vs Single

Command source: `make multitask-domain`, captured through
`scripts/capture_results.py`.

| Domain | Shared top-1 | Single top-1 | Shared NLL | Single NLL | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| `prime_gap` | 10.9% | 10.9% | 2.6639 | 2.6970 | Null; no residue structure in shared envelope |
| `grid_path` | 89.7% | 91.0% | 0.3713 | 0.3497 | Roughly neutral, slight loss |
| `delayed_claim` | 98.4% | 95.7% | 0.0684 | 0.0989 | Positive transfer candidate |
| `arc_agi` | 14.2% | 42.8% | 2.1981 | 3.9258 | Large top-1 interference, but lower NLL under shared |

### Three-Seed Matrix

Command:

```sh
python3 scripts/multitask_ablation_matrix.py --seeds 20240516,20240517,20240518 --epochs 4 --steps-per-domain 3000 --eval-samples 1000
```

Mean trained-domain top-1:

| Variant | Prime | Grid | Claim | ARC |
| --- | ---: | ---: | ---: | ---: |
| Task tags | 11.77% | 90.37% | 95.50% | 26.13% |
| No task tags | 14.07% | 90.00% | 95.97% | 30.90% |

This suggests task tags are not currently buying much, and the fixed ARC row
remains the most fragile part of the shared action envelope.

### Public ARC Loader

The symbolic ARC loader is a separate surface. Grammar v10 loaded all 400
training and 400 evaluation tasks with no parse errors, certifying 24/400 train
tasks and 4/400 eval tasks. With optional induction enabled, the current phase-4
public-corpus run reaches 5/400 eval tasks with no regressions.

Use this to evaluate the verifier-gated ARC substrate, not fixed-scorer
multi-task transfer.

## Hypotheses

| ID | Claim | Falsified if |
| --- | --- | --- |
| H1 | Fixed-task ARC single-task top-1 is partly memorization; shared training should generalize better on same-family ARC OOD variants. | Shared <= single on held-out same-shape/same-family ARC variants across repeated seeds. |
| H2 | `delayed_claim` positive transfer is driven by specific sibling domains rather than all three uniformly. | Leave-one-out removals do not produce a stable or monotonic change in the claim gain. |
| H3 | A shared trunk with per-task adapter heads can recover ARC single-task accuracy while preserving the delayed-claim shared gain. | Adapter heads improve only ARC or only claim, or do not beat the plain shared trunk. |
| H4 | Shared training can improve calibration or NLL even when top-1 dips. | ECE/NLL are worse under shared for every task and every seed. |
| H5 | ARC negative transfer is mostly training budget or trunk capacity, not an unavoidable sharing failure. | Longer training and capacity sweeps both fail to close the ARC gap. |
| H6 | Exact ARC certificate coverage is scorer-invariant; only budgeted proposer/ranker coverage should vary with scorer choice. | Exact gate accepted-rule sets differ when the candidate grammar is unchanged, or budgeted coverage is reported as exact gate coverage. |

## Required Instrumentation

### I-MT.1: Add Multitask Calibration Fields

Status: implemented.

`crlp.multitask.summary.v1` emits top-k, NLL, mean rank, utility, baseline
steps, and calibration fields:

- `ece`
- `reliability_bins`
- `confidence_mean`
- `correct_confidence_mean`
- `incorrect_confidence_mean`

This makes E-MT.5 meaningful and aligns multitask rows with `crlp.compare.v1`.

### I-MT.2: Make Hidden Size Explicit

Status: implemented.

`examples/multitask_domain.c` accepts `--hidden h1xh2` and emits:

- `hidden`
- `parameters`
- `forward_flops`

This makes capacity sweeps auditable.

### I-MT.3: Add Training-Order Modes

Status: implemented.

Shared training accepts:

- `--training-order uniform_random`
- `--training-order round_robin`
- `--training-order arc_heavy`
- `--training-order prime_light`

The selected value is emitted in `training_order`.

### I-MT.4: Create Fixed-Envelope ARC OOD Variants

Status: pending.

To test scorer OOD without switching to the public ARC transform loader, add a
small held-out suite of same-shape cell-color tasks that fit the existing
96-feature envelope:

- same 7x3 input and 3x3 output shape
- output-color action space `0..9`
- train/test split analogous to `0520fde7`
- family metadata such as `split_panel_mask`, `intersection`, `union`,
  `recolor`, `translate`, or `sequence`

This is the scorer OOD testbed for H1. It should not be described as a public
ARC solver benchmark.

## Experiments

### E-MT.1: Longer Shared and Single Runs

Question: was the 24,000-update budget too small?

Run all-domain shared and per-task single baselines at:

```sh
./build/float/multitask_domain --tasks all --eval-tasks all --epochs 16 --steps-per-epoch 12000 --eval-samples 1000 --seed 20240516
```

Repeat for at least three seeds. Keep architecture fixed at `64x32`.

Tests: H5

Decision rule:

- If ARC shared-vs-single gap materially closes with no architecture change,
  the negative transfer was mostly a training-budget artifact.
- If delayed-claim gain disappears, the baseline gain was likely short-run
  variance.

### E-MT.2: Same-Envelope ARC OOD Scorer Test

Question: does shared training generalize better than single-task ARC when the
held-out tasks are not the memorized `0520fde7` test grid?

Train:

- single ARC scorer on the fixed ARC training task
- shared all-domain scorer
- optional no-task-tag shared scorer

Evaluate all on held-out same-envelope ARC variants from I-MT.4.

Tests: H1, H4

Decision rule:

- If shared > single on OOD top-1 or NLL while single wins on `0520fde7`, the
  paper gets a real transfer finding.
- If single wins both in-distribution and OOD, the ARC single-task row is not
  merely memorization under this envelope.

### E-MT.3: Delayed-Claim Leave-One-Out

Question: which sibling domain drives the delayed-claim gain?

The existing matrix already enumerates every non-empty training subset. Re-run
or filter it for:

- all domains
- all except `prime_gap`
- all except `grid_path`
- all except `arc_agi`
- `delayed_claim` only

Tests: H2

Decision rule:

- If removing one sibling collapses the claim gain, that sibling is the likely
  source of transfer.
- If every removal preserves the gain, the gain may come from regularization or
  update noise rather than semantic sibling transfer.

### E-MT.4: Adapter-Headed Sharing

Question: can a shared trunk avoid ARC interference while preserving claim gain?

Architecture:

```text
shared trunk: 96 -> 64 -> 32
task head:    32 -> 1
```

Each step updates the trunk and only the active task head. Compare:

- single-task baseline
- plain shared trunk
- shared trunk plus per-task heads

Tests: H3

Implementation note:

This likely needs either new engine support for exposed intermediate
activations/backprop, or a bespoke adapter model inside `multitask_domain.c`.
It is more than a trivial config change because `NeuralNet` is opaque through
the public API.

Decision rule:

- If adapter heads recover ARC top-1 and preserve delayed-claim gain, this
  becomes the multi-task recipe.
- If adapters only help ARC by destroying claim transfer, the recipe is not yet
  useful.

### E-MT.5: Calibration Audit

Question: is shared training improving uncertainty even when top-1 falls?

After I-MT.1, report for each `{shared, single} x task x seed`:

- top-1
- NLL
- ECE
- reliability bins
- mean confidence when correct
- mean confidence when incorrect

Tests: H4

Decision rule:

- If shared has worse top-1 but better NLL/ECE on ARC, frame ARC as a
  ranking/calibration tradeoff rather than pure degradation.
- If shared worsens both top-1 and calibration, call it negative transfer.

### E-MT.6: Capacity Sweep

Question: is the ARC gap a trunk capacity bottleneck?

After I-MT.2, run:

```text
96 -> 32 -> 16 -> 1
96 -> 64 -> 32 -> 1
96 -> 128 -> 64 -> 1
96 -> 256 -> 128 -> 1
```

Use the same seeds and update budgets as E-MT.1.

Tests: H5

Decision rule:

- If ARC improves monotonically with capacity while claim remains strong,
  capacity is the likely bottleneck.
- If larger trunks do not help, prioritize adapter heads or data/order changes.

### E-MT.7: ARC Gate and Budgeted Proposer Invariance

Question: which ARC measurements are actually scorer-dependent?

Run two separate checks:

1. Exact transform gate with unchanged candidate grammar.
2. Budgeted proposer/ranker coverage at fixed budgets.

Tests: H6

Decision rule:

- Exact gate accepted-rule sets should not change with scorer choice.
- Budgeted coverage may change if the scorer/proposer affects which candidates
  are surfaced inside budget. Report this as proposer/ranker sensitivity, not
  exact-gate correctness.

### E-MT.8: Curriculum and Sampling

Question: is negative ARC transfer caused by task sampling frequency or loss
landscape collision?

Compare:

- uniform random
- round robin
- ARC-heavy, 50% of updates
- prime-light, 10% of updates

Tests: H5, H3 precursor

Decision rule:

- If ARC-heavy recovers ARC without hurting claim, sampling is enough.
- If ARC-heavy hurts other tasks or still fails ARC, architecture is the next
  lever.

### E-ARC.1: Public ARC Induction Family Expansion

Question: can the symbolic ARC substrate lift public eval task certification?

This is separate from the multitask scorer. Use:

```sh
make arc-agi-induction-public ARC_AGI_TASK_DIR=/path/to/ARC-AGI/data ARC_AGI_MAX_TASKS=400
```

Prioritize family expansion only after adding per-task failure diagnostics:

- no candidate generated
- train mismatch
- unsupported shape
- conflicting train-consistent predictions
- correct partial cells but not exact

Decision rule:

- A useful wedge is at least +1 public eval task with zero regressions.
- A strong wedge is at least +2 public eval tasks plus clear train-side lift.

## Order of Attack

Completed instrumentation:

1. **I-MT.1 calibration fields.** Small schema extension, unlocks H4.
2. **I-MT.2 hidden-size reporting.** Makes capacity sweeps auditable.
3. **I-MT.3 training-order modes.** Makes curriculum/sampling tests explicit.

Next experiments:

1. **E-MT.1 longer run.** Establish whether baseline training budget was too
   small.
2. **E-MT.3 delayed-claim leave-one-out.** Existing script already provides most
   of the needed data shape.
3. **I-MT.4 and E-MT.2 ARC OOD scorer test.** Answers the memorization question
   without confusing scorer transfer with public ARC gate coverage.
4. **E-MT.7 gate/proposer invariance.** Cheap once ARC measurement labels are
   clean.
5. **E-MT.6 capacity sweep.** Only after the long-run baseline is known.
6. **E-MT.8 curriculum.** Use the implemented training-order modes once the
   dominant failure mode is known.
7. **E-MT.4 adapter heads.** Architectural work; do it after budget/capacity
   has a clear failure mode.
8. **E-ARC.1 induction-family expansion.** Parallel substrate track; do not
   present it as multitask-scorer OOD evidence.

## Paper Impact

| Finding | Paper change |
| --- | --- |
| Shared beats single on same-envelope ARC OOD | Add a real multi-task transfer finding: shared scoring improves OOD generalization despite worse fixed-task top-1. |
| Adapter heads recover ARC and preserve delayed-claim gain | Promote shared trunk + per-task heads + verifier gates as a recipe. |
| Longer training closes the ARC gap | Reframe negative transfer as under-training, not architectural incompatibility. |
| Capacity closes the ARC gap | Keep architecture simple but report explicit capacity scaling. |
| Exact ARC gate invariant, budgeted proposer varies | Strengthen the verifier-correctness claim while narrowing scorer claims to ranking efficiency. |
| Most hypotheses fail | Narrow the paper: the substrate is a verifier-gating contract; multitask scoring is an ablation with heterogeneous-task interference. |

## Out of Scope

- Larger domain-specific feature envelopes.
- Transformer or attention scorers.
- Claims about beating ARC-AGI SOTA.
- Public ARC solver claims from the fixed multitask scorer.
- Reward shaping for `delayed_claim`.
- Hand-tuning against public ARC evaluation task IDs.

## Success Criteria

The research track is ready to affect the paper when:

- every reported multitask row includes seed, update budget, hidden size,
  parameter count, top-k, NLL, and ECE;
- ARC scorer OOD and public ARC symbolic-gate results are reported in separate
  sections;
- all shared-vs-single claims use at least three seeds;
- public ARC gate claims report task-level exact certifications and regressions,
  not only cell accuracy;
- failures are turned into limitations rather than patched with mismatched
  architectures or feature envelopes.
