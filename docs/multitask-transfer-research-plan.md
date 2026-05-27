# Multitask Transfer Research Plan

Date: 2026-05-17
Status: R-MT.1/R-MT.2 captured; remaining work is follow-up analysis and architecture

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
`the Makefile results capture target`.

| Domain | Shared top-1 | Single top-1 | Shared NLL | Single NLL | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| `prime_gap` | 10.9% | 10.9% | 2.6639 | 2.6970 | Null; no residue structure in shared envelope |
| `grid_path` | 89.7% | 91.0% | 0.3713 | 0.3497 | Roughly neutral, slight loss |
| `delayed_claim` | 98.4% | 95.7% | 0.0684 | 0.0989 | Positive transfer candidate |
| `arc_agi` | 14.2% | 42.8% | 2.1981 | 3.9258 | Large top-1 interference, but lower NLL under shared |

### Three-Seed Matrix

Command:

```sh
make multitask-domain
```

Mean trained-domain top-1:

| Variant | Prime | Grid | Claim | ARC |
| --- | ---: | ---: | ---: | ---: |
| Task tags | 11.77% | 90.37% | 95.50% | 26.13% |
| No task tags | 14.07% | 90.00% | 95.97% | 30.90% |

This suggests task tags are not currently buying much, and the fixed ARC row
remains the most fragile part of the shared action envelope.

### Roadmap Capture: R-MT.1 and R-MT.2

Command source: `the retired multitask roadmap workflow`, captured in
`results/2026-05-17-multitask-roadmap`.

R-MT.1 uses three seeds at 16 epochs x 12,000 shared steps per seed
(192,000 total shared updates). Mean top-1:

| Domain | Shared | Single | Delta |
| --- | ---: | ---: | ---: |
| `prime_gap` | 14.07% | 13.63% | +0.43pp |
| `grid_path` | 90.90% | 91.13% | -0.23pp |
| `delayed_claim` | 98.00% | 97.90% | +0.10pp |
| `arc_agi` | 26.17% | 42.80% | -16.63pp |

R-MT.2 evaluates the same trained ARC scorer on 12 same-envelope held-out
7x3 -> 3x3 panel-composition tasks, still training ARC only on `0520fde7`.

| Model | OOD top-1 | OOD NLL | OOD ECE |
| --- | ---: | ---: | ---: |
| shared | 48.40% | 2.6278 | 0.4481 |
| single-task | 42.83% | 4.3723 | 0.4820 |

Interpretation: longer training narrows but does not close the fixed-task ARC
gap. The same-envelope OOD suite produces the first positive ARC transfer
signal (+5.57pp aggregate), but per-family rows are mixed, so this is a narrow
same-envelope finding rather than a public ARC generalization claim.

R-MT.3 keeps the delayed-claim transfer claim alive at the same long-run
budget. With 48k expected delayed-claim updates per seed, all-domain training
beats claim-only by +1.40pp top-1 (98.00% vs 96.60%). The sibling pattern is
specific rather than uniformly positive: removing `arc_agi` improves the row to
98.50%, while removing `grid_path` is the largest loss at 97.20%.

R-MT.6 rejects the simple "just make the trunk bigger" explanation for the
fixed ARC gap. Across `32x16`, `64x32`, `128x64`, and `256x128` trunks, the
smallest `32x16` trunk has the best shared fixed-ARC top-1 (39.20%) and the
least negative fixed-ARC shared-minus-single delta (-1.23pp). The largest trunk
partially recovers the `64x32` gap (-8.30pp versus -16.63pp), but not
monotonically and not enough to close the gap. Delayed-claim remains strong
across the sweep.

The follow-up ARC fixed-scorer diagnostic points to cell/color mode collapse,
not a core NN engine bug. Uniform random gives ARC about 47.9k updates and
round-robin gives exactly 48k, so the baseline is not simply starved. The
epoch-prefix scan shows the same prediction family after one epoch and after 16
epochs while train loss falls to near zero, so this is not late forgetting.
Learning rates from `0.0015` through `0.012` do not fix exactness. The learned
scorers discover the relevant color set `{0,2}` and crude background/foreground
priors, but do not infer the `0520fde7` panel geometry; the single-task scorer
converges to the same background-heavy wrong grid across all three seeds.
`docs/arc-json-vision-research-note.md` records the follow-up interpretation:
ARC is stored as JSON cell colors, but it assumes human-like objectness,
segmentation, geometry, topology, and latent layer/depth priors. The next ARC
experiment should therefore put those priors in the symbolic verifier grammar
first. The standalone ARC grammar now has an explicit layer/depth family
(`layer_overlap`, `layer_union`, `layer_xor`, `layer_left_only`,
`layer_right_only`), and the budgeted proposer has a split-panel structural
prior based on train shapes and foreground counts. The flat multitask scorer
remains the negative-control baseline rather than the place where these priors
are smuggled in as ordinary features.

R-MT.7 pins down the ARC scorer boundary. The public ARC transform gate and
budgeted proposer path do not currently consume a multitask scorer checkpoint,
so the `shared_scorer_context` versus `single_task_scorer_context` run is an
explicit invariance audit of the scorer-independent path. Across 400 train and
400 eval tasks, accepted rule sets, prediction hashes, loader summaries, and
budgeted proposer aggregates had zero mismatches.

### Public ARC Loader

The symbolic ARC loader is a separate surface. Grammar v15 loaded all 400
training and 400 evaluation tasks with no parse errors, certifying 24/400 train
tasks and 7/400 eval tasks. The train count is unchanged from the
pre-layer-overlap grammar v10 pass, while the explicit layer/depth family and
separator-line compression plus lattice motif repeat move evaluation cell
coverage from 187/98515 to 1057/98515 with no evaluation false rejections in
the budgeted proposer rows. Grammar v15 preserves that coverage while exposing
the `92e50de0` lattice verifier as a `tile_motif_repeat` hypothesis template
that the proposer can induce from train-pair lattice evidence.
The clean evidence surface and miss taxonomy are generated by
`make arc-agi-evidence-surface`.

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

Status: implemented.

To test scorer OOD without switching to the public ARC transform loader,
`examples/multitask_domain.c` now provides `--arc-eval-suite ood` and
`--arc-ood-task-breakdown`. The suite has 12 held-out same-shape cell-color
tasks that fit the existing 96-feature envelope:

- same 7x3 input and 3x3 output shape
- output-color action space `0..9`
- train/test split analogous to `0520fde7`
- family metadata such as `union`, `xor`, `left_only`, `right_only`,
  `copy_left`, `copy_right`, `neither`, `not_left`, `not_right`,
  `left_implies_right`, `right_implies_left`, and `row_any_intersection`

This is the scorer OOD testbed for H1.

### I-MT.5: Add Training-Dynamics and Fixed-ARC Grid Diagnostics

Status: implemented.

`examples/multitask_domain.c` emits `crlp.multitask.epoch.v1`, one row per
active domain per shared-training epoch, with:

- `loss_mean`
- `epoch_updates`
- `train_updates`
- `hidden`
- `learning_rate`
- `seed`
- `training_order`
- `task_set`

Fixed `0520fde7` ARC summary rows also include exact-grid diagnostics:

- `arc_fixed_grid_exact`
- `arc_fixed_cell_accuracy`
- `arc_fixed_prediction_hash`
- `arc_fixed_targets`
- `arc_fixed_predictions`
- `arc_fixed_confidence`
- `arc_fixed_target_probability`
- per-color target/predicted/correct counts

This addresses the R-MT.6 diagnostic gap: fixed ARC can now be read as a 3x3
grid/color failure rather than only as 1,000 repeated balanced cell samples, and
R-MT.8 can separate task starvation from forgetting or optimizer instability.

## Experiments

### E-MT.1: Longer Shared and Single Runs

Question: was the 24,000-update budget too small?

Status: captured in `results/2026-05-17-multitask-roadmap`.

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

Status: captured in `results/2026-05-17-multitask-roadmap`.

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

Status: captured in `results/2026-05-17-multitask-roadmap/rmt3_leave_one_out.md`.

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

Result:

- All-domain training is +1.40pp top-1 versus claim-only.
- Removing `grid_path` causes the largest loss versus all-domain (-0.80pp),
  making it the most likely positive sibling.
- Removing `arc_agi` improves delayed-claim by +0.50pp versus all-domain,
  so ARC appears to be mild interference for this target.
- Removing `prime_gap` is near-neutral (-0.17pp).

### E-MT.4: Adapter-Headed Sharing

Question: can a shared trunk avoid ARC interference while preserving claim gain?

Status: instrumented as `examples/multitask_domain.c --task-heads`; refresh the
canonical R-MT.4 bundle with `make multitask-adapter-heads`.

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

Run:

```sh
make multitask-domain
```

Implementation note: the first implementation uses a bespoke adapter model
inside `multitask_domain.c` instead of changing the opaque public `NeuralNet`
API. The emitted `shared_task_heads` rows carry the same
`crlp.multitask.summary.v1` schema, parameter/FLOP counters, calibration bins,
and fixed-ARC diagnostics as the `shared` and `single_task` rows.

Decision rule:

- If adapter heads recover ARC top-1 and preserve delayed-claim gain, this
  becomes the multi-task recipe.
- If adapters only help ARC by destroying claim transfer, the recipe is not yet
  useful.

### E-MT.5: Calibration Audit

Question: is shared training improving uncertainty even when top-1 falls?

Status: captured in `results/2026-05-17-multitask-roadmap/rmt5_calibration.md`.

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

Result:

- Fixed `0520fde7`: shared has worse top-1 and worse ECE (0.6995 vs 0.5589),
  but substantially better NLL (4.0827 vs 8.2754). This is a ranking/log-loss
  tradeoff, not a calibration win.
- Same-envelope ARC OOD: shared improves top-1, NLL, and ECE (ECE 0.4481 vs
  0.4820; NLL 2.6278 vs 4.3723). This supports the narrow OOD transfer
  interpretation from R-MT.2.
- Other fixed tasks have small ECE deltas; R-MT.5 is mainly an ARC diagnostic.

### E-MT.6: Capacity Sweep

Question: is the ARC gap a trunk capacity bottleneck?

Status: captured in `results/2026-05-17-multitask-roadmap/rmt6_capacity_sweep.md`.

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

Result:

- The fixed ARC gap does not improve monotonically with capacity. Shared
  fixed-ARC top-1 is 39.20% at `32x16`, 26.17% at `64x32`, 26.17% at
  `128x64`, and 34.50% at `256x128`.
- The smallest trunk has the least negative fixed-ARC shared-minus-single
  delta (-1.23pp). The largest trunk is better than `64x32` but still trails
  single-task by -8.30pp.
- Delayed-claim remains strong across the sweep, with shared-minus-single
  deltas from +0.10pp to +0.43pp.
- Capacity is therefore not the dominant explanation by itself. The follow-up
  diagnostic in `rmt_arc_failure_diagnostics.md` narrows the issue further to
  fixed ARC cell/color mode collapse under the current row formulation.
  Reproduce it with `make multitask-failure-diagnostics`; override
  `MULTITASK_FAILURE_DIAGNOSTIC_*` variables for a shorter local probe.
  R-MT.8 remains useful for sampling dynamics, but exact fixed-ARC scorer
  recovery likely needs a better ARC feature/training surface or adapter path.
  Before adding architecture, run the fully-trained tiny-net audit in E-MT.9 so
  the roadmap can distinguish undertraining, saturation, and missing cumulative
  protocol state.

### E-MT.7: ARC Gate and Budgeted Proposer Invariance

Question: which ARC measurements are actually scorer-dependent?

Status: captured in `results/2026-05-17-multitask-roadmap/rmt7_arc_gate_invariance.md`.

Run two separate checks:

1. Exact transform gate with unchanged candidate grammar.
2. Budgeted proposer/ranker coverage at fixed budgets.

Tests: H6

Decision rule:

- Exact gate accepted-rule sets should not change with scorer choice.
- Budgeted coverage may change if the scorer/proposer affects which candidates
  are surfaced inside budget. Report this as proposer/ranker sensitivity, not
  exact-gate correctness.

Result:

- The current public ARC transform gate and budgeted proposer path are
  scorer-independent: the harness labels `shared_scorer_context` and
  `single_task_scorer_context`, but both exercise the same ARC loader/gate path.
- On 400 train tasks and 400 eval tasks, exact gate/loader mismatches were `0`
  and budgeted proposer aggregate mismatches were `0`.
- This strengthens the verifier-correctness boundary while also narrowing the
  claim: R-MT.7 is not evidence that the current shared scorer improves public
  ARC search, because that scorer is not wired into this path.

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

### E-MT.9: Fully Trained Tiny-Net Audit

Question: have the current tiny scorers reached their practical limit under
ordinary training, or is there still headroom from longer training, restarts,
warm starts, and cumulative protocol artifacts?

Run a small capacity ladder under fixed seeds:

```text
96 -> 8 -> 4 -> 1
96 -> 16 -> 8 -> 1
96 -> 32 -> 16 -> 1
96 -> 64 -> 32 -> 1
```

Implemented runner:

```sh
make tiny-net-plateau-audit
```

The runner writes checkpointed learning curves, restart variance, best
checkpoint rows, plateau flags, parameter/FLOP counts, and a width-help verdict
under `build/float/tiny_net_plateau_audit/`. It also emits
`tiny_net_diagnostics.jsonl`, which classifies each domain/model variant as
weak-signal, seed-variance dominated, not plateaued, not width-limited, or a
capacity-helped-but-unsettled row before any architecture claim can be promoted.

For each row, report loss, top-1, NLL, ECE, validation utility, and
certificate-gated utility over checkpoints. Then rerun the best small trunk with
longer budgets and fresh restarts. Finally compare scratch training against
warm-start or artifact-fed variants that can consume rejected-candidate reason
codes and certified-library features.

Tests: H5, H6, and the paper-roadmap R-F items.

Decision rule:

- If extra steps, restarts, and wider trunks do not produce a material
  validation or certified-utility gain, treat the tiny net as fully trained for
  this protocol.
- If warm-start or artifact-fed rows improve while scratch rows plateau, frame
  the remaining headroom as cumulative substrate learning rather than raw
  network capacity.
- If a wider trunk wins after the same plateau checks, keep capacity as a
  controlled variable but do not make it the default explanation.

Current result:

- The 128-epoch audit remains underpowered: 11/48 plateau rows pass the
  threshold and 0/12 domain/variant verdicts are audit-ready. Longer training
  alone did not clear C7.
- The next capacity baseline is explicit and uses the same resumable/sharded
  runner:

```sh
make tiny-net-stronger-baseline
```

By default this extends the ladder to `128x64` and `256x128` through the same
128-epoch checkpoint grid. Use `TINY_NET_STRONGER_BASELINE_*` variables for
short smoke probes, sharding, or a shared resume cache.

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
4. **I-MT.4 same-envelope ARC OOD suite.** Adds fixed-envelope held-out ARC
   panel-composition tasks and task-level metadata.
5. **I-MT.5 epoch and fixed-ARC grid diagnostics.** Adds
   `crlp.multitask.epoch.v1` plus exact-grid/per-cell/per-color ARC fields.
6. **R-MT.4 adapter-head surface.** Adds `--task-heads` plus the native
   `make multitask-adapter-heads` report target; full roadmap results are
   pending.

Completed experiments:

1. **E-MT.1 longer run.** Three-seed 192k-update capture landed.
2. **E-MT.2 ARC OOD scorer test.** Shared is +5.57pp top-1 aggregate on the
   same-envelope OOD suite, mixed by family.
3. **E-MT.3 delayed-claim leave-one-out.** Positive transfer survives, with
   `grid_path` as the most likely positive sibling and `arc_agi` as mild
   interference.
4. **E-MT.5 calibration audit.** Reliability-bin CSVs and markdown summary
   landed in `results/2026-05-17-multitask-roadmap`.
5. **E-MT.6 capacity sweep.** Larger trunks do not monotonically close the
   fixed ARC shared-minus-single gap; delayed-claim stays strong.
6. **E-MT.7 gate/proposer invariance.** ARC accepted rule sets, prediction
   hashes, loader summaries, and budgeted proposer aggregates are invariant
   across the current scorer-context labels.
7. **E-MT.9 fully trained tiny-net audit.** The 128-epoch audit stayed
   underpowered, with 11/48 plateau rows and 0/12 audit-ready verdicts.

Next experiments:

1. **E-MT.9 stronger baseline ladder.** Run the `tiny-net-stronger-baseline`
   target to test whether modestly larger MLPs change the diagnostic verdicts
   under the same audit rules.
2. **E-MT.8 curriculum.** Use the implemented training-order modes once the
   dominant failure mode is known.
3. **E-MT.4 adapter heads.** Run the now-instrumented `--task-heads` comparison
   after plateau and cumulative-artifact checks have a clear failure mode.
4. **E-ARC.1 induction-family expansion.** Parallel substrate track; do not
   present it as multitask-scorer OOD evidence.

## Paper Impact

| Finding | Paper change |
| --- | --- |
| Shared beats single on same-envelope ARC OOD | Add a real multi-task transfer finding: shared scoring improves OOD generalization despite worse fixed-task top-1. |
| Adapter heads recover ARC and preserve delayed-claim gain | Promote shared trunk + per-task heads + verifier gates as a recipe. |
| Longer training closes the ARC gap | Reframe negative transfer as under-training, not architectural incompatibility. |
| Capacity closes the ARC gap | Keep architecture simple but report explicit capacity scaling. |
| Capacity does not close the ARC gap | Run the fully-trained tiny-net audit; prioritize cumulative artifacts, curriculum/sampling, and adapter heads over simply increasing trunk size. |
| Warm-start or artifact-fed tiny nets improve after scratch plateaus | Frame the remaining headroom as cumulative substrate learning: certified traces and rejected-candidate reasons improve future proposal/search. |
| Exact ARC gate invariant, budgeted proposer varies | Strengthen the verifier-correctness claim while narrowing scorer claims to ranking efficiency. |
| Most hypotheses fail | Narrow the paper: the substrate is a verifier-gating contract; multitask scoring is an ablation with heterogeneous-task interference. |

## Out of Scope

- Larger domain-specific feature envelopes.
- Transformer or attention scorers.
- Claims about beating ARC-AGI SOTA.
- Public ARC coverage claims from the fixed multitask scorer.
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

## Cross-reference

The paper's Roadmap appendix (`paper/paper.typ` `tab:roadmap`) is the canonical
list of experiments that defend a paper claim. The E-MT.x / I-MT.x / E-ARC.x
items in this document map 1:1 onto R-MT.x in the paper Roadmap; this doc is
the *operational* plan (ground rules, instrumentation, success criteria),
while the paper Roadmap is the *defensibility* plan (effort estimate, which
claim each experiment defends, submission tiers). Keep this document
authoritative for *how* to run the experiments and the paper Roadmap
authoritative for *which* experiments must land before submission.
