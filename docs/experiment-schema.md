# Generalized Experiment Schema

The project is one verifier-gated action-scoring architecture with first-class
task schemas plus case-study discovery schemas. Prime gaps, grid paths, delayed
claims, program synthesis, and the ARC-AGI sample/loader path must all be described
with the same contract before they are used as evidence. The evidence maturity
differs by task: prime has the complete production learned-gate table, grid has
a budgeted legality gate, delayed-claim still needs a production learned-gate
evaluation, and ARC is a fixed-sample scorer plus a narrow corpus-loader
transform gate rather than a full ARC solver.

There is also a separate shared-envelope multitask probe that includes the same
ARC-AGI-1 sample. That probe is useful for testing whether the core C scorer can
train across heterogeneous action domains, but it is not a substitute for the
standalone schema/result row or for future verifier-gated ARC rule evidence.

## Contract

Each task is specified as:

```text
Task T = {
  state distribution S_T,
  candidate enumerator A_T(s),
  observable features x_T(s, a),
  neural scorer f_theta(x_T),
  exact verifier V_T(s, a),
  symbolic grammar G_T,
  certificate checker C_T(rule),
  rule proposer q_phi(rule | structural(rule)),
  gate executor R_T(rule, s, a),
  trace/replay/RL protocol
}
```

The machine-readable form is emitted as `crlp.experiment_schema.v1`:

```sh
make experiment-schemas
```

The task-schema C API lives in `include/nn_experiment.h`; the implementation is
`src/nn_experiment.c`. The shared symbolic-domain API lives in
`include/nn_symbolic.h`; its implementation is `src/nn_symbolic.c`. The shared
typed program/transform API lives in `include/nn_program.h`; its implementation
is `src/nn_program.c`.

Schema fields describe the intended formal contract. The evidence-status section
below is the controlling source for which production binaries currently execute
which gate and baseline paths.

## Enforced Schema Fields

The schema is not limited to strings. Each task also declares:

- `NNFeatureSchema[]`: one row per scorer input feature, with a typed
  `NNFeatureProvenance`.
- `NNLeakagePolicy`: `forbidden`, `priors_allowed`, `legacy`, or the
  proposer-only `gate_allowed`.
- `NNProposerMode`: `selection` or `discovery`.
- `NNFeatureAudit`: emitted automatically in `crlp.experiment_schema.v1`.

Feature provenance is one of:

```c
NN_FEATURE_STATE_OBSERVATION
NN_FEATURE_STRUCTURAL_PRIOR
NN_FEATURE_VERIFIER_DERIVED
NN_FEATURE_DERIVED_FROM_GATE
```

`NN_LEAKAGE_FORBIDDEN` allows only enabled state/action observations in the
scorer. `NN_LEAKAGE_PRIORS_ALLOWED` allows enabled structural priors but still
forbids verifier-derived and gate-derived scorer inputs. `NN_LEAKAGE_LEGACY`
marks runs that are not directly comparable to clean rows.
`NN_LEAKAGE_GATE_ALLOWED` is for proposer/gate surfaces such as closed-loop
residual-set features; experiment scorer schemas deliberately cannot use it.

The schema writer refuses to emit a task whose enabled scorer features violate
its declared leakage policy. The same reusable audit helper is used by
`NNSymbolicDomain` and `NNClosedLoopRulePolicyConfig` so symbolic proposers
cannot silently add verifier-derived inputs. Comparison rows also carry
`leakage_policy`, so mixed policy tables are visible in `results/latest`.

## Invariants

1. Candidate enumeration is complete before gating.
2. Scorer features are observable state/action features only.
3. Verifier outputs may be labels, rewards, certificates, diagnostics, or
   executed gates, but they are not default scorer inputs.
4. Static proposer inputs are structural rule descriptors only; closed-loop
   proposer inputs may include declared gate-derived residual-set features.
5. Empirical support, utility, and false-rejection counts are targets or
   diagnostics, not proposer features.
6. A rule enters the gate only through the task certificate checker.
7. Every gate-learning claim needs matched-budget neural, random, grammar-order,
   structural-heuristic, and oracle orderings.
8. Every replay claim uses stable action ids.

## Current Instantiations

Verifier-backed first-class task schemas:

| Task | State | Actions | Verifier | Grammar | Certificate | Production gate status |
| --- | --- | --- | --- | --- | --- | --- |
| `prime_gap` | prime `p` | candidate gaps | segmented sieve next-prime verifier | modular rejection predicates | algebraic divisibility proof | learned-rule gate plus matched baselines |
| `grid_path` | grid, start, goal | four moves | BFS shortest-path successor | legality predicates; admissible pruning planned | cannot reject an optimal move | budgeted legality gate; richer baselines pending |
| `delayed_claim` | observable claim features | trust/ignore/request/cautious | deterministic hidden-outcome simulator | threshold predicates; richer claim rules planned | finite-generator safety check in symbolic smoke | production learned-gate table pending |
| `arc_agi` | ARC-AGI-1 task `0520fde7` input grid plus output coordinate | output colors `0..9` | known ARC train/test output grid | copy/mask/intersection/geometric/object transform grammar | train-pair consistency plus unique grammar-consistent test prediction | fixed-sample certified transform gate plus ARC JSON directory loader and matched `NNProgram` proposer rows for narrow geometry/object evidence |

Verifier-backed case-study discovery schemas:

| Task | State | Actions | Verifier | Grammar | Certificate | Evidence status |
| --- | --- | --- | --- | --- | --- | --- |
| `program_synthesis` | list-routine task with input/output examples | candidate library abstraction | run abstraction on every example | monomorphic list/int DSL over map, filter, fold, and selectors | syntactic DSL type check | matched-budget scaffold with random, grammar-order, structural, validation-oracle, static neural, and closed-loop rows |

Leakage status today:

| Task | Leakage policy | Feature audit |
| --- | --- | --- |
| `prime_gap` | `forbidden` | 3 enabled observations; 10 disabled structural-prior residue columns retained only for ablations |
| `grid_path` | `priors_allowed` | 61 observations plus 2 enabled Manhattan structural priors |
| `delayed_claim` | `forbidden` | 10 enabled observations |
| `program_synthesis` | `priors_allowed` | 18 enabled structural-prior DSL abstraction features |
| `arc_agi` | `forbidden` | 49 enabled observations over grid cells, output coordinate, candidate color, and action one-hot |

The first-class `arc_agi` default schema is still one fixed ARC-AGI-1 task. It
emits both a neural cell-scorer comparison row and a certified transform-gate
row. The same binary can also scan ARC-format JSON directories with `--task-dir`
and emit per-task `crlp.arc_transform.summary.v1` rows plus a
`crlp.arc_loader.summary.v1` aggregate. It also maps transform candidates into
typed `NNProgram` rows and emits `crlp.program_proposer.summary.v1` for
grammar-order, structural, random, learned-program, and oracle orderings across
matched program-budget curves. The learned proposer uses the shared
`nn_program_encode_features()` encoder plus ARC task-level shape/color/object
descriptors, and can be pretrained with `--program-train-dir` before evaluating
another split. The loader evidence is intentionally narrow: grammar
version 4 covers two same-height panel masks, same-shape color maps,
foreground/object crops, rotations, flips, transposes, small integer scales,
bbox drawing, small foreground translations, and largest connected-component
crop/mask transforms. The multitask binary uses a separate fixed 96-wide
action-row envelope documented in
[multitask-domain.md](multitask-domain.md).

## Shared Symbolic-Domain Interface

`NNSymbolicDomain` is the generalized formal-system hook:

```text
symbols, axioms, grammar
        ↓
declare rule/scorer feature schemas and leakage policy
        ↓
enumerate NNRule rows
        ↓
check NNCertificate
        ↓
encode structural/residual features
        ↓
train a shared certificate classifier, or use the closed-loop policy where
residual coverage callbacks are implemented
```

`examples/symbolic_domains.c` currently instantiates this interface for
`prime_gap`, `grid_path`, `delayed_claim`, and a compiled ARC-AGI-1 sample. It
emits `crlp.symbolic_domain.summary.v1` per domain, including rule-feature
leakage-policy compliance fields, and
`crlp.symbolic_rule_proposer.summary.v1` for one certificate-status classifier
trained across all four domains. This smoke path checks enumeration,
structural encoding, and certificate labels; it is not the closed-loop residual
policy benchmark.

`examples/program_synthesis.c` instantiates the same interface for the
program-synthesis case study. It uses `NNRuleCoverage` as the task-compression
matrix and the generic closed-loop residual policy for the `closed_loop_neural`
baseline, emitting `crlp.rule_search.summary.v1` rows compatible with the
existing result capture path.

## Shared Program/Transform Interface

`NNProgram` is the task-neutral representation for transform induction,
object-relation operators, library abstractions, and other typed candidates that
are more naturally expressed as programs than as one-step rejection rules.

```text
domain grammar emits typed NNProgram rows
        ↓
optional NNHypothesis templates rank and bind slots from example evidence
        ↓
nn_program_encode_features emits domain-neutral program features
        ↓
canonical id from signature, params, and child ids
        ↓
optional composition under input/output type checks
        ↓
certificate checker accepts or rejects the candidate
        ↓
NNProgramCoverage records support over tasks, examples, cells, obligations, or
other domain events
        ↓
crlp.program_proposer.summary.v1 records matched-budget proposer comparisons
```

The program layer does not replace `NNSymbolicDomain`; it gives symbolic
domains a lower-level object for candidates with typed inputs and outputs plus a
shared structural feature encoder for candidate ranking. `NNHypothesis` is the
thin slot-binding layer above `NNProgram`: domains can bind typed template slots
from observed task facts, compile the result into a canonical `NNProgram`, then
let the domain verifier accept or reject it. ARC appends task-specific
shape/color/object descriptors to those domain-neutral program features before
training its learned proposer. For program synthesis, the same object can name
reusable DSL abstractions. The certificate checker remains domain-owned.

## Rule-Search Summary Rows

`prime_rule_search` and case-study rule-discovery binaries emit
`crlp.rule_search.summary.v1` rows for matched-budget proposer comparisons.
Capacity and truncation-sensitive comparisons are machine-readable through:

| Field | Meaning |
| --- | --- |
| `selection_strategy` | proposer actually used, including `closed_loop_neural` and `unified_scorer` |
| `closed_loop_hidden` | hidden-size label for dedicated closed-loop rows; `null` for unified rows |
| `unified_scorer_hidden` | hidden-size label for unified scorer rows; `null` for closed-loop rows |
| `softmax_group_size` | effective rule group size used by residual-aware training/selection, including any cap |
| `parameters`, `forward_flops` | actual model capacity from `nn_parameter_count` and `nn_forward_flop_count` when a neural proposer constructs a net |
| `unified_action_updates`, `unified_rule_updates` | cross-role sampling counts for unified scorer rows |
| `coverage_cache_hits`, `coverage_cache_misses`, `coverage_cache_writes` | coverage-blob reuse counters for rule-search runs that use `--coverage-cache-dir` |

Unified scorer rows in `prime_rule_search` are Phase 1 cross-role-transfer
rows. They train one 96-wide shared scorer with action rows from the existing
action domains and rule rows from the prime rule trajectory, then deploy that
same scorer for rule selection. The row's `proposer_feature_set` is
`task_role_tagged_action_rule_rows_v1` and its `proposer_target` is
`joint_action_rule_softmax`.

The ARC grammar in the shared symbolic-domain smoke path still contains copy,
mask/intersection, union, object-select-overlap, and color-map transforms. Its
checker accepts a rule only when it matches all train pairs and every
grammar-consistent rule agrees on the test output. That smoke path's
object-select-overlap rule remains an alias for intersection. The standalone
ARC loader is the path that now carries typed `NNProgram` rows and the largest
connected-component primitive. Neither path is a general ARC solver claim.

## Shared Multitask Summary Rows

`examples/multitask_domain.c` emits `crlp.multitask.summary.v1` rows. These rows
are not substitutes for `crlp.experiment_schema.v1`; they summarize one shared
scorer evaluated across one or more domains.

Fields:

| Field | Meaning |
| --- | --- |
| `domain` | evaluated domain, such as `prime_gap`, `grid_path`, `delayed_claim`, or `arc_agi` |
| `task_set` | domains used for training, with `all` meaning every configured domain |
| `eval_task_set` | domains requested for evaluation |
| `model_kind` | current model family, usually `shared` or `single_task` |
| `training_order` | sampling schedule, e.g. `uniform_random`, `round_robin`, `arc_heavy`, `prime_light`, or `domain_only_random` |
| `feature_envelope` | shared row layout, currently `task_tagged_action_rows_v1` or `untagged_action_rows_v1` |
| `train_updates` | number of SGD updates received from this domain |
| `epochs`, `steps_per_epoch`, `hidden`, `parameters`, `forward_flops`, `learning_rate`, `seed` | training and capacity configuration |
| `sample_count`, `candidate_count` | evaluation size and mean action count |
| `top1`, `top5`, `top10`, `nll`, `ece`, `reliability_bins`, `confidence_mean`, `correct_confidence_mean`, `incorrect_confidence_mean`, `mean_rank` | ranking and calibration metrics |
| `decision_utility`, `baseline_steps` | task-specific utility/probe fields when available |

The removed `prime_features` key should not reappear in multitask outputs.
Prime residue predicates are intentionally outside the shared multitask envelope.

## Evidence Status

`prime_gap` has the most complete gate ablation: no gate, fixed hand gate,
learned certified gate, and fixed-budget proposer baselines. It should not be
described as the project goal.

`grid_path` has raw versus budgeted certified legality-gate evaluation under
two visible feature regimes: `no_manhattan_priors` for the clean scorer row and
`with_manhattan_priors` as a priors-allowed ablation. The next version should
add a richer admissible-pruning grammar and the same fixed-budget proposer
baselines used by `prime_gap`.

`delayed_claim` supports supervised training, trace replay, and calibration
evaluation in its production binary. Its threshold grammar carries a
finite-generator certificate in the shared symbolic-domain smoke; the
production learned-gate evaluation table is open work.

## Next Build Targets

Substrate-engineering work that is *not* defending a paper claim and so does
not appear in the paper Roadmap (`paper/paper.typ` `tab:roadmap`). For
experiment-level items see that table; for the multitask operational plan see
`docs/multitask-transfer-research-plan.md`.

1. Move `grid_path` and `delayed_claim` out of monolithic examples into task
   modules with the same shape as `src/prime_gap/` (refactoring; precondition
   for richer per-task targets).
2. Promote the delayed-claim certified gate from symbolic-domain smoke into the
   production delayed-claim evaluation (cross-listed as R-X1 in the paper
   Roadmap).
3. Extend grid-path gates beyond simple legality to admissible pruning rules
   (cross-listed as R-E1.2).
4. Run proposer baselines for every task at matched rule budgets (Axis 2's
   protocol applied to grid-path and delayed-claim; cross-listed under R-E2.1).
5. Run strict versus KL/entropy verifier RL on the certified tasks
   (cross-listed as R-E3.1).
6. Report one cross-task table keyed by the shared schema:
   no gate, spec gate, learned gate, random gate, grammar-order gate,
   structural-heuristic gate, oracle gate, replay successor, strict RL, KL RL
   (substrate-level reporting deliverable that uses the existing schemas).
7. Add metric-provenance rows for random ranking, strongest non-neural
   heuristic, and oracle ranking on the same gated candidate set (substrate-level
   reporting deliverable).
8. Report random-normalized and gate-marginal skill in cross-task tables; keep
   raw top-1 in task-specific sections.
9. Extend ARC beyond the current `NNProgram` ranking experiment into actual
   object-relation, tiling, and transform-induction candidates before treating
   it as production verifier-gated ARC evidence (parallel substrate track;
   cross-listed under R-S.3 for the synthetic-bootstrapping flavor and R-X4
   for the public-corpus parity row).
