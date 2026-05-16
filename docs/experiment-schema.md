# Generalized Experiment Schema

The project is one verifier-gated action-scoring architecture with three
certified task instantiations. Prime gaps, grid paths, and delayed claims must
all be described with the same contract before they are used as certified
evidence.

There is also a separate shared-envelope multitask probe that includes one
ARC-AGI-1 sample. That probe is useful for testing whether the core C scorer can
train across heterogeneous action domains, but it is not yet part of the
certified rule/verifier evidence table.

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
`include/nn_symbolic.h`; its implementation is `src/nn_symbolic.c`.

## Enforced Schema Fields

The schema is no longer only strings. Each task also declares:

- `NNFeatureSchema[]`: one row per scorer input feature, with a typed
  `NNFeatureProvenance`.
- `NNLeakagePolicy`: `forbidden`, `priors_allowed`, or `legacy`.
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

The schema writer refuses to emit a task whose enabled features violate its
declared leakage policy. Comparison rows also carry `leakage_policy`, so mixed
policy tables are visible in `results/latest`.

## Invariants

1. Candidate enumeration is complete before gating.
2. Scorer features are observable state/action features only.
3. Verifier outputs may be labels, rewards, certificates, diagnostics, or
   executed gates, but they are not default scorer inputs.
4. Proposer inputs are structural rule descriptors only.
5. Empirical support, utility, and false-rejection counts are targets or
   diagnostics, not proposer features.
6. A rule enters the gate only through the task certificate checker.
7. Every gate-learning claim needs matched-budget neural, random, grammar-order,
   structural-heuristic, and oracle orderings.
8. Every replay claim uses stable action ids.

## Current Instantiations

Certified verifier-backed tasks:

| Task | State | Actions | Verifier | Grammar | Certificate |
| --- | --- | --- | --- | --- | --- |
| `prime_gap` | prime `p` | candidate gaps | segmented sieve next-prime verifier | modular rejection predicates | algebraic divisibility proof |
| `grid_path` | grid, start, goal | four moves | BFS shortest-path successor | legality and admissible-pruning predicates | cannot reject an optimal move |
| `delayed_claim` | observable claim features | trust/ignore/request/cautious | deterministic hidden-outcome simulator | threshold/dominance/evidence predicates | finite-generator safety check |

Shared-envelope integration task:

| Task | State | Actions | Label source | Status |
| --- | --- | --- | --- | --- |
| `arc_agi` | ARC-AGI-1 task `0520fde7` input grid plus output coordinate | output colors `0..9` | known ARC train/test output grid | supervised cell-color scorer; separate symbolic-domain smoke has a tiny certified transform grammar |

Leakage status today:

| Task | Leakage policy | Feature audit |
| --- | --- | --- |
| `prime_gap` | `forbidden` | 3 enabled observations; 10 disabled structural-prior residue columns retained only for ablations |
| `grid_path` | `priors_allowed` | 61 observations plus 2 enabled Manhattan structural priors |
| `delayed_claim` | `forbidden` | 10 enabled observations |

`arc_agi` is omitted from the leakage table until it has a task schema and a
dataset-level certificate checker. In the current multitask binary it uses the
fixed 96-wide action-row envelope documented in [multitask-domain.md](multitask-domain.md).

## Shared Symbolic-Domain Interface

`NNSymbolicDomain` is the generalized formal-system hook:

```text
symbols, axioms, grammar
        ↓
enumerate NNRule rows
        ↓
check NNCertificate
        ↓
encode structural/residual features
        ↓
train/select with the generic closed-loop rule policy
```

`examples/symbolic_domains.c` currently instantiates this interface for
`prime_gap`, `grid_path`, `delayed_claim`, and a compiled ARC-AGI-1 sample. It
emits `crlp.symbolic_domain.summary.v1` per domain and
`crlp.symbolic_rule_proposer.summary.v1` for one rule proposer trained across
all four domains.

The ARC grammar in that smoke path contains copy, mask/intersection, union,
object-select, and color-map transforms. Its checker accepts a rule only when it
matches all train pairs and every grammar-consistent rule agrees on the test
output. This is still a fixed-sample formal-system smoke, not a general ARC
solver claim.

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
| `model_kind` | current model family, usually `shared_action_scorer` |
| `training_order` | sampling schedule, currently `shuffled` |
| `feature_envelope` | shared row layout, currently `task_tagged_action_rows_v1` or `untagged_action_rows_v1` |
| `train_updates` | number of SGD updates received from this domain |
| `epochs`, `steps_per_epoch`, `learning_rate`, `seed` | training configuration |
| `sample_count`, `candidate_count` | evaluation size and mean action count |
| `top1`, `top5`, `top10`, `nll`, `mean_rank` | ranking metrics |
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

`delayed_claim` currently supports supervised training, trace replay, and
calibration evaluation in its production binary. Its threshold grammar now has a
finite-generator certificate in the shared symbolic-domain smoke, but it still
needs a production learned-gate evaluation table.

## Next Build Targets

1. Move `grid_path` and `delayed_claim` out of monolithic examples into task
   modules with the same shape as `src/prime_gap/`.
2. Promote the delayed-claim certified gate from symbolic-domain smoke into the
   production delayed-claim evaluation.
3. Extend grid-path gates beyond simple legality to admissible pruning rules.
4. Run proposer baselines for every task at matched rule budgets.
5. Run strict versus KL/entropy verifier RL on the certified tasks.
6. Report one cross-task table keyed by the shared schema:
   no gate, spec gate, learned gate, random gate, grammar-order gate,
   structural-heuristic gate, oracle gate, replay successor, strict RL, KL RL.
7. Add metric-provenance rows for random ranking, strongest non-neural
   heuristic, and oracle ranking on the same gated candidate set.
8. Report random-normalized and gate-marginal skill in cross-task tables; keep
   raw top-1 in task-specific sections.
9. Promote ARC from one fixed cell-color sample to a loader over many ARC tasks,
   then add certifiable symbolic ARC transformations before treating ARC as
   verifier-gated evidence.
