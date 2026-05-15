# Generalized Experiment Schema

The project is one verifier-gated action-scoring architecture with three task
instantiations. Prime gaps, grid paths, and delayed claims must all be described
with the same contract before they are used as evidence.

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

The C API lives in `include/nn_experiment.h`; the implementation is
`src/nn_experiment.c`.

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

| Task | State | Actions | Verifier | Grammar | Certificate |
| --- | --- | --- | --- | --- | --- |
| `prime_gap` | prime `p` | candidate gaps | segmented sieve next-prime verifier | modular rejection predicates | algebraic divisibility proof |
| `grid_path` | grid, start, goal | four moves | BFS shortest-path successor | legality and admissible-pruning predicates | cannot reject an optimal move |
| `delayed_claim` | observable claim features | trust/ignore/request/cautious | deterministic hidden-outcome simulator | threshold/dominance/evidence predicates | finite-generator safety check |

## Evidence Status

`prime_gap` has the most complete gate ablation: no gate, fixed hand gate,
learned certified gate, and fixed-budget proposer baselines. It should not be
described as the project goal.

`grid_path` has raw versus budgeted certified legality-gate evaluation. The next
version should add a richer admissible-pruning grammar and the same fixed-budget
proposer baselines used by `prime_gap`.

`delayed_claim` currently supports supervised training, trace replay, and
calibration evaluation. Its threshold/dominance grammar is specified in the
schema but not yet implemented as a learned certified gate.

## Next Build Targets

1. Move `grid_path` and `delayed_claim` out of monolithic examples into task
   modules with the same shape as `src/prime_gap/`.
2. Implement the delayed-claim certified gate grammar.
3. Extend grid-path gates beyond simple legality to admissible pruning rules.
4. Run proposer baselines for every task at matched rule budgets.
5. Run strict versus KL/entropy verifier RL on all three tasks.
6. Report one cross-task table keyed by the shared schema:
   no gate, spec gate, learned gate, random gate, grammar-order gate,
   structural-heuristic gate, oracle gate, replay successor, strict RL, KL RL.
