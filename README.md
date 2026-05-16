# CRLPLRIMES: Certificate-Gated Neural Proposers for Verifier-Grounded Action Scoring

This repository is a compact C neural-network library and an exact-verifier research harness for **certified neural-symbolic action scoring**. The project is not a prime-gap solver. The code exposes one architecture across first-class verifier-backed task schemas, plus shared-action and shared-symbolic cross-domain tests. The production certified-gate evidence is complete for `prime_gap`, partial for `grid_path`, fixed-sample only for `arc_agi`, and not yet implemented as a production gate table for `delayed_claim`:

| Task | State | Actions | Exact verifier | Gate grammar | Certificate | Status |
| --- | --- | --- | --- | --- | --- | --- |
| `prime_gap` | prime `p` | candidate gaps | segmented sieve next-prime verifier | modular rejection predicates | algebraic divisibility proof | learned-rule gate plus matched baselines |
| `grid_path` | grid, start, goal | four moves | BFS shortest-path successor | legality predicates; admissible pruning planned | cannot reject an optimal move | budgeted legality gate; richer baselines pending |
| `delayed_claim` | observable claim features | trust/ignore/request/cautious | deterministic hidden-outcome simulator | threshold predicates; richer claim rules planned | finite-generator safety check in symbolic smoke | production learned-gate table pending |
| `arc_agi` | ARC-AGI-1 task `0520fde7` input plus output coordinate | colors `0..9` | held-out output-cell color lookup | copy/mask/intersection/geometric transform grammar | train-pair consistency plus unique compiled-test prediction | fixed-sample certified transform gate plus ARC JSON directory loader for narrow geometry evidence |
| `program_synthesis` | list/int routine examples | candidate library abstractions | run abstraction on examples | list/int DSL abstraction grammar | monomorphic DSL type checker | discovery case study |

Every first-class task schema publishes the same machine-readable contract with:

```sh
make experiment-schemas
```

The schema is `crlp.experiment_schema.v1` and defines the state distribution, candidate enumerator, typed feature provenance, leakage policy, scorer, verifier, rule grammar, certificate checker, proposer inputs, proposer mode, baselines, trace protocol, replay protocol, and verifier-RL protocol. The public C contracts live in `include/nn_experiment.h` for task schemas, `include/nn_symbolic.h` for shared symbolic rule domains, and `include/nn_program.h` for typed program/transform candidates with canonical ids and coverage evidence.
The working design note is [docs/experiment-schema.md](docs/experiment-schema.md). The shared-model multitask matrix is documented separately in [docs/multitask-domain.md](docs/multitask-domain.md). The formal symbolic-domain interface is documented in [docs/symbolic-domain-interface.md](docs/symbolic-domain-interface.md).

## Generalized Experiment Schema

Each task is specified as:

```text
Task T = {
  state distribution S_T,
  candidate enumerator A_T(s),
  observable state/action features x_T(s, a),
  neural residual scorer f_theta(x_T),
  exact verifier V_T(s, a),
  symbolic rule grammar G_T,
  certificate checker C_T(rule),
  rule proposer q_phi(rule | structural(rule)),
  gate executor R_T(rule, s, a),
  trace/replay/RL protocol
}
```

The reusable architecture is:

```text
state ──▶ rule proposer ─▶ rule candidates ─▶ certificate verifier ─▶ accepted rules
                                                                        │
state ──▶ candidate enumerator ─────────────────────────────────────────┴─▶ gated candidates
                                                                                │
                                                              learned scorer ──▶ softmax ranking
                                                                                │
                                                              exact verifier ──▶ correct action / cost
                                                                                │
                                                                       JSONL trace ─▶ replay
```

Design invariant:

```text
The model proposes. The verifier decides. The trace remembers.
```

The task-independent rules are:

1. Candidate enumeration is complete before gating; the verifier is not allowed to prune the candidate set silently.
2. Every scorer input feature has typed provenance: `state_observation`, `structural_prior`, `verifier_derived`, or `derived_from_gate`.
3. The scorer's enabled features must satisfy the task's `forbidden`, `priors_allowed`, or `legacy` leakage policy before the schema writer will emit the task.
4. The static proposer receives structural rule descriptors only. The closed-loop proposer may also receive gate-derived residual-set features through an audited `gate_allowed` schema; verifier-derived values remain forbidden.
5. A rule can enter the gate only through the task's certificate checker.
6. Every gate-learning claim must be evaluated against neural, random, grammar-order, structural-heuristic, and oracle proposer orderings at matched budgets before it is treated as evidence.
7. JSONL traces use stable action ids so replay cannot silently change semantics when candidates are reordered.

Current leakage audit:

| Task | Leakage policy | Enabled structural priors |
| --- | --- | ---: |
| `prime_gap` | `forbidden` | 0 |
| `grid_path` | `priors_allowed` | 2 |
| `delayed_claim` | `forbidden` | 0 |
| `arc_agi` | `forbidden` | 0 |
| `program_synthesis` | `priors_allowed` | 18 |

The `arc_agi` task is not a certified ARC solver claim. It is one fixed ARC-AGI-1 task (`0520fde7`) promoted to the same schema/result surface as the other experiments, reduced to balanced output-cell color scoring. The first-class binary now carries a tiny transform grammar, emits a certified transform-gate row when train-consistent grammar rules make a unique test prediction, and can scan ARC-format JSON directories with `--task-dir`. `make symbolic-domains` still exercises the shared symbolic interface over the same task. The `object_select_overlap` rule is an overlap-style placeholder, not a general connected-object selector.

## Current Evidence

The canonical result is strongest on `prime_gap`, presented as one task row rather than the project goal. With residue-sieve features excluded from the scorer input, the raw prime scorer reaches **9.13%** top-1; a fixed hand gate reaches **45.12%**; and a certified learned-rule gate reaches **68.87%** at 100% top-10. On `grid_path`, the no-Manhattan-prior scorer reaches **70.80%** raw accuracy and **84.90%** with the budgeted legality gate; the priors-allowed checkpoint is reported as an explicit ablation at **75.80%** raw and **89.60%** gated. On `delayed_claim`, the load-bearing result is trace replay and calibration: the replay successor reaches **94.80%** accuracy versus **93.50%** for the trained parent. On the fixed `arc_agi` task, the neural scorer reaches **28.56%** top-1 with poor calibration (`ECE=0.5574`), while the certified transform gate reaches **100.00%** top-1 on all 9 output cells via two train-consistent rules (`mask_intersection_color_2`, `object_select_overlap_color_2`). A loader run over the public ARC-AGI corpus certifies 17/400 training tasks and 4/400 evaluation tasks under the expanded grammar. That is real transform evidence, not solved ARC.

At a 16-rule budget with neural proposers configured at matched `64x32` capacity, the closed-loop residual proposer reaches **56.18%** work saved on the conditional gap grammar, above grammar order (`55.57%`) and structural heuristic (`55.57%`), within `0.65pp` of the validation oracle (`56.83%`). The unified scorer with joint action+rule training reaches **73.22%** on the unconditional candidate grammar, matching grammar order, structural heuristic, and the validation oracle exactly, and accepting `16/16` rules versus the dedicated closed-loop proposer's `10/16` at the same capacity (`69.40%`). Random ordering trails on both grammars (`46.30%` unconditional, `3.49%` conditional). The full matched-budget grid lives in `results/unified_scorer/phase_{1,2,3}/summary.md`.

## Protocol Axes and Current Results

### Axis 1 — Certified gates reduce the residual action space

Each task should report the same gate ablation: no gate, hand/spec gate, learned certified gate, and matched-budget baselines. The prime-gap task currently has the complete three-gate table:

| Gate | Top-1 | Top-10 | NLL |
| --- | ---: | ---: | ---: |
| None (raw candidate space) | 9.13% | 78.01% | 2.9994 |
| Hand-wheel (small-prime divisibility) | 45.12% | 99.77% | 1.5902 |
| **Learned rule program** (certificate-checked rules) | **68.87%** | **100.00%** | **1.0056** |

The same protocol is partially implemented on `grid_path`: with Manhattan priors disabled during training, raw accuracy is **70.80%** and budgeted legality-gate accuracy is **84.90%**. The priors-allowed checkpoint is reported as an ablation at **75.80%** raw and **89.60%** gated; its comparison rows list `feature_set=with_manhattan_priors` and `enabled_structural_prior_features=2`. `delayed_claim` carries a threshold grammar and finite-generator certificate in the shared symbolic-domain smoke; the production delayed-claim binary does not yet emit the same gate-ablation table as prime and grid. The `45.12%` prime figure is included only as a full-sieve-feature leakage ablation.

### Axis 2 — Rule search must beat fixed-budget symbolic baselines

`prime_rule_search` runs the neural proposer over unconditional candidate filters `reject candidate n when n > m and n % m == r` for `m=3..255`. The expanded grammar can also express conditional gap filters (`if p % m == a then reject gap % m == b`), and `make prime-rule-search-baselines` reports both families at the same 16-rule budget. `--proposer closed-loop` trains a residual-aware policy on greedy coverage trajectories and proposes rules sequentially against the current accepted set; `--proposer unified` consumes the same residual-aware envelope but trains jointly with action-scoring rows. The certificate verifier only accepts rules with a deterministic soundness proof, so an accepted rule provably cannot reject the true next-prime gap.

Rule-search run from `results/latest/comparison.jsonl`:

```text
proposed:        32,637
verified:         4,692   ← rules that survive verifier screens
accepted:            53   ← all unconditional candidate filters
false rejections:     0
direct exact:      7.20   ← average exact-verifier work without rules
expanded exact:    1.45   ← average exact-verifier work after accepted rules
saved:            79.79%
```

The proposer is small (`crlp.rule_trace.v1` schema, scoring rule candidates with the same shared-scorer abstraction as gap actions). Its input feature set is structural-only: condition kind, moduli, reciprocal scales, residues, zero-residue indicators, gcd ratios, and derived candidate residue. It does not receive `train_rejects`, validation support, utility, or false-rejection counters as features; those counts are labels and verifier diagnostics. The neural target is budgeted train-set marginal utility, not raw support. The discipline of the result is in the verifier, not the model: a rule is accepted only if a deterministic algebraic check certifies soundness, so zero false rejections is a property of the certificate, not of training.

Matched-budget proposer baselines run in `make prime-rule-search-baselines`. At a 16-rule budget over the same split and certificate verifier, with neural proposers configured to match capacity at `64x32`, the proposer rows are:

| Grammar | Rule ordering | Capacity | Exact probes | Work saved |
| --- | --- | --- | ---: | ---: |
| Unconditional candidate | Random, seed 20240511 | — | 3.86 | 46.30% |
| Unconditional candidate | Static neural proposer | 16x8 | 1.98 | 72.49% |
| Unconditional candidate | Closed-loop residual proposer | 64x32 | 2.20 | 69.40% |
| Unconditional candidate | Unified scorer (joint action+rule) | 64x32 | 1.93 | **73.22%** |
| Unconditional candidate | Grammar order | — | 1.93 | 73.22% |
| Unconditional candidate | Structural heuristic | — | 1.93 | 73.22% |
| Unconditional candidate | Validation-marginal oracle | — | 1.93 | 73.22% |
| Conditional gap | Random, seed 20240511 | — | 6.94 | 3.49% |
| Conditional gap | Static neural proposer | 16x8 | 7.04 | 2.15% |
| Conditional gap | Closed-loop residual proposer | 64x32 | 3.15 | **56.18%** |
| Conditional gap | Unified scorer (joint action+rule) | 64x32 | 3.20 | 55.57% |
| Conditional gap | Grammar order | — | 3.20 | 55.57% |
| Conditional gap | Structural heuristic | — | 3.20 | 55.57% |
| Conditional gap | Validation-marginal oracle | — | 3.11 | 56.83% |

**What this evidence does and does not show.** The closed-loop residual proposer at `64x32` reaches `56.18%` work saved on conditional gap rules, above grammar order (`55.57%`) and within `0.65pp` of the validation oracle (`56.83%`). The joint-training unified scorer at `64x32` reaches `73.22%` on unconditional candidate rules — matching grammar order, structural heuristic, and the validation oracle exactly — versus `69.40%` for the dedicated closed-loop proposer at the same capacity. The unconditional `69.40%` cell reflects subset-pruning capping the closed-loop accepted set at `10/16` rules; the joint-training unified scorer accepts `16/16`. The static neural proposer remains far behind both grammars (`72.49%` unconditional, `2.15%` conditional) because it scores rules without conditioning on the accepted set. The next neurosymbolic experiment is to port the same matched-budget loop to `grid_path` and `delayed_claim` and check whether the joint-training advantage transfers across domains.

### Axis 3 — Verifier RL must preserve calibration across tasks

Strict verifier policy gradient on sparse exact rewards is a known-bad recipe, and we observe the failure mode cleanly. Same supervised checkpoint, two RL branches, default split:

| Policy | Temp | Top-1 | Top-10 | NLL | ECE | Top-1 confidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Supervised | 1.00 | 9.13% | 78.01% | 2.9170 | 0.0704 | 0.1617 |
| Strict verifier RL | 1.00 | 9.13% | 78.01% | 3.0611 | 0.1271 | 0.2184 |
| Strict verifier RL, temp-scaled | 1.50 | 9.13% | 78.01% | 2.9063 | 0.0613 | 0.1526 |
| KL/entropy-regularized verifier RL | 1.00 | 9.13% | 78.01% | 2.9056 | 0.0604 | 0.1517 |
| KL/entropy-regularized RL, temp-scaled | 1.00 | 9.13% | 78.01% | 2.9056 | 0.0604 | 0.1517 |

Three observations:

- **Ranking is preserved across all variants.** Strict RL does not improve top-1 — it is mostly logit inflation.
- **Strict RL degrades calibration.** Unscaled ECE rises from 0.0704 to 0.1271, and post-hoc temperature scaling needs `T=1.5` to undo it. The RL branch has effectively learned to be overconfident.
- **KL-to-prior + entropy fixes it at training time.** Unscaled ECE drops to 0.0604; the selected temperature stays at `T=1.0`, meaning the policy is already close to calibrated without rescaling. This is the standard RLHF objective applied to exact-verifier rewards, and it works here for the same reason it works there: the reference policy supplies the bulk of the probability mass and the gradient is bounded.

The finding as stated holds for prime-gap. Whether the same logit-inflation failure mode appears under verifier RL on `grid_path` and `delayed_claim`, and whether the same KL/entropy regularizer fixes it, is on the open-work list — the pipeline supports running it, but the comparison is not in the canonical results yet.

## Task Instantiation: Prime Gap

Prime gaps are used as the load-bearing benchmark because the domain has properties most exact-verifier sandboxes lack:

- **A fast deterministic verifier at all scales** (segmented sieves, wheel factorization, block proofs).
- **A compact totally-ordered candidate space** (gaps `2..256` plus an overflow action).
- **An exhausted classical literature** — there is no plausible claim that a small neural net beats a sieve, which sharpens the question to "what does the neural component contribute under perfect feedback?"
- **Rich algebraic structure** the rule grammar can exploit (residue classes, divisibility, certificates).

We are **not** claiming improved prime generation. Exact next-prime verification under any of these pipelines remains the classical algorithm's job, because a correct proof requires ruling out earlier candidates and a wrong neural proposal creates extra proof obligations. The framing is:

```text
Primes are the substrate. The contribution is the protocol: certificate-gated
hybrid scoring, learned symbolic gates, and calibration-preserving verifier RL.
```

The default training split is:

```text
train       p < 800,000
validation  800,000 <= p < 1,000,000
test        1,000,000 <= p <= 3,000,000
reported metrics use 30,000 held-out test states
```

Detailed baselines:

| Method | Temp | Top-1 | Top-10 | NLL | ECE | Top-1 confidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Empirical gap baseline | 1.00 | 15.46% | 78.71% | 2.8585 | 0.0196 | 0.1743 |
| Residue-conditioned frequency | 1.00 | 40.11% | 93.04% | 2.8583 | 0.2769 | 0.1242 |
| Linear minimal scorer | 1.00 | 9.13% | 78.01% | 2.9119 | 0.0289 | 0.1202 |
| Linear minimal, temp-scaled | 0.85 | 9.13% | 78.01% | 2.9005 | 0.0486 | 0.1399 |
| Supervised minimal scorer | 1.00 | 9.13% | 78.01% | 2.9994 | 0.1076 | 0.1989 |
| Supervised minimal, temp-scaled | 1.20 | 9.13% | 78.01% | 2.8991 | 0.0459 | 0.1372 |
| Full-sieve-feature ablation (leakage) | 1.00 | 45.12% | 99.77% | 1.5778 | 0.0743 | 0.4882 |
| Supervised minimal + hand-wheel | 1.00 | 45.12% | 99.77% | 1.5902 | 0.0834 | 0.5075 |
| **Supervised minimal + learned-rule gate** | **1.00** | **68.87%** | **100.00%** | **1.0056** | 0.1402 | 0.6276 |
| Learned-rule survivor replay | 1.00 | 68.87% | 100.00% | 1.0196 | 0.1323 | 0.6736 |
| Strict verifier RL | 1.00 | 9.13% | 78.01% | 3.0611 | 0.1271 | 0.2184 |
| KL/entropy-regularized verifier RL | 1.00 | 9.13% | 78.01% | 2.9056 | 0.0604 | 0.1517 |

Among raw-action-space scorers without residue leakage, the hidden network and linear scorer both collapse to gap-frequency behavior. The `45.12%` full-sieve row is useful only as an upper-bound ablation showing how much signal residue flags hand to the model. The dominating effect is not depth of the scorer; it is the verified symbolic gate.

## Task Instantiations: Delayed Claim and Grid Path

`delayed_claim` is a delayed-label synthetic task with six state features and four actions (`trust`, `ignore`, `request_more_evidence`, `act_cautiously`). A deterministic hidden rule resolves the outcome and defines the best action. The task validates trace replay with parent-hash lineage:

| Model | Accuracy | NLL | ECE | Decision utility | Steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| Trained parent | 93.50% | 0.1580 | 0.0363 | 1.0432 | 1.065 |
| Trace-replay successor | 94.80% | 0.1188 | 0.0166 | 1.0704 | 1.052 |

`grid_path` is a shortest-path task with an exact BFS verifier owning correctness. Candidate actions are all four moves, including out-of-bounds and blocked moves. The neural model sees start/goal coordinates, candidate displacement, the obstacle map, and an action one-hot; it does not receive explicit `out_of_bounds` or `blocked_next` legality flags. BFS supplies labels and verification cost, not a pruned candidate set.

A small structural rule proposer runs on this task. Its grammar is the certified legality gate: reject edge moves that leave the grid, and reject moves into blocked cells. The proposer scores five certified rules from structural features, selection uses train+validation only, and a fixed budget accepts the top three before test evaluation. This is intentionally not a pathfinding solver; it is a cross-domain sanity check that certificate-gated pruning can be represented and evaluated outside prime residues.

| Model / mode | Accuracy | NLL | ECE | Candidates | Decision utility | Steps | Nodes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No-prior parent, raw | 70.80% | 0.7608 | 0.0574 | 4.00 | 0.6350 | 1.365 | 22.18 |
| No-prior parent, budgeted legality gate | 84.90% | 0.4295 | 0.0726 | 3.13 | 0.8113 | 1.164 | 21.98 |
| Priors-allowed parent, raw | 75.80% | 0.7154 | 0.0712 | 4.00 | 0.6975 | 1.327 | 21.46 |
| Priors-allowed parent, prior-disabled eval ablation | 57.40% | 1.0420 | 0.0423 | 4.00 | 0.4675 | 1.674 | 26.98 |
| Priors-allowed parent, budgeted legality gate | 89.60% | 0.3732 | 0.1033 | 3.13 | 0.8700 | 1.116 | 21.25 |
| Priors-allowed parent, gated prior-disabled eval ablation | 68.90% | 0.7403 | 0.0470 | 3.13 | 0.6112 | 1.417 | 26.72 |
| Trace-replay successor, raw | 74.00% | 0.7274 | 0.0602 | 4.00 | 0.6750 | 1.335 | 21.75 |
| Trace-replay successor, budgeted legality gate | 87.60% | 0.3851 | 0.0928 | 3.13 | 0.8450 | 1.137 | 21.55 |

The grid legality proposer accepts 3/5 certified rules under budget with 0 held-out false rejections, rejecting 0.87 impossible moves per held-out state on average. The no-prior row is the clean scorer baseline; the priors-allowed row is retained because it measures exactly how much the two Manhattan structural priors contribute. The useful signal is that a verifier-certified symbolic gate transfers outside prime residues, not that legality pruning solves shortest-path ranking.

## Limitations and Next Work

- **Proposer learning is domain-specific.** Closed-loop and unified proposers match the validation-oracle ceiling on conditional gap rules and match grammar order exactly on unconditional candidate rules. Both reach the available ceiling at `64x32` capacity. Capacity ladders at `128x64` and `256x128` produce no further gain on conditional and small regressions on unconditional. Where the deterministic baselines already saturate the achievable signal, additional capacity does not help.
- **Cross-task verifier-RL comparison.** Axis 3 (strict-vs-regularized RL on the prime-gap task) requires matching runs on `grid_path` and `delayed_claim` before it can be stated as a general property of sparse-exact-reward policy gradient. The pipeline supports it; the comparison is not in the canonical results yet.
- **Richer non-prime gate grammars.** `grid_path` carries a budgeted certified legality gate that is intentionally shallow. The next cross-domain test should learn admissible dead-end or distance-bound pruning rules that reduce BFS verifier work without becoming a hand-coded solver.
- **ARC is still narrow.** The first-class `arc_agi` experiment now has a certified transform gate for one ARC-AGI-1 task and an ARC JSON directory loader, but the current grammar is still a hand-coded set of copy/mask, recolor, crop, rotation/flip/transpose, scale, bbox, and small-translation transforms. It does not induce general transformations across ARC tasks.
- **Statistical rigor.** The default `make prime-gap-experiments` config currently runs 1 seed; the harness supports more (`PRIME_GAP_EXPERIMENT_SEEDS=...`). The minimal-vs-leakage, strict-vs-regularized, and hand-wheel-vs-learned-rule deltas should be reported with confidence intervals across >=5 seeds.
- **Reliability diagrams.** The comparison schema stores `reliability_bins`; the README currently reports only ECE. Calibration claims would be stronger with a small reliability plot.
- **Learned-rule calibration.** The learned rule program improves ranking sharply, but its ECE is worse than the raw minimal and hand-wheel rows in the default run. That calibration tradeoff deserves a focused reliability-diagram pass.

## Reproducibility Substrate

The hybrid architecture is the contribution; the substrate is the discipline that makes it auditable. Four pieces matter:

- **Stable action IDs and trace contract.** Trace rows carry `schema: "crlp.trace.v1"`, `trace_version: 1`, and `action_id_contract: "stable_unique_action_ids"`. Replay resolves `target_index` to the trace row's `action_id`, then maps that stable id back into the regenerated candidate row. Reordering action ids cannot silently change replay semantics; the prime replay regression test exercises that.
- **Checkpoint identity.** Checkpoints carry a `parent_model_hash`, a `training_trace_root`, a `feature_encoder_version`, and a `feature_set` in metadata; the model hash is SHA-256 over the exact checkpoint bytes. Successor models declare their parent. Comparison reports (`crlp.compare.v1`) carry `model_hash`, `parent_hash`, `feature_set`, and a machine-readable `reliability_bins` array.
- **Pure C, no third-party deps.** The neural library is dense layers, sigmoid/ReLU/tanh/linear, MSE and softmax cross-entropy, SGD, batched 4-by-4 tiled forward/backward, cloned networks, action scoring, pairwise ranking loss, cached policy-gradient updates with optional KL/entropy regularization, deterministic checkpoint save/load, SHA-256 hashing, JSONL trace writing. The core is ~3k lines; the prime-gap task is split into focused modules under `src/prime_gap/`.
- **Shared symbolic substrate.** `include/nn_symbolic.h` defines `NNRule`, `NNCertificate`, `NNSymbolicDomain`, residual coverage bitsets, feature-provenance audits for rule/scorer schemas, and a generic closed-loop residual rule policy. `NNClosedLoopRulePolicyConfig` carries the residual feature schema used by the shared policy. Prime uses that shared closed-loop policy; `examples/symbolic_domains.c` runs prime, grid, delayed-claim, and ARC through one formal rule-row and certificate interface. The smoke path certifies rows and trains a certificate-status classifier; it does not exercise production gate execution callbacks for every domain.

End-to-end quick path:

```sh
make
make test
make experiment-schemas
make prime-gap
make prime-gap-trace
make prime-gap-replay
make prime-gap-eval
make prime-rule-search
make delayed-claim
make grid-path
make arc-agi
make arc-agi-loader ARC_AGI_TASK_DIR=/path/to/ARC-AGI/data/training
make multitask-domain
make symbolic-domains
python3 scripts/multitask_ablation_matrix.py
make results
```

`prime-gap-trace` writes `build/float/prime_v1.nnckpt` and `build/float/traces/prime_gap.jsonl`. `prime-gap-replay` trains `build/float/prime_v2.nnckpt` from that trace. `prime-gap-eval` compares both checkpoints on the same frozen benchmark. `prime-rule-search` writes `build/float/prime_rules_v2.crlprule` and `build/float/traces/prime_rule_search.jsonl`. `arc-agi` trains the fixed `0520fde7` cell-color scorer, writes `build/float/arc_agi.nnckpt`, and emits both neural scorer and certified transform-gate rows. `arc-agi-loader` scans ARC-format JSON tasks from `ARC_AGI_TASK_DIR` and emits per-task `crlp.arc_transform.summary.v1` rows plus a `crlp.arc_loader.summary.v1` aggregate. `make results` writes `results/latest/summary.md`, `results/latest/comparison.jsonl`, and `results/latest/commands.txt`; bulky logs stay under ignored `results/**/logs/`. `make symbolic-domains` is currently outside the canonical `make results` capture script, so run it separately when refreshing symbolic-domain smoke evidence.

Cross-task tuning uses a separate noisy workspace so canonical result snapshots stay stable:

```sh
make neurosymbolic-tune
```

This runs `scripts/neurosymbolic_tuning.py` over the prime rule-search grammars, grid legality rule search, delayed-claim replay, and grid replay under one profile. It writes raw task rows to `build/float/neurosymbolic_tuning/raw.jsonl`, a cross-task score table to `normalized.csv`, and a compact `summary.md`. Grid replay tuning defaults to `with-priors`, because that is the feature set currently supported by the trace/replay path. Use `scripts/neurosymbolic_tuning.py --profile full` for larger budgets and sample counts.

The shared-model smoke test trains one action scorer across all configured domains in shuffled order:

```sh
make multitask-domain
```

`examples/multitask_domain.c` wraps prime-gap, grid-path, delayed-claim, and one ARC-AGI-1 sample into one fixed-width action-row envelope. The ARC sample is training task `0520fde7`, represented as a balanced cell-color action task: given a train/test input grid and an output-cell coordinate, choose the output color from actions `0..9`. ARC rows expose the flattened input grid, the output coordinate, and the candidate action, but not the direct overlap/intersection rule. This is a useful integration test for grid-to-grid symbolic tasks, not a full ARC solver claim.

Use `--tasks prime,grid,claim,arc|all` for combination ablations and `--eval-tasks all` to evaluate held-out domains after training on a subset. A matched per-domain update comparison can scale `--steps-per-epoch` by the number of active domains, for example `--tasks prime --steps-per-epoch 3000`, `--tasks prime,grid --steps-per-epoch 6000`, and `--tasks all --steps-per-epoch 12000`. The remaining feature ablation is explicit: `--no-task-tags` removes the task one-hot fields. The shared multitask envelope excludes prime residue flags so the prime row receives no hand-coded modular predicates.

The shared formal-system smoke test trains one certificate-status rule classifier across rule rows from all symbolic domains:

```sh
make symbolic-domains
```

`examples/symbolic_domains.c` exposes the same `NNSymbolicDomain` enumeration, feature-schema, leakage-policy, and certificate callbacks for prime modular filters, grid legality filters, delayed-claim threshold rules, and an ARC grammar (`copy`, `mask`, `intersection`, `object_select_overlap`, `color_map`). The current `object_select_overlap` operator is an alias for intersection in this compiled sample. The binary validates each domain with `nn_symbolic_domain_is_valid()`, prints rule-feature audit fields in each `crlp.symbolic_domain.summary.v1` row, and emits one `crlp.symbolic_rule_proposer.summary.v1` row for the shared certificate classifier; it is not yet the closed-loop residual proposer benchmark.

For the repeatable multi-seed matrix:

```sh
python3 scripts/multitask_ablation_matrix.py
```

The default matrix enumerates all non-empty task combinations across three seeds, evaluating all domains every time. It writes `build/float/multitask_ablation_matrix/raw.jsonl`, `trained_summary.csv`, `heldout_summary.csv`, and `summary.md`. In a 4-domain, 3-seed run without prime residue features, the all-domain tagged model averages 11.77% prime top-1, 90.37% grid top-1, 95.50% claim top-1, and 26.13% ARC cell-color top-1. Removing task tags reaches 14.07%, 90.00%, 95.97%, and 30.90%. The shared scorer trains across all domains; the prime task is weak without symbolic modular predicates; the ARC row is intentionally unsaturated.

## C Neural Network Library

Public headers split conceptually:

```text
include/nn_core.h             allocation, inspection, forward pass, softmax, action scoring, checkpoints, traces
include/nn_train.h            sample training, batch training, action-scorer training, cached policy-gradient with KL/entropy reg
include/nn_experiment.h       shared experiment-schema contract for verifier-backed action tasks
include/nn_symbolic.h         shared rule, certificate, symbolic-domain, and closed-loop residual policy API
include/nn_program.h          shared typed program/transform candidates, canonical ids, coverage, and proposer summary rows
include/nn_tasks_prime_gap.h  command entry points for the prime-gap task module
include/nn.h                  compatibility umbrella for existing users
```

The prime-gap implementation is split across `src/prime_gap/`: data generation, feature encoding, symbolic-wheel discovery, expanded rule search, block-verifier accounting, evaluation, training, run orchestration, and replay/compare entry points each have their own module. Shared private declarations live in `src/prime_gap/internal.h`. `src/nn_tasks_prime_gap.c` remains as a test-only module index so private helpers can be exercised without exporting them from the public task header. The files in `examples/prime_gap.c`, `examples/prime_replay.c`, `examples/prime_compare.c`, and `examples/prime_rule_search.c` are thin command wrappers.

## Build

```sh
make
```

The default scalar type is `float`. Build with double precision when needed:

```sh
make REAL=double
```

## Run the XOR Example

```sh
make run
```

Expected output should converge toward:

```text
0 xor 0 -> close to 0
0 xor 1 -> close to 1
1 xor 0 -> close to 1
1 xor 1 -> close to 0
```

The example also prints basic performance metrics:

```text
parameters
training time
training throughput
final loss
accuracy
inference latency
inference throughput
```

## Run the Standalone Action-Scoring Example

```sh
make action-score
```

This example trains a tiny shared scorer over three candidate rows and prints deterministic ranked actions with scores and probabilities. It is independent of the prime-gap experiment.

## Run Benchmarks

```sh
make benchmark
```

The benchmark uses larger synthetic MLP shapes and reports:

```text
parameter count
estimated dense forward FLOPs
forward latency
forward throughput
estimated forward GFLOP/s
mini-batch training throughput
sample-wise SGD baseline
batched training speedup
estimated dense training GFLOP/s
```

The sample-SGD and batch-SGD lines measure throughput for different update schedules, so their losses are not expected to match after the same epoch count.

The current batch path uses internal 4-by-4 tiles for forward propagation, hidden-delta backpropagation, and gradient accumulation, with scalar fallback code for remainders and small dimensions.

Gradient buffers are allocated lazily the first time mini-batch training runs, so inference-only use and sample-wise training avoid that extra model memory.

## Running the Prime-Gap Experiments

The headline pipeline is `make prime-gap` followed by `make prime-rule-search`.

```sh
make prime-gap
```

Builds a sieve-backed prime-gap dataset and trains a shared candidate scorer:

```text
input  = [log(p), normalized_gap, overflow_flag]
score  = f(input)
policy = softmax(scores across candidate gaps 2..256 plus an overflow action)
```

The default `minimal` feature set intentionally omits precomputed residue divisibility flags. Use `--feature-set full` or `--full-features` only for the leakage/upper-bound ablation where the scorer sees the small-prime sieve flags directly. After a learned rule program has already pruned composites, those sieve flags are mostly dead on the survivor set, so they are diagnostic rather than a clean source of ranking signal. The run evaluates empirical, residue-conditioned, and linear minimal baselines, runs supervised feature ablations (log-only, gap-only, residue-only, full-sieve leakage), fits a scalar softmax temperature on a held-out validation slice, and compares strict and KL/entropy-regularized verifier-RL branches cloned from the same supervised checkpoint.

```sh
make prime-rule-search
```

Runs the prime-gap rule-grammar instantiation of Axis 2. The neural proposer scores unconditional candidate filters of the form `reject candidate n when n > m and n % m == r` over `m=3..255`; `make prime-rule-search-baselines` runs both that grammar and the conditional gap grammar with neural, closed-loop, unified, random, grammar-order, structural-heuristic, and validation-oracle orderings at a fixed 16-rule budget. `make prime-rule-search-conditional` emits a trace/checkpoint for the broader conditional gap grammar. The static neural proposer's feature set is structural-only and excludes support counts, utility, and false-rejection counters. It trains on a budgeted marginal-utility target computed from verifier labels; `--proposer closed-loop` trains on residual coverage trajectories and scores each next rule against the accepted set and current residual state; `--proposer unified` consumes the same residual-aware envelope but trains jointly with action-scoring rows. `--max-marginal-states` controls the sample cap for those targets and defaults to 4096. `--coverage-cache-dir` reuses the precomputed coverage bitset across baseline orderings on the same split. `--closed-loop-hidden` and `--unified-scorer-hidden` select the neural proposer's MLP layer sizes (e.g. `64x32`). The verifier accepts only rules with a deterministic algebraic certificate. The JSONL trace uses schema `crlp.rule_trace.v1` and records grammar version, selection strategy, proposer feature set, proposer target, rule text, model score, verifier result, deterministic-certificate flag, accepted flag, support counts, marginal utility, and false-rejection counts.

CLI output of `make prime-gap` is split into sections:

```text
capability:    top-k accuracy, NLL, true-action probability, mean rank, MRR, gap error
calibration:   top-1 confidence, correct/incorrect confidence, ECE, high-confidence accuracy
prime buckets: capability metrics over four increasing held-out prime ranges
verifier:      exact classical work needed to verify or correct proposals
policy:        supervised, strict RL, regularized RL, and temperature-scaled rows
```

`proposal_steps` measures how quickly the true action appears in the model's proposal order. `wheel_exact` measures how many exact sieve primality probes remain after small-prime divisibility filters eliminate obvious composites. `assisted_wheel` tries the model's top-10 proposals before falling back to wheel scan. `block_cost` estimates segmented interval-proof work, and `assisted_block` tries model proposals before falling back to the block scan.

Larger sweeps:

```sh
make prime-gap-sweep
make prime-gap-regularization-sweep
make prime-gap-experiments
```

The experiment harness runs supervised epoch sweeps and KL/entropy regularization sweeps and writes parsed CSV plus a Markdown summary under `build/float/prime_gap_experiments/`. Curated result snapshots that should survive `make clean` live under `results/`. Override the default one-seed quick run with:

```sh
make prime-gap-experiments PRIME_GAP_EXPERIMENT_SEEDS=20240511,20240512,20240513
```

Optional arguments:

```sh
./build/float/prime_gap [train_limit] [test_limit] [supervised_epochs] [rl_epochs] [max_train_states] [max_test_states] [regularized_reference_weight] [regularized_entropy_weight] [rng_seed] [--feature-set minimal|full|log-only|gap-only|residue-only]
```

## Run Tests

```sh
make test
```

Run sanitizer coverage with:

```sh
make sanitize
```

Run the CI-style local check with:

```sh
make check
```

This runs the unit tests, sanitizer tests, double-precision tests, the standalone action-scoring example, delayed-claim trace/replay/eval, symbolic-domain smoke, the expanded prime-rule-search smoke path, and a small prime-gap trace/replay/eval smoke path.

The tests compare scalar and batched forward results, verify `batch_size=1` training against sample-wise training, finite-difference check batched gradients through the tiled kernels, and check the softmax cross-entropy and shared softmax-scorer gradients.

## API

```c
const size_t layers[] = {2, 4, 1};
NeuralNet *net = nn_create(layers, 3, 42);

size_t parameters = nn_parameter_count(net);
size_t forward_flops = nn_forward_flop_count(net);

nn_real input[] = {1.0, 0.0};
nn_real target[] = {1.0};

nn_train_sample(net, input, target, 0.7);

nn_real output[1];
nn_forward(net, input, output);

nn_free(net);
```

Use `nn_create_config` for non-default activations or softmax cross-entropy:

```c
const size_t layers[] = {12, 32, 16, 1};
NNConfig config = {
    .layer_sizes = layers,
    .layer_count = 4,
    .seed = 42,
    .hidden_activation = NN_ACTIVATION_RELU,
    .output_activation = NN_ACTIVATION_LINEAR,
    .loss = NN_LOSS_MSE,
};
NeuralNet *scorer = nn_create_config(&config);
NeuralNet *copy = nn_clone(scorer);
```

With a linear output layer, `nn_forward` returns logits; call `nn_softmax` when probabilities are needed.

For a shared scalar scorer over candidate actions, train one state at a time with softmax cross-entropy across the candidate rows:

```c
nn_real scores[16];
nn_real probabilities[16];
size_t ranked[16];

nn_score_action_rows(scorer, candidate_inputs, candidate_count, scores, probabilities);
nn_rank_action_scores(scores, candidate_count, ranked);

nn_real loss = nn_train_action_scorer(scorer,
                                      candidate_inputs,
                                      candidate_count,
                                      true_candidate_index,
                                      0.005f);
```

`candidate_inputs` is row-major: `candidate_count * nn_input_size(scorer)` values. The scorer must have one scalar output. Probabilities sum to approximately `1.0`, and ranking is deterministic for equal scores.

Verifier-guided or policy-gradient experiments can reuse a cached candidate forward pass and add KL/entropy regularization against a reference policy:

```c
nn_forward_batch(scorer, candidate_inputs, scores, candidate_count);
nn_real objective = nn_train_softmax_scorer_policy_regularized_cached(
    scorer,
    candidate_count,
    sampled_index,
    advantage,
    reference_probabilities,
    1.20f,
    0.12f,
    0.001f);
```

Training a dataset uses contiguous row-major buffers:

```c
nn_real final_loss = nn_train(net, inputs, targets, sample_count, 0.7, 1000, 1);
```

Mini-batch training uses reusable batch activations, batch deltas, and gradient buffers inside the network:

```c
nn_real final_loss = nn_train_batched(net, inputs, targets, sample_count, 32, 0.05, 10, 1);
```

`inputs` must contain `sample_count * nn_input_size(net)` values. `targets` must contain `sample_count * nn_output_size(net)` values.

## Checkpoints

Checkpoints use a small canonical binary format with:

```text
magic
checkpoint_version
scalar_type
layer_count
layer_sizes
hidden_activation
output_activation
loss
weights
biases
metadata block length
metadata block
```

Metadata stores `parent_model_hash`, `training_trace_root`, `feature_encoder_version`, `feature_set`, `benchmark_report_hash`, and `notes`.

```c
NNCheckpointMeta meta = {0};
meta.checkpoint_version = 1;
meta.feature_encoder_version = 1;

nn_save_checkpoint(net, "model.nnckpt", &meta);

char hash[65];
nn_hash_checkpoint_hex("model.nnckpt", hash);

NNCheckpointMeta loaded_meta;
NeuralNet *loaded = nn_load_checkpoint("model.nnckpt", &loaded_meta);
```

The hash is SHA-256 over the exact checkpoint bytes, so changing weights or metadata changes model identity.

## Trace Schema

Trace rows are JSONL records with schema `crlp.trace.v1`:

```json
{"schema":"crlp.trace.v1","trace_version":1,"action_id_contract":"stable_unique_action_ids","task":"prime_gap","state_hash":"0000000000000000","model_hash":"unknown","feature_encoder_version":1,"candidate_count":0,"action_ids":[],"scores":[],"probabilities":[],"selected_index":0,"target_index":0,"loss":0.0,"verifier_result":"correct","verifier_cost":{},"seed":0,"feature_set":"minimal"}
```

Prime-gap traces add `p`, `true_gap`, `proposal_steps`, `wheel_exact`, `assisted_wheel`, `block_cost`, `assisted_block`, and `log_scale`. Rule-search traces use schema `crlp.rule_trace.v1`. Delayed-claim traces use the deterministic per-row seed for replay.

Use `nn_trace_parse_jsonl` to validate rows before replay. It checks the schema string, trace version, action-id contract, required scalar fields, unique action ids, score/probability array lengths, selected/target index bounds, and probability mass. Replay code resolves `target_index` to the row's stable `action_id`, then maps that action id back into the regenerated candidate rows before training. `nn_trace_json_get_u64`, `nn_trace_json_get_real`, and `nn_trace_json_get_string` are small helpers for task-specific extra fields.

Comparison reports are JSONL records with schema `crlp.compare.v1`, `report_version`, `model_hash`, `parent_hash`, `feature_set`, `feature_encoder_version`, benchmark seed, sample count, capability metrics, calibration metrics, machine-readable `reliability_bins`, and any task-specific verifier or utility fields.

Deterministic fixtures live in `tests/fixtures/` and cover a tiny action scorer, expected scores, expected probabilities, expected ranking, and an example trace row. The tests exercise those fixtures in both `float` and `double` builds.

## Delayed-Claim Task

`delayed_claim` is a small delayed-label synthetic task with six state features:

```text
source_reliability
evidence_strength
contradiction_count
claim_age
local_history_score
noise_level
```

The model scores four actions: `trust`, `ignore`, `request_more_evidence`, and `act_cautiously`. A deterministic hidden rule resolves the later outcome and defines the best action. The task reports accuracy, NLL, calibration error, and decision utility.

```sh
make delayed-claim
make delayed-claim-trace
make delayed-claim-replay
make delayed-claim-eval
```

## Grid-Path Task

`grid_path` is a second exact-verifier task. A deterministic generator creates a small obstructed grid, start, and goal. Candidate actions are all four moves. The neural model ranks moves from observable map and move features, while an exact BFS verifier owns correctness by identifying the shortest-path successor. `grid_path --rule-search` runs the cross-task budgeted legality-gate experiment: a structural proposer ranks five certified rules that reject out-of-bounds and blocked-cell moves, then accepts the top three under a fixed rule budget.

The task reports accuracy, NLL, calibration error, decision utility, proposal steps, fixed-order baseline steps, exact verifier nodes, baseline verifier nodes, and candidate counts before/after the legality gate. Trace replay stores stable move action IDs, then maps them back into the regenerated four-row candidate set before training a successor model.

```sh
make grid-path
make grid-path-rule-search
make grid-path-trace
make grid-path-replay
make grid-path-eval
```

## Shared Multitask / ARC-AGI Sample

`arc_agi` is now also a standalone first-class schema/result target:

```sh
make arc-agi
```

The standalone task is ARC-AGI-1 task `0520fde7`, reduced to output-cell color prediction over actions `0..9`. The neural model sees the flattened input grid, target output coordinate, candidate color, and action one-hot. It does not receive a direct overlap/intersection feature. The same binary now also evaluates a tiny transform grammar and reports the certified transform gate separately from the neural scorer.

The loader mode runs the same transform grammar over ARC-format JSON files:

```sh
make arc-agi-loader ARC_AGI_TASK_DIR=/path/to/ARC-AGI/data/training ARC_AGI_MAX_TASKS=400
./build/float/arc_agi --task-dir /path/to/ARC-AGI/data/evaluation --max-tasks 400 --loader-summary-only
```

On the public ARC-AGI corpus snapshot used for the current evidence pass, grammar version 3 loaded all 400 training tasks with no parse errors and certified 17 tasks; on the evaluation split it certified 4 tasks. The supported families include split-panel masks, same-shape color maps, foreground crops, flips/rotations/transposes, scaling, bbox drawing, and small translations.

`multitask_domain` remains the generalization smoke test for the core scorer API. It trains one scalar action scorer over a fixed 96-feature row envelope while shuffling samples from `prime_gap`, `grid_path`, `delayed_claim`, and `arc_agi`.

```sh
make multitask-domain
python3 scripts/multitask_ablation_matrix.py
```

The multitask ARC row uses the same fixed task and adds the optional task tag from the shared envelope. The matrix script enumerates every non-empty task combination, evaluates all domains each time, and writes JSONL/CSV/Markdown summaries under `build/float/multitask_ablation_matrix/`.

Current all-domain top-1 means over three seeds:

| Variant | Prime | Grid | Claim | ARC |
| --- | ---: | ---: | ---: | ---: |
| Task tags | 11.77% | 90.37% | 95.50% | 26.13% |
| No task tags | 14.07% | 90.00% | 95.97% | 30.90% |

Prime residue features were removed from this shared envelope, so the weak prime row is expected. The useful signal is that one C scorer can train across all four action-row domains without task-specific code in the network itself.
