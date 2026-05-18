# CRLPLRIMES: Certificate-Gated Neural Proposers for Verifier-Grounded Action Scoring

This repository is a compact C neural-network library and an exact-verifier research harness for **certified neural-symbolic action scoring**. The code exposes one architecture across first-class verifier-backed task schemas, plus shared-action and shared-symbolic cross-domain tests. The strongest current anchor is **certified modular gate search**: learn and certify modular rejection rules that shrink an exact verifier's candidate set without changing the correct answer. The `prime_gap` task is the first complete instantiation of that family, not the project boundary. The production certified-gate evidence is complete for `prime_gap`, partial for `grid_path`, threshold+margin wired for `delayed_claim`, and fixed-sample only for `arc_agi`:

| Task | State | Actions | Exact verifier | Gate grammar | Certificate | Status |
| --- | --- | --- | --- | --- | --- | --- |
| `prime_gap` | prime `p` | candidate gaps | segmented sieve next-prime verifier | modular rejection predicates | algebraic divisibility proof | certified modular gate search instantiation with learned-rule gate plus matched baselines |
| `grid_path` | grid, start, goal | four moves | BFS shortest-path successor | legality, local-dead-end, component-trap, clear-path admissible-bound, and distance-dominance predicates | cannot reject an optimal move | budgeted certified gate with family/budget proposer sweep |
| `delayed_claim` | observable claim features | trust/ignore/request/cautious | deterministic hidden-outcome simulator | threshold predicates plus direct utility-margin and indirect risk-margin predicates | finite-generator safety check over train+validation support | production certified direct gate resolves trust/request on clean bands; stress rows expose noise sensitivity and indirect proxy limits |
| `arc_agi` | ARC-AGI-1 task `0520fde7` input plus output coordinate | colors `0..9` | held-out output-cell color lookup | copy/mask/intersection/geometric/object transform grammar | train-pair consistency plus unique compiled-test prediction | fixed-sample certified transform gate plus ARC loader, typed program rows, and learned program-proposer evidence |
| `signal_flight` | Signal ship kinematics, hull, path, and clearance observations | WASD key combinations | deterministic replay of WASD hypotheses in Signal's authoritative simulator | WASD control hypotheses now; richer flight predicates planned | replay-certified progress, rollout progress, hull loss, damage, and death outcomes | counterfactual utility scorer, certified rollout bridge, and protocol-checked websocket pilot; teacher imitation remains an ablation |
| `program_synthesis` | list/int routine examples | candidate library abstractions | run abstraction on examples | list/int DSL abstraction grammar | monomorphic DSL type checker | discovery case study |

Every first-class task schema publishes the same machine-readable contract with:

```sh
make experiment-schemas
```

The schema is `crlp.experiment_schema.v1` and defines the state distribution, candidate enumerator, typed feature provenance, leakage policy, scorer, verifier, rule grammar, certificate checker, proposer inputs, proposer mode, baselines, trace protocol, replay protocol, and verifier-RL protocol. The public C contracts live in `include/nn_experiment.h` for task schemas, `include/nn_symbolic.h` for shared symbolic rule domains, and `include/nn_program.h` for typed program/transform candidates with canonical ids and coverage evidence.
The working design note is [docs/experiment-schema.md](docs/experiment-schema.md). The shared-model multitask matrix is documented separately in [docs/multitask-domain.md](docs/multitask-domain.md). The formal symbolic-domain interface is documented in [docs/symbolic-domain-interface.md](docs/symbolic-domain-interface.md). The code-to-PDF claim-generation plan is tracked in [docs/self-authoring-paper-roadmap.md](docs/self-authoring-paper-roadmap.md). Editorial-agent review policy is tracked in [docs/editorial-agent-prs.md](docs/editorial-agent-prs.md).

The paper-support compiler is the framework boundary between result rows and
PDF claims:

```sh
make paper-claims    # derive claim states and generated paper artifacts
make paper-support   # fail if a required claim is stale/missing/unsupported
make paper-pdf       # runs paper-support before compiling paper/paper.pdf
make paper-ci        # check editorial PR boundaries and compile the paper
```

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
7. JSONL traces use stable action ids, declared ranking policy, and score-margin metadata so replay cannot silently change semantics when candidates are reordered or scores are float-sensitive ties.

Current leakage audit:

| Task | Leakage policy | Enabled structural priors |
| --- | --- | ---: |
| `prime_gap` | `forbidden` | 0 |
| `grid_path` | `priors_allowed` | 2 |
| `delayed_claim` | `forbidden` | 0 |
| `arc_agi` | `forbidden` | 0 |
| `signal_flight` | `forbidden` | 0 |
| `program_synthesis` | `priors_allowed` | 18 |

### Certified Modular Gate Search

`prime_gap` is best read as an instance of certified modular gate search:

```text
enumerate candidates -> propose modular residue rules -> certify soundness
                     -> gate impossible candidates -> rank the residual
```

The current instance rejects candidate next-prime gaps when residue arithmetic proves `p + g` composite. The class is broader than primes: it covers any verifier-backed arithmetic task where a candidate can be safely rejected by residue, period, divisibility, or congruence evidence and where each accepted rule can be replayed as a deterministic certificate. The repo now carries a smoke-level `modular_gate_search` extension for three natural next members: prime constellations, squarefree/k-free divisibility filters, and Fibonacci/Lucas recurrence gates via Pisano-period residues. That is the path from an `A` prime-gap benchmark to an `A+` family claim.

The `arc_agi` task is one fixed ARC-AGI-1 task (`0520fde7`) promoted to the same schema/result surface as the other experiments, reduced to balanced output-cell color scoring. The first-class binary now carries a tiny transform grammar, emits a certified transform-gate row when train-consistent grammar rules make a unique test prediction, maps transform candidates into typed `NNProgram` rows, trains a learned program proposer over domain-neutral program features plus ARC task descriptors, can append learned scale-composition candidates, emits budget curves, and can scan ARC-format JSON directories with `--task-dir`. `make symbolic-domains` still exercises the older shared symbolic interface over the same task; its `object_select_overlap` rule remains an overlap-style placeholder, while the `layer_*` family makes the hidden layer/depth prior explicit in the symbolic verifier grammar. The standalone ARC loader also has connected-component largest-object primitives, learned selected-object priors, connector-only rendering, generic object-relation induction, color-line compression for separator/panel tasks, lattice motif copy/extend for separator-grid tasks, and an optional `--induction` path that binds `NNHypothesis` template slots from object-scene deltas or lattice evidence, scores top-K slot proposals with learned relation priors when available, then compiles them into verifier-checked `NNProgram` rows.

The `signal_flight` task is a simulator-control experiment. `make signal-flight` generates train/test CSV traces from `../signal/build/flight_trace`, trains a tiny WASD action scorer, and writes `crlp.signal_flight.summary.v1` plus a checkpoint under `build/$(REAL)/signal_flight_experiment`. The default data path is counterfactual: sampled states enumerate all nine WASD hypotheses, replay each through Signal's authoritative simulator, and train the neural scorer to rank certified utility rather than imitate the deterministic flight teacher. The summary compares neural ranking against a no-safety goal-heading WASD baseline on utility, regret, damage rate, and death rate. `make signal-flight-rollout-eval` turns the checkpoint into closed-loop `crlp.signal_flight_rollout.summary.v1` rows over certified control-target hypotheses, and `signal_live_brain.mjs` now protocol-checks the Signal websocket headers before active play. `make signal-flight-pain` biases trace generation toward asteroid-contact states with hull-loss rows.

The `signal_mining` task is now the grammar layer above flight. `make signal-mining` enumerates bounded policy hypotheses from target selector, route constraint, speed/contact profile, and interaction primitive; each hypothesis compiles to normal Signal player inputs, is replayed in the deterministic simulator, and gets ranked by replay-certified mining/trade utility. The checkpoint feature set is `signal-mining-grammar-v1`, and trace rows expose the grammar fields next to damage, tow, smelt, repair-kit, and supply-chain outcomes. `make signal-live-brain SIGNAL_LIVE_BRAIN_SERVER=ws://127.0.0.1:9091/ws` runs the external websocket MVP pilot and appends `crlp.signal_live_brain.v1` observation/action rows for continuous training.

## Current Evidence

The canonical result is strongest on the `prime_gap` certified modular gate search instantiation, presented as one task row rather than the project goal. With residue-sieve features excluded from the scorer input, the raw prime scorer reaches **9.13%** top-1; a fixed hand gate reaches **45.12%**; and a certified learned-rule gate reaches **68.87%** at 100% top-10. On `grid_path`, the cheap local/topology gate is separated from cheap admissible bounds and the exact-distance oracle rule: the no-Manhattan-prior scorer reaches **70.80%** raw accuracy, **84.90%** with cheap topology, **87.80%** with topology plus clear-path bounds, and **92.00%** with the mixed topology+bound+exact gate. The priors-allowed checkpoint is reported as an explicit ablation at **75.80%** raw, **89.70%** cheap-local gated, **90.10%** cheap-bound gated, and **95.30%** mixed gated. The v5 grid grammar keeps the certified component-trap predicate, adds certified clear-path admissible-bound predicates, and now selects rules by train/validation residual marginal utility instead of standalone support. Cheap-local marginal held-out rejects top out at 1268, the default cheap-bound budget reaches 1995, the full non-exact cheap-bound ceiling is 2316, and the exact-distance mixed ceiling remains 2545. On `delayed_claim`, trace replay still improves the parent (94.80% accuracy, NLL 0.1188, ECE 0.0166 versus 93.50%, 0.1580, 0.0363), and the production threshold+margin gate now emits matched proposer and stress rows. The direct utility-margin family certifies 4 rules with zero clean held-out false rejections, shrinking the action set from 4 to 1 candidate and taking clean gated accuracy/NLL/ECE to **100.00% / 0.0000 / 0.0000**; it remains exact on a shifted seed band, but observation noise exposes 23/1000 false rejections and a hard uncertainty+noise band exposes 104/1000. The indirect risk-margin family is weaker but safer under the same observation-noise row: 94.60% accuracy, 1.35 candidates, and zero false rejections. On the fixed `arc_agi` task, the neural scorer reaches **28.56%** top-1 with poor calibration (`ECE=0.5574`), while the certified transform gate reaches **100.00%** top-1 on all 9 output cells; grammar v15 makes the same fixed-task certified prediction via `mask_intersection_color_2`, `object_select_overlap_color_2`, and `layer_overlap_color_2`, while the `layer_*` family supplies sibling layer-relation hypotheses for budgeted ranking. The public ARC-AGI corpus loader now records grammar v15 at 24/400 training tasks and 7/400 evaluation tasks, covering 1187/56672 training cells and 1057/98515 evaluation cells. The v15 induction path also synthesizes the `92e50de0` lattice verifier as `hypothesis_tile_motif_repeat_bg_0_period_2` from two slots (`lattice.background=0`, `repeat.period=2`) and ranks it at budget 1 under `--induction`. Training the program proposer on the public training split improves evaluation budget efficiency: at budget 32 learned covers all 1057 evaluation cells versus 948 for structural and 39 for grammar order; by budget 64, learned still covers the full 1057-cell grammar ceiling while structural reaches 1048 and grammar order reaches 87. The new hypothesis smoke under `tests/fixtures/arc_hypothesis` emits a certified `select_then_translate` hypothesis evidence row with 25/25 covered events and ranks it inside budget 8 for structural and learned program orderings. That is the first self-hypothesizing substrate pass.

At a 16-rule budget with neural proposers configured at matched `64x32` capacity, the closed-loop residual proposer reaches **56.18%** work saved on the conditional gap grammar, above grammar order (`55.57%`) and structural heuristic (`55.57%`), within `0.65pp` of the validation oracle (`56.83%`). The unified scorer with joint action+rule training reaches **73.22%** on the unconditional candidate grammar, matching grammar order, structural heuristic, and the validation oracle exactly, and accepting `16/16` rules versus the dedicated closed-loop proposer's `10/16` at the same capacity (`69.40%`). Random ordering trails on both grammars (`46.30%` unconditional, `3.49%` conditional). The full matched-budget grid lives in `results/unified_scorer/phase_{1,2,3}/summary.md`.

Two substrate-loop experiments exercise the next layer above static evidence:

- `make rejected-candidate-learning` learns structural priors from rejected certificate rows across three surfaces: prime modular filters, additional modular-gate families, and delayed-claim threshold gates. On held-out high-modulus prime rules, the trained residual improves the default tight budget from 6 accepted rules / 216 utility to 7 accepted rules / 245 utility with zero validation false rejections. The modular split guard passes, delayed-claim now uses a safety-first residual to remove unsafe tight-budget rows, and the new modular leave-one-family-out artifact shows mixed transfer rather than a uniform cross-family win. The proposal is [docs/experiment-rejected-candidate-learning.md](docs/experiment-rejected-candidate-learning.md).
- `make hypothesis-library-promotion` groups certified `crlp.hypothesis.evidence.v1` rows into reusable library-entry candidates and exports a `crlp.hypothesis_library.prior.v1` slot-prior TSV that `arc_agi --hypothesis-library-prior` feeds back into proposal ranking. `make hypothesis-library-ablation` promotes from `move_to_anchor,move_to_anchor_x`, then evaluates on held-out fixture tasks under a tight 1/2/4/8 budget grid, top-k 1 score-only hypothesis selection, slot-only, linear, tiny-MLP, full-MLP, and cold-start variants so structural/coverage heuristics and pre-ranking train filters cannot hide early saturation. `make hypothesis-library-split-sweep` repeats that stricter check across every two-task `move_to_anchor` promotion split and reports aggregate, decoy-pressure, and per-task transfer tables. On the full fixture suite, `move_to_anchor` promotes from four certified task instances, 150 covered events, two variable slots, and zero false rejections. The proposal is [docs/experiment-certified-hypothesis-library.md](docs/experiment-certified-hypothesis-library.md).

## Protocol Axes and Current Results

### Axis 1 — Certified gates reduce the residual action space

Each task should report the same gate ablation: no gate, hand/spec gate, learned certified gate, and matched-budget baselines. The certified modular gate search instance currently has the complete three-gate table:

| Gate | Top-1 | Top-10 | NLL |
| --- | ---: | ---: | ---: |
| None (raw candidate space) | 9.13% | 78.01% | 2.9994 |
| Hand-wheel (small-prime divisibility) | 45.12% | 99.77% | 1.5902 |
| **Learned rule program** (certificate-checked rules) | **68.87%** | **100.00%** | **1.0056** |

The same protocol is partially implemented on `grid_path`: with Manhattan priors disabled during training, raw accuracy is **70.80%**, cheap-local gate accuracy is **84.90%**, cheap-bound gate accuracy is **87.80%**, and mixed topology+bound+exact gate accuracy is **92.00%**. The priors-allowed checkpoint is reported as an ablation at **75.80%** raw, **89.70%** cheap-local gated, **90.10%** cheap-bound gated, and **95.30%** mixed gated; its comparison rows list `feature_set=with_manhattan_priors` and `enabled_structural_prior_features=2`. `delayed_claim` now emits the same kind of production threshold+margin-gate/proposer table plus shifted/noisy stress rows. The direct grammar resolves `trust`/`request_more_evidence` on clean and shifted generators, while the indirect grammar is a safer but less decisive proxy under observation noise. The `45.12%` prime figure is included only as a full-sieve-feature leakage ablation.

### Axis 2 — Rule search must beat fixed-budget symbolic baselines

The current `prime_rule_search` binary is the first certified modular gate search implementation. It runs the neural proposer over unconditional candidate filters `reject candidate n when n > m and n % m == r` for `m=3..255`. The expanded grammar can also express conditional gap filters (`if p % m == a then reject gap % m == b`), and `make prime-rule-search-baselines` reports both families at the same 16-rule budget. `--proposer closed-loop` trains a residual-aware policy on greedy coverage trajectories and proposes rules sequentially against the current accepted set; `--proposer unified` consumes the same residual-aware envelope but trains jointly with action-scoring rows. The certificate verifier only accepts rules with a deterministic soundness proof, so an accepted rule provably cannot reject the true next-prime gap.

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

**What this evidence does and does not show.** The closed-loop residual proposer at `64x32` reaches `56.18%` work saved on conditional gap rules, above grammar order (`55.57%`) and within `0.65pp` of the validation oracle (`56.83%`). The joint-training unified scorer at `64x32` reaches `73.22%` on unconditional candidate rules — matching grammar order, structural heuristic, and the validation oracle exactly — versus `69.40%` for the dedicated closed-loop proposer at the same capacity. The unconditional `69.40%` cell reflects subset-pruning capping the closed-loop accepted set at `10/16` rules; the joint-training unified scorer accepts `16/16`. The static neural proposer remains far behind both grammars (`72.49%` unconditional, `2.15%` conditional) because it scores rules without conditioning on the accepted set. The grid-path family sweep now ports the matched-budget idea outside arithmetic; the remaining neurosymbolic experiment is to make the richer residual-aware proposer win consistently across multi-seed budget grids and then carry that discipline into `delayed_claim`.

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

The finding as stated holds for the prime-gap certified modular gate instance. Whether the same logit-inflation failure mode appears under verifier RL on `grid_path` and `delayed_claim`, and whether the same KL/entropy regularizer fixes it, is on the open-work list — the pipeline supports running it, but the comparison is not in the canonical results yet.

## Task Family: Certified Modular Gate Search

The current member of this family is the prime-gap instantiation. Prime gaps are used as the load-bearing benchmark because the domain has properties most exact-verifier sandboxes lack:

- **A fast deterministic verifier at all scales** (segmented sieves, wheel factorization, block proofs).
- **A compact totally-ordered candidate space** (gaps `2..256` plus an overflow action).
- **An exhausted classical literature** — there is no plausible claim that a small neural net beats a sieve, which sharpens the question to "what does the neural component contribute under perfect feedback?"
- **Rich algebraic structure** the rule grammar can exploit (residue classes, divisibility, certificates).

We are **not** claiming improved prime generation. Exact next-prime verification under any of these pipelines remains the classical algorithm's job, because a correct proof requires ruling out earlier candidates and a wrong neural proposal creates extra proof obligations. The framing is:

```text
Primes are the substrate. The contribution is the protocol: certificate-gated
hybrid scoring, learned symbolic gates, and calibration-preserving verifier RL.
```

Future members should preserve the same proof shape while changing the arithmetic substrate. Prime constellations test depth inside the prime-residue family. Squarefree, k-free, and polynomial-value filters test divisibility gates outside next-prime search. Fibonacci/Lucas recurrence gates test periodic residue structure through Pisano periods. All of them should keep the same contract: explicit candidate enumeration, structural-only rule proposer inputs, deterministic certificate checks, replayable rule traces, and matched-budget baselines.

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

The production threshold+margin gate now evaluates two claim grammars. The direct family enumerates 538 observable rules: 528 feature/action threshold predicates plus trust/request utility-margin predicates that compare the two residual actions without exposing verifier utility to the neural scorer. The indirect family enumerates 546 rules by replacing direct utility comparisons with a coarser risk-margin proxy. Every accepted rule is certificate-checked against the finite train+validation generator, then evaluated under matched neural, closed-loop residual, random, grammar-order, structural, and validation-oracle proposer orderings. At an 8-rule budget the best direct rows accept 4 certified rules, reject no clean held-out optimal actions, and reduce the clean held-out action set from 4 to 1 candidate:

| Rule ordering | Accepted rules | Held-out false rejections | Held-out candidates | Gate work saved |
| --- | ---: | ---: | ---: | ---: |
| Neural structural | 4 | 0 | 1.00 | 75.00% |
| Closed-loop residual | 4 | 0 | 1.00 | 75.00% |
| Random | 8 | 0 | 2.08 | 47.90% |
| Grammar order | 8 | 0 | 2.60 | 34.90% |
| Structural heuristic | 4 | 0 | 1.00 | 75.00% |
| Validation oracle | 4 | 0 | 1.00 | 75.00% |

The best direct gate rows take the replay successor from 94.80% accuracy, NLL 0.1188, and ECE 0.0166 to 100.00% accuracy, NLL 0.0000, and ECE 0.0000 on the clean held-out generator. The reason is explicit in the accepted grammar: thresholds prune dominated `ignore`/`act_cautiously` actions, while the utility-margin predicates safely arbitrate `trust` versus `request_more_evidence`.

The stress rows keep the claim honest:

| Gate family / row | Band | False rejections | Accuracy | Candidates |
| --- | --- | ---: | ---: | ---: |
| Direct margin, neural/closed-loop | shifted seed | 0 | 100.00% | 1.00 |
| Direct margin, neural/closed-loop | observation noise 0.03 | 23 | 98.50% | 1.07 |
| Direct margin, neural/closed-loop | uncertainty band + noise 0.03 | 104 | 93.70% | 1.31 |
| Indirect risk margin, structural/oracle | clean held-out | 0 | 94.90% | 1.35 |
| Indirect risk margin, structural/oracle | observation noise 0.03 | 0 | 94.60% | 1.35 |
| Indirect risk margin, structural/oracle | uncertainty band + noise 0.03 | 0 | 69.40% | 1.99 |

That changes the interpretation: direct margin rules prove the production certificate/proposer/gate loop can solve the clean deterministic task, while the indirect proxy and noisy bands expose the robustness frontier.

`grid_path` is a shortest-path task with an exact BFS verifier owning correctness. Candidate actions are all four moves, including out-of-bounds and blocked moves. The neural model sees start/goal coordinates, candidate displacement, the obstacle map, and an action one-hot; it does not receive explicit `out_of_bounds` or `blocked_next` legality flags. BFS supplies labels and verification cost, not a pruned candidate set.

A small structural rule proposer runs on this task. Its grammar now covers certified legality, local dead-end, component-trap, clear-path admissible-bound, and shortest-distance dominance rules: reject edge moves that leave the grid, reject moves into blocked cells, reject non-goal local cul-de-sacs, reject moves into components that can reach the goal only by backtracking through the start, reject moves whose Manhattan lower bound is dominated by another successor's clear straight or two-segment Manhattan path to the goal, and reject legal successors dominated by another successor's BFS goal distance. The proposer scores ten certified rules from structural features; default selection uses train+validation residual marginal utility, so overlapping support cannot consume budget unless it adds non-overlapping rejects. The role of this task is cross-domain evidence that certificate-gated pruning can be represented and evaluated outside prime residues.

| Model / mode | Accuracy | NLL | ECE | Candidates | Decision utility | Steps | Nodes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No-prior parent, raw | 70.80% | 0.7608 | 0.0574 | 4.00 | 0.6350 | 1.365 | 22.18 |
| No-prior parent, cheap-local gate | 84.90% | 0.4188 | 0.0647 | 2.98 | 0.8113 | 1.159 | 21.97 |
| No-prior parent, bound-only gate | 74.10% | 0.6549 | 0.0362 | 2.88 | 0.6763 | 1.325 | 21.57 |
| No-prior parent, cheap-bound gate | 87.80% | 0.3139 | 0.0260 | 2.00 | 0.8475 | 1.130 | 21.37 |
| No-prior parent, exact-distance gate | 76.90% | 0.5905 | 0.0390 | 2.65 | 0.7113 | 1.279 | 20.14 |
| No-prior parent, mixed topology+bound+exact gate | 92.00% | 0.2205 | 0.0396 | 1.77 | 0.9000 | 1.084 | 19.95 |
| Priors-allowed parent, raw | 75.80% | 0.7154 | 0.0712 | 4.00 | 0.6975 | 1.327 | 21.46 |
| Priors-allowed parent, prior-disabled eval ablation | 57.40% | 1.0420 | 0.0423 | 4.00 | 0.4675 | 1.674 | 26.98 |
| Priors-allowed parent, cheap-local gate | 89.70% | 0.3626 | 0.0966 | 2.98 | 0.8712 | 1.113 | 21.24 |
| Priors-allowed parent, bound-only gate | 76.00% | 0.6504 | 0.0519 | 2.88 | 0.7000 | 1.316 | 21.23 |
| Priors-allowed parent, cheap-bound gate | 90.10% | 0.2991 | 0.0514 | 2.00 | 0.8762 | 1.111 | 21.02 |
| Priors-allowed parent, exact-distance gate | 79.60% | 0.5759 | 0.0587 | 2.65 | 0.7450 | 1.256 | 19.46 |
| Priors-allowed parent, mixed topology+bound+exact gate | 95.30% | 0.1926 | 0.0697 | 1.77 | 0.9413 | 1.051 | 19.26 |
| Priors-allowed parent, cheap-bound prior-disabled eval ablation | 87.00% | 0.3454 | 0.0727 | 2.00 | 0.8375 | 1.141 | 21.29 |
| Priors-allowed parent, mixed prior-disabled eval ablation | 93.50% | 0.2344 | 0.0966 | 1.77 | 0.9187 | 1.068 | 19.12 |
| Trace-replay successor, raw | 74.00% | 0.7274 | 0.0602 | 4.00 | 0.6750 | 1.335 | 21.75 |
| Trace-replay successor, cheap-bound gate | 88.70% | 0.2986 | 0.0671 | 2.00 | 0.8588 | 1.124 | 21.30 |
| Trace-replay successor, mixed topology+bound+exact gate | 94.50% | 0.1904 | 0.0605 | 1.77 | 0.9313 | 1.060 | 19.44 |

The grid proposer now reports rule families explicitly. The cheap-local/topology family accepts 4/7 certified non-distance rules with 0 held-out false rejections and rejects 1.01 moves per held-out state at the default budget, with no bound or exact-distance calls. The bound-only family has 2 certified clear-path rules; it has 1794 summed held-out support but only 1118 marginal held-out rejects because the straight-line rule is a subset of the two-segment Manhattan-path rule. The default marginal-aware selector therefore accepts only the two-segment bound in the bound-only budget. The cheap-bound family combines topology with admissible bounds: at the default four-rule budget it reaches 1995 marginal held-out rejects and 87.80% no-prior gated accuracy, and at full non-exact budget it reaches 2316 marginal held-out rejects. The mixed topology+bound+exact family reaches 2225 marginal held-out rejects at the default four-rule budget and 2545 at full budget. The full family/budget sweep runs budgets 1..7 for cheap-local/topology, 1..2 for bound-only, 1..9 for cheap-bound, 1 for exact-distance, and 1..10 for mixed, with five neural seeds and five random seeds per budget. The useful signal is now split cleanly: cheap certified pruning transfers outside prime residues, cheap admissible bounds close most of the distance-oracle gap, exact-distance pruning is reported as an oracle-style upper bound, and raw summed support remains visible as a diagnostic rather than driving the default accepted set.

## Limitations and Next Work

- **Proposer learning is domain-specific.** Closed-loop and unified proposers match the validation-oracle ceiling on conditional gap rules and match grammar order exactly on unconditional candidate rules. Both reach the available ceiling at `64x32` capacity. Capacity ladders at `128x64` and `256x128` produce no further gain on conditional and small regressions on unconditional. Where the deterministic baselines already saturate the achievable signal, additional capacity does not help.
- **Fully trained tiny nets.** The next capacity question is not "make it bigger" by default; it is whether the current tiny nets are undertrained, saturated, or missing cumulative protocol state. The roadmap now adds a train-to-plateau audit over smaller trunks, longer budgets, restarts, warm starts, rejected-candidate reason codes, and certified-library features before treating adapter heads or curriculum as the main explanation.
- **Multitask architecture boundary.** The flat 96-feature shared action scorer remains a negative-control baseline for ARC geometry. `multitask_domain --task-heads` now adds a shared trunk with per-domain scalar heads as the R-MT.4 architecture surface, but it is not a paper claim until the full roadmap comparison is captured.
- **Cross-task verifier-RL comparison.** Axis 3 (strict-vs-regularized RL on the prime-gap certified modular gate instance) requires matching runs on `grid_path` and `delayed_claim` before it can be stated as a general property of sparse-exact-reward policy gradient. The pipeline supports it; the comparison is not in the canonical results yet.
- **Delayed-claim external validity.** The delayed-claim production loop now resolves the clean and shifted deterministic generator with certified direct threshold+margin rules, and the indirect risk-margin grammar avoids false rejections under the 0.03 observation-noise stress row. The hard uncertainty+noise band is still weak, especially for the indirect proxy, so the next delayed-claim improvement should add noise-aware certificates or abstaining margin bands before treating the result as broad evidence.
- **Richer non-prime gate grammars.** `grid_path` now carries admissible local-dead-end, component-trap, clear-path bound, and distance-dominance predicates plus a family/budget proposer sweep with marginal, non-overlapping reject counts. Its default selector is marginal-aware, so the remaining grid-path work is repeated seeds and an accuracy/utility-aware selector audit rather than raw support pruning. `delayed_claim` now has direct utility-margin and indirect risk-margin trust/request predicates plus shifted/noisy stress rows; its next test is repeated stress seeds and the same marginal utility accounting on its gate families.
- **ARC is still narrow.** The first-class `arc_agi` experiment now has a certified transform gate for one ARC-AGI-1 task, an ARC JSON directory loader, typed `NNProgram` transform rows, a domain-neutral program feature encoder, a learned program proposer, learned scale-composition assembly, connected-component largest-object primitives, selected-object crop/mask selectors, connector-only rendering, generic object-relation induction, an explicit `layer_*` split-panel family, color-line compression, lattice motif copy/extend, and a thin `NNHypothesis` layer that binds template slots from example-derived object deltas and lattice evidence through a top-K slot proposal queue. The grammar is still mostly hand-coded: copy/mask, layer relations, recolor, crop/compression, tile motif repeat, rotation/flip/transpose, scale, bbox, small translation, largest/selected connected components, relation-induced translation, connection, enclosed-fill transforms, and learned two-step scale compositions. The learned proposer ranks candidates better under tight budgets and now observes relation-kind, selected-object selector, background/color/offset/mode, render parameters, tile-motif repeat slots, and hypothesis-derived transform rows, but public ARC exact coverage remains 24/400 training and 7/400 evaluation tasks.
- **Statistical rigor.** The default `make prime-gap-experiments` config currently runs 1 seed; the harness supports more (`PRIME_GAP_EXPERIMENT_SEEDS=...`). The minimal-vs-leakage, strict-vs-regularized, and hand-wheel-vs-learned-rule deltas should be reported with confidence intervals across >=5 seeds.
- **Reliability diagrams.** The comparison schema stores `reliability_bins`; the README currently reports only ECE. Calibration claims would be stronger with a small reliability plot.
- **Learned-rule calibration.** The learned rule program improves ranking sharply, but its ECE is worse than the raw minimal and hand-wheel rows in the default run. That calibration tradeoff deserves a focused reliability-diagram pass.

## Reproducibility Substrate

The hybrid architecture is the contribution; the substrate is the discipline that makes it auditable. Four pieces matter:

- **Stable action IDs and trace contract.** Trace rows carry `schema: "crlp.trace.v1"`, `trace_version: 1`, `action_id_contract: "stable_unique_action_ids"`, and `ranking_policy: "score_desc_action_id_tiebreak_epsilon"`. Replay resolves `target_index` to the trace row's `action_id`, then maps that stable id back into the regenerated candidate row. New traces also log `selected_action_id`, `target_action_id`, `selected_score_margin`, `rank_score_epsilon`, and `rank_fp_sensitive`; the parser validates logged action ids and rejects records whose selected row contradicts the declared deterministic rank policy. Reordering action ids cannot silently change replay semantics; the prime replay regression test exercises that.
- **Checkpoint identity.** Checkpoints carry a `parent_model_hash`, a `training_trace_root`, a `feature_encoder_version`, and a `feature_set` in metadata; the model hash is SHA-256 over the exact checkpoint bytes. Successor models declare their parent. Comparison reports (`crlp.compare.v1`) carry `model_hash`, `parent_hash`, `feature_set`, and a machine-readable `reliability_bins` array.
- **Pure C, no third-party deps.** The neural library is dense layers, sigmoid/ReLU/tanh/linear, MSE and softmax cross-entropy, SGD, batched 4-by-4 tiled forward/backward, cloned networks, action scoring, pairwise ranking loss, cached policy-gradient updates with optional KL/entropy regularization, deterministic checkpoint save/load, SHA-256 hashing, JSONL trace writing. The core is ~3k lines; the current certified modular gate implementation is split into focused modules under `src/prime_gap/`.
- **Shared symbolic substrate.** `include/nn_symbolic.h` defines `NNRule`, `NNCertificate`, `NNSymbolicDomain`, residual coverage bitsets, feature-provenance audits for rule/scorer schemas, and a generic closed-loop residual rule policy. `NNClosedLoopRulePolicyConfig` carries the residual feature schema used by the shared policy. Prime uses that shared closed-loop policy; `examples/symbolic_domains.c` runs prime, grid, delayed-claim, and ARC through one formal rule-row and certificate interface. The smoke path certifies rows and trains a certificate-status classifier; it does not exercise production gate execution callbacks for every domain.
- **CI-baseline results.** GitHub Actions (`.github/workflows/`) pin a single `ubuntu-24.04` + `gcc-13` runner image. `check.yml` runs `make check`, the ARC induction smoke, and `typst compile paper/paper.typ` on every push and PR. `results.yml` runs nightly (and via `workflow_dispatch`), invokes `make results`, captures `results/latest/environment.json` from `scripts/environment_manifest.py`, canonicalizes timing-sensitive fields out of `comparison.jsonl` via `scripts/canonicalize_results.py`, and opens a PR if the canonical rows diverge from `main`. Local `make results` produces the same `comparison.canonical.jsonl` and `environment.json` artifacts so a developer-laptop run can be diffed against the CI baseline.

End-to-end quick path:

```sh
make
make test
make experiment-schemas
make prime-gap
make prime-gap-trace
make trace-determinism-audit
make trace-determinism-compare
make prime-gap-replay
make prime-gap-eval
make prime-rule-search
make modular-gate-search-smoke
make delayed-claim
make grid-path
make arc-agi
make arc-agi-loader ARC_AGI_TASK_DIR=/path/to/ARC-AGI/data/training
make multitask-domain
make multitask-adapter-heads
make symbolic-domains
python3 scripts/multitask_ablation_matrix.py
make results
```

`prime-gap-trace` writes `build/float/prime_v1.nnckpt` and `build/float/traces/prime_gap.jsonl`. `trace-determinism-audit` writes `build/float/traces/trace_determinism_audit.jsonl` plus `build/float/traces/prime_gap.semantic.jsonl`, which hash the replay-semantic trace fields while leaving raw scores and probabilities as diagnostics. `trace-determinism-compare` runs the same audit for float and double traces, writes the comparison to `results/latest/trace_determinism.jsonl`, and refreshes `results/latest/prime_gap_trace.semantic.jsonl`; `make check` runs this target before `paper-support`, so the C8 determinism claim is enforced locally and in CI. `prime-gap-replay` trains `build/float/prime_v2.nnckpt` from that trace. `prime-gap-eval` compares both checkpoints on the same frozen benchmark. `prime-rule-search` writes `build/float/prime_rules_v2.crlprule` and `build/float/traces/prime_rule_search.jsonl`. `arc-agi` trains the fixed `0520fde7` cell-color scorer, writes `build/float/arc_agi.nnckpt`, and emits neural scorer, certified transform-gate, and budgeted `crlp.program_proposer.summary.v1` rows. `arc-agi-loader` scans ARC-format JSON tasks from `ARC_AGI_TASK_DIR` and emits per-task `crlp.arc_transform.summary.v1` rows, a `crlp.arc_loader.summary.v1` aggregate, and matched program-proposer aggregates. The loader also accepts `--program-train-dir` to pretrain the learned program proposer on another ARC-format split before evaluation. `make results` writes `results/latest/summary.md`, `results/latest/comparison.jsonl`, `results/latest/commands.txt`, `results/latest/trace_determinism.jsonl`, and `results/latest/prime_gap_trace.semantic.jsonl`; bulky logs stay under ignored `results/**/logs/`. `make symbolic-domains` is currently outside the canonical `make results` capture script, so run it separately when refreshing symbolic-domain smoke evidence.

`make modular-gate-search-smoke` runs the broader certified modular gate family smoke. It emits matched-budget `crlp.modular_gate.summary.v1` rows for:

- `prime_constellation`: reject twin-prime shifts when a tuple member is certified composite by a small-prime residue.
- `kfree_squarefree`: reject integer candidates divisible by `p^2` or stronger powers.
- `fibonacci_lucas`: reject recurrence indices whose Fibonacci or Lucas value is zero modulo a prime, certified through Pisano-period residues.

Each task is evaluated at the same rule budget with `grammar_order`, `random`, `structural_heuristic`, `learned_structural`, `train_marginal_oracle`, and `validation_marginal_oracle` orderings. The learned row is a structural-only linear proposer trained on train-set rule coverage labels; verifier counts are labels/diagnostics, not proposer inputs. The smoke writes per-rule `crlp.modular_gate.rule_trace.v1` traces to `build/$(REAL)/traces/modular_gate_search.jsonl` and requires zero false rejections. These rows extend the family boundary; they are not yet the canonical multi-seed gate-ablation benchmark.

Cross-task tuning uses a separate noisy workspace so canonical result snapshots stay stable:

```sh
make neurosymbolic-tune
```

This runs `scripts/neurosymbolic_tuning.py` over the prime rule-search grammars, grid legality rule search, delayed-claim replay, and grid replay under one profile. It writes raw task rows to `build/float/neurosymbolic_tuning/raw.jsonl`, a cross-task score table to `normalized.csv`, and a compact `summary.md`. Grid replay tuning defaults to `with-priors`, because that is the feature set currently supported by the trace/replay path. Use `scripts/neurosymbolic_tuning.py --profile full` for larger budgets and sample counts.

The shared-model smoke test trains one action scorer across all configured domains:

```sh
make multitask-domain
```

`examples/multitask_domain.c` wraps prime-gap, grid-path, delayed-claim, and one ARC-AGI-1 sample into one fixed-width action-row envelope. The ARC sample is training task `0520fde7`, represented as a balanced cell-color action task: given a train/test input grid and an output-cell coordinate, choose the output color from actions `0..9`. ARC rows expose the flattened input grid, the output coordinate, and the candidate action, but not the direct overlap/intersection rule. This is a useful integration test for grid-to-grid symbolic tasks inside the shared action-row envelope.

Use `--tasks prime,grid,claim,arc|all` for combination ablations and `--eval-tasks all` to evaluate held-out domains after training on a subset. A matched per-domain update comparison can scale `--steps-per-epoch` by the number of active domains, for example `--tasks prime --steps-per-epoch 3000`, `--tasks prime,grid --steps-per-epoch 6000`, and `--tasks all --steps-per-epoch 12000`. Capacity and sampling are explicit with `--hidden 64x32` and `--training-order uniform_random|round_robin|arc_heavy|prime_light`; emitted multitask rows include ECE, reliability bins, parameter count, and forward FLOPs. `--task-heads` additionally emits `shared_task_heads` rows from a shared trunk with per-domain scalar heads; `python3 scripts/multitask_adapter_heads.py` captures the R-MT.4 comparison. The remaining feature ablation is explicit: `--no-task-tags` removes the task/domain one-hot fields. The default shared multitask envelope is now the same role-tagged 96-slot envelope used by action and rule candidates, and it excludes prime residue flags so the prime row receives no hand-coded modular predicates.

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

## Running the Certified Modular Gate Search Instance

The headline pipeline is `make prime-gap` followed by `make prime-rule-search`. The target names remain prime-specific because `prime_gap` is the current task id and result-schema key; conceptually, this is the certified modular gate search instance.

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

Runs the prime-gap rule-grammar instantiation of certified modular gate search. The neural proposer scores unconditional candidate filters of the form `reject candidate n when n > m and n % m == r` over `m=3..255`; `make prime-rule-search-baselines` runs both that grammar and the conditional gap grammar with neural, closed-loop, unified, random, grammar-order, structural-heuristic, and validation-oracle orderings at a fixed 16-rule budget. `make prime-rule-search-conditional` emits a trace/checkpoint for the broader conditional gap grammar. The static neural proposer's feature set is structural-only and excludes support counts, utility, and false-rejection counters. It trains on a budgeted marginal-utility target computed from verifier labels; `--proposer closed-loop` trains on residual coverage trajectories and scores each next rule against the accepted set and current residual state; `--proposer unified` consumes the same residual-aware envelope but trains jointly with action-scoring rows. `--max-marginal-states` controls the sample cap for those targets and defaults to 4096. `--coverage-cache-dir` reuses the precomputed coverage bitset across baseline orderings on the same split. `--closed-loop-hidden` and `--unified-scorer-hidden` select the neural proposer's MLP layer sizes (e.g. `64x32`). The verifier accepts only rules with a deterministic algebraic certificate. The JSONL trace uses schema `crlp.rule_trace.v1` and records grammar version, selection strategy, proposer feature set, proposer target, rule text, model score, verifier result, deterministic-certificate flag, accepted flag, support counts, marginal utility, and false-rejection counts.

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

`candidate_inputs` is row-major: `candidate_count * nn_input_size(scorer)` values. The scorer must have one scalar output. Probabilities sum to approximately `1.0`, and ranking is deterministic for equal scores. Trace-producing proposer code should use `nn_rank_action_scores_with_ids(..., NN_RANK_SCORE_EPSILON)` so near ties are broken by stable action id instead of candidate row order; plain `nn_rank_action_scores` remains the index-stable helper for legacy callers and local diagnostics.

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
{"schema":"crlp.trace.v1","trace_version":1,"action_id_contract":"stable_unique_action_ids","task":"prime_gap","state_hash":"0000000000000000","model_hash":"unknown","feature_encoder_version":1,"candidate_count":2,"action_ids":[10,20],"scores":[0.25,-0.25],"probabilities":[0.62,0.38],"selected_index":0,"target_index":1,"loss":0.7,"selected_action_id":10,"target_action_id":20,"selected_score_margin":0.5,"rank_score_epsilon":1e-06,"rank_fp_sensitive":false,"ranking_policy":"score_desc_action_id_tiebreak_epsilon","verifier_result":"correct","verifier_cost":{},"seed":0,"feature_set":"minimal"}
```

Prime-gap traces add `p`, `true_gap`, `proposal_steps`, `wheel_exact`, `assisted_wheel`, `block_cost`, `assisted_block`, and `log_scale`. Rule-search traces use schema `crlp.rule_trace.v1`. Delayed-claim traces use the deterministic per-row seed for replay.

Use `nn_trace_parse_jsonl` to validate rows before replay. It checks the schema string, trace version, action-id contract, required scalar fields, unique action ids, finite score/probability arrays, selected/target index bounds, probability mass, optional logged action ids, optional logged score margin, and declared ranking policy. Replay code resolves `target_index` to the row's stable `action_id`, then maps that action id back into the regenerated candidate rows before training. Float scores remain residual ranking hints rather than semantic truth; action ids and verifier outcomes are the replay contract. `nn_trace_json_get_u64`, `nn_trace_json_get_real`, and `nn_trace_json_get_string` are small helpers for task-specific extra fields.

`scripts/trace_determinism_audit.py` projects trace rows onto `crlp.trace.semantic.v1` rows containing only replay-semantic fields (`state_hash`, ordered `action_ids`, selected/target action ids, verifier result, seed, and ranking policy), computes a semantic SHA-256 hash, and reports rank-sensitivity diagnostics (`rank_fp_sensitive` count/rate plus min/median/p95 score margins). Pass multiple trace paths to emit a `crlp.trace_determinism_compare.v1` row for cross-run semantic-hash checks; the paper-support claim `C8.trace_determinism_audit` requires the float/double semantic hashes to match.

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

`grid_path` is a second exact-verifier task. A deterministic generator creates a small obstructed grid, start, and goal. Candidate actions are all four moves. The neural model ranks moves from observable map and move features, while an exact BFS verifier owns correctness by identifying the shortest-path successor. `grid_path --rule-search` runs the cross-task budgeted certified-gate experiment: a structural proposer scores ten certified rules covering out-of-bounds moves, blocked-cell moves, local dead ends, component traps, clear-path admissible bounds, and shortest-distance dominance, then accepts the top four by residual marginal utility under a fixed rule budget.

The task reports accuracy, NLL, calibration error, decision utility, proposal steps, fixed-order baseline steps, exact verifier nodes, baseline verifier nodes, and candidate counts before/after the certified gate. `grid-path-eval` emits a rule-family sweep: cheap-local legality/topology rules, bound-only clear-path rules, cheap topology+bound rules, the exact-distance dominance rule by itself, and the mixed gate. `grid-path-budget-sweep` runs family/budget proposer baselines over neural marginal-greedy, random, grammar-order, structural-heuristic, and validation-oracle ordering. Trace replay stores stable move action IDs, then maps them back into the regenerated four-row candidate set before training a successor model.

```sh
make grid-path
make grid-path-rule-search
make grid-path-budget-sweep
make grid-path-trace
make grid-path-replay
make grid-path-eval
```

## Shared Multitask / ARC-AGI Sample

`arc_agi` is now also a standalone first-class schema/result target:

```sh
make arc-agi
```

The standalone task is ARC-AGI-1 task `0520fde7`, reduced to output-cell color prediction over actions `0..9`. The neural model sees the flattened input grid, target output coordinate, candidate color, and action one-hot. It does not receive a direct overlap/intersection feature. The same binary now also evaluates a tiny transform grammar, extracts object-scene evidence, maps each primitive or composed transform to a typed `NNProgram`, appends ARC shape/color/object-relation/object-selector parameter descriptors, reports matched program-proposer budget curves, and reports the certified transform gate separately from the neural scorer.

The loader mode runs the same transform grammar over ARC-format JSON files:

```sh
make arc-agi-loader ARC_AGI_TASK_DIR=/path/to/ARC-AGI/data/training ARC_AGI_MAX_TASKS=400
./build/float/arc_agi --program-train-dir /path/to/ARC-AGI/data/training --task-dir /path/to/ARC-AGI/data/evaluation --max-tasks 400 --loader-summary-only --program-budget-grid
```

The loader summaries report corpus coverage for the current transform grammar.
On the public ARC-AGI corpus snapshot used for the current evidence pass,
grammar v15 loaded all 400 training and 400 evaluation tasks with no parse
errors, certified 24 training tasks and 7 evaluation tasks, and covered
1187/56672 training cells plus 1057/98515 evaluation cells. The
connected-component largest-object primitive accounts for two training-only
gains, learned scale composition adds the training-only `f25fbde4` certificate,
and generic object-relation induction adds the training-only `ded97339`,
`00d62c1b`, `a5313dff`, and `a79310a0` certificates. Grammar v15 also carries
learned unique/largest/smallest/interior/border/nearest selected-object priors,
connector-only rendering, and a `layer_*` split-panel family as an explicit
symbolic layer/depth prior. The `compress_color_*` family removes uniform
separator/background rows and columns; it adds the `68b67ca3` evaluation
certificate over the grammar v12 snapshot. The `tile_motif_parity_repeat`
family copies a single motif across same-parity separator-grid panels; it adds
the large `92e50de0` evaluation certificate over the grammar v13 snapshot, and
v15 also exposes the same lattice verifier as a `tile_motif_repeat`
`NNHypothesis` template for proposer-side induction.
The supported families include
split-panel masks, same-shape color maps, foreground crops,
color-line compression, separator-grid motif repeat, flips/rotations/transposes,
scaling, bbox drawing, small translations, largest and selected
connected-component crop/mask transforms, relation-induced translation,
connection, enclosed-fill transforms, and bounded scale-composition transforms.

The clean ARC evidence surface is regenerated with:

```sh
make arc-agi-evidence-surface ARC_AGI_TASK_DIR=/path/to/ARC-AGI/data
```

It writes fixed-task evidence, public train/eval budget curves, per-task eval
rows, and a miss taxonomy under `ARC_EVIDENCE_SURFACE_DIR`.

The family-discovery pass mines ARC induction misses into recurring symbolic
search-space targets:

```sh
make arc-family-discovery ARC_AGI_TASK_DIR=/path/to/ARC-AGI/data
```

It consumes the task-level induction diagnostics JSONL, extracts train-pair
signatures, clusters repeated failure shapes, and emits candidate family stubs
plus pseudo-labels for a future neural family proposer. The output is a search
agenda, not a certificate; promoted families still need train-pair verification
and zero-regression public ARC measurement.

`multitask_domain` remains the generalization smoke test for the core scorer API. It trains one scalar action scorer over a fixed 96-feature row envelope while shuffling samples from `prime_gap`, `grid_path`, `delayed_claim`, and `arc_agi`.

```sh
make multitask-domain
make multitask-adapter-heads
python3 scripts/multitask_ablation_matrix.py
```

The multitask ARC row uses the same fixed task and adds the optional task tag from the shared envelope. The matrix script enumerates every non-empty task combination, evaluates all domains each time, and writes JSONL/CSV/Markdown summaries under `build/float/multitask_ablation_matrix/`.

Current all-domain top-1 means over three seeds:

| Variant | Prime | Grid | Claim | ARC |
| --- | ---: | ---: | ---: | ---: |
| Task tags | 11.77% | 90.37% | 95.50% | 26.13% |
| No task tags | 14.07% | 90.00% | 95.97% | 30.90% |

Prime residue features were removed from this shared envelope, so the weak prime row is expected. The useful signal is that one C scorer can train across all four action-row domains without task-specific code in the network itself.
