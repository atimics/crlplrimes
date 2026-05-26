# Generalized Experiment Schema

The project is one grounding-gated action-scoring architecture with first-class
task schemas plus case-study discovery schemas. Certified modular gate search,
currently instantiated as `prime_gap`, grid paths, delayed claims, Signal flight
control, program synthesis, and the ARC-AGI sample/loader path must all be
described with the same contract before they are used as evidence.
The evidence maturity
differs by task: the `prime_gap` certified modular gate instance has the
complete production learned-gate table, grid has budgeted certified cheap-local
and exact-distance gate families, delayed-claim has parent-hash trace replay plus a
production threshold+direct/indirect-margin gate/proposer loop with shifted/noisy
stress rows, and ARC is a
fixed-sample scorer plus a narrow corpus-loader transform gate. Signal flight now has a counterfactual generated
trace path: each sampled state enumerates WASD hypotheses, replays them through
Signal's authoritative simulator, and trains around certified progress,
hull-loss, damage, and death outcomes. Signal also has a closed-loop rollout
evaluator that first enumerates certified control-target hypotheses (`direct`,
`nav_path`, `left_bypass`, `right_bypass`, `brake_escape`) and then lets trained
checkpoints rank WASD actions around the selected hypothesis. The rollout
compares `goal_heading_no_safety`, `flight_steer_to`, `neural_raw`, and
`neural_guarded` controllers on identical deterministic episodes. The external
websocket pilot records a compatibility hash from Signal's protocol headers
before active play, so protocol drift turns into underpowered live evidence
instead of silent decoder reuse.
The summary rows report neural utility/regret/damage/death metrics beside
primitive and deterministic flight baselines.
Signal mining now adds the next tactical layer: sampled states enumerate
mine/tow/smelt hypotheses, replay each hypothesis through the authoritative
simulator, and train a tiny scorer to rank certified tactical utility rather
than imitate an autopilot trace.

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
  grounding operator O_T(s, a),
  grounding authority level,
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

The current C schema emits both the historical verifier fields
(`exact_verifier`, `verifier_feedback`) and explicit grounding fields:
`grounding_authority`, `grounding_operator`, `candidate_grammar`,
`outcome_row_schema`, `claim_scope`, and `known_non_claims`. In this document,
"grounding operator" is the broader contract term for exact checkers,
deterministic replay, corpus proxies, and review gates that populate those
verifier-named outcome fields.

The task-schema C API lives in `include/nn_experiment.h`; the implementation is
`src/nn_experiment.c`. The shared symbolic-domain API lives in
`include/nn_symbolic.h`; its implementation is `src/nn_symbolic.c`. The shared
typed program/transform API lives in `include/nn_program.h`; its implementation
is `src/nn_program.c`. The shared candidate-verifier protocol row API lives in
`include/nn_protocol.h`; its implementation is `src/nn_protocol.c`.

The protocol layer intentionally owns row shape rather than grammar semantics.
Domain code still defines prime modular rules, grid topology predicates,
delayed-claim thresholds, ARC visual-program compiler primitives, and Signal
control options. Shared code normalizes those rows as:

```text
data -> grammar -> candidate set -> grounding outcome -> learned ranking
     -> discovered rules/library feedback -> result -> limitation
```

ARC is the pressure case for moving this boundary downward. Handwritten
transform templates are now treated as legacy compiler/operator scaffolding;
the learned surface is the typed visual-algebra program plus its coverage holes,
slot choices, neural proposals, and verifier outcomes. A row that succeeds only
because a legacy template owns the transform is useful training evidence, but
not coverage credit or the end-state grammar abstraction.

The canonical JSONL row schemas are:

- `crlp.protocol.candidate_row.v1`
- `crlp.protocol.verifier_outcome.v1`
- `crlp.protocol.verifier_feedback_signal.v1`
- `crlp.protocol.ranker_decision.v1`
- `crlp.protocol.discovered_rule.v1`
- `crlp.protocol.library_promotion.v1`

The candidate/outcome protocol is now the intended evidence spine, not only a
small helper row. Candidate rows can carry source schema/artifact, state hash,
and optional `crlp.neural_frame.v1` fields (`frame_sequence_id`, `frame_index`,
`frame_length`, previous/next summaries, and composition context). Verifier
outcome rows can carry evidence attachments for state hash, parent hash,
trace/root hash, source artifact, authority, signature, replay seed,
failure-stage, residual-class, and domain payload schema. That gives ARC exact
prediction rows, Signal replay rows, and future signed cargo/handoff rows the
same outer contract while leaving the domain payloads domain-owned.
Verifier feedback signal rows are the shared bridge from authority outcomes to
later proposer/ranker supervision. They carry the authority level, outcome
status, reason code, residual class, utility/cost/support counts, and declared
target role, but they cannot promote a rule or candidate without the original
domain grounding operator.

Schema fields describe the intended formal contract. The evidence-status section
below is the controlling source for which production binaries currently execute
which gate and baseline paths.

## Grounding Operators

The architecture uses "grounding operator" as the general term for the component
that turns a candidate into an auditable outcome row. An exact verifier is the
strongest instance, but not the only one.

| Authority | Operator class | Examples | Claim scope |
| --- | --- | --- | --- |
| `exact_check` | proof, lookup, type check, or deterministic verifier | `prime_gap`, `grid_path`, certified ARC transforms, program-synthesis examples | correctness and zero-false-rejection claims inside the declared support |
| `deterministic_replay` | simulator or replay that decides candidate consequences | Signal flight/mining, delayed-claim hidden-outcome simulator | replay-certified utility, damage, death, regret, or decision metrics |
| `corpus_proxy` | deterministic corpus-derived grammar or corruption benchmark | haiku prototypes, public ARC coverage surfaces | corpus-fit, coverage, and near-miss discrimination, not semantic truth |
| `human_review` | human acceptance/rejection over reviewable artifacts | editorial-agent PRs, future preference rows | review-gated assistance or preference evidence, not autonomous claim authority |

Promoted result rows and docs must declare the authority level because the same
scorer role has different implications under each operator. Signal simulator
replay can ground damage and death outcomes; a haiku corruption benchmark can
only ground a corpus-proxy discrimination claim. Both fit the loop, but they do
not license the same language.

## Enforced Schema Fields

The schema is not limited to strings. Each task also declares:

- `NNFeatureSchema[]`: one row per scorer input feature, with a typed
  `NNFeatureProvenance`.
- `NNGroundingAuthority`: `exact_check`, `deterministic_replay`,
  `corpus_proxy`, or `human_review`.
- Grounding scope text: `grounding_operator`, `candidate_grammar`,
  `outcome_row_schema`, `claim_scope`, and `known_non_claims`.
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
`NN_LEAKAGE_GATE_ALLOWED` is for explicitly declared proposer/gate surfaces such
as closed-loop residual-set features; it permits gate-derived features but still
forbids verifier-derived scorer inputs.

The schema writer refuses to emit a task whose enabled scorer features violate
its declared leakage policy. The same reusable audit helper is used by
`NNSymbolicDomain` and `NNClosedLoopRulePolicyConfig` so symbolic proposers
cannot silently add verifier-derived inputs. Comparison rows also carry
`leakage_policy`, so mixed policy tables are visible in `results/latest`.

## Invariants

1. Candidate enumeration is complete before gating.
2. Scorer features are observable state/action features only.
3. Grounding outputs may be labels, rewards, certificates, diagnostics, proxy
   outcomes, or executed gates, but they are not default scorer inputs.
4. Static proposer inputs are structural rule descriptors only; closed-loop
   proposer inputs may include declared gate-derived residual-set features.
5. Empirical support, utility, and false-rejection counts are targets or
   diagnostics, not proposer features.
6. A rule enters the gate only through the task certificate checker.
7. Every gate-learning claim needs matched-budget neural, random, grammar-order,
   structural-heuristic, and oracle orderings.
8. Every replay claim uses stable action ids.
9. Trace-producing rankers declare their deterministic ranking policy and emit action-id plus score-margin guards so float-sensitive ties are visible instead of silently changing replay semantics.
10. Numeric training claims must distinguish semantic certification from local optimizer hygiene. `crlp.numeric_step_certificate.v1` rows may certify that supervised or KL/entropy policy updates stayed finite and within declared parameter/logit/margin bounds, and `crlp.neuron_contract.v1` rows may certify interval bounds for dense neurons. Shared supervised calls emit rows at update granularity, env guard variables can tighten finite, parameter, logit, delta, margin, and shrink-step bounds for a run, and `crlp.numeric_certificate_budget.v1` rows define the per-target p99/max envelopes that `make check` enforces. They do not promote a candidate, rule, or program without the domain grounding operator.
11. Verifier-feedback rows may train proposer/ranker ordering, but they are never promotion evidence unless replayed through the task authority. Rejected-candidate reason learning and hypothesis-library feedback are instances of this shared loop.
12. Learned-grammar coverage is audited before ranker quality. `crlp.learned_grammar_coverage.v1` rows first ask whether the candidate space contains a train-consistent solution, then whether the neural ranker brings it near the top, whether the verifier certifies it, and whether held-out/test-pair transfer survives. ARC rows also report aggregate visual-replacement gaps so legacy-template debt, missing visual candidates, and visual selection failures remain distinct even when sampled coverage holes are truncated. `crlp.experiment_health.v1` rows consolidate those gates with numeric certificate rows, coverage holes, verifier deltas, learning signal, and generated paper-claim support.
13. Capacity and architecture claims are blocked by the plateau audit. Until `C7.tiny_net_plateau_stop_rule` is supported for the affected task/variant, width and adapter-head rows remain underpowered diagnostics rather than registered architecture conclusions.

## Current Instantiations

Grounding-backed first-class task schemas:

| Task | State | Actions | Grounding operator | Authority | Grammar/certificate | Production gate status |
| --- | --- | --- | --- | --- | --- | --- |
| `prime_gap` | prime `p` | candidate gaps | segmented sieve next-prime verifier | `exact_check` | modular predicates plus algebraic divisibility proof | certified modular gate search instance with learned-rule gate plus matched baselines |
| `grid_path` | grid, start, goal | four moves | BFS shortest-path successor | `exact_check` | legality/topology/bound predicates that cannot reject an optimal move | budgeted certified gate with family/budget proposer sweep |
| `delayed_claim` | observable claim features | trust/ignore/request/cautious | deterministic hidden-outcome simulator | `deterministic_replay` | threshold and margin predicates plus finite-generator safety check | production threshold+margin gate plus matched proposer baselines and shifted/noisy stress rows; direct clean residual resolved, proxy/noisy frontier exposed |
| `arc_agi` | ARC-AGI-1 task `0520fde7` input grid plus output coordinate | output colors `0..9` | known ARC train/test output grid | `exact_check` | train-pair consistency plus unique grammar-consistent test prediction | fixed-sample certified transform gate plus ARC JSON directory loader and matched `NNProgram` proposer rows for narrow geometry/object evidence |
| `signal_flight` | Signal ship kinematics, hull, path, clearance observations, and target/control-point geometry | WASD key combinations around certified control-target hypotheses | Signal simulator replay and live telemetry | `deterministic_replay` | replay-certified progress, hull loss, damage, death outcomes, rollout-time control-target certificates, and protocol compatibility | counterfactual utility scorer, certified-hypothesis rollout evaluator, and protocol-checked external websocket pilot |
| `signal_mining` | Signal tactical mining state after deterministic scenario prefixes: ship state, tow chain, rocks, fragments, refinery/dock demand | bounded policy hypotheses from target selector, route constraint, speed/contact profile, and interaction primitive | fresh deterministic replay in Signal's authoritative simulator | `deterministic_replay` | replay-certified utility, smelts, pickups, fractures, tow count, repair-kit deltas, hull loss, damage, and death | grammar hypothesis ranker with fixed-priority grammar baseline via `make signal-mining` |

Grounding-backed case-study discovery schemas:

| Task | State | Actions | Grounding operator | Authority | Grammar/certificate | Evidence status |
| --- | --- | --- | --- | --- | --- | --- |
| `program_synthesis` | list-routine task with input/output examples | candidate library abstraction | run abstraction on every example | `exact_check` | monomorphic list/int DSL type check | matched-budget scaffold with random, grammar-order, structural, validation-oracle, static neural, and closed-loop rows |

Corpus-proxy prototypes such as `haiku_word_lm` and `haiku_understanding` use
the same candidate/grounding/scorer/trace loop but are not production paper
claims. Their grounding operator is a train-split dataset vocabulary,
adjacency graph, and corruption benchmark; the supported wording is
corpus-grounded form or near-miss discrimination, not semantic truth or broad
language understanding.

Leakage status today:

| Task | Leakage policy | Feature audit |
| --- | --- | --- |
| `prime_gap` | `forbidden` | 3 enabled observations; 10 disabled structural-prior residue columns retained only for ablations |
| `grid_path` | `priors_allowed` | 61 observations plus 2 enabled Manhattan structural priors |
| `delayed_claim` | `forbidden` | 10 enabled observations |
| `program_synthesis` | `priors_allowed` | 18 enabled structural-prior DSL abstraction features |
| `arc_agi` | `forbidden` | 49 enabled observations over grid cells, output coordinate, candidate color, and action one-hot |
| `signal_flight` | `forbidden` | 48 enabled observations over WASD candidate controls, kinematics, clearances, path state, hull, and heading |
| `signal_mining` | `forbidden` | 70 enabled observations over grammar hypothesis id, target/route/speed/interaction primitives, ship state, tow state, rock/fragment distances, refinery/dock demand, and prefix context |

## Certified Modular Gate Search

`prime_gap` is the current production member of a broader certified modular gate
search family. A task belongs to this family when candidate rejection can be
expressed as residue, divisibility, period, or congruence rules and every
accepted rule can be checked by a deterministic certificate. The current
prime-gap grammar proves that rejected `p + g` candidates are composite. Future
members should keep the same schema surface while changing the arithmetic
substrate: prime constellations for depth inside prime residues, k-free or
polynomial-value filters for divisibility gates outside next-prime search, and
Fibonacci/Lucas gates for recurrence-period structure.

`examples/modular_gate_search.c` is the first implementation of that broader
family boundary. It emits `crlp.modular_gate.summary.v1` and
`crlp.modular_gate.rule_trace.v1` rows for three smoke-level domains:
`prime_constellation`, `kfree_squarefree`, and `fibonacci_lucas`. The selector is
matched-budget across `grammar_order`, `random`, `structural_heuristic`,
`learned_structural`, `train_marginal_oracle`, and
`validation_marginal_oracle` rows. The learned row is structural-only and trains
against train-set rule coverage labels; support, marginal utility, and
false-rejection counts stay labels or diagnostics. The smoke target requires
zero held-out false rejections. These rows are extension evidence for the family
definition; they are not yet a replacement for the production `prime_gap`
learned-gate table.

The first-class `arc_agi` default schema is still one fixed ARC-AGI-1 task. It
emits both a neural cell-scorer comparison row and a certified transform-gate
row. The same binary can also scan ARC-format JSON directories with `--task-dir`
and emit per-task `crlp.arc_transform.summary.v1` rows plus a
`crlp.arc_loader.summary.v1` aggregate. Transform rows include both
`accepted_rules` and a deterministic `prediction_hash`, which lets invariance
checks compare rule identity and predicted outputs without expanding every grid
cell into CSV. It also maps transform candidates into typed `NNProgram` rows and
emits `crlp.program_proposer.summary.v1` for
grammar-order, structural, random, learned-program, and oracle orderings across
matched program-budget curves. The learned proposer uses the shared
`nn_program_encode_features()` encoder plus ARC task-level shape/color/object
descriptors, and can be pretrained with `--program-train-dir` before evaluating
another split. The loader evidence is intentionally narrow: grammar
version 14 covers two same-height panel masks, explicit layer-relation priors,
same-shape color maps, foreground/object crops, separator/background line
compression, separator-grid motif repeat, rotations, flips, transposes, small
integer scales, bbox drawing, small foreground translations, and largest
connected-component crop/mask transforms. The multitask binary uses a separate fixed 96-wide
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

The value vocabulary now includes Signal/system objects in addition to ARC/DSL
objects: `sim_state`, `trajectory`, `waypoint`, `resource_transfer`,
`receipt_chain`, `handoff_ticket`, `authority`, `zone`, and `ship_snapshot`.
These are type-contract hooks only; they do not make Signal authority or ARC
visual reasoning automatic.

## Protocol Operator Promotion

`scripts/protocol_spine.py` normalizes candidate/outcome evidence into
`crlp.protocol.candidate_row.v1` and `crlp.protocol.verifier_outcome.v1`. The
promotion learner in `scripts/protocol_operator_promotion.py` reads that ledger,
joins rows by domain and stable id, classifies residuals into operator families,
and emits `crlp.protocol.operator_promotion.v1` records plus a
`crlp.protocol.operator_prior.v1` TSV.

The promotion rule is deliberately conservative: an operator family needs
repeated verifier-backed rows and cross-domain support. A promoted row is still
a prior, not a gate. It names the compiler contract and input/output slots that
a domain adapter can try next, but correctness still requires that adapter to
compile the operator and replay the declared grounding operator.

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
| `train_rejects`, `validation_rejects`, `test_rejects` | summed support over accepted rules; overlapping rules can credit the same state-action rejection more than once |
| `train_marginal_rejects`, `validation_marginal_rejects`, `test_marginal_rejects` | accepted-order non-overlapping rejects credited only to the first accepted rule that rejects an action |
| `train_marginal_false_rejections`, `validation_marginal_false_rejections`, `test_marginal_false_rejections` | accepted-order non-overlapping false rejections, using the same first-rejecting-rule attribution |

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
ARC loader is the path that now carries typed `NNProgram` rows, largest
connected-component primitives, layer/compression priors, and separator-grid
motif repeat. Both paths report transform-grammar coverage and certificate
evidence.

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
| `feature_envelope` | shared row layout, currently `task_role_tagged_action_rule_rows_v1` or `untagged_action_rows_v1` |
| `train_updates` | number of SGD updates received from this domain |
| `epochs`, `steps_per_epoch`, `hidden`, `parameters`, `forward_flops`, `learning_rate`, `seed` | training and capacity configuration |
| `sample_count`, `candidate_count` | evaluation size and mean action count |
| `top1`, `top5`, `top10`, `nll`, `ece`, `reliability_bins`, `confidence_mean`, `correct_confidence_mean`, `incorrect_confidence_mean`, `mean_rank` | ranking and calibration metrics |
| `decision_utility`, `baseline_steps` | task-specific utility/probe fields when available |
| `arc_fixed_grid_exact`, `arc_fixed_cell_accuracy`, `arc_fixed_prediction_hash`, `arc_fixed_targets`, `arc_fixed_predictions` | fixed `0520fde7` grid diagnostics on ARC summary rows |
| `arc_fixed_confidence`, `arc_fixed_target_probability`, `arc_fixed_target_color_count`, `arc_fixed_predicted_color_count`, `arc_fixed_correct_target_color_count` | fixed-ARC per-cell/per-color diagnostics for distinguishing cell-mode failures from smooth scorer drift |

The removed `prime_features` key should not reappear in multitask outputs.
Prime residue predicates are intentionally outside the shared multitask envelope.

The same binary also emits `crlp.multitask.epoch.v1` rows during shared training.
Each row reports one domain's per-epoch mean loss, epoch update count, cumulative
domain updates, hidden size, learning rate, seed, task set, feature envelope, and
training order. These rows are diagnostics for curriculum/order and optimization
experiments; they are not aggregate evaluation rows.

## Evidence Status

`prime_gap` has the most complete gate ablation: no gate, fixed hand gate,
learned certified gate, and fixed-budget proposer baselines. It should be
described as the first certified modular gate search instance, not as the
project goal.

`grid_path` has raw versus budgeted certified gate evaluation under two visible
feature regimes: `no_manhattan_priors` for the clean scorer row and
`with_manhattan_priors` as a priors-allowed ablation. Its gate rows now split
cheap-local predicates, bound-only clear-path predicates, cheap topology+bound
predicates, the exact-distance oracle rule, and the mixed family. `--baseline-sweep`
compares neural, random, grammar-order, structural-heuristic, and
validation-oracle rule ordering under one budget, while `--budget-sweep`
runs the family/budget grid with repeated neural and random seeds. Default neural
selection is now marginal-aware: it targets train/validation residual utility, so
the straight-line bound cannot consume budget after the two-segment bound has
already covered its rejects. The raw support/marginal split remains in the result
rows as an overlap diagnostic.

`delayed_claim` supports supervised training, trace replay, calibration
evaluation, matched-budget certified gate evaluation, and shifted/noisy stress
evaluation in its production binary. Its direct threshold+margin grammar
carries a finite-generator certificate: threshold rules prune dominated actions,
while trust/request utility-margin rules resolve the clean and shifted
deterministic residual with zero false rejections. The indirect risk-margin
grammar is less decisive but safer under the observation-noise row. The next
open work is noise-aware certification or abstaining bands for the hard
uncertainty+noise slice, not basic gate wiring.

## Next Build Targets

Substrate-engineering work that is *not* defending a paper claim and so does
not appear in the paper Roadmap (`paper/paper.typ` `tab:roadmap`). For
experiment-level items see that table; for the multitask operational plan see
`docs/multitask-transfer-research-plan.md`. For the code-to-PDF claim layer
that turns evidence rows into generated paper claims and gap reports, see
`docs/self-authoring-paper-roadmap.md`.

1. Move `grid_path` and `delayed_claim` out of monolithic examples into task
   modules with the same shape as `src/prime_gap/` (refactoring; precondition
   for richer per-task targets).
2. Add noise-aware delayed-claim certificates or abstaining margin bands, then
   repeat the stress rows across seeds.
3. Port the grid-path family/budget-sweep discipline to delayed-claim
   and report accepted-order marginal utility alongside per-rule support
   (cross-listed as R-E1.2/R-E2.1 follow-up).
4. Run repeated-seed proposer baselines for delayed-claim at matched rule
   budgets, matching the grid-path budget sweep (cross-listed under R-E2.1).
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
