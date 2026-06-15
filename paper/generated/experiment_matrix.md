# Frozen Experiment Matrix

Date: 2026-06-14

The evidence families are frozen to these rows. New domains belong in roadmap notes until this matrix is deliberately revised.

Authority is a hard reporting axis. Exact checks, deterministic replay, corpus proxies, and mixed diagnostics are not collapsed into a single score.

| Domain/data | Authority | Grammar | Candidate set | Verifier/gate | Neural role split | Headline baseline | Trace artifact | Result table | Takeaway | Limitation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `prime_gap`: next-prime gaps over segmented-sieve windows; broader family includes period-residue sequence filters | `exact_check`: zero-false-rejection gates and exact residual ranking inside the next-prime support | local modular filters over residues, gap classes, and wheel support | next-prime gap actions plus certified modular filters | segmented-sieve replay and modular certificate checks; zero-false-rejection divisibility certificates for the audited next-prime instance | scorer: score residual actions and certified filters after the gate defines the target; proposer: static proposer is a negative baseline; closed-loop and unified proposer rows must be compared to grammar order | scorer baseline is the certified gate; proposer baseline is grammar order plus verifier-feedback variants | `crlp.compare.v1; crlp.rule_trace.v1; crlp.rule_search.summary.v1` | learned filter 68.87% top-1 vs 45.12% hand-wheel and 9.13% raw; seed interval audited | certified modular structure is the candidate mechanism; the neural scorer ranks residual budget choices while Fibonacci/Lucas-style filters remain boundary evidence. | static neural proposer is not superior to deterministic baselines at fixed budget |
| `grid_path`: grid mazes with deterministic shortest-path labels | `exact_check`: shortest-path successor and certificate claims inside generated grids | local moves plus boundary, trap, path-clear, and distance predicates | four moves plus symbolic local/topology/bound predicates | BFS shortest-path successor and rule certificate checks; legality, trap, clear-path, and distance-dominance certificates | scorer: rank legal residual moves under exact shortest-path labels; proposer: budgeted rule-order diagnostics only | BFS successor and cheap symbolic gates | `crlp.compare.v1; crlp.symbolic_domain.trace.v1` | cheap-bound gate lifts priors-enabled raw 75.80% to 90.10% | the same candidate-verifier contract transfers to topology once the grammar stays domain-local. | supporting transfer surface, not a main contribution |
| `delayed_claim`: delayed reward/claim rows with hidden outcomes and margins | `deterministic_replay`: simulator-certified utility and safety under the delayed-claim generator | threshold, margin, and sibling-claim predicates | claim actions, threshold gates, and utility-margin rules | deterministic hidden-outcome simulator and rule certificate checks; finite-generator safety check plus false-rejection counters | scorer: rank utility after simulator safety dominates the decision; proposer: consume simulator/gate rejection reasons as scoped feedback | deterministic clean threshold and margin gates | `crlp.delayed_claim.rule_trace.v1; crlp.rejected_candidate_learning.*` | clean threshold+margin gate reduces four actions to one; noisy rows expose failures | utility ranking must be downstream of safety when false rejection risk is domain-specific. | reason-prior transfer alone can fail safety; claim remains domain-scoped |
| `rejected_candidate_learning`: verifier rejection rows grouped by seed, split, and reason family | `exact_check`: reason-code feedback on arithmetic/modular verifier rows; replay stress rows are limitations only | the original domain grammars that produced rejected candidate rows | rejected verifier rows grouped by task, split, reason family, and seed | original rule/certificate checker that rejected or accepted each row; false-rejection-first budget filter over reason priors and residual scores | scorer: train residual ordering from rejected verifier outcomes; proposer: feed reason codes and residual classes into the next proposal pass | unguided model_score versus reason_prior and trained_residual | `crlp.rejected_candidate_learning.*` | trained residual reaches 7 accepted rules / 245 utility with zero false rejections | rejected verifier rows are reusable supervision, not discarded failed searches. | not a generic reason-code ranker; delayed-claim safety remains scoped |
| `hypothesis_library_feedback`: ARC-like hypothesis fixtures split into promotion and held-out task ids | `exact_check`: train-pair exactness and held-out fixture certification under the typed hypothesis grammar | typed templates, slot bindings, and compiled program candidates | certified hypothesis templates, slot bindings, and compiled programs | train-pair exactness, type checks, and held-out fixture split evaluation; zero-false-rejection promotion before a binding enters the library | scorer: rank bounded program candidates after template certification; proposer: turn zero-false-rejection templates into slot priors for the next search | no_prior and grammar-order tight-budget selection | `crlp.hypothesis.evidence.v1; crlp.hypothesis_library.heldout_result.v1` | slot priors raise certified@1 from 2 to 4; held-out split rows = 30 | certified hypotheses can become library priors and improve tight-budget held-out certification. | public ARC promotion remains scoped evidence, not a headline ARC score |
| `multitask_tiny_net`: shared residual rows from prime, grid, claim, and ARC surfaces | `mixed_diagnostic`: diagnostic scorer rows only; not aggregated into an authority-level leaderboard | domain-local grammars encoded through common residual feature rows | shared residual ranking rows across prime/grid/claim/ARC | task-local labels and certified downstream gates; per-task held-out metrics, OOD ARC variants, calibration bins | scorer: shared residual scorer with per-domain diagnostics; proposer: none promoted; architecture rows are blocked by the plateau audit | single-task scorer and C7 plateau audit | `crlp.multitask.summary.v1; crlp.multitask.tiny_net_plateau.v1; crlp.multitask.tiny_net_diagnostic.v1` | ARC OOD shared +5.57pp; capacity is not monotonic; 128-epoch plateau audit is underpowered | small rankers are useful around certified structure, but one flat head exposes interference. | architecture-size claims stay gated by plateau-ready diagnostics |
| `arc_agi_evidence_surface`: public and fixture ARC tasks with cell, object, transform, and hypothesis candidates | `corpus_proxy`: public-corpus coverage and failure taxonomy, not broad exact ARC competence | typed transforms, object/slot bindings, and bounded program templates | cell colors, typed transforms, programs, and hypothesis-derived candidates | train-pair consistency, unique compiled prediction, and public corpus loader; program/hypothesis certificate rows and scorer-invariant loader summaries | scorer: rank bounded candidates and expose calibration/coverage gaps; proposer: grow certified transform/program grammar coverage rather than claim scorer capacity fixes | corpus loader coverage and train-pair certified transform rows | `crlp.arc_loader.summary.v1; crlp.program_proposer.summary.v1; crlp.hypothesis.evidence.v1` | public eval coverage is 7/400 tasks and 1057 cells under the measured surface | ARC is an evidence surface for grammar coverage and failure taxonomy, not a score chase. | not a broad ARC competence or benchmark-score claim |
| `arc_verified_self_play`: generated ARC-like tasks from forward visual-primitive composition plus existing ARC visual-pipeline attempts | `exact_check`: exact synthetic supervision for inverse proposer training; support requires state-hash leakage disjointness and public ARC fixed-budget replay against no-prior baseline logs | visual primitives for object motion, anchor-relative translation, recolor, tile/scale, crop/extract, projection/panel, counting, fill, and symmetry | forward-composed visual primitive programs, verifier-shaped negative siblings, and exported compiler-backed visual priors | known-correct synthetic output construction, ARC visual-pipeline verifier fields, and full/matched visual-only public ARC replay logs; accepted rows are generated by primitive composition; public coverage requires ARC verifier replay against full and matched visual-only baseline logs | scorer: none as correctness owner; ranker learns inverse program ordering from generated verifier rows; proposer: self-play proposer trained on accepted programs and verifier-reason negative siblings, then exported as compiler-backed visual priors; synthetic top-1 is in-distribution only | full no-prior public replay, matched visual-only no-prior public replay, authored template baseline 7 certified eval tasks, oracle ceiling 14, and nine zero-coverage miss families | `crlp.arc_verified_self_play.summary.v1; crlp.arc_verified_self_play.prior.v1; crlp.arc_synthetic_shape_curriculum.v1; crlp.protocol.verified_selfplay_sample.v1; crlp.neural_grammar_discovery.arc_visual_pipeline_neural_summary.v1` | self-play is registered as an underpowered L1 search-efficiency surface until fixed-budget public replay, leakage disjointness, seed count, and zero-family opening gates pass | the durable ARC asset is the verifier/primitive system; authored solution libraries are baselines, not the training source. | distribution mismatch, leakage audit, and public fixed-budget attribution remain open before self-play can be credited |
| `signal_flight`: Signal flight replay frames, closed-loop rollout rows, live shadow-control comparisons, and live neural-active rows | `deterministic_replay`: simulator replay, rollout replay, protocol-compatible live telemetry, and guarded safety evidence | bounded action/control-option candidates beside the heuristic controller | certified control-target hypotheses plus nine WASD action candidates | offline counterfactual simulator replay, closed-loop rollout replay, live shadow frame comparison, and live active telemetry; replay utility, regret, damage/death counters, certified control-target rollout rows, shadow disagreement logs, live safety gates, and protocol compatibility rows | scorer: rank bounded WASD/control candidates beside heuristic control; proposer: control-target hypotheses remain replay-gated before action ranking | heuristic controller, simulator replay, and live guardrails | `crlp.signal_flight.summary.v1; crlp.signal_flight_rollout.summary.v1; crlp.signal_shadow_replay_audit.v1; crlp.signal_live_active_summary.v1` | offline replay, rollout bridge, and live neural-active claims supported; active baseline comparison remains open | Signal adds application pressure through replay, rollout, shadow, and guarded live artifacts while keeping active-control performance open. | active neural versus heuristic performance is not yet claimed |

## Authority Leaderboards

### `exact_check`

- `prime_gap`: learned filter 68.87% top-1 vs 45.12% hand-wheel and 9.13% raw; seed interval audited
- `grid_path`: cheap-bound gate lifts priors-enabled raw 75.80% to 90.10%
- `rejected_candidate_learning`: trained residual reaches 7 accepted rules / 245 utility with zero false rejections
- `hypothesis_library_feedback`: slot priors raise certified@1 from 2 to 4; held-out split rows = 30
- `arc_verified_self_play`: self-play is registered as an underpowered L1 search-efficiency surface until fixed-budget public replay, leakage disjointness, seed count, and zero-family opening gates pass

### `deterministic_replay`

- `delayed_claim`: clean threshold+margin gate reduces four actions to one; noisy rows expose failures
- `signal_flight`: offline replay, rollout bridge, and live neural-active claims supported; active baseline comparison remains open

### `corpus_proxy`

- `arc_agi_evidence_surface`: public eval coverage is 7/400 tasks and 1057 cells under the measured surface

### `human_review`

- No frozen rows.

### `mixed_diagnostic`

- `multitask_tiny_net`: diagnostic scorer rows only; not aggregated into an authority-level leaderboard


## Repeated Protocol Rows

### `prime_gap`

1. Domain/data: next-prime gaps over segmented-sieve windows; broader family includes period-residue sequence filters
2. Authority: `exact_check` - zero-false-rejection gates and exact residual ranking inside the next-prime support
3. Grammar: local modular filters over residues, gap classes, and wheel support
4. Candidate set: next-prime gap actions plus certified modular filters
5. Verifier/gate: segmented-sieve replay and modular certificate checks; zero-false-rejection divisibility certificates for the audited next-prime instance
6. Neural scorer role: score residual actions and certified filters after the gate defines the target
7. Neural proposer role: static proposer is a negative baseline; closed-loop and unified proposer rows must be compared to grammar order
8. Headline baseline: scorer baseline is the certified gate; proposer baseline is grammar order plus verifier-feedback variants
9. Examples: modular reject rule; wheel support certificate; next-prime gap action; Pisano-period recurrence gate
10. Result table: learned filter 68.87% top-1 vs 45.12% hand-wheel and 9.13% raw; seed interval audited
11. Takeaway: certified modular structure is the candidate mechanism; the neural scorer ranks residual budget choices while Fibonacci/Lucas-style filters remain boundary evidence.
12. Limitation: static neural proposer is not superior to deterministic baselines at fixed budget

### `grid_path`

1. Domain/data: grid mazes with deterministic shortest-path labels
2. Authority: `exact_check` - shortest-path successor and certificate claims inside generated grids
3. Grammar: local moves plus boundary, trap, path-clear, and distance predicates
4. Candidate set: four moves plus symbolic local/topology/bound predicates
5. Verifier/gate: BFS shortest-path successor and rule certificate checks; legality, trap, clear-path, and distance-dominance certificates
6. Neural scorer role: rank legal residual moves under exact shortest-path labels
7. Neural proposer role: budgeted rule-order diagnostics only
8. Headline baseline: BFS successor and cheap symbolic gates
9. Examples: legal move; trap reject; distance-dominance gate
10. Result table: cheap-bound gate lifts priors-enabled raw 75.80% to 90.10%
11. Takeaway: the same candidate-verifier contract transfers to topology once the grammar stays domain-local.
12. Limitation: supporting transfer surface, not a main contribution

### `delayed_claim`

1. Domain/data: delayed reward/claim rows with hidden outcomes and margins
2. Authority: `deterministic_replay` - simulator-certified utility and safety under the delayed-claim generator
3. Grammar: threshold, margin, and sibling-claim predicates
4. Candidate set: claim actions, threshold gates, and utility-margin rules
5. Verifier/gate: deterministic hidden-outcome simulator and rule certificate checks; finite-generator safety check plus false-rejection counters
6. Neural scorer role: rank utility after simulator safety dominates the decision
7. Neural proposer role: consume simulator/gate rejection reasons as scoped feedback
8. Headline baseline: deterministic clean threshold and margin gates
9. Examples: safe threshold claim; margin reject; unsafe reason-prior transfer
10. Result table: clean threshold+margin gate reduces four actions to one; noisy rows expose failures
11. Takeaway: utility ranking must be downstream of safety when false rejection risk is domain-specific.
12. Limitation: reason-prior transfer alone can fail safety; claim remains domain-scoped

### `rejected_candidate_learning`

1. Domain/data: verifier rejection rows grouped by seed, split, and reason family
2. Authority: `exact_check` - reason-code feedback on arithmetic/modular verifier rows; replay stress rows are limitations only
3. Grammar: the original domain grammars that produced rejected candidate rows
4. Candidate set: rejected verifier rows grouped by task, split, reason family, and seed
5. Verifier/gate: original rule/certificate checker that rejected or accepted each row; false-rejection-first budget filter over reason priors and residual scores
6. Neural scorer role: train residual ordering from rejected verifier outcomes
7. Neural proposer role: feed reason codes and residual classes into the next proposal pass
8. Headline baseline: unguided model_score versus reason_prior and trained_residual
9. Examples: unsafe candidate; no-support row; trained residual recovery
10. Result table: trained residual reaches 7 accepted rules / 245 utility with zero false rejections
11. Takeaway: rejected verifier rows are reusable supervision, not discarded failed searches.
12. Limitation: not a generic reason-code ranker; delayed-claim safety remains scoped

### `hypothesis_library_feedback`

1. Domain/data: ARC-like hypothesis fixtures split into promotion and held-out task ids
2. Authority: `exact_check` - train-pair exactness and held-out fixture certification under the typed hypothesis grammar
3. Grammar: typed templates, slot bindings, and compiled program candidates
4. Candidate set: certified hypothesis templates, slot bindings, and compiled programs
5. Verifier/gate: train-pair exactness, type checks, and held-out fixture split evaluation; zero-false-rejection promotion before a binding enters the library
6. Neural scorer role: rank bounded program candidates after template certification
7. Neural proposer role: turn zero-false-rejection templates into slot priors for the next search
8. Headline baseline: no_prior and grammar-order tight-budget selection
9. Examples: slot prior; promoted template; held-out certified@1 row
10. Result table: slot priors raise certified@1 from 2 to 4; held-out split rows = 30
11. Takeaway: certified hypotheses can become library priors and improve tight-budget held-out certification.
12. Limitation: public ARC promotion remains scoped evidence, not a headline ARC score

### `multitask_tiny_net`

1. Domain/data: shared residual rows from prime, grid, claim, and ARC surfaces
2. Authority: `mixed_diagnostic` - diagnostic scorer rows only; not aggregated into an authority-level leaderboard
3. Grammar: domain-local grammars encoded through common residual feature rows
4. Candidate set: shared residual ranking rows across prime/grid/claim/ARC
5. Verifier/gate: task-local labels and certified downstream gates; per-task held-out metrics, OOD ARC variants, calibration bins
6. Neural scorer role: shared residual scorer with per-domain diagnostics
7. Neural proposer role: none promoted; architecture rows are blocked by the plateau audit
8. Headline baseline: single-task scorer and C7 plateau audit
9. Examples: shared trunk row; ARC OOD row; capacity plateau row
10. Result table: ARC OOD shared +5.57pp; capacity is not monotonic; 128-epoch plateau audit is underpowered
11. Takeaway: small rankers are useful around certified structure, but one flat head exposes interference.
12. Limitation: architecture-size claims stay gated by plateau-ready diagnostics

### `arc_agi_evidence_surface`

1. Domain/data: public and fixture ARC tasks with cell, object, transform, and hypothesis candidates
2. Authority: `corpus_proxy` - public-corpus coverage and failure taxonomy, not broad exact ARC competence
3. Grammar: typed transforms, object/slot bindings, and bounded program templates
4. Candidate set: cell colors, typed transforms, programs, and hypothesis-derived candidates
5. Verifier/gate: train-pair consistency, unique compiled prediction, and public corpus loader; program/hypothesis certificate rows and scorer-invariant loader summaries
6. Neural scorer role: rank bounded candidates and expose calibration/coverage gaps
7. Neural proposer role: grow certified transform/program grammar coverage rather than claim scorer capacity fixes
8. Headline baseline: corpus loader coverage and train-pair certified transform rows
9. Examples: typed transform; hypothesis candidate; loader summary
10. Result table: public eval coverage is 7/400 tasks and 1057 cells under the measured surface
11. Takeaway: ARC is an evidence surface for grammar coverage and failure taxonomy, not a score chase.
12. Limitation: not a broad ARC competence or benchmark-score claim

### `arc_verified_self_play`

1. Domain/data: generated ARC-like tasks from forward visual-primitive composition plus existing ARC visual-pipeline attempts
2. Authority: `exact_check` - exact synthetic supervision for inverse proposer training; support requires state-hash leakage disjointness and public ARC fixed-budget replay against no-prior baseline logs
3. Grammar: visual primitives for object motion, anchor-relative translation, recolor, tile/scale, crop/extract, projection/panel, counting, fill, and symmetry
4. Candidate set: forward-composed visual primitive programs, verifier-shaped negative siblings, and exported compiler-backed visual priors
5. Verifier/gate: known-correct synthetic output construction, ARC visual-pipeline verifier fields, and full/matched visual-only public ARC replay logs; accepted rows are generated by primitive composition; public coverage requires ARC verifier replay against full and matched visual-only baseline logs
6. Neural scorer role: none as correctness owner; ranker learns inverse program ordering from generated verifier rows
7. Neural proposer role: self-play proposer trained on accepted programs and verifier-reason negative siblings, then exported as compiler-backed visual priors; synthetic top-1 is in-distribution only
8. Headline baseline: full no-prior public replay, matched visual-only no-prior public replay, authored template baseline 7 certified eval tasks, oracle ceiling 14, and nine zero-coverage miss families
9. Examples: verified self-play sample; forward-composed accepted program; fixed-budget public ablation
10. Result table: self-play is registered as an underpowered L1 search-efficiency surface until fixed-budget public replay, leakage disjointness, seed count, and zero-family opening gates pass
11. Takeaway: the durable ARC asset is the verifier/primitive system; authored solution libraries are baselines, not the training source.
12. Limitation: distribution mismatch, leakage audit, and public fixed-budget attribution remain open before self-play can be credited

### `signal_flight`

1. Domain/data: Signal flight replay frames, closed-loop rollout rows, live shadow-control comparisons, and live neural-active rows
2. Authority: `deterministic_replay` - simulator replay, rollout replay, protocol-compatible live telemetry, and guarded safety evidence
3. Grammar: bounded action/control-option candidates beside the heuristic controller
4. Candidate set: certified control-target hypotheses plus nine WASD action candidates
5. Verifier/gate: offline counterfactual simulator replay, closed-loop rollout replay, live shadow frame comparison, and live active telemetry; replay utility, regret, damage/death counters, certified control-target rollout rows, shadow disagreement logs, live safety gates, and protocol compatibility rows
6. Neural scorer role: rank bounded WASD/control candidates beside heuristic control
7. Neural proposer role: control-target hypotheses remain replay-gated before action ranking
8. Headline baseline: heuristic controller, simulator replay, and live guardrails
9. Examples: WASD candidate; counterfactual utility row; rollout controller row; shadow disagreement row
10. Result table: offline replay, rollout bridge, and live neural-active claims supported; active baseline comparison remains open
11. Takeaway: Signal adds application pressure through replay, rollout, shadow, and guarded live artifacts while keeping active-control performance open.
12. Limitation: active neural versus heuristic performance is not yet claimed
