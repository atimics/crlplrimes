# Frozen Experiment Matrix

Date: 2026-05-19

The paper evidence families are frozen to these rows. New domains belong in roadmap notes until this matrix is deliberately revised.

| Domain/data | Grammar | Candidate set | Verifier/gate | Scorer role | Trace artifact | Result table | Takeaway | Limitation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `prime_gap`: prime gaps over segmented sieve windows | local modular filters over residues, gap classes, and wheel support | next-gap actions plus certified modular rule filters | segmented-sieve replay and modular certificate checks; zero-false-rejection divisibility certificates over train/validation support | rank residual gaps and budgeted certified rules | `crlp.compare.v1; crlp.rule_trace.v1; crlp.rule_search.summary.v1` | learned gate 68.87% top-1 vs 45.12% hand-wheel and 9.13% raw | certified modular structure is the candidate mechanism; the neural scorer ranks residual budget choices. | static neural proposer is not superior to deterministic baselines at fixed budget |
| `grid_path`: grid mazes with deterministic shortest-path labels | local moves plus boundary, trap, path-clear, and distance predicates | four moves plus symbolic local/topology/bound predicates | BFS shortest-path successor and rule certificate checks; legality, trap, clear-path, and distance-dominance certificates | rank legal residual moves after symbolic constraints | `crlp.compare.v1; crlp.symbolic_domain.trace.v1` | cheap-bound gate lifts priors-enabled raw 75.80% to 90.10% | the same candidate-verifier contract transfers to topology once the grammar stays domain-local. | supporting transfer surface, not a main contribution |
| `delayed_claim`: delayed reward/claim rows with hidden outcomes and margins | threshold, margin, and sibling-claim predicates | claim actions, threshold gates, and utility-margin rules | deterministic hidden-outcome simulator and rule certificate checks; finite-generator safety check plus false-rejection counters | rank utility only after safety dominates the objective | `crlp.delayed_claim.rule_trace.v1; crlp.rejected_candidate_learning.*` | clean threshold+margin gate reduces four actions to one; noisy rows expose failures | utility ranking must be downstream of safety when false rejection risk is domain-specific. | reason-prior transfer alone can fail safety; claim remains domain-scoped |
| `rejected_candidate_learning`: verifier rejection rows grouped by seed, split, and reason family | the original domain grammars that produced rejected candidate rows | rejected verifier rows grouped by task, split, reason family, and seed | original rule/certificate checker that rejected or accepted each row; false-rejection-first budget filter over reason priors and residual scores | learn residual ordering from rejection reasons and marginal verifier utility | `crlp.rejected_candidate_learning.*` | trained residual reaches 7 accepted rules / 245 utility with zero false rejections | rejected verifier rows are reusable supervision, not discarded failed searches. | not a generic reason-code ranker; delayed-claim safety remains scoped |
| `hypothesis_library_feedback`: ARC-like hypothesis fixtures split into promotion and held-out task ids | typed templates, slot bindings, and compiled program candidates | certified hypothesis templates, slot bindings, and compiled programs | train-pair exactness, type checks, and held-out fixture split evaluation; zero-false-rejection promotion before a binding enters the library | rank promoted slot priors and residual program candidates under tight budgets | `crlp.hypothesis.evidence.v1; crlp.hypothesis_library.heldout_result.v1` | slot priors raise certified@1 from 2 to 4; held-out split rows = 30 | certified hypotheses can become library priors and improve tight-budget held-out certification. | public ARC promotion remains scoped evidence, not a headline ARC score |
| `multitask_tiny_net`: shared residual rows from prime, grid, claim, and ARC surfaces | domain-local grammars encoded through common residual feature rows | shared residual ranking rows across prime/grid/claim/ARC | task-local labels and certified downstream gates; per-task held-out metrics, OOD ARC variants, calibration bins | shared residual scorer with domain-specific diagnostics | `crlp.multitask.summary.v1; crlp.multitask.tiny_net_plateau.v1` | ARC OOD shared +5.57pp; capacity is not monotonic; plateau audit is underpowered | small rankers are useful around certified structure, but one flat head exposes interference. | architecture-size claims stay gated by the plateau audit |
| `arc_agi_evidence_surface`: public and fixture ARC tasks with cell, object, transform, and hypothesis candidates | typed transforms, object/slot bindings, and bounded program templates | cell colors, typed transforms, programs, and hypothesis-derived candidates | train-pair consistency, unique compiled prediction, and public corpus loader; program/hypothesis certificate rows and scorer-invariant loader summaries | rank bounded candidates; scorer does not own correctness | `crlp.arc_loader.summary.v1; crlp.program_proposer.summary.v1; crlp.hypothesis.evidence.v1` | public eval coverage is 7/400 tasks and 1057 cells under the measured surface | ARC is an evidence surface for grammar coverage and failure taxonomy, not a score chase. | not a broad ARC competence or benchmark-score claim |
| `signal_flight`: Signal flight replay frames, closed-loop rollout rows, live shadow-control comparisons, and live neural-active rows | bounded action/control-option candidates beside the heuristic controller | certified control-target hypotheses plus nine WASD action candidates | offline counterfactual simulator replay, closed-loop rollout replay, live shadow frame comparison, and live active telemetry; replay utility, regret, damage/death counters, certified control-target rollout rows, shadow disagreement logs, live safety gates, and protocol compatibility rows | rank bounded control candidates beside heuristic control | `crlp.signal_flight.summary.v1; crlp.signal_flight_rollout.summary.v1; crlp.signal_shadow_replay_audit.v1; crlp.signal_live_active_summary.v1` | offline replay, rollout bridge, and live neural-active claims supported; active baseline comparison remains open | Signal adds application pressure through replay, rollout, shadow, and guarded live artifacts while keeping active-control performance open. | active neural versus heuristic performance is not yet claimed |

## Repeated Protocol Rows

### `prime_gap`

1. Domain/data: prime gaps over segmented sieve windows
2. Grammar: local modular filters over residues, gap classes, and wheel support
3. Candidate set: next-gap actions plus certified modular rule filters
4. Verifier/gate: segmented-sieve replay and modular certificate checks; zero-false-rejection divisibility certificates over train/validation support
5. Examples: modular reject rule; wheel support certificate; residual gap action
6. Result table: learned gate 68.87% top-1 vs 45.12% hand-wheel and 9.13% raw
7. Takeaway: certified modular structure is the candidate mechanism; the neural scorer ranks residual budget choices.
8. Limitation: static neural proposer is not superior to deterministic baselines at fixed budget

### `grid_path`

1. Domain/data: grid mazes with deterministic shortest-path labels
2. Grammar: local moves plus boundary, trap, path-clear, and distance predicates
3. Candidate set: four moves plus symbolic local/topology/bound predicates
4. Verifier/gate: BFS shortest-path successor and rule certificate checks; legality, trap, clear-path, and distance-dominance certificates
5. Examples: legal move; trap reject; distance-dominance gate
6. Result table: cheap-bound gate lifts priors-enabled raw 75.80% to 90.10%
7. Takeaway: the same candidate-verifier contract transfers to topology once the grammar stays domain-local.
8. Limitation: supporting transfer surface, not a main contribution

### `delayed_claim`

1. Domain/data: delayed reward/claim rows with hidden outcomes and margins
2. Grammar: threshold, margin, and sibling-claim predicates
3. Candidate set: claim actions, threshold gates, and utility-margin rules
4. Verifier/gate: deterministic hidden-outcome simulator and rule certificate checks; finite-generator safety check plus false-rejection counters
5. Examples: safe threshold claim; margin reject; unsafe reason-prior transfer
6. Result table: clean threshold+margin gate reduces four actions to one; noisy rows expose failures
7. Takeaway: utility ranking must be downstream of safety when false rejection risk is domain-specific.
8. Limitation: reason-prior transfer alone can fail safety; claim remains domain-scoped

### `rejected_candidate_learning`

1. Domain/data: verifier rejection rows grouped by seed, split, and reason family
2. Grammar: the original domain grammars that produced rejected candidate rows
3. Candidate set: rejected verifier rows grouped by task, split, reason family, and seed
4. Verifier/gate: original rule/certificate checker that rejected or accepted each row; false-rejection-first budget filter over reason priors and residual scores
5. Examples: unsafe candidate; no-support row; trained residual recovery
6. Result table: trained residual reaches 7 accepted rules / 245 utility with zero false rejections
7. Takeaway: rejected verifier rows are reusable supervision, not discarded failed searches.
8. Limitation: not a generic reason-code ranker; delayed-claim safety remains scoped

### `hypothesis_library_feedback`

1. Domain/data: ARC-like hypothesis fixtures split into promotion and held-out task ids
2. Grammar: typed templates, slot bindings, and compiled program candidates
3. Candidate set: certified hypothesis templates, slot bindings, and compiled programs
4. Verifier/gate: train-pair exactness, type checks, and held-out fixture split evaluation; zero-false-rejection promotion before a binding enters the library
5. Examples: slot prior; promoted template; held-out certified@1 row
6. Result table: slot priors raise certified@1 from 2 to 4; held-out split rows = 30
7. Takeaway: certified hypotheses can become library priors and improve tight-budget held-out certification.
8. Limitation: public ARC promotion remains scoped evidence, not a headline ARC score

### `multitask_tiny_net`

1. Domain/data: shared residual rows from prime, grid, claim, and ARC surfaces
2. Grammar: domain-local grammars encoded through common residual feature rows
3. Candidate set: shared residual ranking rows across prime/grid/claim/ARC
4. Verifier/gate: task-local labels and certified downstream gates; per-task held-out metrics, OOD ARC variants, calibration bins
5. Examples: shared trunk row; ARC OOD row; capacity plateau row
6. Result table: ARC OOD shared +5.57pp; capacity is not monotonic; plateau audit is underpowered
7. Takeaway: small rankers are useful around certified structure, but one flat head exposes interference.
8. Limitation: architecture-size claims stay gated by the plateau audit

### `arc_agi_evidence_surface`

1. Domain/data: public and fixture ARC tasks with cell, object, transform, and hypothesis candidates
2. Grammar: typed transforms, object/slot bindings, and bounded program templates
3. Candidate set: cell colors, typed transforms, programs, and hypothesis-derived candidates
4. Verifier/gate: train-pair consistency, unique compiled prediction, and public corpus loader; program/hypothesis certificate rows and scorer-invariant loader summaries
5. Examples: typed transform; hypothesis candidate; loader summary
6. Result table: public eval coverage is 7/400 tasks and 1057 cells under the measured surface
7. Takeaway: ARC is an evidence surface for grammar coverage and failure taxonomy, not a score chase.
8. Limitation: not a broad ARC competence or benchmark-score claim

### `signal_flight`

1. Domain/data: Signal flight replay frames, closed-loop rollout rows, live shadow-control comparisons, and live neural-active rows
2. Grammar: bounded action/control-option candidates beside the heuristic controller
3. Candidate set: certified control-target hypotheses plus nine WASD action candidates
4. Verifier/gate: offline counterfactual simulator replay, closed-loop rollout replay, live shadow frame comparison, and live active telemetry; replay utility, regret, damage/death counters, certified control-target rollout rows, shadow disagreement logs, live safety gates, and protocol compatibility rows
5. Examples: WASD candidate; counterfactual utility row; rollout controller row; shadow disagreement row
6. Result table: offline replay, rollout bridge, and live neural-active claims supported; active baseline comparison remains open
7. Takeaway: Signal adds application pressure through replay, rollout, shadow, and guarded live artifacts while keeping active-control performance open.
8. Limitation: active neural versus heuristic performance is not yet claimed
