# Experiment Matrix

## arithmetic sequence filtering (prime_gap)

- **Grounding:** exact_check — segmented-sieve replay and modular certificate checks
- **Claim:** certified modular filters shrink the residual action space
- **Result:** learned filter 68.87% top-1 vs 45.12% hand-wheel and 9.13% raw; seed interval audited
- **Limitation:** static neural proposer is not superior to deterministic baselines at fixed budget
- **Takeaway:** certified modular structure is the candidate mechanism; the neural scorer ranks residual budget choices while Fibonacci/Lucas-style filters remain boundary evidence.

## grid path (grid_path)

- **Grounding:** exact_check — BFS shortest-path successor and rule certificate checks
- **Claim:** same gate contract works outside residue arithmetic
- **Result:** cheap-bound gate lifts priors-enabled raw 75.80% to 90.10%
- **Limitation:** supporting transfer surface, not a main contribution
- **Takeaway:** the same candidate-verifier contract transfers to topology once the grammar stays domain-local.

## delayed claim (delayed_claim)

- **Grounding:** deterministic_replay — deterministic hidden-outcome simulator and rule certificate checks
- **Claim:** reason learning needs domain-aware safety constraints
- **Result:** clean threshold+margin gate reduces four actions to one; noisy rows expose failures
- **Limitation:** reason-prior transfer alone can fail safety; claim remains domain-scoped
- **Takeaway:** utility ranking must be downstream of safety when false rejection risk is domain-specific.

## rejected candidate learning (rejected_candidate_learning)

- **Grounding:** exact_check — original rule/certificate checker that rejected or accepted each row
- **Claim:** verifier failures become training signal
- **Result:** trained residual reaches 7 accepted rules / 245 utility with zero false rejections
- **Limitation:** not a generic reason-code ranker; delayed-claim safety remains scoped
- **Takeaway:** rejected verifier rows are reusable supervision, not discarded failed searches.

## hypothesis library feedback (hypothesis_library_feedback)

- **Grounding:** exact_check — train-pair exactness, type checks, and held-out fixture split evaluation
- **Claim:** certified learning can accumulate through reusable library priors
- **Result:** slot priors raise certified@1 from 2 to 4; held-out split rows = 30
- **Limitation:** public ARC promotion remains scoped evidence, not a headline ARC score
- **Takeaway:** certified hypotheses can become library priors and improve tight-budget held-out certification.

## multitask tiny net (multitask_tiny_net)

- **Grounding:** mixed_diagnostic — task-local labels and certified downstream gates
- **Claim:** representation/protocol limits dominate naive width scaling
- **Result:** Go shared +53.3pp, ARC OOD +5.57pp; capacity is not monotonic; 128-epoch plateau audit is underpowered
- **Limitation:** architecture-size claims stay gated by plateau-ready diagnostics
- **Takeaway:** small rankers are useful around certified structure, but one flat head exposes interference.

## arc agi evidence surface (arc_agi_evidence_surface)

- **Grounding:** corpus_proxy — train-pair consistency, unique compiled prediction, and public corpus loader
- **Claim:** ARC exposes priors, grammar coverage, and failure taxonomy
- **Result:** public eval coverage is 7/400 tasks and 1057 cells under the measured surface
- **Limitation:** not a broad ARC competence or benchmark-score claim
- **Takeaway:** ARC is an evidence surface for grammar coverage and failure taxonomy, not a score chase.

## arc verified self play (arc_verified_self_play)

- **Grounding:** exact_check — known-correct synthetic output construction, ARC visual-pipeline verifier fields, and full/matched visual-only public ARC replay logs
- **Claim:** ARC proposer training can be generated from verifier-owned primitives, but support requires fixed-budget public replay rather than synthetic in-distribution top-1
- **Result:** self-play is registered as an underpowered L1 search-efficiency surface until fixed-budget public replay, leakage disjointness, seed count, and zero-family opening gates pass
- **Limitation:** distribution mismatch, leakage audit, and public fixed-budget attribution remain open before self-play can be credited
- **Takeaway:** the durable ARC asset is the verifier/primitive system; authored solution libraries are baselines, not the training source.

## go 9×9 (go_position)

- **Grounding:** deterministic_replay — terminal territory scoring after deterministic random-rollout replay
- **Claim:** deterministic-replay Go action ranking transfers positively from other domains under shared scoring
- **Result:** shared scorer Go top-1 73.3% vs single-task 20.0% (+53.3pp); tournament warm-start from cross-domain checkpoint operational
- **Limitation:** Go top-1 is measured on rollout-grounded labels, not competitive play strength; Signal mining integration uses synthetic data
- **Takeaway:** Go adds board-game application pressure and exhibits the strongest cross-domain transfer signal in the matrix.

## signal flight (signal_flight)

- **Grounding:** deterministic_replay — offline counterfactual simulator replay, closed-loop rollout replay, live shadow frame comparison, and live active telemetry
- **Claim:** application pressure for replay-grounded residual ranking and guarded live neural-active execution
- **Result:** in offline Signal replay, the model ranks recorded WASD choices at top-1 79.76% vs baseline 67.86% (+11.90pp), utility 0.1432 vs 0.1359, regret 0.0073 vs 0.0146, damage 2.38% vs 2.38%, deaths 0.00% vs 0.00%; live shadow frames=74. the live Signal pilot did run checkpoint-ranked neural actions, but only behind guards: 74 recent neural-source rows from checkpoint f12b644ed9, p95 inference 0.089 ms, 74 replay-ready rows, protocol-compatible rows=74, safety-gate rows=74, damage events=2, deaths=0; no baseline comparison is claimed.
- **Limitation:** active neural versus heuristic performance is not yet claimed
- **Takeaway:** Signal adds application pressure through replay, rollout, shadow, and guarded live artifacts while keeping active-control performance open.

