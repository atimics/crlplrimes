# Experiment Proposal: Signal Full-Spectrum Control Stack

Status: implemented as an offline replay ranker, closed-loop rollout evaluator,
and live websocket shadow pilot. `examples/signal_flight.c` trains/saves the
checkpoint; `examples/signal_flight_rollout.c` runs deterministic Signal
rollouts against baselines; `scripts/signal_live_brain.mjs` loads that
checkpoint, defaults to `shadow_neural`, and only lets neural choices drive the
ship under explicit `--brain-mode neural_active`.

## Purpose

Signal is the real-time application test for the verifier-grounded tiny-net
protocol. The core question is whether the same pattern used in the symbolic
experiments works when the environment is an embodied simulator:

```text
state -> enumerate legal candidates -> replay/simulate candidates
      -> train a tiny ranker on certified utility rows
      -> run live in shadow mode
      -> activate only when safety and regret gates pass
```

The experiment should not claim that a tiny network "solves robotics" or
replaces a controller. The claim is narrower and stronger: a very small neural
ranker can improve real-time action selection when it ranks a bounded candidate
set whose outcomes are certified by authoritative replay or simulation.

## Current Supported Result

The first pass is offline replay training over Signal flight controls.

| Field | Current value |
| --- | --- |
| Candidate set | 9 discrete WASD actions: `NONE`, `W`, `A`, `D`, `S`, `WA`, `WD`, `SA`, `SD` |
| Feature row | 48 state/action observations |
| Model size | 2113 parameters |
| Forward cost | 4177 FLOPs per candidate row |
| Train rows | 2016 |
| Test rows | 756 |
| Test groups | 84 counterfactual utility groups |
| Trace schema | `signal.flight_counterfactual.v1` |
| Summary schema | `crlp.signal_flight.summary.v1` |
| Checkpoint | `build/float/signal_flight_experiment/signal_flight.nnckpt` |

Current held-out replay summary:

| Metric | Neural ranker | No-safety baseline | Interpretation |
| --- | ---: | ---: | --- |
| Top-1 | 79.76% | 67.86% | Neural ranker is materially better at picking the replay utility target. |
| Top-3 | 94.05% | n/a | The target is usually near the top of the neural ordering. |
| Selected utility | 0.14317 | 0.13593 | Neural ranker improves realized replay utility. |
| Regret | 0.00735 | 0.01459 | Neural regret is about half the baseline regret. |
| Damage rate | 2.38% | 2.38% | No damage improvement yet. |
| Death rate | 0.00% | 0.00% | No held-out deaths in either row. |
| ECE | 0.31608 | n/a | Calibration is weak and should not drive safety. |

Supported wording:

> On held-out replay-certified Signal flight groups, a 2113-parameter WASD
> ranker improves utility and roughly halves regret relative to the no-safety
> goal-heading heuristic, with equal held-out damage rate and zero deaths in
> the measured trace.

Current closed-loop rollout smoke:

```sh
make signal-flight-rollout-eval SIGNAL_FLIGHT_ROLLOUT_CHECKPOINT=build/float/signal_flight_wall_probe/signal_flight_epochs60.nnckpt
```

The first 24-episode rollout emits `crlp.signal_flight_rollout.summary.v1`.
`neural_guarded` had the best average utility in this smoke
(`0.35635` vs `0.16049` for `flight_steer_to`) with zero deaths and the same
raw damage-event count as the deterministic steering helper, but its damage
episode rate remains slightly worse. Both neural controllers now choose among
certified control-target hypotheses before ranking WASD actions: direct goal,
Signal nav path, left bypass, right bypass, and brake/escape. The guarded run
selected a certified hypothesis on about 40.9% of neural ticks, which makes the
next gap explicit: fallback nav hypotheses still carry too much of the workload.
This is not a deployment gate pass; it is the first closed-loop evidence that
more training should target wall and approach failures.

Unsupported wording:

> The live websocket brain has passed active neural-control gates.

The live brain now logs heuristic and neural decisions on the same frame, but
active neural flight remains a Tier-2 result that requires replay follow-up.

## Protocol Shape

| Field | Instantiation |
| --- | --- |
| Candidate set | Discrete control candidates or higher-level Signal action candidates |
| Verifier | Signal authoritative simulator replay, live shadow replay, or deterministic build/resource legality checks |
| Certificate/gate | Replay utility, damage/death counters, legality checks, resource feasibility, collision/safety predicates |
| Scorer role | Rank candidates that pass the hard gate; never own correctness or safety |
| Trace artifact | `signal.flight_counterfactual.v1`, `crlp.signal_flight.summary.v1`, `crlp.signal_flight_rollout.summary.v1`, `crlp.signal_live_brain.v1`, future `crlp.signal_plan_trace.v1` |
| Protocol point | Real-time control can use tiny neural rankers when candidate enumeration and safety evidence remain outside the model |

## Experimental Tiers

### Tier 0: Offline Replay Ranker

This tier is already partially complete.

Run:

```sh
make signal-flight
```

Data path:

```text
../signal/build/flight_trace
  -> build/float/signal_flight_train.csv
  -> build/float/signal_flight_test.csv
  -> build/float/signal_flight_experiment/signal_flight.nnckpt
  -> build/float/signal_flight_experiment/summary.jsonl
```

Decision rule:

- Test top-1 beats the no-safety baseline.
- Neural selected utility beats the no-safety baseline.
- Neural regret is lower than baseline regret.
- Damage and death rates do not worsen.
- Candidate-mask sanity rates are 100%.

Current status:

- Ranking and regret pass.
- Death rate passes.
- Damage rate ties, so safety improvement is not yet supported.
- Calibration does not pass a safety-use bar.

### Tier 1: Live Shadow Inference

This tier wires checkpoint inference into `scripts/signal_live_brain.mjs` without
letting the network control the ship unless `--brain-mode neural_active` is
provided.

Live behavior:

```text
heuristic action drives ship
neural checkpoint ranks the same candidate set
log heuristic action, neural top-k, scores, features, checkpoint hash
compare the two in live telemetry and follow-up replay
```

Live fields:

- `brain_mode`: `heuristic`, `shadow_neural`, or `neural_active`
- `checkpoint_path`
- `checkpoint_hash`
- `feature_set`
- `candidate_scores`
- `candidate_probabilities`
- `neural_action`
- `heuristic_action`
- `active_action`
- `safety_gate`
- `safety_reject_reasons`
- `inference_ms`

Run:

```sh
make signal-live-brain SIGNAL_LIVE_BRAIN_CHECKPOINT=build/float/signal_flight_experiment/signal_flight.nnckpt
make signal-live-status
make signal-shadow-replay-audit
```

`signal_live_status.mjs` reports active source, neural shadow disagreement rate,
and inference latency percentiles over the rolling window. The shadow replay
audit emits `crlp.signal_shadow_frame.v1` and
`crlp.signal_shadow_replay_audit.v1` rows. It is intentionally conservative:
until a deterministic replay pass consumes the frame rows, neural
counterfactual outcomes are marked unavailable rather than inferred from the
heuristic-executed trajectory.

Decision rule:

- Live shadow inference stays below the tick budget.
- Feature encoding matches the offline checkpoint feature contract.
- Shadow neural regret is lower than heuristic regret under replay follow-up.
- Shadow neural choices do not increase damage/death under replay follow-up.

### Tier 2: Active Neural Flight With Safety Gate

This tier allows the neural ranker to drive only after Tier 1 passes.

Before live activation, run the deterministic closed-loop gate:

```sh
make signal-flight-rollout-eval SIGNAL_FLIGHT_ROLLOUT_CHECKPOINT=build/float/signal_flight_experiment/signal_flight.nnckpt
```

The rollout binary evaluates `goal_heading_no_safety`, `flight_steer_to`,
`neural_raw`, and `neural_guarded` on identical deterministic episodes. It is
the current bridge between one-step replay ranking and live multiplayer control.

The active controller should not be:

```text
neural top-1 -> send input
```

It should be:

```text
enumerate/certify control-target hypotheses
-> select a certified waypoint/path/escape hypothesis
-> enumerate WASD candidates around that hypothesis
-> apply hard action mask
-> neural rank surviving WASD candidates
-> optionally fall back to heuristic if confidence/safety is weak
-> send active action
-> log outcome
```

Hard safety mask candidates:

- Death predicted by replay.
- Hull-loss above threshold.
- Collision within braking distance.
- Clearance below hull/speed-dependent margin.
- Candidate not in the legal input mask.
- Candidate produces no progress when a safe progress action exists.

Decision rule:

- Active neural utility beats heuristic utility.
- Active neural regret beats heuristic regret.
- Damage rate improves or stays within a predeclared non-inferiority margin.
- Death rate remains zero over the evaluation window.
- Emergency fallback rate is reported, not hidden.

### Tier 3: Tactical Option Ranking

Once flight control is stable, move one level up from keypresses to tactical
options. The action set is no longer raw WASD; it is a bounded menu of options
that each compile down to lower-level flight or interaction candidates.

Candidate families:

- pursue target
- retreat to safe range
- intercept hostile
- escort ally
- dock at station
- repair
- mine asteroid
- collect cargo
- deliver cargo
- patrol route

Verifier/gate:

- path feasibility
- fuel or energy feasibility if modeled
- damage/death estimate
- docking legality
- target reachability
- expected time-to-complete
- mission utility

Scorer role:

- Rank tactical options by simulator-certified utility.
- Defer to hard legality and safety gates.
- Emit trace rows suitable for replay and later library promotion.

### Tier 4: Construction And Station Planning

Station construction should be treated as planning over certified candidates,
not as a giant neural output.

Bad formulation:

```text
state -> neural policy -> BUILD_STATION_PLAN
```

Good formulation:

```text
state
-> enumerate buildable modules, slots, links, and next-step sequences
-> reject illegal or unsafe construction candidates
-> tiny ranker orders feasible next build steps
-> simulator records throughput, cost, survival, and regret
-> successful motifs become reusable library priors
```

Candidate set:

- station site candidates
- module type candidates
- module placement slots
- connection/link candidates
- build-order candidates
- resource-route candidates
- defense-coverage candidates
- repair/upgrade candidates

Verifier/gate:

- legal placement
- no overlap with existing geometry
- graph connectivity
- docking/clearance constraints
- power/resource feasibility
- construction cost and inventory availability
- line-of-fire or defensive coverage constraints
- survivability under simulated raids or asteroid fields
- no dead-end logistics states

Scorer role:

- Rank feasible build steps or small build templates.
- Rank promoted construction motifs.
- Trade off utility, cost, time-to-online, defense, and logistics.
- Never override legality, resource, or safety gates.

Trace artifacts:

- `crlp.signal_construction_candidate.v1`
- `crlp.signal_construction_certificate.v1`
- `crlp.signal_construction_summary.v1`
- `crlp.signal_construction_library.v1`

Metrics:

- build completion rate
- time to first productive module
- resource waste
- throughput after N ticks
- defensive coverage
- repair load
- survival rate
- regret versus oracle or strong heuristic
- invalid-action rejection rate

Library promotion:

Repeated successful build motifs should become explicit library entries:

- "starter mining outpost"
- "defensive ring"
- "dock-first trading station"
- "power-before-production"
- "repair-buffered expansion"
- "resource-shortage recovery"

The promotion rule should mirror the certified-hypothesis work:

- support across multiple seeds or maps
- zero legality regressions
- non-negative held-out utility
- stable template id
- explicit invariant and variable slots

## Architecture

### Current Flight Ranker

The current Signal flight ranker is intentionally tiny:

```text
48-feature state/action row
-> dense tiny MLP
-> scalar score per candidate action
-> softmax over 9 candidates
```

This is appropriate for reactive flight because inference must fit inside a
real-time tick. The current model size and FLOP count are small enough for
high-frequency control.

### Proposed Full Stack

Use a hierarchy of small rankers instead of one large policy:

| Layer | Candidate type | Ranker | Gate |
| --- | --- | --- | --- |
| Reactive | WASD controls | flight ranker | collision/death/hull replay mask |
| Tactical | short-horizon options | tactical option ranker | feasibility and safety simulator |
| Operational | route/resource assignments | logistics ranker | resource and path feasibility |
| Construction | modules, slots, build steps | construction ranker | legality, connectivity, resources |
| Library | promoted motifs | motif prior/ranker | held-out certification |

The layers should communicate through explicit candidates and trace rows, not
through hidden recurrent state.

### Why Not One Larger Net

The recent architecture results argue against scaling a flat MLP first:

- Wider multitask trunks did not monotonically improve fixed ARC.
- ARC failures came from representation/formulation, not raw capacity.
- Reason-code learning succeeded only when safety and reason structure matched
  the domain.
- Certified-library priors improved tight-budget ranking without requiring a
  larger network.

Signal should follow the same lesson. If construction or tactics fail, first
improve candidate factoring, verifier coverage, and trace-derived priors before
making the model larger.

## Training Data Plan

### Offline Counterfactual Rows

For each sampled state:

1. Enumerate candidates.
2. Replay each candidate for a fixed horizon.
3. Record progress, utility, damage, hull loss, death, and mask status.
4. Group rows by state.
5. Train ranker to choose the best certified utility row under safety constraints.

### Pain-Biased Rows

The current `make signal-flight-pain` path should be used to oversample difficult
states:

- asteroid-contact states
- station-wall approach and escape states
- low hull
- high speed near obstacles
- docking approach failures
- braking-distance failures
- target overshoot

These rows are needed because average flight traces may be too easy to teach
safety.

### Shadow-Live Rows

Live shadow mode should append rows where:

- heuristic drove the ship
- neural ranker produced top-k alternatives
- later replay or delayed outcome assigns certified utility

This turns deployment into data collection without risking immediate neural
control.

### Construction Rows

Construction rows should be generated from scenario seeds:

- empty map startup
- constrained resource startup
- hostile nearby
- dense asteroid field
- damaged station recovery
- expansion under traffic

Each scenario should produce candidate build steps plus short and long horizon
outcomes.

## Baselines

Flight baselines:

- no-safety goal-heading heuristic
- current live heuristic
- random legal action
- teacher trace action if present
- oracle over replay-certified utility
- neural ranker without safety mask
- neural ranker with safety mask

Tactical baselines:

- nearest-goal heuristic
- fixed priority list
- shortest-time option
- safest option
- oracle over replay groups

Construction baselines:

- fixed build order
- greedy resource throughput
- greedy defense coverage
- nearest legal slot
- random legal build
- oracle over bounded candidate set

## Metrics

Flight:

- top-1 and top-3 against certified utility target
- selected utility
- regret versus oracle
- damage rate
- death rate
- hull loss
- progress per tick
- action entropy
- emergency fallback rate
- inference latency
- live disconnect/reconnect count

Tactics:

- mission completion
- time to completion
- regret versus oracle
- damage/death
- target survival
- resource collected/delivered

Construction:

- invalid candidate rejection rate
- construction completion
- time to first productive module
- station throughput
- power/resource balance
- repair burden
- survival under threat
- held-out motif utility

Calibration:

- ECE by action family
- reliability bins for high-confidence actions
- confidence under safety rejection
- temperature fit on replay validation rows

## Claim Mapping

| Claim | Signal evidence needed |
| --- | --- |
| C5 substrate portability | Signal emits replay-certified rows under the same candidate/verifier/scorer/trace discipline. |
| C7 tiny-net capacity | The flight ranker remains small and is evaluated against latency, regret, and safety metrics. |
| Application vignette | Offline replay ranker improves held-out utility/regret, then shadow-live rows show real-time viability. |
| Full-spectrum control | Tactical and construction layers use the same ranked-candidate protocol with domain-specific gates. |

The first paper-safe claim should be:

> Signal demonstrates that verifier-grounded tiny rankers can improve
> held-out replay-certified real-time control decisions under a low-latency
> action-ranking interface.

The later full-stack claim should require:

> The same protocol extends from reactive control to tactical and construction
> actions through hierarchical candidate sets, simulator/legal gates, and
> promoted action libraries.

## Implementation Roadmap

### R-SIG.1: Offline Replay Ranker

Status: prototype complete.

Needed hardening:

- repeat across at least 3 train/test seed pairs
- include pain-biased traces
- report confidence intervals
- include calibration bins
- compare active safety mask versus no safety mask

### R-SIG.2: Live Shadow Inference

Owner: live brain integration.

Work:

- load `signal_flight.nnckpt` in the live websocket script or a helper process
- implement feature encoding parity with `signal-flight-live-v2`
- log neural top-k and checkpoint hash
- keep heuristic active action
- replay shadow choices after the run

Pass condition:

- shadow neural regret lower than heuristic regret
- no damage/death regression under replay
- p95 inference latency below the tick budget

### R-SIG.3: Active Neural Flight

Owner: live brain integration plus safety evaluator.

Work:

- implement hard safety mask
- add fallback policy
- activate neural top candidate only after safety mask
- log fallback and reject reasons
- run bounded live sessions

Pass condition:

- active neural utility beats heuristic
- regret improves
- death rate remains zero
- damage rate improves or is non-inferior

### R-SIG.4: Tactical Option Ranker

Owner: Signal experiment agent.

Work:

- define tactical option candidate schema
- generate replay-certified tactical traces
- train tiny option ranker
- compare against priority and nearest-goal heuristics

Pass condition:

- tactical ranker improves mission utility/regret without safety regression

### R-SIG.5: Construction Candidate Ranker

Owner: construction experiment agent.

Work:

- define station construction candidate schema
- implement legality/resource/connectivity certificates
- generate scenario traces
- train ranker over build steps
- add construction motif promotion

Pass condition:

- ranker improves throughput, survival, or regret versus fixed build-order
  baselines on held-out scenarios
- zero illegal builds after gate
- promoted motifs improve held-out build ranking

## Research-Agent Briefs

### Flight Agent

Focus only on WASD flight.

Do not add construction or tactics. Produce:

- seed sweep
- pain-probe sweep
- safety-mask ablation
- latency report
- shadow-mode integration checklist

### Live Agent

Focus only on websocket parity and live logging.

Do not change offline training. Produce:

- checkpoint loading path
- feature parity test
- shadow log schema
- p50/p95/p99 inference latency
- replay follow-up for shadow choices

### Tactical Agent

Focus on short-horizon option candidates.

Do not use raw WASD as the target. Produce:

- tactical candidate schema
- simulator replay harness
- baseline comparison
- regret/safety summary

### Construction Agent

Focus on build candidates and certified legality.

Do not start with a giant policy network. Produce:

- candidate schema for module, slot, link, and build step
- resource/connectivity/safety certificate rows
- fixed build-order baselines
- held-out scenario split
- motif promotion proposal

## Risks And Controls

| Risk | Control |
| --- | --- |
| Live model acts on mismatched features | Feature-set/version check and shadow mode before activation |
| Neural ranker improves utility but worsens damage | Hard safety mask and damage/death non-inferiority gate |
| Offline replay does not match live dynamics | Shadow-live replay follow-up and live trace ingestion |
| Action-space explosion for construction | Hierarchical candidates and template/motif ranking |
| Calibration is weak | Do not use confidence as safety; use verifier masks and report reliability |
| Heuristic remains hidden in active path | Log active mode, fallback reason, and active action source |
| Construction motifs overfit scenarios | Held-out map/seed split and zero-illegal-build requirement |

## Paper Use

Signal should be an application vignette until live neural inference and shadow
replay are complete. It can already support the claim that the protocol applies
outside symbolic paper tasks:

- candidate enumeration is real-time WASD control
- verification is authoritative simulator replay
- the neural component is a tiny ranker
- evaluation reports regret, utility, damage, and death
- live deployment is staged through shadow logging and safety gates

Once R-SIG.2 and R-SIG.3 pass, Signal can become a stronger empirical section.
Once R-SIG.5 exists, station construction can become the long-horizon evidence
that the same protocol scales from motor control to planning without changing
the core claim: neural rankers order certified candidates; verifiers, simulators,
and trace artifacts own correctness and safety.
