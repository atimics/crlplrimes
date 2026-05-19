# Experiment Proposal: Vector 4X Multiplayer with Certified Tiny Pilots

Status: proposal. Builds on `examples/signal_flight.c`,
`examples/signal_flight_rollout.c`, the certified hypothesis library
(`scripts/certified_hypothesis_library.py`,
`docs/experiment-certified-hypothesis-library.md`), and the live
shadow-to-active pilot stack (`scripts/signal_live_brain.mjs`,
`scripts/signal_shadow_replay_audit.py`). No game code yet.

## Purpose

Test whether the verifier-grounded tiny-net protocol scales from one Signal
flight pilot to a populated multiplayer 4X game where every ship, fleet, and
faction agent is a separately certified ranker. The game is the proving
ground; the contribution is evidence that the protocol holds at scale
(hundreds of agents per frame) and across the strategy-level decisions
(build, expand, fight, negotiate) that 4X demands.

Negative claims:

- Not a claim that tiny nets beat LLM agents at strategic play.
- Not a claim that the game ships as a product; it ships as evidence.
- Not a replacement for designed systems. The candidate grammar is
  authored; the ranker only chooses.

## Game shape

| Field | Value |
| --- | --- |
| Genre | 2D vector graphics 4X with asteroids-style kinematics |
| Players | Multiplayer (lockstep peer-to-peer for the first build) |
| Unit count | Target ~100 controllable ships per match, ~10 fleets per faction |
| Map | Procedural sector graph with asteroid fields, jump lanes, stations |
| Win condition | TBD; likely victory-point + elimination hybrid |
| Tick model | Deterministic fixed-tick simulator; client renders interpolated |
| Replay | Full match trace under `crlp.vector_4x.trace.v1` |

The game is not the contribution. The contribution is that every NPC unit,
every fleet AI, and every faction agent in the game is a tiny certified
ranker the repo's existing protocol can train, verify, and ship.

## Mapping repo primitives to game systems

The generalized experiment schema in `include/nn_experiment.h` ports
directly. No new abstractions are required for M0-M2.

| Repo primitive | Game role |
| --- | --- |
| Candidate enumerator `A_T(s)` | Legal orders per unit per tick |
| Exact verifier `V_T(s, a)` | Authoritative game simulator: collision, fuel, line of sight, resources |
| Rule grammar `G_T` | Designer-authored tactical hypotheses: intercept, evade, screen, mine, jump |
| Certificate checker `C_T` | Proves a tactic cannot violate game invariants (no clip, no resource cheat, no illegal order) |
| Tiny scorer `f_theta` | Per-agent ranker, 2K-50K params, one per role |
| Trace protocol | Multiplayer determinism and match replay are the same artifact |
| Shadow-to-active gate | Ship new pilot checkpoints behind a flag, escalate per role |

The design invariant ports verbatim:

```text
The model proposes. The verifier decides. The trace remembers.
```

In a 4X this reads: the player and the AI both submit candidate orders; the
simulator decides what actually happens; the trace is the match record and
the training corpus.

## Agent taxonomy

Three certified ranker tiers, each a separate schema instance under
`crlp.experiment_schema.v1`. All three share the same certificate gate,
trace protocol, and shadow-to-active deployment path.

1. **Pilot** — direct control of one ship. Direct port of the existing
   `signal_flight` ranker. ~2K params. Inference budget under 0.1ms per
   candidate row. Leakage policy: `forbidden`.
2. **Fleet** — orders 2-12 ships toward a tactical goal (engage, screen,
   harvest, withdraw). Candidates are tactical hypotheses; the verifier
   runs a short replay rollout and scores certified utility. ~5K-10K
   params. Leakage policy: `forbidden`.
3. **Faction** — 4X-level decisions over a turn or fixed wall-clock
   window: build queue, expansion target, diplomacy stance, research.
   Candidates are strategic hypotheses; the verifier runs a longer macro
   rollout against a reduced state. ~10K-50K params. Leakage policy:
   `priors_allowed`, audited.

The bet is that the protocol's safety properties compose: a faction that
can only choose among certified strategies, executed by fleets that can
only issue certified tactics, flown by pilots that can only take certified
actions, cannot grief the simulation regardless of training state.

## Multiplayer determinism

Multiplayer is the load-bearing constraint and the reason a tiny-net stack
is the right choice here.

- The simulator is the verifier. Both clients run it; the trace protocol
  already enforces stable action ids, declared ranking policy, and
  score-margin metadata for tie resolution. That is exactly the
  determinism contract a lockstep multiplayer game needs.
- Inference at p95 0.089ms per pilot and 4177 FLOPs per candidate row fits
  inside a 16ms frame budget with two orders of magnitude of headroom for
  100 agents per frame.
- Float vs double drift across clients is already a tested concern: `make
  check` exercises both scalar types. A cross-machine determinism harness
  is the obvious extension and is M1 scope.

Open question: peer-to-peer lockstep vs authoritative server. Lockstep is
cheaper to operate and matches the trace-as-source-of-truth model.
Server-authoritative is more robust to cheating. Recommend lockstep for
the first public build, server-authoritative for ranked play.

## Style and personality

Optimal play is boring to play against. The framework gives `won't grief`;
it does not give `feels alive`. Proposed mechanisms, in increasing
ambition:

1. **Per-agent style vector** — 8-16 extra params per NPC, concatenated
   into every feature row. One shared trunk, distinct behavior. Cheap,
   ships first.
2. **Suboptimal utility targets** — train rankers against curated weak-play
   distributions (overcommitting, tunnel-vision, risk-averse). Certificate
   still holds; competence varies. Style becomes a training-data axis.
3. **Mixture-of-experts router** over hypothesis families (engage vs evade
   vs harass). The router is the personality; the experts are the shared
   skills.
4. **Faction-flavored grammars** — different cultures expose different
   hypothesis families to their pilots. Style becomes a content axis.

Each mechanism is independently testable against the existing trace and
replay protocol; none require changes to the certificate checker.

## Training loop

The self-improving game server is the natural endpoint of the existing
self-distillation rollout protocol:

```text
match -> trace -> certify utility rows -> mine new hypotheses
      -> propose new rules -> certificate gate -> retrain rankers
      -> shadow-deploy next checkpoint -> activate per role on regret gate
```

Existing scaffolding the loop can reuse without modification:

- `scripts/certified_hypothesis_library.py` for hypothesis mining and
  certification.
- `scripts/signal_shadow_replay_audit.py` and `scripts/signal_live_brain.mjs`
  for shadow-to-active deployment.
- `scripts/capture_results.py` and `make results` for stable-baseline
  comparison and CI-blocking drift detection.
- `.github/workflows/results.yml` nightly cron as the production cadence.

Every morning, last night's matches train tomorrow's pilots; the
certificate gate is what makes leaving the loop running unattended
defensible.

## Milestones

| M | Deliverable | Evidence |
| --- | --- | --- |
| M0 | Headless deterministic vector simulator with WASD pilot port | Existing `signal_flight` checkpoint flies in the new sim; trace round-trips |
| M1 | Two-player lockstep with one human and one neural pilot per side | `crlp.vector_4x.trace.v1` schema; replay across two clients is byte-identical; cross-machine determinism test in `make check` |
| M2 | Fleet ranker over 3-5 tactical hypotheses | Closed-loop rollout matrix; certified-hypothesis selection rate per fleet; regret vs no-safety baseline |
| M3 | Single-sector 4X loop: build, expand, fight | Faction ranker first checkpoint; match traces feed nightly retrain |
| M4 | Multi-sector + diplomacy + first style vectors | Style A vs Style B distinguishable in replay by behavior signature, not by skill |
| M5 | Public ranked build with daily checkpoint rotation | `make results`-style canonical match metrics; regression PRs on drift |

M0-M2 reuse the existing protocol almost verbatim. M3 is where new schema
work is required for the strategy tier.

## Risks and open questions

- **Content scope.** 4X games are content-heavy. The protocol does not
  reduce that cost; it only ensures the AI on top stays cheap. The
  hypothesis grammar at the fleet and faction tiers is the real authoring
  surface and could blow up.
- **Hypothesis coverage at strategy scale.** The current Signal pilot
  grammar is 5 hypotheses with ~41% certified-selection rate; a faction
  grammar may need hundreds. The certified library pipeline must scale
  with grammar size or M3 stalls.
- **Lockstep tolerance for float drift.** `make check` runs float and
  double on one machine. A cross-machine determinism test is not yet
  wired and is M1 scope.
- **Match length vs trace size.** A 30-minute 100-ship match produces a
  large JSONL trace. A binary trace format or sampling policy is needed
  before M5.
- **Self-improving server safety.** Daily checkpoint rotation must not be
  able to ship a regressed pilot. The shadow-to-active gate is the
  existing answer; it needs an explicit per-role policy (pilot can
  auto-promote on regret-gate pass; faction cannot without human review).
- **Discoverability of "alive" behavior.** Style mechanisms are easy to
  add and hard to evaluate. Need a behavior-signature metric in the trace
  protocol before M4.

## Out of scope

- LLM-driven NPCs, voice, or narrative.
- Hand-authored behavior trees as a fallback layer. They would compete
  with the candidate grammar; pick one.
- Free-form mod support before M5.
- 3D, physics beyond 2D rigid-body, networked physics.

## Supported wording

> A 2D vector multiplayer 4X where every pilot, fleet, and faction agent
> is a separately certified tiny neural ranker, sharing one trace protocol
> with the same simulator that decides legal play.
