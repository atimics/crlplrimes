# Experiment Proposal: Certified Tiny Brains in a Bounded-Constraint Multiplayer 4X

Status: in flight. Single-brain ship pilot is live over the Signal
websocket protocol (`scripts/signal_live_brain.mjs`) against the running
game at `signal.ratimics.com` (source: `github.com/cenetex/signal`). Two
checkpoint families already deploy through the protocol bridge:
`signal-flight-live-v2` (WASD + tractor + brake) and
`signal-mining-grammar-v1` (per-ship tactical option ranker). Both expose
the same `heuristic` / `shadow_neural` / `neural_active` mode ladder.

## Purpose

Signal is a bounded-constraint 4X (3 minerals, finite modules, fixed
recipes, finite station slots). That boundedness is what makes the
verifier-grounded tiny-net protocol the right fit: the game's entity
model defines the agent boundaries and the game's binary action interface
defines the candidate grammar. The AI doesn't bolt on; it inhabits the
substrate the game already published.

Two claims to defend:

1. **Population scale.** The single-brain `neural_active` ship pilot is
   proven. The unproven, load-bearing claim is whether N ship brains in
   one match behave well together.
2. **Game evolution by construction.** Adding a new action to Signal
   should produce a brain that learns it from replay, without
   developer-in-the-loop retraining. This is the killer property of the
   ranker architecture and is not yet realized end-to-end.

Negative claims:

- Not a claim tiny nets beat LLM agents at long-horizon planning.
- Not a claim the game changes shape. The game is shipped; the AI lives
  on top of its existing binary protocol.
- Not a replacement for designed systems. The candidate grammar is
  game-authored; the ranker only chooses among certified options.

## Entities as AI execution sites

Ships and stations are not "agent tiers we chose." They are the only
addressable units the game exposes. Each has identity (callsign,
station identity record), state (kinematics or modules/inventory), and
an action interface (`NET_MSG_INPUT` or the station action enum). A
brain inhabits an entity the way a process inhabits a slot: same
identity, same state, same actions the entity could ever take. There
is no "fleet coordinator" tier because the game has no fleet entity to
host one. There is no "diplomacy AI" tier because the game has no
diplomacy entity. Add a third entity class to Signal and a third brain
tier becomes possible by construction; otherwise there are two.

This is the principled answer to "how many ranker tiers should there
be." The right count equals the number of entity classes the game
defines, which today is two.

## Actions are game-authored

The candidate grammar is not authored on the AI side. It comes off the
wire:

- **Ship grammar.** The `NET_MSG_INPUT` bitfield (thrust, left, right,
  fire, brake, tractor) plus the per-ship action enum (`NET_ACTION_DOCK`,
  `LAUNCH`, `SELL_CARGO`, `REPAIR`, `BUY_PRODUCT`,
  `DELIVER_COMMODITY`, ...).
- **Station grammar.** Module-build queue slots, pricing levels per
  commodity, contract templates, hail topics. Today simulation-authored
  inside Signal; the brain proposes among them rather than redesigning
  them.

The certificate checker's job becomes much smaller in this framing.
It is not "prove this rule is safe in some general sense." It is
"prove this rule does not escape the entity's action interface" — and
since the brain only ever submits actions through the existing
protocol, that property is true by construction. The protocol bridge
has already proven this for ships: every neural decision turns into a
legal `NET_MSG_INPUT` packet or it doesn't get sent.

This collapses the surface area of certification work onto exactly the
research-interesting parts: utility hypotheses about *which* action is
good, not safety hypotheses about *which* action is legal.

## Current state

What exists today, in this repo and the Signal repo together:

| Component | Status | Evidence |
| --- | --- | --- |
| Signal game | Live, multiplayer, browser + native | `signal.ratimics.com`, `cenetex/signal` |
| Binary protocol bridge | Live | `scripts/signal_live_brain.mjs` speaks `NET_MSG_JOIN/INPUT/WORLD_*` directly, with explicit `--protocol-root` drift checks |
| Ship pilot (WASD + tractor + brake) | Live in `neural_active` | `signal-flight-live-v2` checkpoint, 0.089ms p95 inference, 74 active rows logged 2026-05-18 |
| Ship tactical options (mine/tow/smelt) | Live, gated `off` / `fixed` / `shadow_neural` / `neural_active` | `signal-mining-grammar-v1` checkpoint, `--option-mode` flag |
| Shadow-to-active gate | Live per checkpoint | Three-mode ladder on both checkpoints |
| Trace and replay | Live | `live.jsonl` per session, replay-certified utility rows |
| Self-improving loop | Partial | Replay-certified utility mining works; nightly retrain not wired to live matches |
| Multi-brain in one match | **Not yet** | Open. Headline question. |
| Per-type action encoder | **Not yet** | Open. Required for "new action learns by construction." |
| Station brain | **Not yet** | Open. The second entity tier. |

## Game shape (for reference)

| Field | Value |
| --- | --- |
| Genre | 2D vector multiplayer mining 4X with bounded resource/mod surface |
| Economy | 3 minerals (FM, LM, TM) → frames, laser modules, tractor modules; FM+LM+TM → repair kit |
| Combat | Tractor-tension rock release; no separate weapon system |
| Multiplayer | WebSocket, browser + native, server-authoritative |
| Stations | Per-sector hubs with module slots; host pricing, contracts, hail copy |
| Tick | Server-authoritative; brain ticks at 50ms by default |
| Replay | Brain-side JSONL trace; server traces handled separately |

The game is the verifier. Brains propose; the server decides.

## Mapping repo primitives to live game systems

| Repo primitive | Live mapping |
| --- | --- |
| Entity-as-host | Ship session and station identity |
| Candidate enumerator `A_T(s)` | Whatever the entity's action interface emits — read from the wire |
| Exact verifier `V_T(s, a)` | Signal authoritative server (deterministic tick) |
| Rule grammar `G_T` | Utility hypotheses over the game-authored action surface; safety hypotheses unnecessary by construction |
| Certificate checker `C_T` | Replay-certified utility rows; protocol-drift check against Signal source root |
| Tiny scorer `f_theta` | 2113-param ship pilot ranker; separate ship-option ranker; station ranker tbd |
| Trace protocol | `live.jsonl` per session; per-tier schema versioning |
| Shadow-to-active gate | `--brain-mode` and `--option-mode` flags, defaulting safe |

```text
The model proposes. The verifier decides. The trace remembers.
```

## Agent taxonomy

Two entities, two brain tiers. Nothing else.

1. **Ship (live).** WASD + tractor + brake plus tactical option choice
   (mine, tow, smelt, dock, sell, deliver). Combat is implicit: the
   tractor is the firing mechanism, and the ranker's choice of when to
   pull and when to release is the combat behavior. Two checkpoints
   today: `signal-flight-live-v2` for control, `signal-mining-grammar-
   v1` for options. ~2K params at the pilot level, p95 0.089ms.
   `neural_active` proven on a single brain.
2. **Station (open).** A station ranks its own behavior: pricing per
   commodity, contract offers, module build queue, repair-kit
   availability, hail decisions. Today these are simulation-authored
   heuristics in the Signal repo. The proposal is to expose them
   through the same certified-ranker pattern, against the same
   action-interface-as-grammar contract.

## Headline open question: population scale

Single-brain `neural_active` is proven. The unproven, load-bearing
claim is whether N ship brains in one match behave well together.

- **Compute** says yes: 0.089ms × 100 brains ≈ 9ms per tick batch, well
  inside a 50ms control period.
- **Protocol** says probably yes: each brain holds its own websocket
  session with its own callsign and session token. The bridge has no
  shared global state today.
- **Game** says unknown: server fan-out behavior under N concurrent
  brain sessions, latency tails, and packet loss handling are not
  measured. The `neural_active` regret gate was tuned on one brain
  against humans; it is not validated under brain-vs-brain dynamics.
- **Behavior** says unknown: brain-vs-brain matches can collapse into
  degenerate equilibria. Style vectors are the proposed mitigation but
  not yet tested.

This is the first thing to measure.

## Game evolution by construction

The killer property of a `f_theta(state, action)` ranker over a policy
head is that the action space can grow without architectural redesign.
A policy head has a fixed output dimension; adding a new action
requires retraining or grafting a new head. A ranker scores arbitrary
(state, action) pairs, so a new action shows up as a new row at
scoring time. The architecture does not change.

For "add a new action to Signal and the brain learns it" to hold
end-to-end, five pieces are required. Status of each:

1. **Action enum auto-discovery from the wire.** *Today: hardcoded.*
   `scripts/signal_live_brain.mjs` mirrors `NET_ACTION_*` and
   `COMMODITY_*` as constants. The bridge needs to read the enum from a
   schema artifact emitted by the Signal build (or grow the
   protocol-root drift check into full schema reflection).
2. **Per-type action encoder.** *Today: partial.* The feature row is 48
   hand-engineered observations with the action encoded inside it. The
   load-bearing move is to encode actions as
   `(action_type_embedding, argument_embeddings)` —
   `BUY_PRODUCT(commodity_id)` becomes
   `(BUY_PRODUCT_emb, commodity_emb[id])`. Adding
   `COMMODITY_PLASMA_KIT` adds one row to the commodity embedding
   table, randomly initialized, refined from replay. Same trunk, no
   architectural change. **This is the gating ML work.**
3. **Exploration budget for unfamiliar actions.** *Today: blocked.* The
   regret gate is correctly conservative: in `neural_active` it won't
   take an action it has no data on. In `shadow_neural` an exploration
   policy needs to deliberately sample new actions under a small cost
   budget so replay-certified utility rows accrue.
4. **Auto-spawned hypothesis on first sight.** *Today: manual.* Every
   new action should automatically get a degenerate hypothesis ("this
   action exists; expected utility starts at the empirical mean once
   data accrues"). The certified library accepts it trivially because
   the gate is "doesn't escape the entity's action interface" — true
   by construction for any wire-legal action.
5. **Schema versioning across game and brain.** *Today: header-only.*
   The drift check should grow into a compatibility manifest: the
   brain captures Signal's action enum at training time, runtime
   compares against the live enum, and unknown actions are tagged as
   `exploration_candidate` rather than silently ignored.

(2) is load-bearing. Without per-type action embeddings, a one-hot
scorer cannot generalize to an action ID it never saw, and the rest
of the chain doesn't matter. (1), (3), (4), (5) are mostly plumbing
once (2) is in place.

The research claim this enables: **certified tiny rankers absorb game
evolution by construction; policy-head agents absorb it by retraining.**
That is a sentence worth defending against any RL baseline, and it
falls out of the entity-and-action-interface framing rather than
requiring a new mechanism.

## Style and personality

The framework gives `won't grief`. It does not give `feels alive`. With
a population of N ship brains the style problem becomes load-bearing —
N identical optimal pilots is worse than one good one.

1. **Per-brain style vector** — 8-16 extra params concatenated into
   every feature row. One shared trunk, distinct behavior. Cheap.
2. **Suboptimal utility targets** — train against curated weak-play
   distributions. Certificate still holds; competence varies.
3. **Mixture-of-experts router** over hypothesis families (aggressive vs
   conservative tractor use, harvester vs raider). Personality is the
   router; skills are shared.
4. **Callsign-flavored grammars** — different callsign prefixes expose
   different hypothesis families.

All four require no change to the certificate checker. Stations get
the same mechanism on their own ranker: trader vs hoarder vs
aggressive-contract personalities.

## Self-improving loop

Existing scaffolding the loop can reuse without modification:

- `scripts/certified_hypothesis_library.py` — mining and certification.
- `scripts/signal_shadow_replay_audit.py` — shadow-to-active audit.
- `scripts/signal_live_active_summary.py` — active-mode rollups.
- `scripts/capture_results.py` and `make results` — stable-baseline diff
  and CI-blocking drift detection.
- `.github/workflows/results.yml` nightly cron — production cadence.

```text
live match -> live.jsonl -> certify utility rows -> mine hypotheses
           -> propose rules -> certificate gate -> retrain rankers
           -> shadow-deploy next checkpoint -> activate per tier on regret gate
```

The remaining work is the bridge between `live.jsonl` and the certified
hypothesis library, plus a per-tier promotion policy: ship pilot can
auto-promote on regret-gate pass; station needs human review on each
promotion until the loop is stable.

## Milestones

| M | Deliverable | Evidence |
| --- | --- | --- |
| M0 | Single-brain ship pilot live over websocket | **Done.** `neural_active` rows logged 2026-05-18. |
| M1 | Per-ship tactical option ranker | **Done.** `signal-mining-grammar-v1`, `--option-mode` deployed. |
| M2 | Population-scale ship brains in one match | N brains with distinct callsigns; per-brain inference p95, server-side latency tail, regret per brain under brain-vs-brain dynamics. |
| M3 | Style vectors at population scale | Style A vs Style B distinguishable in trace by behavior signature, not by skill, across N brains. |
| M4 | Per-type action encoder + auto-discovery | Action enum read from a Signal-emitted schema artifact; ranker uses `(type_emb, arg_emb)` encoding; one new action added in a Signal test build is scored and learned without checkpoint retraining; exploration policy + auto-spawned hypothesis closed. |
| M5 | Station brain MVP | Authored hypothesis families over the existing station action interface; reduced-state long-horizon rollout verifier; first checkpoint in `shadow_neural`. M4 is a hard prerequisite because the station action space is bigger and more structured than the ship's. |
| M6 | Nightly retrain wired to live match traces | Daily checkpoint rotation with per-tier promotion policy; regression PRs on canonical match metrics. |

M2 is the next concrete milestone and is mostly infrastructure. M4 is
the next ML milestone and is the one that earns the "game evolution by
construction" claim.

## Risks and open questions

- **Brain-vs-brain degeneracy.** N identical optimal ship pilots may
  collapse into avoidance, ramming, or queueing patterns that one-vs-
  humans never exposed. Style vectors are the mitigation; need a
  behavior-signature metric in the trace before M3 is evaluable.
- **Server fan-out under N sessions.** Untested. Could surface tick
  jitter, dropped inputs, or queueing artifacts that the regret gate
  would misattribute to its own decisions.
- **Schema reflection scope.** Reading the action enum from a Signal-
  emitted artifact (M4) requires either a small build-time emitter in
  the Signal repo or a brain-side parser of the C headers. Either is
  feasible; both are integration work.
- **Action-encoder cold-start.** A brand-new action has a randomly
  initialized embedding; the regret gate keeps `neural_active` from
  picking it until replay-certified rows accrue. That's correct but
  means "learns it" has a measurable latency. Needs an explicit SLA
  before claiming the property.
- **Station horizon length.** Reduced-state rollouts of full economic
  plans may be expensive enough to fall outside the verifier's
  comfortable budget. Sampling policy needed before M5.
- **Trace size.** `live.jsonl` per session is fine; N-brain match traces
  scale linearly. Binary format or sampling before M6.

## Out of scope

- Changes to the Signal game protocol. The AI lives on top of what's
  shipped.
- LLM-driven NPCs, voice, or narrative.
- Hand-authored behavior trees as a fallback layer. They would compete
  with the candidate grammar; pick one.
- New input modalities beyond what the entity's action interface emits.
- Any third agent tier. Two entities, two brains.

## Supported wording

> Signal is a bounded-constraint multiplayer 4X whose only addressable
> agents are ships and stations. Each entity's action interface is the
> certified-ranker candidate grammar by construction, so safety
> certification reduces to "doesn't escape the wire format" and the
> remaining research arc is utility — population-scale ship brains,
> per-type action encoding so new actions are learned by construction,
> and a station brain over the existing economic action surface.
