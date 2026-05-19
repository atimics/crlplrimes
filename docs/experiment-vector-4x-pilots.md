# Experiment Proposal: Certified Tiny Pilots at Population Scale in Signal

Status: in flight. Single-brain ship pilot is live over the Signal
websocket protocol (`scripts/signal_live_brain.mjs`) against the running
game at `signal.ratimics.com` (source: `github.com/cenetex/signal`). Two
checkpoint families already deploy through the protocol bridge:
`signal-flight-live-v2` (WASD + tractor + brake) and
`signal-mining-grammar-v1` (per-ship tactical option ranker). Both expose
the same `heuristic` / `shadow_neural` / `neural_active` mode ladder. This
document supersedes the earlier "vector 4X" framing; the game already
exists, so the proposal is about what's next, not what to build.

## Purpose

The ship pilot tier proves the verifier-grounded tiny-net protocol works
end-to-end against a real multiplayer game. The remaining research
contribution is whether the protocol holds at **population scale** —
many independent ship brains per match, each with its own checkpoint and
style — and whether the same protocol generalizes to **stations**:
station-authored behavior over pricing, contracts, module builds, and
hail decisions, on the long-horizon side of the action space.

Two tiers, no others.

Negative claims:

- Not a claim tiny nets beat LLM agents at long-horizon planning.
- Not a claim the game changes shape. The game is shipped; the AI lives
  on top of its existing binary protocol.
- Not a replacement for designed systems. The candidate grammar is
  authored; the ranker only chooses among certified options.

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
| Multi-brain in one match | **Not yet** | Open. Headline question for this proposal. |
| Station brain | **Not yet** | Open. The second tier. |

## Game shape (for reference)

| Field | Value |
| --- | --- |
| Genre | 2D vector multiplayer mining game; combat by tractor-tension rock release |
| Economy | 3 minerals (FM, LM, TM) → frames, laser modules, tractor modules; FM+LM+TM → repair kit |
| Multiplayer | WebSocket, browser + native, server-authoritative |
| Stations | Per-sector hubs with module slots (frame press, laser fab, tractor fab, shipyard, dock); host pricing, contracts, hail copy |
| Tick | Server-authoritative; brain ticks at 50ms by default |
| Replay | Brain-side JSONL trace; server traces handled separately |

The game is the verifier. Brains propose; the server decides.

## Mapping repo primitives to live game systems

The schema in `include/nn_experiment.h` is already instantiated end-to-end
against the live game via the websocket bridge.

| Repo primitive | Live mapping |
| --- | --- |
| Candidate enumerator `A_T(s)` | Ship: 9 WASD combos + tractor/brake, plus tactical option set. Station: pricing levels, contract offers, module build queue, hail targets. |
| Exact verifier `V_T(s, a)` | Signal authoritative server (deterministic tick) |
| Rule grammar `G_T` | Ship: flight + mining hypotheses (`signal-mining-grammar-v1`). Station: economic and contract hypotheses (open). |
| Certificate checker `C_T` | Replay-certified utility rows; protocol-drift check against Signal source root |
| Tiny scorer `f_theta` | 2113-param ship pilot ranker; separate ship-option ranker; station ranker tbd |
| Trace protocol | `live.jsonl` per session; per-tier schema versioning |
| Shadow-to-active gate | `--brain-mode` and `--option-mode` flags, defaulting safe |

```text
The model proposes. The verifier decides. The trace remembers.
```

## Agent taxonomy

Two tiers. Nothing else.

1. **Ship (live).** WASD + tractor + brake plus tactical option choice
   (mine, tow, smelt, dock, sell, deliver). Combat is implicit: the
   tractor is the firing mechanism, and the ranker's choice of when to
   pull and when to release is the combat behavior. Two checkpoints,
   one trunk in practice: `signal-flight-live-v2` for control,
   `signal-mining-grammar-v1` for options, ~2K params at the pilot
   level, p95 0.089ms. `neural_active` proven on a single brain.
2. **Station (open).** A station decides its own behavior: pricing per
   commodity, contract offers, module build queue, repair-kit
   availability, who to hail and what to say. Today these are
   simulation-authored heuristics in the Signal repo. The proposal is
   to expose them through the same certified-ranker pattern: a station
   brain proposes from an authored candidate grammar, the server
   verifies (commits the pricing, runs the contract), the trace
   captures certified utility rows over a longer horizon (hours of
   sector economy, not seconds of flight).

The bet: the protocol's safety properties compose across the two tiers.
Stations that can only choose among certified economic plans, populated
by ships that can only take certified WASD+tractor+option combinations,
cannot grief the live game regardless of training state.

## Headline open question: population scale

Single-brain `neural_active` is proven. The unproven, load-bearing claim
is whether N ship brains in one match behave well together.

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

This is the first thing to measure, before the station tier.

## Style and personality

The framework gives `won't grief`. It does not give `feels alive`. With a
population of N ship brains the style problem becomes load-bearing — N
identical optimal pilots is worse than one good one.

1. **Per-brain style vector** — 8-16 extra params concatenated into every
   feature row. One shared trunk, distinct behavior. Cheap, ships first.
2. **Suboptimal utility targets** — train against curated weak-play
   distributions (over-pulling tractor, slow-to-release, risk-averse on
   repair). Certificate still holds; competence varies. Style as a
   training-data axis.
3. **Mixture-of-experts router** over hypothesis families (aggressive vs
   conservative tractor use, harvester vs raider). Personality is the
   router; skills are shared.
4. **Callsign-flavored grammars** — different callsign prefixes expose
   different hypothesis families. Style as a content axis.

All four are testable through the existing trace protocol and require no
change to the certificate checker. Stations get their own analogue:
trader vs hoarder vs aggressive-contract personalities driven by the
same style-vector mechanism on the station ranker.

## Self-improving loop

The existing replay-certified rollout protocol already produces the
input to a nightly retrain. The missing piece is wiring it to live
match traces:

```text
live match -> live.jsonl -> certify utility rows -> mine hypotheses
           -> propose rules -> certificate gate -> retrain rankers
           -> shadow-deploy next checkpoint -> activate per tier on regret gate
```

Scaffolding already present:

- `scripts/certified_hypothesis_library.py` — mining and certification.
- `scripts/signal_shadow_replay_audit.py` — shadow-to-active audit.
- `scripts/signal_live_active_summary.py` — active-mode rollups.
- `scripts/capture_results.py` and `make results` — stable-baseline diff
  and CI-blocking drift detection.
- `.github/workflows/results.yml` nightly cron — production cadence.

The remaining work is the bridge between `live.jsonl` and the certified
hypothesis library, plus a per-tier promotion policy: ship pilot can
auto-promote on regret-gate pass; station needs human review on each
promotion until the loop is stable.

## Milestones (revised)

| M | Deliverable | Evidence |
| --- | --- | --- |
| M0 | Single-brain ship pilot live over websocket | **Done.** `neural_active` rows logged 2026-05-18. |
| M1 | Per-ship tactical option ranker | **Done.** `signal-mining-grammar-v1`, `--option-mode` deployed. |
| M2 | Population-scale ship brains in one match | N brains with distinct callsigns; per-brain inference p95, server-side latency tail, regret per brain under brain-vs-brain dynamics. |
| M3 | First style vectors at population scale | Style A vs Style B distinguishable in trace by behavior signature, not by skill, across N brains. |
| M4 | Station brain MVP | Authored candidate grammar over pricing, contracts, module build queue, hail; reduced-state long-horizon rollout verifier; first checkpoint in `shadow_neural`. |
| M5 | Nightly retrain wired to live match traces | Daily checkpoint rotation with per-tier promotion policy; regression PRs on canonical match metrics. |

M2 is the next concrete milestone and is mostly infrastructure (spawn N
brains, instrument, measure) rather than new ML.

## Risks and open questions

- **Brain-vs-brain degeneracy.** N identical optimal ship pilots may
  collapse into avoidance, ramming, or queueing patterns that one-vs-
  humans never exposed. Style vectors are the mitigation; need a
  behavior-signature metric in the trace before M3 is evaluable.
- **Server fan-out under N sessions.** Untested. Could surface tick
  jitter, dropped inputs, or queueing artifacts that the brain's regret
  gate would misattribute to its own decisions.
- **Protocol drift.** The bridge already fails fast on header mismatch,
  but a richer compatibility matrix between Signal commits and brain
  checkpoints would let `make results` block on drift the same way it
  blocks on numerical drift.
- **Station horizon length.** Reduced-state rollouts of full economic
  plans may be expensive enough to fall outside the verifier's
  comfortable budget. Sampling policy needed before M4 closes.
- **Station authoring surface.** The candidate grammar (pricing levels,
  contract templates, hail topics) is a real authoring job. Probably
  bounded by the existing protocol's action enum but not proven
  bounded.
- **Trace size.** `live.jsonl` per session is fine; N-brain match traces
  scale linearly. Binary format or sampling before M5.

## Out of scope

- Changes to the Signal game protocol. The AI lives on top of what's
  shipped.
- LLM-driven NPCs, voice, or narrative.
- Hand-authored behavior trees as a fallback layer. They would compete
  with the candidate grammar; pick one.
- New input modalities beyond the existing `NET_MSG_INPUT` and station
  action enums.
- Any third agent tier. Ships and stations are the entire model.

## Supported wording

> A live multiplayer mining game whose ship tier is already flown by a
> 2113-parameter certified ranker over websocket, and whose remaining
> research arc is whether the same protocol holds at population scale
> across many ship brains and extends cleanly to a station brain over
> the existing economic action space.
