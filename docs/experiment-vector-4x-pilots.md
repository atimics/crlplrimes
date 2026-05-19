# Experiment Proposal: Certified Tiny Pilots at Population Scale in Signal

Status: in flight. Single-brain pilot is live over the Signal websocket
protocol (`scripts/signal_live_brain.mjs`) against the running game at
`signal.ratimics.com` (source: `github.com/cenetex/signal`). Two checkpoint
families already deploy through the protocol bridge:
`signal-flight-live-v2` (pilot, WASD + tractor + brake) and
`signal-mining-grammar-v1` (per-ship tactical option ranker). Both expose
the same `heuristic` / `shadow_neural` / `neural_active` mode ladder. This
document supersedes the earlier "vector 4X" framing; the game already
exists, so the proposal is about what's next, not what to build.

## Purpose

The pilot tier proves the verifier-grounded tiny-net protocol works
end-to-end against a real multiplayer game. The remaining research
contribution is whether the protocol holds at **population scale** — many
independent brains per match, each with its own checkpoint and style — and
across the **logistics horizon** (mining, mod choice, repair, expansion)
that Signal's economy already exposes.

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
| Pilot ranker (WASD + tractor + brake) | Live in `neural_active` | `signal-flight-live-v2` checkpoint, 0.089ms p95 inference, 74 active rows logged 2026-05-18 |
| Mining option ranker | Live, gated `off` / `fixed` / `shadow_neural` / `neural_active` | `signal-mining-grammar-v1` checkpoint, `--option-mode` flag |
| Shadow-to-active gate | Live per tier | Three-mode ladder on both checkpoints |
| Trace and replay | Live | `live.jsonl` per session, replay-certified utility rows |
| Self-improving loop | Partial | Replay-certified utility mining works; nightly retrain not wired to live matches |
| Multi-brain in one match | **Not yet** | Open. Headline question for this proposal. |
| Multi-ship coordination ranker | **Not yet** | Open. The fleet tier. |
| Logistics ranker (mods, repair, expansion) | **Not yet** | Open. The economic tier. |

## Game shape (for reference)

| Field | Value |
| --- | --- |
| Genre | 2D vector multiplayer mining game; combat by tractor-tension rock release |
| Economy | 3 minerals (FM, LM, TM) → frames, laser modules, tractor modules; FM+LM+TM → repair kit |
| Multiplayer | WebSocket, browser + native, lockstep-equivalent through the authoritative server |
| Stations | Per-sector hubs with module slots; dock/launch/sell/buy actions in the protocol |
| Tick | Server-authoritative; brain ticks at 50ms by default |
| Replay | Brain-side JSONL trace; server traces handled separately |

The game is the verifier. The brain proposes; the server decides.

## Mapping repo primitives to live game systems

The schema in `include/nn_experiment.h` is already instantiated end-to-end
against the live game via the websocket bridge. No new abstractions
required to extend it.

| Repo primitive | Live mapping |
| --- | --- |
| Candidate enumerator `A_T(s)` | 9 WASD combos + tractor/brake; tactical option set in `signal_mining` |
| Exact verifier `V_T(s, a)` | Signal authoritative server (deterministic tick) |
| Rule grammar `G_T` | Flight hypotheses (direct, nav path, left/right bypass, brake/escape); mining grammar in `signal-mining-grammar-v1` |
| Certificate checker `C_T` | Replay-certified utility rows; protocol-drift check against Signal source root |
| Tiny scorer `f_theta` | 2113-param pilot ranker; separate option ranker |
| Trace protocol | `live.jsonl` per session; per-tier schema versioning |
| Shadow-to-active gate | `--brain-mode` and `--option-mode` flags, defaulting safe |

```text
The model proposes. The verifier decides. The trace remembers.
```

## Agent taxonomy

Three certified ranker tiers. Two exist today as single-instance
checkpoints; the third is open.

1. **Pilot (live)** — WASD + tractor + brake on one ship. The tractor is
   the firing mechanism: combat behavior is implicit in the pilot
   ranker's choice of when to pull and when to release. `signal-flight-
   live-v2`, ~2K params, p95 0.089ms, `neural_active` proven on a single
   brain.
2. **Per-ship tactical (live)** — mining/tow/smelt option ranking over a
   small grammar of certified utility hypotheses. `signal-mining-grammar-
   v1`, runs alongside the pilot via `--option-mode`. Provides the
   `which-option` decision the pilot then executes.
3. **Multi-ship coordination (open)** — role assignment across a small
   team: who harvests, who escorts, who runs cargo home, who flies
   interference. Candidates are role bundles over 2-8 ships. The
   verifier is a short server-state replay rollout. New schema work.
4. **Logistics / economic (open)** — long-horizon decisions over Signal's
   actual economy: which station to dock at, which mods to buy, when to
   sell vs hoard, when to push to the next sector, when to claim
   `BUY_REPAIR_KIT`. Candidate grammar is bounded by the protocol's
   action enum (a real, finite list). The verifier is a longer reduced-
   state rollout. This is the only place the original "faction tier"
   framing survives, and it is much smaller than 4X-strategy makes it
   sound.

The bet: the protocol's safety properties compose across tiers. Logistics
that can only choose among certified economic plans, executed by
coordinators that can only issue certified role bundles, executed by
pilots that can only take certified WASD+tractor combinations, cannot
grief the live game regardless of training state.

## Headline open question: population scale

Single-brain `neural_active` is proven. The unproven, load-bearing claim
is whether N brains in one match behave well together.

- **Compute** says yes: 0.089ms × 100 brains ≈ 9ms per tick batch, well
  inside a 50ms control period.
- **Protocol** says probably yes: each brain holds its own websocket
  session with its own callsign and session token. The bridge has no
  shared global state today.
- **Game** says unknown: the server's fan-out behavior under N concurrent
  brain sessions, latency tails, and packet loss handling are not
  measured. The brain's `neural_active` regret gate was tuned on one
  brain against humans; it is not validated under brain-vs-brain
  dynamics.
- **Behavior** says unknown: brain-vs-brain matches can collapse into
  degenerate equilibria. Style vectors are the proposed mitigation but
  not yet tested.

This is the first thing to measure, before any new ranker tier.

## Style and personality

The framework gives `won't grief`. It does not give `feels alive`. With a
population of N brains the style problem becomes load-bearing — N
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
4. **Faction-flavored grammars** — different callsign-prefixes expose
   different hypothesis families. Style as a content axis.

All four are testable through the existing trace protocol and require no
change to the certificate checker.

## Self-improving loop

The existing replay-certified rollout protocol already produces the input
to a nightly retrain. The missing piece is wiring it to live match
traces:

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
hypothesis library, plus a per-tier promotion policy: pilot can auto-
promote on regret-gate pass; coordination and logistics need human
review on each promotion until the loop is stable.

## Milestones (revised)

| M | Deliverable | Evidence |
| --- | --- | --- |
| M0 | Single-brain pilot live over websocket | **Done.** `neural_active` rows logged 2026-05-18. |
| M1 | Per-ship tactical option ranker | **Done.** `signal-mining-grammar-v1`, `--option-mode` deployed. |
| M2 | Population-scale brains-in-one-match | N brains with distinct callsigns; per-brain inference p95, server-side latency tail, regret per brain under brain-vs-brain dynamics. |
| M3 | First style vectors at population scale | Style A vs Style B distinguishable in trace by behavior signature, not by skill, across N brains. |
| M4 | Multi-ship coordination ranker | Role-bundle candidate grammar, replay-rollout certificate; closed-loop coordination matrix vs single-brain baselines. |
| M5 | Logistics / economic ranker | Long-horizon option grammar over BUY/SELL/DOCK/REPAIR action space; reduced-state rollout verifier. |
| M6 | Nightly retrain wired to live match traces | Daily checkpoint rotation with per-tier promotion policy; regression PRs on canonical match metrics. |

M2 is the next concrete milestone and is mostly infrastructure (spawn N
brains, instrument, measure) rather than new ML.

## Risks and open questions

- **Brain-vs-brain degeneracy.** N identical optimal pilots may collapse
  into avoidance, ramming, or queueing patterns that one-vs-humans never
  exposed. Style vectors are the mitigation; need a behavior-signature
  metric in the trace before M3 is evaluable.
- **Server fan-out under N sessions.** Untested. Could surface tick
  jitter, dropped inputs, or queueing artifacts that the brain's regret
  gate would misattribute to its own decisions.
- **Protocol drift.** The bridge already fails fast on header mismatch,
  but a richer compatibility matrix between Signal commits and brain
  checkpoints would let `make results` block on drift the same way it
  blocks on numerical drift.
- **Coordination grammar authoring.** Role bundles are a real authoring
  surface. Probably small (harvest/escort/courier/interference) but not
  proven small.
- **Logistics horizon length.** Reduced-state rollouts of full economic
  plans may be expensive enough to fall outside the verifier's
  comfortable budget. Sampling policy needed.
- **Trace size.** `live.jsonl` per session is fine; N-brain match traces
  scale linearly. Binary format or sampling before M6.

## Out of scope

- Changes to the Signal game protocol. The AI lives on top of what's
  shipped.
- LLM-driven NPCs, voice, or narrative.
- Hand-authored behavior trees as a fallback layer. They would compete
  with the candidate grammar; pick one.
- New input modalities beyond the existing `NET_MSG_INPUT` action space.

## Supported wording

> A live multiplayer mining game whose pilot tier is already flown by a
> 2113-parameter certified ranker over websocket, and whose remaining
> research arc is whether the same protocol holds at population scale and
> across coordination and economic time horizons.
