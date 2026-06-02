# Design note: branch is free, death is the missing operator

**Status:** design note / not yet implemented. Captures a cross-cutting principle
observed across the engine, the Go subsystem, the codebase, and the paper.

## The principle

Every level of this project *branches* — it enumerates candidates, forks files,
spawns experiments, opens paper variants. Branching is generative and happens by
default; it costs nothing to produce another branch.

What every level needs, and mostly lacks, is **death**: an explicit selection
operator that removes a branch once it is dominated. Death is not free — it
requires a *criterion* (a fitness/domination test), and choosing that criterion
is the real design work.

The repository already demonstrates the principle by contrast:

| Subsystem | Branch (have) | Death (criterion) | Healthy? |
| --- | --- | --- | --- |
| Claim compiler | candidate claims | decision rule → `limitations.md` | **yes** — death is encoded |
| Verified search (prime/grid) | candidate enumeration | certificate rejects (`exact_check`) | yes |
| Go engines | MCTS expansion, 5 forked binaries | *none* (see below) | no |
| Codebase | file forks, scripts | *none* — nothing deletes a dominated fork | no |
| Paper | v1 / v2 documents | *none* — both import the full claim set | drifting |

The one subsystem where death is a first-class operator (the claim compiler) is
the one subsystem that is unambiguously healthy. That is the finding, not a
metaphor: **health tracks where death is encoded.**

## Death is already the architecture

"Enumerate candidates → ground with the verifier → rank the residual" is
branch (enumerate) + death (verifier rejects) + order (ranker). MCTS in Go and
Signal is the same move with a learned prior and a backup operator. The project
does not need to *add* branch-and-death; it needs to (a) name the death operator
as a shared primitive so domains stop reimplementing it, and (b) extend it to the
levels that currently have no death at all.

There are already three death operators in the tree, one per grounding authority:

- **hard death** — a certificate rejects the candidate (`exact_check`): provably gone.
- **replay death** — a simulator outcome kills the candidate (`deterministic_replay`): Signal, Go.
- **soft death** — the learned ranker deprioritizes (never deletes, only downranks).

Unifying these as one interface is the substrate generalization the search loop
has been circling.

## Proposed primitive: `nn_search` (frontier + death + rank)

A single shared search core so domains adapt rather than fork:

```
typedef enum { PRUNE_KILL, PRUNE_KEEP, PRUNE_DOWNRANK } PruneVerdict;

typedef struct {
    /* domain adapter */
    void   (*enumerate)(void *state, Candidate *out, size_t *n);   /* branch  */
    PruneVerdict (*ground)(void *state, const Candidate *c);       /* death   */
    nn_real (*rank)(NeuralNet *net, const Candidate *c);           /* order   */
} SearchOps;

int nn_search_step(NeuralNet *net, void *state, const SearchOps *ops,
                   Candidate *frontier, size_t *frontier_n);
```

Go, Signal, ARC sketch search, and prime rule search become *adapters* over this
core. Test of success: adding domain N+1 is a small adapter, and a bug fix is
impossible to apply to "only one copy." If adding a domain still means copying a
900-line file, the primitive is not real yet.

This same change dissolves the code-fork problem: the five Go engines fork
*because the search loop is not a shared object*. Make it one, and there is
nothing to fork.

## Where death is missing, by level

### Go (missing at four levels — diagnostic)
1. **Game** — no resignation; games run to the move cap, so a lost position emits
   no death signal until terminal scoring. Add a resignation threshold on the
   value estimate.
2. **Memory** — the tournament replay buffer is append-until-full, not a ring;
   samples never expire and training freezes on stale early positions. Make it a
   true ring.
3. **Population** — it is called a *tournament* but implemented as a round-robin
   with shared training: **no agent is ever culled, no winner ever reproduces.**
   The missing evolutionary loop is: each epoch, kill the bottom-k by ELO and
   respawn them as mutated clones of the top-k. This is the single highest-signal
   change — it makes "tournament" true, supplies real selection pressure, and
   makes the flat-baseline-ELO bug moot because progress comes from replacing the
   weak rather than from a frozen reference line.
4. **Domain** — Go *is* a game about death (capturing groups, life-and-death);
   the value head meant to learn it was label-leakage-broken until the
   `nn_search`-adjacent fix, so the engine could not learn the death intrinsic to
   the domain.

### Signal (maps cleanly — death is already literal and replay-certified)
Signal is the cleanest place to encode death-as-selection, because the
authoritative simulator already kills ships and the claim artifacts already log
`damage` and `death` rates as verified outcomes. Unlike Go, the fitness signal
does not have to be invented — replay hands it over directly.

1. **Agent / episode** — the ship's hull reaches zero; literal, telemetered,
   replay-certified death.
2. **Candidate** — WASD candidates that lead to damage/death should be
   hard-pruned by replay (the `deterministic_replay` death operator). This
   partly exists via the safety gates.
3. **Population** — `signal_live_swarm.mjs` is a *swarm* of pilots with no
   culling and no reproduction: the same deathless round-robin as the Go
   tournament. The missing loop is identical — cull the worst pilots by
   replay-certified survival/utility and respawn them as mutated clones of the
   best.
4. **Memory** — shadow/replay frames never expire.

**The two faces of death (Signal's distinctive contribution).** Signal forces a
split that Go's abstract game let us ignore, and getting it wrong is dangerous:

- **death-as-constraint** (object level, *avoid*): for the live deployed pilot,
  death must be prevented — the safety gate prunes lethal actions, and the claim
  requires damage/death no worse than baseline. A pilot must never embrace death.
- **death-as-selection** (meta level, *exploit*): across the swarm/training loop,
  death is *information* — a pilot that dies more is culled, one that
  survives-and-achieves reproduces.

These pull in opposite directions and must stay separated: death-as-selection in
the live path trains a suicidal ship; death-as-constraint in the training loop
gives no selection pressure. This is the biology structure (an organism avoids
its *own* death while evolution *uses* death across the population) and it aligns
with the project invariant that **the model proposes, the verifier decides, and
the model never owns safety.** Death-as-selection belongs in the
training/population loop; death-as-constraint belongs in the replay-gated live
path; the two never cross.

### Codebase
Forks branch but never die. Add a **supersedes registry**: when a fork is added it
declares what it dominates, and CI fails if a dominated artifact still exists.
This is the claim-compiler pattern applied to code — the death function the
codebase currently lacks.

### Paper
v1 and v2 both import the full generated claim set and drift in framing. Either
v2 *supersedes* v1 (and v1 dies), or the two become scoped *views* of one source
emitting different claim subsets — not two copies maintained against one
evidence base.

## The bitter-lesson tie-back

Evolution is branch + death over genomes; the cheap-inference brain is the
*surviving artifact*. Death is the operator that makes scaled search pay off, and
it is the half this project under-implements everywhere except the claim
compiler. Encoding death as a first-class, criterion-driven operator — at the
search, population, code, and paper levels — is therefore not housekeeping; it is
the same move as the project's own thesis, applied consistently.
