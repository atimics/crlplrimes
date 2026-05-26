# Program-Induction Substrate (As Built)

Status: description of code that exists in `main` as of `b7ebf79`.
Not a proposal. This document narrates the architecture the C library
already constitutes, with references to the headers, commits, and
schema names that ship it. Intended use: orientation. A reader walking
up to the repo for the first time should be able to read this and
know what the type signatures enforce, what the wire protocol
guarantees, and what the trunk already scores.

## Pipeline

The library implements a five-component pipeline. Every component has
a typed C interface, a versioned protocol schema, and at least one
in-tree user.

```text
program inducer (LEARNED or CLOSED_LOOP proposer)
   ↓ proposes typed programs
certificate gate (NNCertificate attached, support/mismatch tracked)
   ↓ admits programs to
library promotion record (crlp.protocol.library_promotion.v1)
   ↓ executed at runtime
candidate row (crlp.protocol.candidate_row.v1)
   ↓ scored by
role-tagged scorer (96-wide envelope, joint action+rule trunk)
   ↓ ranker decision (crlp.protocol.ranker_decision.v1)
   ↓ outcome from the world
verifier outcome (crlp.protocol.verifier_outcome.v1)
   ↓ closes back to the inducer
```

Cost characteristics:

- **Inducer** — expensive, rare. Offline or occasional closed-loop.
- **Certificate gate** — cheap. Structural checks plus counter updates.
- **Library** — cheap. Lookups and append-only promotions.
- **Executor** — cheap. DSL evaluation on bounded typed programs.
- **Ranker** — kiloscale (96 features × tiny trunk), runs every
  decision. Small per call by construction.

Total runtime cost is dominated by ranking. Total wall-clock cost
across a campaign is dominated by induction. The split is deliberate.

## Components

### 1. Typed DSL — `include/nn_program.h`

Introduced in `40573e1` (typed program engine layer). Defines:

| Construct | Values |
| --- | --- |
| `NNProgramValueType` | `ANY`, `GRID`, `OBJECT_SET`, `OBJECT`, `MASK`, `SCALAR`, `BOOLEAN`, `SEQUENCE`, `CUSTOM` |
| `NNProgramKind` | `PRIMITIVE`, `COMPOSITION`, `SELECTOR`, `PREDICATE`, `TRANSFORM`, `CUSTOM` |
| `NNProgramProposerStrategy` | `RANDOM`, `GRAMMAR_ORDER`, `STRUCTURAL_HEURISTIC`, `LEARNED`, `CLOSED_LOOP`, `ORACLE` |

Every `NNProgram` carries:

- Typed input and output (`input_type`, `output_type`).
- A canonical id and 64-bit canonical hash, so deduplication and
  caching are first-class.
- An `NNCertificate` with support and mismatch counters across
  train/validation/test.
- A structural feature vector (up to 96 features) for proposer
  scoring.
- Up to 16 typed parameters and 4 child program references for
  composition.

The proposer-strategy enum is load-bearing: the substrate explicitly
admits baselines (`RANDOM`, `GRAMMAR_ORDER`, `ORACLE`) alongside the
learned variants, which is what makes proposer comparisons honest.

### 2. Domain-neutral hypothesis layer — `include/nn_hypothesis.h`

Introduced in `58d6b12` (self-hypothesizing ARC induction). The
header comment states the unification claim directly:

> A hypothesis is a typed operator skeleton (a "template") plus a
> concrete slot binding inferred from observed task evidence.
> Compiling a hypothesis produces an NNProgram (the canonical row
> format used by the verifier and the existing closed-loop policy).
> The hypothesis layer is intentionally domain-neutral so that ARC,
> program_synthesis, and prime_gap can all use the same template +
> slot vocabulary.

Provenance contract enforced by the header: slot values bound from
state are `NN_FEATURE_STATE_OBSERVATION` or
`NN_FEATURE_STRUCTURAL_PRIOR`; verifier-derived values are
`NN_FEATURE_VERIFIER_DERIVED` or `NN_FEATURE_DERIVED_FROM_GATE` and
cannot reach a selection-mode proposer's feature vector.

### 3. Verifier / library protocol — `include/nn_protocol.h`

Introduced in `ec7fb3e` (unified candidate verifier protocol). Five
versioned schemas:

| Schema | Purpose |
| --- | --- |
| `crlp.protocol.candidate_row.v1` | A candidate proposed for verification: stable id, domain, feature vector, provenance, optional example text. |
| `crlp.protocol.verifier_outcome.v1` | Verifier result: `ACCEPTED` / `UNSAFE` / `NO_SUPPORT` / `SAFE_DUPLICATE` / `REJECTED`, with utility, support, false-rejection count, cost, reason code, proof kind. |
| `crlp.protocol.ranker_decision.v1` | Output of the kiloscale ranker: task name, model hash, candidate ids, scores, probabilities, selected index, target index, score margin, ranking policy. |
| `crlp.protocol.discovered_rule.v1` | A new candidate entering the library, with its verifier outcome and the template id it was generated from. |
| `crlp.protocol.library_promotion.v1` | A library promotion event: support across tasks, coverage, false rejections, utility, held-out rows, reason code. |

The five together describe the entire data flow between inducer,
library, ranker, and verifier as a public contract. They are not
aspirations; they're typedef'd structs and parsing code.

### 4. Role-tagged scorer envelope — `include/nn_core.h`

Introduced in `b7ebf79` (extract role-tagged scorer envelope). The
load-bearing primitive of the substrate.

```c
#define NN_SCORER_ENVELOPE_FEATURE_COUNT 96u
#define NN_SCORER_ENVELOPE_FEATURE_SET "task_role_tagged_action_rule_rows_v1"

typedef enum {
    NN_SCORER_ROLE_ACTION = 0,
    NN_SCORER_ROLE_RULE = 1,
} NNScorerRole;

typedef enum {
    NN_SCORER_DOMAIN_PRIME_GAP_ACTION = 0,
    NN_SCORER_DOMAIN_GRID_PATH_ACTION = 1,
    NN_SCORER_DOMAIN_DELAYED_CLAIM_ACTION = 2,
    NN_SCORER_DOMAIN_ARC_AGI_ACTION = 3,
    NN_SCORER_DOMAIN_PRIME_RULE_PROPOSE = 4,
} NNScorerDomain;
```

One 96-wide feature row, fixed schema, joint-trainable trunk. The
row carries a role tag (action vs rule), a domain tag (which task),
candidate index and count, and slot regions for action-common,
action-state, role-payload, and residual features.

This is the kiloscale ranker for everything the substrate ranks.
Action selection across four domains and rule proposal for prime_gap
all flow through the same parameters. Adding a new domain or a new
role is an enum entry plus a slot assignment, not architectural
work.

Joint training across this envelope landed in `8d807bb` (unified
scorer joint training).

### 5. Protocol guards — across the headers

Hardened in `4567b2d` (harden protocol audits and guards). Drift
checks, schema-version assertions, leakage-policy enforcement. The
guards are what let the substrate run unattended: schema mismatches
fail fast, leakage violations are blocked at the experiment-schema
emitter in `include/nn_experiment.h`.

## Active domains and roles

The role-tagged envelope currently registers:

| Domain | Role | Status |
| --- | --- | --- |
| `prime_gap` | action | live, certified-gate flagship |
| `prime_gap` | rule proposal | live, currently the only `RULE` role |
| `grid_path` | action | live, with admissible-bound predicates |
| `delayed_claim` | action | live, threshold + margin gates |
| `arc_agi` | action | live, fixed-sample plus learned program proposer |

`signal_flight` runs through a parallel websocket protocol bridge
(`scripts/signal_live_brain.mjs`) that maps to the substrate's
candidate / verifier / ranker contracts but uses Signal's own wire
schemas. The substrate does not need restructuring for Signal; Signal
is a separate verifier instance emitting the same shape of
replay-certified utility rows.

## Extension

Adding a new ranker domain is a header edit:

1. Add `NN_SCORER_DOMAIN_<NEW>_ACTION` (or `_RULE_PROPOSE`) to
   `NNScorerDomain` in `nn_core.h`.
2. Assign a domain-tag slot in `NNScorerEnvelopeSlot`.
3. If the domain introduces new typed values, extend
   `NNProgramValueType` in `nn_program.h`.
4. Register the domain's verifier hooks against
   `crlp.protocol.candidate_row.v1` and
   `crlp.protocol.verifier_outcome.v1`.

The trunk does not need to be retrained from scratch; the joint
scorer already handles new domains via the role + domain tags. New
programs accrue through the existing library promotion path.

## Gaps relative to the paper contract

Two substrate primitives sit in `main` without a home in
`paper/claims.yaml`'s C1–C9 ladder:

- **Role-tagged scorer envelope** (`b7ebf79`). A method-level claim
  ("one trunk scores both actions and rules across domains via role +
  domain tags") is not asserted. Closest fit is C4 (shared scoring)
  but not specialized there.
- **Unified verifier protocol** (`ec7fb3e`). Five schemas
  constituting the substrate membrane. Currently silent in the paper.
  Candidates: substrate sub-claim under C5, or its own
  framework_claim alongside C8 (paper compiler enforces code-to-PDF
  claim support).

Both are orphan relative to the published contract. The freeze on
`experiment_families_frozen: true` is keeping them quiet; on the
substrate's own terms they are stronger than several of the existing
sub-claims.

## Scope

This document describes only what the code already constitutes.
Forward-looking architecture proposals live in
`docs/self-authoring-paper-roadmap.md`. Paper-supporting experiment
specs live in the `docs/experiment-*.md` family. Specific game or
product roadmap material does not live in this repo. This document
is for orientation, not direction.
