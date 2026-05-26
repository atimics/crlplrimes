# Grounding Operators

The shared architecture is broader than exact verification. Each experiment
declares a domain-specific grounding operator: the mechanism that turns a
candidate into an outcome row the neural scorer may learn from and the claim
compiler may audit.

```text
state/data -> candidate grammar -> candidate set
           -> grounding operator -> outcome/certificate/proxy row
           -> verifier feedback signal -> proposer/ranker training row
           -> neural scorer/ranker -> trace -> claim gate
```

The grounding operator is the authority boundary. It may be a proof checker, a
simulator, a corpus-derived proxy, or a human review step, but the authority
level must be declared before a result is promoted into the paper. The current C
schema still uses the compatibility names `exact_verifier`,
`verifier_feedback`, `verifier_result`, and `verifier_cost`; this note defines
how to scope those verifier-named rows when the authority is replay, proxy, or
review rather than proof.

## Authority Levels

| Level | Name | Grounding operator | Examples | Supported claim shape |
| --- | --- | --- | --- | --- |
| G1 | exact proof/check | Deterministic verifier can decide the candidate outcome for the declared support. | `prime_gap`, `grid_path`, certified ARC transform rows, typed program synthesis examples | correctness, zero-false-rejection gates, exact residual ranking metrics |
| G2 | deterministic simulation/replay | Authoritative simulator or replay can decide candidate consequences. | Signal flight/mining counterfactual replay, delayed-claim hidden-outcome simulator | replay-certified utility/safety metrics under declared simulator state distribution |
| G3 | corpus proxy | Deterministic corpus-derived grammar or corruption benchmark creates proxy outcomes. | haiku word-adjacency and real-vs-corrupt understanding prototypes, public ARC coverage surfaces | corpus-fit, near-miss discrimination, evidence-surface coverage, not semantic truth |
| G4 | human review | Human judgment accepts/rejects proposed edits, preferences, or interpretations. | editorial-agent PRs, future preference rows | reviewable assistance or preference-labeled evidence, not autonomous claim authority |

These levels are not a value ranking. They are a scope ranking. A G3 corpus
proxy can be useful, but it cannot support the same language as a G1 proof or
G2 simulator replay.

## Required Row Fields

Every grounding-backed experiment should expose these fields directly or through
the current verifier-named compatibility fields:

- `grounding_operator`: the named checker, simulator, corpus proxy, or review
  process.
- `grounding_authority`: one of `exact_check`, `deterministic_replay`,
  `corpus_proxy`, or `human_review`.
- `candidate_grammar`: how candidates are enumerated.
- `outcome_row_schema`: the JSONL schema emitted by the grounding operator.
- `trace_schema`: the row schema for scorer decisions.
- `claim_scope`: the strongest wording permitted by the authority level.
- `known_non_claims`: common overclaims the row must not support.

## Verifier Feedback Signal

The shared feedback primitive is `crlp.protocol.verifier_feedback_signal.v1`.
It is the stable adapter between an authority outcome and the next proposer
training pass:

```text
crlp.protocol.verifier_outcome.v1
  -> crlp.protocol.verifier_feedback_signal.v1
  -> proposer/ranker training rows
```

The feedback row carries `grounding_authority`, `outcome_status`,
`reason_code`, optional `residual_class`, utility/cost/support counts, a
`training_signal` label, and a `target_role` such as `proposer` or `ranker`.
The row may shape future candidate ordering, but it does not promote a
candidate by itself. Promotion still requires the original domain authority:
exact verifier, deterministic replay, corpus proxy, or review gate.

This makes rejected-candidate reason learning and certified hypothesis-library
feedback instances of the same loop instead of separate substrate prototypes.
Static proposers may remain baselines, but paper-facing proposer claims should
compare against the authority-fed feedback version when such rows exist.

## Verified Self-Play

An exact verifier can also run in generator mode. The shared contract is:

```text
primitive grammar -> forward composition -> known-correct task/output
                  -> verifier-shaped accepted/rejected rows
                  -> inverse proposer training
                  -> public verifier replay before coverage claims
```

`make arc-verified-self-play` is the current ARC instance. It composes visual
primitives forward, records accepted programs whose outputs are known by
construction, and emits negative siblings with the same rejection-reason fields
used by the ARC visual-pipeline ranker. It exports only the compiler-backed
accepted rows as an `arc_sketch_program` prior and reports unmapped generated
primitive families separately. The row is exact supervision for inverse program
induction, but not a public ARC score. ARC coverage claims still require
`make arc-verified-self-play-public`, which downloads/validates the official
ARC-AGI-1 public JSON corpus when needed, replays the prior through the ARC
verifier, and compares it against full-program and matched visual-only no-prior
public replays, the declared template baseline, oracle ceiling, and
miss-taxonomy families. Runtime compiler traces stay inside that verifier
boundary: the reference-recolor mask compiler currently promotes only after it
emits a visual-combinator sketch and the public replay certifies the task.
The generator prior is an authored prior over primitive pipelines and
miss-taxonomy families, not an authored solution library. The registered
`C5.arc_verified_self_play` claim therefore stays underpowered until fixed-budget
with/without public replay, generated/eval `state_hash` disjointness,
three-seed evidence, and at least one opened zero-coverage family are present.
Synthetic inverse-ranker top-1 remains in-distribution training evidence only.

## Examples

Signal flight is G2. A sampled state enumerates WASD candidates, the Signal
simulator replays each candidate, and the outcome row records utility, regret,
damage, deaths, and replay metadata. The neural scorer can learn to rank under
that replay, but live comparative control claims require a separate active
episode baseline artifact.

Haiku is currently G3. The word-action model uses a dataset vocabulary and
train-split adjacency graph; the understanding model ranks real haiku above
controlled corruptions. The row can support a claim about corpus-grounded
near-miss discrimination. It cannot support a claim that the model understands
haiku semantics unless a stronger topic/coherence proxy or human-reviewed row
is added and scoped accordingly.

Editorial agents are G4. The authority is not the model; it is reviewable human
acceptance through normal PR flow. The supported claim is that editorial
assistance was traceable and review-gated, not that the model authored the
paper.
