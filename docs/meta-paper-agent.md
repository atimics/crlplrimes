# Meta-Level Paper Agent

The paper-writing loop uses the same verifier-grounded architecture as the
experiments, with an LLM occupying the neural-proposer role.

```text
repository artifacts + source papers + claim registry
  -> LLM proposes paper candidates
  -> claim/source/scope verifiers accept or reject each candidate
  -> accepted candidates may enter paper surfaces
  -> rejected candidates become residual feedback for the next proposal pass
```

The LLM does not own truth. It proposes claim sentences, section drafts,
citations, captions, limitations, and agent briefs. The compiler decides which
candidate surfaces are admissible.

## Trace Schema

`make paper-claims` reads optional LLM-authored proposals from
`paper/agent_candidates/*.jsonl` and emits normalized verifier verdicts to
`paper/generated/paper_agent_trace.jsonl` with schema
`crlp.paper_agent_candidate.v1`.

Required row shape:

```json
{
  "schema": "crlp.paper_agent_candidate.v1",
  "trace_version": 1,
  "candidate_id": "stable-hash",
  "candidate_type": "claim_sentence",
  "proposer_kind": "llm",
  "proposer_model": "model-id-or-null",
  "prompt_hash": "prompt-hash-or-null",
  "proposed_text": "Candidate paper text.",
  "claim_ids": ["C1.prime_gate_delta"],
  "claim_family_ids": ["C1"],
  "paper_surfaces": ["abstract"],
  "source_refs": [
    {
      "path": "results/latest/comparison.canonical.jsonl",
      "exists": true,
      "sha256": "source-hash",
      "row_count": 523
    }
  ],
  "required_for_submission": true,
  "verifier": {
    "name": "paper_claim_compiler",
    "status": "accepted",
    "support_status": "supported",
    "decision_rule": {}
  },
  "verifier_status": "accepted",
  "accepted_surface": "paper",
  "rejection_reason": null,
  "residual_feedback": [],
  "scope_constraints": []
}
```

Current generated candidate types:

| Type | Purpose | Acceptance rule |
| --- | --- | --- |
| `claim_sentence` | A sentence that could enter the paper as a claim. | Accepted only when every referenced claim is supported for that surface. |
| `audit_summary` | Compiler-generated statement of claim state. | Accepted for audit/support surfaces, including negative states. |
| `limitation_candidate` | Compiler-generated limitation text for contradicted or underpowered claims. | Accepted only on limitations surfaces. |
| `agent_brief` | Bounded instruction for future research or editorial agents. | Accepted as roadmap/instruction text, not as evidence. |

LLM-authored input rows must add `proposer_kind: "llm"`, `proposer_model`, and
`prompt_hash`. The compiler ignores any submitted verdict status and re-emits
the row as accepted or rejected according to the current claim registry, source
hashes, row counts, surface permissions, and non-claim scope constraints. The
compiler-generated rows use `claim_registry` or `claim_compiler` as
deterministic stand-ins.

## Verifier Layers

The meta-level checker should stay split into independent gates:

1. Claim support gate: claim ids must exist and be `supported` before the text
   can enter promoted paper surfaces.
2. Evidence gate: every `source_ref` must exist with the recorded hash and row
   count, or the candidate is stale.
3. Surface gate: contradicted and underpowered claims can generate accepted
   audit or limitation text, but not accepted body/abstract claim text.
4. Citation gate: related-work and source-backed prose must name source ids and
   evidence spans before it can leave draft status.
5. Scope gate: proposed text must not assert registered non-claims such as
   general task solving or competitive ARC performance.
6. Revision gate: accepted text should carry a stable `candidate_id` so later
   edits are reviewable as changed candidates, not silent rewrites.

## External Input Gate

The schema now has first-class LLM proposal inputs. The current gate rejects a
candidate with explicit `rejection_reason` values such as:

- `missing_prompt_hash`
- `unknown_claim_id`
- `unsupported_claim`
- `missing_source_ref`
- `stale_source_ref`
- `disallowed_surface`
- `contradicts_non_claim`

The repository includes fixture rows showing both outcomes: a supported
`C1.prime_gate_delta` abstract proposal is accepted, while a positive
`C2.prime_proposer_fixed_budget` claim is rejected because the referenced claim
is currently contradicted.

## Run Cadence

The loop is split so deterministic gates are cheap and LLM proposal is explicit:

| Moment | Command | Purpose |
| --- | --- | --- |
| Proposal pass | `make paper-agent-propose` | Read `paper/generated/paper_agent_trace.jsonl` and write a prompt for the next LLM proposal pass. If `PAPER_AGENT_LLM_CMD` is set, run that command and normalize JSONL candidates into `paper/agent_candidates/llm_proposals.jsonl`. |
| Local commit gate | `make paper-pre-commit` or the tracked `.githooks/pre-commit` hook | Run only `scripts/derive_paper_claims.py` when staged paper claim inputs changed. The hook fails if generated paper artifacts change, so the author reviews and stages them deliberately. |
| Hook install | `make install-hooks` | Set `core.hooksPath=.githooks` for this checkout. |
| PR / CI gate | `make paper-support` and `make paper-ci` | Re-run deterministic support checks, editorial checks, and PDF compilation. |

`paper-agent-propose` is not part of pre-commit or CI. The LLM is a proposer,
not a deterministic verifier.

## Next Tightening

- Require `source_refs` for every numeric, comparative, or citation-like
  sentence. A claim sentence without source refs should be rejected even if the
  wording is plausible.
- Require source span ids for related-work claims, not only bibliography keys.
  This prevents the LLM from attaching a real citation to an unsupported
  contrast.
- Add quote-budget and paraphrase checks for source-derived prose before
  accepting related-work paragraphs.
- Add a diff-level gate: when `paper/paper.typ` changes, every new or changed
  claim-like sentence should map to an accepted `candidate_id`.

The direction is deliberately conservative: make the LLM good at proposing
better candidates, not at bypassing the claim compiler.
