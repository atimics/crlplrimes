# Certified Hypothesis Library Promotion

Date: 2026-05-17

This experiment groups certified `crlp.hypothesis.evidence.v1` rows by template, proof kind, and certificate reason. Repeated zero-false-rejection groups become library-entry candidates with invariant and variable slot summaries, then exports certified slot priors that can seed ARC hypothesis proposal ranking.

Prior rows exported: `9`

## Promotion Source Selection Summary

- `hypothesis_cold_start_enabled`: True
- `hypothesis_coverage_selection_enabled`: True
- `hypothesis_neural_prior_enabled`: True
- `hypothesis_prior_hidden`: 24
- `hypothesis_slot_prior_enabled`: True
- `hypothesis_structural_score_enabled`: True
- `hypothesis_top_k`: 32
- `hypothesis_train_filter_enabled`: True
- `library_prior_entries`: 0
- `library_prior_seeded_certified`: 0
- `library_prior_seeded_proposals`: 0
- `library_prior_slot_rows`: 0
- `proposals_attempted`: 38
- `proposals_generated`: 2
- `proposals_selected`: 2
- `proposals_train_consistent`: 2
- `proposals_train_rejected`: 0
- `selection_rows`: 2

## Capacity Ablation Check

| Run | Mode | Top-k | Prior rows | Seeded cert | Selected | Learned certified @1 | @2 | @4 | @8 | Learned utility @8 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no_prior | `slot+mlp24` | 1 | 0 | 0 | 8 | 2 | 2 | 2 | 2 | 0.2119 |
| slot_only | `slot` | 1 | 9 | 16 | 8 | 4 | 4 | 4 | 4 | 0.5169 |
| linear_prior | `slot+linear` | 1 | 9 | 16 | 8 | 4 | 4 | 4 | 4 | 0.5169 |
| tiny_mlp4 | `slot+mlp4` | 1 | 9 | 16 | 8 | 4 | 4 | 4 | 4 | 0.5169 |
| full_mlp24 | `slot+mlp24` | 1 | 9 | 16 | 8 | 4 | 4 | 4 | 4 | 0.5169 |
| full_mlp24_cold | `slot+mlp24+cold` | 1 | 9 | 16 | 8 | 4 | 4 | 4 | 4 | 0.5169 |

## Promotion Candidates

| Template | Promoted | Tasks | Covered events | Variable slots | Library id |
| --- | --- | ---: | ---: | ---: | --- |
| move_to_anchor | yes | 2 | 78 | 1 | `arc_agi:move_to_anchor:library:2x:78` |

## Trace Contract

- Candidate set: certified hypothesis evidence rows.
- Verifier: domain certificate checker that produced the source evidence.
- Certificate/gate: zero false rejections plus proof-kind and reason agreement.
- Scorer role: promoted entries seed the ARC hypothesis slot prior before proposal ranking.
- Trace artifact: `crlp.hypothesis_library.promotion.v1` rows plus `crlp.hypothesis_library.prior.v1` TSV rows.
- Protocol point: repeated successful slot bindings become explicit library objects.
