# Experiment Proposal: Library Promotion From Certified Hypotheses

Status: implemented as `scripts/certified_hypothesis_library.py`, an
`arc_agi --hypothesis-library-prior` feedback path, and a tight-capacity
ablation target.

## Purpose

The hypothesis layer currently compiles individual slot bindings into certified
program rows. This experiment tests the next abstraction step: when a template
is repeatedly certified across tasks with zero false rejections, promote the
pattern into an explicit library-entry candidate.

## Hypothesis

Repeated certified hypothesis rows contain enough structure to separate
invariant slots from variable slots. That separation can define a reusable
library entry without hand-writing a new grammar primitive.

## Protocol Shape

| Field | Instantiation |
| --- | --- |
| Candidate set | `crlp.hypothesis.evidence.v1` rows |
| Verifier | Domain certificate checker that produced each evidence row |
| Certificate/gate | Certified status, proof kind, reason, and zero false rejections |
| Scorer role | Promoted entries seed the ARC hypothesis slot prior before proposal ranking |
| Trace artifact | `crlp.hypothesis_library.promotion.v1` plus `crlp.hypothesis_library.prior.v1` TSV rows |
| Protocol point | Repeated successful slot bindings become explicit library objects |

## Implementation

The Make target runs the ARC induction fixture suite, captures hypothesis
selection and evidence rows, then groups certified evidence by:

- template domain,
- template id,
- template name,
- proof kind, and
- certificate reason.

For each group, the script emits:

- support task count,
- covered event count,
- false-rejection total,
- invariant slots,
- variable slots, and
- a promotion decision.

For promoted groups, the script also emits certified slot-prior rows keyed by
template id, slot name, slot type, and slot value. The ARC loader can ingest
those rows with `--hypothesis-library-prior`; it seeds the same slot log-odds
tables used by the online hypothesis prior, so library promotion now affects
proposal scores before top-K coverage selection.

The ARC loader exposes the split and capacity knobs needed to check whether the
fixture suite saturates too quickly:

- `--task-ids` restricts promotion and evaluation to explicit task-id splits.
- `--program-budget-tight-grid` evaluates budgets 1, 2, 4, and 8.
- `--hypothesis-top-k` caps hypothesis promotion before program ranking.
- `--disable-hypothesis-neural-prior` leaves only certified slot counts.
- `--hypothesis-prior-hidden 0` runs a linear hypothesis prior.
- `--hypothesis-prior-hidden 4` runs a tiny MLP.
- `--disable-hypothesis-cold-start` removes the learned object/relation
  cold-start score from the hypothesis slot ranking.
- `--disable-hypothesis-coverage-selection` removes greedy test-input change
  coverage from hypothesis selection.
- `--disable-hypothesis-structural-score` removes the hand structural
  hypothesis score so prior capacity is visible.
- `--disable-hypothesis-train-filter` ranks inconsistent proposals too, then
  lets the downstream verifier reject them.

Run:

```sh
make hypothesis-library-promotion
make hypothesis-library-ablation
make hypothesis-library-split-sweep
```

The ablation target promotes the library prior from
`move_to_anchor,move_to_anchor_x`, then evaluates on the remaining fixture tasks
with top-k 1, coverage selection off, structural hypothesis score off, and
pre-ranking train filtering off. This leaves inconsistent proposals in the
ranked set so the downstream verifier, not the proposal stage, exposes mistakes.

The split-sweep target repeats that stricter protocol across every two-task
`move_to_anchor` source split. It writes aggregate transfer, decoy pressure, and
per-task transfer tables so the result can distinguish reusable slot transfer
from a lucky single split.

The split-sweep runner now also emits the paper-facing held-out result surface:
`crlp.hypothesis_library.heldout_result.v1` rows compare every prior mode
against the no-prior tight-budget baseline after promotion on the source split.
Use `--source-task-ids` and `--heldout-task-ids` to turn the fixture pair sweep
into a fixed train/test split without changing the evaluator schema.
For public ARC data, `make hypothesis-library-public
HYPOTHESIS_LIBRARY_PUBLIC_ARC_DIR=/path/to/ARC-AGI/data` promotes on the public
training split and evaluates the prior on the public evaluation split while
preserving the same held-out result schema.

Outputs:

- `build/float/hypothesis_library/arc_hypothesis_raw.jsonl`
- `build/float/hypothesis_library/promotions.jsonl`
- `build/float/hypothesis_library/library_prior.tsv`
- `build/float/hypothesis_library/arc_hypothesis_library_prior_raw.jsonl`
- `build/float/hypothesis_library/ablation_*_raw.jsonl`
- `build/float/hypothesis_library/summary.md`
- `build/float/hypothesis_library_split_sweep/summary.md`
- `build/float/hypothesis_library_split_sweep/split_sweep_summary.jsonl`
- `build/float/hypothesis_library_split_sweep/split_sweep_tasks.jsonl`
- `build/float/hypothesis_library_split_sweep/heldout_result.jsonl`
- `build/float/hypothesis_library_split_sweep/heldout_result.csv`

## Promotion Rule

A group is promoted when:

- it appears in at least `--min-support-tasks` task instances,
- total false rejections are zero, and
- covered events meet `--min-covered-events`.

The default support threshold is two tasks. That is intentionally low for the
fixture suite; public-corpus promotion should raise the threshold.

## Success Criteria

The experiment is useful if it produces at least one promoted library candidate
with:

- multiple supporting task instances,
- zero false rejections,
- non-empty variable-slot structure, and
- a stable library id whose slot priors are loaded by the proposer feedback run.

The capacity ablation is useful if the no-prior, slot-only, linear, tiny-MLP,
full-MLP, and full-MLP-plus-cold-start rows do not all saturate at the same
tight budgets. If they do, the fixture is too easy or still leaking too much
information through structural proposal generation.

## Failure Modes

- Certified hypotheses are too sparse to promote.
- Slot values do not cluster into useful invariants and variables.
- Promotion duplicates hand-authored templates without improving budgeted
  proposer behavior.
- A promoted prior overfits the fixture suite and needs a higher support
  threshold before public-corpus use.

Fixture-scale feedback is still a smoke test. Public-corpus promotion should
raise the support threshold and evaluate the library-prior run on a held-out
task split.

## Paper Use

If positive, this is a strong NeurIPS-facing result because it shows the audit
substrate enables a new learning loop: certificates produce reusable library
objects, not only per-task evidence rows.
