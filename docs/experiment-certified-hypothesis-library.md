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
| Scorer role | Promoted entries seed ARC template and slot priors before proposal ranking |
| Trace artifact | `crlp.hypothesis_library.promotion.v1`, `crlp.hypothesis_library.template_prior.v1`, and `crlp.hypothesis_library.prior.v1` TSV rows |
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

For promoted groups, the script also emits one certified template-prior row and
certified slot-prior rows keyed by template id, slot name, slot type, and slot
value. The ARC loader can ingest those rows with
`--hypothesis-library-prior`; it seeds the same template and slot log-odds
tables used by the online hypothesis prior, so library promotion now affects
proposal scores before top-K coverage selection. Selection traces expose
`library_prior_template_rows`, `template_prior_score`, and
`template_family_prior_source` beside the existing slot-prior counters.

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
- `--disable-hypothesis-train-consistency-prior` removes the verifier-derived
  rank bonus without disabling the downstream verifier.

Even when the train filter is disabled, pre-cap train consistency remains a
rank signal. That keeps the experiment neurosymbolic: the symbolic verifier can
constrain a learned or library prior without hiding verifier failures from the
downstream certificate rows. The split-sweep runner can also use
`--stress-no-prior` to disable that rank signal only for the no-prior arm,
creating a labeled search-pressure baseline rather than changing the normal
evaluator. Verifier-derived train consistency is intentionally stronger than
learned, slot, and template priors in the normal evaluator: the priors should
rank among viable symbolic candidates, not bury candidates that already satisfy
the task's train pairs.

Run:

```sh
make hypothesis-library-promotion
make hypothesis-library-ablation
make hypothesis-library-split-sweep
make hypothesis-library-feedback-loop
make hypothesis-library-public-filter HYPOTHESIS_LIBRARY_PUBLIC_ARC_DIR=/path/to/ARC-AGI/data
make hypothesis-library-public-report
make hypothesis-library-public-triage HYPOTHESIS_LIBRARY_PUBLIC_ARC_DIR=/path/to/ARC-AGI/data
make hypothesis-library-neural-discovery HYPOTHESIS_LIBRARY_PUBLIC_ARC_DIR=/path/to/ARC-AGI/data
make hypothesis-library-neural-discovery HYPOTHESIS_LIBRARY_PUBLIC_ARC_DIR=/path/to/ARC-AGI/data HYPOTHESIS_LIBRARY_NEURAL_DISCOVERY_MUTATION_VERIFY=1
make hypothesis-library-neural-failure-proof HYPOTHESIS_LIBRARY_PUBLIC_ARC_DIR=/path/to/ARC-AGI/data HYPOTHESIS_LIBRARY_NEURAL_DISCOVERY_MUTATION_VERIFY=1
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

The split-sweep rows also carry a discovery surface for ARC family debugging.
Each summary, task, and held-out result row records the source family, promotion
family, held-out family, source templates, promoted templates, held-out selected
templates, novel held-out templates, and whether held-out success reused a
source/promoted template or required a novel symbolic template. The same rows
include proposer/search efficiency counters derived from existing
`crlp.arc_hypothesis.selection.v1` rows: attempted/generated/selected proposal
counts, train-consistency rate, proposal-cap hits, seeded-prior counts, and
per-template search budgets. This keeps the claim neurosymbolic: the report
shows which certified symbolic templates the library discovered, where
held-out families reused them, and where the proposer only succeeded after
selecting a different template.

`make hypothesis-library-feedback-loop` is the first closed-loop controller over
that surface. It runs the split sweep, selects held-out failure rows, adds those
failed tasks back to the source side as feedback evidence, re-promotes certified
hypotheses, and re-evaluates only the still-held-out tasks by default. The
default target now synthesizes two color-remapped sibling fixtures for multiple
families (`bbox_panel_selection`, `connect_aligned`,
`extend_aligned_sequence`, `object_extract`, and `recolor_by_reference`) with
`--generated-siblings-per-family 2`, feeds back up to three selected failures,
and keeps the remaining siblings in the held-out evaluation set. Feedback
selection is family-diverse by default (`--max-feedback-tasks-per-family 1`) so
one failure family cannot consume the whole feedback budget before other
families get a chance to contribute evidence. The Make target runs this with
`--stress-no-prior`, so its summary row reports a hard no-prior search-pressure
baseline and the stricter guard against the original source-prior run. A useful
loop must promote a feedback-supported library and improve at least one
still-held-out task relative to the original source prior; otherwise the row
remains a diagnostic limitation. On the current fixture suite this is now an
explicit prior-search result: stressed no-prior certifies only a small subset,
the source prior recovers the source family, and the feedback library prior
recovers held-out sibling families under no-leak evaluation.

For public ARC data, `make hypothesis-library-public
HYPOTHESIS_LIBRARY_PUBLIC_ARC_DIR=/path/to/ARC-AGI/data` promotes on the public
training split and evaluates the prior on the public evaluation split while
preserving the same held-out result schema.

Before running the full public split, use
`make hypothesis-library-public-filter
HYPOTHESIS_LIBRARY_PUBLIC_ARC_DIR=/path/to/ARC-AGI/data` to construct a
non-saturated source/held-out slice. The filter runs promotion evidence on the
training split, then runs both a default strict no-library pass and a stressed
no-prior pass on the evaluation split. Held-out selection defaults to tasks the
default strict pass did not certify, while still requiring stressed traces to
show train-consistent proposal pressure. That is the framework-specific wedge:
the selected failures are not generic "ARC is hard" examples, they are tasks
where our symbolic verifier already sees viable structure but the current
ranking/search protocol does not yet promote it into the selected hypothesis.
If the current default strict pass already certifies every train-consistent
stress candidate, the filter falls back to stress-no-prior failures rather than
reading stale diagnostics or emitting an empty slice.
`make hypothesis-library-public-report` then buckets the resulting diagnostics
into certified tasks, train-consistent hypotheses that were not selected,
selected hypotheses rejected by the verifier, proposal-cap failures without a
train-consistent hypothesis, and low-signal proposal failures.
`make hypothesis-library-public-triage` joins those diagnostics back to the
public task JSON and labels the next structural family to mine.

The current public-filter run adds two public-facing template families:
`masked_motif_lift`, which lifts a foreground/background mask into a block grid
whose repeated motif is the input grid itself, and `same_grid_color_map`, which
certifies same-shape grids whose cells obey a consistent input-color to
output-color map. The color-map family certifies five public training tasks
(`0d3d703e`, `aabf363d`, `b1948b0a`, `c8f0f002`, `d511f180`) and promotes as a
multi-task library under the default support threshold. With these templates,
the public filter has 15 certified source candidates out of 400 training tasks
and 4 held-out pressure candidates out of 400 evaluation tasks. On the filtered
split, stressed no-prior remains `0/4`, while slot-only, linear, tiny-MLP, and
full-MLP priors select certified hypotheses for `4/4`. This is still narrow
coverage, not broad ARC saturation: 385/400 public training tasks remain
no-meaningful-proposal cases under the current template grammar.

The triage surface says the next coverage work should not be another
color-map pass. Across the 1,181 uncertified public diagnostic rows, the largest
failure wedge is same-shape patch/lattice transforms, followed by
separator-driven panel extraction and expansion/tiling. That is a better next
template target than trying to tune ranking around already-covered color-map
families.

`make hypothesis-library-neural-discovery` is the first hard-stop target for
moving discovery pressure away from handwritten ARC features. It consumes public
task JSON plus verifier diagnostics, encodes only raw padded train
input/output grid cells and mechanical validity masks, trains a small neural
label head from verifier labels, and clusters uncertified failures in the
learned latent space. Handwritten triage labels such as separator, shape,
palette, and patch/lattice categories are report-only and are not model input.
The target writes latent clusters, nearest certified neighbors, and a
verification queue for the next sketch miner. HTM is not part of this path; the
symbolic verifier remains the certification authority, while neural state is
used only to propose which latent failure neighborhoods deserve candidate
sketch synthesis.

The sketch miner now compiles each neural cluster suggestion into executable
symbolic mutations. Each mutation contains an uncertified
`crlp.arc_neural_discovery.template_probe.v1` row for the candidate template
family plus optional `crlp.arc_neural_discovery.slot_probe.v1` rows grouped into
complete slot sketches. Template probes bias a family; slot probes can be loaded
by `arc_agi` as concrete hypothesis candidates as well as slot-level ranking
signals. Neither schema counts as certified library support. With
`HYPOTHESIS_LIBRARY_NEURAL_DISCOVERY_MUTATION_VERIFY=1`, the loop runs each
mutation against representative cluster tasks, compares it with a no-mutation
baseline using the same evaluator mode that produced the diagnostic row
(`promotion`, strict `default`, or strict `stress_no_prior`), and records
whether the verifier produced a novel certification, only a diagnostic
search/ranking improvement, or no gain. This keeps the neural side in the
proposal role and the symbolic verifier in the accept/reject role.

`make hypothesis-library-neural-failure-proof` is the falsification target for
the current "slot decoder quality versus missing template family" question. It
joins mutation-verification rows, a promotion-mode replay of the same
representative tasks, and report-only public triage labels. A row supports a
slot-decoder failure only when neural slot probes are loaded or selected but
produce zero train-consistent mutations. It supports a current-template-family
gap only when the promotion replay produces zero train-consistent proposals and
the public triage label points at a structural family not covered by the current
grammar. Accepted-existing rows are kept as controls and excluded from the
failure proof counts.

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
- `build/float/hypothesis_library_feedback_loop/feedback_candidates.jsonl`
- `build/float/hypothesis_library_feedback_loop/feedback_loop_tasks.jsonl`
- `build/float/hypothesis_library_feedback_loop/feedback_loop_summary.jsonl`
- `build/float/hypothesis_library_feedback_loop/summary.md`
- `build/float/hypothesis_library_public_filter/diagnostics.jsonl`
- `build/float/hypothesis_library_public_filter/selected_tasks.jsonl`
- `build/float/hypothesis_library_public_filter/source_task_ids.txt`
- `build/float/hypothesis_library_public_filter/heldout_task_ids.txt`
- `build/float/hypothesis_library_public_filter/filtered_tasks/`
- `build/float/hypothesis_library_public_filter/summary.md`
- `build/float/hypothesis_library_public_filter/arc_public_diagnostics_report.md`
- `build/float/hypothesis_library_public_filter/arc_public_diagnostics_buckets.jsonl`
- `build/float/hypothesis_library_public_filter/arc_public_diagnostics_buckets.csv`
- `build/float/hypothesis_library_public_triage/coverage_triage.jsonl`
- `build/float/hypothesis_library_public_triage/coverage_triage.csv`
- `build/float/hypothesis_library_public_triage/coverage_triage_summary.md`
- `build/float/hypothesis_library_neural_discovery/samples.jsonl`
- `build/float/hypothesis_library_neural_discovery/embeddings.jsonl`
- `build/float/hypothesis_library_neural_discovery/clusters.jsonl`
- `build/float/hypothesis_library_neural_discovery/candidate_sketches.jsonl`
- `build/float/hypothesis_library_neural_discovery/mutation_priors/`
- `build/float/hypothesis_library_neural_discovery/mutation_verification.jsonl`
- `build/float/hypothesis_library_neural_discovery/mutation_verifier_logs/`
- `build/float/hypothesis_library_neural_discovery/failure_proof/task_proof_rows.jsonl`
- `build/float/hypothesis_library_neural_discovery/failure_proof/promotion_replay.jsonl`
- `build/float/hypothesis_library_neural_discovery/failure_proof/summary.md`
- `build/float/hypothesis_library_neural_discovery/verification_queue.jsonl`
- `build/float/hypothesis_library_neural_discovery/summary.md`

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
