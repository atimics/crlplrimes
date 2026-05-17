# ARC Family Discovery

Status: first offline pipeline.

The goal is to let the neural side learn where the symbolic search space should
grow, while the verifier still owns correctness. The loop is:

```text
ARC JSON tasks + induction diagnostics
  -> structural train-pair signatures
  -> recurring cluster keys
  -> symbolic-family stubs
  -> pseudo-labels for a neural family proposer
  -> new candidate generators
  -> exact verifier
```

Run it after `arc_induction_parity.py` has produced the task diagnostics JSONL:

```sh
make arc-family-discovery \
  ARC_AGI_TASK_DIR=/private/tmp/ARC-AGI/data \
  ARC_FAMILY_DIAGNOSTICS=results/2026-05-17-arc-induction-task-diagnostics.tasks.jsonl
```

Or directly:

```sh
python3 scripts/arc_family_discovery.py \
  --task-dir /private/tmp/ARC-AGI/data \
  --diagnostics results/2026-05-17-arc-induction-task-diagnostics.tasks.jsonl \
  --output-dir build/float/arc_family_discovery
```

Outputs:

- `summary.md`: ranked recurring family clusters and template stubs.
- `family_clusters.jsonl`: one row per proposed symbolic family cluster.
- `task_signatures.jsonl`: one structural signature per clustered ARC task.
- `family_labels.jsonl`: pseudo-label rows for a future neural family proposer.

The first public-eval run uses the failed induction buckets
`no_induction_generated,no_induction_attempted`. This is deliberate: the latest
ARC diagnostics show the gap is mostly generation coverage, not ranking. The
largest clusters are currently panel/projection selection, additive object
copy/extension, scale/tile expansion, crop/object extraction, and color rewrite.

The cluster labels are not correctness claims. They are search-space growth
targets. A promoted family is useful only if its generated candidates match
train pairs and increase public ARC certifications with zero regressions.
