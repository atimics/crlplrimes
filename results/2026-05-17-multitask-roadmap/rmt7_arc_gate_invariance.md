# ARC Gate Invariance

Task dir: `/private/tmp/ARC-AGI/data`
Max tasks per split: `400`
Compared contexts: `shared_scorer_context` vs `single_task_scorer_context`.

Current implementation note: the public ARC transform gate and program-proposer path do not consume a multitask scorer checkpoint. The contexts here are therefore an explicit invariance check over the current scorer-independent gate/proposer code path.

| Split | Tasks | Parse errors | Certified tasks | Test-exact tasks | Correct cells | Transform rows | Proposer rows | Budgets |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `train` | 400 | 0 | 24 | 24 | 1187 | 400 | 20 | `8,16,32,64` |
| `eval` | 400 | 0 | 4 | 4 | 187 | 400 | 20 | `8,16,32,64` |

## Result

- exact gate/loader mismatches: `0`
- budgeted proposer aggregate mismatches: `0`
- R-MT.7 holds for the measured code path: accepted rule sets, prediction hashes, loader summaries, and budgeted proposer aggregates are invariant across the two scorer contexts.
