# Next Session Plan — VSA Two-Pass Training

Date: 2026-06-06 | Current coverage: 70/400 (17.5%)

## What's Proved

- VSA save/load works on 10-task runs: save file is 2056 bytes, load pass
  improves cell accuracy from 0.728 to 0.832.
- Stack overflow fixed (`ARCGrid best` moved to static).
- The crash on 400-task runs was stale object files, not logic errors.

## Session Goal

Run the two-pass VSA experiment at scale and measure cross-task transfer.
Target: 75+ certified (18.75%+).

## Step 1: Clean Build + VSA Save/Load Wiring (10 min)

The VSA save/load code is already written and tested on 10-task runs.
Apply it to HEAD, clean rebuild, verify.

Changes needed:
- File-scope `g_vsa_ranker` before cert function
- Config field `vsa_ranker_path`
- `--vsa-ranker` CLI flag
- Load at task loader start (fopen check for existing file)
- Save at task loader end
- Always: `rm -f build/float/arc_agi && make` (stale .o files)

## Step 2: Pass 1 — Train VSA Ranker (2 min)

```
./build/float/arc_agi \
  --task-dir /tmp/arc_extract/ARC-AGI-master/data/training \
  --shuffle --induction --min-induction-evidence 1 \
  --vsa-ranker /tmp/vsa_trained.bin \
  --loader-summary-only
```

Expected: ~70 certified, save file created.

## Step 3: Pass 2 — Evaluate with Loaded Ranker (2 min)

```
./build/float/arc_agi \
  --task-dir /tmp/arc_extract/ARC-AGI-master/data/training \
  --induction --min-induction-evidence 1 \
  --vsa-ranker /tmp/vsa_trained.bin \
  --loader-summary-only
```

Expected: 70-75+ certified. The loaded ranker has cross-task knowledge.

## Step 4: Iterate

If single-epoch doesn't train enough: `--task-epochs 3` on pass 1.
If loaded ranker doesn't improve: increase VSA blend weight from 50 to 200.
If still no gain: try VSA as primary score (replace structural, not blend).

## Success Criteria

| Metric | Current | Target |
|--------|---------|--------|
| Certified tasks | 70 | 75+ |
| VSA two-pass working | No | Yes |
| Runtime (400 tasks) | 35s | <60s |

## Risks

- **Stale object files:** Always clean rebuild.
- **VSA ranker overfitting:** Pass 1 trains on shuffled order. Mitigation:
  multiple shuffle seeds if overfitting detected.
- **Scale-space slowdown:** Fallback plan if VSA doesn't deliver.
