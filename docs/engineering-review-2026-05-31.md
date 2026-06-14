# Engineering Review — crlplrimes

**Date:** 2026-05-31
**Reviewer:** Claude (Opus 4.8), multi-agent review + manual verification + build
**Snapshot reviewed:** HEAD `8a5422b` ("continuous-learning GTP engine for CGOS", 2026-05-31 10:22).

**Change set under review (all from 2026-05-31):**
- `89ccb0d` — "sparse verified MCTS Go engine with GTP protocol": new `examples/go_selfplay.c` (824 lines), `nn_dataset.c`/`nn_dataset_music.c`/`nn_paper.c` additions.
- `8a5422b` — "continuous-learning GTP engine": new `examples/go_gtp.c` (534 lines).
- **Uncommitted working tree:** further edits to `go_gtp.c`, `go_selfplay.c`, `dataset_action_score.c`, `nn_dataset.c`, and a large `nn_dataset_music.c` refactor (inline data → `#include nn_dataset_music_corpus.inc`).
- **Untracked at that snapshot:** `src/nn_dataset_proof.c` (new), `nn_dataset_music_corpus.inc` (1030-line generated data), then-external web harnesses, and `scripts/sensor_transfer_proof_grid.sh`.

> **Verification legend:** **[verified]** = I confirmed it by reading the current code directly. **[reported]** = surfaced by a review agent, mechanism is plausible and consistent with surrounding code, but not independently traced end-to-end — confirm before acting.

**Build status:** Both `go_gtp` and `go_selfplay` compile cleanly under the default `-std=c11 -Wall -Wextra -Wpedantic -O3`, with only two warnings: unused variable `c` (`go_selfplay.c:101`) and unused `err` (`go_selfplay.c:680`).

---

## Executive summary

The two new Go-engine files are well-structured and **memory-safe for the 9×9 case** in the important sense: no confirmed use-after-free, double-free, or board out-of-bounds (array bounds hold; tree recursion is depth-bounded ≤81; `kids[]` is capped). However, the review found:

1. **Two network-reachable uninitialized-stack-read bugs** in the GTP command dispatcher (`go_gtp.c`), trivially triggerable from the bundled Python servers.
2. **A cluster of correctness bugs that undermine the headline "verified" and "continuous-learning" claims** — most importantly a Zobrist hash that omits side-to-move (transposition table serves the opponent's policy/value), a likely-inverted MCTS backprop sign, a training loop whose game-outcome reward is hardcoded to `0.0`, a replay buffer that stops learning after it fills, and an "adaptive visit budget" that is computed and then discarded.
3. **Two `nn_paper.c` claim-evaluator bugs** that make the new `sensor_transfer` claim evaluate on fabricated/zero data or never match at all — i.e. the paper's generated claims for this experiment are not actually backed by the data.

| Severity | Count | Theme |
|---|---|---|
| 🔴 Critical | 3 | Uninit stack reads from network input; Zobrist omits side-to-move |
| 🟠 High | 4 | Unchecked alloc; NN data race; mislabeled suicide/self-atari; claim evaluator parses numbers as strings |
| 🟡 Medium | ~7 | Backprop sign, replay non-ring, lock-held-across-epoch, overloaded head, RAVE perspective, proof label contract |
| ⚪ Low | several | GTP 'I'-skip non-conformance, hardcoded outcome, unimplemented budget, dead code |

**My blunt take:** the engine *runs* and is memory-safe enough to not crash on well-formed self-play, but the specific mechanisms the commit messages advertise — "sparse **verified** MCTS", "**continuous learning**" — are each broken by at least one of the bugs below. If these binaries are feeding the paper's claims, the claims for the Go/sensor-transfer experiments should be treated as unsupported until C3, M5 (selfplay), the GTP outcome/replay bugs, and the two `nn_paper` evaluator bugs are fixed.

---

## 🔴 Critical

### C1. Uninitialized stack buffers read in GTP dispatch **[verified]**
**File:** `examples/go_gtp.c:527–528`
```c
char cmd[64], a1[64], a2[64];
int n = sscanf(p, "%63s %63s %63s", cmd, a1, a2);
```
`sscanf` only writes the buffers it matches and does not NUL-terminate the rest. On a line with `n < 1` tokens, `cmd` is fully uninitialized and every subsequent `strcmp(cmd, ...)` reads indeterminate stack memory (UB — may match garbage or crash). At the reviewed snapshot, external harnesses forwarded HTTP request data verbatim, so this was remotely reachable.
**Fix:** `cmd[0] = a1[0] = a2[0] = '\0';` before `sscanf`, and `if (n < 1) { gtp_out("? empty"); continue; }`.

### C2. `boardsize` and `kgs-genmove_cleanup` use args with no arity check → uninitialized read **[verified]**
**File:** `examples/go_gtp.c:534, 540`
```c
else if (!strcmp(cmd, "boardsize"))            gtp_boardsize(atoi(a1));   // no n check
...
else if (!strcmp(cmd, "kgs-genmove_cleanup"))  gtp_genmove(a1);           // no n check
```
`play` (`:536`, `n>=3`) and `genmove` (`:537`, `n>=2`) are correctly guarded, but these two are not. Sending bare `boardsize\n` runs `atoi` on uninitialized `a1`; bare `kgs-genmove_cleanup\n` passes uninitialized `a1` into `gtp_genmove`, which dereferences `color[0]`. Compounds C1.
**Fix:** require `n >= 2` for both before dispatch.

### C3. Zobrist hash omits side-to-move → transposition table returns the opponent's policy/value **[verified]**
**File:** `examples/go_selfplay.c:51, 77–81, 207, 210`
`zobrist[CELLS][3]` is keyed only by position × stone color. `go_update_hash` XORs only the placed/removed stone, and `go_move`'s pass branch (`:207`) flips `to_play` **without touching `zhash`**. So a position with identical stones but opposite side-to-move hashes identically. The TT (`tt_store`/`tt_lookup`) keys purely on this hash, so Black-to-move and White-to-move positions collide and the search retrieves the **wrong player's** cached policy and value. This directly violates the "verified" guarantee and corrupts search.
**Fix:** add a `zobrist_side` constant and XOR it into `zhash` on every `go_move` including pass; fold it into `go_hash`. (Also see M3 — seed `zhash` nonzero so the empty board doesn't hash to 0.)

---

## 🟠 High

### H1. Unchecked `calloc` in MCTS node allocation → null deref on OOM **[reported]**
**File:** `examples/go_selfplay.c:423–427` (`gn_alloc`)
`GoNode *n = calloc(1, sizeof(GoNode)); n->move = ...;` dereferences `n` with no null check. Across `MCTS_SIMULATIONS` × ~30 nodes/expansion × 150 moves × 100 games × 3 epochs, an allocation failure is plausible and crashes immediately.
**Fix:** check `n`; abort the simulation (return NULL / pass) and propagate.

### H2. Data race: training thread mutates the NN while `genmove` reads it **[reported]**
**File:** `examples/go_gtp.c:357–387` (`train_loop`) vs `:456` (genmove path)
`train_loop` calls `nn_train_sample(train_net, …)` (writes weights) concurrently with `nn_forward(gtp_net, …)` during `genmove` — and `gtp_net == train_net`. The mutex protects only the replay buffer, **not the network**, so the weight arrays are read and written concurrently → UB, torn reads, possible crashes inside the NN lib. Fires on live CGOS whenever a training cycle overlaps a move.
**Fix:** guard NN access with an rwlock, or double-buffer (train a copy, atomically swap the pointer), or pause training during `genmove`.

### H3. `--load` accepts an unvalidated checkpoint from a world-writable path **[reported]**
**File:** `examples/go_gtp.c:496–508` (and CLI load in `dataset_action_score.c`)
On `--load`, the loaded net replaces `gtp_net`/`train_net` with no check that its input dim is `FEATURES` (243) and output is 1. The harnesses load `/tmp/go_seed.nnckpt` (world-writable `/tmp`). A mismatched checkpoint makes `nn_forward(net, feats /*243-wide*/, …)` read/write past the net's real layer buffers → OOB.
**Fix:** after load, verify `nn_input_size(net)==FEATURES && nn_output_size(net)==1`; reject and fall back otherwise.

### H4. `nn_paper` sensor-transfer evaluator parses numeric metrics as strings → always 0.0 **[verified]**
**File:** `src/nn_paper.c:4361–4366`
The new evaluator filters rows on the literal `"\"schema\":\"crlp.dataset_action_score\""` and reads `test_top1` via `paper_json_dup_string_value()`. But (a) `test_top1` is emitted as a bare JSON **number** (`dataset_action_score.c:362`), and `dup_string_value` returns `""` for non-`"`-prefixed values → `strtod("")` = **0.0** for every metric; and (b) `dataset_action_score.c` emits a **dynamic** `evidence_schema` (`= experiment->evidence_schema`, typically `crlp.engine_evidence.v1`), not the literal string matched here — so the filter never matches the real rows, and it reads `transfer_comparison.jsonl` while the new `sensor_transfer_proof_grid.sh` writes `transfer_proof_grid.jsonl`. Net: the claim either resolves on all-zeros or never fires. **The paper's sensor-transfer claim is not backed by data.**
**Fix:** match the real schema string and file; parse `test_top1` with `paper_json_get_double(line, "test_top1", &v)` (already defined at `:551`). Keep `dup_string` only for the genuinely-string `baseline` field.

---

## 🟡 Medium

- **M1 (selfplay backprop sign likely inverted).** `go_selfplay.c:465–468`: `v = sign(score); v *= node->to_play;` where `node->to_play` is the side **to move at the leaf**, then the backprop loop negates per level (`:560`). The leaf value should be from the perspective of the player who just moved *into* the leaf (`-node->to_play`). A one-ply sign flip makes MCTS prefer losing lines. **Verify with a 2-ply unit test** (a forced capture should raise the capturing move's Q). [reported]
- **M2 (suicide vs self-atari mislabeled).** `go_selfplay.c:215, 221, 329`: `is_self_atari` actually returns the 0-liberty (suicide) condition. Used correctly for legality (`:221`), but the "danger discount" / tactical heuristics (`:329`) that intend "group reduced to 1 liberty" therefore never fire. Split into `is_suicide` (0 libs) and a true `is_self_atari` (1 lib after placement, no capture). [reported]
- **M3 (empty board hashes to 0, collides with empty TT slots).** `go_selfplay.c:388, 393`: an empty `TTEntry` has `hash==0`, and the initial board's `zhash` is 0 (zeroed by `go_init`). Add a nonzero seed or a `valid` flag. [reported]
- **M4 (overloaded policy/value head; board planes stomped).** `go_selfplay.c:323, 628, 637`: `mf[FEATURES-1]`/`feats[FEATURES-1]` (a board one-hot plane for cell 80) is overwritten with a move-id/reward scratch value, and a single scalar net output serves as both per-move policy score and position value. Corrupts the last intersection's encoding. Add dedicated feature slots. [reported]
- **M5 (replay buffer is not a ring → learning stops after 50k).** `go_gtp.c:332–340, 369`: `replay_add` drops everything once `replay_count == REPLAY_BUFFER`, and `train_loop` always scans `replay[0..n)` from the front — so after the buffer fills, the net re-trains on the first 50k positions forever and ignores all new games. Use a ring buffer + random minibatch. [reported]
- **M6 (training holds `replay_lock` across the whole epoch loop).** `go_gtp.c:365–375`: `replay_add` (called from `gtp_clear_board`) blocks while training holds the lock for two full epochs over up to 50k samples — can stall the GTP response past the controller timeout. Snapshot under a brief lock, train unlocked. [reported]
- **M7 (proof experiment label contract wrong).** `nn_dataset_proof.c:198` + `nn_dataset.c` proof entry: registered `NN_DATASET_LABEL_SINGLE_TARGET`, but the target heuristic produces **multiple** positive targets per group for all current seeds (60/60). Trainer tolerates it (`1/target_count` weighting) so no crash, but the emitted `target_label_contract` is misleading. Declare `MULTI_TARGET` or tighten the heuristic. [reported]

---

## ⚪ Low

- **L1. GTP coordinate non-conformance (missing 'I' skip).** `go_gtp.c:427` (parse) and `:469` (emit), same in `go_selfplay.c:833`. Standard GTP omits column letter 'I'; this engine uses straight `'A'+x`, so incoming `I` maps to the wrong column and emitted moves use `I` — real CGOS/OGS controllers will misparse and desync. Buffers themselves are safe. [reported]
- **L2. Continuous-learning reward is hardcoded neutral.** `go_gtp.c:410–413`: completed-game samples are stored with `outcome = 0.0f` (the per-color sign flip is a no-op), and there's no `final_score` handling — so the value head trains toward 0 regardless of win/loss. The advertised "continuous learning" does not actually learn from game results. Capture the real outcome before `clear_board`. [reported]
- **L3. "Adaptive visit budget / confidence gap" advertised but not implemented.** `go_selfplay.c:579, 589`: `budget` and `second_p` are computed then `(void)`-discarded; the search always runs a fixed `MCTS_SIMULATIONS`. [reported]
- **L4. RAVE statistics mix perspectives and aren't reset across games.** `go_selfplay.c:362–373, 554`: AMAF wins are credited from the unwinding ply's perspective, not the mover's, and `go_match`/`gtp` never call `rave_reset` between games — effectively noise. [reported]
- **L5. `z_init` reseeds `rand()` with `srand(42)`**, potentially clobbering `main`'s `srand(time(NULL))` and making play deterministic per process. `go_selfplay.c` Zobrist init. Use a private RNG for Zobrist. [reported]
- **L6. Network harnesses bind `0.0.0.0`** and forward client-supplied move lists verbatim as GTP commands — this is what makes C1/C2 remotely reachable. Keep network orchestration inside the C runtime boundary, bind `127.0.0.1`, and validate commands before dispatch. [reported]
- **L7. Unused variables** `c` (`go_selfplay.c:101`) and `err` (`:680`) — the only compiler warnings. [verified]

---

## ✅ What's solid (checked, no action)

- **No confirmed memory-corruption bugs in the engines.** `go_gtp`: `replay[]` and `game_moves[]` writes are bounds-guarded; `sscanf("%63s")` widths are correct (no token-length overflow); `gtp_out` uses a fixed format string (`vprintf(fmt, ap)` with literal `fmt` at all call sites — **no format-string injection**); MCTS `gn_new`/`gn_free` recursion is balanced and `nkids` is capped at `MCTS_MAX_KIDS`. `go_selfplay`: board neighbor checks guard `in_bounds` + edge tests, `MAX_MOVES == MCTS_MAX_KIDS == 82`, tree freed via `gn_free` after each search; recursion depth ≤ CELLS (81). The coordinate bounds checks in `gtp_play` correctly prevent OOB into `board[]`.
- **Music dataset refactor is clean.** `nn_dataset_music_corpus.inc` defines 512 lines + 512 splits consistently; all corpus values are in `0..35` and bounded by `MUSIC_CORPUS_NOTES[36]` — no OOB, no sample-count mismatch. (One latent edge: `--max-candidates < 36` could make a target unreachable and abort the build; default 36 is safe.)
- **`nn_dataset_proof.c` is memory-safe.** `state[24]` filled to 10; resolvent/clause arrays correctly sized; `calloc` checked in the shared append path; normalizations cast to `double` before integer division. (Only the label-contract issue M7.)

---

## Recommended fix order

1. **C1 + C2** — initialize `cmd/a1/a2` and gate the two un-guarded commands. One small block closes both network-reachable uninitialized-read bugs.
2. **C3** — fold side-to-move into the Zobrist hash (unblocks the "verified" TT correctness; everything MCTS-quality depends on it).
3. **H4 + the nn_paper schema/file mismatch** — otherwise the sensor-transfer paper claims are fabricated/zero. Fix before publishing or citing those claims.
4. **M1 (backprop sign)** — verify with a forced-capture unit test; this gates whether MCTS actually plays to win.
5. **H1, H2, H3** — unchecked alloc, NN data race, checkpoint validation (resilience under live CGOS).
6. **L2 + M5 (outcome=0 and non-ring replay) + L3 (unimplemented budget)** — required for "continuous learning" to mean anything.
7. Remaining Medium/Low cleanup (GTP 'I'-skip, RAVE, head overloading, proof label contract, unused vars).

---
*Generated by automated multi-agent review (3 parallel reviewers across the GTP engine, the MCTS self-play engine, and the dataset/claim-evaluator code) with manual verification of all Critical/High findings against the working tree, plus a clean build of both new binaries.*
