# Modularization Roadmap

This document catalogs extraction targets, dependency graphs, and safety invariants for the planned modularization of the CRLP build. No extraction is performed in the current phase; this roadmap guides incremental execution.

## Goals

* Shrink `src/nn.c` (~19,500 lines) into focused translation units.
* Make every extraction a drop-in substitute: external callers do not change API or behavior.
* Keep CI green at every commit; never merge a split that breaks `make promoted`.

## Extraction Queue

### 1. `src/nn.c` → `nn_blas.c`, `nn_forward.c`, `nn_train.c`, `nn_numeric_certificate.c`

| Module | Lines (est.) | Depends on | Safety invariant |
|--------|-------------|------------|-----------------|
| `nn_blas.c` | ~120 | `<cblas.h>` or `<Accelerate/Accelerate.h>` (guarded) | `nn_blas_gemv` / `nn_blas_gemm` produce bit-identical outputs to scalar fallback within epsilon. |
| `nn_forward.c` | ~400 | `nn_blas.c`, `<math.h>` | `dense_forward_layer` and `dense_forward_layer_batch` produce the same activations as the monolithic build for all activation types (SIGMOID, RELU, TANH, LINEAR, EML, EML_PAIRWISE). |
| `nn_train.c` | ~800 | `nn_forward.c`, `nn_blas.c` | Backprop gradients agree within `1e-6` (float) / `1e-12` (double) with the monolithic build on a fixed-seed XOR fixture. |
| `nn_numeric_certificate.c` | ~200 | `nn_forward.c`, `nn_train.c` | Determinism gates pass on the registered seed set; no numeric regression across the split boundary. |

**Order**: `nn_blas.c` first (already prepared via NN_BLAS wrappers), then `nn_forward.c`, then `nn_train.c`, then `nn_numeric_certificate.c`.

### 2. `tests/test_nn.c` → tiered test binaries

| Binary | Tests | Dependency notes |
|--------|-------|-----------------|
| `test_nn_blas` | BLAS wrappers, epsilon checks | Only needs `nn_blas.c` |
| `test_nn_core` | Forward, backward, checkpoint, trace | Needs `nn.o` + `nn_forward.o` + `nn_train.o` |
| `test_nn_experiment` | Experiment, symbolic, bitgrid, tilebot, layer, program, hypothesis, protocol | Needs the core library objects |
| `test_nn_grammar` | Grammar discovery, paper claims | Needs `nn_grammar_discovery.o`, `nn_paper.o` |

### 3. `Makefile` → `mk/core.mk`, `mk/promoted.mk`, `mk/sandbox.mk`, `mk/paper.mk`

| Makefile fragment | Contents | Must preserve |
|-------------------|----------|---------------|
| `mk/core.mk` | Object list, compilation rules, `make core`, test runner build | `NN_BLAS` detection, -I flags, REAL selection |
| `mk/promoted.mk` | Prime gap, delayed claim, grid path, modular gate search binaries | Builds without `../signal` |
| `mk/sandbox.mk` | ARC, haiku, hypothesis library, neural grammar, HOLO targets | Optional; gated by `WITH_EXTENDED=1` |
| `mk/paper.mk` | Paper claims, support report, bibliographic targets | Must regenerate `paper/generated/*` |

### 4. `examples/arc_agi.c` → `src/arc/*.c`

| File | Purpose |
|------|---------|
| `src/arc/arc_loader.c` | ARC task file I/O, grid parsing |
| `src/arc/arc_candidate.c` | Candidate generation and validation |
| `src/arc/arc_ranker.c` | Scoring and ranking |
| `src/arc/arc_visual.c` | Visual program synthesis (sketch) |
| `src/arc/arc_hole_model.c` | ARC hole model (frontier) |
| `src/arc/arc_selfplay.c` | Self-play orchestration |

## Ground Rules

1. **Every split is a PR**. No batch-splitting; each extraction is its own commit with a focused CI run.
2. **No header reorganisation during extraction**. Move `.c` files first; headers stay in `include/` until the split is stable.
3. **Test before and after**. Run `make test` on both the pre-split and post-split builds; results must match.
4. **Keep the monolithic `all` target working**. The `all` target continues to build everything; new targets (`core`, `promoted`) are additive.
5. **Signal guards stay**. `WITH_SIGNAL` detection is already in place; new `mk/sandbox.mk` must not reintroduce hard Signal dependencies.

## Status

**Phase 1 (current)**: P0 build-level fixes — source-policy-check, portable BLAS, tiered targets, Signal guards, evidence-origin metadata. ✅

**Phase 2 (next)**: Extract `nn_blas.c` and `nn_forward.c`. Create `mk/core.mk`.

**Phase 3**: Extract remaining `nn_*.c` modules. Create tiered test binaries.

**Phase 4**: Split `arc_agi.c`. Create `mk/sandbox.mk` and `mk/paper.mk`.

**Phase 5**: Fuzzing harness, CI hardening, cross-platform verification on Ubuntu 24.04.
