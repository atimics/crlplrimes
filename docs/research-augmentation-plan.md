# Research Augmentation Plan — Holographic Certification, Compositional Synthesis, and Geometric Representation Learning

Date: 2026-06-06 | Status: draft

## 1. Proven Claims

### C1: The Gated Loop Works

Enumeration → certified symbolic gate → neural ranking → verification. This loop
delivers 62/400 (15.5%) exact-match certification on the ARC public training set
under operator-valid semantics (train-consistent, holonet best-pick).

**Evidence:** `examples/arc_agi.c:evaluate_arc_transform_gate_candidates` with
holonet best-pick scoring. 30 → 62 certified tasks after adding 9 operator
families, full composition, and 8× geometric augmentation.

### C2: Holographic Certificate Store Provides Fixed-Dimension Memory

The `HolographicRanker` stores (state, action) pairs via bind and recalls via
unbind in O(dim) space regardless of stored count. LeosEngine provides
self-contained FFT-based VSA operations with zero heap allocation in the hot path.

**Evidence:** `src/nn_holographic_ranker.c`, `src/holographic/leos_engine.c`.
Linked into `arc_agi` binary. ~7k bind/s at dim=8192.

### C3: Rejected Candidates Are Useful Data

The hypothesis library prior (save/restore via `--save-hypothesis-prior`) accumulates
proposal/certified counts per template. Reloading a prior from a 62-certified run
increases train-consistent proposals from 267 to 323.

**Evidence:** 6 templates accumulate 8/8 proposals/certified each per pass.

### C4: Geometric Augmentation Compounds with Induction

8× dihedral augmentation (4 rotations × 2 flips) provides the largest single
coverage boost: 33 → 60 certified tasks.

**Evidence:** `arc_transform_task` applies rot90/flip_h to all grids before evaluation.
400 tasks → 3,200 augmented evaluations per pass.

### C5: Full Composition Unlocks Multi-Step Transforms

Relaxing composition from "at least one operand must be scale" to "any composable
pair" enables ~30 composed solves among the 62 certified.

**Evidence:** `compose_tile_repeat_then_draw_line` (6), `compose_pattern_extend_then_crop` (7).

---

## 2. Architecture Gaps

### G1: No Scale Invariance

A 3×3 shape and a 9×9 version are different inputs to every operator.
**Required:** Scale-space representation or multi-resolution grid encoding.

### G2: No Object-Abstraction Layer

Every operator works on raw pixel grids. Two L-shapes at different positions
require brute-force translation sweeps.
**Required:** Slot-based or holographic object encoding.

### G3: No Learnable Grid Representation

The candidate builder enumerates fixed operator families. The 13 hypothesis
templates cannot generalize from 62 examples.
**Required:** A representation layer that learns from solved tasks.

### G4: The Holonet Is Not in the Hot Path

Certification uses `arc_structural_program_score` (hardcoded heuristic), not
the holographic ranker.
**Required:** Wire HolographicRanker into certification scoring.

---

## 3. Research Threads

### Thread 1: Holographic Grid Encoding

**Goal:** Replace pixel-grid scoring with holographic scene vectors.

**Approach:**
- Encode each non-background cell as: `S_cell = bind(pos_x, bind(pos_y, color_vec))`
- Bundle all cells: `S_scene = bundle(S_cells)`
- Query: `recall = unbind(S_scene, pos_key)` retrieves what's at that position
- Cyclic tile: compare similarity of shifted queries to detect period

**Papers:** Plate 1995 (HRR), Gayler 2003 (VSA), Kanerva 2009 (HDC),
Frady 2020 (Resonator Networks), Kleyko 2022 (VSA survey)

### Thread 2: Scale-Invariant Grid Representation

**Goal:** Detect the same shape at different scales.

**Approach:**
- Multi-scale grid pyramid: downsample 1×/2×/3×/4× via max-pool and average-pool
- Apply operators at each scale, upsample results
- Fourier-domain scaling: FFT → truncate/pad → IFFT (preserves frequency content)

**Papers:** Henriques & Vedaldi 2016 (Warped Convolutions), Chi 2020 (FNO),
DeVries & Taylor 2017 (Cutout), Shorten & Khoshgoftaar 2019 (augmentation survey)

### Thread 3: Object-Centric Scene Encoding

**Goal:** Represent grid as a set of objects, not a pixel array.

**Approach:**
- Extract all connected components (currently only largest)
- Per-component features: bbox, area, centroid, color, shape hash
- Encode each object as VSA vector: `S_obj = bind(shape_key, bind(pos_key, color_key))`
- Operators query the scene via similarity search

**Papers:** Locatello 2020 (Slot Attention), Burgess 2019 (MONet),
Santoro 2018 (Relation Networks), Battaglia 2018 (Graph Networks)

### Thread 4: Learned Program Synthesis from Templates

**Goal:** Train hypothesis templates on solved tasks to generalize to unsolved.

**Approach:**
- Multi-pass training: run 8× augmentation through hypothesis system, save priors
- Cross-task similarity: use holographic scene vectors to propagate successful
  templates from solved to similar unsolved tasks

**Papers:** Lake 2017 (Compositional Generalization), Bunel 2018 (Neural Program
Synthesis), Nye 2021 (Scratchpads)

### Thread 5: Spectral Methods for Grid Transforms

**Goal:** Fourier-domain operations for efficient multi-scale processing.

**Approach:**
- LeosEngine FFT for period detection (find peaks in frequency domain)
- Low-pass filter for scale/blur: zero high frequencies → smaller effective grid
- Cross-correlation via FFT multiplication for shape matching

**Papers:** Gu 2022 (S4), Cooley & Tukey 1965 (FFT)

---

## 4. Phase Plan

### Phase 1: Holographic Grid Encoding (2-3 weeks)

1. Implement `grid_to_hologram`: 30×30 grid → 256-dim scene vector via LeosEngine
2. Holographic cyclic tile detector replacing broken `arc_apply_pattern_tile_candidate`
3. Wire HolographicRanker into certification scoring
4. **Target:** 62 → 68+ certified tasks

### Phase 2: Scale-Space Grid Operations (2 weeks)

1. Multi-scale pyramid: downsample 1×/2×/3×/4× via max-pool/average-pool
2. Apply operators at each scale, upsample predictions
3. **Target:** 29/29 scale-up tasks solved (currently ~15/29)

### Phase 3: Object-Centric Representation (2 weeks)

1. Extract all connected components per grid
2. Encode each component with (position, color, area, aspect ratio, perimeter)
3. Component-aware operators (per-object translate, per-object recolor)
4. **Target:** 3+ per-object tasks solved

### Phase 4: Template Pre-Training Loop (1 week)

1. Multi-pass training: build priors from augmented data
2. Cross-task holographic similarity for template propagation
3. **Target:** 70+ certified tasks

---

## 5. Augmented Bibliography

### Group J: Holographic Representations and VSA

- Plate, T. A. (1995). "Holographic Reduced Representations." *IEEE TNN*.
- Gayler, R. W. (2003). "Vector Symbolic Architectures." *ICCS*.
- Kanerva, P. (2009). "Hyperdimensional Computing." *Cognitive Computation*.
- Frady, E. P. et al. (2020). "Resonator Networks." *Neural Computation*.

### Group K: Geometric Data Augmentation and Invariance

- Shorten, C. & Khoshgoftaar, T. M. (2019). "A Survey on Image Data Augmentation." *J. Big Data*.
- Henriques, J. F. & Vedaldi, A. (2016). "Warped Convolutions." *arXiv*.
- DeVries, T. & Taylor, G. W. (2017). "Cutout." *arXiv*.

### Group L: Object-Centric and Relational Learning

- Locatello, F. et al. (2020). "Object-Centric Learning with Slot Attention." *NeurIPS*.
- Burgess, C. P. et al. (2019). "MONet." *arXiv*.
- Santoro, A. et al. (2018). "Relation Networks." *NeurIPS*.

### Group M: Fourier and Spectral Methods

- Chi, L. et al. (2020). "Fourier Neural Operator." *arXiv*.
- Gu, A. et al. (2022). "S4: Structured State Spaces." *ICLR*.

### Group N: Program Synthesis and Compositional Generalization

- Lake, B. M. et al. (2017). "Building Machines That Learn and Think Like People." *BBS*.
- Bunel, R. et al. (2018). "Neural Program Synthesis." *ICLR*.

---

## 6. Evaluation Protocol

| Phase | Metric | Baseline | Target |
|-------|--------|----------|--------|
| Holo encoding | Certified tasks | 62 | 68+ |
| Multi-scale | Scale-up tasks solved | ~15/29 | 29/29 |
| Object-centric | Per-object translate | 0 | 3+ |
| Template training | Certified tasks | 62 | 70+ |

Runs use `--induction` with 8× augmentation. DreamCoder save/restore carries priors
forward between phases. Each phase is additive on the existing operator set.

---

*SPDX-License-Identifier: MIT. Copyright (c) 2026 Anonymous Authors.*
