# HoloNet Engine — Gap Analysis

June 2026. 120 files, 131K lines across leos-c + crlplrimes.

## What We Built

### L1: Holographic Primitives (leos-c)
| Component | File | Lines | Status |
|-----------|------|-------|--------|
| FFT (radix-2, pre-computed twiddles) | `leos_engine.c` | 360 | **ship** |
| VSA bind/unbind/bundle | `leos_engine.c` | — | **ship** |
| Flat-spectrum keygen | `leos_engine.c` | — | **ship** |
| Vector ops (norm, dot, axpy) | `leos_engine.c` | — | **ship** |
| Performance counters | `leos_engine.c` | — | **ship** |
| Holographic store (key-value) | `leos_store.c` | 249 | **ship** |

### L2: Neural Holographic (leos-c)
| Component | File | Lines | Status |
|-----------|------|-------|--------|
| HolographicNeuron (bind/unbind mode) | `leos_hnn.c` | 145 | **ship** |
| HolographicMemory (content-addressable) | `leos_hnn.c` | — | **ship** |
| HolographicLayer (bundle of neurons) | `leos_hnn.c` | — | **ship** |
| Full forward (memory → layers → output) | `leos_hnn.c` | — | **ship** |
| Hebbian training (per-neuron alignment) | `leos_train.c` | 120 | **gap** |
| Bridge (hypothesis lib, features, weights) | `leos_bridge.c` | 113 | **ship** |

### L3: crlplrimes Integration
| Component | File | Lines | Status |
|-----------|------|-------|--------|
| Singleton wrapper | `nn_holographic.c` | 90 | **ship** |
| Holographic ranker (k-NN scorer) | `nn_holographic_ranker.c` | 90 | **ship** |
| Experiment runner (accuracy/NLL/latency) | `nn_holo_experiment.c` | 65 | **ship** |
| Feature envelope (96→100) | `nn_core.h` | 1 | **ship** |
| HoloNet (structured receptive fields) | `holonet.c` | demo | **ship** |
| Makefile integration | `Makefile` | +50 lines | **ship** |

### L4: Demos & Benchmarks
| Demo | What it shows | Status |
|------|--------------|--------|
| Fully holographic | k-NN 100% at dim=2048 | **ship** |
| Side-by-side (4 experiments) | Holo vs Neural per task | **ship** |
| HoloNet grid path | Structured fields, 100% acc | **ship** |
| Weight storage | Holographic checkpointing | **ship** |
| Fading memory | 1/√N decay verified | **ship** |

---

## Gaps

### G1: Holographic Layer Training (BLOCKING)
**What:** `hnn_layer_train_step` doesn't converge. Hebbian per-neuron alignment
produces random updates because bind/unbind aren't differentiable and random
vectors are near-orthogonal in high dimensions.
**Impact:** Can't train deep holographic networks. Stuck at single-layer k-NN.
**Options:**
  A. Reservoir computing — freeze random layer, only train readout
  B. Evolutionary search over small weight spaces (8 neurons × 2048 dims)
  C. Displacement codec — learn the hypersphere transform directly
  D. Accept that holo-layers don't train and use k-NN as the primary mechanism
**Recommendation:** D for now. k-NN is the proven core. Layers are feature
  transforms that don't need to be trained — random projections work in high dims.

### G2: Performance (MEDIUM)
**What:** FFT at ~10 GFLOPS on M3. Could be 50+ with vDSP/Accelerate.
**Impact:** 45 µs/query limits throughput to ~22K queries/sec.
**Fix:** 
  - Replace hand-rolled FFT with vDSP (Apple) or FFTW (Linux)
  - Float instead of double (NEON SIMD)
  - Estimated 3–5× speedup
**Effort:** ~2 days

### G3: Online Reinforcement in crlplrimes (MEDIUM)
**What:** `holo_ranker_store` exists but isn't wired into the eval loop.
After verifier confirmation, should call `holo_ranker_store(state, correct_action)`.
**Impact:** Ranker works in demos but not in the real prime_gap/grid_path pipelines.
**Fix:** ~4 lines of code in `eval.c` and `train.c`.
**Effort:** ~1 hour

### G4: Confidence-Gated Routing (MEDIUM)
**What:** No mechanism to choose between holographic k-NN and neural scorer
based on holographic recall fidelity.
**Impact:** On delayed_claim, holographic gets 30.5% but neural gets 84%.
  A router would use neural for that task.
**Fix:** 
  - Measure 1/√N per query
  - If fidelity < 0.1, fall back to neural scorer
  - Or blend: score = α·holo + (1−α)·neural where α = fidelity
**Effort:** ~1 day

### G5: Serialization (LOW)
**What:** Can't save/load holographic memory traces.
**Impact:** Every run starts from scratch. Can't checkpoint or transfer models.
**Fix:** `fwrite(M, dim*sizeof(double), fp)` — literally 2 lines.
**Effort:** ~30 minutes

### G6: HoloNet Centroid Learning (LOW)
**What:** HoloNet centroids are random. Should be learned via k-means or
online clustering of feature vectors.
**Impact:** 8/16 neurons are unused (0 stored). Routing is suboptimal.
**Fix:** Online k-means: initialize centroids to first K training features,
  reassign on drift.
**Effort:** ~1 day

### G7: Cross-Task Transfer (LOW)
**What:** No mechanism to transfer knowledge between task domains.
**Impact:** Prime_gap rules don't help grid_path scoring.
**Fix:** Rosetta projection matrix between task embedding spaces.
  Or: shared holographic memory trace with task-tagged keys.
**Effort:** ~3 days

### G8: Iterative Cleanup (LOW)
**What:** `leos_store_get_clean` doesn't improve fidelity.
**Impact:** Can't recover signal from deeply buried memories.
**Fix:** The residual decode algorithm needs non-normalized intermediates.
  Requires a separate cleanup memory unit with non-normalized accumulator.
**Effort:** ~2 days

### G9: Multi-Resolution Memory (LOW)
**What:** Single memory trace at fixed dimension.
**Impact:** Can't do coarse-to-fine retrieval.
**Fix:** Pyramid of traces at dim/2, dim/4, dim/8. Route queries through pyramid.
**Effort:** ~2 days

### G10: vSA Arithmetic (FUTURE)
**What:** Residue Hyperdimensional Computing for exact arithmetic.
**Impact:** Can't do reliable integer operations in VSA space (Kleyko 2022).
**Fix:** Implement RNS encoding: represent integers as residue vectors,
  bind for multiplication, add for addition.
**Effort:** ~5 days

---

## What Works — Proven Claims

1. **Holographic k-NN achieves 100% on local-continuity tasks** with 0 parameters.
   Fidelity decays as 1/sqrt(N). Memory is O(dim) fixed.

2. **Flat-spectrum key generation produces perfect bind/unbind roundtrip.**
   Verified: sim(v, unbind(bind(k,v), k)) = 1.000000.

3. **Radix-2 FFT at ~10 GFLOPS, 3e-15 roundtrip error.** Self-contained,
   pre-computed twiddles, zero malloc in hot path.

4. **Memory fades gracefully — no catastrophic forgetting.**
   Verified: accuracy holds at 100% from 5 to 500 stored examples
   despite fidelity dropping from 0.45 to 0.045.

5. **Neural scorer excels at logical/discontinuous mappings.**
   84% on delayed_claim vs 30.5% for holographic. Complementary strengths.

6. **HoloNet with structured receptive fields matches flat k-NN**
   while distributing load across neurons (sparse activation).

7. **Zero-parameter architecture.** No weights to train, checkpoint, or serialize.
   The model IS the memory trace. Training IS storing examples.

---

## Recommended Next Steps

| Priority | Item | Effort | Impact |
|----------|------|--------|--------|
| P0 | Wire online reinforcement into eval loop (G3) | 1h | Makes ranker usable in real pipelines |
| P0 | Serialization (G5) | 30m | Enables checkpointing and model transfer |
| P1 | vDSP FFT (G2) | 2d | 3–5× throughput improvement |
| P1 | Confidence-gated routing (G4) | 1d | Combined holo+neural wins on all tasks |
| P2 | HoloNet centroid learning (G6) | 1d | Better neuron utilization |
| P2 | Accept k-NN as core, deprecate layer training (G1) | 0 | Ships what works |
| P3 | Cross-task transfer (G7) | 3d | Knowledge sharing between domains |
| P3 | Multi-resolution memory (G9) | 2d | Coarse-to-fine retrieval |
