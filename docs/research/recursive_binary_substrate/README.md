# Recursive Binary Substrate Research Sprint

Date: 2026-05-22
Status: pre-scale architecture review

## Decision Summary

Do not scale the recursive binary net as a raw "bigger 3x3 brain" yet.

The literature supports the core direction if we keep the neural layer as a
proposer and promote only verified tools. It does not support expecting local
binary neural maps to discover object layers, arithmetic, ARC programs, and
language coherence from raw grids alone.

The next scale-up should be:

```text
raw grid streams
-> learned layer/object/motion/residue proposals
-> certified local tools
-> neural-guided program search
-> exact reconstruction or task verifier
-> promoted macro library
```

The architecture is worth continuing, but only with pre-scale gates:

1. **Layer induction gate:** learn masks/layers from frame pairs by verified
   reconstruction and compression, not by supervised labels.
2. **Tool alphabet gate:** implement translation, crop, attend, copy, mask,
   compose, symmetry, object/layer overlap, residue/wheel, and arithmetic
   tools as certifiable operators.
3. **Verifier boundary gate:** every promoted abstraction must have an exact
   verifier or be marked diagnostic-only.
4. **Compression gate:** compare raw 3x3 learning against compact features
   such as popcount, D4 orbit, connected components, motion vectors, and
   modular residues.
5. **Out-of-distribution gate:** require held-out size/position/rule-family
   tests before claiming scale generalization.

## Current Repo Architecture Under Review

The current substrate has the right spine:

- `NNBitwiseGate` learns the 16 binary truth tables and promotes only after
  exhaustive truth-table replay.
- `NNBitwiseToolPolicy` searches over certified gates and operands.
- `NNBitwiseMacroLibrary` promotes verified chains and now records symbolic
  fallback use.
- `NNTileBot` trains `3x3 -> center` operators over all 512 tile states.
- `tile-feature-entropy` measures whether a compact feature exactly explains a
  rule before training.
- `life-curriculum` treats inverse dynamics as proposal plus exact forward
  verification rather than a unique inverse.

This matches the strongest pattern in the literature: learned proposal,
symbolic or executable verification, and library growth.

The weak point is also clear: we do not yet have a data-layer system that
discovers layers, slots, motion fields, residue layers, or object masks from raw
streams. Without that, a larger recursive grid net risks memorizing local
appearance rather than discovering reusable structure.

## Literature Findings

### 1. Local rules can be computationally universal, but universality is not learnability.

Cook's Rule 110 result proves that a tiny binary local rule can support
universal computation on an infinite line. This supports the idea that local
binary substrate is expressive enough in principle. It also warns us that
behavior can become undecidable and hard to control. A universal substrate does
not automatically learn useful decompositions or efficient programs.

Implication for us: keep `3x3` local tools as the substrate, but make higher
layers explicit and verifier-gated.

### 2. Neural cellular automata validate learned local dynamics, but show stability limits.

Growing Neural Cellular Automata is highly relevant because it trains local
neural update rules over grids and studies persistence/regeneration. Its model
uses local neighborhood perception, stochastic updates, and living-cell masks.
The important warning is that patterns trained for a finite rollout can become
unstable when run longer than training.

Implication for us: every recursive local operator needs long-horizon and
damage/perturbation tests before promotion into a macro layer.

### 3. Object-centric learning is the closest literature for layered data.

AIR, MONet, Slot Attention, and SAVi/SAVi++ all attack the missing layer in our
system: unsupervised or weakly supervised decomposition of images/videos into
objects, masks, slots, and tracked entities.

The useful idea is not to copy their architectures wholesale. The useful idea is
their training pressure:

```text
decompose into slots/layers -> reconstruct or predict -> reuse slots over time
```

For our engine, the analogous objective should be:

```text
propose masks/layers/transforms
-> compose them
-> exactly reconstruct the target grid
-> minimize description length of the explanation
```

Implication for us: build layer discovery as an explicit engine surface before
scaling ARC or moving-grid tasks.

### 4. Program synthesis literature supports library growth plus neural search.

DreamCoder is the strongest analogue to our intended recursion. It alternates
program search, abstraction/library learning, and neural guidance over the
expanded library. This is very close to our certificate-gate idea:

```text
solve tasks -> compress solutions into abstractions -> train search policy
```

Implication for us: the "recursive" part should mostly live in library growth
and tool-chain compression, not in a monolithic hidden state.

### 5. ARC literature supports priors and verifier-gated task construction.

Chollet's ARC framing emphasizes skill-acquisition efficiency, priors,
abstraction, and generalization difficulty. This aligns with our repo's
existing separation between fixed ARC scorers and verifier-gated ARC loaders.

Implication for us: ARC scale-up should not be a flat cell-color classifier.
It should extend the symbolic/verifier grammar with layer/depth/object priors
and use neural rankers only to choose candidate programs.

### 6. Neural algorithmic reasoning supports algorithm-shaped inductive bias.

Neural Algorithmic Reasoning, Neural GPU, Neural Execution Engines, and NALU all
argue from different directions that algorithmic extrapolation needs the right
computational structure. NALU is especially relevant to arithmetic: generic
MLPs do not reliably extrapolate numeric operations without arithmetic bias.

Implication for us: primes and sieves should use certified arithmetic and
modular tools. A raw grid bitmap of primes is not enough.

### 7. Binary neural networks are useful for implementation efficiency, not the core idea.

Binarized Neural Networks are relevant to runtime efficiency and hardware, but
they are not the same claim as our binary grid substrate. Our main question is
not whether weights can be 1-bit; it is whether binary-state tools can be
learned, certified, composed, and promoted.

Implication for us: defer BNN-style weight binarization until after the
operator/library architecture is stable.

## Domain Assessment

| Domain | Fit | Required additions before scale |
| --- | --- | --- |
| Bitwise gates | Strong | Already works; keep fallback auditing strict. |
| 3x3 local tiles | Strong for finite local rules | Add more features, long-horizon stack tests, and macro compression. |
| Moving grids | Promising | Need layer/mask/motion induction and occlusion handling. |
| ARC | Promising but not raw | Need object/layer/depth priors and verifier-gated program search. |
| Primes/sieves | Promising with arithmetic bias | Need residue/wheel/sieve interval layers and primality certificates. |
| Program construction | Strong | Use DreamCoder-style library growth and executable tests. |
| Haiku | Partial | Grammar/adjacency can certify; semantic coherence remains diagnostic unless a stronger verifier is built. |

## Recommended Architecture Change Before Scaling

Add a **Layer Discovery API** between raw grids and tool policy:

```c
NNLayerProposal {
    mask_grid;
    value_grid;
    transform_hint;
    support_bbox;
    layer_kind;        // object, background, residue, motion, text token, etc.
    reconstruction_cost;
    certificate;
}
```

Training objective:

```text
minimize:
  reconstruction_error(input -> target via proposed layers/tools)
  + description_length(layers)
  + description_length(tool chain)
  + uncertainty_penalty

subject to:
  exact verifier passes before promotion
```

This gives the "layered data" side of the brain. Without it, scaling the
recursive net mostly scales the burden on raw local pixels.

Initial implementation now exists as `include/nn_layer.h` and `src/nn_layer.c`.
It exposes a deterministic motion-layer discovery surface for binary frame
pairs, produces per-layer certificates, recomposes the proposed layers, and
ships a `make layer-discovery` synthetic benchmark. This is intentionally the
smallest certified wedge: translations over separated binary objects, no
occlusion or residual fallback in the default benchmark.

The same module now includes `NNLayerRanker`, a neural proposal ranker trained
on synthetic motion scenes. The network ranks candidate translations for each
connected component, but promotion still depends on exact verifier checks and
full-grid recomposition. This keeps learning pressure separate from
certification: the ranker can reduce proposal search cost, but cannot certify a
wrong transform.

The next layer is also wired: `NNLayerTaskPolicy` learns from a demonstration
pair `(demo_source, demo_target)` and executes the inferred global translation
on a separate `query_source` without receiving the query target. Evaluation only
uses `query_target` after execution as a verifier. This turns the first motion
slice from target-conditioned lookback into a small demo-conditioned executor.
The certified scope is deliberately narrow: global translations over separated
binary objects, not occlusion, per-object motion, arbitrary chains, or ARC-scale
program construction.

## Pre-Scale Experiment Plan

### E1: Synthetic motion layer induction

Generate two-frame binary scenes with 1-4 moving objects, static background,
occlusion, wrapping, and noise. Train/rank layer proposals. Promote only if
recomposition exactly predicts frame 2.

Metrics:

- exact reconstruction rate
- number of layers used
- description length
- held-out object count and velocity generalization

### E2: ARC layer grammar replay

Take existing ARC layer/depth grammar and force the new layer API to express
the same transforms. Compare exact certification count and search budget.

Metrics:

- certified train/eval tasks
- proposal count to first verifier pass
- regressions versus current symbolic loader

### E3: Sieve residue compression

Represent integer intervals as binary grids. Compare raw prime bitmap learning
against residue/wheel layers and certified divisibility tools.

Metrics:

- candidate-prime recall
- compression ratio versus raw bitmap
- verifier cost per accepted prime
- held-out interval generalization

### E4: Life and CA long-horizon stability

Keep the current `3x3` Life rule, but evaluate stacks on longer rollouts,
perturbed states, and larger toroidal boards. Include learned inverse proposals
only as verified preimage proposals.

Metrics:

- exact rollout match over horizon
- preimage proposal verification rate
- failure modes by population count and symmetry bucket

### E5: Program-library compression

Use existing `NNProgram`/hypothesis surfaces to learn chains and compress them
into macros. Compare against grammar order and random search.

Metrics:

- training-program compression at fixed K
- validation task solve rate
- verifier pass count per promoted macro

## Sources Pulled Into Local Pack

| Topic | Local file | Source |
| --- | --- | --- |
| Neural CA | `papers/growing_neural_cellular_automata_distill.html` | https://distill.pub/2020/growing-ca/ |
| CA universality | `papers/rule110_cook_2004.pdf` | https://content.wolfram.com/uploads/sites/13/2018/02/15-1-1.pdf |
| Slot object discovery | `papers/slot_attention_2006.15055.pdf` | https://arxiv.org/abs/2006.15055 |
| Scene decomposition | `papers/monet_1901.11390.pdf` | https://arxiv.org/abs/1901.11390 |
| Sequential object inference | `papers/air_1603.08575.pdf` | https://arxiv.org/abs/1603.08575 |
| Video object slots | `papers/savi_plus_plus_2206.07764.pdf` | https://arxiv.org/abs/2206.07764 |
| Program synthesis library growth | `papers/dreamcoder_2006.08381.pdf` | https://arxiv.org/abs/2006.08381 |
| ARC benchmark/priors | `papers/arc_measure_1911.01547.pdf` | https://arxiv.org/abs/1911.01547 |
| Neural algorithmic reasoning | `papers/neural_algorithmic_reasoning_2105.02761.pdf` | https://arxiv.org/abs/2105.02761 |
| Neural algorithm execution | `papers/neural_gpu_1511.08228.pdf` | https://arxiv.org/abs/1511.08228 |
| Learned subroutine execution | `papers/neural_execution_engines_2006.08084.pdf` | https://arxiv.org/abs/2006.08084 |
| Arithmetic extrapolation | `papers/nalu_1808.00508.pdf` | https://arxiv.org/abs/1808.00508 |
| Binary neural nets | `papers/binarized_neural_networks_1602.02830.pdf` | https://arxiv.org/abs/1602.02830 |

## Commit Recommendation

Commit to the architecture only in this narrower form:

```text
recursive binary substrate
+ certified tool promotion
+ layer/object/residue discovery
+ neural-guided program search
+ exact verifier boundary
```

Do not commit to:

```text
raw binary grids + larger recursive net = general intelligence
```

The next implementation milestone should be the Layer Discovery API plus the
synthetic motion reconstruction benchmark. That benchmark will tell us whether
the brain can learn the missing layered data representation before we scale
ARC, primes, or haiku around it.
