# ARC JSON Vision Research Note

Date: 2026-05-17

## Thesis

The current ARC failure should be treated as a perceptual representation failure,
not as evidence that the neural substrate has hit its reasoning limit.

ARC-AGI is rendered as a 2D colored-grid task, but humans do not solve it as a
flat table of independent integer cells. Humans bring object files, figure-ground
segmentation, continuity, occlusion, contact, containment, and elementary
geometry/topology priors learned in a 3D physical world. The ARC JSON format
removes the developmental path that produced those priors, then silently expects
the solver to already possess them.

For our current scorer, the result is predictable: it learns that colors change,
but it does not know that the changing cells are object states under a geometric
relation.

## What ARC Assumes

The official ARC-AGI repository defines each task as JSON demonstration pairs and
test pairs. A grid is just a rectangular list-of-lists of integers `0..9`; exact
cell match is required for correctness. The browser interface visualizes those
integers as colors, but the stored substrate is symbolic JSON, not perceptual
experience. Source: [fchollet/ARC-AGI](https://github.com/fchollet/ARC-AGI).

Chollet's ARC framing explicitly assumes Core Knowledge priors. In the 2019
paper, ARC's prior list includes objectness, elementary geometry, and topology;
the paper also says solvers with fewer priors are disadvantaged. In the paper's
ARC-specific prior section, objectness includes parsing grids into objects by
color continuity, spatial contiguity, zones, and partitions. Source:
[On the Measure of Intelligence](https://arxiv.org/abs/1911.01547).

The ARC Prize 2024 report repeats this premise: ARC tasks are meant not to
require language or specialized world knowledge, but they do assume Core
Knowledge such as objectness, topology, and elementary arithmetic. Source:
[ARC Prize 2024 Technical Report](https://arcprize.org/media/arc-prize-2024-technical-report.pdf).

That is the sleight of hand: the public data format is a matrix of symbols, but
the benchmark's intended solver class is allowed, and in practice required, to
inject a human-like visual ontology before reasoning.

## Why The Human Route Is Not Flat 2D

The important point is not that ARC needs literal stereoscopic depth. The screen
is 2D. The point is that human 2D perception is backed by object-centric priors
acquired in a 3D physical world.

Developmental-psychology work on Core Knowledge treats object continuity,
cohesion, solidity, and contact as early object principles. Spelke's object
motion work studies infants' expectations that hidden moving objects preserve
continuous paths. Source:
[Early knowledge of object motion: continuity and inertia](https://www.sciencedirect.com/science/article/pii/0010027794900132).

Object-file work gives the mechanism a useful engineering analogy. Kahneman,
Treisman, and Gibbs describe temporary episodic object representations that link
successive states of an object across appearances. Source:
[The reviewing of object files](https://www.sciencedirect.com/science/article/pii/001002859290007O).

So when a human sees an ARC panel where colored cells appear, disappear, overlap,
or move, the perceptual system does not only see color changes. It tracks
candidate objects, surfaces, masks, relative positions, and correspondences. Our
current scorer does none of that.

## What Our Diagnostics Show

The fixed multitask ARC row is task `0520fde7`, reduced to 7x3 input and 3x3
output cell-color scoring. The training examples use two 3x3 panels separated by
color `5`; the output is the panel intersection marked with color `2`.

The current scorer row sees:

- flattened input colors scaled as numeric features,
- output `x/y` and output cell index,
- candidate action color,
- one-hot output cell and action tags.

It does not receive an object/panel decomposition. In particular, it does not
directly receive "left panel cell", "right panel cell", "both occupied",
"left-only", "right-only", "row/column relation", or "separator-defined zones"
as perceptual facts.

The failure diagnostic in
`results/2026-05-17-multitask-roadmap/rmt_arc_failure_diagnostics.md` matches
that representation gap. Reproduce it with `make multitask-failure-diagnostics`;
override `MULTITASK_FAILURE_DIAGNOSTIC_*` variables for a shorter local probe:

- ARC is not simply starved. Uniform and round-robin both give about 48k ARC
  updates, while `arc_heavy` gives about 96k updates.
- It is not late forgetting. The same prediction family appears after one epoch
  and after 16 epochs while ARC train loss falls near zero.
- The tested learning-rate range does not fix it.
- The model discovers the target color set `{0,2}` and a background/foreground
  prior, but not the panel geometry.
- The single-task scorer fails the same way, which points away from multitask
  interference as the root cause.

This is exactly what we would expect from a learner asked to infer object
correspondence from raw color-cell JSON without object priors. It treats "cell
changes color" as the primitive event. The human-interpretable primitive is
"object/mask relation produces this output cell".

## Related ARC Evidence

Several current ARC papers point in the same direction.

Xu et al. find that GPT performance on ARC is strongly affected by textual grid
serialization, and that object-based representations nearly double performance
on their solved-task subset. Source:
[LLMs and ARC: Successes, Failures, and the Importance of Object-based Representations](https://openreview.net/forum?id=E8m8oySvPJ).

Li et al. report that a standard Vision Transformer fails badly on most ARC tasks
even with very large task-specific supervised data, then improves by adding
pixel-level representation, spatial tokenization, and object-based positional
encoding using automatic segmentation. Source:
[ViTARC: 2D Representation, Positions, and Objects](https://arxiv.org/abs/2410.06405).

Ferre's ARC work attacks the benchmark with object-centric models and MDL,
arguing that these models align with human natural programs. Source:
[Object-centric ARC with the MDL Principle](https://arxiv.org/abs/2311.00545).

GPAR casts ARC as generalized planning with PDDL and external functions for
object-centric abstractions, and reports gains on object-centric ARC tasks.
Source:
[Generalized Planning for the Abstraction and Reasoning Corpus](https://arxiv.org/abs/2401.07426).

The general object-centric learning literature says the same thing abstractly:
Slot Attention was motivated by the need for object-centric representations that
support abstract reasoning and generalization to new compositions. Source:
[Object-Centric Learning with Slot Attention](https://papers.nips.cc/paper/2020/hash/8511df98c02ab60aea1b2356c013bc0f-Abstract.html).

## Engineering Implication

The next fix should not be another capacity or optimizer patch. The failure is
in the ARC perceptual interface.

A principled fix should make the prior explicit and auditable:

1. Add a separate ARC object/perception envelope, rather than pretending the flat
   96-feature row is a fair visual substrate.
2. Encode separator-defined zones, panel-local coordinates, connected components,
   masks, bounding boxes, and cell correspondence features.
3. Report this as a new scorer condition, for example `arc_object_envelope_v1`,
   not as a silent improvement to the existing fixed-envelope baseline.
4. Keep the exact symbolic gate separate. The object envelope tests whether the
   learned scorer can use declared visual priors; the symbolic gate tests whether
   verifier-bounded search can certify transformations.
5. Evaluate with foreground recall, background precision, exact grid, NLL, ECE,
   and prediction hashes. Cell accuracy alone is misleading because background
   cells dominate.

For `0520fde7`, we should not hard-code "intersection" as the answer. The clean
move is to expose the perceptual facts from which intersection can be learned:
left-panel occupancy, right-panel occupancy, their conjunction/disjunction, and
relative cell coordinates. Then test not only intersection, but union, xor,
left-only, right-only, copy-left, copy-right, neither, implication, and
row-level variants in the same object envelope.

## Symbolic Layer/Depth Prior

The missing perceptual primitive should live first in the symbolic verification
layer. For ARC grids, this is not literal camera depth. It is a compact
object-file or layer representation:

- background versus foreground,
- separator-defined zones,
- panel-local coordinates,
- object/component id,
- layer identity for left-panel mask, right-panel mask, and overlap,
- optional occlusion or z-order when a task actually contains covering.

The important correction is that these are priors in the verifier grammar, not
ordinary scorer inputs. If the neural scorer receives layer/depth as a feature,
we have only moved the missing human preprocessing into a hidden input channel.
The cleaner protocol is:

1. Declare symbolic perceptual operators, such as `layer_overlap`,
   `layer_union`, `layer_xor`, `layer_left_only`, and `layer_right_only`, that
   parse a grid into aligned layer masks and render candidate outputs.
2. Let the certificate checker decide whether that operator matches all train
   pairs and whether train-consistent operators agree on the test grid.
3. Emit the operator as an accepted rule/program row with provenance under
   `priors_allowed`, not as a flat cell-color observation.
4. Let learned components rank which symbolic operators to try under a matched
   verifier budget.
5. Keep the flat JSON-cell scorer as the negative-control baseline.

For the fixed `0520fde7` panel-intersection task, the first depth/layer target is
simple: convert the 7x3 input into two aligned 3x3 occupancy planes, one for the
left panel and one for the right panel, then render candidate layer relations.
This is now an explicit symbolic rule family (`layer_overlap`, `layer_union`,
`layer_xor`, `layer_left_only`, `layer_right_only`) rather than a hidden neural
feature.

The next promoted prior is deliberately smaller: `compress_color_*` removes
uniform separator/background rows and columns before verification. It came from
the public-eval miss taxonomy rather than from the fixed task, and adds one
evaluation certificate (`68b67ca3`) without changing the flat scorer.

The follow-on object-motion/copy-extend promotion is
`tile_motif_parity_repeat`: infer a regular separator lattice, find the single
motif tile, and copy it to same-parity panels, clipping at ragged edges. This
certifies public evaluation task `92e50de0`, moving evaluation coverage to
7/400 tasks and 1057/98515 cells while leaving the training exact count at
24/400. Grammar v15 keeps that static certificate but also adds a
`tile_motif_repeat` `NNHypothesis` template with two learned slots
(`lattice.background`, `repeat.period`), so `--induction` can synthesize and
rank the same verifier candidate from train-pair lattice evidence instead of
only inheriting it from the static enumerator.

## Decision

Before proceeding with R-MT.8 or another sweep, build one explicit object-centric
ARC scorer condition and compare it against the current flat JSON-cell scorer.

The falsifiable claim is:

> If the ARC blocker is perceptual segmentation/object correspondence, then an
> object-envelope scorer should recover fixed `0520fde7` foreground cells and
> improve same-envelope OOD families without requiring a larger trunk or more
> ARC updates.

If that does not happen, then the problem is deeper than the visual interface and
we should revisit optimizer dynamics or the scorer objective. But the current
evidence says the first missing piece is object perception.
