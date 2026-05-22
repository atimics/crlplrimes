# Bitwise Recursive Substrate

This layer makes the first rewrite step intentionally finite.

For two local input bits `(a, b)`, the complete binary-operation domain is the
set of 16 truth-table masks over rows `00, 01, 10, 11`. The neural job is not to
guess a named gate such as XOR. It is to learn a mask from this closed domain,
then promote the learned model only if exhaustive truth-table replay proves all
four rows match.

After promotion, execution uses the certified gate path rather than neural
inference. A grid shape is then a recursive composition surface: each cell uses
the same promoted local operator, so widths like `0`, `01`/`10`, `010`/`101`,
and larger bit grids inherit the same certificate.

The next layer separates the easy and hard parts:

- Verification is exact: execute the chain and check whether the attended output
  grid matches the target grid.
- Inference is search: infer which truth-table constraints are implied by
  `X,Y -> target`, rank candidate tool applications, execute the selected chain,
  and let the verifier accept or reject the result.
- Auditing distinguishes policy search from rescue search: `NNBitwiseChain`
  now records whether a symbolic shortest-chain fallback was used, and the XOR
  transform regression requires the learned policy path.

For a constraint such as `X + Y + Z = 0` over GF(2), the target construction is
`Z = X xor Y`. The tool policy does not need XOR to be hard-coded as the answer;
it learns to choose certified tools and operands from the current workspace,
then the verifier checks that the resulting `Z` matches the attended target.

The implemented substrate is:

- `NNBitGrid`: gridlike binary cells with alternating-pattern helpers.
- `NNBitwiseGate`: a neural learner plus a `NNCertificate`.
- `nn_bitwise_gate_train_truth_table`: learns any of the 16 truth masks.
- `nn_bitwise_gate_apply`: applies a promoted gate over matching grid shapes.
- `NNBitwiseToolPolicy`: neural policy over certified gates and workspace
  operands.
- `nn_bitwise_chain_solve_grid_transform`: infers and verifies a chain that
  transforms attended input grids into a target grid.
- `NNBitwiseMacroLibrary`: promotes verified chains into reusable certified
  macro-tools, while reporting fallback-promoted tools separately.
- `make bitwise-recursive`: learns and certifies the complete binary domain,
  runs XOR, AND, OR, and NOT-A over recursive alternating bit grids, promotes
  learned macro-tools, and solves a verified `X + Y + Z = 0` transform.

The promotion rule is strict:

```text
neural learner -> exhaustive replay over 4 binary rows -> certificate gate
certified gates + attended grids -> neural chain policy -> exact match verifier
verified chains -> exhaustive composition replay -> promoted macro-tool
```

This keeps the neural layer as proposer/search machinery and keeps the promoted
operator as deterministic symbolic substrate.

## Recursive 3x3 Tile Layer

The next recursive layer treats a `3x3` neighborhood as a tile symbol. A binary
`3x3` tile has `2^9 = 512` states, so local tile operators can still be trained
from synthetic exhaustive rows and certified by replaying the complete tile
table.

The first tiled experiment is `make tile-recursive`. It trains three stacked
`3x3 -> center` neural operators over an `18x18` binary grid. Three such layers
give a `7x7` effective receptive field. Training uses the complete 512-row tile
table, reweighted by Hamming-weight bucket so rare singleton and near-full tiles
are not washed out by the common middle buckets.

The reusable implementation is `NNTileBot` in `include/nn_tilebot.h`. A tile bot
has a local rule, a feature projection, a neural learner, and an exhaustive
`NNCertificate`. `nn_tilebot_train` trains and promotes a local operator only
after all 512 tile rows replay exactly. `nn_tilebot_eval_stack` compares a stack
of learned tile bots against the exact symbolic stack on sampled grids.
`NN_TILE_FEATURE_POPCOUNT` and
`NN_TILE_FEATURE_LIFE_CENTER_NEIGHBOR_COUNT` are the first certified pruning
features; raw `3x3` bits remain available as a diagnostic feature.

The successful stacks are deliberately simple but diagnostic:

- `dilate_x3`: three certified dilation layers.
- `majority_x3`: three certified majority layers.
- `dilate_majority_erode`: a mixed stack of three different certified tile
  operators.

Each learned layer must match all 512 tile rows before the stack result is
treated as certified. The grid-level verifier then compares the three-layer
learned render against the exact symbolic render.

This is the intended synthetic-data path for the recursive substrate:

```text
bits -> certified cell gates -> certified 3x3 tile bots
     -> grids of tile bots -> larger synthetic tasks -> promoted macro-tools
```

## Special-Feature Entropy

`make tile-feature-entropy` measures which hand-built tile features can prune
the 512-row `3x3` space before training a bot. The useful score is not only
information gain:

```text
IG(feature; target) = H(target) - H(target | feature)
efficiency = IG(feature; target) / H(feature)
```

`H(target | feature) = 0` means the feature buckets are sufficient to decide the
target exactly. The efficiency term then prefers compact features over raw
memorization. For example, `raw_mask` is often perfectly informative but uses
512 buckets, while compact `d4_orbit` uses 102 symmetry buckets for
symmetry-invariant targets. `popcount` uses only 10 buckets, so it is the right
pruning feature for count-based tile bots such as dilation, majority, erosion,
and parity.

The same experiment trains a tiny `popcount` feature bot: a `10 -> 1` network
with 11 parameters over the 10 Hamming-weight buckets. It certifies all 512 tile
rows for `dilate_any`, `majority5`, `erode_all`, and `parity9`. This is the
bridge for hard cases: parity should be routed through certified special
features or lower-level XOR macros, not forced through a smooth raw-pixel MLP.
The Life local rule is handled the same way with an `18 -> 1` center-neighbor
feature bot.

## Game Of Life Curriculum

`make life-curriculum` uses Conway-style cellular dynamics as a reversible-search
curriculum. The forward local rule is a `3x3 -> center` operator, so it fits the
same tile-bot substrate. The compact sufficient feature is
`center + neighbor_count`, giving 18 buckets. The trained feature bot is a
`18 -> 1` network with 19 parameters and certifies all 512 local tile rows.

The inverse task is intentionally harder. A Life step is many-to-one, so
“predict the prior state” should be treated as preimage proposal plus exact
verification, not as a unique inverse. On finite toroidal boards the experiment
enumerates the complete forward map:

- `3x3`: 512 states, 128 reachable next states, 384 Garden-of-Eden states,
  maximum 302 preimages for one next state.
- `4x4`: 65,536 states, 17,879 reachable next states, 47,657 Garden-of-Eden
  states, maximum 9,628 preimages for one next state.

The first inverse probe trains a `3x3` canonical-preimage predictor and verifies
its outputs by applying Life forward. It reaches `128/128` verified preimages
for the reachable `3x3` targets. This gives a natural curriculum:

```text
learn certified forward local dynamics
-> enumerate small-board preimage ambiguity
-> train canonical or low-energy preimage proposals
-> verify by forward rollout
-> scale board size and add search when direct inverse prediction becomes ambiguous
```

`4x4` is intentionally a later regime. It has `65,536` tile symbols and
`4,294,967,296` rows for one binary tile-operation table, so exhaustive
certification stops being the cheap inner loop. The practical route is to scale
the population of certified `3x3` bots first, then use those bots as the
operator alphabet for higher layers.

The intended infinite-plane generalization is sparse: a state is finite attended
support plus an implicit background. Translation, cropping, and spatial
attention should become additional certified tools in the same chain policy.
The verifier still only checks the declared attended support unless a domain
adapter supplies a stronger invariant.
