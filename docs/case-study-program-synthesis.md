# Case Study — Program-Synthesis Library Discovery

Status: implemented as `examples/program_synthesis.c`. The binary builds a
list-routine DSL interpreter, enumerates a 32-abstraction library, runs the
matched-budget proposer grid over `random / grammar / structural /
validation-oracle / neural / closed-loop` at budgets `{1, 4, 16, 64}`, and
emits `crlp.rule_search.summary.v1` rows. The Makefile targets are
`make program-synthesis` and `make program-synthesis-baselines`; the schema
is reported through `make experiment-schemas`. The engine in
`include/nn_symbolic.h` and the matched-budget discipline from
`docs/experiment-schema.md` host the task; the implementation adds a small
DSL, an exact program verifier (running the program on input/output
examples), and a rule grammar over library abstractions.

The current run uses 12 training, 8 validation, and 8 test tasks. At budget
16, the validation oracle reaches `100%` validation compression and `12.5%`
test compression; grammar order reaches `75%` validation and `75%` test;
the closed-loop residual proposer reaches `50%` validation and `12.5%`
test, with `100%` training compression. The 12-task training set is below
the regime where greedy-on-training generalises, so the closed-loop result
is overfit-limited rather than capacity-limited (a `16x8 → 256x128` capacity
sweep at fixed budget=16 produces identical `train=100%, val=50%`
compression). The cross-role-transfer path (joint multitask training of
action and rule rows in a shared scorer) is the open lever.

## Why this domain

Program-synthesis rule discovery — discovering reusable library functions
("abstractions") that compress a corpus of training programs — is a domain
that maps unusually cleanly onto the engine's discovery-mode abstractions:

- **Exact verifier.** A library function is "correct" iff it produces the
  expected output on every input in its test set. The verifier is the
  interpreter, external to the neural model, sound by construction.
- **Non-enumerable rule space.** Library candidates are program fragments;
  the space of fragments at depth `d` over a DSL with `n` primitives is
  `Ω(n^d)`. Enumeration is infeasible for non-trivial depth. The proposer's
  job is to allocate verifier-call budget. This is the discovery-mode regime.
- **Field interest.** DreamCoder (Ellis et al., 2021), Stitch (Bowers et al.,
  2023), and the broader SyGuS literature take library discovery seriously.
  A matched-budget honest evaluation of neural library proposers would be
  cited by people who currently rely on uncontrolled comparisons.
- **No new external dependencies needed.** Unlike the theorem-prover case,
  the verifier here is an interpreter we can write in C in a day. The full
  stack ships in this repo.

## Formal contract under `NNSymbolicDomain`

| Concept | Domain instantiation |
| --- | --- |
| state | a programming-task spec (set of input/output examples) |
| action space | candidate programs in a small DSL |
| candidate enumerator | bottom-up enumeration of DSL terms up to depth bound |
| exact verifier | run the candidate program on the spec's examples, check outputs |
| rule grammar | library abstractions: a parametrised DSL term that names a reusable subprogram |
| certificate checker | proof that the abstraction's body is well-typed under the DSL grammar (purely syntactic) |
| utility | number of training-set programs that compress under the abstraction (use the abstraction in their minimum-description-length solution) |

The certificate is purely syntactic: the abstraction's body must parse and
type-check against the DSL grammar. Whether it is *useful* is the verifier's
job (and the proposer's prediction). Soundness here means "the abstraction is
a well-formed function," not "the abstraction is correct" — correctness of
the synthesised programs that *use* the abstraction is the per-state
verifier's job, not the abstraction's certificate.

## Rule grammar

Minimum viable DSL (Phase 1) — list operations, the standard SyGuS benchmark
family:

```text
expr := nil
      | cons(expr, expr)
      | head(expr)
      | tail(expr)
      | map(lambda x. expr, expr)
      | filter(lambda x. expr, expr)
      | fold(lambda acc x. expr, expr, expr)
      | +(expr, expr) | *(expr, expr) | =(expr, expr) | <(expr, expr)
      | x | y | z | 0 | 1 | -1
```

A "library abstraction" is a named DSL term with free variables. For example:

```text
abstraction sum_list(xs) := fold(lambda acc x. +(acc, x), 0, xs)
```

The grammar over abstractions is:

```text
rule := abstraction(name, lambda x_1 ... x_k. body)
```

Where `body` is a closed DSL term over `x_1 ... x_k` and `name` is a fresh
identifier. The certificate verifier checks that `body` parses against the
DSL grammar with `x_i` in scope.

## Why this fits the engine

Mapping into `NNSymbolicDomain` callbacks:

- `enumerate_rules`: bottom-up DSL enumeration up to a depth bound (default
  ≤ 6). Phase 1 caps the candidate set at ≈ 50k abstractions, comparable in
  scale to the prime-gap unconditional grammar.
- `encode_rule`: structural features of the abstraction — term depth, node
  count, primitive-symbol multiset, arity, presence of recursion via `fold`,
  parameter count. ≈ 16 features. Independent of the training task set.
- `encode_residual_features` (closed-loop / unified): for each candidate,
  similarity to already-accepted abstractions in the current library
  (matching subterm count, shared primitives), and residual coverage over
  training tasks the abstraction does not yet help compress.
- `check_certificate`: parser + type-checker on the abstraction body.
- `verify_action`: run the candidate program (which may use accepted
  abstractions) on the task's input/output examples; return the per-example
  pass/fail counts.

`NNRuleCoverage` is a bitset over (training-task × abstraction) pairs: bit
set if the task's known correct solution becomes shorter when expressed using
the abstraction. Greedy set-cover on this matrix is the standard "MDL
library compression" heuristic that DreamCoder and Stitch use. It is *also*
the natural marginal-utility target.

## Dataset

Two viable starting datasets, in increasing realism:

1. **SyGuS-Comp list/integer benchmarks.** Published benchmark suite with
   tens of tasks per category. Small enough to use entire categories as
   training/test splits. Trivial to load from JSON.
2. **DreamCoder list-routines benchmark.** ≈ 100 small list-manipulation
   tasks released with the DreamCoder code. Each task has input/output
   examples; each has a known minimum-length solution. Loader is a small JSON
   parser.

For Phase 1, the DreamCoder list-routines benchmark is the right scale: small
enough that bottom-up enumeration produces ground-truth oracle library sets
for validation. Phase 2 can scale to multi-domain DreamCoder tasks
(list + arithmetic + simple string manipulation).

## Matched-budget baselines

Adapting the discipline from `docs/experiment-schema.md`:

| Baseline | Mechanism |
| --- | --- |
| random | sample abstractions uniformly from the admissible set |
| grammar order | enumerate abstractions in DSL-depth, lex-min order; take first K |
| structural heuristic | rank by hand-coded features: shorter body first, higher primitive variety, lower recursion depth |
| validation oracle | rank by held-out (validation-task-set) compression gain |
| neural (static) | scorer over abstraction structural features, no library/task conditioning |
| closed-loop | proposer scores conditional on already-accepted abstractions and residual training-task coverage |
| unified scorer | shared neural scorer trained alongside the existing four action-scoring tasks |

The metric is **training-program compression at budget K**: with a library of
K accepted abstractions, what fraction of training-program solutions become
shorter (measured in DSL nodes) versus the baseline DSL-only solution. K ∈
{1, 4, 16, 64}. Larger K than prime-gap because library discovery is a
larger-grammar regime.

A defensible positive result is: **the unified scorer at K=16 achieves
training-program compression within 5% of the validation oracle, while
beating grammar order and structural heuristic by ≥ 5pp.**

The oracle ceiling here is **not** greedy set-cover on the same data the
network sees — it is greedy set-cover on a held-out validation task set.
That's the key disanalogy with prime-gap and the reason this domain is more
promising for a positive headline: the structural-prior story has somewhere
to land because validation generalisation, not training fit, is the metric.

## Expected leakage hazards

Three new ones to watch:

1. **Test-program features in `encode_rule`.** If the abstraction features
   include any function of the training-task solutions (e.g., "how many
   training programs textually contain this subterm?"), the encoder is
   doing the verifier's job at feature time. That number is the target,
   not a feature. Same shape as the prime-gap `train_rejects` leak. The
   audit must catch it.
2. **DSL declaration order.** SyGuS benchmarks have a conventional primitive
   order (cons, head, tail, ...). A feature "primitive index" implicitly
   encodes "more basic primitives come first," which correlates with
   abstraction usefulness. Same shape as the theorem-prover library-order
   leak. Either exclude or tag `priors_allowed`.
3. **Train/test task overlap.** DreamCoder benchmarks include task variants
   (sum-list, sum-positive-list, sum-of-squares); a split that puts a task
   variant in test while keeping the base in train leaks structure. Use
   the task-family split, not a per-task split, for the headline.

The first one is the load-bearing hazard. The agent that implements this
should write `encode_rule` so it physically cannot read solution-program
contents. Pass only the abstraction body and the DSL grammar; do not pass
the training set.

## Engine fit

The engine in `include/nn_symbolic.h` covers this domain except:

- `NN_RULE_MAX_NAME = 64`. Abstraction names ("sum_list", "max_of_list_pair")
  fit comfortably.
- `NN_RULE_MAX_PARAMS = 12`. Abstraction parameters are ≤ 4 in practice
  (most useful library functions take 1–2 arguments).
- `NN_RULE_MAX_FEATURES = 64`. Structural features of an abstraction
  (≈ 16) plus residual features (≈ 10) fit.
- `NNCertificate`. The certificate text records "well-typed at depth `d`
  under DSL `G`." Cheap to verify, sound by construction.
- `NNRuleCoverage`. Events are (training-task, abstraction) pairs. With ≈
  100 training tasks and ≈ 50k abstractions, the bitset is ≈ 100 × 50k / 8
  ≈ 625 KB — small enough for the dense representation already in the
  engine.

No new engine work is required for Phase 1. This is the more accessible of
the two case studies precisely because the verifier is an interpreter, not
an external dependency.

## Phased plan

| Phase | Deliverable | Effort |
| --- | --- | --- |
| 0 | DSL interpreter, DreamCoder list-routines loader, abstraction enumerator | 3 days |
| 1 | Matched-budget grid (random, grammar order, structural, oracle, static neural, closed-loop) on list-routines compression at K ∈ {1, 4, 16, 64}. | 2 days |
| 2 | Unified scorer evaluation (per `docs/unified-scorer-proposal.md`). Train shared scorer across prime-gap, grid-path, delayed-claim, ARC, and program-synthesis rule rows. | 1 week |
| 3 | Multi-domain DreamCoder upgrade (list + arithmetic + strings), report compression per domain plus cross-domain transfer. | 1 week |
| 4 | Comparison against DreamCoder / Stitch published numbers on the same benchmarks under the matched-budget discipline. | 1 week |

Phase 0–2 ships as a workshop-paper-ready result. Phase 4 is the strongest
external-comparability claim; it requires care because published library-
discovery numbers are not reported under matched-budget discipline and the
comparison is necessarily one-sided (we report the discipline; they did not).

## Honest scope

This case study is implementable entirely within this repo, no external
runtime dependencies. The DSL interpreter is ≈ 500 lines of C. The DreamCoder
list-routines dataset is a small JSON file in their public release. Phase 4
("compare to DreamCoder / Stitch") requires running their Python pipelines
to obtain published-protocol numbers, but that comparison is in a separate
evaluation environment; this repo only needs the C-side results.

The DSL is deliberately small. A larger DSL (lambda calculus with let-
polymorphism, recursion combinators, recursive types) would make the case
study more realistic but expand the engineering surface by an order of
magnitude. Phase 1's list-routines DSL is the right shakedown.

## Expected ceiling and where the unified scorer earns its keep

The structural prediction:

- The **closed-loop dedicated proposer** will tie greedy-on-training set-
  cover on training compression, the way it ties grammar order on prime-gap.
- The **validation oracle** (greedy on held-out task set) will outperform
  greedy-on-train by 3–10pp depending on split, because library discovery
  generalises poorly from small training sets.
- The **unified scorer** has a real chance to beat closed-loop on
  *validation* compression because the cross-role transfer should produce a
  structural prior that doesn't overfit the training-event set as eagerly.
  This is the positive headline the project is looking for.

If the unified scorer beats closed-loop on held-out compression while losing
on training compression — that's the cleanest demonstration of the
neuro-symbolic substrate's value. It would be impossible to produce on the
prime-gap domain because there is no analogous generalisation gap there.

## Out of scope for this case study

- Inductive-program-synthesis algorithms beyond bottom-up enumeration
  (e.g., PBE search, SMT solver back-ends). The engine doesn't need them.
- Probabilistic DSLs with continuous parameters. Stay discrete.
- Recursive type inference. The DSL is monomorphic (lists of ints).
- Comparison against neural-only program synthesis baselines (Codex, etc.).
  The library-discovery problem is the contribution surface, not end-to-end
  programming.
