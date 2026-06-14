# Engineering Improvement Report and Philosophical Analysis

Date: 2026-06-07
Reviewer: OpenRouter coding agent, local repository inspection
Snapshot: `00a64dd` with an uncommitted working tree
Repository: `crlplrimes`

## Scope and method

This report reviews the repository as an engineering system and as a research
artifact. I inspected the top-level documentation, build system, CI workflows,
core C headers and sources, representative large experiment binaries, generated
paper-support artifacts, previous engineering review material, and current test
surface. I also ran local verification commands where they were cheap and safe.

Commands run locally:

- `make test`: passed.
- `make -j4 all`: passed on this machine, with warnings, because this checkout
  has Apple Accelerate and a sibling `../signal` tree available.
- `make source-policy-check`: failed with a shell syntax error.
- Repository inventory and line-count scans using `find`, `wc -l`, `git status`,
  and targeted `sed`/`nl` reads.

Important caveat: the working tree was dirty before this report was written.
Current uncommitted changes include `Makefile`, `examples/arc_agi.c`,
`examples/substrate_experiments.c`, generated paper files, latest trace output,
and new untracked files such as `examples/hologram_merge.c`,
`examples/vanity_hologram.c`, and `include/base58.h`. This report should be read as an analysis of the
current working tree, not a clean release tag.

## Executive summary

`crlplrimes` is not just a neural-network toy library. It is a compact C
research operating system for a specific epistemic discipline:

> The model proposes. The declared grounding operator decides within its
> authority. The trace remembers.

That invariant is the repository's most valuable idea. The best parts of the
codebase make this idea concrete through task schemas, feature-provenance
audits, symbolic rule certificates, replayable JSONL traces, paper-claim gates,
and matched-baseline result capture. The strongest promoted evidence remains the
certified modular gate-search line around `prime_gap`, plus the `grid_path` and
`delayed_claim` supporting domains. The repository also contains many ambitious
sandboxes: ARC, Signal control, haiku, program synthesis, holographic memory,
EML activations, Go engines, and paper-agent machinery.

The main engineering risk is that the codebase has outgrown its control
surfaces. The repo is now a dense mixture of production evidence contracts,
experimental playgrounds, generated paper governance, external simulator
integration, one-off scripts, and new speculative systems. This creates several
urgent issues:

- The default build is not portable despite Ubuntu CI declarations: `src/nn.c`
  directly includes Apple Accelerate and `Makefile` links `-framework
  Accelerate`; `make all` also depends on a sibling `../signal` checkout.
- `make source-policy-check` is syntactically broken and, if repaired naively,
  conflicts with tracked Python files already present in the repo.
- The `Makefile` has become a 1900+ line orchestration monolith with stale,
  retired, duplicate, and partially disconnected targets.
- Several core files are too large to review confidently: `examples/arc_agi.c`
  is about 26K lines, `src/nn_grammar_discovery.c` about 19K lines,
  `src/nn.c` about 5K lines, and `tests/test_nn.c` about 4.6K lines.
- Generated evidence and paper-claim artifacts are strong in concept but risk
  becoming ritualized if hard-coded rows, stale generated files, and underpowered
  claims are not aggressively separated from live measurements.

The recommended next phase is not to add another experiment. It is to make the
repo's epistemic boundary mechanically boring: portable build, fixed CI gates,
smaller modules, explicit experiment tiers, and artifact freshness checks that
make stale or local-only evidence impossible to confuse with canonical support.

## System map

### Core layers

The codebase has four main layers:

1. Core neural runtime:
   - `include/nn_core.h`
   - `src/nn.c`
   - training, ranking, softmax scorer, checkpoints, trace records, numeric
     update certificates.

2. Experiment and protocol contracts:
   - `include/nn_experiment.h`
   - `include/nn_protocol.h`
   - `include/nn_symbolic.h`
   - task schemas, leakage policies, grounding authorities, candidate rows,
     verifier outcomes, rule traces, library promotion records.

3. Domain implementations and sandboxes:
   - promoted-ish: `src/prime_gap/*`, `examples/grid_path.c`,
     `examples/delayed_claim.c`, Signal replay/mining.
   - sandbox: ARC, program synthesis, haiku, multitask, holographic memory,
     EML, Go, cross-domain transfer.

4. Evidence and publication machinery:
   - `results/latest/*`
   - `paper/claims.yaml`
   - `paper/generated/*`
   - `src/nn_paper.c`
   - `src/nn_grammar_discovery.c`
   - Make targets such as `results`, `paper-support`, `paper-ci`,
     `experiment-health`, and numeric-certificate gates.

### The architectural invariant

The README states a clean generalized architecture:

```text
state -> candidate enumerator -> gated candidates -> learned scorer -> grounding operator -> JSONL trace -> replay
```

The strongest implementation detail is that this is not only prose. It appears
in C contracts:

- `NNFeatureProvenance` distinguishes state observations, structural priors,
  verifier-derived features, and gate-derived features.
- `NNLeakagePolicy` separates forbidden, priors-allowed, legacy, and
  gate-allowed modes.
- `NNGroundingAuthority` separates exact checks, deterministic replay, corpus
  proxies, and human review.
- `NNCertificate`, `NNRule`, `NNVerifierOutcome`, and trace schemas encode the
  claim that correctness belongs to explicit authorities, not to neural scores.

This is the repository's defensible core. Most improvement work should protect
this boundary from accidental dilution.

## Verification status observed in this pass

### Passing locally

`make test` passed. The unit suite covers a surprisingly broad range for a C
research repo: batched forward equivalence, gradient finite differences,
softmax-scorer behavior, guarded numeric certificates, trace parsing,
experiment-schema JSON, paper claim governance, symbolic certificate contracts,
bit-grid/tilebot/layer/program/hypothesis/protocol contracts, prime-gap parser
and feature contracts, replay action IDs, and closed-loop/unified rule search.

`make -j4 all` also passed locally. That is useful, but not proof of clean-clone
portability because this machine has dependencies that CI and other developers
may not have.

### Failing locally

`make source-policy-check` fails with:

```text
/bin/sh: -c: line 1: syntax error: unexpected end of file
make: *** [source-policy-check] Error 2
```

The target is missing a closing `fi` in `Makefile`. This was later resolved by
making source policy explicit: orchestration belongs in the C runtime surface,
and tracked Python source is rejected rather than reclassified as a permanent
exception.

### Local warnings from `make -j4 all`

Representative warnings:

- `examples/cross_domain_transfer.c`: logical-op-parentheses, unused variable,
  unused parameter.
- `examples/haiku_word_lm.c`: variable set but unused.
- `examples/holo_experiments.c`: unused variable.
- vendored Signal/TweetNaCl sources: signedness warnings.

The warnings are not catastrophic, but the project should decide whether
`-Werror` applies to first-party code. Given the claim-governance posture, the
answer should eventually be yes for core and promoted domains, no or quarantined
for vendored dependencies.

## Engineering strengths

### 1. Explicit epistemic contracts

The experiment schema is excellent. It forces each task to name its state
distribution, action enumerator, feature provenance, leakage policy, grounding
operator, certificate checker, proposer inputs, baselines, trace protocol, and
claim scope. This is the difference between a demo and an auditable research
machine.

### 2. Strong separation between ranking and authority

The repo repeatedly encodes the idea that neural models rank residual choices
but do not own correctness. This gives the project a clear answer to a common
failure mode in ML systems: confidence is not truth. The verifier, simulator,
corpus proxy, or human-review gate owns the declared authority.

### 3. Replay-first evidence

Stable action IDs, ranking policies, score-margin metadata, trace determinism,
and replay contracts are exactly the right primitives for a research codebase
that evolves quickly. This makes experiments debuggable across numeric changes,
candidate ordering changes, and model checkpoint changes.

### 4. Mechanical paper governance

The paper-support compiler and claim registry are a major asset. The support
report explicitly marks claims as supported, underpowered, or missing. That is
far healthier than burying weak evidence in prose.

### 5. Good unit-test intent

Even though the test file is large, the unit test suite is not superficial. It
checks numerical behavior, serialization, JSON contracts, domain-specific prime
logic, and cross-role scorer contracts. The tests encode important research
invariants, not only implementation details.

### 6. C implementation discipline in the core

The core runtime uses explicit allocation checks in many places, stable binary
checkpoint formats, scalar-type switches, and deterministic RNG patterns. There
are rough edges, but the core is much more deliberate than the average research
C codebase.

## Major risks and improvement opportunities

### P0. Make the build portable and honest

Problem: CI says Ubuntu, while the core build assumes Apple Accelerate and a
local sibling Signal checkout.

Evidence:

- `src/nn.c` unconditionally includes `<Accelerate/Accelerate.h>` and directly
  calls `cblas_sgemv`, `cblas_dgemv`, `cblas_sgemm`, and `cblas_dgemm`.
- `Makefile` defaults `LDFLAGS` to `-framework Accelerate`.
- `.github/workflows/check.yml` runs on `ubuntu-24.04`, installs `gcc-13` and
  `make`, and then runs `make -j$(nproc) all`.
- `make all` includes `$(SIGNAL_MINING)`, whose recipe compiles many files from
  `../signal/...`.

Impact:

- A clean Ubuntu runner cannot rely on Apple frameworks.
- A clean checkout cannot rely on `../signal` existing.
- The local passing build can create false confidence about CI and contributor
  reproducibility.

Recommended fix:

- Add a BLAS portability layer:
  - `NN_USE_ACCELERATE=1` on macOS.
  - `NN_USE_CBLAS=1` for OpenBLAS/CBLAS on Linux.
  - scalar fallback loops when no BLAS is found.
- In `src/nn.c`, guard Accelerate include:

```c
#if defined(NN_USE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#elif defined(NN_USE_CBLAS)
#include <cblas.h>
#endif
```

- Hide BLAS calls behind `nn_blas_gemv` and `nn_blas_gemm` wrappers.
- Split `all` into `all-core` and `all-integrations`; keep Signal targets out
  of default CI unless a submodule or artifact is present.
- Add a CI matrix for macOS and Ubuntu only after the build scripts honestly
  support both.

### P0. Repair `source-policy-check`

Problem: the target is syntactically invalid and semantically inconsistent with
the repository.

Evidence:

- The shell block ends without `fi`.
- The repo had mixed orchestration artifacts in tracked source, including
  script sources and generated cache files.

Impact:

- `make check` is blocked before the real evidence gates can run.
- Contributors cannot tell whether Python is banned, retired, tolerated for
  utilities, or banned only from promoted evidence pipelines.

Recommended fix:

- Decide the actual policy. A likely good policy is:
  - no tracked `__pycache__` or `*.pyc` ever;
  - Python allowed in `scripts/` only for non-promoted utilities or tests;
  - promoted evidence paths must be C/runtime-owned unless explicitly exempted.
- Replace the current target with an allowlist-based check.
- Remove the tracked `.pyc` file from git.
- Make policy output actionable: print category, reason, and remediation.

### P0. Separate promoted evidence from sandboxes at the build boundary

Problem: the README correctly distinguishes promoted evidence from sandboxes,
but the build graph still interleaves them.

Impact:

- New experiments can accidentally break the promoted evidence path.
- CI time and failure modes become noisy.
- Reviewers cannot quickly answer "what must be correct for the paper claim?"

Recommended fix:

- Introduce explicit target tiers:
  - `make core`: core library and promoted task binaries.
  - `make evidence`: promoted traces, replay, paper-support, numeric gates.
  - `make sandbox`: ARC/haiku/program synthesis/holo/Go/Signal experiments.
  - `make integrations`: external sibling repos and live systems.
  - `make all`: maybe alias to `core`, not every experiment.
- Require every claim in `paper/claims.yaml` to name exactly one tier.
- Reject promoted-claim support if it depends on sandbox or integration artifacts
  not declared as such.

### P1. Modularize the largest files

Problem: several files are too large for effective review and change isolation.

Observed sizes:

- `examples/arc_agi.c`: about 26K lines.
- `src/nn_grammar_discovery.c`: about 19K lines.
- `examples/multitask_domain.c`: about 6.2K lines.
- `src/nn_paper.c`: about 5.8K lines.
- `src/nn.c`: about 5.1K lines.
- `tests/test_nn.c`: about 4.6K lines.

Impact:

- Higher risk of accidental coupling.
- Harder targeted review.
- Harder fuzzing and unit segmentation.
- Harder ownership boundaries between runtime, domain logic, and generated
  artifact compilation.

Recommended module splits:

- Split `src/nn.c` into:
  - `nn_alloc.c`
  - `nn_forward.c`
  - `nn_train.c`
  - `nn_checkpoint.c`
  - `nn_trace.c`
  - `nn_numeric_certificate.c`
- Split `examples/arc_agi.c` into a real `src/arc/` module:
  - grid representation and JSON loader;
  - transform primitives;
  - hypothesis induction;
  - program proposer;
  - CLI/report writer.
- Split `src/nn_grammar_discovery.c` by artifact family:
  - learned grammar coverage;
  - experiment health;
  - paper claim compilation;
  - ARC failure proof generation;
  - cache/parsing utilities.
- Split `tests/test_nn.c` into focused binaries matching modules.

Target rule: no first-party source file over 3000 lines without a written
justification and a module-extraction issue.

### P1. Keep generated evidence live, not theatrical

Problem: the paper-support machinery is strong, but parts of the evidence
surface appear to emit hard-coded or synthetic rows inside the runtime.

Examples:

- `src/nn_grammar_discovery.c` contains learned-grammar coverage rows where
  ARC task counts and metric JSON are literal strings.
- The ARC neural failure proof writer emits a fixed proof/replay row and summary
  text when generating failure-proof artifacts.

This is not automatically wrong. Fixture rows and deterministic synthetic
diagnostics are useful. The danger is that generated rows can look like measured
evidence unless the schema makes their status impossible to miss.

Recommended fix:

- Add an `evidence_origin` field to generated rows:
  - `measured_run`
  - `fixture_regression`
  - `synthetic_diagnostic`
  - `cached_snapshot`
  - `manual_annotation`
- Paper-support evaluators should reject `synthetic_diagnostic` for promoted
  result claims unless the claim explicitly allows it.
- Add artifact freshness metadata: git SHA, command, environment manifest,
  input hashes, generator binary hash, and timestamp policy.
- Build a `make evidence-audit` target that lists which claims depend on cached,
  synthetic, or underpowered rows.

### P1. Repair Makefile maintainability

Problem: the `Makefile` is doing too many jobs: build graph, experiment runner,
paper compiler, CI policy engine, external integration launcher, release helper,
and retired-workflow registry.

Evidence:

- The `.PHONY` line is enormous and difficult to review.
- Retired targets remain in the main graph.
- A stray recipe line appears under the `Holographic Prime Gap` comment without
  an obvious target.
- There are duplicate comments and appended target sections after `clean`.
- Several targets shell out to scripts even while the source policy says scripts
  were retired or should be ported into C.

Recommended fix:

- Keep top-level `Makefile` as a dispatcher only.
- Move target groups into included files:
  - `mk/core.mk`
  - `mk/promoted.mk`
  - `mk/sandbox.mk`
  - `mk/paper.mk`
  - `mk/integrations.mk`
  - `mk/release.mk`
- Add `make help` generated from target comments.
- Add `make lint-makefile` that catches missing `fi`, orphaned recipe lines,
  duplicate target names, and nonexistent dependencies.

### P1. Normalize artifact and generated-file hygiene

Problem: generated paper files and results live in git, while build artifacts
are mostly untracked. That can be correct for canonical snapshots, but the
current dirty working tree contains many generated files, making it hard to
distinguish source edits from regenerated evidence.

Recommended fix:

- Classify files into:
  - source of truth;
  - generated but committed canonical snapshot;
  - local generated artifact;
  - external/cached data.
- Add this classification to `README.md` and `.gitattributes`.
- Add a generated-file header to committed generated artifacts:

```text
generated_by: make paper-support
source_sha: ...
inputs_sha256: ...
do_not_edit: true
```

- Add `make generated-diff-check` to fail if generated files are stale relative
  to their declared inputs.

### P1. Treat external integrations as plugins

Problem: Signal and other external surfaces are valuable pressure tests, but
they make the repo non-hermetic.

Recommended fix:

- Move Signal build logic behind an explicit `WITH_SIGNAL=1` gate.
- If Signal is required for CI, use a git submodule or documented setup action.
- Otherwise, make Signal targets skip with a clear message when `../signal` is
  missing.
- Ensure promoted paper claims cannot silently depend on locally available
  sibling repositories unless the environment manifest records them.

### P2. Tighten C API hygiene

Small but telling examples:

- `include/nn_core.h` includes `<math.h>` twice.
- `NN_SCORER_ENVELOPE_EML_FEATURE_COUNT` is defined twice.
- `nn_set_layer_activation` is declared twice.

These are easy to fix, and doing so matters because headers define the public
surface. Header duplication signals that the system is being extended faster
than it is being pruned.

Recommended fix:

- Run include-what-you-use or a small custom header-lint script.
- Add a public-header review checklist.
- Keep `include/nn.h` as the umbrella, but make each subheader independently
  minimal and documented.

### P2. Improve warning and sanitizer discipline

Recommended policy:

- First-party core and promoted domain files: `-Werror` in CI.
- Sandbox files: warnings visible but non-blocking initially.
- Vendored files: compiled with a reduced warning profile or isolated in a
  vendor target.
- Sanitizer target should avoid `-ffast-math`; fast-math can invalidate some
  undefined-behavior and NaN/finite checks.
- Add UBSan/ASan nightly on a small deterministic promoted-evidence subset.

### P2. Reduce claim-surface drift

The support report currently marks many claims as supported, but also lists
underpowered or missing claims, including ARC verified self-play, cross-domain
structural transfer, Signal mining replay options, and multiple holographic
claims.

Recommended fix:

- Put missing/underpowered claims behind an explicit `aspirational` paper
  surface, not near result claims.
- Add a rule: required paper claims cannot depend on artifacts under
  `build/float` unless `make paper-support` can regenerate or verify them from
  committed inputs.
- Keep the phrase "underpowered" in generated output; it is honest and useful.
  But ensure underpowered claims are visually segregated from supported claims
  in the paper pipeline.

### P2. Split tests by contract tier

The current single test binary is rich but large. Recommended split:

- `test_nn_core`
- `test_nn_checkpoint_trace`
- `test_nn_experiment_protocol`
- `test_nn_symbolic`
- `test_prime_gap`
- `test_paper_claims`
- `test_arc_fixture`

This will speed local iteration and make CI failures more legible.

### P3. Add fuzz/property tests for parsers and JSONL contracts

High-value targets:

- trace parser;
- checkpoint loader;
- JSON cursor/string extraction;
- rule-file loader;
- ARC JSON loader;
- GTP or live-protocol parsers if Go/Signal remain active.

The repo already thinks in schemas. Fuzzing is the natural next step.

## Priority roadmap

### Week 1: restore trust in gates

1. Fix `source-policy-check` syntax and policy contradiction.
2. Remove tracked pyc files.
3. Split `make all` into portable core and optional integrations.
4. Add BLAS fallback or Linux CBLAS support.
5. Make CI run only hermetic targets by default.

### Week 2: stabilize evidence boundaries

1. Add `evidence_origin` to generated rows.
2. Add artifact freshness hashes to paper-support inputs.
3. Create `make evidence-audit`.
4. Move missing/underpowered claims into an explicit aspirational track.
5. Document generated-file policy.

### Month 1: reduce monolith risk

1. Extract `src/nn.c` forward/train/checkpoint/trace modules.
2. Extract ARC into `src/arc/` with a thin CLI wrapper.
3. Split `src/nn_grammar_discovery.c` by report family.
4. Split `tests/test_nn.c` into tiered test binaries.
5. Move Makefile groups into `mk/*.mk` includes.

### Month 2: make promoted research boringly reproducible

1. Hermetic promoted evidence runner.
2. Multi-seed CI/nightly reports for core claims.
3. Artifact diff tooling that distinguishes numeric drift from semantic drift.
4. Fuzz tests for parsers/loaders.
5. Formal promotion checklist for sandbox-to-promoted transitions.

## Suggested target taxonomy

The repo needs a vocabulary that maps directly to build targets and claims:

| Tier | Purpose | Examples | CI policy |
| --- | --- | --- | --- |
| core | library and contracts | `nn`, `nn_experiment`, `nn_protocol`, `nn_symbolic` | every push |
| promoted | canonical evidence | `prime_gap`, `grid_path`, `delayed_claim`, paper-support | every push or nightly depending cost |
| sandbox | exploratory research | ARC, haiku, program synthesis, holo, EML, Go | smoke only, non-blocking or scheduled |
| integration | external systems | Signal live/mining/flight if sibling repo needed | explicit opt-in |
| publication | paper generation and claim registry | Typst, bibliography, generated claims | PR paths and release |

The README already contains the conceptual distinction. The build system should
enforce it.

## Philosophical analysis

### 1. This repo is an epistemology engine

Most ML repositories ask, "Can the model predict?" This repository asks a more
interesting question: "What is allowed to count as knowing?" Its answer is not
confidence, loss, or leaderboard rank. Its answer is a chain:

```text
enumeration -> proposal -> grounding -> certificate/replay -> trace -> claim
```

That is an epistemology. Knowledge is not an internal state of the model; it is
a public relation between a candidate, an authority, and an auditable record.

This is why the repository's most important sentence is:

> The model proposes. The declared grounding operator decides within its
> authority. The trace remembers.

That line is a separation-of-powers doctrine for AI systems.

### 2. Neural-symbolic separation as constitutional design

The neural model is not sovereign. It is an elected parliament of guesses:
useful, adaptive, expressive, but not trusted with final authority. The verifier
is the judiciary: narrower, slower, more procedural, but empowered to say what
is valid inside its jurisdiction. The trace is the archive: it makes both model
and verifier accountable to future replay.

This constitutional pattern is the project's deepest contribution. The danger
is not that neural models are weak. The danger is that neural confidence can
impersonate authority. The repo resists that impersonation by forcing every
claim to name its court of appeal.

### 3. Falsifiability is treated as a first-class artifact

The support report's `underpowered` and `missing` statuses are philosophically
important. They show that the system is willing to say "not yet". That is rare
in research tooling. A paper pipeline that can refuse its own claims is a small
Popperian machine.

However, falsifiability can decay into theater. If generated rows are hard-coded
or stale, the ritual of verification may remain while the substance weakens.
The project should therefore treat artifact freshness as an ethical issue, not
just a build issue. A stale certificate is a broken promise.

### 4. The trace is memory, not truth

The trace remembers, but memory is not automatically truth. A trace records what
the system said happened under a particular schema, binary, environment, and
authority. Its truth depends on replay semantics staying stable. The repo
understands this through trace determinism and stable action IDs, but the next
step is to make environment and artifact provenance just as central as model
hashes.

In philosophical terms, the trace is not correspondence itself; it is the
material substrate that lets correspondence be tested again.

### 5. Sandboxes are hopes; promoted claims are debts

The repository contains many beautiful hopes: ARC visual algebra, holographic
memory, EML universality, live Signal control, self-play, hypothesis libraries,
program synthesis. These are valuable precisely because they are not all forced
to be evidence at once.

A sandbox is a place where an idea may be alive before it is true. A promoted
claim is different: it creates a debt to the reader. The reader is owed a clean
path from prose to artifact to replay. The repo's README already knows this. The
engineering challenge is to make it impossible for a hope to masquerade as a
paid debt.

### 6. Holographic memory is a compelling metaphor with a governance risk

The holographic/VSA thread has philosophical force: memory as fixed-dimensional
superposition, learning as storing, forgetting as graceful interference rather
than catastrophic deletion. This is a lovely counter-image to gradient descent.

But metaphors can outrun evidence. The support report currently marks several
holographic claims as missing. That is healthy. The project should keep the
holographic material as a named architecture axis, but require the same
certificate/replay discipline before it becomes a result. Otherwise the metaphor
will become decorative rather than scientific.

### 7. The paper compiler creates a recursion problem

The repo checks claims about the paper using code in the repo. That is useful
and unavoidable, but it creates a recursion:

```text
the system certifies claims about the system certifying claims
```

This is not invalid. All formal systems eventually lean on a trusted base. But
the trusted base should be small, boring, and easy to audit. Right now the claim
compiler and grammar-discovery machinery are large. Philosophically, the
solution is not to eliminate recursion; it is to minimize the kernel of trust.

Engineering translation: make the claim-evaluator core tiny, split from report
generation, and test it independently with fixtures that hostile reviewers would
understand.

### 8. The project is fighting Goodhart's law

Once claims are mechanically gated, contributors may optimize for passing the
gate rather than increasing truth. This is not hypothetical; every metric system
invites this. The repo's best defense is plural authority:

- exact checks for mathematical tasks;
- deterministic replay for simulator tasks;
- corpus proxy labels for language/ARC diagnostics;
- human review for editorial assistance;
- explicit claim scopes and known non-claims.

The more each authority states its limits, the harder it is for one metric to
become the whole world.

### 9. The deepest design principle: bounded trust

The repo's moral center is bounded trust. Trust the neural model to search, not
to certify. Trust the verifier inside its domain, not beyond it. Trust the trace
as replay material, not as eternal truth. Trust the paper compiler to catch
registered claim failures, not to decide research taste. Trust sandboxes to
generate possibilities, not to support conclusions.

This boundedness is the difference between ambitious engineering and mythology.

## Final assessment

This is a strong and unusual research codebase with a real thesis. Its core idea
is valuable: learned systems should make proposals inside explicit evidence
contracts, and claims should be compiled from replayable artifacts rather than
handwritten confidence. The repo has enough infrastructure to make that thesis
credible.

The immediate danger is not lack of ideas. It is excess surface area. Too many
experiments, generated artifacts, external dependencies, and orchestration paths
now share the same build and review channels. The next engineering win is to
make the promoted path small, portable, hermetic, and boring. Once that spine is
stable, the sandboxes can continue to bloom without threatening the truth-value
of the paper.

Recommended guiding rule for the next phase:

> Promote less, certify harder, and make every artifact say exactly what kind of
> evidence it is.
