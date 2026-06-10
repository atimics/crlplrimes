# AI Agent Next Steps: Taking `crlplrimes` to the Next Level

Date: 2026-06-07
Updated: 2026-06-09
Audience: AI coding/research agents working in this repository
Companion report: `docs/engineering-improvement-report-2026-06-07.md`

## Mission

Make `crlplrimes` a boringly reproducible research harness for certified
neural-symbolic action scoring.

The core invariant to preserve is:

```text
The model proposes. The declared grounding operator decides within its authority.
The trace remembers.
```

Your job is not to add maximum novelty. Your job is to make the promoted evidence
spine small, portable, hermetic, replayable, and difficult to overclaim.

## Operating principles for agents

1. Do not blur promoted evidence and sandboxes.
   - Promoted evidence must be reproducible from documented commands.
   - Sandboxes are allowed to be speculative, but must not support paper-facing
     claims without explicit promotion criteria.

2. Preserve user work.
   - Check `git status --short` before edits.
   - Do not overwrite dirty files unless the user explicitly asks.
   - If generated artifacts change as a side effect, say so clearly.

3. Prefer small mechanical fixes over broad rewrites.
   - Change one boundary at a time.
   - Add tests or checks near the contract being fixed.
   - Keep edits local to existing patterns.

4. Treat build and evidence failures as first-class signals.
   - A failing `make check` is not "just CI"; it is an epistemic break.
   - A stale or synthetic artifact must declare itself as such.

5. Keep claims scoped.
   - Exact checks support exact correctness only inside their declared domain.
   - Deterministic replay supports simulator/replay claims, not broad autonomy.
   - Corpus proxy rows support corpus-fit diagnostics, not semantic truth.
   - Human review supports review-gated assistance, not machine proof.

## Current known state

From the 2026-06-07 inspection:

- `make test` passed locally.
- `make -j4 all` passed locally, but with warnings and local-only assumptions.
- `make source-policy-check` failed with a shell syntax error.
- The default build assumes Apple Accelerate in `src/nn.c` and `Makefile`.
- `make all` includes Signal integration that depends on a sibling `../signal`
  checkout.
- The working tree was dirty before the report was written.

Do not assume a clean repo. Reconfirm before acting.

## Strategic objective

Build a clean evidence spine:

```text
core library
  -> promoted domain binaries
  -> deterministic traces/replay
  -> result snapshots
  -> paper-support claim compiler
  -> paper PDF
```

Everything outside that spine should be explicitly marked as sandbox or
integration.

## Holographic/VSA backlog alignment

The `AnOversizedMooseWithSocks/holostuff` repository was reviewed on
2026-06-09 as an external reference for VSA mechanisms. Treat it as design
inspiration only: no visible license file was present in the reviewed snapshot,
so agents must not vendor or copy code from it.

Use the following fit when working on the holographic backlog:

| Item | Tier | Direction |
| --- | --- | --- |
| VSA grid encoder and ARC candidate scoring | sandbox baseline | Keep instrumented. Full-scale raw-grid VSA persisted (`stored_count=114`) but did not improve the 70/400 ordered ARC baseline. |
| Object-centric scene vectors | sandbox baseline | Implemented as `arc_object_scene_v1`; persisted and scored cleanly but also did not improve the 70/400 ordered ARC baseline. |
| Scale-space / visual-program proposal coverage | sandbox, immediate workstream | `reflect_tile_{2,3,4}x` is verifier-backed and improves the `--unique-task-gate` induction run from 31 exact / 34 certified to 34 exact / 37 certified; `marker_motif_tile` moves it to 36 exact / 39 certified; `axis_tile` moves it to 42 exact / 45 certified; `period_extend` moves it again to 43 exact / 46 certified. `--arc-proposal-trace` now emits `crlp.arc_proposal_trace.v1` labels for family/slot proposal learning; online `trace_task_prior` beats structural and global-trace ordering at budget 64, while flat `trace_linear` and listwise `trace_mlp` trail it. Next lift should be confidence-gated residual neural heads around the prototype prior rather than more manual miss mining. |
| `memory_neuron.c` | sandbox/core-adjacent | Already exists as a prototype; harden with tests and real-loop integration. |
| Capacity-partitioned memory | sandbox | Add a C `HoloForest`-style ablation around `HolographicRanker`; report recall/cost rows before depending on it. |
| Shadow self-organization | sandbox | Judge split/merge/rebuild candidates on held-out verifier or replay rows before swapping. |
| Text generation, unified multimodal demos, image archive | sandbox/parked | Do not promote without a separate evidence proposal. |

The rule is the same as the rest of the repo: mechanisms may propose, route, or
rank, but declared grounding operators own correctness.

## Workstream 0: First response protocol for every agent

Before implementing any nontrivial change, run or inspect:

```sh
git status --short
make -n check
```

If the task touches build, CI, claims, or evidence, also inspect:

```sh
sed -n '1,220p' README.md
sed -n '1,220p' docs/engineering-improvement-report-2026-06-07.md
sed -n '1,220p' paper/generated/support_report.md
```

If the task touches promoted results, identify which claim IDs in
`paper/claims.yaml` are affected.

Acceptance criteria:

- Agent states whether the working tree was already dirty.
- Agent names the tier being touched: core, promoted, sandbox, integration, or
  publication.
- Agent states the minimal validation command it will run.

## Workstream 1: Restore `make check` trust

Priority: P0

Problem:

`source-policy-check` is syntactically broken and semantically conflicts with
tracked Python files.

Tasks:

1. Fix the shell syntax in `source-policy-check`.
2. Replace the current blanket Python ban with an explicit policy.
3. Remove tracked `*.pyc` and `__pycache__` artifacts from version control.
4. Make policy failures actionable.

Recommended policy:

- Tracked `*.pyc` and `__pycache__` are always forbidden.
- Python scripts are allowed under `scripts/` only when classified as utility,
  test, or retired helper.
- Promoted evidence targets must not depend on Python unless the dependency is
  explicitly documented and claim-scoped.

Suggested implementation:

```make
source-policy-check:
	@bad_cache=$$(find . -path ./.git -prune -o \( -name '*.pyc' -o -name '__pycache__' \) -print); \
	if [ -n "$$bad_cache" ]; then \
		echo "forbidden generated Python cache files:"; \
		echo "$$bad_cache"; \
		exit 1; \
	fi
```

Then add a second explicit check for promoted Make targets if needed.

Validation:

```sh
make source-policy-check
make test
```

Acceptance criteria:

- `make source-policy-check` exits 0 on an intended clean checkout.
- If it exits nonzero, it prints specific file paths and remediation.
- `make check` can proceed past source-policy validation.

## Workstream 2: Make the build portable

Priority: P0

Problem:

The core runtime directly depends on Apple Accelerate while CI declares Ubuntu.

Tasks:

1. Add a BLAS portability layer.
2. Support at least three modes:
   - Apple Accelerate on macOS.
   - CBLAS/OpenBLAS on Linux.
   - scalar fallback without BLAS.
3. Stop using `-framework Accelerate` as the unconditional default.
4. Update CI dependencies to match the selected Linux mode.

Suggested shape:

```c
#if defined(NN_USE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#elif defined(NN_USE_CBLAS)
#include <cblas.h>
#endif
```

Hide calls behind wrappers:

```c
static void nn_blas_gemv(...);
static void nn_blas_gemm(...);
```

Use scalar loops when neither BLAS macro is enabled.

Validation:

```sh
make clean
make test
make -j4 all-core
```

If Linux CI is available:

```sh
make test CC=gcc REAL=float
make test CC=gcc REAL=double
```

Acceptance criteria:

- Core tests pass without Apple-only headers.
- macOS still uses Accelerate when requested.
- Linux CI no longer refers to `-framework Accelerate`.

## Workstream 3: Split target tiers

Priority: P0

Problem:

The build interleaves core, promoted evidence, sandboxes, integrations, and paper
publication.

Tasks:

1. Add explicit target tiers:
   - `core`
   - `promoted`
   - `evidence`
   - `sandbox`
   - `integrations`
   - `publication`
2. Make `all` either alias `core` or document that it is expensive and
   non-hermetic.
3. Move Signal-dependent builds behind `WITH_SIGNAL=1` or an explicit
   `integrations` target.
4. Ensure CI uses hermetic targets by default.

Proposed semantics:

| Target | Meaning | External dependencies allowed? |
| --- | --- | --- |
| `core` | library and contract test binaries | no |
| `promoted` | prime/grid/delayed promoted binaries | no |
| `evidence` | canonical result and paper-support path | no, unless declared |
| `sandbox` | ARC, haiku, program synthesis, holo, EML, Go | maybe, but non-promoted |
| `integrations` | Signal/live/external systems | yes, explicit |
| `publication` | Typst, bibliography, generated paper files | limited/documented |

Validation:

```sh
make core
make promoted
make evidence
```

Acceptance criteria:

- `make core` works from a clean checkout without `../signal`.
- `make promoted` works without sandbox dependencies.
- `make integrations` clearly skips or fails with setup instructions when
  external repos are missing.

## Workstream 4: Evidence origin and freshness

Priority: P1

Problem:

Generated rows, cached snapshots, fixture diagnostics, and live measurements can
look too similar.

Tasks:

1. Add `evidence_origin` to generated JSONL rows where possible.
2. Add artifact freshness metadata to paper-support artifacts.
3. Teach paper-support evaluators to reject inappropriate origins for promoted
   claims.
4. Add an evidence audit command.

Suggested `evidence_origin` values:

- `measured_run`
- `fixture_regression`
- `synthetic_diagnostic`
- `cached_snapshot`
- `manual_annotation`

Suggested metadata fields:

```json
{
  "source_git_sha": "...",
  "generator": "...",
  "generator_hash": "...",
  "command": "make results",
  "input_hashes": {...},
  "environment_manifest": "..."
}
```

New target:

```sh
make evidence-audit
```

It should print:

- claim ID;
- support status;
- source artifact;
- evidence origin;
- freshness status;
- whether artifact is required for submission.

Validation:

```sh
make paper-support
make evidence-audit
```

Acceptance criteria:

- Promoted claims cannot be supported solely by `synthetic_diagnostic` rows
  unless explicitly allowed by the claim registry.
- Stale artifacts are visible in a single report.
- Underpowered/missing claims remain clearly separated from supported claims.

## Workstream 5: Reduce monolith risk

Priority: P1

Problem:

Several files are too large for safe review.

Targets:

- `examples/arc_agi.c`
- `src/nn_grammar_discovery.c`
- `src/nn.c`
- `src/nn_paper.c`
- `tests/test_nn.c`
- `Makefile`

Rules for agents:

- Do not rewrite these files wholesale.
- Extract one coherent module at a time.
- Preserve existing CLI behavior.
- Add or move tests with the module.

Recommended extraction order:

1. `src/nn.c`
   - `nn_forward.c`
   - `nn_train.c`
   - `nn_checkpoint.c`
   - `nn_trace.c`
   - `nn_numeric_certificate.c`

2. `tests/test_nn.c`
   - `test_nn_core.c`
   - `test_nn_checkpoint_trace.c`
   - `test_nn_experiment_protocol.c`
   - `test_prime_gap.c`
   - `test_paper_claims.c`

3. `Makefile`
   - `mk/core.mk`
   - `mk/promoted.mk`
   - `mk/sandbox.mk`
   - `mk/integrations.mk`
   - `mk/paper.mk`

4. `examples/arc_agi.c`
   - `src/arc/arc_grid.c`
   - `src/arc/arc_loader.c`
   - `src/arc/arc_transforms.c`
   - `src/arc/arc_hypothesis.c`
   - `src/arc/arc_cli.c`

Validation:

```sh
make test
make promoted
```

Acceptance criteria:

- No behavior changes without explicit tests or regenerated expected artifacts.
- Module extraction reduces line count of the original file.
- Public headers remain minimal.

## Workstream 6: Paper claim hygiene

Priority: P1

Problem:

The paper-support system is a strength, but it must remain more than a ritual.

Tasks:

1. Make missing and underpowered claims visually separate in generated reports.
2. Ensure required claims cannot pass with stale artifacts.
3. Add a small trusted kernel for claim evaluation fixtures.
4. Keep aspirational architecture claims out of result wording.

Agent checklist before editing paper or claims:

- Which claim IDs are affected?
- Is the claim required for submission?
- What artifact supports it?
- Can that artifact be regenerated by a documented command?
- Is the evidence exact, replay, proxy, review, fixture, synthetic, or cached?

Validation:

```sh
make paper-support
make paper-ci
```

Acceptance criteria:

- The support report is easy to scan by status.
- Claim evaluators have fixture tests.
- Required claims fail closed when artifacts are missing or stale.

## Workstream 7: Warnings, sanitizers, and fuzzing

Priority: P2

Tasks:

1. Make first-party core and promoted domains warning-clean.
2. Add `-Werror` for core/promoted CI after cleanup.
3. Isolate vendored warnings.
4. Add parser fuzz/property tests for:
   - trace JSONL parser;
   - checkpoint loader;
   - JSON cursor helpers;
   - rule-file loader;
   - ARC JSON loader.
5. Avoid `-ffast-math` in sanitizer builds unless specifically justified.

Validation:

```sh
make sanitize
make test
```

Acceptance criteria:

- Core/promoted warnings are zero or explicitly documented.
- Fuzz/property tests catch malformed inputs without crashes.
- Sanitizer configs prioritize correctness over speed.

## Workstream 8: Sandbox promotion process

Priority: P2

Problem:

The repo has many compelling sandboxes. They need a formal promotion path.

Promotion checklist for any sandbox:

1. Task schema emitted with grounding authority.
2. Candidate enumeration is complete and explicit.
3. Feature provenance audit passes.
4. Leakage policy is declared and tested.
5. Grounding operator is replayable or exactly checkable.
6. At least one trace/replay determinism test exists.
7. Baselines are matched-budget and included in result rows.
8. Multi-seed or stress rows exist where stochastic claims are made.
9. Claim registry entry scopes non-claims explicitly.
10. Paper-support evaluator fails closed.

Do not promote a sandbox because it is interesting. Promote it only when it can
pay the evidence debt.

## Recommended first five agent tasks

If you are the next agent and no more specific user request is given, do these
in order:

1. Add confidence-gated residual ARC proposal heads around `trace_task_prior`.
   Online prototype memory beats structural/global trace ordering at budget 64;
   flat `trace_linear` and listwise `trace_mlp` are measured and trail it.
2. Fix `source-policy-check` and remove tracked Python cache artifacts.
3. Add `core` and `promoted` targets that avoid Signal and sandbox dependencies.
4. Add a scalar fallback for `src/nn.c` BLAS calls.
5. Update CI to run the hermetic core/promoted path.
6. Add `evidence_origin` to the easiest generated report path and wire one
   paper-support check to use it.

Each task should be a separate small PR or commit.

## Commands agents should know

Fast local checks:

```sh
make test
make source-policy-check
```

Promoted evidence checks:

```sh
make prime-gap-eval
make grid-path-eval
make delayed-claim-eval
make paper-support
```

Broader checks:

```sh
make sanitize
make check
make experimental-check
```

Paper path:

```sh
make results
make paper-claims
make paper-support
make paper-pdf
make paper-ci
```

Use broader checks only after understanding their dependencies and runtime.

## Definition of "next level"

The repo reaches the next level when a new agent can clone it and answer these
questions mechanically:

1. What is core, promoted, sandbox, integration, and publication code?
2. Which claims are supported, underpowered, missing, or aspirational?
3. Which command regenerates each required artifact?
4. Which authority supports each claim?
5. Are the artifacts fresh relative to source and inputs?
6. Can the promoted evidence path run without private local state?
7. Can a reviewer replay traces without trusting the model?

Until those answers are boring, do not optimize for more demos.

## Final instruction to future agents

Be ambitious about the thesis and conservative about claims.

The repository already has enough imagination. What it needs now is a smaller
trusted base, cleaner boundaries, and evidence that remains meaningful after the
current agent leaves.

Promote less. Certify harder. Make the boring path sacred.
