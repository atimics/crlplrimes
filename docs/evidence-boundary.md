# Evidence Boundary

`make evidence` is the local truth pass for promoted evidence. It intentionally
stays smaller than the full paper and sandbox surface so a reviewer can rerun the
claim-bearing spine without building every exploratory runtime.

## Promoted Spine

Promoted evidence currently covers:

- `prime_gap`: trace-first symbolic rule search, replay, checkpoint comparison,
  numeric certificate rows, and trace determinism.
- `grid_path`: exact grid-path grounding and promoted result rows.
- `delayed_claim`: exact delayed-claim grounding and promoted result rows.

The `evidence` target owns source policy checks, promoted binaries, promoted
result rows, promoted numeric certificates, and trace determinism comparison.
`make results` writes the promoted comparison artifacts under `results/latest`.

## Sandbox Surface

ARC, Signal, haiku, program synthesis, multitask, holographic/VSA, and neural
grammar discovery are application-pressure or research sandboxes. They are useful
for finding pressure points, but they are not promoted evidence until their
inputs, baselines, replay, and result capture are owned by the promoted gate.

Use:

- `make sandbox` for non-promoted smoke coverage.
- `make experimental-results` when deliberately refreshing sandbox result rows.
- `make paper-support` when checking the broader paper-support boundary.

## Promotion Checklist

Before moving a sandbox into promoted evidence, require:

1. Hermetic replay inputs available from the repo or deterministic build steps.
2. CI availability without private local state or machine-specific services.
3. Matched baselines at the same budget, capacity, and verifier access.
4. Result rows captured by `make results` and gated by `make evidence`.
5. Numeric certificates and trace determinism when training or model traces are
   part of the claim.
6. A declared grounding operator, authority level, and known non-claims.
7. Leakage-policy review for datasets, generated traces, and proposer inputs.

Promotion should make the claim more boring, not merely more impressive.
