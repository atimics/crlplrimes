# ilXyr and Braid Cloud Runs

CRLPLRIMES can use ilXyr as its research control plane and a GitHub Actions
runner as its current cloud execution machine. Braid owns immutable corpus
releases. These roles stay separate:

```text
Braid release -> ilXyr frozen family -> GitHub cloud runner -> CRLPLRIMES evidence
      data             authority              compute             claim compiler
```

ilXyr does not currently ship a cloud scheduler. Running ilXyr inside a cloud
runner keeps its existing admission, family, settlement, ledger, and evidence
rules without claiming that GitHub Actions has become part of the scientific
authority. The runner supplies compute. ilXyr owns what was allowed to run and
what outcome resolved.

The current ilXyr `local-command` adapter also requires `network=open` at
admission. The smoke runner makes no network requests, but its contract states
that adapter policy truthfully. A future isolated executor should replace this
instead of describing the current runner as network-denied.

## Hard boundaries

- A corpus-proxy job must name a Braid release. The bridge rejects
  `grounding_authority: corpus_proxy` with `corpus.kind: none`.
- The Braid repository commit, release ID, and `release.json` SHA-256 are frozen.
  The official Braid verifier runs before ilXyr freezes the family.
- A scientific job resolves `CURRENT` to the exact GitHub event commit before
  ilXyr freezes the family. A full commit may also be supplied directly. The
  ilXyr and optional Braid commits are always full hashes, and important source
  files carry separate SHA-256 bindings.
- The cloud bridge never invents forecasts, funding, outcome thresholds, or
  claims. Those belong in the reviewed native ilXyr family package.
- Passing through one cloud workflow does not raise evidence authority. A Braid
  corpus proxy stays a corpus proxy; an exact checker stays an exact checker.

The bridge contract is
[`integrations/ilxyr/cloud-job.schema.json`](../integrations/ilxyr/cloud-job.schema.json).
The executable validator and lifecycle driver is
[`scripts/ilxyr_cloud_bridge.mjs`](../scripts/ilxyr_cloud_bridge.mjs).

## Job layout

A job manifest binds four things:

1. the CRLPLRIMES source commit and important source-file hashes;
2. the native ilXyr family manifest and exact ilXyr commit;
3. either no corpus or one verified Braid release;
4. the output directory and cloud-artifact retention period.

For a Braid-backed job, `materialization.kind` can be `build` or `existing`.
`build` runs the pinned official Braid compiler against a checked-in
`braid.yaml` and then requires the frozen release ID and manifest hash to match.
`existing` expects the release under the checkout or supplied with
`--braid-release`. Large existing releases should come from private object
storage or Braid Hosted through a short-lived download. Download transport is
an execution-plane concern; the frozen release ID and hashes remain the same.

The checked-in family directory is a template. Immediately before ilXyr freezes
it, the bridge copies it into the output directory and replaces only declared
tokens such as `__CRLP_COMMIT__`, `__CRLP_ROOT__`, and
`__BRAID_RELEASE_ID__`. `__NODE_EXECUTABLE__` resolves to the exact Node binary
on the cloud runner. Any unknown token fails closed. The rendered package and
all of its hashes are uploaded with the ledger.

The rendered ilXyr family package uses deployment paths under `/workspace`:

```text
/workspace/crlplrimes
/workspace/ilxyr
/workspace/braid
```

The cloud workflow creates those paths before ilXyr starts. ilXyr then invokes
the declared absolute executable directly, with its normal cleared environment
and exact argument vector.

## Local contract check

```sh
make ilxyr-integration-check
```

This validates the strict bridge rules without contacting a cloud service or
running an experiment.

## Cloud use

Open the `ilXyr cloud experiment` workflow, enter the checked-in job-manifest
path, and leave `execute` off for the first pass. The ready-to-use infrastructure
job is `integrations/ilxyr/jobs/bridge-smoke.json`. It has no claim IDs, no
corpus, and no scientific result. That pass:

1. checks out the pinned ilXyr and optional Braid commits;
2. builds the control plane and Braid verifier;
3. verifies source and corpus identities;
4. freezes the family in a fresh ilXyr ledger;
5. runs every ilXyr admission and executable-materialization check;
6. uploads the ledger and bridge receipt.

After reviewing that artifact, run the same frozen job with `execute` enabled.
The second pass runs all family members, settles the frozen family rule, verifies
the ledger, and uploads the result. The experiment runner must still write the
registered CRLPLRIMES JSONL evidence artifacts that the claim compiler expects.

## EML experiment path

The EML scorer work should use this order:

1. write and review the ilXyr family contract, including matched parameter
   budgets, seeds, metrics, success/no-go rules, forecasts, and funding;
2. use Braid only for corpus-backed datasets; keep generated nonlinear tasks
   bound to their generator commit and configuration instead of pretending they
   are corpus evidence;
3. run the family on the cloud bridge;
4. convert the settled, hash-bound run artifacts into
   `results/eml/scorer_benchmarks.jsonl`;
5. let the existing claim compiler decide whether C11 is supported,
   underpowered, or a supported negative.

This preserves the useful part of all three systems: Braid governs data, ilXyr
governs research decisions, and CRLPLRIMES governs claim wording.
