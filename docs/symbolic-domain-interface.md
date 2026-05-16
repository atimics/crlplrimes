# Shared Symbolic-Domain Interface

`include/nn_symbolic.h` is the shared formal-system layer. A domain supplies
symbols, axioms, a rule grammar, a certificate checker, and optional gate/action
callbacks. The neural code sees generic rule rows; the domain owns soundness.

## Contract

The core objects are:

| Object | Role |
| --- | --- |
| `NNRule` | Generic symbolic rule with domain, type, template id, params, structural features, certificate status, and utility counters |
| `NNCertificate` | Soundness result from the domain checker: unchecked, rejected, or certified |
| `NNSymbolicDomain` | Domain descriptor plus rule/scorer feature schemas, leakage policy, enumeration, feature encoding, certificate checking, and gate/action execution when the production path implements them |
| `NNProgram` | Typed program/transform descriptor with input/output types, params, structural features, child program ids, canonical id, certificate, and utility counters |
| `NNRuleCoverage` | Shared bitset for rule-to-event residual coverage |
| `NNProgramCoverage` | Shared bitset for program-to-event residual coverage |
| `NNClosedLoopRulePolicy` | Generic residual-aware neural proposer trained from greedy coverage trajectories |

The separation is intentional:

```text
symbols + grammar + certificate checker  -> domain formal system
NNRule rows + feature provenance audit   -> shared row contract
residual coverage                        -> shared proposer substrate
Neural policy                            -> proposes next verifier call
Certificate checker                      -> decides whether a rule is sound
```

The certificate checker is "sound truth" for this project: it is hand-coded
outside the neural model, and accepted rules enter the gate only through that
checker.

## Typed Program Layer

`include/nn_program.h` adds the shared transform/program substrate that sits
beside `NNRule`. It is deliberately domain-neutral: ARC transforms, program
synthesis library abstractions, theorem-prover lemmas, and future object-level
operators can all be represented as typed candidates before a domain-specific
checker decides whether they are certified.

The core engine features are:

| Feature | API |
| --- | --- |
| Typed signatures | `NNProgramValueType` and `nn_program_is_composable()` prevent composing, for example, an `object -> mask` transform into a `grid -> object_set` extractor in the wrong order |
| Canonical identities | `nn_program_refresh_canonical_id()` derives a stable id and hash from domain, name, kind, signature, params, and child program ids |
| Program composition | `nn_program_compose()` creates a typed composition row and records both child canonical ids |
| Program features | `nn_program_encode_features()` emits a fixed structural feature row that learned proposers can share across domains |
| Program coverage | `NNProgramCoverage` mirrors rule coverage for program-to-event support, residual gain, and accepted-set coverage |
| Proposer evidence | `nn_program_write_proposer_summary_json()` emits `crlp.program_proposer.summary.v1` rows for random, grammar-order, structural, learned, closed-loop, and oracle program orderings |

This does not make ARC object-relation reasoning automatic. It gives each task
a first-class place to put learned transform induction or object-relation
candidates without inventing a new row format, coverage matrix, or evidence
schema per experiment.

## Implemented Domains

`examples/symbolic_domains.c` wires four domains through the same enumeration,
feature-encoding, and certificate interface. In this smoke binary, gate/action
execution callbacks are placeholders; the production gate paths still live in
their task binaries.

| Domain | Grammar | Certificate |
| --- | --- | --- |
| `prime_gap` | candidate modular filters, conditional gap modular filters | algebraic `gcd` compositeness proof |
| `grid_path` | edge legality, blocked-next legality | legality axiom: impossible moves cannot be optimal successors |
| `delayed_claim` | threshold rules over claim features and actions | finite-generator check with zero false rejections |
| `arc_agi` | copy, mask/intersection, union, object-select-overlap, color map | match all train pairs and require all train-consistent grammar rules to agree on the test grid |

Run:

```sh
make symbolic-domains
```

Current smoke output shape:

```text
crlp.symbolic_domain.summary.v1
crlp.symbolic_rule_proposer.summary.v1
```

`nn_symbolic_domain_is_valid()` refuses domains whose declared feature schemas
violate the domain leakage policy or whose encoded rule width does not match the
declared rule schema. The domain summary rows include the rule-feature leakage
policy and a `rule_feature_policy_compliant` flag. In the current smoke all rule
features are structural descriptors under `priors_allowed`; verifier-derived
counts remain labels or diagnostics, not proposer inputs.

The shared proposer smoke currently trains one small MLP on all enumerated rule
rows from all four domains using certificate status as the label. This proves
the interface and training path are unified; it is not yet evidence that the
learned proposer beats grammar-order baselines on every domain, and it is not
the closed-loop residual policy.

## Closed-Loop Policy

The closed-loop residual proposer lives in `src/nn_symbolic.c`:

```c
nn_closed_loop_rule_policy_train(...)
nn_closed_loop_rule_policy_select(...)
```

Domain code provides:

| Callback | Meaning |
| --- | --- |
| `NNResidualRuleFeatureFn` | Encode a rule conditional on accepted rules and residual coverage |
| `NNResidualRuleCandidateFn` | Decide whether a rule is eligible under the current accepted set |
| `NNResidualRuleKeepAcceptedFn` | Optional set-maintenance hook, e.g. drop accepted rules subsumed by a new rule |

`NNClosedLoopRulePolicyConfig` also carries an `NNFeatureSchema[]` and
`NNLeakagePolicy`. Prime's closed-loop rule search declares 15 structural rule
features plus 10 gate-derived residual-set features under `gate_allowed`; the
shared train/select functions reject missing schemas or verifier-derived
features. Prime rule search consumes this shared policy and contributes only
prime-specific feature encoding and subset logic. The `symbolic_domains` smoke
does not consume the closed-loop policy; it trains the simpler
certificate-status classifier described above.

## Remaining Work

The interface is generalized, but the task implementations are not equally
mature:

- `grid_path` and `delayed_claim` need production gate-ablation tables under
  the same matched-budget baselines as prime.
- The symbolic-domain smoke needs real `rule_rejects` and `verify_action`
  callbacks before it can exercise gate execution uniformly.
- The standalone ARC binary has a JSON directory loader, but the shared
  symbolic-domain smoke still uses one compiled sample and is not a general ARC
  solver.
- The standalone ARC loader now has typed `NNProgram` rows, a largest
  connected-component primitive, budget curves, and a learned program proposer
  over shared program features plus ARC task descriptors; the older shared
  symbolic-domain `object_select_overlap` rule is still an alias for
  intersection.
- The shared proposer should be evaluated against random, grammar-order,
  structural-heuristic, and oracle orderings on each non-ARC domain; ARC now
  emits matched program-proposer rows from the standalone loader path.
- The formal-system descriptor is C callbacks, not a declarative symbol and
  axiom language.
