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
| `NNSymbolicDomain` | Domain descriptor plus callbacks for enumeration, feature encoding, certificate checking, gate execution, and action verification |
| `NNRuleCoverage` | Shared bitset for rule-to-event residual coverage |
| `NNClosedLoopRulePolicy` | Generic residual-aware neural proposer trained from greedy coverage trajectories |

The separation is intentional:

```text
symbols + grammar + certificate checker  -> domain formal system
NNRule rows + residual coverage          -> shared proposer substrate
Neural policy                            -> proposes next verifier call
Certificate checker                      -> decides whether a rule is sound
```

The certificate checker is "sound truth" for this project: it is hand-coded
outside the neural model, and accepted rules enter the gate only through that
checker.

## Implemented Domains

`examples/symbolic_domains.c` wires four domains through the same interface:

| Domain | Grammar | Certificate |
| --- | --- | --- |
| `prime_gap` | candidate modular filters, conditional gap modular filters | algebraic `gcd` compositeness proof |
| `grid_path` | edge legality, blocked-next legality | legality axiom: impossible moves cannot be optimal successors |
| `delayed_claim` | threshold rules over claim features and actions | finite-generator check with zero false rejections |
| `arc_agi` | copy, mask/intersection, union, object select, color map | match all train pairs and require all train-consistent grammar rules to agree on the test grid |

Run:

```sh
make symbolic-domains
```

Current smoke output shape:

```text
crlp.symbolic_domain.summary.v1
crlp.symbolic_rule_proposer.summary.v1
```

The shared proposer currently trains one small MLP on all enumerated rule rows
from all four domains. This proves the interface and training path are unified;
it is not yet evidence that the learned proposer beats grammar-order baselines
on every domain.

## Closed-Loop Policy

The closed-loop residual proposer now lives in `src/nn_symbolic.c`:

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

Prime rule search now uses this shared policy and only contributes prime-specific
feature encoding and subset logic.

## Remaining Work

The interface is now generalized, but the task implementations are not equally
mature:

- `grid_path` and `delayed_claim` still need production gate-ablation tables
  under the same matched-budget baselines as prime.
- ARC is one compiled sample, not a dataset loader or a general ARC solver.
- The shared proposer should be evaluated against random, grammar-order,
  structural-heuristic, and oracle orderings on each domain.
- The formal-system descriptor is still C callbacks, not a declarative symbol
  and axiom language.
