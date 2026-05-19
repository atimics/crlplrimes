#ifndef NN_EXPERIMENT_H
#define NN_EXPERIMENT_H

#include <stddef.h>
#include <stdio.h>

#define NN_EXPERIMENT_SCHEMA "crlp.experiment_schema.v1"
#define NN_EXPERIMENT_SCHEMA_VERSION 1u

typedef enum {
    NN_FEATURE_STATE_OBSERVATION = 0,
    NN_FEATURE_STRUCTURAL_PRIOR = 1,
    NN_FEATURE_VERIFIER_DERIVED = 2,
    NN_FEATURE_DERIVED_FROM_GATE = 3,
} NNFeatureProvenance;

typedef enum {
    NN_LEAKAGE_FORBIDDEN = 0,
    NN_LEAKAGE_PRIORS_ALLOWED = 1,
    NN_LEAKAGE_LEGACY = 2,
    NN_LEAKAGE_GATE_ALLOWED = 3,
} NNLeakagePolicy;

typedef enum {
    NN_PROPOSER_SELECTION = 0,
    NN_PROPOSER_DISCOVERY = 1,
} NNProposerMode;

typedef enum {
    NN_GROUNDING_EXACT_CHECK = 0,
    NN_GROUNDING_DETERMINISTIC_REPLAY = 1,
    NN_GROUNDING_CORPUS_PROXY = 2,
    NN_GROUNDING_HUMAN_REVIEW = 3,
} NNGroundingAuthority;

typedef struct {
    const char *name;
    NNFeatureProvenance provenance;
    int enabled_by_default;
} NNFeatureSchema;

typedef struct {
    size_t total_features;
    size_t enabled_features;
    size_t structural_prior_features;
    size_t verifier_derived_features;
    size_t gate_derived_features;
    size_t enabled_structural_prior_features;
    size_t enabled_verifier_derived_features;
    size_t enabled_gate_derived_features;
    int scorer_input_is_policy_compliant;
    int has_warning;
} NNFeatureAudit;

typedef struct {
    const char *task_name;
    const char *role;
    const char *state_space;
    const char *state_distribution;
    const char *action_space;
    const char *candidate_enumerator;
    const char *observable_features;
    NNLeakagePolicy leakage_policy;
    const char *neural_scorer;
    NNGroundingAuthority grounding_authority;
    const char *grounding_operator;
    const char *exact_verifier;
    const char *verifier_feedback;
    const char *candidate_grammar;
    const char *outcome_row_schema;
    const char *claim_scope;
    const char *known_non_claims;
    const char *rule_grammar;
    const char *certificate_checker;
    const char *gate_executor;
    const char *proposer_inputs;
    const char *proposer_targets;
    NNProposerMode proposer_mode;
    const char *baseline_orderings;
    const char *generalization_axis;
    const char *trace_schema;
    const char *replay_protocol;
    const char *rl_protocol;
    const NNFeatureSchema *features;
    size_t feature_count;
} NNExperimentSpec;

const char *nn_feature_provenance_name(NNFeatureProvenance provenance);
const char *nn_leakage_policy_name(NNLeakagePolicy policy);
const char *nn_proposer_mode_name(NNProposerMode mode);
const char *nn_grounding_authority_name(NNGroundingAuthority authority);
int nn_leakage_policy_is_valid(NNLeakagePolicy policy);
int nn_grounding_authority_is_valid(NNGroundingAuthority authority);
int nn_feature_schema_is_valid(const NNFeatureSchema *features, size_t feature_count);
NNFeatureAudit nn_feature_schema_audit(const NNFeatureSchema *features,
                                       size_t feature_count,
                                       NNLeakagePolicy policy);
NNFeatureAudit nn_experiment_feature_audit(const NNExperimentSpec *spec);
int nn_experiment_spec_is_valid(const NNExperimentSpec *spec);
int nn_experiment_write_schema_json(FILE *fp, const NNExperimentSpec *spec);

#endif
