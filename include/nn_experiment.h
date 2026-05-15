#ifndef NN_EXPERIMENT_H
#define NN_EXPERIMENT_H

#include <stdio.h>

#define NN_EXPERIMENT_SCHEMA "crlp.experiment_schema.v1"
#define NN_EXPERIMENT_SCHEMA_VERSION 1u

typedef struct {
    const char *task_name;
    const char *role;
    const char *state_space;
    const char *state_distribution;
    const char *action_space;
    const char *candidate_enumerator;
    const char *observable_features;
    const char *leakage_policy;
    const char *neural_scorer;
    const char *exact_verifier;
    const char *verifier_feedback;
    const char *rule_grammar;
    const char *certificate_checker;
    const char *gate_executor;
    const char *proposer_inputs;
    const char *proposer_targets;
    const char *baseline_orderings;
    const char *generalization_axis;
    const char *trace_schema;
    const char *replay_protocol;
    const char *rl_protocol;
} NNExperimentSpec;

int nn_experiment_spec_is_valid(const NNExperimentSpec *spec);
int nn_experiment_write_schema_json(FILE *fp, const NNExperimentSpec *spec);

#endif
