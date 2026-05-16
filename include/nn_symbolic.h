#ifndef NN_SYMBOLIC_H
#define NN_SYMBOLIC_H

#include "nn_core.h"
#include "nn_experiment.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#define NN_RULE_MAX_NAME 64u
#define NN_RULE_MAX_DOMAIN 32u
#define NN_RULE_MAX_PARAMS 12u
#define NN_RULE_MAX_FEATURES 64u
#define NN_CERTIFICATE_MAX_TEXT 160u
#define NN_SYMBOLIC_MAX_TYPE_COUNTS 8u

typedef enum {
    NN_RULE_KIND_REJECT_ACTION = 0,
    NN_RULE_KIND_TRANSFORM = 1,
    NN_RULE_KIND_THRESHOLD = 2,
    NN_RULE_KIND_ARC_TRANSFORM = 3,
    NN_RULE_KIND_CUSTOM = 255,
} NNRuleKind;

typedef enum {
    NN_CERTIFICATE_UNCHECKED = 0,
    NN_CERTIFICATE_REJECTED = 1,
    NN_CERTIFICATE_CERTIFIED = 2,
} NNCertificateStatus;

typedef struct {
    char name[NN_RULE_MAX_NAME];
    double value;
    int64_t integer;
} NNRuleParam;

typedef struct {
    NNCertificateStatus status;
    int deterministic;
    size_t checked_cases;
    size_t support;
    size_t false_rejections;
    char proof_kind[NN_CERTIFICATE_MAX_TEXT];
    char reason[NN_CERTIFICATE_MAX_TEXT];
} NNCertificate;

typedef struct {
    char domain[NN_RULE_MAX_DOMAIN];
    char name[NN_RULE_MAX_NAME];
    NNRuleKind kind;
    uint32_t template_id;
    NNRuleParam params[NN_RULE_MAX_PARAMS];
    size_t param_count;
    nn_real structural_features[NN_RULE_MAX_FEATURES];
    size_t structural_feature_count;
    NNCertificate certificate;
    int accepted;
    double utility;
    size_t train_rejects;
    size_t validation_rejects;
    size_t test_rejects;
    size_t train_false_rejections;
    size_t validation_false_rejections;
    size_t test_false_rejections;
} NNRule;

typedef struct {
    int correct;
    double utility;
    double cost;
    size_t target_action;
} NNVerifyResult;

struct NNSymbolicDomain;

typedef size_t (*NNSymbolicEnumerateRulesFn)(const struct NNSymbolicDomain *domain,
                                             NNRule *rules,
                                             size_t capacity);
typedef int (*NNSymbolicEncodeRuleFn)(const struct NNSymbolicDomain *domain,
                                      const NNRule *rule,
                                      nn_real *features,
                                      size_t capacity,
                                      size_t *feature_count_out);
typedef int (*NNSymbolicCheckCertificateFn)(const struct NNSymbolicDomain *domain,
                                            const NNRule *rule,
                                            NNCertificate *certificate);
typedef int (*NNSymbolicRuleRejectsFn)(const struct NNSymbolicDomain *domain,
                                       const NNRule *rule,
                                       const void *state,
                                       size_t action,
                                       int *rejects);
typedef int (*NNSymbolicVerifyActionFn)(const struct NNSymbolicDomain *domain,
                                        const void *state,
                                        size_t action,
                                        NNVerifyResult *result);

typedef struct NNSymbolicDomain {
    const char *name;
    const char *description;
    const char *symbols;
    const char *axioms;
    const char *grammar;
    size_t max_rule_count;
    size_t rule_feature_count;
    NNLeakagePolicy leakage_policy;
    const NNFeatureSchema *rule_feature_schema;
    size_t rule_feature_schema_count;
    const NNFeatureSchema *scorer_feature_schema;
    size_t scorer_feature_count;
    size_t max_action_count;
    void *context;
    NNSymbolicEnumerateRulesFn enumerate_rules;
    NNSymbolicEncodeRuleFn encode_rule;
    NNSymbolicCheckCertificateFn check_certificate;
    NNSymbolicRuleRejectsFn rule_rejects;
    NNSymbolicVerifyActionFn verify_action;
} NNSymbolicDomain;

typedef struct {
    uint64_t *bits;
    size_t rule_count;
    size_t event_count;
    size_t word_count;
} NNRuleCoverage;

typedef struct {
    size_t accepted_count;
    size_t event_count;
    size_t covered_events;
    size_t max_rules;
    size_t type_counts[NN_SYMBOLIC_MAX_TYPE_COUNTS];
} NNResidualSummary;

typedef int (*NNResidualRuleFeatureFn)(size_t rule_index,
                                       const size_t *accepted_indices,
                                       size_t accepted_count,
                                       const NNRuleCoverage *coverage,
                                       const uint64_t *covered,
                                       void *context,
                                       nn_real *features,
                                       size_t capacity);
typedef int (*NNResidualRuleCandidateFn)(size_t rule_index,
                                         const size_t *accepted_indices,
                                         size_t accepted_count,
                                         void *context);
typedef int (*NNResidualRuleKeepAcceptedFn)(size_t accepted_rule_index,
                                            size_t candidate_rule_index,
                                            void *context);

typedef struct {
    size_t max_rules;
    size_t feature_count;
    size_t max_train_rules_per_step;
    size_t epochs;
    size_t batch_size;
    size_t hidden1;
    size_t hidden2;
    NNLeakagePolicy leakage_policy;
    const NNFeatureSchema *feature_schema;
    size_t feature_schema_count;
    nn_real learning_rate;
    unsigned int seed;
    int verbose;
} NNClosedLoopRulePolicyConfig;

typedef struct {
    NeuralNet *net;
    double loss;
    size_t sample_count;
} NNClosedLoopRulePolicy;

void nn_certificate_clear(NNCertificate *certificate);
void nn_certificate_accept(NNCertificate *certificate,
                           const char *proof_kind,
                           const char *reason,
                           size_t checked_cases,
                           size_t support);
void nn_certificate_reject(NNCertificate *certificate,
                           const char *proof_kind,
                           const char *reason,
                           size_t checked_cases,
                           size_t false_rejections);
const char *nn_certificate_status_name(NNCertificateStatus status);

void nn_rule_clear(NNRule *rule);
void nn_rule_init(NNRule *rule,
                  const char *domain,
                  const char *name,
                  NNRuleKind kind,
                  uint32_t template_id);
int nn_rule_add_param(NNRule *rule, const char *name, double value, int64_t integer);
int nn_rule_get_param(const NNRule *rule, const char *name, double *value_out);
int nn_rule_get_param_i64(const NNRule *rule, const char *name, int64_t *value_out);
int nn_rule_add_feature(NNRule *rule, nn_real value);
NNFeatureAudit nn_symbolic_domain_rule_feature_audit(const NNSymbolicDomain *domain);
NNFeatureAudit nn_symbolic_domain_scorer_feature_audit(const NNSymbolicDomain *domain);
int nn_symbolic_domain_is_valid(const NNSymbolicDomain *domain);

size_t nn_symbolic_domain_enumerate_rules(const NNSymbolicDomain *domain,
                                          NNRule *rules,
                                          size_t capacity);
int nn_symbolic_domain_encode_rule(const NNSymbolicDomain *domain,
                                   const NNRule *rule,
                                   nn_real *features,
                                   size_t capacity,
                                   size_t *feature_count_out);
int nn_symbolic_domain_check_certificate(const NNSymbolicDomain *domain,
                                         NNRule *rule,
                                         NNCertificate *certificate_out);
int nn_symbolic_domain_rule_rejects(const NNSymbolicDomain *domain,
                                    const NNRule *rule,
                                    const void *state,
                                    size_t action,
                                    int *rejects);
int nn_symbolic_domain_verify_action(const NNSymbolicDomain *domain,
                                     const void *state,
                                     size_t action,
                                     NNVerifyResult *result);

int nn_rule_coverage_init(NNRuleCoverage *coverage,
                          size_t rule_count,
                          size_t event_count);
void nn_rule_coverage_free(NNRuleCoverage *coverage);
void nn_rule_coverage_clear(NNRuleCoverage *coverage);
int nn_rule_coverage_set(NNRuleCoverage *coverage, size_t rule_index, size_t event_index);
int nn_rule_coverage_get(const NNRuleCoverage *coverage, size_t rule_index, size_t event_index);
size_t nn_rule_coverage_count_covered(const uint64_t *covered, size_t word_count);
size_t nn_rule_coverage_gain(const NNRuleCoverage *coverage,
                             size_t rule_index,
                             const uint64_t *covered);
int nn_rule_coverage_add_rule_to_residual(const NNRuleCoverage *coverage,
                                          size_t rule_index,
                                          uint64_t *covered);

void nn_residual_summary_clear(NNResidualSummary *summary);
void nn_residual_summary_from_coverage(NNResidualSummary *summary,
                                       size_t accepted_count,
                                       size_t max_rules,
                                       const NNRuleCoverage *coverage,
                                       const uint64_t *covered);
size_t nn_residual_summary_write_features(const NNResidualSummary *summary,
                                          nn_real *features,
                                          size_t capacity);

void nn_closed_loop_rule_policy_config_defaults(NNClosedLoopRulePolicyConfig *config,
                                                size_t feature_count);
void nn_closed_loop_rule_policy_clear(NNClosedLoopRulePolicy *policy);
void nn_closed_loop_rule_policy_free(NNClosedLoopRulePolicy *policy);
int nn_closed_loop_rule_policy_train(const NNRuleCoverage *coverage,
                                     const NNClosedLoopRulePolicyConfig *config,
                                     NNResidualRuleFeatureFn encode_features,
                                     NNResidualRuleCandidateFn candidate_allowed,
                                     NNResidualRuleKeepAcceptedFn keep_accepted,
                                     void *context,
                                     NNClosedLoopRulePolicy *policy);
int nn_closed_loop_rule_policy_select(const NNRuleCoverage *coverage,
                                      const NNClosedLoopRulePolicyConfig *config,
                                      NNResidualRuleFeatureFn encode_features,
                                      NNResidualRuleCandidateFn candidate_allowed,
                                      NNResidualRuleKeepAcceptedFn keep_accepted,
                                      void *context,
                                      NNClosedLoopRulePolicy *policy,
                                      nn_real *scores,
                                      size_t *accepted_indices,
                                      size_t accepted_capacity,
                                      size_t *accepted_count_out,
                                      size_t *proposal_calls_out);

#endif
