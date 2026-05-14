#ifndef PRIME_GAP_INTERNAL_H
#define PRIME_GAP_INTERNAL_H

#include "nn.h"
#include "nn_tasks_prime_gap.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef NN_PRIME_GAP_G_MAX
#define NN_PRIME_GAP_G_MAX 256
#endif

#ifndef NN_PRIME_GAP_FEATURE_MOD_COUNT
#define NN_PRIME_GAP_FEATURE_MOD_COUNT 10
#endif

#ifndef NN_PRIME_GAP_HIDDEN1
#define NN_PRIME_GAP_HIDDEN1 32
#endif

#ifndef NN_PRIME_GAP_HIDDEN2
#define NN_PRIME_GAP_HIDDEN2 16
#endif

#ifndef NN_SYMBOLIC_RULE_MAX_MOD
#define NN_SYMBOLIC_RULE_MAX_MOD 255
#endif

#ifndef NN_SYMBOLIC_MAX_LEARNED_RULES
#define NN_SYMBOLIC_MAX_LEARNED_RULES 64
#endif

enum {
    G_MAX = NN_PRIME_GAP_G_MAX,
    EXACT_ACTION_COUNT = G_MAX / 2,
    OVERFLOW_ACTION = EXACT_ACTION_COUNT,
    ACTION_COUNT = EXACT_ACTION_COUNT + 1,
    FEATURE_MOD_COUNT = NN_PRIME_GAP_FEATURE_MOD_COUNT,
    FEATURE_COUNT = 3 + FEATURE_MOD_COUNT,
    CONDITION_FEATURE_MOD_COUNT = FEATURE_MOD_COUNT <= 16 ? FEATURE_MOD_COUNT : 16,
    CONDITION_SIGNATURE_COUNT = (1 << CONDITION_FEATURE_MOD_COUNT) + 1,
    DEFAULT_TRAIN_LIMIT = 1000000,
    DEFAULT_TEST_LIMIT = 3000000,
    DEFAULT_SUPERVISED_EPOCHS = 2,
    DEFAULT_RL_EPOCHS = 1,
    DEFAULT_MAX_TRAIN_STATES = 30000,
    DEFAULT_MAX_TEST_STATES = 30000,
    CALIBRATION_BINS = 10,
    PRIME_BUCKET_COUNT = 4,
    SYMBOLIC_RULE_MOD_MIN = 3,
    SYMBOLIC_RULE_FEATURE_COUNT = 4,
    SYMBOLIC_RULE_SCORER_EPOCHS = 20,
};

extern const double DEFAULT_REGULARIZED_RL_REFERENCE_WEIGHT;
extern const double DEFAULT_REGULARIZED_RL_ENTROPY_WEIGHT;
extern const uint32_t FEATURE_MODS[];

typedef struct {
    const char *name;
    int use_log;
    int use_gap;
    int use_overflow;
    int use_residue;
} FeatureMask;

extern const FeatureMask FEATURE_MASK_FULL;
extern const FeatureMask FEATURE_MASK_MINIMAL;
extern const FeatureMask FEATURE_MASK_LOG_ONLY;
extern const FeatureMask FEATURE_MASK_GAP_ONLY;
extern const FeatureMask FEATURE_MASK_RESIDUE_ONLY;

typedef struct {
    uint8_t *composite;
    uint32_t *primes;
    size_t prime_count;
    uint32_t limit;
} Sieve;

typedef struct {
    uint32_t *prime;
    uint32_t *gap;
    size_t total_seen;
    size_t count;
    size_t capacity;
    size_t overflow_count;
} GapDataset;

typedef struct {
    size_t *action_counts;
    size_t condition_totals[CONDITION_SIGNATURE_COUNT];
} ConditionedFrequencyPolicy;

typedef struct {
    size_t candidates;
    size_t mod_tests;
    size_t exact_tests;
} VerifierWork;

typedef struct {
    uint32_t modulus;
    uint32_t residue;
    size_t train_rejects;
    size_t validation_rejects;
    nn_real model_score;
    int domain_valid;
} SymbolicRejectRule;

typedef struct {
    SymbolicRejectRule *items;
    size_t count;
    size_t capacity;
} SymbolicRuleSet;

typedef struct {
    size_t proposed_rules;
    size_t domain_valid_rules;
    size_t accepted_rules;
    size_t train_rejects;
    size_t validation_rejects;
    double proposer_loss;
    size_t samples;
    size_t test_false_rejections;
    double direct_exact_tests;
    double hand_wheel_mod_tests;
    double hand_wheel_exact_tests;
    double learned_mod_tests;
    double learned_exact_tests;
} SymbolicWheelMetrics;

enum {
    RULE_SEARCH_GRAMMAR_VERSION = 2,
    RULE_SEARCH_MAX_MODULI = 256,
    RULE_SEARCH_FEATURE_COUNT = 13,
    RULE_SEARCH_SCORER_EPOCHS = 24,
};

typedef enum {
    RULE_FILTER_CANDIDATE_MOD = 0,
    RULE_FILTER_GAP_MOD = 1,
} RuleFilterKind;

typedef struct {
    uint32_t modulus;
    uint32_t residue;
} RuleStateAtom;

typedef struct {
    RuleFilterKind kind;
    uint32_t modulus;
    uint32_t residue;
} RuleFilterAtom;

typedef struct {
    RuleFilterKind kind;
    uint32_t state_modulus;
    uint32_t state_residue;
    uint32_t filter_modulus;
    uint32_t filter_residue;
    size_t train_rejects;
    size_t train_false_rejections;
    size_t validation_rejects;
    size_t validation_false_rejections;
    nn_real model_score;
    double utility;
    int accepted;
} ExpandedRule;

typedef struct {
    ExpandedRule *items;
    size_t count;
} ExpandedRuleSet;

typedef struct {
    RuleStateAtom *state_atoms;
    RuleFilterAtom *filter_atoms;
    size_t *state_offsets;
    size_t *candidate_offsets;
    size_t *gap_offsets;
    size_t state_count;
    size_t filter_count;
} RuleSearchGrammar;

typedef struct {
    size_t train_limit;
    size_t test_limit;
    size_t max_train_states;
    size_t max_test_states;
    size_t max_rules;
    uint32_t moduli[RULE_SEARCH_MAX_MODULI];
    size_t mod_count;
    int include_state_conditions;
    int include_candidate_filters;
    int include_gap_filters;
    const char *trace_path;
    const char *save_path;
} RuleSearchConfig;

typedef struct {
    size_t proposed_rules;
    size_t verified_rules;
    size_t accepted_rules;
    size_t accepted_conditional_rules;
    size_t accepted_gap_rules;
    size_t accepted_candidate_rules;
    size_t unsafe_rules;
    size_t no_support_rules;
    size_t uncertified_rules;
    size_t samples;
    size_t false_rejections;
    double proposer_loss;
    double direct_exact_tests;
    double hand_wheel_exact_tests;
    double expanded_exact_tests;
    double expanded_mod_tests;
} RuleSearchMetrics;

typedef struct {
    size_t candidates;
    size_t base_primes;
    size_t marks;
    size_t survivor_checks;
} BlockWork;

typedef struct {
    size_t samples;
    size_t top1;
    size_t top5;
    size_t top10;
    size_t top1_prime;
    size_t correct_top1_confidence_count;
    size_t incorrect_top1_confidence_count;
    size_t high_confidence;
    size_t high_confidence_correct;
    size_t verifier_samples;
    size_t calibration_count[CALIBRATION_BINS];
    size_t bucket_samples[PRIME_BUCKET_COUNT];
    size_t bucket_top1[PRIME_BUCKET_COUNT];
    size_t bucket_top5[PRIME_BUCKET_COUNT];
    size_t bucket_top10[PRIME_BUCKET_COUNT];
    double nll;
    double true_action_probability;
    double mean_true_rank;
    double mean_reciprocal_rank;
    double top1_confidence;
    double correct_top1_confidence;
    double incorrect_top1_confidence;
    double expected_calibration_error;
    double calibration_confidence[CALIBRATION_BINS];
    double calibration_correct[CALIBRATION_BINS];
    double bucket_nll[PRIME_BUCKET_COUNT];
    double bucket_true_rank[PRIME_BUCKET_COUNT];
    double mean_abs_gap_error;
    double direct_scan_exact_tests;
    double wheel_scan_mod_tests;
    double wheel_scan_exact_tests;
    double proposal_steps_top10;
    double assisted_wheel_mod_tests_top10;
    double assisted_wheel_exact_tests_top10;
    double block_scan_cost;
    double assisted_block_cost_top10;
    double scorer_seconds;
    double verifier_seconds;
} EvalMetrics;

typedef struct {
    size_t train_limit_arg;
    size_t test_limit_arg;
    size_t supervised_epochs;
    size_t rl_epochs;
    size_t max_train_states;
    size_t max_test_states;
    double regularized_reference_weight;
    double regularized_entropy_weight;
    size_t rng_seed_arg;
    const char *trace_path;
    const char *save_path;
    const char *load_model_path;
    const char *save_rules_path;
    const char *load_rules_path;
    const char *save_learned_model_path;
    const FeatureMask *feature_mask;
} PrimeGapRunConfig;

double now_seconds(void);
uint32_t rng_next(uint32_t *state);
double rng_uniform(uint32_t *state);
size_t rng_bounded(uint32_t *state, size_t bound);
int parse_size_arg(int argc, char **argv, int index, size_t fallback, size_t *value_out);
int parse_double_arg(int argc, char **argv, int index, double fallback, double *value_out);
int parse_size_text(const char *text, size_t *value_out);
int parse_u32_text(const char *text, uint32_t *value_out);
const FeatureMask *feature_mask_from_name(const char *name);
int parse_run_config(int argc, char **argv, PrimeGapRunConfig *config);
int run_config_is_valid(const PrimeGapRunConfig *config);
void print_usage(FILE *stream);

uint32_t validation_start_limit(uint32_t train_limit);
void free_sieve(Sieve *sieve);
int build_sieve(uint32_t limit, Sieve *sieve);
int is_prime_sieve(const Sieve *sieve, uint32_t n);
void free_gap_dataset(GapDataset *dataset);
int append_gap(GapDataset *dataset, uint32_t prime, uint32_t gap);
int build_gap_datasets(const Sieve *sieve,
                       uint32_t train_limit,
                       uint32_t test_limit,
                       GapDataset *train,
                       GapDataset *validation,
                       GapDataset *test);
void shuffle_order(size_t *order, size_t count, uint32_t *rng);
void fill_order(size_t *order, size_t count);
size_t action_gap(size_t action);
size_t gap_to_action(uint32_t gap);
int action_is_overflow(size_t action);

double gap_error_for_action(size_t action, uint32_t true_gap);
void fill_candidate_features_masked(nn_real *inputs,
                                    uint32_t p,
                                    double log_scale,
                                    const FeatureMask *mask);
size_t residue_signature(uint32_t p, size_t action, size_t mod_count);
void apply_feature_wheel_filter(uint32_t p,
                                nn_real *scores,
                                nn_real *probabilities);
void fill_candidate_features(nn_real *inputs, uint32_t p, double log_scale);
void rank_actions(const nn_real *scores, size_t *ranked);
size_t true_rank(const size_t *ranked, size_t true_action);
size_t proposal_steps_with_topk(const size_t *ranked, size_t top_k, size_t true_action);
VerifierWork direct_scan_work(uint32_t true_gap);
VerifierWork wheel_scan_work(const Sieve *sieve, uint32_t p, uint32_t true_gap);
VerifierWork assisted_wheel_work(const Sieve *sieve,
                                 uint32_t p,
                                 uint32_t true_gap,
                                 const size_t *ranked,
                                 size_t top_k);

double block_work_cost(BlockWork work);
BlockWork block_scan_work(const Sieve *sieve, uint32_t p, uint32_t true_gap);
BlockWork assisted_block_work(const Sieve *sieve,
                              uint32_t p,
                              uint32_t true_gap,
                              const size_t *ranked,
                              size_t top_k);

size_t sample_action(const nn_real *probabilities, uint32_t *rng);
void softmax_temperature(const nn_real *scores,
                         nn_real *probabilities,
                         size_t count,
                         double temperature);
double distribution_entropy(const nn_real *probabilities, size_t count);
double distribution_kl(const nn_real *reference,
                       const nn_real *policy,
                       size_t count);
size_t eval_sample_index(const GapDataset *dataset, size_t sample, size_t samples);
void build_frequency_policy(const GapDataset *dataset,
                            size_t *ranked,
                            nn_real *probabilities);
void free_conditioned_frequency_policy(ConditionedFrequencyPolicy *policy);
int build_conditioned_frequency_policy(ConditionedFrequencyPolicy *policy,
                                       const GapDataset *dataset);
void conditioned_frequency_scores(const ConditionedFrequencyPolicy *policy,
                                  uint32_t p,
                                  nn_real *scores,
                                  nn_real *probabilities);
EvalMetrics evaluate_frequency_policy(const Sieve *sieve,
                                      const GapDataset *dataset,
                                      const nn_real *base_probabilities,
                                      size_t max_states,
                                      int include_verifier,
                                      int apply_wheel_filter);
EvalMetrics evaluate_conditioned_frequency_policy(
    const ConditionedFrequencyPolicy *policy,
    const Sieve *sieve,
    const GapDataset *dataset,
    size_t max_states,
    int include_verifier,
    int apply_wheel_filter,
    nn_real *scores,
    nn_real *probabilities);
EvalMetrics evaluate_model(NeuralNet *net,
                           const Sieve *sieve,
                           const GapDataset *dataset,
                           size_t max_states,
                           double log_scale,
                           const FeatureMask *mask,
                           int include_verifier,
                           double temperature,
                           nn_real *candidate_inputs,
                           nn_real *scores,
                           nn_real *probabilities);
EvalMetrics evaluate_model_symbolic_masked(NeuralNet *net,
                                           const Sieve *sieve,
                                           const SymbolicRuleSet *rules,
                                           const GapDataset *dataset,
                                           size_t max_states,
                                           double log_scale,
                                           const FeatureMask *mask,
                                           int include_verifier,
                                           double temperature,
                                           nn_real *candidate_inputs,
                                           nn_real *scores,
                                           nn_real *probabilities);
EvalMetrics evaluate_model_expanded_masked(NeuralNet *net,
                                           const Sieve *sieve,
                                           const ExpandedRuleSet *rules,
                                           const GapDataset *dataset,
                                           size_t max_states,
                                           double log_scale,
                                           const FeatureMask *mask,
                                           int include_verifier,
                                           double temperature,
                                           nn_real *candidate_inputs,
                                           nn_real *scores,
                                           nn_real *probabilities);
EvalMetrics evaluate_model_feature_wheel_masked(NeuralNet *net,
                                                const Sieve *sieve,
                                                const GapDataset *dataset,
                                                size_t max_states,
                                                double log_scale,
                                                const FeatureMask *mask,
                                                int include_verifier,
                                                double temperature,
                                                nn_real *candidate_inputs,
                                                nn_real *scores,
                                                nn_real *probabilities);
double fit_temperature(NeuralNet *net,
                       const GapDataset *dataset,
                       size_t max_states,
                       double log_scale,
                       const FeatureMask *mask,
                       nn_real *candidate_inputs,
                       nn_real *scores,
                       nn_real *probabilities,
                       double *best_nll);
double fit_temperature_symbolic_masked(NeuralNet *net,
                                       const SymbolicRuleSet *rules,
                                       const GapDataset *dataset,
                                       size_t max_states,
                                       double log_scale,
                                       const FeatureMask *mask,
                                       nn_real *candidate_inputs,
                                       nn_real *scores,
                                       nn_real *probabilities,
                                       double *best_nll);
int prime_action_from_action_id(uint32_t action_id, size_t *out_action);
void fill_action_ids(uint32_t *action_ids);
size_t action_row_for_action_id(const uint32_t *action_ids,
                                size_t count,
                                uint32_t action_id);
size_t survivor_index_for_action(const size_t *survivor_actions,
                                 size_t candidate_count,
                                 size_t action);
size_t fill_symbolic_survivor_candidates(const SymbolicRuleSet *rules,
                                         nn_real *candidate_inputs,
                                         uint32_t p,
                                         double log_scale,
                                         const FeatureMask *mask,
                                         size_t *survivor_actions);
size_t fill_expanded_survivor_candidates(const ExpandedRuleSet *rules,
                                         nn_real *candidate_inputs,
                                         uint32_t p,
                                         double log_scale,
                                         const FeatureMask *mask,
                                         size_t *survivor_actions);
void print_comparison_header(void);
void print_comparison_row(const char *label, double temperature, EvalMetrics metrics);
void print_metrics(const char *label, EvalMetrics metrics);
int save_prime_checkpoint_with_meta(NeuralNet *net,
                                    const char *path,
                                    const char *notes,
                                    const char *parent_hash,
                                    const char *rule_hash,
                                    char out_hash[65]);
int save_prime_checkpoint(NeuralNet *net, const char *path, char out_hash[65]);
int emit_prime_gap_trace(NeuralNet *net,
                         const Sieve *sieve,
                         const GapDataset *dataset,
                         size_t max_states,
                         double log_scale,
                         const FeatureMask *mask,
                         const char *path,
                         const char *model_hash,
                         const char *rule_hash,
                         uint64_t seed,
                         nn_real *candidate_inputs,
                         nn_real *scores,
                         nn_real *probabilities);

uint32_t gcd_u32(uint32_t a, uint32_t b);
int compare_symbolic_rules_canonical(const void *left, const void *right);
void free_symbolic_rule_set(SymbolicRuleSet *rules);
int save_symbolic_rule_checkpoint(const SymbolicRuleSet *rules,
                                  const char *path,
                                  const SymbolicWheelMetrics *metrics,
                                  char *hash_out);
int load_symbolic_rules_with_hash(const char *path,
                                  SymbolicRuleSet *rules,
                                  char *hash_out);
void free_expanded_rule_set(ExpandedRuleSet *rules);
int load_expanded_rules_with_hash(const char *path,
                                  ExpandedRuleSet *rules,
                                  RuleSearchConfig *config,
                                  RuleSearchMetrics *metrics,
                                  char *hash_out);
int expanded_rule_rejects(const ExpandedRule *rule,
                          uint32_t p,
                          uint32_t gap,
                          VerifierWork *work);
VerifierWork expanded_rule_scan_work(const Sieve *sieve,
                                      const ExpandedRuleSet *rules,
                                      uint32_t p,
                                      uint32_t true_gap,
                                      int *false_rejection);
RuleSearchMetrics evaluate_expanded_rules(const Sieve *sieve,
                                          const ExpandedRuleSet *rules,
                                          const GapDataset *dataset,
                                          size_t max_states,
                                          RuleSearchMetrics metrics);
int discover_symbolic_wheel(const GapDataset *train,
                            const GapDataset *validation,
                            SymbolicRuleSet *accepted,
                            SymbolicWheelMetrics *metrics);
int symbolic_rule_rejects(const SymbolicRejectRule *rule, uint32_t candidate);
VerifierWork learned_symbolic_scan_work(const Sieve *sieve,
                                        const SymbolicRuleSet *rules,
                                        uint32_t p,
                                        uint32_t true_gap,
                                        int *false_rejection);
SymbolicWheelMetrics evaluate_symbolic_wheel(const Sieve *sieve,
                                             const SymbolicRuleSet *rules,
                                             const GapDataset *dataset,
                                             size_t max_samples,
                                             SymbolicWheelMetrics metrics);
void print_symbolic_wheel_report(const SymbolicRuleSet *rules,
                                 SymbolicWheelMetrics metrics);

int train_supervised(NeuralNet *net,
                     const char *label,
                     const GapDataset *dataset,
                     size_t max_states,
                     size_t epochs,
                     nn_real learning_rate,
                     double log_scale,
                     const FeatureMask *mask,
                     nn_real *candidate_inputs,
                     size_t *order,
                     uint32_t *rng);
int train_pairwise_ranker(NeuralNet *net,
                          const char *label,
                          const GapDataset *dataset,
                          size_t max_states,
                          size_t epochs,
                          nn_real learning_rate,
                          double log_scale,
                          const FeatureMask *mask,
                          nn_real *candidate_inputs,
                          size_t *order,
                          uint32_t *rng);
int train_supervised_symbolic_masked(NeuralNet *net,
                                     const char *label,
                                     const SymbolicRuleSet *rules,
                                     const GapDataset *dataset,
                                     size_t max_states,
                                     size_t epochs,
                                     nn_real learning_rate,
                                     double log_scale,
                                     const FeatureMask *mask,
                                     nn_real *candidate_inputs,
                                     size_t *order,
                                     uint32_t *rng);
int train_strict_verifier_rl(NeuralNet *net,
                             const char *label,
                             const GapDataset *dataset,
                             size_t max_states,
                             size_t epochs,
                             nn_real learning_rate,
                             double log_scale,
                             const FeatureMask *mask,
                             nn_real *candidate_inputs,
                             nn_real *scores,
                             nn_real *probabilities,
                             size_t *order,
                             uint32_t *rng);
int train_regularized_verifier_rl(NeuralNet *net,
                                  NeuralNet *reference_net,
                                  const char *label,
                                  const GapDataset *dataset,
                                  size_t max_states,
                                  size_t epochs,
                                  nn_real learning_rate,
                                  nn_real reference_weight,
                                  nn_real entropy_weight,
                                  double log_scale,
                                  const FeatureMask *mask,
                                  nn_real *candidate_inputs,
                                  nn_real *scores,
                                  nn_real *probabilities,
                                  nn_real *reference_scores,
                                  nn_real *reference_probabilities,
                                  size_t *order,
                                  uint32_t *rng);
int run_supervised_ablation(const char *label,
                            const FeatureMask *mask,
                            const NNConfig *config,
                            const Sieve *sieve,
                            const GapDataset *train,
                            const GapDataset *test,
                            size_t max_train_states,
                            size_t max_test_states,
                            size_t supervised_epochs,
                            double log_scale,
                            nn_real *candidate_inputs,
                            nn_real *scores,
                            nn_real *probabilities,
                            size_t *order,
                            uint32_t *rng);
int run_linear_scorer_baseline(const NNConfig *config,
                               const FeatureMask *mask,
                               const Sieve *sieve,
                               const GapDataset *train,
                               const GapDataset *validation,
                               const GapDataset *test,
                               size_t max_train_states,
                               size_t max_validation_states,
                               size_t max_test_states,
                               size_t supervised_epochs,
                               double log_scale,
                               nn_real *candidate_inputs,
                               nn_real *scores,
                               nn_real *probabilities,
                               size_t *order,
                               uint32_t *rng,
                               EvalMetrics *metrics_out,
                               EvalMetrics *temp_metrics_out,
                               double *temperature_out);

#endif
