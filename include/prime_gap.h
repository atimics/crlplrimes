#ifndef PRIME_GAP_H
#define PRIME_GAP_H

#include "nn.h"

#include <stddef.h>
#include <stdint.h>

enum {
    G_MAX = 256,
    EXACT_ACTION_COUNT = G_MAX / 2,
    OVERFLOW_ACTION = EXACT_ACTION_COUNT,
    ACTION_COUNT = EXACT_ACTION_COUNT + 1,
    INTERVAL_MAX_GAP = 2048,
    INTERVAL_MAX_ACTION_COUNT = INTERVAL_MAX_GAP / 2,
    INTERVAL_SWEEP_COUNT = 3,
    FEATURE_MOD_COUNT = 10,
    FEATURE_COUNT = 3 + FEATURE_MOD_COUNT,
    DEFAULT_TRAIN_LIMIT = 1000000,
    DEFAULT_TEST_LIMIT = 3000000,
    DEFAULT_SUPERVISED_EPOCHS = 2,
    DEFAULT_RL_EPOCHS = 1,
    DEFAULT_MAX_TRAIN_STATES = 30000,
    DEFAULT_MAX_TEST_STATES = 30000,
    CALIBRATION_BINS = 10,
    PRIME_BUCKET_COUNT = 4,
    RESIDUE_SIGNATURE_COUNT = 1u << FEATURE_MOD_COUNT,
    CONDITIONED_SIGNATURE_COUNT = RESIDUE_SIGNATURE_COUNT + 1,
};

enum {
    RESIDUE_FEATURE_NONE = 0,
    RESIDUE_FEATURE_DIVISIBILITY = 1,
    RESIDUE_FEATURE_RAW_NORMALIZED = 2,
};

typedef struct {
    const char *name;
    int use_log;
    int use_gap;
    int use_overflow;
    int residue_mode;
} FeatureMask;

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
    size_t candidates;
    size_t mod_tests;
    size_t exact_tests;
} VerifierWork;

typedef struct {
    size_t candidates;
    size_t base_primes;
    size_t marks;
    size_t survivor_checks;
} BlockWork;

typedef struct {
    size_t *counts;
    size_t *totals;
} ConditionedPolicy;

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
    double temperature;
    double mean_abs_gap_error;
    double direct_scan_exact_tests;
    double wheel_scan_mod_tests;
    double wheel_scan_exact_tests;
    double proposal_steps_top10;
    double assisted_wheel_mod_tests_top10;
    double assisted_wheel_exact_tests_top10;
    double block_scan_cost;
    double assisted_block_cost_top10;
} EvalMetrics;

typedef struct {
    size_t samples;
    size_t top1_prime;
    size_t top5_any_prime;
    size_t top10_any_prime;
    size_t top1_wheel_survivor;
    size_t positive_total;
    double positive_probability;
    double nll;
    double mean_average_precision;
    double recall_at_5;
    double recall_at_10;
    double first_prime_rank;
    double top1_confidence;
} IntervalMetrics;

typedef struct {
    size_t sample_count;
    size_t candidate_count;
    nn_real *targets;
    size_t *positive_counts;
} IntervalTargetCache;

extern const uint32_t FEATURE_MODS[FEATURE_MOD_COUNT];
extern const uint32_t INTERVAL_SWEEP_GAPS[INTERVAL_SWEEP_COUNT];
extern const nn_real SUPERVISED_ENTROPY_WEIGHT;
extern const FeatureMask FEATURE_MASK_FULL;
extern const FeatureMask FEATURE_MASK_LOG_ONLY;
extern const FeatureMask FEATURE_MASK_GAP_ONLY;
extern const FeatureMask FEATURE_MASK_RESIDUE_ONLY;
extern const FeatureMask FEATURE_MASK_RAW_RESIDUE;

double now_seconds(void);
uint32_t rng_next(uint32_t *state);
double rng_uniform(uint32_t *state);
size_t rng_bounded(uint32_t *state, size_t bound);
size_t parse_size_arg(int argc, char **argv, int index, size_t fallback);

void free_sieve(Sieve *sieve);
int build_sieve(uint32_t limit, Sieve *sieve);
int is_prime_sieve(const Sieve *sieve, uint32_t n);

void free_gap_dataset(GapDataset *dataset);
int append_gap(GapDataset *dataset, uint32_t prime, uint32_t gap);
int build_gap_datasets(const Sieve *sieve,
                       uint32_t train_limit,
                       uint32_t test_limit,
                       GapDataset *train,
                       GapDataset *test);
int copy_gap_slice(const GapDataset *source, size_t start, size_t count, GapDataset *target);
int split_train_calibration(const GapDataset *source, GapDataset *fit, GapDataset *calibration);

void shuffle_order(size_t *order, size_t count, uint32_t *rng);
size_t action_gap(size_t action);
uint32_t indexed_gap(size_t action);
size_t gap_to_action(uint32_t gap);
int action_is_overflow(size_t action);
double gap_error_for_action(size_t action, uint32_t true_gap);
size_t candidate_gap_signature(uint32_t p, uint32_t gap);
size_t candidate_signature(uint32_t p, size_t action);
int gap_survives_feature_wheel(uint32_t p, uint32_t gap);
int action_survives_feature_wheel(uint32_t p, size_t action);

void fill_gap_features(nn_real *row,
                       uint32_t p,
                       uint32_t gap,
                       int overflow,
                       double gap_scale,
                       double log_scale,
                       const FeatureMask *mask);
void fill_action_features(nn_real *row,
                          uint32_t p,
                          size_t action,
                          double log_scale,
                          const FeatureMask *mask);
void fill_candidate_features_masked(nn_real *inputs,
                                    uint32_t p,
                                    double log_scale,
                                    const FeatureMask *mask);
size_t fill_wheel_survivor_features(nn_real *inputs,
                                    uint32_t p,
                                    size_t true_action,
                                    double log_scale,
                                    const FeatureMask *mask,
                                    size_t *target_index);
void fill_candidate_features(nn_real *inputs, uint32_t p, double log_scale);

void rank_actions(const nn_real *scores, size_t *ranked);
void rank_actions_count(const nn_real *scores, size_t count, size_t *ranked);
void rank_actions_wheel(const nn_real *scores, uint32_t p, size_t *ranked);
void softmax_temperature(const nn_real *scores,
                         nn_real *probabilities,
                         size_t count,
                         double temperature);
void softmax_temperature_wheel(const nn_real *scores,
                               nn_real *probabilities,
                               uint32_t p,
                               double temperature);
size_t true_rank(const size_t *ranked, size_t true_action);
size_t proposal_steps_with_topk(const size_t *ranked, size_t top_k, size_t true_action);

int wheel_probe_gap(const Sieve *sieve, uint32_t p, uint32_t gap, VerifierWork *work);
int wheel_probe_action_cached(const Sieve *sieve,
                              uint32_t p,
                              size_t action,
                              uint8_t *tested,
                              uint8_t *prime_result,
                              VerifierWork *work);
int wheel_scan_to_gap_cached(const Sieve *sieve,
                             uint32_t p,
                             uint32_t limit_gap,
                             uint8_t *tested,
                             uint8_t *prime_result,
                             VerifierWork *work);
VerifierWork direct_scan_work(uint32_t true_gap);
VerifierWork wheel_scan_work(const Sieve *sieve, uint32_t p, uint32_t true_gap);
VerifierWork assisted_wheel_work(const Sieve *sieve,
                                 uint32_t p,
                                 uint32_t true_gap,
                                 const size_t *ranked,
                                 size_t top_k);
void add_block_work(BlockWork *total, BlockWork part);
double block_work_cost(BlockWork work);
uint32_t segmented_first_prime_gap(const Sieve *sieve,
                                   uint32_t p,
                                   uint32_t limit_gap,
                                   BlockWork *work);
BlockWork block_scan_work(const Sieve *sieve, uint32_t p, uint32_t true_gap);
BlockWork assisted_block_work(const Sieve *sieve,
                              uint32_t p,
                              uint32_t true_gap,
                              const size_t *ranked,
                              size_t top_k);

size_t sample_action(const nn_real *probabilities, uint32_t *rng);
double safe_probability(double probability);
size_t eval_sample_index(const GapDataset *dataset, size_t sample, size_t samples);
size_t prime_bucket_index(const GapDataset *dataset, uint32_t p);
void accumulate_eval_sample(EvalMetrics *metrics,
                            const Sieve *sieve,
                            const GapDataset *dataset,
                            uint32_t p,
                            uint32_t gap,
                            const size_t *ranked,
                            const nn_real *probabilities,
                            int include_verifier);
void finalize_eval_metrics(EvalMetrics *metrics);
EvalMetrics evaluate_model(NeuralNet *net,
                           const Sieve *sieve,
                           const GapDataset *dataset,
                           size_t max_states,
                           double log_scale,
                           const FeatureMask *mask,
                           double temperature,
                           int wheel_survivors_only,
                           int include_verifier,
                           nn_real *candidate_inputs,
                           nn_real *scores,
                           nn_real *probabilities);
EvalMetrics evaluate_static_policy(const Sieve *sieve,
                                   const GapDataset *dataset,
                                   const size_t *ranked,
                                   const nn_real *probabilities,
                                   size_t max_states,
                                   int include_verifier);

void fill_interval_candidate_features(nn_real *inputs,
                                      uint32_t p,
                                      size_t candidate_count,
                                      double gap_scale,
                                      double log_scale,
                                      const FeatureMask *mask);
size_t build_interval_prime_targets(const Sieve *sieve,
                                    uint32_t p,
                                    size_t candidate_count,
                                    nn_real *targets);
void free_interval_target_cache(IntervalTargetCache *cache);
int build_interval_eval_target_cache(const Sieve *sieve,
                                     const GapDataset *dataset,
                                     size_t max_states,
                                     size_t candidate_count,
                                     IntervalTargetCache *cache);
void accumulate_interval_sample(IntervalMetrics *metrics,
                                uint32_t p,
                                const size_t *ranked,
                                const nn_real *probabilities,
                                const nn_real *targets,
                                size_t candidate_count,
                                size_t positive_count);
void finalize_interval_metrics(IntervalMetrics *metrics);
IntervalMetrics evaluate_interval_model(NeuralNet *net,
                                        const Sieve *sieve,
                                        const GapDataset *dataset,
                                        size_t max_states,
                                        size_t candidate_count,
                                        double gap_scale,
                                        double log_scale,
                                        const FeatureMask *mask,
                                        nn_real *candidate_inputs,
                                        nn_real *scores,
                                        nn_real *probabilities,
                                        nn_real *targets);
IntervalMetrics evaluate_interval_model_cached(NeuralNet *net,
                                               const GapDataset *dataset,
                                               const IntervalTargetCache *cache,
                                               double gap_scale,
                                               double log_scale,
                                               const FeatureMask *mask,
                                               nn_real *candidate_inputs,
                                               nn_real *scores,
                                               nn_real *probabilities);
double tune_temperature(NeuralNet *net,
                        const Sieve *sieve,
                        const GapDataset *dataset,
                        size_t max_states,
                        double log_scale,
                        const FeatureMask *mask,
                        nn_real *candidate_inputs,
                        nn_real *scores,
                        nn_real *probabilities);

void print_metrics(const char *label, EvalMetrics metrics);
void print_interval_metrics(const char *label, IntervalMetrics metrics);
void print_calibration_comparison(const char *label,
                                  EvalMetrics supervised,
                                  EvalMetrics entropy,
                                  EvalMetrics strict_rl);

void build_frequency_policy(const GapDataset *dataset, size_t *ranked, nn_real *probabilities);
void score_wheel_frequency_policy(const nn_real *base_probabilities,
                                  uint32_t p,
                                  size_t *ranked,
                                  nn_real *probabilities);
EvalMetrics evaluate_wheel_frequency_policy(const Sieve *sieve,
                                            const GapDataset *dataset,
                                            const nn_real *base_probabilities,
                                            size_t max_states,
                                            int include_verifier,
                                            nn_real *probabilities);
void build_interval_frequency_policy(const Sieve *sieve,
                                     const GapDataset *dataset,
                                     size_t candidate_count,
                                     size_t *ranked,
                                     nn_real *probabilities);
void rank_interval_wheel(const nn_real *scores,
                         uint32_t p,
                         size_t candidate_count,
                         size_t *ranked);
void score_interval_wheel_frequency_policy(const nn_real *base_probabilities,
                                           uint32_t p,
                                           size_t candidate_count,
                                           size_t *ranked,
                                           nn_real *probabilities);
IntervalMetrics evaluate_interval_static_policy(const Sieve *sieve,
                                                const GapDataset *dataset,
                                                const size_t *ranked,
                                                const nn_real *probabilities,
                                                size_t candidate_count,
                                                size_t max_states,
                                                nn_real *targets);
IntervalMetrics evaluate_interval_static_policy_cached(const GapDataset *dataset,
                                                       const IntervalTargetCache *cache,
                                                       const size_t *ranked,
                                                       const nn_real *probabilities);
IntervalMetrics evaluate_interval_wheel_frequency_policy(const Sieve *sieve,
                                                         const GapDataset *dataset,
                                                         const nn_real *base_probabilities,
                                                         size_t candidate_count,
                                                         size_t max_states,
                                                         nn_real *probabilities,
                                                         nn_real *targets);
IntervalMetrics evaluate_interval_wheel_frequency_policy_cached(
    const GapDataset *dataset,
    const IntervalTargetCache *cache,
    const nn_real *base_probabilities,
    nn_real *probabilities);

void free_conditioned_policy(ConditionedPolicy *policy);
int build_residue_conditioned_policy(const GapDataset *dataset, ConditionedPolicy *policy);
void score_residue_conditioned_policy(const ConditionedPolicy *policy,
                                      uint32_t p,
                                      size_t *ranked,
                                      nn_real *probabilities);
EvalMetrics evaluate_residue_conditioned_policy(const ConditionedPolicy *policy,
                                                const Sieve *sieve,
                                                const GapDataset *dataset,
                                                size_t max_states,
                                                int include_verifier,
                                                nn_real *probabilities);

int train_supervised(NeuralNet *net,
                     const char *label,
                     const GapDataset *dataset,
                     size_t max_states,
                     size_t epochs,
                     nn_real learning_rate,
                     double log_scale,
                     const FeatureMask *mask,
                     nn_real entropy_weight,
                     int wheel_survivors_only,
                     nn_real *candidate_inputs,
                     size_t *order,
                     uint32_t *rng);
int train_interval_supervised(NeuralNet *net,
                              const char *label,
                              const Sieve *sieve,
                              const GapDataset *dataset,
                              size_t max_states,
                              size_t candidate_count,
                              double gap_scale,
                              size_t epochs,
                              nn_real learning_rate,
                              double log_scale,
                              const FeatureMask *mask,
                              nn_real *candidate_inputs,
                              nn_real *targets,
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

#endif
