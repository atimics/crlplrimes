#include "prime_gap.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    PRIME_GAP_FEATURE_ENCODER_VERSION = 1,
};

typedef struct {
    FILE *file;
    uint32_t train_limit;
    uint32_t test_limit;
    size_t supervised_epochs;
    size_t rl_epochs;
    size_t max_train_states;
    size_t max_test_states;
    int verifier_diagnostics;
} SummaryWriter;

typedef struct {
    const char *summary_csv_path;
    const char *trace_jsonl_path;
} OutputPaths;

static double ratio(size_t numerator, size_t denominator)
{
    return denominator == 0 ? 0.0 : (double)numerator / (double)denominator;
}

static int parse_output_paths(int argc, char **argv, OutputPaths *paths)
{
    paths->summary_csv_path = NULL;
    paths->trace_jsonl_path = NULL;

    for (int i = 8; i < argc; ++i) {
        if (strcmp(argv[i], "--summary") == 0) {
            if (++i >= argc || paths->summary_csv_path != NULL) {
                return 0;
            }
            paths->summary_csv_path = argv[i];
        } else if (strcmp(argv[i], "--trace") == 0) {
            if (++i >= argc || paths->trace_jsonl_path != NULL) {
                return 0;
            }
            paths->trace_jsonl_path = argv[i];
        } else if (paths->summary_csv_path == NULL) {
            paths->summary_csv_path = argv[i];
        } else if (paths->trace_jsonl_path == NULL) {
            paths->trace_jsonl_path = argv[i];
        } else {
            return 0;
        }
    }

    return 1;
}

static void csv_text(FILE *file, const char *text)
{
    fputc('"', file);
    if (text != NULL) {
        for (const char *cursor = text; *cursor != '\0'; ++cursor) {
            if (*cursor == '"') {
                fputc('"', file);
            }
            fputc(*cursor, file);
        }
    }
    fputc('"', file);
}

static void summary_field_empty(FILE *file)
{
    fputc(',', file);
}

static void summary_field_double(FILE *file, double value)
{
    fprintf(file, ",%.17g", value);
}

static void summary_field_size(FILE *file, size_t value)
{
    fprintf(file, ",%zu", value);
}

static int summary_open(SummaryWriter *writer,
                        const char *path,
                        uint32_t train_limit,
                        uint32_t test_limit,
                        size_t supervised_epochs,
                        size_t rl_epochs,
                        size_t max_train_states,
                        size_t max_test_states,
                        int verifier_diagnostics)
{
    writer->file = NULL;
    writer->train_limit = train_limit;
    writer->test_limit = test_limit;
    writer->supervised_epochs = supervised_epochs;
    writer->rl_epochs = rl_epochs;
    writer->max_train_states = max_train_states;
    writer->max_test_states = max_test_states;
    writer->verifier_diagnostics = verifier_diagnostics;

    if (path == NULL || path[0] == '\0') {
        return 1;
    }

    writer->file = fopen(path, "w");
    if (writer->file == NULL) {
        fprintf(stderr, "failed to open summary CSV: %s\n", path);
        return 0;
    }

    fputs("kind,section,label,train_limit,test_limit,supervised_epochs,rl_epochs,"
          "max_train_states,max_test_states,verifier_diagnostics,window_gap,"
          "candidate_count,samples,temperature,top1,top5,top10,top1_prime,"
          "top5_any_prime,top10_any_prime,nll,p_true,p_any_prime,mean_rank,mrr,"
          "map,recall5,recall10,first_prime_rank,mae_gap,top1_conf,"
          "correct_top1_conf,incorrect_top1_conf,ece,high_conf_coverage,"
          "high_conf_accuracy,avg_primes_in_window,top1_wheel_survivor,"
          "verifier_samples,direct_scan_exact,wheel_scan_exact,wheel_scan_mod,"
          "proposal_steps_top10,assisted_wheel_exact,assisted_wheel_mod,"
          "block_cost,assisted_block_cost\n",
          writer->file);
    return 1;
}

static int summary_close(SummaryWriter *writer)
{
    if (writer->file == NULL) {
        return 1;
    }

    if (fclose(writer->file) != 0) {
        writer->file = NULL;
        return 0;
    }

    writer->file = NULL;
    return 1;
}

static void summary_prefix(SummaryWriter *writer,
                           const char *kind,
                           const char *section,
                           const char *label,
                           uint32_t window_gap,
                           size_t candidate_count,
                           size_t samples)
{
    FILE *file = writer->file;

    csv_text(file, kind);
    fputc(',', file);
    csv_text(file, section);
    fputc(',', file);
    csv_text(file, label);
    fprintf(file,
            ",%u,%u,%zu,%zu,%zu,%zu,%d,%u,%zu,%zu",
            writer->train_limit,
            writer->test_limit,
            writer->supervised_epochs,
            writer->rl_epochs,
            writer->max_train_states,
            writer->max_test_states,
            writer->verifier_diagnostics,
            window_gap,
            candidate_count,
            samples);
}

static void summary_write_eval(SummaryWriter *writer,
                               const char *section,
                               const char *label,
                               EvalMetrics metrics)
{
    FILE *file = writer->file;
    if (file == NULL) {
        return;
    }

    summary_prefix(writer, "eval", section, label, G_MAX, ACTION_COUNT, metrics.samples);
    summary_field_double(file, metrics.temperature);
    summary_field_double(file, ratio(metrics.top1, metrics.samples));
    summary_field_double(file, ratio(metrics.top5, metrics.samples));
    summary_field_double(file, ratio(metrics.top10, metrics.samples));
    summary_field_double(file, ratio(metrics.top1_prime, metrics.samples));
    summary_field_empty(file);
    summary_field_empty(file);
    summary_field_double(file, metrics.nll);
    summary_field_double(file, metrics.true_action_probability);
    summary_field_empty(file);
    summary_field_double(file, metrics.mean_true_rank);
    summary_field_double(file, metrics.mean_reciprocal_rank);
    summary_field_empty(file);
    summary_field_empty(file);
    summary_field_empty(file);
    summary_field_empty(file);
    summary_field_double(file, metrics.mean_abs_gap_error);
    summary_field_double(file, metrics.top1_confidence);
    summary_field_double(file, metrics.correct_top1_confidence);
    summary_field_double(file, metrics.incorrect_top1_confidence);
    summary_field_double(file, metrics.expected_calibration_error);
    summary_field_double(file, ratio(metrics.high_confidence, metrics.samples));
    summary_field_double(file, ratio(metrics.high_confidence_correct, metrics.high_confidence));
    summary_field_empty(file);
    summary_field_empty(file);
    summary_field_size(file, metrics.verifier_samples);
    summary_field_double(file, metrics.direct_scan_exact_tests);
    summary_field_double(file, metrics.wheel_scan_exact_tests);
    summary_field_double(file, metrics.wheel_scan_mod_tests);
    summary_field_double(file, metrics.proposal_steps_top10);
    summary_field_double(file, metrics.assisted_wheel_exact_tests_top10);
    summary_field_double(file, metrics.assisted_wheel_mod_tests_top10);
    summary_field_double(file, metrics.block_scan_cost);
    summary_field_double(file, metrics.assisted_block_cost_top10);
    fputc('\n', file);
}

static void summary_write_interval(SummaryWriter *writer,
                                   const char *section,
                                   const char *label,
                                   uint32_t window_gap,
                                   size_t candidate_count,
                                   IntervalMetrics metrics)
{
    FILE *file = writer->file;
    if (file == NULL) {
        return;
    }

    summary_prefix(writer, "interval", section, label, window_gap, candidate_count, metrics.samples);
    summary_field_empty(file);
    summary_field_empty(file);
    summary_field_empty(file);
    summary_field_empty(file);
    summary_field_double(file, ratio(metrics.top1_prime, metrics.samples));
    summary_field_double(file, ratio(metrics.top5_any_prime, metrics.samples));
    summary_field_double(file, ratio(metrics.top10_any_prime, metrics.samples));
    summary_field_double(file, metrics.nll);
    summary_field_empty(file);
    summary_field_double(file, metrics.positive_probability);
    summary_field_empty(file);
    summary_field_empty(file);
    summary_field_double(file, metrics.mean_average_precision);
    summary_field_double(file, metrics.recall_at_5);
    summary_field_double(file, metrics.recall_at_10);
    summary_field_double(file, metrics.first_prime_rank);
    summary_field_empty(file);
    summary_field_double(file, metrics.top1_confidence);
    summary_field_empty(file);
    summary_field_empty(file);
    summary_field_empty(file);
    summary_field_empty(file);
    summary_field_empty(file);
    summary_field_double(file, ratio(metrics.positive_total, metrics.samples));
    summary_field_double(file, ratio(metrics.top1_wheel_survivor, metrics.samples));
    summary_field_empty(file);
    summary_field_empty(file);
    summary_field_empty(file);
    summary_field_empty(file);
    summary_field_empty(file);
    summary_field_empty(file);
    summary_field_empty(file);
    summary_field_empty(file);
    summary_field_empty(file);
    fputc('\n', file);
}

static uint64_t hash_prime_gap_state(uint32_t p)
{
    uint64_t hash = UINT64_C(1469598103934665603);
    const char prefix[] = "prime_gap";

    for (size_t i = 0; i + 1 < sizeof(prefix); ++i) {
        hash ^= (uint8_t)prefix[i];
        hash *= UINT64_C(1099511628211);
    }
    for (size_t i = 0; i < sizeof(p); ++i) {
        hash ^= (uint8_t)((p >> (8u * i)) & 0xffu);
        hash *= UINT64_C(1099511628211);
    }

    return hash;
}

static void hex_encode(const uint8_t bytes[32], char out_hex[65])
{
    static const char alphabet[] = "0123456789abcdef";
    for (size_t i = 0; i < 32; ++i) {
        out_hex[i * 2] = alphabet[bytes[i] >> 4];
        out_hex[i * 2 + 1] = alphabet[bytes[i] & 0x0f];
    }
    out_hex[64] = '\0';
}

static char *companion_checkpoint_path(const char *trace_path)
{
    const char suffix[] = ".nnckpt";
    size_t trace_len = strlen(trace_path);
    size_t suffix_len = sizeof(suffix);
    if (trace_len > SIZE_MAX - suffix_len) {
        return NULL;
    }

    char *path = malloc(trace_len + suffix_len);
    if (path == NULL) {
        return NULL;
    }

    memcpy(path, trace_path, trace_len);
    memcpy(path + trace_len, suffix, suffix_len);
    return path;
}

static void fill_prime_gap_action_ids(uint32_t action_ids[ACTION_COUNT])
{
    for (size_t action = 0; action < ACTION_COUNT; ++action) {
        action_ids[action] =
            action_is_overflow(action) ? UINT32_MAX : (uint32_t)action_gap(action);
    }
}

static int emit_prime_gap_traces(FILE *file,
                                 NeuralNet *net,
                                 const Sieve *sieve,
                                 const GapDataset *dataset,
                                 size_t max_states,
                                 double log_scale,
                                 const char *model_hash_hex,
                                 uint64_t seed,
                                 nn_real *candidate_inputs,
                                 nn_real *scores,
                                 nn_real *probabilities)
{
    uint32_t action_ids[ACTION_COUNT];
    size_t ranked[ACTION_COUNT];
    fill_prime_gap_action_ids(action_ids);

    size_t samples = dataset->count;
    if (max_states != 0 && max_states < samples) {
        samples = max_states;
    }

    for (size_t sample = 0; sample < samples; ++sample) {
        size_t index = eval_sample_index(dataset, sample, samples);
        uint32_t p = dataset->prime[index];
        uint32_t gap = dataset->gap[index];
        size_t target = gap_to_action(gap);

        fill_candidate_features_masked(candidate_inputs, p, log_scale, &FEATURE_MASK_FULL);
        if (!nn_score_action_rows(net, candidate_inputs, ACTION_COUNT, scores, probabilities)) {
            return 0;
        }
        rank_actions(scores, ranked);

        VerifierWork direct = direct_scan_work(gap);
        VerifierWork wheel = wheel_scan_work(sieve, p, gap);
        VerifierWork assisted = assisted_wheel_work(sieve, p, gap, ranked, 10);
        BlockWork block = block_scan_work(sieve, p, gap);
        BlockWork assisted_block = assisted_block_work(sieve, p, gap, ranked, 10);
        char verifier_cost[512];
        int cost_len = snprintf(verifier_cost,
                                sizeof(verifier_cost),
                                "{\"p\":%u,\"true_gap\":%u,\"proposal_steps\":%zu,"
                                "\"direct_exact\":%zu,\"wheel_exact\":%zu,"
                                "\"wheel_mod\":%zu,\"assisted_wheel_exact\":%zu,"
                                "\"assisted_wheel_mod\":%zu,\"block_cost\":%.17g,"
                                "\"assisted_block_cost\":%.17g}",
                                p,
                                gap,
                                proposal_steps_with_topk(ranked, 10, target),
                                direct.exact_tests,
                                wheel.exact_tests,
                                wheel.mod_tests,
                                assisted.exact_tests,
                                assisted.mod_tests,
                                block_work_cost(block),
                                block_work_cost(assisted_block));
        if (cost_len < 0 || (size_t)cost_len >= sizeof(verifier_cost)) {
            return 0;
        }

        const char *verifier_result = "corrected";
        if (ranked[0] == target) {
            verifier_result = "accepted";
        } else if (target == OVERFLOW_ACTION) {
            verifier_result = "overflow";
        } else if (action_is_overflow(ranked[0]) ||
                   !is_prime_sieve(sieve, p + (uint32_t)action_gap(ranked[0]))) {
            verifier_result = "rejected";
        }

        NNTraceRecord record;
        record.task_name = "prime_gap";
        record.model_hash_hex = model_hash_hex;
        record.state_hash = hash_prime_gap_state(p);
        record.feature_encoder_version = PRIME_GAP_FEATURE_ENCODER_VERSION;
        record.candidate_count = ACTION_COUNT;
        record.action_ids = action_ids;
        record.scores = scores;
        record.probabilities = probabilities;
        record.selected_index = ranked[0];
        record.target_index = target;
        record.loss = (nn_real)(-log(safe_probability((double)probabilities[target])));
        record.verifier_result = verifier_result;
        record.verifier_cost_json = verifier_cost;
        record.seed = seed;

        if (!nn_trace_write_jsonl(file, &record)) {
            return 0;
        }
    }

    return 1;
}

int main(int argc, char **argv)
{
    size_t train_limit_arg = parse_size_arg(argc, argv, 1, DEFAULT_TRAIN_LIMIT);
    size_t test_limit_arg = parse_size_arg(argc, argv, 2, DEFAULT_TEST_LIMIT);
    size_t supervised_epochs = parse_size_arg(argc, argv, 3, DEFAULT_SUPERVISED_EPOCHS);
    size_t rl_epochs = parse_size_arg(argc, argv, 4, DEFAULT_RL_EPOCHS);
    size_t max_train_states = parse_size_arg(argc, argv, 5, DEFAULT_MAX_TRAIN_STATES);
    size_t max_test_states = parse_size_arg(argc, argv, 6, DEFAULT_MAX_TEST_STATES);
    size_t verifier_diagnostics_arg = parse_size_arg(argc, argv, 7, DEFAULT_VERIFIER_DIAGNOSTICS);
    OutputPaths output_paths;
    PrimeGapProfile profile;
    SummaryWriter summary = {0};
    double total_started = now_seconds();
    Sieve sieve = {0, 0, 0, 0};
    GapDataset train = {0, 0, 0, 0, 0, 0};
    GapDataset fit = {0, 0, 0, 0, 0, 0};
    GapDataset calibration = {0, 0, 0, 0, 0, 0};
    GapDataset test = {0, 0, 0, 0, 0, 0};
    ConditionedPolicy residue_policy = {0, 0};
    IntervalTargetCache interval_eval_cache = {0, 0, 0, 0};
    uint32_t rng = 20240511u;
    EvalMetrics supervised_metrics;
    EvalMetrics supervised_temperature_metrics;
    EvalMetrics entropy_metrics;
    EvalMetrics entropy_temperature_metrics;
    EvalMetrics strict_rl_metrics;
    EvalMetrics strict_rl_temperature_metrics;
    double supervised_temperature = 1.0;
    double entropy_temperature = 1.0;
    double strict_rl_temperature = 1.0;
    int have_strict_rl_metrics = 0;
    int ok = 1;

    prime_gap_profile_reset(&profile);
    prime_gap_profile_set(&profile);

    if (!parse_output_paths(argc, argv, &output_paths) ||
        test_limit_arg <= train_limit_arg || train_limit_arg < 16 ||
        test_limit_arg > (size_t)UINT32_MAX - (size_t)INTERVAL_MAX_GAP ||
        train_limit_arg > (size_t)UINT32_MAX ||
        verifier_diagnostics_arg > 1) {
        fputs("usage: prime_gap [train_limit] [test_limit] [supervised_epochs] [rl_epochs] "
              "[max_train_states] [max_test_states] [verifier_diagnostics=0|1] "
              "[summary_csv_path] [--trace trace_jsonl_path]\n",
              stderr);
        return EXIT_FAILURE;
    }

    uint32_t train_limit = (uint32_t)train_limit_arg;
    uint32_t test_limit = (uint32_t)test_limit_arg;
    uint32_t sieve_limit = test_limit + (uint32_t)INTERVAL_MAX_GAP;
    int include_verifier_diagnostics = verifier_diagnostics_arg != 0;

    printf("prime-gap scorer train_limit=%u test_limit=%u g_max=%d actions=%d features=%d "
           "verifier_diagnostics=%s summary_csv=%s trace_jsonl=%s\n",
           train_limit,
           test_limit,
           G_MAX,
           ACTION_COUNT,
           FEATURE_COUNT,
           include_verifier_diagnostics ? "on" : "off",
           output_paths.summary_csv_path == NULL || output_paths.summary_csv_path[0] == '\0'
               ? "off"
               : output_paths.summary_csv_path,
           output_paths.trace_jsonl_path == NULL || output_paths.trace_jsonl_path[0] == '\0'
               ? "off"
               : output_paths.trace_jsonl_path);

    double started = now_seconds();
    if (!build_sieve(sieve_limit, &sieve)) {
        fputs("failed to build sieve\n", stderr);
        return EXIT_FAILURE;
    }
    printf("sieve limit=%u primes=%zu time=%.3fs\n",
           sieve_limit,
           sieve.prime_count,
           now_seconds() - started);

    if (!build_gap_datasets(&sieve, train_limit, test_limit, &train, &test)) {
        fputs("failed to build gap datasets\n", stderr);
        free_sieve(&sieve);
        return EXIT_FAILURE;
    }

    if (!split_train_calibration(&train, &fit, &calibration)) {
        fputs("failed to split fit/calibration datasets\n", stderr);
        free_gap_dataset(&train);
        free_gap_dataset(&test);
        free_sieve(&sieve);
        return EXIT_FAILURE;
    }

    printf("dataset raw_train=%zu fit=%zu calibration=%zu test=%zu "
           "exact_action_coverage train=%.2f%% test=%.2f%% "
           "overflow_actions train=%zu calibration=%zu test=%zu\n",
           train.count,
           fit.count,
           calibration.count,
           test.count,
           train.total_seen == 0
               ? 0.0
               : 100.0 * (double)(train.total_seen - train.overflow_count) /
                     (double)train.total_seen,
           test.total_seen == 0
               ? 0.0
               : 100.0 * (double)(test.total_seen - test.overflow_count) /
                     (double)test.total_seen,
           train.overflow_count,
           calibration.overflow_count,
           test.overflow_count);

    if (fit.count == 0 || calibration.count == 0 || test.count == 0) {
        fputs("empty fit, calibration, or test split\n", stderr);
        free_gap_dataset(&train);
        free_gap_dataset(&fit);
        free_gap_dataset(&calibration);
        free_gap_dataset(&test);
        free_sieve(&sieve);
        return EXIT_FAILURE;
    }

    const size_t layers[] = {FEATURE_COUNT, 32, 16, 1};
    NNConfig config;
    config.layer_sizes = layers;
    config.layer_count = sizeof(layers) / sizeof(layers[0]);
    config.seed = 42;
    config.hidden_activation = NN_ACTIVATION_RELU;
    config.output_activation = NN_ACTIVATION_LINEAR;
    config.loss = NN_LOSS_MSE;

    const size_t linear_layers[] = {FEATURE_COUNT, 1};
    NNConfig linear_config = config;
    linear_config.layer_sizes = linear_layers;
    linear_config.layer_count = sizeof(linear_layers) / sizeof(linear_layers[0]);

    NeuralNet *net = nn_create_config(&config);
    NeuralNet *entropy_net = NULL;
    NeuralNet *wheel_net = NULL;
    NeuralNet *interval_net = NULL;
    NeuralNet *interval_linear_net = NULL;
    NeuralNet *interval_raw_net = NULL;
    NeuralNet *interval_raw_linear_net = NULL;
    size_t max_candidate_rows = ACTION_COUNT > INTERVAL_MAX_ACTION_COUNT
                                    ? ACTION_COUNT
                                    : INTERVAL_MAX_ACTION_COUNT;
    nn_real *candidate_inputs =
        calloc(max_candidate_rows * FEATURE_COUNT, sizeof(*candidate_inputs));
    nn_real *scores = calloc(max_candidate_rows, sizeof(*scores));
    nn_real *probabilities = calloc(max_candidate_rows, sizeof(*probabilities));
    nn_real *target_distribution = calloc(INTERVAL_MAX_ACTION_COUNT, sizeof(*target_distribution));
    size_t *order = calloc(fit.count, sizeof(*order));
    if (net == NULL || candidate_inputs == NULL || scores == NULL || probabilities == NULL ||
        target_distribution == NULL || order == NULL) {
        fputs("failed to allocate model buffers\n", stderr);
        ok = 0;
        goto cleanup;
    }

    for (size_t i = 0; i < fit.count; ++i) {
        order[i] = i;
    }

    if (!summary_open(&summary,
                      output_paths.summary_csv_path,
                      train_limit,
                      test_limit,
                      supervised_epochs,
                      rl_epochs,
                      max_train_states,
                      max_test_states,
                      include_verifier_diagnostics)) {
        ok = 0;
        goto cleanup;
    }

    double log_scale = log((double)test_limit + (double)G_MAX);
    size_t frequency_ranked[ACTION_COUNT];
    nn_real frequency_probabilities[ACTION_COUNT];

    printf("\n=== Baselines ===\n");
    build_frequency_policy(&fit, frequency_ranked, frequency_probabilities);
    EvalMetrics baseline_empirical_metrics =
        evaluate_static_policy(&sieve,
                               &test,
                               frequency_ranked,
                               frequency_probabilities,
                               max_test_states,
                               include_verifier_diagnostics);
    print_metrics("baseline empirical-gap", baseline_empirical_metrics);
    summary_write_eval(&summary, "baselines", "baseline empirical-gap", baseline_empirical_metrics);

    EvalMetrics baseline_wheel_metrics =
        evaluate_wheel_frequency_policy(&sieve,
                                        &test,
                                        frequency_probabilities,
                                        max_test_states,
                                        0,
                                        probabilities);
    print_metrics("baseline wheel-frequency", baseline_wheel_metrics);
    summary_write_eval(&summary, "baselines", "baseline wheel-frequency", baseline_wheel_metrics);

    if (!build_residue_conditioned_policy(&fit, &residue_policy)) {
        fputs("failed to build residue-conditioned baseline\n", stderr);
        ok = 0;
        goto cleanup;
    }
    EvalMetrics baseline_residue_metrics =
        evaluate_residue_conditioned_policy(&residue_policy,
                                            &sieve,
                                            &test,
                                            max_test_states,
                                            include_verifier_diagnostics,
                                            probabilities);
    print_metrics("baseline residue-conditioned", baseline_residue_metrics);
    summary_write_eval(&summary,
                       "baselines",
                       "baseline residue-conditioned",
                       baseline_residue_metrics);

    EvalMetrics model_before_metrics =
        evaluate_model(net,
                       &sieve,
                       &test,
                       max_test_states,
                       log_scale,
                       &FEATURE_MASK_FULL,
                       1.0,
                       0,
                       include_verifier_diagnostics,
                       candidate_inputs,
                       scores,
                       probabilities);
    print_metrics("model before", model_before_metrics);
    summary_write_eval(&summary, "baselines", "model before", model_before_metrics);

    printf("\n=== Feature Ablations ===\n");
    printf("Supervised-only runs; verifier diagnostics skipped to keep these capability-focused.\n");
    EvalMetrics ablation_metrics;
    ok = run_supervised_ablation("baseline linear-full",
                                 &FEATURE_MASK_FULL,
                                 &linear_config,
                                 &sieve,
                                 &fit,
                                 &test,
                                 max_train_states,
                                 max_test_states,
                                 supervised_epochs,
                                 log_scale,
                                 candidate_inputs,
                                 scores,
                                 probabilities,
                                 order,
                                 &rng,
                                 &ablation_metrics);
    if (!ok) {
        goto cleanup;
    }
    summary_write_eval(&summary, "feature-ablations", "baseline linear-full", ablation_metrics);

    ok = run_supervised_ablation("ablation log-only",
                                 &FEATURE_MASK_LOG_ONLY,
                                 &config,
                                 &sieve,
                                 &fit,
                                 &test,
                                 max_train_states,
                                 max_test_states,
                                 supervised_epochs,
                                 log_scale,
                                 candidate_inputs,
                                 scores,
                                 probabilities,
                                 order,
                                 &rng,
                                 &ablation_metrics);
    if (!ok) {
        goto cleanup;
    }
    summary_write_eval(&summary, "feature-ablations", "ablation log-only", ablation_metrics);

    ok = run_supervised_ablation("ablation gap-only",
                                 &FEATURE_MASK_GAP_ONLY,
                                 &config,
                                 &sieve,
                                 &fit,
                                 &test,
                                 max_train_states,
                                 max_test_states,
                                 supervised_epochs,
                                 log_scale,
                                 candidate_inputs,
                                 scores,
                                 probabilities,
                                 order,
                                 &rng,
                                 &ablation_metrics);
    if (!ok) {
        goto cleanup;
    }
    summary_write_eval(&summary, "feature-ablations", "ablation gap-only", ablation_metrics);

    ok = run_supervised_ablation("ablation residue-only",
                                 &FEATURE_MASK_RESIDUE_ONLY,
                                 &config,
                                 &sieve,
                                 &fit,
                                 &test,
                                 max_train_states,
                                 max_test_states,
                                 supervised_epochs,
                                 log_scale,
                                 candidate_inputs,
                                 scores,
                                 probabilities,
                                 order,
                                 &rng,
                                 &ablation_metrics);
    if (!ok) {
        goto cleanup;
    }
    summary_write_eval(&summary, "feature-ablations", "ablation residue-only", ablation_metrics);

    printf("\n=== Full Model Training ===\n");
    ok = train_supervised(net,
                          "model supervised",
                          &fit,
                          max_train_states,
                          supervised_epochs,
                          (nn_real)0.005,
                          log_scale,
                          &FEATURE_MASK_FULL,
                          (nn_real)0,
                          0,
                          candidate_inputs,
                          order,
                          &rng);
    if (!ok) {
        goto cleanup;
    }

    supervised_metrics = evaluate_model(net,
                                        &sieve,
                                        &test,
                                        max_test_states,
                                        log_scale,
                                        &FEATURE_MASK_FULL,
                                        1.0,
                                        0,
                                        include_verifier_diagnostics,
                                        candidate_inputs,
                                        scores,
                                        probabilities);
    print_metrics("model supervised", supervised_metrics);
    summary_write_eval(&summary, "full-model-training", "model supervised", supervised_metrics);

    if (output_paths.trace_jsonl_path != NULL && output_paths.trace_jsonl_path[0] != '\0') {
        char *checkpoint_path = companion_checkpoint_path(output_paths.trace_jsonl_path);
        char model_hash_hex[65];
        uint8_t model_hash[32];
        NNCheckpointMeta checkpoint_meta;
        memset(&checkpoint_meta, 0, sizeof(checkpoint_meta));
        checkpoint_meta.feature_encoder_version = PRIME_GAP_FEATURE_ENCODER_VERSION;

        if (checkpoint_path == NULL ||
            !nn_save_checkpoint(net, checkpoint_path, &checkpoint_meta) ||
            !nn_hash_checkpoint_file(checkpoint_path, model_hash)) {
            fputs("failed to save/hash trace companion checkpoint\n", stderr);
            free(checkpoint_path);
            ok = 0;
            goto cleanup;
        }
        hex_encode(model_hash, model_hash_hex);

        FILE *trace_file = fopen(output_paths.trace_jsonl_path, "w");
        if (trace_file == NULL) {
            fprintf(stderr, "failed to open trace JSONL: %s\n", output_paths.trace_jsonl_path);
            free(checkpoint_path);
            ok = 0;
            goto cleanup;
        }

        if (!emit_prime_gap_traces(trace_file,
                                   net,
                                   &sieve,
                                   &test,
                                   max_test_states,
                                   log_scale,
                                   model_hash_hex,
                                   rng,
                                   candidate_inputs,
                                   scores,
                                   probabilities)) {
            fputs("failed to emit prime-gap trace JSONL\n", stderr);
            fclose(trace_file);
            free(checkpoint_path);
            ok = 0;
            goto cleanup;
        }
        if (fclose(trace_file) != 0) {
            fputs("failed to close trace JSONL\n", stderr);
            free(checkpoint_path);
            ok = 0;
            goto cleanup;
        }

        printf("trace jsonl=%s model_checkpoint=%s model_hash=%s\n",
               output_paths.trace_jsonl_path,
               checkpoint_path,
               model_hash_hex);
        free(checkpoint_path);
    }

    supervised_temperature = tune_temperature(net,
                                              &sieve,
                                              &calibration,
                                              max_train_states,
                                              log_scale,
                                              &FEATURE_MASK_FULL,
                                              candidate_inputs,
                                              scores,
                                              probabilities);
    supervised_temperature_metrics = evaluate_model(net,
                                                    &sieve,
                                                    &test,
                                                    max_test_states,
                                                    log_scale,
                                                    &FEATURE_MASK_FULL,
                                                    supervised_temperature,
                                                    0,
                                                    0,
                                                    candidate_inputs,
                                                    scores,
                                                    probabilities);
    print_metrics("model supervised temp-scaled", supervised_temperature_metrics);
    summary_write_eval(&summary,
                       "full-model-training",
                       "model supervised temp-scaled",
                       supervised_temperature_metrics);

    printf("\n=== Wheel-Survivor Task ===\n");
    printf("Ranks only candidates that survive the small-prime feature wheel; verifier diagnostics skipped.\n");
    EvalMetrics supervised_wheel_metrics =
        evaluate_model(net,
                       &sieve,
                       &test,
                       max_test_states,
                       log_scale,
                       &FEATURE_MASK_FULL,
                       1.0,
                       1,
                       0,
                       candidate_inputs,
                       scores,
                       probabilities);
    print_metrics("model supervised wheel-survivor eval", supervised_wheel_metrics);
    summary_write_eval(&summary,
                       "wheel-survivor",
                       "model supervised wheel-survivor eval",
                       supervised_wheel_metrics);

    wheel_net = nn_create_config(&config);
    if (wheel_net == NULL) {
        fputs("failed to allocate wheel-survivor model\n", stderr);
        ok = 0;
        goto cleanup;
    }

    ok = train_supervised(wheel_net,
                          "model wheel-survivor-supervised",
                          &fit,
                          max_train_states,
                          supervised_epochs,
                          (nn_real)0.005,
                          log_scale,
                          &FEATURE_MASK_FULL,
                          (nn_real)0,
                          1,
                          candidate_inputs,
                          order,
                          &rng);
    if (!ok) {
        goto cleanup;
    }

    EvalMetrics wheel_supervised_metrics =
        evaluate_model(wheel_net,
                       &sieve,
                       &test,
                       max_test_states,
                       log_scale,
                       &FEATURE_MASK_FULL,
                       1.0,
                       1,
                       0,
                       candidate_inputs,
                       scores,
                       probabilities);
    print_metrics("model wheel-survivor-supervised", wheel_supervised_metrics);
    summary_write_eval(&summary,
                       "wheel-survivor",
                       "model wheel-survivor-supervised",
                       wheel_supervised_metrics);

    printf("\n=== Interval Prime Ranking Task ===\n");
    printf("Target is a distribution over every prime p+g in the 2..%d gap window, not only the next gap.\n",
           G_MAX);
    size_t interval_frequency_ranked[INTERVAL_MAX_ACTION_COUNT];
    nn_real interval_frequency_probabilities[INTERVAL_MAX_ACTION_COUNT];
    if (!build_interval_eval_target_cache(
            &sieve, &test, max_test_states, EXACT_ACTION_COUNT, &interval_eval_cache)) {
        fputs("failed to cache interval eval targets\n", stderr);
        ok = 0;
        goto cleanup;
    }
    build_interval_frequency_policy(&sieve,
                                    &fit,
                                    EXACT_ACTION_COUNT,
                                    interval_frequency_ranked,
                                    interval_frequency_probabilities);
    IntervalMetrics baseline_interval_metrics =
        evaluate_interval_static_policy_cached(&test,
                                               &interval_eval_cache,
                                               interval_frequency_ranked,
                                               interval_frequency_probabilities);
    print_interval_metrics("baseline interval-frequency", baseline_interval_metrics);
    summary_write_interval(&summary,
                           "interval-prime-ranking",
                           "baseline interval-frequency",
                           G_MAX,
                           EXACT_ACTION_COUNT,
                           baseline_interval_metrics);

    IntervalMetrics baseline_interval_wheel_metrics =
        evaluate_interval_wheel_frequency_policy_cached(&test,
                                                        &interval_eval_cache,
                                                        interval_frequency_probabilities,
                                                        probabilities);
    print_interval_metrics("baseline interval-wheel-frequency", baseline_interval_wheel_metrics);
    summary_write_interval(&summary,
                           "interval-prime-ranking",
                           "baseline interval-wheel-frequency",
                           G_MAX,
                           EXACT_ACTION_COUNT,
                           baseline_interval_wheel_metrics);

    interval_linear_net = nn_create_config(&linear_config);
    if (interval_linear_net == NULL) {
        fputs("failed to allocate interval linear model\n", stderr);
        ok = 0;
        goto cleanup;
    }
    ok = train_interval_supervised(interval_linear_net,
                                   "model interval-linear",
                                   &sieve,
                                   &fit,
                                   max_train_states,
                                   EXACT_ACTION_COUNT,
                                   (double)G_MAX,
                                   supervised_epochs,
                                   (nn_real)0.005,
                                   log_scale,
                                   &FEATURE_MASK_FULL,
                                   candidate_inputs,
                                   target_distribution,
                                   order,
                                   &rng);
    if (!ok) {
        goto cleanup;
    }
    IntervalMetrics interval_linear_metrics =
        evaluate_interval_model_cached(interval_linear_net,
                                       &test,
                                       &interval_eval_cache,
                                       (double)G_MAX,
                                       log_scale,
                                       &FEATURE_MASK_FULL,
                                       candidate_inputs,
                                       scores,
                                       probabilities);
    print_interval_metrics("model interval-linear", interval_linear_metrics);
    summary_write_interval(&summary,
                           "interval-prime-ranking",
                           "model interval-linear",
                           G_MAX,
                           EXACT_ACTION_COUNT,
                           interval_linear_metrics);

    interval_net = nn_create_config(&config);
    if (interval_net == NULL) {
        fputs("failed to allocate interval model\n", stderr);
        ok = 0;
        goto cleanup;
    }
    ok = train_interval_supervised(interval_net,
                                   "model interval-mlp",
                                   &sieve,
                                   &fit,
                                   max_train_states,
                                   EXACT_ACTION_COUNT,
                                   (double)G_MAX,
                                   supervised_epochs,
                                   (nn_real)0.005,
                                   log_scale,
                                   &FEATURE_MASK_FULL,
                                   candidate_inputs,
                                   target_distribution,
                                   order,
                                   &rng);
    if (!ok) {
        goto cleanup;
    }
    IntervalMetrics interval_mlp_metrics =
        evaluate_interval_model_cached(interval_net,
                                       &test,
                                       &interval_eval_cache,
                                       (double)G_MAX,
                                       log_scale,
                                       &FEATURE_MASK_FULL,
                                       candidate_inputs,
                                       scores,
                                       probabilities);
    print_interval_metrics("model interval-mlp", interval_mlp_metrics);
    summary_write_interval(&summary,
                           "interval-prime-ranking",
                           "model interval-mlp",
                           G_MAX,
                           EXACT_ACTION_COUNT,
                           interval_mlp_metrics);

    printf("Raw-residue interval variants remove explicit small-prime divisibility flags.\n");
    interval_raw_linear_net = nn_create_config(&linear_config);
    if (interval_raw_linear_net == NULL) {
        fputs("failed to allocate raw-residue interval linear model\n", stderr);
        ok = 0;
        goto cleanup;
    }
    ok = train_interval_supervised(interval_raw_linear_net,
                                   "model interval-raw-linear",
                                   &sieve,
                                   &fit,
                                   max_train_states,
                                   EXACT_ACTION_COUNT,
                                   (double)G_MAX,
                                   supervised_epochs,
                                   (nn_real)0.005,
                                   log_scale,
                                   &FEATURE_MASK_RAW_RESIDUE,
                                   candidate_inputs,
                                   target_distribution,
                                   order,
                                   &rng);
    if (!ok) {
        goto cleanup;
    }
    IntervalMetrics interval_raw_linear_metrics =
        evaluate_interval_model_cached(interval_raw_linear_net,
                                       &test,
                                       &interval_eval_cache,
                                       (double)G_MAX,
                                       log_scale,
                                       &FEATURE_MASK_RAW_RESIDUE,
                                       candidate_inputs,
                                       scores,
                                       probabilities);
    print_interval_metrics("model interval-raw-linear", interval_raw_linear_metrics);
    summary_write_interval(&summary,
                           "interval-prime-ranking",
                           "model interval-raw-linear",
                           G_MAX,
                           EXACT_ACTION_COUNT,
                           interval_raw_linear_metrics);

    interval_raw_net = nn_create_config(&config);
    if (interval_raw_net == NULL) {
        fputs("failed to allocate raw-residue interval model\n", stderr);
        ok = 0;
        goto cleanup;
    }
    ok = train_interval_supervised(interval_raw_net,
                                   "model interval-raw-mlp",
                                   &sieve,
                                   &fit,
                                   max_train_states,
                                   EXACT_ACTION_COUNT,
                                   (double)G_MAX,
                                   supervised_epochs,
                                   (nn_real)0.005,
                                   log_scale,
                                   &FEATURE_MASK_RAW_RESIDUE,
                                   candidate_inputs,
                                   target_distribution,
                                   order,
                                   &rng);
    if (!ok) {
        goto cleanup;
    }
    IntervalMetrics interval_raw_mlp_metrics =
        evaluate_interval_model_cached(interval_raw_net,
                                       &test,
                                       &interval_eval_cache,
                                       (double)G_MAX,
                                       log_scale,
                                       &FEATURE_MASK_RAW_RESIDUE,
                                       candidate_inputs,
                                       scores,
                                       probabilities);
    print_interval_metrics("model interval-raw-mlp", interval_raw_mlp_metrics);
    summary_write_interval(&summary,
                           "interval-prime-ranking",
                           "model interval-raw-mlp",
                           G_MAX,
                           EXACT_ACTION_COUNT,
                           interval_raw_mlp_metrics);

    nn_free(interval_raw_linear_net);
    interval_raw_linear_net = NULL;
    nn_free(interval_raw_net);
    interval_raw_net = NULL;

    printf("\n=== Wider Raw-Residue Interval Sweep ===\n");
    size_t sweep_train_states =
        max_train_states == 0 || max_train_states > 5000 ? 5000 : max_train_states;
    size_t sweep_test_states =
        max_test_states == 0 || max_test_states > 5000 ? 5000 : max_test_states;
    size_t sweep_epochs = supervised_epochs == 0 ? 0 : 1;
    printf("Each window trains fresh raw-residue linear and MLP scorers without explicit divisibility flags.\n");
    printf("Wide-window runs are capped at train_states=%zu test_states=%zu epochs=%zu for CLI runtime.\n",
           sweep_train_states,
           sweep_test_states,
           sweep_epochs);
    size_t interval_sweep_count = sizeof(INTERVAL_SWEEP_GAPS) / sizeof(INTERVAL_SWEEP_GAPS[0]);
    for (size_t sweep = 0; sweep < interval_sweep_count; ++sweep) {
        uint32_t window_gap = INTERVAL_SWEEP_GAPS[sweep];
        size_t candidate_count = window_gap / 2u;
        char label[96];

        printf("\n-- interval window 2..%u --\n", window_gap);
        if (!build_interval_eval_target_cache(
                &sieve, &test, sweep_test_states, candidate_count, &interval_eval_cache)) {
            fputs("failed to cache wide interval eval targets\n", stderr);
            ok = 0;
            goto cleanup;
        }
        build_interval_frequency_policy(&sieve,
                                        &fit,
                                        candidate_count,
                                        interval_frequency_ranked,
                                        interval_frequency_probabilities);

        snprintf(label, sizeof(label), "baseline interval-frequency-%u", window_gap);
        IntervalMetrics sweep_frequency_metrics =
            evaluate_interval_static_policy_cached(&test,
                                                   &interval_eval_cache,
                                                   interval_frequency_ranked,
                                                   interval_frequency_probabilities);
        print_interval_metrics(label, sweep_frequency_metrics);
        summary_write_interval(&summary,
                               "wide-interval-sweep",
                               label,
                               window_gap,
                               candidate_count,
                               sweep_frequency_metrics);

        snprintf(label, sizeof(label), "baseline interval-wheel-frequency-%u", window_gap);
        IntervalMetrics sweep_wheel_frequency_metrics =
            evaluate_interval_wheel_frequency_policy_cached(&test,
                                                            &interval_eval_cache,
                                                            interval_frequency_probabilities,
                                                            probabilities);
        print_interval_metrics(label, sweep_wheel_frequency_metrics);
        summary_write_interval(&summary,
                               "wide-interval-sweep",
                               label,
                               window_gap,
                               candidate_count,
                               sweep_wheel_frequency_metrics);

        interval_raw_linear_net = nn_create_config(&linear_config);
        if (interval_raw_linear_net == NULL) {
            fputs("failed to allocate sweep raw-residue interval linear model\n", stderr);
            ok = 0;
            goto cleanup;
        }

        snprintf(label, sizeof(label), "model interval-raw-linear-%u", window_gap);
        ok = train_interval_supervised(interval_raw_linear_net,
                                       label,
                                       &sieve,
                                       &fit,
                                       sweep_train_states,
                                       candidate_count,
                                       (double)window_gap,
                                       sweep_epochs,
                                       (nn_real)0.005,
                                       log_scale,
                                       &FEATURE_MASK_RAW_RESIDUE,
                                       candidate_inputs,
                                       target_distribution,
                                       order,
                                       &rng);
        if (!ok) {
            goto cleanup;
        }
        IntervalMetrics sweep_raw_linear_metrics =
            evaluate_interval_model_cached(interval_raw_linear_net,
                                           &test,
                                           &interval_eval_cache,
                                           (double)window_gap,
                                           log_scale,
                                           &FEATURE_MASK_RAW_RESIDUE,
                                           candidate_inputs,
                                           scores,
                                           probabilities);
        print_interval_metrics(label, sweep_raw_linear_metrics);
        summary_write_interval(&summary,
                               "wide-interval-sweep",
                               label,
                               window_gap,
                               candidate_count,
                               sweep_raw_linear_metrics);
        nn_free(interval_raw_linear_net);
        interval_raw_linear_net = NULL;

        interval_raw_net = nn_create_config(&config);
        if (interval_raw_net == NULL) {
            fputs("failed to allocate sweep raw-residue interval model\n", stderr);
            ok = 0;
            goto cleanup;
        }

        snprintf(label, sizeof(label), "model interval-raw-mlp-%u", window_gap);
        ok = train_interval_supervised(interval_raw_net,
                                       label,
                                       &sieve,
                                       &fit,
                                       sweep_train_states,
                                       candidate_count,
                                       (double)window_gap,
                                       sweep_epochs,
                                       (nn_real)0.005,
                                       log_scale,
                                       &FEATURE_MASK_RAW_RESIDUE,
                                       candidate_inputs,
                                       target_distribution,
                                       order,
                                       &rng);
        if (!ok) {
            goto cleanup;
        }
        IntervalMetrics sweep_raw_mlp_metrics =
            evaluate_interval_model_cached(interval_raw_net,
                                           &test,
                                           &interval_eval_cache,
                                           (double)window_gap,
                                           log_scale,
                                           &FEATURE_MASK_RAW_RESIDUE,
                                           candidate_inputs,
                                           scores,
                                           probabilities);
        print_interval_metrics(label, sweep_raw_mlp_metrics);
        summary_write_interval(&summary,
                               "wide-interval-sweep",
                               label,
                               window_gap,
                               candidate_count,
                               sweep_raw_mlp_metrics);
        nn_free(interval_raw_net);
        interval_raw_net = NULL;
    }

    printf("\n=== Entropy-Regularized Training ===\n");
    entropy_net = nn_create_config(&config);
    if (entropy_net == NULL) {
        fputs("failed to allocate entropy model\n", stderr);
        ok = 0;
        goto cleanup;
    }

    ok = train_supervised(entropy_net,
                          "model entropy-supervised",
                          &fit,
                          max_train_states,
                          supervised_epochs,
                          (nn_real)0.005,
                          log_scale,
                          &FEATURE_MASK_FULL,
                          SUPERVISED_ENTROPY_WEIGHT,
                          0,
                          candidate_inputs,
                          order,
                          &rng);
    if (!ok) {
        goto cleanup;
    }

    entropy_metrics = evaluate_model(entropy_net,
                                     &sieve,
                                     &test,
                                     max_test_states,
                                     log_scale,
                                     &FEATURE_MASK_FULL,
                                     1.0,
                                     0,
                                     0,
                                     candidate_inputs,
                                     scores,
                                     probabilities);
    print_metrics("model entropy-supervised", entropy_metrics);
    summary_write_eval(&summary,
                       "entropy-regularized-training",
                       "model entropy-supervised",
                       entropy_metrics);

    entropy_temperature = tune_temperature(entropy_net,
                                           &sieve,
                                           &calibration,
                                           max_train_states,
                                           log_scale,
                                           &FEATURE_MASK_FULL,
                                           candidate_inputs,
                                           scores,
                                           probabilities);
    entropy_temperature_metrics = evaluate_model(entropy_net,
                                                 &sieve,
                                                 &test,
                                                 max_test_states,
                                                 log_scale,
                                                 &FEATURE_MASK_FULL,
                                                 entropy_temperature,
                                                 0,
                                                 0,
                                                 candidate_inputs,
                                                 scores,
                                                 probabilities);
    print_metrics("model entropy-supervised temp-scaled", entropy_temperature_metrics);
    summary_write_eval(&summary,
                       "entropy-regularized-training",
                       "model entropy-supervised temp-scaled",
                       entropy_temperature_metrics);

    printf("\n=== Strict Verifier RL ===\n");
    if (rl_epochs == 0) {
        printf("skipped because rl_epochs=0\n");
    } else {
        ok = train_strict_verifier_rl(net,
                                      "model strict-rl",
                                      &fit,
                                      max_train_states,
                                      rl_epochs,
                                      (nn_real)0.001,
                                      log_scale,
                                      &FEATURE_MASK_FULL,
                                      candidate_inputs,
                                      scores,
                                      probabilities,
                                      order,
                                      &rng);
        if (!ok) {
            goto cleanup;
        }

        strict_rl_metrics = evaluate_model(net,
                                           &sieve,
                                           &test,
                                           max_test_states,
                                           log_scale,
                                           &FEATURE_MASK_FULL,
                                           1.0,
                                           0,
                                           include_verifier_diagnostics,
                                           candidate_inputs,
                                           scores,
                                           probabilities);
        have_strict_rl_metrics = 1;
        print_metrics("model strict-rl", strict_rl_metrics);
        summary_write_eval(&summary, "strict-verifier-rl", "model strict-rl", strict_rl_metrics);

        strict_rl_temperature = tune_temperature(net,
                                                 &sieve,
                                                 &calibration,
                                                 max_train_states,
                                                 log_scale,
                                                 &FEATURE_MASK_FULL,
                                                 candidate_inputs,
                                                 scores,
                                                 probabilities);
        strict_rl_temperature_metrics = evaluate_model(net,
                                                       &sieve,
                                                       &test,
                                                       max_test_states,
                                                       log_scale,
                                                       &FEATURE_MASK_FULL,
                                                       strict_rl_temperature,
                                                       0,
                                                       0,
                                                       candidate_inputs,
                                                       scores,
                                                       probabilities);
        print_metrics("model strict-rl temp-scaled", strict_rl_temperature_metrics);
        summary_write_eval(&summary,
                           "strict-verifier-rl",
                           "model strict-rl temp-scaled",
                           strict_rl_temperature_metrics);
    }

    if (have_strict_rl_metrics) {
        print_calibration_comparison("calibration comparison raw",
                                     supervised_metrics,
                                     entropy_metrics,
                                     strict_rl_metrics);
        print_calibration_comparison("calibration comparison temp-scaled",
                                     supervised_temperature_metrics,
                                     entropy_temperature_metrics,
                                     strict_rl_temperature_metrics);
    }

cleanup:
    prime_gap_profile_print(&profile, now_seconds() - total_started);
    prime_gap_profile_set(NULL);
    if (!summary_close(&summary)) {
        fputs("failed to close summary CSV\n", stderr);
        ok = 0;
    }
    free_interval_target_cache(&interval_eval_cache);
    free(order);
    free(target_distribution);
    free(probabilities);
    free(scores);
    free(candidate_inputs);
    nn_free(interval_raw_linear_net);
    nn_free(interval_raw_net);
    nn_free(interval_linear_net);
    nn_free(interval_net);
    nn_free(wheel_net);
    nn_free(entropy_net);
    nn_free(net);
    free_conditioned_policy(&residue_policy);
    free_gap_dataset(&train);
    free_gap_dataset(&fit);
    free_gap_dataset(&calibration);
    free_gap_dataset(&test);
    free_sieve(&sieve);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
