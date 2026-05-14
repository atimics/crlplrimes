/* Trace replay and checkpoint comparison CLI entry points. */

#include "internal.h"


typedef struct {
    uint32_t p;
    uint32_t target_action_id;
    double log_scale;
} ReplaySample;

typedef struct {
    ReplaySample *items;
    size_t count;
    size_t capacity;
    char parent_hash[65];
    char rule_hash[65];
} ReplayTrace;

static int append_replay_sample(ReplayTrace *trace, ReplaySample sample)
{
    if (trace->count == trace->capacity) {
        size_t next_capacity = trace->capacity == 0 ? 1024 : trace->capacity * 2;
        ReplaySample *next = calloc(next_capacity, sizeof(*next));
        if (next == NULL) {
            return 0;
        }
        if (trace->count != 0) {
            memcpy(next, trace->items, trace->count * sizeof(*next));
        }
        free(trace->items);
        trace->items = next;
        trace->capacity = next_capacity;
    }

    trace->items[trace->count++] = sample;
    return 1;
}

static void free_replay_trace(ReplayTrace *trace)
{
    free(trace->items);
    memset(trace, 0, sizeof(*trace));
}

static int load_replay_trace(const char *path, ReplayTrace *trace)
{
    memset(trace, 0, sizeof(*trace));
    strcpy(trace->parent_hash, "unknown");
    strcpy(trace->rule_hash, "unknown");

    FILE *fp = fopen(path, "r");
    if (fp == NULL) {
        return 0;
    }

    char line[16384];
    uint32_t action_ids[ACTION_COUNT];
    nn_real scores[ACTION_COUNT];
    nn_real probabilities[ACTION_COUNT];
    int ok = 1;
    while (ok && fgets(line, sizeof(line), fp) != NULL) {
        NNParsedTraceRecord record;
        uint64_t p = 0;
        nn_real log_scale = (nn_real)log((double)DEFAULT_TEST_LIMIT + (double)G_MAX);

        if (!nn_trace_parse_jsonl(line,
                                  &record,
                                  action_ids,
                                  scores,
                                  probabilities,
                                  ACTION_COUNT)) {
            ok = 0;
            break;
        }
        if (strcmp(record.task_name, "prime_gap") != 0) {
            continue;
        }
        if (record.candidate_count != ACTION_COUNT ||
            !nn_trace_json_get_u64(line, "p", &p)) {
            ok = 0;
            break;
        }
        nn_trace_json_get_real(line, "log_scale", &log_scale);
        if (strcmp(trace->parent_hash, "unknown") == 0 &&
            strcmp(record.model_hash_hex, "unknown") != 0) {
            strcpy(trace->parent_hash, record.model_hash_hex);
        }
        if (strcmp(trace->rule_hash, "unknown") == 0) {
            char parsed_rule_hash[65];
            if (nn_trace_json_get_string(line,
                                         "rule_set_hash",
                                         parsed_rule_hash,
                                         sizeof(parsed_rule_hash)) &&
                parsed_rule_hash[0] != '\0') {
                strcpy(trace->rule_hash, parsed_rule_hash);
            }
        }
        uint8_t seen_actions[ACTION_COUNT] = {0};
        for (size_t row = 0; row < record.candidate_count; ++row) {
            size_t action = ACTION_COUNT;
            if (!prime_action_from_action_id(action_ids[row], &action) ||
                seen_actions[action]) {
                ok = 0;
                break;
            }
            seen_actions[action] = 1;
        }
        if (!ok || p > UINT32_MAX || record.target_index >= ACTION_COUNT) {
            ok = 0;
            break;
        }

        ReplaySample sample;
        sample.p = (uint32_t)p;
        sample.target_action_id = action_ids[record.target_index];
        sample.log_scale = (double)log_scale;
        ok = append_replay_sample(trace, sample);
    }

    if (fclose(fp) != 0) {
        ok = 0;
    }
    if (!ok || trace->count == 0) {
        free_replay_trace(trace);
        return 0;
    }
    return 1;
}

static void print_replay_usage(FILE *stream)
{
    fputs("usage: prime_replay trace.jsonl --out model.nnckpt [epochs] [--rules rules.crlprule]\n",
          stream);
}

int nn_prime_replay_main(int argc, char **argv)
{
    if (argc < 4) {
        print_replay_usage(stderr);
        return EXIT_FAILURE;
    }

    const char *trace_path = argv[1];
    const char *out_path = NULL;
    const char *rules_path = NULL;
    size_t epochs = 3;

    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--out") == 0) {
            if (++i >= argc || argv[i][0] == '\0') {
                print_replay_usage(stderr);
                return EXIT_FAILURE;
            }
            out_path = argv[i];
        } else if (strcmp(argv[i], "--rules") == 0) {
            if (++i >= argc || argv[i][0] == '\0') {
                print_replay_usage(stderr);
                return EXIT_FAILURE;
            }
            rules_path = argv[i];
        } else {
            char *end = NULL;
            unsigned long value = strtoul(argv[i], &end, 10);
            if (end == argv[i] || *end != '\0' || value == 0) {
                print_replay_usage(stderr);
                return EXIT_FAILURE;
            }
            epochs = (size_t)value;
        }
    }
    if (out_path == NULL) {
        print_replay_usage(stderr);
        return EXIT_FAILURE;
    }

    SymbolicRuleSet replay_rules = {0};
    char rule_hash[65] = "unknown";
    int use_rules = 0;
    if (rules_path != NULL) {
        if (!load_symbolic_rules_with_hash(rules_path, &replay_rules, rule_hash)) {
            fprintf(stderr, "failed to load symbolic rules: %s\n", rules_path);
            print_replay_usage(stderr);
            return EXIT_FAILURE;
        }
        use_rules = 1;
        printf("loaded replay symbolic rules: %s hash=%s rules=%zu\n",
               rules_path,
               rule_hash,
               replay_rules.count);
    }

    ReplayTrace trace;
    if (!load_replay_trace(trace_path, &trace)) {
        fprintf(stderr, "failed to load replay trace: %s\n", trace_path);
        free_symbolic_rule_set(&replay_rules);
        return EXIT_FAILURE;
    }
    if (!use_rules && strcmp(trace.rule_hash, "unknown") != 0) {
        strcpy(rule_hash, trace.rule_hash);
    }

    const size_t layers[] = {FEATURE_COUNT, NN_PRIME_GAP_HIDDEN1, NN_PRIME_GAP_HIDDEN2, 1};
    NNConfig config;
    config.layer_sizes = layers;
    config.layer_count = sizeof(layers) / sizeof(layers[0]);
    config.seed = 2026;
    config.hidden_activation = NN_ACTIVATION_RELU;
    config.output_activation = NN_ACTIVATION_LINEAR;
    config.loss = NN_LOSS_MSE;

    NeuralNet *net = nn_create_config(&config);
    nn_real *candidate_inputs = calloc((size_t)ACTION_COUNT * FEATURE_COUNT,
                                       sizeof(*candidate_inputs));
    if (net == NULL || candidate_inputs == NULL) {
        fputs("failed to allocate replay model\n", stderr);
        nn_free(net);
        free(candidate_inputs);
        free_replay_trace(&trace);
        free_symbolic_rule_set(&replay_rules);
        return EXIT_FAILURE;
    }

    size_t survivor_actions[ACTION_COUNT];
    uint32_t current_action_ids[ACTION_COUNT];
    fill_action_ids(current_action_ids);
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        double loss = 0.0;
        for (size_t i = 0; i < trace.count; ++i) {
            ReplaySample sample = trace.items[i];
            size_t candidate_count = ACTION_COUNT;
            size_t target_index = action_row_for_action_id(current_action_ids,
                                                           ACTION_COUNT,
                                                           sample.target_action_id);
            if (use_rules) {
                size_t target_action = ACTION_COUNT;
                candidate_count = fill_symbolic_survivor_candidates(&replay_rules,
                                                                    candidate_inputs,
                                                                    sample.p,
                                                                    sample.log_scale,
                                                                    &FEATURE_MASK_FULL,
                                                                    survivor_actions);
                if (!prime_action_from_action_id(sample.target_action_id, &target_action)) {
                    fputs("replay trace target action_id is unknown\n", stderr);
                    nn_free(net);
                    free(candidate_inputs);
                    free_replay_trace(&trace);
                    free_symbolic_rule_set(&replay_rules);
                    return EXIT_FAILURE;
                }
                target_index = survivor_index_for_action(survivor_actions,
                                                         candidate_count,
                                                         target_action);
                if (candidate_count == 0 || target_index == candidate_count) {
                    fputs("replay symbolic rules rejected the target action\n", stderr);
                    nn_free(net);
                    free(candidate_inputs);
                    free_replay_trace(&trace);
                    free_symbolic_rule_set(&replay_rules);
                    return EXIT_FAILURE;
                }
            } else {
                fill_candidate_features_masked(candidate_inputs,
                                               sample.p,
                                               sample.log_scale,
                                               &FEATURE_MASK_FULL);
                if (target_index == ACTION_COUNT) {
                    fputs("replay trace target action_id is not in current candidates\n", stderr);
                    nn_free(net);
                    free(candidate_inputs);
                    free_replay_trace(&trace);
                    free_symbolic_rule_set(&replay_rules);
                    return EXIT_FAILURE;
                }
            }
            nn_real step_loss = nn_train_action_scorer(net,
                                                       candidate_inputs,
                                                       candidate_count,
                                                       target_index,
                                                       (nn_real)0.005);
            if (isnan((double)step_loss)) {
                fputs("replay training failed\n", stderr);
                nn_free(net);
                free(candidate_inputs);
                free_replay_trace(&trace);
                free_symbolic_rule_set(&replay_rules);
                return EXIT_FAILURE;
            }
            loss += (double)step_loss;
        }
        printf("replay epoch %zu loss=%.4f rows=%zu\n",
               epoch + 1,
               loss / (double)trace.count,
               trace.count);
    }

    NNCheckpointMeta meta;
    memset(&meta, 0, sizeof(meta));
    meta.checkpoint_version = 1;
    meta.feature_encoder_version = 1;
    strcpy(meta.parent_model_hash, trace.parent_hash);
    snprintf(meta.training_trace_root,
             sizeof(meta.training_trace_root),
             "%s",
             strcmp(rule_hash, "unknown") == 0 ? "jsonl" : rule_hash);
    snprintf(meta.notes,
             sizeof(meta.notes),
             "%s",
             use_rules ? "prime_gap successor trained from learned-wheel trace replay"
                       : "prime_gap successor trained from trace replay");

    char hash[65];
    int ok = nn_save_checkpoint(net, out_path, &meta) &&
             nn_hash_checkpoint_hex(out_path, hash);
    if (ok) {
        printf("saved replay checkpoint: %s hash=%s parent=%s\n",
               out_path,
               hash,
               trace.parent_hash);
    }

    nn_free(net);
    free(candidate_inputs);
    free_replay_trace(&trace);
    free_symbolic_rule_set(&replay_rules);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

static void print_compare_usage(FILE *stream)
{
    fputs("usage: prime_compare [--rules rules.crlprule] "
          "model_a.nnckpt model_b.nnckpt [model_c.nnckpt ...]\n",
          stream);
}

static void print_eval_reliability_bins_json(const EvalMetrics *metrics)
{
    fputs(",\"reliability_bins\":[", stdout);
    int first = 1;
    for (size_t bin = 0; bin < CALIBRATION_BINS; ++bin) {
        if (metrics->calibration_count[bin] == 0) {
            continue;
        }
        printf("%s{\"low\":%.2f,\"high\":%.2f,\"count\":%zu,"
               "\"confidence\":%.8f,\"accuracy\":%.8f}",
               first ? "" : ",",
               (double)bin / (double)CALIBRATION_BINS,
               (double)(bin + 1) / (double)CALIBRATION_BINS,
               metrics->calibration_count[bin],
               metrics->calibration_confidence[bin],
               metrics->calibration_correct[bin]);
        first = 0;
    }
    fputc(']', stdout);
}

static void print_report_json(const char *model_hash,
                              const NNCheckpointMeta *meta,
                              uint32_t seed,
                              const char *evaluation_mode,
                              const char *rule_hash,
                              EvalMetrics metrics)
{
    double samples = metrics.samples == 0 ? 1.0 : (double)metrics.samples;
    printf("{\"schema\":\"crlp.compare.v1\",\"report_version\":1,\"task\":\"prime_gap\","
           "\"model_hash\":\"%s\",\"parent_hash\":\"%s\","
           "\"evaluation_mode\":\"%s\",\"rule_set_hash\":\"%s\","
           "\"feature_encoder_version\":%u,"
           "\"benchmark_seed\":%u,\"sample_count\":%zu,"
           "\"top1\":%.8f,\"top5\":%.8f,\"top10\":%.8f,"
           "\"nll\":%.8f,\"mrr\":%.8f,\"ece\":%.8f,"
           "\"mean_rank\":%.8f,\"proposal_steps\":%.8f,"
           "\"wheel_exact\":%.8f,\"assisted_wheel\":%.8f,"
           "\"block_cost\":%.8f,\"assisted_block\":%.8f,"
           "\"model_ms_per_sample\":%.8f,\"verifier_ms_per_sample\":%.8f",
           model_hash,
           meta->parent_model_hash[0] == '\0' ? "unknown" : meta->parent_model_hash,
           evaluation_mode == NULL ? "raw" : evaluation_mode,
           rule_hash == NULL ? "unknown" : rule_hash,
           meta->feature_encoder_version,
           seed,
           metrics.samples,
           (double)metrics.top1 / samples,
           (double)metrics.top5 / samples,
           (double)metrics.top10 / samples,
           metrics.nll,
           metrics.mean_reciprocal_rank,
           metrics.expected_calibration_error,
           metrics.mean_true_rank,
           metrics.proposal_steps_top10,
           metrics.wheel_scan_exact_tests,
           metrics.assisted_wheel_exact_tests_top10,
           metrics.block_scan_cost,
           metrics.assisted_block_cost_top10,
           metrics.samples == 0 ? 0.0
                                : 1000.0 * metrics.scorer_seconds /
                                      (double)metrics.samples,
           metrics.verifier_samples == 0 ? 0.0
                                          : 1000.0 * metrics.verifier_seconds /
                                                (double)metrics.verifier_samples);
    print_eval_reliability_bins_json(&metrics);
    printf("}\n");
}

static void print_symbolic_compare_json(const char *rule_hash,
                                        uint32_t seed,
                                        SymbolicWheelMetrics metrics)
{
    printf("{\"schema\":\"crlp.compare.v1\",\"report_version\":1,\"task\":\"prime_gap\","
           "\"model_hash\":\"none\",\"parent_hash\":\"unknown\","
           "\"evaluation_mode\":\"learned_wheel_rules\","
           "\"rule_set_hash\":\"%s\",\"feature_encoder_version\":1,"
           "\"benchmark_seed\":%u,\"sample_count\":%zu,"
           "\"accepted_rules\":%zu,\"false_rejections\":%zu,"
           "\"direct_exact\":%.8f,\"hand_wheel_exact\":%.8f,"
           "\"learned_wheel_exact\":%.8f,\"learned_wheel_mod\":%.8f}\n",
           rule_hash == NULL ? "unknown" : rule_hash,
           seed,
           metrics.samples,
           metrics.accepted_rules,
           metrics.test_false_rejections,
           metrics.direct_exact_tests,
           metrics.hand_wheel_exact_tests,
           metrics.learned_exact_tests,
           metrics.learned_mod_tests);
}

int nn_prime_compare_main(int argc, char **argv)
{
    const char *rules_path = NULL;
    size_t model_count = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--rules") == 0) {
            if (++i >= argc || argv[i][0] == '\0') {
                print_compare_usage(stderr);
                return EXIT_FAILURE;
            }
            rules_path = argv[i];
        } else if (argv[i][0] == '-') {
            print_compare_usage(stderr);
            return EXIT_FAILURE;
        } else {
            ++model_count;
        }
    }

    if (model_count < 2) {
        print_compare_usage(stderr);
        return EXIT_FAILURE;
    }

    const uint32_t train_limit = DEFAULT_TRAIN_LIMIT;
    const uint32_t test_limit = DEFAULT_TEST_LIMIT;
    const uint32_t sieve_limit = test_limit + (uint32_t)G_MAX;
    const uint32_t benchmark_seed = 20240511u;
    const double log_scale = log((double)test_limit + (double)G_MAX);

    SymbolicRuleSet compare_rules = {0};
    char rule_hash[65] = "unknown";
    int use_rules = 0;
    if (rules_path != NULL) {
        if (!load_symbolic_rules_with_hash(rules_path, &compare_rules, rule_hash)) {
            fprintf(stderr, "failed to load symbolic rules: %s\n", rules_path);
            return EXIT_FAILURE;
        }
        use_rules = 1;
    }

    Sieve sieve;
    GapDataset train;
    GapDataset validation;
    GapDataset test;
    if (!build_sieve(sieve_limit, &sieve) ||
        !build_gap_datasets(&sieve, train_limit, test_limit, &train, &validation, &test)) {
        fputs("failed to build benchmark data\n", stderr);
        free_symbolic_rule_set(&compare_rules);
        free_sieve(&sieve);
        return EXIT_FAILURE;
    }

    nn_real *candidate_inputs = calloc((size_t)ACTION_COUNT * FEATURE_COUNT,
                                       sizeof(*candidate_inputs));
    nn_real *scores = calloc(ACTION_COUNT, sizeof(*scores));
    nn_real *probabilities = calloc(ACTION_COUNT, sizeof(*probabilities));
    if (candidate_inputs == NULL || scores == NULL || probabilities == NULL) {
        fputs("failed to allocate benchmark buffers\n", stderr);
        free(candidate_inputs);
        free(scores);
        free(probabilities);
        free_symbolic_rule_set(&compare_rules);
        free_gap_dataset(&train);
        free_gap_dataset(&validation);
        free_gap_dataset(&test);
        free_sieve(&sieve);
        return EXIT_FAILURE;
    }

    puts("jsonl report:");
    if (use_rules) {
        SymbolicWheelMetrics symbolic_metrics;
        memset(&symbolic_metrics, 0, sizeof(symbolic_metrics));
        symbolic_metrics.accepted_rules = compare_rules.count;
        symbolic_metrics = evaluate_symbolic_wheel(&sieve,
                                                   &compare_rules,
                                                   &test,
                                                   DEFAULT_MAX_TEST_STATES,
                                                   symbolic_metrics);
        print_symbolic_compare_json(rule_hash, benchmark_seed, symbolic_metrics);
        printf("learned wheel rules hash=%s direct_exact=%.2f hand_wheel_exact=%.2f "
               "learned_wheel_exact=%.2f false_rejections=%zu\n",
               rule_hash,
               symbolic_metrics.direct_exact_tests,
               symbolic_metrics.hand_wheel_exact_tests,
               symbolic_metrics.learned_exact_tests,
               symbolic_metrics.test_false_rejections);
    }
    int ok = 1;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--rules") == 0) {
            ++i;
            continue;
        }
        NNCheckpointMeta meta;
        NeuralNet *net = nn_load_checkpoint(argv[i], &meta);
        char hash[65] = "unknown";
        if (net == NULL) {
            fprintf(stderr, "failed to load checkpoint: %s\n", argv[i]);
            ok = 0;
            continue;
        }
        nn_hash_checkpoint_hex(argv[i], hash);
        EvalMetrics metrics = evaluate_model(net,
                                             &sieve,
                                             &test,
                                             DEFAULT_MAX_TEST_STATES,
                                             log_scale,
                                             &FEATURE_MASK_FULL,
                                             1,
                                             1.0,
                                             candidate_inputs,
                                             scores,
                                             probabilities);
        print_report_json(hash, &meta, benchmark_seed, "raw", "unknown", metrics);
        print_comparison_row(argv[i], 1.0, metrics);
        if (use_rules) {
            EvalMetrics learned_metrics =
                evaluate_model_symbolic_masked(net,
                                               &sieve,
                                               &compare_rules,
                                               &test,
                                               DEFAULT_MAX_TEST_STATES,
                                               log_scale,
                                               &FEATURE_MASK_FULL,
                                               1,
                                               1.0,
                                               candidate_inputs,
                                               scores,
                                               probabilities);
            print_report_json(hash,
                              &meta,
                              benchmark_seed,
                              "learned_wheel_model",
                              rule_hash,
                              learned_metrics);
            print_comparison_row("learned-wheel+model", 1.0, learned_metrics);
        }
        nn_free(net);
    }

    free(candidate_inputs);
    free(scores);
    free(probabilities);
    free_symbolic_rule_set(&compare_rules);
    free_gap_dataset(&train);
    free_gap_dataset(&validation);
    free_gap_dataset(&test);
    free_sieve(&sieve);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
