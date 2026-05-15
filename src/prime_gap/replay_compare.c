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
    char feature_set[32];
} ReplayTrace;

typedef enum {
    PRIME_RULE_INPUT_NONE = 0,
    PRIME_RULE_INPUT_SYMBOLIC = 1,
    PRIME_RULE_INPUT_EXPANDED = 2,
} PrimeRuleInputKind;

typedef struct {
    PrimeRuleInputKind kind;
    SymbolicRuleSet symbolic;
    ExpandedRuleSet expanded;
    RuleSearchConfig expanded_config;
    RuleSearchMetrics expanded_metrics;
    char hash[65];
} PrimeRuleInput;

static const char *prime_rule_input_label(PrimeRuleInputKind kind)
{
    if (kind == PRIME_RULE_INPUT_SYMBOLIC) {
        return "symbolic";
    }
    if (kind == PRIME_RULE_INPUT_EXPANDED) {
        return "expanded";
    }
    return "none";
}

static void init_prime_rule_input(PrimeRuleInput *input)
{
    memset(input, 0, sizeof(*input));
    input->kind = PRIME_RULE_INPUT_NONE;
    strcpy(input->hash, "unknown");
}

static void free_prime_rule_input(PrimeRuleInput *input)
{
    free_symbolic_rule_set(&input->symbolic);
    free_expanded_rule_set(&input->expanded);
    input->kind = PRIME_RULE_INPUT_NONE;
}

static int load_prime_rule_input(const char *path, PrimeRuleInput *input)
{
    init_prime_rule_input(input);
    if (path == NULL) {
        return 0;
    }

    if (load_symbolic_rules_with_hash(path, &input->symbolic, input->hash)) {
        input->kind = PRIME_RULE_INPUT_SYMBOLIC;
        return 1;
    }
    free_symbolic_rule_set(&input->symbolic);

    if (load_expanded_rules_with_hash(path,
                                      &input->expanded,
                                      &input->expanded_config,
                                      &input->expanded_metrics,
                                      input->hash)) {
        input->kind = PRIME_RULE_INPUT_EXPANDED;
        return 1;
    }
    free_expanded_rule_set(&input->expanded);
    strcpy(input->hash, "unknown");
    return 0;
}

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
        char parsed_feature_set[32];
        if (!nn_trace_json_get_string(line,
                                      "feature_set",
                                      parsed_feature_set,
                                      sizeof(parsed_feature_set)) ||
            parsed_feature_set[0] == '\0') {
            ok = 0;
            break;
        }
        if (trace->feature_set[0] == '\0') {
            strcpy(trace->feature_set, parsed_feature_set);
        } else if (strcmp(trace->feature_set, parsed_feature_set) != 0) {
            ok = 0;
            break;
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
    if (!ok || trace->count == 0 || trace->feature_set[0] == '\0') {
        free_replay_trace(trace);
        return 0;
    }
    return 1;
}

static void print_replay_usage(FILE *stream)
{
    fputs("usage: prime_replay trace.jsonl --out model.nnckpt [epochs] "
          "[--rules symbolic-or-expanded-rules.crlprule]\n",
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

    PrimeRuleInput replay_rules;
    init_prime_rule_input(&replay_rules);
    char rule_hash[65] = "unknown";
    if (rules_path != NULL) {
        if (!load_prime_rule_input(rules_path, &replay_rules)) {
            fprintf(stderr, "failed to load symbolic or expanded rules: %s\n", rules_path);
            print_replay_usage(stderr);
            return EXIT_FAILURE;
        }
        strcpy(rule_hash, replay_rules.hash);
        printf("loaded replay %s rules: %s hash=%s rules=%zu\n",
               prime_rule_input_label(replay_rules.kind),
               rules_path,
               rule_hash,
               replay_rules.kind == PRIME_RULE_INPUT_SYMBOLIC ? replay_rules.symbolic.count
                                                               : replay_rules.expanded.count);
    }

    ReplayTrace trace;
    if (!load_replay_trace(trace_path, &trace)) {
        fprintf(stderr, "failed to load replay trace: %s\n", trace_path);
        free_prime_rule_input(&replay_rules);
        return EXIT_FAILURE;
    }
    if (replay_rules.kind == PRIME_RULE_INPUT_NONE &&
        strcmp(trace.rule_hash, "unknown") != 0) {
        strcpy(rule_hash, trace.rule_hash);
    }
    const FeatureMask *trace_mask = feature_mask_from_name(trace.feature_set);
    if (trace_mask == NULL) {
        fprintf(stderr, "unsupported replay trace feature_set: %s\n", trace.feature_set);
        free_replay_trace(&trace);
        free_prime_rule_input(&replay_rules);
        return EXIT_FAILURE;
    }
    printf("replay feature_set=%s residue_features=%s\n",
           trace_mask->name,
           trace_mask->use_residue ? "on" : "off");

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
        free_prime_rule_input(&replay_rules);
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
            if (replay_rules.kind != PRIME_RULE_INPUT_NONE) {
                size_t target_action = ACTION_COUNT;
                if (!prime_action_from_action_id(sample.target_action_id, &target_action)) {
                    fputs("replay trace target action_id is unknown\n", stderr);
                    nn_free(net);
                    free(candidate_inputs);
                    free_replay_trace(&trace);
                    free_prime_rule_input(&replay_rules);
                    return EXIT_FAILURE;
                }
                if (replay_rules.kind == PRIME_RULE_INPUT_SYMBOLIC) {
                    candidate_count =
                        fill_symbolic_survivor_candidates(&replay_rules.symbolic,
                                                          candidate_inputs,
                                                          sample.p,
                                                          sample.log_scale,
                                                          trace_mask,
                                                          survivor_actions);
                } else {
                    candidate_count =
                        fill_expanded_survivor_candidates(&replay_rules.expanded,
                                                          candidate_inputs,
                                                          sample.p,
                                                          sample.log_scale,
                                                          trace_mask,
                                                          survivor_actions);
                }
                target_index = survivor_index_for_action(survivor_actions,
                                                         candidate_count,
                                                         target_action);
                if (candidate_count == 0 || target_index == candidate_count) {
                    fprintf(stderr,
                            "replay %s rules rejected the target action\n",
                            prime_rule_input_label(replay_rules.kind));
                    nn_free(net);
                    free(candidate_inputs);
                    free_replay_trace(&trace);
                    free_prime_rule_input(&replay_rules);
                    return EXIT_FAILURE;
                }
            } else {
                fill_candidate_features_masked(candidate_inputs,
                                               sample.p,
                                               sample.log_scale,
                                               trace_mask);
                if (target_index == ACTION_COUNT) {
                    fputs("replay trace target action_id is not in current candidates\n", stderr);
                    nn_free(net);
                    free(candidate_inputs);
                    free_replay_trace(&trace);
                    free_prime_rule_input(&replay_rules);
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
                free_prime_rule_input(&replay_rules);
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
             "%s feature_set=%s",
             replay_rules.kind == PRIME_RULE_INPUT_EXPANDED
                 ? "prime_gap successor trained from expanded-rule trace replay"
                 : replay_rules.kind == PRIME_RULE_INPUT_SYMBOLIC
                       ? "prime_gap successor trained from learned-wheel trace replay"
                       : "prime_gap successor trained from trace replay",
             trace_mask->name);
    snprintf(meta.feature_set, sizeof(meta.feature_set), "%s", trace_mask->name);

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
    free_prime_rule_input(&replay_rules);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

static void print_compare_usage(FILE *stream)
{
    fputs("usage: prime_compare [--rules symbolic-or-expanded-rules.crlprule] "
          "[--feature-set minimal|full|log-only|gap-only|residue-only] "
          "[--no-residue-features] [--full-features] "
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
                              const char *feature_set,
                              const char *evaluation_mode,
                              const char *rule_hash,
                              EvalMetrics metrics)
{
    double samples = metrics.samples == 0 ? 1.0 : (double)metrics.samples;
    printf("{\"schema\":\"crlp.compare.v1\",\"report_version\":1,\"task\":\"prime_gap\","
           "\"model_hash\":\"%s\",\"parent_hash\":\"%s\","
           "\"feature_set\":\"%s\",\"evaluation_mode\":\"%s\",\"rule_set_hash\":\"%s\","
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
           feature_set == NULL ? "unknown" : feature_set,
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

static void print_expanded_compare_json(const char *rule_hash,
                                        uint32_t seed,
                                        RuleSearchMetrics metrics)
{
    printf("{\"schema\":\"crlp.compare.v1\",\"report_version\":1,\"task\":\"prime_gap\","
           "\"model_hash\":\"none\",\"parent_hash\":\"unknown\","
           "\"evaluation_mode\":\"expanded_rules\","
           "\"rule_set_hash\":\"%s\",\"feature_encoder_version\":1,"
           "\"benchmark_seed\":%u,\"sample_count\":%zu,"
           "\"accepted_rules\":%zu,\"accepted_conditional_rules\":%zu,"
           "\"accepted_gap_rules\":%zu,\"accepted_candidate_rules\":%zu,"
           "\"false_rejections\":%zu,"
           "\"direct_exact\":%.8f,\"hand_wheel_exact\":%.8f,"
           "\"expanded_exact\":%.8f,\"expanded_mod\":%.8f}\n",
           rule_hash == NULL ? "unknown" : rule_hash,
           seed,
           metrics.samples,
           metrics.accepted_rules,
           metrics.accepted_conditional_rules,
           metrics.accepted_gap_rules,
           metrics.accepted_candidate_rules,
           metrics.false_rejections,
           metrics.direct_exact_tests,
           metrics.hand_wheel_exact_tests,
           metrics.expanded_exact_tests,
           metrics.expanded_mod_tests);
}

int nn_prime_compare_main(int argc, char **argv)
{
    const char *rules_path = NULL;
    const FeatureMask *feature_mask = &FEATURE_MASK_MINIMAL;
    size_t model_count = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--rules") == 0) {
            if (++i >= argc || argv[i][0] == '\0') {
                print_compare_usage(stderr);
                return EXIT_FAILURE;
            }
            rules_path = argv[i];
        } else if (strcmp(argv[i], "--feature-set") == 0) {
            if (++i >= argc || argv[i][0] == '\0') {
                print_compare_usage(stderr);
                return EXIT_FAILURE;
            }
            feature_mask = feature_mask_from_name(argv[i]);
            if (feature_mask == NULL) {
                print_compare_usage(stderr);
                return EXIT_FAILURE;
            }
        } else if (strcmp(argv[i], "--no-residue-features") == 0) {
            feature_mask = &FEATURE_MASK_MINIMAL;
        } else if (strcmp(argv[i], "--full-features") == 0) {
            feature_mask = &FEATURE_MASK_FULL;
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

    PrimeRuleInput compare_rules;
    init_prime_rule_input(&compare_rules);
    if (rules_path != NULL) {
        if (!load_prime_rule_input(rules_path, &compare_rules)) {
            fprintf(stderr, "failed to load symbolic or expanded rules: %s\n", rules_path);
            return EXIT_FAILURE;
        }
    }

    Sieve sieve;
    GapDataset train;
    GapDataset validation;
    GapDataset test;
    if (!build_sieve(sieve_limit, &sieve) ||
        !build_gap_datasets(&sieve, train_limit, test_limit, &train, &validation, &test)) {
        fputs("failed to build benchmark data\n", stderr);
        free_prime_rule_input(&compare_rules);
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
        free_prime_rule_input(&compare_rules);
        free_gap_dataset(&train);
        free_gap_dataset(&validation);
        free_gap_dataset(&test);
        free_sieve(&sieve);
        return EXIT_FAILURE;
    }

    puts("jsonl report:");
    printf("feature_set=%s residue_features=%s\n",
           feature_mask->name,
           feature_mask->use_residue ? "on" : "off");
    if (compare_rules.kind == PRIME_RULE_INPUT_SYMBOLIC) {
        SymbolicWheelMetrics symbolic_metrics;
        memset(&symbolic_metrics, 0, sizeof(symbolic_metrics));
        symbolic_metrics.accepted_rules = compare_rules.symbolic.count;
        symbolic_metrics = evaluate_symbolic_wheel(&sieve,
                                                   &compare_rules.symbolic,
                                                   &test,
                                                   DEFAULT_MAX_TEST_STATES,
                                                   symbolic_metrics);
        print_symbolic_compare_json(compare_rules.hash, benchmark_seed, symbolic_metrics);
        printf("learned wheel rules hash=%s direct_exact=%.2f hand_wheel_exact=%.2f "
               "learned_wheel_exact=%.2f false_rejections=%zu\n",
               compare_rules.hash,
               symbolic_metrics.direct_exact_tests,
               symbolic_metrics.hand_wheel_exact_tests,
               symbolic_metrics.learned_exact_tests,
               symbolic_metrics.test_false_rejections);
    } else if (compare_rules.kind == PRIME_RULE_INPUT_EXPANDED) {
        RuleSearchMetrics expanded_metrics;
        memset(&expanded_metrics, 0, sizeof(expanded_metrics));
        expanded_metrics.proposed_rules = compare_rules.expanded_metrics.proposed_rules;
        expanded_metrics.verified_rules = compare_rules.expanded_metrics.verified_rules;
        expanded_metrics.accepted_rules = compare_rules.expanded.count;
        expanded_metrics.accepted_conditional_rules =
            compare_rules.expanded_metrics.accepted_conditional_rules;
        expanded_metrics.accepted_gap_rules =
            compare_rules.expanded_metrics.accepted_gap_rules;
        expanded_metrics.accepted_candidate_rules =
            compare_rules.expanded_metrics.accepted_candidate_rules;
        expanded_metrics = evaluate_expanded_rules(&sieve,
                                                   &compare_rules.expanded,
                                                   &test,
                                                   DEFAULT_MAX_TEST_STATES,
                                                   expanded_metrics);
        print_expanded_compare_json(compare_rules.hash, benchmark_seed, expanded_metrics);
        printf("expanded rules hash=%s direct_exact=%.2f hand_wheel_exact=%.2f "
               "expanded_exact=%.2f expanded_mod=%.2f false_rejections=%zu "
               "accepted=%zu conditional=%zu candidate=%zu gap=%zu\n",
               compare_rules.hash,
               expanded_metrics.direct_exact_tests,
               expanded_metrics.hand_wheel_exact_tests,
               expanded_metrics.expanded_exact_tests,
               expanded_metrics.expanded_mod_tests,
               expanded_metrics.false_rejections,
               expanded_metrics.accepted_rules,
               expanded_metrics.accepted_conditional_rules,
               expanded_metrics.accepted_candidate_rules,
               expanded_metrics.accepted_gap_rules);
    }
    int ok = 1;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--rules") == 0) {
            ++i;
            continue;
        }
        if (strcmp(argv[i], "--feature-set") == 0) {
            ++i;
            continue;
        }
        if (strcmp(argv[i], "--no-residue-features") == 0 ||
            strcmp(argv[i], "--full-features") == 0) {
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
        if (meta.feature_set[0] != '\0' &&
            strcmp(meta.feature_set, "unknown") != 0 &&
            strcmp(meta.feature_set, feature_mask->name) != 0) {
            fprintf(stderr,
                    "checkpoint feature_set mismatch for %s: checkpoint=%s requested=%s\n",
                    argv[i],
                    meta.feature_set,
                    feature_mask->name);
            nn_free(net);
            ok = 0;
            continue;
        }
        EvalMetrics metrics = evaluate_model(net,
                                             &sieve,
                                             &test,
                                             DEFAULT_MAX_TEST_STATES,
                                             log_scale,
                                             feature_mask,
                                             1,
                                             1.0,
                                             candidate_inputs,
                                             scores,
                                             probabilities);
        print_report_json(hash,
                          &meta,
                          benchmark_seed,
                          feature_mask->name,
                          "raw",
                          "unknown",
                          metrics);
        print_comparison_row(argv[i], 1.0, metrics);
        if (compare_rules.kind != PRIME_RULE_INPUT_NONE) {
            EvalMetrics hand_wheel_metrics =
                evaluate_model_feature_wheel_masked(net,
                                                    &sieve,
                                                    &test,
                                                    DEFAULT_MAX_TEST_STATES,
                                                    log_scale,
                                                    feature_mask,
                                                    1,
                                                    1.0,
                                                    candidate_inputs,
                                                    scores,
                                                    probabilities);
            print_report_json(hash,
                              &meta,
                              benchmark_seed,
                              feature_mask->name,
                              "hand_wheel_model",
                              "hand-wheel",
                              hand_wheel_metrics);
            print_comparison_row("hand-wheel+model", 1.0, hand_wheel_metrics);

            if (compare_rules.kind == PRIME_RULE_INPUT_SYMBOLIC) {
                EvalMetrics learned_metrics =
                    evaluate_model_symbolic_masked(net,
                                                   &sieve,
                                                   &compare_rules.symbolic,
                                                   &test,
                                                   DEFAULT_MAX_TEST_STATES,
                                                   log_scale,
                                                   feature_mask,
                                                   1,
                                                   1.0,
                                                   candidate_inputs,
                                                   scores,
                                                   probabilities);
                print_report_json(hash,
                                  &meta,
                                  benchmark_seed,
                                  feature_mask->name,
                                  "learned_wheel_model",
                                  compare_rules.hash,
                                  learned_metrics);
                print_comparison_row("learned-wheel+model", 1.0, learned_metrics);
            } else {
                EvalMetrics expanded_metrics =
                    evaluate_model_expanded_masked(net,
                                                   &sieve,
                                                   &compare_rules.expanded,
                                                   &test,
                                                   DEFAULT_MAX_TEST_STATES,
                                                   log_scale,
                                                   feature_mask,
                                                   1,
                                                   1.0,
                                                   candidate_inputs,
                                                   scores,
                                                   probabilities);
                print_report_json(hash,
                                  &meta,
                                  benchmark_seed,
                                  feature_mask->name,
                                  "expanded_rule_model",
                                  compare_rules.hash,
                                  expanded_metrics);
                print_comparison_row("expanded-rule+model", 1.0, expanded_metrics);
            }
        }
        nn_free(net);
    }

    free(candidate_inputs);
    free(scores);
    free(probabilities);
    free_prime_rule_input(&compare_rules);
    free_gap_dataset(&train);
    free_gap_dataset(&validation);
    free_gap_dataset(&test);
    free_sieve(&sieve);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
