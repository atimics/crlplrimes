/* Main prime-gap experiment CLI orchestration. */

#include "internal.h"

int nn_prime_gap_main(int argc, char **argv)
{
    PrimeGapRunConfig run_config;
    Sieve sieve;
    GapDataset train;
    GapDataset validation;
    GapDataset test;
    uint32_t rng;
    int ok = 1;
    char model_hash[65] = "unknown";
    char rule_hash[65] = "unknown";
    int rules_loaded_from_checkpoint = 0;

    if (!parse_run_config(argc, argv, &run_config) || !run_config_is_valid(&run_config)) {
        print_usage(stderr);
        return EXIT_FAILURE;
    }

    rng = (uint32_t)run_config.rng_seed_arg;

    uint32_t train_limit = (uint32_t)run_config.train_limit_arg;
    uint32_t test_limit = (uint32_t)run_config.test_limit_arg;
    uint32_t sieve_limit = test_limit + (uint32_t)G_MAX;
    uint32_t validation_start = validation_start_limit(train_limit);
    size_t max_validation_states = run_config.max_test_states;

    printf("prime-gap scorer train_limit=%u validation_start=%u test_limit=%u "
           "g_max=%d actions=%d features=%d\n",
           train_limit,
           validation_start,
           test_limit,
           G_MAX,
           ACTION_COUNT,
           FEATURE_COUNT);
    printf("regularized_rl reference_weight=%.3f entropy_weight=%.3f rng_seed=%u\n",
           run_config.regularized_reference_weight,
           run_config.regularized_entropy_weight,
           rng);
    printf("feature_set=%s residue_features=%s\n",
           run_config.feature_mask->name,
           run_config.feature_mask->use_residue ? "on" : "off");

    double started = now_seconds();
    if (!build_sieve(sieve_limit, &sieve)) {
        fputs("failed to build sieve\n", stderr);
        return EXIT_FAILURE;
    }
    printf("sieve limit=%u primes=%zu time=%.3fs\n",
           sieve_limit,
           sieve.prime_count,
           now_seconds() - started);

    if (!build_gap_datasets(&sieve, train_limit, test_limit, &train, &validation, &test)) {
        fputs("failed to build gap datasets\n", stderr);
        free_sieve(&sieve);
        return EXIT_FAILURE;
    }

    printf("dataset train=%zu exact_action_coverage=%.2f%% "
           "validation=%zu exact_action_coverage=%.2f%% "
           "test=%zu exact_action_coverage=%.2f%% "
           "overflow_actions train=%zu validation=%zu test=%zu\n",
           train.count,
           train.total_seen == 0
               ? 0.0
               : 100.0 * (double)(train.total_seen - train.overflow_count) /
                     (double)train.total_seen,
           validation.count,
           validation.total_seen == 0
               ? 0.0
               : 100.0 * (double)(validation.total_seen - validation.overflow_count) /
                     (double)validation.total_seen,
           test.count,
           test.total_seen == 0
               ? 0.0
               : 100.0 * (double)(test.total_seen - test.overflow_count) /
                     (double)test.total_seen,
           train.overflow_count,
           validation.overflow_count,
           test.overflow_count);

    if (train.count == 0 || validation.count == 0 || test.count == 0) {
        fputs("empty train, validation, or test split\n", stderr);
        free_gap_dataset(&train);
        free_gap_dataset(&validation);
        free_gap_dataset(&test);
        free_sieve(&sieve);
        return EXIT_FAILURE;
    }

    const size_t layers[] = {FEATURE_COUNT, NN_PRIME_GAP_HIDDEN1, NN_PRIME_GAP_HIDDEN2, 1};
    const size_t linear_layers[] = {FEATURE_COUNT, 1};
    NNConfig config;
    config.layer_sizes = layers;
    config.layer_count = sizeof(layers) / sizeof(layers[0]);
    config.seed = 42;
    config.hidden_activation = NN_ACTIVATION_RELU;
    config.output_activation = NN_ACTIVATION_LINEAR;
    config.loss = NN_LOSS_MSE;
    NNConfig linear_config = config;
    linear_config.layer_sizes = linear_layers;
    linear_config.layer_count = sizeof(linear_layers) / sizeof(linear_layers[0]);
    linear_config.hidden_activation = NN_ACTIVATION_LINEAR;
    linear_config.seed = 4201;

    NeuralNet *net = nn_create_config(&config);
    NeuralNet *learned_wheel_net = NULL;
    NeuralNet *pairwise_net = NULL;
    NeuralNet *strict_net = NULL;
    NeuralNet *regularized_net = NULL;
    ConditionedFrequencyPolicy conditioned_policy = {0};
    SymbolicRuleSet learned_symbolic_rules = {0};
    nn_real *candidate_inputs = calloc((size_t)ACTION_COUNT * FEATURE_COUNT, sizeof(*candidate_inputs));
    nn_real *scores = calloc(ACTION_COUNT, sizeof(*scores));
    nn_real *probabilities = calloc(ACTION_COUNT, sizeof(*probabilities));
    nn_real *reference_scores = calloc(ACTION_COUNT, sizeof(*reference_scores));
    nn_real *reference_probabilities = calloc(ACTION_COUNT, sizeof(*reference_probabilities));
    size_t *order = calloc(train.count, sizeof(*order));
    if (net == NULL || candidate_inputs == NULL || scores == NULL || probabilities == NULL ||
        reference_scores == NULL || reference_probabilities == NULL || order == NULL) {
        fputs("failed to allocate model buffers\n", stderr);
        ok = 0;
        goto cleanup;
    }

    fill_order(order, train.count);

    double log_scale = log((double)test_limit + (double)G_MAX);
    const FeatureMask *primary_mask = run_config.feature_mask;

    if (run_config.load_rules_path != NULL) {
        if (!load_symbolic_rules_with_hash(run_config.load_rules_path,
                                           &learned_symbolic_rules,
                                           rule_hash)) {
            fprintf(stderr, "failed to load symbolic rules: %s\n", run_config.load_rules_path);
            ok = 0;
            goto cleanup;
        }
        rules_loaded_from_checkpoint = 1;
        printf("loaded symbolic rules: %s hash=%s rules=%zu\n",
               run_config.load_rules_path,
               rule_hash,
               learned_symbolic_rules.count);
    }

    if (run_config.load_model_path != NULL) {
        NNCheckpointMeta loaded_meta;
        NeuralNet *loaded_net = nn_load_checkpoint(run_config.load_model_path, &loaded_meta);
        if (loaded_net == NULL) {
            fprintf(stderr, "failed to load checkpoint: %s\n", run_config.load_model_path);
            ok = 0;
            goto cleanup;
        }

        nn_free(net);
        net = loaded_net;
        if (!nn_hash_checkpoint_hex(run_config.load_model_path, model_hash)) {
            strcpy(model_hash, "unknown");
        }
        printf("loaded checkpoint: %s hash=%s feature_encoder_version=%u parent=%s\n",
               run_config.load_model_path,
               model_hash,
               loaded_meta.feature_encoder_version,
               loaded_meta.parent_model_hash[0] == '\0' ? "unknown"
                                                         : loaded_meta.parent_model_hash);
        print_metrics("loaded model",
                      evaluate_model(net,
                                     &sieve,
                                     &test,
                                     run_config.max_test_states,
                                     log_scale,
                                     primary_mask,
                                     1,
                                     1.0,
                                     candidate_inputs,
                                     scores,
                                     probabilities));
        if (rules_loaded_from_checkpoint) {
            print_metrics("loaded model learned-wheel",
                          evaluate_model_symbolic_masked(net,
                                                         &sieve,
                                                         &learned_symbolic_rules,
                                                         &test,
                                                         run_config.max_test_states,
                                                         log_scale,
                                                         primary_mask,
                                                         1,
                                                         1.0,
                                                         candidate_inputs,
                                                         scores,
                                                         probabilities));
        }
        if (run_config.trace_path != NULL &&
            !emit_prime_gap_trace(net,
                                  &sieve,
                                  &test,
                                  run_config.max_test_states,
                                  log_scale,
                                  primary_mask,
                                  run_config.trace_path,
                                  model_hash,
                                  rule_hash,
                                  (uint64_t)rng,
                                  candidate_inputs,
                                  scores,
                                  probabilities)) {
            fputs("failed to write trace\n", stderr);
            ok = 0;
        }
        goto cleanup;
    }

    size_t frequency_ranked[ACTION_COUNT];
    nn_real frequency_probabilities[ACTION_COUNT];
    EvalMetrics frequency_metrics;
    EvalMetrics wheel_frequency_metrics;
    EvalMetrics conditioned_metrics;
    EvalMetrics wheel_conditioned_metrics;
    EvalMetrics model_before_metrics;
    EvalMetrics linear_metrics;
    EvalMetrics linear_temp_metrics;
    EvalMetrics learned_wheel_metrics;
    EvalMetrics learned_wheel_temp_metrics;
    EvalMetrics pairwise_metrics;
    double linear_temperature = 1.0;
    double learned_wheel_temperature = 1.0;

    printf("\n=== Baselines ===\n");
    build_frequency_policy(&train, frequency_ranked, frequency_probabilities);
    (void)frequency_ranked;
    frequency_metrics = evaluate_frequency_policy(&sieve,
                                                  &test,
                                                  frequency_probabilities,
                                                  run_config.max_test_states,
                                                  1,
                                                  0);
    print_metrics("baseline empirical-gap", frequency_metrics);
    wheel_frequency_metrics = evaluate_frequency_policy(&sieve,
                                                        &test,
                                                        frequency_probabilities,
                                                        run_config.max_test_states,
                                                        1,
                                                        1);
    print_metrics("baseline hand-wheel empirical", wheel_frequency_metrics);

    if (!build_conditioned_frequency_policy(&conditioned_policy, &train)) {
        fputs("failed to build conditioned frequency policy\n", stderr);
        ok = 0;
        goto cleanup;
    }
    conditioned_metrics = evaluate_conditioned_frequency_policy(&conditioned_policy,
                                                                &sieve,
                                                                &test,
                                                                run_config.max_test_states,
                                                                1,
                                                                0,
                                                                scores,
                                                                probabilities);
    print_metrics("baseline residue-conditioned", conditioned_metrics);
    wheel_conditioned_metrics = evaluate_conditioned_frequency_policy(&conditioned_policy,
                                                                      &sieve,
                                                                      &test,
                                                                      run_config.max_test_states,
                                                                      1,
                                                                      1,
                                                                      scores,
                                                                      probabilities);
    print_metrics("baseline hand-wheel residue-conditioned", wheel_conditioned_metrics);
    free_conditioned_frequency_policy(&conditioned_policy);

    SymbolicWheelMetrics symbolic_metrics;
    if (rules_loaded_from_checkpoint) {
        memset(&symbolic_metrics, 0, sizeof(symbolic_metrics));
        symbolic_metrics.accepted_rules = learned_symbolic_rules.count;
    } else {
        if (!discover_symbolic_wheel(&sieve,
                                     &train,
                                     &validation,
                                     &learned_symbolic_rules,
                                     &symbolic_metrics)) {
            fputs("failed to discover symbolic wheel\n", stderr);
            ok = 0;
            goto cleanup;
        }
    }
    symbolic_metrics = evaluate_symbolic_wheel(&sieve,
                                               &learned_symbolic_rules,
                                               &test,
                                               run_config.max_test_states,
                                               symbolic_metrics);
    print_symbolic_wheel_report(&learned_symbolic_rules, symbolic_metrics);

    if (run_config.save_rules_path != NULL &&
        !save_symbolic_rule_checkpoint(&learned_symbolic_rules,
                                       run_config.save_rules_path,
                                       &symbolic_metrics,
                                       rule_hash)) {
        fputs("failed to save symbolic rules\n", stderr);
        ok = 0;
        goto cleanup;
    }
    if (run_config.save_rules_path != NULL) {
        printf("saved symbolic rules: %s hash=%s rules=%zu\n",
               run_config.save_rules_path,
               rule_hash,
               learned_symbolic_rules.count);
    }

    uint32_t linear_rng = rng;
    fill_order(order, train.count);
    ok = run_linear_scorer_baseline(&linear_config,
                                    primary_mask,
                                    &sieve,
                                    &train,
                                    &validation,
                                    &test,
                                    run_config.max_train_states,
                                    max_validation_states,
                                    run_config.max_test_states,
                                    run_config.supervised_epochs,
                                    log_scale,
                                    candidate_inputs,
                                    scores,
                                    probabilities,
                                    order,
                                    &linear_rng,
                                    &linear_metrics,
                                    &linear_temp_metrics,
                                    &linear_temperature);
    if (!ok) {
        goto cleanup;
    }
    fill_order(order, train.count);

    model_before_metrics = evaluate_model(net,
                                          &sieve,
                                          &test,
                                          run_config.max_test_states,
                                          log_scale,
                                          primary_mask,
                                          1,
                                          1.0,
                                          candidate_inputs,
                                          scores,
                                          probabilities);
    print_metrics("model before", model_before_metrics);

    printf("\n=== Feature Ablations ===\n");
    printf("Supervised-only runs; verifier diagnostics skipped to keep these capability-focused.\n");
    ok = run_supervised_ablation("ablation log-only",
                                 &FEATURE_MASK_LOG_ONLY,
                                 &config,
                                 &sieve,
                                 &train,
                                 &test,
                                 run_config.max_train_states,
                                 run_config.max_test_states,
                                 run_config.supervised_epochs,
                                 log_scale,
                                 candidate_inputs,
                                 scores,
                                 probabilities,
                                 order,
                                 &rng);
    if (!ok) {
        goto cleanup;
    }

    ok = run_supervised_ablation("ablation gap-only",
                                 &FEATURE_MASK_GAP_ONLY,
                                 &config,
                                 &sieve,
                                 &train,
                                 &test,
                                 run_config.max_train_states,
                                 run_config.max_test_states,
                                 run_config.supervised_epochs,
                                 log_scale,
                                 candidate_inputs,
                                 scores,
                                 probabilities,
                                 order,
                                 &rng);
    if (!ok) {
        goto cleanup;
    }

    ok = run_supervised_ablation("ablation residue-only",
                                 &FEATURE_MASK_RESIDUE_ONLY,
                                 &config,
                                 &sieve,
                                 &train,
                                 &test,
                                 run_config.max_train_states,
                                 run_config.max_test_states,
                                 run_config.supervised_epochs,
                                 log_scale,
                                 candidate_inputs,
                                 scores,
                                 probabilities,
                                 order,
                                 &rng);
    if (!ok) {
        goto cleanup;
    }

    ok = run_supervised_ablation("ablation full-sieve-features",
                                 &FEATURE_MASK_FULL,
                                 &config,
                                 &sieve,
                                 &train,
                                 &test,
                                 run_config.max_train_states,
                                 run_config.max_test_states,
                                 run_config.supervised_epochs,
                                 log_scale,
                                 candidate_inputs,
                                 scores,
                                 probabilities,
                                 order,
                                 &rng);
    if (!ok) {
        goto cleanup;
    }

    printf("\n=== Primary Model Training ===\n");
    ok = train_supervised(net,
                          "model supervised",
                          &train,
                          run_config.max_train_states,
                          run_config.supervised_epochs,
                          (nn_real)0.005,
                          log_scale,
                          primary_mask,
                          candidate_inputs,
                          order,
                          &rng);
    if (!ok) {
        goto cleanup;
    }

    double supervised_temperature_nll = 0.0;
    double supervised_temperature = fit_temperature(net,
                                                    &validation,
                                                    max_validation_states,
                                                    log_scale,
                                                    primary_mask,
                                                    candidate_inputs,
                                                    scores,
                                                    probabilities,
                                                    &supervised_temperature_nll);
    printf("model supervised temperature=%.2f validation_nll=%.4f\n",
           supervised_temperature,
           supervised_temperature_nll);

    EvalMetrics supervised_metrics = evaluate_model(net,
                                                    &sieve,
                                                    &test,
                                                    run_config.max_test_states,
                                                    log_scale,
                                                    primary_mask,
                                                    1,
                                                    1.0,
                                                    candidate_inputs,
                                                    scores,
                                                    probabilities);
    EvalMetrics supervised_temp_metrics = evaluate_model(net,
                                                         &sieve,
                                                         &test,
                                                         run_config.max_test_states,
                                                         log_scale,
                                                         primary_mask,
                                                         0,
                                                         supervised_temperature,
                                                         candidate_inputs,
                                                         scores,
                                                         probabilities);
    print_metrics("model supervised", supervised_metrics);
    print_metrics("model supervised temp-scaled", supervised_temp_metrics);
    EvalMetrics supervised_hand_wheel_metrics =
        evaluate_model_feature_wheel_masked(net,
                                            &sieve,
                                            &test,
                                            run_config.max_test_states,
                                            log_scale,
                                            primary_mask,
                                            1,
                                            1.0,
                                            candidate_inputs,
                                            scores,
                                            probabilities);
    EvalMetrics supervised_learned_wheel_metrics =
        evaluate_model_symbolic_masked(net,
                                       &sieve,
                                       &learned_symbolic_rules,
                                       &test,
                                       run_config.max_test_states,
                                       log_scale,
                                       primary_mask,
                                       1,
                                       1.0,
                                       candidate_inputs,
                                       scores,
                                       probabilities);
    print_metrics("model supervised hand-wheel", supervised_hand_wheel_metrics);
    print_metrics("model supervised learned-wheel", supervised_learned_wheel_metrics);

    printf("\n=== Ranking Loss Ablation ===\n");
    pairwise_net = nn_create_config(&config);
    if (pairwise_net == NULL) {
        fputs("failed to allocate pairwise-ranking model\n", stderr);
        ok = 0;
        goto cleanup;
    }
    fill_order(order, train.count);
    ok = train_pairwise_ranker(pairwise_net,
                               "model pairwise-ranking",
                               &train,
                               run_config.max_train_states,
                               run_config.supervised_epochs,
                               (nn_real)0.002,
                               log_scale,
                               primary_mask,
                               candidate_inputs,
                               order,
                               &rng);
    if (!ok) {
        goto cleanup;
    }
    pairwise_metrics = evaluate_model(pairwise_net,
                                      &sieve,
                                      &test,
                                      run_config.max_test_states,
                                      log_scale,
                                      primary_mask,
                                      1,
                                      1.0,
                                      candidate_inputs,
                                      scores,
                                      probabilities);
    print_metrics("model pairwise-ranking", pairwise_metrics);

    printf("\n=== Learned-Wheel Proposal Training ===\n");
    learned_wheel_net = nn_create_config(&config);
    if (learned_wheel_net == NULL) {
        fputs("failed to allocate learned-wheel model\n", stderr);
        ok = 0;
        goto cleanup;
    }
    fill_order(order, train.count);
    ok = train_supervised_symbolic_masked(learned_wheel_net,
                                          "model learned-wheel supervised",
                                          &learned_symbolic_rules,
                                          &train,
                                          run_config.max_train_states,
                                          run_config.supervised_epochs,
                                          (nn_real)0.005,
                                          log_scale,
                                          primary_mask,
                                          candidate_inputs,
                                          order,
                                          &rng);
    if (!ok) {
        goto cleanup;
    }

    double learned_wheel_temperature_nll = 0.0;
    learned_wheel_temperature = fit_temperature_symbolic_masked(learned_wheel_net,
                                                                &learned_symbolic_rules,
                                                                &validation,
                                                                max_validation_states,
                                                                log_scale,
                                                                primary_mask,
                                                                candidate_inputs,
                                                                scores,
                                                                probabilities,
                                                                &learned_wheel_temperature_nll);
    printf("model learned-wheel supervised temperature=%.2f validation_nll=%.4f\n",
           learned_wheel_temperature,
           learned_wheel_temperature_nll);

    learned_wheel_metrics = evaluate_model_symbolic_masked(learned_wheel_net,
                                                           &sieve,
                                                           &learned_symbolic_rules,
                                                           &test,
                                                           run_config.max_test_states,
                                                           log_scale,
                                                           primary_mask,
                                                           1,
                                                           1.0,
                                                           candidate_inputs,
                                                           scores,
                                                           probabilities);
    learned_wheel_temp_metrics = evaluate_model_symbolic_masked(learned_wheel_net,
                                                                &sieve,
                                                                &learned_symbolic_rules,
                                                                &test,
                                                                run_config.max_test_states,
                                                                log_scale,
                                                                primary_mask,
                                                                0,
                                                                learned_wheel_temperature,
                                                                candidate_inputs,
                                                                scores,
                                                                probabilities);
    print_metrics("model learned-wheel supervised", learned_wheel_metrics);
    print_metrics("model learned-wheel supervised temp-scaled",
                  learned_wheel_temp_metrics);

    printf("\n=== Strict Verifier RL ===\n");
    int have_rl_metrics = 0;
    double rl_temperature = 1.0;
    EvalMetrics strict_metrics;
    EvalMetrics strict_temp_metrics;
    double regularized_temperature = 1.0;
    EvalMetrics regularized_metrics;
    EvalMetrics regularized_temp_metrics;

    if (run_config.rl_epochs == 0) {
        printf("skipped because rl_epochs=0\n");
    } else {
        strict_net = nn_clone(net);
        regularized_net = nn_clone(net);
        if (strict_net == NULL || regularized_net == NULL) {
            fputs("failed to clone supervised model\n", stderr);
            ok = 0;
            goto cleanup;
        }

        uint32_t strict_rng = rng;
        uint32_t regularized_rng = rng;

        fill_order(order, train.count);
        ok = train_strict_verifier_rl(strict_net,
                                      "model strict-rl",
                                      &train,
                                      run_config.max_train_states,
                                      run_config.rl_epochs,
                                      (nn_real)0.001,
                                      log_scale,
                                      primary_mask,
                                      candidate_inputs,
                                      scores,
                                      probabilities,
                                      order,
                                      &strict_rng);
        if (!ok) {
            goto cleanup;
        }

        double rl_temperature_nll = 0.0;
        rl_temperature = fit_temperature(strict_net,
                                         &validation,
                                         max_validation_states,
                                         log_scale,
                                         primary_mask,
                                         candidate_inputs,
                                         scores,
                                         probabilities,
                                         &rl_temperature_nll);
        printf("model strict-rl temperature=%.2f validation_nll=%.4f\n",
               rl_temperature,
               rl_temperature_nll);

        strict_metrics = evaluate_model(strict_net,
                                        &sieve,
                                        &test,
                                        run_config.max_test_states,
                                        log_scale,
                                        primary_mask,
                                        1,
                                        1.0,
                                        candidate_inputs,
                                        scores,
                                        probabilities);
        strict_temp_metrics = evaluate_model(strict_net,
                                             &sieve,
                                             &test,
                                             run_config.max_test_states,
                                             log_scale,
                                             primary_mask,
                                             0,
                                             rl_temperature,
                                             candidate_inputs,
                                             scores,
                                             probabilities);
        print_metrics("model strict-rl", strict_metrics);
        print_metrics("model strict-rl temp-scaled", strict_temp_metrics);

        printf("\n=== Regularized Verifier RL ===\n");
        printf("reference_weight=%.3f entropy_weight=%.3f\n",
               run_config.regularized_reference_weight,
               run_config.regularized_entropy_weight);
        fill_order(order, train.count);
        ok = train_regularized_verifier_rl(regularized_net,
                                           net,
                                           "model regularized-rl",
                                           &train,
                                           run_config.max_train_states,
                                           run_config.rl_epochs,
                                           (nn_real)0.001,
                                           (nn_real)run_config.regularized_reference_weight,
                                           (nn_real)run_config.regularized_entropy_weight,
                                           log_scale,
                                           primary_mask,
                                           candidate_inputs,
                                           scores,
                                           probabilities,
                                           reference_scores,
                                           reference_probabilities,
                                           order,
                                           &regularized_rng);
        if (!ok) {
            goto cleanup;
        }

        double regularized_temperature_nll = 0.0;
        regularized_temperature = fit_temperature(regularized_net,
                                                  &validation,
                                                  max_validation_states,
                                                  log_scale,
                                                  primary_mask,
                                                  candidate_inputs,
                                                  scores,
                                                  probabilities,
                                                  &regularized_temperature_nll);
        printf("model regularized-rl temperature=%.2f validation_nll=%.4f\n",
               regularized_temperature,
               regularized_temperature_nll);

        regularized_metrics = evaluate_model(regularized_net,
                                             &sieve,
                                             &test,
                                             run_config.max_test_states,
                                             log_scale,
                                             primary_mask,
                                             1,
                                             1.0,
                                             candidate_inputs,
                                             scores,
                                             probabilities);
        regularized_temp_metrics = evaluate_model(regularized_net,
                                                  &sieve,
                                                  &test,
                                                  run_config.max_test_states,
                                                  log_scale,
                                                  primary_mask,
                                                  0,
                                                  regularized_temperature,
                                                  candidate_inputs,
                                                  scores,
                                                  probabilities);
        print_metrics("model regularized-rl", regularized_metrics);
        print_metrics("model regularized-rl temp-scaled", regularized_temp_metrics);
        have_rl_metrics = 1;
    }

    print_comparison_header();
    print_comparison_row("empirical-gap", 1.0, frequency_metrics);
    print_comparison_row("hand-wheel empirical", 1.0, wheel_frequency_metrics);
    print_comparison_row("residue-conditioned", 1.0, conditioned_metrics);
    print_comparison_row("hand-wheel residue-conditioned", 1.0, wheel_conditioned_metrics);
    print_comparison_row("linear-scorer", 1.0, linear_metrics);
    print_comparison_row("linear-scorer temp-scaled", linear_temperature, linear_temp_metrics);
    print_comparison_row("supervised", 1.0, supervised_metrics);
    print_comparison_row("supervised temp-scaled",
                         supervised_temperature,
                         supervised_temp_metrics);
    print_comparison_row("supervised hand-wheel", 1.0, supervised_hand_wheel_metrics);
    print_comparison_row("supervised learned-wheel", 1.0, supervised_learned_wheel_metrics);
    print_comparison_row("pairwise-ranking", 1.0, pairwise_metrics);
    print_comparison_row("learned-wheel supervised", 1.0, learned_wheel_metrics);
    print_comparison_row("learned-wheel supervised temp-scaled",
                         learned_wheel_temperature,
                         learned_wheel_temp_metrics);
    if (have_rl_metrics) {
        print_comparison_row("strict-rl", 1.0, strict_metrics);
        print_comparison_row("strict-rl temp-scaled", rl_temperature, strict_temp_metrics);
        print_comparison_row("regularized-rl", 1.0, regularized_metrics);
        print_comparison_row("regularized-rl temp-scaled",
                             regularized_temperature,
                             regularized_temp_metrics);
    }

    if (run_config.save_path != NULL) {
        char checkpoint_notes[128];
        snprintf(checkpoint_notes,
                 sizeof(checkpoint_notes),
                 "prime_gap action scorer feature_set=%s",
                 primary_mask->name);
        if (!save_prime_checkpoint_with_meta(net,
                                             run_config.save_path,
                                             checkpoint_notes,
                                             "unknown",
                                             NULL,
                                             model_hash)) {
            fputs("failed to save checkpoint\n", stderr);
            ok = 0;
            goto cleanup;
        }
    }

    if (run_config.save_learned_model_path != NULL) {
        char learned_model_hash[65] = "unknown";
        char learned_notes[128];
        snprintf(learned_notes,
                 sizeof(learned_notes),
                 "prime_gap learned-wheel action scorer feature_set=%s",
                 primary_mask->name);
        if (!save_prime_checkpoint_with_meta(learned_wheel_net,
                                             run_config.save_learned_model_path,
                                             learned_notes,
                                             "unknown",
                                             rule_hash,
                                             learned_model_hash)) {
            fputs("failed to save learned-wheel checkpoint\n", stderr);
            ok = 0;
            goto cleanup;
        }
    }

    if (run_config.trace_path != NULL &&
        !emit_prime_gap_trace(net,
                              &sieve,
                              &test,
                              run_config.max_test_states,
                              log_scale,
                              primary_mask,
                              run_config.trace_path,
                              model_hash,
                              rule_hash,
                              (uint64_t)run_config.rng_seed_arg,
                              candidate_inputs,
                              scores,
                              probabilities)) {
        fputs("failed to write trace\n", stderr);
        ok = 0;
        goto cleanup;
    }

cleanup:
    free_symbolic_rule_set(&learned_symbolic_rules);
    free_conditioned_frequency_policy(&conditioned_policy);
    free(order);
    free(reference_probabilities);
    free(reference_scores);
    free(probabilities);
    free(scores);
    free(candidate_inputs);
    nn_free(regularized_net);
    nn_free(strict_net);
    nn_free(pairwise_net);
    nn_free(learned_wheel_net);
    nn_free(net);
    free_gap_dataset(&train);
    free_gap_dataset(&validation);
    free_gap_dataset(&test);
    free_sieve(&sieve);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
