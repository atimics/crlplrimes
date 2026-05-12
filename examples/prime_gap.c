#include "prime_gap.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    size_t train_limit_arg = parse_size_arg(argc, argv, 1, DEFAULT_TRAIN_LIMIT);
    size_t test_limit_arg = parse_size_arg(argc, argv, 2, DEFAULT_TEST_LIMIT);
    size_t supervised_epochs = parse_size_arg(argc, argv, 3, DEFAULT_SUPERVISED_EPOCHS);
    size_t rl_epochs = parse_size_arg(argc, argv, 4, DEFAULT_RL_EPOCHS);
    size_t max_train_states = parse_size_arg(argc, argv, 5, DEFAULT_MAX_TRAIN_STATES);
    size_t max_test_states = parse_size_arg(argc, argv, 6, DEFAULT_MAX_TEST_STATES);
    size_t verifier_diagnostics_arg = parse_size_arg(argc, argv, 7, DEFAULT_VERIFIER_DIAGNOSTICS);
    PrimeGapProfile profile;
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

    if (test_limit_arg <= train_limit_arg || train_limit_arg < 16 ||
        test_limit_arg > (size_t)UINT32_MAX - (size_t)INTERVAL_MAX_GAP ||
        train_limit_arg > (size_t)UINT32_MAX ||
        verifier_diagnostics_arg > 1) {
        fputs("usage: prime_gap [train_limit] [test_limit] [supervised_epochs] [rl_epochs] "
              "[max_train_states] [max_test_states] [verifier_diagnostics=0|1]\n",
              stderr);
        return EXIT_FAILURE;
    }

    uint32_t train_limit = (uint32_t)train_limit_arg;
    uint32_t test_limit = (uint32_t)test_limit_arg;
    uint32_t sieve_limit = test_limit + (uint32_t)INTERVAL_MAX_GAP;
    int include_verifier_diagnostics = verifier_diagnostics_arg != 0;

    printf("prime-gap scorer train_limit=%u test_limit=%u g_max=%d actions=%d features=%d "
           "verifier_diagnostics=%s\n",
           train_limit,
           test_limit,
           G_MAX,
           ACTION_COUNT,
           FEATURE_COUNT,
           include_verifier_diagnostics ? "on" : "off");

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

    double log_scale = log((double)test_limit + (double)G_MAX);
    size_t frequency_ranked[ACTION_COUNT];
    nn_real frequency_probabilities[ACTION_COUNT];

    printf("\n=== Baselines ===\n");
    build_frequency_policy(&fit, frequency_ranked, frequency_probabilities);
    print_metrics("baseline empirical-gap",
                  evaluate_static_policy(&sieve,
                                         &test,
                                         frequency_ranked,
                                         frequency_probabilities,
                                         max_test_states,
                                         include_verifier_diagnostics));
    print_metrics("baseline wheel-frequency",
                  evaluate_wheel_frequency_policy(&sieve,
                                                  &test,
                                                  frequency_probabilities,
                                                  max_test_states,
                                                  0,
                                                  probabilities));

    if (!build_residue_conditioned_policy(&fit, &residue_policy)) {
        fputs("failed to build residue-conditioned baseline\n", stderr);
        ok = 0;
        goto cleanup;
    }
    print_metrics("baseline residue-conditioned",
                  evaluate_residue_conditioned_policy(&residue_policy,
                                                      &sieve,
                                                      &test,
                                                      max_test_states,
                                                      include_verifier_diagnostics,
                                                      probabilities));

    print_metrics("model before",
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
                                 probabilities));

    printf("\n=== Feature Ablations ===\n");
    printf("Supervised-only runs; verifier diagnostics skipped to keep these capability-focused.\n");
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
                                 &rng);
    if (!ok) {
        goto cleanup;
    }

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
                                 &rng);
    if (!ok) {
        goto cleanup;
    }

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
                                 &rng);
    if (!ok) {
        goto cleanup;
    }

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
                                 &rng);
    if (!ok) {
        goto cleanup;
    }

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

    printf("\n=== Wheel-Survivor Task ===\n");
    printf("Ranks only candidates that survive the small-prime feature wheel; verifier diagnostics skipped.\n");
    print_metrics("model supervised wheel-survivor eval",
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
                                 probabilities));

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

    print_metrics("model wheel-survivor-supervised",
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
                                 probabilities));

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
    print_interval_metrics("baseline interval-frequency",
                           evaluate_interval_static_policy_cached(&test,
                                                                  &interval_eval_cache,
                                                                  interval_frequency_ranked,
                                                                  interval_frequency_probabilities));
    print_interval_metrics("baseline interval-wheel-frequency",
                           evaluate_interval_wheel_frequency_policy_cached(&test,
                                                                           &interval_eval_cache,
                                                                           interval_frequency_probabilities,
                                                                           probabilities));

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
    print_interval_metrics("model interval-linear",
                           evaluate_interval_model_cached(interval_linear_net,
                                                          &test,
                                                          &interval_eval_cache,
                                                          (double)G_MAX,
                                                          log_scale,
                                                          &FEATURE_MASK_FULL,
                                                          candidate_inputs,
                                                          scores,
                                                          probabilities));

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
    print_interval_metrics("model interval-mlp",
                           evaluate_interval_model_cached(interval_net,
                                                          &test,
                                                          &interval_eval_cache,
                                                          (double)G_MAX,
                                                          log_scale,
                                                          &FEATURE_MASK_FULL,
                                                          candidate_inputs,
                                                          scores,
                                                          probabilities));

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
    print_interval_metrics("model interval-raw-linear",
                           evaluate_interval_model_cached(interval_raw_linear_net,
                                                          &test,
                                                          &interval_eval_cache,
                                                          (double)G_MAX,
                                                          log_scale,
                                                          &FEATURE_MASK_RAW_RESIDUE,
                                                          candidate_inputs,
                                                          scores,
                                                          probabilities));

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
    print_interval_metrics("model interval-raw-mlp",
                           evaluate_interval_model_cached(interval_raw_net,
                                                          &test,
                                                          &interval_eval_cache,
                                                          (double)G_MAX,
                                                          log_scale,
                                                          &FEATURE_MASK_RAW_RESIDUE,
                                                          candidate_inputs,
                                                          scores,
                                                          probabilities));

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
        print_interval_metrics(label,
                               evaluate_interval_static_policy_cached(&test,
                                                                      &interval_eval_cache,
                                                                      interval_frequency_ranked,
                                                                      interval_frequency_probabilities));

        snprintf(label, sizeof(label), "baseline interval-wheel-frequency-%u", window_gap);
        print_interval_metrics(label,
                               evaluate_interval_wheel_frequency_policy_cached(&test,
                                                                               &interval_eval_cache,
                                                                               interval_frequency_probabilities,
                                                                               probabilities));

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
        print_interval_metrics(label,
                               evaluate_interval_model_cached(interval_raw_linear_net,
                                                              &test,
                                                              &interval_eval_cache,
                                                              (double)window_gap,
                                                              log_scale,
                                                              &FEATURE_MASK_RAW_RESIDUE,
                                                              candidate_inputs,
                                                              scores,
                                                              probabilities));
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
        print_interval_metrics(label,
                               evaluate_interval_model_cached(interval_raw_net,
                                                              &test,
                                                              &interval_eval_cache,
                                                              (double)window_gap,
                                                              log_scale,
                                                              &FEATURE_MASK_RAW_RESIDUE,
                                                              candidate_inputs,
                                                              scores,
                                                              probabilities));
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
