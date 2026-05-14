/* Supervised, ranking, verifier-RL, and ablation training loops.
 * Included by ../nn_tasks_prime_gap.c; not compiled as a standalone translation unit.
 */

static int train_supervised(NeuralNet *net,
                            const char *label,
                            const GapDataset *dataset,
                            size_t max_states,
                            size_t epochs,
                            nn_real learning_rate,
                            double log_scale,
                            const FeatureMask *mask,
                            nn_real *candidate_inputs,
                            size_t *order,
                            uint32_t *rng)
{
    size_t train_steps = dataset->count;
    if (max_states != 0 && max_states < train_steps) {
        train_steps = max_states;
    }

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        shuffle_order(order, dataset->count, rng);
        double started = now_seconds();
        double loss = 0.0;

        for (size_t step = 0; step < train_steps; ++step) {
            size_t index = order[step];
            fill_candidate_features_masked(candidate_inputs, dataset->prime[index], log_scale, mask);
            nn_real step_loss = nn_train_action_scorer(net,
                                                        candidate_inputs,
                                                        ACTION_COUNT,
                                                        gap_to_action(dataset->gap[index]),
                                                        learning_rate);
            if (isnan((double)step_loss)) {
                return 0;
            }
            loss += (double)step_loss;
        }

        double elapsed = now_seconds() - started;
        printf("%s epoch %zu loss=%.4f updates=%zu %.0f states/s\n",
               label,
               epoch + 1,
               loss / (double)train_steps,
               train_steps,
               elapsed > 0.0 ? (double)train_steps / elapsed : 0.0);
    }

    return 1;
}

static int train_pairwise_ranker(NeuralNet *net,
                                 const char *label,
                                 const GapDataset *dataset,
                                 size_t max_states,
                                 size_t epochs,
                                 nn_real learning_rate,
                                 double log_scale,
                                 const FeatureMask *mask,
                                 nn_real *candidate_inputs,
                                 size_t *order,
                                 uint32_t *rng)
{
    size_t train_steps = dataset->count;
    if (max_states != 0 && max_states < train_steps) {
        train_steps = max_states;
    }

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        shuffle_order(order, dataset->count, rng);
        double started = now_seconds();
        double loss = 0.0;

        for (size_t step = 0; step < train_steps; ++step) {
            size_t index = order[step];
            fill_candidate_features_masked(candidate_inputs, dataset->prime[index], log_scale, mask);
            nn_real step_loss = nn_train_pairwise_action_ranker(net,
                                                                candidate_inputs,
                                                                ACTION_COUNT,
                                                                gap_to_action(dataset->gap[index]),
                                                                (nn_real)1,
                                                                learning_rate);
            if (isnan((double)step_loss)) {
                return 0;
            }
            loss += (double)step_loss;
        }

        double elapsed = now_seconds() - started;
        printf("%s epoch %zu pairwise_hinge=%.4f updates=%zu %.0f states/s\n",
               label,
               epoch + 1,
               loss / (double)train_steps,
               train_steps,
               elapsed > 0.0 ? (double)train_steps / elapsed : 0.0);
    }

    return 1;
}

static int train_supervised_symbolic_masked(NeuralNet *net,
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
                                            uint32_t *rng)
{
    size_t train_steps = dataset->count;
    size_t survivor_actions[ACTION_COUNT];
    if (max_states != 0 && max_states < train_steps) {
        train_steps = max_states;
    }

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        shuffle_order(order, dataset->count, rng);
        double started = now_seconds();
        double loss = 0.0;
        size_t masked_out_targets = 0;
        size_t survivor_total = 0;

        for (size_t step = 0; step < train_steps; ++step) {
            size_t index = order[step];
            size_t target = gap_to_action(dataset->gap[index]);
            size_t survivor_count = fill_symbolic_survivor_candidates(rules,
                                                                      candidate_inputs,
                                                                      dataset->prime[index],
                                                                      log_scale,
                                                                      mask,
                                                                      survivor_actions);
            size_t survivor_target =
                survivor_index_for_action(survivor_actions, survivor_count, target);
            if (survivor_count == 0 || survivor_target == survivor_count) {
                ++masked_out_targets;
                continue;
            }
            survivor_total += survivor_count;

            nn_real step_loss = nn_train_action_scorer(net,
                                                       candidate_inputs,
                                                       survivor_count,
                                                       survivor_target,
                                                       learning_rate);
            if (isnan((double)step_loss)) {
                return 0;
            }
            loss += (double)step_loss;
        }

        double elapsed = now_seconds() - started;
        double divisor = train_steps == masked_out_targets
                             ? 1.0
                             : (double)(train_steps - masked_out_targets);
        printf("%s epoch %zu loss=%.4f updates=%zu masked_targets=%zu "
               "mean_survivors=%.2f %.0f states/s\n",
               label,
               epoch + 1,
               loss / divisor,
               train_steps - masked_out_targets,
               masked_out_targets,
               survivor_total / divisor,
               elapsed > 0.0 ? (double)train_steps / elapsed : 0.0);
    }

    return 1;
}

static int train_strict_verifier_rl(NeuralNet *net,
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
                                    uint32_t *rng)
{
    size_t train_steps = dataset->count;
    double baseline = 0.0;
    if (max_states != 0 && max_states < train_steps) {
        train_steps = max_states;
    }

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        shuffle_order(order, dataset->count, rng);
        double started = now_seconds();
        double objective = 0.0;
        size_t exact_hits = 0;

        for (size_t step = 0; step < train_steps; ++step) {
            size_t index = order[step];
            size_t target = gap_to_action(dataset->gap[index]);

            fill_candidate_features_masked(candidate_inputs, dataset->prime[index], log_scale, mask);
            if (!nn_score_action_rows(net, candidate_inputs, ACTION_COUNT, scores, probabilities)) {
                return 0;
            }

            size_t selected = sample_action(probabilities, rng);
            double reward = selected == target ? 1.0 : 0.0;
            double advantage = reward - baseline;

            baseline += 0.01 * (reward - baseline);
            exact_hits += reward > 0.0 ? 1u : 0u;

            nn_real step_loss = nn_train_softmax_scorer_cached(net,
                                                               ACTION_COUNT,
                                                               selected,
                                                               (nn_real)advantage,
                                                               learning_rate);
            if (isnan((double)step_loss)) {
                return 0;
            }
            objective += (double)step_loss;
        }

        double elapsed = now_seconds() - started;
        printf("%s epoch %zu sampled_exact=%.2f%% baseline=%.3f objective=%.4f updates=%zu %.0f states/s\n",
               label,
               epoch + 1,
               100.0 * (double)exact_hits / (double)train_steps,
               baseline,
               objective / (double)train_steps,
               train_steps,
               elapsed > 0.0 ? (double)train_steps / elapsed : 0.0);
    }

    return 1;
}

static int train_regularized_verifier_rl(NeuralNet *net,
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
                                         uint32_t *rng)
{
    size_t train_steps = dataset->count;
    double baseline = 0.0;
    if (max_states != 0 && max_states < train_steps) {
        train_steps = max_states;
    }

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        shuffle_order(order, dataset->count, rng);
        double started = now_seconds();
        double objective = 0.0;
        double mean_entropy = 0.0;
        double mean_reference_kl = 0.0;
        size_t exact_hits = 0;

        for (size_t step = 0; step < train_steps; ++step) {
            size_t index = order[step];
            size_t target = gap_to_action(dataset->gap[index]);

            fill_candidate_features_masked(candidate_inputs, dataset->prime[index], log_scale, mask);
            if (!nn_score_action_rows(net, candidate_inputs, ACTION_COUNT, scores, probabilities)) {
                return 0;
            }
            if (!nn_score_action_rows(reference_net,
                                      candidate_inputs,
                                      ACTION_COUNT,
                                      reference_scores,
                                      reference_probabilities)) {
                return 0;
            }

            size_t selected = sample_action(probabilities, rng);
            double reward = selected == target ? 1.0 : 0.0;
            double advantage = reward - baseline;

            baseline += 0.01 * (reward - baseline);
            exact_hits += reward > 0.0 ? 1u : 0u;
            mean_entropy += distribution_entropy(probabilities, ACTION_COUNT);
            mean_reference_kl +=
                distribution_kl(reference_probabilities, probabilities, ACTION_COUNT);

            nn_real step_loss = nn_train_softmax_scorer_policy_regularized_cached(
                net,
                ACTION_COUNT,
                selected,
                (nn_real)advantage,
                reference_probabilities,
                reference_weight,
                entropy_weight,
                learning_rate);
            if (isnan((double)step_loss)) {
                return 0;
            }
            objective += (double)step_loss;
        }

        double elapsed = now_seconds() - started;
        printf("%s epoch %zu sampled_exact=%.2f%% baseline=%.3f objective=%.4f "
               "entropy=%.3f ref_kl=%.4f updates=%zu %.0f states/s\n",
               label,
               epoch + 1,
               100.0 * (double)exact_hits / (double)train_steps,
               baseline,
               objective / (double)train_steps,
               mean_entropy / (double)train_steps,
               mean_reference_kl / (double)train_steps,
               train_steps,
               elapsed > 0.0 ? (double)train_steps / elapsed : 0.0);
    }

    return 1;
}

static int run_supervised_ablation(const char *label,
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
                                   uint32_t *rng)
{
    NeuralNet *net = nn_create_config(config);
    int ok;

    if (net == NULL) {
        return 0;
    }

    printf("\ntraining %s (%s features)\n", label, mask->name);
    ok = train_supervised(net,
                          label,
                          train,
                          max_train_states,
                          supervised_epochs,
                          (nn_real)0.005,
                          log_scale,
                          mask,
                          candidate_inputs,
                          order,
                          rng);
    if (ok) {
        print_metrics(label,
                      evaluate_model(net,
                                     sieve,
                                     test,
                                     max_test_states,
                                     log_scale,
                                     mask,
                                     0,
                                     1.0,
                                     candidate_inputs,
                                     scores,
                                     probabilities));
    }

    nn_free(net);
    return ok;
}

static int run_linear_scorer_baseline(const NNConfig *config,
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
                                      double *temperature_out)
{
    NeuralNet *net = nn_create_config(config);
    int ok;

    if (net == NULL) {
        return 0;
    }

    printf("\ntraining baseline linear-scorer (full features, no hidden layer)\n");
    ok = train_supervised(net,
                          "baseline linear-scorer",
                          train,
                          max_train_states,
                          supervised_epochs,
                          (nn_real)0.005,
                          log_scale,
                          &FEATURE_MASK_FULL,
                          candidate_inputs,
                          order,
                          rng);
    if (ok) {
        double validation_nll = 0.0;
        double temperature = fit_temperature(net,
                                             validation,
                                             max_validation_states,
                                             log_scale,
                                             &FEATURE_MASK_FULL,
                                             candidate_inputs,
                                             scores,
                                             probabilities,
                                             &validation_nll);
        printf("baseline linear-scorer temperature=%.2f validation_nll=%.4f\n",
               temperature,
               validation_nll);

        *metrics_out = evaluate_model(net,
                                      sieve,
                                      test,
                                      max_test_states,
                                      log_scale,
                                      &FEATURE_MASK_FULL,
                                      0,
                                      1.0,
                                      candidate_inputs,
                                      scores,
                                      probabilities);
        *temp_metrics_out = evaluate_model(net,
                                           sieve,
                                           test,
                                           max_test_states,
                                           log_scale,
                                           &FEATURE_MASK_FULL,
                                           0,
                                           temperature,
                                           candidate_inputs,
                                           scores,
                                           probabilities);
        *temperature_out = temperature;
        print_metrics("baseline linear-scorer", *metrics_out);
        print_metrics("baseline linear-scorer temp-scaled", *temp_metrics_out);
    }

    nn_free(net);
    return ok;
}

