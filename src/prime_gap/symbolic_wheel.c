/* Legacy learned symbolic wheel discovery, persistence, and reporting.
 * Included by ../nn_tasks_prime_gap.c; not compiled as a standalone translation unit.
 */

static size_t symbolic_rule_count(void)
{
    size_t count = 0;
    for (uint32_t modulus = SYMBOLIC_RULE_MOD_MIN;
         modulus <= (uint32_t)NN_SYMBOLIC_RULE_MAX_MOD;
         ++modulus) {
        count += modulus;
    }
    return count;
}

static void fill_symbolic_rule_offsets(size_t *offsets)
{
    size_t offset = 0;
    for (uint32_t modulus = 0; modulus <= (uint32_t)NN_SYMBOLIC_RULE_MAX_MOD + 1u; ++modulus) {
        offsets[modulus] = 0;
    }
    for (uint32_t modulus = SYMBOLIC_RULE_MOD_MIN;
         modulus <= (uint32_t)NN_SYMBOLIC_RULE_MAX_MOD;
         ++modulus) {
        offsets[modulus] = offset;
        offset += modulus;
    }
    offsets[(uint32_t)NN_SYMBOLIC_RULE_MAX_MOD + 1u] = offset;
}

static void init_symbolic_rules(SymbolicRejectRule *rules, const size_t *offsets)
{
    for (uint32_t modulus = SYMBOLIC_RULE_MOD_MIN;
         modulus <= (uint32_t)NN_SYMBOLIC_RULE_MAX_MOD;
         ++modulus) {
        for (uint32_t residue = 0; residue < modulus; ++residue) {
            SymbolicRejectRule *rule = &rules[offsets[modulus] + residue];
            rule->modulus = modulus;
            rule->residue = residue;
        }
    }
}

static void mark_domain_invalid_symbolic_residues(const Sieve *sieve,
                                                  uint8_t *invalid,
                                                  const size_t *offsets)
{
    for (size_t prime_index = 0; prime_index < sieve->prime_count; ++prime_index) {
        uint32_t prime = sieve->primes[prime_index];
        for (uint32_t modulus = SYMBOLIC_RULE_MOD_MIN;
             modulus <= (uint32_t)NN_SYMBOLIC_RULE_MAX_MOD && modulus < prime;
             ++modulus) {
            invalid[offsets[modulus] + (prime % modulus)] = 1;
        }
    }
}

static void accumulate_symbolic_rule_rejects(SymbolicRejectRule *rules,
                                             const size_t *offsets,
                                             const GapDataset *dataset,
                                             int validation)
{
    for (size_t sample = 0; sample < dataset->count; ++sample) {
        uint32_t p = dataset->prime[sample];
        uint32_t true_gap = dataset->gap[sample];

        for (uint32_t gap = 2; gap < true_gap; gap += 2) {
            uint32_t candidate = p + gap;
            for (uint32_t modulus = SYMBOLIC_RULE_MOD_MIN;
                 modulus <= (uint32_t)NN_SYMBOLIC_RULE_MAX_MOD && modulus < candidate;
                 ++modulus) {
                SymbolicRejectRule *rule = &rules[offsets[modulus] + (candidate % modulus)];
                if (validation) {
                    ++rule->validation_rejects;
                } else {
                    ++rule->train_rejects;
                }
            }
        }
    }
}

static uint32_t gcd_u32(uint32_t a, uint32_t b)
{
    while (b != 0) {
        uint32_t tmp = a % b;
        a = b;
        b = tmp;
    }
    return a;
}

static void fill_symbolic_rule_features(const SymbolicRejectRule *rule,
                                        size_t max_train_rejects,
                                        nn_real *features)
{
    double modulus = (double)rule->modulus;
    double residue = (double)rule->residue;
    double max_modulus = (double)NN_SYMBOLIC_RULE_MAX_MOD;
    double support_scale = max_train_rejects == 0 ? 1.0 : (double)max_train_rejects;
    uint32_t common = gcd_u32(rule->modulus, rule->residue);

    features[0] = (nn_real)(modulus / max_modulus);
    features[1] = (nn_real)(residue / modulus);
    features[2] = rule->residue == 0 ? (nn_real)1 : (nn_real)0;
    features[3] = (nn_real)((double)common / modulus);
    features[4] = (nn_real)((double)rule->train_rejects / support_scale);
}

static int score_symbolic_rules_with_model(SymbolicRejectRule *rules,
                                           size_t count,
                                           double *loss_out)
{
    size_t max_train_rejects = 0;
    for (size_t i = 0; i < count; ++i) {
        if (rules[i].train_rejects > max_train_rejects) {
            max_train_rejects = rules[i].train_rejects;
        }
    }

    nn_real *inputs = calloc(count * SYMBOLIC_RULE_FEATURE_COUNT, sizeof(*inputs));
    nn_real *targets = calloc(count, sizeof(*targets));
    nn_real *scores = calloc(count, sizeof(*scores));
    if (inputs == NULL || targets == NULL || scores == NULL) {
        free(inputs);
        free(targets);
        free(scores);
        return 0;
    }

    double support_scale = max_train_rejects == 0 ? 1.0 : (double)max_train_rejects;
    for (size_t i = 0; i < count; ++i) {
        fill_symbolic_rule_features(&rules[i],
                                    max_train_rejects,
                                    &inputs[i * SYMBOLIC_RULE_FEATURE_COUNT]);
        targets[i] = rules[i].domain_valid
                         ? (nn_real)((double)rules[i].train_rejects / support_scale)
                         : (nn_real)0;
    }

    const size_t layers[] = {SYMBOLIC_RULE_FEATURE_COUNT, 16, 8, 1};
    NNConfig config;
    config.layer_sizes = layers;
    config.layer_count = sizeof(layers) / sizeof(layers[0]);
    config.seed = 7319;
    config.hidden_activation = NN_ACTIVATION_RELU;
    config.output_activation = NN_ACTIVATION_LINEAR;
    config.loss = NN_LOSS_MSE;

    NeuralNet *net = nn_create_config(&config);
    if (net == NULL) {
        free(inputs);
        free(targets);
        free(scores);
        return 0;
    }

    nn_real loss = nn_train_batched(net,
                                    inputs,
                                    targets,
                                    count,
                                    256,
                                    (nn_real)0.02,
                                    SYMBOLIC_RULE_SCORER_EPOCHS,
                                    1);
    nn_forward_batch(net, inputs, scores, count);
    for (size_t i = 0; i < count; ++i) {
        rules[i].model_score = scores[i];
    }
    if (loss_out != NULL) {
        *loss_out = (double)loss;
    }

    nn_free(net);
    free(inputs);
    free(targets);
    free(scores);
    return 1;
}

static int compare_symbolic_rules(const void *left, const void *right)
{
    const SymbolicRejectRule *a = (const SymbolicRejectRule *)left;
    const SymbolicRejectRule *b = (const SymbolicRejectRule *)right;

    if ((double)a->model_score < (double)b->model_score) {
        return 1;
    }
    if ((double)a->model_score > (double)b->model_score) {
        return -1;
    }
    if (a->train_rejects != b->train_rejects) {
        return a->train_rejects < b->train_rejects ? 1 : -1;
    }
    if (a->validation_rejects != b->validation_rejects) {
        return a->validation_rejects < b->validation_rejects ? 1 : -1;
    }
    if (a->modulus != b->modulus) {
        return a->modulus < b->modulus ? -1 : 1;
    }
    if (a->residue != b->residue) {
        return a->residue < b->residue ? -1 : 1;
    }
    return 0;
}

static int compare_symbolic_rules_canonical(const void *left, const void *right)
{
    const SymbolicRejectRule *a = (const SymbolicRejectRule *)left;
    const SymbolicRejectRule *b = (const SymbolicRejectRule *)right;

    if (a->modulus != b->modulus) {
        return a->modulus < b->modulus ? -1 : 1;
    }
    if (a->residue != b->residue) {
        return a->residue < b->residue ? -1 : 1;
    }
    return 0;
}


static int symbolic_rule_subset_of(const SymbolicRejectRule *candidate,
                                   const SymbolicRejectRule *accepted)
{
    return candidate->modulus % accepted->modulus == 0 &&
           candidate->residue % accepted->modulus == accepted->residue;
}

static int symbolic_rule_is_covered(const SymbolicRuleSet *accepted,
                                    const SymbolicRejectRule *candidate)
{
    for (size_t i = 0; i < accepted->count; ++i) {
        if (symbolic_rule_subset_of(candidate, &accepted->items[i])) {
            return 1;
        }
    }
    return 0;
}

static void symbolic_rule_remove_subsets(SymbolicRuleSet *accepted,
                                         const SymbolicRejectRule *candidate)
{
    size_t write = 0;
    for (size_t read = 0; read < accepted->count; ++read) {
        if (symbolic_rule_subset_of(&accepted->items[read], candidate)) {
            continue;
        }
        if (write != read) {
            accepted->items[write] = accepted->items[read];
        }
        ++write;
    }
    accepted->count = write;
}

static void free_symbolic_rule_set(SymbolicRuleSet *rules)
{
    free(rules->items);
    memset(rules, 0, sizeof(*rules));
}

static int save_symbolic_rule_checkpoint(const SymbolicRuleSet *rules,
                                         const char *path,
                                         const SymbolicWheelMetrics *metrics,
                                         char out_hash[65])
{
    if (rules == NULL || path == NULL) {
        return 0;
    }

    FILE *fp = fopen(path, "w");
    if (fp == NULL) {
        return 0;
    }

    int ok = fprintf(fp, "crlp.symbolic_rules.v1\n") >= 0 &&
             fprintf(fp, "rule_count %zu\n", rules->count) >= 0 &&
             fprintf(fp, "max_modulus %u\n", (uint32_t)NN_SYMBOLIC_RULE_MAX_MOD) >= 0 &&
             fprintf(fp, "feature_mod_count %u\n", (uint32_t)FEATURE_MOD_COUNT) >= 0;
    if (metrics != NULL) {
        ok = ok &&
             fprintf(fp,
                     "proposed_rules %zu\n"
                     "domain_valid_rules %zu\n"
                     "accepted_train_reject_support %zu\n"
                     "validation_reject_support %zu\n"
                     "proposer_loss %.17g\n",
                     metrics->proposed_rules,
                     metrics->domain_valid_rules,
                     metrics->train_rejects,
                     metrics->validation_rejects,
                     metrics->proposer_loss) >= 0;
    } else {
        ok = ok &&
             fprintf(fp,
                     "proposed_rules 0\n"
                     "domain_valid_rules 0\n"
                     "accepted_train_reject_support 0\n"
                     "validation_reject_support 0\n"
                     "proposer_loss 0\n") >= 0;
    }

    for (size_t i = 0; ok && i < rules->count; ++i) {
        const SymbolicRejectRule *rule = &rules->items[i];
        ok = fprintf(fp,
                     "rule %u %u %zu %zu %.9g\n",
                     rule->modulus,
                     rule->residue,
                     rule->train_rejects,
                     rule->validation_rejects,
                     (double)rule->model_score) >= 0;
    }

    if (fclose(fp) != 0) {
        ok = 0;
    }
    if (!ok) {
        return 0;
    }
    if (out_hash != NULL && !nn_hash_checkpoint_hex(path, out_hash)) {
        return 0;
    }
    return 1;
}

static int load_symbolic_rule_checkpoint(const char *path, SymbolicRuleSet *rules)
{
    memset(rules, 0, sizeof(*rules));
    if (path == NULL) {
        return 0;
    }

    FILE *fp = fopen(path, "r");
    if (fp == NULL) {
        return 0;
    }

    char magic[64];
    size_t count = 0;
    unsigned int max_modulus = 0;
    unsigned int feature_mod_count = 0;
    int ok = fscanf(fp, "%63s", magic) == 1 &&
             strcmp(magic, "crlp.symbolic_rules.v1") == 0 &&
             fscanf(fp, " rule_count %zu", &count) == 1 &&
             fscanf(fp, " max_modulus %u", &max_modulus) == 1 &&
             fscanf(fp, " feature_mod_count %u", &feature_mod_count) == 1;

    size_t ignored_size = 0;
    double ignored_loss = 0.0;
    ok = ok &&
         fscanf(fp, " proposed_rules %zu", &ignored_size) == 1 &&
         fscanf(fp, " domain_valid_rules %zu", &ignored_size) == 1 &&
         fscanf(fp, " accepted_train_reject_support %zu", &ignored_size) == 1 &&
         fscanf(fp, " validation_reject_support %zu", &ignored_size) == 1 &&
         fscanf(fp, " proposer_loss %lf", &ignored_loss) == 1;
    ok = ok && max_modulus == (unsigned int)NN_SYMBOLIC_RULE_MAX_MOD &&
         feature_mod_count == (unsigned int)FEATURE_MOD_COUNT;

    if (ok) {
        rules->items = calloc(count == 0 ? 1 : count, sizeof(*rules->items));
        rules->capacity = count;
        ok = rules->items != NULL;
    }

    for (size_t i = 0; ok && i < count; ++i) {
        unsigned int modulus = 0;
        unsigned int residue = 0;
        size_t train_rejects = 0;
        size_t validation_rejects = 0;
        double model_score = 0.0;
        ok = fscanf(fp,
                    " rule %u %u %zu %zu %lf",
                    &modulus,
                    &residue,
                    &train_rejects,
                    &validation_rejects,
                    &model_score) == 5 &&
             modulus >= SYMBOLIC_RULE_MOD_MIN &&
             modulus <= max_modulus &&
             residue < modulus;
        if (!ok) {
            break;
        }

        rules->items[i].modulus = (uint32_t)modulus;
        rules->items[i].residue = (uint32_t)residue;
        rules->items[i].train_rejects = train_rejects;
        rules->items[i].validation_rejects = validation_rejects;
        rules->items[i].model_score = (nn_real)model_score;
        rules->items[i].domain_valid = 1;
        ++rules->count;
    }

    if (fclose(fp) != 0) {
        ok = 0;
    }
    if (!ok || rules->count != count) {
        free_symbolic_rule_set(rules);
        return 0;
    }

    qsort(rules->items,
          rules->count,
          sizeof(*rules->items),
          compare_symbolic_rules_canonical);
    return 1;
}

static int load_symbolic_rules_with_hash(const char *path,
                                         SymbolicRuleSet *rules,
                                         char out_hash[65])
{
    if (!load_symbolic_rule_checkpoint(path, rules)) {
        return 0;
    }
    if (out_hash != NULL && !nn_hash_checkpoint_hex(path, out_hash)) {
        strcpy(out_hash, "unknown");
    }
    return 1;
}

static int discover_symbolic_wheel(const Sieve *sieve,
                                   const GapDataset *train,
                                   const GapDataset *validation,
                                   SymbolicRuleSet *accepted,
                                   SymbolicWheelMetrics *metrics)
{
    size_t offsets[(uint32_t)NN_SYMBOLIC_RULE_MAX_MOD + 2u];
    size_t count = symbolic_rule_count();
    SymbolicRejectRule *rules = calloc(count, sizeof(*rules));
    uint8_t *invalid = calloc(count, sizeof(*invalid));

    memset(accepted, 0, sizeof(*accepted));
    memset(metrics, 0, sizeof(*metrics));
    metrics->proposed_rules = count;

    if (rules == NULL || invalid == NULL) {
        free(rules);
        free(invalid);
        return 0;
    }

    fill_symbolic_rule_offsets(offsets);
    init_symbolic_rules(rules, offsets);
    mark_domain_invalid_symbolic_residues(sieve, invalid, offsets);

    for (size_t i = 0; i < count; ++i) {
        rules[i].domain_valid = invalid[i] ? 0 : 1;
        metrics->domain_valid_rules += rules[i].domain_valid ? 1u : 0u;
    }

    accumulate_symbolic_rule_rejects(rules, offsets, train, 0);
    accumulate_symbolic_rule_rejects(rules, offsets, validation, 1);
    if (!score_symbolic_rules_with_model(rules, count, &metrics->proposer_loss)) {
        free_symbolic_rule_set(accepted);
        free(rules);
        free(invalid);
        return 0;
    }
    qsort(rules, count, sizeof(*rules), compare_symbolic_rules);

    accepted->capacity = NN_SYMBOLIC_MAX_LEARNED_RULES;
    accepted->items = calloc(accepted->capacity, sizeof(*accepted->items));
    if (accepted->items == NULL) {
        free(rules);
        free(invalid);
        return 0;
    }

    for (size_t i = 0; i < count; ++i) {
        if (!rules[i].domain_valid || rules[i].train_rejects == 0) {
            continue;
        }
        if (symbolic_rule_is_covered(accepted, &rules[i])) {
            continue;
        }

        symbolic_rule_remove_subsets(accepted, &rules[i]);
        if (accepted->count >= accepted->capacity) {
            continue;
        }
        accepted->items[accepted->count++] = rules[i];
    }

    metrics->accepted_rules = accepted->count;
    qsort(accepted->items,
          accepted->count,
          sizeof(*accepted->items),
          compare_symbolic_rules_canonical);
    for (size_t i = 0; i < accepted->count; ++i) {
        metrics->train_rejects += accepted->items[i].train_rejects;
        metrics->validation_rejects += accepted->items[i].validation_rejects;
    }
    free(rules);
    free(invalid);
    return 1;
}

static int symbolic_rule_rejects(const SymbolicRejectRule *rule, uint32_t candidate)
{
    return candidate > rule->modulus && candidate % rule->modulus == rule->residue;
}

static int learned_symbolic_probe_gap(const Sieve *sieve,
                                      const SymbolicRuleSet *rules,
                                      uint32_t p,
                                      uint32_t gap,
                                      VerifierWork *work,
                                      int *rejected)
{
    uint32_t candidate = p + gap;

    ++work->candidates;
    *rejected = 0;
    for (size_t i = 0; i < rules->count; ++i) {
        ++work->mod_tests;
        if (symbolic_rule_rejects(&rules->items[i], candidate)) {
            *rejected = 1;
            return 0;
        }
    }

    ++work->exact_tests;
    return is_prime_sieve(sieve, candidate);
}

static VerifierWork learned_symbolic_scan_work(const Sieve *sieve,
                                               const SymbolicRuleSet *rules,
                                               uint32_t p,
                                               uint32_t true_gap,
                                               int *false_rejection)
{
    VerifierWork work = {0, 0, 0};
    *false_rejection = 0;

    for (uint32_t gap = 2; gap <= true_gap; gap += 2) {
        int rejected = 0;
        int prime = learned_symbolic_probe_gap(sieve, rules, p, gap, &work, &rejected);
        if (rejected && gap == true_gap) {
            *false_rejection = 1;
            return work;
        }
        if (prime) {
            return work;
        }
    }

    return work;
}

static SymbolicWheelMetrics evaluate_symbolic_wheel(const Sieve *sieve,
                                                    const SymbolicRuleSet *rules,
                                                    const GapDataset *dataset,
                                                    size_t max_states,
                                                    SymbolicWheelMetrics metrics)
{
    size_t samples = dataset->count;
    if (max_states != 0 && max_states < samples) {
        samples = max_states;
    }

    for (size_t sample = 0; sample < samples; ++sample) {
        size_t index = eval_sample_index(dataset, sample, samples);
        uint32_t p = dataset->prime[index];
        uint32_t gap = dataset->gap[index];
        VerifierWork direct = direct_scan_work(gap);
        VerifierWork hand = wheel_scan_work(sieve, p, gap);
        int false_rejection = 0;
        VerifierWork learned = learned_symbolic_scan_work(sieve,
                                                          rules,
                                                          p,
                                                          gap,
                                                          &false_rejection);

        metrics.direct_exact_tests += (double)direct.exact_tests;
        metrics.hand_wheel_mod_tests += (double)hand.mod_tests;
        metrics.hand_wheel_exact_tests += (double)hand.exact_tests;
        metrics.learned_mod_tests += (double)learned.mod_tests;
        metrics.learned_exact_tests += (double)learned.exact_tests;
        metrics.test_false_rejections += false_rejection ? 1u : 0u;
    }

    metrics.samples = samples;
    if (samples != 0) {
        double denominator = (double)samples;
        metrics.direct_exact_tests /= denominator;
        metrics.hand_wheel_mod_tests /= denominator;
        metrics.hand_wheel_exact_tests /= denominator;
        metrics.learned_mod_tests /= denominator;
        metrics.learned_exact_tests /= denominator;
    }
    return metrics;
}

static void print_symbolic_rule(const SymbolicRejectRule *rule)
{
    printf("n %% %u == %u", rule->modulus, rule->residue);
}

static void print_symbolic_wheel_report(const SymbolicRuleSet *rules,
                                        SymbolicWheelMetrics metrics)
{
    double learned_saved = metrics.direct_exact_tests - metrics.learned_exact_tests;
    double hand_saved = metrics.direct_exact_tests - metrics.hand_wheel_exact_tests;
    double learned_saved_pct = metrics.direct_exact_tests == 0.0
                                   ? 0.0
                                   : 100.0 * learned_saved / metrics.direct_exact_tests;
    double hand_saved_pct = metrics.direct_exact_tests == 0.0
                                ? 0.0
                                : 100.0 * hand_saved / metrics.direct_exact_tests;

    printf("\n[learned symbolic wheel]\n");
    printf("  grammar: reject candidate n when n > m and n %% m == r; m=3..%u\n",
           (uint32_t)NN_SYMBOLIC_RULE_MAX_MOD);
    printf("  proposer/verifier\n");
    printf("    neural_rule_scorer_epochs=%u final_mse=%.6f\n",
           (uint32_t)SYMBOLIC_RULE_SCORER_EPOCHS,
           metrics.proposer_loss);
    printf("    proposed_rules=%zu domain_valid=%zu accepted=%zu max_rules=%u\n",
           metrics.proposed_rules,
           metrics.domain_valid_rules,
           metrics.accepted_rules,
           (uint32_t)NN_SYMBOLIC_MAX_LEARNED_RULES);
    printf("    accepted_train_reject_support=%zu validation_reject_support=%zu\n",
           metrics.train_rejects,
           metrics.validation_rejects);
    printf("  accepted rules");
    for (size_t i = 0; i < rules->count; ++i) {
        printf("%s", i == 0 ? ": " : ", ");
        print_symbolic_rule(&rules->items[i]);
        if (i == 15 && rules->count > 16) {
            printf(", ...");
            break;
        }
    }
    if (rules->count == 0) {
        printf(": none");
    }
    printf("\n");
    printf("  verifier diagnostics\n");
    printf("    samples=%zu false_rejections=%zu\n",
           metrics.samples,
           metrics.test_false_rejections);
    printf("    direct exact=%.2f\n", metrics.direct_exact_tests);
    printf("    hand wheel exact=%.2f saved=%.2f (%.1f%%) mod_tests=%.2f\n",
           metrics.hand_wheel_exact_tests,
           hand_saved,
           hand_saved_pct,
           metrics.hand_wheel_mod_tests);
    printf("    learned wheel exact=%.2f saved=%.2f (%.1f%%) mod_tests=%.2f\n",
           metrics.learned_exact_tests,
           learned_saved,
           learned_saved_pct,
           metrics.learned_mod_tests);
}

