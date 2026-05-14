/* Candidate feature encoding, ranking helpers, and wheel verifier work accounting. */

#include "internal.h"

double gap_error_for_action(size_t action, uint32_t true_gap)
{
    if (action_is_overflow(action) && true_gap > G_MAX) {
        return 0.0;
    }

    return fabs((double)action_gap(action) - (double)true_gap);
}

void fill_candidate_features_masked(nn_real *inputs,
                                    uint32_t p,
                                    double log_scale,
                                    const FeatureMask *mask)
{
    if (mask == NULL) {
        mask = &FEATURE_MASK_MINIMAL;
    }

    double log_p = log((double)p) / log_scale;

    for (size_t action = 0; action < ACTION_COUNT; ++action) {
        nn_real *row = &inputs[action * FEATURE_COUNT];

        row[0] = mask->use_log ? (nn_real)log_p : (nn_real)0;
        row[1] = mask->use_gap ? (nn_real)((double)action_gap(action) / (double)G_MAX)
                               : (nn_real)0;
        row[2] = mask->use_overflow && action_is_overflow(action) ? (nn_real)1 : (nn_real)0;
        if (action_is_overflow(action)) {
            for (size_t i = 0; i < FEATURE_MOD_COUNT; ++i) {
                row[3 + i] = 0;
            }
            continue;
        }

        uint32_t gap = (uint32_t)action_gap(action);
        uint32_t candidate = p + gap;
        for (size_t i = 0; i < FEATURE_MOD_COUNT; ++i) {
            uint32_t mod = FEATURE_MODS[i];
            row[3 + i] = mask->use_residue && candidate > mod && candidate % mod == 0
                             ? (nn_real)1
                             : (nn_real)0;
        }
    }
}

size_t residue_signature(uint32_t p, size_t action, size_t mod_count)
{
    if (action_is_overflow(action)) {
        return (size_t)1u << mod_count;
    }

    uint32_t candidate = p + (uint32_t)action_gap(action);
    size_t signature = 0;
    for (size_t i = 0; i < mod_count; ++i) {
        uint32_t mod = FEATURE_MODS[i];
        if (candidate > mod && candidate % mod == 0) {
            signature |= (size_t)1u << i;
        }
    }
    return signature;
}

static int feature_wheel_rejects_action(uint32_t p, size_t action)
{
    if (action_is_overflow(action)) {
        return 0;
    }

    uint32_t candidate = p + (uint32_t)action_gap(action);
    for (size_t i = 0; i < FEATURE_MOD_COUNT; ++i) {
        uint32_t mod = FEATURE_MODS[i];
        if (candidate > mod && candidate % mod == 0) {
            return 1;
        }
    }
    return 0;
}

void apply_feature_wheel_filter(uint32_t p,
                                nn_real *scores,
                                nn_real *probabilities)
{
    double survivor_probability = 0.0;
    size_t survivors = 0;

    for (size_t action = 0; action < ACTION_COUNT; ++action) {
        if (feature_wheel_rejects_action(p, action)) {
            scores[action] = (nn_real)-1.0e9;
            probabilities[action] = (nn_real)0;
            continue;
        }

        survivor_probability += (double)probabilities[action];
        ++survivors;
    }

    if (survivors == 0 || survivor_probability <= 0.0) {
        nn_real uniform = (nn_real)(1.0 / (double)ACTION_COUNT);
        for (size_t action = 0; action < ACTION_COUNT; ++action) {
            scores[action] = uniform;
            probabilities[action] = uniform;
        }
        return;
    }

    for (size_t action = 0; action < ACTION_COUNT; ++action) {
        probabilities[action] =
            (nn_real)((double)probabilities[action] / survivor_probability);
    }
}

void fill_candidate_features(nn_real *inputs, uint32_t p, double log_scale)
{
    fill_candidate_features_masked(inputs, p, log_scale, &FEATURE_MASK_MINIMAL);
}

void rank_actions(const nn_real *scores, size_t *ranked)
{
    nn_rank_action_scores(scores, ACTION_COUNT, ranked);
}

size_t true_rank(const size_t *ranked, size_t true_action)
{
    for (size_t i = 0; i < ACTION_COUNT; ++i) {
        if (ranked[i] == true_action) {
            return i;
        }
    }

    return ACTION_COUNT;
}

size_t proposal_steps_with_topk(const size_t *ranked, size_t top_k, size_t true_action)
{
    uint8_t tried[ACTION_COUNT] = {0};
    size_t steps = 0;

    if (top_k > ACTION_COUNT) {
        top_k = ACTION_COUNT;
    }

    for (size_t i = 0; i < top_k; ++i) {
        size_t action = ranked[i];
        if (tried[action]) {
            continue;
        }

        tried[action] = 1;
        ++steps;
        if (action == true_action) {
            return steps;
        }
    }

    for (size_t action = 0; action <= true_action; ++action) {
        if (tried[action]) {
            continue;
        }

        ++steps;
        if (action == true_action) {
            return steps;
        }
    }

    return steps;
}

static int wheel_probe_gap(const Sieve *sieve, uint32_t p, uint32_t gap, VerifierWork *work)
{
    uint32_t candidate = p + gap;

    ++work->candidates;
    for (size_t i = 0; i < FEATURE_MOD_COUNT; ++i) {
        uint32_t mod = FEATURE_MODS[i];
        if (candidate <= mod) {
            continue;
        }

        ++work->mod_tests;
        if (candidate % mod == 0) {
            return 0;
        }
    }

    ++work->exact_tests;
    return is_prime_sieve(sieve, candidate);
}

static int wheel_probe_action_cached(const Sieve *sieve,
                                     uint32_t p,
                                     size_t action,
                                     uint8_t *tested,
                                     uint8_t *prime_result,
                                     VerifierWork *work)
{
    if (action_is_overflow(action)) {
        return 0;
    }

    if (!tested[action]) {
        tested[action] = 1;
        prime_result[action] =
            (uint8_t)wheel_probe_gap(sieve, p, (uint32_t)action_gap(action), work);
    }

    return prime_result[action] != 0;
}

static int wheel_scan_to_gap_cached(const Sieve *sieve,
                                    uint32_t p,
                                    uint32_t limit_gap,
                                    uint8_t *tested,
                                    uint8_t *prime_result,
                                    VerifierWork *work)
{
    for (uint32_t gap = 2; gap <= limit_gap; gap += 2) {
        int is_prime = 0;
        if (gap <= G_MAX) {
            is_prime = wheel_probe_action_cached(
                sieve, p, gap_to_action(gap), tested, prime_result, work);
        } else {
            is_prime = wheel_probe_gap(sieve, p, gap, work);
        }

        if (is_prime) {
            return 1;
        }
    }

    return 0;
}

VerifierWork direct_scan_work(uint32_t true_gap)
{
    VerifierWork work;
    work.candidates = true_gap / 2u;
    work.mod_tests = 0;
    work.exact_tests = true_gap / 2u;
    return work;
}

VerifierWork wheel_scan_work(const Sieve *sieve, uint32_t p, uint32_t true_gap)
{
    uint8_t tested[EXACT_ACTION_COUNT] = {0};
    uint8_t prime_result[EXACT_ACTION_COUNT] = {0};
    VerifierWork work = {0, 0, 0};

    wheel_scan_to_gap_cached(sieve, p, true_gap, tested, prime_result, &work);
    return work;
}

VerifierWork assisted_wheel_work(const Sieve *sieve,
                                 uint32_t p,
                                 uint32_t true_gap,
                                 const size_t *ranked,
                                 size_t top_k)
{
    uint8_t tested[EXACT_ACTION_COUNT] = {0};
    uint8_t prime_result[EXACT_ACTION_COUNT] = {0};
    VerifierWork work = {0, 0, 0};

    if (top_k > ACTION_COUNT) {
        top_k = ACTION_COUNT;
    }

    for (size_t i = 0; i < top_k; ++i) {
        size_t proposed = ranked[i];
        if (action_is_overflow(proposed)) {
            wheel_scan_to_gap_cached(sieve, p, true_gap, tested, prime_result, &work);
            return work;
        }

        if (!wheel_probe_action_cached(sieve, p, proposed, tested, prime_result, &work)) {
            continue;
        }

        wheel_scan_to_gap_cached(
            sieve, p, (uint32_t)action_gap(proposed), tested, prime_result, &work);
        return work;
    }

    wheel_scan_to_gap_cached(sieve, p, true_gap, tested, prime_result, &work);
    return work;
}
