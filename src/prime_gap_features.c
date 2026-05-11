#include "prime_gap.h"

#include <math.h>
#include <stdlib.h>

size_t action_gap(size_t action)
{
    if (action == OVERFLOW_ACTION) {
        return G_MAX + 2;
    }

    return (action + 1) * 2;
}

uint32_t indexed_gap(size_t action)
{
    return (uint32_t)((action + 1u) * 2u);
}

size_t gap_to_action(uint32_t gap)
{
    if (gap > G_MAX) {
        return OVERFLOW_ACTION;
    }

    return (size_t)(gap / 2u - 1u);
}

int action_is_overflow(size_t action)
{
    return action == OVERFLOW_ACTION;
}

double gap_error_for_action(size_t action, uint32_t true_gap)
{
    if (action_is_overflow(action) && true_gap > G_MAX) {
        return 0.0;
    }

    return fabs((double)action_gap(action) - (double)true_gap);
}

size_t candidate_gap_signature(uint32_t p, uint32_t gap)
{
    size_t signature = 0;
    uint32_t candidate = p + gap;
    for (size_t i = 0; i < FEATURE_MOD_COUNT; ++i) {
        uint32_t mod = FEATURE_MODS[i];
        if (candidate > mod && candidate % mod == 0) {
            signature |= (size_t)1u << i;
        }
    }

    return signature;
}

size_t candidate_signature(uint32_t p, size_t action)
{
    if (action_is_overflow(action)) {
        return RESIDUE_SIGNATURE_COUNT;
    }

    return candidate_gap_signature(p, (uint32_t)action_gap(action));
}

int gap_survives_feature_wheel(uint32_t p, uint32_t gap)
{
    return candidate_gap_signature(p, gap) == 0;
}

int action_survives_feature_wheel(uint32_t p, size_t action)
{
    return action_is_overflow(action) || candidate_signature(p, action) == 0;
}

void fill_gap_features(nn_real *row,
                              uint32_t p,
                              uint32_t gap,
                              int overflow,
                              double gap_scale,
                              double log_scale,
                              const FeatureMask *mask)
{
    if (mask == NULL) {
        mask = &FEATURE_MASK_FULL;
    }

    double log_p = log((double)p) / log_scale;
    row[0] = mask->use_log ? (nn_real)log_p : (nn_real)0;
    row[1] = mask->use_gap ? (nn_real)((double)gap / gap_scale) : (nn_real)0;
    row[2] = mask->use_overflow && overflow ? (nn_real)1 : (nn_real)0;
    if (overflow) {
        for (size_t i = 0; i < FEATURE_MOD_COUNT; ++i) {
            row[3 + i] = 0;
        }
        return;
    }

    uint32_t candidate = p + gap;
    for (size_t i = 0; i < FEATURE_MOD_COUNT; ++i) {
        uint32_t mod = FEATURE_MODS[i];
        uint32_t residue = candidate % mod;

        if (mask->residue_mode == RESIDUE_FEATURE_DIVISIBILITY) {
            row[3 + i] = candidate > mod && residue == 0 ? (nn_real)1 : (nn_real)0;
        } else if (mask->residue_mode == RESIDUE_FEATURE_RAW_NORMALIZED) {
            row[3 + i] = (nn_real)((double)residue / (double)mod);
        } else {
            row[3 + i] = 0;
        }
    }
}

void fill_action_features(nn_real *row,
                                 uint32_t p,
                                 size_t action,
                                 double log_scale,
                                 const FeatureMask *mask)
{
    fill_gap_features(row,
                      p,
                      (uint32_t)action_gap(action),
                      action_is_overflow(action),
                      (double)G_MAX,
                      log_scale,
                      mask);
}

void fill_candidate_features_masked(nn_real *inputs,
                                           uint32_t p,
                                           double log_scale,
                                           const FeatureMask *mask)
{
    double profiled = prime_gap_profile_start();

    for (size_t action = 0; action < ACTION_COUNT; ++action) {
        fill_action_features(&inputs[action * FEATURE_COUNT], p, action, log_scale, mask);
    }

    prime_gap_profile_stop(PRIME_GAP_PROFILE_FEATURES, profiled);
}

size_t fill_wheel_survivor_features(nn_real *inputs,
                                           uint32_t p,
                                           size_t true_action,
                                           double log_scale,
                                           const FeatureMask *mask,
                                           size_t *target_index)
{
    size_t count = 0;
    double profiled = prime_gap_profile_start();
    *target_index = ACTION_COUNT;

    for (size_t action = 0; action < ACTION_COUNT; ++action) {
        if (!action_survives_feature_wheel(p, action)) {
            continue;
        }

        if (action == true_action) {
            *target_index = count;
        }
        fill_action_features(&inputs[count * FEATURE_COUNT], p, action, log_scale, mask);
        ++count;
    }

    prime_gap_profile_stop(PRIME_GAP_PROFILE_FEATURES, profiled);
    return count;
}

void fill_candidate_features(nn_real *inputs, uint32_t p, double log_scale)
{
    fill_candidate_features_masked(inputs, p, log_scale, &FEATURE_MASK_FULL);
}

void rank_actions(const nn_real *scores, size_t *ranked)
{
    for (size_t i = 0; i < ACTION_COUNT; ++i) {
        ranked[i] = i;
    }

    for (size_t i = 0; i < ACTION_COUNT; ++i) {
        size_t best = i;
        for (size_t j = i + 1; j < ACTION_COUNT; ++j) {
            if (scores[ranked[j]] > scores[ranked[best]]) {
                best = j;
            }
        }

        size_t tmp = ranked[i];
        ranked[i] = ranked[best];
        ranked[best] = tmp;
    }
}

void rank_actions_count(const nn_real *scores, size_t count, size_t *ranked)
{
    for (size_t i = 0; i < count; ++i) {
        ranked[i] = i;
    }

    for (size_t i = 0; i < count; ++i) {
        size_t best = i;
        for (size_t j = i + 1; j < count; ++j) {
            if (scores[ranked[j]] > scores[ranked[best]]) {
                best = j;
            }
        }

        size_t tmp = ranked[i];
        ranked[i] = ranked[best];
        ranked[best] = tmp;
    }
}

void rank_actions_wheel(const nn_real *scores, uint32_t p, size_t *ranked)
{
    size_t survivor_count = 0;
    size_t rejected_count = 0;
    size_t rejected[ACTION_COUNT];

    for (size_t action = 0; action < ACTION_COUNT; ++action) {
        if (action_survives_feature_wheel(p, action)) {
            ranked[survivor_count++] = action;
        } else {
            rejected[rejected_count++] = action;
        }
    }

    for (size_t i = 0; i < survivor_count; ++i) {
        size_t best = i;
        for (size_t j = i + 1; j < survivor_count; ++j) {
            if (scores[ranked[j]] > scores[ranked[best]]) {
                best = j;
            }
        }

        size_t tmp = ranked[i];
        ranked[i] = ranked[best];
        ranked[best] = tmp;
    }

    for (size_t i = 0; i < rejected_count; ++i) {
        ranked[survivor_count + i] = rejected[i];
    }
}

void softmax_temperature(const nn_real *scores,
                                nn_real *probabilities,
                                size_t count,
                                double temperature)
{
    if (temperature <= 0.0) {
        temperature = 1.0;
    }

    nn_real max_score = (nn_real)((double)scores[0] / temperature);
    for (size_t i = 1; i < count; ++i) {
        nn_real scaled = (nn_real)((double)scores[i] / temperature);
        if (scaled > max_score) {
            max_score = scaled;
        }
    }

    nn_real denominator = 0;
    for (size_t i = 0; i < count; ++i) {
        probabilities[i] = (nn_real)exp((double)scores[i] / temperature - (double)max_score);
        denominator += probabilities[i];
    }

    if (denominator <= (nn_real)0) {
        nn_real uniform = (nn_real)1 / (nn_real)count;
        for (size_t i = 0; i < count; ++i) {
            probabilities[i] = uniform;
        }
        return;
    }

    for (size_t i = 0; i < count; ++i) {
        probabilities[i] /= denominator;
    }
}

void softmax_temperature_wheel(const nn_real *scores,
                                      nn_real *probabilities,
                                      uint32_t p,
                                      double temperature)
{
    int have_score = 0;
    size_t survivor_count = 0;
    nn_real max_score = 0;

    if (temperature <= 0.0) {
        temperature = 1.0;
    }

    for (size_t action = 0; action < ACTION_COUNT; ++action) {
        probabilities[action] = 0;
        if (!action_survives_feature_wheel(p, action)) {
            continue;
        }

        ++survivor_count;
        nn_real scaled = (nn_real)((double)scores[action] / temperature);
        if (!have_score || scaled > max_score) {
            max_score = scaled;
            have_score = 1;
        }
    }

    if (!have_score) {
        nn_real uniform = (nn_real)1 / (nn_real)ACTION_COUNT;
        for (size_t action = 0; action < ACTION_COUNT; ++action) {
            probabilities[action] = uniform;
        }
        return;
    }

    nn_real denominator = 0;
    for (size_t action = 0; action < ACTION_COUNT; ++action) {
        if (!action_survives_feature_wheel(p, action)) {
            continue;
        }

        probabilities[action] =
            (nn_real)exp((double)scores[action] / temperature - (double)max_score);
        denominator += probabilities[action];
    }

    if (denominator <= (nn_real)0) {
        nn_real uniform = survivor_count == 0
                              ? (nn_real)1 / (nn_real)ACTION_COUNT
                              : (nn_real)1 / (nn_real)survivor_count;
        for (size_t action = 0; action < ACTION_COUNT; ++action) {
            probabilities[action] = action_survives_feature_wheel(p, action) ? uniform : 0;
        }
        return;
    }

    for (size_t action = 0; action < ACTION_COUNT; ++action) {
        probabilities[action] /= denominator;
    }
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
