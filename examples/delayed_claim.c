#include "nn.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    DC_STATE_FEATURES = 6,
    DC_ACTION_COUNT = 4,
    DC_FEATURE_COUNT = DC_STATE_FEATURES + DC_ACTION_COUNT,
    DC_TRAIN_SAMPLES = 4000,
    DC_TEST_SAMPLES = 1000,
    DC_TRAIN_SEED_START = 1000,
    DC_TEST_SEED_START = 900000,
    DC_REPLAY_EPOCHS = 6,
    DC_CALIBRATION_BINS = 10,
};

static const nn_real DC_REPLAY_LEARNING_RATE = (nn_real)0.008;
static const nn_real DC_REPLAY_HARD_WEIGHT = (nn_real)1.0;
static const nn_real DC_REPLAY_SOFT_WEIGHT = (nn_real)0.35;

typedef struct {
    nn_real state[DC_STATE_FEATURES];
    size_t target;
    nn_real utility[DC_ACTION_COUNT];
} ClaimState;

typedef struct {
    size_t samples;
    size_t correct;
    double nll;
    double ece;
    double utility;
    double proposal_steps;
    double baseline_steps;
    size_t bin_count[DC_CALIBRATION_BINS];
    double bin_confidence[DC_CALIBRATION_BINS];
    double bin_accuracy[DC_CALIBRATION_BINS];
} ClaimMetrics;

static uint32_t dc_rng_next(uint32_t *state)
{
    *state = (*state * 1664525u) + 1013904223u;
    return *state;
}

static nn_real dc_rng_unit(uint32_t *state)
{
    return (nn_real)((double)dc_rng_next(state) / (double)UINT32_MAX);
}

static int parse_size_text(const char *text, size_t *out_value)
{
    if (text == NULL || text[0] == '\0' || text[0] == '-' || text[0] == '+') {
        return 0;
    }

    errno = 0;
    char *end = NULL;
    unsigned long value = strtoul(text, &end, 10);
    if (end == text || *end != '\0' || errno == ERANGE || value == 0) {
        return 0;
    }

    *out_value = (size_t)value;
    return 1;
}

static int parse_u64_text(const char *text, uint64_t *out_value)
{
    if (text == NULL || text[0] == '\0' || text[0] == '-' || text[0] == '+') {
        return 0;
    }

    errno = 0;
    char *end = NULL;
    unsigned long long value = strtoull(text, &end, 10);
    if (end == text || *end != '\0' || errno == ERANGE) {
        return 0;
    }

    *out_value = (uint64_t)value;
    return 1;
}

static int parse_real_text(const char *text, nn_real *out_value)
{
    if (text == NULL || text[0] == '\0') {
        return 0;
    }

    errno = 0;
    char *end = NULL;
    double value = strtod(text, &end);
    if (end == text || *end != '\0' || errno == ERANGE || !isfinite(value) || value < 0.0) {
        return 0;
    }

    *out_value = (nn_real)value;
    return 1;
}

static double dc_safe_probability(double probability)
{
    return probability < 1e-12 ? 1e-12 : probability;
}

static ClaimState generate_claim_state(uint64_t seed)
{
    uint32_t rng = (uint32_t)(seed ^ (seed >> 32) ^ 0x9e3779b9u);
    ClaimState state;
    for (size_t i = 0; i < DC_STATE_FEATURES; ++i) {
        state.state[i] = dc_rng_unit(&rng);
    }

    double reliability = (double)state.state[0];
    double evidence = (double)state.state[1];
    double contradiction = (double)state.state[2];
    double age = (double)state.state[3];
    double history = (double)state.state[4];
    double noise = (double)state.state[5];
    double truth_score = 0.45 * reliability + 0.35 * evidence + 0.20 * history -
                         0.30 * contradiction - 0.15 * noise - 0.10 * age;
    int resolves_true = truth_score >= 0.38;
    double uncertainty = fabs(truth_score - 0.38);

    state.utility[0] = resolves_true ? (nn_real)2.0 : (nn_real)-2.5;
    state.utility[1] = resolves_true ? (nn_real)-0.3 : (nn_real)0.3;
    state.utility[2] = (nn_real)(0.95 - 0.6 * uncertainty - 0.25 * age);
    state.utility[3] = resolves_true ? (nn_real)1.1 : (nn_real)-0.2;
    if (contradiction > 0.70 || noise > 0.80) {
        state.utility[2] += (nn_real)0.35;
        state.utility[0] -= (nn_real)0.75;
    }

    state.target = 0;
    for (size_t action = 1; action < DC_ACTION_COUNT; ++action) {
        if (state.utility[action] > state.utility[state.target]) {
            state.target = action;
        }
    }
    return state;
}

static void fill_claim_candidate_features(nn_real *candidate_inputs, const ClaimState *state)
{
    for (size_t action = 0; action < DC_ACTION_COUNT; ++action) {
        nn_real *row = &candidate_inputs[action * DC_FEATURE_COUNT];
        memcpy(row, state->state, DC_STATE_FEATURES * sizeof(*row));
        for (size_t i = 0; i < DC_ACTION_COUNT; ++i) {
            row[DC_STATE_FEATURES + i] = action == i ? (nn_real)1 : (nn_real)0;
        }
    }
}

static uint32_t claim_action_id_for_action(size_t action)
{
    return (uint32_t)action;
}

static int claim_action_from_action_id(uint32_t action_id, size_t *out_action)
{
    if (action_id >= DC_ACTION_COUNT || out_action == NULL) {
        return 0;
    }

    *out_action = (size_t)action_id;
    return 1;
}

static void fill_claim_action_ids(uint32_t *action_ids)
{
    for (size_t action = 0; action < DC_ACTION_COUNT; ++action) {
        action_ids[action] = claim_action_id_for_action(action);
    }
}

static size_t claim_row_for_action_id(const uint32_t *action_ids,
                                      size_t candidate_count,
                                      uint32_t action_id)
{
    for (size_t row = 0; row < candidate_count; ++row) {
        if (action_ids[row] == action_id) {
            return row;
        }
    }
    return candidate_count;
}

static NeuralNet *create_claim_model(unsigned int seed)
{
    static const size_t layers[] = {DC_FEATURE_COUNT, 24, 12, 1};
    NNConfig config;
    config.layer_sizes = layers;
    config.layer_count = sizeof(layers) / sizeof(layers[0]);
    config.seed = seed;
    config.hidden_activation = NN_ACTIVATION_RELU;
    config.output_activation = NN_ACTIVATION_LINEAR;
    config.loss = NN_LOSS_MSE;
    return nn_create_config(&config);
}

static ClaimMetrics evaluate_claim_model(NeuralNet *net,
                                         uint64_t seed_start,
                                         size_t samples,
                                         nn_real *candidate_inputs,
                                         nn_real *scores,
                                         nn_real *probabilities)
{
    ClaimMetrics metrics;
    memset(&metrics, 0, sizeof(metrics));
    size_t ranked[DC_ACTION_COUNT];

    for (size_t i = 0; i < samples; ++i) {
        ClaimState state = generate_claim_state(seed_start + (uint64_t)i);
        fill_claim_candidate_features(candidate_inputs, &state);
        if (!nn_score_action_rows(net,
                                  candidate_inputs,
                                  DC_ACTION_COUNT,
                                  scores,
                                  probabilities) ||
            !nn_rank_action_scores(scores, DC_ACTION_COUNT, ranked)) {
            continue;
        }

        size_t selected = ranked[0];
        size_t target_rank = DC_ACTION_COUNT;
        for (size_t rank = 0; rank < DC_ACTION_COUNT; ++rank) {
            if (ranked[rank] == state.target) {
                target_rank = rank;
                break;
            }
        }
        double confidence = (double)probabilities[selected];
        size_t bin = confidence >= 1.0 ? DC_CALIBRATION_BINS - 1
                                       : (size_t)(confidence * DC_CALIBRATION_BINS);
        if (bin >= DC_CALIBRATION_BINS) {
            bin = DC_CALIBRATION_BINS - 1;
        }

        ++metrics.samples;
        metrics.correct += selected == state.target ? 1u : 0u;
        metrics.nll += -log(dc_safe_probability((double)probabilities[state.target]));
        metrics.utility += (double)state.utility[selected];
        metrics.proposal_steps += (double)target_rank + 1.0;
        metrics.baseline_steps += (double)state.target + 1.0;
        ++metrics.bin_count[bin];
        metrics.bin_confidence[bin] += confidence;
        metrics.bin_accuracy[bin] += selected == state.target ? 1.0 : 0.0;
    }

    double sample_count = metrics.samples == 0 ? 1.0 : (double)metrics.samples;
    metrics.nll /= sample_count;
    metrics.utility /= sample_count;
    metrics.proposal_steps /= sample_count;
    metrics.baseline_steps /= sample_count;
    for (size_t bin = 0; bin < DC_CALIBRATION_BINS; ++bin) {
        if (metrics.bin_count[bin] == 0) {
            continue;
        }
        double count = (double)metrics.bin_count[bin];
        double confidence = metrics.bin_confidence[bin] / count;
        double accuracy = metrics.bin_accuracy[bin] / count;
        metrics.ece += (count / sample_count) * fabs(confidence - accuracy);
        metrics.bin_confidence[bin] = confidence;
        metrics.bin_accuracy[bin] = accuracy;
    }
    return metrics;
}

static int train_claim_model(NeuralNet *net,
                             uint64_t seed_start,
                             size_t samples,
                             size_t epochs,
                             nn_real *candidate_inputs)
{
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        double loss = 0.0;
        for (size_t i = 0; i < samples; ++i) {
            ClaimState state = generate_claim_state(seed_start + (uint64_t)i);
            fill_claim_candidate_features(candidate_inputs, &state);
            nn_real step_loss = nn_train_action_scorer(net,
                                                       candidate_inputs,
                                                       DC_ACTION_COUNT,
                                                       state.target,
                                                       (nn_real)0.01);
            if (isnan((double)step_loss)) {
                return 0;
            }
            loss += (double)step_loss;
        }
        printf("delayed_claim epoch %zu loss=%.4f samples=%zu\n",
               epoch + 1,
               loss / (double)samples,
               samples);
    }
    return 1;
}

static void print_claim_metrics(const char *label, const char *hash, ClaimMetrics metrics)
{
    double samples = metrics.samples == 0 ? 1.0 : (double)metrics.samples;
    double saved = metrics.baseline_steps - metrics.proposal_steps;
    double saved_pct = metrics.baseline_steps <= 0.0 ? 0.0 : 100.0 * saved / metrics.baseline_steps;
    printf("[%s] hash=%s accuracy=%.2f%% nll=%.4f ece=%.4f decision_utility=%.4f "
           "proposal_steps=%.2f baseline_steps=%.2f saved=%.2f (%.1f%%) samples=%zu\n",
           label,
           hash == NULL ? "unknown" : hash,
           100.0 * (double)metrics.correct / samples,
           metrics.nll,
           metrics.ece,
           metrics.utility,
           metrics.proposal_steps,
           metrics.baseline_steps,
           saved,
           saved_pct,
           metrics.samples);
}

static int save_claim_checkpoint(NeuralNet *net,
                                 const char *path,
                                 const char *parent_hash,
                                 const char *notes,
                                 char out_hash[65])
{
    NNCheckpointMeta meta;
    memset(&meta, 0, sizeof(meta));
    meta.checkpoint_version = 1;
    meta.feature_encoder_version = 1;
    strcpy(meta.parent_model_hash, parent_hash == NULL ? "unknown" : parent_hash);
    strcpy(meta.training_trace_root, "delayed_claim");
    strcpy(meta.notes, notes);
    strcpy(meta.feature_set, "claim-state-action-v1");
    return nn_save_checkpoint(net, path, &meta) && nn_hash_checkpoint_hex(path, out_hash);
}

static uint64_t claim_state_hash(uint64_t seed)
{
    uint64_t hash = 1469598103934665603ull;
    hash ^= seed;
    hash *= 1099511628211ull;
    return hash;
}

static int emit_claim_trace(NeuralNet *net,
                            const char *path,
                            const char *model_hash,
                            uint64_t seed_start,
                            size_t samples,
                            nn_real *candidate_inputs,
                            nn_real *scores,
                            nn_real *probabilities)
{
    FILE *fp = fopen(path, "w");
    if (fp == NULL) {
        return 0;
    }

    uint32_t action_ids[DC_ACTION_COUNT];
    fill_claim_action_ids(action_ids);
    size_t ranked[DC_ACTION_COUNT];
    int ok = 1;
    for (size_t i = 0; ok && i < samples; ++i) {
        uint64_t seed = seed_start + (uint64_t)i;
        ClaimState state = generate_claim_state(seed);
        uint32_t target_action_id = claim_action_id_for_action(state.target);
        size_t target_index = claim_row_for_action_id(action_ids,
                                                      DC_ACTION_COUNT,
                                                      target_action_id);
        fill_claim_candidate_features(candidate_inputs, &state);
        ok = nn_score_action_rows(net,
                                  candidate_inputs,
                                  DC_ACTION_COUNT,
                                  scores,
                                  probabilities) &&
             nn_rank_action_scores(scores, DC_ACTION_COUNT, ranked);
        if (!ok) {
            break;
        }
        if (target_index == DC_ACTION_COUNT) {
            ok = 0;
            break;
        }
        size_t proposal_steps = DC_ACTION_COUNT;
        for (size_t rank = 0; rank < DC_ACTION_COUNT; ++rank) {
            if (action_ids[ranked[rank]] == target_action_id) {
                proposal_steps = rank + 1;
                break;
            }
        }
        size_t selected_action = DC_ACTION_COUNT;
        if (!claim_action_from_action_id(action_ids[ranked[0]], &selected_action)) {
            ok = 0;
            break;
        }

        char cost_json[128];
        char extra_json[128];
        snprintf(cost_json,
                 sizeof(cost_json),
                 "{\"decision_utility\":%.6f,\"proposal_steps\":%zu,"
                 "\"baseline_steps\":%zu}",
                 (double)state.utility[selected_action],
                 proposal_steps,
                 state.target + 1);
        snprintf(extra_json,
                 sizeof(extra_json),
                 "\"decision_utility\":%.6f,\"proposal_steps\":%zu,"
                 "\"baseline_steps\":%zu",
                 (double)state.utility[selected_action],
                 proposal_steps,
                 state.target + 1);

        NNTraceRecord record;
        memset(&record, 0, sizeof(record));
        record.task_name = "delayed_claim";
        record.model_hash_hex = model_hash == NULL ? "unknown" : model_hash;
        record.state_hash = claim_state_hash(seed);
        record.feature_encoder_version = 1;
        record.candidate_count = DC_ACTION_COUNT;
        record.action_ids = action_ids;
        record.scores = scores;
        record.probabilities = probabilities;
        record.selected_index = ranked[0];
        record.target_index = target_index;
        record.loss =
            (nn_real)(-log(dc_safe_probability((double)probabilities[target_index])));
        record.verifier_result =
            action_ids[ranked[0]] == target_action_id ? "correct" : "corrected";
        record.verifier_cost_json = cost_json;
        record.seed = seed;
        record.extra_json = extra_json;
        ok = nn_trace_write_jsonl(fp, &record);
    }

    if (fclose(fp) != 0) {
        ok = 0;
    }
    if (ok) {
        printf("wrote delayed_claim trace: %s rows=%zu\n", path, samples);
    }
    return ok;
}

typedef struct {
    uint64_t *seeds;
    uint32_t *target_action_ids;
    nn_real *reference_probabilities;
    size_t count;
    size_t capacity;
    char parent_hash[65];
} ClaimTrace;

typedef struct {
    const char *trace_path;
    const char *out_path;
    size_t epochs;
    nn_real learning_rate;
    nn_real hard_weight;
    nn_real soft_weight;
} ClaimReplayConfig;

static int claim_trace_append(ClaimTrace *trace,
                              uint64_t seed,
                              uint32_t target_action_id,
                              const nn_real *reference_probabilities)
{
    if (trace->count == trace->capacity) {
        size_t next_capacity = trace->capacity == 0 ? 1024 : trace->capacity * 2;
        uint64_t *next_seeds = calloc(next_capacity, sizeof(*next_seeds));
        uint32_t *next_target_action_ids =
            calloc(next_capacity, sizeof(*next_target_action_ids));
        nn_real *next_reference_probabilities =
            calloc(next_capacity * DC_ACTION_COUNT, sizeof(*next_reference_probabilities));
        if (next_seeds == NULL || next_target_action_ids == NULL ||
            next_reference_probabilities == NULL) {
            free(next_seeds);
            free(next_target_action_ids);
            free(next_reference_probabilities);
            return 0;
        }
        if (trace->count != 0) {
            memcpy(next_seeds, trace->seeds, trace->count * sizeof(*next_seeds));
            memcpy(next_target_action_ids,
                   trace->target_action_ids,
                   trace->count * sizeof(*next_target_action_ids));
            memcpy(next_reference_probabilities,
                   trace->reference_probabilities,
                   trace->count * DC_ACTION_COUNT * sizeof(*next_reference_probabilities));
        }
        free(trace->seeds);
        free(trace->target_action_ids);
        free(trace->reference_probabilities);
        trace->seeds = next_seeds;
        trace->target_action_ids = next_target_action_ids;
        trace->reference_probabilities = next_reference_probabilities;
        trace->capacity = next_capacity;
    }
    trace->seeds[trace->count] = seed;
    trace->target_action_ids[trace->count] = target_action_id;
    memcpy(&trace->reference_probabilities[trace->count * DC_ACTION_COUNT],
           reference_probabilities,
           DC_ACTION_COUNT * sizeof(*reference_probabilities));
    ++trace->count;
    return 1;
}

static void claim_trace_free(ClaimTrace *trace)
{
    free(trace->seeds);
    free(trace->target_action_ids);
    free(trace->reference_probabilities);
    memset(trace, 0, sizeof(*trace));
}

static int load_claim_trace(const char *path, ClaimTrace *trace);

static int delayed_claim_reordered_trace_self_test(void)
{
    const char *path = "/tmp/crlplrimes_delayed_claim_reordered_trace.jsonl";
    FILE *fp = fopen(path, "w");
    if (fp == NULL) {
        return 0;
    }

    fprintf(fp,
            "{\"schema\":\"crlp.trace.v1\",\"trace_version\":1,"
            "\"action_id_contract\":\"stable_unique_action_ids\",\"task\":\"delayed_claim\","
            "\"state_hash\":\"0000000000000001\",\"model_hash\":\"fixture\","
            "\"feature_encoder_version\":1,\"candidate_count\":4,"
            "\"action_ids\":[2,0,3,1],\"scores\":[2,0,-1,-2],"
            "\"probabilities\":[0.7,0.1,0.1,0.1],"
            "\"selected_index\":0,\"target_index\":0,\"loss\":0.1,"
            "\"verifier_result\":\"correct\",\"verifier_cost\":{},\"seed\":1000}\n");
    if (fclose(fp) != 0) {
        return 0;
    }

    ClaimTrace trace;
    int ok = load_claim_trace(path, &trace);
    if (ok) {
        uint32_t current_action_ids[DC_ACTION_COUNT];
        nn_real row_reference_probabilities[DC_ACTION_COUNT];
        fill_claim_action_ids(current_action_ids);
        for (size_t row = 0; row < DC_ACTION_COUNT; ++row) {
            size_t action = DC_ACTION_COUNT;
            ok = ok && claim_action_from_action_id(current_action_ids[row], &action);
            if (ok) {
                row_reference_probabilities[row] =
                    trace.reference_probabilities[action];
            }
        }
        ok = ok && trace.count == 1 && trace.target_action_ids[0] == 2 &&
             claim_row_for_action_id(current_action_ids,
                                     DC_ACTION_COUNT,
                                     trace.target_action_ids[0]) == 2 &&
             fabs((double)row_reference_probabilities[2] - 0.7) < 1e-6;
    }
    claim_trace_free(&trace);
    remove(path);
    printf("%s delayed_claim reordered trace action-id replay\n", ok ? "ok" : "FAIL");
    return ok;
}

static int load_claim_trace(const char *path, ClaimTrace *trace)
{
    memset(trace, 0, sizeof(*trace));
    strcpy(trace->parent_hash, "unknown");
    FILE *fp = fopen(path, "r");
    if (fp == NULL) {
        return 0;
    }

    char line[8192];
    uint32_t action_ids[DC_ACTION_COUNT];
    nn_real scores[DC_ACTION_COUNT];
    nn_real probabilities[DC_ACTION_COUNT];
    int ok = 1;
    while (ok && fgets(line, sizeof(line), fp) != NULL) {
        NNParsedTraceRecord record;
        if (!nn_trace_parse_jsonl(line,
                                  &record,
                                  action_ids,
                                  scores,
                                  probabilities,
                                  DC_ACTION_COUNT)) {
            ok = 0;
            break;
        }
        if (strcmp(record.task_name, "delayed_claim") != 0) {
            continue;
        }
        nn_real reference_probabilities[DC_ACTION_COUNT] = {0};
        uint8_t seen_actions[DC_ACTION_COUNT] = {0};
        for (size_t row = 0; row < record.candidate_count; ++row) {
            size_t action = DC_ACTION_COUNT;
            if (!claim_action_from_action_id(action_ids[row], &action) ||
                seen_actions[action]) {
                ok = 0;
                break;
            }
            seen_actions[action] = 1;
            reference_probabilities[action] = probabilities[row];
        }
        if (!ok || record.candidate_count != DC_ACTION_COUNT ||
            record.target_index >= DC_ACTION_COUNT) {
            ok = 0;
            break;
        }
        if (strcmp(trace->parent_hash, "unknown") == 0 &&
            strcmp(record.model_hash_hex, "unknown") != 0) {
            strcpy(trace->parent_hash, record.model_hash_hex);
        }
        ok = claim_trace_append(trace,
                                record.seed,
                                action_ids[record.target_index],
                                reference_probabilities);
    }
    if (fclose(fp) != 0) {
        ok = 0;
    }
    if (!ok || trace->count == 0) {
        claim_trace_free(trace);
        return 0;
    }
    return 1;
}

static int replay_claim_trace(const ClaimReplayConfig *config)
{
    ClaimTrace trace;
    if (!load_claim_trace(config->trace_path, &trace)) {
        fprintf(stderr, "failed to load trace: %s\n", config->trace_path);
        return 0;
    }

    NeuralNet *net = create_claim_model(9090);
    nn_real *candidate_inputs = calloc((size_t)DC_ACTION_COUNT * DC_FEATURE_COUNT,
                                       sizeof(*candidate_inputs));
    nn_real *scores = calloc(DC_ACTION_COUNT, sizeof(*scores));
    if (net == NULL || candidate_inputs == NULL || scores == NULL) {
        nn_free(net);
        free(candidate_inputs);
        free(scores);
        claim_trace_free(&trace);
        return 0;
    }

    uint32_t current_action_ids[DC_ACTION_COUNT];
    fill_claim_action_ids(current_action_ids);
    for (size_t epoch = 0; epoch < config->epochs; ++epoch) {
        double loss = 0.0;
        for (size_t i = 0; i < trace.count; ++i) {
            ClaimState state = generate_claim_state(trace.seeds[i]);
            size_t target_index = claim_row_for_action_id(current_action_ids,
                                                          DC_ACTION_COUNT,
                                                          trace.target_action_ids[i]);
            nn_real row_reference_probabilities[DC_ACTION_COUNT];

            if (target_index == DC_ACTION_COUNT) {
                nn_free(net);
                free(candidate_inputs);
                free(scores);
                claim_trace_free(&trace);
                return 0;
            }
            for (size_t row = 0; row < DC_ACTION_COUNT; ++row) {
                size_t action = DC_ACTION_COUNT;
                if (!claim_action_from_action_id(current_action_ids[row], &action)) {
                    nn_free(net);
                    free(candidate_inputs);
                    free(scores);
                    claim_trace_free(&trace);
                    return 0;
                }
                row_reference_probabilities[row] =
                    trace.reference_probabilities[i * DC_ACTION_COUNT + action];
            }

            fill_claim_candidate_features(candidate_inputs, &state);
            if (!nn_score_action_rows(net, candidate_inputs, DC_ACTION_COUNT, scores, NULL)) {
                nn_free(net);
                free(candidate_inputs);
                free(scores);
                claim_trace_free(&trace);
                return 0;
            }
            nn_real step_loss = nn_train_softmax_scorer_policy_regularized_cached(
                net,
                DC_ACTION_COUNT,
                target_index,
                config->hard_weight,
                row_reference_probabilities,
                config->soft_weight,
                (nn_real)0,
                config->learning_rate);
            if (isnan((double)step_loss)) {
                nn_free(net);
                free(candidate_inputs);
                free(scores);
                claim_trace_free(&trace);
                return 0;
            }
            loss += (double)step_loss;
        }
        printf("delayed_claim replay epoch %zu loss=%.4f rows=%zu\n",
               epoch + 1,
               loss / (double)trace.count,
               trace.count);
    }

    char hash[65];
    int ok = save_claim_checkpoint(net,
                                   config->out_path,
                                   trace.parent_hash,
                                   "delayed_claim successor trained from soft trace replay",
                                   hash);
    if (ok) {
        printf("saved delayed_claim replay checkpoint: %s hash=%s parent=%s "
               "epochs=%zu hard_weight=%.2f soft_weight=%.2f\n",
               config->out_path,
               hash,
               trace.parent_hash,
               config->epochs,
               (double)config->hard_weight,
               (double)config->soft_weight);
    }
    nn_free(net);
    free(candidate_inputs);
    free(scores);
    claim_trace_free(&trace);
    return ok;
}

static void print_claim_reliability_bins_json(const ClaimMetrics *metrics)
{
    fputs(",\"reliability_bins\":[", stdout);
    int first = 1;
    for (size_t bin = 0; bin < DC_CALIBRATION_BINS; ++bin) {
        if (metrics->bin_count[bin] == 0) {
            continue;
        }
        printf("%s{\"low\":%.2f,\"high\":%.2f,\"count\":%zu,"
               "\"confidence\":%.8f,\"accuracy\":%.8f}",
               first ? "" : ",",
               (double)bin / (double)DC_CALIBRATION_BINS,
               (double)(bin + 1) / (double)DC_CALIBRATION_BINS,
               metrics->bin_count[bin],
               metrics->bin_confidence[bin],
               metrics->bin_accuracy[bin]);
        first = 0;
    }
    fputc(']', stdout);
}

static int compare_claim_models(int argc, char **argv)
{
    nn_real *candidate_inputs = calloc((size_t)DC_ACTION_COUNT * DC_FEATURE_COUNT,
                                       sizeof(*candidate_inputs));
    nn_real *scores = calloc(DC_ACTION_COUNT, sizeof(*scores));
    nn_real *probabilities = calloc(DC_ACTION_COUNT, sizeof(*probabilities));
    if (candidate_inputs == NULL || scores == NULL || probabilities == NULL) {
        free(candidate_inputs);
        free(scores);
        free(probabilities);
        return 0;
    }

    int ok = 1;
    for (int i = 2; i < argc; ++i) {
        NNCheckpointMeta meta;
        NeuralNet *net = nn_load_checkpoint(argv[i], &meta);
        char hash[65] = "unknown";
        if (net == NULL) {
            fprintf(stderr, "failed to load model: %s\n", argv[i]);
            ok = 0;
            continue;
        }
        nn_hash_checkpoint_hex(argv[i], hash);
        ClaimMetrics metrics = evaluate_claim_model(net,
                                                    DC_TEST_SEED_START,
                                                    DC_TEST_SAMPLES,
                                                    candidate_inputs,
                                                    scores,
                                                    probabilities);
        printf("{\"schema\":\"crlp.compare.v1\",\"report_version\":1,\"task\":\"delayed_claim\","
               "\"model_hash\":\"%s\",\"parent_hash\":\"%s\","
               "\"feature_encoder_version\":%u,"
               "\"benchmark_seed\":%u,\"sample_count\":%zu,"
               "\"accuracy\":%.8f,\"nll\":%.8f,\"ece\":%.8f,"
               "\"decision_utility\":%.8f,\"proposal_steps\":%.8f,"
               "\"baseline_steps\":%.8f,\"search_saved\":%.8f",
               hash,
               meta.parent_model_hash[0] == '\0' ? "unknown" : meta.parent_model_hash,
               meta.feature_encoder_version,
               (unsigned int)DC_TEST_SEED_START,
               metrics.samples,
               metrics.samples == 0 ? 0.0 : (double)metrics.correct / (double)metrics.samples,
               metrics.nll,
               metrics.ece,
               metrics.utility,
               metrics.proposal_steps,
               metrics.baseline_steps,
               metrics.baseline_steps - metrics.proposal_steps);
        print_claim_reliability_bins_json(&metrics);
        printf("}\n");
        print_claim_metrics(argv[i], hash, metrics);
        nn_free(net);
    }

    free(candidate_inputs);
    free(scores);
    free(probabilities);
    return ok;
}

static void print_usage(FILE *stream)
{
    fputs("usage: delayed_claim [--trace path] [--save path] "
          "[--trace-train] [--trace-seed seed] [--trace-samples count]\n"
          "       delayed_claim --replay trace.jsonl --out model.nnckpt "
          "[--epochs count] [--hard-weight value] [--soft-weight value]\n"
          "       delayed_claim --compare model_a.nnckpt model_b.nnckpt [...]\n",
          stream);
}

int main(int argc, char **argv)
{
    if (argc == 2 && strcmp(argv[1], "--self-test-reordered-trace") == 0) {
        return delayed_claim_reordered_trace_self_test() ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    if (argc >= 2 && strcmp(argv[1], "--replay") == 0) {
        if (argc < 5) {
            print_usage(stderr);
            return EXIT_FAILURE;
        }

        ClaimReplayConfig replay_config;
        replay_config.trace_path = argv[2];
        replay_config.out_path = NULL;
        replay_config.epochs = DC_REPLAY_EPOCHS;
        replay_config.learning_rate = DC_REPLAY_LEARNING_RATE;
        replay_config.hard_weight = DC_REPLAY_HARD_WEIGHT;
        replay_config.soft_weight = DC_REPLAY_SOFT_WEIGHT;

        for (int i = 3; i < argc; ++i) {
            if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
                replay_config.out_path = argv[++i];
            } else if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) {
                if (!parse_size_text(argv[++i], &replay_config.epochs)) {
                    print_usage(stderr);
                    return EXIT_FAILURE;
                }
            } else if (strcmp(argv[i], "--learning-rate") == 0 && i + 1 < argc) {
                if (!parse_real_text(argv[++i], &replay_config.learning_rate) ||
                    replay_config.learning_rate <= (nn_real)0) {
                    print_usage(stderr);
                    return EXIT_FAILURE;
                }
            } else if (strcmp(argv[i], "--hard-weight") == 0 && i + 1 < argc) {
                if (!parse_real_text(argv[++i], &replay_config.hard_weight)) {
                    print_usage(stderr);
                    return EXIT_FAILURE;
                }
            } else if (strcmp(argv[i], "--soft-weight") == 0 && i + 1 < argc) {
                if (!parse_real_text(argv[++i], &replay_config.soft_weight)) {
                    print_usage(stderr);
                    return EXIT_FAILURE;
                }
            } else {
                print_usage(stderr);
                return EXIT_FAILURE;
            }
        }

        if (replay_config.out_path == NULL ||
            (replay_config.hard_weight == (nn_real)0 &&
             replay_config.soft_weight == (nn_real)0)) {
            print_usage(stderr);
            return EXIT_FAILURE;
        }
        return replay_claim_trace(&replay_config) ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    if (argc >= 2 && strcmp(argv[1], "--compare") == 0) {
        if (argc < 4) {
            print_usage(stderr);
            return EXIT_FAILURE;
        }
        return compare_claim_models(argc, argv) ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    const char *trace_path = NULL;
    const char *save_path = NULL;
    uint64_t trace_seed_start = DC_TEST_SEED_START;
    size_t trace_samples = DC_TEST_SAMPLES;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--trace") == 0 && i + 1 < argc) {
            trace_path = argv[++i];
        } else if (strcmp(argv[i], "--save") == 0 && i + 1 < argc) {
            save_path = argv[++i];
        } else if (strcmp(argv[i], "--trace-train") == 0) {
            trace_seed_start = DC_TRAIN_SEED_START;
            trace_samples = DC_TRAIN_SAMPLES;
        } else if (strcmp(argv[i], "--trace-seed") == 0 && i + 1 < argc) {
            if (!parse_u64_text(argv[++i], &trace_seed_start)) {
                print_usage(stderr);
                return EXIT_FAILURE;
            }
        } else if (strcmp(argv[i], "--trace-samples") == 0 && i + 1 < argc) {
            if (!parse_size_text(argv[++i], &trace_samples)) {
                print_usage(stderr);
                return EXIT_FAILURE;
            }
        } else {
            print_usage(stderr);
            return EXIT_FAILURE;
        }
    }

    NeuralNet *net = create_claim_model(5150);
    nn_real *candidate_inputs = calloc((size_t)DC_ACTION_COUNT * DC_FEATURE_COUNT,
                                       sizeof(*candidate_inputs));
    nn_real *scores = calloc(DC_ACTION_COUNT, sizeof(*scores));
    nn_real *probabilities = calloc(DC_ACTION_COUNT, sizeof(*probabilities));
    if (net == NULL || candidate_inputs == NULL || scores == NULL || probabilities == NULL) {
        fputs("failed to allocate delayed_claim model\n", stderr);
        nn_free(net);
        free(candidate_inputs);
        free(scores);
        free(probabilities);
        return EXIT_FAILURE;
    }

    int ok = train_claim_model(net, DC_TRAIN_SEED_START, DC_TRAIN_SAMPLES, 4, candidate_inputs);
    ClaimMetrics metrics = evaluate_claim_model(net,
                                                DC_TEST_SEED_START,
                                                DC_TEST_SAMPLES,
                                                candidate_inputs,
                                                scores,
                                                probabilities);
    char hash[65] = "unknown";
    if (ok && save_path != NULL) {
        ok = save_claim_checkpoint(net,
                                   save_path,
                                   "unknown",
                                   "delayed_claim trained checkpoint",
                                   hash);
        if (ok) {
            printf("saved delayed_claim checkpoint: %s hash=%s\n", save_path, hash);
        }
    }
    print_claim_metrics("delayed_claim", hash, metrics);

    if (ok && trace_path != NULL) {
        ok = emit_claim_trace(net,
                              trace_path,
                              hash,
                              trace_seed_start,
                              trace_samples,
                              candidate_inputs,
                              scores,
                              probabilities);
    }

    nn_free(net);
    free(candidate_inputs);
    free(scores);
    free(probabilities);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
