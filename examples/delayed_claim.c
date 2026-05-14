#include "nn.h"

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
    DC_CALIBRATION_BINS = 10,
};

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
        ++metrics.bin_count[bin];
        metrics.bin_confidence[bin] += confidence;
        metrics.bin_accuracy[bin] += selected == state.target ? 1.0 : 0.0;
    }

    double sample_count = metrics.samples == 0 ? 1.0 : (double)metrics.samples;
    metrics.nll /= sample_count;
    metrics.utility /= sample_count;
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
    printf("[%s] hash=%s accuracy=%.2f%% nll=%.4f ece=%.4f decision_utility=%.4f samples=%zu\n",
           label,
           hash == NULL ? "unknown" : hash,
           100.0 * (double)metrics.correct / samples,
           metrics.nll,
           metrics.ece,
           metrics.utility,
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

    uint32_t action_ids[DC_ACTION_COUNT] = {0, 1, 2, 3};
    size_t ranked[DC_ACTION_COUNT];
    int ok = 1;
    for (size_t i = 0; ok && i < samples; ++i) {
        uint64_t seed = seed_start + (uint64_t)i;
        ClaimState state = generate_claim_state(seed);
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

        char cost_json[128];
        char extra_json[128];
        snprintf(cost_json,
                 sizeof(cost_json),
                 "{\"decision_utility\":%.6f}",
                 (double)state.utility[ranked[0]]);
        snprintf(extra_json, sizeof(extra_json), "\"decision_utility\":%.6f", (double)state.utility[ranked[0]]);

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
        record.target_index = state.target;
        record.loss = (nn_real)(-log(dc_safe_probability((double)probabilities[state.target])));
        record.verifier_result = ranked[0] == state.target ? "correct" : "corrected";
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
    size_t *targets;
    size_t count;
    size_t capacity;
    char parent_hash[65];
} ClaimTrace;

static int claim_trace_append(ClaimTrace *trace, uint64_t seed, size_t target)
{
    if (trace->count == trace->capacity) {
        size_t next_capacity = trace->capacity == 0 ? 1024 : trace->capacity * 2;
        uint64_t *next_seeds = calloc(next_capacity, sizeof(*next_seeds));
        size_t *next_targets = calloc(next_capacity, sizeof(*next_targets));
        if (next_seeds == NULL || next_targets == NULL) {
            free(next_seeds);
            free(next_targets);
            return 0;
        }
        if (trace->count != 0) {
            memcpy(next_seeds, trace->seeds, trace->count * sizeof(*next_seeds));
            memcpy(next_targets, trace->targets, trace->count * sizeof(*next_targets));
        }
        free(trace->seeds);
        free(trace->targets);
        trace->seeds = next_seeds;
        trace->targets = next_targets;
        trace->capacity = next_capacity;
    }
    trace->seeds[trace->count] = seed;
    trace->targets[trace->count] = target;
    ++trace->count;
    return 1;
}

static void claim_trace_free(ClaimTrace *trace)
{
    free(trace->seeds);
    free(trace->targets);
    memset(trace, 0, sizeof(*trace));
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
        if (record.candidate_count != DC_ACTION_COUNT || record.target_index >= DC_ACTION_COUNT) {
            ok = 0;
            break;
        }
        if (strcmp(trace->parent_hash, "unknown") == 0 &&
            strcmp(record.model_hash_hex, "unknown") != 0) {
            strcpy(trace->parent_hash, record.model_hash_hex);
        }
        ok = claim_trace_append(trace, record.seed, record.target_index);
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

static int replay_claim_trace(const char *trace_path, const char *out_path)
{
    ClaimTrace trace;
    if (!load_claim_trace(trace_path, &trace)) {
        fprintf(stderr, "failed to load trace: %s\n", trace_path);
        return 0;
    }

    NeuralNet *net = create_claim_model(9090);
    nn_real *candidate_inputs = calloc((size_t)DC_ACTION_COUNT * DC_FEATURE_COUNT,
                                       sizeof(*candidate_inputs));
    if (net == NULL || candidate_inputs == NULL) {
        nn_free(net);
        free(candidate_inputs);
        claim_trace_free(&trace);
        return 0;
    }

    for (size_t epoch = 0; epoch < 3; ++epoch) {
        double loss = 0.0;
        for (size_t i = 0; i < trace.count; ++i) {
            ClaimState state = generate_claim_state(trace.seeds[i]);
            fill_claim_candidate_features(candidate_inputs, &state);
            nn_real step_loss = nn_train_action_scorer(net,
                                                       candidate_inputs,
                                                       DC_ACTION_COUNT,
                                                       trace.targets[i],
                                                       (nn_real)0.01);
            if (isnan((double)step_loss)) {
                nn_free(net);
                free(candidate_inputs);
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
                                   out_path,
                                   trace.parent_hash,
                                   "delayed_claim successor trained from trace replay",
                                   hash);
    if (ok) {
        printf("saved delayed_claim replay checkpoint: %s hash=%s parent=%s\n",
               out_path,
               hash,
               trace.parent_hash);
    }
    nn_free(net);
    free(candidate_inputs);
    claim_trace_free(&trace);
    return ok;
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
                                                    900000,
                                                    DC_TEST_SAMPLES,
                                                    candidate_inputs,
                                                    scores,
                                                    probabilities);
        printf("{\"schema\":\"crlp.compare.v1\",\"report_version\":1,\"task\":\"delayed_claim\","
               "\"model_hash\":\"%s\",\"parent_hash\":\"%s\","
               "\"feature_encoder_version\":%u,"
               "\"benchmark_seed\":900000,\"sample_count\":%zu,"
               "\"accuracy\":%.8f,\"nll\":%.8f,\"ece\":%.8f,"
               "\"decision_utility\":%.8f}\n",
               hash,
               meta.parent_model_hash[0] == '\0' ? "unknown" : meta.parent_model_hash,
               meta.feature_encoder_version,
               metrics.samples,
               metrics.samples == 0 ? 0.0 : (double)metrics.correct / (double)metrics.samples,
               metrics.nll,
               metrics.ece,
               metrics.utility);
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
    fputs("usage: delayed_claim [--trace path] [--save path]\n"
          "       delayed_claim --replay trace.jsonl --out model.nnckpt\n"
          "       delayed_claim --compare model_a.nnckpt model_b.nnckpt [...]\n",
          stream);
}

int main(int argc, char **argv)
{
    if (argc >= 2 && strcmp(argv[1], "--replay") == 0) {
        if (argc < 5 || strcmp(argv[3], "--out") != 0) {
            print_usage(stderr);
            return EXIT_FAILURE;
        }
        return replay_claim_trace(argv[2], argv[4]) ? EXIT_SUCCESS : EXIT_FAILURE;
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
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--trace") == 0 && i + 1 < argc) {
            trace_path = argv[++i];
        } else if (strcmp(argv[i], "--save") == 0 && i + 1 < argc) {
            save_path = argv[++i];
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

    int ok = train_claim_model(net, 1000, DC_TRAIN_SAMPLES, 4, candidate_inputs);
    ClaimMetrics metrics = evaluate_claim_model(net,
                                                900000,
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
                              900000,
                              DC_TEST_SAMPLES,
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
