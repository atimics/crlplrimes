#include "nn.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    TILE_WIDTH = 3,
    TILE_CELLS = TILE_WIDTH * TILE_WIDTH,
    TILE_ROWS = 1 << TILE_CELLS,
    LIFE_FEATURE_BUCKETS = 18,
    INVERSE_EPOCHS = 12000,
    INVERSE_RETRIES = 4,
};

typedef struct {
    NeuralNet *net;
    size_t exact_canonical;
    size_t verified_preimage;
    double bit_accuracy;
    double loss;
} InverseReport;

static uint32_t popcount_u64(uint64_t value)
{
    uint32_t count = 0u;
    while (value != 0u) {
        count += (uint32_t)(value & UINT64_C(1));
        value >>= 1u;
    }
    return count;
}

static uint8_t tile_bit(uint16_t mask, int x, int y)
{
    return (uint8_t)((mask >> (unsigned)(y * TILE_WIDTH + x)) & 1u);
}

static uint8_t life_tile_next(uint16_t mask)
{
    uint8_t center = tile_bit(mask, 1, 1);
    uint32_t neighbors = popcount_u64(mask) - center;
    return neighbors == 3u || (center != 0u && neighbors == 2u) ? 1u : 0u;
}

static size_t life_feature_bucket(uint16_t mask)
{
    uint8_t center = tile_bit(mask, 1, 1);
    uint32_t neighbors = popcount_u64(mask) - center;
    return (size_t)(center * 9u + neighbors);
}

static double entropy_binary(size_t a, size_t b)
{
    size_t total = a + b;
    double entropy = 0.0;
    if (total == 0u) {
        return 0.0;
    }
    if (a != 0u) {
        double p = (double)a / (double)total;
        entropy -= p * (log(p) / log(2.0));
    }
    if (b != 0u) {
        double p = (double)b / (double)total;
        entropy -= p * (log(p) / log(2.0));
    }
    return entropy;
}

static void report_life_feature_entropy(void)
{
    size_t target_counts[2] = {0u, 0u};
    size_t bucket_counts[LIFE_FEATURE_BUCKETS][2];
    size_t bucket_totals[LIFE_FEATURE_BUCKETS];
    double conditional_entropy = 0.0;
    double feature_entropy = 0.0;

    memset(bucket_counts, 0, sizeof(bucket_counts));
    memset(bucket_totals, 0, sizeof(bucket_totals));

    for (uint16_t mask = 0u; mask < TILE_ROWS; ++mask) {
        uint8_t y = life_tile_next(mask);
        size_t bucket = life_feature_bucket(mask);
        ++target_counts[y ? 1u : 0u];
        ++bucket_counts[bucket][y ? 1u : 0u];
        ++bucket_totals[bucket];
    }

    double target_entropy = entropy_binary(target_counts[0], target_counts[1]);
    for (size_t bucket = 0u; bucket < LIFE_FEATURE_BUCKETS; ++bucket) {
        if (bucket_totals[bucket] == 0u) {
            continue;
        }
        double p = (double)bucket_totals[bucket] / (double)TILE_ROWS;
        feature_entropy -= p * (log(p) / log(2.0));
        conditional_entropy +=
            p * entropy_binary(bucket_counts[bucket][0], bucket_counts[bucket][1]);
    }

    printf("life_forward_entropy rows=%d positives=%zu/%d H(Y)=%.6f "
           "feature=center+neighbor_count buckets=%d H(F)=%.6f H(Y|F)=%.6f "
           "IG=%.6f IG/H(F)=%.6f\n",
           TILE_ROWS,
           target_counts[1],
           TILE_ROWS,
           target_entropy,
           LIFE_FEATURE_BUCKETS,
           feature_entropy,
           conditional_entropy,
           target_entropy - conditional_entropy,
           feature_entropy > 0.0 ? (target_entropy - conditional_entropy) / feature_entropy :
                                   0.0);
}

static int train_life_feature_bot(void)
{
    nn_real inputs[LIFE_FEATURE_BUCKETS * LIFE_FEATURE_BUCKETS];
    nn_real targets[LIFE_FEATURE_BUCKETS];
    size_t layers[] = {LIFE_FEATURE_BUCKETS, 1u};
    NNConfig config;
    config.layer_sizes = layers;
    config.layer_count = sizeof(layers) / sizeof(layers[0]);
    config.seed = 31091u;
    config.hidden_activation = NN_ACTIVATION_TANH;
    config.output_activation = NN_ACTIVATION_SIGMOID;
    config.loss = NN_LOSS_MSE;

    memset(inputs, 0, sizeof(inputs));
    memset(targets, 0, sizeof(targets));
    for (uint16_t mask = 0u; mask < TILE_ROWS; ++mask) {
        size_t bucket = life_feature_bucket(mask);
        inputs[bucket * LIFE_FEATURE_BUCKETS + bucket] = (nn_real)1;
        targets[bucket] = life_tile_next(mask) ? (nn_real)1 : (nn_real)0;
    }

    NeuralNet *net = nn_create_config(&config);
    if (net == NULL) {
        return 0;
    }

    (void)nn_train_batched(net,
                           inputs,
                           targets,
                           LIFE_FEATURE_BUCKETS,
                           LIFE_FEATURE_BUCKETS,
                           (nn_real)0.7,
                           20000u,
                           1);

    size_t correct = 0u;
    double min_margin = 1.0;
    nn_real row[LIFE_FEATURE_BUCKETS];
    for (uint16_t mask = 0u; mask < TILE_ROWS; ++mask) {
        size_t bucket = life_feature_bucket(mask);
        nn_real output = 0;
        memset(row, 0, sizeof(row));
        row[bucket] = (nn_real)1;
        nn_forward(net, row, &output);
        uint8_t expected = life_tile_next(mask);
        uint8_t predicted = output >= (nn_real)0.5 ? 1u : 0u;
        double margin = fabs((double)output - 0.5);
        if (margin < min_margin) {
            min_margin = margin;
        }
        correct += predicted == expected ? 1u : 0u;
    }

    printf("life_feature_bot inputs=%d parameters=%zu certified_rows=%zu/%d "
           "min_margin=%.6f\n",
           LIFE_FEATURE_BUCKETS,
           nn_parameter_count(net),
           correct,
           TILE_ROWS,
           min_margin);
    nn_free(net);
    return correct == TILE_ROWS;
}

static void print_preimage_stats(int n)
{
    NNLifePreimageStats stats;
    if (!nn_life_preimage_stats_torus((size_t)n, &stats)) {
        printf("preimage_stats grid=%dx%d allocation_failed\n", n, n);
        return;
    }
    printf("preimage_stats grid=%dx%d torus states=%zu reachable=%zu garden=%zu "
           "ambiguous=%zu singleton=%zu avg_preimages=%.3f max_preimages=%u "
           "H(prior|next)=%.3f H(next)=%.3f\n",
           n,
           n,
           stats.state_count,
           stats.reachable,
           stats.garden_of_eden,
           stats.ambiguous_targets,
           stats.singleton_targets,
           stats.avg_preimages_reachable,
           stats.max_preimages,
           stats.inverse_entropy_bits,
           stats.image_entropy_bits);
}

static size_t build_inverse_dataset_3x3(const uint32_t *counts,
                                        const uint64_t *canonical,
                                        nn_real **inputs_out,
                                        nn_real **targets_out)
{
    const int n = 3;
    const size_t cells = (size_t)n * (size_t)n;
    const size_t states = (size_t)UINT64_C(1) << (unsigned)cells;
    size_t samples = 0u;
    for (size_t target = 0u; target < states; ++target) {
        samples += counts[target] != 0u ? 1u : 0u;
    }

    nn_real *inputs = calloc(samples * cells, sizeof(*inputs));
    nn_real *targets = calloc(samples * cells, sizeof(*targets));
    if (inputs == NULL || targets == NULL) {
        free(inputs);
        free(targets);
        return 0u;
    }

    size_t row = 0u;
    for (size_t target = 0u; target < states; ++target) {
        if (counts[target] == 0u) {
            continue;
        }
        uint64_t prior = canonical[target];
        for (size_t bit = 0u; bit < cells; ++bit) {
            inputs[row * cells + bit] =
                ((target >> bit) & 1u) != 0u ? (nn_real)1 : (nn_real)0;
            targets[row * cells + bit] =
                ((prior >> bit) & UINT64_C(1)) != 0u ? (nn_real)1 : (nn_real)0;
        }
        ++row;
    }

    *inputs_out = inputs;
    *targets_out = targets;
    return samples;
}

static InverseReport train_inverse_3x3(void)
{
    const int n = 3;
    const size_t cells = (size_t)n * (size_t)n;
    uint32_t *counts = NULL;
    uint64_t *canonical = NULL;
    nn_real *inputs = NULL;
    nn_real *targets = NULL;
    size_t samples = 0u;
    InverseReport best;
    memset(&best, 0, sizeof(best));
    best.loss = 1e9;

    if (!nn_life_preimage_tables_torus((size_t)n, &counts, &canonical)) {
        return best;
    }
    samples = build_inverse_dataset_3x3(counts, canonical, &inputs, &targets);
    if (samples == 0u) {
        free(counts);
        free(canonical);
        return best;
    }

    for (size_t attempt = 0u; attempt < INVERSE_RETRIES; ++attempt) {
        size_t layers[] = {cells, 64u, cells};
        NNConfig config;
        config.layer_sizes = layers;
        config.layer_count = sizeof(layers) / sizeof(layers[0]);
        config.seed = 41011u + (unsigned int)(attempt * 409u);
        config.hidden_activation = NN_ACTIVATION_TANH;
        config.output_activation = NN_ACTIVATION_SIGMOID;
        config.loss = NN_LOSS_MSE;
        NeuralNet *net = nn_create_config(&config);
        if (net == NULL) {
            continue;
        }

        (void)nn_train_batched(net,
                               inputs,
                               targets,
                               samples,
                               samples,
                               (nn_real)0.08,
                               INVERSE_EPOCHS,
                               1);

        InverseReport report;
        memset(&report, 0, sizeof(report));
        report.net = net;
        size_t correct_bits = 0u;
        nn_real output[9];
        size_t row = 0u;
        for (size_t target = 0u; target < ((size_t)UINT64_C(1) << (unsigned)cells); ++target) {
            if (counts[target] == 0u) {
                continue;
            }
            nn_forward(net, &inputs[row * cells], output);
            uint64_t predicted = 0u;
            for (size_t bit = 0u; bit < cells; ++bit) {
                uint8_t b = output[bit] >= (nn_real)0.5 ? 1u : 0u;
                uint8_t expected = ((canonical[target] >> bit) & UINT64_C(1)) != 0u ? 1u : 0u;
                predicted |= b != 0u ? UINT64_C(1) << (unsigned)bit : 0u;
                correct_bits += b == expected ? 1u : 0u;
                double error = (double)output[bit] - (double)expected;
                report.loss += 0.5 * error * error;
            }
            report.exact_canonical += predicted == canonical[target] ? 1u : 0u;
            report.verified_preimage +=
                nn_life_step_torus((size_t)n, predicted) == (uint64_t)target ? 1u : 0u;
            ++row;
        }

        report.loss /= (double)(samples * cells);
        report.bit_accuracy = (double)correct_bits / (double)(samples * cells);
        if (report.verified_preimage > best.verified_preimage ||
            (report.verified_preimage == best.verified_preimage &&
             report.exact_canonical > best.exact_canonical) ||
            (report.verified_preimage == best.verified_preimage &&
             report.exact_canonical == best.exact_canonical && report.loss < best.loss)) {
            nn_free(best.net);
            best = report;
        } else {
            nn_free(net);
        }
    }

    printf("inverse_predictor grid=3x3 torus samples=%zu parameters=%zu "
           "canonical_exact=%zu/%zu verified_preimage=%zu/%zu bit_accuracy=%.6f "
           "loss=%.8f\n",
           samples,
           best.net == NULL ? 0u : nn_parameter_count(best.net),
           best.exact_canonical,
           samples,
           best.verified_preimage,
           samples,
           best.bit_accuracy,
           best.loss);

    free(inputs);
    free(targets);
    free(counts);
    free(canonical);
    return best;
}

int main(void)
{
    printf("life_curriculum tile=3x3\n");
    report_life_feature_entropy();
    if (!train_life_feature_bot()) {
        return EXIT_FAILURE;
    }

    print_preimage_stats(3);
    print_preimage_stats(4);

    InverseReport inverse = train_inverse_3x3();
    int ok = inverse.net != NULL && inverse.verified_preimage > 0u;
    nn_free(inverse.net);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
