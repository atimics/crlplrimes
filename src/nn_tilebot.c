#include "nn_tilebot.h"
#include "nn_train.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    TB_D4_ORBIT_COUNT = 102u,
};

static int tb_finite(nn_real value)
{
    return isfinite((double)value);
}

static uint32_t tb_rng_next(uint32_t *state)
{
    *state = (*state * 1664525u) + 1013904223u;
    return *state;
}

static uint32_t tb_popcount_u64(uint64_t value)
{
    uint32_t count = 0u;
    while (value != 0u) {
        count += (uint32_t)(value & UINT64_C(1));
        value >>= 1u;
    }
    return count;
}

static uint8_t tb_bit_at(uint16_t mask, int x, int y)
{
    return (uint8_t)((mask >> (unsigned)(y * (int)NN_TILEBOT_WIDTH + x)) & 1u);
}

static unsigned tb_corner_count(uint16_t mask)
{
    return tb_bit_at(mask, 0, 0) + tb_bit_at(mask, 2, 0) + tb_bit_at(mask, 0, 2) +
           tb_bit_at(mask, 2, 2);
}

static unsigned tb_edge_count(uint16_t mask)
{
    return tb_bit_at(mask, 1, 0) + tb_bit_at(mask, 0, 1) + tb_bit_at(mask, 2, 1) +
           tb_bit_at(mask, 1, 2);
}

static uint16_t tb_transform_mask(uint16_t mask, unsigned transform)
{
    uint16_t out = 0u;
    for (int y = 0; y < (int)NN_TILEBOT_WIDTH; ++y) {
        for (int x = 0; x < (int)NN_TILEBOT_WIDTH; ++x) {
            int tx = x;
            int ty = y;
            switch (transform) {
            case 0:
                tx = x;
                ty = y;
                break;
            case 1:
                tx = (int)NN_TILEBOT_WIDTH - 1 - y;
                ty = x;
                break;
            case 2:
                tx = (int)NN_TILEBOT_WIDTH - 1 - x;
                ty = (int)NN_TILEBOT_WIDTH - 1 - y;
                break;
            case 3:
                tx = y;
                ty = (int)NN_TILEBOT_WIDTH - 1 - x;
                break;
            case 4:
                tx = (int)NN_TILEBOT_WIDTH - 1 - x;
                ty = y;
                break;
            case 5:
                tx = x;
                ty = (int)NN_TILEBOT_WIDTH - 1 - y;
                break;
            case 6:
                tx = y;
                ty = x;
                break;
            default:
                tx = (int)NN_TILEBOT_WIDTH - 1 - y;
                ty = (int)NN_TILEBOT_WIDTH - 1 - x;
                break;
            }
            if (tb_bit_at(mask, x, y) != 0u) {
                out |= (uint16_t)(1u << (unsigned)(ty * (int)NN_TILEBOT_WIDTH + tx));
            }
        }
    }
    return out;
}

static uint16_t tb_d4_canonical(uint16_t mask)
{
    uint16_t best = tb_transform_mask(mask, 0u);
    for (unsigned i = 1u; i < 8u; ++i) {
        uint16_t candidate = tb_transform_mask(mask, i);
        if (candidate < best) {
            best = candidate;
        }
    }
    return best;
}

static size_t tb_d4_orbit_index(uint16_t mask)
{
    uint16_t canonical = tb_d4_canonical(mask);
    size_t index = 0u;
    for (uint16_t candidate = 0u; candidate < NN_TILEBOT_ROWS; ++candidate) {
        if (tb_d4_canonical(candidate) != candidate) {
            continue;
        }
        if (candidate == canonical) {
            return index;
        }
        ++index;
    }
    return NN_TILEBOT_MAX_BUCKETS;
}

static double tb_entropy_binary(size_t zero_count, size_t one_count)
{
    size_t total = zero_count + one_count;
    double entropy = 0.0;
    if (total == 0u) {
        return 0.0;
    }
    if (zero_count != 0u) {
        double p = (double)zero_count / (double)total;
        entropy -= p * (log(p) / log(2.0));
    }
    if (one_count != 0u) {
        double p = (double)one_count / (double)total;
        entropy -= p * (log(p) / log(2.0));
    }
    return entropy;
}

static size_t tb_feature_input_width(NNTileFeature feature)
{
    switch (feature) {
    case NN_TILE_FEATURE_RAW_BITS:
        return NN_TILEBOT_CELLS;
    case NN_TILE_FEATURE_CENTER:
        return 2u;
    case NN_TILE_FEATURE_POPCOUNT:
        return NN_TILEBOT_CELLS + 1u;
    case NN_TILE_FEATURE_BOUNDARY_COUNT:
        return NN_TILEBOT_CELLS;
    case NN_TILE_FEATURE_CORNER_COUNT:
    case NN_TILE_FEATURE_EDGE_COUNT:
        return 5u;
    case NN_TILE_FEATURE_CORNER_EDGE_CENTER:
        return 50u;
    case NN_TILE_FEATURE_LIFE_CENTER_NEIGHBOR_COUNT:
        return 18u;
    case NN_TILE_FEATURE_D4_ORBIT:
        return TB_D4_ORBIT_COUNT;
    case NN_TILE_FEATURE_RAW_MASK:
    default:
        return NN_TILEBOT_ROWS;
    }
}

static int tb_feature_input_width_is_valid(size_t width)
{
    return width > 0u && width <= NN_TILEBOT_MAX_BUCKETS;
}

const char *nn_tile_rule_name(NNTileRule rule)
{
    switch (rule) {
    case NN_TILE_RULE_DILATE:
        return "dilate_any";
    case NN_TILE_RULE_MAJORITY:
        return "majority5";
    case NN_TILE_RULE_ERODE:
        return "erode_all";
    case NN_TILE_RULE_PARITY:
        return "parity9";
    case NN_TILE_RULE_COPY_CENTER:
        return "copy_center";
    case NN_TILE_RULE_EDGE_ANY:
        return "edge_any";
    case NN_TILE_RULE_CORNER_PARITY:
        return "corner_parity";
    case NN_TILE_RULE_LIFE:
        return "life";
    default:
        return "unknown";
    }
}

const char *nn_tile_feature_name(NNTileFeature feature)
{
    switch (feature) {
    case NN_TILE_FEATURE_RAW_BITS:
        return "raw_bits";
    case NN_TILE_FEATURE_RAW_MASK:
        return "raw_mask";
    case NN_TILE_FEATURE_POPCOUNT:
        return "popcount";
    case NN_TILE_FEATURE_CENTER:
        return "center";
    case NN_TILE_FEATURE_BOUNDARY_COUNT:
        return "boundary_count";
    case NN_TILE_FEATURE_CORNER_COUNT:
        return "corner_count";
    case NN_TILE_FEATURE_EDGE_COUNT:
        return "edge_count";
    case NN_TILE_FEATURE_CORNER_EDGE_CENTER:
        return "corner_edge_center";
    case NN_TILE_FEATURE_D4_ORBIT:
        return "d4_orbit";
    case NN_TILE_FEATURE_LIFE_CENTER_NEIGHBOR_COUNT:
        return "center_neighbor_count";
    default:
        return "unknown";
    }
}

size_t nn_tile_feature_bucket_count(NNTileFeature feature)
{
    switch (feature) {
    case NN_TILE_FEATURE_RAW_BITS:
    case NN_TILE_FEATURE_RAW_MASK:
        return NN_TILEBOT_ROWS;
    case NN_TILE_FEATURE_POPCOUNT:
        return NN_TILEBOT_CELLS + 1u;
    case NN_TILE_FEATURE_CENTER:
        return 2u;
    case NN_TILE_FEATURE_BOUNDARY_COUNT:
        return NN_TILEBOT_CELLS;
    case NN_TILE_FEATURE_CORNER_COUNT:
    case NN_TILE_FEATURE_EDGE_COUNT:
        return 5u;
    case NN_TILE_FEATURE_CORNER_EDGE_CENTER:
        return 50u;
    case NN_TILE_FEATURE_D4_ORBIT:
        return TB_D4_ORBIT_COUNT;
    case NN_TILE_FEATURE_LIFE_CENTER_NEIGHBOR_COUNT:
        return 18u;
    default:
        return 0u;
    }
}

uint8_t nn_tile_rule_eval(NNTileRule rule, uint16_t mask)
{
    uint32_t count = tb_popcount_u64((uint64_t)(mask & 0x01ffu));
    uint8_t center = tb_bit_at(mask, 1, 1);
    uint32_t neighbors = count - center;
    switch (rule) {
    case NN_TILE_RULE_DILATE:
        return count > 0u ? 1u : 0u;
    case NN_TILE_RULE_MAJORITY:
        return count >= 5u ? 1u : 0u;
    case NN_TILE_RULE_ERODE:
        return count == NN_TILEBOT_CELLS ? 1u : 0u;
    case NN_TILE_RULE_PARITY:
        return (uint8_t)(count & 1u);
    case NN_TILE_RULE_COPY_CENTER:
        return center;
    case NN_TILE_RULE_EDGE_ANY:
        return tb_edge_count(mask) > 0u ? 1u : 0u;
    case NN_TILE_RULE_CORNER_PARITY:
        return (uint8_t)(tb_corner_count(mask) & 1u);
    case NN_TILE_RULE_LIFE:
        return neighbors == 3u || (center != 0u && neighbors == 2u) ? 1u : 0u;
    default:
        return 0u;
    }
}

size_t nn_tile_feature_bucket(NNTileFeature feature, uint16_t mask)
{
    unsigned corners = tb_corner_count(mask);
    unsigned edges = tb_edge_count(mask);
    unsigned center = tb_bit_at(mask, 1, 1);
    switch (feature) {
    case NN_TILE_FEATURE_POPCOUNT:
        return tb_popcount_u64(mask & 0x01ffu);
    case NN_TILE_FEATURE_CENTER:
        return center;
    case NN_TILE_FEATURE_BOUNDARY_COUNT:
        return corners + edges;
    case NN_TILE_FEATURE_CORNER_COUNT:
        return corners;
    case NN_TILE_FEATURE_EDGE_COUNT:
        return edges;
    case NN_TILE_FEATURE_CORNER_EDGE_CENTER:
        return (size_t)(center * 25u + corners * 5u + edges);
    case NN_TILE_FEATURE_D4_ORBIT:
        return tb_d4_orbit_index(mask);
    case NN_TILE_FEATURE_LIFE_CENTER_NEIGHBOR_COUNT:
        return (size_t)(center * 9u + (tb_popcount_u64(mask & 0x01ffu) - center));
    case NN_TILE_FEATURE_RAW_BITS:
    case NN_TILE_FEATURE_RAW_MASK:
    default:
        return (size_t)(mask & 0x01ffu);
    }
}

int nn_tile_feature_vector(NNTileFeature feature, uint16_t mask, nn_real *out, size_t capacity)
{
    if (out == NULL) {
        return 0;
    }
    size_t width = tb_feature_input_width(feature);
    if (!tb_feature_input_width_is_valid(width) || capacity < width) {
        return 0;
    }
    memset(out, 0, width * sizeof(*out));
    if (feature == NN_TILE_FEATURE_RAW_BITS) {
        for (size_t bit = 0u; bit < NN_TILEBOT_CELLS; ++bit) {
            out[bit] = ((mask >> bit) & 1u) != 0u ? (nn_real)1 : (nn_real)0;
        }
        return 1;
    }

    size_t bucket = nn_tile_feature_bucket(feature, mask);
    if (bucket >= width) {
        return 0;
    }
    out[bucket] = (nn_real)1;
    return 1;
}

int nn_tile_feature_score(NNTileRule rule,
                          NNTileFeature feature,
                          NNTileFeatureScore *out_score)
{
    if (out_score == NULL) {
        return 0;
    }

    size_t counts[NN_TILEBOT_MAX_BUCKETS][2];
    size_t bucket_totals[NN_TILEBOT_MAX_BUCKETS];
    size_t target_counts[2] = {0u, 0u};
    memset(counts, 0, sizeof(counts));
    memset(bucket_totals, 0, sizeof(bucket_totals));

    for (uint16_t mask = 0u; mask < NN_TILEBOT_ROWS; ++mask) {
        size_t bucket = nn_tile_feature_bucket(feature, mask);
        uint8_t target = nn_tile_rule_eval(rule, mask);
        if (bucket >= NN_TILEBOT_MAX_BUCKETS) {
            return 0;
        }
        ++counts[bucket][target ? 1u : 0u];
        ++bucket_totals[bucket];
        ++target_counts[target ? 1u : 0u];
    }

    double target_entropy = tb_entropy_binary(target_counts[0], target_counts[1]);
    double feature_entropy = 0.0;
    double conditional_entropy = 0.0;
    size_t used_buckets = 0u;
    for (size_t bucket = 0u; bucket < NN_TILEBOT_MAX_BUCKETS; ++bucket) {
        size_t total = bucket_totals[bucket];
        if (total == 0u) {
            continue;
        }
        ++used_buckets;
        double p = (double)total / (double)NN_TILEBOT_ROWS;
        feature_entropy -= p * (log(p) / log(2.0));
        conditional_entropy += p * tb_entropy_binary(counts[bucket][0], counts[bucket][1]);
    }

    out_score->rule = rule;
    out_score->feature = feature;
    out_score->bucket_count = used_buckets;
    out_score->target_entropy = target_entropy;
    out_score->feature_entropy = feature_entropy;
    out_score->conditional_entropy = conditional_entropy;
    out_score->information_gain = target_entropy - conditional_entropy;
    out_score->efficiency = feature_entropy > 0.0 ? out_score->information_gain / feature_entropy : 0.0;
    return 1;
}

void nn_tilebot_train_config_init(NNTileBotTrainConfig *config)
{
    if (config == NULL) {
        return;
    }
    config->hidden_width = 8u;
    config->max_epochs = 6000u;
    config->retry_count = 8u;
    config->learning_rate = (nn_real)0.08;
    config->min_margin = (nn_real)0.05;
    config->seed = 20240522u;
    config->shuffle = 1;
    config->hamming_weight_balance = 1;
}

static void tb_report_init(NNTileBotTrainReport *report, NNTileRule rule, NNTileFeature feature)
{
    if (report == NULL) {
        return;
    }
    memset(report, 0, sizeof(*report));
    snprintf(report->schema, sizeof(report->schema), "%s", NN_TILEBOT_SCHEMA);
    report->rule = rule;
    report->feature = feature;
    report->tile_rows = NN_TILEBOT_ROWS;
    report->input_width = tb_feature_input_width(feature);
    report->bucket_count = nn_tile_feature_bucket_count(feature);
    report->final_loss = (nn_real)NAN;
    report->min_margin = (nn_real)-INFINITY;
}

void nn_tilebot_init(NNTileBot *bot)
{
    if (bot == NULL) {
        return;
    }
    memset(bot, 0, sizeof(*bot));
    bot->rule = NN_TILE_RULE_DILATE;
    bot->feature = NN_TILE_FEATURE_RAW_BITS;
    bot->input_width = NN_TILEBOT_CELLS;
    bot->bucket_count = NN_TILEBOT_ROWS;
    nn_certificate_clear(&bot->certificate);
    tb_report_init(&bot->report, bot->rule, bot->feature);
}

void nn_tilebot_free(NNTileBot *bot)
{
    if (bot == NULL) {
        return;
    }
    nn_free(bot->net);
    bot->net = NULL;
    nn_certificate_clear(&bot->certificate);
}

static size_t tb_balanced_repeat(uint16_t mask,
                                 const size_t count_bucket[NN_TILEBOT_CELLS + 1u],
                                 size_t max_bucket,
                                 int enabled)
{
    if (!enabled) {
        return 1u;
    }
    size_t bucket = tb_popcount_u64(mask & 0x01ffu);
    size_t count = count_bucket[bucket];
    size_t repeat = count == 0u ? 1u : max_bucket / count;
    return repeat == 0u ? 1u : repeat;
}

static NeuralNet *tb_create_net(size_t input_width, size_t hidden_width, unsigned int seed)
{
    size_t layers_with_hidden[] = {input_width, hidden_width, 1u};
    size_t layers_linear[] = {input_width, 1u};
    NNConfig config;
    if (hidden_width == 0u) {
        config.layer_sizes = layers_linear;
        config.layer_count = sizeof(layers_linear) / sizeof(layers_linear[0]);
    } else {
        config.layer_sizes = layers_with_hidden;
        config.layer_count = sizeof(layers_with_hidden) / sizeof(layers_with_hidden[0]);
    }
    config.seed = seed;
    config.hidden_activation = NN_ACTIVATION_TANH;
    config.output_activation = NN_ACTIVATION_SIGMOID;
    config.loss = NN_LOSS_MSE;
    return nn_create_config(&config);
}

static int tb_eval_bot_net(NeuralNet *net,
                           NNTileRule rule,
                           NNTileFeature feature,
                           NNTileBotTrainReport *report)
{
    if (net == NULL || report == NULL) {
        return 0;
    }

    size_t input_width = tb_feature_input_width(feature);
    if (!tb_feature_input_width_is_valid(input_width)) {
        return 0;
    }
    nn_real features[NN_TILEBOT_MAX_BUCKETS];

    size_t correct = 0u;
    double loss = 0.0;
    nn_real min_margin = (nn_real)INFINITY;
    int finite = 1;
    for (uint16_t mask = 0u; mask < NN_TILEBOT_ROWS; ++mask) {
        nn_real output = 0;
        uint8_t expected = nn_tile_rule_eval(rule, mask);
        if (!nn_tile_feature_vector(feature, mask, features, NN_TILEBOT_MAX_BUCKETS)) {
            finite = 0;
            continue;
        }
        nn_forward(net, features, &output);
        if (!tb_finite(output)) {
            finite = 0;
            continue;
        }
        uint8_t predicted = output >= (nn_real)0.5 ? 1u : 0u;
        nn_real target = expected ? (nn_real)1 : (nn_real)0;
        nn_real error = output - target;
        nn_real margin = output >= (nn_real)0.5 ? output - (nn_real)0.5 : (nn_real)0.5 - output;
        loss += 0.5 * (double)(error * error);
        correct += predicted == expected ? 1u : 0u;
        if (margin < min_margin) {
            min_margin = margin;
        }
    }

    report->correct_rows = correct;
    report->accuracy = (double)correct / (double)NN_TILEBOT_ROWS;
    report->final_loss = finite ? (nn_real)(loss / (double)NN_TILEBOT_ROWS) : (nn_real)NAN;
    report->min_margin = finite ? min_margin : (nn_real)-INFINITY;
    return finite;
}

int nn_tilebot_train(NNTileBot *bot,
                     NNTileRule rule,
                     NNTileFeature feature,
                     const NNTileBotTrainConfig *config)
{
    if (bot == NULL || (size_t)rule >= NN_TILE_RULE_COUNT ||
        (size_t)feature >= NN_TILE_FEATURE_COUNT) {
        return 0;
    }

    NNTileBotTrainConfig default_config;
    NNTileBotTrainConfig active_config;
    if (config == NULL) {
        nn_tilebot_train_config_init(&default_config);
        active_config = default_config;
    } else {
        active_config = *config;
    }
    if (active_config.max_epochs == 0u) {
        active_config.max_epochs = 1u;
    }
    if (active_config.retry_count == 0u) {
        active_config.retry_count = 1u;
    }
    if (active_config.learning_rate <= (nn_real)0 || !tb_finite(active_config.learning_rate)) {
        active_config.learning_rate = (nn_real)0.08;
    }
    if (!tb_finite(active_config.min_margin)) {
        active_config.min_margin = (nn_real)0;
    }

    nn_tilebot_free(bot);
    bot->rule = rule;
    bot->feature = feature;
    bot->input_width = tb_feature_input_width(feature);
    bot->bucket_count = nn_tile_feature_bucket_count(feature);
    tb_report_init(&bot->report, rule, feature);
    if (!tb_feature_input_width_is_valid(bot->input_width)) {
        return 0;
    }

    size_t count_bucket[NN_TILEBOT_CELLS + 1u] = {0u};
    size_t max_bucket = 0u;
    for (uint16_t mask = 0u; mask < NN_TILEBOT_ROWS; ++mask) {
        ++count_bucket[tb_popcount_u64(mask & 0x01ffu)];
    }
    for (size_t i = 0u; i <= NN_TILEBOT_CELLS; ++i) {
        if (count_bucket[i] > max_bucket) {
            max_bucket = count_bucket[i];
        }
    }

    size_t train_rows = 0u;
    for (uint16_t mask = 0u; mask < NN_TILEBOT_ROWS; ++mask) {
        train_rows += tb_balanced_repeat(mask,
                                         count_bucket,
                                         max_bucket,
                                         active_config.hamming_weight_balance);
    }
    if (train_rows == 0u || bot->input_width > SIZE_MAX / train_rows) {
        return 0;
    }

    nn_real *inputs = (nn_real *)calloc(train_rows * bot->input_width, sizeof(*inputs));
    nn_real *targets = (nn_real *)calloc(train_rows, sizeof(*targets));
    if (inputs == NULL || targets == NULL) {
        free(inputs);
        free(targets);
        return 0;
    }

    size_t row = 0u;
    for (uint16_t mask = 0u; mask < NN_TILEBOT_ROWS; ++mask) {
        size_t repeat = tb_balanced_repeat(mask,
                                           count_bucket,
                                           max_bucket,
                                           active_config.hamming_weight_balance);
        for (size_t i = 0u; i < repeat; ++i) {
            if (!nn_tile_feature_vector(feature,
                                        mask,
                                        &inputs[row * bot->input_width],
                                        bot->input_width)) {
                free(inputs);
                free(targets);
                return 0;
            }
            targets[row] = nn_tile_rule_eval(rule, mask) ? (nn_real)1 : (nn_real)0;
            ++row;
        }
    }
    bot->report.train_rows = train_rows;

    NNTileBotTrainReport best_report;
    tb_report_init(&best_report, rule, feature);
    for (size_t attempt = 0u; attempt < active_config.retry_count; ++attempt) {
        NeuralNet *net = tb_create_net(bot->input_width,
                                       active_config.hidden_width,
                                       active_config.seed + (unsigned int)(attempt * 65537u));
        if (net == NULL) {
            continue;
        }
        (void)nn_train_batched(net,
                               inputs,
                               targets,
                               train_rows,
                               train_rows,
                               active_config.learning_rate,
                               active_config.max_epochs,
                               active_config.shuffle);

        NNTileBotTrainReport attempt_report;
        tb_report_init(&attempt_report, rule, feature);
        attempt_report.train_rows = train_rows;
        attempt_report.epochs = active_config.max_epochs;
        attempt_report.attempts = attempt + 1u;
        (void)tb_eval_bot_net(net, rule, feature, &attempt_report);

        int better = best_report.attempts == 0u ||
                     attempt_report.correct_rows > best_report.correct_rows ||
                     (attempt_report.correct_rows == best_report.correct_rows &&
                      attempt_report.min_margin > best_report.min_margin);
        if (better) {
            nn_free(bot->net);
            bot->net = net;
            best_report = attempt_report;
        } else {
            nn_free(net);
        }
        if (best_report.correct_rows == NN_TILEBOT_ROWS &&
            best_report.min_margin >= active_config.min_margin) {
            break;
        }
    }

    free(inputs);
    free(targets);

    bot->report = best_report;
    if (bot->report.correct_rows == NN_TILEBOT_ROWS &&
        bot->report.min_margin >= active_config.min_margin) {
        bot->report.promoted = 1;
        nn_certificate_accept(&bot->certificate,
                              "exhaustive_tile_table",
                              "tile bot matches every 3x3 input row",
                              NN_TILEBOT_ROWS,
                              NN_TILEBOT_ROWS);
        return 1;
    }

    nn_certificate_reject(&bot->certificate,
                          "exhaustive_tile_table",
                          "tile bot did not match every 3x3 input row",
                          NN_TILEBOT_ROWS,
                          bot->report.correct_rows);
    return 0;
}

int nn_tilebot_predict(const NNTileBot *bot,
                       uint16_t mask,
                       uint8_t *out_bit,
                       nn_real *confidence)
{
    if (bot == NULL || out_bit == NULL) {
        return 0;
    }
    mask &= 0x01ffu;
    if (bot->certificate.status == NN_CERTIFICATE_CERTIFIED) {
        *out_bit = nn_tile_rule_eval(bot->rule, mask);
        if (confidence != NULL) {
            *confidence = (nn_real)1;
        }
        return 1;
    }
    if (bot->net == NULL || !tb_feature_input_width_is_valid(bot->input_width)) {
        return 0;
    }
    nn_real features[NN_TILEBOT_MAX_BUCKETS];
    nn_real output = 0;
    int ok = nn_tile_feature_vector(bot->feature, mask, features, NN_TILEBOT_MAX_BUCKETS);
    if (ok) {
        nn_forward(bot->net, features, &output);
        ok = tb_finite(output);
    }
    if (!ok) {
        return 0;
    }
    *out_bit = output >= (nn_real)0.5 ? 1u : 0u;
    if (confidence != NULL) {
        *confidence = *out_bit ? output : (nn_real)1 - output;
    }
    return 1;
}

static int tb_offset_coord(size_t coord,
                           size_t limit,
                           int delta,
                           NNTileBoundary boundary,
                           size_t *out)
{
    if (out == NULL || limit == 0u || coord >= limit) {
        return 0;
    }
    size_t magnitude = delta < 0 ? (size_t)(-(delta + 1)) + 1u : (size_t)delta;
    if (boundary == NN_TILE_BOUNDARY_TORUS) {
        size_t step = magnitude % limit;
        *out = delta < 0 ? (coord + limit - step) % limit : (coord + step) % limit;
        return 1;
    }
    if (delta < 0) {
        if (coord < magnitude) {
            return 0;
        }
        *out = coord - magnitude;
        return 1;
    }
    if (magnitude > (limit - 1u) - coord) {
        return 0;
    }
    *out = coord + magnitude;
    return 1;
}

static uint8_t tb_grid_bit(const uint8_t *grid, size_t width, size_t x, size_t y)
{
    return (uint8_t)(grid[(size_t)y * width + (size_t)x] & 1u);
}

static uint16_t tb_read_grid_tile(const uint8_t *grid,
                                  size_t width,
                                  size_t height,
                                  size_t x,
                                  size_t y,
                                  NNTileBoundary boundary)
{
    uint16_t mask = 0u;
    unsigned bit = 0u;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            size_t sx = 0u;
            size_t sy = 0u;
            if (tb_offset_coord(x, width, dx, boundary, &sx) &&
                tb_offset_coord(y, height, dy, boundary, &sy) &&
                tb_grid_bit(grid, width, sx, sy) != 0u) {
                mask |= (uint16_t)(1u << bit);
            }
            ++bit;
        }
    }
    return mask;
}

int nn_tilebot_apply_grid(const NNTileBot *bot,
                          const uint8_t *input,
                          size_t width,
                          size_t height,
                          NNTileBoundary boundary,
                          uint8_t *output)
{
    if (bot == NULL || input == NULL || output == NULL || width == 0u || height == 0u ||
        width > SIZE_MAX / height) {
        return 0;
    }
    for (size_t y = 0u; y < height; ++y) {
        for (size_t x = 0u; x < width; ++x) {
            uint8_t out = 0u;
            if (!nn_tilebot_predict(bot,
                                    tb_read_grid_tile(input, width, height, x, y, boundary),
                                    &out,
                                    NULL)) {
                return 0;
            }
            output[y * width + x] = out;
        }
    }
    return 1;
}

int nn_tile_apply_rule_grid(NNTileRule rule,
                            const uint8_t *input,
                            size_t width,
                            size_t height,
                            NNTileBoundary boundary,
                            uint8_t *output)
{
    if (input == NULL || output == NULL || width == 0u || height == 0u ||
        width > SIZE_MAX / height) {
        return 0;
    }
    for (size_t y = 0u; y < height; ++y) {
        for (size_t x = 0u; x < width; ++x) {
            output[y * width + x] =
                nn_tile_rule_eval(rule, tb_read_grid_tile(input, width, height, x, y, boundary));
        }
    }
    return 1;
}

static void tb_random_grid(uint8_t *grid,
                           size_t cells,
                           unsigned density_percent,
                           uint32_t *rng)
{
    for (size_t i = 0u; i < cells; ++i) {
        grid[i] = (tb_rng_next(rng) % 100u) < density_percent ? 1u : 0u;
    }
}

int nn_tilebot_eval_stack(const NNTileBot *bots,
                          const NNTileRule *rules,
                          size_t layer_count,
                          size_t width,
                          size_t height,
                          unsigned density_percent,
                          size_t grid_count,
                          unsigned int seed,
                          NNTileBoundary boundary,
                          NNTileStackEvalReport *out_report)
{
    if (bots == NULL || layer_count == 0u || layer_count > NN_TILEBOT_MAX_STACK_LAYERS ||
        width == 0u || height == 0u || grid_count == 0u || out_report == NULL ||
        width > SIZE_MAX / height) {
        return 0;
    }
    size_t cells = width * height;
    uint8_t *exact_a = (uint8_t *)calloc(cells, sizeof(*exact_a));
    uint8_t *exact_b = (uint8_t *)calloc(cells, sizeof(*exact_b));
    uint8_t *learned_a = (uint8_t *)calloc(cells, sizeof(*learned_a));
    uint8_t *learned_b = (uint8_t *)calloc(cells, sizeof(*learned_b));
    if (exact_a == NULL || exact_b == NULL || learned_a == NULL || learned_b == NULL) {
        free(exact_a);
        free(exact_b);
        free(learned_a);
        free(learned_b);
        return 0;
    }

    memset(out_report, 0, sizeof(*out_report));
    out_report->layer_count = layer_count;
    out_report->evaluated_grids = grid_count;
    uint32_t rng = seed == 0u ? 1u : seed;
    for (size_t trial = 0u; trial < grid_count; ++trial) {
        tb_random_grid(exact_a, cells, density_percent, &rng);
        memcpy(learned_a, exact_a, cells * sizeof(*learned_a));
        for (size_t layer = 0u; layer < layer_count; ++layer) {
            NNTileRule rule = rules == NULL ? bots[layer].rule : rules[layer];
            if (!nn_tile_apply_rule_grid(rule, exact_a, width, height, boundary, exact_b) ||
                !nn_tilebot_apply_grid(&bots[layer], learned_a, width, height, boundary, learned_b)) {
                free(exact_a);
                free(exact_b);
                free(learned_a);
                free(learned_b);
                return 0;
            }
            memcpy(exact_a, exact_b, cells * sizeof(*exact_a));
            memcpy(learned_a, learned_b, cells * sizeof(*learned_a));
        }
        size_t correct = 0u;
        for (size_t cell = 0u; cell < cells; ++cell) {
            correct += exact_a[cell] == learned_a[cell] ? 1u : 0u;
        }
        out_report->correct_cells += correct;
        out_report->total_cells += cells;
        out_report->exact_grids += correct == cells ? 1u : 0u;
    }
    out_report->cell_accuracy =
        out_report->total_cells == 0u ? 0.0 :
                                        (double)out_report->correct_cells /
                                            (double)out_report->total_cells;

    free(exact_a);
    free(exact_b);
    free(learned_a);
    free(learned_b);
    return 1;
}

void nn_tile_macro_init(NNTileMacroTool *macro)
{
    if (macro == NULL) {
        return;
    }
    memset(macro, 0, sizeof(*macro));
    snprintf(macro->schema, sizeof(macro->schema), "%s", NN_TILEBOT_MACRO_SCHEMA);
    nn_certificate_clear(&macro->certificate);
}

int nn_tile_macro_promote(NNTileMacroTool *macro, const NNTileBot *bots, size_t layer_count)
{
    if (macro == NULL || bots == NULL || layer_count == 0u ||
        layer_count > NN_TILEBOT_MAX_STACK_LAYERS) {
        return 0;
    }
    nn_tile_macro_init(macro);
    macro->layer_count = layer_count;
    macro->receptive_field_width = 1u + 2u * layer_count;
    for (size_t i = 0u; i < layer_count; ++i) {
        macro->rules[i] = bots[i].rule;
        macro->features[i] = bots[i].feature;
        if (bots[i].certificate.status != NN_CERTIFICATE_CERTIFIED) {
            nn_certificate_reject(&macro->certificate,
                                  "certified_tile_stack",
                                  "all stack layers must be certified tile bots",
                                  layer_count,
                                  i);
            return 0;
        }
    }
    macro->promoted = 1;
    nn_certificate_accept(&macro->certificate,
                          "certified_tile_stack",
                          "composition of certified 3x3 tile bots",
                          layer_count,
                          layer_count);
    return 1;
}

int nn_tile_macro_apply(const NNTileMacroTool *macro,
                        const uint8_t *input,
                        size_t width,
                        size_t height,
                        NNTileBoundary boundary,
                        uint8_t *output)
{
    if (macro == NULL || !macro->promoted ||
        macro->certificate.status != NN_CERTIFICATE_CERTIFIED || input == NULL ||
        output == NULL || width == 0u || height == 0u || width > SIZE_MAX / height) {
        return 0;
    }
    size_t cells = width * height;
    uint8_t *a = (uint8_t *)calloc(cells, sizeof(*a));
    uint8_t *b = (uint8_t *)calloc(cells, sizeof(*b));
    if (a == NULL || b == NULL) {
        free(a);
        free(b);
        return 0;
    }
    memcpy(a, input, cells * sizeof(*a));
    int ok = 1;
    for (size_t i = 0u; ok && i < macro->layer_count; ++i) {
        ok = nn_tile_apply_rule_grid(macro->rules[i], a, width, height, boundary, b);
        if (ok) {
            memcpy(a, b, cells * sizeof(*a));
        }
    }
    if (ok) {
        memcpy(output, a, cells * sizeof(*output));
    }
    free(a);
    free(b);
    return ok;
}

uint64_t nn_life_step_torus(size_t side, uint64_t state)
{
    if (side == 0u || side > 8u || side > SIZE_MAX / side || side * side > 64u) {
        return 0u;
    }
    uint64_t out = 0u;
    for (size_t y = 0u; y < side; ++y) {
        for (size_t x = 0u; x < side; ++x) {
            uint32_t neighbors = 0u;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (dx == 0 && dy == 0) {
                        continue;
                    }
                    size_t sx = (size_t)(((int)x + dx + (int)side) % (int)side);
                    size_t sy = (size_t)(((int)y + dy + (int)side) % (int)side);
                    neighbors += (uint32_t)((state >> (sy * side + sx)) & UINT64_C(1));
                }
            }
            uint8_t center = (uint8_t)((state >> (y * side + x)) & UINT64_C(1));
            uint8_t next = neighbors == 3u || (center != 0u && neighbors == 2u) ? 1u : 0u;
            if (next != 0u) {
                out |= UINT64_C(1) << (y * side + x);
            }
        }
    }
    return out;
}

int nn_life_preimage_tables_torus(size_t side,
                                  uint32_t **counts_out,
                                  uint64_t **canonical_out)
{
    if (counts_out == NULL || side == 0u || side > 4u) {
        return 0;
    }
    size_t cells = side * side;
    size_t states = (size_t)UINT64_C(1) << cells;
    uint32_t *counts = (uint32_t *)calloc(states, sizeof(*counts));
    uint64_t *canonical =
        canonical_out == NULL ? NULL : (uint64_t *)calloc(states, sizeof(*canonical));
    if (counts == NULL || (canonical_out != NULL && canonical == NULL)) {
        free(counts);
        free(canonical);
        return 0;
    }

    for (uint64_t prior = 0u; prior < (uint64_t)states; ++prior) {
        uint64_t next = nn_life_step_torus(side, prior);
        if (canonical != NULL &&
            (counts[next] == 0u ||
             tb_popcount_u64(prior) < tb_popcount_u64(canonical[next]) ||
             (tb_popcount_u64(prior) == tb_popcount_u64(canonical[next]) &&
              prior < canonical[next]))) {
            canonical[next] = prior;
        }
        ++counts[next];
    }

    *counts_out = counts;
    if (canonical_out != NULL) {
        *canonical_out = canonical;
    }
    return 1;
}

int nn_life_preimage_stats_torus(size_t side, NNLifePreimageStats *out_stats)
{
    if (out_stats == NULL || side == 0u || side > 4u) {
        return 0;
    }
    uint32_t *counts = NULL;
    if (!nn_life_preimage_tables_torus(side, &counts, NULL)) {
        return 0;
    }

    size_t cells = side * side;
    size_t states = (size_t)UINT64_C(1) << cells;
    memset(out_stats, 0, sizeof(*out_stats));
    out_stats->side = side;
    out_stats->state_count = states;
    for (size_t target = 0u; target < states; ++target) {
        uint32_t count = counts[target];
        if (count == 0u) {
            ++out_stats->garden_of_eden;
            continue;
        }
        ++out_stats->reachable;
        out_stats->singleton_targets += count == 1u ? 1u : 0u;
        out_stats->ambiguous_targets += count > 1u ? 1u : 0u;
        if (count > out_stats->max_preimages) {
            out_stats->max_preimages = count;
        }
        out_stats->inverse_entropy_bits +=
            ((double)count / (double)states) * (log((double)count) / log(2.0));
    }
    out_stats->avg_preimages_reachable =
        out_stats->reachable == 0u ? 0.0 : (double)states / (double)out_stats->reachable;
    out_stats->image_entropy_bits = (double)cells - out_stats->inverse_entropy_bits;

    free(counts);
    return 1;
}

int nn_life_canonical_preimage_torus(size_t side,
                                     uint64_t target,
                                     uint64_t *out_prior,
                                     uint32_t *out_preimage_count)
{
    if (out_prior == NULL || out_preimage_count == NULL || side == 0u || side > 4u) {
        return 0;
    }
    size_t states = (size_t)UINT64_C(1) << (side * side);
    if (target >= (uint64_t)states) {
        return 0;
    }
    uint32_t *counts = NULL;
    uint64_t *canonical = NULL;
    if (!nn_life_preimage_tables_torus(side, &counts, &canonical)) {
        return 0;
    }
    *out_prior = canonical[target];
    *out_preimage_count = counts[target];
    int ok = *out_preimage_count != 0u;
    free(counts);
    free(canonical);
    return ok;
}
