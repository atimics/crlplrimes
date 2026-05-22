#include "nn.h"

#include <stdio.h>
#include <stdlib.h>

typedef struct {
    NNTileRule rule;
    NNTileFeature feature;
} FeatureBotSpec;

static const NNTileRule RULES[] = {
    NN_TILE_RULE_DILATE,
    NN_TILE_RULE_MAJORITY,
    NN_TILE_RULE_ERODE,
    NN_TILE_RULE_PARITY,
    NN_TILE_RULE_COPY_CENTER,
    NN_TILE_RULE_EDGE_ANY,
    NN_TILE_RULE_CORNER_PARITY,
    NN_TILE_RULE_LIFE,
};

static const NNTileFeature FEATURES[] = {
    NN_TILE_FEATURE_POPCOUNT,
    NN_TILE_FEATURE_CENTER,
    NN_TILE_FEATURE_BOUNDARY_COUNT,
    NN_TILE_FEATURE_CORNER_COUNT,
    NN_TILE_FEATURE_EDGE_COUNT,
    NN_TILE_FEATURE_CORNER_EDGE_CENTER,
    NN_TILE_FEATURE_D4_ORBIT,
    NN_TILE_FEATURE_LIFE_CENTER_NEIGHBOR_COUNT,
    NN_TILE_FEATURE_RAW_MASK,
};

static const FeatureBotSpec FEATURE_BOTS[] = {
    {NN_TILE_RULE_DILATE, NN_TILE_FEATURE_POPCOUNT},
    {NN_TILE_RULE_MAJORITY, NN_TILE_FEATURE_POPCOUNT},
    {NN_TILE_RULE_ERODE, NN_TILE_FEATURE_POPCOUNT},
    {NN_TILE_RULE_PARITY, NN_TILE_FEATURE_POPCOUNT},
    {NN_TILE_RULE_LIFE, NN_TILE_FEATURE_LIFE_CENTER_NEIGHBOR_COUNT},
};

static void print_feature_scores(void)
{
    for (size_t r = 0u; r < sizeof(RULES) / sizeof(RULES[0]); ++r) {
        printf("target=%s\n", nn_tile_rule_name(RULES[r]));
        for (size_t f = 0u; f < sizeof(FEATURES) / sizeof(FEATURES[0]); ++f) {
            NNTileFeatureScore score;
            if (!nn_tile_feature_score(RULES[r], FEATURES[f], &score)) {
                continue;
            }
            printf("  feature=%-26s buckets=%3zu H(Y)=%.6f H(F)=%.6f "
                   "H(Y|F)=%.6f IG=%.6f efficiency=%.6f exact=%s\n",
                   nn_tile_feature_name(FEATURES[f]),
                   score.bucket_count,
                   score.target_entropy,
                   score.feature_entropy,
                   score.conditional_entropy,
                   score.information_gain,
                   score.efficiency,
                   score.conditional_entropy < 1e-9 ? "yes" : "no");
        }
    }
}

static int train_feature_bots(void)
{
    NNTileBotTrainConfig config;
    nn_tilebot_train_config_init(&config);
    config.hidden_width = 0u;
    config.max_epochs = 20000u;
    config.retry_count = 4u;
    config.learning_rate = (nn_real)0.7;
    config.min_margin = (nn_real)0.01;

    int ok = 1;
    for (size_t i = 0u; i < sizeof(FEATURE_BOTS) / sizeof(FEATURE_BOTS[0]); ++i) {
        NNTileBot bot;
        nn_tilebot_init(&bot);
        config.seed = 45000u + (unsigned int)(i * 977u);
        ok = nn_tilebot_train(&bot,
                              FEATURE_BOTS[i].rule,
                              FEATURE_BOTS[i].feature,
                              &config) &&
             ok;
        printf("feature_bot target=%-14s feature=%-26s input=%2zu buckets=%3zu "
               "parameters=%zu certified_rows=%zu/%u min_margin=%.6f status=%s\n",
               nn_tile_rule_name(FEATURE_BOTS[i].rule),
               nn_tile_feature_name(FEATURE_BOTS[i].feature),
               bot.input_width,
               bot.bucket_count,
               nn_parameter_count(bot.net),
               bot.report.correct_rows,
               NN_TILEBOT_ROWS,
               (double)bot.report.min_margin,
               nn_certificate_status_name(bot.certificate.status));
        nn_tilebot_free(&bot);
    }
    return ok;
}

int main(void)
{
    printf("tile_feature_entropy schema=%s tile=3x3 rows=%u\n",
           NN_TILEBOT_SCHEMA,
           NN_TILEBOT_ROWS);
    print_feature_scores();
    return train_feature_bots() ? EXIT_SUCCESS : EXIT_FAILURE;
}
