#ifndef NN_TILEBOT_H
#define NN_TILEBOT_H

#include "nn_symbolic.h"

#include <stddef.h>
#include <stdint.h>

#define NN_TILEBOT_SCHEMA "crlp.tilebot.v1"
#define NN_TILEBOT_MACRO_SCHEMA "crlp.tilebot.macro.v1"
#define NN_TILEBOT_WIDTH 3u
#define NN_TILEBOT_CELLS 9u
#define NN_TILEBOT_ROWS 512u
#define NN_TILEBOT_MAX_BUCKETS 512u
#define NN_TILEBOT_MAX_STACK_LAYERS 8u

typedef enum {
    NN_TILE_RULE_DILATE = 0,
    NN_TILE_RULE_MAJORITY = 1,
    NN_TILE_RULE_ERODE = 2,
    NN_TILE_RULE_PARITY = 3,
    NN_TILE_RULE_COPY_CENTER = 4,
    NN_TILE_RULE_EDGE_ANY = 5,
    NN_TILE_RULE_CORNER_PARITY = 6,
    NN_TILE_RULE_LIFE = 7,
    NN_TILE_RULE_COUNT = 8,
} NNTileRule;

typedef enum {
    NN_TILE_FEATURE_RAW_BITS = 0,
    NN_TILE_FEATURE_RAW_MASK = 1,
    NN_TILE_FEATURE_POPCOUNT = 2,
    NN_TILE_FEATURE_CENTER = 3,
    NN_TILE_FEATURE_BOUNDARY_COUNT = 4,
    NN_TILE_FEATURE_CORNER_COUNT = 5,
    NN_TILE_FEATURE_EDGE_COUNT = 6,
    NN_TILE_FEATURE_CORNER_EDGE_CENTER = 7,
    NN_TILE_FEATURE_D4_ORBIT = 8,
    NN_TILE_FEATURE_LIFE_CENTER_NEIGHBOR_COUNT = 9,
    NN_TILE_FEATURE_COUNT = 10,
} NNTileFeature;

typedef enum {
    NN_TILE_BOUNDARY_ZERO = 0,
    NN_TILE_BOUNDARY_TORUS = 1,
} NNTileBoundary;

typedef struct {
    size_t hidden_width;
    size_t max_epochs;
    size_t retry_count;
    nn_real learning_rate;
    nn_real min_margin;
    unsigned int seed;
    int shuffle;
    int hamming_weight_balance;
} NNTileBotTrainConfig;

typedef struct {
    char schema[32];
    NNTileRule rule;
    NNTileFeature feature;
    size_t tile_rows;
    size_t input_width;
    size_t bucket_count;
    size_t train_rows;
    size_t epochs;
    size_t attempts;
    size_t correct_rows;
    nn_real final_loss;
    nn_real min_margin;
    double accuracy;
    int promoted;
} NNTileBotTrainReport;

typedef struct {
    NNTileRule rule;
    NNTileFeature feature;
    size_t input_width;
    size_t bucket_count;
    NeuralNet *net;
    NNCertificate certificate;
    NNTileBotTrainReport report;
} NNTileBot;

typedef struct {
    NNTileRule rule;
    NNTileFeature feature;
    size_t bucket_count;
    double target_entropy;
    double feature_entropy;
    double conditional_entropy;
    double information_gain;
    double efficiency;
} NNTileFeatureScore;

typedef struct {
    size_t layer_count;
    size_t evaluated_grids;
    size_t exact_grids;
    size_t total_cells;
    size_t correct_cells;
    double cell_accuracy;
} NNTileStackEvalReport;

typedef struct {
    char schema[32];
    NNTileRule rules[NN_TILEBOT_MAX_STACK_LAYERS];
    NNTileFeature features[NN_TILEBOT_MAX_STACK_LAYERS];
    size_t layer_count;
    size_t receptive_field_width;
    NNCertificate certificate;
    int promoted;
} NNTileMacroTool;

typedef struct {
    size_t side;
    size_t state_count;
    size_t reachable;
    size_t garden_of_eden;
    size_t singleton_targets;
    size_t ambiguous_targets;
    uint32_t max_preimages;
    double avg_preimages_reachable;
    double inverse_entropy_bits;
    double image_entropy_bits;
} NNLifePreimageStats;

const char *nn_tile_rule_name(NNTileRule rule);
const char *nn_tile_feature_name(NNTileFeature feature);
size_t nn_tile_feature_bucket_count(NNTileFeature feature);
uint8_t nn_tile_rule_eval(NNTileRule rule, uint16_t mask);
size_t nn_tile_feature_bucket(NNTileFeature feature, uint16_t mask);
int nn_tile_feature_vector(NNTileFeature feature, uint16_t mask, nn_real *out, size_t capacity);
int nn_tile_feature_score(NNTileRule rule,
                          NNTileFeature feature,
                          NNTileFeatureScore *out_score);

void nn_tilebot_train_config_init(NNTileBotTrainConfig *config);
void nn_tilebot_init(NNTileBot *bot);
void nn_tilebot_free(NNTileBot *bot);
/* bot must be initialized with nn_tilebot_init, or be zero-initialized, before training. */
int nn_tilebot_train(NNTileBot *bot,
                     NNTileRule rule,
                     NNTileFeature feature,
                     const NNTileBotTrainConfig *config);
int nn_tilebot_predict(const NNTileBot *bot,
                       uint16_t mask,
                       uint8_t *out_bit,
                       nn_real *confidence);
int nn_tilebot_apply_grid(const NNTileBot *bot,
                          const uint8_t *input,
                          size_t width,
                          size_t height,
                          NNTileBoundary boundary,
                          uint8_t *output);
int nn_tile_apply_rule_grid(NNTileRule rule,
                            const uint8_t *input,
                            size_t width,
                            size_t height,
                            NNTileBoundary boundary,
                            uint8_t *output);
int nn_tilebot_eval_stack(const NNTileBot *bots,
                          const NNTileRule *rules,
                          size_t layer_count,
                          size_t width,
                          size_t height,
                          unsigned density_percent,
                          size_t grid_count,
                          unsigned int seed,
                          NNTileBoundary boundary,
                          NNTileStackEvalReport *out_report);

void nn_tile_macro_init(NNTileMacroTool *macro);
int nn_tile_macro_promote(NNTileMacroTool *macro, const NNTileBot *bots, size_t layer_count);
int nn_tile_macro_apply(const NNTileMacroTool *macro,
                        const uint8_t *input,
                        size_t width,
                        size_t height,
                        NNTileBoundary boundary,
                        uint8_t *output);

uint64_t nn_life_step_torus(size_t side, uint64_t state);
int nn_life_preimage_stats_torus(size_t side, NNLifePreimageStats *out_stats);
/* Allocates counts and, when canonical_out is non-NULL, canonical prior tables. Caller frees. */
int nn_life_preimage_tables_torus(size_t side, uint32_t **counts_out, uint64_t **canonical_out);
int nn_life_canonical_preimage_torus(size_t side,
                                     uint64_t target,
                                     uint64_t *out_prior,
                                     uint32_t *out_preimage_count);

#endif
