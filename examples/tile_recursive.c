#include "nn.h"

#include <stdio.h>
#include <stdlib.h>

enum {
    STACK_LAYERS = 3,
    GRID_WIDTH = 18,
    EVAL_GRIDS = 512,
};

typedef struct {
    const char *name;
    NNTileRule rules[STACK_LAYERS];
    NNTileFeature feature;
    size_t hidden_width;
    size_t max_epochs;
    size_t retry_count;
    nn_real learning_rate;
    nn_real min_margin;
    int require_certificate;
} StackSpec;

static const StackSpec STACKS[] = {
    {
        "dilate_x3",
        {NN_TILE_RULE_DILATE, NN_TILE_RULE_DILATE, NN_TILE_RULE_DILATE},
        NN_TILE_FEATURE_POPCOUNT,
        0u,
        20000u,
        4u,
        (nn_real)0.7,
        (nn_real)0.01,
        1,
    },
    {
        "majority_x3",
        {NN_TILE_RULE_MAJORITY, NN_TILE_RULE_MAJORITY, NN_TILE_RULE_MAJORITY},
        NN_TILE_FEATURE_POPCOUNT,
        0u,
        20000u,
        4u,
        (nn_real)0.7,
        (nn_real)0.01,
        1,
    },
    {
        "dilate_majority_erode",
        {NN_TILE_RULE_DILATE, NN_TILE_RULE_MAJORITY, NN_TILE_RULE_ERODE},
        NN_TILE_FEATURE_POPCOUNT,
        0u,
        20000u,
        4u,
        (nn_real)0.7,
        (nn_real)0.01,
        1,
    },
    {
        "parity_x3_raw_gradient_probe",
        {NN_TILE_RULE_PARITY, NN_TILE_RULE_PARITY, NN_TILE_RULE_PARITY},
        NN_TILE_FEATURE_RAW_BITS,
        8u,
        1000u,
        1u,
        (nn_real)0.08,
        (nn_real)0.05,
        0,
    },
};

static int run_stack(const StackSpec *spec)
{
    NNTileBot bots[STACK_LAYERS];
    NNTileBotTrainConfig config;
    int ok = 1;
    int all_certified = 1;
    static const unsigned densities[] = {10u, 50u, 90u};

    for (size_t i = 0u; i < STACK_LAYERS; ++i) {
        nn_tilebot_init(&bots[i]);
    }
    nn_tilebot_train_config_init(&config);
    config.hidden_width = spec->hidden_width;
    config.max_epochs = spec->max_epochs;
    config.retry_count = spec->retry_count;
    config.learning_rate = spec->learning_rate;
    config.min_margin = spec->min_margin;
    config.hamming_weight_balance = 1;

    printf("stack=%s feature=%s hidden=%zu\n",
           spec->name,
           nn_tile_feature_name(spec->feature),
           spec->hidden_width);
    for (size_t layer = 0u; layer < STACK_LAYERS; ++layer) {
        config.seed = 7000u + (unsigned int)(layer * 977u) +
                      (unsigned int)(spec->rules[layer] * 37u) +
                      (unsigned int)(spec->hidden_width * 13u);
        int trained = nn_tilebot_train(&bots[layer],
                                       spec->rules[layer],
                                       spec->feature,
                                       &config);
        int certified = bots[layer].certificate.status == NN_CERTIFICATE_CERTIFIED;
        all_certified = all_certified && certified;
        ok = ok && bots[layer].net != NULL;
        if (spec->require_certificate) {
            ok = ok && trained && certified;
        }
        printf("  layer=%zu rule=%s input=%zu buckets=%zu parameters=%zu "
               "certified_rows=%zu/%u min_margin=%.6f loss=%.8f status=%s\n",
               layer + 1u,
               nn_tile_rule_name(spec->rules[layer]),
               bots[layer].input_width,
               bots[layer].bucket_count,
               nn_parameter_count(bots[layer].net),
               bots[layer].report.correct_rows,
               NN_TILEBOT_ROWS,
               (double)bots[layer].report.min_margin,
               (double)bots[layer].report.final_loss,
               nn_certificate_status_name(bots[layer].certificate.status));
    }

    for (size_t i = 0u; ok && i < sizeof(densities) / sizeof(densities[0]); ++i) {
        NNTileStackEvalReport report;
        ok = nn_tilebot_eval_stack(bots,
                                   spec->rules,
                                   STACK_LAYERS,
                                   GRID_WIDTH,
                                   GRID_WIDTH,
                                   densities[i],
                                   EVAL_GRIDS,
                                   9000u + (unsigned int)(densities[i] * 11u),
                                   NN_TILE_BOUNDARY_ZERO,
                                   &report);
        if (ok) {
            printf("  eval density=%.2f exact_grids=%zu/%d cell_accuracy=%.6f\n",
                   (double)densities[i] / 100.0,
                   report.exact_grids,
                   EVAL_GRIDS,
                   report.cell_accuracy);
        }
    }

    if (all_certified) {
        NNTileMacroTool macro;
        nn_tile_macro_init(&macro);
        ok = ok && nn_tile_macro_promote(&macro, bots, STACK_LAYERS);
        printf("  macro status=%s receptive_field=%zux%zu\n",
               nn_certificate_status_name(macro.certificate.status),
               macro.receptive_field_width,
               macro.receptive_field_width);
    } else {
        printf("  macro status=rejected reason=uncertified_layer\n");
    }

    for (size_t layer = 0u; layer < STACK_LAYERS; ++layer) {
        nn_tilebot_free(&bots[layer]);
    }
    return ok;
}

int main(void)
{
    int ok = 1;
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("recursive_3x3_stack schema=%s tile_rows=%u grid=%dx%d layers=%d\n",
           NN_TILEBOT_SCHEMA,
           NN_TILEBOT_ROWS,
           GRID_WIDTH,
           GRID_WIDTH,
           STACK_LAYERS);
    printf("tile_symbols=%u receptive_field_after_3_layers=7x7\n", NN_TILEBOT_ROWS);

    for (size_t i = 0u; i < sizeof(STACKS) / sizeof(STACKS[0]); ++i) {
        ok = run_stack(&STACKS[i]) && ok;
    }
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
