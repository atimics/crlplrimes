#include "nn.h"

#include <stdio.h>
#include <stdlib.h>

enum {
    FEATURE_COUNT = 3,
    CANDIDATE_COUNT = 4,
    TRAINING_STEPS = 1200,
};

typedef struct {
    unsigned int action_id;
    const char *name;
    nn_real features[FEATURE_COUNT];
} ActionRow;

int main(void)
{
    const size_t layers[] = {FEATURE_COUNT, 8, 1};
    NNConfig config;
    config.layer_sizes = layers;
    config.layer_count = sizeof(layers) / sizeof(layers[0]);
    config.seed = 42;
    config.hidden_activation = NN_ACTIVATION_RELU;
    config.output_activation = NN_ACTIVATION_LINEAR;
    config.loss = NN_LOSS_MSE;

    ActionRow rows[CANDIDATE_COUNT] = {
        {10, "ignore", {0.05f, 0.05f, 0.10f}},
        {20, "downweight", {0.35f, 0.30f, 0.25f}},
        {30, "request_verification", {1.00f, 0.85f, 0.15f}},
        {40, "act_cautiously", {0.65f, 0.45f, 0.40f}},
    };
    nn_real candidate_inputs[CANDIDATE_COUNT * FEATURE_COUNT];
    NNActionCandidate candidates[CANDIDATE_COUNT];
    size_t ranked[CANDIDATE_COUNT];

    for (size_t action = 0; action < CANDIDATE_COUNT; ++action) {
        for (size_t feature = 0; feature < FEATURE_COUNT; ++feature) {
            candidate_inputs[action * FEATURE_COUNT + feature] = rows[action].features[feature];
        }
        candidates[action].action_id = rows[action].action_id;
        candidates[action].features = rows[action].features;
        candidates[action].score = 0;
        candidates[action].probability = 0;
    }

    NeuralNet *net = nn_create_config(&config);
    if (net == NULL) {
        fputs("failed to create action scorer\n", stderr);
        return EXIT_FAILURE;
    }

    nn_real loss = 0;
    for (size_t step = 0; step < TRAINING_STEPS; ++step) {
        loss = nn_train_action_scorer(net,
                                      candidate_inputs,
                                      CANDIDATE_COUNT,
                                      2,
                                      (nn_real)0.02);
    }

    if (!nn_score_actions(net, candidates, CANDIDATE_COUNT) ||
        !nn_rank_actions(candidates, CANDIDATE_COUNT, ranked)) {
        fputs("failed to score action candidates\n", stderr);
        nn_free(net);
        return EXIT_FAILURE;
    }

    printf("generic action scorer loss=%.4f\n", (double)loss);
    printf("target action=request_verification\n");
    puts("ranked proposals");
    for (size_t rank = 0; rank < CANDIDATE_COUNT; ++rank) {
        size_t index = ranked[rank];
        printf("  %zu. id=%u %-20s score=% .4f probability=%.4f\n",
               rank + 1,
               candidates[index].action_id,
               rows[index].name,
               (double)candidates[index].score,
               (double)candidates[index].probability);
    }

    nn_free(net);
    return ranked[0] == 2 ? EXIT_SUCCESS : EXIT_FAILURE;
}
