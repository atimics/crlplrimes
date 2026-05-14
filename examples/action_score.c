#include "nn.h"

#include <stdio.h>
#include <stdlib.h>

int main(void)
{
    const size_t layers[] = {3, 6, 1};
    const nn_real candidate_inputs[] = {
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    };
    const char *labels[] = {"trust", "ignore", "request_more_evidence"};
    nn_real scores[3];
    nn_real probabilities[3];
    size_t ranked[3];
    NNConfig config;

    config.layer_sizes = layers;
    config.layer_count = sizeof(layers) / sizeof(layers[0]);
    config.seed = 77;
    config.hidden_activation = NN_ACTIVATION_RELU;
    config.output_activation = NN_ACTIVATION_LINEAR;
    config.loss = NN_LOSS_MSE;

    NeuralNet *net = nn_create_config(&config);
    if (net == NULL) {
        fputs("failed to create action scorer\n", stderr);
        return EXIT_FAILURE;
    }

    for (size_t epoch = 0; epoch < 200; ++epoch) {
        nn_train_action_scorer(net, candidate_inputs, 3, 2, (nn_real)0.02);
    }

    if (!nn_score_action_rows(net, candidate_inputs, 3, scores, probabilities) ||
        !nn_rank_action_scores(scores, 3, ranked)) {
        fputs("failed to score actions\n", stderr);
        nn_free(net);
        return EXIT_FAILURE;
    }

    puts("ranked candidate actions:");
    for (size_t rank = 0; rank < 3; ++rank) {
        size_t action = ranked[rank];
        printf("  %zu. %s score=%.4f probability=%.4f\n",
               rank + 1,
               labels[action],
               (double)scores[action],
               (double)probabilities[action]);
    }

    nn_free(net);
    return EXIT_SUCCESS;
}
