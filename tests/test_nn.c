#include "nn.h"
#include "prime_gap.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if NN_REAL_IS_FLOAT
#define TEST_EPS ((nn_real)1e-2)
#define TEST_ABS_TOL ((nn_real)2e-2)
#define TEST_REL_TOL ((nn_real)8e-2)
#else
#define TEST_EPS ((nn_real)1e-5)
#define TEST_ABS_TOL ((nn_real)1e-5)
#define TEST_REL_TOL ((nn_real)1e-4)
#endif

static int failures = 0;

static nn_real abs_real(nn_real value)
{
    return value < (nn_real)0 ? -value : value;
}

static int nearly_equal(nn_real a, nn_real b, nn_real abs_tol, nn_real rel_tol)
{
    nn_real diff = abs_real(a - b);
    nn_real scale = abs_real(a) > abs_real(b) ? abs_real(a) : abs_real(b);
    return diff <= abs_tol || diff <= rel_tol * scale;
}

static void check(int condition, const char *name)
{
    if (condition) {
        printf("ok   %s\n", name);
        return;
    }

    printf("FAIL %s\n", name);
    ++failures;
}

static void fill_inputs(nn_real *inputs, size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        inputs[i] = (nn_real)(((i * 37u) % 101u) / 100.0);
    }
}

static void fill_targets(nn_real *targets, size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        targets[i] = (nn_real)(((i * 19u + 7u) % 89u) / 88.0);
    }
}

static nn_real average_loss(NeuralNet *net,
                            const nn_real *inputs,
                            const nn_real *targets,
                            size_t sample_count)
{
    size_t input_width = nn_input_size(net);
    size_t output_width = nn_output_size(net);
    nn_real loss = 0;
    nn_real output[16];

    if (output_width > sizeof(output) / sizeof(output[0])) {
        return (nn_real)NAN;
    }

    for (size_t sample = 0; sample < sample_count; ++sample) {
        nn_forward(net, &inputs[sample * input_width], output);
        for (size_t i = 0; i < output_width; ++i) {
            nn_real error = output[i] - targets[sample * output_width + i];
            loss += (nn_real)0.5 * error * error;
        }
    }

    return loss / (nn_real)sample_count;
}

static nn_real average_cross_entropy_loss(NeuralNet *net,
                                          const nn_real *inputs,
                                          const nn_real *targets,
                                          size_t sample_count)
{
    size_t input_width = nn_input_size(net);
    size_t output_width = nn_output_size(net);
    nn_real loss = 0;
    nn_real output[16];

    if (output_width > sizeof(output) / sizeof(output[0])) {
        return (nn_real)NAN;
    }

    for (size_t sample = 0; sample < sample_count; ++sample) {
        nn_forward(net, &inputs[sample * input_width], output);
        loss += nn_softmax_cross_entropy(output, &targets[sample * output_width], output_width);
    }

    return loss / (nn_real)sample_count;
}

static nn_real scorer_loss(NeuralNet *net,
                           const nn_real *candidate_inputs,
                           size_t candidate_count,
                           size_t target_index)
{
    nn_real scores[8];
    nn_real targets[8] = {0};

    if (candidate_count > sizeof(scores) / sizeof(scores[0]) || target_index >= candidate_count) {
        return (nn_real)NAN;
    }

    nn_forward_batch(net, candidate_inputs, scores, candidate_count);
    targets[target_index] = (nn_real)1;
    return nn_softmax_cross_entropy(scores, targets, candidate_count);
}

static nn_real scorer_entropy_loss(NeuralNet *net,
                                   const nn_real *candidate_inputs,
                                   size_t candidate_count,
                                   size_t target_index,
                                   nn_real entropy_weight)
{
    nn_real scores[8];
    nn_real probabilities[8];
    nn_real targets[8] = {0};
    nn_real entropy = 0;

    if (candidate_count > sizeof(scores) / sizeof(scores[0]) || target_index >= candidate_count) {
        return (nn_real)NAN;
    }

    nn_forward_batch(net, candidate_inputs, scores, candidate_count);
    nn_softmax(scores, probabilities, candidate_count);
    targets[target_index] = (nn_real)1;
    for (size_t i = 0; i < candidate_count; ++i) {
        if (probabilities[i] > (nn_real)0) {
            entropy -= probabilities[i] * (nn_real)log((double)probabilities[i]);
        }
    }

    return nn_softmax_cross_entropy(scores, targets, candidate_count) -
           entropy_weight * entropy;
}

static nn_real scorer_targets_loss(NeuralNet *net,
                                   const nn_real *candidate_inputs,
                                   const nn_real *targets,
                                   size_t candidate_count)
{
    nn_real scores[8];
    nn_real target_sum = 0;

    if (candidate_count > sizeof(scores) / sizeof(scores[0])) {
        return (nn_real)NAN;
    }

    nn_forward_batch(net, candidate_inputs, scores, candidate_count);
    for (size_t i = 0; i < candidate_count; ++i) {
        target_sum += targets[i];
    }
    if (target_sum <= (nn_real)0) {
        return (nn_real)NAN;
    }

    return nn_softmax_cross_entropy(scores, targets, candidate_count) / target_sum;
}

static int test_forward_batch_matches_scalar(void)
{
    const size_t layers[] = {5, 7, 3};
    const size_t sample_count = 5;
    NeuralNet *net = nn_create(layers, 3, 123);
    nn_real inputs[5 * 5];
    nn_real scalar_outputs[5 * 3];
    nn_real batch_outputs[5 * 3];
    int ok = 1;

    if (net == NULL) {
        return 0;
    }

    fill_inputs(inputs, sizeof(inputs) / sizeof(inputs[0]));

    for (size_t sample = 0; sample < sample_count; ++sample) {
        nn_forward(net, &inputs[sample * layers[0]], &scalar_outputs[sample * layers[2]]);
    }
    nn_forward_batch(net, inputs, batch_outputs, sample_count);

    for (size_t i = 0; i < sizeof(batch_outputs) / sizeof(batch_outputs[0]); ++i) {
        if (!nearly_equal(scalar_outputs[i], batch_outputs[i], (nn_real)1e-6, (nn_real)1e-5)) {
            ok = 0;
            break;
        }
    }

    nn_free(net);
    return ok;
}

static int test_gradient_buffers_are_lazy(void)
{
    const size_t layers[] = {5, 6, 3};
    NeuralNet *net = nn_create(layers, 3, 321);
    nn_real input[5];
    nn_real target[3];
    nn_real output[3];
    int ok = 1;

    if (net == NULL) {
        return 0;
    }

    fill_inputs(input, sizeof(input) / sizeof(input[0]));
    fill_targets(target, sizeof(target) / sizeof(target[0]));

    ok = ok && !nn_test_has_gradient_buffers(net);
    nn_forward(net, input, output);
    ok = ok && !nn_test_has_gradient_buffers(net);
    nn_train_sample(net, input, target, (nn_real)0.03);
    ok = ok && !nn_test_has_gradient_buffers(net);
    nn_train_batch(net, input, target, 1, (nn_real)0.03);
    ok = ok && nn_test_has_gradient_buffers(net);

    nn_free(net);
    return ok;
}

static int test_batch_size_one_matches_sample_training(void)
{
    const size_t layers[] = {5, 6, 3};
    const size_t sample_count = 5;
    NeuralNet *sample_net = nn_create(layers, 3, 456);
    NeuralNet *batch_net = nn_create(layers, 3, 456);
    nn_real inputs[5 * 5];
    nn_real targets[5 * 3];
    int ok = 1;

    if (sample_net == NULL || batch_net == NULL) {
        nn_free(sample_net);
        nn_free(batch_net);
        return 0;
    }

    fill_inputs(inputs, sizeof(inputs) / sizeof(inputs[0]));
    fill_targets(targets, sizeof(targets) / sizeof(targets[0]));

    for (size_t sample = 0; sample < sample_count; ++sample) {
        nn_train_sample(sample_net,
                        &inputs[sample * layers[0]],
                        &targets[sample * layers[2]],
                        (nn_real)0.03);
    }

    nn_train_batched(batch_net, inputs, targets, sample_count, 1, (nn_real)0.03, 1, 0);

    for (size_t i = 0; i < nn_parameter_count(sample_net); ++i) {
        nn_real a = nn_test_get_parameter(sample_net, i);
        nn_real b = nn_test_get_parameter(batch_net, i);
        if (!nearly_equal(a, b, (nn_real)1e-6, (nn_real)1e-5)) {
            ok = 0;
            break;
        }
    }

    nn_free(sample_net);
    nn_free(batch_net);
    return ok;
}

static int test_batched_gradient_matches_finite_difference(void)
{
    const size_t layers[] = {5, 6, 5};
    const size_t sample_count = 5;
    const nn_real learning_rate = (nn_real)1e-3;
    NeuralNet *base_net = nn_create(layers, 3, 789);
    NeuralNet *update_net = nn_create(layers, 3, 789);
    nn_real inputs[5 * 5];
    nn_real targets[5 * 5];
    int ok = 1;

    if (base_net == NULL || update_net == NULL) {
        nn_free(base_net);
        nn_free(update_net);
        return 0;
    }

    fill_inputs(inputs, sizeof(inputs) / sizeof(inputs[0]));
    fill_targets(targets, sizeof(targets) / sizeof(targets[0]));

    nn_train_batch(update_net, inputs, targets, sample_count, learning_rate);

    for (size_t i = 0; i < nn_parameter_count(base_net); ++i) {
        nn_real original = nn_test_get_parameter(base_net, i);
        nn_real updated = nn_test_get_parameter(update_net, i);
        nn_real analytic = (original - updated) / learning_rate;

        nn_test_set_parameter(base_net, i, original + TEST_EPS);
        nn_real plus_loss = average_loss(base_net, inputs, targets, sample_count);
        nn_test_set_parameter(base_net, i, original - TEST_EPS);
        nn_real minus_loss = average_loss(base_net, inputs, targets, sample_count);
        nn_test_set_parameter(base_net, i, original);

        nn_real numeric = (plus_loss - minus_loss) / ((nn_real)2 * TEST_EPS);
        if (!nearly_equal(analytic, numeric, TEST_ABS_TOL, TEST_REL_TOL)) {
            fprintf(stderr,
                    "gradient mismatch param=%zu analytic=%g numeric=%g\n",
                    i,
                    (double)analytic,
                    (double)numeric);
            ok = 0;
            break;
        }
    }

    nn_free(base_net);
    nn_free(update_net);
    return ok;
}

static int test_softmax_cross_entropy_config_learns(void)
{
    const size_t layers[] = {3, 8, 3};
    const size_t sample_count = 3;
    const nn_real inputs[] = {
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
    };
    const nn_real targets[] = {
        1, 0, 0,
        0, 1, 0,
        0, 0, 1,
    };
    NNConfig config;
    config.layer_sizes = layers;
    config.layer_count = 3;
    config.seed = 2468;
    config.hidden_activation = NN_ACTIVATION_RELU;
    config.output_activation = NN_ACTIVATION_LINEAR;
    config.loss = NN_LOSS_SOFTMAX_CROSS_ENTROPY;

    NeuralNet *net = nn_create_config(&config);
    int ok = 1;
    if (net == NULL) {
        return 0;
    }

    nn_real before = average_cross_entropy_loss(net, inputs, targets, sample_count);
    nn_train_batched(net, inputs, targets, sample_count, sample_count, (nn_real)0.05, 300, 1);
    nn_real after = average_cross_entropy_loss(net, inputs, targets, sample_count);

    ok = ok && after < before * (nn_real)0.5;

    for (size_t sample = 0; sample < sample_count; ++sample) {
        nn_real logits[3];
        nn_forward(net, &inputs[sample * 3], logits);

        size_t best = 0;
        for (size_t i = 1; i < 3; ++i) {
            if (logits[i] > logits[best]) {
                best = i;
            }
        }

        ok = ok && best == sample;
    }

    nn_free(net);
    return ok;
}

static int test_softmax_scorer_gradient_matches_finite_difference(void)
{
    const size_t layers[] = {4, 5, 1};
    const size_t candidate_count = 4;
    const size_t target_index = 2;
    const nn_real learning_rate = (nn_real)1e-3;
    const nn_real candidate_inputs[] = {
        0.10, 0.20, 0.30, 0.40,
        0.30, 0.10, 0.70, 0.20,
        0.80, 0.20, 0.10, 0.60,
        0.25, 0.50, 0.25, 0.75,
    };
    NNConfig config;
    config.layer_sizes = layers;
    config.layer_count = 3;
    config.seed = 1357;
    config.hidden_activation = NN_ACTIVATION_TANH;
    config.output_activation = NN_ACTIVATION_LINEAR;
    config.loss = NN_LOSS_MSE;

    NeuralNet *base_net = nn_create_config(&config);
    NeuralNet *update_net = nn_create_config(&config);
    int ok = 1;

    if (base_net == NULL || update_net == NULL) {
        nn_free(base_net);
        nn_free(update_net);
        return 0;
    }

    nn_train_softmax_scorer(update_net,
                            candidate_inputs,
                            candidate_count,
                            target_index,
                            learning_rate);

    for (size_t i = 0; i < nn_parameter_count(base_net); ++i) {
        nn_real original = nn_test_get_parameter(base_net, i);
        nn_real updated = nn_test_get_parameter(update_net, i);
        nn_real analytic = (original - updated) / learning_rate;

        nn_test_set_parameter(base_net, i, original + TEST_EPS);
        nn_real plus_loss = scorer_loss(base_net, candidate_inputs, candidate_count, target_index);
        nn_test_set_parameter(base_net, i, original - TEST_EPS);
        nn_real minus_loss = scorer_loss(base_net, candidate_inputs, candidate_count, target_index);
        nn_test_set_parameter(base_net, i, original);

        nn_real numeric = (plus_loss - minus_loss) / ((nn_real)2 * TEST_EPS);
        if (!nearly_equal(analytic, numeric, TEST_ABS_TOL, TEST_REL_TOL)) {
            fprintf(stderr,
                    "scorer gradient mismatch param=%zu analytic=%g numeric=%g\n",
                    i,
                    (double)analytic,
                    (double)numeric);
            ok = 0;
            break;
        }
    }

    nn_free(base_net);
    nn_free(update_net);
    return ok;
}

static int test_softmax_scorer_entropy_gradient_matches_finite_difference(void)
{
    const size_t layers[] = {4, 5, 1};
    const size_t candidate_count = 4;
    const size_t target_index = 2;
    const nn_real learning_rate = (nn_real)1e-3;
    const nn_real entropy_weight = (nn_real)0.05;
    const nn_real candidate_inputs[] = {
        0.10, 0.20, 0.30, 0.40,
        0.30, 0.10, 0.70, 0.20,
        0.80, 0.20, 0.10, 0.60,
        0.25, 0.50, 0.25, 0.75,
    };
    NNConfig config;
    config.layer_sizes = layers;
    config.layer_count = 3;
    config.seed = 9753;
    config.hidden_activation = NN_ACTIVATION_TANH;
    config.output_activation = NN_ACTIVATION_LINEAR;
    config.loss = NN_LOSS_MSE;

    NeuralNet *base_net = nn_create_config(&config);
    NeuralNet *update_net = nn_create_config(&config);
    int ok = 1;

    if (base_net == NULL || update_net == NULL) {
        nn_free(base_net);
        nn_free(update_net);
        return 0;
    }

    nn_train_softmax_scorer_entropy(update_net,
                                    candidate_inputs,
                                    candidate_count,
                                    target_index,
                                    entropy_weight,
                                    learning_rate);

    for (size_t i = 0; i < nn_parameter_count(base_net); ++i) {
        nn_real original = nn_test_get_parameter(base_net, i);
        nn_real updated = nn_test_get_parameter(update_net, i);
        nn_real analytic = (original - updated) / learning_rate;

        nn_test_set_parameter(base_net, i, original + TEST_EPS);
        nn_real plus_loss = scorer_entropy_loss(base_net,
                                                candidate_inputs,
                                                candidate_count,
                                                target_index,
                                                entropy_weight);
        nn_test_set_parameter(base_net, i, original - TEST_EPS);
        nn_real minus_loss = scorer_entropy_loss(base_net,
                                                 candidate_inputs,
                                                 candidate_count,
                                                 target_index,
                                                 entropy_weight);
        nn_test_set_parameter(base_net, i, original);

        nn_real numeric = (plus_loss - minus_loss) / ((nn_real)2 * TEST_EPS);
        if (!nearly_equal(analytic, numeric, TEST_ABS_TOL, TEST_REL_TOL)) {
            fprintf(stderr,
                    "entropy scorer gradient mismatch param=%zu analytic=%g numeric=%g\n",
                    i,
                    (double)analytic,
                    (double)numeric);
            ok = 0;
            break;
        }
    }

    nn_free(base_net);
    nn_free(update_net);
    return ok;
}

static int test_softmax_scorer_targets_gradient_matches_finite_difference(void)
{
    const size_t layers[] = {4, 5, 1};
    const size_t candidate_count = 4;
    const nn_real learning_rate = (nn_real)1e-3;
    const nn_real candidate_inputs[] = {
        0.10, 0.20, 0.30, 0.40,
        0.30, 0.10, 0.70, 0.20,
        0.80, 0.20, 0.10, 0.60,
        0.25, 0.50, 0.25, 0.75,
    };
    const nn_real targets[] = {0, 2, 1, 0};
    NNConfig config;
    config.layer_sizes = layers;
    config.layer_count = 3;
    config.seed = 7531;
    config.hidden_activation = NN_ACTIVATION_TANH;
    config.output_activation = NN_ACTIVATION_LINEAR;
    config.loss = NN_LOSS_MSE;

    NeuralNet *base_net = nn_create_config(&config);
    NeuralNet *update_net = nn_create_config(&config);
    int ok = 1;

    if (base_net == NULL || update_net == NULL) {
        nn_free(base_net);
        nn_free(update_net);
        return 0;
    }

    nn_train_softmax_scorer_targets(update_net,
                                    candidate_inputs,
                                    candidate_count,
                                    targets,
                                    learning_rate);

    for (size_t i = 0; i < nn_parameter_count(base_net); ++i) {
        nn_real original = nn_test_get_parameter(base_net, i);
        nn_real updated = nn_test_get_parameter(update_net, i);
        nn_real analytic = (original - updated) / learning_rate;

        nn_test_set_parameter(base_net, i, original + TEST_EPS);
        nn_real plus_loss = scorer_targets_loss(base_net, candidate_inputs, targets, candidate_count);
        nn_test_set_parameter(base_net, i, original - TEST_EPS);
        nn_real minus_loss = scorer_targets_loss(base_net, candidate_inputs, targets, candidate_count);
        nn_test_set_parameter(base_net, i, original);

        nn_real numeric = (plus_loss - minus_loss) / ((nn_real)2 * TEST_EPS);
        if (!nearly_equal(analytic, numeric, TEST_ABS_TOL, TEST_REL_TOL)) {
            fprintf(stderr,
                    "target scorer gradient mismatch param=%zu analytic=%g numeric=%g\n",
                    i,
                    (double)analytic,
                    (double)numeric);
            ok = 0;
            break;
        }
    }

    nn_free(base_net);
    nn_free(update_net);
    return ok;
}

static int test_softmax_cross_entropy_requires_linear_output(void)
{
    const size_t layers[] = {3, 4, 3};
    NNConfig config;
    config.layer_sizes = layers;
    config.layer_count = 3;
    config.seed = 999;
    config.hidden_activation = NN_ACTIVATION_RELU;
    config.output_activation = NN_ACTIVATION_SIGMOID;
    config.loss = NN_LOSS_SOFTMAX_CROSS_ENTROPY;

    NeuralNet *net = nn_create_config(&config);
    nn_free(net);
    return net == NULL;
}

static int test_softmax_scorer_cached_matches_regular(void)
{
    const size_t layers[] = {4, 5, 1};
    const size_t candidate_count = 4;
    const size_t selected_index = 1;
    const nn_real learning_rate = (nn_real)1e-3;
    const nn_real weight = (nn_real)-0.35;
    const nn_real candidate_inputs[] = {
        0.10, 0.20, 0.30, 0.40,
        0.30, 0.10, 0.70, 0.20,
        0.80, 0.20, 0.10, 0.60,
        0.25, 0.50, 0.25, 0.75,
    };
    nn_real scores[4];
    NNConfig config;
    config.layer_sizes = layers;
    config.layer_count = 3;
    config.seed = 8642;
    config.hidden_activation = NN_ACTIVATION_RELU;
    config.output_activation = NN_ACTIVATION_LINEAR;
    config.loss = NN_LOSS_MSE;

    NeuralNet *regular_net = nn_create_config(&config);
    NeuralNet *cached_net = nn_create_config(&config);
    int ok = 1;

    if (regular_net == NULL || cached_net == NULL) {
        nn_free(regular_net);
        nn_free(cached_net);
        return 0;
    }

    nn_train_softmax_scorer_weighted(regular_net,
                                     candidate_inputs,
                                     candidate_count,
                                     selected_index,
                                     weight,
                                     learning_rate);
    nn_forward_batch(cached_net, candidate_inputs, scores, candidate_count);
    nn_train_softmax_scorer_cached(cached_net,
                                   candidate_count,
                                   selected_index,
                                   weight,
                                   learning_rate);

    for (size_t i = 0; i < nn_parameter_count(regular_net); ++i) {
        nn_real a = nn_test_get_parameter(regular_net, i);
        nn_real b = nn_test_get_parameter(cached_net, i);
        if (!nearly_equal(a, b, (nn_real)1e-6, (nn_real)1e-5)) {
            ok = 0;
            break;
        }
    }

    nn_free(regular_net);
    nn_free(cached_net);
    return ok;
}

static int test_action_scorer_rows_probabilities_sum_to_one(void)
{
    const size_t layers[] = {2, 3, 1};
    const size_t candidate_count = 3;
    const nn_real candidate_inputs[] = {
        0.0, 0.0,
        1.0, 0.0,
        0.0, 1.0,
    };
    nn_real scores[3];
    nn_real probabilities[3];
    nn_real batch_scores[3];
    NNConfig config;
    config.layer_sizes = layers;
    config.layer_count = 3;
    config.seed = 24601;
    config.hidden_activation = NN_ACTIVATION_RELU;
    config.output_activation = NN_ACTIVATION_LINEAR;
    config.loss = NN_LOSS_MSE;

    NeuralNet *net = nn_create_config(&config);
    if (net == NULL) {
        return 0;
    }

    int ok = nn_score_action_rows(net,
                                  candidate_inputs,
                                  candidate_count,
                                  scores,
                                  probabilities);
    nn_forward_batch(net, candidate_inputs, batch_scores, candidate_count);

    nn_real sum = 0;
    for (size_t i = 0; i < candidate_count; ++i) {
        sum += probabilities[i];
        ok = ok && nearly_equal(scores[i], batch_scores[i], (nn_real)1e-6, (nn_real)1e-6);
    }
    ok = ok && nearly_equal(sum, (nn_real)1, (nn_real)1e-6, (nn_real)1e-6);

    nn_free(net);
    return ok;
}

static int test_action_scorer_struct_api_scores_candidates(void)
{
    const size_t layers[] = {2, 3, 1};
    const nn_real features[] = {
        0.0, 0.0,
        1.0, 0.0,
        0.0, 1.0,
    };
    NNActionCandidate candidates[] = {
        {10, &features[0], 0, 0},
        {20, &features[2], 0, 0},
        {30, &features[4], 0, 0},
    };
    NNConfig config;
    config.layer_sizes = layers;
    config.layer_count = 3;
    config.seed = 13579;
    config.hidden_activation = NN_ACTIVATION_TANH;
    config.output_activation = NN_ACTIVATION_LINEAR;
    config.loss = NN_LOSS_MSE;

    NeuralNet *net = nn_create_config(&config);
    if (net == NULL) {
        return 0;
    }

    int ok = nn_score_actions(net, candidates, 3);
    nn_real sum = candidates[0].probability + candidates[1].probability + candidates[2].probability;
    ok = ok && nearly_equal(sum, (nn_real)1, (nn_real)1e-6, (nn_real)1e-6);
    ok = ok && candidates[0].probability > (nn_real)0 &&
         candidates[1].probability > (nn_real)0 &&
         candidates[2].probability > (nn_real)0;

    nn_free(net);
    return ok;
}

static int test_action_scorer_ranking_is_deterministic_under_ties(void)
{
    NNActionCandidate candidates[] = {
        {30, NULL, (nn_real)1, 0},
        {10, NULL, (nn_real)1, 0},
        {20, NULL, (nn_real)2, 0},
        {10, NULL, (nn_real)1, 0},
    };
    size_t ranked[4];

    return nn_rank_actions(candidates, 4, ranked) &&
           ranked[0] == 2 &&
           ranked[1] == 1 &&
           ranked[2] == 3 &&
           ranked[3] == 0;
}

static NNCheckpointMeta checkpoint_meta_fixture(void)
{
    NNCheckpointMeta meta;
    memset(&meta, 0, sizeof(meta));
    meta.feature_encoder_version = 7;
    meta.normalization_version = 3;
    for (size_t i = 0; i < 32; ++i) {
        meta.normalization_hash[i] = (uint8_t)(0x10u + i);
        meta.parent_model_hash[i] = (uint8_t)(0x30u + i);
        meta.training_trace_root[i] = (uint8_t)(0x50u + i);
        meta.training_view_root[i] = (uint8_t)(0x70u + i);
        meta.benchmark_report_hash[i] = (uint8_t)(0x90u + i);
    }
    return meta;
}

static int hashes_equal(const uint8_t a[32], const uint8_t b[32])
{
    return memcmp(a, b, 32) == 0;
}

static int test_checkpoint_round_trip_preserves_inference(void)
{
    const char *path = "/tmp/crlp-checkpoint-roundtrip.nnckpt";
    const size_t layers[] = {3, 4, 2};
    const nn_real input[] = {0.25, 0.50, 0.75};
    nn_real before[2];
    nn_real after[2];
    NNCheckpointMeta meta = checkpoint_meta_fixture();
    NNCheckpointMeta loaded_meta;
    NNConfig config;
    config.layer_sizes = layers;
    config.layer_count = 3;
    config.seed = 4242;
    config.hidden_activation = NN_ACTIVATION_TANH;
    config.output_activation = NN_ACTIVATION_LINEAR;
    config.loss = NN_LOSS_MSE;

    remove(path);
    NeuralNet *net = nn_create_config(&config);
    if (net == NULL) {
        return 0;
    }
    nn_forward(net, input, before);

    int ok = nn_save_checkpoint(net, path, &meta);
    NeuralNet *loaded = nn_load_checkpoint(path, &loaded_meta);
    ok = ok && loaded != NULL &&
         memcmp(&meta, &loaded_meta, sizeof(meta)) == 0 &&
         nn_input_size(loaded) == nn_input_size(net) &&
         nn_output_size(loaded) == nn_output_size(net) &&
         nn_parameter_count(loaded) == nn_parameter_count(net);
    if (loaded != NULL) {
        nn_forward(loaded, input, after);
        ok = ok &&
             nearly_equal(before[0], after[0], (nn_real)1e-6, (nn_real)1e-6) &&
             nearly_equal(before[1], after[1], (nn_real)1e-6, (nn_real)1e-6);
    }

    nn_free(loaded);
    nn_free(net);
    remove(path);
    return ok;
}

static int test_checkpoint_deterministic_bytes(void)
{
    const char *path_a = "/tmp/crlp-checkpoint-deterministic-a.nnckpt";
    const char *path_b = "/tmp/crlp-checkpoint-deterministic-b.nnckpt";
    const size_t layers[] = {2, 3, 1};
    uint8_t hash_a[32];
    uint8_t hash_b[32];
    NNCheckpointMeta meta = checkpoint_meta_fixture();
    NNConfig config;
    config.layer_sizes = layers;
    config.layer_count = 3;
    config.seed = 8080;
    config.hidden_activation = NN_ACTIVATION_RELU;
    config.output_activation = NN_ACTIVATION_LINEAR;
    config.loss = NN_LOSS_MSE;

    remove(path_a);
    remove(path_b);
    NeuralNet *net = nn_create_config(&config);
    if (net == NULL) {
        return 0;
    }

    int ok = nn_save_checkpoint(net, path_a, &meta) &&
             nn_save_checkpoint(net, path_b, &meta) &&
             nn_hash_checkpoint_file(path_a, hash_a) &&
             nn_hash_checkpoint_file(path_b, hash_b) &&
             hashes_equal(hash_a, hash_b);

    nn_free(net);
    remove(path_a);
    remove(path_b);
    return ok;
}

static int test_checkpoint_hash_changes_with_weight(void)
{
    const char *path_a = "/tmp/crlp-checkpoint-hash-a.nnckpt";
    const char *path_b = "/tmp/crlp-checkpoint-hash-b.nnckpt";
    const size_t layers[] = {2, 3, 1};
    uint8_t hash_a[32];
    uint8_t hash_b[32];
    NNCheckpointMeta meta = checkpoint_meta_fixture();
    NNConfig config;
    config.layer_sizes = layers;
    config.layer_count = 3;
    config.seed = 9090;
    config.hidden_activation = NN_ACTIVATION_RELU;
    config.output_activation = NN_ACTIVATION_LINEAR;
    config.loss = NN_LOSS_MSE;

    remove(path_a);
    remove(path_b);
    NeuralNet *net = nn_create_config(&config);
    if (net == NULL) {
        return 0;
    }

    nn_real original = nn_test_get_parameter(net, 0);
    int ok = nn_save_checkpoint(net, path_a, &meta) &&
             nn_test_set_parameter(net, 0, original + (nn_real)0.125) &&
             nn_save_checkpoint(net, path_b, &meta) &&
             nn_hash_checkpoint_file(path_a, hash_a) &&
             nn_hash_checkpoint_file(path_b, hash_b) &&
             !hashes_equal(hash_a, hash_b);

    nn_free(net);
    remove(path_a);
    remove(path_b);
    return ok;
}

static int test_checkpoint_hash_matches_sha256_vector(void)
{
    const char *path = "/tmp/crlp-checkpoint-sha256-vector.txt";
    const uint8_t expected[32] = {
        0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea,
        0x41, 0x41, 0x40, 0xde, 0x5d, 0xae, 0x22, 0x23,
        0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c,
        0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00, 0x15, 0xad,
    };
    uint8_t actual[32];

    remove(path);
    FILE *file = fopen(path, "wb");
    if (file == NULL) {
        return 0;
    }
    int ok = fwrite("abc", 1, 3, file) == 3;
    ok = fclose(file) == 0 && ok;
    ok = ok && nn_hash_checkpoint_file(path, actual) && hashes_equal(actual, expected);
    remove(path);
    return ok;
}

static int test_trace_jsonl_writer_outputs_valid_shape(void)
{
    const char *path = "/tmp/crlp-trace-writer.jsonl";
    const uint32_t action_ids[] = {10, 20};
    const nn_real scores[] = {(nn_real)1.5, (nn_real)-0.25};
    const nn_real probabilities[] = {(nn_real)0.85, (nn_real)0.15};
    char contents[2048];
    NNTraceRecord record;
    record.task_name = "task\"with\nescape";
    record.model_hash_hex = "abc123";
    record.state_hash = UINT64_C(0x1234);
    record.feature_encoder_version = 5;
    record.candidate_count = 2;
    record.action_ids = action_ids;
    record.scores = scores;
    record.probabilities = probabilities;
    record.selected_index = 0;
    record.target_index = 1;
    record.loss = (nn_real)0.42;
    record.verifier_result = "corrected";
    record.verifier_cost_json = "{\"checks\":3}";
    record.seed = 99;

    remove(path);
    FILE *file = fopen(path, "wb");
    if (file == NULL) {
        return 0;
    }
    int ok = nn_trace_write_jsonl(file, &record);
    ok = fclose(file) == 0 && ok;

    file = fopen(path, "rb");
    if (file == NULL) {
        remove(path);
        return 0;
    }
    size_t got = fread(contents, 1, sizeof(contents) - 1, file);
    contents[got] = '\0';
    ok = fclose(file) == 0 && ok;

    ok = ok &&
         strstr(contents, "\"schema\":\"crlp.trace.v1\"") != NULL &&
         strstr(contents, "\"task\":\"task\\\"with\\nescape\"") != NULL &&
         strstr(contents, "\"state_hash\":\"0000000000001234\"") != NULL &&
         strstr(contents, "\"model_hash\":\"abc123\"") != NULL &&
         strstr(contents, "\"action_ids\":[10,20]") != NULL &&
         strstr(contents, "\"verifier_cost\":{\"checks\":3}") != NULL &&
         contents[got > 0 ? got - 1 : 0] == '\n';

    remove(path);
    return ok;
}

static int test_trace_jsonl_writer_rejects_invalid_payloads(void)
{
    const uint32_t action_ids[] = {10, 20};
    nn_real scores[] = {(nn_real)1.5, (nn_real)-0.25};
    nn_real probabilities[] = {(nn_real)0.85, (nn_real)0.15};
    NNTraceRecord record;
    record.task_name = "task";
    record.model_hash_hex = "abc123";
    record.state_hash = UINT64_C(0x1234);
    record.feature_encoder_version = 5;
    record.candidate_count = 2;
    record.action_ids = action_ids;
    record.scores = scores;
    record.probabilities = probabilities;
    record.selected_index = 0;
    record.target_index = 1;
    record.loss = (nn_real)0.42;
    record.verifier_result = "corrected";
    record.verifier_cost_json = "{\"checks\":3}";
    record.seed = 99;

    FILE *file = tmpfile();
    if (file == NULL) {
        return 0;
    }

    int ok = 1;
    scores[1] = (nn_real)INFINITY;
    ok = ok && !nn_trace_write_jsonl(file, &record);
    scores[1] = (nn_real)-0.25;

    probabilities[0] = (nn_real)NAN;
    ok = ok && !nn_trace_write_jsonl(file, &record);
    probabilities[0] = (nn_real)0.85;

    record.loss = (nn_real)INFINITY;
    ok = ok && !nn_trace_write_jsonl(file, &record);
    record.loss = (nn_real)0.42;

    record.verifier_cost_json = "[]";
    ok = ok && !nn_trace_write_jsonl(file, &record);
    record.verifier_cost_json = "{\"checks\":3}";

    record.verifier_cost_json = "{\n\"checks\":3}";
    ok = ok && !nn_trace_write_jsonl(file, &record);
    record.verifier_cost_json = "{\"checks\":3}";

    ok = ok && nn_trace_write_jsonl(file, &record);
    ok = fclose(file) == 0 && ok;
    return ok;
}

static int test_prime_gap_features_are_divisibility_flags(void)
{
    nn_real inputs[ACTION_COUNT * FEATURE_COUNT];
    nn_real masked[ACTION_COUNT * FEATURE_COUNT];
    fill_candidate_features(inputs, 11, log(1000.0));
    fill_candidate_features_masked(masked, 11, log(1000.0), &FEATURE_MASK_GAP_ONLY);

    const nn_real *gap2 = &inputs[0 * FEATURE_COUNT];
    const nn_real *gap4 = &inputs[1 * FEATURE_COUNT];
    const nn_real *overflow = &inputs[OVERFLOW_ACTION * FEATURE_COUNT];
    const nn_real *masked_gap4 = &masked[1 * FEATURE_COUNT];
    const nn_real *masked_overflow = &masked[OVERFLOW_ACTION * FEATURE_COUNT];

    return gap4[2] == (nn_real)0 && gap4[3] == (nn_real)1 &&
           gap4[4] == (nn_real)1 && gap4[5] == (nn_real)0 &&
           gap2[7] == (nn_real)0 && overflow[2] == (nn_real)1 &&
           masked_gap4[0] == (nn_real)0 && masked_gap4[1] > (nn_real)0 &&
           masked_gap4[3] == (nn_real)0 && masked_overflow[2] == (nn_real)1;
}

static int test_prime_gap_raw_residue_features_are_not_divisibility_flags(void)
{
    nn_real inputs[ACTION_COUNT * FEATURE_COUNT];
    fill_candidate_features_masked(inputs, 11, log(1000.0), &FEATURE_MASK_RAW_RESIDUE);

    const nn_real *gap2 = &inputs[0 * FEATURE_COUNT];
    const nn_real *gap4 = &inputs[1 * FEATURE_COUNT];

    return nearly_equal(gap2[3], (nn_real)(1.0 / 3.0), (nn_real)1e-6, (nn_real)1e-6) &&
           gap4[3] == (nn_real)0 &&
           gap4[4] == (nn_real)0 &&
           gap4[3] != (nn_real)1 &&
           gap2[1] > (nn_real)0 &&
           gap2[2] == (nn_real)0;
}

static int test_prime_gap_frequency_policy_is_smoothed(void)
{
    GapDataset dataset;
    size_t ranked[ACTION_COUNT];
    nn_real probabilities[ACTION_COUNT];
    double sum = 0.0;
    int ok;

    memset(&dataset, 0, sizeof(dataset));
    ok = append_gap(&dataset, 11, 2) && append_gap(&dataset, 13, 4) &&
         append_gap(&dataset, 17, 4);
    if (!ok) {
        free_gap_dataset(&dataset);
        return 0;
    }

    build_frequency_policy(&dataset, ranked, probabilities);
    for (size_t i = 0; i < ACTION_COUNT; ++i) {
        sum += (double)probabilities[i];
    }

    ok = ranked[0] == gap_to_action(4) &&
         probabilities[gap_to_action(4)] > probabilities[gap_to_action(2)] &&
         probabilities[OVERFLOW_ACTION] > (nn_real)0 &&
         fabs(sum - 1.0) < 1e-5;

    free_gap_dataset(&dataset);
    return ok;
}

static int test_prime_gap_residue_conditioned_policy_scores_candidates(void)
{
    GapDataset dataset;
    ConditionedPolicy policy;
    size_t ranked[ACTION_COUNT];
    nn_real probabilities[ACTION_COUNT];
    double sum = 0.0;
    int ok;

    memset(&dataset, 0, sizeof(dataset));
    memset(&policy, 0, sizeof(policy));
    ok = append_gap(&dataset, 11, 2) && append_gap(&dataset, 17, 2) &&
         append_gap(&dataset, 29, 2);
    if (!ok) {
        free_gap_dataset(&dataset);
        return 0;
    }

    ok = build_residue_conditioned_policy(&dataset, &policy);
    if (ok) {
        score_residue_conditioned_policy(&policy, 11, ranked, probabilities);
        for (size_t i = 0; i < ACTION_COUNT; ++i) {
            sum += (double)probabilities[i];
        }

        ok = ranked[0] == gap_to_action(2) &&
             candidate_signature(11, gap_to_action(4)) != 0 &&
             probabilities[ranked[0]] > probabilities[gap_to_action(4)] &&
             fabs(sum - 1.0) < 1e-5;
    }

    free_conditioned_policy(&policy);
    free_gap_dataset(&dataset);
    return ok;
}

static int test_prime_gap_wheel_verifier_accounting(void)
{
    Sieve sieve;
    size_t ranked[ACTION_COUNT];
    VerifierWork direct;
    VerifierWork wheel;
    VerifierWork assisted;
    BlockWork block;
    BlockWork assisted_block;
    int ok = 1;

    if (!build_sieve(100, &sieve)) {
        return 0;
    }

    for (size_t i = 0; i < ACTION_COUNT; ++i) {
        ranked[i] = i;
    }
    ranked[0] = 2;
    ranked[2] = 0;

    ok = ok && proposal_steps_with_topk(ranked, 1, 0) == 2;

    direct = direct_scan_work(6);
    wheel = wheel_scan_work(&sieve, 23, 6);
    assisted = assisted_wheel_work(&sieve, 11, 2, ranked, 1);

    ok = ok && direct.exact_tests == 3;
    ok = ok && wheel.exact_tests == 1;
    ok = ok && wheel.mod_tests > 0;
    ok = ok && assisted.exact_tests > wheel_scan_work(&sieve, 11, 2).exact_tests;

    block = block_scan_work(&sieve, 23, 6);
    assisted_block = assisted_block_work(&sieve, 11, 2, ranked, 1);
    ok = ok && block.survivor_checks == 3;
    ok = ok && block.marks >= 2;
    ok = ok && block_work_cost(assisted_block) > block_work_cost(block_scan_work(&sieve, 11, 2));

    free_sieve(&sieve);
    return ok;
}

static int test_prime_gap_dataset_tracks_overflow_actions(void)
{
    Sieve sieve;
    GapDataset train;
    GapDataset test;
    int ok = 1;

    if (!build_sieve(200 + G_MAX, &sieve)) {
        return 0;
    }

    if (!build_gap_datasets(&sieve, 50, 200, &train, &test)) {
        free_sieve(&sieve);
        return 0;
    }

    ok = ok && train.total_seen == train.count;
    ok = ok && test.total_seen == test.count;
    ok = ok && train.overflow_count <= train.count;
    ok = ok && test.overflow_count <= test.count;
    ok = ok && gap_to_action(G_MAX + 2) == OVERFLOW_ACTION;
    ok = ok && train.count > 0 && test.count > 0;

    free_gap_dataset(&train);
    free_gap_dataset(&test);
    free_sieve(&sieve);
    return ok;
}

static int test_prime_gap_train_calibration_split(void)
{
    GapDataset source;
    GapDataset fit;
    GapDataset calibration;
    int ok = 1;

    memset(&source, 0, sizeof(source));
    memset(&fit, 0, sizeof(fit));
    memset(&calibration, 0, sizeof(calibration));

    for (uint32_t i = 0; i < 10; ++i) {
        ok = ok && append_gap(&source, 11 + i * 2, 2 + 2 * (i % 3));
    }
    if (!ok) {
        free_gap_dataset(&source);
        return 0;
    }

    ok = split_train_calibration(&source, &fit, &calibration);
    ok = ok && fit.count == 8 && calibration.count == 2;
    ok = ok && fit.prime[0] == source.prime[0] && fit.prime[7] == source.prime[7];
    ok = ok && calibration.prime[0] == source.prime[8] &&
         calibration.prime[1] == source.prime[9];
    ok = ok && fit.total_seen == fit.count && calibration.total_seen == calibration.count;

    free_gap_dataset(&calibration);
    free_gap_dataset(&fit);
    free_gap_dataset(&source);
    return ok;
}

static int test_prime_gap_wheel_survivor_features_include_target(void)
{
    nn_real inputs[ACTION_COUNT * FEATURE_COUNT];
    size_t target_index = ACTION_COUNT;
    size_t true_action = gap_to_action(2);
    size_t count = fill_wheel_survivor_features(inputs,
                                                11,
                                                true_action,
                                                log(1000.0),
                                                &FEATURE_MASK_FULL,
                                                &target_index);
    const nn_real *target = &inputs[target_index * FEATURE_COUNT];

    return count > 0 && count < ACTION_COUNT && target_index < count &&
           action_survives_feature_wheel(11, true_action) &&
           !action_survives_feature_wheel(11, gap_to_action(4)) &&
           nearly_equal(target[1], (nn_real)(2.0 / (double)G_MAX), (nn_real)1e-6, (nn_real)1e-6) &&
           target[2] == (nn_real)0 && target[3] == (nn_real)0;
}

static int test_prime_gap_wheel_frequency_policy_masks_rejected_actions(void)
{
    GapDataset dataset;
    size_t ranked[ACTION_COUNT];
    nn_real base_probabilities[ACTION_COUNT];
    nn_real wheel_probabilities[ACTION_COUNT];
    double sum = 0.0;
    int ok;

    memset(&dataset, 0, sizeof(dataset));
    ok = append_gap(&dataset, 11, 4) && append_gap(&dataset, 13, 4) &&
         append_gap(&dataset, 17, 2);
    if (!ok) {
        free_gap_dataset(&dataset);
        return 0;
    }

    build_frequency_policy(&dataset, ranked, base_probabilities);
    score_wheel_frequency_policy(base_probabilities, 11, ranked, wheel_probabilities);
    for (size_t i = 0; i < ACTION_COUNT; ++i) {
        sum += (double)wheel_probabilities[i];
    }

    ok = wheel_probabilities[gap_to_action(4)] == (nn_real)0 &&
         ranked[0] != gap_to_action(4) &&
         fabs(sum - 1.0) < 1e-5;

    free_gap_dataset(&dataset);
    return ok;
}

static int test_prime_gap_interval_targets_include_all_primes(void)
{
    Sieve sieve;
    nn_real targets[EXACT_ACTION_COUNT];
    double sum = 0.0;
    size_t positives;
    int ok;

    if (!build_sieve(300, &sieve)) {
        return 0;
    }

    positives = build_interval_prime_targets(&sieve, 11, EXACT_ACTION_COUNT, targets);
    for (size_t action = 0; action < EXACT_ACTION_COUNT; ++action) {
        sum += (double)targets[action];
    }

    ok = positives > 1 &&
         targets[gap_to_action(2)] > (nn_real)0 &&
         targets[gap_to_action(4)] == (nn_real)0 &&
         targets[gap_to_action(6)] > (nn_real)0 &&
         fabs(sum - 1.0) < 1e-5;

    free_sieve(&sieve);
    return ok;
}

static int test_prime_gap_interval_wheel_frequency_masks_rejected_actions(void)
{
    nn_real base_probabilities[EXACT_ACTION_COUNT];
    nn_real wheel_probabilities[EXACT_ACTION_COUNT];
    size_t ranked[EXACT_ACTION_COUNT];
    double sum = 0.0;

    for (size_t action = 0; action < EXACT_ACTION_COUNT; ++action) {
        base_probabilities[action] = (nn_real)1 / (nn_real)EXACT_ACTION_COUNT;
    }

    score_interval_wheel_frequency_policy(
        base_probabilities, 11, EXACT_ACTION_COUNT, ranked, wheel_probabilities);
    for (size_t action = 0; action < EXACT_ACTION_COUNT; ++action) {
        sum += (double)wheel_probabilities[action];
    }

    return wheel_probabilities[gap_to_action(4)] == (nn_real)0 &&
           ranked[0] != gap_to_action(4) &&
           fabs(sum - 1.0) < 1e-5;
}

static int test_prime_gap_interval_target_cache_matches_direct_targets(void)
{
    Sieve sieve = {0, 0, 0, 0};
    GapDataset train = {0, 0, 0, 0, 0, 0};
    GapDataset test = {0, 0, 0, 0, 0, 0};
    IntervalTargetCache cache = {0, 0, 0, 0};
    nn_real direct[EXACT_ACTION_COUNT];
    int ok = 1;

    if (!build_sieve(300 + G_MAX, &sieve)) {
        return 0;
    }
    if (!build_gap_datasets(&sieve, 50, 300, &train, &test)) {
        free_sieve(&sieve);
        return 0;
    }
    if (!build_interval_eval_target_cache(&sieve, &test, 5, EXACT_ACTION_COUNT, &cache)) {
        free_gap_dataset(&train);
        free_gap_dataset(&test);
        free_sieve(&sieve);
        return 0;
    }

    for (size_t sample = 0; sample < cache.sample_count; ++sample) {
        size_t index = eval_sample_index(&test, sample, cache.sample_count);
        size_t positives =
            build_interval_prime_targets(&sieve, test.prime[index], EXACT_ACTION_COUNT, direct);
        const nn_real *cached = &cache.targets[sample * cache.candidate_count];

        ok = ok && positives == cache.positive_counts[sample];
        for (size_t action = 0; action < cache.candidate_count; ++action) {
            ok = ok && fabs((double)(direct[action] - cached[action])) < 1e-7;
        }
    }

    free_interval_target_cache(&cache);
    free_gap_dataset(&train);
    free_gap_dataset(&test);
    free_sieve(&sieve);
    return ok;
}

static int test_prime_gap_profile_accumulates_phase_time(void)
{
    PrimeGapProfile profile;

    prime_gap_profile_reset(&profile);
    prime_gap_profile_set(&profile);
    prime_gap_profile_add(PRIME_GAP_PROFILE_FEATURES, 0.25);
    prime_gap_profile_add(PRIME_GAP_PROFILE_FEATURES, 0.50);
    prime_gap_profile_set(NULL);

    return profile.calls[PRIME_GAP_PROFILE_FEATURES] == 2 &&
           fabs(profile.seconds[PRIME_GAP_PROFILE_FEATURES] - 0.75) < 1e-12 &&
           profile.calls[PRIME_GAP_PROFILE_FORWARD] == 0;
}

static int test_prime_gap_wide_interval_targets_fit_buffer(void)
{
    Sieve sieve;
    nn_real targets[INTERVAL_MAX_ACTION_COUNT];
    size_t positives;
    double sum = 0.0;
    int ok;

    if (!build_sieve(11 + INTERVAL_MAX_GAP, &sieve)) {
        return 0;
    }

    positives = build_interval_prime_targets(
        &sieve, 11, INTERVAL_MAX_ACTION_COUNT, targets);
    for (size_t action = 0; action < INTERVAL_MAX_ACTION_COUNT; ++action) {
        sum += (double)targets[action];
    }

    ok = positives > 0 && fabs(sum - 1.0) < 1e-5 &&
         targets[INTERVAL_MAX_ACTION_COUNT - 1] >= (nn_real)0;

    free_sieve(&sieve);
    return ok;
}

int main(void)
{
    check(test_forward_batch_matches_scalar(), "batched forward matches scalar");
    check(test_gradient_buffers_are_lazy(), "gradient buffers are lazy");
    check(test_batch_size_one_matches_sample_training(), "batch size 1 matches sample training");
    check(test_batched_gradient_matches_finite_difference(),
          "batched gradient matches finite difference");
    check(test_softmax_cross_entropy_config_learns(),
          "softmax cross-entropy config learns logits");
    check(test_softmax_scorer_gradient_matches_finite_difference(),
          "softmax scorer gradient matches finite difference");
    check(test_softmax_scorer_entropy_gradient_matches_finite_difference(),
          "softmax scorer entropy gradient matches finite difference");
    check(test_softmax_scorer_targets_gradient_matches_finite_difference(),
          "softmax scorer target distribution gradient matches finite difference");
    check(test_softmax_cross_entropy_requires_linear_output(),
          "softmax cross-entropy requires linear output");
    check(test_softmax_scorer_cached_matches_regular(),
          "softmax scorer cached update matches regular update");
    check(test_action_scorer_rows_probabilities_sum_to_one(),
          "action scorer rows produce normalized probabilities");
    check(test_action_scorer_struct_api_scores_candidates(),
          "action scorer struct API scores candidates");
    check(test_action_scorer_ranking_is_deterministic_under_ties(),
          "action scorer ranking is deterministic under ties");
    check(test_checkpoint_round_trip_preserves_inference(),
          "checkpoint round trip preserves inference");
    check(test_checkpoint_deterministic_bytes(),
          "checkpoint writes deterministic bytes");
    check(test_checkpoint_hash_changes_with_weight(),
          "checkpoint hash changes when weights change");
    check(test_checkpoint_hash_matches_sha256_vector(),
          "checkpoint hash matches SHA-256 test vector");
    check(test_trace_jsonl_writer_outputs_valid_shape(),
          "trace JSONL writer emits valid record shape");
    check(test_trace_jsonl_writer_rejects_invalid_payloads(),
          "trace JSONL writer rejects invalid payloads");
    check(test_prime_gap_features_are_divisibility_flags(),
          "prime-gap features are divisibility flags");
    check(test_prime_gap_raw_residue_features_are_not_divisibility_flags(),
          "prime-gap raw residue features are not divisibility flags");
    check(test_prime_gap_frequency_policy_is_smoothed(),
          "prime-gap frequency policy is smoothed");
    check(test_prime_gap_residue_conditioned_policy_scores_candidates(),
          "prime-gap residue-conditioned policy scores candidates");
    check(test_prime_gap_wheel_verifier_accounting(),
          "prime-gap wheel verifier accounting");
    check(test_prime_gap_dataset_tracks_overflow_actions(),
          "prime-gap dataset tracks overflow actions");
    check(test_prime_gap_train_calibration_split(),
          "prime-gap train/calibration split is chronological");
    check(test_prime_gap_wheel_survivor_features_include_target(),
          "prime-gap wheel-survivor features include target");
    check(test_prime_gap_wheel_frequency_policy_masks_rejected_actions(),
          "prime-gap wheel-frequency masks rejected actions");
    check(test_prime_gap_interval_targets_include_all_primes(),
          "prime-gap interval targets include all primes");
    check(test_prime_gap_interval_wheel_frequency_masks_rejected_actions(),
          "prime-gap interval wheel-frequency masks rejected actions");
    check(test_prime_gap_interval_target_cache_matches_direct_targets(),
          "prime-gap interval target cache matches direct targets");
    check(test_prime_gap_profile_accumulates_phase_time(),
          "prime-gap profile accumulates phase time");
    check(test_prime_gap_wide_interval_targets_fit_buffer(),
          "prime-gap wide interval targets fit buffer");

    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
