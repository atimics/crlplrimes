#include "nn.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "../src/nn_tasks_prime_gap.c"
#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

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

static int test_clone_matches_and_detaches(void)
{
    const size_t layers[] = {3, 4, 2};
    NeuralNet *net = nn_create(layers, 3, 2026);
    NeuralNet *clone = NULL;
    nn_real input[] = {(nn_real)0.1, (nn_real)0.5, (nn_real)0.9};
    nn_real target[] = {(nn_real)0.25, (nn_real)0.75};
    nn_real original_before[2];
    nn_real clone_before[2];
    nn_real original_after[2];
    nn_real clone_after[2];
    int ok = 1;

    if (net == NULL) {
        return 0;
    }

    clone = nn_clone(net);
    if (clone == NULL) {
        nn_free(net);
        return 0;
    }

    nn_forward(net, input, original_before);
    nn_forward(clone, input, clone_before);
    ok = ok && nearly_equal(original_before[0], clone_before[0], (nn_real)1e-6, (nn_real)1e-6) &&
         nearly_equal(original_before[1], clone_before[1], (nn_real)1e-6, (nn_real)1e-6);

    nn_train_sample(clone, input, target, (nn_real)0.1);
    nn_forward(net, input, original_after);
    nn_forward(clone, input, clone_after);
    ok = ok && nearly_equal(original_before[0], original_after[0], (nn_real)1e-6, (nn_real)1e-6) &&
         nearly_equal(original_before[1], original_after[1], (nn_real)1e-6, (nn_real)1e-6);
    ok = ok && (!nearly_equal(original_after[0], clone_after[0], (nn_real)1e-8, (nn_real)1e-8) ||
                !nearly_equal(original_after[1], clone_after[1], (nn_real)1e-8, (nn_real)1e-8));

    nn_free(clone);
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
    NeuralNet *invalidated_net = nn_create_config(&config);
    int ok = 1;

    if (regular_net == NULL || cached_net == NULL || invalidated_net == NULL) {
        nn_free(regular_net);
        nn_free(cached_net);
        nn_free(invalidated_net);
        return 0;
    }

    nn_forward_batch(invalidated_net, candidate_inputs, scores, candidate_count);
    nn_forward(invalidated_net, candidate_inputs, scores);
    ok = ok && isnan((double)nn_train_softmax_scorer_cached(invalidated_net,
                                                            candidate_count,
                                                            selected_index,
                                                            weight,
                                                            learning_rate));

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
    ok = ok && isnan((double)nn_train_softmax_scorer_cached(cached_net,
                                                            candidate_count,
                                                            selected_index,
                                                            weight,
                                                            learning_rate));

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
    nn_free(invalidated_net);
    return ok;
}

static int test_policy_regularized_cached_matches_policy_update(void)
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
    const nn_real reference[] = {(nn_real)0.40, (nn_real)0.30, (nn_real)0.20, (nn_real)0.10};
    nn_real scores[4];
    NNConfig config;
    config.layer_sizes = layers;
    config.layer_count = 3;
    config.seed = 9753;
    config.hidden_activation = NN_ACTIVATION_RELU;
    config.output_activation = NN_ACTIVATION_LINEAR;
    config.loss = NN_LOSS_MSE;

    NeuralNet *regular_net = nn_create_config(&config);
    NeuralNet *policy_net = nn_create_config(&config);
    NeuralNet *regularized_net = nn_create_config(&config);
    int ok = 1;

    if (regular_net == NULL || policy_net == NULL || regularized_net == NULL) {
        nn_free(regular_net);
        nn_free(policy_net);
        nn_free(regularized_net);
        return 0;
    }

    nn_train_softmax_scorer_weighted(regular_net,
                                     candidate_inputs,
                                     candidate_count,
                                     selected_index,
                                     weight,
                                     learning_rate);
    nn_forward_batch(policy_net, candidate_inputs, scores, candidate_count);
    nn_train_softmax_scorer_policy_regularized_cached(policy_net,
                                                      candidate_count,
                                                      selected_index,
                                                      weight,
                                                      NULL,
                                                      (nn_real)0,
                                                      (nn_real)0,
                                                      learning_rate);

    for (size_t i = 0; i < nn_parameter_count(regular_net); ++i) {
        nn_real a = nn_test_get_parameter(regular_net, i);
        nn_real b = nn_test_get_parameter(policy_net, i);
        if (!nearly_equal(a, b, (nn_real)1e-6, (nn_real)1e-5)) {
            ok = 0;
            break;
        }
    }

    nn_forward_batch(regularized_net, candidate_inputs, scores, candidate_count);
    nn_real loss = nn_train_softmax_scorer_policy_regularized_cached(regularized_net,
                                                                     candidate_count,
                                                                     selected_index,
                                                                     weight,
                                                                     reference,
                                                                     (nn_real)0.05,
                                                                     (nn_real)0.01,
                                                                     learning_rate);
    ok = ok && !isnan((double)loss);
    ok = ok && isnan((double)nn_train_softmax_scorer_policy_regularized_cached(
                         regularized_net,
                         candidate_count,
                         selected_index,
                         weight,
                         reference,
                         (nn_real)0.05,
                         (nn_real)0.01,
                         learning_rate));

    nn_free(regular_net);
    nn_free(policy_net);
    nn_free(regularized_net);
    return ok;
}

static int files_match(const char *a_path, const char *b_path)
{
    FILE *a = fopen(a_path, "rb");
    FILE *b = fopen(b_path, "rb");
    int ok = 1;

    if (a == NULL || b == NULL) {
        if (a != NULL) {
            fclose(a);
        }
        if (b != NULL) {
            fclose(b);
        }
        return 0;
    }

    for (;;) {
        unsigned char a_buffer[256];
        unsigned char b_buffer[256];
        size_t a_count = fread(a_buffer, 1, sizeof(a_buffer), a);
        size_t b_count = fread(b_buffer, 1, sizeof(b_buffer), b);
        if (a_count != b_count || memcmp(a_buffer, b_buffer, a_count) != 0) {
            ok = 0;
            break;
        }
        if (a_count < sizeof(a_buffer)) {
            ok = feof(a) && feof(b) && !ferror(a) && !ferror(b);
            break;
        }
    }

    fclose(a);
    fclose(b);
    return ok;
}

static int hashes_equal(const char *a, const char *b)
{
    return strcmp(a, b) == 0;
}

static int test_action_scorer_api_scores_probabilities_and_ranking(void)
{
    const size_t layers[] = {3, 4, 1};
    const nn_real candidate_inputs[] = {
        0.10, 0.20, 0.30,
        0.30, 0.10, 0.40,
        0.80, 0.20, 0.10,
    };
    NNConfig config;
    config.layer_sizes = layers;
    config.layer_count = 3;
    config.seed = 1122;
    config.hidden_activation = NN_ACTIVATION_RELU;
    config.output_activation = NN_ACTIVATION_LINEAR;
    config.loss = NN_LOSS_MSE;

    NeuralNet *net = nn_create_config(&config);
    nn_real scores[3];
    nn_real probabilities[3];
    size_t ranked[4];
    nn_real tied_scores[] = {(nn_real)1, (nn_real)2, (nn_real)2, (nn_real)0};
    int ok = 1;

    if (net == NULL) {
        return 0;
    }

    ok = ok && nn_score_action_rows(net, candidate_inputs, 3, scores, probabilities);
    nn_real sum = probabilities[0] + probabilities[1] + probabilities[2];
    ok = ok && nearly_equal(sum, (nn_real)1, (nn_real)1e-6, (nn_real)1e-6);
    ok = ok && nn_rank_action_scores(tied_scores, 4, ranked);
    ok = ok && ranked[0] == 1 && ranked[1] == 2 && ranked[2] == 0 && ranked[3] == 3;

    nn_free(net);
    return ok;
}

static int test_pairwise_action_ranker_orders_target(void)
{
    const size_t layers[] = {3, 5, 1};
    const nn_real candidate_inputs[] = {
        1.00, 0.00, 0.00,
        0.00, 1.00, 0.00,
        0.00, 0.00, 1.00,
    };
    NNConfig config;
    config.layer_sizes = layers;
    config.layer_count = 3;
    config.seed = 7711;
    config.hidden_activation = NN_ACTIVATION_RELU;
    config.output_activation = NN_ACTIVATION_LINEAR;
    config.loss = NN_LOSS_MSE;

    NeuralNet *net = nn_create_config(&config);
    nn_real scores[3];
    nn_real probabilities[3];
    size_t ranked[3];
    int ok = 1;

    if (net == NULL) {
        return 0;
    }

    for (size_t step = 0; step < 200; ++step) {
        nn_real loss = nn_train_pairwise_action_ranker(net,
                                                       candidate_inputs,
                                                       3,
                                                       2,
                                                       (nn_real)0.5,
                                                       (nn_real)0.02);
        if (isnan((double)loss)) {
            ok = 0;
            break;
        }
    }

    ok = ok && nn_score_action_rows(net, candidate_inputs, 3, scores, probabilities) &&
         nn_rank_action_scores(scores, 3, ranked);
    ok = ok && ranked[0] == 2 && scores[2] > scores[0] && scores[2] > scores[1];
    ok = ok && isnan((double)nn_train_pairwise_action_ranker(net,
                                                             candidate_inputs,
                                                             3,
                                                             2,
                                                             (nn_real)0,
                                                             (nn_real)0.02));

    nn_free(net);
    return ok;
}

static int test_checkpoint_roundtrip_hash_and_invalid_files(void)
{
    const char *path_a = "/tmp/crlplrimes_test_a.nnckpt";
    const char *path_b = "/tmp/crlplrimes_test_b.nnckpt";
    const char *path_c = "/tmp/crlplrimes_test_c.nnckpt";
    const char *path_bad = "/tmp/crlplrimes_test_bad.nnckpt";
    const char *path_bad_size = "/tmp/crlplrimes_test_bad_size.nnckpt";
    const size_t layers[] = {3, 5, 2};
    const nn_real input[] = {(nn_real)0.2, (nn_real)0.4, (nn_real)0.6};
    nn_real before[2];
    nn_real after[2];
    char hash_a[65];
    char hash_b[65];
    char hash_c[65];
    char hash_d[65];
    NNCheckpointMeta meta;
    NNCheckpointMeta loaded_meta;
    NNConfig config;
    NeuralNet *net;
    NeuralNet *loaded;
    int ok = 1;

    config.layer_sizes = layers;
    config.layer_count = 3;
    config.seed = 4242;
    config.hidden_activation = NN_ACTIVATION_TANH;
    config.output_activation = NN_ACTIVATION_LINEAR;
    config.loss = NN_LOSS_MSE;
    net = nn_create_config(&config);
    if (net == NULL) {
        return 0;
    }

    memset(&meta, 0, sizeof(meta));
    meta.checkpoint_version = 1;
    meta.feature_encoder_version = 7;
    strcpy(meta.parent_model_hash, "unknown");
    strcpy(meta.training_trace_root, "trace-root");
    strcpy(meta.benchmark_report_hash, "bench-root");
    strcpy(meta.notes, "fixture checkpoint");
    strcpy(meta.feature_set, "fixture-features");

    nn_forward(net, input, before);
    ok = ok && nn_save_checkpoint(net, path_a, &meta);
    ok = ok && nn_save_checkpoint(net, path_b, &meta);
    ok = ok && files_match(path_a, path_b);
    ok = ok && nn_hash_checkpoint_hex(path_a, hash_a);
    ok = ok && nn_hash_checkpoint_hex(path_b, hash_b);
    ok = ok && hashes_equal(hash_a, hash_b);

    loaded = nn_load_checkpoint(path_a, &loaded_meta);
    ok = ok && loaded != NULL;
    if (loaded != NULL) {
        nn_forward(loaded, input, after);
        ok = ok && nearly_equal(before[0], after[0], (nn_real)1e-6, (nn_real)1e-6) &&
             nearly_equal(before[1], after[1], (nn_real)1e-6, (nn_real)1e-6);
        ok = ok && loaded_meta.checkpoint_version == 1 &&
             loaded_meta.feature_encoder_version == 7 &&
             strcmp(loaded_meta.training_trace_root, "trace-root") == 0 &&
             strcmp(loaded_meta.feature_set, "fixture-features") == 0;
        nn_free(loaded);
    }

    nn_test_set_parameter(net, 0, nn_test_get_parameter(net, 0) + (nn_real)0.125);
    ok = ok && nn_save_checkpoint(net, path_c, &meta);
    ok = ok && nn_hash_checkpoint_hex(path_c, hash_c);
    ok = ok && !hashes_equal(hash_a, hash_c);

    strcpy(meta.notes, "metadata changed");
    ok = ok && nn_save_checkpoint(net, path_b, &meta);
    ok = ok && nn_hash_checkpoint_hex(path_b, hash_d);
    ok = ok && !hashes_equal(hash_c, hash_d);

    FILE *bad = fopen(path_bad, "wb");
    if (bad != NULL) {
        fputs("bad checkpoint", bad);
        fclose(bad);
    }
    ok = ok && nn_load_checkpoint(path_bad, NULL) == NULL;

    ok = ok && nn_save_checkpoint(net, path_bad_size, &meta);
    FILE *bad_size = fopen(path_bad_size, "r+b");
    if (bad_size != NULL) {
        unsigned char zero_size[8] = {0};
        fseek(bad_size, 24, SEEK_SET);
        fwrite(zero_size, 1, sizeof(zero_size), bad_size);
        fclose(bad_size);
    }
    ok = ok && nn_load_checkpoint(path_bad_size, NULL) == NULL;

    remove(path_a);
    remove(path_b);
    remove(path_c);
    remove(path_bad);
    remove(path_bad_size);
    nn_free(net);
    return ok;
}

static int test_trace_writer_emits_parseable_jsonl(void)
{
    const char *path = "/tmp/crlplrimes_test_trace.jsonl";
    const uint32_t action_ids[] = {10, 20};
    const nn_real scores[] = {(nn_real)0.25, (nn_real)-0.25};
    const nn_real probabilities[] = {(nn_real)0.62, (nn_real)0.38};
    uint32_t parsed_action_ids[2];
    nn_real parsed_scores[2];
    nn_real parsed_probabilities[2];
    NNParsedTraceRecord parsed;
    NNTraceRecord record;
    char line[1024];
    int ok = 1;

    memset(&record, 0, sizeof(record));
    record.task_name = "unit";
    record.model_hash_hex = "abc";
    record.state_hash = 0x1234u;
    record.feature_encoder_version = 3;
    record.candidate_count = 2;
    record.action_ids = action_ids;
    record.scores = scores;
    record.probabilities = probabilities;
    record.selected_index = 0;
    record.target_index = 1;
    record.loss = (nn_real)0.7;
    record.verifier_result = "corrected";
    record.verifier_cost_json = "{\"steps\":2}";
    record.seed = 99;
    record.extra_json = "\"p\":11,\"true_gap\":2";

    FILE *fp = fopen(path, "w");
    if (fp == NULL) {
        return 0;
    }
    ok = ok && nn_trace_write_jsonl(fp, &record);
    fclose(fp);

    fp = fopen(path, "r");
    if (fp == NULL) {
        return 0;
    }
    ok = ok && fgets(line, sizeof(line), fp) != NULL;
    fclose(fp);
    remove(path);

    ok = ok && line[0] == '{' && strstr(line, "\"schema\":\"crlp.trace.v1\"") != NULL &&
         strstr(line, "\"trace_version\":1") != NULL &&
         strstr(line, "\"action_id_contract\":\"stable_unique_action_ids\"") != NULL &&
         strstr(line, "\"candidate_count\":2") != NULL &&
         strstr(line, "\"action_ids\":[10,20]") != NULL &&
         strstr(line, "\"verifier_cost\":{\"steps\":2}") != NULL &&
         strstr(line, "\"p\":11") != NULL;
    ok = ok && nn_trace_parse_jsonl(line,
                                    &parsed,
                                    parsed_action_ids,
                                    parsed_scores,
                                    parsed_probabilities,
                                    2);
    ok = ok && strcmp(parsed.schema, NN_TRACE_SCHEMA) == 0 &&
         parsed.trace_version == NN_TRACE_VERSION &&
         strcmp(parsed.action_id_contract, NN_TRACE_ACTION_ID_CONTRACT) == 0 &&
         strcmp(parsed.task_name, "unit") == 0 &&
         parsed.state_hash == 0x1234u &&
         parsed.candidate_count == 2 &&
         parsed.selected_index == 0 &&
         parsed.target_index == 1 &&
         parsed_action_ids[0] == 10 &&
         parsed_action_ids[1] == 20 &&
         nearly_equal(parsed_probabilities[0], probabilities[0], (nn_real)1e-6, (nn_real)1e-6);
    return ok;
}

static int parse_action_fixture(const char *path,
                                nn_real weights[2],
                                nn_real *bias,
                                nn_real inputs[6],
                                nn_real expected_scores[3],
                                nn_real expected_probabilities[3],
                                size_t expected_ranking[3])
{
    FILE *fp = fopen(path, "r");
    char line[256];
    size_t input_row = 0;
    int reading_inputs = 0;
    int saw_weights = 0;
    int saw_bias = 0;
    int saw_scores = 0;
    int saw_probabilities = 0;
    int saw_ranking = 0;

    if (fp == NULL) {
        return 0;
    }

    while (fgets(line, sizeof(line), fp) != NULL) {
        if (line[0] == '#' || line[0] == '\n') {
            continue;
        }

        if (reading_inputs && input_row < 3) {
            double a = 0.0;
            double b = 0.0;
            if (sscanf(line, "%lf %lf", &a, &b) == 2) {
                inputs[input_row * 2] = (nn_real)a;
                inputs[input_row * 2 + 1] = (nn_real)b;
                ++input_row;
                if (input_row == 3) {
                    reading_inputs = 0;
                }
                continue;
            }
        }

        if (strncmp(line, "inputs", 6) == 0) {
            reading_inputs = 1;
            continue;
        }

        double a = 0.0;
        double b = 0.0;
        double c = 0.0;
        size_t r0 = 0;
        size_t r1 = 0;
        size_t r2 = 0;
        if (sscanf(line, "weights %lf %lf", &a, &b) == 2) {
            weights[0] = (nn_real)a;
            weights[1] = (nn_real)b;
            saw_weights = 1;
        } else if (sscanf(line, "bias %lf", &a) == 1) {
            *bias = (nn_real)a;
            saw_bias = 1;
        } else if (sscanf(line, "expected_scores %lf %lf %lf", &a, &b, &c) == 3) {
            expected_scores[0] = (nn_real)a;
            expected_scores[1] = (nn_real)b;
            expected_scores[2] = (nn_real)c;
            saw_scores = 1;
        } else if (sscanf(line, "expected_probabilities %lf %lf %lf", &a, &b, &c) == 3) {
            expected_probabilities[0] = (nn_real)a;
            expected_probabilities[1] = (nn_real)b;
            expected_probabilities[2] = (nn_real)c;
            saw_probabilities = 1;
        } else if (sscanf(line, "expected_ranking %zu %zu %zu", &r0, &r1, &r2) == 3) {
            expected_ranking[0] = r0;
            expected_ranking[1] = r1;
            expected_ranking[2] = r2;
            saw_ranking = 1;
        }
    }

    fclose(fp);
    return input_row == 3 && saw_weights && saw_bias && saw_scores && saw_probabilities &&
           saw_ranking;
}

static int test_fixture_checkpoint_load_and_scoring(void)
{
    const char *fixture_path = "tests/fixtures/action_scorer_fixture.txt";
    const char *checkpoint_path = "/tmp/crlplrimes_fixture_checkpoint.nnckpt";
    const size_t layers[] = {2, 1};
    nn_real weights[2];
    nn_real bias;
    nn_real inputs[6];
    nn_real expected_scores[3];
    nn_real expected_probabilities[3];
    size_t expected_ranking[3];
    nn_real scores[3];
    nn_real probabilities[3];
    size_t ranking[3];
    NNConfig config;
    NeuralNet *net;
    NeuralNet *loaded;
    int ok = 1;

    if (!parse_action_fixture(fixture_path,
                              weights,
                              &bias,
                              inputs,
                              expected_scores,
                              expected_probabilities,
                              expected_ranking)) {
        return 0;
    }

    config.layer_sizes = layers;
    config.layer_count = 2;
    config.seed = 1;
    config.hidden_activation = NN_ACTIVATION_LINEAR;
    config.output_activation = NN_ACTIVATION_LINEAR;
    config.loss = NN_LOSS_MSE;
    net = nn_create_config(&config);
    if (net == NULL) {
        return 0;
    }

    ok = ok && nn_test_set_parameter(net, 0, weights[0]) &&
         nn_test_set_parameter(net, 1, weights[1]) &&
         nn_test_set_parameter(net, 2, bias);
    ok = ok && nn_save_checkpoint(net, checkpoint_path, NULL);
    loaded = nn_load_checkpoint(checkpoint_path, NULL);
    ok = ok && loaded != NULL;
    if (loaded != NULL) {
        ok = ok && nn_score_action_rows(loaded, inputs, 3, scores, probabilities) &&
             nn_rank_action_scores(scores, 3, ranking);
        for (size_t i = 0; i < 3; ++i) {
            ok = ok && nearly_equal(scores[i],
                                    expected_scores[i],
                                    (nn_real)1e-6,
                                    (nn_real)1e-6);
            ok = ok && nearly_equal(probabilities[i],
                                    expected_probabilities[i],
                                    (nn_real)1e-6,
                                    (nn_real)1e-6);
            ok = ok && ranking[i] == expected_ranking[i];
        }
        nn_free(loaded);
    }

    remove(checkpoint_path);
    nn_free(net);
    return ok;
}

static int test_fixture_trace_parser(void)
{
    const char *fixture_path = "tests/fixtures/example_trace.jsonl";
    FILE *fp = fopen(fixture_path, "r");
    char line[1024];
    NNParsedTraceRecord parsed;
    uint32_t action_ids[3];
    nn_real scores[3];
    nn_real probabilities[3];
    uint64_t p = 0;
    int ok = 1;

    if (fp == NULL) {
        return 0;
    }
    ok = ok && fgets(line, sizeof(line), fp) != NULL;
    fclose(fp);

    ok = ok && nn_trace_parse_jsonl(line, &parsed, action_ids, scores, probabilities, 3);
    ok = ok && strcmp(parsed.schema, NN_TRACE_SCHEMA) == 0 &&
         parsed.trace_version == NN_TRACE_VERSION &&
         strcmp(parsed.action_id_contract, NN_TRACE_ACTION_ID_CONTRACT) == 0 &&
         strcmp(parsed.task_name, "fixture") == 0 &&
         strcmp(parsed.model_hash_hex, "fixture-model") == 0 &&
         parsed.state_hash == 0x1234u &&
         parsed.candidate_count == 3 &&
         parsed.selected_index == 0 &&
         parsed.target_index == 2 &&
         action_ids[0] == 10 &&
         action_ids[2] == 30 &&
         nn_trace_json_get_u64(line, "seed", &p) &&
         p == 42;

    const char *bad =
        "{\"schema\":\"crlp.trace.v1\",\"trace_version\":1,"
        "\"action_id_contract\":\"stable_unique_action_ids\",\"task\":\"fixture\","
        "\"state_hash\":\"0000000000001234\",\"model_hash\":\"fixture-model\","
        "\"feature_encoder_version\":1,\"candidate_count\":2,\"action_ids\":[1,2],"
        "\"scores\":[0,1],\"probabilities\":[0.7,0.7],"
        "\"selected_index\":0,\"target_index\":1,\"loss\":0.1,"
        "\"verifier_result\":\"corrected\",\"verifier_cost\":{},\"seed\":42}";
    ok = ok && !nn_trace_parse_jsonl(bad, &parsed, action_ids, scores, probabilities, 3);

    const char *duplicate_ids =
        "{\"schema\":\"crlp.trace.v1\",\"trace_version\":1,"
        "\"action_id_contract\":\"stable_unique_action_ids\",\"task\":\"fixture\","
        "\"state_hash\":\"0000000000001234\",\"model_hash\":\"fixture-model\","
        "\"feature_encoder_version\":1,\"candidate_count\":2,\"action_ids\":[1,1],"
        "\"scores\":[0,1],\"probabilities\":[0.5,0.5],"
        "\"selected_index\":0,\"target_index\":1,\"loss\":0.1,"
        "\"verifier_result\":\"corrected\",\"verifier_cost\":{},\"seed\":42}";
    ok = ok && !nn_trace_parse_jsonl(
                   duplicate_ids, &parsed, action_ids, scores, probabilities, 3);
    return ok;
}

static int test_prime_gap_rejects_negative_size_args(void)
{
    char *argv[] = {"prime_gap", "-1"};
    size_t value = 123;

    return !parse_size_arg(2, argv, 1, DEFAULT_TRAIN_LIMIT, &value) && value == 123;
}

static int test_prime_gap_rejects_invalid_regularization_args(void)
{
    char *negative_weight[] = {
        "prime_gap", "1000", "2000", "1", "1", "100", "100", "-0.1", "0.1",
    };
    char *bad_weight[] = {
        "prime_gap", "1000", "2000", "1", "1", "100", "100", "abc", "0.1",
    };
    PrimeGapRunConfig config;

    return parse_run_config(9, negative_weight, &config) && !run_config_is_valid(&config) &&
           !parse_run_config(9, bad_weight, &config);
}

static int test_prime_gap_parses_rng_seed(void)
{
    char *argv[] = {
        "prime_gap", "1000", "2000", "1", "1", "100", "100", "0.4", "0.04", "12345",
    };
    PrimeGapRunConfig config;

    return parse_run_config(10, argv, &config) && run_config_is_valid(&config) &&
           config.rng_seed_arg == 12345;
}

static int test_prime_gap_feature_set_defaults_to_minimal(void)
{
    char *default_argv[] = {"prime_gap", "1000", "2000"};
    char *full_argv[] = {"prime_gap", "1000", "2000", "--feature-set", "full"};
    char *minimal_argv[] = {"prime_gap", "1000", "2000", "--no-residue-features"};
    PrimeGapRunConfig config;

    return parse_run_config(3, default_argv, &config) && run_config_is_valid(&config) &&
           config.feature_mask == &FEATURE_MASK_MINIMAL &&
           parse_run_config(5, full_argv, &config) && run_config_is_valid(&config) &&
           config.feature_mask == &FEATURE_MASK_FULL &&
           parse_run_config(4, minimal_argv, &config) && run_config_is_valid(&config) &&
           config.feature_mask == &FEATURE_MASK_MINIMAL;
}

static int test_prime_gap_features_are_divisibility_flags(void)
{
    nn_real inputs[ACTION_COUNT * FEATURE_COUNT];
    nn_real full[ACTION_COUNT * FEATURE_COUNT];
    nn_real masked[ACTION_COUNT * FEATURE_COUNT];
    nn_real minimal[ACTION_COUNT * FEATURE_COUNT];
    fill_candidate_features(inputs, 11, log(1000.0));
    fill_candidate_features_masked(full, 11, log(1000.0), &FEATURE_MASK_FULL);
    fill_candidate_features_masked(masked, 11, log(1000.0), &FEATURE_MASK_GAP_ONLY);
    fill_candidate_features_masked(minimal, 11, log(1000.0), &FEATURE_MASK_MINIMAL);

    const nn_real *default_gap4 = &inputs[1 * FEATURE_COUNT];
    const nn_real *gap2 = &full[0 * FEATURE_COUNT];
    const nn_real *gap4 = &full[1 * FEATURE_COUNT];
    const nn_real *overflow = &full[OVERFLOW_ACTION * FEATURE_COUNT];
    const nn_real *masked_gap4 = &masked[1 * FEATURE_COUNT];
    const nn_real *masked_overflow = &masked[OVERFLOW_ACTION * FEATURE_COUNT];
    const nn_real *minimal_gap4 = &minimal[1 * FEATURE_COUNT];

    return default_gap4[0] > (nn_real)0 && default_gap4[1] > (nn_real)0 &&
           default_gap4[3] == (nn_real)0 &&
           gap4[2] == (nn_real)0 && gap4[3] == (nn_real)1 &&
           gap4[4] == (nn_real)1 && gap4[5] == (nn_real)0 &&
           gap2[7] == (nn_real)0 && overflow[2] == (nn_real)1 &&
           masked_gap4[0] == (nn_real)0 && masked_gap4[1] > (nn_real)0 &&
           masked_gap4[3] == (nn_real)0 && masked_overflow[2] == (nn_real)1 &&
           minimal_gap4[0] > (nn_real)0 && minimal_gap4[1] > (nn_real)0 &&
           minimal_gap4[2] == (nn_real)0 && minimal_gap4[3] == (nn_real)0 &&
           minimal_gap4[4] == (nn_real)0;
}

static int test_rule_proposer_features_are_structural(void)
{
    SymbolicRejectRule symbolic_a;
    SymbolicRejectRule symbolic_b;
    memset(&symbolic_a, 0, sizeof(symbolic_a));
    memset(&symbolic_b, 0, sizeof(symbolic_b));
    symbolic_a.modulus = 15;
    symbolic_a.residue = 5;
    symbolic_a.train_rejects = 10;
    symbolic_a.validation_rejects = 20;
    symbolic_b = symbolic_a;
    symbolic_b.train_rejects = 200;
    symbolic_b.validation_rejects = 400;

    nn_real symbolic_features_a[SYMBOLIC_RULE_FEATURE_COUNT];
    nn_real symbolic_features_b[SYMBOLIC_RULE_FEATURE_COUNT];
    fill_symbolic_rule_features(&symbolic_a, symbolic_features_a);
    fill_symbolic_rule_features(&symbolic_b, symbolic_features_b);

    int ok = symbolic_features_a[0] > (nn_real)0 &&
             symbolic_features_a[1] > (nn_real)0 &&
             symbolic_features_a[2] == (nn_real)0 &&
             symbolic_features_a[3] > (nn_real)0;
    for (size_t i = 0; i < SYMBOLIC_RULE_FEATURE_COUNT; ++i) {
        ok = ok && symbolic_features_a[i] == symbolic_features_b[i];
    }

    ExpandedRule expanded_a;
    ExpandedRule expanded_b;
    memset(&expanded_a, 0, sizeof(expanded_a));
    memset(&expanded_b, 0, sizeof(expanded_b));
    expanded_a.kind = RULE_FILTER_GAP_MOD;
    expanded_a.state_modulus = 15;
    expanded_a.state_residue = 4;
    expanded_a.filter_modulus = 15;
    expanded_a.filter_residue = 11;
    expanded_a.train_rejects = 10;
    expanded_a.train_false_rejections = 1;
    expanded_a.validation_rejects = 20;
    expanded_a.validation_false_rejections = 2;
    expanded_b = expanded_a;
    expanded_b.train_rejects = 500;
    expanded_b.train_false_rejections = 0;
    expanded_b.validation_rejects = 700;
    expanded_b.validation_false_rejections = 0;

    nn_real expanded_features_a[RULE_SEARCH_FEATURE_COUNT];
    nn_real expanded_features_b[RULE_SEARCH_FEATURE_COUNT];
    fill_expanded_rule_features(&expanded_a, 23, expanded_features_a);
    fill_expanded_rule_features(&expanded_b, 23, expanded_features_b);

    ok = ok && expanded_features_a[0] == (nn_real)1 &&
         expanded_features_a[4] > (nn_real)0 &&
         expanded_features_a[5] == (nn_real)1 &&
         expanded_features_a[10] == (nn_real)1 &&
         expanded_features_a[11] == (nn_real)0 &&
         expanded_features_a[12] == (nn_real)1;
    for (size_t i = 0; i < RULE_SEARCH_FEATURE_COUNT; ++i) {
        ok = ok && expanded_features_a[i] == expanded_features_b[i];
    }

    return ok;
}

static int test_rule_search_unconditional_range_builds_structural_program(void)
{
    char *argv[] = {
        "prime_rule_search",
        "--candidate-only",
        "--unconditional-only",
        "--moduli-range",
        "3:5",
    };
    RuleSearchConfig config;
    RuleSearchGrammar grammar;
    ExpandedRuleSet rules;
    memset(&grammar, 0, sizeof(grammar));
    memset(&rules, 0, sizeof(rules));

    int ok = parse_rule_search_config(5, argv, &config);
    ok = ok && config.include_state_conditions == 0 &&
         config.include_candidate_filters == 1 &&
         config.include_gap_filters == 0 && config.mod_count == 3 &&
         config.moduli[0] == 3 && config.moduli[1] == 4 &&
         config.moduli[2] == 5;

    ok = ok && build_rule_search_grammar(&config, &grammar);
    ok = ok && grammar.state_count == 1 && grammar.filter_count == 12 &&
         grammar.state_atoms[0].modulus == 1 && grammar.state_atoms[0].residue == 0;

    ok = ok && build_expanded_rules(&grammar, &rules);
    ok = ok && rules.count == 12 && rules.items[0].state_modulus == 1 &&
         rules.items[0].state_residue == 0 &&
         rules.items[0].kind == RULE_FILTER_CANDIDATE_MOD &&
         rules.items[0].filter_modulus == 3 &&
         rules.items[11].filter_modulus == 5 &&
         rules.items[11].filter_residue == 4;

    free_expanded_rule_set(&rules);
    free_rule_search_grammar(&grammar);
    return ok;
}

static int test_expanded_rule_survivor_mask_rejects_candidate_filters(void)
{
    ExpandedRule rule;
    memset(&rule, 0, sizeof(rule));
    rule.kind = RULE_FILTER_CANDIDATE_MOD;
    rule.state_modulus = 1;
    rule.state_residue = 0;
    rule.filter_modulus = 3;
    rule.filter_residue = 0;
    rule.accepted = 1;

    ExpandedRuleSet rules;
    rules.items = &rule;
    rules.count = 1;

    nn_real inputs[ACTION_COUNT * FEATURE_COUNT];
    size_t survivor_actions[ACTION_COUNT];
    size_t survivor_count = fill_expanded_survivor_candidates(&rules,
                                                              inputs,
                                                              11,
                                                              log(1000.0),
                                                              &FEATURE_MASK_MINIMAL,
                                                              survivor_actions);

    size_t gap2 = gap_to_action(2);
    size_t gap4 = gap_to_action(4);
    return survivor_count > 0 &&
           survivor_index_for_action(survivor_actions, survivor_count, gap2) !=
               survivor_count &&
           survivor_index_for_action(survivor_actions, survivor_count, gap4) ==
               survivor_count;
}

static int test_expanded_rule_checkpoint_roundtrip(void)
{
#if NN_REAL_IS_FLOAT
    const char *rules_path = "build/test_expanded_rules_float.crlprule";
#else
    const char *rules_path = "build/test_expanded_rules_double.crlprule";
#endif
    ExpandedRule rule;
    memset(&rule, 0, sizeof(rule));
    rule.kind = RULE_FILTER_CANDIDATE_MOD;
    rule.state_modulus = 1;
    rule.state_residue = 0;
    rule.filter_modulus = 3;
    rule.filter_residue = 0;
    rule.train_rejects = 12;
    rule.validation_rejects = 7;
    rule.model_score = (nn_real)0.25;
    rule.accepted = 1;

    ExpandedRuleSet rules;
    rules.items = &rule;
    rules.count = 1;

    RuleSearchConfig config = default_rule_search_config();
    config.include_state_conditions = 0;
    config.include_candidate_filters = 1;
    config.include_gap_filters = 0;
    config.mod_count = 1;
    config.moduli[0] = 3;

    RuleSearchMetrics metrics;
    memset(&metrics, 0, sizeof(metrics));
    metrics.proposed_rules = 3;
    metrics.verified_rules = 1;
    metrics.accepted_rules = 1;
    metrics.accepted_candidate_rules = 1;
    metrics.direct_exact_tests = 2.0;
    metrics.hand_wheel_exact_tests = 1.0;
    metrics.expanded_exact_tests = 1.0;

    ExpandedRuleSet loaded_rules;
    RuleSearchConfig loaded_config;
    RuleSearchMetrics loaded_metrics;
    char saved_hash[65];
    char loaded_hash[65];
    memset(&loaded_rules, 0, sizeof(loaded_rules));
    memset(&loaded_config, 0, sizeof(loaded_config));
    memset(&loaded_metrics, 0, sizeof(loaded_metrics));

    int ok = save_expanded_rule_checkpoint(&rules,
                                           &config,
                                           &metrics,
                                           rules_path,
                                           saved_hash);
    ok = ok && load_expanded_rules_with_hash(rules_path,
                                             &loaded_rules,
                                             &loaded_config,
                                             &loaded_metrics,
                                             loaded_hash);
    if (ok) {
        ok = strcmp(saved_hash, loaded_hash) == 0 && strlen(saved_hash) == 64 &&
             loaded_rules.count == 1 &&
             loaded_rules.items[0].kind == RULE_FILTER_CANDIDATE_MOD &&
             loaded_rules.items[0].state_modulus == 1 &&
             loaded_rules.items[0].filter_modulus == 3 &&
             loaded_rules.items[0].filter_residue == 0 &&
             loaded_rules.items[0].train_rejects == 12 &&
             loaded_config.include_state_conditions == 0 &&
             loaded_config.include_candidate_filters == 1 &&
             loaded_config.include_gap_filters == 0 &&
             loaded_config.mod_count == 1 && loaded_config.moduli[0] == 3 &&
             loaded_metrics.accepted_rules == 1 &&
             loaded_metrics.accepted_candidate_rules == 1 &&
             loaded_metrics.accepted_gap_rules == 0 &&
             loaded_metrics.accepted_conditional_rules == 0;
    }

    free_expanded_rule_set(&loaded_rules);
    return ok;
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

static int test_prime_gap_conditioned_frequency_uses_residue_flags(void)
{
    GapDataset dataset;
    ConditionedFrequencyPolicy policy;
    nn_real scores[ACTION_COUNT];
    nn_real probabilities[ACTION_COUNT];
    double sum = 0.0;
    int ok;

    memset(&dataset, 0, sizeof(dataset));
    memset(&policy, 0, sizeof(policy));
    ok = append_gap(&dataset, 11, 2) && append_gap(&dataset, 17, 4) &&
         append_gap(&dataset, 59, 4);
    if (!ok || !build_conditioned_frequency_policy(&policy, &dataset)) {
        free_conditioned_frequency_policy(&policy);
        free_gap_dataset(&dataset);
        return 0;
    }

    conditioned_frequency_scores(&policy, 17, scores, probabilities);
    for (size_t i = 0; i < ACTION_COUNT; ++i) {
        sum += (double)probabilities[i];
    }

    ok = scores[gap_to_action(4)] > scores[gap_to_action(2)] &&
         probabilities[OVERFLOW_ACTION] > (nn_real)0 &&
         fabs(sum - 1.0) < 1e-5;

    free_conditioned_frequency_policy(&policy);
    free_gap_dataset(&dataset);
    return ok;
}

static int test_prime_gap_temperature_softmax_flattens_confidence(void)
{
    const nn_real scores[] = {(nn_real)0, (nn_real)1, (nn_real)3};
    nn_real probabilities_cold[3];
    nn_real probabilities_hot[3];

    softmax_temperature(scores, probabilities_cold, 3, 1.0);
    softmax_temperature(scores, probabilities_hot, 3, 4.0);

    double cold_sum = (double)probabilities_cold[0] + (double)probabilities_cold[1] +
                      (double)probabilities_cold[2];
    double hot_sum = (double)probabilities_hot[0] + (double)probabilities_hot[1] +
                     (double)probabilities_hot[2];

    return probabilities_cold[2] > probabilities_cold[1] &&
           probabilities_hot[2] > probabilities_hot[1] &&
           probabilities_hot[2] < probabilities_cold[2] &&
           fabs(cold_sum - 1.0) < 1e-5 && fabs(hot_sum - 1.0) < 1e-5;
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

static int test_prime_gap_symbolic_rule_discovery_is_verified(void)
{
    Sieve sieve;
    GapDataset train;
    GapDataset validation;
    GapDataset test;
    SymbolicRuleSet rules;
    SymbolicWheelMetrics metrics;
    int ok = 1;

    if (!build_sieve(1000 + G_MAX, &sieve)) {
        return 0;
    }
    if (!build_gap_datasets(&sieve, 200, 1000, &train, &validation, &test)) {
        free_sieve(&sieve);
        return 0;
    }

    ok = ok && discover_symbolic_wheel(&train, &validation, &rules, &metrics);
    if (ok) {
        metrics = evaluate_symbolic_wheel(&sieve, &rules, &test, 0, metrics);
        ok = ok && rules.count > 0;
        ok = ok && metrics.test_false_rejections == 0;
        ok = ok && metrics.learned_exact_tests <= metrics.hand_wheel_exact_tests;
        ok = ok && rules.items[0].modulus == 3 && rules.items[0].residue == 0;

#if NN_REAL_IS_FLOAT
        const char *rules_path = "build/test_symbolic_rules_float.crlprule";
#else
        const char *rules_path = "build/test_symbolic_rules_double.crlprule";
#endif
        SymbolicRuleSet loaded_rules;
        SymbolicWheelMetrics loaded_metrics;
        char saved_hash[65];
        char loaded_hash[65];
        memset(&loaded_rules, 0, sizeof(loaded_rules));
        memset(&loaded_metrics, 0, sizeof(loaded_metrics));
        ok = ok && save_symbolic_rule_checkpoint(&rules, rules_path, &metrics, saved_hash);
        ok = ok && load_symbolic_rules_with_hash(rules_path, &loaded_rules, loaded_hash);
        if (ok) {
            loaded_metrics.accepted_rules = loaded_rules.count;
            loaded_metrics = evaluate_symbolic_wheel(&sieve,
                                                     &loaded_rules,
                                                     &test,
                                                     0,
                                                     loaded_metrics);
            ok = ok && strcmp(saved_hash, loaded_hash) == 0 &&
                 loaded_rules.count == rules.count &&
                 loaded_rules.items[0].modulus == rules.items[0].modulus &&
                 loaded_rules.items[0].residue == rules.items[0].residue &&
                 loaded_metrics.test_false_rejections == 0;
        }
        free_symbolic_rule_set(&loaded_rules);
    }

    free_symbolic_rule_set(&rules);
    free_gap_dataset(&train);
    free_gap_dataset(&validation);
    free_gap_dataset(&test);
    free_sieve(&sieve);
    return ok;
}

static int test_prime_gap_replay_uses_trace_action_ids(void)
{
#if NN_REAL_IS_FLOAT
    const char *trace_path = "build/test_prime_replay_action_ids_float.jsonl";
#else
    const char *trace_path = "build/test_prime_replay_action_ids_double.jsonl";
#endif
    FILE *fp = fopen(trace_path, "w");
    if (fp == NULL) {
        return 0;
    }

    fprintf(fp,
            "{\"schema\":\"crlp.trace.v1\",\"trace_version\":1,"
            "\"action_id_contract\":\"stable_unique_action_ids\",\"task\":\"prime_gap\","
            "\"state_hash\":\"0000000000000001\",\"model_hash\":\"fixture\","
            "\"feature_encoder_version\":1,\"candidate_count\":%d,\"action_ids\":[",
            ACTION_COUNT);
    for (size_t i = 0; i < ACTION_COUNT; ++i) {
        uint32_t action_id = (uint32_t)i;
        if (i == 0) {
            action_id = 7;
        } else if (i == 7) {
            action_id = 0;
        }
        fprintf(fp, "%s%u", i == 0 ? "" : ",", action_id);
    }
    fputs("],\"scores\":[", fp);
    for (size_t i = 0; i < ACTION_COUNT; ++i) {
        fputs(i == 0 ? "0" : ",0", fp);
    }
    fputs("],\"probabilities\":[", fp);
    for (size_t i = 0; i < ACTION_COUNT; ++i) {
        fputs(i == 0 ? "1" : ",0", fp);
    }
    fprintf(fp,
            "],\"selected_index\":0,\"target_index\":0,\"loss\":0,"
            "\"verifier_result\":\"correct\",\"verifier_cost\":{},\"seed\":42,"
            "\"feature_set\":\"minimal\","
            "\"p\":101,\"log_scale\":%.17g}\n",
            log(1000.0));
    if (fclose(fp) != 0) {
        return 0;
    }

    ReplayTrace trace;
    int ok = load_replay_trace(trace_path, &trace);
    uint32_t current_action_ids[ACTION_COUNT];
    fill_action_ids(current_action_ids);
    ok = ok && trace.count == 1 && trace.items[0].p == 101 &&
         strcmp(trace.feature_set, "minimal") == 0 &&
         trace.items[0].target_action_id == 7 &&
         action_row_for_action_id(current_action_ids,
                                  ACTION_COUNT,
                                  trace.items[0].target_action_id) == 7;
    free_replay_trace(&trace);
    return ok;
}

static int test_prime_gap_replay_requires_feature_set(void)
{
#if NN_REAL_IS_FLOAT
    const char *trace_path = "build/test_prime_replay_missing_feature_set_float.jsonl";
#else
    const char *trace_path = "build/test_prime_replay_missing_feature_set_double.jsonl";
#endif
    FILE *fp = fopen(trace_path, "w");
    if (fp == NULL) {
        return 0;
    }

    fprintf(fp,
            "{\"schema\":\"crlp.trace.v1\",\"trace_version\":1,"
            "\"action_id_contract\":\"stable_unique_action_ids\",\"task\":\"prime_gap\","
            "\"state_hash\":\"0000000000000001\",\"model_hash\":\"fixture\","
            "\"feature_encoder_version\":1,\"candidate_count\":%d,\"action_ids\":[",
            ACTION_COUNT);
    for (size_t i = 0; i < ACTION_COUNT; ++i) {
        fprintf(fp, "%s%zu", i == 0 ? "" : ",", i);
    }
    fputs("],\"scores\":[", fp);
    for (size_t i = 0; i < ACTION_COUNT; ++i) {
        fputs(i == 0 ? "0" : ",0", fp);
    }
    fputs("],\"probabilities\":[", fp);
    for (size_t i = 0; i < ACTION_COUNT; ++i) {
        fputs(i == 0 ? "1" : ",0", fp);
    }
    fprintf(fp,
            "],\"selected_index\":0,\"target_index\":0,\"loss\":0,"
            "\"verifier_result\":\"correct\",\"verifier_cost\":{},\"seed\":42,"
            "\"p\":101,\"log_scale\":%.17g}\n",
            log(1000.0));
    if (fclose(fp) != 0) {
        return 0;
    }

    ReplayTrace trace;
    int loaded = load_replay_trace(trace_path, &trace);
    if (loaded) {
        free_replay_trace(&trace);
    }
    return !loaded;
}

static int test_prime_gap_dataset_tracks_overflow_actions(void)
{
    Sieve sieve;
    GapDataset train;
    GapDataset validation;
    GapDataset test;
    int ok = 1;

    if (!build_sieve(200 + G_MAX, &sieve)) {
        return 0;
    }

    if (!build_gap_datasets(&sieve, 50, 200, &train, &validation, &test)) {
        free_sieve(&sieve);
        return 0;
    }

    ok = ok && train.total_seen == train.count;
    ok = ok && validation.total_seen == validation.count;
    ok = ok && test.total_seen == test.count;
    ok = ok && train.overflow_count <= train.count;
    ok = ok && validation.overflow_count <= validation.count;
    ok = ok && test.overflow_count <= test.count;
    ok = ok && gap_to_action(G_MAX + 2) == OVERFLOW_ACTION;
    ok = ok && train.count > 0 && validation.count > 0 && test.count > 0;

    free_gap_dataset(&train);
    free_gap_dataset(&validation);
    free_gap_dataset(&test);
    free_sieve(&sieve);
    return ok;
}

int main(void)
{
    check(test_forward_batch_matches_scalar(), "batched forward matches scalar");
    check(test_clone_matches_and_detaches(), "clone matches and detaches");
    check(test_gradient_buffers_are_lazy(), "gradient buffers are lazy");
    check(test_batch_size_one_matches_sample_training(), "batch size 1 matches sample training");
    check(test_batched_gradient_matches_finite_difference(),
          "batched gradient matches finite difference");
    check(test_softmax_cross_entropy_config_learns(),
          "softmax cross-entropy config learns logits");
    check(test_softmax_scorer_gradient_matches_finite_difference(),
          "softmax scorer gradient matches finite difference");
    check(test_softmax_cross_entropy_requires_linear_output(),
          "softmax cross-entropy requires linear output");
    check(test_softmax_scorer_cached_matches_regular(),
          "softmax scorer cached update matches regular update");
    check(test_policy_regularized_cached_matches_policy_update(),
          "policy-regularized cached update matches policy update");
    check(test_action_scorer_api_scores_probabilities_and_ranking(),
          "action scorer scores probabilities and ranks deterministically");
    check(test_pairwise_action_ranker_orders_target(),
          "pairwise action ranker orders target");
    check(test_checkpoint_roundtrip_hash_and_invalid_files(),
          "checkpoint roundtrip hash and invalid files");
    check(test_trace_writer_emits_parseable_jsonl(),
          "trace writer emits parseable jsonl");
    check(test_fixture_checkpoint_load_and_scoring(),
          "fixture checkpoint load and scoring");
    check(test_fixture_trace_parser(),
          "fixture trace parser validates schema");
    check(test_prime_gap_rejects_negative_size_args(),
          "prime-gap parser rejects negative size args");
    check(test_prime_gap_rejects_invalid_regularization_args(),
          "prime-gap parser rejects invalid regularization args");
    check(test_prime_gap_parses_rng_seed(), "prime-gap parser accepts rng seed");
    check(test_prime_gap_feature_set_defaults_to_minimal(),
          "prime-gap feature set defaults to minimal");
    check(test_prime_gap_features_are_divisibility_flags(),
          "prime-gap features are divisibility flags");
    check(test_rule_proposer_features_are_structural(),
          "rule proposer features are structural");
    check(test_rule_search_unconditional_range_builds_structural_program(),
          "rule search unconditional range builds structural program");
    check(test_expanded_rule_survivor_mask_rejects_candidate_filters(),
          "expanded rule survivor mask rejects candidate filters");
    check(test_expanded_rule_checkpoint_roundtrip(),
          "expanded rule checkpoint roundtrip");
    check(test_prime_gap_frequency_policy_is_smoothed(),
          "prime-gap frequency policy is smoothed");
    check(test_prime_gap_conditioned_frequency_uses_residue_flags(),
          "prime-gap conditioned frequency uses residue flags");
    check(test_prime_gap_temperature_softmax_flattens_confidence(),
          "prime-gap temperature softmax flattens confidence");
    check(test_prime_gap_wheel_verifier_accounting(),
          "prime-gap wheel verifier accounting");
    check(test_prime_gap_symbolic_rule_discovery_is_verified(),
          "prime-gap symbolic rule discovery is verified");
    check(test_prime_gap_replay_uses_trace_action_ids(),
          "prime-gap replay uses trace action ids");
    check(test_prime_gap_replay_requires_feature_set(),
          "prime-gap replay requires feature set");
    check(test_prime_gap_dataset_tracks_overflow_actions(),
          "prime-gap dataset tracks overflow actions");

    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
