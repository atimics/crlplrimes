#include "nn.h"

#include <stdio.h>
#include <stdlib.h>

enum {
    LAYER_DISCOVERY_SCENES = 64,
    LAYER_DISCOVERY_WIDTH = 24,
    LAYER_DISCOVERY_HEIGHT = 24,
    LAYER_DISCOVERY_OBJECTS = 5,
    LAYER_DISCOVERY_MAX_TRANSLATION = 3,
};

static int layer_result_is_exact(const NNLayerDiscoveryResult *result)
{
    return result != NULL &&
           result->certificate.status == NN_CERTIFICATE_CERTIFIED &&
           result->reconstruction_errors == 0u &&
           result->layer_count == result->certified_layers;
}

int main(void)
{
    NNLayerDiscoveryConfig config;
    nn_layer_discovery_config_init(&config);
    config.max_translation = LAYER_DISCOVERY_MAX_TRANSLATION;
    config.max_layers = NN_LAYER_MAX_PROPOSALS;
    config.include_static_layers = 1;
    config.allow_residual_layer = 0;

    size_t exact = 0u;
    size_t total_layers = 0u;
    size_t total_certified_layers = 0u;
    size_t total_errors = 0u;
    size_t max_errors = 0u;
    double total_description_length = 0.0;

    printf("layer_discovery schema=%s scenes=%d grid=%dx%d objects=%d max_translation=%d\n",
           NN_LAYER_DISCOVERY_SCHEMA,
           LAYER_DISCOVERY_SCENES,
           LAYER_DISCOVERY_WIDTH,
           LAYER_DISCOVERY_HEIGHT,
           LAYER_DISCOVERY_OBJECTS,
           LAYER_DISCOVERY_MAX_TRANSLATION);

    for (size_t scene = 0u; scene < LAYER_DISCOVERY_SCENES; ++scene) {
        NNBitGrid source = {0};
        NNBitGrid target = {0};
        NNLayerDiscoveryResult result;
        nn_layer_discovery_result_init(&result);
        unsigned int seed = 20240522u + (unsigned int)(scene * 17u);

        int ok = nn_layer_generate_motion_scene(&source,
                                                &target,
                                                LAYER_DISCOVERY_WIDTH,
                                                LAYER_DISCOVERY_HEIGHT,
                                                LAYER_DISCOVERY_OBJECTS,
                                                LAYER_DISCOVERY_MAX_TRANSLATION,
                                                seed) &&
                 nn_layer_discover_motion(&source, &target, &config, &result);
        if (!ok) {
            fprintf(stderr, "layer_discovery failed scene=%zu seed=%u\n", scene, seed);
            nn_layer_discovery_result_free(&result);
            nn_bitgrid_free(&source);
            nn_bitgrid_free(&target);
            return EXIT_FAILURE;
        }

        int scene_exact = layer_result_is_exact(&result);
        exact += scene_exact ? 1u : 0u;
        total_layers += result.layer_count;
        total_certified_layers += result.certified_layers;
        total_errors += result.reconstruction_errors;
        if (result.reconstruction_errors > max_errors) {
            max_errors = result.reconstruction_errors;
        }
        total_description_length += result.description_length;

        if (scene < 8u || !scene_exact) {
            printf("scene=%zu seed=%u exact=%d layers=%zu certified=%zu errors=%zu dl=%.3f\n",
                   scene,
                   seed,
                   scene_exact,
                   result.layer_count,
                   result.certified_layers,
                   result.reconstruction_errors,
                   result.description_length);
        }
        if (scene == 0u) {
            (void)nn_layer_discovery_write_json(stdout, &result);
        }

        nn_layer_discovery_result_free(&result);
        nn_bitgrid_free(&source);
        nn_bitgrid_free(&target);
    }

    printf("summary exact=%zu/%d avg_layers=%.3f avg_certified_layers=%.3f "
           "avg_errors=%.3f max_errors=%zu avg_description_length=%.3f\n",
           exact,
           LAYER_DISCOVERY_SCENES,
           (double)total_layers / (double)LAYER_DISCOVERY_SCENES,
           (double)total_certified_layers / (double)LAYER_DISCOVERY_SCENES,
           (double)total_errors / (double)LAYER_DISCOVERY_SCENES,
           max_errors,
           total_description_length / (double)LAYER_DISCOVERY_SCENES);

    NNLayerRankerConfig ranker_config;
    NNLayerRanker ranker;
    nn_layer_ranker_config_init(&ranker_config);
    nn_layer_ranker_init(&ranker);
    ranker_config.width = LAYER_DISCOVERY_WIDTH;
    ranker_config.height = LAYER_DISCOVERY_HEIGHT;
    ranker_config.object_count = LAYER_DISCOVERY_OBJECTS;
    ranker_config.max_translation = LAYER_DISCOVERY_MAX_TRANSLATION;
    ranker_config.seed = 20240522u;

    int ranker_ok = nn_layer_ranker_train(&ranker, &ranker_config);
    (void)nn_layer_ranker_write_json(stdout, &ranker);
    printf("ranker_eval exact=%zu/%zu top1=%.2f%% avg_verifier_checks=%.3f "
           "parameters=%zu flops=%zu\n",
           ranker.report.certified_scenes,
           ranker.report.eval_scenes,
           ranker.report.top1_accuracy * 100.0,
           ranker.report.average_verifier_checks,
           ranker.report.parameters,
           ranker.report.forward_flops);
    nn_layer_ranker_free(&ranker);

    NNLayerTaskPolicyConfig task_config;
    NNLayerTaskPolicy task_policy;
    nn_layer_task_policy_config_init(&task_config);
    nn_layer_task_policy_init(&task_policy);
    task_config.width = LAYER_DISCOVERY_WIDTH;
    task_config.height = LAYER_DISCOVERY_HEIGHT;
    task_config.object_count = LAYER_DISCOVERY_OBJECTS;
    task_config.max_translation = LAYER_DISCOVERY_MAX_TRANSLATION;
    task_config.seed = 20240522u;

    int task_ok = nn_layer_task_policy_train(&task_policy, &task_config);
    (void)nn_layer_task_policy_write_json(stdout, &task_policy);
    printf("task_policy_eval exact=%zu/%zu top1=%.2f%% parameters=%zu flops=%zu\n",
           task_policy.report.solved_tasks,
           task_policy.report.eval_tasks,
           task_policy.report.top1_accuracy * 100.0,
           task_policy.report.parameters,
           task_policy.report.forward_flops);
    nn_layer_task_policy_free(&task_policy);

    return exact == LAYER_DISCOVERY_SCENES && ranker_ok && task_ok ? EXIT_SUCCESS
                                                                   : EXIT_FAILURE;
}
