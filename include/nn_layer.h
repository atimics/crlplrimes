#ifndef NN_LAYER_H
#define NN_LAYER_H

#include "nn_bitgrid.h"
#include "nn_symbolic.h"

#include <stddef.h>
#include <stdio.h>

#define NN_LAYER_DISCOVERY_SCHEMA "crlp.layer_discovery.v1"
#define NN_LAYER_RANKER_SCHEMA "crlp.layer_ranker.v1"
#define NN_LAYER_TASK_POLICY_SCHEMA "crlp.layer_task_policy.v1"
#define NN_LAYER_MAX_PROPOSALS 32u
#define NN_LAYER_RANKER_FEATURE_COUNT 16u
#define NN_LAYER_TASK_FEATURE_COUNT 16u
#define NN_LAYER_RANKER_MAX_CANDIDATES 81u

typedef enum {
    NN_LAYER_KIND_UNKNOWN = 0,
    NN_LAYER_KIND_STATIC = 1,
    NN_LAYER_KIND_TRANSLATION = 2,
    NN_LAYER_KIND_RESIDUAL = 3,
} NNLayerKind;

typedef struct {
    size_t max_layers;
    int max_translation;
    int include_static_layers;
    int allow_residual_layer;
} NNLayerDiscoveryConfig;

typedef struct {
    char schema[32];
    NNLayerKind kind;
    int dx;
    int dy;
    size_t min_x;
    size_t min_y;
    size_t width;
    size_t height;
    size_t source_cells;
    size_t target_cells;
    size_t reconstructed_cells;
    size_t reconstruction_errors;
    size_t candidate_count;
    size_t selected_rank;
    size_t verifier_checks;
    double description_length;
    NNBitGrid mask;
    NNBitGrid values;
    NNCertificate certificate;
} NNLayerProposal;

typedef struct {
    char schema[32];
    size_t layer_count;
    size_t certified_layers;
    size_t source_cells;
    size_t target_cells;
    size_t reconstructed_cells;
    size_t reconstruction_errors;
    size_t top1_layers;
    size_t verifier_checks;
    double description_length;
    NNBitGrid reconstruction;
    NNCertificate certificate;
    NNLayerProposal layers[NN_LAYER_MAX_PROPOSALS];
} NNLayerDiscoveryResult;

typedef struct {
    size_t hidden_width;
    size_t epochs;
    size_t train_scenes;
    size_t eval_scenes;
    size_t width;
    size_t height;
    size_t object_count;
    int max_translation;
    nn_real learning_rate;
    unsigned int seed;
} NNLayerRankerConfig;

typedef struct {
    char schema[32];
    size_t train_scenes;
    size_t eval_scenes;
    size_t training_groups;
    size_t training_updates;
    size_t eval_layers;
    size_t top1_layers;
    size_t certified_scenes;
    size_t verifier_checks;
    nn_real final_loss;
    double top1_accuracy;
    double certified_rate;
    double average_verifier_checks;
    size_t parameters;
    size_t forward_flops;
} NNLayerRankerReport;

typedef struct {
    NeuralNet *net;
    NNLayerRankerConfig config;
    NNLayerRankerReport report;
} NNLayerRanker;

typedef struct {
    size_t hidden_width;
    size_t epochs;
    size_t train_tasks;
    size_t eval_tasks;
    size_t width;
    size_t height;
    size_t object_count;
    int max_translation;
    nn_real learning_rate;
    unsigned int seed;
} NNLayerTaskPolicyConfig;

typedef struct {
    char schema[32];
    size_t train_tasks;
    size_t eval_tasks;
    size_t training_updates;
    size_t solved_tasks;
    size_t top1_tasks;
    size_t verifier_checks;
    nn_real final_loss;
    double exact_rate;
    double top1_accuracy;
    size_t parameters;
    size_t forward_flops;
} NNLayerTaskPolicyReport;

typedef struct {
    NeuralNet *net;
    NNLayerTaskPolicyConfig config;
    NNLayerTaskPolicyReport report;
} NNLayerTaskPolicy;

const char *nn_layer_kind_name(NNLayerKind kind);
void nn_layer_discovery_config_init(NNLayerDiscoveryConfig *config);
void nn_layer_discovery_result_init(NNLayerDiscoveryResult *result);
void nn_layer_discovery_result_free(NNLayerDiscoveryResult *result);
void nn_layer_ranker_config_init(NNLayerRankerConfig *config);
void nn_layer_ranker_init(NNLayerRanker *ranker);
void nn_layer_ranker_free(NNLayerRanker *ranker);
int nn_layer_ranker_train(NNLayerRanker *ranker, const NNLayerRankerConfig *config);
void nn_layer_task_policy_config_init(NNLayerTaskPolicyConfig *config);
void nn_layer_task_policy_init(NNLayerTaskPolicy *policy);
void nn_layer_task_policy_free(NNLayerTaskPolicy *policy);
int nn_layer_task_policy_train(NNLayerTaskPolicy *policy,
                               const NNLayerTaskPolicyConfig *config);
int nn_layer_task_policy_predict(const NNLayerTaskPolicy *policy,
                                 const NNBitGrid *demo_source,
                                 const NNBitGrid *demo_target,
                                 const NNBitGrid *query_source,
                                 NNBitGrid *out_prediction,
                                 int *out_dx,
                                 int *out_dy);
int nn_layer_discover_motion(const NNBitGrid *source,
                             const NNBitGrid *target,
                             const NNLayerDiscoveryConfig *config,
                             NNLayerDiscoveryResult *out_result);
int nn_layer_discover_motion_ranked(const NNBitGrid *source,
                                    const NNBitGrid *target,
                                    const NNLayerDiscoveryConfig *config,
                                    const NNLayerRanker *ranker,
                                    NNLayerDiscoveryResult *out_result);
int nn_layer_recompose(const NNLayerDiscoveryResult *result, NNBitGrid *out);
int nn_layer_apply_global_translation(const NNBitGrid *source,
                                      int dx,
                                      int dy,
                                      NNBitGrid *out);
int nn_layer_discovery_write_json(FILE *fp, const NNLayerDiscoveryResult *result);
int nn_layer_ranker_write_json(FILE *fp, const NNLayerRanker *ranker);
int nn_layer_task_policy_write_json(FILE *fp, const NNLayerTaskPolicy *policy);
int nn_layer_generate_motion_scene(NNBitGrid *source,
                                   NNBitGrid *target,
                                   size_t width,
                                   size_t height,
                                   size_t object_count,
                                   int max_translation,
                                   unsigned int seed);
int nn_layer_generate_global_motion_task(NNBitGrid *demo_source,
                                         NNBitGrid *demo_target,
                                         NNBitGrid *query_source,
                                         NNBitGrid *query_target,
                                         size_t width,
                                         size_t height,
                                         size_t object_count,
                                         int max_translation,
                                         unsigned int seed,
                                         int *out_dx,
                                         int *out_dy);

#endif
