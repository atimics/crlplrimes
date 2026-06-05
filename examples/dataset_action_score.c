#include "nn.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *dataset_path;
    const char *replay_trace_path;
    const char *adapter_name;
    const char *corpus_path;
    const char *trace_path;
    const char *save_path;
    const char *load_path;
    const char *evidence_path;
    const char *task_name;
    size_t feature_count;
    size_t hidden_width;
    size_t epochs;
    size_t rollout_passes;
    size_t max_groups;
    size_t max_candidates_per_group;
    double learning_rate;
    unsigned int seed;
} DSConfig;

typedef struct {
    size_t groups;
    size_t rows;
    size_t correct;
    size_t top5;
    size_t targets;
    double nll;
    double mean_candidates;
    double mean_targets;
} DSMetrics;

static NeuralNet *create_model(const DSConfig *config)
{
    size_t layers[3];
    layers[0] = config->feature_count;
    layers[1] = config->hidden_width;
    layers[2] = 1;
    NNConfig nn_config;
    nn_config.layer_sizes = layers;
    nn_config.layer_count = 3;
    nn_config.seed = config->seed;
    nn_config.hidden_activation = NN_ACTIVATION_RELU;
    nn_config.output_activation = NN_ACTIVATION_LINEAR;
    nn_config.loss = NN_LOSS_MSE;
    return nn_create_config(&nn_config);
}

static int fill_group_inputs(const NNDataset *dataset, const NNDatasetGroup *group, nn_real *inputs)
{
    for (size_t i = 0; i < group->count; ++i) {
        const NNDatasetRow *row = &dataset->rows[group->begin + i];
        memcpy(inputs + i * dataset->feature_count,
               row->features,
               dataset->feature_count * sizeof(inputs[0]));
    }
    return 1;
}

static int train_model(NeuralNet *net, const NNDataset *dataset, const DSConfig *config)
{
    size_t max_candidates = 0;
    for (size_t i = 0; i < dataset->group_count; ++i) {
        if (dataset->groups[i].split == NN_DATASET_SPLIT_TRAIN &&
            dataset->groups[i].count > max_candidates) {
            max_candidates = dataset->groups[i].count;
        }
    }
    if (max_candidates == 0) {
        fputs("dataset has no train groups\n", stderr);
        return 0;
    }
    nn_real *inputs =
        (nn_real *)calloc(max_candidates * dataset->feature_count, sizeof(inputs[0]));
    if (inputs == NULL) {
        return 0;
    }
    for (size_t pass = 0; pass < config->rollout_passes; ++pass) {
        for (size_t epoch = 0; epoch < config->epochs; ++epoch) {
            for (size_t i = 0; i < dataset->group_count; ++i) {
                const NNDatasetGroup *group = &dataset->groups[i];
                if (group->split != NN_DATASET_SPLIT_TRAIN) {
                    continue;
                }
                fill_group_inputs(dataset, group, inputs);
                nn_real target_weight = group->target_count > 0u
                                            ? (nn_real)1 / (nn_real)group->target_count
                                            : (nn_real)1;
                for (size_t j = 0; j < group->count; ++j) {
                    if (!dataset->rows[group->begin + j].target) {
                        continue;
                    }
                    nn_train_softmax_scorer_weighted(net,
                                                     inputs,
                                                     group->count,
                                                     j,
                                                     target_weight,
                                                     (nn_real)config->learning_rate);
                }
            }
        }
    }
    free(inputs);
    return 1;
}

static int group_index_is_target(const NNDataset *dataset,
                                 const NNDatasetGroup *group,
                                 size_t offset)
{
    return dataset != NULL && group != NULL && offset < group->count &&
           dataset->rows[group->begin + offset].target;
}

static int evaluate_split(NeuralNet *net,
                          const NNDataset *dataset,
                          NNDatasetSplit split,
                          FILE *trace,
                          const DSConfig *config,
                          const char *model_hash,
                          DSMetrics *metrics)
{
    memset(metrics, 0, sizeof(*metrics));
    size_t max_candidates = 0;
    for (size_t i = 0; i < dataset->group_count; ++i) {
        if (dataset->groups[i].split == split && dataset->groups[i].count > max_candidates) {
            max_candidates = dataset->groups[i].count;
        }
    }
    if (max_candidates == 0) {
        return 1;
    }
    nn_real *inputs =
        (nn_real *)calloc(max_candidates * dataset->feature_count, sizeof(inputs[0]));
    nn_real *scores = (nn_real *)calloc(max_candidates, sizeof(scores[0]));
    nn_real *probabilities = (nn_real *)calloc(max_candidates, sizeof(probabilities[0]));
    uint32_t *action_ids = (uint32_t *)calloc(max_candidates, sizeof(action_ids[0]));
    size_t *ranked = (size_t *)calloc(max_candidates, sizeof(ranked[0]));
    size_t *target_indices = (size_t *)calloc(max_candidates, sizeof(target_indices[0]));
    if (inputs == NULL || scores == NULL || probabilities == NULL || action_ids == NULL ||
        ranked == NULL || target_indices == NULL) {
        free(inputs);
        free(scores);
        free(probabilities);
        free(action_ids);
        free(ranked);
        free(target_indices);
        return 0;
    }

    for (size_t i = 0; i < dataset->group_count; ++i) {
        const NNDatasetGroup *group = &dataset->groups[i];
        if (group->split != split) {
            continue;
        }
        fill_group_inputs(dataset, group, inputs);
        size_t target_count = 0;
        for (size_t j = 0; j < group->count; ++j) {
            action_ids[j] = dataset->rows[group->begin + j].action_id;
            if (dataset->rows[group->begin + j].target) {
                target_indices[target_count++] = j;
            }
        }
        if (!nn_score_action_rows(net, inputs, group->count, scores, probabilities) ||
            !nn_rank_action_scores_with_ids(scores,
                                            action_ids,
                                            group->count,
                                            ranked,
                                            NN_RANK_SCORE_EPSILON)) {
            free(inputs);
            free(scores);
            free(probabilities);
            free(action_ids);
            free(ranked);
            free(target_indices);
            return 0;
        }
        metrics->groups += 1;
        metrics->rows += group->count;
        metrics->targets += group->target_count;
        metrics->mean_candidates += (double)group->count;
        metrics->mean_targets += (double)group->target_count;
        if (group_index_is_target(dataset, group, ranked[0])) {
            metrics->correct += 1;
        }
        for (size_t r = 0; r < group->count && r < 5; ++r) {
            if (group_index_is_target(dataset, group, ranked[r])) {
                metrics->top5 += 1;
                break;
            }
        }
        double p = 0.0;
        for (size_t j = 0; j < group->count; ++j) {
            if (dataset->rows[group->begin + j].target) {
                p += (double)probabilities[j];
            }
        }
        if (p < 1e-12) {
            p = 1e-12;
        }
        metrics->nll += -log(p);
        if (trace != NULL) {
            NNTraceRecord record;
            memset(&record, 0, sizeof(record));
            record.task_name = config->task_name;
            record.model_hash_hex = model_hash;
            record.state_hash = group->state_hash;
            record.feature_encoder_version = 1;
            record.candidate_count = group->count;
            record.action_ids = action_ids;
            record.scores = scores;
            record.probabilities = probabilities;
            record.selected_index = ranked[0];
            record.target_index = group->target_offset;
            record.target_indices = target_indices;
            record.target_count = target_count;
            record.loss = (nn_real)(-log(p));
            record.verifier_result = "dataset_target";
            record.seed = config->seed;
            record.extra_json = "\"authority\":\"dataset_or_programmatic_verifier\"";
            record.features = &inputs[group->target_offset * dataset->feature_count];
            record.feature_count = dataset->feature_count;
            if (!nn_trace_write_jsonl(trace, &record)) {
                free(inputs);
                free(scores);
                free(probabilities);
                free(action_ids);
                free(ranked);
                free(target_indices);
                return 0;
            }
        }
    }
    if (metrics->groups > 0) {
        metrics->nll /= (double)metrics->groups;
        metrics->mean_candidates /= (double)metrics->groups;
        metrics->mean_targets /= (double)metrics->groups;
    }
    free(inputs);
    free(scores);
    free(probabilities);
    free(action_ids);
    free(ranked);
    free(target_indices);
    return 1;
}

static void print_metrics_json(const char *prefix, const DSMetrics *metrics)
{
    double top1 = metrics->groups > 0 ? (double)metrics->correct / (double)metrics->groups : 0.0;
    double top5 = metrics->groups > 0 ? (double)metrics->top5 / (double)metrics->groups : 0.0;
    printf("\"%s_groups\":%zu,\"%s_rows\":%zu,\"%s_top1\":%.8f,"
           "\"%s_top5\":%.8f,\"%s_nll\":%.8f,\"%s_mean_candidates\":%.8f,"
           "\"%s_targets\":%zu,\"%s_mean_targets\":%.8f",
           prefix,
           metrics->groups,
           prefix,
           metrics->rows,
           prefix,
           top1,
           prefix,
           top5,
           prefix,
           metrics->nll,
           prefix,
           metrics->mean_candidates,
           prefix,
           metrics->targets,
           prefix,
           metrics->mean_targets);
}

static int append_engine_evidence(const char *path,
                                  const DSConfig *config,
                                  const NNDatasetExperiment *experiment,
                                  const char *model_hash,
                                  const char *checkpoint_path,
                                  NeuralNet *net,
                                  const DSMetrics *train_metrics,
                                  const DSMetrics *valid_metrics,
                                  const DSMetrics *test_metrics)
{
    if (path == NULL) {
        return 1;
    }
    FILE *fp = fopen(path, "a");
    if (fp == NULL) {
        fprintf(stderr, "failed to open evidence %s: %s\n", path, strerror(errno));
        return 0;
    }
    int supports_rule_search =
        nn_dataset_experiment_has_capability(experiment, NN_DATASET_CAP_RULE_SEARCH);
    int supports_replay = config->replay_trace_path != NULL ||
                          nn_dataset_experiment_has_capability(experiment,
                                                               NN_DATASET_CAP_REPLAY);
    const char *rule_schema =
        experiment != NULL && experiment->rule_search_schema != NULL
            ? experiment->rule_search_schema
            : "none";
    const char *replay_schema =
        config->replay_trace_path != NULL
            ? "crlp.trace.v1_to_dataset_replay_v1"
            : (experiment != NULL && experiment->replay_schema != NULL
            ? experiment->replay_schema
            : "none");
    const char *evidence_schema =
        experiment != NULL && experiment->evidence_schema != NULL
            ? experiment->evidence_schema
            : "crlp.engine_evidence.v1";
    double train_top1 = train_metrics->groups > 0
                            ? (double)train_metrics->correct / (double)train_metrics->groups
                            : 0.0;
    double valid_top1 = valid_metrics->groups > 0
                            ? (double)valid_metrics->correct / (double)valid_metrics->groups
                            : 0.0;
    double test_top1 = test_metrics->groups > 0
                           ? (double)test_metrics->correct / (double)test_metrics->groups
                           : 0.0;
    double test_top5 = test_metrics->groups > 0
                           ? (double)test_metrics->top5 / (double)test_metrics->groups
                           : 0.0;
    int complete_candidate_groups =
        train_metrics->groups > 0 && valid_metrics->groups > 0 &&
        test_metrics->groups > 0 && train_metrics->rows >= train_metrics->groups &&
        valid_metrics->rows >= valid_metrics->groups &&
        test_metrics->rows >= test_metrics->groups;
    int has_verifier_targets = train_metrics->targets > 0 &&
                               valid_metrics->targets > 0 &&
                               test_metrics->targets > 0;
    int has_checkpoint_identity =
        model_hash != NULL && model_hash[0] != '\0' &&
        checkpoint_path != NULL && checkpoint_path[0] != '\0';
    int has_rule_search_schema = rule_schema != NULL && strcmp(rule_schema, "none") != 0;
    int has_replay_schema = replay_schema != NULL && strcmp(replay_schema, "none") != 0;
    int test_top5_gate = test_top5 >= 0.80;
    int test_top1_floor = test_top1 >= 0.60;
    int smoke_promotion_ready = complete_candidate_groups && has_verifier_targets &&
                                has_checkpoint_identity && has_rule_search_schema &&
                                has_replay_schema && test_top5_gate && test_top1_floor;
    const char *target_label_contract =
        experiment != NULL && experiment->label_mode == NN_DATASET_LABEL_MULTI_TARGET
            ? "multi_target_verifier_positive"
            : "single_target_verifier_positive";
    const char *safe_model_hash = model_hash != NULL ? model_hash : "none";
    const char *safe_checkpoint_path =
        checkpoint_path != NULL ? checkpoint_path : "none";
    fprintf(fp,
            "{\"schema\":\"%s\",\"task\":\"%s\",\"experiment\":\"%s\","
            "\"domain\":\"%s\",\"feature_set\":\"%s\",\"verifier\":\"%s\","
            "\"capabilities\":{\"dataset\":true,\"rule_search\":%s,"
            "\"replay\":%s,\"evidence\":true},"
            "\"rule_search_schema\":\"%s\",\"replay_schema\":\"%s\","
            "\"train_groups\":%zu,\"valid_groups\":%zu,\"test_groups\":%zu,"
            "\"train_rows\":%zu,\"valid_rows\":%zu,\"test_rows\":%zu,"
            "\"train_targets\":%zu,\"valid_targets\":%zu,\"test_targets\":%zu,"
            "\"train_top1\":%.8f,\"valid_top1\":%.8f,\"test_top1\":%.8f,"
            "\"test_top5\":%.8f,\"test_mean_candidates\":%.8f,"
            "\"test_mean_targets\":%.8f,"
            "\"parameters\":%zu,\"forward_flops\":%zu,"
            "\"model_hash\":\"%s\",\"checkpoint_path\":\"%s\","
            "\"evidence_level\":\"dataset_engine_smoke\","
            "\"promotion_level\":\"dataset_engine_promoted_smoke\","
            "\"candidate_group_contract\":\"complete_grouped_candidates\","
            "\"target_label_contract\":\"%s\","
            "\"complete_candidate_groups\":%s,"
            "\"has_verifier_targets\":%s,"
            "\"has_checkpoint_identity\":%s,"
            "\"has_rule_search_schema\":%s,"
            "\"has_replay_schema\":%s,"
            "\"test_top5_gate\":%s,"
            "\"test_top1_floor\":%s,"
            "\"smoke_promotion_ready\":%s}\n",
            evidence_schema,
            config->task_name,
            config->replay_trace_path != NULL
                ? "trace_replay"
                : (experiment != NULL ? experiment->name : "csv"),
            config->replay_trace_path != NULL
                ? "trace_replay"
                : (experiment != NULL ? experiment->domain : "csv"),
            config->replay_trace_path != NULL
                ? "trace_replay_logged_score_v1"
                : (experiment != NULL ? experiment->feature_schema : "dataset_action_rows_v1"),
            config->replay_trace_path != NULL
                ? "trace_target_indices"
                : (experiment != NULL ? experiment->verifier_schema : "csv_target"),
            supports_rule_search ? "true" : "false",
            supports_replay ? "true" : "false",
            rule_schema,
            replay_schema,
            train_metrics->groups,
            valid_metrics->groups,
            test_metrics->groups,
            train_metrics->rows,
            valid_metrics->rows,
            test_metrics->rows,
            train_metrics->targets,
            valid_metrics->targets,
            test_metrics->targets,
            train_top1,
            valid_top1,
            test_top1,
            test_top5,
            test_metrics->mean_candidates,
            test_metrics->mean_targets,
            nn_parameter_count(net),
            nn_forward_flop_count(net),
            safe_model_hash,
            safe_checkpoint_path,
            target_label_contract,
            complete_candidate_groups ? "true" : "false",
            has_verifier_targets ? "true" : "false",
            has_checkpoint_identity ? "true" : "false",
            has_rule_search_schema ? "true" : "false",
            has_replay_schema ? "true" : "false",
            test_top5_gate ? "true" : "false",
            test_top1_floor ? "true" : "false",
            smoke_promotion_ready ? "true" : "false");
    int ok = ferror(fp) == 0;
    fclose(fp);
    return ok;
}

static void sanitize_path_component(const char *input, char *output, size_t output_size)
{
    if (output_size == 0) {
        return;
    }
    size_t out = 0;
    if (input == NULL || input[0] == '\0') {
        input = "dataset_action_score";
    }
    for (size_t i = 0; input[i] != '\0' && out + 1u < output_size; ++i) {
        char c = input[i];
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9') || c == '_' || c == '-') {
            output[out++] = c;
        } else {
            output[out++] = '_';
        }
    }
    output[out] = '\0';
}

static int derive_evidence_checkpoint_path(const DSConfig *config,
                                           char *output,
                                           size_t output_size)
{
    if (config == NULL || config->evidence_path == NULL || output == NULL ||
        output_size == 0) {
        return 0;
    }
    char task[160];
    sanitize_path_component(config->task_name, task, sizeof(task));
    const char *slash = strrchr(config->evidence_path, '/');
    int written = 0;
    if (slash == NULL) {
        written = snprintf(output, output_size, "%s.nnckpt", task);
    } else {
        size_t dir_len = (size_t)(slash - config->evidence_path);
        if (dir_len + 1u >= output_size) {
            return 0;
        }
        memcpy(output, config->evidence_path, dir_len);
        output[dir_len] = '\0';
        written = snprintf(output + dir_len,
                           output_size - dir_len,
                           "/%s.nnckpt",
                           task);
    }
    return written > 0 && (size_t)written < output_size;
}

static void print_usage(FILE *stream)
{
    fputs("usage: dataset_action_score (--dataset path --features N | --replay-trace path | --adapter threshold) "
          "[--experiment name] "
          "[--corpus path] [--task name] [--epochs N] [--hidden N] [--lr R] [--seed N] "
          "[--rollout-passes N] [--max-groups N] [--max-candidates N] "
          "[--trace path] [--save path] [--evidence path]\n"
          "CSV columns: split,group_id,state_hash,action_id,target,f0,...,fN\n",
          stream);
}

static int parse_config(int argc, char **argv, DSConfig *config)
{
    config->dataset_path = NULL;
    config->replay_trace_path = NULL;
    config->adapter_name = NULL;
    config->corpus_path = NULL;
    config->trace_path = NULL;
    config->save_path = NULL;
    config->load_path = NULL;
    config->evidence_path = NULL;
    config->task_name = "dataset_action_score";
    config->feature_count = 0;
    config->hidden_width = 16;
    config->epochs = 200;
    config->rollout_passes = 1;
    config->max_groups = 0;
    config->max_candidates_per_group = 0;
    config->learning_rate = 0.02;
    config->seed = 20260524u;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--dataset") == 0 && i + 1 < argc) {
            config->dataset_path = argv[++i];
        } else if (strcmp(argv[i], "--replay-trace") == 0 && i + 1 < argc) {
            config->replay_trace_path = argv[++i];
        } else if (strcmp(argv[i], "--corpus") == 0 && i + 1 < argc) {
            config->corpus_path = argv[++i];
        } else if ((strcmp(argv[i], "--adapter") == 0 ||
                    strcmp(argv[i], "--experiment") == 0) &&
                   i + 1 < argc) {
            config->adapter_name = argv[++i];
        } else if (strcmp(argv[i], "--trace") == 0 && i + 1 < argc) {
            config->trace_path = argv[++i];
        } else if (strcmp(argv[i], "--save") == 0 && i + 1 < argc) {
            config->save_path = argv[++i];
        } else if (strcmp(argv[i], "--load") == 0 && i + 1 < argc) {
            config->load_path = argv[++i];
        } else if (strcmp(argv[i], "--evidence") == 0 && i + 1 < argc) {
            config->evidence_path = argv[++i];
        } else if (strcmp(argv[i], "--task") == 0 && i + 1 < argc) {
            config->task_name = argv[++i];
        } else if (strcmp(argv[i], "--features") == 0 && i + 1 < argc) {
            if (!nn_parse_size_nonnegative_text(argv[++i], &config->feature_count)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--hidden") == 0 && i + 1 < argc) {
            if (!nn_parse_size_nonnegative_text(argv[++i], &config->hidden_width)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) {
            if (!nn_parse_size_nonnegative_text(argv[++i], &config->epochs)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--rollout-passes") == 0 && i + 1 < argc) {
            if (!nn_parse_size_nonnegative_text(argv[++i], &config->rollout_passes)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--max-groups") == 0 && i + 1 < argc) {
            if (!nn_parse_size_nonnegative_text(argv[++i], &config->max_groups)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--max-candidates") == 0 && i + 1 < argc) {
            if (!nn_parse_size_nonnegative_text(argv[++i], &config->max_candidates_per_group)) {
                return 0;
            }
        } else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) {
            config->learning_rate = strtod(argv[++i], NULL);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            config->seed = (unsigned int)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--help") == 0) {
            print_usage(stdout);
            exit(EXIT_SUCCESS);
        } else {
            return 0;
        }
    }
    const NNDatasetExperiment *experiment = nn_dataset_find_experiment(config->adapter_name);
    if (config->adapter_name != NULL && experiment == NULL) {
        return 0;
    }
    if (experiment != NULL) {
        if (config->feature_count == 0) {
            config->feature_count = experiment->feature_count;
        } else if (config->feature_count != experiment->feature_count) {
            return 0;
        }
    } else if (config->replay_trace_path != NULL) {
        if (config->feature_count == 0) {
            config->feature_count = NN_DATASET_TRACE_REPLAY_FEATURE_COUNT;
        } else if (config->feature_count != NN_DATASET_TRACE_REPLAY_FEATURE_COUNT) {
            return 0;
        }
    }
    return ((config->dataset_path != NULL) + (config->replay_trace_path != NULL) +
                (experiment != NULL) ==
            1) &&
           config->feature_count > 0 && config->hidden_width > 0 &&
           config->epochs > 0 && config->rollout_passes > 0 &&
           config->learning_rate > 0.0;
}

int main(int argc, char **argv)
{
    DSConfig config;
    if (!parse_config(argc, argv, &config)) {
        print_usage(stderr);
        return EXIT_FAILURE;
    }

    NNDataset dataset;
    const NNDatasetExperiment *experiment = nn_dataset_find_experiment(config.adapter_name);
    int loaded = 0;
    if (experiment != NULL) {
        NNDatasetExperimentConfig experiment_config;
        experiment_config.feature_count = config.feature_count;
        experiment_config.max_groups = config.max_groups;
        experiment_config.max_candidates_per_group = config.max_candidates_per_group;
        experiment_config.corpus_path = config.corpus_path;
        experiment_config.seed = config.seed;
        loaded = nn_dataset_generate_experiment(experiment, &experiment_config, &dataset);
    } else {
        if (config.replay_trace_path != NULL) {
            loaded = nn_dataset_load_trace_replay(config.replay_trace_path,
                                                  config.max_candidates_per_group,
                                                  &dataset);
        } else {
            loaded = nn_dataset_load_csv(config.dataset_path, config.feature_count, &dataset);
        }
    }
    if (!loaded) {
        return EXIT_FAILURE;
    }
    NeuralNet *net = NULL;
    if (config.load_path != NULL) {
        NNCheckpointMeta meta;
        net = nn_load_checkpoint(config.load_path, &meta);
        if (net != NULL) {
            /* Continue training with loaded weights */
        }
    }
    if (net == NULL) {
        net = create_model(&config);
    }
    if (net == NULL) {
        nn_dataset_free(&dataset);
        return EXIT_FAILURE;
    }
    if (!train_model(net, &dataset, &config)) {
        nn_free(net);
        nn_dataset_free(&dataset);
        return EXIT_FAILURE;
    }

    char auto_save_path[1024];
    const char *checkpoint_path = config.save_path;
    if (checkpoint_path == NULL && config.evidence_path != NULL) {
        if (!derive_evidence_checkpoint_path(&config, auto_save_path, sizeof(auto_save_path))) {
            fprintf(stderr, "failed to derive checkpoint path for evidence run\n");
            nn_free(net);
            nn_dataset_free(&dataset);
            return EXIT_FAILURE;
        }
        checkpoint_path = auto_save_path;
    }

    char model_hash[65] = "unsaved";
    if (checkpoint_path != NULL) {
        NNCheckpointMeta meta;
        nn_checkpoint_meta_init(&meta, 1);
        if (!nn_checkpoint_meta_set_training_trace_root(&meta, config.task_name) ||
            !nn_checkpoint_meta_set_notes(&meta, "generic dataset action scorer") ||
            !nn_checkpoint_meta_set_feature_set(&meta,
                                                config.replay_trace_path != NULL
                                                    ? "trace_replay_logged_score_v1"
                                                    : (experiment != NULL
                                                    ? experiment->feature_schema
                                                    : "dataset_action_rows_v1")) ||
            !nn_save_checkpoint(net, checkpoint_path, &meta) ||
            !nn_hash_checkpoint_hex(checkpoint_path, model_hash)) {
            fprintf(stderr, "failed to save checkpoint: %s\n", checkpoint_path);
            nn_free(net);
            nn_dataset_free(&dataset);
            return EXIT_FAILURE;
        }
    }

    FILE *trace = NULL;
    if (config.trace_path != NULL) {
        trace = fopen(config.trace_path, "w");
        if (trace == NULL) {
            fprintf(stderr, "failed to open trace %s: %s\n", config.trace_path, strerror(errno));
            nn_free(net);
            nn_dataset_free(&dataset);
            return EXIT_FAILURE;
        }
    }

    DSMetrics train_metrics;
    DSMetrics valid_metrics;
    DSMetrics test_metrics;
    int ok = evaluate_split(net,
                            &dataset,
                            NN_DATASET_SPLIT_TRAIN,
                            trace,
                            &config,
                            model_hash,
                            &train_metrics) &&
             evaluate_split(net,
                            &dataset,
                            NN_DATASET_SPLIT_VALID,
                            trace,
                            &config,
                            model_hash,
                            &valid_metrics) &&
             evaluate_split(net,
                            &dataset,
                            NN_DATASET_SPLIT_TEST,
                            trace,
                            &config,
                            model_hash,
                            &test_metrics);
    if (trace != NULL) {
        fclose(trace);
    }
    if (ok) {
        const char *summary_adapter = config.replay_trace_path != NULL
                                          ? "trace_replay"
                                          : (config.adapter_name != NULL ? config.adapter_name
                                                                         : "csv");
        const char *summary_domain = config.replay_trace_path != NULL
                                         ? "trace_replay"
                                         : (experiment != NULL ? experiment->domain : "csv");
        const char *summary_feature_set =
            config.replay_trace_path != NULL
                ? "trace_replay_logged_score_v1"
                : (experiment != NULL ? experiment->feature_schema
                                      : "dataset_action_rows_v1");
        const char *summary_verifier = config.replay_trace_path != NULL
                                           ? "trace_target_indices"
                                           : (experiment != NULL ? experiment->verifier_schema
                                                                 : "csv_target");
        printf("{\"schema\":\"crlp.dataset_action_score.summary.v1\","
               "\"task\":\"%s\",\"dataset\":\"%s\",\"adapter\":\"%s\","
               "\"experiment\":\"%s\",\"domain\":\"%s\","
               "\"feature_set\":\"%s\",\"verifier\":\"%s\","
               "\"feature_count\":%zu,\"hidden_width\":%zu,\"parameters\":%zu,"
               "\"forward_flops\":%zu,\"epochs\":%zu,\"rollout_passes\":%zu,"
               "\"learning_rate\":%.8f,"
               "\"seed\":%u,\"model_hash\":\"%s\",",
               config.task_name,
               config.replay_trace_path != NULL
                   ? config.replay_trace_path
                   : (config.dataset_path != NULL ? config.dataset_path : "generated"),
               summary_adapter,
               summary_adapter,
               summary_domain,
               summary_feature_set,
               summary_verifier,
               config.feature_count,
               config.hidden_width,
               nn_parameter_count(net),
               nn_forward_flop_count(net),
               config.epochs,
               config.rollout_passes,
               config.learning_rate,
               config.seed,
               model_hash);
        print_metrics_json("train", &train_metrics);
        printf(",");
        print_metrics_json("valid", &valid_metrics);
        printf(",");
        print_metrics_json("test", &test_metrics);
        printf("}\n");
        ok = append_engine_evidence(config.evidence_path,
                                    &config,
                                    experiment,
                                    model_hash,
                                    checkpoint_path,
                                    net,
                                    &train_metrics,
                                    &valid_metrics,
                                    &test_metrics);
    }

    nn_free(net);
    nn_dataset_free(&dataset);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
