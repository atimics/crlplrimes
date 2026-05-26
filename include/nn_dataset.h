#ifndef NN_DATASET_H
#define NN_DATASET_H

#include "nn_core.h"

#include <stddef.h>
#include <stdint.h>

typedef enum {
    NN_DATASET_SPLIT_TRAIN = 0,
    NN_DATASET_SPLIT_VALID = 1,
    NN_DATASET_SPLIT_TEST = 2,
} NNDatasetSplit;

typedef struct {
    NNDatasetSplit split;
    uint64_t group_id;
    uint64_t state_hash;
    uint32_t action_id;
    int target;
    nn_real *features;
} NNDatasetRow;

typedef struct {
    NNDatasetSplit split;
    uint64_t group_id;
    uint64_t state_hash;
    size_t begin;
    size_t count;
    size_t target_offset;
    size_t target_count;
} NNDatasetGroup;

typedef struct {
    NNDatasetRow *rows;
    size_t row_count;
    size_t row_capacity;
    NNDatasetGroup *groups;
    size_t group_count;
    size_t group_capacity;
    size_t feature_count;
} NNDataset;

enum {
    NN_DATASET_TRACE_REPLAY_FEATURE_COUNT = 10,
};

typedef struct {
    size_t feature_count;
    size_t max_groups;
    size_t max_candidates_per_group;
    const char *corpus_path;
    unsigned int seed;
} NNDatasetExperimentConfig;

typedef int (*NNDatasetExperimentGenerateFn)(const NNDatasetExperimentConfig *config,
                                             NNDataset *dataset);

typedef enum {
    NN_DATASET_LABEL_SINGLE_TARGET = 0,
    NN_DATASET_LABEL_MULTI_TARGET = 1,
} NNDatasetLabelMode;

typedef enum {
    NN_DATASET_CAP_DATASET = 1u << 0,
    NN_DATASET_CAP_RULE_SEARCH = 1u << 1,
    NN_DATASET_CAP_REPLAY = 1u << 2,
    NN_DATASET_CAP_EVIDENCE = 1u << 3,
} NNDatasetExperimentCapability;

typedef struct {
    const char *name;
    const char *description;
    const char *domain;
    const char *feature_schema;
    const char *verifier_schema;
    size_t feature_count;
    NNDatasetLabelMode label_mode;
    NNDatasetExperimentGenerateFn generate;
    unsigned int capabilities;
    const char *rule_search_schema;
    const char *replay_schema;
    const char *evidence_schema;
} NNDatasetExperiment;

typedef NNDatasetExperimentConfig NNDatasetAdapterConfig;
typedef NNDatasetExperimentGenerateFn NNDatasetAdapterGenerateFn;
typedef NNDatasetExperiment NNDatasetAdapter;

int nn_dataset_parse_split(const char *text, NNDatasetSplit *out);
void nn_dataset_init(NNDataset *dataset, size_t feature_count);
void nn_dataset_free(NNDataset *dataset);
int nn_dataset_append_candidate(NNDataset *dataset,
                                NNDatasetSplit split,
                                uint64_t group_id,
                                uint64_t state_hash,
                                uint32_t action_id,
                                int target,
                                const nn_real *features);
int nn_dataset_finalize_group(NNDataset *dataset, size_t begin, size_t end);
int nn_dataset_load_csv(const char *path, size_t feature_count, NNDataset *dataset);
int nn_dataset_load_trace_replay(const char *path,
                                 size_t max_candidates_per_group,
                                 NNDataset *dataset);
int nn_dataset_generate_experiment(const NNDatasetExperiment *experiment,
                                   const NNDatasetExperimentConfig *config,
                                   NNDataset *dataset);

const NNDatasetExperiment *nn_dataset_find_experiment(const char *name);
size_t nn_dataset_experiment_count(void);
const NNDatasetExperiment *nn_dataset_experiment_at(size_t index);
int nn_dataset_experiment_has_capability(const NNDatasetExperiment *experiment,
                                         unsigned int capability);

const NNDatasetAdapter *nn_dataset_find_adapter(const char *name);
size_t nn_dataset_adapter_count(void);
const NNDatasetAdapter *nn_dataset_adapter_at(size_t index);

#endif
