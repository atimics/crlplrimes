#include "nn_dataset.h"
#include "nn_bitgrid.h"

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    NN_DATASET_BITWISE_FEATURE_COUNT = 15,
    NN_DATASET_HAIKU_FEATURE_COUNT = 12,
    NN_DATASET_MUSIC_FEATURE_COUNT = 18,
    NN_DATASET_PROOF_FEATURE_COUNT = 16,
};

#include "nn_dataset_music.c"
#include "nn_dataset_proof.c"

typedef struct {
    int known;
    int initial_node;
    uint8_t truth_mask;
    uint8_t op_mask;
    uint8_t left_mask;
    uint8_t right_mask;
    size_t steps;
} NNDatasetBitwiseExpr;

int nn_dataset_parse_split(const char *text, NNDatasetSplit *out)
{
    if (text == NULL || out == NULL) {
        return 0;
    }
    if (strcmp(text, "train") == 0) {
        *out = NN_DATASET_SPLIT_TRAIN;
        return 1;
    }
    if (strcmp(text, "valid") == 0 || strcmp(text, "validation") == 0) {
        *out = NN_DATASET_SPLIT_VALID;
        return 1;
    }
    if (strcmp(text, "test") == 0) {
        *out = NN_DATASET_SPLIT_TEST;
        return 1;
    }
    return 0;
}

void nn_dataset_init(NNDataset *dataset, size_t feature_count)
{
    if (dataset == NULL) {
        return;
    }
    memset(dataset, 0, sizeof(*dataset));
    dataset->feature_count = feature_count;
}

void nn_dataset_free(NNDataset *dataset)
{
    if (dataset == NULL) {
        return;
    }
    for (size_t i = 0; i < dataset->row_count; ++i) {
        free(dataset->rows[i].features);
    }
    free(dataset->rows);
    free(dataset->groups);
    memset(dataset, 0, sizeof(*dataset));
}

static int nn_dataset_append_row(NNDataset *dataset, const NNDatasetRow *row)
{
    if (dataset == NULL || row == NULL) {
        return 0;
    }
    if (dataset->row_count >= dataset->row_capacity) {
        size_t next_capacity = dataset->row_capacity == 0 ? 1024 : dataset->row_capacity * 2u;
        NNDatasetRow *next =
            (NNDatasetRow *)realloc(dataset->rows, next_capacity * sizeof(next[0]));
        if (next == NULL) {
            return 0;
        }
        dataset->rows = next;
        dataset->row_capacity = next_capacity;
    }
    dataset->rows[dataset->row_count++] = *row;
    return 1;
}

static int nn_dataset_append_group(NNDataset *dataset, const NNDatasetGroup *group)
{
    if (dataset == NULL || group == NULL) {
        return 0;
    }
    if (dataset->group_count >= dataset->group_capacity) {
        size_t next_capacity = dataset->group_capacity == 0 ? 256 : dataset->group_capacity * 2u;
        NNDatasetGroup *next =
            (NNDatasetGroup *)realloc(dataset->groups, next_capacity * sizeof(next[0]));
        if (next == NULL) {
            return 0;
        }
        dataset->groups = next;
        dataset->group_capacity = next_capacity;
    }
    dataset->groups[dataset->group_count++] = *group;
    return 1;
}

int nn_dataset_append_candidate(NNDataset *dataset,
                                NNDatasetSplit split,
                                uint64_t group_id,
                                uint64_t state_hash,
                                uint32_t action_id,
                                int target,
                                const nn_real *features)
{
    if (dataset == NULL || dataset->feature_count == 0 || features == NULL) {
        return 0;
    }
    NNDatasetRow row;
    memset(&row, 0, sizeof(row));
    row.split = split;
    row.group_id = group_id;
    row.state_hash = state_hash;
    row.action_id = action_id;
    row.target = target;
    row.features = (nn_real *)calloc(dataset->feature_count, sizeof(row.features[0]));
    if (row.features == NULL) {
        return 0;
    }
    memcpy(row.features, features, dataset->feature_count * sizeof(row.features[0]));
    if (!nn_dataset_append_row(dataset, &row)) {
        free(row.features);
        return 0;
    }
    return 1;
}

int nn_dataset_finalize_group(NNDataset *dataset, size_t begin, size_t end)
{
    if (dataset == NULL || end < begin || end > dataset->row_count) {
        return 0;
    }
    if (end <= begin) {
        return 1;
    }
    NNDatasetGroup group;
    memset(&group, 0, sizeof(group));
    group.split = dataset->rows[begin].split;
    group.group_id = dataset->rows[begin].group_id;
    group.state_hash = dataset->rows[begin].state_hash;
    group.begin = begin;
    group.count = end - begin;
    group.target_offset = (size_t)-1;
    group.target_count = 0;
    for (size_t i = begin; i < end; ++i) {
        if (dataset->rows[i].split != group.split ||
            dataset->rows[i].group_id != group.group_id) {
            fprintf(stderr, "dataset rows must be grouped by split and group_id\n");
            return 0;
        }
        if (dataset->rows[i].target) {
            if (group.target_offset == (size_t)-1) {
                group.target_offset = i - begin;
            }
            group.target_count += 1u;
        }
    }
    if (group.target_offset == (size_t)-1) {
        fprintf(stderr, "group %llu has no target row\n", (unsigned long long)group.group_id);
        return 0;
    }
    return nn_dataset_append_group(dataset, &group);
}

int nn_dataset_generate_experiment(const NNDatasetExperiment *experiment,
                                   const NNDatasetExperimentConfig *config,
                                   NNDataset *dataset)
{
    if (experiment == NULL || experiment->generate == NULL || config == NULL ||
        dataset == NULL || config->feature_count != experiment->feature_count) {
        return 0;
    }
    return experiment->generate(config, dataset);
}

static int parse_u64_field(const char *text, uint64_t *out)
{
    errno = 0;
    char *end = NULL;
    unsigned long long value = strtoull(text, &end, 10);
    if (end == text || *end != '\0' || errno == ERANGE) {
        return 0;
    }
    *out = (uint64_t)value;
    return 1;
}

static int parse_row(char *line, size_t feature_count, NNDatasetRow *row)
{
    memset(row, 0, sizeof(*row));
    char *field = strtok(line, ",");
    if (field == NULL || !nn_dataset_parse_split(field, &row->split)) {
        return 0;
    }
    field = strtok(NULL, ",");
    if (field == NULL || !parse_u64_field(field, &row->group_id)) {
        return 0;
    }
    field = strtok(NULL, ",");
    if (field == NULL || !parse_u64_field(field, &row->state_hash)) {
        return 0;
    }
    field = strtok(NULL, ",");
    uint64_t action = 0;
    if (field == NULL || !parse_u64_field(field, &action) || action > UINT32_MAX) {
        return 0;
    }
    row->action_id = (uint32_t)action;
    field = strtok(NULL, ",");
    if (field == NULL) {
        return 0;
    }
    row->target = atoi(field) != 0;
    row->features = (nn_real *)calloc(feature_count, sizeof(row->features[0]));
    if (row->features == NULL) {
        return 0;
    }
    for (size_t i = 0; i < feature_count; ++i) {
        field = strtok(NULL, ",");
        if (field == NULL) {
            free(row->features);
            row->features = NULL;
            return 0;
        }
        row->features[i] = (nn_real)strtod(field, NULL);
    }
    return 1;
}

int nn_dataset_load_csv(const char *path, size_t feature_count, NNDataset *dataset)
{
    if (path == NULL || feature_count == 0 || dataset == NULL) {
        return 0;
    }
    nn_dataset_init(dataset, feature_count);
    FILE *fp = fopen(path, "r");
    if (fp == NULL) {
        fprintf(stderr, "failed to open dataset %s: %s\n", path, strerror(errno));
        return 0;
    }

    char line[8192];
    size_t group_begin = 0;
    int have_group = 0;
    NNDatasetSplit current_split = NN_DATASET_SPLIT_TRAIN;
    uint64_t current_group = 0;
    while (fgets(line, sizeof(line), fp) != NULL) {
        size_t len = strlen(line);
        while (len > 0 && (line[len - 1u] == '\n' || line[len - 1u] == '\r')) {
            line[--len] = '\0';
        }
        if (line[0] == '\0' || line[0] == '#') {
            continue;
        }
        if (strncmp(line, "split,", 6) == 0) {
            continue;
        }
        NNDatasetRow row;
        if (!parse_row(line, feature_count, &row)) {
            fclose(fp);
            fprintf(stderr, "failed to parse dataset row\n");
            return 0;
        }
        if (have_group &&
            (row.split != current_split || row.group_id != current_group)) {
            if (!nn_dataset_finalize_group(dataset, group_begin, dataset->row_count)) {
                free(row.features);
                fclose(fp);
                return 0;
            }
            group_begin = dataset->row_count;
        }
        current_split = row.split;
        current_group = row.group_id;
        have_group = 1;
        if (!nn_dataset_append_row(dataset, &row)) {
            free(row.features);
            fclose(fp);
            return 0;
        }
    }
    fclose(fp);
    if (have_group && !nn_dataset_finalize_group(dataset, group_begin, dataset->row_count)) {
        return 0;
    }
    return dataset->row_count > 0 && dataset->group_count > 0;
}

static const char *nn_dataset_json_key(const char *line, const char *key)
{
    if (line == NULL || key == NULL) {
        return NULL;
    }
    char pattern[96];
    int written = snprintf(pattern, sizeof(pattern), "\"%s\":", key);
    if (written <= 0 || (size_t)written >= sizeof(pattern)) {
        return NULL;
    }
    const char *match = strstr(line, pattern);
    return match == NULL ? NULL : match + (size_t)written;
}

static int nn_dataset_parse_size_array(const char *line,
                                       const char *key,
                                       size_t *values,
                                       size_t expected_count)
{
    const char *p = nn_dataset_json_key(line, key);
    if (p == NULL || values == NULL || *p != '[') {
        return 0;
    }
    ++p;
    for (size_t i = 0; i < expected_count; ++i) {
        if (*p == '-' || *p == '+') {
            return 0;
        }
        errno = 0;
        char *end = NULL;
        unsigned long long value = strtoull(p, &end, 10);
        if (end == p || errno == ERANGE) {
            return 0;
        }
        values[i] = (size_t)value;
        p = end;
        if (i + 1u < expected_count) {
            if (*p != ',') {
                return 0;
            }
            ++p;
        }
    }
    return *p == ']';
}

static int nn_dataset_trace_target_mask(const char *line,
                                        size_t candidate_count,
                                        int *target_mask,
                                        size_t *out_target_count)
{
    uint64_t target_count_u64 = 0;
    if (!nn_trace_json_get_u64(line, "target_count", &target_count_u64) ||
        target_count_u64 == 0 || target_count_u64 > candidate_count) {
        return 0;
    }
    size_t target_count = (size_t)target_count_u64;
    size_t *target_indices = (size_t *)calloc(target_count, sizeof(target_indices[0]));
    if (target_indices == NULL) {
        return 0;
    }
    int ok = nn_dataset_parse_size_array(line, "target_indices", target_indices, target_count);
    if (ok) {
        memset(target_mask, 0, candidate_count * sizeof(target_mask[0]));
        for (size_t i = 0; i < target_count; ++i) {
            if (target_indices[i] >= candidate_count || target_mask[target_indices[i]]) {
                ok = 0;
                break;
            }
            target_mask[target_indices[i]] = 1;
        }
    }
    free(target_indices);
    if (ok && out_target_count != NULL) {
        *out_target_count = target_count;
    }
    return ok;
}

static NNDatasetSplit nn_dataset_replay_split(size_t group_index)
{
    size_t bucket = group_index % 5u;
    if (bucket < 3u) {
        return NN_DATASET_SPLIT_TRAIN;
    }
    if (bucket == 3u) {
        return NN_DATASET_SPLIT_VALID;
    }
    return NN_DATASET_SPLIT_TEST;
}

static nn_real nn_dataset_safe_scale(size_t value, size_t max_value)
{
    if (max_value == 0u) {
        return (nn_real)0;
    }
    return (nn_real)((double)value / (double)max_value);
}

int nn_dataset_load_trace_replay(const char *path,
                                 size_t max_candidates_per_group,
                                 NNDataset *dataset)
{
    if (path == NULL || dataset == NULL) {
        return 0;
    }
    size_t candidate_capacity = max_candidates_per_group == 0u
                                    ? 1024u
                                    : max_candidates_per_group;
    nn_dataset_init(dataset, NN_DATASET_TRACE_REPLAY_FEATURE_COUNT);
    FILE *fp = fopen(path, "r");
    if (fp == NULL) {
        fprintf(stderr, "failed to open replay trace %s: %s\n", path, strerror(errno));
        return 0;
    }

    char line[65536];
    uint32_t *action_ids = (uint32_t *)calloc(candidate_capacity, sizeof(action_ids[0]));
    nn_real *scores = (nn_real *)calloc(candidate_capacity, sizeof(scores[0]));
    nn_real *probabilities = (nn_real *)calloc(candidate_capacity, sizeof(probabilities[0]));
    size_t *ranked = (size_t *)calloc(candidate_capacity, sizeof(ranked[0]));
    int *target_mask = (int *)calloc(candidate_capacity, sizeof(target_mask[0]));
    if (action_ids == NULL || scores == NULL || probabilities == NULL || ranked == NULL ||
        target_mask == NULL) {
        fclose(fp);
        free(action_ids);
        free(scores);
        free(probabilities);
        free(ranked);
        free(target_mask);
        nn_dataset_free(dataset);
        return 0;
    }

    size_t group_index = 0;
    int ok = 1;
    while (fgets(line, sizeof(line), fp) != NULL) {
        size_t len = strlen(line);
        if (len + 1u == sizeof(line) && line[len - 1u] != '\n') {
            fprintf(stderr, "replay trace line too long: %s\n", path);
            ok = 0;
            break;
        }
        NNParsedTraceRecord record;
        if (!nn_trace_parse_jsonl(line,
                                  &record,
                                  action_ids,
                                  scores,
                                  probabilities,
                                  candidate_capacity)) {
            continue;
        }
        size_t target_count = 0;
        if (!nn_dataset_trace_target_mask(line,
                                          record.candidate_count,
                                          target_mask,
                                          &target_count) ||
            target_count == 0u ||
            !nn_rank_action_scores_with_ids(scores,
                                            action_ids,
                                            record.candidate_count,
                                            ranked,
                                            record.rank_score_epsilon)) {
            ok = 0;
            break;
        }
        size_t *rank_by_index =
            (size_t *)calloc(record.candidate_count, sizeof(rank_by_index[0]));
        if (rank_by_index == NULL) {
            ok = 0;
            break;
        }
        for (size_t rank = 0; rank < record.candidate_count; ++rank) {
            rank_by_index[ranked[rank]] = rank;
        }
        nn_real top_score = scores[ranked[0]];
        size_t begin = dataset->row_count;
        NNDatasetSplit split = nn_dataset_replay_split(group_index);
        uint64_t group_id = group_index + 1u;
        size_t denom = record.candidate_count > 1u ? record.candidate_count - 1u : 1u;
        for (size_t i = 0; i < record.candidate_count; ++i) {
            uint32_t action_id = action_ids[i];
            nn_real features[NN_DATASET_TRACE_REPLAY_FEATURE_COUNT];
            features[0] = (nn_real)1;
            features[1] = (nn_real)((double)(action_id & 0xffffu) / 65535.0);
            features[2] = (nn_real)((double)(action_id >> 16) / 65535.0);
            features[3] = nn_dataset_safe_scale(i, denom);
            features[4] = nn_dataset_safe_scale(record.candidate_count, candidate_capacity);
            features[5] = scores[i];
            features[6] = probabilities[i];
            features[7] = nn_dataset_safe_scale(rank_by_index[i], denom);
            features[8] = i == record.selected_index ? (nn_real)1 : (nn_real)0;
            features[9] = top_score - scores[i];
            if (!isfinite((double)features[5]) || !isfinite((double)features[9]) ||
                !nn_dataset_append_candidate(dataset,
                                             split,
                                             group_id,
                                             record.state_hash,
                                             action_id,
                                             target_mask[i],
                                             features)) {
                ok = 0;
                break;
            }
        }
        free(rank_by_index);
        if (!ok || !nn_dataset_finalize_group(dataset, begin, dataset->row_count)) {
            ok = 0;
            break;
        }
        group_index += 1u;
    }

    fclose(fp);
    free(action_ids);
    free(scores);
    free(probabilities);
    free(ranked);
    free(target_mask);
    if (!ok || group_index == 0u) {
        nn_dataset_free(dataset);
        return 0;
    }
    return 1;
}

static NNDatasetSplit split_for_index(size_t index, size_t train_count, size_t valid_count)
{
    if (index < train_count) {
        return NN_DATASET_SPLIT_TRAIN;
    }
    if (index < train_count + valid_count) {
        return NN_DATASET_SPLIT_VALID;
    }
    return NN_DATASET_SPLIT_TEST;
}

static int generate_threshold_adapter(const NNDatasetAdapterConfig *config, NNDataset *dataset)
{
    nn_dataset_init(dataset, 4);
    const double xs[] = {0.00, 0.10, 0.80, 1.00, 0.20, 0.90, 0.30, 0.70};
    const size_t count = sizeof(xs) / sizeof(xs[0]);
    const size_t train_count = 4;
    const size_t valid_count = 2;
    for (size_t i = 0; i < count; ++i) {
        NNDatasetSplit split = split_for_index(i, train_count, valid_count);
        uint64_t group_id = (uint64_t)i;
        uint64_t state_hash = 100u + (uint64_t)i;
        int target_action = xs[i] >= 0.5 ? 1 : 0;
        size_t begin = dataset->row_count;
        for (uint32_t action = 0; action < 2u; ++action) {
            nn_real features[4];
            features[0] = (nn_real)1.0;
            features[1] = (nn_real)xs[i];
            features[2] = (nn_real)action;
            features[3] = (nn_real)((double)action - xs[i]);
            if (!nn_dataset_append_candidate(dataset,
                                             split,
                                             group_id,
                                             state_hash,
                                             action,
                                             (int)action == target_action,
                                             features)) {
                return 0;
            }
        }
        if (!nn_dataset_finalize_group(dataset, begin, dataset->row_count)) {
            return 0;
        }
    }
    (void)config;
    return 1;
}

static int generate_threshold_multi_adapter(const NNDatasetAdapterConfig *config,
                                            NNDataset *dataset)
{
    nn_dataset_init(dataset, 4);
    const double xs[] = {0.05, 0.15, 0.65, 0.95, 0.25, 0.75, 0.35, 0.85};
    const size_t count = sizeof(xs) / sizeof(xs[0]);
    const size_t train_count = 4;
    const size_t valid_count = 2;
    for (size_t i = 0; i < count; ++i) {
        NNDatasetSplit split = split_for_index(i, train_count, valid_count);
        uint64_t group_id = (uint64_t)i;
        uint64_t state_hash = 200u + (uint64_t)i;
        size_t begin = dataset->row_count;
        for (uint32_t action = 0; action < 3u; ++action) {
            double action_value = (double)action / 2.0;
            nn_real features[4];
            features[0] = (nn_real)1.0;
            features[1] = (nn_real)xs[i];
            features[2] = (nn_real)action_value;
            features[3] = (nn_real)(action_value - xs[i]);
            if (!nn_dataset_append_candidate(dataset,
                                             split,
                                             group_id,
                                             state_hash,
                                             action,
                                             action_value >= xs[i],
                                             features)) {
                return 0;
            }
        }
        if (!nn_dataset_finalize_group(dataset, begin, dataset->row_count)) {
            return 0;
        }
    }
    (void)config;
    return 1;
}

static void encode_bitwise_features(uint8_t target_mask, uint8_t candidate_mask, nn_real features[NN_DATASET_BITWISE_FEATURE_COUNT])
{
    size_t cursor = 0;
    size_t matches = 0;
    features[cursor++] = (nn_real)1.0;
    features[cursor++] = (nn_real)((double)candidate_mask / 15.0);
    for (uint8_t bit = 0; bit < 4u; ++bit) {
        features[cursor++] = (nn_real)((candidate_mask >> bit) & 1u);
    }
    for (uint8_t bit = 0; bit < 4u; ++bit) {
        features[cursor++] = (nn_real)((target_mask >> bit) & 1u);
    }
    for (uint8_t bit = 0; bit < 4u; ++bit) {
        int same = ((candidate_mask >> bit) & 1u) == ((target_mask >> bit) & 1u);
        matches += same ? 1u : 0u;
        features[cursor++] = same ? (nn_real)1.0 : (nn_real)0.0;
    }
    features[cursor++] = (nn_real)((double)matches / 4.0);
    features[cursor++] = (nn_real)((double)(4u - matches) / 4.0);
}

static int generate_bitwise_truth_table_adapter(const NNDatasetAdapterConfig *config,
                                                NNDataset *dataset)
{
    static const uint8_t train_masks[] = {0x0, 0x1, 0x2, 0x4, 0x6, 0x8, 0xa, 0xc, 0xe, 0xf};
    static const uint8_t valid_masks[] = {0x3, 0x7, 0xb};
    static const uint8_t test_masks[] = {0x5, 0x9, 0xd};
    const uint8_t *splits[] = {train_masks, valid_masks, test_masks};
    const size_t counts[] = {
        sizeof(train_masks) / sizeof(train_masks[0]),
        sizeof(valid_masks) / sizeof(valid_masks[0]),
        sizeof(test_masks) / sizeof(test_masks[0]),
    };
    const NNDatasetSplit split_ids[] = {
        NN_DATASET_SPLIT_TRAIN,
        NN_DATASET_SPLIT_VALID,
        NN_DATASET_SPLIT_TEST,
    };

    nn_dataset_init(dataset, NN_DATASET_BITWISE_FEATURE_COUNT);
    uint64_t group_id = 0;
    for (size_t split_index = 0; split_index < 3u; ++split_index) {
        for (size_t i = 0; i < counts[split_index]; ++i) {
            uint8_t target_mask = splits[split_index][i];
            size_t begin = dataset->row_count;
            for (uint8_t candidate = 0; candidate < 16u; ++candidate) {
                nn_real features[NN_DATASET_BITWISE_FEATURE_COUNT];
                encode_bitwise_features(target_mask, candidate, features);
                if (!nn_dataset_append_candidate(dataset,
                                                 split_ids[split_index],
                                                 group_id,
                                                 0xB1700000u + (uint64_t)target_mask,
                                                 candidate,
                                                 candidate == target_mask,
                                                 features)) {
                    return 0;
                }
            }
            if (!nn_dataset_finalize_group(dataset, begin, dataset->row_count)) {
                return 0;
            }
            ++group_id;
        }
    }
    (void)config;
    return 1;
}

static int bitwise_tool_allowed(uint16_t allowed_truth_masks, uint8_t op)
{
    return op < NN_BITWISE_BINARY_TRUTH_TABLES &&
           ((allowed_truth_masks >> op) & 1u) != 0u;
}

static int bitwise_initial_node(uint8_t truth_mask, int *node_out)
{
    if (truth_mask == NN_BITWISE_A) {
        *node_out = 0;
        return 1;
    }
    if (truth_mask == NN_BITWISE_B) {
        *node_out = 1;
        return 1;
    }
    if (truth_mask == NN_BITWISE_FALSE) {
        *node_out = 2;
        return 1;
    }
    if (truth_mask == NN_BITWISE_TRUE) {
        *node_out = 3;
        return 1;
    }
    return 0;
}

static void bitwise_expr_seed(NNDatasetBitwiseExpr exprs[NN_BITWISE_BINARY_TRUTH_TABLES])
{
    for (uint8_t mask = 0; mask < NN_BITWISE_BINARY_TRUTH_TABLES; ++mask) {
        int initial = -1;
        memset(&exprs[mask], 0, sizeof(exprs[mask]));
        exprs[mask].initial_node = -1;
        exprs[mask].truth_mask = mask;
        if (bitwise_initial_node(mask, &initial)) {
            exprs[mask].known = 1;
            exprs[mask].initial_node = initial;
        }
    }
}

static void bitwise_expr_discover(uint16_t allowed_truth_masks,
                                  size_t max_steps,
                                  NNDatasetBitwiseExpr exprs[NN_BITWISE_BINARY_TRUTH_TABLES])
{
    bitwise_expr_seed(exprs);
    int changed = 1;
    while (changed) {
        changed = 0;
        for (uint8_t op = 0; op < NN_BITWISE_BINARY_TRUTH_TABLES; ++op) {
            if (!bitwise_tool_allowed(allowed_truth_masks, op)) {
                continue;
            }
            for (uint8_t left = 0; left < NN_BITWISE_BINARY_TRUTH_TABLES; ++left) {
                if (!exprs[left].known) {
                    continue;
                }
                for (uint8_t right = 0; right < NN_BITWISE_BINARY_TRUTH_TABLES; ++right) {
                    if (!exprs[right].known) {
                        continue;
                    }
                    uint8_t result = nn_bitwise_apply_truth_mask_to_masks(op, left, right);
                    size_t steps = exprs[left].steps + exprs[right].steps + 1u;
                    if (steps > max_steps) {
                        continue;
                    }
                    if (!exprs[result].known || steps < exprs[result].steps) {
                        exprs[result].known = 1;
                        exprs[result].initial_node = -1;
                        exprs[result].truth_mask = result;
                        exprs[result].op_mask = op;
                        exprs[result].left_mask = left;
                        exprs[result].right_mask = right;
                        exprs[result].steps = steps;
                        changed = 1;
                    }
                }
            }
        }
    }
}

static int bitwise_chain_seed(NNBitwiseChain *chain)
{
    if (chain == NULL) {
        return 0;
    }
    nn_bitwise_chain_init(chain);
    chain->nodes[0].kind = NN_BITWISE_CHAIN_INPUT_X;
    chain->nodes[0].truth_mask = NN_BITWISE_A;
    chain->nodes[1].kind = NN_BITWISE_CHAIN_INPUT_Y;
    chain->nodes[1].truth_mask = NN_BITWISE_B;
    chain->nodes[2].kind = NN_BITWISE_CHAIN_CONST_ZERO;
    chain->nodes[2].truth_mask = NN_BITWISE_FALSE;
    chain->nodes[3].kind = NN_BITWISE_CHAIN_CONST_ONE;
    chain->nodes[3].truth_mask = NN_BITWISE_TRUE;
    chain->node_count = 4u;
    return 1;
}

static int bitwise_chain_find_mask(const NNBitwiseChain *chain,
                                   uint8_t truth_mask,
                                   size_t *index_out)
{
    if (chain == NULL) {
        return 0;
    }
    for (size_t i = 0; i < chain->node_count; ++i) {
        if (chain->nodes[i].truth_mask == (truth_mask & 0x0fu)) {
            if (index_out != NULL) {
                *index_out = i;
            }
            return 1;
        }
    }
    return 0;
}

static int bitwise_chain_append(NNBitwiseChain *chain,
                                uint8_t op_mask,
                                size_t left_index,
                                size_t right_index,
                                uint8_t truth_mask)
{
    if (chain == NULL || chain->node_count >= NN_BITWISE_CHAIN_MAX_NODES ||
        left_index >= chain->node_count || right_index >= chain->node_count) {
        return 0;
    }
    NNBitwiseChainNode *node = &chain->nodes[chain->node_count++];
    memset(node, 0, sizeof(*node));
    node->kind = NN_BITWISE_CHAIN_DERIVED;
    node->op_mask = op_mask;
    node->truth_mask = (uint8_t)(truth_mask & 0x0fu);
    node->left_index = left_index;
    node->right_index = right_index;
    chain->step_count = chain->node_count - 4u;
    return 1;
}

static int bitwise_chain_append_expr(const NNDatasetBitwiseExpr exprs[NN_BITWISE_BINARY_TRUTH_TABLES],
                                     uint8_t truth_mask,
                                     NNBitwiseChain *chain,
                                     size_t *index_out)
{
    size_t existing = 0;
    if (bitwise_chain_find_mask(chain, truth_mask, &existing)) {
        *index_out = existing;
        return 1;
    }
    const NNDatasetBitwiseExpr *expr = &exprs[truth_mask & 0x0fu];
    if (!expr->known) {
        return 0;
    }
    if (expr->initial_node >= 0) {
        *index_out = (size_t)expr->initial_node;
        return 1;
    }
    size_t left_index = 0;
    size_t right_index = 0;
    if (!bitwise_chain_append_expr(exprs, expr->left_mask, chain, &left_index) ||
        !bitwise_chain_append_expr(exprs, expr->right_mask, chain, &right_index) ||
        !bitwise_chain_append(chain, expr->op_mask, left_index, right_index, expr->truth_mask)) {
        return 0;
    }
    *index_out = chain->node_count - 1u;
    return 1;
}

static size_t bitwise_expr_steps_missing(const NNDatasetBitwiseExpr exprs[NN_BITWISE_BINARY_TRUTH_TABLES],
                                         const NNBitwiseChain *prefix,
                                         uint8_t truth_mask);

static size_t bitwise_expr_steps_for_child(const NNDatasetBitwiseExpr exprs[NN_BITWISE_BINARY_TRUTH_TABLES],
                                           const NNBitwiseChain *prefix,
                                           uint8_t truth_mask)
{
    if (bitwise_chain_find_mask(prefix, truth_mask, NULL)) {
        return 0u;
    }
    return bitwise_expr_steps_missing(exprs, prefix, truth_mask);
}

static size_t bitwise_expr_steps_missing(const NNDatasetBitwiseExpr exprs[NN_BITWISE_BINARY_TRUTH_TABLES],
                                         const NNBitwiseChain *prefix,
                                         uint8_t truth_mask)
{
    const NNDatasetBitwiseExpr *expr = &exprs[truth_mask & 0x0fu];
    if (!expr->known) {
        return SIZE_MAX;
    }
    if (bitwise_chain_find_mask(prefix, truth_mask, NULL)) {
        return 0u;
    }
    if (expr->initial_node >= 0) {
        return 0u;
    }
    size_t left = bitwise_expr_steps_for_child(exprs, prefix, expr->left_mask);
    size_t right = bitwise_expr_steps_for_child(exprs, prefix, expr->right_mask);
    if (left == SIZE_MAX || right == SIZE_MAX ||
        left > SIZE_MAX - right - 1u) {
        return SIZE_MAX;
    }
    return left + right + 1u;
}

static int bitwise_candidate_can_complete(const NNDatasetBitwiseExpr exprs[NN_BITWISE_BINARY_TRUTH_TABLES],
                                          const NNBitwiseChain *prefix,
                                          uint8_t target_mask,
                                          uint8_t result_mask,
                                          size_t max_steps)
{
    if (prefix == NULL || prefix->step_count >= max_steps) {
        return 0;
    }
    if (result_mask == (target_mask & 0x0fu)) {
        return 1;
    }
    size_t before = bitwise_expr_steps_missing(exprs, prefix, target_mask);
    if (before == SIZE_MAX || before == 0u) {
        return 0;
    }
    NNBitwiseChain next = *prefix;
    if (next.node_count >= NN_BITWISE_CHAIN_MAX_NODES) {
        return 0;
    }
    next.nodes[next.node_count].truth_mask = (uint8_t)(result_mask & 0x0fu);
    next.node_count += 1u;
    next.step_count += 1u;
    size_t remaining_budget = max_steps - next.step_count;
    size_t missing_steps = bitwise_expr_steps_missing(exprs, &next, target_mask);
    return missing_steps != SIZE_MAX && missing_steps < before &&
           missing_steps <= remaining_budget;
}

static int bitwise_build_oracle_chain(const NNDatasetBitwiseExpr exprs[NN_BITWISE_BINARY_TRUTH_TABLES],
                                      uint8_t target_mask,
                                      NNBitwiseChain *chain)
{
    size_t output_index = 0;
    if (!bitwise_chain_seed(chain) ||
        !bitwise_chain_append_expr(exprs, target_mask, chain, &output_index)) {
        return 0;
    }
    chain->target_truth_mask = (uint8_t)(target_mask & 0x0fu);
    chain->output_index = output_index;
    chain->solved = 1;
    return 1;
}

static void encode_bitwise_macro_features(nn_real *features,
                                          uint8_t target_mask,
                                          uint8_t left_mask,
                                          uint8_t right_mask,
                                          uint8_t op_mask,
                                          uint8_t result_mask,
                                          size_t node_count,
                                          size_t step_index)
{
    memset(features, 0, NN_BITWISE_TOOL_FEATURE_COUNT * sizeof(*features));
    for (size_t bit = 0; bit < 4u; ++bit) {
        features[bit] = (nn_real)((target_mask >> bit) & 1u);
        features[4u + bit] = (nn_real)((left_mask >> bit) & 1u);
        features[8u + bit] = (nn_real)((right_mask >> bit) & 1u);
        features[12u + bit] = (nn_real)((op_mask >> bit) & 1u);
        features[16u + bit] = (nn_real)((result_mask >> bit) & 1u);
    }

    size_t matches = 0;
    for (size_t bit = 0; bit < 4u; ++bit) {
        matches += ((target_mask >> bit) & 1u) == ((result_mask >> bit) & 1u) ? 1u : 0u;
    }
    features[20] = (nn_real)((double)matches / 4.0);
    features[21] = result_mask == target_mask ? (nn_real)1 : (nn_real)0;
    features[22] = (nn_real)((double)node_count / (double)NN_BITWISE_CHAIN_MAX_NODES);
    features[23] = (nn_real)((double)step_index / (double)NN_BITWISE_CHAIN_MAX_NODES);
    features[24] = op_mask == NN_BITWISE_XOR ? (nn_real)1 : (nn_real)0;
    features[25] = op_mask == NN_BITWISE_AND ? (nn_real)1 : (nn_real)0;
    features[26] = op_mask == NN_BITWISE_OR ? (nn_real)1 : (nn_real)0;
    features[27] = op_mask == NN_BITWISE_NOT_A ? (nn_real)1 : (nn_real)0;
    features[28] = (nn_real)((double)(4u - matches) / 4.0);
    features[29] = left_mask == target_mask ? (nn_real)1 : (nn_real)0;
    features[30] = right_mask == target_mask ? (nn_real)1 : (nn_real)0;
    features[31] = (nn_real)1;
}

static uint32_t bitwise_macro_action_id(uint8_t op, size_t left, size_t right)
{
    return ((uint32_t)op << 16u) | ((uint32_t)left << 8u) | (uint32_t)right;
}

static int generate_bitwise_macro_step_adapter(const NNDatasetAdapterConfig *config,
                                               NNDataset *dataset)
{
    static const uint8_t train_masks[] = {0x1, 0x2, 0x4, 0x6, 0x8, 0xe};
    static const uint8_t valid_masks[] = {0x3, 0x7, 0xb};
    static const uint8_t test_masks[] = {0x5, 0x9, 0xd};
    const uint8_t *splits[] = {train_masks, valid_masks, test_masks};
    const size_t counts[] = {
        sizeof(train_masks) / sizeof(train_masks[0]),
        sizeof(valid_masks) / sizeof(valid_masks[0]),
        sizeof(test_masks) / sizeof(test_masks[0]),
    };
    const NNDatasetSplit split_ids[] = {
        NN_DATASET_SPLIT_TRAIN,
        NN_DATASET_SPLIT_VALID,
        NN_DATASET_SPLIT_TEST,
    };
    const uint16_t allowed = nn_bitwise_default_tool_mask();
    const size_t max_steps = 4u;
    NNDatasetBitwiseExpr exprs[NN_BITWISE_BINARY_TRUTH_TABLES];
    bitwise_expr_discover(allowed, max_steps, exprs);

    nn_dataset_init(dataset, NN_BITWISE_TOOL_FEATURE_COUNT);
    uint64_t group_id = 0;
    for (size_t split_index = 0; split_index < 3u; ++split_index) {
        for (size_t target_index = 0; target_index < counts[split_index]; ++target_index) {
            uint8_t target_mask = splits[split_index][target_index];
            NNBitwiseChain oracle;
            NNBitwiseChain prefix;
            if (!bitwise_build_oracle_chain(exprs, target_mask, &oracle) ||
                !bitwise_chain_seed(&prefix)) {
                return 0;
            }
            for (size_t step = 4u; step < oracle.node_count; ++step) {
                const NNBitwiseChainNode *wanted = &oracle.nodes[step];
                size_t begin = dataset->row_count;
                for (uint8_t op = 0; op < NN_BITWISE_BINARY_TRUTH_TABLES; ++op) {
                    if (!bitwise_tool_allowed(allowed, op)) {
                        continue;
                    }
                    for (size_t left = 0; left < prefix.node_count; ++left) {
                        for (size_t right = 0; right < prefix.node_count; ++right) {
                            uint8_t result = nn_bitwise_apply_truth_mask_to_masks(
                                op,
                                prefix.nodes[left].truth_mask,
                                prefix.nodes[right].truth_mask);
                            if (bitwise_chain_find_mask(&prefix, result, NULL)) {
                                continue;
                            }
                            nn_real features[NN_BITWISE_TOOL_FEATURE_COUNT];
                            encode_bitwise_macro_features(features,
                                                          target_mask,
                                                          prefix.nodes[left].truth_mask,
                                                          prefix.nodes[right].truth_mask,
                                                          op,
                                                          result,
                                                          prefix.node_count,
                                                          prefix.step_count);
                            int target = bitwise_candidate_can_complete(exprs,
                                                                        &prefix,
                                                                        target_mask,
                                                                        result,
                                                                        max_steps);
                            if (!nn_dataset_append_candidate(dataset,
                                                             split_ids[split_index],
                                                             group_id,
                                                             0xB17A0000u + group_id,
                                                             bitwise_macro_action_id(op, left, right),
                                                             target,
                                                             features)) {
                                return 0;
                            }
                        }
                    }
                }
                if (!nn_dataset_finalize_group(dataset, begin, dataset->row_count)) {
                    return 0;
                }
                if (!bitwise_chain_append(&prefix,
                                          wanted->op_mask,
                                          wanted->left_index,
                                          wanted->right_index,
                                          wanted->truth_mask)) {
                    return 0;
                }
                ++group_id;
            }
        }
    }
    (void)config;
    return dataset->group_count > 0u;
}

typedef struct {
    const char *word;
    uint8_t syllables;
    uint8_t topic_mask;
} NNDatasetHaikuWord;

static const NNDatasetHaikuWord HAIKU_WORDS[] = {
    {"old", 1, 0},       {"pond", 1, 1},     {"waits", 1, 0},
    {"in", 1, 0},        {"rain", 1, 3},     {"frog", 1, 1},
    {"jumps", 1, 0},     {"under", 2, 0},    {"soft", 1, 0},
    {"moon", 1, 2},      {"light", 1, 2},    {"ripples", 2, 1},
    {"carry", 2, 0},     {"night", 1, 2},    {"cold", 1, 4},
    {"wind", 1, 1},      {"through", 1, 0},  {"pine", 1, 1},
    {"snow", 1, 5},      {"mountain", 2, 1}, {"shadows", 2, 2},
    {"cross", 1, 0},     {"river", 2, 1},    {"stars", 1, 2},
    {"open", 2, 0},      {"quiet", 2, 0},    {"stone", 1, 1},
    {"autumn", 2, 4},    {"leaves", 1, 1},   {"fall", 1, 4},
};

static const uint8_t HAIKU_LINES[][8] = {
    {0, 1, 2, 3, 4, 255, 255, 255},
    {5, 6, 7, 8, 9, 10, 255, 255},
    {11, 12, 13, 255, 255, 255, 255, 255},
    {14, 15, 16, 17, 18, 255, 255, 255},
    {19, 20, 21, 22, 255, 255, 255, 255},
    {23, 24, 26, 4, 255, 255, 255, 255},
    {27, 28, 3, 4, 255, 255, 255, 255},
};

static const uint8_t HAIKU_LINE_SPLITS[] = {
    NN_DATASET_SPLIT_TRAIN,
    NN_DATASET_SPLIT_TRAIN,
    NN_DATASET_SPLIT_TRAIN,
    NN_DATASET_SPLIT_VALID,
    NN_DATASET_SPLIT_VALID,
    NN_DATASET_SPLIT_TEST,
    NN_DATASET_SPLIT_TEST,
};

enum {
    HAIKU_CORPUS_MAX_LINE_WORDS = 32,
    HAIKU_CORPUS_WORD_LEN = 64,
};

typedef struct {
    uint16_t words[HAIKU_CORPUS_MAX_LINE_WORDS];
    uint8_t count;
    uint8_t line_mod;
    NNDatasetSplit split;
} NNDatasetHaikuCorpusLine;

typedef struct {
    NNDatasetHaikuWord *words;
    size_t word_count;
    size_t word_capacity;
    NNDatasetHaikuCorpusLine *lines;
    size_t line_count;
    size_t line_capacity;
} NNDatasetHaikuCorpus;

static int haiku_normalize_word(const char *raw, char *out, size_t out_size)
{
    size_t n = 0;
    for (size_t i = 0; raw[i] != '\0'; ++i) {
        unsigned char c = (unsigned char)raw[i];
        if (!isalpha(c)) {
            return 0;
        }
        if (n + 1u >= out_size) {
            return 0;
        }
        out[n++] = (char)tolower(c);
    }
    out[n] = '\0';
    return n > 0u;
}

static void haiku_strip_line(char *line)
{
    size_t len = strlen(line);
    while (len > 0u && (line[len - 1u] == '\n' || line[len - 1u] == '\r' ||
                        isspace((unsigned char)line[len - 1u]))) {
        line[--len] = '\0';
    }
}

static int haiku_syllable_proxy(const char *word)
{
    int count = 0;
    int prev_vowel = 0;
    for (size_t i = 0; word[i] != '\0'; ++i) {
        char c = word[i];
        int vowel = c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u' || c == 'y';
        if (vowel && !prev_vowel) {
            ++count;
        }
        prev_vowel = vowel;
    }
    return count > 0 ? count : 1;
}

static uint8_t haiku_topic_mask(const char *word)
{
    uint8_t mask = 0;
    static const char *const nature[] = {
        "pond", "rain", "frog", "pine", "snow", "river", "stone", "leaves",
        "wind", "moon", "stars", "cedar", "sparrows", "dawn", "ice", "sun",
    };
    static const char *const season[] = {
        "autumn", "winter", "snow", "ice", "rain", "plum", "buds", "spring",
    };
    static const char *const light[] = {
        "moon", "light", "night", "stars", "dawn", "lanterns", "sun",
    };
    for (size_t i = 0; i < sizeof(nature) / sizeof(nature[0]); ++i) {
        if (strcmp(word, nature[i]) == 0) {
            mask |= 1u;
            break;
        }
    }
    for (size_t i = 0; i < sizeof(season) / sizeof(season[0]); ++i) {
        if (strcmp(word, season[i]) == 0) {
            mask |= 2u;
            break;
        }
    }
    for (size_t i = 0; i < sizeof(light) / sizeof(light[0]); ++i) {
        if (strcmp(word, light[i]) == 0) {
            mask |= 4u;
            break;
        }
    }
    return mask;
}

static void haiku_corpus_free(NNDatasetHaikuCorpus *corpus)
{
    if (corpus == NULL) {
        return;
    }
    for (size_t i = 0; i < corpus->word_count; ++i) {
        free((void *)corpus->words[i].word);
    }
    free(corpus->words);
    free(corpus->lines);
    memset(corpus, 0, sizeof(*corpus));
}

static int haiku_corpus_find_word(const NNDatasetHaikuCorpus *corpus, const char *word)
{
    for (size_t i = 0; i < corpus->word_count; ++i) {
        if (strcmp(corpus->words[i].word, word) == 0) {
            return (int)i;
        }
    }
    return -1;
}

static char *haiku_strdup(const char *text)
{
    size_t len = strlen(text);
    char *copy = (char *)calloc(len + 1u, sizeof(copy[0]));
    if (copy != NULL) {
        memcpy(copy, text, len + 1u);
    }
    return copy;
}

static int haiku_corpus_add_word(NNDatasetHaikuCorpus *corpus, const char *word)
{
    int existing = haiku_corpus_find_word(corpus, word);
    if (existing >= 0) {
        return existing;
    }
    if (corpus->word_count >= corpus->word_capacity) {
        size_t next_capacity = corpus->word_capacity == 0u ? 256u : corpus->word_capacity * 2u;
        NNDatasetHaikuWord *next =
            (NNDatasetHaikuWord *)realloc(corpus->words, next_capacity * sizeof(next[0]));
        if (next == NULL) {
            return -1;
        }
        corpus->words = next;
        corpus->word_capacity = next_capacity;
    }
    char *copy = haiku_strdup(word);
    if (copy == NULL) {
        return -1;
    }
    NNDatasetHaikuWord *entry = &corpus->words[corpus->word_count];
    entry->word = copy;
    entry->syllables = (uint8_t)haiku_syllable_proxy(word);
    entry->topic_mask = haiku_topic_mask(word);
    return (int)corpus->word_count++;
}

static int haiku_corpus_add_line(NNDatasetHaikuCorpus *corpus,
                                 int line_mod,
                                 NNDatasetSplit split,
                                 const uint16_t *words,
                                 size_t count)
{
    if (count == 0u || count > HAIKU_CORPUS_MAX_LINE_WORDS) {
        return 0;
    }
    if (corpus->line_count >= corpus->line_capacity) {
        size_t next_capacity = corpus->line_capacity == 0u ? 256u : corpus->line_capacity * 2u;
        NNDatasetHaikuCorpusLine *next =
            (NNDatasetHaikuCorpusLine *)realloc(corpus->lines, next_capacity * sizeof(next[0]));
        if (next == NULL) {
            return 0;
        }
        corpus->lines = next;
        corpus->line_capacity = next_capacity;
    }
    NNDatasetHaikuCorpusLine *line = &corpus->lines[corpus->line_count++];
    memset(line, 0, sizeof(*line));
    line->count = (uint8_t)count;
    line->line_mod = (uint8_t)line_mod;
    line->split = split;
    memcpy(line->words, words, count * sizeof(words[0]));
    return 1;
}

static int haiku_parse_corpus_line(char *line,
                                   NNDatasetHaikuCorpus *corpus,
                                   uint16_t *out_words,
                                   size_t *out_count)
{
    *out_count = 0;
    char *cursor = line;
    while (*cursor != '\0') {
        while (*cursor != '\0' && isspace((unsigned char)*cursor)) {
            ++cursor;
        }
        if (*cursor == '\0') {
            break;
        }
        char *start = cursor;
        while (*cursor != '\0' && !isspace((unsigned char)*cursor)) {
            ++cursor;
        }
        char saved = *cursor;
        *cursor = '\0';
        char word[HAIKU_CORPUS_WORD_LEN];
        if (!haiku_normalize_word(start, word, sizeof(word))) {
            *cursor = saved;
            return 0;
        }
        int id = haiku_corpus_add_word(corpus, word);
        if (id < 0 || *out_count >= HAIKU_CORPUS_MAX_LINE_WORDS || id > UINT16_MAX) {
            *cursor = saved;
            return 0;
        }
        out_words[(*out_count)++] = (uint16_t)id;
        *cursor = saved;
    }
    return *out_count > 0u;
}

static int haiku_load_corpus_file(const char *path, NNDatasetHaikuCorpus *corpus)
{
    memset(corpus, 0, sizeof(*corpus));
    FILE *fp = fopen(path, "r");
    if (fp == NULL) {
        fprintf(stderr, "failed to open haiku corpus %s: %s\n", path, strerror(errno));
        return 0;
    }
    uint16_t stanza[3][HAIKU_CORPUS_MAX_LINE_WORDS];
    size_t stanza_counts[3] = {0, 0, 0};
    int current_line = 0;
    size_t poem_index = 0;
    char line[512];
    while (fgets(line, sizeof(line), fp) != NULL) {
        haiku_strip_line(line);
        if (line[0] == '\0') {
            if (current_line == 3) {
                NNDatasetSplit split = poem_index % 5u < 3u
                                           ? NN_DATASET_SPLIT_TRAIN
                                           : poem_index % 5u == 3u ? NN_DATASET_SPLIT_VALID
                                                                   : NN_DATASET_SPLIT_TEST;
                for (int i = 0; i < 3; ++i) {
                    if (!haiku_corpus_add_line(corpus, i, split, stanza[i], stanza_counts[i])) {
                        fclose(fp);
                        haiku_corpus_free(corpus);
                        return 0;
                    }
                }
                ++poem_index;
            }
            memset(stanza_counts, 0, sizeof(stanza_counts));
            current_line = 0;
            continue;
        }
        if (current_line >= 3 ||
            !haiku_parse_corpus_line(line, corpus, stanza[current_line],
                                     &stanza_counts[current_line])) {
            memset(stanza_counts, 0, sizeof(stanza_counts));
            current_line = 0;
            continue;
        }
        ++current_line;
    }
    if (current_line == 3) {
        NNDatasetSplit split = poem_index % 5u < 3u
                                   ? NN_DATASET_SPLIT_TRAIN
                                   : poem_index % 5u == 3u ? NN_DATASET_SPLIT_VALID
                                                           : NN_DATASET_SPLIT_TEST;
        for (int i = 0; i < 3; ++i) {
            if (!haiku_corpus_add_line(corpus, i, split, stanza[i], stanza_counts[i])) {
                fclose(fp);
                haiku_corpus_free(corpus);
                return 0;
            }
        }
    }
    fclose(fp);
    return corpus->word_count > 0u && corpus->line_count > 0u;
}

static size_t haiku_line_len(size_t line_index)
{
    size_t len = 0;
    while (len < 8u && HAIKU_LINES[line_index][len] != 255u) {
        ++len;
    }
    return len;
}

static int haiku_line_target_syllables(size_t line_index)
{
    return line_index % 3u == 1u ? 7 : 5;
}

static int haiku_transition_exists(int line_mod, int prev_word, int next_word)
{
    for (size_t line = 0; line < sizeof(HAIKU_LINES) / sizeof(HAIKU_LINES[0]); ++line) {
        if ((int)(line % 3u) != line_mod) {
            continue;
        }
        int prev = -1;
        size_t len = haiku_line_len(line);
        for (size_t i = 0; i < len; ++i) {
            int word = HAIKU_LINES[line][i];
            if (prev == prev_word && word == next_word) {
                return 1;
            }
            prev = word;
        }
    }
    return 0;
}

static int haiku_continuation_viable(int line_mod, int prev_word, int remaining_syllables)
{
    if (remaining_syllables == 0) {
        return 1;
    }
    if (remaining_syllables < 0) {
        return 0;
    }
    for (size_t word = 0; word < sizeof(HAIKU_WORDS) / sizeof(HAIKU_WORDS[0]); ++word) {
        int syllables = HAIKU_WORDS[word].syllables;
        if (syllables > remaining_syllables ||
            !haiku_transition_exists(line_mod, prev_word, (int)word)) {
            continue;
        }
        if (haiku_continuation_viable(line_mod, (int)word, remaining_syllables - syllables)) {
            return 1;
        }
    }
    return 0;
}

static int haiku_corpus_transition_exists(const NNDatasetHaikuCorpus *corpus,
                                          int line_mod,
                                          int prev_word,
                                          int next_word)
{
    for (size_t line_index = 0; line_index < corpus->line_count; ++line_index) {
        const NNDatasetHaikuCorpusLine *line = &corpus->lines[line_index];
        if ((int)line->line_mod != line_mod) {
            continue;
        }
        int prev = -1;
        for (size_t i = 0; i < line->count; ++i) {
            int word = line->words[i];
            if (prev == prev_word && word == next_word) {
                return 1;
            }
            prev = word;
        }
    }
    return 0;
}

static int haiku_corpus_continuation_viable(const NNDatasetHaikuCorpus *corpus,
                                            int line_mod,
                                            int prev_word,
                                            int remaining_syllables,
                                            size_t candidate_limit)
{
    if (remaining_syllables == 0) {
        return 1;
    }
    if (remaining_syllables < 0) {
        return 0;
    }
    size_t limit = candidate_limit == 0u || candidate_limit > corpus->word_count
                       ? corpus->word_count
                       : candidate_limit;
    for (size_t word = 0; word < limit; ++word) {
        int syllables = corpus->words[word].syllables;
        if (syllables > remaining_syllables ||
            !haiku_corpus_transition_exists(corpus, line_mod, prev_word, (int)word)) {
            continue;
        }
        if (haiku_corpus_continuation_viable(corpus,
                                             line_mod,
                                             (int)word,
                                             remaining_syllables - syllables,
                                             candidate_limit)) {
            return 1;
        }
    }
    return 0;
}

static uint64_t haiku_state_hash(size_t line, size_t prefix_len, int prev_word, int used_syllables)
{
    uint64_t hash = 1469598103934665603ull;
    hash ^= (uint64_t)line + 0x9e3779b97f4a7c15ull;
    hash *= 1099511628211ull;
    hash ^= (uint64_t)prefix_len;
    hash *= 1099511628211ull;
    hash ^= (uint64_t)(uint32_t)(prev_word + 1);
    hash *= 1099511628211ull;
    hash ^= (uint64_t)(uint32_t)used_syllables;
    return hash;
}

static void encode_haiku_features(nn_real features[NN_DATASET_HAIKU_FEATURE_COUNT],
                                  int line_mod,
                                  int prev_word,
                                  int candidate_word,
                                  int used_syllables,
                                  int target_syllables,
                                  int adjacent,
                                  int viable)
{
    const NNDatasetHaikuWord *candidate = &HAIKU_WORDS[candidate_word];
    memset(features, 0, NN_DATASET_HAIKU_FEATURE_COUNT * sizeof(features[0]));
    features[0] = (nn_real)1.0;
    features[1] = (nn_real)((double)line_mod / 2.0);
    features[2] = (nn_real)((double)used_syllables / 7.0);
    features[3] = (nn_real)((double)(target_syllables - used_syllables) / 7.0);
    features[4] = (nn_real)((double)candidate->syllables / 7.0);
    features[5] = adjacent ? (nn_real)1.0 : (nn_real)0.0;
    features[6] = viable ? (nn_real)1.0 : (nn_real)0.0;
    features[7] = prev_word < 0 ? (nn_real)1.0 : (nn_real)0.0;
    features[8] = (candidate->topic_mask & 1u) ? (nn_real)1.0 : (nn_real)0.0;
    features[9] = (candidate->topic_mask & 2u) ? (nn_real)1.0 : (nn_real)0.0;
    features[10] = (candidate->topic_mask & 4u) ? (nn_real)1.0 : (nn_real)0.0;
    features[11] = (nn_real)((double)candidate_word /
                             (double)((sizeof(HAIKU_WORDS) / sizeof(HAIKU_WORDS[0])) - 1u));
}

static int generate_haiku_adjacency_adapter(const NNDatasetAdapterConfig *config,
                                            NNDataset *dataset)
{
    if (config->corpus_path != NULL && config->corpus_path[0] != '\0') {
        NNDatasetHaikuCorpus corpus;
        if (!haiku_load_corpus_file(config->corpus_path, &corpus)) {
            return 0;
        }
        size_t max_groups = config->max_groups == 0u ? SIZE_MAX : config->max_groups;
        size_t max_candidates = config->max_candidates_per_group == 0u ||
                                        config->max_candidates_per_group > corpus.word_count
                                    ? corpus.word_count
                                    : config->max_candidates_per_group;
        nn_dataset_init(dataset, NN_DATASET_HAIKU_FEATURE_COUNT);
        uint64_t group_id = 0;
        for (size_t line_index = 0;
             line_index < corpus.line_count && dataset->group_count < max_groups;
             ++line_index) {
            const NNDatasetHaikuCorpusLine *line = &corpus.lines[line_index];
            int target_syllables = haiku_line_target_syllables(line->line_mod);
            int total_syllables = 0;
            for (size_t i = 0; i < line->count; ++i) {
                total_syllables += corpus.words[line->words[i]].syllables;
            }
            if (total_syllables != target_syllables) {
                continue;
            }
            int used_syllables = 0;
            int prev_word = -1;
            for (size_t prefix = 0;
                 prefix < line->count && dataset->group_count < max_groups;
                 ++prefix) {
                size_t begin = dataset->row_count;
                for (size_t word = 0; word < max_candidates; ++word) {
                    int adjacent = haiku_corpus_transition_exists(&corpus,
                                                                  line->line_mod,
                                                                  prev_word,
                                                                  (int)word);
                    int remaining =
                        target_syllables - used_syllables - corpus.words[word].syllables;
                    int viable =
                        adjacent &&
                        haiku_corpus_continuation_viable(&corpus,
                                                         line->line_mod,
                                                         (int)word,
                                                         remaining,
                                                         max_candidates);
                    nn_real features[NN_DATASET_HAIKU_FEATURE_COUNT];
                    memset(features, 0, sizeof(features));
                    features[0] = (nn_real)1.0;
                    features[1] = (nn_real)((double)line->line_mod / 2.0);
                    features[2] = (nn_real)((double)used_syllables / 7.0);
                    features[3] = (nn_real)((double)(target_syllables - used_syllables) / 7.0);
                    features[4] = (nn_real)((double)corpus.words[word].syllables / 7.0);
                    features[5] = adjacent ? (nn_real)1.0 : (nn_real)0.0;
                    features[6] = viable ? (nn_real)1.0 : (nn_real)0.0;
                    features[7] = prev_word < 0 ? (nn_real)1.0 : (nn_real)0.0;
                    features[8] = (corpus.words[word].topic_mask & 1u) ? (nn_real)1.0 : (nn_real)0.0;
                    features[9] = (corpus.words[word].topic_mask & 2u) ? (nn_real)1.0 : (nn_real)0.0;
                    features[10] = (corpus.words[word].topic_mask & 4u) ? (nn_real)1.0 : (nn_real)0.0;
                    features[11] = max_candidates <= 1u
                                       ? (nn_real)0
                                       : (nn_real)((double)word / (double)(max_candidates - 1u));
                    if (!nn_dataset_append_candidate(dataset,
                                                     line->split,
                                                     group_id,
                                                     haiku_state_hash(line_index,
                                                                      prefix,
                                                                      prev_word,
                                                                      used_syllables),
                                                     (uint32_t)word,
                                                     viable,
                                                     features)) {
                        haiku_corpus_free(&corpus);
                        return 0;
                    }
                }
                if (!nn_dataset_finalize_group(dataset, begin, dataset->row_count)) {
                    haiku_corpus_free(&corpus);
                    return 0;
                }
                prev_word = line->words[prefix];
                used_syllables += corpus.words[prev_word].syllables;
                ++group_id;
            }
        }
        haiku_corpus_free(&corpus);
        return dataset->group_count > 0u;
    }

    const size_t word_count = sizeof(HAIKU_WORDS) / sizeof(HAIKU_WORDS[0]);
    const size_t line_count = sizeof(HAIKU_LINES) / sizeof(HAIKU_LINES[0]);
    size_t max_groups = config->max_groups == 0u ? SIZE_MAX : config->max_groups;
    size_t max_candidates = config->max_candidates_per_group == 0u
                                ? word_count
                                : config->max_candidates_per_group;
    nn_dataset_init(dataset, NN_DATASET_HAIKU_FEATURE_COUNT);
    uint64_t group_id = 0;
    for (size_t line = 0; line < line_count && dataset->group_count < max_groups; ++line) {
        size_t len = haiku_line_len(line);
        int line_mod = (int)(line % 3u);
        int target_syllables = haiku_line_target_syllables(line);
        int used_syllables = 0;
        int prev_word = -1;
        for (size_t prefix = 0; prefix < len && dataset->group_count < max_groups; ++prefix) {
            size_t begin = dataset->row_count;
            size_t emitted = 0;
            for (size_t word = 0; word < word_count && emitted < max_candidates; ++word) {
                int adjacent = haiku_transition_exists(line_mod, prev_word, (int)word);
                int remaining = target_syllables - used_syllables - HAIKU_WORDS[word].syllables;
                int viable = adjacent &&
                             haiku_continuation_viable(line_mod, (int)word, remaining);
                nn_real features[NN_DATASET_HAIKU_FEATURE_COUNT];
                encode_haiku_features(features,
                                      line_mod,
                                      prev_word,
                                      (int)word,
                                      used_syllables,
                                      target_syllables,
                                      adjacent,
                                      viable);
                if (!nn_dataset_append_candidate(dataset,
                                                 (NNDatasetSplit)HAIKU_LINE_SPLITS[line],
                                                 group_id,
                                                 haiku_state_hash(line, prefix, prev_word, used_syllables),
                                                 (uint32_t)word,
                                                 viable,
                                                 features)) {
                    return 0;
                }
                ++emitted;
            }
            if (!nn_dataset_finalize_group(dataset, begin, dataset->row_count)) {
                return 0;
            }
            prev_word = HAIKU_LINES[line][prefix];
            used_syllables += HAIKU_WORDS[prev_word].syllables;
            ++group_id;
        }
    }
    return dataset->group_count > 0u;
}


enum {
    NN_DATASET_GRID_SIZE = 7,
    NN_DATASET_GRID_CELLS = NN_DATASET_GRID_SIZE * NN_DATASET_GRID_SIZE,
    NN_DATASET_GRID_ACTION_COUNT = 4,
    NN_DATASET_GRID_FEATURE_COUNT = 10 + NN_DATASET_GRID_CELLS + NN_DATASET_GRID_ACTION_COUNT,
    NN_DATASET_GRID_INF = 1000000,
    NN_DATASET_CLAIM_STATE_FEATURES = 6,
    NN_DATASET_CLAIM_ACTION_COUNT = 4,
    NN_DATASET_CLAIM_FEATURE_COUNT = NN_DATASET_CLAIM_STATE_FEATURES + NN_DATASET_CLAIM_ACTION_COUNT,
    NN_DATASET_PRIME_FEATURE_COUNT = 8,
    NN_DATASET_ARC_FEATURE_COUNT = 16,
    NN_DATASET_PROGRAM_FEATURE_COUNT = 18,
    NN_DATASET_SIGNAL_ACTION_COUNT = 9,
    NN_DATASET_SIGNAL_FEATURE_COUNT = 20,
};

typedef struct {
    uint8_t blocked[NN_DATASET_GRID_CELLS];
    int sx;
    int sy;
    int gx;
    int gy;
} NNDatasetGridState;

typedef struct {
    nn_real state[NN_DATASET_CLAIM_STATE_FEATURES];
    size_t target;
    nn_real utility[NN_DATASET_CLAIM_ACTION_COUNT];
} NNDatasetClaimState;

typedef struct {
    int output_type;
    int depth;
    int nodes;
    int primitives;
    int uses_map;
    int uses_filter;
    int uses_fold;
    int uses_compare;
    int uses_arithmetic;
    int uses_ordering;
    int arity;
    int composed;
    int predicate;
} NNDatasetProgramOp;

static uint64_t dataset_mix64(uint64_t x)
{
    x ^= x >> 33;
    x *= UINT64_C(0xff51afd7ed558ccd);
    x ^= x >> 33;
    x *= UINT64_C(0xc4ceb9fe1a85ec53);
    x ^= x >> 33;
    return x;
}

static uint32_t dataset_rng_next(uint32_t *state)
{
    *state = (*state * 1664525u) + 1013904223u;
    return *state;
}

static double dataset_rng_unit(uint32_t *state)
{
    return (double)dataset_rng_next(state) / (double)UINT32_MAX;
}

static int dataset_grid_cell(int x, int y)
{
    return y * NN_DATASET_GRID_SIZE + x;
}

static int dataset_grid_in_bounds(int x, int y)
{
    return x >= 0 && x < NN_DATASET_GRID_SIZE && y >= 0 && y < NN_DATASET_GRID_SIZE;
}

static const int DATASET_GRID_DX[NN_DATASET_GRID_ACTION_COUNT] = {0, 1, 0, -1};
static const int DATASET_GRID_DY[NN_DATASET_GRID_ACTION_COUNT] = {-1, 0, 1, 0};

static void dataset_grid_distances_from_goal(const NNDatasetGridState *state, int *dist)
{
    int queue[NN_DATASET_GRID_CELLS];
    size_t head = 0;
    size_t tail = 0;
    for (size_t i = 0; i < NN_DATASET_GRID_CELLS; ++i) {
        dist[i] = NN_DATASET_GRID_INF;
    }
    int goal = dataset_grid_cell(state->gx, state->gy);
    dist[goal] = 0;
    queue[tail++] = goal;
    while (head < tail) {
        int cell = queue[head++];
        int x = cell % NN_DATASET_GRID_SIZE;
        int y = cell / NN_DATASET_GRID_SIZE;
        for (size_t action = 0; action < NN_DATASET_GRID_ACTION_COUNT; ++action) {
            int nx = x + DATASET_GRID_DX[action];
            int ny = y + DATASET_GRID_DY[action];
            if (!dataset_grid_in_bounds(nx, ny)) {
                continue;
            }
            int next = dataset_grid_cell(nx, ny);
            if (state->blocked[next] || dist[next] != NN_DATASET_GRID_INF) {
                continue;
            }
            dist[next] = dist[cell] + 1;
            queue[tail++] = next;
        }
    }
}

static int dataset_generate_grid_state(uint64_t seed, NNDatasetGridState *state)
{
    for (uint64_t attempt = 0; attempt < 1000u; ++attempt) {
        uint32_t rng = (uint32_t)(seed ^ (seed >> 32) ^ (attempt * UINT64_C(0x9e3779b9)));
        memset(state, 0, sizeof(*state));
        state->sx = (int)(dataset_rng_next(&rng) % NN_DATASET_GRID_SIZE);
        state->sy = (int)(dataset_rng_next(&rng) % NN_DATASET_GRID_SIZE);
        do {
            state->gx = (int)(dataset_rng_next(&rng) % NN_DATASET_GRID_SIZE);
            state->gy = (int)(dataset_rng_next(&rng) % NN_DATASET_GRID_SIZE);
        } while (state->gx == state->sx && state->gy == state->sy);
        for (int y = 0; y < NN_DATASET_GRID_SIZE; ++y) {
            for (int x = 0; x < NN_DATASET_GRID_SIZE; ++x) {
                int cell = dataset_grid_cell(x, y);
                if ((x == state->sx && y == state->sy) ||
                    (x == state->gx && y == state->gy)) {
                    continue;
                }
                state->blocked[cell] = dataset_rng_unit(&rng) < 0.18 ? 1u : 0u;
            }
        }
        int dist[NN_DATASET_GRID_CELLS];
        dataset_grid_distances_from_goal(state, dist);
        int start = dataset_grid_cell(state->sx, state->sy);
        if (dist[start] > 0 && dist[start] < NN_DATASET_GRID_INF) {
            return 1;
        }
    }
    return 0;
}

static void dataset_fill_grid_features(const NNDatasetGridState *state,
                                       size_t action,
                                       const int *dist,
                                       nn_real *features)
{
    int nx = state->sx + DATASET_GRID_DX[action];
    int ny = state->sy + DATASET_GRID_DY[action];
    int feature_x = state->sx;
    int feature_y = state->sy;
    if (dataset_grid_in_bounds(nx, ny)) {
        feature_x = nx;
        feature_y = ny;
    }
    int manhattan_before = abs(state->gx - state->sx) + abs(state->gy - state->sy);
    int manhattan_after = abs(state->gx - feature_x) + abs(state->gy - feature_y);
    memset(features, 0, NN_DATASET_GRID_FEATURE_COUNT * sizeof(features[0]));
    features[0] = (nn_real)((double)state->sx / (double)(NN_DATASET_GRID_SIZE - 1));
    features[1] = (nn_real)((double)state->sy / (double)(NN_DATASET_GRID_SIZE - 1));
    features[2] = (nn_real)((double)state->gx / (double)(NN_DATASET_GRID_SIZE - 1));
    features[3] = (nn_real)((double)state->gy / (double)(NN_DATASET_GRID_SIZE - 1));
    features[4] = (nn_real)((double)(state->gx - feature_x) / (double)(NN_DATASET_GRID_SIZE - 1));
    features[5] = (nn_real)((double)(state->gy - feature_y) / (double)(NN_DATASET_GRID_SIZE - 1));
    features[6] = (nn_real)((double)manhattan_after / (double)(2 * (NN_DATASET_GRID_SIZE - 1)));
    features[7] = (nn_real)((double)(manhattan_before - manhattan_after) / 2.0);
    features[8] = (nn_real)DATASET_GRID_DX[action];
    features[9] = (nn_real)DATASET_GRID_DY[action];
    (void)dist;
    for (size_t cell = 0; cell < NN_DATASET_GRID_CELLS; ++cell) {
        features[10 + cell] = state->blocked[cell] ? (nn_real)1 : (nn_real)0;
    }
    for (size_t i = 0; i < NN_DATASET_GRID_ACTION_COUNT; ++i) {
        features[10 + NN_DATASET_GRID_CELLS + i] = i == action ? (nn_real)1 : (nn_real)0;
    }
}

static int generate_grid_path_adapter(const NNDatasetAdapterConfig *config, NNDataset *dataset)
{
    nn_dataset_init(dataset, NN_DATASET_GRID_FEATURE_COUNT);
    size_t max_groups = config->max_groups == 0u ? 96u : config->max_groups;
    size_t train_count = (max_groups * 3u) / 5u;
    size_t valid_count = max_groups / 5u;
    uint64_t seed_base = config->seed == 0u ? UINT64_C(20260524) : (uint64_t)config->seed;
    for (size_t i = 0; i < max_groups; ++i) {
        NNDatasetGridState state;
        if (!dataset_generate_grid_state(seed_base + i * 17u, &state)) {
            return 0;
        }
        int dist[NN_DATASET_GRID_CELLS];
        dataset_grid_distances_from_goal(&state, dist);
        int best_dist = NN_DATASET_GRID_INF;
        for (size_t action = 0; action < NN_DATASET_GRID_ACTION_COUNT; ++action) {
            int nx = state.sx + DATASET_GRID_DX[action];
            int ny = state.sy + DATASET_GRID_DY[action];
            int next_dist = NN_DATASET_GRID_INF;
            if (dataset_grid_in_bounds(nx, ny) && !state.blocked[dataset_grid_cell(nx, ny)]) {
                next_dist = dist[dataset_grid_cell(nx, ny)];
            }
            if (next_dist < best_dist) {
                best_dist = next_dist;
            }
        }
        NNDatasetSplit split = split_for_index(i, train_count, valid_count);
        size_t begin = dataset->row_count;
        uint64_t state_hash = dataset_mix64(seed_base + i * 17u);
        for (size_t action = 0; action < NN_DATASET_GRID_ACTION_COUNT; ++action) {
            int nx = state.sx + DATASET_GRID_DX[action];
            int ny = state.sy + DATASET_GRID_DY[action];
            int next_dist = NN_DATASET_GRID_INF;
            if (dataset_grid_in_bounds(nx, ny) && !state.blocked[dataset_grid_cell(nx, ny)]) {
                next_dist = dist[dataset_grid_cell(nx, ny)];
            }
            nn_real features[NN_DATASET_GRID_FEATURE_COUNT];
            dataset_fill_grid_features(&state, action, dist, features);
            if (!nn_dataset_append_candidate(dataset,
                                             split,
                                             (uint64_t)i,
                                             state_hash,
                                             (uint32_t)action,
                                             next_dist == best_dist,
                                             features)) {
                return 0;
            }
        }
        if (!nn_dataset_finalize_group(dataset, begin, dataset->row_count)) {
            return 0;
        }
    }
    return 1;
}

static double dataset_claim_truth(const nn_real *features)
{
    return 0.45 * (double)features[0] + 0.35 * (double)features[1] +
           0.20 * (double)features[4] - 0.30 * (double)features[2] -
           0.15 * (double)features[5] - 0.10 * (double)features[3];
}

static void dataset_claim_utilities(const nn_real *features,
                                    nn_real utility[NN_DATASET_CLAIM_ACTION_COUNT])
{
    double truth = dataset_claim_truth(features);
    int resolves_true = truth >= 0.38;
    double uncertainty = truth >= 0.38 ? truth - 0.38 : 0.38 - truth;
    utility[0] = resolves_true ? (nn_real)2.0 : (nn_real)-2.5;
    utility[1] = resolves_true ? (nn_real)-0.3 : (nn_real)0.3;
    utility[2] = (nn_real)(0.95 - 0.6 * uncertainty - 0.25 * (double)features[3]);
    utility[3] = resolves_true ? (nn_real)1.1 : (nn_real)-0.2;
    if ((double)features[2] > 0.70 || (double)features[5] > 0.80) {
        utility[2] += (nn_real)0.35;
        utility[0] -= (nn_real)0.75;
    }
}

static NNDatasetClaimState dataset_generate_claim_state(uint64_t seed)
{
    uint32_t rng = (uint32_t)(seed ^ (seed >> 32) ^ UINT32_C(0x9e3779b9));
    NNDatasetClaimState state;
    memset(&state, 0, sizeof(state));
    for (size_t i = 0; i < NN_DATASET_CLAIM_STATE_FEATURES; ++i) {
        state.state[i] = (nn_real)dataset_rng_unit(&rng);
    }
    dataset_claim_utilities(state.state, state.utility);
    state.target = 0;
    for (size_t action = 1; action < NN_DATASET_CLAIM_ACTION_COUNT; ++action) {
        if (state.utility[action] > state.utility[state.target]) {
            state.target = action;
        }
    }
    return state;
}

static int generate_delayed_claim_adapter(const NNDatasetAdapterConfig *config, NNDataset *dataset)
{
    nn_dataset_init(dataset, NN_DATASET_CLAIM_FEATURE_COUNT);
    size_t max_groups = config->max_groups == 0u ? 96u : config->max_groups;
    size_t train_count = (max_groups * 3u) / 5u;
    size_t valid_count = max_groups / 5u;
    uint64_t seed_base = config->seed == 0u ? UINT64_C(1000) : (uint64_t)config->seed;
    for (size_t i = 0; i < max_groups; ++i) {
        NNDatasetClaimState state = dataset_generate_claim_state(seed_base + i);
        NNDatasetSplit split = split_for_index(i, train_count, valid_count);
        size_t begin = dataset->row_count;
        uint64_t state_hash = dataset_mix64(seed_base + i);
        for (size_t action = 0; action < NN_DATASET_CLAIM_ACTION_COUNT; ++action) {
            nn_real features[NN_DATASET_CLAIM_FEATURE_COUNT];
            memcpy(features, state.state, NN_DATASET_CLAIM_STATE_FEATURES * sizeof(features[0]));
            for (size_t j = 0; j < NN_DATASET_CLAIM_ACTION_COUNT; ++j) {
                features[NN_DATASET_CLAIM_STATE_FEATURES + j] = action == j ? (nn_real)1 : (nn_real)0;
            }
            if (!nn_dataset_append_candidate(dataset,
                                             split,
                                             (uint64_t)i,
                                             state_hash,
                                             (uint32_t)action,
                                             action == state.target,
                                             features)) {
                return 0;
            }
        }
        if (!nn_dataset_finalize_group(dataset, begin, dataset->row_count)) {
            return 0;
        }
    }
    return 1;
}

static int dataset_is_prime_u64(uint64_t value)
{
    if (value < 2u) {
        return 0;
    }
    if (value == 2u) {
        return 1;
    }
    if ((value & 1u) == 0u) {
        return 0;
    }
    for (uint64_t d = 3u; d * d <= value; d += 2u) {
        if (value % d == 0u) {
            return 0;
        }
    }
    return 1;
}

static uint64_t dataset_nth_odd_prime(size_t index)
{
    uint64_t value = 3u;
    size_t seen = 0;
    while (1) {
        if (dataset_is_prime_u64(value)) {
            if (seen == index) {
                return value;
            }
            ++seen;
        }
        value += 2u;
    }
}

static int generate_prime_gap_candidate_adapter(const NNDatasetAdapterConfig *config,
                                                NNDataset *dataset)
{
    nn_dataset_init(dataset, NN_DATASET_PRIME_FEATURE_COUNT);
    static const uint32_t gaps[] = {2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30};
    size_t max_groups = config->max_groups == 0u ? 96u : config->max_groups;
    size_t train_count = (max_groups * 3u) / 5u;
    size_t valid_count = max_groups / 5u;
    for (size_t i = 0; i < max_groups; ++i) {
        uint64_t p = dataset_nth_odd_prime(i + 4u);
        uint32_t best_gap = 0;
        for (uint32_t gap = 2u; gap <= 64u; gap += 2u) {
            if (dataset_is_prime_u64(p + gap)) {
                best_gap = gap;
                break;
            }
        }
        if (best_gap == 0u) {
            return 0;
        }
        NNDatasetSplit split = split_for_index(i, train_count, valid_count);
        size_t begin = dataset->row_count;
        for (size_t g = 0; g < sizeof(gaps) / sizeof(gaps[0]); ++g) {
            uint32_t gap = gaps[g];
            nn_real features[NN_DATASET_PRIME_FEATURE_COUNT];
            features[0] = (nn_real)1;
            features[1] = (nn_real)((double)(p % 3u) / 2.0);
            features[2] = (nn_real)((double)(p % 5u) / 4.0);
            features[3] = (nn_real)((double)(p % 7u) / 6.0);
            features[4] = (nn_real)((double)gap / 64.0);
            features[5] = (nn_real)(((p + gap) % 3u) == 0u ? 1 : 0);
            features[6] = (nn_real)(((p + gap) % 5u) == 0u ? 1 : 0);
            features[7] = (nn_real)(((p + gap) % 7u) == 0u ? 1 : 0);
            if (!nn_dataset_append_candidate(dataset,
                                             split,
                                             (uint64_t)i,
                                             dataset_mix64(p),
                                             gap,
                                             gap == best_gap,
                                             features)) {
                return 0;
            }
        }
        if (!nn_dataset_finalize_group(dataset, begin, dataset->row_count)) {
            return 0;
        }
    }
    return 1;
}

static const uint8_t DATASET_ARC_INPUTS[3][21] = {
    {1,0,0,5,0,1,0, 0,1,0,5,1,1,1, 1,0,0,5,0,0,0},
    {1,1,0,5,0,1,0, 0,0,1,5,1,1,1, 1,1,0,5,0,1,0},
    {0,0,1,5,0,0,0, 1,1,0,5,1,0,1, 0,1,1,5,1,0,1},
};
static const uint8_t DATASET_ARC_OUTPUTS[3][9] = {
    {0,0,0, 0,2,0, 0,0,0},
    {0,2,0, 0,0,2, 0,2,0},
    {0,0,0, 2,0,0, 0,0,2},
};
static const uint8_t DATASET_ARC_TEST_INPUT[21] = {
    1,0,1,5,1,0,1, 0,1,0,5,1,0,1, 1,0,1,5,0,1,0,
};
static const uint8_t DATASET_ARC_TEST_OUTPUT[9] = {
    2,0,2, 0,0,0, 0,0,0,
};

static void dataset_arc_features(const uint8_t *input,
                                 size_t cell,
                                 size_t color,
                                 nn_real features[NN_DATASET_ARC_FEATURE_COUNT])
{
    size_t row = cell / 3u;
    size_t col = cell % 3u;
    uint8_t left = input[row * 7u + col];
    uint8_t right = input[row * 7u + 4u + col];
    memset(features, 0, NN_DATASET_ARC_FEATURE_COUNT * sizeof(features[0]));
    features[0] = (nn_real)1;
    features[1] = (nn_real)((double)row / 2.0);
    features[2] = (nn_real)((double)col / 2.0);
    features[3] = (nn_real)((double)color / 9.0);
    features[4] = left == 1u ? (nn_real)1 : (nn_real)0;
    features[5] = right == 1u ? (nn_real)1 : (nn_real)0;
    features[6] = left == right ? (nn_real)1 : (nn_real)0;
    features[7] = left != right ? (nn_real)1 : (nn_real)0;
    features[8] = left == 1u && right == 1u ? (nn_real)1 : (nn_real)0;
    features[9] = left == 1u || right == 1u ? (nn_real)1 : (nn_real)0;
    features[10] = color == 0u ? (nn_real)1 : (nn_real)0;
    features[11] = color == 2u ? (nn_real)1 : (nn_real)0;
    features[12] = (left == 1u && right == 1u && color == 2u) ? (nn_real)1 : (nn_real)0;
    features[13] = (left != right && color == 0u) ? (nn_real)1 : (nn_real)0;
    features[14] = (nn_real)((double)(row + col) / 4.0);
    features[15] = (nn_real)((double)(row == col ? 1 : 0));
}

static int dataset_append_arc_cells(NNDataset *dataset,
                                    NNDatasetSplit split,
                                    uint64_t group_base,
                                    const uint8_t *input,
                                    const uint8_t *output)
{
    for (size_t cell = 0; cell < 9u; ++cell) {
        size_t begin = dataset->row_count;
        uint64_t group_id = group_base + cell;
        for (size_t color = 0; color < 10u; ++color) {
            nn_real features[NN_DATASET_ARC_FEATURE_COUNT];
            dataset_arc_features(input, cell, color, features);
            if (!nn_dataset_append_candidate(dataset,
                                             split,
                                             group_id,
                                             dataset_mix64(group_id),
                                             (uint32_t)color,
                                             color == output[cell],
                                             features)) {
                return 0;
            }
        }
        if (!nn_dataset_finalize_group(dataset, begin, dataset->row_count)) {
            return 0;
        }
    }
    return 1;
}

static int generate_arc_cell_color_adapter(const NNDatasetAdapterConfig *config,
                                           NNDataset *dataset)
{
    nn_dataset_init(dataset, NN_DATASET_ARC_FEATURE_COUNT);
    (void)config;
    for (size_t pair = 0; pair < 3u; ++pair) {
        NNDatasetSplit split = pair < 2u ? NN_DATASET_SPLIT_TRAIN : NN_DATASET_SPLIT_VALID;
        if (!dataset_append_arc_cells(dataset,
                                      split,
                                      100u + pair * 16u,
                                      DATASET_ARC_INPUTS[pair],
                                      DATASET_ARC_OUTPUTS[pair])) {
            return 0;
        }
    }
    return dataset_append_arc_cells(dataset,
                                    NN_DATASET_SPLIT_TEST,
                                    1000u,
                                    DATASET_ARC_TEST_INPUT,
                                    DATASET_ARC_TEST_OUTPUT);
}

static const NNDatasetProgramOp DATASET_PROGRAM_OPS[] = {
    {0,1,1,1,0,0,0,0,0,0,1,0,0}, {0,2,2,1,0,0,0,0,0,0,1,0,0},
    {0,3,4,2,0,0,0,0,0,1,1,1,0}, {0,3,5,2,1,0,0,0,1,0,1,1,0},
    {0,3,5,2,0,1,0,1,0,0,1,0,1}, {0,3,5,2,0,1,0,1,0,0,1,0,1},
    {1,2,3,1,0,0,1,0,1,0,1,0,0}, {1,2,2,1,0,0,0,0,0,0,1,0,0},
    {2,2,2,1,0,0,0,1,0,0,1,0,1}, {1,4,7,3,1,0,1,0,1,0,1,1,0},
    {1,4,8,3,1,0,1,0,1,0,1,1,0}, {1,3,6,2,0,1,1,1,1,0,1,1,1},
};

static void dataset_program_features(size_t op,
                                     size_t target_op,
                                     nn_real features[NN_DATASET_PROGRAM_FEATURE_COUNT])
{
    const NNDatasetProgramOp *spec = &DATASET_PROGRAM_OPS[op];
    memset(features, 0, NN_DATASET_PROGRAM_FEATURE_COUNT * sizeof(features[0]));
    features[0] = (nn_real)((double)target_op / (double)(sizeof(DATASET_PROGRAM_OPS) / sizeof(DATASET_PROGRAM_OPS[0]) - 1u));
    features[1] = spec->output_type == 0 ? (nn_real)1 : (nn_real)0;
    features[2] = spec->output_type == 1 ? (nn_real)1 : (nn_real)0;
    features[3] = spec->output_type == 2 ? (nn_real)1 : (nn_real)0;
    features[4] = (nn_real)((double)spec->depth / 8.0);
    features[5] = (nn_real)((double)spec->nodes / 10.0);
    features[6] = (nn_real)((double)spec->primitives / 4.0);
    features[7] = spec->uses_map ? (nn_real)1 : (nn_real)0;
    features[8] = spec->uses_filter ? (nn_real)1 : (nn_real)0;
    features[9] = spec->uses_fold ? (nn_real)1 : (nn_real)0;
    features[10] = spec->uses_compare ? (nn_real)1 : (nn_real)0;
    features[11] = spec->uses_arithmetic ? (nn_real)1 : (nn_real)0;
    features[12] = spec->uses_ordering ? (nn_real)1 : (nn_real)0;
    features[13] = (nn_real)((double)spec->arity / 3.0);
    features[14] = (nn_real)((double)op / (double)(sizeof(DATASET_PROGRAM_OPS) / sizeof(DATASET_PROGRAM_OPS[0]) - 1u));
    features[15] = spec->composed ? (nn_real)1 : (nn_real)0;
    features[16] = spec->predicate ? (nn_real)1 : (nn_real)0;
    size_t delta = op > target_op ? op - target_op : target_op - op;
    features[17] = (nn_real)(1.0 / (1.0 + (double)delta));
}

static int generate_program_synthesis_ops_adapter(const NNDatasetAdapterConfig *config,
                                                  NNDataset *dataset)
{
    nn_dataset_init(dataset, NN_DATASET_PROGRAM_FEATURE_COUNT);
    static const size_t targets[] = {0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    size_t max_groups = config->max_groups == 0u ? sizeof(targets) / sizeof(targets[0]) : config->max_groups;
    size_t train_count = (max_groups * 3u) / 5u;
    size_t valid_count = max_groups / 5u;
    size_t op_count = sizeof(DATASET_PROGRAM_OPS) / sizeof(DATASET_PROGRAM_OPS[0]);
    for (size_t i = 0; i < max_groups; ++i) {
        size_t target = targets[i % (sizeof(targets) / sizeof(targets[0]))];
        NNDatasetSplit split = split_for_index(i, train_count, valid_count);
        size_t begin = dataset->row_count;
        for (size_t op = 0; op < op_count; ++op) {
            nn_real features[NN_DATASET_PROGRAM_FEATURE_COUNT];
            dataset_program_features(op, target, features);
            int target_row = op == target;
            if (!nn_dataset_append_candidate(dataset,
                                             split,
                                             (uint64_t)i,
                                             dataset_mix64(5000u + i),
                                             (uint32_t)op,
                                             target_row,
                                             features)) {
                return 0;
            }
        }
        if (!nn_dataset_finalize_group(dataset, begin, dataset->row_count)) {
            return 0;
        }
    }
    return 1;
}

static void dataset_signal_controls(size_t action, double *turn, double *thrust)
{
    static const double turns[NN_DATASET_SIGNAL_ACTION_COUNT] = {0,-1,1,0,-1,1,0,-1,1};
    static const double thrusts[NN_DATASET_SIGNAL_ACTION_COUNT] = {0,0,0,1,1,1,-1,-1,-1};
    *turn = turns[action];
    *thrust = thrusts[action];
}

static int generate_signal_wasd_adapter(const NNDatasetAdapterConfig *config, NNDataset *dataset)
{
    nn_dataset_init(dataset, NN_DATASET_SIGNAL_FEATURE_COUNT);
    size_t max_groups = config->max_groups == 0u ? 96u : config->max_groups;
    size_t train_count = (max_groups * 3u) / 5u;
    size_t valid_count = max_groups / 5u;
    uint32_t rng = config->seed == 0u ? UINT32_C(20260524) : config->seed;
    for (size_t i = 0; i < max_groups; ++i) {
        double heading_error = 2.0 * dataset_rng_unit(&rng) - 1.0;
        double speed = dataset_rng_unit(&rng);
        double clearance = dataset_rng_unit(&rng);
        double target_dist = dataset_rng_unit(&rng);
        double best_utility = -1e9;
        double utilities[NN_DATASET_SIGNAL_ACTION_COUNT];
        for (size_t action = 0; action < NN_DATASET_SIGNAL_ACTION_COUNT; ++action) {
            double turn = 0.0;
            double thrust = 0.0;
            dataset_signal_controls(action, &turn, &thrust);
            double align = 1.0 - (heading_error - 0.35 * turn) * (heading_error - 0.35 * turn);
            double progress = thrust > 0.0 ? align * (1.0 - 0.25 * speed) : -0.1 * target_dist;
            double safety = clearance < 0.25 && thrust > 0.0 ? -1.5 : 0.25 * clearance;
            utilities[action] = progress + safety - 0.05 * (turn < 0.0 ? -turn : turn);
            if (utilities[action] > best_utility) {
                best_utility = utilities[action];
            }
        }
        NNDatasetSplit split = split_for_index(i, train_count, valid_count);
        size_t begin = dataset->row_count;
        for (size_t action = 0; action < NN_DATASET_SIGNAL_ACTION_COUNT; ++action) {
            double turn = 0.0;
            double thrust = 0.0;
            dataset_signal_controls(action, &turn, &thrust);
            nn_real features[NN_DATASET_SIGNAL_FEATURE_COUNT];
            memset(features, 0, sizeof(features));
            features[0] = (nn_real)1;
            features[1] = (nn_real)heading_error;
            features[2] = (nn_real)speed;
            features[3] = (nn_real)clearance;
            features[4] = (nn_real)target_dist;
            features[5] = (nn_real)turn;
            features[6] = (nn_real)thrust;
            features[7] = turn < 0.0 ? (nn_real)1 : (nn_real)0;
            features[8] = turn > 0.0 ? (nn_real)1 : (nn_real)0;
            features[9] = thrust > 0.0 ? (nn_real)1 : (nn_real)0;
            features[10] = thrust < 0.0 ? (nn_real)1 : (nn_real)0;
            features[11] = action == 0u ? (nn_real)1 : (nn_real)0;
            features[12] = (nn_real)(heading_error * turn);
            features[13] = (nn_real)(speed * thrust);
            features[14] = (nn_real)(clearance * thrust);
            features[15] = (nn_real)(target_dist * thrust);
            features[16] = clearance < 0.25 ? (nn_real)1 : (nn_real)0;
            features[17] = speed > 0.75 ? (nn_real)1 : (nn_real)0;
            features[18] = (nn_real)((double)action / 8.0);
            features[19] = (nn_real)utilities[action];
            if (!nn_dataset_append_candidate(dataset,
                                             split,
                                             (uint64_t)i,
                                             dataset_mix64(9000u + i),
                                             (uint32_t)action,
                                             utilities[action] + 1e-9 >= best_utility,
                                             features)) {
                return 0;
            }
        }
        if (!nn_dataset_finalize_group(dataset, begin, dataset->row_count)) {
            return 0;
        }
    }
    return 1;
}


extern int gen_go_adapter(const NNDatasetAdapterConfig *cfg, NNDataset *ds);
static const NNDatasetExperiment NN_DATASET_EXPERIMENTS[] = {
    {
        "threshold",
        "synthetic threshold action verifier; choose action 1 when x >= 0.5",
        "synthetic",
        "threshold_v1",
        "threshold_x_ge_0_5",
        4,
        NN_DATASET_LABEL_SINGLE_TARGET,
        generate_threshold_adapter,
        NN_DATASET_CAP_DATASET | NN_DATASET_CAP_EVIDENCE,
        "none",
        "none",
        "crlp.engine_evidence.v1",
    },
    {
        "threshold_multi",
        "synthetic multi-target threshold verifier; any action value greater than x is valid",
        "synthetic",
        "threshold_multi_v1",
        "threshold_action_ge_x",
        4,
        NN_DATASET_LABEL_MULTI_TARGET,
        generate_threshold_multi_adapter,
        NN_DATASET_CAP_DATASET | NN_DATASET_CAP_EVIDENCE,
        "none",
        "none",
        "crlp.engine_evidence.v1",
    },
    {
        "bitwise_truth_table",
        "bitwise truth-table action verifier; choose the candidate mask matching the observed table",
        "bitwise",
        "bitwise_tt_v1",
        "exact_truth_mask_match",
        NN_DATASET_BITWISE_FEATURE_COUNT,
        NN_DATASET_LABEL_SINGLE_TARGET,
        generate_bitwise_truth_table_adapter,
        NN_DATASET_CAP_DATASET | NN_DATASET_CAP_REPLAY | NN_DATASET_CAP_EVIDENCE,
        "none",
        "truth_table_replay_v1",
        "crlp.engine_evidence.v1",
    },
    {
        "bitwise_macro_step",
        "bitwise recursive macro-step verifier; choose the oracle tool application for a target truth mask",
        "bitwise",
        "bitwise_tool_v1",
        "chain_completion_oracle_step",
        NN_BITWISE_TOOL_FEATURE_COUNT,
        NN_DATASET_LABEL_MULTI_TARGET,
        generate_bitwise_macro_step_adapter,
        NN_DATASET_CAP_DATASET | NN_DATASET_CAP_RULE_SEARCH |
            NN_DATASET_CAP_REPLAY | NN_DATASET_CAP_EVIDENCE,
        "bitwise_macro_rule_search_v1",
        "bitwise_macro_replay_v1",
        "crlp.engine_evidence.v1",
    },
    {
        "haiku_adjacency",
        "haiku next-word verifier using corpus-derived line adjacency and syllable viability",
        "haiku",
        "haiku_adj_v1",
        "corpus_adjacency_syllable_viability",
        NN_DATASET_HAIKU_FEATURE_COUNT,
        NN_DATASET_LABEL_MULTI_TARGET,
        generate_haiku_adjacency_adapter,
        NN_DATASET_CAP_DATASET | NN_DATASET_CAP_RULE_SEARCH |
            NN_DATASET_CAP_REPLAY | NN_DATASET_CAP_EVIDENCE,
        "haiku_corpus_rule_search_v1",
        "haiku_corpus_replay_v1",
        "crlp.engine_evidence.v1",
    },
    {
        "grid_path",
        "grid shortest-path action verifier over generated obstacle maps",
        "grid",
        "grid_path_v3_dataset",
        "bfs_shortest_successor",
        NN_DATASET_GRID_FEATURE_COUNT,
        NN_DATASET_LABEL_MULTI_TARGET,
        generate_grid_path_adapter,
        NN_DATASET_CAP_DATASET | NN_DATASET_CAP_RULE_SEARCH |
            NN_DATASET_CAP_REPLAY | NN_DATASET_CAP_EVIDENCE,
        "grid_path_rule_search_v1",
        "grid_path_replay_v1",
        "crlp.engine_evidence.v1",
    },
    {
        "delayed_claim",
        "delayed-claim deterministic replay verifier over four claim actions",
        "delayed_claim",
        "claim_state_action_v1_dataset",
        "hidden_utility_argmax",
        NN_DATASET_CLAIM_FEATURE_COUNT,
        NN_DATASET_LABEL_SINGLE_TARGET,
        generate_delayed_claim_adapter,
        NN_DATASET_CAP_DATASET | NN_DATASET_CAP_RULE_SEARCH |
            NN_DATASET_CAP_REPLAY | NN_DATASET_CAP_EVIDENCE,
        "delayed_claim_rule_search_v1",
        "delayed_claim_trace_replay_v1",
        "crlp.engine_evidence.v1",
    },
    {
        "prime_gap_candidate",
        "next-prime gap candidate verifier using exact primality checks",
        "prime_gap",
        "prime_gap_candidate_v1",
        "first_prime_gap",
        NN_DATASET_PRIME_FEATURE_COUNT,
        NN_DATASET_LABEL_SINGLE_TARGET,
        generate_prime_gap_candidate_adapter,
        NN_DATASET_CAP_DATASET | NN_DATASET_CAP_RULE_SEARCH |
            NN_DATASET_CAP_REPLAY | NN_DATASET_CAP_EVIDENCE,
        "prime_gap_rule_search_v2",
        "prime_gap_trace_replay_v2",
        "crlp.engine_evidence.v1",
    },
    {
        "arc_cell_color",
        "ARC cell-color candidate verifier for the 0520fde7-style split-grid task",
        "arc",
        "arc_cell_color_v1",
        "expected_output_color",
        NN_DATASET_ARC_FEATURE_COUNT,
        NN_DATASET_LABEL_SINGLE_TARGET,
        generate_arc_cell_color_adapter,
        NN_DATASET_CAP_DATASET | NN_DATASET_CAP_RULE_SEARCH |
            NN_DATASET_CAP_REPLAY | NN_DATASET_CAP_EVIDENCE,
        "arc_hypothesis_rule_search_v1",
        "arc_cell_color_replay_v1",
        "crlp.engine_evidence.v1",
    },
    {
        "program_synthesis_ops",
        "program-synthesis candidate abstraction verifier over fixture DSL ops",
        "program_synthesis",
        "program_synth_ops_v1",
        "dsl_task_target_op",
        NN_DATASET_PROGRAM_FEATURE_COUNT,
        NN_DATASET_LABEL_SINGLE_TARGET,
        generate_program_synthesis_ops_adapter,
        NN_DATASET_CAP_DATASET | NN_DATASET_CAP_RULE_SEARCH |
            NN_DATASET_CAP_REPLAY | NN_DATASET_CAP_EVIDENCE,
        "program_synthesis_rule_search_v1",
        "program_synthesis_replay_v1",
        "crlp.engine_evidence.v1",
    },
    {
        "signal_wasd",
        "Signal-style WASD candidate verifier over synthetic counterfactual utility states",
        "signal",
        "signal_wasd_v1",
        "counterfactual_utility_argmax",
        NN_DATASET_SIGNAL_FEATURE_COUNT,
        NN_DATASET_LABEL_MULTI_TARGET,
        generate_signal_wasd_adapter,
        NN_DATASET_CAP_DATASET | NN_DATASET_CAP_RULE_SEARCH |
            NN_DATASET_CAP_REPLAY | NN_DATASET_CAP_EVIDENCE,
        "signal_control_rule_search_v1",
        "signal_counterfactual_replay_v1",
        "crlp.engine_evidence.v1",
    },
    {
        "proof_resolution",
        "propositional resolution prover using exact resolution-check verifier",
        "proof",
        "proof_res_v1",
        "resolution_soundness",
        NN_DATASET_PROOF_FEATURE_COUNT,
        NN_DATASET_LABEL_SINGLE_TARGET,
        generate_proof_resolution_adapter,
        NN_DATASET_CAP_DATASET | NN_DATASET_CAP_REPLAY | NN_DATASET_CAP_EVIDENCE,
        "proof_resolution_rule_search_v1",
        "proof_resolution_replay_v1",
        "crlp.engine_evidence.v1",
    },

    {
        "music_sequence",
        "toy music sequence verifier using corpus-derived note adjacency and phrase structure",
        "music",
        "music_seq_v1",
        "corpus_adjacency_phrase_viability",
        NN_DATASET_MUSIC_FEATURE_COUNT,
        NN_DATASET_LABEL_SINGLE_TARGET,
        generate_music_sequence_adapter,
        NN_DATASET_CAP_DATASET | NN_DATASET_CAP_REPLAY | NN_DATASET_CAP_EVIDENCE,
        "music_sequence_rule_search_v1",
        "music_corpus_replay_v1",
        "crlp.engine_evidence.v1",
    },

    {
        "go_position",
        "9x9 Go position action scorer under deterministic replay verifier",
        "go",
        "go_position_v1",
        "deterministic_rollout_score_replay",
        96,
        NN_DATASET_LABEL_MULTI_TARGET,
        gen_go_adapter,
        NN_DATASET_CAP_DATASET | NN_DATASET_CAP_RULE_SEARCH |
            NN_DATASET_CAP_REPLAY | NN_DATASET_CAP_EVIDENCE,
        "go_position_rule_search_v1",
        "go_position_replay_v1",
        "crlp.engine_evidence.v1",
    },

};

const NNDatasetExperiment *nn_dataset_find_experiment(const char *name)
{
    if (name == NULL) {
        return NULL;
    }
    for (size_t i = 0; i < nn_dataset_experiment_count(); ++i) {
        if (strcmp(NN_DATASET_EXPERIMENTS[i].name, name) == 0) {
            return &NN_DATASET_EXPERIMENTS[i];
        }
    }
    return NULL;
}

size_t nn_dataset_experiment_count(void)
{
    return sizeof(NN_DATASET_EXPERIMENTS) / sizeof(NN_DATASET_EXPERIMENTS[0]);
}

const NNDatasetExperiment *nn_dataset_experiment_at(size_t index)
{
    if (index >= nn_dataset_experiment_count()) {
        return NULL;
    }
    return &NN_DATASET_EXPERIMENTS[index];
}

int nn_dataset_experiment_has_capability(const NNDatasetExperiment *experiment,
                                         unsigned int capability)
{
    return experiment != NULL && (experiment->capabilities & capability) == capability;
}

const NNDatasetAdapter *nn_dataset_find_adapter(const char *name)
{
    return nn_dataset_find_experiment(name);
}

size_t nn_dataset_adapter_count(void)
{
    return nn_dataset_experiment_count();
}

const NNDatasetAdapter *nn_dataset_adapter_at(size_t index)
{
    return nn_dataset_experiment_at(index);
}
