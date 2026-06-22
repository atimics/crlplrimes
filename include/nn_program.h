#ifndef NN_PROGRAM_H
#define NN_PROGRAM_H

#include "nn_core.h"
#include "nn_symbolic.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#define NN_PROGRAM_MAX_NAME 64u
#define NN_PROGRAM_MAX_DOMAIN 32u
#define NN_PROGRAM_MAX_PARAMS 16u
#define NN_PROGRAM_MAX_FEATURES 96u
#define NN_PROGRAM_MAX_CHILDREN 4u
#define NN_PROGRAM_MAX_CANONICAL_ID 192u
#define NN_PROGRAM_FEATURE_ENCODER_VERSION 1u
#define NN_PROGRAM_FEATURE_COUNT 32u

typedef enum {
    NN_PROGRAM_VALUE_ANY = 0,
    NN_PROGRAM_VALUE_GRID = 1,
    NN_PROGRAM_VALUE_OBJECT_SET = 2,
    NN_PROGRAM_VALUE_OBJECT = 3,
    NN_PROGRAM_VALUE_MASK = 4,
    NN_PROGRAM_VALUE_SCALAR = 5,
    NN_PROGRAM_VALUE_BOOLEAN = 6,
    NN_PROGRAM_VALUE_SEQUENCE = 7,
    NN_PROGRAM_VALUE_SIM_STATE = 8,
    NN_PROGRAM_VALUE_TRAJECTORY = 9,
    NN_PROGRAM_VALUE_WAYPOINT = 10,
    NN_PROGRAM_VALUE_RESOURCE_TRANSFER = 11,
    NN_PROGRAM_VALUE_RECEIPT_CHAIN = 12,
    NN_PROGRAM_VALUE_HANDOFF_TICKET = 13,
    NN_PROGRAM_VALUE_AUTHORITY = 14,
    NN_PROGRAM_VALUE_ZONE = 15,
    NN_PROGRAM_VALUE_SHIP_SNAPSHOT = 16,
    NN_PROGRAM_VALUE_CUSTOM = 255,
} NNProgramValueType;

typedef enum {
    NN_PROGRAM_KIND_PRIMITIVE = 0,
    NN_PROGRAM_KIND_COMPOSITION = 1,
    NN_PROGRAM_KIND_SELECTOR = 2,
    NN_PROGRAM_KIND_PREDICATE = 3,
    NN_PROGRAM_KIND_TRANSFORM = 4,
    NN_PROGRAM_KIND_CUSTOM = 255,
} NNProgramKind;

typedef enum {
    NN_PROGRAM_PROPOSER_RANDOM = 0,
    NN_PROGRAM_PROPOSER_GRAMMAR_ORDER = 1,
    NN_PROGRAM_PROPOSER_STRUCTURAL_HEURISTIC = 2,
    NN_PROGRAM_PROPOSER_LEARNED = 3,
    NN_PROGRAM_PROPOSER_CLOSED_LOOP = 4,
    NN_PROGRAM_PROPOSER_TRACE_PRIOR = 5,
    NN_PROGRAM_PROPOSER_TRACE_TASK_PRIOR = 6,
    NN_PROGRAM_PROPOSER_TRACE_LINEAR = 7,
    NN_PROGRAM_PROPOSER_TRACE_MLP = 8,
    NN_PROGRAM_PROPOSER_ORACLE = 9,
} NNProgramProposerStrategy;

typedef struct {
    char name[NN_PROGRAM_MAX_NAME];
    double value;
    int64_t integer;
} NNProgramParam;

typedef struct {
    char domain[NN_PROGRAM_MAX_DOMAIN];
    char name[NN_PROGRAM_MAX_NAME];
    NNProgramKind kind;
    NNProgramValueType input_type;
    NNProgramValueType output_type;
    uint32_t template_id;
    NNProgramParam params[NN_PROGRAM_MAX_PARAMS];
    size_t param_count;
    nn_real structural_features[NN_PROGRAM_MAX_FEATURES];
    size_t structural_feature_count;
    char child_ids[NN_PROGRAM_MAX_CHILDREN][NN_PROGRAM_MAX_CANONICAL_ID];
    size_t child_count;
    char canonical_id[NN_PROGRAM_MAX_CANONICAL_ID];
    uint64_t canonical_hash;
    NNCertificate certificate;
    int accepted;
    double utility;
    size_t train_support;
    size_t validation_support;
    size_t test_support;
    size_t train_mismatches;
    size_t validation_mismatches;
    size_t test_mismatches;
} NNProgram;

typedef struct {
    uint64_t *bits;
    size_t program_count;
    size_t event_count;
    size_t word_count;
} NNProgramCoverage;

typedef struct {
    const char *domain;
    const char *task;
    NNProgramProposerStrategy strategy;
    size_t budget;
    size_t proposed_programs;
    size_t certified_programs;
    size_t accepted_programs;
    size_t event_count;
    size_t covered_events;
    size_t false_rejections;
    double utility;
    const char *program_set_hash;
} NNProgramProposerSummary;

const char *nn_program_value_type_name(NNProgramValueType type);
const char *nn_program_kind_name(NNProgramKind kind);
const char *nn_program_proposer_strategy_name(NNProgramProposerStrategy strategy);

void nn_program_clear(NNProgram *program);
void nn_program_init(NNProgram *program,
                     const char *domain,
                     const char *name,
                     NNProgramKind kind,
                     NNProgramValueType input_type,
                     NNProgramValueType output_type,
                     uint32_t template_id);
int nn_program_add_param(NNProgram *program,
                         const char *name,
                         double value,
                         int64_t integer);
int nn_program_get_param(const NNProgram *program, const char *name, double *value_out);
int nn_program_get_param_i64(const NNProgram *program, const char *name, int64_t *value_out);
int nn_program_add_feature(NNProgram *program, nn_real value);
const NNFeatureSchema *nn_program_feature_schema(size_t *feature_count_out);
int nn_program_encode_features(const NNProgram *program,
                               nn_real *features,
                               size_t capacity,
                               size_t *feature_count_out);
int nn_program_is_composable(const NNProgram *first, const NNProgram *second);
int nn_program_compose(NNProgram *out,
                       const NNProgram *first,
                       const NNProgram *second,
                       const char *name);
uint64_t nn_program_refresh_canonical_id(NNProgram *program);

int nn_program_coverage_init(NNProgramCoverage *coverage,
                             size_t program_count,
                             size_t event_count);
void nn_program_coverage_free(NNProgramCoverage *coverage);
void nn_program_coverage_clear(NNProgramCoverage *coverage);
int nn_program_coverage_set(NNProgramCoverage *coverage,
                            size_t program_index,
                            size_t event_index);
int nn_program_coverage_get(const NNProgramCoverage *coverage,
                            size_t program_index,
                            size_t event_index);
size_t nn_program_coverage_count_covered(const uint64_t *covered, size_t word_count);
size_t nn_program_coverage_gain(const NNProgramCoverage *coverage,
                                size_t program_index,
                                const uint64_t *covered);
int nn_program_coverage_add_program_to_residual(const NNProgramCoverage *coverage,
                                                size_t program_index,
                                                uint64_t *covered);

int nn_program_write_proposer_summary_json(FILE *fp,
                                           const NNProgramProposerSummary *summary);

#endif
