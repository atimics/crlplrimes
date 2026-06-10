#include "nn_program.h"
#include "nn_text.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static const NNFeatureSchema NN_PROGRAM_FEATURE_SCHEMA[NN_PROGRAM_FEATURE_COUNT] = {
    {"program_bias", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"program_kind_norm", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"program_input_type_norm", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"program_output_type_norm", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"program_template_id_norm", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"program_param_count_norm", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"program_child_count_norm", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"program_structural_feature_count_norm", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"program_has_children", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"program_is_composition", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"program_input_output_same_type", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"program_input_is_grid", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"program_output_is_grid", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"program_output_is_object_set", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"program_output_is_object", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"program_output_is_mask", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"program_param_mean_abs", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"program_param_max_abs", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"program_param_mean_signed", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"program_param_integer_mean_abs", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"program_payload_00", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"program_payload_01", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"program_payload_02", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"program_payload_03", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"program_payload_04", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"program_payload_05", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"program_payload_06", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"program_payload_07", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"program_payload_08", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"program_payload_09", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"program_payload_10", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"program_payload_11", NN_FEATURE_STRUCTURAL_PRIOR, 1},
};

static nn_real nn_program_clamped_unit(double value, double scale)
{
    if (!isfinite(value) || scale <= 0.0) {
        return (nn_real)0;
    }
    double scaled = value / scale;
    if (scaled > 1.0) {
        scaled = 1.0;
    } else if (scaled < -1.0) {
        scaled = -1.0;
    }
    return (nn_real)scaled;
}

static void nn_program_copy_text(char *dst, size_t dst_size, const char *src)
{
    if (dst == NULL || dst_size == 0) {
        return;
    }
    if (src == NULL) {
        dst[0] = '\0';
        return;
    }
    snprintf(dst, dst_size, "%s", src);
}

static size_t nn_program_popcount_u64(uint64_t value)
{
#if defined(__GNUC__) || defined(__clang__)
    return (size_t)__builtin_popcountll((unsigned long long)value);
#else
    size_t count = 0;
    while (value != 0) {
        value &= value - 1u;
        ++count;
    }
    return count;
#endif
}

static uint64_t nn_program_hash_bytes(uint64_t hash, const void *data, size_t size)
{
    const unsigned char *bytes = (const unsigned char *)data;
    for (size_t i = 0; i < size; ++i) {
        hash ^= (uint64_t)bytes[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static uint64_t nn_program_hash_text(uint64_t hash, const char *text)
{
    if (text == NULL) {
        return nn_program_hash_bytes(hash, "", 1);
    }
    return nn_program_hash_bytes(hash, text, strlen(text) + 1u);
}

static uint64_t nn_program_hash_u64(uint64_t hash, uint64_t value)
{
    return nn_program_hash_bytes(hash, &value, sizeof(value));
}

static uint64_t nn_program_hash_double(uint64_t hash, double value)
{
    return nn_program_hash_bytes(hash, &value, sizeof(value));
}

static int nn_program_type_matches(NNProgramValueType produced,
                                   NNProgramValueType consumed)
{
    return produced == consumed ||
           produced == NN_PROGRAM_VALUE_ANY ||
           consumed == NN_PROGRAM_VALUE_ANY;
}

const char *nn_program_value_type_name(NNProgramValueType type)
{
    switch (type) {
    case NN_PROGRAM_VALUE_ANY:
        return "any";
    case NN_PROGRAM_VALUE_GRID:
        return "grid";
    case NN_PROGRAM_VALUE_OBJECT_SET:
        return "object_set";
    case NN_PROGRAM_VALUE_OBJECT:
        return "object";
    case NN_PROGRAM_VALUE_MASK:
        return "mask";
    case NN_PROGRAM_VALUE_SCALAR:
        return "scalar";
    case NN_PROGRAM_VALUE_BOOLEAN:
        return "boolean";
    case NN_PROGRAM_VALUE_SEQUENCE:
        return "sequence";
    case NN_PROGRAM_VALUE_SIM_STATE:
        return "sim_state";
    case NN_PROGRAM_VALUE_TRAJECTORY:
        return "trajectory";
    case NN_PROGRAM_VALUE_WAYPOINT:
        return "waypoint";
    case NN_PROGRAM_VALUE_RESOURCE_TRANSFER:
        return "resource_transfer";
    case NN_PROGRAM_VALUE_RECEIPT_CHAIN:
        return "receipt_chain";
    case NN_PROGRAM_VALUE_HANDOFF_TICKET:
        return "handoff_ticket";
    case NN_PROGRAM_VALUE_AUTHORITY:
        return "authority";
    case NN_PROGRAM_VALUE_ZONE:
        return "zone";
    case NN_PROGRAM_VALUE_SHIP_SNAPSHOT:
        return "ship_snapshot";
    case NN_PROGRAM_VALUE_CUSTOM:
        return "custom";
    }
    return "unknown";
}

const char *nn_program_kind_name(NNProgramKind kind)
{
    switch (kind) {
    case NN_PROGRAM_KIND_PRIMITIVE:
        return "primitive";
    case NN_PROGRAM_KIND_COMPOSITION:
        return "composition";
    case NN_PROGRAM_KIND_SELECTOR:
        return "selector";
    case NN_PROGRAM_KIND_PREDICATE:
        return "predicate";
    case NN_PROGRAM_KIND_TRANSFORM:
        return "transform";
    case NN_PROGRAM_KIND_CUSTOM:
        return "custom";
    }
    return "unknown";
}

const char *nn_program_proposer_strategy_name(NNProgramProposerStrategy strategy)
{
    switch (strategy) {
    case NN_PROGRAM_PROPOSER_RANDOM:
        return "random";
    case NN_PROGRAM_PROPOSER_GRAMMAR_ORDER:
        return "grammar_order";
    case NN_PROGRAM_PROPOSER_STRUCTURAL_HEURISTIC:
        return "structural_heuristic";
    case NN_PROGRAM_PROPOSER_LEARNED:
        return "learned";
    case NN_PROGRAM_PROPOSER_CLOSED_LOOP:
        return "closed_loop";
    case NN_PROGRAM_PROPOSER_TRACE_PRIOR:
        return "trace_prior";
    case NN_PROGRAM_PROPOSER_TRACE_TASK_PRIOR:
        return "trace_task_prior";
    case NN_PROGRAM_PROPOSER_TRACE_LINEAR:
        return "trace_linear";
    case NN_PROGRAM_PROPOSER_TRACE_MLP:
        return "trace_mlp";
    case NN_PROGRAM_PROPOSER_ORACLE:
        return "oracle";
    }
    return "unknown";
}

void nn_program_clear(NNProgram *program)
{
    if (program == NULL) {
        return;
    }
    memset(program, 0, sizeof(*program));
    nn_certificate_clear(&program->certificate);
}

uint64_t nn_program_refresh_canonical_id(NNProgram *program)
{
    if (program == NULL) {
        return 0;
    }
    uint64_t hash = UINT64_C(1469598103934665603);
    hash = nn_program_hash_text(hash, program->domain);
    hash = nn_program_hash_text(hash, program->name);
    hash = nn_program_hash_u64(hash, (uint64_t)program->kind);
    hash = nn_program_hash_u64(hash, (uint64_t)program->input_type);
    hash = nn_program_hash_u64(hash, (uint64_t)program->output_type);
    hash = nn_program_hash_u64(hash, (uint64_t)program->template_id);
    hash = nn_program_hash_u64(hash, (uint64_t)program->param_count);
    for (size_t i = 0; i < program->param_count; ++i) {
        hash = nn_program_hash_text(hash, program->params[i].name);
        hash = nn_program_hash_double(hash, program->params[i].value);
        hash = nn_program_hash_u64(hash, (uint64_t)program->params[i].integer);
    }
    hash = nn_program_hash_u64(hash, (uint64_t)program->child_count);
    for (size_t i = 0; i < program->child_count; ++i) {
        hash = nn_program_hash_text(hash, program->child_ids[i]);
    }
    program->canonical_hash = hash;

    int written = snprintf(program->canonical_id,
                           sizeof(program->canonical_id),
                           "%s:%s:%s:%s>%s:%u:%016llx",
                           program->domain,
                           program->name,
                           nn_program_kind_name(program->kind),
                           nn_program_value_type_name(program->input_type),
                           nn_program_value_type_name(program->output_type),
                           (unsigned int)program->template_id,
                           (unsigned long long)program->canonical_hash);
    if (written < 0) {
        program->canonical_id[0] = '\0';
    }
    return program->canonical_hash;
}

void nn_program_init(NNProgram *program,
                     const char *domain,
                     const char *name,
                     NNProgramKind kind,
                     NNProgramValueType input_type,
                     NNProgramValueType output_type,
                     uint32_t template_id)
{
    if (program == NULL) {
        return;
    }
    nn_program_clear(program);
    nn_program_copy_text(program->domain, sizeof(program->domain), domain);
    nn_program_copy_text(program->name, sizeof(program->name), name);
    program->kind = kind;
    program->input_type = input_type;
    program->output_type = output_type;
    program->template_id = template_id;
    nn_program_refresh_canonical_id(program);
}

int nn_program_add_param(NNProgram *program,
                         const char *name,
                         double value,
                         int64_t integer)
{
    if (program == NULL || name == NULL ||
        program->param_count >= NN_PROGRAM_MAX_PARAMS) {
        return 0;
    }
    NNProgramParam *param = &program->params[program->param_count++];
    nn_program_copy_text(param->name, sizeof(param->name), name);
    param->value = value;
    param->integer = integer;
    nn_program_refresh_canonical_id(program);
    return 1;
}

int nn_program_get_param(const NNProgram *program, const char *name, double *value_out)
{
    if (program == NULL || name == NULL || value_out == NULL) {
        return 0;
    }
    for (size_t i = 0; i < program->param_count; ++i) {
        if (strcmp(program->params[i].name, name) == 0) {
            *value_out = program->params[i].value;
            return 1;
        }
    }
    return 0;
}

int nn_program_get_param_i64(const NNProgram *program, const char *name, int64_t *value_out)
{
    if (program == NULL || name == NULL || value_out == NULL) {
        return 0;
    }
    for (size_t i = 0; i < program->param_count; ++i) {
        if (strcmp(program->params[i].name, name) == 0) {
            *value_out = program->params[i].integer;
            return 1;
        }
    }
    return 0;
}

int nn_program_add_feature(NNProgram *program, nn_real value)
{
    if (program == NULL ||
        program->structural_feature_count >= NN_PROGRAM_MAX_FEATURES) {
        return 0;
    }
    program->structural_features[program->structural_feature_count++] = value;
    return 1;
}

const NNFeatureSchema *nn_program_feature_schema(size_t *feature_count_out)
{
    if (feature_count_out != NULL) {
        *feature_count_out = NN_PROGRAM_FEATURE_COUNT;
    }
    return NN_PROGRAM_FEATURE_SCHEMA;
}

int nn_program_encode_features(const NNProgram *program,
                               nn_real *features,
                               size_t capacity,
                               size_t *feature_count_out)
{
    if (program == NULL || features == NULL || capacity < NN_PROGRAM_FEATURE_COUNT) {
        return 0;
    }
    memset(features, 0, NN_PROGRAM_FEATURE_COUNT * sizeof(*features));

    double param_abs_sum = 0.0;
    double param_signed_sum = 0.0;
    double param_max_abs = 0.0;
    double integer_abs_sum = 0.0;
    for (size_t i = 0; i < program->param_count; ++i) {
        double abs_value = fabs(program->params[i].value);
        param_abs_sum += abs_value;
        param_signed_sum += program->params[i].value;
        if (abs_value > param_max_abs) {
            param_max_abs = abs_value;
        }
        integer_abs_sum += fabs((double)program->params[i].integer);
    }
    double param_count = program->param_count == 0 ? 1.0 : (double)program->param_count;

    features[0] = (nn_real)1;
    features[1] = nn_program_clamped_unit((double)program->kind, 255.0);
    features[2] = nn_program_clamped_unit((double)program->input_type, 255.0);
    features[3] = nn_program_clamped_unit((double)program->output_type, 255.0);
    features[4] = nn_program_clamped_unit((double)program->template_id, 1024.0);
    features[5] = nn_program_clamped_unit((double)program->param_count,
                                          (double)NN_PROGRAM_MAX_PARAMS);
    features[6] = nn_program_clamped_unit((double)program->child_count,
                                          (double)NN_PROGRAM_MAX_CHILDREN);
    features[7] = nn_program_clamped_unit((double)program->structural_feature_count,
                                          (double)NN_PROGRAM_MAX_FEATURES);
    features[8] = program->child_count != 0 ? (nn_real)1 : (nn_real)0;
    features[9] = program->kind == NN_PROGRAM_KIND_COMPOSITION ? (nn_real)1 : (nn_real)0;
    features[10] = program->input_type == program->output_type ? (nn_real)1 : (nn_real)0;
    features[11] = program->input_type == NN_PROGRAM_VALUE_GRID ? (nn_real)1 : (nn_real)0;
    features[12] = program->output_type == NN_PROGRAM_VALUE_GRID ? (nn_real)1 : (nn_real)0;
    features[13] = program->output_type == NN_PROGRAM_VALUE_OBJECT_SET ? (nn_real)1 : (nn_real)0;
    features[14] = program->output_type == NN_PROGRAM_VALUE_OBJECT ? (nn_real)1 : (nn_real)0;
    features[15] = program->output_type == NN_PROGRAM_VALUE_MASK ? (nn_real)1 : (nn_real)0;
    features[16] = nn_program_clamped_unit(param_abs_sum / param_count, 16.0);
    features[17] = nn_program_clamped_unit(param_max_abs, 64.0);
    features[18] = nn_program_clamped_unit(param_signed_sum / param_count, 16.0);
    features[19] = nn_program_clamped_unit(integer_abs_sum / param_count, 64.0);

    size_t payload_count = NN_PROGRAM_FEATURE_COUNT - 20u;
    if (payload_count > program->structural_feature_count) {
        payload_count = program->structural_feature_count;
    }
    for (size_t i = 0; i < payload_count; ++i) {
        features[20u + i] = program->structural_features[i];
    }
    if (feature_count_out != NULL) {
        *feature_count_out = NN_PROGRAM_FEATURE_COUNT;
    }
    return 1;
}

int nn_program_is_composable(const NNProgram *first, const NNProgram *second)
{
    if (first == NULL || second == NULL) {
        return 0;
    }
    if (strcmp(first->domain, second->domain) != 0) {
        return 0;
    }
    return nn_program_type_matches(first->output_type, second->input_type);
}

int nn_program_compose(NNProgram *out,
                       const NNProgram *first,
                       const NNProgram *second,
                       const char *name)
{
    if (out == NULL || first == NULL || second == NULL ||
        !nn_program_is_composable(first, second)) {
        return 0;
    }
    nn_program_init(out,
                    first->domain,
                    name != NULL ? name : "composition",
                    NN_PROGRAM_KIND_COMPOSITION,
                    first->input_type,
                    second->output_type,
                    0);
    out->child_count = 2;
    nn_program_copy_text(out->child_ids[0],
                         sizeof(out->child_ids[0]),
                         first->canonical_id);
    nn_program_copy_text(out->child_ids[1],
                         sizeof(out->child_ids[1]),
                         second->canonical_id);
    nn_program_refresh_canonical_id(out);
    return 1;
}

int nn_program_coverage_init(NNProgramCoverage *coverage,
                             size_t program_count,
                             size_t event_count)
{
    if (coverage == NULL) {
        return 0;
    }
    memset(coverage, 0, sizeof(*coverage));
    coverage->program_count = program_count;
    coverage->event_count = event_count;
    coverage->word_count = (event_count + 63u) / 64u;
    if (program_count == 0 || event_count == 0 || coverage->word_count == 0) {
        return 1;
    }
    if (coverage->word_count > SIZE_MAX / program_count ||
        program_count * coverage->word_count > SIZE_MAX / sizeof(*coverage->bits)) {
        memset(coverage, 0, sizeof(*coverage));
        return 0;
    }
    coverage->bits = calloc(program_count * coverage->word_count, sizeof(*coverage->bits));
    if (coverage->bits == NULL) {
        memset(coverage, 0, sizeof(*coverage));
        return 0;
    }
    return 1;
}

void nn_program_coverage_free(NNProgramCoverage *coverage)
{
    if (coverage == NULL) {
        return;
    }
    free(coverage->bits);
    memset(coverage, 0, sizeof(*coverage));
}

void nn_program_coverage_clear(NNProgramCoverage *coverage)
{
    if (coverage == NULL || coverage->bits == NULL) {
        return;
    }
    memset(coverage->bits,
           0,
           coverage->program_count * coverage->word_count * sizeof(*coverage->bits));
}

int nn_program_coverage_set(NNProgramCoverage *coverage,
                            size_t program_index,
                            size_t event_index)
{
    if (coverage == NULL || coverage->bits == NULL ||
        program_index >= coverage->program_count || event_index >= coverage->event_count) {
        return 0;
    }
    size_t word = event_index / 64u;
    uint64_t mask = UINT64_C(1) << (event_index % 64u);
    coverage->bits[program_index * coverage->word_count + word] |= mask;
    return 1;
}

int nn_program_coverage_get(const NNProgramCoverage *coverage,
                            size_t program_index,
                            size_t event_index)
{
    if (coverage == NULL || coverage->bits == NULL ||
        program_index >= coverage->program_count || event_index >= coverage->event_count) {
        return 0;
    }
    size_t word = event_index / 64u;
    uint64_t mask = UINT64_C(1) << (event_index % 64u);
    return (coverage->bits[program_index * coverage->word_count + word] & mask) != 0;
}

size_t nn_program_coverage_count_covered(const uint64_t *covered, size_t word_count)
{
    if (covered == NULL) {
        return 0;
    }
    size_t count = 0;
    for (size_t word = 0; word < word_count; ++word) {
        count += nn_program_popcount_u64(covered[word]);
    }
    return count;
}

size_t nn_program_coverage_gain(const NNProgramCoverage *coverage,
                                size_t program_index,
                                const uint64_t *covered)
{
    if (coverage == NULL || coverage->bits == NULL || covered == NULL ||
        program_index >= coverage->program_count) {
        return 0;
    }
    const uint64_t *program_bits =
        &coverage->bits[program_index * coverage->word_count];
    size_t gain = 0;
    for (size_t word = 0; word < coverage->word_count; ++word) {
        gain += nn_program_popcount_u64(program_bits[word] & ~covered[word]);
    }
    return gain;
}

int nn_program_coverage_add_program_to_residual(const NNProgramCoverage *coverage,
                                                size_t program_index,
                                                uint64_t *covered)
{
    if (coverage == NULL || coverage->bits == NULL || covered == NULL ||
        program_index >= coverage->program_count) {
        return 0;
    }
    const uint64_t *program_bits =
        &coverage->bits[program_index * coverage->word_count];
    for (size_t word = 0; word < coverage->word_count; ++word) {
        covered[word] |= program_bits[word];
    }
    return 1;
}

static void nn_program_json_string(FILE *fp, const char *value)
{
    (void)nn_write_json_string(fp, value);
}

int nn_program_write_proposer_summary_json(FILE *fp,
                                           const NNProgramProposerSummary *summary)
{
    if (fp == NULL || summary == NULL) {
        return 0;
    }
    fprintf(fp, "{\"schema\":\"crlp.program_proposer.summary.v1\"");
    fprintf(fp, ",\"domain\":");
    nn_program_json_string(fp, summary->domain);
    fprintf(fp, ",\"task\":");
    nn_program_json_string(fp, summary->task);
    fprintf(fp, ",\"strategy\":");
    nn_program_json_string(fp, nn_program_proposer_strategy_name(summary->strategy));
    fprintf(fp,
            ",\"budget\":%zu,\"proposed_programs\":%zu,"
            "\"certified_programs\":%zu,\"accepted_programs\":%zu,"
            "\"event_count\":%zu,\"covered_events\":%zu,"
            "\"false_rejections\":%zu,\"utility\":%.12g",
            summary->budget,
            summary->proposed_programs,
            summary->certified_programs,
            summary->accepted_programs,
            summary->event_count,
            summary->covered_events,
            summary->false_rejections,
            summary->utility);
    fprintf(fp, ",\"program_set_hash\":");
    nn_program_json_string(fp, summary->program_set_hash);
    fprintf(fp, "}\n");
    return ferror(fp) == 0;
}
