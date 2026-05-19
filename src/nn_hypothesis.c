#include "nn_hypothesis.h"
#include "nn_text.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static const NNFeatureSchema NN_HYPOTHESIS_FEATURE_SCHEMA[NN_HYPOTHESIS_FEATURE_COUNT] = {
    {"hypothesis_bias", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"hypothesis_template_id_norm", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"hypothesis_stage_count_norm", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"hypothesis_slot_count_norm", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"hypothesis_bound_slot_fraction", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"hypothesis_input_is_grid", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"hypothesis_output_is_grid", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"hypothesis_output_is_object_set", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"hypothesis_output_is_object", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"hypothesis_has_selector_stage", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"hypothesis_has_transform_stage", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"hypothesis_color_slots_norm", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"hypothesis_offset_slots_norm", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"hypothesis_selector_slot_norm", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"hypothesis_relation_slot_norm", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"hypothesis_render_slot_norm", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"hypothesis_slot_value_mean_abs", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"hypothesis_slot_value_max_abs", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"hypothesis_slot_value_signed_mean", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"hypothesis_slot_integer_mean_abs", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"hypothesis_color_slot_mean_norm", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"hypothesis_offset_l1_norm", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"hypothesis_input_output_same_type", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"hypothesis_has_default_only_required_slot", NN_FEATURE_STRUCTURAL_PRIOR, 1},
};

static nn_real nn_hypothesis_clamped_unit(double value, double scale)
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

static void nn_hypothesis_copy_text(char *dst, size_t dst_size, const char *src)
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

static uint64_t nn_hypothesis_hash_bytes(uint64_t hash, const void *data, size_t size)
{
    const unsigned char *bytes = (const unsigned char *)data;
    for (size_t i = 0; i < size; ++i) {
        hash ^= (uint64_t)bytes[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static uint64_t nn_hypothesis_hash_text(uint64_t hash, const char *text)
{
    if (text == NULL) {
        return nn_hypothesis_hash_bytes(hash, "", 1);
    }
    return nn_hypothesis_hash_bytes(hash, text, strlen(text) + 1u);
}

static uint64_t nn_hypothesis_hash_u64(uint64_t hash, uint64_t value)
{
    return nn_hypothesis_hash_bytes(hash, &value, sizeof(value));
}

static uint64_t nn_hypothesis_hash_double(uint64_t hash, double value)
{
    return nn_hypothesis_hash_bytes(hash, &value, sizeof(value));
}

static void nn_hypothesis_json_string(FILE *fp, const char *value)
{
    (void)nn_write_json_string(fp, value);
}

const char *nn_hypothesis_slot_type_name(NNHypothesisSlotType type)
{
    switch (type) {
    case NN_HYPOTHESIS_SLOT_INTEGER:
        return "integer";
    case NN_HYPOTHESIS_SLOT_COLOR:
        return "color";
    case NN_HYPOTHESIS_SLOT_OFFSET:
        return "offset";
    case NN_HYPOTHESIS_SLOT_SELECTOR_KIND:
        return "selector_kind";
    case NN_HYPOTHESIS_SLOT_RELATION_KIND:
        return "relation_kind";
    case NN_HYPOTHESIS_SLOT_RENDER_MODE:
        return "render_mode";
    case NN_HYPOTHESIS_SLOT_REFERENCE_COLOR:
        return "reference_color";
    case NN_HYPOTHESIS_SLOT_SCALAR:
        return "scalar";
    case NN_HYPOTHESIS_SLOT_BOOLEAN:
        return "boolean";
    case NN_HYPOTHESIS_SLOT_CUSTOM:
        return "custom";
    }
    return "unknown";
}

int nn_hypothesis_template_is_valid(const NNHypothesisTemplate *template_ref)
{
    if (template_ref == NULL) {
        return 0;
    }
    if (template_ref->stage_count == 0 ||
        template_ref->stage_count > NN_HYPOTHESIS_MAX_STAGES) {
        return 0;
    }
    if (template_ref->slot_count > NN_HYPOTHESIS_MAX_SLOTS) {
        return 0;
    }
    if (template_ref->stages[0].input_type != template_ref->input_type &&
        template_ref->input_type != NN_PROGRAM_VALUE_ANY &&
        template_ref->stages[0].input_type != NN_PROGRAM_VALUE_ANY) {
        return 0;
    }
    NNProgramValueType last_output =
        template_ref->stages[template_ref->stage_count - 1u].output_type;
    if (last_output != template_ref->output_type &&
        template_ref->output_type != NN_PROGRAM_VALUE_ANY &&
        last_output != NN_PROGRAM_VALUE_ANY) {
        return 0;
    }
    for (size_t i = 1; i < template_ref->stage_count; ++i) {
        NNProgramValueType produced = template_ref->stages[i - 1u].output_type;
        NNProgramValueType consumed = template_ref->stages[i].input_type;
        if (produced != consumed && produced != NN_PROGRAM_VALUE_ANY &&
            consumed != NN_PROGRAM_VALUE_ANY) {
            return 0;
        }
    }
    for (size_t i = 0; i < template_ref->slot_count; ++i) {
        if (template_ref->slots[i].name[0] == '\0') {
            return 0;
        }
        for (size_t j = i + 1u; j < template_ref->slot_count; ++j) {
            if (strcmp(template_ref->slots[i].name,
                       template_ref->slots[j].name) == 0) {
                return 0;
            }
        }
    }
    return 1;
}

void nn_hypothesis_template_init(NNHypothesisTemplate *template_ref,
                                 const char *domain,
                                 const char *name,
                                 uint32_t template_id,
                                 NNProgramValueType input_type,
                                 NNProgramValueType output_type)
{
    if (template_ref == NULL) {
        return;
    }
    memset(template_ref, 0, sizeof(*template_ref));
    nn_hypothesis_copy_text(template_ref->domain,
                            sizeof(template_ref->domain),
                            domain);
    nn_hypothesis_copy_text(template_ref->name,
                            sizeof(template_ref->name),
                            name);
    template_ref->template_id = template_id;
    template_ref->input_type = input_type;
    template_ref->output_type = output_type;
}

int nn_hypothesis_template_add_stage(NNHypothesisTemplate *template_ref,
                                     const char *stage_name,
                                     NNProgramKind kind,
                                     NNProgramValueType input_type,
                                     NNProgramValueType output_type,
                                     uint32_t stage_template_id)
{
    if (template_ref == NULL || stage_name == NULL ||
        template_ref->stage_count >= NN_HYPOTHESIS_MAX_STAGES) {
        return 0;
    }
    NNHypothesisStage *stage = &template_ref->stages[template_ref->stage_count];
    nn_hypothesis_copy_text(stage->name, sizeof(stage->name), stage_name);
    stage->kind = kind;
    stage->input_type = input_type;
    stage->output_type = output_type;
    stage->template_id = stage_template_id;
    ++template_ref->stage_count;
    return 1;
}

int nn_hypothesis_template_add_slot(NNHypothesisTemplate *template_ref,
                                    const char *slot_name,
                                    NNHypothesisSlotType type,
                                    int required,
                                    double default_value,
                                    int64_t default_integer)
{
    if (template_ref == NULL || slot_name == NULL ||
        template_ref->slot_count >= NN_HYPOTHESIS_MAX_SLOTS) {
        return 0;
    }
    NNHypothesisSlotSpec *spec =
        &template_ref->slots[template_ref->slot_count];
    nn_hypothesis_copy_text(spec->name, sizeof(spec->name), slot_name);
    spec->type = type;
    spec->required = required ? 1 : 0;
    spec->default_value = default_value;
    spec->default_integer = default_integer;
    ++template_ref->slot_count;
    return 1;
}

void nn_hypothesis_clear(NNHypothesis *hypothesis)
{
    if (hypothesis == NULL) {
        return;
    }
    memset(hypothesis, 0, sizeof(*hypothesis));
}

int nn_hypothesis_init(NNHypothesis *hypothesis,
                       const NNHypothesisTemplate *template_ref)
{
    if (hypothesis == NULL || !nn_hypothesis_template_is_valid(template_ref)) {
        return 0;
    }
    nn_hypothesis_clear(hypothesis);
    hypothesis->template_ref = template_ref;
    hypothesis->slot_count = template_ref->slot_count;
    for (size_t i = 0; i < template_ref->slot_count; ++i) {
        const NNHypothesisSlotSpec *spec = &template_ref->slots[i];
        NNHypothesisSlotValue *slot = &hypothesis->slots[i];
        nn_hypothesis_copy_text(slot->name, sizeof(slot->name), spec->name);
        slot->type = spec->type;
        slot->value = spec->default_value;
        slot->integer = spec->default_integer;
        slot->bound = spec->required ? 0 : 1;
    }
    nn_hypothesis_refresh_canonical_id(hypothesis);
    return 1;
}

static NNHypothesisSlotValue *nn_hypothesis_find_slot(NNHypothesis *hypothesis,
                                                      const char *slot_name)
{
    if (hypothesis == NULL || slot_name == NULL) {
        return NULL;
    }
    for (size_t i = 0; i < hypothesis->slot_count; ++i) {
        if (strcmp(hypothesis->slots[i].name, slot_name) == 0) {
            return &hypothesis->slots[i];
        }
    }
    return NULL;
}

int nn_hypothesis_bind(NNHypothesis *hypothesis,
                       const char *slot_name,
                       double value,
                       int64_t integer)
{
    NNHypothesisSlotValue *slot = nn_hypothesis_find_slot(hypothesis, slot_name);
    if (slot == NULL) {
        return 0;
    }
    slot->value = value;
    slot->integer = integer;
    slot->bound = 1;
    nn_hypothesis_refresh_canonical_id(hypothesis);
    return 1;
}

int nn_hypothesis_get(const NNHypothesis *hypothesis,
                      const char *slot_name,
                      double *value_out,
                      int64_t *integer_out)
{
    if (hypothesis == NULL || slot_name == NULL) {
        return 0;
    }
    for (size_t i = 0; i < hypothesis->slot_count; ++i) {
        if (strcmp(hypothesis->slots[i].name, slot_name) == 0) {
            if (value_out != NULL) {
                *value_out = hypothesis->slots[i].value;
            }
            if (integer_out != NULL) {
                *integer_out = hypothesis->slots[i].integer;
            }
            return 1;
        }
    }
    return 0;
}

int nn_hypothesis_is_fully_bound(const NNHypothesis *hypothesis)
{
    if (hypothesis == NULL || hypothesis->template_ref == NULL) {
        return 0;
    }
    for (size_t i = 0; i < hypothesis->slot_count; ++i) {
        if (!hypothesis->slots[i].bound) {
            return 0;
        }
    }
    return 1;
}

uint64_t nn_hypothesis_refresh_canonical_id(NNHypothesis *hypothesis)
{
    if (hypothesis == NULL || hypothesis->template_ref == NULL) {
        return 0;
    }
    const NNHypothesisTemplate *template_ref = hypothesis->template_ref;
    uint64_t hash = UINT64_C(1469598103934665603);
    hash = nn_hypothesis_hash_text(hash, template_ref->domain);
    hash = nn_hypothesis_hash_text(hash, template_ref->name);
    hash = nn_hypothesis_hash_u64(hash, (uint64_t)template_ref->template_id);
    hash = nn_hypothesis_hash_u64(hash, (uint64_t)template_ref->input_type);
    hash = nn_hypothesis_hash_u64(hash, (uint64_t)template_ref->output_type);
    hash = nn_hypothesis_hash_u64(hash, (uint64_t)template_ref->stage_count);
    for (size_t i = 0; i < template_ref->stage_count; ++i) {
        hash = nn_hypothesis_hash_text(hash, template_ref->stages[i].name);
        hash = nn_hypothesis_hash_u64(hash, (uint64_t)template_ref->stages[i].kind);
        hash = nn_hypothesis_hash_u64(hash, (uint64_t)template_ref->stages[i].input_type);
        hash = nn_hypothesis_hash_u64(hash, (uint64_t)template_ref->stages[i].output_type);
        hash = nn_hypothesis_hash_u64(hash, (uint64_t)template_ref->stages[i].template_id);
    }
    hash = nn_hypothesis_hash_u64(hash, (uint64_t)hypothesis->slot_count);
    for (size_t i = 0; i < hypothesis->slot_count; ++i) {
        hash = nn_hypothesis_hash_text(hash, hypothesis->slots[i].name);
        hash = nn_hypothesis_hash_u64(hash, (uint64_t)hypothesis->slots[i].type);
        hash = nn_hypothesis_hash_double(hash, hypothesis->slots[i].value);
        hash = nn_hypothesis_hash_u64(hash, (uint64_t)hypothesis->slots[i].integer);
        hash = nn_hypothesis_hash_u64(hash, hypothesis->slots[i].bound ? 1u : 0u);
    }
    hypothesis->canonical_hash = hash;
    int written = snprintf(hypothesis->canonical_id,
                           sizeof(hypothesis->canonical_id),
                           "%s:%s:%u:%016llx",
                           template_ref->domain,
                           template_ref->name,
                           (unsigned int)template_ref->template_id,
                           (unsigned long long)hash);
    if (written < 0) {
        hypothesis->canonical_id[0] = '\0';
    }
    return hash;
}

const NNFeatureSchema *nn_hypothesis_feature_schema(size_t *feature_count_out)
{
    if (feature_count_out != NULL) {
        *feature_count_out = NN_HYPOTHESIS_FEATURE_COUNT;
    }
    return NN_HYPOTHESIS_FEATURE_SCHEMA;
}

NNFeatureAudit nn_hypothesis_feature_audit(void)
{
    return nn_feature_schema_audit(NN_HYPOTHESIS_FEATURE_SCHEMA,
                                   NN_HYPOTHESIS_FEATURE_COUNT,
                                   NN_LEAKAGE_PRIORS_ALLOWED);
}

int nn_hypothesis_encode_features(const NNHypothesis *hypothesis,
                                  nn_real *features,
                                  size_t capacity,
                                  size_t *feature_count_out)
{
    if (hypothesis == NULL || hypothesis->template_ref == NULL ||
        features == NULL || capacity < NN_HYPOTHESIS_FEATURE_COUNT) {
        return 0;
    }
    memset(features, 0, NN_HYPOTHESIS_FEATURE_COUNT * sizeof(*features));

    const NNHypothesisTemplate *template_ref = hypothesis->template_ref;
    size_t bound = 0;
    size_t color_slots = 0;
    size_t offset_slots = 0;
    size_t selector_slots = 0;
    size_t relation_slots = 0;
    size_t render_slots = 0;
    int has_selector_stage = 0;
    int has_transform_stage = 0;
    int default_only_required = 0;

    double value_abs_sum = 0.0;
    double value_signed_sum = 0.0;
    double value_max_abs = 0.0;
    double integer_abs_sum = 0.0;
    double color_sum = 0.0;
    double offset_l1 = 0.0;

    for (size_t i = 0; i < template_ref->stage_count; ++i) {
        if (template_ref->stages[i].kind == NN_PROGRAM_KIND_SELECTOR) {
            has_selector_stage = 1;
        }
        if (template_ref->stages[i].kind == NN_PROGRAM_KIND_TRANSFORM) {
            has_transform_stage = 1;
        }
    }

    for (size_t i = 0; i < hypothesis->slot_count; ++i) {
        const NNHypothesisSlotValue *slot = &hypothesis->slots[i];
        if (slot->bound) {
            ++bound;
        }
        double abs_value = fabs(slot->value);
        value_abs_sum += abs_value;
        value_signed_sum += slot->value;
        if (abs_value > value_max_abs) {
            value_max_abs = abs_value;
        }
        integer_abs_sum += fabs((double)slot->integer);
        switch (slot->type) {
        case NN_HYPOTHESIS_SLOT_COLOR:
        case NN_HYPOTHESIS_SLOT_REFERENCE_COLOR:
            ++color_slots;
            color_sum += slot->value;
            break;
        case NN_HYPOTHESIS_SLOT_OFFSET:
            ++offset_slots;
            offset_l1 += abs_value;
            break;
        case NN_HYPOTHESIS_SLOT_SELECTOR_KIND:
            ++selector_slots;
            break;
        case NN_HYPOTHESIS_SLOT_RELATION_KIND:
            ++relation_slots;
            break;
        case NN_HYPOTHESIS_SLOT_RENDER_MODE:
            ++render_slots;
            break;
        default:
            break;
        }
        if (i < template_ref->slot_count) {
            const NNHypothesisSlotSpec *spec = &template_ref->slots[i];
            if (spec->required && !slot->bound) {
                default_only_required = 1;
            }
        }
    }

    double slot_div = hypothesis->slot_count == 0
                          ? 1.0
                          : (double)hypothesis->slot_count;

    features[0] = (nn_real)1;
    features[1] = nn_hypothesis_clamped_unit((double)template_ref->template_id, 1024.0);
    features[2] = nn_hypothesis_clamped_unit((double)template_ref->stage_count,
                                             (double)NN_HYPOTHESIS_MAX_STAGES);
    features[3] = nn_hypothesis_clamped_unit((double)hypothesis->slot_count,
                                             (double)NN_HYPOTHESIS_MAX_SLOTS);
    features[4] = nn_hypothesis_clamped_unit((double)bound, slot_div);
    features[5] = template_ref->input_type == NN_PROGRAM_VALUE_GRID
                      ? (nn_real)1
                      : (nn_real)0;
    features[6] = template_ref->output_type == NN_PROGRAM_VALUE_GRID
                      ? (nn_real)1
                      : (nn_real)0;
    features[7] = template_ref->output_type == NN_PROGRAM_VALUE_OBJECT_SET
                      ? (nn_real)1
                      : (nn_real)0;
    features[8] = template_ref->output_type == NN_PROGRAM_VALUE_OBJECT
                      ? (nn_real)1
                      : (nn_real)0;
    features[9] = has_selector_stage ? (nn_real)1 : (nn_real)0;
    features[10] = has_transform_stage ? (nn_real)1 : (nn_real)0;
    features[11] = nn_hypothesis_clamped_unit((double)color_slots, 4.0);
    features[12] = nn_hypothesis_clamped_unit((double)offset_slots, 4.0);
    features[13] = nn_hypothesis_clamped_unit((double)selector_slots, 2.0);
    features[14] = nn_hypothesis_clamped_unit((double)relation_slots, 2.0);
    features[15] = nn_hypothesis_clamped_unit((double)render_slots, 2.0);
    features[16] = nn_hypothesis_clamped_unit(value_abs_sum / slot_div, 16.0);
    features[17] = nn_hypothesis_clamped_unit(value_max_abs, 64.0);
    features[18] = nn_hypothesis_clamped_unit(value_signed_sum / slot_div, 16.0);
    features[19] = nn_hypothesis_clamped_unit(integer_abs_sum / slot_div, 64.0);
    features[20] = nn_hypothesis_clamped_unit(
        color_slots == 0 ? 0.0 : color_sum / (double)color_slots, 9.0);
    features[21] = nn_hypothesis_clamped_unit(offset_l1, 16.0);
    features[22] = template_ref->input_type == template_ref->output_type
                       ? (nn_real)1
                       : (nn_real)0;
    features[23] = default_only_required ? (nn_real)1 : (nn_real)0;

    if (feature_count_out != NULL) {
        *feature_count_out = NN_HYPOTHESIS_FEATURE_COUNT;
    }
    return 1;
}

static int nn_hypothesis_slot_matches_stage(const char *slot_name,
                                            const char *stage_name)
{
    if (slot_name == NULL || stage_name == NULL) {
        return 0;
    }
    size_t stage_len = strlen(stage_name);
    if (stage_len == 0) {
        return 0;
    }
    if (strncmp(slot_name, stage_name, stage_len) != 0) {
        return 0;
    }
    return slot_name[stage_len] == '.';
}

static const char *nn_hypothesis_slot_local_name(const char *slot_name,
                                                 const char *stage_name)
{
    size_t stage_len = strlen(stage_name);
    return slot_name + stage_len + 1u;
}

static int nn_hypothesis_apply_slots_to_stage(const NNHypothesis *hypothesis,
                                              NNProgram *stage_program,
                                              size_t stage_index)
{
    const NNHypothesisTemplate *template_ref = hypothesis->template_ref;
    const NNHypothesisStage *stage = &template_ref->stages[stage_index];
    for (size_t i = 0; i < hypothesis->slot_count; ++i) {
        const NNHypothesisSlotValue *slot = &hypothesis->slots[i];
        const char *param_name = NULL;
        if (nn_hypothesis_slot_matches_stage(slot->name, stage->name)) {
            param_name = nn_hypothesis_slot_local_name(slot->name, stage->name);
        } else if (strchr(slot->name, '.') == NULL) {
            param_name = slot->name;
        }
        if (param_name == NULL || param_name[0] == '\0') {
            continue;
        }
        if (!nn_program_add_param(stage_program,
                                  param_name,
                                  slot->value,
                                  slot->integer)) {
            return 0;
        }
    }
    return 1;
}

int nn_hypothesis_compile(const NNHypothesis *hypothesis,
                          NNProgram *program_out)
{
    if (hypothesis == NULL || program_out == NULL ||
        hypothesis->template_ref == NULL ||
        !nn_hypothesis_template_is_valid(hypothesis->template_ref) ||
        !nn_hypothesis_is_fully_bound(hypothesis)) {
        return 0;
    }
    const NNHypothesisTemplate *template_ref = hypothesis->template_ref;
    NNProgram stages[NN_HYPOTHESIS_MAX_STAGES];
    for (size_t i = 0; i < template_ref->stage_count; ++i) {
        nn_program_init(&stages[i],
                        template_ref->domain,
                        template_ref->stages[i].name,
                        template_ref->stages[i].kind,
                        template_ref->stages[i].input_type,
                        template_ref->stages[i].output_type,
                        template_ref->stages[i].template_id);
        if (!nn_hypothesis_apply_slots_to_stage(hypothesis, &stages[i], i)) {
            return 0;
        }
    }

    if (template_ref->stage_count == 1) {
        *program_out = stages[0];
    } else {
        NNProgram accumulator;
        char composed_name[NN_PROGRAM_MAX_NAME];
        snprintf(composed_name,
                 sizeof(composed_name),
                 "%s_then_%s",
                 template_ref->stages[0].name,
                 template_ref->stages[1].name);
        if (!nn_program_compose(&accumulator,
                                &stages[0],
                                &stages[1],
                                composed_name)) {
            return 0;
        }
        for (size_t i = 2; i < template_ref->stage_count; ++i) {
            NNProgram next;
            snprintf(composed_name,
                     sizeof(composed_name),
                     "%s_then_%s",
                     accumulator.name,
                     template_ref->stages[i].name);
            if (!nn_program_compose(&next,
                                    &accumulator,
                                    &stages[i],
                                    composed_name)) {
                return 0;
            }
            accumulator = next;
        }
        *program_out = accumulator;
    }

    /* Rename the composed program so its canonical id binds back to the
     * hypothesis template identity. */
    snprintf(program_out->name, sizeof(program_out->name), "%s", template_ref->name);

    nn_program_add_param(program_out,
                         "hypothesis_template_id",
                         (double)template_ref->template_id,
                         (int64_t)template_ref->template_id);
    nn_program_add_param(program_out,
                         "hypothesis_hash_lo",
                         (double)(uint32_t)(hypothesis->canonical_hash & 0xffffffffu),
                         (int64_t)(hypothesis->canonical_hash & 0xffffffffu));
    nn_program_add_param(program_out,
                         "hypothesis_hash_hi",
                         (double)(uint32_t)((hypothesis->canonical_hash >> 32) & 0xffffffffu),
                         (int64_t)((hypothesis->canonical_hash >> 32) & 0xffffffffu));
    nn_program_refresh_canonical_id(program_out);
    return 1;
}

int nn_hypothesis_write_evidence_json(FILE *fp,
                                      const NNHypothesis *hypothesis,
                                      const NNHypothesisEvidence *evidence)
{
    if (fp == NULL || hypothesis == NULL || hypothesis->template_ref == NULL ||
        evidence == NULL) {
        return 0;
    }
    const NNHypothesisTemplate *template_ref = hypothesis->template_ref;
    const NNProgram *program = evidence->program;
    NNFeatureAudit audit = nn_hypothesis_feature_audit();

    fputs("{\"schema\":\"crlp.hypothesis.evidence.v1\",\"report_version\":1", fp);
    fputs(",\"task\":", fp);
    nn_hypothesis_json_string(fp, evidence->task != NULL ? evidence->task : template_ref->domain);
    fputs(",\"task_instance\":", fp);
    nn_hypothesis_json_string(fp, evidence->task_instance != NULL
                                      ? evidence->task_instance
                                      : "unknown");
    fputs(",\"template_domain\":", fp);
    nn_hypothesis_json_string(fp, template_ref->domain);
    fputs(",\"template_name\":", fp);
    nn_hypothesis_json_string(fp, template_ref->name);
    fprintf(fp,
            ",\"template_id\":%u,\"stage_count\":%zu,\"slot_count\":%zu,"
            "\"feature_encoder_version\":%u",
            (unsigned int)template_ref->template_id,
            template_ref->stage_count,
            hypothesis->slot_count,
            (unsigned int)NN_HYPOTHESIS_FEATURE_ENCODER_VERSION);
    fputs(",\"canonical_id\":", fp);
    nn_hypothesis_json_string(fp, hypothesis->canonical_id);
    fprintf(fp,
            ",\"canonical_hash\":\"%016llx\"",
            (unsigned long long)hypothesis->canonical_hash);
    fputs(",\"program_canonical_id\":", fp);
    nn_hypothesis_json_string(fp, program != NULL ? program->canonical_id : "");
    fprintf(fp,
            ",\"program_canonical_hash\":\"%016llx\"",
            (unsigned long long)(program != NULL ? program->canonical_hash : 0u));
    fputs(",\"certificate_status\":", fp);
    nn_hypothesis_json_string(fp, nn_certificate_status_name(evidence->certificate_status));
    fputs(",\"proof_kind\":", fp);
    nn_hypothesis_json_string(fp, evidence->proof_kind != NULL ? evidence->proof_kind : "");
    fputs(",\"reason\":", fp);
    nn_hypothesis_json_string(fp, evidence->reason != NULL ? evidence->reason : "");
    fprintf(fp,
            ",\"event_count\":%zu,\"covered_events\":%zu,\"support\":%zu,"
            "\"false_rejections\":%zu,\"utility\":%.8f",
            evidence->event_count,
            evidence->covered_events,
            evidence->support,
            evidence->false_rejections,
            evidence->utility);

    fputs(",\"slots\":[", fp);
    for (size_t i = 0; i < hypothesis->slot_count; ++i) {
        const NNHypothesisSlotValue *slot = &hypothesis->slots[i];
        if (i != 0) {
            fputc(',', fp);
        }
        fputs("{\"name\":", fp);
        nn_hypothesis_json_string(fp, slot->name);
        fputs(",\"type\":", fp);
        nn_hypothesis_json_string(fp, nn_hypothesis_slot_type_name(slot->type));
        fprintf(fp,
                ",\"value\":%.8f,\"integer\":%lld,\"bound\":%s}",
                slot->value,
                (long long)slot->integer,
                slot->bound ? "true" : "false");
    }
    fputs("]", fp);
    fprintf(fp,
            ",\"feature_audit\":{\"total\":%zu,\"structural\":%zu,"
            "\"verifier_derived\":%zu,\"gate_derived\":%zu,"
            "\"scorer_input_compliant\":%s}}\n",
            audit.total_features,
            audit.structural_prior_features,
            audit.verifier_derived_features,
            audit.gate_derived_features,
            audit.scorer_input_is_policy_compliant ? "true" : "false");
    return 1;
}
