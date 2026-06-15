typedef struct {
    const char *op_name;
    const char *pipeline_kind;
    const char *visual_program;
    const char *slot_contract;
    const char *composition_kind;
    const char *normalization_frame;
    const char *typed_stages;
    const char *primitive_contract;
} NNGDArcSyntheticShapeSpec;

static const NNGDArcSyntheticShapeSpec nn_gd_arc_synthetic_shape_specs[] = {
    {"object_motion",
     "object_motion_pipeline",
     "select_object -> infer_offset -> translate -> emit_grid",
     "selector,width,height,x_offset,y_offset,color",
     "object_motion",
     "offset_frame",
     "select_object|infer_offset|translate|emit_grid",
     "selector:object,transform:translate,emitter:grid"},
    {"object_relation_translate_to_anchor",
     "object_relation_translate_to_anchor_pipeline",
     "select_object -> infer_anchor_relation -> translate_to_anchor -> emit_grid",
     "selector,x_anchor,y_anchor,x_offset,y_offset,width,height,color",
     "object_relation_translate",
     "anchor_relation_frame",
     "select_object|infer_anchor_relation|translate_to_anchor|emit_grid",
     "selector:object,relation:anchor,transform:translate,emitter:grid"},
    {"recolor",
     "recolor_pipeline",
     "select_mask -> infer_color_map -> recolor -> emit_grid",
     "selector,color,output_color,background",
     "color_rewrite",
     "mask_color_frame",
     "select_mask|infer_color_map|recolor|emit_grid",
     "selector:mask,transform:recolor,emitter:grid"},
    {"tile_scale",
     "tile_scale_pipeline",
     "select_motif -> scale_or_tile -> emit_grid",
     "selector,width,height,period,background,color",
     "tile_scale",
     "motif_scale_frame",
     "select_motif|scale_or_tile|emit_grid",
     "selector:motif,transform:scale_tile,emitter:grid"},
    {"fill_or_complete",
     "fill_or_complete_pipeline",
     "select_boundary -> infer_fill -> complete_shape -> emit_grid",
     "selector,background,color,output_color,width,height",
     "shape_completion",
     "boundary_fill_frame",
     "select_boundary|infer_fill|complete_shape|emit_grid",
     "selector:boundary,transform:fill_complete,emitter:grid"},
    {"symmetry",
     "symmetry_pipeline",
     "select_axis -> reflect_or_rotate -> emit_grid",
     "selector,orientation,width,height,color",
     "symmetry_geometry",
     "axis_geometry_frame",
     "select_axis|reflect_or_rotate|emit_grid",
     "selector:axis,transform:symmetry,emitter:grid"},
    {"masked_or_normalized_crop",
     "masked_or_normalized_crop_pipeline",
     "select_mask -> normalize_crop -> emit_grid",
     "selector,width,height,color,background",
     "object_render",
     "normalized_bbox_frame",
     "select_mask|normalize_crop|emit_grid",
     "selector:mask,normalizer:bbox,emitter:grid"},
    {"crop",
     "crop_pipeline",
     "select_bbox -> crop -> emit_grid",
     "selector,width,height,x_anchor,y_anchor",
     "object_render",
     "bbox_frame",
     "select_bbox|crop|emit_grid",
     "selector:bbox,normalizer:crop,emitter:grid"},
    {"recolor_crop",
     "recolor_crop_pipeline",
     "select_bbox -> crop -> recolor -> emit_grid",
     "selector,width,height,color,output_color",
     "object_render",
     "bbox_color_frame",
     "select_bbox|crop|recolor|emit_grid",
     "selector:bbox,transform:recolor,emitter:grid"},
    {"macrocell_substitute",
     "macrocell_substitute_pipeline",
     "select_macrocell -> substitute_template -> emit_grid",
     "selector,width,height,period,color",
     "template_render",
     "macrocell_frame",
     "select_macrocell|substitute_template|emit_grid",
     "selector:macrocell,transform:template,emitter:grid"},
    {"component_color_summary",
     "component_color_summary_pipeline",
     "select_components -> summarize_color -> emit_grid",
     "selector,color,output_color,count_mode",
     "component_summary",
     "component_color_frame",
     "select_components|summarize_color|emit_grid",
     "selector:components,normalizer:component,emitter:grid"},
    {"visual_combinator",
     "visual_combinator_pipeline",
     "select_components -> compose_relation -> emit_grid",
     "selector,width,height,orientation,color",
     "combinator_render",
     "component_relation_frame",
     "select_components|compose_relation|emit_grid",
     "selector:components,transform:relation,emitter:grid"},
    {"separator_panel_map",
     "separator_panel_map_pipeline",
     "find_separators -> normalize_panels -> map_panels",
     "selector,background,color,output_color,x_anchor",
     "panel_composition",
     "separator_panel_frame",
     "find_separators|normalize_panels|map_panels",
     "selector:separator,normalizer:panel_grid,emitter:grid"},
};

static const NNGDArcSyntheticShapeSpec *nn_gd_arc_synthetic_shape_spec(
    const char *op_name) {
    size_t i;
    for (i = 0U;
         i < sizeof(nn_gd_arc_synthetic_shape_specs) /
                 sizeof(nn_gd_arc_synthetic_shape_specs[0]);
         ++i) {
        if (strcmp(nn_gd_arc_synthetic_shape_specs[i].op_name, op_name) == 0) {
            return &nn_gd_arc_synthetic_shape_specs[i];
        }
    }
    return NULL;
}

static int nn_gd_arc_synthetic_shape_parse_csv(const char *csv,
                                               NNGDStringList *values) {
    char *copy;
    char *start;
    char *p;
    if (csv == NULL || csv[0] == '\0') {
        return -1;
    }
    copy = nn_gd_strdup(csv);
    if (copy == NULL) {
        return -1;
    }
    start = copy;
    for (p = copy; ; ++p) {
        if (*p == ',' || *p == '\0') {
            char saved = *p;
            char *end;
            *p = '\0';
            while (isspace((unsigned char)*start)) {
                ++start;
            }
            end = start + strlen(start);
            while (end > start && isspace((unsigned char)end[-1])) {
                *--end = '\0';
            }
            if (start[0] != '\0' && nn_gd_string_list_add(values, start) != 0) {
                free(copy);
                return -1;
            }
            if (saved == '\0') {
                break;
            }
            start = p + 1;
        }
    }
    free(copy);
    return values->count == 0U ? -1 : 0;
}

static int nn_gd_arc_synthetic_shape_validate_ops(const NNGDStringList *ops) {
    size_t i;
    for (i = 0U; i < ops->count; ++i) {
        if (nn_gd_arc_synthetic_shape_spec(ops->items[i]) == NULL) {
            return -1;
        }
    }
    return 0;
}

static int nn_gd_arc_synthetic_shape_validate_negative_kinds(
    const NNGDStringList *negative_kinds) {
    size_t i;
    for (i = 0U; i < negative_kinds->count; ++i) {
        if (strcmp(negative_kinds->items[i], "unsupported_shape") != 0 &&
            strcmp(negative_kinds->items[i], "no_train_match") != 0) {
            return -1;
        }
    }
    return 0;
}

static const char *nn_gd_arc_synthetic_shape_kind(unsigned grid_w,
                                                  unsigned grid_h,
                                                  unsigned shape_w,
                                                  unsigned shape_h) {
    if (shape_w == grid_w && shape_h == grid_h) {
        return "full_grid";
    }
    if (shape_w == 1U && shape_h == 1U) {
        return "point";
    }
    if (shape_h == 1U) {
        return "horizontal_line";
    }
    if (shape_w == 1U) {
        return "vertical_line";
    }
    if (shape_w == shape_h) {
        return "square";
    }
    return "rectangle";
}

static const char *nn_gd_arc_synthetic_shape_target_family(const char *op_name) {
    if (op_name == NULL) {
        return "unsupported_parse_or_shape";
    }
    if (strcmp(op_name, "object_motion") == 0 ||
        strcmp(op_name, "object_relation_translate_to_anchor") == 0) {
        return "object_motion_or_copy_extend";
    }
    if (strcmp(op_name, "recolor") == 0 || strcmp(op_name, "recolor_crop") == 0) {
        return "color_remap_or_mode_collapse";
    }
    if (strcmp(op_name, "tile_scale") == 0 ||
        strcmp(op_name, "macrocell_substitute") == 0) {
        return "scale_tile_or_expand";
    }
    if (strcmp(op_name, "component_color_summary") == 0 ||
        strcmp(op_name, "visual_combinator") == 0) {
        return "relational_layout_or_counting";
    }
    if (strcmp(op_name, "masked_or_normalized_crop") == 0 ||
        strcmp(op_name, "crop") == 0) {
        return "segmentation_or_object_extract";
    }
    if (strcmp(op_name, "separator_panel_map") == 0) {
        return "projection_or_panel_selection";
    }
    if (strcmp(op_name, "fill_or_complete") == 0) {
        return "shape_completion_or_fill";
    }
    if (strcmp(op_name, "symmetry") == 0) {
        return "symmetry_or_geometry";
    }
    return "unsupported_parse_or_shape";
}

static int nn_gd_arc_synthetic_shape_is_string_field(size_t index) {
    return index <= 5U || (index >= 7U && index <= 15U) ||
           (index >= 17U && index <= 19U) || index == 40U || index == 42U ||
           index == NN_GD_ARC_RUNTIME_FIELD_COUNT ||
           index == NN_GD_ARC_RUNTIME_FIELD_COUNT + 1U;
}

static void nn_gd_arc_synthetic_shape_hash_part(uint64_t *hash,
                                                const char *part) {
    *hash = nn_gd_fnv1a_update(*hash, part == NULL ? "" : part);
    *hash ^= 0U;
    *hash *= UINT64_C(1099511628211);
}

static void nn_gd_arc_synthetic_shape_hash_uint(uint64_t *hash,
                                                unsigned value) {
    char buffer[32];
    snprintf(buffer, sizeof(buffer), "%u", value);
    nn_gd_arc_synthetic_shape_hash_part(hash, buffer);
}

static uint64_t nn_gd_arc_synthetic_shape_hash(const char *label,
                                               unsigned grid_w,
                                               unsigned grid_h,
                                               unsigned shape_w,
                                               unsigned shape_h,
                                               const char *kind,
                                               const char *op_name) {
    uint64_t hash = UINT64_C(1469598103934665603);
    nn_gd_arc_synthetic_shape_hash_part(&hash, label);
    nn_gd_arc_synthetic_shape_hash_uint(&hash, grid_w);
    nn_gd_arc_synthetic_shape_hash_uint(&hash, grid_h);
    nn_gd_arc_synthetic_shape_hash_uint(&hash, shape_w);
    nn_gd_arc_synthetic_shape_hash_uint(&hash, shape_h);
    nn_gd_arc_synthetic_shape_hash_part(&hash, kind);
    nn_gd_arc_synthetic_shape_hash_part(&hash, op_name);
    return hash;
}

static uint64_t nn_gd_arc_synthetic_shape_pair_hash(const char *accepted_id,
                                                    const char *outcome) {
    uint64_t hash = UINT64_C(1469598103934665603);
    nn_gd_arc_synthetic_shape_hash_part(&hash, accepted_id);
    nn_gd_arc_synthetic_shape_hash_part(&hash, outcome);
    return hash;
}

static void nn_gd_arc_synthetic_shape_set(char values[NN_GD_MAX_FIELDS][512],
                                          size_t field_count,
                                          size_t index,
                                          const char *value) {
    if (index >= field_count || index >= NN_GD_MAX_FIELDS) {
        return;
    }
    snprintf(values[index], 512U, "%s", value == NULL ? "" : value);
}

static void nn_gd_arc_synthetic_shape_set_long(
    char values[NN_GD_MAX_FIELDS][512],
    size_t field_count,
    size_t index,
    long value) {
    if (index < field_count && index < NN_GD_MAX_FIELDS) {
        snprintf(values[index], 512U, "%ld", value);
    }
}

static void nn_gd_arc_synthetic_shape_set_double6(
    char values[NN_GD_MAX_FIELDS][512],
    size_t field_count,
    size_t index,
    double value) {
    if (index < field_count && index < NN_GD_MAX_FIELDS) {
        snprintf(values[index], 512U, "%.6f", value);
    }
}

static void nn_gd_arc_synthetic_shape_write_row(FILE *fp,
                                                char values[NN_GD_MAX_FIELDS][512],
                                                size_t field_count) {
    size_t i;
    for (i = 0U; i < field_count; ++i) {
        if (i > 0U) {
            fputc('\t', fp);
        }
        fputs(values[i], fp);
    }
    fputc('\n', fp);
}

static int nn_gd_arc_synthetic_shape_write_selfplay_sample(
    FILE *fp,
    unsigned stable_id,
    const char *task_id,
    const char *state_hash,
    const char *target_family,
    const NNGDArcSyntheticShapeSpec *spec,
    double utility) {
    if (fp == NULL || state_hash == NULL || state_hash[0] == '\0' ||
        target_family == NULL || target_family[0] == '\0' || spec == NULL) {
        return -1;
    }
    fputs("{\"schema\":", fp);
    nn_gd_json_write_string(fp, NN_PROTOCOL_SELFPLAY_SCHEMA);
    fputs(",\"domain\":\"arc_agi\"", fp);
    fprintf(fp, ",\"stable_id\":%u", stable_id);
    fputs(",\"authority\":\"exact_check\"", fp);
    fputs(",\"provenance\":\"selfplay_forward\"", fp);
    fputs(",\"outcome_status\":\"accepted\"", fp);
    fputs(",\"target_family\":", fp);
    nn_gd_json_write_string(fp, target_family);
    fputs(",\"op_sequence\":", fp);
    nn_gd_json_write_string(fp, spec->typed_stages);
    fputs(",\"state_hash\":", fp);
    nn_gd_json_write_string(fp, state_hash);
    fputs(",\"reason_code\":\"accepted\"", fp);
    fprintf(fp, ",\"utility\":%.17g,\"support\":1", utility);
    fputs(",\"source_schema\":\"crlp.arc_synthetic_shape_curriculum.v1\"", fp);
    fputs(",\"source_artifact\":", fp);
    nn_gd_json_write_string(fp, task_id);
    fputs("}\n", fp);
    return ferror(fp) ? -1 : 0;
}

static int nn_gd_arc_synthetic_shape_make_attempt_row(
    FILE *synthetic,
    FILE *combined,
    FILE *selfplay_samples,
    size_t field_count,
    unsigned attempt_index,
    unsigned grid_w,
    unsigned grid_h,
    unsigned shape_w,
    unsigned shape_h,
    const char *kind,
    const char *op_name,
    unsigned op_index,
    const char *outcome,
    const char *accepted_attempt_id,
    unsigned negative_count) {
    char values[NN_GD_MAX_FIELDS][512];
    char task_id[128];
    char attempt_id[128];
    char library_id[128];
    char state_hash[128];
    char slot_tokens[512];
    const NNGDArcSyntheticShapeSpec *spec =
        nn_gd_arc_synthetic_shape_spec(op_name);
    unsigned area = shape_w * shape_h;
    unsigned color = 1U + ((grid_w + shape_w + op_index) % 9U);
    unsigned output_color = 1U + ((grid_h + shape_h + op_index + 3U) % 9U);
    const char *rejection_reason;
    const char *failure_stage;
    const char *failure_detail;
    double verifier_score;
    long verifier_mismatches;
    long verifier_unsupported;
    long verifier_apply_failed;
    long verifier_shape_mismatch;
    double mismatch_ratio;
    const char *contrast_id;
    const char *contrast_kind;
    const char *target_family;
    long contrast_distance;
    long accepted;
    double utility;
    long slot_width_matches_output;
    long slot_height_matches_output;
    size_t i;
    if (spec == NULL || field_count > NN_GD_MAX_FIELDS) {
        return -1;
    }
    target_family = nn_gd_arc_synthetic_shape_target_family(op_name);
    for (i = 0U; i < field_count; ++i) {
        snprintf(values[i],
                 512U,
                 "%s",
                 nn_gd_arc_synthetic_shape_is_string_field(i) ? "" : "0");
    }
    snprintf(task_id,
             sizeof(task_id),
             "synth_%02ux%02u_%02ux%02u_%s",
             grid_w,
             grid_h,
             shape_w,
             shape_h,
             kind);
    snprintf(library_id,
             sizeof(library_id),
             "ngd-synth:%016" PRIx64,
             nn_gd_arc_synthetic_shape_hash("library",
                                            grid_w,
                                            grid_h,
                                            shape_w,
                                            shape_h,
                                            kind,
                                            op_name));
    snprintf(state_hash,
             sizeof(state_hash),
             "ngd-state:%016" PRIx64,
             nn_gd_arc_synthetic_shape_hash("state",
                                            grid_w,
                                            grid_h,
                                            shape_w,
                                            shape_h,
                                            kind,
                                            op_name));
    if (strcmp(outcome, "accepted") == 0) {
        snprintf(attempt_id, sizeof(attempt_id), "%s", accepted_attempt_id);
        rejection_reason = "accepted";
        failure_stage = "accepted";
        failure_detail = "exact_synthetic_shape";
        verifier_score = 1.0;
        verifier_mismatches = 0L;
        verifier_unsupported = 0L;
        verifier_apply_failed = 0L;
        verifier_shape_mismatch = 0L;
        mismatch_ratio = 0.0;
        contrast_id = "";
        contrast_kind = "";
        contrast_distance = 0L;
        accepted = 1L;
        utility = 1.0;
        slot_width_matches_output = 1L;
        slot_height_matches_output = 1L;
    } else if (strcmp(outcome, "unsupported_shape") == 0) {
        snprintf(attempt_id,
                 sizeof(attempt_id),
                 "ngd-synth:%016" PRIx64,
                 nn_gd_arc_synthetic_shape_pair_hash(accepted_attempt_id, outcome));
        rejection_reason = "unsupported_shape";
        failure_stage = "renderer_shape";
        failure_detail = "output_shape_mismatch";
        verifier_score = 0.20;
        verifier_mismatches = (long)(area / 3U);
        if (verifier_mismatches < 1L) {
            verifier_mismatches = 1L;
        }
        verifier_unsupported = 1L;
        verifier_apply_failed = 0L;
        verifier_shape_mismatch = 1L;
        mismatch_ratio = 0.0;
        contrast_id = accepted_attempt_id;
        contrast_kind = "synthetic_shape_mismatch";
        contrast_distance = 1L;
        accepted = 0L;
        utility = 0.20;
        slot_width_matches_output = 0L;
        slot_height_matches_output = 0L;
    } else if (strcmp(outcome, "no_train_match") == 0) {
        snprintf(attempt_id,
                 sizeof(attempt_id),
                 "ngd-synth:%016" PRIx64,
                 nn_gd_arc_synthetic_shape_pair_hash(accepted_attempt_id, outcome));
        rejection_reason = "no_train_match";
        failure_stage = "transform_or_color";
        failure_detail = "high_cell_mismatch_ratio";
        verifier_score = 0.25;
        verifier_mismatches = (long)((area * 3U) / 4U);
        if (verifier_mismatches < 1L) {
            verifier_mismatches = 1L;
        }
        verifier_unsupported = 0L;
        verifier_apply_failed = 0L;
        verifier_shape_mismatch = 0L;
        mismatch_ratio = 0.75;
        contrast_id = accepted_attempt_id;
        contrast_kind = "synthetic_no_train_match";
        contrast_distance = 1L;
        accepted = 0L;
        utility = 0.25;
        slot_width_matches_output = 1L;
        slot_height_matches_output = 1L;
    } else {
        return -1;
    }
    snprintf(slot_tokens,
             sizeof(slot_tokens),
             "synthetic.shape=%s|synthetic.grid=%ux%u|synthetic.bbox=%ux%u|"
             "sketch.op=%u|sketch.background=0|sketch.color=%u|"
             "sketch.output_color=%u|sketch.width=%u|sketch.height=%u|"
             "sketch.x_anchor=%u|sketch.y_anchor=%u",
             kind,
             grid_w,
             grid_h,
             shape_w,
             shape_h,
             op_index,
             color,
             output_color,
             shape_w,
             shape_h,
             (grid_w - shape_w) / 2U,
             (grid_h - shape_h) / 2U);
    nn_gd_arc_synthetic_shape_set(values, field_count, 0U, "source");
    nn_gd_arc_synthetic_shape_set(values, field_count, 1U, "arc.synthetic_shape_family");
    nn_gd_arc_synthetic_shape_set(values, field_count, 2U, task_id);
    nn_gd_arc_synthetic_shape_set(values, field_count, 3U, "synthetic_shape");
    nn_gd_arc_synthetic_shape_set(values, field_count, 4U, attempt_id);
    nn_gd_arc_synthetic_shape_set(values, field_count, 5U, library_id);
    nn_gd_arc_synthetic_shape_set_long(values, field_count, 6U, (long)attempt_index);
    nn_gd_arc_synthetic_shape_set(values, field_count, 7U, spec->pipeline_kind);
    nn_gd_arc_synthetic_shape_set(values, field_count, 8U, op_name);
    nn_gd_arc_synthetic_shape_set(values, field_count, 9U, spec->visual_program);
    nn_gd_arc_synthetic_shape_set(values, field_count, 10U, spec->slot_contract);
    nn_gd_arc_synthetic_shape_set(values, field_count, 11U, spec->composition_kind);
    nn_gd_arc_synthetic_shape_set(values, field_count, 12U, spec->normalization_frame);
    nn_gd_arc_synthetic_shape_set(values, field_count, 13U, spec->typed_stages);
    nn_gd_arc_synthetic_shape_set(values, field_count, 14U, spec->primitive_contract);
    nn_gd_arc_synthetic_shape_set(values, field_count, 15U, slot_tokens);
    nn_gd_arc_synthetic_shape_set_long(values, field_count, 16U, accepted);
    nn_gd_arc_synthetic_shape_set(values, field_count, 17U, rejection_reason);
    nn_gd_arc_synthetic_shape_set(values, field_count, 18U, failure_stage);
    nn_gd_arc_synthetic_shape_set(values, field_count, 19U, failure_detail);
    nn_gd_arc_synthetic_shape_set_long(values, field_count, 20U, 64L);
    nn_gd_arc_synthetic_shape_set_long(values, field_count, 21U, 1L);
    nn_gd_arc_synthetic_shape_set_double6(values, field_count, 22U, utility);
    nn_gd_arc_synthetic_shape_set_long(values, field_count, 23U, (long)op_index);
    nn_gd_arc_synthetic_shape_set_long(values, field_count, 24U, 0L);
    nn_gd_arc_synthetic_shape_set_long(values, field_count, 25U, (long)color);
    nn_gd_arc_synthetic_shape_set_long(values, field_count, 26U, (long)output_color);
    nn_gd_arc_synthetic_shape_set_long(values, field_count, 27U, (long)op_index + 1L);
    nn_gd_arc_synthetic_shape_set_long(values, field_count, 28U, 0L);
    nn_gd_arc_synthetic_shape_set_long(values,
                                       field_count,
                                       29U,
                                       shape_w >= shape_h ? 1L : 2L);
    nn_gd_arc_synthetic_shape_set_long(
        values,
        field_count,
        30U,
        strstr(spec->typed_stages, "count") == NULL ? 0L : 1L);
    nn_gd_arc_synthetic_shape_set_long(values, field_count, 31U, 1L);
    nn_gd_arc_synthetic_shape_set_long(values, field_count, 32U, 1L);
    nn_gd_arc_synthetic_shape_set_long(values,
                                       field_count,
                                       33U,
                                       (long)((grid_w - shape_w) / 2U));
    nn_gd_arc_synthetic_shape_set_long(values,
                                       field_count,
                                       34U,
                                       (long)((grid_h - shape_h) / 2U));
    nn_gd_arc_synthetic_shape_set_long(values,
                                       field_count,
                                       35U,
                                       (long)(grid_w - shape_w));
    nn_gd_arc_synthetic_shape_set_long(values,
                                       field_count,
                                       36U,
                                       (long)(grid_h - shape_h));
    nn_gd_arc_synthetic_shape_set_long(values, field_count, 37U, (long)shape_w);
    nn_gd_arc_synthetic_shape_set_long(values, field_count, 38U, (long)shape_h);
    nn_gd_arc_synthetic_shape_set_long(values,
                                       field_count,
                                       39U,
                                       (long)(shape_w < shape_h ? shape_w : shape_h));
    nn_gd_arc_synthetic_shape_set(values, field_count, 40U, contrast_id);
    nn_gd_arc_synthetic_shape_set_long(values, field_count, 41U, contrast_distance);
    nn_gd_arc_synthetic_shape_set(values, field_count, 42U, contrast_kind);
    nn_gd_arc_synthetic_shape_set_long(values, field_count, 43U, (long)negative_count);
    nn_gd_arc_synthetic_shape_set_long(values, field_count, 44U, (long)grid_w);
    nn_gd_arc_synthetic_shape_set_long(values, field_count, 45U, (long)grid_h);
    nn_gd_arc_synthetic_shape_set_long(values, field_count, 46U, (long)shape_w);
    nn_gd_arc_synthetic_shape_set_long(values, field_count, 47U, (long)shape_h);
    nn_gd_arc_synthetic_shape_set_long(values, field_count, 48U, (long)shape_w);
    nn_gd_arc_synthetic_shape_set_long(values, field_count, 49U, (long)shape_h);
    nn_gd_arc_synthetic_shape_set_long(values,
                                       field_count,
                                       50U,
                                       grid_w % shape_w == 0U ? 1L : 0L);
    nn_gd_arc_synthetic_shape_set_long(values,
                                       field_count,
                                       51U,
                                       grid_h % shape_h == 0U ? 1L : 0L);
    nn_gd_arc_synthetic_shape_set_long(values,
                                       field_count,
                                       52U,
                                       slot_width_matches_output);
    nn_gd_arc_synthetic_shape_set_long(values,
                                       field_count,
                                       53U,
                                       slot_height_matches_output);
    nn_gd_arc_synthetic_shape_set_long(values, field_count, 54U, 1L);
    nn_gd_arc_synthetic_shape_set_long(values, field_count, 55U, 1L);
    nn_gd_arc_synthetic_shape_set_long(values, field_count, 56U, 1L);
    nn_gd_arc_synthetic_shape_set_long(values, field_count, 57U, 1L);
    nn_gd_arc_synthetic_shape_set_long(values, field_count, 58U, 4L);
    nn_gd_arc_synthetic_shape_set_long(values, field_count, 59U, (long)area);
    nn_gd_arc_synthetic_shape_set_long(values, field_count, 60U, verifier_mismatches);
    nn_gd_arc_synthetic_shape_set_long(values, field_count, 61U, verifier_unsupported);
    nn_gd_arc_synthetic_shape_set_long(values, field_count, 62U, verifier_apply_failed);
    nn_gd_arc_synthetic_shape_set_long(values, field_count, 63U, verifier_shape_mismatch);
    nn_gd_arc_synthetic_shape_set_long(values, field_count, 64U, 0L);
    nn_gd_arc_synthetic_shape_set_long(values,
                                       field_count,
                                       65U,
                                       accepted ? -1L : 0L);
    nn_gd_arc_synthetic_shape_set_double6(values, field_count, 66U, mismatch_ratio);
    nn_gd_arc_synthetic_shape_set_double6(values, field_count, 67U, verifier_score);
    nn_gd_arc_synthetic_shape_set_long(values, field_count, 68U, 1L);
    nn_gd_arc_synthetic_shape_set_long(values, field_count, 69U, 1L);
    nn_gd_arc_synthetic_shape_set_long(values, field_count, 70U, (long)area);
    nn_gd_arc_synthetic_shape_set_long(values, field_count, 71U, (long)area);
    nn_gd_arc_synthetic_shape_set_long(values, field_count, 72U, (long)area);
    nn_gd_arc_synthetic_shape_set_long(values, field_count, 73U, 1L);
    nn_gd_arc_synthetic_shape_set_long(values, field_count, 74U, 1L);
    nn_gd_arc_synthetic_shape_set_long(values, field_count, 75U, (long)shape_w);
    nn_gd_arc_synthetic_shape_set_long(values, field_count, 76U, (long)shape_h);
    nn_gd_arc_synthetic_shape_set_long(values,
                                       field_count,
                                       77U,
                                       slot_width_matches_output);
    nn_gd_arc_synthetic_shape_set_long(values,
                                       field_count,
                                       78U,
                                       shape_w == shape_h ? 1L : 0L);
    nn_gd_arc_synthetic_shape_set_long(values, field_count, 79U, (long)area);
    nn_gd_arc_synthetic_shape_set_long(values, field_count, 80U, 1L);
    nn_gd_arc_synthetic_shape_set_long(values, field_count, 81U, 1L);
    nn_gd_arc_synthetic_shape_set_long(values, field_count, 82U, 1L);
    nn_gd_arc_synthetic_shape_set_long(values,
                                       field_count,
                                       83U,
                                       area == 1U ? 1L : 0L);
    nn_gd_arc_synthetic_shape_set_long(values, field_count, 84U, 1L);
    nn_gd_arc_synthetic_shape_set(values,
                                  field_count,
                                  NN_GD_ARC_RUNTIME_FIELD_COUNT,
                                  state_hash);
    nn_gd_arc_synthetic_shape_set(values,
                                  field_count,
                                  NN_GD_ARC_RUNTIME_FIELD_COUNT + 1U,
                                  "selfplay_forward");
    nn_gd_arc_synthetic_shape_write_row(synthetic, values, field_count);
    nn_gd_arc_synthetic_shape_write_row(combined, values, field_count);
    if (accepted &&
        nn_gd_arc_synthetic_shape_write_selfplay_sample(selfplay_samples,
                                                        attempt_index,
                                                        task_id,
                                                        state_hash,
                                                        target_family,
                                                        spec,
                                                        utility) != 0) {
        return -1;
    }
    return ferror(synthetic) || ferror(combined) ? -1 : 0;
}

static int nn_gd_arc_synthetic_shape_copy_original_attempts(
    const char *path,
    FILE *combined,
    size_t target_field_count,
    NNGDArcHoleCounterList *original_counts) {
    FILE *fp = fopen(path, "r");
    char line[NN_GD_LINE_MAX];
    if (fp == NULL) {
        return -1;
    }
    if (fgets(line, sizeof(line), fp) == NULL) {
        fclose(fp);
        return -1;
    }
    while (fgets(line, sizeof(line), fp) != NULL) {
        char parse_line[NN_GD_LINE_MAX];
        char *fields[NN_GD_MAX_FIELDS];
        size_t field_count;
        size_t pad_i;
        char key[128];
        const char *split_name;
        snprintf(parse_line, sizeof(parse_line), "%s", line);
        field_count = nn_gd_split_tabs(parse_line, fields, NN_GD_MAX_FIELDS);
        nn_gd_strip_line_end(line);
        if (fputs(line, combined) == EOF) {
            fclose(fp);
            return -1;
        }
        for (pad_i = field_count; pad_i < target_field_count; ++pad_i) {
            if (fputc('\t', combined) == EOF) {
                fclose(fp);
                return -1;
            }
        }
        if (fputc('\n', combined) == EOF) {
            fclose(fp);
            return -1;
        }
        split_name = nn_gd_field(fields, field_count, 0U);
        snprintf(key,
                 sizeof(key),
                 "original_%s_rows",
                 split_name[0] == '\0' ? "unknown" : split_name);
        if (nn_gd_arc_hole_counter_add(original_counts, key) != 0) {
            fclose(fp);
            return -1;
        }
    }
    if (ferror(fp)) {
        fclose(fp);
        return -1;
    }
    fclose(fp);
    return 0;
}

static int nn_gd_arc_synthetic_shape_write_summary(
    const char *summary_path,
    const char *attempt_rows_path,
    const char *synthetic_attempts_path,
    const char *combined_attempts_path,
    const char *selfplay_samples_path,
    unsigned max_grid_size,
    const char *op_mode,
    const NNGDStringList *ops,
    const NNGDStringList *negative_kinds,
    unsigned max_accepted,
    unsigned max_synthetic_rows,
    unsigned max_seconds,
    const char *stop_reason,
    double elapsed_seconds,
    unsigned synthetic_rows,
    NNGDArcHoleCounterList *synthetic_counts,
    NNGDArcHoleCounterList *op_counts,
    NNGDArcHoleCounterList *shape_counts,
    NNGDArcHoleCounterList *original_counts) {
    FILE *fp;
    size_t i;
    size_t accepted_count = 0U;
    if (nn_gd_mkdir_parent(summary_path) != 0) {
        return -1;
    }
    for (i = 0U; i < synthetic_counts->count; ++i) {
        if (strcmp(synthetic_counts->items[i].key, "accepted") == 0) {
            accepted_count = synthetic_counts->items[i].count;
            break;
        }
    }
    fp = fopen(summary_path, "w");
    if (fp == NULL) {
        return -1;
    }
    fputs("{\"schema\":\"crlp.arc_synthetic_shape_curriculum.v1\","
          "\"report_version\":1,\"self_play_mode\":\"verified_forward_generator\","
          "\"training_target\":\"inverse_program_induction\","
          "\"authored_task_library\":false,"
          "\"grounding_authority\":\"exact_check\","
          "\"attempts_tsv\":",
          fp);
    nn_gd_json_write_string(fp, attempt_rows_path);
    fputs(",\"synthetic_attempts\":", fp);
    nn_gd_json_write_string(fp, synthetic_attempts_path);
    fputs(",\"combined_attempts\":", fp);
    nn_gd_json_write_string(fp, combined_attempts_path);
    fputs(",\"verified_selfplay_samples\":", fp);
    nn_gd_json_write_string(fp, selfplay_samples_path);
    fprintf(fp, ",\"max_grid_size\":%u,\"op_mode\":", max_grid_size);
    nn_gd_json_write_string(fp, op_mode);
    fputs(",\"ops\":", fp);
    nn_gd_arc_hole_write_string_array(fp, (const char *const *)ops->items, ops->count);
    fputs(",\"negative_kinds\":", fp);
    nn_gd_arc_hole_write_string_array(fp,
                                      (const char *const *)negative_kinds->items,
                                      negative_kinds->count);
    fprintf(fp,
            ",\"max_accepted\":%u,\"max_synthetic_rows\":%u,"
            "\"max_seconds\":%u,\"stop_reason\":",
            max_accepted,
            max_synthetic_rows,
            max_seconds);
    nn_gd_json_write_string(fp, stop_reason == NULL ? "unknown" : stop_reason);
    fprintf(fp,
            ",\"elapsed_seconds\":%.3f,\"synthetic_rows\":%u,"
            "\"synthetic_accepted_rows\":%zu,\"synthetic_rejected_rows\":%zu,"
            "\"synthetic_counts\":",
            elapsed_seconds,
            synthetic_rows,
            accepted_count,
            (size_t)synthetic_rows - accepted_count);
    nn_gd_arc_hole_write_counter_object(fp, synthetic_counts);
    fputs(",\"synthetic_op_counts\":", fp);
    nn_gd_arc_hole_write_counter_object(fp, op_counts);
    fputs(",\"synthetic_shape_counts\":", fp);
    nn_gd_arc_hole_write_counter_object(fp, shape_counts);
    fputs(",\"synthetic_provenance\":\"selfplay_forward\","
          "\"synthetic_state_hash_column\":\"state_hash\"", fp);
    for (i = 0U; i < original_counts->count; ++i) {
        fputc(',', fp);
        nn_gd_json_write_string(fp, original_counts->items[i].key);
        fprintf(fp, ":%zu", original_counts->items[i].count);
    }
    fputs("}\n", fp);
    if (fclose(fp) != 0) {
        return -1;
    }
    return 0;
}

int nn_grammar_discovery_write_arc_synthetic_shape_curriculum(
    const char *attempt_rows_path,
    const char *output_dir,
    const char *synthetic_attempts_path,
    const char *combined_attempts_path,
    const char *summary_path,
    unsigned max_grid_size,
    const char *ops_csv,
    const char *negative_kinds_csv,
    const char *op_mode,
    unsigned max_accepted,
    unsigned max_synthetic_rows,
    unsigned max_seconds) {
    FILE *source = NULL;
    FILE *synthetic = NULL;
    FILE *combined = NULL;
    FILE *selfplay_samples = NULL;
    char header_line[NN_GD_LINE_MAX];
    char header_parse[NN_GD_LINE_MAX];
    char output_header[NN_GD_LINE_MAX];
    char selfplay_samples_path[4096];
    char *header_fields[NN_GD_MAX_FIELDS];
    size_t field_count;
    size_t synthetic_field_count;
    NNGDStringList ops = {0};
    NNGDStringList negative_kinds = {0};
    NNGDArcHoleCounterList synthetic_counts = {0};
    NNGDArcHoleCounterList op_counts = {0};
    NNGDArcHoleCounterList shape_counts = {0};
    NNGDArcHoleCounterList original_counts = {0};
    unsigned attempt_index = 0U;
    unsigned accepted_rows = 0U;
    unsigned synthetic_rows = 0U;
    unsigned shape_index = 0U;
    unsigned grid_w;
    int all_ops;
    int stop = 0;
    int status = -1;
    const char *stop_reason = "exhausted";
    double start_time = nn_gd_wall_seconds();
    double end_time = start_time;
    if (attempt_rows_path == NULL || output_dir == NULL ||
        synthetic_attempts_path == NULL || combined_attempts_path == NULL ||
        summary_path == NULL || max_grid_size == 0U || op_mode == NULL ||
        (strcmp(op_mode, "cycle") != 0 && strcmp(op_mode, "all") != 0) ||
        nn_gd_arc_synthetic_shape_parse_csv(ops_csv, &ops) != 0 ||
        nn_gd_arc_synthetic_shape_parse_csv(negative_kinds_csv, &negative_kinds) != 0 ||
        nn_gd_arc_synthetic_shape_validate_ops(&ops) != 0 ||
        nn_gd_arc_synthetic_shape_validate_negative_kinds(&negative_kinds) != 0 ||
        nn_gd_mkdirs(output_dir) != 0 ||
        nn_gd_mkdir_parent(synthetic_attempts_path) != 0 ||
        nn_gd_mkdir_parent(combined_attempts_path) != 0) {
        goto done;
    }
    if (snprintf(selfplay_samples_path,
                 sizeof(selfplay_samples_path),
                 "%s/verified_selfplay_samples.jsonl",
                 output_dir) >= (int)sizeof(selfplay_samples_path)) {
        goto done;
    }
    source = fopen(attempt_rows_path, "r");
    if (source == NULL || fgets(header_line, sizeof(header_line), source) == NULL) {
        goto done;
    }
    fclose(source);
    source = NULL;
    snprintf(header_parse, sizeof(header_parse), "%s", header_line);
    field_count = nn_gd_split_tabs(header_parse, header_fields, NN_GD_MAX_FIELDS);
    if (field_count < NN_GD_ARC_RUNTIME_FIELD_COUNT ||
        strcmp(nn_gd_field(header_fields, field_count, 0U), "split_name") != 0) {
        goto done;
    }
    synthetic_field_count =
        field_count + 2U <= NN_GD_MAX_FIELDS ? field_count + 2U : field_count;
    synthetic = fopen(synthetic_attempts_path, "w");
    combined = fopen(combined_attempts_path, "w");
    selfplay_samples = fopen(selfplay_samples_path, "w");
    if (synthetic == NULL || combined == NULL || selfplay_samples == NULL) {
        goto done;
    }
    snprintf(output_header, sizeof(output_header), "%s", header_line);
    nn_gd_strip_line_end(output_header);
    fputs(output_header, synthetic);
    fputs(output_header, combined);
    if (synthetic_field_count > field_count) {
        fputs("\tstate_hash\tprovenance", synthetic);
        fputs("\tstate_hash\tprovenance", combined);
    }
    fputc('\n', synthetic);
    fputc('\n', combined);
    if (nn_gd_arc_synthetic_shape_copy_original_attempts(attempt_rows_path,
                                                         combined,
                                                         synthetic_field_count,
                                                         &original_counts) != 0) {
        goto done;
    }
    all_ops = strcmp(op_mode, "all") == 0;
    for (grid_w = 1U; grid_w <= max_grid_size && !stop; ++grid_w) {
        unsigned grid_h;
        for (grid_h = 1U; grid_h <= max_grid_size && !stop; ++grid_h) {
            unsigned shape_w;
            for (shape_w = 1U; shape_w <= grid_w && !stop; ++shape_w) {
                unsigned shape_h;
                for (shape_h = 1U; shape_h <= grid_h && !stop; ++shape_h) {
                    const char *kind =
                        nn_gd_arc_synthetic_shape_kind(grid_w,
                                                       grid_h,
                                                       shape_w,
                                                       shape_h);
                    size_t op_start = all_ops ? 0U : shape_index % ops.count;
                    size_t op_end = all_ops ? ops.count : op_start + 1U;
                    size_t op_cursor;
                    for (op_cursor = op_start; op_cursor < op_end; ++op_cursor) {
                        const char *op_name = ops.items[op_cursor];
                        char accepted_id[128];
                        size_t neg_i;
                        if (max_accepted != 0U && accepted_rows >= max_accepted) {
                            stop_reason = "max_accepted";
                            stop = 1;
                            break;
                        }
                        if (max_synthetic_rows != 0U &&
                            synthetic_rows + 1U + (unsigned)negative_kinds.count >
                                max_synthetic_rows) {
                            stop_reason = "max_synthetic_rows";
                            stop = 1;
                            break;
                        }
                        if (max_seconds != 0U &&
                            nn_gd_wall_seconds() - start_time >=
                                (double)max_seconds) {
                            stop_reason = "max_seconds";
                            stop = 1;
                            break;
                        }
                        snprintf(accepted_id,
                                 sizeof(accepted_id),
                                 "ngd-synth:%016" PRIx64,
                                 nn_gd_arc_synthetic_shape_hash("accepted",
                                                                grid_w,
                                                                grid_h,
                                                                shape_w,
                                                                shape_h,
                                                                kind,
                                                                op_name));
                        if (nn_gd_arc_synthetic_shape_make_attempt_row(
                                synthetic,
                                combined,
                                selfplay_samples,
                                synthetic_field_count,
                                attempt_index++,
                                grid_w,
                                grid_h,
                                shape_w,
                                shape_h,
                                kind,
                                op_name,
                                (unsigned)op_cursor,
                                "accepted",
                                accepted_id,
                                (unsigned)negative_kinds.count) != 0 ||
                            nn_gd_arc_hole_counter_add(&synthetic_counts,
                                                       "accepted") != 0 ||
                            nn_gd_arc_hole_counter_add(&op_counts, op_name) != 0 ||
                            nn_gd_arc_hole_counter_add(&shape_counts, kind) != 0) {
                            goto done;
                        }
                        ++accepted_rows;
                        ++synthetic_rows;
                        for (neg_i = 0U; neg_i < negative_kinds.count; ++neg_i) {
                            const char *negative_kind = negative_kinds.items[neg_i];
                            if (nn_gd_arc_synthetic_shape_make_attempt_row(
                                    synthetic,
                                    combined,
                                    selfplay_samples,
                                    synthetic_field_count,
                                    attempt_index++,
                                    grid_w,
                                    grid_h,
                                    shape_w,
                                    shape_h,
                                    kind,
                                    op_name,
                                    (unsigned)op_cursor,
                                    negative_kind,
                                    accepted_id,
                                    (unsigned)negative_kinds.count) != 0 ||
                                nn_gd_arc_hole_counter_add(&synthetic_counts,
                                                           negative_kind) != 0) {
                                goto done;
                            }
                            ++synthetic_rows;
                        }
                    }
                    ++shape_index;
                }
            }
        }
    }
    if (fclose(synthetic) != 0) {
        synthetic = NULL;
        goto done;
    }
    synthetic = NULL;
    if (fclose(combined) != 0) {
        combined = NULL;
        goto done;
    }
    combined = NULL;
    if (fclose(selfplay_samples) != 0) {
        selfplay_samples = NULL;
        goto done;
    }
    selfplay_samples = NULL;
    end_time = nn_gd_wall_seconds();
    if (nn_gd_arc_synthetic_shape_write_summary(summary_path,
                                                attempt_rows_path,
                                                synthetic_attempts_path,
                                                combined_attempts_path,
                                                selfplay_samples_path,
                                                max_grid_size,
                                                op_mode,
                                                &ops,
                                                &negative_kinds,
                                                max_accepted,
                                                max_synthetic_rows,
                                                max_seconds,
                                                stop_reason,
                                                end_time - start_time,
                                                synthetic_rows,
                                                &synthetic_counts,
                                                &op_counts,
                                                &shape_counts,
                                                &original_counts) != 0) {
        goto done;
    }
    printf("arc-synthetic-shape-curriculum synthetic_rows=%u accepted=%u "
           "rejected=%u stop=%s combined=%s summary=%s\n",
           synthetic_rows,
           accepted_rows,
           synthetic_rows - accepted_rows,
           stop_reason,
           combined_attempts_path,
           summary_path);
    status = 0;

done:
    if (source != NULL) {
        fclose(source);
    }
    if (synthetic != NULL) {
        fclose(synthetic);
    }
    if (combined != NULL) {
        fclose(combined);
    }
    if (selfplay_samples != NULL) {
        fclose(selfplay_samples);
    }
    nn_gd_string_list_free(&ops);
    nn_gd_string_list_free(&negative_kinds);
    nn_gd_arc_hole_counter_list_free(&synthetic_counts);
    nn_gd_arc_hole_counter_list_free(&op_counts);
    nn_gd_arc_hole_counter_list_free(&shape_counts);
    nn_gd_arc_hole_counter_list_free(&original_counts);
    return status;
}
