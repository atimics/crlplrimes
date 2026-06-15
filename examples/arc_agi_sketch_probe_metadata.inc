static const char *arc_sketch_op_name(int op)
{
    switch (op) {
    case ARC_SKETCH_OP_CROP:
        return "crop";
    case ARC_SKETCH_OP_TILE_REPEAT:
        return "tile_repeat";
    case ARC_SKETCH_OP_MACROCELL_SUBSTITUTE:
        return "macrocell_substitute";
    case ARC_SKETCH_OP_LATTICE_SAMPLE:
        return "lattice_sample";
    case ARC_SKETCH_OP_COMPONENT_COLOR_SUMMARY:
        return "component_color_summary";
    case ARC_SKETCH_OP_NORMALIZED_CROP:
        return "masked_or_normalized_crop";
    case ARC_SKETCH_OP_POINT_LINE_PROJECT:
        return "point_line_project";
    case ARC_SKETCH_OP_OBJECT_MASK_EXTRACT:
        return "object_mask_extract";
    case ARC_SKETCH_OP_AXIS_PROJECTION_COMPRESS:
        return "axis_projection_compress";
    case ARC_SKETCH_OP_SEPARATOR_PANEL_MAP:
        return "separator_panel_map";
    case ARC_SKETCH_OP_PALETTE_SCALE:
        return "palette_scale";
    case ARC_SKETCH_OP_OBJECT_COLOR_TILE:
        return "object_color_tile";
    case ARC_SKETCH_OP_MARKER_COUNT_MOTIF_TILE:
        return "marker_count_motif_tile";
    case ARC_SKETCH_OP_COMPONENT_COUNT_LINE:
        return "component_count_line";
    case ARC_SKETCH_OP_RECOLOR_CROP:
        return "recolor_crop";
    case ARC_SKETCH_OP_MARKER_POLYLINE:
        return "marker_polyline";
    case ARC_SKETCH_OP_SEPARATOR_INTERSECTION_COLOR:
        return "separator_intersection_color";
    case ARC_SKETCH_OP_SEPARATOR_PANEL_RELATION_MASK:
        return "separator_panel_relation_mask";
    case ARC_SKETCH_OP_VISUAL_COMBINATOR:
        return "visual_combinator";
    case ARC_SKETCH_OP_EXTEND_ALIGNED_SEQUENCE:
        return "extend_aligned_sequence";
    case ARC_SKETCH_OP_OBJECT_TRANSLATE:
        return "object_translate";
    default:
        return "unknown";
    }
}

static const char *arc_visual_pipeline_name(ARCVisualPipelineKind kind)
{
    switch (kind) {
    case ARC_VISUAL_PIPELINE_CROP:
        return "crop_pipeline";
    case ARC_VISUAL_PIPELINE_TILE_REPEAT:
        return "tile_repeat_pipeline";
    case ARC_VISUAL_PIPELINE_MACROCELL_SUBSTITUTE:
        return "macrocell_substitute_pipeline";
    case ARC_VISUAL_PIPELINE_LATTICE_SAMPLE:
        return "lattice_sample_pipeline";
    case ARC_VISUAL_PIPELINE_COMPONENT_COLOR_SUMMARY:
        return "component_color_summary_pipeline";
    case ARC_VISUAL_PIPELINE_NORMALIZED_CROP:
        return "masked_or_normalized_crop_pipeline";
    case ARC_VISUAL_PIPELINE_POINT_LINE_PROJECT:
        return "point_line_project_pipeline";
    case ARC_VISUAL_PIPELINE_OBJECT_MASK_EXTRACT:
        return "object_mask_extract_pipeline";
    case ARC_VISUAL_PIPELINE_AXIS_PROJECTION_COMPRESS:
        return "axis_projection_compress_pipeline";
    case ARC_VISUAL_PIPELINE_SEPARATOR_PANEL_MAP:
        return "separator_panel_map_pipeline";
    case ARC_VISUAL_PIPELINE_PALETTE_SCALE:
        return "palette_scale_pipeline";
    case ARC_VISUAL_PIPELINE_OBJECT_COLOR_TILE:
        return "object_color_tile_pipeline";
    case ARC_VISUAL_PIPELINE_MARKER_COUNT_MOTIF_TILE:
        return "marker_count_motif_tile_pipeline";
    case ARC_VISUAL_PIPELINE_COMPONENT_COUNT_LINE:
        return "component_count_line_pipeline";
    case ARC_VISUAL_PIPELINE_RECOLOR_CROP:
        return "recolor_crop_pipeline";
    case ARC_VISUAL_PIPELINE_MARKER_POLYLINE:
        return "marker_polyline_pipeline";
    case ARC_VISUAL_PIPELINE_SEPARATOR_INTERSECTION_COLOR:
        return "separator_intersection_color_pipeline";
    case ARC_VISUAL_PIPELINE_SEPARATOR_PANEL_RELATION_MASK:
        return "separator_panel_relation_mask_pipeline";
    case ARC_VISUAL_PIPELINE_VISUAL_COMBINATOR:
        return "visual_combinator_pipeline";
    case ARC_VISUAL_PIPELINE_OBJECT_TRANSLATE:
        return "object_translate_pipeline";
    default:
        return "unknown_pipeline";
    }
}

static const char *arc_sketch_probe_reason_name(ARCSketchProbeReason reason)
{
    switch (reason) {
    case ARC_SKETCH_PROBE_REASON_ACCEPTED:
        return "accepted";
    case ARC_SKETCH_PROBE_REASON_UNSUPPORTED_SHAPE:
        return "unsupported_shape";
    case ARC_SKETCH_PROBE_REASON_NO_TRAIN_MATCH:
        return "no_train_match";
    case ARC_SKETCH_PROBE_REASON_DUPLICATE:
        return "duplicate";
    case ARC_SKETCH_PROBE_REASON_CAP_HIT:
        return "cap_hit";
    case ARC_SKETCH_PROBE_REASON_INVALID_ARGUMENT:
        return "invalid_argument";
    default:
        return "unknown";
    }
}

typedef struct {
    int op;
    const char *composition_kind;
    const char *normalization_frame;
    const char *typed_stages;
    const char *visual_program;
    const char *slot_contract;
    const char *primitive_contract;
} ARCVisualOpMetadata;

static const ARCVisualOpMetadata *arc_visual_op_metadata_for_op(int op)
{
    static const ARCVisualOpMetadata metadata[] = {
        {
            ARC_SKETCH_OP_CROP,
            "region_extraction",
            "grid_or_bbox_frame",
            "select_region|normalize_origin|emit_grid",
            "select_region -> normalize_origin -> emit_grid",
            "selector,background,x_anchor,y_anchor,x_offset,y_offset,width,height",
            "selector:region,normalizer:origin,emitter:grid",
        },
        {
            ARC_SKETCH_OP_TILE_REPEAT,
            "lattice_transform",
            "lattice_frame",
            "select_lattice_motif|repeat_motif|emit_grid",
            "select_lattice_motif -> repeat_motif -> emit_grid",
            "selector,background,period",
            "selector:lattice,transform:repeat,emitter:grid",
        },
        {
            ARC_SKETCH_OP_MACROCELL_SUBSTITUTE,
            "lattice_transform",
            "macrocell_frame",
            "sample_cell|map_macrocell|render_expanded_grid",
            "sample_cell -> map_macrocell -> render_expanded_grid",
            "background,color,output_color,period,x_anchor",
            "selector:lattice,transform:macrocell,normalizer:scale",
        },
        {
            ARC_SKETCH_OP_LATTICE_SAMPLE,
            "lattice_transform",
            "lattice_frame",
            "normalize_lattice_phase|sample_lattice|emit_grid",
            "normalize_lattice_phase -> sample_lattice -> emit_grid",
            "background,width,height,period,x_offset,y_offset,x_anchor",
            "selector:lattice,normalizer:phase,transform:sample",
        },
        {
            ARC_SKETCH_OP_COMPONENT_COLOR_SUMMARY,
            "object_summary",
            "centroid_bbox_frame",
            "select_components|normalize_centroids|scatter_colors",
            "select_components -> normalize_centroids -> scatter_colors",
            "background,color,output_color,width,height",
            "selector:components,normalizer:centroid_bbox,emitter:color_grid",
        },
        {
            ARC_SKETCH_OP_NORMALIZED_CROP,
            "region_extraction",
            "normalized_bbox_frame",
            "select_region|normalize_background|emit_grid",
            "select_region -> normalize_background -> emit_grid",
            "selector,background,output_color,x_anchor,y_anchor,x_offset,y_offset,width,height",
            "selector:region,normalizer:background,emitter:grid",
        },
        {
            ARC_SKETCH_OP_POINT_LINE_PROJECT,
            "projection_transform",
            "point_relation_frame",
            "select_points|project_rows_cols|mark_intersections",
            "select_points -> project_rows_cols -> mark_intersections",
            "background,output_color",
            "selector:points,projector:row_col,emitter:grid",
        },
        {
            ARC_SKETCH_OP_OBJECT_MASK_EXTRACT,
            "region_extraction",
            "object_bbox_frame",
            "select_object_mask|crop_bbox|render_mask",
            "select_object_mask -> crop_bbox -> render_mask",
            "background,color,output_color,x_anchor,y_anchor,x_offset,y_offset,width,height",
            "selector:object,normalizer:bbox,normalizer:framed_bbox,emitter:mask",
        },
        {
            ARC_SKETCH_OP_AXIS_PROJECTION_COMPRESS,
            "projection_transform",
            "axis_frame",
            "project_axis|compress_line|emit_grid",
            "project_axis -> compress_line -> emit_grid",
            "background,output_color,x_anchor,y_anchor",
            "selector:axis,projector:compress,emitter:line",
        },
        {
            ARC_SKETCH_OP_SEPARATOR_PANEL_MAP,
            "panel_composition",
            "separator_panel_frame",
            "find_separators|normalize_panels|map_panels",
            "find_separators -> normalize_panels -> map_panels",
            "background,color,output_color,x_anchor",
            "selector:separator,normalizer:panel_grid,emitter:grid",
        },
        {
            ARC_SKETCH_OP_PALETTE_SCALE,
            "count_render",
            "palette_count_frame",
            "count_palette|scale_grid_by_count|emit_grid",
            "count_palette -> scale_grid_by_count -> emit_grid",
            "background,x_anchor",
            "selector:palette,normalizer:count,transform:scale",
        },
        {
            ARC_SKETCH_OP_OBJECT_COLOR_TILE,
            "object_render",
            "object_color_frame",
            "select_object_color|render_solid_rect",
            "select_object_color -> render_solid_rect",
            "background,color,output_color,x_anchor,width,height",
            "selector:object_color,emitter:solid_rect",
        },
        {
            ARC_SKETCH_OP_MARKER_COUNT_MOTIF_TILE,
            "count_render",
            "marker_count_frame",
            "count_marker_cells|select_motif_window|repeat_window_by_count",
            "count_marker_cells -> select_motif_window -> repeat_window_by_count",
            "background,color,output_color,width,height,x_anchor,y_anchor,x_offset,y_offset,period",
            "selector:marker_count,selector:motif,transform:repeat",
        },
        {
            ARC_SKETCH_OP_COMPONENT_COUNT_LINE,
            "count_render",
            "component_count_frame",
            "count_color|render_count_line",
            "count_color -> render_count_line",
            "background,color,output_color,x_anchor,y_anchor",
            "selector:component_count,emitter:line",
        },
        {
            ARC_SKETCH_OP_RECOLOR_CROP,
            "repair_composition",
            "counterexample_region_frame",
            "select_region|apply_color_repair|emit_grid",
            "select_region -> apply_color_repair -> emit_grid",
            "selector,background,color,output_color,x_anchor,y_anchor,x_offset,y_offset,width,height,period",
            "selector:region,normalizer:bbox,repair:color_mask,emitter:grid",
        },
        {
            ARC_SKETCH_OP_MARKER_POLYLINE,
            "repair_composition",
            "same_shape_marker_frame",
            "select_markers|order_markers|draw_polyline",
            "select_markers -> order_markers -> draw_polyline",
            "background,output_color,x_anchor,y_anchor",
            "selector:markers,composer:polyline,emitter:grid",
        },
        {
            ARC_SKETCH_OP_SEPARATOR_INTERSECTION_COLOR,
            "panel_composition",
            "separator_intersection_frame",
            "find_tolerant_separator_lines|intersect_lines|emit_color",
            "find_tolerant_separator_lines -> intersect_lines -> emit_color",
            "background,color,output_color,x_anchor,period",
            "selector:separator_lines,normalizer:intersection,emitter:color_cell",
        },
        {
            ARC_SKETCH_OP_SEPARATOR_PANEL_RELATION_MASK,
            "panel_composition",
            "separator_relation_frame",
            "find_separator|align_two_panels|emit_relation_mask",
            "find_separator -> align_two_panels -> emit_relation_mask",
            "background,color,output_color,x_anchor,period",
            "selector:separator,normalizer:aligned_panels,composer:panel_relation,emitter:mask",
        },
        {
            ARC_SKETCH_OP_VISUAL_COMBINATOR,
            "visual_composition",
            "object_lattice_frame",
            "select_components|normalize_component_origins|classify_component_shape_color|render_component_mask|emit_component_grid",
            "select_components -> normalize_component_origins -> classify_component_shape_color -> render_component_mask -> emit_component_grid",
            "background,color,output_color,width,height,x_anchor,y_anchor,x_offset,y_offset,period",
            "selector:components,selector:component_predicate,normalizer:object_lattice,normalizer:component_mask,normalizer:component_selector_layout,composer:visual_program,composer:aligned_span,emitter:occupancy_grid,emitter:count_template,emitter:component_shape_grid,emitter:component_mask,emitter:aligned_span,classifier:component_shape_color,classifier:marker_glyph_color",
        },
        {
            ARC_SKETCH_OP_EXTEND_ALIGNED_SEQUENCE,
            "sequence_completion",
            "aligned_point_sequence_frame",
            "select_aligned_points|infer_step|extend_sequence|overlay_extension",
            "select_aligned_points -> infer_step -> extend_sequence -> overlay_extension",
            "background,color,output_color,x_anchor,y_anchor",
            "selector:aligned_points,normalizer:step,composer:sequence_extension,emitter:overlay",
        },
        {
            ARC_SKETCH_OP_OBJECT_TRANSLATE,
            "object_motion",
            "object_bbox_frame",
            "select_object|translate_object|render_object_motion",
            "select_object -> translate_object -> render_object_motion",
            "background,color,output_color,x_anchor,y_anchor,x_offset,y_offset,period",
            "selector:object,normalizer:bbox,transform:translate,emitter:blank_copy_or_move",
        },
    };
    for (size_t i = 0; i < sizeof(metadata) / sizeof(metadata[0]); ++i) {
        if (metadata[i].op == op) {
            return &metadata[i];
        }
    }
    return NULL;
}

static int arc_sketch_probe_spec_equal(const ARCSketchProgramSpec *lhs,
                                       const ARCSketchProgramSpec *rhs)
{
    return lhs != NULL && rhs != NULL &&
           lhs->selector == rhs->selector &&
           lhs->op == rhs->op &&
           lhs->background == rhs->background &&
           lhs->color == rhs->color &&
           lhs->output_color == rhs->output_color &&
           lhs->x_anchor == rhs->x_anchor &&
           lhs->y_anchor == rhs->y_anchor &&
           lhs->x_offset == rhs->x_offset &&
           lhs->y_offset == rhs->y_offset &&
           lhs->width == rhs->width &&
           lhs->height == rhs->height &&
           lhs->period == rhs->period;
}

static uint64_t arc_sketch_probe_spec_hash(const char *family,
                                           const char *task_id,
                                           const ARCSketchProgramSpec *spec);

static int arc_slot_contract_has_slot(const char *contract, const char *slot)
{
    size_t slot_len;
    const char *start;
    const char *p;
    if (contract == NULL || slot == NULL || slot[0] == '\0') {
        return 0;
    }
    slot_len = strlen(slot);
    start = contract;
    for (p = contract; ; ++p) {
        if (*p == ',' || *p == '\0') {
            if ((size_t)(p - start) == slot_len &&
                strncmp(start, slot, slot_len) == 0) {
                return 1;
            }
            if (*p == '\0') {
                break;
            }
            start = p + 1;
        }
    }
    return 0;
}

static int64_t arc_sketch_spec_slot_value(const ARCSketchProgramSpec *spec,
                                          const char *slot)
{
    if (strcmp(slot, "selector") == 0) {
        return spec->selector;
    }
    if (strcmp(slot, "background") == 0) {
        return spec->background;
    }
    if (strcmp(slot, "color") == 0) {
        return spec->color;
    }
    if (strcmp(slot, "output_color") == 0) {
        return spec->output_color;
    }
    if (strcmp(slot, "x_anchor") == 0) {
        return spec->x_anchor;
    }
    if (strcmp(slot, "y_anchor") == 0) {
        return spec->y_anchor;
    }
    if (strcmp(slot, "x_offset") == 0) {
        return spec->x_offset;
    }
    if (strcmp(slot, "y_offset") == 0) {
        return spec->y_offset;
    }
    if (strcmp(slot, "width") == 0) {
        return (int64_t)spec->width;
    }
    if (strcmp(slot, "height") == 0) {
        return (int64_t)spec->height;
    }
    if (strcmp(slot, "period") == 0) {
        return (int64_t)spec->period;
    }
    return INT64_MIN;
}

static size_t arc_sketch_active_slot_distance(const ARCSketchProgramSpec *accepted,
                                              const ARCSketchProgramSpec *rejected)
{
    static const char *slots[] = {
        "selector",
        "background",
        "color",
        "output_color",
        "x_anchor",
        "y_anchor",
        "x_offset",
        "y_offset",
        "width",
        "height",
        "period",
    };
    const ARCVisualOpMetadata *metadata;
    size_t distance = 0;
    if (accepted == NULL || rejected == NULL || accepted->op != rejected->op) {
        return SIZE_MAX;
    }
    metadata = arc_visual_op_metadata_for_op(accepted->op);
    if (metadata == NULL) {
        return SIZE_MAX;
    }
    for (size_t i = 0; i < sizeof(slots) / sizeof(slots[0]); ++i) {
        int64_t lhs;
        int64_t rhs;
        if (!arc_slot_contract_has_slot(metadata->slot_contract, slots[i])) {
            continue;
        }
        lhs = arc_sketch_spec_slot_value(accepted, slots[i]);
        rhs = arc_sketch_spec_slot_value(rejected, slots[i]);
        if (lhs != rhs) {
            ++distance;
        }
    }
    return distance;
}

static const char *arc_visual_pipeline_failure_stage(
    const ARCVisualPipelineAttempt *attempt)
{
    if (attempt == NULL) {
        return "unknown";
    }
    if (attempt->accepted || attempt->reason == ARC_SKETCH_PROBE_REASON_ACCEPTED) {
        return "accepted";
    }
    if (attempt->reason == ARC_SKETCH_PROBE_REASON_DUPLICATE ||
        attempt->reason == ARC_SKETCH_PROBE_REASON_CAP_HIT) {
        return "search_control";
    }
    if (attempt->reason == ARC_SKETCH_PROBE_REASON_INVALID_ARGUMENT) {
        return "compiler";
    }
    if (attempt->telemetry.apply_failed_pairs > 0 &&
        attempt->telemetry.checked_cells == 0) {
        return "selector_or_transform";
    }
    if (attempt->telemetry.shape_mismatch_pairs > 0) {
        return "renderer_shape";
    }
    if (attempt->telemetry.unsupported_pairs > 0 &&
        attempt->telemetry.checked_cells == 0) {
        return "normalizer_shape";
    }
    if (attempt->telemetry.mismatches > 0) {
        if (attempt->telemetry.checked_cells > 0 &&
            attempt->telemetry.mismatches * 2u > attempt->telemetry.checked_cells) {
            return "transform_or_color";
        }
        return "renderer_cells";
    }
    if (attempt->reason == ARC_SKETCH_PROBE_REASON_NO_TRAIN_MATCH) {
        return "verifier_distance";
    }
    if (attempt->reason == ARC_SKETCH_PROBE_REASON_UNSUPPORTED_SHAPE) {
        return "unsupported_shape";
    }
    return "unknown";
}

static const char *arc_visual_pipeline_failure_detail(
    const ARCVisualPipelineAttempt *attempt)
{
    const char *stage;
    if (attempt == NULL) {
        return "missing_attempt";
    }
    stage = arc_visual_pipeline_failure_stage(attempt);
    if (strcmp(stage, "accepted") == 0) {
        return "exact_train_match";
    }
    if (strcmp(stage, "search_control") == 0) {
        return arc_sketch_probe_reason_name(attempt->reason);
    }
    if (strcmp(stage, "selector_or_transform") == 0) {
        return "apply_failed_before_cell_comparison";
    }
    if (strcmp(stage, "renderer_shape") == 0) {
        return "output_shape_mismatch";
    }
    if (strcmp(stage, "normalizer_shape") == 0) {
        return "no_comparable_output_cells";
    }
    if (strcmp(stage, "transform_or_color") == 0) {
        return "high_cell_mismatch_ratio";
    }
    if (strcmp(stage, "renderer_cells") == 0) {
        return "low_cell_mismatch_ratio";
    }
    if (strcmp(stage, "verifier_distance") == 0) {
        return "shape_supported_but_train_not_exact";
    }
    return arc_sketch_probe_reason_name(attempt->reason);
}

static void arc_visual_pipeline_attempt_id(const ARCVisualPipelineAttempt *attempt,
                                           char *out,
                                           size_t out_size)
{
    uint64_t hash =
        arc_sketch_probe_spec_hash(attempt->family, attempt->task_id, &attempt->spec);
    snprintf(out,
             out_size,
             "ngd-attempt:%s:%s:%016llx:%zu",
             attempt->family,
             attempt->task_id,
             (unsigned long long)hash,
             attempt->attempt_index);
}

static void arc_visual_pipeline_library_id(const ARCVisualPipelineAttempt *attempt,
                                           char *out,
                                           size_t out_size)
{
    uint64_t hash =
        arc_sketch_probe_spec_hash(attempt->family, attempt->task_id, &attempt->spec);
    snprintf(out,
             out_size,
             "ngd:%s:%s:%016llx",
             attempt->family,
             attempt->task_id,
             (unsigned long long)hash);
}

static void arc_visual_pipeline_attempts_link_contrasts(
    ARCVisualPipelineAttempt *attempts,
    size_t attempt_count)
{
    size_t *accepted_indices = NULL;
    size_t accepted_count = 0;
    for (size_t i = 0; i < attempt_count; ++i) {
        attempts[i].has_contrast_accepted = 0;
        attempts[i].contrast_accepted_log_index = 0;
        attempts[i].contrast_slot_distance = 0;
        attempts[i].contrast_rejected_sibling_count = 0;
    }
    accepted_indices = malloc(attempt_count == 0 ? sizeof(*accepted_indices)
                                                 : attempt_count * sizeof(*accepted_indices));
    if (accepted_indices == NULL) {
        return;
    }
    for (size_t i = 0; i < attempt_count; ++i) {
        if (attempts[i].accepted) {
            accepted_indices[accepted_count++] = i;
        }
    }
    for (size_t rejected_index = 0; rejected_index < attempt_count; ++rejected_index) {
        ARCVisualPipelineAttempt *rejected = &attempts[rejected_index];
        size_t best_index = SIZE_MAX;
        size_t best_distance = SIZE_MAX;
        if (rejected->accepted ||
            (rejected->reason != ARC_SKETCH_PROBE_REASON_UNSUPPORTED_SHAPE &&
             rejected->reason != ARC_SKETCH_PROBE_REASON_NO_TRAIN_MATCH)) {
            continue;
        }
        for (size_t accepted_cursor = 0; accepted_cursor < accepted_count; ++accepted_cursor) {
            size_t accepted_index = accepted_indices[accepted_cursor];
            ARCVisualPipelineAttempt *accepted = &attempts[accepted_index];
            size_t distance;
            if (strcmp(accepted->family, rejected->family) != 0 ||
                strcmp(accepted->task_id, rejected->task_id) != 0 ||
                strcmp(accepted->split, rejected->split) != 0 ||
                accepted->pipeline.kind != rejected->pipeline.kind ||
                accepted->spec.op != rejected->spec.op) {
                continue;
            }
            distance = arc_sketch_active_slot_distance(&accepted->spec, &rejected->spec);
            if (distance == 0 || distance == SIZE_MAX || distance > 2) {
                continue;
            }
            if (distance < best_distance) {
                best_distance = distance;
                best_index = accepted_index;
                if (distance == 1) {
                    break;
                }
            }
        }
        if (best_index != SIZE_MAX) {
            rejected->has_contrast_accepted = 1;
            rejected->contrast_accepted_log_index = best_index;
            rejected->contrast_slot_distance = best_distance;
            ++attempts[best_index].contrast_rejected_sibling_count;
        }
    }
    free(accepted_indices);
}

static void arc_component_inventory_fill(ARCComponentInventoryFeatures *features,
                                         const ARCTask *task,
                                         const ARCSketchProgramSpec *spec)
{
    ARCScene input_scene;
    ARCScene output_scene;
    uint8_t output_colors[ARC_COLOR_COUNT];
    size_t output_color_count = 0;
    size_t shape_widths[ARC_MAX_SCENE_OBJECTS];
    size_t shape_heights[ARC_MAX_SCENE_OBJECTS];
    size_t shape_count = 0;
    size_t selected_min_x = SIZE_MAX;
    size_t selected_min_y = SIZE_MAX;
    size_t selected_max_x = 0;
    size_t selected_max_y = 0;
    if (features == NULL) {
        return;
    }
    memset(features, 0, sizeof(*features));
    if (task == NULL || spec == NULL || task->train_count == 0 ||
        !task->train[0].has_output) {
        return;
    }
    const ARCGrid *input = &task->train[0].input;
    const ARCGrid *output = &task->train[0].output;
    if (!arc_extract_scene(input, spec->background, &input_scene)) {
        return;
    }
    (void)arc_extract_scene(output, spec->background, &output_scene);
    features->input_component_count = input_scene.object_count;
    features->output_component_count = output_scene.object_count;
    output_color_count = arc_grid_nonzero_colors(output, output_colors, ARC_COLOR_COUNT);
    for (size_t i = 0; i < output_color_count; ++i) {
        if (output_colors[i] != spec->background) {
            ++features->output_non_background_color_count;
        }
    }
    features->output_foreground_cell_count =
        arc_grid_cells(output) - arc_grid_color_cell_count(output, spec->background);
    features->selected_component_area_min = SIZE_MAX;
    for (size_t i = 0; i < input_scene.object_count; ++i) {
        const ARCObject *object = &input_scene.objects[i];
        size_t width;
        size_t height;
        int seen_shape = 0;
        if (object->color == spec->background) {
            continue;
        }
        if (spec->color != spec->background && object->color != spec->color) {
            continue;
        }
        width = object->max_x - object->min_x + 1u;
        height = object->max_y - object->min_y + 1u;
        ++features->selected_component_count;
        features->selected_component_area_total += object->count;
        if (object->count < features->selected_component_area_min) {
            features->selected_component_area_min = object->count;
        }
        if (object->count > features->selected_component_area_max) {
            features->selected_component_area_max = object->count;
        }
        if (object->min_x < selected_min_x) {
            selected_min_x = object->min_x;
        }
        if (object->min_y < selected_min_y) {
            selected_min_y = object->min_y;
        }
        if (object->max_x > selected_max_x) {
            selected_max_x = object->max_x;
        }
        if (object->max_y > selected_max_y) {
            selected_max_y = object->max_y;
        }
        for (size_t shape = 0; shape < shape_count; ++shape) {
            if (shape_widths[shape] == width && shape_heights[shape] == height) {
                seen_shape = 1;
                break;
            }
        }
        if (!seen_shape && shape_count < ARC_MAX_SCENE_OBJECTS) {
            shape_widths[shape_count] = width;
            shape_heights[shape_count] = height;
            ++shape_count;
        }
    }
    if (features->selected_component_area_min == SIZE_MAX) {
        features->selected_component_area_min = 0;
    }
    features->selected_component_shape_count = shape_count;
    features->selected_component_same_shape =
        features->selected_component_count > 0 && shape_count <= 1u;
    if (features->selected_component_count > 0 &&
        selected_min_x <= selected_max_x &&
        selected_min_y <= selected_max_y) {
        features->selected_component_bbox_width = selected_max_x - selected_min_x + 1u;
        features->selected_component_bbox_height = selected_max_y - selected_min_y + 1u;
        features->selected_bbox_matches_output_shape =
            features->selected_component_bbox_width == output->width &&
            features->selected_component_bbox_height == output->height;
        features->selected_bbox_rotated_output_shape =
            features->selected_component_bbox_width == output->height &&
            features->selected_component_bbox_height == output->width;
    }
    features->selected_count_matches_output_components =
        features->selected_component_count > 0 &&
        features->selected_component_count == features->output_component_count;
    features->selected_count_matches_output_foreground =
        features->selected_component_count > 0 &&
        features->selected_component_count == features->output_foreground_cell_count;
    features->selected_area_matches_output_foreground =
        features->selected_component_area_total > 0 &&
        features->selected_component_area_total ==
            features->output_foreground_cell_count;
}

static void arc_visual_pipeline_attempt_fill_task_features(
    ARCVisualPipelineAttempt *attempt,
    const ARCTask *task)
{
    if (attempt == NULL || task == NULL || task->train_count == 0 ||
        !task->train[0].has_output) {
        return;
    }
    const ARCGrid *first_input = &task->train[0].input;
    const ARCGrid *first_output = &task->train[0].output;
    arc_component_inventory_fill(&attempt->component_inventory, task, &attempt->spec);
    attempt->input_width = first_input->width;
    attempt->input_height = first_input->height;
    attempt->output_width = first_output->width;
    attempt->output_height = first_output->height;
    if (first_input->width > 0 && first_output->width % first_input->width == 0) {
        attempt->shape_scale_x = first_output->width / first_input->width;
    }
    if (first_input->height > 0 && first_output->height % first_input->height == 0) {
        attempt->shape_scale_y = first_output->height / first_input->height;
    }
    if (attempt->spec.period > 0) {
        attempt->period_divides_input_width =
            first_input->width % attempt->spec.period == 0;
        attempt->period_divides_input_height =
            first_input->height % attempt->spec.period == 0;
    }
    attempt->slot_width_matches_output =
        attempt->spec.width > 0 && attempt->spec.width == first_output->width;
    attempt->slot_height_matches_output =
        attempt->spec.height > 0 && attempt->spec.height == first_output->height;
    for (size_t pair = 0; pair < task->train_count; ++pair) {
        const ARCGrid *input = &task->train[pair].input;
        const ARCGrid *output = &task->train[pair].output;
        if (arc_grid_color_cell_count(input, attempt->spec.color) > 0) {
            attempt->color_in_input = 1;
        }
        if (task->train[pair].has_output &&
            arc_grid_color_cell_count(output, attempt->spec.color) > 0) {
            attempt->color_in_output = 1;
        }
        if (task->train[pair].has_output &&
            arc_grid_color_cell_count(output, attempt->spec.output_color) > 0) {
            attempt->output_color_in_output = 1;
        }
        if (arc_grid_color_cell_count(input, attempt->spec.background) > 0) {
            attempt->background_in_input = 1;
        }
    }
}

static int arc_text_list_contains_token(const char *text,
                                        const char *token,
                                        char delimiter)
{
    const char *cursor = text;
    size_t token_len;
    if (text == NULL || token == NULL || token[0] == '\0') {
        return 0;
    }
    token_len = strlen(token);
    while (*cursor != '\0') {
        while (*cursor == delimiter || isspace((unsigned char)*cursor)) {
            ++cursor;
        }
        const char *start = cursor;
        while (*cursor != '\0' && *cursor != delimiter) {
            ++cursor;
        }
        const char *end = cursor;
        while (end > start && isspace((unsigned char)end[-1])) {
            --end;
        }
        if ((size_t)(end - start) == token_len &&
            strncmp(start, token, token_len) == 0) {
            return 1;
        }
    }
    return 0;
}

static int arc_text_list_allows_int_pin(const char *text,
                                        const char *pin_name,
                                        int value)
{
    const char *cursor = text;
    char prefix[64];
    size_t prefix_len;
    int saw_pin = 0;
    if (text == NULL || pin_name == NULL || pin_name[0] == '\0') {
        return 1;
    }
    snprintf(prefix, sizeof(prefix), "pin:%s=", pin_name);
    prefix_len = strlen(prefix);
    while (*cursor != '\0') {
        while (*cursor == ',' || isspace((unsigned char)*cursor)) {
            ++cursor;
        }
        const char *start = cursor;
        while (*cursor != '\0' && *cursor != ',') {
            ++cursor;
        }
        const char *end = cursor;
        while (end > start && isspace((unsigned char)end[-1])) {
            --end;
        }
        if ((size_t)(end - start) > prefix_len &&
            strncmp(start, prefix, prefix_len) == 0) {
            char *parse_end = NULL;
            long pinned = strtol(start + prefix_len, &parse_end, 10);
            saw_pin = 1;
            if (parse_end == end && pinned == value) {
                return 1;
            }
        }
    }
    return !saw_pin;
}

static int arc_text_list_contains_int_pin(const char *text,
                                          const char *pin_name,
                                          int value)
{
    const char *cursor = text;
    char prefix[64];
    size_t prefix_len;
    if (text == NULL || pin_name == NULL || pin_name[0] == '\0') {
        return 0;
    }
    snprintf(prefix, sizeof(prefix), "pin:%s=", pin_name);
    prefix_len = strlen(prefix);
    while (*cursor != '\0') {
        while (*cursor == ',' || isspace((unsigned char)*cursor)) {
            ++cursor;
        }
        const char *start = cursor;
        while (*cursor != '\0' && *cursor != ',') {
            ++cursor;
        }
        const char *end = cursor;
        while (end > start && isspace((unsigned char)end[-1])) {
            --end;
        }
        if ((size_t)(end - start) > prefix_len &&
            strncmp(start, prefix, prefix_len) == 0) {
            char *parse_end = NULL;
            long pinned = strtol(start + prefix_len, &parse_end, 10);
            if (parse_end == end && pinned == value) {
                return 1;
            }
        }
    }
    return 0;
}

static int arc_any_hint_matches_values(const char *hints,
                                       const char *values,
                                       char value_delimiter)
{
    const char *cursor = hints;
    char token[128];
    if (hints == NULL || values == NULL) {
        return 0;
    }
    while (*cursor != '\0') {
        while (*cursor == ',' || isspace((unsigned char)*cursor)) {
            ++cursor;
        }
        const char *start = cursor;
        while (*cursor != '\0' && *cursor != ',') {
            ++cursor;
        }
        const char *end = cursor;
        while (end > start && isspace((unsigned char)end[-1])) {
            --end;
        }
        size_t len = (size_t)(end - start);
        if (len > 0 && len < sizeof(token)) {
            memcpy(token, start, len);
            token[len] = '\0';
            if (arc_text_list_contains_token(values, token, value_delimiter)) {
                return 1;
            }
        }
    }
    return 0;
}

static int arc_sketch_probe_input_allows_op(const ARCSketchProbeInput *input, int op)
{
    const ARCVisualOpMetadata *metadata;
    if (input == NULL || input->op_hints[0] == '\0' ||
        strcmp(input->op_hints, "*") == 0) {
        return 1;
    }
    if (arc_text_list_contains_int_pin(input->op_hints, "op", op)) {
        return 1;
    }
    if (arc_text_list_contains_token(input->op_hints, arc_sketch_op_name(op), ',')) {
        return 1;
    }
    metadata = arc_visual_op_metadata_for_op(op);
    if (metadata == NULL) {
        return 0;
    }
    return arc_text_list_contains_token(input->op_hints,
                                        metadata->composition_kind,
                                        ',') ||
           arc_text_list_contains_token(input->op_hints,
                                        metadata->normalization_frame,
                                        ',') ||
           arc_text_list_contains_token(input->op_hints,
                                        metadata->typed_stages,
                                        ',') ||
           arc_text_list_contains_token(input->op_hints,
                                        metadata->primitive_contract,
                                        ',') ||
           arc_any_hint_matches_values(input->op_hints,
                                       metadata->typed_stages,
                                       '|') ||
           arc_any_hint_matches_values(input->op_hints,
                                       metadata->primitive_contract,
                                       ',');
}

static int arc_sketch_spec_allows_pins(const ARCSketchProbeInput *input,
                                       const ARCSketchProgramSpec *spec)
{
    const char *hints;
    if (input == NULL || spec == NULL) {
        return 1;
    }
    hints = input->op_hints;
    if (hints[0] == '\0' || strcmp(hints, "*") == 0) {
        return 1;
    }
    return arc_text_list_allows_int_pin(hints, "op", spec->op) &&
           arc_text_list_allows_int_pin(hints, "selector", spec->selector) &&
           arc_text_list_allows_int_pin(hints, "background", spec->background) &&
           arc_text_list_allows_int_pin(hints, "color", spec->color) &&
           arc_text_list_allows_int_pin(hints, "output_color", spec->output_color) &&
           arc_text_list_allows_int_pin(hints, "x_anchor", spec->x_anchor) &&
           arc_text_list_allows_int_pin(hints, "y_anchor", spec->y_anchor) &&
           arc_text_list_allows_int_pin(hints, "x_offset", spec->x_offset) &&
           arc_text_list_allows_int_pin(hints, "y_offset", spec->y_offset) &&
           arc_text_list_allows_int_pin(hints, "width", (int)spec->width) &&
           arc_text_list_allows_int_pin(hints, "height", (int)spec->height) &&
           arc_text_list_allows_int_pin(hints, "period", (int)spec->period);
}

static uint64_t arc_sketch_probe_spec_hash(const char *family,
                                           const char *task_id,
                                           const ARCSketchProgramSpec *spec)
{
    uint64_t hash = UINT64_C(1469598103934665603);
    if (family != NULL) {
        for (const unsigned char *p = (const unsigned char *)family; *p != '\0'; ++p) {
            hash = arc_hash_mix(hash, *p);
        }
    }
    if (task_id != NULL) {
        for (const unsigned char *p = (const unsigned char *)task_id; *p != '\0'; ++p) {
            hash = arc_hash_mix(hash, *p);
        }
    }
    if (spec != NULL) {
        hash = arc_hash_mix(hash, (uint64_t)(uint32_t)spec->selector);
        hash = arc_hash_mix(hash, (uint64_t)(uint32_t)spec->op);
        hash = arc_hash_mix(hash, spec->background);
        hash = arc_hash_mix(hash, spec->color);
        hash = arc_hash_mix(hash, spec->output_color);
        hash = arc_hash_mix(hash, (uint64_t)(uint32_t)spec->x_anchor);
        hash = arc_hash_mix(hash, (uint64_t)(uint32_t)spec->y_anchor);
        hash = arc_hash_mix(hash, (uint64_t)(uint32_t)spec->x_offset);
        hash = arc_hash_mix(hash, (uint64_t)(uint32_t)spec->y_offset);
        hash = arc_hash_mix(hash, (uint64_t)spec->width);
        hash = arc_hash_mix(hash, (uint64_t)spec->height);
        hash = arc_hash_mix(hash, (uint64_t)spec->period);
    }
    return hash;
}

static void arc_public_task_path(const char *public_arc_dir,
                                 const char *split,
                                 const char *task_id,
                                 char *out,
                                 size_t out_size)
{
    const char *split_dir = split;
    if (strcmp(split, "public_training") == 0 ||
        strcmp(split, "training") == 0 ||
        strcmp(split, "train") == 0) {
        split_dir = "training";
    } else if (strcmp(split, "public_evaluation") == 0 ||
               strcmp(split, "evaluation") == 0 ||
               strcmp(split, "eval") == 0) {
        split_dir = "evaluation";
    }
    snprintf(out, out_size, "%s/%s/%s.json", public_arc_dir, split_dir, task_id);
}

static ARCVisualPipelineAttemptLog *arc_active_visual_pipeline_attempt_log = NULL;

static int arc_visual_pipeline_attempt_log_append(
    ARCVisualPipelineAttemptLog *log,
    const ARCSketchProbeInput *input,
    const char *task_path,
    const ARCTask *task,
    const ARCVisualPipelineSpec *pipeline,
    const ARCSketchProgramSpec *spec,
    const ARCTrainMatchTelemetry *telemetry,
    size_t attempt_index,
    int accepted,
    ARCSketchProbeReason reason);

static void arc_sketch_probe_log_spec_attempt(const ARCSketchProbeInput *input,
                                              const char *task_path,
                                              const ARCTask *task,
                                              const ARCSketchProgramSpec *spec,
                                              const ARCTrainMatchTelemetry *telemetry,
                                              const ARCSketchProbeSummary *summary,
                                              int accepted,
                                              ARCSketchProbeReason reason)
{
    ARCVisualPipelineSpec pipeline;
    if (arc_active_visual_pipeline_attempt_log == NULL ||
        input == NULL || task_path == NULL || spec == NULL || summary == NULL ||
        !arc_visual_pipeline_from_sketch_spec(spec, &pipeline)) {
        return;
    }
    (void)arc_visual_pipeline_attempt_log_append(arc_active_visual_pipeline_attempt_log,
                                                 input,
                                                 task_path,
                                                 task,
                                                 &pipeline,
                                                 spec,
                                                 telemetry,
                                                 summary->attempted,
                                                 accepted,
                                                 reason);
}

static int arc_sketch_probe_append_if_exact_with_reason(
    const ARCSketchProbeInput *input,
    const char *task_path,
    const ARCTask *task,
    const ARCSketchProgramSpec *spec,
    ARCSketchProbeRecord *records,
    size_t *record_count,
    ARCSketchProgramSpec *emitted_specs,
    size_t *emitted_count,
    size_t sketches_per_task,
    ARCSketchProbeSummary *summary,
    ARCSketchProbeReason *reason_out)
{
    ARCTransformCandidate candidate;
    size_t checked = 0;
    size_t unsupported = 0;
    ARCTrainMatchTelemetry telemetry;
    memset(&telemetry, 0, sizeof(telemetry));
    if (reason_out != NULL) {
        *reason_out = ARC_SKETCH_PROBE_REASON_INVALID_ARGUMENT;
    }
    if (input == NULL || task == NULL || spec == NULL || records == NULL ||
        record_count == NULL || emitted_specs == NULL || emitted_count == NULL ||
        summary == NULL) {
        return 0;
    }
    ++summary->attempted;
    if (!arc_sketch_spec_allows_pins(input, spec)) {
        ++summary->pruned_infeasible;
        return 0;
    }
    if (*emitted_count >= sketches_per_task) {
        ++summary->cap_hit;
        if (reason_out != NULL) {
            *reason_out = ARC_SKETCH_PROBE_REASON_CAP_HIT;
        }
        return 0;
    }
    for (size_t i = 0; i < *emitted_count; ++i) {
        if (arc_sketch_probe_spec_equal(&emitted_specs[i], spec)) {
            ++summary->duplicate;
            if (reason_out != NULL) {
                *reason_out = ARC_SKETCH_PROBE_REASON_DUPLICATE;
            }
            arc_sketch_probe_log_spec_attempt(input,
                                              task_path,
                                              task,
                                              spec,
                                              NULL,
                                              summary,
                                              0,
                                              ARC_SKETCH_PROBE_REASON_DUPLICATE);
            return 0;
        }
    }
    if (!arc_candidate_from_sketch_program_values(&candidate, spec)) {
        telemetry.train_pairs = task->train_count;
        telemetry.unsupported_pairs = task->train_count;
        telemetry.apply_failed_pairs = task->train_count;
        telemetry.first_failing_pair = task->train_count == 0 ? SIZE_MAX : 0;
        telemetry.verifier_score = 0.0;
        ++summary->unsupported_shape;
        if (reason_out != NULL) {
            *reason_out = ARC_SKETCH_PROBE_REASON_UNSUPPORTED_SHAPE;
        }
        arc_sketch_probe_log_spec_attempt(input,
                                          task_path,
                                          task,
                                          spec,
                                          &telemetry,
                                          summary,
                                          0,
                                          ARC_SKETCH_PROBE_REASON_UNSUPPORTED_SHAPE);
        return 0;
    }
    if (!arc_candidate_train_match_telemetry(task, &candidate, &telemetry)) {
        checked = telemetry.checked_cells;
        unsupported = telemetry.unsupported_pairs;
        if (unsupported > 0 || checked == 0) {
            ++summary->unsupported_shape;
            if (reason_out != NULL) {
                *reason_out = ARC_SKETCH_PROBE_REASON_UNSUPPORTED_SHAPE;
            }
            arc_sketch_probe_log_spec_attempt(input,
                                              task_path,
                                              task,
                                              spec,
                                              &telemetry,
                                              summary,
                                              0,
                                              ARC_SKETCH_PROBE_REASON_UNSUPPORTED_SHAPE);
        } else {
            ++summary->no_train_match;
            if (reason_out != NULL) {
                *reason_out = ARC_SKETCH_PROBE_REASON_NO_TRAIN_MATCH;
            }
            arc_sketch_probe_log_spec_attempt(input,
                                              task_path,
                                              task,
                                              spec,
                                              &telemetry,
                                              summary,
                                              0,
                                              ARC_SKETCH_PROBE_REASON_NO_TRAIN_MATCH);
        }
        return 0;
    }
    if (*record_count >= ARC_SKETCH_PROBE_RECORD_CAPACITY) {
        ++summary->cap_hit;
        if (reason_out != NULL) {
            *reason_out = ARC_SKETCH_PROBE_REASON_CAP_HIT;
        }
        arc_sketch_probe_log_spec_attempt(input,
                                          task_path,
                                          task,
                                          spec,
                                          NULL,
                                          summary,
                                          0,
                                          ARC_SKETCH_PROBE_REASON_CAP_HIT);
        return 0;
    }
    uint64_t hash = arc_sketch_probe_spec_hash(input->family, input->task_id, spec);
    ARCSketchProbeRecord *record = &records[*record_count];
    memset(record, 0, sizeof(*record));
    snprintf(record->family, sizeof(record->family), "%s", input->family);
    snprintf(record->task_id, sizeof(record->task_id), "%s", input->task_id);
    snprintf(record->split, sizeof(record->split), "%s", input->split);
    snprintf(record->task_path, sizeof(record->task_path), "%s", task_path);
    snprintf(record->coverage_frontier_id,
             sizeof(record->coverage_frontier_id),
             "%s",
             input->coverage_frontier_id);
    snprintf(record->parent_attempt_id,
             sizeof(record->parent_attempt_id),
             "%s",
             input->parent_attempt_id);
    snprintf(record->repair_operator,
             sizeof(record->repair_operator),
             "%s",
             input->repair_operator);
    snprintf(record->dominant_residual_class,
             sizeof(record->dominant_residual_class),
             "%s",
             input->dominant_residual_class);
    snprintf(record->dominant_failure_stage,
             sizeof(record->dominant_failure_stage),
             "%s",
             input->dominant_failure_stage);
    snprintf(record->repair_strategy,
             sizeof(record->repair_strategy),
             "%s",
             input->repair_strategy);
    snprintf(record->library_id,
             sizeof(record->library_id),
             "ngd:%s:%s:%016llx",
             input->family,
             input->task_id,
             (unsigned long long)hash);
    record->spec = *spec;
    record->support_tasks = input->support_tasks;
    record->proposal_count = input->proposal_count == 0 ? 1u : input->proposal_count;
    record->utility_mean = input->utility_mean;
    record->sketch_index = *emitted_count + 1u;
    emitted_specs[*emitted_count] = *spec;
    ++*emitted_count;
    ++*record_count;
    ++summary->emitted;
    if (reason_out != NULL) {
        *reason_out = ARC_SKETCH_PROBE_REASON_ACCEPTED;
    }
    arc_sketch_probe_log_spec_attempt(input,
                                      task_path,
                                      task,
                                      spec,
                                      &telemetry,
                                      summary,
                                      1,
                                      ARC_SKETCH_PROBE_REASON_ACCEPTED);
    return 1;
}

static int arc_sketch_probe_append_if_exact(const ARCSketchProbeInput *input,
                                            const char *task_path,
                                            const ARCTask *task,
                                            const ARCSketchProgramSpec *spec,
                                            ARCSketchProbeRecord *records,
                                            size_t *record_count,
                                            ARCSketchProgramSpec *emitted_specs,
                                            size_t *emitted_count,
                                            size_t sketches_per_task,
                                            ARCSketchProbeSummary *summary)
{
    return arc_sketch_probe_append_if_exact_with_reason(input,
                                                       task_path,
                                                       task,
                                                       spec,
                                                       records,
                                                       record_count,
                                                       emitted_specs,
                                                       emitted_count,
                                                       sketches_per_task,
                                                       summary,
                                                       NULL);
}

static int arc_visual_pipeline_attempt_log_append(
    ARCVisualPipelineAttemptLog *log,
    const ARCSketchProbeInput *input,
    const char *task_path,
    const ARCTask *task,
    const ARCVisualPipelineSpec *pipeline,
    const ARCSketchProgramSpec *spec,
    const ARCTrainMatchTelemetry *telemetry,
    size_t attempt_index,
    int accepted,
    ARCSketchProbeReason reason)
{
    if (log == NULL || input == NULL || task_path == NULL ||
        pipeline == NULL || spec == NULL) {
        return 1;
    }
    if (log->count >= log->capacity) {
        size_t next_capacity = log->capacity == 0 ? 1024u : log->capacity * 2u;
        ARCVisualPipelineAttempt *next =
            realloc(log->items, next_capacity * sizeof(*next));
        if (next == NULL) {
            return 0;
        }
        log->items = next;
        log->capacity = next_capacity;
    }
    ARCVisualPipelineAttempt *attempt = &log->items[log->count++];
    memset(attempt, 0, sizeof(*attempt));
    snprintf(attempt->family, sizeof(attempt->family), "%s", input->family);
    snprintf(attempt->task_id, sizeof(attempt->task_id), "%s", input->task_id);
    snprintf(attempt->split, sizeof(attempt->split), "%s", input->split);
    snprintf(attempt->task_path, sizeof(attempt->task_path), "%s", task_path);
    snprintf(attempt->coverage_frontier_id,
             sizeof(attempt->coverage_frontier_id),
             "%s",
             input->coverage_frontier_id);
    snprintf(attempt->parent_attempt_id,
             sizeof(attempt->parent_attempt_id),
             "%s",
             input->parent_attempt_id);
    snprintf(attempt->repair_operator,
             sizeof(attempt->repair_operator),
             "%s",
             input->repair_operator);
    snprintf(attempt->dominant_residual_class,
             sizeof(attempt->dominant_residual_class),
             "%s",
             input->dominant_residual_class);
    snprintf(attempt->dominant_failure_stage,
             sizeof(attempt->dominant_failure_stage),
             "%s",
             input->dominant_failure_stage);
    snprintf(attempt->repair_strategy,
             sizeof(attempt->repair_strategy),
             "%s",
             input->repair_strategy);
    attempt->pipeline = *pipeline;
    attempt->spec = *spec;
    const ARCVisualOpMetadata *metadata = arc_visual_op_metadata_for_op(spec->op);
    if (metadata != NULL) {
        attempt->pipeline.composition_kind = metadata->composition_kind;
        attempt->pipeline.normalization_frame = metadata->normalization_frame;
        attempt->pipeline.typed_stages = metadata->typed_stages;
        attempt->pipeline.primitive_contract = metadata->primitive_contract;
    }
    attempt->attempt_index = attempt_index;
    attempt->support_tasks = input->support_tasks;
    attempt->proposal_count = input->proposal_count == 0 ? 1u : input->proposal_count;
    attempt->utility_mean = input->utility_mean;
    attempt->accepted = accepted ? 1 : 0;
    attempt->reason = reason;
    if (telemetry != NULL) {
        attempt->telemetry = *telemetry;
    }
    snprintf(attempt->failure_stage,
             sizeof(attempt->failure_stage),
             "%s",
             arc_visual_pipeline_failure_stage(attempt));
    snprintf(attempt->failure_detail,
             sizeof(attempt->failure_detail),
             "%s",
             arc_visual_pipeline_failure_detail(attempt));
    arc_visual_pipeline_attempt_fill_task_features(attempt, task);
    return 1;
}

static int arc_sketch_probe_append_visual_pipeline_if_exact(
    const ARCSketchProbeInput *input,
    const char *task_path,
    const ARCTask *task,
    const ARCVisualPipelineSpec *pipeline,
    ARCSketchProbeRecord *records,
    size_t *record_count,
    ARCSketchProgramSpec *emitted_specs,
    size_t *emitted_count,
    size_t sketches_per_task,
    ARCSketchProbeSummary *summary,
    ARCVisualPipelineAttemptLog *attempt_log)
{
    ARCSketchProgramSpec spec;
    ARCSketchProbeReason reason = ARC_SKETCH_PROBE_REASON_INVALID_ARGUMENT;
    int accepted = 0;
    if (summary == NULL) {
        return 0;
    }
    if (emitted_count != NULL && *emitted_count >= sketches_per_task) {
        ++summary->cap_hit;
        return 0;
    }
    if (!arc_visual_pipeline_to_sketch_spec(pipeline, &spec)) {
        if (attempt_log != NULL && input != NULL && task_path != NULL &&
            pipeline != NULL) {
            ARCSketchProgramSpec empty_spec;
            memset(&empty_spec, 0, sizeof(empty_spec));
            arc_visual_pipeline_attempt_log_append(attempt_log,
                                                   input,
                                                   task_path,
                                                   task,
                                                   pipeline,
                                                   &empty_spec,
                                                   NULL,
                                                   summary->attempted,
                                                   0,
                                                   ARC_SKETCH_PROBE_REASON_UNSUPPORTED_SHAPE);
        }
        ++summary->unsupported_shape;
        return 0;
    }
    if (spec.op == ARC_SKETCH_OP_VISUAL_COMBINATOR &&
        spec.x_anchor == 3 &&
        !arc_visual_component_mask_spec_feasible_for_task(task, &spec)) {
        ++summary->pruned_infeasible;
        return 0;
    }
    accepted = arc_sketch_probe_append_if_exact_with_reason(input,
                                                           task_path,
                                                           task,
                                                           &spec,
                                                           records,
                                                           record_count,
                                                           emitted_specs,
                                                           emitted_count,
                                                           sketches_per_task,
                                                           summary,
                                                           &reason);
    (void)attempt_log;
    (void)reason;
    return accepted;
}

static int arc_sketch_spec_matches_first_train_pair(const ARCTask *task,
