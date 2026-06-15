static int arc_color_added_bbox(const ARCGrid *input,
                                const ARCGrid *output,
                                uint8_t output_color,
                                size_t *min_x,
                                size_t *min_y,
                                size_t *max_x,
                                size_t *max_y)
{
    int found = 0;
    if (input == NULL || output == NULL || min_x == NULL ||
        min_y == NULL || max_x == NULL || max_y == NULL ||
        !arc_grid_same_shape(input, output)) {
        return 0;
    }
    *min_x = input->width;
    *min_y = input->height;
    *max_x = 0;
    *max_y = 0;
    for (size_t y = 0; y < output->height; ++y) {
        for (size_t x = 0; x < output->width; ++x) {
            size_t cell = y * output->width + x;
            if (output->cells[cell] != output_color ||
                input->cells[cell] == output_color) {
                continue;
            }
            if (!found || x < *min_x) {
                *min_x = x;
            }
            if (!found || y < *min_y) {
                *min_y = y;
            }
            if (!found || x > *max_x) {
                *max_x = x;
            }
            if (!found || y > *max_y) {
                *max_y = y;
            }
            found = 1;
        }
    }
    return found;
}

typedef struct {
    size_t attempted_specs;
    size_t accepted_specs;
    size_t unsupported_shape;
    size_t no_train_match;
    size_t duplicate;
    size_t cap_hit;
    size_t all_color_attempts;
    size_t all_color_accepted;
    size_t single_object_attempts;
    size_t single_object_accepted;
    int have_best_rejected;
    double best_rejected_verifier_score;
    ARCSketchProbeReason best_rejected_reason;
    ARCTrainMatchTelemetry best_rejected_telemetry;
    ARCSketchProgramSpec best_rejected_spec;
} ARCObjectTranslateMineStats;

static void arc_object_translate_stats_record_reason(
    ARCObjectTranslateMineStats *stats,
    ARCSketchProbeReason reason)
{
    if (stats == NULL) {
        return;
    }
    switch (reason) {
    case ARC_SKETCH_PROBE_REASON_UNSUPPORTED_SHAPE:
        ++stats->unsupported_shape;
        break;
    case ARC_SKETCH_PROBE_REASON_NO_TRAIN_MATCH:
        ++stats->no_train_match;
        break;
    case ARC_SKETCH_PROBE_REASON_DUPLICATE:
        ++stats->duplicate;
        break;
    case ARC_SKETCH_PROBE_REASON_CAP_HIT:
        ++stats->cap_hit;
        break;
    case ARC_SKETCH_PROBE_REASON_ACCEPTED:
    case ARC_SKETCH_PROBE_REASON_INVALID_ARGUMENT:
    default:
        break;
    }
}

static void arc_object_translate_stats_record_rejected(
    ARCObjectTranslateMineStats *stats,
    const ARCSketchProgramSpec *spec,
    const ARCTrainMatchTelemetry *telemetry,
    ARCSketchProbeReason reason)
{
    if (stats == NULL || spec == NULL || telemetry == NULL) {
        return;
    }
    if (!stats->have_best_rejected ||
        telemetry->verifier_score > stats->best_rejected_verifier_score) {
        stats->have_best_rejected = 1;
        stats->best_rejected_verifier_score = telemetry->verifier_score;
        stats->best_rejected_reason = reason;
        stats->best_rejected_telemetry = *telemetry;
        stats->best_rejected_spec = *spec;
    }
}

static int arc_object_translate_probe_spec(
    const ARCSketchProbeInput *input,
    const char *task_path,
    const ARCTask *task,
    const ARCSketchProgramSpec *spec,
    int all_color_mode,
    ARCSketchProbeRecord *records,
    size_t *record_count,
    ARCSketchProgramSpec *emitted_specs,
    size_t *emitted_count,
    size_t sketches_per_task,
    ARCSketchProbeSummary *summary,
    ARCObjectTranslateMineStats *stats)
{
    ARCTransformCandidate candidate;
    ARCTrainMatchTelemetry telemetry;
    ARCSketchProbeReason reason = ARC_SKETCH_PROBE_REASON_INVALID_ARGUMENT;
    int train_match = 0;
    memset(&telemetry, 0, sizeof(telemetry));
    if (stats != NULL) {
        ++stats->attempted_specs;
        if (all_color_mode) {
            ++stats->all_color_attempts;
        } else {
            ++stats->single_object_attempts;
        }
    }
    if (input == NULL || task == NULL || spec == NULL) {
        return 0;
    }
    if (!arc_candidate_from_sketch_program_values(&candidate, spec)) {
        telemetry.train_pairs = task->train_count;
        telemetry.unsupported_pairs = task->train_count;
        telemetry.apply_failed_pairs = task->train_count;
        telemetry.first_failing_pair = task->train_count == 0 ? SIZE_MAX : 0;
        telemetry.verifier_score = 0.0;
        reason = ARC_SKETCH_PROBE_REASON_UNSUPPORTED_SHAPE;
        arc_object_translate_stats_record_reason(stats, reason);
        arc_object_translate_stats_record_rejected(stats, spec, &telemetry, reason);
        return 0;
    }
    train_match = arc_candidate_train_match_telemetry(task, &candidate, &telemetry);
    if (!train_match) {
        reason = telemetry.unsupported_pairs > 0 || telemetry.checked_cells == 0
                     ? ARC_SKETCH_PROBE_REASON_UNSUPPORTED_SHAPE
                     : ARC_SKETCH_PROBE_REASON_NO_TRAIN_MATCH;
        arc_object_translate_stats_record_reason(stats, reason);
        arc_object_translate_stats_record_rejected(stats, spec, &telemetry, reason);
        return 0;
    }
    if (!arc_sketch_probe_append_if_exact_with_reason(input,
                                                      task_path,
                                                      task,
                                                      spec,
                                                      records,
                                                      record_count,
                                                      emitted_specs,
                                                      emitted_count,
                                                      sketches_per_task,
                                                      summary,
                                                      &reason)) {
        arc_object_translate_stats_record_reason(stats, reason);
        return 0;
    }
    if (stats != NULL) {
        ++stats->accepted_specs;
        if (all_color_mode) {
            ++stats->all_color_accepted;
        } else {
            ++stats->single_object_accepted;
        }
    }
    return 1;
}

static void arc_print_object_translate_miner_trace(
    const ARCSketchProbeInput *input,
    const ARCTask *task,
    const ARCObjectTranslateMineStats *stats)
{
    if (input == NULL || task == NULL || stats == NULL ||
        strcmp(input->family, "arc.generative_prior") != 0 ||
        stats->attempted_specs == 0) {
        return;
    }
    fputs("{\"schema\":\"crlp.arc_object_translate_miner.trace.v1\","
          "\"report_version\":1,\"task\":\"arc_agi\",\"task_instance\":",
          stdout);
    nn_write_json_string(stdout, task->id);
    fprintf(stdout,
            ",\"attempted_specs\":%zu,\"accepted_specs\":%zu,"
            "\"unsupported_shape\":%zu,\"no_train_match\":%zu,"
            "\"duplicate\":%zu,\"cap_hit\":%zu,"
            "\"all_color_attempts\":%zu,\"all_color_accepted\":%zu,"
            "\"single_object_attempts\":%zu,\"single_object_accepted\":%zu",
            stats->attempted_specs,
            stats->accepted_specs,
            stats->unsupported_shape,
            stats->no_train_match,
            stats->duplicate,
            stats->cap_hit,
            stats->all_color_attempts,
            stats->all_color_accepted,
            stats->single_object_attempts,
            stats->single_object_accepted);
    if (stats->have_best_rejected) {
        const ARCTrainMatchTelemetry *telemetry = &stats->best_rejected_telemetry;
        const ARCSketchProgramSpec *spec = &stats->best_rejected_spec;
        fprintf(stdout,
                ",\"best_rejected_reason\":");
        nn_write_json_string(stdout,
                             arc_sketch_probe_reason_name(
                                 stats->best_rejected_reason));
        fprintf(stdout,
                ",\"best_rejected_verifier_score\":%.8f,"
                "\"best_rejected_mismatches\":%zu,"
                "\"best_rejected_unsupported_pairs\":%zu,"
                "\"best_rejected_first_pair_mismatches\":%zu,"
                "\"best_rejected_first_mismatch_min_x\":%zu,"
                "\"best_rejected_first_mismatch_min_y\":%zu,"
                "\"best_rejected_first_mismatch_max_x\":%zu,"
                "\"best_rejected_first_mismatch_max_y\":%zu,"
                "\"best_rejected_selector\":%d,"
                "\"best_rejected_color\":%u,"
                "\"best_rejected_output_color\":%u,"
                "\"best_rejected_dx\":%d,"
                "\"best_rejected_dy\":%d,"
                "\"best_rejected_render_mode\":%zu",
                telemetry->verifier_score,
                telemetry->mismatches,
                telemetry->unsupported_pairs,
                telemetry->first_pair_mismatches,
                telemetry->first_mismatch_min_x == SIZE_MAX
                    ? 0u
                    : telemetry->first_mismatch_min_x,
                telemetry->first_mismatch_min_y == SIZE_MAX
                    ? 0u
                    : telemetry->first_mismatch_min_y,
                telemetry->first_mismatch_max_x,
                telemetry->first_mismatch_max_y,
                spec->x_anchor,
                (unsigned)spec->color,
                (unsigned)spec->output_color,
                spec->x_offset,
                spec->y_offset,
                spec->period);
    }
    fputs("}\n", stdout);
}

static void arc_mine_object_translate_sketches(
    const ARCSketchProbeInput *input,
    const char *task_path,
    const ARCTask *task,
    ARCSketchProbeRecord *records,
    size_t *record_count,
    ARCSketchProgramSpec *emitted_specs,
    size_t *emitted_count,
    size_t sketches_per_task,
    ARCSketchProbeSummary *summary)
{
    ARCObjectTranslateMineStats stats;
    static const uint8_t selectors[] = {
        ARC_OBJECT_SELECTOR_UNIQUE_COLOR,
        ARC_OBJECT_SELECTOR_LARGEST_COLOR,
        ARC_OBJECT_SELECTOR_SMALLEST_COLOR,
        ARC_OBJECT_SELECTOR_INTERIOR_COLOR,
        ARC_OBJECT_SELECTOR_TOUCHING_BORDER_COLOR,
    };
    if (task == NULL || task->train_count == 0 || !task->train[0].has_output ||
        !arc_grid_same_shape(&task->train[0].input, &task->train[0].output)) {
        return;
    }
    memset(&stats, 0, sizeof(stats));
    const ARCGrid *first_input = &task->train[0].input;
    const ARCGrid *first_output = &task->train[0].output;
    uint8_t backgrounds[ARC_COLOR_COUNT];
    size_t background_count =
        arc_task_relation_backgrounds(task,
                                      backgrounds,
                                      sizeof(backgrounds) / sizeof(backgrounds[0]));
    arc_add_unique_color(backgrounds,
                         sizeof(backgrounds) / sizeof(backgrounds[0]),
                         &background_count,
                         arc_grid_dominant_color(first_input));
    arc_add_unique_color(backgrounds,
                         sizeof(backgrounds) / sizeof(backgrounds[0]),
                         &background_count,
                         0);
    for (size_t background_index = 0;
         background_index < background_count;
         ++background_index) {
        uint8_t background = backgrounds[background_index];
        ARCScene input_scene;
        ARCScene output_scene;
        if (!arc_extract_scene(first_input, background, &input_scene) ||
            !arc_extract_scene(first_output, background, &output_scene)) {
            continue;
        }
        uint8_t input_colors[ARC_COLOR_COUNT];
        uint8_t output_colors[ARC_COLOR_COUNT];
        size_t input_color_count =
            arc_grid_nonzero_colors(first_input, input_colors, ARC_COLOR_COUNT);
        size_t output_color_count =
            arc_grid_nonzero_colors(first_output, output_colors, ARC_COLOR_COUNT);
        for (size_t input_color_index = 0;
             input_color_index < input_color_count;
             ++input_color_index) {
            uint8_t input_color = input_colors[input_color_index];
            size_t input_min_x = 0;
            size_t input_min_y = 0;
            size_t input_max_x = 0;
            size_t input_max_y = 0;
            if (input_color == background ||
                !arc_grid_color_bbox(first_input,
                                     input_color,
                                     &input_min_x,
                                     &input_min_y,
                                     &input_max_x,
                                     &input_max_y)) {
                continue;
            }
            arc_add_unique_color(output_colors,
                                 sizeof(output_colors) / sizeof(output_colors[0]),
                                 &output_color_count,
                                 input_color);
            for (size_t output_color_index = 0;
                 output_color_index < output_color_count;
                 ++output_color_index) {
                uint8_t output_color = output_colors[output_color_index];
                if (output_color == background) {
                    continue;
                }
                for (size_t render_mode = 0;
                     render_mode < ARC_OBJECT_TRANSLATE_RENDER_KIND_COUNT;
                     ++render_mode) {
                    size_t output_min_x = 0;
                    size_t output_min_y = 0;
                    size_t output_max_x = 0;
                    size_t output_max_y = 0;
                    int have_output_bbox =
                        render_mode == ARC_OBJECT_TRANSLATE_RENDER_COPY
                            ? arc_color_added_bbox(first_input,
                                                   first_output,
                                                   output_color,
                                                   &output_min_x,
                                                   &output_min_y,
                                                   &output_max_x,
                                                   &output_max_y)
                            : arc_grid_color_bbox(first_output,
                                                  output_color,
                                                  &output_min_x,
                                                  &output_min_y,
                                                  &output_max_x,
                                                  &output_max_y);
                    if (!have_output_bbox) {
                        continue;
                    }
                    if ((output_max_x - output_min_x) !=
                            (input_max_x - input_min_x) ||
                        (output_max_y - output_min_y) !=
                            (input_max_y - input_min_y)) {
                        continue;
                    }
                    int dx = (int)output_min_x - (int)input_min_x;
                    int dy = (int)output_min_y - (int)input_min_y;
                    if (dx == 0 && dy == 0 && output_color == input_color) {
                        continue;
                    }
                    ARCSketchProgramSpec spec;
                    memset(&spec, 0, sizeof(spec));
                    spec.selector = ARC_SKETCH_SELECTOR_FOREGROUND_BBOX_PANEL;
                    spec.op = ARC_SKETCH_OP_OBJECT_TRANSLATE;
                    spec.background = background;
                    spec.color = input_color;
                    spec.output_color = output_color;
                    spec.x_anchor = ARC_OBJECT_SELECTOR_ALL_COLOR;
                    spec.y_anchor = 0;
                    spec.x_offset = dx;
                    spec.y_offset = dy;
                    spec.period = render_mode;
                    arc_object_translate_probe_spec(input,
                                                    task_path,
                                                    task,
                                                    &spec,
                                                    1,
                                                    records,
                                                    record_count,
                                                    emitted_specs,
                                                    emitted_count,
                                                    sketches_per_task,
                                                    summary,
                                                    &stats);
                }
            }
        }
        for (size_t input_index = 0;
             input_index < input_scene.object_count;
             ++input_index) {
            const ARCObject *input_object = &input_scene.objects[input_index];
            if (input_object->color == background) {
                continue;
            }
            for (size_t selector_index = 0;
                 selector_index < sizeof(selectors) / sizeof(selectors[0]);
                 ++selector_index) {
                uint8_t selector = selectors[selector_index];
                ARCObject selected;
                if (!arc_select_scene_object(&input_scene,
                                             selector,
                                             input_object->color,
                                             0,
                                             &selected) ||
                    selected.id != input_object->id) {
                    continue;
                }
                for (size_t output_index = 0;
                     output_index < output_scene.object_count;
                     ++output_index) {
                    const ARCObject *output_object =
                        &output_scene.objects[output_index];
                    if (output_object->color == background ||
                        !arc_objects_same_shape(input_object, output_object)) {
                        continue;
                    }
                    int dx = (int)output_object->min_x - (int)input_object->min_x;
                    int dy = (int)output_object->min_y - (int)input_object->min_y;
                    if (dx == 0 && dy == 0 &&
                        output_object->color == input_object->color) {
                        continue;
                    }
                    for (size_t render_mode = 0;
                         render_mode < ARC_OBJECT_TRANSLATE_RENDER_KIND_COUNT;
                         ++render_mode) {
                        ARCSketchProgramSpec spec;
                        memset(&spec, 0, sizeof(spec));
                        spec.selector = ARC_SKETCH_SELECTOR_FOREGROUND_BBOX_PANEL;
                        spec.op = ARC_SKETCH_OP_OBJECT_TRANSLATE;
                        spec.background = background;
                        spec.color = input_object->color;
                        spec.output_color = output_object->color;
                        spec.x_anchor = selector;
                        spec.y_anchor = 0;
                        spec.x_offset = dx;
                        spec.y_offset = dy;
                        spec.period = render_mode;
                        arc_object_translate_probe_spec(input,
                                                        task_path,
                                                        task,
                                                        &spec,
                                                        0,
                                                        records,
                                                        record_count,
                                                        emitted_specs,
                                                        emitted_count,
                                                        sketches_per_task,
                                                        summary,
                                                        &stats);
                    }
                }
            }
        }
    }
    arc_mine_reference_recolor_mask_sketches(input,
                                             task_path,
                                             task,
                                             records,
                                             record_count,
                                             emitted_specs,
                                             emitted_count,
                                             sketches_per_task,
                                             summary);
    arc_print_object_translate_miner_trace(input, task, &stats);
}

static void arc_mine_extend_aligned_sequence_sketches(
    const ARCSketchProbeInput *input,
    const char *task_path,
    const ARCTask *task,
    ARCSketchProbeRecord *records,
    size_t *record_count,
    ARCSketchProgramSpec *emitted_specs,
    size_t *emitted_count,
    size_t sketches_per_task,
    ARCSketchProbeSummary *summary)
{
    if (task == NULL || task->train_count == 0 || !task->train[0].has_output) {
        return;
    }
    const ARCGrid *first_input = &task->train[0].input;
    const ARCGrid *first_output = &task->train[0].output;
    uint8_t backgrounds[ARC_COLOR_COUNT];
    size_t background_count =
        arc_task_relation_backgrounds(task,
                                      backgrounds,
                                      sizeof(backgrounds) / sizeof(backgrounds[0]));
    arc_add_unique_color(backgrounds,
                         sizeof(backgrounds) / sizeof(backgrounds[0]),
                         &background_count,
                         arc_grid_dominant_color(first_input));
    arc_add_unique_color(backgrounds,
                         sizeof(backgrounds) / sizeof(backgrounds[0]),
                         &background_count,
                         0);
    uint8_t input_colors[ARC_COLOR_COUNT];
    size_t input_color_count =
        arc_grid_nonzero_colors(first_input, input_colors, ARC_COLOR_COUNT);
    uint8_t output_colors[ARC_COLOR_COUNT];
    size_t output_color_count =
        arc_grid_nonzero_colors(first_output, output_colors, ARC_COLOR_COUNT);
    for (size_t background_index = 0;
         background_index < background_count;
         ++background_index) {
        uint8_t background = backgrounds[background_index];
        for (size_t color_index = 0;
             color_index < input_color_count;
             ++color_index) {
            uint8_t color = input_colors[color_index];
            if (color == background) {
                continue;
            }
            arc_add_unique_color(output_colors,
                                 sizeof(output_colors) / sizeof(output_colors[0]),
                                 &output_color_count,
                                 color);
            for (size_t output_index = 0;
                 output_index < output_color_count;
                 ++output_index) {
                uint8_t output_color = output_colors[output_index];
                if (output_color == background) {
                    continue;
                }
                for (int direction = 0;
                     direction < ARC_SEQUENCE_DIRECTION_KIND_COUNT;
                     ++direction) {
                    for (int extend_mode = 0;
                         extend_mode < ARC_SEQUENCE_EXTEND_KIND_COUNT;
                         ++extend_mode) {
                        ARCSketchProgramSpec spec;
                        memset(&spec, 0, sizeof(spec));
                        spec.selector = ARC_SKETCH_SELECTOR_LATTICE;
                        spec.op = ARC_SKETCH_OP_EXTEND_ALIGNED_SEQUENCE;
                        spec.background = background;
                        spec.color = color;
                        spec.output_color = output_color;
                        spec.x_anchor = direction;
                        spec.y_anchor = extend_mode;
                        if (!arc_sketch_spec_matches_first_train_pair(task, &spec)) {
                            continue;
                        }
                        arc_sketch_probe_append_if_exact(input,
                                                         task_path,
                                                         task,
                                                         &spec,
                                                         records,
                                                         record_count,
                                                         emitted_specs,
                                                         emitted_count,
                                                         sketches_per_task,
                                                         summary);
                    }
                }
            }
        }
    }
}

typedef struct {
    size_t input_width;
    size_t input_height;
    size_t output_width;
    size_t output_height;
    size_t shape_scale_x;
    size_t shape_scale_y;
    size_t input_component_count;
    size_t output_component_count;
} ARCGenerativePriorTaskFeatures;

typedef struct {
    int op;
    double score;
    double rejection_penalty;
    double mean_verifier_score;
    size_t matched_rows;
    size_t accepted_count;
    size_t rejected_count;
    size_t near_miss_rejected_count;
    size_t hard_rejected_count;
    size_t specificity;
    int eligible;
} ARCRuntimePriorOpScore;

static void arc_generative_prior_task_features(
    const ARCTask *task,
    ARCGenerativePriorTaskFeatures *features)
{
    if (features == NULL) {
        return;
    }
    memset(features, 0, sizeof(*features));
    if (task == NULL || task->train_count == 0 ||
        !task->train[0].has_output) {
        return;
    }
    const ARCGrid *input = &task->train[0].input;
    const ARCGrid *output = &task->train[0].output;
    features->input_width = input->width;
    features->input_height = input->height;
    features->output_width = output->width;
    features->output_height = output->height;
    if (input->width != 0 && output->width % input->width == 0) {
        features->shape_scale_x = output->width / input->width;
    }
    if (input->height != 0 && output->height % input->height == 0) {
        features->shape_scale_y = output->height / input->height;
    }
    features->input_component_count = arc_count_components(input, 0);
    features->output_component_count = arc_count_components(output, 0);
}

static int arc_generative_prior_field_matches(size_t row_value,
                                              size_t task_value,
                                              size_t *specificity)
{
    if (row_value == 0) {
        return 1;
    }
    if (row_value != task_value) {
        return 0;
    }
    if (specificity != NULL) {
        ++*specificity;
    }
    return 1;
}

static int arc_generative_prior_row_matches(
    const ARCGenerativePriorRow *row,
    const ARCGenerativePriorTaskFeatures *features,
    size_t *specificity_out)
{
    size_t specificity = 0;
    if (row == NULL || features == NULL) {
        return 0;
    }
    if (!arc_generative_prior_field_matches(row->input_width,
                                            features->input_width,
                                            &specificity) ||
        !arc_generative_prior_field_matches(row->input_height,
                                            features->input_height,
                                            &specificity) ||
        !arc_generative_prior_field_matches(row->output_width,
                                            features->output_width,
                                            &specificity) ||
        !arc_generative_prior_field_matches(row->output_height,
                                            features->output_height,
                                            &specificity) ||
        !arc_generative_prior_field_matches(row->shape_scale_x,
                                            features->shape_scale_x,
                                            &specificity) ||
        !arc_generative_prior_field_matches(row->shape_scale_y,
                                            features->shape_scale_y,
                                            &specificity) ||
        !arc_generative_prior_field_matches(row->input_component_count,
                                            features->input_component_count,
                                            &specificity) ||
        !arc_generative_prior_field_matches(row->output_component_count,
                                            features->output_component_count,
                                            &specificity)) {
        return 0;
    }
    if (specificity_out != NULL) {
        *specificity_out = specificity;
    }
    return 1;
}

static int arc_generative_prior_score_for_op(
    const ARCProgramLearnedState *learned,
    const ARCGenerativePriorTaskFeatures *features,
    int op,
    ARCRuntimePriorOpScore *out)
{
    ARCRuntimePriorOpScore score;
    double verifier_score_sum = 0.0;
    size_t best_specificity = 0;
    memset(&score, 0, sizeof(score));
    score.op = op;
    if (learned == NULL || features == NULL || learned->generative_priors == NULL) {
        if (out != NULL) {
            *out = score;
        }
        return 0;
    }
    for (size_t i = 0; i < learned->generative_prior_count; ++i) {
        const ARCGenerativePriorRow *row = &learned->generative_priors[i];
        size_t specificity = 0;
        if (row->op != op ||
            !arc_generative_prior_row_matches(row, features, &specificity)) {
            continue;
        }
        double support = log1p((double)row->accepted_count);
        double specificity_weight = 1.0 + 0.35 * (double)specificity;
        score.score += row->score * specificity_weight + 0.01 * support;
        score.rejection_penalty += row->rejection_penalty * specificity_weight;
        verifier_score_sum += row->mean_verifier_score * (double)(row->accepted_count +
                                                                  row->rejected_count);
        score.matched_rows += 1u;
        score.accepted_count += row->accepted_count;
        score.rejected_count += row->rejected_count;
        score.near_miss_rejected_count += row->near_miss_rejected_count;
        score.hard_rejected_count += row->hard_rejected_count;
        if (specificity > best_specificity) {
            best_specificity = specificity;
        }
    }
    score.specificity = best_specificity;
    if (score.accepted_count + score.rejected_count > 0) {
        score.mean_verifier_score =
            verifier_score_sum / (double)(score.accepted_count + score.rejected_count);
    }
    if (out != NULL) {
        *out = score;
    }
    return score.matched_rows != 0;
}

static int arc_runtime_prior_op_supported(int op)
{
    switch (op) {
    case ARC_SKETCH_OP_PALETTE_SCALE:
    case ARC_SKETCH_OP_OBJECT_COLOR_TILE:
    case ARC_SKETCH_OP_MARKER_COUNT_MOTIF_TILE:
    case ARC_SKETCH_OP_COMPONENT_COUNT_LINE:
    case ARC_SKETCH_OP_RECOLOR_CROP:
    case ARC_SKETCH_OP_MARKER_POLYLINE:
    case ARC_SKETCH_OP_SEPARATOR_INTERSECTION_COLOR:
    case ARC_SKETCH_OP_SEPARATOR_PANEL_MAP:
    case ARC_SKETCH_OP_VISUAL_COMBINATOR:
    case ARC_SKETCH_OP_OBJECT_TRANSLATE:
        return 1;
    default:
        return 0;
    }
}

static size_t arc_runtime_prior_min_specificity(int op)
{
    switch (op) {
    case ARC_SKETCH_OP_PALETTE_SCALE:
    case ARC_SKETCH_OP_COMPONENT_COUNT_LINE:
    case ARC_SKETCH_OP_SEPARATOR_INTERSECTION_COLOR:
    case ARC_SKETCH_OP_OBJECT_TRANSLATE:
        return 2u;
    case ARC_SKETCH_OP_SEPARATOR_PANEL_MAP:
        return 3u;
    case ARC_SKETCH_OP_OBJECT_COLOR_TILE:
    case ARC_SKETCH_OP_MARKER_COUNT_MOTIF_TILE:
    case ARC_SKETCH_OP_MARKER_POLYLINE:
    case ARC_SKETCH_OP_RECOLOR_CROP:
        return 4u;
    case ARC_SKETCH_OP_VISUAL_COMBINATOR:
        return 6u;
    default:
        return 8u;
    }
}

static int arc_runtime_prior_task_allowed(
    int op,
    const ARCGenerativePriorTaskFeatures *features)
{
    if (features == NULL) {
        return 0;
    }
    size_t input_area = features->input_width * features->input_height;
    size_t output_area = features->output_width * features->output_height;
    switch (op) {
    case ARC_SKETCH_OP_OBJECT_TRANSLATE:
        return features->input_width == features->output_width &&
               features->input_height == features->output_height &&
               input_area <= 400u && output_area <= 400u &&
               features->input_component_count <= 16u &&
               features->output_component_count <= 20u;
    case ARC_SKETCH_OP_VISUAL_COMBINATOR:
        return input_area <= 100u && output_area <= 100u &&
               features->input_component_count <= 12u &&
               features->output_component_count <= 12u;
    case ARC_SKETCH_OP_RECOLOR_CROP:
    case ARC_SKETCH_OP_MARKER_POLYLINE:
        return input_area <= 400u && output_area <= 400u;
    default:
        return 1;
    }
}

static void arc_print_generative_prior_runtime_trace(
    const ARCTask *task,
    const ARCRuntimePriorOpScore *score,
    const char *decision,
    size_t rank,
    size_t op_cap,
    size_t emitted_delta)
{
    if (task == NULL || score == NULL || decision == NULL) {
        return;
    }
    fputs("{\"schema\":\"crlp.arc_generative_prior.runtime_trace.v1\","
          "\"report_version\":1,\"task\":\"arc_agi\",\"task_instance\":",
          stdout);
    nn_write_json_string(stdout, task->id);
    fputs(",\"op_name\":", stdout);
    nn_write_json_string(stdout, arc_sketch_op_name(score->op));
    fprintf(stdout,
            ",\"op_id\":%d,\"decision\":",
            score->op);
    nn_write_json_string(stdout, decision);
    fprintf(stdout,
            ",\"rank\":%zu,\"op_cap\":%zu,\"matched_prior_rows\":%zu,"
            "\"specificity\":%zu,\"accepted_count\":%zu,\"rejected_count\":%zu,"
            "\"near_miss_rejected_count\":%zu,\"hard_rejected_count\":%zu,"
            "\"mean_verifier_score\":%.8f,\"rejection_penalty\":%.8f,"
            "\"rejection_trained_score\":%.8f,\"emitted_sketches\":%zu,"
            "\"rejection_trained\":true}\n",
            rank,
            op_cap,
            score->matched_rows,
            score->specificity,
            score->accepted_count,
            score->rejected_count,
            score->near_miss_rejected_count,
            score->hard_rejected_count,
            score->mean_verifier_score,
            score->rejection_penalty,
            score->score,
            emitted_delta);
}

static void arc_runtime_mine_prior_op(const ARCSketchProbeInput *input,
                                      const char *task_path,
                                      const ARCTask *task,
                                      int op,
                                      ARCSketchProbeRecord *records,
                                      size_t *record_count,
                                      ARCSketchProgramSpec *emitted_specs,
                                      size_t *emitted_count,
                                      size_t sketches_per_task,
                                      ARCSketchProbeSummary *summary)
{
    switch (op) {
    case ARC_SKETCH_OP_PALETTE_SCALE:
        arc_mine_palette_scale_sketches(input,
                                        task_path,
                                        task,
                                        records,
                                        record_count,
                                        emitted_specs,
                                        emitted_count,
                                        sketches_per_task,
                                        summary,
                                        NULL);
        break;
    case ARC_SKETCH_OP_OBJECT_COLOR_TILE:
        arc_mine_object_color_tile_sketches(input,
                                            task_path,
                                            task,
                                            records,
                                            record_count,
                                            emitted_specs,
                                            emitted_count,
                                            sketches_per_task,
                                            summary,
                                            NULL);
        break;
    case ARC_SKETCH_OP_MARKER_COUNT_MOTIF_TILE:
        arc_mine_marker_count_motif_tile_sketches(input,
                                                  task_path,
                                                  task,
                                                  records,
                                                  record_count,
                                                  emitted_specs,
                                                  emitted_count,
                                                  sketches_per_task,
                                                  summary,
                                                  NULL);
        break;
    case ARC_SKETCH_OP_COMPONENT_COUNT_LINE:
        arc_mine_component_count_line_sketches(input,
                                               task_path,
                                               task,
                                               records,
                                               record_count,
                                               emitted_specs,
                                               emitted_count,
                                               sketches_per_task,
                                               summary,
                                               NULL);
        break;
    case ARC_SKETCH_OP_RECOLOR_CROP:
        arc_mine_recolor_crop_repair_sketches(input,
                                              task_path,
                                              task,
                                              records,
                                              record_count,
                                              emitted_specs,
                                              emitted_count,
                                              sketches_per_task,
                                              summary);
        break;
    case ARC_SKETCH_OP_MARKER_POLYLINE:
        arc_mine_marker_polyline_sketches(input,
                                          task_path,
                                          task,
                                          records,
                                          record_count,
                                          emitted_specs,
                                          emitted_count,
                                          sketches_per_task,
                                          summary);
        break;
    case ARC_SKETCH_OP_SEPARATOR_INTERSECTION_COLOR:
        arc_mine_separator_intersection_color_sketches(input,
                                                       task_path,
                                                       task,
                                                       records,
                                                       record_count,
                                                       emitted_specs,
                                                       emitted_count,
                                                       sketches_per_task,
                                                       summary);
        break;
    case ARC_SKETCH_OP_SEPARATOR_PANEL_MAP:
        arc_mine_separator_panel_map_sketches(input,
                                              task_path,
                                              task,
                                              records,
                                              record_count,
                                              emitted_specs,
                                              emitted_count,
                                              sketches_per_task,
                                              summary);
        break;
    case ARC_SKETCH_OP_VISUAL_COMBINATOR:
        arc_mine_visual_combinator_sketches(input,
                                            task_path,
                                            task,
                                            records,
                                            record_count,
                                            emitted_specs,
                                            emitted_count,
                                            sketches_per_task,
                                            summary,
                                            NULL);
        break;
    case ARC_SKETCH_OP_OBJECT_TRANSLATE:
        arc_mine_object_translate_sketches(input,
                                           task_path,
                                           task,
                                           records,
                                           record_count,
                                           emitted_specs,
                                           emitted_count,
                                           sketches_per_task,
                                           summary);
        break;
    default:
        break;
    }
}

static void arc_runtime_mine_with_generative_prior(
    const ARCTask *task,
    const ARCProgramLearnedState *learned,
    ARCSketchProbeRecord *records,
    size_t *record_count,
    ARCSketchProgramSpec *emitted_specs,
    size_t *emitted_count,
    size_t sketches_per_task,
    ARCSketchProbeSummary *summary,
    ARCHypothesisSelectionStats *stats)
{
    if (task == NULL || learned == NULL || records == NULL ||
        record_count == NULL || emitted_specs == NULL || emitted_count == NULL ||
        summary == NULL || learned->generative_priors == NULL ||
        learned->generative_prior_count == 0 ||
        *emitted_count >= sketches_per_task) {
        return;
    }
    ARCGenerativePriorTaskFeatures features;
    arc_generative_prior_task_features(task, &features);
    static const int prior_ops[] = {
        ARC_SKETCH_OP_OBJECT_TRANSLATE,
        ARC_SKETCH_OP_RECOLOR_CROP,
        ARC_SKETCH_OP_SEPARATOR_PANEL_MAP,
        ARC_SKETCH_OP_VISUAL_COMBINATOR,
        ARC_SKETCH_OP_PALETTE_SCALE,
        ARC_SKETCH_OP_OBJECT_COLOR_TILE,
        ARC_SKETCH_OP_MARKER_COUNT_MOTIF_TILE,
        ARC_SKETCH_OP_COMPONENT_COUNT_LINE,
        ARC_SKETCH_OP_MARKER_POLYLINE,
        ARC_SKETCH_OP_SEPARATOR_INTERSECTION_COLOR,
    };
    ARCRuntimePriorOpScore scores[sizeof(prior_ops) / sizeof(prior_ops[0])];
    size_t score_count = 0;
    for (size_t i = 0; i < sizeof(prior_ops) / sizeof(prior_ops[0]); ++i) {
        int op = prior_ops[i];
        (void)arc_generative_prior_score_for_op(learned, &features, op, &scores[score_count]);
        scores[score_count].eligible =
            arc_runtime_prior_op_supported(op) &&
            scores[score_count].matched_rows != 0 &&
            scores[score_count].score > 0.0 &&
            scores[score_count].specificity >= arc_runtime_prior_min_specificity(op) &&
            arc_runtime_prior_task_allowed(op, &features);
        ++score_count;
    }
    for (size_t i = 0; i + 1u < score_count; ++i) {
        for (size_t j = i + 1u; j < score_count; ++j) {
            if ((scores[j].eligible && !scores[i].eligible) ||
                (scores[j].eligible == scores[i].eligible &&
                 (scores[j].score > scores[i].score ||
                  (scores[j].score == scores[i].score &&
                   scores[j].specificity > scores[i].specificity)))) {
                ARCRuntimePriorOpScore tmp = scores[i];
                scores[i] = scores[j];
                scores[j] = tmp;
            }
        }
    }
    ARCSketchProbeInput prior_input;
    memset(&prior_input, 0, sizeof(prior_input));
    snprintf(prior_input.family, sizeof(prior_input.family), "%s", "arc.generative_prior");
    snprintf(prior_input.task_id, sizeof(prior_input.task_id), "%s", task->id);
    snprintf(prior_input.split, sizeof(prior_input.split), "%s", "runtime");
    snprintf(prior_input.op_hints, sizeof(prior_input.op_hints), "%s", "*");
    prior_input.support_tasks = task->train_count;
    prior_input.proposal_count = 1;
    prior_input.utility_mean = 1.0;
    size_t op_runs = 0;
    size_t op_cap = learned->arc_generative_prior_op_cap;
    for (size_t i = 0; i < score_count && *emitted_count < sketches_per_task; ++i) {
        const char *decision = "selected";
        size_t before = *emitted_count;
        size_t emitted_delta = 0;
        if (!arc_runtime_prior_op_supported(scores[i].op)) {
            decision = "filtered_unsupported_op";
            arc_print_generative_prior_runtime_trace(task,
                                                     &scores[i],
                                                     decision,
                                                     i + 1u,
                                                     op_cap,
                                                     0);
            continue;
        }
        if (scores[i].matched_rows == 0) {
            decision = "filtered_no_prior_match";
            arc_print_generative_prior_runtime_trace(task,
                                                     &scores[i],
                                                     decision,
                                                     i + 1u,
                                                     op_cap,
                                                     0);
            continue;
        }
        if (scores[i].score <= 0.0) {
            decision = "filtered_rejection_score";
            arc_print_generative_prior_runtime_trace(task,
                                                     &scores[i],
                                                     decision,
                                                     i + 1u,
                                                     op_cap,
                                                     0);
            continue;
        }
        if (scores[i].specificity < arc_runtime_prior_min_specificity(scores[i].op)) {
            decision = "filtered_low_specificity";
            arc_print_generative_prior_runtime_trace(task,
                                                     &scores[i],
                                                     decision,
                                                     i + 1u,
                                                     op_cap,
                                                     0);
            continue;
        }
        if (!arc_runtime_prior_task_allowed(scores[i].op, &features)) {
            decision = "filtered_task_guard";
            arc_print_generative_prior_runtime_trace(task,
                                                     &scores[i],
                                                     decision,
                                                     i + 1u,
                                                     op_cap,
                                                     0);
            continue;
        }
        if (op_runs >= op_cap) {
            decision = "not_selected_op_cap";
            arc_print_generative_prior_runtime_trace(task,
                                                     &scores[i],
                                                     decision,
                                                     i + 1u,
                                                     op_cap,
                                                     0);
            continue;
        }
        if (*emitted_count >= sketches_per_task) {
            decision = "not_selected_candidate_cap";
            arc_print_generative_prior_runtime_trace(task,
                                                     &scores[i],
                                                     decision,
                                                     i + 1u,
                                                     op_cap,
                                                     0);
            break;
        }
        arc_runtime_mine_prior_op(&prior_input,
                                  "runtime",
                                  task,
                                  scores[i].op,
                                  records,
                                  record_count,
                                  emitted_specs,
                                  emitted_count,
                                  sketches_per_task,
                                  summary);
        ++op_runs;
        emitted_delta = *emitted_count - before;
        arc_print_generative_prior_runtime_trace(task,
                                                 &scores[i],
                                                 decision,
                                                 i + 1u,
                                                 op_cap,
                                                 emitted_delta);
        if (stats != NULL) {
            ++stats->runtime_generative_prior_ops;
            stats->runtime_generative_prior_mined_sketches += emitted_delta;
        }
    }
}

static size_t arc_induce_runtime_mined_sketch_hypotheses(
    const ARCHypothesisTemplateBinding *binding,
    const ARCTask *task,
    ARCProgramLearnedState *learned,
    const ARCTransformCandidate *existing_candidates,
    size_t existing_count,
    ARCHypothesisProposal *proposals,
    size_t proposal_capacity,
    size_t proposal_count,
    ARCHypothesisSelectionStats *stats,
    size_t template_index,
    const ARCProgramTaskFeatures *task_features)
{
    if (binding == NULL || binding->template_fn == NULL ||
        binding->to_candidate == NULL ||
        task == NULL || task_features == NULL ||
        existing_candidates == NULL || proposals == NULL ||
        proposal_count >= proposal_capacity) {
        return proposal_count;
    }

    size_t remaining = proposal_capacity - proposal_count;
    size_t cap = learned == NULL ? 0u : learned->arc_runtime_sketch_mine_cap;
    if (cap > remaining) {
        cap = remaining;
    }
    if (cap == 0) {
        return proposal_count;
    }

    ARCSketchProbeRecord *records =
        calloc(ARC_SKETCH_PROBE_RECORD_CAPACITY, sizeof(*records));
    if (records == NULL) {
        return proposal_count;
    }

    ARCSketchProbeInput input;
    ARCSketchProbeSummary summary;
    ARCSketchProgramSpec emitted_specs[ARC_HYPOTHESIS_PROPOSAL_CAPACITY];
    size_t record_count = 0;
    size_t emitted_count = 0;
    memset(&input, 0, sizeof(input));
    memset(&summary, 0, sizeof(summary));
    memset(emitted_specs, 0, sizeof(emitted_specs));
    snprintf(input.family, sizeof(input.family), "%s", "arc.runtime_visual_mining");
    snprintf(input.task_id, sizeof(input.task_id), "%s", task->id);
    snprintf(input.split, sizeof(input.split), "%s", "runtime");
    snprintf(input.op_hints, sizeof(input.op_hints), "%s", "*");
    input.support_tasks = task->train_count;
    input.proposal_count = 1;
    input.utility_mean = 1.0;
    snprintf(summary.family, sizeof(summary.family), "%s", input.family);
    snprintf(summary.task_id, sizeof(summary.task_id), "%s", input.task_id);
    snprintf(summary.split, sizeof(summary.split), "%s", input.split);
    snprintf(summary.task_path, sizeof(summary.task_path), "%s", "runtime");

#define ARC_RUNTIME_MINE_IF_ROOM(call_expr) \
    do {                                    \
        if (emitted_count < cap) {          \
            call_expr;                      \
        }                                   \
    } while (0)

    ARC_RUNTIME_MINE_IF_ROOM(arc_mine_crop_sketches(&input,
                                                    "runtime",
                                                    task,
                                                    records,
                                                    &record_count,
                                                    emitted_specs,
                                                    &emitted_count,
                                                    cap,
                                                    &summary));
    ARC_RUNTIME_MINE_IF_ROOM(arc_mine_object_mask_extract_sketches(&input,
                                                                  "runtime",
                                                                  task,
                                                                  records,
                                                                  &record_count,
                                                                  emitted_specs,
                                                                  &emitted_count,
                                                                  cap,
                                                                  &summary));
    ARC_RUNTIME_MINE_IF_ROOM(arc_mine_macrocell_sketches(&input,
                                                        "runtime",
                                                        task,
                                                        records,
                                                        &record_count,
                                                        emitted_specs,
                                                        &emitted_count,
                                                        cap,
                                                        &summary));
    ARC_RUNTIME_MINE_IF_ROOM(arc_mine_lattice_sample_sketches(&input,
                                                             "runtime",
                                                             task,
                                                             records,
                                                             &record_count,
                                                             emitted_specs,
                                                             &emitted_count,
                                                             cap,
                                                             &summary));
    ARC_RUNTIME_MINE_IF_ROOM(arc_mine_diagonal_compact_sketches(&input,
                                                               "runtime",
                                                               task,
                                                               records,
                                                               &record_count,
                                                               emitted_specs,
                                                               &emitted_count,
                                                               cap,
                                                               &summary));
    ARC_RUNTIME_MINE_IF_ROOM(arc_mine_point_line_project_sketches(&input,
                                                                 "runtime",
                                                                 task,
                                                                 records,
                                                                 &record_count,
                                                                 emitted_specs,
                                                                 &emitted_count,
                                                                 cap,
                                                                 &summary));
    ARC_RUNTIME_MINE_IF_ROOM(arc_mine_component_summary_sketches(&input,
                                                                "runtime",
                                                                task,
                                                                records,
                                                                &record_count,
                                                                emitted_specs,
                                                                &emitted_count,
                                                                cap,
                                                                &summary));
    ARC_RUNTIME_MINE_IF_ROOM(arc_mine_axis_projection_compress_sketches(&input,
                                                                       "runtime",
                                                                       task,
                                                                       records,
                                                                       &record_count,
                                                                       emitted_specs,
                                                                       &emitted_count,
                                                                       cap,
                                                                       &summary));
    ARC_RUNTIME_MINE_IF_ROOM(arc_mine_separator_panel_relation_mask_sketches(&input,
                                                                            "runtime",
                                                                            task,
                                                                            records,
                                                                            &record_count,
                                                                            emitted_specs,
                                                                            &emitted_count,
                                                                            cap,
                                                                            &summary,
                                                                            NULL));
    ARC_RUNTIME_MINE_IF_ROOM(arc_mine_extend_aligned_sequence_sketches(&input,
                                                                      "runtime",
                                                                      task,
                                                                      records,
                                                                      &record_count,
                                                                      emitted_specs,
                                                                      &emitted_count,
                                                                      cap,
                                                                      &summary));
    ARC_RUNTIME_MINE_IF_ROOM(arc_runtime_mine_with_generative_prior(task,
                                                                   learned,
                                                                   records,
                                                                   &record_count,
                                                                   emitted_specs,
                                                                   &emitted_count,
                                                                   cap,
                                                                   &summary,
                                                                   stats));

#undef ARC_RUNTIME_MINE_IF_ROOM

    if (stats != NULL) {
        stats->runtime_mined_sketches += emitted_count;
    }

    for (size_t i = 0; i < emitted_count; ++i) {
        const ARCSketchProgramSpec *spec = &emitted_specs[i];
        proposal_count = arc_try_add_sketch_program(binding,
                                                    task,
                                                    learned,
                                                    existing_candidates,
                                                    existing_count,
                                                    proposals,
                                                    proposal_capacity,
                                                    proposal_count,
                                                    stats,
                                                    template_index,
                                                    task_features,
                                                    spec->selector,
                                                    spec->op,
                                                    spec->background,
                                                    spec->color,
                                                    spec->output_color,
                                                    spec->x_anchor,
                                                    spec->y_anchor,
                                                    spec->x_offset,
                                                    spec->y_offset,
                                                    spec->width,
                                                    spec->height,
                                                    spec->period);
    }

    free(records);
    return proposal_count;
}
