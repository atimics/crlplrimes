static int arc_apply_multiscale_candidate(const ARCTransformCandidate *c, const ARCGrid *in, ARCGrid *out);
static int arc_apply_object_label_candidate(const ARCTransformCandidate *c, const ARCGrid *in, ARCGrid *out);
static int arc_apply_holo_tile_candidate(const ARCTransformCandidate *c, const ARCGrid *in, ARCGrid *out);
static int arc_grid_to_hologram(const ARCGrid *grid, double *scene_out);
static int arc_apply_candidate(const ARCTransformCandidate *candidate,
                               const ARCGrid *input,
                               ARCGrid *output)
{
    if (arc_candidate_is_composed(candidate)) {
        ARCTransformCandidate first;
        ARCTransformCandidate second;
        ARCGrid intermediate;
        arc_candidate_from_step(&candidate->steps[0], &first);
        arc_candidate_from_step(&candidate->steps[1], &second);
        return arc_apply_candidate(&first, input, &intermediate) &&
               arc_apply_candidate(&second, &intermediate, output);
    }

    switch (candidate->op) {
    case ARC_RULE_COPY_LEFT:
    case ARC_RULE_COPY_RIGHT:
    case ARC_RULE_INTERSECTION:
    case ARC_RULE_UNION:
    case ARC_RULE_OBJECT_SELECT:
    case ARC_RULE_LAYER_OVERLAP:
    case ARC_RULE_LAYER_UNION:
    case ARC_RULE_LAYER_XOR:
    case ARC_RULE_LAYER_LEFT_ONLY:
    case ARC_RULE_LAYER_RIGHT_ONLY:
    case ARC_RULE_COLOR_MAP_LEFT:
        return arc_apply_split_panel_candidate(candidate, input, output);
    case ARC_RULE_GRID_IDENTITY:
        *output = *input;
        return 1;
    case ARC_RULE_GRID_COLOR_MAP:
        arc_init_grid(output, input->width, input->height);
        for (size_t cell = 0; cell < arc_grid_cells(input); ++cell) {
            output->cells[cell] = candidate->color_map[input->cells[cell]];
        }
        return 1;
    case ARC_RULE_GRID_FOREGROUND_RECOLOR:
        arc_init_grid(output, input->width, input->height);
        for (size_t cell = 0; cell < arc_grid_cells(input); ++cell) {
            output->cells[cell] =
                input->cells[cell] == candidate->background_color ? 0 : candidate->output_color;
        }
        return 1;
    case ARC_RULE_CROP_BACKGROUND:
        return arc_apply_crop_background(candidate, input, output, 0);
    case ARC_RULE_CROP_BACKGROUND_RECOLOR:
        return arc_apply_crop_background(candidate, input, output, 1);
    case ARC_RULE_CROP_COLOR:
        return arc_apply_crop_color(candidate, input, output, 0);
    case ARC_RULE_CROP_COLOR_MASK:
        return arc_apply_crop_color(candidate, input, output, 1);
    case ARC_RULE_COMPRESS_COLOR_LINES:
        return arc_apply_compress_color_lines(candidate, input, output);
    case ARC_RULE_TILE_MOTIF_PARITY_REPEAT:
        return arc_apply_tile_motif_parity_repeat_candidate(candidate, input, output);
    case ARC_RULE_DILATE:
        return arc_apply_dilate_candidate(candidate, input, output);
    case ARC_RULE_GRAVITY_DOWN: case ARC_RULE_GRAVITY_UP: case ARC_RULE_GRAVITY_LEFT: case ARC_RULE_GRAVITY_RIGHT:
        return arc_apply_gravity_candidate(candidate, input, output);
    case ARC_RULE_TILE_REPEAT:
        return arc_apply_tile_repeat_candidate(candidate, input, output);
    case ARC_RULE_GRID_REPEAT:
        return arc_apply_grid_repeat_candidate(candidate, input, output);
    case ARC_RULE_MARKER_MOTIF_TILE:
        return arc_apply_marker_motif_tile_candidate(candidate, input, output);
    case ARC_RULE_AXIS_TILE:
        return arc_apply_axis_tile_candidate(candidate, input, output);
    case ARC_RULE_PERIOD_EXTEND:
        return arc_apply_period_extend_candidate(candidate, input, output);
    case ARC_RULE_EXTRACT_COLOR: case ARC_RULE_ERASE_COLOR:
        return arc_apply_extract_erase_color_candidate(candidate, input, output);
    case ARC_RULE_RECOLOR_BY_REFERENCE:
        return arc_apply_recolor_by_reference_candidate(candidate, input, output);
    case ARC_RULE_PATTERN_EXTEND:
        return arc_apply_pattern_extend_candidate(candidate, input, output);
    case ARC_RULE_DRAW_LINE:
        return arc_apply_draw_line_candidate(candidate, input, output);
    case ARC_RULE_SYMMETRY_H: case ARC_RULE_SYMMETRY_V:
        return arc_apply_symmetry_candidate(candidate, input, output);
    case ARC_RULE_MULTISCALE:
        return arc_apply_multiscale_candidate(candidate, input, output);
    case ARC_RULE_OBJECT_LABEL:
        return arc_apply_object_label_candidate(candidate, input, output);
    case ARC_RULE_HOLO_TILE:
        return arc_apply_holo_tile_candidate(candidate, input, output);
    case ARC_RULE_OBJECT_CROP_LARGEST:
    case ARC_RULE_OBJECT_CROP_LARGEST_RECOLOR:
    case ARC_RULE_OBJECT_MASK_LARGEST:
        return arc_apply_largest_component_candidate(candidate, input, output);
    case ARC_RULE_OBJECT_TRANSLATE:
        return arc_apply_object_translate_candidate(candidate, input, output);
    case ARC_RULE_OBJECT_CONNECT_ALIGNED:
        return arc_apply_object_connect_candidate(candidate, input, output);
    case ARC_RULE_OBJECT_FILL_ENCLOSED:
        return arc_apply_object_fill_enclosed_candidate(candidate, input, output);
    case ARC_RULE_HYPOTHESIS_APPLIED:
        return arc_apply_hypothesis_candidate(candidate, input, output);
    case ARC_RULE_OBJECT_RELATION_INDUCED:
        switch (arc_object_relation_kind(candidate)) {
        case ARC_OBJECT_RELATION_TRANSLATE:
            return arc_apply_object_translate_candidate(candidate, input, output);
        case ARC_OBJECT_RELATION_CONNECT_ALIGNED:
            return arc_apply_object_connect_candidate(candidate, input, output);
        case ARC_OBJECT_RELATION_FILL_ENCLOSED:
            return arc_apply_object_fill_enclosed_candidate(candidate, input, output);
        default:
            return 0;
        }
    case ARC_RULE_ROTATE90:
        if (input->height > ARC_MAX_GRID_WIDTH || input->width > ARC_MAX_GRID_HEIGHT) {
            return 0;
        }
        arc_init_grid(output, input->height, input->width);
        for (size_t y = 0; y < input->height; ++y) {
            for (size_t x = 0; x < input->width; ++x) {
                size_t out_x = input->height - 1u - y;
                size_t out_y = x;
                output->cells[out_y * output->width + out_x] = input->cells[y * input->width + x];
            }
        }
        return 1;
    case ARC_RULE_ROTATE180:
        arc_init_grid(output, input->width, input->height);
        for (size_t y = 0; y < input->height; ++y) {
            for (size_t x = 0; x < input->width; ++x) {
                size_t out_x = input->width - 1u - x;
                size_t out_y = input->height - 1u - y;
                output->cells[out_y * output->width + out_x] = input->cells[y * input->width + x];
            }
        }
        return 1;
    case ARC_RULE_ROTATE270:
        if (input->height > ARC_MAX_GRID_WIDTH || input->width > ARC_MAX_GRID_HEIGHT) {
            return 0;
        }
        arc_init_grid(output, input->height, input->width);
        for (size_t y = 0; y < input->height; ++y) {
            for (size_t x = 0; x < input->width; ++x) {
                size_t out_x = y;
                size_t out_y = input->width - 1u - x;
                output->cells[out_y * output->width + out_x] = input->cells[y * input->width + x];
            }
        }
        return 1;
    case ARC_RULE_FLIP_HORIZONTAL:
        arc_init_grid(output, input->width, input->height);
        for (size_t y = 0; y < input->height; ++y) {
            for (size_t x = 0; x < input->width; ++x) {
                output->cells[y * output->width + (input->width - 1u - x)] =
                    input->cells[y * input->width + x];
            }
        }
        return 1;
    case ARC_RULE_FLIP_VERTICAL:
        arc_init_grid(output, input->width, input->height);
        for (size_t y = 0; y < input->height; ++y) {
            for (size_t x = 0; x < input->width; ++x) {
                output->cells[(input->height - 1u - y) * output->width + x] =
                    input->cells[y * input->width + x];
            }
        }
        return 1;
    case ARC_RULE_TRANSPOSE:
        if (input->height > ARC_MAX_GRID_WIDTH || input->width > ARC_MAX_GRID_HEIGHT) {
            return 0;
        }
        arc_init_grid(output, input->height, input->width);
        for (size_t y = 0; y < input->height; ++y) {
            for (size_t x = 0; x < input->width; ++x) {
                output->cells[x * output->width + y] = input->cells[y * input->width + x];
            }
        }
        return 1;
    case ARC_RULE_SCALE:
        if (candidate->scale == 0 ||
            input->width * candidate->scale > ARC_MAX_GRID_WIDTH ||
            input->height * candidate->scale > ARC_MAX_GRID_HEIGHT) {
            return 0;
        }
        arc_init_grid(output, input->width * candidate->scale, input->height * candidate->scale);
        for (size_t y = 0; y < input->height; ++y) {
            for (size_t x = 0; x < input->width; ++x) {
                uint8_t value = input->cells[y * input->width + x];
                for (size_t yy = 0; yy < candidate->scale; ++yy) {
                    for (size_t xx = 0; xx < candidate->scale; ++xx) {
                        size_t out_x = x * candidate->scale + xx;
                        size_t out_y = y * candidate->scale + yy;
                        output->cells[out_y * output->width + out_x] = value;
                    }
                }
            }
        }
        return 1;
    case ARC_RULE_REFLECT_TILE: {
        size_t scale = candidate->scale == 0 ? 2u : candidate->scale;
        if (scale < 2 ||
            input->width * scale > ARC_MAX_GRID_WIDTH ||
            input->height * scale > ARC_MAX_GRID_HEIGHT) {
            return 0;
        }
        arc_init_grid(output, input->width * scale, input->height * scale);
        for (size_t y = 0; y < output->height; ++y) {
            size_t tile_y = y / input->height;
            size_t src_y = y % input->height;
            if (tile_y & 1u) src_y = input->height - 1u - src_y;
            for (size_t x = 0; x < output->width; ++x) {
                size_t tile_x = x / input->width;
                size_t src_x = x % input->width;
                if (tile_x & 1u) src_x = input->width - 1u - src_x;
                output->cells[y * output->width + x] =
                    input->cells[src_y * input->width + src_x];
            }
        }
        return 1;
    }
    case ARC_RULE_BBOX_OUTLINE:
    case ARC_RULE_BBOX_FILL: {
        size_t min_x = 0;
        size_t min_y = 0;
        size_t max_x = 0;
        size_t max_y = 0;
        if (!arc_find_non_background_bbox(input,
                                          candidate->background_color,
                                          &min_x,
                                          &min_y,
                                          &max_x,
                                          &max_y)) {
            return 0;
        }
        arc_init_grid(output, input->width, input->height);
        for (size_t y = min_y; y <= max_y; ++y) {
            for (size_t x = min_x; x <= max_x; ++x) {
                if (candidate->op == ARC_RULE_BBOX_FILL ||
                    x == min_x || x == max_x || y == min_y || y == max_y) {
                    output->cells[y * output->width + x] = candidate->output_color;
                }
            }
        }
        return 1;
    }
    case ARC_RULE_TRANSLATE:
        arc_init_grid(output, input->width, input->height);
        for (size_t y = 0; y < input->height; ++y) {
            for (size_t x = 0; x < input->width; ++x) {
                uint8_t value = input->cells[y * input->width + x];
                if (value == candidate->background_color) {
                    continue;
                }
                int out_x = (int)x + candidate->dx;
                int out_y = (int)y + candidate->dy;
                if (out_x < 0 || out_y < 0 ||
                    out_x >= (int)input->width || out_y >= (int)input->height) {
                    return 0;
                }
                output->cells[(size_t)out_y * output->width + (size_t)out_x] = value;
            }
        }
        return 1;
    default:
        return 0;
    }
}

static int arc_candidate_matches_train(const ARCTask *task,
                                       const ARCTransformCandidate *candidate,
                                       size_t *checked_cells_out,
                                       size_t *mismatches_out,
                                       size_t *unsupported_pairs_out)
{
    ARCGrid produced;
    size_t checked_cells = 0;
    size_t mismatches = 0;
    size_t unsupported_pairs = 0;
    for (size_t pair = 0; pair < task->train_count; ++pair) {
        if (!task->train[pair].has_output ||
            !arc_apply_candidate(candidate, &task->train[pair].input, &produced) ||
            !arc_grid_same_shape(&produced, &task->train[pair].output)) {
            ++unsupported_pairs;
            continue;
        }
        size_t cells = arc_grid_cells(&task->train[pair].output);
        checked_cells += cells;
        for (size_t cell = 0; cell < cells; ++cell) {
            if (produced.cells[cell] != task->train[pair].output.cells[cell]) {
                ++mismatches;
            }
        }
    }
    if (checked_cells_out != NULL) {
        *checked_cells_out = checked_cells;
    }
    if (mismatches_out != NULL) {
        *mismatches_out = mismatches;
    }
    if (unsupported_pairs_out != NULL) {
        *unsupported_pairs_out = unsupported_pairs;
    }
    return unsupported_pairs == 0 && checked_cells > 0 && mismatches == 0;
}

static int arc_candidate_train_match_telemetry(const ARCTask *task,
                                               const ARCTransformCandidate *candidate,
                                               ARCTrainMatchTelemetry *telemetry)
{
    ARCGrid produced;
    int first_failure_set = 0;
    if (telemetry == NULL) {
        return arc_candidate_matches_train(task, candidate, NULL, NULL, NULL);
    }
    memset(telemetry, 0, sizeof(*telemetry));
    telemetry->first_failing_pair = SIZE_MAX;
    telemetry->first_mismatch_min_x = SIZE_MAX;
    telemetry->first_mismatch_min_y = SIZE_MAX;
    telemetry->first_mismatch_max_x = SIZE_MAX;
    telemetry->first_mismatch_max_y = SIZE_MAX;
    if (task == NULL || candidate == NULL) {
        telemetry->verifier_score = 0.0;
        return 0;
    }
    telemetry->train_pairs = task->train_count;
    for (size_t pair = 0; pair < task->train_count; ++pair) {
        const ARCGrid *input = &task->train[pair].input;
        const ARCGrid *expected = &task->train[pair].output;
        if (!task->train[pair].has_output) {
            ++telemetry->unsupported_pairs;
            ++telemetry->missing_output_pairs;
            if (!first_failure_set) {
                telemetry->first_failing_pair = pair;
                telemetry->first_input_width = input->width;
                telemetry->first_input_height = input->height;
                first_failure_set = 1;
            }
            continue;
        }
        if (!arc_apply_candidate(candidate, input, &produced)) {
            ++telemetry->unsupported_pairs;
            ++telemetry->apply_failed_pairs;
            if (!first_failure_set) {
                telemetry->first_failing_pair = pair;
                telemetry->first_input_width = input->width;
                telemetry->first_input_height = input->height;
                telemetry->first_expected_width = expected->width;
                telemetry->first_expected_height = expected->height;
                first_failure_set = 1;
            }
            continue;
        }
        if (!arc_grid_same_shape(&produced, expected)) {
            ++telemetry->unsupported_pairs;
            ++telemetry->shape_mismatch_pairs;
            if (!first_failure_set) {
                telemetry->first_failing_pair = pair;
                telemetry->first_input_width = input->width;
                telemetry->first_input_height = input->height;
                telemetry->first_expected_width = expected->width;
                telemetry->first_expected_height = expected->height;
                telemetry->first_produced_width = produced.width;
                telemetry->first_produced_height = produced.height;
                first_failure_set = 1;
            }
            continue;
        }
        size_t cells = arc_grid_cells(expected);
        size_t pair_mismatches = 0;
        size_t mismatch_min_x = SIZE_MAX;
        size_t mismatch_min_y = SIZE_MAX;
        size_t mismatch_max_x = 0;
        size_t mismatch_max_y = 0;
        telemetry->checked_cells += cells;
        for (size_t cell = 0; cell < cells; ++cell) {
            if (produced.cells[cell] != expected->cells[cell]) {
                ++pair_mismatches;
                size_t x = cell % expected->width;
                size_t y = cell / expected->width;
                if (x < mismatch_min_x) {
                    mismatch_min_x = x;
                }
                if (y < mismatch_min_y) {
                    mismatch_min_y = y;
                }
                if (x > mismatch_max_x) {
                    mismatch_max_x = x;
                }
                if (y > mismatch_max_y) {
                    mismatch_max_y = y;
                }
            }
        }
        if (pair_mismatches > 0 && !first_failure_set) {
            telemetry->first_failing_pair = pair;
            telemetry->first_input_width = input->width;
            telemetry->first_input_height = input->height;
            telemetry->first_expected_width = expected->width;
            telemetry->first_expected_height = expected->height;
            telemetry->first_produced_width = produced.width;
            telemetry->first_produced_height = produced.height;
            telemetry->first_pair_checked_cells = cells;
            telemetry->first_pair_mismatches = pair_mismatches;
            telemetry->first_mismatch_min_x = mismatch_min_x;
            telemetry->first_mismatch_min_y = mismatch_min_y;
            telemetry->first_mismatch_max_x = mismatch_max_x;
            telemetry->first_mismatch_max_y = mismatch_max_y;
            first_failure_set = 1;
        }
        telemetry->mismatches += pair_mismatches;
    }
    if (!first_failure_set) {
        telemetry->first_failing_pair = SIZE_MAX;
    }
    if (telemetry->unsupported_pairs == 0 &&
        telemetry->checked_cells > 0 &&
        telemetry->mismatches == 0) {
        telemetry->verifier_score = 1.0;
        return 1;
    }
    double checked_score = telemetry->checked_cells == 0
                               ? 0.0
                               : 1.0 -
                                     (double)telemetry->mismatches /
                                         (double)telemetry->checked_cells;
    double support_score = telemetry->train_pairs == 0
                               ? 0.0
                               : 1.0 -
                                     (double)telemetry->unsupported_pairs /
                                         (double)telemetry->train_pairs;
    if (checked_score < 0.0) {
        checked_score = 0.0;
    }
    if (support_score < 0.0) {
        support_score = 0.0;
    }
    telemetry->verifier_score = 0.65 * checked_score + 0.35 * support_score;
    return 0;
}

static void arc_append_accepted_rule(char *buffer, size_t buffer_size, const char *rule_name)
{
    size_t used = strlen(buffer);
    if (used + 1u >= buffer_size) {
        return;
    }
    int written = snprintf(buffer + used,
                           buffer_size - used,
                           "%s%s",
                           used == 0 ? "" : ",",
                           rule_name);
    if (written < 0 || (size_t)written >= buffer_size - used) {
        buffer[buffer_size - 1u] = '\0';
    }
}

static int arc_apply_candidate_to_tests(const ARCTask *task,
                                        const ARCTransformCandidate *candidate,
                                        ARCGrid *predictions,
                                        size_t *unsupported_pairs_out)
{
    size_t unsupported_pairs = 0;
    for (size_t pair = 0; pair < task->test_count; ++pair) {
        if (!arc_apply_candidate(candidate, &task->test[pair].input, &predictions[pair])) {
            ++unsupported_pairs;
        }
    }
    if (unsupported_pairs_out != NULL) {
        *unsupported_pairs_out = unsupported_pairs;
    }
    return unsupported_pairs == 0;
}

static uint64_t arc_hash_mix(uint64_t hash, uint64_t value)
{
    hash ^= value + UINT64_C(0x9e3779b97f4a7c15) + (hash << 6) + (hash >> 2);
    return hash;
}

static size_t arc_task_test_cells(const ARCTask *task)
{
    size_t cells = 0;
    for (size_t pair = 0; pair < task->test_count; ++pair) {
        if (task->test[pair].has_output) {
            cells += arc_grid_cells(&task->test[pair].output);
        }
    }
    return cells;
}

static uint64_t arc_prediction_hash(const ARCGrid *predictions, size_t count)
{
    uint64_t hash = UINT64_C(1469598103934665603);
    for (size_t pair = 0; pair < count; ++pair) {
        hash = arc_hash_mix(hash, predictions[pair].width);
        hash = arc_hash_mix(hash, predictions[pair].height);
        size_t cells = arc_grid_cells(&predictions[pair]);
        for (size_t cell = 0; cell < cells; ++cell) {
            hash = arc_hash_mix(hash, predictions[pair].cells[cell]);
        }
    }
    return hash;
}

static int arc_candidate_is_object_program(const ARCTransformCandidate *candidate)
{
    if (arc_candidate_is_composed(candidate)) {
        return 0;
    }
    return candidate->op == ARC_RULE_OBJECT_CROP_LARGEST ||
           candidate->op == ARC_RULE_OBJECT_CROP_LARGEST_RECOLOR ||
           candidate->op == ARC_RULE_OBJECT_MASK_LARGEST ||
           candidate->op == ARC_RULE_OBJECT_TRANSLATE ||
           candidate->op == ARC_RULE_OBJECT_CONNECT_ALIGNED ||
           candidate->op == ARC_RULE_OBJECT_FILL_ENCLOSED ||
           candidate->op == ARC_RULE_OBJECT_RELATION_INDUCED ||
           candidate->op == ARC_RULE_HYPOTHESIS_APPLIED;
}

static int arc_candidate_is_selected_object_program(const ARCTransformCandidate *candidate)
{
    return candidate != NULL &&
           !arc_candidate_is_composed(candidate) &&
           candidate->selector_kind != ARC_OBJECT_SELECTOR_ALL_COLOR &&
           (candidate->op == ARC_RULE_OBJECT_CROP_LARGEST ||
            candidate->op == ARC_RULE_OBJECT_CROP_LARGEST_RECOLOR ||
            candidate->op == ARC_RULE_OBJECT_MASK_LARGEST);
}

static size_t arc_candidate_family(const ARCTransformCandidate *candidate)
{
    if (candidate == NULL) return 0;
    if (arc_candidate_is_hypothesis_induced(candidate)) return 15;
    return candidate->op < ARC_RULE_COUNT ? arc_op_info[candidate->op].family : 0;
}

static int arc_candidate_is_hypothesis_induced(const ARCTransformCandidate *candidate)
{
    return candidate != NULL &&
           (candidate->op == ARC_RULE_HYPOTHESIS_APPLIED ||
            strncmp(candidate->name, "hypothesis_", strlen("hypothesis_")) == 0);
}

static double arc_structural_program_score(const ARCTransformCandidate *candidate)
{
    if (candidate == NULL) return 0.0;
    if (arc_candidate_is_hypothesis_induced(candidate))
        return 130.0 - (double)(abs(candidate->dx) + abs(candidate->dy));
    double base = candidate->op < ARC_RULE_COUNT ? arc_op_info[candidate->op].score : 0.0;
    if (base == 0.0) return 0.0;
    if (candidate->op == ARC_RULE_TRANSLATE || candidate->op == ARC_RULE_OBJECT_TRANSLATE)
        base -= (double)(abs(candidate->dx) + abs(candidate->dy));
    else if (candidate->op == ARC_RULE_SCALE || candidate->op == ARC_RULE_TILE_MOTIF_PARITY_REPEAT)
        base -= (double)candidate->scale;
    else if (candidate->op == ARC_RULE_MARKER_MOTIF_TILE)
        base -= (double)candidate->selector_kind;
    else if (candidate->op == ARC_RULE_AXIS_TILE)
        base -= (double)(abs(candidate->dx - 1) + abs(candidate->dy - 1) + candidate->render_mode);
    else if (candidate->op == ARC_RULE_PERIOD_EXTEND)
        base -= (double)(abs(candidate->dx - 1) + abs(candidate->dy - 1));
    return base;
}

static int arc_task_has_split_panel_shape(const ARCTask *task)
{
    if (task == NULL || task->train_count == 0) {
        return 0;
    }
    for (size_t pair = 0; pair < task->train_count; ++pair) {
        const ARCGrid *input = &task->train[pair].input;
        const ARCGrid *output = &task->train[pair].output;
        if (!task->train[pair].has_output ||
            input->height == 0 ||
            input->height != output->height ||
            input->width != (2u * output->width + 1u) ||
            output->width == 0) {
            return 0;
        }
        size_t separator_x = output->width;
        uint8_t separator = input->cells[separator_x];
        if (separator == 0) {
            return 0;
        }
        for (size_t y = 0; y < input->height; ++y) {
            if (input->cells[y * input->width + separator_x] != separator) {
                return 0;
            }
        }
    }
    return 1;
}

static uint8_t arc_task_dominant_output_color(const ARCTask *task)
{
    size_t counts[ARC_COLOR_COUNT] = {0};
    if (task == NULL) {
        return 0;
    }
    for (size_t pair = 0; pair < task->train_count; ++pair) {
        if (!task->train[pair].has_output) {
            continue;
        }
        const ARCGrid *output = &task->train[pair].output;
        size_t cells = arc_grid_cells(output);
        for (size_t cell = 0; cell < cells; ++cell) {
            uint8_t color = output->cells[cell];
            if (color != 0 && color < ARC_COLOR_COUNT) {
                ++counts[color];
            }
        }
    }

    uint8_t best_color = 0;
    size_t best_count = 0;
    for (uint8_t color = 1; color < ARC_COLOR_COUNT; ++color) {
        if (counts[color] > best_count) {
            best_count = counts[color];
            best_color = color;
        }
    }
    return best_color;
}

static size_t arc_layer_relation_count(const ARCTransformCandidate *candidate,
                                       const ARCGrid *input,
                                       const ARCGrid *output)
{
    if (candidate == NULL || input == NULL || output == NULL ||
        input->width != 2u * output->width + 1u ||
        input->height != output->height) {
        return 0;
    }
    size_t right_offset = output->width + 1u;
    size_t count = 0;
    for (size_t y = 0; y < output->height; ++y) {
        for (size_t x = 0; x < output->width; ++x) {
            int left_on = input->cells[y * input->width + x] != 0;
            int right_on = input->cells[y * input->width + right_offset + x] != 0;
            count += arc_layer_relation_value(candidate->op, left_on, right_on) ? 1u : 0u;
        }
    }
    return count;
}

static double arc_task_layer_count_error(const ARCTask *task,
                                         const ARCTransformCandidate *candidate)
{
    if (task == NULL || candidate == NULL || !arc_candidate_is_layer_program(candidate) ||
        !arc_task_has_split_panel_shape(task) || task->train_count == 0) {
        return 1.0e9;
    }
    double error = 0.0;
    size_t pairs = 0;
    for (size_t pair = 0; pair < task->train_count; ++pair) {
        if (!task->train[pair].has_output) {
            continue;
        }
        size_t predicted = arc_layer_relation_count(candidate,
                                                    &task->train[pair].input,
                                                    &task->train[pair].output);
        size_t observed = arc_count_non_background(&task->train[pair].output, 0);
        error += fabs((double)predicted - (double)observed);
        ++pairs;
    }
    return pairs == 0 ? 1.0e9 : error / (double)pairs;
}

static double arc_task_conditioned_program_score(const ARCTask *task,
                                                 const ARCTransformCandidate *candidate)
{
    double score = arc_structural_program_score(candidate);
    if (!arc_candidate_is_layer_program(candidate)) {
        return score;
    }
    if (!arc_task_has_split_panel_shape(task)) {
        return score - 24.0;
    }

    score += 60.0;
    uint8_t output_color = arc_task_dominant_output_color(task);
    score += output_color != 0 && candidate->output_color == output_color ? 40.0 : -40.0;

    double count_error = arc_task_layer_count_error(task, candidate);
    if (count_error <= 0.0) {
        score += 35.0;
    } else if (count_error < 8.0) {
        score += 24.0 / (1.0 + count_error);
    } else {
        score -= 16.0;
    }
    return score;
}

static int arc_candidate_to_program(const ARCTransformCandidate *candidate,
                                    NNProgram *program)
{
    if (candidate == NULL || program == NULL) {
        return 0;
    }

    if (arc_candidate_is_composed(candidate)) {
        ARCTransformCandidate first_candidate;
        ARCTransformCandidate second_candidate;
        NNProgram first_program;
        NNProgram second_program;
        arc_candidate_from_step(&candidate->steps[0], &first_candidate);
        arc_candidate_from_step(&candidate->steps[1], &second_candidate);
        if (!arc_candidate_to_program(&first_candidate, &first_program) ||
            !arc_candidate_to_program(&second_candidate, &second_program) ||
            !nn_program_compose(program, &first_program, &second_program, candidate->name)) {
            return 0;
        }
        nn_program_add_param(program,
                             "first_op",
                             (double)first_candidate.op,
                             (int64_t)first_candidate.op);
        nn_program_add_param(program,
                             "second_op",
                             (double)second_candidate.op,
                             (int64_t)second_candidate.op);
        nn_program_add_param(program,
                             "first_family",
                             (double)arc_candidate_family(&first_candidate),
                             (int64_t)arc_candidate_family(&first_candidate));
        nn_program_add_param(program,
                             "second_family",
                             (double)arc_candidate_family(&second_candidate),
                             (int64_t)arc_candidate_family(&second_candidate));
    } else if (candidate->op == ARC_RULE_HYPOTHESIS_APPLIED) {
        NNHypothesis hypothesis;
        return arc_hypothesis_from_candidate(candidate, &hypothesis) &&
               nn_hypothesis_compile(&hypothesis, program);
    } else if (candidate->op == ARC_RULE_OBJECT_RELATION_INDUCED) {
        NNProgram scene;
        NNProgram relation;
        size_t relation_kind = arc_object_relation_kind(candidate);
        int relation_mode = relation_kind == ARC_OBJECT_RELATION_CONNECT_ALIGNED
                                ? candidate->dx
                                : 0;
        nn_program_init(&scene,
                        "arc_agi",
                        "object_scene",
                        NN_PROGRAM_KIND_TRANSFORM,
                        NN_PROGRAM_VALUE_GRID,
                        NN_PROGRAM_VALUE_OBJECT_SET,
                        110);
        nn_program_add_param(&scene,
                             "background",
                             (double)candidate->background_color,
                             candidate->background_color);
        nn_program_init(&relation,
                        "arc_agi",
                        "object_relation_induced",
                        NN_PROGRAM_KIND_TRANSFORM,
                        NN_PROGRAM_VALUE_OBJECT_SET,
                        NN_PROGRAM_VALUE_GRID,
                        116);
        nn_program_add_param(&relation,
                             "relation_kind",
                             (double)relation_kind,
                             (int64_t)relation_kind);
        nn_program_add_param(&relation,
                             "selector_kind",
                             (double)arc_candidate_selector_kind(candidate),
                             (int64_t)arc_candidate_selector_kind(candidate));
        nn_program_add_param(&relation,
                             "output_selector_kind",
                             (double)arc_candidate_output_selector_kind(candidate),
                             (int64_t)arc_candidate_output_selector_kind(candidate));
        nn_program_add_param(&relation,
                             "reference_color",
                             (double)candidate->reference_color,
                             candidate->reference_color);
        nn_program_add_param(&relation,
                             "render_mode",
                             (double)arc_candidate_render_mode(candidate),
                             arc_candidate_render_mode(candidate));
        nn_program_add_param(&relation,
                             "input_color",
                             (double)candidate->input_color,
                             candidate->input_color);
        nn_program_add_param(&relation,
                             "output_color",
                             (double)candidate->output_color,
                             candidate->output_color);
        nn_program_add_param(&relation,
                             "background",
                             (double)candidate->background_color,
                             candidate->background_color);
        nn_program_add_param(&relation, "dx", (double)candidate->dx, candidate->dx);
        nn_program_add_param(&relation, "dy", (double)candidate->dy, candidate->dy);
        nn_program_add_param(&relation, "mode", (double)relation_mode, relation_mode);
        nn_program_add_feature(&relation, (nn_real)((double)relation_kind / 4.0));
        nn_program_add_feature(&relation,
                               (nn_real)((double)arc_candidate_selector_kind(candidate) /
                                         (double)(ARC_OBJECT_SELECTOR_KIND_COUNT - 1)));
        nn_program_add_feature(
            &relation,
            (nn_real)((double)arc_candidate_output_selector_kind(candidate) /
                      (double)(ARC_OBJECT_SELECTOR_KIND_COUNT - 1)));
        nn_program_add_feature(&relation,
                               (nn_real)((double)arc_candidate_render_mode(candidate) /
                                         (double)(ARC_OBJECT_RENDER_KIND_COUNT - 1)));
        nn_program_add_feature(&relation, (nn_real)((double)candidate->reference_color / 9.0));
        nn_program_add_feature(&relation, (nn_real)((double)candidate->input_color / 9.0));
        nn_program_add_feature(&relation, (nn_real)((double)candidate->output_color / 9.0));
        nn_program_add_feature(&relation,
                               (nn_real)((double)candidate->dx / (double)ARC_MAX_GRID_WIDTH));
        nn_program_add_feature(&relation,
                               (nn_real)((double)candidate->dy / (double)ARC_MAX_GRID_HEIGHT));
        if (!nn_program_compose(program, &scene, &relation, candidate->name)) {
            return 0;
        }
    } else if (arc_candidate_is_layer_program(candidate)) {
        NNProgram layers;
        NNProgram relation;
        char relation_name[NN_PROGRAM_MAX_NAME];
        snprintf(relation_name,
                 sizeof(relation_name),
                 "render_layer_%s",
                 arc_layer_relation_name(candidate->op));
        nn_program_init(&layers,
                        "arc_agi",
                        "split_panel_layers",
                        NN_PROGRAM_KIND_TRANSFORM,
                        NN_PROGRAM_VALUE_GRID,
                        NN_PROGRAM_VALUE_OBJECT_SET,
                        120);
        nn_program_add_param(&layers, "background", 0.0, 0);
        nn_program_init(&relation,
                        "arc_agi",
                        relation_name,
                        NN_PROGRAM_KIND_TRANSFORM,
                        NN_PROGRAM_VALUE_OBJECT_SET,
                        NN_PROGRAM_VALUE_GRID,
                        121);
        nn_program_add_param(&relation,
                             "relation",
                             (double)(candidate->op - ARC_RULE_LAYER_OVERLAP),
                             (int64_t)(candidate->op - ARC_RULE_LAYER_OVERLAP));
        nn_program_add_param(&relation,
                             "output_color",
                             (double)candidate->output_color,
                             candidate->output_color);
        nn_program_add_param(&relation,
                             "relation_name_hash",
                             (double)(candidate->op - ARC_RULE_LAYER_OVERLAP),
                             (int64_t)(candidate->op - ARC_RULE_LAYER_OVERLAP));
        nn_program_add_feature(&relation,
                               (nn_real)((double)(candidate->op - ARC_RULE_LAYER_OVERLAP) / 4.0));
        nn_program_add_feature(&relation, (nn_real)((double)candidate->output_color / 9.0));
        if (!nn_program_compose(program, &layers, &relation, candidate->name)) {
            return 0;
        }
    } else if (candidate->op == ARC_RULE_OBJECT_TRANSLATE) {
        NNProgram scene;
        NNProgram selector;
        NNProgram mover;
        NNProgram renderer;
        NNProgram selected;
        NNProgram moved;
        nn_program_init(&scene,
                        "arc_agi",
                        "object_scene",
                        NN_PROGRAM_KIND_TRANSFORM,
                        NN_PROGRAM_VALUE_GRID,
                        NN_PROGRAM_VALUE_OBJECT_SET,
                        110);
        nn_program_add_param(&scene,
                             "background",
                             (double)candidate->background_color,
                             candidate->background_color);
        nn_program_init(&selector,
                        "arc_agi",
                        "select_object_by_selector",
                        NN_PROGRAM_KIND_SELECTOR,
                        NN_PROGRAM_VALUE_OBJECT_SET,
                        NN_PROGRAM_VALUE_OBJECT,
                        111);
        nn_program_add_param(&selector,
                             "color",
                             (double)candidate->input_color,
                             candidate->input_color);
        nn_program_add_param(&selector,
                             "selector_kind",
                             (double)arc_candidate_selector_kind(candidate),
                             arc_candidate_selector_kind(candidate));
        nn_program_add_param(&selector,
                             "reference_color",
                             (double)candidate->reference_color,
                             candidate->reference_color);
        nn_program_init(&mover,
                        "arc_agi",
                        "translate_object",
                        NN_PROGRAM_KIND_TRANSFORM,
                        NN_PROGRAM_VALUE_OBJECT,
                        NN_PROGRAM_VALUE_OBJECT,
                        112);
        nn_program_add_param(&mover, "dx", (double)candidate->dx, candidate->dx);
        nn_program_add_param(&mover, "dy", (double)candidate->dy, candidate->dy);
        nn_program_init(&renderer,
                        "arc_agi",
                        "render_object_grid",
                        NN_PROGRAM_KIND_TRANSFORM,
                        NN_PROGRAM_VALUE_OBJECT,
                        NN_PROGRAM_VALUE_GRID,
                        113);
        nn_program_add_param(&renderer,
                             "background",
                             (double)candidate->background_color,
                             candidate->background_color);
        nn_program_add_param(&renderer,
                             "output_color",
                             (double)candidate->output_color,
                             candidate->output_color);
        if (!nn_program_compose(&selected,
                                &scene,
                                &selector,
                                "object_scene_then_select_color") ||
            !nn_program_compose(&moved,
                                &selected,
                                &mover,
                                "object_scene_select_then_translate") ||
            !nn_program_compose(program, &moved, &renderer, candidate->name)) {
            return 0;
        }
    } else if (candidate->op == ARC_RULE_OBJECT_CONNECT_ALIGNED) {
        NNProgram scene;
        NNProgram connector;
        nn_program_init(&scene,
                        "arc_agi",
                        "object_scene",
                        NN_PROGRAM_KIND_TRANSFORM,
                        NN_PROGRAM_VALUE_GRID,
                        NN_PROGRAM_VALUE_OBJECT_SET,
                        110);
        nn_program_add_param(&scene,
                             "background",
                             (double)candidate->background_color,
                             candidate->background_color);
        nn_program_init(&connector,
                        "arc_agi",
                        "connect_aligned_color_objects",
                        NN_PROGRAM_KIND_TRANSFORM,
                        NN_PROGRAM_VALUE_OBJECT_SET,
                        NN_PROGRAM_VALUE_GRID,
                        114);
        nn_program_add_param(&connector,
                             "color",
                             (double)candidate->input_color,
                             candidate->input_color);
        nn_program_add_param(&connector,
                             "output_color",
                             (double)candidate->output_color,
                             candidate->output_color);
        nn_program_add_param(&connector, "mode", (double)candidate->dx, candidate->dx);
        nn_program_add_param(&connector,
                             "render_mode",
                             (double)arc_candidate_render_mode(candidate),
                             arc_candidate_render_mode(candidate));
        if (!nn_program_compose(program, &scene, &connector, candidate->name)) {
            return 0;
        }
    } else if (candidate->op == ARC_RULE_OBJECT_FILL_ENCLOSED) {
        NNProgram scene;
        NNProgram filler;
        nn_program_init(&scene,
                        "arc_agi",
                        "object_scene",
                        NN_PROGRAM_KIND_TRANSFORM,
                        NN_PROGRAM_VALUE_GRID,
                        NN_PROGRAM_VALUE_OBJECT_SET,
                        110);
        nn_program_add_param(&scene,
                             "background",
                             (double)candidate->background_color,
                             candidate->background_color);
        nn_program_init(&filler,
                        "arc_agi",
                        "fill_enclosed_by_color",
                        NN_PROGRAM_KIND_TRANSFORM,
                        NN_PROGRAM_VALUE_OBJECT_SET,
                        NN_PROGRAM_VALUE_GRID,
                        115);
        nn_program_add_param(&filler,
                             "border_color",
                             (double)candidate->input_color,
                             candidate->input_color);
        nn_program_add_param(&filler,
                             "fill_color",
                             (double)candidate->output_color,
                             candidate->output_color);
        if (!nn_program_compose(program, &scene, &filler, candidate->name)) {
            return 0;
        }
    } else if (arc_candidate_is_object_program(candidate)) {
        NNProgram components;
        NNProgram largest;
        NNProgram renderer;
        NNProgram selected;
        nn_program_init(&components,
                        "arc_agi",
                        "connected_components",
                        NN_PROGRAM_KIND_TRANSFORM,
                        NN_PROGRAM_VALUE_GRID,
                        NN_PROGRAM_VALUE_OBJECT_SET,
                        100);
        nn_program_add_param(&components,
                             "background",
                             (double)candidate->background_color,
                             candidate->background_color);
        nn_program_init(&largest,
                        "arc_agi",
                        candidate->selector_kind == ARC_OBJECT_SELECTOR_ALL_COLOR
                            ? "largest_component"
                            : "select_object_by_selector",
                        NN_PROGRAM_KIND_SELECTOR,
                        NN_PROGRAM_VALUE_OBJECT_SET,
                        NN_PROGRAM_VALUE_OBJECT,
                        101);
        nn_program_add_param(&largest,
                             "selector_kind",
                             (double)arc_candidate_selector_kind(candidate),
                             arc_candidate_selector_kind(candidate));
        nn_program_add_param(&largest,
                             "color",
                             (double)candidate->input_color,
                             candidate->input_color);
        nn_program_add_param(&largest,
                             "reference_color",
                             (double)candidate->reference_color,
                             candidate->reference_color);
        nn_program_init(&renderer,
                        "arc_agi",
                        candidate->op == ARC_RULE_OBJECT_MASK_LARGEST ? "object_mask" : "object_crop",
                        NN_PROGRAM_KIND_TRANSFORM,
                        NN_PROGRAM_VALUE_OBJECT,
                        NN_PROGRAM_VALUE_GRID,
                        candidate->op == ARC_RULE_OBJECT_MASK_LARGEST ? 102u : 103u);
        nn_program_add_param(&renderer,
                             "output_color",
                             (double)candidate->output_color,
                             candidate->output_color);
        if (!nn_program_compose(&selected,
                                &components,
                                &largest,
                                "connected_components_then_largest") ||
            !nn_program_compose(program, &selected, &renderer, candidate->name)) {
            return 0;
        }
    } else {
        nn_program_init(program,
                        "arc_agi",
                        candidate->name,
                        NN_PROGRAM_KIND_TRANSFORM,
                        NN_PROGRAM_VALUE_GRID,
                        NN_PROGRAM_VALUE_GRID,
                        (uint32_t)candidate->op);
    }

    nn_program_add_param(program, "op", (double)candidate->op, (int64_t)candidate->op);
    nn_program_add_param(program,
                         "input_color",
                         (double)candidate->input_color,
                         candidate->input_color);
    nn_program_add_param(program,
                         "output_color",
                         (double)candidate->output_color,
                         candidate->output_color);
    nn_program_add_param(program,
                         "background",
                         (double)candidate->background_color,
                         candidate->background_color);
    nn_program_add_param(program,
                         "selector_kind",
                         (double)arc_candidate_selector_kind(candidate),
                         arc_candidate_selector_kind(candidate));
    nn_program_add_param(program,
                         "output_selector_kind",
                         (double)arc_candidate_output_selector_kind(candidate),
                         arc_candidate_output_selector_kind(candidate));
    nn_program_add_param(program,
                         "reference_color",
                         (double)candidate->reference_color,
                         candidate->reference_color);
    nn_program_add_param(program,
                         "render_mode",
                         (double)arc_candidate_render_mode(candidate),
                         arc_candidate_render_mode(candidate));
    nn_program_add_param(program, "dx", (double)candidate->dx, candidate->dx);
    nn_program_add_param(program, "dy", (double)candidate->dy, candidate->dy);
    nn_program_add_param(program, "scale", (double)candidate->scale, (int64_t)candidate->scale);
    if (candidate->op == ARC_RULE_GRID_COLOR_MAP) {
        for (uint8_t color = 0; color < ARC_COLOR_COUNT; ++color) {
            char name[16];
            snprintf(name, sizeof(name), "map_%u", (unsigned)color);
            nn_program_add_param(program,
                                 name,
                                 (double)candidate->color_map[color],
                                 candidate->color_map[color]);
        }
    }
    nn_program_add_feature(program, (nn_real)((double)candidate->op / 39.0));
    nn_program_add_feature(program, (nn_real)((double)arc_candidate_family(candidate) / 17.0));
    nn_program_add_feature(program, (nn_real)((double)candidate->background_color / 9.0));
    nn_program_add_feature(program, (nn_real)((double)candidate->output_color / 9.0));
    nn_program_add_feature(program, (nn_real)(arc_candidate_is_object_program(candidate) ? 1.0 : 0.0));
    nn_program_add_feature(program, (nn_real)((double)(abs(candidate->dx) + abs(candidate->dy)) / 8.0));
    nn_program_add_feature(program, (nn_real)((double)candidate->scale / 4.0));
    nn_program_add_feature(program, (nn_real)(candidate->input_color == candidate->output_color ? 1.0 : 0.0));
    nn_program_add_feature(program, (nn_real)((double)candidate->input_color / 9.0));
    nn_program_add_feature(program,
                           (nn_real)((double)candidate->dx / (double)ARC_MAX_GRID_WIDTH));
    nn_program_add_feature(program,
                           (nn_real)((double)candidate->dy / (double)ARC_MAX_GRID_HEIGHT));
    nn_program_add_feature(program, (nn_real)(arc_candidate_is_object_relation(candidate) ? 1.0 : 0.0));
    nn_program_add_feature(program,
                           (nn_real)((double)arc_candidate_selector_kind(candidate) /
                                     (double)(ARC_OBJECT_SELECTOR_KIND_COUNT - 1)));
    nn_program_add_feature(
        program,
        (nn_real)((double)arc_candidate_output_selector_kind(candidate) /
                  (double)(ARC_OBJECT_SELECTOR_KIND_COUNT - 1)));
    nn_program_add_feature(program,
                           (nn_real)((double)arc_candidate_render_mode(candidate) /
                                     (double)(ARC_OBJECT_RENDER_KIND_COUNT - 1)));
    nn_program_add_feature(program, (nn_real)((double)candidate->reference_color / 9.0));
    nn_program_refresh_canonical_id(program);
    return 1;
}

static int arc_encode_program_features(const ARCTransformCandidate *candidate,
                                       const ARCProgramTaskFeatures *task_features,
                                       nn_real *features,
                                       size_t capacity)
{
    if (candidate == NULL || task_features == NULL || features == NULL ||
        capacity < ARC_PROGRAM_FEATURE_COUNT) {
        return 0;
    }
    NNProgram program;
    size_t encoded_count = 0;
    if (!arc_candidate_to_program(candidate, &program) ||
        !nn_program_encode_features(&program,
                                    features,
                                    NN_PROGRAM_FEATURE_COUNT,
                                    &encoded_count) ||
        encoded_count != NN_PROGRAM_FEATURE_COUNT) {
        return 0;
    }
    arc_program_task_features_write(task_features,
                                    &features[NN_PROGRAM_FEATURE_COUNT]);
    return 1;
}

static int arc_build_program_feature_matrix(const ARCTransformCandidate *candidates,
                                            size_t candidate_count,
                                            const ARCProgramTaskFeatures *task_features,
                                            nn_real *features,
                                            size_t feature_count)
{
    if (feature_count != ARC_PROGRAM_FEATURE_COUNT) {
        return 0;
    }
    for (size_t i = 0; i < candidate_count; ++i) {
        if (!arc_encode_program_features(&candidates[i],
                                         task_features,
                                         &features[i * feature_count],
                                         feature_count)) {
            return 0;
        }
    }
    return 1;
}

static size_t arc_evaluate_program_candidates(const ARCTask *task,
                                              const ARCTransformCandidate *candidates,
                                              size_t candidate_count,
                                              ARCProgramEval *evals)
{
    ARCGrid predictions[ARC_MAX_PAIRS];
    size_t task_cells = arc_task_test_cells(task);
    for (size_t i = 0; i < candidate_count; ++i) {
        NNProgram program;
        memset(&evals[i], 0, sizeof(evals[i]));
        if (arc_candidate_to_program(&candidates[i], &program)) {
            evals[i].program_hash = program.canonical_hash;
            evals[i].object_program = arc_candidate_is_object_program(&candidates[i]);
        }
        evals[i].train_consistent =
            arc_candidate_matches_train(task,
                                        &candidates[i],
                                        &evals[i].train_checked_cells,
                                        &evals[i].train_mismatches,
                                        &evals[i].unsupported_train_pairs);
        if (!evals[i].train_consistent) {
            continue;
        }
        evals[i].test_supported =
            arc_apply_candidate_to_tests(task,
                                         &candidates[i],
                                         predictions,
                                         &evals[i].unsupported_test_pairs);
        if (!evals[i].test_supported) {
            continue;
        }
        evals[i].prediction_hash = arc_prediction_hash(predictions, task->test_count);
        evals[i].test_cells = task_cells;
        for (size_t pair = 0; pair < task->test_count; ++pair) {
            if (!task->test[pair].has_output ||
                !arc_grid_same_shape(&predictions[pair], &task->test[pair].output)) {
                continue;
            }
            size_t cells = arc_grid_cells(&task->test[pair].output);
            for (size_t cell = 0; cell < cells; ++cell) {
                evals[i].test_correct_cells +=
                    predictions[pair].cells[cell] == task->test[pair].output.cells[cell] ? 1u : 0u;
            }
        }
    }
    return candidate_count;
}

static NNProgramProposerStrategy arc_program_strategy_to_nn(ARCProgramStrategy strategy)
{
    switch (strategy) {
    case ARC_PROGRAM_STRATEGY_GRAMMAR:
        return NN_PROGRAM_PROPOSER_GRAMMAR_ORDER;
    case ARC_PROGRAM_STRATEGY_STRUCTURAL:
        return NN_PROGRAM_PROPOSER_STRUCTURAL_HEURISTIC;
    case ARC_PROGRAM_STRATEGY_RANDOM:
        return NN_PROGRAM_PROPOSER_RANDOM;
    case ARC_PROGRAM_STRATEGY_LEARNED:
        return NN_PROGRAM_PROPOSER_LEARNED;
    case ARC_PROGRAM_STRATEGY_TRACE_PRIOR:
        return NN_PROGRAM_PROPOSER_TRACE_PRIOR;
    case ARC_PROGRAM_STRATEGY_TRACE_TASK_PRIOR:
        return NN_PROGRAM_PROPOSER_TRACE_TASK_PRIOR;
    case ARC_PROGRAM_STRATEGY_TRACE_LINEAR:
        return NN_PROGRAM_PROPOSER_TRACE_LINEAR;
    case ARC_PROGRAM_STRATEGY_TRACE_MLP:
        return NN_PROGRAM_PROPOSER_TRACE_MLP;
    case ARC_PROGRAM_STRATEGY_TRACE_HOLO_RESIDUAL:
        return NN_PROGRAM_PROPOSER_TRACE_HOLO_RESIDUAL;
    case ARC_PROGRAM_STRATEGY_TRACE_HOLO_BUCKET:
        return NN_PROGRAM_PROPOSER_TRACE_HOLO_BUCKET;
    case ARC_PROGRAM_STRATEGY_ORACLE:
        return NN_PROGRAM_PROPOSER_ORACLE;
    }
    return NN_PROGRAM_PROPOSER_GRAMMAR_ORDER;
}

static uint64_t arc_task_hash(const ARCTask *task)
{
    uint64_t hash = UINT64_C(1469598103934665603);
    for (const unsigned char *p = (const unsigned char *)task->id; *p != '\0'; ++p) {
        hash = arc_hash_mix(hash, *p);
    }
    return hash;
}

static NeuralNet *create_arc_program_model(unsigned int seed)
{
    static const size_t layers[] = {ARC_PROGRAM_FEATURE_COUNT, 64, 32, 1};
    NNConfig config;
    config.layer_sizes = layers;
    config.layer_count = sizeof(layers) / sizeof(layers[0]);
    config.seed = seed;
    config.hidden_activation = NN_ACTIVATION_RELU;
    config.output_activation = NN_ACTIVATION_LINEAR;
    config.loss = NN_LOSS_MSE;
    return nn_create_config(&config);
}

static NeuralNet *create_arc_trace_mlp_model(unsigned int seed)
{
    static const size_t layers[] = {ARC_PROGRAM_FEATURE_COUNT, 32, 16, 1};
    NNConfig config;
    config.layer_sizes = layers;
    config.layer_count = sizeof(layers) / sizeof(layers[0]);
    config.seed = seed ^ 0x517cc1abu;
    config.hidden_activation = NN_ACTIVATION_RELU;
    config.output_activation = NN_ACTIVATION_LINEAR;
    config.loss = NN_LOSS_MSE;
    return nn_create_config(&config);
}

static int arc_proposal_trace_prior_init(ARCProposalTracePrior *prior, unsigned int seed)
{
    if (prior == NULL) {
        return 0;
    }
    memset(prior, 0, sizeof(*prior));
    prior->mlp = create_arc_trace_mlp_model(seed);
    if (prior->mlp == NULL) {
        return 0;
    }
    prior->holo_residual = holo_ranker_create(ARC_PROGRAM_FEATURE_COUNT);
    if (prior->holo_residual == NULL) {
        nn_free(prior->mlp);
        prior->mlp = NULL;
        return 0;
    }
    prior->holo_pressure =
        holo_partitioned_ranker_create(ARC_PROGRAM_FEATURE_COUNT,
                                       32u,
                                       ((uint64_t)seed << 32) ^ UINT64_C(0x484f4c4f50524553));
    if (prior->holo_pressure == NULL) {
        holo_ranker_destroy(prior->holo_residual);
        prior->holo_residual = NULL;
        nn_free(prior->mlp);
        prior->mlp = NULL;
        return 0;
    }
    return 1;
}

static void arc_proposal_trace_prior_free(ARCProposalTracePrior *prior)
{
    if (prior == NULL) {
        return;
    }
    nn_free(prior->mlp);
    prior->mlp = NULL;
    holo_ranker_destroy(prior->holo_residual);
    prior->holo_residual = NULL;
    holo_partitioned_ranker_destroy(prior->holo_pressure);
    prior->holo_pressure = NULL;
}

static NeuralNet *create_arc_hypothesis_prior_model(unsigned int seed, size_t hidden)
{
    size_t layers[3] = {ARC_HYPOTHESIS_PRIOR_FEATURE_COUNT, hidden, 1};
    NNConfig config;
    config.layer_sizes = layers;
    config.layer_count = hidden == 0 ? 2u : 3u;
    if (hidden == 0) {
        layers[1] = 1;
    }
    config.seed = seed ^ 0x9e43a71bu;
    config.hidden_activation = NN_ACTIVATION_RELU;
    config.output_activation = NN_ACTIVATION_LINEAR;
    config.loss = NN_LOSS_MSE;
    return nn_create_config(&config);
}

static int arc_program_learned_save_state(const ARCProgramLearnedState *state, const char *path) {
    /* Save fixed-size learned state for multipass training.
       Skips dynamic generative_priors (rebuilt per run). */
    FILE *f = fopen(path, "wb"); fprintf(stderr, "save_state: path=%s f=%p\n", path, (void*)f);
    if (!f) return 0;
    uint32_t magic = 0x43524C50; /* "CRLP" */
    uint32_t version = 1;
    if (fwrite(&magic, sizeof(magic), 1, f) != 1 ||
        fwrite(&version, sizeof(version), 1, f) != 1) { fclose(f); return 0; }
    /* Write fixed-size prefix (everything before generative_priors pointer) */
    size_t prefix_size = offsetof(ARCProgramLearnedState, generative_priors);
    if (fwrite(state, prefix_size, 1, f) != 1) { fclose(f); return 0; }
    /* Write generative_prior_count (data is not saved, rebuilt on load) */
    size_t zero = 0;
    if (fwrite(&zero, sizeof(zero), 1, f) != 1) { fclose(f); return 0; }
    fclose(f);
    return 1;
}

static int arc_program_learned_load_state(ARCProgramLearnedState *state, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    uint32_t magic = 0, version = 0;
    if (fread(&magic, 4, 1, f) != 1 || fread(&version, 4, 1, f) != 1 ||
        magic != 0x43524C50 || version != 1) { fclose(f); return 0; }
    if (fread(state->family_score, sizeof(state->family_score), 1, f) != 1) { fclose(f); return 0; }
    if (fread(state->relation_score, sizeof(state->relation_score), 1, f) != 1) { fclose(f); return 0; }
    if (fread(state->relation_color_score, sizeof(state->relation_color_score), 1, f) != 1) { fclose(f); return 0; }
    if (fread(state->relation_background_score, sizeof(state->relation_background_score), 1, f) != 1) { fclose(f); return 0; }
    if (fread(state->relation_selector_score, sizeof(state->relation_selector_score), 1, f) != 1) { fclose(f); return 0; }
    if (fread(state->relation_output_selector_score, sizeof(state->relation_output_selector_score), 1, f) != 1) { fclose(f); return 0; }
    if (fread(state->relation_render_score, sizeof(state->relation_render_score), 1, f) != 1) { fclose(f); return 0; }
    if (fread(state->object_selector_score, sizeof(state->object_selector_score), 1, f) != 1) { fclose(f); return 0; }
    if (fread(state->object_color_score, sizeof(state->object_color_score), 1, f) != 1) { fclose(f); return 0; }
    if (fread(state->object_render_score, sizeof(state->object_render_score), 1, f) != 1) { fclose(f); return 0; }
    state->generative_priors = NULL;
    state->generative_prior_capacity = 0;
    state->generative_prior_count = 0;
    fclose(f);
    return 1;
}

static int arc_program_learned_init(ARCProgramLearnedState *state,
                                    const ARCConfig *config,
                                    unsigned int seed)
{
    memset(state, 0, sizeof(*state));
    state->hypothesis_prior_hidden =
        config == NULL ? ARC_HYPOTHESIS_PRIOR_DEFAULT_HIDDEN
                       : config->hypothesis_prior_hidden;
    state->hypothesis_top_k =
        config == NULL ? ARC_HYPOTHESIS_TOP_K : config->hypothesis_top_k;
    state->arc_runtime_sketch_mine_cap =
        config == NULL ? 16u : config->arc_runtime_sketch_mine_cap;
    state->arc_generative_prior_op_cap =
        config == NULL ? 2u : config->arc_generative_prior_op_cap;
    state->hypothesis_neural_prior_enabled =
        config == NULL ? 1 : config->hypothesis_neural_prior_enabled;
    state->hypothesis_slot_prior_enabled =
        config == NULL ? 1 : config->hypothesis_slot_prior_enabled;
    state->hypothesis_cold_start_enabled =
        config == NULL ? 1 : config->hypothesis_cold_start_enabled;
    state->hypothesis_structural_score_enabled =
        config == NULL ? 1 : config->hypothesis_structural_score_enabled;
    state->hypothesis_coverage_selection_enabled =
        config == NULL ? 1 : config->hypothesis_coverage_selection_enabled;
    state->hypothesis_train_filter_enabled =
        config == NULL ? 1 : config->hypothesis_train_filter_enabled;
    state->hypothesis_train_consistency_prior_enabled =
        config == NULL ? 1 : config->hypothesis_train_consistency_prior_enabled;
    state->arc_visual_only = config == NULL ? 0 : config->arc_visual_only;
    state->net = create_arc_program_model(seed);
    if (state->hypothesis_neural_prior_enabled) {
        state->hypothesis_prior_net =
            create_arc_hypothesis_prior_model(seed, state->hypothesis_prior_hidden);
    }
    if (state->net == NULL ||
        (state->hypothesis_neural_prior_enabled &&
         state->hypothesis_prior_net == NULL)) {
        nn_free(state->net);
        nn_free(state->hypothesis_prior_net);
        memset(state, 0, sizeof(*state));
        return 0;
    }
    return 1;
}

static void arc_program_learned_free(ARCProgramLearnedState *state)
{
    if (state == NULL) {
        return;
    }
    nn_free(state->net);
    nn_free(state->hypothesis_prior_net);
    free(state->generative_priors);
    memset(state, 0, sizeof(*state));
}

static size_t arc_hypothesis_template_bucket(uint32_t template_id)
{
    switch (template_id) {
    case 1024:
        return 0;
    case 1025:
        return 1;
    case 1026:
        return 2;
    case 1027:
        return 3;
    case 1028:
        return 4;
    case 1029:
        return 5;
    case 1030:
        return 6;
    case 1031:
        return 7;
    case 1032:
        return 8;
    case 1033:
        return 9;
    case 1034:
        return 10;
    case 1035:
        return 11;
    default:
        return (size_t)(template_id % ARC_HYPOTHESIS_TEMPLATE_BUCKETS);
    }
}

static size_t arc_hypothesis_slot_type_bucket(NNHypothesisSlotType type)
{
    switch (type) {
    case NN_HYPOTHESIS_SLOT_INTEGER:
    case NN_HYPOTHESIS_SLOT_COLOR:
    case NN_HYPOTHESIS_SLOT_OFFSET:
    case NN_HYPOTHESIS_SLOT_SELECTOR_KIND:
    case NN_HYPOTHESIS_SLOT_RELATION_KIND:
    case NN_HYPOTHESIS_SLOT_RENDER_MODE:
    case NN_HYPOTHESIS_SLOT_REFERENCE_COLOR:
    case NN_HYPOTHESIS_SLOT_SCALAR:
    case NN_HYPOTHESIS_SLOT_BOOLEAN:
        return (size_t)type;
    case NN_HYPOTHESIS_SLOT_CUSTOM:
    default:
        return 0;
    }
}

static uint64_t arc_hypothesis_prior_mix(uint64_t hash, uint64_t value)
{
    hash ^= value + UINT64_C(0x9e3779b97f4a7c15) + (hash << 6) + (hash >> 2);
    return hash;
}

static size_t arc_hypothesis_slot_value_bucket_for(uint32_t template_id,
                                                   const char *slot_name,
                                                   NNHypothesisSlotType slot_type,
                                                   int64_t slot_integer)
{
    uint64_t hash = template_id;
    hash = arc_hypothesis_prior_mix(hash, (uint64_t)slot_type);
    hash = arc_hypothesis_prior_mix(hash, (uint64_t)slot_integer);
    for (const unsigned char *p = (const unsigned char *)slot_name;
         p != NULL && *p != '\0';
         ++p) {
        hash = arc_hypothesis_prior_mix(hash, *p);
    }
    return (size_t)(hash % ARC_HYPOTHESIS_SLOT_VALUE_BUCKETS);
}

static size_t arc_hypothesis_slot_value_bucket(const NNHypothesis *hypothesis,
                                               const NNHypothesisSlotValue *slot)
{
    uint32_t template_id = hypothesis != NULL && hypothesis->template_ref != NULL
                               ? hypothesis->template_ref->template_id
                               : 0;
    return arc_hypothesis_slot_value_bucket_for(template_id,
                                                slot->name,
                                                slot->type,
                                                slot->integer);
}

static int arc_hypothesis_slot_type_from_name(const char *name,
                                              NNHypothesisSlotType *type_out)
{
    if (name == NULL || type_out == NULL) {
        return 0;
    }
    if (strcmp(name, "integer") == 0) {
        *type_out = NN_HYPOTHESIS_SLOT_INTEGER;
    } else if (strcmp(name, "color") == 0) {
        *type_out = NN_HYPOTHESIS_SLOT_COLOR;
    } else if (strcmp(name, "offset") == 0) {
        *type_out = NN_HYPOTHESIS_SLOT_OFFSET;
    } else if (strcmp(name, "selector_kind") == 0) {
        *type_out = NN_HYPOTHESIS_SLOT_SELECTOR_KIND;
    } else if (strcmp(name, "relation_kind") == 0) {
        *type_out = NN_HYPOTHESIS_SLOT_RELATION_KIND;
    } else if (strcmp(name, "render_mode") == 0) {
        *type_out = NN_HYPOTHESIS_SLOT_RENDER_MODE;
    } else if (strcmp(name, "reference_color") == 0) {
        *type_out = NN_HYPOTHESIS_SLOT_REFERENCE_COLOR;
    } else if (strcmp(name, "scalar") == 0) {
        *type_out = NN_HYPOTHESIS_SLOT_SCALAR;
    } else if (strcmp(name, "boolean") == 0) {
        *type_out = NN_HYPOTHESIS_SLOT_BOOLEAN;
    } else if (strcmp(name, "custom") == 0) {
        *type_out = NN_HYPOTHESIS_SLOT_CUSTOM;
    } else {
        return 0;
    }
    return 1;
}

static int arc_exact_sketch_task_id_from_library_id(const char *library_id,
                                                    char *task_id,
                                                    size_t task_id_size)
{
    if (library_id == NULL || task_id == NULL || task_id_size == 0 ||
        strncmp(library_id, "ngd:", 4) != 0) {
        return 0;
    }
    const char *family_start = library_id + 4;
    const char *task_start = strchr(family_start, ':');
    if (task_start == NULL) {
        return 0;
    }
    ++task_start;
    const char *task_end = strchr(task_start, ':');
    if (task_end == NULL || task_end == task_start) {
        return 0;
    }
    size_t len = (size_t)(task_end - task_start);
    if (len >= task_id_size) {
        len = task_id_size - 1u;
    }
    memcpy(task_id, task_start, len);
    task_id[len] = '\0';
    return len > 0;
}

static int arc_exact_sketch_slot_bit(const char *slot_name, unsigned int *bit_out)
{
    static const char *slots[ARC_EXACT_SKETCH_PRIOR_SLOT_COUNT] = {
        "sketch.selector",
        "sketch.op",
        "sketch.background",
        "sketch.color",
        "sketch.output_color",
        "sketch.x_anchor",
        "sketch.y_anchor",
        "sketch.x_offset",
        "sketch.y_offset",
        "sketch.width",
        "sketch.height",
        "sketch.period",
    };
    if (slot_name == NULL || bit_out == NULL) {
        return 0;
    }
    for (unsigned int i = 0; i < ARC_EXACT_SKETCH_PRIOR_SLOT_COUNT; ++i) {
        if (strcmp(slot_name, slots[i]) == 0) {
            *bit_out = i;
            return 1;
        }
    }
    return 0;
}

static ARCExactSketchPrior *arc_exact_sketch_prior_find_or_add(
    ARCProgramLearnedState *state,
    const char *library_id)
{
    char task_id[ARC_MAX_TASK_ID];
    if (state == NULL || library_id == NULL ||
        !arc_exact_sketch_task_id_from_library_id(library_id,
                                                  task_id,
                                                  sizeof(task_id))) {
        return NULL;
    }
    for (size_t i = 0; i < state->exact_sketch_prior_count; ++i) {
        if (strcmp(state->exact_sketch_priors[i].library_id, library_id) == 0) {
            return &state->exact_sketch_priors[i];
        }
    }
    if (state->exact_sketch_prior_count >= ARC_EXACT_SKETCH_PRIOR_CAPACITY) {
        return NULL;
    }
    ARCExactSketchPrior *prior =
        &state->exact_sketch_priors[state->exact_sketch_prior_count++];
    memset(prior, 0, sizeof(*prior));
    snprintf(prior->library_id, sizeof(prior->library_id), "%s", library_id);
    snprintf(prior->task_id, sizeof(prior->task_id), "%s", task_id);
    return prior;
}

static void arc_exact_sketch_prior_observe_slot(ARCProgramLearnedState *state,
                                                const char *library_id,
                                                const char *slot_name,
                                                int64_t slot_integer,
                                                size_t proposal_count,
                                                size_t certified_count,
                                                double utility_mean)
{
    unsigned int bit = 0;
    if (certified_count == 0 ||
        !arc_exact_sketch_slot_bit(slot_name, &bit)) {
        return;
    }
    ARCExactSketchPrior *prior =
        arc_exact_sketch_prior_find_or_add(state, library_id);
    if (prior == NULL) {
        return;
    }
    switch (bit) {
    case 0:
        prior->spec.selector = (int)slot_integer;
        break;
    case 1:
        prior->spec.op = (int)slot_integer;
        break;
    case 2:
        prior->spec.background = (uint8_t)slot_integer;
        break;
    case 3:
        prior->spec.color = (uint8_t)slot_integer;
        break;
    case 4:
        prior->spec.output_color = (uint8_t)slot_integer;
        break;
    case 5:
        prior->spec.x_anchor = (int)slot_integer;
        break;
    case 6:
        prior->spec.y_anchor = (int)slot_integer;
        break;
    case 7:
        prior->spec.x_offset = (int)slot_integer;
        break;
    case 8:
        prior->spec.y_offset = (int)slot_integer;
        break;
    case 9:
        prior->spec.width = slot_integer < 0 ? 0u : (size_t)slot_integer;
        break;
    case 10:
        prior->spec.height = slot_integer < 0 ? 0u : (size_t)slot_integer;
        break;
    case 11:
        prior->spec.period = slot_integer < 0 ? 0u : (size_t)slot_integer;
        break;
    default:
        return;
    }
    prior->seen_slot_mask |= (1u << bit);
    prior->proposal_count += proposal_count;
    prior->certified_count += certified_count;
    if (utility_mean > prior->utility_mean) {
        prior->utility_mean = utility_mean;
    }
}

static void arc_hypothesis_seed_slot_prior(ARCProgramLearnedState *state,
                                           uint32_t template_id,
                                           const char *slot_name,
                                           NNHypothesisSlotType slot_type,
                                           int64_t slot_integer,
                                           size_t proposal_count,
                                           size_t certified_count)
{
    if (state == NULL || slot_name == NULL || proposal_count == 0) {
        return;
    }
    if (certified_count > proposal_count) {
        certified_count = proposal_count;
    }
    size_t template_bucket = arc_hypothesis_template_bucket(template_id);
    size_t type_bucket = arc_hypothesis_slot_type_bucket(slot_type);
    size_t value_bucket = arc_hypothesis_slot_value_bucket_for(template_id,
                                                               slot_name,
                                                               slot_type,
                                                               slot_integer);
    state->hypothesis_slot_proposal_count[template_bucket]
                                         [type_bucket]
                                         [value_bucket] += proposal_count;
    state->hypothesis_slot_certified_count[template_bucket]
                                          [type_bucket]
                                          [value_bucket] += certified_count;
    ++state->hypothesis_library_slot_rows;
    state->hypothesis_library_seeded_proposals += proposal_count;
    state->hypothesis_library_seeded_certified += certified_count;
}

static void arc_hypothesis_seed_template_prior(ARCProgramLearnedState *state,
                                               uint32_t template_id,
                                               size_t proposal_count,
                                               size_t certified_count)
{
    if (state == NULL || proposal_count == 0) {
        return;
    }
    if (certified_count > proposal_count) {
        certified_count = proposal_count;
    }
    size_t template_bucket = arc_hypothesis_template_bucket(template_id);
    state->hypothesis_template_proposal_count[template_bucket] += proposal_count;
    state->hypothesis_template_certified_count[template_bucket] += certified_count;
    ++state->hypothesis_library_template_rows;
    state->hypothesis_library_seeded_proposals += proposal_count;
    state->hypothesis_library_seeded_certified += certified_count;
}

static void arc_hypothesis_seed_template_probe(ARCProgramLearnedState *state,
                                               uint32_t template_id,
                                               size_t proposal_count)
{
    if (state == NULL || proposal_count == 0) {
        return;
    }
    size_t template_bucket = arc_hypothesis_template_bucket(template_id);
    state->hypothesis_template_probe_count[template_bucket] += proposal_count;
    ++state->hypothesis_probe_template_rows;
    state->hypothesis_probe_seeded_proposals += proposal_count;
}

static void arc_hypothesis_seed_slot_probe(ARCProgramLearnedState *state,
                                           uint32_t template_id,
                                           const char *library_id,
                                           const char *slot_name,
                                           NNHypothesisSlotType slot_type,
                                           double slot_value,
                                           int64_t slot_integer,
                                           size_t proposal_count)
{
    if (state == NULL || library_id == NULL || slot_name == NULL ||
        proposal_count == 0) {
        return;
    }
    size_t template_bucket = arc_hypothesis_template_bucket(template_id);
    size_t type_bucket = arc_hypothesis_slot_type_bucket(slot_type);
    size_t value_bucket = arc_hypothesis_slot_value_bucket_for(template_id,
                                                               slot_name,
                                                               slot_type,
                                                               slot_integer);
    state->hypothesis_slot_probe_count[template_bucket]
                                      [type_bucket]
                                      [value_bucket] += proposal_count;
    ++state->hypothesis_probe_slot_rows;
    state->hypothesis_probe_slot_seeded_proposals += proposal_count;
    if (state->hypothesis_slot_probe_row_count <
        ARC_HYPOTHESIS_SLOT_PROBE_CAPACITY) {
        size_t index = state->hypothesis_slot_probe_row_count++;
        state->hypothesis_slot_probe_rows[index].template_id = template_id;
        snprintf(state->hypothesis_slot_probe_rows[index].library_id,
                 sizeof(state->hypothesis_slot_probe_rows[index].library_id),
                 "%s",
                 library_id);
        snprintf(state->hypothesis_slot_probe_rows[index].slot_name,
                 sizeof(state->hypothesis_slot_probe_rows[index].slot_name),
                 "%s",
                 slot_name);
        state->hypothesis_slot_probe_rows[index].slot_type = slot_type;
        state->hypothesis_slot_probe_rows[index].slot_value = slot_value;
        state->hypothesis_slot_probe_rows[index].slot_integer = slot_integer;
        state->hypothesis_slot_probe_rows[index].proposal_count = proposal_count;
    }
}

static size_t arc_split_tab_fields(char *line, char **fields, size_t capacity)
{
    if (line == NULL || fields == NULL || capacity == 0) {
        return 0;
    }
    line[strcspn(line, "\r\n")] = '\0';
    size_t count = 0;
    char *cursor = line;
    while (count < capacity) {
        fields[count++] = cursor;
        char *tab = strchr(cursor, '\t');
        if (tab == NULL) {
            break;
        }
        *tab = '\0';
        cursor = tab + 1;
    }
    return count;
}

static int arc_program_learned_load_hypothesis_library_prior(
    ARCProgramLearnedState *state,
    const char *path)
{
    if (state == NULL || path == NULL || path[0] == '\0') {
        return 0;
    }
    FILE *fp = fopen(path, "r");
    if (fp == NULL) {
        fprintf(stderr, "failed to open hypothesis library prior: %s\n", path);
        return 0;
    }

    char line[2048];
    char last_library_id[ARC_TRANSFORM_NAME_BYTES];
    last_library_id[0] = '\0';
    size_t line_no = 0;
    while (fgets(line, sizeof(line), fp) != NULL) {
        ++line_no;
        if (line[0] == '\0' || line[0] == '\n' || line[0] == '#') {
            continue;
        }
        if (strncmp(line, "schema_version\t", strlen("schema_version\t")) == 0) {
            continue;
        }
        char *fields[13];
        size_t field_count = arc_split_tab_fields(line, fields, sizeof(fields) / sizeof(fields[0]));
        int is_slot_prior = strcmp(fields[0], "crlp.hypothesis_library.prior.v1") == 0;
        int is_template_prior =
            strcmp(fields[0], "crlp.hypothesis_library.template_prior.v1") == 0;
        int is_template_probe =
            strcmp(fields[0], "crlp.arc_neural_discovery.template_probe.v1") == 0;
        int is_slot_probe =
            strcmp(fields[0], "crlp.arc_neural_discovery.slot_probe.v1") == 0;
        if (field_count != 13 ||
            (!is_slot_prior && !is_template_prior && !is_template_probe &&
             !is_slot_probe)) {
            fprintf(stderr,
                    "invalid hypothesis library prior row at %s:%zu\n",
                    path,
                    line_no);
            fclose(fp);
            return 0;
        }

        size_t template_id_size = 0;
        int64_t slot_integer = 0;
        double slot_value = 0.0;
        double utility_mean = 0.0;
        size_t proposal_count = 0;
        size_t certified_count = 0;
        NNHypothesisSlotType slot_type = NN_HYPOTHESIS_SLOT_CUSTOM;
        if (!nn_parse_size_text(fields[1], &template_id_size) ||
            !nn_parse_size_text(fields[8], &proposal_count) ||
            !nn_parse_size_nonnegative_text(fields[9], &certified_count) ||
            !nn_parse_double_text(fields[12], &utility_mean) ||
            ((is_slot_prior || is_slot_probe) &&
             (!arc_hypothesis_slot_type_from_name(fields[5], &slot_type) ||
              !nn_parse_double_text(fields[6], &slot_value) ||
              !nn_parse_i64_text(fields[7], &slot_integer)))) {
            fprintf(stderr,
                    "invalid hypothesis library prior fields at %s:%zu\n",
                    path,
                    line_no);
            fclose(fp);
            return 0;
        }

        if (is_template_probe) {
            arc_hypothesis_seed_template_probe(state,
                                               (uint32_t)template_id_size,
                                               proposal_count);
        } else if (is_slot_probe) {
            arc_hypothesis_seed_slot_probe(state,
                                           (uint32_t)template_id_size,
                                           fields[3],
                                           fields[4],
                                           slot_type,
                                           slot_value,
                                           slot_integer,
                                           proposal_count);
        } else if (is_template_prior) {
            arc_hypothesis_seed_template_prior(state,
                                               (uint32_t)template_id_size,
                                               proposal_count,
                                               certified_count);
        } else {
            arc_hypothesis_seed_slot_prior(state,
                                           (uint32_t)template_id_size,
                                           fields[4],
                                           slot_type,
                                           slot_integer,
                                           proposal_count,
                                           certified_count);
            if (template_id_size == 1036 && certified_count > 0) {
                arc_exact_sketch_prior_observe_slot(state,
                                                    fields[3],
                                                    fields[4],
                                                    slot_integer,
                                                    proposal_count,
                                                    certified_count,
                                                    utility_mean);
            }
        }
        if (!is_template_probe && !is_slot_probe &&
            strcmp(last_library_id, fields[3]) != 0) {
            snprintf(last_library_id, sizeof(last_library_id), "%s", fields[3]);
            ++state->hypothesis_library_entries;
        }
    }

    fclose(fp);
    return 1;
}

static void arc_hypothesis_template_prior_counts(const ARCProgramLearnedState *state, uint32_t template_id, size_t *proposals, size_t *certified);
