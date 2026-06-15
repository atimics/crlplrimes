                                                    const ARCSketchProgramSpec *spec)
{
    ARCTransformCandidate candidate;
    ARCGrid produced;
    if (task == NULL || spec == NULL || task->train_count == 0 ||
        !task->train[0].has_output ||
        !arc_candidate_from_sketch_program_values(&candidate, spec) ||
        !arc_apply_candidate(&candidate, &task->train[0].input, &produced)) {
        return 0;
    }
    return arc_grid_equal(&produced, &task->train[0].output);
}

static void arc_mine_macrocell_sketches(const ARCSketchProbeInput *input,
                                        const char *task_path,
                                        const ARCTask *task,
                                        ARCSketchProbeRecord *records,
                                        size_t *record_count,
                                        ARCSketchProgramSpec *emitted_specs,
                                        size_t *emitted_count,
                                        size_t sketches_per_task,
                                        ARCSketchProbeSummary *summary)
{
    if (task == NULL || task->train_count == 0) {
        return;
    }
    const ARCGrid *first_input = &task->train[0].input;
    const ARCGrid *first_output = &task->train[0].output;
    if (first_input->width == 0 || first_input->height == 0 ||
        first_output->width % first_input->width != 0 ||
        first_output->height % first_input->height != 0) {
        ++summary->unsupported_shape;
        return;
    }
    size_t scale_x = first_output->width / first_input->width;
    size_t scale_y = first_output->height / first_input->height;
    if (scale_x == 0 || scale_x != scale_y || scale_x > 6u) {
        ++summary->unsupported_shape;
        return;
    }
    for (size_t pair = 1; pair < task->train_count; ++pair) {
        if (task->train[pair].output.width != task->train[pair].input.width * scale_x ||
            task->train[pair].output.height != task->train[pair].input.height * scale_x) {
            ++summary->unsupported_shape;
            return;
        }
    }
    uint8_t backgrounds[ARC_COLOR_COUNT];
    size_t background_count =
        arc_task_relation_backgrounds(task,
                                      backgrounds,
                                      sizeof(backgrounds) / sizeof(backgrounds[0]));
    int has_zero_background = 0;
    int has_dominant_background = 0;
    uint8_t dominant_background = arc_grid_dominant_color(first_input);
    for (size_t i = 0; i < background_count; ++i) {
        if (backgrounds[i] == 0) {
            has_zero_background = 1;
        }
        if (backgrounds[i] == dominant_background) {
            has_dominant_background = 1;
        }
    }
    if (!has_zero_background && background_count < ARC_COLOR_COUNT) {
        backgrounds[background_count++] = 0;
    }
    if (!has_dominant_background && background_count < ARC_COLOR_COUNT) {
        backgrounds[background_count++] = dominant_background;
    }
    uint8_t colors[ARC_COLOR_COUNT];
    size_t color_count = arc_grid_nonzero_colors(first_input, colors, ARC_COLOR_COUNT);
    for (size_t background_index = 0;
         background_index < background_count;
         ++background_index) {
        uint8_t background = backgrounds[background_index];
        ARCSketchProgramSpec spec;
        memset(&spec, 0, sizeof(spec));
        spec.selector = ARC_SKETCH_SELECTOR_LATTICE;
        spec.op = ARC_SKETCH_OP_MACROCELL_SUBSTITUTE;
        spec.background = background;
        spec.color = background;
        spec.output_color = background;
        spec.period = scale_x;
        for (int motif_kind = 0; motif_kind <= 15; ++motif_kind) {
            spec.x_anchor = motif_kind;
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
        for (size_t color_index = 0; color_index < color_count; ++color_index) {
            uint8_t color = colors[color_index];
            if (color == background) {
                continue;
            }
            spec.color = color;
            spec.output_color = color;
            for (int motif_kind = 0; motif_kind <= 15; ++motif_kind) {
                spec.x_anchor = motif_kind;
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

static void arc_mine_lattice_sample_sketches(const ARCSketchProbeInput *input,
                                             const char *task_path,
                                             const ARCTask *task,
                                             ARCSketchProbeRecord *records,
                                             size_t *record_count,
                                             ARCSketchProgramSpec *emitted_specs,
                                             size_t *emitted_count,
                                             size_t sketches_per_task,
                                             ARCSketchProbeSummary *summary)
{
    if (task == NULL || task->train_count == 0) {
        return;
    }
    const ARCGrid *first_output = &task->train[0].output;
    for (size_t period = 2; period <= 6; ++period) {
        for (size_t x_offset = 0; x_offset < period; ++x_offset) {
            for (size_t y_offset = 0; y_offset < period; ++y_offset) {
                ARCSketchProgramSpec spec;
                memset(&spec, 0, sizeof(spec));
                spec.selector = ARC_SKETCH_SELECTOR_LATTICE;
                spec.op = ARC_SKETCH_OP_LATTICE_SAMPLE;
                spec.background = arc_grid_dominant_color(&task->train[0].input);
                spec.x_offset = (int)x_offset;
                spec.y_offset = (int)y_offset;
                spec.width = first_output->width;
                spec.height = first_output->height;
                spec.period = period;
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

static void arc_mine_diagonal_compact_sketches(const ARCSketchProbeInput *input,
                                               const char *task_path,
                                               const ARCTask *task,
                                               ARCSketchProbeRecord *records,
                                               size_t *record_count,
                                               ARCSketchProgramSpec *emitted_specs,
                                               size_t *emitted_count,
                                               size_t sketches_per_task,
                                               ARCSketchProbeSummary *summary)
{
    if (task == NULL || task->train_count == 0) {
        return;
    }
    const ARCGrid *first_input = &task->train[0].input;
    const ARCGrid *first_output = &task->train[0].output;
    if (first_input->width != first_input->height ||
        (first_input->width % 2u) == 0 ||
        first_output->width != 3u ||
        first_output->height != 3u) {
        ++summary->unsupported_shape;
        return;
    }
    uint8_t backgrounds[ARC_COLOR_COUNT];
    size_t background_count =
        arc_task_relation_backgrounds(task,
                                      backgrounds,
                                      sizeof(backgrounds) / sizeof(backgrounds[0]));
    int has_zero_background = 0;
    int has_dominant_background = 0;
    uint8_t dominant_background = arc_grid_dominant_color(first_input);
    for (size_t i = 0; i < background_count; ++i) {
        if (backgrounds[i] == 0) {
            has_zero_background = 1;
        }
        if (backgrounds[i] == dominant_background) {
            has_dominant_background = 1;
        }
    }
    if (!has_zero_background && background_count < ARC_COLOR_COUNT) {
        backgrounds[background_count++] = 0;
    }
    if (!has_dominant_background && background_count < ARC_COLOR_COUNT) {
        backgrounds[background_count++] = dominant_background;
    }
    for (size_t background_index = 0;
         background_index < background_count;
         ++background_index) {
        ARCSketchProgramSpec spec;
        memset(&spec, 0, sizeof(spec));
        spec.selector = ARC_SKETCH_SELECTOR_LATTICE;
        spec.op = ARC_SKETCH_OP_LATTICE_SAMPLE;
        spec.background = backgrounds[background_index];
        spec.width = first_output->width;
        spec.height = first_output->height;
        spec.x_anchor = 1;
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

static void arc_mine_point_line_project_sketches(const ARCSketchProbeInput *input,
                                                 const char *task_path,
                                                 const ARCTask *task,
                                                 ARCSketchProbeRecord *records,
                                                 size_t *record_count,
                                                 ARCSketchProgramSpec *emitted_specs,
                                                 size_t *emitted_count,
                                                 size_t sketches_per_task,
                                                 ARCSketchProbeSummary *summary)
{
    if (task == NULL || task->train_count == 0) {
        return;
    }
    const ARCGrid *first_input = &task->train[0].input;
    const ARCGrid *first_output = &task->train[0].output;
    if (!arc_grid_same_shape(first_input, first_output)) {
        ++summary->unsupported_shape;
        return;
    }
    uint8_t backgrounds[ARC_COLOR_COUNT];
    size_t background_count =
        arc_task_relation_backgrounds(task,
                                      backgrounds,
                                      sizeof(backgrounds) / sizeof(backgrounds[0]));
    if (background_count < ARC_COLOR_COUNT) {
        int has_zero = 0;
        for (size_t i = 0; i < background_count; ++i) {
            has_zero = has_zero || backgrounds[i] == 0;
        }
        if (!has_zero) {
            backgrounds[background_count++] = 0;
        }
    }
    uint8_t output_colors[ARC_COLOR_COUNT];
    size_t output_color_count =
        arc_grid_nonzero_colors(first_output, output_colors, ARC_COLOR_COUNT);
    arc_add_unique_color(output_colors,
                         ARC_COLOR_COUNT,
                         &output_color_count,
                         arc_grid_dominant_color(first_output));
    for (size_t background_index = 0;
         background_index < background_count;
         ++background_index) {
        for (size_t output_color_index = 0;
             output_color_index < output_color_count;
             ++output_color_index) {
            ARCSketchProgramSpec spec;
            memset(&spec, 0, sizeof(spec));
            spec.selector = ARC_SKETCH_SELECTOR_LATTICE;
            spec.op = ARC_SKETCH_OP_POINT_LINE_PROJECT;
            spec.background = backgrounds[background_index];
            spec.output_color = output_colors[output_color_index];
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

static void arc_mine_marker_polyline_sketches(const ARCSketchProbeInput *input,
                                              const char *task_path,
                                              const ARCTask *task,
                                              ARCSketchProbeRecord *records,
                                              size_t *record_count,
                                              ARCSketchProgramSpec *emitted_specs,
                                              size_t *emitted_count,
                                              size_t sketches_per_task,
                                              ARCSketchProbeSummary *summary)
{
    if (task == NULL || task->train_count == 0 || !task->train[0].has_output ||
        !arc_grid_same_shape(&task->train[0].input, &task->train[0].output)) {
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
    uint8_t output_colors[ARC_COLOR_COUNT];
    size_t output_color_count =
        arc_grid_nonzero_colors(first_output, output_colors, ARC_COLOR_COUNT);
    arc_add_unique_color(output_colors,
                         ARC_COLOR_COUNT,
                         &output_color_count,
                         arc_grid_most_common_non_background_color(first_output,
                                                                    arc_grid_dominant_color(first_output)));
    for (size_t background_index = 0;
         background_index < background_count;
         ++background_index) {
        for (size_t output_index = 0;
             output_index < output_color_count;
             ++output_index) {
            for (int order_mode = 0; order_mode <= 33; ++order_mode) {
                for (int route_mode = 0; route_mode <= 3; ++route_mode) {
                    ARCSketchProgramSpec spec;
                    memset(&spec, 0, sizeof(spec));
                    spec.selector = ARC_SKETCH_SELECTOR_LATTICE;
                    spec.op = ARC_SKETCH_OP_MARKER_POLYLINE;
                    spec.background = backgrounds[background_index];
                    spec.output_color = output_colors[output_index];
                    spec.x_anchor = order_mode;
                    spec.y_anchor = route_mode;
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

static void arc_mine_component_summary_sketches(const ARCSketchProbeInput *input,
                                                const char *task_path,
                                                const ARCTask *task,
                                                ARCSketchProbeRecord *records,
                                                size_t *record_count,
                                                ARCSketchProgramSpec *emitted_specs,
                                                size_t *emitted_count,
                                                size_t sketches_per_task,
                                                ARCSketchProbeSummary *summary)
{
    if (task == NULL || task->train_count == 0) {
        return;
    }
    uint8_t backgrounds[2];
    size_t background_count =
        arc_task_relation_backgrounds(task,
                                      backgrounds,
                                      sizeof(backgrounds) / sizeof(backgrounds[0]));
    const ARCGrid *first_input = &task->train[0].input;
    const ARCGrid *first_output = &task->train[0].output;
    uint8_t colors[ARC_COLOR_COUNT];
    size_t color_count = arc_grid_nonzero_colors(first_input, colors, ARC_COLOR_COUNT);
    uint8_t output_colors[ARC_COLOR_COUNT];
    size_t output_color_count =
        arc_grid_nonzero_colors(first_output, output_colors, ARC_COLOR_COUNT);
    arc_add_unique_color(output_colors,
                         ARC_COLOR_COUNT,
                         &output_color_count,
                         arc_grid_dominant_color(first_output));
    for (size_t background_index = 0;
         background_index < background_count;
         ++background_index) {
        uint8_t background = backgrounds[background_index];
        ARCSketchProgramSpec spec;
        memset(&spec, 0, sizeof(spec));
        spec.selector = ARC_SKETCH_SELECTOR_FOREGROUND_BBOX_PANEL;
        spec.op = ARC_SKETCH_OP_COMPONENT_COLOR_SUMMARY;
        spec.background = background;
        spec.color = background;
        spec.output_color = arc_grid_dominant_color(first_output);
        spec.width = first_output->width;
        spec.height = first_output->height;
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
        for (size_t color_index = 0; color_index < color_count; ++color_index) {
            spec.color = colors[color_index];
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
        for (size_t mode = 1; mode <= 2; ++mode) {
            for (int frame = 0; frame <= 1; ++frame) {
                for (size_t output_index = 0;
                     output_index < output_color_count;
                     ++output_index) {
                    spec.color = background;
                    spec.output_color = output_colors[output_index];
                    spec.period = mode;
                    spec.x_anchor = frame;
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
                    for (size_t color_index = 0; color_index < color_count; ++color_index) {
                        spec.color = colors[color_index];
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

static void arc_mine_crop_sketches(const ARCSketchProbeInput *input,
                                   const char *task_path,
                                   const ARCTask *task,
                                   ARCSketchProbeRecord *records,
                                   size_t *record_count,
                                   ARCSketchProgramSpec *emitted_specs,
                                   size_t *emitted_count,
                                   size_t sketches_per_task,
                                   ARCSketchProbeSummary *summary)
{
    if (task == NULL || task->train_count == 0) {
        return;
    }
    const ARCGrid *first_input = &task->train[0].input;
    const ARCGrid *first_output = &task->train[0].output;
    if (first_output->width == 0 || first_output->height == 0 ||
        first_output->width > first_input->width ||
        first_output->height > first_input->height) {
        return;
    }
    uint8_t backgrounds[2];
    size_t background_count =
        arc_task_relation_backgrounds(task,
                                      backgrounds,
                                      sizeof(backgrounds) / sizeof(backgrounds[0]));
    for (size_t background_index = 0;
         background_index < background_count;
         ++background_index) {
        uint8_t background = backgrounds[background_index];
        for (int op = ARC_SKETCH_OP_CROP; op <= ARC_SKETCH_OP_NORMALIZED_CROP; op += 5) {
            if (!arc_sketch_probe_input_allows_op(input, op)) {
                continue;
            }
            for (size_t origin_y = 0;
                 origin_y + first_output->height <= first_input->height;
                 ++origin_y) {
                for (size_t origin_x = 0;
                     origin_x + first_output->width <= first_input->width;
                     ++origin_x) {
                    for (int selector = ARC_SKETCH_SELECTOR_GRID_PANEL;
                         selector <= ARC_SKETCH_SELECTOR_FOREGROUND_BBOX_PANEL;
                         ++selector) {
                        size_t bbox_min_x = 0;
                        size_t bbox_min_y = 0;
                        size_t bbox_max_x = first_input->width - 1u;
                        size_t bbox_max_y = first_input->height - 1u;
                        if (selector == ARC_SKETCH_SELECTOR_FOREGROUND_BBOX_PANEL &&
                            !arc_grid_non_background_bbox(first_input,
                                                          background,
                                                          &bbox_min_x,
                                                          &bbox_min_y,
                                                          &bbox_max_x,
                                                          &bbox_max_y)) {
                            continue;
                        }
                        size_t base_width = bbox_max_x - bbox_min_x + 1u;
                        size_t base_height = bbox_max_y - bbox_min_y + 1u;
                        if (first_output->width > base_width ||
                            first_output->height > base_height) {
                            continue;
                        }
                        for (int x_mode = 0; x_mode <= 2; ++x_mode) {
                            size_t x_span = base_width - first_output->width;
                            int64_t x_base = (int64_t)bbox_min_x;
                            if (x_mode == 1) {
                                x_base += (int64_t)(x_span / 2u);
                            } else if (x_mode == 2) {
                                x_base += (int64_t)x_span;
                            }
                            int x_offset = (int)((int64_t)origin_x - x_base);
                            for (int y_mode = 0; y_mode <= 2; ++y_mode) {
                                size_t y_span = base_height - first_output->height;
                                int64_t y_base = (int64_t)bbox_min_y;
                                if (y_mode == 1) {
                                    y_base += (int64_t)(y_span / 2u);
                                } else if (y_mode == 2) {
                                    y_base += (int64_t)y_span;
                                }
                                ARCSketchProgramSpec spec;
                                memset(&spec, 0, sizeof(spec));
                                spec.selector = selector;
                                spec.op = op;
                                spec.background = background;
                                spec.output_color = arc_grid_dominant_color(first_output);
                                spec.x_anchor = x_mode;
                                spec.y_anchor = y_mode;
                                spec.x_offset = x_offset;
                                spec.y_offset = (int)((int64_t)origin_y - y_base);
                                spec.width = first_output->width;
                                spec.height = first_output->height;
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
    }
    uint8_t colors[ARC_COLOR_COUNT];
    size_t color_count = arc_grid_nonzero_colors(first_input, colors, ARC_COLOR_COUNT);
    for (size_t color_index = 0; color_index < color_count; ++color_index) {
        for (int op = ARC_SKETCH_OP_CROP; op <= ARC_SKETCH_OP_NORMALIZED_CROP; op += 5) {
            if (!arc_sketch_probe_input_allows_op(input, op)) {
                continue;
            }
            ARCSketchProgramSpec spec;
            memset(&spec, 0, sizeof(spec));
            spec.selector = ARC_SKETCH_SELECTOR_COLOR_BBOX;
            spec.op = op;
            spec.background = arc_grid_dominant_color(first_input);
            spec.color = colors[color_index];
            spec.output_color = arc_grid_dominant_color(first_output);
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

static void arc_recolor_crop_try_spec(const ARCSketchProbeInput *input,
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
    if (!arc_sketch_spec_matches_first_train_pair(task, spec)) {
        return;
    }
    arc_sketch_probe_append_if_exact(input,
                                     task_path,
                                     task,
                                     spec,
                                     records,
                                     record_count,
                                     emitted_specs,
                                     emitted_count,
                                     sketches_per_task,
                                     summary);
}

static void arc_mine_recolor_crop_repair_sketches(
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
    for (size_t i = 0; i < background_count; ++i) {
        arc_add_unique_color(input_colors,
                             ARC_COLOR_COUNT,
                             &input_color_count,
                             backgrounds[i]);
    }
    uint8_t output_colors[ARC_COLOR_COUNT];
    size_t output_color_count =
        arc_grid_nonzero_colors(first_output, output_colors, ARC_COLOR_COUNT);
    arc_add_unique_color(output_colors,
                         ARC_COLOR_COUNT,
                         &output_color_count,
                         arc_grid_dominant_color(first_output));
    arc_add_unique_color(output_colors,
                         ARC_COLOR_COUNT,
                         &output_color_count,
                         0);
    for (size_t background_index = 0;
         background_index < background_count;
         ++background_index) {
        uint8_t background = backgrounds[background_index];
        for (size_t color_index = 0; color_index < input_color_count; ++color_index) {
            uint8_t color = input_colors[color_index];
            for (size_t output_index = 0;
                 output_index < output_color_count;
                 ++output_index) {
                uint8_t output_color = output_colors[output_index];
                for (size_t mode = 1; mode <= 8; ++mode) {
                    ARCSketchProgramSpec spec;
                    memset(&spec, 0, sizeof(spec));
                    spec.op = ARC_SKETCH_OP_RECOLOR_CROP;
                    spec.background = background;
                    spec.color = color;
                    spec.output_color = output_color;
                    spec.period = mode;

                    spec.selector = ARC_SKETCH_SELECTOR_FOREGROUND_BBOX_PANEL;
                    arc_recolor_crop_try_spec(input,
                                              task_path,
                                              task,
                                              &spec,
                                              records,
                                              record_count,
                                              emitted_specs,
                                              emitted_count,
                                              sketches_per_task,
                                              summary);

                    int object_selectors[] = {
                        0,
                        ARC_OBJECT_SELECTOR_UNIQUE_COLOR,
                        ARC_OBJECT_SELECTOR_LARGEST_COLOR,
                        ARC_OBJECT_SELECTOR_SMALLEST_COLOR,
                        ARC_OBJECT_SELECTOR_INTERIOR_COLOR,
                        ARC_OBJECT_SELECTOR_TOUCHING_BORDER_COLOR,
                    };
                    for (size_t selector_index = 0;
                         selector_index < sizeof(object_selectors) /
                                              sizeof(object_selectors[0]);
                         ++selector_index) {
                        spec.selector = ARC_SKETCH_SELECTOR_COLOR_BBOX;
                        spec.x_anchor = object_selectors[selector_index];
                        spec.y_anchor = 0;
                        spec.width = 0;
                        spec.height = 0;
                        spec.x_offset = 0;
                        spec.y_offset = 0;
                        arc_recolor_crop_try_spec(input,
                                                  task_path,
                                                  task,
                                                  &spec,
                                                  records,
                                                  record_count,
                                                  emitted_specs,
                                                  emitted_count,
                                                  sketches_per_task,
                                                  summary);
                        if (first_output->width == 0 ||
                            first_output->height == 0 ||
                            first_output->width > first_input->width ||
                            first_output->height > first_input->height) {
                            continue;
                        }
                        spec.width = first_output->width;
                        spec.height = first_output->height;
                        int offsets[] = {0, 1, -1, 2, -2};
                        for (size_t x_offset_index = 0;
                             x_offset_index < sizeof(offsets) / sizeof(offsets[0]);
                             ++x_offset_index) {
                            spec.x_offset = offsets[x_offset_index];
                            for (size_t y_offset_index = 0;
                                 y_offset_index < sizeof(offsets) / sizeof(offsets[0]);
                                 ++y_offset_index) {
                                spec.y_offset = offsets[y_offset_index];
                                arc_recolor_crop_try_spec(input,
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

                    if (first_output->width == 0 ||
                        first_output->height == 0 ||
                        first_output->width > first_input->width ||
                        first_output->height > first_input->height) {
                        continue;
                    }
                    spec.selector = ARC_SKETCH_SELECTOR_GRID_PANEL;
                    spec.width = first_output->width;
                    spec.height = first_output->height;
                    spec.x_anchor = 0;
                    spec.y_anchor = 0;
                    for (size_t origin_y = 0;
                         origin_y + first_output->height <= first_input->height;
                         ++origin_y) {
                        spec.y_offset = (int)origin_y;
                        for (size_t origin_x = 0;
                             origin_x + first_output->width <= first_input->width;
                             ++origin_x) {
                            spec.x_offset = (int)origin_x;
                            arc_recolor_crop_try_spec(input,
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
}

static int arc_int_seen(const int *values, size_t count, int value)
{
    for (size_t i = 0; i < count; ++i) {
        if (values[i] == value) {
            return 1;
        }
    }
    return 0;
}

static void arc_append_object_mask_extract_variants(
    const ARCSketchProbeInput *input,
    const char *task_path,
    const ARCTask *task,
    const ARCSketchProgramSpec *base_spec,
    ARCSketchProbeRecord *records,
    size_t *record_count,
    ARCSketchProgramSpec *emitted_specs,
    size_t *emitted_count,
    size_t sketches_per_task,
    ARCSketchProbeSummary *summary)
{
    if (task == NULL || task->train_count == 0 || base_spec == NULL ||
        !task->train[0].has_output) {
        return;
    }
    arc_sketch_probe_append_if_exact(input,
                                     task_path,
                                     task,
                                     base_spec,
                                     records,
                                     record_count,
                                     emitted_specs,
                                     emitted_count,
                                     sketches_per_task,
                                     summary);
    const ARCGrid *first_input = &task->train[0].input;
    const ARCGrid *first_output = &task->train[0].output;
    ARCGrid tight;
    if (!arc_apply_sketch_object_mask_extract(base_spec, first_input, &tight) ||
        first_output->width == 0 || first_output->height == 0 ||
        first_output->width > ARC_MAX_GRID_WIDTH ||
        first_output->height > ARC_MAX_GRID_HEIGHT ||
        (tight.width == first_output->width && tight.height == first_output->height)) {
        return;
    }
    int delta_x = (int)first_output->width - (int)tight.width;
    int delta_y = (int)first_output->height - (int)tight.height;
    int x_offsets[5];
    int y_offsets[5];
    size_t x_count = 0;
    size_t y_count = 0;
    int x_candidates[] = {0, -delta_x, -(delta_x / 2), 1, -1};
    int y_candidates[] = {0, -delta_y, -(delta_y / 2), 1, -1};
    for (size_t i = 0; i < sizeof(x_candidates) / sizeof(x_candidates[0]); ++i) {
        if (!arc_int_seen(x_offsets, x_count, x_candidates[i])) {
            x_offsets[x_count++] = x_candidates[i];
        }
    }
    for (size_t i = 0; i < sizeof(y_candidates) / sizeof(y_candidates[0]); ++i) {
        if (!arc_int_seen(y_offsets, y_count, y_candidates[i])) {
            y_offsets[y_count++] = y_candidates[i];
        }
    }
    for (size_t xi = 0; xi < x_count; ++xi) {
        for (size_t yi = 0; yi < y_count; ++yi) {
            ARCSketchProgramSpec framed = *base_spec;
            framed.width = first_output->width;
            framed.height = first_output->height;
            framed.x_offset = x_offsets[xi];
            framed.y_offset = y_offsets[yi];
            arc_sketch_probe_append_if_exact(input,
                                             task_path,
                                             task,
                                             &framed,
                                             records,
                                             record_count,
                                             emitted_specs,
                                             emitted_count,
                                             sketches_per_task,
                                             summary);
        }
    }
}

static void arc_mine_object_mask_extract_sketches(const ARCSketchProbeInput *input,
                                                  const char *task_path,
                                                  const ARCTask *task,
                                                  ARCSketchProbeRecord *records,
                                                  size_t *record_count,
                                                  ARCSketchProgramSpec *emitted_specs,
                                                  size_t *emitted_count,
                                                  size_t sketches_per_task,
                                                  ARCSketchProbeSummary *summary)
{
    if (task == NULL || task->train_count == 0) {
        return;
    }
    const ARCGrid *first_input = &task->train[0].input;
    const ARCGrid *first_output = &task->train[0].output;
    uint8_t backgrounds[ARC_COLOR_COUNT];
    size_t background_count =
        arc_task_relation_backgrounds(task,
                                      backgrounds,
                                      sizeof(backgrounds) / sizeof(backgrounds[0]));
    uint8_t input_colors[ARC_COLOR_COUNT];
    size_t input_color_count = arc_grid_nonzero_colors(first_input,
                                                       input_colors,
                                                       ARC_COLOR_COUNT);
    uint8_t output_colors[ARC_COLOR_COUNT];
    size_t output_color_count = arc_grid_nonzero_colors(first_output,
                                                        output_colors,
                                                        ARC_COLOR_COUNT);
    arc_add_unique_color(output_colors,
                         ARC_COLOR_COUNT,
                         &output_color_count,
                         arc_grid_dominant_color(first_output));
    uint8_t selectors[] = {
        ARC_OBJECT_SELECTOR_UNIQUE_COLOR,
        ARC_OBJECT_SELECTOR_LARGEST_COLOR,
        ARC_OBJECT_SELECTOR_SMALLEST_COLOR,
        ARC_OBJECT_SELECTOR_INTERIOR_COLOR,
        ARC_OBJECT_SELECTOR_TOUCHING_BORDER_COLOR,
    };
    for (size_t background_index = 0;
         background_index < background_count;
         ++background_index) {
        uint8_t background = backgrounds[background_index];
        ARCSketchProgramSpec component_spec;
        memset(&component_spec, 0, sizeof(component_spec));
        component_spec.selector = ARC_SKETCH_SELECTOR_FOREGROUND_BBOX_PANEL;
        component_spec.op = ARC_SKETCH_OP_OBJECT_MASK_EXTRACT;
        component_spec.background = background;
        component_spec.color = background;
        component_spec.output_color = background;
        component_spec.x_anchor = ARC_OBJECT_SELECTOR_ALL_COLOR;
        for (int mode = 0; mode <= 3; ++mode) {
            component_spec.y_anchor = mode;
            if (mode == 1 || mode == 3) {
                for (size_t output_color_index = 0;
                     output_color_index < output_color_count;
                     ++output_color_index) {
                    component_spec.output_color = output_colors[output_color_index];
                    arc_append_object_mask_extract_variants(input,
                                                            task_path,
                                                            task,
                                                            &component_spec,
                                                            records,
                                                            record_count,
                                                            emitted_specs,
                                                            emitted_count,
                                                            sketches_per_task,
                                                            summary);
                }
            } else {
                component_spec.output_color = background;
                arc_append_object_mask_extract_variants(input,
                                                        task_path,
                                                        task,
                                                        &component_spec,
                                                        records,
                                                        record_count,
                                                        emitted_specs,
                                                        emitted_count,
                                                        sketches_per_task,
                                                        summary);
            }
        }
        for (size_t color_index = 0;
             color_index < input_color_count;
             ++color_index) {
            uint8_t color = input_colors[color_index];
            if (color == background) {
                continue;
            }
            for (size_t selector_index = 0;
                 selector_index < sizeof(selectors) / sizeof(selectors[0]);
                 ++selector_index) {
                ARCSketchProgramSpec spec;
                memset(&spec, 0, sizeof(spec));
                spec.selector = ARC_SKETCH_SELECTOR_COLOR_BBOX;
                spec.op = ARC_SKETCH_OP_OBJECT_MASK_EXTRACT;
                spec.background = background;
                spec.color = color;
                spec.x_anchor = selectors[selector_index];
                for (int mode = 0; mode <= 3; ++mode) {
                    spec.y_anchor = mode;
                    if (mode == 1 || mode == 3) {
                        for (size_t output_color_index = 0;
                             output_color_index < output_color_count;
                             ++output_color_index) {
                            spec.output_color = output_colors[output_color_index];
                            arc_append_object_mask_extract_variants(input,
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
                    } else {
                        spec.output_color = background;
                        arc_append_object_mask_extract_variants(input,
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

static void arc_mine_object_color_tile_sketches(const ARCSketchProbeInput *input,
                                                const char *task_path,
                                                const ARCTask *task,
                                                ARCSketchProbeRecord *records,
                                                size_t *record_count,
                                                ARCSketchProgramSpec *emitted_specs,
                                                size_t *emitted_count,
                                                size_t sketches_per_task,
                                                ARCSketchProbeSummary *summary,
                                                ARCVisualPipelineAttemptLog *attempt_log)
{
    if (task == NULL || task->train_count == 0) {
        return;
    }
    const ARCGrid *first_input = &task->train[0].input;
    const ARCGrid *first_output = &task->train[0].output;
    if (first_output->width == 0 || first_output->height == 0 ||
        first_output->width > ARC_MAX_GRID_WIDTH ||
        first_output->height > ARC_MAX_GRID_HEIGHT) {
        ++summary->unsupported_shape;
        return;
    }
    uint8_t backgrounds[ARC_COLOR_COUNT];
    size_t background_count =
        arc_task_relation_backgrounds(task,
                                      backgrounds,
                                      sizeof(backgrounds) / sizeof(backgrounds[0]));
    arc_add_unique_color(backgrounds,
                         sizeof(backgrounds) / sizeof(backgrounds[0]),
                         &background_count,
                         arc_grid_dominant_color(first_input));
    uint8_t selectors[] = {
        ARC_OBJECT_SELECTOR_LARGEST_COLOR,
        ARC_OBJECT_SELECTOR_SMALLEST_COLOR,
        ARC_OBJECT_SELECTOR_INTERIOR_COLOR,
        ARC_OBJECT_SELECTOR_TOUCHING_BORDER_COLOR,
    };
    for (size_t background_index = 0;
         background_index < background_count;
         ++background_index) {
        for (size_t selector_index = 0;
             selector_index < sizeof(selectors) / sizeof(selectors[0]);
             ++selector_index) {
            ARCVisualPipelineSpec pipeline;
            memset(&pipeline, 0, sizeof(pipeline));
            pipeline.kind = ARC_VISUAL_PIPELINE_OBJECT_COLOR_TILE;
            pipeline.background = backgrounds[background_index];
            pipeline.color = pipeline.background;
            pipeline.output_color = pipeline.background;
            pipeline.selector_kind = selectors[selector_index];
            pipeline.width = first_output->width;
            pipeline.height = first_output->height;
            arc_sketch_probe_append_visual_pipeline_if_exact(input,
                                                            task_path,
                                                            task,
                                                            &pipeline,
                                                            records,
                                                            record_count,
                                                            emitted_specs,
                                                            emitted_count,
                                                            sketches_per_task,
                                                            summary,
                                                            attempt_log);
        }
    }
}

static void arc_mine_palette_scale_sketches(const ARCSketchProbeInput *input,
                                            const char *task_path,
                                            const ARCTask *task,
                                            ARCSketchProbeRecord *records,
                                            size_t *record_count,
                                            ARCSketchProgramSpec *emitted_specs,
                                            size_t *emitted_count,
                                            size_t sketches_per_task,
                                            ARCSketchProbeSummary *summary,
                                            ARCVisualPipelineAttemptLog *attempt_log)
{
    if (task == NULL || task->train_count == 0) {
        return;
    }
    const ARCGrid *first_input = &task->train[0].input;
    ARCVisualPipelineSpec pipeline;
    memset(&pipeline, 0, sizeof(pipeline));
    pipeline.kind = ARC_VISUAL_PIPELINE_PALETTE_SCALE;
    pipeline.background = 0;
    pipeline.include_background = 1;
    arc_sketch_probe_append_visual_pipeline_if_exact(input,
                                                    task_path,
                                                    task,
                                                    &pipeline,
                                                    records,
                                                    record_count,
                                                    emitted_specs,
                                                    emitted_count,
                                                    sketches_per_task,
                                                    summary,
                                                    attempt_log);

    uint8_t backgrounds[ARC_COLOR_COUNT];
    size_t background_count =
        arc_task_relation_backgrounds(task,
                                      backgrounds,
                                      sizeof(backgrounds) / sizeof(backgrounds[0]));
    arc_add_unique_color(backgrounds,
                         sizeof(backgrounds) / sizeof(backgrounds[0]),
                         &background_count,
                         arc_grid_dominant_color(first_input));
    for (size_t background_index = 0;
         background_index < background_count;
         ++background_index) {
        memset(&pipeline, 0, sizeof(pipeline));
        pipeline.kind = ARC_VISUAL_PIPELINE_PALETTE_SCALE;
        pipeline.background = backgrounds[background_index];
        pipeline.include_background = 0;
        arc_sketch_probe_append_visual_pipeline_if_exact(input,
                                                        task_path,
                                                        task,
                                                        &pipeline,
                                                        records,
                                                        record_count,
                                                        emitted_specs,
                                                        emitted_count,
                                                        sketches_per_task,
                                                        summary,
                                                        attempt_log);
    }
}

static void arc_mine_marker_count_motif_tile_sketches(const ARCSketchProbeInput *input,
                                                      const char *task_path,
                                                      const ARCTask *task,
                                                      ARCSketchProbeRecord *records,
                                                      size_t *record_count,
                                                      ARCSketchProgramSpec *emitted_specs,
                                                      size_t *emitted_count,
                                                      size_t sketches_per_task,
                                                      ARCSketchProbeSummary *summary,
                                                      ARCVisualPipelineAttemptLog *attempt_log)
{
    if (task == NULL || task->train_count == 0) {
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
    uint8_t colors[ARC_COLOR_COUNT];
    size_t color_count = arc_grid_nonzero_colors(first_input, colors, ARC_COLOR_COUNT);
    for (size_t background_index = 0;
         background_index < background_count;
         ++background_index) {
        uint8_t background = backgrounds[background_index];
        for (size_t marker_index = 0; marker_index < color_count; ++marker_index) {
            uint8_t marker = colors[marker_index];
            if (marker == background) {
                continue;
            }
            size_t marker_count = arc_count_color_cells(first_input, marker);
            if (marker_count == 0) {
                continue;
            }
            for (int orientation = 0; orientation <= 1; ++orientation) {
                if ((orientation == 0 && first_output->width % marker_count != 0) ||
                    (orientation == 1 && first_output->height % marker_count != 0)) {
                    continue;
                }
                size_t motif_width = orientation == 0
                                         ? first_output->width / marker_count
                                         : first_output->width;
                size_t motif_height = orientation == 0
                                          ? first_output->height
                                          : first_output->height / marker_count;
                if (motif_width == 0 || motif_height == 0 ||
                    motif_width > ARC_MAX_GRID_WIDTH ||
                    motif_height > ARC_MAX_GRID_HEIGHT) {
                    continue;
                }
                for (int x_align = 0; x_align <= 2; ++x_align) {
                    for (int y_align = 0; y_align <= 2; ++y_align) {
                        ARCVisualPipelineSpec pipeline;
                        memset(&pipeline, 0, sizeof(pipeline));
                        pipeline.kind = ARC_VISUAL_PIPELINE_MARKER_COUNT_MOTIF_TILE;
                        pipeline.background = background;
                        pipeline.color = marker;
                        pipeline.output_color = background;
                        pipeline.width = motif_width;
                        pipeline.height = motif_height;
                        pipeline.orientation = orientation;
                        pipeline.x_align = x_align;
                        pipeline.y_align = y_align;
                        arc_sketch_probe_append_visual_pipeline_if_exact(input,
                                                                        task_path,
                                                                        task,
                                                                        &pipeline,
                                                                        records,
                                                                        record_count,
                                                                        emitted_specs,
                                                                        emitted_count,
                                                                        sketches_per_task,
                                                                        summary,
                                                                        attempt_log);
                        for (size_t motif_index = 0; motif_index < color_count; ++motif_index) {
                            uint8_t motif_color = colors[motif_index];
                            if (motif_color == background || motif_color == marker) {
                                continue;
                            }
                            pipeline.output_color = motif_color;
                            arc_sketch_probe_append_visual_pipeline_if_exact(input,
                                                                            task_path,
                                                                            task,
                                                                            &pipeline,
                                                                            records,
                                                                            record_count,
                                                                            emitted_specs,
                                                                            emitted_count,
                                                                            sketches_per_task,
                                                                            summary,
                                                                            attempt_log);
                        }
                    }
                }
            }
        }
    }
}

static void arc_mine_component_count_line_sketches(const ARCSketchProbeInput *input,
                                                  const char *task_path,
                                                  const ARCTask *task,
                                                  ARCSketchProbeRecord *records,
                                                  size_t *record_count,
                                                  ARCSketchProgramSpec *emitted_specs,
                                                  size_t *emitted_count,
                                                  size_t sketches_per_task,
                                                  ARCSketchProbeSummary *summary,
                                                  ARCVisualPipelineAttemptLog *attempt_log)
{
    if (task == NULL || task->train_count == 0) {
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
    uint8_t input_colors[ARC_COLOR_COUNT];
    size_t input_color_count = arc_grid_nonzero_colors(first_input,
                                                       input_colors,
                                                       ARC_COLOR_COUNT);
    uint8_t output_colors[ARC_COLOR_COUNT];
    size_t output_color_count = arc_grid_nonzero_colors(first_output,
                                                        output_colors,
                                                        ARC_COLOR_COUNT);
    for (size_t background_index = 0;
         background_index < background_count;
         ++background_index) {
        uint8_t background = backgrounds[background_index];
        for (int count_mode = 0; count_mode <= 1; ++count_mode) {
            for (int orientation = 0; orientation <= 1; ++orientation) {
                ARCVisualPipelineSpec pipeline;
                memset(&pipeline, 0, sizeof(pipeline));
                pipeline.kind = ARC_VISUAL_PIPELINE_COMPONENT_COUNT_LINE;
                pipeline.background = background;
                pipeline.color = background;
                pipeline.output_color = background;
                pipeline.orientation = orientation;
                pipeline.count_mode = count_mode;
                arc_sketch_probe_append_visual_pipeline_if_exact(input,
                                                                task_path,
                                                                task,
                                                                &pipeline,
                                                                records,
                                                                record_count,
                                                                emitted_specs,
                                                                emitted_count,
                                                                sketches_per_task,
                                                                summary,
                                                                attempt_log);
                for (size_t color_index = 0;
                     color_index < input_color_count;
                     ++color_index) {
                    uint8_t color = input_colors[color_index];
                    if (color == background) {
                        continue;
                    }
                    pipeline.color = color;
                    pipeline.output_color = background;
                    arc_sketch_probe_append_visual_pipeline_if_exact(input,
                                                                    task_path,
                                                                    task,
                                                                    &pipeline,
                                                                    records,
                                                                    record_count,
                                                                    emitted_specs,
                                                                    emitted_count,
                                                                    sketches_per_task,
                                                                    summary,
                                                                    attempt_log);
                    for (size_t output_color_index = 0;
                         output_color_index < output_color_count;
                         ++output_color_index) {
                        pipeline.output_color = output_colors[output_color_index];
                        arc_sketch_probe_append_visual_pipeline_if_exact(input,
                                                                        task_path,
                                                                        task,
                                                                        &pipeline,
                                                                        records,
                                                                        record_count,
                                                                        emitted_specs,
                                                                        emitted_count,
                                                                        sketches_per_task,
                                                                        summary,
                                                                        attempt_log);
                    }
                }
            }
        }
    }
}

static void arc_mine_axis_projection_compress_sketches(const ARCSketchProbeInput *input,
                                                       const char *task_path,
                                                       const ARCTask *task,
                                                       ARCSketchProbeRecord *records,
                                                       size_t *record_count,
                                                       ARCSketchProgramSpec *emitted_specs,
                                                       size_t *emitted_count,
                                                       size_t sketches_per_task,
                                                       ARCSketchProbeSummary *summary)
{
    if (task == NULL || task->train_count == 0) {
        return;
    }
    const ARCGrid *first_input = &task->train[0].input;
    const ARCGrid *first_output = &task->train[0].output;
    uint8_t backgrounds[ARC_COLOR_COUNT];
    size_t background_count =
        arc_task_relation_backgrounds(task,
                                      backgrounds,
                                      sizeof(backgrounds) / sizeof(backgrounds[0]));
    uint8_t output_colors[ARC_COLOR_COUNT];
    size_t output_color_count = arc_grid_nonzero_colors(first_output,
                                                        output_colors,
                                                        ARC_COLOR_COUNT);
    arc_add_unique_color(output_colors,
                         ARC_COLOR_COUNT,
                         &output_color_count,
                         arc_grid_dominant_color(first_output));
    arc_add_unique_color(output_colors,
                         ARC_COLOR_COUNT,
                         &output_color_count,
                         arc_grid_most_common_non_background_color(first_input,
                                                                    arc_grid_dominant_color(first_input)));
    for (size_t background_index = 0;
         background_index < background_count;
         ++background_index) {
        for (int axis = 0; axis <= 1; ++axis) {
            for (int mode = 0; mode <= 3; ++mode) {
                for (size_t output_color_index = 0;
                     output_color_index < output_color_count;
                     ++output_color_index) {
                    ARCSketchProgramSpec spec;
                    memset(&spec, 0, sizeof(spec));
                    spec.selector = ARC_SKETCH_SELECTOR_LATTICE;
                    spec.op = ARC_SKETCH_OP_AXIS_PROJECTION_COMPRESS;
                    spec.background = backgrounds[background_index];
                    spec.output_color = output_colors[output_color_index];
                    spec.x_anchor = axis;
                    spec.y_anchor = mode;
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

static void arc_mine_separator_panel_map_sketches(const ARCSketchProbeInput *input,
                                                  const char *task_path,
                                                  const ARCTask *task,
                                                  ARCSketchProbeRecord *records,
                                                  size_t *record_count,
                                                  ARCSketchProgramSpec *emitted_specs,
                                                  size_t *emitted_count,
                                                  size_t sketches_per_task,
                                                  ARCSketchProbeSummary *summary)
{
    if (task == NULL || task->train_count == 0) {
        return;
    }
    const ARCGrid *first_input = &task->train[0].input;
    uint8_t backgrounds[ARC_COLOR_COUNT];
    size_t background_count =
        arc_task_relation_backgrounds(task,
                                      backgrounds,
                                      sizeof(backgrounds) / sizeof(backgrounds[0]));
    uint8_t separator_colors[ARC_COLOR_COUNT];
    size_t separator_color_count = arc_grid_nonzero_colors(first_input,
                                                           separator_colors,
                                                           ARC_COLOR_COUNT);
    for (size_t background_index = 0;
         background_index < background_count;
         ++background_index) {
        for (size_t separator_index = 0;
             separator_index <= separator_color_count;
             ++separator_index) {
            uint8_t separator = separator_index == separator_color_count
                                    ? backgrounds[background_index]
                                    : separator_colors[separator_index];
            if (separator == backgrounds[background_index] &&
                separator_index != separator_color_count) {
                continue;
            }
            for (int mode = 0; mode <= 3; ++mode) {
                ARCSketchProgramSpec spec;
                memset(&spec, 0, sizeof(spec));
                spec.selector = ARC_SKETCH_SELECTOR_GRID_PANEL;
                spec.op = ARC_SKETCH_OP_SEPARATOR_PANEL_MAP;
                spec.background = backgrounds[background_index];
                spec.color = separator;
                spec.output_color = arc_grid_dominant_color(&task->train[0].output);
                spec.x_anchor = mode;
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

static void arc_mine_separator_intersection_color_sketches(
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
    if (task == NULL || task->train_count == 0 ||
        !task->train[0].has_output ||
        task->train[0].output.width != 1u ||
        task->train[0].output.height != 1u) {
        return;
    }
    const ARCGrid *first_input = &task->train[0].input;
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
    uint8_t colors[ARC_COLOR_COUNT];
    size_t color_count = arc_grid_nonzero_colors(first_input, colors, ARC_COLOR_COUNT);
    for (size_t background_index = 0;
         background_index < background_count;
         ++background_index) {
        uint8_t background = backgrounds[background_index];
        for (size_t row_color_index = 0;
             row_color_index <= color_count;
             ++row_color_index) {
            uint8_t row_color =
                row_color_index == color_count ? background : colors[row_color_index];
            for (size_t col_color_index = 0;
                 col_color_index <= color_count;
                 ++col_color_index) {
                uint8_t col_color =
                    col_color_index == color_count ? background : colors[col_color_index];
                for (size_t tolerance = 0; tolerance <= 4u; ++tolerance) {
                    ARCSketchProgramSpec spec;
                    memset(&spec, 0, sizeof(spec));
                    spec.selector = ARC_SKETCH_SELECTOR_GRID_PANEL;
                    spec.op = ARC_SKETCH_OP_SEPARATOR_INTERSECTION_COLOR;
                    spec.background = background;
                    spec.color = row_color;
                    spec.output_color = col_color;
                    spec.period = tolerance;
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

static void arc_mine_separator_panel_relation_mask_sketches(
    const ARCSketchProbeInput *input,
    const char *task_path,
    const ARCTask *task,
    ARCSketchProbeRecord *records,
    size_t *record_count,
    ARCSketchProgramSpec *emitted_specs,
    size_t *emitted_count,
    size_t sketches_per_task,
    ARCSketchProbeSummary *summary,
    ARCVisualPipelineAttemptLog *attempt_log)
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
    uint8_t separator_colors[ARC_COLOR_COUNT];
    size_t separator_color_count =
        arc_grid_nonzero_colors(first_input, separator_colors, ARC_COLOR_COUNT);
    uint8_t output_colors[ARC_COLOR_COUNT];
    size_t output_color_count =
        arc_grid_nonzero_colors(first_output, output_colors, ARC_COLOR_COUNT);
    for (size_t background_index = 0;
         background_index < background_count;
         ++background_index) {
        uint8_t background = backgrounds[background_index];
        arc_add_unique_color(output_colors,
                             sizeof(output_colors) / sizeof(output_colors[0]),
                             &output_color_count,
                             arc_grid_most_common_non_background_color(first_output,
                                                                       background));
        for (size_t separator_index = 0;
             separator_index <= separator_color_count;
             ++separator_index) {
            uint8_t separator =
                separator_index == separator_color_count
                    ? background
                    : separator_colors[separator_index];
            for (size_t output_index = 0;
                 output_index < output_color_count;
                 ++output_index) {
                uint8_t output_color = output_colors[output_index];
                if (output_color == background) {
                    continue;
                }
                for (int orientation = 0; orientation <= 2; ++orientation) {
                    for (size_t relation_mode = 0; relation_mode <= 5u; ++relation_mode) {
                        ARCVisualPipelineSpec pipeline;
                        memset(&pipeline, 0, sizeof(pipeline));
                        pipeline.kind = ARC_VISUAL_PIPELINE_SEPARATOR_PANEL_RELATION_MASK;
                        pipeline.background = background;
                        pipeline.color = separator;
                        pipeline.output_color = output_color;
                        pipeline.x_anchor = orientation;
                        pipeline.period = relation_mode;
                        arc_sketch_probe_append_visual_pipeline_if_exact(input,
                                                                        task_path,
                                                                        task,
                                                                        &pipeline,
                                                                        records,
                                                                        record_count,
                                                                        emitted_specs,
                                                                        emitted_count,
                                                                        sketches_per_task,
                                                                        summary,
                                                                        attempt_log);
                    }
                }
            }
        }
    }
}

static int arc_task_train_outputs_match_input_shapes(const ARCTask *task)
{
    if (task == NULL || task->train_count == 0) {
        return 0;
    }
    for (size_t pair = 0; pair < task->train_count; ++pair) {
        if (!task->train[pair].has_output ||
            !arc_grid_same_shape(&task->train[pair].input, &task->train[pair].output)) {
            return 0;
        }
    }
    return 1;
}

static void arc_append_visual_combinator_frame_variants(
    const ARCSketchProbeInput *input,
    const char *task_path,
    const ARCTask *task,
    const ARCVisualPipelineSpec *pipeline,
    int include_input_frame,
    ARCSketchProbeRecord *records,
    size_t *record_count,
    ARCSketchProgramSpec *emitted_specs,
    size_t *emitted_count,
    size_t sketches_per_task,
    ARCSketchProbeSummary *summary,
    ARCVisualPipelineAttemptLog *attempt_log)
{
    ARCVisualPipelineSpec input_frame;
    arc_sketch_probe_append_visual_pipeline_if_exact(input,
                                                    task_path,
                                                    task,
                                                    pipeline,
                                                    records,
                                                    record_count,
                                                    emitted_specs,
                                                    emitted_count,
                                                    sketches_per_task,
                                                    summary,
                                                    attempt_log);
    if (!include_input_frame || pipeline == NULL) {
        return;
    }
    input_frame = *pipeline;
    input_frame.width = 0;
    input_frame.height = 0;
    arc_sketch_probe_append_visual_pipeline_if_exact(input,
                                                    task_path,
                                                    task,
                                                    &input_frame,
                                                    records,
                                                    record_count,
                                                    emitted_specs,
                                                    emitted_count,
                                                    sketches_per_task,
                                                    summary,
                                                    attempt_log);
}

static void arc_mine_visual_combinator_sketches(
    const ARCSketchProbeInput *input,
    const char *task_path,
    const ARCTask *task,
    ARCSketchProbeRecord *records,
    size_t *record_count,
    ARCSketchProgramSpec *emitted_specs,
    size_t *emitted_count,
    size_t sketches_per_task,
    ARCSketchProbeSummary *summary,
    ARCVisualPipelineAttemptLog *attempt_log)
{
    if (task == NULL || task->train_count == 0 || !task->train[0].has_output) {
        return;
    }
    const ARCGrid *first_input = &task->train[0].input;
    const ARCGrid *first_output = &task->train[0].output;
    int output_uses_input_frame = arc_task_train_outputs_match_input_shapes(task);
    if (first_output->width == 0 || first_output->height == 0 ||
        first_output->width > ARC_MAX_GRID_WIDTH ||
        first_output->height > ARC_MAX_GRID_HEIGHT) {
        ++summary->unsupported_shape;
        return;
    }
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
    int allow_component_mask =
        arc_text_list_contains_token(input->op_hints, "emitter:component_mask", ',') ||
        arc_text_list_contains_token(input->op_hints, "normalizer:component_mask", ',') ||
        arc_text_list_contains_token(input->op_hints, "selector:component_predicate", ',') ||
        arc_text_list_contains_token(input->op_hints, "normalizer:component_selector_layout", ',') ||
        arc_text_list_contains_token(input->op_hints, "pin:x_anchor=3", ',');
    int allow_aligned_span =
        arc_text_list_contains_token(input->op_hints, "emitter:aligned_span", ',') ||
        arc_text_list_contains_token(input->op_hints, "composer:aligned_span", ',') ||
        arc_text_list_contains_token(input->op_hints, "pin:x_anchor=4", ',');
    int allow_component_predicate_mask =
        arc_text_list_contains_token(input->op_hints, "selector:component_predicate", ',') ||
        arc_text_list_contains_token(input->op_hints, "normalizer:component_selector_layout", ',');
    for (int predicate_mode = 7; predicate_mode <= 18; ++predicate_mode) {
        if (arc_text_list_contains_int_pin(input->op_hints, "y_anchor", predicate_mode)) {
            allow_component_predicate_mask = 1;
            break;
        }
    }
    for (size_t background_index = 0;
         background_index < background_count;
         ++background_index) {
        uint8_t background = backgrounds[background_index];
        arc_add_unique_color(output_colors,
                             sizeof(output_colors) / sizeof(output_colors[0]),
                             &output_color_count,
                             arc_grid_most_common_non_background_color(first_output,
                                                                       background));
        size_t mask_driver_output_index = SIZE_MAX;
        for (size_t k = 0; k < output_color_count; ++k) {
            if (output_colors[k] != background) {
                mask_driver_output_index = k;
                break;
            }
        }
        for (size_t color_index = 0;
             color_index <= input_color_count;
            ++color_index) {
            uint8_t color =
                color_index == input_color_count ? background : input_colors[color_index];
            for (size_t output_index = 0;
                 output_index < output_color_count;
                 ++output_index) {
                uint8_t output_color = output_colors[output_index];
                if (output_color == background) {
                    continue;
                }
                if (!arc_text_list_allows_int_pin(input->op_hints, "x_anchor", 0)) {
                    goto skip_occupancy_grid;
                }
                for (int coordinate_mode = 0; coordinate_mode <= 5; ++coordinate_mode) {
                    if (!arc_text_list_allows_int_pin(input->op_hints,
                                                      "y_anchor",
                                                      coordinate_mode)) {
                        continue;
                    }
                    ARCVisualPipelineSpec pipeline;
                    memset(&pipeline, 0, sizeof(pipeline));
                    pipeline.kind = ARC_VISUAL_PIPELINE_VISUAL_COMBINATOR;
                    pipeline.selector = ARC_SKETCH_SELECTOR_LATTICE;
                    pipeline.background = background;
                    pipeline.color = color;
                    pipeline.output_color = output_color;
                    pipeline.width = first_output->width;
                    pipeline.height = first_output->height;
                    pipeline.x_anchor = 0;
                    pipeline.y_anchor = coordinate_mode;
                    pipeline.period = 0;
                    arc_append_visual_combinator_frame_variants(input,
                                                                task_path,
                                                                task,
                                                                &pipeline,
                                                                output_uses_input_frame,
                                                                records,
                                                                record_count,
                                                                emitted_specs,
                                                                emitted_count,
                                                                sketches_per_task,
                                                                summary,
                                                                attempt_log);
                }
skip_occupancy_grid:
                if (!arc_text_list_allows_int_pin(input->op_hints, "x_anchor", 1)) {
                    goto skip_count_template;
                }
                for (int pattern_mode = 0; pattern_mode <= 3; ++pattern_mode) {
                    if (!arc_text_list_allows_int_pin(input->op_hints,
                                                      "y_anchor",
                                                      pattern_mode)) {
                        continue;
                    }
                    ARCVisualPipelineSpec pipeline;
                    memset(&pipeline, 0, sizeof(pipeline));
                    pipeline.kind = ARC_VISUAL_PIPELINE_VISUAL_COMBINATOR;
                    pipeline.selector = ARC_SKETCH_SELECTOR_LATTICE;
                    pipeline.background = background;
                    pipeline.color = color;
                    pipeline.output_color = output_color;
                    pipeline.width = first_output->width;
                    pipeline.height = first_output->height;
                    pipeline.x_anchor = 1;
                    pipeline.y_anchor = pattern_mode;
                    pipeline.period = 0;
                    arc_append_visual_combinator_frame_variants(input,
                                                                task_path,
                                                                task,
                                                                &pipeline,
                                                                output_uses_input_frame,
                                                                records,
                                                                record_count,
                                                                emitted_specs,
                                                                emitted_count,
                                                                sketches_per_task,
                                                                summary,
                                                                attempt_log);
                }
skip_count_template:
                if (!allow_aligned_span ||
                    !output_uses_input_frame ||
                    color == background ||
                    !arc_text_list_allows_int_pin(input->op_hints, "x_anchor", 4)) {
                    goto skip_aligned_span;
                }
                for (int span_mode = ARC_OBJECT_CONNECT_HORIZONTAL;
                     span_mode <= ARC_OBJECT_CONNECT_BOTH;
                     ++span_mode) {
                    if (!arc_text_list_allows_int_pin(input->op_hints,
                                                      "y_anchor",
                                                      span_mode)) {
                        continue;
                    }
                    ARCVisualPipelineSpec pipeline;
                    memset(&pipeline, 0, sizeof(pipeline));
                    pipeline.kind = ARC_VISUAL_PIPELINE_VISUAL_COMBINATOR;
                    pipeline.selector = ARC_SKETCH_SELECTOR_LATTICE;
                    pipeline.background = background;
                    pipeline.color = color;
                    pipeline.output_color = output_color;
                    pipeline.width = first_output->width;
                    pipeline.height = first_output->height;
                    pipeline.x_anchor = 4;
                    pipeline.y_anchor = span_mode;
                    pipeline.period = 0;
                    arc_append_visual_combinator_frame_variants(input,
                                                                task_path,
                                                                task,
                                                                &pipeline,
                                                                output_uses_input_frame,
                                                                records,
                                                                record_count,
                                                                emitted_specs,
                                                                emitted_count,
                                                                sketches_per_task,
                                                                summary,
                                                                attempt_log);
                }
skip_aligned_span:
                if (!arc_text_list_allows_int_pin(input->op_hints, "x_anchor", 2)) {
                    goto skip_component_shape_grid;
                }
                for (int classifier_mode = 0; classifier_mode <= 6; ++classifier_mode) {
                    if (!arc_text_list_allows_int_pin(input->op_hints,
                                                      "period",
                                                      classifier_mode)) {
                        continue;
                    }
                    size_t alt_color_count = classifier_mode <= 1 ? 1u : output_color_count;
                    for (size_t alt_index = 0;
                         alt_index < alt_color_count;
                         ++alt_index) {
                        uint8_t alt_color =
                            classifier_mode <= 1 ? background : output_colors[alt_index];
                        if (classifier_mode > 1 && alt_color == background) {
                            continue;
                        }
                        if (classifier_mode > 1 && alt_color == output_color) {
                            continue;
                        }
                        for (int coordinate_mode = 0;
                             coordinate_mode <= 5;
                             ++coordinate_mode) {
                            if (!arc_text_list_allows_int_pin(input->op_hints,
                                                              "y_anchor",
                                                              coordinate_mode)) {
                                continue;
                            }
                            ARCVisualPipelineSpec pipeline;
                            memset(&pipeline, 0, sizeof(pipeline));
                            pipeline.kind = ARC_VISUAL_PIPELINE_VISUAL_COMBINATOR;
                            pipeline.selector = ARC_SKETCH_SELECTOR_LATTICE;
                            pipeline.background = background;
                            pipeline.color = color;
                            pipeline.output_color = output_color;
                            pipeline.width = first_output->width;
                            pipeline.height = first_output->height;
                            pipeline.x_anchor = 2;
                            pipeline.y_anchor = coordinate_mode;
                            pipeline.x_offset = (int)alt_color;
                            pipeline.period = (size_t)classifier_mode;
                            arc_append_visual_combinator_frame_variants(input,
                                                                        task_path,
                                                                        task,
                                                                        &pipeline,
                                                                        output_uses_input_frame,
                                                                        records,
                                                                        record_count,
                                                                        emitted_specs,
                                                                        emitted_count,
                                                                        sketches_per_task,
                                                                        summary,
                                                                        attempt_log);
                        }
                    }
                }
skip_component_shape_grid:
                if (!allow_component_mask ||
                    output_index != mask_driver_output_index ||
                    !arc_text_list_allows_int_pin(input->op_hints, "x_anchor", 3)) {
                    continue;
                }
                for (int mask_mode = 0; mask_mode <= 18; ++mask_mode) {
                    if (mask_mode > 6 && !allow_component_predicate_mask) {
                        continue;
                    }
                    if (!arc_text_list_allows_int_pin(input->op_hints,
                                                      "y_anchor",
                                                      mask_mode)) {
                        continue;
                    }
                    for (int color_mode = 0; color_mode <= 3; ++color_mode) {
                        if (!arc_text_list_allows_int_pin(input->op_hints,
                                                          "period",
                                                          color_mode)) {
                            continue;
                        }
                        size_t alt_color_count = color_mode == 2 ? output_color_count : 1u;
                        size_t mask_output_count =
                            (color_mode == 1 || color_mode == 3) ? 1u : output_color_count;
                        for (size_t mask_output_index = 0;
                             mask_output_index < mask_output_count;
                             ++mask_output_index) {
                            uint8_t mask_output_color =
                                color_mode == 1 ? output_color : output_colors[mask_output_index];
                            if (mask_output_color == background) {
                                continue;
                            }
                            for (size_t alt_index = 0;
                                 alt_index < alt_color_count;
                                 ++alt_index) {
                                uint8_t alt_color =
                                    color_mode == 2 ? output_colors[alt_index] : background;
                                if (color_mode == 2 &&
                                    (alt_color == background || alt_color == mask_output_color)) {
                                    continue;
                                }
                                for (int render_mode = 0;
                                     render_mode <= 8;
                                     ++render_mode) {
                                    if (!arc_text_list_allows_int_pin(input->op_hints,
                                                                      "y_offset",
                                                                      render_mode)) {
                                        continue;
                                    }
                                    ARCVisualPipelineSpec pipeline;
                                    memset(&pipeline, 0, sizeof(pipeline));
                                    pipeline.kind = ARC_VISUAL_PIPELINE_VISUAL_COMBINATOR;
                                    pipeline.selector = ARC_SKETCH_SELECTOR_LATTICE;
                                    pipeline.background = background;
                                    pipeline.color = color;
                                    pipeline.output_color = mask_output_color;
                                    pipeline.width = first_output->width;
                                    pipeline.height = first_output->height;
                                    pipeline.x_anchor = 3;
                                    pipeline.y_anchor = mask_mode;
                                    pipeline.x_offset = (int)alt_color;
                                    pipeline.y_offset = render_mode;
                                    pipeline.period = (size_t)color_mode;
                                    if (color_mode == 3) {
                                        ARCSketchProgramSpec spec_for_map;
                                        memset(&spec_for_map, 0, sizeof(spec_for_map));
                                        if (!arc_visual_pipeline_to_sketch_spec(
                                                &pipeline,
                                                &spec_for_map) ||
                                            !arc_visual_component_mask_marker_glyph_map_for_task(
                                                task,
                                                &spec_for_map,
                                                &pipeline.x_offset)) {
                                            continue;
                                        }
                                    }
                                    arc_append_visual_combinator_frame_variants(input,
                                                                                task_path,
                                                                                task,
                                                                                &pipeline,
                                                                                output_uses_input_frame,
                                                                                records,
                                                                                record_count,
                                                                                emitted_specs,
                                                                                emitted_count,
                                                                                sketches_per_task,
                                                                                summary,
                                                                                attempt_log);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
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
    ARCSketchProbeSummary *summary);

static void arc_print_reference_recolor_mask_miner_trace(
    const ARCSketchProbeInput *input,
    const ARCTask *task,
    size_t attempted_specs,
    size_t accepted_specs)
{
    if (input == NULL || task == NULL ||
        strcmp(input->family, "arc.generative_prior") != 0 ||
        attempted_specs == 0) {
        return;
    }
    fputs("{\"schema\":\"crlp.arc_reference_recolor_mask_miner.trace.v1\","
          "\"report_version\":1,\"task\":\"arc_agi\",\"task_instance\":",
          stdout);
    nn_write_json_string(stdout, task->id);
    fprintf(stdout,
            ",\"attempted_specs\":%zu,\"accepted_specs\":%zu,"
            "\"compiler_target\":\"visual_combinator\","
            "\"composition_kind\":\"reference_conditioned_recolor\","
            "\"normalization_frame\":\"reference_target_bbox\","
            "\"typed_stages\":\"select_reference|derive_relative_mask|select_target|recolor_overlay\","
            "\"primitive_contract\":\"compiler_replay\","
            "\"visual_program\":\"reference_recolor_mask\"}\n",
            attempted_specs,
            accepted_specs);
}

static void arc_mine_reference_recolor_mask_sketches(
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
    size_t attempted_specs = 0;
    size_t accepted_specs = 0;
    if (task == NULL || task->train_count == 0 || !task->train[0].has_output ||
        !arc_task_train_outputs_match_input_shapes(task)) {
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
    for (size_t background_index = 0;
         background_index < background_count;
         ++background_index) {
        uint8_t background = backgrounds[background_index];
        uint8_t output_color =
            arc_grid_most_common_non_background_color(first_output, background);
        if (output_color == background || output_color >= ARC_COLOR_COUNT) {
            continue;
        }
        for (size_t color_index = 0;
             color_index < input_color_count;
             ++color_index) {
            uint8_t target_color = input_colors[color_index];
            ARCVisualPipelineSpec pipeline;
            ARCSketchProgramSpec spec_for_map;
            if (target_color == background) {
                continue;
            }
            memset(&pipeline, 0, sizeof(pipeline));
            pipeline.kind = ARC_VISUAL_PIPELINE_VISUAL_COMBINATOR;
            pipeline.selector = ARC_SKETCH_SELECTOR_LATTICE;
            pipeline.background = background;
            pipeline.color = target_color;
            pipeline.output_color = output_color;
            pipeline.width = 0;
            pipeline.height = 0;
            pipeline.x_anchor = 3;
            pipeline.y_anchor = 6;
            pipeline.y_offset = 0;
            pipeline.period = 3;
            memset(&spec_for_map, 0, sizeof(spec_for_map));
            if (!arc_visual_pipeline_to_sketch_spec(&pipeline, &spec_for_map) ||
                !arc_visual_component_mask_marker_glyph_map_for_task(task,
                                                                     &spec_for_map,
                                                                     &pipeline.x_offset)) {
                continue;
            }
            ++attempted_specs;
            if (arc_sketch_probe_append_visual_pipeline_if_exact(input,
                                                                 task_path,
                                                                 task,
                                                                 &pipeline,
                                                                 records,
                                                                 record_count,
                                                                 emitted_specs,
                                                                 emitted_count,
                                                                 sketches_per_task,
                                                                 summary,
                                                                 NULL)) {
                ++accepted_specs;
            }
        }
    }
    arc_print_reference_recolor_mask_miner_trace(input,
                                                 task,
                                                 attempted_specs,
                                                 accepted_specs);
}

static void arc_mine_learned_visual_grammar_sketches(
    const ARCSketchProbeInput *input,
    const char *task_path,
    const ARCTask *task,
    ARCSketchProbeRecord *records,
    size_t *record_count,
    ARCSketchProgramSpec *emitted_specs,
    size_t *emitted_count,
    size_t sketches_per_task,
    ARCSketchProbeSummary *summary,
    ARCVisualPipelineAttemptLog *attempt_log)
{
    if (input == NULL || input->op_hints[0] == '\0' ||
        strcmp(input->op_hints, "*") == 0) {
        return;
    }
    if (arc_sketch_probe_input_allows_op(input, ARC_SKETCH_OP_PALETTE_SCALE)) {
        arc_mine_palette_scale_sketches(input,
                                        task_path,
                                        task,
                                        records,
                                        record_count,
                                        emitted_specs,
                                        emitted_count,
                                        sketches_per_task,
                                        summary,
                                        attempt_log);
    }
    if (arc_sketch_probe_input_allows_op(input, ARC_SKETCH_OP_MACROCELL_SUBSTITUTE)) {
        arc_mine_macrocell_sketches(input,
                                    task_path,
                                    task,
                                    records,
                                    record_count,
                                    emitted_specs,
                                    emitted_count,
                                    sketches_per_task,
                                    summary);
    }
    if (arc_sketch_probe_input_allows_op(input, ARC_SKETCH_OP_LATTICE_SAMPLE)) {
        arc_mine_lattice_sample_sketches(input,
                                         task_path,
                                         task,
                                         records,
                                         record_count,
                                         emitted_specs,
                                         emitted_count,
                                         sketches_per_task,
                                         summary);
        arc_mine_diagonal_compact_sketches(input,
                                           task_path,
                                           task,
                                           records,
                                           record_count,
                                           emitted_specs,
                                           emitted_count,
                                           sketches_per_task,
                                           summary);
    }
    if (arc_sketch_probe_input_allows_op(input, ARC_SKETCH_OP_POINT_LINE_PROJECT)) {
        arc_mine_point_line_project_sketches(input,
                                             task_path,
                                             task,
                                             records,
                                             record_count,
                                             emitted_specs,
                                             emitted_count,
                                             sketches_per_task,
                                             summary);
    }
    if (arc_sketch_probe_input_allows_op(input, ARC_SKETCH_OP_MARKER_POLYLINE) ||
        arc_text_list_contains_token(input->op_hints,
                                     "counterexample_selector_phase_repair",
                                     ',') ||
        arc_text_list_contains_token(input->op_hints,
                                     "composer:polyline",
                                     ',')) {
        arc_mine_marker_polyline_sketches(input,
                                          task_path,
                                          task,
                                          records,
                                          record_count,
                                          emitted_specs,
                                          emitted_count,
                                          sketches_per_task,
                                          summary);
    }
    if (arc_sketch_probe_input_allows_op(input, ARC_SKETCH_OP_SEPARATOR_PANEL_MAP)) {
        arc_mine_separator_panel_map_sketches(input,
                                             task_path,
                                             task,
                                             records,
                                             record_count,
                                             emitted_specs,
                                             emitted_count,
                                             sketches_per_task,
                                             summary);
    }
    if (arc_sketch_probe_input_allows_op(input, ARC_SKETCH_OP_SEPARATOR_INTERSECTION_COLOR) ||
        arc_text_list_contains_token(input->op_hints,
                                     "separator_intersection_frame",
                                     ',') ||
        arc_text_list_contains_token(input->op_hints,
                                     "normalizer:intersection",
                                     ',')) {
        arc_mine_separator_intersection_color_sketches(input,
                                                       task_path,
                                                       task,
                                                       records,
                                                       record_count,
                                                       emitted_specs,
                                                       emitted_count,
                                                       sketches_per_task,
                                                       summary);
    }
    if (arc_sketch_probe_input_allows_op(input, ARC_SKETCH_OP_SEPARATOR_PANEL_RELATION_MASK) ||
        arc_text_list_contains_token(input->op_hints,
                                     "separator_relation_frame",
                                     ',') ||
        arc_text_list_contains_token(input->op_hints,
                                     "composer:panel_relation",
                                     ',')) {
        arc_mine_separator_panel_relation_mask_sketches(input,
                                                       task_path,
                                                       task,
                                                       records,
                                                       record_count,
                                                       emitted_specs,
                                                       emitted_count,
                                                       sketches_per_task,
                                                       summary,
                                                       attempt_log);
    }
    if (arc_sketch_probe_input_allows_op(input, ARC_SKETCH_OP_COMPONENT_COLOR_SUMMARY)) {
        arc_mine_component_summary_sketches(input,
                                           task_path,
                                           task,
                                           records,
                                           record_count,
                                           emitted_specs,
                                           emitted_count,
                                           sketches_per_task,
                                           summary);
    }
    if (arc_sketch_probe_input_allows_op(input, ARC_SKETCH_OP_VISUAL_COMBINATOR) ||
        arc_text_list_contains_token(input->op_hints,
                                     "object_lattice_frame",
                                     ',') ||
        arc_text_list_contains_token(input->op_hints,
                                     "normalizer:object_lattice",
                                     ',') ||
        arc_text_list_contains_token(input->op_hints,
                                     "composer:visual_program",
                                     ',')) {
        arc_mine_visual_combinator_sketches(input,
                                           task_path,
                                           task,
                                           records,
                                           record_count,
                                           emitted_specs,
                                           emitted_count,
                                           sketches_per_task,
                                           summary,
                                           attempt_log);
    }
    if (arc_sketch_probe_input_allows_op(input, ARC_SKETCH_OP_CROP) ||
        arc_sketch_probe_input_allows_op(input, ARC_SKETCH_OP_NORMALIZED_CROP)) {
        arc_mine_crop_sketches(input,
                               task_path,
                               task,
                               records,
                               record_count,
                               emitted_specs,
                               emitted_count,
                               sketches_per_task,
                               summary);
    }
    if (arc_sketch_probe_input_allows_op(input, ARC_SKETCH_OP_RECOLOR_CROP) ||
        arc_text_list_contains_token(input->op_hints,
                                     "counterexample_cell_mask_cleanup_repair",
                                     ',') ||
        arc_text_list_contains_token(input->op_hints,
                                     "counterexample_transform_color_repair",
                                     ',') ||
        arc_text_list_contains_token(input->op_hints,
                                     "counterexample_shape_frame_repair",
                                     ',')) {
        arc_mine_recolor_crop_repair_sketches(input,
                                              task_path,
                                              task,
                                              records,
                                              record_count,
                                              emitted_specs,
                                              emitted_count,
                                              sketches_per_task,
                                              summary);
    }
    if (arc_sketch_probe_input_allows_op(input, ARC_SKETCH_OP_OBJECT_COLOR_TILE)) {
        arc_mine_object_color_tile_sketches(input,
                                           task_path,
                                           task,
                                           records,
                                           record_count,
                                           emitted_specs,
                                           emitted_count,
                                           sketches_per_task,
                                           summary,
                                           attempt_log);
    }
    if (arc_sketch_probe_input_allows_op(input, ARC_SKETCH_OP_OBJECT_MASK_EXTRACT)) {
        arc_mine_object_mask_extract_sketches(input,
                                             task_path,
                                             task,
                                             records,
                                             record_count,
                                             emitted_specs,
                                             emitted_count,
                                             sketches_per_task,
                                             summary);
    }
    if (arc_sketch_probe_input_allows_op(input, ARC_SKETCH_OP_OBJECT_TRANSLATE) ||
        arc_text_list_contains_token(input->op_hints,
                                     "object_motion",
                                     ',') ||
        arc_text_list_contains_token(input->op_hints,
                                     "transform:translate",
                                     ',')) {
        arc_mine_object_translate_sketches(input,
                                          task_path,
                                          task,
                                          records,
                                          record_count,
                                          emitted_specs,
                                          emitted_count,
                                          sketches_per_task,
                                          summary);
    }
    if (arc_sketch_probe_input_allows_op(input, ARC_SKETCH_OP_MARKER_COUNT_MOTIF_TILE)) {
        arc_mine_marker_count_motif_tile_sketches(input,
                                                 task_path,
                                                 task,
                                                 records,
                                                 record_count,
                                                 emitted_specs,
                                                 emitted_count,
                                                 sketches_per_task,
                                                 summary,
                                                 attempt_log);
    }
    if (arc_sketch_probe_input_allows_op(input, ARC_SKETCH_OP_COMPONENT_COUNT_LINE)) {
        arc_mine_component_count_line_sketches(input,
                                              task_path,
                                              task,
                                              records,
                                              record_count,
                                              emitted_specs,
                                              emitted_count,
                                              sketches_per_task,
                                              summary,
                                              attempt_log);
    }
    if (arc_sketch_probe_input_allows_op(input, ARC_SKETCH_OP_AXIS_PROJECTION_COMPRESS)) {
        arc_mine_axis_projection_compress_sketches(input,
                                                   task_path,
                                                   task,
                                                   records,
                                                   record_count,
                                                   emitted_specs,
                                                   emitted_count,
                                                   sketches_per_task,
                                                   summary);
    }
}
