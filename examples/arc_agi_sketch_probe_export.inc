static int arc_read_sketch_probe_inputs(const char *path,
                                        ARCSketchProbeInput *inputs,
                                        size_t capacity,
                                        size_t *count_out)
{
    FILE *fp = fopen(path, "r");
    char line[4096];
    size_t count = 0;
    if (fp == NULL || inputs == NULL || count_out == NULL) {
        if (fp != NULL) {
            fclose(fp);
        }
        return 0;
    }
    while (fgets(line, sizeof(line), fp) != NULL) {
        if (strncmp(line, "family\t", strlen("family\t")) == 0 ||
            line[0] == '\0' || line[0] == '\n' || line[0] == '#') {
            continue;
        }
        if (count >= capacity) {
            fclose(fp);
            return 0;
        }
        char *fields[13];
        size_t field_count = arc_split_tab_fields(line, fields, 13);
        if (field_count < 6 || field_count > 13) {
            fclose(fp);
            return 0;
        }
        ARCSketchProbeInput *input = &inputs[count];
        memset(input, 0, sizeof(*input));
        snprintf(input->family, sizeof(input->family), "%s", fields[0]);
        snprintf(input->task_id, sizeof(input->task_id), "%s", fields[1]);
        snprintf(input->split, sizeof(input->split), "%s", fields[2]);
        input->support_tasks = (size_t)strtoull(fields[3], NULL, 10);
        input->proposal_count = (size_t)strtoull(fields[4], NULL, 10);
        input->utility_mean = strtod(fields[5], NULL);
        if (field_count >= 7) {
            snprintf(input->op_hints, sizeof(input->op_hints), "%s", fields[6]);
        }
        if (field_count >= 8) {
            snprintf(input->coverage_frontier_id,
                     sizeof(input->coverage_frontier_id),
                     "%s",
                     fields[7]);
        }
        if (field_count >= 9) {
            snprintf(input->parent_attempt_id,
                     sizeof(input->parent_attempt_id),
                     "%s",
                     fields[8]);
        }
        if (field_count >= 10) {
            snprintf(input->repair_operator,
                     sizeof(input->repair_operator),
                     "%s",
                     fields[9]);
        }
        if (field_count >= 11) {
            snprintf(input->dominant_residual_class,
                     sizeof(input->dominant_residual_class),
                     "%s",
                     fields[10]);
        }
        if (field_count >= 12) {
            snprintf(input->dominant_failure_stage,
                     sizeof(input->dominant_failure_stage),
                     "%s",
                     fields[11]);
        }
        if (field_count >= 13) {
            snprintf(input->repair_strategy,
                     sizeof(input->repair_strategy),
                     "%s",
                     fields[12]);
        }
        ++count;
    }
    fclose(fp);
    *count_out = count;
    return 1;
}

static void arc_write_sketch_probe_slot_values(FILE *fp, const ARCSketchProgramSpec *spec)
{
    fprintf(fp,
            "{\"sketch.background\":%u,\"sketch.color\":%u,"
            "\"sketch.height\":%zu,\"sketch.op\":%d,"
            "\"sketch.output_color\":%u,\"sketch.period\":%zu,"
            "\"sketch.selector\":%d,\"sketch.width\":%zu,"
            "\"sketch.x_anchor\":%d,\"sketch.x_offset\":%d,"
            "\"sketch.y_anchor\":%d,\"sketch.y_offset\":%d}",
            (unsigned)spec->background,
            (unsigned)spec->color,
            spec->height,
            spec->op,
            (unsigned)spec->output_color,
            spec->period,
            spec->selector,
            spec->width,
            spec->x_anchor,
            spec->x_offset,
            spec->y_anchor,
            spec->y_offset);
}

static void arc_write_visual_pipeline_values(FILE *fp,
                                             const ARCVisualPipelineSpec *pipeline)
{
    const char *composition_kind =
        pipeline->composition_kind == NULL ? "" : pipeline->composition_kind;
    const char *normalization_frame =
        pipeline->normalization_frame == NULL ? "" : pipeline->normalization_frame;
    const char *typed_stages =
        pipeline->typed_stages == NULL ? "" : pipeline->typed_stages;
    const char *primitive_contract =
        pipeline->primitive_contract == NULL ? "" : pipeline->primitive_contract;
    fputs("{\"pipeline_kind\":", fp);
    nn_write_json_string(fp, arc_visual_pipeline_name(pipeline->kind));
    fputs(",\"composition_kind\":", fp);
    nn_write_json_string(fp, composition_kind);
    fputs(",\"normalization_frame\":", fp);
    nn_write_json_string(fp, normalization_frame);
    fputs(",\"typed_stages\":", fp);
    nn_write_json_string(fp, typed_stages);
    fputs(",\"primitive_contract\":", fp);
    nn_write_json_string(fp, primitive_contract);
    fprintf(fp,
            ",\"selector\":%d,\"background\":%u,\"color\":%u,\"output_color\":%u,"
            "\"selector_kind\":%u,\"include_background\":%s,"
            "\"orientation\":%d,\"count_mode\":%d,"
            "\"x_align\":%d,\"y_align\":%d,"
            "\"x_anchor\":%d,\"y_anchor\":%d,"
            "\"x_offset\":%d,\"y_offset\":%d,"
            "\"width\":%zu,\"height\":%zu,\"period\":%zu}",
            pipeline->selector,
            (unsigned)pipeline->background,
            (unsigned)pipeline->color,
            (unsigned)pipeline->output_color,
            (unsigned)pipeline->selector_kind,
            pipeline->include_background ? "true" : "false",
            pipeline->orientation,
            pipeline->count_mode,
            pipeline->x_align,
            pipeline->y_align,
            pipeline->x_anchor,
            pipeline->y_anchor,
            pipeline->x_offset,
            pipeline->y_offset,
            pipeline->width,
            pipeline->height,
            pipeline->period);
}

static int arc_probe_provenance_present(const char *coverage_frontier_id,
                                        const char *parent_attempt_id,
                                        const char *repair_operator,
                                        const char *dominant_residual_class,
                                        const char *dominant_failure_stage,
                                        const char *repair_strategy)
{
    return (coverage_frontier_id != NULL && coverage_frontier_id[0] != '\0') ||
           (parent_attempt_id != NULL && parent_attempt_id[0] != '\0') ||
           (repair_operator != NULL && repair_operator[0] != '\0') ||
           (dominant_residual_class != NULL && dominant_residual_class[0] != '\0') ||
           (dominant_failure_stage != NULL && dominant_failure_stage[0] != '\0') ||
           (repair_strategy != NULL && repair_strategy[0] != '\0');
}

static void arc_write_probe_provenance(FILE *fp,
                                       const char *coverage_frontier_id,
                                       const char *parent_attempt_id,
                                       const char *repair_operator,
                                       const char *dominant_residual_class,
                                       const char *dominant_failure_stage,
                                       const char *repair_strategy)
{
    fputs("{\"coverage_frontier_id\":", fp);
    nn_write_json_string(fp, coverage_frontier_id == NULL ? "" : coverage_frontier_id);
    fputs(",\"parent_attempt_id\":", fp);
    nn_write_json_string(fp, parent_attempt_id == NULL ? "" : parent_attempt_id);
    fputs(",\"repair_operator\":", fp);
    nn_write_json_string(fp, repair_operator == NULL ? "" : repair_operator);
    fputs(",\"dominant_residual_class\":", fp);
    nn_write_json_string(fp,
                         dominant_residual_class == NULL ? "" : dominant_residual_class);
    fputs(",\"dominant_failure_stage\":", fp);
    nn_write_json_string(fp,
                         dominant_failure_stage == NULL ? "" : dominant_failure_stage);
    fputs(",\"repair_strategy\":", fp);
    nn_write_json_string(fp, repair_strategy == NULL ? "" : repair_strategy);
    fputs("}", fp);
}

static int arc_write_visual_pipeline_attempts(
    const char *path,
    const ARCVisualPipelineAttempt *attempts,
    size_t attempt_count)
{
    if (path == NULL) {
        return 1;
    }
    FILE *fp = fopen(path, "w");
    if (fp == NULL) {
        return 0;
    }
    for (size_t i = 0; i < attempt_count; ++i) {
        const ARCVisualPipelineAttempt *attempt = &attempts[i];
        char id[ARC_TRANSFORM_NAME_BYTES * 2];
        fputs("{\"schema\":\"crlp.neural_grammar_discovery.arc_visual_pipeline_attempt.v1\","
              "\"report_version\":1,\"attempt_id\":",
              fp);
        arc_visual_pipeline_attempt_id(attempt, id, sizeof(id));
        nn_write_json_string(fp, id);
        fputs(",\"library_id\":", fp);
        arc_visual_pipeline_library_id(attempt, id, sizeof(id));
        nn_write_json_string(fp, id);
        fputs(",\"family\":", fp);
        nn_write_json_string(fp, attempt->family);
        fputs(",\"task_id\":", fp);
        nn_write_json_string(fp, attempt->task_id);
        fputs(",\"split\":", fp);
        nn_write_json_string(fp, attempt->split);
        fputs(",\"task_path\":", fp);
        nn_write_json_string(fp, attempt->task_path);
        if (arc_probe_provenance_present(attempt->coverage_frontier_id,
                                         attempt->parent_attempt_id,
                                         attempt->repair_operator,
                                         attempt->dominant_residual_class,
                                         attempt->dominant_failure_stage,
                                         attempt->repair_strategy)) {
            fputs(",\"candidate_provenance\":", fp);
            arc_write_probe_provenance(fp,
                                       attempt->coverage_frontier_id,
                                       attempt->parent_attempt_id,
                                       attempt->repair_operator,
                                       attempt->dominant_residual_class,
                                       attempt->dominant_failure_stage,
                                       attempt->repair_strategy);
        }
        fprintf(fp,
                ",\"attempt_index\":%zu,\"accepted\":%s,\"rejection_reason\":",
                attempt->attempt_index,
                attempt->accepted ? "true" : "false");
        nn_write_json_string(fp, arc_sketch_probe_reason_name(attempt->reason));
        fputs(",\"failure_stage\":", fp);
        nn_write_json_string(fp, attempt->failure_stage);
        fputs(",\"failure_detail\":", fp);
        nn_write_json_string(fp, attempt->failure_detail);
        fputs(",\"pipeline\":", fp);
        arc_write_visual_pipeline_values(fp, &attempt->pipeline);
        fputs(",\"slot_values\":", fp);
        arc_write_sketch_probe_slot_values(fp, &attempt->spec);
        fputs(",\"pipeline_kind\":", fp);
        nn_write_json_string(fp, arc_visual_pipeline_name(attempt->pipeline.kind));
        fputs(",\"op_name\":", fp);
        nn_write_json_string(fp, arc_sketch_op_name(attempt->spec.op));
        const ARCVisualOpMetadata *metadata =
            arc_visual_op_metadata_for_op(attempt->spec.op);
        if (metadata != NULL) {
            fputs(",\"visual_program\":", fp);
            nn_write_json_string(fp, metadata->visual_program);
            fputs(",\"slot_contract\":", fp);
            nn_write_json_string(fp, metadata->slot_contract);
            fputs(",\"composition_kind\":", fp);
            nn_write_json_string(fp, metadata->composition_kind);
            fputs(",\"normalization_frame\":", fp);
            nn_write_json_string(fp, metadata->normalization_frame);
            fputs(",\"typed_stages\":", fp);
            nn_write_json_string(fp, metadata->typed_stages);
            fputs(",\"primitive_contract\":", fp);
            nn_write_json_string(fp, metadata->primitive_contract);
        }
        if (attempt->has_contrast_accepted &&
            attempt->contrast_accepted_log_index < attempt_count) {
            const ARCVisualPipelineAttempt *accepted =
                &attempts[attempt->contrast_accepted_log_index];
            fputs(",\"contrast_pair_kind\":\"same_task_same_op_active_slot\","
                  "\"contrast_accepted_attempt_id\":",
                  fp);
            arc_visual_pipeline_attempt_id(accepted, id, sizeof(id));
            nn_write_json_string(fp, id);
            fprintf(fp,
                    ",\"contrast_slot_distance\":%zu,"
                    "\"contrast_rejected_sibling_count\":0",
                    attempt->contrast_slot_distance);
        } else {
            fprintf(fp,
                    ",\"contrast_pair_kind\":\"%s\","
                    "\"contrast_accepted_attempt_id\":\"\","
                    "\"contrast_slot_distance\":0,"
                    "\"contrast_rejected_sibling_count\":%zu",
                    attempt->accepted ? "accepted_anchor" : "",
                    attempt->contrast_rejected_sibling_count);
        }
        fprintf(fp,
                ",\"task_slot_features\":{\"input_width\":%zu,"
                "\"input_height\":%zu,\"output_width\":%zu,"
                "\"output_height\":%zu,\"shape_scale_x\":%zu,"
                "\"shape_scale_y\":%zu,\"period_divides_input_width\":%s,"
                "\"period_divides_input_height\":%s,"
                "\"slot_width_matches_output\":%s,"
                "\"slot_height_matches_output\":%s,"
                "\"color_in_input\":%s,\"color_in_output\":%s,"
                "\"output_color_in_output\":%s,"
                "\"background_in_input\":%s},"
                "\"component_inventory\":{\"input_component_count\":%zu,"
                "\"selected_component_count\":%zu,"
                "\"selected_component_area_min\":%zu,"
                "\"selected_component_area_max\":%zu,"
                "\"selected_component_area_total\":%zu,"
                "\"selected_component_same_shape\":%s,"
                "\"selected_component_shape_count\":%zu,"
                "\"selected_component_bbox_width\":%zu,"
                "\"selected_component_bbox_height\":%zu,"
                "\"output_foreground_cell_count\":%zu,"
                "\"output_component_count\":%zu,"
                "\"output_non_background_color_count\":%zu,"
                "\"selected_count_matches_output_components\":%s,"
                "\"selected_count_matches_output_foreground\":%s,"
                "\"selected_area_matches_output_foreground\":%s,"
                "\"selected_bbox_matches_output_shape\":%s,"
                "\"selected_bbox_rotated_output_shape\":%s},"
                "\"verifier_distance\":{\"train_pairs\":%zu,"
                "\"checked_cells\":%zu,\"mismatches\":%zu,"
                "\"unsupported_pairs\":%zu,\"apply_failed_pairs\":%zu,"
                "\"shape_mismatch_pairs\":%zu,\"missing_output_pairs\":%zu,"
                "\"first_failing_pair\":%ld,"
                "\"first_input_width\":%zu,\"first_input_height\":%zu,"
                "\"first_expected_width\":%zu,\"first_expected_height\":%zu,"
                "\"first_produced_width\":%zu,\"first_produced_height\":%zu,"
                "\"first_pair_checked_cells\":%zu,\"first_pair_mismatches\":%zu,"
                "\"first_mismatch_min_x\":%ld,\"first_mismatch_min_y\":%ld,"
                "\"first_mismatch_max_x\":%ld,\"first_mismatch_max_y\":%ld,"
                "\"mismatch_ratio\":%.8f,"
                "\"verifier_score\":%.8f},"
                "\"proposal_count\":%zu,\"support_tasks\":%zu,"
                "\"utility_mean\":%.8f}\n",
                attempt->input_width,
                attempt->input_height,
                attempt->output_width,
                attempt->output_height,
                attempt->shape_scale_x,
                attempt->shape_scale_y,
                attempt->period_divides_input_width ? "true" : "false",
                attempt->period_divides_input_height ? "true" : "false",
                attempt->slot_width_matches_output ? "true" : "false",
                attempt->slot_height_matches_output ? "true" : "false",
                attempt->color_in_input ? "true" : "false",
                attempt->color_in_output ? "true" : "false",
                attempt->output_color_in_output ? "true" : "false",
                attempt->background_in_input ? "true" : "false",
                attempt->component_inventory.input_component_count,
                attempt->component_inventory.selected_component_count,
                attempt->component_inventory.selected_component_area_min,
                attempt->component_inventory.selected_component_area_max,
                attempt->component_inventory.selected_component_area_total,
                attempt->component_inventory.selected_component_same_shape
                    ? "true"
                    : "false",
                attempt->component_inventory.selected_component_shape_count,
                attempt->component_inventory.selected_component_bbox_width,
                attempt->component_inventory.selected_component_bbox_height,
                attempt->component_inventory.output_foreground_cell_count,
                attempt->component_inventory.output_component_count,
                attempt->component_inventory.output_non_background_color_count,
                attempt->component_inventory.selected_count_matches_output_components
                    ? "true"
                    : "false",
                attempt->component_inventory.selected_count_matches_output_foreground
                    ? "true"
                    : "false",
                attempt->component_inventory.selected_area_matches_output_foreground
                    ? "true"
                    : "false",
                attempt->component_inventory.selected_bbox_matches_output_shape
                    ? "true"
                    : "false",
                attempt->component_inventory.selected_bbox_rotated_output_shape
                    ? "true"
                    : "false",
                attempt->telemetry.train_pairs,
                attempt->telemetry.checked_cells,
                attempt->telemetry.mismatches,
                attempt->telemetry.unsupported_pairs,
                attempt->telemetry.apply_failed_pairs,
                attempt->telemetry.shape_mismatch_pairs,
                attempt->telemetry.missing_output_pairs,
                attempt->telemetry.first_failing_pair == SIZE_MAX
                    ? -1L
                    : (long)attempt->telemetry.first_failing_pair,
                attempt->telemetry.first_input_width,
                attempt->telemetry.first_input_height,
                attempt->telemetry.first_expected_width,
                attempt->telemetry.first_expected_height,
                attempt->telemetry.first_produced_width,
                attempt->telemetry.first_produced_height,
                attempt->telemetry.first_pair_checked_cells,
                attempt->telemetry.first_pair_mismatches,
                attempt->telemetry.first_mismatch_min_x == SIZE_MAX
                    ? -1L
                    : (long)attempt->telemetry.first_mismatch_min_x,
                attempt->telemetry.first_mismatch_min_y == SIZE_MAX
                    ? -1L
                    : (long)attempt->telemetry.first_mismatch_min_y,
                attempt->telemetry.first_mismatch_max_x == SIZE_MAX
                    ? -1L
                    : (long)attempt->telemetry.first_mismatch_max_x,
                attempt->telemetry.first_mismatch_max_y == SIZE_MAX
                    ? -1L
                    : (long)attempt->telemetry.first_mismatch_max_y,
                attempt->telemetry.checked_cells == 0
                    ? 1.0
                    : (double)attempt->telemetry.mismatches /
                          (double)attempt->telemetry.checked_cells,
                attempt->telemetry.verifier_score,
                attempt->proposal_count,
                attempt->support_tasks,
                attempt->utility_mean);
    }
    int ok = !ferror(fp);
    fclose(fp);
    return ok;
}

static int arc_write_sketch_probe_sources(const char *path,
                                          const ARCSketchProbeRecord *records,
                                          size_t record_count,
                                          const ARCSketchProbeSummary *summaries,
                                          size_t summary_count)
{
    FILE *fp = fopen(path, "w");
    if (fp == NULL) {
        return 0;
    }
    for (size_t i = 0; i < summary_count; ++i) {
        const ARCSketchProbeSummary *summary = &summaries[i];
        fputs("{\"schema\":\"crlp.neural_grammar_discovery.arc_sketch_program_probe_summary.v1\","
              "\"report_version\":1,\"family\":",
              fp);
        nn_write_json_string(fp, summary->family);
        fputs(",\"task_id\":", fp);
        nn_write_json_string(fp, summary->task_id);
        fputs(",\"split\":", fp);
        nn_write_json_string(fp, summary->split);
        fputs(",\"task_path\":", fp);
        nn_write_json_string(fp, summary->task_path);
        fprintf(fp,
                ",\"attempted\":%zu,\"emitted\":%zu,"
                "\"rejection_counts\":{\"pruned_infeasible\":%zu,"
                "\"unsupported_shape\":%zu,"
                "\"no_train_match\":%zu,\"duplicate\":%zu,\"cap_hit\":%zu}}\n",
                summary->attempted,
                summary->emitted,
                summary->pruned_infeasible,
                summary->unsupported_shape,
                summary->no_train_match,
                summary->duplicate,
                summary->cap_hit);
    }
    for (size_t i = 0; i < record_count; ++i) {
        const ARCSketchProbeRecord *record = &records[i];
        fputs("{\"schema\":\"crlp.neural_grammar_discovery.arc_sketch_program_probe.v1\","
              "\"report_version\":1,\"library_id\":",
              fp);
        nn_write_json_string(fp, record->library_id);
        fputs(",\"template_id\":1036,\"template_name\":\"arc_sketch_program\",\"family\":", fp);
        nn_write_json_string(fp, record->family);
        fprintf(fp, ",\"candidate_group_size\":%zu,\"task_id\":", record->support_tasks);
        nn_write_json_string(fp, record->task_id);
        fputs(",\"split\":", fp);
        nn_write_json_string(fp, record->split);
        fputs(",\"task_path\":", fp);
        nn_write_json_string(fp, record->task_path);
        if (arc_probe_provenance_present(record->coverage_frontier_id,
                                         record->parent_attempt_id,
                                         record->repair_operator,
                                         record->dominant_residual_class,
                                         record->dominant_failure_stage,
                                         record->repair_strategy)) {
            fputs(",\"candidate_provenance\":", fp);
            arc_write_probe_provenance(fp,
                                       record->coverage_frontier_id,
                                       record->parent_attempt_id,
                                       record->repair_operator,
                                       record->dominant_residual_class,
                                       record->dominant_failure_stage,
                                       record->repair_strategy);
        }
        fprintf(fp,
                ",\"sketch_index\":%zu,\"slot_values\":",
                record->sketch_index);
        arc_write_sketch_probe_slot_values(fp, &record->spec);
        fprintf(fp,
                ",\"proposal_count\":%zu,\"support_tasks\":%zu,"
                "\"utility_mean\":%.8f,\"compiler\":\"arc_sketch_program_probe_compiler\","
                "\"op_name\":",
                record->proposal_count,
                record->support_tasks,
                record->utility_mean);
        nn_write_json_string(fp, arc_sketch_op_name(record->spec.op));
        const ARCVisualOpMetadata *metadata =
            arc_visual_op_metadata_for_op(record->spec.op);
        if (metadata != NULL) {
            fputs(",\"visual_program\":", fp);
            nn_write_json_string(fp, metadata->visual_program);
            fputs(",\"slot_contract\":", fp);
            nn_write_json_string(fp, metadata->slot_contract);
            fputs(",\"composition_kind\":", fp);
            nn_write_json_string(fp, metadata->composition_kind);
            fputs(",\"normalization_frame\":", fp);
            nn_write_json_string(fp, metadata->normalization_frame);
            fputs(",\"typed_stages\":", fp);
            nn_write_json_string(fp, metadata->typed_stages);
            fputs(",\"primitive_contract\":", fp);
            nn_write_json_string(fp, metadata->primitive_contract);
        }
        fputs(",\"promotion_path\":\"tsv_prior_loaded_by_arc_agi_verifier\"}\n", fp);
    }
    int ok = !ferror(fp);
    fclose(fp);
    return ok;
}

static void arc_write_sketch_probe_prior_row(FILE *fp,
                                             const char *schema,
                                             const ARCSketchProbeRecord *record,
                                             const char *slot_name,
                                             const char *slot_type,
                                             int64_t slot_integer,
                                             size_t certified_count)
{
    fprintf(fp,
            "%s\t1036\tarc_sketch_program\t%s\t%s\t%s\t%.8f\t%lld\t%zu\t%zu\t%zu\t%zu\t%.8f\n",
            schema,
            record->library_id,
            slot_name,
            slot_type,
            (double)slot_integer,
            (long long)slot_integer,
            record->proposal_count,
            certified_count,
            record->support_tasks,
            record->support_tasks,
            record->utility_mean);
}

static int arc_write_sketch_probe_prior(const char *path,
                                        const ARCSketchProbeRecord *records,
                                        size_t record_count)
{
    FILE *fp = fopen(path, "w");
    if (fp == NULL) {
        return 0;
    }
    fputs("schema_version\ttemplate_id\ttemplate_name\tlibrary_id\tslot_name\t"
          "slot_type\tslot_value\tslot_integer\tproposal_count\tcertified_count\t"
          "support_tasks\tcovered_events\tutility_mean\n",
          fp);
    for (size_t i = 0; i < record_count; ++i) {
        const ARCSketchProbeRecord *record = &records[i];
        size_t certified_count = record->support_tasks == 0 ? 1u : record->support_tasks;
        arc_write_sketch_probe_prior_row(fp,
                                         "crlp.hypothesis_library.template_prior.v1",
                                         record,
                                         "__template__",
                                         "template",
                                         0,
                                         certified_count);
        arc_write_sketch_probe_prior_row(fp,
                                         "crlp.hypothesis_library.prior.v1",
                                         record,
                                         "sketch.selector",
                                         "integer",
                                         record->spec.selector,
                                         certified_count);
        arc_write_sketch_probe_prior_row(fp,
                                         "crlp.hypothesis_library.prior.v1",
                                         record,
                                         "sketch.op",
                                         "integer",
                                         record->spec.op,
                                         certified_count);
        arc_write_sketch_probe_prior_row(fp,
                                         "crlp.hypothesis_library.prior.v1",
                                         record,
                                         "sketch.background",
                                         "color",
                                         record->spec.background,
                                         certified_count);
        arc_write_sketch_probe_prior_row(fp,
                                         "crlp.hypothesis_library.prior.v1",
                                         record,
                                         "sketch.color",
                                         "color",
                                         record->spec.color,
                                         certified_count);
        arc_write_sketch_probe_prior_row(fp,
                                         "crlp.hypothesis_library.prior.v1",
                                         record,
                                         "sketch.output_color",
                                         "color",
                                         record->spec.output_color,
                                         certified_count);
        arc_write_sketch_probe_prior_row(fp,
                                         "crlp.hypothesis_library.prior.v1",
                                         record,
                                         "sketch.x_anchor",
                                         "integer",
                                         record->spec.x_anchor,
                                         certified_count);
        arc_write_sketch_probe_prior_row(fp,
                                         "crlp.hypothesis_library.prior.v1",
                                         record,
                                         "sketch.y_anchor",
                                         "integer",
                                         record->spec.y_anchor,
                                         certified_count);
        arc_write_sketch_probe_prior_row(fp,
                                         "crlp.hypothesis_library.prior.v1",
                                         record,
                                         "sketch.x_offset",
                                         "offset",
                                         record->spec.x_offset,
                                         certified_count);
        arc_write_sketch_probe_prior_row(fp,
                                         "crlp.hypothesis_library.prior.v1",
                                         record,
                                         "sketch.y_offset",
                                         "offset",
                                         record->spec.y_offset,
                                         certified_count);
        arc_write_sketch_probe_prior_row(fp,
                                         "crlp.hypothesis_library.prior.v1",
                                         record,
                                         "sketch.width",
                                         "integer",
                                         (int64_t)record->spec.width,
                                         certified_count);
        arc_write_sketch_probe_prior_row(fp,
                                         "crlp.hypothesis_library.prior.v1",
                                         record,
                                         "sketch.height",
                                         "integer",
                                         (int64_t)record->spec.height,
                                         certified_count);
        arc_write_sketch_probe_prior_row(fp,
                                         "crlp.hypothesis_library.prior.v1",
                                         record,
                                         "sketch.period",
                                         "integer",
                                         (int64_t)record->spec.period,
                                         certified_count);
    }
    int ok = !ferror(fp);
    fclose(fp);
    return ok;
}

static int ARC_NOINLINE arc_export_sketch_program_probes(const ARCConfig *config)
{
    if (config == NULL ||
        config->arc_sketch_probe_candidates_path == NULL ||
        config->arc_sketch_probe_public_arc_dir == NULL ||
        config->arc_sketch_probe_sources_path == NULL ||
        config->arc_sketch_probe_prior_path == NULL) {
        return 0;
    }
    ARCSketchProbeInput *inputs =
        calloc(ARC_SKETCH_PROBE_INPUT_CAPACITY, sizeof(*inputs));
    ARCSketchProbeRecord *records =
        calloc(ARC_SKETCH_PROBE_RECORD_CAPACITY, sizeof(*records));
    ARCSketchProbeSummary *summaries =
        calloc(ARC_SKETCH_PROBE_INPUT_CAPACITY, sizeof(*summaries));
    ARCVisualPipelineAttemptLog attempt_log;
    memset(&attempt_log, 0, sizeof(attempt_log));
    if (inputs == NULL || records == NULL || summaries == NULL) {
        free(inputs);
        free(records);
        free(summaries);
        return 0;
    }
    size_t input_count = 0;
    size_t record_count = 0;
    size_t summary_count = 0;
    int ok = arc_read_sketch_probe_inputs(config->arc_sketch_probe_candidates_path,
                                          inputs,
                                          ARC_SKETCH_PROBE_INPUT_CAPACITY,
                                          &input_count);
    if (!ok) {
        free(inputs);
        free(records);
        free(summaries);
        return 0;
    }
    size_t cap = config->arc_sketch_probe_sketches_per_task == 0
                     ? 8u
                     : config->arc_sketch_probe_sketches_per_task;
    if (cap > 64u) {
        cap = 64u;
    }
    arc_active_visual_pipeline_attempt_log = &attempt_log;
    for (size_t i = 0; i < input_count; ++i) {
        ARCTask task;
        char task_path[ARC_MAX_TASK_PATH];
        arc_public_task_path(config->arc_sketch_probe_public_arc_dir,
                             inputs[i].split,
                             inputs[i].task_id,
                             task_path,
                             sizeof(task_path));
        ARCSketchProbeSummary *summary = &summaries[summary_count++];
        memset(summary, 0, sizeof(*summary));
        snprintf(summary->family, sizeof(summary->family), "%s", inputs[i].family);
        snprintf(summary->task_id, sizeof(summary->task_id), "%s", inputs[i].task_id);
        snprintf(summary->split, sizeof(summary->split), "%s", inputs[i].split);
        snprintf(summary->task_path, sizeof(summary->task_path), "%s", task_path);
        if (!arc_load_task_file(task_path, inputs[i].task_id, &task)) {
            ++summary->unsupported_shape;
            continue;
        }
        ARCSketchProgramSpec emitted_specs[64];
        size_t emitted_count = 0;
        memset(emitted_specs, 0, sizeof(emitted_specs));
        if (inputs[i].op_hints[0] != '\0' && strcmp(inputs[i].op_hints, "*") != 0) {
            arc_mine_learned_visual_grammar_sketches(&inputs[i],
                                                     task_path,
                                                     &task,
                                                     records,
                                                     &record_count,
                                                     emitted_specs,
                                                     &emitted_count,
                                                     cap,
                                                     summary,
                                                     &attempt_log);
            continue;
        }
        if (strcmp(inputs[i].family, "arc.patch_lattice_template_family") == 0 ||
            strcmp(inputs[i].family, "arc.tiling_canvas_growth_family") == 0) {
            if (arc_sketch_probe_input_allows_op(&inputs[i], ARC_SKETCH_OP_PALETTE_SCALE)) {
                arc_mine_palette_scale_sketches(&inputs[i],
                                                task_path,
                                                &task,
                                                records,
                                                &record_count,
                                                emitted_specs,
                                                &emitted_count,
                                                cap,
                                                summary,
                                                &attempt_log);
            }
            if (arc_sketch_probe_input_allows_op(&inputs[i], ARC_SKETCH_OP_MACROCELL_SUBSTITUTE)) {
                arc_mine_macrocell_sketches(&inputs[i],
                                            task_path,
                                            &task,
                                            records,
                                            &record_count,
                                            emitted_specs,
                                            &emitted_count,
                                            cap,
                                            summary);
            }
            if (arc_sketch_probe_input_allows_op(&inputs[i], ARC_SKETCH_OP_LATTICE_SAMPLE)) {
                arc_mine_lattice_sample_sketches(&inputs[i],
                                                 task_path,
                                                 &task,
                                                 records,
                                                 &record_count,
                                                 emitted_specs,
                                                 &emitted_count,
                                                 cap,
                                                 summary);
            }
            if (strcmp(inputs[i].family, "arc.patch_lattice_template_family") == 0 &&
                arc_sketch_probe_input_allows_op(&inputs[i], ARC_SKETCH_OP_POINT_LINE_PROJECT)) {
                arc_mine_point_line_project_sketches(&inputs[i],
                                                     task_path,
                                                     &task,
                                                     records,
                                                     &record_count,
                                                     emitted_specs,
                                                     &emitted_count,
                                                     cap,
                                                     summary);
            }
            if (strcmp(inputs[i].family, "arc.patch_lattice_template_family") == 0 &&
                arc_sketch_probe_input_allows_op(&inputs[i], ARC_SKETCH_OP_MARKER_POLYLINE)) {
                arc_mine_marker_polyline_sketches(&inputs[i],
                                                  task_path,
                                                  &task,
                                                  records,
                                                  &record_count,
                                                  emitted_specs,
                                                  &emitted_count,
                                                  cap,
                                                  summary);
            }
        }
        if (strcmp(inputs[i].family, "arc.separator_panel_extraction_family") == 0) {
            if (arc_sketch_probe_input_allows_op(&inputs[i], ARC_SKETCH_OP_SEPARATOR_PANEL_MAP)) {
                arc_mine_separator_panel_map_sketches(&inputs[i],
                                                     task_path,
                                                     &task,
                                                     records,
                                                     &record_count,
                                                     emitted_specs,
                                                     &emitted_count,
                                                     cap,
                                                     summary);
            }
            if (arc_sketch_probe_input_allows_op(&inputs[i], ARC_SKETCH_OP_SEPARATOR_INTERSECTION_COLOR)) {
                arc_mine_separator_intersection_color_sketches(&inputs[i],
                                                              task_path,
                                                              &task,
                                                              records,
                                                              &record_count,
                                                              emitted_specs,
                                                              &emitted_count,
                                                              cap,
                                                              summary);
            }
            if (arc_sketch_probe_input_allows_op(&inputs[i], ARC_SKETCH_OP_SEPARATOR_PANEL_RELATION_MASK)) {
                arc_mine_separator_panel_relation_mask_sketches(&inputs[i],
                                                               task_path,
                                                               &task,
                                                               records,
                                                               &record_count,
                                                               emitted_specs,
                                                               &emitted_count,
                                                               cap,
                                                               summary,
                                                               &attempt_log);
            }
            if (arc_sketch_probe_input_allows_op(&inputs[i], ARC_SKETCH_OP_COMPONENT_COLOR_SUMMARY)) {
                arc_mine_component_summary_sketches(&inputs[i],
                                                   task_path,
                                                   &task,
                                                   records,
                                                   &record_count,
                                                   emitted_specs,
                                                   &emitted_count,
                                                   cap,
                                                   summary);
            }
            if (arc_sketch_probe_input_allows_op(&inputs[i], ARC_SKETCH_OP_CROP) ||
                arc_sketch_probe_input_allows_op(&inputs[i], ARC_SKETCH_OP_NORMALIZED_CROP)) {
                arc_mine_crop_sketches(&inputs[i],
                                       task_path,
                                       &task,
                                       records,
                                       &record_count,
                                       emitted_specs,
                                       &emitted_count,
                                       cap,
                                       summary);
            }
            if (arc_sketch_probe_input_allows_op(&inputs[i], ARC_SKETCH_OP_RECOLOR_CROP)) {
                arc_mine_recolor_crop_repair_sketches(&inputs[i],
                                                      task_path,
                                                      &task,
                                                      records,
                                                      &record_count,
                                                      emitted_specs,
                                                      &emitted_count,
                                                      cap,
                                                      summary);
            }
        }
        if (strcmp(inputs[i].family, "arc.crop_object_selection_family") == 0) {
            if (arc_sketch_probe_input_allows_op(&inputs[i], ARC_SKETCH_OP_OBJECT_COLOR_TILE)) {
                arc_mine_object_color_tile_sketches(&inputs[i],
                                                   task_path,
                                                   &task,
                                                   records,
                                                   &record_count,
                                                   emitted_specs,
                                                   &emitted_count,
                                                   cap,
                                                   summary,
                                                   &attempt_log);
            }
            if (arc_sketch_probe_input_allows_op(&inputs[i], ARC_SKETCH_OP_OBJECT_MASK_EXTRACT)) {
                arc_mine_object_mask_extract_sketches(&inputs[i],
                                                     task_path,
                                                     &task,
                                                     records,
                                                     &record_count,
                                                     emitted_specs,
                                                     &emitted_count,
                                                     cap,
                                                     summary);
            }
            if (arc_sketch_probe_input_allows_op(&inputs[i], ARC_SKETCH_OP_LATTICE_SAMPLE)) {
                arc_mine_diagonal_compact_sketches(&inputs[i],
                                                   task_path,
                                                   &task,
                                                   records,
                                                   &record_count,
                                                   emitted_specs,
                                                   &emitted_count,
                                                   cap,
                                                   summary);
            }
            if (arc_sketch_probe_input_allows_op(&inputs[i], ARC_SKETCH_OP_CROP) ||
                arc_sketch_probe_input_allows_op(&inputs[i], ARC_SKETCH_OP_NORMALIZED_CROP)) {
                arc_mine_crop_sketches(&inputs[i],
                                       task_path,
                                       &task,
                                       records,
                                       &record_count,
                                       emitted_specs,
                                       &emitted_count,
                                       cap,
                                       summary);
            }
            if (arc_sketch_probe_input_allows_op(&inputs[i], ARC_SKETCH_OP_RECOLOR_CROP)) {
                arc_mine_recolor_crop_repair_sketches(&inputs[i],
                                                      task_path,
                                                      &task,
                                                      records,
                                                      &record_count,
                                                      emitted_specs,
                                                      &emitted_count,
                                                      cap,
                                                      summary);
            }
            if (arc_sketch_probe_input_allows_op(&inputs[i], ARC_SKETCH_OP_LATTICE_SAMPLE)) {
                arc_mine_lattice_sample_sketches(&inputs[i],
                                                 task_path,
                                                 &task,
                                                 records,
                                                 &record_count,
                                                 emitted_specs,
                                                 &emitted_count,
                                                 cap,
                                                 summary);
            }
            if (arc_sketch_probe_input_allows_op(&inputs[i], ARC_SKETCH_OP_VISUAL_COMBINATOR)) {
                arc_mine_visual_combinator_sketches(&inputs[i],
                                                   task_path,
                                                   &task,
                                                   records,
                                                   &record_count,
                                                   emitted_specs,
                                                   &emitted_count,
                                                   cap,
                                                   summary,
                                                   &attempt_log);
            }
        }
        if (strcmp(inputs[i].family, "arc.projection_compression_family") == 0) {
            if (arc_sketch_probe_input_allows_op(&inputs[i], ARC_SKETCH_OP_MARKER_COUNT_MOTIF_TILE)) {
                arc_mine_marker_count_motif_tile_sketches(&inputs[i],
                                                         task_path,
                                                         &task,
                                                         records,
                                                         &record_count,
                                                         emitted_specs,
                                                         &emitted_count,
                                                         cap,
                                                         summary,
                                                         &attempt_log);
            }
            if (arc_sketch_probe_input_allows_op(&inputs[i], ARC_SKETCH_OP_COMPONENT_COUNT_LINE)) {
                arc_mine_component_count_line_sketches(&inputs[i],
                                                      task_path,
                                                      &task,
                                                      records,
                                                      &record_count,
                                                      emitted_specs,
                                                      &emitted_count,
                                                      cap,
                                                      summary,
                                                      &attempt_log);
            }
            if (arc_sketch_probe_input_allows_op(&inputs[i], ARC_SKETCH_OP_COMPONENT_COLOR_SUMMARY)) {
                arc_mine_component_summary_sketches(&inputs[i],
                                                   task_path,
                                                   &task,
                                                   records,
                                                   &record_count,
                                                   emitted_specs,
                                                   &emitted_count,
                                                   cap,
                                                   summary);
            }
            if (arc_sketch_probe_input_allows_op(&inputs[i], ARC_SKETCH_OP_VISUAL_COMBINATOR)) {
                arc_mine_visual_combinator_sketches(&inputs[i],
                                                   task_path,
                                                   &task,
                                                   records,
                                                   &record_count,
                                                   emitted_specs,
                                                   &emitted_count,
                                                   cap,
                                                   summary,
                                                   &attempt_log);
            }
            if (arc_sketch_probe_input_allows_op(&inputs[i], ARC_SKETCH_OP_AXIS_PROJECTION_COMPRESS)) {
                arc_mine_axis_projection_compress_sketches(&inputs[i],
                                                          task_path,
                                                          &task,
                                                          records,
                                                          &record_count,
                                                          emitted_specs,
                                                          &emitted_count,
                                                          cap,
                                                          summary);
            }
            if (arc_sketch_probe_input_allows_op(&inputs[i], ARC_SKETCH_OP_SEPARATOR_PANEL_MAP)) {
                arc_mine_separator_panel_map_sketches(&inputs[i],
                                                     task_path,
                                                     &task,
                                                     records,
                                                     &record_count,
                                                     emitted_specs,
                                                     &emitted_count,
                                                     cap,
                                                     summary);
            }
            if (arc_sketch_probe_input_allows_op(&inputs[i], ARC_SKETCH_OP_SEPARATOR_INTERSECTION_COLOR)) {
                arc_mine_separator_intersection_color_sketches(&inputs[i],
                                                              task_path,
                                                              &task,
                                                              records,
                                                              &record_count,
                                                              emitted_specs,
                                                              &emitted_count,
                                                              cap,
                                                              summary);
            }
            if (arc_sketch_probe_input_allows_op(&inputs[i], ARC_SKETCH_OP_SEPARATOR_PANEL_RELATION_MASK)) {
                arc_mine_separator_panel_relation_mask_sketches(&inputs[i],
                                                               task_path,
                                                               &task,
                                                               records,
                                                               &record_count,
                                                               emitted_specs,
                                                               &emitted_count,
                                                               cap,
                                                               summary,
                                                               &attempt_log);
            }
        }
        if (strcmp(inputs[i].family, "arc.compositional_transform_family") == 0) {
            if (arc_sketch_probe_input_allows_op(&inputs[i], ARC_SKETCH_OP_PALETTE_SCALE)) {
                arc_mine_palette_scale_sketches(&inputs[i],
                                                task_path,
                                                &task,
                                                records,
                                                &record_count,
                                                emitted_specs,
                                                &emitted_count,
                                                cap,
                                                summary,
                                                &attempt_log);
            }
            if (arc_sketch_probe_input_allows_op(&inputs[i], ARC_SKETCH_OP_OBJECT_COLOR_TILE)) {
                arc_mine_object_color_tile_sketches(&inputs[i],
                                                   task_path,
                                                   &task,
                                                   records,
                                                   &record_count,
                                                   emitted_specs,
                                                   &emitted_count,
                                                   cap,
                                                   summary,
                                                   &attempt_log);
            }
            if (arc_sketch_probe_input_allows_op(&inputs[i], ARC_SKETCH_OP_OBJECT_MASK_EXTRACT)) {
                arc_mine_object_mask_extract_sketches(&inputs[i],
                                                     task_path,
                                                     &task,
                                                     records,
                                                     &record_count,
                                                     emitted_specs,
                                                     &emitted_count,
                                                     cap,
                                                     summary);
            }
            if (arc_sketch_probe_input_allows_op(&inputs[i], ARC_SKETCH_OP_VISUAL_COMBINATOR)) {
                arc_mine_visual_combinator_sketches(&inputs[i],
                                                   task_path,
                                                   &task,
                                                   records,
                                                   &record_count,
                                                   emitted_specs,
                                                   &emitted_count,
                                                   cap,
                                                   summary,
                                                   &attempt_log);
            }
            if (arc_sketch_probe_input_allows_op(&inputs[i], ARC_SKETCH_OP_RECOLOR_CROP)) {
                arc_mine_recolor_crop_repair_sketches(&inputs[i],
                                                      task_path,
                                                      &task,
                                                      records,
                                                      &record_count,
                                                      emitted_specs,
                                                      &emitted_count,
                                                      cap,
                                                      summary);
            }
            if (arc_sketch_probe_input_allows_op(&inputs[i], ARC_SKETCH_OP_AXIS_PROJECTION_COMPRESS)) {
                arc_mine_axis_projection_compress_sketches(&inputs[i],
                                                          task_path,
                                                          &task,
                                                          records,
                                                          &record_count,
                                                          emitted_specs,
                                                          &emitted_count,
                                                          cap,
                                                          summary);
            }
            if (arc_sketch_probe_input_allows_op(&inputs[i], ARC_SKETCH_OP_SEPARATOR_PANEL_MAP)) {
                arc_mine_separator_panel_map_sketches(&inputs[i],
                                                     task_path,
                                                     &task,
                                                     records,
                                                     &record_count,
                                                     emitted_specs,
                                                     &emitted_count,
                                                     cap,
                                                     summary);
            }
            if (arc_sketch_probe_input_allows_op(&inputs[i], ARC_SKETCH_OP_SEPARATOR_INTERSECTION_COLOR)) {
                arc_mine_separator_intersection_color_sketches(&inputs[i],
                                                              task_path,
                                                              &task,
                                                              records,
                                                              &record_count,
                                                              emitted_specs,
                                                              &emitted_count,
                                                              cap,
                                                              summary);
            }
            if (arc_sketch_probe_input_allows_op(&inputs[i], ARC_SKETCH_OP_SEPARATOR_PANEL_RELATION_MASK)) {
                arc_mine_separator_panel_relation_mask_sketches(&inputs[i],
                                                               task_path,
                                                               &task,
                                                               records,
                                                               &record_count,
                                                               emitted_specs,
                                                               &emitted_count,
                                                               cap,
                                                               summary,
                                                               &attempt_log);
            }
        }
    }
    arc_active_visual_pipeline_attempt_log = NULL;
    arc_visual_pipeline_attempts_link_contrasts(attempt_log.items, attempt_log.count);
    ok = arc_write_sketch_probe_sources(config->arc_sketch_probe_sources_path,
                                        records,
                                        record_count,
                                        summaries,
                                        summary_count) &&
         arc_write_sketch_probe_prior(config->arc_sketch_probe_prior_path,
                                      records,
                                      record_count) &&
         arc_write_visual_pipeline_attempts(config->arc_sketch_probe_attempts_path,
                                            attempt_log.items,
                                            attempt_log.count);
    fprintf(stderr,
            "exported %zu ARC sketch program probes and %zu visual pipeline attempts from %zu promoted rows\n",
            record_count,
            attempt_log.count,
            input_count);
    free(inputs);
    free(records);
    free(summaries);
    free(attempt_log.items);
    return ok;
}
