typedef enum {
    ARC_HYPOTHESIS_INSERT_DROPPED = 0,
    ARC_HYPOTHESIS_INSERT_ADDED = 1,
    ARC_HYPOTHESIS_INSERT_REPLACED = 2,
} ARCHypothesisInsertResult;

static int arc_hypothesis_record_pre_cap_proposal(ARCHypothesisSelectionStats *stats,
                                                  size_t template_index,
                                                  const ARCTask *task,
                                                  const ARCTransformCandidate *candidate,
                                                  int train_filter_enabled,
                                                  int *train_consistent_out)
{
    if (task == NULL || candidate == NULL) {
        return 0;
    }
    int train_consistent =
        arc_candidate_matches_train(task, candidate, NULL, NULL, NULL);
    if (train_consistent_out != NULL) {
        *train_consistent_out = train_consistent;
    }
    if (stats == NULL) {
        return train_filter_enabled ? train_consistent : 1;
    }
    ++stats->proposals_attempted;
    ++stats->proposals_generated;
    if (template_index < ARC_HYPOTHESIS_TEMPLATE_REGISTRY_CAPACITY) {
        ++stats->template_attempted[template_index];
        ++stats->template_generated[template_index];
    }
    if (train_consistent) {
        ++stats->proposals_train_consistent_pre_cap;
        if (template_index < ARC_HYPOTHESIS_TEMPLATE_REGISTRY_CAPACITY) {
            ++stats->template_train_consistent_pre_cap[template_index];
        }
    }
    return train_filter_enabled ? train_consistent : 1;
}

static ARCHypothesisInsertResult arc_hypothesis_insert_proposal(
    ARCHypothesisProposal *proposals,
    size_t *proposal_count,
    size_t proposal_capacity,
    const ARCHypothesisProposal *proposal)
{
    if (proposals == NULL || proposal_count == NULL ||
        proposal_capacity == 0 || proposal == NULL) {
        return ARC_HYPOTHESIS_INSERT_DROPPED;
    }
    size_t count = *proposal_count;
    size_t pos = count;
    ARCHypothesisInsertResult result = ARC_HYPOTHESIS_INSERT_ADDED;
    if (count < proposal_capacity) {
        proposals[pos] = *proposal;
        ++count;
    } else if (proposal->score > proposals[count - 1u].score) {
        proposals[count - 1u] = *proposal;
        pos = count - 1u;
        result = ARC_HYPOTHESIS_INSERT_REPLACED;
    } else {
        return ARC_HYPOTHESIS_INSERT_DROPPED;
    }

    while (pos > 0 && proposals[pos].score > proposals[pos - 1u].score) {
        ARCHypothesisProposal tmp = proposals[pos - 1u];
        proposals[pos - 1u] = proposals[pos];
        proposals[pos] = tmp;
        --pos;
    }
    *proposal_count = count;
    return result;
}

static void arc_hypothesis_record_insert_result(ARCHypothesisSelectionStats *stats,
                                                size_t template_index,
                                                ARCHypothesisInsertResult insert_result)
{
    if (stats == NULL) {
        return;
    }
    if (insert_result == ARC_HYPOTHESIS_INSERT_REPLACED) {
        ++stats->proposals_heap_replaced;
    } else if (insert_result == ARC_HYPOTHESIS_INSERT_DROPPED) {
        ++stats->proposals_heap_dropped;
        if (template_index < ARC_HYPOTHESIS_TEMPLATE_REGISTRY_CAPACITY) {
            ++stats->template_heap_dropped[template_index];
        }
    }
}

static size_t arc_try_add_hypothesis_proposal(
    ARCProgramLearnedState *learned,
    const ARCTask *task,
    const ARCTransformCandidate *existing_candidates,
    size_t existing_count,
    ARCHypothesisProposal *proposals,
    size_t proposal_capacity,
    size_t proposal_count,
    ARCHypothesisSelectionStats *stats,
    size_t template_index,
    const ARCProgramTaskFeatures *task_features,
    const NNHypothesis *hypothesis,
    const ARCTransformCandidate *candidate,
    const ARCObject *input_object,
    const ARCObject *output_object)
{
    if (task == NULL || existing_candidates == NULL || proposals == NULL ||
        proposal_capacity == 0 || hypothesis == NULL || candidate == NULL ||
        candidate->name[0] == '\0' ||
        arc_candidate_name_exists(existing_candidates, existing_count, candidate->name) ||
        arc_hypothesis_proposal_name_exists(proposals,
                                            proposal_count,
                                            candidate->name)) {
        return proposal_count;
    }
    int train_consistent_pre_cap = 0;
    if (!arc_hypothesis_record_pre_cap_proposal(stats,
                                                template_index,
                                                task,
                                                candidate,
                                                learned == NULL
                                                    ? 1
                                                    : learned->hypothesis_train_filter_enabled,
                                                &train_consistent_pre_cap)) {
        return proposal_count;
    }
    ARCHypothesisProposal proposal;
    proposal.hypothesis = *hypothesis;
    proposal.candidate = *candidate;
    proposal.score = arc_hypothesis_slot_proposal_score(learned,
                                                        hypothesis,
                                                        candidate,
                                                        input_object,
                                                        output_object,
                                                        task_features,
                                                        train_consistent_pre_cap);
    ARCHypothesisInsertResult insert_result =
        arc_hypothesis_insert_proposal(proposals,
                                       &proposal_count,
                                       proposal_capacity,
                                       &proposal);
    arc_hypothesis_record_insert_result(stats, template_index, insert_result);
    return proposal_count;
}

static size_t arc_grid_nonzero_colors(const ARCGrid *grid,
                                      uint8_t *colors,
                                      size_t capacity)
{
    uint8_t seen[ARC_COLOR_COUNT];
    size_t count = 0;
    if (grid == NULL || colors == NULL || capacity == 0) {
        return 0;
    }
    memset(seen, 0, sizeof(seen));
    for (size_t cell = 0; cell < arc_grid_cells(grid); ++cell) {
        uint8_t color = grid->cells[cell];
        if (color >= ARC_COLOR_COUNT || color == 0 || seen[color]) {
            continue;
        }
        seen[color] = 1;
        if (count < capacity) {
            colors[count++] = color;
        }
    }
    return count;
}

static size_t arc_grid_changed_output_colors(const ARCGrid *input,
                                             const ARCGrid *output,
                                             uint8_t *colors,
                                             size_t capacity)
{
    uint8_t seen[ARC_COLOR_COUNT];
    size_t count = 0;
    if (input == NULL || output == NULL || colors == NULL || capacity == 0 ||
        !arc_grid_same_shape(input, output)) {
        return 0;
    }
    memset(seen, 0, sizeof(seen));
    for (size_t cell = 0; cell < arc_grid_cells(output); ++cell) {
        uint8_t color = output->cells[cell];
        if (color >= ARC_COLOR_COUNT || color == 0 ||
            color == input->cells[cell] || seen[color]) {
            continue;
        }
        seen[color] = 1;
        if (count < capacity) {
            colors[count++] = color;
        }
    }
    return count;
}

static int arc_grid_has_changed_cells(const ARCGrid *input, const ARCGrid *output)
{
    if (input == NULL || output == NULL || !arc_grid_same_shape(input, output)) {
        return 0;
    }
    for (size_t cell = 0; cell < arc_grid_cells(output); ++cell) {
        if (input->cells[cell] != output->cells[cell]) {
            return 1;
        }
    }
    return 0;
}

static size_t arc_grid_color_cell_count(const ARCGrid *grid, uint8_t color)
{
    size_t count = 0;
    if (grid == NULL) {
        return 0;
    }
    for (size_t cell = 0; cell < arc_grid_cells(grid); ++cell) {
        count += grid->cells[cell] == color ? 1u : 0u;
    }
    return count;
}

static int arc_color_map_has_changed_color(const ARCTransformCandidate *candidate)
{
    if (candidate == NULL) {
        return 0;
    }
    for (uint8_t color = 0; color < ARC_COLOR_COUNT; ++color) {
        if (candidate->color_map[color] != color) {
            return 1;
        }
    }
    return 0;
}

static size_t arc_induce_color_map_hypotheses(const ARCHypothesisTemplateBinding *binding,
                                              const ARCTask *task,
                                              ARCProgramLearnedState *learned,
                                              const ARCTransformCandidate *existing_candidates,
                                              size_t existing_count,
                                              ARCHypothesisProposal *proposals,
                                              size_t proposal_capacity,
                                              size_t proposal_count,
                                              ARCHypothesisSelectionStats *stats,
                                              size_t template_index)
{
    if (binding == NULL || binding->template_fn == NULL ||
        binding->to_candidate == NULL ||
        task == NULL || task->train_count == 0 ||
        existing_candidates == NULL || proposals == NULL ||
        proposal_capacity == 0) {
        return proposal_count;
    }

    ARCTransformCandidate observed;
    if (!arc_build_observed_color_map_candidate(task, &observed) ||
        !arc_color_map_has_changed_color(&observed)) {
        return proposal_count;
    }

    NNHypothesis hypothesis;
    ARCTransformCandidate candidate;
    if (!nn_hypothesis_init(&hypothesis, binding->template_fn())) {
        return proposal_count;
    }
    int ok = 1;
    for (uint8_t color = 0; color < ARC_COLOR_COUNT; ++color) {
        char slot_name[16];
        arc_color_map_slot_name(color, slot_name, sizeof(slot_name));
        ok = ok && nn_hypothesis_bind(&hypothesis,
                                      slot_name,
                                      (double)observed.color_map[color],
                                      observed.color_map[color]);
    }
    if (!ok || !binding->to_candidate(&hypothesis, &candidate)) {
        return proposal_count;
    }

    ARCProgramTaskFeatures task_features;
    arc_program_task_features(task, &task_features);
    return arc_try_add_hypothesis_proposal(learned,
                                           task,
                                           existing_candidates,
                                           existing_count,
                                           proposals,
                                           proposal_capacity,
                                           proposal_count,
                                           stats,
                                           template_index,
                                           &task_features,
                                           &hypothesis,
                                           &candidate,
                                           NULL,
                                           NULL);
}

static size_t arc_induce_translate_hypotheses(const ARCHypothesisTemplateBinding *binding,
                                              const ARCTask *task,
                                              ARCProgramLearnedState *learned,
                                              const ARCTransformCandidate *existing_candidates,
                                              size_t existing_count,
                                              ARCHypothesisProposal *proposals,
                                              size_t proposal_capacity,
                                              size_t proposal_count,
                                              ARCHypothesisSelectionStats *stats,
                                              size_t template_index)
{
    if (binding == NULL || binding->template_fn == NULL ||
        binding->to_candidate == NULL ||
        task == NULL || task->train_count == 0 ||
        existing_candidates == NULL || proposals == NULL ||
        proposal_capacity == 0) {
        return proposal_count;
    }

    ARCProgramTaskFeatures task_features;
    arc_program_task_features(task, &task_features);
    uint8_t backgrounds[2];
    size_t background_count =
        arc_task_relation_backgrounds(task,
                                      backgrounds,
                                      sizeof(backgrounds) / sizeof(backgrounds[0]));
    for (size_t background_index = 0;
         background_index < background_count;
         ++background_index) {
        uint8_t background = backgrounds[background_index];
        for (size_t train_pair = 0; train_pair < task->train_count; ++train_pair) {
            ARCScene input_scene;
            ARCScene output_scene;
            if (!task->train[train_pair].has_output ||
                !arc_extract_scene(&task->train[train_pair].input,
                                   background,
                                   &input_scene) ||
                !arc_extract_scene(&task->train[train_pair].output,
                                   background,
                                   &output_scene)) {
                continue;
            }
            for (size_t input_index = 0;
                 input_index < input_scene.object_count;
                 ++input_index) {
                const ARCObject *input_object = &input_scene.objects[input_index];
                uint8_t selector_kind = ARC_OBJECT_SELECTOR_UNIQUE_COLOR;
                uint8_t reference_color = 0;
                if (input_object->color == background ||
                    !arc_infer_selector_for_scene_object(&input_scene,
                                                         input_object,
                                                         &selector_kind,
                                                         &reference_color)) {
                    continue;
                }
                int reference_color_used =
                    selector_kind == ARC_OBJECT_SELECTOR_NEAREST_TO_COLOR;
                for (size_t output_index = 0;
                     output_index < output_scene.object_count;
                     ++output_index) {
                    const ARCObject *output_object = &output_scene.objects[output_index];
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

                    NNHypothesis hypothesis;
                    ARCTransformCandidate candidate;
                    if (!nn_hypothesis_init(&hypothesis, binding->template_fn())) {
                        continue;
                    }
                    int ok = nn_hypothesis_bind(&hypothesis,
                                                "object_scene.background",
                                                (double)background,
                                                background);
                    ok = ok && nn_hypothesis_bind(&hypothesis,
                                                  "select.selector_kind",
                                                  (double)selector_kind,
                                                  selector_kind);
                    ok = ok && nn_hypothesis_bind(&hypothesis,
                                                  "select.color",
                                                  (double)input_object->color,
                                                  input_object->color);
                    ok = ok && nn_hypothesis_bind(&hypothesis,
                                                  "select.reference_color_used",
                                                  (double)reference_color_used,
                                                  reference_color_used);
                    ok = ok && nn_hypothesis_bind(&hypothesis,
                                                  "select.reference_color",
                                                  (double)reference_color,
                                                  reference_color);
                    ok = ok && nn_hypothesis_bind(&hypothesis,
                                                  "translate.dx",
                                                  (double)dx,
                                                  dx);
                    ok = ok && nn_hypothesis_bind(&hypothesis,
                                                  "translate.dy",
                                                  (double)dy,
                                                  dy);
                    ok = ok && nn_hypothesis_bind(&hypothesis,
                                                  "render.background",
                                                  (double)background,
                                                  background);
                    ok = ok && nn_hypothesis_bind(&hypothesis,
                                                  "render.output_color",
                                                  (double)output_object->color,
                                                  output_object->color);
                    if (!ok ||
                        !binding->to_candidate(&hypothesis, &candidate)) {
                        continue;
                    }
                    proposal_count =
                        arc_try_add_hypothesis_proposal(learned,
                                                        task,
                                                        existing_candidates,
                                                        existing_count,
                                                        proposals,
                                                        proposal_capacity,
                                                        proposal_count,
                                                        stats,
                                                        template_index,
                                                        &task_features,
                                                        &hypothesis,
                                                        &candidate,
                                                        input_object,
                                                        output_object);
                }
            }
        }
    }

    return proposal_count;
}

static void arc_consider_move_to_anchor_proposal(
    const ARCHypothesisTemplateBinding *binding,
    ARCProgramLearnedState *learned,
    const ARCTask *task,
    const ARCTransformCandidate *existing_candidates,
    size_t existing_count,
    ARCHypothesisProposal *proposals,
    size_t *proposal_count,
    size_t proposal_capacity,
    ARCHypothesisSelectionStats *stats,
    size_t template_index,
    const ARCProgramTaskFeatures *task_features,
    uint8_t background,
    const ARCObject *subject,
    uint8_t subject_selector_kind,
    const ARCObject *anchor,
    uint8_t anchor_selector_kind,
    const ARCObject *output_object,
    int axis,
    int offset)
{
    if (binding == NULL || binding->template_fn == NULL ||
        binding->to_candidate == NULL || proposals == NULL ||
        proposal_count == NULL || proposal_capacity == 0 ||
        task == NULL || task_features == NULL ||
        subject == NULL || anchor == NULL || output_object == NULL) {
        return;
    }
    NNHypothesis hypothesis;
    ARCTransformCandidate candidate;
    if (!nn_hypothesis_init(&hypothesis, binding->template_fn())) {
        return;
    }
    int ok = nn_hypothesis_bind(&hypothesis,
                                "object_scene.background",
                                (double)background,
                                background);
    ok = ok && nn_hypothesis_bind(&hypothesis,
                                  "subject.selector_kind",
                                  (double)subject_selector_kind,
                                  subject_selector_kind);
    ok = ok && nn_hypothesis_bind(&hypothesis,
                                  "subject.color",
                                  (double)subject->color,
                                  subject->color);
    ok = ok && nn_hypothesis_bind(&hypothesis,
                                  "anchor.selector_kind",
                                  (double)anchor_selector_kind,
                                  anchor_selector_kind);
    ok = ok && nn_hypothesis_bind(&hypothesis,
                                  "anchor.color",
                                  (double)anchor->color,
                                  anchor->color);
    ok = ok && nn_hypothesis_bind(&hypothesis,
                                  "align.axis",
                                  (double)axis,
                                  axis);
    ok = ok && nn_hypothesis_bind(&hypothesis,
                                  "align.offset",
                                  (double)offset,
                                  offset);
    ok = ok && nn_hypothesis_bind(&hypothesis,
                                  "render.output_color",
                                  (double)output_object->color,
                                  output_object->color);
    if (!ok || !binding->to_candidate(&hypothesis, &candidate)) {
        return;
    }
    *proposal_count =
        arc_try_add_hypothesis_proposal(learned,
                                        task,
                                        existing_candidates,
                                        existing_count,
                                        proposals,
                                        proposal_capacity,
                                        *proposal_count,
                                        stats,
                                        template_index,
                                        task_features,
                                        &hypothesis,
                                        &candidate,
                                        subject,
                                        output_object);
}

static size_t arc_induce_move_to_anchor_hypotheses(const ARCHypothesisTemplateBinding *binding,
                                                   const ARCTask *task,
                                                   ARCProgramLearnedState *learned,
                                                   const ARCTransformCandidate *existing_candidates,
                                                   size_t existing_count,
                                                   ARCHypothesisProposal *proposals,
                                                   size_t proposal_capacity,
                                                   size_t proposal_count,
                                                   ARCHypothesisSelectionStats *stats,
                                                   size_t template_index)
{
    if (binding == NULL || binding->template_fn == NULL ||
        binding->to_candidate == NULL ||
        task == NULL || task->train_count == 0 ||
        existing_candidates == NULL || proposals == NULL ||
        proposal_capacity == 0) {
        return proposal_count;
    }

    ARCProgramTaskFeatures task_features;
    arc_program_task_features(task, &task_features);
    uint8_t backgrounds[2];
    size_t background_count =
        arc_task_relation_backgrounds(task,
                                      backgrounds,
                                      sizeof(backgrounds) / sizeof(backgrounds[0]));
    for (size_t background_index = 0;
         background_index < background_count;
         ++background_index) {
        uint8_t background = backgrounds[background_index];
        for (size_t train_pair = 0; train_pair < task->train_count; ++train_pair) {
            ARCScene input_scene;
            ARCScene output_scene;
            if (!task->train[train_pair].has_output ||
                !arc_extract_scene(&task->train[train_pair].input,
                                   background,
                                   &input_scene) ||
                !arc_extract_scene(&task->train[train_pair].output,
                                   background,
                                   &output_scene)) {
                continue;
            }
            for (size_t subject_index = 0;
                 subject_index < input_scene.object_count;
                 ++subject_index) {
                const ARCObject *subject = &input_scene.objects[subject_index];
                uint8_t subject_selector_kind = ARC_OBJECT_SELECTOR_UNIQUE_COLOR;
                if (subject->color == background ||
                    !arc_infer_non_reference_selector_for_scene_object(
                        &input_scene,
                        subject,
                        &subject_selector_kind)) {
                    continue;
                }
                for (size_t anchor_index = 0;
                     anchor_index < input_scene.object_count;
                     ++anchor_index) {
                    const ARCObject *anchor = &input_scene.objects[anchor_index];
                    uint8_t anchor_selector_kind = ARC_OBJECT_SELECTOR_UNIQUE_COLOR;
                    if (anchor->id == subject->id ||
                        anchor->color == background ||
                        anchor->color == subject->color ||
                        !arc_infer_non_reference_selector_for_scene_object(
                            &input_scene,
                            anchor,
                            &anchor_selector_kind)) {
                        continue;
                    }
                    for (size_t output_index = 0;
                         output_index < output_scene.object_count;
                         ++output_index) {
                        const ARCObject *output_object =
                            &output_scene.objects[output_index];
                        if (output_object->color == background ||
                            !arc_objects_same_shape(subject, output_object)) {
                            continue;
                        }
                        int offset_x = (int)output_object->min_x -
                                       (int)anchor->min_x;
                        int offset_y = (int)output_object->min_y -
                                       (int)anchor->min_y;
                        if (output_object->min_y == subject->min_y) {
                            arc_consider_move_to_anchor_proposal(binding,
                                                                 learned,
                                                                 task,
                                                                 existing_candidates,
                                                                 existing_count,
                                                                 proposals,
                                                                 &proposal_count,
                                                                 proposal_capacity,
                                                                 stats,
                                                                 template_index,
                                                                 &task_features,
                                                                 background,
                                                                 subject,
                                                                 subject_selector_kind,
                                                                 anchor,
                                                                 anchor_selector_kind,
                                                                 output_object,
                                                                 0,
                                                                 offset_x);
                        }
                        if (output_object->min_x == subject->min_x) {
                            arc_consider_move_to_anchor_proposal(binding,
                                                                 learned,
                                                                 task,
                                                                 existing_candidates,
                                                                 existing_count,
                                                                 proposals,
                                                                 &proposal_count,
                                                                 proposal_capacity,
                                                                 stats,
                                                                 template_index,
                                                                 &task_features,
                                                                 background,
                                                                 subject,
                                                                 subject_selector_kind,
                                                                 anchor,
                                                                 anchor_selector_kind,
                                                                 output_object,
                                                                 1,
                                                                 offset_y);
                        }
                        if (offset_x == offset_y) {
                            arc_consider_move_to_anchor_proposal(binding,
                                                                 learned,
                                                                 task,
                                                                 existing_candidates,
                                                                 existing_count,
                                                                 proposals,
                                                                 &proposal_count,
                                                                 proposal_capacity,
                                                                 stats,
                                                                 template_index,
                                                                 &task_features,
                                                                 background,
                                                                 subject,
                                                                 subject_selector_kind,
                                                                 anchor,
                                                                 anchor_selector_kind,
                                                                 output_object,
                                                                 2,
                                                                 offset_x);
                        }
                    }
                }
            }
        }
    }

    return proposal_count;
}

static size_t arc_induce_recolor_hypotheses(const ARCHypothesisTemplateBinding *binding,
                                            const ARCTask *task,
                                            ARCProgramLearnedState *learned,
                                            const ARCTransformCandidate *existing_candidates,
                                            size_t existing_count,
                                            ARCHypothesisProposal *proposals,
                                            size_t proposal_capacity,
                                            size_t proposal_count,
                                            ARCHypothesisSelectionStats *stats,
                                            size_t template_index)
{
    if (binding == NULL || binding->template_fn == NULL ||
        binding->to_candidate == NULL ||
        task == NULL || task->train_count == 0 ||
        existing_candidates == NULL || proposals == NULL ||
        proposal_capacity == 0) {
        return proposal_count;
    }

    ARCProgramTaskFeatures task_features;
    arc_program_task_features(task, &task_features);
    uint8_t backgrounds[2];
    size_t background_count =
        arc_task_relation_backgrounds(task,
                                      backgrounds,
                                      sizeof(backgrounds) / sizeof(backgrounds[0]));
    for (size_t background_index = 0;
         background_index < background_count;
         ++background_index) {
        uint8_t background = backgrounds[background_index];
        for (size_t train_pair = 0; train_pair < task->train_count; ++train_pair) {
            ARCScene input_scene;
            ARCScene output_scene;
            if (!task->train[train_pair].has_output ||
                !arc_grid_same_shape(&task->train[train_pair].input,
                                     &task->train[train_pair].output) ||
                !arc_extract_scene(&task->train[train_pair].input,
                                   background,
                                   &input_scene) ||
                !arc_extract_scene(&task->train[train_pair].output,
                                   background,
                                   &output_scene)) {
                continue;
            }
            for (size_t input_index = 0;
                 input_index < input_scene.object_count;
                 ++input_index) {
                const ARCObject *input_object = &input_scene.objects[input_index];
                uint8_t selector_kind = ARC_OBJECT_SELECTOR_UNIQUE_COLOR;
                uint8_t reference_color = 0;
                if (input_object->color == background ||
                    !arc_infer_selector_for_scene_object(&input_scene,
                                                         input_object,
                                                         &selector_kind,
                                                         &reference_color)) {
                    continue;
                }
                int reference_color_used =
                    selector_kind == ARC_OBJECT_SELECTOR_NEAREST_TO_COLOR;
                for (size_t output_index = 0;
                     output_index < output_scene.object_count;
                     ++output_index) {
                    const ARCObject *output_object = &output_scene.objects[output_index];
                    if (output_object->color == background ||
                        output_object->color == input_object->color ||
                        output_object->min_x != input_object->min_x ||
                        output_object->min_y != input_object->min_y ||
                        !arc_objects_same_shape(input_object, output_object)) {
                        continue;
                    }
                    for (uint8_t render_mode = ARC_OBJECT_RENDER_OVERLAY;
                         render_mode < ARC_OBJECT_RENDER_KIND_COUNT;
                         ++render_mode) {
                        NNHypothesis hypothesis;
                        ARCTransformCandidate candidate;
                        if (!nn_hypothesis_init(&hypothesis, binding->template_fn())) {
                            continue;
                        }
                        int ok = nn_hypothesis_bind(&hypothesis,
                                                    "object_scene.background",
                                                    (double)background,
                                                    background);
                        ok = ok && nn_hypothesis_bind(&hypothesis,
                                                      "select.selector_kind",
                                                      (double)selector_kind,
                                                      selector_kind);
                        ok = ok && nn_hypothesis_bind(&hypothesis,
                                                      "select.color",
                                                      (double)input_object->color,
                                                      input_object->color);
                        ok = ok && nn_hypothesis_bind(&hypothesis,
                                                      "select.ref_used",
                                                      (double)reference_color_used,
                                                      reference_color_used);
                        ok = ok && nn_hypothesis_bind(&hypothesis,
                                                      "select.ref_color",
                                                      (double)reference_color,
                                                      reference_color);
                        ok = ok && nn_hypothesis_bind(&hypothesis,
                                                      "render.output_color",
                                                      (double)output_object->color,
                                                      output_object->color);
                        ok = ok && nn_hypothesis_bind(&hypothesis,
                                                      "render.mode",
                                                      (double)render_mode,
                                                      render_mode);
                        if (!ok || !binding->to_candidate(&hypothesis, &candidate)) {
                            continue;
                        }
                        proposal_count =
                            arc_try_add_hypothesis_proposal(learned,
                                                            task,
                                                            existing_candidates,
                                                            existing_count,
                                                            proposals,
                                                            proposal_capacity,
                                                            proposal_count,
                                                            stats,
                                                            template_index,
                                                            &task_features,
                                                            &hypothesis,
                                                            &candidate,
                                                            input_object,
                                                            output_object);
                    }
                }
            }
        }
    }
    return proposal_count;
}

static size_t arc_induce_connect_hypotheses(const ARCHypothesisTemplateBinding *binding,
                                            const ARCTask *task,
                                            ARCProgramLearnedState *learned,
                                            const ARCTransformCandidate *existing_candidates,
                                            size_t existing_count,
                                            ARCHypothesisProposal *proposals,
                                            size_t proposal_capacity,
                                            size_t proposal_count,
                                            ARCHypothesisSelectionStats *stats,
                                            size_t template_index)
{
    if (binding == NULL || binding->template_fn == NULL ||
        binding->to_candidate == NULL ||
        task == NULL || task->train_count == 0 ||
        existing_candidates == NULL || proposals == NULL ||
        proposal_capacity == 0) {
        return proposal_count;
    }

    ARCProgramTaskFeatures task_features;
    arc_program_task_features(task, &task_features);
    uint8_t backgrounds[2];
    size_t background_count =
        arc_task_relation_backgrounds(task,
                                      backgrounds,
                                      sizeof(backgrounds) / sizeof(backgrounds[0]));
    for (size_t background_index = 0;
         background_index < background_count;
         ++background_index) {
        uint8_t background = backgrounds[background_index];
        for (size_t train_pair = 0; train_pair < task->train_count; ++train_pair) {
            if (!task->train[train_pair].has_output ||
                !arc_grid_same_shape(&task->train[train_pair].input,
                                     &task->train[train_pair].output) ||
                !arc_grid_has_changed_cells(&task->train[train_pair].input,
                                            &task->train[train_pair].output)) {
                continue;
            }
            uint8_t input_colors[ARC_COLOR_COUNT];
            uint8_t output_colors[ARC_COLOR_COUNT];
            size_t input_color_count =
                arc_grid_nonzero_colors(&task->train[train_pair].input,
                                        input_colors,
                                        ARC_COLOR_COUNT);
            size_t output_color_count =
                arc_grid_changed_output_colors(&task->train[train_pair].input,
                                               &task->train[train_pair].output,
                                               output_colors,
                                               ARC_COLOR_COUNT);
            ARCScene scene;
            if (!arc_extract_scene(&task->train[train_pair].input,
                                   background,
                                   &scene)) {
                continue;
            }
            for (size_t input_index = 0; input_index < input_color_count; ++input_index) {
                uint8_t input_color = input_colors[input_index];
                if (input_color == background) {
                    continue;
                }
                for (int mode = ARC_OBJECT_CONNECT_HORIZONTAL;
                     mode <= ARC_OBJECT_CONNECT_BOTH;
                     ++mode) {
                    if (!arc_scene_has_aligned_color_pair(&scene, input_color, mode)) {
                        continue;
                    }
                    for (size_t output_index = 0;
                         output_index < output_color_count;
                         ++output_index) {
                        uint8_t output_color = output_colors[output_index];
                        for (uint8_t render_mode = ARC_OBJECT_RENDER_OVERLAY;
                             render_mode < ARC_OBJECT_RENDER_KIND_COUNT;
                             ++render_mode) {
                            NNHypothesis hypothesis;
                            ARCTransformCandidate candidate;
                            if (!nn_hypothesis_init(&hypothesis, binding->template_fn())) {
                                continue;
                            }
                            int ok = nn_hypothesis_bind(&hypothesis,
                                                        "object_scene.background",
                                                        (double)background,
                                                        background);
                            ok = ok && nn_hypothesis_bind(&hypothesis,
                                                          "connect.color",
                                                          (double)input_color,
                                                          input_color);
                            ok = ok && nn_hypothesis_bind(&hypothesis,
                                                          "connect.output_color",
                                                          (double)output_color,
                                                          output_color);
                            ok = ok && nn_hypothesis_bind(&hypothesis,
                                                          "connect.mode",
                                                          (double)mode,
                                                          mode);
                            ok = ok && nn_hypothesis_bind(&hypothesis,
                                                          "connect.render_mode",
                                                          (double)render_mode,
                                                          render_mode);
                            if (!ok || !binding->to_candidate(&hypothesis, &candidate)) {
                                continue;
                            }
                            proposal_count =
                                arc_try_add_hypothesis_proposal(learned,
                                                                task,
                                                                existing_candidates,
                                                                existing_count,
                                                                proposals,
                                                                proposal_capacity,
                                                                proposal_count,
                                                                stats,
                                                                template_index,
                                                                &task_features,
                                                                &hypothesis,
                                                                &candidate,
                                                                NULL,
                                                                NULL);
                        }
                    }
                }
            }
        }
    }
    return proposal_count;
}

static size_t arc_induce_fill_hypotheses(const ARCHypothesisTemplateBinding *binding,
                                         const ARCTask *task,
                                         ARCProgramLearnedState *learned,
                                         const ARCTransformCandidate *existing_candidates,
                                         size_t existing_count,
                                         ARCHypothesisProposal *proposals,
                                         size_t proposal_capacity,
                                         size_t proposal_count,
                                         ARCHypothesisSelectionStats *stats,
                                         size_t template_index)
{
    if (binding == NULL || binding->template_fn == NULL ||
        binding->to_candidate == NULL ||
        task == NULL || task->train_count == 0 ||
        existing_candidates == NULL || proposals == NULL ||
        proposal_capacity == 0) {
        return proposal_count;
    }

    ARCProgramTaskFeatures task_features;
    arc_program_task_features(task, &task_features);
    uint8_t backgrounds[2];
    size_t background_count =
        arc_task_relation_backgrounds(task,
                                      backgrounds,
                                      sizeof(backgrounds) / sizeof(backgrounds[0]));
    for (size_t background_index = 0;
         background_index < background_count;
         ++background_index) {
        uint8_t background = backgrounds[background_index];
        for (size_t train_pair = 0; train_pair < task->train_count; ++train_pair) {
            if (!task->train[train_pair].has_output ||
                !arc_grid_same_shape(&task->train[train_pair].input,
                                     &task->train[train_pair].output) ||
                !arc_grid_has_changed_cells(&task->train[train_pair].input,
                                            &task->train[train_pair].output)) {
                continue;
            }
            uint8_t border_colors[ARC_COLOR_COUNT];
            uint8_t output_colors[ARC_COLOR_COUNT];
            size_t border_color_count =
                arc_grid_nonzero_colors(&task->train[train_pair].input,
                                        border_colors,
                                        ARC_COLOR_COUNT);
            size_t output_color_count =
                arc_grid_changed_output_colors(&task->train[train_pair].input,
                                               &task->train[train_pair].output,
                                               output_colors,
                                               ARC_COLOR_COUNT);
            for (size_t border_index = 0;
                 border_index < border_color_count;
                 ++border_index) {
                uint8_t border_color = border_colors[border_index];
                if (border_color == background) {
                    continue;
                }
                for (size_t output_index = 0;
                     output_index < output_color_count;
                     ++output_index) {
                    uint8_t output_color = output_colors[output_index];
                    if (output_color == border_color) {
                        continue;
                    }
                    NNHypothesis hypothesis;
                    ARCTransformCandidate candidate;
                    if (!nn_hypothesis_init(&hypothesis, binding->template_fn())) {
                        continue;
                    }
                    int ok = nn_hypothesis_bind(&hypothesis,
                                                "object_scene.background",
                                                (double)background,
                                                background);
                    ok = ok && nn_hypothesis_bind(&hypothesis,
                                                  "fill.border_color",
                                                  (double)border_color,
                                                  border_color);
                    ok = ok && nn_hypothesis_bind(&hypothesis,
                                                  "fill.output_color",
                                                  (double)output_color,
                                                  output_color);
                    if (!ok || !binding->to_candidate(&hypothesis, &candidate)) {
                        continue;
                    }
                    proposal_count =
                        arc_try_add_hypothesis_proposal(learned,
                                                        task,
                                                        existing_candidates,
                                                        existing_count,
                                                        proposals,
                                                        proposal_capacity,
                                                        proposal_count,
                                                        stats,
                                                        template_index,
                                                        &task_features,
                                                        &hypothesis,
                                                        &candidate,
                                                        NULL,
                                                        NULL);
                }
            }
        }
    }
    return proposal_count;
}

static size_t arc_induce_extract_hypotheses(const ARCHypothesisTemplateBinding *binding,
                                            const ARCTask *task,
                                            ARCProgramLearnedState *learned,
                                            const ARCTransformCandidate *existing_candidates,
                                            size_t existing_count,
                                            ARCHypothesisProposal *proposals,
                                            size_t proposal_capacity,
                                            size_t proposal_count,
                                            ARCHypothesisSelectionStats *stats,
                                            size_t template_index)
{
    if (binding == NULL || binding->template_fn == NULL ||
        binding->to_candidate == NULL ||
        task == NULL || task->train_count == 0 ||
        existing_candidates == NULL || proposals == NULL ||
        proposal_capacity == 0) {
        return proposal_count;
    }

    ARCProgramTaskFeatures task_features;
    arc_program_task_features(task, &task_features);
    uint8_t backgrounds[2];
    size_t background_count =
        arc_task_relation_backgrounds(task,
                                      backgrounds,
                                      sizeof(backgrounds) / sizeof(backgrounds[0]));
    for (size_t background_index = 0;
         background_index < background_count;
         ++background_index) {
        uint8_t background = backgrounds[background_index];
        for (size_t train_pair = 0; train_pair < task->train_count; ++train_pair) {
            ARCScene input_scene;
            if (!task->train[train_pair].has_output ||
                !arc_extract_scene(&task->train[train_pair].input,
                                   background,
                                   &input_scene)) {
                continue;
            }
            uint8_t output_colors[ARC_COLOR_COUNT];
            size_t output_color_count =
                arc_grid_nonzero_colors(&task->train[train_pair].output,
                                        output_colors,
                                        ARC_COLOR_COUNT);
            for (size_t input_index = 0;
                 input_index < input_scene.object_count;
                 ++input_index) {
                const ARCObject *input_object = &input_scene.objects[input_index];
                uint8_t selector_kind = ARC_OBJECT_SELECTOR_UNIQUE_COLOR;
                uint8_t reference_color = 0;
                if (input_object->color == background ||
                    !arc_infer_selector_for_scene_object(&input_scene,
                                                         input_object,
                                                         &selector_kind,
                                                         &reference_color)) {
                    continue;
                }
                int reference_color_used =
                    selector_kind == ARC_OBJECT_SELECTOR_NEAREST_TO_COLOR;
                for (int mode = 0; mode <= 2; ++mode) {
                    if (mode == 0) {
                        NNHypothesis hypothesis;
                        ARCTransformCandidate candidate;
                        if (!nn_hypothesis_init(&hypothesis, binding->template_fn())) {
                            continue;
                        }
                        int ok = nn_hypothesis_bind(&hypothesis,
                                                    "object_scene.background",
                                                    (double)background,
                                                    background);
                        ok = ok && nn_hypothesis_bind(&hypothesis,
                                                      "select.selector_kind",
                                                      (double)selector_kind,
                                                      selector_kind);
                        ok = ok && nn_hypothesis_bind(&hypothesis,
                                                      "select.color",
                                                      (double)input_object->color,
                                                      input_object->color);
                        ok = ok && nn_hypothesis_bind(&hypothesis,
                                                      "select.ref_used",
                                                      (double)reference_color_used,
                                                      reference_color_used);
                        ok = ok && nn_hypothesis_bind(&hypothesis,
                                                      "select.ref_color",
                                                      (double)reference_color,
                                                      reference_color);
                        ok = ok && nn_hypothesis_bind(&hypothesis,
                                                      "extract.mode",
                                                      (double)mode,
                                                      mode);
                        ok = ok && nn_hypothesis_bind(&hypothesis,
                                                      "extract.output_color",
                                                      (double)input_object->color,
                                                      input_object->color);
                        if (!ok || !binding->to_candidate(&hypothesis, &candidate)) {
                            continue;
                        }
                        proposal_count =
                            arc_try_add_hypothesis_proposal(learned,
                                                            task,
                                                            existing_candidates,
                                                            existing_count,
                                                            proposals,
                                                            proposal_capacity,
                                                            proposal_count,
                                                            stats,
                                                            template_index,
                                                            &task_features,
                                                            &hypothesis,
                                                            &candidate,
                                                            input_object,
                                                            input_object);
                        continue;
                    }
                    for (size_t output_index = 0;
                         output_index < output_color_count;
                         ++output_index) {
                        uint8_t output_color = output_colors[output_index];
                        NNHypothesis hypothesis;
                        ARCTransformCandidate candidate;
                        if (!nn_hypothesis_init(&hypothesis, binding->template_fn())) {
                            continue;
                        }
                        int ok = nn_hypothesis_bind(&hypothesis,
                                                    "object_scene.background",
                                                    (double)background,
                                                    background);
                        ok = ok && nn_hypothesis_bind(&hypothesis,
                                                      "select.selector_kind",
                                                      (double)selector_kind,
                                                      selector_kind);
                        ok = ok && nn_hypothesis_bind(&hypothesis,
                                                      "select.color",
                                                      (double)input_object->color,
                                                      input_object->color);
                        ok = ok && nn_hypothesis_bind(&hypothesis,
                                                      "select.ref_used",
                                                      (double)reference_color_used,
                                                      reference_color_used);
                        ok = ok && nn_hypothesis_bind(&hypothesis,
                                                      "select.ref_color",
                                                      (double)reference_color,
                                                      reference_color);
                        ok = ok && nn_hypothesis_bind(&hypothesis,
                                                      "extract.mode",
                                                      (double)mode,
                                                      mode);
                        ok = ok && nn_hypothesis_bind(&hypothesis,
                                                      "extract.output_color",
                                                      (double)output_color,
                                                      output_color);
                        if (!ok || !binding->to_candidate(&hypothesis, &candidate)) {
                            continue;
                        }
                        proposal_count =
                            arc_try_add_hypothesis_proposal(learned,
                                                            task,
                                                            existing_candidates,
                                                            existing_count,
                                                            proposals,
                                                            proposal_capacity,
                                                            proposal_count,
                                                            stats,
                                                            template_index,
                                                            &task_features,
                                                            &hypothesis,
                                                            &candidate,
                                                            input_object,
                                                            input_object);
                    }
                }
            }
        }
    }
    return proposal_count;
}

static size_t arc_induce_sequence_hypotheses(const ARCHypothesisTemplateBinding *binding,
                                             const ARCTask *task,
                                             ARCProgramLearnedState *learned,
                                             const ARCTransformCandidate *existing_candidates,
                                             size_t existing_count,
                                             ARCHypothesisProposal *proposals,
                                             size_t proposal_capacity,
                                             size_t proposal_count,
                                             ARCHypothesisSelectionStats *stats,
                                             size_t template_index)
{
    if (binding == NULL || binding->template_fn == NULL ||
        binding->to_candidate == NULL ||
        task == NULL || task->train_count == 0 ||
        existing_candidates == NULL || proposals == NULL ||
        proposal_capacity == 0) {
        return proposal_count;
    }

    ARCProgramTaskFeatures task_features;
    arc_program_task_features(task, &task_features);
    uint8_t backgrounds[2];
    size_t background_count =
        arc_task_relation_backgrounds(task,
                                      backgrounds,
                                      sizeof(backgrounds) / sizeof(backgrounds[0]));
    for (size_t background_index = 0;
         background_index < background_count;
         ++background_index) {
        uint8_t background = backgrounds[background_index];
        for (size_t train_pair = 0; train_pair < task->train_count; ++train_pair) {
            const ARCGrid *input = &task->train[train_pair].input;
            const ARCGrid *output = &task->train[train_pair].output;
            if (!task->train[train_pair].has_output ||
                !arc_grid_same_shape(input, output) ||
                !arc_grid_has_changed_cells(input, output)) {
                continue;
            }
            uint8_t input_colors[ARC_COLOR_COUNT];
            uint8_t output_colors[ARC_COLOR_COUNT];
            size_t input_color_count =
                arc_grid_nonzero_colors(input, input_colors, ARC_COLOR_COUNT);
            size_t output_color_count =
                arc_grid_changed_output_colors(input,
                                               output,
                                               output_colors,
                                               ARC_COLOR_COUNT);
            for (size_t input_index = 0;
                 input_index < input_color_count;
                 ++input_index) {
                uint8_t input_color = input_colors[input_index];
                if (input_color == background ||
                    arc_grid_color_cell_count(input, input_color) < 2u) {
                    continue;
                }
                for (size_t output_index = 0;
                     output_index < output_color_count;
                     ++output_index) {
                    uint8_t output_color = output_colors[output_index];
                    for (int direction = 0;
                         direction < ARC_SEQUENCE_DIRECTION_KIND_COUNT;
                         ++direction) {
                        for (int extend_mode = 0;
                             extend_mode < ARC_SEQUENCE_EXTEND_KIND_COUNT;
                             ++extend_mode) {
                            NNHypothesis hypothesis;
                            ARCTransformCandidate candidate;
                            if (!nn_hypothesis_init(&hypothesis,
                                                    binding->template_fn())) {
                                continue;
                            }
                            int ok =
                                nn_hypothesis_bind(&hypothesis,
                                                   "object_scene.background",
                                                   (double)background,
                                                   background);
                            ok = ok && nn_hypothesis_bind(&hypothesis,
                                                          "sequence.color",
                                                          (double)input_color,
                                                          input_color);
                            ok = ok && nn_hypothesis_bind(&hypothesis,
                                                          "sequence.output_color",
                                                          (double)output_color,
                                                          output_color);
                            ok = ok && nn_hypothesis_bind(&hypothesis,
                                                          "sequence.direction",
                                                          (double)direction,
                                                          direction);
                            ok = ok && nn_hypothesis_bind(&hypothesis,
                                                          "sequence.extend_mode",
                                                          (double)extend_mode,
                                                          extend_mode);
                            if (!ok ||
                                !binding->to_candidate(&hypothesis, &candidate)) {
                                continue;
                            }
                            static ARCGrid produced;
                            if (!arc_apply_candidate(&candidate, input, &produced) ||
                                !arc_grid_equal(&produced, output)) {
                                continue;
                            }
                            proposal_count =
                                arc_try_add_hypothesis_proposal(learned,
                                                                task,
                                                                existing_candidates,
                                                                existing_count,
                                                                proposals,
                                                                proposal_capacity,
                                                                proposal_count,
                                                                stats,
                                                                template_index,
                                                                &task_features,
                                                                &hypothesis,
                                                                &candidate,
                                                                NULL,
                                                                NULL);
                        }
                    }
                }
            }
        }
    }
    return proposal_count;
}

static size_t arc_induce_tile_motif_hypotheses(const ARCHypothesisTemplateBinding *binding,
                                               const ARCTask *task,
                                               ARCProgramLearnedState *learned,
                                               const ARCTransformCandidate *existing_candidates,
                                               size_t existing_count,
                                               ARCHypothesisProposal *proposals,
                                               size_t proposal_capacity,
                                               size_t proposal_count,
                                               ARCHypothesisSelectionStats *stats,
                                               size_t template_index)
{
    if (binding == NULL || binding->template_fn == NULL ||
        binding->to_candidate == NULL ||
        task == NULL || task->train_count == 0 ||
        existing_candidates == NULL || proposals == NULL ||
        proposal_capacity == 0) {
        return proposal_count;
    }

    ARCProgramTaskFeatures task_features;
    arc_program_task_features(task, &task_features);
    uint8_t backgrounds[2];
    size_t background_count =
        arc_task_relation_backgrounds(task,
                                      backgrounds,
                                      sizeof(backgrounds) / sizeof(backgrounds[0]));
    for (size_t background_index = 0;
         background_index < background_count;
         ++background_index) {
        uint8_t background = backgrounds[background_index];
        for (size_t period = 2; period <= 3; ++period) {
            NNHypothesis hypothesis;
            ARCTransformCandidate candidate;
            if (!nn_hypothesis_init(&hypothesis, binding->template_fn())) {
                continue;
            }
            int ok = nn_hypothesis_bind(&hypothesis,
                                        "lattice.background",
                                        (double)background,
                                        background);
            ok = ok && nn_hypothesis_bind(&hypothesis,
                                          "repeat.period",
                                          (double)period,
                                          (int64_t)period);
            if (!ok || !binding->to_candidate(&hypothesis, &candidate)) {
                continue;
            }

            int has_pair_evidence = 0;
            for (size_t train_pair = 0;
                 train_pair < task->train_count;
                 ++train_pair) {
                static ARCGrid produced;
                if (!task->train[train_pair].has_output ||
                    !arc_grid_same_shape(&task->train[train_pair].input,
                                         &task->train[train_pair].output) ||
                    !arc_apply_candidate(&candidate,
                                         &task->train[train_pair].input,
                                         &produced) ||
                    !arc_grid_equal(&produced,
                                    &task->train[train_pair].output)) {
                    continue;
                }
                has_pair_evidence = 1;
                break;
            }
            if (!has_pair_evidence) {
                continue;
            }

            proposal_count =
                arc_try_add_hypothesis_proposal(learned,
                                                task,
                                                existing_candidates,
                                                existing_count,
                                                proposals,
                                                proposal_capacity,
                                                proposal_count,
                                                stats,
                                                template_index,
                                                &task_features,
                                                &hypothesis,
                                                &candidate,
                                                NULL,
                                                NULL);
        }
    }
    return proposal_count;
}

static size_t arc_induce_masked_motif_hypotheses(const ARCHypothesisTemplateBinding *binding,
                                                 const ARCTask *task,
                                                 ARCProgramLearnedState *learned,
                                                 const ARCTransformCandidate *existing_candidates,
                                                 size_t existing_count,
                                                 ARCHypothesisProposal *proposals,
                                                 size_t proposal_capacity,
                                                 size_t proposal_count,
                                                 ARCHypothesisSelectionStats *stats,
                                                 size_t template_index)
{
    if (binding == NULL || binding->template_fn == NULL ||
        binding->to_candidate == NULL ||
        task == NULL || task->train_count == 0 ||
        existing_candidates == NULL || proposals == NULL ||
        proposal_capacity == 0) {
        return proposal_count;
    }

    ARCProgramTaskFeatures task_features;
    arc_program_task_features(task, &task_features);
    uint8_t backgrounds[2];
    size_t background_count =
        arc_task_relation_backgrounds(task,
                                      backgrounds,
                                      sizeof(backgrounds) / sizeof(backgrounds[0]));
    for (size_t background_index = 0;
         background_index < background_count;
         ++background_index) {
        uint8_t background = backgrounds[background_index];
        NNHypothesis hypothesis;
        ARCTransformCandidate candidate;
        if (!nn_hypothesis_init(&hypothesis, binding->template_fn())) {
            continue;
        }
        int ok = nn_hypothesis_bind(&hypothesis,
                                    "mask.background",
                                    (double)background,
                                    background);
        if (!ok || !binding->to_candidate(&hypothesis, &candidate)) {
            continue;
        }

        int has_pair_evidence = 0;
        for (size_t train_pair = 0; train_pair < task->train_count; ++train_pair) {
            const ARCGridPair *pair = &task->train[train_pair];
            static ARCGrid produced;
            if (!pair->has_output ||
                pair->input.width * pair->input.width != pair->output.width ||
                pair->input.height * pair->input.height != pair->output.height ||
                !arc_apply_candidate(&candidate, &pair->input, &produced) ||
                !arc_grid_equal(&produced, &pair->output)) {
                continue;
            }
            has_pair_evidence = 1;
            break;
        }
        if (!has_pair_evidence) {
            continue;
        }

        proposal_count =
            arc_try_add_hypothesis_proposal(learned,
                                            task,
                                            existing_candidates,
                                            existing_count,
                                            proposals,
                                            proposal_capacity,
                                            proposal_count,
                                            stats,
                                            template_index,
                                            &task_features,
                                            &hypothesis,
                                            &candidate,
                                            NULL,
                                            NULL);
    }
    return proposal_count;
}

static const char *arc_panel_anchor_short_name(int mode)
{
    switch (mode) {
    case 0:
        return "start";
    case 1:
        return "center";
    case 2:
        return "end";
    }
    return "unknown";
}

static int arc_panel_origin(size_t input_extent,
                            size_t panel_extent,
                            int anchor_mode,
                            int offset,
                            size_t *origin_out)
{
    if (panel_extent == 0 || panel_extent > input_extent ||
        anchor_mode < 0 || anchor_mode > 2 || origin_out == NULL) {
        return 0;
    }
    size_t span = input_extent - panel_extent;
    int64_t base = 0;
    if (anchor_mode == 1) {
        base = (int64_t)(span / 2u);
    } else if (anchor_mode == 2) {
        base = (int64_t)span;
    }
    int64_t origin = base + (int64_t)offset;
    if (origin < 0 ||
        origin + (int64_t)panel_extent > (int64_t)input_extent) {
        return 0;
    }
    *origin_out = (size_t)origin;
    return 1;
}

static int arc_grid_crop_matches(const ARCGrid *input,
                                 const ARCGrid *output,
                                 size_t origin_x,
                                 size_t origin_y)
{
    if (input == NULL || output == NULL ||
        output->width == 0 || output->height == 0 ||
        origin_x + output->width > input->width ||
        origin_y + output->height > input->height) {
        return 0;
    }
    for (size_t y = 0; y < output->height; ++y) {
        for (size_t x = 0; x < output->width; ++x) {
            size_t input_cell = (origin_y + y) * input->width + origin_x + x;
            size_t output_cell = y * output->width + x;
            if (input->cells[input_cell] != output->cells[output_cell]) {
                return 0;
            }
        }
    }
    return 1;
}

static int arc_grid_non_background_bbox(const ARCGrid *grid,
                                        uint8_t background,
                                        size_t *min_x_out,
                                        size_t *min_y_out,
                                        size_t *max_x_out,
                                        size_t *max_y_out)
{
    if (grid == NULL || grid->width == 0 || grid->height == 0) {
        return 0;
    }
    int found = 0;
    size_t min_x = grid->width;
    size_t min_y = grid->height;
    size_t max_x = 0;
    size_t max_y = 0;
    for (size_t y = 0; y < grid->height; ++y) {
        for (size_t x = 0; x < grid->width; ++x) {
            uint8_t color = grid->cells[y * grid->width + x];
            if (color == background) {
                continue;
            }
            if (!found || x < min_x) {
                min_x = x;
            }
            if (!found || y < min_y) {
                min_y = y;
            }
            if (!found || x > max_x) {
                max_x = x;
            }
            if (!found || y > max_y) {
                max_y = y;
            }
            found = 1;
        }
    }
    if (!found) {
        return 0;
    }
    if (min_x_out != NULL) {
        *min_x_out = min_x;
    }
    if (min_y_out != NULL) {
        *min_y_out = min_y;
    }
    if (max_x_out != NULL) {
        *max_x_out = max_x;
    }
    if (max_y_out != NULL) {
        *max_y_out = max_y;
    }
    return 1;
}

static int arc_grid_color_bbox(const ARCGrid *grid,
                               uint8_t color,
                               size_t *min_x_out,
                               size_t *min_y_out,
                               size_t *max_x_out,
                               size_t *max_y_out)
{
    if (grid == NULL || grid->width == 0 || grid->height == 0) {
        return 0;
    }
    int found = 0;
    size_t min_x = grid->width;
    size_t min_y = grid->height;
    size_t max_x = 0;
    size_t max_y = 0;
    for (size_t y = 0; y < grid->height; ++y) {
        for (size_t x = 0; x < grid->width; ++x) {
            if (grid->cells[y * grid->width + x] != color) {
                continue;
            }
            if (!found || x < min_x) {
                min_x = x;
            }
            if (!found || y < min_y) {
                min_y = y;
            }
            if (!found || x > max_x) {
                max_x = x;
            }
            if (!found || y > max_y) {
                max_y = y;
            }
            found = 1;
        }
    }
    if (!found) {
        return 0;
    }
    if (min_x_out != NULL) {
        *min_x_out = min_x;
    }
    if (min_y_out != NULL) {
        *min_y_out = min_y;
    }
    if (max_x_out != NULL) {
        *max_x_out = max_x;
    }
    if (max_y_out != NULL) {
        *max_y_out = max_y;
    }
    return 1;
}

static int arc_bbox_panel_origin(const ARCGrid *input,
                                 uint8_t background,
                                 size_t panel_width,
                                 size_t panel_height,
                                 int x_anchor,
                                 int y_anchor,
                                 int x_offset,
                                 int y_offset,
                                 size_t *origin_x_out,
                                 size_t *origin_y_out)
{
    size_t min_x = 0;
    size_t min_y = 0;
    size_t max_x = 0;
    size_t max_y = 0;
    if (input == NULL || origin_x_out == NULL || origin_y_out == NULL ||
        panel_width == 0 || panel_height == 0 ||
        x_anchor < 0 || x_anchor > 2 ||
        y_anchor < 0 || y_anchor > 2 ||
        !arc_grid_non_background_bbox(input,
                                      background,
                                      &min_x,
                                      &min_y,
                                      &max_x,
                                      &max_y)) {
        return 0;
    }
    size_t bbox_width = max_x - min_x + 1u;
    size_t bbox_height = max_y - min_y + 1u;
    if (panel_width > bbox_width || panel_height > bbox_height) {
        return 0;
    }
    size_t x_span = bbox_width - panel_width;
    size_t y_span = bbox_height - panel_height;
    int64_t x_base = (int64_t)min_x;
    int64_t y_base = (int64_t)min_y;
    if (x_anchor == 1) {
        x_base += (int64_t)(x_span / 2u);
    } else if (x_anchor == 2) {
        x_base += (int64_t)x_span;
    }
    if (y_anchor == 1) {
        y_base += (int64_t)(y_span / 2u);
    } else if (y_anchor == 2) {
        y_base += (int64_t)y_span;
    }
    int64_t origin_x = x_base + (int64_t)x_offset;
    int64_t origin_y = y_base + (int64_t)y_offset;
    if (origin_x < 0 || origin_y < 0 ||
        origin_x + (int64_t)panel_width > (int64_t)input->width ||
        origin_y + (int64_t)panel_height > (int64_t)input->height) {
        return 0;
    }
    *origin_x_out = (size_t)origin_x;
    *origin_y_out = (size_t)origin_y;
    return 1;
}

static size_t arc_induce_panel_selection_hypotheses(const ARCHypothesisTemplateBinding *binding,
                                                    const ARCTask *task,
                                                    ARCProgramLearnedState *learned,
                                                    const ARCTransformCandidate *existing_candidates,
                                                    size_t existing_count,
                                                    ARCHypothesisProposal *proposals,
                                                    size_t proposal_capacity,
                                                    size_t proposal_count,
                                                    ARCHypothesisSelectionStats *stats,
                                                    size_t template_index)
{
    if (binding == NULL || binding->template_fn == NULL ||
        binding->to_candidate == NULL ||
        task == NULL || task->train_count == 0 ||
        existing_candidates == NULL || proposals == NULL ||
        proposal_capacity == 0 ||
        !task->train[0].has_output) {
        return proposal_count;
    }

    const ARCGrid *first_input = &task->train[0].input;
    const ARCGrid *first_output = &task->train[0].output;
    if (first_output->width == 0 || first_output->height == 0 ||
        first_output->width > first_input->width ||
        first_output->height > first_input->height ||
        (first_output->width == first_input->width &&
         first_output->height == first_input->height)) {
        return proposal_count;
    }
    for (size_t pair = 1; pair < task->train_count; ++pair) {
        const ARCGrid *input = &task->train[pair].input;
        const ARCGrid *output = &task->train[pair].output;
        if (!task->train[pair].has_output ||
            output->width != first_output->width ||
            output->height != first_output->height ||
            output->width > input->width ||
            output->height > input->height) {
            return proposal_count;
        }
    }

    ARCProgramTaskFeatures task_features;
    arc_program_task_features(task, &task_features);
    uint8_t background = arc_grid_dominant_color(first_input);
    for (size_t origin_y = 0;
         origin_y + first_output->height <= first_input->height;
         ++origin_y) {
        for (size_t origin_x = 0;
             origin_x + first_output->width <= first_input->width;
             ++origin_x) {
            if (!arc_grid_crop_matches(first_input,
                                       first_output,
                                       origin_x,
                                       origin_y)) {
                continue;
            }
            for (int x_mode = 0; x_mode <= 2; ++x_mode) {
                size_t x_span = first_input->width - first_output->width;
                int64_t x_base = x_mode == 0 ? 0
                                 : x_mode == 1 ? (int64_t)(x_span / 2u)
                                               : (int64_t)x_span;
                int x_offset = (int)((int64_t)origin_x - x_base);
                for (int y_mode = 0; y_mode <= 2; ++y_mode) {
                    size_t y_span = first_input->height - first_output->height;
                    int64_t y_base = y_mode == 0 ? 0
                                     : y_mode == 1 ? (int64_t)(y_span / 2u)
                                                   : (int64_t)y_span;
                    int y_offset = (int)((int64_t)origin_y - y_base);

                    NNHypothesis hypothesis;
                    ARCTransformCandidate candidate;
                    if (!nn_hypothesis_init(&hypothesis, binding->template_fn())) {
                        continue;
                    }
                    int ok = nn_hypothesis_bind(&hypothesis,
                                                "panel.background",
                                                (double)background,
                                                background);
                    ok = ok && nn_hypothesis_bind(&hypothesis,
                                                  "panel.x_anchor",
                                                  (double)x_mode,
                                                  x_mode);
                    ok = ok && nn_hypothesis_bind(&hypothesis,
                                                  "panel.y_anchor",
                                                  (double)y_mode,
                                                  y_mode);
                    ok = ok && nn_hypothesis_bind(&hypothesis,
                                                  "panel.x_offset",
                                                  (double)x_offset,
                                                  x_offset);
                    ok = ok && nn_hypothesis_bind(&hypothesis,
                                                  "panel.y_offset",
                                                  (double)y_offset,
                                                  y_offset);
                    ok = ok && nn_hypothesis_bind(&hypothesis,
                                                  "panel.width",
                                                  (double)first_output->width,
                                                  (int64_t)first_output->width);
                    ok = ok && nn_hypothesis_bind(&hypothesis,
                                                  "panel.height",
                                                  (double)first_output->height,
                                                  (int64_t)first_output->height);
                    if (!ok || !binding->to_candidate(&hypothesis, &candidate)) {
                        continue;
                    }
                    proposal_count =
                        arc_try_add_hypothesis_proposal(learned,
                                                        task,
                                                        existing_candidates,
                                                        existing_count,
                                                        proposals,
                                                        proposal_capacity,
                                                        proposal_count,
                                                        stats,
                                                        template_index,
                                                        &task_features,
                                                        &hypothesis,
                                                        &candidate,
                                                        NULL,
                                                        NULL);
                }
            }
        }
    }
    return proposal_count;
}

static size_t arc_induce_bbox_panel_selection_hypotheses(const ARCHypothesisTemplateBinding *binding,
                                                         const ARCTask *task,
                                                         ARCProgramLearnedState *learned,
                                                         const ARCTransformCandidate *existing_candidates,
                                                         size_t existing_count,
                                                         ARCHypothesisProposal *proposals,
                                                         size_t proposal_capacity,
                                                         size_t proposal_count,
                                                         ARCHypothesisSelectionStats *stats,
                                                         size_t template_index)
{
    if (binding == NULL || binding->template_fn == NULL ||
        binding->to_candidate == NULL ||
        task == NULL || task->train_count == 0 ||
        existing_candidates == NULL || proposals == NULL ||
        proposal_capacity == 0 ||
        !task->train[0].has_output) {
        return proposal_count;
    }

    const ARCGrid *first_input = &task->train[0].input;
    const ARCGrid *first_output = &task->train[0].output;
    if (first_output->width == 0 || first_output->height == 0 ||
        first_output->width > first_input->width ||
        first_output->height > first_input->height ||
        (first_output->width == first_input->width &&
         first_output->height == first_input->height)) {
        return proposal_count;
    }
    for (size_t pair = 1; pair < task->train_count; ++pair) {
        const ARCGrid *input = &task->train[pair].input;
        const ARCGrid *output = &task->train[pair].output;
        if (!task->train[pair].has_output ||
            output->width != first_output->width ||
            output->height != first_output->height ||
            output->width > input->width ||
            output->height > input->height) {
            return proposal_count;
        }
    }

    uint8_t background = arc_grid_dominant_color(first_input);
    size_t bbox_min_x = 0;
    size_t bbox_min_y = 0;
    size_t bbox_max_x = 0;
    size_t bbox_max_y = 0;
    if (!arc_grid_non_background_bbox(first_input,
                                      background,
                                      &bbox_min_x,
                                      &bbox_min_y,
                                      &bbox_max_x,
                                      &bbox_max_y)) {
        return proposal_count;
    }
    size_t bbox_width = bbox_max_x - bbox_min_x + 1u;
    size_t bbox_height = bbox_max_y - bbox_min_y + 1u;
    if (first_output->width > bbox_width ||
        first_output->height > bbox_height) {
        return proposal_count;
    }

    ARCProgramTaskFeatures task_features;
    arc_program_task_features(task, &task_features);
    for (size_t origin_y = 0;
         origin_y + first_output->height <= first_input->height;
         ++origin_y) {
        for (size_t origin_x = 0;
             origin_x + first_output->width <= first_input->width;
             ++origin_x) {
            if (!arc_grid_crop_matches(first_input,
                                       first_output,
                                       origin_x,
                                       origin_y)) {
                continue;
            }
            for (int x_mode = 0; x_mode <= 2; ++x_mode) {
                size_t x_span = bbox_width - first_output->width;
                int64_t x_base = (int64_t)bbox_min_x;
                if (x_mode == 1) {
                    x_base += (int64_t)(x_span / 2u);
                } else if (x_mode == 2) {
                    x_base += (int64_t)x_span;
                }
                int x_offset = (int)((int64_t)origin_x - x_base);
                for (int y_mode = 0; y_mode <= 2; ++y_mode) {
                    size_t y_span = bbox_height - first_output->height;
                    int64_t y_base = (int64_t)bbox_min_y;
                    if (y_mode == 1) {
                        y_base += (int64_t)(y_span / 2u);
                    } else if (y_mode == 2) {
                        y_base += (int64_t)y_span;
                    }
                    int y_offset = (int)((int64_t)origin_y - y_base);

                    NNHypothesis hypothesis;
                    ARCTransformCandidate candidate;
                    if (!nn_hypothesis_init(&hypothesis, binding->template_fn())) {
                        continue;
                    }
                    int ok = nn_hypothesis_bind(&hypothesis,
                                                "panel.background",
                                                (double)background,
                                                background);
                    ok = ok && nn_hypothesis_bind(&hypothesis,
                                                  "panel.x_anchor",
                                                  (double)x_mode,
                                                  x_mode);
                    ok = ok && nn_hypothesis_bind(&hypothesis,
                                                  "panel.y_anchor",
                                                  (double)y_mode,
                                                  y_mode);
                    ok = ok && nn_hypothesis_bind(&hypothesis,
                                                  "panel.x_offset",
                                                  (double)x_offset,
                                                  x_offset);
                    ok = ok && nn_hypothesis_bind(&hypothesis,
                                                  "panel.y_offset",
                                                  (double)y_offset,
                                                  y_offset);
                    ok = ok && nn_hypothesis_bind(&hypothesis,
                                                  "panel.width",
                                                  (double)first_output->width,
                                                  (int64_t)first_output->width);
                    ok = ok && nn_hypothesis_bind(&hypothesis,
                                                  "panel.height",
                                                  (double)first_output->height,
                                                  (int64_t)first_output->height);
                    if (!ok || !binding->to_candidate(&hypothesis, &candidate)) {
                        continue;
                    }
                    proposal_count =
                        arc_try_add_hypothesis_proposal(learned,
                                                        task,
                                                        existing_candidates,
                                                        existing_count,
                                                        proposals,
                                                        proposal_capacity,
                                                        proposal_count,
                                                        stats,
                                                        template_index,
                                                        &task_features,
                                                        &hypothesis,
                                                        &candidate,
                                                        NULL,
                                                        NULL);
                }
            }
        }
    }
    return proposal_count;
}
