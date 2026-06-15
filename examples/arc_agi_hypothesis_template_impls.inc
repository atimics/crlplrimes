typedef struct {
    uint8_t color_map[ARC_COLOR_COUNT];
} ARCColorMapHypothesisSpec;

static int arc_color_map_spec_from_hypothesis(const NNHypothesis *hypothesis,
                                              ARCColorMapHypothesisSpec *spec)
{
    if (hypothesis == NULL || spec == NULL ||
        hypothesis->template_ref != arc_hypothesis_template_same_grid_color_map() ||
        !nn_hypothesis_is_fully_bound(hypothesis)) {
        return 0;
    }
    for (uint8_t color = 0; color < ARC_COLOR_COUNT; ++color) {
        char slot_name[16];
        int64_t mapped = 0;
        arc_color_map_slot_name(color, slot_name, sizeof(slot_name));
        if (!nn_hypothesis_get(hypothesis, slot_name, NULL, &mapped) ||
            mapped < 0 || mapped >= ARC_COLOR_COUNT) {
            return 0;
        }
        spec->color_map[color] = (uint8_t)mapped;
    }
    return 1;
}

static void arc_color_map_candidate_from_spec(const ARCColorMapHypothesisSpec *spec,
                                              ARCTransformCandidate *candidate)
{
    arc_init_candidate(candidate, ARC_RULE_GRID_COLOR_MAP, "same_grid_color_map");
    memcpy(candidate->color_map, spec->color_map, sizeof(candidate->color_map));
}

static int arc_candidate_from_color_map_hypothesis(const NNHypothesis *hypothesis,
                                                   ARCTransformCandidate *candidate)
{
    ARCColorMapHypothesisSpec spec;
    if (candidate == NULL ||
        !arc_color_map_spec_from_hypothesis(hypothesis, &spec)) {
        return 0;
    }
    arc_color_map_candidate_from_spec(&spec, candidate);
    candidate->op = ARC_RULE_HYPOTHESIS_APPLIED;
    if (!arc_store_hypothesis_candidate(candidate, hypothesis)) {
        return 0;
    }

    char encoded[ARC_COLOR_COUNT + 1u];
    for (uint8_t color = 0; color < ARC_COLOR_COUNT; ++color) {
        encoded[color] = (char)('0' + spec.color_map[color]);
    }
    encoded[ARC_COLOR_COUNT] = '\0';
    snprintf(candidate->name,
             sizeof(candidate->name),
             "hypothesis_%s_%s",
             hypothesis->template_ref->name,
             encoded);
    return 1;
}

static int arc_apply_color_map_hypothesis(const NNHypothesis *hypothesis,
                                          const ARCGrid *input,
                                          ARCGrid *output)
{
    ARCColorMapHypothesisSpec spec;
    if (!arc_color_map_spec_from_hypothesis(hypothesis, &spec) ||
        input == NULL || output == NULL) {
        return 0;
    }
    arc_init_grid(output, input->width, input->height);
    for (size_t cell = 0; cell < arc_grid_cells(input); ++cell) {
        uint8_t color = input->cells[cell];
        if (color >= ARC_COLOR_COUNT) {
            return 0;
        }
        output->cells[cell] = spec.color_map[color];
    }
    return 1;
}

static const NNHypothesisTemplate *arc_hypothesis_template_object_translate(void)
{
    static NNHypothesisTemplate template_ref;
    static int initialized = 0;
    if (initialized) {
        return &template_ref;
    }
    nn_hypothesis_template_init(&template_ref,
                                "arc_agi",
                                "select_then_translate",
                                1024,
                                NN_PROGRAM_VALUE_GRID,
                                NN_PROGRAM_VALUE_GRID);
    nn_hypothesis_template_add_stage(&template_ref,
                                     "object_scene",
                                     NN_PROGRAM_KIND_TRANSFORM,
                                     NN_PROGRAM_VALUE_GRID,
                                     NN_PROGRAM_VALUE_OBJECT_SET,
                                     110);
    nn_hypothesis_template_add_stage(&template_ref,
                                     "select",
                                     NN_PROGRAM_KIND_SELECTOR,
                                     NN_PROGRAM_VALUE_OBJECT_SET,
                                     NN_PROGRAM_VALUE_OBJECT,
                                     111);
    nn_hypothesis_template_add_stage(&template_ref,
                                     "translate",
                                     NN_PROGRAM_KIND_TRANSFORM,
                                     NN_PROGRAM_VALUE_OBJECT,
                                     NN_PROGRAM_VALUE_OBJECT,
                                     112);
    nn_hypothesis_template_add_stage(&template_ref,
                                     "render",
                                     NN_PROGRAM_KIND_TRANSFORM,
                                     NN_PROGRAM_VALUE_OBJECT,
                                     NN_PROGRAM_VALUE_GRID,
                                     113);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "object_scene.background",
                                    NN_HYPOTHESIS_SLOT_COLOR,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "select.selector_kind",
                                    NN_HYPOTHESIS_SLOT_SELECTOR_KIND,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "select.color",
                                    NN_HYPOTHESIS_SLOT_COLOR,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "select.reference_color_used",
                                    NN_HYPOTHESIS_SLOT_BOOLEAN,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "select.reference_color",
                                    NN_HYPOTHESIS_SLOT_REFERENCE_COLOR,
                                    0, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "translate.dx",
                                    NN_HYPOTHESIS_SLOT_OFFSET,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "translate.dy",
                                    NN_HYPOTHESIS_SLOT_OFFSET,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "render.background",
                                    NN_HYPOTHESIS_SLOT_COLOR,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "render.output_color",
                                    NN_HYPOTHESIS_SLOT_COLOR,
                                    1, 0.0, 0);
    initialized = 1;
    return &template_ref;
}

typedef struct {
    uint8_t background;
    uint8_t selector_kind;
    uint8_t input_color;
    uint8_t reference_color_used;
    uint8_t reference_color;
    int dx;
    int dy;
    uint8_t output_color;
} ARCTranslateHypothesisSpec;

static int arc_translate_spec_from_hypothesis(const NNHypothesis *hypothesis,
                                              ARCTranslateHypothesisSpec *spec)
{
    if (hypothesis == NULL || spec == NULL ||
        hypothesis->template_ref != arc_hypothesis_template_object_translate() ||
        !nn_hypothesis_is_fully_bound(hypothesis)) {
        return 0;
    }
    int64_t background = 0;
    int64_t selector_kind = 0;
    int64_t input_color = 0;
    int64_t reference_color_used = 0;
    int64_t reference_color = 0;
    int64_t dx = 0;
    int64_t dy = 0;
    int64_t render_background = 0;
    int64_t output_color = 0;
    if (!nn_hypothesis_get(hypothesis,
                           "object_scene.background",
                           NULL,
                           &background) ||
        !nn_hypothesis_get(hypothesis,
                           "select.selector_kind",
                           NULL,
                           &selector_kind) ||
        !nn_hypothesis_get(hypothesis, "select.color", NULL, &input_color) ||
        !nn_hypothesis_get(hypothesis,
                           "select.reference_color_used",
                           NULL,
                           &reference_color_used) ||
        !nn_hypothesis_get(hypothesis,
                           "select.reference_color",
                           NULL,
                           &reference_color) ||
        !nn_hypothesis_get(hypothesis, "translate.dx", NULL, &dx) ||
        !nn_hypothesis_get(hypothesis, "translate.dy", NULL, &dy) ||
        !nn_hypothesis_get(hypothesis,
                           "render.background",
                           NULL,
                           &render_background) ||
        !nn_hypothesis_get(hypothesis,
                           "render.output_color",
                           NULL,
                           &output_color)) {
        return 0;
    }
    if (background < 0 || background >= ARC_COLOR_COUNT ||
        render_background != background ||
        selector_kind < 0 || selector_kind >= ARC_OBJECT_SELECTOR_KIND_COUNT ||
        input_color < 0 || input_color >= ARC_COLOR_COUNT ||
        reference_color_used < 0 || reference_color_used > 1 ||
        reference_color < 0 || reference_color >= ARC_COLOR_COUNT ||
        output_color < 0 || output_color >= ARC_COLOR_COUNT ||
        dx < -ARC_MAX_GRID_WIDTH || dx > ARC_MAX_GRID_WIDTH ||
        dy < -ARC_MAX_GRID_HEIGHT || dy > ARC_MAX_GRID_HEIGHT) {
        return 0;
    }
    int reference_used = reference_color_used != 0;
    if ((reference_used &&
         selector_kind != ARC_OBJECT_SELECTOR_NEAREST_TO_COLOR) ||
        (!reference_used &&
         selector_kind == ARC_OBJECT_SELECTOR_NEAREST_TO_COLOR) ||
        (!reference_used && reference_color != 0)) {
        return 0;
    }

    spec->background = (uint8_t)background;
    spec->selector_kind = (uint8_t)selector_kind;
    spec->input_color = (uint8_t)input_color;
    spec->reference_color_used = (uint8_t)reference_color_used;
    spec->reference_color = (uint8_t)reference_color;
    spec->dx = (int)dx;
    spec->dy = (int)dy;
    spec->output_color = (uint8_t)output_color;
    return 1;
}

static int arc_candidate_from_translate_hypothesis(const NNHypothesis *hypothesis,
                                                   ARCTransformCandidate *candidate)
{
    ARCTranslateHypothesisSpec spec;
    if (candidate == NULL ||
        !arc_translate_spec_from_hypothesis(hypothesis, &spec)) {
        return 0;
    }

    arc_init_candidate(candidate, ARC_RULE_HYPOTHESIS_APPLIED, "");
    candidate->background_color = spec.background;
    candidate->selector_kind = spec.selector_kind;
    candidate->output_selector_kind = ARC_OBJECT_SELECTOR_UNIQUE_COLOR;
    candidate->input_color = spec.input_color;
    candidate->reference_color = spec.reference_color;
    candidate->output_color = spec.output_color;
    candidate->dx = spec.dx;
    candidate->dy = spec.dy;
    if (!arc_store_hypothesis_candidate(candidate, hypothesis)) {
        return 0;
    }
    if (spec.reference_color_used) {
        snprintf(candidate->name,
                 sizeof(candidate->name),
                 "hypothesis_%s_%s_ref_%u_bg_%u_color_%u_to_%u_%+d_%+d",
                 hypothesis->template_ref->name,
                 arc_object_selector_short_name(candidate->selector_kind),
                 (unsigned)candidate->reference_color,
                 (unsigned)candidate->background_color,
                 (unsigned)candidate->input_color,
                 (unsigned)candidate->output_color,
                 candidate->dx,
                 candidate->dy);
    } else {
        snprintf(candidate->name,
                 sizeof(candidate->name),
                 "hypothesis_%s_%s_ref_none_bg_%u_color_%u_to_%u_%+d_%+d",
                 hypothesis->template_ref->name,
                 arc_object_selector_short_name(candidate->selector_kind),
                 (unsigned)candidate->background_color,
                 (unsigned)candidate->input_color,
                 (unsigned)candidate->output_color,
                 candidate->dx,
                 candidate->dy);
    }
    return 1;
}

static int arc_apply_translate_hypothesis(const NNHypothesis *hypothesis,
                                          const ARCGrid *input,
                                          ARCGrid *output)
{
    ARCTranslateHypothesisSpec spec;
    ARCScene scene;
    ARCObject object;
    if (!arc_translate_spec_from_hypothesis(hypothesis, &spec) ||
        !arc_extract_scene(input, spec.background, &scene) ||
        !arc_select_scene_object(&scene,
                                 spec.selector_kind,
                                 spec.input_color,
                                 spec.reference_color,
                                 &object)) {
        return 0;
    }

    arc_init_grid(output, input->width, input->height);
    for (size_t cell = 0; cell < arc_grid_cells(output); ++cell) {
        output->cells[cell] = spec.background;
    }
    for (size_t y = object.min_y; y <= object.max_y; ++y) {
        for (size_t x = object.min_x; x <= object.max_x; ++x) {
            size_t cell = y * input->width + x;
            if (input->cells[cell] != spec.input_color) {
                continue;
            }
            int out_x = (int)x + spec.dx;
            int out_y = (int)y + spec.dy;
            if (out_x < 0 || out_y < 0 ||
                out_x >= (int)input->width || out_y >= (int)input->height) {
                return 0;
            }
            output->cells[(size_t)out_y * output->width + (size_t)out_x] =
                spec.output_color;
        }
    }
    return 1;
}

static const NNHypothesisTemplate *arc_hypothesis_template_move_to_anchor(void)
{
    static NNHypothesisTemplate template_ref;
    static int initialized = 0;
    if (initialized) {
        return &template_ref;
    }
    nn_hypothesis_template_init(&template_ref,
                                "arc_agi",
                                "move_to_anchor",
                                1025,
                                NN_PROGRAM_VALUE_GRID,
                                NN_PROGRAM_VALUE_GRID);
    nn_hypothesis_template_add_stage(&template_ref,
                                     "object_scene",
                                     NN_PROGRAM_KIND_TRANSFORM,
                                     NN_PROGRAM_VALUE_GRID,
                                     NN_PROGRAM_VALUE_OBJECT_SET,
                                     120);
    nn_hypothesis_template_add_stage(&template_ref,
                                     "select_subject",
                                     NN_PROGRAM_KIND_SELECTOR,
                                     NN_PROGRAM_VALUE_OBJECT_SET,
                                     NN_PROGRAM_VALUE_OBJECT,
                                     121);
    nn_hypothesis_template_add_stage(&template_ref,
                                     "select_anchor",
                                     NN_PROGRAM_KIND_SELECTOR,
                                     NN_PROGRAM_VALUE_OBJECT,
                                     NN_PROGRAM_VALUE_OBJECT,
                                     122);
    nn_hypothesis_template_add_stage(&template_ref,
                                     "align",
                                     NN_PROGRAM_KIND_TRANSFORM,
                                     NN_PROGRAM_VALUE_OBJECT,
                                     NN_PROGRAM_VALUE_OBJECT,
                                     123);
    nn_hypothesis_template_add_stage(&template_ref,
                                     "render",
                                     NN_PROGRAM_KIND_TRANSFORM,
                                     NN_PROGRAM_VALUE_OBJECT,
                                     NN_PROGRAM_VALUE_GRID,
                                     124);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "object_scene.background",
                                    NN_HYPOTHESIS_SLOT_COLOR,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "subject.selector_kind",
                                    NN_HYPOTHESIS_SLOT_SELECTOR_KIND,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "subject.color",
                                    NN_HYPOTHESIS_SLOT_COLOR,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "anchor.selector_kind",
                                    NN_HYPOTHESIS_SLOT_SELECTOR_KIND,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "anchor.color",
                                    NN_HYPOTHESIS_SLOT_COLOR,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "align.axis",
                                    NN_HYPOTHESIS_SLOT_INTEGER,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "align.offset",
                                    NN_HYPOTHESIS_SLOT_OFFSET,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "render.output_color",
                                    NN_HYPOTHESIS_SLOT_COLOR,
                                    1, 0.0, 0);
    initialized = 1;
    return &template_ref;
}

typedef struct {
    uint8_t background;
    uint8_t subject_selector_kind;
    uint8_t subject_color;
    uint8_t anchor_selector_kind;
    uint8_t anchor_color;
    int axis;
    int offset;
    uint8_t output_color;
} ARCMoveToAnchorHypothesisSpec;

static const char *arc_anchor_axis_short_name(int axis)
{
    switch (axis) {
    case 0:
        return "x";
    case 1:
        return "y";
    case 2:
        return "both";
    }
    return "unknown";
}

static int arc_move_to_anchor_spec_from_hypothesis(const NNHypothesis *hypothesis,
                                                   ARCMoveToAnchorHypothesisSpec *spec)
{
    if (hypothesis == NULL || spec == NULL ||
        hypothesis->template_ref != arc_hypothesis_template_move_to_anchor() ||
        !nn_hypothesis_is_fully_bound(hypothesis)) {
        return 0;
    }
    int64_t background = 0;
    int64_t subject_selector_kind = 0;
    int64_t subject_color = 0;
    int64_t anchor_selector_kind = 0;
    int64_t anchor_color = 0;
    int64_t axis = 0;
    int64_t offset = 0;
    int64_t output_color = 0;
    if (!nn_hypothesis_get(hypothesis,
                           "object_scene.background",
                           NULL,
                           &background) ||
        !nn_hypothesis_get(hypothesis,
                           "subject.selector_kind",
                           NULL,
                           &subject_selector_kind) ||
        !nn_hypothesis_get(hypothesis,
                           "subject.color",
                           NULL,
                           &subject_color) ||
        !nn_hypothesis_get(hypothesis,
                           "anchor.selector_kind",
                           NULL,
                           &anchor_selector_kind) ||
        !nn_hypothesis_get(hypothesis,
                           "anchor.color",
                           NULL,
                           &anchor_color) ||
        !nn_hypothesis_get(hypothesis, "align.axis", NULL, &axis) ||
        !nn_hypothesis_get(hypothesis, "align.offset", NULL, &offset) ||
        !nn_hypothesis_get(hypothesis,
                           "render.output_color",
                           NULL,
                           &output_color)) {
        return 0;
    }
    if (background < 0 || background >= ARC_COLOR_COUNT ||
        subject_selector_kind <= ARC_OBJECT_SELECTOR_ALL_COLOR ||
        subject_selector_kind >= ARC_OBJECT_SELECTOR_KIND_COUNT ||
        anchor_selector_kind <= ARC_OBJECT_SELECTOR_ALL_COLOR ||
        anchor_selector_kind >= ARC_OBJECT_SELECTOR_KIND_COUNT ||
        subject_selector_kind == ARC_OBJECT_SELECTOR_NEAREST_TO_COLOR ||
        anchor_selector_kind == ARC_OBJECT_SELECTOR_NEAREST_TO_COLOR ||
        subject_color < 0 || subject_color >= ARC_COLOR_COUNT ||
        anchor_color < 0 || anchor_color >= ARC_COLOR_COUNT ||
        subject_color == anchor_color ||
        axis < 0 || axis > 2 ||
        offset < -ARC_MAX_GRID_WIDTH || offset > ARC_MAX_GRID_WIDTH ||
        output_color < 0 || output_color >= ARC_COLOR_COUNT) {
        return 0;
    }
    spec->background = (uint8_t)background;
    spec->subject_selector_kind = (uint8_t)subject_selector_kind;
    spec->subject_color = (uint8_t)subject_color;
    spec->anchor_selector_kind = (uint8_t)anchor_selector_kind;
    spec->anchor_color = (uint8_t)anchor_color;
    spec->axis = (int)axis;
    spec->offset = (int)offset;
    spec->output_color = (uint8_t)output_color;
    return 1;
}

static int arc_candidate_from_move_to_anchor_hypothesis(const NNHypothesis *hypothesis,
                                                       ARCTransformCandidate *candidate)
{
    ARCMoveToAnchorHypothesisSpec spec;
    if (candidate == NULL ||
        !arc_move_to_anchor_spec_from_hypothesis(hypothesis, &spec)) {
        return 0;
    }
    arc_init_candidate(candidate, ARC_RULE_HYPOTHESIS_APPLIED, "");
    candidate->background_color = spec.background;
    candidate->selector_kind = spec.subject_selector_kind;
    candidate->output_selector_kind = spec.anchor_selector_kind;
    candidate->input_color = spec.subject_color;
    candidate->reference_color = spec.anchor_color;
    candidate->output_color = spec.output_color;
    candidate->dx = spec.axis;
    candidate->dy = spec.offset;
    if (!arc_store_hypothesis_candidate(candidate, hypothesis)) {
        return 0;
    }
    snprintf(candidate->name,
             sizeof(candidate->name),
             "hypothesis_%s_subject_%s_%u_anchor_%s_%u_axis_%s_off_%+d_bg_%u_to_%u",
             hypothesis->template_ref->name,
             arc_object_selector_short_name(spec.subject_selector_kind),
             (unsigned)spec.subject_color,
             arc_object_selector_short_name(spec.anchor_selector_kind),
             (unsigned)spec.anchor_color,
             arc_anchor_axis_short_name(spec.axis),
             spec.offset,
             (unsigned)spec.background,
             (unsigned)spec.output_color);
    return 1;
}

static int arc_apply_move_to_anchor_hypothesis(const NNHypothesis *hypothesis,
                                               const ARCGrid *input,
                                               ARCGrid *output)
{
    ARCMoveToAnchorHypothesisSpec spec;
    ARCScene scene;
    ARCObject subject;
    ARCObject anchor;
    if (!arc_move_to_anchor_spec_from_hypothesis(hypothesis, &spec) ||
        !arc_extract_scene(input, spec.background, &scene) ||
        !arc_select_scene_object(&scene,
                                 spec.subject_selector_kind,
                                 spec.subject_color,
                                 0,
                                 &subject) ||
        !arc_select_scene_object(&scene,
                                 spec.anchor_selector_kind,
                                 spec.anchor_color,
                                 0,
                                 &anchor)) {
        return 0;
    }

    int out_min_x = (int)subject.min_x;
    int out_min_y = (int)subject.min_y;
    if (spec.axis == 0 || spec.axis == 2) {
        out_min_x = (int)anchor.min_x + spec.offset;
    }
    if (spec.axis == 1 || spec.axis == 2) {
        out_min_y = (int)anchor.min_y + spec.offset;
    }

    arc_init_grid(output, input->width, input->height);
    for (size_t cell = 0; cell < arc_grid_cells(output); ++cell) {
        output->cells[cell] = spec.background;
    }
    for (size_t y = subject.min_y; y <= subject.max_y; ++y) {
        for (size_t x = subject.min_x; x <= subject.max_x; ++x) {
            size_t cell = y * input->width + x;
            if (input->cells[cell] != spec.subject_color) {
                continue;
            }
            int out_x = out_min_x + (int)x - (int)subject.min_x;
            int out_y = out_min_y + (int)y - (int)subject.min_y;
            if (out_x < 0 || out_y < 0 ||
                out_x >= (int)input->width || out_y >= (int)input->height) {
                return 0;
            }
            output->cells[(size_t)out_y * output->width + (size_t)out_x] =
                spec.output_color;
        }
    }
    return 1;
}

static const NNHypothesisTemplate *arc_hypothesis_template_recolor_by_reference(void)
{
    static NNHypothesisTemplate template_ref;
    static int initialized = 0;
    if (initialized) {
        return &template_ref;
    }
    nn_hypothesis_template_init(&template_ref,
                                "arc_agi",
                                "recolor_by_reference",
                                1026,
                                NN_PROGRAM_VALUE_GRID,
                                NN_PROGRAM_VALUE_GRID);
    nn_hypothesis_template_add_stage(&template_ref,
                                     "object_scene",
                                     NN_PROGRAM_KIND_TRANSFORM,
                                     NN_PROGRAM_VALUE_GRID,
                                     NN_PROGRAM_VALUE_OBJECT_SET,
                                     130);
    nn_hypothesis_template_add_stage(&template_ref,
                                     "select",
                                     NN_PROGRAM_KIND_SELECTOR,
                                     NN_PROGRAM_VALUE_OBJECT_SET,
                                     NN_PROGRAM_VALUE_OBJECT,
                                     131);
    nn_hypothesis_template_add_stage(&template_ref,
                                     "recolor",
                                     NN_PROGRAM_KIND_TRANSFORM,
                                     NN_PROGRAM_VALUE_OBJECT,
                                     NN_PROGRAM_VALUE_OBJECT,
                                     132);
    nn_hypothesis_template_add_stage(&template_ref,
                                     "render",
                                     NN_PROGRAM_KIND_TRANSFORM,
                                     NN_PROGRAM_VALUE_OBJECT,
                                     NN_PROGRAM_VALUE_GRID,
                                     133);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "object_scene.background",
                                    NN_HYPOTHESIS_SLOT_COLOR,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "select.selector_kind",
                                    NN_HYPOTHESIS_SLOT_SELECTOR_KIND,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "select.color",
                                    NN_HYPOTHESIS_SLOT_COLOR,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "select.ref_used",
                                    NN_HYPOTHESIS_SLOT_BOOLEAN,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "select.ref_color",
                                    NN_HYPOTHESIS_SLOT_REFERENCE_COLOR,
                                    0, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "render.output_color",
                                    NN_HYPOTHESIS_SLOT_COLOR,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "render.mode",
                                    NN_HYPOTHESIS_SLOT_RENDER_MODE,
                                    1, 0.0, 0);
    initialized = 1;
    return &template_ref;
}

typedef struct {
    uint8_t background;
    uint8_t selector_kind;
    uint8_t input_color;
    uint8_t reference_color_used;
    uint8_t reference_color;
    uint8_t output_color;
    uint8_t render_mode;
} ARCRecolorHypothesisSpec;

static int arc_recolor_spec_from_hypothesis(const NNHypothesis *hypothesis,
                                            ARCRecolorHypothesisSpec *spec)
{
    if (hypothesis == NULL || spec == NULL ||
        hypothesis->template_ref != arc_hypothesis_template_recolor_by_reference() ||
        !nn_hypothesis_is_fully_bound(hypothesis)) {
        return 0;
    }
    int64_t background = 0;
    int64_t selector_kind = 0;
    int64_t input_color = 0;
    int64_t ref_used = 0;
    int64_t ref_color = 0;
    int64_t output_color = 0;
    int64_t render_mode = 0;
    if (!nn_hypothesis_get(hypothesis,
                           "object_scene.background",
                           NULL,
                           &background) ||
        !nn_hypothesis_get(hypothesis,
                           "select.selector_kind",
                           NULL,
                           &selector_kind) ||
        !nn_hypothesis_get(hypothesis, "select.color", NULL, &input_color) ||
        !nn_hypothesis_get(hypothesis, "select.ref_used", NULL, &ref_used) ||
        !nn_hypothesis_get(hypothesis, "select.ref_color", NULL, &ref_color) ||
        !nn_hypothesis_get(hypothesis,
                           "render.output_color",
                           NULL,
                           &output_color) ||
        !nn_hypothesis_get(hypothesis, "render.mode", NULL, &render_mode)) {
        return 0;
    }
    if (background < 0 || background >= ARC_COLOR_COUNT ||
        selector_kind <= ARC_OBJECT_SELECTOR_ALL_COLOR ||
        selector_kind >= ARC_OBJECT_SELECTOR_KIND_COUNT ||
        input_color < 0 || input_color >= ARC_COLOR_COUNT ||
        ref_used < 0 || ref_used > 1 ||
        ref_color < 0 || ref_color >= ARC_COLOR_COUNT ||
        output_color < 0 || output_color >= ARC_COLOR_COUNT ||
        render_mode < 0 || render_mode >= ARC_OBJECT_RENDER_KIND_COUNT) {
        return 0;
    }
    if ((ref_used != 0 && selector_kind != ARC_OBJECT_SELECTOR_NEAREST_TO_COLOR) ||
        (ref_used == 0 && selector_kind == ARC_OBJECT_SELECTOR_NEAREST_TO_COLOR) ||
        (ref_used == 0 && ref_color != 0)) {
        return 0;
    }
    spec->background = (uint8_t)background;
    spec->selector_kind = (uint8_t)selector_kind;
    spec->input_color = (uint8_t)input_color;
    spec->reference_color_used = (uint8_t)ref_used;
    spec->reference_color = (uint8_t)ref_color;
    spec->output_color = (uint8_t)output_color;
    spec->render_mode = (uint8_t)render_mode;
    return 1;
}

static int arc_candidate_from_recolor_hypothesis(const NNHypothesis *hypothesis,
                                                 ARCTransformCandidate *candidate)
{
    ARCRecolorHypothesisSpec spec;
    if (candidate == NULL ||
        !arc_recolor_spec_from_hypothesis(hypothesis, &spec)) {
        return 0;
    }
    arc_init_candidate(candidate, ARC_RULE_HYPOTHESIS_APPLIED, "");
    candidate->background_color = spec.background;
    candidate->selector_kind = spec.selector_kind;
    candidate->output_selector_kind = ARC_OBJECT_SELECTOR_UNIQUE_COLOR;
    candidate->input_color = spec.input_color;
    candidate->reference_color = spec.reference_color;
    candidate->output_color = spec.output_color;
    candidate->render_mode = spec.render_mode;
    if (!arc_store_hypothesis_candidate(candidate, hypothesis)) {
        return 0;
    }
    snprintf(candidate->name,
             sizeof(candidate->name),
             "hypothesis_%s_%s_ref_%s%u_bg_%u_color_%u_to_%u_%s",
             hypothesis->template_ref->name,
             arc_object_selector_short_name(spec.selector_kind),
             spec.reference_color_used ? "" : "none_",
             spec.reference_color_used ? (unsigned)spec.reference_color : 0u,
             (unsigned)spec.background,
             (unsigned)spec.input_color,
             (unsigned)spec.output_color,
             spec.render_mode == ARC_OBJECT_RENDER_ONLY ? "only" : "overlay");
    return 1;
}

static int arc_apply_recolor_hypothesis(const NNHypothesis *hypothesis,
                                        const ARCGrid *input,
                                        ARCGrid *output)
{
    ARCRecolorHypothesisSpec spec;
    ARCScene scene;
    ARCObject object;
    if (!arc_recolor_spec_from_hypothesis(hypothesis, &spec) ||
        !arc_extract_scene(input, spec.background, &scene) ||
        !arc_select_scene_object(&scene,
                                 spec.selector_kind,
                                 spec.input_color,
                                 spec.reference_color,
                                 &object)) {
        return 0;
    }
    if (spec.render_mode == ARC_OBJECT_RENDER_ONLY) {
        arc_init_grid(output, input->width, input->height);
        for (size_t cell = 0; cell < arc_grid_cells(output); ++cell) {
            output->cells[cell] = spec.background;
        }
    } else {
        *output = *input;
    }
    for (size_t y = object.min_y; y <= object.max_y; ++y) {
        for (size_t x = object.min_x; x <= object.max_x; ++x) {
            size_t cell = y * input->width + x;
            if (input->cells[cell] == spec.input_color) {
                output->cells[cell] = spec.output_color;
            }
        }
    }
    return 1;
}

static const NNHypothesisTemplate *arc_hypothesis_template_connect_aligned(void)
{
    static NNHypothesisTemplate template_ref;
    static int initialized = 0;
    if (initialized) {
        return &template_ref;
    }
    nn_hypothesis_template_init(&template_ref,
                                "arc_agi",
                                "connect_aligned",
                                1027,
                                NN_PROGRAM_VALUE_GRID,
                                NN_PROGRAM_VALUE_GRID);
    nn_hypothesis_template_add_stage(&template_ref,
                                     "object_scene",
                                     NN_PROGRAM_KIND_TRANSFORM,
                                     NN_PROGRAM_VALUE_GRID,
                                     NN_PROGRAM_VALUE_OBJECT_SET,
                                     140);
    nn_hypothesis_template_add_stage(&template_ref,
                                     "connect",
                                     NN_PROGRAM_KIND_TRANSFORM,
                                     NN_PROGRAM_VALUE_OBJECT_SET,
                                     NN_PROGRAM_VALUE_GRID,
                                     141);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "object_scene.background",
                                    NN_HYPOTHESIS_SLOT_COLOR,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "connect.color",
                                    NN_HYPOTHESIS_SLOT_COLOR,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "connect.output_color",
                                    NN_HYPOTHESIS_SLOT_COLOR,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "connect.mode",
                                    NN_HYPOTHESIS_SLOT_INTEGER,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "connect.render_mode",
                                    NN_HYPOTHESIS_SLOT_RENDER_MODE,
                                    1, 0.0, 0);
    initialized = 1;
    return &template_ref;
}

typedef struct {
    uint8_t background;
    uint8_t input_color;
    uint8_t output_color;
    int mode;
    uint8_t render_mode;
} ARCConnectHypothesisSpec;

static int arc_connect_spec_from_hypothesis(const NNHypothesis *hypothesis,
                                            ARCConnectHypothesisSpec *spec)
{
    if (hypothesis == NULL || spec == NULL ||
        hypothesis->template_ref != arc_hypothesis_template_connect_aligned() ||
        !nn_hypothesis_is_fully_bound(hypothesis)) {
        return 0;
    }
    int64_t background = 0;
    int64_t input_color = 0;
    int64_t output_color = 0;
    int64_t mode = 0;
    int64_t render_mode = 0;
    if (!nn_hypothesis_get(hypothesis,
                           "object_scene.background",
                           NULL,
                           &background) ||
        !nn_hypothesis_get(hypothesis, "connect.color", NULL, &input_color) ||
        !nn_hypothesis_get(hypothesis,
                           "connect.output_color",
                           NULL,
                           &output_color) ||
        !nn_hypothesis_get(hypothesis, "connect.mode", NULL, &mode) ||
        !nn_hypothesis_get(hypothesis,
                           "connect.render_mode",
                           NULL,
                           &render_mode)) {
        return 0;
    }
    if (background < 0 || background >= ARC_COLOR_COUNT ||
        input_color < 0 || input_color >= ARC_COLOR_COUNT ||
        output_color < 0 || output_color >= ARC_COLOR_COUNT ||
        mode < ARC_OBJECT_CONNECT_HORIZONTAL ||
        mode > ARC_OBJECT_CONNECT_BOTH ||
        render_mode < 0 || render_mode >= ARC_OBJECT_RENDER_KIND_COUNT) {
        return 0;
    }
    spec->background = (uint8_t)background;
    spec->input_color = (uint8_t)input_color;
    spec->output_color = (uint8_t)output_color;
    spec->mode = (int)mode;
    spec->render_mode = (uint8_t)render_mode;
    return 1;
}

static void arc_connect_candidate_from_spec(const ARCConnectHypothesisSpec *spec,
                                            ARCTransformCandidate *candidate)
{
    arc_init_candidate(candidate, ARC_RULE_OBJECT_CONNECT_ALIGNED, "");
    candidate->background_color = spec->background;
    candidate->input_color = spec->input_color;
    candidate->output_color = spec->output_color;
    candidate->selector_kind = ARC_OBJECT_SELECTOR_ALL_COLOR;
    candidate->output_selector_kind = ARC_OBJECT_SELECTOR_UNIQUE_COLOR;
    candidate->dx = spec->mode;
    candidate->render_mode = spec->render_mode;
}

static int arc_candidate_from_connect_hypothesis(const NNHypothesis *hypothesis,
                                                 ARCTransformCandidate *candidate)
{
    ARCConnectHypothesisSpec spec;
    if (candidate == NULL ||
        !arc_connect_spec_from_hypothesis(hypothesis, &spec)) {
        return 0;
    }
    arc_connect_candidate_from_spec(&spec, candidate);
    candidate->op = ARC_RULE_HYPOTHESIS_APPLIED;
    if (!arc_store_hypothesis_candidate(candidate, hypothesis)) {
        return 0;
    }
    snprintf(candidate->name,
             sizeof(candidate->name),
             "hypothesis_%s_bg_%u_color_%u_to_%u_mode_%d_%s",
             hypothesis->template_ref->name,
             (unsigned)spec.background,
             (unsigned)spec.input_color,
             (unsigned)spec.output_color,
             spec.mode,
             spec.render_mode == ARC_OBJECT_RENDER_ONLY ? "only" : "overlay");
    return 1;
}

static int arc_apply_connect_hypothesis(const NNHypothesis *hypothesis,
                                        const ARCGrid *input,
                                        ARCGrid *output)
{
    ARCConnectHypothesisSpec spec;
    ARCTransformCandidate candidate;
    if (!arc_connect_spec_from_hypothesis(hypothesis, &spec)) {
        return 0;
    }
    arc_connect_candidate_from_spec(&spec, &candidate);
    return arc_apply_object_connect_candidate(&candidate, input, output);
}

static const NNHypothesisTemplate *arc_hypothesis_template_fill_enclosed(void)
{
    static NNHypothesisTemplate template_ref;
    static int initialized = 0;
    if (initialized) {
        return &template_ref;
    }
    nn_hypothesis_template_init(&template_ref,
                                "arc_agi",
                                "fill_enclosed",
                                1028,
                                NN_PROGRAM_VALUE_GRID,
                                NN_PROGRAM_VALUE_GRID);
    nn_hypothesis_template_add_stage(&template_ref,
                                     "object_scene",
                                     NN_PROGRAM_KIND_TRANSFORM,
                                     NN_PROGRAM_VALUE_GRID,
                                     NN_PROGRAM_VALUE_OBJECT_SET,
                                     150);
    nn_hypothesis_template_add_stage(&template_ref,
                                     "fill",
                                     NN_PROGRAM_KIND_TRANSFORM,
                                     NN_PROGRAM_VALUE_OBJECT_SET,
                                     NN_PROGRAM_VALUE_GRID,
                                     151);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "object_scene.background",
                                    NN_HYPOTHESIS_SLOT_COLOR,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "fill.border_color",
                                    NN_HYPOTHESIS_SLOT_COLOR,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "fill.output_color",
                                    NN_HYPOTHESIS_SLOT_COLOR,
                                    1, 0.0, 0);
    initialized = 1;
    return &template_ref;
}

typedef struct {
    uint8_t background;
    uint8_t border_color;
    uint8_t output_color;
} ARCFillHypothesisSpec;

static int arc_fill_spec_from_hypothesis(const NNHypothesis *hypothesis,
                                         ARCFillHypothesisSpec *spec)
{
    if (hypothesis == NULL || spec == NULL ||
        hypothesis->template_ref != arc_hypothesis_template_fill_enclosed() ||
        !nn_hypothesis_is_fully_bound(hypothesis)) {
        return 0;
    }
    int64_t background = 0;
    int64_t border_color = 0;
    int64_t output_color = 0;
    if (!nn_hypothesis_get(hypothesis,
                           "object_scene.background",
                           NULL,
                           &background) ||
        !nn_hypothesis_get(hypothesis,
                           "fill.border_color",
                           NULL,
                           &border_color) ||
        !nn_hypothesis_get(hypothesis,
                           "fill.output_color",
                           NULL,
                           &output_color)) {
        return 0;
    }
    if (background < 0 || background >= ARC_COLOR_COUNT ||
        border_color < 0 || border_color >= ARC_COLOR_COUNT ||
        output_color < 0 || output_color >= ARC_COLOR_COUNT ||
        border_color == output_color) {
        return 0;
    }
    spec->background = (uint8_t)background;
    spec->border_color = (uint8_t)border_color;
    spec->output_color = (uint8_t)output_color;
    return 1;
}

static void arc_fill_candidate_from_spec(const ARCFillHypothesisSpec *spec,
                                         ARCTransformCandidate *candidate)
{
    arc_init_candidate(candidate, ARC_RULE_OBJECT_FILL_ENCLOSED, "");
    candidate->background_color = spec->background;
    candidate->input_color = spec->border_color;
    candidate->output_color = spec->output_color;
    candidate->selector_kind = ARC_OBJECT_SELECTOR_ALL_COLOR;
    candidate->output_selector_kind = ARC_OBJECT_SELECTOR_UNIQUE_COLOR;
    candidate->render_mode = ARC_OBJECT_RENDER_OVERLAY;
}

static int arc_candidate_from_fill_hypothesis(const NNHypothesis *hypothesis,
                                              ARCTransformCandidate *candidate)
{
    ARCFillHypothesisSpec spec;
    if (candidate == NULL || !arc_fill_spec_from_hypothesis(hypothesis, &spec)) {
        return 0;
    }
    arc_fill_candidate_from_spec(&spec, candidate);
    candidate->op = ARC_RULE_HYPOTHESIS_APPLIED;
    if (!arc_store_hypothesis_candidate(candidate, hypothesis)) {
        return 0;
    }
    snprintf(candidate->name,
             sizeof(candidate->name),
             "hypothesis_%s_bg_%u_border_%u_to_%u",
             hypothesis->template_ref->name,
             (unsigned)spec.background,
             (unsigned)spec.border_color,
             (unsigned)spec.output_color);
    return 1;
}

static int arc_apply_fill_hypothesis(const NNHypothesis *hypothesis,
                                     const ARCGrid *input,
                                     ARCGrid *output)
{
    ARCFillHypothesisSpec spec;
    ARCTransformCandidate candidate;
    if (!arc_fill_spec_from_hypothesis(hypothesis, &spec)) {
        return 0;
    }
    arc_fill_candidate_from_spec(&spec, &candidate);
    return arc_apply_object_fill_enclosed_candidate(&candidate, input, output);
}

static const NNHypothesisTemplate *arc_hypothesis_template_object_extract(void)
{
    static NNHypothesisTemplate template_ref;
    static int initialized = 0;
    if (initialized) {
        return &template_ref;
    }
    nn_hypothesis_template_init(&template_ref,
                                "arc_agi",
                                "object_extract",
                                1029,
                                NN_PROGRAM_VALUE_GRID,
                                NN_PROGRAM_VALUE_GRID);
    nn_hypothesis_template_add_stage(&template_ref,
                                     "object_scene",
                                     NN_PROGRAM_KIND_TRANSFORM,
                                     NN_PROGRAM_VALUE_GRID,
                                     NN_PROGRAM_VALUE_OBJECT_SET,
                                     160);
    nn_hypothesis_template_add_stage(&template_ref,
                                     "select",
                                     NN_PROGRAM_KIND_SELECTOR,
                                     NN_PROGRAM_VALUE_OBJECT_SET,
                                     NN_PROGRAM_VALUE_OBJECT,
                                     161);
    nn_hypothesis_template_add_stage(&template_ref,
                                     "extract",
                                     NN_PROGRAM_KIND_TRANSFORM,
                                     NN_PROGRAM_VALUE_OBJECT,
                                     NN_PROGRAM_VALUE_GRID,
                                     162);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "object_scene.background",
                                    NN_HYPOTHESIS_SLOT_COLOR,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "select.selector_kind",
                                    NN_HYPOTHESIS_SLOT_SELECTOR_KIND,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "select.color",
                                    NN_HYPOTHESIS_SLOT_COLOR,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "select.ref_used",
                                    NN_HYPOTHESIS_SLOT_BOOLEAN,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "select.ref_color",
                                    NN_HYPOTHESIS_SLOT_REFERENCE_COLOR,
                                    0, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "extract.mode",
                                    NN_HYPOTHESIS_SLOT_INTEGER,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "extract.output_color",
                                    NN_HYPOTHESIS_SLOT_COLOR,
                                    1, 0.0, 0);
    initialized = 1;
    return &template_ref;
}

typedef struct {
    uint8_t background;
    uint8_t selector_kind;
    uint8_t input_color;
    uint8_t reference_color_used;
    uint8_t reference_color;
    int mode;
    uint8_t output_color;
} ARCExtractHypothesisSpec;

static const char *arc_extract_mode_name(int mode)
{
    switch (mode) {
    case 0:
        return "crop";
    case 1:
        return "crop_recolor";
    case 2:
        return "mask";
    }
    return "unknown";
}

static ARCTransformRule arc_extract_mode_rule(int mode)
{
    switch (mode) {
    case 0:
        return ARC_RULE_OBJECT_CROP_LARGEST;
    case 1:
        return ARC_RULE_OBJECT_CROP_LARGEST_RECOLOR;
    case 2:
        return ARC_RULE_OBJECT_MASK_LARGEST;
    }
    return ARC_RULE_OBJECT_CROP_LARGEST;
}

static int arc_extract_spec_from_hypothesis(const NNHypothesis *hypothesis,
                                            ARCExtractHypothesisSpec *spec)
{
    if (hypothesis == NULL || spec == NULL ||
        hypothesis->template_ref != arc_hypothesis_template_object_extract() ||
        !nn_hypothesis_is_fully_bound(hypothesis)) {
        return 0;
    }
    int64_t background = 0;
    int64_t selector_kind = 0;
    int64_t input_color = 0;
    int64_t ref_used = 0;
    int64_t ref_color = 0;
    int64_t mode = 0;
    int64_t output_color = 0;
    if (!nn_hypothesis_get(hypothesis,
                           "object_scene.background",
                           NULL,
                           &background) ||
        !nn_hypothesis_get(hypothesis,
                           "select.selector_kind",
                           NULL,
                           &selector_kind) ||
        !nn_hypothesis_get(hypothesis, "select.color", NULL, &input_color) ||
        !nn_hypothesis_get(hypothesis, "select.ref_used", NULL, &ref_used) ||
        !nn_hypothesis_get(hypothesis, "select.ref_color", NULL, &ref_color) ||
        !nn_hypothesis_get(hypothesis, "extract.mode", NULL, &mode) ||
        !nn_hypothesis_get(hypothesis,
                           "extract.output_color",
                           NULL,
                           &output_color)) {
        return 0;
    }
    if (background < 0 || background >= ARC_COLOR_COUNT ||
        selector_kind <= ARC_OBJECT_SELECTOR_ALL_COLOR ||
        selector_kind >= ARC_OBJECT_SELECTOR_KIND_COUNT ||
        input_color < 0 || input_color >= ARC_COLOR_COUNT ||
        ref_used < 0 || ref_used > 1 ||
        ref_color < 0 || ref_color >= ARC_COLOR_COUNT ||
        mode < 0 || mode > 2 ||
        output_color < 0 || output_color >= ARC_COLOR_COUNT) {
        return 0;
    }
    if ((ref_used != 0 && selector_kind != ARC_OBJECT_SELECTOR_NEAREST_TO_COLOR) ||
        (ref_used == 0 && selector_kind == ARC_OBJECT_SELECTOR_NEAREST_TO_COLOR) ||
        (ref_used == 0 && ref_color != 0)) {
        return 0;
    }
    spec->background = (uint8_t)background;
    spec->selector_kind = (uint8_t)selector_kind;
    spec->input_color = (uint8_t)input_color;
    spec->reference_color_used = (uint8_t)ref_used;
    spec->reference_color = (uint8_t)ref_color;
    spec->mode = (int)mode;
    spec->output_color = (uint8_t)output_color;
    return 1;
}

static void arc_extract_candidate_from_spec(const ARCExtractHypothesisSpec *spec,
                                            ARCTransformCandidate *candidate)
{
    arc_init_candidate(candidate, arc_extract_mode_rule(spec->mode), "");
    candidate->background_color = spec->background;
    candidate->selector_kind = spec->selector_kind;
    candidate->output_selector_kind = ARC_OBJECT_SELECTOR_UNIQUE_COLOR;
    candidate->input_color = spec->input_color;
    candidate->reference_color = spec->reference_color;
    candidate->output_color = spec->mode == 0 ? spec->input_color : spec->output_color;
    candidate->render_mode = ARC_OBJECT_RENDER_ONLY;
}

static int arc_candidate_from_extract_hypothesis(const NNHypothesis *hypothesis,
                                                 ARCTransformCandidate *candidate)
{
    ARCExtractHypothesisSpec spec;
    if (candidate == NULL ||
        !arc_extract_spec_from_hypothesis(hypothesis, &spec)) {
        return 0;
    }
    arc_extract_candidate_from_spec(&spec, candidate);
    candidate->op = ARC_RULE_HYPOTHESIS_APPLIED;
    if (!arc_store_hypothesis_candidate(candidate, hypothesis)) {
        return 0;
    }
    snprintf(candidate->name,
             sizeof(candidate->name),
             "hypothesis_%s_%s_%s_ref_%s%u_bg_%u_color_%u_to_%u",
             hypothesis->template_ref->name,
             arc_extract_mode_name(spec.mode),
             arc_object_selector_short_name(spec.selector_kind),
             spec.reference_color_used ? "" : "none_",
             spec.reference_color_used ? (unsigned)spec.reference_color : 0u,
             (unsigned)spec.background,
             (unsigned)spec.input_color,
             (unsigned)spec.output_color);
    return 1;
}

static int arc_apply_extract_hypothesis(const NNHypothesis *hypothesis,
                                        const ARCGrid *input,
                                        ARCGrid *output)
{
    ARCExtractHypothesisSpec spec;
    ARCTransformCandidate candidate;
    if (!arc_extract_spec_from_hypothesis(hypothesis, &spec)) {
        return 0;
    }
    arc_extract_candidate_from_spec(&spec, &candidate);
    return arc_apply_largest_component_candidate(&candidate, input, output);
}

static const NNHypothesisTemplate *arc_hypothesis_template_extend_aligned_sequence(void)
{
    static NNHypothesisTemplate template_ref;
    static int initialized = 0;
    if (initialized) {
        return &template_ref;
    }
    nn_hypothesis_template_init(&template_ref,
                                "arc_agi",
                                "extend_aligned_sequence",
                                1030,
                                NN_PROGRAM_VALUE_GRID,
                                NN_PROGRAM_VALUE_GRID);
    nn_hypothesis_template_add_stage(&template_ref,
                                     "object_scene",
                                     NN_PROGRAM_KIND_TRANSFORM,
                                     NN_PROGRAM_VALUE_GRID,
                                     NN_PROGRAM_VALUE_OBJECT_SET,
                                     170);
    nn_hypothesis_template_add_stage(&template_ref,
                                     "infer_sequence",
                                     NN_PROGRAM_KIND_SELECTOR,
                                     NN_PROGRAM_VALUE_OBJECT_SET,
                                     NN_PROGRAM_VALUE_OBJECT_SET,
                                     171);
    nn_hypothesis_template_add_stage(&template_ref,
                                     "render_extension",
                                     NN_PROGRAM_KIND_TRANSFORM,
                                     NN_PROGRAM_VALUE_OBJECT_SET,
                                     NN_PROGRAM_VALUE_GRID,
                                     172);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "object_scene.background",
                                    NN_HYPOTHESIS_SLOT_COLOR,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "sequence.color",
                                    NN_HYPOTHESIS_SLOT_COLOR,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "sequence.output_color",
                                    NN_HYPOTHESIS_SLOT_COLOR,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "sequence.direction",
                                    NN_HYPOTHESIS_SLOT_INTEGER,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "sequence.extend_mode",
                                    NN_HYPOTHESIS_SLOT_INTEGER,
                                    1, 0.0, 0);
    initialized = 1;
    return &template_ref;
}

typedef struct {
    uint8_t background;
    uint8_t input_color;
    uint8_t output_color;
    int direction;
    int extend_mode;
} ARCSequenceHypothesisSpec;

typedef struct {
    int x;
    int y;
    int projection;
} ARCSequencePoint;

static const char *arc_sequence_direction_name(int direction)
{
    switch (direction) {
    case ARC_SEQUENCE_DIRECTION_HORIZONTAL:
        return "horizontal";
    case ARC_SEQUENCE_DIRECTION_VERTICAL:
        return "vertical";
    case ARC_SEQUENCE_DIRECTION_DIAGONAL_DOWN_RIGHT:
        return "diag_down_right";
    case ARC_SEQUENCE_DIRECTION_DIAGONAL_DOWN_LEFT:
        return "diag_down_left";
    }
    return "unknown";
}

static const char *arc_sequence_extend_mode_name(int extend_mode)
{
    switch (extend_mode) {
    case ARC_SEQUENCE_EXTEND_POSITIVE:
        return "positive";
    case ARC_SEQUENCE_EXTEND_NEGATIVE:
        return "negative";
    case ARC_SEQUENCE_EXTEND_BOTH:
        return "both";
    }
    return "unknown";
}

static int arc_sequence_spec_from_hypothesis(const NNHypothesis *hypothesis,
                                             ARCSequenceHypothesisSpec *spec)
{
    if (hypothesis == NULL || spec == NULL ||
        hypothesis->template_ref != arc_hypothesis_template_extend_aligned_sequence() ||
        !nn_hypothesis_is_fully_bound(hypothesis)) {
        return 0;
    }
    int64_t background = 0;
    int64_t input_color = 0;
    int64_t output_color = 0;
    int64_t direction = 0;
    int64_t extend_mode = 0;
    if (!nn_hypothesis_get(hypothesis,
                           "object_scene.background",
                           NULL,
                           &background) ||
        !nn_hypothesis_get(hypothesis, "sequence.color", NULL, &input_color) ||
        !nn_hypothesis_get(hypothesis,
                           "sequence.output_color",
                           NULL,
                           &output_color) ||
        !nn_hypothesis_get(hypothesis,
                           "sequence.direction",
                           NULL,
                           &direction) ||
        !nn_hypothesis_get(hypothesis,
                           "sequence.extend_mode",
                           NULL,
                           &extend_mode)) {
        return 0;
    }
    if (background < 0 || background >= ARC_COLOR_COUNT ||
        input_color < 0 || input_color >= ARC_COLOR_COUNT ||
        output_color < 0 || output_color >= ARC_COLOR_COUNT ||
        direction < 0 || direction >= ARC_SEQUENCE_DIRECTION_KIND_COUNT ||
        extend_mode < 0 || extend_mode >= ARC_SEQUENCE_EXTEND_KIND_COUNT ||
        input_color == background) {
        return 0;
    }
    spec->background = (uint8_t)background;
    spec->input_color = (uint8_t)input_color;
    spec->output_color = (uint8_t)output_color;
    spec->direction = (int)direction;
    spec->extend_mode = (int)extend_mode;
    return 1;
}

static int arc_sequence_point_projection(size_t x,
                                         size_t y,
                                         int direction,
                                         int *line_out,
                                         int *projection_out)
{
    int ix = (int)x;
    int iy = (int)y;
    switch (direction) {
    case ARC_SEQUENCE_DIRECTION_HORIZONTAL:
        *line_out = iy;
        *projection_out = ix;
        return 1;
    case ARC_SEQUENCE_DIRECTION_VERTICAL:
        *line_out = ix;
        *projection_out = iy;
        return 1;
    case ARC_SEQUENCE_DIRECTION_DIAGONAL_DOWN_RIGHT:
        *line_out = ix - iy;
        *projection_out = ix;
        return 1;
    case ARC_SEQUENCE_DIRECTION_DIAGONAL_DOWN_LEFT:
        *line_out = ix + iy;
        *projection_out = ix;
        return 1;
    default:
        return 0;
    }
}

static int arc_sequence_collect_points(const ARCGrid *input,
                                       uint8_t input_color,
                                       int direction,
                                       ARCSequencePoint *points,
                                       size_t *point_count_out,
                                       int *step_out)
{
    if (input == NULL || points == NULL ||
        point_count_out == NULL || step_out == NULL) {
        return 0;
    }
    size_t point_count = 0;
    int line = 0;
    int have_line = 0;
    for (size_t y = 0; y < input->height; ++y) {
        for (size_t x = 0; x < input->width; ++x) {
            if (input->cells[y * input->width + x] != input_color) {
                continue;
            }
            int current_line = 0;
            int projection = 0;
            if (!arc_sequence_point_projection(x,
                                               y,
                                               direction,
                                               &current_line,
                                               &projection)) {
                return 0;
            }
            if (!have_line) {
                line = current_line;
                have_line = 1;
            } else if (line != current_line) {
                return 0;
            }
            if (point_count >= ARC_MAX_GRID_CELLS) {
                return 0;
            }
            points[point_count].x = (int)x;
            points[point_count].y = (int)y;
            points[point_count].projection = projection;
            ++point_count;
        }
    }
    if (point_count < 2u) {
        return 0;
    }
    for (size_t i = 1; i < point_count; ++i) {
        ARCSequencePoint key = points[i];
        size_t j = i;
        while (j > 0 && key.projection < points[j - 1u].projection) {
            points[j] = points[j - 1u];
            --j;
        }
        points[j] = key;
    }
    int step = points[1].projection - points[0].projection;
    if (step <= 0) {
        return 0;
    }
    for (size_t i = 2; i < point_count; ++i) {
        if (points[i].projection - points[i - 1u].projection != step) {
            return 0;
        }
    }
    if (direction == ARC_SEQUENCE_DIRECTION_DIAGONAL_DOWN_RIGHT ||
        direction == ARC_SEQUENCE_DIRECTION_DIAGONAL_DOWN_LEFT) {
        for (size_t i = 1; i < point_count; ++i) {
            int dx = points[i].x - points[i - 1u].x;
            int dy = points[i].y - points[i - 1u].y;
            if (direction == ARC_SEQUENCE_DIRECTION_DIAGONAL_DOWN_RIGHT) {
                if (dx != step || dy != step) {
                    return 0;
                }
            } else if (dx != step || dy != -step) {
                return 0;
            }
        }
    }
    *point_count_out = point_count;
    *step_out = step;
    return 1;
}

static void arc_sequence_delta(int direction, int step, int *dx, int *dy)
{
    *dx = 0;
    *dy = 0;
    switch (direction) {
    case ARC_SEQUENCE_DIRECTION_HORIZONTAL:
        *dx = step;
        break;
    case ARC_SEQUENCE_DIRECTION_VERTICAL:
        *dy = step;
        break;
    case ARC_SEQUENCE_DIRECTION_DIAGONAL_DOWN_RIGHT:
        *dx = step;
        *dy = step;
        break;
    case ARC_SEQUENCE_DIRECTION_DIAGONAL_DOWN_LEFT:
        *dx = step;
        *dy = -step;
        break;
    default:
        break;
    }
}

static size_t arc_sequence_extend_from_point(const ARCSequenceHypothesisSpec *spec,
                                             const ARCGrid *input,
                                             ARCGrid *output,
                                             ARCSequencePoint point,
                                             int dx,
                                             int dy)
{
    size_t changed = 0;
    int x = point.x + dx;
    int y = point.y + dy;
    while (x >= 0 && y >= 0 &&
           x < (int)input->width && y < (int)input->height) {
        size_t cell = (size_t)y * input->width + (size_t)x;
        if (output->cells[cell] == spec->background) {
            output->cells[cell] = spec->output_color;
            ++changed;
        }
        x += dx;
        y += dy;
    }
    return changed;
}

static int arc_candidate_from_sequence_hypothesis(const NNHypothesis *hypothesis,
                                                  ARCTransformCandidate *candidate)
{
    ARCSequenceHypothesisSpec spec;
    if (candidate == NULL ||
        !arc_sequence_spec_from_hypothesis(hypothesis, &spec)) {
        return 0;
    }
    arc_init_candidate(candidate, ARC_RULE_HYPOTHESIS_APPLIED, "");
    candidate->background_color = spec.background;
    candidate->input_color = spec.input_color;
    candidate->output_color = spec.output_color;
    candidate->selector_kind = ARC_OBJECT_SELECTOR_ALL_COLOR;
    candidate->output_selector_kind = ARC_OBJECT_SELECTOR_UNIQUE_COLOR;
    candidate->dx = spec.direction;
    candidate->dy = spec.extend_mode;
    candidate->render_mode = ARC_OBJECT_RENDER_OVERLAY;
    if (!arc_store_hypothesis_candidate(candidate, hypothesis)) {
        return 0;
    }
    snprintf(candidate->name,
             sizeof(candidate->name),
             "hypothesis_%s_bg_%u_color_%u_to_%u_%s_%s",
             hypothesis->template_ref->name,
             (unsigned)spec.background,
             (unsigned)spec.input_color,
             (unsigned)spec.output_color,
             arc_sequence_direction_name(spec.direction),
             arc_sequence_extend_mode_name(spec.extend_mode));
    return 1;
}

static int arc_apply_sequence_hypothesis(const NNHypothesis *hypothesis,
                                         const ARCGrid *input,
                                         ARCGrid *output)
{
    ARCSequenceHypothesisSpec spec;
    ARCSequencePoint points[ARC_MAX_GRID_CELLS];
    size_t point_count = 0;
    int step = 0;
    if (!arc_sequence_spec_from_hypothesis(hypothesis, &spec) ||
        !arc_sequence_collect_points(input,
                                     spec.input_color,
                                     spec.direction,
                                     points,
                                     &point_count,
                                     &step)) {
        return 0;
    }
    int dx = 0;
    int dy = 0;
    arc_sequence_delta(spec.direction, step, &dx, &dy);
    if (dx == 0 && dy == 0) {
        return 0;
    }
    *output = *input;
    size_t changed = 0;
    if (spec.extend_mode == ARC_SEQUENCE_EXTEND_POSITIVE ||
        spec.extend_mode == ARC_SEQUENCE_EXTEND_BOTH) {
        changed += arc_sequence_extend_from_point(&spec,
                                                  input,
                                                  output,
                                                  points[point_count - 1u],
                                                  dx,
                                                  dy);
    }
    if (spec.extend_mode == ARC_SEQUENCE_EXTEND_NEGATIVE ||
        spec.extend_mode == ARC_SEQUENCE_EXTEND_BOTH) {
        changed += arc_sequence_extend_from_point(&spec,
                                                  input,
                                                  output,
                                                  points[0],
                                                  -dx,
                                                  -dy);
    }
    return changed > 0;
}

static int arc_apply_sketch_extend_aligned_sequence(
    const ARCSketchProgramSpec *spec,
    const ARCGrid *input,
    ARCGrid *output)
{
    ARCSequenceHypothesisSpec sequence;
    ARCSequencePoint points[ARC_MAX_GRID_CELLS];
    size_t point_count = 0;
    int step = 0;
    if (spec == NULL || input == NULL || output == NULL ||
        spec->color == spec->background ||
        spec->x_anchor < 0 ||
        spec->x_anchor >= ARC_SEQUENCE_DIRECTION_KIND_COUNT ||
        spec->y_anchor < 0 ||
        spec->y_anchor >= ARC_SEQUENCE_EXTEND_KIND_COUNT) {
        return 0;
    }
    memset(&sequence, 0, sizeof(sequence));
    sequence.background = spec->background;
    sequence.input_color = spec->color;
    sequence.output_color = spec->output_color;
    sequence.direction = spec->x_anchor;
    sequence.extend_mode = spec->y_anchor;
    if (!arc_sequence_collect_points(input,
                                     sequence.input_color,
                                     sequence.direction,
                                     points,
                                     &point_count,
                                     &step)) {
        return 0;
    }
    int dx = 0;
    int dy = 0;
    arc_sequence_delta(sequence.direction, step, &dx, &dy);
    if (dx == 0 && dy == 0) {
        return 0;
    }
    *output = *input;
    size_t changed = 0;
    if (sequence.extend_mode == ARC_SEQUENCE_EXTEND_POSITIVE ||
        sequence.extend_mode == ARC_SEQUENCE_EXTEND_BOTH) {
        changed += arc_sequence_extend_from_point(&sequence,
                                                  input,
                                                  output,
                                                  points[point_count - 1u],
                                                  dx,
                                                  dy);
    }
    if (sequence.extend_mode == ARC_SEQUENCE_EXTEND_NEGATIVE ||
        sequence.extend_mode == ARC_SEQUENCE_EXTEND_BOTH) {
        changed += arc_sequence_extend_from_point(&sequence,
                                                  input,
                                                  output,
                                                  points[0],
                                                  -dx,
                                                  -dy);
    }
    return changed > 0;
}

static const NNHypothesisTemplate *arc_hypothesis_template_tile_motif_repeat(void)
{
    static NNHypothesisTemplate template_ref;
    static int initialized = 0;
    if (initialized) {
        return &template_ref;
    }
    nn_hypothesis_template_init(&template_ref,
                                "arc_agi",
                                "tile_motif_repeat",
                                1031,
                                NN_PROGRAM_VALUE_GRID,
                                NN_PROGRAM_VALUE_GRID);
    nn_hypothesis_template_add_stage(&template_ref,
                                     "separator_lattice",
                                     NN_PROGRAM_KIND_TRANSFORM,
                                     NN_PROGRAM_VALUE_GRID,
                                     NN_PROGRAM_VALUE_OBJECT_SET,
                                     180);
    nn_hypothesis_template_add_stage(&template_ref,
                                     "select_motif_tile",
                                     NN_PROGRAM_KIND_SELECTOR,
                                     NN_PROGRAM_VALUE_OBJECT_SET,
                                     NN_PROGRAM_VALUE_OBJECT,
                                     181);
    nn_hypothesis_template_add_stage(&template_ref,
                                     "repeat_by_parity",
                                     NN_PROGRAM_KIND_TRANSFORM,
                                     NN_PROGRAM_VALUE_OBJECT,
                                     NN_PROGRAM_VALUE_GRID,
                                     182);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "lattice.background",
                                    NN_HYPOTHESIS_SLOT_COLOR,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "repeat.period",
                                    NN_HYPOTHESIS_SLOT_INTEGER,
                                    1, 2.0, 2);
    initialized = 1;
    return &template_ref;
}

typedef struct {
    uint8_t background;
    size_t period;
} ARCTileMotifHypothesisSpec;

static int arc_tile_motif_spec_from_hypothesis(const NNHypothesis *hypothesis,
                                               ARCTileMotifHypothesisSpec *spec)
{
    if (hypothesis == NULL || spec == NULL ||
        hypothesis->template_ref != arc_hypothesis_template_tile_motif_repeat() ||
        !nn_hypothesis_is_fully_bound(hypothesis)) {
        return 0;
    }
    int64_t background = 0;
    int64_t period = 0;
    if (!nn_hypothesis_get(hypothesis,
                           "lattice.background",
                           NULL,
                           &background) ||
        !nn_hypothesis_get(hypothesis, "repeat.period", NULL, &period)) {
        return 0;
    }
    if (background < 0 || background >= ARC_COLOR_COUNT ||
        period < 2 || period > ARC_MAX_GRID_WIDTH) {
        return 0;
    }
    spec->background = (uint8_t)background;
    spec->period = (size_t)period;
    return 1;
}

static void arc_tile_motif_candidate_from_spec(const ARCTileMotifHypothesisSpec *spec,
                                               ARCTransformCandidate *candidate)
{
    arc_init_candidate(candidate, ARC_RULE_TILE_MOTIF_PARITY_REPEAT, "");
    candidate->background_color = spec->background;
    candidate->selector_kind = ARC_OBJECT_SELECTOR_ALL_COLOR;
    candidate->output_selector_kind = ARC_OBJECT_SELECTOR_UNIQUE_COLOR;
    candidate->render_mode = ARC_OBJECT_RENDER_OVERLAY;
    candidate->scale = spec->period;
}

static int arc_candidate_from_tile_motif_hypothesis(const NNHypothesis *hypothesis,
                                                    ARCTransformCandidate *candidate)
{
    ARCTileMotifHypothesisSpec spec;
    if (candidate == NULL ||
        !arc_tile_motif_spec_from_hypothesis(hypothesis, &spec)) {
        return 0;
    }
    arc_tile_motif_candidate_from_spec(&spec, candidate);
    candidate->op = ARC_RULE_HYPOTHESIS_APPLIED;
    if (!arc_store_hypothesis_candidate(candidate, hypothesis)) {
        return 0;
    }
    snprintf(candidate->name,
             sizeof(candidate->name),
             "hypothesis_%s_bg_%u_period_%zu",
             hypothesis->template_ref->name,
             (unsigned)spec.background,
             spec.period);
    return 1;
}

static int arc_apply_tile_motif_hypothesis(const NNHypothesis *hypothesis,
                                           const ARCGrid *input,
                                           ARCGrid *output)
{
    ARCTileMotifHypothesisSpec spec;
    ARCTransformCandidate candidate;
    if (!arc_tile_motif_spec_from_hypothesis(hypothesis, &spec)) {
        return 0;
    }
    arc_tile_motif_candidate_from_spec(&spec, &candidate);
    return arc_apply_tile_motif_parity_repeat_candidate(&candidate, input, output);
}

static const NNHypothesisTemplate *arc_hypothesis_template_masked_motif_lift(void)
{
    static NNHypothesisTemplate template_ref;
    static int initialized = 0;
    if (initialized) {
        return &template_ref;
    }
    nn_hypothesis_template_init(&template_ref,
                                "arc_agi",
                                "masked_motif_lift",
                                1034,
                                NN_PROGRAM_VALUE_GRID,
                                NN_PROGRAM_VALUE_GRID);
    nn_hypothesis_template_add_stage(&template_ref,
                                     "bind_source_motif",
                                     NN_PROGRAM_KIND_SELECTOR,
                                     NN_PROGRAM_VALUE_GRID,
                                     NN_PROGRAM_VALUE_GRID,
                                     190);
    nn_hypothesis_template_add_stage(&template_ref,
                                     "detect_mask",
                                     NN_PROGRAM_KIND_TRANSFORM,
                                     NN_PROGRAM_VALUE_GRID,
                                     NN_PROGRAM_VALUE_OBJECT_SET,
                                     191);
    nn_hypothesis_template_add_stage(&template_ref,
                                     "render_masked_motif_grid",
                                     NN_PROGRAM_KIND_TRANSFORM,
                                     NN_PROGRAM_VALUE_OBJECT_SET,
                                     NN_PROGRAM_VALUE_GRID,
                                     192);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "mask.background",
                                    NN_HYPOTHESIS_SLOT_COLOR,
                                    1, 0.0, 0);
    initialized = 1;
    return &template_ref;
}

typedef struct {
    uint8_t background;
} ARCMaskedMotifHypothesisSpec;

static int arc_masked_motif_spec_from_hypothesis(const NNHypothesis *hypothesis,
                                                 ARCMaskedMotifHypothesisSpec *spec)
{
    if (hypothesis == NULL || spec == NULL ||
        hypothesis->template_ref != arc_hypothesis_template_masked_motif_lift() ||
        !nn_hypothesis_is_fully_bound(hypothesis)) {
        return 0;
    }
    int64_t background = 0;
    if (!nn_hypothesis_get(hypothesis,
                           "mask.background",
                           NULL,
                           &background) ||
        background < 0 || background >= ARC_COLOR_COUNT) {
        return 0;
    }
    spec->background = (uint8_t)background;
    return 1;
}

static int arc_apply_masked_motif_lift(const ARCMaskedMotifHypothesisSpec *spec,
                                       const ARCGrid *input,
                                       ARCGrid *output)
{
    if (spec == NULL || input == NULL || output == NULL ||
        input->width == 0 || input->height == 0 ||
        input->width * input->width > ARC_MAX_GRID_WIDTH ||
        input->height * input->height > ARC_MAX_GRID_HEIGHT) {
        return 0;
    }
    arc_init_grid(output, input->width * input->width, input->height * input->height);
    for (size_t cell = 0; cell < arc_grid_cells(output); ++cell) {
        output->cells[cell] = spec->background;
    }
    size_t foreground_cells = 0;
    for (size_t mask_y = 0; mask_y < input->height; ++mask_y) {
        for (size_t mask_x = 0; mask_x < input->width; ++mask_x) {
            uint8_t mask_value = input->cells[mask_y * input->width + mask_x];
            if (mask_value == spec->background) {
                continue;
            }
            ++foreground_cells;
            size_t origin_x = mask_x * input->width;
            size_t origin_y = mask_y * input->height;
            for (size_t motif_y = 0; motif_y < input->height; ++motif_y) {
                for (size_t motif_x = 0; motif_x < input->width; ++motif_x) {
                    size_t out_cell = (origin_y + motif_y) * output->width +
                                      origin_x + motif_x;
                    output->cells[out_cell] =
                        input->cells[motif_y * input->width + motif_x];
                }
            }
        }
    }
    return foreground_cells > 0;
}

static int arc_candidate_from_masked_motif_hypothesis(const NNHypothesis *hypothesis,
                                                      ARCTransformCandidate *candidate)
{
    ARCMaskedMotifHypothesisSpec spec;
    if (candidate == NULL ||
        !arc_masked_motif_spec_from_hypothesis(hypothesis, &spec)) {
        return 0;
    }
    arc_init_candidate(candidate, ARC_RULE_HYPOTHESIS_APPLIED, "");
    candidate->background_color = spec.background;
    candidate->selector_kind = ARC_OBJECT_SELECTOR_ALL_COLOR;
    candidate->output_selector_kind = ARC_OBJECT_SELECTOR_UNIQUE_COLOR;
    candidate->render_mode = ARC_OBJECT_RENDER_ONLY;
    if (!arc_store_hypothesis_candidate(candidate, hypothesis)) {
        return 0;
    }
    snprintf(candidate->name,
             sizeof(candidate->name),
             "hypothesis_%s_bg_%u",
             hypothesis->template_ref->name,
             (unsigned)spec.background);
    return 1;
}

static int arc_apply_masked_motif_hypothesis(const NNHypothesis *hypothesis,
                                             const ARCGrid *input,
                                             ARCGrid *output)
{
    ARCMaskedMotifHypothesisSpec spec;
    if (!arc_masked_motif_spec_from_hypothesis(hypothesis, &spec)) {
        return 0;
    }
    return arc_apply_masked_motif_lift(&spec, input, output);
}

static const NNHypothesisTemplate *arc_hypothesis_template_panel_selection(void)
{
    static NNHypothesisTemplate template_ref;
    static int initialized = 0;
    if (initialized) {
        return &template_ref;
    }
    nn_hypothesis_template_init(&template_ref,
                                "arc_agi",
                                "projection_panel_selection",
                                1032,
                                NN_PROGRAM_VALUE_GRID,
                                NN_PROGRAM_VALUE_GRID);
    nn_hypothesis_template_add_stage(&template_ref,
                                     "partition_panels",
                                     NN_PROGRAM_KIND_TRANSFORM,
                                     NN_PROGRAM_VALUE_GRID,
                                     NN_PROGRAM_VALUE_OBJECT_SET,
                                     180);
    nn_hypothesis_template_add_stage(&template_ref,
                                     "select_panel",
                                     NN_PROGRAM_KIND_SELECTOR,
                                     NN_PROGRAM_VALUE_OBJECT_SET,
                                     NN_PROGRAM_VALUE_GRID,
                                     181);
    nn_hypothesis_template_add_stage(&template_ref,
                                     "render_panel",
                                     NN_PROGRAM_KIND_TRANSFORM,
                                     NN_PROGRAM_VALUE_GRID,
                                     NN_PROGRAM_VALUE_GRID,
                                     182);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "panel.background",
                                    NN_HYPOTHESIS_SLOT_COLOR,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "panel.x_anchor",
                                    NN_HYPOTHESIS_SLOT_INTEGER,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "panel.y_anchor",
                                    NN_HYPOTHESIS_SLOT_INTEGER,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "panel.x_offset",
                                    NN_HYPOTHESIS_SLOT_OFFSET,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "panel.y_offset",
                                    NN_HYPOTHESIS_SLOT_OFFSET,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "panel.width",
                                    NN_HYPOTHESIS_SLOT_INTEGER,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "panel.height",
                                    NN_HYPOTHESIS_SLOT_INTEGER,
                                    1, 0.0, 0);
    initialized = 1;
    return &template_ref;
}

static const NNHypothesisTemplate *arc_hypothesis_template_bbox_panel_selection(void)
{
    static NNHypothesisTemplate template_ref;
    static int initialized = 0;
    if (initialized) {
        return &template_ref;
    }
    nn_hypothesis_template_init(&template_ref,
                                "arc_agi",
                                "projection_bbox_panel_selection",
                                1033,
                                NN_PROGRAM_VALUE_GRID,
                                NN_PROGRAM_VALUE_GRID);
    nn_hypothesis_template_add_stage(&template_ref,
                                     "detect_content_bbox",
                                     NN_PROGRAM_KIND_SELECTOR,
                                     NN_PROGRAM_VALUE_GRID,
                                     NN_PROGRAM_VALUE_OBJECT_SET,
                                     183);
    nn_hypothesis_template_add_stage(&template_ref,
                                     "select_bbox_panel",
                                     NN_PROGRAM_KIND_SELECTOR,
                                     NN_PROGRAM_VALUE_OBJECT_SET,
                                     NN_PROGRAM_VALUE_GRID,
                                     184);
    nn_hypothesis_template_add_stage(&template_ref,
                                     "render_panel",
                                     NN_PROGRAM_KIND_TRANSFORM,
                                     NN_PROGRAM_VALUE_GRID,
                                     NN_PROGRAM_VALUE_GRID,
                                     185);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "panel.background",
                                    NN_HYPOTHESIS_SLOT_COLOR,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "panel.x_anchor",
                                    NN_HYPOTHESIS_SLOT_INTEGER,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "panel.y_anchor",
                                    NN_HYPOTHESIS_SLOT_INTEGER,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "panel.x_offset",
                                    NN_HYPOTHESIS_SLOT_OFFSET,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "panel.y_offset",
                                    NN_HYPOTHESIS_SLOT_OFFSET,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "panel.width",
                                    NN_HYPOTHESIS_SLOT_INTEGER,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "panel.height",
                                    NN_HYPOTHESIS_SLOT_INTEGER,
                                    1, 0.0, 0);
    initialized = 1;
    return &template_ref;
}

typedef struct {
    uint8_t background;
    int x_anchor;
    int y_anchor;
    int x_offset;
    int y_offset;
    size_t width;
    size_t height;
} ARCPanelSelectionHypothesisSpec;

static int arc_panel_selection_spec_from_hypothesis(const NNHypothesis *hypothesis,
                                                    ARCPanelSelectionHypothesisSpec *spec)
{
    if (hypothesis == NULL || spec == NULL ||
        hypothesis->template_ref != arc_hypothesis_template_panel_selection() ||
        !nn_hypothesis_is_fully_bound(hypothesis)) {
        return 0;
    }
    int64_t background = 0;
    int64_t x_anchor = 0;
    int64_t y_anchor = 0;
    int64_t x_offset = 0;
    int64_t y_offset = 0;
    int64_t width = 0;
    int64_t height = 0;
    if (!nn_hypothesis_get(hypothesis,
                           "panel.background",
                           NULL,
                           &background) ||
        !nn_hypothesis_get(hypothesis, "panel.x_anchor", NULL, &x_anchor) ||
        !nn_hypothesis_get(hypothesis, "panel.y_anchor", NULL, &y_anchor) ||
        !nn_hypothesis_get(hypothesis, "panel.x_offset", NULL, &x_offset) ||
        !nn_hypothesis_get(hypothesis, "panel.y_offset", NULL, &y_offset) ||
        !nn_hypothesis_get(hypothesis, "panel.width", NULL, &width) ||
        !nn_hypothesis_get(hypothesis, "panel.height", NULL, &height)) {
        return 0;
    }
    if (background < 0 || background >= ARC_COLOR_COUNT ||
        x_anchor < 0 || x_anchor > 2 ||
        y_anchor < 0 || y_anchor > 2 ||
        x_offset < -ARC_MAX_GRID_WIDTH ||
        x_offset > ARC_MAX_GRID_WIDTH ||
        y_offset < -ARC_MAX_GRID_HEIGHT ||
        y_offset > ARC_MAX_GRID_HEIGHT ||
        width <= 0 || width > ARC_MAX_GRID_WIDTH ||
        height <= 0 || height > ARC_MAX_GRID_HEIGHT) {
        return 0;
    }
    spec->background = (uint8_t)background;
    spec->x_anchor = (int)x_anchor;
    spec->y_anchor = (int)y_anchor;
    spec->x_offset = (int)x_offset;
    spec->y_offset = (int)y_offset;
    spec->width = (size_t)width;
    spec->height = (size_t)height;
    return 1;
}

static int arc_bbox_panel_selection_spec_from_hypothesis(const NNHypothesis *hypothesis,
                                                         ARCPanelSelectionHypothesisSpec *spec)
{
    if (hypothesis == NULL || spec == NULL ||
        hypothesis->template_ref != arc_hypothesis_template_bbox_panel_selection() ||
        !nn_hypothesis_is_fully_bound(hypothesis)) {
        return 0;
    }
    int64_t background = 0;
    int64_t x_anchor = 0;
    int64_t y_anchor = 0;
    int64_t x_offset = 0;
    int64_t y_offset = 0;
    int64_t width = 0;
    int64_t height = 0;
    if (!nn_hypothesis_get(hypothesis,
                           "panel.background",
                           NULL,
                           &background) ||
        !nn_hypothesis_get(hypothesis, "panel.x_anchor", NULL, &x_anchor) ||
        !nn_hypothesis_get(hypothesis, "panel.y_anchor", NULL, &y_anchor) ||
        !nn_hypothesis_get(hypothesis, "panel.x_offset", NULL, &x_offset) ||
        !nn_hypothesis_get(hypothesis, "panel.y_offset", NULL, &y_offset) ||
        !nn_hypothesis_get(hypothesis, "panel.width", NULL, &width) ||
        !nn_hypothesis_get(hypothesis, "panel.height", NULL, &height)) {
        return 0;
    }
    if (background < 0 || background >= ARC_COLOR_COUNT ||
        x_anchor < 0 || x_anchor > 2 ||
        y_anchor < 0 || y_anchor > 2 ||
        x_offset < -ARC_MAX_GRID_WIDTH ||
        x_offset > ARC_MAX_GRID_WIDTH ||
        y_offset < -ARC_MAX_GRID_HEIGHT ||
        y_offset > ARC_MAX_GRID_HEIGHT ||
        width <= 0 || width > ARC_MAX_GRID_WIDTH ||
        height <= 0 || height > ARC_MAX_GRID_HEIGHT) {
        return 0;
    }
    spec->background = (uint8_t)background;
    spec->x_anchor = (int)x_anchor;
    spec->y_anchor = (int)y_anchor;
    spec->x_offset = (int)x_offset;
    spec->y_offset = (int)y_offset;
    spec->width = (size_t)width;
    spec->height = (size_t)height;
    return 1;
}

static int arc_candidate_from_panel_selection_hypothesis(const NNHypothesis *hypothesis,
                                                         ARCTransformCandidate *candidate)
{
    ARCPanelSelectionHypothesisSpec spec;
    if (candidate == NULL ||
        !arc_panel_selection_spec_from_hypothesis(hypothesis, &spec)) {
        return 0;
    }
    arc_init_candidate(candidate, ARC_RULE_HYPOTHESIS_APPLIED, "");
    candidate->background_color = spec.background;
    candidate->selector_kind = ARC_OBJECT_SELECTOR_ALL_COLOR;
    candidate->output_selector_kind = ARC_OBJECT_SELECTOR_UNIQUE_COLOR;
    candidate->dx = spec.x_offset;
    candidate->dy = spec.y_offset;
    candidate->scale = spec.width;
    if (!arc_store_hypothesis_candidate(candidate, hypothesis)) {
        return 0;
    }
    snprintf(candidate->name,
             sizeof(candidate->name),
             "hypothesis_%s_%s_%+d_%s_%+d_%zux%zu_bg_%u",
             hypothesis->template_ref->name,
             arc_panel_anchor_short_name(spec.x_anchor),
             spec.x_offset,
             arc_panel_anchor_short_name(spec.y_anchor),
             spec.y_offset,
             spec.width,
             spec.height,
             (unsigned)spec.background);
    return 1;
}

static int arc_candidate_from_bbox_panel_selection_hypothesis(const NNHypothesis *hypothesis,
                                                              ARCTransformCandidate *candidate)
{
    ARCPanelSelectionHypothesisSpec spec;
    if (candidate == NULL ||
        !arc_bbox_panel_selection_spec_from_hypothesis(hypothesis, &spec)) {
        return 0;
    }
    arc_init_candidate(candidate, ARC_RULE_HYPOTHESIS_APPLIED, "");
    candidate->background_color = spec.background;
    candidate->selector_kind = ARC_OBJECT_SELECTOR_ALL_COLOR;
    candidate->output_selector_kind = ARC_OBJECT_SELECTOR_UNIQUE_COLOR;
    candidate->dx = spec.x_offset;
    candidate->dy = spec.y_offset;
    candidate->scale = spec.width;
    if (!arc_store_hypothesis_candidate(candidate, hypothesis)) {
        return 0;
    }
    snprintf(candidate->name,
             sizeof(candidate->name),
             "hypothesis_%s_bbox_%s_%+d_%s_%+d_%zux%zu_bg_%u",
             hypothesis->template_ref->name,
             arc_panel_anchor_short_name(spec.x_anchor),
             spec.x_offset,
             arc_panel_anchor_short_name(spec.y_anchor),
             spec.y_offset,
             spec.width,
             spec.height,
             (unsigned)spec.background);
    return 1;
}

static int arc_apply_panel_selection_hypothesis(const NNHypothesis *hypothesis,
                                                const ARCGrid *input,
                                                ARCGrid *output)
{
    ARCPanelSelectionHypothesisSpec spec;
    size_t origin_x = 0;
    size_t origin_y = 0;
    if (!arc_panel_selection_spec_from_hypothesis(hypothesis, &spec) ||
        input == NULL || output == NULL ||
        !arc_panel_origin(input->width,
                          spec.width,
                          spec.x_anchor,
                          spec.x_offset,
                          &origin_x) ||
        !arc_panel_origin(input->height,
                          spec.height,
                          spec.y_anchor,
                          spec.y_offset,
                          &origin_y)) {
        return 0;
    }
    arc_init_grid(output, spec.width, spec.height);
    for (size_t y = 0; y < spec.height; ++y) {
        for (size_t x = 0; x < spec.width; ++x) {
            size_t input_cell = (origin_y + y) * input->width + origin_x + x;
            output->cells[y * output->width + x] = input->cells[input_cell];
        }
    }
    return 1;
}

static int arc_apply_bbox_panel_selection_hypothesis(const NNHypothesis *hypothesis,
                                                     const ARCGrid *input,
                                                     ARCGrid *output)
{
    ARCPanelSelectionHypothesisSpec spec;
    size_t origin_x = 0;
    size_t origin_y = 0;
    if (!arc_bbox_panel_selection_spec_from_hypothesis(hypothesis, &spec) ||
        input == NULL || output == NULL ||
        !arc_bbox_panel_origin(input,
                               spec.background,
                               spec.width,
                               spec.height,
                               spec.x_anchor,
                               spec.y_anchor,
                               spec.x_offset,
                               spec.y_offset,
                               &origin_x,
                               &origin_y)) {
        return 0;
    }
    arc_init_grid(output, spec.width, spec.height);
    for (size_t y = 0; y < spec.height; ++y) {
        for (size_t x = 0; x < spec.width; ++x) {
            size_t input_cell = (origin_y + y) * input->width + origin_x + x;
            output->cells[y * output->width + x] = input->cells[input_cell];
        }
    }
    return 1;
}

static const ARCHypothesisTemplateBinding *arc_hypothesis_template_registry(size_t *count)
{
    static const ARCHypothesisTemplateBinding bindings[] = {
        {
            "same_grid_color_map",
            1035,
            arc_hypothesis_template_same_grid_color_map,
            arc_induce_color_map_hypotheses,
            arc_candidate_from_color_map_hypothesis,
            arc_apply_color_map_hypothesis,
        },
        {
            "select_then_translate",
            1024,
            arc_hypothesis_template_object_translate,
            arc_induce_translate_hypotheses,
            arc_candidate_from_translate_hypothesis,
            arc_apply_translate_hypothesis,
        },
        {
            "move_to_anchor",
            1025,
            arc_hypothesis_template_move_to_anchor,
            arc_induce_move_to_anchor_hypotheses,
            arc_candidate_from_move_to_anchor_hypothesis,
            arc_apply_move_to_anchor_hypothesis,
        },
        {
            "recolor_by_reference",
            1026,
            arc_hypothesis_template_recolor_by_reference,
            arc_induce_recolor_hypotheses,
            arc_candidate_from_recolor_hypothesis,
            arc_apply_recolor_hypothesis,
        },
        {
            "connect_aligned",
            1027,
            arc_hypothesis_template_connect_aligned,
            arc_induce_connect_hypotheses,
            arc_candidate_from_connect_hypothesis,
            arc_apply_connect_hypothesis,
        },
        {
            "fill_enclosed",
            1028,
            arc_hypothesis_template_fill_enclosed,
            arc_induce_fill_hypotheses,
            arc_candidate_from_fill_hypothesis,
            arc_apply_fill_hypothesis,
        },
        {
            "object_extract",
            1029,
            arc_hypothesis_template_object_extract,
            arc_induce_extract_hypotheses,
            arc_candidate_from_extract_hypothesis,
            arc_apply_extract_hypothesis,
        },
        {
            "extend_aligned_sequence",
            1030,
            arc_hypothesis_template_extend_aligned_sequence,
            arc_induce_sequence_hypotheses,
            arc_candidate_from_sequence_hypothesis,
            arc_apply_sequence_hypothesis,
        },
        {
            "tile_motif_repeat",
            1031,
            arc_hypothesis_template_tile_motif_repeat,
            arc_induce_tile_motif_hypotheses,
            arc_candidate_from_tile_motif_hypothesis,
            arc_apply_tile_motif_hypothesis,
        },
        {
            "masked_motif_lift",
            1034,
            arc_hypothesis_template_masked_motif_lift,
            arc_induce_masked_motif_hypotheses,
            arc_candidate_from_masked_motif_hypothesis,
            arc_apply_masked_motif_hypothesis,
        },
        {
            "projection_panel_selection",
            1032,
            arc_hypothesis_template_panel_selection,
            arc_induce_panel_selection_hypotheses,
            arc_candidate_from_panel_selection_hypothesis,
            arc_apply_panel_selection_hypothesis,
        },
        {
            "projection_bbox_panel_selection",
            1033,
            arc_hypothesis_template_bbox_panel_selection,
            arc_induce_bbox_panel_selection_hypotheses,
            arc_candidate_from_bbox_panel_selection_hypothesis,
            arc_apply_bbox_panel_selection_hypothesis,
        },
        {
            "arc_sketch_program",
            1036,
            arc_hypothesis_template_sketch_program,
            arc_induce_sketch_program_hypotheses,
            arc_candidate_from_sketch_program_hypothesis,
            arc_apply_sketch_program_hypothesis,
        },
    };
    if (count != NULL) {
        *count = sizeof(bindings) / sizeof(bindings[0]);
    }
    return bindings;
}

static const ARCHypothesisTemplateBinding *arc_hypothesis_binding_for_template_id(uint32_t template_id)
{
    size_t count = 0;
    const ARCHypothesisTemplateBinding *bindings =
        arc_hypothesis_template_registry(&count);
    for (size_t i = 0; i < count; ++i) {
        if (bindings[i].template_id == template_id) {
            return &bindings[i];
        }
    }
    return NULL;
}

static int arc_apply_hypothesis_candidate(const ARCTransformCandidate *candidate,
                                          const ARCGrid *input,
                                          ARCGrid *output)
{
    NNHypothesis hypothesis;
    if (!arc_hypothesis_from_candidate(candidate, &hypothesis)) {
        return 0;
    }
    const ARCHypothesisTemplateBinding *binding =
        arc_hypothesis_binding_for_template_id(candidate->hypothesis_template_id);
    if (binding == NULL || binding->apply == NULL) {
        return 0;
    }
    return binding->apply(&hypothesis, input, output);
}

static void arc_emit_hypothesis_evidence_json(const char *task_id,
                                              const NNHypothesis *hypothesis,
                                              const NNProgram *program,
                                              const ARCProgramEval *eval,
                                              size_t test_cells)
{
    if (hypothesis == NULL || hypothesis->template_ref == NULL ||
        program == NULL) {
        return;
    }
    size_t events = eval != NULL && eval->test_cells != 0
                        ? eval->test_cells
                        : test_cells;
    size_t covered = eval != NULL ? eval->test_correct_cells : 0;
    NNHypothesisEvidence evidence = {
        .task = "arc_agi",
        .task_instance = task_id != NULL ? task_id : "unknown",
        .program = program,
        .certificate_status = eval != NULL && eval->train_consistent && eval->test_supported
                                  ? NN_CERTIFICATE_CERTIFIED
                                  : NN_CERTIFICATE_REJECTED,
        .proof_kind = eval != NULL && eval->train_consistent
                          ? "arc_train_pairs_consistent"
                          : "arc_hypothesis_unchecked",
        .reason = eval != NULL && eval->train_consistent
                      ? "hypothesis-derived transform matches train pairs"
                      : "hypothesis-derived transform did not certify on train pairs",
        .event_count = events,
        .covered_events = covered,
        .support = eval != NULL ? eval->train_checked_cells : 0,
        .false_rejections = eval != NULL ? eval->train_mismatches : 0,
        .utility = events == 0 ? 0.0 : (double)covered / (double)events,
    };
    nn_hypothesis_write_evidence_json(stdout, hypothesis, &evidence);
}

static size_t arc_emit_induction_smoke(const ARCTask *task,
                                       const ARCTransformCandidate *candidates,
                                       const ARCProgramEval *evals,
                                       size_t candidate_count)
{
    size_t test_cells = arc_task_test_cells(task);
    size_t emitted = 0;
    for (size_t i = 0; i < candidate_count && emitted < 8; ++i) {
        if (!arc_candidate_is_hypothesis_induced(&candidates[i])) {
            continue;
        }
        NNHypothesis hypothesis;
        NNProgram program;
        if (!arc_hypothesis_from_candidate(&candidates[i], &hypothesis) ||
            !nn_hypothesis_compile(&hypothesis, &program)) {
            continue;
        }
        arc_emit_hypothesis_evidence_json(task->id,
                                          &hypothesis,
                                          &program,
                                          &evals[i],
                                          test_cells);
        ++emitted;
    }
    return emitted;
}

static void arc_emit_builtin_program_summaries(const ARCTask *task, const ARCConfig *config)
{
    ARCTransformCandidate candidates[ARC_RULE_CANDIDATE_COUNT];
    ARCProgramEval evals[ARC_RULE_CANDIDATE_COUNT];
    ARCProgramLearnedState learned;
    ARCProgramProposerAggregate aggregate;
    size_t budgets[ARC_PROGRAM_BUDGET_GRID_COUNT];
    size_t budget_count = arc_program_budget_list(config,
                                                  budgets,
                                                  ARC_PROGRAM_BUDGET_GRID_COUNT);
    if (!arc_program_learned_init(&learned,
                                  config,
                                  (unsigned int)((config == NULL ? 20240516u : config->seed) &
                                                 UINT32_MAX))) {
        return;
    }
    size_t candidate_count = arc_build_candidate_set(task,
                                                     &learned,
                                                     candidates,
                                                     ARC_RULE_CANDIDATE_COUNT,
                                                     config != NULL && config->induction_smoke,
                                                     NULL);
    arc_evaluate_program_candidates(task, candidates, candidate_count, evals);
    for (size_t budget_index = 0; budget_index < budget_count; ++budget_index) {
        for (size_t strategy = 0; strategy < ARC_PROGRAM_STRATEGY_COUNT; ++strategy) {
            arc_program_aggregate_init(&aggregate,
                                       task->id,
                                       (ARCProgramStrategy)strategy,
                                       budgets[budget_index]);
            arc_program_aggregate_task(&aggregate,
                                       task,
                                       candidates,
                                       evals,
                                       candidate_count,
                                       (ARCProgramStrategy)strategy,
                                       &learned,
                                       NULL,
                                       budgets[budget_index]);
            arc_program_print_aggregate(&aggregate);
        }
    }
    if (config != NULL && config->induction_smoke) {
        arc_emit_induction_smoke(task, candidates, evals, candidate_count);
    }
    arc_program_learned_free(&learned);
}
