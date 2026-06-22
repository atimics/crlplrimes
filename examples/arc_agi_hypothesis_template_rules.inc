/* Internal ARC search, transform, hypothesis-template, and program-summary helpers for arc_agi.c. */
static void arc_color_map_slot_name(uint8_t color, char *name, size_t name_size);
static int arc_candidate_from_translate_hypothesis(const NNHypothesis *hypothesis,
                                                   ARCTransformCandidate *candidate);
static int arc_candidate_from_move_to_anchor_hypothesis(const NNHypothesis *hypothesis,
                                                       ARCTransformCandidate *candidate);
static int arc_candidate_from_recolor_hypothesis(const NNHypothesis *hypothesis,
                                                 ARCTransformCandidate *candidate);
static int arc_candidate_from_connect_hypothesis(const NNHypothesis *hypothesis,
                                                 ARCTransformCandidate *candidate);
static int arc_candidate_from_fill_hypothesis(const NNHypothesis *hypothesis,
                                              ARCTransformCandidate *candidate);
static int arc_candidate_from_extract_hypothesis(const NNHypothesis *hypothesis,
                                                 ARCTransformCandidate *candidate);
static int arc_candidate_from_sequence_hypothesis(const NNHypothesis *hypothesis,
                                                  ARCTransformCandidate *candidate);
static int arc_candidate_from_tile_motif_hypothesis(const NNHypothesis *hypothesis,
                                                    ARCTransformCandidate *candidate);
static int arc_candidate_from_masked_motif_hypothesis(const NNHypothesis *hypothesis,
                                                      ARCTransformCandidate *candidate);
static int arc_candidate_from_panel_selection_hypothesis(const NNHypothesis *hypothesis,
                                                         ARCTransformCandidate *candidate);
static int arc_candidate_from_bbox_panel_selection_hypothesis(const NNHypothesis *hypothesis,
                                                              ARCTransformCandidate *candidate);
static int arc_candidate_from_sketch_program_hypothesis(const NNHypothesis *hypothesis,
                                                        ARCTransformCandidate *candidate);
static int arc_apply_color_map_hypothesis(const NNHypothesis *hypothesis,
                                          const ARCGrid *input,
                                          ARCGrid *output);
static int arc_apply_translate_hypothesis(const NNHypothesis *hypothesis,
                                          const ARCGrid *input,
                                          ARCGrid *output);
static int arc_apply_move_to_anchor_hypothesis(const NNHypothesis *hypothesis,
                                               const ARCGrid *input,
                                               ARCGrid *output);
static int arc_apply_recolor_hypothesis(const NNHypothesis *hypothesis,
                                        const ARCGrid *input,
                                        ARCGrid *output);
static int arc_apply_connect_hypothesis(const NNHypothesis *hypothesis,
                                        const ARCGrid *input,
                                        ARCGrid *output);
static int arc_apply_fill_hypothesis(const NNHypothesis *hypothesis,
                                     const ARCGrid *input,
                                     ARCGrid *output);
static int arc_apply_extract_hypothesis(const NNHypothesis *hypothesis,
                                        const ARCGrid *input,
                                        ARCGrid *output);
static int arc_apply_sequence_hypothesis(const NNHypothesis *hypothesis,
                                         const ARCGrid *input,
                                         ARCGrid *output);
static int arc_apply_tile_motif_hypothesis(const NNHypothesis *hypothesis,
                                           const ARCGrid *input,
                                           ARCGrid *output);
static int arc_apply_masked_motif_hypothesis(const NNHypothesis *hypothesis,
                                             const ARCGrid *input,
                                             ARCGrid *output);
static int arc_apply_panel_selection_hypothesis(const NNHypothesis *hypothesis,
                                                const ARCGrid *input,
                                                ARCGrid *output);
static int arc_apply_bbox_panel_selection_hypothesis(const NNHypothesis *hypothesis,
                                                     const ARCGrid *input,
                                                     ARCGrid *output);
static int arc_apply_sketch_program_hypothesis(const NNHypothesis *hypothesis,
                                               const ARCGrid *input,
                                               ARCGrid *output);
static int arc_apply_sketch_extend_aligned_sequence(
    const ARCSketchProgramSpec *spec,
    const ARCGrid *input,
    ARCGrid *output);
static size_t arc_induce_color_map_hypotheses(const ARCHypothesisTemplateBinding *binding,
                                              const ARCTask *task,
                                              ARCProgramLearnedState *learned,
                                              const ARCTransformCandidate *existing_candidates,
                                              size_t existing_count,
                                              ARCHypothesisProposal *proposals,
                                              size_t proposal_capacity,
                                              size_t proposal_count,
                                              ARCHypothesisSelectionStats *stats,
                                              size_t template_index);
static size_t arc_induce_translate_hypotheses(const ARCHypothesisTemplateBinding *binding,
                                              const ARCTask *task,
                                              ARCProgramLearnedState *learned,
                                              const ARCTransformCandidate *existing_candidates,
                                              size_t existing_count,
                                              ARCHypothesisProposal *proposals,
                                              size_t proposal_capacity,
                                              size_t proposal_count,
                                              ARCHypothesisSelectionStats *stats,
                                              size_t template_index);
static size_t arc_induce_move_to_anchor_hypotheses(const ARCHypothesisTemplateBinding *binding,
                                                   const ARCTask *task,
                                                   ARCProgramLearnedState *learned,
                                                   const ARCTransformCandidate *existing_candidates,
                                                   size_t existing_count,
                                                   ARCHypothesisProposal *proposals,
                                                   size_t proposal_capacity,
                                                   size_t proposal_count,
                                                   ARCHypothesisSelectionStats *stats,
                                                   size_t template_index);
static size_t arc_induce_recolor_hypotheses(const ARCHypothesisTemplateBinding *binding,
                                            const ARCTask *task,
                                            ARCProgramLearnedState *learned,
                                            const ARCTransformCandidate *existing_candidates,
                                            size_t existing_count,
                                            ARCHypothesisProposal *proposals,
                                            size_t proposal_capacity,
                                            size_t proposal_count,
                                            ARCHypothesisSelectionStats *stats,
                                            size_t template_index);
static size_t arc_induce_connect_hypotheses(const ARCHypothesisTemplateBinding *binding,
                                            const ARCTask *task,
                                            ARCProgramLearnedState *learned,
                                            const ARCTransformCandidate *existing_candidates,
                                            size_t existing_count,
                                            ARCHypothesisProposal *proposals,
                                            size_t proposal_capacity,
                                            size_t proposal_count,
                                            ARCHypothesisSelectionStats *stats,
                                            size_t template_index);
static size_t arc_induce_fill_hypotheses(const ARCHypothesisTemplateBinding *binding,
                                         const ARCTask *task,
                                         ARCProgramLearnedState *learned,
                                         const ARCTransformCandidate *existing_candidates,
                                         size_t existing_count,
                                         ARCHypothesisProposal *proposals,
                                         size_t proposal_capacity,
                                         size_t proposal_count,
                                         ARCHypothesisSelectionStats *stats,
                                         size_t template_index);
static size_t arc_induce_extract_hypotheses(const ARCHypothesisTemplateBinding *binding,
                                            const ARCTask *task,
                                            ARCProgramLearnedState *learned,
                                            const ARCTransformCandidate *existing_candidates,
                                            size_t existing_count,
                                            ARCHypothesisProposal *proposals,
                                            size_t proposal_capacity,
                                            size_t proposal_count,
                                            ARCHypothesisSelectionStats *stats,
                                            size_t template_index);
static size_t arc_induce_sequence_hypotheses(const ARCHypothesisTemplateBinding *binding,
                                             const ARCTask *task,
                                             ARCProgramLearnedState *learned,
                                             const ARCTransformCandidate *existing_candidates,
                                             size_t existing_count,
                                             ARCHypothesisProposal *proposals,
                                             size_t proposal_capacity,
                                             size_t proposal_count,
                                             ARCHypothesisSelectionStats *stats,
                                             size_t template_index);
static size_t arc_induce_tile_motif_hypotheses(const ARCHypothesisTemplateBinding *binding,
                                               const ARCTask *task,
                                               ARCProgramLearnedState *learned,
                                               const ARCTransformCandidate *existing_candidates,
                                               size_t existing_count,
                                               ARCHypothesisProposal *proposals,
                                               size_t proposal_capacity,
                                               size_t proposal_count,
                                               ARCHypothesisSelectionStats *stats,
                                               size_t template_index);
static size_t arc_induce_masked_motif_hypotheses(const ARCHypothesisTemplateBinding *binding,
                                                 const ARCTask *task,
                                                 ARCProgramLearnedState *learned,
                                                 const ARCTransformCandidate *existing_candidates,
                                                 size_t existing_count,
                                                 ARCHypothesisProposal *proposals,
                                                 size_t proposal_capacity,
                                                 size_t proposal_count,
                                                 ARCHypothesisSelectionStats *stats,
                                                 size_t template_index);
static size_t arc_induce_panel_selection_hypotheses(const ARCHypothesisTemplateBinding *binding,
                                                    const ARCTask *task,
                                                    ARCProgramLearnedState *learned,
                                                    const ARCTransformCandidate *existing_candidates,
                                                    size_t existing_count,
                                                    ARCHypothesisProposal *proposals,
                                                    size_t proposal_capacity,
                                                    size_t proposal_count,
                                                    ARCHypothesisSelectionStats *stats,
                                                    size_t template_index);
static size_t arc_induce_bbox_panel_selection_hypotheses(const ARCHypothesisTemplateBinding *binding,
                                                         const ARCTask *task,
                                                         ARCProgramLearnedState *learned,
                                                         const ARCTransformCandidate *existing_candidates,
                                                         size_t existing_count,
                                                         ARCHypothesisProposal *proposals,
                                                         size_t proposal_capacity,
                                                         size_t proposal_count,
                                                         ARCHypothesisSelectionStats *stats,
                                                         size_t template_index);
static size_t arc_induce_sketch_program_hypotheses(const ARCHypothesisTemplateBinding *binding,
                                                   const ARCTask *task,
                                                   ARCProgramLearnedState *learned,
                                                   const ARCTransformCandidate *existing_candidates,
                                                   size_t existing_count,
                                                   ARCHypothesisProposal *proposals,
                                                   size_t proposal_capacity,
                                                   size_t proposal_count,
                                                   ARCHypothesisSelectionStats *stats,
                                                   size_t template_index);
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
    const ARCProgramTaskFeatures *task_features);

static double safe_probability(double probability)
{
    return probability < 1e-12 ? 1e-12 : probability;
}

static void arc_init_grid(ARCGrid *grid, size_t width, size_t height)
{
    memset(grid, 0, sizeof(*grid));
    grid->width = width;
    grid->height = height;
}

static size_t arc_grid_cells(const ARCGrid *grid)
{
    return grid->width * grid->height;
}

static int arc_grid_same_shape(const ARCGrid *lhs, const ARCGrid *rhs)
{
    return lhs->width == rhs->width && lhs->height == rhs->height;
}

static int arc_grid_equal(const ARCGrid *lhs, const ARCGrid *rhs)
{
    size_t cells = arc_grid_cells(lhs);
    return arc_grid_same_shape(lhs, rhs) &&
           memcmp(lhs->cells, rhs->cells, cells * sizeof(lhs->cells[0])) == 0;
}

static void init_builtin_arc_task(ARCTask *task)
{
    memset(task, 0, sizeof(*task));
    strcpy(task->id, "0520fde7");
    task->train_count = ARC_TRAIN_PAIR_COUNT;
    task->test_count = 1;
    for (size_t pair = 0; pair < ARC_TRAIN_PAIR_COUNT; ++pair) {
        arc_init_grid(&task->train[pair].input, ARC_INPUT_WIDTH, ARC_INPUT_HEIGHT);
        memcpy(task->train[pair].input.cells,
               ARC_0520FDE7_TRAIN_INPUTS[pair],
               sizeof(ARC_0520FDE7_TRAIN_INPUTS[pair]));
        arc_init_grid(&task->train[pair].output, ARC_OUTPUT_WIDTH, ARC_OUTPUT_HEIGHT);
        memcpy(task->train[pair].output.cells,
               ARC_0520FDE7_TRAIN_OUTPUTS[pair],
               sizeof(ARC_0520FDE7_TRAIN_OUTPUTS[pair]));
        task->train[pair].has_output = 1;
    }
    arc_init_grid(&task->test[0].input, ARC_INPUT_WIDTH, ARC_INPUT_HEIGHT);
    memcpy(task->test[0].input.cells, ARC_0520FDE7_TEST_INPUT, sizeof(ARC_0520FDE7_TEST_INPUT));
    arc_init_grid(&task->test[0].output, ARC_OUTPUT_WIDTH, ARC_OUTPUT_HEIGHT);
    memcpy(task->test[0].output.cells, ARC_0520FDE7_TEST_OUTPUT, sizeof(ARC_0520FDE7_TEST_OUTPUT));
    task->test[0].has_output = 1;
}

static void arc_init_candidate(ARCTransformCandidate *candidate, ARCTransformRule op, const char *name)
{
    memset(candidate, 0, sizeof(*candidate));
    candidate->op = op;
    for (uint8_t color = 0; color < ARC_COLOR_COUNT; ++color) {
        candidate->color_map[color] = color;
    }
    snprintf(candidate->name, sizeof(candidate->name), "%s", name);
}

static int arc_store_hypothesis_candidate(ARCTransformCandidate *candidate,
                                          const NNHypothesis *hypothesis)
{
    if (candidate == NULL || hypothesis == NULL ||
        hypothesis->template_ref == NULL ||
        !nn_hypothesis_is_fully_bound(hypothesis) ||
        hypothesis->slot_count > NN_HYPOTHESIS_MAX_SLOTS) {
        return 0;
    }
    candidate->op = ARC_RULE_HYPOTHESIS_APPLIED;
    candidate->hypothesis_template_id = hypothesis->template_ref->template_id;
    candidate->hypothesis_canonical_hash = hypothesis->canonical_hash;
    candidate->hypothesis_slot_count = (uint8_t)hypothesis->slot_count;
    for (size_t i = 0; i < hypothesis->slot_count; ++i) {
        candidate->hypothesis_slot_integers[i] = hypothesis->slots[i].integer;
    }
    return 1;
}

static int arc_hypothesis_from_candidate(const ARCTransformCandidate *candidate,
                                         NNHypothesis *hypothesis)
{
    if (candidate == NULL || hypothesis == NULL ||
        candidate->op != ARC_RULE_HYPOTHESIS_APPLIED) {
        return 0;
    }
    const ARCHypothesisTemplateBinding *binding =
        arc_hypothesis_binding_for_template_id(candidate->hypothesis_template_id);
    if (binding == NULL || binding->template_fn == NULL ||
        !nn_hypothesis_init(hypothesis, binding->template_fn()) ||
        candidate->hypothesis_slot_count != hypothesis->slot_count) {
        return 0;
    }
    for (size_t i = 0; i < hypothesis->slot_count; ++i) {
        const NNHypothesisSlotSpec *slot = &hypothesis->template_ref->slots[i];
        int64_t integer = candidate->hypothesis_slot_integers[i];
        if (!nn_hypothesis_bind(hypothesis,
                                slot->name,
                                (double)integer,
                                integer)) {
            return 0;
        }
    }
    return hypothesis->canonical_hash == candidate->hypothesis_canonical_hash;
}

static int arc_candidate_is_composed(const ARCTransformCandidate *candidate)
{
    return candidate != NULL &&
           candidate->op == ARC_RULE_COMPOSED &&
           candidate->step_count == 2;
}

static void arc_step_from_candidate(const ARCTransformCandidate *candidate, ARCTransformStep *step)
{
    memset(step, 0, sizeof(*step));
    step->op = candidate->op;
    step->input_color = candidate->input_color;
    step->output_color = candidate->output_color;
    step->background_color = candidate->background_color;
    step->selector_kind = candidate->selector_kind;
    step->output_selector_kind = candidate->output_selector_kind;
    step->reference_color = candidate->reference_color;
    step->render_mode = candidate->render_mode;
    memcpy(step->color_map, candidate->color_map, sizeof(step->color_map));
    step->dx = candidate->dx;
    step->dy = candidate->dy;
    step->scale = candidate->scale;
    snprintf(step->name, sizeof(step->name), "%s", candidate->name);
}

static void arc_candidate_from_step(const ARCTransformStep *step, ARCTransformCandidate *candidate)
{
    arc_init_candidate(candidate, step->op, step->name);
    candidate->input_color = step->input_color;
    candidate->output_color = step->output_color;
    candidate->background_color = step->background_color;
    candidate->selector_kind = step->selector_kind;
    candidate->output_selector_kind = step->output_selector_kind;
    candidate->reference_color = step->reference_color;
    candidate->render_mode = step->render_mode;
    memcpy(candidate->color_map, step->color_map, sizeof(candidate->color_map));
    candidate->dx = step->dx;
    candidate->dy = step->dy;
    candidate->scale = step->scale;
}

static int arc_candidate_is_object_relation(const ARCTransformCandidate *candidate)
{
    return candidate != NULL &&
           (candidate->op == ARC_RULE_OBJECT_RELATION_INDUCED ||
            candidate->op == ARC_RULE_OBJECT_TRANSLATE ||
            candidate->op == ARC_RULE_OBJECT_CONNECT_ALIGNED ||
            candidate->op == ARC_RULE_OBJECT_FILL_ENCLOSED);
}

static int arc_candidate_is_layer_program(const ARCTransformCandidate *candidate)
{
    return candidate != NULL &&
           (candidate->op == ARC_RULE_LAYER_OVERLAP ||
            candidate->op == ARC_RULE_LAYER_UNION ||
            candidate->op == ARC_RULE_LAYER_XOR ||
            candidate->op == ARC_RULE_LAYER_LEFT_ONLY ||
            candidate->op == ARC_RULE_LAYER_RIGHT_ONLY);
}

static int arc_layer_relation_value(ARCTransformRule op, int left_on, int right_on)
{
    switch (op) {
    case ARC_RULE_LAYER_OVERLAP:
        return left_on && right_on;
    case ARC_RULE_LAYER_UNION:
        return left_on || right_on;
    case ARC_RULE_LAYER_XOR:
        return left_on != right_on;
    case ARC_RULE_LAYER_LEFT_ONLY:
        return left_on && !right_on;
    case ARC_RULE_LAYER_RIGHT_ONLY:
        return right_on && !left_on;
    default:
        return 0;
    }
}

static const char *arc_layer_relation_name(ARCTransformRule op)
{
    switch (op) {
    case ARC_RULE_LAYER_OVERLAP:
        return "overlap";
    case ARC_RULE_LAYER_UNION:
        return "union";
    case ARC_RULE_LAYER_XOR:
        return "xor";
    case ARC_RULE_LAYER_LEFT_ONLY:
        return "left_only";
    case ARC_RULE_LAYER_RIGHT_ONLY:
        return "right_only";
    default:
        return "unknown";
    }
}

static size_t arc_object_relation_kind(const ARCTransformCandidate *candidate)
{
    if (candidate == NULL) {
        return 0;
    }
    if (candidate->op == ARC_RULE_OBJECT_RELATION_INDUCED) {
        return candidate->scale;
    }
    if (candidate->op == ARC_RULE_OBJECT_TRANSLATE) {
        return ARC_OBJECT_RELATION_TRANSLATE;
    }
    if (candidate->op == ARC_RULE_OBJECT_CONNECT_ALIGNED) {
        return ARC_OBJECT_RELATION_CONNECT_ALIGNED;
    }
    if (candidate->op == ARC_RULE_OBJECT_FILL_ENCLOSED) {
        return ARC_OBJECT_RELATION_FILL_ENCLOSED;
    }
    return 0;
}

static const char *arc_object_relation_kind_name(size_t kind)
{
    switch (kind) {
    case ARC_OBJECT_RELATION_TRANSLATE:
        return "translate";
    case ARC_OBJECT_RELATION_CONNECT_ALIGNED:
        return "connect_aligned";
    case ARC_OBJECT_RELATION_FILL_ENCLOSED:
        return "fill_enclosed";
    }
    return "unknown";
}

static const char *arc_object_selector_short_name(uint8_t selector_kind)
{
    switch (selector_kind) {
    case ARC_OBJECT_SELECTOR_UNIQUE_COLOR:
        return "uniq";
    case ARC_OBJECT_SELECTOR_LARGEST_COLOR:
        return "max";
    case ARC_OBJECT_SELECTOR_SMALLEST_COLOR:
        return "min";
    case ARC_OBJECT_SELECTOR_INTERIOR_COLOR:
        return "inner";
    case ARC_OBJECT_SELECTOR_TOUCHING_BORDER_COLOR:
        return "edge";
    case ARC_OBJECT_SELECTOR_NEAREST_TO_COLOR:
        return "near";
    case ARC_OBJECT_SELECTOR_ALL_COLOR:
    default:
        return "all";
    }
}

static uint8_t arc_candidate_selector_kind(const ARCTransformCandidate *candidate)
{
    if (candidate == NULL) {
        return ARC_OBJECT_SELECTOR_ALL_COLOR;
    }
    if (candidate->selector_kind != 0) {
        return candidate->selector_kind;
    }
    if (candidate->op == ARC_RULE_OBJECT_TRANSLATE ||
        arc_object_relation_kind(candidate) == ARC_OBJECT_RELATION_TRANSLATE ||
        candidate->op == ARC_RULE_OBJECT_FILL_ENCLOSED ||
        arc_object_relation_kind(candidate) == ARC_OBJECT_RELATION_FILL_ENCLOSED) {
        return ARC_OBJECT_SELECTOR_UNIQUE_COLOR;
    }
    return ARC_OBJECT_SELECTOR_ALL_COLOR;
}

static uint8_t arc_candidate_output_selector_kind(const ARCTransformCandidate *candidate)
{
    if (candidate == NULL || candidate->output_selector_kind == 0) {
        return ARC_OBJECT_SELECTOR_UNIQUE_COLOR;
    }
    return candidate->output_selector_kind;
}

static uint8_t arc_candidate_render_mode(const ARCTransformCandidate *candidate)
{
    if (candidate == NULL ||
        candidate->render_mode >= ARC_OBJECT_RENDER_KIND_COUNT) {
        return ARC_OBJECT_RENDER_OVERLAY;
    }
    return candidate->render_mode;
}

static int arc_candidate_is_hypothesis_induced(const ARCTransformCandidate *candidate);
/* Unified operator registry: single struct replaces 4 separate tables.
   Adding a new operator: enum + 1 line here + dispatch case + apply fn. */
static const struct {
    int composable;   /* can appear as first in a composition */
    int can_end;      /* can appear as last in a composition */
    size_t family;    /* operator family for learned scoring */
    double score;     /* structural program score */
} arc_op_info[ARC_RULE_COUNT] = {
    [ARC_RULE_COPY_LEFT]                = {0,0,6,102.0},
    [ARC_RULE_COPY_RIGHT]               = {0,0,6,102.0},
    [ARC_RULE_INTERSECTION]             = {0,0,6,104.0},
    [ARC_RULE_UNION]                    = {0,0,6,104.0},
    [ARC_RULE_OBJECT_SELECT]            = {0,0,6,104.0},
    [ARC_RULE_COLOR_MAP_LEFT]           = {0,0,6,102.0},
    [ARC_RULE_GRID_IDENTITY]            = {0,1,9,68.0},
    [ARC_RULE_GRID_COLOR_MAP]           = {1,1,1,120.0},
    [ARC_RULE_GRID_FOREGROUND_RECOLOR]  = {1,1,10,62.0},
    [ARC_RULE_CROP_BACKGROUND]          = {1,1,4,100.0},
    [ARC_RULE_CROP_BACKGROUND_RECOLOR]  = {0,1,4,84.0},
    [ARC_RULE_CROP_COLOR]               = {1,1,4,100.0},
    [ARC_RULE_CROP_COLOR_MASK]          = {1,1,4,84.0},
    [ARC_RULE_ROTATE90]                 = {1,1,2,110.0},
    [ARC_RULE_ROTATE180]                = {1,1,2,110.0},
    [ARC_RULE_ROTATE270]                = {1,1,2,110.0},
    [ARC_RULE_FLIP_HORIZONTAL]          = {1,1,2,110.0},
    [ARC_RULE_FLIP_VERTICAL]            = {1,1,2,110.0},
    [ARC_RULE_TRANSPOSE]                = {1,1,2,110.0},
    [ARC_RULE_SCALE]                    = {1,1,3,105.0},
    [ARC_RULE_BBOX_OUTLINE]             = {1,1,7,78.0},
    [ARC_RULE_BBOX_FILL]                = {1,1,7,78.0},
    [ARC_RULE_TRANSLATE]                = {1,1,8,72.0},
    [ARC_RULE_OBJECT_CROP_LARGEST]      = {0,1,5,96.0},
    [ARC_RULE_OBJECT_CROP_LARGEST_RECOLOR]={0,1,5,96.0},
    [ARC_RULE_OBJECT_MASK_LARGEST]      = {0,1,5,96.0},
    [ARC_RULE_COMPOSED]                 = {0,0,11,-40.0},
    [ARC_RULE_OBJECT_TRANSLATE]         = {0,1,12,98.0},
    [ARC_RULE_OBJECT_CONNECT_ALIGNED]   = {0,1,13,99.0},
    [ARC_RULE_OBJECT_FILL_ENCLOSED]     = {0,1,14,97.0},
    [ARC_RULE_OBJECT_RELATION_INDUCED]  = {0,1,12,90.0},
    [ARC_RULE_HYPOTHESIS_APPLIED]       = {0,0,15,130.0},
    [ARC_RULE_LAYER_OVERLAP]            = {0,0,6,104.0},
    [ARC_RULE_LAYER_UNION]              = {0,0,6,104.0},
    [ARC_RULE_LAYER_XOR]                = {0,0,6,104.0},
    [ARC_RULE_LAYER_LEFT_ONLY]          = {0,0,6,104.0},
    [ARC_RULE_LAYER_RIGHT_ONLY]         = {0,0,6,104.0},
    [ARC_RULE_COMPRESS_COLOR_LINES]     = {1,1,4,100.0},
    [ARC_RULE_TILE_MOTIF_PARITY_REPEAT] = {0,1,16,106.0},
    [ARC_RULE_GRAVITY_DOWN]             = {1,1,17,98.0},
    [ARC_RULE_GRAVITY_UP]               = {1,1,17,98.0},
    [ARC_RULE_GRAVITY_LEFT]             = {1,1,17,98.0},
    [ARC_RULE_GRAVITY_RIGHT]            = {1,1,17,98.0},
    [ARC_RULE_TILE_REPEAT]              = {1,1,18,104.0},
    [ARC_RULE_GRID_REPEAT]              = {1,1,19,106.0},
    [ARC_RULE_EXTRACT_COLOR]            = {1,1,20,92.0},
    [ARC_RULE_ERASE_COLOR]              = {1,1,20,90.0},
    [ARC_RULE_RECOLOR_BY_REFERENCE]     = {1,1,21,100.0},
    [ARC_RULE_PATTERN_EXTEND]           = {1,1,22,102.0},
    [ARC_RULE_DRAW_LINE]                = {1,1,23,96.0},
    [ARC_RULE_SYMMETRY_H]               = {1,1,24,100.0},
    [ARC_RULE_SYMMETRY_V]               = {1,1,24,100.0},
    [ARC_RULE_MULTISCALE]               = {1,0,25,110.0},
    [ARC_RULE_OBJECT_LABEL]             = {1,0,26,94.0},
    [ARC_RULE_PERIOD_TILE]              = {1,0,22,108.0},
    [ARC_RULE_DILATE]                   = {1,1,27,95.0},
    [ARC_RULE_REFLECT_TILE]             = {1,1,28,112.0},
    [ARC_RULE_MARKER_MOTIF_TILE]        = {1,1,29,111.0},
    [ARC_RULE_AXIS_TILE]                = {1,1,30,109.0},
    [ARC_RULE_PERIOD_EXTEND]            = {1,1,31,108.0},
};

static int arc_rule_is_composable(ARCTransformRule op) {
    return op < ARC_RULE_COUNT ? arc_op_info[op].composable : 0;
}

static int arc_rule_can_end_composition(ARCTransformRule op) {
    return op < ARC_RULE_COUNT ? arc_op_info[op].can_end : 0;
}

static int arc_init_composed_candidate(ARCTransformCandidate *candidate,
                                       const ARCTransformCandidate *first,
                                       const ARCTransformCandidate *second)
{
    if (candidate == NULL || first == NULL || second == NULL ||
        arc_candidate_is_composed(first) || arc_candidate_is_composed(second) ||
        !arc_rule_can_end_composition(second->op) ||
        (!arc_rule_is_composable(first->op) && !arc_rule_is_composable(second->op))) {
        return 0;
    }
    arc_init_candidate(candidate, ARC_RULE_COMPOSED, "");
    candidate->step_count = 2;
    candidate->input_color = first->input_color;
    candidate->output_color = second->output_color;
    candidate->background_color = first->background_color;
    candidate->selector_kind = first->selector_kind;
    candidate->output_selector_kind = second->output_selector_kind;
    candidate->reference_color = first->reference_color;
    candidate->render_mode = second->render_mode;
    candidate->dx = first->dx + second->dx;
    candidate->dy = first->dy + second->dy;
    candidate->scale = second->scale;
    arc_step_from_candidate(first, &candidate->steps[0]);
    arc_step_from_candidate(second, &candidate->steps[1]);
    snprintf(candidate->name,
             sizeof(candidate->name),
             "compose_%s_then_%s",
             first->name,
             second->name);
    return 1;
}

static int arc_add_candidate(ARCTransformCandidate *candidates,
                             size_t capacity,
                             size_t *count,
                             const ARCTransformCandidate *candidate)
{
    if (*count >= capacity) {
        return 0;
    }
    candidates[*count] = *candidate;
    ++*count;
    return 1;
}

static int arc_add_named_candidate(ARCTransformCandidate *candidates,
                                   size_t capacity,
                                   size_t *count,
                                   ARCTransformRule op,
                                   const char *name)
{
    ARCTransformCandidate candidate;
    arc_init_candidate(&candidate, op, name);
    return arc_add_candidate(candidates, capacity, count, &candidate);
}

static int arc_build_observed_color_map_candidate(const ARCTask *task,
                                                  ARCTransformCandidate *candidate)
{
    arc_init_candidate(candidate, ARC_RULE_GRID_COLOR_MAP, "grid_color_map_same_shape");
    int seen[ARC_COLOR_COUNT];
    memset(seen, 0, sizeof(seen));
    int mapped_any = 0;
    for (size_t pair = 0; pair < task->train_count; ++pair) {
        const ARCGrid *input = &task->train[pair].input;
        const ARCGrid *output = &task->train[pair].output;
        if (!task->train[pair].has_output || !arc_grid_same_shape(input, output)) {
            return 0;
        }
        size_t cells = arc_grid_cells(input);
        for (size_t cell = 0; cell < cells; ++cell) {
            uint8_t in = input->cells[cell];
            uint8_t out = output->cells[cell];
            if (!seen[in]) {
                candidate->color_map[in] = out;
                seen[in] = 1;
                mapped_any = 1;
            } else if (candidate->color_map[in] != out) {
                return 0;
            }
        }
    }
    return mapped_any;
}

static uint8_t arc_grid_dominant_color(const ARCGrid *grid)
{
    size_t counts[ARC_COLOR_COUNT];
    memset(counts, 0, sizeof(counts));
    for (size_t cell = 0; cell < arc_grid_cells(grid); ++cell) {
        if (grid->cells[cell] < ARC_COLOR_COUNT) {
            ++counts[grid->cells[cell]];
        }
    }
    uint8_t best_color = 0;
    size_t best_count = counts[0];
    for (uint8_t color = 1; color < ARC_COLOR_COUNT; ++color) {
        if (counts[color] > best_count) {
            best_count = counts[color];
            best_color = color;
        }
    }
    return best_color;
}

static void arc_add_unique_color(uint8_t *colors, size_t capacity, size_t *count, uint8_t color)
{
    if (colors == NULL || count == NULL || *count >= capacity) {
        return;
    }
    for (size_t i = 0; i < *count; ++i) {
        if (colors[i] == color) {
            return;
        }
    }
    colors[*count] = color;
    ++*count;
}

static size_t arc_task_relation_backgrounds(const ARCTask *task,
                                            uint8_t *backgrounds,
                                            size_t capacity)
{
    size_t count = 0;
    arc_add_unique_color(backgrounds, capacity, &count, 0);
    if (task == NULL || task->train_count == 0) {
        return count;
    }

    int have_background = 0;
    uint8_t observed_background = 0;
    for (size_t pair = 0; pair < task->train_count; ++pair) {
        if (!task->train[pair].has_output) {
            return count;
        }
        uint8_t input_background = arc_grid_dominant_color(&task->train[pair].input);
        uint8_t output_background = arc_grid_dominant_color(&task->train[pair].output);
        if (input_background != output_background) {
            return count;
        }
        if (!have_background) {
            observed_background = input_background;
            have_background = 1;
        } else if (observed_background != input_background) {
            return count;
        }
    }
    if (have_background) {
        arc_add_unique_color(backgrounds, capacity, &count, observed_background);
    }
    return count;
}

static int arc_objects_same_shape(const ARCObject *lhs, const ARCObject *rhs)
{
    size_t lhs_width = lhs->max_x - lhs->min_x + 1u;
    size_t lhs_height = lhs->max_y - lhs->min_y + 1u;
    size_t rhs_width = rhs->max_x - rhs->min_x + 1u;
    size_t rhs_height = rhs->max_y - rhs->min_y + 1u;
    return lhs->count == rhs->count &&
           lhs_width == rhs_width &&
           lhs_height == rhs_height &&
           lhs->shape_hash == rhs->shape_hash;
}

static size_t arc_scene_count_color_objects(const ARCScene *scene, uint8_t color)
{
    size_t count = 0;
    for (size_t i = 0; i < scene->object_count; ++i) {
        count += scene->objects[i].color == color ? 1u : 0u;
    }
    return count;
}

static double arc_object_distance2(const ARCObject *lhs, const ARCObject *rhs)
{
    double dx = lhs->centroid_x - rhs->centroid_x;
    double dy = lhs->centroid_y - rhs->centroid_y;
    return dx * dx + dy * dy;
}

static int arc_select_scene_object(const ARCScene *scene,
                                   uint8_t selector_kind,
                                   uint8_t color,
                                   uint8_t reference_color,
                                   ARCObject *object_out)
{
    if (selector_kind == ARC_OBJECT_SELECTOR_ALL_COLOR) {
        selector_kind = ARC_OBJECT_SELECTOR_UNIQUE_COLOR;
    }
    if (selector_kind == ARC_OBJECT_SELECTOR_UNIQUE_COLOR) {
        return arc_scene_unique_color_object(scene, color, object_out);
    }

    ARCObject reference;
    if (selector_kind == ARC_OBJECT_SELECTOR_NEAREST_TO_COLOR &&
        !arc_scene_unique_color_object(scene, reference_color, &reference)) {
        return 0;
    }

    int found = 0;
    int tied = 0;
    ARCObject best;
    double best_score = 0.0;
    memset(&best, 0, sizeof(best));
    for (size_t i = 0; i < scene->object_count; ++i) {
        const ARCObject *object = &scene->objects[i];
        double score = 0.0;
        if (object->color != color) {
            continue;
        }
        switch (selector_kind) {
        case ARC_OBJECT_SELECTOR_LARGEST_COLOR:
            score = (double)object->count;
            break;
        case ARC_OBJECT_SELECTOR_SMALLEST_COLOR:
            score = -(double)object->count;
            break;
        case ARC_OBJECT_SELECTOR_INTERIOR_COLOR:
            if (object->touches_border) {
                continue;
            }
            score = 0.0;
            break;
        case ARC_OBJECT_SELECTOR_TOUCHING_BORDER_COLOR:
            if (!object->touches_border) {
                continue;
            }
            score = 0.0;
            break;
        case ARC_OBJECT_SELECTOR_NEAREST_TO_COLOR:
            if (object->id == reference.id) {
                continue;
            }
            score = -arc_object_distance2(object, &reference);
            break;
        default:
            return 0;
        }
        if (!found || score > best_score) {
            best = *object;
            best_score = score;
            found = 1;
            tied = 0;
        } else if (fabs(score - best_score) < 1e-9) {
            tied = 1;
        }
    }
    if (!found || tied) {
        return 0;
    }
    if (object_out != NULL) {
        *object_out = best;
    }
    return 1;
}

static int arc_infer_selector_for_scene_object(const ARCScene *scene,
                                               const ARCObject *target,
                                               uint8_t *selector_kind_out,
                                               uint8_t *reference_color_out)
{
    if (scene == NULL || target == NULL ||
        selector_kind_out == NULL || reference_color_out == NULL) {
        return 0;
    }
    static const uint8_t selectors[] = {
        ARC_OBJECT_SELECTOR_UNIQUE_COLOR,
        ARC_OBJECT_SELECTOR_LARGEST_COLOR,
        ARC_OBJECT_SELECTOR_SMALLEST_COLOR,
        ARC_OBJECT_SELECTOR_INTERIOR_COLOR,
        ARC_OBJECT_SELECTOR_TOUCHING_BORDER_COLOR,
    };
    for (size_t i = 0; i < sizeof(selectors) / sizeof(selectors[0]); ++i) {
        ARCObject selected;
        if (arc_select_scene_object(scene,
                                    selectors[i],
                                    target->color,
                                    0,
                                    &selected) &&
            selected.id == target->id) {
            *selector_kind_out = selectors[i];
            *reference_color_out = 0;
            return 1;
        }
    }
    for (uint8_t reference_color = 0;
         reference_color < ARC_COLOR_COUNT;
         ++reference_color) {
        ARCObject selected;
        if (reference_color == scene->background_color ||
            reference_color == target->color ||
            !arc_scene_unique_color_object(scene, reference_color, NULL)) {
            continue;
        }
        if (arc_select_scene_object(scene,
                                    ARC_OBJECT_SELECTOR_NEAREST_TO_COLOR,
                                    target->color,
                                    reference_color,
                                    &selected) &&
            selected.id == target->id) {
            *selector_kind_out = ARC_OBJECT_SELECTOR_NEAREST_TO_COLOR;
            *reference_color_out = reference_color;
            return 1;
        }
    }
    return 0;
}

static int arc_infer_non_reference_selector_for_scene_object(const ARCScene *scene,
                                                            const ARCObject *target,
                                                            uint8_t *selector_kind_out)
{
    uint8_t selector_kind = ARC_OBJECT_SELECTOR_ALL_COLOR;
    uint8_t reference_color = 0;
    if (selector_kind_out == NULL ||
        !arc_infer_selector_for_scene_object(scene,
                                             target,
                                             &selector_kind,
                                             &reference_color) ||
        selector_kind == ARC_OBJECT_SELECTOR_NEAREST_TO_COLOR) {
        return 0;
    }
    *selector_kind_out = selector_kind;
    return 1;
}

static int arc_build_object_translation_candidate(const ARCTask *task,
                                                  uint8_t background,
                                                  uint8_t input_color,
                                                  uint8_t output_color,
                                                  uint8_t selector_kind,
                                                  uint8_t output_selector_kind,
                                                  uint8_t reference_color,
                                                  ARCTransformCandidate *candidate)
{
    int have_offset = 0;
    int dx = 0;
    int dy = 0;
    int selector_used = selector_kind == ARC_OBJECT_SELECTOR_UNIQUE_COLOR;
    int output_selector_used =
        output_selector_kind == ARC_OBJECT_SELECTOR_UNIQUE_COLOR;
    for (size_t pair = 0; pair < task->train_count; ++pair) {
        if (!task->train[pair].has_output ||
            !arc_grid_same_shape(&task->train[pair].input, &task->train[pair].output)) {
            return 0;
        }
        ARCScene input_scene;
        ARCScene output_scene;
        ARCObject input_object;
        ARCObject output_object;
        if (!arc_extract_scene(&task->train[pair].input, background, &input_scene) ||
            !arc_extract_scene(&task->train[pair].output, background, &output_scene) ||
            !arc_select_scene_object(&input_scene,
                                     selector_kind,
                                     input_color,
                                     reference_color,
                                     &input_object) ||
            !arc_select_scene_object(&output_scene,
                                     output_selector_kind,
                                     output_color,
                                     reference_color,
                                     &output_object) ||
            !arc_objects_same_shape(&input_object, &output_object)) {
            return 0;
        }
        if (selector_kind != ARC_OBJECT_SELECTOR_UNIQUE_COLOR &&
            arc_scene_count_color_objects(&input_scene, input_color) > 1u) {
            selector_used = 1;
        }
        if (output_selector_kind != ARC_OBJECT_SELECTOR_UNIQUE_COLOR &&
            arc_scene_count_color_objects(&output_scene, output_color) > 1u) {
            output_selector_used = 1;
        }
        int pair_dx = (int)output_object.min_x - (int)input_object.min_x;
        int pair_dy = (int)output_object.min_y - (int)input_object.min_y;
        if (pair_dx == 0 && pair_dy == 0) {
            return 0;
        }
        if (!have_offset) {
            dx = pair_dx;
            dy = pair_dy;
            have_offset = 1;
        } else if (dx != pair_dx || dy != pair_dy) {
            return 0;
        }
    }
    if (!have_offset || !selector_used || !output_selector_used) {
        return 0;
    }

    arc_init_candidate(candidate, ARC_RULE_OBJECT_TRANSLATE, "");
    candidate->input_color = input_color;
    candidate->output_color = output_color;
    candidate->background_color = background;
    candidate->selector_kind = selector_kind;
    candidate->output_selector_kind = output_selector_kind;
    candidate->reference_color = reference_color;
    candidate->dx = dx;
    candidate->dy = dy;
    if (selector_kind == ARC_OBJECT_SELECTOR_UNIQUE_COLOR &&
        output_selector_kind == ARC_OBJECT_SELECTOR_UNIQUE_COLOR) {
        snprintf(candidate->name,
                 sizeof(candidate->name),
                 "object_translate_bg_%u_color_%u_to_%u_%+d_%+d",
                 (unsigned)background,
                 (unsigned)input_color,
                 (unsigned)output_color,
                 dx,
                 dy);
    } else {
        snprintf(candidate->name,
                 sizeof(candidate->name),
                 "object_translate_%s_to_%s_ref_%u_bg_%u_color_%u_to_%u_%+d_%+d",
                 arc_object_selector_short_name(selector_kind),
                 arc_object_selector_short_name(output_selector_kind),
                 (unsigned)reference_color,
                 (unsigned)background,
                 (unsigned)input_color,
                 (unsigned)output_color,
                 dx,
                 dy);
    }
    return 1;
}

static int arc_scene_has_aligned_color_pair(const ARCScene *scene, uint8_t color, int mode)
{
    for (size_t i = 0; i < scene->object_count; ++i) {
        const ARCObject *lhs = &scene->objects[i];
        if (lhs->color != color) {
            continue;
        }
        for (size_t j = i + 1u; j < scene->object_count; ++j) {
            const ARCObject *rhs = &scene->objects[j];
            if (rhs->color != color) {
                continue;
            }
            int horizontal = fabs(lhs->centroid_y - rhs->centroid_y) < 1e-9;
            int vertical = fabs(lhs->centroid_x - rhs->centroid_x) < 1e-9;
            if ((mode == ARC_OBJECT_CONNECT_HORIZONTAL && horizontal) ||
                (mode == ARC_OBJECT_CONNECT_VERTICAL && vertical) ||
                (mode == ARC_OBJECT_CONNECT_BOTH && (horizontal || vertical))) {
                return 1;
            }
        }
    }
    return 0;
}

static int arc_apply_object_connect_candidate(const ARCTransformCandidate *candidate,
                                              const ARCGrid *input,
                                              ARCGrid *output)
{
    ARCScene scene;
    if (!arc_extract_scene(input, candidate->background_color, &scene) ||
        !arc_scene_has_aligned_color_pair(&scene, candidate->input_color, candidate->dx)) {
        return 0;
    }

    if (arc_candidate_render_mode(candidate) == ARC_OBJECT_RENDER_ONLY) {
        arc_init_grid(output, input->width, input->height);
        for (size_t cell = 0; cell < arc_grid_cells(output); ++cell) {
            output->cells[cell] = candidate->background_color;
        }
    } else {
        *output = *input;
    }
    if (candidate->dx == ARC_OBJECT_CONNECT_HORIZONTAL ||
        candidate->dx == ARC_OBJECT_CONNECT_BOTH) {
        for (size_t y = 0; y < input->height; ++y) {
            int found = 0;
            size_t min_x = input->width;
            size_t max_x = 0;
            for (size_t x = 0; x < input->width; ++x) {
                if (input->cells[y * input->width + x] != candidate->input_color) {
                    continue;
                }
                if (!found || x < min_x) {
                    min_x = x;
                }
                if (!found || x > max_x) {
                    max_x = x;
                }
                found = 1;
            }
            if (!found || min_x == max_x) {
                continue;
            }
            for (size_t x = min_x; x <= max_x; ++x) {
                output->cells[y * output->width + x] = candidate->output_color;
            }
        }
    }
    if (candidate->dx == ARC_OBJECT_CONNECT_VERTICAL ||
        candidate->dx == ARC_OBJECT_CONNECT_BOTH) {
        for (size_t x = 0; x < input->width; ++x) {
            int found = 0;
            size_t min_y = input->height;
            size_t max_y = 0;
            for (size_t y = 0; y < input->height; ++y) {
                if (input->cells[y * input->width + x] != candidate->input_color) {
                    continue;
                }
                if (!found || y < min_y) {
                    min_y = y;
                }
                if (!found || y > max_y) {
                    max_y = y;
                }
                found = 1;
            }
            if (!found || min_y == max_y) {
                continue;
            }
            for (size_t y = min_y; y <= max_y; ++y) {
                output->cells[y * output->width + x] = candidate->output_color;
            }
        }
    }
    return 1;
}

static int arc_build_object_connect_candidate(const ARCTask *task,
                                              uint8_t background,
                                              uint8_t input_color,
                                              uint8_t output_color,
                                              int mode,
                                              uint8_t render_mode,
                                              ARCTransformCandidate *candidate)
{
    const char *mode_name = mode == ARC_OBJECT_CONNECT_HORIZONTAL
                                ? "horizontal"
                                : mode == ARC_OBJECT_CONNECT_VERTICAL ? "vertical" : "both";
    arc_init_candidate(candidate, ARC_RULE_OBJECT_CONNECT_ALIGNED, "");
    candidate->input_color = input_color;
    candidate->output_color = output_color;
    candidate->background_color = background;
    candidate->dx = mode;
    candidate->selector_kind = ARC_OBJECT_SELECTOR_ALL_COLOR;
    candidate->output_selector_kind = ARC_OBJECT_SELECTOR_UNIQUE_COLOR;
    candidate->render_mode = render_mode;
    if (render_mode == ARC_OBJECT_RENDER_ONLY) {
        snprintf(candidate->name,
                 sizeof(candidate->name),
                 "object_connect_bg_%u_color_%u_to_%u_%s_line_only",
                 (unsigned)background,
                 (unsigned)input_color,
                 (unsigned)output_color,
                 mode_name);
    } else {
        snprintf(candidate->name,
                 sizeof(candidate->name),
                 "object_connect_bg_%u_color_%u_to_%u_%s",
                 (unsigned)background,
                 (unsigned)input_color,
                 (unsigned)output_color,
                 mode_name);
    }

    int changed_any = 0;
    for (size_t pair = 0; pair < task->train_count; ++pair) {
        static ARCGrid produced;
        if (!task->train[pair].has_output ||
            !arc_grid_same_shape(&task->train[pair].input, &task->train[pair].output) ||
            !arc_apply_object_connect_candidate(candidate,
                                                &task->train[pair].input,
                                                &produced) ||
            !arc_grid_equal(&produced, &task->train[pair].output)) {
            return 0;
        }
        changed_any = changed_any || !arc_grid_equal(&task->train[pair].input, &produced);
    }
    return changed_any;
}

static int arc_apply_object_fill_enclosed_candidate(const ARCTransformCandidate *candidate,
                                                    const ARCGrid *input,
                                                    ARCGrid *output)
{
    uint8_t reachable[ARC_MAX_GRID_CELLS];
    size_t queue[ARC_MAX_GRID_CELLS];
    if (input->width == 0 || input->height == 0) {
        return 0;
    }
    memset(reachable, 0, sizeof(reachable));
    *output = *input;

    size_t head = 0;
    size_t tail = 0;
    for (size_t x = 0; x < input->width; ++x) {
        size_t top = x;
        size_t bottom = (input->height - 1u) * input->width + x;
        if (input->cells[top] != candidate->input_color && !reachable[top]) {
            reachable[top] = 1;
            queue[tail++] = top;
        }
        if (input->cells[bottom] != candidate->input_color && !reachable[bottom]) {
            reachable[bottom] = 1;
            queue[tail++] = bottom;
        }
    }
    for (size_t y = 0; y < input->height; ++y) {
        size_t left = y * input->width;
        size_t right = y * input->width + input->width - 1u;
        if (input->cells[left] != candidate->input_color && !reachable[left]) {
            reachable[left] = 1;
            queue[tail++] = left;
        }
        if (input->cells[right] != candidate->input_color && !reachable[right]) {
            reachable[right] = 1;
            queue[tail++] = right;
        }
    }

    while (head < tail) {
        size_t cell = queue[head++];
        size_t x = cell % input->width;
        size_t y = cell / input->width;
        const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        for (size_t i = 0; i < sizeof(offsets) / sizeof(offsets[0]); ++i) {
            int nx = (int)x + offsets[i][0];
            int ny = (int)y + offsets[i][1];
            if (nx < 0 || ny < 0 ||
                nx >= (int)input->width || ny >= (int)input->height) {
                continue;
            }
            size_t next = (size_t)ny * input->width + (size_t)nx;
            if (reachable[next] || input->cells[next] == candidate->input_color) {
                continue;
            }
            reachable[next] = 1;
            queue[tail++] = next;
        }
    }

    int changed = 0;
    for (size_t cell = 0; cell < arc_grid_cells(input); ++cell) {
        if (input->cells[cell] == candidate->input_color || reachable[cell]) {
            continue;
        }
        output->cells[cell] = candidate->output_color;
        changed = 1;
    }
    return changed;
}

static int arc_build_object_fill_enclosed_candidate(const ARCTask *task,
                                                    uint8_t border_color,
                                                    uint8_t fill_color,
                                                    ARCTransformCandidate *candidate)
{
    arc_init_candidate(candidate, ARC_RULE_OBJECT_FILL_ENCLOSED, "");
    candidate->input_color = border_color;
    candidate->output_color = fill_color;
    candidate->background_color = 0;
    candidate->selector_kind = ARC_OBJECT_SELECTOR_ALL_COLOR;
    candidate->output_selector_kind = ARC_OBJECT_SELECTOR_UNIQUE_COLOR;
    candidate->render_mode = ARC_OBJECT_RENDER_OVERLAY;
    snprintf(candidate->name,
             sizeof(candidate->name),
             "object_fill_enclosed_%u_to_%u",
             (unsigned)border_color,
             (unsigned)fill_color);

    int changed_any = 0;
    for (size_t pair = 0; pair < task->train_count; ++pair) {
        static ARCGrid produced;
        if (!task->train[pair].has_output ||
            !arc_grid_same_shape(&task->train[pair].input, &task->train[pair].output) ||
            !arc_apply_object_fill_enclosed_candidate(candidate,
                                                      &task->train[pair].input,
                                                      &produced) ||
            !arc_grid_equal(&produced, &task->train[pair].output)) {
            return 0;
        }
        changed_any = changed_any || !arc_grid_equal(&task->train[pair].input, &produced);
    }
    return changed_any;
}

static int arc_build_selected_object_candidate(const ARCTask *task,
                                               ARCTransformRule op,
                                               uint8_t background,
                                               uint8_t input_color,
                                               uint8_t output_color,
                                               uint8_t selector_kind,
                                               uint8_t reference_color,
                                               ARCTransformCandidate *candidate)
{
    arc_init_candidate(candidate, op, "");
    candidate->input_color = input_color;
    candidate->output_color = output_color;
    candidate->background_color = background;
    candidate->selector_kind = selector_kind;
    candidate->output_selector_kind = ARC_OBJECT_SELECTOR_UNIQUE_COLOR;
    candidate->reference_color = reference_color;
    candidate->render_mode = ARC_OBJECT_RENDER_ONLY;

    const char *op_name = op == ARC_RULE_OBJECT_MASK_LARGEST
                              ? "mask"
                              : op == ARC_RULE_OBJECT_CROP_LARGEST_RECOLOR ? "crop_recolor"
                                                                            : "crop";
    int selector_used = selector_kind == ARC_OBJECT_SELECTOR_UNIQUE_COLOR;
    int changed_any = 0;
    for (size_t pair = 0; pair < task->train_count; ++pair) {
        ARCScene scene;
        static ARCGrid produced;
        if (!task->train[pair].has_output ||
            !arc_extract_scene(&task->train[pair].input, background, &scene) ||
            !arc_apply_candidate(candidate, &task->train[pair].input, &produced) ||
            !arc_grid_equal(&produced, &task->train[pair].output)) {
            return 0;
        }
        if (selector_kind != ARC_OBJECT_SELECTOR_UNIQUE_COLOR &&
            arc_scene_count_color_objects(&scene, input_color) > 1u) {
            selector_used = 1;
        }
        changed_any = changed_any || !arc_grid_equal(&task->train[pair].input, &produced);
    }
    if (!selector_used || !changed_any) {
        return 0;
    }

    if (selector_kind == ARC_OBJECT_SELECTOR_NEAREST_TO_COLOR) {
        snprintf(candidate->name,
                 sizeof(candidate->name),
                 "object_select_%s_%s_ref_%u_bg_%u_color_%u_to_%u",
                 op_name,
                 arc_object_selector_short_name(selector_kind),
                 (unsigned)reference_color,
                 (unsigned)background,
                 (unsigned)input_color,
                 (unsigned)output_color);
    } else {
        snprintf(candidate->name,
                 sizeof(candidate->name),
                 "object_select_%s_%s_bg_%u_color_%u_to_%u",
                 op_name,
                 arc_object_selector_short_name(selector_kind),
                 (unsigned)background,
                 (unsigned)input_color,
                 (unsigned)output_color);
    }
    return 1;
}

static int arc_build_object_relation_candidate(const ARCTask *task,
                                               size_t relation_kind,
                                               uint8_t background,
                                               uint8_t input_color,
                                               uint8_t output_color,
                                               uint8_t selector_kind,
                                               uint8_t output_selector_kind,
                                               uint8_t reference_color,
                                               uint8_t render_mode,
                                               int mode,
                                               ARCTransformCandidate *candidate)
{
    ARCTransformCandidate induced;
    memset(&induced, 0, sizeof(induced));

    switch (relation_kind) {
    case ARC_OBJECT_RELATION_TRANSLATE:
        if (!arc_build_object_translation_candidate(task,
                                                    background,
                                                    input_color,
                                                    output_color,
                                                    selector_kind,
                                                    output_selector_kind,
                                                    reference_color,
                                                    &induced)) {
            return 0;
        }
        if (selector_kind == ARC_OBJECT_SELECTOR_UNIQUE_COLOR &&
            output_selector_kind == ARC_OBJECT_SELECTOR_UNIQUE_COLOR) {
            snprintf(induced.name,
                     sizeof(induced.name),
                     "object_relation_%s_bg_%u_color_%u_to_%u_%+d_%+d",
                     arc_object_relation_kind_name(relation_kind),
                     (unsigned)induced.background_color,
                     (unsigned)induced.input_color,
                     (unsigned)induced.output_color,
                     induced.dx,
                     induced.dy);
        } else {
            snprintf(induced.name,
                     sizeof(induced.name),
                     "object_relation_%s_%s_to_%s_ref_%u_bg_%u_color_%u_to_%u_%+d_%+d",
                     arc_object_relation_kind_name(relation_kind),
                     arc_object_selector_short_name(selector_kind),
                     arc_object_selector_short_name(output_selector_kind),
                     (unsigned)reference_color,
                     (unsigned)induced.background_color,
                     (unsigned)induced.input_color,
                     (unsigned)induced.output_color,
                     induced.dx,
                     induced.dy);
        }
        break;
    case ARC_OBJECT_RELATION_CONNECT_ALIGNED: {
        if (!arc_build_object_connect_candidate(task,
                                                background,
                                                input_color,
                                                output_color,
                                                mode,
                                                render_mode,
                                                &induced)) {
            return 0;
        }
        const char *mode_name =
            mode == ARC_OBJECT_CONNECT_HORIZONTAL
                ? "horizontal"
                : mode == ARC_OBJECT_CONNECT_VERTICAL ? "vertical" : "both";
        if (render_mode == ARC_OBJECT_RENDER_ONLY) {
            snprintf(induced.name,
                     sizeof(induced.name),
                     "object_relation_%s_bg_%u_color_%u_to_%u_%s_line_only",
                     arc_object_relation_kind_name(relation_kind),
                     (unsigned)induced.background_color,
                     (unsigned)induced.input_color,
                     (unsigned)induced.output_color,
                     mode_name);
        } else {
            snprintf(induced.name,
                     sizeof(induced.name),
                     "object_relation_%s_bg_%u_color_%u_to_%u_%s",
                     arc_object_relation_kind_name(relation_kind),
                     (unsigned)induced.background_color,
                     (unsigned)induced.input_color,
                     (unsigned)induced.output_color,
                     mode_name);
        }
        break;
    }
    case ARC_OBJECT_RELATION_FILL_ENCLOSED:
        if (!arc_build_object_fill_enclosed_candidate(task,
                                                      input_color,
                                                      output_color,
                                                      &induced)) {
            return 0;
        }
        snprintf(induced.name,
                 sizeof(induced.name),
                 "object_relation_%s_%u_to_%u",
                 arc_object_relation_kind_name(relation_kind),
                 (unsigned)induced.input_color,
                 (unsigned)induced.output_color);
        break;
    default:
        return 0;
    }

    induced.op = ARC_RULE_OBJECT_RELATION_INDUCED;
    induced.scale = relation_kind;
    *candidate = induced;
    return 1;
}

static size_t arc_build_candidates(const ARCTask *task,
                                   ARCTransformCandidate *candidates,
                                   size_t capacity)
{
    size_t count = 0;
    ARCTransformCandidate candidate;
    static const struct {
        ARCTransformRule op;
        const char *prefix;
    } colored_templates[] = {
        {ARC_RULE_COPY_LEFT, "copy_left_color_"},
        {ARC_RULE_COPY_RIGHT, "copy_right_color_"},
        {ARC_RULE_INTERSECTION, "mask_intersection_color_"},
        {ARC_RULE_UNION, "mask_union_color_"},
        {ARC_RULE_OBJECT_SELECT, "object_select_overlap_color_"},
        {ARC_RULE_LAYER_OVERLAP, "layer_overlap_color_"},
        {ARC_RULE_LAYER_UNION, "layer_union_color_"},
        {ARC_RULE_LAYER_XOR, "layer_xor_color_"},
        {ARC_RULE_LAYER_LEFT_ONLY, "layer_left_only_color_"},
        {ARC_RULE_LAYER_RIGHT_ONLY, "layer_right_only_color_"},
    };

    for (size_t template_id = 0;
         template_id < sizeof(colored_templates) / sizeof(colored_templates[0]);
         ++template_id) {
        for (uint8_t color = 1; color < ARC_COLOR_COUNT; ++color) {
            if (count >= capacity) {
                return count;
            }
            arc_init_candidate(&candidate, colored_templates[template_id].op, "");
            candidate.output_color = color;
            snprintf(candidate.name,
                     sizeof(candidate.name),
                     "%s%u",
                     colored_templates[template_id].prefix,
                     (unsigned)color);
            arc_add_candidate(candidates, capacity, &count, &candidate);
        }
    }

    arc_add_named_candidate(candidates, capacity, &count, ARC_RULE_COLOR_MAP_LEFT, "color_map_left_identity");
    arc_add_named_candidate(candidates, capacity, &count, ARC_RULE_GRID_IDENTITY, "grid_identity");
    arc_add_named_candidate(candidates, capacity, &count, ARC_RULE_ROTATE90, "rotate90");
    arc_add_named_candidate(candidates, capacity, &count, ARC_RULE_ROTATE180, "rotate180");
    arc_add_named_candidate(candidates, capacity, &count, ARC_RULE_ROTATE270, "rotate270");
    arc_add_named_candidate(candidates, capacity, &count, ARC_RULE_FLIP_HORIZONTAL, "flip_horizontal");
    arc_add_named_candidate(candidates, capacity, &count, ARC_RULE_FLIP_VERTICAL, "flip_vertical");
    arc_add_named_candidate(candidates, capacity, &count, ARC_RULE_TRANSPOSE, "transpose");

    for (size_t scale = 1; scale <= 4; ++scale) {
        arc_init_candidate(&candidate, ARC_RULE_SCALE, "");
        candidate.scale = scale;
        snprintf(candidate.name, sizeof(candidate.name), "scale_%zux", scale);
        arc_add_candidate(candidates, capacity, &count, &candidate);
    }
    if (!g_arc_active_config || !g_arc_active_config->disable_reflect_tile) {
        for (size_t scale = 2; scale <= 4; ++scale) {
            arc_init_candidate(&candidate, ARC_RULE_REFLECT_TILE, "");
            candidate.scale = scale;
            snprintf(candidate.name, sizeof(candidate.name), "reflect_tile_%zux", scale);
            arc_add_candidate(candidates, capacity, &count, &candidate);
        }
    }
    if (!g_arc_active_config || !g_arc_active_config->disable_marker_motif_tile) {
    for (uint8_t background = 0; background < ARC_COLOR_COUNT; ++background) {
        for (uint8_t marker = 1; marker < ARC_COLOR_COUNT; ++marker) {
            if (marker == background) {
                continue;
            }
            arc_init_candidate(&candidate, ARC_RULE_MARKER_MOTIF_TILE, "");
            candidate.background_color = background;
            candidate.input_color = marker;
            candidate.selector_kind = 0;
            snprintf(candidate.name,
                     sizeof(candidate.name),
                     "marker_motif_tile_bg_%u_color_%u",
                     (unsigned)background,
                     (unsigned)marker);
            arc_add_candidate(candidates, capacity, &count, &candidate);
        }
        arc_init_candidate(&candidate, ARC_RULE_MARKER_MOTIF_TILE, "");
        candidate.background_color = background;
        candidate.selector_kind = 1;
        snprintf(candidate.name,
                 sizeof(candidate.name),
                 "marker_motif_tile_bg_%u_dominant",
                 (unsigned)background);
        arc_add_candidate(candidates, capacity, &count, &candidate);

        arc_init_candidate(&candidate, ARC_RULE_MARKER_MOTIF_TILE, "");
        candidate.background_color = background;
        candidate.selector_kind = 2;
        snprintf(candidate.name,
                 sizeof(candidate.name),
                 "marker_motif_tile_bg_%u_rarest",
                 (unsigned)background);
        arc_add_candidate(candidates, capacity, &count, &candidate);
    }
    }
    if (!g_arc_active_config || !g_arc_active_config->disable_axis_tile) {
    static const struct {
        int dx;
        int dy;
        uint8_t mode;
        const char *name;
    } axis_tile_modes[] = {
        {2, 1, 0, "axis_tile_x2_repeat"},
        {2, 1, 1, "axis_tile_x2_reflect"},
        {2, 1, 2, "axis_tile_x2_reflect_first"},
        {1, 2, 0, "axis_tile_y2_repeat"},
        {1, 2, 1, "axis_tile_y2_reflect"},
        {1, 2, 2, "axis_tile_y2_reflect_first"},
    };
    for (size_t axis_index = 0;
         axis_index < sizeof(axis_tile_modes) / sizeof(axis_tile_modes[0]);
         ++axis_index) {
        arc_init_candidate(&candidate, ARC_RULE_AXIS_TILE, "");
        candidate.dx = axis_tile_modes[axis_index].dx;
        candidate.dy = axis_tile_modes[axis_index].dy;
        candidate.render_mode = axis_tile_modes[axis_index].mode;
        snprintf(candidate.name,
                 sizeof(candidate.name),
                 "%s",
                 axis_tile_modes[axis_index].name);
        arc_add_candidate(candidates, capacity, &count, &candidate);
    }
    }
    if (!g_arc_active_config || !g_arc_active_config->disable_period_extend) {
        arc_init_candidate(&candidate, ARC_RULE_PERIOD_EXTEND, "");
        candidate.dx = 2;
        candidate.dy = 1;
        snprintf(candidate.name, sizeof(candidate.name), "period_extend_x2");
        arc_add_candidate(candidates, capacity, &count, &candidate);

        arc_init_candidate(&candidate, ARC_RULE_PERIOD_EXTEND, "");
        candidate.dx = 1;
        candidate.dy = 2;
        snprintf(candidate.name, sizeof(candidate.name), "period_extend_y2");
        arc_add_candidate(candidates, capacity, &count, &candidate);
    }

    if (count < capacity && arc_build_observed_color_map_candidate(task, &candidate)) arc_add_candidate(candidates, capacity, &count, &candidate);
    for (size_t p = 0; p < task->train_count && count < capacity; ++p) {
        const ARCGrid *ip = &task->train[p].input, *op = &task->train[p].output;
        if (!task->train[p].has_output || !arc_grid_same_shape(ip, op)) continue;
        ARCTransformCandidate pc; arc_init_candidate(&pc, ARC_RULE_GRID_COLOR_MAP, ""); int m=0;
        for (size_t ce=0;ce<arc_grid_cells(ip);++ce){uint8_t i_n=ip->cells[ce],o_t=op->cells[ce];if(pc.color_map[i_n]==0||pc.color_map[i_n]==o_t){pc.color_map[i_n]=o_t;if(i_n!=o_t)m=1;}}
        if(m){snprintf(pc.name,sizeof(pc.name),"grid_color_map_pair_%zu",p);arc_add_candidate(candidates,capacity,&count,&pc);}
    }

    for (uint8_t background = 0; background < ARC_COLOR_COUNT; ++background) {
        arc_init_candidate(&candidate, ARC_RULE_CROP_BACKGROUND, "");
        candidate.background_color = background;
        snprintf(candidate.name, sizeof(candidate.name), "crop_non_background_%u", (unsigned)background);
        arc_add_candidate(candidates, capacity, &count, &candidate);

        for (uint8_t color = 1; color < ARC_COLOR_COUNT; ++color) {
            arc_init_candidate(&candidate, ARC_RULE_GRID_FOREGROUND_RECOLOR, "");
            candidate.background_color = background;
            candidate.output_color = color;
            snprintf(candidate.name,
                     sizeof(candidate.name),
                     "grid_recolor_non_background_%u_to_%u",
                     (unsigned)background,
                     (unsigned)color);
            arc_add_candidate(candidates, capacity, &count, &candidate);

            arc_init_candidate(&candidate, ARC_RULE_CROP_BACKGROUND_RECOLOR, "");
            candidate.background_color = background;
            candidate.output_color = color;
            snprintf(candidate.name,
                     sizeof(candidate.name),
                     "crop_recolor_non_background_%u_to_%u",
                     (unsigned)background,
                     (unsigned)color);
            arc_add_candidate(candidates, capacity, &count, &candidate);

            arc_init_candidate(&candidate, ARC_RULE_BBOX_OUTLINE, "");
            candidate.background_color = background;
            candidate.output_color = color;
            snprintf(candidate.name,
                     sizeof(candidate.name),
                     "bbox_outline_%u_to_%u",
                     (unsigned)background,
                     (unsigned)color);
            arc_add_candidate(candidates, capacity, &count, &candidate);

            arc_init_candidate(&candidate, ARC_RULE_BBOX_FILL, "");
            candidate.background_color = background;
            candidate.output_color = color;
            snprintf(candidate.name,
                     sizeof(candidate.name),
                     "bbox_fill_%u_to_%u",
                     (unsigned)background,
                     (unsigned)color);
            arc_add_candidate(candidates, capacity, &count, &candidate);
        }
    }

    for (uint8_t background = 0; background < ARC_COLOR_COUNT; ++background) {
        arc_init_candidate(&candidate, ARC_RULE_OBJECT_CROP_LARGEST, "");
        candidate.background_color = background;
        snprintf(candidate.name,
                 sizeof(candidate.name),
                 "object_crop_largest_non_background_%u",
                 (unsigned)background);
        arc_add_candidate(candidates, capacity, &count, &candidate);

        for (uint8_t color = 1; color < ARC_COLOR_COUNT; ++color) {
            arc_init_candidate(&candidate, ARC_RULE_OBJECT_CROP_LARGEST_RECOLOR, "");
            candidate.background_color = background;
            candidate.output_color = color;
            snprintf(candidate.name,
                     sizeof(candidate.name),
                     "object_crop_largest_non_background_%u_to_%u",
                     (unsigned)background,
                     (unsigned)color);
            arc_add_candidate(candidates, capacity, &count, &candidate);

            arc_init_candidate(&candidate, ARC_RULE_OBJECT_MASK_LARGEST, "");
            candidate.background_color = background;
            candidate.output_color = color;
            snprintf(candidate.name,
                     sizeof(candidate.name),
                     "object_mask_largest_non_background_%u_to_%u",
                     (unsigned)background,
                     (unsigned)color);
            arc_add_candidate(candidates, capacity, &count, &candidate);
        }
    }

    for (uint8_t input_color = 1; input_color < ARC_COLOR_COUNT; ++input_color) {
        arc_init_candidate(&candidate, ARC_RULE_CROP_COLOR, "");
        candidate.input_color = input_color;
        snprintf(candidate.name, sizeof(candidate.name), "crop_color_%u", (unsigned)input_color);
        arc_add_candidate(candidates, capacity, &count, &candidate);

        for (uint8_t output_color = 1; output_color < ARC_COLOR_COUNT; ++output_color) {
            arc_init_candidate(&candidate, ARC_RULE_CROP_COLOR_MASK, "");
            candidate.input_color = input_color;
            candidate.output_color = output_color;
            snprintf(candidate.name,
                     sizeof(candidate.name),
                     "crop_color_%u_mask_to_%u",
                     (unsigned)input_color,
                     (unsigned)output_color);
            arc_add_candidate(candidates, capacity, &count, &candidate);
        }
    }

    static const struct {
        int dx;
        int dy;
        const char *suffix;
    } compress_modes[] = {
        {0, 1, "rows"},
        {1, 0, "cols"},
        {1, 1, "rows_cols"},
    };
    for (uint8_t color = 0; color < ARC_COLOR_COUNT; ++color) {
        for (size_t mode = 0; mode < sizeof(compress_modes) / sizeof(compress_modes[0]); ++mode) {
            arc_init_candidate(&candidate, ARC_RULE_COMPRESS_COLOR_LINES, "");
            candidate.background_color = color;
            candidate.dx = compress_modes[mode].dx;
            candidate.dy = compress_modes[mode].dy;
            snprintf(candidate.name,
                     sizeof(candidate.name),
                     "compress_color_%u_%s",
                     (unsigned)color,
                     compress_modes[mode].suffix);
            arc_add_candidate(candidates, capacity, &count, &candidate);
        }
    }
    for (uint8_t background = 0; background < ARC_COLOR_COUNT; ++background) {
        for (size_t per = 2; per <= 4; ++per) {
            if (count >= capacity) break;
            if (arc_build_tile_motif_parity_repeat_candidate(task, background, per, &candidate))
                arc_add_candidate(candidates, capacity, &count, &candidate);
        }
    }

    uint8_t relation_backgrounds[2];
    size_t relation_background_count =
        arc_task_relation_backgrounds(task,
                                      relation_backgrounds,
                                      sizeof(relation_backgrounds) /
                                          sizeof(relation_backgrounds[0]));
    static const uint8_t object_selectors[] = {
        ARC_OBJECT_SELECTOR_UNIQUE_COLOR,
        ARC_OBJECT_SELECTOR_LARGEST_COLOR,
        ARC_OBJECT_SELECTOR_SMALLEST_COLOR,
        ARC_OBJECT_SELECTOR_INTERIOR_COLOR,
        ARC_OBJECT_SELECTOR_TOUCHING_BORDER_COLOR,
    };
    for (size_t background_index = 0;
         background_index < relation_background_count;
         ++background_index) {
        uint8_t background = relation_backgrounds[background_index];
        for (uint8_t input_color = 0; input_color < ARC_COLOR_COUNT; ++input_color) {
            if (input_color == background) {
                continue;
            }
            for (size_t selector_index = 0;
                 selector_index < sizeof(object_selectors) / sizeof(object_selectors[0]);
                 ++selector_index) {
                uint8_t selector = object_selectors[selector_index];
                if (arc_build_selected_object_candidate(task,
                                                        ARC_RULE_OBJECT_CROP_LARGEST,
                                                        background,
                                                        input_color,
                                                        input_color,
                                                        selector,
                                                        0,
                                                        &candidate)) {
                    arc_add_candidate(candidates, capacity, &count, &candidate);
                }
                for (uint8_t output_color = 1;
                     output_color < ARC_COLOR_COUNT;
                     ++output_color) {
                    if (arc_build_selected_object_candidate(task,
                                                            ARC_RULE_OBJECT_CROP_LARGEST_RECOLOR,
                                                            background,
                                                            input_color,
                                                            output_color,
                                                            selector,
                                                            0,
                                                            &candidate)) {
                        arc_add_candidate(candidates, capacity, &count, &candidate);
                    }
                    if (arc_build_selected_object_candidate(task,
                                                            ARC_RULE_OBJECT_MASK_LARGEST,
                                                            background,
                                                            input_color,
                                                            output_color,
                                                            selector,
                                                            0,
                                                            &candidate)) {
                        arc_add_candidate(candidates, capacity, &count, &candidate);
                    }
                }
            }
            for (uint8_t reference_color = 0;
                 reference_color < ARC_COLOR_COUNT;
                 ++reference_color) {
                if (reference_color == background ||
                    reference_color == input_color) {
                    continue;
                }
                if (arc_build_selected_object_candidate(task,
                                                        ARC_RULE_OBJECT_CROP_LARGEST,
                                                        background,
                                                        input_color,
                                                        input_color,
                                                        ARC_OBJECT_SELECTOR_NEAREST_TO_COLOR,
                                                        reference_color,
                                                        &candidate)) {
                    arc_add_candidate(candidates, capacity, &count, &candidate);
                }
                for (uint8_t output_color = 1;
                     output_color < ARC_COLOR_COUNT;
                     ++output_color) {
                    if (arc_build_selected_object_candidate(task,
                                                            ARC_RULE_OBJECT_CROP_LARGEST_RECOLOR,
                                                            background,
                                                            input_color,
                                                            output_color,
                                                            ARC_OBJECT_SELECTOR_NEAREST_TO_COLOR,
                                                            reference_color,
                                                            &candidate)) {
                        arc_add_candidate(candidates, capacity, &count, &candidate);
                    }
                    if (arc_build_selected_object_candidate(task,
                                                            ARC_RULE_OBJECT_MASK_LARGEST,
                                                            background,
                                                            input_color,
                                                            output_color,
                                                            ARC_OBJECT_SELECTOR_NEAREST_TO_COLOR,
                                                            reference_color,
                                                            &candidate)) {
                        arc_add_candidate(candidates, capacity, &count, &candidate);
                    }
                }
            }
        }
    }
    static const uint8_t translate_selectors[] = {
        ARC_OBJECT_SELECTOR_UNIQUE_COLOR,
        ARC_OBJECT_SELECTOR_LARGEST_COLOR,
        ARC_OBJECT_SELECTOR_SMALLEST_COLOR,
        ARC_OBJECT_SELECTOR_INTERIOR_COLOR,
        ARC_OBJECT_SELECTOR_TOUCHING_BORDER_COLOR,
    };
    static const uint8_t translate_output_selectors[] = {
        ARC_OBJECT_SELECTOR_UNIQUE_COLOR,
        ARC_OBJECT_SELECTOR_LARGEST_COLOR,
        ARC_OBJECT_SELECTOR_SMALLEST_COLOR,
        ARC_OBJECT_SELECTOR_INTERIOR_COLOR,
        ARC_OBJECT_SELECTOR_TOUCHING_BORDER_COLOR,
    };
    for (size_t background_index = 0;
         background_index < relation_background_count;
         ++background_index) {
        uint8_t background = relation_backgrounds[background_index];
        for (uint8_t input_color = 0; input_color < ARC_COLOR_COUNT; ++input_color) {
            if (input_color == background) {
                continue;
            }
            for (uint8_t output_color = 0; output_color < ARC_COLOR_COUNT; ++output_color) {
                if (output_color == background) {
                    continue;
                }
                for (size_t selector_index = 0;
                     selector_index < sizeof(translate_selectors) /
                                          sizeof(translate_selectors[0]);
                     ++selector_index) {
                    for (size_t output_selector_index = 0;
                         output_selector_index < sizeof(translate_output_selectors) /
                                                     sizeof(translate_output_selectors[0]);
                         ++output_selector_index) {
                        if (arc_build_object_relation_candidate(
                                task,
                                ARC_OBJECT_RELATION_TRANSLATE,
                                background,
                                input_color,
                                output_color,
                                translate_selectors[selector_index],
                                translate_output_selectors[output_selector_index],
                                0,
                                ARC_OBJECT_RENDER_ONLY,
                                0,
                                &candidate)) {
                            arc_add_candidate(candidates, capacity, &count, &candidate);
                        }
                    }
                }
                for (uint8_t reference_color = 0;
                     reference_color < ARC_COLOR_COUNT;
                     ++reference_color) {
                    if (reference_color == background ||
                        reference_color == input_color) {
                        continue;
                    }
                    if (arc_build_object_relation_candidate(task,
                                                            ARC_OBJECT_RELATION_TRANSLATE,
                                                            background,
                                                            input_color,
                                                            output_color,
                                                            ARC_OBJECT_SELECTOR_NEAREST_TO_COLOR,
                                                            ARC_OBJECT_SELECTOR_UNIQUE_COLOR,
                                                            reference_color,
                                                            ARC_OBJECT_RENDER_ONLY,
                                                            0,
                                                            &candidate)) {
                        arc_add_candidate(candidates, capacity, &count, &candidate);
                    }
                }
                for (int mode = ARC_OBJECT_CONNECT_HORIZONTAL;
                     mode <= ARC_OBJECT_CONNECT_BOTH;
                     ++mode) {
                    for (uint8_t render_mode = ARC_OBJECT_RENDER_OVERLAY;
                         render_mode < ARC_OBJECT_RENDER_KIND_COUNT;
                         ++render_mode) {
                        if (arc_build_object_relation_candidate(task,
                                                                ARC_OBJECT_RELATION_CONNECT_ALIGNED,
                                                                background,
                                                                input_color,
                                                                output_color,
                                                                ARC_OBJECT_SELECTOR_ALL_COLOR,
                                                                ARC_OBJECT_SELECTOR_UNIQUE_COLOR,
                                                                0,
                                                                render_mode,
                                                                mode,
                                                                &candidate)) {
                            arc_add_candidate(candidates, capacity, &count, &candidate);
                        }
                    }
                }
            }
        }
    }

    for (uint8_t color = 1; color < ARC_COLOR_COUNT; ++color) {
        for (uint8_t fill_color = 1; fill_color < ARC_COLOR_COUNT; ++fill_color) {
            if (arc_build_object_relation_candidate(task,
                                                    ARC_OBJECT_RELATION_FILL_ENCLOSED,
                                                    0,
                                                    color,
                                                    fill_color,
                                                    ARC_OBJECT_SELECTOR_ALL_COLOR,
                                                    ARC_OBJECT_SELECTOR_UNIQUE_COLOR,
                                                    0,
                                                    ARC_OBJECT_RENDER_OVERLAY,
                                                    0,
                                                    &candidate)) {
                arc_add_candidate(candidates, capacity, &count, &candidate);
            }
        }
    }

    for (int dx = -5; dx <= 5; ++dx) {
        for (int dy = -5; dy <= 5; ++dy) {
            if (dx == 0 && dy == 0) continue;
            arc_init_candidate(&candidate, ARC_RULE_TRANSLATE, "");
            candidate.background_color = 0; candidate.dx = dx; candidate.dy = dy;
            snprintf(candidate.name, sizeof(candidate.name), "translate_nonzero_%+d_%+d", dx, dy);
            arc_add_candidate(candidates, capacity, &count, &candidate);
        }
    }
    for (uint8_t bg = 0; bg < ARC_COLOR_COUNT; ++bg) {
        static const ARCTransformRule gr[] = {ARC_RULE_GRAVITY_DOWN,ARC_RULE_GRAVITY_UP,ARC_RULE_GRAVITY_LEFT,ARC_RULE_GRAVITY_RIGHT};
        static const char *gn[] = {"down","up","left","right"};
        for(size_t g=0;g<4;++g){if(count>=capacity)break;arc_init_candidate(&candidate,gr[g],"");candidate.background_color=bg;snprintf(candidate.name,sizeof(candidate.name),"gravity_%s_bg_%u",gn[g],(unsigned)bg);arc_add_candidate(candidates,capacity,&count,&candidate);}
    }
    for(uint8_t bg=0;bg<ARC_COLOR_COUNT;++bg){if(count>=capacity)break;arc_init_candidate(&candidate,ARC_RULE_TILE_REPEAT,"");candidate.background_color=bg;snprintf(candidate.name,sizeof(candidate.name),"tile_repeat_bg_%u",(unsigned)bg);arc_add_candidate(candidates,capacity,&count,&candidate);}
    if(count<capacity){arc_init_candidate(&candidate,ARC_RULE_GRID_REPEAT,"");snprintf(candidate.name,sizeof(candidate.name),"grid_repeat");arc_add_candidate(candidates,capacity,&count,&candidate);}
    for(uint8_t c=1;c<ARC_COLOR_COUNT;++c){if(count>=capacity)break;arc_init_candidate(&candidate,ARC_RULE_EXTRACT_COLOR,"");candidate.input_color=c;snprintf(candidate.name,sizeof(candidate.name),"extract_color_%u",(unsigned)c);arc_add_candidate(candidates,capacity,&count,&candidate);}
    for(uint8_t c=1;c<ARC_COLOR_COUNT;++c){if(count>=capacity)break;arc_init_candidate(&candidate,ARC_RULE_ERASE_COLOR,"");candidate.input_color=c;snprintf(candidate.name,sizeof(candidate.name),"erase_color_%u",(unsigned)c);arc_add_candidate(candidates,capacity,&count,&candidate);}
    {int multi=0;for(size_t p=0;p<task->train_count;++p){const ARCGrid*g=&task->train[p].input;int seen[10]={0};int d=0;for(size_t c=0;c<arc_grid_cells(g);++c){uint8_t v=g->cells[c];if(v&&!seen[v]){seen[v]=1;++d;}}if(d>=2){multi=1;break;}}if(multi&&count<capacity){arc_init_candidate(&candidate,ARC_RULE_RECOLOR_BY_REFERENCE,"");candidate.background_color=0;snprintf(candidate.name,sizeof(candidate.name),"recolor_by_reference");arc_add_candidate(candidates,capacity,&count,&candidate);}}
    if(count<capacity){arc_init_candidate(&candidate,ARC_RULE_PATTERN_EXTEND,"");candidate.background_color=0;snprintf(candidate.name,sizeof(candidate.name),"pattern_extend");arc_add_candidate(candidates,capacity,&count,&candidate);}
    if(count<capacity){arc_init_candidate(&candidate,ARC_RULE_DRAW_LINE,"");candidate.background_color=0;snprintf(candidate.name,sizeof(candidate.name),"draw_line");arc_add_candidate(candidates,capacity,&count,&candidate);}
    if(count<capacity){arc_init_candidate(&candidate,ARC_RULE_SYMMETRY_H,"");snprintf(candidate.name,sizeof(candidate.name),"symmetry_h");arc_add_candidate(candidates,capacity,&count,&candidate);}
    if(count<capacity){arc_init_candidate(&candidate,ARC_RULE_SYMMETRY_V,"");snprintf(candidate.name,sizeof(candidate.name),"symmetry_v");arc_add_candidate(candidates,capacity,&count,&candidate);}
    if(count<capacity){arc_init_candidate(&candidate,ARC_RULE_MULTISCALE,"");candidate.background_color=0;snprintf(candidate.name,sizeof(candidate.name),"multiscale");arc_add_candidate(candidates,capacity,&count,&candidate);}
    if(count<capacity){arc_init_candidate(&candidate,ARC_RULE_OBJECT_LABEL,"");candidate.background_color=0;snprintf(candidate.name,sizeof(candidate.name),"object_label");arc_add_candidate(candidates,capacity,&count,&candidate);}
    for(uint8_t dc=1;dc<ARC_COLOR_COUNT;++dc){if(count>=capacity)break;arc_init_candidate(&candidate,ARC_RULE_DILATE,"");candidate.input_color=dc;snprintf(candidate.name,sizeof(candidate.name),"dilate_color_%u",(unsigned)dc);arc_add_candidate(candidates,capacity,&count,&candidate);}
    if(count<capacity){arc_init_candidate(&candidate,ARC_RULE_PERIOD_TILE,"");candidate.background_color=0;snprintf(candidate.name,sizeof(candidate.name),"period_tile");arc_add_candidate(candidates,capacity,&count,&candidate);}
    { ARCTransformCandidate cons[32]; size_t nc=0;
      for(size_t i=0;i<count&&nc<32;++i)if(arc_rule_is_composable(candidates[i].op)&&!arc_candidate_is_composed(&candidates[i])){ARCTransformCandidate t=candidates[i];size_t c2=0,m2=0,u2=0;if(arc_candidate_matches_train(task,&t,&c2,&m2,&u2))cons[nc++]=candidates[i];}
      for(size_t a=0;a<nc&&count+4<capacity;++a)for(size_t b=0;b<nc&&count+4<capacity;++b){if(a==b)continue;ARCTransformCandidate comp;if(arc_init_composed_candidate(&comp,&cons[a],&cons[b]))arc_add_candidate(candidates,capacity,&count,&comp);}
    }
    return count;return count;
}

static int arc_find_non_background_bbox(const ARCGrid *grid,
                                        uint8_t background,
                                        size_t *min_x,
                                        size_t *min_y,
                                        size_t *max_x,
                                        size_t *max_y)
{
    int found = 0;
    *min_x = grid->width;
    *min_y = grid->height;
    *max_x = 0;
    *max_y = 0;
    for (size_t y = 0; y < grid->height; ++y) {
        for (size_t x = 0; x < grid->width; ++x) {
            if (grid->cells[y * grid->width + x] == background) {
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

static int arc_find_color_bbox(const ARCGrid *grid,
                               uint8_t color,
                               size_t *min_x,
                               size_t *min_y,
                               size_t *max_x,
                               size_t *max_y)
{
    int found = 0;
    *min_x = grid->width;
    *min_y = grid->height;
    *max_x = 0;
    *max_y = 0;
    for (size_t y = 0; y < grid->height; ++y) {
        for (size_t x = 0; x < grid->width; ++x) {
            if (grid->cells[y * grid->width + x] != color) {
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

static void arc_component_clear(ARCComponent *component)
{
    memset(component, 0, sizeof(*component));
    component->min_x = ARC_MAX_GRID_WIDTH;
    component->min_y = ARC_MAX_GRID_HEIGHT;
}

static int arc_find_largest_component(const ARCGrid *grid,
                                      uint8_t background,
                                      ARCComponent *component)
{
    uint8_t visited[ARC_MAX_GRID_CELLS];
    uint8_t current_mask[ARC_MAX_GRID_CELLS];
    size_t queue[ARC_MAX_GRID_CELLS];
    memset(visited, 0, sizeof(visited));
    arc_component_clear(component);

    size_t cells = arc_grid_cells(grid);
    for (size_t start = 0; start < cells; ++start) {
        if (visited[start] || grid->cells[start] == background) {
            continue;
        }
        memset(current_mask, 0, sizeof(current_mask));
        size_t head = 0;
        size_t tail = 0;
        size_t count = 0;
        size_t min_x = grid->width;
        size_t min_y = grid->height;
        size_t max_x = 0;
        size_t max_y = 0;

        visited[start] = 1;
        queue[tail++] = start;
        while (head < tail) {
            size_t cell = queue[head++];
            size_t x = cell % grid->width;
            size_t y = cell / grid->width;
            current_mask[cell] = 1;
            ++count;
            if (x < min_x) {
                min_x = x;
            }
            if (y < min_y) {
                min_y = y;
            }
            if (x > max_x) {
                max_x = x;
            }
            if (y > max_y) {
                max_y = y;
            }

            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            for (size_t i = 0; i < 4; ++i) {
                int nx = (int)x + offsets[i][0];
                int ny = (int)y + offsets[i][1];
                if (nx < 0 || ny < 0 ||
                    nx >= (int)grid->width || ny >= (int)grid->height) {
                    continue;
                }
                size_t next = (size_t)ny * grid->width + (size_t)nx;
                if (visited[next] || grid->cells[next] == background) {
                    continue;
                }
                visited[next] = 1;
                queue[tail++] = next;
            }
        }

        int better = count > component->count ||
                     (count == component->count && component->count > 0 &&
                      (min_y < component->min_y ||
                       (min_y == component->min_y && min_x < component->min_x)));
        if (better) {
            component->count = count;
            component->min_x = min_x;
            component->min_y = min_y;
            component->max_x = max_x;
            component->max_y = max_y;
            memcpy(component->mask, current_mask, sizeof(component->mask));
        }
    }
    return component->count > 0;
}

static uint64_t arc_scene_hash_mix(uint64_t hash, uint64_t value)
{
    hash ^= value + UINT64_C(0x9e3779b97f4a7c15) + (hash << 6) + (hash >> 2);
    return hash;
}

static uint64_t arc_shape_hash_from_mask(const ARCGrid *grid,
                                         const uint8_t *mask,
                                         size_t min_x,
                                         size_t min_y,
                                         size_t max_x,
                                         size_t max_y,
                                         size_t count)
{
    uint64_t hash = UINT64_C(1469598103934665603);
    size_t width = max_x >= min_x ? max_x - min_x + 1u : 0u;
    size_t height = max_y >= min_y ? max_y - min_y + 1u : 0u;
    hash = arc_scene_hash_mix(hash, width);
    hash = arc_scene_hash_mix(hash, height);
    hash = arc_scene_hash_mix(hash, count);
    for (size_t y = min_y; y <= max_y && y < grid->height; ++y) {
        for (size_t x = min_x; x <= max_x && x < grid->width; ++x) {
            size_t cell = y * grid->width + x;
            hash = arc_scene_hash_mix(hash, mask[cell] ? 1u : 0u);
        }
    }
    return hash;
}

static void arc_scene_clear(ARCScene *scene, uint8_t background)
{
    memset(scene, 0, sizeof(*scene));
    scene->background_color = background;
}

static int arc_ranges_overlap(size_t lhs_min,
                              size_t lhs_max,
                              size_t rhs_min,
                              size_t rhs_max)
{
    return lhs_min <= rhs_max && rhs_min <= lhs_max;
}

static int arc_objects_adjacent(const ARCObject *lhs, const ARCObject *rhs)
{
    int horizontal_touch =
        (lhs->max_x + 1u == rhs->min_x || rhs->max_x + 1u == lhs->min_x) &&
        arc_ranges_overlap(lhs->min_y, lhs->max_y, rhs->min_y, rhs->max_y);
    int vertical_touch =
        (lhs->max_y + 1u == rhs->min_y || rhs->max_y + 1u == lhs->min_y) &&
        arc_ranges_overlap(lhs->min_x, lhs->max_x, rhs->min_x, rhs->max_x);
    return horizontal_touch || vertical_touch;
}

static int arc_objects_aligned(const ARCObject *lhs, const ARCObject *rhs)
{
    return lhs->min_x == rhs->min_x ||
           lhs->max_x == rhs->max_x ||
           lhs->min_y == rhs->min_y ||
           lhs->max_y == rhs->max_y ||
           fabs(lhs->centroid_x - rhs->centroid_x) < 1e-9 ||
           fabs(lhs->centroid_y - rhs->centroid_y) < 1e-9;
}

static void arc_scene_add_relation_evidence(ARCScene *scene)
{
    for (size_t i = 0; i < scene->object_count; ++i) {
        for (size_t j = i + 1u; j < scene->object_count; ++j) {
            const ARCObject *lhs = &scene->objects[i];
            const ARCObject *rhs = &scene->objects[j];
            if (lhs->count == rhs->count &&
                lhs->shape_hash == rhs->shape_hash) {
                ++scene->repeated_shape_pairs;
            }
            if (arc_objects_aligned(lhs, rhs)) {
                ++scene->aligned_pairs;
            }
            if (arc_objects_adjacent(lhs, rhs)) {
                ++scene->adjacent_pairs;
            }
        }
    }
}

static int arc_extract_scene(const ARCGrid *grid, uint8_t background, ARCScene *scene)
{
    uint8_t visited[ARC_MAX_GRID_CELLS];
    uint8_t current_mask[ARC_MAX_GRID_CELLS];
    size_t queue[ARC_MAX_GRID_CELLS];
    memset(visited, 0, sizeof(visited));
    arc_scene_clear(scene, background);

    size_t cells = arc_grid_cells(grid);
    for (size_t start = 0; start < cells; ++start) {
        uint8_t color = grid->cells[start];
        if (visited[start] || color == background) {
            continue;
        }
        if (scene->object_count >= ARC_MAX_SCENE_OBJECTS) {
            return 0;
        }

        memset(current_mask, 0, sizeof(current_mask));
        size_t head = 0;
        size_t tail = 0;
        size_t count = 0;
        size_t min_x = grid->width;
        size_t min_y = grid->height;
        size_t max_x = 0;
        size_t max_y = 0;
        double sum_x = 0.0;
        double sum_y = 0.0;

        visited[start] = 1;
        queue[tail++] = start;
        while (head < tail) {
            size_t cell = queue[head++];
            size_t x = cell % grid->width;
            size_t y = cell / grid->width;
            current_mask[cell] = 1;
            ++count;
            sum_x += (double)x;
            sum_y += (double)y;
            if (x < min_x) {
                min_x = x;
            }
            if (y < min_y) {
                min_y = y;
            }
            if (x > max_x) {
                max_x = x;
            }
            if (y > max_y) {
                max_y = y;
            }

            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            for (size_t i = 0; i < sizeof(offsets) / sizeof(offsets[0]); ++i) {
                int nx = (int)x + offsets[i][0];
                int ny = (int)y + offsets[i][1];
                if (nx < 0 || ny < 0 ||
                    nx >= (int)grid->width || ny >= (int)grid->height) {
                    continue;
                }
                size_t next = (size_t)ny * grid->width + (size_t)nx;
                if (visited[next] || grid->cells[next] != color) {
                    continue;
                }
                visited[next] = 1;
                queue[tail++] = next;
            }
        }

        ARCObject *object = &scene->objects[scene->object_count];
        object->id = scene->object_count;
        object->color = color;
        object->count = count;
        object->min_x = min_x;
        object->min_y = min_y;
        object->max_x = max_x;
        object->max_y = max_y;
        object->centroid_x = count == 0 ? 0.0 : sum_x / (double)count;
        object->centroid_y = count == 0 ? 0.0 : sum_y / (double)count;
        object->shape_hash = arc_shape_hash_from_mask(grid,
                                                       current_mask,
                                                       min_x,
                                                       min_y,
                                                       max_x,
                                                       max_y,
                                                       count);
        object->touches_border = min_x == 0 || min_y == 0 ||
                                 max_x + 1u == grid->width ||
                                 max_y + 1u == grid->height;
        ++scene->object_count;
    }

    arc_scene_add_relation_evidence(scene);
    return 1;
}

static int arc_scene_unique_color_object(const ARCScene *scene,
                                         uint8_t color,
                                         ARCObject *object_out)
{
    int found = 0;
    ARCObject object;
    memset(&object, 0, sizeof(object));
    for (size_t i = 0; i < scene->object_count; ++i) {
        if (scene->objects[i].color != color) {
            continue;
        }
        if (found) {
            return 0;
        }
        object = scene->objects[i];
        found = 1;
    }
    if (!found) {
        return 0;
    }
    if (object_out != NULL) {
        *object_out = object;
    }
    return 1;
}

static size_t arc_count_components(const ARCGrid *grid, uint8_t background)
{
    uint8_t visited[ARC_MAX_GRID_CELLS];
    size_t queue[ARC_MAX_GRID_CELLS];
    size_t components = 0;
    memset(visited, 0, sizeof(visited));

    size_t cells = arc_grid_cells(grid);
    for (size_t start = 0; start < cells; ++start) {
        if (visited[start] || grid->cells[start] == background) {
            continue;
        }
        ++components;
        size_t head = 0;
        size_t tail = 0;
        visited[start] = 1;
        queue[tail++] = start;
        while (head < tail) {
            size_t cell = queue[head++];
            size_t x = cell % grid->width;
            size_t y = cell / grid->width;
            const int offsets[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            for (size_t i = 0; i < 4; ++i) {
                int nx = (int)x + offsets[i][0];
                int ny = (int)y + offsets[i][1];
                if (nx < 0 || ny < 0 ||
                    nx >= (int)grid->width || ny >= (int)grid->height) {
                    continue;
                }
                size_t next = (size_t)ny * grid->width + (size_t)nx;
                if (visited[next] || grid->cells[next] == background) {
                    continue;
                }
                visited[next] = 1;
                queue[tail++] = next;
            }
        }
    }
    return components;
}

static size_t arc_count_colors(const ARCGrid *grid)
{
    uint8_t seen[ARC_COLOR_COUNT];
    memset(seen, 0, sizeof(seen));
    size_t cells = arc_grid_cells(grid);
    size_t count = 0;
    for (size_t cell = 0; cell < cells; ++cell) {
        uint8_t color = grid->cells[cell];
        if (color < ARC_COLOR_COUNT && !seen[color]) {
            seen[color] = 1;
            ++count;
        }
    }
    return count;
}

static size_t arc_count_non_background(const ARCGrid *grid, uint8_t background)
{
    size_t cells = arc_grid_cells(grid);
    size_t count = 0;
    for (size_t cell = 0; cell < cells; ++cell) {
        count += grid->cells[cell] != background ? 1u : 0u;
    }
    return count;
}

static double arc_safe_ratio(double numerator, double denominator)
{
    return denominator == 0.0 ? 0.0 : numerator / denominator;
}

static double arc_clamp_signed_unit(double value)
{
    if (value > 1.0) {
        return 1.0;
    }
    if (value < -1.0) {
        return -1.0;
    }
    return value;
}

static void arc_program_task_features_add_grid_pair(ARCProgramTaskFeatures *features,
                                                    const ARCGrid *input,
                                                    const ARCGrid *output)
{
    double input_cells = (double)arc_grid_cells(input);
    double output_cells = (double)arc_grid_cells(output);
    size_t in_min_x = 0;
    size_t in_min_y = 0;
    size_t in_max_x = 0;
    size_t in_max_y = 0;
    size_t out_min_x = 0;
    size_t out_min_y = 0;
    size_t out_max_x = 0;
    size_t out_max_y = 0;
    int in_bbox = arc_find_non_background_bbox(input, 0, &in_min_x, &in_min_y, &in_max_x, &in_max_y);
    int out_bbox =
        arc_find_non_background_bbox(output, 0, &out_min_x, &out_min_y, &out_max_x, &out_max_y);
    double in_bbox_width = in_bbox ? (double)(in_max_x - in_min_x + 1u) : 0.0;
    double in_bbox_height = in_bbox ? (double)(in_max_y - in_min_y + 1u) : 0.0;
    double out_bbox_width = out_bbox ? (double)(out_max_x - out_min_x + 1u) : 0.0;
    double out_bbox_height = out_bbox ? (double)(out_max_y - out_min_y + 1u) : 0.0;
    ARCScene input_scene;
    ARCScene output_scene;
    int have_input_scene = arc_extract_scene(input, 0, &input_scene);
    int have_output_scene = arc_extract_scene(output, 0, &output_scene);

    features->input_width += (double)input->width / (double)ARC_MAX_GRID_WIDTH;
    features->input_height += (double)input->height / (double)ARC_MAX_GRID_HEIGHT;
    features->output_width += (double)output->width / (double)ARC_MAX_GRID_WIDTH;
    features->output_height += (double)output->height / (double)ARC_MAX_GRID_HEIGHT;
    features->width_ratio += arc_safe_ratio((double)output->width, (double)input->width);
    features->height_ratio += arc_safe_ratio((double)output->height, (double)input->height);
    features->area_ratio += arc_safe_ratio(output_cells, input_cells);
    features->same_shape_fraction +=
        input->width == output->width && input->height == output->height ? 1.0 : 0.0;
    features->output_smaller_fraction += output_cells < input_cells ? 1.0 : 0.0;
    features->output_larger_fraction += output_cells > input_cells ? 1.0 : 0.0;
    features->input_color_count += (double)arc_count_colors(input) / (double)ARC_COLOR_COUNT;
    features->output_color_count += (double)arc_count_colors(output) / (double)ARC_COLOR_COUNT;
    features->input_foreground_fraction +=
        arc_safe_ratio((double)arc_count_non_background(input, 0), input_cells);
    features->output_foreground_fraction +=
        arc_safe_ratio((double)arc_count_non_background(output, 0), output_cells);
    features->input_component_count +=
        (double)arc_count_components(input, 0) / (double)ARC_MAX_GRID_CELLS;
    features->output_component_count +=
        (double)arc_count_components(output, 0) / (double)ARC_MAX_GRID_CELLS;
    features->input_bbox_width_fraction += arc_safe_ratio(in_bbox_width, (double)input->width);
    features->input_bbox_height_fraction += arc_safe_ratio(in_bbox_height, (double)input->height);
    features->output_bbox_width_fraction += arc_safe_ratio(out_bbox_width, (double)output->width);
    features->output_bbox_height_fraction += arc_safe_ratio(out_bbox_height, (double)output->height);
    features->bbox_area_ratio +=
        arc_safe_ratio(out_bbox_width * out_bbox_height, in_bbox_width * in_bbox_height);
    features->component_delta +=
        ((double)arc_count_components(output, 0) - (double)arc_count_components(input, 0)) /
        (double)ARC_MAX_GRID_CELLS;
    features->color_delta +=
        ((double)arc_count_colors(output) - (double)arc_count_colors(input)) /
        (double)ARC_COLOR_COUNT;
    features->foreground_delta +=
        arc_safe_ratio((double)arc_count_non_background(output, 0), output_cells) -
        arc_safe_ratio((double)arc_count_non_background(input, 0), input_cells);
    if (have_input_scene) {
        features->input_object_count +=
            (double)input_scene.object_count / (double)ARC_MAX_SCENE_OBJECTS;
        features->input_repeated_shape_pairs +=
            (double)input_scene.repeated_shape_pairs / (double)ARC_MAX_SCENE_RELATION_PAIRS;
        features->input_aligned_pairs +=
            (double)input_scene.aligned_pairs / (double)ARC_MAX_SCENE_RELATION_PAIRS;
        features->input_adjacent_pairs +=
            (double)input_scene.adjacent_pairs / (double)ARC_MAX_SCENE_RELATION_PAIRS;
    }
    if (have_output_scene) {
        features->output_object_count +=
            (double)output_scene.object_count / (double)ARC_MAX_SCENE_OBJECTS;
        features->output_repeated_shape_pairs +=
            (double)output_scene.repeated_shape_pairs / (double)ARC_MAX_SCENE_RELATION_PAIRS;
        features->output_aligned_pairs +=
            (double)output_scene.aligned_pairs / (double)ARC_MAX_SCENE_RELATION_PAIRS;
        features->output_adjacent_pairs +=
            (double)output_scene.adjacent_pairs / (double)ARC_MAX_SCENE_RELATION_PAIRS;
    }
}

static void arc_program_task_features_normalize(ARCProgramTaskFeatures *features, double count)
{
    if (count <= 0.0) {
        return;
    }
#define ARC_NORMALIZE_TASK_FEATURE(field) \
    features->field = arc_clamp_signed_unit(features->field / count)
    ARC_NORMALIZE_TASK_FEATURE(input_width);
    ARC_NORMALIZE_TASK_FEATURE(input_height);
    ARC_NORMALIZE_TASK_FEATURE(output_width);
    ARC_NORMALIZE_TASK_FEATURE(output_height);
    ARC_NORMALIZE_TASK_FEATURE(width_ratio);
    ARC_NORMALIZE_TASK_FEATURE(height_ratio);
    ARC_NORMALIZE_TASK_FEATURE(area_ratio);
    ARC_NORMALIZE_TASK_FEATURE(same_shape_fraction);
    ARC_NORMALIZE_TASK_FEATURE(output_smaller_fraction);
    ARC_NORMALIZE_TASK_FEATURE(output_larger_fraction);
    ARC_NORMALIZE_TASK_FEATURE(input_color_count);
    ARC_NORMALIZE_TASK_FEATURE(output_color_count);
    ARC_NORMALIZE_TASK_FEATURE(input_foreground_fraction);
    ARC_NORMALIZE_TASK_FEATURE(output_foreground_fraction);
    ARC_NORMALIZE_TASK_FEATURE(input_component_count);
    ARC_NORMALIZE_TASK_FEATURE(output_component_count);
    ARC_NORMALIZE_TASK_FEATURE(input_bbox_width_fraction);
    ARC_NORMALIZE_TASK_FEATURE(input_bbox_height_fraction);
    ARC_NORMALIZE_TASK_FEATURE(output_bbox_width_fraction);
    ARC_NORMALIZE_TASK_FEATURE(output_bbox_height_fraction);
    ARC_NORMALIZE_TASK_FEATURE(bbox_area_ratio);
    ARC_NORMALIZE_TASK_FEATURE(component_delta);
    ARC_NORMALIZE_TASK_FEATURE(color_delta);
    ARC_NORMALIZE_TASK_FEATURE(foreground_delta);
    ARC_NORMALIZE_TASK_FEATURE(input_object_count);
    ARC_NORMALIZE_TASK_FEATURE(output_object_count);
    ARC_NORMALIZE_TASK_FEATURE(input_repeated_shape_pairs);
    ARC_NORMALIZE_TASK_FEATURE(output_repeated_shape_pairs);
    ARC_NORMALIZE_TASK_FEATURE(input_aligned_pairs);
    ARC_NORMALIZE_TASK_FEATURE(output_aligned_pairs);
    ARC_NORMALIZE_TASK_FEATURE(input_adjacent_pairs);
    ARC_NORMALIZE_TASK_FEATURE(output_adjacent_pairs);
#undef ARC_NORMALIZE_TASK_FEATURE
}

static void arc_program_task_features(const ARCTask *task, ARCProgramTaskFeatures *features)
{
    memset(features, 0, sizeof(*features));
    for (size_t pair = 0; pair < task->train_count; ++pair) {
        arc_program_task_features_add_grid_pair(features,
                                                &task->train[pair].input,
                                                &task->train[pair].output);
    }
    arc_program_task_features_normalize(features, (double)task->train_count);
}

static void arc_program_task_features_write(const ARCProgramTaskFeatures *features,
                                            nn_real *out)
{
    out[0] = (nn_real)features->input_width;
    out[1] = (nn_real)features->input_height;
    out[2] = (nn_real)features->output_width;
    out[3] = (nn_real)features->output_height;
    out[4] = (nn_real)features->width_ratio;
    out[5] = (nn_real)features->height_ratio;
    out[6] = (nn_real)features->area_ratio;
    out[7] = (nn_real)features->same_shape_fraction;
    out[8] = (nn_real)features->output_smaller_fraction;
    out[9] = (nn_real)features->output_larger_fraction;
    out[10] = (nn_real)features->input_color_count;
    out[11] = (nn_real)features->output_color_count;
    out[12] = (nn_real)features->input_foreground_fraction;
    out[13] = (nn_real)features->output_foreground_fraction;
    out[14] = (nn_real)features->input_component_count;
    out[15] = (nn_real)features->output_component_count;
    out[16] = (nn_real)features->input_bbox_width_fraction;
    out[17] = (nn_real)features->input_bbox_height_fraction;
    out[18] = (nn_real)features->output_bbox_width_fraction;
    out[19] = (nn_real)features->output_bbox_height_fraction;
    out[20] = (nn_real)features->bbox_area_ratio;
    out[21] = (nn_real)features->component_delta;
    out[22] = (nn_real)features->color_delta;
    out[23] = (nn_real)features->foreground_delta;
    out[24] = (nn_real)features->input_object_count;
    out[25] = (nn_real)features->output_object_count;
    out[26] = (nn_real)features->input_repeated_shape_pairs;
    out[27] = (nn_real)features->output_repeated_shape_pairs;
    out[28] = (nn_real)features->input_aligned_pairs;
    out[29] = (nn_real)features->output_aligned_pairs;
    out[30] = (nn_real)features->input_adjacent_pairs;
    out[31] = (nn_real)features->output_adjacent_pairs;
}

static int arc_apply_largest_component_candidate(const ARCTransformCandidate *candidate,
                                                 const ARCGrid *input,
                                                 ARCGrid *output)
{
    if (candidate->selector_kind != ARC_OBJECT_SELECTOR_ALL_COLOR) {
        ARCScene scene;
        ARCObject object;
        if (!arc_extract_scene(input, candidate->background_color, &scene) ||
            !arc_select_scene_object(&scene,
                                     arc_candidate_selector_kind(candidate),
                                     candidate->input_color,
                                     candidate->reference_color,
                                     &object)) {
            return 0;
        }

        if (candidate->op == ARC_RULE_OBJECT_MASK_LARGEST) {
            arc_init_grid(output, input->width, input->height);
            for (size_t cell = 0; cell < arc_grid_cells(output); ++cell) {
                output->cells[cell] = 0;
            }
            for (size_t y = object.min_y; y <= object.max_y; ++y) {
                for (size_t x = object.min_x; x <= object.max_x; ++x) {
                    size_t cell = y * input->width + x;
                    if (input->cells[cell] == candidate->input_color) {
                        output->cells[cell] = candidate->output_color;
                    }
                }
            }
            return 1;
        }

        size_t width = object.max_x - object.min_x + 1u;
        size_t height = object.max_y - object.min_y + 1u;
        arc_init_grid(output, width, height);
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                size_t source = (object.min_y + y) * input->width + object.min_x + x;
                size_t target = y * width + x;
                if (input->cells[source] != candidate->input_color) {
                    output->cells[target] = 0;
                } else if (candidate->op == ARC_RULE_OBJECT_CROP_LARGEST_RECOLOR) {
                    output->cells[target] = candidate->output_color;
                } else {
                    output->cells[target] = input->cells[source];
                }
            }
        }
        return 1;
    }

    ARCComponent component;
    if (!arc_find_largest_component(input, candidate->background_color, &component)) {
        return 0;
    }

    if (candidate->op == ARC_RULE_OBJECT_MASK_LARGEST) {
        arc_init_grid(output, input->width, input->height);
        for (size_t cell = 0; cell < arc_grid_cells(input); ++cell) {
            output->cells[cell] = component.mask[cell] ? candidate->output_color : 0;
        }
        return 1;
    }

    size_t width = component.max_x - component.min_x + 1u;
    size_t height = component.max_y - component.min_y + 1u;
    arc_init_grid(output, width, height);
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            size_t source = (component.min_y + y) * input->width + component.min_x + x;
            size_t target = y * width + x;
            if (!component.mask[source]) {
                output->cells[target] = 0;
            } else if (candidate->op == ARC_RULE_OBJECT_CROP_LARGEST_RECOLOR) {
                output->cells[target] = candidate->output_color;
            } else {
                output->cells[target] = input->cells[source];
            }
        }
    }
    return 1;
}

static int arc_apply_split_panel_candidate(const ARCTransformCandidate *candidate,
                                           const ARCGrid *input,
                                           ARCGrid *output)
{
    if (input->width < 3 || input->height == 0 || (input->width % 2u) == 0) {
        return 0;
    }
    size_t output_width = (input->width - 1u) / 2u;
    size_t output_height = input->height;
    if (output_width == 0 || output_width > ARC_MAX_GRID_WIDTH ||
        output_height > ARC_MAX_GRID_HEIGHT ||
        output_width * output_height > ARC_MAX_GRID_CELLS) {
        return 0;
    }

    arc_init_grid(output, output_width, output_height);
    size_t right_offset = output_width + 1u;
    for (size_t y = 0; y < output_height; ++y) {
        for (size_t x = 0; x < output_width; ++x) {
            uint8_t left = input->cells[y * input->width + x];
            uint8_t right = input->cells[y * input->width + right_offset + x];
            int left_on = left != 0;
            int right_on = right != 0;
            size_t out = y * output_width + x;
            switch (candidate->op) {
            case ARC_RULE_COPY_LEFT:
                output->cells[out] = left_on ? candidate->output_color : 0;
                break;
            case ARC_RULE_COPY_RIGHT:
                output->cells[out] = right_on ? candidate->output_color : 0;
                break;
            case ARC_RULE_INTERSECTION:
                output->cells[out] = left_on && right_on ? candidate->output_color : 0;
                break;
            case ARC_RULE_UNION:
                output->cells[out] = left_on || right_on ? candidate->output_color : 0;
                break;
            case ARC_RULE_OBJECT_SELECT:
                output->cells[out] = left_on && right_on ? candidate->output_color : 0;
                break;
            case ARC_RULE_LAYER_OVERLAP:
            case ARC_RULE_LAYER_UNION:
            case ARC_RULE_LAYER_XOR:
            case ARC_RULE_LAYER_LEFT_ONLY:
            case ARC_RULE_LAYER_RIGHT_ONLY:
                output->cells[out] = arc_layer_relation_value(candidate->op, left_on, right_on)
                                         ? candidate->output_color
                                         : 0;
                break;
            case ARC_RULE_COLOR_MAP_LEFT:
                output->cells[out] = left;
                break;
            default:
                return 0;
            }
        }
    }
    return 1;
}

static int arc_apply_crop_background(const ARCTransformCandidate *candidate,
                                     const ARCGrid *input,
                                     ARCGrid *output,
                                     int recolor)
{
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
    size_t width = max_x - min_x + 1u;
    size_t height = max_y - min_y + 1u;
    arc_init_grid(output, width, height);
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            uint8_t value = input->cells[(min_y + y) * input->width + min_x + x];
            if (recolor) {
                value = value == candidate->background_color ? 0 : candidate->output_color;
            }
            output->cells[y * width + x] = value;
        }
    }
    return 1;
}

static int arc_apply_crop_color(const ARCTransformCandidate *candidate,
                                const ARCGrid *input,
                                ARCGrid *output,
                                int mask_only)
{
    size_t min_x = 0;
    size_t min_y = 0;
    size_t max_x = 0;
    size_t max_y = 0;
    if (!arc_find_color_bbox(input,
                             candidate->input_color,
                             &min_x,
                             &min_y,
                             &max_x,
                             &max_y)) {
        return 0;
    }
    size_t width = max_x - min_x + 1u;
    size_t height = max_y - min_y + 1u;
    arc_init_grid(output, width, height);
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            uint8_t value = input->cells[(min_y + y) * input->width + min_x + x];
            if (mask_only) {
                value = value == candidate->input_color ? candidate->output_color : 0;
            }
            output->cells[y * width + x] = value;
        }
    }
    return 1;
}

static int arc_apply_compress_color_lines(const ARCTransformCandidate *candidate,
                                          const ARCGrid *input,
                                          ARCGrid *output)
{
    int keep_rows[ARC_MAX_GRID_HEIGHT];
    int keep_cols[ARC_MAX_GRID_WIDTH];
    size_t kept_rows = 0;
    size_t kept_cols = 0;
    int remove_rows = candidate->dy != 0;
    int remove_cols = candidate->dx != 0;
    if (!remove_rows && !remove_cols) {
        return 0;
    }

    for (size_t y = 0; y < input->height; ++y) {
        int all_color = 1;
        for (size_t x = 0; x < input->width; ++x) {
            if (input->cells[y * input->width + x] != candidate->background_color) {
                all_color = 0;
                break;
            }
        }
        keep_rows[y] = !(remove_rows && all_color);
        kept_rows += keep_rows[y] ? 1u : 0u;
    }
    for (size_t x = 0; x < input->width; ++x) {
        int all_color = 1;
        for (size_t y = 0; y < input->height; ++y) {
            if (input->cells[y * input->width + x] != candidate->background_color) {
                all_color = 0;
                break;
            }
        }
        keep_cols[x] = !(remove_cols && all_color);
        kept_cols += keep_cols[x] ? 1u : 0u;
    }
    if (kept_rows == 0 || kept_cols == 0 ||
        kept_rows > ARC_MAX_GRID_HEIGHT || kept_cols > ARC_MAX_GRID_WIDTH ||
        kept_rows * kept_cols > ARC_MAX_GRID_CELLS) {
        return 0;
    }

    arc_init_grid(output, kept_cols, kept_rows);
    size_t out_y = 0;
    for (size_t y = 0; y < input->height; ++y) {
        if (!keep_rows[y]) {
            continue;
        }
        size_t out_x = 0;
        for (size_t x = 0; x < input->width; ++x) {
            if (!keep_cols[x]) {
                continue;
            }
            output->cells[out_y * output->width + out_x] =
                input->cells[y * input->width + x];
            ++out_x;
        }
        ++out_y;
    }
    return 1;
}

typedef struct {
    size_t min;
    size_t max;
} ARCRange;

static size_t arc_full_color_rows(const ARCGrid *grid,
                                  uint8_t color,
                                  size_t *rows,
                                  size_t capacity)
{
    size_t count = 0;
    for (size_t y = 0; y < grid->height; ++y) {
        int all_color = 1;
        for (size_t x = 0; x < grid->width; ++x) {
            if (grid->cells[y * grid->width + x] != color) {
                all_color = 0;
                break;
            }
        }
        if (all_color && count < capacity) {
            rows[count++] = y;
        }
    }
    return count;
}

static size_t arc_full_color_cols(const ARCGrid *grid,
                                  uint8_t color,
                                  size_t *cols,
                                  size_t capacity)
{
    size_t count = 0;
    for (size_t x = 0; x < grid->width; ++x) {
        int all_color = 1;
        for (size_t y = 0; y < grid->height; ++y) {
            if (grid->cells[y * grid->width + x] != color) {
                all_color = 0;
                break;
            }
        }
        if (all_color && count < capacity) {
            cols[count++] = x;
        }
    }
    return count;
}

static size_t arc_ranges_from_separators(size_t extent,
                                         const size_t *separators,
                                         size_t separator_count,
                                         ARCRange *ranges,
                                         size_t capacity)
{
    size_t count = 0;
    size_t start = 0;
    for (size_t i = 0; i < separator_count; ++i) {
        size_t separator = separators[i];
        if (separator > start && count < capacity) {
            ranges[count].min = start;
            ranges[count].max = separator - 1u;
            ++count;
        }
        start = separator + 1u;
    }
    if (start < extent && count < capacity) {
        ranges[count].min = start;
        ranges[count].max = extent - 1u;
        ++count;
    }
    return count;
}

static size_t arc_regular_tile_extent(const ARCRange *ranges, size_t count)
{
    if (count == 0) {
        return 0;
    }
    size_t base = ranges[0].max - ranges[0].min + 1u;
    for (size_t i = 0; i < count; ++i) {
        size_t extent = ranges[i].max - ranges[i].min + 1u;
        if (extent != base && !(i + 1u == count && extent <= base)) {
            return 0;
        }
    }
    return base;
}

static int arc_find_tile_motif_lattice(const ARCGrid *input,
                                       uint8_t background,
                                       uint8_t *separator_out,
                                       ARCRange *row_ranges,
                                       size_t *row_count_out,
                                       ARCRange *col_ranges,
                                       size_t *col_count_out,
                                       size_t *tile_height_out,
                                       size_t *tile_width_out)
{
    uint8_t separator = 0;
    size_t found = 0;
    ARCRange found_rows[ARC_MAX_GRID_HEIGHT];
    ARCRange found_cols[ARC_MAX_GRID_WIDTH];
    size_t found_row_count = 0;
    size_t found_col_count = 0;
    size_t found_tile_height = 0;
    size_t found_tile_width = 0;

    for (uint8_t color = 0; color < ARC_COLOR_COUNT; ++color) {
        if (color == background) {
            continue;
        }
        size_t full_rows[ARC_MAX_GRID_HEIGHT];
        size_t full_cols[ARC_MAX_GRID_WIDTH];
        size_t full_row_count =
            arc_full_color_rows(input, color, full_rows, ARC_MAX_GRID_HEIGHT);
        size_t full_col_count =
            arc_full_color_cols(input, color, full_cols, ARC_MAX_GRID_WIDTH);
        if (full_row_count == 0 || full_col_count == 0) {
            continue;
        }

        ARCRange candidate_rows[ARC_MAX_GRID_HEIGHT];
        ARCRange candidate_cols[ARC_MAX_GRID_WIDTH];
        size_t candidate_row_count =
            arc_ranges_from_separators(input->height,
                                       full_rows,
                                       full_row_count,
                                       candidate_rows,
                                       ARC_MAX_GRID_HEIGHT);
        size_t candidate_col_count =
            arc_ranges_from_separators(input->width,
                                       full_cols,
                                       full_col_count,
                                       candidate_cols,
                                       ARC_MAX_GRID_WIDTH);
        size_t tile_height =
            arc_regular_tile_extent(candidate_rows, candidate_row_count);
        size_t tile_width =
            arc_regular_tile_extent(candidate_cols, candidate_col_count);
        if (candidate_row_count < 2 || candidate_col_count < 2 ||
            tile_height == 0 || tile_width == 0) {
            continue;
        }

        separator = color;
        found_row_count = candidate_row_count;
        found_col_count = candidate_col_count;
        found_tile_height = tile_height;
        found_tile_width = tile_width;
        memcpy(found_rows, candidate_rows, sizeof(candidate_rows));
        memcpy(found_cols, candidate_cols, sizeof(candidate_cols));
        ++found;
    }

    if (found != 1) {
        return 0;
    }
    if (separator_out != NULL) {
        *separator_out = separator;
    }
    if (row_ranges != NULL) {
        memcpy(row_ranges, found_rows, sizeof(found_rows));
    }
    if (col_ranges != NULL) {
        memcpy(col_ranges, found_cols, sizeof(found_cols));
    }
    if (row_count_out != NULL) {
        *row_count_out = found_row_count;
    }
    if (col_count_out != NULL) {
        *col_count_out = found_col_count;
    }
    if (tile_height_out != NULL) {
        *tile_height_out = found_tile_height;
    }
    if (tile_width_out != NULL) {
        *tile_width_out = found_tile_width;
    }
    return 1;
}

static int arc_tile_has_motif(const ARCGrid *input,
                              const ARCRange *row_range,
                              const ARCRange *col_range,
                              uint8_t background,
                              uint8_t separator)
{
    for (size_t y = row_range->min; y <= row_range->max; ++y) {
        for (size_t x = col_range->min; x <= col_range->max; ++x) {
            uint8_t value = input->cells[y * input->width + x];
            if (value != background && value != separator) {
                return 1;
            }
        }
    }
    return 0;
}

static int arc_apply_tile_motif_parity_repeat_candidate(const ARCTransformCandidate *candidate,
                                                        const ARCGrid *input,
                                                        ARCGrid *output)
{
    uint8_t separator = 0;
    ARCRange row_ranges[ARC_MAX_GRID_HEIGHT];
    ARCRange col_ranges[ARC_MAX_GRID_WIDTH];
    size_t row_count = 0;
    size_t col_count = 0;
    size_t tile_height = 0;
    size_t tile_width = 0;
    size_t period = candidate->scale == 0 ? 2u : candidate->scale;
    if (period < 2 ||
        !arc_find_tile_motif_lattice(input,
                                     candidate->background_color,
                                     &separator,
                                     row_ranges,
                                     &row_count,
                                     col_ranges,
                                     &col_count,
                                     &tile_height,
                                     &tile_width)) {
        return 0;
    }

    size_t source_row = 0;
    size_t source_col = 0;
    size_t motif_count = 0;
    for (size_t tile_y = 0; tile_y < row_count; ++tile_y) {
        for (size_t tile_x = 0; tile_x < col_count; ++tile_x) {
            if (!arc_tile_has_motif(input,
                                    &row_ranges[tile_y],
                                    &col_ranges[tile_x],
                                    candidate->background_color,
                                    separator)) {
                continue;
            }
            source_row = tile_y;
            source_col = tile_x;
            ++motif_count;
        }
    }
    if (motif_count != 1 ||
        row_ranges[source_row].max - row_ranges[source_row].min + 1u != tile_height ||
        col_ranges[source_col].max - col_ranges[source_col].min + 1u != tile_width) {
        return 0;
    }

    *output = *input;
    int changed = 0;
    const ARCRange *source_y = &row_ranges[source_row];
    const ARCRange *source_x = &col_ranges[source_col];
    for (size_t tile_y = 0; tile_y < row_count; ++tile_y) {
        if (tile_y % period != source_row % period) {
            continue;
        }
        size_t out_height = row_ranges[tile_y].max - row_ranges[tile_y].min + 1u;
        size_t copy_height = out_height < tile_height ? out_height : tile_height;
        for (size_t tile_x = 0; tile_x < col_count; ++tile_x) {
            if (tile_x % period != source_col % period) {
                continue;
            }
            size_t out_width = col_ranges[tile_x].max - col_ranges[tile_x].min + 1u;
            size_t copy_width = out_width < tile_width ? out_width : tile_width;
            for (size_t dy = 0; dy < copy_height; ++dy) {
                for (size_t dx = 0; dx < copy_width; ++dx) {
                    size_t source_cell =
                        (source_y->min + dy) * input->width + source_x->min + dx;
                    size_t output_cell =
                        (row_ranges[tile_y].min + dy) * output->width +
                        col_ranges[tile_x].min + dx;
                    uint8_t value = input->cells[source_cell];
                    if (output->cells[output_cell] != value) {
                        output->cells[output_cell] = value;
                        changed = 1;
                    }
                }
            }
        }
    }
    return changed;
}

static int arc_candidate_matches_train(const ARCTask *t, const ARCTransformCandidate *c, size_t *ch, size_t *mm, size_t *up);
static int arc_build_tile_motif_parity_repeat_candidate(const ARCTask *task,
                                                        uint8_t background,
                                                        size_t period,
                                                        ARCTransformCandidate *candidate)
{
    arc_init_candidate(candidate, ARC_RULE_TILE_MOTIF_PARITY_REPEAT, "");
    candidate->background_color = background;
    candidate->scale = period;
    snprintf(candidate->name,
             sizeof(candidate->name),
             "tile_motif_parity_repeat_bg_%u_period_%zu",
             (unsigned)background,
             period);

    int changed_any = 0;
    for (size_t pair = 0; pair < task->train_count; ++pair) {
        static ARCGrid produced;
        if (!task->train[pair].has_output ||
            !arc_grid_same_shape(&task->train[pair].input, &task->train[pair].output) ||
            !arc_apply_tile_motif_parity_repeat_candidate(candidate,
                                                          &task->train[pair].input,
                                                          &produced) ||
            !arc_grid_equal(&produced, &task->train[pair].output)) {
            return 0;
        }
        changed_any = changed_any || !arc_grid_equal(&task->train[pair].input, &produced);
    }
    return changed_any;
}

static int arc_apply_object_translate_candidate(const ARCTransformCandidate *candidate,
                                                const ARCGrid *input,
                                                ARCGrid *output)
{
    ARCScene scene;
    ARCObject object;
    if (!arc_extract_scene(input, candidate->background_color, &scene) ||
        !arc_select_scene_object(&scene,
                                 arc_candidate_selector_kind(candidate),
                                 candidate->input_color,
                                 candidate->reference_color,
                                 &object)) {
        return 0;
    }

    arc_init_grid(output, input->width, input->height);
    for (size_t cell = 0; cell < arc_grid_cells(output); ++cell) {
        output->cells[cell] = candidate->background_color;
    }
    for (size_t y = object.min_y; y <= object.max_y; ++y) {
        for (size_t x = object.min_x; x <= object.max_x; ++x) {
            size_t cell = y * input->width + x;
            if (input->cells[cell] != candidate->input_color) {
                continue;
            }
            int out_x = (int)x + candidate->dx;
            int out_y = (int)y + candidate->dy;
            if (out_x < 0 || out_y < 0 ||
                out_x >= (int)input->width || out_y >= (int)input->height) {
                return 0;
            }
            output->cells[(size_t)out_y * output->width + (size_t)out_x] =
                candidate->output_color;
        }
    }
    return 1;
}


static int arc_apply_dilate_candidate(const ARCTransformCandidate*c,const ARCGrid*in,ARCGrid*out){
    /* Expand input_color cells to 4-neighbors. Keeps existing non-background. */
    arc_init_grid(out,in->width,in->height);
    for(size_t i=0;i<arc_grid_cells(in);++i)out->cells[i]=in->cells[i];
    uint8_t col=c->input_color;if(!col)col=1;
    for(size_t y=0;y<in->height;++y)for(size_t x=0;x<in->width;++x){
        if(in->cells[y*in->width+x]!=col)continue;
        if(x>0)out->cells[y*in->width+(x-1)]=col;
        if(x+1<in->width)out->cells[y*in->width+(x+1)]=col;
        if(y>0)out->cells[(y-1)*in->width+x]=col;
        if(y+1<in->height)out->cells[(y+1)*in->width+x]=col;
    }
    return 1;
}
static int arc_apply_gravity_candidate(const ARCTransformCandidate*c,const ARCGrid*in,ARCGrid*out){uint8_t bg=c->background_color;arc_init_grid(out,in->width,in->height);if(c->op==ARC_RULE_GRAVITY_DOWN){for(size_t x=0;x<in->width;++x){size_t wr=in->height;for(size_t y=in->height;y>0;--y){size_t ry=y-1;uint8_t v=in->cells[ry*in->width+x];if(v!=bg){--wr;out->cells[wr*in->width+x]=v;}}}}else if(c->op==ARC_RULE_GRAVITY_UP){for(size_t x=0;x<in->width;++x){size_t wr=0;for(size_t y=0;y<in->height;++y){uint8_t v=in->cells[y*in->width+x];if(v!=bg){out->cells[wr*in->width+x]=v;++wr;}}}}else if(c->op==ARC_RULE_GRAVITY_LEFT){for(size_t y=0;y<in->height;++y){size_t wc=0;for(size_t x=0;x<in->width;++x){uint8_t v=in->cells[y*in->width+x];if(v!=bg){out->cells[y*in->width+wc]=v;++wc;}}}}else if(c->op==ARC_RULE_GRAVITY_RIGHT){for(size_t y=0;y<in->height;++y){size_t wc=in->width;for(size_t x=in->width;x>0;--x){size_t rx=x-1;uint8_t v=in->cells[y*in->width+rx];if(v!=bg){--wc;out->cells[y*in->width+wc]=v;}}}}return 1;}
static int arc_apply_tile_repeat_candidate(const ARCTransformCandidate*c,const ARCGrid*in,ARCGrid*out){size_t mx=in->width,my=in->height,Mx=0,My=0;int found=0;for(size_t y=0;y<in->height;++y)for(size_t x=0;x<in->width;++x)if(in->cells[y*in->width+x]!=c->background_color){if(x<mx)mx=x;if(y<my)my=y;if(x>Mx)Mx=x;if(y>My)My=y;found=1;}if(!found){arc_init_grid(out,in->width,in->height);return 1;}size_t tw=Mx-mx+1,th=My-my+1;if(!tw||!th||tw>ARC_MAX_GRID_WIDTH||th>ARC_MAX_GRID_HEIGHT)return 0;if(out->width>ARC_MAX_GRID_WIDTH||out->height>ARC_MAX_GRID_HEIGHT||out->width*out->height>ARC_MAX_GRID_CELLS)return 0;arc_init_grid(out,out->width,out->height);for(size_t y=0;y<out->height;++y)for(size_t x=0;x<out->width;++x)out->cells[y*out->width+x]=in->cells[(my+(y%th))*in->width+(mx+(x%tw))];return 1;}
static int arc_apply_grid_repeat_candidate(const ARCTransformCandidate*c,const ARCGrid*in,ARCGrid*out){(void)c;if(!in->width||!in->height)return 0;if(out->width>ARC_MAX_GRID_WIDTH||out->height>ARC_MAX_GRID_HEIGHT||out->width*out->height>ARC_MAX_GRID_CELLS)return 0;arc_init_grid(out,out->width,out->height);for(size_t y=0;y<out->height;++y)for(size_t x=0;x<out->width;++x)out->cells[y*out->width+x]=in->cells[(y%in->height)*in->width+(x%in->width)];return 1;}
static int arc_marker_motif_select_color(const ARCTransformCandidate*c,const ARCGrid*in,uint8_t*marker_out){if(!c||!in||!marker_out)return 0;if(c->selector_kind==0){*marker_out=c->input_color;return c->input_color<ARC_COLOR_COUNT&&c->input_color!=c->background_color;}size_t count[ARC_COLOR_COUNT]={0};for(size_t i=0;i<arc_grid_cells(in);++i){uint8_t v=in->cells[i];if(v<ARC_COLOR_COUNT&&v!=c->background_color)++count[v];}uint8_t best=ARC_COLOR_COUNT;if(c->selector_kind==1){size_t best_count=0;for(uint8_t v=0;v<ARC_COLOR_COUNT;++v)if(v!=c->background_color&&count[v]>best_count){best_count=count[v];best=v;}}else{size_t best_count=(size_t)-1;for(uint8_t v=0;v<ARC_COLOR_COUNT;++v)if(v!=c->background_color&&count[v]>0&&count[v]<best_count){best_count=count[v];best=v;}}if(best>=ARC_COLOR_COUNT)return 0;*marker_out=best;return 1;}
static int arc_apply_marker_motif_tile_candidate(const ARCTransformCandidate*c,const ARCGrid*in,ARCGrid*out){if(!c||!in||!out||!in->width||!in->height||in->width*in->width>ARC_MAX_GRID_WIDTH||in->height*in->height>ARC_MAX_GRID_HEIGHT)return 0;uint8_t marker=0;if(!arc_marker_motif_select_color(c,in,&marker))return 0;arc_init_grid(out,in->width*in->width,in->height*in->height);for(size_t i=0;i<arc_grid_cells(out);++i)out->cells[i]=c->background_color;int used=0;for(size_t by=0;by<in->height;++by){for(size_t bx=0;bx<in->width;++bx){if(in->cells[by*in->width+bx]!=marker)continue;used=1;for(size_t y=0;y<in->height;++y){for(size_t x=0;x<in->width;++x){size_t out_x=bx*in->width+x;size_t out_y=by*in->height+y;out->cells[out_y*out->width+out_x]=in->cells[y*in->width+x];}}}}return used;}
static int arc_axis_tile_source(size_t coord,size_t span,size_t tiles,uint8_t mode){if(span==0||tiles==0)return -1;size_t tile=coord/span;size_t local=coord%span;if(mode==0||tiles==1)return (int)local;int reverse=(tile&1u)!=0;if(mode==2)reverse=!reverse;return reverse?(int)(span-1u-local):(int)local;}
static int arc_apply_axis_tile_candidate(const ARCTransformCandidate*c,const ARCGrid*in,ARCGrid*out){if(!c||!in||!out||!in->width||!in->height)return 0;size_t sx=c->dx>0?(size_t)c->dx:1u;size_t sy=c->dy>0?(size_t)c->dy:1u;if(sx<1||sy<1||sx>4||sy>4||in->width*sx>ARC_MAX_GRID_WIDTH||in->height*sy>ARC_MAX_GRID_HEIGHT)return 0;arc_init_grid(out,in->width*sx,in->height*sy);for(size_t y=0;y<out->height;++y){int src_y=arc_axis_tile_source(y,in->height,sy,c->render_mode);if(src_y<0)return 0;for(size_t x=0;x<out->width;++x){int src_x=arc_axis_tile_source(x,in->width,sx,c->render_mode);if(src_x<0)return 0;out->cells[y*out->width+x]=in->cells[(size_t)src_y*in->width+(size_t)src_x];}}return 1;}
static size_t arc_row_period(const ARCGrid*in,size_t y){for(size_t p=1;p<=in->width;++p){int ok=1;for(size_t x=0;x<in->width;++x)if(in->cells[y*in->width+x]!=in->cells[y*in->width+(x%p)]){ok=0;break;}if(ok)return p;}return in->width;}
static size_t arc_col_period(const ARCGrid*in,size_t x){for(size_t p=1;p<=in->height;++p){int ok=1;for(size_t y=0;y<in->height;++y)if(in->cells[y*in->width+x]!=in->cells[(y%p)*in->width+x]){ok=0;break;}if(ok)return p;}return in->height;}
static int arc_apply_period_extend_candidate(const ARCTransformCandidate*c,const ARCGrid*in,ARCGrid*out){if(!c||!in||!out||!in->width||!in->height)return 0;size_t sx=c->dx>0?(size_t)c->dx:1u;size_t sy=c->dy>0?(size_t)c->dy:1u;if(!((sx==2&&sy==1)||(sx==1&&sy==2))||in->width*sx>ARC_MAX_GRID_WIDTH||in->height*sy>ARC_MAX_GRID_HEIGHT)return 0;arc_init_grid(out,in->width*sx,in->height*sy);if(sx==2){for(size_t y=0;y<in->height;++y){size_t p=arc_row_period(in,y);for(size_t x=0;x<out->width;++x)out->cells[y*out->width+x]=in->cells[y*in->width+(x%p)];}}else{for(size_t x=0;x<in->width;++x){size_t p=arc_col_period(in,x);for(size_t y=0;y<out->height;++y)out->cells[y*out->width+x]=in->cells[(y%p)*in->width+x];}}return 1;}
static int arc_apply_extract_erase_color_candidate(const ARCTransformCandidate*c,const ARCGrid*in,ARCGrid*out){arc_init_grid(out,in->width,in->height);uint8_t t=c->input_color;if(c->op==ARC_RULE_EXTRACT_COLOR){for(size_t i=0;i<arc_grid_cells(in);++i)out->cells[i]=in->cells[i]==t?t:0;}else{for(size_t i=0;i<arc_grid_cells(in);++i)out->cells[i]=in->cells[i]==t?0:in->cells[i];}return 1;}
static int arc_apply_recolor_by_reference_candidate(const ARCTransformCandidate*c,const ARCGrid*in,ARCGrid*out){uint8_t bg=c->background_color;size_t cnt[10]={0};size_t fp[10];for(size_t i=0;i<10;++i)fp[i]=(size_t)-1;size_t nbg=0;for(size_t i=0;i<arc_grid_cells(in);++i){uint8_t v=in->cells[i];if(v!=bg){if(fp[v]==(size_t)-1)fp[v]=i;++cnt[v];++nbg;}}if(!nbg){arc_init_grid(out,in->width,in->height);return 1;}uint8_t dom=0;size_t mc=0;for(uint8_t i=1;i<10;++i)if(cnt[i]>mc){mc=cnt[i];dom=i;}uint8_t ref=0;size_t bl=(in->height-1)*in->width;uint8_t blc=in->cells[bl];if(blc!=bg&&blc!=dom)ref=blc;else{size_t mn=(size_t)-1;for(uint8_t i=1;i<10;++i)if(i!=dom&&i!=bg&&cnt[i]>0&&cnt[i]<mn){mn=cnt[i];ref=i;}}if(!ref){*out=*in;return 1;}arc_init_grid(out,in->width,in->height);for(size_t i=0;i<arc_grid_cells(in);++i){uint8_t v=in->cells[i];if(v==dom)out->cells[i]=ref;else if(v==ref||v==bg)out->cells[i]=0;else out->cells[i]=v;}return 1;}
static int arc_apply_pattern_extend_candidate(const ARCTransformCandidate*c,const ARCGrid*in,ARCGrid*out){static size_t pos[900][2];size_t n=0;for(size_t y=0;y<in->height&&n<900;++y)for(size_t x=0;x<in->width&&n<900;++x)if(in->cells[y*in->width+x]!=c->background_color){pos[n][0]=x;pos[n][1]=y;++n;}if(n<2){*out=*in;return 1;}int64_t sdx=0,sdy=0;size_t st=0;for(size_t i=0;i+1<n;++i){int64_t dx=(int64_t)pos[i+1][0]-(int64_t)pos[i][0];int64_t dy=(int64_t)pos[i+1][1]-(int64_t)pos[i][1];if(dx>=-10&&dx<=10&&dy>=-10&&dy<=10){sdx+=dx;sdy+=dy;++st;}}if(!st){*out=*in;return 1;}int64_t adx=sdx/(int64_t)st,ady=sdy/(int64_t)st;size_t lx=pos[n-1][0],ly=pos[n-1][1];uint8_t col=in->cells[ly*in->width+lx];arc_init_grid(out,in->width,in->height);for(size_t i=0;i<arc_grid_cells(in);++i)out->cells[i]=in->cells[i];int64_t nx=(int64_t)lx+adx,ny=(int64_t)ly+ady;if(nx>=0&&nx<(int64_t)in->width&&ny>=0&&ny<(int64_t)in->height)out->cells[ny*in->width+nx]=col;return 1;}
static int arc_apply_draw_line_candidate(const ARCTransformCandidate*c,const ARCGrid*in,ARCGrid*out){(void)c;static size_t p[10][900][2];size_t cnt[10]={0};for(size_t y=0;y<in->height;++y)for(size_t x=0;x<in->width;++x){uint8_t v=in->cells[y*in->width+x];if(v&&v<10&&cnt[v]<900){p[v][cnt[v]][0]=x;p[v][cnt[v]][1]=y;++cnt[v];}}arc_init_grid(out,in->width,in->height);for(size_t i=0;i<arc_grid_cells(in);++i)out->cells[i]=in->cells[i];for(uint8_t col=1;col<10;++col){if(cnt[col]<2)continue;int64_t x0=p[col][0][0],y0=p[col][0][1],x1=p[col][cnt[col]-1][0],y1=p[col][cnt[col]-1][1];int64_t dx=x1>x0?1:(x1<x0?-1:0),dy=y1>y0?1:(y1<y0?-1:0);int64_t stp=dx!=0?(x1-x0)*dx:(y1-y0)*dy;for(int64_t s=0;s<=stp;++s){int64_t x=x0+s*dx,y=y0+s*dy;if(x>=0&&x<(int64_t)in->width&&y>=0&&y<(int64_t)in->height)out->cells[y*in->width+x]=col;}}return 1;}
static int arc_apply_symmetry_candidate(const ARCTransformCandidate*c,const ARCGrid*in,ARCGrid*out){arc_init_grid(out,in->width,in->height);if(c->op==ARC_RULE_SYMMETRY_H){for(size_t y=0;y<in->height;++y)for(size_t x=0;x<in->width/2;++x){uint8_t v=in->cells[y*in->width+x];if(v)out->cells[y*in->width+(in->width-1-x)]=v;else{uint8_t r=in->cells[y*in->width+(in->width-1-x)];if(r)out->cells[y*in->width+x]=r;}}}else{for(size_t y=0;y<in->height/2;++y)for(size_t x=0;x<in->width;++x){uint8_t v=in->cells[y*in->width+x];if(v)out->cells[(in->height-1-y)*in->width+x]=v;else{uint8_t r=in->cells[(in->height-1-y)*in->width+x];if(r)out->cells[y*in->width+x]=r;}}}return 1;}
