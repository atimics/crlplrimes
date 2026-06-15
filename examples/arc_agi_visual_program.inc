typedef enum {
    ARC_VISUAL_STEP_COUNT_PALETTE = 0,
    ARC_VISUAL_STEP_SCALE_GRID_BY_COUNT = 1,
    ARC_VISUAL_STEP_SELECT_OBJECT_COLOR = 2,
    ARC_VISUAL_STEP_RENDER_SOLID_RECT = 3,
    ARC_VISUAL_STEP_COUNT_MARKER_CELLS = 4,
    ARC_VISUAL_STEP_SELECT_MOTIF_WINDOW = 5,
    ARC_VISUAL_STEP_REPEAT_WINDOW_BY_COUNT = 6,
    ARC_VISUAL_STEP_COUNT_COLOR = 7,
    ARC_VISUAL_STEP_RENDER_COUNT_LINE = 8,
    ARC_VISUAL_STEP_SELECT_COMPONENT_ORIGINS = 9,
    ARC_VISUAL_STEP_RENDER_OCCUPANCY_GRID = 10,
    ARC_VISUAL_STEP_RENDER_COUNT_TEMPLATE = 11,
    ARC_VISUAL_STEP_RENDER_COMPONENT_SHAPE_GRID = 12,
    ARC_VISUAL_STEP_RENDER_COMPONENT_MASK = 13,
    ARC_VISUAL_STEP_RENDER_ALIGNED_SPANS = 14,
} ARCVisualStepKind;

typedef struct {
    ARCVisualStepKind kind;
} ARCVisualStep;

typedef struct {
    size_t count;
    uint8_t color;
    size_t window_x;
    size_t window_y;
    size_t window_width;
    size_t window_height;
    size_t component_count;
    size_t component_min_x[ARC_MAX_SCENE_OBJECTS];
    size_t component_min_y[ARC_MAX_SCENE_OBJECTS];
    size_t component_max_x[ARC_MAX_SCENE_OBJECTS];
    size_t component_max_y[ARC_MAX_SCENE_OBJECTS];
    size_t component_area[ARC_MAX_SCENE_OBJECTS];
    uint8_t component_color[ARC_MAX_SCENE_OBJECTS];
} ARCVisualState;

static int arc_visual_pipeline_to_sketch_spec(const ARCVisualPipelineSpec *pipeline,
                                              ARCSketchProgramSpec *spec)
{
    if (pipeline == NULL || spec == NULL) {
        return 0;
    }
    memset(spec, 0, sizeof(*spec));
    switch (pipeline->kind) {
    case ARC_VISUAL_PIPELINE_CROP:
        spec->selector = pipeline->selector;
        spec->op = ARC_SKETCH_OP_CROP;
        spec->background = pipeline->background;
        spec->color = pipeline->color;
        spec->output_color = pipeline->output_color;
        spec->x_anchor = pipeline->x_anchor;
        spec->y_anchor = pipeline->y_anchor;
        spec->x_offset = pipeline->x_offset;
        spec->y_offset = pipeline->y_offset;
        spec->width = pipeline->width;
        spec->height = pipeline->height;
        return 1;
    case ARC_VISUAL_PIPELINE_TILE_REPEAT:
        spec->selector = pipeline->selector;
        spec->op = ARC_SKETCH_OP_TILE_REPEAT;
        spec->background = pipeline->background;
        spec->color = pipeline->color;
        spec->output_color = pipeline->output_color;
        spec->x_anchor = pipeline->x_anchor;
        spec->y_anchor = pipeline->y_anchor;
        spec->x_offset = pipeline->x_offset;
        spec->y_offset = pipeline->y_offset;
        spec->width = pipeline->width;
        spec->height = pipeline->height;
        spec->period = pipeline->period;
        return 1;
    case ARC_VISUAL_PIPELINE_MACROCELL_SUBSTITUTE:
        spec->selector = ARC_SKETCH_SELECTOR_LATTICE;
        spec->op = ARC_SKETCH_OP_MACROCELL_SUBSTITUTE;
        spec->background = pipeline->background;
        spec->color = pipeline->color;
        spec->output_color = pipeline->output_color;
        spec->x_anchor = pipeline->x_anchor;
        spec->period = pipeline->period;
        return 1;
    case ARC_VISUAL_PIPELINE_LATTICE_SAMPLE:
        spec->selector = ARC_SKETCH_SELECTOR_LATTICE;
        spec->op = ARC_SKETCH_OP_LATTICE_SAMPLE;
        spec->background = pipeline->background;
        spec->color = pipeline->color;
        spec->output_color = pipeline->output_color;
        spec->x_anchor = pipeline->x_anchor;
        spec->y_anchor = pipeline->y_anchor;
        spec->x_offset = pipeline->x_offset;
        spec->y_offset = pipeline->y_offset;
        spec->width = pipeline->width;
        spec->height = pipeline->height;
        spec->period = pipeline->period;
        return 1;
    case ARC_VISUAL_PIPELINE_COMPONENT_COLOR_SUMMARY:
        spec->selector = ARC_SKETCH_SELECTOR_FOREGROUND_BBOX_PANEL;
        spec->op = ARC_SKETCH_OP_COMPONENT_COLOR_SUMMARY;
        spec->background = pipeline->background;
        spec->color = pipeline->color;
        spec->output_color = pipeline->output_color;
        spec->width = pipeline->width;
        spec->height = pipeline->height;
        return 1;
    case ARC_VISUAL_PIPELINE_NORMALIZED_CROP:
        spec->selector = pipeline->selector;
        spec->op = ARC_SKETCH_OP_NORMALIZED_CROP;
        spec->background = pipeline->background;
        spec->color = pipeline->color;
        spec->output_color = pipeline->output_color;
        spec->x_anchor = pipeline->x_anchor;
        spec->y_anchor = pipeline->y_anchor;
        spec->x_offset = pipeline->x_offset;
        spec->y_offset = pipeline->y_offset;
        spec->width = pipeline->width;
        spec->height = pipeline->height;
        return 1;
    case ARC_VISUAL_PIPELINE_POINT_LINE_PROJECT:
        spec->selector = ARC_SKETCH_SELECTOR_LATTICE;
        spec->op = ARC_SKETCH_OP_POINT_LINE_PROJECT;
        spec->background = pipeline->background;
        spec->color = pipeline->color;
        spec->output_color = pipeline->output_color;
        return 1;
    case ARC_VISUAL_PIPELINE_OBJECT_MASK_EXTRACT:
        spec->selector = pipeline->selector;
        spec->op = ARC_SKETCH_OP_OBJECT_MASK_EXTRACT;
        spec->background = pipeline->background;
        spec->color = pipeline->color;
        spec->output_color = pipeline->output_color;
        spec->x_anchor = pipeline->x_anchor;
        spec->y_anchor = pipeline->y_anchor;
        return 1;
    case ARC_VISUAL_PIPELINE_AXIS_PROJECTION_COMPRESS:
        spec->selector = ARC_SKETCH_SELECTOR_LATTICE;
        spec->op = ARC_SKETCH_OP_AXIS_PROJECTION_COMPRESS;
        spec->background = pipeline->background;
        spec->color = pipeline->color;
        spec->output_color = pipeline->output_color;
        spec->x_anchor = pipeline->x_anchor;
        spec->y_anchor = pipeline->y_anchor;
        return 1;
    case ARC_VISUAL_PIPELINE_SEPARATOR_PANEL_MAP:
        spec->selector = ARC_SKETCH_SELECTOR_GRID_PANEL;
        spec->op = ARC_SKETCH_OP_SEPARATOR_PANEL_MAP;
        spec->background = pipeline->background;
        spec->color = pipeline->color;
        spec->output_color = pipeline->output_color;
        spec->x_anchor = pipeline->x_anchor;
        return 1;
    case ARC_VISUAL_PIPELINE_PALETTE_SCALE:
        spec->selector = ARC_SKETCH_SELECTOR_LATTICE;
        spec->op = ARC_SKETCH_OP_PALETTE_SCALE;
        spec->background = pipeline->background;
        spec->color = pipeline->background;
        spec->output_color = pipeline->background;
        spec->x_anchor = pipeline->include_background ? 0 : 1;
        return 1;
    case ARC_VISUAL_PIPELINE_OBJECT_COLOR_TILE:
        spec->selector = ARC_SKETCH_SELECTOR_FOREGROUND_BBOX_PANEL;
        spec->op = ARC_SKETCH_OP_OBJECT_COLOR_TILE;
        spec->background = pipeline->background;
        spec->color = pipeline->color;
        spec->output_color = pipeline->output_color;
        spec->x_anchor = pipeline->selector_kind;
        spec->width = pipeline->width;
        spec->height = pipeline->height;
        return 1;
    case ARC_VISUAL_PIPELINE_MARKER_COUNT_MOTIF_TILE:
        spec->selector = ARC_SKETCH_SELECTOR_LATTICE;
        spec->op = ARC_SKETCH_OP_MARKER_COUNT_MOTIF_TILE;
        spec->background = pipeline->background;
        spec->color = pipeline->color;
        spec->output_color = pipeline->output_color;
        spec->x_anchor = pipeline->x_align;
        spec->y_anchor = pipeline->y_align;
        spec->width = pipeline->width;
        spec->height = pipeline->height;
        spec->period = (size_t)pipeline->orientation;
        return 1;
    case ARC_VISUAL_PIPELINE_COMPONENT_COUNT_LINE:
        spec->selector = ARC_SKETCH_SELECTOR_LATTICE;
        spec->op = ARC_SKETCH_OP_COMPONENT_COUNT_LINE;
        spec->background = pipeline->background;
        spec->color = pipeline->color;
        spec->output_color = pipeline->output_color;
        spec->x_anchor = pipeline->orientation;
        spec->y_anchor = pipeline->count_mode;
        return 1;
    case ARC_VISUAL_PIPELINE_RECOLOR_CROP:
        spec->selector = pipeline->selector;
        spec->op = ARC_SKETCH_OP_RECOLOR_CROP;
        spec->background = pipeline->background;
        spec->color = pipeline->color;
        spec->output_color = pipeline->output_color;
        spec->x_anchor = pipeline->x_anchor;
        spec->y_anchor = pipeline->y_anchor;
        spec->x_offset = pipeline->x_offset;
        spec->y_offset = pipeline->y_offset;
        spec->width = pipeline->width;
        spec->height = pipeline->height;
        spec->period = pipeline->period;
        return 1;
    case ARC_VISUAL_PIPELINE_MARKER_POLYLINE:
        spec->selector = ARC_SKETCH_SELECTOR_LATTICE;
        spec->op = ARC_SKETCH_OP_MARKER_POLYLINE;
        spec->background = pipeline->background;
        spec->color = pipeline->color;
        spec->output_color = pipeline->output_color;
        spec->x_anchor = pipeline->x_anchor;
        spec->y_anchor = pipeline->y_anchor;
        return 1;
    case ARC_VISUAL_PIPELINE_SEPARATOR_INTERSECTION_COLOR:
        spec->selector = ARC_SKETCH_SELECTOR_GRID_PANEL;
        spec->op = ARC_SKETCH_OP_SEPARATOR_INTERSECTION_COLOR;
        spec->background = pipeline->background;
        spec->color = pipeline->color;
        spec->output_color = pipeline->output_color;
        spec->x_anchor = pipeline->x_anchor;
        spec->period = pipeline->period;
        return 1;
    case ARC_VISUAL_PIPELINE_SEPARATOR_PANEL_RELATION_MASK:
        spec->selector = ARC_SKETCH_SELECTOR_GRID_PANEL;
        spec->op = ARC_SKETCH_OP_SEPARATOR_PANEL_RELATION_MASK;
        spec->background = pipeline->background;
        spec->color = pipeline->color;
        spec->output_color = pipeline->output_color;
        spec->x_anchor = pipeline->x_anchor;
        spec->period = pipeline->period;
        return 1;
    case ARC_VISUAL_PIPELINE_VISUAL_COMBINATOR:
        spec->selector = pipeline->selector;
        spec->op = ARC_SKETCH_OP_VISUAL_COMBINATOR;
        spec->background = pipeline->background;
        spec->color = pipeline->color;
        spec->output_color = pipeline->output_color;
        spec->x_anchor = pipeline->x_anchor;
        spec->y_anchor = pipeline->y_anchor;
        spec->x_offset = pipeline->x_offset;
        spec->y_offset = pipeline->y_offset;
        spec->width = pipeline->width;
        spec->height = pipeline->height;
        spec->period = pipeline->period;
        return 1;
    case ARC_VISUAL_PIPELINE_OBJECT_TRANSLATE:
        spec->selector = pipeline->selector;
        spec->op = ARC_SKETCH_OP_OBJECT_TRANSLATE;
        spec->background = pipeline->background;
        spec->color = pipeline->color;
        spec->output_color = pipeline->output_color;
        spec->x_anchor = pipeline->x_anchor;
        spec->y_anchor = pipeline->y_anchor;
        spec->x_offset = pipeline->x_offset;
        spec->y_offset = pipeline->y_offset;
        spec->period = pipeline->period;
        return 1;
    default:
        return 0;
    }
}

static int arc_visual_pipeline_from_sketch_spec(const ARCSketchProgramSpec *spec,
                                                ARCVisualPipelineSpec *pipeline)
{
    if (spec == NULL || pipeline == NULL) {
        return 0;
    }
    memset(pipeline, 0, sizeof(*pipeline));
    switch (spec->op) {
    case ARC_SKETCH_OP_CROP:
        pipeline->kind = ARC_VISUAL_PIPELINE_CROP;
        break;
    case ARC_SKETCH_OP_TILE_REPEAT:
        pipeline->kind = ARC_VISUAL_PIPELINE_TILE_REPEAT;
        break;
    case ARC_SKETCH_OP_MACROCELL_SUBSTITUTE:
        pipeline->kind = ARC_VISUAL_PIPELINE_MACROCELL_SUBSTITUTE;
        break;
    case ARC_SKETCH_OP_LATTICE_SAMPLE:
        pipeline->kind = ARC_VISUAL_PIPELINE_LATTICE_SAMPLE;
        break;
    case ARC_SKETCH_OP_COMPONENT_COLOR_SUMMARY:
        pipeline->kind = ARC_VISUAL_PIPELINE_COMPONENT_COLOR_SUMMARY;
        break;
    case ARC_SKETCH_OP_NORMALIZED_CROP:
        pipeline->kind = ARC_VISUAL_PIPELINE_NORMALIZED_CROP;
        break;
    case ARC_SKETCH_OP_POINT_LINE_PROJECT:
        pipeline->kind = ARC_VISUAL_PIPELINE_POINT_LINE_PROJECT;
        break;
    case ARC_SKETCH_OP_OBJECT_MASK_EXTRACT:
        pipeline->kind = ARC_VISUAL_PIPELINE_OBJECT_MASK_EXTRACT;
        break;
    case ARC_SKETCH_OP_AXIS_PROJECTION_COMPRESS:
        pipeline->kind = ARC_VISUAL_PIPELINE_AXIS_PROJECTION_COMPRESS;
        break;
    case ARC_SKETCH_OP_SEPARATOR_PANEL_MAP:
        pipeline->kind = ARC_VISUAL_PIPELINE_SEPARATOR_PANEL_MAP;
        break;
    case ARC_SKETCH_OP_PALETTE_SCALE:
        pipeline->kind = ARC_VISUAL_PIPELINE_PALETTE_SCALE;
        pipeline->include_background = spec->x_anchor == 0;
        break;
    case ARC_SKETCH_OP_OBJECT_COLOR_TILE:
        pipeline->kind = ARC_VISUAL_PIPELINE_OBJECT_COLOR_TILE;
        pipeline->selector_kind = (uint8_t)spec->x_anchor;
        break;
    case ARC_SKETCH_OP_MARKER_COUNT_MOTIF_TILE:
        pipeline->kind = ARC_VISUAL_PIPELINE_MARKER_COUNT_MOTIF_TILE;
        pipeline->orientation = (int)spec->period;
        pipeline->x_align = spec->x_anchor;
        pipeline->y_align = spec->y_anchor;
        break;
    case ARC_SKETCH_OP_COMPONENT_COUNT_LINE:
        pipeline->kind = ARC_VISUAL_PIPELINE_COMPONENT_COUNT_LINE;
        pipeline->orientation = spec->x_anchor;
        pipeline->count_mode = spec->y_anchor;
        break;
    case ARC_SKETCH_OP_RECOLOR_CROP:
        pipeline->kind = ARC_VISUAL_PIPELINE_RECOLOR_CROP;
        break;
    case ARC_SKETCH_OP_MARKER_POLYLINE:
        pipeline->kind = ARC_VISUAL_PIPELINE_MARKER_POLYLINE;
        break;
    case ARC_SKETCH_OP_SEPARATOR_INTERSECTION_COLOR:
        pipeline->kind = ARC_VISUAL_PIPELINE_SEPARATOR_INTERSECTION_COLOR;
        break;
    case ARC_SKETCH_OP_SEPARATOR_PANEL_RELATION_MASK:
        pipeline->kind = ARC_VISUAL_PIPELINE_SEPARATOR_PANEL_RELATION_MASK;
        break;
    case ARC_SKETCH_OP_VISUAL_COMBINATOR:
        pipeline->kind = ARC_VISUAL_PIPELINE_VISUAL_COMBINATOR;
        break;
    case ARC_SKETCH_OP_OBJECT_TRANSLATE:
        pipeline->kind = ARC_VISUAL_PIPELINE_OBJECT_TRANSLATE;
        break;
    default:
        return 0;
    }
    pipeline->selector = spec->selector;
    pipeline->background = spec->background;
    pipeline->color = spec->color;
    pipeline->output_color = spec->output_color;
    pipeline->x_anchor = spec->x_anchor;
    pipeline->y_anchor = spec->y_anchor;
    pipeline->x_offset = spec->x_offset;
    pipeline->y_offset = spec->y_offset;
    pipeline->width = spec->width;
    pipeline->height = spec->height;
    pipeline->period = spec->period;
    return 1;
}

static size_t arc_rank_value_scaled(const size_t *values,
                                    size_t count,
                                    size_t value,
                                    size_t extent)
{
    size_t unique[ARC_MAX_SCENE_OBJECTS];
    size_t unique_count = 0;
    size_t rank = 0;
    if (count == 0 || extent == 0) {
        return 0;
    }
    for (size_t i = 0; i < count; ++i) {
        int seen = 0;
        for (size_t j = 0; j < unique_count; ++j) {
            if (unique[j] == values[i]) {
                seen = 1;
                break;
            }
        }
        if (!seen && unique_count < ARC_MAX_SCENE_OBJECTS) {
            unique[unique_count++] = values[i];
        }
    }
    for (size_t i = 0; i + 1u < unique_count; ++i) {
        for (size_t j = i + 1u; j < unique_count; ++j) {
            if (unique[j] < unique[i]) {
                size_t tmp = unique[i];
                unique[i] = unique[j];
                unique[j] = tmp;
            }
        }
    }
    for (size_t i = 0; i < unique_count; ++i) {
        if (unique[i] == value) {
            rank = i;
            break;
        }
    }
    if (unique_count <= 1u) {
        return 0;
    }
    size_t scaled = (rank * (extent - 1u)) / (unique_count - 1u);
    return scaled >= extent ? extent - 1u : scaled;
}

static size_t arc_interval_pack_row(const ARCVisualState *state,
                                    size_t index,
                                    size_t extent)
{
    size_t starts[ARC_MAX_SCENE_OBJECTS];
    size_t ends[ARC_MAX_SCENE_OBJECTS];
    size_t order[ARC_MAX_SCENE_OBJECTS];
    size_t rows[ARC_MAX_SCENE_OBJECTS];
    size_t row_ends[ARC_MAX_SCENE_OBJECTS];
    size_t row_count = 0;
    if (state == NULL || index >= state->component_count || extent == 0) {
        return 0;
    }
    for (size_t i = 0; i < state->component_count; ++i) {
        starts[i] = state->component_min_y[i];
        ends[i] = state->component_max_y[i];
        order[i] = i;
        rows[i] = 0;
    }
    for (size_t i = 0; i + 1u < state->component_count; ++i) {
        for (size_t j = i + 1u; j < state->component_count; ++j) {
            if (starts[order[j]] < starts[order[i]] ||
                (starts[order[j]] == starts[order[i]] &&
                 state->component_min_x[order[j]] < state->component_min_x[order[i]])) {
                size_t tmp = order[i];
                order[i] = order[j];
                order[j] = tmp;
            }
        }
    }
    for (size_t cursor = 0; cursor < state->component_count; ++cursor) {
        size_t object_index = order[cursor];
        size_t assigned_row = row_count;
        for (size_t row = 0; row < row_count; ++row) {
            if (starts[object_index] > row_ends[row] + 1u) {
                assigned_row = row;
                break;
            }
        }
        if (assigned_row == row_count) {
            if (row_count >= ARC_MAX_SCENE_OBJECTS) {
                return 0;
            }
            row_ends[row_count++] = ends[object_index];
        } else if (ends[object_index] > row_ends[assigned_row]) {
            row_ends[assigned_row] = ends[object_index];
        }
        rows[object_index] = assigned_row;
    }
    if (row_count == 0) {
        return 0;
    }
    size_t scaled = row_count <= 1u
                        ? 0u
                        : (rows[index] * (extent - 1u)) / (row_count - 1u);
    return scaled >= extent ? extent - 1u : scaled;
}

static int arc_visual_state_component_bounds(const ARCVisualState *state,
                                             size_t *min_x_out,
                                             size_t *min_y_out,
                                             size_t *max_x_out,
                                             size_t *max_y_out)
{
    if (state == NULL || state->component_count == 0) {
        return 0;
    }
    size_t min_x = state->component_min_x[0];
    size_t min_y = state->component_min_y[0];
    size_t max_x = state->component_max_x[0];
    size_t max_y = state->component_max_y[0];
    for (size_t i = 1; i < state->component_count; ++i) {
        if (state->component_min_x[i] < min_x) {
            min_x = state->component_min_x[i];
        }
        if (state->component_min_y[i] < min_y) {
            min_y = state->component_min_y[i];
        }
        if (state->component_max_x[i] > max_x) {
            max_x = state->component_max_x[i];
        }
        if (state->component_max_y[i] > max_y) {
            max_y = state->component_max_y[i];
        }
    }
    *min_x_out = min_x;
    *min_y_out = min_y;
    *max_x_out = max_x;
    *max_y_out = max_y;
    return 1;
}

static int arc_visual_count_template_position(size_t pattern,
                                              size_t index,
                                              size_t width,
                                              size_t height,
                                              size_t *x_out,
                                              size_t *y_out)
{
    if (width == 0 || height == 0 || x_out == NULL || y_out == NULL) {
        return 0;
    }
    if (pattern == 1u && width >= 3u && height >= 3u) {
        static const uint8_t corner_center_order[][2] = {
            {0, 0},
            {2, 0},
            {1, 1},
            {0, 2},
            {2, 2},
        };
        if (index < sizeof(corner_center_order) / sizeof(corner_center_order[0])) {
            size_t px = corner_center_order[index][0];
            size_t py = corner_center_order[index][1];
            *x_out = (px * (width - 1u)) / 2u;
            *y_out = (py * (height - 1u)) / 2u;
            return 1;
        }
    } else if (pattern == 2u && width >= 3u && height >= 3u) {
        size_t row = index / 2u;
        size_t side = index % 2u;
        if (row < height) {
            *x_out = side == 0u ? 0u : width - 1u;
            *y_out = row;
            return 1;
        }
    } else if (pattern == 3u && width >= 3u && height >= 3u) {
        size_t diag = index / 2u;
        size_t side = index % 2u;
        if (diag < width && diag < height) {
            *x_out = side == 0u ? diag : width - 1u - diag;
            *y_out = diag;
            return 1;
        }
    }
    if (index >= width * height) {
        return 0;
    }
    *x_out = index % width;
    *y_out = index / width;
    return 1;
}

static size_t arc_visual_component_row_color_count(const ARCGrid *input,
                                                   const ARCVisualState *state,
                                                   size_t index,
                                                   size_t y)
{
    if (input == NULL || state == NULL || index >= state->component_count ||
        y >= input->height) {
        return 0;
    }
    size_t min_x = state->component_min_x[index];
    size_t max_x = state->component_max_x[index];
    uint8_t color = state->component_color[index];
    size_t count = 0;
    if (max_x >= input->width) {
        max_x = input->width - 1u;
    }
    for (size_t x = min_x; x <= max_x; ++x) {
        if (input->cells[y * input->width + x] == color) {
            ++count;
        }
    }
    return count;
}

static size_t arc_visual_component_col_color_count(const ARCGrid *input,
                                                   const ARCVisualState *state,
                                                   size_t index,
                                                   size_t x)
{
    if (input == NULL || state == NULL || index >= state->component_count ||
        x >= input->width) {
        return 0;
    }
    size_t min_y = state->component_min_y[index];
    size_t max_y = state->component_max_y[index];
    uint8_t color = state->component_color[index];
    size_t count = 0;
    if (max_y >= input->height) {
        max_y = input->height - 1u;
    }
    for (size_t y = min_y; y <= max_y; ++y) {
        if (input->cells[y * input->width + x] == color) {
            ++count;
        }
    }
    return count;
}

static uint8_t arc_visual_component_classifier_color(const ARCSketchProgramSpec *spec,
                                                     const ARCGrid *input,
                                                     const ARCVisualState *state,
                                                     size_t index)
{
    size_t width = state->component_max_x[index] - state->component_min_x[index] + 1u;
    size_t height = state->component_max_y[index] - state->component_min_y[index] + 1u;
    size_t bbox_area = width * height;
    uint8_t primary = spec->output_color == spec->background
                          ? state->component_color[index]
                          : spec->output_color;
    uint8_t secondary = (spec->x_offset >= 0 && spec->x_offset < ARC_COLOR_COUNT)
                            ? (uint8_t)spec->x_offset
                            : spec->color;
    if (primary == spec->background) {
        primary = state->component_color[index];
    }
    if (secondary == spec->background) {
        secondary = spec->color == spec->background
                        ? state->component_color[index]
                        : spec->color;
    }
    switch (spec->period) {
    case 0:
        return primary;
    case 1:
        return state->component_color[index];
    case 2:
        return width >= height ? primary : secondary;
    case 3:
        return height >= width ? primary : secondary;
    case 4:
        return bbox_area > 0 && state->component_area[index] * 2u >= bbox_area
                   ? primary
                   : secondary;
    case 5: {
        size_t top = arc_visual_component_row_color_count(input,
                                                          state,
                                                          index,
                                                          state->component_min_y[index]);
        size_t bottom = arc_visual_component_row_color_count(input,
                                                             state,
                                                             index,
                                                             state->component_max_y[index]);
        return top >= bottom ? primary : secondary;
    }
    case 6: {
        size_t left = arc_visual_component_col_color_count(input,
                                                           state,
                                                           index,
                                                           state->component_min_x[index]);
        size_t right = arc_visual_component_col_color_count(input,
                                                            state,
                                                            index,
                                                            state->component_max_x[index]);
        return left >= right ? primary : secondary;
    }
    default:
        return primary;
    }
}

static size_t arc_visual_select_component_index(const ARCVisualState *state,
                                                int selector_mode)
{
    size_t best = SIZE_MAX;
    if (state == NULL || state->component_count == 0) {
        return SIZE_MAX;
    }
    best = 0;
    for (size_t i = 1; i < state->component_count; ++i) {
        size_t best_width =
            state->component_max_x[best] - state->component_min_x[best] + 1u;
        size_t best_height =
            state->component_max_y[best] - state->component_min_y[best] + 1u;
        size_t width = state->component_max_x[i] - state->component_min_x[i] + 1u;
        size_t height = state->component_max_y[i] - state->component_min_y[i] + 1u;
        size_t best_area = state->component_area[best];
        size_t area = state->component_area[i];
        int replace = 0;
        switch (selector_mode) {
        case 4:
            replace = area < best_area ||
                      (area == best_area && state->component_min_y[i] < state->component_min_y[best]) ||
                      (area == best_area && state->component_min_y[i] == state->component_min_y[best] &&
                       state->component_min_x[i] < state->component_min_x[best]);
            break;
        case 5:
            replace = width * height < best_width * best_height ||
                      (width * height == best_width * best_height &&
                       state->component_area[i] < state->component_area[best]);
            break;
        case 2:
        case 3:
        default:
            replace = area > best_area ||
                      (area == best_area && state->component_min_y[i] < state->component_min_y[best]) ||
                      (area == best_area && state->component_min_y[i] == state->component_min_y[best] &&
                       state->component_min_x[i] < state->component_min_x[best]);
            break;
        }
        if (replace) {
            best = i;
        }
    }
    return best;
}

static size_t arc_visual_component_width(const ARCVisualState *state,
                                         size_t index)
{
    if (state == NULL || index >= state->component_count ||
        state->component_max_x[index] < state->component_min_x[index]) {
        return 0;
    }
    return state->component_max_x[index] - state->component_min_x[index] + 1u;
}

static size_t arc_visual_component_height(const ARCVisualState *state,
                                          size_t index)
{
    if (state == NULL || index >= state->component_count ||
        state->component_max_y[index] < state->component_min_y[index]) {
        return 0;
    }
    return state->component_max_y[index] - state->component_min_y[index] + 1u;
}

static int arc_visual_component_mask_group_mode(int y_anchor)
{
    return y_anchor == 0 || y_anchor == 1 || y_anchor == 6 ||
           (y_anchor >= 7 && y_anchor <= 18);
}

static int arc_visual_component_mask_scaled_mode(int y_anchor)
{
    return y_anchor == 1 || y_anchor == 3;
}

static int arc_visual_component_mask_requires_exact_shape(int y_anchor)
{
    return y_anchor == 0 || y_anchor == 2 || y_anchor == 4 ||
           y_anchor == 7 || y_anchor == 8;
}

static size_t arc_visual_component_color_frequency(const ARCVisualState *state,
                                                   uint8_t color)
{
    size_t count = 0;
    if (state == NULL) {
        return 0;
    }
    for (size_t i = 0; i < state->component_count; ++i) {
        if (state->component_color[i] == color) {
            ++count;
        }
    }
    return count;
}

static int arc_visual_component_mask_selects_index(
    const ARCVisualState *state,
    const ARCSketchProgramSpec *spec,
    size_t index,
    size_t anchor_index)
{
    size_t width;
    size_t height;
    size_t anchor_width;
    size_t anchor_height;
    if (state == NULL || spec == NULL || index >= state->component_count ||
        anchor_index >= state->component_count) {
        return 0;
    }
    switch (spec->y_anchor) {
    case 0:
    case 1:
    case 6:
        return 1;
    case 7:
        width = arc_visual_component_width(state, index);
        height = arc_visual_component_height(state, index);
        return width == spec->width && height == spec->height;
    case 8:
        width = arc_visual_component_width(state, index);
        height = arc_visual_component_height(state, index);
        return width == spec->height && height == spec->width;
    case 9:
    case 13:
        width = arc_visual_component_width(state, index);
        height = arc_visual_component_height(state, index);
        anchor_width = arc_visual_component_width(state, anchor_index);
        anchor_height = arc_visual_component_height(state, anchor_index);
        return width == anchor_width && height == anchor_height;
    case 10:
    case 14:
        return state->component_area[index] ==
               state->component_area[anchor_index];
    case 11:
    case 15:
        return state->component_color[index] ==
               state->component_color[anchor_index];
    case 12:
    case 16:
        return arc_visual_component_color_frequency(
                   state,
                   state->component_color[index]) == 1u;
    case 17:
        width = arc_visual_component_width(state, index);
        height = arc_visual_component_height(state, index);
        return width == spec->width && height == spec->height;
    case 18:
        width = arc_visual_component_width(state, index);
        height = arc_visual_component_height(state, index);
        return width == spec->height && height == spec->width;
    default:
        return index == anchor_index;
    }
}

static int arc_visual_marker_signal_index(const ARCVisualState *state,
                                          const ARCSketchProgramSpec *spec,
                                          size_t *index_out)
{
    size_t anchor_index;
    size_t best = SIZE_MAX;
    if (state == NULL || spec == NULL || index_out == NULL ||
        state->component_count < 2) {
        return 0;
    }
    anchor_index = arc_visual_select_component_index(state, spec->y_anchor);
    if (anchor_index == SIZE_MAX) {
        return 0;
    }
    for (size_t i = 0; i < state->component_count; ++i) {
        if (arc_visual_component_mask_selects_index(state, spec, i, anchor_index)) {
            continue;
        }
        if (best == SIZE_MAX ||
            state->component_area[i] < state->component_area[best] ||
            (state->component_area[i] == state->component_area[best] &&
             state->component_min_y[i] < state->component_min_y[best]) ||
            (state->component_area[i] == state->component_area[best] &&
             state->component_min_y[i] == state->component_min_y[best] &&
             state->component_min_x[i] < state->component_min_x[best])) {
            best = i;
        }
    }
    if (best == SIZE_MAX) {
        return 0;
    }
    *index_out = best;
    return 1;
}

static int arc_visual_marker_glyph_key(const ARCGrid *input,
                                       const ARCVisualState *state,
                                       size_t index,
                                       unsigned *key_out)
{
    size_t width;
    size_t height;
    unsigned top_mask = 0;
    if (input == NULL || state == NULL || key_out == NULL ||
        index >= state->component_count) {
        return 0;
    }
    width = arc_visual_component_width(state, index);
    height = arc_visual_component_height(state, index);
    if (width == 0 || height == 0 || width > 3u || height > 3u ||
        state->component_min_y[index] >= input->height) {
        return 0;
    }
    for (size_t x = 0; x < width; ++x) {
        size_t grid_x = state->component_min_x[index] + x;
        size_t grid_y = state->component_min_y[index];
        if (grid_x >= input->width) {
            continue;
        }
        if (input->cells[grid_y * input->width + grid_x] ==
            state->component_color[index]) {
            top_mask |= 1u << (2u - (unsigned)x);
        }
    }
    if (top_mask == 0 || top_mask >= 8u) {
        return 0;
    }
    *key_out = top_mask;
    return 1;
}

static int arc_visual_marker_color_glyph_key(const ARCGrid *input,
                                             const ARCSketchProgramSpec *spec,
                                             unsigned *key_out)
{
    int found = 0;
    uint8_t best_color = 0;
    size_t best_count = 0;
    size_t best_min_x = 0;
    size_t best_min_y = 0;
    size_t best_max_x = 0;
    if (input == NULL || spec == NULL || key_out == NULL) {
        return 0;
    }
    for (uint8_t color = 0; color < ARC_COLOR_COUNT; ++color) {
        size_t count = 0;
        size_t min_x = input->width;
        size_t min_y = input->height;
        size_t max_x = 0;
        size_t max_y = 0;
        size_t width;
        size_t height;
        if (color == spec->background ||
            (spec->color != spec->background && color == spec->color)) {
            continue;
        }
        for (size_t y = 0; y < input->height; ++y) {
            for (size_t x = 0; x < input->width; ++x) {
                if (input->cells[y * input->width + x] != color) {
                    continue;
                }
                if (count == 0) {
                    min_x = max_x = x;
                    min_y = max_y = y;
                } else {
                    if (x < min_x) {
                        min_x = x;
                    }
                    if (x > max_x) {
                        max_x = x;
                    }
                    if (y < min_y) {
                        min_y = y;
                    }
                    if (y > max_y) {
                        max_y = y;
                    }
                }
                ++count;
            }
        }
        if (count == 0) {
            continue;
        }
        width = max_x - min_x + 1u;
        height = max_y - min_y + 1u;
        if (width == 0 || height == 0 || width > 3u || height > 3u) {
            continue;
        }
        if (!found || count < best_count ||
            (count == best_count && min_y < best_min_y) ||
            (count == best_count && min_y == best_min_y && min_x < best_min_x)) {
            found = 1;
            best_color = color;
            best_count = count;
            best_min_x = min_x;
            best_min_y = min_y;
            best_max_x = max_x;
        }
    }
    if (found) {
        unsigned top_mask = 0;
        for (size_t x = best_min_x; x <= best_max_x; ++x) {
            if (input->cells[best_min_y * input->width + x] == best_color) {
                top_mask |= 1u << (2u - (unsigned)(x - best_min_x));
            }
        }
        if (top_mask > 0 && top_mask < 8u) {
            *key_out = top_mask;
            return 1;
        }
    }
    return 0;
}

static uint8_t arc_visual_marker_glyph_map_color(
    const ARCSketchProgramSpec *spec,
    const ARCGrid *input,
    const ARCVisualState *state)
{
    size_t marker_index = SIZE_MAX;
    unsigned key = 0;
    uint32_t packed;
    unsigned color;
    if (spec == NULL || input == NULL || state == NULL || spec->x_offset <= 0) {
        return spec != NULL ? spec->output_color : 0;
    }
    if (!arc_visual_marker_color_glyph_key(input, spec, &key) &&
        (!arc_visual_marker_signal_index(state, spec, &marker_index) ||
         !arc_visual_marker_glyph_key(input, state, marker_index, &key))) {
        return spec->output_color;
    }
    packed = (uint32_t)spec->x_offset;
    color = (packed >> (key * 4u)) & 0xFu;
    if (color == 0 || color >= ARC_COLOR_COUNT) {
        return spec->output_color;
    }
    return (uint8_t)color;
}

static int arc_visual_component_contains_cell(const ARCGrid *input,
                                              const ARCVisualState *state,
                                              size_t index,
                                              size_t x,
                                              size_t y)
{
    if (input == NULL || state == NULL || index >= state->component_count ||
        x >= input->width || y >= input->height ||
        x < state->component_min_x[index] || x > state->component_max_x[index] ||
        y < state->component_min_y[index] || y > state->component_max_y[index]) {
        return 0;
    }
    return input->cells[y * input->width + x] == state->component_color[index];
}

static int arc_visual_select_components_into_state(const ARCGrid *input,
                                                   const ARCSketchProgramSpec *spec,
                                                   ARCVisualState *state)
{
    ARCScene scene;
    if (input == NULL || spec == NULL || state == NULL ||
        !arc_extract_scene(input, spec->background, &scene)) {
        return 0;
    }
    state->component_count = 0;
    for (size_t i = 0; i < scene.object_count; ++i) {
        const ARCObject *object = &scene.objects[i];
        if (object->color == spec->background) {
            continue;
        }
        if (spec->color != spec->background && object->color != spec->color) {
            continue;
        }
        if (state->component_count >= ARC_MAX_SCENE_OBJECTS) {
            return 0;
        }
        state->component_min_x[state->component_count] = object->min_x;
        state->component_min_y[state->component_count] = object->min_y;
        state->component_max_x[state->component_count] = object->max_x;
        state->component_max_y[state->component_count] = object->max_y;
        state->component_area[state->component_count] = object->count;
        state->component_color[state->component_count] = object->color;
        ++state->component_count;
    }
    return state->component_count > 0;
}

static size_t arc_visual_component_at_cell(const ARCGrid *input,
                                           const ARCVisualState *state,
                                           const ARCSketchProgramSpec *spec,
                                           size_t x,
                                           size_t y,
                                           size_t selected_index)
{
    if (state == NULL || spec == NULL) {
        return SIZE_MAX;
    }
    if (!arc_visual_component_mask_group_mode(spec->y_anchor)) {
        return arc_visual_component_contains_cell(input, state, selected_index, x, y)
                   ? selected_index
                   : SIZE_MAX;
    }
    for (size_t i = 0; i < state->component_count; ++i) {
        if (!arc_visual_component_mask_selects_index(state, spec, i, selected_index)) {
            continue;
        }
        if (arc_visual_component_contains_cell(input, state, i, x, y)) {
            return i;
        }
    }
    return SIZE_MAX;
}

static uint8_t arc_visual_component_mask_color(const ARCSketchProgramSpec *spec,
                                               const ARCGrid *input,
                                               const ARCVisualState *state,
                                               size_t index)
{
    if (spec->period == 1u) {
        return state->component_color[index];
    }
    if (spec->period == 2u) {
        return arc_visual_component_classifier_color(spec, input, state, index);
    }
    if (spec->period == 3u) {
        return arc_visual_marker_glyph_map_color(spec, input, state);
    }
    if (spec->output_color != spec->background) {
        return spec->output_color;
    }
    return state->component_color[index];
}

static int arc_visual_component_mask_source_box(const ARCGrid *input,
                                                const ARCVisualState *state,
                                                const ARCSketchProgramSpec *spec,
                                                size_t selected_index,
                                                size_t *min_x,
                                                size_t *min_y,
                                                size_t *max_x,
                                                size_t *max_y)
{
    int found = 0;
    if (input == NULL || state == NULL || spec == NULL ||
        selected_index >= state->component_count ||
        min_x == NULL || min_y == NULL || max_x == NULL || max_y == NULL) {
        return 0;
    }
    if (spec->y_anchor == 6) {
        if (input->width == 0 || input->height == 0) {
            return 0;
        }
        *min_x = 0;
        *min_y = 0;
        *max_x = input->width - 1u;
        *max_y = input->height - 1u;
        return 1;
    }
    if (spec->y_anchor >= 13 && spec->y_anchor <= 18) {
        for (size_t i = 0; i < state->component_count; ++i) {
            if (arc_visual_component_mask_selects_index(state, spec, i, selected_index)) {
                found = 1;
                break;
            }
        }
        if (!found || input->width == 0 || input->height == 0) {
            return 0;
        }
        *min_x = 0;
        *min_y = 0;
        *max_x = input->width - 1u;
        *max_y = input->height - 1u;
        return 1;
    }
    if (!arc_visual_component_mask_group_mode(spec->y_anchor)) {
        *min_x = state->component_min_x[selected_index];
        *min_y = state->component_min_y[selected_index];
        *max_x = state->component_max_x[selected_index];
        *max_y = state->component_max_y[selected_index];
        return 1;
    }
    for (size_t i = 0; i < state->component_count; ++i) {
        if (!arc_visual_component_mask_selects_index(state, spec, i, selected_index)) {
            continue;
        }
        if (!found) {
            *min_x = state->component_min_x[i];
            *min_y = state->component_min_y[i];
            *max_x = state->component_max_x[i];
            *max_y = state->component_max_y[i];
            found = 1;
            continue;
        }
        if (state->component_min_x[i] < *min_x) {
            *min_x = state->component_min_x[i];
        }
        if (state->component_min_y[i] < *min_y) {
            *min_y = state->component_min_y[i];
        }
        if (state->component_max_x[i] > *max_x) {
            *max_x = state->component_max_x[i];
        }
        if (state->component_max_y[i] > *max_y) {
            *max_y = state->component_max_y[i];
        }
    }
    return found;
}

static int arc_visual_component_mask_render_shape(size_t source_width,
                                                  size_t source_height,
                                                  int mode,
                                                  size_t *width_out,
                                                  size_t *height_out)
{
    if (source_width == 0 || source_height == 0 ||
        width_out == NULL || height_out == NULL) {
        return 0;
    }
    if (mode == 1 || mode == 3) {
        *width_out = source_height;
        *height_out = source_width;
    } else {
        *width_out = source_width;
        *height_out = source_height;
    }
    return 1;
}

static int arc_visual_component_mask_unmap(size_t display_x,
                                           size_t display_y,
                                           size_t source_width,
                                           size_t source_height,
                                           int mode,
                                           size_t *source_x_out,
                                           size_t *source_y_out)
{
    if (source_x_out == NULL || source_y_out == NULL ||
        source_width == 0 || source_height == 0) {
        return 0;
    }
    switch (mode) {
    case 1:
        if (display_x >= source_height || display_y >= source_width) {
            return 0;
        }
        *source_x_out = display_y;
        *source_y_out = source_height - 1u - display_x;
        return 1;
    case 2:
        if (display_x >= source_width || display_y >= source_height) {
            return 0;
        }
        *source_x_out = source_width - 1u - display_x;
        *source_y_out = source_height - 1u - display_y;
        return 1;
    case 3:
        if (display_x >= source_height || display_y >= source_width) {
            return 0;
        }
        *source_x_out = source_width - 1u - display_y;
        *source_y_out = display_x;
        return 1;
    case 4:
        if (display_x >= source_width || display_y >= source_height) {
            return 0;
        }
        *source_x_out = source_width - 1u - display_x;
        *source_y_out = display_y;
        return 1;
    case 5:
        if (display_x >= source_width || display_y >= source_height) {
            return 0;
        }
        *source_x_out = display_x;
        *source_y_out = source_height - 1u - display_y;
        return 1;
    default:
        if (display_x >= source_width || display_y >= source_height) {
            return 0;
        }
        *source_x_out = display_x;
        *source_y_out = display_y;
        return 1;
    }
}

static int arc_visual_component_mask_neighbor_active(const uint8_t *mask,
                                                     size_t width,
                                                     size_t height,
                                                     size_t x,
                                                     size_t y)
{
    if (mask == NULL || width == 0 || height == 0 ||
        x >= width || y >= height) {
        return 0;
    }
    return mask[y * width + x] != 0;
}

static void arc_visual_component_mask_apply_style(const uint8_t *source,
                                                  const size_t *source_component,
                                                  size_t selected_index,
                                                  size_t width,
                                                  size_t height,
                                                  int mode,
                                                  uint8_t *styled,
                                                  size_t *styled_component)
{
    if (source == NULL || source_component == NULL || styled == NULL ||
        styled_component == NULL || width == 0 || height == 0 ||
        width * height > ARC_MAX_GRID_CELLS) {
        return;
    }
    for (size_t i = 0; i < width * height; ++i) {
        styled[i] = source[i];
        styled_component[i] = source_component[i];
    }
    if (mode == 6) {
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                size_t index = y * width + x;
                int edge = x == 0 || y == 0 || x + 1u == width || y + 1u == height;
                if (!source[index]) {
                    styled[index] = 0;
                    continue;
                }
                if (!edge &&
                    arc_visual_component_mask_neighbor_active(source, width, height, x - 1u, y) &&
                    arc_visual_component_mask_neighbor_active(source, width, height, x + 1u, y) &&
                    arc_visual_component_mask_neighbor_active(source, width, height, x, y - 1u) &&
                    arc_visual_component_mask_neighbor_active(source, width, height, x, y + 1u)) {
                    styled[index] = 0;
                }
            }
        }
    } else if (mode == 7) {
        uint8_t exterior[ARC_MAX_GRID_CELLS];
        size_t queue[ARC_MAX_GRID_CELLS];
        size_t head = 0;
        size_t tail = 0;
        memset(exterior, 0, sizeof(exterior));
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                if (!(x == 0 || y == 0 || x + 1u == width || y + 1u == height)) {
                    continue;
                }
                size_t index = y * width + x;
                if (!source[index] && !exterior[index]) {
                    exterior[index] = 1;
                    queue[tail++] = index;
                }
            }
        }
        while (head < tail) {
            size_t index = queue[head++];
            size_t x = index % width;
            size_t y = index / width;
            const int dx[4] = {-1, 1, 0, 0};
            const int dy[4] = {0, 0, -1, 1};
            for (size_t n = 0; n < 4; ++n) {
                int nx = (int)x + dx[n];
                int ny = (int)y + dy[n];
                if (nx < 0 || ny < 0 ||
                    nx >= (int)width || ny >= (int)height) {
                    continue;
                }
                size_t next = (size_t)ny * width + (size_t)nx;
                if (!source[next] && !exterior[next]) {
                    exterior[next] = 1;
                    queue[tail++] = next;
                }
            }
        }
        for (size_t i = 0; i < width * height; ++i) {
            if (!source[i] && !exterior[i]) {
                styled[i] = 1;
                styled_component[i] = selected_index;
            }
        }
    } else if (mode == 8) {
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                size_t index = y * width + x;
                int edge = x == 0 || y == 0 || x + 1u == width || y + 1u == height;
                if (!source[index] || edge ||
                    !arc_visual_component_mask_neighbor_active(source, width, height, x - 1u, y) ||
                    !arc_visual_component_mask_neighbor_active(source, width, height, x + 1u, y) ||
                    !arc_visual_component_mask_neighbor_active(source, width, height, x, y - 1u) ||
                    !arc_visual_component_mask_neighbor_active(source, width, height, x, y + 1u)) {
                    styled[index] = 0;
                }
            }
        }
    }
}

static int arc_visual_combinator_frame_shape(const ARCGrid *input,
                                             const ARCSketchProgramSpec *spec,
                                             size_t *width_out,
                                             size_t *height_out)
{
    size_t width;
    size_t height;
    if (input == NULL || spec == NULL || width_out == NULL || height_out == NULL ||
        input->width == 0 || input->height == 0) {
        return 0;
    }
    if (spec->op == ARC_SKETCH_OP_VISUAL_COMBINATOR &&
        spec->width == 0 && spec->height == 0) {
        width = input->width;
        height = input->height;
    } else {
        if (spec->width == 0 || spec->height == 0) {
            return 0;
        }
        width = spec->width;
        height = spec->height;
    }
    if (width == 0 || height == 0 ||
        width > ARC_MAX_GRID_WIDTH || height > ARC_MAX_GRID_HEIGHT ||
        width * height > ARC_MAX_GRID_CELLS) {
        return 0;
    }
    *width_out = width;
    *height_out = height;
    return 1;
}

static int arc_visual_component_mask_shape_for_input(const ARCGrid *input,
                                                     const ARCSketchProgramSpec *spec,
                                                     size_t *render_width,
                                                     size_t *render_height)
{
    ARCVisualState state;
    size_t selected_index = SIZE_MAX;
    size_t min_x = 0;
    size_t min_y = 0;
    size_t max_x = 0;
    size_t max_y = 0;
    size_t source_width = 0;
    size_t source_height = 0;
    int transform_mode;
    memset(&state, 0, sizeof(state));
    if (input == NULL || spec == NULL || render_width == NULL ||
        render_height == NULL || input->width == 0 || input->height == 0 ||
        ((spec->width == 0) != (spec->height == 0)) ||
        spec->width > ARC_MAX_GRID_WIDTH ||
        spec->height > ARC_MAX_GRID_HEIGHT) {
        return 0;
    }
    if (!arc_visual_select_components_into_state(input, spec, &state)) {
        return 0;
    }
    selected_index = arc_visual_select_component_index(&state, spec->y_anchor);
    if (selected_index == SIZE_MAX) {
        return 0;
    }
    if (!arc_visual_component_mask_source_box(input,
                                              &state,
                                              spec,
                                              selected_index,
                                              &min_x,
                                              &min_y,
                                              &max_x,
                                              &max_y)) {
        return 0;
    }
    if (min_x > max_x || min_y > max_y ||
        max_x >= input->width || max_y >= input->height) {
        return 0;
    }
    source_width = max_x - min_x + 1u;
    source_height = max_y - min_y + 1u;
    if (source_width == 0 || source_height == 0 ||
        source_width * source_height > ARC_MAX_GRID_CELLS) {
        return 0;
    }
    transform_mode = spec->y_offset;
    if (transform_mode < 0 || transform_mode > 8) {
        transform_mode = 0;
    }
    return arc_visual_component_mask_render_shape(source_width,
                                                  source_height,
                                                  transform_mode,
                                                  render_width,
                                                  render_height);
}

static int arc_visual_component_mask_spec_feasible_for_task(
    const ARCTask *task,
    const ARCSketchProgramSpec *spec)
{
    if (task == NULL || spec == NULL ||
        spec->op != ARC_SKETCH_OP_VISUAL_COMBINATOR ||
        spec->x_anchor != 3) {
        return 1;
    }
    if (task->train_count == 0) {
        return 0;
    }
    for (size_t pair = 0; pair < task->train_count; ++pair) {
        size_t frame_width = 0;
        size_t frame_height = 0;
        size_t render_width = 0;
        size_t render_height = 0;
        const ARCGrid *input = &task->train[pair].input;
        const ARCGrid *expected = &task->train[pair].output;
        if (!task->train[pair].has_output ||
            !arc_visual_combinator_frame_shape(input, spec, &frame_width, &frame_height) ||
            expected->width != frame_width ||
            expected->height != frame_height) {
            return 0;
        }
        if (!arc_visual_component_mask_shape_for_input(input,
                                                       spec,
                                                       &render_width,
                                                       &render_height)) {
            return 0;
        }
        if (arc_visual_component_mask_requires_exact_shape(spec->y_anchor)) {
            if (render_width != frame_width || render_height != frame_height) {
                return 0;
            }
        }
        if (spec->y_anchor == 5 &&
            (render_width > frame_width || render_height > frame_height)) {
            return 0;
        }
    }
    return 1;
}

static int arc_visual_component_mask_marker_glyph_map_for_task(
    const ARCTask *task,
    const ARCSketchProgramSpec *spec,
    int *packed_map_out)
{
    int color_by_key[8];
    size_t mapped_count = 0;
    uint32_t packed = 0;
    if (task == NULL || spec == NULL || packed_map_out == NULL ||
        task->train_count == 0 || spec->op != ARC_SKETCH_OP_VISUAL_COMBINATOR ||
        spec->x_anchor != 3) {
        return 0;
    }
    for (size_t i = 0; i < sizeof(color_by_key) / sizeof(color_by_key[0]); ++i) {
        color_by_key[i] = -1;
    }
    for (size_t pair = 0; pair < task->train_count; ++pair) {
        ARCVisualState state;
        size_t marker_index = SIZE_MAX;
        unsigned key = 0;
        uint8_t expected_color;
        memset(&state, 0, sizeof(state));
        if (!task->train[pair].has_output ||
            (!arc_visual_marker_color_glyph_key(&task->train[pair].input,
                                                spec,
                                                &key) &&
             (!arc_visual_select_components_into_state(&task->train[pair].input,
                                                       spec,
                                                       &state) ||
              !arc_visual_marker_signal_index(&state, spec, &marker_index) ||
              !arc_visual_marker_glyph_key(&task->train[pair].input,
                                           &state,
                                           marker_index,
                                           &key)))) {
            return 0;
        }
        expected_color = arc_grid_most_common_non_background_color(
            &task->train[pair].output,
            spec->background);
        if (expected_color == spec->background || expected_color >= ARC_COLOR_COUNT ||
            key >= sizeof(color_by_key) / sizeof(color_by_key[0])) {
            return 0;
        }
        if (color_by_key[key] >= 0 && color_by_key[key] != (int)expected_color) {
            return 0;
        }
        if (color_by_key[key] < 0) {
            color_by_key[key] = (int)expected_color;
            ++mapped_count;
        }
    }
    if (mapped_count < 2) {
        return 0;
    }
    for (unsigned key = 0; key < 8u; ++key) {
        if (color_by_key[key] < 0) {
            continue;
        }
        packed |= ((uint32_t)color_by_key[key] & 0xFu) << (key * 4u);
    }
    if (packed == 0 || packed > 0x7fffffffu) {
        return 0;
    }
    *packed_map_out = (int)packed;
    return 1;
}

static int arc_visual_execute_step(const ARCVisualStep *step,
                                   const ARCSketchProgramSpec *spec,
                                   const ARCGrid *input,
                                   ARCGrid *output,
                                   ARCVisualState *state)
{
    if (step == NULL || spec == NULL || input == NULL || output == NULL ||
        state == NULL) {
        return 0;
    }
    switch (step->kind) {
    case ARC_VISUAL_STEP_COUNT_PALETTE:
        state->count = arc_grid_distinct_color_count(input,
                                                     spec->background,
                                                     spec->x_anchor == 0);
        return state->count > 0;
    case ARC_VISUAL_STEP_SCALE_GRID_BY_COUNT:
        if (state->count == 0 ||
            input->width * state->count > ARC_MAX_GRID_WIDTH ||
            input->height * state->count > ARC_MAX_GRID_HEIGHT) {
            return 0;
        }
        arc_init_grid(output, input->width * state->count, input->height * state->count);
        for (size_t y = 0; y < input->height; ++y) {
            for (size_t x = 0; x < input->width; ++x) {
                uint8_t value = input->cells[y * input->width + x];
                for (size_t yy = 0; yy < state->count; ++yy) {
                    for (size_t xx = 0; xx < state->count; ++xx) {
                        output->cells[(y * state->count + yy) * output->width +
                                      x * state->count + xx] = value;
                    }
                }
            }
        }
        return 1;
    case ARC_VISUAL_STEP_SELECT_OBJECT_COLOR:
        if (spec->color == spec->background) {
            return arc_select_any_color_grid_component(input,
                                                       spec->background,
                                                       (uint8_t)spec->x_anchor,
                                                       &state->color);
        } else {
            ARCScene scene;
            ARCObject object;
            if (!arc_extract_scene(input, spec->background, &scene)) {
                return 0;
            }
            uint8_t selector = (uint8_t)spec->x_anchor;
            if (selector >= ARC_OBJECT_SELECTOR_KIND_COUNT ||
                selector == ARC_OBJECT_SELECTOR_ALL_COLOR) {
                selector = ARC_OBJECT_SELECTOR_LARGEST_COLOR;
            }
            if (!arc_select_scene_object(&scene,
                                         selector,
                                         spec->color,
                                         spec->output_color,
                                         &object)) {
                return 0;
            }
            state->color = object.color;
            return 1;
        }
    case ARC_VISUAL_STEP_RENDER_SOLID_RECT: {
        if (spec->width == 0 || spec->height == 0 ||
            spec->width > ARC_MAX_GRID_WIDTH || spec->height > ARC_MAX_GRID_HEIGHT) {
            return 0;
        }
        uint8_t fill = spec->output_color == spec->background
                           ? state->color
                           : spec->output_color;
        arc_init_grid(output, spec->width, spec->height);
        for (size_t cell = 0; cell < arc_grid_cells(output); ++cell) {
            output->cells[cell] = fill;
        }
        return 1;
    }
    case ARC_VISUAL_STEP_COUNT_MARKER_CELLS:
        if (spec->color == spec->background) {
            return 0;
        }
        state->count = arc_count_color_cells(input, spec->color);
        return state->count > 0;
    case ARC_VISUAL_STEP_SELECT_MOTIF_WINDOW: {
        uint8_t motif_color = spec->background;
        size_t min_x = 0;
        size_t min_y = 0;
        size_t max_x = 0;
        size_t max_y = 0;
        if (!arc_largest_component_bbox_excluding_color(input,
                                                        spec->background,
                                                        spec->color,
                                                        spec->output_color,
                                                        &motif_color,
                                                        &min_x,
                                                        &min_y,
                                                        &max_x,
                                                        &max_y)) {
            return 0;
        }
        size_t bbox_width = max_x - min_x + 1u;
        size_t bbox_height = max_y - min_y + 1u;
        size_t motif_width = spec->width == 0 ? bbox_width : spec->width;
        size_t motif_height = spec->height == 0 ? bbox_height : spec->height;
        if (motif_width < bbox_width || motif_height < bbox_height) {
            return 0;
        }
        int64_t origin_x = (int64_t)min_x;
        int64_t origin_y = (int64_t)min_y;
        if (spec->x_anchor == 1) {
            origin_x -= (int64_t)((motif_width - bbox_width) / 2u);
        } else if (spec->x_anchor == 2) {
            origin_x = (int64_t)(max_x + 1u) - (int64_t)motif_width;
        }
        if (spec->y_anchor == 1) {
            origin_y -= (int64_t)((motif_height - bbox_height) / 2u);
        } else if (spec->y_anchor == 2) {
            origin_y = (int64_t)(max_y + 1u) - (int64_t)motif_height;
        }
        if (origin_x < 0 || origin_y < 0 ||
            origin_x + (int64_t)motif_width > (int64_t)input->width ||
            origin_y + (int64_t)motif_height > (int64_t)input->height) {
            return 0;
        }
        state->color = motif_color;
        state->window_x = (size_t)origin_x;
        state->window_y = (size_t)origin_y;
        state->window_width = motif_width;
        state->window_height = motif_height;
        return 1;
    }
    case ARC_VISUAL_STEP_REPEAT_WINDOW_BY_COUNT: {
        int vertical = spec->period == 1;
        size_t output_width = vertical ? state->window_width
                                       : state->window_width * state->count;
        size_t output_height = vertical ? state->window_height * state->count
                                        : state->window_height;
        if (state->count == 0 ||
            state->window_width == 0 || state->window_height == 0 ||
            output_width > ARC_MAX_GRID_WIDTH ||
            output_height > ARC_MAX_GRID_HEIGHT) {
            return 0;
        }
        arc_init_grid(output, output_width, output_height);
        for (size_t cell = 0; cell < arc_grid_cells(output); ++cell) {
            output->cells[cell] = spec->background;
        }
        for (size_t repeat = 0; repeat < state->count; ++repeat) {
            size_t base_x = vertical ? 0 : repeat * state->window_width;
            size_t base_y = vertical ? repeat * state->window_height : 0;
            for (size_t y = 0; y < state->window_height; ++y) {
                for (size_t x = 0; x < state->window_width; ++x) {
                    uint8_t value =
                        input->cells[(state->window_y + y) * input->width +
                                     state->window_x + x];
                    if (value == spec->color) {
                        value = spec->background;
                    }
                    if (value != spec->background && value != state->color) {
                        return 0;
                    }
                    output->cells[(base_y + y) * output->width + base_x + x] = value;
                }
            }
        }
        return 1;
    }
    case ARC_VISUAL_STEP_COUNT_COLOR: {
        uint8_t color = spec->color;
        if (color == spec->background) {
            uint8_t colors[ARC_COLOR_COUNT];
            size_t color_count = arc_grid_nonzero_colors(input, colors, ARC_COLOR_COUNT);
            if (color_count != 1) {
                return 0;
            }
            color = colors[0];
        }
        state->color = color;
        state->count = spec->y_anchor == 1
                           ? arc_count_color_cells(input, color)
                           : arc_count_color_components(input,
                                                        color,
                                                        NULL,
                                                        NULL,
                                                        NULL,
                                                        NULL);
        return state->count > 0;
    }
    case ARC_VISUAL_STEP_RENDER_COUNT_LINE: {
        if (state->count == 0 ||
            state->count > ARC_MAX_GRID_WIDTH ||
            state->count > ARC_MAX_GRID_HEIGHT) {
            return 0;
        }
        uint8_t fill = spec->output_color == spec->background
                           ? state->color
                           : spec->output_color;
        if (spec->x_anchor == 1) {
            arc_init_grid(output, 1, state->count);
        } else {
            arc_init_grid(output, state->count, 1);
        }
        for (size_t cell = 0; cell < arc_grid_cells(output); ++cell) {
            output->cells[cell] = fill;
        }
        return 1;
    }
    case ARC_VISUAL_STEP_SELECT_COMPONENT_ORIGINS: {
        return arc_visual_select_components_into_state(input, spec, state);
    }
    case ARC_VISUAL_STEP_RENDER_OCCUPANCY_GRID: {
        size_t frame_width = 0;
        size_t frame_height = 0;
        size_t min_x = 0;
        size_t min_y = 0;
        size_t max_x = 0;
        size_t max_y = 0;
        size_t starts_x[ARC_MAX_SCENE_OBJECTS];
        size_t starts_y[ARC_MAX_SCENE_OBJECTS];
        if (!arc_visual_combinator_frame_shape(input, spec, &frame_width, &frame_height) ||
            state->component_count == 0) {
            return 0;
        }
        if (!arc_visual_state_component_bounds(state, &min_x, &min_y, &max_x, &max_y)) {
            return 0;
        }
        for (size_t i = 0; i < state->component_count; ++i) {
            starts_x[i] = state->component_min_x[i];
            starts_y[i] = state->component_min_y[i];
        }
        arc_init_grid(output, frame_width, frame_height);
        for (size_t cell = 0; cell < arc_grid_cells(output); ++cell) {
            output->cells[cell] = spec->background;
        }
        for (size_t i = 0; i < state->component_count; ++i) {
            size_t out_x = 0;
            size_t out_y = 0;
            size_t object_width =
                state->component_max_x[i] - state->component_min_x[i] + 1u;
            size_t object_height =
                state->component_max_y[i] - state->component_min_y[i] + 1u;
            switch (spec->y_anchor) {
            case 0:
                out_x = (state->component_min_x[i] * frame_width) / input->width;
                out_y = (state->component_min_y[i] * frame_height) / input->height;
                break;
            case 1:
                out_x = ((state->component_min_x[i] + state->component_max_x[i] + 1u) *
                         frame_width) /
                        (2u * input->width);
                out_y = ((state->component_min_y[i] + state->component_max_y[i] + 1u) *
                         frame_height) /
                        (2u * input->height);
                break;
            case 2: {
                size_t span_x = max_x - min_x + 1u;
                size_t span_y = max_y - min_y + 1u;
                out_x = ((state->component_min_x[i] - min_x) * frame_width) / span_x;
                out_y = ((state->component_min_y[i] - min_y) * frame_height) / span_y;
                break;
            }
            case 3:
                out_x = arc_rank_value_scaled(starts_x,
                                              state->component_count,
                                              state->component_min_x[i],
                                              frame_width);
                out_y = arc_rank_value_scaled(starts_y,
                                              state->component_count,
                                              state->component_min_y[i],
                                              frame_height);
                break;
            case 4:
                if (object_width == 0 || object_height == 0) {
                    return 0;
                }
                out_x = state->component_min_x[i] / object_width;
                out_y = state->component_min_y[i] / object_height;
                break;
            case 5:
                out_x = arc_rank_value_scaled(starts_x,
                                              state->component_count,
                                              state->component_min_x[i],
                                              frame_width);
                out_y = arc_interval_pack_row(state, i, frame_height);
                break;
            default:
                return 0;
            }
            if (out_x >= frame_width) {
                out_x = frame_width - 1u;
            }
            if (out_y >= frame_height) {
                out_y = frame_height - 1u;
            }
            output->cells[out_y * output->width + out_x] =
                spec->period == 1u ? spec->color : spec->output_color;
        }
        return 1;
    }
    case ARC_VISUAL_STEP_RENDER_COUNT_TEMPLATE: {
        size_t frame_width = 0;
        size_t frame_height = 0;
        if (!arc_visual_combinator_frame_shape(input, spec, &frame_width, &frame_height) ||
            state->component_count == 0) {
            return 0;
        }
        arc_init_grid(output, frame_width, frame_height);
        for (size_t cell = 0; cell < arc_grid_cells(output); ++cell) {
            output->cells[cell] = spec->background;
        }
        uint8_t fill = spec->output_color == spec->background
                           ? spec->color
                           : spec->output_color;
        for (size_t i = 0; i < state->component_count; ++i) {
            size_t out_x = 0;
            size_t out_y = 0;
            if (!arc_visual_count_template_position(spec->y_anchor,
                                                    i,
                                                    frame_width,
                                                    frame_height,
                                                    &out_x,
                                                    &out_y)) {
                return 0;
            }
            output->cells[out_y * output->width + out_x] = fill;
        }
        return 1;
    }
    case ARC_VISUAL_STEP_RENDER_ALIGNED_SPANS: {
        size_t frame_width = 0;
        size_t frame_height = 0;
        int mode = spec->y_anchor;
        if (!arc_visual_combinator_frame_shape(input, spec, &frame_width, &frame_height) ||
            frame_width != input->width || frame_height != input->height ||
            spec->color == spec->background ||
            spec->output_color == spec->background ||
            mode < ARC_OBJECT_CONNECT_HORIZONTAL || mode > ARC_OBJECT_CONNECT_BOTH) {
            return 0;
        }
        *output = *input;
        if (mode == ARC_OBJECT_CONNECT_HORIZONTAL ||
            mode == ARC_OBJECT_CONNECT_BOTH) {
            for (size_t y = 0; y < input->height; ++y) {
                int found = 0;
                size_t min_x = input->width;
                size_t max_x = 0;
                for (size_t x = 0; x < input->width; ++x) {
                    if (input->cells[y * input->width + x] != spec->color) {
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
                for (size_t x = min_x + 1u; x < max_x; ++x) {
                    size_t cell = y * output->width + x;
                    if (input->cells[cell] == spec->background) {
                        output->cells[cell] = spec->output_color;
                    }
                }
            }
        }
        if (mode == ARC_OBJECT_CONNECT_VERTICAL ||
            mode == ARC_OBJECT_CONNECT_BOTH) {
            for (size_t x = 0; x < input->width; ++x) {
                int found = 0;
                size_t min_y = input->height;
                size_t max_y = 0;
                for (size_t y = 0; y < input->height; ++y) {
                    if (input->cells[y * input->width + x] != spec->color) {
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
                for (size_t y = min_y + 1u; y < max_y; ++y) {
                    size_t cell = y * output->width + x;
                    if (input->cells[cell] == spec->background) {
                        output->cells[cell] = spec->output_color;
                    }
                }
            }
        }
        return 1;
    }
    case ARC_VISUAL_STEP_RENDER_COMPONENT_SHAPE_GRID: {
        size_t frame_width = 0;
        size_t frame_height = 0;
        size_t min_x = 0;
        size_t min_y = 0;
        size_t max_x = 0;
        size_t max_y = 0;
        size_t starts_x[ARC_MAX_SCENE_OBJECTS];
        size_t starts_y[ARC_MAX_SCENE_OBJECTS];
        if (!arc_visual_combinator_frame_shape(input, spec, &frame_width, &frame_height) ||
            state->component_count == 0) {
            return 0;
        }
        if (!arc_visual_state_component_bounds(state, &min_x, &min_y, &max_x, &max_y)) {
            return 0;
        }
        for (size_t i = 0; i < state->component_count; ++i) {
            starts_x[i] = state->component_min_x[i];
            starts_y[i] = state->component_min_y[i];
        }
        arc_init_grid(output, frame_width, frame_height);
        for (size_t cell = 0; cell < arc_grid_cells(output); ++cell) {
            output->cells[cell] = spec->background;
        }
        for (size_t i = 0; i < state->component_count; ++i) {
            size_t out_x = 0;
            size_t out_y = 0;
            size_t object_width =
                state->component_max_x[i] - state->component_min_x[i] + 1u;
            size_t object_height =
                state->component_max_y[i] - state->component_min_y[i] + 1u;
            switch (spec->y_anchor) {
            case 0:
                out_x = (state->component_min_x[i] * frame_width) / input->width;
                out_y = (state->component_min_y[i] * frame_height) / input->height;
                break;
            case 1:
                out_x = ((state->component_min_x[i] + state->component_max_x[i] + 1u) *
                         frame_width) /
                        (2u * input->width);
                out_y = ((state->component_min_y[i] + state->component_max_y[i] + 1u) *
                         frame_height) /
                        (2u * input->height);
                break;
            case 2: {
                size_t span_x = max_x - min_x + 1u;
                size_t span_y = max_y - min_y + 1u;
                out_x = ((state->component_min_x[i] - min_x) * frame_width) / span_x;
                out_y = ((state->component_min_y[i] - min_y) * frame_height) / span_y;
                break;
            }
            case 3:
                out_x = arc_rank_value_scaled(starts_x,
                                              state->component_count,
                                              state->component_min_x[i],
                                              frame_width);
                out_y = arc_rank_value_scaled(starts_y,
                                              state->component_count,
                                              state->component_min_y[i],
                                              frame_height);
                break;
            case 4:
                if (object_width == 0 || object_height == 0) {
                    return 0;
                }
                out_x = state->component_min_x[i] / object_width;
                out_y = state->component_min_y[i] / object_height;
                break;
            case 5:
                out_x = arc_rank_value_scaled(starts_x,
                                              state->component_count,
                                              state->component_min_x[i],
                                              frame_width);
                out_y = arc_interval_pack_row(state, i, frame_height);
                break;
            default:
                return 0;
            }
            if (out_x >= frame_width) {
                out_x = frame_width - 1u;
            }
            if (out_y >= frame_height) {
                out_y = frame_height - 1u;
            }
            output->cells[out_y * output->width + out_x] =
                arc_visual_component_classifier_color(spec, input, state, i);
        }
        return 1;
    }
    case ARC_VISUAL_STEP_RENDER_COMPONENT_MASK: {
        size_t frame_width = 0;
        size_t frame_height = 0;
        size_t selected_index = SIZE_MAX;
        size_t min_x = 0;
        size_t min_y = 0;
        size_t max_x = 0;
        size_t max_y = 0;
        size_t source_width = 0;
        size_t source_height = 0;
        size_t render_width = 0;
        size_t render_height = 0;
        size_t dest_x = 0;
        size_t dest_y = 0;
        int transform_mode = spec->y_offset;
        int style_mode = spec->y_offset;
        int scaled = arc_visual_component_mask_scaled_mode(spec->y_anchor);
        uint8_t local_mask[ARC_MAX_GRID_CELLS];
        uint8_t styled_mask[ARC_MAX_GRID_CELLS];
        size_t local_component[ARC_MAX_GRID_CELLS];
        size_t styled_component[ARC_MAX_GRID_CELLS];
        if (!arc_visual_combinator_frame_shape(input, spec, &frame_width, &frame_height) ||
            state->component_count == 0) {
            return 0;
        }
        selected_index = arc_visual_select_component_index(state, spec->y_anchor);
        if (selected_index == SIZE_MAX) {
            return 0;
        }
        if (!arc_visual_component_mask_source_box(input,
                                                  state,
                                                  spec,
                                                  selected_index,
                                                  &min_x,
                                                  &min_y,
                                                  &max_x,
                                                  &max_y)) {
            return 0;
        }
        source_width = max_x - min_x + 1u;
        source_height = max_y - min_y + 1u;
        if (source_width == 0 || source_height == 0) {
            return 0;
        }
        if (source_width * source_height > ARC_MAX_GRID_CELLS) {
            return 0;
        }
        if (transform_mode < 0 || transform_mode > 8) {
            transform_mode = 0;
        }
        if (style_mode < 6 || style_mode > 8) {
            style_mode = 0;
        }
        if (!arc_visual_component_mask_render_shape(source_width,
                                                    source_height,
                                                    transform_mode,
                                                    &render_width,
                                                    &render_height)) {
            return 0;
        }
        if (arc_visual_component_mask_requires_exact_shape(spec->y_anchor)) {
            if (render_width != frame_width || render_height != frame_height) {
                return 0;
            }
        }
        if (spec->y_anchor == 5) {
            if (render_width > frame_width || render_height > frame_height) {
                return 0;
            }
            dest_x = (frame_width - render_width) / 2u;
            dest_y = (frame_height - render_height) / 2u;
        }
        for (size_t y = 0; y < source_height; ++y) {
            for (size_t x = 0; x < source_width; ++x) {
                size_t index = y * source_width + x;
                size_t component_index = arc_visual_component_at_cell(input,
                                                                      state,
                                                                      spec,
                                                                      min_x + x,
                                                                      min_y + y,
                                                                      selected_index);
                local_mask[index] = component_index != SIZE_MAX ? 1u : 0u;
                local_component[index] =
                    component_index != SIZE_MAX ? component_index : selected_index;
            }
        }
        arc_visual_component_mask_apply_style(local_mask,
                                              local_component,
                                              selected_index,
                                              source_width,
                                              source_height,
                                              style_mode,
                                              styled_mask,
                                              styled_component);
        arc_init_grid(output, frame_width, frame_height);
        for (size_t cell = 0; cell < arc_grid_cells(output); ++cell) {
            output->cells[cell] = spec->background;
        }
        for (size_t out_y = 0; out_y < frame_height; ++out_y) {
            for (size_t out_x = 0; out_x < frame_width; ++out_x) {
                size_t local_x = out_x;
                size_t local_y = out_y;
                size_t source_x = 0;
                size_t source_y = 0;
                size_t display_x = 0;
                size_t display_y = 0;
                size_t source_index = 0;
                if (spec->y_anchor == 5) {
                    if (out_x < dest_x || out_y < dest_y ||
                        out_x >= dest_x + render_width ||
                        out_y >= dest_y + render_height) {
                        continue;
                    }
                    local_x = out_x - dest_x;
                    local_y = out_y - dest_y;
                }
                if (scaled) {
                    display_x = (local_x * render_width) / frame_width;
                    display_y = (local_y * render_height) / frame_height;
                } else {
                    if (local_x >= render_width || local_y >= render_height) {
                        continue;
                    }
                    display_x = local_x;
                    display_y = local_y;
                }
                if (!arc_visual_component_mask_unmap(display_x,
                                                     display_y,
                                                     source_width,
                                                     source_height,
                                                     transform_mode,
                                                     &source_x,
                                                     &source_y)) {
                    continue;
                }
                source_index = source_y * source_width + source_x;
                if (styled_mask[source_index]) {
                    uint8_t paint =
                        arc_visual_component_mask_color(spec,
                                                        input,
                                                        state,
                                                        styled_component[source_index]);
                    if (style_mode == 7 && spec->output_color != spec->background) {
                        paint = local_mask[source_index]
                                    ? state->component_color[styled_component[source_index]]
                                    : spec->output_color;
                    }
                    output->cells[out_y * output->width + out_x] = paint;
                }
            }
        }
        return 1;
    }
    default:
        return 0;
    }
}

static int arc_visual_program_execute(const ARCVisualStep *steps,
                                      size_t step_count,
                                      const ARCSketchProgramSpec *spec,
                                      const ARCGrid *input,
                                      ARCGrid *output)
{
    ARCVisualState state;
    memset(&state, 0, sizeof(state));
    if (steps == NULL || step_count == 0) {
        return 0;
    }
    for (size_t i = 0; i < step_count; ++i) {
        if (!arc_visual_execute_step(&steps[i], spec, input, output, &state)) {
            return 0;
        }
    }
    return 1;
}

static int arc_apply_sketch_palette_scale(const ARCSketchProgramSpec *spec,
                                          const ARCGrid *input,
                                          ARCGrid *output)
{
    if (spec == NULL || input == NULL || output == NULL ||
        input->width == 0 || input->height == 0) {
        return 0;
    }
    const ARCVisualStep steps[] = {
        {ARC_VISUAL_STEP_COUNT_PALETTE},
        {ARC_VISUAL_STEP_SCALE_GRID_BY_COUNT},
    };
    return arc_visual_program_execute(steps,
                                      sizeof(steps) / sizeof(steps[0]),
                                      spec,
                                      input,
                                      output);
}

static int arc_apply_sketch_object_color_tile(const ARCSketchProgramSpec *spec,
                                              const ARCGrid *input,
                                              ARCGrid *output)
{
    if (spec == NULL || input == NULL || output == NULL ||
        spec->width == 0 || spec->height == 0 ||
        spec->width > ARC_MAX_GRID_WIDTH || spec->height > ARC_MAX_GRID_HEIGHT) {
        return 0;
    }
    const ARCVisualStep steps[] = {
        {ARC_VISUAL_STEP_SELECT_OBJECT_COLOR},
        {ARC_VISUAL_STEP_RENDER_SOLID_RECT},
    };
    return arc_visual_program_execute(steps,
                                      sizeof(steps) / sizeof(steps[0]),
                                      spec,
                                      input,
                                      output);
}

static int arc_apply_sketch_marker_count_motif_tile(const ARCSketchProgramSpec *spec,
                                                    const ARCGrid *input,
                                                    ARCGrid *output)
{
    if (spec == NULL || input == NULL || output == NULL ||
        spec->color == spec->background) {
        return 0;
    }
    const ARCVisualStep steps[] = {
        {ARC_VISUAL_STEP_COUNT_MARKER_CELLS},
        {ARC_VISUAL_STEP_SELECT_MOTIF_WINDOW},
        {ARC_VISUAL_STEP_REPEAT_WINDOW_BY_COUNT},
    };
    return arc_visual_program_execute(steps,
                                      sizeof(steps) / sizeof(steps[0]),
                                      spec,
                                      input,
                                      output);
}

static int arc_apply_sketch_component_count_line(const ARCSketchProgramSpec *spec,
                                                 const ARCGrid *input,
                                                 ARCGrid *output)
{
    if (spec == NULL || input == NULL || output == NULL) {
        return 0;
    }
    const ARCVisualStep steps[] = {
        {ARC_VISUAL_STEP_COUNT_COLOR},
        {ARC_VISUAL_STEP_RENDER_COUNT_LINE},
    };
    return arc_visual_program_execute(steps,
                                      sizeof(steps) / sizeof(steps[0]),
                                      spec,
                                      input,
                                      output);
}

static int arc_apply_sketch_visual_combinator(const ARCSketchProgramSpec *spec,
                                              const ARCGrid *input,
                                              ARCGrid *output)
{
    size_t frame_width = 0;
    size_t frame_height = 0;
    if (spec == NULL || input == NULL || output == NULL ||
        !arc_visual_combinator_frame_shape(input, spec, &frame_width, &frame_height)) {
        return 0;
    }
    (void)frame_width;
    (void)frame_height;
    if (spec->x_anchor == 0) {
        const ARCVisualStep steps[] = {
            {ARC_VISUAL_STEP_SELECT_COMPONENT_ORIGINS},
            {ARC_VISUAL_STEP_RENDER_OCCUPANCY_GRID},
        };
        return arc_visual_program_execute(steps,
                                          sizeof(steps) / sizeof(steps[0]),
                                          spec,
                                          input,
                                          output);
    }
    if (spec->x_anchor == 1) {
        const ARCVisualStep steps[] = {
            {ARC_VISUAL_STEP_SELECT_COMPONENT_ORIGINS},
            {ARC_VISUAL_STEP_RENDER_COUNT_TEMPLATE},
        };
        return arc_visual_program_execute(steps,
                                          sizeof(steps) / sizeof(steps[0]),
                                          spec,
                                          input,
                                          output);
    }
    if (spec->x_anchor == 2) {
        const ARCVisualStep steps[] = {
            {ARC_VISUAL_STEP_SELECT_COMPONENT_ORIGINS},
            {ARC_VISUAL_STEP_RENDER_COMPONENT_SHAPE_GRID},
        };
        return arc_visual_program_execute(steps,
                                          sizeof(steps) / sizeof(steps[0]),
                                          spec,
                                          input,
                                          output);
    }
    if (spec->x_anchor == 3) {
        const ARCVisualStep steps[] = {
            {ARC_VISUAL_STEP_SELECT_COMPONENT_ORIGINS},
            {ARC_VISUAL_STEP_RENDER_COMPONENT_MASK},
        };
        return arc_visual_program_execute(steps,
                                          sizeof(steps) / sizeof(steps[0]),
                                          spec,
                                          input,
                                          output);
    }
    if (spec->x_anchor == 4) {
        const ARCVisualStep steps[] = {
            {ARC_VISUAL_STEP_RENDER_ALIGNED_SPANS},
        };
        return arc_visual_program_execute(steps,
                                          sizeof(steps) / sizeof(steps[0]),
                                          spec,
                                          input,
                                          output);
    }
    return 0;
}

static int arc_apply_sketch_component_color_summary(const ARCSketchProgramSpec *spec,
                                                    const ARCGrid *input,
                                                    ARCGrid *output)
{
    if (spec == NULL || input == NULL || output == NULL ||
        spec->width == 0 || spec->height == 0 ||
        spec->width > ARC_MAX_GRID_WIDTH || spec->height > ARC_MAX_GRID_HEIGHT) {
        return 0;
    }
    size_t min_x = 0;
    size_t min_y = 0;
    size_t max_x = 0;
    size_t max_y = 0;
    if (!arc_grid_non_background_bbox(input,
                                      spec->background,
                                      &min_x,
                                      &min_y,
                                      &max_x,
                                      &max_y)) {
        return 0;
    }
    ARCScene scene;
    if (!arc_extract_scene(input, spec->background, &scene)) {
        return 0;
    }
    if (spec->period == 1u || spec->period == 2u) {
        int full_frame = spec->x_anchor == 1;
        if (full_frame) {
            min_x = 0;
            min_y = 0;
            max_x = input->width - 1u;
            max_y = input->height - 1u;
        }
        arc_init_grid(output, spec->width, spec->height);
        for (size_t cell = 0; cell < arc_grid_cells(output); ++cell) {
            output->cells[cell] = spec->background;
        }
        double span_x = (double)(max_x - min_x + 1u);
        double span_y = (double)(max_y - min_y + 1u);
        for (size_t object_index = 0; object_index < scene.object_count; ++object_index) {
            const ARCObject *object = &scene.objects[object_index];
            if (object->color == spec->background) {
                continue;
            }
            if (spec->color != spec->background && object->color != spec->color) {
                continue;
            }
            double rel_x = (object->centroid_x - (double)min_x) / span_x;
            double rel_y = (object->centroid_y - (double)min_y) / span_y;
            if (rel_x < 0.0) {
                rel_x = 0.0;
            }
            if (rel_y < 0.0) {
                rel_y = 0.0;
            }
            size_t out_x = (size_t)(rel_x * (double)spec->width);
            size_t out_y = (size_t)(rel_y * (double)spec->height);
            if (out_x >= spec->width) {
                out_x = spec->width - 1u;
            }
            if (out_y >= spec->height) {
                out_y = spec->height - 1u;
            }
            output->cells[out_y * output->width + out_x] =
                spec->period == 1u ? spec->output_color : object->color;
        }
        return 1;
    }
    uint8_t ignore_color = spec->color == spec->background
                               ? arc_grid_most_common_non_background_color(input,
                                                                            spec->background)
                               : spec->color;
    arc_init_grid(output, spec->width, spec->height);
    for (size_t cell = 0; cell < arc_grid_cells(output); ++cell) {
        output->cells[cell] = spec->output_color;
    }
    double span_x = (double)(max_x - min_x + 1u);
    double span_y = (double)(max_y - min_y + 1u);
    for (size_t object_index = 0; object_index < scene.object_count; ++object_index) {
        const ARCObject *object = &scene.objects[object_index];
        if (object->color == ignore_color || object->color == spec->background) {
            continue;
        }
        double rel_x = (object->centroid_x - (double)min_x) / span_x;
        double rel_y = (object->centroid_y - (double)min_y) / span_y;
        if (rel_x < 0.0) {
            rel_x = 0.0;
        }
        if (rel_y < 0.0) {
            rel_y = 0.0;
        }
        size_t out_x = (size_t)(rel_x * (double)spec->width);
        size_t out_y = (size_t)(rel_y * (double)spec->height);
        if (out_x >= spec->width) {
            out_x = spec->width - 1u;
        }
        if (out_y >= spec->height) {
            out_y = spec->height - 1u;
        }
        output->cells[out_y * output->width + out_x] = object->color;
    }
    return 1;
}

static int arc_apply_sketch_object_mask_extract(const ARCSketchProgramSpec *spec,
                                                const ARCGrid *input,
                                                ARCGrid *output)
{
    size_t min_x = 0;
    size_t min_y = 0;
    size_t max_x = 0;
    size_t max_y = 0;
    uint8_t selected_mask[ARC_MAX_GRID_CELLS];
    int use_component_mask = 0;
    memset(selected_mask, 0, sizeof(selected_mask));
    if (spec == NULL || input == NULL || output == NULL ||
        input->width == 0 || input->height == 0) {
        return 0;
    }
    if (spec->color == spec->background &&
        spec->x_anchor == ARC_OBJECT_SELECTOR_ALL_COLOR) {
        ARCComponent component;
        if (!arc_find_largest_component(input, spec->background, &component)) {
            return 0;
        }
        min_x = component.min_x;
        min_y = component.min_y;
        max_x = component.max_x;
        max_y = component.max_y;
        memcpy(selected_mask, component.mask, sizeof(selected_mask));
        use_component_mask = 1;
    } else {
        ARCScene scene;
        ARCObject object;
        uint8_t selector = (uint8_t)spec->x_anchor;
        if (selector >= ARC_OBJECT_SELECTOR_KIND_COUNT) {
            selector = ARC_OBJECT_SELECTOR_LARGEST_COLOR;
        }
        if (!arc_extract_scene(input, spec->background, &scene) ||
            !arc_select_scene_object(&scene,
                                     selector,
                                     spec->color,
                                     spec->output_color,
                                     &object)) {
            return 0;
        }
        min_x = object.min_x;
        min_y = object.min_y;
        max_x = object.max_x;
        max_y = object.max_y;
    }
    if (min_x > max_x || min_y > max_y ||
        max_x >= input->width || max_y >= input->height) {
        return 0;
    }
    size_t bbox_width = max_x - min_x + 1u;
    size_t bbox_height = max_y - min_y + 1u;
    size_t width = spec->width == 0 ? bbox_width : spec->width;
    size_t height = spec->height == 0 ? bbox_height : spec->height;
    int64_t origin_x = (int64_t)min_x;
    int64_t origin_y = (int64_t)min_y;
    if (spec->width != 0 || spec->height != 0) {
        origin_x += spec->x_offset;
        origin_y += spec->y_offset;
    }
    if (width == 0 || height == 0 ||
        width > ARC_MAX_GRID_WIDTH || height > ARC_MAX_GRID_HEIGHT) {
        return 0;
    }
    arc_init_grid(output, width, height);
    for (size_t cell = 0; cell < arc_grid_cells(output); ++cell) {
        output->cells[cell] = spec->background;
    }
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            int64_t sx = origin_x + (int64_t)x;
            int64_t sy = origin_y + (int64_t)y;
            if (sx < 0 || sy < 0 ||
                sx >= (int64_t)input->width || sy >= (int64_t)input->height) {
                continue;
            }
            size_t source = (size_t)sy * input->width + (size_t)sx;
            size_t target = y * output->width + x;
            uint8_t value = input->cells[source];
            int selected = use_component_mask ? selected_mask[source] : value == spec->color;
            if (spec->y_anchor == 2) {
                output->cells[target] = value;
            } else if (spec->y_anchor == 3) {
                output->cells[target] =
                    value == spec->background ? spec->output_color : value;
            } else if (selected) {
                output->cells[target] =
                    spec->y_anchor == 1 ? spec->output_color : value;
            }
        }
    }
    return 1;
}

static int arc_apply_sketch_object_translate(const ARCSketchProgramSpec *spec,
                                             const ARCGrid *input,
                                             ARCGrid *output)
{
    ARCScene scene;
    ARCObject object;
    uint8_t selector;
    uint8_t reference_color;
    size_t selected_cells = 0;
    size_t min_x = 0;
    size_t min_y = 0;
    size_t max_x = 0;
    size_t max_y = 0;
    int translate_all_color = 0;
    if (spec == NULL || input == NULL || output == NULL ||
        input->width == 0 || input->height == 0 ||
        spec->period >= ARC_OBJECT_TRANSLATE_RENDER_KIND_COUNT) {
        return 0;
    }
    if (spec->x_anchor < 0 ||
        spec->x_anchor >= ARC_OBJECT_SELECTOR_KIND_COUNT) {
        return 0;
    }
    selector = (uint8_t)spec->x_anchor;
    reference_color = selector == ARC_OBJECT_SELECTOR_NEAREST_TO_COLOR
                          ? (uint8_t)spec->y_anchor
                          : 0;
    if (reference_color >= ARC_COLOR_COUNT) {
        return 0;
    }
    translate_all_color = selector == ARC_OBJECT_SELECTOR_ALL_COLOR;
    if (translate_all_color) {
        if (spec->color == spec->background) {
            return 0;
        }
        max_x = input->width - 1u;
        max_y = input->height - 1u;
    } else {
        if (!arc_extract_scene(input, spec->background, &scene) ||
            !arc_select_scene_object(&scene,
                                     selector,
                                     spec->color,
                                     reference_color,
                                     &object)) {
            return 0;
        }
        min_x = object.min_x;
        min_y = object.min_y;
        max_x = object.max_x;
        max_y = object.max_y;
    }

    for (size_t y = min_y; y <= max_y; ++y) {
        for (size_t x = min_x; x <= max_x; ++x) {
            size_t cell = y * input->width + x;
            if (input->cells[cell] != spec->color) {
                continue;
            }
            int out_x = (int)x + spec->x_offset;
            int out_y = (int)y + spec->y_offset;
            if (out_x < 0 || out_y < 0 ||
                out_x >= (int)input->width || out_y >= (int)input->height) {
                return 0;
            }
            ++selected_cells;
        }
    }
    if (selected_cells == 0) {
        return 0;
    }

    if (spec->period == ARC_OBJECT_TRANSLATE_RENDER_BLANK) {
        arc_init_grid(output, input->width, input->height);
        for (size_t cell = 0; cell < arc_grid_cells(output); ++cell) {
            output->cells[cell] = spec->background;
        }
    } else {
        *output = *input;
    }

    if (spec->period == ARC_OBJECT_TRANSLATE_RENDER_MOVE) {
        for (size_t y = min_y; y <= max_y; ++y) {
            for (size_t x = min_x; x <= max_x; ++x) {
                size_t cell = y * input->width + x;
                if (input->cells[cell] == spec->color) {
                    output->cells[cell] = spec->background;
                }
            }
        }
    }
    for (size_t y = min_y; y <= max_y; ++y) {
        for (size_t x = min_x; x <= max_x; ++x) {
            size_t cell = y * input->width + x;
            if (input->cells[cell] != spec->color) {
                continue;
            }
            size_t out_x = (size_t)((int)x + spec->x_offset);
            size_t out_y = (size_t)((int)y + spec->y_offset);
            output->cells[out_y * output->width + out_x] = spec->output_color;
        }
    }
    return 1;
}

static int arc_apply_sketch_recolor_crop(const ARCSketchProgramSpec *spec,
                                         const ARCGrid *input,
                                         ARCGrid *output)
{
    size_t origin_x = 0;
    size_t origin_y = 0;
    size_t width = spec == NULL ? 0 : spec->width;
    size_t height = spec == NULL ? 0 : spec->height;
    size_t max_x = 0;
    size_t max_y = 0;
    if (spec == NULL || input == NULL || output == NULL ||
        input->width == 0 || input->height == 0) {
        return 0;
    }
    if (spec->selector == ARC_SKETCH_SELECTOR_GRID_PANEL) {
        if (width == 0 || height == 0 ||
            !arc_panel_origin(input->width,
                              width,
                              spec->x_anchor,
                              spec->x_offset,
                              &origin_x) ||
            !arc_panel_origin(input->height,
                              height,
                              spec->y_anchor,
                              spec->y_offset,
                              &origin_y)) {
            return 0;
        }
    } else if (spec->selector == ARC_SKETCH_SELECTOR_FOREGROUND_BBOX_PANEL) {
        if (spec->period == 7u || spec->period == 8u) {
            int found = 0;
            origin_x = input->width;
            origin_y = input->height;
            max_x = 0;
            max_y = 0;
            for (size_t y = 0; y < input->height; ++y) {
                for (size_t x = 0; x < input->width; ++x) {
                    uint8_t value = input->cells[y * input->width + x];
                    if (value == spec->background || value == spec->color) {
                        continue;
                    }
                    if (!found || x < origin_x) {
                        origin_x = x;
                    }
                    if (!found || y < origin_y) {
                        origin_y = y;
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
        } else {
            if (!arc_grid_non_background_bbox(input,
                                              spec->background,
                                              &origin_x,
                                              &origin_y,
                                              &max_x,
                                              &max_y)) {
                return 0;
            }
        }
        if (width == 0) {
            width = max_x - origin_x + 1u;
        } else {
            int64_t adjusted = (int64_t)origin_x + (int64_t)spec->x_offset;
            if (adjusted < 0) {
                return 0;
            }
            origin_x = (size_t)adjusted;
        }
        if (height == 0) {
            height = max_y - origin_y + 1u;
        } else {
            int64_t adjusted = (int64_t)origin_y + (int64_t)spec->y_offset;
            if (adjusted < 0) {
                return 0;
            }
            origin_y = (size_t)adjusted;
        }
    } else if (spec->selector == ARC_SKETCH_SELECTOR_COLOR_BBOX) {
        if (spec->x_anchor >= ARC_OBJECT_SELECTOR_UNIQUE_COLOR &&
            spec->x_anchor <= ARC_OBJECT_SELECTOR_TOUCHING_BORDER_COLOR) {
            ARCScene scene;
            ARCObject object;
            if (!arc_extract_scene(input, spec->background, &scene) ||
                !arc_select_scene_object(&scene,
                                         (uint8_t)spec->x_anchor,
                                         spec->color,
                                         0,
                                         &object)) {
                return 0;
            }
            origin_x = object.min_x;
            origin_y = object.min_y;
            max_x = object.max_x;
            max_y = object.max_y;
        } else if (!arc_grid_color_bbox(input,
                                        spec->color,
                                        &origin_x,
                                        &origin_y,
                                        &max_x,
                                        &max_y)) {
            return 0;
        }
        if (width == 0) {
            width = max_x - origin_x + 1u;
        } else {
            int64_t adjusted = (int64_t)origin_x + (int64_t)spec->x_offset;
            if (adjusted < 0) {
                return 0;
            }
            origin_x = (size_t)adjusted;
        }
        if (height == 0) {
            height = max_y - origin_y + 1u;
        } else {
            int64_t adjusted = (int64_t)origin_y + (int64_t)spec->y_offset;
            if (adjusted < 0) {
                return 0;
            }
            origin_y = (size_t)adjusted;
        }
    } else {
        return 0;
    }
    if (!arc_sketch_copy_region(input, origin_x, origin_y, width, height, output)) {
        return 0;
    }
    for (size_t cell = 0; cell < arc_grid_cells(output); ++cell) {
        uint8_t value = output->cells[cell];
        switch (spec->period) {
        case 1:
            if (value == spec->color) {
                output->cells[cell] = spec->output_color;
            }
            break;
        case 2:
            if (value == spec->background) {
                output->cells[cell] = spec->output_color;
            }
            break;
        case 3:
            output->cells[cell] =
                value == spec->color ? spec->output_color : spec->background;
            break;
        case 4:
            if (value == spec->color) {
                output->cells[cell] = spec->background;
            }
            break;
        case 5:
            if (value != spec->background) {
                output->cells[cell] = spec->output_color;
            }
            break;
        case 6:
            if (value == spec->background) {
                output->cells[cell] = spec->output_color;
            } else if (value == spec->color) {
                output->cells[cell] = spec->background;
            }
            break;
        case 7:
            if (value == spec->color) {
                output->cells[cell] = spec->background;
            }
            break;
        case 8:
            break;
        default:
            break;
        }
    }
    return 1;
}

typedef struct {
    size_t x;
    size_t y;
    uint8_t color;
} ARCMarkerPoint;

static size_t arc_factorial_size(size_t value)
{
    size_t result = 1;
    for (size_t i = 2; i <= value; ++i) {
        result *= i;
    }
    return result;
}

static void arc_permute_marker_points(const ARCMarkerPoint *sorted,
                                      size_t count,
                                      size_t index,
                                      ARCMarkerPoint *ordered)
{
    ARCMarkerPoint pool[ARC_COLOR_COUNT];
    size_t pool_count = count;
    for (size_t i = 0; i < count; ++i) {
        pool[i] = sorted[i];
    }
    for (size_t out = 0; out < count; ++out) {
        size_t block = arc_factorial_size(pool_count - 1u);
        size_t pick = block == 0 ? 0 : (index / block) % pool_count;
        ordered[out] = pool[pick];
        for (size_t j = pick + 1u; j < pool_count; ++j) {
            pool[j - 1u] = pool[j];
        }
        index = block == 0 ? 0 : index % block;
        --pool_count;
    }
}

static int arc_marker_point_compare_mode(const ARCMarkerPoint *lhs,
                                         const ARCMarkerPoint *rhs,
                                         int mode)
{
    int lhs_primary = 0;
    int rhs_primary = 0;
    int lhs_secondary = 0;
    int rhs_secondary = 0;
    switch (mode) {
    case 24:
        lhs_primary = (int)lhs->x;
        rhs_primary = (int)rhs->x;
        lhs_secondary = (int)lhs->y;
        rhs_secondary = (int)rhs->y;
        break;
    case 25:
        lhs_primary = -(int)lhs->x;
        rhs_primary = -(int)rhs->x;
        lhs_secondary = (int)lhs->y;
        rhs_secondary = (int)rhs->y;
        break;
    case 26:
        lhs_primary = (int)lhs->y;
        rhs_primary = (int)rhs->y;
        lhs_secondary = (int)lhs->x;
        rhs_secondary = (int)rhs->x;
        break;
    case 27:
        lhs_primary = -(int)lhs->y;
        rhs_primary = -(int)rhs->y;
        lhs_secondary = (int)lhs->x;
        rhs_secondary = (int)rhs->x;
        break;
    case 28:
        lhs_primary = (int)(lhs->x + lhs->y);
        rhs_primary = (int)(rhs->x + rhs->y);
        lhs_secondary = (int)lhs->x;
        rhs_secondary = (int)rhs->x;
        break;
    case 29:
        lhs_primary = -(int)(lhs->x + lhs->y);
        rhs_primary = -(int)(rhs->x + rhs->y);
        lhs_secondary = (int)lhs->x;
        rhs_secondary = (int)rhs->x;
        break;
    case 30:
        lhs_primary = (int)lhs->x - (int)lhs->y;
        rhs_primary = (int)rhs->x - (int)rhs->y;
        lhs_secondary = (int)lhs->x;
        rhs_secondary = (int)rhs->x;
        break;
    case 31:
        lhs_primary = (int)lhs->y - (int)lhs->x;
        rhs_primary = (int)rhs->y - (int)rhs->x;
        lhs_secondary = (int)lhs->y;
        rhs_secondary = (int)rhs->y;
        break;
    default:
        lhs_primary = (int)lhs->color;
        rhs_primary = (int)rhs->color;
        lhs_secondary = (int)(lhs->x + lhs->y);
        rhs_secondary = (int)(rhs->x + rhs->y);
        break;
    }
    if (lhs_primary != rhs_primary) {
        return lhs_primary < rhs_primary ? -1 : 1;
    }
    if (lhs_secondary != rhs_secondary) {
        return lhs_secondary < rhs_secondary ? -1 : 1;
    }
    return lhs->color < rhs->color ? -1 : (lhs->color > rhs->color ? 1 : 0);
}

static void arc_order_marker_points_by_mode(const ARCMarkerPoint *points,
                                            size_t count,
                                            int mode,
                                            ARCMarkerPoint *ordered)
{
    for (size_t i = 0; i < count; ++i) {
        ordered[i] = points[i];
    }
    if (mode == 32 || mode == 33) {
        size_t used[ARC_COLOR_COUNT];
        memset(used, 0, sizeof(used));
        size_t current_x = mode == 32 ? 0u : ARC_MAX_GRID_WIDTH - 1u;
        size_t current_y = mode == 32 ? 0u : ARC_MAX_GRID_HEIGHT - 1u;
        for (size_t out = 0; out < count; ++out) {
            size_t best = count;
            size_t best_distance = 0;
            for (size_t i = 0; i < count; ++i) {
                if (used[i]) {
                    continue;
                }
                size_t dx = points[i].x > current_x
                                ? points[i].x - current_x
                                : current_x - points[i].x;
                size_t dy = points[i].y > current_y
                                ? points[i].y - current_y
                                : current_y - points[i].y;
                size_t distance = dx * dx + dy * dy;
                if (best == count || distance < best_distance ||
                    (distance == best_distance &&
                     points[i].color < points[best].color)) {
                    best = i;
                    best_distance = distance;
                }
            }
            if (best == count) {
                return;
            }
            ordered[out] = points[best];
            used[best] = 1;
            current_x = points[best].x;
            current_y = points[best].y;
        }
        return;
    }
    for (size_t i = 0; i + 1u < count; ++i) {
        for (size_t j = i + 1u; j < count; ++j) {
            if (arc_marker_point_compare_mode(&ordered[j], &ordered[i], mode) < 0) {
                ARCMarkerPoint tmp = ordered[i];
                ordered[i] = ordered[j];
                ordered[j] = tmp;
            }
        }
    }
}

static void arc_marker_polyline_draw_hv_segment(ARCGrid *output,
                                                uint8_t background,
                                                uint8_t output_color,
                                                size_t x1,
                                                size_t y1,
                                                size_t x2,
                                                size_t y2,
                                                int vertical_first)
{
    if (vertical_first) {
        for (size_t y = y1 < y2 ? y1 : y2; y <= (y1 < y2 ? y2 : y1); ++y) {
            size_t cell = y * output->width + x1;
            if (output->cells[cell] == background) {
                output->cells[cell] = output_color;
            }
        }
        for (size_t x = x1 < x2 ? x1 : x2; x <= (x1 < x2 ? x2 : x1); ++x) {
            size_t cell = y2 * output->width + x;
            if (output->cells[cell] == background) {
                output->cells[cell] = output_color;
            }
        }
    } else {
        for (size_t x = x1 < x2 ? x1 : x2; x <= (x1 < x2 ? x2 : x1); ++x) {
            size_t cell = y1 * output->width + x;
            if (output->cells[cell] == background) {
                output->cells[cell] = output_color;
            }
        }
        for (size_t y = y1 < y2 ? y1 : y2; y <= (y1 < y2 ? y2 : y1); ++y) {
            size_t cell = y * output->width + x2;
            if (output->cells[cell] == background) {
                output->cells[cell] = output_color;
            }
        }
    }
}

static int arc_apply_sketch_marker_polyline(const ARCSketchProgramSpec *spec,
                                            const ARCGrid *input,
                                            ARCGrid *output)
{
    ARCMarkerPoint points[ARC_COLOR_COUNT];
    ARCMarkerPoint ordered[ARC_COLOR_COUNT];
    size_t counts[ARC_COLOR_COUNT];
    size_t point_count = 0;
    if (spec == NULL || input == NULL || output == NULL ||
        input->width == 0 || input->height == 0) {
        return 0;
    }
    memset(counts, 0, sizeof(counts));
    memset(points, 0, sizeof(points));
    for (size_t y = 0; y < input->height; ++y) {
        for (size_t x = 0; x < input->width; ++x) {
            uint8_t value = input->cells[y * input->width + x];
            if (value == spec->background || value >= ARC_COLOR_COUNT) {
                continue;
            }
            if (counts[value] == 0) {
                points[point_count].x = x;
                points[point_count].y = y;
                points[point_count].color = value;
                ++point_count;
            }
            ++counts[value];
        }
    }
    if (point_count < 2u || point_count > 6u) {
        return 0;
    }
    for (size_t i = 0; i < point_count; ++i) {
        if (counts[points[i].color] != 1u) {
            return 0;
        }
    }
    for (size_t i = 0; i + 1u < point_count; ++i) {
        for (size_t j = i + 1u; j < point_count; ++j) {
            if (points[j].color < points[i].color) {
                ARCMarkerPoint tmp = points[i];
                points[i] = points[j];
                points[j] = tmp;
            }
        }
    }
    if (spec->x_anchor >= 0 && spec->x_anchor < 24 && point_count <= 4u) {
        arc_permute_marker_points(points,
                                  point_count,
                                  (size_t)spec->x_anchor,
                                  ordered);
    } else {
        arc_order_marker_points_by_mode(points, point_count, spec->x_anchor, ordered);
    }
    *output = *input;
    for (size_t i = 0; i + 1u < point_count; ++i) {
        size_t x1 = ordered[i].x;
        size_t y1 = ordered[i].y;
        size_t x2 = ordered[i + 1u].x;
        size_t y2 = ordered[i + 1u].y;
        if (spec->y_anchor == 2) {
            arc_marker_polyline_draw_hv_segment(output,
                                                spec->background,
                                                spec->output_color,
                                                x1,
                                                y1,
                                                x2,
                                                y2,
                                                0);
            arc_marker_polyline_draw_hv_segment(output,
                                                spec->background,
                                                spec->output_color,
                                                x1,
                                                y1,
                                                x2,
                                                y2,
                                                1);
        } else if (spec->y_anchor == 3) {
            size_t min_x = x1 < x2 ? x1 : x2;
            size_t max_x = x1 < x2 ? x2 : x1;
            size_t min_y = y1 < y2 ? y1 : y2;
            size_t max_y = y1 < y2 ? y2 : y1;
            for (size_t x = min_x; x <= max_x; ++x) {
                size_t top = min_y * output->width + x;
                size_t bottom = max_y * output->width + x;
                if (output->cells[top] == spec->background) {
                    output->cells[top] = spec->output_color;
                }
                if (output->cells[bottom] == spec->background) {
                    output->cells[bottom] = spec->output_color;
                }
            }
            for (size_t y = min_y; y <= max_y; ++y) {
                size_t left = y * output->width + min_x;
                size_t right = y * output->width + max_x;
                if (output->cells[left] == spec->background) {
                    output->cells[left] = spec->output_color;
                }
                if (output->cells[right] == spec->background) {
                    output->cells[right] = spec->output_color;
                }
            }
        } else {
            arc_marker_polyline_draw_hv_segment(output,
                                                spec->background,
                                                spec->output_color,
                                                x1,
                                                y1,
                                                x2,
                                                y2,
                                                spec->y_anchor == 1);
        }
    }
    return 1;
}

static uint8_t arc_line_projection_color(const ARCGrid *input,
                                         uint8_t background,
                                         uint8_t output_color,
                                         int axis,
                                         size_t index,
                                         int mode)
{
    size_t counts[ARC_COLOR_COUNT];
    memset(counts, 0, sizeof(counts));
    uint8_t first = background;
    uint8_t last = background;
    size_t total = 0;
    size_t extent = axis == 0 ? input->width : input->height;
    for (size_t offset = 0; offset < extent; ++offset) {
        size_t x = axis == 0 ? offset : index;
        size_t y = axis == 0 ? index : offset;
        uint8_t color = input->cells[y * input->width + x];
        if (color == background || color >= ARC_COLOR_COUNT) {
            continue;
        }
        if (total == 0) {
            first = color;
        }
        last = color;
        ++counts[color];
        ++total;
    }
    if (total == 0) {
        return background;
    }
    if (mode == 0) {
        return first;
    }
    if (mode == 1) {
        return last;
    }
    if (mode == 3) {
        return output_color;
    }
    uint8_t best = first;
    size_t best_count = 0;
    for (uint8_t color = 0; color < ARC_COLOR_COUNT; ++color) {
        if (counts[color] > best_count) {
            best = color;
            best_count = counts[color];
        }
    }
    return best;
}

static int arc_apply_sketch_axis_projection_compress(const ARCSketchProgramSpec *spec,
                                                     const ARCGrid *input,
                                                     ARCGrid *output)
{
    if (spec == NULL || input == NULL || output == NULL ||
        input->width == 0 || input->height == 0) {
        return 0;
    }
    int axis = spec->x_anchor == 1 ? 1 : 0;
    int mode = spec->y_anchor;
    if (axis == 0) {
        arc_init_grid(output, 1, input->height);
        for (size_t y = 0; y < input->height; ++y) {
            output->cells[y] =
                arc_line_projection_color(input,
                                          spec->background,
                                          spec->output_color,
                                          axis,
                                          y,
                                          mode);
        }
    } else {
        arc_init_grid(output, input->width, 1);
        for (size_t x = 0; x < input->width; ++x) {
            output->cells[x] =
                arc_line_projection_color(input,
                                          spec->background,
                                          spec->output_color,
                                          axis,
                                          x,
                                          mode);
        }
    }
    return 1;
}

static int arc_find_separator_ranges_for_color(const ARCGrid *input,
                                               uint8_t separator,
                                               ARCRange *row_ranges,
                                               size_t *row_count_out,
                                               ARCRange *col_ranges,
                                               size_t *col_count_out)
{
    size_t full_rows[ARC_MAX_GRID_HEIGHT];
    size_t full_cols[ARC_MAX_GRID_WIDTH];
    size_t full_row_count =
        arc_full_color_rows(input, separator, full_rows, ARC_MAX_GRID_HEIGHT);
    size_t full_col_count =
        arc_full_color_cols(input, separator, full_cols, ARC_MAX_GRID_WIDTH);
    if (full_row_count == 0 && full_col_count == 0) {
        return 0;
    }
    if (row_ranges != NULL && row_count_out != NULL) {
        *row_count_out = arc_ranges_from_separators(input->height,
                                                    full_rows,
                                                    full_row_count,
                                                    row_ranges,
                                                    ARC_MAX_GRID_HEIGHT);
    }
    if (col_ranges != NULL && col_count_out != NULL) {
        *col_count_out = arc_ranges_from_separators(input->width,
                                                    full_cols,
                                                    full_col_count,
                                                    col_ranges,
                                                    ARC_MAX_GRID_WIDTH);
    }
    return 1;
}

static int arc_detect_separator_color(const ARCGrid *input,
                                      uint8_t background,
                                      uint8_t *separator_out)
{
    uint8_t found = 0;
    size_t found_count = 0;
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
        if (full_row_count == 0 && full_col_count == 0) {
            continue;
        }
        found = color;
        ++found_count;
    }
    if (found_count != 1) {
        return 0;
    }
    *separator_out = found;
    return 1;
}

static size_t arc_panel_non_background_count(const ARCGrid *input,
                                             const ARCRange *row_range,
                                             const ARCRange *col_range,
                                             uint8_t background,
                                             uint8_t separator)
{
    size_t count = 0;
    for (size_t y = row_range->min; y <= row_range->max; ++y) {
        for (size_t x = col_range->min; x <= col_range->max; ++x) {
            uint8_t color = input->cells[y * input->width + x];
            if (color != background && color != separator) {
                ++count;
            }
        }
    }
    return count;
}

static int arc_panel_content_bbox(const ARCGrid *input,
                                  const ARCRange *row_range,
                                  const ARCRange *col_range,
                                  uint8_t background,
                                  uint8_t separator,
                                  size_t *min_x_out,
                                  size_t *min_y_out,
                                  size_t *max_x_out,
                                  size_t *max_y_out)
{
    int found = 0;
    size_t min_x = col_range->max;
    size_t min_y = row_range->max;
    size_t max_x = col_range->min;
    size_t max_y = row_range->min;
    for (size_t y = row_range->min; y <= row_range->max; ++y) {
        for (size_t x = col_range->min; x <= col_range->max; ++x) {
            uint8_t color = input->cells[y * input->width + x];
            if (color == background || color == separator) {
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
    *min_x_out = min_x;
    *min_y_out = min_y;
    *max_x_out = max_x;
    *max_y_out = max_y;
    return 1;
}

static int arc_copy_panel_range(const ARCGrid *input,
                                const ARCRange *row_range,
                                const ARCRange *col_range,
                                ARCGrid *output)
{
    size_t width = col_range->max - col_range->min + 1u;
    size_t height = row_range->max - row_range->min + 1u;
    if (width == 0 || height == 0 ||
        width > ARC_MAX_GRID_WIDTH || height > ARC_MAX_GRID_HEIGHT) {
        return 0;
    }
    arc_init_grid(output, width, height);
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            output->cells[y * output->width + x] =
                input->cells[(row_range->min + y) * input->width + col_range->min + x];
        }
    }
    return 1;
}

typedef struct {
    size_t index;
    uint8_t color;
    size_t support;
} ARCSeparatorLineCandidate;

static size_t arc_tolerant_separator_rows(const ARCGrid *input,
                                          uint8_t background,
                                          uint8_t requested_color,
                                          size_t tolerance,
                                          ARCSeparatorLineCandidate *out,
                                          size_t capacity)
{
    size_t count = 0;
    for (size_t y = 0; y < input->height; ++y) {
        size_t color_counts[ARC_COLOR_COUNT];
        memset(color_counts, 0, sizeof(color_counts));
        for (size_t x = 0; x < input->width; ++x) {
            uint8_t color = input->cells[y * input->width + x];
            if (color < ARC_COLOR_COUNT) {
                ++color_counts[color];
            }
        }
        for (uint8_t color = 0; color < ARC_COLOR_COUNT; ++color) {
            if (color == background || color_counts[color] == 0) {
                continue;
            }
            if (requested_color != background && color != requested_color) {
                continue;
            }
            if (color_counts[color] + tolerance < input->width) {
                continue;
            }
            if (count < capacity) {
                out[count].index = y;
                out[count].color = color;
                out[count].support = color_counts[color];
            }
            ++count;
        }
    }
    return count < capacity ? count : capacity;
}

static size_t arc_tolerant_separator_cols(const ARCGrid *input,
                                          uint8_t background,
                                          uint8_t requested_color,
                                          size_t tolerance,
                                          ARCSeparatorLineCandidate *out,
                                          size_t capacity)
{
    size_t count = 0;
    for (size_t x = 0; x < input->width; ++x) {
        size_t color_counts[ARC_COLOR_COUNT];
        memset(color_counts, 0, sizeof(color_counts));
        for (size_t y = 0; y < input->height; ++y) {
            uint8_t color = input->cells[y * input->width + x];
            if (color < ARC_COLOR_COUNT) {
                ++color_counts[color];
            }
        }
        for (uint8_t color = 0; color < ARC_COLOR_COUNT; ++color) {
            if (color == background || color_counts[color] == 0) {
                continue;
            }
            if (requested_color != background && color != requested_color) {
                continue;
            }
            if (color_counts[color] + tolerance < input->height) {
                continue;
            }
            if (count < capacity) {
                out[count].index = x;
                out[count].color = color;
                out[count].support = color_counts[color];
            }
            ++count;
        }
    }
    return count < capacity ? count : capacity;
}

static int arc_apply_sketch_separator_intersection_color(const ARCSketchProgramSpec *spec,
                                                        const ARCGrid *input,
                                                        ARCGrid *output)
{
    ARCSeparatorLineCandidate rows[ARC_MAX_GRID_HEIGHT * ARC_COLOR_COUNT];
    ARCSeparatorLineCandidate cols[ARC_MAX_GRID_WIDTH * ARC_COLOR_COUNT];
    size_t tolerance = spec == NULL ? 0 : spec->period;
    if (spec == NULL || input == NULL || output == NULL ||
        input->width == 0 || input->height == 0 ||
        tolerance > 4u) {
        return 0;
    }
    size_t row_count = arc_tolerant_separator_rows(input,
                                                   spec->background,
                                                   spec->color,
                                                   tolerance,
                                                   rows,
                                                   sizeof(rows) / sizeof(rows[0]));
    size_t col_count = arc_tolerant_separator_cols(input,
                                                   spec->background,
                                                   spec->output_color,
                                                   tolerance,
                                                   cols,
                                                   sizeof(cols) / sizeof(cols[0]));
    if (row_count == 0 || col_count == 0) {
        return 0;
    }
    size_t best_row = 0;
    size_t best_col = 0;
    size_t best_score = 0;
    uint8_t best_color = spec->background;
    int found = 0;
    for (size_t row = 0; row < row_count; ++row) {
        for (size_t col = 0; col < col_count; ++col) {
            uint8_t value = input->cells[rows[row].index * input->width + cols[col].index];
            if (value == spec->background) {
                continue;
            }
            size_t score = rows[row].support + cols[col].support;
            if (!found || score > best_score ||
                (score == best_score && value < best_color)) {
                found = 1;
                best_score = score;
                best_row = row;
                best_col = col;
                best_color = value;
            }
        }
    }
    if (!found) {
        return 0;
    }
    (void)best_row;
    (void)best_col;
    arc_init_grid(output, 1, 1);
    output->cells[0] = best_color;
    return 1;
}

static int arc_separator_relation_mask_value(uint8_t first,
                                             uint8_t second,
                                             uint8_t background,
                                             uint8_t separator,
                                             size_t mode)
{
    int first_fg = first != background && first != separator;
    int second_fg = second != background && second != separator;
    switch (mode) {
    case 0:
        return first_fg || second_fg;
    case 1:
        return first_fg && second_fg;
    case 2:
        return first_fg != second_fg;
    case 3:
        return !first_fg && !second_fg;
    case 4:
        return first_fg && !second_fg;
    case 5:
        return !first_fg && second_fg;
    default:
        return 0;
    }
}

static int arc_apply_separator_panel_relation_orientation(
    const ARCSketchProgramSpec *spec,
    const ARCGrid *input,
    ARCGrid *output,
    const ARCRange *row_ranges,
    size_t row_count,
    const ARCRange *col_ranges,
    size_t col_count,
    uint8_t separator,
    int orientation)
{
    ARCRange first_rows;
    ARCRange first_cols;
    ARCRange second_rows;
    ARCRange second_cols;
    size_t width = 0;
    size_t height = 0;
    if (orientation == 1) {
        if (row_count != 1u || col_count != 2u) {
            return 0;
        }
        first_rows = row_ranges[0];
        second_rows = row_ranges[0];
        first_cols = col_ranges[0];
        second_cols = col_ranges[1];
    } else if (orientation == 2) {
        if (row_count != 2u || col_count != 1u) {
            return 0;
        }
        first_rows = row_ranges[0];
        second_rows = row_ranges[1];
        first_cols = col_ranges[0];
        second_cols = col_ranges[0];
    } else {
        return 0;
    }
    width = first_cols.max - first_cols.min + 1u;
    height = first_rows.max - first_rows.min + 1u;
    if (width == 0 || height == 0 ||
        width != second_cols.max - second_cols.min + 1u ||
        height != second_rows.max - second_rows.min + 1u ||
        width > ARC_MAX_GRID_WIDTH || height > ARC_MAX_GRID_HEIGHT ||
        spec->period > 5u) {
        return 0;
    }
    arc_init_grid(output, width, height);
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            uint8_t first =
                input->cells[(first_rows.min + y) * input->width + first_cols.min + x];
            uint8_t second =
                input->cells[(second_rows.min + y) * input->width + second_cols.min + x];
            output->cells[y * output->width + x] =
                arc_separator_relation_mask_value(first,
                                                  second,
                                                  spec->background,
                                                  separator,
                                                  spec->period)
                    ? spec->output_color
                    : spec->background;
        }
    }
    return 1;
}

static int arc_apply_sketch_separator_panel_relation_mask(
    const ARCSketchProgramSpec *spec,
    const ARCGrid *input,
    ARCGrid *output)
{
    ARCRange row_ranges[ARC_MAX_GRID_HEIGHT];
    ARCRange col_ranges[ARC_MAX_GRID_WIDTH];
    size_t row_count = 0;
    size_t col_count = 0;
    uint8_t separator = spec == NULL ? 0 : spec->color;
    if (spec == NULL || input == NULL || output == NULL ||
        spec->period > 5u || separator >= ARC_COLOR_COUNT) {
        return 0;
    }
    if (separator == spec->background &&
        !arc_detect_separator_color(input, spec->background, &separator)) {
        return 0;
    }
    if (!arc_find_separator_ranges_for_color(input,
                                             separator,
                                             row_ranges,
                                             &row_count,
                                             col_ranges,
                                             &col_count)) {
        return 0;
    }
    if (spec->x_anchor == 1 || spec->x_anchor == 2) {
        return arc_apply_separator_panel_relation_orientation(spec,
                                                             input,
                                                             output,
                                                             row_ranges,
                                                             row_count,
                                                             col_ranges,
                                                             col_count,
                                                             separator,
                                                             spec->x_anchor);
    }
    return arc_apply_separator_panel_relation_orientation(spec,
                                                         input,
                                                         output,
                                                         row_ranges,
                                                         row_count,
                                                         col_ranges,
                                                         col_count,
                                                         separator,
                                                         1) ||
           arc_apply_separator_panel_relation_orientation(spec,
                                                         input,
                                                         output,
                                                         row_ranges,
                                                         row_count,
                                                         col_ranges,
                                                         col_count,
                                                         separator,
                                                         2);
}

static int arc_apply_sketch_separator_panel_map(const ARCSketchProgramSpec *spec,
                                                const ARCGrid *input,
                                                ARCGrid *output)
{
    ARCRange row_ranges[ARC_MAX_GRID_HEIGHT];
    ARCRange col_ranges[ARC_MAX_GRID_WIDTH];
    size_t row_count = 0;
    size_t col_count = 0;
    uint8_t separator = spec == NULL ? 0 : spec->color;
    if (spec == NULL || input == NULL || output == NULL ||
        separator >= ARC_COLOR_COUNT) {
        return 0;
    }
    if (separator == spec->background &&
        !arc_detect_separator_color(input, spec->background, &separator)) {
        return 0;
    }
    if (
        !arc_find_separator_ranges_for_color(input,
                                             separator,
                                             row_ranges,
                                             &row_count,
                                             col_ranges,
                                             &col_count) ||
        row_count == 0 || col_count == 0) {
        return 0;
    }
    if (spec->x_anchor == 0) {
        size_t kept_rows = 0;
        size_t kept_cols = 0;
        int keep_rows[ARC_MAX_GRID_HEIGHT];
        int keep_cols[ARC_MAX_GRID_WIDTH];
        for (size_t y = 0; y < input->height; ++y) {
            int full_separator = 1;
            for (size_t x = 0; x < input->width; ++x) {
                if (input->cells[y * input->width + x] != separator) {
                    full_separator = 0;
                    break;
                }
            }
            keep_rows[y] = !full_separator;
            kept_rows += keep_rows[y] ? 1u : 0u;
        }
        for (size_t x = 0; x < input->width; ++x) {
            int full_separator = 1;
            for (size_t y = 0; y < input->height; ++y) {
                if (input->cells[y * input->width + x] != separator) {
                    full_separator = 0;
                    break;
                }
            }
            keep_cols[x] = !full_separator;
            kept_cols += keep_cols[x] ? 1u : 0u;
        }
        if (kept_rows == 0 || kept_cols == 0) {
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
    if (spec->x_anchor == 3) {
        size_t cell_width = 0;
        size_t cell_height = 0;
        size_t bbox_min_x[ARC_MAX_GRID_HEIGHT][ARC_MAX_GRID_WIDTH];
        size_t bbox_min_y[ARC_MAX_GRID_HEIGHT][ARC_MAX_GRID_WIDTH];
        size_t bbox_max_x[ARC_MAX_GRID_HEIGHT][ARC_MAX_GRID_WIDTH];
        size_t bbox_max_y[ARC_MAX_GRID_HEIGHT][ARC_MAX_GRID_WIDTH];
        uint8_t panel_has_content[ARC_MAX_GRID_HEIGHT][ARC_MAX_GRID_WIDTH];
        memset(panel_has_content, 0, sizeof(panel_has_content));
        for (size_t row = 0; row < row_count; ++row) {
            for (size_t col = 0; col < col_count; ++col) {
                if (!arc_panel_content_bbox(input,
                                            &row_ranges[row],
                                            &col_ranges[col],
                                            spec->background,
                                            separator,
                                            &bbox_min_x[row][col],
                                            &bbox_min_y[row][col],
                                            &bbox_max_x[row][col],
                                            &bbox_max_y[row][col])) {
                    continue;
                }
                size_t width = bbox_max_x[row][col] - bbox_min_x[row][col] + 1u;
                size_t height = bbox_max_y[row][col] - bbox_min_y[row][col] + 1u;
                if (width > cell_width) {
                    cell_width = width;
                }
                if (height > cell_height) {
                    cell_height = height;
                }
                panel_has_content[row][col] = 1;
            }
        }
        if (cell_width == 0 || cell_height == 0 ||
            cell_width * col_count > ARC_MAX_GRID_WIDTH ||
            cell_height * row_count > ARC_MAX_GRID_HEIGHT) {
            return 0;
        }
        arc_init_grid(output, cell_width * col_count, cell_height * row_count);
        for (size_t cell = 0; cell < arc_grid_cells(output); ++cell) {
            output->cells[cell] = spec->background;
        }
        for (size_t row = 0; row < row_count; ++row) {
            for (size_t col = 0; col < col_count; ++col) {
                if (!panel_has_content[row][col]) {
                    continue;
                }
                size_t width = bbox_max_x[row][col] - bbox_min_x[row][col] + 1u;
                size_t height = bbox_max_y[row][col] - bbox_min_y[row][col] + 1u;
                for (size_t y = 0; y < height; ++y) {
                    for (size_t x = 0; x < width; ++x) {
                        uint8_t value =
                            input->cells[(bbox_min_y[row][col] + y) * input->width +
                                         bbox_min_x[row][col] + x];
                        if (value == separator) {
                            value = spec->background;
                        }
                        output->cells[(row * cell_height + y) * output->width +
                                      col * cell_width + x] = value;
                    }
                }
            }
        }
        return 1;
    }
    size_t best_row = 0;
    size_t best_col = 0;
    size_t best_count = 0;
    int found = 0;
    for (size_t row = 0; row < row_count; ++row) {
        for (size_t col = 0; col < col_count; ++col) {
            size_t count = arc_panel_non_background_count(input,
                                                          &row_ranges[row],
                                                          &col_ranges[col],
                                                          spec->background,
                                                          separator);
            if (count == 0) {
                continue;
            }
            int better = !found || count > best_count;
            if (spec->x_anchor == 2) {
                better = !found || count < best_count;
            }
            if (better) {
                found = 1;
                best_count = count;
                best_row = row;
                best_col = col;
            }
        }
    }
    if (!found) {
        return 0;
    }
    return arc_copy_panel_range(input, &row_ranges[best_row], &col_ranges[best_col], output);
}
