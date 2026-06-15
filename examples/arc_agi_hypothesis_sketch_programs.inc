static const NNHypothesisTemplate *arc_hypothesis_template_sketch_program(void)
{
    static NNHypothesisTemplate template_ref;
    static int initialized = 0;
    if (initialized) {
        return &template_ref;
    }
    nn_hypothesis_template_init(&template_ref,
                                "arc_agi",
                                "arc_sketch_program",
                                1036,
                                NN_PROGRAM_VALUE_GRID,
                                NN_PROGRAM_VALUE_GRID);
    nn_hypothesis_template_add_stage(&template_ref,
                                     "select",
                                     NN_PROGRAM_KIND_SELECTOR,
                                     NN_PROGRAM_VALUE_GRID,
                                     NN_PROGRAM_VALUE_GRID,
                                     360);
    nn_hypothesis_template_add_stage(&template_ref,
                                     "apply",
                                     NN_PROGRAM_KIND_TRANSFORM,
                                     NN_PROGRAM_VALUE_GRID,
                                     NN_PROGRAM_VALUE_GRID,
                                     361);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "sketch.selector",
                                    NN_HYPOTHESIS_SLOT_CUSTOM,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "sketch.op",
                                    NN_HYPOTHESIS_SLOT_CUSTOM,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "sketch.background",
                                    NN_HYPOTHESIS_SLOT_COLOR,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "sketch.color",
                                    NN_HYPOTHESIS_SLOT_COLOR,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "sketch.output_color",
                                    NN_HYPOTHESIS_SLOT_COLOR,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "sketch.x_anchor",
                                    NN_HYPOTHESIS_SLOT_INTEGER,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "sketch.y_anchor",
                                    NN_HYPOTHESIS_SLOT_INTEGER,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "sketch.x_offset",
                                    NN_HYPOTHESIS_SLOT_OFFSET,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "sketch.y_offset",
                                    NN_HYPOTHESIS_SLOT_OFFSET,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "sketch.width",
                                    NN_HYPOTHESIS_SLOT_INTEGER,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "sketch.height",
                                    NN_HYPOTHESIS_SLOT_INTEGER,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "sketch.period",
                                    NN_HYPOTHESIS_SLOT_INTEGER,
                                    1, 0.0, 0);
    initialized = 1;
    return &template_ref;
}

static int arc_sketch_program_spec_from_hypothesis(const NNHypothesis *hypothesis,
                                                   ARCSketchProgramSpec *spec)
{
    int64_t selector = 0;
    int64_t op = 0;
    int64_t background = 0;
    int64_t color = 0;
    int64_t output_color = 0;
    int64_t x_anchor = 0;
    int64_t y_anchor = 0;
    int64_t x_offset = 0;
    int64_t y_offset = 0;
    int64_t width = 0;
    int64_t height = 0;
    int64_t period = 0;
    if (hypothesis == NULL || spec == NULL ||
        hypothesis->template_ref != arc_hypothesis_template_sketch_program() ||
        !nn_hypothesis_is_fully_bound(hypothesis) ||
        !nn_hypothesis_get(hypothesis, "sketch.selector", NULL, &selector) ||
        !nn_hypothesis_get(hypothesis, "sketch.op", NULL, &op) ||
        !nn_hypothesis_get(hypothesis, "sketch.background", NULL, &background) ||
        !nn_hypothesis_get(hypothesis, "sketch.color", NULL, &color) ||
        !nn_hypothesis_get(hypothesis, "sketch.output_color", NULL, &output_color) ||
        !nn_hypothesis_get(hypothesis, "sketch.x_anchor", NULL, &x_anchor) ||
        !nn_hypothesis_get(hypothesis, "sketch.y_anchor", NULL, &y_anchor) ||
        !nn_hypothesis_get(hypothesis, "sketch.x_offset", NULL, &x_offset) ||
        !nn_hypothesis_get(hypothesis, "sketch.y_offset", NULL, &y_offset) ||
        !nn_hypothesis_get(hypothesis, "sketch.width", NULL, &width) ||
        !nn_hypothesis_get(hypothesis, "sketch.height", NULL, &height) ||
        !nn_hypothesis_get(hypothesis, "sketch.period", NULL, &period) ||
        background < 0 || background >= ARC_COLOR_COUNT ||
        color < 0 || color >= ARC_COLOR_COUNT ||
        output_color < 0 || output_color >= ARC_COLOR_COUNT ||
        width < 0 || height < 0 || period < 0) {
        return 0;
    }
    spec->selector = (int)selector;
    spec->op = (int)op;
    spec->background = (uint8_t)background;
    spec->color = (uint8_t)color;
    spec->output_color = (uint8_t)output_color;
    spec->x_anchor = (int)x_anchor;
    spec->y_anchor = (int)y_anchor;
    spec->x_offset = (int)x_offset;
    spec->y_offset = (int)y_offset;
    spec->width = (size_t)width;
    spec->height = (size_t)height;
    spec->period = (size_t)period;
    return 1;
}

static int arc_sketch_copy_region(const ARCGrid *input,
                                  size_t origin_x,
                                  size_t origin_y,
                                  size_t width,
                                  size_t height,
                                  ARCGrid *output)
{
    if (input == NULL || output == NULL ||
        width == 0 || height == 0 ||
        origin_x + width > input->width ||
        origin_y + height > input->height) {
        return 0;
    }
    arc_init_grid(output, width, height);
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            output->cells[y * output->width + x] =
                input->cells[(origin_y + y) * input->width + origin_x + x];
        }
    }
    return 1;
}

static int arc_sketch_macrocell_active(size_t x, size_t y, size_t period, int motif_kind)
{
    if (period == 0) {
        return 0;
    }
    size_t mid = period / 2u;
    switch (motif_kind) {
    case 0:
        return 1;
    case 1:
        return (x == 0 || x + 1u == period) && (y == 0 || y + 1u == period);
    case 2:
        return x == 0 || y == 0 || x + 1u == period || y + 1u == period;
    case 3:
        return x == y || x + y + 1u == period;
    case 4:
        return x == mid && y == mid;
    case 5:
        return x == mid && y == mid;
    case 6:
        return x == mid || y == mid;
    case 7:
        return y == mid;
    case 8:
        return x == mid;
    case 9:
        return x == y;
    case 10:
        return x + y + 1u == period;
    case 11:
        return ((x + y) % 2u) == 0u;
    case 12:
        return ((x + y) % 2u) == 1u;
    case 13:
        return !(x == mid && y == mid);
    case 14:
        return x > 0 && y > 0 && x + 1u < period && y + 1u < period;
    case 15:
        return (x == 0 || y == 0 || x + 1u == period || y + 1u == period) &&
               !((x == 0 || x + 1u == period) && (y == 0 || y + 1u == period));
    default:
        return x == mid && y == mid;
    }
}

static int arc_apply_sketch_macrocell_substitute(const ARCSketchProgramSpec *spec,
                                                 const ARCGrid *input,
                                                 ARCGrid *output)
{
    if (spec == NULL || input == NULL || output == NULL ||
        spec->period == 0 ||
        input->width * spec->period > ARC_MAX_GRID_WIDTH ||
        input->height * spec->period > ARC_MAX_GRID_HEIGHT) {
        return 0;
    }
    arc_init_grid(output, input->width * spec->period, input->height * spec->period);
    for (size_t y = 0; y < output->height; ++y) {
        for (size_t x = 0; x < output->width; ++x) {
            output->cells[y * output->width + x] = spec->background;
        }
    }
    int any_trigger = spec->color == spec->background;
    for (size_t y = 0; y < input->height; ++y) {
        for (size_t x = 0; x < input->width; ++x) {
            uint8_t value = input->cells[y * input->width + x];
            if ((any_trigger && value == spec->background) ||
                (!any_trigger && value != spec->color)) {
                continue;
            }
            uint8_t motif_color =
                spec->output_color == spec->background ? value : spec->output_color;
            for (size_t yy = 0; yy < spec->period; ++yy) {
                for (size_t xx = 0; xx < spec->period; ++xx) {
                    int active = 0;
                    if (spec->x_anchor == 5 &&
                        input->width == spec->period &&
                        input->height == spec->period) {
                        active = input->cells[yy * input->width + xx] == spec->background;
                    } else {
                        active = arc_sketch_macrocell_active(xx,
                                                             yy,
                                                             spec->period,
                                                             spec->x_anchor);
                    }
                    if (!active) {
                        continue;
                    }
                    size_t out_x = x * spec->period + xx;
                    size_t out_y = y * spec->period + yy;
                    output->cells[out_y * output->width + out_x] = motif_color;
                }
            }
        }
    }
    return 1;
}

static int arc_apply_sketch_lattice_sample(const ARCSketchProgramSpec *spec,
                                           const ARCGrid *input,
                                           ARCGrid *output)
{
    if (spec != NULL && spec->x_anchor == 1) {
        if (input == NULL || output == NULL ||
            input->width != input->height ||
            input->width < 3u ||
            (input->width % 2u) == 0 ||
            spec->width != 3u ||
            spec->height != 3u) {
            return 0;
        }
        size_t last = input->width - 1u;
        arc_init_grid(output, spec->width, spec->height);
        for (size_t cell = 0; cell < arc_grid_cells(output); ++cell) {
            output->cells[cell] = spec->background;
        }
        for (size_t y = 0; y < input->height; ++y) {
            for (size_t x = 0; x < input->width; ++x) {
                uint8_t value = input->cells[y * input->width + x];
                if (value == spec->background) {
                    continue;
                }
                if (x != y && x + y != last) {
                    return 0;
                }
                size_t out_x = 1u;
                size_t out_y = 1u;
                if (x < y) {
                    out_x = 0u;
                } else if (x > y) {
                    out_x = 2u;
                } else if (x == 0u) {
                    out_x = 0u;
                } else if (x == last) {
                    out_x = 2u;
                }
                if (x + y < last) {
                    out_y = 0u;
                } else if (x + y > last) {
                    out_y = 2u;
                } else if (y == 0u) {
                    out_y = 0u;
                } else if (y == last) {
                    out_y = 2u;
                }
                uint8_t *target = &output->cells[out_y * output->width + out_x];
                if (*target != spec->background && *target != value) {
                    return 0;
                }
                *target = value;
            }
        }
        return 1;
    }
    if (spec == NULL || input == NULL || output == NULL ||
        spec->period == 0 || spec->width == 0 || spec->height == 0 ||
        spec->width > ARC_MAX_GRID_WIDTH || spec->height > ARC_MAX_GRID_HEIGHT) {
        return 0;
    }
    int64_t last_x =
        (int64_t)spec->x_offset + (int64_t)(spec->width - 1u) * (int64_t)spec->period;
    int64_t last_y =
        (int64_t)spec->y_offset + (int64_t)(spec->height - 1u) * (int64_t)spec->period;
    if (spec->x_offset < 0 || spec->y_offset < 0 ||
        last_x < 0 || last_y < 0 ||
        last_x >= (int64_t)input->width || last_y >= (int64_t)input->height) {
        return 0;
    }
    arc_init_grid(output, spec->width, spec->height);
    for (size_t y = 0; y < spec->height; ++y) {
        for (size_t x = 0; x < spec->width; ++x) {
            size_t in_x = (size_t)spec->x_offset + x * spec->period;
            size_t in_y = (size_t)spec->y_offset + y * spec->period;
            output->cells[y * output->width + x] =
                input->cells[in_y * input->width + in_x];
        }
    }
    return 1;
}

static int arc_apply_sketch_point_line_project(const ARCSketchProgramSpec *spec,
                                               const ARCGrid *input,
                                               ARCGrid *output)
{
    if (spec == NULL || input == NULL || output == NULL ||
        input->width == 0 || input->height == 0) {
        return 0;
    }
    typedef struct {
        size_t x;
        size_t y;
        uint8_t color;
    } ARCPointLineSeed;
    ARCPointLineSeed seeds[ARC_COLOR_COUNT];
    size_t seed_count = 0;
    for (size_t y = 0; y < input->height; ++y) {
        for (size_t x = 0; x < input->width; ++x) {
            uint8_t color = input->cells[y * input->width + x];
            if (color == spec->background) {
                continue;
            }
            if (seed_count >= sizeof(seeds) / sizeof(seeds[0])) {
                return 0;
            }
            seeds[seed_count].x = x;
            seeds[seed_count].y = y;
            seeds[seed_count].color = color;
            ++seed_count;
        }
    }
    if (seed_count < 2u) {
        return 0;
    }
    arc_init_grid(output, input->width, input->height);
    for (size_t cell = 0; cell < arc_grid_cells(output); ++cell) {
        output->cells[cell] = spec->background;
    }
    for (size_t seed = 0; seed < seed_count; ++seed) {
        for (size_t y = 0; y < output->height; ++y) {
            output->cells[y * output->width + seeds[seed].x] = seeds[seed].color;
        }
        for (size_t x = 0; x < output->width; ++x) {
            output->cells[seeds[seed].y * output->width + x] = seeds[seed].color;
        }
    }
    for (size_t lhs = 0; lhs < seed_count; ++lhs) {
        for (size_t rhs = lhs + 1u; rhs < seed_count; ++rhs) {
            if (seeds[lhs].x == seeds[rhs].x || seeds[lhs].y == seeds[rhs].y) {
                continue;
            }
            output->cells[seeds[lhs].y * output->width + seeds[rhs].x] =
                spec->output_color;
            output->cells[seeds[rhs].y * output->width + seeds[lhs].x] =
                spec->output_color;
        }
    }
    return 1;
}

static uint8_t arc_grid_most_common_non_background_color(const ARCGrid *grid,
                                                         uint8_t background)
{
    size_t counts[ARC_COLOR_COUNT];
    memset(counts, 0, sizeof(counts));
    if (grid == NULL) {
        return background;
    }
    for (size_t cell = 0; cell < arc_grid_cells(grid); ++cell) {
        uint8_t color = grid->cells[cell];
        if (color < ARC_COLOR_COUNT && color != background) {
            ++counts[color];
        }
    }
    uint8_t best = background;
    size_t best_count = 0;
    for (uint8_t color = 0; color < ARC_COLOR_COUNT; ++color) {
        if (counts[color] > best_count) {
            best_count = counts[color];
            best = color;
        }
    }
    return best;
}

static size_t arc_grid_distinct_color_count(const ARCGrid *grid,
                                            uint8_t background,
                                            int include_background)
{
    uint8_t seen[ARC_COLOR_COUNT];
    size_t count = 0;
    memset(seen, 0, sizeof(seen));
    if (grid == NULL) {
        return 0;
    }
    for (size_t cell = 0; cell < arc_grid_cells(grid); ++cell) {
        uint8_t color = grid->cells[cell];
        if (color >= ARC_COLOR_COUNT ||
            (!include_background && color == background) ||
            seen[color]) {
            continue;
        }
        seen[color] = 1;
        ++count;
    }
    return count;
}

static int arc_select_any_color_grid_component(const ARCGrid *grid,
                                               uint8_t background,
                                               uint8_t selector_kind,
                                               uint8_t *color_out)
{
    if (grid == NULL || color_out == NULL) {
        return 0;
    }
    uint8_t visited[ARC_MAX_GRID_CELLS];
    size_t queue[ARC_MAX_GRID_CELLS];
    memset(visited, 0, sizeof(visited));
    int found = 0;
    int tied = 0;
    double best_score = 0.0;
    uint8_t best_color = background;
    if (selector_kind != ARC_OBJECT_SELECTOR_SMALLEST_COLOR &&
        selector_kind != ARC_OBJECT_SELECTOR_INTERIOR_COLOR &&
        selector_kind != ARC_OBJECT_SELECTOR_TOUCHING_BORDER_COLOR) {
        selector_kind = ARC_OBJECT_SELECTOR_LARGEST_COLOR;
    }
    size_t cells = arc_grid_cells(grid);
    for (size_t start = 0; start < cells; ++start) {
        uint8_t color = grid->cells[start];
        if (visited[start] || color == background) {
            continue;
        }
        size_t head = 0;
        size_t tail = 0;
        size_t count = 0;
        int touches_border = 0;
        visited[start] = 1;
        queue[tail++] = start;
        while (head < tail) {
            size_t cell = queue[head++];
            size_t x = cell % grid->width;
            size_t y = cell / grid->width;
            ++count;
            if (x == 0 || y == 0 ||
                x + 1u == grid->width ||
                y + 1u == grid->height) {
                touches_border = 1;
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
        double score = 0.0;
        switch (selector_kind) {
        case ARC_OBJECT_SELECTOR_SMALLEST_COLOR:
            score = -(double)count;
            break;
        case ARC_OBJECT_SELECTOR_INTERIOR_COLOR:
            if (touches_border) {
                continue;
            }
            score = (double)count;
            break;
        case ARC_OBJECT_SELECTOR_TOUCHING_BORDER_COLOR:
            if (!touches_border) {
                continue;
            }
            score = (double)count;
            break;
        default:
            score = (double)count;
            break;
        }
        if (!found || score > best_score) {
            best_color = color;
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
    *color_out = best_color;
    return 1;
}

static size_t arc_count_color_cells(const ARCGrid *grid, uint8_t color)
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

static size_t arc_count_color_components(const ARCGrid *grid,
                                         uint8_t color,
                                         size_t *min_x_out,
                                         size_t *min_y_out,
                                         size_t *max_x_out,
                                         size_t *max_y_out)
{
    uint8_t visited[ARC_MAX_GRID_CELLS];
    size_t queue[ARC_MAX_GRID_CELLS];
    size_t component_count = 0;
    memset(visited, 0, sizeof(visited));
    if (grid == NULL) {
        return 0;
    }
    if (min_x_out != NULL) {
        *min_x_out = grid->width;
    }
    if (min_y_out != NULL) {
        *min_y_out = grid->height;
    }
    if (max_x_out != NULL) {
        *max_x_out = 0;
    }
    if (max_y_out != NULL) {
        *max_y_out = 0;
    }
    size_t cells = arc_grid_cells(grid);
    for (size_t start = 0; start < cells; ++start) {
        if (visited[start] || grid->cells[start] != color) {
            continue;
        }
        ++component_count;
        size_t head = 0;
        size_t tail = 0;
        visited[start] = 1;
        queue[tail++] = start;
        while (head < tail) {
            size_t cell = queue[head++];
            size_t x = cell % grid->width;
            size_t y = cell / grid->width;
            if (min_x_out != NULL && x < *min_x_out) {
                *min_x_out = x;
            }
            if (min_y_out != NULL && y < *min_y_out) {
                *min_y_out = y;
            }
            if (max_x_out != NULL && x > *max_x_out) {
                *max_x_out = x;
            }
            if (max_y_out != NULL && y > *max_y_out) {
                *max_y_out = y;
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
    }
    return component_count;
}

static int arc_largest_component_bbox_excluding_color(const ARCGrid *grid,
                                                      uint8_t background,
                                                      uint8_t excluded,
                                                      uint8_t required_color,
                                                      uint8_t *color_out,
                                                      size_t *min_x_out,
                                                      size_t *min_y_out,
                                                      size_t *max_x_out,
                                                      size_t *max_y_out)
{
    uint8_t visited[ARC_MAX_GRID_CELLS];
    size_t queue[ARC_MAX_GRID_CELLS];
    memset(visited, 0, sizeof(visited));
    if (grid == NULL || color_out == NULL ||
        min_x_out == NULL || min_y_out == NULL ||
        max_x_out == NULL || max_y_out == NULL) {
        return 0;
    }
    int found = 0;
    int tied = 0;
    size_t best_count = 0;
    uint8_t best_color = background;
    size_t best_min_x = 0;
    size_t best_min_y = 0;
    size_t best_max_x = 0;
    size_t best_max_y = 0;
    size_t cells = arc_grid_cells(grid);
    for (size_t start = 0; start < cells; ++start) {
        uint8_t color = grid->cells[start];
        if (visited[start] ||
            color == background ||
            color == excluded ||
            (required_color != background && color != required_color)) {
            continue;
        }
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
        if (!found || count > best_count) {
            found = 1;
            tied = 0;
            best_count = count;
            best_color = color;
            best_min_x = min_x;
            best_min_y = min_y;
            best_max_x = max_x;
            best_max_y = max_y;
        } else if (count == best_count) {
            tied = 1;
        }
    }
    if (!found || tied) {
        return 0;
    }
    *color_out = best_color;
    *min_x_out = best_min_x;
    *min_y_out = best_min_y;
    *max_x_out = best_max_x;
    *max_y_out = best_max_y;
    return 1;
}

#include "arc_agi_visual_program.inc"
static int arc_apply_sketch_program_hypothesis(const NNHypothesis *hypothesis,
                                               const ARCGrid *input,
                                               ARCGrid *output)
{
    ARCSketchProgramSpec spec;
    if (!arc_sketch_program_spec_from_hypothesis(hypothesis, &spec) ||
        input == NULL || output == NULL) {
        return 0;
    }
    if (spec.op == ARC_SKETCH_OP_TILE_REPEAT &&
        spec.selector == ARC_SKETCH_SELECTOR_LATTICE) {
        ARCTransformCandidate candidate;
        arc_init_candidate(&candidate, ARC_RULE_TILE_MOTIF_PARITY_REPEAT, "sketch_tile_repeat");
        candidate.background_color = spec.background;
        candidate.scale = spec.period;
        return arc_apply_tile_motif_parity_repeat_candidate(&candidate, input, output);
    }
    if (spec.op == ARC_SKETCH_OP_MACROCELL_SUBSTITUTE) {
        return arc_apply_sketch_macrocell_substitute(&spec, input, output);
    }
    if (spec.op == ARC_SKETCH_OP_LATTICE_SAMPLE) {
        return arc_apply_sketch_lattice_sample(&spec, input, output);
    }
    if (spec.op == ARC_SKETCH_OP_COMPONENT_COLOR_SUMMARY) {
        return arc_apply_sketch_component_color_summary(&spec, input, output);
    }
    if (spec.op == ARC_SKETCH_OP_POINT_LINE_PROJECT) {
        return arc_apply_sketch_point_line_project(&spec, input, output);
    }
    if (spec.op == ARC_SKETCH_OP_OBJECT_MASK_EXTRACT) {
        return arc_apply_sketch_object_mask_extract(&spec, input, output);
    }
    if (spec.op == ARC_SKETCH_OP_AXIS_PROJECTION_COMPRESS) {
        return arc_apply_sketch_axis_projection_compress(&spec, input, output);
    }
    if (spec.op == ARC_SKETCH_OP_SEPARATOR_PANEL_MAP) {
        return arc_apply_sketch_separator_panel_map(&spec, input, output);
    }
    if (spec.op == ARC_SKETCH_OP_SEPARATOR_INTERSECTION_COLOR) {
        return arc_apply_sketch_separator_intersection_color(&spec, input, output);
    }
    if (spec.op == ARC_SKETCH_OP_SEPARATOR_PANEL_RELATION_MASK) {
        return arc_apply_sketch_separator_panel_relation_mask(&spec, input, output);
    }
    if (spec.op == ARC_SKETCH_OP_PALETTE_SCALE) {
        return arc_apply_sketch_palette_scale(&spec, input, output);
    }
    if (spec.op == ARC_SKETCH_OP_OBJECT_COLOR_TILE) {
        return arc_apply_sketch_object_color_tile(&spec, input, output);
    }
    if (spec.op == ARC_SKETCH_OP_MARKER_COUNT_MOTIF_TILE) {
        return arc_apply_sketch_marker_count_motif_tile(&spec, input, output);
    }
    if (spec.op == ARC_SKETCH_OP_COMPONENT_COUNT_LINE) {
        return arc_apply_sketch_component_count_line(&spec, input, output);
    }
    if (spec.op == ARC_SKETCH_OP_VISUAL_COMBINATOR) {
        return arc_apply_sketch_visual_combinator(&spec, input, output);
    }
    if (spec.op == ARC_SKETCH_OP_RECOLOR_CROP) {
        return arc_apply_sketch_recolor_crop(&spec, input, output);
    }
    if (spec.op == ARC_SKETCH_OP_MARKER_POLYLINE) {
        return arc_apply_sketch_marker_polyline(&spec, input, output);
    }
    if (spec.op == ARC_SKETCH_OP_EXTEND_ALIGNED_SEQUENCE) {
        return arc_apply_sketch_extend_aligned_sequence(&spec, input, output);
    }
    if (spec.op == ARC_SKETCH_OP_OBJECT_TRANSLATE) {
        return arc_apply_sketch_object_translate(&spec, input, output);
    }
    if (spec.op != ARC_SKETCH_OP_CROP && spec.op != ARC_SKETCH_OP_NORMALIZED_CROP) {
        return 0;
    }

    size_t origin_x = 0;
    size_t origin_y = 0;
    size_t width = spec.width;
    size_t height = spec.height;
    if (spec.selector == ARC_SKETCH_SELECTOR_GRID_PANEL) {
        if (!arc_panel_origin(input->width,
                              width,
                              spec.x_anchor,
                              spec.x_offset,
                              &origin_x) ||
            !arc_panel_origin(input->height,
                              height,
                              spec.y_anchor,
                              spec.y_offset,
                              &origin_y)) {
            return 0;
        }
    } else if (spec.selector == ARC_SKETCH_SELECTOR_FOREGROUND_BBOX_PANEL) {
        if (!arc_bbox_panel_origin(input,
                                   spec.background,
                                   width,
                                   height,
                                   spec.x_anchor,
                                   spec.y_anchor,
                                   spec.x_offset,
                                   spec.y_offset,
                                   &origin_x,
                                   &origin_y)) {
            return 0;
        }
    } else if (spec.selector == ARC_SKETCH_SELECTOR_COLOR_BBOX) {
        size_t max_x = 0;
        size_t max_y = 0;
        if (!arc_grid_color_bbox(input,
                                 spec.color,
                                 &origin_x,
                                 &origin_y,
                                 &max_x,
                                 &max_y)) {
            return 0;
        }
        width = max_x - origin_x + 1u;
        height = max_y - origin_y + 1u;
    } else {
        return 0;
    }
    if (!arc_sketch_copy_region(input, origin_x, origin_y, width, height, output)) {
        return 0;
    }
    if (spec.op == ARC_SKETCH_OP_NORMALIZED_CROP) {
        for (size_t cell = 0; cell < arc_grid_cells(output); ++cell) {
            if (output->cells[cell] == spec.background) {
                output->cells[cell] = spec.output_color;
            }
        }
    }
    return 1;
}

static int arc_candidate_from_sketch_program_hypothesis(const NNHypothesis *hypothesis,
                                                        ARCTransformCandidate *candidate)
{
    ARCSketchProgramSpec spec;
    if (candidate == NULL ||
        !arc_sketch_program_spec_from_hypothesis(hypothesis, &spec)) {
        return 0;
    }
    arc_init_candidate(candidate, ARC_RULE_HYPOTHESIS_APPLIED, "arc_sketch_program");
    if (!arc_store_hypothesis_candidate(candidate, hypothesis)) {
        return 0;
    }
    snprintf(candidate->name,
             sizeof(candidate->name),
             "hypothesis_%s_s%d_op%d_bg%u_c%u_%zux%zu_%+d_%+d_p%zu",
             hypothesis->template_ref->name,
             spec.selector,
             spec.op,
             (unsigned)spec.background,
             (unsigned)spec.color,
             spec.width,
             spec.height,
             spec.x_offset,
             spec.y_offset,
             spec.period);
    return 1;
}

static int arc_bind_sketch_program(NNHypothesis *hypothesis,
                                   int selector,
                                   int op,
                                   uint8_t background,
                                   uint8_t color,
                                   uint8_t output_color,
                                   int x_anchor,
                                   int y_anchor,
                                   int x_offset,
                                   int y_offset,
                                   size_t width,
                                   size_t height,
                                   size_t period)
{
    return nn_hypothesis_bind(hypothesis, "sketch.selector", (double)selector, selector) &&
           nn_hypothesis_bind(hypothesis, "sketch.op", (double)op, op) &&
           nn_hypothesis_bind(hypothesis, "sketch.background", (double)background, background) &&
           nn_hypothesis_bind(hypothesis, "sketch.color", (double)color, color) &&
           nn_hypothesis_bind(hypothesis, "sketch.output_color", (double)output_color, output_color) &&
           nn_hypothesis_bind(hypothesis, "sketch.x_anchor", (double)x_anchor, x_anchor) &&
           nn_hypothesis_bind(hypothesis, "sketch.y_anchor", (double)y_anchor, y_anchor) &&
           nn_hypothesis_bind(hypothesis, "sketch.x_offset", (double)x_offset, x_offset) &&
           nn_hypothesis_bind(hypothesis, "sketch.y_offset", (double)y_offset, y_offset) &&
           nn_hypothesis_bind(hypothesis, "sketch.width", (double)width, (int64_t)width) &&
           nn_hypothesis_bind(hypothesis, "sketch.height", (double)height, (int64_t)height) &&
           nn_hypothesis_bind(hypothesis, "sketch.period", (double)period, (int64_t)period);
}

static int arc_candidate_from_sketch_program_values(ARCTransformCandidate *candidate,
                                                    const ARCSketchProgramSpec *spec)
{
    NNHypothesis hypothesis;
    if (candidate == NULL || spec == NULL ||
        !nn_hypothesis_init(&hypothesis, arc_hypothesis_template_sketch_program()) ||
        !arc_bind_sketch_program(&hypothesis,
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
                                 spec->period) ||
        !arc_candidate_from_sketch_program_hypothesis(&hypothesis, candidate)) {
        return 0;
    }
    return 1;
}

static size_t arc_try_add_sketch_program(const ARCHypothesisTemplateBinding *binding,
                                         const ARCTask *task,
                                         ARCProgramLearnedState *learned,
                                         const ARCTransformCandidate *existing_candidates,
                                         size_t existing_count,
                                         ARCHypothesisProposal *proposals,
                                         size_t proposal_capacity,
                                         size_t proposal_count,
                                         ARCHypothesisSelectionStats *stats,
                                         size_t template_index,
                                         const ARCProgramTaskFeatures *task_features,
                                         int selector,
                                         int op,
                                         uint8_t background,
                                         uint8_t color,
                                         uint8_t output_color,
                                         int x_anchor,
                                         int y_anchor,
                                         int x_offset,
                                         int y_offset,
                                         size_t width,
                                         size_t height,
                                         size_t period)
{
    NNHypothesis hypothesis;
    ARCTransformCandidate candidate;
    if (!nn_hypothesis_init(&hypothesis, binding->template_fn()) ||
        !arc_bind_sketch_program(&hypothesis,
                                 selector,
                                 op,
                                 background,
                                 color,
                                 output_color,
                                 x_anchor,
                                 y_anchor,
                                 x_offset,
                                 y_offset,
                                 width,
                                 height,
                                 period) ||
        !binding->to_candidate(&hypothesis, &candidate)) {
        return proposal_count;
    }
    return arc_try_add_hypothesis_proposal(learned,
                                           task,
                                           existing_candidates,
                                           existing_count,
                                           proposals,
                                           proposal_capacity,
                                           proposal_count,
                                           stats,
                                           template_index,
                                           task_features,
                                           &hypothesis,
                                           &candidate,
                                           NULL,
                                           NULL);
}

static size_t arc_induce_sketch_panel_crop_hypotheses(
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
    const ARCGrid *first_input = &task->train[0].input;
    const ARCGrid *first_output = &task->train[0].output;
    if (!task->train[0].has_output ||
        first_output->width == 0 || first_output->height == 0 ||
        first_output->width > first_input->width ||
        first_output->height > first_input->height) {
        return proposal_count;
    }
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
                        int y_offset = (int)((int64_t)origin_y - y_base);
                        proposal_count = arc_try_add_sketch_program(
                            binding,
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
                            selector,
                            ARC_SKETCH_OP_CROP,
                            background,
                            0,
                            0,
                            x_mode,
                            y_mode,
                            x_offset,
                            y_offset,
                            first_output->width,
                            first_output->height,
                            0);
                    }
                }
            }
        }
    }
    return proposal_count;
}

static size_t arc_induce_sketch_color_bbox_hypotheses(
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
    const ARCGrid *first_input = &task->train[0].input;
    const ARCGrid *first_output = &task->train[0].output;
    if (!task->train[0].has_output || first_output->width == 0 ||
        first_output->height == 0) {
        return proposal_count;
    }
    uint8_t background = arc_grid_dominant_color(first_input);
    uint8_t colors[ARC_COLOR_COUNT];
    size_t color_count = arc_grid_nonzero_colors(first_input, colors, ARC_COLOR_COUNT);
    for (size_t color_index = 0; color_index < color_count; ++color_index) {
        uint8_t color = colors[color_index];
        size_t min_x = 0;
        size_t min_y = 0;
        size_t max_x = 0;
        size_t max_y = 0;
        if (!arc_grid_color_bbox(first_input, color, &min_x, &min_y, &max_x, &max_y)) {
            continue;
        }
        size_t width = max_x - min_x + 1u;
        size_t height = max_y - min_y + 1u;
        if (width != first_output->width ||
            height != first_output->height ||
            !arc_grid_crop_matches(first_input, first_output, min_x, min_y)) {
            continue;
        }
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
                                                    ARC_SKETCH_SELECTOR_COLOR_BBOX,
                                                    ARC_SKETCH_OP_CROP,
                                                    background,
                                                    color,
                                                    0,
                                                    0,
                                                    0,
                                                    0,
                                                    0,
                                                    0,
                                                    0,
                                                    0);
    }
    return proposal_count;
}

static size_t arc_induce_sketch_tile_hypotheses(
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
                                                        ARC_SKETCH_SELECTOR_LATTICE,
                                                        ARC_SKETCH_OP_TILE_REPEAT,
                                                        background,
                                                        0,
                                                        0,
                                                        0,
                                                        0,
                                                        0,
                                                        0,
                                                        0,
                                                        0,
                                                        period);
        }
    }
    return proposal_count;
}

static size_t arc_induce_sketch_program_hypotheses(const ARCHypothesisTemplateBinding *binding,
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
    ARCProgramTaskFeatures task_features;
    arc_program_task_features(task, &task_features);
    proposal_count =
        arc_induce_sketch_panel_crop_hypotheses(binding,
                                                task,
                                                learned,
                                                existing_candidates,
                                                existing_count,
                                                proposals,
                                                proposal_capacity,
                                                proposal_count,
                                                stats,
                                                template_index,
                                                &task_features);
    proposal_count =
        arc_induce_sketch_color_bbox_hypotheses(binding,
                                                task,
                                                learned,
                                                existing_candidates,
                                                existing_count,
                                                proposals,
                                                proposal_capacity,
                                                proposal_count,
                                                stats,
                                                template_index,
                                                &task_features);
    proposal_count =
        arc_induce_sketch_tile_hypotheses(binding,
                                          task,
                                          learned,
                                          existing_candidates,
                                          existing_count,
                                          proposals,
                                          proposal_capacity,
                                          proposal_count,
                                          stats,
                                          template_index,
                                          &task_features);
    if (learned != NULL && learned->arc_visual_only) {
        proposal_count =
            arc_induce_runtime_mined_sketch_hypotheses(binding,
                                                       task,
                                                       learned,
                                                       existing_candidates,
                                                       existing_count,
                                                       proposals,
                                                       proposal_capacity,
                                                       proposal_count,
                                                       stats,
                                                       template_index,
                                                       &task_features);
    }
    return proposal_count;
}
