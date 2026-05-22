#include "nn_layer.h"
#include "nn_train.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int dx;
    int dy;
    size_t matched;
    size_t errors;
    size_t placed;
} LayerTranslationScore;

typedef struct {
    LayerTranslationScore score;
    uint32_t action_id;
} LayerCandidate;

static uint32_t layer_rng_next(uint32_t *state)
{
    *state = (*state * 1664525u) + 1013904223u;
    return *state;
}

static size_t layer_cell_count(const NNBitGrid *grid)
{
    return nn_bitgrid_cell_count(grid);
}

static size_t layer_active_count(const NNBitGrid *grid)
{
    if (grid == NULL || grid->cells == NULL) {
        return 0u;
    }
    size_t count = 0u;
    size_t cells = layer_cell_count(grid);
    for (size_t i = 0u; i < cells; ++i) {
        count += (grid->cells[i] & 1u) != 0u ? 1u : 0u;
    }
    return count;
}

static int layer_offset(size_t coord, int delta, size_t limit, size_t *out)
{
    if (out == NULL || limit == 0u || coord >= limit) {
        return 0;
    }
    if (delta < 0) {
        size_t magnitude = (size_t)(-(delta + 1)) + 1u;
        if (coord < magnitude) {
            return 0;
        }
        *out = coord - magnitude;
        return 1;
    }
    size_t magnitude = (size_t)delta;
    if (magnitude > (limit - 1u) - coord) {
        return 0;
    }
    *out = coord + magnitude;
    return 1;
}

static void layer_proposal_init(NNLayerProposal *proposal)
{
    if (proposal == NULL) {
        return;
    }
    memset(proposal, 0, sizeof(*proposal));
    snprintf(proposal->schema, sizeof(proposal->schema), "%s.layer", NN_LAYER_DISCOVERY_SCHEMA);
    proposal->kind = NN_LAYER_KIND_UNKNOWN;
    nn_certificate_clear(&proposal->certificate);
}

static void layer_proposal_free(NNLayerProposal *proposal)
{
    if (proposal == NULL) {
        return;
    }
    nn_bitgrid_free(&proposal->mask);
    nn_bitgrid_free(&proposal->values);
    layer_proposal_init(proposal);
}

static void layer_ranker_report_init(NNLayerRankerReport *report)
{
    if (report == NULL) {
        return;
    }
    memset(report, 0, sizeof(*report));
    snprintf(report->schema, sizeof(report->schema), "%s", NN_LAYER_RANKER_SCHEMA);
    report->final_loss = (nn_real)NAN;
}

static void layer_task_policy_report_init(NNLayerTaskPolicyReport *report)
{
    if (report == NULL) {
        return;
    }
    memset(report, 0, sizeof(*report));
    snprintf(report->schema, sizeof(report->schema), "%s", NN_LAYER_TASK_POLICY_SCHEMA);
    report->final_loss = (nn_real)NAN;
}

const char *nn_layer_kind_name(NNLayerKind kind)
{
    switch (kind) {
    case NN_LAYER_KIND_STATIC:
        return "static";
    case NN_LAYER_KIND_TRANSLATION:
        return "translation";
    case NN_LAYER_KIND_RESIDUAL:
        return "residual";
    case NN_LAYER_KIND_UNKNOWN:
    default:
        return "unknown";
    }
}

void nn_layer_discovery_config_init(NNLayerDiscoveryConfig *config)
{
    if (config == NULL) {
        return;
    }
    config->max_layers = NN_LAYER_MAX_PROPOSALS;
    config->max_translation = 4;
    config->include_static_layers = 1;
    config->allow_residual_layer = 0;
}

void nn_layer_ranker_config_init(NNLayerRankerConfig *config)
{
    if (config == NULL) {
        return;
    }
    config->hidden_width = 24u;
    config->epochs = 8u;
    config->train_scenes = 128u;
    config->eval_scenes = 64u;
    config->width = 24u;
    config->height = 24u;
    config->object_count = 5u;
    config->max_translation = 3;
    config->learning_rate = (nn_real)0.035;
    config->seed = 20240522u;
}

void nn_layer_task_policy_config_init(NNLayerTaskPolicyConfig *config)
{
    if (config == NULL) {
        return;
    }
    config->hidden_width = 24u;
    config->epochs = 8u;
    config->train_tasks = 128u;
    config->eval_tasks = 64u;
    config->width = 24u;
    config->height = 24u;
    config->object_count = 5u;
    config->max_translation = 3;
    config->learning_rate = (nn_real)0.035;
    config->seed = 20240522u;
}

void nn_layer_discovery_result_init(NNLayerDiscoveryResult *result)
{
    if (result == NULL) {
        return;
    }
    memset(result, 0, sizeof(*result));
    snprintf(result->schema, sizeof(result->schema), "%s", NN_LAYER_DISCOVERY_SCHEMA);
    nn_certificate_clear(&result->certificate);
    for (size_t i = 0u; i < NN_LAYER_MAX_PROPOSALS; ++i) {
        layer_proposal_init(&result->layers[i]);
    }
}

void nn_layer_ranker_init(NNLayerRanker *ranker)
{
    if (ranker == NULL) {
        return;
    }
    memset(ranker, 0, sizeof(*ranker));
    nn_layer_ranker_config_init(&ranker->config);
    layer_ranker_report_init(&ranker->report);
}

void nn_layer_task_policy_init(NNLayerTaskPolicy *policy)
{
    if (policy == NULL) {
        return;
    }
    memset(policy, 0, sizeof(*policy));
    nn_layer_task_policy_config_init(&policy->config);
    layer_task_policy_report_init(&policy->report);
}

void nn_layer_ranker_free(NNLayerRanker *ranker)
{
    if (ranker == NULL) {
        return;
    }
    nn_free(ranker->net);
    nn_layer_ranker_init(ranker);
}

void nn_layer_task_policy_free(NNLayerTaskPolicy *policy)
{
    if (policy == NULL) {
        return;
    }
    nn_free(policy->net);
    nn_layer_task_policy_init(policy);
}

void nn_layer_discovery_result_free(NNLayerDiscoveryResult *result)
{
    if (result == NULL) {
        return;
    }
    for (size_t i = 0u; i < NN_LAYER_MAX_PROPOSALS; ++i) {
        nn_bitgrid_free(&result->layers[i].mask);
        nn_bitgrid_free(&result->layers[i].values);
    }
    nn_bitgrid_free(&result->reconstruction);
    nn_layer_discovery_result_init(result);
}

static void layer_component_bbox(const NNBitGrid *grid,
                                 const size_t *component,
                                 size_t component_count,
                                 size_t *min_x,
                                 size_t *min_y,
                                 size_t *width,
                                 size_t *height)
{
    size_t lo_x = grid->width;
    size_t lo_y = grid->height;
    size_t hi_x = 0u;
    size_t hi_y = 0u;
    for (size_t i = 0u; i < component_count; ++i) {
        size_t index = component[i];
        size_t x = index % grid->width;
        size_t y = index / grid->width;
        if (x < lo_x) {
            lo_x = x;
        }
        if (y < lo_y) {
            lo_y = y;
        }
        if (x > hi_x) {
            hi_x = x;
        }
        if (y > hi_y) {
            hi_y = y;
        }
    }
    *min_x = component_count == 0u ? 0u : lo_x;
    *min_y = component_count == 0u ? 0u : lo_y;
    *width = component_count == 0u ? 0u : hi_x - lo_x + 1u;
    *height = component_count == 0u ? 0u : hi_y - lo_y + 1u;
}

static LayerTranslationScore layer_score_translation(const NNBitGrid *target,
                                                     const NNBitGrid *reconstruction,
                                                     const size_t *component,
                                                     size_t component_count,
                                                     int dx,
                                                     int dy)
{
    LayerTranslationScore score;
    memset(&score, 0, sizeof(score));
    score.dx = dx;
    score.dy = dy;
    for (size_t i = 0u; i < component_count; ++i) {
        size_t index = component[i];
        size_t x = index % target->width;
        size_t y = index / target->width;
        size_t tx = 0u;
        size_t ty = 0u;
        if (!layer_offset(x, dx, target->width, &tx) ||
            !layer_offset(y, dy, target->height, &ty)) {
            ++score.errors;
            continue;
        }
        size_t target_index = ty * target->width + tx;
        if ((reconstruction->cells[target_index] & 1u) != 0u) {
            ++score.errors;
            continue;
        }
        ++score.placed;
        if ((target->cells[target_index] & 1u) != 0u) {
            ++score.matched;
        } else {
            ++score.errors;
        }
    }
    return score;
}

static int layer_score_is_better(LayerTranslationScore candidate,
                                 LayerTranslationScore best,
                                 int have_best)
{
    if (!have_best) {
        return 1;
    }
    if (candidate.errors != best.errors) {
        return candidate.errors < best.errors;
    }
    if (candidate.matched != best.matched) {
        return candidate.matched > best.matched;
    }
    int candidate_motion = abs(candidate.dx) + abs(candidate.dy);
    int best_motion = abs(best.dx) + abs(best.dy);
    if (candidate_motion != best_motion) {
        return candidate_motion < best_motion;
    }
    if (candidate.dy != best.dy) {
        return candidate.dy < best.dy;
    }
    return candidate.dx < best.dx;
}

static LayerTranslationScore layer_find_translation(const NNBitGrid *target,
                                                   const NNBitGrid *reconstruction,
                                                   const size_t *component,
                                                   size_t component_count,
                                                   const NNLayerDiscoveryConfig *config)
{
    LayerTranslationScore best;
    memset(&best, 0, sizeof(best));
    int have_best = 0;
    for (int dy = -config->max_translation; dy <= config->max_translation; ++dy) {
        for (int dx = -config->max_translation; dx <= config->max_translation; ++dx) {
            if (!config->include_static_layers && dx == 0 && dy == 0) {
                continue;
            }
            LayerTranslationScore candidate =
                layer_score_translation(target, reconstruction, component, component_count, dx, dy);
            if (layer_score_is_better(candidate, best, have_best)) {
                best = candidate;
                have_best = 1;
            }
        }
    }
    return best;
}

static uint32_t layer_candidate_action_id(int dx, int dy, int max_translation)
{
    int span = max_translation * 2 + 1;
    return (uint32_t)((dy + max_translation) * span + (dx + max_translation));
}

static void layer_encode_candidate_features(const NNBitGrid *source,
                                            const NNBitGrid *target,
                                            const NNBitGrid *reconstruction,
                                            const size_t *component,
                                            size_t component_count,
                                            LayerTranslationScore score,
                                            const NNLayerDiscoveryConfig *config,
                                            nn_real *features)
{
    size_t min_x = 0u;
    size_t min_y = 0u;
    size_t bbox_width = 0u;
    size_t bbox_height = 0u;
    layer_component_bbox(source, component, component_count, &min_x, &min_y, &bbox_width, &bbox_height);
    double max_translation = config->max_translation <= 0 ? 1.0 : (double)config->max_translation;
    double cells = (double)layer_cell_count(source);
    double component_scale = component_count == 0u ? 1.0 : (double)component_count;
    double center_x = (double)min_x + ((double)bbox_width * 0.5);
    double center_y = (double)min_y + ((double)bbox_height * 0.5);
    size_t target_active = layer_active_count(target);
    size_t reconstruction_active = layer_active_count(reconstruction);

    features[0] = (nn_real)1;
    features[1] = (nn_real)((double)score.dx / max_translation);
    features[2] = (nn_real)((double)score.dy / max_translation);
    features[3] = (nn_real)((double)abs(score.dx) / max_translation);
    features[4] = (nn_real)((double)abs(score.dy) / max_translation);
    features[5] = (nn_real)((double)component_count / cells);
    features[6] = (nn_real)((double)bbox_width / (double)source->width);
    features[7] = (nn_real)((double)bbox_height / (double)source->height);
    features[8] = (nn_real)(center_x / (double)source->width);
    features[9] = (nn_real)(center_y / (double)source->height);
    features[10] = (nn_real)((double)score.matched / component_scale);
    features[11] = (nn_real)((double)score.errors / component_scale);
    features[12] = (nn_real)((double)score.placed / component_scale);
    features[13] = (nn_real)((double)target_active / cells);
    features[14] = (nn_real)((double)reconstruction_active / cells);
    features[15] = (nn_real)((double)(abs(score.dx) + abs(score.dy)) / (2.0 * max_translation));
}

static size_t layer_enumerate_translation_candidates(const NNBitGrid *source,
                                                     const NNBitGrid *target,
                                                     const NNBitGrid *reconstruction,
                                                     const size_t *component,
                                                     size_t component_count,
                                                     const NNLayerDiscoveryConfig *config,
                                                     LayerCandidate *candidates,
                                                     nn_real *features,
                                                     size_t capacity)
{
    size_t count = 0u;
    for (int dy = -config->max_translation; dy <= config->max_translation; ++dy) {
        for (int dx = -config->max_translation; dx <= config->max_translation; ++dx) {
            if (!config->include_static_layers && dx == 0 && dy == 0) {
                continue;
            }
            if (count >= capacity) {
                return 0u;
            }
            LayerTranslationScore score =
                layer_score_translation(target, reconstruction, component, component_count, dx, dy);
            candidates[count].score = score;
            candidates[count].action_id =
                layer_candidate_action_id(dx, dy, config->max_translation);
            layer_encode_candidate_features(source,
                                            target,
                                            reconstruction,
                                            component,
                                            component_count,
                                            score,
                                            config,
                                            &features[count * NN_LAYER_RANKER_FEATURE_COUNT]);
            ++count;
        }
    }
    return count;
}

static size_t layer_oracle_candidate_index(const LayerCandidate *candidates, size_t count)
{
    size_t best_index = 0u;
    int have_best = 0;
    LayerTranslationScore best;
    memset(&best, 0, sizeof(best));
    for (size_t i = 0u; i < count; ++i) {
        if (layer_score_is_better(candidates[i].score, best, have_best)) {
            best = candidates[i].score;
            best_index = i;
            have_best = 1;
        }
    }
    return best_index;
}

static int layer_candidate_is_certifiable(LayerTranslationScore score, size_t component_count)
{
    return score.errors == 0u && score.matched == component_count;
}

static int layer_select_translation_ranked(const NNBitGrid *source,
                                           const NNBitGrid *target,
                                           const NNBitGrid *reconstruction,
                                           const size_t *component,
                                           size_t component_count,
                                           const NNLayerDiscoveryConfig *config,
                                           const NNLayerRanker *ranker,
                                           LayerTranslationScore *out_score,
                                           size_t *out_candidate_count,
                                           size_t *out_selected_rank,
                                           size_t *out_verifier_checks)
{
    if (out_score == NULL || out_candidate_count == NULL || out_selected_rank == NULL ||
        out_verifier_checks == NULL) {
        return 0;
    }
    *out_candidate_count = 0u;
    *out_selected_rank = 0u;
    *out_verifier_checks = 0u;

    if (ranker == NULL) {
        *out_score = layer_find_translation(target, reconstruction, component, component_count, config);
        return 1;
    }
    if (ranker->net == NULL) {
        return 0;
    }

    LayerCandidate candidates[NN_LAYER_RANKER_MAX_CANDIDATES];
    nn_real features[NN_LAYER_RANKER_MAX_CANDIDATES * NN_LAYER_RANKER_FEATURE_COUNT];
    nn_real scores[NN_LAYER_RANKER_MAX_CANDIDATES];
    nn_real probabilities[NN_LAYER_RANKER_MAX_CANDIDATES];
    uint32_t action_ids[NN_LAYER_RANKER_MAX_CANDIDATES];
    size_t ranked[NN_LAYER_RANKER_MAX_CANDIDATES];
    size_t candidate_count = layer_enumerate_translation_candidates(source,
                                                                    target,
                                                                    reconstruction,
                                                                    component,
                                                                    component_count,
                                                                    config,
                                                                    candidates,
                                                                    features,
                                                                    NN_LAYER_RANKER_MAX_CANDIDATES);
    if (candidate_count == 0u) {
        return 0;
    }
    for (size_t i = 0u; i < candidate_count; ++i) {
        action_ids[i] = candidates[i].action_id;
    }
    if (!nn_score_action_rows(ranker->net, features, candidate_count, scores, probabilities) ||
        !nn_rank_action_scores_with_ids(scores,
                                        action_ids,
                                        candidate_count,
                                        ranked,
                                        NN_RANK_SCORE_EPSILON)) {
        return 0;
    }

    *out_candidate_count = candidate_count;
    for (size_t rank = 0u; rank < candidate_count; ++rank) {
        size_t index = ranked[rank];
        ++(*out_verifier_checks);
        if (layer_candidate_is_certifiable(candidates[index].score, component_count)) {
            *out_score = candidates[index].score;
            *out_selected_rank = rank;
            return 1;
        }
    }

    size_t fallback = layer_oracle_candidate_index(candidates, candidate_count);
    *out_score = candidates[fallback].score;
    *out_selected_rank = candidate_count;
    return 1;
}

static int layer_apply_component(NNBitGrid *grid,
                                 const size_t *component,
                                 size_t component_count,
                                 int dx,
                                 int dy)
{
    if (grid == NULL || grid->cells == NULL) {
        return 0;
    }
    for (size_t i = 0u; i < component_count; ++i) {
        size_t index = component[i];
        size_t x = index % grid->width;
        size_t y = index / grid->width;
        size_t tx = 0u;
        size_t ty = 0u;
        if (!layer_offset(x, dx, grid->width, &tx) || !layer_offset(y, dy, grid->height, &ty)) {
            return 0;
        }
        grid->cells[ty * grid->width + tx] = 1u;
    }
    return 1;
}

int nn_layer_apply_global_translation(const NNBitGrid *source,
                                      int dx,
                                      int dy,
                                      NNBitGrid *out)
{
    if (source == NULL || source->cells == NULL || out == NULL) {
        return 0;
    }
    if (out->cells == NULL) {
        if (!nn_bitgrid_init(out, source->width, source->height)) {
            return 0;
        }
    } else if (!nn_bitgrid_same_shape(source, out)) {
        return 0;
    }
    if (!nn_bitgrid_fill(out, 0u)) {
        return 0;
    }
    size_t cells = layer_cell_count(source);
    for (size_t i = 0u; i < cells; ++i) {
        if ((source->cells[i] & 1u) == 0u) {
            continue;
        }
        size_t x = i % source->width;
        size_t y = i / source->width;
        size_t tx = 0u;
        size_t ty = 0u;
        if (!layer_offset(x, dx, source->width, &tx) ||
            !layer_offset(y, dy, source->height, &ty)) {
            return 0;
        }
        out->cells[ty * out->width + tx] = 1u;
    }
    return 1;
}

static LayerTranslationScore layer_score_global_translation(const NNBitGrid *source,
                                                           const NNBitGrid *target,
                                                           int dx,
                                                           int dy)
{
    LayerTranslationScore score;
    NNBitGrid translated = {0};
    memset(&score, 0, sizeof(score));
    score.dx = dx;
    score.dy = dy;
    if (!nn_bitgrid_same_shape(source, target) ||
        !nn_bitgrid_init(&translated, source->width, source->height)) {
        score.errors = source == NULL ? 1u : layer_cell_count(source);
        return score;
    }

    size_t source_cells = layer_cell_count(source);
    for (size_t i = 0u; i < source_cells; ++i) {
        if ((source->cells[i] & 1u) == 0u) {
            continue;
        }
        size_t x = i % source->width;
        size_t y = i / source->width;
        size_t tx = 0u;
        size_t ty = 0u;
        if (!layer_offset(x, dx, source->width, &tx) ||
            !layer_offset(y, dy, source->height, &ty)) {
            ++score.errors;
            continue;
        }
        translated.cells[ty * translated.width + tx] = 1u;
        ++score.placed;
    }
    size_t target_cells = layer_cell_count(target);
    for (size_t i = 0u; i < target_cells; ++i) {
        uint8_t actual = translated.cells[i] & 1u;
        uint8_t expected = target->cells[i] & 1u;
        if (actual != expected) {
            ++score.errors;
        }
        if (actual != 0u && expected != 0u) {
            ++score.matched;
        }
    }
    nn_bitgrid_free(&translated);
    return score;
}

static void layer_encode_task_candidate_features(const NNBitGrid *demo_source,
                                                 const NNBitGrid *demo_target,
                                                 const NNBitGrid *query_source,
                                                 LayerTranslationScore score,
                                                 int max_translation,
                                                 nn_real *features)
{
    double max_delta = max_translation <= 0 ? 1.0 : (double)max_translation;
    double cells = (double)layer_cell_count(demo_source);
    double source_active = (double)layer_active_count(demo_source);
    double target_active = (double)layer_active_count(demo_target);
    double query_active = (double)layer_active_count(query_source);
    double source_scale = source_active <= 0.0 ? 1.0 : source_active;

    features[0] = (nn_real)1;
    features[1] = (nn_real)((double)score.dx / max_delta);
    features[2] = (nn_real)((double)score.dy / max_delta);
    features[3] = (nn_real)((double)abs(score.dx) / max_delta);
    features[4] = (nn_real)((double)abs(score.dy) / max_delta);
    features[5] = (nn_real)(source_active / cells);
    features[6] = (nn_real)(target_active / cells);
    features[7] = (nn_real)(query_active / cells);
    features[8] = (nn_real)((double)score.matched / source_scale);
    features[9] = (nn_real)((double)score.errors / cells);
    features[10] = (nn_real)((double)score.placed / source_scale);
    features[11] = (nn_real)(score.errors == 0u ? 1 : 0);
    features[12] = (nn_real)((double)(abs(score.dx) + abs(score.dy)) / (2.0 * max_delta));
    features[13] = (nn_real)((source_active - target_active) / cells);
    features[14] = (nn_real)((query_active - source_active) / cells);
    features[15] = (nn_real)((score.dx == 0 && score.dy == 0) ? 1 : 0);
}

static size_t layer_enumerate_task_candidates(const NNBitGrid *demo_source,
                                              const NNBitGrid *demo_target,
                                              const NNBitGrid *query_source,
                                              int max_translation,
                                              LayerCandidate *candidates,
                                              nn_real *features,
                                              size_t capacity)
{
    size_t count = 0u;
    for (int dy = -max_translation; dy <= max_translation; ++dy) {
        for (int dx = -max_translation; dx <= max_translation; ++dx) {
            if (count >= capacity) {
                return 0u;
            }
            LayerTranslationScore score =
                layer_score_global_translation(demo_source, demo_target, dx, dy);
            candidates[count].score = score;
            candidates[count].action_id = layer_candidate_action_id(dx, dy, max_translation);
            layer_encode_task_candidate_features(demo_source,
                                                 demo_target,
                                                 query_source,
                                                 score,
                                                 max_translation,
                                                 &features[count * NN_LAYER_TASK_FEATURE_COUNT]);
            ++count;
        }
    }
    return count;
}

static size_t layer_task_oracle_candidate_index(const LayerCandidate *candidates, size_t count)
{
    size_t best_index = 0u;
    int have_best = 0;
    LayerTranslationScore best;
    memset(&best, 0, sizeof(best));
    for (size_t i = 0u; i < count; ++i) {
        if (layer_score_is_better(candidates[i].score, best, have_best)) {
            best = candidates[i].score;
            best_index = i;
            have_best = 1;
        }
    }
    return best_index;
}

static double layer_description_length(size_t source_cells,
                                       size_t bbox_width,
                                       size_t bbox_height,
                                       int dx,
                                       int dy)
{
    return 8.0 + (double)source_cells + 0.125 * (double)(bbox_width * bbox_height) +
           (double)(abs(dx) + abs(dy));
}

static int layer_fill_proposal(NNLayerProposal *proposal,
                               const NNBitGrid *source,
                               const size_t *component,
                               size_t component_count,
                               LayerTranslationScore score)
{
    if (proposal == NULL || source == NULL || source->cells == NULL) {
        return 0;
    }
    layer_proposal_free(proposal);
    if (!nn_bitgrid_init(&proposal->mask, source->width, source->height) ||
        !nn_bitgrid_init(&proposal->values, source->width, source->height)) {
        layer_proposal_free(proposal);
        return 0;
    }
    snprintf(proposal->schema, sizeof(proposal->schema), "%s.layer", NN_LAYER_DISCOVERY_SCHEMA);
    proposal->kind = score.dx == 0 && score.dy == 0 ? NN_LAYER_KIND_STATIC :
                                                    NN_LAYER_KIND_TRANSLATION;
    proposal->dx = score.dx;
    proposal->dy = score.dy;
    proposal->source_cells = component_count;
    proposal->target_cells = score.matched;
    proposal->reconstructed_cells = score.placed;
    proposal->reconstruction_errors = score.errors;
    layer_component_bbox(source,
                         component,
                         component_count,
                         &proposal->min_x,
                         &proposal->min_y,
                         &proposal->width,
                         &proposal->height);
    proposal->description_length =
        layer_description_length(component_count, proposal->width, proposal->height, score.dx, score.dy);
    for (size_t i = 0u; i < component_count; ++i) {
        size_t index = component[i];
        proposal->mask.cells[index] = 1u;
        proposal->values.cells[index] = source->cells[index] & 1u;
    }
    if (score.errors == 0u && score.matched == component_count) {
        nn_certificate_accept(&proposal->certificate,
                              "motion_layer_translation",
                              "component translation reconstructs target support",
                              component_count,
                              component_count);
    } else {
        nn_certificate_reject(&proposal->certificate,
                              "motion_layer_translation",
                              "component translation leaves reconstruction errors",
                              component_count,
                              score.errors);
    }
    return 1;
}

static size_t layer_collect_component(const NNBitGrid *source,
                                      size_t start,
                                      uint8_t *visited,
                                      size_t *queue,
                                      size_t *component)
{
    size_t head = 0u;
    size_t tail = 0u;
    size_t count = 0u;
    queue[tail++] = start;
    visited[start] = 1u;
    while (head < tail) {
        size_t index = queue[head++];
        component[count++] = index;
        size_t x = index % source->width;
        size_t y = index / source->width;
        static const int dxs[4] = {-1, 1, 0, 0};
        static const int dys[4] = {0, 0, -1, 1};
        for (size_t i = 0u; i < 4u; ++i) {
            size_t nx = 0u;
            size_t ny = 0u;
            if (!layer_offset(x, dxs[i], source->width, &nx) ||
                !layer_offset(y, dys[i], source->height, &ny)) {
                continue;
            }
            size_t next = ny * source->width + nx;
            if (!visited[next] && (source->cells[next] & 1u) != 0u) {
                visited[next] = 1u;
                queue[tail++] = next;
            }
        }
    }
    return count;
}

static int layer_add_residual_if_possible(const NNBitGrid *target, NNLayerDiscoveryResult *result)
{
    if (target == NULL || result == NULL || result->layer_count >= NN_LAYER_MAX_PROPOSALS) {
        return 0;
    }
    NNLayerProposal *proposal = &result->layers[result->layer_count];
    layer_proposal_free(proposal);
    if (!nn_bitgrid_init(&proposal->mask, target->width, target->height) ||
        !nn_bitgrid_init(&proposal->values, target->width, target->height)) {
        layer_proposal_free(proposal);
        return 0;
    }
    snprintf(proposal->schema, sizeof(proposal->schema), "%s.layer", NN_LAYER_DISCOVERY_SCHEMA);
    proposal->kind = NN_LAYER_KIND_RESIDUAL;
    proposal->width = target->width;
    proposal->height = target->height;
    size_t missing = 0u;
    size_t cells = layer_cell_count(target);
    for (size_t i = 0u; i < cells; ++i) {
        if ((target->cells[i] & 1u) != 0u && (result->reconstruction.cells[i] & 1u) == 0u) {
            proposal->mask.cells[i] = 1u;
            proposal->values.cells[i] = 1u;
            result->reconstruction.cells[i] = 1u;
            ++missing;
        }
    }
    proposal->source_cells = missing;
    proposal->target_cells = missing;
    proposal->reconstructed_cells = missing;
    proposal->candidate_count = 1u;
    proposal->selected_rank = 0u;
    proposal->verifier_checks = 1u;
    proposal->description_length = 16.0 + (double)missing;
    nn_certificate_accept(&proposal->certificate,
                          "residual_target_layer",
                          "residual paints unexplained target support",
                          missing,
                          missing);
    ++result->layer_count;
    ++result->certified_layers;
    ++result->top1_layers;
    ++result->verifier_checks;
    result->description_length += proposal->description_length;
    return 1;
}

static int layer_ranker_config_is_valid(const NNLayerRankerConfig *config)
{
    if (config == NULL || config->hidden_width == 0u || config->epochs == 0u ||
        config->train_scenes == 0u || config->eval_scenes == 0u ||
        config->width < 8u || config->height < 8u || config->object_count == 0u ||
        config->object_count > NN_LAYER_MAX_PROPOSALS ||
        config->max_translation < 1 || config->max_translation > 4 ||
        config->learning_rate <= (nn_real)0 || !isfinite((double)config->learning_rate)) {
        return 0;
    }
    int span = config->max_translation * 2 + 1;
    return (size_t)(span * span) <= NN_LAYER_RANKER_MAX_CANDIDATES;
}

static int layer_ranker_train_scene(NeuralNet *net,
                                    const NNBitGrid *source,
                                    const NNBitGrid *target,
                                    const NNLayerDiscoveryConfig *discovery_config,
                                    nn_real learning_rate,
                                    nn_real *loss_sum,
                                    size_t *groups,
                                    size_t *updates)
{
    NNBitGrid reconstruction = {0};
    if (net == NULL || loss_sum == NULL || groups == NULL || updates == NULL ||
        !nn_bitgrid_same_shape(source, target) ||
        !nn_bitgrid_init(&reconstruction, source->width, source->height)) {
        return 0;
    }

    size_t cells = layer_cell_count(source);
    uint8_t *visited = (uint8_t *)calloc(cells, sizeof(*visited));
    size_t *queue = (size_t *)calloc(cells, sizeof(*queue));
    size_t *component = (size_t *)calloc(cells, sizeof(*component));
    LayerCandidate candidates[NN_LAYER_RANKER_MAX_CANDIDATES];
    nn_real features[NN_LAYER_RANKER_MAX_CANDIDATES * NN_LAYER_RANKER_FEATURE_COUNT];
    int ok = visited != NULL && queue != NULL && component != NULL;

    for (size_t index = 0u; ok && index < cells; ++index) {
        if (visited[index] || (source->cells[index] & 1u) == 0u) {
            continue;
        }
        size_t component_count = layer_collect_component(source, index, visited, queue, component);
        size_t candidate_count = layer_enumerate_translation_candidates(source,
                                                                        target,
                                                                        &reconstruction,
                                                                        component,
                                                                        component_count,
                                                                        discovery_config,
                                                                        candidates,
                                                                        features,
                                                                        NN_LAYER_RANKER_MAX_CANDIDATES);
        if (candidate_count == 0u) {
            ok = 0;
            break;
        }
        size_t target_index = layer_oracle_candidate_index(candidates, candidate_count);
        nn_real loss = nn_train_softmax_scorer(net,
                                               features,
                                               candidate_count,
                                               target_index,
                                               learning_rate);
        if (!isfinite((double)loss)) {
            ok = 0;
            break;
        }
        *loss_sum += loss;
        ++(*groups);
        ++(*updates);
        ok = layer_apply_component(&reconstruction,
                                   component,
                                   component_count,
                                   candidates[target_index].score.dx,
                                   candidates[target_index].score.dy);
    }

    free(visited);
    free(queue);
    free(component);
    nn_bitgrid_free(&reconstruction);
    return ok;
}

int nn_layer_ranker_train(NNLayerRanker *ranker, const NNLayerRankerConfig *config)
{
    if (ranker == NULL) {
        return 0;
    }

    NNLayerRankerConfig default_config;
    if (config == NULL) {
        nn_layer_ranker_config_init(&default_config);
        config = &default_config;
    }
    if (!layer_ranker_config_is_valid(config)) {
        return 0;
    }

    nn_layer_ranker_free(ranker);
    ranker->config = *config;
    layer_ranker_report_init(&ranker->report);
    ranker->report.train_scenes = config->train_scenes;
    ranker->report.eval_scenes = config->eval_scenes;

    size_t layers[] = {NN_LAYER_RANKER_FEATURE_COUNT, config->hidden_width, 1u};
    NNConfig nn_config = {
        .layer_sizes = layers,
        .layer_count = sizeof(layers) / sizeof(layers[0]),
        .seed = config->seed,
        .hidden_activation = NN_ACTIVATION_TANH,
        .output_activation = NN_ACTIVATION_LINEAR,
        .loss = NN_LOSS_MSE,
    };
    ranker->net = nn_create_config(&nn_config);
    if (ranker->net == NULL) {
        return 0;
    }
    ranker->report.parameters = nn_parameter_count(ranker->net);
    ranker->report.forward_flops = nn_forward_flop_count(ranker->net);

    NNLayerDiscoveryConfig discovery_config;
    nn_layer_discovery_config_init(&discovery_config);
    discovery_config.max_translation = config->max_translation;
    discovery_config.max_layers = NN_LAYER_MAX_PROPOSALS;
    discovery_config.include_static_layers = 1;
    discovery_config.allow_residual_layer = 0;

    nn_real last_epoch_loss = (nn_real)NAN;
    for (size_t epoch = 0u; epoch < config->epochs; ++epoch) {
        nn_real loss_sum = (nn_real)0;
        size_t groups = 0u;
        for (size_t scene = 0u; scene < config->train_scenes; ++scene) {
            NNBitGrid source = {0};
            NNBitGrid target = {0};
            unsigned int seed = config->seed + (unsigned int)(epoch * 1000003u + scene * 17u);
            int ok = nn_layer_generate_motion_scene(&source,
                                                    &target,
                                                    config->width,
                                                    config->height,
                                                    config->object_count,
                                                    config->max_translation,
                                                    seed) &&
                     layer_ranker_train_scene(ranker->net,
                                              &source,
                                              &target,
                                              &discovery_config,
                                              config->learning_rate,
                                              &loss_sum,
                                              &groups,
                                              &ranker->report.training_updates);
            nn_bitgrid_free(&source);
            nn_bitgrid_free(&target);
            if (!ok) {
                return 0;
            }
        }
        ranker->report.training_groups += groups;
        last_epoch_loss = groups == 0u ? (nn_real)NAN : loss_sum / (nn_real)groups;
    }
    ranker->report.final_loss = last_epoch_loss;

    size_t total_layers = 0u;
    size_t top1_layers = 0u;
    size_t certified_scenes = 0u;
    size_t verifier_checks = 0u;
    for (size_t scene = 0u; scene < config->eval_scenes; ++scene) {
        NNBitGrid source = {0};
        NNBitGrid target = {0};
        NNLayerDiscoveryResult result;
        nn_layer_discovery_result_init(&result);
        unsigned int seed =
            config->seed + 900000001u + (unsigned int)(scene * 7919u);
        int ok = nn_layer_generate_motion_scene(&source,
                                                &target,
                                                config->width,
                                                config->height,
                                                config->object_count,
                                                config->max_translation,
                                                seed) &&
                 nn_layer_discover_motion_ranked(&source,
                                                 &target,
                                                 &discovery_config,
                                                 ranker,
                                                 &result);
        if (ok && result.certificate.status == NN_CERTIFICATE_CERTIFIED &&
            result.reconstruction_errors == 0u) {
            ++certified_scenes;
        }
        if (ok) {
            total_layers += result.layer_count;
            top1_layers += result.top1_layers;
            verifier_checks += result.verifier_checks;
        }
        nn_layer_discovery_result_free(&result);
        nn_bitgrid_free(&source);
        nn_bitgrid_free(&target);
        if (!ok) {
            return 0;
        }
    }
    ranker->report.eval_layers = total_layers;
    ranker->report.top1_layers = top1_layers;
    ranker->report.certified_scenes = certified_scenes;
    ranker->report.verifier_checks = verifier_checks;
    ranker->report.top1_accuracy =
        total_layers == 0u ? 0.0 : (double)top1_layers / (double)total_layers;
    ranker->report.certified_rate =
        config->eval_scenes == 0u ? 0.0 : (double)certified_scenes / (double)config->eval_scenes;
    ranker->report.average_verifier_checks =
        total_layers == 0u ? 0.0 : (double)verifier_checks / (double)total_layers;
    return 1;
}

static int layer_task_policy_config_is_valid(const NNLayerTaskPolicyConfig *config)
{
    if (config == NULL || config->hidden_width == 0u || config->epochs == 0u ||
        config->train_tasks == 0u || config->eval_tasks == 0u ||
        config->width < 8u || config->height < 8u || config->object_count == 0u ||
        config->object_count > NN_LAYER_MAX_PROPOSALS ||
        config->max_translation < 1 || config->max_translation > 4 ||
        config->learning_rate <= (nn_real)0 || !isfinite((double)config->learning_rate)) {
        return 0;
    }
    int span = config->max_translation * 2 + 1;
    return (size_t)(span * span) <= NN_LAYER_RANKER_MAX_CANDIDATES;
}

int nn_layer_task_policy_predict(const NNLayerTaskPolicy *policy,
                                 const NNBitGrid *demo_source,
                                 const NNBitGrid *demo_target,
                                 const NNBitGrid *query_source,
                                 NNBitGrid *out_prediction,
                                 int *out_dx,
                                 int *out_dy)
{
    if (policy == NULL || policy->net == NULL || out_prediction == NULL ||
        !nn_bitgrid_same_shape(demo_source, demo_target) ||
        query_source == NULL || query_source->cells == NULL ||
        demo_source->width != query_source->width ||
        demo_source->height != query_source->height) {
        return 0;
    }

    LayerCandidate candidates[NN_LAYER_RANKER_MAX_CANDIDATES];
    nn_real features[NN_LAYER_RANKER_MAX_CANDIDATES * NN_LAYER_TASK_FEATURE_COUNT];
    nn_real scores[NN_LAYER_RANKER_MAX_CANDIDATES];
    nn_real probabilities[NN_LAYER_RANKER_MAX_CANDIDATES];
    uint32_t action_ids[NN_LAYER_RANKER_MAX_CANDIDATES];
    size_t ranked[NN_LAYER_RANKER_MAX_CANDIDATES];
    size_t candidate_count = layer_enumerate_task_candidates(demo_source,
                                                             demo_target,
                                                             query_source,
                                                             policy->config.max_translation,
                                                             candidates,
                                                             features,
                                                             NN_LAYER_RANKER_MAX_CANDIDATES);
    if (candidate_count == 0u) {
        return 0;
    }
    for (size_t i = 0u; i < candidate_count; ++i) {
        action_ids[i] = candidates[i].action_id;
    }
    if (!nn_score_action_rows(policy->net, features, candidate_count, scores, probabilities) ||
        !nn_rank_action_scores_with_ids(scores,
                                        action_ids,
                                        candidate_count,
                                        ranked,
                                        NN_RANK_SCORE_EPSILON)) {
        return 0;
    }

    for (size_t rank = 0u; rank < candidate_count; ++rank) {
        LayerTranslationScore selected = candidates[ranked[rank]].score;
        if (nn_layer_apply_global_translation(query_source,
                                              selected.dx,
                                              selected.dy,
                                              out_prediction)) {
            if (out_dx != NULL) {
                *out_dx = selected.dx;
            }
            if (out_dy != NULL) {
                *out_dy = selected.dy;
            }
            return 1;
        }
    }
    return 0;
}

int nn_layer_task_policy_train(NNLayerTaskPolicy *policy,
                               const NNLayerTaskPolicyConfig *config)
{
    if (policy == NULL) {
        return 0;
    }
    NNLayerTaskPolicyConfig default_config;
    if (config == NULL) {
        nn_layer_task_policy_config_init(&default_config);
        config = &default_config;
    }
    if (!layer_task_policy_config_is_valid(config)) {
        return 0;
    }

    nn_layer_task_policy_free(policy);
    policy->config = *config;
    layer_task_policy_report_init(&policy->report);
    policy->report.train_tasks = config->train_tasks;
    policy->report.eval_tasks = config->eval_tasks;

    size_t layers[] = {NN_LAYER_TASK_FEATURE_COUNT, config->hidden_width, 1u};
    NNConfig nn_config = {
        .layer_sizes = layers,
        .layer_count = sizeof(layers) / sizeof(layers[0]),
        .seed = config->seed,
        .hidden_activation = NN_ACTIVATION_TANH,
        .output_activation = NN_ACTIVATION_LINEAR,
        .loss = NN_LOSS_MSE,
    };
    policy->net = nn_create_config(&nn_config);
    if (policy->net == NULL) {
        return 0;
    }
    policy->report.parameters = nn_parameter_count(policy->net);
    policy->report.forward_flops = nn_forward_flop_count(policy->net);

    LayerCandidate candidates[NN_LAYER_RANKER_MAX_CANDIDATES];
    nn_real features[NN_LAYER_RANKER_MAX_CANDIDATES * NN_LAYER_TASK_FEATURE_COUNT];
    nn_real last_epoch_loss = (nn_real)NAN;
    for (size_t epoch = 0u; epoch < config->epochs; ++epoch) {
        nn_real loss_sum = (nn_real)0;
        size_t updates = 0u;
        for (size_t task = 0u; task < config->train_tasks; ++task) {
            NNBitGrid demo_source = {0};
            NNBitGrid demo_target = {0};
            NNBitGrid query_source = {0};
            NNBitGrid query_target = {0};
            unsigned int seed = config->seed + (unsigned int)(epoch * 1000003u + task * 17u);
            int ok = nn_layer_generate_global_motion_task(&demo_source,
                                                          &demo_target,
                                                          &query_source,
                                                          &query_target,
                                                          config->width,
                                                          config->height,
                                                          config->object_count,
                                                          config->max_translation,
                                                          seed,
                                                          NULL,
                                                          NULL);
            if (ok) {
                size_t candidate_count = layer_enumerate_task_candidates(&demo_source,
                                                                         &demo_target,
                                                                         &query_source,
                                                                         config->max_translation,
                                                                         candidates,
                                                                         features,
                                                                         NN_LAYER_RANKER_MAX_CANDIDATES);
                size_t target_index = layer_task_oracle_candidate_index(candidates, candidate_count);
                nn_real loss = candidate_count == 0u
                                   ? (nn_real)NAN
                                   : nn_train_softmax_scorer(policy->net,
                                                             features,
                                                             candidate_count,
                                                             target_index,
                                                             config->learning_rate);
                ok = isfinite((double)loss);
                if (ok) {
                    loss_sum += loss;
                    ++updates;
                    ++policy->report.training_updates;
                }
            }
            nn_bitgrid_free(&demo_source);
            nn_bitgrid_free(&demo_target);
            nn_bitgrid_free(&query_source);
            nn_bitgrid_free(&query_target);
            if (!ok) {
                return 0;
            }
        }
        last_epoch_loss = updates == 0u ? (nn_real)NAN : loss_sum / (nn_real)updates;
    }
    policy->report.final_loss = last_epoch_loss;

    for (size_t task = 0u; task < config->eval_tasks; ++task) {
        NNBitGrid demo_source = {0};
        NNBitGrid demo_target = {0};
        NNBitGrid query_source = {0};
        NNBitGrid query_target = {0};
        NNBitGrid prediction = {0};
        int dx = 0;
        int dy = 0;
        int predicted_dx = 0;
        int predicted_dy = 0;
        unsigned int seed = config->seed + 900000001u + (unsigned int)(task * 7919u);
        int ok = nn_layer_generate_global_motion_task(&demo_source,
                                                      &demo_target,
                                                      &query_source,
                                                      &query_target,
                                                      config->width,
                                                      config->height,
                                                      config->object_count,
                                                      config->max_translation,
                                                      seed,
                                                      &dx,
                                                      &dy) &&
                 nn_layer_task_policy_predict(policy,
                                              &demo_source,
                                              &demo_target,
                                              &query_source,
                                              &prediction,
                                              &predicted_dx,
                                              &predicted_dy);
        if (ok) {
            ++policy->report.verifier_checks;
            if (predicted_dx == dx && predicted_dy == dy) {
                ++policy->report.top1_tasks;
            }
            if (nn_bitgrid_hamming_distance(&prediction, &query_target) == 0u) {
                ++policy->report.solved_tasks;
            }
        }
        nn_bitgrid_free(&demo_source);
        nn_bitgrid_free(&demo_target);
        nn_bitgrid_free(&query_source);
        nn_bitgrid_free(&query_target);
        nn_bitgrid_free(&prediction);
        if (!ok) {
            return 0;
        }
    }
    policy->report.exact_rate =
        config->eval_tasks == 0u ? 0.0 : (double)policy->report.solved_tasks / (double)config->eval_tasks;
    policy->report.top1_accuracy =
        config->eval_tasks == 0u ? 0.0 : (double)policy->report.top1_tasks / (double)config->eval_tasks;
    return 1;
}

static int layer_discover_motion_internal(const NNBitGrid *source,
                                          const NNBitGrid *target,
                                          const NNLayerDiscoveryConfig *config,
                                          const NNLayerRanker *ranker,
                                          NNLayerDiscoveryResult *out_result)
{
    if (!nn_bitgrid_same_shape(source, target) || out_result == NULL) {
        return 0;
    }
    NNLayerDiscoveryConfig default_config;
    if (config == NULL) {
        nn_layer_discovery_config_init(&default_config);
        config = &default_config;
    }
    if (config->max_layers == 0u || config->max_layers > NN_LAYER_MAX_PROPOSALS ||
        config->max_translation < 0) {
        return 0;
    }

    nn_layer_discovery_result_free(out_result);
    snprintf(out_result->schema, sizeof(out_result->schema), "%s", NN_LAYER_DISCOVERY_SCHEMA);
    if (!nn_bitgrid_init(&out_result->reconstruction, source->width, source->height)) {
        return 0;
    }
    out_result->source_cells = layer_active_count(source);
    out_result->target_cells = layer_active_count(target);

    size_t cells = layer_cell_count(source);
    uint8_t *visited = (uint8_t *)calloc(cells, sizeof(*visited));
    size_t *queue = (size_t *)calloc(cells, sizeof(*queue));
    size_t *component = (size_t *)calloc(cells, sizeof(*component));
    if (visited == NULL || queue == NULL || component == NULL) {
        free(visited);
        free(queue);
        free(component);
        return 0;
    }

    int ok = 1;
    for (size_t index = 0u; ok && index < cells; ++index) {
        if (visited[index] || (source->cells[index] & 1u) == 0u) {
            continue;
        }
        if (out_result->layer_count >= config->max_layers) {
            ok = 0;
            break;
        }
        size_t component_count = layer_collect_component(source, index, visited, queue, component);
        LayerTranslationScore score;
        size_t candidate_count = 0u;
        size_t selected_rank = 0u;
        size_t verifier_checks = 0u;
        ok = layer_select_translation_ranked(source,
                                             target,
                                             &out_result->reconstruction,
                                             component,
                                             component_count,
                                             config,
                                             ranker,
                                             &score,
                                             &candidate_count,
                                             &selected_rank,
                                             &verifier_checks);
        if (!ok) {
            break;
        }
        NNLayerProposal *proposal = &out_result->layers[out_result->layer_count];
        ok = layer_fill_proposal(proposal, source, component, component_count, score) &&
             layer_apply_component(&out_result->reconstruction, component, component_count, score.dx, score.dy);
        if (!ok) {
            break;
        }
        out_result->description_length += proposal->description_length;
        out_result->certified_layers +=
            proposal->certificate.status == NN_CERTIFICATE_CERTIFIED ? 1u : 0u;
        proposal->candidate_count = candidate_count;
        proposal->selected_rank = selected_rank;
        proposal->verifier_checks = verifier_checks;
        out_result->verifier_checks += verifier_checks;
        out_result->top1_layers += selected_rank == 0u ? 1u : 0u;
        ++out_result->layer_count;
    }

    free(visited);
    free(queue);
    free(component);
    if (!ok) {
        nn_certificate_reject(&out_result->certificate,
                              "motion_layer_reconstruction",
                              "layer search failed before full reconstruction",
                              out_result->source_cells,
                              out_result->source_cells);
        return 0;
    }

    if (config->allow_residual_layer &&
        nn_bitgrid_hamming_distance(&out_result->reconstruction, target) != 0u) {
        (void)layer_add_residual_if_possible(target, out_result);
    }

    out_result->reconstruction_errors =
        nn_bitgrid_hamming_distance(&out_result->reconstruction, target);
    out_result->reconstructed_cells = layer_active_count(&out_result->reconstruction);
    if (out_result->reconstruction_errors == 0u) {
        nn_certificate_accept(&out_result->certificate,
                              "motion_layer_reconstruction",
                              "composed motion layers exactly reconstruct target grid",
                              cells,
                              cells);
        return 1;
    }
    nn_certificate_reject(&out_result->certificate,
                          "motion_layer_reconstruction",
                          "composed motion layers do not reconstruct target grid",
                          cells,
                          out_result->reconstruction_errors);
    return 1;
}

int nn_layer_discover_motion(const NNBitGrid *source,
                             const NNBitGrid *target,
                             const NNLayerDiscoveryConfig *config,
                             NNLayerDiscoveryResult *out_result)
{
    return layer_discover_motion_internal(source, target, config, NULL, out_result);
}

int nn_layer_discover_motion_ranked(const NNBitGrid *source,
                                    const NNBitGrid *target,
                                    const NNLayerDiscoveryConfig *config,
                                    const NNLayerRanker *ranker,
                                    NNLayerDiscoveryResult *out_result)
{
    if (ranker == NULL || ranker->net == NULL) {
        return 0;
    }
    return layer_discover_motion_internal(source, target, config, ranker, out_result);
}

int nn_layer_recompose(const NNLayerDiscoveryResult *result, NNBitGrid *out)
{
    if (result == NULL || out == NULL || result->reconstruction.cells == NULL) {
        return 0;
    }
    return nn_bitgrid_copy(out, &result->reconstruction);
}

int nn_layer_discovery_write_json(FILE *fp, const NNLayerDiscoveryResult *result)
{
    if (fp == NULL || result == NULL) {
        return 0;
    }
    fprintf(fp,
            "{\"schema\":\"%s.summary\","
            "\"layers\":%zu,"
            "\"certified_layers\":%zu,"
            "\"source_cells\":%zu,"
            "\"target_cells\":%zu,"
            "\"reconstructed_cells\":%zu,"
            "\"reconstruction_errors\":%zu,"
            "\"top1_layers\":%zu,"
            "\"verifier_checks\":%zu,"
            "\"description_length\":%.6f,"
            "\"certificate\":\"%s\"}\n",
            NN_LAYER_DISCOVERY_SCHEMA,
            result->layer_count,
            result->certified_layers,
            result->source_cells,
            result->target_cells,
            result->reconstructed_cells,
            result->reconstruction_errors,
            result->top1_layers,
            result->verifier_checks,
            result->description_length,
            nn_certificate_status_name(result->certificate.status));
    for (size_t i = 0u; i < result->layer_count; ++i) {
        const NNLayerProposal *layer = &result->layers[i];
        fprintf(fp,
                "{\"schema\":\"%s.layer\","
                "\"index\":%zu,"
                "\"kind\":\"%s\","
                "\"dx\":%d,"
                "\"dy\":%d,"
                "\"bbox\":[%zu,%zu,%zu,%zu],"
                "\"source_cells\":%zu,"
                "\"reconstruction_errors\":%zu,"
                "\"candidate_count\":%zu,"
                "\"selected_rank\":%zu,"
                "\"verifier_checks\":%zu,"
                "\"description_length\":%.6f,"
                "\"certificate\":\"%s\"}\n",
                NN_LAYER_DISCOVERY_SCHEMA,
                i,
                nn_layer_kind_name(layer->kind),
                layer->dx,
                layer->dy,
                layer->min_x,
                layer->min_y,
                layer->width,
                layer->height,
                layer->source_cells,
                layer->reconstruction_errors,
                layer->candidate_count,
                layer->selected_rank,
                layer->verifier_checks,
                layer->description_length,
                nn_certificate_status_name(layer->certificate.status));
    }
    return ferror(fp) == 0;
}

int nn_layer_ranker_write_json(FILE *fp, const NNLayerRanker *ranker)
{
    if (fp == NULL || ranker == NULL) {
        return 0;
    }
    const NNLayerRankerReport *report = &ranker->report;
    fprintf(fp,
            "{\"schema\":\"%s.summary\","
            "\"train_scenes\":%zu,"
            "\"eval_scenes\":%zu,"
            "\"training_groups\":%zu,"
            "\"training_updates\":%zu,"
            "\"eval_layers\":%zu,"
            "\"top1_layers\":%zu,"
            "\"certified_scenes\":%zu,"
            "\"verifier_checks\":%zu,"
            "\"top1_accuracy\":%.6f,"
            "\"certified_rate\":%.6f,"
            "\"average_verifier_checks\":%.6f,"
            "\"final_loss\":%.6f,"
            "\"parameters\":%zu,"
            "\"forward_flops\":%zu}\n",
            NN_LAYER_RANKER_SCHEMA,
            report->train_scenes,
            report->eval_scenes,
            report->training_groups,
            report->training_updates,
            report->eval_layers,
            report->top1_layers,
            report->certified_scenes,
            report->verifier_checks,
            report->top1_accuracy,
            report->certified_rate,
            report->average_verifier_checks,
            (double)report->final_loss,
            report->parameters,
            report->forward_flops);
    return ferror(fp) == 0;
}

int nn_layer_task_policy_write_json(FILE *fp, const NNLayerTaskPolicy *policy)
{
    if (fp == NULL || policy == NULL) {
        return 0;
    }
    const NNLayerTaskPolicyReport *report = &policy->report;
    fprintf(fp,
            "{\"schema\":\"%s.summary\","
            "\"train_tasks\":%zu,"
            "\"eval_tasks\":%zu,"
            "\"training_updates\":%zu,"
            "\"solved_tasks\":%zu,"
            "\"top1_tasks\":%zu,"
            "\"verifier_checks\":%zu,"
            "\"exact_rate\":%.6f,"
            "\"top1_accuracy\":%.6f,"
            "\"final_loss\":%.6f,"
            "\"parameters\":%zu,"
            "\"forward_flops\":%zu}\n",
            NN_LAYER_TASK_POLICY_SCHEMA,
            report->train_tasks,
            report->eval_tasks,
            report->training_updates,
            report->solved_tasks,
            report->top1_tasks,
            report->verifier_checks,
            report->exact_rate,
            report->top1_accuracy,
            (double)report->final_loss,
            report->parameters,
            report->forward_flops);
    return ferror(fp) == 0;
}

static int layer_shape_cell(size_t shape, size_t x, size_t y)
{
    static const uint8_t masks[][9] = {
        {1, 1, 0, 1, 1, 0, 0, 0, 0},
        {0, 1, 0, 1, 1, 1, 0, 1, 0},
        {1, 0, 0, 1, 1, 1, 0, 0, 1},
        {1, 1, 1, 0, 1, 0, 0, 1, 0},
        {1, 1, 1, 1, 0, 1, 1, 1, 1},
    };
    size_t shape_count = sizeof(masks) / sizeof(masks[0]);
    return masks[shape % shape_count][y * 3u + x] != 0u;
}

static int layer_shape_fits(const NNBitGrid *source,
                            const NNBitGrid *target,
                            size_t x0,
                            size_t y0,
                            int dx,
                            int dy,
                            size_t shape)
{
    for (size_t y = 0u; y < 3u; ++y) {
        for (size_t x = 0u; x < 3u; ++x) {
            if (!layer_shape_cell(shape, x, y)) {
                continue;
            }
            size_t sx = x0 + x;
            size_t sy = y0 + y;
            size_t tx = 0u;
            size_t ty = 0u;
            if (sx >= source->width || sy >= source->height ||
                !layer_offset(sx, dx, target->width, &tx) ||
                !layer_offset(sy, dy, target->height, &ty)) {
                return 0;
            }
            if ((source->cells[sy * source->width + sx] & 1u) != 0u ||
                (target->cells[ty * target->width + tx] & 1u) != 0u) {
                return 0;
            }
            for (int ny = -1; ny <= 1; ++ny) {
                for (int nx = -1; nx <= 1; ++nx) {
                    size_t nsx = 0u;
                    size_t nsy = 0u;
                    size_t ntx = 0u;
                    size_t nty = 0u;
                    if (layer_offset(sx, nx, source->width, &nsx) &&
                        layer_offset(sy, ny, source->height, &nsy) &&
                        (source->cells[nsy * source->width + nsx] & 1u) != 0u) {
                        return 0;
                    }
                    if (layer_offset(tx, nx, target->width, &ntx) &&
                        layer_offset(ty, ny, target->height, &nty) &&
                        (target->cells[nty * target->width + ntx] & 1u) != 0u) {
                        return 0;
                    }
                }
            }
        }
    }
    return 1;
}

static void layer_paint_shape(NNBitGrid *grid, size_t x0, size_t y0, size_t shape)
{
    for (size_t y = 0u; y < 3u; ++y) {
        for (size_t x = 0u; x < 3u; ++x) {
            if (layer_shape_cell(shape, x, y)) {
                grid->cells[(y0 + y) * grid->width + x0 + x] = 1u;
            }
        }
    }
}

static int layer_generate_scene_with_translation(NNBitGrid *source,
                                                 NNBitGrid *target,
                                                 size_t width,
                                                 size_t height,
                                                 size_t object_count,
                                                 int dx,
                                                 int dy,
                                                 unsigned int seed)
{
    if (source == NULL || target == NULL || width < 8u || height < 8u ||
        object_count == 0u || object_count > NN_LAYER_MAX_PROPOSALS) {
        return 0;
    }
    if (!nn_bitgrid_init(source, width, height) || !nn_bitgrid_init(target, width, height)) {
        nn_bitgrid_free(source);
        nn_bitgrid_free(target);
        return 0;
    }

    uint32_t rng = seed == 0u ? 1u : seed;
    for (size_t object = 0u; object < object_count; ++object) {
        int placed = 0;
        for (size_t attempt = 0u; attempt < 512u && !placed; ++attempt) {
            size_t shape = (size_t)(layer_rng_next(&rng) % 5u);
            size_t x0 = 1u + (size_t)(layer_rng_next(&rng) % (uint32_t)(width - 4u));
            size_t y0 = 1u + (size_t)(layer_rng_next(&rng) % (uint32_t)(height - 4u));
            if (!layer_shape_fits(source, target, x0, y0, dx, dy, shape)) {
                continue;
            }
            layer_paint_shape(source, x0, y0, shape);
            size_t tx = 0u;
            size_t ty = 0u;
            (void)layer_offset(x0, dx, target->width, &tx);
            (void)layer_offset(y0, dy, target->height, &ty);
            layer_paint_shape(target, tx, ty, shape);
            placed = 1;
        }
        if (!placed) {
            nn_bitgrid_free(source);
            nn_bitgrid_free(target);
            return 0;
        }
    }
    return 1;
}

int nn_layer_generate_motion_scene(NNBitGrid *source,
                                   NNBitGrid *target,
                                   size_t width,
                                   size_t height,
                                   size_t object_count,
                                   int max_translation,
                                   unsigned int seed)
{
    if (source == NULL || target == NULL || width < 8u || height < 8u ||
        object_count == 0u || object_count > NN_LAYER_MAX_PROPOSALS ||
        max_translation < 1) {
        return 0;
    }
    if (!nn_bitgrid_init(source, width, height) || !nn_bitgrid_init(target, width, height)) {
        nn_bitgrid_free(source);
        nn_bitgrid_free(target);
        return 0;
    }
    uint32_t rng = seed == 0u ? 1u : seed;
    int span = max_translation * 2 + 1;
    for (size_t object = 0u; object < object_count; ++object) {
        int placed = 0;
        for (size_t attempt = 0u; attempt < 512u && !placed; ++attempt) {
            size_t shape = (size_t)(layer_rng_next(&rng) % 5u);
            int dx = (int)(layer_rng_next(&rng) % (uint32_t)span) - max_translation;
            int dy = (int)(layer_rng_next(&rng) % (uint32_t)span) - max_translation;
            if (dx == 0 && dy == 0 && (object % 4u) != 0u) {
                dx = object % 2u == 0u ? 1 : -1;
            }
            size_t x0 = 1u + (size_t)(layer_rng_next(&rng) % (uint32_t)(width - 4u));
            size_t y0 = 1u + (size_t)(layer_rng_next(&rng) % (uint32_t)(height - 4u));
            if (!layer_shape_fits(source, target, x0, y0, dx, dy, shape)) {
                continue;
            }
            layer_paint_shape(source, x0, y0, shape);
            size_t tx = 0u;
            size_t ty = 0u;
            (void)layer_offset(x0, dx, target->width, &tx);
            (void)layer_offset(y0, dy, target->height, &ty);
            layer_paint_shape(target, tx, ty, shape);
            placed = 1;
        }
        if (!placed) {
            nn_bitgrid_free(source);
            nn_bitgrid_free(target);
            return 0;
        }
    }
    return 1;
}

int nn_layer_generate_global_motion_task(NNBitGrid *demo_source,
                                         NNBitGrid *demo_target,
                                         NNBitGrid *query_source,
                                         NNBitGrid *query_target,
                                         size_t width,
                                         size_t height,
                                         size_t object_count,
                                         int max_translation,
                                         unsigned int seed,
                                         int *out_dx,
                                         int *out_dy)
{
    if (demo_source == NULL || demo_target == NULL || query_source == NULL ||
        query_target == NULL || width < 8u || height < 8u || object_count == 0u ||
        object_count > NN_LAYER_MAX_PROPOSALS || max_translation < 1) {
        return 0;
    }

    uint32_t rng = seed == 0u ? 1u : seed;
    int span = max_translation * 2 + 1;
    for (size_t attempt = 0u; attempt < 64u; ++attempt) {
        int dx = (int)(layer_rng_next(&rng) % (uint32_t)span) - max_translation;
        int dy = (int)(layer_rng_next(&rng) % (uint32_t)span) - max_translation;
        if (dx == 0 && dy == 0) {
            dx = 1;
        }

        NNBitGrid ds = {0};
        NNBitGrid dt = {0};
        NNBitGrid qs = {0};
        NNBitGrid qt = {0};
        unsigned int demo_seed = layer_rng_next(&rng);
        unsigned int query_seed = layer_rng_next(&rng);
        int ok = layer_generate_scene_with_translation(&ds,
                                                       &dt,
                                                       width,
                                                       height,
                                                       object_count,
                                                       dx,
                                                       dy,
                                                       demo_seed) &&
                 layer_generate_scene_with_translation(&qs,
                                                       &qt,
                                                       width,
                                                       height,
                                                       object_count,
                                                       dx,
                                                       dy,
                                                       query_seed);
        if (ok) {
            *demo_source = ds;
            *demo_target = dt;
            *query_source = qs;
            *query_target = qt;
            if (out_dx != NULL) {
                *out_dx = dx;
            }
            if (out_dy != NULL) {
                *out_dy = dy;
            }
            return 1;
        }
        nn_bitgrid_free(&ds);
        nn_bitgrid_free(&dt);
        nn_bitgrid_free(&qs);
        nn_bitgrid_free(&qt);
    }
    return 0;
}
