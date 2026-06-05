#include "memory_neuron.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* ---- Internal helpers ---- */

/* Find K smallest element indices in dists array using insertion sort.
 * K is tiny (typically 4) so O(n*K) is fine. */
static void find_k_nearest(const double *dists, size_t n, size_t k,
                           size_t *out_indices, double *out_dists) {
    size_t found = 0;
    for (size_t i = 0; i < n && found < k; i++) {
        out_indices[found] = i;
        out_dists[found] = dists[i];
        found++;
    }
    if (found == 0) return;
    for (size_t i = 1; i < found; i++) {
        size_t key_idx = out_indices[i];
        double key_dist = out_dists[i];
        size_t j = i;
        while (j > 0 && out_dists[j - 1] > key_dist) {
            out_indices[j] = out_indices[j - 1];
            out_dists[j] = out_dists[j - 1];
            j--;
        }
        out_indices[j] = key_idx;
        out_dists[j] = key_dist;
    }
    for (size_t i = k; i < n; i++) {
        if (dists[i] >= out_dists[k - 1]) continue;
        size_t pos = k - 1;
        while (pos > 0 && out_dists[pos - 1] > dists[i]) pos--;
        for (size_t j = k - 1; j > pos; j--) {
            out_indices[j] = out_indices[j - 1];
            out_dists[j] = out_dists[j - 1];
        }
        out_indices[pos] = i;
        out_dists[pos] = dists[i];
    }
}

static double feature_distance(const MemoryNeuron *neuron,
                               const double *features, size_t fc) {
    double d = 0.0;
    for (size_t i = 0; i < fc; i++) {
        double df = features[i] - neuron->f_centroid[i];
        d += df * df;
    }
    return d;
}

static void centroid_nudge(double *centroid, const double *features,
                           size_t fc, double lr) {
    for (size_t i = 0; i < fc; i++)
        centroid[i] += lr * (features[i] - centroid[i]);
}

static void extract_context(MemoryNeuron *neuron,
                            LeosEngine *eng, size_t dim,
                            const double *state_vec,
                            const double *candidate_vec) {
    (void)dim;
    if (neuron->stored == 0) {
        for (int i = 0; i < MEMORY_NEURON_CONTEXT_FEATURES; i++)
            neuron->context[i] = 0.0;
        neuron->fidelity = 0.0;
        return;
    }
    /* Use the ranker's actual recall to get the context vector */
    holo_ranker_recall(neuron->ranker, state_vec, neuron->scratch);
    leos_engine_normalize(eng, neuron->scratch);
    neuron->context[0] = leos_engine_dot(eng, state_vec, neuron->scratch);
    neuron->context[1] = leos_engine_dot(eng, candidate_vec, neuron->scratch);
    neuron->context[2] = neuron->context[1];
    neuron->fidelity = 1.0 / sqrt((double)neuron->stored);
    neuron->context[3] = neuron->fidelity;
}

/* ---- Lifecycle ---- */

MemoryNeuronGrid *memory_neuron_grid_create(size_t dim,
                                            size_t rows, size_t cols,
                                            size_t feature_count,
                                            size_t k_active,
                                            size_t tick_interval,
                                            double centroid_lr) {
    if (dim == 0 || rows == 0 || cols == 0 || k_active == 0) return NULL;
    if (k_active > rows * cols) k_active = rows * cols;

    MemoryNeuronGrid *grid = calloc(1, sizeof(MemoryNeuronGrid));
    if (!grid) return NULL;

    grid->rows = rows;
    grid->cols = cols;
    grid->count = rows * cols;
    grid->dim = dim;
    grid->feature_count = feature_count > 0 ? feature_count : 1;
    grid->k_active = k_active;
    grid->slow_tick_interval = tick_interval > 0 ? tick_interval : 1;
    grid->tick = 0;
    grid->centroid_lr = centroid_lr;

    grid->neurons = calloc(grid->count, sizeof(MemoryNeuron));
    grid->eng = leos_engine_create(dim);
    grid->scratch_state = calloc(dim, sizeof(double));
    grid->scratch_action = calloc(dim, sizeof(double));
    grid->scratch_context = calloc(dim, sizeof(double));
    grid->context_out = calloc(MEMORY_NEURON_CONTEXT_FEATURES, sizeof(double));

    if (!grid->neurons || !grid->eng || !grid->scratch_state ||
        !grid->scratch_action || !grid->scratch_context || !grid->context_out) {
        memory_neuron_grid_destroy(grid);
        return NULL;
    }

    for (size_t i = 0; i < grid->count; i++) {
        MemoryNeuron *n = &grid->neurons[i];
        n->ranker = holo_ranker_create(dim);
        n->centroid = calloc(dim, sizeof(double));
        n->f_centroid = calloc(grid->feature_count, sizeof(double));
        n->scratch = calloc(dim, sizeof(double));
        if (!n->ranker || !n->centroid || !n->f_centroid || !n->scratch) {
            memory_neuron_grid_destroy(grid);
            return NULL;
        }

        uint64_t seed = 42 + i * 997;
        for (size_t f = 0; f < grid->feature_count; f++) {
            seed = (seed + 0x9e3779b97f4a7c15ULL);
            seed = (seed ^ (seed >> 30)) * 0xbf58476d1ce4e5b9ULL;
            seed = (seed ^ (seed >> 27)) * 0x94d049bb133111ebULL;
            n->f_centroid[f] = ((double)(seed % 1000) / 1000.0 - 0.5) * 2.0;
        }
        leos_engine_keygen(grid->eng, seed, n->centroid);

        n->stored = 0;
        n->fidelity = 0.0;
        for (int j = 0; j < MEMORY_NEURON_CONTEXT_FEATURES; j++)
            n->context[j] = 0.0;
    }

    return grid;
}

void memory_neuron_grid_destroy(MemoryNeuronGrid *grid) {
    if (!grid) return;
    if (grid->neurons) {
        for (size_t i = 0; i < grid->count; i++) {
            MemoryNeuron *n = &grid->neurons[i];
            holo_ranker_destroy(n->ranker);
            free(n->centroid);
            free(n->f_centroid);
            free(n->scratch);
        }
        free(grid->neurons);
    }
    leos_engine_destroy(grid->eng);
    free(grid->scratch_state);
    free(grid->scratch_action);
    free(grid->scratch_context);
    free(grid->context_out);
    free(grid);
}

/* ---- Two-Tick Interface ---- */

int memory_neuron_grid_context(MemoryNeuronGrid *grid,
                               const double *state_vec,
                               const double *candidate_vec,
                               const double *features,
                               double *context_out) {
    if (!grid || !state_vec || !candidate_vec || !context_out) return -1;

    double *dists = calloc(grid->count, sizeof(double));
    if (!dists) return -1;
    for (size_t i = 0; i < grid->count; i++)
        dists[i] = feature_distance(&grid->neurons[i],
                                    features, grid->feature_count);

    size_t *active = calloc(grid->k_active, sizeof(size_t));
    double *active_dists = calloc(grid->k_active, sizeof(double));
    if (!active || !active_dists) {
        free(dists); free(active); free(active_dists);
        return -1;
    }
    find_k_nearest(dists, grid->count, grid->k_active, active, active_dists);
    free(dists);

    double total_w = 0.0;
    for (int j = 0; j < MEMORY_NEURON_CONTEXT_FEATURES; j++)
        context_out[j] = 0.0;

    for (size_t i = 0; i < grid->k_active; i++) {
        if (active[i] >= grid->count) continue;
        MemoryNeuron *n = &grid->neurons[active[i]];
        if (n->stored == 0) continue;
        double w = 1.0 / (active_dists[i] + 0.01);
        total_w += w;
        for (int j = 0; j < MEMORY_NEURON_CONTEXT_FEATURES; j++)
            context_out[j] += n->context[j] * w;
    }

    if (total_w > 0.0) {
        for (int j = 0; j < MEMORY_NEURON_CONTEXT_FEATURES; j++)
            context_out[j] /= total_w;
    }

    free(active);
    free(active_dists);
    return 0;
}

int memory_neuron_grid_store(MemoryNeuronGrid *grid,
                             const double *state_vec,
                             const double *action_vec,
                             const double *features) {
    if (!grid || !state_vec || !action_vec || !features) return -1;

    grid->tick++;
    if (grid->tick % grid->slow_tick_interval != 0)
        return 0;

    double *dists = calloc(grid->count, sizeof(double));
    if (!dists) return -1;
    for (size_t i = 0; i < grid->count; i++)
        dists[i] = feature_distance(&grid->neurons[i],
                                    features, grid->feature_count);

    size_t *active = calloc(grid->k_active, sizeof(size_t));
    double *active_dists = calloc(grid->k_active, sizeof(double));
    if (!active || !active_dists) {
        free(dists); free(active); free(active_dists);
        return -1;
    }
    find_k_nearest(dists, grid->count, grid->k_active, active, active_dists);
    free(dists);

    for (size_t i = 0; i < grid->k_active; i++) {
        if (active[i] >= grid->count) continue;
        MemoryNeuron *n = &grid->neurons[active[i]];

        holo_ranker_store(n->ranker, state_vec, action_vec);
        n->stored++;

        centroid_nudge(n->f_centroid, features,
                       grid->feature_count, grid->centroid_lr);

        /* Recompute context: use state binding with centroid as proxy recall */
        double *recalled = calloc(grid->dim, sizeof(double));
        if (recalled) {
            leos_engine_bind(grid->eng, state_vec, n->centroid, recalled);
            leos_engine_normalize(grid->eng, recalled);
            extract_context(n, grid->eng, grid->dim, state_vec, recalled);
            n->context[1] = leos_engine_dot(grid->eng, action_vec, recalled);
            n->context[2] = n->context[1];
            free(recalled);
        }
    }

    free(active);
    free(active_dists);
    return 0;
}

/* ---- Scorer Envelope Extension ---- */

void memory_neuron_envelope_extend(const double *base_row_96,
                                   const double *context_4,
                                   double *row_100) {
    memcpy(row_100, base_row_96, 96 * sizeof(double));
    memcpy(row_100 + 96, context_4, MEMORY_NEURON_CONTEXT_FEATURES * sizeof(double));
}

/* ---- Diagnostics ---- */

void memory_neuron_grid_stored_counts(const MemoryNeuronGrid *grid,
                                      size_t *counts_out) {
    if (!grid || !counts_out) return;
    for (size_t i = 0; i < grid->count; i++)
        counts_out[i] = grid->neurons[i].stored;
}

double memory_neuron_grid_avg_fidelity(const MemoryNeuronGrid *grid) {
    if (!grid || grid->count == 0) return 0.0;
    double sum = 0.0;
    for (size_t i = 0; i < grid->count; i++)
        sum += grid->neurons[i].fidelity;
    return sum / (double)grid->count;
}

size_t memory_neuron_grid_total_stored(const MemoryNeuronGrid *grid) {
    if (!grid) return 0;
    size_t sum = 0;
    for (size_t i = 0; i < grid->count; i++)
        sum += grid->neurons[i].stored;
    return sum;
}

size_t memory_neuron_grid_tick(const MemoryNeuronGrid *grid) {
    return grid ? grid->tick : 0;
}

