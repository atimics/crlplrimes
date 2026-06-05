#ifndef MEMORY_NEURON_H
#define MEMORY_NEURON_H

#include "nn_holographic_ranker.h"
#include "holographic/leos_engine.h"
#include <stddef.h>
#include <stdint.h>

/*
 * Memory Neuron -- a holographic memory cell paired with a traditional neuron.
 *
 * Architecture:
 *   Memory neurons live in an MxN grid alongside traditional neural-net
 *   neurons. Each memory neuron accumulates verified (state, action) pairs
 *   into a holographic trace on a slow tick, then feeds 4 context features
 *   into its paired traditional neuron on every fast tick.
 *
 * Two ticks:
 *   Fast tick (every decision): memory neurons are read-only. They provide
 *     4 context features appended to the standard scorer envelope.
 *   Slow tick (every K decisions): the K nearest memory neurons store
 *     bind(state, verified_action) into their traces and recompute context.
 *
 * Routing:
 *   Each memory neuron has a centroid in feature space. At each slow tick,
 *   only the K nearest neurons to the current feature vector are active.
 *   Centroids drift toward stored states via online clustering.
 *
 * The neural net learns through normal SGD whether and how to use the
 * 4 memory context features. No post-hoc blending needed.
 *
 * Default configuration (from holonet experiments):
 *   MxN = 4x4 = 16 memory neurons
 *   K = 4 active per tick (25% sparse activation)
 *   DIM = 2048 (16KB per trace)
 *   slow_tick_interval = 8
 *   centroid_learning_rate = 0.01
 */

#define MEMORY_NEURON_CONTEXT_FEATURES 4
#define MEMORY_NEURON_DEFAULT_DIM 2048
#define MEMORY_NEURON_DEFAULT_ROWS 4
#define MEMORY_NEURON_DEFAULT_COLS 4
#define MEMORY_NEURON_DEFAULT_K 4
#define MEMORY_NEURON_DEFAULT_TICK_INTERVAL 8
#define MEMORY_NEURON_DEFAULT_CENTROID_LR 0.01

/* A single memory neuron: holographic trace + feature-space centroid. */
typedef struct {
    HolographicRanker *ranker;   /* holographic store for this field */
    double *centroid;             /* DIM-dimensional center of receptive field */
    double *f_centroid;           /* feature-space centroid (feature_count dims) */
    double *scratch;              /* DIM doubles, for intermediate results */
    size_t stored;                /* number of pairs stored */
    double fidelity;              /* 1/sqrt(stored), cached */
    double context[MEMORY_NEURON_CONTEXT_FEATURES]; /* latest context features */
} MemoryNeuron;

/* A grid of memory neurons paired with a traditional neural network. */
typedef struct {
    MemoryNeuron *neurons;        /* row-major: neurons[row * cols + col] */
    size_t rows;
    size_t cols;
    size_t count;                 /* rows * cols */
    size_t dim;                   /* VSA dimension (power of 2) */
    size_t feature_count;         /* dimension of feature-space centroids */
    size_t k_active;              /* top-K active neurons per tick */
    size_t slow_tick_interval;    /* update memory every N ticks */
    size_t tick;                  /* current tick counter */
    double centroid_lr;           /* online clustering learning rate */
    LeosEngine *eng;              /* shared VSA compute context */
    double *scratch_state;        /* DIM doubles, for state embedding */
    double *scratch_action;       /* DIM doubles, for action embedding */
    double *scratch_context;      /* DIM doubles, for recall computation */
    double *context_out;          /* MEMORY_NEURON_CONTEXT_FEATURES doubles */
} MemoryNeuronGrid;

/* ---- Lifecycle ---- */

MemoryNeuronGrid *memory_neuron_grid_create(size_t dim,
                                            size_t rows, size_t cols,
                                            size_t feature_count,
                                            size_t k_active,
                                            size_t tick_interval,
                                            double centroid_lr);

void memory_neuron_grid_destroy(MemoryNeuronGrid *grid);

/* ---- Two-Tick Interface ---- */

/* Fast tick: extract 4 memory context features for the current state
 * and candidate action. Read-only -- no memory updates.
 *
 * Returns 0 on success, -1 if no neurons have been trained yet. */
int memory_neuron_grid_context(MemoryNeuronGrid *grid,
                               const double *state_vec,
                               const double *candidate_vec,
                               const double *features,
                               double *context_out);

/* Slow tick: store a verified (state, action) pair into the K nearest
 * memory neurons and recompute their context features.
 *
 * Call only after the verifier has confirmed the action was correct.
 * Increments the internal tick counter. If tick % tick_interval != 0,
 * this is a no-op (returns 0 immediately).
 *
 * Returns 0 on success, -1 on error. */
int memory_neuron_grid_store(MemoryNeuronGrid *grid,
                             const double *state_vec,
                             const double *action_vec,
                             const double *features);

/* Combined fast-tick scorer envelope extension:
 * Given the existing 96-feature envelope row and the 4 memory context
 * features, fills a 100-feature row (96 original + 4 context).
 * row_capacity must be at least 100. */
void memory_neuron_envelope_extend(const double *base_row_96,
                                   const double *context_4,
                                   double *row_100);

/* ---- Diagnostics ---- */

void memory_neuron_grid_stored_counts(const MemoryNeuronGrid *grid,
                                      size_t *counts_out);

double memory_neuron_grid_avg_fidelity(const MemoryNeuronGrid *grid);

size_t memory_neuron_grid_total_stored(const MemoryNeuronGrid *grid);

size_t memory_neuron_grid_tick(const MemoryNeuronGrid *grid);

#endif /* MEMORY_NEURON_H */

