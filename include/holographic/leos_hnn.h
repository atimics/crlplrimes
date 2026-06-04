#ifndef LEOS_HNN_H
#define LEOS_HNN_H

#include "leos_engine.h"
#include <stddef.h>
#include <stdint.h>

/*
 * leOS-HNN: Micro Symbolic Neural Holographic Networks
 *
 * Three computational primitives that unify memory and compute
 * in a single vector space on the unit hypersphere S^{d-1}:
 *
 *   HolographicNeuron  — bind(input, weight) replaces dot+activation
 *   HolographicMemory  — content-addressable key-value store
 *   HolographicLayer   — bundle of neurons, vector in = vector out
 *
 * These are NOT metaphors.  The VSA operations (bind, unbind, bundle)
 * are Turing-complete @kleyko2022vsa.  A holographic layer computes
 * by binding and superposing, not by dot products and activation
 * functions.  Memory and computation are the same operation.
 */

typedef struct LeosHNN HolographicNet;

/* -----------------------------------------------------------------------
 * Holographic Neuron
 *
 * Regular neuron:   y = sigma(w · x + b)         [scalar out]
 * Holographic neuron: y = bind(w, x)             [vector out, same dim]
 *                   y = unbind(w, x)             [content-addressable]
 *
 * The output is a full vector on the hypersphere, not a scalar.
 * This means information is preserved through the layer — unbind
 * can recover the input given the weight, which a dot product destroys.
 * ----------------------------------------------------------------------- */

typedef enum {
    HOLOGRAPHIC_NEURON_BIND = 0,
    HOLOGRAPHIC_NEURON_UNBIND = 1,
    HOLOGRAPHIC_NEURON_EML = 2,
} HolographicNeuronMode;

typedef struct {
    double             *weight; /* dim doubles, unit-norm, flat-spectrum key */
    HolographicNeuronMode mode; /* bind, unbind, or eml */
} HolographicNeuron;

/* -----------------------------------------------------------------------
 * Holographic Memory (content-addressable)
 *
 * Store:   M += bind(key, value)              [accumulate into trace]
 * Recall:  value_hat = unbind(M, key)         [≈ value + noise]
 * Cleanup: iterative residual decode
 *
 * Differs from LeosStore: this is an in-layer memory unit that
 * operates as part of the network graph, not as external storage.
 * ----------------------------------------------------------------------- */

typedef struct {
    double *M;          /* memory trace (dim doubles, un-normalized) */
    double *scratch;    /* dim doubles, for intermediate results */
    size_t  stored;     /* number of pairs stored */
} HolographicMemory;

/* -----------------------------------------------------------------------
 * Holographic Layer
 *
 * A layer bundles N holographic neurons.  Given input vector x:
 *   y = normalize( sum_{i=1..N} neuron_i(x) )
 *
 * Each neuron_i(x) is either bind(w_i, x) or unbind(w_i, x).
 * The output y is a single vector on S^{d-1}.
 *
 * Stacking layers: y_{l+1} = HolographicLayer_l(y_l)
 * The entire network preserves dimension — every intermediate
 * representation is a full vector on the hypersphere.
 * ----------------------------------------------------------------------- */

typedef struct {
    HolographicNeuron *neurons;
    size_t             count;
    double            *output;    /* dim doubles, scratch for layer output */
} HolographicLayer;

/* -----------------------------------------------------------------------
 * The full network: memory units + holographic layers
 *
 * Architecture:
 *   input vector (on S^{d-1})
 *     → holographic memory recall (retrieve relevant context)
 *     → holographic layer 1 (bind/unbind with learned keys)
 *     → holographic layer 2
 *     → ... 
 *     → holographic memory store (encode result)
 *     → output vector (on S^{d-1})
 *
 * The output can be fed to a tiny neural scorer (dot product with
 * candidate embeddings) for ranking — this is where micro-neural
 * meets holographic-symbolic.
 * ----------------------------------------------------------------------- */

struct LeosHNN {
    LeosEngine        *eng;
    HolographicMemory *memories;
    size_t             memory_count;
    HolographicLayer  *layers;
    size_t             layer_count;
};

/* -----------------------------------------------------------------------
 * API
 * ----------------------------------------------------------------------- */

/* Create a holographic network. dim must be power of 2. */
HolographicNet *hnn_create(size_t dim);

/* Destroy. */
void hnn_destroy(HolographicNet *net);

/* Add a content-addressable memory unit. Returns index. */
size_t hnn_add_memory(HolographicNet *net);

/* Store a key-value pair in memory unit `mem_idx`. */
void hnn_memory_store(HolographicNet *net, size_t mem_idx,
                      const double *key, const double *value);

/* Recall: retrieve value for key from memory unit `mem_idx`. */
void hnn_memory_recall(HolographicNet *net, size_t mem_idx,
                       const double *key, double *out);

/* Add a holographic layer with `neuron_count` neurons.
 * Weights are initialized randomly (flat-spectrum keys).
 * `inverted_ratio` sets fraction of neurons in unbind mode (0.0 to 1.0).
 * Returns layer index. */
size_t hnn_add_layer(HolographicNet *net, size_t neuron_count,
                     double inverted_ratio);

/* Add a mixed-mode layer: neuron_count neurons with given mode fractions
 * (bind_frac + unbind_frac + eml_frac should sum to 1.0 or less).
 * Remaining neurons default to bind mode. Returns layer index. */
size_t hnn_add_layer_mixed(HolographicNet *net, size_t neuron_count,
                           double bind_frac, double unbind_frac,
                           double eml_frac);

/* Forward pass through layer `layer_idx`: out = layer(in). */
void hnn_layer_forward(HolographicNet *net, size_t layer_idx,
                       const double *in, double *out);

/* Forward pass through entire network:
 *   recall mem[0](input) → layer[0] → layer[1] → ... → store mem[1](output)
 * mem[0] and mem[1] must exist. Returns output vector in `out`. */
void hnn_forward(HolographicNet *net, const double *key,
                 const double *input, double *out);

/* Hebbian-like update: reinforce the connection between key and current
 * network output. Adds bind(key, output) to memory unit's trace. */
void hnn_reinforce(HolographicNet *net, size_t mem_idx,
                   const double *key, const double *output);

/* Get the engine for direct VSA operations. */
LeosEngine *hnn_engine(HolographicNet *net);

#endif /* LEOS_HNN_H */
