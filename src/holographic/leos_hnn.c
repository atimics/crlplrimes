#include "leos_hnn.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

static uint64_t xorshift64(uint64_t *state) {
    uint64_t x = *state;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    *state = x;
    return x;
}

HolographicNet *hnn_create(size_t dim) {
    LeosEngine *eng = leos_engine_create(dim);
    if (!eng) return NULL;
    HolographicNet *net = calloc(1, sizeof(HolographicNet));
    if (!net) { leos_engine_destroy(eng); return NULL; }
    net->eng = eng;
    return net;
}

void hnn_destroy(HolographicNet *net) {
    if (!net) return;
    for (size_t i = 0; i < net->memory_count; i++) {
        free(net->memories[i].M);
        free(net->memories[i].scratch);
    }
    free(net->memories);
    for (size_t i = 0; i < net->layer_count; i++) {
        for (size_t j = 0; j < net->layers[i].count; j++)
            free(net->layers[i].neurons[j].weight);
        free(net->layers[i].neurons);
        free(net->layers[i].output);
    }
    free(net->layers);
    leos_engine_destroy(net->eng);
    free(net);
}

LeosEngine *hnn_engine(HolographicNet *net) { return net->eng; }

size_t hnn_add_memory(HolographicNet *net) {
    size_t idx = net->memory_count;
    net->memories = realloc(net->memories, (idx + 1) * sizeof(HolographicMemory));
    if (!net->memories) return (size_t)-1;
    size_t dim = leos_engine_dim(net->eng);
    HolographicMemory *mem = &net->memories[idx];
    mem->M       = calloc(dim, sizeof(double));
    mem->scratch = calloc(dim, sizeof(double));
    mem->stored  = 0;
    net->memory_count = idx + 1;
    return idx;
}

void hnn_memory_store(HolographicNet *net, size_t mem_idx,
                      const double *key, const double *value) {
    if (mem_idx >= net->memory_count) return;
    HolographicMemory *mem = &net->memories[mem_idx];
    leos_engine_bind(net->eng, key, value, mem->scratch);
    leos_engine_add(net->eng, mem->M, mem->scratch, mem->M);
    mem->stored++;
}

void hnn_memory_recall(HolographicNet *net, size_t mem_idx,
                       const double *key, double *out) {
    if (mem_idx >= net->memory_count || net->memories[mem_idx].stored == 0) {
        memcpy(out, key, leos_engine_dim(net->eng) * sizeof(double));
        return;
    }
    HolographicMemory *mem = &net->memories[mem_idx];
    memcpy(mem->scratch, mem->M, leos_engine_dim(net->eng) * sizeof(double));
    leos_engine_normalize(net->eng, mem->scratch);
    leos_engine_unbind(net->eng, mem->scratch, key, out);
}

size_t hnn_add_layer(HolographicNet *net, size_t neuron_count,
                     double inverted_ratio) {
    size_t dim = leos_engine_dim(net->eng);
    size_t idx = net->layer_count;
    net->layers = realloc(net->layers, (idx + 1) * sizeof(HolographicLayer));
    if (!net->layers) return (size_t)-1;
    HolographicLayer *layer = &net->layers[idx];
    layer->count   = neuron_count;
    layer->neurons = calloc(neuron_count, sizeof(HolographicNeuron));
    layer->output  = calloc(dim, sizeof(double));
    if (!layer->neurons || !layer->output) return (size_t)-1;
    uint64_t rng = 42 + idx * 1000;
    size_t n_inverted = (size_t)(neuron_count * inverted_ratio);
    for (size_t i = 0; i < neuron_count; i++) {
        layer->neurons[i].weight = malloc(dim * sizeof(double));
        layer->neurons[i].mode   = (i < n_inverted)
            ? HOLOGRAPHIC_NEURON_UNBIND : HOLOGRAPHIC_NEURON_BIND;
        uint64_t seed = xorshift64(&rng);
        leos_engine_keygen(net->eng, seed, layer->neurons[i].weight);
    }
    net->layer_count = idx + 1;
    return idx;
}

size_t hnn_add_layer_mixed(HolographicNet *net, size_t neuron_count,
                           double bind_frac, double unbind_frac,
                           double eml_frac) {
    size_t dim = leos_engine_dim(net->eng);
    size_t idx = net->layer_count;
    net->layers = realloc(net->layers, (idx + 1) * sizeof(HolographicLayer));
    if (!net->layers) return (size_t)-1;
    HolographicLayer *layer = &net->layers[idx];
    layer->count   = neuron_count;
    layer->neurons = calloc(neuron_count, sizeof(HolographicNeuron));
    layer->output  = calloc(dim, sizeof(double));
    if (!layer->neurons || !layer->output) return (size_t)-1;

    size_t n_bind   = (size_t)(neuron_count * bind_frac);
    size_t n_unbind = (size_t)(neuron_count * unbind_frac);
    size_t n_eml    = (size_t)(neuron_count * eml_frac);
    uint64_t rng = 42 + idx * 1000;

    size_t i = 0;
    for (; i < n_bind && i < neuron_count; i++) {
        layer->neurons[i].weight = malloc(dim * sizeof(double));
        layer->neurons[i].mode   = HOLOGRAPHIC_NEURON_BIND;
        leos_engine_keygen(net->eng, xorshift64(&rng), layer->neurons[i].weight);
    }
    for (size_t j = 0; j < n_unbind && i < neuron_count; j++, i++) {
        layer->neurons[i].weight = malloc(dim * sizeof(double));
        layer->neurons[i].mode   = HOLOGRAPHIC_NEURON_UNBIND;
        leos_engine_keygen(net->eng, xorshift64(&rng), layer->neurons[i].weight);
    }
    for (size_t j = 0; j < n_eml && i < neuron_count; j++, i++) {
        layer->neurons[i].weight = malloc(dim * sizeof(double));
        layer->neurons[i].mode   = HOLOGRAPHIC_NEURON_EML;
        leos_engine_keygen(net->eng, xorshift64(&rng), layer->neurons[i].weight);
    }
    /* remaining default to bind */
    for (; i < neuron_count; i++) {
        layer->neurons[i].weight = malloc(dim * sizeof(double));
        layer->neurons[i].mode   = HOLOGRAPHIC_NEURON_BIND;
        leos_engine_keygen(net->eng, xorshift64(&rng), layer->neurons[i].weight);
    }

    net->layer_count = idx + 1;
    return idx;
}

void hnn_layer_forward(HolographicNet *net, size_t layer_idx,
                       const double *in, double *out) {
    if (layer_idx >= net->layer_count) return;
    HolographicLayer *layer = &net->layers[layer_idx];
    size_t dim = leos_engine_dim(net->eng);
    leos_engine_zero(net->eng, layer->output);
    double *scratch = malloc(dim * sizeof(double));
    if (!scratch) return;
    for (size_t i = 0; i < layer->count; i++) {
        HolographicNeuron *n = &layer->neurons[i];
        switch (n->mode) {
        case HOLOGRAPHIC_NEURON_UNBIND:
            leos_engine_unbind(net->eng, in, n->weight, scratch);
            break;
        case HOLOGRAPHIC_NEURON_EML:
            leos_engine_eml_bind(net->eng, n->weight, in, scratch);
            break;
        case HOLOGRAPHIC_NEURON_BIND:
        default:
            leos_engine_bind(net->eng, n->weight, in, scratch);
            break;
        }
        leos_engine_add(net->eng, layer->output, scratch, layer->output);
    }
    leos_engine_normalize(net->eng, layer->output);
    memcpy(out, layer->output, dim * sizeof(double));
    free(scratch);
}

void hnn_forward(HolographicNet *net, const double *key,
                 const double *input, double *out) {
    size_t dim = leos_engine_dim(net->eng);
    double *current = malloc(dim * sizeof(double));
    double *next    = malloc(dim * sizeof(double));
    if (!current || !next) { free(current); free(next); return; }
    if (net->memory_count > 0)
        hnn_memory_recall(net, 0, key, current);
    else
        memcpy(current, input, dim * sizeof(double));
    for (size_t l = 0; l < net->layer_count; l++) {
        hnn_layer_forward(net, l, current, next);
        double *tmp = current; current = next; next = tmp;
    }
    if (net->memory_count > 1 && key)
        hnn_memory_store(net, 1, key, current);
    memcpy(out, current, dim * sizeof(double));
    leos_engine_normalize(net->eng, out);
    free(current); free(next);
}

void hnn_reinforce(HolographicNet *net, size_t mem_idx,
                   const double *key, const double *output) {
    hnn_memory_store(net, mem_idx, key, output);
}
