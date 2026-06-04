#include "leos_bridge.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

struct CrlpBridge {
    HolographicNet *net;
    LeosEngine     *eng;
    size_t          dim;
    size_t mem_rules, mem_weights, mem_recent, mem_context;
    size_t layer_features;
    size_t rule_stored, weight_stored, reinforce_count;
};

static uint64_t fnv1a(const char *s) {
    uint64_t h = 14695981039346656037ULL;
    while (*s) { h ^= (uint64_t)(unsigned char)*s++; h *= 1099511628211ULL; }
    return h;
}

CrlpBridge *crlp_bridge_create(size_t dim) {
    CrlpBridge *b = calloc(1, sizeof(CrlpBridge));
    if (!b) return NULL;
    b->net = hnn_create(dim);
    if (!b->net) { free(b); return NULL; }
    b->eng = hnn_engine(b->net);
    b->dim = dim;
    b->mem_rules   = hnn_add_memory(b->net);
    b->mem_weights = hnn_add_memory(b->net);
    b->mem_recent  = hnn_add_memory(b->net);
    b->mem_context = hnn_add_memory(b->net);
    b->layer_features = hnn_add_layer(b->net, 4, 0.0);
    return b;
}

void crlp_bridge_destroy(CrlpBridge *b) {
    if (!b) return;
    hnn_destroy(b->net);
    free(b);
}

void crlp_bridge_store_rule(CrlpBridge *b, const double *task_sig, const double *rule_vec) {
    hnn_memory_store(b->net, b->mem_rules, task_sig, rule_vec);
    b->rule_stored++;
}

void crlp_bridge_recall_context(CrlpBridge *b, const double *task_sig, double *context_out) {
    hnn_memory_recall(b->net, b->mem_rules, task_sig, context_out);
}

void crlp_bridge_extract_features(CrlpBridge *b, const double *state_vec,
                                  const double *candidate_vec, double features_out[4]) {
    double *ctx = malloc(b->dim * sizeof(double));
    crlp_bridge_recall_context(b, state_vec, ctx);
    features_out[0] = leos_engine_dot(b->eng, state_vec, ctx);

    double *recent = malloc(b->dim * sizeof(double));
    hnn_memory_recall(b->net, b->mem_recent, state_vec, recent);
    features_out[1] = leos_engine_dot(b->eng, state_vec, recent);

    features_out[2] = leos_engine_dot(b->eng, ctx, candidate_vec);

    size_t n = b->rule_stored + b->reinforce_count;
    features_out[3] = n > 0 ? 1.0 / sqrt((double)n) : 0.0;

    free(ctx); free(recent);
}

void crlp_bridge_store_weight(CrlpBridge *b, const char *weight_key, const double *weight_vec) {
    double *key = malloc(b->dim * sizeof(double));
    leos_engine_keygen(b->eng, fnv1a(weight_key), key);
    hnn_memory_store(b->net, b->mem_weights, key, weight_vec);
    b->weight_stored++;
    free(key);
}

void crlp_bridge_recall_weight(CrlpBridge *b, const char *weight_key,
                               double *weight_out, double *fidelity_out) {
    double *key = malloc(b->dim * sizeof(double));
    leos_engine_keygen(b->eng, fnv1a(weight_key), key);
    hnn_memory_recall(b->net, b->mem_weights, key, weight_out);
    if (fidelity_out)
        *fidelity_out = b->weight_stored > 0 ? 1.0/sqrt((double)b->weight_stored) : 1.0;
    free(key);
}

void crlp_bridge_reinforce(CrlpBridge *b, const double *state_vec, const double *action_vec) {
    hnn_memory_store(b->net, b->mem_recent, state_vec, action_vec);
    b->reinforce_count++;
}

size_t crlp_bridge_rule_count(const CrlpBridge *b)      { return b->rule_stored; }
size_t crlp_bridge_weight_count(const CrlpBridge *b)    { return b->weight_stored; }
size_t crlp_bridge_reinforce_count(const CrlpBridge *b) { return b->reinforce_count; }

double crlp_bridge_rule_fidelity(const CrlpBridge *b) {
    return b->rule_stored > 0 ? 1.0/sqrt((double)b->rule_stored) : 1.0;
}
double crlp_bridge_weight_fidelity(const CrlpBridge *b) {
    return b->weight_stored > 0 ? 1.0/sqrt((double)b->weight_stored) : 1.0;
}
double crlp_bridge_reinforce_fidelity(const CrlpBridge *b) {
    return b->reinforce_count > 0 ? 1.0/sqrt((double)b->reinforce_count) : 1.0;
}

void crlp_bridge_stats(const CrlpBridge *b) {
    printf("=== CrlpBridge (dim=%zu) ===\n", b->dim);
    printf("  rules:       %5zu  (fidelity est: %.4f)\n", b->rule_stored, crlp_bridge_rule_fidelity(b));
    printf("  weights:     %5zu  (fidelity est: %.4f)\n", b->weight_stored, crlp_bridge_weight_fidelity(b));
    printf("  reinforces:  %5zu  (fidelity est: %.4f)\n", b->reinforce_count, crlp_bridge_reinforce_fidelity(b));
    printf("  total:       %5zu\n", b->rule_stored + b->weight_stored + b->reinforce_count);
}

int crlp_bridge_save_reinforce_trace(const CrlpBridge *b, const char *path) {
    if (!b || !path) return -1;
    HolographicMemory *mem = &b->net->memories[b->mem_recent];
    FILE *fp = fopen(path, "wb");
    if (!fp) return -1;
    if (fwrite(&b->dim, sizeof(size_t), 1, fp) != 1) { fclose(fp); return -1; }
    if (fwrite(mem->M, sizeof(double), b->dim, fp) != b->dim) { fclose(fp); return -1; }
    fclose(fp);
    return 0;
}

int crlp_bridge_merge_reinforce_trace(CrlpBridge *b, const char *path) {
    if (!b || !path) return -1;
    FILE *fp = fopen(path, "rb");
    if (!fp) return -1;
    size_t dim;
    if (fread(&dim, sizeof(size_t), 1, fp) != 1 || dim != b->dim) { fclose(fp); return -1; }
    double *ext = malloc(dim * sizeof(double));
    if (!ext) { fclose(fp); return -1; }
    if (fread(ext, sizeof(double), dim, fp) != dim) { free(ext); fclose(fp); return -1; }
    fclose(fp);
    HolographicMemory *mem = &b->net->memories[b->mem_recent];
    leos_engine_axpy(b->eng, 1.0, ext, mem->M, mem->M);
    b->reinforce_count += 1;
    free(ext);
    return 0;
}
