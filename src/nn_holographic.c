#include "nn_holographic.h"
#include "leos_bridge.h"
#include <stdlib.h>
#include <stdio.h>

static CrlpBridge *g_bridge = NULL;
static size_t g_dim = 0;

int nn_holographic_init(size_t dim) {
    if (g_bridge) return 0; /* already initialized */
    g_bridge = crlp_bridge_create(dim);
    if (!g_bridge) return -1;
    g_dim = dim;
    return 0;
}

void nn_holographic_shutdown(void) {
    crlp_bridge_destroy(g_bridge);
    g_bridge = NULL;
    g_dim = 0;
}

void nn_holographic_store_rule(const double *task_sig, const double *rule_vec) {
    if (!g_bridge) return;
    crlp_bridge_store_rule(g_bridge, task_sig, rule_vec);
}

void nn_holographic_recall_context(const double *task_sig, double *context_out) {
    if (!g_bridge) return;
    crlp_bridge_recall_context(g_bridge, task_sig, context_out);
}

void nn_holographic_extract_features(const double *state_vec,
                                     const double *candidate_vec,
                                     double *features_out) {
    if (!g_bridge) {
        for (int i = 0; i < NN_HOLOGRAPHIC_FEATURE_COUNT; i++)
            features_out[i] = 0.0;
        return;
    }
    crlp_bridge_extract_features(g_bridge, state_vec, candidate_vec, features_out);
}

void nn_holographic_reinforce(const double *state_vec, const double *action_vec) {
    if (!g_bridge) return;
    crlp_bridge_reinforce(g_bridge, state_vec, action_vec);
}

void nn_holographic_store_weight(const char *weight_key, const double *weight_vec) {
    if (!g_bridge) return;
    crlp_bridge_store_weight(g_bridge, weight_key, weight_vec);
}

void nn_holographic_recall_weight(const char *weight_key,
                                  double *weight_out, double *fidelity_out) {
    if (!g_bridge) return;
    crlp_bridge_recall_weight(g_bridge, weight_key, weight_out, fidelity_out);
}

size_t nn_holographic_rule_count(void) {
    return g_bridge ? crlp_bridge_rule_count(g_bridge) : 0;
}

size_t nn_holographic_reinforce_count(void) {
    return g_bridge ? crlp_bridge_reinforce_count(g_bridge) : 0;
}

double nn_holographic_rule_fidelity(void) {
    return g_bridge ? crlp_bridge_rule_fidelity(g_bridge) : 0.0;
}

void nn_holographic_stats(void) {
    if (g_bridge) crlp_bridge_stats(g_bridge);
}

int nn_holographic_save_trace(const char *path) {
    if (!g_bridge) return -1;
    return crlp_bridge_save_reinforce_trace(g_bridge, path);
}

int nn_holographic_merge_trace(const char *path) {
    if (!g_bridge) return -1;
    return crlp_bridge_merge_reinforce_trace(g_bridge, path);
}
