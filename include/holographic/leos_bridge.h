#ifndef LEOS_BRIDGE_H
#define LEOS_BRIDGE_H

#include "leos_hnn.h"
#include <stddef.h>
#include <stdint.h>

/*
 * leOS Bridge — integration layer for crlplrimes.
 *
 * Three subsystems, one holographic substrate:
 *
 *   1. Hypothesis Library   — certified rules stored holographically.
 *      Store:  bind(task_sig, rule_vec) → accumulate into trace.
 *      Recall: unbind(trace, task_sig)  → blended context vector.
 *
 *   2. Scorer Features       — holographic recall as feature engineering.
 *      Given a state, retrieve the "what has worked here before" signal
 *      and feed it as additional features into the 96-dim scorer envelope.
 *
 *   3. Weight Storage        — the scorer's own weights live in memory.
 *      Store:  bind(layer_key, weight_vec) for each weight.
 *      Recall: unbind(trace, layer_key) for each weight.
 *      Reinforcement: re-store after every N successful verifications.
 */

typedef struct CrlpBridge CrlpBridge;

/* ---- Lifecycle ---- */

/* Create bridge with given dimension (power of 2). */
CrlpBridge *crlp_bridge_create(size_t dim);

/* Destroy and free all resources. */
void crlp_bridge_destroy(CrlpBridge *b);

/* ---- Hypothesis Library ---- */

/* Store a certified rule: task_signature → rule_vector.
 * Both are dim-dimensional, unit-norm vectors on the hypersphere. */
void crlp_bridge_store_rule(CrlpBridge *b,
                            const double *task_sig,
                            const double *rule_vec);

/* Retrieve the blended context of all stored rules for a task.
 * Returns a single vector on S^{d-1} that captures the weighted
 * superposition of all rules relevant to task_sig.
 * Fidelity decays as 1/sqrt(N) where N is the total stored rules. */
void crlp_bridge_recall_context(CrlpBridge *b,
                                const double *task_sig,
                                double *context_out);

/* ---- Scorer Feature Extraction ---- */

/* Extract holographic features from a state vector.
 * Computes:
 *   f[0] = similarity(state, holographic_context(state))
 *   f[1] = similarity(state, holographic_recall(state, "recent"))
 *   f[2] = similarity(holographic_context(state), best_candidate)
 *   f[3] = holographic_recall_confidence (estimated fidelity)
 *
 * These can be appended to the existing 96-feature scorer envelope. */
void crlp_bridge_extract_features(CrlpBridge *b,
                                  const double *state_vec,
                                  const double *candidate_vec,
                                  double features_out[4]);

/* ---- Weight Storage ---- */

/* Store a weight vector under a named key (e.g., "layer0_w0").
 * Used for holographic checkpointing — the network's weights
 * become part of the same memory trace as everything else. */
void crlp_bridge_store_weight(CrlpBridge *b,
                              const char *weight_key,
                              const double *weight_vec);

/* Recall a stored weight. Returns fidelity in *fidelity_out if non-NULL. */
void crlp_bridge_recall_weight(CrlpBridge *b,
                               const char *weight_key,
                               double *weight_out,
                               double *fidelity_out);

/* ---- Reinforcement ---- */

/* Reinforce a state→action association. Call after the verifier
 * confirms the action was correct. Each reinforcement call adds
 * bind(state_vec, action_vec) to the long-term memory trace.
 *
 * This is the long-running process: every successful verification
 * strengthens the memory. No separate training loop needed. */
void crlp_bridge_reinforce(CrlpBridge *b,
                           const double *state_vec,
                           const double *action_vec);

/* ---- Diagnostics ---- */

/* Number of items stored in each subsystem. */
size_t crlp_bridge_rule_count(const CrlpBridge *b);
size_t crlp_bridge_weight_count(const CrlpBridge *b);
size_t crlp_bridge_reinforce_count(const CrlpBridge *b);

/* Estimated fidelity for each memory unit. */
double crlp_bridge_rule_fidelity(const CrlpBridge *b);
double crlp_bridge_weight_fidelity(const CrlpBridge *b);
double crlp_bridge_reinforce_fidelity(const CrlpBridge *b);

/* Dump state to stdout. */
void crlp_bridge_stats(const CrlpBridge *b);

/* Save the reinforce memory trace to a binary file.
 * Format: dim (size_t), then dim doubles. Returns 0 on success. */
int crlp_bridge_save_reinforce_trace(const CrlpBridge *b, const char *path);

/* Load and merge a reinforce memory trace from a binary file.
 * Adds the loaded trace to the current memory. Returns 0 on success. */
int crlp_bridge_merge_reinforce_trace(CrlpBridge *b, const char *path);

#endif /* LEOS_BRIDGE_H */
