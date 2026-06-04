#ifndef NN_HOLOGRAPHIC_H
#define NN_HOLOGRAPHIC_H

#include <stddef.h>
#include <stdint.h>

/*
 * nn_holographic — holographic memory substrate for crlplrimes.
 *
 * Wraps the leOS bridge. Three roles:
 *   1. Hypothesis library — store/recall certified rules
 *   2. Scorer features   — 4 floats appended to the 96-feature envelope
 *   3. Reinforcement     — strengthen state→action after verification
 *
 * Thread-unsafe. Single global instance per process.
 */

#define NN_HOLOGRAPHIC_FEATURE_COUNT 4

/* Initialize the holographic subsystem. dim must be power of 2.
 * Call once at startup. Returns 0 on success. */
int nn_holographic_init(size_t dim);

/* Shutdown and free resources. */
void nn_holographic_shutdown(void);

/* Store a certified rule: task_signature → rule_vector.
 * Both are dim-dimensional unit-norm vectors. */
void nn_holographic_store_rule(const double *task_sig, const double *rule_vec);

/* Retrieve blended context for a task signature. */
void nn_holographic_recall_context(const double *task_sig, double *context_out);

/* Extract 4 holographic features from (state, candidate).
 * Appends to the standard 96-feature scorer envelope.
 *   f[96] = cos(state, holographic_context(state))
 *   f[97] = cos(candidate, holographic_recall_recent(state))
 *   f[98] = cos(holographic_context(state), candidate)
 *   f[99] = holographic_confidence (lower bound)
 *
 * features_out must have room for NN_HOLOGRAPHIC_FEATURE_COUNT floats. */
void nn_holographic_extract_features(const double *state_vec,
                                     const double *candidate_vec,
                                     double *features_out);

/* Reinforce: strengthen the state→action association.
 * Call after the verifier confirms the action was correct.
 * Multiple calls accumulate — this is the long-running process. */
void nn_holographic_reinforce(const double *state_vec,
                              const double *action_vec);

/* Store a neural network weight holographically.
 * weight_key is e.g. "hidden_0_w0". */
void nn_holographic_store_weight(const char *weight_key,
                                 const double *weight_vec);

/* Recall a stored weight. fidelity_out may be NULL. */
void nn_holographic_recall_weight(const char *weight_key,
                                  double *weight_out,
                                  double *fidelity_out);

/* Diagnostics. */
size_t nn_holographic_rule_count(void);
size_t nn_holographic_reinforce_count(void);
double nn_holographic_rule_fidelity(void);
void   nn_holographic_stats(void);

/* Save the reinforce memory trace to a binary file. Returns 0 on success. */
int nn_holographic_save_trace(const char *path);

/* Load and merge a reinforce memory trace. Returns 0 on success. */
int nn_holographic_merge_trace(const char *path);

#endif /* NN_HOLOGRAPHIC_H */
