#ifndef NN_HOLO_GATE_H
#define NN_HOLO_GATE_H

#include "nn_core.h"

void holo_gate_blend(double holo_fidelity,
                     const double *holo_scores,
                     const nn_real *neural_scores,
                     size_t n_candidates,
                     double *blended_out);

double holo_gate_eval(struct HoloRunner *hr, NeuralNet *neural_net,
                      const double *test_states,  const double *test_action_vecs,
                      const double *test_features, const int *test_labels,
                      size_t n_test, size_t n_candidates, size_t n_features);

void holo_gate_print_compare(const char *name,
                             double holo_acc, double neural_acc,
                             double gated_acc, double fidelity);

#endif /* NN_HOLO_GATE_H */
