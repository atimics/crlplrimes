#ifndef NN_HOLO_GATE_H
#define NN_HOLO_GATE_H

#include "nn_core.h"
#include "nn_holo_runner.h"

void holo_gate_blend(double holo_fidelity,
                     const double *holo_scores,
                     const nn_real *neural_scores,
                     size_t n_candidates,
                     double *blended_out);

#endif /* NN_HOLO_GATE_H */
