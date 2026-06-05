#include "nn_holo_gate.h"
#include <math.h>

void holo_gate_blend(double holo_fidelity,
                     const double *holo_scores,
                     const nn_real *neural_scores,
                     size_t n_candidates,
                     double *blended_out) {
    double alpha = holo_fidelity * 3.0;  /* amplified: trust holo more at moderate fidelity */
    if (alpha > 1.0) alpha = 1.0;
    for (size_t i = 0; i < n_candidates; i++)
        blended_out[i] = alpha * holo_scores[i] + (1.0 - alpha) * (double)neural_scores[i];
}
