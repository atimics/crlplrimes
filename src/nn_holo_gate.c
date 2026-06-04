#include <stdio.h>
#include <time.h>
#include "nn_holo_runner.h"
#include "nn_core.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

/*
 * Confidence-Gated Router
 *
 * Blends holographic and neural scores based on holo fidelity.
 *
 *   α = fidelity³ = (1/√N)³
 *   blended_score[i] = α · holo_score[i] + (1−α) · neural_score[i]
 *
 * When N is small (strong holo recall): α ≈ 1.0 → trust holo.
 * When N is large (noisy holo recall): α ≈ 0.0 → trust neural.
 *
 * The cubic exponent makes the transition sharp:
 *   N=1:    α=1.000  → pure holo
 *   N=8:    α=0.044  → mostly neural
 *   N=100:  α=0.001  → pure neural
 *
 * This automatically routes grid-like tasks (local, few examples per region)
 * to holo, and claim-like tasks (logical, many examples) to neural.
 */

void holo_gate_blend(double holo_fidelity,
                     const double *holo_scores,
                     const nn_real *neural_scores,
                     size_t n_candidates,
                     double *blended_out) {
    double alpha = holo_fidelity * holo_fidelity * holo_fidelity;

    for (size_t i = 0; i < n_candidates; i++)
        blended_out[i] = alpha * holo_scores[i] + (1.0 - alpha) * (double)neural_scores[i];
}

/*
 * Gated evaluation: run both scorers, blend, measure accuracy.
 *
 * Returns accuracy of the gated system.
 */
double holo_gate_eval(HoloRunner *hr,
                      NeuralNet *neural_net,
                      const double *test_states,
                      const double *test_action_vecs,
                      const double *test_features,
                      const int    *test_labels,
                      size_t n_test, size_t n_candidates, size_t n_features) {
    double fidelity = 1.0 / sqrt((double)(hr->train_count > 0 ? hr->train_count : 1));
    size_t correct = 0;

    for (size_t s = 0; s < n_test; s++) {
        /* Holographic scores */
        double holo_sc[64];
        holo_runner_eval(hr,
                         &test_states[s * holo_runner_dim(hr)],
                         test_action_vecs,
                         n_candidates,
                         holo_sc,
                         (size_t)test_labels[s]);

        /* Neural scores */
        nn_real neural_sc[64];
        nn_score_action_rows(neural_net,
            (nn_real*)&test_features[s * n_candidates * n_features],
            n_candidates, neural_sc, NULL);

        /* Blend */
        double blended[64];
        holo_gate_blend(fidelity, holo_sc, neural_sc, n_candidates, blended);

        /* Top-1 */
        size_t best = 0;
        for (size_t i = 1; i < n_candidates; i++)
            if (blended[i] > blended[best]) best = i;
        if (best == (size_t)test_labels[s]) correct++;
    }

    return (double)correct / n_test;
}

/*
 * Print gated comparison table.
 */
void holo_gate_print_compare(const char *name,
                             double holo_acc, double neural_acc,
                             double gated_acc, double fidelity) {
    double alpha = fidelity * fidelity * fidelity;
    printf("  %-16s | %7.1f%% | %7.1f%% | %7.1f%% | %6.3f | %.0f%% holo\n",
           name, neural_acc*100, holo_acc*100, gated_acc*100,
           fidelity, alpha*100);
}
