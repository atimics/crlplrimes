#include "leos_engine.h"
#include "nn_holographic.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define DIM 2048

int main(void) {
    printf("=== Holographic Integration Smoke Test ===\n\n");

    /* Init */
    if (nn_holographic_init(DIM) != 0) {
        printf("FAIL: nn_holographic_init\n");
        return 1;
    }
    printf("Init OK (dim=%d)\n", DIM);

    /* Store 3 rules */
    double task[3][DIM], rule[3][DIM];
    for (int i = 0; i < 3; i++) {
        /* Seed-based keygen: we need the engine, but we can use the bridge to recall.
         * For the test, just feed zeros — the bridge generates keys internally. */
        /* Actually, the bridge's store_rule takes pre-generated vectors.
         * We need the engine to generate them. Let's use a raw engine for this. */
    }

    /* Better approach: use raw engine for vector generation */
    LeosEngine *eng = leos_engine_create(DIM);
    if (!eng) { printf("FAIL: engine\n"); return 1; }

    double *ts = malloc(DIM * sizeof(double));
    double *rv = malloc(DIM * sizeof(double));
    double *ctx = malloc(DIM * sizeof(double));
    double *feat = malloc(4 * sizeof(double));

    for (int i = 0; i < 3; i++) {
        leos_engine_keygen(eng, (uint64_t)(i * 2 + 1), ts);
        leos_engine_keygen(eng, (uint64_t)(i * 2 + 2), rv);
        nn_holographic_store_rule(ts, rv);
    }
    printf("Stored 3 rules (%zu total)\n", nn_holographic_rule_count());

    /* Extract features */
    leos_engine_keygen(eng, 100, ts);  /* state */
    leos_engine_keygen(eng, 101, rv);  /* candidate */
    nn_holographic_extract_features(ts, rv, feat);

    printf("\nFeatures for novel state:\n");
    printf("  f_context:    %+.4f\n", feat[0]);
    printf("  f_recent:     %+.4f\n", feat[1]);
    printf("  f_candidate:  %+.4f\n", feat[2]);
    printf("  f_confidence: %+.4f\n", feat[3]);

    /* Reinforce 10 times */
    for (int i = 0; i < 10; i++)
        nn_holographic_reinforce(ts, rv);
    printf("\nReinforced 10x (%zu total)\n", nn_holographic_reinforce_count());

    /* Extract features again */
    nn_holographic_extract_features(ts, rv, feat);
    printf("Features after reinforcement:\n");
    printf("  f_context:    %+.4f\n", feat[0]);
    printf("  f_recent:     %+.4f\n", feat[1]);
    printf("  f_candidate:  %+.4f\n", feat[2]);
    printf("  f_confidence: %+.4f\n", feat[3]);

    /* Stats */
    nn_holographic_stats();

    /* Shutdown */
    nn_holographic_shutdown();
    leos_engine_destroy(eng);
    free(ts); free(rv); free(ctx); free(feat);

    printf("\n=== PASS ===\n");
    return 0;
}
