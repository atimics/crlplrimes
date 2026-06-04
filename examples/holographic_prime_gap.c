/*
 * Holographic Prime Gap — Side-by-Side Comparison
 *
 * Key insight: holographic memory works by CLUSTERING similar states.
 * Primes with the same residue pattern share a holographic key, so
 * reinforcement accumulates across the cluster. This is what makes
 * the memory actually useful.
 */
#include "nn_core.h"
#include "nn_train.h"
#include "nn_holographic.h"
#include "leos_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define DIM         2048
#define N_CAND        10
#define N_STATIC      14
#define N_HOLO         4
#define N_BASE        N_STATIC
#define N_AUG        (N_STATIC + N_HOLO)
#define N_TRAIN      400
#define N_TEST       200
#define N_MODULI       7
#define N_CLUSTERS    64

static const int small_mods[N_MODULI] = {2, 3, 5, 7, 11, 13, 17};

static int cluster_id(int prime) {
    int h = 0;
    for (int i = 0; i < N_MODULI; i++)
        h = h * 31 + (prime % small_mods[i]);
    return (h & 0x7FFFFFFF) % N_CLUSTERS;
}

static int correct_gap(int prime) {
    int sum = 0;
    for (int i = 0; i < N_MODULI; i++) sum += prime % small_mods[i];
    return sum % N_CAND;
}

static double run(int use_holo, const double *fbase, const double *faug,
                  const size_t *targets, size_t n_tr, size_t n_te,
                  double *tr_acc, double *te_acc) {
    size_t fc = use_holo ? N_AUG : N_BASE;
    const double *f = use_holo ? faug : fbase;
    const size_t L[] = {fc, 14, 7, 1};
    NNConfig cfg = {L, 4, (unsigned)(42+use_holo*99),
                    NN_ACTIVATION_RELU, NN_ACTIVATION_LINEAR, NN_LOSS_MSE};
    NeuralNet *net = nn_create_config(&cfg);
    for (size_t ep = 0; ep < 10; ep++)
        for (size_t s = 0; s < n_tr; s++)
            nn_train_softmax_scorer(net, (nn_real*)&f[s*N_CAND*fc], N_CAND, targets[s], 0.02);
    size_t tc=0, ec=0;
    for (size_t s = 0; s < n_tr; s++) {
        nn_real sc[N_CAND]; nn_score_action_rows(net, (nn_real*)&f[s*N_CAND*fc], N_CAND, sc, NULL);
        int t=0; for (int i=1;i<N_CAND;i++) if(sc[i]>sc[t]) t=i;
        if((size_t)t==targets[s]) tc++;
    }
    for (size_t s = 0; s < n_te; s++) {
        size_t idx=n_tr+s; nn_real sc[N_CAND];
        nn_score_action_rows(net, (nn_real*)&f[idx*N_CAND*fc], N_CAND, sc, NULL);
        int t=0; for (int i=1;i<N_CAND;i++) if(sc[i]>sc[t]) t=i;
        if((size_t)t==targets[idx]) ec++;
    }
    nn_free(net);
    *tr_acc=(double)tc/n_tr; *te_acc=(double)ec/n_te; return 0;
}

int main(void) {
    printf("╔══════════════════════════════════════════════╗\n");
    printf("║  Holographic Prime Gap — %d clusters       ║\n", N_CLUSTERS);
    printf("║  %d static + %d holo = %d features          ║\n", N_STATIC, N_HOLO, N_AUG);
    printf("╚══════════════════════════════════════════════╝\n\n");

    nn_holographic_init(DIM);
    LeosEngine *eng = leos_engine_create(DIM);
    size_t n = N_TRAIN + N_TEST;
    double *fb = calloc(n*N_CAND*N_BASE, sizeof(double));
    double *fa = calloc(n*N_CAND*N_AUG, sizeof(double));
    size_t *tg = calloc(n, sizeof(size_t));

    printf("Generating %zu states, %d candidates, %d clusters...\n", n, N_CAND, N_CLUSTERS);

    for (size_t s = 0; s < n; s++) {
        int prime = 100 + (int)(s * 3 + 7);
        size_t corr = (size_t)correct_gap(prime);
        tg[s] = corr;
        int cid = cluster_id(prime);

        /* State: cluster embedding (all primes in same cluster = same key) */
        double *state = malloc(DIM*sizeof(double));
        leos_engine_keygen(eng, (uint64_t)cid, state);

        /* Correct action embedding */
        double *corr_act = malloc(DIM*sizeof(double));
        leos_engine_keygen(eng, (uint64_t)(cid * 100 + (int)corr), corr_act);

        /* Candidate embeddings: one per gap value */
        double *cand[N_CAND];
        for (int c = 0; c < N_CAND; c++) {
            cand[c] = malloc(DIM*sizeof(double));
            leos_engine_keygen(eng, (uint64_t)(cid * 100 + c), cand[c]);
        }

        for (int c = 0; c < N_CAND; c++) {
            double *b = &fb[s*N_CAND*N_BASE + c*N_BASE];
            double *a = &fa[s*N_CAND*N_AUG   + c*N_AUG];

            for (int m = 0; m < N_MODULI; m++)
                b[m] = (double)(prime % small_mods[m]) + ((double)((s*7+c)%100)/100.0-0.5)*0.6;
            for (int m = 0; m < N_MODULI; m++) b[N_MODULI + m] = (double)((prime / small_mods[m]) % 10);
            b[7] = log((double)prime);
            memcpy(a, b, N_STATIC*sizeof(double));
            nn_holographic_extract_features(state, cand[c], &a[N_STATIC]);
        }

        if (s < N_TRAIN) {
            nn_holographic_reinforce(state, corr_act);
            nn_holographic_store_rule(state, corr_act);
        }
        free(state); free(corr_act);
        for (int c=0;c<N_CAND;c++) free(cand[c]);
    }

    printf("\n  Run | BaseTr | BaseTe | HoloTr | HoloTe | Delta\n");
    printf("  ----|--------|--------|--------|--------|------\n");
    double sb=0,sh=0,sbt=0,sht=0;
    for (int r=0;r<5;r++) {
        double btr,bte,htr,hte;
        run(0,fb,fa,tg,N_TRAIN,N_TEST,&btr,&bte);
        run(1,fb,fa,tg,N_TRAIN,N_TEST,&htr,&hte);
        printf("  %3d | %5.1f%% | %5.1f%% | %5.1f%% | %5.1f%% | %+5.1f%%\n",
               r+1,btr*100,bte*100,htr*100,hte*100,(hte-bte)*100);
        sb+=bte; sh+=hte; sbt+=btr; sht+=htr;
    }
    printf("  ----|--------|--------|--------|--------|------\n");
    printf("  avg | %5.1f%% | %5.1f%% | %5.1f%% | %5.1f%% | %+5.1f%%\n",
           sbt/5*100,sb/5*100,sht/5*100,sh/5*100,(sh-sb)/5*100);
    printf("\n  chance: %.1f%%\n",100.0/N_CAND);
    nn_holographic_stats();

    free(fb); free(fa); free(tg);
    leos_engine_destroy(eng); nn_holographic_shutdown();
    return 0;
}
