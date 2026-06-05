#include "memory_neuron.h"
#include "nn_core.h"
#include "nn_train.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define DIM 2048
#define N_FEAT 8
#define N_CAND 4
#define N_TRAIN 1200
#define N_TEST 400

static uint64_t sm(uint64_t *s) {
    uint64_t z = (*s += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

int main(void) {
    printf("=== Memory Neuron Grid Demo ===\n");
    printf("Grid: 4x4, K=4, slow tick every 4, %d train / %d test\n\n",
           N_TRAIN, N_TEST);

    MemoryNeuronGrid *grid = memory_neuron_grid_create(
        DIM, 4, 4, N_FEAT, 4, 4, 0.01);
    LeosEngine *eng = leos_engine_create(DIM);

    double *act_vecs = calloc(N_CAND * DIM, sizeof(double));
    for (int c = 0; c < N_CAND; c++)
        leos_engine_keygen(eng, (uint64_t)(c * 700), &act_vecs[c * DIM]);

    int gx = 10, gy = 10;
    int dx[] = {0, 0, -1, 1}, dy[] = {-1, 1, 0, 0};
    uint64_t sd = 42;

    int nf_base = N_FEAT;
    int nf_mem  = N_FEAT + 4;
    double *base_train = calloc(N_TRAIN * N_CAND * nf_base, sizeof(double));
    double *mem_train  = calloc(N_TRAIN * N_CAND * nf_mem, sizeof(double));
    int *train_labels = calloc(N_TRAIN, sizeof(int));
    double *base_test = calloc(N_TEST * N_CAND * nf_base, sizeof(double));
    double *mem_test  = calloc(N_TEST * N_CAND * nf_mem, sizeof(double));
    int *test_labels = calloc(N_TEST, sizeof(int));

    for (int i = 0; i < N_TRAIN + N_TEST; i++) {
        int x = (int)(sm(&sd) % 12), y = (int)(sm(&sd) % 12);
        int lb = 0; double bd = 1e9;
        for (int c = 0; c < N_CAND; c++) {
            int nx = x + dx[c], ny = y + dy[c];
            if (nx >= 0 && nx < 12 && ny >= 0 && ny < 12) {
                double d = sqrt((nx-gx)*(nx-gx) + (ny-gy)*(ny-gy));
                if (d < bd) { bd = d; lb = c; }
            }
        }

        double sv[DIM], av[DIM];
        leos_engine_keygen(eng, (uint64_t)(x * 1000 + y), sv);
        leos_engine_normalize(eng, sv);
        memcpy(av, &act_vecs[lb * DIM], DIM * sizeof(double));

        double fv[N_FEAT];
        fv[0] = (double)x / 12.0;  fv[1] = (double)y / 12.0;
        fv[2] = (double)gx / 12.0; fv[3] = (double)gy / 12.0;
        fv[4] = (double)dx[0];     fv[5] = (double)dy[0];
        fv[6] = (double)dx[3];     fv[7] = (double)dy[3];

        int is_tr = (i < N_TRAIN);
        int idx = is_tr ? i : (i - N_TRAIN);
        double *bbuf = is_tr ? base_train : base_test;
        double *mbuf = is_tr ? mem_train  : mem_test;
        int *lbls = is_tr ? train_labels : test_labels;
        lbls[idx] = lb;

        for (int c = 0; c < N_CAND; c++) {
            double *b = &bbuf[(idx * N_CAND + c) * nf_base];
            b[0] = fv[0]; b[1] = fv[1]; b[2] = fv[2]; b[3] = fv[3];
            b[4] = (double)dx[c]; b[5] = (double)dy[c];
            b[6] = fv[6]; b[7] = fv[7];

            double *m = &mbuf[(idx * N_CAND + c) * nf_mem];
            memcpy(m, b, N_FEAT * sizeof(double));
            double cand_vec[DIM];
            memcpy(cand_vec, &act_vecs[c * DIM], DIM * sizeof(double));
            memory_neuron_grid_context(grid, sv, cand_vec, fv, &m[N_FEAT]);
        }

        if (is_tr)
            memory_neuron_grid_store(grid, sv, av, fv);
    }

    int bc = 0;
    {
        const size_t L[] = {(size_t)nf_base, 16, 8, 1};
        NNConfig cfg = {L, 4, 42, NN_ACTIVATION_RELU,
                        NN_ACTIVATION_LINEAR, NN_LOSS_MSE};
        NeuralNet *nn = nn_create_config(&cfg);
        for (int ep = 0; ep < 10; ep++)
            for (int s = 0; s < N_TRAIN; s++)
                nn_train_softmax_scorer(nn,
                    (nn_real*)&base_train[s*N_CAND*nf_base],
                    N_CAND, (size_t)train_labels[s], 0.02);
        for (int s = 0; s < N_TEST; s++) {
            nn_real sc[N_CAND];
            nn_score_action_rows(nn,
                (nn_real*)&base_test[s*N_CAND*nf_base],
                N_CAND, sc, NULL);
            int b = 0;
            for (int i = 1; i < N_CAND; i++)
                if (sc[i] > sc[b]) b = i;
            if (b == test_labels[s]) bc++;
        }
        nn_free(nn);
    }

    int mc = 0;
    {
        const size_t L[] = {(size_t)nf_mem, 16, 8, 1};
        NNConfig cfg = {L, 4, 43, NN_ACTIVATION_RELU,
                        NN_ACTIVATION_LINEAR, NN_LOSS_MSE};
        NeuralNet *nn = nn_create_config(&cfg);
        for (int ep = 0; ep < 10; ep++)
            for (int s = 0; s < N_TRAIN; s++)
                nn_train_softmax_scorer(nn,
                    (nn_real*)&mem_train[s*N_CAND*nf_mem],
                    N_CAND, (size_t)train_labels[s], 0.02);
        for (int s = 0; s < N_TEST; s++) {
            nn_real sc[N_CAND];
            nn_score_action_rows(nn,
                (nn_real*)&mem_test[s*N_CAND*nf_mem],
                N_CAND, sc, NULL);
            int b = 0;
            for (int i = 1; i < N_CAND; i++)
                if (sc[i] > sc[b]) b = i;
            if (b == test_labels[s]) mc++;
        }
        nn_free(nn);
    }

    double ba = (double)bc / N_TEST * 100;
    double ma = (double)mc / N_TEST * 100;

    printf("  %-30s | %8s | %7s\n", "Architecture", "Acc", "Delta");
    printf("  %-30s-|-%8s-|-%7s\n",
           "------------------------------", "--------", "-------");
    printf("  %-30s | %7.1f%% |\n",
           "Neural (8 domain features)", ba);
    printf("  %-30s | %7.1f%% | %+6.1f%%\n",
           "Neural + Memory (8 + 4 ctx)", ma, ma - ba);

    printf("\n  Grid: ticks=%zu  stored=%zu  avg_fid=%.4f\n",
           memory_neuron_grid_tick(grid),
           memory_neuron_grid_total_stored(grid),
           memory_neuron_grid_avg_fidelity(grid));

    size_t counts[16];
    memory_neuron_grid_stored_counts(grid, counts);
    printf("  Storage/neuron: ");
    for (int i = 0; i < 16; i++) printf("%zu ", counts[i]);
    printf("\n");

    free(act_vecs); free(base_train); free(mem_train);
    free(base_test); free(mem_test);
    free(train_labels); free(test_labels);
    memory_neuron_grid_destroy(grid);
    leos_engine_destroy(eng);
    return 0;
}
