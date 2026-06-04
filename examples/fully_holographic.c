#include "leos_hnn.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define DIM    2048
#define N_TRAIN 200
#define N_TEST  100
#define N_CLASSES 5

static double ns(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec*1e9+ts.tv_nsec; }

static uint64_t fnv1a(const char *s) {
    uint64_t h = 14695981039346656037ULL;
    while (*s) { h ^= (uint64_t)(unsigned char)*s++; h *= 1099511628211ULL; }
    return h;
}

int main(void) {
    printf("=== Fully Holographic Neural Network ===\n");
    printf("No weight arrays — the model IS the memory\n\n");

    HolographicNet *net = hnn_create(DIM);
    LeosEngine *eng = hnn_engine(net);
    size_t mem_train = hnn_add_memory(net);
    size_t mem_layer = hnn_add_memory(net);
    size_t layer0 = hnn_add_layer_mixed(net, 8, 0.625, 0.125, 0.25);

    /* Generate class signatures */
    double *class_sigs[N_CLASSES];
    for (int c = 0; c < N_CLASSES; c++) {
        class_sigs[c] = malloc(DIM*sizeof(double));
        leos_engine_keygen(eng, (uint64_t)(c*100), class_sigs[c]);
    }

    /* Store training data */
    double **inputs = malloc(N_TRAIN * sizeof(double*));
    int *labels = malloc(N_TRAIN * sizeof(int));
    for (int i = 0; i < N_TRAIN; i++) {
        labels[i] = i % N_CLASSES;
        inputs[i] = malloc(DIM*sizeof(double));
        memcpy(inputs[i], class_sigs[labels[i]], DIM*sizeof(double));
        double *noise = malloc(DIM*sizeof(double));
        leos_engine_keygen(eng, (uint64_t)(i*1000+500), noise);
        leos_engine_axpy(eng, 0.3, inputs[i], noise, inputs[i]);
        leos_engine_normalize(eng, inputs[i]);
        free(noise);
        hnn_memory_store(net, mem_train, inputs[i], class_sigs[labels[i]]);
    }
    printf("Stored %d examples. Memory: %.0f KB\n\n", N_TRAIN, DIM*8.0/1024.0);

    /* Test: k-NN */
    int correct = 0;
    for (int t = 0; t < 50; t++) {
        int tl = t % N_CLASSES;
        double *ti = malloc(DIM*sizeof(double));
        memcpy(ti, class_sigs[tl], DIM*sizeof(double));
        double *nz = malloc(DIM*sizeof(double));
        leos_engine_keygen(eng, (uint64_t)(t*2000+700), nz);
        leos_engine_axpy(eng, 0.3, ti, nz, ti); leos_engine_normalize(eng, ti); free(nz);

        double *rc = malloc(DIM*sizeof(double));
        hnn_memory_recall(net, mem_train, ti, rc);
        int best=0; double bs=leos_engine_dot(eng, rc, class_sigs[0]);
        for (int c=1;c<N_CLASSES;c++) {
            double s=leos_engine_dot(eng, rc, class_sigs[c]);
            if (s>bs) { bs=s; best=c; }
        }
        if (best==tl) correct++;
        free(ti); free(rc);
    }
    printf("k-NN (holographic memory): %.1f%% (chance %.1f%%)\n\n",
           (double)correct/50*100, 100.0/N_CLASSES);

    /* Store layer weights holographically */
    for (int i = 0; i < 8; i++) {
        char key[32]; snprintf(key, sizeof(key), "neuron_%d", i);
        double *wk = malloc(DIM*sizeof(double));
        double *wv = malloc(DIM*sizeof(double));
        leos_engine_keygen(eng, fnv1a(key), wk);
        leos_engine_keygen(eng, (uint64_t)(i*300+100), wv);
        hnn_memory_store(net, mem_layer, wk, wv);
        free(wk); free(wv);
    }
    printf("Stored 8 neuron weights. Fidelity: %.3f\n\n", 1.0/sqrt(8.0));

    /* Fully holographic forward pass */
    correct = 0;
    double t0 = ns();
    for (int t = 0; t < N_TEST; t++) {
        int tl = t % N_CLASSES;
        double *ti = malloc(DIM*sizeof(double));
        memcpy(ti, class_sigs[tl], DIM*sizeof(double));
        double *nz = malloc(DIM*sizeof(double));
        leos_engine_keygen(eng, (uint64_t)(t*3000+900), nz);
        leos_engine_axpy(eng, 0.3, ti, nz, ti); leos_engine_normalize(eng, ti); free(nz);

        double *ctx = malloc(DIM*sizeof(double));
        hnn_memory_recall(net, mem_train, ti, ctx);
        double *lo = malloc(DIM*sizeof(double));
        hnn_layer_forward(net, layer0, ctx, lo);

        int best=0; double bs=leos_engine_dot(eng, lo, class_sigs[0]);
        for (int c=1;c<N_CLASSES;c++) {
            double s=leos_engine_dot(eng, lo, class_sigs[c]);
            if (s>bs) { bs=s; best=c; }
        }
        if (best==tl) correct++;
        free(ti); free(ctx); free(lo);
    }
    double dt = (ns()-t0)/N_TEST;
    printf("Fully holographic forward: %.1f%% (%d/%d) in %.0f us/forward\n",
           (double)correct/N_TEST*100, correct, N_TEST, dt/1000);

    /* Fading memory test */
    printf("\n--- Graceful forgetting ---\n");
    printf("Distractors | k-NN Acc | Theory\n");
    for (int nd = 0; nd <= 800; nd = nd==0 ? 50 : nd*2) {
        HolographicNet *nc = hnn_create(DIM); LeosEngine *ec = hnn_engine(nc);
        size_t mc = hnn_add_memory(nc);
        for (int i=0;i<N_TRAIN;i++) hnn_memory_store(nc,mc,inputs[i],class_sigs[labels[i]]);
        for (int d=0;d<nd;d++) {
            double *dk=malloc(DIM*sizeof(double)),*dv=malloc(DIM*sizeof(double));
            leos_engine_keygen(ec,(uint64_t)(d+10000),dk);
            leos_engine_keygen(ec,(uint64_t)(d+20000),dv);
            hnn_memory_store(nc,mc,dk,dv); free(dk);free(dv);
        }
        int cn=0;
        for(int t=0;t<30;t++){
            int tl=t%N_CLASSES;
            double *ti=malloc(DIM*sizeof(double));
            memcpy(ti,class_sigs[tl],DIM*sizeof(double));
            double *nz=malloc(DIM*sizeof(double));
            leos_engine_keygen(ec,(uint64_t)(t*999+1),nz);
            leos_engine_axpy(ec,0.3,ti,nz,ti);leos_engine_normalize(ec,ti);free(nz);
            double *rc=malloc(DIM*sizeof(double));
            hnn_memory_recall(nc,mc,ti,rc);
            int b=0;double bs=leos_engine_dot(ec,rc,class_sigs[0]);
            for(int c=1;c<N_CLASSES;c++){double s=leos_engine_dot(ec,rc,class_sigs[c]);if(s>bs){bs=s;b=c;}}
            if(b==tl)cn++; free(ti);free(rc);
        }
        int total=N_TRAIN+nd;
        printf("%11d | %7.1f%% | %.4f\n", nd, (double)cn/30*100, 1.0/sqrt((double)(total>0?total:1)));
        hnn_destroy(nc);
    }

    printf("\n=== Architecture ===\n");
    printf("Store:  M += bind(key, value)\n");
    printf("Recall: value_hat = unbind(M, key)\n");
    printf("Forward: recall -> layer -> classify\n");
    printf("Forget: add distractors -> fidelity 1/sqrt(N)\n");
    printf("Memory: %.0f KB fixed (dim=%d)\n", DIM*8.0/1024.0, DIM);

    for (int c=0;c<N_CLASSES;c++) free(class_sigs[c]);
    for (int i=0;i<N_TRAIN;i++) free(inputs[i]);
    free(inputs); free(labels);
    hnn_destroy(net);
    return 0;
}
