#include "nn_holo_runner.h"
#include "nn_holographic_ranker.h"
#include "nn_core.h"
#include "nn_train.h"
#include "leos_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static double ns(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec*1e9+ts.tv_nsec; }


HoloRunner *holo_runner_create(size_t dim) {
    HoloRunner *hr = calloc(1, sizeof(HoloRunner));
    if (!hr) return NULL;
    hr->ranker = holo_ranker_create(dim);
    hr->eng = leos_engine_create(dim);
    hr->dim = dim;
    if (!hr->ranker || !hr->eng) { holo_runner_destroy(hr); return NULL; }
    return hr;
}

void holo_runner_destroy(HoloRunner *hr) {
    if (!hr) return;
    holo_ranker_destroy(hr->ranker);
    leos_engine_destroy(hr->eng);
    free(hr);
}

void holo_runner_train(HoloRunner *hr, const double *s, const double *a) {
    holo_ranker_store(hr->ranker, s, a); hr->train_count++;
}

void holo_runner_eval(HoloRunner *hr, const double *st, const double *av,
                      size_t nc, double *sc, size_t tl) {
    double t0 = ns();
    holo_ranker_score(hr->ranker, st, av, nc, sc);
    hr->total_time_ns += ns()-t0;
    double mx=sc[0]; for(size_t i=1;i<nc;i++)if(sc[i]>mx)mx=sc[i];
    double se=0; for(size_t i=0;i<nc;i++)se+=exp(sc[i]-mx);
    double pr=exp(sc[tl]-mx)/se; hr->sum_nll+=-log(pr+1e-10);
    size_t b=0; for(size_t i=1;i<nc;i++)if(sc[i]>sc[b])b=i;
    if(b==tl)hr->correct++;
    hr->test_count++;
}

double holo_runner_accuracy(const HoloRunner *hr) {
    return hr->test_count?(double)hr->correct/hr->test_count:0;
}
double holo_runner_nll(const HoloRunner *hr) {
    return hr->test_count?hr->sum_nll/hr->test_count:0;
}

void holo_runner_neural_baseline(HoloRunner *hr, const double *tf, const int *tl,
                                 const double *ef, const int *el,
                                 size_t nt, size_t ne, size_t nf, size_t nc) {
    const size_t L[]={nf,16,8,1};
    NNConfig cfg={L,4,42,NN_ACTIVATION_RELU,NN_ACTIVATION_LINEAR,NN_LOSS_MSE};
    NeuralNet*nn=nn_create_config(&cfg); if(!nn)return;
    for(size_t ep=0;ep<8;ep++)for(size_t s=0;s<nt;s++)
        nn_train_softmax_scorer(nn,(nn_real*)&tf[s*nc*nf],nc,(size_t)tl[s],0.015);
    size_t cor=0;double sn=0;
    for(size_t s=0;s<ne;s++){nn_real sc[64];nn_score_action_rows(nn,(nn_real*)&ef[s*nc*nf],nc,sc,NULL);
        nn_real mx=sc[0];for(size_t i=1;i<nc;i++)if(sc[i]>mx)mx=sc[i];
        nn_real se=0;for(size_t i=0;i<nc;i++)se+=exp(sc[i]-mx);
        nn_real pr=exp(sc[el[s]]-mx)/se;sn+=-log((double)pr+1e-10);
        size_t b=0;for(size_t i=1;i<nc;i++)if(sc[i]>sc[b])b=i;if(b==(size_t)el[s])cor++;}
    hr->neural_acc=(double)cor/ne;hr->neural_nll=sn/ne;hr->neural_ran=1;
    nn_free(nn);
}

void holo_runner_print_compare(const HoloRunner *hr) {
    double ha=holo_runner_accuracy(hr);
    printf("  Holo: %.1f%% NLL=%.3f | Neural: %.1f%% NLL=%.3f | Params: 0 vs 1185\n",
           ha*100,holo_runner_nll(hr),hr->neural_acc*100,hr->neural_nll);
}

int holo_runner_save(const HoloRunner *hr, const char *p) { return holo_ranker_save(hr->ranker,p); }
int holo_runner_load(HoloRunner *hr, const char *p)      { return holo_ranker_load(hr->ranker,p); }

void holo_runner_embed_state(HoloRunner *hr, uint64_t seed, double *out) { leos_engine_keygen(hr->eng,seed,out); }
size_t holo_runner_dim(const HoloRunner *hr) { return hr->dim; }
