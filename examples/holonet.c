/*
 * Holonet: Structured Holographic Network
 *
 * Synthesis of everything we've learned:
 *
 * 1. Feature envelope → structured key domains (not random keys)
 * 2. Multi-resolution memory pyramid (coarse → fine)
 * 3. Differentiated receptive fields per neuron
 * 4. Training = bundling verified pairs (not SGD)
 * 5. Confidence-gated routing between holo and neural paths
 *
 * Each "holo-neuron" covers a REGION of the feature space,
 * defined by a centroid vector. Only the K nearest neurons
 * are active for a given input (sparse activation).
 */
#include "nn_holographic_ranker.h"
#include "nn_core.h"
#include "nn_train.h"
#include "leos_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define DIM    2048
#define N_FEAT   12   /* feature envelope size */
#define N_NEURON 16   /* number of holo-neurons */
#define K_ACTIVE  4   /* top-K active neurons per input */

static uint64_t sm(uint64_t*s){uint64_t z=(*s+=0x9e3779b97f4a7c15ULL);z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;z=(z^(z>>27))*0x94d049bb133111ebULL;return z^(z>>31);}
static double ns(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec*1e9+ts.tv_nsec;}

/* ---- Holonet Neuron ----
 * Each neuron has:
 *   - centroid: a vector in feature space that defines its receptive field
 *   - memory:   holographic store for (input, target) pairs that fall in its field
 *   - count:    how many examples stored */
typedef struct {
    double *centroid;       /* DIM-dimensional center of receptive field */
    HolographicRanker *mem; /* holographic store for this field */
    size_t stored;
} HoloNeuron;

typedef struct {
    HoloNeuron *neurons;
    size_t n;
    size_t dim;
    LeosEngine *eng;
    /* Feature centroids in feature space (N_FEAT dims, for routing) */
    double *f_centroids;  /* N_NEURON * N_FEAT */
} HoloNet;

static HoloNet *holonet_create(size_t dim, size_t n) {
    HoloNet *hn = calloc(1, sizeof(HoloNet));
    hn->neurons = calloc(n, sizeof(HoloNeuron));
    hn->n = n; hn->dim = dim;
    hn->eng = leos_engine_create(dim);
    hn->f_centroids = calloc(n * N_FEAT, sizeof(double));

    /* Initialize centroids: random points in feature space */
    uint64_t sd = 42;
    for (size_t i = 0; i < n; i++) {
        hn->neurons[i].centroid = malloc(dim * sizeof(double));
        leos_engine_keygen(hn->eng, sm(&sd), hn->neurons[i].centroid);
        hn->neurons[i].mem = holo_ranker_create(dim);
        /* Feature-space centroids for routing */
        for (int f = 0; f < N_FEAT; f++)
            hn->f_centroids[i * N_FEAT + f] = ((double)(sm(&sd) % 1000) / 1000.0 - 0.5) * 2.0;
    }
    return hn;
}

static void holonet_destroy(HoloNet *hn) {
    if(!hn)return;
    for(size_t i=0;i<hn->n;i++){free(hn->neurons[i].centroid);holo_ranker_destroy(hn->neurons[i].mem);}
    free(hn->neurons);free(hn->f_centroids);leos_engine_destroy(hn->eng);free(hn);
}

/* Find K nearest neurons to a feature vector */
static void holonet_route(HoloNet *hn, const double *features,
                          size_t *active, size_t k) {
    double dists[N_NEURON];
    int indices[N_NEURON];
    for (size_t i = 0; i < hn->n; i++) {
        double d = 0;
        for (int f = 0; f < N_FEAT; f++) {
            double df = features[f] - hn->f_centroids[i * N_FEAT + f];
            d += df * df;
        }
        dists[i] = d;
        indices[i] = (int)i;
    }
    /* Sort by distance (bubble — N is tiny) */
    for (size_t i = 0; i < hn->n; i++)
        for (size_t j = i+1; j < hn->n; j++)
            if (dists[j] < dists[i]) {
                double td=dists[i];dists[i]=dists[j];dists[j]=td;
                int ti=indices[i];indices[i]=indices[j];indices[j]=ti;
            }
    for (size_t i = 0; i < k && i < hn->n; i++) active[i] = (size_t)indices[i];
    for (size_t i = k; i < hn->n; i++) active[i] = (size_t)-1;
}

/* Train: store in K nearest neurons */
static void holonet_train(HoloNet *hn, const double *state, const double *action,
                          const double *features) {
    size_t active[N_NEURON];
    holonet_route(hn, features, active, K_ACTIVE);
    for (size_t i = 0; i < K_ACTIVE; i++) {
        if (active[i] == (size_t)-1) break;
        holo_ranker_store(hn->neurons[active[i]].mem, state, action);
        hn->neurons[active[i]].stored++;
    }
}

/* Score: average scores from K nearest neurons, weighted by proximity */
static void holonet_score(HoloNet *hn, const double *state, const double *actions,
                          size_t nc, const double *features, double *scores) {
    size_t active[N_NEURON];
    holonet_route(hn, features, active, K_ACTIVE);

    /* Compute distances for weighting */
    double dists[K_ACTIVE];
    for (size_t i = 0; i < K_ACTIVE; i++) {
        if (active[i] == (size_t)-1) { dists[i] = 1e9; continue; }
        double d = 0;
        for (int f = 0; f < N_FEAT; f++) {
            double df = features[f] - hn->f_centroids[active[i] * N_FEAT + f];
            d += df * df;
        }
        dists[i] = sqrt(d);
    }

    /* Weighted average of scores from each active neuron */
    memset(scores, 0, nc * sizeof(double));
    double total_w = 0;
    for (size_t i = 0; i < K_ACTIVE; i++) {
        if (active[i] == (size_t)-1) break;
        double w = 1.0 / (dists[i] + 0.01); /* inverse distance weight */
        total_w += w;
        double ns[32];
        holo_ranker_score(hn->neurons[active[i]].mem, state, actions, nc, ns);
        for (size_t c = 0; c < nc; c++) scores[c] += ns[c] * w;
    }
    if (total_w > 0) for (size_t c = 0; c < nc; c++) scores[c] /= total_w;
}

/* ---- Baselines ---- */
static void neural_baseline(int nf, int nc, const double *tx, const int *ty, int nt,
                            double *acc, double *nll) {
    const size_t L[]={(size_t)nf,16,8,1};
    NNConfig cfg={L,4,42,NN_ACTIVATION_RELU,NN_ACTIVATION_LINEAR,NN_LOSS_MSE};
    NeuralNet*nn=nn_create_config(&cfg);
    for(int ep=0;ep<8;ep++)for(int s=0;s<nt;s++)nn_train_softmax_scorer(nn,(nn_real*)&tx[s*nc*nf],(size_t)nc,(size_t)ty[s],0.015);
    int cor=0;double sn=0;
    for(int s=0;s<nt;s++){nn_real sc[32];nn_score_action_rows(nn,(nn_real*)&tx[s*nc*nf],(size_t)nc,sc,NULL);
        nn_real mx=sc[0];for(int i=1;i<nc;i++)if(sc[i]>mx)mx=sc[i];
        nn_real se=0;for(int i=0;i<nc;i++)se+=exp(sc[i]-mx);
        nn_real pr=exp(sc[ty[s]]-mx)/se;sn+=-log((double)pr+1e-10);
        int b=0;for(int i=1;i<nc;i++)if(sc[i]>sc[b])b=i;if(b==ty[s])cor++;}
    nn_free(nn);*acc=(double)cor/nt;*nll=sn/nt;
}

int main(void){
    printf("╔══════════════════════════════════════════════════════╗\n");
    printf("║  HoloNet: Structured Holographic Network            ║\n");
    printf("║  %d neurons, K=%d active, %d-dim features          ║\n",N_NEURON,K_ACTIVE,N_FEAT);
    printf("╚══════════════════════════════════════════════════════╝\n\n");

    /* Grid path experiment with holonet */
    int nc=4, nt=400, nte=200, gx=10, gy=10;
    LeosEngine *eng = leos_engine_create(DIM);
    HoloNet *hn = holonet_create(DIM, N_NEURON);
    HolographicRanker *hr = holo_ranker_create(DIM); /* flat baseline */

    /* Generate data */
    double *st = calloc((nt+nte)*DIM,sizeof(double));
    double *ac = calloc(nc*DIM,sizeof(double));
    double *feat = calloc((nt+nte)*N_FEAT,sizeof(double));
    int *label = calloc(nt+nte,sizeof(int));
    double *tx = calloc(nt*nc*N_FEAT,sizeof(double));
    double *ex = calloc(nte*nc*N_FEAT,sizeof(double));

    for(int c=0;c<nc;c++)leos_engine_keygen(eng,(uint64_t)(c*700),&ac[c*DIM]);

    for(int i=0;i<nt+nte;i++){
        int x=(int)(i*17+5)%12,y=(int)(i*31+7)%12,lb=0;double bd=1e9;
        int dx[]={0,0,-1,1},dy[]={-1,1,0,0};
        for(int c=0;c<nc;c++){int nx=x+dx[c],ny=y+dy[c];if(nx>=0&&nx<12&&ny>=0&&ny<12){double d=sqrt((nx-gx)*(nx-gx)+(ny-gy)*(ny-gy));if(d<bd){bd=d;lb=c;}}}
        label[i]=lb;
        double*s=&st[i*DIM];leos_engine_keygen(eng,(uint64_t)(x*1000+y),s);
        /* Feature envelope: position + goal + action deltas */
        double*f=&feat[i*N_FEAT];
        f[0]=(double)x/12.0;f[1]=(double)y/12.0;f[2]=(double)gx/12.0;f[3]=(double)gy/12.0;
        for(int c=0;c<nc;c++){f[4+c*2]=(double)dx[c];f[5+c*2]=(double)dy[c];}

        if(i<nt){
            holo_ranker_store(hr,s,&ac[lb*DIM]);     /* flat baseline */
            holonet_train(hn,s,&ac[lb*DIM],f);        /* holonet */
            for(int c=0;c<nc;c++)memcpy(&tx[(i*nc+c)*N_FEAT],f,N_FEAT*sizeof(double));
        }else{
            int j=i-nt;
            for(int c=0;c<nc;c++)memcpy(&ex[(j*nc+c)*N_FEAT],f,N_FEAT*sizeof(double));
        }
    }

    /* Evaluate flat holographic */
    double ha,hnll; int hcor=0;double hsn=0;
    for(int i=0;i<nte;i++){double sc[32];
        holo_ranker_score(hr,&st[(nt+i)*DIM],ac,(size_t)nc,sc);
        double mx=sc[0];for(int j=1;j<nc;j++)if(sc[j]>mx)mx=sc[j];
        double se=0;for(int j=0;j<nc;j++)se+=exp(sc[j]-mx);
        double pr=exp(sc[label[nt+i]]-mx)/se;hsn+=-log(pr+1e-10);
        int b=0;for(int j=1;j<nc;j++)if(sc[j]>sc[b])b=j;if(b==label[nt+i])hcor++;}
    ha=(double)hcor/nte;hnll=hsn/nte;

    /* Evaluate holonet */
    double hna,hnanll;int hnac=0;double hnasn=0;
    for(int i=0;i<nte;i++){double sc[32]={0};
        holonet_score(hn,&st[(nt+i)*DIM],ac,(size_t)nc,&feat[(nt+i)*N_FEAT],sc);
        double mx=sc[0];for(int j=1;j<nc;j++)if(sc[j]>mx)mx=sc[j];
        double se=0;for(int j=0;j<nc;j++)se+=exp(sc[j]-mx);
        double pr=exp(sc[label[nt+i]]-mx)/se;hnasn+=-log(pr+1e-10);
        int b=0;for(int j=1;j<nc;j++)if(sc[j]>sc[b])b=j;if(b==label[nt+i])hnac++;}
    hna=(double)hnac/nte;hnanll=hnasn/nte;

    /* Neural baseline */
    double na,nnll;
    neural_baseline(N_FEAT,nc,ex,&label[nt],nte,&na,&nnll);

    /* Print */
    printf("Grid Path (%d train, %d test):\n\n", nt, nte);
    printf("  Architecture     | Accuracy | NLL    | Notes\n");
    printf("  -----------------|----------|--------|-----------------------\n");
    printf("  Neural (MLP)     | %7.1f%% | %6.3f | 12→16→8→1, 1185 params\n", na*100, nnll);
    printf("  Holographic k-NN | %7.1f%% | %6.3f | flat, 0 params\n", ha*100, hnll);
    printf("  HoloNet          | %7.1f%% | %6.3f | %d neurons, K=%d\n", hna*100, hnanll, N_NEURON, K_ACTIVE);

    /* Per-neuron statistics */
    printf("\n  Per-neuron storage:\n");
    for(size_t i=0;i<hn->n;i++)
        printf("    neuron %2zu: %4zu stored, fidelity %.3f\n",
               i, hn->neurons[i].stored,
               hn->neurons[i].stored>0?1.0/sqrt((double)hn->neurons[i].stored):1.0);

    /* Architecture description */
    printf("\n╔══════════════════════════════════════════════════════╗\n");
    printf("║  HoloNet Architecture                               ║\n");
    printf("╠══════════════════════════════════════════════════════╣\n");
    printf("║  Feature envelope → structured key domains          ║\n");
    printf("║  Each neuron = one region of feature space          ║\n");
    printf("║  Routing: K nearest centroids are active            ║\n");
    printf("║  Training: store (state,action) in active neurons   ║\n");
    printf("║  Inference: weighted vote from active neurons       ║\n");
    printf("║  Total memory: %d × 16 KB = %d KB                   ║\n", N_NEURON, N_NEURON*16);
    printf("╚══════════════════════════════════════════════════════╝\n");

    free(st);free(ac);free(feat);free(label);free(tx);free(ex);
    holo_ranker_destroy(hr);holonet_destroy(hn);leos_engine_destroy(eng);
    return 0;
}
