/*
 * Hybrid Holographic-Neural Architecture
 *
 * Three variants tested side-by-side:
 *
 *   A. Holographic k-NN: unbind(M, query) → dot(candidates) → rank
 *   B. Neural scorer: MLP(96→16→8→1) trained with SGD
 *   C. Holographic neurons: each neuron is a mini holographic store.
 *      N neurons, each stores bind(input_fragment, output_component).
 *      Forward = normalize(sum over neurons of neuron.recall(input))
 *
 * The question: does making each neuron a tiny hologram beat both?
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

#define DIM   2048
#define NHOLO    8   /* number of holographic neurons */

static uint64_t sm(uint64_t*s){uint64_t z=(*s+=0x9e3779b97f4a7c15ULL);z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;z=(z^(z>>27))*0x94d049bb133111ebULL;return z^(z>>31);}

/* ---- Holographic Neuron Array ----
 * Each neuron has its own memory trace.
 * Forward: recall from each neuron, bundle results.
 * Train: store correction in each neuron. */
typedef struct {
    HolographicRanker **neurons;
    size_t n;
    size_t dim;
    double *scratch;
} HoloNeuronArray;

static HoloNeuronArray *hna_create(size_t n, size_t dim) {
    HoloNeuronArray *a = calloc(1, sizeof(HoloNeuronArray));
    a->neurons = calloc(n, sizeof(HolographicRanker*));
    a->n = n; a->dim = dim;
    a->scratch = calloc(dim, sizeof(double));
    for (size_t i = 0; i < n; i++) a->neurons[i] = holo_ranker_create(dim);
    return a;
}
static void hna_destroy(HoloNeuronArray *a) {
    if(!a)return;
    for(size_t i=0;i<a->n;i++) holo_ranker_destroy(a->neurons[i]);
    free(a->neurons); free(a->scratch); free(a);
}
static void hna_train(HoloNeuronArray *a, const double *inp, const double *tgt) {
    /* Each neuron stores bind(fragment_of_input, fragment_of_target).
     * Neurons specialize via hashing: neuron i uses features from offset i. */
    for (size_t i = 0; i < a->n; i++) {
        /* Extract a "view" of input and target by shifting */
        double *vi = malloc(a->dim * sizeof(double));
        double *vt = malloc(a->dim * sizeof(double));
        LeosEngine *eng = leos_engine_create(a->dim);
        leos_engine_keygen(eng, (uint64_t)((uintptr_t)inp + i*1000), vi);
        leos_engine_keygen(eng, (uint64_t)((uintptr_t)tgt + i*2000), vt);
        holo_ranker_store(a->neurons[i], vi, vt);
        leos_engine_destroy(eng);
        free(vi); free(vt);
    }
}
static void hna_score(HoloNeuronArray *a, const double *inp, const double *acts,
                      size_t nc, double *scores) {
    LeosEngine *eng = leos_engine_create(a->dim);
    memset(a->scratch, 0, a->dim * sizeof(double));

    /* Recall from each neuron and bundle */
    for (size_t i = 0; i < a->n; i++) {
        double *vi = malloc(a->dim * sizeof(double));
        leos_engine_keygen(eng, (uint64_t)((uintptr_t)inp + i*1000), vi);
        double *rc = malloc(a->dim * sizeof(double));
        holo_ranker_score(a->neurons[i], vi, acts, nc, rc);
        /* rc now has scores per candidate. Average them. */
        for (size_t c = 0; c < nc; c++) scores[c] += rc[c];
        free(vi); free(rc);
    }
    /* Normalize scores */
    for (size_t c = 0; c < nc; c++) scores[c] /= (double)a->n;
    leos_engine_destroy(eng);
}

/* ---- Neural scorer baseline ---- */
static void neural_score(int nf, int nc, const double *tx, const int *ty,
                         int nt, double *acc, double *nll) {
    const size_t L[] = {(size_t)nf, 16, 8, 1};
    NNConfig cfg = {L, 4, 42, NN_ACTIVATION_RELU, NN_ACTIVATION_LINEAR, NN_LOSS_MSE};
    NeuralNet *nn = nn_create_config(&cfg);
    for (int ep = 0; ep < 8; ep++)
        for (int s = 0; s < nt; s++)
            nn_train_softmax_scorer(nn, (nn_real*)&tx[s*nc*nf], (size_t)nc, (size_t)ty[s], 0.015);
    int cor = 0; double sn = 0;
    for (int s = 0; s < nt; s++) {
        nn_real sc[32];
        nn_score_action_rows(nn, (nn_real*)&tx[s*nc*nf], (size_t)nc, sc, NULL);
        nn_real mx = sc[0]; for (int i=1;i<nc;i++) if(sc[i]>mx) mx=sc[i];
        nn_real se=0; for(int i=0;i<nc;i++) se+=exp(sc[i]-mx);
        nn_real pr = exp(sc[ty[s]]-mx)/se; sn += -log((double)pr+1e-10);
        int b=0; for(int i=1;i<nc;i++) if(sc[i]>sc[b]) b=i; if(b==ty[s]) cor++;
    }
    nn_free(nn);
    *acc = (double)cor/nt; *nll = sn/nt;
}

/* ---- Holographic ranker baseline ---- */
static void holo_score(HolographicRanker *hr, const double *st, const double *acts,
                       int nc, const int *ty, int nt, double *acc, double *nll) {
    int cor = 0; double sn = 0;
    LeosEngine *eng = leos_engine_create(DIM);
    for (int s = 0; s < nt; s++) {
        double sc[32];
        holo_ranker_score(hr, &st[s*DIM], acts, (size_t)nc, sc);
        double mx=sc[0]; for(int i=1;i<nc;i++) if(sc[i]>mx) mx=sc[i];
        double se=0; for(int i=0;i<nc;i++) se+=exp(sc[i]-mx);
        double pr=exp(sc[ty[s]]-mx)/se; sn+=-log(pr+1e-10);
        int b=0; for(int i=1;i<nc;i++) if(sc[i]>sc[b]) b=i; if(b==ty[s]) cor++;
    }
    leos_engine_destroy(eng);
    *acc = (double)cor/nt; *nll = sn/nt;
}

/* ---- Run one experiment ---- */
typedef struct { double nn_a, nn_n, ho_a, ho_n, hy_a, hy_n; } Result;

static Result run_one(int nt, int nte, int nc, int nf,
                      void(*gen)(HolographicRanker*,HoloNeuronArray*,
                                 LeosEngine*,int,int,int*,int*,double*,double*,double*,double*)) {
    HolographicRanker *hr = holo_ranker_create(DIM);
    HoloNeuronArray *hna = hna_create(NHOLO, DIM);
    LeosEngine *eng = leos_engine_create(DIM);
    int *ty=calloc(nt,sizeof(int)),*ey=calloc(nte,sizeof(int));
    double *tx=calloc(nt*nc*nf,sizeof(double)),*ex=calloc(nte*nc*nf,sizeof(double));
    double *ac=calloc(nc*DIM,sizeof(double)),*st=calloc((nt+nte)*DIM,sizeof(double));

    gen(hr, hna, eng, nt, nte, ty, ey, tx, ex, ac, st);

    Result r;
    neural_score(nf, nc, ex, ey, nte, &r.nn_a, &r.nn_n);
    holo_score(hr, &st[nt*DIM], ac, nc, ey, nte, &r.ho_a, &r.ho_n);

    /* Hybrid: holographic neurons score test states */
    {
        int cor=0; double sn=0;
        for(int s=0;s<nte;s++){
            double sc[32]={0};
            hna_score(hna, &st[(nt+s)*DIM], ac, (size_t)nc, sc);
            double mx=sc[0];for(int i=1;i<nc;i++)if(sc[i]>mx)mx=sc[i];
            double se=0;for(int i=0;i<nc;i++)se+=exp(sc[i]-mx);
            double pr=exp(sc[ey[s]]-mx)/se;sn+=-log(pr+1e-10);
            int b=0;for(int i=1;i<nc;i++)if(sc[i]>sc[b])b=i;if(b==ey[s])cor++;
        }
        r.hy_a=(double)cor/nte;r.hy_n=sn/nte;
    }

    free(ty);free(ey);free(tx);free(ex);free(ac);free(st);
    holo_ranker_destroy(hr);hna_destroy(hna);leos_engine_destroy(eng);
    return r;
}

/* ---- Experiment generators ---- */
static void g1(HolographicRanker*hr,HoloNeuronArray*ha,LeosEngine*eng,int nt,int nte,int*ty,int*ey,double*tx,double*ex,double*ac,double*st){
    int nc=7,nf=32;double*sg[7];uint64_t sd=42;
    for(int c=0;c<nc;c++){sg[c]=malloc(DIM*8);leos_engine_keygen(eng,(uint64_t)(c*100),sg[c]);}
    for(int i=0;i<nt+nte;i++){int lb=(int)(sm(&sd)%nc);double*s=&st[i*DIM];memcpy(s,sg[lb],DIM*8);
        double*nz=malloc(DIM*8);leos_engine_keygen(eng,sm(&sd),nz);leos_engine_axpy(eng,0.6,s,nz,s);leos_engine_normalize(eng,s);free(nz);
        if(i<nt){ty[i]=lb;holo_ranker_store(hr,s,sg[lb]);hna_train(ha,s,sg[lb]);for(int c=0;c<nc;c++)memcpy(&tx[(i*nc+c)*nf],sg[c],nf*8);}
        else{ey[i-nt]=lb;for(int c=0;c<nc;c++)memcpy(&ex[((i-nt)*nc+c)*nf],sg[c],nf*8);}}
    for(int c=0;c<nc;c++)free(sg[c]);
}
static void g2(HolographicRanker*hr,HoloNeuronArray*ha,LeosEngine*eng,int nt,int nte,int*ty,int*ey,double*tx,double*ex,double*ac,double*st){
    int nc=8,nf=16,md[]={2,3,5,7,11,13},nm=6;
    for(int c=0;c<nc;c++)leos_engine_keygen(eng,(uint64_t)(c*500),&ac[c*DIM]);
    for(int i=0;i<nt+nte;i++){int p=100+(int)(i*7+13),lb=0;for(int m=0;m<nm;m++)lb+=p%md[m];lb%=nc;
        double*s=&st[i*DIM];leos_engine_keygen(eng,(uint64_t)p,s);
        if(i<nt){ty[i]=lb;holo_ranker_store(hr,s,&ac[lb*DIM]);hna_train(ha,s,&ac[lb*DIM]);
            for(int c=0;c<nc;c++){double*f=&tx[(i*nc+c)*nf];for(int m=0;m<nm;m++)f[m]=(double)(p%md[m]);f[nm]=log((double)p);}}
        else{ey[i-nt]=lb;for(int c=0;c<nc;c++){double*f=&ex[((i-nt)*nc+c)*nf];for(int m=0;m<nm;m++)f[m]=(double)(p%md[m]);f[nm]=log((double)p);}}}
}
static void g3(HolographicRanker*hr,HoloNeuronArray*ha,LeosEngine*eng,int nt,int nte,int*ty,int*ey,double*tx,double*ex,double*ac,double*st){
    int nc=4,nf=12,gx=10,gy=10;
    for(int c=0;c<nc;c++)leos_engine_keygen(eng,(uint64_t)(c*700),&ac[c*DIM]);
    for(int i=0;i<nt+nte;i++){int x=(int)(i*17+5)%12,y=(int)(i*31+7)%12,lb=0;double bd=1e9;
        int dx[]={0,0,-1,1},dy[]={-1,1,0,0};
        for(int c=0;c<nc;c++){int nx=x+dx[c],ny=y+dy[c];if(nx>=0&&nx<12&&ny>=0&&ny<12){double d=sqrt((nx-gx)*(nx-gx)+(ny-gy)*(ny-gy));if(d<bd){bd=d;lb=c;}}}
        double*s=&st[i*DIM];leos_engine_keygen(eng,(uint64_t)(x*1000+y),s);
        if(i<nt){ty[i]=lb;holo_ranker_store(hr,s,&ac[lb*DIM]);hna_train(ha,s,&ac[lb*DIM]);
            for(int c=0;c<nc;c++){double*f=&tx[(i*nc+c)*nf];f[0]=(double)x;f[1]=(double)y;f[2]=(double)gx;f[3]=(double)gy;f[4]=(double)dx[c];f[5]=(double)dy[c];}}
        else{ey[i-nt]=lb;for(int c=0;c<nc;c++){double*f=&ex[((i-nt)*nc+c)*nf];f[0]=(double)x;f[1]=(double)y;f[2]=(double)gx;f[3]=(double)gy;f[4]=(double)dx[c];f[5]=(double)dy[c];}}}
}
static void g4(HolographicRanker*hr,HoloNeuronArray*ha,LeosEngine*eng,int nt,int nte,int*ty,int*ey,double*tx,double*ex,double*ac,double*st){
    int nc=3,nf=10;
    for(int c=0;c<nc;c++)leos_engine_keygen(eng,(uint64_t)(c*900),&ac[c*DIM]);
    for(int i=0;i<nt+nte;i++){double ev=(double)((i*13+7)%100)/100,ri=(double)((i*29+11)%100)/100;
        int lb=(ev>0.6)?0:((ev>0.3&&ri<0.5)?1:2);
        double*s=&st[i*DIM];leos_engine_keygen(eng,(uint64_t)(i*100+1),s);
        if(i<nt){ty[i]=lb;holo_ranker_store(hr,s,&ac[lb*DIM]);hna_train(ha,s,&ac[lb*DIM]);
            for(int c=0;c<nc;c++){double*f=&tx[(i*nc+c)*nf];f[0]=ev;f[1]=ri;f[2]=(double)c;}}
        else{ey[i-nt]=lb;for(int c=0;c<nc;c++){double*f=&ex[((i-nt)*nc+c)*nf];f[0]=ev;f[1]=ri;f[2]=(double)c;}}}
}

int main(void){
    printf("╔══════════════════════════════════════════════════════╗\n");
    printf("║  Hybrid: Holographic Neurons vs Baselines           ║\n");
    printf("║  Each neuron = tiny holographic store (NHOLO=%d)    ║\n", NHOLO);
    printf("╚══════════════════════════════════════════════════════╝\n\n");

    printf("  %-16s | %7s | %7s | %7s | %7s\n",
           "Experiment", "Neural", "HoloNN", "Hybrid", "Best");
    printf("  %-16s-|-%7s-|-%7s-|-%7s-|-%7s\n",
           "----------------","-------","-------","-------","-------");

    Result r1 = run_one(300,200,7,32,g1);
    printf("  %-16s | %6.1f%% | %6.1f%% | %6.1f%% | %6.1f%%\n",
           "Noisy classes", r1.nn_a*100, r1.ho_a*100, r1.hy_a*100,
           fmax(fmax(r1.nn_a,r1.ho_a),r1.hy_a)*100);

    Result r2 = run_one(400,200,8,16,g2);
    printf("  %-16s | %6.1f%% | %6.1f%% | %6.1f%% | %6.1f%%\n",
           "Prime residues", r2.nn_a*100, r2.ho_a*100, r2.hy_a*100,
           fmax(fmax(r2.nn_a,r2.ho_a),r2.hy_a)*100);

    Result r3 = run_one(300,200,4,12,g3);
    printf("  %-16s | %6.1f%% | %6.1f%% | %6.1f%% | %6.1f%%\n",
           "Grid path", r3.nn_a*100, r3.ho_a*100, r3.hy_a*100,
           fmax(fmax(r3.nn_a,r3.ho_a),r3.hy_a)*100);

    Result r4 = run_one(300,200,3,10,g4);
    printf("  %-16s | %6.1f%% | %6.1f%% | %6.1f%% | %6.1f%%\n",
           "Delayed claim", r4.nn_a*100, r4.ho_a*100, r4.hy_a*100,
           fmax(fmax(r4.nn_a,r4.ho_a),r4.hy_a)*100);

    printf("\n  Architecture: %d neurons × tiny holographic stores.\n", NHOLO);
    printf("  Forward: recall from each neuron → bundle → score.\n");
    printf("  Train: store bind(input_view, target_view) per neuron.\n");
    return 0;
}
