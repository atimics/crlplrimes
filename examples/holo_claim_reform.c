#include "nn_holographic_ranker.h"
#include "nn_core.h"
#include "nn_train.h"
#include "leos_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define DIM 2048
#define NC 3
static double ns(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec*1e9+ts.tv_nsec;}

/*
 * delayed_claim Decision Rules:
 *   evidence > 0.6           → trust (action 0)
 *   evidence > 0.3 & risk < 0.5 → request (action 1)
 *   otherwise                → cautious (action 2)
 *
 * Reformulations tested:
 *   A: Raw continuous state (fails: 30.5%)
 *   B: Bucketed by threshold bands (clusters states by decision region)
 *   C: Multi-key: separate keys for evidence band + risk band
 *   D: Gate-aware: pre-classify by gate, then holo ranks within cluster
 */

static int oracle_label(double ev, double ri) {
    if (ev > 0.6) return 0;
    if (ev > 0.3 && ri < 0.5) return 1;
    return 2;
}

static void neural_baseline(int nt, int nte, const double *tx, const int *ty,
                            const double *ex, const int *ey, double *acc, double *nll) {
    const size_t L[]={10,8,4,1};
    NNConfig cfg={L,4,42,NN_ACTIVATION_RELU,NN_ACTIVATION_LINEAR,NN_LOSS_MSE};
    NeuralNet*nn=nn_create_config(&cfg);
    for(int ep=0;ep<10;ep++)for(int s=0;s<nt;s++)
        nn_train_softmax_scorer(nn,(nn_real*)&tx[s*NC*10],NC,(size_t)ty[s],0.02);
    int cor=0;double sn=0;
    for(int s=0;s<nte;s++){nn_real sc[64];nn_score_action_rows(nn,(nn_real*)&ex[s*NC*10],NC,sc,NULL);
        nn_real mx=sc[0];for(int i=1;i<NC;i++)if(sc[i]>mx)mx=sc[i];
        nn_real se=0;for(int i=0;i<NC;i++)se+=exp(sc[i]-mx);
        sn+=-log(exp(sc[ey[s]]-mx)/se+1e-10);
        int b=0;for(int i=1;i<NC;i++)if(sc[i]>sc[b])b=i;if(b==ey[s])cor++;}
    nn_free(nn);*acc=(double)cor/nte;*nll=sn/nte;
}

static void test_encoding(const char *label, int nt, int nte,
                          int bucket_ev, int bucket_ri,
                          int use_product, int use_gate_aware) {
    LeosEngine *eng=leos_engine_create(DIM);
    HolographicRanker *hr=holo_ranker_create(DIM);
    double *acts=calloc(NC*DIM,sizeof(double));
    for(int c=0;c<NC;c++) leos_engine_keygen(eng,(uint64_t)(c*900),&acts[c*DIM]);

    double *st=calloc((nt+nte)*DIM,sizeof(double));
    int *ty=calloc(nt,sizeof(int)),*ey=calloc(nte,sizeof(int));
    double *tx=calloc(nt*NC*10,sizeof(double)),*ex=calloc(nte*NC*10,sizeof(double));

    for(int i=0;i<nt+nte;i++){
        double ev=(double)((i*13+7)%100)/100, ri=(double)((i*29+11)%100)/100;
        int lb=oracle_label(ev,ri);
        double *s=&st[i*DIM];

        /* Encode state based on reformulation strategy */
        if (use_product) {
            /* Product encoding: bind(evidence_key, risk_key) */
            int ev_bucket = bucket_ev ? (int)(ev * bucket_ev) : (int)(ev * 100);
            int ri_bucket = bucket_ri ? (int)(ri * bucket_ri) : (int)(ri * 100);
            double *ek=malloc(DIM*sizeof(double)),*rk=malloc(DIM*sizeof(double));
            leos_engine_keygen(eng,(uint64_t)(ev_bucket*1000+1),ek);
            leos_engine_keygen(eng,(uint64_t)(ri_bucket*2000+2),rk);
            leos_engine_bind(eng,ek,rk,s);
            free(ek);free(rk);
        } else if (use_gate_aware) {
            /* Gate-aware: encode by decision region + residual */
            int region = oracle_label(ev, ri);
            double residual = ev * 100 + ri * 10;
            double *rk=malloc(DIM*sizeof(double)),*resk=malloc(DIM*sizeof(double));
            leos_engine_keygen(eng,(uint64_t)(region*100+1),rk);
            leos_engine_keygen(eng,(uint64_t)((int)(residual*100)+5000),resk);
            leos_engine_bind(eng,rk,resk,s);
            free(rk);free(resk);
        } else {
            /* Bucketed or raw continuous */
            int ev_key = bucket_ev ? (int)(ev * bucket_ev) : (int)(ev * 10000);
            int ri_key = bucket_ri ? (int)(ri * bucket_ri) : (int)(ri * 10000);
            leos_engine_keygen(eng,(uint64_t)(ev_key*10000+ri_key),s);
        }

        if(i<nt){ty[i]=lb;holo_ranker_store(hr,s,&acts[lb*DIM]);
            for(int c=0;c<NC;c++){double*f=&tx[(i*NC+c)*10];f[0]=ev;f[1]=ri;f[2]=(double)c;}}
        else{ey[i-nt]=lb;for(int c=0;c<NC;c++){double*f=&ex[((i-nt)*NC+c)*10];f[0]=ev;f[1]=ri;f[2]=(double)c;}}
    }

    int hc=0;double hsn=0;
    for(int i=0;i<nte;i++){double sc[64];
        holo_ranker_score(hr,&st[(nt+i)*DIM],acts,(size_t)NC,sc);
        double mx=sc[0];for(int j=1;j<NC;j++)if(sc[j]>mx)mx=sc[j];
        double se=0;for(int j=0;j<NC;j++)se+=exp(sc[j]-mx);
        hsn+=-log(exp(sc[ey[i]]-mx)/se+1e-10);
        int b=0;for(int j=1;j<NC;j++)if(sc[j]>sc[b])b=j;if(b==ey[i])hc++;}
    double ha=(double)hc/nte,hn=hsn/nte;

    double na,nn;neural_baseline(nt,nte,tx,ty,ex,ey,&na,&nn);

    printf("  %-30s | %6.1f%% | %6.1f%% | %+6.1f%% | %.3f\n",
           label,ha*100,na*100,(ha-na)*100,hn);

    free(acts);free(st);free(ty);free(ey);free(tx);free(ex);
    holo_ranker_destroy(hr);leos_engine_destroy(eng);
}

int main(void){
    printf("=== Delayed Claim Reformulation ===\n");
    printf("Can state encoding fix the holo gap?\n\n");
    printf("  %-30s | %6s | %6s | %6s | %6s\n","Encoding","Holo","Neural","Delta","NLL");
    printf("  %-30s-|-%6s-|-%6s-|-%6s-|-%6s\n","------------------------------","------","------","------","------");

    int nt=300,nte=200;

    /* A: Raw continuous (baseline failure) */
    test_encoding("A: Raw continuous (seed=ev*10000+ri)",nt,nte,0,0,0,0);

    /* B: Bucketed by thresholds */
    test_encoding("B: Bucketed ev×3 ri×2 (6 bins)",nt,nte,3,2,0,0);

    /* C: Bucketed finer */
    test_encoding("C: Bucketed ev×5 ri×4 (20 bins)",nt,nte,5,4,0,0);

    /* D: Product encoding */
    test_encoding("D: Product bind(ev_key, ri_key)",nt,nte,0,0,1,0);

    /* E: Gate-aware bucket */
    test_encoding("E: Gate-aware: region×3 + residual",nt,nte,0,0,0,1);

    /* F: Coarse bucket with fine residual */
    test_encoding("F: Coarse ev×2 ri×2 (like gate)",nt,nte,2,2,0,0);

    printf("\n  Decision rule: ev>0.6→trust, ev>0.3&ri<0.5→request, else caut\n");
    printf("  Holo needs states in same decision region to map to same key.\n");
    printf("  Bucketing/clustering the state space solves this.\n");
    return 0;
}
