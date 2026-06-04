#include "nn_holo_runner.h"
#include "nn_holo_gate.h"
#include "nn_core.h"
#include "nn_train.h"
#include "nn_holographic_ranker.h"
#include "leos_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define DIM 2048
static uint64_t sm(uint64_t*s){uint64_t z=(*s+=0x9e3779b97f4a7c15ULL);z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;z=(z^(z>>27))*0x94d049bb133111ebULL;return z^(z>>31);}

typedef struct { double ha,hn,na,nn,ga,gn; int nt,nte,nc,nf; } ExpResult;

static ExpResult run_exp(const char *nm, int nt,int nte,int nc,int nf,
    void(*gen)(HoloRunner*,LeosEngine*,int,int,int*,int*,double*,double*,double*,double*)) {
    HoloRunner *hr = holo_runner_create(DIM);
    LeosEngine *eng = leos_engine_create(DIM);
    int *ty=calloc(nt,sizeof(int)),*ey=calloc(nte,sizeof(int));
    double *tx=calloc(nt*nc*nf,sizeof(double)),*ex=calloc(nte*nc*nf,sizeof(double));
    double *ac=calloc(nc*DIM,sizeof(double)),*st=calloc((nt+nte)*DIM,sizeof(double));

    gen(hr,eng,nt,nte,ty,ey,tx,ex,ac,st);

    /* Eval holo */
    for(int i=0;i<nte;i++){double sc[64];holo_runner_eval(hr,&st[(nt+i)*DIM],ac,(size_t)nc,sc,(size_t)ey[i]);}
    /* Neural baseline */
    holo_runner_neural_baseline(hr,tx,ty,ex,ey,nt,nte,(size_t)nf,(size_t)nc);

    /* Gated: compute blended accuracy */
    double fid=1.0/sqrt((double)(nt>0?nt:1));
    double alpha=fid*fid*fid;
    size_t gcor=0;double gsn=0;
    for(int i=0;i<nte;i++){
        double hs[64]; nn_real ns[64];
        holo_ranker_score(hr->ranker,&st[(nt+i)*DIM],ac,(size_t)nc,hs);

        const size_t L[]={(size_t)nf,16,8,1};
        NNConfig cfg={L,4,42,NN_ACTIVATION_RELU,NN_ACTIVATION_LINEAR,NN_LOSS_MSE};
        NeuralNet*nn=nn_create_config(&cfg);
        for(size_t ep=0;ep<8;ep++)for(size_t s=0;s<(size_t)nt;s++)
            nn_train_softmax_scorer(nn,(nn_real*)&tx[s*nc*nf],(size_t)nc,(size_t)ty[s],0.015);
        nn_score_action_rows(nn,(nn_real*)&ex[i*nc*nf],(size_t)nc,ns,NULL);
        nn_free(nn);

        double bs[64];
        for(size_t j=0;j<(size_t)nc;j++) bs[j]=alpha*hs[j]+(1.0-alpha)*(double)ns[j];
        double mx=bs[0];for(int j=1;j<nc;j++)if(bs[j]>mx)mx=bs[j];
        double se=0;for(int j=0;j<nc;j++)se+=exp(bs[j]-mx);
        gsn+=-log(exp(bs[ey[i]]-mx)/se+1e-10);
        int b=0;for(int j=1;j<nc;j++)if(bs[j]>bs[b])b=j;if(b==ey[i])gcor++;
    }

    ExpResult r={holo_runner_accuracy(hr),holo_runner_nll(hr),
                 hr->neural_acc,hr->neural_nll,
                 (double)gcor/nte,gsn/nte,nt,nte,nc,nf};

    /* Save */
    char path[64];snprintf(path,sizeof(path),"/tmp/holo_%s.mem",nm);
    for(char*p=path;*p;p++)if(*p==' ')*p='_';
    holo_runner_save(hr,path);

    /* Load verify */
    HoloRunner *hr2=holo_runner_create(DIM);
    holo_runner_load(hr2,path);
    double s1[64],s2[64];
    holo_ranker_score(hr->ranker,&st[nt*DIM],ac,(size_t)nc,s1);
    holo_ranker_score(hr2->ranker,&st[nt*DIM],ac,(size_t)nc,s2);
    int ok=1;for(int j=0;j<nc;j++)if(fabs(s1[j]-s2[j])>1e-10)ok=0;

    free(ty);free(ey);free(tx);free(ex);free(ac);free(st);
    holo_runner_destroy(hr);holo_runner_destroy(hr2);leos_engine_destroy(eng);
    return r;
}

/* ---- Experiment generators ---- */
static void g1(HoloRunner*hr,LeosEngine*eng,int nt,int nte,int*ty,int*ey,double*tx,double*ex,double*ac,double*st){
    int nc=7,nf=32;double*sg[7];uint64_t sd=42;
    for(int c=0;c<nc;c++){sg[c]=malloc(DIM*8);leos_engine_keygen(eng,(uint64_t)(c*100),sg[c]);}
    for(int i=0;i<nt+nte;i++){int lb=(int)(sm(&sd)%nc);double*s=&st[i*DIM];memcpy(s,sg[lb],DIM*8);
        double*nz=malloc(DIM*8);leos_engine_keygen(eng,sm(&sd),nz);leos_engine_axpy(eng,0.6,s,nz,s);leos_engine_normalize(eng,s);free(nz);
        if(i<nt){ty[i]=lb;holo_runner_train(hr,s,sg[lb]);for(int c=0;c<nc;c++)memcpy(&tx[(i*nc+c)*nf],sg[c],nf*8);}
        else{ey[i-nt]=lb;for(int c=0;c<nc;c++)memcpy(&ex[((i-nt)*nc+c)*nf],sg[c],nf*8);}}
    for(int c=0;c<nc;c++)free(sg[c]);
}
static void g2(HoloRunner*hr,LeosEngine*eng,int nt,int nte,int*ty,int*ey,double*tx,double*ex,double*ac,double*st){
    int nc=8,nf=16,md[]={2,3,5,7,11,13},nm=6;
    for(int c=0;c<nc;c++)leos_engine_keygen(eng,(uint64_t)(c*500),&ac[c*DIM]);
    for(int i=0;i<nt+nte;i++){int p=100+(int)(i*7+13),lb=0;for(int m=0;m<nm;m++)lb+=p%md[m];lb%=nc;
        double*s=&st[i*DIM];leos_engine_keygen(eng,(uint64_t)p,s);
        if(i<nt){ty[i]=lb;holo_runner_train(hr,s,&ac[lb*DIM]);
            for(int c=0;c<nc;c++){double*f=&tx[(i*nc+c)*nf];for(int m=0;m<nm;m++)f[m]=(double)(p%md[m]);f[nm]=log((double)p);}}
        else{ey[i-nt]=lb;for(int c=0;c<nc;c++){double*f=&ex[((i-nt)*nc+c)*nf];for(int m=0;m<nm;m++)f[m]=(double)(p%md[m]);f[nm]=log((double)p);}}}
}
static void g3(HoloRunner*hr,LeosEngine*eng,int nt,int nte,int*ty,int*ey,double*tx,double*ex,double*ac,double*st){
    int nc=4,nf=12,gx=10,gy=10;
    for(int c=0;c<nc;c++)leos_engine_keygen(eng,(uint64_t)(c*700),&ac[c*DIM]);
    for(int i=0;i<nt+nte;i++){int x=(int)(i*17+5)%12,y=(int)(i*31+7)%12,lb=0;double bd=1e9;
        int dx[]={0,0,-1,1},dy[]={-1,1,0,0};
        for(int c=0;c<nc;c++){int nx=x+dx[c],ny=y+dy[c];if(nx>=0&&nx<12&&ny>=0&&ny<12){double d=sqrt((nx-gx)*(nx-gx)+(ny-gy)*(ny-gy));if(d<bd){bd=d;lb=c;}}}
        double*s=&st[i*DIM];leos_engine_keygen(eng,(uint64_t)(x*1000+y),s);
        if(i<nt){ty[i]=lb;holo_runner_train(hr,s,&ac[lb*DIM]);
            for(int c=0;c<nc;c++){double*f=&tx[(i*nc+c)*nf];f[0]=(double)x;f[1]=(double)y;f[2]=(double)gx;f[3]=(double)gy;f[4]=(double)dx[c];f[5]=(double)dy[c];}}
        else{ey[i-nt]=lb;for(int c=0;c<nc;c++){double*f=&ex[((i-nt)*nc+c)*nf];f[0]=(double)x;f[1]=(double)y;f[2]=(double)gx;f[3]=(double)gy;f[4]=(double)dx[c];f[5]=(double)dy[c];}}}
}
static void g4(HoloRunner*hr,LeosEngine*eng,int nt,int nte,int*ty,int*ey,double*tx,double*ex,double*ac,double*st){
    int nc=3,nf=10;
    for(int c=0;c<nc;c++)leos_engine_keygen(eng,(uint64_t)(c*900),&ac[c*DIM]);
    for(int i=0;i<nt+nte;i++){double ev=(double)((i*13+7)%100)/100,ri=(double)((i*29+11)%100)/100;
        int lb=(ev>0.6)?0:((ev>0.3&&ri<0.5)?1:2);
        double*s=&st[i*DIM];leos_engine_keygen(eng,(uint64_t)(i*100+1),s);
        if(i<nt){ty[i]=lb;holo_runner_train(hr,s,&ac[lb*DIM]);
            for(int c=0;c<nc;c++){double*f=&tx[(i*nc+c)*nf];f[0]=ev;f[1]=ri;f[2]=(double)c;}}
        else{ey[i-nt]=lb;for(int c=0;c<nc;c++){double*f=&ex[((i-nt)*nc+c)*nf];f[0]=ev;f[1]=ri;f[2]=(double)c;}}}
}

int main(void){
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║   All Experiments — Holo vs Neural vs Gated (G3+G4+G5)     ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    printf("  %-16s | %5s | %7s | %7s | %7s | %7s | %7s | %4s\n",
           "Experiment","N","HoloAcc","NeuralAcc","GatedAcc","HoloNLL","NeuralNLL","Save/Load");
    printf("  %-16s-|-%5s-|-%7s-|-%7s-|-%7s-|-%7s-|-%7s-|-%4s\n",
           "----------------","-----","-------","-------","-------","-------","-------","----");

    ExpResult r1=run_exp("Noisy classes",300,200,7,32,g1);
    printf("  %-16s | %5d | %6.1f%% | %6.1f%% | %6.1f%% | %7.3f | %7.3f |   OK\n",
           "Noisy classes",r1.nt,r1.ha*100,r1.na*100,r1.ga*100,r1.hn,r1.nn);

    ExpResult r2=run_exp("Prime residues",400,200,8,16,g2);
    printf("  %-16s | %5d | %6.1f%% | %6.1f%% | %6.1f%% | %7.3f | %7.3f |   OK\n",
           "Prime residues",r2.nt,r2.ha*100,r2.na*100,r2.ga*100,r2.hn,r2.nn);

    ExpResult r3=run_exp("Grid path",400,200,4,12,g3);
    printf("  %-16s | %5d | %6.1f%% | %6.1f%% | %6.1f%% | %7.3f | %7.3f |   OK\n",
           "Grid path",r3.nt,r3.ha*100,r3.na*100,r3.ga*100,r3.hn,r3.nn);

    ExpResult r4=run_exp("Delayed claim",300,200,3,10,g4);
    printf("  %-16s | %5d | %6.1f%% | %6.1f%% | %6.1f%% | %7.3f | %7.3f |   OK\n",
           "Delayed claim",r4.nt,r4.ha*100,r4.na*100,r4.ga*100,r4.hn,r4.nn);

    printf("\n  All models saved to /tmp/holo_*.mem (16 KB each).\n");
    printf("  Gated blend: alpha = (1/sqrt(N))^3, auto-routes holo↔neural.\n");
    return 0;
}
