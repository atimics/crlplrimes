#include "nn_holographic_ranker.h"
#include "nn_holo_runner.h"
#include "nn_core.h"
#include "nn_train.h"
#include "leos_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#define DIM 2048
static uint64_t sm(uint64_t*s){uint64_t z=(*s+=0x9e3779b97f4a7c15ULL);z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;z=(z^(z>>27))*0x94d049bb133111ebULL;return z^(z>>31);}

static void universal_features(HolographicRanker *r, const double *st, const double *ac, int nc, double *f){
    double sc[32];holo_ranker_score(r,st,ac,(size_t)nc,sc);
    f[0]=sc[0];f[1]=sc[1%nc];f[2]=sc[2%nc];f[3]=sc[nc>3?3:0];
}

typedef struct {char n[32];double ha,na,hfa;int nc,nt;}R;
static R run(const char*nm,int nt,int nte,int nc,int nf,void(*g)(HoloRunner*,LeosEngine*,int,int,int*,int*,double*,double*,double*,double*)){
    HoloRunner*hr=holo_runner_create(DIM);LeosEngine*eng=leos_engine_create(DIM);
    int*ty=calloc(nt,sizeof(int)),*ey=calloc(nte,sizeof(int));
    double*tx=calloc(nt*nc*nf,sizeof(double)),*ex=calloc(nte*nc*nf,sizeof(double));
    double*ac=calloc(nc*DIM,sizeof(double)),*st=calloc((nt+nte)*DIM,sizeof(double));
    g(hr,eng,nt,nte,ty,ey,tx,ex,ac,st);
    for(int i=0;i<nte;i++){double sc[64];holo_runner_eval(hr,&st[(nt+i)*DIM],ac,(size_t)nc,sc,(size_t)ey[i]);}
    double ha=holo_runner_accuracy(hr);
    holo_runner_neural_baseline(hr,tx,ty,ex,ey,nt,nte,(size_t)nf,(size_t)nc);
    double na=hr->neural_acc;

    #define NHF 4
    double*htr=calloc(nt*nc*NHF,sizeof(double)),*hte=calloc(nte*nc*NHF,sizeof(double));
    for(int s=0;s<nt;s++)for(int c=0;c<nc;c++)universal_features(hr->ranker,&st[s*DIM],ac,(size_t)nc,&htr[(s*nc+c)*NHF]);
    for(int s=0;s<nte;s++)for(int c=0;c<nc;c++)universal_features(hr->ranker,&st[(nt+s)*DIM],ac,(size_t)nc,&hte[(s*nc+c)*NHF]);
    const size_t L[]={NHF,8,4,1};NNConfig cfg={L,4,42,NN_ACTIVATION_RELU,NN_ACTIVATION_LINEAR,NN_LOSS_MSE};
    NeuralNet*nn=nn_create_config(&cfg);
    for(int ep=0;ep<10;ep++)for(int s=0;s<nt;s++)nn_train_softmax_scorer(nn,(nn_real*)&htr[s*nc*NHF],(size_t)nc,(size_t)ty[s],0.02);
    int cor=0;for(int s=0;s<nte;s++){nn_real sc[64];nn_score_action_rows(nn,(nn_real*)&hte[s*nc*NHF],(size_t)nc,sc,NULL);
        int b=0;for(int i=1;i<nc;i++)if(sc[i]>sc[b])b=i;if(b==ey[s])cor++;}
    double hfa=(double)cor/nte;nn_free(nn);
    R r={0};snprintf(r.n,sizeof(r.n),"%s",nm);r.ha=ha;r.na=na;r.hfa=hfa;r.nc=nc;r.nt=nt;
    free(ty);free(ey);free(tx);free(ex);free(ac);free(st);free(htr);free(hte);
    holo_runner_destroy(hr);leos_engine_destroy(eng);return r;
}

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
    printf("=== Universal Holographic Feature Envelope ===\n");
    printf("Can 4 holo features replace domain-specific features?\n\n");
    printf("  %-16s | %4s | %7s | %7s | %7s\n","Task","Cand","HoloNN","DomainNN","HoloFeat");
    printf("  %-16s-|-%4s-|-%7s-|-%7s-|-%7s\n","----------------","----","-------","-------","-------");

    R r1=run("Noisy classes",300,200,7,32,g1);
    printf("  %-16s | %4d | %6.1f%% | %6.1f%% | %6.1f%%\n","Noisy classes",r1.nc,r1.ha*100,r1.na*100,r1.hfa*100);
    R r2=run("Prime residues",400,200,8,16,g2);
    printf("  %-16s | %4d | %6.1f%% | %6.1f%% | %6.1f%%\n","Prime residues",r2.nc,r2.ha*100,r2.na*100,r2.hfa*100);
    R r3=run("Grid path",400,200,4,12,g3);
    printf("  %-16s | %4d | %6.1f%% | %6.1f%% | %6.1f%%\n","Grid path",r3.nc,r3.ha*100,r3.na*100,r3.hfa*100);
    R r4=run("Delayed claim",300,200,3,10,g4);
    printf("  %-16s | %4d | %6.1f%% | %6.1f%% | %6.1f%%\n","Delayed claim",r4.nc,r4.ha*100,r4.na*100,r4.hfa*100);

    printf("\n  HoloNN = k-NN, 0 params. DomainNN = MLP with domain features.\n");
    printf("  HoloFeat = MLP with 4 universal holographic features.\n");
    return 0;
}
