#include "nn_holo_experiment.h"
#include "nn_core.h"
#include "nn_train.h"
#include "leos_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define DIM 2048
static uint64_t sm(uint64_t *s){uint64_t z=(*s+=0x9e3779b97f4a7c15ULL);z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;z=(z^(z>>27))*0x94d049bb133111ebULL;return z^(z>>31);}

static void neural_score(int nf,int nc,const double*tx,const int*ty,int nt,double*acc,double*nll,size_t*pr){
    const size_t L[]={(size_t)nf,16,8,1};NNConfig cfg={L,4,42,NN_ACTIVATION_RELU,NN_ACTIVATION_LINEAR,NN_LOSS_MSE};
    NeuralNet*nn=nn_create_config(&cfg);
    for(int ep=0;ep<8;ep++)for(int s=0;s<nt;s++)nn_train_softmax_scorer(nn,(nn_real*)&tx[s*nc*nf],(size_t)nc,(size_t)ty[s],0.015);
    int cor=0;double sn=0;
    for(int s=0;s<nt;s++){nn_real sc[32];nn_score_action_rows(nn,(nn_real*)&tx[s*nc*nf],(size_t)nc,sc,NULL);
        nn_real mx=sc[0];for(int i=1;i<nc;i++)if(sc[i]>mx)mx=sc[i];nn_real se=0;for(int i=0;i<nc;i++)se+=exp(sc[i]-mx);
        nn_real pr=exp(sc[ty[s]]-mx)/se;sn+=-log((double)pr+1e-10);int b=0;for(int i=1;i<nc;i++)if(sc[i]>sc[b])b=i;if(b==ty[s])cor++;}
    nn_free(nn);*acc=(double)cor/nt;*nll=sn/nt;*pr=0;for(int l=0;l<3;l++)*pr+=L[l]*L[l+1]+L[l+1];
}

static void run_exp(const char*nm,int nf,int nc,int nt,int nte,
                    void(*gen)(HoloExperiment*,LeosEngine*,int,int,int*,int*,double*,double*,double*,double*)){
    HoloExperiment*hx=holo_experiment_create(DIM);LeosEngine*eng=leos_engine_create(DIM);
    int*ty=calloc(nt,sizeof(int)),*ey=calloc(nte,sizeof(int));
    double*tx=calloc(nt*nc*nf,sizeof(double)),*ex=calloc(nte*nc*nf,sizeof(double));
    double*ac=calloc(nc*DIM,sizeof(double)),*st=calloc((nt+nte)*DIM,sizeof(double));
    gen(hx,eng,nt,nte,ty,ey,tx,ex,ac,st);
    double na,nn;size_t np;neural_score(nf,nc,ex,ey,nte,&na,&nn,&np);
    double ha=holo_experiment_accuracy(hx),hn=holo_experiment_nll(hx),hl=holo_experiment_avg_latency_us(hx);
    printf("  %-20s | %6.1f%% | %6.1f%% | %+6.1f%% | %6.3f | %6.3f | %5.0f us\n",nm,na*100,ha*100,(ha-na)*100,nn,hn,hl);
    free(ty);free(ey);free(tx);free(ex);free(ac);free(st);holo_experiment_destroy(hx);leos_engine_destroy(eng);
}

/* Exp 1: Noisy classification */
static void e1(HoloExperiment*hx,LeosEngine*eng,int nt,int nte,int*ty,int*ey,double*tx,double*ex,double*ac,double*st){
    int nc=7,nf=32;double*sg[7];uint64_t sd=42;
    for(int c=0;c<nc;c++){sg[c]=malloc(DIM*8);leos_engine_keygen(eng,(uint64_t)(c*100),sg[c]);}
    for(int i=0;i<nt+nte;i++){int lb=(int)(sm(&sd)%nc);double*s=&st[i*DIM];memcpy(s,sg[lb],DIM*8);
        double*nz=malloc(DIM*8);leos_engine_keygen(eng,sm(&sd),nz);leos_engine_axpy(eng,0.6,s,nz,s);leos_engine_normalize(eng,s);free(nz);
        if(i<nt){ty[i]=lb;holo_experiment_train(hx,s,sg[lb]);for(int c=0;c<nc;c++)memcpy(&tx[(i*nc+c)*nf],sg[c],nf*8);}
        else{ey[i-nt]=lb;for(int c=0;c<nc;c++)memcpy(&ex[((i-nt)*nc+c)*nf],sg[c],nf*8);}}
    for(int i=0;i<nte;i++){double sc[32];holo_experiment_score(hx,&st[(nt+i)*DIM],sg[0],(size_t)nc,sc,(size_t)ey[i]);}
    for(int c=0;c<nc;c++)free(sg[c]);
}

/* Exp 2: Prime residues */
static void e2(HoloExperiment*hx,LeosEngine*eng,int nt,int nte,int*ty,int*ey,double*tx,double*ex,double*ac,double*st){
    int nc=8,nf=16,md[]={2,3,5,7,11,13},nm=6;
    for(int c=0;c<nc;c++)leos_engine_keygen(eng,(uint64_t)(c*500),&ac[c*DIM]);
    for(int i=0;i<nt+nte;i++){int p=100+(int)(i*7+13),lb=0;for(int m=0;m<nm;m++)lb+=p%md[m];lb%=nc;
        double*s=&st[i*DIM];leos_engine_keygen(eng,(uint64_t)p,s);
        if(i<nt){ty[i]=lb;holo_experiment_train(hx,s,&ac[lb*DIM]);
            for(int c=0;c<nc;c++){double*f=&tx[(i*nc+c)*nf];for(int m=0;m<nm;m++)f[m]=(double)(p%md[m]);f[nm]=log((double)p);}}
        else{ey[i-nt]=lb;for(int c=0;c<nc;c++){double*f=&ex[((i-nt)*nc+c)*nf];for(int m=0;m<nm;m++)f[m]=(double)(p%md[m]);f[nm]=log((double)p);}}}
    for(int i=0;i<nte;i++){double sc[32];holo_experiment_score(hx,&st[(nt+i)*DIM],ac,(size_t)nc,sc,(size_t)ey[i]);}
}

/* Exp 3: Grid path */
static void e3(HoloExperiment*hx,LeosEngine*eng,int nt,int nte,int*ty,int*ey,double*tx,double*ex,double*ac,double*st){
    int nc=4,nf=12,gx=10,gy=10;
    for(int c=0;c<nc;c++)leos_engine_keygen(eng,(uint64_t)(c*700),&ac[c*DIM]);
    for(int i=0;i<nt+nte;i++){int x=(int)(i*17+5)%12,y=(int)(i*31+7)%12,lb=0;double bd=1e9;
        int dx[]={0,0,-1,1},dy[]={-1,1,0,0};
        for(int c=0;c<nc;c++){int nx=x+dx[c],ny=y+dy[c];if(nx>=0&&nx<12&&ny>=0&&ny<12){double d=sqrt((nx-gx)*(nx-gx)+(ny-gy)*(ny-gy));if(d<bd){bd=d;lb=c;}}}
        double*s=&st[i*DIM];leos_engine_keygen(eng,(uint64_t)(x*1000+y),s);
        if(i<nt){ty[i]=lb;holo_experiment_train(hx,s,&ac[lb*DIM]);
            for(int c=0;c<nc;c++){double*f=&tx[(i*nc+c)*nf];f[0]=(double)x;f[1]=(double)y;f[2]=(double)gx;f[3]=(double)gy;f[4]=(double)dx[c];f[5]=(double)dy[c];}}
        else{ey[i-nt]=lb;for(int c=0;c<nc;c++){double*f=&ex[((i-nt)*nc+c)*nf];f[0]=(double)x;f[1]=(double)y;f[2]=(double)gx;f[3]=(double)gy;f[4]=(double)dx[c];f[5]=(double)dy[c];}}}
    for(int i=0;i<nte;i++){double sc[32];holo_experiment_score(hx,&st[(nt+i)*DIM],ac,(size_t)nc,sc,(size_t)ey[i]);}
}

/* Exp 4: Delayed claim */
static void e4(HoloExperiment*hx,LeosEngine*eng,int nt,int nte,int*ty,int*ey,double*tx,double*ex,double*ac,double*st){
    int nc=3,nf=10;
    for(int c=0;c<nc;c++)leos_engine_keygen(eng,(uint64_t)(c*900),&ac[c*DIM]);
    for(int i=0;i<nt+nte;i++){double ev=(double)((i*13+7)%100)/100,ri=(double)((i*29+11)%100)/100;
        int lb=(ev>0.6)?0:((ev>0.3&&ri<0.5)?1:2);
        double*s=&st[i*DIM];leos_engine_keygen(eng,(uint64_t)(i*100+1),s);
        if(i<nt){ty[i]=lb;holo_experiment_train(hx,s,&ac[lb*DIM]);
            for(int c=0;c<nc;c++){double*f=&tx[(i*nc+c)*nf];f[0]=ev;f[1]=ri;f[2]=(double)c;}}
        else{ey[i-nt]=lb;for(int c=0;c<nc;c++){double*f=&ex[((i-nt)*nc+c)*nf];f[0]=ev;f[1]=ri;f[2]=(double)c;}}}
    for(int i=0;i<nte;i++){double sc[32];holo_experiment_score(hx,&st[(nt+i)*DIM],ac,(size_t)nc,sc,(size_t)ey[i]);}
}

int main(void){
    printf("╔══════════════════════════════════════════════════════╗\n");
    printf("║   Holographic Ranker — All Experiments              ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n\n");
    printf("  %-20s | %6s | %6s | %6s | %6s | %6s | %7s\n","Experiment","NN Acc","HoloAcc","Delta","NN NLL","HoloNLL","Latency");
    printf("  %-20s-|-%6s-|-%6s-|-%6s-|-%6s-|-%6s-|-%7s\n","--------------------","------","------","------","------","------","-------");
    run_exp("Noisy classes",32,7,300,200,e1);
    run_exp("Prime residues",16,8,400,200,e2);
    run_exp("Grid path",12,4,300,200,e3);
    run_exp("Delayed claim",10,3,300,200,e4);
    printf("\n  Zero-parameter holographic ranker, dim=%d.\n",DIM);
    return 0;
}
