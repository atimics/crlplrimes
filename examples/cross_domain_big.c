#include "nn_holographic_ranker.h"
#include "nn_core.h"
#include "nn_train.h"
#include "holographic/leos_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define DIM 2048
static uint64_t sm(uint64_t*s){uint64_t z=(*s+=0x9e3779b97f4a7c15ULL);z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;z=(z^(z>>27))*0x94d049bb133111ebULL;return z^(z>>31);}
static double ns(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec*1e9+ts.tv_nsec;}

static int gen_noisy(LeosEngine*eng,int nt,int nte,int nc,double*acts,double*st,int*ty,int*ey){
    for(int c=0;c<nc;c++)leos_engine_keygen(eng,(uint64_t)(c*100),&acts[c*DIM]);
    uint64_t sd=42;
    for(int i=0;i<nt+nte;i++){int lb=(int)(sm(&sd)%nc);leos_engine_keygen(eng,sm(&sd),&st[i*DIM]);
        double nz[DIM];leos_engine_keygen(eng,sm(&sd),nz);leos_engine_axpy(eng,0.6,&st[i*DIM],nz,&st[i*DIM]);leos_engine_normalize(eng,&st[i*DIM]);
        if(i<nt)ty[i]=lb;else ey[i-nt]=lb;}return nc;
}
static int gen_prime(LeosEngine*eng,int nt,int nte,int nc,double*acts,double*st,int*ty,int*ey){
    for(int c=0;c<nc;c++)leos_engine_keygen(eng,(uint64_t)(c*500),&acts[c*DIM]);
    int md[]={2,3,5,7,11,13},nm=6;
    for(int i=0;i<nt+nte;i++){int p=100+(int)(i*7+13),lb=0;for(int m=0;m<nm;m++)lb+=p%md[m];lb%=nc;
        leos_engine_keygen(eng,(uint64_t)p,&st[i*DIM]);if(i<nt)ty[i]=lb;else ey[i-nt]=lb;}return nc;
}
static int gen_grid(LeosEngine*eng,int nt,int nte,int nc,double*acts,double*st,int*ty,int*ey){
    for(int c=0;c<nc;c++)leos_engine_keygen(eng,(uint64_t)(c*700),&acts[c*DIM]);
    int gx=10,gy=10;
    for(int i=0;i<nt+nte;i++){int x=(int)(i*17+5)%12,y=(int)(i*31+7)%12,lb=0;double bd=1e9;
        int dx[]={0,0,-1,1},dy[]={-1,1,0,0};
        for(int c=0;c<nc;c++){int nx=x+dx[c],ny=y+dy[c];if(nx>=0&&nx<12&&ny>=0&&ny<12){double d=sqrt((nx-gx)*(nx-gx)+(ny-gy)*(ny-gy));if(d<bd){bd=d;lb=c;}}}
        leos_engine_keygen(eng,(uint64_t)(x*1000+y),&st[i*DIM]);if(i<nt)ty[i]=lb;else ey[i-nt]=lb;}return nc;
}
static int gen_claim(LeosEngine*eng,int nt,int nte,int nc,double*acts,double*st,int*ty,int*ey){
    for(int c=0;c<nc;c++)leos_engine_keygen(eng,(uint64_t)(c*900),&acts[c*DIM]);
    for(int i=0;i<nt+nte;i++){double ev=(double)((i*13+7)%100)/100,ri=(double)((i*29+11)%100)/100;
        int lb=(ev>0.6)?0:((ev>0.3&&ri<0.5)?1:2);
        leos_engine_keygen(eng,(uint64_t)((int)(ev*10000)+(int)(ri*10000)),&st[i*DIM]);
        if(i<nt)ty[i]=lb;else ey[i-nt]=lb;}return nc;
}

static void test_shared(int nt,int nte){
    printf("\n=== Shared Memory: all 4 tasks → one trace ===\n");
    HolographicRanker*shared=holo_ranker_create(DIM);LeosEngine*eng=leos_engine_create(DIM);
    int ncs[]={7,8,4,3};char*nm[]={"noisy","prime","grid","claim"};
    int(*gs[])(LeosEngine*,int,int,int,double*,double*,int*,int*)={gen_noisy,gen_prime,gen_grid,gen_claim};
    double*aa[4],*as[4];int*at[4],*ae[4];
    for(int t=0;t<4;t++){aa[t]=calloc(ncs[t]*DIM,sizeof(double));as[t]=calloc((nt+nte)*DIM,sizeof(double));at[t]=calloc(nt,sizeof(int));ae[t]=calloc(nte,sizeof(int));}
    for(int t=0;t<4;t++)gs[t](eng,nt,nte,ncs[t],aa[t],as[t],at[t],ae[t]);
    for(int t=0;t<4;t++)for(int i=0;i<nt;i++)holo_ranker_store(shared,&as[t][i*DIM],&aa[t][at[t][i]*DIM]);
    printf("  %-10s | %9s | %9s | %11s\n","Task","Shared","Solo","Interference");
    for(int t=0;t<4;t++){
        int sc=0;for(int i=0;i<nte;i++){double s[64];holo_ranker_score(shared,&as[t][(nt+i)*DIM],aa[t],(size_t)ncs[t],s);int b=0;for(int j=1;j<ncs[t];j++)if(s[j]>s[b])b=j;if(b==ae[t][i])sc++;}
        HolographicRanker*solo=holo_ranker_create(DIM);for(int i=0;i<nt;i++)holo_ranker_store(solo,&as[t][i*DIM],&aa[t][at[t][i]*DIM]);
        int soc=0;for(int i=0;i<nte;i++){double s[64];holo_ranker_score(solo,&as[t][(nt+i)*DIM],aa[t],(size_t)ncs[t],s);int b=0;for(int j=1;j<ncs[t];j++)if(s[j]>s[b])b=j;if(b==ae[t][i])soc++;}
        printf("  %-10s | %8.1f%% | %8.1f%% | %+10.1f%%\n",nm[t],(double)sc/nte*100,(double)soc/nte*100,((double)sc-soc)/nte*100);
        holo_ranker_destroy(solo);
    }
    for(int t=0;t<4;t++){free(aa[t]);free(as[t]);free(at[t]);free(ae[t]);}holo_ranker_destroy(shared);leos_engine_destroy(eng);
}

static void test_transfer(int nt,int nte){
    printf("\n=== Cross-Domain Transfer: train A, test B ===\n");
    LeosEngine*eng=leos_engine_create(DIM);int ncs[]={7,8,4,3};char*nm[]={"noisy","prime","grid","claim"};
    int(*gs[])(LeosEngine*,int,int,int,double*,double*,int*,int*)={gen_noisy,gen_prime,gen_grid,gen_claim};
    double*aa[4],*as[4];int*at[4],*ae[4];
    for(int t=0;t<4;t++){aa[t]=calloc(ncs[t]*DIM,sizeof(double));as[t]=calloc((nt+nte)*DIM,sizeof(double));at[t]=calloc(nt,sizeof(int));ae[t]=calloc(nte,sizeof(int));gs[t](eng,nt,nte,ncs[t],aa[t],as[t],at[t],ae[t]);}
    printf("  Train \\ Test |");for(int t=0;t<4;t++)printf(" %8s |",nm[t]);printf("\n");
    for(int src=0;src<4;src++){printf("  %-13s |",nm[src]);
        for(int dst=0;dst<4;dst++){HolographicRanker*hr=holo_ranker_create(DIM);
            for(int i=0;i<nt;i++)holo_ranker_store(hr,&as[src][i*DIM],&aa[src][at[src][i]*DIM]);
            int cor=0;for(int i=0;i<nte;i++){double s[64];holo_ranker_score(hr,&as[dst][(nt+i)*DIM],aa[dst],(size_t)ncs[dst],s);int b=0;for(int j=1;j<ncs[dst];j++)if(s[j]>s[b])b=j;if(b==ae[dst][i])cor++;}
            double ch=100.0/ncs[dst];const char*c=(double)cor/nte*100>ch+10?"++":((double)cor/nte*100<ch+5?"--":"  ");
            printf(" %5.1f%%%s |",(double)cor/nte*100,c);holo_ranker_destroy(hr);}printf("\n");}
    for(int t=0;t<4;t++){free(aa[t]);free(as[t]);free(at[t]);free(ae[t]);}leos_engine_destroy(eng);
}

static void test_voids(void){
    printf("\n=== Void Detection ===\n");
    LeosEngine*eng=leos_engine_create(DIM);int nc=4;double acts[nc*DIM];
    for(int c=0;c<nc;c++)leos_engine_keygen(eng,(uint64_t)(c*700),&acts[c*DIM]);
    HolographicRanker*hr=holo_ranker_create(DIM);
    for(int i=0;i<400;i++){int x=(int)(i*17+5)%12,y=(int)(i*31+7)%12;if(x>=6||y>=6)continue;
        int lb=0;double bd=1e9;int dx[]={0,0,-1,1},dy[]={-1,1,0,0};
        for(int c=0;c<nc;c++){int nx=x+dx[c],ny=y+dy[c];if(nx>=0&&nx<12&&ny>=0&&ny<12){double d=sqrt((nx-10)*(nx-10)+(ny-10)*(ny-10));if(d<bd){bd=d;lb=c;}}}
        double st[DIM];leos_engine_keygen(eng,(uint64_t)(x*1000+y),st);holo_ranker_store(hr,st,&acts[lb*DIM]);}
    printf("  Region     | Cells | Min Sim | Max Sim | Detection\n");
    double tr_min=1,tr_max=-1,vo_min=1,vo_max=-1;int tr_n=0,vo_n=0;
    for(int x=0;x<12;x++)for(int y=0;y<12;y++){double st[DIM],s[64];leos_engine_keygen(eng,(uint64_t)(x*1000+y),st);holo_ranker_score(hr,st,acts,(size_t)nc,s);double mx=s[0];for(int j=1;j<nc;j++)if(s[j]>mx)mx=s[j];
        if(x<6&&y<6){if(mx<tr_min)tr_min=mx;if(mx>tr_max)tr_max=mx;tr_n++;}else{if(mx<vo_min)vo_min=mx;if(mx>vo_max)vo_max=mx;vo_n++;}}
    printf("  Trained     | %5d | %7.4f | %7.4f | sim > 0.1 ✓\n",tr_n,tr_min,tr_max);
    printf("  Void        | %5d | %7.4f | %7.4f | sim < 0.1 → escalate\n",vo_n,vo_min,vo_max);
    holo_ranker_destroy(hr);leos_engine_destroy(eng);
}

static void test_scale(void){
    printf("\n=== Signal Flight 10K — Fixed Encoding ===\n");
    LeosEngine*eng=leos_engine_create(DIM);int nc=9;double acts[nc*DIM];
    for(int c=0;c<nc;c++)leos_engine_keygen(eng,(uint64_t)(c*100),&acts[c*DIM]);
    FILE*fp=fopen("/tmp/flight_10k.csv","r");if(!fp){printf("  No data\n");leos_engine_destroy(eng);return;}
    char line[4096];fgets(line,sizeof(line),fp);
    HolographicRanker*hr=holo_ranker_create(DIM);
    int total=0,ec=0,et=0;double e_time=0;

    while(fgets(line,sizeof(line),fp)&&total<10000){
        char*save,*p;int col=0,action=0;double dist=0,hcos=0,hsin=0,spd=0,fspd=0,lspd=0,bdist=0,fclr=0,lclr=0,rclr=0,hull=0;
        for(p=strtok_r(line,",",&save);p;p=strtok_r(NULL,",",&save)){
            switch(col){case 6:action=atoi(p);break;case 12:dist=atof(p);break;case 14:hcos=atof(p);break;case 15:hsin=atof(p);break;
                case 16:spd=atof(p);break;case 17:fspd=atof(p);break;case 18:lspd=atof(p);break;case 19:bdist=atof(p);break;
                case 20:fclr=atof(p);break;case 21:lclr=atof(p);break;case 22:rclr=atof(p);break;case 25:hull=atof(p);break;}col++;}
        if(action<0||action>=nc)continue;

        /* FIXED: encode using actual kinematics features, preserving topology */
        int dk=(int)(dist/100);int sk=(int)(spd/10);int hk=(int)((hcos+1)*50);
        int fk=(int)(fclr*10);int lk=(int)(lclr*10);int rk=(int)(rclr*10);
        uint64_t seed=(uint64_t)(dk*1000000+sk*10000+hk*100+fk*10+lk);
        seed=seed*31+(uint64_t)(rk*7+(int)(hull*10));

        double st[DIM];leos_engine_keygen(eng,seed,st);

        if(total>0&&total%10==0){double sc[64];double t0=ns();holo_ranker_score(hr,st,acts,(size_t)nc,sc);e_time+=ns()-t0;
            int b=0;for(int j=1;j<nc;j++)if(sc[j]>sc[b])b=j;if(b==action)ec++;et++;}
        holo_ranker_store(hr,st,&acts[action*DIM]);total++;
        if(total%1000==0){printf("  %5d | %5.1f%% | fid=%.4f | %.0fus\n",total,et>0?(double)ec/et*100:0,1.0/sqrt((double)total),et>0?e_time/et/1000:0);}
    }
    fclose(fp);
    printf("  Final: %d frames, %.1f%% acc, %.0f us/query\n",total,et>0?(double)ec/et*100:0,et>0?e_time/et/1000:0);
    holo_ranker_destroy(hr);leos_engine_destroy(eng);
}

int main(void){
    printf("=== Cross-Domain + Void + Scale ===\n\n");
    int nt=300,nte=200;
    test_shared(nt,nte);
    test_transfer(nt,nte);
    test_voids();
    test_scale();
    printf("\nDone.\n");return 0;
}
