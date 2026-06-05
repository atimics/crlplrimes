/* arc_discovery.c — Holo-guided ARC transform discovery.
   Loads ARC tasks, builds VSA holo store, retrieves similar tasks.
   Usage: arc_discovery <task_dir> <output.jsonl>
   Compile: cc -Iinclude -std=c11 -O3 -o arc_discovery examples/arc_discovery.c \
            build/float/nn_holographic_ranker.o build/float/nn_holo_gate.o build/float/leos_engine.o -lm
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <dirent.h>
#include <sys/stat.h>

#include "nn_holographic_ranker.h"
#include "holographic/leos_engine.h"

#define MAX_TASKS 500
#define MAX_PAIRS 5
#define MAX_CELLS 1024
#define DIM 2048
#define FEAT_DIM 64

typedef struct { size_t w, h; uint8_t c[MAX_CELLS]; } Grid;
typedef struct { Grid in, out; } Pair;
typedef struct { char id[64]; size_t np; Pair p[MAX_PAIRS]; } Task;

static void mkdir_p(const char *path) {
    char tmp[1024]; snprintf(tmp, sizeof(tmp), "%s", path);
    for (char *p = tmp+1; *p; p++) { if (*p=='/') { *p=0; mkdir(tmp,0755); *p='/'; } }
    mkdir(tmp, 0755);
}

static int load_task(const char *path, Task *t) {
    FILE *f = fopen(path, "r"); if (!f) return 0;
    char buf[131072]; size_t len = fread(buf,1,sizeof(buf)-1,f); fclose(f);
    if (len<50) return 0; buf[len]=0;
    memset(t,0,sizeof(*t));
    const char *s = strrchr(path,'/'); s = s ? s+1 : path;
    strncpy(t->id,s,63); char *d = strchr(t->id,'.'); if(d) *d=0;
    
    char *p = strstr(buf,"\"train\":"); if(!p) return 0;
    p = strchr(p,'['); if(!p) return 0; p++;
    while (t->np < MAX_PAIRS && *p && *p!=']') {
        char *inp = strstr(p,"\"input\":"), *outp = strstr(p,"\"output\":");
        if(!inp||!outp) break;
        Grid *ig=&t->p[t->np].in, *og=&t->p[t->np].out;
        ig->h=ig->w=0; char *rp = strchr(inp+8,'[')+1;
        while(*rp&&*rp!=']'){if(*rp=='['){rp++;size_t rw=0;
        while(*rp&&*rp!=']'){if(*rp>='0'&&*rp<='9'){ig->c[ig->h*ig->w+rw]=*rp-'0';rw++;}rp++;}
        if(*rp==']')rp++;ig->h++;ig->w=rw;}else rp++;}
        og->h=og->w=0; rp=strchr(outp+8,'[')+1;
        while(*rp&&*rp!=']'){if(*rp=='['){rp++;size_t rw=0;
        while(*rp&&*rp!=']'){if(*rp>='0'&&*rp<='9'){og->c[og->h*og->w+rw]=*rp-'0';rw++;}rp++;}
        if(*rp==']')rp++;og->h++;og->w=rw;}else rp++;}
        t->np++; p=outp;
    }
    return t->np>0;
}

static void task_feats(const Task *t, double *f) {
    size_t fi=0;
    for(size_t i=0;i<t->np&&i<2;i++){f[fi++]=t->p[i].in.w/30.0;f[fi++]=t->p[i].in.h/30.0;
        f[fi++]=t->p[i].out.w/30.0;f[fi++]=t->p[i].out.h/30.0;}
    double cols[10]={0}; double tot=0;
    for(size_t i=0;i<t->np&&i<2;i++){
        for(size_t y=0;y<t->p[i].in.h;y++)for(size_t x=0;x<t->p[i].in.w;x++)
            {int c=t->p[i].in.c[y*t->p[i].in.w+x];if(c>=0&&c<10){cols[c]++;tot++;}}
        for(size_t y=0;y<t->p[i].out.h;y++)for(size_t x=0;x<t->p[i].out.w;x++)
            {int c=t->p[i].out.c[y*t->p[i].out.w+x];if(c>=0&&c<10){cols[c]++;tot++;}}}
    for(int c=0;c<10;c++)f[fi++]=tot>0?cols[c]/tot:0;
    while(fi<FEAT_DIM)f[fi++]=0;
}

static void embed(LeosEngine *eng, const double *f, double *out) {
    memset(out,0,DIM*sizeof(double));
    for(size_t i=0;i<FEAT_DIM&&i<DIM;i++){
        double ph=f[i]*3.0; size_t bs=(i*DIM)/FEAT_DIM,be=((i+1)*DIM)/FEAT_DIM;
        for(size_t d=bs;d<be;d++)out[d]=sin(ph*(double)(d-bs+1));}
    leos_engine_normalize(eng,out);
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr,"usage: %s <task_dir> <output_dir>\n",argv[0]); return 1; }
    mkdir_p(argv[2]);
    char opath[1024]; snprintf(opath,sizeof(opath),"%s/arc_discovery.jsonl",argv[2]);
    FILE *of = fopen(opath,"w"); if(!of) return 1;
    
    DIR *dir = opendir(argv[1]); if(!dir){fclose(of);return 1;}
    Task *tasks = calloc(MAX_TASKS,sizeof(Task)); size_t n=0;
    struct dirent *e;
    while((e=readdir(dir))&&n<MAX_TASKS){
        if(!strstr(e->d_name,".json"))continue;
        char tp[1024]; snprintf(tp,sizeof(tp),"%s/%s",argv[1],e->d_name);
        if(load_task(tp,&tasks[n]))n++;}
    closedir(dir);
    fprintf(stderr,"Loaded %zu tasks\n",n);
    if(n<10){fclose(of);return 0;}
    
    size_t ntr=(n*7)/10, nte=n-ntr;
    LeosEngine *eng = leos_engine_create(DIM);
    HolographicRanker *holo = holo_ranker_create(DIM);
    double f[FEAT_DIM], se[DIM], ae[DIM];
    
    /* Store training tasks in holo */
    for(size_t i=0;i<ntr;i++){
        task_feats(&tasks[i],f); embed(eng,f,se); embed(eng,f,ae);
        holo_ranker_store(holo,se,ae);}
    fprintf(stderr,"Holo: %zu stored, fid=%.3f\n",ntr,holo_ranker_fidelity(holo));
    
    /* Pre-build action embeddings for all training tasks */
    double (*aes)[DIM] = calloc(ntr,sizeof(double[DIM]));
    for(size_t j=0;j<ntr;j++){task_feats(&tasks[j],f);embed(eng,f,aes[j]);}
    
    /* Test: for each test task, query holo, find best training match */
    size_t hit=0, dim_hit=0;
    double sc[MAX_TASKS];
    for(size_t i=ntr;i<n;i++){
        task_feats(&tasks[i],f); embed(eng,f,se);
        holo_ranker_score(holo,se,(double*)aes,ntr,sc);
        size_t best=0;
        for(size_t j=1;j<ntr;j++)if(sc[j]>sc[best])best=j;
        
        /* Self-retrieval: does best match have the same output dimensions? */
        int dm = (tasks[best].p[0].out.w==tasks[i].p[0].out.w &&
                  tasks[best].p[0].out.h==tasks[i].p[0].out.h);
        if(dm) dim_hit++;
        /* VSA similarity threshold */
        double sim = leos_engine_dot(eng,se,aes[best]);
        if(sim>0.5)hit++;
        
        fprintf(of,"{\"id\":\"%s\",\"best\":\"%s\",\"sim\":%.4f,\"dim_match\":%s}\n",
                tasks[i].id,tasks[best].id,sim,dm?"true":"false");
    }
    
    fprintf(of,"{\"summary\":{\"train\":%zu,\"test\":%zu,\"dim_hits\":%zu,\"sim_hits\":%zu}}\n",
            ntr,nte,dim_hit,hit);
    fprintf(stderr,"Results: dim_match=%zu/%zu (%.1f%%) sim>0.5=%zu/%zu (%.1f%%)\n",
            dim_hit,nte,(double)dim_hit/nte*100,hit,nte,(double)hit/nte*100);
    
    free(aes); holo_ranker_destroy(holo); leos_engine_destroy(eng);
    free(tasks); fclose(of);
    return 0;
}
