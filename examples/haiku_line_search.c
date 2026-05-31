#include "nn.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { MAX_LEX=512, MAX_LN=256, H_FEAT=26 };

typedef struct { char word[32]; int syl; } HWord;
static HWord LEX[MAX_LEX]; static size_t LEX_N;
static double UG[MAX_LEX], BG[MAX_LEX][MAX_LEX], TG[MAX_LEX][MAX_LEX][MAX_LEX], LS[MAX_LEX];

static int syllables(const char *w){int n=0;for(const char*p=w;*p;++p)if(strchr("aeiouy",*p)){if(p==w||!strchr("aeiouy",*(p-1)))++n;}return n>0?n:1;}
static size_t add_word(const char *w){for(size_t i=0;i<LEX_N;++i)if(!strcmp(LEX[i].word,w))return i;if(LEX_N>=MAX_LEX)return MAX_LEX;size_t n=strlen(w);if(n>=32)n=31;memcpy(LEX[LEX_N].word,w,n);LEX[LEX_N].word[n]=0;LEX[LEX_N].syl=syllables(w);return LEX_N++;}

static void build_stats(const char *path){
    FILE*fp=fopen(path,"r");if(!fp)return;
    memset(UG,0,sizeof(UG));memset(BG,0,sizeof(BG));memset(TG,0,sizeof(TG));memset(LS,0,sizeof(LS));
    char ln[MAX_LN];size_t total=0,n_blank=0;
    while(fgets(ln,sizeof(ln),fp)){size_t l=strlen(ln);while(l>0&&(ln[l-1]=='\n'||ln[l-1]=='\r'))ln[--l]=0;
        if(!l){n_blank=0;continue;}
        size_t wa[32];size_t nw=0;char*s=ln;
        for(char*p=ln;;++p){if(*p==' '||*p==0){size_t wl=(size_t)(p-s);if(wl>0){char b[64];memcpy(b,s,wl);b[wl]=0;size_t wi=add_word(b);if(wi<MAX_LEX&&nw<32)wa[nw++]=wi;}if(!*p)break;s=p+1;}}
        if(!nw)continue;if(n_blank==0)LS[wa[0]]+=1.0;
        for(size_t i=0;i<nw;++i){UG[wa[i]]+=1.0;total++;if(i+1<nw)BG[wa[i]][wa[i+1]]+=1.0;if(i+2<nw)TG[wa[i]][wa[i+1]][wa[i+2]]+=1.0;}++n_blank;}
    fclose(fp);if(!total)return;
    for(size_t i=0;i<LEX_N;++i)UG[i]/=(double)total;
    for(size_t i=0;i<LEX_N;++i){double s=0;for(size_t j=0;j<LEX_N;++j)s+=BG[i][j];if(s>0)for(size_t j=0;j<LEX_N;++j)BG[i][j]/=s;}
    double ls=0;for(size_t i=0;i<LEX_N;++i)ls+=LS[i];if(ls>0)for(size_t i=0;i<LEX_N;++i)LS[i]/=ls;}
static double CLUSTER_SIM[MAX_LEX][MAX_LEX];
static void build_clusters(void){
    /* Precompute word similarity from bigram-partner overlap.
       Two words are similar if they share many bigram partners. */
    for(size_t i=0;i<LEX_N;++i)for(size_t j=0;j<LEX_N;++j){
        double overlap=0,norm_i=0,norm_j=0;
        for(size_t k=0;k<LEX_N;++k){
            if(BG[i][k]>0&&BG[j][k]>0)overlap+=1.0;
            if(BG[i][k]>0)norm_i+=1.0;if(BG[j][k]>0)norm_j+=1.0;}
        double denom=sqrt(norm_i*norm_j);
        CLUSTER_SIM[i][j]=denom>0.001?overlap/denom:0.0;
    }
}

/* POS-approximate sensors without a tagger */
static int ends_with(const char *w, const char *suffix) {
    size_t wl=strlen(w),sl=strlen(suffix);
    return wl>=sl&&!strcmp(w+wl-sl,suffix);
}
static int is_function_word(const char *w) {
    static const char *fws[] = {"the","a","of","by","in","on","at","for","with","to","and","is","it","as","an"};
    for(size_t i=0;i<sizeof(fws)/sizeof(fws[0]);++i)if(!strcmp(w,fws[i]))return 1;
    return 0;
}
static int is_likely_adj(const char *w){return ends_with(w,"y")||ends_with(w,"ing")||ends_with(w,"ed")||ends_with(w,"ous")||ends_with(w,"ful")||ends_with(w,"less")||ends_with(w,"ish");}
static int is_likely_verb(const char *w){return ends_with(w,"s")||ends_with(w,"ed")||ends_with(w,"ing")||ends_with(w,"es");}
static int is_likely_noun(const char *w){return !is_function_word(w)&&!is_likely_verb(w)&&!is_likely_adj(w);}

static void enc_word(size_t wi,int syl,int tgt,int wpos,nn_real*f,size_t pw,size_t ppw,
                     const char *forbidden,int adj_streak,int has_verb,int has_noun){
    memset(f,0,H_FEAT*sizeof(*f));int ws=LEX[wi].syl;
    const char *w=LEX[wi].word;
    /* Base sensors (0-5) */
    f[0]=1.0;f[1]=(nn_real)syl/(nn_real)tgt;f[2]=(nn_real)(tgt-syl)/7.0;f[3]=(nn_real)ws/3.0;
    f[4]=(nn_real)wpos/8.0;f[5]=wpos==0?1.0:0.0;
    /* Corpus binary sensors (6-9) */
    f[6]=UG[wi]>0.0?1.0:0.0;
    f[7]=(pw<LEX_N&&BG[pw][wi]>0.0)?1.0:0.0;
    f[8]=(ppw<LEX_N&&pw<LEX_N&&TG[ppw][pw][wi]>0.0)?1.0:0.0;
    f[9]=wpos==0?(LS[wi]>0.0?1.0:0.0):0.0;
    /* Structural sensors (10-13) */
    f[10]=(nn_real)strlen(w)/12.0;
    f[11]=ws==1?1.0:0.0;f[12]=ws==2?1.0:0.0;
    f[13]=tgt-syl-ws==0?1.0:0.0;
    /* POS-approximate sensors (14-17) */
    f[14]=is_function_word(w)?1.0:0.0;
    f[15]=is_likely_adj(w)?1.0:0.0;
    f[16]=is_likely_verb(w)?1.0:0.0;
    f[17]=is_likely_noun(w)?1.0:0.0;
    /* Quality sensors (18-21) */
    f[18]=(tgt-syl-ws==0&&is_function_word(w))?1.0:0.0;          /* func word at line end */
    f[19]=(adj_streak>=2&&is_likely_adj(w))?1.0:0.0;              /* adjective chain >=3 */
    f[20]=(tgt-syl-ws==0&&!has_verb&&!is_likely_verb(w))?1.0:0.0; /* no verb in line */
    f[21]=(tgt-syl-ws==0&&!has_noun&&!is_likely_noun(w))?1.0:0.0; /* no noun in line */
    /* Cross-line sensors (22-23) */
    f[22]=0.0;if(forbidden&&forbidden[0]){char fb[256];strcpy(fb,forbidden);
        char*tok=strtok(fb," ");while(tok){if(!strcmp(w,tok)){f[22]=1.0;break;}tok=strtok(NULL," ");}}
    f[23]=UG[wi]>0.02?1.0:0.0;
    /* Semantic cluster similarity to previous word */
    f[24]=(pw<LEX_N)?(nn_real)CLUSTER_SIM[pw][wi]:0.0;
    /* Semantic cluster similarity to line 2 (for line 3 candidates) */
    f[25]=0.0;
}

static int line_score(NeuralNet*net,int tgt,size_t bw,char*out,size_t cap,unsigned lseed,const char*forbidden){
    (void)bw;if(cap<256)return 0;char pfx[256]={0};int syl=0,st=0,adj_streak=0,has_verb=0,has_noun=0;size_t pw=LEX_N,ppw=LEX_N;
    while(syl<tgt&&st<tgt*6){size_t cw[MAX_LEX];nn_real cs[MAX_LEX];size_t cn=0;nn_real mx=-1e30;nn_real in[H_FEAT];
        for(size_t wi=0;wi<LEX_N;++wi){int ws=LEX[wi].syl;if(syl+ws>tgt)continue;
            enc_word(wi,syl,tgt,st,in,pw,(st>=2?ppw:LEX_N),forbidden,adj_streak,has_verb,has_noun);nn_real sc=0;nn_forward(net,in,&sc);
            /* Hard trigram multiplier: invalid trigrams get 0.15x score */
            if(st>=2&&ppw<LEX_N&&pw<LEX_N&&TG[ppw][pw][wi]<=0.0)sc*=(nn_real)0.15;
            /* Cross-line repeat penalty already handled by f[22] sensor */
            /* Semantic contrast: if line 3 word is too similar to line 2 words, penalize */
            if(forbidden&&forbidden[0]&&st>=0 && tgt==5){char fb[256];strcpy(fb,forbidden);
                char*tok=strtok(fb," ");while(tok){for(size_t li=0;li<LEX_N;++li)if(!strcmp(LEX[li].word,tok)&&CLUSTER_SIM[wi][li]>0.5){sc*=(nn_real)0.4;break;}tok=strtok(NULL," ");}}
            if(sc>mx)mx=sc;if(cn<MAX_LEX){cw[cn]=wi;cs[cn]=sc;++cn;}}
        if(!cn)break;
        nn_real tmp=0.7,sum=0;for(size_t ci=0;ci<cn;++ci){cs[ci]=(nn_real)exp((double)(cs[ci]-mx)/tmp);sum+=cs[ci];}
        nn_real r=(nn_real)((st*17+syl*31+pw*13+tgt*7+lseed)%1000)/1000.0f,cum=0;size_t bw2=LEX_N;
        for(size_t ci=0;ci<cn;++ci){cum+=cs[ci]/sum;if(r<=cum){bw2=cw[ci];break;}}if(bw2>=LEX_N)bw2=cw[0];
        const char*w=LEX[bw2].word;size_t wl=strlen(w),pl=strlen(pfx);if(pl+wl+2>=255)break;
        if(pl>0){pfx[pl]=' ';pfx[pl+1]=0;}strcat(pfx,w);syl+=LEX[bw2].syl;ppw=(st>=1?pw:LEX_N);pw=bw2;
        if(is_likely_adj(LEX[bw2].word))++adj_streak;else adj_streak=0;
        if(is_likely_verb(LEX[bw2].word))has_verb=1;
        if(is_likely_noun(LEX[bw2].word))has_noun=1;
        ++st;}
    size_t pl=strlen(pfx);if(pl>0&&pl<cap){memcpy(out,pfx,pl);out[pl]=0;return 1;}out[0]=0;return 0;
}

static int triplet(NeuralNet*net,size_t bw,char*out,size_t cap,unsigned seed){
    char ls[3][256];int t[]={5,7,5};size_t p=0;
    for(int li=0;li<3;++li){const char*fb=NULL;  /* words to penalize repeating */
        if(li==2){static char fb_buf[512];snprintf(fb_buf,sizeof(fb_buf),"%s %s",ls[0],ls[1]);fb=fb_buf;}  /* penalize repeats from lines 1+2 */
        if(!line_score(net,t[li],bw,ls[li],256,(unsigned)(li*7919+seed),fb))return 0;
        size_t l=strlen(ls[li]);if(p+l+2>cap)return 0;if(li>0)out[p++]='\n';memcpy(out+p,ls[li],l);p+=l;}out[p]=0;return 1;
}

int main(int argc,char**argv){
    size_t bw=8,ep=64;unsigned seed=20240522;const char*cp=NULL;
    for(int i=1;i<argc;++i){if(!strcmp(argv[i],"--beam")&&i+1<argc)bw=(size_t)strtoul(argv[++i],NULL,10);
        else if(!strcmp(argv[i],"--epochs")&&i+1<argc)ep=(size_t)strtoul(argv[++i],NULL,10);
        else if(!strcmp(argv[i],"--seed")&&i+1<argc)seed=(unsigned)strtoul(argv[++i],NULL,10);
        else if(!strcmp(argv[i],"--corpus")&&i+1<argc)cp=argv[++i];}
    if(!bw)bw=1;if(!cp){fprintf(stderr,"usage: ... --corpus path\n");return 1;}
    build_stats(cp);build_clusters();fprintf(stderr,"lexicon: %zu words\n",LEX_N);
    FILE*fp=fopen(cp,"r");if(!fp)return 1;size_t ns=0;char ln[MAX_LN];
    while(fgets(ln,sizeof(ln),fp)){size_t l=strlen(ln);while(l>0&&(ln[l-1]=='\n'||ln[l-1]=='\r'))ln[--l]=0;if(!l)continue;
        char*s=ln;for(char*p=ln;;++p){if(*p==' '||*p==0){size_t wl=(size_t)(p-s);if(wl>0){char b[64];memcpy(b,s,wl);b[wl]=0;
            for(size_t wi=0;wi<LEX_N;++wi)if(!strcmp(LEX[wi].word,b)){ns+=8;break;}}if(!*p)break;s=p+1;}}}rewind(fp);
    if(!ns)ns=64;
    nn_real*in=(nn_real*)calloc(ns*H_FEAT,sizeof(*in));nn_real*tg=(nn_real*)calloc(ns,sizeof(*tg));if(!in||!tg){fclose(fp);return 1;}
    size_t si=0;while(fgets(ln,sizeof(ln),fp)&&si<ns){size_t l=strlen(ln);while(l>0&&(ln[l-1]=='\n'||ln[l-1]=='\r'))ln[--l]=0;if(!l)continue;
        size_t wa[32];size_t nw=0;char*s=ln;for(char*p=ln;;++p){if(*p==' '||*p==0){size_t wl=(size_t)(p-s);if(wl>0){char b[64];memcpy(b,s,wl);b[wl]=0;
            for(size_t wi=0;wi<LEX_N;++wi)if(!strcmp(LEX[wi].word,b)){if(nw<32)wa[nw++]=wi;break;}}if(!*p)break;s=p+1;}}if(!nw)continue;
        int tgt=0;for(size_t i=0;i<nw;++i)tgt+=LEX[wa[i]].syl;if(tgt<6)tgt=5;else tgt=7;int syl=0;
        for(size_t wp=0;wp<nw&&si<ns;++wp){size_t aw=wa[wp];
            enc_word(aw,syl,tgt,(int)wp,&in[si*H_FEAT],wp>0?wa[wp-1]:LEX_N,wp>1?wa[wp-2]:LEX_N,NULL,0,0,0);tg[si]=1.0;++si;
            for(int neg=0;neg<7&&si<ns;++neg){size_t bw2;int bs2,dw=0;
                do{bw2=((size_t)(si*31+wp*17+neg*13+dw))%LEX_N;bs2=LEX[bw2].syl;++dw;}while((bw2==aw||syl+bs2>tgt)&&dw<1000);
                enc_word(bw2,syl,tgt,(int)wp,&in[si*H_FEAT],wp>0?wa[wp-1]:LEX_N,wp>1?wa[wp-2]:LEX_N,NULL,0,0,0);tg[si]=0.0;++si;}syl+=LEX[aw].syl;}}
    fclose(fp);ns=si;fprintf(stderr,"samples: %zu\n",ns);
    const size_t layers[]={H_FEAT,12,1};NNConfig cfg={.layer_sizes=layers,.layer_count=3,.seed=seed,.hidden_activation=NN_ACTIVATION_RELU,.output_activation=NN_ACTIVATION_SIGMOID,.loss=NN_LOSS_MSE};
    NeuralNet*net=nn_create_config(&cfg);if(!net){free(in);free(tg);return 1;}
    nn_real loss=(nn_real)nn_train(net,in,tg,ns,(nn_real)0.05,ep,1);fprintf(stderr,"loss=%.4f\n",(double)loss);free(in);free(tg);
    for(int i=0;i<5;++i){char h[768];if(triplet(net,bw,h,sizeof(h),(unsigned)(i*7919+seed)))printf("%s\n\n",h);}
    nn_free(net);return 0;
}
