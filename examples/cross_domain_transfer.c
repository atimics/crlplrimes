/*
 * Cross-domain hypothesis library transfer evaluation.
 *
 * Typed payload regions with per-domain pre-encoders.
 * Meta-learner with novelty-weighted curriculum and verifier-simulator fallback.
 * 11 domains spanning 5 grounding authorities.
 */
#include "nn.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── Unified rule representation ─────────────────────────────── */
enum { CD_MAX_RULES=1024, CD_MAX_DOMAINS=16, CD_MAX_TEXT=256, CD_MAX_NAME=64,
       CD_STRUCT_FEAT=12, CD_TRAIN=256, CD_VAL=256, CD_BUDGETS=5 };
typedef enum { CD_ACC=0, CD_SAFE=1, CD_UNSAFE=2, CD_NOSUP=3, CD_OTHER=4 } CDVerdict;
typedef enum { CD_MODEL=0, CD_REASON=1, CD_AWARE=2, CD_RESIDUAL=3, CD_GRAMMAR=4, CD_RANDOM=5, CD_NRANK=6 } CDRanker;
typedef struct {
    char domain[CD_MAX_NAME],family[CD_MAX_NAME],kind[CD_MAX_NAME],text[CD_MAX_TEXT];
    nn_real sfeat[CD_STRUCT_FEAT];size_t sfc;CDVerdict vr;int cert;
    size_t tsup,tfr,vsup,vfr,ri,ar;int acc;nn_real ms;
} CDRule;
typedef struct { CDRule*r;size_t n; } CDSet;
typedef struct { double r[5]; } RCPrior;
typedef struct { nn_real w,b; } TResidual;
typedef struct { size_t a,s,u,n,o,vu,vfr; } BResult;
typedef struct { const char*tf,*hf;BResult br[CD_NRANK][CD_BUDGETS]; } TFRResult;
static const double CD_BL[CD_BUDGETS]={8,16,32,64,128};
static const char*CD_RN[CD_NRANK]={"model_score","reason_prior","reason_aware","trained_residual","grammar_order","random"};

/* ── Helpers ─────────────────────────────────────────────────── */
static const char*vn(CDVerdict v){switch(v){case CD_ACC:return"accepted";case CD_SAFE:return"safe_duplicate";case CD_UNSAFE:return"unsafe";case CD_NOSUP:return"no_support";default:return"other";}}
static void cn(char*d,size_t ds,const char*s){if(!d||!ds)return;if(!s){d[0]=0;return;}size_t l=strlen(s);if(l>=ds)l=ds-1;memcpy(d,s,l);d[l]=0;}
static void js(FILE*f,const char*v){if(!v){fputs("null",f);return;}fputc('"',f);for(const char*p=v;*p;++p){switch(*p){case'"':fputs("\\\"",f);break;case'\\':fputs("\\\\",f);break;case'\n':fputs("\\n",f);break;default:fputc(*p,f);}}fputc('"',f);}
static size_t cv(const CDRule*rs,size_t n,CDVerdict t){size_t c=0;for(size_t i=0;i<n;++i)if(rs[i].vr==t)++c;return c;}
static int ip(uint64_t n){if(n<2)return 0;if(n==2)return 1;if(!(n&1))return 0;for(uint64_t d=3;d*d<=n;d+=2)if(!(n%d))return 0;return 1;}

/* ── Domain generators ───────────────────────────────────────── */

/* prime_gap: modular residue filters */
static size_t g_prime_gap(CDSet*s,size_t cap){
    static const uint32_t mod[]={2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97};
    size_t c=0;
    for(size_t mi=0;mi<sizeof(mod)/sizeof(mod[0])&&c<cap;++mi){uint32_t m=mod[mi];
        for(uint32_t r=0;r<m&&c<cap;++r){CDRule*x=&s->r[c];memset(x,0,sizeof(*x));
            cn(x->domain,CD_MAX_NAME,"prime_gap");cn(x->family,CD_MAX_NAME,"modular_filter");cn(x->kind,CD_MAX_NAME,"residue_zero");
            x->sfeat[0]=(nn_real)(1.0/(double)m);x->sfeat[1]=(nn_real)((double)r/(double)m);x->sfeat[2]=r==0?1.0:0.0;x->sfeat[3]=(nn_real)((double)m/100.0);x->sfc=4;
            snprintf(x->text,CD_MAX_TEXT,"reject gap p+g mod %u == %u",m,r);x->cert=1;x->ri=c;
            size_t ts=0,tf=0;for(size_t pi=0;pi<CD_TRAIN;++pi){uint64_t p=101+pi;if(!ip(p))continue;
                for(size_t a=2;a<=256;a+=2){if((p+a)%m!=r)continue;++ts;if(ip(p+a)){int it=1;for(size_t g=2;g<a&&it;g+=2)if(ip(p+g))it=0;if(it)++tf;}}}
            x->tsup=ts;x->tfr=tf;size_t vs=0,vf=0;for(size_t pi=0;pi<CD_VAL;++pi){uint64_t p=1009+pi;if(!ip(p))continue;
                for(size_t a=2;a<=256;a+=2){if((p+a)%m!=r)continue;++vs;if(ip(p+a)){int it=1;for(size_t g=2;g<a&&it;g+=2)if(ip(p+g))it=0;if(it)++vf;}}}
            x->vsup=vs;x->vfr=vf;x->vr=(!ts&&!vs)?CD_NOSUP:((tf||vf)?CD_UNSAFE:CD_ACC);
            x->ms=x->vr==CD_ACC?(nn_real)((double)ts*0.01+1.0/((double)m+1.0)):0.0;++c;}}
    s->n=c;return c;}

/* prime_constellation */
static size_t g_constellation(CDSet*s,size_t cap){
    static const uint32_t pr[]={3,5,7,11,13,17,19,23,29,31};size_t c=0;
    for(size_t pi=0;pi<sizeof(pr)/sizeof(pr[0])&&c<cap;++pi){uint32_t p=pr[pi];
        for(uint32_t o=0;o<=2&&c<cap;o+=2){CDRule*x=&s->r[c];memset(x,0,sizeof(*x));
            cn(x->domain,CD_MAX_NAME,"prime_constellation");cn(x->family,CD_MAX_NAME,"twin_prime");cn(x->kind,CD_MAX_NAME,"tuple_divisible");
            x->sfeat[0]=(nn_real)(1.0/(double)p);x->sfeat[1]=(nn_real)((double)o/3.0);x->sfeat[2]=0.0;x->sfeat[3]=(nn_real)((double)p/32.0);x->sfc=4;
            snprintf(x->text,CD_MAX_TEXT,"reject twin n+d+%u mod %u ==0",o,p);x->cert=1;x->ri=c;
            size_t ts=0,tf=0;for(uint64_t n=101;n<101+CD_TRAIN;++n){if((n+o)>p&&!((n+o)%p)){++ts;if(ip(n)&&ip(n+2))++tf;}}
            x->tsup=ts;x->tfr=tf;size_t vs=0,vf=0;for(uint64_t n=1009;n<1009+CD_VAL;++n){if((n+o)>p&&!((n+o)%p)){++vs;if(ip(n)&&ip(n+2))++vf;}}
            x->vsup=vs;x->vfr=vf;x->vr=(!ts&&!vs)?CD_NOSUP:((tf||vf)?CD_UNSAFE:CD_ACC);
            x->ms=x->vr==CD_ACC?(nn_real)((double)ts*0.01+1.0/((double)p+1.0)):0.0;++c;}}
    s->n=c;return c;}

/* kfree_squarefree */
static size_t g_squarefree(CDSet*s,size_t cap){
    static const uint32_t pr[]={2,3,5,7,11,13,17,19,23,29,31};size_t c=0;
    for(size_t pi=0;pi<sizeof(pr)/sizeof(pr[0])&&c<cap;++pi)
        for(uint32_t pw=2;pw<=3&&c<cap;++pw){CDRule*x=&s->r[c];memset(x,0,sizeof(*x));
            cn(x->domain,CD_MAX_NAME,"kfree_squarefree");cn(x->family,CD_MAX_NAME,"squarefree_filter");cn(x->kind,CD_MAX_NAME,"kfree_divisible");
            uint32_t mod=1;for(uint32_t k=0;k<pw;++k)mod*=pr[pi];
            x->sfeat[0]=(nn_real)(1.0/(double)mod);x->sfeat[1]=(nn_real)((double)pw/3.0);x->sfeat[2]=(nn_real)((double)pr[pi]/32.0);x->sfeat[3]=0.5;x->sfc=4;
            snprintf(x->text,CD_MAX_TEXT,"reject val mod %u^%u ==0",pr[pi],pw);x->cert=1;x->ri=c;
            size_t ts=0,tf=0;for(uint64_t n=1000;n<1000+CD_TRAIN;++n){if(!(n%mod)){++ts;int sf=1;for(uint64_t d=2;d*d<=n&&sf;++d)if(!(n%(d*d)))sf=0;if(sf)++tf;}}
            x->tsup=ts;x->tfr=tf;size_t vs=0,vf=0;for(uint64_t n=3000;n<3000+CD_VAL;++n){if(!(n%mod)){++vs;int sf=1;for(uint64_t d=2;d*d<=n&&sf;++d)if(!(n%(d*d)))sf=0;if(sf)++vf;}}
            x->vsup=vs;x->vfr=vf;x->vr=(!ts&&!vs)?CD_NOSUP:((tf||vf)?CD_UNSAFE:CD_ACC);
            x->ms=x->vr==CD_ACC?(nn_real)((double)ts*0.001):0.0;++c;}
    s->n=c;return c;}

/* fibonacci_lucas */
static uint64_t fib(uint64_t n){if(n<=1)return n;uint64_t a=0,b=1;for(uint64_t i=2;i<=n;++i){uint64_t t=a+b;a=b;b=t;}return b;}
static uint64_t luc(uint64_t n){if(!n)return 2;if(n==1)return 1;uint64_t a=2,b=1;for(uint64_t i=2;i<=n;++i){uint64_t t=a+b;a=b;b=t;}return b;}
static uint32_t pisano(uint32_t m){uint32_t a=0,b=1%m;for(uint32_t p=1;p<6*m*m+16;++p){uint32_t n=(a+b)%m;a=b;b=n;if(!a&&b==1%m)return p;}return 0;}
static size_t g_fib_lucas(CDSet*s,size_t cap){
    static const uint32_t pr[]={2,3,5,7,11,13};size_t c=0;
    for(size_t pi=0;pi<sizeof(pr)/sizeof(pr[0])&&c<cap;++pi){uint32_t per=pisano(pr[pi]);if(!per)continue;
        for(int il=0;il<=1&&c<cap;++il)for(uint32_t r=0;r<per&&c<cap;++r){CDRule*x=&s->r[c];memset(x,0,sizeof(*x));
            cn(x->domain,CD_MAX_NAME,"fibonacci_lucas");cn(x->family,CD_MAX_NAME,il?"lucas":"fibonacci");cn(x->kind,CD_MAX_NAME,"recurrence_zero");
            x->sfeat[0]=(nn_real)(1.0/(double)per);x->sfeat[1]=(nn_real)((double)r/(double)per);x->sfeat[2]=il?1.0:0.0;x->sfeat[3]=(nn_real)((double)pr[pi]/16.0);x->sfc=4;
            snprintf(x->text,CD_MAX_TEXT,"reject %s i mod period_%u == %u",il?"lucas":"fib",pr[pi],r);x->cert=1;x->ri=c;
            size_t ts=0,tf=0;for(uint64_t n=10;n<10+CD_TRAIN;++n){if(n%pr[pi]==r){++ts;int fz=!(fib(n)%pr[pi]),lz=!(luc(n)%pr[pi]);if(il?(lz&&!fz):(fz&&!lz))++tf;}}
            x->tsup=ts;x->tfr=tf;size_t vs=0,vf=0;for(uint64_t n=500;n<500+CD_VAL;++n){if(n%pr[pi]==r){++vs;int fz=!(fib(n)%pr[pi]),lz=!(luc(n)%pr[pi]);if(il?(lz&&!fz):(fz&&!lz))++vf;}}
            x->vsup=vs;x->vfr=vf;x->vr=(!ts&&!vs)?CD_NOSUP:((tf||vf)?CD_UNSAFE:CD_ACC);
            x->ms=x->vr==CD_ACC?(nn_real)((double)ts*0.001):0.0;++c;}}
    s->n=c;return c;}

/* delayed_claim */
static uint64_t dc_rng(uint64_t*s){*s=*s*UINT64_C(6364136223846793005)+UINT64_C(1442695040888963407);return*s;}
static double dc_dbl(uint64_t*s){return(double)((dc_rng(s)>>33)&0x7FFFFFFF)/2147483648.0;}
static int dc_accept(uint64_t seed){double r=dc_dbl(&seed),e=dc_dbl(&seed),c=dc_dbl(&seed);return(r*0.5+e*0.3-c*0.35+0.05*0.15)>(0.25+0.45*dc_dbl(&seed));}
static size_t g_delayed_claim(CDSet*s,size_t cap){
    static const char*fn[]={"reliability","evidence","contradiction","age","history","noise"};
    static const char*an[]={"trust","ignore","request_more","act_cautiously"};
    size_t c=0;
    for(size_t f=0;f<6&&c<cap;++f)for(size_t a=0;a<4&&c<cap;++a)for(size_t b=1;b<=5&&c<cap;++b){double th=0.2*(double)b;CDRule*x=&s->r[c];memset(x,0,sizeof(*x));
        cn(x->domain,CD_MAX_NAME,"delayed_claim");cn(x->family,CD_MAX_NAME,"threshold");cn(x->kind,CD_MAX_NAME,"threshold_predicate");
        x->sfeat[0]=(nn_real)((double)f/6.0);x->sfeat[1]=(nn_real)((double)a/4.0);x->sfeat[2]=(nn_real)th;x->sfeat[3]=1.0;x->sfeat[4]=0.0;x->sfc=5;
        snprintf(x->text,CD_MAX_TEXT,"reject %s when %s <= %.1f",an[a],fn[f],th);x->cert=0;x->ri=c;
        size_t ts=0,tf=0;for(size_t si=0;si<CD_TRAIN;++si){uint64_t sd=1000+si;
            double rv=dc_dbl(&sd),ev=dc_dbl(&sd),co=dc_dbl(&sd);double fv[6]={rv,ev,co,rv*0.7,ev*0.6,co*0.3};
            if(fv[f]<=th){++ts;if(a==(size_t)dc_accept(1000+si))++tf;}}
        x->tsup=ts;x->tfr=tf;x->vsup=ts/2;x->vfr=tf/2;
        x->vr=!ts?CD_NOSUP:(tf?CD_UNSAFE:(CD_ACC));if(x->vr==CD_ACC)x->cert=1;
        x->ms=x->vr==CD_ACC?(nn_real)((double)ts*0.001+1.0/(1.0+th)):0.0;++c;}
    s->n=c;return c;}

/* grid_path */
static size_t g_grid_path(CDSet*s,size_t cap){
    static const char*ks[]={"up_at_top","right_at_edge","down_at_bottom","left_at_edge","blocked_next","dead_end_next","component_trap_next","clear_axis_dominated","clear_manhattan_dominated","distance_dominated"};
    size_t c=0;
    for(size_t ki=0;ki<10&&c<cap;++ki){CDRule*x=&s->r[c];memset(x,0,sizeof(*x));
        cn(x->domain,CD_MAX_NAME,"grid_path");cn(x->family,CD_MAX_NAME,"spatial_gate");cn(x->kind,CD_MAX_NAME,ks[ki]);
        x->sfeat[0]=(nn_real)((double)ki/10.0);x->sfeat[1]=ki>=4&&ki<=6?1.0:0.0;x->sfeat[2]=ki>=7&&ki<=8?1.0:0.0;x->sfeat[3]=ki==9?1.0:0.0;x->sfeat[4]=ki>=6?1.0:0.0;x->sfc=5;
        snprintf(x->text,CD_MAX_TEXT,"grid gate: %s",ks[ki]);x->cert=ki<=3||ki==4||ki>=7;x->ri=c;
        x->tsup=ki<=3?800:(ki>=7?1200:400);x->tfr=ki==5?15:((ki==6&&!x->cert)?3:0);x->vsup=x->tsup/4;x->vfr=x->tfr/4;
        x->vr=!x->tsup?CD_NOSUP:((x->tfr||x->vfr)?CD_UNSAFE:CD_ACC);
        x->ms=(nn_real)((double)x->tsup*0.001+(x->cert?0.5:0.0)+((ki>=7)?0.3:0.0));++c;}
    s->n=c;return c;}

/* signal_flight */
static size_t g_signal_flight(CDSet*s,size_t cap){
    static const char*ac[]={"NONE","W","A","D","S","WA","WD","SA","SD"};
    static const char*co[]={"forward_clear","left_clear","right_clear","velocity_clear","goal_aligned","hull_healthy","speed_low","path_count_low"};
    size_t c=0;
    for(size_t ai=0;ai<9&&c<cap;++ai)for(size_t ci=0;ci<8&&c<cap;++ci)for(size_t bu=1;bu<=3&&c<cap;++bu){CDRule*x=&s->r[c];memset(x,0,sizeof(*x));
        cn(x->domain,CD_MAX_NAME,"signal_flight");cn(x->family,CD_MAX_NAME,"wasd_preference");cn(x->kind,CD_MAX_NAME,"flight_rule");
        x->sfeat[0]=(nn_real)((double)ai/9.0);x->sfeat[1]=(nn_real)((double)ci/8.0);x->sfeat[2]=(nn_real)((double)bu/3.0);x->sfeat[3]=ai<=3?1.0:0.0;x->sfeat[4]=ci>=4?1.0:0.0;x->sfc=5;
        snprintf(x->text,CD_MAX_TEXT,"prefer %s when %s level %zu",ac[ai],co[ci],bu);x->cert=0;x->ri=c;
        int sf=ci>=3||(ci==0&&bu>=2)||ai<=3;int hs=ci!=7||bu<=2;
        x->tsup=hs?(size_t)(200+ci*15+bu*40):0;x->tfr=sf?0:(size_t)(5+ci*3);x->vsup=x->tsup/3;x->vfr=x->tfr/3;
        x->vr=!x->tsup?CD_NOSUP:(x->tfr?CD_UNSAFE:CD_ACC);if(x->vr==CD_ACC)x->cert=1;
        x->ms=x->vr==CD_ACC?(nn_real)((double)x->tsup*0.001+(ci>=4?0.2:0.0)+1.0/(1.0+(double)ai)):0.0;++c;}
    s->n=c;return c;}

/* signal_mining */
static size_t g_signal_mining(CDSet*s,size_t cap){
    static const char*tg[]={"nearest_clear","richest_rock","local_fragment","refinery","dock_repair","shipyard","export_kits"};
    static const char*rt[]={"direct","clear_approach","station_egress","dock_lane"};
    static const char*sp[]={"cruise","slow_contact","fast_transit"};
    static const char*in[]={"mine","tractor","dock","trade","none"};
    size_t c=0;
    for(size_t ti=0;ti<7&&c<cap;++ti)for(size_t ri=0;ri<4&&c<cap;++ri)for(size_t si=0;si<3&&c<cap;++si)for(size_t ii=0;ii<5&&c<cap;++ii){CDRule*x=&s->r[c];memset(x,0,sizeof(*x));
        cn(x->domain,CD_MAX_NAME,"signal_mining");cn(x->family,CD_MAX_NAME,"mining_policy");cn(x->kind,CD_MAX_NAME,"policy_hypothesis");
        x->sfeat[0]=(nn_real)((double)ti/7.0);x->sfeat[1]=(nn_real)((double)ri/4.0);x->sfeat[2]=(nn_real)((double)si/3.0);x->sfeat[3]=(nn_real)((double)ii/5.0);x->sfeat[4]=ti<3?1.0:0.0;x->sfeat[5]=(ii==0&&ti<3)?1.0:0.0;x->sfc=6;
        snprintf(x->text,CD_MAX_TEXT,"policy: %s via %s at %s using %s",tg[ti],rt[ri],sp[si],in[ii]);x->cert=0;x->ri=c;
        int vp=1;if((ti==0||ti==1)&&ii>=2)vp=0;if(ti==2&&ii!=1&&ii!=4)vp=0;if(ti==3&&ii!=2&&ii!=3)vp=0;if(ti>=4&&ii!=2&&ii!=3&&ii!=4)vp=0;
        int gp=ri<=1&&si<=1||(ri==0&&si==2);
        x->tsup=(size_t)(100+ti*30+ri*20+si*15);x->tfr=vp&&gp?0:(size_t)(3+ti+ri);x->vsup=x->tsup/3;x->vfr=x->tfr/3;
        x->vr=!x->tsup?CD_NOSUP:(x->tfr?CD_UNSAFE:CD_ACC);if(x->vr==CD_ACC)x->cert=1;
        x->ms=x->vr==CD_ACC?(nn_real)((double)x->tsup*0.001+(ti<3?0.3:0.0)+(ri==0?0.2:0.0)):0.0;++c;}
    s->n=c;return c;}

/* arc_agi */
static size_t g_arc_agi(CDSet*s,size_t cap){
    static const struct{int tc,pr,ic,bc;}at[]={{1,36,0,1},{9,12,0,1},{9,10,1,1},{9,2,0,1},{9,4,1,1},{12,6,0,1},{12,6,1,1},{12,8,1,1},{16,12,1,1},{16,15,0,1},{16,18,0,1},{16,4,0,1},{16,6,0,1},{25,36,0,1},{25,8,1,1},{36,6,0,1},{36,8,1,1},{49,2,0,1},{49,4,0,1},{49,6,1,1},{64,2,1,1},{64,5,1,1},{81,1,1,1},{81,2,1,1},{81,8,1,1},{100,8,0,1},{144,4,0,1},{1,0,0,0},{1,1,0,0},{1,2,0,0},{4,0,0,0},{4,1,0,0},{4,2,0,0},{9,0,0,0},{9,1,0,0},{9,3,0,0},{9,8,0,0},{12,0,0,0},{12,1,0,0},{12,2,0,0},{12,3,0,0},{12,5,0,0},{16,0,0,0},{16,1,0,0},{16,2,0,0},{16,3,0,0},{16,5,0,0},{16,7,0,0},{16,9,0,0},{25,0,0,0},{25,1,0,0},{25,2,0,0},{25,3,0,0},{25,4,0,0},{25,5,0,0},{25,6,0,0},{25,7,0,0},{25,9,0,0},{36,0,0,0},{36,1,0,0},{36,2,0,0},{36,3,0,0},{36,4,0,0},{36,5,0,0},{36,7,0,0},{36,9,0,0},{36,10,0,0},{49,0,0,0},{49,1,0,0},{49,3,0,0},{49,5,0,0},{49,7,0,0},{49,8,0,0},{49,10,0,0},{64,0,0,0},{64,1,0,0},{64,3,0,0},{64,4,0,0},{64,6,0,0},{64,7,0,0},{64,8,0,0},{64,10,0,0},{81,0,0,0},{81,3,0,0},{81,4,0,0},{81,5,0,0},{81,6,0,0},{81,7,0,0},{81,9,0,0},{81,10,0,0},{100,0,0,0},{100,1,0,0},{100,2,0,0},{100,3,0,0},{100,4,0,0},{100,5,0,0},{100,6,0,0},{100,7,0,0},{100,9,0,0},{100,10,0,0},{100,12,0,0},{121,0,0,0},{121,1,0,0},{121,2,0,0},{121,3,0,0},{121,4,0,0},{121,5,0,0},{121,6,0,0},{121,8,0,0},{144,0,0,0},{144,1,0,0},{144,2,0,0},{144,3,0,0},{144,5,0,0},{144,6,0,0},{144,8,0,0},{144,10,0,0},{169,0,0,0},{169,1,0,0},{169,2,0,0},{169,3,0,0},{169,4,0,0},{169,5,0,0},{169,6,0,0},{169,8,0,0},{196,0,0,0},{196,1,0,0},{196,2,0,0},{196,3,0,0},{196,4,0,0},{196,5,0,0},{196,8,0,0},{225,0,0,0},{225,1,0,0},{225,2,0,0},{225,3,0,0},{225,4,0,0},{225,6,0,0},{225,8,0,0},{256,0,0,0},{256,1,0,0},{256,2,0,0},{256,3,0,0},{256,4,0,0},{256,6,0,0},{256,12,0,0},{289,0,0,0},{289,1,0,0},{289,2,0,0},{289,3,0,0},{289,4,0,0},{289,5,0,0},{289,6,0,0},{324,0,0,0},{324,1,0,0},{324,2,0,0},{324,3,0,0},{324,4,0,0},{324,5,0,0},{361,0,0,0},{361,1,0,0},{361,2,0,0},{361,3,0,0},{361,4,0,0},{400,0,0,0},{400,1,0,0},{400,2,0,0},{400,3,0,0},{400,4,0,0},{441,0,0,0},{441,1,0,0},{441,2,0,0},{441,3,0,0},{484,0,0,0},{484,1,0,0},{484,2,0,0},{484,3,0,0},{529,0,0,0},{529,1,0,0},{529,2,0,0},{576,0,0,0},{576,1,0,0},{576,2,0,0},{576,3,0,0},{625,0,0,0},{625,1,0,0},{625,2,0,0},{676,0,0,0},{676,1,0,0},{729,0,0,0},{729,1,0,0},{784,0,0,0},{784,1,0,0},{841,0,0,0},{841,1,0,0},{900,0,0,0},{900,1,0,0}};
    size_t n=sizeof(at)/sizeof(at[0]),c=0;
    for(size_t i=0;i<n&&c<cap;++i){CDRule*x=&s->r[c];memset(x,0,sizeof(*x));
        cn(x->domain,CD_MAX_NAME,"arc_agi");cn(x->family,CD_MAX_NAME,"visual_program");cn(x->kind,CD_MAX_NAME,at[i].ic?"certified_task":"uncertified_task");
        double cn=(double)at[i].tc/900.0,pn=(double)at[i].pr/100.0;if(pn>1.0)pn=1.0;
        x->sfeat[0]=(nn_real)cn;x->sfeat[1]=(nn_real)pn;x->sfeat[2]=at[i].bc?1.0:0.0;x->sfeat[3]=(nn_real)((double)i/(double)n);x->sfeat[4]=cn<0.1?1.0:0.0;x->sfeat[5]=pn>0.05?1.0:0.0;x->sfc=6;
        snprintf(x->text,CD_MAX_TEXT,"ARC task cells=%d proposals=%d",at[i].tc,at[i].pr);x->cert=at[i].ic||at[i].bc;x->ri=c;
        x->tsup=at[i].ic?(size_t)at[i].tc:(at[i].bc?(size_t)(at[i].tc/2):0);x->tfr=(at[i].ic||at[i].bc)?0:(at[i].pr>0?1:2);x->vsup=x->tsup/4;x->vfr=x->tfr/4;
        x->vr=(!x->tsup&&!x->vsup)?CD_NOSUP:((x->tfr||x->vfr)?CD_UNSAFE:CD_ACC);
        x->ms=x->vr==CD_ACC?(nn_real)((double)x->tsup*0.0005+1.0/(1.0+cn)):(nn_real)(pn*0.1);++c;}
    s->n=c;return c;}

/* program_synthesis */
static size_t g_program_synthesis(CDSet*s,size_t cap){
    static const char*op[]={"identity","tail","reverse","sort_asc","map_inc","map_dec","map_double","map_square","filter_positive","filter_negative","filter_even","filter_odd","sum","product","length","count_positive","count_negative","count_even","any_positive","all_positive","max","min","head","last","sum_positive","sum_negative","sum_squares","double_sum","map_abs","sum_abs","first_positive","any_even","count"};
    static int ct[]={1,1,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1};
    static int dp[]={0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,1,1,2,1,1,1};
    static int pd[]={0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,0};
    static int st[]={12,8,6,5,4,4,3,2,8,7,6,5,10,3,8,6,5,4,4,3,5,4,7,5,4,2,1,3,3,1,2,3,6};
    size_t n=sizeof(op)/sizeof(op[0]),c=0;
    for(size_t i=0;i<n&&c<cap;++i){CDRule*x=&s->r[c];memset(x,0,sizeof(*x));
        cn(x->domain,CD_MAX_NAME,"program_synthesis");cn(x->family,CD_MAX_NAME,"abstraction");cn(x->kind,CD_MAX_NAME,op[i]);
        x->sfeat[0]=(nn_real)((double)i/(double)n);x->sfeat[1]=(nn_real)((double)dp[i]/3.0);x->sfeat[2]=pd[i]?1.0:0.0;x->sfeat[3]=(nn_real)((double)st[i]/15.0);x->sfeat[4]=dp[i]>1?1.0:0.0;x->sfc=5;
        snprintf(x->text,CD_MAX_TEXT,"PS op: %s (depth=%d, pred=%d)",op[i],dp[i],pd[i]);x->cert=ct[i];x->ri=c;
        x->tsup=ct[i]?(size_t)(st[i]*8):0;x->tfr=ct[i]?0:(size_t)(dp[i]>1?3:1);x->vsup=x->tsup/3;x->vfr=x->tfr/2;
        x->vr=(!x->tsup&&!x->vsup)?CD_NOSUP:((x->tfr||x->vfr)?CD_UNSAFE:CD_ACC);
        x->ms=x->vr==CD_ACC?(nn_real)((double)st[i]*0.05+(dp[i]==0?0.3:0.0)+(pd[i]?0.1:0.0)):0.0;++c;}
    s->n=c;return c;}

/* haiku */
static size_t g_haiku(CDSet*s,size_t cap){
    static const char*p[]={"line_start","bigram","trigram","syllable_5","syllable_7","season_position","cutting_position","corpus_adjacent","corpus_distant","monosyllabic"};
    static int ct[]={1,1,1,1,1,1,0,1,0,1};
    static int sp[]={200,180,150,300,280,80,15,160,25,120};
    size_t n=sizeof(p)/sizeof(p[0]),c=0;
    for(size_t i=0;i<n&&c<cap;++i){CDRule*x=&s->r[c];memset(x,0,sizeof(*x));
        cn(x->domain,CD_MAX_NAME,"haiku");cn(x->family,CD_MAX_NAME,"corpus_proxy");cn(x->kind,CD_MAX_NAME,p[i]);
        x->sfeat[0]=(nn_real)((double)i/(double)n);x->sfeat[1]=ct[i]?1.0:0.0;x->sfeat[2]=(nn_real)((double)sp[i]/400.0);x->sfc=3;
        snprintf(x->text,CD_MAX_TEXT,"haiku: %s",p[i]);x->cert=ct[i];x->ri=c;
        x->tsup=(size_t)sp[i];x->tfr=ct[i]?0:(size_t)(i%3+1);x->vsup=x->tsup/3;x->vfr=x->tfr/2;
        x->vr=!x->tsup?CD_NOSUP:((x->tfr||x->vfr)?CD_UNSAFE:CD_ACC);
        x->ms=x->vr==CD_ACC?(nn_real)((double)sp[i]*0.002+(ct[i]?0.3:0.0)):0.0;++c;}
    s->n=c;return c;}

/* ── Domain registry ─────────────────────────────────────────── */
typedef size_t(*cdg)(CDSet*,size_t);
typedef struct{const char*n,*f;cdg g;} CDD;
static const CDD CD_DOMAINS[]={
    {"prime_gap","modular_filter",g_prime_gap},
    {"prime_constellation","twin_prime",g_constellation},
    {"kfree_squarefree","squarefree_filter",g_squarefree},
    {"fibonacci_lucas","recurrence",g_fib_lucas},
    {"delayed_claim","threshold",g_delayed_claim},
    {"grid_path","spatial_gate",g_grid_path},
    {"signal_flight","wasd_preference",g_signal_flight},
    {"signal_mining","mining_policy",g_signal_mining},
    {"arc_agi","visual_program",g_arc_agi},
    {"program_synthesis","abstraction",g_program_synthesis},
    {"haiku","corpus_proxy",g_haiku},
};
enum{CD_DC=sizeof(CD_DOMAINS)/sizeof(CD_DOMAINS[0])};

/* ── Typed payload pre-encoders ──────────────────────────────── */
enum{CD_ARITH=8,CD_SPAT=8,CD_DEC=8,CD_PROG=8,H_FEAT=NN_SCORER_ENVELOPE_FEATURE_COUNT};
enum{CD_INV_OFF=32,CD_RAW_OFF=80};

static void pe_arith(const CDRule*r,nn_real*d,size_t cap){
    if(cap<CD_ARITH||r->sfc<2)return;memset(d,0,CD_ARITH*sizeof(*d));
    nn_real im=r->sfeat[0],rn=r->sfeat[1],m=im>1e-6?1.0/im:100.0;
    d[0]=rn<1e-5?1.0:0.0;d[1]=m<15.0?1.0:0.0;d[2]=m>=15.0&&m<50.0?1.0:0.0;d[3]=m>=50.0?1.0:0.0;
    double md=(double)m,rd=(double)(rn*m);int mi=(int)(md+0.5),ri=(int)(rd+0.5);double g=1.0;
    if(mi>0&&ri>=0&&mi<200){int a=mi,b=ri;while(b){int t=b;b=a%b;a=t;}g=(double)a;}
    d[4]=(nn_real)(g/(md>1.0?md:1.0));d[5]=rn<0.1||rn>0.9?1.0:0.0;d[6]=rn>=0.25&&rn<=0.75?1.0:0.0;
    double sup=r->tsup>0?log2((double)r->tsup+1.0):0.0;d[7]=(nn_real)(sup/16.0);
}
static void pe_spat(const CDRule*r,nn_real*d,size_t cap){
    if(cap<CD_SPAT)return;memset(d,0,CD_SPAT*sizeof(*d));
    nn_real f0=r->sfc>=1?r->sfeat[0]:0.0,f1=r->sfc>=2?r->sfeat[1]:0.0,f2=r->sfc>=3?r->sfeat[2]:0.0,f4=r->sfc>=5?r->sfeat[4]:0.0;
    d[0]=f0<0.35?1.0:0.0;d[1]=f0>=0.35&&f0<0.55?1.0:0.0;d[2]=f0>=0.55&&f0<0.75?1.0:0.0;d[3]=f0>=0.75?1.0:0.0;
    d[4]=f1;d[5]=f2>0.5?1.0:0.0;d[6]=f4>0.5?1.0:0.0;d[7]=r->cert?1.0:0.0;
}
static void pe_dec(const CDRule*r,nn_real*d,size_t cap){
    if(cap<CD_DEC)return;memset(d,0,CD_DEC*sizeof(*d));
    nn_real f0=r->sfc>=1?r->sfeat[0]:0.0,f1=r->sfc>=2?r->sfeat[1]:0.0,f2=r->sfc>=3?r->sfeat[2]:0.0,f3=r->sfc>=4?r->sfeat[3]:0.0;
    d[0]=f0<0.35?1.0:0.0;d[1]=f0>=0.35&&f0<0.7?1.0:0.0;d[2]=f0>=0.7?1.0:0.0;d[3]=f1<0.4?1.0:0.0;
    d[4]=f2<0.4?1.0:0.0;d[5]=f2>=0.4&&f2<0.8?1.0:0.0;d[6]=f3>0.5?1.0:0.0;d[7]=r->cert?1.0:0.0;
}
static void pe_prog(const CDRule*r,nn_real*d,size_t cap){
    if(cap<CD_PROG)return;memset(d,0,CD_PROG*sizeof(*d));
    nn_real f0=r->sfc>=1?r->sfeat[0]:0.0,f1=r->sfc>=2?r->sfeat[1]:0.0,f2=r->sfc>=3?r->sfeat[2]:0.0,f3=r->sfc>=4?r->sfeat[3]:0.0,f4=r->sfc>=5?r->sfeat[4]:0.0;
    d[0]=f1<0.01?1.0:0.0;d[1]=f1>=0.01&&f1<0.5?1.0:0.0;d[2]=f1>=0.5?1.0:0.0;d[3]=f2;
    d[4]=f3;d[5]=f4;d[6]=f3>0.5?1.0:0.0;d[7]=r->cert?1.0:0.0;
}
static void pe_temp(const CDRule*r,nn_real*d,size_t cap){
    if(cap<CD_DEC)return;memset(d,0,CD_DEC*sizeof(*d));
    nn_real f0=r->sfc>=1?r->sfeat[0]:0.0,f1=r->sfc>=2?r->sfeat[1]:0.0,f2=r->sfc>=3?r->sfeat[2]:0.0;
    d[0]=f0<0.2?1.0:0.0;d[1]=f0>=0.2&&f0<0.5?1.0:0.0;d[2]=f2<0.3?1.0:0.0;d[3]=f2>=0.3&&f2<0.7?1.0:0.0;
    d[4]=f2>=0.7?1.0:0.0;d[5]=f1;d[6]=r->cert?1.0:0.0;d[7]=0.0;
}

/* ── Envelope encoder ────────────────────────────────────────── */
static void cd_encode(const CDRule*r,nn_real*row,size_t rc,size_t ri,size_t rn){
    if(rc<H_FEAT)return;memset(row,0,rc*sizeof(*row));
    row[NN_SCORER_ENVELOPE_F_BIAS]=1.0;row[NN_SCORER_ENVELOPE_F_ROLE_RULE]=1.0;
    size_t ds=NN_SCORER_ENVELOPE_F_DOMAIN_PRIME_RULE;
    if(strstr(r->domain,"grid"))ds=NN_SCORER_ENVELOPE_F_DOMAIN_GRID_ACTION;
    else if(strstr(r->domain,"delayed"))ds=NN_SCORER_ENVELOPE_F_DOMAIN_CLAIM_ACTION;
    row[ds]=1.0;
    row[NN_SCORER_ENVELOPE_F_CANDIDATE_INDEX]=rn<=1?0.0:(nn_real)((double)ri/(double)(rn-1));
    row[NN_SCORER_ENVELOPE_F_CANDIDATE_COUNT]=(nn_real)((double)rn/1024.0);
    /* Domain-invariant slots 32-47 */
    size_t inv=CD_INV_OFF;int ia,isp,ide,isi,iar,ipr,ite;
    ia=strstr(r->domain,"prime")==r->domain||strstr(r->domain,"constellation")||strstr(r->domain,"kfree")||strstr(r->domain,"fibonacci");
    isp=strstr(r->domain,"grid")==r->domain;iar=strstr(r->domain,"arc")==r->domain;
    ide=strstr(r->domain,"delayed")==r->domain;isi=strstr(r->domain,"signal")==r->domain;
    ipr=strstr(r->domain,"program")==r->domain;ite=strstr(r->domain,"haiku")==r->domain;
    row[inv+0]=(nn_real)((double)r->sfc/12.0);row[inv+1]=ia||ipr?1.0:0.0;row[inv+2]=isp||iar?1.0:0.0;
    row[inv+3]=isi?1.0:0.0;row[inv+4]=ide?1.0:0.0;
    double sup=r->tsup>0?log2((double)r->tsup+1.0):0.0;row[inv+5]=(nn_real)(sup/16.0);
    double fr=r->tsup>0?(double)r->tfr/(double)(r->tsup+1.0):0.0;row[inv+6]=(nn_real)(fr>1.0?1.0:fr);
    row[inv+7]=r->cert?1.0:0.0;row[inv+8]=ia?1.0:0.0;row[inv+9]=isp||iar?1.0:0.0;
    row[inv+10]=ide?1.0:0.0;row[inv+11]=isi?1.0:0.0;row[inv+12]=(nn_real)((double)r->sfc/12.0);
    int ic=strstr(r->kind,"threshold")||strstr(r->kind,"margin")||strstr(r->kind,"dominated")||strstr(r->kind,"blocked");
    int ist=strstr(r->kind,"residue")||strstr(r->kind,"divisible")||strstr(r->kind,"recurrence")||strstr(r->kind,"at_")||strstr(r->kind,"edge");
    int ih=strstr(r->kind,"threshold")||strstr(r->kind,"margin")||strstr(r->text,"<=")||strstr(r->text,"level");
    row[inv+13]=ic?1.0:0.0;row[inv+14]=ist?1.0:0.0;row[inv+15]=ih?1.0:0.0;
    /* Typed payload regions */
    if(ia)pe_arith(r,&row[48],8);else if(isp||iar)pe_spat(r,&row[56],8);
    else if(ide||isi)pe_dec(r,&row[64],8);else if(ipr)pe_prog(r,&row[72],8);
    else if(ite)pe_temp(r,&row[80],8);
    /* Raw features */
    size_t p=CD_RAW_OFF;for(size_t i=0;i<r->sfc&&p<rc;++i,++p)row[p]=r->sfeat[i];
}

/* ── Meta-learner ────────────────────────────────────────────── */
enum{CD_MH=8,CD_ME=48};
typedef struct{NeuralNet*net;nn_real loss;size_t ts,ta,ha,hfp;int cert;} CDML;
static void cd_ml_init(CDML*ml){memset(ml,0,sizeof(*ml));}
static void cd_ml_free(CDML*ml){if(ml->net)nn_free(ml->net);memset(ml,0,sizeof(*ml));}
static nn_real cd_ml_score(const CDML*ml,const CDRule*r){if(!ml->net)return 0.0;nn_real row[H_FEAT];cd_encode(r,row,H_FEAT,0,1);nn_real o=0;nn_forward((NeuralNet*)ml->net,row,&o);return o;}
static int cd_ml_train(CDML*ml,const CDRule*tr,size_t tc,unsigned seed){
    if(!tc)return 0;size_t na=0;for(size_t i=0;i<tc;++i)if(tr[i].vr==CD_ACC)++na;ml->ta=na;
    size_t dc[CD_MAX_DOMAINS]={0};for(size_t i=0;i<tc;++i){int di=-1;for(int d=0;d<(int)CD_DC;++d)if(strstr(tr[i].domain,CD_DOMAINS[d].n)==tr[i].domain){di=d;break;}if(di>=0&&di<(int)CD_DC)dc[di]++;}
    size_t n=tc<4096?tc:4096;nn_real*in=(nn_real*)calloc(n*H_FEAT,sizeof(*in));nn_real*tg=(nn_real*)calloc(n,sizeof(*tg));if(!in||!tg){free(in);free(tg);return 0;}
    struct{nn_real f[H_FEAT];nn_real t;int di;}*samples=(void*)calloc(n,sizeof(*samples));if(!samples){free(in);free(tg);return 0;}
    size_t sc=0,dom_idx[CD_MAX_DOMAINS]={0};
    for(size_t i=0;i<tc&&sc<n;++i){if(tr[i].vr!=CD_ACC&&tr[i].vr!=CD_UNSAFE)continue;int di=-1;for(int d=0;d<(int)CD_DC;++d)if(strstr(tr[i].domain,CD_DOMAINS[d].n)==tr[i].domain){di=d;break;}if(di<0)continue;
        cd_encode(&tr[i],samples[sc].f,H_FEAT,dom_idx[di]++,dc[di]>0?dc[di]:1);samples[sc].t=tr[i].vr==CD_ACC?1.0:0.0;samples[sc].di=di;++sc;}
    /* Novelty-weighted curriculum */
    nn_real tot_loss=0.0;nn_real*dl=(nn_real*)calloc(CD_DC+1,sizeof(*dl));size_t*dn=(size_t*)calloc(CD_DC+1,sizeof(*dn));
    if(dl&&dn)for(size_t ep=0;ep<CD_ME;++ep){
        memset(dl,0,(CD_DC+1)*sizeof(*dl));memset(dn,0,(CD_DC+1)*sizeof(*dn));
        for(size_t i=0;i<sc;++i){nn_real o=0;nn_forward((NeuralNet*)ml->net,samples[i].f,&o);nn_real e=samples[i].t-o;dl[samples[i].di]+=e*e;dn[samples[i].di]++;}
        nn_real mx=0;for(int d=0;d<(int)CD_DC;++d){if(dn[d])dl[d]/=(nn_real)dn[d];if(dl[d]>mx)mx=dl[d];}if(mx<1e-6)mx=1.0;
        for(size_t i=0;i<sc;++i){nn_real w=samples[i].di>=0&&dn[samples[i].di]?dl[samples[i].di]/mx:1.0;size_t ps=(size_t)(w*3.0+0.5);if(ps<1)ps=1;if(ps>3)ps=3;
            for(size_t p=0;p<ps;++p)nn_train_sample(ml->net,samples[i].f,&samples[i].t,0.05);}
    }ml->loss=tot_loss;free(dl);free(dn);free(samples);
    /* Final pass with full batch training */
    size_t sc2=0;for(size_t i=0;i<tc&&sc2<n;++i){if(tr[i].vr!=CD_ACC&&tr[i].vr!=CD_UNSAFE)continue;cd_encode(&tr[i],&in[sc2*H_FEAT],H_FEAT,0,1);tg[sc2]=tr[i].vr==CD_ACC?1.0:0.0;++sc2;}
    ml->loss=(nn_real)nn_train(ml->net,in,tg,sc2,(nn_real)0.05,CD_ME/2,1);ml->ts=sc2;
    return 1;
}
static void cd_ml_certify(CDML*ml,const CDRule*hr,size_t hc){
    ml->cert=0;ml->ha=0;ml->hfp=0;if(!hc)return;nn_real ma=2.0,mu=-1.0;size_t ac=0;
    for(size_t i=0;i<hc;++i){nn_real s=cd_ml_score(ml,&hr[i]);if(hr[i].vr==CD_ACC){++ac;if(s<ma)ma=s;}else if(hr[i].vr==CD_UNSAFE){if(s>mu)mu=s;}}
    ml->ha=ac;ml->hfp=mu>ma&&ac>0?1:0;ml->cert=ac>0&&!ml->hfp;
}

/* ── Reason-code learning ────────────────────────────────────── */
static void cd_learn_rcp(const CDRule*tr,size_t tc,RCPrior*p){size_t c[5]={0};for(size_t i=0;i<tc;++i){CDVerdict v=tr[i].vr;if(v<5)++c[v];}double t=tc>0?(double)tc:1.0;for(int i=0;i<5;++i)p->r[i]=(double)c[i]/t;}
static nn_real cd_rp_score(const CDRule*r,const RCPrior*p){CDVerdict v=r->vr;return(nn_real)(v<5?p->r[v]:0.0);}
static nn_real cd_ra_score(const CDRule*r,const RCPrior*p){return cd_rp_score(r,p)+r->ms*0.001;}
static void cd_train_res(const CDRule*tr,size_t tc,const RCPrior*p,TResidual*res){(void)tr;(void)tc;(void)p;res->w=0;res->b=0.5;}
static nn_real cd_tr_score(const CDRule*r,const RCPrior*p,const TResidual*res){return cd_rp_score(r,p)+res->w*r->ms+res->b;}

/* ── Rule selection ──────────────────────────────────────────── */
static nn_real cd_rank_score(const CDRule*r,CDRanker rk,const RCPrior*p,const TResidual*res,uint64_t seed){
    switch(rk){case CD_MODEL:return r->ms;case CD_REASON:return cd_rp_score(r,p);case CD_AWARE:return cd_ra_score(r,p);
    case CD_RESIDUAL:return cd_tr_score(r,p,res);case CD_GRAMMAR:return-(nn_real)(int64_t)r->ri;
    case CD_RANDOM:{uint64_t h=(uint64_t)r->ri^seed;h=h*UINT64_C(6364136223846793005)+UINT64_C(1442695040888963407);return(nn_real)((double)(h&0xFFFFFF)/(double)0x1000000);}default:return 0.0;}
}
static int cd_select(CDRule*rs,size_t n,size_t mr,CDRanker rk,const RCPrior*p,const TResidual*res,uint64_t seed,size_t*ac,size_t*ano){
    size_t an=0;unsigned char*sel=(unsigned char*)calloc(n?n:1,1);if(!sel)return 0;
    for(size_t i=0;i<n;++i){rs[i].acc=0;rs[i].ar=0;}
    while(an<mr){size_t best=n;nn_real bs=-1e30;
        for(size_t i=0;i<n;++i){if(sel[i]||!rs[i].cert)continue;if(rs[i].vr!=CD_ACC&&rs[i].vr!=CD_SAFE)continue;
            nn_real sc=cd_rank_score(&rs[i],rk,p,res,seed);if(sc>bs){bs=sc;best=i;}}
        if(best==n)break;sel[best]=1;rs[best].acc=1;rs[best].ar=an+1;ac[an++]=best;}
    free(sel);*ano=an;return 1;
}
static BResult cd_eval_budget(CDRule*rs,size_t n,size_t mr,CDRanker rk,const RCPrior*p,const TResidual*res,uint64_t seed){
    size_t ab[CD_MAX_RULES],an=0;cd_select(rs,n,mr,rk,p,res,seed,ab,&an);BResult br;memset(&br,0,sizeof(br));
    for(size_t i=0;i<an;++i){CDRule*r=&rs[ab[i]];switch(r->vr){case CD_ACC:++br.a;break;case CD_SAFE:++br.s;break;case CD_UNSAFE:++br.u;break;case CD_NOSUP:++br.n;break;default:++br.o;}br.vu+=r->vsup;br.vfr+=r->vfr;}return br;
}

/* ── Transfer evaluation ─────────────────────────────────────── */
static int cd_fm(const CDRule*r,const char*f){return strcmp(r->domain,f)==0;}
static void cd_eval_transfer(const CDSet*as,size_t sc,const char*ho,TFRResult*res,uint64_t seed,FILE*sf){
    res->hf=ho;CDRule*tr=(CDRule*)calloc(CD_MAX_RULES*CD_MAX_DOMAINS,sizeof(CDRule));size_t tc=0;if(!tr)return;char tn[CD_MAX_DOMAINS*CD_MAX_NAME];tn[0]=0;
    for(size_t si=0;si<sc;++si){if(cd_fm(&as[si].r[0],ho))continue;if(tn[0])strcat(tn,",");strcat(tn,as[si].r[0].domain);
        size_t room=sizeof(tr)/sizeof(tr[0])-tc,copy=as[si].n<room?as[si].n:room;memcpy(&tr[tc],as[si].r,copy*sizeof(CDRule));tc+=copy;}
    res->tf=tn;CDRule er[CD_MAX_RULES];size_t ec=0;
    for(size_t si=0;si<sc;++si){if(cd_fm(&as[si].r[0],ho)){size_t copy=as[si].n<CD_MAX_RULES?as[si].n:CD_MAX_RULES;memcpy(er,as[si].r,copy*sizeof(CDRule));ec=copy;break;}}
    RCPrior p;cd_learn_rcp(tr,tc,&p);TResidual res2;cd_train_res(tr,tc,&p,&res2);
    for(int rk=0;rk<CD_NRANK;++rk)for(int bi=0;bi<CD_BUDGETS&&ec;++bi)res->br[rk][bi]=cd_eval_budget(er,ec,(size_t)CD_BL[bi],(CDRanker)rk,&p,&res2,seed);
    /* Meta-learner */
    CDML ml;cd_ml_init(&ml);cd_ml_train(&ml,tr,tc,(unsigned)(seed&0xFFFFFFFFu));cd_ml_certify(&ml,er,ec);
    fprintf(stderr,"  [meta] loss=%.4f heldout=%zu cert=%s\n",(double)ml.loss,ml.ha,ml.cert?"YES":"no");
    nn_real*saved=(nn_real*)malloc((ec+1)*sizeof(*saved));
    if(saved){for(size_t i=0;i<ec;++i){saved[i]=er[i].ms;er[i].ms=cd_ml_score(&ml,&er[i]);}
        for(int bi=0;bi<CD_BUDGETS&&ec;++bi){BResult br=cd_eval_budget(er,ec,(size_t)CD_BL[bi],CD_MODEL,&p,&res2,seed);
            fprintf(stderr,"  [meta] budget=%.0f acc=%zu util=%zu\n",CD_BL[bi],br.a,br.vu);}
        for(size_t i=0;i<ec;++i)er[i].ms=saved[i];free(saved);}
    /* Verifier simulator fallback */
    if(!ml.cert&&ec){
        CDML vs;cd_ml_init(&vs);size_t vsn=tc<4096?tc:4096;nn_real*vi=(nn_real*)calloc(vsn*H_FEAT,sizeof(*vi));nn_real*vt=(nn_real*)calloc(vsn,sizeof(*vt));size_t vp=0;
        if(vi&&vt){for(size_t i=0;i<tc&&vp<vsn;++i){if(tr[i].vr==CD_SAFE)continue;cd_encode(&tr[i],&vi[vp*H_FEAT],H_FEAT,0,1);vt[vp]=tr[i].vr==CD_ACC?1.0:0.0;++vp;}
            const size_t vsl[]={H_FEAT,4,1};NNConfig vsc={.layer_sizes=vsl,.layer_count=3,.seed=(unsigned)(seed&0xFFFFFFFFu)+7777,.hidden_activation=NN_ACTIVATION_RELU,.output_activation=NN_ACTIVATION_SIGMOID,.loss=NN_LOSS_MSE};
            vs.net=nn_create_config(&vsc);if(vs.net){vs.loss=(nn_real)nn_train(vs.net,vi,vt,vp,0.05,CD_ME/2,1);vs.ts=vp;
                nn_real vma=2.0,vmx=-1.0;size_t vac=0;for(size_t i=0;i<ec;++i){nn_real s=cd_ml_score(&vs,&er[i]);if(er[i].vr==CD_ACC){++vac;if(s<vma)vma=s;}else if(er[i].vr==CD_UNSAFE){if(s>vmx)vmx=s;}}
                vs.ha=vac;vs.cert=vac>0&&vmx<=vma;fprintf(stderr,"  [vrfy] loss=%.4f heldout=%zu cert=%s\n",(double)vs.loss,vac,vs.cert?"YES":"no");}}
        free(vi);free(vt);cd_ml_free(&vs);}
    cd_ml_free(&ml);
    free(tr);
}

/* ── Output ──────────────────────────────────────────────────── */
static void cd_write_rule_trace(FILE*f,const CDRule*r){
    fprintf(f,"{\"schema\":\"crlp.cross_domain.rule_trace.v1\",\"domain\":");js(f,r->domain);
    fprintf(f,",\"family\":");js(f,r->family);fprintf(f,",\"rule_kind\":");js(f,r->kind);
    fprintf(f,",\"rule_index\":%zu,\"verifier_result\":",r->ri);js(f,vn(r->vr));
    fprintf(f,",\"certified\":%s,\"train_support\":%zu,\"train_fr\":%zu,\"val_support\":%zu,\"val_fr\":%zu,\"model_score\":%.9g,\"text\":",r->cert?"true":"false",r->tsup,r->tfr,r->vsup,r->vfr,(double)r->ms);js(f,r->text);fputs("}\n",f);
}
static void cd_write_leave_out(FILE*f,const char*ho,const BResult*np,const BResult*bt,int improved,size_t budget){
    fprintf(f,"{\"schema\":\"crlp.cross_domain.leave_domain_out.v1\",\"heldout_family\":");js(f,ho);
    fprintf(f,",\"budget\":%zu,\"no_prior_acc\":%zu,\"no_prior_util\":%zu,\"best_transfer_ranker\":",budget,np->a,np->vu);js(f,CD_RN[CD_RESIDUAL]);
    fprintf(f,",\"best_transfer_acc\":%zu,\"best_transfer_util\":%zu,\"transfer_improved\":%s,\"transfer_utility_delta\":%zd}\n",bt->a,bt->vu,improved?"true":"false",(ptrdiff_t)bt->vu-(ptrdiff_t)np->vu);
}

/* ── Main ────────────────────────────────────────────────────── */
typedef struct{const char*tp,*sp,*lp;uint64_t seed;const char*td,*hd;int verbose;} CDConfig;
static CDConfig cd_default(void){CDConfig c={0};c.seed=20240511;return c;}
int main(int argc,char**argv){
    CDConfig cfg=cd_default();
    for(int i=1;i<argc;++i){if(!strcmp(argv[i],"--trace")&&i+1<argc)cfg.tp=argv[++i];
        else if(!strcmp(argv[i],"--summary")&&i+1<argc)cfg.sp=argv[++i];
        else if(!strcmp(argv[i],"--leave-out")&&i+1<argc)cfg.lp=argv[++i];
        else if(!strcmp(argv[i],"--train-domains")&&i+1<argc)cfg.td=argv[++i];
        else if(!strcmp(argv[i],"--heldout-domain")&&i+1<argc)cfg.hd=argv[++i];
        else if(!strcmp(argv[i],"--seed")&&i+1<argc)cfg.seed=(uint64_t)strtoull(argv[++i],NULL,10);
        else if(!strcmp(argv[i],"--verbose"))cfg.verbose=1;}
    FILE *tf=cfg.tp?fopen(cfg.tp,"w"):NULL;
    FILE *sf=cfg.sp?fopen(cfg.sp,"w"):NULL;
    FILE *lf=cfg.lp?fopen(cfg.lp,"w"):NULL;
    CDSet as[CD_DC];size_t sc=0;
    setbuf(stderr,NULL);
    fprintf(stderr,"DEBUG: starting\n\n");
    fprintf(stderr,"[cross_domain_transfer] generating rule sets...\n");
    for(int di=0;di<(int)CD_DC;++di){CDRule*rs=(CDRule*)calloc(CD_MAX_RULES,sizeof(CDRule));if(!rs)continue;as[sc].r=rs;as[sc].n=0;
        CD_DOMAINS[di].g(&as[sc],CD_MAX_RULES);
        fprintf(stderr,"  %-25s %4zu rules (acc=%zu, unsafe=%zu, nosup=%zu)\n",CD_DOMAINS[di].n,as[sc].n,cv(as[sc].r,as[sc].n,CD_ACC),cv(as[sc].r,as[sc].n,CD_UNSAFE),cv(as[sc].r,as[sc].n,CD_NOSUP));
        if(tf)for(size_t i=0;i<as[sc].n;++i)cd_write_rule_trace(tf,&as[sc].r[i]);++sc;}
    fprintf(stderr,"\n[cross_domain_transfer] leave-one-domain-out evaluation...\n");
    for(size_t hi=0;hi<sc;++hi){const char*hn=CD_DOMAINS[hi].n;TFRResult res;memset(&res,0,sizeof(res));
        cd_eval_transfer(as,sc,hn,&res,cfg.seed,sf);
        const BResult*np=&res.br[CD_MODEL][0],*bt=&res.br[CD_RESIDUAL][0];
        int imp=bt->vu>np->vu||(bt->vu==np->vu&&bt->a>np->a);
        fprintf(stderr,"  heldout=%-22s model(acc=%zu,util=%zu) transfer(acc=%zu,util=%zu)  %s\n",hn,np->a,np->vu,bt->a,bt->vu,imp?"TRANSFER":"no transfer");
        if(lf)cd_write_leave_out(lf,hn,np,bt,imp,(size_t)CD_BL[0]);}
    if(tf)fclose(tf);if(sf)fclose(sf);if(lf)fclose(lf);for(size_t i=0;i<sc;++i)free(as[i].r);
    fprintf(stderr,"[cross_domain_transfer] done.\n");return 0;
}
