#include "nn_text_vsa.h"
#include "nn_holographic_ranker.h"
#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HAIKU_DIM 1024
#define MAX_POEMS 5000
#define MAX_TOKENS 32
#define MAX_TLEN 64
#define MAX_VOCAB 8000
#define MAX_CLAIMS 8
#define N_CTX 4
#define TOP_N 5

typedef struct { char t[MAX_TLEN]; uint64_t seed; } VEntry;
typedef struct { int ids[MAX_TOKENS]; size_t nc; char *text; int ctx; } Poem;

static uint64_t fh(const char *s) {
    uint64_t h = UINT64_C(14695981039346656037);
    for (const unsigned char *p = (const unsigned char *)s; *p; ++p) {
        h ^= (uint64_t)*p; h *= UINT64_C(1099511628211);
    }
    return h;
}

static int va(const char *t, VEntry *v, size_t *n) {
    uint64_t h = fh(t);
    for (size_t i = 0; i < *n; ++i)
        if (v[i].seed == h && !strcmp(v[i].t, t)) return (int)i;
    if (*n >= MAX_VOCAB) return -1;
    size_t c = strlen(t); if (c >= MAX_TLEN) c = MAX_TLEN-1;
    memcpy(v[*n].t, t, c); v[*n].t[c] = 0; v[*n].seed = h;
    return (int)((*n)++);
}

static const char *CN[N_CTX] = {"spring","autumn","winter","summer"};
static const char *CT[N_CTX] = {
    "spring bloom green mild rain petal blossom fresh warm",
    "autumn fall brown crisp leaf harvest dusk cool gold",
    "winter frost cold bare snow ice still white grey",
    "summer heat warm bright sun blaze fire haze light"
};

static const char *CL[MAX_CLAIMS] = {
    "verified gate filter improves top one ranking over raw baseline",
    "shared scorer beats single task scorer on out of domain ranking",
    "holographic memory achieves exact retrieval on isolated pairs",
    "rejected candidate feedback loop improves proposal quality",
    "claim compiler mechanically checks every paper assertion",
    "modular residue filter proves zero false rejections over support",
    "cross domain transfer benefits from shared hypothesis library",
    "trace determinism audit catches replay sensitive claim drift"
};

static const char *CC[MAX_CLAIMS] = {
    "gate filter ranking verification baseline",
    "scorer transfer generalization domain multitask",
    "memory retrieval holographic storage vector",
    "feedback learning proposal rejection candidate",
    "compiler claim evidence paper assertion",
    "modular residue proof certificate false rejection",
    "transfer cross domain library hypothesis shared",
    "determinism audit trace replay drift"
};

static int clfy(const char *t) {
    int b = 0, bs = -1;
    char lo[4096]; size_t n = strlen(t);
    if (n > 4095) n = 4095;
    for (size_t i = 0; i < n; ++i) lo[i] = (char)tolower((unsigned char)t[i]);
    lo[n] = 0;
    for (int c = 0; c < N_CTX; ++c) {
        int s = 0; const char *p = CT[c];
        while (*p) {
            while (*p == ' ') ++p; if (!*p) break;
            const char *e = p; while (*e && *e != ' ') ++e;
            size_t el = (size_t)(e-p); char w[64];
            if (el > 63) el = 63; memcpy(w, p, el); w[el] = 0;
            if (strstr(lo, w)) ++s; p = e;
        }
        if (s > bs) { bs = s; b = c; }
    }
    return b;
}

static double dt(const double *a, const double *b, size_t d) {
    double s = 0; for (size_t i = 0; i < d; ++i) s += a[i]*b[i]; return s;
}

static void tk(const double *sc, size_t n, int k, int *out) {
    for (int i = 0; i < k; ++i) out[i] = -1;
    for (size_t i = 0; i < n; ++i) {
        int p = k-1;
        while (p >= 0 && (out[p] < 0 || sc[i] > sc[out[p]])) --p;
        ++p; if (p >= k) continue;
        for (int j = k-1; j > p; --j) out[j] = out[j-1];
        out[p] = (int)i;
    }
}

int main(void) {
    printf("=== Haiku + Claim Holographic VSA: Partitioned Spaces ===\n");
    printf("dim=%d poems=%d claims=%d\n\n", HAIKU_DIM, MAX_POEMS, MAX_CLAIMS);

    LeosEngine *eng = leos_engine_create(HAIKU_DIM);
    VEntry *vb = calloc(MAX_VOCAB, sizeof(VEntry));
    Poem *pm = calloc(MAX_POEMS, sizeof(Poem));
    size_t vn = 0, pn = 0;

    FILE *fp = fopen("data/haiku_synthetic_5000.txt", "r");
    if (!fp) { fprintf(stderr,"corpus fail\n"); free(vb); free(pm); leos_engine_destroy(eng); return 1; }
    char ln[1024], stz[4096] = ""; int sl = 0;
    while (fgets(ln, sizeof(ln), fp) && pn < MAX_POEMS) {
        size_t ll = strlen(ln);
        while (ll && (ln[ll-1]=='\n'||ln[ll-1]=='\r')) ln[--ll]=0;
        if (!ll) {
            if (sl == 3) {
                Poem *p = &pm[pn]; p->text = strdup(stz); p->ctx = clfy(stz); p->nc = 0;
                char *x = stz;
                while (*x && p->nc < MAX_TOKENS) {
                    while (*x == ' ') ++x; if (!*x) break;
                    char *e = x; while (*e && *e!=' '&&*e!='\n') ++e;
                    char sv = *e; *e = 0;
                    int id = va(x, vb, &vn); if (id >= 0) p->ids[p->nc++] = id;
                    *e = sv; x = sv ? e+1 : e;
                }
                ++pn;
            }
            sl = 0; stz[0] = 0; continue;
        }
        if (sl > 0) strcat(stz, " "); strcat(stz, ln); ++sl;
    }
    fclose(fp);
    printf("loaded %zu poems, %zu tokens\n\n", pn, vn);

    /* context vectors */
    double *cx = calloc(N_CTX * HAIKU_DIM, sizeof(double));
    for (int c = 0; c < N_CTX; ++c)
        nn_text_vsa_encode(eng, CT[c], &cx[c*HAIKU_DIM]);

    /* --- HAIKU RANKER --- */
    HolographicRanker *hr = holo_ranker_create(HAIKU_DIM);
    double **pv = calloc(pn, sizeof(double*));
    for (size_t i = 0; i < pn; ++i) {
        pv[i] = calloc(HAIKU_DIM, sizeof(double));
        uint64_t sd[MAX_TOKENS];
        for (size_t t = 0; t < pm[i].nc; ++t) sd[t] = vb[pm[i].ids[t]].seed;
        nn_text_vsa_encode_tokens(eng, pm[i].ids, pm[i].nc, sd, pv[i]);
        const double *c = &cx[pm[i].ctx * HAIKU_DIM];
        holo_ranker_store(hr, c, pv[i]);
    }
    double hf = holo_ranker_fidelity(hr);
    printf("--- Haiku space: %zu poems, fidelity=%.4f (1/sqrt(N)=%.4f) ---\n",
           pn, hf, 1.0/sqrt((double)pn));

    double *rc = calloc(HAIKU_DIM, sizeof(double));
    for (int c = 0; c < N_CTX; ++c) {
        holo_ranker_recall(hr, &cx[c*HAIKU_DIM], rc);
        double *sc = calloc(pn, sizeof(double));
        for (size_t i = 0; i < pn; ++i) sc[i] = dt(rc, pv[i], HAIKU_DIM);
        int tp[TOP_N]; tk(sc, pn, TOP_N, tp);
        printf("\n  %s:", CN[c]);
        for (int k = 0; k < TOP_N && tp[k] >= 0; ++k)
            printf("\n    [%s, %.3f] %s", CN[pm[tp[k]].ctx], sc[tp[k]], pm[tp[k]].text);
        printf("\n");
        free(sc);
    }

    /* --- CLAIM RANKER (separate space, clean fidelity) --- */
    HolographicRanker *cr = holo_ranker_create(HAIKU_DIM);
    double **cv = calloc(MAX_CLAIMS, sizeof(double*));
    double **cc = calloc(MAX_CLAIMS, sizeof(double*));
    for (int i = 0; i < MAX_CLAIMS; ++i) {
        cv[i] = calloc(HAIKU_DIM, sizeof(double));
        cc[i] = calloc(HAIKU_DIM, sizeof(double));
        nn_text_vsa_encode(eng, CL[i], cv[i]);
        nn_text_vsa_encode(eng, CC[i], cc[i]);
        holo_ranker_store(cr, cc[i], cv[i]);
    }
    printf("\n--- Claim space: %d claims, fidelity=%.4f (clean, no poem noise) ---\n",
           MAX_CLAIMS, holo_ranker_fidelity(cr));

    for (int q = 0; q < 3; ++q) {
        holo_ranker_recall(cr, cc[q], rc);
        double *sc = calloc(MAX_CLAIMS, sizeof(double));
        for (int i = 0; i < MAX_CLAIMS; ++i) sc[i] = dt(rc, cv[i], HAIKU_DIM);
        int tp[TOP_N]; tk(sc, MAX_CLAIMS, TOP_N, tp);
        printf("\n  query: \"%s\"\n  ctx:   \"%s\"\n", CL[q], CC[q]);
        for (int k = 0; k < TOP_N && tp[k] >= 0; ++k)
            printf("    #%d [%.3f] %s\n", k+1, sc[tp[k]], CL[tp[k]]);
        free(sc);
    }

    /* --- Cross-domain fan-out --- */
    printf("\n--- Cross-domain fan-out: \"pattern structure rule evidence proof\" ---\n");
    double *qv = calloc(HAIKU_DIM, sizeof(double));
    nn_text_vsa_encode(eng, "pattern structure rule evidence proof", qv);

    holo_ranker_recall(hr, qv, rc);
    double *hs = calloc(pn, sizeof(double));
    for (size_t i = 0; i < pn; ++i) hs[i] = dt(rc, pv[i], HAIKU_DIM);
    int ht[3]; tk(hs, pn, 3, ht);

    holo_ranker_recall(cr, qv, rc);
    double *cs = calloc(MAX_CLAIMS, sizeof(double));
    for (int i = 0; i < MAX_CLAIMS; ++i) cs[i] = dt(rc, cv[i], HAIKU_DIM);
    int ct[3]; tk(cs, MAX_CLAIMS, 3, ct);

    printf("  Haiku space (fid=%.4f):\n", hf);
    for (int k = 0; k < 3 && ht[k] >= 0; ++k)
        printf("    [%.3f] %s\n", hs[ht[k]], pm[ht[k]].text);
    printf("  Claim space (fid=%.4f):\n", holo_ranker_fidelity(cr));
    for (int k = 0; k < 3 && ct[k] >= 0; ++k)
        printf("    [%.3f] %s\n", cs[ct[k]], CL[ct[k]]);

    /* --- Determinism --- */
    printf("\n--- Similarity ---\n");
    double v1[HAIKU_DIM], v2[HAIKU_DIM];
    nn_text_vsa_encode(eng, "cold winter frost", v1);
    nn_text_vsa_encode(eng, "cold winter frost", v2);
    printf("self-sim 'cold winter frost':    %.4f\n", dt(v1,v2,HAIKU_DIM));
    nn_text_vsa_encode(eng, "warm summer breeze", v2);
    printf("'cold winter frost' vs 'warm summer breeze': %.4f\n", dt(v1,v2,HAIKU_DIM));
    nn_text_vsa_encode(eng, "cold winter night", v2);
    printf("'cold winter frost' vs 'cold winter night':  %.4f\n", dt(v1,v2,HAIKU_DIM));
    nn_text_vsa_encode(eng, "verified gate filter improves top one ranking", v1);
    nn_text_vsa_encode(eng, "modular residue filter proves zero false rejections", v2);
    printf("claim(verify+filter) vs claim(modular+filter):  %.4f\n", dt(v1,v2,HAIKU_DIM));

    for (size_t i = 0; i < pn; ++i) { free(pm[i].text); free(pv[i]); }
    for (int i = 0; i < MAX_CLAIMS; ++i) { free(cv[i]); free(cc[i]); }
    free(pm); free(vb); free(cx); free(pv); free(cv); free(cc);
    free(rc); free(hs); free(cs); free(qv);
    holo_ranker_destroy(hr); holo_ranker_destroy(cr); leos_engine_destroy(eng);
    return 0;
}
