#include "nn_bibliography.h"
#include "nn_text_vsa.h"
#include "nn_tfidf.h"
#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BIB_LINE_MAX 65536
#define AFFINITY_THRESHOLD 0.4

static int is_sentence_end(int c) { return c == '.' || c == '!' || c == '?'; }

static size_t split_sentences(const char *text, char ***out) {
    size_t cap = 128, n = 0;
    char **sents = calloc(cap, sizeof(char*));
    if (!sents) return 0;
    const char *p = text, *start = NULL;
    while (*p) {
        while (*p && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) p++;
        if (!*p) break;
        start = p;
        while (*p && !is_sentence_end(*p) && (size_t)(p-start) < 2000) {
            if (*p == '\n' && (p[1]=='\n'||p[1]=='\r')) break;
            p++;
        }
        if (*p == '.' || *p == '!' || *p == '?') p++;
        while (*p == ' ' || *p == '\n' || *p == '\r') p++;
        size_t len = (size_t)(p-start);
        if (len < 3) continue;
        while (len && (start[len-1]==' '||start[len-1]=='\n'||
               start[len-1]=='\r'||start[len-1]=='\t')) len--;
        if (len < 3) continue;
        if (n >= cap) { cap*=2; char **ns=realloc(sents,cap*sizeof(char*)); if(!ns){n=0;break;} sents=ns; }
        sents[n]=calloc(len+1,1); if(!sents[n])break;
        memcpy(sents[n],start,len); sents[n][len]='\0'; n++;
    }
    *out = sents; return n;
}

static void free_sentences(char **s, size_t n) {
    for (size_t i=0;i<n;i++) free(s[i]); free(s);
}

static const char *DEFAULT_TOPICS[] = {
    "verifier gated reasoning training reward model step by step correctness",
    "holographic memory hyperdimensional computing HDC vector symbolic architecture VSA binding retrieval superposition distribute represent",
    "formal methods synthesis grammar verification proof counterexample",
    "neurosymbolic scoring ranking program generation learned proposer",
    "machine learning accountability reproducibility cards datasheets audit",
    "reinforcement learning policy gradient preference optimization rlhf",
    "code generation formal theorem solver search alphageometry sketch",
    "calibration evaluation benchmark generalization measurement ood"
};

static void encode_text(const BibliographyStore *bib, LeosEngine *eng,
                        const char *text, double *out) {
    if (bib->tfidf_vocab) tfidf_encode(bib->tfidf_vocab, text, out);
    else nn_text_vsa_encode(eng, text, out);
}

static const char *GATE_CLAIMS[8] = {
    "verified gate filter improves top one ranking over raw baseline",
    "shared scorer beats single task scorer on out of domain ranking",
    "holographic memory achieves exact retrieval on isolated pairs",
    "rejected candidate feedback loop improves proposal quality",
    "claim compiler mechanically checks every paper assertion",
    "modular residue filter proves zero false rejections",
    "cross domain transfer benefits from shared hypothesis library",
    "trace determinism audit catches replay sensitive claim drift",
};
static const char *GATE_DOMAINS[8] = {
    "prime_gap verification ranking gate filter arithmetic",
    "multitask shared scorer generalization cross domain transfer",
    "holographic memory VSA retrieval storage binding",
    "rejected candidate learning feedback proposal loop",
    "claim compiler paper evidence assertion mechanical check",
    "modular arithmetic residue filter certificate proof",
    "cross domain transfer hypothesis library shared",
    "determinism audit trace replay reproducibility",
};
#define N_GATES 8
#define GATE_TOPK 3

static double find_best_span(const double *scores, size_t n, int *out_start, int *out_end);

BibliographyStore *bibliography_create(const char *manifest_path,
                                       const char *text_dir,
                                       const char *const *topic_contexts,
                                       size_t topic_count) {
    BibliographyStore *bib = calloc(1, sizeof(*bib));
    if (!bib) return NULL;
    if (!topic_contexts || topic_count == 0) { topic_contexts = DEFAULT_TOPICS; topic_count = 8; }
    if (topic_count > BIBLIOGRAPHY_MAX_TOPICS) topic_count = BIBLIOGRAPHY_MAX_TOPICS;
    bib->topic_count = topic_count;
    for (size_t t = 0; t < topic_count; t++)
        bib->sentence_layers[t] = holo_layer_create(BIBLIOGRAPHY_DIM, 8, BIBLIOGRAPHY_DIM, 0.005);
    bib->topic_vecs = calloc(topic_count, sizeof(double*));
    LeosEngine *eng = leos_engine_create(BIBLIOGRAPHY_DIM);
    for (size_t t = 0; t < topic_count; t++) {
        bib->topic_vecs[t] = calloc(BIBLIOGRAPHY_DIM, sizeof(double));
        encode_text(bib, eng, topic_contexts[t], bib->topic_vecs[t]);
        size_t n = strlen(topic_contexts[t]); if (n > 31) n = 31;
        memcpy(bib->topic_names[t], topic_contexts[t], n); bib->topic_names[t][n] = '\0';
    }
    leos_engine_destroy(eng);
    bib->term_bridge = text_mutator_create("data/term_thesaurus.tsv");
    memset(bib->paper_rankers, 0, sizeof(bib->paper_rankers));
    memset(bib->paper_centroids, 0, sizeof(bib->paper_centroids));

    char **paths = NULL; size_t np = 0, pc = 0;
    char *keys[BIBLIOGRAPHY_MAX_PAPERS]; size_t nk = 0;
    if (manifest_path) {
        FILE *mf = fopen(manifest_path, "r");
        if (mf) {
            char line[BIB_LINE_MAX];
            if (fgets(line, sizeof(line), mf)) {
                while (fgets(line, sizeof(line), mf) && np < BIBLIOGRAPHY_MAX_PAPERS) {
                    char *tab = strchr(line, '\t');
                    size_t bl = tab ? (size_t)(tab - line) : strlen(line);
                    while (bl && (line[bl-1]==' '||line[bl-1]=='\r')) bl--;
                    if (bl >= BIBLIOGRAPHY_BIBKEY_MAX) bl = BIBLIOGRAPHY_BIBKEY_MAX-1;
                    char *path = malloc(512);
                    snprintf(path, 512, "%s/%.*s.txt", text_dir, (int)bl, line);
                    if (np >= pc) { pc = pc ? pc*2 : 64; paths = realloc(paths, pc*sizeof(char*)); }
                    paths[np++] = path;
                    keys[nk] = malloc(bl+1); memcpy(keys[nk], line, bl); keys[nk][bl] = '\0'; nk++;
                }
            }
            fclose(mf);
        }
    }
    if (np > 0) bib->tfidf_vocab = tfidf_vocab_create((const char *const*)paths, np, BIBLIOGRAPHY_DIM);
    for (size_t i = 0; i < nk; i++) {
        BibPaper *bp = &bib->papers[bib->paper_count];
        memcpy(bp->bibkey, keys[i], strlen(keys[i])+1);
        bibliography_encode_paper(bib, bp->bibkey, paths[i]);
    }

    /* Train claim gateways */
    if (bib->tfidf_vocab && bib->paper_count > 0) {
        bib->n_gates = N_GATES;
        bib->claim_gates = calloc(N_GATES, sizeof(HolographicRanker*));
        bib->gate_claim_vecs = calloc(N_GATES, sizeof(double*));
        bib->gate_candidates = calloc(bib->paper_count * BIBLIOGRAPHY_DIM, sizeof(double));
        for (size_t i = 0; i < bib->paper_count; i++)
            memcpy(&bib->gate_candidates[i*BIBLIOGRAPHY_DIM], bib->papers[i].abstract_vec, BIBLIOGRAPHY_DIM*sizeof(double));

        LeosEngine *geng = leos_engine_create(BIBLIOGRAPHY_DIM);
        /* Encode claim vectors */
        for (size_t c = 0; c < N_GATES; c++) {
            char e[1024]; snprintf(e, sizeof(e), "domain %s claim %s", GATE_DOMAINS[c], GATE_CLAIMS[c]);
            bib->gate_claim_vecs[c] = calloc(BIBLIOGRAPHY_DIM, sizeof(double));
            tfidf_encode(bib->tfidf_vocab, e, bib->gate_claim_vecs[c]);
        }
        /* Gram-Schmidt orthogonalize */
        for (size_t c = 0; c < N_GATES; c++) {
            for (size_t p = 0; p < c; p++) {
                double proj = leos_engine_dot(geng, bib->gate_claim_vecs[c], bib->gate_claim_vecs[p]);
                for (size_t d = 0; d < BIBLIOGRAPHY_DIM; d++) bib->gate_claim_vecs[c][d] -= proj * bib->gate_claim_vecs[p][d];
            }
            leos_engine_normalize(geng, bib->gate_claim_vecs[c]);
        }
        /* Train each gate with rank-weighted positive + negative associations */
        for (size_t c = 0; c < N_GATES; c++) {
            bib->claim_gates[c] = holo_ranker_create(BIBLIOGRAPHY_DIM);
            char bk[GATE_TOPK][BIBLIOGRAPHY_BIBKEY_MAX]; double sc[GATE_TOPK];
            bibliography_query_claim(bib, GATE_CLAIMS[c], GATE_DOMAINS[c], bk, sc, GATE_TOPK);
            for (size_t k = 0; k < GATE_TOPK && bk[k][0]; k++) {
                int w = GATE_TOPK - (int)k;
                for (size_t pi = 0; pi < bib->paper_count; pi++)
                    if (!strcmp(bib->papers[pi].bibkey, bk[k]))
                        { for (int r=0; r<w; r++) holo_ranker_store(bib->claim_gates[c], bib->gate_claim_vecs[c], bib->papers[pi].abstract_vec); break; }
            }
            /* Negatives: papers scoring top-10 by gate but not in TF-IDF top-K */
            double *as = calloc(bib->paper_count, sizeof(double));
            holo_ranker_score(bib->claim_gates[c], bib->gate_claim_vecs[c], bib->gate_candidates, bib->paper_count, as);
            size_t *ord = calloc(bib->paper_count, sizeof(size_t));
            for (size_t i=0; i<bib->paper_count; i++) ord[i]=i;
            for (size_t i=0; i<bib->paper_count; i++) for (size_t j=i+1; j<bib->paper_count; j++) if (as[ord[i]]<as[ord[j]]){size_t t=ord[i];ord[i]=ord[j];ord[j]=t;}
            size_t ns = bib->paper_count<10?bib->paper_count:10;
            for (size_t si=0; si<ns; si++) {
                const char *gk = bib->papers[ord[si]].bibkey; int in=0;
                for (size_t k=0; k<GATE_TOPK; k++) if (!strcmp(gk,bk[k])){in=1;break;}
                if (in) continue;
                double *neg = calloc(BIBLIOGRAPHY_DIM, sizeof(double));
                for (size_t d=0; d<BIBLIOGRAPHY_DIM; d++) neg[d] = -bib->papers[ord[si]].abstract_vec[d];
                holo_ranker_store(bib->claim_gates[c], bib->gate_claim_vecs[c], neg);
                free(neg);
            }
            free(as); free(ord);
        }
        leos_engine_destroy(geng);
        bib->gates_ready = 1;
    }

    for (size_t i = 0; i < np; i++) free(paths[i]); free(paths);
    for (size_t i = 0; i < nk; i++) free(keys[i]);
    return bib;
}

void bibliography_destroy(BibliographyStore *bib) {
    if (!bib) return;
    for (size_t i = 0; i < bib->paper_count; i++) {
        free_sentences(bib->papers[i].sentences, bib->papers[i].sentence_count);
        free(bib->papers[i].abstract_vec); free(bib->papers[i].binding_vecs);
        free(bib->papers[i].binding_indices);
        if (bib->paper_rankers[i]) holo_ranker_destroy(bib->paper_rankers[i]);
        free(bib->paper_centroids[i]);
    }
    for (size_t t = 0; t < bib->topic_count; t++) holo_layer_destroy(bib->sentence_layers[t]);
    for (size_t t = 0; t < bib->topic_count; t++) free(bib->topic_vecs[t]); free(bib->topic_vecs);
    if (bib->term_bridge) text_mutator_destroy(bib->term_bridge);
    if (bib->tfidf_vocab) tfidf_vocab_destroy(bib->tfidf_vocab);
    for (size_t c = 0; c < bib->n_gates; c++) { if (bib->claim_gates[c]) holo_ranker_destroy(bib->claim_gates[c]); free(bib->gate_claim_vecs[c]); }
    free(bib->claim_gates); free(bib->gate_claim_vecs); free(bib->gate_candidates);
    free(bib);
}

int bibliography_encode_paper(BibliographyStore *bib, const char *bibkey, const char *text_path) {
    if (bib->paper_count >= BIBLIOGRAPHY_MAX_PAPERS) return -1;
    LeosEngine *eng = leos_engine_create(BIBLIOGRAPHY_DIM);
    if (!eng) return -1;
    FILE *fp = fopen(text_path, "r");
    if (!fp) { leos_engine_destroy(eng); return -1; }
    fseek(fp, 0, SEEK_END); long sz = ftell(fp);
    if (sz <= 0 || sz > 500000) { fclose(fp); leos_engine_destroy(eng); return -1; }
    fseek(fp, 0, SEEK_SET);
    char *buf = calloc((size_t)sz+1, 1);
    if (!buf) { fclose(fp); leos_engine_destroy(eng); return -1; }
    size_t nr = fread(buf, 1, (size_t)sz, fp); fclose(fp); buf[nr] = '\0';

    size_t paper_idx = bib->paper_count;
    BibPaper *bp = &bib->papers[paper_idx];
    size_t bl = strlen(bibkey); if (bl >= BIBLIOGRAPHY_BIBKEY_MAX) bl = BIBLIOGRAPHY_BIBKEY_MAX-1;
    memcpy(bp->bibkey, bibkey, bl); bp->bibkey[bl] = '\0';

    double *pv = calloc(BIBLIOGRAPHY_DIM, sizeof(double));
    encode_text(bib, eng, buf, pv);
    double ma = -1e9;
    for (size_t t = 0; t < bib->topic_count; t++) {
        bp->topic_affinities[t] = leos_engine_dot(eng, pv, bib->topic_vecs[t]);
        if (bp->topic_affinities[t] > ma) { ma = bp->topic_affinities[t]; bp->primary_topic = (int)t; }
    }
    bp->abstract_vec = pv;

    char **sents = NULL; size_t sn = split_sentences(buf, &sents); free(buf);
    if (sn > BIB_MAX_SENTENCES) sn = BIB_MAX_SENTENCES;
    if (sn == 0) { leos_engine_destroy(eng); return -1; }
    bp->sentences = sents; bp->sentence_count = sn;

    /* Encode each sentence with TF-IDF and score against primary topic.
     * Keep above-median sentences for storage in the holographic ranker. */
    double *sent_scores = calloc(sn, sizeof(double));
    for (size_t i = 0; i < sn; i++) {
        double *sv = calloc(BIBLIOGRAPHY_DIM, sizeof(double));
        encode_text(bib, eng, sents[i], sv);
        sent_scores[i] = leos_engine_dot(eng, sv, bib->topic_vecs[bp->primary_topic]);
        free(sv);
    }
    double *ssort = calloc(sn, sizeof(double));
    memcpy(ssort, sent_scores, sn*sizeof(double));
    for (size_t i = 0; i < sn; i++)
        for (size_t j = i+1; j < sn; j++)
            if (ssort[i] > ssort[j]) { double t=ssort[i]; ssort[i]=ssort[j]; ssort[j]=t; }
    double median = ssort[sn/2]; free(ssort);

    size_t bcount = 0;
    for (size_t i = 0; i < sn; i++) if (sent_scores[i] >= median) bcount++;
    bp->binding_count = bcount;
    bp->binding_vecs = calloc(bcount * BIBLIOGRAPHY_DIM, sizeof(double));
    bp->binding_indices = calloc(bcount, sizeof(int));

    size_t bi = 0;
    for (size_t i = 0; i < sn; i++) {
        if (sent_scores[i] < median) continue;
        double *sv = calloc(BIBLIOGRAPHY_DIM, sizeof(double));
        encode_text(bib, eng, sents[i], sv);
        memcpy(&bp->binding_vecs[bi * BIBLIOGRAPHY_DIM], sv, BIBLIOGRAPHY_DIM*sizeof(double));
        bp->binding_indices[bi] = (int)i;
        bi++;
        free(sv);
    }
    free(sent_scores);
    bib->paper_count++;

    /* Per-paper holographic ranker: each sentence self-keyed.
     * M = Σ bind(s_i, s_i). unbind(M, claim) ≈ Σ s_i * dot(s_i, claim).
     * Scoring s_j against context gives sentence-claim relevance. */
    bib->paper_rankers[paper_idx] = holo_ranker_create(BIBLIOGRAPHY_DIM);
    bib->paper_centroids[paper_idx] = calloc(BIBLIOGRAPHY_DIM, sizeof(double));
    memcpy(bib->paper_centroids[paper_idx], pv, BIBLIOGRAPHY_DIM*sizeof(double));

    for (size_t i = 0; i < bcount; i++) {
        const double *sv = &bp->binding_vecs[i * BIBLIOGRAPHY_DIM];
        holo_ranker_store(bib->paper_rankers[paper_idx], sv, sv);
    }
    leos_engine_destroy(eng);
    return 0;
}

int bibliography_query_claim(const BibliographyStore *bib, const char *claim_text,
                             const char *domain_context, char out_bibkeys[][BIBLIOGRAPHY_BIBKEY_MAX],
                             double *out_scores, size_t N) {
    if (!bib || !claim_text || !out_bibkeys || !out_scores || N == 0) return -1;
    LeosEngine *eng = leos_engine_create(BIBLIOGRAPHY_DIM);
    if (!eng) return -1;
    char enriched[1024];
    if (domain_context && domain_context[0]) snprintf(enriched, sizeof(enriched), "domain %s claim %s", domain_context, claim_text);
    else snprintf(enriched, sizeof(enriched), "claim %s", claim_text);
    double *cv = calloc(BIBLIOGRAPHY_DIM, sizeof(double));
    encode_text(bib, eng, enriched, cv);

    double *ps = calloc(bib->paper_count, sizeof(double));
    /* Sentence-level holographic retrieval with span detection.
     * Score every sentence in every paper, group into spans,
     * paper score = best span sum. */
    for (size_t i = 0; i < bib->paper_count; i++) {
        const BibPaper *bp = &bib->papers[i];
        if (!bib->paper_rankers[i] || !bp->binding_vecs || bp->binding_count == 0) continue;
        double *ss = calloc(bp->binding_count, sizeof(double));
        holo_ranker_score(bib->paper_rankers[i], cv, bp->binding_vecs, bp->binding_count, ss);

        int span_start, span_end;
        ps[i] = find_best_span(ss, bp->binding_count, &span_start, &span_end);
        free(ss);
    }
    if (bib->term_bridge) {
        char *div = text_mutator_diversify(bib->term_bridge, claim_text);
        if (div && strcmp(div, claim_text) != 0) {
            char e2[1024];
            if (domain_context && domain_context[0]) snprintf(e2, sizeof(e2), "domain %s claim %s", domain_context, div);
            else snprintf(e2, sizeof(e2), "claim %s", div);
            double *sv = calloc(BIBLIOGRAPHY_DIM, sizeof(double));
            encode_text(bib, eng, e2, sv);
            for (size_t i = 0; i < bib->paper_count; i++) {
                if (!bib->papers[i].abstract_vec) continue;
                double s2 = leos_engine_dot(eng, sv, bib->papers[i].abstract_vec);
                if (s2 > ps[i]) ps[i] = s2;
            }
            free(sv);
        }
        free(div);
    }
    for (size_t i = 0; i < N; i++) out_scores[i] = -1e9;
    for (size_t i = 0; i < bib->paper_count; i++) {
        int pos = (int)N-1;
        while (pos >= 0 && ps[i] > out_scores[pos]) pos--;
        pos++; if (pos >= (int)N) continue;
        for (int j = (int)N-1; j > pos; j--) { out_scores[j] = out_scores[j-1]; memcpy(out_bibkeys[j], out_bibkeys[j-1], BIBLIOGRAPHY_BIBKEY_MAX); }
        out_scores[pos] = ps[i];
        size_t bl = strlen(bib->papers[i].bibkey); if (bl >= BIBLIOGRAPHY_BIBKEY_MAX) bl = BIBLIOGRAPHY_BIBKEY_MAX-1;
        memcpy(out_bibkeys[pos], bib->papers[i].bibkey, bl); out_bibkeys[pos][bl] = '\0';
    }
    free(cv); free(ps); leos_engine_destroy(eng);
    return 0;
}


/* ---- Sentence-span detection ----
 * Groups consecutive above-threshold sentence scores into spans.
 * Returns the best span (highest sum score) for a paper. */

#define SPAN_GAP_TOLERANCE 2  /* allow up to 2 below-threshold sentences within a span */
static double find_best_span(const double *scores, size_t n,

                             int *out_start, int *out_end) {
    double median = 0;
    if (n > 0) {
        double *scopy = calloc(n, sizeof(double));
        memcpy(scopy, scores, n*sizeof(double));
        for (size_t i=0; i<n; i++) for (size_t j=i+1; j<n; j++) if (scopy[i]>scopy[j]){double t=scopy[i];scopy[i]=scopy[j];scopy[j]=t;}
        median = scopy[n/2];
        free(scopy);
    }
    double best_sum = 0; int best_start = 0, best_end = 0;
    size_t i = 0;
    while (i < n) {
        /* Find start of span: first sentence above threshold */
        while (i < n && scores[i] < median) i++;
        if (i >= n) break;
        size_t span_start = i;
        double span_sum = 0;
        int gaps = 0;
        /* Extend span, allowing gaps */
        while (i < n) {
            if (scores[i] >= median) { span_sum += scores[i]; gaps = 0; }
            else { gaps++; if (gaps > SPAN_GAP_TOLERANCE) break; }
            i++;
        }
        if (span_sum > best_sum) { best_sum = span_sum; best_start = (int)span_start; best_end = (int)(i - gaps - 1); }
    }
    *out_start = best_start; *out_end = best_end;
    return best_sum;
}

size_t bibliography_ordinal_query(const BibliographyStore *bib, const char *claim_text,
                                  const char *domain_context, char out_bibkeys[][BIBLIOGRAPHY_BIBKEY_MAX],
                                  size_t N) {
    double *ps = calloc(bib->paper_count, sizeof(double));
    if (!ps) return 0;
    LeosEngine *eng = leos_engine_create(BIBLIOGRAPHY_DIM);
    if (!eng) { free(ps); return 0; }
    char enriched[1024];
    if (domain_context && domain_context[0]) snprintf(enriched, sizeof(enriched), "domain %s claim %s", domain_context, claim_text);
    else snprintf(enriched, sizeof(enriched), "claim %s", claim_text);
    double *cv = calloc(BIBLIOGRAPHY_DIM, sizeof(double));
    encode_text(bib, eng, enriched, cv);
    /* Sentence-level holographic retrieval with span detection.
     * Score every sentence in every paper, group into spans,
     * paper score = best span sum. */
    for (size_t i = 0; i < bib->paper_count; i++) {
        const BibPaper *bp = &bib->papers[i];
        if (!bib->paper_rankers[i] || !bp->binding_vecs || bp->binding_count == 0) continue;
        double *ss = calloc(bp->binding_count, sizeof(double));
        holo_ranker_score(bib->paper_rankers[i], cv, bp->binding_vecs, bp->binding_count, ss);

        int span_start, span_end;
        ps[i] = find_best_span(ss, bp->binding_count, &span_start, &span_end);
        free(ss);
    }
    if (bib->term_bridge) {
        char *div = text_mutator_diversify(bib->term_bridge, claim_text);
        if (div && strcmp(div, claim_text) != 0) {
            char e2[1024];
            if (domain_context && domain_context[0]) snprintf(e2, sizeof(e2), "domain %s claim %s", domain_context, div);
            else snprintf(e2, sizeof(e2), "claim %s", div);
            double *sv = calloc(BIBLIOGRAPHY_DIM, sizeof(double));
            encode_text(bib, eng, e2, sv);
            for (size_t i = 0; i < bib->paper_count; i++) {
                if (!bib->papers[i].abstract_vec) continue;
                double s2 = leos_engine_dot(eng, sv, bib->papers[i].abstract_vec);
                if (s2 > ps[i]) ps[i] = s2;
            }
            free(sv);
        }
        free(div);
    }
    size_t *order = calloc(bib->paper_count, sizeof(size_t));
    for (size_t i = 0; i < bib->paper_count; i++) order[i] = i;
    for (size_t i = 0; i < bib->paper_count; i++)
        for (size_t j = i+1; j < bib->paper_count; j++)
            if (ps[order[i]] < ps[order[j]]) { size_t t = order[i]; order[i] = order[j]; order[j] = t; }
    size_t on = N < bib->paper_count ? N : bib->paper_count;
    for (size_t k = 0; k < on; k++) {
        size_t bl = strlen(bib->papers[order[k]].bibkey); if (bl >= BIBLIOGRAPHY_BIBKEY_MAX) bl = BIBLIOGRAPHY_BIBKEY_MAX-1;
        memcpy(out_bibkeys[k], bib->papers[order[k]].bibkey, bl); out_bibkeys[k][bl] = '\0';
    }
    free(cv); free(ps); free(order); leos_engine_destroy(eng);
    return on;
}

size_t bibliography_paper_count(const BibliographyStore *bib) { return bib?bib->paper_count:0; }
size_t bibliography_sentence_count(const BibliographyStore *bib) {
    if (!bib) return 0; size_t t = 0;
    for (size_t i = 0; i < bib->paper_count; i++) t += bib->papers[i].sentence_count; return t;
}
double bibliography_sentence_fidelity(const BibliographyStore *bib) {
    if (!bib) return 0; double f = 0; size_t n = 0;
    for (size_t t = 0; t < bib->topic_count; t++) { f += holo_layer_fidelity(bib->sentence_layers[t]); n++; }
    return n ? f/(double)n : 0;
}
double bibliography_abstract_fidelity(const BibliographyStore *bib) {
    if (!bib || bib->paper_count == 0) return 0; double total = 0;
    for (size_t i = 0; i < bib->paper_count; i++) if (bib->paper_rankers[i]) total += holo_ranker_fidelity(bib->paper_rankers[i]);
    return total / (double)bib->paper_count;
}
double bibliography_topic_fidelity(const BibliographyStore *bib) { return bibliography_abstract_fidelity(bib); }
