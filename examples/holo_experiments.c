#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>

#include "nn_holographic_ranker.h"
#include "memory_neuron.h"
#include "nn_core.h"
#include "nn_holo_gate.h"
#include "holographic/leos_engine.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ---- Utilities ---- */

static void mkdir_p(const char *path) {
    char tmp[1024];
    snprintf(tmp, sizeof(tmp), "%s", path);
    for (char *p = tmp + 1; *p; p++) {
        if (*p == '/') { *p = 0; mkdir(tmp, 0755); *p = '/'; }
    }
    mkdir(tmp, 0755);
}

static void gen_unit_vec(LeosEngine *eng, uint64_t seed, double *out) {
    leos_engine_keygen(eng, seed, out);
}

/* Feature-aware VSA embedding: expands feature vector into VSA space.
   Each feature modulates a band of dimensions via sinusoidal encoding.
   Similar feature vectors -> similar VSA vectors. */
static void embed_features(LeosEngine *eng, const double *feats, size_t nf, size_t dim, double *out) {
    memset(out, 0, dim * sizeof(double));
    for (size_t f = 0; f < nf && f < dim; f++) {
        double phase = feats[f] * 3.0;
        size_t bs = (f * dim) / nf, be = ((f + 1) * dim) / nf;
        for (size_t d = bs; d < be; d++)
            out[d] = sin(phase * (double)(d - bs + 1));
    }
    leos_engine_normalize(eng, out);
}

/* Score-based VSA embedding: fallback when features aren't available. */
static void embed_scores(LeosEngine *eng, const double *scores, size_t ns, size_t dim, double *out) {
    embed_features(eng, scores, ns, dim, out);
}

/* Parse a line into state_hash, target, verified, scores, and optional features.
   Format: state_hash\tstate_val\ttarget\tverified\tscore0\t...
   state_val may be "features:f0,f1,..." or "hash:NNN" */
typedef struct {
    char *state_val;       /* raw state_val string (caller owns) */
    size_t target;
    int verified;
    size_t n_candidates;
    double *scores;        /* calloc'd, caller frees */
    int has_features;
    double features[2048];
    size_t n_features;
} ParsedRow;

static int parse_row(char *line, ParsedRow *row) {
    memset(row, 0, sizeof(*row));
    char *p = line;
    
    /* state_hash */
    char *t = strchr(p, '\t'); if (!t) return 0; *t = '\0';
    p = t + 1;
    
    /* state_val */
    row->state_val = p;
    t = strchr(p, '\t'); if (!t) return 0; *t = '\0'; p = t + 1;
    
    /* target */
    row->target = (size_t)atoi(p);
    t = strchr(p, '\t'); if (!t) return 0; p = t + 1;
    
    /* verified */
    row->verified = atoi(p);
    t = strchr(p, '\t'); if (!t) return 0; p = t + 1;
    
    /* Check for features prefix */
    if (strncmp(row->state_val, "features:", 9) == 0) {
        row->has_features = 1;
        char *fs = row->state_val + 9;
        char *tok = strtok(fs, ",");
        while (tok && row->n_features < 2048) {
            row->features[row->n_features++] = atof(tok);
            tok = strtok(NULL, ",");
        }
    }
    
    /* scores */
    char *score_start = p;
    row->n_candidates = 1;
    while (*p) { if (*p == '\t') row->n_candidates++; p++; }
    
    row->scores = calloc(row->n_candidates, sizeof(double));
    if (!row->scores) return 0;
    p = score_start;
    for (size_t j = 0; j < row->n_candidates; j++) {
        row->scores[j] = atof(p);
        char *nx = strchr(p, '\t');
        if (nx) p = nx + 1;
    }
    return 1;
}

/* Embed a parsed row's state: use features if available, else scores. */
static void embed_row(LeosEngine *eng, const ParsedRow *row, size_t dim, double *out) {
    if (row->has_features)
        embed_features(eng, row->features, row->n_features, dim, out);
    else
        embed_scores(eng, row->scores, row->n_candidates, dim, out);
}

/* ---- B.1: Exact Retrieval Fidelity ---- */
static int run_exact_retrieval(const char *od) {
    mkdir_p(od);
    char path[1024]; snprintf(path, sizeof(path), "%s/retrieval_fidelity.jsonl", od);
    FILE *fp = fopen(path, "w"); if (!fp) return 1;
    
    size_t dim = 2048;
    LeosEngine *eng = leos_engine_create(dim);
    size_t test_ns[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000};
    size_t n_tests = sizeof(test_ns) / sizeof(test_ns[0]), max_n = test_ns[n_tests - 1];
    
    double *states = calloc(max_n * dim, sizeof(double));
    double *actions = calloc(max_n * dim, sizeof(double));
    for (size_t i = 0; i < max_n; i++) {
        gen_unit_vec(eng, 20240601ULL + i * 2, &states[i * dim]);
        gen_unit_vec(eng, 20240601ULL + i * 2 + 1, &actions[i * dim]);
    }
    
    for (size_t ti = 0; ti < n_tests; ti++) {
        size_t n = test_ns[ti];
        HolographicRanker *r = holo_ranker_create(dim);
        for (size_t i = 0; i < n; i++) holo_ranker_store(r, &states[i*dim], &actions[i*dim]);
        
        double *sc = calloc(n, sizeof(double));
        size_t hits = 0;
        for (size_t i = 0; i < n; i++) {
            holo_ranker_score(r, &states[i*dim], actions, n, sc);
            size_t best = 0;
            for (size_t j = 1; j < n; j++) if (sc[j] > sc[best]) best = j;
            if (best == i) hits++;
        }
        free(sc);
        
        fprintf(fp, "{\"schema\":\"crlp.holo_retrieval_fidelity.v1\",\"dim\":%zu,\"n\":%zu,"
                "\"top1_hits\":%zu,\"top1_acc\":%.8f,\"fidelity\":%.8f,\"bound\":%.8f}\n",
                dim, n, hits, (double)hits/n, holo_ranker_fidelity(r), 1.0/sqrt((double)n));
        holo_ranker_destroy(r);
    }
    free(states); free(actions); leos_engine_destroy(eng); fclose(fp);
    return 0;
}

/* ---- B.2: Memory Neuron Routing ---- */
static int run_memory_routing(const char *od) {
    mkdir_p(od);
    char path[1024]; snprintf(path, sizeof(path), "%s/memory_neuron_routing.jsonl", od);
    FILE *fp = fopen(path, "w"); if (!fp) return 1;
    
    size_t dim = 2048, rows = 4, cols = 4, k_active = 4, tick_int = 8;
    LeosEngine *eng = leos_engine_create(dim);
    MemoryNeuronGrid *grid = memory_neuron_grid_create(dim, rows, cols, 16, k_active, tick_int, 0.01);
    if (!grid) return 1;
    
    double cluster_centers[4][16], state_emb[2048], action_emb[2048], features[16];
    for (size_t c = 0; c < 4; c++) {
        for (size_t d = 0; d < 16; d++) cluster_centers[c][d] = ((c >> (d%4)) & 1) ? 1.0 : -1.0;
        double nrm = 0; for (size_t d = 0; d < 16; d++) nrm += cluster_centers[c][d]*cluster_centers[c][d];
        nrm = sqrt(nrm); for (size_t d = 0; d < 16; d++) cluster_centers[c][d] /= nrm;
    }
    
    size_t total_ticks = 2000, per_cluster = total_ticks / 4;
    double prev_centroids[16][2048]; int has_prev = 0;
    
    for (size_t tick = 0; tick < total_ticks; tick++) {
        size_t cl = tick / per_cluster; if (cl >= 4) cl = 3;
        uint64_t seed = 20240602ULL + tick * 3;
        gen_unit_vec(eng, seed, state_emb); gen_unit_vec(eng, seed+1, action_emb);
        for (size_t d = 0; d < 16; d++) {
            double ns = ((double)((seed*1103515245+12345)&0x7fffffff)/0x7fffffff - 0.5)*0.2;
            features[d] = cluster_centers[cl][d] + ns;
        }
        double fn = 0; for (size_t d = 0; d < 16; d++) fn += features[d]*features[d];
        fn = sqrt(fn); if (fn > 0) for (size_t d = 0; d < 16; d++) features[d] /= fn;
        memory_neuron_grid_store(grid, state_emb, action_emb, features);
        
        if (tick > 0 && tick % 200 == 0) {
            double mx_drift = 0, sm_drift = 0, inter_sim = 0, intra_sim = 0;
            size_t occ = 0, ip = 0, ia = 0;
            size_t cell_stored[16] = {0};
            
            for (size_t r = 0; r < rows; r++) for (size_t c = 0; c < cols; c++) {
                MemoryNeuron *mn = &grid->neurons[r*cols+c];
                cell_stored[r*cols+c] = mn->stored;
                if (!mn->stored) continue;
                occ++;
                if (has_prev) {
                    double dr = 0;
                    for (size_t d = 0; d < dim; d++) { double delta = mn->centroid[d]-prev_centroids[r*cols+c][d]; dr += delta*delta; }
                    dr = sqrt(dr); if (dr > mx_drift) mx_drift = dr; sm_drift += dr;
                }
                memcpy(prev_centroids[r*cols+c], mn->centroid, dim*sizeof(double));
            }
            
            for (size_t r1 = 0; r1 < rows; r1++) for (size_t c1 = 0; c1 < cols; c1++)
                for (size_t r2 = 0; r2 < rows; r2++) for (size_t c2 = 0; c2 < cols; c2++) {
                    if (r1==r2 && c1==c2) continue;
                    MemoryNeuron *a = &grid->neurons[r1*cols+c1], *b = &grid->neurons[r2*cols+c2];
                    if (a->stored && b->stored) { inter_sim += leos_engine_dot(eng,a->centroid,b->centroid); ip++; }
                }
            for (size_t r = 0; r < rows; r++) for (size_t c = 0; c < cols; c++)
                if (grid->neurons[r*cols+c].stored) { intra_sim += grid->neurons[r*cols+c].fidelity; ia++; }
            
            intra_sim = ia ? intra_sim/ia : 0; inter_sim = ip ? inter_sim/ip : 0;
            sm_drift = occ ? sm_drift/occ : -1;
            
            fprintf(fp, "{\"schema\":\"crlp.memory_neuron_routing.v1\",\"dim\":%zu,\"grid\":\"%zux%zu\","
                    "\"tick\":%zu,\"occupied\":%zu,\"max_drift\":%.8f,\"mean_drift\":%.8f,"
                    "\"inter_cell_cosine\":%.8f,\"intra_fidelity\":%.8f,"
                    "\"cell_stored\":[%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu,%zu]}\n",
                    dim, rows, cols, tick, occ, has_prev?mx_drift:-1.0, sm_drift, inter_sim, intra_sim,
                    cell_stored[0],cell_stored[1],cell_stored[2],cell_stored[3],
                    cell_stored[4],cell_stored[5],cell_stored[6],cell_stored[7],
                    cell_stored[8],cell_stored[9],cell_stored[10],cell_stored[11],
                    cell_stored[12],cell_stored[13],cell_stored[14],cell_stored[15]);
            fflush(fp);
            has_prev = 1;
        }
    }
    
    memory_neuron_grid_destroy(grid); leos_engine_destroy(eng); fclose(fp);
    return 0;
}

/* ---- B.3: Holographic Generalization ---- */
static int run_generalization(const char *od) {
    mkdir_p(od);
    char path[1024]; snprintf(path, sizeof(path), "%s/holo_generalization.jsonl", od);
    FILE *fp = fopen(path, "w"); if (!fp) return 1;
    
    size_t dim = 2048, n_store = 200, n_held = 50;
    LeosEngine *eng = leos_engine_create(dim);
    double *ba = calloc(dim, sizeof(double)), *bb = calloc(dim, sizeof(double));
    double *sa = calloc((n_store+n_held)*dim, sizeof(double)), *aa = calloc(n_store*dim, sizeof(double));
    double *sb = calloc(n_held*dim, sizeof(double)), *ab = calloc(n_held*dim, sizeof(double));
    double *rec = calloc(dim, sizeof(double)), *sc = calloc(n_store, sizeof(double));
    
    gen_unit_vec(eng, 20240603ULL, ba); gen_unit_vec(eng, 20240603ULL+10000, bb);
    
    for (size_t i = 0; i < n_store+n_held; i++) {
        double nv[2048]; gen_unit_vec(eng, 20240603ULL+100+i, nv);
        for (size_t d = 0; d < dim; d++) sa[i*dim+d] = 0.95*ba[d] + 0.05*nv[d];
        leos_engine_normalize(eng, &sa[i*dim]);
    }
    for (size_t i = 0; i < n_store; i++) gen_unit_vec(eng, 20240603ULL+200+i, &aa[i*dim]);
    for (size_t i = 0; i < n_held; i++) {
        double nv[2048]; gen_unit_vec(eng, 20240603ULL+300+i, nv);
        for (size_t d = 0; d < dim; d++) sb[i*dim+d] = 0.95*bb[d] + 0.05*nv[d];
        leos_engine_normalize(eng, &sb[i*dim]);
        gen_unit_vec(eng, 20240603ULL+400+i, &ab[i*dim]);
    }
    
    HolographicRanker *r = holo_ranker_create(dim);
    for (size_t i = 0; i < n_store; i++) holo_ranker_store(r, &sa[i*dim], &aa[i*dim]);
    
    double sum_sim_a = 0, sum_sim_b = 0;
    for (size_t i = 0; i < n_held; i++) {
        holo_ranker_recall(r, &sa[(n_store+i)*dim], rec);
        holo_ranker_score(r, &sa[(n_store+i)*dim], aa, n_store, sc);
        size_t best = 0; for (size_t j = 1; j < n_store; j++) if (sc[j] > sc[best]) best = j;
        sum_sim_a += leos_engine_dot(eng, rec, &aa[best*dim]);
        
        holo_ranker_recall(r, &sb[i*dim], rec);
        holo_ranker_score(r, &sb[i*dim], aa, n_store, sc);
        best = 0; for (size_t j = 1; j < n_store; j++) if (sc[j] > sc[best]) best = j;
        sum_sim_b += leos_engine_dot(eng, rec, &aa[best*dim]);
    }
    
    double chance = 0;
    for (size_t i = 0; i < 100; i++) {
        double *ra = calloc(dim, sizeof(double)), *rb = calloc(dim, sizeof(double));
        gen_unit_vec(eng, 20240603ULL+500+i*2, ra); gen_unit_vec(eng, 20240603ULL+500+i*2+1, rb);
        chance += leos_engine_dot(eng, ra, rb); free(ra); free(rb);
    }
    chance /= 100;
    
    double ma = sum_sim_a/n_held, mb = sum_sim_b/n_held;
    fprintf(fp, "{\"schema\":\"crlp.holo_generalization.v1\",\"dim\":%zu,\"n_stored\":%zu,\"n_held\":%zu,"
            "\"similar_cosine\":%.8f,\"dissimilar_cosine\":%.8f,\"chance\":%.8f,\"above_chance\":%s}\n",
            dim, n_store, n_held, ma, mb, chance, ma > chance ? "true" : "false");
    
    free(ba); free(bb); free(sa); free(aa); free(sb); free(ab); free(rec); free(sc);
    holo_ranker_destroy(r); leos_engine_destroy(eng); fclose(fp);
    return 0;
}

/* ---- A.3: HoloGate Blend ---- */
static int run_holo_gate_blend(const char *od) {
    mkdir_p(od);
    char path[1024]; snprintf(path, sizeof(path), "%s/holo_gate_blend.jsonl", od);
    FILE *fp = fopen(path, "w"); if (!fp) return 1;
    
    size_t dim = 2048, fid_levels[] = {1,2,3,4,8,16,32}, nl = 7, nq = 400;
    LeosEngine *eng = leos_engine_create(dim);
    double *states = calloc(32*dim, sizeof(double)), *actions = calloc(32*dim, sizeof(double));
    for (size_t i = 0; i < 32; i++) {
        gen_unit_vec(eng, 20240801ULL+i*2, &states[i*dim]);
        gen_unit_vec(eng, 20240801ULL+i*2+1, &actions[i*dim]);
    }
    
    double *qs = calloc(nq*dim, sizeof(double));
    size_t *tl = calloc(nq, sizeof(size_t));
    double *hs = calloc(32, sizeof(double)), *bs = calloc(32, sizeof(double));
    nn_real *ns = calloc(32, sizeof(nn_real));
    
    for (size_t li = 0; li < nl; li++) {
        size_t ns_store = fid_levels[li];
        for (size_t q = 0; q < nq; q++) {
            double nv[2048]; gen_unit_vec(eng, 20240801ULL+10000+li*nq+q, nv);
            int sim = (q < (nq*65)/100);
            double sw = sim ? 0.95 : 0.3, nw = sqrt(1-sw*sw);
            for (size_t d = 0; d < dim; d++) qs[q*dim+d] = sw*states[d] + nw*nv[d];
            leos_engine_normalize(eng, &qs[q*dim]);
            tl[q] = sim ? 0 : (size_t)((q*1103515245+12345)%ns_store);
        }
        
        HolographicRanker *r = holo_ranker_create(dim);
        for (size_t i = 0; i < ns_store; i++) holo_ranker_store(r, &states[i*dim], &actions[i*dim]);
        double fid = holo_ranker_fidelity(r);
        size_t ht = 0, nt = 0, bt = 0;
        
        for (size_t q = 0; q < nq; q++) {
            holo_ranker_score(r, &qs[q*dim], actions, ns_store, hs);
            for (size_t j = 0; j < ns_store; j++) {
                uint64_t ns_seed = 20240801ULL+30000+li*100000+q*ns_store+j;
                double nz = ((double)((ns_seed*1103515245+12345)&0x7fffffff)/0x7fffffff-0.5)*1.2;
                ns[j] = (nn_real)((j==tl[q]?1.0:0.0) + nz);
            }
            holo_gate_blend(fid, hs, ns, ns_store, bs);
            size_t bh=0,bn=0,bb=0;
            for (size_t j=1; j<ns_store; j++) { if(hs[j]>hs[bh])bh=j; if(ns[j]>ns[bn])bn=j; if(bs[j]>bs[bb])bb=j; }
            if(bh==tl[q])ht++; if(bn==tl[q])nt++; if(bb==tl[q])bt++;
        }
        double max_s = (double)ht/nq > (double)nt/nq ? (double)ht/nq : (double)nt/nq;
        fprintf(fp, "{\"schema\":\"crlp.holo_gate_blend.v1\",\"dim\":%zu,\"n\":%zu,\"fid\":%.8f,"
                "\"alpha\":%.8f,\"holo\":%.8f,\"neural\":%.8f,\"blend\":%.8f,\"delta\":%.8f}\n",
                dim, ns_store, fid, fid*fid*fid, (double)ht/nq, (double)nt/nq, (double)bt/nq, (double)bt/nq-max_s);
        holo_ranker_destroy(r);
    }
    free(states); free(actions); free(qs); free(tl); free(hs); free(bs); free(ns);
    leos_engine_destroy(eng); fclose(fp);
    return 0;
}

/* ---- Domain Integration: holo across all 5 domains ---- */
#define N_CELLS 16

static int run_domain_integration(const char *input_path, const char *domain, const char *od) {
    mkdir_p(od);
    char path[1024]; snprintf(path, sizeof(path), "%s/%s_holo_blend.jsonl", od, domain);
    FILE *fp = fopen(path, "w"); if (!fp) return 1;
    
    FILE *in = fopen(input_path, "r"); if (!in) { fclose(fp); return 0; }
    
    size_t max_rows = 5000, dim = 2048;
    ParsedRow *rows = calloc(max_rows, sizeof(ParsedRow));
    char *line_buf = calloc(65536, 1);
    size_t n = 0, n_cand = 0;
    
    while (n < max_rows && fgets(line_buf, 65536, in)) {
        char *line_copy = strdup(line_buf);
        if (parse_row(line_copy, &rows[n]) && rows[n].n_candidates > 0) {
            if (!n_cand) n_cand = rows[n].n_candidates;
            n++;
        } else free(line_copy);
    }
    fclose(in); free(line_buf);
    if (n < 20) { fclose(fp); return 0; }
    
    size_t n_train = (n*7)/10, n_test = n - n_train;
    printf("  domain=%s loaded=%zu cand=%zu train=%zu test=%zu\n", domain, n, n_cand, n_train, n_test);
    
    LeosEngine *eng = leos_engine_create(dim);
    if (!eng) return 1;
    
    double *act_emb = calloc(n_cand * dim, sizeof(double));
    for (size_t j = 0; j < n_cand; j++) gen_unit_vec(eng, 20241001ULL+j, &act_emb[j*dim]);
    
    size_t store_sizes[] = {25, 50, 100, 200, 350, 500};
    size_t n_fracs = 6;
    if (n_train < 100) { store_sizes[0] = n_train/4; store_sizes[1] = n_train/2; n_fracs = 2; }
    
    double *st_emb = calloc(dim, sizeof(double));
    double *ho_sc = calloc(n_cand, sizeof(double));
    double *bl_sc = calloc(n_cand, sizeof(double));
    nn_real *ne_sc = calloc(n_cand, sizeof(nn_real));
    
    for (size_t fi = 0; fi < n_fracs; fi++) {
        size_t n_store = store_sizes[fi];
        if (n_store > n_train) n_store = n_train;
        
        /* 16-cell hash-based partition: consistent routing for similar states */
        #define N_CELLS 16
        HolographicRanker *cells[N_CELLS];
        for (int c = 0; c < N_CELLS; c++) cells[c] = holo_ranker_create(dim);
        size_t cell_counts[N_CELLS] = {0}, stored = 0;
        for (size_t i = 0; i < n_store && i < n_train; i++) {
            if (!rows[i].verified) continue;
            embed_row(eng, &rows[i], dim, st_emb);
            int cell = 0;
            for (size_t d = 0; d < dim && d < 32; d++) cell ^= (int)(st_emb[d] * 7919.0);
            cell &= 0xF;
            if (cells[cell]) {
                holo_ranker_store(cells[cell], st_emb, &act_emb[rows[i].target * dim]);
                cell_counts[cell]++; stored++;
            }
        }
        
        double fid_sum = 0; int active = 0;
        for (int c = 0; c < N_CELLS; c++)
            if (cells[c] && cell_counts[c]) { fid_sum += holo_ranker_fidelity(cells[c]); active++; }
        double fidelity = active ? fid_sum / active : 0;
        
        size_t holo_top1 = 0, neural_top1 = 0, blend_top1 = 0;
        
        for (size_t i = n_train; i < n; i++) {
            size_t correct = rows[i].target;
            embed_row(eng, &rows[i], dim, st_emb);
            
            for (size_t j = 0; j < n_cand; j++) ho_sc[j] = -1e10;
            for (int c = 0; c < N_CELLS; c++) {
                if (!cells[c] || !cell_counts[c]) continue;
                double cs[256];
                holo_ranker_score(cells[c], st_emb, act_emb, n_cand, cs);
                for (size_t j = 0; j < n_cand; j++) if (cs[j] > ho_sc[j]) ho_sc[j] = cs[j];
            }
            
            for (size_t j = 0; j < n_cand; j++) ne_sc[j] = (nn_real)rows[i].scores[j];
            
            /* Normalize & blend */
            double hmin=ho_sc[0],hmax=ho_sc[0],nmin=ne_sc[0],nmax=ne_sc[0];
            for (size_t j=1; j<n_cand; j++) {
                if(ho_sc[j]<hmin)hmin=ho_sc[j]; if(ho_sc[j]>hmax)hmax=ho_sc[j];
                if(ne_sc[j]<nmin)nmin=ne_sc[j]; if(ne_sc[j]>nmax)nmax=ne_sc[j];
            }
            double hr=hmax-hmin; if(hr<1e-10)hr=1; double nr=nmax-nmin; if(nr<1e-10)nr=1;
            double hn[256]; nn_real nn_n[256];
            for (size_t j=0; j<n_cand; j++) { hn[j]=(ho_sc[j]-hmin)/hr; nn_n[j]=(nn_real)((ne_sc[j]-nmin)/nr); }
            holo_gate_blend(fidelity, hn, nn_n, n_cand, bl_sc);
            
            size_t bh=0,bn=0,bb=0;
            for (size_t j=1; j<n_cand; j++) {
                if(ho_sc[j]>ho_sc[bh])bh=j; if(ne_sc[j]>ne_sc[bn])bn=j; if(bl_sc[j]>bl_sc[bb])bb=j;
            }
            if(bh==correct)holo_top1++; if(bn==correct)neural_top1++; if(bb==correct)blend_top1++;
        }
        
        double ha=(double)holo_top1/n_test, na=(double)neural_top1/n_test, ba=(double)blend_top1/n_test;
        fprintf(fp, "{\"schema\":\"crlp.domain_holo_blend.v1\",\"domain\":\"%s\",\"dim\":%zu,"
                "\"n_train\":%zu,\"n_stored\":%zu,\"n_test\":%zu,\"n_cand\":%zu,"
                "\"fidelity\":%.8f,\"holo_top1\":%.8f,\"neural_top1\":%.8f,\"blended_top1\":%.8f}\n",
                domain, dim, n_store, stored, n_test, n_cand, fidelity, ha, na, ba);
        fflush(fp);
        
        double cfs[4]; size_t ccts[4];
        for (int c=0;c<4;c++) {
            cfs[c]=cells[c]&&cell_counts[c]?holo_ranker_fidelity(cells[c]):0;
            ccts[c]=cell_counts[c];
        }
        printf("  %-14s store=%3zu cells=[%.2f,%.2f,%.2f,%.2f] cnt=[%zu,%zu,%zu,%zu] holo=%.1f%% neur=%.1f%% blend=%.1f%%\n",
               domain, n_store, cfs[0],cfs[1],cfs[2],cfs[3], ccts[0],ccts[1],ccts[2],ccts[3], ha*100,na*100,ba*100);
        
        for (int c = 0; c < N_CELLS; c++) if (cells[c]) holo_ranker_destroy(cells[c]);
    }
    
    free(st_emb); free(ho_sc); free(bl_sc); free(ne_sc); free(act_emb);
    for (size_t i = 0; i < n; i++) free(rows[i].scores);
    free(rows); leos_engine_destroy(eng); fclose(fp);
    return 0;
}

static int run_learned_augment(const char *input_path, const char *domain, const char *od);
static int run_program_holo(const char *input_path, const char *od);
static int run_arc_discovery(const char *task_dir, const char *od);

/* ---- Main ---- */
int main(int argc, char **argv) {
    const char *mode = NULL, *od = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--mode") && i+1 < argc) mode = argv[++i];
        else if (!strcmp(argv[i], "--output-dir") && i+1 < argc) od = argv[++i];
    }
    if (!mode || !od) {
        fprintf(stderr, "usage: %s --mode <mode> --output-dir <dir>\n", argv[0]);
        fprintf(stderr, "  modes: exact-retrieval, memory-routing, generalization, holo-gate-blend, domain-integration\n");
        return 1;
    }
    printf("=== holo %s ===\n", mode);
    
    if (!strcmp(mode, "exact-retrieval"))     return run_exact_retrieval(od);
    if (!strcmp(mode, "memory-routing"))       return run_memory_routing(od);
    if (!strcmp(mode, "generalization"))       return run_generalization(od);
    if (!strcmp(mode, "holo-gate-blend"))      return run_holo_gate_blend(od);
    
    if (!strcmp(mode, "learned-augment")) {
        const char *ip = od, *dn = "unknown";
        for (int i = 1; i < argc; i++) {
            if (!strcmp(argv[i], "--input") && i+1 < argc) ip = argv[++i];
            else if (!strcmp(argv[i], "--domain") && i+1 < argc) dn = argv[++i];
        }
        return run_learned_augment(ip, dn, od);
    }
    if (!strcmp(mode, "arc-discovery")) {
        const char *td = od;
        for (int i = 1; i < argc; i++) {
            if (!strcmp(argv[i], "--task-dir") && i+1 < argc) td = argv[++i];
        }
        return run_arc_discovery(td, od);
    }
    if (!strcmp(mode, "program-holo")) {
        const char *ip = od;
        for (int i = 1; i < argc; i++) {
            if (!strcmp(argv[i], "--input") && i+1 < argc) ip = argv[++i];
        }
        return run_program_holo(ip, od);
    }
    if (!strcmp(mode, "domain-integration")) {
        const char *ip = od, *dn = "unknown";
        for (int i = 1; i < argc; i++) {
            if (!strcmp(argv[i], "--input") && i+1 < argc) ip = argv[++i];
            else if (!strcmp(argv[i], "--domain") && i+1 < argc) dn = argv[++i];
        }
        return run_domain_integration(ip, dn, od);
    }
    
    fprintf(stderr, "unknown mode: %s\n", mode);
    return 1;
}

/* ---- Learned Augmentation: treat holo score as a neural feature ---- */
static int run_learned_augment(const char *input_path, const char *domain, const char *od) {
    mkdir_p(od);
    char path[1024]; snprintf(path, sizeof(path), "%s/%s_augment.jsonl", od, domain);
    FILE *fp = fopen(path, "w"); if (!fp) return 1;
    
    FILE *in = fopen(input_path, "r"); if (!in) { fclose(fp); return 0; }
    
    size_t max_rows = 5000, dim = 2048;
    ParsedRow *rows = calloc(max_rows, sizeof(ParsedRow));
    char *line_buf = calloc(65536, 1);
    size_t n = 0, n_cand = 0;
    
    while (n < max_rows && fgets(line_buf, 65536, in)) {
        char *cp = strdup(line_buf);
        if (parse_row(cp, &rows[n]) && rows[n].n_candidates > 0) { if (!n_cand) n_cand = rows[n].n_candidates; n++; }
        else free(cp);
    }
    fclose(in); free(line_buf);
    if (n < 50) { fclose(fp); return 0; }
    
    size_t n_train = (n*6)/10, n_val = (n*8)/10;
    size_t n_val_count = n_val - n_train, n_test = n - n_val;
    printf("  domain=%s n=%zu cand=%zu train=%zu val=%zu test=%zu\n", domain, n, n_cand, n_train, n_val_count, n_test);
    
    LeosEngine *eng = leos_engine_create(dim);
    double *act_emb = calloc(n_cand * dim, sizeof(double));
    for (size_t j = 0; j < n_cand; j++) gen_unit_vec(eng, 20241101ULL+j, &act_emb[j*dim]);
    
    double *st_emb = calloc(dim, sizeof(double));
    double *ho_sc = calloc(n_cand, sizeof(double));
    
    /* Build holo store from training data */
    HolographicRanker *r = holo_ranker_create(dim);
    size_t stored = 0;
    for (size_t i = 0; i < n_train; i++) {
        if (!rows[i].verified) continue;
        embed_row(eng, &rows[i], dim, st_emb);
        holo_ranker_store(r, st_emb, &act_emb[rows[i].target * dim]);
        stored++;
    }
    printf("  stored=%zu fid=%.4f\n", stored, holo_ranker_fidelity(r));
    
    /* Grid search on validation: learn optimal weight for holo score augmentation.
       augmented_score[j] = neural_score[j] + weight * holo_score[j]
       This simulates what retraining with holo features would learn. */
    double best_weight = 0, best_val_acc = 0;
    for (int wi = 0; wi <= 200; wi++) {
        double w = (double)(wi - 100) / 100.0;  /* -1.0 to 1.0 */
        size_t correct = 0;
        for (size_t i = n_train; i < n_val; i++) {
            embed_row(eng, &rows[i], dim, st_emb);
            holo_ranker_score(r, st_emb, act_emb, n_cand, ho_sc);
            
            /* Normalize both score sets per query */
            double hmin=ho_sc[0],hmax=ho_sc[0],nmin=rows[i].scores[0],nmax=rows[i].scores[0];
            for (size_t j=1; j<n_cand; j++) {
                if(ho_sc[j]<hmin)hmin=ho_sc[j]; if(ho_sc[j]>hmax)hmax=ho_sc[j];
                if(rows[i].scores[j]<nmin)nmin=rows[i].scores[j]; if(rows[i].scores[j]>nmax)nmax=rows[i].scores[j];
            }
            double hr=hmax-hmin; if(hr<1e-10)hr=1; double nr=nmax-nmin; if(nr<1e-10)nr=1;
            double best_score = -1e10; size_t best_j = 0;
            for (size_t j = 0; j < n_cand; j++) {
                double hn = (ho_sc[j]-hmin)/hr, nn = (rows[i].scores[j]-nmin)/nr;
                double aug = nn + w * hn;
                if (aug > best_score) { best_score = aug; best_j = j; }
            }
            if (best_j == rows[i].target) correct++;
        }
        double acc = (double)correct / n_val_count;
        if (acc > best_val_acc) { best_val_acc = acc; best_weight = w; }
    }
    
    /* Evaluate on test set: compare neural alone, holo alone, augmented */
    size_t neural_top1 = 0, holo_top1 = 0, aug_top1 = 0, blend_top1 = 0;
    double fid = holo_ranker_fidelity(r);
    
    for (size_t i = n_val; i < n; i++) {
        embed_row(eng, &rows[i], dim, st_emb);
        holo_ranker_score(r, st_emb, act_emb, n_cand, ho_sc);
        
        /* Neural argmax */
        size_t bn = 0;
        for (size_t j = 1; j < n_cand; j++) if (rows[i].scores[j] > rows[i].scores[bn]) bn = j;
        if (bn == rows[i].target) neural_top1++;
        
        /* Holo argmax */
        size_t bh = 0;
        for (size_t j = 1; j < n_cand; j++) if (ho_sc[j] > ho_sc[bh]) bh = j;
        if (bh == rows[i].target) holo_top1++;
        
        /* Augmented argmax (normalized per query) */
        double hmin2=ho_sc[0],hmax2=ho_sc[0],nmin2=rows[i].scores[0],nmax2=rows[i].scores[0];
        for (size_t j=1; j<n_cand; j++) {
            if(ho_sc[j]<hmin2)hmin2=ho_sc[j]; if(ho_sc[j]>hmax2)hmax2=ho_sc[j];
            if(rows[i].scores[j]<nmin2)nmin2=rows[i].scores[j]; if(rows[i].scores[j]>nmax2)nmax2=rows[i].scores[j];
        }
        double hr2=hmax2-hmin2; if(hr2<1e-10)hr2=1; double nr2=nmax2-nmin2; if(nr2<1e-10)nr2=1;
        double best_aug = -1e10; size_t ba = 0;
        for (size_t j = 0; j < n_cand; j++) {
            double hn2 = (ho_sc[j]-hmin2)/hr2, nn2 = (rows[i].scores[j]-nmin2)/nr2;
            double aug = nn2 + best_weight * hn2;
            if (aug > best_aug) { best_aug = aug; ba = j; }
        }
        if (ba == rows[i].target) aug_top1++;
        
        /* HoloGate blend for comparison */
        double hmin=ho_sc[0],hmax=ho_sc[0],nmin=rows[i].scores[0],nmax=rows[i].scores[0];
        for (size_t j=1; j<n_cand; j++) {
            if(ho_sc[j]<hmin)hmin=ho_sc[j]; if(ho_sc[j]>hmax)hmax=ho_sc[j];
            if(rows[i].scores[j]<nmin)nmin=rows[i].scores[j]; if(rows[i].scores[j]>nmax)nmax=rows[i].scores[j];
        }
        double hr=hmax-hmin; if(hr<1e-10)hr=1; double nr=nmax-nmin; if(nr<1e-10)nr=1;
        double hn_bl[256], bl[256]; nn_real nn_bl[256];
        for (size_t j=0; j<n_cand; j++) { hn_bl[j]=(ho_sc[j]-hmin)/hr; nn_bl[j]=(nn_real)((rows[i].scores[j]-nmin)/nr); }
        holo_gate_blend(fid, hn_bl, nn_bl, n_cand, bl);
        size_t bb = 0;
        for (size_t j=1; j<n_cand; j++) if(bl[j]>bl[bb]) bb=j;
        if (bb == rows[i].target) blend_top1++;
    }
    
    double na = (double)neural_top1/n_test, ha = (double)holo_top1/n_test;
    double aa = (double)aug_top1/n_test, ba2 = (double)blend_top1/n_test;
    
    fprintf(fp, "{\"schema\":\"crlp.learned_augment.v1\",\"domain\":\"%s\",\"n_train\":%zu,\"n_val\":%zu,\"n_test\":%zu,"
            "\"n_cand\":%zu,\"best_weight\":%.4f,\"neural\":%.8f,\"holo\":%.8f,\"augmented\":%.8f,\"blend\":%.8f}\n",
            domain, n_train, n_val_count, n_test, n_cand, best_weight, na, ha, aa, ba2);
    
    printf("  %-14s weight=%.3f neural=%.1f%% holo=%.1f%% augmented=%.1f%% blend=%.1f%%\n",
           domain, best_weight, na*100, ha*100, aa*100, ba2*100);
    
    free(st_emb); free(ho_sc); free(act_emb);
    for (size_t i = 0; i < n; i++) free(rows[i].scores);
    free(rows); holo_ranker_destroy(r); leos_engine_destroy(eng); fclose(fp);
    return 0;
}

/* ---- Program-level holo: task features -> transform retrieval ---- */
static int run_program_holo(const char *input_path, const char *od) {
    mkdir_p(od);
    char path[1024]; snprintf(path, sizeof(path), "%s/program_holo.jsonl", od);
    FILE *fp = fopen(path, "w"); if (!fp) return 1;
    
    FILE *in = fopen(input_path, "r"); if (!in) { fclose(fp); return 0; }
    
    size_t max_rows = 500, dim = 2048, n_cand = 10;
    ParsedRow *rows = calloc(max_rows, sizeof(ParsedRow));
    char *line_buf = calloc(65536, 1);
    size_t n = 0;
    
    while (n < max_rows && fgets(line_buf, 65536, in)) {
        char *cp = strdup(line_buf);
        if (parse_row(cp, &rows[n]) && rows[n].n_candidates > 0) n++;
        else free(cp);
    }
    fclose(in); free(line_buf);
    if (n < 10) { fclose(fp); return 0; }
    
    size_t n_train = (n*7)/10, n_test = n - n_train;
    printf("  program-holo: %zu train, %zu test, %zu candidates\n", n_train, n_test, n_cand);
    
    LeosEngine *eng = leos_engine_create(dim);
    HolographicRanker *r = holo_ranker_create(dim);
    double *st_emb = calloc(dim, sizeof(double));
    double *act_emb = calloc(n_cand * dim, sizeof(double));
    
    /* Generate action embeddings for transform signatures */
    for (size_t j = 0; j < n_cand; j++)
        gen_unit_vec(eng, 20250000ULL + (uint64_t)j, &act_emb[j * dim]);
    
    /* Store training task signatures */
    size_t stored = 0;
    for (size_t i = 0; i < n_train; i++) {
        embed_row(eng, &rows[i], dim, st_emb);
        /* Store the target signature as the action */
        size_t best_j = 0;
        for (size_t j = 1; j < n_cand; j++)
            if (rows[i].scores[j] > rows[i].scores[best_j]) best_j = j;
        holo_ranker_store(r, st_emb, &act_emb[best_j * dim]);
        stored++;
    }
    
    /* Evaluate retrieval on test tasks */
    size_t top1 = 0, top5 = 0;
    double *sc = calloc(n_cand, sizeof(double));
    
    for (size_t i = n_train; i < n; i++) {
        embed_row(eng, &rows[i], dim, st_emb);
        holo_ranker_score(r, st_emb, act_emb, n_cand, sc);
        
        /* Find top-1 and check if it matches the ground truth */
        size_t best = 0, correct = 0;
        for (size_t j = 1; j < n_cand; j++) {
            if (sc[j] > sc[best]) best = j;
            if (rows[i].scores[j] > rows[i].scores[correct]) correct = j;
        }
        if (best == correct) top1++;
        
        /* Top-5: is correct in top 5? */
        size_t rank = 0;
        for (size_t j = 0; j < n_cand; j++)
            if (sc[j] > sc[correct]) rank++;
        if (rank < 5) top5++;
    }
    
    double fid = holo_ranker_fidelity(r);
    fprintf(fp, "{\"schema\":\"crlp.program_holo.v1\",\"n_train\":%zu,\"n_stored\":%zu,"
            "\"n_test\":%zu,\"fidelity\":%.4f,\"top1\":%.4f,\"top5\":%.4f}\n",
            n_train, stored, n_test, fid, (double)top1/n_test, (double)top5/n_test);
    
    printf("  program-holo: fid=%.4f top1=%.1f%% top5=%.1f%%\n",
           fid, (double)top1/n_test*100, (double)top5/n_test*100);
    
    free(st_emb); free(act_emb); free(sc);
    for (size_t i = 0; i < n; i++) free(rows[i].scores);
    free(rows); holo_ranker_destroy(r); leos_engine_destroy(eng); fclose(fp);
    return 0;
}

/* ---- ARC Holo-Discovery: cross-task transform retrieval ---- */
#include <dirent.h>

/* Minimal ARC task structure */
#define ARC_MAX_PAIRS 5
#define ARC_MAX_CELLS 1024

typedef struct {
    size_t width, height;
    uint8_t cells[ARC_MAX_CELLS];
} MiniGrid;

typedef struct {
    MiniGrid input, output;
} MiniPair;

typedef struct {
    char id[64];
    size_t pair_count;
    MiniPair pairs[ARC_MAX_PAIRS];
} MiniTask;

/* Load a single ARC task JSON - minimal parser for grid data */
static int load_mini_task(const char *path, MiniTask *task) {
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    char buf[131072];
    size_t len = fread(buf, 1, sizeof(buf)-1, f);
    fclose(f);
    if (len < 50) return 0;
    buf[len] = '\0';
    
    memset(task, 0, sizeof(*task));
    task->pair_count = 0;
    
    /* Extract task ID from filename */
    const char *slash = strrchr(path, '/');
    const char *name = slash ? slash + 1 : path;
    strncpy(task->id, name, 63);
    char *dot = strchr(task->id, '.');
    if (dot) *dot = '\0';
    
    /* Parse train pairs: find "train":[[...]] */
    char *train_start = strstr(buf, "\"train\":");
    if (!train_start) return 0;
    char *p = strchr(train_start, '[');
    if (!p) return 0;
    p++; /* skip opening [ of train array */
    
    /* Parse each pair */
    while (task->pair_count < ARC_MAX_PAIRS && *p && *p != ']') {
        /* Find "input" */
        char *inp = strstr(p, "\"input\":");
        char *outp = strstr(p, "\"output\":");
        if (!inp || !outp) break;
        
        /* Parse input grid */
        inp = strchr(inp + 8, '['); /* skip to grid rows */
        if (!inp) break;
        MiniGrid *ig = &task->pairs[task->pair_count].input;
        ig->height = 0; ig->width = 0;
        char *rp = inp + 1;
        while (*rp && *rp != ']') {
            if (*rp == '[') {
                rp++;
                while (*rp && *rp != ']') {
                    if (*rp >= '0' && *rp <= '9') {
                        if (ig->height * ig->width < ARC_MAX_CELLS) {
                            ig->cells[ig->height * ig->width] = (uint8_t)(*rp - '0');
                            ig->width++;
                        }
                    }
                    rp++;
                }
                if (*rp == ']') rp++;
                ig->height++;
            } else rp++;
        }
        if (ig->height > 0 && ig->width % ig->height == 0) ig->width /= ig->height;
        
        /* Parse output grid */
        outp = strchr(outp + 8, '[');
        if (!outp) break;
        MiniGrid *og = &task->pairs[task->pair_count].output;
        og->height = 0; og->width = 0;
        rp = outp + 1;
        while (*rp && *rp != ']') {
            if (*rp == '[') {
                rp++;
                size_t row_w = 0;
                while (*rp && *rp != ']') {
                    if (*rp >= '0' && *rp <= '9') {
                        if (og->height * og->width + row_w < ARC_MAX_CELLS) {
                            og->cells[og->height * og->width + row_w] = (uint8_t)(*rp - '0');
                            row_w++;
                        }
                    }
                    rp++;
                }
                if (*rp == ']') rp++;
                if (row_w > 0) og->height++;
                og->width = row_w;
            } else rp++;
        }
        
        task->pair_count++;
        p = outp;
    }
    
    return task->pair_count > 0;
}

/* Extract 64-dim task features for VSA embedding */
static void mini_task_features(const MiniTask *task, double *feats) {
    size_t fi = 0;
    for (size_t p = 0; p < task->pair_count && p < 2; p++) {
        feats[fi++] = task->pairs[p].input.width / 30.0;
        feats[fi++] = task->pairs[p].input.height / 30.0;
        feats[fi++] = task->pairs[p].output.width / 30.0;
        feats[fi++] = task->pairs[p].output.height / 30.0;
    }
    for (size_t p = 0; p < task->pair_count && p < 2; p++) {
        for (size_t y = 0; y < task->pairs[p].input.height; y++)
            for (size_t x = 0; x < task->pairs[p].input.width; x++)
                feats[fi++] = task->pairs[p].input.cells[y * task->pairs[p].input.width + x] / 9.0;
    }
    while (fi < 64) feats[fi++] = 0.0;
}

static void embed_mini_feats(LeosEngine *eng, const double *feats, size_t dim, double *out) {
    memset(out, 0, dim * sizeof(double));
    for (size_t f = 0; f < 64 && f < dim; f++) {
        double phase = feats[f] * 3.0;
        size_t bs = (f * dim) / 64, be = ((f + 1) * dim) / 64;
        for (size_t d = bs; d < be; d++)
            out[d] = sin(phase * (double)(d - bs + 1));
    }
    leos_engine_normalize(eng, out);
}

static int run_arc_discovery(const char *task_dir, const char *od) {
    mkdir_p(od);
    char path[1024]; snprintf(path, sizeof(path), "%s/arc_discovery.jsonl", od);
    FILE *fp = fopen(path, "w"); if (!fp) return 1;
    
    /* Load all tasks */
    DIR *dir = opendir(task_dir);
    if (!dir) { fprintf(stderr, "cannot open %s\n", task_dir); fclose(fp); return 1; }
    
    MiniTask *tasks = calloc(500, sizeof(MiniTask));
    size_t n_tasks = 0;
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL && n_tasks < 500) {
        if (!strstr(entry->d_name, ".json")) continue;
        char tpath[1024]; snprintf(tpath, sizeof(tpath), "%s/%s", task_dir, entry->d_name);
        if (load_mini_task(tpath, &tasks[n_tasks])) n_tasks++;
    }
    closedir(dir);
    printf("Loaded %zu tasks\n", n_tasks);
    if (n_tasks < 10) { fclose(fp); return 0; }
    
    size_t dim = 2048;
    LeosEngine *eng = leos_engine_create(dim);
    HolographicRanker *holo = holo_ranker_create(dim);
    
    /* Build holo store from first 70% tasks */
    size_t n_train = (n_tasks * 7) / 10;
    double state_emb[2048], sig_emb[2048];
    double feats[64];
    
    for (size_t i = 0; i < n_train; i++) {
        mini_task_features(&tasks[i], feats);
        embed_mini_feats(eng, feats, dim, state_emb);
        embed_mini_feats(eng, feats, dim, sig_emb);  /* task signature = features */
        holo_ranker_store(holo, state_emb, sig_emb);
    }
    
    /* Test: for each test task, query holo, find similar training tasks */
    size_t solved = 0, improved = 0;
    
    for (size_t i = n_train; i < n_tasks; i++) {
        mini_task_features(&tasks[i], feats);
        embed_mini_feats(eng, feats, dim, state_emb);
        
        /* Query holo store for similar tasks using VSA retrieval */
        /* Build action embedding table for all training tasks */
        double (*act_embs)[2048] = calloc(500, sizeof(double[2048]));
        double ho_scores[500];
        holo_ranker_score(holo, state_emb, (double*)act_embs, n_train, ho_scores);
        
        /* Find best match by holo score */
        size_t best_idx = 0;
        for (size_t j = 1; j < n_train; j++)
            if (ho_scores[j] > ho_scores[best_idx]) best_idx = j;
        
        /* Check if best match has matching output dimensions (proxy for transform compatibility) */
        int dim_match = (tasks[best_idx].pairs[0].output.width == tasks[i].pairs[0].output.width &&
                         tasks[best_idx].pairs[0].output.height == tasks[i].pairs[0].output.height);
        
        if (dim_match) solved++;
        
        fprintf(fp, "{\"schema\":\"crlp.arc_discovery.v1\",\"task\":\"%s\",\"best_match\":\"%s\",\"match\":%s}\n",
                tasks[i].id, tasks[best_idx].id, dim_match ? "true" : "false");
    }
    
    fprintf(fp, "{\"schema\":\"crlp.arc_discovery.summary.v1\",\"n_train\":%zu,\"n_test\":%zu,\"solved\":%zu}\n",
            n_train, n_tasks - n_train, solved);
    free(act_embs);
    
    printf("ARC Discovery: %zu/%zu test tasks matched by holo retrieval (%.1f%%)\n",
           solved, n_tasks - n_train, (double)solved / (n_tasks - n_train) * 100.0);
    
    holo_ranker_destroy(holo); leos_engine_destroy(eng);
    free(tasks); fclose(fp);
    return 0;
}
