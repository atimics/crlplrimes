/*
 * hologram_merge.c — Merge holographic traces from multiple grind sessions.
 *
 * Reads one or more JSON files produced by vanity_hologram, decodes their
 * trace vectors, adds them (holographic superposition), and writes a
 * combined JSON proof bundle.
 *
 * Build:  make build/float/hologram_merge
 * Usage:  ./build/float/hologram_merge a.json b.json --out merged.json
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>

#define MAX_FILES 64
#define MAX_LINE  131072

typedef struct {
    size_t  dim;
    size_t  positions;
    double *trace;
    double  fidelity;
    int     valid;
} HologramFile;

static size_t hex_decode(const char *hex, unsigned char *out, size_t out_cap) {
    size_t len = strlen(hex);
    if (len % 2 != 0) return 0;
    size_t byte_len = len / 2;
    if (byte_len > out_cap) return 0;
    for (size_t i = 0; i < byte_len; i++) {
        char hi = hex[i * 2], lo = hex[i * 2 + 1];
        int v = 0;
        if (hi >= '0' && hi <= '9') v = (hi - '0') << 4;
        else if (hi >= 'a' && hi <= 'f') v = (hi - 'a' + 10) << 4;
        else if (hi >= 'A' && hi <= 'F') v = (hi - 'A' + 10) << 4;
        else return 0;
        if (lo >= '0' && lo <= '9') v |= (lo - '0');
        else if (lo >= 'a' && lo <= 'f') v |= (lo - 'a' + 10);
        else if (lo >= 'A' && lo <= 'F') v |= (lo - 'A' + 10);
        else return 0;
        out[i] = (unsigned char)v;
    }
    return byte_len;
}

static int json_get_string(const char *line, const char *key,
                           char *out, size_t out_cap) {
    char search[256];
    snprintf(search, sizeof(search), "\"%s\":\"", key);
    const char *p = strstr(line, search);
    if (!p) { snprintf(search, sizeof(search), "\"%s\": \"", key); p = strstr(line, search); }
    if (!p) {
        snprintf(search, sizeof(search), "\"%s\": \"", key);
        p = strstr(line, search);
    }
    if (!p) return -1;
    p += strlen(search);
    const char *end = strchr(p, '"');
    if (!end) return -1;
    size_t len = (size_t)(end - p);
    if (len >= out_cap) len = out_cap - 1;
    memcpy(out, p, len);
    out[len] = '\0';
    return 0;
}

static int json_get_size(const char *line, const char *key, size_t *out) {
    char search[256];
    snprintf(search, sizeof(search), "\"%s\":", key);
    const char *p = strstr(line, search);
    if (!p) {
        snprintf(search, sizeof(search), "\"%s\": \"", key);
        p = strstr(line, search);
    }
    if (!p) return -1;
    p += strlen(search);
    *out = (size_t)atoll(p);
    return 0;
}

static int json_get_double(const char *line, const char *key, double *out) {
    char search[256];
    snprintf(search, sizeof(search), "\"%s\":", key);
    const char *p = strstr(line, search);
    if (!p) {
        snprintf(search, sizeof(search), "\"%s\": \"", key);
        p = strstr(line, search);
    }
    if (!p) return -1;
    p += strlen(search);
    *out = atof(p);
    return 0;
}

static int load_hologram(const char *path, HologramFile *hf) {
    memset(hf, 0, sizeof(*hf));
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Error: cannot open %s\n", path); return -1; }
    /* Read entire file (handles pretty-printed multi-line JSON). */
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    if (fsize <= 0 || fsize >= MAX_LINE) {
        fprintf(stderr, "Error: file too large or empty: %s\n", path);
        fclose(f); return -1;
    }
    fseek(f, 0, SEEK_SET);
    char *line = (char *)malloc((size_t)fsize + 1);
    if (!line || fread(line, 1, (size_t)fsize, f) != (size_t)fsize) {
        fprintf(stderr, "Error: read failed %s\n", path);
        free(line); fclose(f); return -1;
    }
    line[fsize] = '\0';
    fclose(f);

    const char *ho = strstr(line, "\"hologram\":{");
    const char *ctx = ho ? ho : line;

    if (json_get_size(ctx, "dim", &hf->dim) != 0 ||
        json_get_size(ctx, "positions", &hf->positions) != 0) {
        fprintf(stderr, "Error: missing dim/positions in %s\n", path);
        free(line); return -1;
    }

    char trace_hex[131072];
    if (json_get_string(ctx, "traceHex", trace_hex, sizeof(trace_hex)) != 0) {
        fprintf(stderr, "Error: missing traceHex in %s\n", path);
        free(line); return -1;
    }

    size_t byte_len = hf->dim * sizeof(double);
    unsigned char *raw = (unsigned char *)malloc(byte_len);
    if (!raw || hex_decode(trace_hex, raw, byte_len) != byte_len) {
        fprintf(stderr, "Error: trace hex decode failed (%s)\n", path);
        free(raw); free(line); return -1;
    }

    hf->trace = (double *)raw;
    json_get_double(ctx, "fidelity", &hf->fidelity);
    hf->valid = 1;
    printf("  loaded %s: dim=%zu pos=%zu fidelity=%.6f\n",
           path, hf->dim, hf->positions, hf->fidelity);
    free(line);
    return 0;
}

int main(int argc, char **argv) {
    const char *out_path = NULL;
    const char *files[MAX_FILES];
    int nfiles = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--out") && i + 1 < argc)
            out_path = argv[++i];
        else if (argv[i][0] != '-') {
            if (nfiles >= MAX_FILES) {
                fprintf(stderr, "Error: too many input files (max %d)\n", MAX_FILES);
                return 1;
            }
            files[nfiles++] = argv[i];
        }
    }

    if (nfiles < 2 || !out_path) {
        fprintf(stderr, "usage: %s a.json b.json [...] --out merged.json\n", argv[0]);
        return 1;
    }

    HologramFile hf[MAX_FILES];
    size_t dim = 0, total_pos = 0;
    for (int i = 0; i < nfiles; i++) {
        if (load_hologram(files[i], &hf[i]) != 0) return 1;
        if (i == 0) dim = hf[i].dim;
        if (hf[i].dim != dim) {
            fprintf(stderr, "Error: dim mismatch in %s\n", files[i]);
            return 1;
        }
        total_pos += hf[i].positions;
    }

    double *merged = (double *)calloc(dim, sizeof(double));
    if (!merged) { fprintf(stderr, "Error: malloc\n"); return 1; }

    for (int i = 0; i < nfiles; i++)
        for (size_t j = 0; j < dim; j++)
            merged[j] += hf[i].trace[j];

    double norm = 0.0;
    for (size_t j = 0; j < dim; j++) norm += merged[j] * merged[j];
    norm = sqrt(norm);
    if (norm > 0.0)
        for (size_t j = 0; j < dim; j++) merged[j] /= norm;

    double fidelity = total_pos > 0 ? 1.0 / sqrt((double)total_pos) : 0.0;

    size_t byte_len = dim * sizeof(double);
    char *th = (char *)malloc(byte_len * 2 + 1);
    const unsigned char *raw = (const unsigned char *)merged;
    for (size_t i = 0; i < byte_len; i++)
        sprintf(th + i * 2, "%02x", raw[i]);
    th[byte_len * 2] = '\0';

    printf("\n  Cross-correlations:\n");
    for (int i = 0; i < nfiles; i++) {
        double dot = 0.0;
        for (size_t j = 0; j < dim; j++) dot += merged[j] * hf[i].trace[j];
        printf("    dot(merged, trace_%d) = %+.6f\n", i, dot);
    }
    printf("    noise floor = %.6f\n", 2.0 / sqrt((double)dim));

    FILE *out = fopen(out_path, "w");
    if (!out) { fprintf(stderr, "Error: cannot write %s\n", out_path); return 1; }
    fprintf(out, "{\"contributors\":%d,\"totalPositions\":%llu,",
            nfiles, (unsigned long long)total_pos);
    fprintf(out, "\"hologram\":{\"dim\":%zu,\"traceHex\":\"%s\","
            "\"positions\":%llu,\"fidelity\":%.6f}",
            dim, th, (unsigned long long)total_pos, fidelity);
    fprintf(out, "}\n");
    fclose(out);

    printf("  wrote: %s\n", out_path);
    for (int i = 0; i < nfiles; i++) free(hf[i].trace);
    free(merged); free(th);
    return 0;
}

