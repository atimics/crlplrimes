/*
 * vanity_hologram.c — Post-hoc holographic proof of a vanity grind.
 *
 * Takes a master seed (Ed25519 secret key bytes, hex-encoded) and attempt
 * count, reconstructs the deterministic seed chain, builds a compact
 * holographic trace of the scanned keypairs, and outputs a JSON proof
 * bundle suitable for Arweave token metadata.
 *
 * Build:  make build/float/vanity_hologram
 * Usage:  ./build/float/vanity_hologram --seed-hex <64 hex> --attempts N
 *              [--dim D] [--sample-rate R] [--out file.json]
 *
 * The hologram trace is additive — traces from different grind sessions
 * (different seeds) can be summed to produce a combined proof of collective
 * work.  Merging requires no knowledge of individual seeds.
 */

#include <sodium.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "leos_engine.h"

#define DEFAULT_DIM        512
#define DEFAULT_SAMPLE_RATE 8    /* accumulate every Nth key */
#define MAX_TRACE_HEX      65536

/* Hash 32 bytes into a uint64_t for deterministic keygen seeding. */
static uint64_t hash32_to_u64(const uint8_t bytes[32]) {
    uint64_t h = 0;
    for (int i = 0; i < 4; i++) {
        uint64_t word = 0;
        for (int j = 0; j < 8; j++) word = (word << 8) | (uint64_t)bytes[i * 8 + j];
        h ^= word;
    }
    /* SplitMix64 finalizer */
    h ^= h >> 30; h *= 0xbf58476d1ce4e5b9ULL;
    h ^= h >> 27; h *= 0x94d049bb133111ebULL;
    h ^= h >> 31;
    return h;
}

static void hex_decode(const char *hex, uint8_t *out, int out_len) {
    for (int i = 0; i < out_len; i++) {
        char hi = hex[i * 2], lo = hex[i * 2 + 1];
        int v = 0;
        if (hi >= '0' && hi <= '9') v = (hi - '0') << 4;
        else if (hi >= 'a' && hi <= 'f') v = (hi - 'a' + 10) << 4;
        else if (hi >= 'A' && hi <= 'F') v = (hi - 'A' + 10) << 4;
        if (lo >= '0' && lo <= '9') v |= (lo - '0');
        else if (lo >= 'a' && lo <= 'f') v |= (lo - 'a' + 10);
        else if (lo >= 'A' && lo <= 'F') v |= (lo - 'A' + 10);
        out[i] = (uint8_t)v;
    }
}

/* Hex-encode the trace doubles into out.
   Production Arweave: upload raw binary as separate file, reference URI. */
static size_t encode_trace_hex(const double *trace, size_t dim,
                               char *out, size_t out_cap) {
    size_t byte_len = dim * sizeof(double);
    if (byte_len * 2 + 1 > out_cap) return 0;
    const uint8_t *raw = (const uint8_t *)trace;
    for (size_t i = 0; i < byte_len; i++)
        sprintf(out + i * 2, "%02x", raw[i]);
    out[byte_len * 2] = '\0';
    return byte_len * 2;
}

int main(int argc, char **argv) {
    const char *seed_hex = NULL;
    const char *out_path = NULL;
    uint64_t    attempts = 0;
    size_t      dim = DEFAULT_DIM;
    int         sample_rate = DEFAULT_SAMPLE_RATE;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--seed-hex") && i + 1 < argc)
            seed_hex = argv[++i];
        else if (!strcmp(argv[i], "--attempts") && i + 1 < argc)
            attempts = (uint64_t)atoll(argv[++i]);
        else if (!strcmp(argv[i], "--dim") && i + 1 < argc)
            dim = (size_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--sample-rate") && i + 1 < argc)
            sample_rate = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--out") && i + 1 < argc)
            out_path = argv[++i];
        else {
            fprintf(stderr,
                "usage: %s --seed-hex <HEX> --attempts N [--dim D] "
                "[--sample-rate R] [--out FILE]\n", argv[0]);
            return 1;
        }
    }

    if (!seed_hex || attempts == 0) {
        fprintf(stderr, "Error: --seed-hex and --attempts are required\n");
        return 1;
    }
    if (strlen(seed_hex) != 64) {
        fprintf(stderr, "Error: --seed-hex must be 64 hex chars (32 bytes)\n");
        return 1;
    }
    if (sample_rate < 1) sample_rate = 1;

    if (sodium_init() < 0) {
        fprintf(stderr, "Error: libsodium init failed\n");
        return 1;
    }

    /* Decode master seed. */
    uint8_t master_seed[32];
    hex_decode(seed_hex, master_seed, 32);

    /* Allocate hologram trace (zero-initialized). */
    double *trace = NULL;
    double *pos_vec = NULL;
    double *key_vec = NULL;
    double *bound = NULL;
    LeosEngine *eng = NULL;

    trace   = (double *)calloc(dim, sizeof(double));
    pos_vec = (double *)malloc(dim * sizeof(double));
    key_vec = (double *)malloc(dim * sizeof(double));
    bound   = (double *)malloc(dim * sizeof(double));
    if (!trace || !pos_vec || !key_vec || !bound) {
        fprintf(stderr, "Error: malloc failed\n");
        goto cleanup;
    }

    /* Create LEOS engine. Dimension must be power of 2. */
    eng = leos_engine_create(dim);
    if (!eng) {
        fprintf(stderr, "Error: failed to create LEOS engine (dim=%zu)\n", dim);
        goto cleanup;
    }

    /* Reconstruct the seed chain and accumulate hologram.
       seed_{i+1} = pk_i[0..31]  (deterministic, one-way).
       Accumulate every sample_rate-th keypair. */
    uint8_t seed[32], pk[32], sk[64];
    memcpy(seed, master_seed, 32);

    /* Derive first keypair from master seed, then use its pk as next seed. */
    if (crypto_sign_seed_keypair(pk, sk, seed) != 0) {
        fprintf(stderr, "Error: keypair derivation failed\n");
        goto cleanup;
    }
    memcpy(seed, pk, 32);

    uint64_t accumulated = 0;
    fprintf(stderr, "Building hologram: dim=%zu sample_rate=%d total_attempts=%llu\n",
            dim, sample_rate, (unsigned long long)attempts);

    for (uint64_t i = 0; i < attempts; i++) {
        if (crypto_sign_seed_keypair(pk, sk, seed) != 0) {
            fprintf(stderr, "Error: keypair derivation at %llu\n",
                    (unsigned long long)i);
            goto cleanup;
        }

        if (i % (uint64_t)sample_rate == 0) {
            /* Position vector: deterministic from index. */
            leos_engine_keygen(eng, i * 0x9e3779b97f4a7c15ULL, pos_vec);
            /* Key vector: deterministic from public key. */
            leos_engine_keygen(eng, hash32_to_u64(pk), key_vec);
            /* bind(position, key) */
            leos_engine_bind(eng, pos_vec, key_vec, bound);
            /* Accumulate: trace += bound */
            leos_engine_add(eng, trace, bound, trace);
            accumulated++;
        }

        /* Advance chain: next seed = current pk. */
        memcpy(seed, pk, 32);

        if ((i & 0xFFFFF) == 0xFFFFF) {
            fprintf(stderr, "\r  progress: %llu / %llu  accumulated: %llu",
                    (unsigned long long)(i + 1),
                    (unsigned long long)attempts,
                    (unsigned long long)accumulated);
            fflush(stderr);
        }
    }
    fprintf(stderr, "\r  progress: %llu / %llu  accumulated: %llu\n",
            (unsigned long long)attempts,
            (unsigned long long)attempts,
            (unsigned long long)accumulated);

    /* Normalize the trace for stability. */
    leos_engine_normalize(eng, trace);

    /* Compute fidelity: 1/sqrt(accumulated). */
    double fidelity = accumulated > 0 ? 1.0 / sqrt((double)accumulated) : 0.0;

    /* Hex-encode the trace. */
    char trace_hex[MAX_TRACE_HEX];
    size_t hex_len = encode_trace_hex(trace, dim, trace_hex, sizeof(trace_hex));
    if (hex_len == 0) {
        fprintf(stderr, "Error: base58 encoding overflow\n");
        return 1;
    }

    /* Build JSON output. */
    char json[65536];
    int off = 0;
    off += snprintf(json + off, sizeof(json) - (size_t)off, "{");
    off += snprintf(json + off, sizeof(json) - (size_t)off,
        "\"attempts\":%llu,", (unsigned long long)attempts);
    off += snprintf(json + off, sizeof(json) - (size_t)off,
        "\"sampleRate\":%d,", sample_rate);
    off += snprintf(json + off, sizeof(json) - (size_t)off,
        "\"hologram\":{");
    off += snprintf(json + off, sizeof(json) - (size_t)off,
        "\"dim\":%zu,", dim);
    off += snprintf(json + off, sizeof(json) - (size_t)off,
        "\"traceHex\":\"%s\",", trace_hex);
    off += snprintf(json + off, sizeof(json) - (size_t)off,
        "\"positions\":%llu,", (unsigned long long)accumulated);
    off += snprintf(json + off, sizeof(json) - (size_t)off,
        "\"fidelity\":%.6f", fidelity);
    off += snprintf(json + off, sizeof(json) - (size_t)off, "}");

    /* Fidelity diagnostic: expected signal-to-noise ratio. */
    {
        double noise_floor = 2.0 / sqrt((double)dim);
        double snr = fidelity / noise_floor;
        off += snprintf(json + off, sizeof(json) - (size_t)off,
            ",\"selfCheck\":{\"fidelity\":%.6f,\"noiseFloor\":%.6f,\"snr\":%.4f,\"accumulated\":%llu}",
            fidelity, noise_floor, snr, (unsigned long long)accumulated);

        fprintf(stderr, "  Hologram: dim=%zu accumulated=%llu fidelity=%.6f noise_floor=%.6f snr=%.4f\n",
                dim, (unsigned long long)accumulated, fidelity, noise_floor, snr);
    }

    off += snprintf(json + off, sizeof(json) - (size_t)off, "}\n");

    if (out_path) {
        FILE *f = fopen(out_path, "w");
        if (!f) { fprintf(stderr, "Error: cannot write %s\n", out_path); return 1; }
        fprintf(f, "%s", json);
        fclose(f);
        fprintf(stderr, "  Wrote: %s\n", out_path);
    } else {
        printf("%s", json);
    }

    free(trace); free(pos_vec); free(key_vec); free(bound);
    leos_engine_destroy(eng);
    return 0;

cleanup:
    free(trace); free(pos_vec); free(key_vec); free(bound);
    if (eng) leos_engine_destroy(eng);
    return 1;
}

