#include "prime_gap.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>


const uint32_t FEATURE_MODS[FEATURE_MOD_COUNT] = {3, 5, 7, 11, 13, 17, 19, 23, 29, 31};
const uint32_t INTERVAL_SWEEP_GAPS[INTERVAL_SWEEP_COUNT] = {512, 1024, 2048};
const nn_real SUPERVISED_ENTROPY_WEIGHT = (nn_real)0.10;

const FeatureMask FEATURE_MASK_FULL = {"full", 1, 1, 1, RESIDUE_FEATURE_DIVISIBILITY};
const FeatureMask FEATURE_MASK_LOG_ONLY = {"log-only", 1, 0, 0, RESIDUE_FEATURE_NONE};
const FeatureMask FEATURE_MASK_GAP_ONLY = {"gap-only", 0, 1, 1, RESIDUE_FEATURE_NONE};
const FeatureMask FEATURE_MASK_RESIDUE_ONLY = {"residue-only", 0, 0, 1, RESIDUE_FEATURE_DIVISIBILITY};
const FeatureMask FEATURE_MASK_RAW_RESIDUE = {"raw-residue", 1, 1, 0, RESIDUE_FEATURE_RAW_NORMALIZED};

double now_seconds(void)
{
#if defined(CLOCK_MONOTONIC)
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0) {
        return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
    }
#endif
    return (double)clock() / (double)CLOCKS_PER_SEC;
}

uint32_t rng_next(uint32_t *state)
{
    *state = (*state * 1664525u) + 1013904223u;
    return *state;
}

double rng_uniform(uint32_t *state)
{
    return (double)rng_next(state) / (double)UINT32_MAX;
}

size_t rng_bounded(uint32_t *state, size_t bound)
{
    return bound == 0 ? 0 : (size_t)(rng_next(state) % (uint32_t)bound);
}

size_t parse_size_arg(int argc, char **argv, int index, size_t fallback)
{
    if (index >= argc) {
        return fallback;
    }

    char *end = NULL;
    unsigned long value = strtoul(argv[index], &end, 10);
    return end != argv[index] && *end == '\0' ? (size_t)value : fallback;
}

void free_sieve(Sieve *sieve)
{
    free(sieve->composite);
    free(sieve->primes);
    memset(sieve, 0, sizeof(*sieve));
}

int build_sieve(uint32_t limit, Sieve *sieve)
{
    memset(sieve, 0, sizeof(*sieve));
    sieve->limit = limit;
    sieve->composite = calloc((size_t)limit + 1, sizeof(*sieve->composite));
    if (sieve->composite == NULL) {
        return 0;
    }

    if (limit >= 1) {
        sieve->composite[0] = 1;
        sieve->composite[1] = 1;
    }

    for (uint32_t p = 2; (uint64_t)p * (uint64_t)p <= limit; ++p) {
        if (sieve->composite[p]) {
            continue;
        }

        for (uint64_t multiple = (uint64_t)p * (uint64_t)p; multiple <= limit; multiple += p) {
            sieve->composite[multiple] = 1;
        }
    }

    for (uint32_t n = 2; n <= limit; ++n) {
        sieve->prime_count += sieve->composite[n] ? 0u : 1u;
    }

    sieve->primes = calloc(sieve->prime_count, sizeof(*sieve->primes));
    if (sieve->primes == NULL) {
        free_sieve(sieve);
        return 0;
    }

    size_t index = 0;
    for (uint32_t n = 2; n <= limit; ++n) {
        if (!sieve->composite[n]) {
            sieve->primes[index++] = n;
        }
    }

    return 1;
}

int is_prime_sieve(const Sieve *sieve, uint32_t n)
{
    return n <= sieve->limit && n >= 2 && !sieve->composite[n];
}

void free_gap_dataset(GapDataset *dataset)
{
    free(dataset->prime);
    free(dataset->gap);
    memset(dataset, 0, sizeof(*dataset));
}

int append_gap(GapDataset *dataset, uint32_t prime, uint32_t gap)
{
    if (dataset->count == dataset->capacity) {
        size_t next_capacity = dataset->capacity == 0 ? 1024 : dataset->capacity * 2;
        uint32_t *next_prime = calloc(next_capacity, sizeof(*next_prime));
        uint32_t *next_gap = calloc(next_capacity, sizeof(*next_gap));
        if (next_prime == NULL || next_gap == NULL) {
            free(next_prime);
            free(next_gap);
            return 0;
        }

        if (dataset->count != 0) {
            memcpy(next_prime, dataset->prime, dataset->count * sizeof(*next_prime));
            memcpy(next_gap, dataset->gap, dataset->count * sizeof(*next_gap));
        }

        free(dataset->prime);
        free(dataset->gap);
        dataset->prime = next_prime;
        dataset->gap = next_gap;
        dataset->capacity = next_capacity;
    }

    dataset->prime[dataset->count] = prime;
    dataset->gap[dataset->count] = gap;
    ++dataset->count;
    return 1;
}

int build_gap_datasets(const Sieve *sieve,
                              uint32_t train_limit,
                              uint32_t test_limit,
                              GapDataset *train,
                              GapDataset *test)
{
    memset(train, 0, sizeof(*train));
    memset(test, 0, sizeof(*test));

    for (size_t i = 0; i + 1 < sieve->prime_count; ++i) {
        uint32_t p = sieve->primes[i];
        uint32_t next = sieve->primes[i + 1];
        uint32_t gap = next - p;

        if (p < 3 || (gap & 1u) != 0) {
            continue;
        }

        GapDataset *target = NULL;
        if (p < train_limit) {
            target = train;
        } else if (p >= train_limit && next <= test_limit) {
            target = test;
        }

        if (target == NULL) {
            continue;
        }

        ++target->total_seen;
        if (gap > G_MAX) {
            ++target->overflow_count;
        }

        if (!append_gap(target, p, gap)) {
            free_gap_dataset(train);
            free_gap_dataset(test);
            return 0;
        }
    }

    return 1;
}

int copy_gap_slice(const GapDataset *source, size_t start, size_t count, GapDataset *target)
{
    memset(target, 0, sizeof(*target));

    if (start > source->count) {
        return 0;
    }
    if (count > source->count - start) {
        count = source->count - start;
    }

    for (size_t i = 0; i < count; ++i) {
        uint32_t p = source->prime[start + i];
        uint32_t gap = source->gap[start + i];
        ++target->total_seen;
        if (gap > G_MAX) {
            ++target->overflow_count;
        }
        if (!append_gap(target, p, gap)) {
            free_gap_dataset(target);
            return 0;
        }
    }

    return 1;
}

int split_train_calibration(const GapDataset *source,
                                   GapDataset *fit,
                                   GapDataset *calibration)
{
    size_t calibration_count = source->count / 5u;
    size_t fit_count;

    if (source->count < 2) {
        return 0;
    }
    if (calibration_count == 0) {
        calibration_count = 1;
    }
    fit_count = source->count - calibration_count;
    if (fit_count == 0) {
        return 0;
    }

    if (!copy_gap_slice(source, 0, fit_count, fit)) {
        return 0;
    }
    if (!copy_gap_slice(source, fit_count, calibration_count, calibration)) {
        free_gap_dataset(fit);
        return 0;
    }

    return 1;
}

void shuffle_order(size_t *order, size_t count, uint32_t *rng)
{
    for (size_t i = count; i > 1; --i) {
        size_t j = rng_bounded(rng, i);
        size_t tmp = order[i - 1];
        order[i - 1] = order[j];
        order[j] = tmp;
    }
}
