/* Prime sieve, dataset construction, ordering helpers, and action/gap mapping. */

#include "internal.h"


uint32_t validation_start_limit(uint32_t train_limit)
{
    uint32_t holdout = train_limit / 5u;
    if (holdout == 0 || holdout >= train_limit - 3u) {
        return train_limit;
    }
    return train_limit - holdout;
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
                       GapDataset *validation,
                       GapDataset *test)
{
    memset(train, 0, sizeof(*train));
    memset(validation, 0, sizeof(*validation));
    memset(test, 0, sizeof(*test));

    uint32_t validation_start = validation_start_limit(train_limit);

    for (size_t i = 0; i + 1 < sieve->prime_count; ++i) {
        uint32_t p = sieve->primes[i];
        uint32_t next = sieve->primes[i + 1];
        uint32_t gap = next - p;

        if (p < 3 || (gap & 1u) != 0) {
            continue;
        }

        GapDataset *target = NULL;
        if (p < validation_start) {
            target = train;
        } else if (p < train_limit) {
            target = validation;
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
            free_gap_dataset(validation);
            free_gap_dataset(test);
            return 0;
        }
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

void fill_order(size_t *order, size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        order[i] = i;
    }
}

size_t action_gap(size_t action)
{
    if (action == OVERFLOW_ACTION) {
        return G_MAX + 2;
    }

    return (action + 1) * 2;
}

size_t gap_to_action(uint32_t gap)
{
    if (gap > G_MAX) {
        return OVERFLOW_ACTION;
    }

    return (size_t)(gap / 2u - 1u);
}

int action_is_overflow(size_t action)
{
    return action == OVERFLOW_ACTION;
}
