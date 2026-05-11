#include "prime_gap.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

int wheel_probe_gap(const Sieve *sieve, uint32_t p, uint32_t gap, VerifierWork *work)
{
    uint32_t candidate = p + gap;

    ++work->candidates;
    for (size_t i = 0; i < FEATURE_MOD_COUNT; ++i) {
        uint32_t mod = FEATURE_MODS[i];
        if (candidate <= mod) {
            continue;
        }

        ++work->mod_tests;
        if (candidate % mod == 0) {
            return 0;
        }
    }

    ++work->exact_tests;
    return is_prime_sieve(sieve, candidate);
}

int wheel_probe_action_cached(const Sieve *sieve,
                                     uint32_t p,
                                     size_t action,
                                     uint8_t *tested,
                                     uint8_t *prime_result,
                                     VerifierWork *work)
{
    if (action_is_overflow(action)) {
        return 0;
    }

    if (!tested[action]) {
        tested[action] = 1;
        prime_result[action] =
            (uint8_t)wheel_probe_gap(sieve, p, (uint32_t)action_gap(action), work);
    }

    return prime_result[action] != 0;
}

int wheel_scan_to_gap_cached(const Sieve *sieve,
                                    uint32_t p,
                                    uint32_t limit_gap,
                                    uint8_t *tested,
                                    uint8_t *prime_result,
                                    VerifierWork *work)
{
    for (uint32_t gap = 2; gap <= limit_gap; gap += 2) {
        int is_prime = 0;
        if (gap <= G_MAX) {
            is_prime = wheel_probe_action_cached(
                sieve, p, gap_to_action(gap), tested, prime_result, work);
        } else {
            is_prime = wheel_probe_gap(sieve, p, gap, work);
        }

        if (is_prime) {
            return 1;
        }
    }

    return 0;
}

VerifierWork direct_scan_work(uint32_t true_gap)
{
    VerifierWork work;
    work.candidates = true_gap / 2u;
    work.mod_tests = 0;
    work.exact_tests = true_gap / 2u;
    return work;
}

VerifierWork wheel_scan_work(const Sieve *sieve, uint32_t p, uint32_t true_gap)
{
    uint8_t tested[EXACT_ACTION_COUNT] = {0};
    uint8_t prime_result[EXACT_ACTION_COUNT] = {0};
    VerifierWork work = {0, 0, 0};

    wheel_scan_to_gap_cached(sieve, p, true_gap, tested, prime_result, &work);
    return work;
}

VerifierWork assisted_wheel_work(const Sieve *sieve,
                                        uint32_t p,
                                        uint32_t true_gap,
                                        const size_t *ranked,
                                        size_t top_k)
{
    uint8_t tested[EXACT_ACTION_COUNT] = {0};
    uint8_t prime_result[EXACT_ACTION_COUNT] = {0};
    VerifierWork work = {0, 0, 0};

    if (top_k > ACTION_COUNT) {
        top_k = ACTION_COUNT;
    }

    for (size_t i = 0; i < top_k; ++i) {
        size_t proposed = ranked[i];
        if (action_is_overflow(proposed)) {
            wheel_scan_to_gap_cached(sieve, p, true_gap, tested, prime_result, &work);
            return work;
        }

        if (!wheel_probe_action_cached(sieve, p, proposed, tested, prime_result, &work)) {
            continue;
        }

        wheel_scan_to_gap_cached(
            sieve, p, (uint32_t)action_gap(proposed), tested, prime_result, &work);
        return work;
    }

    wheel_scan_to_gap_cached(sieve, p, true_gap, tested, prime_result, &work);
    return work;
}

void add_block_work(BlockWork *total, BlockWork part)
{
    total->candidates += part.candidates;
    total->base_primes += part.base_primes;
    total->marks += part.marks;
    total->survivor_checks += part.survivor_checks;
}

double block_work_cost(BlockWork work)
{
    return (double)work.base_primes + (double)work.marks + (double)work.survivor_checks;
}

uint32_t segmented_first_prime_gap(const Sieve *sieve,
                                          uint32_t p,
                                          uint32_t limit_gap,
                                          BlockWork *work)
{
    uint32_t candidate_count = limit_gap / 2u;
    uint8_t local_composite[EXACT_ACTION_COUNT];
    uint8_t *composite = local_composite;
    uint32_t start_number = p + 2u;
    uint32_t end_number = p + limit_gap;
    uint32_t found_gap = 0;

    memset(work, 0, sizeof(*work));
    work->candidates = candidate_count;

    if (candidate_count == 0) {
        return 0;
    }

    if (candidate_count <= EXACT_ACTION_COUNT) {
        memset(local_composite, 0, candidate_count * sizeof(*local_composite));
    } else {
        composite = calloc(candidate_count, sizeof(*composite));
        if (composite == NULL) {
            return 0;
        }
    }

    for (size_t i = 0; i < sieve->prime_count; ++i) {
        uint32_t prime = sieve->primes[i];
        uint64_t prime_squared = (uint64_t)prime * (uint64_t)prime;
        if (prime == 2) {
            continue;
        }
        if (prime_squared > end_number) {
            break;
        }

        ++work->base_primes;

        uint64_t first = ((uint64_t)start_number + prime - 1u) / prime * prime;
        if (first < prime_squared) {
            first = prime_squared;
        }
        if ((first & 1u) == 0) {
            first += prime;
        }

        for (uint64_t value = first; value <= end_number; value += (uint64_t)2u * prime) {
            uint32_t gap = (uint32_t)(value - p);
            if (gap < 2 || (gap & 1u) != 0) {
                continue;
            }

            uint32_t index = gap / 2u - 1u;
            if (index < candidate_count && !composite[index]) {
                composite[index] = 1;
                ++work->marks;
            }
        }
    }

    for (uint32_t index = 0; index < candidate_count; ++index) {
        ++work->survivor_checks;
        if (!composite[index]) {
            found_gap = (index + 1u) * 2u;
            break;
        }
    }

    if (composite != local_composite) {
        free(composite);
    }

    return found_gap;
}

BlockWork block_scan_work(const Sieve *sieve, uint32_t p, uint32_t true_gap)
{
    BlockWork work;
    segmented_first_prime_gap(sieve, p, true_gap, &work);
    return work;
}

BlockWork assisted_block_work(const Sieve *sieve,
                                     uint32_t p,
                                     uint32_t true_gap,
                                     const size_t *ranked,
                                     size_t top_k)
{
    BlockWork total = {0, 0, 0, 0};

    if (top_k > ACTION_COUNT) {
        top_k = ACTION_COUNT;
    }

    for (size_t i = 0; i < top_k; ++i) {
        size_t proposed = ranked[i];
        BlockWork part;
        uint32_t found_gap;

        if (action_is_overflow(proposed)) {
            add_block_work(&total, block_scan_work(sieve, p, true_gap));
            return total;
        }

        found_gap =
            segmented_first_prime_gap(sieve, p, (uint32_t)action_gap(proposed), &part);
        add_block_work(&total, part);

        if (found_gap != 0) {
            return total;
        }
    }

    add_block_work(&total, block_scan_work(sieve, p, true_gap));
    return total;
}
