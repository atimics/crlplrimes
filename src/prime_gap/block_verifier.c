/* Segmented block verifier diagnostics and assisted block scans.
 * Included by ../nn_tasks_prime_gap.c; not compiled as a standalone translation unit.
 */

static void add_block_work(BlockWork *total, BlockWork part)
{
    total->candidates += part.candidates;
    total->base_primes += part.base_primes;
    total->marks += part.marks;
    total->survivor_checks += part.survivor_checks;
}

static double block_work_cost(BlockWork work)
{
    return (double)work.base_primes + (double)work.marks + (double)work.survivor_checks;
}

static uint32_t segmented_first_prime_gap(const Sieve *sieve,
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

static BlockWork block_scan_work(const Sieve *sieve, uint32_t p, uint32_t true_gap)
{
    BlockWork work;
    segmented_first_prime_gap(sieve, p, true_gap, &work);
    return work;
}

static uint32_t extend_segmented_first_prime_gap(const Sieve *sieve,
                                                 uint32_t p,
                                                 uint32_t from_gap,
                                                 uint32_t to_gap,
                                                 uint8_t *composite,
                                                 BlockWork *work)
{
    uint32_t old_candidate_count = from_gap / 2u;
    uint32_t candidate_count = to_gap / 2u;
    uint64_t start_number = (uint64_t)p + (uint64_t)from_gap + 2u;
    uint64_t end_number = (uint64_t)p + (uint64_t)to_gap;

    memset(work, 0, sizeof(*work));
    if (to_gap <= from_gap) {
        return 0;
    }

    work->candidates = candidate_count - old_candidate_count;

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
            uint32_t gap = (uint32_t)(value - (uint64_t)p);
            if (gap < 2 || (gap & 1u) != 0) {
                continue;
            }

            uint32_t index = gap / 2u - 1u;
            if (index >= old_candidate_count && index < candidate_count && !composite[index]) {
                composite[index] = 1;
                ++work->marks;
            }
        }
    }

    for (uint32_t index = old_candidate_count; index < candidate_count; ++index) {
        ++work->survivor_checks;
        if (!composite[index]) {
            return (index + 1u) * 2u;
        }
    }

    return 0;
}

static BlockWork assisted_block_work(const Sieve *sieve,
                                     uint32_t p,
                                     uint32_t true_gap,
                                     const size_t *ranked,
                                     size_t top_k)
{
    BlockWork total = {0, 0, 0, 0};
    uint32_t max_gap = true_gap;
    uint32_t verified_gap = 0;
    uint32_t candidate_count;
    uint8_t local_composite[EXACT_ACTION_COUNT];
    uint8_t *composite = local_composite;

    if (top_k > ACTION_COUNT) {
        top_k = ACTION_COUNT;
    }

    for (size_t i = 0; i < top_k; ++i) {
        size_t proposed = ranked[i];
        if (!action_is_overflow(proposed) && action_gap(proposed) > max_gap) {
            max_gap = (uint32_t)action_gap(proposed);
        }
    }

    candidate_count = max_gap / 2u;
    if (candidate_count == 0) {
        return total;
    }

    if (candidate_count <= EXACT_ACTION_COUNT) {
        memset(local_composite, 0, candidate_count * sizeof(*local_composite));
    } else {
        composite = calloc(candidate_count, sizeof(*composite));
        if (composite == NULL) {
            return block_scan_work(sieve, p, true_gap);
        }
    }

    for (size_t i = 0; i < top_k; ++i) {
        size_t proposed = ranked[i];
        BlockWork part;
        uint32_t found_gap;

        if (action_is_overflow(proposed)) {
            break;
        }

        uint32_t proposed_gap = (uint32_t)action_gap(proposed);
        if (proposed_gap <= verified_gap) {
            continue;
        }

        found_gap =
            extend_segmented_first_prime_gap(sieve, p, verified_gap, proposed_gap, composite, &part);
        add_block_work(&total, part);
        verified_gap = proposed_gap;

        if (found_gap != 0) {
            if (composite != local_composite) {
                free(composite);
            }
            return total;
        }
    }

    if (true_gap > verified_gap) {
        BlockWork part;
        extend_segmented_first_prime_gap(sieve, p, verified_gap, true_gap, composite, &part);
        add_block_work(&total, part);
    }

    if (composite != local_composite) {
        free(composite);
    }
    return total;
}

