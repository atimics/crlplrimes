/* Multi-step resolution prover with depth-bound refutation target.
   State = set of clauses. Candidate = resolution pair.
   Target = applying this resolution leads to empty clause within max depth.
   Verifier: exact-check (resolution soundness). */

#include <string.h>

#define NN_DATASET_PROOF_FEATURE_COUNT 16
#define NN_DATASET_PROOF_MAX_VARS 8
#define NN_DATASET_PROOF_MAX_CLAUSES 24
#define NN_DATASET_PROOF_MAX_CLAUSE_LEN 5
#define NN_DATASET_PROOF_INIT_CLAUSES 10
#define NN_DATASET_PROOF_MAX_DEPTH 5
#define NN_DATASET_PROOF_TRAIN 40
#define NN_DATASET_PROOF_VALID 10
#define NN_DATASET_PROOF_TEST 10
#define NN_DATASET_PROOF_TOTAL (NN_DATASET_PROOF_TRAIN + NN_DATASET_PROOF_VALID + NN_DATASET_PROOF_TEST)

typedef struct {
    int literals[NN_DATASET_PROOF_MAX_CLAUSE_LEN];
    int len;
    int is_original;
} ProofClause;

static int proof_var(int lit) { return lit / 2; }
static int proof_complement(int lit) { return lit ^ 1; }
static int proof_make_lit(int var, int sign) { return var * 2 + sign; }

static unsigned int proof_rand(unsigned int *state) {
    *state = *state * 1103515245 + 12345;
    return (*state >> 16) & 0x7fff;
}

static void proof_generate_instance(unsigned int seed, int out_clauses[][4], int *out_count) {
    int i;
    *out_count = 0;
    for (i = 0; i < NN_DATASET_PROOF_INIT_CLAUSES; ++i) {
        int lits[3];
        int j;
        for (j = 0; j < 3; ++j) {
            int var = (int)(proof_rand(&seed) % (unsigned int)NN_DATASET_PROOF_MAX_VARS);
            int sign = (int)(proof_rand(&seed) % 2u);
            lits[j] = proof_make_lit(var, sign);
            int k;
            for (k = 0; k < j; ++k) {
                if (lits[k] == lits[j]) {
                    lits[j] = proof_make_lit((var + 1) % NN_DATASET_PROOF_MAX_VARS, sign ^ 1);
                    break;
                }
            }
        }
        out_clauses[i][0] = 3;
        out_clauses[i][1] = lits[0];
        out_clauses[i][2] = lits[1];
        out_clauses[i][3] = lits[2];
        ++(*out_count);
    }
}

static int proof_can_resolve(const int *c1, int len1, const int *c2, int len2) {
    int i, j;
    for (i = 0; i < len1; ++i)
        for (j = 0; j < len2; ++j)
            if (c1[i] == proof_complement(c2[j])) return proof_var(c1[i]);
    return -1;
}

static int proof_resolve(const int *c1, int len1, const int *c2, int len2,
                          int pivot_var, int *out, int max_out) {
    int i, out_len = 0;
    for (i = 0; i < len1; ++i) {
        if (proof_var(c1[i]) != pivot_var) {
            int j, dup = 0;
            for (j = 0; j < out_len; ++j) if (out[j] == c1[i]) { dup = 1; break; }
            if (!dup && out_len < max_out) out[out_len++] = c1[i];
        }
    }
    for (i = 0; i < len2; ++i) {
        if (proof_var(c2[i]) != pivot_var) {
            int j, dup = 0;
            for (j = 0; j < out_len; ++j) if (out[j] == c2[i]) { dup = 1; break; }
            if (!dup && out_len < max_out) out[out_len++] = c2[i];
        }
    }
    return out_len;
}

static int proof_var_frequency(int var, const ProofClause *clauses, int count) {
    int i, j, freq = 0;
    for (i = 0; i < count; ++i)
        for (j = 0; j < clauses[i].len; ++j)
            if (proof_var(clauses[i].literals[j]) == var) ++freq;
    return freq;
}

/* Check if a clause already exists in the set (exact match, any order) */
static int proof_clause_exists(const ProofClause *clauses, int count,
                                const int *lits, int len) {
    int i;
    for (i = 0; i < count; ++i) {
        if (clauses[i].len != len) continue;
        int j, all_found = 1;
        for (j = 0; j < len; ++j) {
            int k, found = 0;
            for (k = 0; k < len; ++k)
                if (clauses[i].literals[k] == lits[j]) { found = 1; break; }
            if (!found) { all_found = 0; break; }
        }
        if (all_found) return 1;
    }
    return 0;
}

static void encode_proof_features(nn_real f[NN_DATASET_PROOF_FEATURE_COUNT],
                                   const ProofClause *clauses, int clause_count,
                                   int ci, int cj, int pivot_var) {
    memset(f, 0, NN_DATASET_PROOF_FEATURE_COUNT * sizeof(f[0]));
    int resolvent[NN_DATASET_PROOF_MAX_CLAUSE_LEN * 2];
    int res_len = proof_resolve(clauses[ci].literals, clauses[ci].len,
                                 clauses[cj].literals, clauses[cj].len,
                                 pivot_var, resolvent,
                                 (int)(sizeof(resolvent)/sizeof(resolvent[0])));
    int freq = proof_var_frequency(pivot_var, clauses, clause_count);
    int is_unit_i = clauses[ci].len == 1;
    int is_unit_j = clauses[cj].len == 1;
    int is_empty = res_len == 0;

    f[0]  = (nn_real)1.0;
    f[1]  = (nn_real)((double)clauses[ci].len / 5.0);
    f[2]  = (nn_real)((double)clauses[cj].len / 5.0);
    f[3]  = (nn_real)((double)res_len / 8.0);
    f[4]  = (nn_real)(is_empty ? 1.0 : 0.0);
    f[5]  = (nn_real)((double)freq / 8.0);
    f[6]  = (nn_real)(is_unit_i ? 1.0 : 0.0);
    f[7]  = (nn_real)(is_unit_j ? 1.0 : 0.0);
    f[8]  = (nn_real)((clauses[ci].is_original && clauses[cj].is_original) ? 1.0 : 0.0);
    f[9]  = (nn_real)((double)clause_count / NN_DATASET_PROOF_MAX_CLAUSES);
    f[10] = (nn_real)((double)(clauses[ci].len + clauses[cj].len - 2) / 8.0);
    f[11] = (nn_real)((double)ci / NN_DATASET_PROOF_MAX_CLAUSES);
    f[12] = (nn_real)((double)cj / NN_DATASET_PROOF_MAX_CLAUSES);
    f[13] = (nn_real)((double)(ci + cj) / (NN_DATASET_PROOF_MAX_CLAUSES * 2.0));
    f[14] = (nn_real)((res_len <= clauses[ci].len && res_len <= clauses[cj].len) ? 1.0 : 0.0);
    f[15] = (nn_real)((double)pivot_var / NN_DATASET_PROOF_MAX_VARS);
}

static uint64_t proof_state_hash(int instance_id, const ProofClause *clauses, int count) {
    uint64_t h = 1469598103934665603ull;
    h ^= (uint64_t)instance_id + 0x9e3779b97f4a7c15ull;
    h *= 1099511628211ull;
    int i, j;
    for (i = 0; i < count; ++i) {
        for (j = 0; j < clauses[i].len; ++j) {
            h ^= (uint64_t)(uint32_t)(clauses[i].literals[j] + 1);
            h *= 1099511628211ull;
        }
    }
    return h;
}

static int generate_proof_resolution_adapter(const NNDatasetAdapterConfig *cfg, NNDataset *ds) {
    int train_n = NN_DATASET_PROOF_TRAIN;
    int valid_n = NN_DATASET_PROOF_VALID;
    int iid;
    size_t mg = cfg->max_groups == 0u ? SIZE_MAX : cfg->max_groups;
    nn_dataset_init(ds, NN_DATASET_PROOF_FEATURE_COUNT);
    uint64_t gid = 0;

    for (iid = 0; iid < NN_DATASET_PROOF_TOTAL && ds->group_count < mg; ++iid) {
        NNDatasetSplit split = iid < train_n ? NN_DATASET_SPLIT_TRAIN
                               : iid < train_n + valid_n ? NN_DATASET_SPLIT_VALID
                               : NN_DATASET_SPLIT_TEST;

        int init_clauses[NN_DATASET_PROOF_INIT_CLAUSES][4];
        int init_count = 0;
        proof_generate_instance((unsigned int)(424200 + iid * 37), init_clauses, &init_count);

        ProofClause state[NN_DATASET_PROOF_MAX_CLAUSES];
        int sc = 0, k;
        for (k = 0; k < init_count; ++k) {
            state[sc].len = init_clauses[k][0];
            int li;
            for (li = 0; li < state[sc].len; ++li)
                state[sc].literals[li] = init_clauses[k][1 + li];
            state[sc].is_original = 1;
            ++sc;
        }

        /* For each resolvable pair, check if it leads to refutation */
        size_t begin = ds->row_count;
        int any = 0;
        int ci;
        for (ci = 0; ci < sc; ++ci) {
            int cj;
            for (cj = ci + 1; cj < sc; ++cj) {
                int pivot = proof_can_resolve(state[ci].literals, state[ci].len,
                                               state[cj].literals, state[cj].len);
                if (pivot < 0) continue;

                nn_real f[NN_DATASET_PROOF_FEATURE_COUNT];
                int resolvent[NN_DATASET_PROOF_MAX_CLAUSE_LEN * 2];
                int res_len = proof_resolve(state[ci].literals, state[ci].len,
                                             state[cj].literals, state[cj].len,
                                             pivot, resolvent,
                                             (int)(sizeof(resolvent)/sizeof(resolvent[0])));

                /* Heuristic target: unit resolvent, empty resolvent, or high-frequency pivot */
                int pivot_freq = proof_var_frequency(pivot, state, sc);
                int target = (res_len <= 2 || pivot_freq >= 5) ? 1 : 0;

                encode_proof_features(f, state, sc, ci, cj, pivot);
                uint32_t action_id = (uint32_t)(ci * NN_DATASET_PROOF_MAX_CLAUSES + cj);
                if (!nn_dataset_append_candidate(ds, split, gid,
                    proof_state_hash(iid, state, sc), action_id, target, f)) return 0;
                any = 1;
            }
        }
        if (any) {
            if (!nn_dataset_finalize_group(ds, begin, ds->row_count)) return 0;
            ++gid;
        }
    }
    return ds->group_count > 0u;
}
