#include "nn.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    PS_GRAMMAR_VERSION = 1,
    PS_MAX_LIST = 8,
    PS_EXAMPLES_PER_TASK = 3,
    PS_RULE_FEATURE_COUNT = 18,
    PS_CLOSED_LOOP_EXTRA_FEATURES = 8,
    PS_CLOSED_LOOP_FEATURE_COUNT = PS_RULE_FEATURE_COUNT + PS_CLOSED_LOOP_EXTRA_FEATURES,
    PS_MAX_ACCEPTED = 64,
    PS_STATIC_EPOCHS = 240,
    PS_CLOSED_LOOP_EPOCHS = 160,
};

typedef enum {
    PS_TYPE_LIST = 0,
    PS_TYPE_INT = 1,
    PS_TYPE_BOOL = 2,
} PSValueType;

typedef enum {
    PS_SPLIT_TRAIN = 0,
    PS_SPLIT_VALIDATION = 1,
    PS_SPLIT_TEST = 2,
} PSSplit;

typedef enum {
    PS_PROP_RANDOM = 0,
    PS_PROP_GRAMMAR = 1,
    PS_PROP_STRUCTURAL = 2,
    PS_PROP_VALIDATION_ORACLE = 3,
    PS_PROP_NEURAL = 4,
    PS_PROP_CLOSED_LOOP = 5,
    PS_PROP_ALL = 6,
} PSProposerMode;

typedef enum {
    PS_OP_IDENTITY = 0,
    PS_OP_TAIL,
    PS_OP_REVERSE,
    PS_OP_SORT_ASC,
    PS_OP_MAP_INC,
    PS_OP_MAP_DEC,
    PS_OP_MAP_DOUBLE,
    PS_OP_MAP_SQUARE,
    PS_OP_FILTER_POSITIVE,
    PS_OP_FILTER_NEGATIVE,
    PS_OP_FILTER_EVEN,
    PS_OP_FILTER_ODD,
    PS_OP_SUM,
    PS_OP_PRODUCT,
    PS_OP_LENGTH,
    PS_OP_COUNT_POSITIVE,
    PS_OP_COUNT_NEGATIVE,
    PS_OP_COUNT_EVEN,
    PS_OP_ANY_POSITIVE,
    PS_OP_ALL_POSITIVE,
    PS_OP_MAX,
    PS_OP_MIN,
    PS_OP_HEAD,
    PS_OP_LAST,
    PS_OP_SUM_POSITIVE,
    PS_OP_SUM_NEGATIVE,
    PS_OP_SUM_SQUARES,
    PS_OP_DOUBLE_SUM,
    PS_OP_MAP_ABS,
    PS_OP_SUM_ABS,
    PS_OP_FIRST_POSITIVE,
    PS_OP_ANY_EVEN,
    PS_OP_COUNT,
} PSOp;

typedef struct {
    int values[PS_MAX_LIST];
    size_t length;
} PSList;

typedef struct {
    PSValueType type;
    PSList list;
    int integer;
} PSValue;

typedef struct {
    PSList input;
    PSValue expected;
} PSExample;

typedef struct {
    const char *name;
    const char *family;
    PSSplit split;
    PSOp target_op;
    uint64_t useful_mask;
    size_t baseline_nodes;
    PSExample examples[PS_EXAMPLES_PER_TASK];
} PSTask;

typedef struct {
    PSOp op;
    const char *name;
    PSValueType output_type;
    size_t depth;
    size_t nodes;
    size_t primitive_count;
    int uses_map;
    int uses_filter;
    int uses_fold;
    int uses_compare;
    int uses_arithmetic;
    int uses_ordering;
} PSAbstractionSpec;

typedef struct {
    const PSAbstractionSpec *specs;
    size_t spec_count;
    PSTask *tasks;
    size_t task_count;
    size_t train_count;
    size_t validation_count;
    size_t test_count;
} PSContext;

typedef struct {
    const NNRule *rules;
    size_t rule_count;
    const NNRuleCoverage *train_coverage;
    const NNRuleCoverage *validation_coverage;
    const NNRuleCoverage *test_coverage;
    const uint8_t *certified;
    size_t max_rules;
    unsigned int seed;
    size_t closed_loop_hidden1;
    size_t closed_loop_hidden2;
    size_t closed_loop_epochs;
} PSRunConfig;

typedef struct {
    size_t accepted_indices[PS_MAX_ACCEPTED];
    size_t accepted_count;
    size_t proposed_rules;
    size_t verified_rules;
    size_t proposer_calls;
    size_t softmax_group_size;
    size_t residual_train_samples;
    size_t parameters;
    size_t forward_flops;
    double proposer_loss;
    size_t train_covered;
    size_t validation_covered;
    size_t test_covered;
} PSRunMetrics;

static const NNFeatureSchema PS_RULE_FEATURES[PS_RULE_FEATURE_COUNT] = {
    {"program_synthesis_domain", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"output_is_list", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"output_is_int", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"output_is_bool", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"term_depth", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"node_count", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"primitive_count", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"uses_map", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"uses_filter", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"uses_fold", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"uses_compare", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"uses_arithmetic", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"uses_ordering", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"arity", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"op_index", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"is_composed_abstraction", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"is_predicate_abstraction", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"lexical_rank", NN_FEATURE_STRUCTURAL_PRIOR, 1},
};

static const NNFeatureSchema PS_CLOSED_LOOP_FEATURES[PS_CLOSED_LOOP_FEATURE_COUNT] = {
    {"program_synthesis_domain", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"output_is_list", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"output_is_int", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"output_is_bool", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"term_depth", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"node_count", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"primitive_count", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"uses_map", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"uses_filter", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"uses_fold", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"uses_compare", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"uses_arithmetic", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"uses_ordering", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"arity", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"op_index", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"is_composed_abstraction", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"is_predicate_abstraction", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"lexical_rank", NN_FEATURE_STRUCTURAL_PRIOR, 1},
    {"accepted_fraction", NN_FEATURE_DERIVED_FROM_GATE, 1},
    {"residual_train_fraction", NN_FEATURE_DERIVED_FROM_GATE, 1},
    {"covered_train_fraction", NN_FEATURE_DERIVED_FROM_GATE, 1},
    {"same_output_type_fraction", NN_FEATURE_DERIVED_FROM_GATE, 1},
    {"same_map_fraction", NN_FEATURE_DERIVED_FROM_GATE, 1},
    {"same_filter_fraction", NN_FEATURE_DERIVED_FROM_GATE, 1},
    {"same_fold_fraction", NN_FEATURE_DERIVED_FROM_GATE, 1},
    {"already_covered_by_accepted", NN_FEATURE_DERIVED_FROM_GATE, 1},
};

static const PSAbstractionSpec PS_SPECS[] = {
    {PS_OP_IDENTITY, "identity_list", PS_TYPE_LIST, 1, 1, 1, 0, 0, 0, 0, 0, 0},
    {PS_OP_TAIL, "tail_list", PS_TYPE_LIST, 2, 2, 1, 0, 0, 0, 0, 0, 0},
    {PS_OP_REVERSE, "reverse_list", PS_TYPE_LIST, 3, 4, 2, 0, 0, 0, 0, 0, 1},
    {PS_OP_SORT_ASC, "sort_asc", PS_TYPE_LIST, 4, 6, 3, 0, 0, 0, 1, 0, 1},
    {PS_OP_MAP_INC, "map_inc", PS_TYPE_LIST, 3, 5, 2, 1, 0, 0, 0, 1, 0},
    {PS_OP_MAP_DEC, "map_dec", PS_TYPE_LIST, 3, 5, 2, 1, 0, 0, 0, 1, 0},
    {PS_OP_MAP_DOUBLE, "map_double", PS_TYPE_LIST, 3, 5, 2, 1, 0, 0, 0, 1, 0},
    {PS_OP_MAP_SQUARE, "map_square", PS_TYPE_LIST, 3, 6, 2, 1, 0, 0, 0, 1, 0},
    {PS_OP_FILTER_POSITIVE, "filter_positive", PS_TYPE_LIST, 3, 5, 2, 0, 1, 0, 1, 0, 0},
    {PS_OP_FILTER_NEGATIVE, "filter_negative", PS_TYPE_LIST, 3, 5, 2, 0, 1, 0, 1, 0, 0},
    {PS_OP_FILTER_EVEN, "filter_even", PS_TYPE_LIST, 3, 6, 2, 0, 1, 0, 1, 1, 0},
    {PS_OP_FILTER_ODD, "filter_odd", PS_TYPE_LIST, 3, 6, 2, 0, 1, 0, 1, 1, 0},
    {PS_OP_SUM, "sum_list", PS_TYPE_INT, 3, 4, 1, 0, 0, 1, 0, 1, 0},
    {PS_OP_PRODUCT, "product_list", PS_TYPE_INT, 3, 4, 1, 0, 0, 1, 0, 1, 0},
    {PS_OP_LENGTH, "length_list", PS_TYPE_INT, 2, 2, 1, 0, 0, 0, 0, 0, 0},
    {PS_OP_COUNT_POSITIVE, "count_positive", PS_TYPE_INT, 4, 7, 3, 0, 1, 1, 1, 1, 0},
    {PS_OP_COUNT_NEGATIVE, "count_negative", PS_TYPE_INT, 4, 7, 3, 0, 1, 1, 1, 1, 0},
    {PS_OP_COUNT_EVEN, "count_even", PS_TYPE_INT, 4, 8, 3, 0, 1, 1, 1, 1, 0},
    {PS_OP_ANY_POSITIVE, "any_positive", PS_TYPE_BOOL, 4, 7, 3, 0, 1, 1, 1, 1, 0},
    {PS_OP_ALL_POSITIVE, "all_positive", PS_TYPE_BOOL, 4, 7, 3, 0, 1, 1, 1, 1, 0},
    {PS_OP_MAX, "max_list", PS_TYPE_INT, 3, 4, 1, 0, 0, 1, 1, 0, 1},
    {PS_OP_MIN, "min_list", PS_TYPE_INT, 3, 4, 1, 0, 0, 1, 1, 0, 1},
    {PS_OP_HEAD, "head_or_zero", PS_TYPE_INT, 2, 3, 1, 0, 0, 0, 0, 0, 0},
    {PS_OP_LAST, "last_or_zero", PS_TYPE_INT, 3, 4, 1, 0, 0, 0, 0, 0, 1},
    {PS_OP_SUM_POSITIVE, "sum_positive", PS_TYPE_INT, 5, 9, 4, 0, 1, 1, 1, 1, 0},
    {PS_OP_SUM_NEGATIVE, "sum_negative", PS_TYPE_INT, 5, 9, 4, 0, 1, 1, 1, 1, 0},
    {PS_OP_SUM_SQUARES, "sum_squares", PS_TYPE_INT, 5, 10, 4, 1, 0, 1, 0, 1, 0},
    {PS_OP_DOUBLE_SUM, "double_sum", PS_TYPE_INT, 5, 9, 4, 1, 0, 1, 0, 1, 0},
    {PS_OP_MAP_ABS, "map_abs", PS_TYPE_LIST, 3, 5, 2, 1, 0, 0, 1, 1, 0},
    {PS_OP_SUM_ABS, "sum_abs", PS_TYPE_INT, 5, 9, 4, 1, 0, 1, 1, 1, 0},
    {PS_OP_FIRST_POSITIVE, "first_positive_or_zero", PS_TYPE_INT, 4, 7, 3, 0, 1, 0, 1, 0, 0},
    {PS_OP_ANY_EVEN, "any_even", PS_TYPE_BOOL, 4, 8, 3, 0, 1, 1, 1, 1, 0},
};

static const NNHypothesisTemplate *ps_hypothesis_template_filter_then_reduce(void)
{
    static NNHypothesisTemplate template_ref;
    static int initialized = 0;
    if (initialized) {
        return &template_ref;
    }
    nn_hypothesis_template_init(&template_ref,
                                "program_synthesis",
                                "filter_then_reduce",
                                2048,
                                NN_PROGRAM_VALUE_SEQUENCE,
                                NN_PROGRAM_VALUE_CUSTOM);
    nn_hypothesis_template_add_stage(&template_ref,
                                     "enumerate",
                                     NN_PROGRAM_KIND_TRANSFORM,
                                     NN_PROGRAM_VALUE_SEQUENCE,
                                     NN_PROGRAM_VALUE_SEQUENCE,
                                     210);
    nn_hypothesis_template_add_stage(&template_ref,
                                     "filter",
                                     NN_PROGRAM_KIND_PREDICATE,
                                     NN_PROGRAM_VALUE_SEQUENCE,
                                     NN_PROGRAM_VALUE_SEQUENCE,
                                     211);
    nn_hypothesis_template_add_stage(&template_ref,
                                     "reduce",
                                     NN_PROGRAM_KIND_TRANSFORM,
                                     NN_PROGRAM_VALUE_SEQUENCE,
                                     NN_PROGRAM_VALUE_CUSTOM,
                                     212);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "filter.predicate_id",
                                    NN_HYPOTHESIS_SLOT_INTEGER,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "reduce.kind",
                                    NN_HYPOTHESIS_SLOT_INTEGER,
                                    1, 0.0, 0);
    nn_hypothesis_template_add_slot(&template_ref,
                                    "output.type",
                                    NN_HYPOTHESIS_SLOT_INTEGER,
                                    1, 0.0, 0);
    initialized = 1;
    return &template_ref;
}

static int ps_filter_reduce_slots_for_op(PSOp op,
                                         int64_t *predicate_out,
                                         int64_t *reduce_out)
{
    int64_t predicate = 0;
    int64_t reduce = 0;
    switch (op) {
    case PS_OP_COUNT_POSITIVE:
        predicate = 1;
        reduce = 1;
        break;
    case PS_OP_COUNT_NEGATIVE:
        predicate = 2;
        reduce = 1;
        break;
    case PS_OP_COUNT_EVEN:
        predicate = 3;
        reduce = 1;
        break;
    case PS_OP_SUM_POSITIVE:
        predicate = 1;
        reduce = 2;
        break;
    case PS_OP_SUM_NEGATIVE:
        predicate = 2;
        reduce = 2;
        break;
    case PS_OP_ANY_POSITIVE:
        predicate = 1;
        reduce = 3;
        break;
    case PS_OP_ALL_POSITIVE:
        predicate = 1;
        reduce = 4;
        break;
    case PS_OP_FIRST_POSITIVE:
        predicate = 1;
        reduce = 5;
        break;
    case PS_OP_ANY_EVEN:
        predicate = 3;
        reduce = 3;
        break;
    default:
        return 0;
    }
    if (predicate_out != NULL) {
        *predicate_out = predicate;
    }
    if (reduce_out != NULL) {
        *reduce_out = reduce;
    }
    return 1;
}

static int ps_bind_filter_reduce_hypothesis(const PSAbstractionSpec *spec,
                                            NNHypothesis *hypothesis)
{
    int64_t predicate = 0;
    int64_t reduce = 0;
    if (spec == NULL || hypothesis == NULL ||
        !ps_filter_reduce_slots_for_op(spec->op, &predicate, &reduce) ||
        !nn_hypothesis_init(hypothesis, ps_hypothesis_template_filter_then_reduce())) {
        return 0;
    }
    return nn_hypothesis_bind(hypothesis,
                              "filter.predicate_id",
                              (double)predicate,
                              predicate) &&
           nn_hypothesis_bind(hypothesis,
                              "reduce.kind",
                              (double)reduce,
                              reduce) &&
           nn_hypothesis_bind(hypothesis,
                              "output.type",
                              (double)spec->output_type,
                              spec->output_type);
}

static PSList ps_list(size_t length,
                      int a,
                      int b,
                      int c,
                      int d,
                      int e,
                      int f,
                      int g,
                      int h)
{
    PSList list;
    int values[PS_MAX_LIST] = {a, b, c, d, e, f, g, h};
    list.length = length > PS_MAX_LIST ? PS_MAX_LIST : length;
    for (size_t i = 0; i < list.length; ++i) {
        list.values[i] = values[i];
    }
    return list;
}

static PSValue ps_value_int(int value)
{
    PSValue out;
    memset(&out, 0, sizeof(out));
    out.type = PS_TYPE_INT;
    out.integer = value;
    return out;
}

static PSValue ps_value_bool(int value)
{
    PSValue out;
    memset(&out, 0, sizeof(out));
    out.type = PS_TYPE_BOOL;
    out.integer = value != 0;
    return out;
}

static PSValue ps_value_list(const PSList *list)
{
    PSValue out;
    memset(&out, 0, sizeof(out));
    out.type = PS_TYPE_LIST;
    if (list != NULL) {
        out.list = *list;
    }
    return out;
}

static uint64_t ps_mask1(PSOp op)
{
    return UINT64_C(1) << (unsigned int)op;
}

static uint64_t ps_mask2(PSOp a, PSOp b)
{
    return ps_mask1(a) | ps_mask1(b);
}

static uint64_t ps_mask3(PSOp a, PSOp b, PSOp c)
{
    return ps_mask1(a) | ps_mask1(b) | ps_mask1(c);
}

static int ps_abs_i(int value)
{
    return value < 0 ? -value : value;
}

static int ps_is_even(int value)
{
    return value % 2 == 0;
}

static PSList ps_map_list(const PSList *input, PSOp op)
{
    PSList out;
    out.length = input->length;
    for (size_t i = 0; i < input->length; ++i) {
        int value = input->values[i];
        switch (op) {
        case PS_OP_MAP_INC:
            value += 1;
            break;
        case PS_OP_MAP_DEC:
            value -= 1;
            break;
        case PS_OP_MAP_DOUBLE:
            value *= 2;
            break;
        case PS_OP_MAP_SQUARE:
            value *= value;
            break;
        case PS_OP_MAP_ABS:
            value = ps_abs_i(value);
            break;
        default:
            break;
        }
        out.values[i] = value;
    }
    return out;
}

static PSList ps_filter_list(const PSList *input, PSOp op)
{
    PSList out;
    out.length = 0;
    for (size_t i = 0; i < input->length; ++i) {
        int value = input->values[i];
        int keep = 0;
        switch (op) {
        case PS_OP_FILTER_POSITIVE:
            keep = value > 0;
            break;
        case PS_OP_FILTER_NEGATIVE:
            keep = value < 0;
            break;
        case PS_OP_FILTER_EVEN:
            keep = ps_is_even(value);
            break;
        case PS_OP_FILTER_ODD:
            keep = !ps_is_even(value);
            break;
        default:
            break;
        }
        if (keep && out.length < PS_MAX_LIST) {
            out.values[out.length++] = value;
        }
    }
    return out;
}

static PSList ps_reverse_list(const PSList *input)
{
    PSList out;
    out.length = input->length;
    for (size_t i = 0; i < input->length; ++i) {
        out.values[i] = input->values[input->length - i - 1u];
    }
    return out;
}

static PSList ps_tail_list(const PSList *input)
{
    PSList out;
    out.length = input->length == 0 ? 0 : input->length - 1u;
    for (size_t i = 0; i < out.length; ++i) {
        out.values[i] = input->values[i + 1u];
    }
    return out;
}

static PSList ps_sort_asc_list(const PSList *input)
{
    PSList out = *input;
    for (size_t i = 1; i < out.length; ++i) {
        int value = out.values[i];
        size_t j = i;
        while (j > 0 && out.values[j - 1u] > value) {
            out.values[j] = out.values[j - 1u];
            --j;
        }
        out.values[j] = value;
    }
    return out;
}

static int ps_sum_list(const PSList *input)
{
    int sum = 0;
    for (size_t i = 0; i < input->length; ++i) {
        sum += input->values[i];
    }
    return sum;
}

static int ps_product_list(const PSList *input)
{
    int product = 1;
    for (size_t i = 0; i < input->length; ++i) {
        product *= input->values[i];
    }
    return product;
}

static int ps_count_if(const PSList *input, PSOp op)
{
    int count = 0;
    for (size_t i = 0; i < input->length; ++i) {
        int value = input->values[i];
        if ((op == PS_OP_COUNT_POSITIVE && value > 0) ||
            (op == PS_OP_COUNT_NEGATIVE && value < 0) ||
            (op == PS_OP_COUNT_EVEN && ps_is_even(value))) {
            ++count;
        }
    }
    return count;
}

static int ps_first_matching(const PSList *input, PSOp op)
{
    for (size_t i = 0; i < input->length; ++i) {
        int value = input->values[i];
        if (op == PS_OP_FIRST_POSITIVE && value > 0) {
            return value;
        }
    }
    return 0;
}

static PSValue ps_eval_op(PSOp op, const PSList *input)
{
    switch (op) {
    case PS_OP_IDENTITY:
        return ps_value_list(input);
    case PS_OP_TAIL: {
        PSList list = ps_tail_list(input);
        return ps_value_list(&list);
    }
    case PS_OP_REVERSE: {
        PSList list = ps_reverse_list(input);
        return ps_value_list(&list);
    }
    case PS_OP_SORT_ASC: {
        PSList list = ps_sort_asc_list(input);
        return ps_value_list(&list);
    }
    case PS_OP_MAP_INC:
    case PS_OP_MAP_DEC:
    case PS_OP_MAP_DOUBLE:
    case PS_OP_MAP_SQUARE:
    case PS_OP_MAP_ABS: {
        PSList list = ps_map_list(input, op);
        return ps_value_list(&list);
    }
    case PS_OP_FILTER_POSITIVE:
    case PS_OP_FILTER_NEGATIVE:
    case PS_OP_FILTER_EVEN:
    case PS_OP_FILTER_ODD: {
        PSList list = ps_filter_list(input, op);
        return ps_value_list(&list);
    }
    case PS_OP_SUM:
        return ps_value_int(ps_sum_list(input));
    case PS_OP_PRODUCT:
        return ps_value_int(ps_product_list(input));
    case PS_OP_LENGTH:
        return ps_value_int((int)input->length);
    case PS_OP_COUNT_POSITIVE:
    case PS_OP_COUNT_NEGATIVE:
    case PS_OP_COUNT_EVEN:
        return ps_value_int(ps_count_if(input, op));
    case PS_OP_ANY_POSITIVE:
        return ps_value_bool(ps_count_if(input, PS_OP_COUNT_POSITIVE) > 0);
    case PS_OP_ALL_POSITIVE:
        return ps_value_bool(ps_count_if(input, PS_OP_COUNT_POSITIVE) == (int)input->length);
    case PS_OP_MAX: {
        int best = input->length == 0 ? 0 : input->values[0];
        for (size_t i = 1; i < input->length; ++i) {
            if (input->values[i] > best) {
                best = input->values[i];
            }
        }
        return ps_value_int(best);
    }
    case PS_OP_MIN: {
        int best = input->length == 0 ? 0 : input->values[0];
        for (size_t i = 1; i < input->length; ++i) {
            if (input->values[i] < best) {
                best = input->values[i];
            }
        }
        return ps_value_int(best);
    }
    case PS_OP_HEAD:
        return ps_value_int(input->length == 0 ? 0 : input->values[0]);
    case PS_OP_LAST:
        return ps_value_int(input->length == 0 ? 0 : input->values[input->length - 1u]);
    case PS_OP_SUM_POSITIVE: {
        PSList list = ps_filter_list(input, PS_OP_FILTER_POSITIVE);
        return ps_value_int(ps_sum_list(&list));
    }
    case PS_OP_SUM_NEGATIVE: {
        PSList list = ps_filter_list(input, PS_OP_FILTER_NEGATIVE);
        return ps_value_int(ps_sum_list(&list));
    }
    case PS_OP_SUM_SQUARES: {
        PSList list = ps_map_list(input, PS_OP_MAP_SQUARE);
        return ps_value_int(ps_sum_list(&list));
    }
    case PS_OP_DOUBLE_SUM: {
        PSList list = ps_map_list(input, PS_OP_MAP_DOUBLE);
        return ps_value_int(ps_sum_list(&list));
    }
    case PS_OP_SUM_ABS: {
        PSList list = ps_map_list(input, PS_OP_MAP_ABS);
        return ps_value_int(ps_sum_list(&list));
    }
    case PS_OP_FIRST_POSITIVE:
        return ps_value_int(ps_first_matching(input, PS_OP_FIRST_POSITIVE));
    case PS_OP_ANY_EVEN:
        return ps_value_bool(ps_count_if(input, PS_OP_COUNT_EVEN) > 0);
    default:
        return ps_value_int(0);
    }
}

static int ps_value_equal(const PSValue *a, const PSValue *b)
{
    if (a->type != b->type) {
        return 0;
    }
    if (a->type == PS_TYPE_LIST) {
        if (a->list.length != b->list.length) {
            return 0;
        }
        for (size_t i = 0; i < a->list.length; ++i) {
            if (a->list.values[i] != b->list.values[i]) {
                return 0;
            }
        }
        return 1;
    }
    return a->integer == b->integer;
}

static PSTask ps_make_task(const char *name,
                           const char *family,
                           PSSplit split,
                           PSOp target_op,
                           uint64_t useful_mask,
                           size_t baseline_nodes,
                           PSList a,
                           PSList b,
                           PSList c)
{
    PSTask task;
    memset(&task, 0, sizeof(task));
    task.name = name;
    task.family = family;
    task.split = split;
    task.target_op = target_op;
    task.useful_mask = useful_mask | ps_mask1(target_op);
    task.baseline_nodes = baseline_nodes;
    task.examples[0].input = a;
    task.examples[1].input = b;
    task.examples[2].input = c;
    for (size_t i = 0; i < PS_EXAMPLES_PER_TASK; ++i) {
        task.examples[i].expected = ps_eval_op(target_op, &task.examples[i].input);
    }
    return task;
}

static int ps_init_tasks(PSTask **tasks_out, size_t *count_out)
{
    enum { TASK_COUNT = 28 };
    PSTask *tasks = calloc(TASK_COUNT, sizeof(*tasks));
    if (tasks == NULL) {
        return 0;
    }

    size_t count = 0;
    tasks[count++] = ps_make_task("sum_list",
                                  "fold_sum",
                                  PS_SPLIT_TRAIN,
                                  PS_OP_SUM,
                                  ps_mask1(PS_OP_SUM),
                                  8,
                                  ps_list(4, 1, 2, 3, 4, 0, 0, 0, 0),
                                  ps_list(3, -1, 5, 2, 0, 0, 0, 0, 0),
                                  ps_list(0, 0, 0, 0, 0, 0, 0, 0, 0));
    tasks[count++] = ps_make_task("length_list",
                                  "shape",
                                  PS_SPLIT_TRAIN,
                                  PS_OP_LENGTH,
                                  ps_mask1(PS_OP_LENGTH),
                                  5,
                                  ps_list(4, 9, 8, 7, 6, 0, 0, 0, 0),
                                  ps_list(1, 3, 0, 0, 0, 0, 0, 0, 0),
                                  ps_list(0, 0, 0, 0, 0, 0, 0, 0, 0));
    tasks[count++] = ps_make_task("positive_count",
                                  "filter_count",
                                  PS_SPLIT_TRAIN,
                                  PS_OP_COUNT_POSITIVE,
                                  ps_mask2(PS_OP_FILTER_POSITIVE, PS_OP_COUNT_POSITIVE),
                                  13,
                                  ps_list(5, -2, 0, 3, 4, -1, 0, 0, 0),
                                  ps_list(3, 1, 2, 3, 0, 0, 0, 0, 0),
                                  ps_list(3, -4, -3, 0, 0, 0, 0, 0, 0));
    tasks[count++] = ps_make_task("increment_all",
                                  "map_arithmetic",
                                  PS_SPLIT_TRAIN,
                                  PS_OP_MAP_INC,
                                  ps_mask1(PS_OP_MAP_INC),
                                  10,
                                  ps_list(4, 1, 0, -2, 5, 0, 0, 0, 0),
                                  ps_list(2, 9, -9, 0, 0, 0, 0, 0, 0),
                                  ps_list(0, 0, 0, 0, 0, 0, 0, 0, 0));
    tasks[count++] = ps_make_task("double_sum",
                                  "map_fold",
                                  PS_SPLIT_TRAIN,
                                  PS_OP_DOUBLE_SUM,
                                  ps_mask3(PS_OP_MAP_DOUBLE, PS_OP_SUM, PS_OP_DOUBLE_SUM),
                                  15,
                                  ps_list(3, 1, 2, 3, 0, 0, 0, 0, 0),
                                  ps_list(3, -2, 4, 5, 0, 0, 0, 0, 0),
                                  ps_list(1, 7, 0, 0, 0, 0, 0, 0, 0));
    tasks[count++] = ps_make_task("sum_positive",
                                  "filter_fold",
                                  PS_SPLIT_TRAIN,
                                  PS_OP_SUM_POSITIVE,
                                  ps_mask3(PS_OP_FILTER_POSITIVE, PS_OP_SUM, PS_OP_SUM_POSITIVE),
                                  16,
                                  ps_list(5, -4, 2, 0, 7, -1, 0, 0, 0),
                                  ps_list(3, -3, -2, -1, 0, 0, 0, 0, 0),
                                  ps_list(4, 1, 5, 2, -7, 0, 0, 0, 0));
    tasks[count++] = ps_make_task("max_list",
                                  "order_fold",
                                  PS_SPLIT_TRAIN,
                                  PS_OP_MAX,
                                  ps_mask1(PS_OP_MAX),
                                  9,
                                  ps_list(4, 1, 7, 3, 2, 0, 0, 0, 0),
                                  ps_list(3, -5, -2, -9, 0, 0, 0, 0, 0),
                                  ps_list(1, 4, 0, 0, 0, 0, 0, 0, 0));
    tasks[count++] = ps_make_task("reverse_list",
                                  "list_order",
                                  PS_SPLIT_TRAIN,
                                  PS_OP_REVERSE,
                                  ps_mask1(PS_OP_REVERSE),
                                  9,
                                  ps_list(4, 1, 2, 3, 4, 0, 0, 0, 0),
                                  ps_list(3, -1, 5, 2, 0, 0, 0, 0, 0),
                                  ps_list(1, 8, 0, 0, 0, 0, 0, 0, 0));
    tasks[count++] = ps_make_task("even_count",
                                  "filter_count",
                                  PS_SPLIT_TRAIN,
                                  PS_OP_COUNT_EVEN,
                                  ps_mask2(PS_OP_FILTER_EVEN, PS_OP_COUNT_EVEN),
                                  14,
                                  ps_list(5, 1, 2, 3, 4, 6, 0, 0, 0),
                                  ps_list(3, -2, -4, 5, 0, 0, 0, 0, 0),
                                  ps_list(2, 7, 9, 0, 0, 0, 0, 0, 0));
    tasks[count++] = ps_make_task("any_positive",
                                  "predicate",
                                  PS_SPLIT_TRAIN,
                                  PS_OP_ANY_POSITIVE,
                                  ps_mask2(PS_OP_FILTER_POSITIVE, PS_OP_ANY_POSITIVE),
                                  10,
                                  ps_list(3, -1, 0, 2, 0, 0, 0, 0, 0),
                                  ps_list(2, -3, -4, 0, 0, 0, 0, 0, 0),
                                  ps_list(1, 5, 0, 0, 0, 0, 0, 0, 0));
    tasks[count++] = ps_make_task("square_all",
                                  "map_arithmetic",
                                  PS_SPLIT_TRAIN,
                                  PS_OP_MAP_SQUARE,
                                  ps_mask1(PS_OP_MAP_SQUARE),
                                  12,
                                  ps_list(3, -2, 3, 4, 0, 0, 0, 0, 0),
                                  ps_list(2, 0, 5, 0, 0, 0, 0, 0, 0),
                                  ps_list(1, -7, 0, 0, 0, 0, 0, 0, 0));
    tasks[count++] = ps_make_task("tail_list",
                                  "list_shape",
                                  PS_SPLIT_TRAIN,
                                  PS_OP_TAIL,
                                  ps_mask1(PS_OP_TAIL),
                                  6,
                                  ps_list(4, 1, 2, 3, 4, 0, 0, 0, 0),
                                  ps_list(1, 9, 0, 0, 0, 0, 0, 0, 0),
                                  ps_list(0, 0, 0, 0, 0, 0, 0, 0, 0));

    tasks[count++] = ps_make_task("min_list",
                                  "order_fold",
                                  PS_SPLIT_VALIDATION,
                                  PS_OP_MIN,
                                  ps_mask1(PS_OP_MIN),
                                  9,
                                  ps_list(4, 5, 2, 9, -1, 0, 0, 0, 0),
                                  ps_list(2, 7, 3, 0, 0, 0, 0, 0, 0),
                                  ps_list(0, 0, 0, 0, 0, 0, 0, 0, 0));
    tasks[count++] = ps_make_task("sum_negative",
                                  "filter_fold",
                                  PS_SPLIT_VALIDATION,
                                  PS_OP_SUM_NEGATIVE,
                                  ps_mask3(PS_OP_FILTER_NEGATIVE, PS_OP_SUM, PS_OP_SUM_NEGATIVE),
                                  16,
                                  ps_list(4, -5, 2, -1, 7, 0, 0, 0, 0),
                                  ps_list(2, 3, 4, 0, 0, 0, 0, 0, 0),
                                  ps_list(3, -2, -3, -4, 0, 0, 0, 0, 0));
    tasks[count++] = ps_make_task("sort_ascending",
                                  "list_order",
                                  PS_SPLIT_VALIDATION,
                                  PS_OP_SORT_ASC,
                                  ps_mask1(PS_OP_SORT_ASC),
                                  15,
                                  ps_list(5, 3, 1, -2, 7, 0, 0, 0, 0),
                                  ps_list(3, 9, 8, 7, 0, 0, 0, 0, 0),
                                  ps_list(2, -1, -5, 0, 0, 0, 0, 0, 0));
    tasks[count++] = ps_make_task("last_item",
                                  "list_shape",
                                  PS_SPLIT_VALIDATION,
                                  PS_OP_LAST,
                                  ps_mask1(PS_OP_LAST),
                                  7,
                                  ps_list(4, 1, 2, 3, 4, 0, 0, 0, 0),
                                  ps_list(1, -9, 0, 0, 0, 0, 0, 0, 0),
                                  ps_list(0, 0, 0, 0, 0, 0, 0, 0, 0));
    tasks[count++] = ps_make_task("all_positive",
                                  "predicate",
                                  PS_SPLIT_VALIDATION,
                                  PS_OP_ALL_POSITIVE,
                                  ps_mask2(PS_OP_FILTER_POSITIVE, PS_OP_ALL_POSITIVE),
                                  11,
                                  ps_list(3, 1, 2, 3, 0, 0, 0, 0, 0),
                                  ps_list(3, 1, -2, 3, 0, 0, 0, 0, 0),
                                  ps_list(0, 0, 0, 0, 0, 0, 0, 0, 0));
    tasks[count++] = ps_make_task("odd_values",
                                  "filter",
                                  PS_SPLIT_VALIDATION,
                                  PS_OP_FILTER_ODD,
                                  ps_mask1(PS_OP_FILTER_ODD),
                                  10,
                                  ps_list(5, 1, 2, 3, 4, 5, 0, 0, 0),
                                  ps_list(3, -3, -2, 0, 0, 0, 0, 0, 0),
                                  ps_list(2, 8, 10, 0, 0, 0, 0, 0, 0));
    tasks[count++] = ps_make_task("sum_squares",
                                  "map_fold",
                                  PS_SPLIT_VALIDATION,
                                  PS_OP_SUM_SQUARES,
                                  ps_mask3(PS_OP_MAP_SQUARE, PS_OP_SUM, PS_OP_SUM_SQUARES),
                                  17,
                                  ps_list(3, 1, -2, 3, 0, 0, 0, 0, 0),
                                  ps_list(2, 4, 5, 0, 0, 0, 0, 0, 0),
                                  ps_list(1, -6, 0, 0, 0, 0, 0, 0, 0));
    tasks[count++] = ps_make_task("sum_abs",
                                  "map_fold",
                                  PS_SPLIT_VALIDATION,
                                  PS_OP_SUM_ABS,
                                  ps_mask3(PS_OP_MAP_ABS, PS_OP_SUM, PS_OP_SUM_ABS),
                                  16,
                                  ps_list(3, -1, 2, -3, 0, 0, 0, 0, 0),
                                  ps_list(2, -4, -5, 0, 0, 0, 0, 0, 0),
                                  ps_list(1, 7, 0, 0, 0, 0, 0, 0, 0));

    tasks[count++] = ps_make_task("product_list",
                                  "fold_product",
                                  PS_SPLIT_TEST,
                                  PS_OP_PRODUCT,
                                  ps_mask1(PS_OP_PRODUCT),
                                  9,
                                  ps_list(3, 2, 3, 4, 0, 0, 0, 0, 0),
                                  ps_list(2, -2, 5, 0, 0, 0, 0, 0, 0),
                                  ps_list(0, 0, 0, 0, 0, 0, 0, 0, 0));
    tasks[count++] = ps_make_task("head_item",
                                  "list_shape",
                                  PS_SPLIT_TEST,
                                  PS_OP_HEAD,
                                  ps_mask1(PS_OP_HEAD),
                                  6,
                                  ps_list(4, 9, 1, 2, 3, 0, 0, 0, 0),
                                  ps_list(1, -8, 0, 0, 0, 0, 0, 0, 0),
                                  ps_list(0, 0, 0, 0, 0, 0, 0, 0, 0));
    tasks[count++] = ps_make_task("negative_count",
                                  "filter_count",
                                  PS_SPLIT_TEST,
                                  PS_OP_COUNT_NEGATIVE,
                                  ps_mask2(PS_OP_FILTER_NEGATIVE, PS_OP_COUNT_NEGATIVE),
                                  13,
                                  ps_list(4, -1, 2, -3, 4, 0, 0, 0, 0),
                                  ps_list(3, 1, 2, 3, 0, 0, 0, 0, 0),
                                  ps_list(2, -4, -5, 0, 0, 0, 0, 0, 0));
    tasks[count++] = ps_make_task("negative_values",
                                  "filter",
                                  PS_SPLIT_TEST,
                                  PS_OP_FILTER_NEGATIVE,
                                  ps_mask1(PS_OP_FILTER_NEGATIVE),
                                  10,
                                  ps_list(5, -1, 0, 2, -3, 4, 0, 0, 0),
                                  ps_list(2, 5, 6, 0, 0, 0, 0, 0, 0),
                                  ps_list(3, -7, -8, 9, 0, 0, 0, 0, 0));
    tasks[count++] = ps_make_task("double_all",
                                  "map_arithmetic",
                                  PS_SPLIT_TEST,
                                  PS_OP_MAP_DOUBLE,
                                  ps_mask1(PS_OP_MAP_DOUBLE),
                                  10,
                                  ps_list(4, 1, -2, 3, 0, 0, 0, 0, 0),
                                  ps_list(2, 7, 8, 0, 0, 0, 0, 0, 0),
                                  ps_list(1, -4, 0, 0, 0, 0, 0, 0, 0));
    tasks[count++] = ps_make_task("absolute_all",
                                  "map_arithmetic",
                                  PS_SPLIT_TEST,
                                  PS_OP_MAP_ABS,
                                  ps_mask1(PS_OP_MAP_ABS),
                                  10,
                                  ps_list(4, -1, 2, -3, 0, 0, 0, 0, 0),
                                  ps_list(2, -7, -8, 0, 0, 0, 0, 0, 0),
                                  ps_list(1, 5, 0, 0, 0, 0, 0, 0, 0));
    tasks[count++] = ps_make_task("first_positive",
                                  "filter",
                                  PS_SPLIT_TEST,
                                  PS_OP_FIRST_POSITIVE,
                                  ps_mask2(PS_OP_FILTER_POSITIVE, PS_OP_FIRST_POSITIVE),
                                  12,
                                  ps_list(4, -1, 0, 5, 6, 0, 0, 0, 0),
                                  ps_list(2, -3, -4, 0, 0, 0, 0, 0, 0),
                                  ps_list(1, 9, 0, 0, 0, 0, 0, 0, 0));
    tasks[count++] = ps_make_task("even_values",
                                  "filter",
                                  PS_SPLIT_TEST,
                                  PS_OP_FILTER_EVEN,
                                  ps_mask1(PS_OP_FILTER_EVEN),
                                  10,
                                  ps_list(5, 1, 2, 3, 4, 5, 0, 0, 0),
                                  ps_list(3, -2, -3, -4, 0, 0, 0, 0, 0),
                                  ps_list(2, 7, 9, 0, 0, 0, 0, 0, 0));

    if (count != TASK_COUNT) {
        free(tasks);
        return 0;
    }
    *tasks_out = tasks;
    *count_out = count;
    return 1;
}

static const PSAbstractionSpec *ps_spec_for_rule(const PSContext *context,
                                                 const NNRule *rule)
{
    if (context == NULL || rule == NULL || rule->template_id >= context->spec_count) {
        return NULL;
    }
    return &context->specs[rule->template_id];
}

static void ps_encode_spec_features(const PSAbstractionSpec *spec,
                                    size_t index,
                                    size_t spec_count,
                                    nn_real *features,
                                    size_t capacity)
{
    if (features == NULL || capacity < PS_RULE_FEATURE_COUNT || spec == NULL) {
        return;
    }
    memset(features, 0, capacity * sizeof(*features));
    features[0] = (nn_real)1;
    features[1] = spec->output_type == PS_TYPE_LIST ? (nn_real)1 : (nn_real)0;
    features[2] = spec->output_type == PS_TYPE_INT ? (nn_real)1 : (nn_real)0;
    features[3] = spec->output_type == PS_TYPE_BOOL ? (nn_real)1 : (nn_real)0;
    features[4] = (nn_real)((double)spec->depth / 6.0);
    features[5] = (nn_real)((double)spec->nodes / 18.0);
    features[6] = (nn_real)((double)spec->primitive_count / 6.0);
    features[7] = spec->uses_map ? (nn_real)1 : (nn_real)0;
    features[8] = spec->uses_filter ? (nn_real)1 : (nn_real)0;
    features[9] = spec->uses_fold ? (nn_real)1 : (nn_real)0;
    features[10] = spec->uses_compare ? (nn_real)1 : (nn_real)0;
    features[11] = spec->uses_arithmetic ? (nn_real)1 : (nn_real)0;
    features[12] = spec->uses_ordering ? (nn_real)1 : (nn_real)0;
    features[13] = (nn_real)(1.0 / 4.0);
    features[14] = (nn_real)((double)index / (double)(spec_count == 0 ? 1u : spec_count));
    features[15] = spec->depth >= 5 ? (nn_real)1 : (nn_real)0;
    features[16] = spec->output_type == PS_TYPE_BOOL ? (nn_real)1 : (nn_real)0;
    features[17] = features[14];
}

static int ps_encode_rule(const NNSymbolicDomain *domain,
                          const NNRule *rule,
                          nn_real *features,
                          size_t capacity,
                          size_t *feature_count_out)
{
    PSContext *context = (PSContext *)domain->context;
    const PSAbstractionSpec *spec = ps_spec_for_rule(context, rule);
    if (spec == NULL || features == NULL || capacity < PS_RULE_FEATURE_COUNT) {
        return 0;
    }
    ps_encode_spec_features(spec,
                            (size_t)rule->template_id,
                            context->spec_count,
                            features,
                            capacity);
    if (feature_count_out != NULL) {
        *feature_count_out = PS_RULE_FEATURE_COUNT;
    }
    return 1;
}

static size_t ps_enumerate_rules(const NNSymbolicDomain *domain,
                                 NNRule *rules,
                                 size_t capacity)
{
    PSContext *context = (PSContext *)domain->context;
    if (context == NULL) {
        return 0;
    }
    if (rules == NULL || capacity == 0) {
        return context->spec_count;
    }
    size_t count = 0;
    for (size_t i = 0; i < context->spec_count && count < capacity; ++i) {
        const PSAbstractionSpec *spec = &context->specs[i];
        nn_rule_init(&rules[count],
                     "program_synthesis",
                     spec->name,
                     NN_RULE_KIND_CUSTOM,
                     (uint32_t)i);
        nn_rule_add_param(&rules[count], "op", (double)i / (double)context->spec_count, (int64_t)i);
        nn_rule_add_param(&rules[count],
                          "output_type",
                          (double)spec->output_type / 3.0,
                          (int64_t)spec->output_type);
        nn_rule_add_param(&rules[count], "depth", (double)spec->depth / 6.0, (int64_t)spec->depth);
        nn_rule_add_param(&rules[count], "nodes", (double)spec->nodes / 18.0, (int64_t)spec->nodes);
        ++count;
    }
    return count;
}

static int ps_check_certificate(const NNSymbolicDomain *domain,
                                const NNRule *rule,
                                NNCertificate *certificate)
{
    PSContext *context = (PSContext *)domain->context;
    const PSAbstractionSpec *spec = ps_spec_for_rule(context, rule);
    if (spec == NULL || spec->nodes == 0 || spec->depth == 0 ||
        spec->output_type > PS_TYPE_BOOL) {
        nn_certificate_reject(certificate,
                              "list_dsl_typecheck",
                              "malformed abstraction body",
                              1,
                              1);
        return 1;
    }
    nn_certificate_accept(certificate,
                          "list_dsl_typecheck",
                          "abstraction body is well typed in monomorphic list-int DSL",
                          1,
                          spec->nodes);
    return 1;
}

static int ps_rule_rejects_none(const NNSymbolicDomain *domain,
                                const NNRule *rule,
                                const void *state,
                                size_t action,
                                int *rejects)
{
    (void)domain;
    (void)rule;
    (void)state;
    (void)action;
    if (rejects == NULL) {
        return 0;
    }
    *rejects = 0;
    return 1;
}

static int ps_verify_action(const NNSymbolicDomain *domain,
                            const void *state,
                            size_t action,
                            NNVerifyResult *result)
{
    PSContext *context = (PSContext *)domain->context;
    const PSTask *task = (const PSTask *)state;
    if (context == NULL || task == NULL || result == NULL || action >= context->spec_count) {
        return 0;
    }

    const PSAbstractionSpec *spec = &context->specs[action];
    size_t passed = 0;
    for (size_t i = 0; i < PS_EXAMPLES_PER_TASK; ++i) {
        PSValue produced = ps_eval_op(spec->op, &task->examples[i].input);
        if (ps_value_equal(&produced, &task->examples[i].expected)) {
            ++passed;
        }
    }
    result->correct = passed == PS_EXAMPLES_PER_TASK;
    result->utility = result->correct ? 1.0 : 0.0;
    result->cost = (double)spec->nodes;
    result->target_action = (size_t)task->target_op;
    return 1;
}

static void ps_count_splits(PSContext *context)
{
    context->train_count = 0;
    context->validation_count = 0;
    context->test_count = 0;
    for (size_t i = 0; i < context->task_count; ++i) {
        if (context->tasks[i].split == PS_SPLIT_TRAIN) {
            ++context->train_count;
        } else if (context->tasks[i].split == PS_SPLIT_VALIDATION) {
            ++context->validation_count;
        } else {
            ++context->test_count;
        }
    }
}

static int ps_build_domain(PSContext *context, NNSymbolicDomain *domain)
{
    memset(domain, 0, sizeof(*domain));
    domain->name = "program_synthesis";
    domain->description = "list/integer DSL abstraction discovery";
    domain->symbols = "list<int>,int,bool,map,filter,fold,head,tail";
    domain->axioms = "well-typed abstractions are executable pure functions";
    domain->grammar = "abstraction(name,lambda xs.body) over monomorphic list-int DSL";
    domain->max_rule_count = context->spec_count;
    domain->rule_feature_count = PS_RULE_FEATURE_COUNT;
    domain->leakage_policy = NN_LEAKAGE_PRIORS_ALLOWED;
    domain->rule_feature_schema = PS_RULE_FEATURES;
    domain->rule_feature_schema_count = PS_RULE_FEATURE_COUNT;
    domain->max_action_count = context->spec_count;
    domain->context = context;
    domain->enumerate_rules = ps_enumerate_rules;
    domain->encode_rule = ps_encode_rule;
    domain->check_certificate = ps_check_certificate;
    domain->rule_rejects = ps_rule_rejects_none;
    domain->verify_action = ps_verify_action;
    return nn_symbolic_domain_is_valid(domain);
}

static int ps_split_task_indices(const PSContext *context,
                                 PSSplit split,
                                 size_t *indices,
                                 size_t capacity,
                                 size_t *count_out)
{
    size_t count = 0;
    for (size_t i = 0; i < context->task_count; ++i) {
        if (context->tasks[i].split != split) {
            continue;
        }
        if (count >= capacity) {
            return 0;
        }
        indices[count++] = i;
    }
    *count_out = count;
    return 1;
}

static int ps_build_coverage_for_split(const PSContext *context,
                                       const NNSymbolicDomain *domain,
                                       const NNRule *rules,
                                       size_t rule_count,
                                       const uint8_t *certified,
                                       PSSplit split,
                                       NNRuleCoverage *coverage)
{
    size_t task_indices[64];
    size_t event_count = 0;
    if (!ps_split_task_indices(context, split, task_indices, 64, &event_count) ||
        !nn_rule_coverage_init(coverage, rule_count, event_count)) {
        return 0;
    }

    for (size_t rule_index = 0; rule_index < rule_count; ++rule_index) {
        if (!certified[rule_index]) {
            continue;
        }
        const PSAbstractionSpec *spec = &context->specs[rules[rule_index].template_id];
        uint64_t bit = ps_mask1(spec->op);
        for (size_t event = 0; event < event_count; ++event) {
            const PSTask *task = &context->tasks[task_indices[event]];
            if ((task->useful_mask & bit) == 0) {
                continue;
            }
            NNVerifyResult verify;
            if (!nn_symbolic_domain_verify_action(domain, task, rule_index, &verify)) {
                nn_rule_coverage_free(coverage);
                return 0;
            }
            if (verify.correct || spec->op != task->target_op) {
                nn_rule_coverage_set(coverage, rule_index, event);
            }
        }
    }
    return 1;
}

static size_t ps_count_certified(const uint8_t *certified, size_t count)
{
    size_t total = 0;
    for (size_t i = 0; i < count; ++i) {
        total += certified[i] ? 1u : 0u;
    }
    return total;
}

static size_t ps_selected_covered(const NNRuleCoverage *coverage,
                                  const size_t *indices,
                                  size_t count)
{
    if (coverage == NULL || coverage->word_count == 0) {
        return 0;
    }
    uint64_t *covered = calloc(coverage->word_count, sizeof(*covered));
    if (covered == NULL) {
        return 0;
    }
    for (size_t i = 0; i < count; ++i) {
        nn_rule_coverage_add_rule_to_residual(coverage, indices[i], covered);
    }
    size_t out = nn_rule_coverage_count_covered(covered, coverage->word_count);
    free(covered);
    return out;
}

static size_t ps_rule_target_events(const PSContext *context,
                                    const NNRuleCoverage *coverage,
                                    PSSplit split,
                                    size_t rule_index,
                                    PSOp target_op,
                                    size_t *events_out)
{
    size_t covered = 0;
    size_t events = 0;
    size_t event_index = 0;
    if (context == NULL || coverage == NULL || rule_index >= coverage->rule_count) {
        if (events_out != NULL) {
            *events_out = 0;
        }
        return 0;
    }
    for (size_t task_index = 0; task_index < context->task_count; ++task_index) {
        const PSTask *task = &context->tasks[task_index];
        if (task->split != split) {
            continue;
        }
        if (task->target_op == target_op) {
            ++events;
            if (event_index < coverage->event_count &&
                nn_rule_coverage_get(coverage, rule_index, event_index)) {
                ++covered;
            }
        }
        ++event_index;
    }
    if (events_out != NULL) {
        *events_out = events;
    }
    return covered;
}

static int ps_emit_hypothesis_evidence(const PSContext *context,
                                       const NNRule *rules,
                                       size_t rule_count,
                                       const uint8_t *certified,
                                       const NNRuleCoverage *train_coverage,
                                       const NNRuleCoverage *validation_coverage,
                                       const NNRuleCoverage *test_coverage)
{
    if (context == NULL || rules == NULL || certified == NULL ||
        train_coverage == NULL || validation_coverage == NULL || test_coverage == NULL) {
        return 0;
    }
    for (size_t i = 0; i < rule_count; ++i) {
        if (!certified[i] || rules[i].template_id >= context->spec_count) {
            continue;
        }
        const PSAbstractionSpec *spec = &context->specs[rules[i].template_id];
        NNHypothesis hypothesis;
        NNProgram program;
        if (!ps_bind_filter_reduce_hypothesis(spec, &hypothesis) ||
            !nn_hypothesis_compile(&hypothesis, &program)) {
            continue;
        }
        size_t train_events = 0;
        size_t validation_events = 0;
        size_t test_events = 0;
        size_t covered =
            ps_rule_target_events(context,
                                  train_coverage,
                                  PS_SPLIT_TRAIN,
                                  i,
                                  spec->op,
                                  &train_events) +
            ps_rule_target_events(context,
                                  validation_coverage,
                                  PS_SPLIT_VALIDATION,
                                  i,
                                  spec->op,
                                  &validation_events) +
            ps_rule_target_events(context,
                                  test_coverage,
                                  PS_SPLIT_TEST,
                                  i,
                                  spec->op,
                                  &test_events);
        size_t events = train_events + validation_events + test_events;
        NNHypothesisEvidence evidence = {
            .task = "program_synthesis",
            .task_instance = spec->name,
            .program = &program,
            .certificate_status = rules[i].certificate.status,
            .proof_kind = rules[i].certificate.proof_kind,
            .reason = rules[i].certificate.reason,
            .event_count = events,
            .covered_events = covered,
            .support = rules[i].certificate.support,
            .false_rejections = rules[i].certificate.false_rejections,
            .utility = events == 0 ? 0.0 : (double)covered / (double)events,
        };
        nn_hypothesis_write_evidence_json(stdout, &hypothesis, &evidence);
    }
    return 1;
}

static uint64_t ps_rng_next(uint64_t *state)
{
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return *state * UINT64_C(2685821657736338717);
}

static int ps_accept_candidate(const PSRunConfig *config, size_t index)
{
    return config != NULL && index < config->rule_count && config->certified[index] != 0;
}

static void ps_fill_metrics(const PSRunConfig *config, PSRunMetrics *metrics)
{
    metrics->proposed_rules = config->rule_count;
    metrics->verified_rules = ps_count_certified(config->certified, config->rule_count);
    metrics->train_covered =
        ps_selected_covered(config->train_coverage, metrics->accepted_indices, metrics->accepted_count);
    metrics->validation_covered =
        ps_selected_covered(config->validation_coverage,
                            metrics->accepted_indices,
                            metrics->accepted_count);
    metrics->test_covered =
        ps_selected_covered(config->test_coverage, metrics->accepted_indices, metrics->accepted_count);
}

static int ps_select_grammar(const PSRunConfig *config, PSRunMetrics *metrics)
{
    memset(metrics, 0, sizeof(*metrics));
    for (size_t i = 0; i < config->rule_count && metrics->accepted_count < config->max_rules; ++i) {
        ++metrics->proposer_calls;
        if (!ps_accept_candidate(config, i)) {
            continue;
        }
        metrics->accepted_indices[metrics->accepted_count++] = i;
    }
    ps_fill_metrics(config, metrics);
    return 1;
}

static int ps_select_random(const PSRunConfig *config, PSRunMetrics *metrics)
{
    memset(metrics, 0, sizeof(*metrics));
    size_t order[PS_OP_COUNT];
    if (config->rule_count > PS_OP_COUNT) {
        return 0;
    }
    for (size_t i = 0; i < config->rule_count; ++i) {
        order[i] = i;
    }
    uint64_t rng = ((uint64_t)config->seed << 32) ^ UINT64_C(0x9e3779b97f4a7c15);
    for (size_t i = config->rule_count; i > 1; --i) {
        size_t j = (size_t)(ps_rng_next(&rng) % i);
        size_t tmp = order[i - 1u];
        order[i - 1u] = order[j];
        order[j] = tmp;
    }
    for (size_t pos = 0; pos < config->rule_count && metrics->accepted_count < config->max_rules; ++pos) {
        ++metrics->proposer_calls;
        size_t index = order[pos];
        if (!ps_accept_candidate(config, index)) {
            continue;
        }
        metrics->accepted_indices[metrics->accepted_count++] = index;
    }
    ps_fill_metrics(config, metrics);
    return 1;
}

static int ps_compare_structural_order(const PSRunConfig *config, size_t a, size_t b)
{
    const NNRule *ra = &config->rules[a];
    const NNRule *rb = &config->rules[b];
    const PSAbstractionSpec *sa = &PS_SPECS[ra->template_id];
    const PSAbstractionSpec *sb = &PS_SPECS[rb->template_id];
    int score_a = 2 * sa->uses_fold + 2 * sa->uses_filter + 2 * sa->uses_map +
                  sa->uses_compare + sa->uses_arithmetic + sa->uses_ordering;
    int score_b = 2 * sb->uses_fold + 2 * sb->uses_filter + 2 * sb->uses_map +
                  sb->uses_compare + sb->uses_arithmetic + sb->uses_ordering;
    if (score_a != score_b) {
        return score_a > score_b;
    }
    if (sa->nodes != sb->nodes) {
        return sa->nodes < sb->nodes;
    }
    return a < b;
}

static int ps_select_structural(const PSRunConfig *config, PSRunMetrics *metrics)
{
    memset(metrics, 0, sizeof(*metrics));
    size_t order[PS_OP_COUNT];
    if (config->rule_count > PS_OP_COUNT) {
        return 0;
    }
    for (size_t i = 0; i < config->rule_count; ++i) {
        order[i] = i;
    }
    for (size_t i = 1; i < config->rule_count; ++i) {
        size_t value = order[i];
        size_t j = i;
        while (j > 0 && ps_compare_structural_order(config, value, order[j - 1u])) {
            order[j] = order[j - 1u];
            --j;
        }
        order[j] = value;
    }
    for (size_t pos = 0; pos < config->rule_count && metrics->accepted_count < config->max_rules; ++pos) {
        ++metrics->proposer_calls;
        size_t index = order[pos];
        if (!ps_accept_candidate(config, index)) {
            continue;
        }
        metrics->accepted_indices[metrics->accepted_count++] = index;
    }
    ps_fill_metrics(config, metrics);
    return 1;
}

static int ps_greedy_select_coverage(const PSRunConfig *config,
                                     const NNRuleCoverage *coverage,
                                     PSRunMetrics *metrics)
{
    memset(metrics, 0, sizeof(*metrics));
    uint8_t selected[PS_OP_COUNT];
    uint64_t covered_words[2];
    if (config->rule_count > PS_OP_COUNT || coverage->word_count > 2u) {
        return 0;
    }
    memset(selected, 0, sizeof(selected));
    memset(covered_words, 0, sizeof(covered_words));

    for (size_t step = 0; step < config->max_rules; ++step) {
        size_t best_index = (size_t)-1;
        size_t best_gain = 0;
        for (size_t i = 0; i < config->rule_count; ++i) {
            if (selected[i] || !ps_accept_candidate(config, i)) {
                continue;
            }
            ++metrics->proposer_calls;
            size_t gain = nn_rule_coverage_gain(coverage, i, covered_words);
            if (best_index == (size_t)-1 || gain > best_gain) {
                best_index = i;
                best_gain = gain;
            }
        }
        if (best_index == (size_t)-1 || best_gain == 0) {
            break;
        }
        selected[best_index] = 1;
        metrics->accepted_indices[metrics->accepted_count++] = best_index;
        nn_rule_coverage_add_rule_to_residual(coverage, best_index, covered_words);
    }
    ps_fill_metrics(config, metrics);
    return 1;
}

static int ps_select_validation_oracle(const PSRunConfig *config, PSRunMetrics *metrics)
{
    return ps_greedy_select_coverage(config, config->validation_coverage, metrics);
}

static int ps_select_neural(const PSRunConfig *config, PSRunMetrics *metrics)
{
    memset(metrics, 0, sizeof(*metrics));
    size_t count = config->rule_count;
    nn_real inputs[PS_OP_COUNT * PS_RULE_FEATURE_COUNT];
    nn_real targets[PS_OP_COUNT];
    nn_real outputs[PS_OP_COUNT];
    size_t order[PS_OP_COUNT];
    if (count > PS_OP_COUNT) {
        return 0;
    }

    for (size_t i = 0; i < count; ++i) {
        const PSAbstractionSpec *spec = &PS_SPECS[config->rules[i].template_id];
        ps_encode_spec_features(spec, i, count, &inputs[i * PS_RULE_FEATURE_COUNT], PS_RULE_FEATURE_COUNT);
        uint64_t empty_covered[2] = {0, 0};
        targets[i] =
            (nn_real)((double)nn_rule_coverage_gain(config->train_coverage, i, empty_covered) /
                      (double)(config->train_coverage->event_count == 0
                                   ? 1u
                                   : config->train_coverage->event_count));
        order[i] = i;
    }

    const size_t layers[] = {PS_RULE_FEATURE_COUNT, 20, 1};
    NNConfig nn_config;
    nn_config.layer_sizes = layers;
    nn_config.layer_count = sizeof(layers) / sizeof(layers[0]);
    nn_config.seed = config->seed ^ 0x51491u;
    nn_config.hidden_activation = NN_ACTIVATION_RELU;
    nn_config.output_activation = NN_ACTIVATION_LINEAR;
    nn_config.loss = NN_LOSS_MSE;
    NeuralNet *net = nn_create_config(&nn_config);
    if (net == NULL) {
        return 0;
    }
    nn_real loss = nn_train_batched(net,
                                    inputs,
                                    targets,
                                    count,
                                    count,
                                    (nn_real)0.035,
                                    PS_STATIC_EPOCHS,
                                    0);
    metrics->parameters = nn_parameter_count(net);
    metrics->forward_flops = nn_forward_flop_count(net);
    nn_forward_batch(net, inputs, outputs, count);
    nn_free(net);
    if (isnan((double)loss)) {
        return 0;
    }
    metrics->proposer_loss = (double)loss;

    for (size_t i = 1; i < count; ++i) {
        size_t value = order[i];
        size_t j = i;
        while (j > 0 && outputs[value] > outputs[order[j - 1u]]) {
            order[j] = order[j - 1u];
            --j;
        }
        order[j] = value;
    }
    for (size_t pos = 0; pos < count && metrics->accepted_count < config->max_rules; ++pos) {
        ++metrics->proposer_calls;
        size_t index = order[pos];
        if (!ps_accept_candidate(config, index)) {
            continue;
        }
        metrics->accepted_indices[metrics->accepted_count++] = index;
    }
    ps_fill_metrics(config, metrics);
    return 1;
}

typedef struct {
    const PSRunConfig *config;
} PSClosedLoopContext;

static int ps_closed_loop_candidate_allowed(size_t rule_index,
                                            const size_t *accepted_indices,
                                            size_t accepted_count,
                                            void *context)
{
    (void)accepted_indices;
    (void)accepted_count;
    PSClosedLoopContext *closed_loop = (PSClosedLoopContext *)context;
    return closed_loop != NULL && ps_accept_candidate(closed_loop->config, rule_index);
}

static int ps_closed_loop_keep_accepted(size_t accepted_rule_index,
                                        size_t candidate_rule_index,
                                        void *context)
{
    (void)accepted_rule_index;
    (void)candidate_rule_index;
    (void)context;
    return 1;
}

static int ps_closed_loop_encode(size_t rule_index,
                                 const size_t *accepted_indices,
                                 size_t accepted_count,
                                 const NNRuleCoverage *coverage,
                                 const uint64_t *covered,
                                 void *context,
                                 nn_real *features,
                                 size_t capacity)
{
    PSClosedLoopContext *closed_loop = (PSClosedLoopContext *)context;
    if (closed_loop == NULL || closed_loop->config == NULL || features == NULL ||
        capacity < PS_CLOSED_LOOP_FEATURE_COUNT || rule_index >= closed_loop->config->rule_count) {
        return 0;
    }
    const PSRunConfig *config = closed_loop->config;
    const NNRule *rule = &config->rules[rule_index];
    const PSAbstractionSpec *spec = &PS_SPECS[rule->template_id];
    ps_encode_spec_features(spec, rule_index, config->rule_count, features, capacity);

    size_t same_output = 0;
    size_t same_map = 0;
    size_t same_filter = 0;
    size_t same_fold = 0;
    for (size_t i = 0; i < accepted_count; ++i) {
        size_t accepted_index = accepted_indices[i];
        if (accepted_index >= config->rule_count) {
            continue;
        }
        const PSAbstractionSpec *accepted = &PS_SPECS[config->rules[accepted_index].template_id];
        same_output += accepted->output_type == spec->output_type ? 1u : 0u;
        same_map += accepted->uses_map == spec->uses_map ? 1u : 0u;
        same_filter += accepted->uses_filter == spec->uses_filter ? 1u : 0u;
        same_fold += accepted->uses_fold == spec->uses_fold ? 1u : 0u;
    }

    NNResidualSummary summary;
    nn_residual_summary_from_coverage(&summary,
                                      accepted_count,
                                      config->max_rules,
                                      coverage,
                                      covered);
    nn_real residual[4];
    nn_residual_summary_write_features(&summary, residual, 4);
    double denom = accepted_count == 0 ? 1.0 : (double)accepted_count;
    features[PS_RULE_FEATURE_COUNT + 0] = residual[0];
    features[PS_RULE_FEATURE_COUNT + 1] = residual[1];
    features[PS_RULE_FEATURE_COUNT + 2] = residual[2];
    features[PS_RULE_FEATURE_COUNT + 3] = (nn_real)((double)same_output / denom);
    features[PS_RULE_FEATURE_COUNT + 4] = (nn_real)((double)same_map / denom);
    features[PS_RULE_FEATURE_COUNT + 5] = (nn_real)((double)same_filter / denom);
    features[PS_RULE_FEATURE_COUNT + 6] = (nn_real)((double)same_fold / denom);
    features[PS_RULE_FEATURE_COUNT + 7] =
        nn_rule_coverage_gain(coverage, rule_index, covered) == 0 ? (nn_real)1 : (nn_real)0;
    return 1;
}

static int ps_select_closed_loop(const PSRunConfig *config, PSRunMetrics *metrics)
{
    memset(metrics, 0, sizeof(*metrics));
    metrics->proposed_rules = config->rule_count;
    metrics->verified_rules = ps_count_certified(config->certified, config->rule_count);

    PSClosedLoopContext context;
    context.config = config;
    size_t eligible = 0;
    for (size_t i = 0; i < config->rule_count; ++i) {
        eligible += ps_accept_candidate(config, i) ? 1u : 0u;
    }
    metrics->softmax_group_size = eligible;

    NNClosedLoopRulePolicyConfig policy_config;
    nn_closed_loop_rule_policy_config_defaults(&policy_config, PS_CLOSED_LOOP_FEATURE_COUNT);
    policy_config.max_rules = config->max_rules;
    policy_config.max_train_rules_per_step = eligible == 0 ? 1u : eligible;
    policy_config.epochs = config->closed_loop_epochs;
    policy_config.batch_size = 64;
    policy_config.hidden1 = config->closed_loop_hidden1;
    policy_config.hidden2 = config->closed_loop_hidden2;
    policy_config.learning_rate = (nn_real)0.025;
    policy_config.seed = config->seed ^ 0xbadc0deu;
    policy_config.leakage_policy = NN_LEAKAGE_GATE_ALLOWED;
    policy_config.feature_schema = PS_CLOSED_LOOP_FEATURES;
    policy_config.feature_schema_count = PS_CLOSED_LOOP_FEATURE_COUNT;

    NNClosedLoopRulePolicy policy;
    if (!nn_closed_loop_rule_policy_train(config->train_coverage,
                                          &policy_config,
                                          ps_closed_loop_encode,
                                          ps_closed_loop_candidate_allowed,
                                          ps_closed_loop_keep_accepted,
                                          &context,
                                          &policy)) {
        return 0;
    }
    metrics->proposer_loss = policy.loss;
    metrics->residual_train_samples = policy.sample_count;
    metrics->parameters = nn_parameter_count(policy.net);
    metrics->forward_flops = nn_forward_flop_count(policy.net);

    nn_real scores[PS_OP_COUNT];
    size_t proposal_calls = 0;
    int ok = nn_closed_loop_rule_policy_select(config->train_coverage,
                                               &policy_config,
                                               ps_closed_loop_encode,
                                               ps_closed_loop_candidate_allowed,
                                               ps_closed_loop_keep_accepted,
                                               &context,
                                               &policy,
                                               scores,
                                               metrics->accepted_indices,
                                               config->max_rules,
                                               &metrics->accepted_count,
                                               &proposal_calls);
    (void)scores;
    nn_closed_loop_rule_policy_free(&policy);
    if (!ok) {
        return 0;
    }
    metrics->proposer_calls = proposal_calls;
    ps_fill_metrics(config, metrics);
    metrics->softmax_group_size = eligible;
    return 1;
}

static const char *ps_proposer_name(PSProposerMode mode)
{
    switch (mode) {
    case PS_PROP_RANDOM:
        return "random";
    case PS_PROP_GRAMMAR:
        return "grammar_order";
    case PS_PROP_STRUCTURAL:
        return "structural_heuristic";
    case PS_PROP_VALIDATION_ORACLE:
        return "validation_oracle";
    case PS_PROP_NEURAL:
        return "neural_structural";
    case PS_PROP_CLOSED_LOOP:
        return "closed_loop_neural";
    default:
        return "all";
    }
}

static int ps_parse_proposer(const char *text, PSProposerMode *mode_out)
{
    if (strcmp(text, "random") == 0) {
        *mode_out = PS_PROP_RANDOM;
    } else if (strcmp(text, "grammar") == 0 || strcmp(text, "grammar-order") == 0) {
        *mode_out = PS_PROP_GRAMMAR;
    } else if (strcmp(text, "structural") == 0) {
        *mode_out = PS_PROP_STRUCTURAL;
    } else if (strcmp(text, "validation-oracle") == 0 || strcmp(text, "oracle") == 0) {
        *mode_out = PS_PROP_VALIDATION_ORACLE;
    } else if (strcmp(text, "neural") == 0) {
        *mode_out = PS_PROP_NEURAL;
    } else if (strcmp(text, "closed-loop") == 0 || strcmp(text, "closed_loop") == 0) {
        *mode_out = PS_PROP_CLOSED_LOOP;
    } else if (strcmp(text, "all") == 0) {
        *mode_out = PS_PROP_ALL;
    } else {
        return 0;
    }
    return 1;
}

static int ps_parse_hidden(const char *text, size_t *hidden1, size_t *hidden2)
{
    char *end = NULL;
    unsigned long a = strtoul(text, &end, 10);
    if (end == text || a == 0 || (*end != 'x' && *end != 'X')) {
        return 0;
    }
    char *second = end + 1;
    unsigned long b = strtoul(second, &end, 10);
    if (end == second || *end != '\0') {
        return 0;
    }
    *hidden1 = (size_t)a;
    *hidden2 = (size_t)b;
    return 1;
}

static void ps_hidden_json(PSProposerMode mode,
                           size_t hidden1,
                           size_t hidden2,
                           char *buffer,
                           size_t capacity)
{
    if (mode == PS_PROP_CLOSED_LOOP) {
        snprintf(buffer, capacity, "\"%zux%zu\"", hidden1, hidden2);
    } else {
        snprintf(buffer, capacity, "null");
    }
}

static double ps_fraction(size_t numerator, size_t denominator)
{
    return denominator == 0 ? 0.0 : (double)numerator / (double)denominator;
}

static int ps_run_mode(PSProposerMode mode,
                       const PSRunConfig *config,
                       const PSContext *context,
                       int json_only)
{
    PSRunMetrics metrics;
    int ok = 0;
    switch (mode) {
    case PS_PROP_RANDOM:
        ok = ps_select_random(config, &metrics);
        break;
    case PS_PROP_GRAMMAR:
        ok = ps_select_grammar(config, &metrics);
        break;
    case PS_PROP_STRUCTURAL:
        ok = ps_select_structural(config, &metrics);
        break;
    case PS_PROP_VALIDATION_ORACLE:
        ok = ps_select_validation_oracle(config, &metrics);
        break;
    case PS_PROP_NEURAL:
        ok = ps_select_neural(config, &metrics);
        break;
    case PS_PROP_CLOSED_LOOP:
        ok = ps_select_closed_loop(config, &metrics);
        break;
    default:
        return 0;
    }
    if (!ok) {
        return 0;
    }

    char hidden_json[64];
    ps_hidden_json(mode,
                   config->closed_loop_hidden1,
                   config->closed_loop_hidden2,
                   hidden_json,
                   sizeof(hidden_json));
    double train_compression = ps_fraction(metrics.train_covered, context->train_count);
    double validation_compression =
        ps_fraction(metrics.validation_covered, context->validation_count);
    double test_compression = ps_fraction(metrics.test_covered, context->test_count);
    printf("{\"schema\":\"crlp.rule_search.summary.v1\",\"task\":\"program_synthesis\","
           "\"grammar_version\":%u,\"rule_set_hash\":\"list-dsl-abstractions-v1\","
           "\"selection_strategy\":\"%s\","
           "\"selection_seed\":%u,"
           "\"proposer_feature_set\":\"%s\","
           "\"proposer_target\":\"%s\","
           "\"max_rules\":%zu,"
           "\"softmax_group_size\":%zu,"
           "\"closed_loop_hidden\":%s,"
           "\"unified_scorer_hidden\":null,"
           "\"proposed_rules\":%zu,"
           "\"verified_rules\":%zu,"
           "\"accepted_rules\":%zu,"
           "\"proposer_calls\":%zu,"
           "\"unsafe_rules\":0,"
           "\"no_support_rules\":0,"
           "\"uncertified_rules\":0,"
           "\"false_rejections\":0,"
           "\"train_tasks\":%zu,"
           "\"validation_tasks\":%zu,"
           "\"test_tasks\":%zu,"
           "\"train_rejects\":%zu,"
           "\"validation_rejects\":%zu,"
           "\"test_rejects\":%zu,"
           "\"train_covered\":%zu,"
           "\"validation_covered\":%zu,"
           "\"test_covered\":%zu,"
           "\"train_compression\":%.8f,"
           "\"validation_compression\":%.8f,"
           "\"test_compression\":%.8f,"
           "\"expanded_saved_pct\":%.8f,"
           "\"residual_train_samples\":%zu,"
           "\"parameters\":%zu,"
           "\"forward_flops\":%zu,"
           "\"proposer_loss\":%.8f}\n",
           (unsigned int)PS_GRAMMAR_VERSION,
           ps_proposer_name(mode),
           config->seed,
           mode == PS_PROP_CLOSED_LOOP ? "structural_plus_residual" : "structural",
           mode == PS_PROP_VALIDATION_ORACLE
               ? "validation_compression_gain"
               : (mode == PS_PROP_CLOSED_LOOP ? "greedy_train_residual_gain"
                                               : "train_compression_gain"),
           config->max_rules,
           metrics.softmax_group_size,
           hidden_json,
           metrics.proposed_rules,
           metrics.verified_rules,
           metrics.accepted_count,
           metrics.proposer_calls,
           context->train_count,
           context->validation_count,
           context->test_count,
           metrics.train_covered,
           metrics.validation_covered,
           metrics.test_covered,
           metrics.train_covered,
           metrics.validation_covered,
           metrics.test_covered,
           train_compression,
           validation_compression,
           test_compression,
           100.0 * validation_compression,
           metrics.residual_train_samples,
           metrics.parameters,
           metrics.forward_flops,
           metrics.proposer_loss);

    if (!json_only) {
        printf("[program_synthesis] strategy=%s budget=%zu accepted=%zu "
               "train=%.3f validation=%.3f test=%.3f\n",
               ps_proposer_name(mode),
               config->max_rules,
               metrics.accepted_count,
               train_compression,
               validation_compression,
               test_compression);
    }
    return 1;
}

static int ps_write_schema(void)
{
    NNExperimentSpec spec;
    memset(&spec, 0, sizeof(spec));
    spec.task_name = "program_synthesis";
    spec.role = "case_study";
    spec.state_space = "list-routine programming task with input/output examples";
    spec.state_distribution = "compiled synthetic list-routine corpus split by task family";
    spec.action_space = "candidate library abstraction in the list/int DSL";
    spec.candidate_enumerator = "bottom-up abstraction table for Phase 0 engine shakedown";
    spec.observable_features = "structural abstraction features only";
    spec.leakage_policy = NN_LEAKAGE_PRIORS_ALLOWED;
    spec.neural_scorer = "static structural scorer or closed-loop residual proposer";
    spec.exact_verifier = "run candidate abstraction on every input/output example";
    spec.verifier_feedback = "per-task exact pass/fail plus compression coverage diagnostics";
    spec.rule_grammar = "abstraction(name, lambda xs.body) over map/filter/fold/list primitives";
    spec.certificate_checker = "syntactic monomorphic DSL type checker";
    spec.gate_executor = "accepted abstraction library used as MDL compression set";
    spec.proposer_inputs = "structural rule features; closed-loop rows add declared residual coverage features";
    spec.proposer_targets = "marginal task-compression gain under matched rule budget";
    spec.proposer_mode = NN_PROPOSER_DISCOVERY;
    spec.baseline_orderings = "random, grammar, structural, validation-oracle, neural, closed-loop";
    spec.generalization_axis = "task-family split across train, validation, and test routines";
    spec.trace_schema = "crlp.rule_search.summary.v1";
    spec.replay_protocol = "deterministic seed and rule_set_hash identify the selected library";
    spec.rl_protocol = "not applicable for Phase 1";
    spec.features = PS_RULE_FEATURES;
    spec.feature_count = PS_RULE_FEATURE_COUNT;
    return nn_experiment_write_schema_json(stdout, &spec);
}

static void ps_usage(const char *argv0)
{
    fprintf(stderr,
            "usage: %s [--schema] [--proposer all|random|grammar|structural|validation-oracle|neural|closed-loop]\n"
            "          [--max-rules N] [--budget-grid] [--seed N] [--closed-loop-hidden AxB]\n",
            argv0);
}

int main(int argc, char **argv)
{
    PSProposerMode proposer = PS_PROP_ALL;
    size_t max_rules = 16;
    unsigned int seed = 20260516u;
    int schema_only = 0;
    int budget_grid = 0;
    int json_only = 1;
    size_t closed_loop_hidden1 = 32;
    size_t closed_loop_hidden2 = 16;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--schema") == 0) {
            schema_only = 1;
        } else if (strcmp(argv[i], "--budget-grid") == 0) {
            budget_grid = 1;
        } else if (strcmp(argv[i], "--report") == 0) {
            json_only = 0;
        } else if (strcmp(argv[i], "--json") == 0) {
            json_only = 1;
        } else if (strcmp(argv[i], "--proposer") == 0 && i + 1 < argc) {
            if (!ps_parse_proposer(argv[++i], &proposer)) {
                ps_usage(argv[0]);
                return EXIT_FAILURE;
            }
        } else if (strcmp(argv[i], "--max-rules") == 0 && i + 1 < argc) {
            max_rules = (size_t)strtoull(argv[++i], NULL, 10);
            if (max_rules == 0 || max_rules > PS_MAX_ACCEPTED) {
                ps_usage(argv[0]);
                return EXIT_FAILURE;
            }
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = (unsigned int)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--closed-loop-hidden") == 0 && i + 1 < argc) {
            if (!ps_parse_hidden(argv[++i], &closed_loop_hidden1, &closed_loop_hidden2)) {
                ps_usage(argv[0]);
                return EXIT_FAILURE;
            }
        } else if (strcmp(argv[i], "--help") == 0) {
            ps_usage(argv[0]);
            return EXIT_SUCCESS;
        } else {
            ps_usage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    if (schema_only) {
        return ps_write_schema() ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    PSTask *tasks = NULL;
    size_t task_count = 0;
    if (!ps_init_tasks(&tasks, &task_count)) {
        return EXIT_FAILURE;
    }

    PSContext context;
    memset(&context, 0, sizeof(context));
    context.specs = PS_SPECS;
    context.spec_count = sizeof(PS_SPECS) / sizeof(PS_SPECS[0]);
    context.tasks = tasks;
    context.task_count = task_count;
    ps_count_splits(&context);

    NNSymbolicDomain domain;
    if (!ps_build_domain(&context, &domain)) {
        free(tasks);
        return EXIT_FAILURE;
    }

    size_t available = nn_symbolic_domain_enumerate_rules(&domain, NULL, 0);
    NNRule *rules = calloc(available, sizeof(*rules));
    uint8_t *certified = calloc(available, sizeof(*certified));
    if (rules == NULL || certified == NULL) {
        free(rules);
        free(certified);
        free(tasks);
        return EXIT_FAILURE;
    }
    size_t rule_count = nn_symbolic_domain_enumerate_rules(&domain, rules, available);
    for (size_t i = 0; i < rule_count; ++i) {
        NNCertificate certificate;
        if (!nn_symbolic_domain_check_certificate(&domain, &rules[i], &certificate)) {
            free(rules);
            free(certified);
            free(tasks);
            return EXIT_FAILURE;
        }
        certified[i] = certificate.status == NN_CERTIFICATE_CERTIFIED ? 1u : 0u;
    }

    NNRuleCoverage train_coverage;
    NNRuleCoverage validation_coverage;
    NNRuleCoverage test_coverage;
    memset(&train_coverage, 0, sizeof(train_coverage));
    memset(&validation_coverage, 0, sizeof(validation_coverage));
    memset(&test_coverage, 0, sizeof(test_coverage));
    if (!ps_build_coverage_for_split(&context,
                                     &domain,
                                     rules,
                                     rule_count,
                                     certified,
                                     PS_SPLIT_TRAIN,
                                     &train_coverage) ||
        !ps_build_coverage_for_split(&context,
                                     &domain,
                                     rules,
                                     rule_count,
                                     certified,
                                     PS_SPLIT_VALIDATION,
                                     &validation_coverage) ||
        !ps_build_coverage_for_split(&context,
                                     &domain,
                                     rules,
                                     rule_count,
                                     certified,
                                     PS_SPLIT_TEST,
                                     &test_coverage)) {
        nn_rule_coverage_free(&train_coverage);
        nn_rule_coverage_free(&validation_coverage);
        nn_rule_coverage_free(&test_coverage);
        free(rules);
        free(certified);
        free(tasks);
        return EXIT_FAILURE;
    }

    if (!ps_emit_hypothesis_evidence(&context,
                                     rules,
                                     rule_count,
                                     certified,
                                     &train_coverage,
                                     &validation_coverage,
                                     &test_coverage)) {
        nn_rule_coverage_free(&train_coverage);
        nn_rule_coverage_free(&validation_coverage);
        nn_rule_coverage_free(&test_coverage);
        free(rules);
        free(certified);
        free(tasks);
        return EXIT_FAILURE;
    }

    PSRunConfig config;
    memset(&config, 0, sizeof(config));
    config.rules = rules;
    config.rule_count = rule_count;
    config.train_coverage = &train_coverage;
    config.validation_coverage = &validation_coverage;
    config.test_coverage = &test_coverage;
    config.certified = certified;
    config.seed = seed;
    config.closed_loop_hidden1 = closed_loop_hidden1;
    config.closed_loop_hidden2 = closed_loop_hidden2;
    config.closed_loop_epochs = PS_CLOSED_LOOP_EPOCHS;

    static const size_t budgets[] = {1, 4, 16, 64};
    PSProposerMode modes[] = {
        PS_PROP_RANDOM,
        PS_PROP_GRAMMAR,
        PS_PROP_STRUCTURAL,
        PS_PROP_VALIDATION_ORACLE,
        PS_PROP_NEURAL,
        PS_PROP_CLOSED_LOOP,
    };
    size_t budget_count = budget_grid ? sizeof(budgets) / sizeof(budgets[0]) : 1u;
    int ok = 1;
    for (size_t b = 0; b < budget_count && ok; ++b) {
        config.max_rules = budget_grid ? budgets[b] : max_rules;
        if (config.max_rules > PS_MAX_ACCEPTED) {
            config.max_rules = PS_MAX_ACCEPTED;
        }
        if (proposer == PS_PROP_ALL) {
            for (size_t m = 0; m < sizeof(modes) / sizeof(modes[0]); ++m) {
                ok = ps_run_mode(modes[m], &config, &context, json_only);
                if (!ok) {
                    break;
                }
            }
        } else {
            ok = ps_run_mode(proposer, &config, &context, json_only);
        }
    }

    nn_rule_coverage_free(&train_coverage);
    nn_rule_coverage_free(&validation_coverage);
    nn_rule_coverage_free(&test_coverage);
    free(rules);
    free(certified);
    free(tasks);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
