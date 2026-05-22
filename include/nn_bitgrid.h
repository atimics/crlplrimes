#ifndef NN_BITGRID_H
#define NN_BITGRID_H

#include "nn_symbolic.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#define NN_BITGRID_GATE_SCHEMA "crlp.bitgrid.gate.v1"
#define NN_BITGRID_MAX_REASON 160u
#define NN_BITWISE_BINARY_TRUTH_TABLES 16u
#define NN_BITWISE_TOOL_FEATURE_COUNT 32u
#define NN_BITWISE_CHAIN_MAX_NODES 32u
#define NN_BITWISE_MACRO_SCHEMA "crlp.bitgrid.macro_tool.v1"
#define NN_BITWISE_MACRO_MAX_TOOLS 16u

typedef enum {
    NN_BITWISE_FALSE = 0x0,
    NN_BITWISE_NOR = 0x1,
    NN_BITWISE_NOT_A_AND_B = 0x2,
    NN_BITWISE_NOT_A = 0x3,
    NN_BITWISE_A_AND_NOT_B = 0x4,
    NN_BITWISE_NOT_B = 0x5,
    NN_BITWISE_XOR = 0x6,
    NN_BITWISE_NAND = 0x7,
    NN_BITWISE_AND = 0x8,
    NN_BITWISE_XNOR = 0x9,
    NN_BITWISE_B = 0xa,
    NN_BITWISE_NOT_A_OR_B = 0xb,
    NN_BITWISE_A = 0xc,
    NN_BITWISE_A_OR_NOT_B = 0xd,
    NN_BITWISE_OR = 0xe,
    NN_BITWISE_TRUE = 0xf,
} NNBitwiseOp;

typedef struct {
    size_t width;
    size_t height;
    uint8_t *cells;
} NNBitGrid;

typedef struct {
    size_t hidden_width;
    size_t max_epochs;
    size_t batch_size;
    size_t retry_count;
    nn_real learning_rate;
    nn_real min_margin;
    unsigned int seed;
    int shuffle;
} NNBitwiseTrainConfig;

typedef struct {
    char schema[32];
    NNBitwiseOp op;
    uint8_t truth_mask;
    size_t arity;
    size_t input_width;
    size_t output_width;
    size_t epochs;
    size_t attempts;
    size_t sample_count;
    nn_real final_loss;
    nn_real min_margin;
    double accuracy;
    int learned_truth_table;
    int promoted;
} NNBitwiseTrainReport;

typedef struct {
    NNBitwiseOp op;
    uint8_t truth_mask;
    size_t arity;
    NeuralNet *net;
    uint8_t truth_table[4];
    uint8_t predicted_table[4];
    NNCertificate certificate;
    NNBitwiseTrainReport report;
} NNBitwiseGate;

typedef enum {
    NN_BITWISE_CHAIN_INPUT_X = 0,
    NN_BITWISE_CHAIN_INPUT_Y = 1,
    NN_BITWISE_CHAIN_CONST_ZERO = 2,
    NN_BITWISE_CHAIN_CONST_ONE = 3,
    NN_BITWISE_CHAIN_DERIVED = 4,
} NNBitwiseChainNodeKind;

typedef struct {
    NNBitwiseChainNodeKind kind;
    uint8_t truth_mask;
    uint8_t op_mask;
    size_t left_index;
    size_t right_index;
} NNBitwiseChainNode;

typedef struct {
    NNBitwiseChainNode nodes[NN_BITWISE_CHAIN_MAX_NODES];
    size_t node_count;
    size_t output_index;
    size_t step_count;
    uint8_t target_truth_mask;
    int solved;
    int used_symbolic_fallback;
} NNBitwiseChain;

typedef struct {
    size_t hidden_width;
    size_t epochs;
    nn_real learning_rate;
    unsigned int seed;
} NNBitwiseToolPolicyConfig;

typedef struct {
    NeuralNet *net;
    uint16_t allowed_truth_masks;
    size_t max_steps;
    size_t training_groups;
    size_t training_updates;
    double oracle_coverage;
    nn_real final_loss;
} NNBitwiseToolPolicy;

typedef struct {
    char schema[32];
    uint8_t truth_mask;
    NNBitwiseChain chain;
    NNCertificate certificate;
    size_t support;
    size_t step_count;
    int promoted;
    int used_symbolic_fallback;
} NNBitwiseMacroTool;

typedef struct {
    NNBitwiseMacroTool tools[NN_BITWISE_MACRO_MAX_TOOLS];
    size_t tool_count;
    size_t candidate_count;
    size_t promoted_count;
    size_t fallback_promoted_count;
    uint16_t promoted_truth_masks;
} NNBitwiseMacroLibrary;

int nn_bitgrid_init(NNBitGrid *grid, size_t width, size_t height);
void nn_bitgrid_free(NNBitGrid *grid);
size_t nn_bitgrid_cell_count(const NNBitGrid *grid);
int nn_bitgrid_same_shape(const NNBitGrid *a, const NNBitGrid *b);
int nn_bitgrid_get(const NNBitGrid *grid, size_t x, size_t y, uint8_t *out_bit);
int nn_bitgrid_set(NNBitGrid *grid, size_t x, size_t y, uint8_t bit);
int nn_bitgrid_fill(NNBitGrid *grid, uint8_t bit);
int nn_bitgrid_fill_alternating(NNBitGrid *grid, uint8_t first_bit);
int nn_bitgrid_copy(NNBitGrid *dst, const NNBitGrid *src);
size_t nn_bitgrid_hamming_distance(const NNBitGrid *a, const NNBitGrid *b);
int nn_bitgrid_write_pattern(const NNBitGrid *grid, char *buffer, size_t buffer_size);

const char *nn_bitwise_op_name(NNBitwiseOp op);
size_t nn_bitwise_op_arity(NNBitwiseOp op);
uint8_t nn_bitwise_op_truth_mask(NNBitwiseOp op);
const char *nn_bitwise_truth_mask_name(uint8_t truth_mask);
int nn_bitwise_op_eval(NNBitwiseOp op, uint8_t a, uint8_t b, uint8_t *out_bit);
int nn_bitwise_truth_mask_eval(uint8_t truth_mask, uint8_t a, uint8_t b, uint8_t *out_bit);

void nn_bitwise_train_config_init(NNBitwiseTrainConfig *config);
void nn_bitwise_gate_init(NNBitwiseGate *gate);
void nn_bitwise_gate_free(NNBitwiseGate *gate);
int nn_bitwise_gate_train(NNBitwiseGate *gate,
                          NNBitwiseOp op,
                          const NNBitwiseTrainConfig *config);
int nn_bitwise_gate_train_truth_table(NNBitwiseGate *gate,
                                      uint8_t truth_mask,
                                      const NNBitwiseTrainConfig *config);
int nn_bitwise_gate_predict_cell(const NNBitwiseGate *gate,
                                 uint8_t a,
                                 uint8_t b,
                                 uint8_t *out_bit,
                                 nn_real *confidence);
int nn_bitwise_gate_apply(const NNBitwiseGate *gate,
                          const NNBitGrid *a,
                          const NNBitGrid *b,
                          NNBitGrid *out);
int nn_bitwise_gate_write_json(FILE *fp, const NNBitwiseGate *gate);

uint8_t nn_bitwise_apply_truth_mask_to_masks(uint8_t op_mask,
                                             uint8_t left_mask,
                                             uint8_t right_mask);
uint16_t nn_bitwise_default_tool_mask(void);
void nn_bitwise_chain_init(NNBitwiseChain *chain);
void nn_bitwise_tool_policy_config_init(NNBitwiseToolPolicyConfig *config);
void nn_bitwise_tool_policy_init(NNBitwiseToolPolicy *policy);
void nn_bitwise_tool_policy_free(NNBitwiseToolPolicy *policy);
int nn_bitwise_tool_policy_train(NNBitwiseToolPolicy *policy,
                                 uint16_t allowed_truth_masks,
                                 size_t max_steps,
                                 const NNBitwiseToolPolicyConfig *config);
int nn_bitwise_chain_solve_truth_table(const NNBitwiseToolPolicy *policy,
                                       const NNBitwiseGate gates[NN_BITWISE_BINARY_TRUTH_TABLES],
                                       uint8_t target_truth_mask,
                                       NNBitwiseChain *chain);
int nn_bitwise_infer_grid_truth_constraints(const NNBitGrid *x,
                                            const NNBitGrid *y,
                                            const NNBitGrid *target,
                                            uint8_t *required_truth_mask,
                                            uint8_t *observed_rows_mask);
int nn_bitwise_chain_solve_grid_transform(
    const NNBitwiseToolPolicy *policy,
    const NNBitwiseGate gates[NN_BITWISE_BINARY_TRUTH_TABLES],
    const NNBitGrid *x,
    const NNBitGrid *y,
    const NNBitGrid *target,
    NNBitwiseChain *chain);
int nn_bitwise_chain_apply(const NNBitwiseChain *chain,
                           const NNBitwiseGate gates[NN_BITWISE_BINARY_TRUTH_TABLES],
                           const NNBitGrid *x,
                           const NNBitGrid *y,
                           NNBitGrid *out);
int nn_bitwise_chain_eval_truth_mask(
    const NNBitwiseChain *chain,
    const NNBitwiseGate gates[NN_BITWISE_BINARY_TRUTH_TABLES],
    uint8_t *out_truth_mask);
int nn_bitwise_chain_write(FILE *fp, const NNBitwiseChain *chain);
void nn_bitwise_macro_library_init(NNBitwiseMacroLibrary *library);
int nn_bitwise_macro_library_learn(
    NNBitwiseMacroLibrary *library,
    const NNBitwiseToolPolicy *policy,
    const NNBitwiseGate gates[NN_BITWISE_BINARY_TRUTH_TABLES],
    uint16_t target_truth_masks);
const NNBitwiseMacroTool *nn_bitwise_macro_library_find(const NNBitwiseMacroLibrary *library,
                                                        uint8_t truth_mask);
int nn_bitwise_macro_tool_apply(const NNBitwiseMacroTool *tool,
                                const NNBitwiseGate gates[NN_BITWISE_BINARY_TRUTH_TABLES],
                                const NNBitGrid *x,
                                const NNBitGrid *y,
                                NNBitGrid *out);
int nn_bitwise_macro_library_write_json(FILE *fp, const NNBitwiseMacroLibrary *library);

#endif
