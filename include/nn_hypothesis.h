#ifndef NN_HYPOTHESIS_H
#define NN_HYPOTHESIS_H

#include "nn_core.h"
#include "nn_experiment.h"
#include "nn_program.h"
#include "nn_symbolic.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

/*
 * Hypothesis layer.
 *
 * A hypothesis is a typed operator skeleton (a "template") plus a concrete
 * slot binding inferred from observed task evidence. Compiling a hypothesis
 * produces an NNProgram (the canonical row format used by the verifier and
 * the existing closed-loop policy). The hypothesis layer is intentionally
 * domain-neutral so that ARC, program_synthesis, and prime_gap can all use
 * the same template + slot vocabulary.
 *
 * Provenance contract:
 *   - Slot values bound from example state (colors, offsets, selectors,
 *     reference objects) are NN_FEATURE_STATE_OBSERVATION or
 *     NN_FEATURE_STRUCTURAL_PRIOR.
 *   - Evidence derived from the verifier (certified support, residual gain)
 *     is NN_FEATURE_VERIFIER_DERIVED or NN_FEATURE_DERIVED_FROM_GATE and
 *     must never appear in the hypothesis-state feature vector that a
 *     selection-mode proposer scores against.
 */

#define NN_HYPOTHESIS_MAX_NAME 64u
#define NN_HYPOTHESIS_MAX_DOMAIN 32u
#define NN_HYPOTHESIS_MAX_STAGES 6u
#define NN_HYPOTHESIS_MAX_SLOTS 16u
#define NN_HYPOTHESIS_MAX_SLOT_NAME 32u
#define NN_HYPOTHESIS_MAX_STAGE_NAME 48u
#define NN_HYPOTHESIS_MAX_CANONICAL_ID 224u
#define NN_HYPOTHESIS_FEATURE_COUNT 24u
#define NN_HYPOTHESIS_FEATURE_ENCODER_VERSION 1u

typedef enum {
    NN_HYPOTHESIS_SLOT_INTEGER = 0,
    NN_HYPOTHESIS_SLOT_COLOR = 1,
    NN_HYPOTHESIS_SLOT_OFFSET = 2,
    NN_HYPOTHESIS_SLOT_SELECTOR_KIND = 3,
    NN_HYPOTHESIS_SLOT_RELATION_KIND = 4,
    NN_HYPOTHESIS_SLOT_RENDER_MODE = 5,
    NN_HYPOTHESIS_SLOT_REFERENCE_COLOR = 6,
    NN_HYPOTHESIS_SLOT_SCALAR = 7,
    NN_HYPOTHESIS_SLOT_BOOLEAN = 8,
    NN_HYPOTHESIS_SLOT_CUSTOM = 255,
} NNHypothesisSlotType;

typedef struct {
    char name[NN_HYPOTHESIS_MAX_SLOT_NAME];
    NNHypothesisSlotType type;
    int required;
    double default_value;
    int64_t default_integer;
} NNHypothesisSlotSpec;

typedef struct {
    char name[NN_HYPOTHESIS_MAX_STAGE_NAME];
    NNProgramKind kind;
    NNProgramValueType input_type;
    NNProgramValueType output_type;
    uint32_t template_id;
} NNHypothesisStage;

typedef struct {
    char domain[NN_HYPOTHESIS_MAX_DOMAIN];
    char name[NN_HYPOTHESIS_MAX_NAME];
    uint32_t template_id;
    NNProgramValueType input_type;
    NNProgramValueType output_type;
    NNHypothesisStage stages[NN_HYPOTHESIS_MAX_STAGES];
    size_t stage_count;
    NNHypothesisSlotSpec slots[NN_HYPOTHESIS_MAX_SLOTS];
    size_t slot_count;
} NNHypothesisTemplate;

typedef struct {
    char name[NN_HYPOTHESIS_MAX_SLOT_NAME];
    NNHypothesisSlotType type;
    double value;
    int64_t integer;
    int bound;
} NNHypothesisSlotValue;

typedef struct {
    const NNHypothesisTemplate *template_ref;
    NNHypothesisSlotValue slots[NN_HYPOTHESIS_MAX_SLOTS];
    size_t slot_count;
    char canonical_id[NN_HYPOTHESIS_MAX_CANONICAL_ID];
    uint64_t canonical_hash;
    size_t evidence_support;
    double evidence_residual_gain;
} NNHypothesis;

typedef struct {
    const char *task;
    const char *task_instance;
    const NNProgram *program;
    NNCertificateStatus certificate_status;
    const char *proof_kind;
    const char *reason;
    size_t event_count;
    size_t covered_events;
    size_t support;
    size_t false_rejections;
    double utility;
} NNHypothesisEvidence;

const char *nn_hypothesis_slot_type_name(NNHypothesisSlotType type);

int nn_hypothesis_template_is_valid(const NNHypothesisTemplate *template_ref);

void nn_hypothesis_template_init(NNHypothesisTemplate *template_ref,
                                 const char *domain,
                                 const char *name,
                                 uint32_t template_id,
                                 NNProgramValueType input_type,
                                 NNProgramValueType output_type);
int nn_hypothesis_template_add_stage(NNHypothesisTemplate *template_ref,
                                     const char *stage_name,
                                     NNProgramKind kind,
                                     NNProgramValueType input_type,
                                     NNProgramValueType output_type,
                                     uint32_t stage_template_id);
int nn_hypothesis_template_add_slot(NNHypothesisTemplate *template_ref,
                                    const char *slot_name,
                                    NNHypothesisSlotType type,
                                    int required,
                                    double default_value,
                                    int64_t default_integer);

void nn_hypothesis_clear(NNHypothesis *hypothesis);
int nn_hypothesis_init(NNHypothesis *hypothesis,
                       const NNHypothesisTemplate *template_ref);
int nn_hypothesis_bind(NNHypothesis *hypothesis,
                       const char *slot_name,
                       double value,
                       int64_t integer);
int nn_hypothesis_get(const NNHypothesis *hypothesis,
                      const char *slot_name,
                      double *value_out,
                      int64_t *integer_out);
int nn_hypothesis_is_fully_bound(const NNHypothesis *hypothesis);

uint64_t nn_hypothesis_refresh_canonical_id(NNHypothesis *hypothesis);

const NNFeatureSchema *nn_hypothesis_feature_schema(size_t *feature_count_out);
int nn_hypothesis_encode_features(const NNHypothesis *hypothesis,
                                  nn_real *features,
                                  size_t capacity,
                                  size_t *feature_count_out);
NNFeatureAudit nn_hypothesis_feature_audit(void);

int nn_hypothesis_compile(const NNHypothesis *hypothesis,
                          NNProgram *program_out);
int nn_hypothesis_write_evidence_json(FILE *fp,
                                      const NNHypothesis *hypothesis,
                                      const NNHypothesisEvidence *evidence);

#endif
