static size_t arc_task_test_input_cells(const ARCTask *task)
{
    size_t cells = 0;
    if (task == NULL) {
        return 0;
    }
    for (size_t pair = 0; pair < task->test_count; ++pair) {
        cells += arc_grid_cells(&task->test[pair].input);
    }
    return cells;
}

static void arc_hypothesis_set_predicted_change_coverage(const ARCTask *task,
                                                         const ARCTransformCandidate *candidate,
                                                         NNProgramCoverage *coverage,
                                                         size_t program_index)
{
    ARCGrid predictions[ARC_MAX_PAIRS];
    size_t unsupported = 0;
    if (task == NULL || candidate == NULL || coverage == NULL ||
        !arc_apply_candidate_to_tests(task, candidate, predictions, &unsupported)) {
        return;
    }

    size_t event = 0;
    for (size_t pair = 0; pair < task->test_count; ++pair) {
        const ARCGrid *input = &task->test[pair].input;
        size_t cells = arc_grid_cells(input);
        if (!arc_grid_same_shape(&predictions[pair], input)) {
            for (size_t cell = 0; cell < cells; ++cell) {
                nn_program_coverage_set(coverage, program_index, event + cell);
            }
            event += cells;
            continue;
        }
        for (size_t cell = 0; cell < cells; ++cell) {
            if (predictions[pair].cells[cell] != input->cells[cell]) {
                nn_program_coverage_set(coverage, program_index, event);
            }
            ++event;
        }
    }
}

static size_t arc_hypothesis_registry_index_for_template_id(uint32_t template_id)
{
    size_t binding_count = 0;
    const ARCHypothesisTemplateBinding *bindings =
        arc_hypothesis_template_registry(&binding_count);
    for (size_t i = 0;
         i < binding_count && i < ARC_HYPOTHESIS_TEMPLATE_REGISTRY_CAPACITY;
         ++i) {
        if (bindings[i].template_id == template_id) {
            return i;
        }
    }
    return ARC_HYPOTHESIS_TEMPLATE_REGISTRY_CAPACITY;
}

static int arc_hypothesis_template_allowed(const ARCProgramLearnedState *learned,
                                           const ARCHypothesisTemplateBinding *binding)
{
    if (binding == NULL) {
        return 0;
    }
    if (learned == NULL || !learned->arc_visual_only) {
        return 1;
    }
    return strcmp(binding->name, "arc_sketch_program") == 0;
}

static int arc_candidate_is_visual_program_hypothesis(
    const ARCTransformCandidate *candidate)
{
    return candidate != NULL &&
           candidate->hypothesis_template_id ==
               (uint32_t)ARC_HYPOTHESIS_TEMPLATE_SKETCH_PROGRAM_ID;
}

static int arc_hypothesis_proposal_preferred(
    const ARCHypothesisProposal *proposal,
    size_t gain,
    const ARCHypothesisProposal *best,
    size_t best_gain,
    double best_score)
{
    if (proposal == NULL) {
        return 0;
    }
    if (best == NULL || gain > best_gain) {
        return 1;
    }
    if (gain < best_gain) {
        return 0;
    }
    int proposal_visual =
        arc_candidate_is_visual_program_hypothesis(&proposal->candidate);
    int best_visual = arc_candidate_is_visual_program_hypothesis(&best->candidate);
    if (proposal_visual != best_visual) {
        return proposal_visual;
    }
    if (proposal->score != best_score) {
        return proposal->score > best_score;
    }
    return strcmp(proposal->candidate.name, best->candidate.name) < 0;
}

static size_t arc_induce_exact_sketch_prior_hypotheses(
    const ARCTask *task,
    ARCProgramLearnedState *learned,
    const ARCTransformCandidate *existing_candidates,
    size_t existing_count,
    ARCHypothesisProposal *proposals,
    size_t proposal_capacity,
    size_t proposal_count,
    ARCHypothesisSelectionStats *stats)
{
    if (task == NULL || learned == NULL || proposals == NULL ||
        learned->exact_sketch_prior_count == 0) {
        return proposal_count;
    }
    const ARCHypothesisTemplateBinding *binding =
        arc_hypothesis_binding_for_template_id(
            (uint32_t)ARC_HYPOTHESIS_TEMPLATE_SKETCH_PROGRAM_ID);
    if (binding == NULL || binding->template_fn == NULL ||
        binding->to_candidate == NULL ||
        !arc_hypothesis_template_allowed(learned, binding)) {
        return proposal_count;
    }
    ARCProgramTaskFeatures task_features;
    arc_program_task_features(task, &task_features);
    size_t template_index = arc_hypothesis_registry_index_for_template_id(
        (uint32_t)ARC_HYPOTHESIS_TEMPLATE_SKETCH_PROGRAM_ID);
    for (size_t i = 0; i < learned->exact_sketch_prior_count; ++i) {
        const ARCExactSketchPrior *prior = &learned->exact_sketch_priors[i];
        if (prior->certified_count == 0 ||
            prior->seen_slot_mask != ARC_EXACT_SKETCH_PRIOR_COMPLETE_MASK ||
            strcmp(prior->task_id, task->id) != 0) {
            continue;
        }
        proposal_count =
            arc_try_add_sketch_program(binding,
                                       task,
                                       learned,
                                       existing_candidates,
                                       existing_count,
                                       proposals,
                                       proposal_capacity,
                                       proposal_count,
                                       stats,
                                       template_index,
                                       &task_features,
                                       prior->spec.selector,
                                       prior->spec.op,
                                       prior->spec.background,
                                       prior->spec.color,
                                       prior->spec.output_color,
                                       prior->spec.x_anchor,
                                       prior->spec.y_anchor,
                                       prior->spec.x_offset,
                                       prior->spec.y_offset,
                                       prior->spec.width,
                                       prior->spec.height,
                                       prior->spec.period);
    }
    return proposal_count;
}

static size_t arc_select_hypothesis_proposals(const ARCTask *task,
                                              const ARCHypothesisProposal *proposals,
                                              size_t proposal_count,
                                              ARCTransformCandidate *candidates,
                                              size_t capacity,
                                              size_t base_count,
                                              size_t top_k,
                                              int use_coverage_selection,
                                              int train_filter_enabled,
                                              ARCHypothesisSelectionStats *stats)
{
    if (task == NULL || proposals == NULL || candidates == NULL ||
        base_count >= capacity || proposal_count == 0) {
        return base_count;
    }

    ARCHypothesisProposal *valid =
        calloc(ARC_HYPOTHESIS_PROPOSAL_CAPACITY, sizeof(*valid));
    if (valid == NULL) {
        return base_count;
    }
    size_t valid_count = 0;
    for (size_t i = 0; i < proposal_count && valid_count < ARC_HYPOTHESIS_PROPOSAL_CAPACITY; ++i) {
        size_t checked_cells = 0;
        size_t mismatches = 0;
        size_t unsupported = 0;
        if (arc_candidate_name_exists(candidates, base_count, proposals[i].candidate.name)) {
            if (stats != NULL) {
                ++stats->proposals_duplicate_existing;
            }
            continue;
        }
        int train_consistent = arc_candidate_matches_train(task,
                                                           &proposals[i].candidate,
                                                           &checked_cells,
                                                           &mismatches,
                                                           &unsupported);
        if (!train_consistent) {
            if (stats != NULL) {
                ++stats->proposals_train_rejected;
            }
            if (train_filter_enabled) {
                continue;
            }
        }
        valid[valid_count] = proposals[i];
        valid[valid_count].hypothesis.evidence_support = checked_cells;
        valid[valid_count].hypothesis.evidence_residual_gain =
            checked_cells == 0 ? 0.0 : 1.0 - (double)mismatches / (double)checked_cells;
        ++valid_count;
        if (stats != NULL && train_consistent) {
            ++stats->proposals_train_consistent;
            size_t template_index =
                arc_hypothesis_registry_index_for_template_id(
                    proposals[i].candidate.hypothesis_template_id);
            if (template_index < ARC_HYPOTHESIS_TEMPLATE_REGISTRY_CAPACITY) {
                ++stats->template_train_consistent_retained[template_index];
            }
        }
    }
    if (valid_count == 0) {
        free(valid);
        return base_count;
    }

    size_t count = base_count;
    size_t task_cells = arc_task_test_input_cells(task);
    if (stats != NULL) {
        stats->coverage_events = task_cells;
    }
    NNProgramCoverage coverage;
    memset(&coverage, 0, sizeof(coverage));
    uint64_t *covered = NULL;
    int have_coverage = use_coverage_selection &&
                        task_cells > 0 &&
                        nn_program_coverage_init(&coverage, valid_count, task_cells);
    if (have_coverage && coverage.word_count != 0) {
        covered = calloc(coverage.word_count, sizeof(*covered));
        if (covered == NULL) {
            nn_program_coverage_free(&coverage);
            have_coverage = 0;
        }
    }
    if (have_coverage) {
        for (size_t i = 0; i < valid_count; ++i) {
            arc_hypothesis_set_predicted_change_coverage(task,
                                                         &valid[i].candidate,
                                                         &coverage,
                                                         i);
        }
        if (stats != NULL) {
            stats->coverage_used = 1;
        }
    }

    uint8_t selected[ARC_HYPOTHESIS_PROPOSAL_CAPACITY];
    memset(selected, 0, sizeof(selected));
    size_t selected_count = 0;
    if (have_coverage && covered != NULL) {
        for (size_t step = 0;
             step < top_k && count < capacity;
             ++step) {
            size_t best = (size_t)-1;
            size_t best_gain = 0;
            double best_score = -1.0e300;
            for (size_t i = 0; i < valid_count; ++i) {
                if (selected[i]) {
                    continue;
                }
                size_t gain = nn_program_coverage_gain(&coverage, i, covered);
                if (arc_hypothesis_proposal_preferred(&valid[i],
                                                      gain,
                                                      best == (size_t)-1
                                                          ? NULL
                                                          : &valid[best],
                                                      best_gain,
                                                      best_score)) {
                    best = i;
                    best_gain = gain;
                    best_score = valid[i].score;
                }
            }
            if (best == (size_t)-1 || best_gain == 0) {
                break;
            }
            selected[best] = 1;
            ++selected_count;
            nn_program_coverage_add_program_to_residual(&coverage, best, covered);
            if (!arc_candidate_name_exists(candidates, count, valid[best].candidate.name)) {
                arc_add_candidate(candidates, capacity, &count, &valid[best].candidate);
                if (stats != NULL) {
                    ++stats->proposals_selected;
                    ++stats->proposals_selected_by_coverage;
                    size_t template_index =
                        arc_hypothesis_registry_index_for_template_id(
                            valid[best].candidate.hypothesis_template_id);
                    if (template_index < ARC_HYPOTHESIS_TEMPLATE_REGISTRY_CAPACITY) {
                        ++stats->template_selected[template_index];
                    }
                }
            }
        }
    }

    if (selected_count == 0) {
        if (stats != NULL) {
            stats->coverage_fallback = 1;
        }
        size_t proposal_limit = valid_count < top_k
                                    ? valid_count
                                    : top_k;
        for (size_t i = 0; i < proposal_limit && count < capacity; ++i) {
            if (!arc_candidate_name_exists(candidates, count, valid[i].candidate.name)) {
                arc_add_candidate(candidates, capacity, &count, &valid[i].candidate);
                if (stats != NULL) {
                    ++stats->proposals_selected;
                    ++stats->proposals_selected_by_score_fallback;
                    size_t template_index =
                        arc_hypothesis_registry_index_for_template_id(
                            valid[i].candidate.hypothesis_template_id);
                    if (template_index < ARC_HYPOTHESIS_TEMPLATE_REGISTRY_CAPACITY) {
                        ++stats->template_selected[template_index];
                    }
                }
            }
        }
    }

    free(covered);
    if (have_coverage) {
        nn_program_coverage_free(&coverage);
    }
    free(valid);
    return count;
}

static size_t arc_induce_neural_slot_probe_hypotheses(
    const ARCTask *task,
    ARCProgramLearnedState *learned,
    const ARCTransformCandidate *existing_candidates,
    size_t existing_count,
    ARCHypothesisProposal *proposals,
    size_t proposal_capacity,
    size_t proposal_count,
    ARCHypothesisSelectionStats *stats)
{
    if (task == NULL || learned == NULL || existing_candidates == NULL ||
        proposals == NULL || proposal_capacity == 0 ||
        learned->hypothesis_slot_probe_row_count == 0) {
        return proposal_count;
    }
    ARCProgramTaskFeatures task_features;
    arc_program_task_features(task, &task_features);
    uint8_t consumed[ARC_HYPOTHESIS_SLOT_PROBE_CAPACITY];
    memset(consumed, 0, sizeof(consumed));
    for (size_t i = 0;
         i < learned->hypothesis_slot_probe_row_count &&
         i < ARC_HYPOTHESIS_SLOT_PROBE_CAPACITY;
         ++i) {
        if (consumed[i]) {
            continue;
        }
        const ARCNeuralSlotProbeRow *first =
            &learned->hypothesis_slot_probe_rows[i];
        const ARCHypothesisTemplateBinding *binding =
            arc_hypothesis_binding_for_template_id(first->template_id);
        if (binding == NULL || binding->template_fn == NULL ||
            binding->to_candidate == NULL) {
            consumed[i] = 1;
            continue;
        }
        NNHypothesis hypothesis;
        if (!nn_hypothesis_init(&hypothesis, binding->template_fn())) {
            consumed[i] = 1;
            continue;
        }
        for (size_t j = i;
             j < learned->hypothesis_slot_probe_row_count &&
             j < ARC_HYPOTHESIS_SLOT_PROBE_CAPACITY;
             ++j) {
            ARCNeuralSlotProbeRow *row = &learned->hypothesis_slot_probe_rows[j];
            if (row->template_id != first->template_id ||
                strcmp(row->library_id, first->library_id) != 0) {
                continue;
            }
            consumed[j] = 1;
            (void)nn_hypothesis_bind(&hypothesis,
                                     row->slot_name,
                                     row->slot_value,
                                     row->slot_integer);
        }
        if (!nn_hypothesis_is_fully_bound(&hypothesis)) {
            continue;
        }
        ARCTransformCandidate candidate;
        if (!binding->to_candidate(&hypothesis, &candidate)) {
            continue;
        }
        size_t template_index =
            arc_hypothesis_registry_index_for_template_id(first->template_id);
        proposal_count = arc_try_add_hypothesis_proposal(learned,
                                                         task,
                                                         existing_candidates,
                                                         existing_count,
                                                         proposals,
                                                         proposal_capacity,
                                                         proposal_count,
                                                         stats,
                                                         template_index,
                                                         &task_features,
                                                         &hypothesis,
                                                         &candidate,
                                                         NULL,
                                                         NULL);
    }
    return proposal_count;
}

static size_t arc_build_candidate_set(const ARCTask *task,
                                      ARCProgramLearnedState *learned,
                                      ARCTransformCandidate *candidates,
                                      size_t capacity,
                                      int include_hypothesis_induction,
                                      ARCHypothesisSelectionStats *hypothesis_stats)
{
    int visual_only = learned != NULL && learned->arc_visual_only;
    size_t count = visual_only ? 0 : arc_build_candidates(task, candidates, capacity);
    if (hypothesis_stats != NULL) {
        memset(hypothesis_stats, 0, sizeof(*hypothesis_stats));
        hypothesis_stats->proposal_capacity = ARC_HYPOTHESIS_PROPOSAL_CAPACITY;
        hypothesis_stats->hypothesis_prior_hidden =
            learned == NULL ? ARC_HYPOTHESIS_PRIOR_DEFAULT_HIDDEN
                            : learned->hypothesis_prior_hidden;
        hypothesis_stats->hypothesis_top_k =
            learned == NULL ? ARC_HYPOTHESIS_TOP_K : learned->hypothesis_top_k;
        hypothesis_stats->runtime_sketch_mine_cap =
            learned == NULL ? 0u : learned->arc_runtime_sketch_mine_cap;
        hypothesis_stats->runtime_generative_prior_rows =
            learned == NULL ? 0u : learned->generative_prior_count;
        hypothesis_stats->runtime_generative_prior_op_cap =
            learned == NULL ? 0u : learned->arc_generative_prior_op_cap;
        hypothesis_stats->hypothesis_neural_prior_enabled =
            learned == NULL ? 1 : learned->hypothesis_neural_prior_enabled;
        hypothesis_stats->hypothesis_slot_prior_enabled =
            learned == NULL ? 1 : learned->hypothesis_slot_prior_enabled;
        hypothesis_stats->hypothesis_cold_start_enabled =
            learned == NULL ? 1 : learned->hypothesis_cold_start_enabled;
        hypothesis_stats->hypothesis_structural_score_enabled =
            learned == NULL ? 1 : learned->hypothesis_structural_score_enabled;
        hypothesis_stats->hypothesis_coverage_selection_enabled =
            learned == NULL ? 1 : learned->hypothesis_coverage_selection_enabled;
        hypothesis_stats->hypothesis_train_filter_enabled =
            learned == NULL ? 1 : learned->hypothesis_train_filter_enabled;
        hypothesis_stats->hypothesis_train_consistency_prior_enabled =
            learned == NULL ? 1 : learned->hypothesis_train_consistency_prior_enabled;
        hypothesis_stats->arc_visual_only = visual_only;
        if (learned != NULL) {
            hypothesis_stats->library_prior_entries = learned->hypothesis_library_entries;
            hypothesis_stats->library_prior_template_rows =
                learned->hypothesis_library_template_rows;
            hypothesis_stats->library_prior_slot_rows = learned->hypothesis_library_slot_rows;
            hypothesis_stats->library_prior_seeded_proposals =
                learned->hypothesis_library_seeded_proposals;
            hypothesis_stats->library_prior_seeded_certified =
                learned->hypothesis_library_seeded_certified;
            hypothesis_stats->template_probe_rows =
                learned->hypothesis_probe_template_rows;
            hypothesis_stats->template_probe_seeded_proposals =
                learned->hypothesis_probe_seeded_proposals;
            hypothesis_stats->slot_probe_rows =
                learned->hypothesis_probe_slot_rows;
            hypothesis_stats->slot_probe_seeded_proposals =
                learned->hypothesis_probe_slot_seeded_proposals;
        }
    }
    if (include_hypothesis_induction) {
        ARCHypothesisProposal *proposals =
            calloc(ARC_HYPOTHESIS_PROPOSAL_CAPACITY, sizeof(*proposals));
        if (proposals == NULL) {
            return count;
        }
        size_t proposal_count = 0;
        size_t binding_count = 0;
        const ARCHypothesisTemplateBinding *bindings =
            arc_hypothesis_template_registry(&binding_count);
        if (hypothesis_stats != NULL) {
            hypothesis_stats->registered_templates = binding_count;
            if (learned != NULL) {
                for (size_t i = 0;
                     i < binding_count && i < ARC_HYPOTHESIS_TEMPLATE_REGISTRY_CAPACITY;
                     ++i) {
                    size_t proposals = 0;
                    size_t certified = 0;
                    arc_hypothesis_template_prior_counts(learned,
                                                         bindings[i].template_id,
                                                         &proposals,
                                                         &certified);
                    hypothesis_stats->template_prior_proposals[i] = proposals;
                    hypothesis_stats->template_prior_certified[i] = certified;
                    hypothesis_stats->template_prior_score[i] =
                        (double)ARC_HYPOTHESIS_TEMPLATE_PRIOR_WEIGHT *
                        arc_hypothesis_template_log_odds_for_id(learned,
                                                                bindings[i].template_id,
                                                                NULL);
                    hypothesis_stats->template_probe_proposals[i] =
                        arc_hypothesis_template_probe_count(learned,
                                                            bindings[i].template_id);
                    hypothesis_stats->template_probe_score[i] =
                        (double)ARC_HYPOTHESIS_TEMPLATE_PROBE_WEIGHT *
                        log1p((double)hypothesis_stats->template_probe_proposals[i]);
                    hypothesis_stats->slot_probe_score[i] =
                        arc_hypothesis_slot_probe_score_for_template(
                            learned,
                            bindings[i].template_id);
                }
            }
        }
        for (size_t i = 0; i < binding_count; ++i) {
            if (bindings[i].induce == NULL ||
                !arc_hypothesis_template_allowed(learned, &bindings[i])) {
                continue;
            }
            proposal_count = bindings[i].induce(&bindings[i],
                                                task,
                                                learned,
                                                candidates,
                                                count,
                                                proposals,
                                                ARC_HYPOTHESIS_PROPOSAL_CAPACITY,
                                                proposal_count,
                                                hypothesis_stats,
                                                i);
        }
        proposal_count = arc_induce_exact_sketch_prior_hypotheses(task,
                                                                  learned,
                                                                  candidates,
                                                                  count,
                                                                  proposals,
                                                                  ARC_HYPOTHESIS_PROPOSAL_CAPACITY,
                                                                  proposal_count,
                                                                  hypothesis_stats);
        proposal_count = arc_induce_neural_slot_probe_hypotheses(task,
                                                                 learned,
                                                                 candidates,
                                                                 count,
                                                                 proposals,
                                                                 ARC_HYPOTHESIS_PROPOSAL_CAPACITY,
                                                                 proposal_count,
                                                                 hypothesis_stats);
        if (hypothesis_stats != NULL) {
            memset(hypothesis_stats->template_retained,
                   0,
                   sizeof(hypothesis_stats->template_retained));
            for (size_t proposal_index = 0;
                 proposal_index < proposal_count;
                 ++proposal_index) {
                size_t template_index =
                    arc_hypothesis_registry_index_for_template_id(
                        proposals[proposal_index].candidate.hypothesis_template_id);
                if (template_index < ARC_HYPOTHESIS_TEMPLATE_REGISTRY_CAPACITY) {
                    ++hypothesis_stats->template_retained[template_index];
                }
            }
            hypothesis_stats->proposals_retained = proposal_count;
            hypothesis_stats->proposal_cap_hit =
                proposal_count >= ARC_HYPOTHESIS_PROPOSAL_CAPACITY;
        }
        count = arc_select_hypothesis_proposals(task,
                                                proposals,
                                                proposal_count,
                                                candidates,
                                                capacity,
                                                count,
                                                learned == NULL ? ARC_HYPOTHESIS_TOP_K
                                                                : learned->hypothesis_top_k,
                                                learned == NULL
                                                    ? 1
                                                    : learned->hypothesis_coverage_selection_enabled,
                                                learned == NULL
                                                    ? 1
                                                    : learned->hypothesis_train_filter_enabled,
                                                hypothesis_stats);
        free(proposals);
    }
    return visual_only
               ? count
               : arc_add_learned_compositions(task, learned, candidates, capacity, count);
}

static void arc_program_aggregate_init(ARCProgramProposerAggregate *aggregate,
                                       const char *task_name,
                                       ARCProgramStrategy strategy,
                                       size_t budget)
{
    memset(aggregate, 0, sizeof(*aggregate));
    aggregate->summary.domain = "arc_agi";
    aggregate->summary.task = task_name;
    aggregate->summary.strategy = arc_program_strategy_to_nn(strategy);
    aggregate->summary.budget = budget;
    aggregate->program_set_hash = UINT64_C(1469598103934665603);
}

static void arc_program_aggregate_task(ARCProgramProposerAggregate *aggregate,
                                       const ARCTask *task,
                                       const ARCTransformCandidate *candidates,
                                       const ARCProgramEval *evals,
                                       size_t candidate_count,
                                       ARCProgramStrategy strategy,
                                       ARCProgramLearnedState *learned,
                                       ARCProposalTracePrior *trace_prior,
                                       size_t budget)
{
    ARCProgramRank ranks[ARC_RULE_CANDIDATE_COUNT];
    ARCProgramTaskFeatures task_features;
    size_t limit = candidate_count < budget ? candidate_count : budget;
    size_t task_cells = arc_task_test_cells(task);
    int have_prediction = 0;
    int unique_prediction = 1;
    uint64_t prediction_hash = 0;
    size_t prediction_correct_cells = 0;
    NNProgramCoverage coverage;
    uint64_t *covered = NULL;
    int have_coverage = nn_program_coverage_init(&coverage, candidate_count, task_cells);
    if (have_coverage && coverage.word_count != 0) {
        covered = calloc(coverage.word_count, sizeof(*covered));
        if (covered == NULL) {
            nn_program_coverage_free(&coverage);
            have_coverage = 0;
        }
    }
    if (have_coverage) {
        for (size_t i = 0; i < candidate_count; ++i) {
            for (size_t event = 0; event < evals[i].test_correct_cells; ++event) {
                nn_program_coverage_set(&coverage, i, event);
            }
        }
    }

    arc_program_task_features(task, &task_features);
    arc_build_program_ranks(task,
                            candidates,
                            evals,
                            candidate_count,
                            budget,
                            strategy,
                            learned,
                            trace_prior,
                            &task_features,
                            ranks);
    aggregate->summary.event_count += task_cells;
    for (size_t rank = 0; rank < limit; ++rank) {
        size_t index = ranks[rank].candidate_index;
        ++aggregate->summary.proposed_programs;
        aggregate->program_set_hash =
            arc_hash_mix(aggregate->program_set_hash, evals[index].program_hash);
        if (!evals[index].train_consistent) {
            continue;
        }
        ++aggregate->summary.certified_programs;
        if (!evals[index].test_supported) {
            continue;
        }
        ++aggregate->summary.accepted_programs;
        if (!have_prediction) {
            prediction_hash = evals[index].prediction_hash;
            prediction_correct_cells =
                have_coverage && covered != NULL
                    ? nn_program_coverage_gain(&coverage, index, covered)
                    : evals[index].test_correct_cells;
            have_prediction = 1;
        } else if (prediction_hash != evals[index].prediction_hash) {
            unique_prediction = 0;
        }
    }

    if (have_prediction && unique_prediction) {
        ++aggregate->unique_prediction_tasks;
        aggregate->summary.covered_events += prediction_correct_cells;
        if (prediction_correct_cells == task_cells && task_cells > 0) {
            ++aggregate->exact_tasks;
        } else if (task_cells > prediction_correct_cells) {
            aggregate->summary.false_rejections += task_cells - prediction_correct_cells;
        }
    }
    if (aggregate->summary.event_count != 0) {
        aggregate->summary.utility =
            (double)aggregate->summary.covered_events /
            (double)aggregate->summary.event_count;
    }
    snprintf(aggregate->program_set_hash_text,
             sizeof(aggregate->program_set_hash_text),
             "%016llx",
             (unsigned long long)aggregate->program_set_hash);
    aggregate->summary.program_set_hash = aggregate->program_set_hash_text;
    free(covered);
    if (have_coverage) {
        nn_program_coverage_free(&coverage);
    }
}

static int arc_program_learned_observe(ARCProgramLearnedState *state,
                                       const ARCTask *task,
                                       const ARCTransformCandidate *candidates,
                                       const ARCProgramEval *evals,
                                       size_t candidate_count,
                                       nn_real learning_rate)
{
    size_t family_seen[32];
    size_t family_success[32];
    size_t relation_seen[ARC_OBJECT_RELATION_KIND_COUNT];
    size_t relation_success[ARC_OBJECT_RELATION_KIND_COUNT];
    size_t relation_color_seen[ARC_OBJECT_RELATION_KIND_COUNT][ARC_COLOR_COUNT][ARC_COLOR_COUNT];
    size_t relation_color_success[ARC_OBJECT_RELATION_KIND_COUNT][ARC_COLOR_COUNT][ARC_COLOR_COUNT];
    size_t relation_background_seen[ARC_OBJECT_RELATION_KIND_COUNT][ARC_COLOR_COUNT];
    size_t relation_background_success[ARC_OBJECT_RELATION_KIND_COUNT][ARC_COLOR_COUNT];
    size_t relation_selector_seen[ARC_OBJECT_RELATION_KIND_COUNT][ARC_OBJECT_SELECTOR_KIND_COUNT];
    size_t relation_selector_success[ARC_OBJECT_RELATION_KIND_COUNT][ARC_OBJECT_SELECTOR_KIND_COUNT];
    size_t relation_output_selector_seen[ARC_OBJECT_RELATION_KIND_COUNT][ARC_OBJECT_SELECTOR_KIND_COUNT];
    size_t relation_output_selector_success[ARC_OBJECT_RELATION_KIND_COUNT][ARC_OBJECT_SELECTOR_KIND_COUNT];
    size_t relation_render_seen[ARC_OBJECT_RELATION_KIND_COUNT][ARC_OBJECT_RENDER_KIND_COUNT];
    size_t relation_render_success[ARC_OBJECT_RELATION_KIND_COUNT][ARC_OBJECT_RENDER_KIND_COUNT];
    size_t object_selector_seen[ARC_OBJECT_SELECTOR_KIND_COUNT];
    size_t object_selector_success[ARC_OBJECT_SELECTOR_KIND_COUNT];
    size_t object_color_seen[ARC_COLOR_COUNT][ARC_COLOR_COUNT];
    size_t object_color_success[ARC_COLOR_COUNT][ARC_COLOR_COUNT];
    size_t object_render_seen[ARC_OBJECT_RENDER_KIND_COUNT];
    size_t object_render_success[ARC_OBJECT_RENDER_KIND_COUNT];
    memset(family_seen, 0, sizeof(family_seen));
    memset(family_success, 0, sizeof(family_success));
    memset(relation_seen, 0, sizeof(relation_seen));
    memset(relation_success, 0, sizeof(relation_success));
    memset(relation_color_seen, 0, sizeof(relation_color_seen));
    memset(relation_color_success, 0, sizeof(relation_color_success));
    memset(relation_background_seen, 0, sizeof(relation_background_seen));
    memset(relation_background_success, 0, sizeof(relation_background_success));
    memset(relation_selector_seen, 0, sizeof(relation_selector_seen));
    memset(relation_selector_success, 0, sizeof(relation_selector_success));
    memset(relation_output_selector_seen, 0, sizeof(relation_output_selector_seen));
    memset(relation_output_selector_success, 0, sizeof(relation_output_selector_success));
    memset(relation_render_seen, 0, sizeof(relation_render_seen));
    memset(relation_render_success, 0, sizeof(relation_render_success));
    memset(object_selector_seen, 0, sizeof(object_selector_seen));
    memset(object_selector_success, 0, sizeof(object_selector_success));
    memset(object_color_seen, 0, sizeof(object_color_seen));
    memset(object_color_success, 0, sizeof(object_color_success));
    memset(object_render_seen, 0, sizeof(object_render_seen));
    memset(object_render_success, 0, sizeof(object_render_success));
    for (size_t i = 0; i < candidate_count; ++i) {
        if (arc_candidate_is_composed(&candidates[i])) {
            continue;
        }
        size_t family = arc_candidate_family(&candidates[i]);
        if (family >= sizeof(family_seen) / sizeof(family_seen[0])) {
            continue;
        }
        ++family_seen[family];
        if (evals[i].train_consistent) {
            ++family_success[family];
        }
        size_t relation_kind = arc_object_relation_kind(&candidates[i]);
        if (relation_kind != 0 &&
            relation_kind < ARC_OBJECT_RELATION_KIND_COUNT &&
            candidates[i].input_color < ARC_COLOR_COUNT &&
            candidates[i].output_color < ARC_COLOR_COUNT) {
            uint8_t input_color = candidates[i].input_color;
            uint8_t output_color = candidates[i].output_color;
            uint8_t background = candidates[i].background_color;
            uint8_t selector = arc_candidate_selector_kind(&candidates[i]);
            uint8_t output_selector = arc_candidate_output_selector_kind(&candidates[i]);
            uint8_t render_mode = arc_candidate_render_mode(&candidates[i]);
            ++relation_seen[relation_kind];
            ++relation_color_seen[relation_kind][input_color][output_color];
            ++relation_background_seen[relation_kind][background];
            ++relation_selector_seen[relation_kind][selector];
            ++relation_output_selector_seen[relation_kind][output_selector];
            ++relation_render_seen[relation_kind][render_mode];
            if (evals[i].train_consistent) {
                ++relation_success[relation_kind];
                ++relation_color_success[relation_kind][input_color][output_color];
                ++relation_background_success[relation_kind][background];
                ++relation_selector_success[relation_kind][selector];
                ++relation_output_selector_success[relation_kind][output_selector];
                ++relation_render_success[relation_kind][render_mode];
            }
        }
        if (arc_candidate_is_selected_object_program(&candidates[i]) &&
            candidates[i].input_color < ARC_COLOR_COUNT &&
            candidates[i].output_color < ARC_COLOR_COUNT) {
            uint8_t selector = arc_candidate_selector_kind(&candidates[i]);
            uint8_t render_mode = arc_candidate_render_mode(&candidates[i]);
            ++object_selector_seen[selector];
            ++object_color_seen[candidates[i].input_color][candidates[i].output_color];
            ++object_render_seen[render_mode];
            if (evals[i].train_consistent) {
                ++object_selector_success[selector];
                ++object_color_success[candidates[i].input_color][candidates[i].output_color];
                ++object_render_success[render_mode];
            }
        }
    }
    for (size_t family = 0; family < sizeof(family_seen) / sizeof(family_seen[0]); ++family) {
        if (family_seen[family] == 0) {
            continue;
        }
        double observation = (double)family_success[family] / (double)family_seen[family];
        state->family_score[family] = 0.85 * state->family_score[family] + 0.15 * observation;
    }
    for (size_t relation_kind = 1; relation_kind < ARC_OBJECT_RELATION_KIND_COUNT; ++relation_kind) {
        if (relation_seen[relation_kind] != 0) {
            double observation =
                (double)relation_success[relation_kind] / (double)relation_seen[relation_kind];
            state->relation_score[relation_kind] =
                0.85 * state->relation_score[relation_kind] + 0.15 * observation;
        }
        for (uint8_t input_color = 0; input_color < ARC_COLOR_COUNT; ++input_color) {
            for (uint8_t output_color = 0; output_color < ARC_COLOR_COUNT; ++output_color) {
                if (relation_color_seen[relation_kind][input_color][output_color] == 0) {
                    continue;
                }
                double observation =
                    (double)relation_color_success[relation_kind][input_color][output_color] /
                    (double)relation_color_seen[relation_kind][input_color][output_color];
                state->relation_color_score[relation_kind][input_color][output_color] =
                    0.85 * state->relation_color_score[relation_kind][input_color][output_color] +
                    0.15 * observation;
            }
        }
        for (uint8_t background = 0; background < ARC_COLOR_COUNT; ++background) {
            if (relation_background_seen[relation_kind][background] == 0) {
                continue;
            }
            double observation =
                (double)relation_background_success[relation_kind][background] /
                (double)relation_background_seen[relation_kind][background];
            state->relation_background_score[relation_kind][background] =
                0.85 * state->relation_background_score[relation_kind][background] +
                0.15 * observation;
        }
        for (uint8_t selector = 0; selector < ARC_OBJECT_SELECTOR_KIND_COUNT; ++selector) {
            if (relation_selector_seen[relation_kind][selector] == 0) {
                continue;
            }
            double observation =
                (double)relation_selector_success[relation_kind][selector] /
                (double)relation_selector_seen[relation_kind][selector];
            state->relation_selector_score[relation_kind][selector] =
                0.85 * state->relation_selector_score[relation_kind][selector] +
                0.15 * observation;
        }
        for (uint8_t selector = 0; selector < ARC_OBJECT_SELECTOR_KIND_COUNT; ++selector) {
            if (relation_output_selector_seen[relation_kind][selector] == 0) {
                continue;
            }
            double observation =
                (double)relation_output_selector_success[relation_kind][selector] /
                (double)relation_output_selector_seen[relation_kind][selector];
            state->relation_output_selector_score[relation_kind][selector] =
                0.85 * state->relation_output_selector_score[relation_kind][selector] +
                0.15 * observation;
        }
        for (uint8_t render_mode = 0; render_mode < ARC_OBJECT_RENDER_KIND_COUNT; ++render_mode) {
            if (relation_render_seen[relation_kind][render_mode] == 0) {
                continue;
            }
            double observation =
                (double)relation_render_success[relation_kind][render_mode] /
                (double)relation_render_seen[relation_kind][render_mode];
            state->relation_render_score[relation_kind][render_mode] =
                0.85 * state->relation_render_score[relation_kind][render_mode] +
                0.15 * observation;
        }
    }
    for (uint8_t selector = 0; selector < ARC_OBJECT_SELECTOR_KIND_COUNT; ++selector) {
        if (object_selector_seen[selector] == 0) {
            continue;
        }
        double observation =
            (double)object_selector_success[selector] / (double)object_selector_seen[selector];
        state->object_selector_score[selector] =
            0.85 * state->object_selector_score[selector] + 0.15 * observation;
    }
    for (uint8_t input_color = 0; input_color < ARC_COLOR_COUNT; ++input_color) {
        for (uint8_t output_color = 0; output_color < ARC_COLOR_COUNT; ++output_color) {
            if (object_color_seen[input_color][output_color] == 0) {
                continue;
            }
            double observation =
                (double)object_color_success[input_color][output_color] /
                (double)object_color_seen[input_color][output_color];
            state->object_color_score[input_color][output_color] =
                0.85 * state->object_color_score[input_color][output_color] +
                0.15 * observation;
        }
    }
    for (uint8_t render_mode = 0; render_mode < ARC_OBJECT_RENDER_KIND_COUNT; ++render_mode) {
        if (object_render_seen[render_mode] == 0) {
            continue;
        }
        double observation =
            (double)object_render_success[render_mode] / (double)object_render_seen[render_mode];
        state->object_render_score[render_mode] =
            0.85 * state->object_render_score[render_mode] + 0.15 * observation;
    }
    return arc_program_learned_train_hypothesis_prior(state,
                                                      candidates,
                                                      evals,
                                                      candidate_count,
                                                      learning_rate) &&
           arc_program_learned_train_task(state,
                                          task,
                                          candidates,
                                          evals,
                                          candidate_count,
                                          learning_rate);
}

static void arc_program_print_aggregate(const ARCProgramProposerAggregate *aggregate)
{
    nn_program_write_proposer_summary_json(stdout, &aggregate->summary);
}

static size_t arc_program_budget_list(const ARCConfig *config,
                                      size_t *budgets,
                                      size_t capacity)
{
    if (config != NULL && config->program_budget_grid) {
        const size_t *grid = config->program_budget_grid == 2
                                 ? ARC_PROGRAM_BUDGET_TIGHT_GRID
                                 : ARC_PROGRAM_BUDGET_GRID;
        size_t count = ARC_PROGRAM_BUDGET_GRID_COUNT < capacity
                           ? ARC_PROGRAM_BUDGET_GRID_COUNT
                           : capacity;
        for (size_t i = 0; i < count; ++i) {
            budgets[i] = grid[i];
        }
        return count;
    }
    if (capacity == 0) {
        return 0;
    }
    budgets[0] = config == NULL ? ARC_PROGRAM_PROPOSER_BUDGET : config->program_budget;
    return 1;
}

/* Internal ARC hypothesis template and program-summary helpers for arc_agi.c. */
static void arc_color_map_slot_name(uint8_t color, char *name, size_t name_size)
{
    snprintf(name, name_size, "map.%u", (unsigned)color);
}

static const NNHypothesisTemplate *arc_hypothesis_template_same_grid_color_map(void)
{
    static NNHypothesisTemplate template_ref;
    static int initialized = 0;
    if (initialized) {
        return &template_ref;
    }
    nn_hypothesis_template_init(&template_ref,
                                "arc_agi",
                                "same_grid_color_map",
                                1035,
                                NN_PROGRAM_VALUE_GRID,
                                NN_PROGRAM_VALUE_GRID);
    nn_hypothesis_template_add_stage(&template_ref,
                                     "infer_color_map",
                                     NN_PROGRAM_KIND_TRANSFORM,
                                     NN_PROGRAM_VALUE_GRID,
                                     NN_PROGRAM_VALUE_GRID,
                                     200);
    for (uint8_t color = 0; color < ARC_COLOR_COUNT; ++color) {
        char slot_name[16];
        arc_color_map_slot_name(color, slot_name, sizeof(slot_name));
        nn_hypothesis_template_add_slot(&template_ref,
                                        slot_name,
                                        NN_HYPOTHESIS_SLOT_COLOR,
                                        1,
                                        (double)color,
                                        color);
    }
    initialized = 1;
    return &template_ref;
}
