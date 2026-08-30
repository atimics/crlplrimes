typedef struct {
    const char *domain;
    const char *domain_display;
    const char *data_domain;
    const char *grounding_authority;
    const char *authority_claim_scope;
    const char *grammar;
    const char *candidate_set;
    const char *verifier;
    const char *certificate_gate;
    const char *scorer_role;
    const char *neural_scorer_role;
    const char *neural_proposer_role;
    const char *headline_baseline;
    const char *trace_artifact;
    const char *examples[4];
    size_t example_count;
    const char *result_table;
    const char *result;
    const char *claim;
    const char *takeaway;
    const char *limitation;
    const char *claim_ids[4];
    size_t claim_id_count;
} PaperExperimentMatrixRow;

static const PaperExperimentMatrixRow PAPER_EXPERIMENT_MATRIX_ROWS[] = {
    {
        "prime_gap",
        "arithmetic sequence filtering",
        "next-prime gaps over segmented-sieve windows; broader family includes period-residue sequence filters",
        "exact_check",
        "zero-false-rejection gates and exact residual ranking inside the next-prime support",
        "local modular filters over residues, gap classes, and wheel support",
        "next-prime gap actions plus certified modular filters",
        "segmented-sieve replay and modular certificate checks",
        "zero-false-rejection divisibility certificates for the audited next-prime instance",
        "rank residual gaps and budgeted certified filters",
        "score residual actions and certified filters after the gate defines the target",
        "static proposer is a negative baseline; closed-loop and unified proposer rows must be compared to grammar order",
        "scorer baseline is the certified gate; proposer baseline is grammar order plus verifier-feedback variants",
        "crlp.compare.v1; crlp.rule_trace.v1; crlp.rule_search.summary.v1",
        {"modular reject rule",
         "wheel support certificate",
         "next-prime gap action",
         "Pisano-period recurrence gate"},
        4U,
        "expanded certified filter 68.87% top-1 vs 45.12% hand-wheel and 9.13% raw with the ranker fixed; five-seed rank invariance audited",
        "expanded certified filter 68.87% top-1 vs 45.12% hand-wheel and 9.13% raw with the ranker fixed; five-seed rank invariance audited",
        "certified modular filters shrink the residual action space",
        "certified modular structure is the candidate mechanism; the neural scorer ranks residual budget choices while Fibonacci/Lucas-style filters remain boundary evidence.",
        "static neural proposer is not superior to deterministic baselines at fixed budget",
        {"C1.prime_gate_delta", "C1.prime_gate_delta_ci", "C2.prime_proposer_fixed_budget", NULL},
        3U,
    },
    {
        "grid_path",
        "grid path",
        "grid mazes with deterministic shortest-path labels",
        "exact_check",
        "shortest-path successor and certificate claims inside generated grids",
        "local moves plus boundary, trap, path-clear, and distance predicates",
        "four moves plus symbolic local/topology/bound predicates",
        "BFS shortest-path successor and rule certificate checks",
        "legality, trap, clear-path, and distance-dominance certificates",
        "rank legal residual moves after symbolic constraints",
        "rank legal residual moves under exact shortest-path labels",
        "budgeted rule-order diagnostics only",
        "BFS successor and cheap symbolic gates",
        "crlp.compare.v1; crlp.symbolic_domain.trace.v1",
        {"legal move", "trap reject", "distance-dominance gate", NULL},
        3U,
        "cheap-bound gate lifts priors-enabled raw 75.80% to 90.10%",
        "cheap-bound gate lifts priors-enabled raw 75.80% to 90.10%",
        "same gate contract works outside residue arithmetic",
        "the same candidate-verifier contract transfers to topology once the grammar stays domain-local.",
        "supporting transfer surface, not a main contribution",
        {NULL, NULL, NULL, NULL},
        0U,
    },
    {
        "delayed_claim",
        "delayed claim",
        "delayed reward/claim rows with hidden outcomes and margins",
        "deterministic_replay",
        "simulator-certified utility and safety under the delayed-claim generator",
        "threshold, margin, and sibling-claim predicates",
        "claim actions, threshold gates, and utility-margin rules",
        "deterministic hidden-outcome simulator and rule certificate checks",
        "finite-generator safety check plus false-rejection counters",
        "rank utility only after safety dominates the objective",
        "rank utility after simulator safety dominates the decision",
        "consume simulator/gate rejection reasons as scoped feedback",
        "deterministic clean threshold and margin gates",
        "crlp.delayed_claim.rule_trace.v1; crlp.rejected_candidate_learning.*",
        {"safe threshold claim", "margin reject", "unsafe reason-prior transfer", NULL},
        3U,
        "clean threshold+margin gate reduces four actions to one; noisy rows expose failures",
        "clean threshold+margin gate reduces four actions to one; noisy rows expose failures",
        "reason learning needs domain-aware safety constraints",
        "utility ranking must be downstream of safety when false rejection risk is domain-specific.",
        "reason-prior transfer alone can fail safety; claim remains domain-scoped",
        {"C4.delayed_claim_sibling_transfer", NULL, NULL, NULL},
        1U,
    },
    {
        "rejected_candidate_learning",
        "rejected candidate learning",
        "verifier rejection rows grouped by seed, split, and reason family",
        "exact_check",
        "reason-code feedback on arithmetic/modular verifier rows; replay stress rows are limitations only",
        "the original domain grammars that produced rejected candidate rows",
        "rejected verifier rows grouped by task, split, reason family, and seed",
        "original rule/certificate checker that rejected or accepted each row",
        "false-rejection-first budget filter over reason priors and residual scores",
        "learn residual ordering from rejection reasons and marginal verifier utility",
        "train residual ordering from rejected verifier outcomes",
        "feed reason codes and residual classes into the next proposal pass",
        "unguided model_score versus reason_prior and trained_residual",
        "crlp.rejected_candidate_learning.*",
        {"unsafe candidate", "no-support row", "trained residual recovery", NULL},
        3U,
        "trained residual reaches 7 accepted rules / 245 utility with zero false rejections",
        "trained residual reaches 7 accepted rules / 245 utility with zero false rejections",
        "verifier failures become training signal",
        "rejected verifier rows are reusable supervision, not discarded failed searches.",
        "not a generic reason-code ranker; delayed-claim safety remains scoped",
        {"C5.rejected_candidate_reason_learning", NULL, NULL, NULL},
        1U,
    },
    {
        "hypothesis_library_feedback",
        "hypothesis library feedback",
        "ARC-like hypothesis fixtures split into promotion and held-out task ids",
        "exact_check",
        "train-pair exactness and held-out fixture certification under the typed hypothesis grammar",
        "typed templates, slot bindings, and compiled program candidates",
        "certified hypothesis templates, slot bindings, and compiled programs",
        "train-pair exactness, type checks, and held-out fixture split evaluation",
        "zero-false-rejection promotion before a binding enters the library",
        "rank promoted slot priors and residual program candidates under tight budgets",
        "rank bounded program candidates after template certification",
        "turn zero-false-rejection templates into slot priors for the next search",
        "no_prior and grammar-order tight-budget selection",
        "crlp.hypothesis.evidence.v1; crlp.hypothesis_library.heldout_result.v1",
        {"slot prior", "promoted template", "held-out certified@1 row", NULL},
        3U,
        "slot priors raise certified@1 from 2 to 4; held-out split rows = 30",
        "slot priors raise certified@1 from 2 to 4; held-out split rows = 30",
        "certified learning can accumulate through reusable library priors",
        "certified hypotheses can become library priors and improve tight-budget held-out certification.",
        "public ARC promotion remains scoped evidence, not a headline ARC score",
        {"C5.hypothesis_library_promotion", NULL, NULL, NULL},
        1U,
    },
    {
        "multitask_tiny_net",
        "multitask tiny net",
        "shared residual rows from prime, grid, claim, and ARC surfaces",
        "mixed_diagnostic",
        "diagnostic scorer rows only; not aggregated into an authority-level leaderboard",
        "domain-local grammars encoded through common residual feature rows",
        "shared residual ranking rows across prime/grid/claim/ARC",
        "task-local labels and certified downstream gates",
        "per-task held-out metrics, OOD ARC variants, calibration bins",
        "shared residual scorer with domain-specific diagnostics",
        "shared residual scorer with per-domain diagnostics",
        "none promoted; architecture rows are blocked by the plateau audit",
        "single-task scorer and C7 plateau audit",
        "crlp.multitask.summary.v1; crlp.multitask.tiny_net_plateau.v1; crlp.multitask.tiny_net_diagnostic.v1",
        {"shared trunk row", "ARC OOD row", "capacity plateau row", NULL},
        3U,
        "ARC OOD shared +5.57pp; capacity is not monotonic; 128-epoch plateau audit is underpowered",
        "ARC OOD shared +5.57pp; capacity is not monotonic; 128-epoch plateau audit is underpowered",
        "representation/protocol limits dominate naive width scaling",
        "small rankers are useful around certified structure, but one flat head exposes interference.",
        "architecture-size claims stay gated by plateau-ready diagnostics",
        {"C4.arc_ood_shared_vs_single", "C4.multitask_capacity_tradeoff", "C7.tiny_net_plateau_stop_rule", NULL},
        3U,
    },
    {
        "arc_agi_evidence_surface",
        "arc agi evidence surface",
        "public and fixture ARC tasks with cell, object, transform, and hypothesis candidates",
        "corpus_proxy",
        "public-corpus coverage and failure taxonomy, not broad exact ARC competence",
        "typed transforms, object/slot bindings, and bounded program templates",
        "cell colors, typed transforms, programs, and hypothesis-derived candidates",
        "train-pair consistency, unique compiled prediction, and public corpus loader",
        "program/hypothesis certificate rows and scorer-invariant loader summaries",
        "rank bounded candidates; scorer does not own correctness",
        "rank bounded candidates and expose calibration/coverage gaps",
        "grow certified transform/program grammar coverage rather than claim scorer capacity fixes",
        "corpus loader coverage and train-pair certified transform rows",
        "crlp.arc_loader.summary.v1; crlp.program_proposer.summary.v1; crlp.hypothesis.evidence.v1",
        {"typed transform", "hypothesis candidate", "loader summary", NULL},
        3U,
        "public eval coverage is 7/400 tasks and 1057 cells under the measured surface",
        "public eval coverage is 7/400 tasks and 1057 cells under the measured surface",
        "ARC exposes priors, grammar coverage, and failure taxonomy",
        "ARC is an evidence surface for grammar coverage and failure taxonomy, not a score chase.",
        "not a broad ARC competence or benchmark-score claim",
        {"C1.arc_gate_scorer_invariance", "C1.arc_public_evidence_surface", "C3.arc_ood_calibration_tradeoff", NULL},
        3U,
    },
    {
        "arc_verified_self_play",
        "arc verified self play",
        "generated ARC-like tasks from forward visual-primitive composition plus existing ARC visual-pipeline attempts",
        "exact_check",
        "exact synthetic supervision for inverse proposer training; support requires state-hash leakage disjointness and public ARC fixed-budget replay against no-prior baseline logs",
        "visual primitives for object motion, anchor-relative translation, recolor, tile/scale, crop/extract, projection/panel, counting, fill, and symmetry",
        "forward-composed visual primitive programs, verifier-shaped negative siblings, and exported compiler-backed visual priors",
        "known-correct synthetic output construction, ARC visual-pipeline verifier fields, and full/matched visual-only public ARC replay logs",
        "accepted rows are generated by primitive composition; public coverage requires ARC verifier replay against full and matched visual-only baseline logs",
        "train inverse visual-program induction from exact self-play rows",
        "none as correctness owner; ranker learns inverse program ordering from generated verifier rows",
        "self-play proposer trained on accepted programs and verifier-reason negative siblings, then exported as compiler-backed visual priors; synthetic top-1 is in-distribution only",
        "full no-prior public replay, matched visual-only no-prior public replay, authored template baseline 7 certified eval tasks, oracle ceiling 14, and nine zero-coverage miss families",
        "crlp.arc_verified_self_play.summary.v1; crlp.arc_verified_self_play.prior.v1; crlp.arc_synthetic_shape_curriculum.v1; crlp.protocol.verified_selfplay_sample.v1; crlp.neural_grammar_discovery.arc_visual_pipeline_neural_summary.v1",
        {"verified self-play sample", "forward-composed accepted program", "fixed-budget public ablation", NULL},
        3U,
        "self-play is registered as an underpowered L1 search-efficiency surface until fixed-budget public replay, leakage disjointness, seed count, and zero-family opening gates pass",
        "self-play is registered as an underpowered L1 search-efficiency surface until fixed-budget public replay, leakage disjointness, seed count, and zero-family opening gates pass",
        "ARC proposer training can be generated from verifier-owned primitives, but support requires fixed-budget public replay rather than synthetic in-distribution top-1",
        "the durable ARC asset is the verifier/primitive system; authored solution libraries are baselines, not the training source.",
        "distribution mismatch, leakage audit, and public fixed-budget attribution remain open before self-play can be credited",
        {"C5.arc_verified_self_play", NULL, NULL, NULL},
        1U,
    },
    {
        "signal_flight",
        "signal flight",
        "Signal flight replay frames, closed-loop rollout rows, live shadow-control comparisons, and live neural-active rows",
        "deterministic_replay",
        "simulator replay, rollout replay, protocol-compatible live telemetry, and guarded safety evidence",
        "bounded action/control-option candidates beside the heuristic controller",
        "certified control-target hypotheses plus nine WASD action candidates",
        "offline counterfactual simulator replay, closed-loop rollout replay, live shadow frame comparison, and live active telemetry",
        "replay utility, regret, damage/death counters, certified control-target rollout rows, shadow disagreement logs, live safety gates, and protocol compatibility rows",
        "rank bounded control candidates beside heuristic control",
        "rank bounded WASD/control candidates beside heuristic control",
        "control-target hypotheses remain replay-gated before action ranking",
        "heuristic controller, simulator replay, and live guardrails",
        "crlp.signal_flight.summary.v1; crlp.signal_flight_rollout.summary.v1; crlp.signal_shadow_replay_audit.v1; crlp.signal_live_active_summary.v1",
        {"WASD candidate", "counterfactual utility row", "rollout controller row", "shadow disagreement row"},
        4U,
        "offline replay, rollout bridge, and live neural-active claims supported; active baseline comparison remains open",
        "offline replay, rollout bridge, and guarded live neural-active evidence are reported; active baseline comparison remains open",
        "application pressure for replay-grounded residual ranking and guarded live neural-active execution",
        "Signal adds application pressure through replay, rollout, shadow, and guarded live artifacts while keeping active-control performance open.",
        "active neural versus heuristic performance is not yet claimed",
        {"C9.signal_flight_replay_control", "C9.signal_live_neural_active", NULL, NULL},
        2U,
    },
};

static const size_t PAPER_EXPERIMENT_MATRIX_ROW_COUNT =
    sizeof(PAPER_EXPERIMENT_MATRIX_ROWS) / sizeof(PAPER_EXPERIMENT_MATRIX_ROWS[0]);

static const PaperClaimResult *paper_find_claim(
    const PaperClaimResult claims[PAPER_CLAIM_COUNT],
    const char *id) {
    size_t i;
    for (i = 0U; i < PAPER_CLAIM_COUNT; ++i) {
        if (strcmp(claims[i].spec->id, id) == 0) {
            return &claims[i];
        }
    }
    return NULL;
}

static void paper_today_iso(char out[11]) {
    time_t now;
    struct tm tm_value;
    now = time(NULL);
    if (now == (time_t)-1) {
        strcpy(out, "unknown");
        return;
    }
    if (localtime_r(&now, &tm_value) == NULL ||
        strftime(out, 11U, "%Y-%m-%d", &tm_value) == 0U) {
        strcpy(out, "unknown");
    }
}

static void paper_matrix_join_examples(const PaperExperimentMatrixRow *row,
                                       char *out,
                                       size_t out_size) {
    size_t i;
    out[0] = '\0';
    for (i = 0U; i < row->example_count; ++i) {
        if (i > 0U) {
            paper_appendf(out, out_size, "; ");
        }
        paper_appendf(out, out_size, "%s", row->examples[i]);
    }
}

static void paper_matrix_effective_result(
    const PaperExperimentMatrixRow *row,
    const PaperClaimResult claims[PAPER_CLAIM_COUNT],
    char *result,
    size_t result_size,
    char *result_table,
    size_t result_table_size) {
    const PaperClaimResult *replay;
    const PaperClaimResult *live;
    snprintf(result, result_size, "%s", row->result);
    snprintf(result_table, result_table_size, "%s", row->result_table);
    if (strcmp(row->domain, "signal_flight") != 0) {
        return;
    }
    replay = paper_find_claim(claims, "C9.signal_flight_replay_control");
    live = paper_find_claim(claims, "C9.signal_live_neural_active");
    if (replay != NULL && replay->summary[0] != '\0' &&
        live != NULL && live->summary[0] != '\0') {
        snprintf(result,
                 result_size,
                 "%s %s",
                 replay->summary,
                 live->summary);
    }
    if (live != NULL && strcmp(live->status, "supported") == 0) {
        snprintf(result_table,
                 result_table_size,
                 "offline replay, rollout bridge, and live neural-active claims supported; active baseline comparison remains open");
    } else if (live != NULL) {
        snprintf(result_table,
                 result_table_size,
                 "offline replay and rollout bridge supported; live neural-active evidence %s pending protocol-compatible rows; active baseline comparison remains open",
                 live->status);
    }
}

static int paper_matrix_validate_row(const PaperExperimentMatrixRow *row) {
    if (row->domain == NULL || row->domain[0] == '\0' ||
        row->data_domain == NULL || row->data_domain[0] == '\0' ||
        row->grounding_authority == NULL ||
        row->grounding_authority[0] == '\0' ||
        strcmp(row->grounding_authority, "unclassified") == 0 ||
        row->authority_claim_scope == NULL ||
        row->authority_claim_scope[0] == '\0' ||
        row->grammar == NULL || row->grammar[0] == '\0' ||
        row->candidate_set == NULL || row->candidate_set[0] == '\0' ||
        row->verifier == NULL || row->verifier[0] == '\0' ||
        row->certificate_gate == NULL || row->certificate_gate[0] == '\0' ||
        row->scorer_role == NULL || row->scorer_role[0] == '\0' ||
        row->neural_scorer_role == NULL ||
        row->neural_scorer_role[0] == '\0' ||
        row->neural_proposer_role == NULL ||
        row->neural_proposer_role[0] == '\0' ||
        row->headline_baseline == NULL ||
        row->headline_baseline[0] == '\0' ||
        row->trace_artifact == NULL || row->trace_artifact[0] == '\0' ||
        row->result_table == NULL || row->result_table[0] == '\0' ||
        row->takeaway == NULL || row->takeaway[0] == '\0' ||
        row->limitation == NULL || row->limitation[0] == '\0') {
        return -1;
    }
    if (row->example_count < 2U || row->example_count > 4U) {
        return -1;
    }
    return 0;
}

static void paper_write_json_string_array(FILE *fp,
                                          const char *const *items,
                                          size_t count) {
    size_t i;
    fputc('[', fp);
    for (i = 0U; i < count; ++i) {
        if (i > 0U) {
            fputs(", ", fp);
        }
        paper_write_json_string(fp, items[i]);
    }
    fputc(']', fp);
}

static void paper_write_matrix_claim_statuses(
    FILE *fp,
    const PaperExperimentMatrixRow *row,
    const PaperClaimResult claims[PAPER_CLAIM_COUNT]) {
    size_t i;
    fputc('{', fp);
    for (i = 0U; i < row->claim_id_count; ++i) {
        const PaperClaimResult *claim = paper_find_claim(claims, row->claim_ids[i]);
        if (i > 0U) {
            fputs(", ", fp);
        }
        paper_write_json_string(fp, row->claim_ids[i]);
        fputs(": ", fp);
        paper_write_json_string(fp, claim == NULL ? "not_registered" : claim->status);
    }
    fputc('}', fp);
}

static int paper_write_experiment_matrix_jsonl(
    const char *path,
    const PaperClaimResult claims[PAPER_CLAIM_COUNT]) {
    FILE *fp;
    char date[11];
    size_t i;
    if (paper_mkdir_parent(path) != 0) {
        return -1;
    }
    fp = fopen(path, "w");
    if (fp == NULL) {
        return -1;
    }
    paper_today_iso(date);
    for (i = 0U; i < PAPER_EXPERIMENT_MATRIX_ROW_COUNT; ++i) {
        const PaperExperimentMatrixRow *row = &PAPER_EXPERIMENT_MATRIX_ROWS[i];
        char result[4096];
        char result_table[1024];
        paper_matrix_effective_result(row,
                                      claims,
                                      result,
                                      sizeof(result),
                                      result_table,
                                      sizeof(result_table));
        fputs("{\"authority_claim_scope\": ", fp);
        paper_write_json_string(fp, row->authority_claim_scope);
        fputs(", \"candidate_set\": ", fp);
        paper_write_json_string(fp, row->candidate_set);
        fputs(", \"certificate_gate\": ", fp);
        paper_write_json_string(fp, row->certificate_gate);
        fputs(", \"claim\": ", fp);
        paper_write_json_string(fp, row->claim);
        fputs(", \"claim_ids\": ", fp);
        paper_write_json_string_array(fp, row->claim_ids, row->claim_id_count);
        fputs(", \"claim_statuses\": ", fp);
        paper_write_matrix_claim_statuses(fp, row, claims);
        fputs(", \"data_domain\": ", fp);
        paper_write_json_string(fp, row->data_domain);
        fputs(", \"date\": ", fp);
        paper_write_json_string(fp, date);
        fputs(", \"domain\": ", fp);
        paper_write_json_string(fp, row->domain);
        fputs(", \"domain_display\": ", fp);
        paper_write_json_string(fp, row->domain_display);
        fputs(", \"examples\": ", fp);
        paper_write_json_string_array(fp, row->examples, row->example_count);
        fputs(", \"grammar\": ", fp);
        paper_write_json_string(fp, row->grammar);
        fputs(", \"grounding_authority\": ", fp);
        paper_write_json_string(fp, row->grounding_authority);
        fputs(", \"headline_baseline\": ", fp);
        paper_write_json_string(fp, row->headline_baseline);
        fputs(", \"limitation\": ", fp);
        paper_write_json_string(fp, row->limitation);
        fputs(", \"neural_proposer_role\": ", fp);
        paper_write_json_string(fp, row->neural_proposer_role);
        fputs(", \"neural_scorer_role\": ", fp);
        paper_write_json_string(fp, row->neural_scorer_role);
        fputs(", \"result\": ", fp);
        paper_write_json_string(fp, result);
        fputs(", \"result_table\": ", fp);
        paper_write_json_string(fp, result_table);
        fputs(", \"schema\": ", fp);
        paper_write_json_string(fp, "crlp.experiment_matrix.row.v1");
        fputs(", \"scorer_role\": ", fp);
        paper_write_json_string(fp, row->scorer_role);
        fputs(", \"takeaway\": ", fp);
        paper_write_json_string(fp, row->takeaway);
        fputs(", \"trace_artifact\": ", fp);
        paper_write_json_string(fp, row->trace_artifact);
        fputs(", \"verifier\": ", fp);
        paper_write_json_string(fp, row->verifier);
        fputs("}\n", fp);
    }
    fclose(fp);
    return 0;
}

static void paper_write_md_escaped(FILE *fp, const char *value) {
    const char *cursor = value == NULL ? "" : value;
    while (*cursor != '\0') {
        if (*cursor == '|') {
            fputs("\\|", fp);
        } else if (*cursor == '\n' || *cursor == '\r') {
            fputc(' ', fp);
        } else {
            fputc((unsigned char)*cursor, fp);
        }
        ++cursor;
    }
}

static void paper_write_md_examples(FILE *fp, const PaperExperimentMatrixRow *row) {
    char examples[512];
    paper_matrix_join_examples(row, examples, sizeof(examples));
    paper_write_md_escaped(fp, examples);
}

static int paper_write_experiment_matrix_markdown(
    const char *path,
    const PaperClaimResult claims[PAPER_CLAIM_COUNT]) {
    FILE *fp;
    char date[11];
    size_t i;
    static const char *const authorities[] = {
        "exact_check",
        "deterministic_replay",
        "corpus_proxy",
        "human_review",
    };
    if (paper_mkdir_parent(path) != 0) {
        return -1;
    }
    fp = fopen(path, "w");
    if (fp == NULL) {
        return -1;
    }
    paper_today_iso(date);
    fprintf(fp,
            "# Frozen Experiment Matrix\n\nDate: %s\n\n"
            "The evidence families are frozen to these rows. New domains belong in roadmap notes until this matrix is deliberately revised.\n\n"
            "Authority is a hard reporting axis. Exact checks, deterministic replay, corpus proxies, and mixed diagnostics are not collapsed into a single score.\n\n"
            "| Domain/data | Authority | Grammar | Candidate set | Verifier/gate | Neural role split | Headline baseline | Trace artifact | Result table | Takeaway | Limitation |\n"
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n",
            date);
    for (i = 0U; i < PAPER_EXPERIMENT_MATRIX_ROW_COUNT; ++i) {
        const PaperExperimentMatrixRow *row = &PAPER_EXPERIMENT_MATRIX_ROWS[i];
        char result[4096];
        char result_table[1024];
        paper_matrix_effective_result(row,
                                      claims,
                                      result,
                                      sizeof(result),
                                      result_table,
                                      sizeof(result_table));
        fputs("| `", fp);
        paper_write_md_escaped(fp, row->domain);
        fputs("`: ", fp);
        paper_write_md_escaped(fp, row->data_domain);
        fputs(" | `", fp);
        paper_write_md_escaped(fp, row->grounding_authority);
        fputs("`: ", fp);
        paper_write_md_escaped(fp, row->authority_claim_scope);
        fputs(" | ", fp);
        paper_write_md_escaped(fp, row->grammar);
        fputs(" | ", fp);
        paper_write_md_escaped(fp, row->candidate_set);
        fputs(" | ", fp);
        paper_write_md_escaped(fp, row->verifier);
        fputs("; ", fp);
        paper_write_md_escaped(fp, row->certificate_gate);
        fputs(" | scorer: ", fp);
        paper_write_md_escaped(fp, row->neural_scorer_role);
        fputs("; proposer: ", fp);
        paper_write_md_escaped(fp, row->neural_proposer_role);
        fputs(" | ", fp);
        paper_write_md_escaped(fp, row->headline_baseline);
        fputs(" | `", fp);
        paper_write_md_escaped(fp, row->trace_artifact);
        fputs("` | ", fp);
        paper_write_md_escaped(fp, result_table);
        fputs(" | ", fp);
        paper_write_md_escaped(fp, row->takeaway);
        fputs(" | ", fp);
        paper_write_md_escaped(fp, row->limitation);
        fputs(" |\n", fp);
    }
    fputs("\n## Authority Leaderboards\n\n", fp);
    for (i = 0U; i < sizeof(authorities) / sizeof(authorities[0]); ++i) {
        size_t j;
        int found = 0;
        fprintf(fp, "### `%s`\n\n", authorities[i]);
        for (j = 0U; j < PAPER_EXPERIMENT_MATRIX_ROW_COUNT; ++j) {
            const PaperExperimentMatrixRow *row = &PAPER_EXPERIMENT_MATRIX_ROWS[j];
            char result[4096];
            char result_table[1024];
            if (strcmp(row->grounding_authority, authorities[i]) != 0) {
                continue;
            }
            paper_matrix_effective_result(row,
                                          claims,
                                          result,
                                          sizeof(result),
                                          result_table,
                                          sizeof(result_table));
            fputs("- `", fp);
            paper_write_md_escaped(fp, row->domain);
            fputs("`: ", fp);
            paper_write_md_escaped(fp, result_table);
            fputc('\n', fp);
            found = 1;
        }
        if (!found) {
            fputs("- No frozen rows.\n", fp);
        }
        fputc('\n', fp);
    }
    fputs("### `mixed_diagnostic`\n\n", fp);
    for (i = 0U; i < PAPER_EXPERIMENT_MATRIX_ROW_COUNT; ++i) {
        const PaperExperimentMatrixRow *row = &PAPER_EXPERIMENT_MATRIX_ROWS[i];
        if (strcmp(row->grounding_authority, "exact_check") == 0 ||
            strcmp(row->grounding_authority, "deterministic_replay") == 0 ||
            strcmp(row->grounding_authority, "corpus_proxy") == 0 ||
            strcmp(row->grounding_authority, "human_review") == 0) {
            continue;
        }
        fputs("- `", fp);
        paper_write_md_escaped(fp, row->domain);
        fputs("`: ", fp);
        paper_write_md_escaped(fp, row->authority_claim_scope);
        fputc('\n', fp);
    }
    fputs("\n\n## Repeated Protocol Rows\n\n", fp);
    for (i = 0U; i < PAPER_EXPERIMENT_MATRIX_ROW_COUNT; ++i) {
        const PaperExperimentMatrixRow *row = &PAPER_EXPERIMENT_MATRIX_ROWS[i];
        char result[4096];
        char result_table[1024];
        paper_matrix_effective_result(row,
                                      claims,
                                      result,
                                      sizeof(result),
                                      result_table,
                                      sizeof(result_table));
        fprintf(fp, "### `%s`\n\n", row->domain);
        fputs("1. Domain/data: ", fp);
        paper_write_md_escaped(fp, row->data_domain);
        fputs("\n2. Authority: `", fp);
        paper_write_md_escaped(fp, row->grounding_authority);
        fputs("` - ", fp);
        paper_write_md_escaped(fp, row->authority_claim_scope);
        fputs("\n3. Grammar: ", fp);
        paper_write_md_escaped(fp, row->grammar);
        fputs("\n4. Candidate set: ", fp);
        paper_write_md_escaped(fp, row->candidate_set);
        fputs("\n5. Verifier/gate: ", fp);
        paper_write_md_escaped(fp, row->verifier);
        fputs("; ", fp);
        paper_write_md_escaped(fp, row->certificate_gate);
        fputs("\n6. Neural scorer role: ", fp);
        paper_write_md_escaped(fp, row->neural_scorer_role);
        fputs("\n7. Neural proposer role: ", fp);
        paper_write_md_escaped(fp, row->neural_proposer_role);
        fputs("\n8. Headline baseline: ", fp);
        paper_write_md_escaped(fp, row->headline_baseline);
        fputs("\n9. Examples: ", fp);
        paper_write_md_examples(fp, row);
        fputs("\n10. Result table: ", fp);
        paper_write_md_escaped(fp, result_table);
        fputs("\n11. Takeaway: ", fp);
        paper_write_md_escaped(fp, row->takeaway);
        fputs("\n12. Limitation: ", fp);
        paper_write_md_escaped(fp, row->limitation);
        if (i + 1U < PAPER_EXPERIMENT_MATRIX_ROW_COUNT) {
            fputs("\n\n", fp);
        } else {
            fputc('\n', fp);
        }
    }
    fclose(fp);
    return 0;
}

static int paper_write_experiment_matrix_typ(
    const char *path,
    const PaperClaimResult claims[PAPER_CLAIM_COUNT]) {
    FILE *fp;
    size_t i;
    if (paper_mkdir_parent(path) != 0) {
        return -1;
    }
    fp = fopen(path, "w");
    if (fp == NULL) {
        return -1;
    }
    fputs("// Generated by the C paper runtime; do not edit by hand.\n",
          fp);
    fputs("#let experiment_matrix_rows = (\n", fp);
    for (i = 0U; i < PAPER_EXPERIMENT_MATRIX_ROW_COUNT; ++i) {
        const PaperExperimentMatrixRow *row = &PAPER_EXPERIMENT_MATRIX_ROWS[i];
        char examples[512];
        char result[4096];
        char result_table[1024];
        paper_matrix_join_examples(row, examples, sizeof(examples));
        paper_matrix_effective_result(row,
                                      claims,
                                      result,
                                      sizeof(result),
                                      result_table,
                                      sizeof(result_table));
        fputs("  (\n    domain: ", fp);
        paper_write_typ_string(fp, row->domain);
        fputs(",\n    domain_display: ", fp);
        paper_write_typ_string(fp, row->domain_display);
        fputs(",\n    data_domain: ", fp);
        paper_write_typ_string(fp, row->data_domain);
        fputs(",\n    grounding_authority: ", fp);
        paper_write_typ_string(fp, row->grounding_authority);
        fputs(",\n    authority_claim_scope: ", fp);
        paper_write_typ_string(fp, row->authority_claim_scope);
        fputs(",\n    grammar: ", fp);
        paper_write_typ_string(fp, row->grammar);
        fputs(",\n    candidate_set: ", fp);
        paper_write_typ_string(fp, row->candidate_set);
        fputs(",\n    verifier: ", fp);
        paper_write_typ_string(fp, row->verifier);
        fputs(",\n    certificate_gate: ", fp);
        paper_write_typ_string(fp, row->certificate_gate);
        fputs(",\n    scorer_role: ", fp);
        paper_write_typ_string(fp, row->scorer_role);
        fputs(",\n    neural_scorer_role: ", fp);
        paper_write_typ_string(fp, row->neural_scorer_role);
        fputs(",\n    neural_proposer_role: ", fp);
        paper_write_typ_string(fp, row->neural_proposer_role);
        fputs(",\n    headline_baseline: ", fp);
        paper_write_typ_string(fp, row->headline_baseline);
        fputs(",\n    trace_artifact: ", fp);
        paper_write_typ_string(fp, row->trace_artifact);
        fputs(",\n    examples: ", fp);
        paper_write_typ_string(fp, examples);
        fputs(",\n    result_table: ", fp);
        paper_write_typ_string(fp, result_table);
        fputs(",\n    result: ", fp);
        paper_write_typ_string(fp, result);
        fputs(",\n    claim: ", fp);
        paper_write_typ_string(fp, row->claim);
        fputs(",\n    takeaway: ", fp);
        paper_write_typ_string(fp, row->takeaway);
        fputs(",\n    limitation: ", fp);
        paper_write_typ_string(fp, row->limitation);
        fputs(",\n  ),\n", fp);
    }
    fputs(")\n", fp);
    fclose(fp);
    return 0;
}

static int paper_write_experiment_matrix(
    const char *output_dir,
    const PaperClaimResult claims[PAPER_CLAIM_COUNT]) {
    char jsonl_path[4096];
    char md_path[4096];
    char typ_path[4096];
    size_t i;
    for (i = 0U; i < PAPER_EXPERIMENT_MATRIX_ROW_COUNT; ++i) {
        if (paper_matrix_validate_row(&PAPER_EXPERIMENT_MATRIX_ROWS[i]) != 0) {
            return -1;
        }
    }
    snprintf(jsonl_path, sizeof(jsonl_path), "%s/experiment_matrix.jsonl", output_dir);
    snprintf(md_path, sizeof(md_path), "%s/experiment_matrix.md", output_dir);
    snprintf(typ_path, sizeof(typ_path), "%s/experiment_matrix.typ", output_dir);
    if (paper_write_experiment_matrix_jsonl(jsonl_path, claims) != 0 ||
        paper_write_experiment_matrix_markdown(md_path, claims) != 0 ||
        paper_write_experiment_matrix_typ(typ_path, claims) != 0) {
        return -1;
    }
    return 0;
}
