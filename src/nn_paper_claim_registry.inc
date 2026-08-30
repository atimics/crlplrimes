typedef struct {
    const char *id;
    const char *family;
    const char *title;
    const char *source;
    int required;
    const char *claim_text;
    const char *explanation_text;
    const char *chart_kind;
    const char *chart_unit;
    const char *chart_y_label;
    const char *freshness_prerequisite;
    const char *refresh_command;
    const char *evidence_origin;
    int allow_synthetic_support;
} PaperClaimSpec;

typedef struct {
    const PaperClaimSpec *spec;
    char status[24];
    char summary[PAPER_CLAIM_SUMMARY_SIZE];
    char explanation[PAPER_CLAIM_EXPLANATION_SIZE];
    char chart_rows[PAPER_CLAIM_CHART_SIZE];
    double raw_top1;
    double hand_top1;
    double learned_top1;
    double raw_top10;
    double hand_top10;
    double learned_top10;
    double learned_vs_hand_delta_pp;
    long accepted_rules;
    long false_rejections;
    long sample_count;
} PaperClaimResult;

typedef struct {
    char headers[80][128];
    char values[80][256];
    size_t count;
} PaperCsvRecord;

static const PaperClaimSpec PAPER_CLAIM_SPECS[PAPER_CLAIM_COUNT] = {
    {
        "C1.prime_gate_delta",
        "C1",
        "Expanded certified next-prime filter improves top-1 over raw and hand-wheel gates",
        "results/latest/comparison.canonical.jsonl",
        0,
        "Holding the scorer checkpoint fixed, the expanded certified next-prime filter improves top-1 ranking over raw and hand-wheel filter conditions without verifier-counted true-answer deletions.",
        "",
        "",
        "",
        "",
        "results/latest/commands.txt",
        "make results",
        "measured_run",
        0,
    },
    {
        "C1.prime_gate_delta_ci",
        "C1",
        "Top-1 rank is invariant across five registered seed checkpoints",
        "results/latest/prime_gate_delta_seed_interval.jsonl",
        0,
        "The expanded-filter top-1 metric is invariant across five registered checkpoints on one fixed prime-range evaluation set; this is not a confidence interval or broad robustness result.",
        "",
        "",
        "",
        "",
        "results/latest/comparison.canonical.jsonl",
        "make prime-gate-seed-audit",
        "measured_run",
        0,
    },
    {
        "C2.prime_proposer_fixed_budget",
        "C2",
        "Next-prime static proposer is a fixed-budget negative baseline",
        "results/latest/comparison.canonical.jsonl",
        0,
        "The static neural next-prime proposer does not beat deterministic symbolic orderings under matched budgets.",
        "",
        "",
        "",
        "",
        "results/latest/commands.txt",
        "make results",
        "measured_run",
        0,
    },
    {
        "C4.arc_ood_shared_vs_single",
        "C4",
        "Shared scorer beats single-task scorer on same-envelope ARC OOD",
        "results/2026-05-17-multitask-roadmap/rmt2_aggregate.csv",
        1,
        "The shared scorer beats the single-task scorer on same-envelope ARC OOD top-1 and log loss.",
        "This supports the narrow residual-generalization claim: the shared scorer loses on the fixed ARC training task, but improves held-out same-envelope ARC ranking and log loss.",
        "delta_bar",
        "pp",
        "shared minus single top-1",
        "",
        "",
        "measured_run",
        0,
    },
    {
        "C1.arc_gate_scorer_invariance",
        "C1",
        "Public ARC certified gate is invariant to scorer context labels",
        "results/2026-05-17-multitask-roadmap/rmt7_arc_gate_invariance_summary.csv",
        1,
        "The public ARC certified gate and proposer summaries are invariant to scorer context labels.",
        "This separates the certified ARC gate from learned scorer choice: the public ARC gate/proposer measurements are currently scorer-independent, so scorer claims and gate claims remain cleanly separated.",
        "coverage_bar",
        "correct cells",
        "certified correct cells",
        "",
        "",
        "measured_run",
        0,
    },
    {
        "C1.arc_public_evidence_surface",
        "C1",
        "Public ARC corpus parity row is measured and scoped",
        "results/2026-05-17-arc-evidence-surface/arc_evidence_summary.csv",
        1,
        "The public ARC row is a measured coverage surface, not a broad ARC competence claim.",
        "This is not a broad ARC-performance claim. It is the public-corpus parity row: the compiler can now distinguish measured ARC gate coverage, version-to-version movement, and the much larger unresolved residual.",
        "coverage_bar",
        "correct cells",
        "evaluation correct cells",
        "",
        "",
        "measured_run",
        0,
    },
    {
        "C3.arc_ood_calibration_tradeoff",
        "C3",
        "ARC shared scorer has a fixed-task versus OOD calibration tradeoff",
        "results/2026-05-17-multitask-roadmap/rmt5_calibration_summary.csv",
        1,
        "Shared ARC scoring improves some calibration surfaces while worsening others.",
        "The calibration claim is explicitly conditional: shared training is not simply better calibrated everywhere. It is worse on fixed ARC ECE but better on OOD ECE, while improving log loss in both settings.",
        "delta_bar",
        "pp",
        "shared ECE improvement vs single",
        "",
        "",
        "measured_run",
        0,
    },
    {
        "C4.delayed_claim_sibling_transfer",
        "C4",
        "Delayed-claim transfer gain is sibling-specific",
        "results/2026-05-17-multitask-roadmap/rmt3_leave_one_out.csv",
        1,
        "Delayed-claim transfer is positive in the registered shared envelope, but sibling-specific.",
        "This replaces the earlier broad positive-transfer claim with a narrower one: delayed_claim benefits from shared training in this envelope, grid_path appears to be the helpful sibling, and ARC is mild interference.",
        "delta_bar",
        "pp",
        "top-1 delta vs claim-only",
        "",
        "",
        "measured_run",
        0,
    },
    {
        "C4.multitask_capacity_tradeoff",
        "C4",
        "Capacity sweep does not explain away the ARC fixed-task gap",
        "results/2026-05-17-multitask-roadmap/rmt6_capacity_sweep_deltas.csv",
        0,
        "The capacity sweep remains diagnostic until the C7 plateau audit is ready.",
        "C7 is the blocking gate for capacity claims: width deltas stay in limitations until the affected task and variant rows are audit-ready.",
        "delta_bar",
        "pp",
        "shared minus single top-1",
        "",
        "",
        "measured_run",
        0,
    },
    {
        "C4.adapter_heads_shared_trunk",
        "C4",
        "Shared trunk plus domain heads is gated as architecture evidence",
        "results/2026-05-17-multitask-roadmap/rmt4_adapter_heads_deltas.csv",
        0,
        "Shared trunk plus task heads must pass C7 before being presented as the default recipe.",
        "The adapter-head result is treated as an architecture gate, not a broad one-model claim. C7 must be ready before a shared trunk plus domain heads can be promoted.",
        "delta_bar",
        "pp",
        "task-head top-1 delta",
        "",
        "",
        "measured_run",
        0,
    },
    {
        "C5.rejected_candidate_reason_learning",
        "C5",
        "Rejected verifier candidates become useful reason labels",
        "results/2026-05-17-substrate-extensions/rejected_candidate_reason_learning.jsonl",
        1,
        "Rejected verifier candidates can become reason labels that repair later proposals.",
        "This is the first feedback loop: rejected verifier rows become reason labels, and those labels improve the next proposal pass. The claim is scoped to the arithmetic-sequence/modular sweeps; delayed_claim remains a stress case.",
        "coverage_bar",
        "validation utility",
        "budget-8 validation utility",
        "",
        "",
        "measured_run",
        0,
    },
    {
        "C5.hypothesis_library_promotion",
        "C5",
        "Held-out hypothesis-library promotion is gated by certificates",
        "results/2026-05-17-substrate-extensions/hypothesis_library_promotions.jsonl",
        1,
        "Repeatedly safe hypotheses can be promoted into a library that helps future held-out tasks.",
        "This is the success-side feedback loop: repeatedly certified hypotheses become proposal priors only after the verifier has already accepted them.",
        "coverage_bar",
        "budget-1 certifications",
        "certified tasks",
        "",
        "",
        "measured_run",
        0,
    },
    {
        "C7.tiny_net_plateau_stop_rule",
        "C7",
        "Tiny-net architecture claims are gated by plateau audits",
        "results/2026-05-17-multitask-roadmap/tiny_net_plateau_summary.jsonl",
        0,
        "Capacity and architecture claims remain underpowered until tiny-net rows reach plateau-ready verdicts.",
        "Diagnostics separate weak signal, seed variance, non-plateaued rows, and width help before architecture claims can be promoted.",
        "",
        "",
        "",
        "",
        "",
        "measured_run",
        0,
    },
    {
        "C8.trace_determinism_audit",
        "C8",
        "Trace determinism audit catches replay-sensitive claim drift",
        "results/latest/trace_determinism.jsonl",
        1,
        "The trace determinism audit verifies that replay meaning survives numeric/runtime changes.",
        "",
        "",
        "",
        "",
        "build/double/traces/prime_gap.jsonl",
        "make trace-determinism-compare",
        "measured_run",
        0,
    },
    {
        "C8.dataset_engine_evidence_surface",
        "C8",
        "Dataset-first engine ports expose shared evidence without promoting domain claims",
        "build/float/dataset_action_score_artifacts/engine_evidence.jsonl",
        1,
        "The dataset-first action-scoring engine exposes complete candidate rows, verifier labels, shared replay/schema declarations, and engine evidence across the registered smoke adapters.",
        "This is an infrastructure claim: the rows prove that domains compile to one auditable engine surface, not that the smoke adapters replace richer domain-specific evidence.",
        "coverage_bar",
        "rows",
        "engine evidence rows",
        "build/float/dataset_action_score",
        "make dataset-action-score-engine-evidence",
        "measured_run",
        0,
    },
    {
        "C9.signal_flight_replay_control",
        "C9",
        "Signal replay model improves bounded offline control metrics",
        "results/2026-05-17-signal-flight/summary.jsonl",
        1,
        "Offline Signal replay uses bounded neural rankers without claiming ownership of safety.",
        "",
        "coverage_bar",
        "utility",
        "selected utility",
        "",
        "",
        "measured_run",
        0,
    },
    {
        "C9.signal_live_neural_active",
        "C9",
        "Live Signal pilot uses guarded checkpoint-ranked neural actions",
        "results/2026-05-17-signal-flight/live_active_summary.jsonl",
        1,
        "The live Signal pilot ran checkpoint-ranked neural actions only behind guards.",
        "",
        "coverage_bar",
        "rows",
        "live rows",
        "",
        "",
        "measured_run",
        0,
    },
    {
        "C5.arc_verified_self_play",
        "C5",
        "ARC verified self-play is a fixed-budget search-efficiency claim",
        "build/float/arc_verified_self_play/summary.jsonl",
        0,
        "Verified self-play may improve ARC proposer search only when fixed-budget public replay recovers certified oracle coverage.",
        "This claim is registered as underpowered until public eval replay, fixed-budget with/without self-play ablation, leakage disjointness, zero-family opening, and seed-count gates are all present.",
        "coverage_bar",
        "certified tasks",
        "public eval certified tasks",
        "",
        "",
        "measured_run",
        0,
    },
    {
        "C10.cross_domain_structural_transfer",
        "C10",
        "Cross-domain meta-learner certifies structural transfer with gated refusal",
        "build/float/traces/cross_domain_leave_out.jsonl",
        0,
        "Cross-domain meta-learner with typed payload regions certifies transfer on 9/11 domains.",
        "",
        "",
        "",
        "",
        "",
        "",
        "measured_run",
        0,
    },
    {
        "C9.signal_mining_replay_options",
        "C9",
        "Signal mining option-ranker evaluates bounded policy hypotheses through replay",
        "build/float/signal_mining_experiment/summary.jsonl",
        0,
        "Signal mining evaluates 420 policy grammar hypotheses through replay.",
        "",
        "",
        "",
        "",
        "",
        "",
        "measured_run",
        0,
    },
    {
        "C11.eml_scorer_benchmarks",
        "C11",
        "EML scorer comparisons require matched multi-seed evidence",
        "results/eml/scorer_benchmarks.jsonl",
        0,
        "Any EML scorer performance delta must come from a persistent, matched-parameter, multi-seed artifact.",
        "The current smoke benchmarks are not claim evidence, so the paper reports no EML accuracy delta until this artifact exists.",
        "",
        "",
        "",
        "",
        "make eml-claim-evidence",
        "measured_run",
        0,
    },
    {
        "C11.eml_binding_recall",
        "C11",
        "EML binding recall requires persistent multi-seed evidence",
        "results/eml/binding_recall.jsonl",
        0,
        "Any EML binding recall comparison must come from a persistent, multi-seed artifact with its configuration and hashes recorded.",
        "The current exploratory binding result is not promoted into paper prose until this artifact exists.",
        "",
        "",
        "",
        "",
        "make eml-claim-evidence",
        "measured_run",
        0,
    },
    {
        "D.1.claim_compiler_self_check",
        "D",
        "Claim registry and empirical-percentage source gate align",
        "paper/claims.yaml",
        1,
        "Runtime claim metadata matches the registry, and paper generation rejects literal empirical percentages that bypass generated claim surfaces.",
        "",
        "",
        "",
        "",
        "",
        "make paper-support",
        "measured_run",
        0,
    },
};

static void paper_appendf(char *buffer,
                          size_t buffer_size,
                          const char *fmt,
                          ...) {
    size_t used;
    va_list args;
    if (buffer == NULL || buffer_size == 0U) {
        return;
    }
    used = strlen(buffer);
    if (used >= buffer_size - 1U) {
        return;
    }
    va_start(args, fmt);
    (void)vsnprintf(buffer + used, buffer_size - used, fmt, args);
    va_end(args);
}

static void paper_claim_set_missing(PaperClaimResult *claim) {
    snprintf(claim->status, sizeof(claim->status), "%s", "missing");
    snprintf(claim->summary,
             sizeof(claim->summary),
             "missing source artifact %s.",
             claim->spec->source);
    claim->chart_rows[0] = '\0';
}

static void paper_claim_init(PaperClaimResult *claim, const PaperClaimSpec *spec) {
    memset(claim, 0, sizeof(*claim));
    claim->spec = spec;
    paper_claim_set_missing(claim);
    snprintf(claim->explanation, sizeof(claim->explanation), "%s", spec->explanation_text);
}

static int paper_claim_has_text(const char *text) {
    return text != NULL && text[0] != '\0';
}

static int paper_path_mtime(const char *path, time_t *out_mtime) {
    struct stat st;
    if (!paper_claim_has_text(path) || out_mtime == NULL || stat(path, &st) != 0) {
        return 0;
    }
    *out_mtime = st.st_mtime;
    return 1;
}

static void paper_claim_set_stale(PaperClaimResult *claim,
                                  const char *freshness_prerequisite,
                                  const char *refresh_command,
                                  int prerequisite_missing) {
    const char *command = paper_claim_has_text(refresh_command)
                              ? refresh_command
                              : "the registered refresh command";
    snprintf(claim->status, sizeof(claim->status), "%s", "stale");
    if (prerequisite_missing) {
        snprintf(claim->summary,
                 sizeof(claim->summary),
                 "cannot verify source freshness for %s because prerequisite %s is missing; rerun `%s`.",
                 claim->spec->source,
                 freshness_prerequisite,
                 command);
    } else {
        snprintf(claim->summary,
                 sizeof(claim->summary),
                 "source artifact %s is older than freshness prerequisite %s; rerun `%s`.",
                 claim->spec->source,
                 freshness_prerequisite,
                 command);
    }
    claim->chart_rows[0] = '\0';
}

static void paper_apply_staleness_gate(PaperClaimResult claims[PAPER_CLAIM_COUNT]) {
    const double mtime_slop_seconds = 1.0;
    size_t i;
    for (i = 0U; i < PAPER_CLAIM_COUNT; ++i) {
        time_t source_mtime;
        time_t prerequisite_mtime;
        const PaperClaimSpec *spec = claims[i].spec;
        if (spec == NULL ||
            !paper_claim_has_text(spec->freshness_prerequisite) ||
            strcmp(claims[i].status, "supported") != 0) {
            continue;
        }
        if (!paper_path_mtime(spec->source, &source_mtime)) {
            paper_claim_set_missing(&claims[i]);
            continue;
        }
        if (!paper_path_mtime(spec->freshness_prerequisite, &prerequisite_mtime)) {
            paper_claim_set_stale(&claims[i],
                                  spec->freshness_prerequisite,
                                  spec->refresh_command,
                                  1);
            continue;
        }
        if (difftime(source_mtime, prerequisite_mtime) < -mtime_slop_seconds) {
            paper_claim_set_stale(&claims[i],
                                  spec->freshness_prerequisite,
                                  spec->refresh_command,
                                  0);
        }
    }
}

typedef struct {
    char *source;
    char *command;
    char *freshness_prerequisite;
} PaperClaimRegistryEvidence;

static void paper_claim_registry_evidence_free(PaperClaimRegistryEvidence *evidence) {
    if (evidence == NULL) {
        return;
    }
    free(evidence->source);
    free(evidence->command);
    free(evidence->freshness_prerequisite);
    memset(evidence, 0, sizeof(*evidence));
}

static char *paper_read_text_file(const char *path) {
    FILE *fp;
    char *content;
    size_t capacity = 4096U;
    size_t used = 0U;
    if (!paper_claim_has_text(path)) {
        return NULL;
    }
    fp = fopen(path, "rb");
    if (fp == NULL) {
        return NULL;
    }
    content = (char *)malloc(capacity);
    if (content == NULL) {
        fclose(fp);
        return NULL;
    }
    for (;;) {
        size_t available = capacity - used - 1U;
        size_t read_count;
        char *next;
        if (available == 0U) {
            capacity *= 2U;
            next = (char *)realloc(content, capacity);
            if (next == NULL) {
                free(content);
                fclose(fp);
                return NULL;
            }
            content = next;
            available = capacity - used - 1U;
        }
        read_count = fread(content + used, 1U, available, fp);
        used += read_count;
        if (read_count < available) {
            if (ferror(fp)) {
                free(content);
                fclose(fp);
                return NULL;
            }
            break;
        }
    }
    content[used] = '\0';
    fclose(fp);
    return content;
}

static const char *paper_json_find_key_value_bounded(const char *start,
                                                     const char *end,
                                                     const char *key) {
    char pattern[128];
    int written;
    const char *cursor;
    if (start == NULL || end == NULL || start > end || key == NULL || key[0] == '\0') {
        return NULL;
    }
    written = snprintf(pattern, sizeof(pattern), "\"%s\":", key);
    if (written <= 0 || (size_t)written >= sizeof(pattern)) {
        return NULL;
    }
    cursor = start;
    while (cursor < end) {
        cursor = strstr(cursor, pattern);
        if (cursor == NULL || cursor >= end) {
            return NULL;
        }
        if (cursor + (size_t)written <= end) {
            return cursor + (size_t)written;
        }
        ++cursor;
    }
    return NULL;
}

static int paper_registry_find_claim_object(const char *content,
                                            const char *claim_id,
                                            const char **out_start,
                                            const char **out_end) {
    const char *content_end;
    const char *cursor;
    if (content == NULL || claim_id == NULL || out_start == NULL || out_end == NULL) {
        return -1;
    }
    content_end = content + strlen(content);
    cursor = content;
    while (cursor < content_end) {
        const char *id_value = paper_json_find_key_value_bounded(cursor, content_end, "id");
        char *parsed_id;
        const char *object_start;
        const char *object_end;
        if (id_value == NULL) {
            return 0;
        }
        parsed_id = paper_json_dup_string_value(id_value);
        if (parsed_id == NULL) {
            return -1;
        }
        if (strcmp(parsed_id, claim_id) != 0) {
            free(parsed_id);
            cursor = id_value + 1;
            continue;
        }
        free(parsed_id);
        object_start = id_value;
        while (object_start > content && *object_start != '{') {
            --object_start;
        }
        if (*object_start != '{') {
            return -1;
        }
        object_end = paper_json_value_end(object_start);
        if (object_end == NULL || object_end > content_end) {
            return -1;
        }
        *out_start = object_start;
        *out_end = object_end;
        return 1;
    }
    return 0;
}

static int paper_registry_read_claim_evidence(const char *content,
                                              const char *claim_id,
                                              PaperClaimRegistryEvidence *out) {
    const char *claim_start;
    const char *claim_end;
    const char *evidence_start;
    const char *evidence_end;
    int found;
    if (out == NULL) {
        return -1;
    }
    memset(out, 0, sizeof(*out));
    found = paper_registry_find_claim_object(content, claim_id, &claim_start, &claim_end);
    if (found <= 0) {
        return found;
    }
    evidence_start = paper_skip_ws(
        paper_json_find_key_value_bounded(claim_start, claim_end, "evidence"));
    if (evidence_start == NULL || *evidence_start != '{') {
        return -1;
    }
    evidence_end = paper_json_value_end(evidence_start);
    if (evidence_end == NULL || evidence_end > claim_end) {
        return -1;
    }
    out->source = paper_json_dup_string_value(
        paper_json_find_key_value_bounded(evidence_start, evidence_end, "source"));
    out->command = paper_json_dup_string_value(
        paper_json_find_key_value_bounded(evidence_start, evidence_end, "command"));
    out->freshness_prerequisite = paper_json_dup_string_value(
        paper_json_find_key_value_bounded(evidence_start,
                                          evidence_end,
                                          "freshness_prerequisite"));
    if (out->source == NULL || out->command == NULL || out->freshness_prerequisite == NULL) {
        paper_claim_registry_evidence_free(out);
        return -1;
    }
    return paper_claim_has_text(out->source) && paper_claim_has_text(out->command) ? 1 : -1;
}

static int paper_validate_claim_registry(const char *claims_path,
                                         char *error,
                                         size_t error_size) {
    char *content;
    size_t i;
    if (error != NULL && error_size > 0U) {
        error[0] = '\0';
    }
    content = paper_read_text_file(claims_path);
    if (content == NULL) {
        snprintf(error, error_size, "cannot read claim registry %s", claims_path);
        return -1;
    }
    for (i = 0U; i < PAPER_CLAIM_COUNT; ++i) {
        PaperClaimRegistryEvidence evidence;
        const PaperClaimSpec *spec = &PAPER_CLAIM_SPECS[i];
        int found = paper_registry_read_claim_evidence(content, spec->id, &evidence);
        if (found <= 0) {
            snprintf(error,
                     error_size,
                     "claim registry is missing evidence metadata for %s",
                     spec->id);
            free(content);
            return -1;
        }
        if (strcmp(evidence.source, spec->source) != 0) {
            snprintf(error,
                     error_size,
                     "claim %s source mismatch: registry=%s runtime=%s",
                     spec->id,
                     evidence.source,
                     spec->source);
            paper_claim_registry_evidence_free(&evidence);
            free(content);
            return -1;
        }
        if (paper_claim_has_text(spec->refresh_command) &&
            strcmp(evidence.command, spec->refresh_command) != 0) {
            snprintf(error,
                     error_size,
                     "claim %s refresh command mismatch: registry=%s runtime=%s",
                     spec->id,
                     evidence.command,
                     spec->refresh_command);
            paper_claim_registry_evidence_free(&evidence);
            free(content);
            return -1;
        }
        if (paper_claim_has_text(spec->freshness_prerequisite) &&
            strcmp(evidence.freshness_prerequisite, spec->freshness_prerequisite) != 0) {
            snprintf(error,
                     error_size,
                     "claim %s freshness prerequisite mismatch: registry=%s runtime=%s",
                     spec->id,
                     evidence.freshness_prerequisite,
                     spec->freshness_prerequisite);
            paper_claim_registry_evidence_free(&evidence);
            free(content);
            return -1;
        }
        paper_claim_registry_evidence_free(&evidence);
    }
    free(content);
    return 0;
}

static int paper_validate_empirical_percentage_literals(const char *paper_path,
                                                        char *error,
                                                        size_t error_size) {
    FILE *fp;
    char *line;
    long line_number = 0;
    if (error != NULL && error_size > 0U) {
        error[0] = '\0';
    }
    if (!paper_claim_has_text(paper_path)) {
        snprintf(error, error_size, "paper source path is empty");
        return -1;
    }
    fp = fopen(paper_path, "r");
    if (fp == NULL) {
        snprintf(error, error_size, "cannot read paper source %s", paper_path);
        return -1;
    }
    while ((line = paper_read_line_dynamic(fp)) != NULL) {
        const char *cursor;
        line_number += 1;
        if (strstr(line, "width:") != NULL || strstr(line, "height:") != NULL) {
            free(line);
            continue;
        }
        for (cursor = line; *cursor != '\0'; ++cursor) {
            const char *before;
            if (*cursor != '%') {
                continue;
            }
            before = cursor;
            while (before > line && isspace((unsigned char)before[-1])) {
                --before;
            }
            if (before > line && isdigit((unsigned char)before[-1])) {
                snprintf(error,
                         error_size,
                         "literal empirical percentage at %s:%ld; use a generated claim value",
                         paper_path,
                         line_number);
                free(line);
                fclose(fp);
                return -1;
            }
        }
        free(line);
    }
    fclose(fp);
    return 0;
}

static int paper_json_get_bool(const char *line, const char *key, int *out) {
    const char *cursor = paper_skip_ws(paper_json_find_key_value(line, key));
    if (cursor == NULL) {
        return 0;
    }
    if (strncmp(cursor, "true", 4U) == 0) {
        *out = 1;
        return 1;
    }
    if (strncmp(cursor, "false", 5U) == 0) {
        *out = 0;
        return 1;
    }
    return 0;
}

static void paper_line_list_free(PaperLineList *list) {
    size_t i;
    if (list == NULL) {
        return;
    }
    for (i = 0U; i < list->count; ++i) {
        free(list->items[i]);
    }
    free(list->items);
    memset(list, 0, sizeof(*list));
}

static int paper_line_list_append_owned(PaperLineList *list, char *line) {
    char **items;
    size_t capacity;
    if (list == NULL || line == NULL) {
        free(line);
        return -1;
    }
    if (list->count + 1U > list->capacity) {
        capacity = list->capacity == 0U ? 256U : list->capacity * 2U;
        items = (char **)realloc(list->items, capacity * sizeof(char *));
        if (items == NULL) {
            free(line);
            return -1;
        }
        list->items = items;
        list->capacity = capacity;
    }
    list->items[list->count++] = line;
    return 0;
}

static int paper_dyn_appendf(char **buffer,
                             size_t *used,
                             size_t *capacity,
                             const char *fmt,
                             ...) {
    va_list args;
    int needed;
    char *next;
    for (;;) {
        if (*buffer == NULL || *capacity == 0U) {
            *capacity = 256U;
            *used = 0U;
            *buffer = (char *)malloc(*capacity);
            if (*buffer == NULL) {
                return -1;
            }
            (*buffer)[0] = '\0';
        }
        va_start(args, fmt);
        needed = vsnprintf(*buffer + *used, *capacity - *used, fmt, args);
        va_end(args);
        if (needed < 0) {
            return -1;
        }
        if (*used + (size_t)needed < *capacity) {
            *used += (size_t)needed;
            return 0;
        }
        while (*used + (size_t)needed >= *capacity) {
            *capacity *= 2U;
        }
        next = (char *)realloc(*buffer, *capacity);
        if (next == NULL) {
            return -1;
        }
        *buffer = next;
    }
}

static int paper_trace_canonical_append(PaperLineList *lines,
                                        const NNParsedTraceRecord *record,
                                        const uint32_t *action_ids) {
    char *line = NULL;
    size_t used = 0U;
    size_t capacity = 0U;
    size_t i;
    int sensitive = fabs((double)record->selected_score_margin) <=
                    (double)record->rank_score_epsilon;
    if (paper_dyn_appendf(&line,
                          &used,
                          &capacity,
                          "{\"schema\":\"crlp.trace.semantic.v1\","
                          "\"source_schema\":\"crlp.trace.v1\","
                          "\"row_index\":%zu,"
                          "\"task\":\"%s\","
                          "\"trace_version\":%u,"
                          "\"action_id_contract\":\"%s\","
                          "\"state_hash\":\"%016llx\","
                          "\"feature_encoder_version\":%u,"
                          "\"candidate_count\":%zu,"
                          "\"action_ids\":[",
                          lines->count,
                          record->task_name,
                          (unsigned)record->trace_version,
                          record->action_id_contract,
                          (unsigned long long)record->state_hash,
                          (unsigned)record->feature_encoder_version,
                          record->candidate_count) != 0) {
        free(line);
        return -1;
    }
    for (i = 0U; i < record->candidate_count; ++i) {
        if (paper_dyn_appendf(&line,
                              &used,
                              &capacity,
                              "%s%u",
                              i == 0U ? "" : ",",
                              (unsigned)action_ids[i]) != 0) {
            free(line);
            return -1;
        }
    }
    if (paper_dyn_appendf(&line,
                          &used,
                          &capacity,
                          "],\"selected_action_id\":%u,"
                          "\"target_action_id\":%u,"
                          "\"verifier_result\":\"%s\","
                          "\"seed\":%llu,"
                          "\"ranking_policy\":\"%s\","
                          "\"rank_fp_sensitive\":%s}",
                          (unsigned)record->selected_action_id,
                          (unsigned)record->target_action_id,
                          record->verifier_result,
                          (unsigned long long)record->seed,
                          record->ranking_policy,
                          sensitive ? "true" : "false") != 0) {
        free(line);
        return -1;
    }
    return paper_line_list_append_owned(lines, line);
}

static uint64_t paper_trace_hash_lines(const PaperLineList *lines) {
    uint64_t hash = UINT64_C(1469598103934665603);
    size_t i;
    for (i = 0U; i < lines->count; ++i) {
        const unsigned char *p = (const unsigned char *)lines->items[i];
        while (*p != '\0') {
            hash ^= (uint64_t)(*p++);
            hash *= UINT64_C(1099511628211);
        }
        hash ^= (uint64_t)'\n';
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static int paper_trace_lines_equal(const PaperLineList *a, const PaperLineList *b) {
    size_t i;
    if (a->count != b->count) {
        return 0;
    }
    for (i = 0U; i < a->count; ++i) {
        if (strcmp(a->items[i], b->items[i]) != 0) {
            return 0;
        }
    }
    return 1;
}

static int paper_trace_write_canonical_file(const char *path,
                                            const PaperLineList *lines) {
    FILE *fp;
    size_t i;
    if (path == NULL || path[0] == '\0') {
        return 0;
    }
    if (paper_mkdir_parent(path) != 0) {
        return -1;
    }
    fp = fopen(path, "w");
    if (fp == NULL) {
        return -1;
    }
    for (i = 0U; i < lines->count; ++i) {
        if (fprintf(fp, "%s\n", lines->items[i]) < 0) {
            fclose(fp);
            return -1;
        }
    }
    if (fclose(fp) != 0) {
        return -1;
    }
    return 0;
}

static int paper_trace_audit_one(const char *path,
                                 PaperLineList *canonical_lines,
                                 long *row_count_out,
                                 long *invalid_rows_out,
                                 long *sensitive_rows_out,
                                 long *logged_mismatch_out,
                                 double *min_margin_out,
                                 double *max_epsilon_out) {
    FILE *fp;
    char *line;
    long row_count = 0;
    long invalid_rows = 0;
    long sensitive_rows = 0;
    long logged_mismatch = 0;
    double min_margin = 0.0;
    double max_epsilon = 0.0;
    int have_margin = 0;
    fp = fopen(path, "r");
    if (fp == NULL) {
        return -1;
    }
    while ((line = paper_read_line_dynamic(fp)) != NULL) {
        long candidate_count_long = 0;
        size_t candidate_count;
        uint32_t *action_ids = NULL;
        nn_real *scores = NULL;
        nn_real *probabilities = NULL;
        NNParsedTraceRecord record;
        int logged_sensitive;
        int computed_sensitive;
        paper_chomp(line);
        if (line[0] != '{') {
            free(line);
            continue;
        }
        if (!paper_json_get_long(line, "candidate_count", &candidate_count_long) ||
            candidate_count_long <= 0) {
            ++invalid_rows;
            free(line);
            continue;
        }
        candidate_count = (size_t)candidate_count_long;
        action_ids = (uint32_t *)calloc(candidate_count, sizeof(uint32_t));
        scores = (nn_real *)calloc(candidate_count, sizeof(nn_real));
        probabilities = (nn_real *)calloc(candidate_count, sizeof(nn_real));
        if (action_ids == NULL || scores == NULL || probabilities == NULL ||
            !nn_trace_parse_jsonl(line,
                                  &record,
                                  action_ids,
                                  scores,
                                  probabilities,
                                  candidate_count)) {
            ++invalid_rows;
            free(action_ids);
            free(scores);
            free(probabilities);
            free(line);
            continue;
        }
        computed_sensitive = fabs((double)record.selected_score_margin) <=
                             (double)record.rank_score_epsilon;
        if (computed_sensitive) {
            ++sensitive_rows;
        }
        if (paper_json_get_bool(line, "rank_fp_sensitive", &logged_sensitive) &&
            logged_sensitive != computed_sensitive) {
            ++logged_mismatch;
        }
        if (!have_margin || (double)record.selected_score_margin < min_margin) {
            min_margin = (double)record.selected_score_margin;
            have_margin = 1;
        }
        if ((double)record.rank_score_epsilon > max_epsilon) {
            max_epsilon = (double)record.rank_score_epsilon;
        }
        if (paper_trace_canonical_append(canonical_lines, &record, action_ids) != 0) {
            free(action_ids);
            free(scores);
            free(probabilities);
            free(line);
            fclose(fp);
            return -1;
        }
        ++row_count;
        free(action_ids);
        free(scores);
        free(probabilities);
        free(line);
    }
    if (fclose(fp) != 0) {
        return -1;
    }
    *row_count_out = row_count;
    *invalid_rows_out = invalid_rows;
    *sensitive_rows_out = sensitive_rows;
    *logged_mismatch_out = logged_mismatch;
    *min_margin_out = have_margin ? min_margin : 0.0;
    *max_epsilon_out = max_epsilon;
    return 0;
}

int nn_paper_write_trace_determinism(const char *const *trace_paths,
                                     size_t trace_path_count,
                                     const char *output_path,
                                     const char *canonical_output_path,
                                     int gate) {
    FILE *fp;
    PaperLineList *canonical = NULL;
    long *row_counts = NULL;
    uint64_t *hashes = NULL;
    long total_invalid = 0;
    long total_sensitive = 0;
    long total_logged_mismatch = 0;
    int all_equal = 1;
    int row_counts_equal = 1;
    int status = -1;
    size_t i;
    if (trace_paths == NULL || trace_path_count == 0U || output_path == NULL) {
        return -1;
    }
    if (canonical_output_path != NULL && canonical_output_path[0] != '\0' &&
        trace_path_count != 1U) {
        return -1;
    }
    canonical = (PaperLineList *)calloc(trace_path_count, sizeof(PaperLineList));
    row_counts = (long *)calloc(trace_path_count, sizeof(long));
    hashes = (uint64_t *)calloc(trace_path_count, sizeof(uint64_t));
    if (canonical == NULL || row_counts == NULL || hashes == NULL ||
        paper_mkdir_parent(output_path) != 0) {
        goto cleanup;
    }
    fp = fopen(output_path, "w");
    if (fp == NULL) {
        goto cleanup;
    }
    for (i = 0U; i < trace_path_count; ++i) {
        long invalid = 0;
        long sensitive = 0;
        long logged_mismatch = 0;
        double min_margin = 0.0;
        double max_epsilon = 0.0;
        if (paper_trace_audit_one(trace_paths[i],
                                  &canonical[i],
                                  &row_counts[i],
                                  &invalid,
                                  &sensitive,
                                  &logged_mismatch,
                                  &min_margin,
                                  &max_epsilon) != 0) {
            fclose(fp);
            goto cleanup;
        }
        hashes[i] = paper_trace_hash_lines(&canonical[i]);
        total_invalid += invalid;
        total_sensitive += sensitive;
        total_logged_mismatch += logged_mismatch;
        fputs("{\"schema\":\"crlp.trace_determinism_audit.v1\",", fp);
        fputs("\"trace_path\":", fp);
        paper_write_json_string(fp, trace_paths[i]);
        fprintf(fp,
                ",\"row_count\":%ld,"
                "\"semantic_hash\":\"%016llx\","
                "\"semantic_schema\":\"crlp.trace.semantic.v1\","
                "\"rank_fp_sensitive_rows\":%ld,"
                "\"rank_fp_sensitive_rate\":%.15g,"
                "\"rank_fp_sensitive_logged_mismatches\":%ld,"
                "\"negative_margin_rows\":%ld,"
                "\"score_margin_min\":%.15g,"
                "\"rank_score_epsilon_max\":%.15g,"
                "\"invalid_rows\":%ld}\n",
                row_counts[i],
                (unsigned long long)hashes[i],
                sensitive,
                row_counts[i] == 0 ? 0.0 : (double)sensitive / (double)row_counts[i],
                logged_mismatch,
                min_margin < 0.0 ? 1L : 0L,
                min_margin,
                max_epsilon,
                invalid);
    }
    if (trace_path_count > 1U) {
        for (i = 1U; i < trace_path_count; ++i) {
            if (!paper_trace_lines_equal(&canonical[0], &canonical[i])) {
                all_equal = 0;
            }
            if (row_counts[i] != row_counts[0]) {
                row_counts_equal = 0;
            }
        }
        fputs("{\"schema\":\"crlp.trace_determinism_compare.v1\",", fp);
        fputs("\"baseline_trace_path\":", fp);
        paper_write_json_string(fp, trace_paths[0]);
        fputs(",\"trace_paths\":[", fp);
        for (i = 0U; i < trace_path_count; ++i) {
            if (i > 0U) {
                fputc(',', fp);
            }
            paper_write_json_string(fp, trace_paths[i]);
        }
        fputs("],\"semantic_hashes\":{", fp);
        for (i = 0U; i < trace_path_count; ++i) {
            if (i > 0U) {
                fputc(',', fp);
            }
            paper_write_json_string(fp, trace_paths[i]);
            fprintf(fp, ":\"%016llx\"", (unsigned long long)hashes[i]);
        }
        fputs("},\"row_counts\":{", fp);
        for (i = 0U; i < trace_path_count; ++i) {
            if (i > 0U) {
                fputc(',', fp);
            }
            paper_write_json_string(fp, trace_paths[i]);
            fprintf(fp, ":%ld", row_counts[i]);
        }
        fprintf(fp,
                "},\"all_semantic_hashes_equal\":%s,"
                "\"row_counts_equal\":%s,"
                "\"differing_paths\":[]}\n",
                all_equal ? "true" : "false",
                row_counts_equal ? "true" : "false");
    }
    if (gate) {
        int passed = total_invalid == 0 && total_sensitive == 0 &&
                     total_logged_mismatch == 0 && all_equal && row_counts_equal &&
                     row_counts[0] > 0;
        fprintf(fp,
                "{\"schema\":\"crlp.trace_determinism_gate.v1\","
                "\"passed\":%s,\"failure_count\":%d}\n",
                passed ? "true" : "false",
                passed ? 0 : 1);
        if (!passed) {
            fclose(fp);
            goto cleanup;
        }
    }
    if (fclose(fp) != 0) {
        goto cleanup;
    }
    if (canonical_output_path != NULL && canonical_output_path[0] != '\0' &&
        paper_trace_write_canonical_file(canonical_output_path, &canonical[0]) != 0) {
        goto cleanup;
    }
    status = 0;
cleanup:
    if (canonical != NULL) {
        for (i = 0U; i < trace_path_count; ++i) {
            paper_line_list_free(&canonical[i]);
        }
    }
    free(canonical);
    free(row_counts);
    free(hashes);
    return status;
}

static int paper_json_string_equals(const char *line,
                                    const char *key,
                                    const char *expected) {
    char *value = paper_json_dup_key_string(line, key);
    int equal;
    if (value == NULL) {
        return 0;
    }
    equal = strcmp(value, expected) == 0;
    free(value);
    return equal;
}

static int paper_json_get_nested_double(const char *line,
                                        const char *outer_key,
                                        const char *inner_key,
                                        double *out) {
    const char *cursor = paper_json_find_key_value(line, outer_key);
    if (cursor == NULL) {
        return 0;
    }
    return paper_json_get_double(cursor, inner_key, out);
}

static int paper_read_first_jsonl(const char *path, char **out_line) {
    FILE *fp;
    char *line;
    fp = fopen(path, "r");
    if (fp == NULL) {
        return -1;
    }
    while ((line = paper_read_line_dynamic(fp)) != NULL) {
        paper_chomp(line);
        if (line[0] != '\0') {
            fclose(fp);
            *out_line = line;
            return 1;
        }
        free(line);
    }
    fclose(fp);
    return 0;
}

static long paper_count_jsonl_rows(const char *path) {
    FILE *fp;
    char *line;
    long count = 0;
    fp = fopen(path, "r");
    if (fp == NULL) {
        return -1;
    }
    while ((line = paper_read_line_dynamic(fp)) != NULL) {
        paper_chomp(line);
        if (line[0] != '\0') {
            ++count;
        }
        free(line);
    }
    fclose(fp);
    return count;
}

static char *paper_trim_in_place(char *value) {
    char *end;
    while (*value != '\0' && isspace((unsigned char)*value)) {
        ++value;
    }
    end = value + strlen(value);
    while (end > value && isspace((unsigned char)end[-1])) {
        *--end = '\0';
    }
    return value;
}

static size_t paper_csv_split_in_place(char *line, char **fields, size_t max_fields) {
    char *read;
    char *write;
    int in_quote = 0;
    size_t count = 0U;
    if (max_fields == 0U) {
        return 0U;
    }
    fields[count++] = line;
    write = line;
    for (read = line; *read != '\0'; ++read) {
        if (*read == '"') {
            in_quote = !in_quote;
            continue;
        }
        if (*read == ',' && !in_quote) {
            *write++ = '\0';
            if (count < max_fields) {
                fields[count++] = write;
            }
            continue;
        }
        *write++ = *read;
    }
    *write = '\0';
    return count;
}

static void paper_csv_record_copy(PaperCsvRecord *record,
                                  char **headers,
                                  size_t header_count,
                                  char **values,
                                  size_t value_count) {
    size_t i;
    record->count = header_count < 80U ? header_count : 80U;
    for (i = 0U; i < record->count; ++i) {
        char *header = paper_trim_in_place(headers[i]);
        char *value = i < value_count ? paper_trim_in_place(values[i]) : "";
        snprintf(record->headers[i], sizeof(record->headers[i]), "%s", header);
        snprintf(record->values[i], sizeof(record->values[i]), "%s", value);
    }
}

static const char *paper_csv_record_get(const PaperCsvRecord *record,
                                        const char *name) {
    size_t i;
    for (i = 0U; i < record->count; ++i) {
        if (strcmp(record->headers[i], name) == 0) {
            return record->values[i];
        }
    }
    return NULL;
}

static int paper_csv_record_get_double(const PaperCsvRecord *record,
                                       const char *name,
                                       double *out) {
    const char *value = paper_csv_record_get(record, name);
    char *end = NULL;
    double parsed;
    if (value == NULL || value[0] == '\0') {
        return 0;
    }
    errno = 0;
    parsed = strtod(value, &end);
    if (end == value || errno == ERANGE || !isfinite(parsed)) {
        return 0;
    }
    *out = parsed;
    return 1;
}

static int paper_csv_record_get_long(const PaperCsvRecord *record,
                                     const char *name,
                                     long *out) {
    const char *value = paper_csv_record_get(record, name);
    char *end = NULL;
    long parsed;
    if (value == NULL || value[0] == '\0') {
        return 0;
    }
    errno = 0;
    parsed = strtol(value, &end, 10);
    if (end == value || errno == ERANGE) {
        return 0;
    }
    *out = parsed;
    return 1;
}

static int paper_csv_load_matching_row(const char *path,
                                       const char **pred_cols,
                                       const char **pred_vals,
                                       size_t pred_count,
                                       PaperCsvRecord *record) {
    FILE *fp;
    char *header_line;
    char *line;
    char *headers[80];
    char *values[80];
    size_t header_count;
    fp = fopen(path, "r");
    if (fp == NULL) {
        return -1;
    }
    header_line = paper_read_line_dynamic(fp);
    if (header_line == NULL) {
        fclose(fp);
        return 0;
    }
    paper_chomp(header_line);
    header_count = paper_csv_split_in_place(header_line, headers, 80U);
    while ((line = paper_read_line_dynamic(fp)) != NULL) {
        size_t value_count;
        size_t i;
        int matched = 1;
        paper_chomp(line);
        if (line[0] == '\0') {
            free(line);
            continue;
        }
        value_count = paper_csv_split_in_place(line, values, 80U);
        paper_csv_record_copy(record, headers, header_count, values, value_count);
        for (i = 0U; i < pred_count; ++i) {
            const char *actual = paper_csv_record_get(record, pred_cols[i]);
            if (actual == NULL || strcmp(actual, pred_vals[i]) != 0) {
                matched = 0;
                break;
            }
        }
        free(line);
        if (matched) {
            free(header_line);
            fclose(fp);
            return 1;
        }
    }
    free(header_line);
    fclose(fp);
    return 0;
}

static long paper_csv_count_rows(const char *path) {
    FILE *fp;
    char *line;
    long count = 0;
    int saw_header = 0;
    fp = fopen(path, "r");
    if (fp == NULL) {
        return -1;
    }
    while ((line = paper_read_line_dynamic(fp)) != NULL) {
        paper_chomp(line);
        if (line[0] != '\0') {
            if (saw_header) {
                ++count;
            } else {
                saw_header = 1;
            }
        }
        free(line);
    }
    fclose(fp);
    return count;
}

static void paper_claim_slug(const char *id, char out[96]) {
    size_t i;
    size_t j = 0U;
    for (i = 0U; id[i] != '\0' && j + 1U < 96U; ++i) {
        unsigned char ch = (unsigned char)id[i];
        out[j++] = isalnum(ch) ? (char)ch : '_';
    }
    out[j] = '\0';
}

static void paper_write_typ_string(FILE *fp, const char *value) {
    const unsigned char *cursor = (const unsigned char *)(value == NULL ? "" : value);
    fputc('"', fp);
    while (*cursor != '\0') {
        if (*cursor == '"' || *cursor == '\\') {
            fputc('\\', fp);
            fputc((int)*cursor, fp);
        } else if (*cursor == '\n') {
            fputs("\\n", fp);
        } else if (*cursor == '\r') {
            fputs("\\r", fp);
        } else if (*cursor == '\t') {
            fputs("\\t", fp);
        } else {
            fputc((int)*cursor, fp);
        }
        ++cursor;
    }
    fputc('"', fp);
}

static void paper_write_typ_float(FILE *fp, double value) {
    if (!isfinite(value)) {
        fputs("none", fp);
    } else {
        fprintf(fp, "%.15g", value);
    }
}

static int paper_write_text_file(const char *path, const char *text) {
    FILE *fp;
    if (paper_mkdir_parent(path) != 0) {
        return -1;
    }
    fp = fopen(path, "w");
    if (fp == NULL) {
        return -1;
    }
    fputs(text, fp);
    fclose(fp);
    return 0;
}
