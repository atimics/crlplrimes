#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static void mkdir_p(const char *path) {
    char tmp[1024];
    snprintf(tmp, sizeof(tmp), "%s", path);
    for (char *p = tmp + 1; *p; p++) {
        if (*p == '/') { *p = 0; mkdir(tmp, 0755); *p = '/'; }
    }
    mkdir(tmp, 0755);
}

static int run_cmd(const char *cmd) {
    printf("  %s\n", cmd);
    fflush(stdout);
    int rc = system(cmd);
    if (rc != 0) fprintf(stderr, "  FAILED (%d)\n", rc);
    return rc;
}

static void write_ok(FILE *fp, const char *surface, const char *trace) {
    fprintf(fp, "{\"schema\":\"crlp.substrate_experiment.v1\",\"surface\":\"%s\",\"status\":\"completed\",\"trace\":\"%s\"}\n", surface, trace);
}
static void write_fail(FILE *fp, const char *surface) {
    fprintf(fp, "{\"schema\":\"crlp.substrate_experiment.v1\",\"surface\":\"%s\",\"status\":\"failed\"}\n", surface);
}

static void write_family_sweep(FILE *fp, const char *family, int ok, int seed) {
    fprintf(fp, "{\"schema\":\"crlp.reason_family_sweep.v1\",\"family\":\"%s\",\"status\":\"%s\",\"seed\":%d}\n",
            family, ok ? "completed" : "failed", seed);
}

/* Extract an integer JSON field value from a flat JSON object line. */
static int json_extract_int(const char *line, const char *field) {
    char key[256];
    snprintf(key, sizeof(key), "\"%s\":", field);
    const char *p = strstr(line, key);
    if (!p) {
        snprintf(key, sizeof(key), "\"%s\": ", field);
        p = strstr(line, key);
    }
    if (!p) return -1;
    p += strlen(key);
    return atoi(p);
}

/* Extract a double JSON field value. */
static double json_extract_double(const char *line, const char *field) {
    char key[256];
    snprintf(key, sizeof(key), "\"%s\":", field);
    const char *p = strstr(line, key);
    if (!p) {
        snprintf(key, sizeof(key), "\"%s\": ", field);
        p = strstr(line, key);
    }
    if (!p) return -1.0;
    p += strlen(key);
    return atof(p);
}

/* Extract a double from a bracket-summary line like "[grid_path] utility=0.6987 ...". */
static double bracket_extract_double(const char *line, const char *key) {
    char fullkey[128];
    snprintf(fullkey, sizeof(fullkey), " %s=", key);
    const char *p = strstr(line, fullkey);
    if (!p) return -1.0;
    p += strlen(fullkey);
    return atof(p);
}

/* Extract a string JSON field value. Writes into out_buf (max out_cap). Returns 0 on success. */
static int json_extract_string(const char *line, const char *field, char *out_buf, size_t out_cap) {
    char key[256];
    snprintf(key, sizeof(key), "\"%s\":\"", field);
    const char *p = strstr(line, key);
    if (!p) {
        snprintf(key, sizeof(key), "\"%s\": \"", field);
        p = strstr(line, key);
    }
    if (!p) return -1;
    p += strlen(key);
    const char *end = strchr(p, '"');
    if (!end) return -1;
    size_t len = (size_t)(end - p);
    if (len >= out_cap) len = out_cap - 1;
    memcpy(out_buf, p, len);
    out_buf[len] = '\0';
    return 0;
}

/* Write a compounding evidence row. */
static void write_compounding_evidence(FILE *fp,
                                       const char *domain,
                                       int iteration,
                                       int certified,
                                       int template_id,
                                       const char *reason_code,
                                       int false_rejections,
                                       double utility,
                                       const char *provenance) {
    fprintf(fp,
        "{\"schema\":\"crlp.compounding.evidence.v1\","
        "\"domain\":\"%s\","
        "\"iteration\":%d,"
        "\"certified\":%s,"
        "\"template_id\":%d,"
        "\"reason_code\":\"%s\","
        "\"false_rejections\":%d,"
        "\"utility\":%.6f,"
        "\"provenance\":\"%s\"}\n",
        domain, iteration,
        certified ? "true" : "false",
        template_id,
        reason_code ? reason_code : "",
        false_rejections,
        utility,
        provenance ? provenance : "none");
    fflush(fp);
}

/* Write a compounding summary row. */
static void write_compounding_summary(FILE *fp,
                                      int iteration,
                                      const char *domain,
                                      int new_certified,
                                      int new_rejected,
                                      double utility_mean,
                                      double false_rejection_rate) {
    fprintf(fp,
        "{\"schema\":\"crlp.compounding.summary.v1\","
        "\"iteration\":%d,"
        "\"domain\":\"%s\","
        "\"new_certified\":%d,"
        "\"new_rejected\":%d,"
        "\"utility_mean\":%.6f,"
        "\"false_rejection_rate\":%.6f}\n",
        iteration, domain,
        new_certified, new_rejected,
        utility_mean, false_rejection_rate);
    fflush(fp);
}

/* ---- rejected-candidate-learning ---- */
static int run_rejected_candidate_learning(const char *bd, const char *od) {
    mkdir_p(od);

    char prime[1024], mod[1024], dc[1024];
    snprintf(prime, sizeof(prime), "%s/prime_rule_search", bd);
    snprintf(mod, sizeof(mod), "%s/modular_gate_search", bd);
    snprintf(dc, sizeof(dc), "%s/delayed_claim", bd);

    char rp[1024], sp[1024], pp[1024], fsp[1024];
    snprintf(rp, sizeof(rp), "%s/reason_learning.jsonl", od);
    snprintf(sp, sizeof(sp), "%s/safety_report.jsonl", od);
    snprintf(pp, sizeof(pp), "%s/split_sweep.jsonl", od);
    snprintf(fsp, sizeof(fsp), "%s/reason_family_sweep.jsonl", od);

    FILE *rf = fopen(rp, "w");
    FILE *sf = fopen(sp, "w");
    FILE *pf = fopen(pp, "w");
    FILE *ff = fopen(fsp, "w");
    if (!rf || !sf || !pf || !ff) {
        perror(od);
        if (rf) fclose(rf); if (sf) fclose(sf);
        if (pf) fclose(pf); if (ff) fclose(ff);
        return 1;
    }

    /* Surface 1: prime modular filters -- closed-loop proposer */
    {
        char cmd[2048], tr[1024];
        snprintf(tr, sizeof(tr), "%s/traces/prime_rule_search.jsonl", bd);
        snprintf(cmd, sizeof(cmd),
            "%s --candidate-only --moduli-range 3:31 --max-rules 4 "
            "--proposer closed-loop --proposer-seed 20240511 "
            "--closed-loop-hidden 64x32 "
            "--trace %s --save %s/prime_reason_learning.crlprule",
            prime, tr, od);
        int ok = (run_cmd(cmd) == 0);
        if (ok) write_ok(rf, "prime_modular_filters", tr);
        else write_fail(rf, "prime_modular_filters");
        fflush(rf);
        write_family_sweep(ff, "prime_modular_filters", ok, 20240511);
    }

    /* Surface 2: modular gate families */
    {
        char cmd[2048], tr[1024];
        snprintf(tr, sizeof(tr), "%s/traces/modular_gate_search.jsonl", bd);
        snprintf(cmd, sizeof(cmd),
            "%s --task fibonacci_lucas "
            "--max-rules 8 --selection-seed 20240511 "
            "--trace %s",
            mod, tr);
        int ok = (run_cmd(cmd) == 0);
        if (ok) write_ok(rf, "modular_gate_search", tr);
        else write_fail(rf, "modular_gate_search");
        fflush(rf);
        write_family_sweep(ff, "modular_gate_search", ok, 20240511);
    }

    /* Surface 3: delayed-claim threshold gates */
    {
        char cmd[2048];
        snprintf(cmd, sizeof(cmd),
            "%s --rule-search %s/delayed_claim_v2.nnckpt "
            "--rule-family direct --max-rules 8 --proposer all "
            "--proposer-seed 20240511",
            dc, bd);
        int ok = (run_cmd(cmd) == 0);
        if (ok) write_ok(rf, "delayed_claim_threshold", "");
        else write_fail(rf, "delayed_claim_threshold");
        fflush(rf);
        write_family_sweep(ff, "delayed_claim_threshold", ok, 20240511);
    }

    /* Safety surface: squarefree gate */
    {
        char cmd[2048];
        snprintf(cmd, sizeof(cmd),
            "%s --task kfree "
            "--max-rules 4 --selection-seed 20240511 "
            "--trace %s/safety_modular.jsonl",
            mod, od);
        if (run_cmd(cmd) == 0) write_ok(sf, "kfree_safety", "");
        else write_fail(sf, "kfree_safety");
        fflush(sf);
    }

    /* Split sweep: conditional vs unconditional grammar */
    for (int s = 0; s < 2; s++) {
        int seed = 20240511 + s * 2;
        char cmd[2048];
        snprintf(cmd, sizeof(cmd),
            "%s --candidate-only --moduli-range 3:31 --max-rules 4 "
            "--proposer closed-loop --proposer-seed %d "
            "--closed-loop-hidden 64x32 "
            "%s "
            "--trace %s/split_%s.jsonl --save %s/split_%s.crlprule",
            prime, seed,
            s == 0 ? "--unconditional-only" : "",
            od, s == 0 ? "unconditional" : "conditional",
            od, s == 0 ? "unconditional" : "conditional");
        if (run_cmd(cmd) == 0)
            fprintf(pf, "{\"schema\":\"crlp.split_sweep.v1\",\"grammar\":\"%s\",\"seed\":%d,\"status\":\"completed\"}\n",
                    s == 0 ? "unconditional" : "conditional", seed);
        else
            fprintf(pf, "{\"schema\":\"crlp.split_sweep.v1\",\"grammar\":\"%s\",\"seed\":%d,\"status\":\"failed\"}\n",
                    s == 0 ? "unconditional" : "conditional", seed);
        fflush(pf);
    }

    fclose(rf); fclose(sf); fclose(pf); fclose(ff);
    printf("  wrote %s, %s, %s, %s\n", rp, sp, pp, fsp);
    return 0;
}

/* ---- rejected-candidate-seed-audit ---- */
static int run_seed_audit(const char *bd, const char *od) {
    mkdir_p(od);

    char prime[1024], mod[1024], dc[1024];
    snprintf(prime, sizeof(prime), "%s/prime_rule_search", bd);
    snprintf(mod, sizeof(mod), "%s/modular_gate_search", bd);
    snprintf(dc, sizeof(dc), "%s/delayed_claim", bd);

    char sp[1024], ip[1024], fp[1024];
    snprintf(sp, sizeof(sp), "%s/seed_stability.jsonl", od);
    snprintf(ip, sizeof(ip), "%s/seed_interval_summary.jsonl", od);
    snprintf(fp, sizeof(fp), "%s/reason_family_seed_interval.jsonl", od);

    FILE *sf = fopen(sp, "w"), *ifp = fopen(ip, "w"), *ff = fopen(fp, "w");
    if (!sf || !ifp || !ff) { perror(od); return 1; }

    int seeds[] = {20240511, 20240512, 20240513, 20240514, 20240515};
    int nseeds = 5;

    const char *families[] = {"prime_modular_filters", "modular_gate_search", "delayed_claim_threshold"};

    int any_failure = 0;

    for (int f = 0; f < 3; f++) {
        int family_ok = 0;
        for (int i = 0; i < nseeds; i++) {
            char cmd[2048];
            const char *family = families[f];
            if (f == 0) {
                snprintf(cmd, sizeof(cmd),
                    "%s --candidate-only --moduli-range 3:31 --max-rules 4 "
                    "--proposer closed-loop --proposer-seed %d "
                    "--closed-loop-hidden 64x32 "
                    "--trace %s/seed_%s_%d.jsonl --save %s/seed_%s_%d.crlprule",
                    prime, seeds[i], od, family, seeds[i], od, family, seeds[i]);
            } else if (f == 1) {
                snprintf(cmd, sizeof(cmd),
                    "%s --task fibonacci_lucas "
                    "--max-rules 8 --selection-seed %d "
                    "--trace %s/seed_%s_%d.jsonl",
                    mod, seeds[i], od, family, seeds[i]);
            } else {
                snprintf(cmd, sizeof(cmd),
                    "%s --rule-search %s/delayed_claim_v2.nnckpt "
                    "--rule-family direct --max-rules 8 --proposer all "
                    "--proposer-seed %d",
                    dc, bd, seeds[i]);
            }
            int rc = run_cmd(cmd);
            fprintf(sf, "{\"schema\":\"crlp.seed_audit.v1\",\"family\":\"%s\",\"seed\":%d,\"status\":\"%s\"}\n",
                    family, seeds[i], rc == 0 ? "completed" : "failed");
            fprintf(ff, "{\"schema\":\"crlp.reason_family_seed_interval.v1\",\"family\":\"%s\",\"seed\":%d,\"status\":\"%s\"}\n",
                    family, seeds[i], rc == 0 ? "completed" : "failed");
            if (rc == 0) family_ok++;
            else any_failure = 1;
            fflush(sf);
            fflush(ff);
        }
        fprintf(ifp, "{\"schema\":\"crlp.seed_interval_summary.v1\",\"family\":\"%s\",\"seeds_tested\":%d,\"seeds_passed\":%d,\"min_seed\":%d,\"max_seed\":%d}\n",
                families[f], nseeds, family_ok, seeds[0], seeds[nseeds - 1]);
        fflush(ifp);
    }

    fclose(sf); fclose(ifp); fclose(ff);
    printf("  wrote %s, %s, %s\n", sp, ip, fp);
    return any_failure;
}

/* ---- hypothesis-library-split-sweep ---- */
static int run_split_sweep(const char *bd, const char *od) {
    mkdir_p(od);

    char arc[1024];
    snprintf(arc, sizeof(arc), "%s/arc_agi", bd);

    char sweep_path[1024], heldout_path[1024], stdout_path[1024];
    snprintf(sweep_path, sizeof(sweep_path), "%s/split_sweep.jsonl", od);
    snprintf(heldout_path, sizeof(heldout_path), "%s/heldout_result.jsonl", od);
    snprintf(stdout_path, sizeof(stdout_path), "%s/split_stdout.txt", od);

    FILE *sf = fopen(sweep_path, "w");
    FILE *hf = fopen(heldout_path, "w");
    if (!sf || !hf) { perror(od); return 1; }

    for (int i = 0; i < 3; i++) {
        char cmd[2048];
        snprintf(cmd, sizeof(cmd),
            "%s --task-dir tests/fixtures/arc_hypothesis --max-tasks 14 "
            "--program-budget-grid --loader-summary-only "
            "--induction --min-induction-evidence 13 "
            "--seed %d "
            "> %s 2>&1",
            arc, 20240511 + i * 7, stdout_path);

        int rc = run_cmd(cmd);
        fprintf(sf, "{\"schema\":\"crlp.split_sweep.v1\",\"split\":%d,\"seed\":%d,\"status\":\"%s\"}\n",
                i, 20240511 + i * 7, rc == 0 ? "completed" : "failed");
        fflush(sf);

        if (rc == 0) {
            FILE *outf = fopen(stdout_path, "r");
            if (outf) {
                char line[4096];
                while (fgets(line, sizeof(line), outf)) {
                    if (strstr(line, "\"crlp.arc_loader.summary.v1\"")) {
                        int test_exact  = json_extract_int(line, "test_exact_tasks");
                        int certified   = json_extract_int(line, "certified_tasks");
                        int scored      = json_extract_int(line, "scored_tasks");
                        int loaded      = json_extract_int(line, "loaded_tasks");
                        double cell_acc = json_extract_double(line, "cell_accuracy");
                        fprintf(hf,
                            "{\"schema\":\"crlp.heldout_result.v1\",\"split\":%d,\"seed\":%d,"
                            "\"loaded_tasks\":%d,\"scored_tasks\":%d,"
                            "\"test_exact_tasks\":%d,\"certified_tasks\":%d,"
                            "\"cell_accuracy\":%.8f}\n",
                            i, 20240511 + i * 7,
                            loaded, scored, test_exact, certified, cell_acc);
                        break;
                    }
                }
                fclose(outf);
            }
        }
        fflush(hf);
    }

    fclose(sf); fclose(hf);
    printf("  wrote %s, %s\n", sweep_path, heldout_path);
    return 0;
}

/* ─── Compounding Loop ──────────────────────────────────────────────── */

/* Count evidence rows for a domain+iteration for summary metrics. */
static void count_evidence_for_summary(FILE *ef, const char *domain, int iteration,
                                       int *certified_out, int *rejected_out,
                                       double *utility_sum_out) {
    /* ef is the evidence file already open for reading; we scan it. */
    rewind(ef);
    char line[4096];
    int cert = 0, rej = 0;
    double util_sum = 0.0;
    char iter_key[64];
    snprintf(iter_key, sizeof(iter_key), "\"iteration\":%d", iteration);
    char dom_key[256];
    snprintf(dom_key, sizeof(dom_key), "\"domain\":\"%s\"", domain);

    while (fgets(line, sizeof(line), ef)) {
        if (!strstr(line, iter_key)) continue;
        if (!strstr(line, dom_key)) continue;
        if (strstr(line, "\"certified\":true")) {
            cert++;
            double u = json_extract_double(line, "utility");
            if (u >= 0.0) util_sum += u;
        } else if (strstr(line, "\"certified\":false")) {
            rej++;
        }
    }
    *certified_out = cert;
    *rejected_out = rej;
    *utility_sum_out = util_sum;
}

/*
 * Build a hypothesis-library-prior TSV file from compounding evidence rows.
 * Reads the evidence file, extracts certified rows for ARC domain across all
 * iterations up to `max_iteration`, and writes them in the format expected
 * by arc_agi's --hypothesis-library-prior flag.
 */
static int build_arc_prior_from_evidence(const char *evidence_path,
                                         const char *prior_path,
                                         int max_iteration) {
    FILE *ef = fopen(evidence_path, "r");
    if (!ef) {
        fprintf(stderr, "build_arc_prior: cannot open evidence %s\n", evidence_path);
        return 0;
    }
    FILE *pf = fopen(prior_path, "w");
    if (!pf) {
        fprintf(stderr, "build_arc_prior: cannot open prior %s\n", prior_path);
        fclose(ef);
        return 0;
    }
    fprintf(pf, "schema_version\t1\n");

    /* We write template_prior rows for each certified template_id seen.
       Since evidence rows may share template_ids, we track distinct ones. */
    #define MAX_DISTINCT_TEMPLATES 256
    uint32_t tids[MAX_DISTINCT_TEMPLATES];
    size_t proposals[MAX_DISTINCT_TEMPLATES];
    size_t certified_counts[MAX_DISTINCT_TEMPLATES];
    size_t distinct = 0;

    char line[4096];
    while (fgets(line, sizeof(line), ef)) {
        if (!strstr(line, "\"schema\":\"crlp.compounding.evidence.v1\"")) continue;
        if (!strstr(line, "\"domain\":\"arc_source\"") &&
            !strstr(line, "\"domain\":\"arc_heldout\"") &&
            !strstr(line, "\"domain\":\"arc\"")) continue;
        if (!strstr(line, "\"certified\":true")) continue;
        int iter = json_extract_int(line, "iteration");
        if (iter < 0 || iter > max_iteration) continue;
        int tid = json_extract_int(line, "template_id");
        if (tid < 0) continue;

        /* Find or add this template_id */
        size_t found = distinct;
        for (size_t j = 0; j < distinct; j++) {
            if (tids[j] == (uint32_t)tid) { found = j; break; }
        }
        if (found == distinct) {
            if (distinct >= MAX_DISTINCT_TEMPLATES) continue;
            tids[distinct] = (uint32_t)tid;
            proposals[distinct] = 0;
            certified_counts[distinct] = 0;
            distinct++;
        }
        proposals[found]++;
        certified_counts[found]++;
    }

    for (size_t i = 0; i < distinct; i++) {
        fprintf(pf, "crlp.hypothesis_library.template_prior.v1\t%u\t-\t-\t-\t-\t-\t-\t%zu\t%zu\t-\t-\t1.0\n",
                tids[i], proposals[i], certified_counts[i]);
    }

    fclose(ef);
    fclose(pf);
    fprintf(stderr, "  built ARC prior with %zu distinct templates: %s\n", distinct, prior_path);
    return (int)distinct;
}

/*
 * Parse arc_agi output (stdout captured to file) for loader summary and
 * extract certification metrics. Write compounding evidence rows.
 */
static int parse_arc_output_for_evidence(const char *stdout_path,
                                         const char *domain,
                                         int iteration,
                                         const char *provenance,
                                         FILE *ef,
                                         int *certified_out,
                                         int *rejected_out,
                                         double *utility_out) {
    FILE *outf = fopen(stdout_path, "r");
    if (!outf) return 0;
    char line[4096];
    int certified = 0, rejected = 0;
    double utility_sum = 0.0;
    while (fgets(line, sizeof(line), outf)) {
        if (strstr(line, "\"crlp.arc_loader.summary.v1\"")) {
            int cert = json_extract_int(line, "certified_tasks");
            int loaded = json_extract_int(line, "loaded_tasks");
            int test_exact = json_extract_int(line, "test_exact_tasks");
            double cell_acc = json_extract_double(line, "cell_accuracy");
            if (cert < 0) cert = 0;
            if (loaded < 0) loaded = 0;
            if (test_exact < 0) test_exact = 0;
            if (cell_acc < 0.0) cell_acc = 0.0;

            /* Write one certified summary row (not one per task). */
            if (cert > 0) {
                write_compounding_evidence(ef, domain, iteration, 1, 0, "", 0, cell_acc, provenance);
            }
            certified += cert;

            /* Non-certified loaded tasks: one rejected summary row. */
            int non_cert = loaded - cert;
            if (non_cert < 0) non_cert = 0;
            if (non_cert > 0) {
                write_compounding_evidence(ef, domain, iteration, 0, 0, "not_certified", non_cert, 0.0, provenance);
            }
            rejected += non_cert;
            utility_sum += cell_acc * (double)cert;
        }
    }
    fclose(outf);
    *certified_out = certified;
    *rejected_out = rejected;
    *utility_out = (certified > 0) ? (utility_sum / (double)certified) : 0.0;
    return 1;
}


/* Merge ARC hypothesis-library prior TSV files from iterations 0..max_iter.
   Sums proposal and certified counts per template_id across files.
   Writes a cumulative prior ready for --hypothesis-library-prior. */
static int merge_arc_prior_files(const char *od, int max_iteration,
                                 const char *output_path) {
    #define MP_MAX_TEMPLATES 512
    uint32_t tids[MP_MAX_TEMPLATES];
    size_t   props[MP_MAX_TEMPLATES];
    size_t   certs[MP_MAX_TEMPLATES];
    size_t   distinct = 0;

    for (int iter = 0; iter <= max_iteration; iter++) {
        char path[1024];
        snprintf(path, sizeof(path), "%s/arc_iter%02d_prior.tsv", od, iter);
        FILE *in = fopen(path, "r");
        if (!in) continue;

        char line[2048];
        while (fgets(line, sizeof(line), in)) {
            /* Skip header and blank lines */
            if (strncmp(line, "schema_version", 14) == 0) continue;
            if (line[0] == '\n' || line[0] == '\0') continue;

            /* Parse tab-separated: field0=schema field1=tid field8=proposals field9=certified */
            char *f0 = strtok(line, "\t\n");
            if (!f0) continue;
            char *f1 = strtok(NULL, "\t\n"); /* template_id */
            for (int skip = 0; skip < 5; skip++) strtok(NULL, "\t\n"); /* fields 2-6 */
            char *f8 = strtok(NULL, "\t\n"); /* proposals */
            char *f9 = strtok(NULL, "\t\n"); /* certified */
            if (!f1 || !f8 || !f9) continue;

            uint32_t tid = (uint32_t)atoi(f1);
            size_t   prop = (size_t)atoi(f8);
            size_t   cert = (size_t)atoi(f9);

            size_t found = distinct;
            for (size_t j = 0; j < distinct; j++) {
                if (tids[j] == tid) { found = j; break; }
            }
            if (found == distinct) {
                if (distinct >= MP_MAX_TEMPLATES) continue;
                tids[distinct] = tid;
                props[distinct] = 0;
                certs[distinct] = 0;
                distinct++;
            }
            props[found] += prop;
            certs[found] += cert;
        }
        fclose(in);
    }

    FILE *out = fopen(output_path, "w");
    if (!out) return 0;
    fprintf(out, "schema_version\t1\n");
    for (size_t i = 0; i < distinct; i++) {
        fprintf(out,
            "crlp.hypothesis_library.template_prior.v1\t%u"
            "\t-\t-\t-\t-\t-\t-\t%zu\t%zu\t-\t-\t1.0\n",
            tids[i], props[i], certs[i]);
    }
    fclose(out);
    fprintf(stderr, "  merged ARC cumulative prior (%zu templates, %d iterations): %s\n",
            distinct, max_iteration + 1, output_path);
    return (int)distinct;
}


/* Parse ARC source-task stdout for selection report lines and build a
   hypothesis-library prior TSV from per-template retained/certified counts.
   This works independently of --save-hypothesis-prior (which depends on
   ARC_RULE_HYPOTHESIS_APPLIED candidates). */
static int build_arc_prior_from_selection_stdout(const char *stdout_path,
                                                  const char *prior_path) {
    #define BAP_MAX_TEMPLATES 256
    uint32_t tids[BAP_MAX_TEMPLATES];
    size_t   retained[BAP_MAX_TEMPLATES];
    size_t   certified[BAP_MAX_TEMPLATES];
    size_t   distinct = 0;

    FILE *in = fopen(stdout_path, "r");
    if (!in) return 0;

    char line[16384];
    while (fgets(line, sizeof(line), in)) {
        /* Look for selection report lines */
        if (!strstr(line, "\"crlp.arc_hypothesis.selection.v1\"")) continue;
        if (!strstr(line, "\"templates\"")) continue;

        /* Extract each template block: "template_id":NNNN,"name":"...","retained":N,... */
        const char *p = line;
        while ((p = strstr(p, "\"template_id\":")) != NULL) {
            p += strlen("\"template_id\":");
            uint32_t tid = (uint32_t)atoi(p);

            /* Find retained count */
            int ret = 0, cert = 0;
            const char *rp = strstr(p, "\"retained\":");
            if (rp) { rp += strlen("\"retained\":"); ret = atoi(rp); }

            /* Find train_consistent_retained count (certified) */
            const char *cp = strstr(p, "\"train_consistent_retained\":");
            if (cp) { cp += strlen("\"train_consistent_retained\":"); cert = atoi(cp); }

            if (ret > 0 || cert > 0) {
                size_t found = distinct;
                for (size_t j = 0; j < distinct; j++) {
                    if (tids[j] == tid) { found = j; break; }
                }
                if (found == distinct) {
                    if (distinct >= BAP_MAX_TEMPLATES) break;
                    tids[distinct] = tid;
                    retained[distinct] = 0;
                    certified[distinct] = 0;
                    distinct++;
                }
                retained[found] += (size_t)ret;
                certified[found] += (size_t)cert;
            }
        }
    }
    fclose(in);

    if (distinct == 0) return 0;

    FILE *out = fopen(prior_path, "w");
    if (!out) return 0;
    fprintf(out, "schema_version\t1\n");
    for (size_t i = 0; i < distinct; i++) {
        fprintf(out,
            "crlp.hypothesis_library.template_prior.v1\t%u"
            "\t-\t-\t-\t-\t-\t%zu\t%zu\t-\t-\t1.0\n",
            tids[i], retained[i] + certified[i], certified[i]);
    }
    fclose(out);
    fprintf(stderr, "  built selection-derived ARC prior (%zu templates): %s\n",
            distinct, prior_path);
    return (int)distinct;
}

/* Run ARC domain for one iteration of the compounding loop.
   Uses source/heldout split: learns on source, evaluates on heldout. */
static int run_arc_compounding_iteration(const char *bd, const char *od,
                                         int iteration, FILE *ef, FILE *sf,
                                         const char *arc_task_dir) {
    char arc[1024];
    snprintf(arc, sizeof(arc), "%s/arc_agi", bd);

    /* Source (promotion) and heldout (evaluation) task IDs */
    /* Detect whether we are using the public ARC dataset (400 tasks)
       vs the small fixture set (14 tasks). */
    int is_public = (strstr(arc_task_dir, "arc_agi_public") != NULL);

    /* Rotate source task pairs per iteration so the cumulative prior
       accumulates evidence from different templates over time. */
    static const char *source_sets[] = {
        "move_to_anchor,move_to_anchor_x",
        "bbox_panel_selection,connect_aligned",
        "extend_aligned_sequence,fill_enclosed",
        "masked_motif_lift,move_to_anchor_both_positive",
        "move_to_anchor_y,object_extract",
        "panel_selection,recolor_by_reference",
        "same_grid_color_map,translate",
    };
    static const int source_set_count = 7;
    int set_idx = iteration % source_set_count;
    const char *source_ids = source_sets[set_idx];

    /* Heldout: all remaining fixture tasks */
    const char *heldout_ids =
        "bbox_panel_selection,connect_aligned,extend_aligned_sequence,"
        "fill_enclosed,masked_motif_lift,move_to_anchor,move_to_anchor_both_positive,"
        "move_to_anchor_x,move_to_anchor_y,object_extract,panel_selection,"
        "recolor_by_reference,same_grid_color_map,translate";

    char prior_path[1024], src_stdout[1024], hout_cold[1024], hout_warm[1024];
    char learned_state_path[1024], prev_learned_state_path[1024];
    snprintf(prior_path, sizeof(prior_path), "%s/arc_iter%02d_prior.tsv", od, iteration);
    snprintf(learned_state_path, sizeof(learned_state_path),
             "%s/arc_learned_state_%02d.bin", od, iteration);
    snprintf(prev_learned_state_path, sizeof(prev_learned_state_path),
             "%s/arc_learned_state_%02d.bin", od, iteration - 1);
    snprintf(src_stdout, sizeof(src_stdout), "%s/arc_iter%02d_source_stdout.txt", od, iteration);
    snprintf(hout_cold, sizeof(hout_cold), "%s/arc_iter%02d_heldout_cold_stdout.txt", od, iteration);
    snprintf(hout_warm, sizeof(hout_warm), "%s/arc_iter%02d_heldout_warm_stdout.txt", od, iteration);

    int src_cert = 0, src_rej = 0, cold_cert = 0, cold_rej = 0, warm_cert = 0, warm_rej = 0;
    double src_util = 0.0, cold_util = 0.0, warm_util = 0.0;

    /* ── Step 1: Run source tasks (promotion) ── */
    {
        char cmd[4096];
        if (is_public) {
            /* Public data: sample N random tasks per iteration */
            snprintf(cmd, sizeof(cmd),
                "%s --task-dir %s "
                "--max-tasks 100 --program-budget-grid "
                "--induction --min-induction-evidence 0 --seed %d --shuffle "
                "--save-hypothesis-prior %s "
                "--save-learned-state %s "
                "%s%s "
                "> %s 2>&1",
                arc, arc_task_dir,
                20240511 + iteration * 7, prior_path,
                learned_state_path,
                iteration > 0 ? "--load-learned-state " : "",
                iteration > 0 ? prev_learned_state_path : "",
                src_stdout);
        } else {
            snprintf(cmd, sizeof(cmd),
                "%s --task-dir %s --task-ids %s "
                "--max-tasks 14 --program-budget-grid "
                "--induction --min-induction-evidence 8 --seed %d --shuffle "
                "--save-hypothesis-prior %s "
                "--save-learned-state %s "
                "%s%s "
                "> %s 2>&1",
                arc, arc_task_dir, source_ids,
                20240511 + iteration * 7, prior_path,
                learned_state_path,
                iteration > 0 ? "--load-learned-state " : "",
                iteration > 0 ? prev_learned_state_path : "",
                src_stdout);
        }
        int rc = run_cmd(cmd);
        if (rc != 0) fprintf(stderr, "  ARC source tasks failed (iteration %d)\n", iteration);
        parse_arc_output_for_evidence(src_stdout, "arc_source", iteration, "none", ef, &src_cert, &src_rej, &src_util);

        /* Build prior from selection report in stdout (bypasses ARC_RULE_HYPOTHESIS_APPLIED) */
        build_arc_prior_from_selection_stdout(src_stdout, prior_path);
    }

    /* ── Build cumulative prior from all iterations so far ── */
    char cumulative_prior_path[1024];
    snprintf(cumulative_prior_path, sizeof(cumulative_prior_path),
             "%s/arc_cumulative_prior_%02d.tsv", od, iteration);
    merge_arc_prior_files(od, iteration, cumulative_prior_path);

    /* ── Step 2: Run heldout tasks cold (baseline, no priors) ── */
    {
        char cmd[4096];
        if (is_public) {
            snprintf(cmd, sizeof(cmd),
                "%s --task-dir %s "
                "--max-tasks 200 --program-budget-grid --loader-summary-only "
                "--induction --min-induction-evidence 0 --seed %d --shuffle "
                "--disable-hypothesis-slot-prior "
                "> %s 2>&1",
                arc, arc_task_dir,
                20240511 + iteration * 7 + 1000, hout_cold);
        } else {
            snprintf(cmd, sizeof(cmd),
                "%s --task-dir %s --task-ids %s "
                "--exclude-task-ids %s "
                "--max-tasks 14 --program-budget-grid --loader-summary-only "
                "--induction --min-induction-evidence 8 --seed %d --shuffle "
                "--disable-hypothesis-slot-prior "
                "> %s 2>&1",
                arc, arc_task_dir, heldout_ids, source_ids,
                20240511 + iteration * 7, hout_cold);
        }
        int rc = run_cmd(cmd);
        if (rc != 0) fprintf(stderr, "  ARC heldout cold failed (iteration %d)\n", iteration);
        parse_arc_output_for_evidence(hout_cold, "arc_heldout", iteration, "none", ef, &cold_cert, &cold_rej, &cold_util);
    }

    /* ── Step 3: Run heldout tasks warm (with learned prior) ── */
    {
        char cmd[4096];
        /* Only use prior if source tasks produced one */
        char prior_arg[2048] = "";
        if (src_cert > 0) {
            snprintf(prior_arg, sizeof(prior_arg), "--hypothesis-library-prior %s", cumulative_prior_path);
        }
        if (is_public) {
            snprintf(cmd, sizeof(cmd),
                "%s --task-dir %s "
                "--max-tasks 200 --program-budget-grid --loader-summary-only "
                "--induction --min-induction-evidence 0 --seed %d --shuffle "
                "%s%s "
                "%s"
                "> %s 2>&1",
                arc, arc_task_dir,
                20240511 + iteration * 7 + 1000,
                iteration > 0 ? "--load-learned-state " : "",
                iteration > 0 ? prev_learned_state_path : "",
                prior_arg, hout_warm);
        } else {
            snprintf(cmd, sizeof(cmd),
                "%s --task-dir %s --task-ids %s "
                "--exclude-task-ids %s "
                "--max-tasks 14 --program-budget-grid --loader-summary-only "
                "--induction --min-induction-evidence 8 --seed %d --shuffle "
                "%s%s "
                "%s"
                "> %s 2>&1",
                arc, arc_task_dir, heldout_ids, source_ids,
                20240511 + iteration * 7,
                iteration > 0 ? "--load-learned-state " : "",
                iteration > 0 ? prev_learned_state_path : "",
                prior_arg, hout_warm);
        }
        int rc = run_cmd(cmd);
        if (rc != 0) fprintf(stderr, "  ARC heldout warm failed (iteration %d, prior=%s)\n", iteration, prior_path);
        parse_arc_output_for_evidence(hout_warm, "arc_heldout", iteration, "cumulative_prior",
                                      ef, &warm_cert, &warm_rej, &warm_util);
    }

    /* ── Write summary ── */
    {
        double rate = 0.0;
        int total = warm_cert + warm_rej;
        if (total > 0) rate = (double)warm_rej / (double)total;
        write_compounding_summary(sf, iteration, "arc_heldout",
                                  warm_cert, warm_rej, warm_util, rate);
    }
    fprintf(stderr, "  ARC iter %d: source cert=%d rej=%d | cold cert=%d rej=%d | warm cert=%d rej=%d util=%.4f\n",
            iteration, src_cert, src_rej, cold_cert, cold_rej, warm_cert, warm_rej, warm_util);

    return 0;
}

/*
 * Parse a prime_rule_search trace file for verifier outcome rows and write
 * compounding evidence.
 */
static int parse_prime_trace_for_evidence(const char *trace_path,
                                          const char *domain,
                                          int iteration,
                                          const char *provenance,
                                          FILE *ef,
                                          int *certified_out,
                                          int *rejected_out,
                                          double *utility_out) {
    FILE *tf = fopen(trace_path, "r");
    if (!tf) return 0;
    char line[4096];
    int certified = 0, rejected = 0;
    double utility_sum = 0.0;
    while (fgets(line, sizeof(line), tf)) {
        /* Look for verifier outcome rows */
        if (strstr(line, "\"crlp.nn_protocol.verifier_outcome.v1\"") ||
            strstr(line, "\"verifier_outcome\"")) {
            int supp   = json_extract_int(line, "support");
            int falser = json_extract_int(line, "false_rejections");
            double u   = json_extract_double(line, "utility");
            int tid    = json_extract_int(line, "template_id");
            if (tid < 0) tid = json_extract_int(line, "stable_id");

            char reason[256] = "";
            json_extract_string(line, "reason_code", reason, sizeof(reason));

            if (supp > 0) {
                write_compounding_evidence(ef, domain, iteration, 1, tid,
                                           reason, falser, u, provenance);
                certified++;
                if (u >= 0.0) utility_sum += u;
            } else {
                write_compounding_evidence(ef, domain, iteration, 0, tid,
                                           reason[0] ? reason : "no_support",
                                           falser, u, provenance);
                rejected++;
            }
        }
        /* Also check for rule_search.summary rows (prime domain primary output) */
        if (strstr(line, "\"crlp.rule_search.summary.v1\"")) {
            int accepted = json_extract_int(line, "accepted_rules");
            int proposed = json_extract_int(line, "proposed_rules");
            int verified = json_extract_int(line, "verified_rules");
            int unsafe = json_extract_int(line, "unsafe_rules");
            int no_support = json_extract_int(line, "no_support_rules");
            if (accepted < 0) accepted = 0;
            if (proposed < 0) proposed = 0;
            if (verified < 0) verified = 0;
            if (unsafe < 0) unsafe = 0;
            if (no_support < 0) no_support = 0;
            /* Accepted rules are certified */
            for (int a = 0; a < accepted; a++) {
                write_compounding_evidence(ef, domain, iteration, 1, 0, "", 0, 1.0, provenance);
                certified++;
                utility_sum += 1.0;
            }
            /* Unsafe + no_support rules are rejected */
            int rejected_count = unsafe + no_support;
            if (rejected_count > 1000) rejected_count = 1000; /* cap to keep output reasonable */
            for (int r = 0; r < rejected_count; r++) {
                write_compounding_evidence(ef, domain, iteration, 0, 0, "rejected", 0, 0.0, provenance);
                rejected++;
            }
        }
        /* Also check for compare/certificate rows in prime gap traces */
        if (strstr(line, "\"crlp.compare.v1\"")) {
            int falser = json_extract_int(line, "false_rejections");
            double u   = json_extract_double(line, "utility");
            if (falser < 0) falser = 0;
            if (u < 0.0) u = 0.0;
            /* A compare row with non-zero utility implies certification */
            if (u > 0.0) {
                write_compounding_evidence(ef, domain, iteration, 1, 0, "", falser, u, provenance);
                certified++;
                utility_sum += u;
            } else {
                write_compounding_evidence(ef, domain, iteration, 0, 0, "low_utility", falser, u, provenance);
                rejected++;
            }
        }
    }
    fclose(tf);
    *certified_out = certified;
    *rejected_out = rejected;
    *utility_out = (certified > 0) ? (utility_sum / (double)certified) : 0.0;
    return 1;
}

/* Run prime domain for one compounding iteration. */
static int run_prime_compounding_iteration(const char *bd, const char *od,
                                           int iteration, FILE *ef, FILE *sf) {
    char prime[1024], trace[1024];
    snprintf(prime, sizeof(prime), "%s/prime_rule_search", bd);
    snprintf(trace, sizeof(trace), "%s/prime_iter%02d_trace.jsonl", od, iteration);

    char cmd[4096];
    char stdout_path[1024];
    snprintf(stdout_path, sizeof(stdout_path), "%s/prime_iter%02d_stdout.txt", od, iteration);
    snprintf(cmd, sizeof(cmd),
        "%s --candidate-only --moduli-range 3:31 --max-rules 4 "
        "--proposer closed-loop --proposer-seed %d "
        "--closed-loop-hidden 64x32 "
        "--trace %s --save %s/prime_iter%02d.crlprule "
        "> %s 2>&1",
        prime, 20240511 + iteration * 3, trace, od, iteration, stdout_path);
    int rc = run_cmd(cmd);
    if (rc != 0) {
        fprintf(stderr, "  prime domain failed (iteration %d)\n", iteration);
    }

    int cert = 0, rej = 0;
    double util = 0.0;
    parse_prime_trace_for_evidence(stdout_path, "prime", iteration, "none", ef, &cert, &rej, &util);

    double rate = 0.0;
    int total = cert + rej;
    if (total > 0) rate = (double)rej / (double)total;
    write_compounding_summary(sf, iteration, "prime", cert, rej, util, rate);
    fprintf(stderr, "  prime iter %d: cert=%d rej=%d util=%.4f\n", iteration, cert, rej, util);
    return 0;
}

/* Run grid_path domain for one compounding iteration. */
static int run_grid_compounding_iteration(const char *bd, const char *od,
                                          int iteration, FILE *ef, FILE *sf) {
    char grid[1024], trace[1024];
    snprintf(grid, sizeof(grid), "%s/grid_path", bd);
    snprintf(trace, sizeof(trace), "%s/grid_iter%02d_trace.jsonl", od, iteration);

    char cmd[4096];
    char g_stdout_path[1024];
    snprintf(g_stdout_path, sizeof(g_stdout_path), "%s/grid_iter%02d_stdout.txt", od, iteration);
    snprintf(cmd, sizeof(cmd),
        "%s --save %s/grid_iter%02d.nnckpt --trace %s --trace-train "
        "> %s 2>&1",
        grid, od, iteration, trace, g_stdout_path);
    int rc = run_cmd(cmd);
    if (rc != 0) {
        fprintf(stderr, "  grid_path domain failed (iteration %d)\n", iteration);
    }

    /* Parse captured stdout for comparison/certification rows */
    int cert = 0, rej = 0;
    double util_val = 0.0;
    FILE *tf = fopen(g_stdout_path, "r");
    if (tf) {
        char line[4096];
        while (fgets(line, sizeof(line), tf)) {
            /* Parse bracket-summary lines: "[grid_path] utility=0.6987 rejected=0.00 ..." */
            if (strstr(line, "[grid_path]")) {
                double u = bracket_extract_double(line, "utility");
                double rej_count = bracket_extract_double(line, "rejected");
                if (u < 0.0) u = 0.0;
                if (rej_count < 0.0) rej_count = 0.0;
                if (u > 0.0) {
                    write_compounding_evidence(ef, "grid_path", iteration, 1, 0, "", (int)rej_count, u, "none");
                    cert++;
                    util_val += u;
                }
                if (rej_count > 0.0) {
                    int irc = (int)rej_count;
                    if (irc > 100) irc = 100; /* cap */
                    for (int r = 0; r < irc; r++) {
                        write_compounding_evidence(ef, "grid_path", iteration, 0, 0, "rejected", 0, 0.0, "none");
                        rej++;
                    }
                }
            }
            if (strstr(line, "\"crlp.compare.v1\"")) {
                int falser = json_extract_int(line, "false_rejections");
                double u   = json_extract_double(line, "utility");
                int rejects = json_extract_int(line, "rejected_candidates");
                if (u < 0.0) u = 0.0;
                if (falser < 0) falser = 0;
                if (rejects < 0) rejects = 0;
                if (u > 0.0) {
                    write_compounding_evidence(ef, "grid_path", iteration, 1, 0, "", falser, u, "none");
                    cert++;
                    util_val += u;
                } else if (rejects > 0) {
                    write_compounding_evidence(ef, "grid_path", iteration, 0, 0, "rejected", falser, u, "none");
                    rej++;
                }
            }
        }
        fclose(tf);
    }
    if (cert > 0) util_val /= (double)cert;

    double rate = 0.0;
    int total = cert + rej;
    if (total > 0) rate = (double)rej / (double)total;
    write_compounding_summary(sf, iteration, "grid_path", cert, rej, util_val, rate);
    fprintf(stderr, "  grid_path iter %d: cert=%d rej=%d util=%.4f\n", iteration, cert, rej, util_val);
    return 0;
}

/* Run delayed_claim domain for one compounding iteration. */
static int run_dc_compounding_iteration(const char *bd, const char *od,
                                        int iteration, FILE *ef, FILE *sf) {
    char dc[1024], trace[1024];
    snprintf(dc, sizeof(dc), "%s/delayed_claim", bd);
    snprintf(trace, sizeof(trace), "%s/dc_iter%02d_trace.jsonl", od, iteration);

    char cmd[4096];
    char dc_stdout_path[1024];
    snprintf(dc_stdout_path, sizeof(dc_stdout_path), "%s/dc_iter%02d_stdout.txt", od, iteration);
    snprintf(cmd, sizeof(cmd),
        "%s --save %s/dc_iter%02d.nnckpt --trace %s --trace-train "
        "> %s 2>&1",
        dc, od, iteration, trace, dc_stdout_path);
    int rc = run_cmd(cmd);
    if (rc != 0) {
        fprintf(stderr, "  delayed_claim domain failed (iteration %d)\n", iteration);
    }

    int dccert = 0, dcrej = 0;
    double dcutil = 0.0;
    FILE *tf = fopen(dc_stdout_path, "r");
    if (tf) {
        char line[4096];
        while (fgets(line, sizeof(line), tf)) {
            /* Parse bracket-summary lines: "[delayed_claim] decision_utility=1.0432 ..." */
            if (strstr(line, "[delayed_claim]")) {
                double u = bracket_extract_double(line, "decision_utility");
                if (u < 0.0) u = 0.0;
                if (u > 0.0) {
                    write_compounding_evidence(ef, "delayed_claim", iteration, 1, 0, "", 0, u, "none");
                    dccert++;
                    dcutil += u;
                }
            }
            if (strstr(line, "\"crlp.compare.v1\"")) {
                int falser = json_extract_int(line, "false_rejections");
                int cflag  = json_extract_int(line, "certified");
                double u   = json_extract_double(line, "utility");
                if (u < 0.0) u = 0.0;
                if (falser < 0) falser = 0;
                if (cflag > 0 || u > 0.0) {
                    write_compounding_evidence(ef, "delayed_claim", iteration, 1, 0, "", falser, u, "none");
                    dccert++;
                    dcutil += u;
                } else {
                    write_compounding_evidence(ef, "delayed_claim", iteration, 0, 0, "not_certified", falser, u, "none");
                    dcrej++;
                }
            }
        }
        fclose(tf);
    }
    if (dccert > 0) dcutil /= (double)dccert;

    double rate = 0.0;
    int total = dccert + dcrej;
    if (total > 0) rate = (double)dcrej / (double)total;
    write_compounding_summary(sf, iteration, "delayed_claim", dccert, dcrej, dcutil, rate);
    fprintf(stderr, "  delayed_claim iter %d: cert=%d rej=%d util=%.4f\n", iteration, dccert, dcrej, dcutil);
    return 0;
}

/* ─── Compounding Loop Runner ───────────────────────────────────────── */

static int run_compounding_loop(const char *bd, const char *od, int iterations, const char *arc_task_dir) {
    mkdir_p(od);

    char evidence_path[1024], summary_path[1024];
    snprintf(evidence_path, sizeof(evidence_path), "%s/evidence.jsonl", od);
    snprintf(summary_path, sizeof(summary_path), "%s/compounding_summary.jsonl", od);

    /* Open evidence file for append (accumulates across iterations) */
    FILE *ef = fopen(evidence_path, "w");
    FILE *sf = fopen(summary_path, "w");
    if (!ef || !sf) { perror(od); return 1; }

    fprintf(stderr, "=== compounding-loop: %d iterations, output %s ===\n", iterations, od);

    for (int iter = 0; iter < iterations; iter++) {
        fprintf(stderr, "\n--- iteration %d ---\n", iter);

        /* Run all four domains for this iteration. */
        run_prime_compounding_iteration(bd, od, iter, ef, sf);
        run_grid_compounding_iteration(bd, od, iter, ef, sf);
        run_dc_compounding_iteration(bd, od, iter, ef, sf);
        run_arc_compounding_iteration(bd, od, iter, ef, sf, arc_task_dir);
    }

    fclose(ef);
    fclose(sf);
    printf("  wrote %s, %s\n", evidence_path, summary_path);
    return 0;
}

/* ─── Main ──────────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    const char *mode = NULL, *bd = "build/double", *od = NULL;
    const char *arc_task_dir = "tests/fixtures/arc_hypothesis";
    int iterations = 3;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--mode") && i+1 < argc) mode = argv[++i];
        else if (!strcmp(argv[i], "--build-dir") && i+1 < argc) bd = argv[++i];
        else if (!strcmp(argv[i], "--output-dir") && i+1 < argc) od = argv[++i];
        else if (!strcmp(argv[i], "--iterations") && i+1 < argc) iterations = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--arc-task-dir") && i+1 < argc) arc_task_dir = argv[++i];
        else if (!strcmp(argv[i], "--real") && i+1 < argc) { /* ignore: captured by Makefile */ ++i; }
    }
    if (!mode || !od) {
        fprintf(stderr, "usage: %s --mode <mode> --output-dir <dir> [--build-dir <dir>] [--iterations N]\n", argv[0]);
        fprintf(stderr, "  modes: rejected-candidate-learning, rejected-candidate-seed-audit,\n");
        fprintf(stderr, "         hypothesis-library-split-sweep, compounding-loop\n");
        return 1;
    }
    if (iterations < 1) iterations = 1;
    if (iterations > 10) iterations = 10;

    printf("=== %s ===\n", mode);
    if (!strcmp(mode, "rejected-candidate-learning"))
        return run_rejected_candidate_learning(bd, od);
    if (!strcmp(mode, "rejected-candidate-seed-audit"))
        return run_seed_audit(bd, od);
    if (!strcmp(mode, "hypothesis-library-split-sweep"))
        return run_split_sweep(bd, od);
    if (!strcmp(mode, "compounding-loop"))
        return run_compounding_loop(bd, od, iterations, arc_task_dir);
    fprintf(stderr, "unknown mode: %s\n", mode);
    return 1;
}
