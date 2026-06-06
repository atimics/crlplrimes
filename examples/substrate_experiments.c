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

/* Extract an integer JSON field value from a flat JSON object line.
   Looks for "fieldname":VALUE and parses VALUE. */
static int json_extract_int(const char *line, const char *field) {
    char key[256];
    snprintf(key, sizeof(key), "\"%s\":", field);
    const char *p = strstr(line, key);
    if (!p) return -1;
    p += strlen(key);
    return atoi(p);
}

/* Extract a double JSON field value. */
static double json_extract_double(const char *line, const char *field) {
    char key[256];
    snprintf(key, sizeof(key), "\"%s\":", field);
    const char *p = strstr(line, key);
    if (!p) return -1.0;
    p += strlen(key);
    return atof(p);
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

    FILE *rf = fopen(rp, "w"), *sf = fopen(sp, "w"), *pf = fopen(pp, "w"), *ff = fopen(fsp, "w");
    if (!rf || !sf || !pf || !ff) { perror(od); return 1; }

    /* Surface 1: prime modular filters -- closed-loop proposer */
    {
        char cmd[2048], tr[1024];
        snprintf(tr, sizeof(tr), "%s/traces/prime_rule_search.jsonl", bd);
        snprintf(cmd, sizeof(cmd),
            "%s --candidate-only --moduli-range 3:255 --max-rules 8 "
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
            "%s --candidate-only --moduli-range 3:255 --max-rules 8 "
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
                    "%s --candidate-only --moduli-range 3:255 --max-rules 8 "
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

    /* Run loader summary on the full hypothesis task set, capture stdout to parse heldout metrics */
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

        /* Parse the captured stdout for the loader summary line */
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

int main(int argc, char **argv) {
    const char *mode = NULL, *bd = "build/double", *od = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--mode") && i+1 < argc) mode = argv[++i];
        else if (!strcmp(argv[i], "--build-dir") && i+1 < argc) bd = argv[++i];
        else if (!strcmp(argv[i], "--output-dir") && i+1 < argc) od = argv[++i];
    }
    if (!mode || !od) {
        fprintf(stderr, "usage: %s --mode <mode> --output-dir <dir> [--build-dir <dir>]\n", argv[0]);
        fprintf(stderr, "  modes: rejected-candidate-learning, rejected-candidate-seed-audit, hypothesis-library-split-sweep\n");
        return 1;
    }
    printf("=== %s ===\n", mode);
    if (!strcmp(mode, "rejected-candidate-learning"))
        return run_rejected_candidate_learning(bd, od);
    if (!strcmp(mode, "rejected-candidate-seed-audit"))
        return run_seed_audit(bd, od);
    if (!strcmp(mode, "hypothesis-library-split-sweep"))
        return run_split_sweep(bd, od);
    fprintf(stderr, "unknown mode: %s\n", mode);
    return 1;
}
