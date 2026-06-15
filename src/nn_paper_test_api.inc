#ifdef NN_ENABLE_TEST_API
static int paper_test_set_mtime(const char *path, time_t mtime) {
    struct utimbuf times;
    times.actime = mtime;
    times.modtime = mtime;
    return utime(path, &times) == 0;
}

static int paper_test_write_timed_file(const char *path, time_t mtime) {
    return paper_write_text_file(path, "paper claim governance test fixture\n") == 0 &&
           paper_test_set_mtime(path, mtime);
}

static int paper_test_staleness_case(const char *source_path,
                                     const char *freshness_prerequisite,
                                     const char *initial_status,
                                     const char *expected_status,
                                     const char *expected_summary_substring) {
    PaperClaimResult claims[PAPER_CLAIM_COUNT];
    PaperClaimSpec spec = {
        .id = "T.paper_staleness_fixture",
        .family = "T",
        .title = "Paper staleness fixture",
        .source = source_path,
        .required = 1,
        .claim_text = "fixture",
        .explanation_text = "",
        .chart_kind = "",
        .chart_unit = "",
        .chart_y_label = "",
        .freshness_prerequisite = freshness_prerequisite,
        .refresh_command = "make test-refresh",
    };
    memset(claims, 0, sizeof(claims));
    paper_claim_init(&claims[0], &spec);
    snprintf(claims[0].status, sizeof(claims[0].status), "%s", initial_status);
    snprintf(claims[0].summary, sizeof(claims[0].summary), "%s", "supported fixture");
    paper_apply_staleness_gate(claims);
    return strcmp(claims[0].status, expected_status) == 0 &&
           (expected_summary_substring == NULL ||
            strstr(claims[0].summary, expected_summary_substring) != NULL);
}

int nn_paper_test_claim_governance_contract(const char *work_dir,
                                            const char *claims_path) {
    char source_old[4096];
    char source_new[4096];
    char prereq_old[4096];
    char prereq_new[4096];
    char prereq_missing[4096];
    char registry_error[512];
    int written;
    if (!paper_claim_has_text(work_dir) || !paper_claim_has_text(claims_path)) {
        return 0;
    }
    if (paper_mkdirs(work_dir) != 0) {
        return 0;
    }
    written = snprintf(source_old, sizeof(source_old), "%s/source_old.txt", work_dir);
    if (written <= 0 || (size_t)written >= sizeof(source_old)) {
        return 0;
    }
    written = snprintf(source_new, sizeof(source_new), "%s/source_new.txt", work_dir);
    if (written <= 0 || (size_t)written >= sizeof(source_new)) {
        return 0;
    }
    written = snprintf(prereq_old, sizeof(prereq_old), "%s/prereq_old.txt", work_dir);
    if (written <= 0 || (size_t)written >= sizeof(prereq_old)) {
        return 0;
    }
    written = snprintf(prereq_new, sizeof(prereq_new), "%s/prereq_new.txt", work_dir);
    if (written <= 0 || (size_t)written >= sizeof(prereq_new)) {
        return 0;
    }
    written = snprintf(prereq_missing,
                       sizeof(prereq_missing),
                       "%s/prereq_missing.txt",
                       work_dir);
    if (written <= 0 || (size_t)written >= sizeof(prereq_missing)) {
        return 0;
    }
    (void)remove(prereq_missing);
    if (!paper_test_write_timed_file(source_old, (time_t)1000) ||
        !paper_test_write_timed_file(source_new, (time_t)1010) ||
        !paper_test_write_timed_file(prereq_old, (time_t)1000) ||
        !paper_test_write_timed_file(prereq_new, (time_t)1005)) {
        return 0;
    }
    registry_error[0] = '\0';
    return paper_validate_claim_registry(claims_path,
                                         registry_error,
                                         sizeof(registry_error)) == 0 &&
           paper_test_staleness_case(source_old,
                                     prereq_new,
                                     "supported",
                                     "stale",
                                     "older than freshness prerequisite") &&
           paper_test_staleness_case(source_new,
                                     prereq_old,
                                     "supported",
                                     "supported",
                                     "supported fixture") &&
           paper_test_staleness_case(source_new,
                                     prereq_missing,
                                     "supported",
                                     "stale",
                                     "cannot verify source freshness") &&
           paper_test_staleness_case(source_old,
                                     prereq_new,
                                     "contradicted",
                                     "contradicted",
                                     "supported fixture");
}
#endif
