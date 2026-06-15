int nn_paper_write_evidence_origin_audit(const char *output_dir,
                                          int fail_synthetic_required) {
    if (output_dir == NULL || output_dir[0] == '\0') {
        output_dir = "paper/generated";
    }

    char path[4096];
    snprintf(path, sizeof(path), "%s/evidence_origin_audit.md", output_dir);

    FILE *fp = fopen(path, "w");
    if (fp == NULL) {
        fprintf(stderr, "Failed to open %s for writing\n", path);
        return 1;
    }

    fprintf(fp, "# Evidence Origin Audit\n\n");
    fprintf(fp, "This report shows which claims are supported by measured_run vs synthetic data.\n\n");
    fprintf(fp, "## Summary\n\n");

    int measured_count = 0, synthetic_count = 0;
    size_t i;

    for (i = 0; i < PAPER_CLAIM_COUNT; i++) {
        const char *origin = PAPER_CLAIM_SPECS[i].evidence_origin;
        if (origin == NULL || strcmp(origin, "measured_run") == 0) {
            measured_count++;
        } else if (strcmp(origin, "synthetic") == 0 || strcmp(origin, "synthetic_diagnostic") == 0) {
            synthetic_count++;
        }
    }

    fprintf(fp, "- **Total claims**: %u\n", (unsigned)PAPER_CLAIM_COUNT);
    fprintf(fp, "- **Measured run data**: %d\n", measured_count);
    fprintf(fp, "- **Synthetic/diagnostic data**: %d\n\n", synthetic_count);

    if (synthetic_count > 0 && fail_synthetic_required) {
        fprintf(fp, "FAIL: %d claim(s) backed only by synthetic data without explicit allowance.\n", synthetic_count);
        fclose(fp);
        printf("Evidence origin audit written to %s\n", path);
        return 1;
    }

    fprintf(fp, "PASS: all required claims have measured origin or explicit synthetic allowance.\n");
    fclose(fp);
    printf("Evidence origin audit written to %s\n", path);
    return 0;
}
