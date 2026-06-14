#include <ctype.h>
#include <errno.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#define LINE_CAP 131072
#define TEXT_CAP 4096
#define CMD_CAP 8192

static uint64_t fnv1a(const char *text)
{
    uint64_t hash = UINT64_C(1469598103934665603);
    const unsigned char *p = (const unsigned char *)text;
    while (*p) {
        hash ^= (uint64_t)*p++;
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static const char *json_value(const char *line, const char *key)
{
    char needle[128];
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    const char *p = strstr(line, needle);
    if (p == NULL) {
        return NULL;
    }
    p += strlen(needle);
    while (*p && isspace((unsigned char)*p)) {
        ++p;
    }
    if (*p != ':') {
        return NULL;
    }
    ++p;
    while (*p && isspace((unsigned char)*p)) {
        ++p;
    }
    return p;
}

static bool json_string(const char *line, const char *key, char *out, size_t cap)
{
    const char *p = json_value(line, key);
    if (p == NULL || *p != '"') {
        return false;
    }
    ++p;
    size_t used = 0;
    while (*p && *p != '"') {
        char ch = *p++;
        if (ch == '\\' && *p) {
            ch = *p++;
        }
        if (used + 1u < cap) {
            out[used++] = ch;
        }
    }
    if (*p != '"') {
        return false;
    }
    if (cap > 0) {
        out[used] = '\0';
    }
    return true;
}

static double json_number(const char *line, const char *key, double fallback)
{
    const char *p = json_value(line, key);
    if (p == NULL) {
        return fallback;
    }
    errno = 0;
    char *end = NULL;
    double value = strtod(p, &end);
    return (end == p || errno == ERANGE) ? fallback : value;
}

static bool verifier_ok(const char *value)
{
    return strcmp(value, "correct") == 0 ||
           strcmp(value, "accepted") == 0 ||
           strcmp(value, "corrected") == 0 ||
           strcmp(value, "dataset_target") == 0;
}

static bool write_scores(FILE *out, const char *line)
{
    const char *p = json_value(line, "scores");
    if (p == NULL || *p != '[') {
        return false;
    }
    ++p;
    int count = 0;
    while (*p && *p != ']') {
        while (*p && (isspace((unsigned char)*p) || *p == ',')) {
            ++p;
        }
        if (*p == ']') {
            break;
        }
        char *end = NULL;
        double score = strtod(p, &end);
        if (end == p) {
            break;
        }
        fprintf(out, "\t%.6f", score);
        ++count;
        p = end;
    }
    return count > 0;
}

static int score_count(const char *line)
{
    const char *p = json_value(line, "scores");
    if (p == NULL || *p != '[') {
        return 0;
    }
    ++p;
    int count = 0;
    while (*p && *p != ']') {
        while (*p && (isspace((unsigned char)*p) || *p == ',')) {
            ++p;
        }
        if (*p == ']') {
            break;
        }
        char *end = NULL;
        (void)strtod(p, &end);
        if (end == p) {
            break;
        }
        ++count;
        p = end;
    }
    return count;
}

static void write_state_key(FILE *out, const char *domain, const char *line, const char *state_hash)
{
    const char *p = json_value(line, "features");
    if (p == NULL || *p != '[') {
        fprintf(out, "hash:%u", (unsigned)(fnv1a(state_hash) & UINT64_C(0xffffffff)));
        return;
    }
    ++p;
    fprintf(out, "features:");
    int index = 0;
    bool first = true;
    while (*p && *p != ']') {
        while (*p && (isspace((unsigned char)*p) || *p == ',')) {
            ++p;
        }
        if (*p == ']') {
            break;
        }
        char *end = NULL;
        double value = strtod(p, &end);
        if (end == p) {
            break;
        }
        if (strcmp(domain, "arc") == 0 && (index == 0 || index == 1 || index == 6)) {
            value = 0.0;
        }
        fprintf(out, "%s%.6f", first ? "" : ",", value);
        first = false;
        ++index;
        p = end;
    }
}

int main(int argc, char **argv)
{
    if (argc != 4) {
        fprintf(stderr, "usage: %s <domain> <trace.jsonl> <out-dir>\n", argv[0]);
        return 2;
    }
    const char *domain = argv[1];
    const char *trace = argv[2];
    const char *out_dir = argv[3];
    char mkdir_cmd[CMD_CAP];
    snprintf(mkdir_cmd, sizeof(mkdir_cmd), "mkdir -p '%s'", out_dir);
    if (system(mkdir_cmd) != 0) {
        return 2;
    }
    char input_path[TEXT_CAP];
    snprintf(input_path, sizeof(input_path), "%s/%s_holo_input.txt", out_dir, domain);
    FILE *in = fopen(trace, "r");
    if (in == NULL) {
        perror(trace);
        return 2;
    }
    FILE *out = fopen(input_path, "w");
    if (out == NULL) {
        perror(input_path);
        fclose(in);
        return 2;
    }
    char line[LINE_CAP];
    size_t rows = 0;
    int candidate_count = 0;
    while (fgets(line, sizeof(line), in) != NULL) {
        char schema[TEXT_CAP] = {0};
        if (!json_string(line, "schema", schema, sizeof(schema)) ||
            strcmp(schema, "crlp.trace.v1") != 0) {
            continue;
        }
        char state_hash[TEXT_CAP] = {0};
        char verifier[TEXT_CAP] = {0};
        json_string(line, "state_hash", state_hash, sizeof(state_hash));
        json_string(line, "verifier_result", verifier, sizeof(verifier));
        int scores = score_count(line);
        if (scores <= 0) {
            continue;
        }
        long target = (long)json_number(line, "target_action_id", 0.0);
        fprintf(out, "%s\t", state_hash);
        write_state_key(out, domain, line, state_hash);
        fprintf(out, "\t%ld\t%d", target, verifier_ok(verifier) ? 1 : 0);
        (void)write_scores(out, line);
        if (candidate_count == 0) {
            candidate_count = scores;
        }
        fputc('\n', out);
        ++rows;
    }
    fclose(in);
    fclose(out);
    if (rows < 20) {
        fprintf(stderr, "SKIP %s: only %zu rows\n", domain, rows);
        return 0;
    }
    printf("DOMAIN %s: %zu rows, %d candidates\n", domain, rows, candidate_count);
    const char *binary = "build/float/holo_experiments";
    if (access(binary, X_OK) != 0) {
        binary = "build/holo_experiments";
    }
    char cmd[CMD_CAP];
    snprintf(cmd, sizeof(cmd),
             "%s --mode domain-integration --domain '%s' --input '%s' --output-dir '%s'",
             binary, domain, input_path, out_dir);
    int rc = system(cmd);
    if (rc == -1) {
        return 2;
    }
    return WIFEXITED(rc) ? WEXITSTATUS(rc) : 2;
}
