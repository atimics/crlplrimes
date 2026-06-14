#include <ctype.h>
#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LINE_CAP 65536
#define VALUE_CAP 4096

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

static double json_number(const char *line, const char *key, double fallback)
{
    const char *p = json_value(line, key);
    if (p == NULL) {
        return fallback;
    }
    errno = 0;
    char *end = NULL;
    double v = strtod(p, &end);
    return (end == p || errno == ERANGE) ? fallback : v;
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

static void write_scores(FILE *out, const char *line)
{
    const char *p = json_value(line, "scores");
    if (p == NULL || *p != '[') {
        return;
    }
    ++p;
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
        p = end;
    }
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s <trace.jsonl> <output.txt>\n", argv[0]);
        return 2;
    }
    FILE *in = fopen(argv[1], "r");
    if (in == NULL) {
        perror(argv[1]);
        return 2;
    }
    FILE *out = fopen(argv[2], "w");
    if (out == NULL) {
        perror(argv[2]);
        fclose(in);
        return 2;
    }
    char line[LINE_CAP];
    size_t rows = 0;
    while (fgets(line, sizeof(line), in) != NULL) {
        char state_hash[VALUE_CAP] = {0};
        char verifier[VALUE_CAP] = {0};
        json_string(line, "state_hash", state_hash, sizeof(state_hash));
        json_string(line, "verifier_result", verifier, sizeof(verifier));
        long state_id = (long)json_number(line, "p", 0.0);
        long target = (long)json_number(line, "target_action_id", 0.0);
        int accepted = strcmp(verifier, "accepted") == 0 || strcmp(verifier, "corrected") == 0;
        fprintf(out, "%s\t%ld\t%ld\t%d", state_hash, state_id, target, accepted);
        write_scores(out, line);
        fputc('\n', out);
        ++rows;
    }
    fclose(in);
    fclose(out);
    printf("Wrote %zu rows\n", rows);
    return 0;
}
