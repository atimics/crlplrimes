#define SIGNAL_DEMO_TEST
#include "../examples/signal_demo.c"

static unsigned assertions;
static void check(int condition, const char *name)
{
    ++assertions;
    if (!condition) fail("Test failed: %s", name);
}

/* Failures are isolated so each check also covers the CLI error boundary. */
#define REJECT(fragment, expression) do { \
    pid_t child = fork(); \
    require(child >= 0, "Cannot fork rejection test"); \
    if (!child) { \
        if (setjmp(demo_failure)) { \
            int matched = strstr(demo_error, fragment) != NULL; \
            if (!matched) fprintf(stderr, "Expected %s; received %s\n", fragment, demo_error); \
            release_allocations(); \
            _exit(matched ? 0 : 2); \
        } \
        expression; \
        _exit(3); \
    } \
    int status = 0; \
    require(waitpid(child, &status, 0) == child, "Cannot wait for rejection test"); \
    check(WIFEXITED(status) && WEXITSTATUS(status) == 0, fragment); \
} while (0)

static const char *model_hash = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
static Json *config, *training, *rollout;
static char temporary[PATH_MAX];

static void fixture(void)
{
    config = read_json(join(SIGNAL_DEMO_ROOT, "integrations/signal-demo/config.json"));
    training = parse_json("{\"schema\":\"crlp.signal_flight.summary.v1\",\"training_mode\":\"counterfactual_utility\",\"feature_mode\":\"live_observable\",\"parameter_count\":2113,\"train_rows\":18,\"test_rows\":9}");
    Json *schedule = get(config, "training");
    put(training, "epochs", get(schedule, "epochs")); put(training, "seed", get(schedule, "seed")); put(training, "learning_rate", get(schedule, "learning_rate")); put(training, "model_hash", string(model_hash));
    rollout = array();
    Json *settings = get(config, "rollout");
    for (size_t n = 0; n < 4; ++n) {
        Json *row = parse_json("{\"schema\":\"crlp.signal_flight_rollout.summary.v1\",\"wall_probes\":true,\"avg_utility\":0.2,\"damage_episode_rate\":0.25,\"death_rate\":0}");
        put(row, "controller", string(controllers[n])); put(row, "checkpoint_hash", string(model_hash));
        put(row, "episodes", get(settings, "episodes")); put(row, "ticks_per_episode", get(settings, "ticks")); put(row, "seed", get(settings, "seed"));
        append(rollout, row);
    }
}
static Json *metrics(void) { return result_metrics(training, rollout, config, model_hash); }
static void negative_result(void)
{
    put(rollout->tail->value, "avg_utility", number(-0.2));
    check(value(metrics(), "utility_delta") == -0.4, "negative measured result is retained");
}
static void changed_rule(void)
{
    put(get(config, "outcome"), "threshold", number(0.1));
    REJECT("fixed neural_guarded", validate_config(config));
}
static void overlapping_seeds(void)
{
    put(get(config, "rollout"), "seed", get(get(get(config, "data"), "heldout"), "seed"));
    REJECT("seeds must differ", validate_config(config));
}
static void same_directory(void)
{
    const char *alias = join(temporary, "alias");
    require(!symlink(temporary, alias), "Cannot create path alias");
    REJECT("two distinct", distinct_runs(temporary, temporary));
    REJECT("two distinct", distinct_runs(temporary, alias));
    unlink(alias);
}
static void native_evidence(void)
{
    Json *bindings = parse_json("[{\"name\":\"outputs/result.json\",\"sha256\":\"a\"}]");
    Json *status = parse_json("{\"latest_run\":{\"metrics\":{\"utility_delta\":0.2},\"source_attestation\":{\"artifacts\":[{\"path\":\"outputs/result.json\",\"sha256\":\"a\"}]}},\"latest_evidence\":{\"resolved_outcome\":\"improved\"}}");
    Json *expected = parse_json("{\"utility_delta\":0.2}");
    verify_native_record(status, expected, "improved", bindings);
    REJECT("native ilXyr record", verify_native_record(status, parse_json("{\"utility_delta\":0.3}"), "improved", bindings));
    put(bindings->items->value, "sha256", string("b"));
    REJECT("Artifact bindings differ", verify_native_record(status, expected, "improved", bindings));
}
static void future_model(void)
{
    put(training, "feature_mode", string("replay_outcome"));
    REJECT("outcome features", metrics());
}
static void changed_checkpoint(void)
{
    put(rollout->items->value, "checkpoint_hash", string("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"));
    REJECT("different checkpoint", metrics());
}
static void controllers_contract(void)
{
    Json *shortened = array();
    for (JsonItem *i = rollout->items; i != rollout->tail; i = i->next) append(shortened, i->value);
    REJECT("four distinct", result_metrics(training, shortened, config, model_hash));
    put(rollout->items->value, "controller", get(rollout->items->next->value, "controller"));
    REJECT("four distinct", metrics());
}
static void changed_seed(void)
{
    Json *row = rollout->items->value;
    put(row, "seed", number(value(row, "seed") + 1));
    REJECT("seed differs", metrics());
}
static void nonfinite_result(void)
{
    put(rollout->items->value, "avg_utility", number(NAN));
    REJECT("Non-finite", metrics());
}
static void changed_schedule(void)
{
    put(training, "epochs", number(value(training, "epochs") + 1));
    REJECT("schedule differs", metrics());
}
static void changed_artifact(void)
{
    const char *path = join(temporary, "artifact.csv");
    write_text(path, "original\n");
    Json *bindings = array(); append(bindings, binding("train-csv", path));
    verify_bindings(bindings);
    write_text(path, "changed\n");
    REJECT("Artifact changed", verify_bindings(bindings));
    unlink(path);
}
static void trace_file(const char *path, int count, int duplicate)
{
    FILE *f = fopen(path, "wb");
    require(f != NULL, "Cannot create trace fixture");
    fputs("schema,seed,episode,tick,action,certificate_status,proof_kind,utility,damage_events,death_events,hull_loss\r\n", f);
    for (int n = 0; n < count; ++n)
        fprintf(f, "signal.flight_counterfactual.v1,4242,0,0,%d,certified,deterministic_authoritative_replay,0.1,0,0,0\r\n", duplicate && n == 8 ? 0 : n);
    require(!fclose(f), "Cannot close trace fixture");
}
static void complete_actions(void)
{
    const char *path = join(temporary, "trace.csv");
    trace_file(path, 9, 0);
    check(value(inspect_trace(path), "rows") == 9, "complete replay group");
    trace_file(path, 8, 0);
    REJECT("all nine actions", inspect_trace(path));
    trace_file(path, 9, 1);
    REJECT("all nine actions", inspect_trace(path));
    unlink(path);
}
static void json_and_csv_bytes(void)
{
    Json *j = parse_json("{\"text\":\"a,b\\r\\n\\\"a,b\\\",\\\"quoted \\\"\\\"value\\\"\\\"\\\"\\n\\u03bb\\ud83d\\ude80\",\"n\":1.25e-2}");
    const char *expected = "a,b\r\n\"a,b\",\"quoted \"\"value\"\"\"\nλ🚀";
    check(!strcmp(field(j, "text"), expected), "JSON decodes CSV and UTF-8 bytes");
    const char *path = join(temporary, "roundtrip.json");
    write_json(path, j);
    check(equal(j, read_json(path)), "JSON byte roundtrip");
    const char *literal = "\"a,b\",\"quoted \"\"value\"\"\",\"two\nlines\"\r\n";
    char *csv = text_copy(literal, strlen(literal));
    CsvRecord row = csv_record(&csv);
    check(row.count == 3 && !strcmp(row.fields[0], "a,b") && !strcmp(row.fields[1], "quoted \"value\"") && !strcmp(row.fields[2], "two\nlines"), "CSV quotes and newlines");
    REJECT("Repeated JSON key", parse_json("{\"a\":1,\"a\":2}"));
    REJECT("Non-finite", parse_json("1e400"));
    REJECT("Invalid UTF-8", parse_json("{\"value\":\"\xff\"}"));
    REJECT("Invalid JSON value", parse_json("\v{}"));
    FILE *nul = fopen(path, "wb"); require(nul != NULL, "Cannot write NUL fixture");
    const char nul_data[] = "{}\0INVALID";
    require(fwrite(nul_data, 1, sizeof(nul_data) - 1, nul) == sizeof(nul_data) - 1 && !fclose(nul), "Cannot finish NUL fixture");
    REJECT("Embedded NUL", read_json(path));
    write_text(path, "{\"value\":1}\v\n");
    REJECT("Trailing JSON text", json_lines(path));
    unlink(path);
}
static void recursive_release(void)
{
    const char *dir = join(temporary, "releases/name/v1/hash");
    mkdirs(dir);
    const char *path = join(dir, "release.json");
    write_text(path, "{}\n");
    Json *found = array(); discover_releases(join(temporary, "releases"), found, 0);
    check(found->count == 1 && !strcmp(str(found->items->value), path), "nested Braid release discovery");
    unlink(path); rmdir(dir); rmdir(parent(dir)); rmdir(parent(parent(dir))); rmdir(join(temporary, "releases"));
}
static void command_contract(void)
{
    char *captured = CMD(temporary, NULL, 1, 5, 0, "/bin/echo", "literal $(text); value");
    check(!strcmp(captured, "literal $(text); value\n"), "argv preserves literal shell characters");
    REJECT("Command failed", CMD(temporary, NULL, 1, 5, 0, "/usr/bin/false"));
    REJECT("timed out", CMD(temporary, NULL, 1, 0, 0, "/bin/sleep", "3"));
    const char *marker = join(temporary, "orphan-marker");
    demo_phase_deadline = seconds_now() + 1;
    REJECT("timed out", CMD(temporary, NULL, 1, 10, 0, "/bin/sh", "-c", "(sleep 2; printf leaked > \"$1\") & wait", "signal-demo-timeout", marker));
    demo_phase_deadline = 0;
    CMD(temporary, NULL, 0, 5, 0, "/bin/sleep", "2");
    check(!exists(marker), "deadline kills child and grandchild before marker write");
    const char *path = join(temporary, "hash.txt"); write_text(path, "abc");
    check(!strcmp(digest(path), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"), "SHA-256 known vector");
    unlink(path);
}

static void nested_cancellation(void)
{
    const char *marker = join(temporary, "nested-orphan"), *ready = format("%s.ready", marker);
    fflush(NULL);
    pid_t outer = fork();
    require(outer >= 0, "Cannot fork nested cancellation test");
    if (!outer) {
        install_signal_cleanup();
        CMD(temporary, NULL, 0, 10, 0, demo_self, "--nested-runtime", marker);
        _exit(4);
    }
    double deadline = seconds_now() + 5;
    while (!exists(ready) && seconds_now() < deadline) {
        struct timespec pause = {0, 10000000}; nanosleep(&pause, NULL);
    }
    int started = exists(ready);
    kill(outer, SIGTERM);
    int status = 0;
    while (waitpid(outer, &status, 0) < 0 && errno == EINTR) {}
    CMD(temporary, NULL, 0, 5, 0, "/bin/sleep", "3");
    check(started && WIFEXITED(status) && WEXITSTATUS(status) == 128 + SIGTERM && !exists(marker), "outer cancellation reaches nested trainer and grandchild");
    unlink(ready);
}

int main(int argc, char **argv)
{
    if (setjmp(demo_failure)) { fprintf(stderr, "%s\n", demo_error); release_allocations(); return 1; }
    if (argc == 3 && !strcmp(argv[1], "--nested-runtime")) {
        install_signal_cleanup();
        demo_execute_phase = 1;
        demo_phase_deadline = seconds_now() + 280;
        CMD(parent(argv[2]), NULL, 0, 10, 0, "/bin/sh", "-c", "trap '' TERM; printf ready > \"$2\"; (sleep 3; printf leaked > \"$1\") & wait", "signal-demo-nested", argv[2], format("%s.ready", argv[2]));
        release_allocations();
        return 0;
    }
    snprintf(temporary, sizeof(temporary), "/tmp/signal-demo-contract-XXXXXX");
    require(mkdtemp(temporary) != NULL, "Cannot create temporary test directory");
    demo_root = SIGNAL_DEMO_ROOT;
    char executable[PATH_MAX];
    require(realpath(argv[0], executable) != NULL, "Cannot resolve test executable");
    demo_self = executable;
    struct { const char *name; void (*test)(void); } cases[] = {
        {"negative measured result", negative_result}, {"fixed outcome rule", changed_rule},
        {"separate split seeds", overlapping_seeds}, {"distinct real directories", same_directory},
        {"native evidence binding", native_evidence}, {"live features", future_model},
        {"checkpoint identity", changed_checkpoint}, {"four controllers", controllers_contract},
        {"frozen rollout seed", changed_seed}, {"finite metrics", nonfinite_result},
        {"frozen training schedule", changed_schedule}, {"artifact bytes", changed_artifact},
        {"nine replay actions", complete_actions}, {"JSON and CSV bytes", json_and_csv_bytes},
        {"nested Braid releases", recursive_release}, {"process and hash runtime", command_contract},
        {"nested cancellation", nested_cancellation}
    };
    for (size_t n = 0; n < sizeof(cases) / sizeof(cases[0]); ++n) {
        fixture();
        cases[n].test();
        printf("ok %s\n", cases[n].name);
        fflush(stdout);
        release_allocations();
    }
    require(!rmdir(temporary), "Cannot remove temporary test directory");
    printf("17 contract tests passed (%u assertions)\n", assertions);
    return 0;
}
