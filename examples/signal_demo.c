#define _POSIX_C_SOURCE 200809L
#define _XOPEN_SOURCE 700
#ifdef __APPLE__
#define _DARWIN_C_SOURCE
#endif
#include "signal_demo_runtime.inc"

static const char *demo_root;
static const char *demo_self;
static const char *controllers[] = {"goal_heading_no_safety", "flight_steer_to", "neural_raw", "neural_guarded"};
static const char *metric_names[] = {"utility_delta", "neural_utility", "heuristic_utility", "neural_damage_rate", "heuristic_damage_rate", "neural_death_rate", "heuristic_death_rate", "train_rows", "heldout_rows", "rollout_episodes", "parameter_count"};
static const char *metric_units[] = {"utility", "utility", "utility", "fraction", "fraction", "fraction", "fraction", "rows", "rows", "episodes", "parameters"};

static void require(int condition, const char *message) { if (!condition) fail("%s", message); }
static void validate_config(Json *config)
{
    require(!strcmp(field(config, "schema"), "crlp.signal_demo.config.v1"), "Unexpected demo config schema");
    Json *rule = parse_json("{\"primary_metric\":\"utility_delta\",\"threshold\":0,\"comparison\":\"neural_guarded minus flight_steer_to\"}");
    require(equal(get(config, "outcome"), rule), "This version supports the fixed neural_guarded minus flight_steer_to utility rule at zero");
    require(!strcmp(field(get(config, "training"), "feature_mode"), "live_observable"), "The model needs live-observable features");
    Json *data = get(config, "data");
    double train = value(get(data, "train"), "seed"), heldout = value(get(data, "heldout"), "seed"), rollout = value(get(config, "rollout"), "seed");
    require(train != heldout && train != rollout && heldout != rollout, "Train, held-out, and rollout seeds must differ");
}

static Json *source_state(const char *root, const char *expected)
{
    char *path = absolute(root);
    char *head = trim(CMD(path, NULL, 1, 30, 0, "git", "rev-parse", "HEAD"));
    if (expected && strcmp(head, expected)) fail("Source pin differs at %s: %s", path, head);
    if (*trim(CMD(path, NULL, 1, 30, 0, "git", "diff", "HEAD", "--name-only"))) fail("Commit tracked source changes first: %s", path);
    Json *state = object();
    put(state, "root", string(path));
    put(state, "commit", string(head));
    return state;
}
static Json *binding(const char *name, const char *path)
{
    Json *b = object();
    put(b, "name", string(name));
    put(b, "path", string(absolute(path)));
    put(b, "sha256", string(digest(path)));
    return b;
}
static void verify_bindings(Json *bindings)
{
    require(bindings->kind == J_ARRAY, "Artifact bindings must be an array");
    for (JsonItem *i = bindings->items; i; i = i->next)
        if (strcmp(digest(field(i->value, "path")), field(i->value, "sha256"))) fail("Artifact changed: %s", field(i->value, "name"));
}
static const char *input_path(Json *inputs, const char *name)
{
    const char *result = NULL;
    for (JsonItem *i = inputs->items; i; i = i->next) {
        if (!strcmp(field(i->value, "name"), name)) {
            require(!result, "Repeated input name");
            result = field(i->value, "path");
        }
    }
    if (!result) fail("Missing input: %s", name);
    return result;
}

typedef struct { char *fields[256]; size_t count; } CsvRecord;
static CsvRecord csv_record(char **cursor)
{
    CsvRecord row = {{0}, 0};
    char *p = *cursor;
    if (!*p) return row;
    for (;;) {
        require(row.count < 256, "CSV has too many columns");
        char buffer[65536];
        size_t n = 0;
        int quoted = *p == '"';
        if (quoted) ++p;
        for (;;) {
            if (quoted && *p == '"') {
                if (p[1] == '"') ++p;
                else { ++p; break; }
            } else if (!*p || (!quoted && (*p == ',' || *p == '\r' || *p == '\n'))) {
                require(!quoted, "Unterminated quoted CSV field");
                break;
            }
            require(n + 1 < sizeof(buffer), "CSV field exceeds 64 KiB");
            buffer[n++] = *p++;
        }
        row.fields[row.count++] = text_copy(buffer, n);
        if (*p == ',') { ++p; continue; }
        require(!*p || *p == '\r' || *p == '\n', "Unexpected text after CSV field");
        if (*p == '\r') ++p;
        if (*p == '\n') ++p;
        break;
    }
    *cursor = p;
    return row;
}
static size_t csv_column(CsvRecord header, const char *name)
{
    size_t found = SIZE_MAX;
    for (size_t n = 0; n < header.count; ++n) if (!strcmp(header.fields[n], name)) {
        require(found == SIZE_MAX, "Repeated CSV column"); found = n;
    }
    if (found == SIZE_MAX) fail("Missing trace column: %s", name);
    return found;
}
static double csv_number(const char *text, const char *name)
{
    char *end = NULL;
    errno = 0;
    double v = strtod(text, &end);
    if (!*text || *end || errno == ERANGE || !isfinite(v)) fail("Non-finite or invalid trace field: %s", name);
    return v;
}
typedef struct { uint64_t seed, episode, tick; unsigned action; } TraceRow;
static uint64_t trace_integer(const char *text, const char *name)
{
    double n = csv_number(text, name);
    require(n >= 0 && n <= 9007199254740991.0 && floor(n) == n, "Trace indices must be nonnegative integers");
    return (uint64_t)n;
}
static int trace_order(const void *a, const void *b)
{
    const TraceRow *x = a, *y = b;
    if (x->seed != y->seed) return x->seed < y->seed ? -1 : 1;
    if (x->episode != y->episode) return x->episode < y->episode ? -1 : 1;
    if (x->tick != y->tick) return x->tick < y->tick ? -1 : 1;
    return 0;
}
static Json *inspect_trace(const char *path)
{
    char *text = read_text(path), *cursor = text;
    size_t capacity = 1;
    for (char *p = text; *p; ++p) if (*p == '\n' || *p == '\r') ++capacity;
    TraceRow *rows = allocate(capacity * sizeof(*rows));
    CsvRecord header = csv_record(&cursor);
    const char *names[] = {"schema", "seed", "episode", "tick", "action", "certificate_status", "proof_kind", "utility", "damage_events", "death_events", "hull_loss"};
    size_t columns[11];
    for (size_t n = 0; n < 11; ++n) columns[n] = csv_column(header, names[n]);
    size_t count = 0;
    while (*cursor) {
        CsvRecord row = csv_record(&cursor);
        if (row.count == 1 && !*row.fields[0]) continue;
        require(row.count == header.count, "Trace column count differs from the header");
        require(!strcmp(row.fields[columns[0]], "signal.flight_counterfactual.v1"), "Unexpected trace schema");
        require(!strcmp(row.fields[columns[5]], "certified") && !strcmp(row.fields[columns[6]], "deterministic_authoritative_replay"), "Uncertified trace row");
        require(count < capacity, "Trace row count exceeds input bounds");
        rows[count].seed = trace_integer(row.fields[columns[1]], "seed");
        rows[count].episode = trace_integer(row.fields[columns[2]], "episode");
        rows[count].tick = trace_integer(row.fields[columns[3]], "tick");
        uint64_t action = trace_integer(row.fields[columns[4]], "action");
        require(action < 9, "Each replay group must contain all nine actions exactly once");
        rows[count++].action = (unsigned)action;
        for (size_t n = 7; n < 11; ++n) (void)csv_number(row.fields[columns[n]], names[n]);
    }
    if (!count) fail("Empty Signal trace: %s", path);
    qsort(rows, count, sizeof(*rows), trace_order);
    size_t groups = 0;
    Json *seeds = array();
    for (size_t n = 0; n < count;) {
        size_t next = n;
        unsigned actions = 0;
        while (next < count && !trace_order(&rows[n], &rows[next])) {
            unsigned bit = 1u << rows[next++].action;
            require(!(actions & bit), "Each replay group must contain all nine actions exactly once");
            actions |= bit;
        }
        require(actions == 511 && next - n == 9, "Each replay group must contain all nine actions exactly once");
        if (!n || rows[n].seed != rows[n - 1].seed) append(seeds, number((double)rows[n].seed));
        ++groups;
        n = next;
    }
    Json *result = object();
    put(result, "rows", number((double)count)); put(result, "groups", number((double)groups));
    put(result, "seeds", seeds); put(result, "sha256", string(digest(path)));
    return result;
}
static Json *json_lines(const char *path)
{
    char *text = read_text(path), *saved = NULL;
    Json *lines = array();
    for (char *line = strtok_r(text, "\n", &saved); line; line = strtok_r(NULL, "\n", &saved)) {
        const char *content = line;
        while (*content && strchr(" \t\r", *content)) ++content;
        if (*content) append(lines, parse_json(line));
    }
    return lines;
}

static Json *result_metrics(Json *training, Json *rollout, Json *config, const char *model_hash)
{
    require(!strcmp(field(training, "schema"), "crlp.signal_flight.summary.v1"), "Unexpected training schema");
    require(!strcmp(field(training, "training_mode"), "counterfactual_utility"), "Training supervision differs from the configuration");
    require(!strcmp(field(training, "feature_mode"), "live_observable"), "Training used outcome features");
    Json *schedule = get(config, "training");
    require(value(training, "epochs") == value(schedule, "epochs") && value(training, "seed") == value(schedule, "seed"), "Training schedule differs from the configuration");
    double actual_rate = value(training, "learning_rate"), expected_rate = value(schedule, "learning_rate");
    require(isfinite(actual_rate) && fabs(actual_rate - expected_rate) <= 1e-6 * fmax(fabs(actual_rate), fabs(expected_rate)), "Learning rate differs from the configuration");
    require(!strcmp(field(training, "model_hash"), model_hash), "Training summary names a different checkpoint");
    require(value(training, "parameter_count") == 2113, "Unexpected model size");
    require(rollout->kind == J_ARRAY && rollout->count == 4, "Rollout needs four distinct matched controllers");
    Json *matched[4] = {NULL};
    Json *settings = get(config, "rollout");
    for (JsonItem *i = rollout->items; i; i = i->next) {
        Json *row = i->value;
        size_t controller = 0;
        while (controller < 4 && strcmp(field(row, "controller"), controllers[controller])) ++controller;
        require(controller < 4 && !matched[controller], "Rollout needs four distinct matched controllers");
        matched[controller] = row;
        require(!strcmp(field(row, "schema"), "crlp.signal_flight_rollout.summary.v1"), "Unexpected rollout schema");
        require(!strcmp(field(row, "checkpoint_hash"), model_hash), "Rollout used a different checkpoint");
        const char *keys[] = {"episodes", "ticks_per_episode", "seed"};
        const char *config_keys[] = {"episodes", "ticks", "seed"};
        for (size_t n = 0; n < 3; ++n)
            if (value(row, keys[n]) != value(settings, config_keys[n])) fail("Rollout %s differs from the frozen configuration", keys[n]);
        require(truth(get(row, "wall_probes")), "Rollout must include wall probes");
        for (JsonItem *v = row->items; v; v = v->next)
            if (v->value->kind == J_NUMBER && !isfinite(v->value->number)) fail("Non-finite rollout metric: %s", v->key);
    }
    Json *neural = matched[3], *baseline = matched[1], *metrics = object();
    double numbers[] = {value(neural, "avg_utility") - value(baseline, "avg_utility"), value(neural, "avg_utility"), value(baseline, "avg_utility"), value(neural, "damage_episode_rate"), value(baseline, "damage_episode_rate"), value(neural, "death_rate"), value(baseline, "death_rate"), value(training, "train_rows"), value(training, "test_rows"), value(settings, "episodes"), value(training, "parameter_count")};
    for (size_t n = 0; n < 11; ++n) {
        require(isfinite(numbers[n]), "Non-finite measured metric");
        put(metrics, metric_names[n], number(numbers[n]));
    }
    return metrics;
}
static void verify_native_record(Json *status, Json *metrics, const char *outcome, Json *bindings)
{
    Json *run = get(status, "latest_run");
    require(equal(get(run, "metrics"), metrics) && !strcmp(field(get(status, "latest_evidence"), "resolved_outcome"), outcome), "Metrics or outcome differ from the native ilXyr record");
    Json *artifacts = get(get(run, "source_attestation"), "artifacts");
    Json *recorded = object(), *expected = object();
    require(artifacts->kind == J_ARRAY && bindings->kind == J_ARRAY, "Artifact bindings require arrays");
    for (JsonItem *i = artifacts->items; i; i = i->next) {
        const char *name = field(i->value, "path");
        require(!find(recorded, name), "Native artifact names are repeated");
        put(recorded, name, get(i->value, "sha256"));
    }
    for (JsonItem *i = bindings->items; i; i = i->next) {
        const char *name = field(i->value, "name");
        require(!find(expected, name), "Artifact binding names are repeated");
        put(expected, name, get(i->value, "sha256"));
    }
    require(equal(recorded, expected), "Artifact bindings differ from the native ilXyr record");
}
static void distinct_runs(const char *first, const char *second)
{
    require(strcmp(absolute(first), absolute(second)) != 0, "Repeatability requires two distinct run directories");
}

typedef struct {
    const char *config, *signal_root, *braid_root, *ilxyr_root, *output;
    const char *prepared, *workspace, *first, *second, *receipt;
} DemoArgs;

/* Braid currently nests releases under name/version/release-id. */
static void discover_releases(const char *directory, Json *paths, unsigned depth)
{
    require(depth <= 16, "Braid release tree exceeds 16 levels");
    DIR *dir = opendir(directory);
    if (!dir) fail("Cannot read release directory: %s", directory);
    struct dirent *entry;
    while ((entry = readdir(dir))) {
        if (!strcmp(entry->d_name, ".") || !strcmp(entry->d_name, "..")) continue;
        char *path = join(directory, entry->d_name);
        struct stat s;
        if (lstat(path, &s)) { closedir(dir); fail("Cannot inspect release path: %s", path); }
        if (S_ISDIR(s.st_mode)) discover_releases(path, paths, depth + 1);
        else if (S_ISREG(s.st_mode) && !strcmp(entry->d_name, "release.json")) append(paths, string(path));
    }
    closedir(dir);
}

static const char *braid_spec_template =
    "{\"apiVersion\":\"braid/v1\",\"kind\":\"DatasetBuild\","
    "\"metadata\":{\"version\":\"v1\",\"description\":\"Deterministic Signal simulation observations\"},"
    "\"spec\":{\"purposes\":[\"research\"],"
    "\"rights\":{\"allowedLicenses\":[\"LicenseRef-Signal-Simulation\"],\"requireAttribution\":true,\"requirePinnedSnapshots\":true},"
    "\"chunking\":{\"targetCharacters\":100000000,\"overlapCharacters\":0},"
    "\"quality\":{\"minimumCharacters\":1,\"maximumCharacters\":100000000,\"minimumAlphabeticRatio\":0,\"maximumRepeatedLineRatio\":1,\"nearDuplicateHammingDistance\":0},"
    "\"selection\":{\"strategy\":\"ranked\",\"maximumDocuments\":2},"
    "\"evaluation\":{\"minimumDocuments\":2,\"minimumSourceDocuments\":2,\"maximumRejectionRatio\":0,\"minimumDocumentsBySplit\":{\"train\":1,\"validation\":1}},"
    "\"output\":{\"directory\":\"releases\",\"formats\":[\"jsonl\"]},\"publication\":{\"target\":\"none\"}}}";

static const char *number_arg(Json *j, const char *key) { return format("%.17g", value(j, key)); }
static char *snapshot_hash(char *output)
{
    char *saved = NULL;
    for (char *line = strtok_r(output, "\n", &saved); line; line = strtok_r(NULL, "\n", &saved)) {
        if (strncmp(line, "sha256: ", 8) || strlen(line + 8) != 64) continue;
        int valid = 1;
        for (char *p = line + 8; *p; ++p) if (!isdigit((unsigned char)*p) && (*p < 'a' || *p > 'f')) valid = 0;
        if (valid) return line + 8;
    }
    fail("Braid snapshot did not return a SHA-256");
    return NULL;
}
static Json *prepare(DemoArgs args)
{
    const char *config_path = absolute(args.config);
    Json *config = read_json(config_path);
    validate_config(config);
    const char *out = absolute(args.output);
    if (exists(out)) fail("Choose a fresh output directory: %s", out);
    Json *source_pins = get(config, "sources");
    Json *crlp = source_state(demo_root, NULL);
    Json *signal = source_state(args.signal_root, field(get(source_pins, "signal"), "commit"));
    Json *braid = source_state(args.braid_root, field(get(source_pins, "braid"), "commit"));
    mkdirs(out);
    const char *logs = join(out, "logs");
    mkdirs(logs);
    const char *node = which("node");
    char *node_version = trim(CMD(demo_root, NULL, 1, 30, 0, node, "-p", "process.versions.node.split('.')[0]"));
    require(atoi(node_version) >= 22, "Node 22 or newer is required");
    const char *signal_root = field(signal, "root"), *build = join(signal_root, "build-connected");
    CMD(demo_root, join(logs, "signal-configure.log"), 0, 1200, 0, "cmake", "-S", signal_root, "-B", build, "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_TESTS_ONLY=ON", "-DBUILD_TOOLS=ON");
    CMD(demo_root, join(logs, "signal-build.log"), 0, 1200, 0, "cmake", "--build", build, "--target", "flight_trace", "--parallel", "4");
    CMD(demo_root, join(logs, "model-build.log"), 0, 1200, 0, "make", "-B", "-j4", "NN_BLAS=none", format("SIGNAL_FLIGHT_SIGNAL_DIR=%s", signal_root), "build/float/signal_flight", "build/float/signal_flight_rollout");
    const char *braid_root = field(braid, "root"), *braid_cli = join(braid_root, "dist/cli.js");
    CMD(braid_root, join(logs, "braid-tool-build.log"), 0, 1200, 0, which("pnpm"), "build");
    require(exists(braid_cli), "Install pinned Braid dependencies with pnpm install --frozen-lockfile");
    Json *traces = object(), *sources = array(), *data = get(config, "data");
    const char *splits[] = {"train", "heldout"}, *braid_splits[] = {"train", "validation"};
    for (size_t n = 0; n < 2; ++n) {
        const char *split = splits[n], *raw = join(join(out, "raw"), split), *path = join(raw, "trace.csv");
        mkdirs(raw);
        Json *settings = get(data, split);
        const char *generation[] = {join(build, "flight_trace"), "--episodes", number_arg(settings, "episodes"), "--ticks", number_arg(data, "ticks"), "--seed", number_arg(settings, "seed"), "--format", "csv", "--counterfactual", "--sample-stride", number_arg(data, "sample_stride"), "--horizon-ticks", number_arg(data, "horizon_ticks"), "--explore-random", number_arg(data, "explore_random"), "--out", path, truth(get(data, "wall_probe")) ? "--wall-probe" : NULL, NULL};
        command(demo_root, join(logs, format("generate-%s.log", split)), 0, 1200, 1, generation);
        put(traces, split, inspect_trace(path));
        char *hash = snapshot_hash(CMD(demo_root, NULL, 1, 1200, 0, node, braid_cli, "snapshot", raw));
        Json *source = parse_json("{\"kind\":\"directory\",\"extensions\":[\".csv\"],\"license\":\"LicenseRef-Signal-Simulation\",\"attribution\":\"Generated from Signal by the connected flight demonstration\",\"domain\":\"flight\"}");
        put(source, "id", string(split)); put(source, "path", string(format("raw/%s", split))); put(source, "split", string(braid_splits[n]));
        Json *snapshot = object(); put(snapshot, "revision", get(signal, "commit")); put(snapshot, "sha256", string(hash)); put(source, "snapshot", snapshot);
        append(sources, source);
    }
    for (JsonItem *a = get(get(traces, "train"), "seeds")->items; a; a = a->next)
        for (JsonItem *b = get(get(traces, "heldout"), "seeds")->items; b; b = b->next)
            require(!equal(a->value, b->value), "Trace splits share seeds");
    Json *spec = parse_json(braid_spec_template);
    put(get(spec, "metadata"), "name", get(config, "id"));
    put(get(get(spec, "spec"), "selection"), "seed", get(config, "id"));
    put(get(spec, "spec"), "sources", sources);
    write_json(join(out, "braid.json"), spec);
    CMD(demo_root, join(logs, "braid-build.log"), 0, 1200, 0, node, braid_cli, "build", join(out, "braid.json"), "--output", join(out, "releases"));
    Json *manifests = array();
    discover_releases(join(out, "releases"), manifests, 0);
    require(manifests->count == 1, "Expected one Braid release");
    const char *manifest_path = str(manifests->items->value), *release = parent(manifest_path);
    Json *manifest = read_json(manifest_path);
    require(!strcmp(field(manifest, "status"), "RELEASED"), "Braid release did not pass its checks");
    CMD(demo_root, join(logs, "braid-verify.log"), 0, 1200, 0, node, braid_cli, "verify", release, "--expected", field(manifest, "releaseId"));
    const char *exports = join(out, "exports");
    mkdirs(exports);
    Json *inputs = array();
    append(inputs, binding("config", config_path)); append(inputs, binding("braid-release", manifest_path));
    append(inputs, binding("demo-runner", demo_self)); append(inputs, binding("trace-generator", join(build, "flight_trace")));
    append(inputs, binding("node", node)); append(inputs, binding("braid-cli", braid_cli));
    for (size_t n = 0; n < 2; ++n) {
        Json *records = json_lines(join(join(release, "data"), format("%s.jsonl", braid_splits[n])));
        require(records->count == 1, "Braid must preserve one complete split document");
        const char *exported = join(exports, format("%s.csv", splits[n]));
        write_text(exported, format("%s\n", field(records->items->value, "text")));
        if (strcmp(digest(exported), field(get(traces, splits[n]), "sha256"))) fail("Braid changed the %s CSV bytes", splits[n]);
        append(inputs, binding(format("%s-csv", splits[n]), exported));
    }
    append(inputs, binding("trainer", join(demo_root, "build/float/signal_flight")));
    append(inputs, binding("evaluator", join(demo_root, "build/float/signal_flight_rollout")));
    const char *output_names[] = {"checkpoint.nnckpt", "training.jsonl", "rollout.jsonl", "result.json"};
    Json *outputs = array();
    for (size_t n = 0; n < 4; ++n) { Json *item = object(); put(item, "name", string(output_names[n])); put(item, "path", string(join(out, output_names[n]))); append(outputs, item); }
    Json *prepared = object(), *states = object(), *braid_release = object();
    put(states, "crlplrimes", crlp); put(states, "signal", signal); put(states, "braid", braid);
    put(braid_release, "release_id", get(manifest, "releaseId")); put(braid_release, "release_directory", string(release)); put(braid_release, "manifest_sha256", string(digest(manifest_path)));
    put(prepared, "schema", string("crlp.signal_demo.prepared.v1")); put(prepared, "config", config); put(prepared, "output", string(out));
    put(prepared, "sources", states); put(prepared, "braid", braid_release); put(prepared, "traces", traces); put(prepared, "inputs", inputs); put(prepared, "outputs", outputs);
    write_json(join(out, "prepared.json"), prepared);
    printf("%s\n", join(out, "prepared.json"));
    return prepared;
}

static void execute(DemoArgs args)
{
    demo_phase_deadline = seconds_now() + 280;
    demo_execute_phase = 1;
    Json *prepared = read_json(args.prepared), *config = get(prepared, "config"), *inputs = get(prepared, "inputs");
    validate_config(config);
    verify_bindings(inputs);
    for (JsonItem *i = get(prepared, "sources")->items; i; i = i->next) source_state(field(i->value, "root"), field(i->value, "commit"));
    const char *out = field(prepared, "output");
    for (JsonItem *i = get(prepared, "outputs")->items; i; i = i->next) require(!exists(field(i->value, "path")), "Execution requires fresh output files");
    const char *splits[] = {"train", "heldout"};
    for (size_t n = 0; n < 2; ++n)
        require(equal(inspect_trace(input_path(inputs, format("%s-csv", splits[n]))), get(get(prepared, "traces"), splits[n])), "Trace metadata changed");
    Json *train = get(config, "training"), *settings = get(config, "rollout");
    CMD(demo_root, join(out, "logs/train.log"), 0, 1200, 0, input_path(inputs, "trainer"), "--train", input_path(inputs, "train-csv"), "--test", input_path(inputs, "heldout-csv"), "--epochs", number_arg(train, "epochs"), "--learning-rate", number_arg(train, "learning_rate"), "--seed", number_arg(train, "seed"), "--live-features", "--save", join(out, "checkpoint.nnckpt"), "--summary", join(out, "training.jsonl"));
    CMD(demo_root, join(out, "logs/rollout.log"), 0, 1200, 0, input_path(inputs, "evaluator"), "--checkpoint", join(out, "checkpoint.nnckpt"), "--episodes", number_arg(settings, "episodes"), "--ticks", number_arg(settings, "ticks"), "--seed", number_arg(settings, "seed"), "--summary", join(out, "rollout.jsonl"));
    Json *summaries = json_lines(join(out, "training.jsonl")), *rollout = json_lines(join(out, "rollout.jsonl"));
    require(summaries->count == 1, "Expected one training summary");
    const char *model_hash = digest(join(out, "checkpoint.nnckpt"));
    Json *metrics = result_metrics(summaries->items->value, rollout, config, model_hash);
    require(value(metrics, "train_rows") == value(get(get(prepared, "traces"), "train"), "rows") && value(metrics, "heldout_rows") == value(get(get(prepared, "traces"), "heldout"), "rows"), "Trainer did not consume the complete Braid exports");
    verify_bindings(inputs);
    Json *result = object(), *outputs = array();
    const char *names[] = {"checkpoint.nnckpt", "training.jsonl", "rollout.jsonl"};
    for (size_t n = 0; n < 3; ++n) append(outputs, binding(names[n], join(out, names[n])));
    put(result, "schema", string("crlp.signal_demo.result.v1")); put(result, "metrics", metrics); put(result, "rollout", rollout);
    put(result, "braid_release_id", get(get(prepared, "braid"), "release_id")); put(result, "checkpoint_sha256", string(model_hash));
    put(result, "inputs", inputs); put(result, "outputs", outputs); put(result, "scope", string("One deterministic local flight study with fixed seeds and four matched controllers."));
    write_json(join(out, "result.json"), result);
    Json *stdout_record = object(); put(stdout_record, "metrics", metrics); emit_json(stdout, stdout_record); fputc('\n', stdout);
    demo_phase_deadline = 0;
    demo_execute_phase = 0;
}

static void verify(DemoArgs args)
{
    const char *out = absolute(args.output);
    Json *prepared = read_json(join(out, "prepared.json")), *config = get(prepared, "config");
    validate_config(config);
    require(!strcmp(absolute(field(prepared, "output")), out), "Prepared execution belongs to a different output directory");
    Json *result = read_json(join(out, "result.json"));
    verify_bindings(get(prepared, "inputs")); verify_bindings(get(result, "outputs"));
    Json *native = get(read_json(join(out, "ilxyr/frozen-config.json")), "config");
    Json *bindings = read_json(join(out, "ilxyr/artifact-bindings.json"));
    verify_bindings(bindings);
    const char *braid_cli = join(field(get(get(prepared, "sources"), "braid"), "root"), "dist/cli.js");
    Json *braid = get(prepared, "braid");
    CMD(demo_root, join(out, "logs/braid-final-verify.log"), 0, 1200, 0, input_path(get(prepared, "inputs"), "node"), braid_cli, "verify", field(braid, "release_directory"), "--expected", field(braid, "release_id"));
    const char *ilxyr = field(get(native, "ilxyr"), "cli"), *ledger_path = join(out, "ilxyr/ledger");
    Json *ledger = parse_json(CMD(demo_root, NULL, 1, 1200, 0, ilxyr, "verify", ledger_path));
    require(truth(get(ledger, "valid")), "ilXyr ledger verification failed");
    Json *status = parse_json(CMD(demo_root, NULL, 1, 1200, 0, ilxyr, "status", ledger_path, field(native, "id")));
    Json *training = json_lines(join(out, "training.jsonl"));
    require(training->count == 1, "Expected one training summary");
    Json *expected = result_metrics(training->items->value, json_lines(join(out, "rollout.jsonl")), config, digest(join(out, "checkpoint.nnckpt")));
    Json *receipt = read_json(join(out, "ilxyr/receipt.json"));
    require(!strcmp(absolute(field(receipt, "ledger")), ledger_path), "Receipt belongs to a different ledger");
    require(equal(get(result, "metrics"), expected) && equal(expected, get(receipt, "metrics")), "Recorded metrics differ from the measured artifacts");
    const char *outcome = field(receipt, "outcome");
    verify_native_record(status, expected, outcome, bindings);
    require(truth(get(receipt, "executed")) && (!strcmp(outcome, "improved") || !strcmp(outcome, "not_improved")), "Experiment did not finish");
    require((!strcmp(outcome, "improved")) == (value(expected, "utility_delta") > 0), "Outcome differs from its frozen rule");
    Json *check = object();
    put(check, "schema", string("crlp.signal_demo.verification.v1")); put(check, "valid", boolean(1)); put(check, "outcome", string(outcome));
    put(check, "metrics", expected); put(check, "braid_release_id", get(braid, "release_id")); put(check, "checkpoint_sha256", string(digest(join(out, "checkpoint.nnckpt")))); put(check, "ledger", ledger); put(check, "result_sha256", string(digest(join(out, "result.json"))));
    write_json(join(out, "verification.json"), check);
    emit_json(stdout, check); fputc('\n', stdout);
}

static void run(DemoArgs args)
{
    Json *prepared = prepare(args), *config = get(prepared, "config");
    const char *out = field(prepared, "output"), *prepared_path = join(out, "prepared.json");
    Json *ilxyr = source_state(args.ilxyr_root, field(get(get(config, "sources"), "ilxyr"), "commit"));
    const char *ilxyr_root = field(ilxyr, "root");
    CMD(ilxyr_root, join(out, "logs/ilxyr-build.log"), 0, 1200, 0, "cargo", "build", "--locked", "-p", "ilxyr-cli");
    put(ilxyr, "cli", string(join(ilxyr_root, "target/debug/ilxyr")));
    Json *native = object(), *source = object(), *runner = object(), *argv = array(), *inputs = array(), *metrics = array();
    Json *crlp = get(get(prepared, "sources"), "crlplrimes");
    put(source, "repository", string("https://github.com/atimics/crlplrimes.git")); put(source, "root", get(crlp, "root")); put(source, "commit", get(crlp, "commit"));
    append(argv, string("execute")); append(argv, string("--prepared")); append(argv, string(prepared_path));
    put(runner, "mode", string("native")); put(runner, "program", string(demo_self)); put(runner, "args", argv); put(runner, "cwd", string(demo_root)); put(runner, "timeout_seconds", number(300));
    for (JsonItem *i = get(prepared, "inputs")->items; i; i = i->next) append(inputs, i->value);
    append(inputs, binding("prepared", prepared_path));
    for (size_t n = 0; n < 11; ++n) {
        Json *m = object(); char *description = text_copy(metric_names[n], strlen(metric_names[n]));
        for (char *p = description; *p; ++p) if (*p == '_') *p = ' ';
        put(m, "name", string(metric_names[n])); put(m, "unit", string(metric_units[n])); put(m, "description", string(description)); append(metrics, m);
    }
    put(native, "id", get(config, "id")); put(native, "title", string("Braid to tiny flight model to measured Signal rollout")); put(native, "seed", get(get(config, "rollout"), "seed"));
    put(native, "source", source); put(native, "ilxyr", ilxyr); put(native, "runner", runner); put(native, "inputs", inputs); put(native, "outputs", get(prepared, "outputs")); put(native, "metrics", metrics);
    put(native, "outcome", parse_json("{\"primary_metric\":\"utility_delta\",\"threshold\":0}"));
    write_json(join(out, "ilxyr-config.json"), native);
    CMD(demo_root, join(out, "logs/ilxyr-execution.log"), 0, 400, 0, which("node"), join(demo_root, "scripts/signal_demo_ilxyr.mjs"), "--config", join(out, "ilxyr-config.json"), "--output", join(out, "ilxyr"), "--execute");
    verify((DemoArgs){.output = out});
}

static Json *source_commits(Json *prepared)
{
    Json *result = object();
    for (JsonItem *i = get(prepared, "sources")->items; i; i = i->next) put(result, i->key, get(i->value, "commit"));
    return result;
}
static void compare(DemoArgs args)
{
    const char *runs[] = {absolute(args.first), absolute(args.second)};
    distinct_runs(runs[0], runs[1]);
    Json *prepared[2], *results[2];
    for (size_t n = 0; n < 2; ++n) {
        prepared[n] = read_json(join(runs[n], "prepared.json")); results[n] = read_json(join(runs[n], "result.json"));
        verify((DemoArgs){.output = runs[n]});
        verify_bindings(get(prepared[n], "inputs")); verify_bindings(get(results[n], "outputs"));
    }
    Json *checks = object();
    put(checks, "config", boolean(equal(get(prepared[0], "config"), get(prepared[1], "config"))));
    put(checks, "source_commits", boolean(equal(source_commits(prepared[0]), source_commits(prepared[1]))));
    put(checks, "braid_release_id", boolean(equal(get(results[0], "braid_release_id"), get(results[1], "braid_release_id"))));
    put(checks, "exported_csv", boolean(equal(get(prepared[0], "traces"), get(prepared[1], "traces"))));
    put(checks, "checkpoint", boolean(equal(get(results[0], "checkpoint_sha256"), get(results[1], "checkpoint_sha256"))));
    put(checks, "metrics", boolean(equal(get(results[0], "metrics"), get(results[1], "metrics"))));
    put(checks, "all_controllers", boolean(equal(get(results[0], "rollout"), get(results[1], "rollout"))));
    for (JsonItem *i = checks->items; i; i = i->next) if (!truth(i->value)) fail("Repeated runs differ: %s", i->key);
    Json *receipt = object(), *bindings = array();
    for (size_t n = 0; n < 2; ++n) append(bindings, binding(format("run-%zu", n + 1), join(runs[n], "result.json")));
    put(receipt, "schema", string("crlp.signal_demo.repeatability.v1")); put(receipt, "checks", checks); put(receipt, "runs", bindings);
    if (args.receipt) write_json(args.receipt, receipt);
    emit_json(stdout, receipt); fputc('\n', stdout);
}
static void reproduce(DemoArgs args)
{
    const char *workspace = absolute(args.workspace);
    if (exists(workspace)) fail("Choose a fresh demonstration workspace: %s", workspace);
    source_state(demo_root, NULL);
    const char *config_path = join(demo_root, "integrations/signal-demo/config.json");
    Json *config = read_json(config_path);
    validate_config(config);
    const char *braid_root = args.braid_root ? absolute(args.braid_root) : join(workspace, "braid");
    if (args.braid_root) source_state(braid_root, field(get(get(config, "sources"), "braid"), "commit"));
    mkdirs(workspace);
    for (JsonItem *i = get(config, "sources")->items; i; i = i->next) {
        if (args.braid_root && !strcmp(i->key, "braid")) continue;
        const char *path = join(workspace, i->key);
        mkdirs(path);
        CMD(path, NULL, 0, 1200, 0, "git", "init", "--quiet");
        CMD(path, NULL, 0, 1200, 0, "git", "remote", "add", "origin", field(i->value, "repository"));
        CMD(path, NULL, 0, 1200, 0, "git", "fetch", "--depth", "1", "origin", field(i->value, "commit"));
        CMD(path, NULL, 0, 1200, 0, "git", "checkout", "--detach", "FETCH_HEAD");
    }
    CMD(braid_root, NULL, 0, 1200, 0, "pnpm", "install", "--frozen-lockfile");
    for (size_t n = 1; n <= 2; ++n) run((DemoArgs){.config = config_path, .signal_root = join(workspace, "signal"), .braid_root = braid_root, .ilxyr_root = join(workspace, "ilxyr"), .output = join(workspace, format("run-%zu", n))});
    compare((DemoArgs){.first = join(workspace, "run-1"), .second = join(workspace, "run-2"), .receipt = join(workspace, "repeatability.json")});
}

#ifndef SIGNAL_DEMO_TEST
static void usage(void)
{
    fputs("signal_demo prepare|run --signal-root DIR --braid-root DIR --output DIR [--config FILE] [--ilxyr-root DIR]\n"
          "signal_demo execute --prepared FILE\n"
          "signal_demo verify --output DIR\n"
          "signal_demo reproduce --workspace DIR [--braid-root DIR]\n"
          "signal_demo compare FIRST SECOND [--receipt FILE]\n", stderr);
}
int main(int argc, char **argv)
{
    if (setjmp(demo_failure)) { fprintf(stderr, "signal-demo: %s\n", demo_error); release_allocations(); return 1; }
    if (argc < 2 || !strcmp(argv[1], "--help")) { usage(); return argc < 2 ? 1 : 0; }
    install_signal_cleanup();
    demo_self = strchr(argv[0], '/') ? absolute(argv[0]) : which(argv[0]);
    demo_root = absolute(SIGNAL_DEMO_ROOT);
    DemoArgs args = {.config = join(demo_root, "integrations/signal-demo/config.json")};
    for (int n = 2; n < argc; ++n) {
        const char *key = argv[n];
        if (strncmp(key, "--", 2)) {
            if (strcmp(argv[1], "compare") || args.second) fail("Unexpected argument: %s", key);
            if (!args.first) args.first = key; else args.second = key;
            continue;
        }
        if (++n == argc) fail("Missing argument for %s", key);
        const char *v = argv[n];
        if (!strcmp(key, "--config")) args.config = v;
        else if (!strcmp(key, "--signal-root")) args.signal_root = v;
        else if (!strcmp(key, "--braid-root")) args.braid_root = v;
        else if (!strcmp(key, "--ilxyr-root")) args.ilxyr_root = v;
        else if (!strcmp(key, "--output")) args.output = v;
        else if (!strcmp(key, "--prepared")) args.prepared = v;
        else if (!strcmp(key, "--workspace")) args.workspace = v;
        else if (!strcmp(key, "--receipt")) args.receipt = v;
        else fail("Unknown option: %s", key);
    }
    if (!strcmp(argv[1], "prepare") || !strcmp(argv[1], "run")) {
        require(args.signal_root && args.braid_root && args.output, "Signal root, Braid root, and output are required");
        if (!strcmp(argv[1], "run")) { require(args.ilxyr_root != NULL, "ilXyr root is required"); run(args); } else prepare(args);
    } else if (!strcmp(argv[1], "execute")) { require(args.prepared != NULL, "Prepared file is required"); execute(args); }
    else if (!strcmp(argv[1], "verify")) { require(args.output != NULL, "Output directory is required"); verify(args); }
    else if (!strcmp(argv[1], "reproduce")) { require(args.workspace != NULL, "Workspace is required"); reproduce(args); }
    else if (!strcmp(argv[1], "compare")) { require(args.first && args.second, "Two run directories are required"); compare(args); }
    else fail("Unknown mode: %s", argv[1]);
    release_allocations();
    return 0;
}
#endif
