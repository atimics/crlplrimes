#!/bin/sh
set -eu

if [ "$#" -ne 1 ]; then
    echo "usage: test_prime_gap_cli.sh path/to/prime_gap" >&2
    exit 2
fi

prime_gap_bin=$1

fail()
{
    echo "FAIL $1" >&2
    if [ "${2:-}" != "" ] && [ -f "$2" ]; then
        sed -n '1,40p' "$2" >&2
    fi
    exit 1
}

pass()
{
    echo "ok   $1"
}

tmp_output()
{
    mktemp "${TMPDIR:-/tmp}/prime-gap-cli.XXXXXX"
}

expect_usage_failure()
{
    name=$1
    shift
    output=$(tmp_output)

    if "$prime_gap_bin" "$@" >"$output" 2>&1; then
        fail "$name accepted invalid arguments" "$output"
    fi
    if ! grep -q "usage: prime_gap" "$output"; then
        fail "$name did not print usage" "$output"
    fi
    if grep -q "sieve limit=" "$output"; then
        fail "$name started the experiment after invalid arguments" "$output"
    fi

    rm -f "$output"
    pass "$name"
}

expect_output_contains()
{
    name=$1
    expected=$2
    shift 2
    output=$(tmp_output)

    if ! "$prime_gap_bin" "$@" >"$output" 2>&1; then
        fail "$name failed unexpectedly" "$output"
    fi
    if ! grep -Fq "$expected" "$output"; then
        fail "$name missing expected output: $expected" "$output"
    fi

    rm -f "$output"
    pass "$name"
}

expect_output_omits()
{
    name=$1
    unexpected=$2
    shift 2
    output=$(tmp_output)

    if ! "$prime_gap_bin" "$@" >"$output" 2>&1; then
        fail "$name failed unexpectedly" "$output"
    fi
    if grep -Fq "$unexpected" "$output"; then
        fail "$name included unexpected output: $unexpected" "$output"
    fi

    rm -f "$output"
    pass "$name"
}

expect_summary_csv()
{
    name=$1
    shift
    output=$(tmp_output)
    summary=$(tmp_output)
    rm -f "$summary"

    if ! "$prime_gap_bin" "$@" "$summary" >"$output" 2>&1; then
        fail "$name failed unexpectedly" "$output"
    fi
    if [ ! -s "$summary" ]; then
        fail "$name did not write summary CSV" "$output"
    fi
    if ! head -n 1 "$summary" | grep -Fq "kind,section,label"; then
        fail "$name missing CSV header" "$summary"
    fi
    if ! grep -Fq '"eval","baselines","baseline empirical-gap"' "$summary"; then
        fail "$name missing eval summary row" "$summary"
    fi
    if ! grep -Fq '"interval","interval-prime-ranking","baseline interval-frequency"' "$summary"; then
        fail "$name missing interval summary row" "$summary"
    fi

    rm -f "$output" "$summary"
    pass "$name"
}

expect_trace_jsonl()
{
    name=$1
    shift
    output=$(tmp_output)
    trace=$(tmp_output)
    checkpoint="${trace}.nnckpt"
    rm -f "$trace" "$checkpoint"

    if ! "$prime_gap_bin" "$@" --trace "$trace" >"$output" 2>&1; then
        fail "$name failed unexpectedly" "$output"
    fi
    if [ ! -s "$trace" ]; then
        fail "$name did not write trace JSONL" "$output"
    fi
    if [ ! -s "$checkpoint" ]; then
        fail "$name did not write companion checkpoint" "$output"
    fi
    if ! grep -Fq '"schema":"crlp.trace.v1"' "$trace"; then
        fail "$name missing trace schema" "$trace"
    fi
    if ! grep -Fq '"task":"prime_gap"' "$trace"; then
        fail "$name missing prime-gap task name" "$trace"
    fi
    if ! grep -Fq '"verifier_cost":{"p":' "$trace"; then
        fail "$name missing verifier cost payload" "$trace"
    fi
    if ! grep -Fq "model_checkpoint=$checkpoint" "$output"; then
        fail "$name did not print companion checkpoint path" "$output"
    fi

    rm -f "$output" "$trace" "$checkpoint"
    pass "$name"
}

expect_trace_jsonl_before_positionals()
{
    name=$1
    shift
    output=$(tmp_output)
    trace=$(tmp_output)
    checkpoint="${trace}.nnckpt"
    rm -f "$trace" "$checkpoint"

    if ! "$prime_gap_bin" --trace "$trace" "$@" >"$output" 2>&1; then
        fail "$name failed unexpectedly" "$output"
    fi
    if [ ! -s "$trace" ]; then
        fail "$name did not write trace JSONL" "$output"
    fi
    if [ ! -s "$checkpoint" ]; then
        fail "$name did not write companion checkpoint" "$output"
    fi
    if ! grep -Fq "trace_jsonl=$trace" "$output"; then
        fail "$name did not parse trace flag before positionals" "$output"
    fi

    rm -f "$output" "$trace" "$checkpoint"
    pass "$name"
}

expect_usage_failure "prime-gap CLI rejects interval sieve-limit overflow" 16 4294967000
expect_usage_failure "prime-gap CLI rejects non-increasing train/test range" 1000 1000
expect_usage_failure "prime-gap CLI rejects tiny train range" 15 3000
expect_usage_failure "prime-gap CLI rejects nonnumeric numeric argument" 1000 nope
expect_usage_failure "prime-gap CLI rejects invalid verifier diagnostics mode" 1000 3000 0 0 10 10 2
expect_usage_failure "prime-gap CLI rejects unknown flag" 1000 3000 --bogus
expect_usage_failure "prime-gap CLI rejects too many arguments" 1000 3000 0 0 10 10 0 a b c
expect_usage_failure "prime-gap CLI rejects missing trace path" 1000 3000 0 0 10 10 0 --trace

expect_output_contains \
    "prime-gap CLI caps unlimited wide sweep" \
    "Wide-window runs are capped at train_states=5000 test_states=5000 epochs=0" \
    1000 3000 0 0 0 0

expect_output_contains \
    "prime-gap CLI preserves below-cap wide sweep limits" \
    "Wide-window runs are capped at train_states=123 test_states=45 epochs=0" \
    1000 3000 0 0 123 45

expect_output_contains \
    "prime-gap CLI defaults verifier diagnostics off" \
    "verifier_diagnostics=off" \
    1000 3000 0 0 20 20

expect_output_omits \
    "prime-gap CLI omits verifier detail by default" \
    "wheel exact" \
    1000 3000 0 0 20 20

expect_output_contains \
    "prime-gap CLI enables verifier diagnostics explicitly" \
    "verifier_diagnostics=on" \
    1000 3000 0 0 20 20 1

expect_output_contains \
    "prime-gap CLI prints verifier detail when enabled" \
    "wheel exact" \
    1000 3000 0 0 20 20 1

expect_summary_csv \
    "prime-gap CLI writes structured summary CSV" \
    1000 3000 0 0 20 20 0

expect_trace_jsonl \
    "prime-gap CLI writes trace JSONL" \
    1000 3000 0 0 20 20 0

expect_trace_jsonl_before_positionals \
    "prime-gap CLI accepts trace flag before numeric args" \
    1000 3000 0 0 20 20 0
