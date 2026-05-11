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

expect_usage_failure "prime-gap CLI rejects interval sieve-limit overflow" 16 4294967000
expect_usage_failure "prime-gap CLI rejects non-increasing train/test range" 1000 1000
expect_usage_failure "prime-gap CLI rejects tiny train range" 15 3000

expect_output_contains \
    "prime-gap CLI caps unlimited wide sweep" \
    "Wide-window runs are capped at train_states=5000 test_states=5000 epochs=0" \
    1000 3000 0 0 0 0

expect_output_contains \
    "prime-gap CLI preserves below-cap wide sweep limits" \
    "Wide-window runs are capped at train_states=123 test_states=45 epochs=0" \
    1000 3000 0 0 123 45
