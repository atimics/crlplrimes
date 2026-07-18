#!/bin/sh
set -eu

results_dir=${1:-results/latest}
canonical="$results_dir/comparison.canonical.jsonl"
commands="$results_dir/commands.txt"
environment="$results_dir/environment.json"
summary="$results_dir/summary.md"
replication="$results_dir/replication.summary.jsonl"
cost_verdict="$results_dir/cost_verdict.jsonl"
robustness_verdict="$results_dir/robustness_verdict.jsonl"

for path in "$canonical" "$commands" "$environment" "$summary" "$replication" "$cost_verdict" "$robustness_verdict"; do
    if [ ! -s "$path" ]; then echo "results-check: missing or empty $path" >&2; exit 1; fi
done

hash_file() {
    if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
    else shasum -a 256 "$1" | awk '{print $1}'; fi
}

expected_canonical=$(sed -n 's/.*"canonical_sha256":"\([0-9a-f]*\)".*/\1/p' "$environment")
expected_commands=$(sed -n 's/.*"commands_sha256":"\([0-9a-f]*\)".*/\1/p' "$environment")
expected_replication=$(sed -n 's/.*"replication_sha256":"\([0-9a-f]*\)".*/\1/p' "$environment")
expected_cost_verdict=$(sed -n 's/.*"cost_verdict_sha256":"\([0-9a-f]*\)".*/\1/p' "$environment")
expected_robustness_verdict=$(sed -n 's/.*"robustness_verdict_sha256":"\([0-9a-f]*\)".*/\1/p' "$environment")
actual_canonical=$(hash_file "$canonical")
actual_commands=$(hash_file "$commands")
actual_replication=$(hash_file "$replication")
actual_cost_verdict=$(hash_file "$cost_verdict")
actual_robustness_verdict=$(hash_file "$robustness_verdict")

test "$expected_canonical" = "$actual_canonical" || { echo "results-check: canonical hash mismatch" >&2; exit 1; }
test "$expected_commands" = "$actual_commands" || { echo "results-check: commands hash mismatch" >&2; exit 1; }
test "$expected_replication" = "$actual_replication" || { echo "results-check: replication hash mismatch" >&2; exit 1; }
test "$expected_cost_verdict" = "$actual_cost_verdict" || { echo "results-check: cost verdict hash mismatch" >&2; exit 1; }
test "$expected_robustness_verdict" = "$actual_robustness_verdict" || { echo "results-check: robustness verdict hash mismatch" >&2; exit 1; }
grep -q "$actual_canonical" "$summary" || { echo "results-check: summary does not identify canonical artifact" >&2; exit 1; }
grep -q "$actual_commands" "$summary" || { echo "results-check: summary does not identify command manifest" >&2; exit 1; }
grep -q "$actual_replication" "$summary" || { echo "results-check: summary does not identify replication artifact" >&2; exit 1; }
grep -q "$actual_cost_verdict" "$summary" || { echo "results-check: summary does not identify cost verdict" >&2; exit 1; }
grep -q "$actual_robustness_verdict" "$summary" || { echo "results-check: summary does not identify robustness verdict" >&2; exit 1; }

test "$(grep -c '"generator_seed_count":5' "$replication")" -ge 7 || {
    echo "results-check: replication summary is incomplete" >&2; exit 1;
}
grep -q '"comparison":"cheap_bound_vs_raw".*"verdict":"improves_total_cost".*"efficiency_claim_promotable":true' "$cost_verdict" || {
    echo "results-check: cheap-bound total-cost promotion gate failed" >&2; exit 1;
}
grep -q '"support_aware_false_rejections":0.*"bounded_safety_pass":true' "$robustness_verdict" || {
    echo "results-check: bounded-safety promotion gate failed" >&2; exit 1;
}

for task in prime_gap grid_path delayed_claim; do
    grep -q "\"task\":\"$task\"" "$canonical" || { echo "results-check: missing promoted task $task" >&2; exit 1; }
done

if awk 'index($0, "\"schema\":\"crlp.compare.v1\"") &&
        $0 ~ /"task":"(arc_agi|program_synthesis|haiku_word_lm|haiku_understanding|signal_flight|signal_mining)"/ { found=1 }
        END { exit !found }' "$canonical"; then
    echo "results-check: sandbox task leaked into promoted canonical results" >&2
    exit 1
fi

echo "results-check: bundle is internally consistent ($actual_canonical)"
