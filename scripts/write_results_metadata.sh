#!/bin/sh
set -eu

results_dir=${1:-results/latest}
real=${2:-float}
cc=${3:-cc}

canonical="$results_dir/comparison.canonical.jsonl"
commands="$results_dir/commands.txt"
replication="$results_dir/replication.summary.jsonl"
cost_verdict="$results_dir/cost_verdict.jsonl"
robustness_verdict="$results_dir/robustness_verdict.jsonl"

test -s "$canonical"
test -s "$commands"
test -s "$replication"
test -s "$cost_verdict"
test -s "$robustness_verdict"

hash_file() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

json_escape() {
    awk 'BEGIN { ORS="" }
        { gsub(/\\/, "\\\\"); gsub(/\"/, "\\\"");
          if (NR > 1) printf "\\n"; printf "%s", $0 }'
}

commit=unknown
dirty=unknown
if git rev-parse HEAD >/dev/null 2>&1; then
    commit=$(git rev-parse HEAD)
    if git diff --quiet --ignore-submodules HEAD --; then dirty=false; else dirty=true; fi
fi

generated_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
compiler_version=$($cc --version 2>/dev/null | sed -n '1p' || printf unknown)
canonical_sha=$(hash_file "$canonical")
commands_sha=$(hash_file "$commands")
replication_sha=$(hash_file "$replication")
cost_verdict_sha=$(hash_file "$cost_verdict")
robustness_verdict_sha=$(hash_file "$robustness_verdict")
row_count=$(wc -l < "$canonical" | tr -d ' ')
command_count=$(wc -l < "$commands" | tr -d ' ')
compiler_json=$(printf '%s' "$compiler_version" | json_escape)

printf '{"schema":"crlp.environment_manifest.v2","generated_utc":"%s","source_commit":"%s","worktree_dirty":%s,"real":"%s","cc":"%s","compiler_version":"%s","canonical_path":"%s","canonical_sha256":"%s","canonical_rows":%s,"commands_path":"%s","commands_sha256":"%s","command_count":%s,"replication_path":"%s","replication_sha256":"%s","cost_verdict_path":"%s","cost_verdict_sha256":"%s","robustness_verdict_path":"%s","robustness_verdict_sha256":"%s"}\n' \
    "$generated_utc" "$commit" "$dirty" "$real" "$cc" "$compiler_json" \
    "$canonical" "$canonical_sha" "$row_count" "$commands" "$commands_sha" \
    "$command_count" "$replication" "$replication_sha" "$cost_verdict" \
    "$cost_verdict_sha" "$robustness_verdict" "$robustness_verdict_sha" \
    > "$results_dir/environment.json"
