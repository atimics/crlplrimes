#!/bin/sh
set -eu

if [ "$#" -lt 3 ]; then
    echo "usage: $0 RESULTS_DIR REAL TARGET..." >&2
    exit 2
fi

results_dir=$1
real=$2
shift 2

mkdir -p "$results_dir/logs"

for target in "$@"; do
    echo "make REAL=$real $target" >> "$results_dir/commands.txt"
    if ! make REAL="$real" "$target" > "$results_dir/logs/$target.log" 2>&1; then
        cat "$results_dir/logs/$target.log" >&2
        exit 1
    fi
    awk '/^\{/{print}' "$results_dir/logs/$target.log" >> "$results_dir/comparison.jsonl"
done
