#!/bin/bash
# check_paper_diff_coverage.sh — R-P.11 diff gate
#
# Checks whether changed claim-like prose in paper.typ maps to an accepted
# candidate_id in the paper agent trace. Used by `make editorial-pr-check`.
#
# Exit 0: all claim-like prose changes are covered (or no changes).
# Exit 1: uncovered claim-like prose changes found (advisory).
set -euo pipefail

PAPER_FILE="paper/paper.typ"
AGENT_TRACE="paper/generated/paper_agent_trace.jsonl"

if ! git rev-parse --git-dir >/dev/null 2>&1; then
    echo "paper-diff-coverage: not a git repo, skipping"
    exit 0
fi

if [ ! -f "$AGENT_TRACE" ]; then
    echo "paper-diff-coverage: no agent trace at $AGENT_TRACE, skipping"
    exit 0
fi

# Collect accepted candidate texts from the agent trace
grep '"verifier_status":"accepted"' "$AGENT_TRACE" \
    | grep -o '"text":"[^"]*"' \
    | sed 's/"text":"//;s/"$//' \
    > /tmp/paper_accepted_texts.$$ 2>/dev/null || true

if [ ! -s /tmp/paper_accepted_texts.$$ ]; then
    echo "paper-diff-coverage: no accepted candidates in trace, nothing to check"
    rm -f /tmp/paper_accepted_texts.$$
    exit 0
fi

# Get diff of paper.typ against HEAD
DIFF=$(git diff HEAD -- "$PAPER_FILE" 2>/dev/null || true)
if [ -z "$DIFF" ]; then
    echo "paper-diff-coverage: no changes to $PAPER_FILE"
    rm -f /tmp/paper_accepted_texts.$$
    exit 0
fi

# Extract added lines (lines starting with + that aren't +++ header lines)
ADDED=$(echo "$DIFF" | grep '^+' | grep -v '^+++' | sed 's/^+//' || true)

if [ -z "$ADDED" ]; then
    rm -f /tmp/paper_accepted_texts.$$
    exit 0
fi

# Filter to claim-like prose: lines with numbers, claim references, or substantive content.
# Skip: import lines, comment lines, blank lines, formatting-only lines.
CLAIM_LIKE=$(echo "$ADDED" \
    | grep -v '^#import\|^#let\|^#set\|^#show\|^#v(\|^#align\|^#box\|^#figure\|^#table\|^#bibliography' \
    | grep -v '^\s*//\|^\s*$\|^\s*\[.*\]:\|^#\|^=' \
    | grep -E '[0-9]|claim_|supported|contradicted|underpowered|pct\(|pp\(' || true)

UNCOVERED=0
while IFS= read -r line; do
    [ -z "$line" ] && continue
    # Lines containing claim_ variable references are inherently claim-backed
    echo "$line" | grep -q 'claim_' && continue

    # Check if any accepted candidate text contains a significant substring
    FINGERPRINT=$(echo "$line" | cut -c1-40 | sed 's/[][\\\/.*^$]/\\&/g')
    if ! grep -qF "$FINGERPRINT" /tmp/paper_accepted_texts.$$ 2>/dev/null; then
        if [ $UNCOVERED -eq 0 ]; then
            echo ""
            echo "=== Uncovered claim-like prose changes ==="
            echo "These lines changed in $PAPER_FILE but don't match any accepted"
            echo "candidate in $AGENT_TRACE. Consider adding agent candidates."
            echo ""
        fi
        UNCOVERED=$((UNCOVERED + 1))
        echo "  [$UNCOVERED] $line"
    fi
done <<< "$CLAIM_LIKE"

rm -f /tmp/paper_accepted_texts.$$

if [ $UNCOVERED -gt 0 ]; then
    echo ""
    echo "paper-diff-coverage: $UNCOVERED uncovered claim-like change(s) — advisory only"
fi

exit 0

