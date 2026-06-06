#!/bin/bash
# version_bump.sh — detect change type and compute next semver.
#
# Rules:
#   Claims changed            → MAJOR  (x.0.0)
#   Only paper files changed  → PATCH  (0.0.x)
#   Pipeline code changed     → MINOR  (0.x.0)
#   Everything else           → MINOR  (0.x.0)
#
# Usage:
#   scripts/version_bump.sh --dry-run         # print what would happen, don't write
#   scripts/version_bump.sh --base REF        # compare vs REF instead of working tree

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

DRY_RUN=true   # safe by default — only write when explicitly asked
BASE_REF=""
WRITE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --write)   WRITE=true; DRY_RUN=false; shift ;;
        --base)    BASE_REF="$2"; shift 2 ;;
        *) echo "unknown: $1"; exit 1 ;;
    esac
done

# ---- Determine changed files ----
if [ -n "$BASE_REF" ]; then
    CHANGED=$(git diff --name-only "$BASE_REF" 2>/dev/null || echo "")
else
    CHANGED=$( (git diff --cached --name-only 2>/dev/null; git diff --name-only 2>/dev/null) | sort -u )
fi

if [ -z "$CHANGED" ]; then
    echo "No changes detected."
    exit 0
fi

# ---- Classification ----
PAPER_PREFIXES=(
    "paper/paper.typ" "paper/references.bib"
    "paper/generated/" "paper/sources/" "paper/agent_candidates/"
    "data/" "results/"
)
PIPELINE_FILES=(
    "src/nn_bibliography.c" "include/nn_bibliography.h"
    "examples/bibliography_vsa.c"
    "src/nn_tfidf.c" "include/nn_tfidf.h"
    "src/nn_term_bridge.c" "include/nn_term_bridge.h"
    "src/nn_holographic_ranker.c" "include/nn_holographic_ranker.h"
    "src/nn_holographic_layer.c" "include/nn_holographic_layer.h"
    "src/holographic/leos_engine.c" "include/holographic/leos_engine.h"
    "src/nn_text_vsa.c" "include/nn_text_vsa.h"
    "scripts/collect_reference_sources.sh"
)
CLAIM_FILES=(
    "paper/claims.yaml"
)

only_paper=true
touches_pipeline=false
touches_claims=false

while IFS= read -r f; do
    [ -z "$f" ] && continue
    is_paper=false
    for p in "${PAPER_PREFIXES[@]}"; do
        [[ "$f" == "$p"* ]] && is_paper=true && break
    done
    $is_paper || only_paper=false
    for pf in "${PIPELINE_FILES[@]}"; do
        [[ "$f" == "$pf" ]] && touches_pipeline=true && break
    done
    for cf in "${CLAIM_FILES[@]}"; do
        [[ "$f" == "$cf" ]] && touches_claims=true && break
    done
    if [[ "$f" == "examples/bibliography_vsa.c" ]]; then
        if [ -n "$BASE_REF" ]; then
            git diff "$BASE_REF" -- "$f" 2>/dev/null | grep -qE 'TEST_CLAIMS|DOMAINS|CLAIM_IDS' && touches_claims=true || true
        else
            { git diff --cached -- "$f" 2>/dev/null; git diff -- "$f" 2>/dev/null; } | grep -qE 'TEST_CLAIMS|DOMAINS|CLAIM_IDS' && touches_claims=true || true
        fi
    fi
    if [[ "$f" == "Makefile" ]]; then
        if [ -n "$BASE_REF" ]; then
            git diff "$BASE_REF" -- "$f" 2>/dev/null | grep -qE 'BIBLIOGRAPHY|paper-bibliography|bibliography-vsa|tfidf' && touches_pipeline=true || true
        else
            { git diff --cached -- "$f" 2>/dev/null; git diff -- "$f" 2>/dev/null; } | grep -qE 'BIBLIOGRAPHY|paper-bibliography|bibliography-vsa|tfidf' && touches_pipeline=true || true
        fi
    fi
done <<< "$CHANGED"

# ---- Determine bump ----
if $touches_claims; then
    BUMP="major"; LABEL="MAJOR (claims changed)"
elif $only_paper; then
    BUMP="patch"; LABEL="PATCH (paper only)"
elif $touches_pipeline; then
    BUMP="minor"; LABEL="MINOR (pipeline)"
else
    BUMP="minor"; LABEL="MINOR (code)"
fi

VER=$(cat VERSION)
IFS='.' read -r MAJ MIN PAT <<< "$VER"
case "$BUMP" in
    major) MAJ=$((MAJ + 1)); MIN=0; PAT=0 ;;
    minor) MIN=$((MIN + 1)); PAT=0 ;;
    patch) PAT=$((PAT + 1)) ;;
esac
NEW_VER="$MAJ.$MIN.$PAT"

echo "$LABEL: $VER → $NEW_VER"

if $WRITE; then
    echo "$NEW_VER" > VERSION
    echo "VERSION written: $NEW_VER"
fi
