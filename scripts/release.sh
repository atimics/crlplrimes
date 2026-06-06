#!/bin/bash
# release.sh — detect change type, bump VERSION, tag, and push.
# The version bump classification reflects the project's inverted structure:
# the code generates the paper, so claims=MAJOR, pipeline=MINOR, paper-only=PATCH.
#
# Usage:
#   scripts/release.sh              # bump, commit VERSION, tag, push
#   scripts/release.sh --dry-run    # show what would happen

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

if ! $DRY_RUN; then
    if ! git diff --quiet || ! git diff --cached --quiet; then
        echo "ERROR: working tree is dirty. Commit or stash changes first."
        exit 1
    fi
fi

# Detect bump type and compute new version (safe — doesn't write)
BUMP_OUTPUT=$(./scripts/version_bump.sh --base HEAD)
echo "$BUMP_OUTPUT"

NEW_VER=$(echo "$BUMP_OUTPUT" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | tail -1)
if [ -z "$NEW_VER" ]; then
    echo "No version change needed."
    exit 0
fi

if $DRY_RUN; then
    echo "Would tag: v$NEW_VER"
    exit 0
fi

# Write VERSION, commit, tag, push
echo "$NEW_VER" > VERSION
git add VERSION
git commit -m "release: v$NEW_VER"

TAG="v$NEW_VER"
git tag -a "$TAG" -m "Release $TAG"

echo "Tagged $TAG. Pushing..."
git push origin HEAD
git push origin "$TAG"

echo ""
echo "Release $TAG pushed — GitHub Actions will build and attach paper artifacts."
