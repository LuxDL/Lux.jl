#!/bin/bash
set -ueo pipefail

BASE_REF="origin/main"
if ! git rev-parse --verify --quiet "$BASE_REF" >/dev/null; then
    BASE_REF="main"
fi

MERGE_BASE=$(git merge-base "$1" "$BASE_REF")
if [ "$MERGE_BASE" = "$(git rev-parse "$1")" ] && git rev-parse --verify --quiet "${1}~1" >/dev/null; then
    echo "${1}~1"
else
    echo "$MERGE_BASE"
fi
