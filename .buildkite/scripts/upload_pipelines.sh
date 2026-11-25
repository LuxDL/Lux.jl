#!/bin/bash
set -euo pipefail

# Detect changed paths for a commit and upload matching Buildkite pipelines
# directly from this step, avoiding extra upload-queue jobs and full clones.
#
# Usage: ./upload_pipelines.sh [--downstream] [--dry-run] [commit]

DOWNSTREAM=false
DRY_RUN=false
COMMIT_HASH=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --downstream)
            DOWNSTREAM=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -*)
            echo >&2 "Unknown option: $1"
            exit 1
            ;;
        *)
            COMMIT_HASH="$1"
            shift
            ;;
    esac
done

COMMIT_HASH="${COMMIT_HASH:-${BUILDKITE_COMMIT:-HEAD}}"
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

CHANGED_FILES=()
while IFS= read -r line; do
    [[ -n "$line" ]] && CHANGED_FILES+=("$line")
done < <("$SCRIPT_DIR/diff.sh" "$COMMIT_HASH")

if [[ ${#CHANGED_FILES[@]} -eq 0 ]]; then
    echo "No changed files detected for $COMMIT_HASH; skipping pipeline uploads."
    exit 0
fi

has_changes() {
    local pattern file
    for pattern in "$@"; do
        for file in "${CHANGED_FILES[@]}"; do
            if [[ "$pattern" == */ ]]; then
                [[ "$file" == "$pattern"* ]] && return 0
            else
                [[ "$file" == "$pattern" ]] && return 0
            fi
        done
    done
    return 1
}

UPLOADED=0
upload_if_changed() {
    local label="$1"
    local pipeline="$2"
    shift 2

    if has_changes "$@"; then
        echo "Uploading $label pipeline ($pipeline)"
        if [[ "$DRY_RUN" == "false" ]]; then
            buildkite-agent pipeline upload "$pipeline"
        fi
        UPLOADED=$((UPLOADED + 1))
    fi
}

if [[ "$DOWNSTREAM" == "true" ]]; then
    upload_if_changed "Downstream" ".buildkite/downstream.yml" \
        "src/" \
        "ext/" \
        "lib/" \
        "Project.toml" \
        ".buildkite/downstream.yml"
else
    upload_if_changed "Lux" ".buildkite/testing.yml" \
        "src/" \
        "ext/" \
        "test/" \
        "Project.toml" \
        ".buildkite/testing.yml" \
        "lib/LuxTestUtils/src/" \
        "lib/LuxTestUtils/ext/" \
        "lib/LuxCore/src/" \
        "lib/LuxCore/ext/" \
        "lib/MLDataDevices/src/" \
        "lib/MLDataDevices/ext/" \
        "lib/WeightInitializers/src/" \
        "lib/WeightInitializers/ext/" \
        "lib/LuxLib/src/" \
        "lib/LuxLib/ext/"

    upload_if_changed "LuxCUDA" ".buildkite/testing_luxcuda.yml" \
        "lib/LuxCUDA/" \
        ".buildkite/testing_luxcuda.yml"

    upload_if_changed "WeightInitializers" ".buildkite/testing_weightinitializers.yml" \
        "lib/WeightInitializers/" \
        ".buildkite/testing_weightinitializers.yml"

    upload_if_changed "MLDataDevices" ".buildkite/testing_mldatadevices.yml" \
        "lib/MLDataDevices/" \
        ".buildkite/testing_mldatadevices.yml"

    upload_if_changed "LuxLib" ".buildkite/testing_luxlib.yml" \
        "lib/LuxLib/" \
        ".buildkite/testing_luxlib.yml" \
        "lib/LuxTestUtils/src/" \
        "lib/LuxTestUtils/ext/" \
        "lib/LuxCore/src/" \
        "lib/LuxCore/ext/" \
        "lib/MLDataDevices/src/" \
        "lib/MLDataDevices/ext/"

    upload_if_changed "LuxTestUtils" ".buildkite/testing_luxtestutils.yml" \
        "lib/LuxTestUtils/" \
        ".buildkite/testing_luxtestutils.yml"
fi

if [[ "$UPLOADED" -eq 0 ]]; then
    echo "No watched paths changed; no pipelines uploaded."
fi
