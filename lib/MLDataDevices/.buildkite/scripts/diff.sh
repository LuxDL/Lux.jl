#!/bin/bash
set -ueo pipefail

# Script to output the diff where the branch was created
# Usage: ./diff.sh $BUILDKITE_COMMIT

COMMIT_HASH=$1
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

BRANCH_POINT_COMMIT=$($SCRIPT_DIR/find_branch_point.sh "$COMMIT_HASH")
echo >&2 "Cannot find latest build. Running diff against: $BRANCH_POINT_COMMIT"
diff=$(git diff --name-only "$BRANCH_POINT_COMMIT")
echo "$diff"
