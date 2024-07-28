#!/bin/bash
set -ue

diff -u <(git rev-list --first-parent "$1") \
        <(git rev-list --first-parent master) | \
        sed -ne 's/^ //p' | head -1
