#!/usr/bin/env bash

set -euo pipefail

DIR="$( cd "$( dirname "$0" )" && pwd )"
{
    cd "$DIR/.."
    poetry install
    for case in tests/guppy_test_cases/*.py; do
        poetry run python "$case" >"$case.json"
    done
}
