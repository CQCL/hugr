# List the available commands
help:
    @just --list --justfile {{justfile()}}

# Prepare the environment for development, installing all the dependencies and
# setting up the pre-commit hooks.
setup:
    poetry install
    [[ -n "${HUGR_JUST_INHIBIT_GIT_HOOKS:-}" ]] || poetry run pre-commit install -t pre-commit

# Run the pre-commit checks.
check:
    HUGR_TEST_SCHEMA=1 poetry run pre-commit run --all-files

# Run all the tests.
test language="[rust|python]" : (_run_lang language \
        "HUGR_TEST_SCHEMA=1 cargo test --all-features" \
        "poetry run pytest"
    )

# Run all the benchmarks.
bench language="[rust|python]": (_run_lang language \
        "cargo bench" \
        "true"
    )

# Auto-fix all clippy warnings.
fix language="[rust|python]": (_run_lang language \
        "cargo clippy --all-targets --all-features --workspace --fix --allow-staged --allow-dirty" \
        "poetry run ruff check --fix"
    )

# Format the code.
format language="[rust|python]": (_run_lang language \
        "cargo fmt" \
        "poetry run ruff format"
    )

# Generate a test coverage report.
coverage language="[rust|python]": (_run_lang language \
        "cargo llvm-cov --lcov > lcov.info" \
        "poetry run pytest --cov=./ --cov-report=html"
    )

# Load a shell with all the dependencies installed
shell:
    poetry shell

# Update the HUGR schema.
update-schema:
    poetry update
    poetry run python scripts/generate_schema.py specification/schema/


# Runs a rust and a python command, depending on the `language` variable.
#
# If `language` is set to `rust` or `python`, only run the command for that language.
# Otherwise, run both commands.
_run_lang language rust_cmd python_cmd:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ "{{ language }}" = "rust" ]; then
        set -x
        {{ rust_cmd }}
    elif [ "{{ language }}" = "python" ]; then
        set -x
        {{ python_cmd }}
    else
        set -x
        {{ rust_cmd }}
        {{ python_cmd }}
    fi
