# List the available commands
help:
    @just --list --justfile {{justfile()}}

# Prepare the environment for development, installing all the dependencies and
# setting up the pre-commit hooks.
setup:
    uv sync
    [[ -n "${HUGR_JUST_INHIBIT_GIT_HOOKS:-}" ]] || uv run pre-commit install -t pre-commit

# Run the pre-commit checks.
check:
    uv run pre-commit run --all-files

# Run all the tests.
test: test-rust test-python
# Run all rust tests.
test-rust *TEST_ARGS:
    @# We cannot use --workspace --all-features as `hugr-model`s pyo3 feature cannot be
    @# built into a binary build (without using `maturin`)
    @#
    @# This feature list should be kept in sync with the `hugr-py/pyproject.toml`
    cargo test \
        --workspace \
        --exclude 'hugr-py' \
        --features 'hugr/declarative hugr/llvm hugr/llvm-test hugr/zstd' {{TEST_ARGS}}
# Run all python tests.
test-python *TEST_ARGS:
    uv run maturin develop --uv
    cargo build -p hugr-cli
    HUGR_RENDER_DOT=1 uv run pytest {{TEST_ARGS}}

# Run all the benchmarks.
bench language="[rust|python]": (_run_lang language \
        "cargo bench" \
        "true"
    )

# Auto-fix all clippy warnings.
fix language="[rust|python]": (_run_lang language \
        "cargo clippy --all-targets --all-features --workspace --fix --allow-staged --allow-dirty" \
        "uv run ruff check --fix"
    )

# Format the code.
format language="[rust|python]": (_run_lang language \
        "cargo fmt --all" \
        "uv run ruff format"
    )

# Generate a test coverage report.
coverage language="[rust|python]": (_run_lang language \
        "cargo llvm-cov --lcov > lcov.info" \
        "uv run pytest --cov=./ --cov-report=html"
    )

# Run unsoundness checks using miri
miri *TEST_ARGS:
    PROPTEST_DISABLE_FAILURE_PERSISTENCE=true MIRIFLAGS='-Zmiri-env-forward=PROPTEST_DISABLE_FAILURE_PERSISTENCE' cargo +nightly miri test {{TEST_ARGS}}

# Update the HUGR schema.
update-schema:
    uv run scripts/generate_schema.py specification/schema/

# Update the `hugr-model` capnproto definitions.
update-model-capnp:
    # Always use the latest version of capnproto-rust
    cargo install capnpc
    @# When modifying the schema version, update the `ci-rs.yml` file too.
    capnp compile -orust:hugr-model/src --src-prefix=hugr-model hugr-model/capnp/hugr-v0.capnp

# Update snapshots used in the pytest tests.
update-pytest-snapshots:
    uv run pytest --snapshot-update

# Generate serialized declarations for the standard extensions and prelude.
gen-extensions:
    cargo run -p hugr-cli gen-extensions -o specification/std_extensions
    cp -r specification/std_extensions/* hugr-py/src/hugr/std/_json_defs/

# Build the python documentation in hugr-py/docs.
build-py-docs:
    cd hugr-py/docs && ./build.sh

# Run rust semver-checks to detect breaking changes since the last release.
semver-checks:
    cargo semver-checks

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
