# List the available commands
help:
    @just --list --justfile {{justfile()}}

# Run all the rust tests
test:
    cargo test --all-features

# Auto-fix all clippy warnings
fix:
    cargo clippy --all-targets --all-features --workspace --fix --allow-staged
    poetry run ruff check --fix

# Run the pre-commit checks
check:
    ./.github/pre-commit

# Format the code
format:
    cargo fmt
    poetry run ruff format

# Generate a test coverage report
coverage:
    cargo llvm-cov --lcov > lcov.info

# Load a poetry shell with the dependencies installed
pyshell:
    poetry shell

# Run the python tests
pytest:
    poetry run pytest

# Generate a python test coverage report
pycoverage:
    poetry run pytest --cov=./ --cov-report=html

# Update the HUGR schema
update-schema:
    poetry run python scripts/generate_schema.py specification/schema/
