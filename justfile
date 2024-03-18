# List the available commands
help:
    @just --list --justfile {{justfile()}}

# Run all the rust tests
test:
    cargo test --all-features

# Auto-fix all clippy warnings
fix:
    cargo clippy --all-targets --all-features --workspace --fix --allow-staged

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

# Run pre-commit checks on the python code
pycheck:
    poetry run ruff format --check
    poetry run ruff check

# Load a poetry shell with the dependencies installed
pyshell:
    poetry install

# Run the python tests
pytest:
    poetry run pytest

# Generate a python test coverage report
pycoverage:
    poetry run pytest --cov=./ --cov-report=html
