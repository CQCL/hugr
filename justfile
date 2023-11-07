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

# Generate a test coverage report
coverage:
    cargo llvm-cov --lcov > lcov.info