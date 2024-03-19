# Welcome to the HUGR development guide <!-- omit in toc -->

This guide is intended to help you get started with developing HUGR.

If you find any errors or omissions in this document, please [open an issue](https://github.com/CQCL/hugr/issues/new)!

## #️⃣ Setting up the development environment

You can setup the development environment in two ways:

### The Nix way

The easiest way to setup the development environment is to use the provided
[`devenv.nix`](devenv.nix) file. This will setup a development shell with all the
required dependencies.

To use this, you will need to install [devenv](https://devenv.sh/getting-started/).
Once you have it running, open a shell with:

```bash
devenv shell
```

All the required dependencies should be available. You can automate loading the
shell by setting up [direnv](https://devenv.sh/automatic-shell-activation/).

### Manual setup

To setup the environment manually you will need:

- Rust: https://www.rust-lang.org/tools/install
- Just: https://just.systems/
- Poetry: https://python-poetry.org/

Once you have these installed, you can install the required python dependencies and setup pre-commit hooks with:

```bash
just setup
```

## 🏃 Running the tests

To compile and test the code, run:

```bash
just test
# or, to test only the rust code or the python code
just test rust
just test python
```

Run the rust benchmarks with:

```bash
cargo bench
```

Finally, if you have rust nightly installed, you can run `miri` to detect
undefined behaviour in the code. Note that the _devenv_ shell only has rust
stable available.

```bash
cargo +nightly miri test
```

Run `just` to see all available commands.

## 💅 Coding Style

The rustfmt tool is used to enforce a consistent rust coding style. The CI will fail if the code is not formatted correctly.

To format your code, run:

```bash
just format
```

We also use various linters to catch common mistakes and enforce best practices. To run these, use:

```bash
just check
```

To quickly fix common issues, run:

```bash
just fix
# or, to fix only the rust code or the python code
just fix rust
just fix python
```

## 📈 Code Coverage

We run coverage checks on the CI. Once you submit a PR, you can review the
line-by-line coverage report on
[codecov](https://app.codecov.io/gh/CQCL/hugr/commits?branch=All%20branches).

To run the coverage checks locally, first install `cargo-llvm-cov`.

```bash
cargo install cargo-llvm-cov
```

Then run the tests:

```bash
just coverage
```

and open it with your favourite coverage viewer. In VSCode, you can use
[`coverage-gutters`](https://marketplace.visualstudio.com/items?itemName=ryanluker.vscode-coverage-gutters).

## 🌐 Contributing to HUGR

We welcome contributions to HUGR! Please open [an issue](https://github.com/CQCL/hugr/issues/new) or [pull request](https://github.com/CQCL/hugr/compare) if you have any questions or suggestions.

PRs should be made against the `main` branch, and should pass all CI checks before being merged. This includes using the [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/) format for the PR title.

The general format of a contribution title should be:

```
<type>(<scope>)!: <description>
```

Where the scope is optional, and the `!` is only included if this is a semver breaking change that requires a major version bump.

We accept the following contribution types:

- feat: New features.
- fix: Bug fixes.
- docs: Improvements to the documentation.
- style: Formatting, missing semi colons, etc; no code change.
- refactor: Refactoring code without changing behaviour.
- perf: Code refactoring focused on improving performance.
- test: Adding missing tests, refactoring tests; no production code change.
- ci: CI related changes. These changes are not published in the changelog.
- chore: Updating build tasks, package manager configs, etc. These changes are not published in the changelog.
- revert: Reverting previous commits.
