# Welcome to the hugr-llvm development guide <!-- omit in toc -->

This guide is intended to help you get started with developing hugr-llvm.

If you find any errors or omissions in this document, please [open an issue](https://github.com/CQCL/hugr-llvm/issues/new)!

## #️⃣ Setting up the development environment

You can set up the development environment in two ways:

### The Nix way

The easiest way to set up the development environment is to use the provided
[`devenv.nix`](devenv.nix) file. This will set up a development shell with all the
required dependencies.

To use this, you will need to install [devenv](https://devenv.sh/getting-started/).
Once you have it running, open a shell with:

```bash
devenv shell
```

All the required dependencies should be available. You can automate loading the
shell by setting up [direnv](https://devenv.sh/automatic-shell-activation/).

### Manual setup

To set up the environment manually you will need:

- Rust `>=1.75`: <https://www.rust-lang.org/tools/install>
- llvm `== 14.0`: we use the rust bindings
[llvm-sys](https://crates.io/crates/llvm-sys) to [llvm](https://llvm.org/),
- Poetry `>=1.8`: <https://python-poetry.org/>

Once you have these installed, verify that your setup is working

```bash
cargo test
```

## 💅 Coding Style

We use `rustfmt` to enforce a consistent coding style. The CI will fail if the code is not formatted correctly.

To format your code, run:

```bash
cargo format
```

We also use various linters to catch common mistakes and enforce best practices. To run these, see [our CI config](./.github/workflows/ci-rs.yml). TODO Provide a better way, contributions welcome.

## 📈 Code Coverage

We run coverage checks on the CI. Once you submit a PR, you can review the
line-by-line coverage report on
[codecov](https://app.codecov.io/gh/CQCL/hugr-llvm/commits?branch=All%20branches).

To run the coverage checks locally, first install `cargo-llvm-cov`.

```bash
cargo install cargo-llvm-cov
```

Then run the tests, see [our CI config](/.github/workflows/ci-rs.yml).

```bash
just coverage
```

This will generate a coverage file that can be opened with your favourite coverage viewer. In VSCode, you can use
[`coverage-gutters`](https://marketplace.visualstudio.com/items?itemName=ryanluker.vscode-coverage-gutters).

## 🌐 Contributing to hugr-llvm

We welcome contributions to hugr-llvm! Please open [an issue](https://github.com/CQCL/hugr-llvm/issues/new) or [pull request](https://github.com/CQCL/hugr-llvm/compare) if you have any questions or suggestions.

PRs should be made against the `main` branch, and should pass all CI checks before being merged. This includes using the [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/) format for the PR title.

Some tests may be skipped based on the changes made. To run all the tests in
your PR mark it with a 'run-ci-checks' label and push new commits to it.

The general format of a contribution title should be:

```
<type>(<scope>)!: <description>
```

Where the scope is optional, and the `!` is only included if this is a semver breaking change that requires a major version bump.

We accept the following contribution types:

- feat: New features.
- fix: Bug fixes.
- docs: Improvements to the documentation.
- style: Formatting, missing semicolons, etc; no code change.
- refactor: Refactoring code without changing behaviour.
- perf: Code refactoring focused on improving performance.
- test: Adding missing tests, refactoring tests; no production code change.
- ci: CI-related changes. These changes are not published in the changelog.
- chore: Updating build tasks, package manager configs, etc. These changes are not published in the changelog.
- revert: Reverting previous commits.

## :shipit: Releasing new versions

We use automation to bump the version number and generate changelog entries
based on the [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/) labels. Release PRs are created automatically
for each package when new changes are merged into the `main` branch. Once the PR is
approved by someone in the [release team](.github/CODEOWNERS) and is merged, the new package
is published on PyPI or crates.io as appropriate.

The changelog can be manually edited before merging the release PR. Note however
that modifying the diff before other changes are merged will cause the
automation to close the release PR and create a new one to avoid conflicts.

### Rust crate release

Rust releases are managed by `release-plz`. This tool will automatically detect
breaking changes even when they are not marked as such in the commit message,
and bump the version accordingly.

To modify the version being released, update the `Cargo.toml`,
`CHANGELOG.md`, PR name, and PR description in the release PR with the desired version. You may also have to update the dates.
Rust pre-release versions should be formatted as `0.1.0-alpha.1` (or `-beta`, or `-rc`).

### Patch releases

Sometimes we need to release a patch version to fix a critical bug, but we don't want
to include all the changes that have been merged into the main branch. In this case,
you can create a new branch from the latest release tag and cherry-pick the commits
you want to include in the patch release.

You will need to modify the version and changelog manually in this case. Check
the existing release PRs for examples on how to do this. Once the branch is
ready, create a [github release](https://github.com/CQCL/hugr-llvm/releases/new).
The tag should follow the format used in the previous releases, e.g. `hugr-llvm-v0.1.1`.

For rust crates, you will need someone from the release team to manually
publish the new version to crates.io by running `cargo release`.
