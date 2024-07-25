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

- Just: https://just.systems/
- Rust `>=1.75`: https://www.rust-lang.org/tools/install
- Poetry `>=1.8`: https://python-poetry.org/

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

## Serialization

If you want to make a change that modifies the serialization schema, you must
ensure backwards-compatibility by writing a method to convert from the existing
format to the new one. We suggest the following process. (For concreteness we
assume that you are upgrading from v5 to v6.)

1.  Add a test case in `hugr-core/src/hugr/serialize/upgrade/test.rs` that
    exercises the part of the schema that will change in v6.
2.  Run the tests. This will create a new JSON file in the `testcases`
    subdirectory. Commit this to the repo.
3.  Implement the schema-breaking change. Expect the test you added in step 1
    (and possibly others) to fail.
4.  In `hugr/hugr-core/src/hugr/serialize.rs`:
    - Add a new line `V6(SerHugr),` in `enum Versioned`, and change the previous
      line to `V5(serde_json::Value),`.
    - In `Versioned::upgrade()` insert the line
      `Self::V5(json) => self = Self::V6(upgrade::v5_to_v6(json).and_then(go)?),`
      and change `V5` to `V6` in the line
      `Self::V5(ser_hugr) => return Ok(ser_hugr),`.
    - Change `new_latest()` to return `Self::V6(t)`.
5.  In `hugr-core/src/hugr/serialize/upgrade.rs` add a stub implementation of
    `v5_to_v6()`.
6.  In `hugr-py/src/hugr/__init__.py` update `get_serialisation_version()` to
    return `"v6"`.
7.  Run `just update-schema` to generate new v6 schema files. Commit these to
    the repo.
8.  In `hugr-core/src/hugr/serialize/test.rs`, in the `include_schema` macro
    change `v5` to `v6`.
9.  Implement `v5_to_v6()`.
10. Ensure all tests are passing.

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
CHANGELOG.md, PR name, and PR description in the release PR with the desired version. You may also have to update the dates.
Rust pre-release versions should be formatted as `0.1.0-alpha.1` (or `-beta`, or `-rc`).

### Python package release

Python releases are managed by `release-please`. This tool always bumps the
minor version (or the pre-release version if the previous version was a
pre-release).

To override the version getting released, you must merge a PR to `main` containing
`Release-As: 0.1.0` in the description.
Python pre-release versions should be formatted as `0.1.0a1` (or `b1`, `rc1`).

### Patch releases

Sometimes we need to release a patch version to fix a critical bug, but we don't want
to include all the changes that have been merged into the main branch. In this case,
you can create a new branch from the latest release tag and cherry-pick the commits
you want to include in the patch release.

#### Rust patch releases

You can use [`release-plz`](https://release-plz.ieni.dev/) to automatically generate the changelogs and bump the package versions.

```bash
release-plz update
```

Once the branch is ready, create a draft PR so that the release team can review
it.

Now someone from the release team can run `release-plz` on the **unmerged**
branch to create the github releases and publish to crates.io.

```bash
# Make sure you are logged in to `crates.io`
cargo login <your_crates_io_token>
# Get a github token with permissions to create releases
GITHUB_TOKEN=<your_github_token>
# Run release-plz
release-plz release --git-token $GITHUB_TOKEN
```

#### Python patch releases

You will need to modify the version and changelog manually in this case. Check
the existing release PRs for examples on how to do this. Once the branch is
ready, create a draft PR so that the release team can review it.

The wheel building process and publication to PyPI is handled by the CI.
Just create a [github release](https://github.com/CQCL/hugr/releases/new) from the **unmerged** branch.
The release tag should follow the format used in the previous releases, e.g. `hugr-py-v0.1.1`.
