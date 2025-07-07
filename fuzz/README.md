# Fuzz testing

This project uses `cargo-fuzz` for doing fuzz testing for hugr.

## Requisites

1. Install `cargo-fuzz` with: `cargo install cargo-fuzz`
2. Build with `cargo fuzz build`

> [!NOTE]
> The `libFuzzer` used by `cargo-fuzz` needs **nightly**.

## Fuzz targets

You can list the fuzzing targets with:
`cargo fuzz list`

### Model: Random

The [fuzz_random](./fuzz_targets/fuzz_random.rs) target uses the coverage-guided
`libFuzzer` fuzzing engine to generate random bytes that we then try to
convert to a package with `hugr_model::v0::ast::Package::from_str()`.

To run this target:
`cargo fuzz run fuzz_random`

It is recommended to provide the `libFuzzer` with a corpus to speed up the
generation of test inputs. For this we can use the fixtures in
`hugr/hugr-model/tests/fixtures`:
`cargo fuzz run fuzz_random ../hugr-model/tests/fixtures`

If you want `libFuzzer` to mutate the examples with ascii characters only:
`cargo fuzz run fuzz_random -- -only_ascii=1`

### Model: Structure

The [fuzz_structure](./fuzz_targets/fuzz_structure.rs) target uses `libFuzzer` to do
[structure-aware](https://rust-fuzz.github.io/book/cargo-fuzz/structure-aware-fuzzing.html)
modifications of the `hugr_model::v0::ast::Package` and its members.

To run this target:
`cargo fuzz run fuzz_structure`

> [!NOTE]
> This target needs some slight modifications to the `hugr-model` source
> code so the structs and enums can derive the `Arbitrary` implementations
> needed by `libFuzzer`.
> The `arbitrary` features for `ordered-float` and `smol_str` are also needed.

## Results

The fuzzing process will be terminated once a crash is detected, and the offending input
will be saved to the `artifacts/<target>` directory. You can reproduce the crash by doing:
`cargo fuzz run fuzz_structure artifacts/<target>/crash-XXXXXX`

If you want to keep the fuzzing process, even after a crash has been detected,
you can provide the options `-fork=1` and `-ignore_crashes=1`.

## Providing options to `libFuzzer`

You can provide lots of options to `libFuzzer` by doing `cargo fuzz run <target> -- -flag1=val1 -flag2=val2`.

To see all the available options:
`cargo fuzz run <target> -- -help=1`
