#![allow(clippy::unit_arg)] // Required for black_box uses

use criterion::{black_box, criterion_group, AxisScale, Criterion, PlotConfiguration};
use hugr::builder::{BuildError, CFGBuilder, DFGBuilder, Dataflow, DataflowHugr, HugrBuilder};
use hugr::extension::prelude::{BOOL_T, USIZE_T};
use hugr::types::Signature;
use hugr::{type_row, Hugr};

pub fn simple_dfg_hugr() -> Hugr {
    let dfg_builder =
        DFGBuilder::new(Signature::new(type_row![BOOL_T], type_row![BOOL_T])).unwrap();
    let [i1] = dfg_builder.input_wires_arr();
    dfg_builder.finish_prelude_hugr_with_outputs([i1]).unwrap()
}

pub fn simple_cfg_builder<T: AsMut<Hugr> + AsRef<Hugr>>(
    cfg_builder: &mut CFGBuilder<T>,
) -> Result<(), BuildError> {
    let sum2_variants = vec![type_row![USIZE_T], type_row![USIZE_T]];
    let mut entry_b = cfg_builder.entry_builder(sum2_variants.clone(), type_row![])?;
    let entry = {
        let [inw] = entry_b.input_wires_arr();

        let sum = entry_b.make_sum(1, sum2_variants, [inw])?;
        entry_b.finish_with_outputs(sum, [])?
    };
    let mut middle_b = cfg_builder
        .simple_block_builder(Signature::new(type_row![USIZE_T], type_row![USIZE_T]), 1)?;
    let middle = {
        let c = middle_b.add_load_const(hugr::ops::Value::unary_unit_sum());
        let [inw] = middle_b.input_wires_arr();
        middle_b.finish_with_outputs(c, [inw])?
    };
    let exit = cfg_builder.exit_block();
    cfg_builder.branch(&entry, 0, &middle)?;
    cfg_builder.branch(&middle, 0, &exit)?;
    cfg_builder.branch(&entry, 1, &exit)?;
    Ok(())
}

pub fn simple_cfg_hugr() -> Hugr {
    let mut cfg_builder =
        CFGBuilder::new(Signature::new(type_row![USIZE_T], type_row![USIZE_T])).unwrap();
    simple_cfg_builder(&mut cfg_builder).unwrap();
    cfg_builder.finish_prelude_hugr().unwrap()
}

fn bench_builder(c: &mut Criterion) {
    let mut group = c.benchmark_group("builder");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    group.bench_function("simple_dfg", |b| b.iter(|| black_box(simple_dfg_hugr())));
    group.bench_function("simple_cfg", |b| b.iter(|| black_box(simple_cfg_hugr())));
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets =
        bench_builder,
}
