#![allow(clippy::unit_arg)] // Required for black_box uses

use criterion::{black_box, criterion_group, AxisScale, BenchmarkId, Criterion, PlotConfiguration};
use hugr::builder::{
    BuildError, CFGBuilder, Container, DFGBuilder, Dataflow, DataflowHugr, DataflowSubContainer,
    HugrBuilder, ModuleBuilder,
};
use hugr::extension::prelude::{BOOL_T, QB_T, USIZE_T};
use hugr::extension::PRELUDE_REGISTRY;
use hugr::ops::OpName;
use hugr::std_extensions::arithmetic::float_ops::FLOAT_OPS_REGISTRY;
use hugr::std_extensions::arithmetic::float_types::{ConstF64, FLOAT64_TYPE};
use hugr::types::Signature;
use hugr::{type_row, CircuitUnit, Extension, Hugr};
use lazy_static::lazy_static;
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

trait Serializer {
    fn serialize(&self, hugr: &Hugr) -> Vec<u8>;
    fn deserialize(&self, bytes: &[u8]) -> Hugr;
}

struct JsonSer;
impl Serializer for JsonSer {
    fn serialize(&self, hugr: &Hugr) -> Vec<u8> {
        serde_json::to_vec(hugr).unwrap()
    }
    fn deserialize(&self, bytes: &[u8]) -> Hugr {
        serde_json::from_slice(bytes).unwrap()
    }
}

fn roundtrip(hugr: &Hugr, serializer: impl Serializer) -> Hugr {
    let bytes = serializer.serialize(hugr);
    serializer.deserialize(&bytes)
}

lazy_static! {
    static ref QUANTUM_EXT: Extension = {
        let mut extension = Extension::new(
            "bench.quantum".try_into().unwrap(),
            hugr::extension::Version::new(0, 0, 0),
        );

        extension
            .add_op(
                OpName::new_inline("H"),
                "".into(),
                Signature::new_endo(QB_T),
            )
            .unwrap();
        extension
            .add_op(
                OpName::new_inline("Rz"),
                "".into(),
                Signature::new(type_row![QB_T, FLOAT64_TYPE], type_row![QB_T]),
            )
            .unwrap();

        extension
            .add_op(
                OpName::new_inline("CX"),
                "".into(),
                Signature::new_endo(type_row![QB_T, QB_T]),
            )
            .unwrap();
        extension
    };
}

pub fn circuit(layers: usize) -> Hugr {
    let h_gate = QUANTUM_EXT
        .instantiate_extension_op("H", [], &PRELUDE_REGISTRY)
        .unwrap();
    let cx_gate = QUANTUM_EXT
        .instantiate_extension_op("CX", [], &PRELUDE_REGISTRY)
        .unwrap();
    let rz = QUANTUM_EXT
        .instantiate_extension_op("Rz", [], &FLOAT_OPS_REGISTRY)
        .unwrap();
    let signature =
        Signature::new_endo(type_row![QB_T, QB_T]).with_extension_delta(QUANTUM_EXT.name().clone());
    let mut module_builder = ModuleBuilder::new();
    let mut f_build = module_builder.define_function("main", signature).unwrap();

    let wires: Vec<_> = f_build.input_wires().collect();

    let mut linear = f_build.as_circuit(wires);

    for _ in 0..layers {
        linear
            .append(h_gate.clone(), [0])
            .unwrap()
            .append(cx_gate.clone(), [0, 1])
            .unwrap()
            .append(cx_gate.clone(), [1, 0])
            .unwrap();

        let angle = linear.add_constant(ConstF64::new(0.5));
        linear
            .append_and_consume(
                rz.clone(),
                [CircuitUnit::Linear(0), CircuitUnit::Wire(angle)],
            )
            .unwrap();
    }

    let outs = linear.finish();
    f_build.finish_with_outputs(outs).unwrap();

    module_builder.finish_hugr(&FLOAT_OPS_REGISTRY).unwrap()
}

fn bench_builder(c: &mut Criterion) {
    let mut group = c.benchmark_group("builder");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    group.bench_function("simple_dfg", |b| b.iter(|| black_box(simple_dfg_hugr())));
    group.bench_function("simple_cfg", |b| b.iter(|| black_box(simple_cfg_hugr())));
    group.finish();
}

fn bench_serialization(c: &mut Criterion) {
    c.bench_function("simple_cfg_serialize", |b| {
        let h = simple_cfg_hugr();
        b.iter(|| {
            black_box(roundtrip(&h, JsonSer));
        });
    });
    let mut group = c.benchmark_group("circuit_roundtrip");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    for size in [0, 1, 10, 100, 1000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let h = circuit(size);
            b.iter(|| {
                black_box(roundtrip(&h, JsonSer));
            });
        });
    }
    group.finish();

    let mut group = c.benchmark_group("circuit_serialize");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    for size in [0, 1, 10, 100, 1000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let h = circuit(size);
            b.iter(|| {
                black_box(JsonSer.serialize(&h));
            });
        });
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets =
        bench_builder, bench_serialization
}
