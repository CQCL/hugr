#![allow(clippy::unit_arg)] // Required for black_box uses

pub mod examples;

use criterion::{black_box, criterion_group, AxisScale, BenchmarkId, Criterion, PlotConfiguration};
#[allow(unused)]
use hugr::std_extensions::STD_REG;
use hugr::Hugr;

pub use examples::{circuit, simple_cfg_hugr, simple_dfg_hugr};

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

#[cfg(feature = "model_unstable")]
struct CapnpSer;

#[cfg(feature = "model_unstable")]
impl Serializer for CapnpSer {
    fn serialize(&self, hugr: &Hugr) -> Vec<u8> {
        let bump = bumpalo::Bump::new();
        let module = hugr_core::export::export_hugr(hugr, &bump);
        let package = hugr_model::v0::table::Package {
            modules: vec![module],
        };
        hugr_model::v0::binary::write_to_vec(&package)
    }

    fn deserialize(&self, bytes: &[u8]) -> Hugr {
        let bump = bumpalo::Bump::new();
        let package = hugr_model::v0::binary::read_from_slice(bytes, &bump).unwrap();
        hugr_core::import::import_hugr(&package.modules[0], &STD_REG).unwrap()
    }
}

fn roundtrip(hugr: &Hugr, serializer: impl Serializer) -> Hugr {
    let bytes = serializer.serialize(hugr);
    serializer.deserialize(&bytes)
}

fn bench_builder(c: &mut Criterion) {
    let mut group = c.benchmark_group("builder");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    group.bench_function("simple_dfg", |b| b.iter(|| black_box(simple_dfg_hugr())));
    group.bench_function("simple_cfg", |b| b.iter(|| black_box(simple_cfg_hugr())));
    group.finish();
}

fn bench_serialization(c: &mut Criterion) {
    c.bench_function("simple_cfg_serialize/json", |b| {
        let h = simple_cfg_hugr();
        b.iter(|| {
            black_box(roundtrip(&h, JsonSer));
        });
    });

    let mut group = c.benchmark_group("circuit_roundtrip/json");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    for size in [0, 1, 10, 100, 1000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let h = circuit(size).0;
            b.iter(|| {
                black_box(roundtrip(&h, JsonSer));
            });
        });
    }
    group.finish();

    let mut group = c.benchmark_group("circuit_serialize/json");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    for size in [0, 1, 10, 100, 1000].iter() {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let h = circuit(size).0;
            b.iter(|| {
                black_box(JsonSer.serialize(&h));
            });
        });
    }
    group.finish();

    #[cfg(feature = "model_unstable")]
    {
        let mut group = c.benchmark_group("circuit_roundtrip/capnp");
        group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
        for size in [0, 1, 10, 100, 1000].iter() {
            group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
                let h = circuit(size).0;
                b.iter(|| {
                    black_box(roundtrip(&h, CapnpSer));
                });
            });
        }
        group.finish();
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets =
        bench_builder, bench_serialization
}
