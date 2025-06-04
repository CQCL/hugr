#![allow(clippy::unit_arg)] // Required for black_box uses

pub mod examples;

use criterion::{AxisScale, BenchmarkId, Criterion, PlotConfiguration, criterion_group};
use hugr::Hugr;
use hugr::envelope::{EnvelopeConfig, EnvelopeFormat};
#[allow(unused)]
use hugr::std_extensions::STD_REG;
use std::hint::black_box;

pub use examples::{BENCH_EXTENSIONS, circuit, simple_cfg_hugr, simple_dfg_hugr};

trait Serializer {
    fn serialize(&self, hugr: &Hugr) -> Vec<u8>;
    fn deserialize(&self, bytes: &[u8]) -> Hugr;
}

struct JsonSer;
impl Serializer for JsonSer {
    fn serialize(&self, hugr: &Hugr) -> Vec<u8> {
        let mut cfg = EnvelopeConfig::default();
        cfg.format = EnvelopeFormat::PackageJson;
        cfg.zstd = None;

        let mut bytes = Vec::new();
        hugr.store(&mut bytes, cfg).unwrap();
        bytes
    }
    fn deserialize(&self, bytes: &[u8]) -> Hugr {
        Hugr::load(bytes, Some(&BENCH_EXTENSIONS)).unwrap()
    }
}

struct CapnpSer;

impl Serializer for CapnpSer {
    fn serialize(&self, hugr: &Hugr) -> Vec<u8> {
        let mut cfg = EnvelopeConfig::default();
        cfg.format = EnvelopeFormat::ModelWithExtensions;
        cfg.zstd = Some(Default::default());

        let mut bytes = Vec::new();
        hugr.store(&mut bytes, cfg).unwrap();
        bytes
    }

    fn deserialize(&self, bytes: &[u8]) -> Hugr {
        Hugr::load(bytes, Some(&BENCH_EXTENSIONS)).unwrap()
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
    for size in &[0, 1, 10, 100, 1000] {
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
    for size in &[0, 1, 10, 100, 1000] {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let h = circuit(size).0;
            b.iter(|| {
                black_box(JsonSer.serialize(&h));
            });
        });
    }
    group.finish();

    let mut group = c.benchmark_group("circuit_roundtrip/capnp");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    for size in &[0, 1, 10, 100, 1000] {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let h = circuit(size).0;
            b.iter(|| {
                black_box(roundtrip(&h, CapnpSer));
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
