#![allow(clippy::unit_arg)] // Required for black_box uses

pub mod examples;

use criterion::{AxisScale, BatchSize, BenchmarkId, Criterion, PlotConfiguration, criterion_group};
use hugr::envelope::{EnvelopeConfig, EnvelopeFormat};
use hugr::hugr::hugrmut::HugrMut;
use hugr::ops::handle::NodeHandle;
#[allow(unused)]
use hugr::std_extensions::STD_REG;
use hugr::{Hugr, HugrView};
use std::hint::black_box;

pub use examples::{
    BENCH_EXTENSIONS, circuit, dfg_calling_defn_decl, simple_cfg_hugr, simple_dfg_hugr,
};

trait Serializer {
    fn serialize(&self, hugr: &Hugr) -> Vec<u8>;
    fn deserialize(&self, bytes: &[u8]) -> Hugr;
}

struct JsonSer;
impl Serializer for JsonSer {
    fn serialize(&self, hugr: &Hugr) -> Vec<u8> {
        let cfg = EnvelopeConfig::new(EnvelopeFormat::PackageJson).disable_compression();

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
        let cfg =
            EnvelopeConfig::new(EnvelopeFormat::ModelWithExtensions).with_zstd(Default::default());

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

fn bench_insertion(c: &mut Criterion) {
    let mut group = c.benchmark_group("insertion");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    group.bench_function("insert_from_view", |b| {
        let mut h1 = simple_dfg_hugr();
        let h2 = simple_cfg_hugr();
        b.iter(|| black_box(h1.insert_from_view(h1.entrypoint(), &h2)))
    });
    group.bench_function("insert_hugr", |b| {
        b.iter_batched(
            || (simple_dfg_hugr(), simple_cfg_hugr()),
            |(mut h, insert)| black_box(h.insert_hugr(h.entrypoint(), insert)),
            BatchSize::SmallInput,
        )
    });
    group.bench_function("insert_view_forest", |b| {
        let (insert, decl, defn) = dfg_calling_defn_decl();
        b.iter_batched(
            || {
                let h = simple_dfg_hugr();
                let nodes = insert.entry_descendants().chain([defn.node(), decl.node()]);
                let roots = [
                    (insert.entrypoint(), h.entrypoint()),
                    (defn.node(), h.module_root()),
                    (decl.node(), h.module_root()),
                ];
                (h, &insert, nodes, roots)
            },
            |(mut h, insert, nodes, roots)| black_box(h.insert_view_forest(insert, nodes, roots)),
            BatchSize::SmallInput,
        )
    });
    group.bench_function("insert_forest", |b| {
        b.iter_batched(
            || {
                let h = simple_dfg_hugr();
                let (insert, decl, defn) = dfg_calling_defn_decl();
                let roots = [
                    (insert.entrypoint(), h.entrypoint()),
                    (defn.node(), h.module_root()),
                    (decl.node(), h.module_root()),
                ];
                (h, insert, roots)
            },
            |(mut h, insert, roots)| black_box(h.insert_forest(insert, roots)),
            BatchSize::SmallInput,
        )
    });
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
        bench_builder, bench_insertion, bench_serialization
}
