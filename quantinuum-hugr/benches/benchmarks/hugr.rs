#![allow(clippy::unit_arg)] // Required for black_box uses

use criterion::{criterion_group, AxisScale, Criterion, PlotConfiguration};

fn bench_it_works(c: &mut Criterion) {
    let mut group = c.benchmark_group("it_works");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));
    group.bench_function("it_works", |b| b.iter(|| 42));
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets =
        bench_it_works,
}
