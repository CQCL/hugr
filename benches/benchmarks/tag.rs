#![allow(clippy::unit_arg)] // Required for black_box uses

use criterion::{black_box, criterion_group, AxisScale, BenchmarkId, Criterion, PlotConfiguration};
use hugr::ops::OpTag;

/// Run `OpTag::is_superset`.
fn bench_superset(c: &mut Criterion) {
    let mut group = c.benchmark_group("it_works");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    let tag = OpTag::Const;
    let parent = OpTag::Any;
    group.bench_with_input(
        BenchmarkId::new("tag_superset", format!("{:?}", (tag, parent))),
        &(tag, parent),
        |b, &(t, p)| b.iter(|| t.is_superset(p)),
    );

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets =
        bench_superset,
}
