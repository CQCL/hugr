// Required for black_box uses
#![allow(clippy::unit_arg)]
use hugr::extension::prelude::{QB_T, USIZE_T};
use hugr::ops::AliasDecl;
use hugr::types::{Signature, Type, TypeBound};

use criterion::{black_box, criterion_group, AxisScale, Criterion, PlotConfiguration};

/// Construct a complex type.
fn make_complex_type() -> Type {
    let qb = QB_T;
    let int = USIZE_T;
    let q_register = Type::new_tuple(vec![qb; 8]);
    let b_register = Type::new_tuple(vec![int; 8]);
    let q_alias = Type::new_alias(AliasDecl::new("QReg", TypeBound::Any));
    let sum = Type::new_sum([q_register, q_alias]);
    Type::new_function(Signature::new(vec![sum], vec![b_register]))
}

fn bench_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("types");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    group.bench_function("construction", |b| {
        b.iter(|| black_box(make_complex_type()))
    });

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets =
        bench_construction,
}
