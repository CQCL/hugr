// Required for black_box uses
#![allow(clippy::unit_arg)]
use hugr::hugr::views::SiblingSubgraph;

use criterion::{black_box, criterion_group, AxisScale, BenchmarkId, Criterion, PlotConfiguration};

use super::hugr::circuit;

fn bench_singleton_subgraph(c: &mut Criterion) {
    let mut group = c.benchmark_group("singleton_subgraph");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    let num_layers = [10, 100, 1000];

    for layers in num_layers.into_iter() {
        let (hugr, layer_ids) = circuit(layers);

        // Get a subgraph with a single node.
        group.bench_with_input(
            BenchmarkId::from_parameter(layers),
            &layers,
            |b, &layers| {
                // Pick a node from the middle of the circuit.
                let node = layer_ids.iter().nth(layers / 2).unwrap().cx1;
                b.iter(|| black_box(SiblingSubgraph::try_from_nodes([node], &hugr)))
            },
        );
    }

    group.finish();
}

fn bench_multinode_subgraph(c: &mut Criterion) {
    let mut group = c.benchmark_group("multinode_subgraph");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    let num_layers = [10, 100, 1000];

    for layers in num_layers.into_iter() {
        let (hugr, layer_ids) = circuit(layers);

        // Get a subgraph with a single node.
        group.bench_with_input(
            BenchmarkId::from_parameter(layers),
            &layers,
            |b, &_layers| {
                // Pick all the hadamard nodes
                let nodes: Vec<_> = layer_ids.iter().map(|ids| ids.h).collect();
                b.iter(|| black_box(SiblingSubgraph::try_from_nodes(nodes.clone(), &hugr)))
            },
        );
    }

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets =
        bench_singleton_subgraph,
        bench_multinode_subgraph,
}
