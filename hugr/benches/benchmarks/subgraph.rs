// Required for black_box uses
#![allow(clippy::unit_arg)]
use hugr::hugr::views::SiblingSubgraph;

use super::hugr::circuit;
use criterion::{AxisScale, BenchmarkId, Criterion, PlotConfiguration, criterion_group};
use std::hint::black_box;

fn bench_singleton_subgraph(c: &mut Criterion) {
    let mut group = c.benchmark_group("singleton_subgraph");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    let num_layers = [10, 100, 1000];

    for layers in num_layers {
        let (hugr, layer_ids) = circuit(layers);

        // Get a subgraph with a single node.
        group.bench_with_input(
            BenchmarkId::from_parameter(layers),
            &layers,
            |b, &layers| {
                // Pick a node from the middle of the circuit.
                let node = layer_ids[layers / 2].cx1;
                b.iter(|| black_box(SiblingSubgraph::try_from_nodes([node], &hugr)));
            },
        );
    }

    group.finish();
}

fn bench_fewnode_subgraph(c: &mut Criterion) {
    let mut group = c.benchmark_group("fewnode_subgraph");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    let num_layers = [10, 100, 1000];

    for layers in num_layers {
        let (hugr, layer_ids) = circuit(layers);

        // Get a subgraph with a fixed number of nodes in the middle of the circuit.
        group.bench_with_input(
            BenchmarkId::from_parameter(layers),
            &layers,
            |b, &_layers| {
                // Pick all the nodes in four layers in the middle of the circuit.
                let nodes: Vec<_> = layer_ids
                    .iter()
                    .skip(layers / 2)
                    .take(4)
                    .flat_map(|ids| [ids.h, ids.cx1, ids.cx2])
                    .collect();
                b.iter(|| black_box(SiblingSubgraph::try_from_nodes(nodes.clone(), &hugr)));
            },
        );
    }

    group.finish();
}

fn bench_multinode_subgraph(c: &mut Criterion) {
    let mut group = c.benchmark_group("multinode_subgraph");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    let num_layers = [10, 100, 1000];

    for layers in num_layers {
        let (hugr, layer_ids) = circuit(layers);

        // Get a subgraph with a single node.
        group.bench_with_input(
            BenchmarkId::from_parameter(layers),
            &layers,
            |b, &_layers| {
                // Pick all the hadamard nodes
                let nodes: Vec<_> = layer_ids.iter().map(|ids| ids.h).collect();
                b.iter(|| black_box(SiblingSubgraph::try_from_nodes(nodes.clone(), &hugr)));
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
        bench_fewnode_subgraph,
        bench_multinode_subgraph,
}
