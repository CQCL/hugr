//! Benchmarks
#![allow(dead_code)]

mod benchmarks;

use criterion::criterion_main;

criterion_main! {
    benchmarks::hugr::benches,
    benchmarks::subgraph::benches,
    benchmarks::types::benches,
}
