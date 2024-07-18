//! This benchmarks running the main JavaScript transform in parcel

use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use std::path::Path;

use parcel_plugin_transformer_js::transformer::test_helpers::run_swc_core_transform;

pub fn transformer_benchmark(c: &mut Criterion) {
  let setup = || {
    let stub_file_path = Path::new(env!("CARGO_MANIFEST_DIR"));
    let stub_file_path =
      stub_file_path.join("benches/parcel_plugin_transformer_js_benchmark/stub.js");
    std::fs::read_to_string(stub_file_path).unwrap()
  };

  c.bench_function("transform JavaScript file", |b| {
    b.iter_batched(
      setup,
      |code| run_swc_core_transform(&code),
      BatchSize::SmallInput,
    );
  });
}

criterion_group!(benches, transformer_benchmark);
criterion_main!(benches);
