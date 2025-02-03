use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hashcaster_boolcheck::builder::{BoolCheckBuilder, DummyPackage};
use hashcaster_primitives::{
    binary_field::BinaryField128b, poly::multinear_lagrangian::MultilinearLagrangianPolynomial,
};
use std::array;

fn benchmark_extend_n_tables(c: &mut Criterion) {
    // Parameters for the test
    const N: usize = 3; // Number of tables
    const M: usize = 1; // Unused in this test

    // Define dimensions and table sizes
    let dims = 20;
    let table_size = 1 << dims;

    // Generate test data for the input tables
    let tables: [MultilinearLagrangianPolynomial; N] = array::from_fn(|i| {
        #[allow(clippy::cast_sign_loss)]
        (i as u128 * 10..i as u128 * 10 + table_size as u128).map(BinaryField128b::from).collect()
    });

    // Create a BoolCheckBuilder instance with test data
    let points = Default::default();
    let bool_check = BoolCheckBuilder::<N, M, 2, DummyPackage<N, M>> {
        polys: tables,
        points: &points,
        gammas: Default::default(),
        claims: Default::default(),
        algebraic_operations: Default::default(),
    };

    // Generate the ternary mapping
    let (_, trit_mapping) = bool_check.trit_mapping();

    // Benchmark the `extend_n_tables` function
    c.bench_function("extend_n_tables", |b| {
        b.iter(|| bool_check.extend_n_tables(black_box(&trit_mapping)));
    });
}

criterion_group!(benches, benchmark_extend_n_tables);
criterion_main!(benches);
