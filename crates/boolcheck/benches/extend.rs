use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hashcaster_boolcheck::builder::BoolCheckBuilder;
use hashcaster_field::binary_field::BinaryField128b;
use hashcaster_poly::multinear_lagrangian::MultilinearLagrangianPolynomial;
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
        (i as u128 * 10..i as u128 * 10 + table_size as u128)
            .map(BinaryField128b::from)
            .collect::<Vec<_>>()
            .into()
    });

    // Create a BoolCheckBuilder instance with test data
    let bool_check = BoolCheckBuilder::<N, M> {
        c: 2, // Recursion depth
        polys: tables,
        ..Default::default()
    };

    // Generate the ternary mapping
    let (_, trit_mapping) = bool_check.trit_mapping();

    // Define the linear and quadratic functions
    let f_lin = |args: &[BinaryField128b; N]| args.iter().copied().sum();
    let f_quad = |args: &[BinaryField128b; N]| args.iter().map(|&x| x * x).sum();

    // Benchmark the `extend_n_tables` function
    c.bench_function("extend_n_tables", |b| {
        b.iter(|| {
            bool_check.extend_n_tables(
                black_box(&trit_mapping),
                black_box(f_lin),
                black_box(f_quad),
            )
        });
    });
}

criterion_group!(benches, benchmark_extend_n_tables);
criterion_main!(benches);
