use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hashcaster_primitives::{binary_field::BinaryField128b, matrix_efficient::EfficientMatrix};
use rand::Rng;

/// Generates a random `BinaryField128b` element.
fn random_binary_field() -> BinaryField128b {
    let mut rng = rand::thread_rng();
    BinaryField128b::from(rng.gen::<u128>())
}

/// Benchmark for the `apply` function in both EfficientMatrix and BoxedEfficientMatrix.
fn benchmark_apply(c: &mut Criterion) {
    // Generate a random EfficientMatrix
    let cols: [_; 128] = std::array::from_fn(|_| random_binary_field());
    let matrix = EfficientMatrix::from_cols(&cols);

    // Generate a random BinaryField128b input
    let input = random_binary_field();

    // Benchmark apply function for EfficientMatrix
    c.bench_function("EfficientMatrix::apply", |b| {
        b.iter(|| {
            // Black-box prevents compiler optimizations
            black_box(matrix.apply(black_box(input)));
        });
    });
}

// Register benchmark group
criterion_group!(benches, benchmark_apply);
criterion_main!(benches);
