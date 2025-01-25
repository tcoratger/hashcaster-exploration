use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use hashcaster_keccak::pcs::HashcasterKeccak;
use std::time::Duration;

fn configure_criterion() -> Criterion {
    Criterion::default()
        .measurement_time(Duration::from_secs(20))
        .sample_size(50)
        .warm_up_time(Duration::from_secs(5))
}

fn benchmark_snark(c: &mut Criterion) {
    let log_permutations = 10; // Example : 2^10 permutations
    let num_permutations = 1 << log_permutations;

    let snark = HashcasterKeccak::new(num_permutations);

    let mut group = c.benchmark_group("HashcasterKeccak");
    group.throughput(Throughput::Elements(num_permutations as u64));

    group.bench_with_input(BenchmarkId::new("Prove", log_permutations), &snark, |b, snark| {
        b.iter_custom(|iters| {
            let mut total_duration = Duration::ZERO;

            for _ in 0..iters {
                let input = snark.generate_input();
                let start = std::time::Instant::now();
                let proof = snark.prove(input);
                total_duration += start.elapsed();

                std::mem::drop(proof);
            }

            total_duration
        });
    });

    group.finish();
}

criterion_group! {
    name = benches;
    config = configure_criterion();
    targets = benchmark_snark
}
criterion_main!(benches);
