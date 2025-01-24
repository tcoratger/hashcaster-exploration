use criterion::{
    criterion_group, criterion_main, measurement::Measurement, BatchSize, BenchmarkGroup,
    BenchmarkId, Criterion, Throughput,
};
use hashcaster_keccak::pcs::HashcasterKeccak;
use rayon::current_num_threads;

pub fn bench(
    group: &mut BenchmarkGroup<'_, impl Measurement>,
    name: impl AsRef<str>,
    num_permutations: impl IntoIterator<Item = usize>,
) {
    for num_permutations in num_permutations {
        let snark = HashcasterKeccak::new(num_permutations);
        let id = BenchmarkId::new(
            name.as_ref(),
            format!(
                "num_threads={}/num_permutations={}",
                current_num_threads(),
                snark.num_permutations()
            ),
        );
        group.throughput(Throughput::Elements(snark.num_permutations() as _));
        group.bench_function(id, |b| {
            b.iter_batched(
                || snark.generate_input(),
                |input| snark.prove(input),
                BatchSize::LargeInput,
            );
        });
    }
}

fn bench_keccak(c: &mut Criterion) {
    let mut group = c.benchmark_group("keccak");

    bench(&mut group, "aes_tower_groestl_mt", po2(10..13));
}

pub fn po2(exps: impl IntoIterator<Item = usize>) -> impl Iterator<Item = usize> {
    exps.into_iter().map(|exp| 1 << exp)
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = bench_keccak,
);
criterion_main!(benches);
