use hashcaster_keccak::pcs::HashcasterKeccak;
use std::{
    fmt::Debug,
    hint::black_box,
    time::{Duration, Instant},
};

#[derive(Clone, Debug, clap::ValueEnum)]
enum Hash {
    Keccak,
}

#[derive(Clone, Debug, clap::Parser)]
#[command(version, about)]
struct Args {
    #[arg(long, value_enum)]
    hash: Hash,
    #[arg(long)]
    log_permutations: usize,
    #[arg(long)]
    sample_size: Option<usize>,
    #[arg(long, default_value_t = false)]
    trace: bool,
}

fn run_with_hash(hash: &Hash, log_permutations: usize, sample_size: Option<usize>) {
    let num_permutations = 1 << log_permutations;

    match sample_size {
        Some(size) => {
            let (_, time, throughput, proof_size) = match hash {
                Hash::Keccak => bench(num_permutations, size),
            };

            println!(
                "      time: {}\nthroughput: {}\nproof size: {}",
                human_time(time),
                human_throughput(throughput),
                human_size(proof_size),
            );
        }
        None => match hash {
            Hash::Keccak => run(num_permutations),
        },
    }
}

fn main() {
    let args: Args = clap::Parser::parse();

    run_with_hash(&args.hash, args.log_permutations, args.sample_size);
}

fn routine(snark: &HashcasterKeccak) -> (Duration, usize) {
    let input = black_box(snark.generate_input());

    let start = Instant::now();
    let proof = snark.prove(input);
    let elapsed = start.elapsed();

    let proof_size = HashcasterKeccak::serialize_proof(&proof).len();
    drop(black_box(proof));

    (elapsed, proof_size)
}

fn warm_up(snark: &HashcasterKeccak) {
    let mut total_elapsed = Duration::default();
    while total_elapsed.as_secs_f64() < 3.0 {
        total_elapsed += routine(snark).0;
    }
}

pub fn bench(num_permutations: usize, sample_size: usize) -> (usize, Duration, f64, f64) {
    let snark = HashcasterKeccak::new(num_permutations);

    warm_up(&snark);

    let mut total_elapsed = Duration::default();
    let mut total_proof_size = 0;
    for _ in 0..sample_size {
        let (elapsed, proof_size) = routine(&snark);
        total_elapsed += elapsed;
        total_proof_size += proof_size;
    }

    let num_permutations = snark.num_permutations();
    let time = total_elapsed / sample_size as u32;
    let throughput = num_permutations as f64 / time.as_secs_f64();
    let proof_size = total_proof_size as f64 / sample_size as f64;
    (num_permutations, time, throughput, proof_size)
}

pub fn run(num_permutations: usize) {
    let snark = HashcasterKeccak::new(num_permutations);
    let input = black_box(snark.generate_input());
    let proof = snark.prove(input);
    drop(black_box(proof));
}

fn human_time(time: Duration) -> String {
    let time = time.as_nanos();
    if time < 1_000 {
        format!("{time} ns")
    } else if time < 1_000_000 {
        format!("{:.2} µs", time as f64 / 1_000.0)
    } else if time < 1_000_000_000 {
        format!("{:.2} ms", time as f64 / 1_000_000.0)
    } else {
        format!("{:.2} s", time as f64 / 1_000_000_000.0)
    }
}

fn human_size(size: f64) -> String {
    if size < 1000.0 {
        format!("{size:.2} B")
    } else if size < 1000.0 * 2f64.powi(10) {
        format!("{:.2} KB", size / 2f64.powi(10))
    } else {
        format!("{:.2} MB", size / 2f64.powi(20))
    }
}

fn human_throughput(throughput: f64) -> String {
    if throughput < 1_000.0 {
        format!("{throughput:.2} /s")
    } else if throughput < 1_000_000.0 {
        format!("{:.2} K/s", throughput / 1_000.0)
    } else {
        format!("{:.2} M/s", throughput / 1_000_000.0)
    }
}
