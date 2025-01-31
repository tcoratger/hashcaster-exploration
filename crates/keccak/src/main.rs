#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(generic_arg_infer)]

use hashcaster_keccak::{keccak_no_pcs::Keccak, keccak_pcs::HashcasterKeccak};
use rand::{
    rngs::{OsRng, StdRng},
    Rng, SeedableRng,
};
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
    // let args: Args = clap::Parser::parse();

    // run_with_hash(&args.hash, args.log_permutations, args.sample_size);

    // Total number of variables.
    const NUM_VARS: usize = 20;
    // Switch parameter for BoolCheck.
    const C: usize = 5;
    // Number of active variables for LinCheck.
    const NUM_ACTIVE_VARS: usize = 10;

    let rng = &mut OsRng;

    println!("... Initializing protocol ...");

    let start = Instant::now();

    // Initialize the protocol with the given parameters.
    let mut protocol = Keccak::<C>::new(NUM_VARS, NUM_ACTIVE_VARS, rng);

    println!("Generating Keccak inputs took {} ms", (start.elapsed().as_millis()));

    // Execute the BoolCheck protocol.
    protocol.boolcheck(rng);

    // Execute the Multiclaim protocol.
    protocol.multiclaim(rng);

    // Execute the LinCheck protocol.
    protocol.lincheck(rng);

    let end = Instant::now();

    println!("Protocol execution took {} ms", (end - start).as_millis());

    println!("Keccak completed successfully.");
}

fn routine<RNG: Rng>(snark: &HashcasterKeccak, rng: &mut RNG) -> (Duration, usize) {
    let input = black_box(snark.generate_input(rng));

    let start = Instant::now();
    let proof = snark.prove(input);
    let elapsed = start.elapsed();

    let proof_size = HashcasterKeccak::serialize_proof(&proof).len();
    drop(black_box(proof));

    (elapsed, proof_size)
}

fn warm_up<RNG: Rng>(snark: &HashcasterKeccak, rng: &mut RNG) {
    let mut total_elapsed = Duration::default();
    while total_elapsed.as_secs_f64() < 3.0 {
        total_elapsed += routine(snark, rng).0;
    }
}

pub fn bench(num_permutations: usize, sample_size: usize) -> (usize, Duration, f64, f64) {
    let rng = &mut OsRng;

    let snark = HashcasterKeccak::new(num_permutations);

    warm_up(&snark, rng);

    let mut total_elapsed = Duration::default();
    let mut total_proof_size = 0;
    for _ in 0..sample_size {
        let (elapsed, proof_size) = routine(&snark, rng);
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
    let input = black_box(snark.generate_input(&mut StdRng::from_entropy()));
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
