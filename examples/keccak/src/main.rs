use crate::chi::ChiPackage;
use chi::chi_round_witness;
use hashcaster_boolcheck::{algebraic::AlgebraicOps, builder::BoolCheckBuilder, BoolCheckOutput};
use hashcaster_lincheck::{builder::LinCheckBuilder, prodcheck::ProdCheckOutput};
use hashcaster_multiclaim::builder::MulticlaimBuilder;
use hashcaster_primitives::{
    binary_field::BinaryField128b,
    linear_trait::LinearOperations,
    poly::{
        evaluation::Evaluations,
        multinear_lagrangian::MultilinearLagrangianPolynomial,
        point::{Point, Points},
        univariate::UnivariatePolynomial,
    },
    sumcheck::{Sumcheck, SumcheckBuilder},
};
use itertools::Itertools;
use linear::{keccak_linround_witness, KeccakLinear};
use num_traits::MulAdd;
use std::{array, time::Instant};

pub mod chi;
pub mod linear;
pub mod matrix;
pub mod rho_pi;
pub mod theta;

/// Struct to encapsulate protocol state and reusable components.
#[derive(Debug)]
pub struct Keccak {
    /// Number of variables in the protocol.
    num_vars: usize,
    /// Switch parameter for BoolCheck.
    c: usize,
    /// Number of active variables for LinCheck.
    num_active_vars: usize,
    /// Set of random points used in the protocol.
    points: Points,
    /// Polynomials for the protocol.
    polys: [MultilinearLagrangianPolynomial; 5],
    /// Chi package for algebraic operations.
    chi: ChiPackage,
    /// Optional vector of challenges for the protocol.
    challenges: Points,
    /// Optional output from the BoolCheck protocol.
    boolcheck_output: Option<BoolCheckOutput>,
    /// Optional output from the Multiclaim protocol.
    multiclaim_output: Option<UnivariatePolynomial>,
    /// Linear witness for the protocol.
    witness_linear: [MultilinearLagrangianPolynomial; 5],
    /// Evaluation claims for the protocol.
    evaluation_claims: [BinaryField128b; 5],
}

impl Keccak {
    /// Initialize the protocol with configuration and random data.
    pub fn new(num_vars: usize, c: usize, num_active_vars: usize) -> Self {
        // Generate random points for the protocol.
        let points: Points = (0..num_vars).map(|_| BinaryField128b::random()).collect();

        // Generate 5 multilinear lagrangian polynomials with random coefficients.
        let polys: [MultilinearLagrangianPolynomial; 5] =
            array::from_fn(|_| (0..1 << num_vars).map(|_| BinaryField128b::random()).collect());

        // Initialize the Chi package.
        let chi = ChiPackage;

        println!("... Generating witness...");
        let start = Instant::now();

        // Generate witness for the Keccak linear round.
        let witness_linear = keccak_linround_witness(array::from_fn(|i| polys[i].as_slice()));

        // Generate witness for the Chi round.
        let witness_chi = chi_round_witness(&witness_linear);

        // Evaluate the claims using the Chi witness and points.
        let evaluation_claims: [_; 5] = array::from_fn(|i| witness_chi[i].evaluate_at(&points));

        let duration = start.elapsed();
        println!(">>>> Witness generation took {} ms", duration.as_millis());

        // Return the initialized protocol.
        Self {
            num_vars,
            c,
            num_active_vars,
            points,
            polys,
            chi,
            challenges: Default::default(),
            boolcheck_output: None,
            multiclaim_output: None,
            witness_linear,
            evaluation_claims,
        }
    }

    /// Perform the BoolCheck protocol phase.
    pub fn boolcheck(&mut self) {
        println!("... Starting BoolCheck protocol...");
        let start = Instant::now();

        // Generate a random gamma value for folding.
        let gamma = Point(BinaryField128b::random());

        // Initialize the BoolCheck prover builder.
        let mut boolcheck_builder = BoolCheckBuilder::new(
            self.chi.clone(),
            self.c,
            self.points.clone(),
            self.evaluation_claims,
            self.witness_linear.clone(),
        );

        // Build the BoolCheck prover.
        let mut boolcheck_prover = boolcheck_builder.build(&gamma);

        // Initialize the claim.
        let mut claim =
            UnivariatePolynomial::new(self.evaluation_claims.to_vec()).evaluate_at(&gamma);

        // Verify that the claim agrees with the Boolcheck prover
        assert_eq!(
            claim, boolcheck_prover.claim,
            "Claim does not match prover claim after initialization"
        );

        // Reset the challenges
        self.challenges = Points(Vec::new());

        // Perform the BoolCheck rounds.
        for _ in 0..self.num_vars {
            // Compute the round polynomial.
            let round_poly = boolcheck_prover.round_polynomial().coeffs(claim);

            // Generate a random challenge.
            let challenge = Point(BinaryField128b::random());

            // Validate the length of the round polynomial
            assert_eq!(round_poly.len(), 4, "Round polynomial length mismatch");

            // Update the claim by evaluating the round polynomial at the challenge.
            claim = round_poly.evaluate_at(&challenge);

            // Bind the challenge
            boolcheck_prover.bind(&challenge);

            // Store the challenge
            self.challenges.push(challenge);

            // Assert that the updated claim matches the prover's claim.
            assert_eq!(
                claim, boolcheck_prover.claim,
                "BoolCheck claim does not match prover claim after round."
            );
        }

        // Finalize the BoolCheck protocol and retrieve the output.
        let output = boolcheck_prover.finish();

        // Assert the Frobenius evaluations have the expected length.
        assert_eq!(
            output.frob_evals.len(),
            128 * 5,
            "Frobenius evaluations should have length 128 * 5."
        );

        // Clone the Frobenius evaluations
        let mut coord_evals = output.frob_evals.clone();

        // Untwist the Frobenius evaluations
        coord_evals.as_mut_slice().chunks_mut(128).for_each(|chunk| {
            let mut tmp = Evaluations::from(chunk.to_vec());
            tmp.untwist();

            for (i, val) in tmp.iter().enumerate() {
                chunk[i] = *val;
            }
        });

        // Trick for padding
        coord_evals.push(BinaryField128b::ZERO);

        // Compute the claimed evaluations and fold them
        let claimed_evaluations =
            UnivariatePolynomial::new(self.chi.algebraic(&coord_evals, 0, 1)[0].to_vec());
        let folded_claimed_evaluations = claimed_evaluations.evaluate_at(&gamma);

        // Validate the final claim
        assert_eq!(
            folded_claimed_evaluations * *(self.points.eq_eval(&self.challenges)),
            claim,
            "Final claim mismatch"
        );

        let duration = start.elapsed();
        println!(">>>> BoolCheck protocol took {} ms", duration.as_millis());

        self.boolcheck_output = Some(output);
    }

    /// Perform the Multiclaim protocol phase.
    pub fn multiclaim(&mut self) {
        println!("... Starting Multiclaim protocol...");
        let start = Instant::now();

        // Generate a random gamma for folding.
        let gamma = Point(BinaryField128b::random());

        // Compute gamma^128 for evaluation.
        let mut gamma128 = gamma.0;
        for _ in 0..7 {
            gamma128 *= gamma128;
        }

        // Initialize the inverse orbit points
        let mut points_inv_orbit = vec![];
        let mut tmp = self.challenges.clone();
        for _ in 0..128 {
            tmp.iter_mut().for_each(|x| **x = **x * **x);
            points_inv_orbit.push(tmp.clone());
        }
        points_inv_orbit.reverse();

        // Initialize the Multiclaim prover builder.
        let mut multiclaim_builder = MulticlaimBuilder::new(
            self.witness_linear.clone(),
            self.challenges.clone(),
            self.boolcheck_output.clone().unwrap().frob_evals,
        );

        // Build the multiclaim prover
        let mut multiclaim_prover = multiclaim_builder.build(&gamma);

        // Initialize the claim.
        let mut claim =
            UnivariatePolynomial::new(self.boolcheck_output.clone().unwrap().frob_evals.0)
                .evaluate_at(&gamma);

        // Reset the challenges
        self.challenges = Points(Vec::new());

        // Perform the Multiclaim rounds.
        for _ in 0..self.num_vars {
            // Compute the round polynomial.
            let round_poly = multiclaim_prover.round_polynomial().coeffs(claim);

            // Generate a random challenge.
            let challenge = Point(BinaryField128b::random());

            // Validate the length of the round polynomial
            assert_eq!(round_poly.len(), 3, "Round polynomial length mismatch");

            // Update the claim
            claim = round_poly.evaluate_at(&challenge);

            // Bind the challenge
            multiclaim_prover.bind(&challenge);

            // Store the challenge
            self.challenges.push(challenge);
        }

        // Finish the protocol
        let multiclaim_output = multiclaim_prover.finish();

        // Compute the equality evaluations at the challenges
        let eq_evaluations: UnivariatePolynomial =
            points_inv_orbit.iter().map(|pts| pts.eq_eval(&self.challenges).0).collect();

        // Compute the equality evaluation at gamma
        let eq_evaluation = eq_evaluations.evaluate_at(&gamma);

        // Validate the claim
        assert_eq!(multiclaim_output.evaluate_at(&Point(gamma128)) * eq_evaluation, claim);

        let end = Instant::now();

        println!(">> Multiopen took {} ms", (end - start).as_millis());

        // Update the multiclaim output
        self.multiclaim_output = Some(multiclaim_output);
    }

    /// Execute the LinCheck protocol phase.
    pub fn lincheck(&mut self) {
        println!("... Starting LinCheck protocol...");
        let start = Instant::now();

        // Fetch the current challenges for future use.
        let points = self.challenges.clone();

        // Generate a random gamma for folding.
        let gamma = Point(BinaryField128b::random());

        let evaluations: [_; 5] =
            self.multiclaim_output.clone().unwrap().coeffs.try_into().unwrap();

        // Initialize the LinCheck prover builder.
        let mut lincheck_builder = LinCheckBuilder::new(
            self.polys.clone(),
            self.challenges.clone(),
            KeccakLinear::new(),
            self.num_active_vars,
            evaluations,
        );

        // Build the LinCheck prover.
        let mut lincheck_prover = lincheck_builder.build(&gamma);

        // Initialize the claim
        let mut claim = UnivariatePolynomial::new(evaluations.to_vec()).evaluate_at(&gamma);

        let linlayer_preparation = Instant::now();

        println!(
            ">>>> Data preparation for lincheck (clone/restrict) took {} ms",
            (linlayer_preparation - start).as_millis()
        );

        // Reset the challenges
        self.challenges = Points(Vec::new());

        // Perform the LinCheck rounds.
        for _ in 0..self.num_active_vars {
            // Compute the round polynomial.
            let round_poly = lincheck_prover.round_polynomial().coeffs(claim);

            // Generate a random challenge
            let challenge = Point(BinaryField128b::random());

            // Validate the length of the round polynomial
            assert_eq!(round_poly.len(), 3, "Round polynomial length mismatch");

            // Update the claim
            claim = round_poly.evaluate_at(&challenge);

            // Bind the challenge
            lincheck_prover.bind(&challenge);

            // Store the challenge
            self.challenges.push(challenge);
        }

        // Finish the protocol
        let ProdCheckOutput { p_evaluations, .. } = lincheck_prover.finish();

        // Validate the length of the evaluations
        assert_eq!(p_evaluations.len(), 5, "Lincheck evaluations length mismatch");

        // Equality polynomial corresponding to the active variables
        let eq_active_vars = Points(points[..self.num_active_vars].to_vec()).to_eq_poly();
        // Equality polynomial corresponding to the random challenges
        let eq_challenges = self.challenges.to_eq_poly();

        // ----------------

        // Generate a series of scaled equality polynomials by iteratively multiplying by gamma.
        let adj_eq_vec: Vec<_> = (0..5)
            // Start with an initial multiplier of ONE and update it for each iteration.
            .scan(BinaryField128b::ONE, |mult, _| {
                // Store the current value of the multiplier.
                let current_mult = *mult;
                // Update the multiplier by multiplying it with gamma for the next iteration.
                *mult *= *gamma;
                // Map over the equality polynomials, scaling each one by the current multiplier.
                Some(eq_active_vars.iter().map(move |x| *x * current_mult))
            })
            .flatten()
            .collect();

        // Initialize a new KeccakLinear matrix object to perform transformations.
        let m = KeccakLinear::new();

        // Create a mutable array initialized to ZERO for storing the results of the matrix
        // application.
        let mut target = vec![BinaryField128b::ZERO; 5 * (1 << self.num_active_vars)];

        // Apply the transposed linear transformation of the Keccak matrix to the adjusted equality
        // vector, and store the result in the `target` array.
        m.apply_transposed(&adj_eq_vec, &mut target);

        // Compute evaluations of equality constraints by processing each polynomial segment.
        let eq_evals: Vec<_> = (0..5)
            // Iterate over each segment of the target corresponding to the equality constraints.
            .map(|i| {
                // Select the current segment of the target array.
                target[i * (1 << self.num_active_vars)..(i + 1) * (1 << self.num_active_vars)]
                    // Pair the segment elements with the equality challenge values.
                    .iter()
                    .zip(eq_challenges.iter())
                    // Compute the sum of the products of corresponding elements.
                    .fold(BinaryField128b::ZERO, |acc, (a, b)| a.mul_add(*b, acc))
            })
            .collect();

        // Compute the final expected claim by summing the products of the linear and equality
        // evaluations.
        let expected_claim = p_evaluations
            .iter()
            .zip_eq(eq_evals.iter())
            .fold(BinaryField128b::ZERO, |acc, (a, b)| a.mul_add(*b, acc));

        // Ensure the expected claim matches the current claim. If not, the protocol fails.
        assert_eq!(expected_claim, claim);

        // Record the end time for the linear layer processing.
        let end = Instant::now();

        // Extend the list of challenges with additional points beyond the active variables.
        self.challenges.extend(points[self.num_active_vars..].iter().cloned());

        // Verify that each polynomial evaluates correctly at the updated challenge points.
        for i in 0..5 {
            // Ensure the evaluation of the i-th polynomial matches the corresponding expected
            // value.
            assert_eq!(self.polys[i].evaluate_at(&self.challenges), p_evaluations[i]);
        }

        println!(">> Linlayer took {} ms", (end - start).as_millis());
    }
}

/// Main entry point to execute the protocol phases.
pub fn main() {
    // Total number of variables.
    const NUM_VARS: usize = 20;
    // Switch parameter for BoolCheck.
    const C: usize = 5;
    // Number of active variables for LinCheck.
    const NUM_ACTIVE_VARS: usize = 10;

    println!("... Initializing protocol ...");

    let start = Instant::now();

    // Initialize the protocol with the given parameters.
    let mut protocol = Keccak::new(NUM_VARS, C, NUM_ACTIVE_VARS);

    // Execute the BoolCheck protocol.
    protocol.boolcheck();

    // Execute the Multiclaim protocol.
    protocol.multiclaim();

    // Execute the LinCheck protocol.
    protocol.lincheck();

    let end = Instant::now();

    println!("Protocol execution took {} ms", (end - start).as_millis());

    println!("Keccak completed successfully.");
}
