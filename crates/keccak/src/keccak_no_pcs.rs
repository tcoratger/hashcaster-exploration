use crate::{
    chi::{chi_round_witness, ChiPackage},
    linear::{keccak_linround_witness, KeccakLinear},
};
use hashcaster_boolcheck::{algebraic::AlgebraicOps, builder::BoolCheckBuilder, BoolCheckOutput};
use hashcaster_lincheck::{builder::LinCheckBuilder, prodcheck::ProdCheckOutput};
use hashcaster_multiclaim::builder::MulticlaimBuilder;
use hashcaster_primitives::{
    binary_field::BinaryField128b,
    linear_trait::LinearOperations,
    poly::{
        evaluation::FixedEvaluations,
        multinear_lagrangian::MultilinearLagrangianPolynomial,
        point::Points,
        univariate::{FixedUnivariatePolynomial, UnivariatePolynomial},
    },
    sumcheck::{Sumcheck, SumcheckBuilder},
};
use itertools::Itertools;
use num_traits::{MulAdd, Pow};
use rand::Rng;
use std::{array, time::Instant};

/// Struct to encapsulate protocol state and reusable components.
#[derive(Debug)]
pub struct Keccak<const C: usize> {
    /// Number of variables in the protocol.
    num_vars: usize,
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
    boolcheck_output: Option<BoolCheckOutput<{ 128 * 5 }, 3>>,
    /// Optional output from the Multiclaim protocol.
    multiclaim_output: Option<FixedEvaluations<5>>,
    /// Linear witness for the protocol.
    witness_linear: [MultilinearLagrangianPolynomial; 5],
    /// Evaluation claims for the protocol.
    evaluation_claims: [BinaryField128b; 5],
}

impl<const C: usize> Keccak<C> {
    /// Initialize the protocol with configuration and random data.
    pub fn new<RNG: Rng>(num_vars: usize, num_active_vars: usize, rng: &mut RNG) -> Self {
        // Generate random points for the protocol.
        let points = Points::random(num_vars, rng);

        // Generate 5 multilinear lagrangian polynomials with random coefficients.
        let polys: [_; 5] =
            array::from_fn(|_| MultilinearLagrangianPolynomial::random(1 << num_vars, rng));

        // Initialize the Chi package.
        let chi = ChiPackage;

        println!("... Generating witness...");
        let start = Instant::now();

        // Generate witness for the Keccak linear round.
        let witness_linear =
            keccak_linround_witness([&polys[0], &polys[1], &polys[2], &polys[3], &polys[4]]);

        // Generate witness for the Chi round.
        let witness_chi = chi_round_witness(&witness_linear);

        let witness_finish = Instant::now();
        println!(">>>> Witness generation took {} ms", (witness_finish - start).as_millis());

        // Evaluate the claims using the Chi witness and points.
        let evaluation_claims: [_; 5] = array::from_fn(|i| witness_chi[i].evaluate_at(&points));

        let evaluation_finish = Instant::now();
        println!(
            ">>>> Evaluation claims took {} ms",
            (evaluation_finish - witness_finish).as_millis()
        );
        println!(
            ">>>> Total time for witness and evaluation took {} ms",
            (evaluation_finish - start).as_millis()
        );

        // Return the initialized protocol.
        Self {
            num_vars,
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
    pub fn boolcheck<RNG: Rng>(&mut self, rng: &mut RNG) {
        println!("... Starting BoolCheck protocol...");
        let start = Instant::now();

        // Generate a random gamma value for folding.
        let gamma = BinaryField128b::random(rng);

        // Initialize the BoolCheck prover builder.
        let boolcheck_builder = BoolCheckBuilder::<_, _, C, _>::new(
            &self.chi,
            &self.points,
            self.evaluation_claims,
            &self.witness_linear,
        );

        let end_boolcheck_init = Instant::now();

        println!(
            ">>>> Boolcheck initialization took {} ms",
            (end_boolcheck_init - start).as_millis()
        );

        // Build the BoolCheck prover.
        let mut boolcheck_prover = boolcheck_builder.build(&gamma);

        let end_boolcheck_extension = Instant::now();

        println!(
            ">>>> Boolcheck table extension took {} ms",
            (end_boolcheck_extension - end_boolcheck_init).as_millis()
        );

        // Initialize the claim.
        let mut claim = FixedUnivariatePolynomial::new(self.evaluation_claims).evaluate_at(&gamma);

        // Verify that the claim agrees with the Boolcheck prover
        assert_eq!(
            claim, boolcheck_prover.claim,
            "Claim does not match prover claim after initialization"
        );

        // Reset the challenges
        self.challenges = Points::default();

        // Perform the BoolCheck rounds.
        for _ in 0..self.num_vars {
            // Compute the round polynomial.
            let round_poly = boolcheck_prover.round_polynomial().coeffs(claim);

            // Generate a random challenge.
            let challenge = BinaryField128b::random(rng);

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

        let boolcheck_final = Instant::now();

        println!(
            ">>>> Boolcheck rounds took {} ms",
            (boolcheck_final - end_boolcheck_extension).as_millis()
        );

        // Assert the Frobenius evaluations have the expected length.
        assert_eq!(
            output.frob_evals.len(),
            128 * 5,
            "Frobenius evaluations should have length 128 * 5."
        );

        // Clone the Frobenius evaluations
        let mut coord_evals = output.frob_evals.clone();

        // Untwist the Frobenius evaluations
        coord_evals.untwist();

        // Compute the claimed evaluations and fold them
        let claimed_evaluations =
            FixedUnivariatePolynomial::new(self.chi.algebraic(coord_evals.as_ref(), 0, 1)[0]);
        let folded_claimed_evaluations = claimed_evaluations.evaluate_at(&gamma);

        // Validate the final claim
        assert_eq!(
            folded_claimed_evaluations * (self.points.eq_eval(&self.challenges)),
            claim,
            "Final claim mismatch"
        );

        let end_boolcheck_verify = Instant::now();
        println!(
            ">>>> BoolCheck verifier took {} ms",
            (end_boolcheck_verify - boolcheck_final).as_millis()
        );
        println!(
            ">>>> BoolCheck total protocol took {} ms",
            (end_boolcheck_verify - start).as_millis()
        );

        self.boolcheck_output = Some(output);
    }

    /// Perform the Multiclaim protocol phase.
    pub fn multiclaim<RNG: Rng>(&mut self, rng: &mut RNG) {
        println!("... Starting Multiclaim protocol...");
        let start = Instant::now();

        // Generate a random gamma for folding.
        let gamma = BinaryField128b::random(rng);

        // Compute gamma^128 for evaluation.
        let gamma128 = gamma.pow(128);

        // Initialize the inverse orbit points
        let points_inv_orbit = self.challenges.to_points_inv_orbit();

        let challenges = self.challenges.clone();

        // Initialize the Multiclaim prover builder.
        let boolcheck_output = self.boolcheck_output.clone().unwrap();
        let multiclaim_builder =
            MulticlaimBuilder::new(&self.witness_linear, &challenges, &boolcheck_output.frob_evals);

        // Build the multiclaim prover
        let mut multiclaim_prover = multiclaim_builder.build(&gamma);

        // Initialize the claim.
        let mut claim =
            FixedUnivariatePolynomial::new(self.boolcheck_output.clone().unwrap().frob_evals.0)
                .evaluate_at(&gamma);

        // Reset the challenges
        self.challenges = Points::default();

        // Perform the Multiclaim rounds.
        for _ in 0..self.num_vars {
            // Compute the round polynomial.
            let round_poly = multiclaim_prover.round_polynomial().coeffs(claim);

            // Generate a random challenge.
            let challenge = BinaryField128b::random(rng);

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
            points_inv_orbit.iter().map(|pts| pts.eq_eval(&self.challenges)).collect();

        // Compute the equality evaluation at gamma
        let eq_evaluation = eq_evaluations.evaluate_at(&gamma);

        // Validate the claim
        assert_eq!(
            FixedUnivariatePolynomial::new(multiclaim_output.0).evaluate_at(&gamma128) *
                eq_evaluation,
            claim
        );

        let end = Instant::now();

        println!(">> Multiopen took {} ms", (end - start).as_millis());

        // Update the multiclaim output
        self.multiclaim_output = Some(multiclaim_output);
    }

    /// Execute the LinCheck protocol phase.
    pub fn lincheck<RNG: Rng>(&mut self, rng: &mut RNG) {
        println!("... Starting LinCheck protocol...");
        let start = Instant::now();

        // Fetch the current challenges for future use.
        let points = self.challenges.clone();

        // Generate a random gamma for folding.
        let gamma = BinaryField128b::random(rng);

        let evaluations: [_; 5] = self.multiclaim_output.clone().unwrap().0;

        // Initialize the LinCheck prover builder.
        let matrix = KeccakLinear::new();
        let lincheck_builder = LinCheckBuilder::new(
            &self.polys,
            &self.challenges,
            &matrix,
            self.num_active_vars,
            evaluations,
        );

        // Build the LinCheck prover.
        let mut lincheck_prover = lincheck_builder.build(&gamma);

        // Initialize the claim
        let mut claim = FixedUnivariatePolynomial::new(evaluations).evaluate_at(&gamma);

        let linlayer_preparation = Instant::now();

        println!(
            ">>>> Data preparation for lincheck (clone/restrict) took {} ms",
            (linlayer_preparation - start).as_millis()
        );

        // Reset the challenges
        self.challenges = Points::default();

        // Perform the LinCheck rounds.
        for _ in 0..self.num_active_vars {
            // Compute the round polynomial.
            let round_poly = lincheck_prover.round_polynomial().coeffs(claim);

            // Generate a random challenge
            let challenge = BinaryField128b::random(rng);

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

        // Generate a series of scaled equality polynomials by iteratively multiplying by gamma.
        let adj_eq_vec: Vec<_> = (0..5)
            // Start with an initial multiplier of ONE and update it for each iteration.
            .scan(BinaryField128b::ONE, |mult, _| {
                // Store the current value of the multiplier.
                let current_mult = *mult;
                // Update the multiplier by multiplying it with gamma for the next iteration.
                *mult *= gamma;
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
        self.challenges.extend(points[self.num_active_vars..].iter().copied());

        // Verify that each polynomial evaluates correctly at the updated challenge points.
        for i in 0..5 {
            // Ensure the evaluation of the i-th polynomial matches the corresponding expected
            // value.
            assert_eq!(self.polys[i].evaluate_at(&self.challenges), p_evaluations[i]);
        }

        println!(">> Linlayer took {} ms", (end - start).as_millis());
    }
}

#[cfg(test)]
mod tests {
    use rand::rngs::OsRng;

    use super::*;

    #[test]
    fn test_keccak_without_pcs() {
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
}
