use crate::chi::ChiPackage;
use hashcaster_boolcheck::{algebraic::AlgebraicOps, builder::BoolCheckBuilder};
use hashcaster_lincheck::builder::LinCheckBuilder;
use hashcaster_multiclaim::builder::MulticlaimBuilder;
use hashcaster_pcs::{
    challenger::F128Challenger,
    proof::{FriPcsProof, SumcheckProof},
    BatchFriPcs,
};
use hashcaster_primitives::{
    binary_field::BinaryField128b,
    linear_trait::LinearOperations,
    poly::{
        multinear_lagrangian::MultilinearLagrangianPolynomial,
        point::{Point, Points},
        univariate::UnivariatePolynomial,
    },
    sumcheck::{EvaluationProvider, Sumcheck, SumcheckBuilder},
};
use itertools::Itertools;
use num_traits::MulAdd;
use p3_challenger::{CanObserve, CanSample};
use serde::{Deserialize, Serialize};
use std::array::from_fn;

const NUM_VARS_PER_PERMUTATIONS: usize = 2;
const BOOL_CHECK_C: usize = 5;
const LIN_CHECK_NUM_VARS: usize = 10;
const PCS_LOG_INV_RATE: usize = 1;
const SECURITY_BITS: usize = 100;
const BATCH_SIZE: usize = 5;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HashcasterKeccakProof {
    initial_claims: [BinaryField128b; 5],
    rounds: [(SumcheckProof, SumcheckProof, SumcheckProof); 24],
    input_open_proof: FriPcsProof,
}

#[derive(Debug)]
pub struct HashcasterKeccak {
    /// Number of permutations
    num_permutations: usize,
    /// The Polynomial Commitment Scheme
    pcs: BatchFriPcs,
}

impl HashcasterKeccak {
    pub fn new(num_permutations: usize) -> Self {
        let num_vars = (num_permutations.ilog2() as usize + NUM_VARS_PER_PERMUTATIONS).max(10);

        Self {
            num_permutations: 3 << (num_vars - 3),
            pcs: BatchFriPcs::new(SECURITY_BITS, PCS_LOG_INV_RATE, num_vars, BATCH_SIZE),
        }
    }

    pub const fn num_permutations(&self) -> usize {
        self.num_permutations
    }

    const fn num_vars(&self) -> usize {
        self.num_permutations.ilog2() as usize + NUM_VARS_PER_PERMUTATIONS
    }

    fn generate_input(&self) -> [MultilinearLagrangianPolynomial; 5] {
        let num_vars = self.num_vars();
        from_fn(|_| MultilinearLagrangianPolynomial::random(1 << num_vars))
    }

    // fn prove(&self, input: [MultilinearLagrangianPolynomial; 5]) -> HashcasterKeccakProof {
    //     let mut challenger = F128Challenger::new_keccak256();

    //     let layers = (0..24usize).fold(vec![input], |mut layers, _| {
    //         let last = layers.last().unwrap();
    //         let lin = keccak_linround_witness(last.map(|poly| poly.as_slice()));
    //         let chi = chi_round_witness(&lin);
    //         layers.extend([lin, chi]);
    //         layers
    //     });

    //     let CommitOutput { commitment, committed, codeword } =
    //         self.pcs.commit(&layers[0].to_vec().into());

    //     commitment.iter().for_each(|scalar| challenger.observe(scalar));
    // }

    pub fn prove_chi(
        &self,
        input: &[MultilinearLagrangianPolynomial; 5],
        points: &Points,
        claims: &[BinaryField128b; 5],
        challenger: &mut F128Challenger,
    ) -> (SumcheckProof, SumcheckProof, Points) {
        // Determine the number of variables in the polynomials.
        let num_vars = self.num_vars();

        // Step 1: Perform the BoolCheck sumcheck.
        let (bool_check_proof, updated_points) = {
            // Define the Chi package for the BoolCheck protocol.
            let chi = ChiPackage;

            // Initialize the BoolCheck builder with the Chi package, points, claims, and input.
            let mut builder =
                BoolCheckBuilder::new(chi, BOOL_CHECK_C, points.clone(), *claims, input.clone());

            // Perform the BoolCheck sumcheck using the helper function.
            perform_sumcheck(num_vars, &mut builder, challenger, claims)
        };

        // Step 2: Perform the Multiclaim sumcheck.
        let (multi_open_proof, final_points) = {
            // Initialize the Multiclaim builder with the input, updated points, and claims.
            let mut builder =
                MulticlaimBuilder::new(input.clone(), updated_points, claims.to_vec().into());

            // Perform the Multiclaim sumcheck using the helper function.
            perform_sumcheck(num_vars, &mut builder, challenger, claims)
        };

        // Return the proofs for the BoolCheck and Multiclaim protocols, along with the final
        // points.
        (bool_check_proof, multi_open_proof, final_points)
    }

    #[allow(clippy::unused_self)]
    fn prove_lin(
        &self,
        matrix: impl LinearOperations,
        input: &[MultilinearLagrangianPolynomial; 5],
        points: &Points,
        claims: &[BinaryField128b; 5],
        challenger: &mut F128Challenger,
    ) -> (SumcheckProof, Points) {
        // Perform the sumcheck process for LinCheck using the shared helper function.
        let mut lincheck_builder = LinCheckBuilder::new(
            input.clone(),
            points.clone(),
            matrix,
            LIN_CHECK_NUM_VARS,
            *claims,
        );

        perform_sumcheck(LIN_CHECK_NUM_VARS, &mut lincheck_builder, challenger, claims)
    }

    pub fn verify_lin(
        &self,
        matrix: &impl LinearOperations,
        points: &Points,
        claims: &[BinaryField128b; 5],
        lin_check_proof: &SumcheckProof,
        challenger: &mut F128Challenger,
    ) -> Points {
        // Verify the number of round polynomials in the LinCheck proof.
        assert_eq!(lin_check_proof.round_polys.len(), LIN_CHECK_NUM_VARS);

        // Verify the number of evaluations in the LinCheck proof.
        assert_eq!(lin_check_proof.evals.len(), 5);

        // Perform the common verification steps
        let (claim, mut rs, gamma) = perform_verification(
            challenger,
            lin_check_proof,
            &UnivariatePolynomial::new(claims.to_vec()),
        );

        // Equality polynomial corresponding to the active variables
        let eq_active_vars = Points(points[..LIN_CHECK_NUM_VARS].to_vec()).to_eq_poly();
        // Equality polynomial corresponding to the random challenges
        let eq_challenges = rs.to_eq_poly();

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

        // Create a mutable array initialized to ZERO for storing the results of the matrix
        // application.
        let mut target = vec![BinaryField128b::ZERO; 5 * (1 << LIN_CHECK_NUM_VARS)];

        // Apply the transposed matrix to the scaled equality polynomials.
        matrix.apply_transposed(&adj_eq_vec, &mut target);

        // Compute evaluations of equality constraints by processing each polynomial segment.
        let eq_evals: Vec<_> = (0..5)
            // Iterate over each segment of the target corresponding to the equality constraints.
            .map(|i| {
                // Select the current segment of the target array.
                target[i * (1 << LIN_CHECK_NUM_VARS)..(i + 1) * (1 << LIN_CHECK_NUM_VARS)]
                    // Pair the segment elements with the equality challenge values.
                    .iter()
                    .zip(eq_challenges.iter())
                    // Compute the sum of the products of corresponding elements.
                    .fold(BinaryField128b::ZERO, |acc, (a, b)| a.mul_add(*b, acc))
            })
            .collect();

        // Compute the final expected claim by summing the products of the linear and equality
        // evaluations.
        let expected_claim = lin_check_proof
            .evals
            .iter()
            .zip_eq(eq_evals.iter())
            .fold(BinaryField128b::ZERO, |acc, (a, b)| a.mul_add(*b, acc));

        // Verify the claim by checking if it matches the expected claim.
        assert_eq!(claim, expected_claim);

        // Add the points to the challenges vector.
        rs.extend_from_slice(&points[LIN_CHECK_NUM_VARS..]);

        rs
    }

    pub fn verify_chi(
        &self,
        points: &Points,
        claims: &[BinaryField128b; 5],
        bool_check_proof: &SumcheckProof,
        multi_open_proof: &SumcheckProof,
        challenger: &mut F128Challenger,
    ) -> Points {
        // Verify the number of round polynomials in the LinCheck proof.
        assert_eq!(bool_check_proof.round_polys.len(), self.num_vars());

        // Verify the number of round polynomials in the Multiclaim proof.
        assert_eq!(multi_open_proof.round_polys.len(), self.num_vars());

        // Verify the number of evaluations in the BoolCheck proof.
        assert_eq!(bool_check_proof.evals.len(), 128 * 5);

        // Verify the number of evaluations in the Multiclaim proof.
        assert_eq!(multi_open_proof.evals.len(), 5);

        let updated_points = {
            // Setup the ChiPackage
            let chi = ChiPackage {};

            // Perform common verification for BoolCheck
            let (claim, rs, gamma) = perform_verification(
                challenger,
                bool_check_proof,
                &UnivariatePolynomial::new(claims.to_vec()),
            );

            // Fetch the frobenius evaluations from the proof.
            let frob_evals = &bool_check_proof.evals;

            // Challenger observes the extracted evaluations.
            challenger.observe_slice(frob_evals);

            // Clone and store the frobenius evaluations
            let mut coord_evals = frob_evals.clone();

            // Untwist the frobenius evaluations
            coord_evals.untwist();

            // Trick for padding
            coord_evals.push(BinaryField128b::ZERO);

            // Compute the claimed evaluations and fold them
            let claimed_evaluations =
                UnivariatePolynomial::new(chi.algebraic(&coord_evals, 0, 1)[0].to_vec());
            let folded_claimed_evaluations = claimed_evaluations.evaluate_at(&gamma);

            // Validate the final claim
            assert_eq!(
                folded_claimed_evaluations * *(points.eq_eval(&rs)),
                claim,
                "Final claim mismatch"
            );

            rs
        };

        let final_points = {
            // Perform common verification for MultiOpen
            let (claim, rs, gamma) = perform_verification(
                challenger,
                multi_open_proof,
                &UnivariatePolynomial::new(bool_check_proof.evals.0.clone()),
            );

            // Fetch the evaluations from the proof.
            let evals = &multi_open_proof.evals;

            // Challenger observes the extracted evaluations.
            challenger.observe_slice(evals);

            // Initialize the inverse orbit points
            let points_inv_orbit = updated_points.to_points_inv_orbit();

            // Compute gamma^128 for evaluation.
            let mut gamma128 = gamma.0;
            for _ in 0..7 {
                gamma128 *= gamma128;
            }

            // Compute the equality evaluations at the challenges
            let eq_evaluations: UnivariatePolynomial =
                points_inv_orbit.iter().map(|pts| pts.eq_eval(&rs).0).collect();

            // Compute the equality evaluation at gamma
            let eq_evaluation = eq_evaluations.evaluate_at(&gamma);

            // Validate the claim
            assert_eq!(
                UnivariatePolynomial::new(evals.clone().0).evaluate_at(&Point(gamma128)) *
                    eq_evaluation,
                claim
            );

            rs
        };

        final_points
    }
}

// Helper function to perform a sumcheck round.
fn perform_sumcheck<B>(
    num_vars: usize,
    builder: &mut B,
    challenger: &mut F128Challenger,
    claims: &[BinaryField128b],
) -> (SumcheckProof, Points)
where
    B: SumcheckBuilder,
{
    // Sample the initial folding challenge.
    let gamma = Point(challenger.sample());

    // Build the prover for the sumcheck protocol.
    let mut prover = builder.build(&gamma);

    // Initialize the claim by evaluating the claims at the initial challenge.
    let mut claim = UnivariatePolynomial::new(claims.to_vec()).evaluate_at(&gamma);

    // Initialize vectors to store round polynomials and challenges.
    let mut round_polys = Vec::new();
    let mut rs = Points(Vec::new());

    for _ in 0..num_vars {
        // Compute the round polynomial for the current sumcheck round.
        let round_poly = prover.round_polynomial();

        // Challenger observes the coefficients of the round polynomial.
        challenger.observe_slice(&round_poly.0);

        // Challenger samples a new random challenge for this round.
        let r = Point(challenger.sample());

        // Update the claim by evaluating the round polynomial at the sampled challenge.
        claim = round_poly.coeffs(claim).evaluate_at(&r);

        // Bind the prover state to the new challenge.
        prover.bind(&r);

        // Store the round polynomial for this round.
        round_polys.push(round_poly);

        // Store the challenge for this round.
        rs.push(r);
    }

    // Finalize the prover and extract evaluations from the result.
    let evals = prover.finish().evals();

    // Challenger observes the extracted evaluations.
    challenger.observe_slice(&evals);

    // Return the proof and updated points (challenges).
    (SumcheckProof { round_polys, evals }, rs)
}

fn perform_verification(
    challenger: &mut F128Challenger,
    proof: &SumcheckProof,
    initial_claim_poly: &UnivariatePolynomial,
) -> (BinaryField128b, Points, Point) {
    // Sample a gamma
    let gamma = Point(challenger.sample());

    // Evaluate the initial claim at gamma
    let mut claim = initial_claim_poly.evaluate_at(&gamma);

    // Initialize a vector to store the challenges
    let mut rs = Points(Vec::new());

    // Iterate over the round polynomials in the proof
    for round_poly in &proof.round_polys {
        // Challenger observes the round polynomial coefficients
        challenger.observe_slice(&round_poly.0);

        // Challenger samples a new challenge for this round
        let r = Point(challenger.sample());

        // Update the claim by evaluating the round polynomial at the sampled challenge
        claim = round_poly.coeffs(claim).evaluate_at(&r);

        // Store the sampled challenge
        rs.push(r);
    }

    // Fetch and observe the proof's evaluations
    challenger.observe_slice(&proof.evals);

    (claim, rs, gamma)
}
