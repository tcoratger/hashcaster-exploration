use crate::{
    chi::{chi_round_witness, ChiPackage},
    linear::{keccak_linround_witness, KeccakLinear},
};
use binius_core::tower::{AESTowerFamily, TowerFamily};
use binius_field::{arch::OptimalUnderlier, PackedField};
use binius_hash::{Groestl256, GroestlDigest, GroestlDigestCompression};
use binius_math::IsomorphicEvaluationDomainFactory;
use hashcaster_boolcheck::{algebraic::AlgebraicOps, builder::BoolCheckBuilder};
use hashcaster_lincheck::builder::LinCheckBuilder;
use hashcaster_multiclaim::builder::MulticlaimBuilder;
use hashcaster_pcs::{
    challenger::F128Challenger,
    proof::{FriPcsProof, SumcheckProof},
    BatchFRIPCS128,
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
use num_traits::{MulAdd, Pow};
use p3_challenger::{CanObserve, CanSample};
use std::array::{self, from_fn};

const NUM_VARS_PER_PERMUTATIONS: usize = 2;
const BOOL_CHECK_C: usize = 5;
const LIN_CHECK_NUM_VARS: usize = 10;
const PCS_LOG_INV_RATE: usize = 1;
const SECURITY_BITS: usize = 100;
const BATCH_SIZE: usize = 5;

#[derive(Clone, Debug)]
pub struct HashcasterKeccakProof {
    commitment: GroestlDigest<<Tower as TowerFamily>::B8>,
    initial_claims: [BinaryField128b; 5],
    rounds: [(SumcheckProof, SumcheckProof, SumcheckProof); 24],
    input_open_proof: FriPcsProof,
}

/// An alias for the optimal underylying field of the tower.
///
/// 128-bit value that is used for 128-bit SIMD operations
type U = OptimalUnderlier;

/// The AES optimized tower family.
type Tower = AESTowerFamily;

/// The domain factory for the tower.
type DomainFactory = IsomorphicEvaluationDomainFactory<<Tower as TowerFamily>::B8>;

/// Binius Groestl256 hash function.
type BiniusGroestl256 = Groestl256<<Tower as TowerFamily>::B128, <Tower as TowerFamily>::B8>;

/// Binius compression function for Grøstl hash digests based on the Grøstl output transformation.
type BiniusGroestlDigestCompression = GroestlDigestCompression<<Tower as TowerFamily>::B8>;

/// Binius Groestl hash digest.
type BiniusGroestlDigest = GroestlDigest<<Tower as TowerFamily>::B8>;

#[allow(missing_debug_implementations)]
pub struct HashcasterKeccak {
    /// Number of permutations
    num_permutations: usize,
    /// The Polynomial Commitment Scheme
    pcs: BatchFRIPCS128<
        Tower,
        U,
        BiniusGroestlDigest,
        DomainFactory,
        BiniusGroestl256,
        BiniusGroestlDigestCompression,
    >,
}

impl HashcasterKeccak {
    pub fn new(num_permutations: usize) -> Self {
        let num_vars = (num_permutations.ilog2() as usize + NUM_VARS_PER_PERMUTATIONS).max(10);

        Self {
            num_permutations: 3 << (num_vars - 3),
            pcs: BatchFRIPCS128::new(SECURITY_BITS, PCS_LOG_INV_RATE, num_vars, BATCH_SIZE),
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

    fn prove(&self, input: [MultilinearLagrangianPolynomial; 5]) -> HashcasterKeccakProof {
        let mut challenger = F128Challenger::new_keccak256();

        let layers = (0..24usize).fold(vec![input], |mut layers, _| {
            layers.extend({
                let last = layers.last().unwrap();
                let lin =
                    keccak_linround_witness([&last[0], &last[1], &last[2], &last[3], &last[4]]);
                [lin.clone(), chi_round_witness(&lin)]
            });
            layers
        });

        let (input_packed, commitment, committed) = self.pcs.commit(&layers[0].to_vec().into());

        // Observe the commitment
        commitment.iter().for_each(|scalar| challenger.observe(scalar));

        // Sample some random points
        let mut points: Points = challenger.sample_vec(self.num_vars()).into();

        // Compute the initial claim
        let initial_claims: [_; 5] =
            layers.last().unwrap().each_ref().map(|poly| poly.evaluate_at(&points));

        // Challenger observe the initial claim
        challenger.observe_slice(&initial_claims);

        let mut layers_rev = layers.iter().rev().skip(1);

        // Initial claim to be used in the main loop.
        let mut claims = initial_claims;

        let rounds: [_; 24] = array::from_fn(|_| {
            let (bool_check_proof, multi_open_proof, lin_check_proof);

            (bool_check_proof, multi_open_proof, points) =
                self.prove_chi(layers_rev.next().unwrap(), &points, &claims, &mut challenger);

            claims = multi_open_proof.evals.clone().0.try_into().unwrap();

            (lin_check_proof, points) = self.prove_lin(
                KeccakLinear::new(),
                layers_rev.next().unwrap(),
                &points,
                &claims,
                &mut challenger,
            );

            claims = lin_check_proof.evals.clone().0.try_into().unwrap();

            (bool_check_proof, multi_open_proof, lin_check_proof)
        });

        let input_open_proof = self.pcs.open(&input_packed, &committed, &points);

        HashcasterKeccakProof { commitment, initial_claims, rounds, input_open_proof }
    }

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

        // Return:
        // - BoolCheck proof
        // - Multiclaim proof
        // - Final points
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
            let gamma128 = gamma.0.pow(128);

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
