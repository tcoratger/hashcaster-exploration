use crate::chi::ChiPackage;
use hashcaster_boolcheck::builder::BoolCheckBuilder;
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
