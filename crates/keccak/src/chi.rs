use hashcaster_boolcheck::algebraic::AlgebraicOps;
use hashcaster_primitives::{
    binary_field::BinaryField128b, poly::multinear_lagrangian::MultilinearLagrangianPolynomial,
};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::{
    array,
    sync::atomic::{AtomicPtr, Ordering},
};

/// Computes the "negation" of a field element in a specialized way.
///
/// # Explanation
/// - The function computes `111...1` (all bits set to 1) in the standard basis and adds the input
///   field element `x`.
/// - This operation effectively performs the operation \( \text{neg}(x) = (2^n - 1) + x \).
///
/// # Parameters
/// - `x`: The input field element of type `BinaryField128b`.
///
/// # Returns
/// - A `BinaryField128b` instance representing the "negation".
fn neg(x: BinaryField128b) -> BinaryField128b {
    BinaryField128b::MAX + x
}

/// Computes the chi-compressed transformation for 5 field elements.
///
/// # Explanation
/// - The `chi` operation combines linear and quadratic terms to transform the input array into a
///   new array.
/// - Each output is computed as: \[ y_i = x_i + \neg(x_{i+1}) \& x_{i+2} \]
///
/// # Parameters
/// - `arg`: An array of 5 `BinaryField128b` field elements.
///
/// # Returns
/// - A new array of 5 transformed `BinaryField128b` elements.
fn chi_compressed(arg: [BinaryField128b; 5]) -> [BinaryField128b; 5] {
    [
        arg[0] + (neg(arg[1]) & arg[2]),
        arg[1] + (neg(arg[2]) & arg[3]),
        arg[2] + (neg(arg[3]) & arg[4]),
        arg[3] + (neg(arg[4]) & arg[0]),
        arg[4] + (neg(arg[0]) & arg[1]),
    ]
}

/// A struct representing the Chi transformation package.
#[derive(Debug, Clone, Default)]
pub struct ChiPackage;

impl AlgebraicOps<5, 5> for ChiPackage {
    /// Computes the linear component of the Chi transformation.
    ///
    /// # Formula
    /// - \[ y_i = x_i + x_{i+2} \]
    ///
    /// # Parameters
    /// - `arg`: A reference to an array of 5 field elements.
    ///
    /// # Returns
    /// - A new array of 5 transformed field elements.
    fn linear(&self, arg: &[BinaryField128b; 5]) -> [BinaryField128b; 5] {
        [arg[0] + arg[2], arg[1] + arg[3], arg[2] + arg[4], arg[3] + arg[0], arg[4] + arg[1]]
    }

    /// Computes the quadratic component of the Chi transformation.
    ///
    /// # Formula
    /// - \[ y_i = x_{i+1} \& x_{i+2} \]
    ///
    /// # Parameters
    /// - `arg`: A reference to an array of 5 field elements.
    ///
    /// # Returns
    /// - A new array of 5 transformed field elements.
    fn quadratic(&self, arg: &[BinaryField128b; 5]) -> [BinaryField128b; 5] {
        [arg[1] & arg[2], arg[2] & arg[3], arg[3] & arg[4], arg[4] & arg[0], arg[0] & arg[1]]
    }

    /// Computes the algebraic terms for the Chi transformation using a combination
    /// of linear and quadratic terms.
    ///
    /// # Parameters
    /// - `data`: A slice of field elements representing the input data.
    /// - `idx_a`: The base index for the computation.
    /// - `offset`: The offset between successive terms in the input data.
    ///
    /// # Returns
    /// - A 2D array \([3][5]\) representing the computed algebraic terms.
    fn algebraic(
        &self,
        data: &[BinaryField128b],
        idx_a: usize,
        offset: usize,
    ) -> [[BinaryField128b; 5]; 3] {
        let mut idxs: [usize; 5] = array::from_fn(|j| idx_a * 2 + j * offset * 128);
        let mut ret = [[BinaryField128b::ZERO; 5]; 3];

        (0..128).for_each(|i| {
            let basis = BinaryField128b::basis(i);
            let one = BinaryField128b::ONE;

            // Linear and quadratic terms
            (0..5).for_each(|j| {
                let next = (j + 1) % 5;
                let next_next = (j + 2) % 5;

                // Linear terms for ret[0] and ret[1]
                ret[0][j] +=
                    basis * (data[idxs[j]] + (one + data[idxs[next]]) * data[idxs[next_next]]);
                ret[1][j] += basis *
                    (data[idxs[j] + 1] +
                        (one + data[idxs[next] + 1]) * data[idxs[next_next] + 1]);

                // Quadratic terms for ret[2]
                ret[2][j] += basis *
                    ((data[idxs[next]] + data[idxs[next] + 1]) *
                        (data[idxs[next_next]] + data[idxs[next_next] + 1]));
            });

            // Update indices for the next iteration
            idxs.iter_mut().for_each(|idx| *idx += offset);
        });

        ret
    }
}

/// Computes the Chi transformation for each element in the input polynomials
/// and stores the result in new polynomials.
///
/// # Parameters
/// - `polys`: A reference to an array of 5 multilinear lagrangian polynomials.
///
/// # Returns
/// - A new array of 5 multilinear lagrangian polynomials representing the transformed values.
pub fn chi_round_witness(
    polys: &[MultilinearLagrangianPolynomial; 5],
) -> [MultilinearLagrangianPolynomial; 5] {
    // Ensure all input vectors have the same length
    let l = polys[0].len();
    assert!(polys.iter().all(|p| p.len() == l), "Input vectors must have the same length");

    // Initialize result vectors
    let mut ret =
        array::from_fn(|_| MultilinearLagrangianPolynomial::new(vec![BinaryField128b::ZERO; l]));

    // Create atomic pointers for each result vector
    let ret_ptrs: Vec<_> = ret.iter_mut().map(|r| AtomicPtr::new(r.as_mut_ptr())).collect();

    let iter = (0..l).into_par_iter();

    // Perform the chi-compressed computation in parallel
    iter.for_each(|i| {
        // Compute chi-compressed values for the current index
        let tmp = chi_compressed([polys[0][i], polys[1][i], polys[2][i], polys[3][i], polys[4][i]]);

        // Update result vectors using atomic pointers
        for (j, value) in tmp.into_iter().enumerate() {
            unsafe {
                *ret_ptrs[j].load(Ordering::SeqCst).add(i) = value;
            }
        }
    });

    ret
}

#[cfg(test)]
mod tests {
    use super::*;
    use hashcaster_boolcheck::{builder::BoolCheckBuilder, BoolCheckOutput};
    use hashcaster_primitives::{
        poly::{
            evaluation::Evaluations,
            multinear_lagrangian::MultilinearLagrangianPolynomial,
            point::{Point, Points},
            univariate::UnivariatePolynomial,
        },
        sumcheck::{Sumcheck, SumcheckBuilder},
    };
    use itertools::Itertools;
    use rand::rngs::OsRng;
    use std::array;

    #[test]
    fn test_chi_compressed() {
        // Setup the Chi package.
        let chi = ChiPackage;

        let rng = &mut OsRng;

        // Generate random field elements as input.
        let input: [_; 5] = array::from_fn(|_| BinaryField128b::random(rng));

        // Compute the linear and quadratic components of the Chi package.
        let linear_part = chi.linear(&input);
        let quadratic_part = chi.quadratic(&input);

        // Combine the linear and quadratic components to obtain the final result.
        let lhs: [_; 5] = array::from_fn(|i| linear_part[i] + quadratic_part[i]);

        // Compute the chi compressed value.
        let rhs = chi_compressed(input);

        // Ensure that the computed value matches the expected value.
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn test_chi_algebraic() {
        // Setup the Chi package.
        let chi = ChiPackage;

        let rng = &mut OsRng;

        // Generate random field elements as input.
        let input_a: [_; 5] = array::from_fn(|_| BinaryField128b::random(rng));
        let input_b: [_; 5] = array::from_fn(|_| BinaryField128b::random(rng));

        // Compute the chi compressed for both inputs.
        let lhs_a = chi_compressed(input_a);
        let lhs_b = chi_compressed(input_b);

        // Compute the combined chi compressed value.
        let _lhs_ab: [_; 5] = chi_compressed(array::from_fn(|i| input_a[i] + input_b[i]));

        // Compute the interleaved input data, bit by bit.
        let data = input_a
            .iter()
            .flat_map(|x| (0..128).map(|i| BinaryField128b::from((x.into_inner() >> i) & 1 != 0)))
            .interleave(input_b.iter().flat_map(|x| {
                (0..128).map(|i| BinaryField128b::from((x.into_inner() >> i) & 1 != 0))
            }))
            .collect::<Vec<_>>();
        // Compute the algebraic components of the Chi package for the interleaved input data.
        let rhs = chi.algebraic(&data, 0, 2);

        // Ensure that the computed value matches the expected value.
        assert_eq!(rhs[0], lhs_a);
        assert_eq!(rhs[1], lhs_b);
    }

    #[test]
    fn test_chi_round() {
        // Setup the number of variables.
        const NUM_VARS: usize = 20;

        // Setup the switch parameter.
        const SWITCH: usize = 5;

        let rng = &mut OsRng;

        // Setup some random points
        let points = Points::random(NUM_VARS, rng);

        // Create 5 multilinear lagrangian polynomials with `2^NUM_VARS` coefficients each
        let polys: [MultilinearLagrangianPolynomial; 5] =
            array::from_fn(|_| MultilinearLagrangianPolynomial::random(1 << NUM_VARS, rng));

        // Compute chi witness
        let witness = chi_round_witness(&polys);

        // Evaluate the claim
        let evaluation_claims: [_; 5] = array::from_fn(|i| witness[i].evaluate_at(&points));

        // Setup the Chi package.
        let chi = ChiPackage;

        // Generate a random gamma for folding
        let gamma = Point::random(rng);

        let start = std::time::Instant::now();

        // Setup the prover builder
        let prover_builder = BoolCheckBuilder::<_, _, SWITCH, _>::new(
            chi.clone(),
            points.clone(),
            evaluation_claims,
            polys.clone(),
        );
        let mut prover = prover_builder.build(&gamma);

        // Initialize the claim
        let mut claim = UnivariatePolynomial::new(evaluation_claims.to_vec()).evaluate_at(&gamma);

        // Verify that the claim agrees with the prover
        assert_eq!(claim, prover.claim, "Claim does not match prover claim after initialization");

        // Setup an empty vector to store the challenges
        let mut challenges = Points::default();

        // Main loop
        for i in 0..NUM_VARS {
            // Compute the round polynomial
            let round_poly = prover.round_polynomial().coeffs(claim);

            // Generate a random challenge
            let challenge = Point::random(rng);

            // Validate the length of the round polynomial
            assert_eq!(round_poly.len(), 4, "Round polynomial length mismatch");

            // Update the claim
            claim = round_poly.evaluate_at(&challenge);

            // Bind the challenge
            prover.bind(&challenge);

            // Store the challenge
            challenges.push(challenge);

            // Verify the claim
            assert_eq!(claim, prover.claim, "Claim does not match prover claim after round {i}");
        }

        // Finalize the protocol and fetch the result
        let BoolCheckOutput { mut frob_evals, .. } = prover.finish();

        let end = std::time::Instant::now();

        // Check the length of the Frobenius evaluations
        assert_eq!(frob_evals.len(), 128 * 5, "Frobenius evaluations length mismatch");

        // Untwist the Frobenius evaluations
        frob_evals.as_mut_slice().chunks_mut(128).for_each(|chunk| {
            let mut tmp = Evaluations::from(chunk.to_vec());
            tmp.untwist();

            for (i, val) in tmp.iter().enumerate() {
                chunk[i] = *val;
            }
        });

        // Compute the expected coordinate evaluations
        let expected_coord_evals: Vec<_> = polys
            .iter()
            .flat_map(|poly| {
                (0..128).map(|i| {
                    MultilinearLagrangianPolynomial::new(
                        poly.iter()
                            .map(|value| BinaryField128b::from((value.into_inner() >> i) & 1 != 0))
                            .collect(),
                    )
                    .evaluate_at(&challenges)
                })
            })
            .collect();

        // Validate the Frobenius evaluations
        assert_eq!(frob_evals.0, expected_coord_evals, "Frobenius evaluations mismatch");

        // Trick for padding
        frob_evals.push(BinaryField128b::ZERO);

        // Compute the claimed evaluations and fold them
        let claimed_evaluations =
            UnivariatePolynomial::new(chi.algebraic(&frob_evals, 0, 1)[0].to_vec());
        let folded_claimed_evaluations = claimed_evaluations.evaluate_at(&gamma);

        // Validate the final claim
        assert_eq!(
            folded_claimed_evaluations * *(points.eq_eval(&challenges)),
            claim,
            "Final claim mismatch"
        );

        println!("Time elapsed: {} ms", (end - start).as_millis());
    }
}
