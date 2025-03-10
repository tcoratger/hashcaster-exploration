#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use hashcaster_lincheck::prodcheck::ProdCheck;
use hashcaster_primitives::{
    array_ref,
    binary_field::BinaryField128b,
    matrix_efficient::EfficientMatrix,
    poly::{
        compressed::CompressedPoly, evaluation::FixedEvaluations,
        multinear_lagrangian::MultilinearLagrangianPolynomial, point::Points,
        univariate::FixedUnivariatePolynomial,
    },
    sumcheck::Sumcheck,
};
use num_traits::MulAdd;
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};

pub mod builder;

/// Represents a multi-claim verification object.
///
/// This struct encapsulates data and operations for handling multilinear
/// lagrangian polynomials and their corresponding claims over a series of rounds.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MultiClaim<'a, const N: usize> {
    /// The array of multilinear lagrangian polynomials used in the proof.
    pub polys: &'a [MultilinearLagrangianPolynomial; N],

    /// The gamma value used for Frobenius transformations.
    pub gamma: BinaryField128b,

    /// The underlying product check object that manages the folding and evaluations.
    pub object: ProdCheck<1>,
}

impl<const N: usize> Sumcheck<N, 2> for MultiClaim<'_, N> {
    type Output = FixedEvaluations<N>;

    fn round_polynomial(&mut self) -> CompressedPoly<2> {
        self.object.round_polynomial()
    }

    fn bind(&mut self, challenge: &BinaryField128b) {
        self.object.bind(challenge);
    }

    fn finish(self) -> Self::Output {
        // Compute the folded openings.
        // - The first opening is initialized to zero.
        // - Subsequent openings are evaluated for each polynomial at the challenges derived during
        //   the sumcheck.

        // Initialize an array with all elements set to zero.
        let mut coeffs = [BinaryField128b::ZERO; N];

        // Compute and store the subsequent openings (starting from index 1).
        for (i, coeff) in coeffs.iter_mut().enumerate().take(N).skip(1) {
            *coeff = self.polys[i].evaluate_at(&self.object.challenges);
        }

        // Construct the fixed-size univariate polynomial with preallocated coefficients.
        let mut ret = FixedUnivariatePolynomial::new(coeffs);

        // Adjust the first opening:
        // - Use a univariate evaluation to account for gamma powers.
        // - Update the first opening with the adjusted value.
        ret.coeffs[0] = ret.evaluate_at(&self.gamma) + self.object.p_polys[0][0];

        // Return the resulting openings.
        FixedEvaluations::new(ret.coeffs)
    }
}

impl<'a, const N: usize> MultiClaim<'a, N> {
    /// Creates a new instance of `MultiClaim` with the provided parameters.
    ///
    /// # Parameters
    /// - `poly`: The initial multilinear lagrangian polynomial.
    /// - `points`: Evaluation points for the polynomial.
    /// - `openings`: Precomputed opening values at Frobenius orbit points.
    /// - `gamma_pows`: Powers of the gamma value for Frobenius transformations.
    /// - `polys`: Array of polynomials involved in the claim.
    ///
    /// # Returns
    /// A new `MultiClaim` instance with precomputed equality polynomials and the initial claim.
    pub fn new(
        poly: MultilinearLagrangianPolynomial,
        points: &Points,
        openings: &[BinaryField128b],
        gamma_pows: &[BinaryField128b],
        polys: &'a [MultilinearLagrangianPolynomial; N],
    ) -> Self {
        // Compute the Frobenius linear combination matrix.
        // This matrix represents the operation \( M_\gamma = \sum \gamma_i \cdot Frob^{-i} \).
        let m = EfficientMatrix::from_frobenius_inv_lc(array_ref!(gamma_pows, 0, 128));

        // Generate the equality polynomial for the evaluation points
        // 1. Compute the Lagrange basis at the evaluation points
        // 2. Modify the polynomial `eq` to account for Frobenius twists.
        let mut eq = points.to_eq_poly();
        eq.par_iter_mut().for_each(|x| *x = m.apply(*x));

        // Compute the initial claim
        // This is computed as \( \text{Initial Claim} = \sum_{i=0}^{127} \gamma^i \cdot o_i \),
        // where \( o_i \) is the opening at Frobenius orbit point \( i \).
        let claim = gamma_pows[0..128]
            .iter()
            .zip(openings.iter())
            .fold(BinaryField128b::ZERO, |acc, (x, y)| x.mul_add(*y, acc));

        // Create a multiclaim object
        // - The object is initialized with:
        //  - The folded polynomial
        //  - The equality polynomial modified for Frobenius twists
        //  - The initial claim
        //  - No check of the claim
        // - The original polynomials
        // - Precomputed value of \( \gamma^{128} \).
        Self {
            object: ProdCheck::new([poly], [eq], claim, false),
            polys,
            gamma: gamma_pows.get(128).map_or_else(Default::default, |g| (*g)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use builder::MulticlaimBuilder;
    use hashcaster_primitives::sumcheck::SumcheckBuilder;
    use num_traits::MulAdd;
    use rand::rngs::OsRng;
    use std::array;

    #[test]
    fn test_new_default_inputs() {
        // Create a default `MultilinearLagrangianPolynomial`
        let poly = MultilinearLagrangianPolynomial {
            coeffs: vec![BinaryField128b::from(1), BinaryField128b::from(2)],
        };

        // Create points for evaluation
        let points = Points::from(vec![BinaryField128b::default()]);

        // Create openings (all zeros)
        let openings = vec![BinaryField128b::ZERO; 128];

        // Create gamma powers (all zeros except the first one)
        let mut gamma_pows = vec![BinaryField128b::ZERO; 129];
        gamma_pows[0] = BinaryField128b::from(1);

        // Create polynomials (empty for now)
        let polys: [MultilinearLagrangianPolynomial; 2] = array::from_fn(|_| {
            vec![BinaryField128b::from(1), BinaryField128b::from(2), BinaryField128b::from(3)]
                .into()
        });

        // Call the `new` function
        let claim = MultiClaim::new(poly, &points, &openings, &gamma_pows, &polys);

        // Validate the fields of the returned claim
        assert_eq!(claim.polys, &polys, "Polynomials should match the input polys.");
        assert_eq!(
            claim.gamma,
            BinaryField128b::from(0),
            "Gamma should match the precomputed value of gamma^128."
        );
        assert_eq!(
            claim.object.claim,
            BinaryField128b::ZERO,
            "Initial claim should be zero when all openings are zero."
        );
    }

    #[test]
    fn test_new_with_nonzero_openings() {
        // Create a non-default `MultilinearLagrangianPolynomial`
        let poly = MultilinearLagrangianPolynomial {
            coeffs: vec![BinaryField128b::from(1), BinaryField128b::from(3)],
        };

        // Create points for evaluation
        let points = Points::from(vec![BinaryField128b::default()]);

        // Create openings with some non-zero values
        let openings = vec![BinaryField128b::from(5); 128];

        // Create gamma powers (non-zero values)
        let gamma_pows: Vec<_> = (0..129).map(BinaryField128b::from).collect();

        // Create polynomials
        let polys: [MultilinearLagrangianPolynomial; 2] = array::from_fn(|_| {
            vec![BinaryField128b::from(2), BinaryField128b::from(4), BinaryField128b::from(6)]
                .into()
        });

        // Call the `new` function
        let claim = MultiClaim::new(poly, &points, &openings, &gamma_pows, &polys);

        // Compute expected initial claim
        let mut expected_claim = BinaryField128b::from(0);
        for i in 0..128 {
            expected_claim += BinaryField128b::from(i) * BinaryField128b::from(5);
        }

        // Validate the fields of the returned claim
        assert_eq!(claim.polys, &polys, "Polynomials should match the input polys.");
        assert_eq!(
            claim.gamma,
            BinaryField128b::from(128),
            "Gamma should match the precomputed value of gamma^128."
        );
        assert_eq!(
            claim.object.claim, expected_claim,
            "Initial claim should match the computed value."
        );
    }

    #[test]
    fn test_multiclaim_complete() {
        // Number of variables
        const NUM_VARS: usize = 20;

        let rng = &mut OsRng;

        // Create a multilinear polynomial with `2^NUM_VARS` coefficients
        let poly = MultilinearLagrangianPolynomial::random(1 << NUM_VARS, rng);

        // Create points for evaluation
        let points = Points::random(NUM_VARS, rng);

        // Map the points to the inverse Frobenius orbit
        let points_inv_orbit: Vec<Points> =
            (0..128).map(|i| points.iter().map(|x| x.frobenius(-i)).collect()).collect();

        // Evaluate the polynomial at the points on the inverse Frobenius orbit
        let evaluations_inv_orbit = FixedEvaluations::<128>::new(std::array::from_fn(|i| {
            poly.evaluate_at(&points_inv_orbit[i])
        }));

        // Setup a multiclaim builder
        let polys = [poly.clone()];
        let prover_builder = MulticlaimBuilder::new(&polys, &points, &evaluations_inv_orbit);

        // Generate a random gamma for folding
        let gamma = BinaryField128b::random(rng);

        // Builder the prover via folding
        let mut prover = prover_builder.build(&gamma);

        // Compute powers of gamma for future checks
        let gamma_pows: [_; 128] = BinaryField128b::compute_gammas_folding(&gamma);

        // Compute the claim
        let mut claim = gamma_pows
            .iter()
            .zip(evaluations_inv_orbit.iter())
            .fold(BinaryField128b::ZERO, |acc, (x, y)| x.mul_add(*y, acc));

        // Setup an empty vector to store the challanges in the main loop
        let mut challenges = Points::from(Vec::<BinaryField128b>::with_capacity(NUM_VARS));

        // Principal loop of the prover
        for _ in 0..NUM_VARS {
            // Compute the round polynomial
            let round_polynomial = prover.round_polynomial().coeffs(claim);

            // Check that the round polynomial is of degree 2
            assert_eq!(round_polynomial.len(), 3, "Round polynomial should have degree 2.");

            // Random challenge
            let challenge = BinaryField128b::random(rng);

            // Update the claim with the round polynomial and the challenge
            claim = round_polynomial.evaluate_at(&challenge);

            // Push the challenge to the vector
            challenges.push(challenge);

            // Bind the prover to the challenge
            prover.bind(&challenge);
        }

        // Compute the equality evaluations at the challenges
        let eq_evaluations = gamma_pows
            .iter()
            .zip(points_inv_orbit.iter())
            .fold(BinaryField128b::ZERO, |acc, (gamma, pts)| {
                gamma.mul_add(pts.eq_eval(&challenges), acc)
            });

        // Verify the final claim
        assert_eq!(
            poly.evaluate_at(&challenges) * eq_evaluations,
            claim,
            "Claim should match the product of the polynomial evaluations and the equality evaluations."
        );
    }
}
