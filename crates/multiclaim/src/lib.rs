#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

use hashcaster_lincheck::prodcheck::ProdCheck;
use hashcaster_poly::{
    compressed::CompressedPoly,
    multinear_lagrangian::MultilinearLagrangianPolynomial,
    point::{Point, Points},
    univariate::UnivariatePolynomial,
};
use hashcaster_primitives::{
    array_ref, binary_field::BinaryField128b, matrix_efficient::EfficientMatrix,
};
use num_traits::MulAdd;
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
use std::array;

pub mod builder;

/// Represents a multi-claim verification object.
///
/// This struct encapsulates data and operations for handling multilinear
/// lagrangian polynomials and their corresponding claims over a series of rounds.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MultiClaim<const N: usize> {
    /// The array of multilinear lagrangian polynomials used in the proof.
    pub polys: [MultilinearLagrangianPolynomial; N],

    /// The gamma value used for Frobenius transformations.
    pub gamma: Point,

    /// The underlying product check object that manages the folding and evaluations.
    pub object: ProdCheck<1>,
}

impl<const N: usize> Default for MultiClaim<N> {
    fn default() -> Self {
        Self {
            polys: array::from_fn(|_| Default::default()),
            gamma: Default::default(),
            object: Default::default(),
        }
    }
}

impl<const N: usize> MultiClaim<N> {
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
        polys: [MultilinearLagrangianPolynomial; N],
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
            gamma: gamma_pows.get(128).map_or_else(Default::default, |g| (*g).into()),
        }
    }

    /// Computes the compressed polynomial for the current round of the proof.
    ///
    /// # Returns
    /// The compressed polynomial after applying folding operations.
    pub fn compute_round_polynomial(&mut self) -> CompressedPoly {
        self.object.compute_round_polynomial()
    }

    /// Binds a challenge point to the `MultiClaim`.
    ///
    /// This updates the state of the underlying product check with the provided challenge point.
    ///
    /// # Parameters
    /// - `challenge`: The challenge point used for folding operations.
    pub fn bind(&mut self, challenge: &Point) {
        self.object.bind(challenge);
    }

    /// Finalizes the `MultiClaim` by computing the univariate polynomial of folded openings.
    ///
    /// # Returns
    /// A univariate polynomial representing the folded openings, adjusted for gamma powers.
    ///
    /// # Process
    /// - Initializes the first opening to zero.
    /// - Evaluates subsequent openings using the challenges derived during the sumcheck process.
    /// - Adjusts the first opening with the univariate evaluation at the gamma value.
    pub fn finish(&self) -> UnivariatePolynomial {
        // Compute the folded openings.
        // - The first opening is initialized to zero,
        // - Subsequent openings are evaluated for each polynomial at the challenges derived during
        //   the sumcheck.
        let mut ret = UnivariatePolynomial::new(
            std::iter::once(BinaryField128b::ZERO)
                .chain((1..N).map(|i| self.polys[i].evaluate_at(&self.object.challenges)))
                .collect(),
        );

        // Adjust the first opening:
        // - Use a univariate evaluation to account for gamma powers.
        // - Update the first opening with the adjusted value.
        ret[0] = ret.evaluate_at(&self.gamma) + self.object.p_polys[0][0];

        // Return the resulting openings.
        ret
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use builder::MulticlaimBuilder;
    use hashcaster_poly::{evaluation::Evaluations, point::Point};
    use num_traits::MulAdd;

    #[test]
    fn test_new_default_inputs() {
        // Create a default `MultilinearLagrangianPolynomial`
        let poly = MultilinearLagrangianPolynomial {
            coeffs: vec![BinaryField128b::from(1), BinaryField128b::from(2)],
        };

        // Create points for evaluation
        let points = Points::from(vec![Point::default()]);

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
        let claim = MultiClaim::new(poly, &points, &openings, &gamma_pows, polys.clone());

        // Validate the fields of the returned claim
        assert_eq!(claim.polys, polys, "Polynomials should match the input polys.");
        assert_eq!(
            claim.gamma,
            Point(BinaryField128b::from(0)),
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
        let points = Points::from(vec![Point::default()]);

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
        let claim = MultiClaim::new(poly, &points, &openings, &gamma_pows, polys.clone());

        // Compute expected initial claim
        let mut expected_claim = BinaryField128b::from(0);
        for i in 0..128 {
            expected_claim += BinaryField128b::from(i) * BinaryField128b::from(5);
        }

        // Validate the fields of the returned claim
        assert_eq!(claim.polys, polys, "Polynomials should match the input polys.");
        assert_eq!(
            claim.gamma,
            Point(BinaryField128b::from(128)),
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

        // Create a multilinear polynomial with `2^NUM_VARS` coefficients
        let poly = MultilinearLagrangianPolynomial::new(
            (0..1 << NUM_VARS).map(|_| BinaryField128b::random()).collect(),
        );

        // Create points for evaluation
        let points: Points = (0..NUM_VARS).map(|_| Point(BinaryField128b::random())).collect();

        // Map the points to the inverse Frobenius orbit
        let points_inv_orbit: Vec<Points> =
            (0..128).map(|i| points.iter().map(|x| Point(x.frobenius(-i))).collect()).collect();

        // Evaluate the polynomial at the points on the inverse Frobenius orbit
        let evaluations_inv_orbit: Evaluations =
            (0..128).map(|i| poly.evaluate_at(&points_inv_orbit[i])).collect();

        // Setup a multiclaim builder
        let prover_builder =
            MulticlaimBuilder::new([poly.clone()], points, evaluations_inv_orbit.clone());

        // Generate a random gamma for folding
        let gamma = Point(BinaryField128b::random());

        // Builder the prover via folding
        let mut prover = prover_builder.build(&gamma);

        // Compute powers of gamma for future checks
        let gamma_pows: [_; 128] = BinaryField128b::compute_gammas_folding(*gamma);

        // Compute the claim
        let mut claim = gamma_pows
            .iter()
            .zip(evaluations_inv_orbit.iter())
            .fold(BinaryField128b::ZERO, |acc, (x, y)| x.mul_add(*y, acc));

        // Setup an empty vector to store the challanges in the main loop
        let mut challenges = Points::from(Vec::<Point>::with_capacity(NUM_VARS));

        // Principal loop of the prover
        for _ in 0..NUM_VARS {
            // Compute the round polynomial
            let round_polynomial = prover.compute_round_polynomial().coeffs(claim);

            // Check that the round polynomial is of degree 2
            assert_eq!(round_polynomial.len(), 3, "Round polynomial should have degree 2.");

            // Random challenge
            let challenge = Point(BinaryField128b::random());

            // Update the claim with the round polynomial and the challenge
            claim = round_polynomial[0] +
                round_polynomial[1] * *challenge +
                round_polynomial[2] * *challenge * *challenge;

            // Push the challenge to the vector
            challenges.push(challenge.clone());

            // Bind the prover to the challenge
            prover.bind(&challenge);
        }

        // Compute the equality evaluations at the challenges
        let eq_evaluations = gamma_pows
            .iter()
            .zip(points_inv_orbit.iter())
            .fold(BinaryField128b::ZERO, |acc, (gamma, pts)| {
                gamma.mul_add(*pts.eq_eval(&challenges), acc)
            });

        // Verify the final claim
        assert_eq!(
            poly.evaluate_at(&challenges) * eq_evaluations,
            claim,
            "Claim should match the product of the polynomial evaluations and the equality evaluations."
        );
    }
}
