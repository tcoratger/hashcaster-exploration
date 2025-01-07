use hashcaster_field::{
    array_ref, binary_field::BinaryField128b, matrix_efficient::EfficientMatrix,
};
use hashcaster_lincheck::prodcheck::ProdCheck;
use hashcaster_poly::{multinear_lagrangian::MultilinearLagrangianPolynomial, point::Points};
use num_traits::Zero;
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
use std::array;

pub mod builder;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MultiClaim<const N: usize> {
    pub polys: [Vec<BinaryField128b>; N],
    pub gamma: BinaryField128b,
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
    pub fn new(
        poly: MultilinearLagrangianPolynomial,
        points: &Points,
        openings: &[BinaryField128b],
        gamma_pows: &[BinaryField128b],
        polys: [Vec<BinaryField128b>; N],
    ) -> Self {
        // Compute the Frobenius linear combination matrix.
        // This matrix represents the operation \( M_\gamma = \sum \gamma_i \cdot Frob^{-i} \).
        let m = EfficientMatrix::from_frobenius_inv_lc(array_ref!(gamma_pows, 0, 128));

        // Generate the equality polynomial for the evaluation points
        // 1. Compute the Lagrange basis at the evaluation points
        // 2. Modify the polynomial `eq` to account for Frobenius twists.
        let mut eq = MultilinearLagrangianPolynomial::new_eq_poly(points);
        eq.par_iter_mut().for_each(|x| *x = m.apply(*x));

        // Compute the initial claim
        // This is computed as \( \text{Initial Claim} = \sum_{i=0}^{127} \gamma^i \cdot o_i \),
        // where \( o_i \) is the opening at Frobenius orbit point \( i \).
        let claim = gamma_pows[0..128]
            .iter()
            .zip(openings.iter())
            .map(|(x, y)| *x * y)
            .fold(BinaryField128b::zero(), |x, y| x + y);

        // Create a multiclaim object
        // - The object is initialized with:
        //  - The folded polynomial
        //  - The equality polynomial modified for Frobenius twists
        //  - The initial claim
        //  - No check of the claim
        // - The original polynomials
        // - Precomputed value of \( \gamma^{128} \).
        Self { object: ProdCheck::new([poly], [eq], claim, false), polys, gamma: gamma_pows[128] }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hashcaster_poly::point::Point;

    #[test]
    fn test_new_default_inputs() {
        // Create a default `MultilinearLagrangianPolynomial`
        let poly = MultilinearLagrangianPolynomial {
            coeffs: vec![BinaryField128b::from(1), BinaryField128b::from(2)],
        };

        // Create points for evaluation
        let points = Points::from(vec![Point::default()]);

        // Create openings (all zeros)
        let openings = vec![BinaryField128b::zero(); 128];

        // Create gamma powers (all zeros except the first one)
        let mut gamma_pows = vec![BinaryField128b::zero(); 129];
        gamma_pows[0] = BinaryField128b::from(1);

        // Create polynomials (empty for now)
        let polys: [Vec<BinaryField128b>; 2] = array::from_fn(|_| {
            vec![BinaryField128b::from(1), BinaryField128b::from(2), BinaryField128b::from(3)]
        });

        // Call the `new` function
        let claim = MultiClaim::new(poly, &points, &openings, &gamma_pows, polys.clone());

        // Validate the fields of the returned claim
        assert_eq!(claim.polys, polys, "Polynomials should match the input polys.");
        assert_eq!(
            claim.gamma,
            BinaryField128b::from(0),
            "Gamma should match the precomputed value of gamma^128."
        );
        assert_eq!(
            claim.object.claim,
            BinaryField128b::zero(),
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
        let polys: [Vec<BinaryField128b>; 2] = array::from_fn(|_| {
            vec![BinaryField128b::from(2), BinaryField128b::from(4), BinaryField128b::from(6)]
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
            BinaryField128b::from(128),
            "Gamma should match the precomputed value of gamma^128."
        );
        assert_eq!(
            claim.object.claim, expected_claim,
            "Initial claim should match the computed value."
        );
    }
}
