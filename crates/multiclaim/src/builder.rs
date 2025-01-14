use crate::MultiClaim;
use hashcaster_primitives::{
    binary_field::BinaryField128b,
    poly::{
        evaluation::Evaluations,
        multinear_lagrangian::MultilinearLagrangianPolynomial,
        point::{Point, Points},
    },
    sumcheck::SumcheckBuilder,
};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::array;

/// A builder for creating a `MultiClaim`.
///
/// # Type Parameters
/// - `N`: The number of polynomials being processed.
///
/// The builder helps manage and validate inputs like polynomials, evaluation points, and openings.
/// It supports creating a `MultiClaim` by combining the inputs using a gamma parameter.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MulticlaimBuilder<const N: usize> {
    /// Multilinear Lagrangian polynomials for the claim.
    pub polys: [MultilinearLagrangianPolynomial; N],
    /// Evaluation points for the claim, represented as a `Points` collection.
    pub points: Points,
    /// Openings for the polynomials, represented as a flat vector of `BinaryField128b`.
    pub openings: Evaluations,
}

impl<const N: usize> Default for MulticlaimBuilder<N> {
    fn default() -> Self {
        Self {
            polys: array::from_fn(|_| Default::default()),
            points: Default::default(),
            openings: Default::default(),
        }
    }
}

impl<const N: usize> MulticlaimBuilder<N> {
    /// Creates a new `MulticlaimBuilder` with the provided inputs.
    ///
    /// # Parameters
    /// - `polys`: An array of `N` polynomials, each represented as a vector of `BinaryField128b`.
    /// - `points`: Evaluation points for the claim.
    /// - `openings`: A vector of polynomial openings, with a length of `N * 128`.
    ///
    /// # Panics
    /// - If the number of openings is not `N * 128`.
    /// - If the length of any polynomial does not match `2^number_of_points`.
    pub fn new(
        polys: [MultilinearLagrangianPolynomial; N],
        points: Points,
        openings: Evaluations,
    ) -> Self {
        // Check that the number of openings is correct.
        assert_eq!(openings.len(), N * 128, "Invalid number of openings");

        // Check that the length of each polynomial is 2^{number of points}.
        let expected_poly_len = 1 << points.len();
        assert!(
            polys.iter().all(|poly| poly.len() == expected_poly_len),
            "Invalid polynomial length"
        );

        Self { polys, points, openings }
    }
}

impl<const N: usize> SumcheckBuilder for MulticlaimBuilder<N>
where
    [(); 128 * N]:,
{
    type Sumcheck = MultiClaim<N>;

    fn build(&mut self, gamma: &Point) -> Self::Sumcheck {
        // Compute the powers of gamma for the folding process.
        let gamma_pows: [_; 128 * N] = BinaryField128b::compute_gammas_folding(**gamma);

        // Determine the number of coefficients in the polynomials (2^number_of_points).
        let l = 1 << self.points.len();

        // Combine all polynomials into a single composite polynomial.
        let poly: Vec<_> = (0..l)
            .into_par_iter()
            .map(|i| {
                // Fold contributions from all polynomials for the current index.
                (0..N)
                    .fold(BinaryField128b::ZERO, |p, j| p + self.polys[j][i] * gamma_pows[128 * j])
            })
            .collect();

        // Combine all openings into a single composite opening.
        let openings: Vec<_> = (0..128)
            .map(|i| {
                // Fold contributions from all openings for the current Frobenius point.
                (0..N).fold(BinaryField128b::ZERO, |o, j| {
                    o + self.openings[i + j * 128] * gamma_pows[128 * j]
                })
            })
            .collect();

        // Construct and return the `MultiClaim`.
        MultiClaim::new(poly.into(), &self.points, &openings, &gamma_pows, self.polys.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hashcaster_primitives::{
        binary_field::BinaryField128b,
        poly::point::{Point, Points},
    };

    #[test]
    fn test_multiclaim_builder_default() {
        // Create a default MulticlaimBuilder instance with N = 3.
        let builder: MulticlaimBuilder<3> = MulticlaimBuilder::default();

        // Verify that the default polys array is empty.
        for poly in &builder.polys {
            assert!(poly.is_empty());
        }

        // Verify that the default points collection is empty.
        assert!(builder.points.is_empty());

        // Verify that the default openings vector is empty.
        assert!(builder.openings.is_empty());
    }

    #[test]
    fn test_multiclaim_builder_new_valid() {
        // Define valid polynomials for N = 2.
        let polys: [MultilinearLagrangianPolynomial; 2] = [
            vec![BinaryField128b::from(1), BinaryField128b::from(2)].into(),
            vec![BinaryField128b::from(3), BinaryField128b::from(4)].into(),
        ];

        // Define valid points.
        let points = Points::from(vec![Point::from(1)]);

        // Define valid openings for N = 2 with 128 * N elements.
        let openings: Evaluations = vec![BinaryField128b::from(0); 2 * 128].into();

        // Create a new MulticlaimBuilder instance.
        let builder = MulticlaimBuilder::new(polys.clone(), points.clone(), openings.clone());

        // Verify that the builder contains the expected fields.
        assert_eq!(builder, MulticlaimBuilder { polys, points, openings });
    }

    #[test]
    #[should_panic(expected = "Invalid number of openings")]
    fn test_multiclaim_builder_new_invalid_openings() {
        // Define valid polynomials for N = 2.
        let polys: [MultilinearLagrangianPolynomial; 2] = [
            vec![BinaryField128b::from(1), BinaryField128b::from(2)].into(),
            vec![BinaryField128b::from(3), BinaryField128b::from(4)].into(),
        ];

        // Define valid points.
        let points = Points::from(vec![Point::from(1)]);

        // Define an invalid number of openings (not N * 128).
        let openings = vec![BinaryField128b::from(0); 100];

        // Attempt to create a new MulticlaimBuilder instance (should panic).
        MulticlaimBuilder::new(polys, points, openings.into());
    }

    #[test]
    #[should_panic(expected = "Invalid polynomial length")]
    fn test_multiclaim_builder_new_invalid_polynomial_length() {
        // Define invalid polynomials with incorrect lengths for N = 2.
        let polys: [MultilinearLagrangianPolynomial; 2] = [
            vec![BinaryField128b::from(1)].into(), // Length is 1, should be 2^number of points.
            vec![BinaryField128b::from(3), BinaryField128b::from(4)].into(),
        ];

        // Define valid points (1 point, so polynomials should have length 2).
        let points = Points::from(vec![Point::from(1)]);

        // Define valid openings for N = 2 with 128 * N elements.
        let openings = vec![BinaryField128b::from(0); 2 * 128];

        // Attempt to create a new MulticlaimBuilder instance (should panic).
        MulticlaimBuilder::new(polys, points, openings.into());
    }

    #[test]
    fn test_multiclaim_builder_new_edge_case() {
        // Define valid polynomials for N = 1.
        let polys: [MultilinearLagrangianPolynomial; 1] = [vec![BinaryField128b::from(42)].into()];

        // Define an empty points collection (0 points, so polynomials should have length 1).
        let points = Points::default();

        // Define valid openings for N = 1 with 128 * N elements.
        let openings: Evaluations = vec![BinaryField128b::from(0); 128].into();

        // Create a new MulticlaimBuilder instance.
        let builder = MulticlaimBuilder::new(polys.clone(), points.clone(), openings.clone());

        // Verify that the builder contains the expected fields.
        assert_eq!(builder, MulticlaimBuilder { polys, points, openings });
    }

    #[test]
    fn test_multiclaim_builder_build_simple_case() {
        // Define polynomials for N = 2
        let p11 = BinaryField128b::from(1);
        let p12 = BinaryField128b::from(2);
        let p21 = BinaryField128b::from(3);
        let p22 = BinaryField128b::from(4);
        let polys: [MultilinearLagrangianPolynomial; 2] = [
            vec![p11, p12].into(), // Polynomial 1
            vec![p21, p22].into(), // Polynomial 2
        ];

        // Define points (1 point, so polynomials should have length 2)
        // First point in the point set
        let p1 = Point::from(1);
        // Wrap the point into the Points structure
        let points = Points::from(vec![p1]);

        // Define openings for N = 2 with 128 * N elements
        let mut openings = vec![];
        for i in 0..2 * 128 {
            // Populate openings with incremental values
            openings.push(BinaryField128b::from(i));
        }

        // Create a new MulticlaimBuilder instance
        let mut builder = MulticlaimBuilder::new(polys, points, openings.clone().into());

        // Define gamma (random point for testing)
        let gamma = Point::from(BinaryField128b::from(2));

        // Build the MultiClaim
        let claim = builder.build(&gamma);

        // Compute gamma powers for validation
        let gamma_pows: [_; 128 * 2] = BinaryField128b::compute_gammas_folding((**gamma).into());

        // Compute the expected folded polynomial
        let poly = vec![
            p11 + p21 * gamma_pows[128], // Folded polynomial coefficient 1
            p12 + p22 * gamma_pows[128], // Folded polynomial coefficient 2
        ];

        // Assert the folded polynomials match
        assert_eq!(claim.object.p_polys, [poly.into()]);

        // Compute the expected claim value manually
        let mut expected_claim = BinaryField128b::ZERO;
        for i in 0..128 {
            expected_claim += (openings[i] + openings[i + 128] * gamma_pows[128]) * gamma_pows[i];
        }

        // Assert the computed claim matches the expected value
        assert_eq!(claim.object.claim, expected_claim);
    }
}
