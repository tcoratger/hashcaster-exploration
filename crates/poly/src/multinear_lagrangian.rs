use hashcaster_field::binary_field::BinaryField128b;
use num_traits::{One, Zero};
use rayon::{
    iter::{
        IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator,
        ParallelIterator,
    },
    slice::ParallelSliceMut,
};
use std::ops::{Deref, DerefMut};

/// A structure representing a multilinear Lagrangian polynomial.
/// This structure holds the coefficients of the polynomial in a vector.
///
/// Each coefficient corresponds to the evaluation of the polynomial at a specific point
/// in the Boolean hypercube `{0,1}^n`. For example, for `n = 2`, the coefficients
/// represent the evaluations `p(0,0)`, `p(0,1)`, `p(1,0)`, `p(1,1)`, where `p(x)` is the
/// polynomial being constructed.
///
/// The multilinear polynomial is defined as:
///
/// ```text
/// p(X_1, X_2, ..., X_n) = sum_{x in {0,1}^n} p(x) * prod_{i=1}^n phi_{x_i}(X_i)
/// ```
///
/// where `phi_{x_i}(X_i)` is the basis polynomial that ensures the term evaluates
/// to `p(x)` only at `x` and vanishes otherwise.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MultilinearLagrangianPolynomial {
    /// Coefficients of the polynomial, stored as a vector of [`BinaryField128b`] elements.
    ///
    /// Each entry corresponds to the evaluation of the polynomial at a point in the Boolean
    /// hypercube `{0,1}^n`, in lexicographical order of the points. For example, for `n = 2`:
    /// - `coeffs[0]` corresponds to `p(0,0)`,
    /// - `coeffs[1]` corresponds to `p(0,1)`,
    /// - `coeffs[2]` corresponds to `p(1,0)`,
    /// - `coeffs[3]` corresponds to `p(1,1)`.
    pub coeffs: Vec<BinaryField128b>,
}

impl MultilinearLagrangianPolynomial {
    /// Creates a new [`MultilinearLagrangianPolynomial`] with the given coefficients.
    ///
    /// # Arguments
    /// * `coeffs` - A vector of [`BinaryField128b`] elements representing the polynomial's
    ///   coefficients.
    ///
    /// # Returns
    /// A new instance of [`MultilinearLagrangianPolynomial`].
    pub const fn new(coeffs: Vec<BinaryField128b>) -> Self {
        Self { coeffs }
    }

    /// Constructs the equality polynomial based on the provided points.
    ///
    /// # Arguments
    /// * `points` - A slice of `BinaryField128b` elements representing the input points.
    ///
    /// # Returns
    /// An instance of `MultilinearLagrangianPolynomial` containing the coefficients of the equality
    /// polynomial.
    ///
    /// # Explanation
    /// The equality polynomial is a multivariate polynomial constructed to encode the relationship
    /// between multiple input points. It ensures that the polynomial evaluates to `1` when the
    /// input `z` aligns with the combination of input points `points` and appropriately
    /// interpolates over the domain for multivariate inputs.
    ///
    /// ## Definition
    /// Given a set of points `points = [b_1, b_2, ..., b_m]` in the finite field, the equality
    /// polynomial `eq(z, points)` is defined iteratively as:
    ///
    /// ```text
    /// eq(z, points) = prod_{i=1}^m (z_i * b_i + (1 - z_i) * (1 - b_i)).
    /// ```
    /// Here, `z = (z_1, z_2, ..., z_m)` is the evaluation point, and `points` represent the
    /// multivariate configuration for which the polynomial encodes the behavior.
    ///
    /// ## Utility in Multivariate Context
    /// The equality polynomial extends to encode the behavior across multiple input points. It
    /// forms the basis for evaluating multilinear extensions in higher-dimensional spaces. For
    /// a multilinear extension `f_MLE` of a function `f`, we use:
    ///
    /// ```text
    /// f_MLE(z) = sum_{b in {0,1}^m} eq(z, b) * f(b),
    /// ```
    /// where `f(b)` is the value of the function at point `b`, and `eq(z, b)` ensures interpolation
    /// for the multivariate domain.
    ///
    /// ## Multivariate Example
    /// For `points = [pt1, pt2]`:
    /// ```text
    /// eq_poly(z) =
    ///   coeffs[0] * (1 - pt1) * (1 - pt2) +
    ///   coeffs[1] * pt1 * (1 - pt2) +
    ///   coeffs[2] * (1 - pt1) * pt2 +
    ///   coeffs[3] * pt1 * pt2.
    /// ```
    /// The coefficients `coeffs` are updated iteratively to encode this multivariate behavior. This
    /// ensures that the polynomial evaluates correctly for combinations of inputs.
    ///
    /// ## How This Implementation Works
    /// - The polynomial is constructed iteratively starting from a single coefficient (initialized
    ///   to `1`).
    /// - Coefficients are updated using the recurrence relation to encode the multivariate
    ///   relationships.
    /// - The result captures the equality polynomial over the domain, supporting evaluation in
    ///   multilinear extensions and related computations.
    pub fn new_eq_poly(points: &[BinaryField128b]) -> Self {
        // Initialize the coefficients with a single 1 (neutral element for multiplication).
        let mut coeffs = vec![BinaryField128b::one()];

        // Preallocate memory for all coefficients, filling with zeros beyond the initial size.
        coeffs.resize(1 << points.len(), BinaryField128b::zero());

        // Iterate over the points to construct the equality polynomial.
        for (i, point) in points.iter().enumerate() {
            // Split the coefficient vector into two parts: `left` and `right`.
            // `left` contains existing coefficients, `right` will store the new coefficients.
            let (left, right) = coeffs.split_at_mut(1 << i);

            // Update coefficients in parallel using iterators over `left` and `right`.
            left.par_iter_mut().zip(right.par_iter_mut()).for_each(|(left_val, right_val)| {
                // Compute the new coefficient in `right` as the product of `left_val` and the
                // current point.
                *right_val = *left_val * point;
                // Update the existing coefficient in `left` by adding the computed `right_val`.
                *left_val += *right_val;
            });
        }

        // Return the constructed equality polynomial.
        Self { coeffs }
    }
}

impl Deref for MultilinearLagrangianPolynomial {
    type Target = Vec<BinaryField128b>;

    fn deref(&self) -> &Self::Target {
        &self.coeffs
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MultilinearLagrangianPolynomials(Vec<MultilinearLagrangianPolynomial>);

impl MultilinearLagrangianPolynomials {
    /// Constructs the sequence of equality polynomials for a given set of points.
    ///
    /// # Arguments
    /// * `points` - A slice of `BinaryField128b` elements representing the input points.
    ///
    /// # Returns
    /// A [`MultilinearLagrangianPolynomials`] instance containing the sequence of equality
    /// polynomials.
    ///
    /// # Explanation
    /// This method generates a sequence of equality polynomials, where each polynomial encodes the
    /// relationships of subsets of the input points. The sequence starts with a base case
    /// polynomial `eq_0(x) = 1` and iteratively constructs each subsequent polynomial using
    /// a recurrence relation.
    ///
    /// ## Sequence Construction
    /// The equality polynomials in the sequence are defined recursively:
    /// - `eq_0(x) = 1` (base case),
    /// - `eq_k(x) = eq_{k-1}(x) * (1 + m_k * x)`, where `m_k` is the multiplier for the \(k\)-th
    ///   point.
    ///
    /// ## Utility
    /// This sequence is useful in efficiently constructing multilinear extensions, where each
    /// polynomial in the sequence serves as a building block for interpolating functions over
    /// the Boolean hypercube \( \{0, 1\}^n \).
    pub fn new_eq_poly_sequence(points: &[BinaryField128b]) -> Self {
        // Start with the base case: eq_0(x) = 1.
        let mut polynomials =
            vec![MultilinearLagrangianPolynomial::new(vec![BinaryField128b::one()])];

        // Iterate over the points in reverse order.
        for (i, &multiplier) in points.iter().rev().enumerate() {
            // Reference the previously computed polynomial in the sequence.
            let previous = &polynomials[i];

            // Allocate space for the new polynomial coefficients.
            // The new polynomial will have twice the size of the previous one.
            let mut new_coeffs = vec![BinaryField128b::zero(); 1 << (i + 1)];

            // Compute the new polynomial coefficients using the recurrence relation.
            new_coeffs.par_chunks_exact_mut(2).zip(previous.par_iter()).for_each(
                |(chunk, &prev_coeff)| {
                    // Calculate the updated coefficients.
                    let multiplied = multiplier * prev_coeff;
                    chunk[0] = prev_coeff + multiplied; // Update the first coefficient.
                    chunk[1] = multiplied; // Update the second coefficient.
                },
            );

            // Append the new polynomial to the list.
            polynomials.push(MultilinearLagrangianPolynomial::new(new_coeffs));
        }

        // Return the constructed sequence of equality polynomials.
        Self(polynomials)
    }
}

impl Deref for MultilinearLagrangianPolynomials {
    type Target = Vec<MultilinearLagrangianPolynomial>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for MultilinearLagrangianPolynomials {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::unreadable_literal)]
    fn test_eq_poly() {
        // Define a simple set of points.
        let points =
            vec![BinaryField128b::from(1), BinaryField128b::from(2), BinaryField128b::from(3)];

        // Compute the equality polynomial for the given points.
        let result = MultilinearLagrangianPolynomial::new_eq_poly(&points);

        // Assert that the computed equality polynomial matches the expected result.
        assert_eq!(
            result,
            MultilinearLagrangianPolynomial {
                coeffs: vec![
                    BinaryField128b::from(23667462636862719611022351736646926339),
                    BinaryField128b::from(79536576834697464894010492239055159303),
                    BinaryField128b::from(222989354442297682615051044822747971586),
                    BinaryField128b::from(106141905937829921662600755429978931204),
                    BinaryField128b::from(71478132110251412414531161936105570306),
                    BinaryField128b::from(294913050526722114925455479975104741381),
                    BinaryField128b::from(87449637247104542199890968644786585603),
                    BinaryField128b::from(225730887183604071602915146884576182279)
                ]
            }
        );
    }

    #[test]
    #[allow(clippy::unreadable_literal)]
    fn test_eq_poly_evaluate() {
        // Define a simple set of points.
        let pt1 = BinaryField128b::from(3434);
        let pt2 = BinaryField128b::from(6765);

        // Define the points in a vector.
        let points = vec![pt1, pt2];

        // Compute the equality polynomial for the given points.
        let result = MultilinearLagrangianPolynomial::new_eq_poly(&points);

        // Evaluate the equality polynomial at the given points.
        let expected_eq_poly =
            result[0] * (BinaryField128b::one() - pt1) * (BinaryField128b::one() - pt2) +
                result[2] * (BinaryField128b::one() - pt1) * pt2 +
                result[1] * (BinaryField128b::one() - pt2) * pt1 +
                result[3] * pt1 * pt2;

        // Verify that the equality polynomial evaluates to 1.
        // This ensures that the computed polynomial satisfies the expected equality conditions.
        assert_eq!(expected_eq_poly, BinaryField128b::one());
    }

    #[test]
    #[allow(clippy::unreadable_literal)]
    fn test_eq_poly_sequence() {
        // Define a vector of input points in the field BinaryField128b.
        let points = vec![
            BinaryField128b::from(1),
            BinaryField128b::from(2),
            BinaryField128b::from(3),
            BinaryField128b::from(4),
        ];

        // Compute the equality polynomial sequence using the eq_poly_sequence function.
        let eq_sequence = MultilinearLagrangianPolynomials::new_eq_poly_sequence(&points);

        // Assert that the computed equality polynomial sequence matches the expected result.
        // This ensures the function is working as intended.
        assert_eq!(
            eq_sequence,
            MultilinearLagrangianPolynomials(vec![
                MultilinearLagrangianPolynomial::new(vec![BinaryField128b::from(
                    257870231182273679343338569694386847745
                )]),
                MultilinearLagrangianPolynomial::new(vec![
                    BinaryField128b::from(257870231182273679343338569694386847749),
                    BinaryField128b::from(4)
                ]),
                MultilinearLagrangianPolynomial::new(vec![
                    BinaryField128b::from(276728653372472173290161332362114236431),
                    BinaryField128b::from(24175334173338157438437990908848766986),
                    BinaryField128b::from(24175334173338157438437990908848766989),
                    BinaryField128b::from(24175334173338157438437990908848766985)
                ]),
                MultilinearLagrangianPolynomial::new(vec![
                    BinaryField128b::from(262194116391218557050997340512544882712),
                    BinaryField128b::from(28499219382283035146096761727006801943),
                    BinaryField128b::from(36391510607255973141463116147421347857),
                    BinaryField128b::from(12382329933390930187138101121107623963),
                    BinaryField128b::from(318146307338789859576041968956220637212),
                    BinaryField128b::from(336838576029515239038751755741412982801),
                    BinaryField128b::from(323629372821402637551770173079877058582),
                    BinaryField128b::from(299454038648064480113332182171028291615)
                ]),
                MultilinearLagrangianPolynomial::new(vec![
                    BinaryField128b::from(156988152385753943545944842966784802826),
                    BinaryField128b::from(238399472904936139640581627536708993042),
                    BinaryField128b::from(103112231144489255218858846742755934222),
                    BinaryField128b::from(118147824772591482324176031488095027225),
                    BinaryField128b::from(214143658130290692373901413934757314572),
                    BinaryField128b::from(247871520449118298941928385465288753181),
                    BinaryField128b::from(114993504431031574539843754968093818891),
                    BinaryField128b::from(127368045919134702485539060344707612688),
                    BinaryField128b::from(138015824183221341818021192535835148297),
                    BinaryField128b::from(181502608447665156913871231917135757333),
                    BinaryField128b::from(311457361334396773361093192980979777548),
                    BinaryField128b::from(30864341024960054608398038725843484701),
                    BinaryField128b::from(197871324294196201012813621595002109966),
                    BinaryField128b::from(137763997785582402678037463463867973656),
                    BinaryField128b::from(30950013923125879264268791915275616264),
                    BinaryField128b::from(326990117386703710211842172749841694743)
                ])
            ])
        );
    }

    #[test]
    fn test_eq_poly_sequence_random_values() {
        use rand::Rng;

        // Define the number of iterations for the test.
        let iterations = 10;

        // Seed a random number generator.
        let mut rng = rand::thread_rng();

        for _ in 0..iterations {
            // Step 1: Generate random points for the test.
            let num_points = rng.gen_range(2..6); // Random number of points between 2 and 5.
            let points: Vec<BinaryField128b> =
                (0..num_points).map(|_| BinaryField128b::random()).collect();

            // Step 2: Compute the equality polynomial using `new_eq_poly`.
            let result = MultilinearLagrangianPolynomial::new_eq_poly(&points);

            // Step 3: Reconstruct the equality polynomial manually to verify correctness.
            let mut expected_eq_poly = BinaryField128b::zero();
            for i in 0..(1 << num_points) {
                // Convert `i` to binary representation to match the current point combination.
                let binary_combination: Vec<bool> =
                    (0..num_points).map(|j| (i & (1 << j)) != 0).collect();

                // Calculate the term for the current binary combination.
                let mut term = result[i];
                for (bit, point) in binary_combination.iter().zip(&points) {
                    term *= if *bit {
                        *point // Include the point if the bit is 1.
                    } else {
                        BinaryField128b::one() - *point // Complement if the bit is 0.
                    };
                }
                expected_eq_poly += term;
            }

            // Step 4: Assert that the computed equality polynomial evaluates to 1.
            assert_eq!(expected_eq_poly, BinaryField128b::one());
        }
    }

    #[test]
    fn test_eq_poly_sequence_cross_check() {
        // Generate a random vector of 20 points in the finite field.
        let points: Vec<BinaryField128b> = (0..20).map(|_| BinaryField128b::random()).collect();

        // Compute the equality polynomial sequence for the points.
        let eq_sequence = MultilinearLagrangianPolynomials::new_eq_poly_sequence(&points);

        // Verify the sequence length matches the expected size (points.len() + 1).
        assert_eq!(eq_sequence.len(), points.len() + 1);

        // Verify the initial polynomial is [1].
        assert_eq!(
            eq_sequence[0],
            MultilinearLagrangianPolynomial::new(vec![BinaryField128b::one()])
        );

        // Cross-check each polynomial in the sequence with its direct computation.
        eq_sequence.iter().enumerate().skip(1).for_each(|(i, poly)| {
            assert_eq!(
                poly,
                &MultilinearLagrangianPolynomial::new_eq_poly(&points[points.len() - i..])
            );
        });
    }
}
