use crate::multinear_lagrangian::{
    MultilinearLagrangianPolynomial, MultilinearLagrangianPolynomials,
};
use hashcaster_primitives::binary_field::BinaryField128b;
use itertools::Itertools;
use rayon::{
    iter::{
        IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator,
        ParallelIterator,
    },
    slice::ParallelSliceMut,
};
use std::ops::{Deref, DerefMut};

/// A point represented as a field element in a binary field.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Point(pub BinaryField128b);

impl From<BinaryField128b> for Point {
    fn from(field_element: BinaryField128b) -> Self {
        Self(field_element)
    }
}

impl Deref for Point {
    type Target = BinaryField128b;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// A collection of points represented as field elements in a binary field.
///
/// # Description
/// The `Points` struct is a wrapper around a vector of `Point` elements, providing additional
/// functionality for operations commonly performed on collections of points in binary fields.
/// It is designed to work seamlessly with binary field arithmetic and operations such as
/// equality polynomial evaluations.
///
/// # Mathematical Representation
/// A collection of points can be represented as:
///
/// ```text
/// Points = {p_1, p_2, ..., p_n}
/// ```
///
/// Each point is a field element in the binary field, encapsulated in the `Point` struct.
///
/// # Use Cases
/// - Storing and manipulating collections of field elements.
/// - Performing operations on points, such as equality polynomial evaluations.
/// - Interfacing with mathematical constructs that require points.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Points(pub Vec<Point>);

impl Points {
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
    pub fn to_eq_poly(&self) -> MultilinearLagrangianPolynomial {
        // Initialize the coefficients with a single 1 (neutral element for multiplication).
        let mut coeffs = vec![BinaryField128b::ONE];

        // Preallocate memory for all coefficients, filling with zeros beyond the initial size.
        coeffs.resize(1 << self.len(), BinaryField128b::ZERO);

        // Iterate over the points to construct the equality polynomial.
        self.iter().enumerate().for_each(|(i, point)| {
            // Split the coefficient vector into two parts: `left` and `right`.
            // `left` contains existing coefficients, `right` will store the new coefficients.
            let (left, right) = coeffs.split_at_mut(1 << i);

            // Update coefficients in parallel using iterators over `left` and `right`.
            left.par_iter_mut().zip(right.par_iter_mut()).for_each(|(left_val, right_val)| {
                // Compute the new coefficient in `right` as the product of `left_val` and the
                // current point.
                *right_val = *left_val * **point;
                // Update the existing coefficient in `left` by adding the computed `right_val`.
                *left_val += *right_val;
            });
        });

        // Return the constructed equality polynomial.
        MultilinearLagrangianPolynomial { coeffs }
    }

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
    pub fn to_eq_poly_sequence(&self) -> MultilinearLagrangianPolynomials {
        // Start with the base case: eq_0(x) = 1.
        let mut polynomials =
            vec![MultilinearLagrangianPolynomial::new(vec![BinaryField128b::ONE])];

        // Iterate over the points in reverse order.
        for (i, multiplier) in self.iter().rev().enumerate() {
            // Reference the previously computed polynomial in the sequence.
            let previous = &polynomials[i];

            // Allocate space for the new polynomial coefficients.
            // The new polynomial will have twice the size of the previous one.
            let mut new_coeffs = vec![BinaryField128b::ZERO; 1 << (i + 1)];

            // Compute the new polynomial coefficients using the recurrence relation.
            new_coeffs.par_chunks_exact_mut(2).zip(previous.par_iter()).for_each(
                |(chunk, &prev_coeff)| {
                    // Calculate the updated coefficients.
                    let multiplied = **multiplier * prev_coeff;
                    chunk[0] = prev_coeff + multiplied; // Update the first coefficient.
                    chunk[1] = multiplied; // Update the second coefficient.
                },
            );

            // Append the new polynomial to the list.
            polynomials.push(MultilinearLagrangianPolynomial::new(new_coeffs));
        }

        // Return the constructed sequence of equality polynomials.
        MultilinearLagrangianPolynomials(polynomials)
    }

    /// Computes the evaluation of the equality polynomial for two sets of points.
    ///
    /// # Description
    /// The equality evaluation `eq_eval` is a function that measures how "close" two sets of points
    /// are in a binary field. It computes the product of terms that evaluate to `1` for
    /// matching points and decrease otherwise. This operation is commonly used to compare or verify
    /// consistency between two multivariate inputs.
    ///
    /// # Mathematical Definition
    /// For two sets of points `self = {x_1, x_2, ..., x_n}` and `other = {y_1, y_2, ..., y_n}`, the
    /// equality evaluation is defined as:
    ///
    /// ```text
    /// eq_eval(self, other) = Π_{i=1}^n (1 + x_i + y_i)
    /// ```
    ///
    /// Here:
    /// - `x_i` and `y_i` are elements of the binary field.
    /// - The term `(1 + x_i + y_i)` evaluates to `1` if `x_i = y_i`, and decreases otherwise.
    /// - The product ensures that all points must match for the result to remain `1`.
    ///
    /// # Parameters
    /// - `self`: A reference to the first set of points (`Points`).
    /// - `other`: A reference to the second set of points (`Points`).
    ///
    /// # Returns
    /// - A single `Point` representing the evaluation of the equality polynomial for the two sets
    ///   of points.
    ///
    /// # Notes
    /// - The input sets `self` and `other` must have the same length. Otherwise, the function will
    ///   panic.
    /// - The operation is defined for binary field elements and assumes valid binary field
    ///   arithmetic.
    pub fn eq_eval(&self, other: &Self) -> Point {
        self.iter()
            .zip_eq(other.iter())
            .fold(BinaryField128b::ONE, |acc, (x, y)| acc * (BinaryField128b::ONE + **x + **y))
            .into()
    }

    /// Computes the evaluation of the equality polynomial for a set of points and a slice of
    /// points.
    ///
    /// # Description
    /// The `eq_eval_slice` function evaluates the equality polynomial for `self` (a collection of
    /// points) and `other` (a slice of points). It measures the similarity between the two sets
    /// in terms of binary field values, returning a product of terms that assess point-by-point
    /// equality.
    ///
    /// # Mathematical Definition
    /// For `self = {x_1, x_2, ..., x_n}` and `other = {y_1, y_2, ..., y_m}`, where `m <= n`,
    /// the evaluation is computed as:
    ///
    /// ```text
    /// eq_eval_slice(self, other) = Π_{i=1}^m (1 + x_i + y_i)
    /// ```
    ///
    /// # Parameters
    /// - `self`: A reference to the set of points (`Points`).
    /// - `other`: A slice of points (`&[Point]`).
    ///
    /// # Returns
    /// - A `Point` representing the product of equality terms for the given sets of points.
    ///
    /// # Panics
    /// - Panics if the length of `other` exceeds the length of `self`.
    pub fn eq_eval_slice(&self, other: &[Point]) -> Point {
        self.iter()
            .zip_eq(other.iter())
            .fold(BinaryField128b::ONE, |acc, (x, y)| acc * (BinaryField128b::ONE + **x + **y))
            .into()
    }
}

impl From<Vec<BinaryField128b>> for Points {
    fn from(points: Vec<BinaryField128b>) -> Self {
        Self(points.into_iter().map(Point).collect())
    }
}

impl From<Vec<Point>> for Points {
    fn from(points: Vec<Point>) -> Self {
        Self(points)
    }
}

impl From<&[Point]> for Points {
    fn from(points: &[Point]) -> Self {
        Self(points.to_vec())
    }
}

impl Deref for Points {
    type Target = Vec<Point>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Points {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl FromIterator<Point> for Points {
    fn from_iter<T: IntoIterator<Item = Point>>(iter: T) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl FromIterator<BinaryField128b> for Points {
    fn from_iter<T: IntoIterator<Item = BinaryField128b>>(iter: T) -> Self {
        Self(iter.into_iter().map(Point::from).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hashcaster_primitives::binary_field::BinaryField128b;

    #[test]
    fn test_eq_eval() {
        // Define two sets of points in the binary field.
        let points_a = Points::from(vec![
            BinaryField128b::from(1),
            BinaryField128b::from(2),
            BinaryField128b::from(3),
        ]);

        let points_b = Points::from(vec![
            BinaryField128b::from(4),
            BinaryField128b::from(5),
            BinaryField128b::from(6),
        ]);

        // Perform the eq_eval operation.
        let result = points_a.eq_eval(&points_b);

        // Manually compute the expected result.
        let expected = BinaryField128b::ONE *
            (BinaryField128b::ONE + BinaryField128b::from(1) + BinaryField128b::from(4)) *
            (BinaryField128b::ONE + BinaryField128b::from(2) + BinaryField128b::from(5)) *
            (BinaryField128b::ONE + BinaryField128b::from(3) + BinaryField128b::from(6));

        // Assert that the computed result matches the expected result.
        assert_eq!(result, Point::from(expected));
    }

    #[test]
    fn test_eq_eval_identity() {
        // Define identical sets of points in the binary field.
        let points = Points::from(vec![
            BinaryField128b::from(7),
            BinaryField128b::from(8),
            BinaryField128b::from(9),
        ]);

        // Perform the eq_eval operation with identical sets.
        let result = points.eq_eval(&points);

        // Assert that the result is consistent with identical inputs.
        //
        // The expected output is One since the operation is commutative and associative.
        assert_eq!(result, Point::from(BinaryField128b::ONE));
    }

    #[test]
    fn test_eq_eval_empty() {
        // Define two empty sets of points.
        let points_a = Points::default();
        let points_b = Points::default();

        // Perform the eq_eval operation.
        let result = points_a.eq_eval(&points_b);

        // The expected result for empty inputs is BinaryField128b::ONE.
        assert_eq!(result, Point::from(BinaryField128b::ONE));
    }

    #[test]
    fn test_eq_eval_slice_basic() {
        // Define two sets of points in the binary field.
        let points_a = Points::from(vec![
            BinaryField128b::from(1),
            BinaryField128b::from(2),
            BinaryField128b::from(3),
        ]);

        let points_b = vec![
            Point::from(BinaryField128b::from(4)),
            Point::from(BinaryField128b::from(5)),
            Point::from(BinaryField128b::from(6)),
        ];

        // Perform the eq_eval_slice operation.
        let result = points_a.eq_eval_slice(&points_b);

        // Manually compute the expected result.
        let expected = BinaryField128b::ONE *
            (BinaryField128b::ONE + BinaryField128b::from(1) + BinaryField128b::from(4)) *
            (BinaryField128b::ONE + BinaryField128b::from(2) + BinaryField128b::from(5)) *
            (BinaryField128b::ONE + BinaryField128b::from(3) + BinaryField128b::from(6));

        // Assert that the computed result matches the expected result.
        assert_eq!(result, Point::from(expected));
    }

    #[test]
    fn test_eq_eval_slice_partial() {
        // Define a larger set of points in the binary field.
        let points_a = Points::from(vec![BinaryField128b::from(1), BinaryField128b::from(2)]);

        // Define a smaller slice of points.
        let points_b = [
            Point::from(BinaryField128b::from(3)),
            Point::from(BinaryField128b::from(4)),
            Point::from(BinaryField128b::from(5)),
            Point::from(BinaryField128b::from(6)),
        ];

        // Perform the eq_eval_slice operation.
        let result = points_a.eq_eval_slice(&points_b[..2]);

        // Manually compute the expected result.
        let expected = BinaryField128b::ONE *
            (BinaryField128b::ONE + BinaryField128b::from(1) + BinaryField128b::from(3)) *
            (BinaryField128b::ONE + BinaryField128b::from(2) + BinaryField128b::from(4));

        // Assert that the computed result matches the expected result.
        assert_eq!(result, Point::from(expected));
    }

    #[test]
    fn test_eq_eval_slice_identity() {
        // Define a set of points in the binary field.
        let points_a = Points::from(vec![BinaryField128b::from(7), BinaryField128b::from(8)]);

        // Define an identical slice of points.
        let points_b =
            vec![Point::from(BinaryField128b::from(7)), Point::from(BinaryField128b::from(8))];

        // Perform the eq_eval_slice operation.
        let result = points_a.eq_eval_slice(&points_b);

        // Assert that the result is consistent with identical inputs.
        assert_eq!(result, Point::from(BinaryField128b::ONE));
    }

    #[test]
    fn test_eq_eval_slice_empty_slice() {
        // Define a non-empty set of points in the binary field.
        let points_a = Points::default();

        // Define an empty slice of points.
        let points_b: Vec<Point> = vec![
            Point(BinaryField128b::from(1)),
            Point(BinaryField128b::from(2)),
            Point(BinaryField128b::from(3)),
        ];

        // Perform the eq_eval_slice operation.
        let result = points_a.eq_eval_slice(&points_b[..0]);

        // The expected result for an empty slice is BinaryField128b::ONE.
        assert_eq!(result, Point::from(BinaryField128b::ONE));
    }

    #[test]
    #[should_panic = "itertools: .zip_eq() reached end of one iterator before the other"]
    fn test_eq_eval_slice_mismatched_lengths() {
        // Define a larger set of points.
        let points_a = Points::from(vec![BinaryField128b::from(1), BinaryField128b::from(2)]);

        // Define a longer slice of points.
        let points_b = vec![
            Point::from(BinaryField128b::from(3)),
            Point::from(BinaryField128b::from(4)),
            Point::from(BinaryField128b::from(5)),
        ];

        // Attempt to perform the eq_eval_slice operation (should panic).
        points_a.eq_eval_slice(&points_b);
    }

    #[test]
    fn test_eq_poly_sequence_cross_check() {
        // Generate a random vector of 20 points in the finite field.
        let points: Points = (0..20).map(|_| BinaryField128b::random()).collect();

        // Compute the equality polynomial sequence for the points.
        let eq_sequence = points.to_eq_poly_sequence();

        // Verify the sequence length matches the expected size (points.len() + 1).
        assert_eq!(eq_sequence.len(), points.len() + 1);

        // Verify the initial polynomial is [1].
        assert_eq!(
            eq_sequence[0],
            MultilinearLagrangianPolynomial::new(vec![BinaryField128b::ONE])
        );

        // Cross-check each polynomial in the sequence with its direct computation.
        eq_sequence.iter().enumerate().skip(1).for_each(|(i, poly)| {
            let pts: Points = points[points.len() - i..].into();
            assert_eq!(*poly, pts.to_eq_poly());
        });
    }
}
