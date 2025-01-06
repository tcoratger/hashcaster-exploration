use std::ops::{Deref, DerefMut};

use hashcaster_field::binary_field::BinaryField128b;
use itertools::Itertools;
use num_traits::One;

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
pub struct Points(Vec<Point>);

impl Points {
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
            .fold(BinaryField128b::one(), |acc, (x, y)| acc * (BinaryField128b::one() + **x + **y))
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
            .fold(BinaryField128b::one(), |acc, (x, y)| acc * (BinaryField128b::one() + **x + **y))
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

#[cfg(test)]
mod tests {
    use super::*;
    use hashcaster_field::binary_field::BinaryField128b;

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
        let expected = BinaryField128b::one() *
            (BinaryField128b::one() + BinaryField128b::from(1) + BinaryField128b::from(4)) *
            (BinaryField128b::one() + BinaryField128b::from(2) + BinaryField128b::from(5)) *
            (BinaryField128b::one() + BinaryField128b::from(3) + BinaryField128b::from(6));

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
        assert_eq!(result, Point::from(BinaryField128b::one()));
    }

    #[test]
    fn test_eq_eval_empty() {
        // Define two empty sets of points.
        let points_a = Points::default();
        let points_b = Points::default();

        // Perform the eq_eval operation.
        let result = points_a.eq_eval(&points_b);

        // The expected result for empty inputs is BinaryField128b::one().
        assert_eq!(result, Point::from(BinaryField128b::one()));
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
        let expected = BinaryField128b::one() *
            (BinaryField128b::one() + BinaryField128b::from(1) + BinaryField128b::from(4)) *
            (BinaryField128b::one() + BinaryField128b::from(2) + BinaryField128b::from(5)) *
            (BinaryField128b::one() + BinaryField128b::from(3) + BinaryField128b::from(6));

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
        let expected = BinaryField128b::one() *
            (BinaryField128b::one() + BinaryField128b::from(1) + BinaryField128b::from(3)) *
            (BinaryField128b::one() + BinaryField128b::from(2) + BinaryField128b::from(4));

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
        assert_eq!(result, Point::from(BinaryField128b::one()));
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

        // The expected result for an empty slice is BinaryField128b::one().
        assert_eq!(result, Point::from(BinaryField128b::one()));
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
}
