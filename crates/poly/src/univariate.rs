use crate::point::Point;
use hashcaster_field::binary_field::BinaryField128b;
use num_traits::{MulAddAssign, Zero};
use std::ops::{Deref, DerefMut, Mul, MulAssign};

/// Represents a univariate polynomial with coefficients over a binary field.
///
/// # Fields
/// - `coeffs`: A vector of `BinaryField128b` elements representing the coefficients of the
///   polynomial. The coefficients are stored in increasing order of degree, where the 0th index
///   corresponds to the constant term.
///
/// # Example
/// ```rust,ignore
/// use hashcaster_field::binary_field::BinaryField128b;
/// use univariate_polynomial::UnivariatePolynomial;
///
/// let coeffs = vec![BinaryField128b::from(1), BinaryField128b::from(2), BinaryField128b::from(3)];
/// let poly = UnivariatePolynomial::new(coeffs);
/// let at = BinaryField128b::from(2);
/// let result = poly.evaluate_at(&at);
/// println!("Result: {:?}", result);
/// ```
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct UnivariatePolynomial {
    /// The coefficients of the polynomial, stored in increasing order of degree.
    pub coeffs: Vec<BinaryField128b>,
}

impl UnivariatePolynomial {
    /// Creates a new polynomial from a vector of coefficients.
    ///
    /// # Parameters
    /// - `coeffs`: A vector of `BinaryField128b` coefficients, with the constant term at index 0.
    ///
    /// # Returns
    /// A `UnivariatePolynomial` instance representing the polynomial with the given coefficients.
    pub const fn new(coeffs: Vec<BinaryField128b>) -> Self {
        Self { coeffs }
    }

    /// Evaluates the polynomial at a given point using Horner's method.
    ///
    /// # Parameters
    /// - `at`: The point at which to evaluate the polynomial, represented as a `BinaryField128b`.
    ///
    /// # Returns
    /// The result of evaluating the polynomial at the given point, as a `BinaryField128b`.
    ///
    /// # Methodology
    /// This method employs Horner's method for efficient polynomial evaluation:
    /// `P(x) = (((a_n \cdot x + a_{n-1}) \cdot x + a_{n-2}) \cdot x + ... + a_0)`
    pub fn evaluate_at(&self, at: &BinaryField128b) -> BinaryField128b {
        // Start with an accumulator initialized to zero.
        self.coeffs.iter().rev().fold(BinaryField128b::zero(), |mut acc, &coeff| {
            // Multiply the accumulator by `at` and add the current coefficient.
            acc.mul_add_assign(at, coeff);
            // Return the updated accumulator for the next iteration.
            acc
        })
    }

    /// Constructs a univariate polynomial from its evaluations at three points:
    /// - t = 0, t = 1, and t = ∞.
    ///
    /// The polynomial is assumed to be of degree 2:
    /// `P(t) = c_0 + c_1 * t + c_2 * t^2`.
    ///
    /// We have:
    /// - `W(0) = P(0) = c_0`,
    /// - `W(1) = P(1) = c_0 + c_1 + c_2`,
    /// - `W(∞) = P(∞) = c_2`.
    ///
    /// # Parameters
    /// - `evaluations`: An array of three evaluations:
    ///   - `evaluations[0]` is the evaluation at t = 0 (W(0) = c_0).
    ///   - `evaluations[1]` is the evaluation at t = 1 (W(1) = c_0 + c_1 + c_2).
    ///   - `evaluations[2]` is the evaluation at t = ∞ (W(∞) = c_2).
    ///
    /// # Returns
    /// A `UnivariatePolynomial` instance with coefficients `[c_0, c_1, c_2]`, where:
    /// - `c_0 = W(0)` (constant coefficient),
    /// - `c_1 = W(1) - W(0) - W(∞)` (linear coefficient),
    /// - `c_2 = W(∞)` (quadratic coefficient).
    pub fn from_evaluations_deg2(evaluations: [BinaryField128b; 3]) -> Self {
        Self {
            coeffs: vec![
                evaluations[0],
                evaluations[1] - evaluations[0] - evaluations[2],
                evaluations[2],
            ],
        }
    }

    /// Multiplies a degree-2 polynomial (`self`) with a degree-1 polynomial (`other`).
    ///
    /// This is not a general-purpose multiplication method, but a specific for multiple reasons:
    /// - The protocol is expected to deal explicitly with degree-2 and degree-1 polynomials.
    /// - A general-purpose multiplication method would be more complex and can be implemented in
    ///   the future.
    ///
    /// # Parameters
    /// - `other`: A reference to another `UnivariatePolynomial` of degree 1.
    ///
    /// # Returns
    /// A new `UnivariatePolynomial` of degree 3 that represents the product of the two input
    /// polynomials.
    ///
    /// # Assumptions
    /// - `self` must be a degree-2 polynomial (length 3).
    /// - `other` must be a degree-1 polynomial (length 2).
    ///
    /// # Formula
    /// Given:
    /// - `self` = \( c_0 + c_1 \cdot x + c_2 \cdot x^2 \)
    /// - `other` = \( d_0 + d_1 \cdot x \),
    ///
    /// The result is:
    /// \[
    /// \text{result} = (c_0 + c_1 \cdot x + c_2 \cdot x^2) \cdot (d_0 + d_1 \cdot x)
    /// \]
    ///
    /// Expanding:
    /// \[
    /// \text{result} = c_0 \cdot d_0
    ///              + (c_0 \cdot d_1 + c_1 \cdot d_0) \cdot x
    ///              + (c_1 \cdot d_1 + c_2 \cdot d_0) \cdot x^2
    ///              + c_2 \cdot d_1 \cdot x^3
    /// \]
    ///
    /// # Panics
    /// - If `self` is not a degree-2 polynomial.
    /// - If `other` is not a degree-1 polynomial.
    pub fn multiply_degree2_by_degree1(&self, other: &Self) -> Self {
        // Ensure the input polynomials are of the expected degrees.
        assert_eq!(self.len(), 3, "Expected lhs to be a degree 2 polynomial");
        assert_eq!(other.len(), 2, "Expected rhs to be a degree 1 polynomial");

        // Compute the coefficients of the resulting degree-3 polynomial.
        Self {
            coeffs: vec![
                // Degree 0 coefficient: c0 * d0
                other[0] * self[0],
                // Degree 1 coefficient: c0 * d1 + c1 * d0
                other[0] * self[1] + other[1] * self[0],
                // Degree 2 coefficient: c1 * d1 + c2 * d0
                other[0] * self[2] + other[1] * self[1],
                // Degree 3 coefficient: c2 * d1
                other[1] * self[2],
            ],
        }
    }
}

impl Mul<Point> for UnivariatePolynomial {
    type Output = Self;

    fn mul(self, point: Point) -> Self::Output {
        self.iter().map(|&c| c * *point).collect::<Vec<_>>().into()
    }
}

impl Mul<&Point> for UnivariatePolynomial {
    type Output = Self;

    fn mul(self, point: &Point) -> Self::Output {
        self.iter().map(|&c| c * **point).collect::<Vec<_>>().into()
    }
}

impl MulAssign<Point> for UnivariatePolynomial {
    fn mul_assign(&mut self, point: Point) {
        for c in self.iter_mut() {
            *c *= *point;
        }
    }
}

impl MulAssign<&Point> for UnivariatePolynomial {
    fn mul_assign(&mut self, point: &Point) {
        for c in self.iter_mut() {
            *c *= **point;
        }
    }
}

impl From<Vec<BinaryField128b>> for UnivariatePolynomial {
    fn from(coeffs: Vec<BinaryField128b>) -> Self {
        Self::new(coeffs)
    }
}

impl Deref for UnivariatePolynomial {
    type Target = Vec<BinaryField128b>;

    fn deref(&self) -> &Self::Target {
        &self.coeffs
    }
}

impl DerefMut for UnivariatePolynomial {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.coeffs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::One;

    #[test]
    fn test_univariate_polynomial_evaluate_at() {
        // Define the coefficients for the polynomial P(x) = 3 + 2x + x^2
        let coeffs = vec![
            BinaryField128b::from(3), // a_0
            BinaryField128b::from(2), // a_1
            BinaryField128b::from(1), // a_2
        ];

        // Create the polynomial
        let poly = UnivariatePolynomial::new(coeffs);

        // Define the point to evaluate the polynomial at
        let x = BinaryField128b::from(2);

        // Manually compute the expected value
        // P(2) = 3 + 2 * 2 + 1 * (2^2)
        let expected = BinaryField128b::from(3) +
            BinaryField128b::from(2) * BinaryField128b::from(2) +
            BinaryField128b::from(1) * BinaryField128b::from(2) * BinaryField128b::from(2);

        // Evaluate the polynomial at x
        let result = poly.evaluate_at(&x);

        // Assert that the result matches the expected value
        assert_eq!(result, expected, "Wrong evaluation result");
    }

    #[test]
    fn test_from_evaluations_deg2() {
        // Define evaluations at t = 0, t = 1, and t = ∞.
        let evals = [
            BinaryField128b::from(3), // W(0) = c_0
            BinaryField128b::from(9), // W(1) = c_0 + c_1 + c_2
            BinaryField128b::from(5), // W(∞) = c_2
        ];

        // Construct the polynomial.
        let poly = UnivariatePolynomial::from_evaluations_deg2(evals);

        // Expected coefficients:
        // c_0 = W(0) = 3
        // c_1 = W(1) - W(0) - W(∞) = 9 - 3 - 5 = 1
        // c_2 = W(∞) = 5
        let expected_coeffs = vec![
            BinaryField128b::from(3),
            BinaryField128b::from(9) - BinaryField128b::from(3) - BinaryField128b::from(5),
            BinaryField128b::from(5),
        ];

        // Assert that the coefficients match the expected values.
        assert_eq!(poly.coeffs, expected_coeffs, "Incorrect polynomial coefficients.");

        // Verify evaluation of the polynomial matches the input evaluations.
        let at_0 = BinaryField128b::zero();
        let at_1 = BinaryField128b::one();
        assert_eq!(poly.evaluate_at(&at_0), evals[0], "Evaluation at t = 0 failed.");
        assert_eq!(poly.evaluate_at(&at_1), evals[1], "Evaluation at t = 1 failed.");
    }

    #[test]
    fn test_mul_univariate_poly_by_point() {
        // Define the coefficients for the polynomial P(x) = 3 + 2x + x^2
        let coeffs = vec![
            BinaryField128b::from(3), // a_0
            BinaryField128b::from(2), // a_1
            BinaryField128b::from(1), // a_2
        ];
        let poly = UnivariatePolynomial::new(coeffs);

        // Define the multiplier point
        let point = Point::from(BinaryField128b::from(2));

        // Multiply the polynomial by the point
        let result_poly = poly.clone() * point;

        // Expected coefficients after multiplication
        let expected_coeffs = vec![
            BinaryField128b::from(3) * BinaryField128b::from(2),
            BinaryField128b::from(2) * BinaryField128b::from(2),
            BinaryField128b::from(1) * BinaryField128b::from(2),
        ];

        // Assert that the result matches the expected coefficients
        assert_eq!(result_poly.coeffs, expected_coeffs, "Multiplication by point failed.");
    }

    #[test]
    fn test_mul_univariate_poly_by_reference_to_point() {
        // Define the coefficients for the polynomial P(x) = 5 + 4x + 3x^2
        let coeffs = vec![
            BinaryField128b::from(5), // a_0
            BinaryField128b::from(4), // a_1
            BinaryField128b::from(3), // a_2
        ];
        let poly = UnivariatePolynomial::new(coeffs);

        // Define the multiplier point as a reference
        let point = Point::from(BinaryField128b::from(3));

        // Multiply the polynomial by a reference to the point
        let result_poly = poly.clone() * &point;

        // Expected coefficients after multiplication
        let expected_coeffs = vec![
            BinaryField128b::from(5) * BinaryField128b::from(3),
            BinaryField128b::from(4) * BinaryField128b::from(3),
            BinaryField128b::from(3) * BinaryField128b::from(3),
        ];

        // Assert that the result matches the expected coefficients
        assert_eq!(
            result_poly.coeffs, expected_coeffs,
            "Multiplication by reference to point failed."
        );
    }

    #[test]
    fn test_mul_assign_univariate_poly_by_point() {
        // Define the coefficients for the polynomial P(x) = 7 + 6x + 2x^2
        let coeffs = vec![
            BinaryField128b::from(7), // a_0
            BinaryField128b::from(6), // a_1
            BinaryField128b::from(2), // a_2
        ];
        let mut poly = UnivariatePolynomial::new(coeffs);

        // Define the multiplier point
        let point = Point::from(BinaryField128b::from(4));

        // Multiply the polynomial by the point in place
        poly *= point;

        // Expected coefficients after multiplication
        let expected_coeffs = vec![
            BinaryField128b::from(7) * BinaryField128b::from(4),
            BinaryField128b::from(6) * BinaryField128b::from(4),
            BinaryField128b::from(2) * BinaryField128b::from(4),
        ];

        // Assert that the result matches the expected coefficients
        assert_eq!(poly.coeffs, expected_coeffs, "In-place multiplication by point failed.");
    }

    #[test]
    fn test_mul_assign_univariate_poly_by_reference_to_point() {
        // Define the coefficients for the polynomial P(x) = 10 + 9x + 8x^2
        let coeffs = vec![
            BinaryField128b::from(10), // a_0
            BinaryField128b::from(9),  // a_1
            BinaryField128b::from(8),  // a_2
        ];
        let mut poly = UnivariatePolynomial::new(coeffs);

        // Define the multiplier point as a reference
        let point = Point::from(BinaryField128b::from(5));

        // Multiply the polynomial by a reference to the point in place
        poly *= &point;

        // Expected coefficients after multiplication
        let expected_coeffs = vec![
            BinaryField128b::from(10) * BinaryField128b::from(5),
            BinaryField128b::from(9) * BinaryField128b::from(5),
            BinaryField128b::from(8) * BinaryField128b::from(5),
        ];

        // Assert that the result matches the expected coefficients
        assert_eq!(
            poly.coeffs, expected_coeffs,
            "In-place multiplication by reference to point failed."
        );
    }

    #[test]
    fn test_multiply_degree2_by_degree1() {
        // Define a degree-2 polynomial: P(x) = 3 + 2x + x^2
        let c0 = BinaryField128b::from(3);
        let c1 = BinaryField128b::from(2);
        let c2 = BinaryField128b::from(1);
        let poly_deg2 = UnivariatePolynomial::new(vec![c0, c1, c2]);

        // Define a degree-1 polynomial: Q(x) = 4 + 5x
        let d0 = BinaryField128b::from(4);
        let d1 = BinaryField128b::from(5);
        let poly_deg1 = UnivariatePolynomial::new(vec![d0, d1]);

        // Multiply the polynomials
        let result = poly_deg2.multiply_degree2_by_degree1(&poly_deg1);

        // Expected coefficients of the result:
        // R(x) = (3 + 2x + x^2) * (4 + 5x)
        let expected_poly = UnivariatePolynomial::new(vec![
            c0 * d0,           // c0 * d0
            c0 * d1 + c1 * d0, // c0 * d1 + c1 * d0
            c1 * d1 + c2 * d0, // c1 * d1 + c2 * d0
            c2 * d1,           // c2 * d1
        ]);

        // Assert that the result matches the expected polynomial
        assert_eq!(
            result, expected_poly,
            "Multiplication of degree-2 and degree-1 polynomials failed."
        );
    }
}