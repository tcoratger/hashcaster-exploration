use hashcaster_field::binary_field::BinaryField128b;
use num_traits::{MulAddAssign, Zero};
use std::ops::Deref;

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
}

impl Deref for UnivariatePolynomial {
    type Target = Vec<BinaryField128b>;

    fn deref(&self) -> &Self::Target {
        &self.coeffs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
