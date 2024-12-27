use std::ops::Deref;

use hashcaster_field::binary_field::BinaryField128b;
use num_traits::identities::Zero;

use crate::univariate::UnivariatePolynomial;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CompressedPoly(Vec<BinaryField128b>);

impl CompressedPoly {
    pub const fn new(coeffs: Vec<BinaryField128b>) -> Self {
        Self(coeffs)
    }

    pub fn compress(poly: &[BinaryField128b]) -> (Self, BinaryField128b) {
        let sum = poly.iter().skip(1).fold(BinaryField128b::zero(), |a, b| a + b);
        (Self(std::iter::once(&poly[0]).chain(&poly[2..]).copied().collect()), sum)
    }

    /// Reconstructs the full polynomial from its compressed form and the provided sum.
    ///
    /// # Description
    /// The `coeffs` function takes a compressed polynomial and the sum of its coefficients
    /// evaluated at `x = 1`. It reconstructs the original polynomial coefficients
    /// by leveraging the compressed coefficients and the sum.
    ///
    /// # Parameters
    /// - `self`: A reference to the `CompressedPoly` object containing the compressed coefficients.
    /// - `sum`: The value of the polynomial evaluated at `x = 1`, i.e., `P(1) = ∑c_i`.
    ///
    /// # Returns
    /// A `Vec<BinaryField128b>` representing the reconstructed polynomial coefficients.
    ///
    /// # Example
    /// Given the polynomial:
    /// ```
    /// P(x) = 3 + 4x + 5x ^ 2 + 6x ^ 3
    /// ```
    /// - Compressed coefficients: `[3, 5, 6]` (skipping `c1 = 4`).
    /// - Sum at `x = 1`: `P(1) = 3 + 4 + 5 + 6 = 18`.
    ///
    /// The function reconstructs the coefficients as:
    /// ```
    /// [3, 4, 5, 6]
    /// ```
    pub fn coeffs(&self, sum: BinaryField128b) -> UnivariatePolynomial {
        // Step 1: Extract the constant term `c0`.
        // Example: For P(x) = 3 + 4x + 5x^2 + 6x^3, c0 = 3.
        let c0 = self[0];

        // Step 2: Compute the value of the polynomial at `x = 1`, `ev_1`.
        // `ev_1 = c0 + sum`.
        // Example: sum = 18, c0 = 3 => ev_1 = 3 + 18 = 21.
        let ev_1 = c0 + sum;

        // Step 3: Compute the missing first coefficient `c1`.
        // `c1 = ev_1 - ∑(c_i for i ≠ 1)`.
        // This is done by summing all stored coefficients and adding `ev_1` to adjust.
        // Example: Compressed coefficients = [3, 5, 6].
        // Sum of stored coefficients = 3 + 5 + 6 = 14.
        // c1 = ev_1 - (sum of stored coefficients excluding c1) = 21 - 14 = 7.
        let c1 = self.iter().fold(BinaryField128b::zero(), |a, b| a + *b) + ev_1;

        // Step 4: Combine all coefficients:
        // - Start with `c0`,
        // - Add `c1`,
        // - Append the remaining coefficients from the compressed form.
        // Example: Result = [3, 7, 5, 6].
        std::iter::once(c0)
            .chain(std::iter::once(c1))
            .chain(self[1..].iter().copied())
            .collect::<Vec<_>>()
            .into()
    }
}

impl Deref for CompressedPoly {
    type Target = Vec<BinaryField128b>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hashcaster_field::binary_field::BinaryField128b;

    #[test]
    fn test_compress() {
        let poly = [
            BinaryField128b::new(1),
            BinaryField128b::new(2),
            BinaryField128b::new(3),
            BinaryField128b::new(4),
        ];

        // Compress the polynomial
        let (compressed, sum) = CompressedPoly::compress(&poly);

        let expected_compressed = CompressedPoly(vec![
            BinaryField128b::new(1),
            BinaryField128b::new(3),
            BinaryField128b::new(4),
        ]);

        let expected_sum =
            BinaryField128b::new(2) + BinaryField128b::new(3) + BinaryField128b::new(4);

        // Verify the compressed polynomial and the sum
        assert_eq!(
            compressed, expected_compressed,
            "Compressed polynomial does not match the expected result"
        );
        assert_eq!(sum, expected_sum, "Sum of coefficients does not match the expected result");
    }

    #[test]
    fn test_compress_single_element() {
        let poly = [BinaryField128b::new(42)];
        let result = std::panic::catch_unwind(|| CompressedPoly::compress(&poly));
        assert!(result.is_err(), "Expected compress to panic for a single-element polynomial");
    }

    #[test]
    fn test_compress_empty() {
        let poly: [BinaryField128b; 0] = [];
        let result = std::panic::catch_unwind(|| CompressedPoly::compress(&poly));
        assert!(result.is_err(), "Expected compress to panic for an empty polynomial");
    }

    #[test]
    fn test_sum_standard_case() {
        let poly = [
            BinaryField128b::new(1),
            BinaryField128b::new(2),
            BinaryField128b::new(3),
            BinaryField128b::new(4),
        ];

        let (_, sum) = CompressedPoly::compress(&poly);

        let expected_sum =
            BinaryField128b::new(2) + BinaryField128b::new(3) + BinaryField128b::new(4);

        assert_eq!(sum, expected_sum, "Sum of coefficients does not match the expected result");
    }

    #[test]
    fn test_sum_single_coefficient() {
        let poly = [BinaryField128b::new(42)];
        let result = std::panic::catch_unwind(|| CompressedPoly::compress(&poly));
        assert!(result.is_err(), "Expected compress to panic for a single-element polynomial");
    }

    #[test]
    fn test_sum_all_zero_coefficients() {
        let poly = [BinaryField128b::new(0), BinaryField128b::new(0), BinaryField128b::new(0)];

        let (_, sum) = CompressedPoly::compress(&poly);

        assert_eq!(sum, BinaryField128b::zero(), "Sum of zero coefficients should be zero");
    }

    #[test]
    fn test_sum_empty_coefficients() {
        let poly: [BinaryField128b; 0] = [];
        let result = std::panic::catch_unwind(|| CompressedPoly::compress(&poly));
        assert!(result.is_err(), "Expected compress to panic for an empty polynomial");
    }

    #[test]
    fn test_coeffs_reconstruction_standard_case() {
        // Define a univariate polynomial:
        // P(x) = 1 + 2x + 3x^2 + 4x^3
        let poly = vec![
            BinaryField128b::new(1),
            BinaryField128b::new(2),
            BinaryField128b::new(3),
            BinaryField128b::new(4),
        ];

        // Compress the polynomial
        let (compressed, sum) = CompressedPoly::compress(&poly);

        // Reconstruct the coefficients
        let reconstructed = compressed.coeffs(sum);

        // Verify the reconstructed polynomial matches the original
        assert_eq!(
            reconstructed,
            UnivariatePolynomial::new(poly),
            "Reconstructed polynomial does not match the original"
        );
    }

    #[test]
    fn test_coeffs_all_zero_coefficients() {
        // Define a univariate polynomial:
        // P(x) = 0 + 0x + 0x^2 + 0x^3
        let poly = vec![
            BinaryField128b::new(0),
            BinaryField128b::new(0),
            BinaryField128b::new(0),
            BinaryField128b::new(0),
        ];

        // Compress the polynomial
        let (compressed, sum) = CompressedPoly::compress(&poly);

        // Reconstruct the coefficients
        let reconstructed = compressed.coeffs(sum);

        // Verify the reconstructed polynomial matches the original
        assert_eq!(
            reconstructed,
            UnivariatePolynomial::new(poly),
            "Reconstructed polynomial does not match the original when all coefficients are zero"
        );
    }

    #[test]
    fn test_coeffs_large_coefficients() {
        // Define a univariate polynomial:
        // P(x) = 1_000_000 + 2_000_000x + 3_000_000x^2 + 4_000_000x^3
        let poly = vec![
            BinaryField128b::new(1_000_000),
            BinaryField128b::new(2_000_000),
            BinaryField128b::new(3_000_000),
            BinaryField128b::new(4_000_000),
        ];

        // Compress the polynomial
        let (compressed, sum) = CompressedPoly::compress(&poly);

        // Reconstruct the coefficients
        let reconstructed = compressed.coeffs(sum);

        // Verify the reconstructed polynomial matches the original
        assert_eq!(
            reconstructed,
            UnivariatePolynomial::new(poly),
            "Reconstructed polynomial does not match the original for large coefficients"
        );
    }

    #[test]
    fn test_coeffs_single_coefficient() {
        // Define a univariate polynomial:
        // P(x) = 42
        let poly = [BinaryField128b::new(42)];

        // Compress the polynomial (expect a panic)
        let result = std::panic::catch_unwind(|| {
            let (compressed, sum) = CompressedPoly::compress(&poly);
            compressed.coeffs(sum)
        });

        // Verify that the function panicked
        assert!(result.is_err(), "Expected coeffs to panic for a single-element polynomial");
    }

    #[test]
    fn test_coeffs_empty_coefficients() {
        // Define an empty univariate polynomial
        let poly: [BinaryField128b; 0] = [];

        // Compress the polynomial (expect a panic)
        let result = std::panic::catch_unwind(|| {
            let (compressed, sum) = CompressedPoly::compress(&poly);
            compressed.coeffs(sum)
        });

        // Verify that the function panicked
        assert!(result.is_err(), "Expected coeffs to panic for an empty polynomial");
    }
}
