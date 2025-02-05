use crate::binary_field::BinaryField128b;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::ops::Deref;

use super::univariate::FixedUnivariatePolynomial;

/// A compressed representation of a univariate polynomial in a binary field.
///
/// # Description
/// The `CompressedPoly` struct represents a polynomial with compressed coefficients,
/// skipping certain terms for optimization. It allows efficient storage and reconstruction
/// of the original polynomial given the sum of its coefficients evaluated at `x = 1`.
///
/// # Fields
/// - `0`: A `Vec<BinaryField128b>` containing the compressed coefficients. The first coefficient is
///   stored explicitly, followed by selected other coefficients. Some coefficients (e.g., `c1`) are
///   omitted to reduce storage.
///
/// # Example
/// Given the polynomial:
/// ```text
/// P(x) = 3 + 4x + 5x^2 + 6x^3
/// ```
/// - Compressed coefficients: `[3, 5, 6]`
/// - Missing coefficient: `c1 = 4`
///
/// The struct supports compressing a polynomial, reconstructing it, and retrieving
/// the original coefficients.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompressedPoly<const N: usize>(pub [BinaryField128b; N]);

impl<const N: usize> Serialize for CompressedPoly<N> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.collect_seq(self.0.iter())
    }
}

impl<'de, const N: usize> Deserialize<'de> for CompressedPoly<N> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        // Deserialize into a Vec first
        let vec = Vec::<BinaryField128b>::deserialize(deserializer)?;

        // Ensure the vector has the correct length
        if vec.len() != N {
            return Err(serde::de::Error::invalid_length(
                vec.len(),
                &format!("Expected length {N}").as_str(),
            ));
        }

        // Convert Vec into a fixed-size array
        let array: [BinaryField128b; N] = vec
            .try_into()
            .map_err(|_| serde::de::Error::custom("Failed to convert Vec to fixed-size array"))?;

        Ok(Self(array))
    }
}

#[derive(Debug)]
pub struct Assert<const COND: bool>;

pub trait IsTrue {}
impl IsTrue for Assert<true> {}

impl<const N: usize> CompressedPoly<N>
where
    Assert<{ N > 1 }>: IsTrue,
{
    /// Compresses a polynomial by omitting certain coefficients.
    ///
    /// # Parameters
    /// - `poly`: A slice of `BinaryField128b` representing the polynomial coefficients.
    ///
    /// # Returns
    /// A tuple containing:
    /// - `CompressedPoly`: The compressed polynomial.
    /// - `BinaryField128b`: The sum of the coefficients, used for reconstruction.
    ///
    /// # Example
    /// Input: `[3, 4, 5, 6]` (coefficients of `P(x) = 3 + 4x + 5x^2 + 6x^3`)
    /// Output: `(CompressedPoly([3, 5, 6]), 15)` (sum = 4 + 5 + 6 = 15)
    pub fn compress(poly: &[BinaryField128b; N]) -> (CompressedPoly<{ N - 1 }>, BinaryField128b) {
        let sum = poly.iter().skip(1).fold(BinaryField128b::ZERO, |a, b| a + b);

        // Manual initialization of the compressed polynomial
        let mut compressed = [BinaryField128b::ZERO; N - 1];
        // Copy the first coefficient
        compressed[0] = poly[0];
        // Copy the remaining coefficients
        compressed[1..N - 1].copy_from_slice(&poly[2..]);
        (CompressedPoly(compressed), sum)
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
    /// A univariate polynomial with the coefficients of the original polynomial.
    ///
    /// # Example
    /// Given the polynomial:
    /// ```text
    /// P(x) = 3 + 4x + 5x ^ 2 + 6x ^ 3
    /// ```
    /// - Compressed coefficients: `[3, 5, 6]` (skipping `c1 = 4`).
    /// - Sum at `x = 1`: `P(1) = 3 + 4 + 5 + 6 = 18`.
    ///
    /// The function reconstructs the coefficients as:
    /// ```text
    /// [3, 4, 5, 6]
    /// ```
    pub fn coeffs(&self, sum: BinaryField128b) -> FixedUnivariatePolynomial<{ N + 1 }> {
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
        let c1 = self.iter().fold(BinaryField128b::ZERO, |a, b| a + *b) + ev_1;

        // Step 4: Reconstruct the full polynomial by inserting `c1` at index `1`.
        //
        // The full polynomial will be stored in a fixed-size array.
        let mut full_coeffs = [BinaryField128b::ZERO; N + 1];

        // Step 4.1: Assign `c0` to index `0`
        full_coeffs[0] = c0;

        // Step 4.2: Assign the recovered `c1` to index `1`
        full_coeffs[1] = c1;

        // Step 4.3: Copy the remaining stored coefficients from `self[1..]` into the reconstructed
        // array. These are all the coefficients except `c0` and `c1`, placed starting from
        // index `2`.
        full_coeffs[2..].copy_from_slice(&self[1..]);

        // Step 5: Return the reconstructed polynomial wrapped in `FixedUnivariatePolynomial`.
        FixedUnivariatePolynomial::new(full_coeffs)
    }
}

impl<const N: usize> Deref for CompressedPoly<N> {
    type Target = [BinaryField128b; N];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binary_field::BinaryField128b;

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

        let expected_compressed = CompressedPoly([
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

    // #[test]
    // fn test_sum_single_coefficient() {
    //     let poly = [BinaryField128b::new(42)];
    //     let result = std::panic::catch_unwind(|| CompressedPoly::compress::<0>(&poly));
    //     assert!(result.is_err(), "Expected compress to panic for a single-element polynomial");
    // }

    #[test]
    fn test_sum_all_zero_coefficients() {
        let poly = [BinaryField128b::ZERO, BinaryField128b::ZERO, BinaryField128b::ZERO];

        let (_, sum) = CompressedPoly::compress(&poly);

        assert_eq!(sum, BinaryField128b::ZERO, "Sum of zero coefficients should be zero");
    }

    #[test]
    fn test_coeffs_reconstruction_standard_case() {
        // Define a univariate polynomial:
        // P(x) = 1 + 2x + 3x^2 + 4x^3
        let poly = [
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
            FixedUnivariatePolynomial::new(poly),
            "Reconstructed polynomial does not match the original"
        );
    }

    #[test]
    fn test_coeffs_all_zero_coefficients() {
        // Define a univariate polynomial:
        // P(x) = 0 + 0x + 0x^2 + 0x^3
        let poly = [
            BinaryField128b::ZERO,
            BinaryField128b::ZERO,
            BinaryField128b::ZERO,
            BinaryField128b::ZERO,
        ];

        // Compress the polynomial
        let (compressed, sum) = CompressedPoly::compress(&poly);

        // Reconstruct the coefficients
        let reconstructed = compressed.coeffs(sum);

        // Verify the reconstructed polynomial matches the original
        assert_eq!(
            reconstructed,
            FixedUnivariatePolynomial::new(poly),
            "Reconstructed polynomial does not match the original when all coefficients are zero"
        );
    }

    #[test]
    fn test_coeffs_large_coefficients() {
        // Define a univariate polynomial:
        // P(x) = 1_000_000 + 2_000_000x + 3_000_000x^2 + 4_000_000x^3
        let poly = [
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
            FixedUnivariatePolynomial::new(poly),
            "Reconstructed polynomial does not match the original for large coefficients"
        );
    }
}
