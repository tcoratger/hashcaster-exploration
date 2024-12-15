use std::ops::Deref;

use hashcaster_field::binary_field::BinaryField128b;
use num_traits::identities::Zero;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CompressedPoly(Vec<BinaryField128b>);

impl CompressedPoly {
    pub fn compress(poly: &[BinaryField128b]) -> Self {
        Self(std::iter::once(&poly[0]).chain(&poly[2..]).copied().collect())
    }

    pub fn sum(&self) -> BinaryField128b {
        self.iter().skip(1).fold(BinaryField128b::zero(), |a, b| a + b)
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
        // Define a sample polynomial as an array of BinaryField128b values
        let poly = [
            BinaryField128b::new(1), // 1st coefficient
            BinaryField128b::new(2), // 2nd coefficient (skipped)
            BinaryField128b::new(3), // 3rd coefficient
            BinaryField128b::new(4), // 4th coefficient
        ];

        // Compress the polynomial
        let compressed = CompressedPoly::compress(&poly);

        // Define the expected compressed coefficients
        let expected = CompressedPoly(vec![
            BinaryField128b::new(1), // 1st coefficient
            BinaryField128b::new(3), // 3rd coefficient
            BinaryField128b::new(4), // 4th coefficient
        ]);

        // Assert that the compressed polynomial matches the expected result
        assert_eq!(
            compressed, expected,
            "Compressed polynomial does not match the expected result"
        );
    }

    #[test]
    fn test_compress_single_element() {
        // Define a single-element polynomial
        let poly = [BinaryField128b::new(42)];

        // Attempt to compress the single-element polynomial and catch any panic
        let result = std::panic::catch_unwind(|| CompressedPoly::compress(&poly));

        // Assert that compressing a single-element polynomial results in a panic
        assert!(result.is_err(), "Expected compress to panic for a single-element polynomial");
    }

    #[test]
    fn test_compress_empty() {
        // Define an empty polynomial
        let poly: [BinaryField128b; 0] = [];

        // Attempt to compress the empty polynomial
        let result = std::panic::catch_unwind(|| CompressedPoly::compress(&poly));

        // Assert that compressing an empty polynomial results in a panic
        assert!(result.is_err(), "Expected compress to panic for an empty polynomial");
    }

    #[test]
    fn test_sum_standard_case() {
        // Define a compressed polynomial with multiple coefficients
        let compressed = CompressedPoly(vec![
            BinaryField128b::new(1), // 1st coefficient (not included in the sum)
            BinaryField128b::new(3), // 2nd coefficient
            BinaryField128b::new(4), // 3rd coefficient
            BinaryField128b::new(5), // 4th coefficient
        ]);

        // Calculate the sum of all coefficients except the 1st
        let computed_sum = compressed.sum();

        // Define the expected sum
        let expected_sum =
            BinaryField128b::new(3) + BinaryField128b::new(4) + BinaryField128b::new(5);

        // Assert that the computed sum matches the expected result
        assert_eq!(
            computed_sum, expected_sum,
            "Sum of compressed polynomial coefficients does not match the expected result"
        );
    }

    #[test]
    fn test_sum_single_coefficient() {
        // Define a compressed polynomial with only one coefficient
        let compressed = CompressedPoly(vec![BinaryField128b::new(42)]);

        // The sum should be zero since there are no additional coefficients
        let computed_sum = compressed.sum();
        assert_eq!(
            computed_sum,
            BinaryField128b::zero(),
            "Sum of a single-coefficient compressed polynomial should be zero"
        );
    }

    #[test]
    fn test_sum_all_zero_coefficients() {
        // Define a compressed polynomial with all zero coefficients
        let compressed = CompressedPoly(vec![
            BinaryField128b::new(0),
            BinaryField128b::new(0),
            BinaryField128b::new(0),
        ]);

        // The sum should be zero
        let computed_sum = compressed.sum();
        assert_eq!(
            computed_sum,
            BinaryField128b::zero(),
            "Sum of a compressed polynomial with all zero coefficients should be zero"
        );
    }

    #[test]
    fn test_sum_empty_coefficients() {
        // Define an empty compressed polynomial
        let compressed = CompressedPoly(vec![]);

        // The sum should also be zero
        let computed_sum = compressed.sum();
        assert_eq!(
            computed_sum,
            BinaryField128b::zero(),
            "Sum of an empty compressed polynomial should be zero"
        );
    }
}
