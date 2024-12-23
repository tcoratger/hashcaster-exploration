use std::ops::Deref;

use hashcaster_field::binary_field::BinaryField128b;
use num_traits::identities::Zero;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CompressedPoly(Vec<BinaryField128b>);

impl CompressedPoly {
    pub fn compress(poly: &[BinaryField128b]) -> (Self, BinaryField128b) {
        let sum = poly.iter().skip(1).fold(BinaryField128b::zero(), |a, b| a + b);
        (Self(std::iter::once(&poly[0]).chain(&poly[2..]).copied().collect()), sum)
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
}
