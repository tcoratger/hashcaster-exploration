use crate::backend::karatsuba::{karatsuba1, karatsuba2, mont_reduce};
use num_traits::{One, Zero};
use std::ops::{Add, Deref, Mul, Neg, Sub};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BinaryField128b(u128);

impl BinaryField128b {
    #[inline(always)]
    pub const fn new(val: u128) -> Self {
        Self(val)
    }
}

impl Zero for BinaryField128b {
    fn zero() -> Self {
        Self(0)
    }

    fn is_zero(&self) -> bool {
        self.0 == 0
    }
}

impl One for BinaryField128b {
    fn one() -> Self {
        // This is the multiplicative identity in binary fields.
        // This means that for all `a` in the field, `a * 1 = a`.
        Self(257870231182273679343338569694386847745)
    }
}

impl Neg for BinaryField128b {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        // Negation in binary fields of characteristic 2 is a no-op.
        self
    }
}

impl Add<Self> for BinaryField128b {
    type Output = Self;

    #[inline(always)]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn add(self, other: Self) -> Self::Output {
        // Addition in binary fields of characteristic 2 is equivalent to XOR.
        Self(self.0 ^ other.0)
    }
}

impl Add<&Self> for BinaryField128b {
    type Output = Self;

    #[inline(always)]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn add(self, other: &Self) -> Self::Output {
        // Addition in binary fields of characteristic 2 is equivalent to XOR.
        Self(self.0 ^ other.0)
    }
}

impl Sub<Self> for BinaryField128b {
    type Output = Self;

    #[inline(always)]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn sub(self, other: Self) -> Self::Output {
        // Subtraction in binary fields of characteristic 2 is equivalent to addition (XOR).
        self + other
    }
}

impl Sub<&Self> for BinaryField128b {
    type Output = Self;

    #[inline(always)]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn sub(self, other: &Self) -> Self::Output {
        // Subtraction in binary fields of characteristic 2 is equivalent to addition (XOR).
        self + other
    }
}

impl Mul<Self> for BinaryField128b {
    type Output = Self;

    #[inline(always)]
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, other: Self) -> Self::Output {
        // Multiplication in binary fields is AND.
        Self(self.0 & other.0)
    }
}

impl Mul<&Self> for BinaryField128b {
    type Output = Self;

    #[inline(always)]
    fn mul(self, other: &Self) -> Self::Output {
        unsafe {
            // Perform Karatsuba decomposition on the operands (self and other).
            // This breaks each 128-bit input into high and low 64-bit parts and computes:
            // - `h`: High product (`self.hi * other.hi`)
            // - `m`: Middle product (`(self.hi ^ self.lo) * (other.hi ^ other.lo)`)
            // - `l`: Low product (`self.lo * other.lo`)
            let (h, m, l) = karatsuba1(std::mem::transmute(self.0), std::mem::transmute(other.0));

            // Combine partial products (`h`, `m`, `l`) into a 256-bit result.
            // - `h`: Combined upper 128 bits.
            // - `l`: Combined lower 128 bits.
            let (h, l) = karatsuba2(h, m, l);

            // Perform Montgomery reduction on the 256-bit result (`h`, `l`) to
            // produce a reduced 128-bit result modulo the field polynomial.
            std::mem::transmute(mont_reduce(h, l))
        }
    }
}

impl Deref for BinaryField128b {
    type Target = u128;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use crate::backend::test_utils::expected_pmull_result;

    use super::*;
    use std::arch::aarch64::*;

    #[test]
    fn test_binary_field_initialization() {
        // Test initialization with a specific value
        assert_eq!(BinaryField128b::new(42).0, 42);

        // Test default initialization (should be 0)
        assert_eq!(BinaryField128b::default().0, 0);
    }

    #[test]
    fn test_binary_field_comparisons() {
        // Initialize two fields
        let a = BinaryField128b::new(10);
        let b = BinaryField128b::new(20);

        // Test comparison operators
        assert!(a < b);
        assert!(b > a);
        assert!(a <= a);
        assert!(b >= b);
        assert_eq!(a, BinaryField128b::new(10));
        assert_ne!(a, b);
    }

    #[test]
    fn test_binary_field_addition() {
        // Initialize two fields with distinct values.
        // Binary: 1010 (decimal: 10)
        let a = BinaryField128b::new(0b1010);
        // Binary: 0101 (decimal: 5)
        let b = BinaryField128b::new(0b0101);

        // Test the addition of two fields using XOR.
        // Expected result: 1010 XOR 0101 = 1111 (decimal: 15)
        assert_eq!(a + b, BinaryField128b::new(0b1111));

        // Test addition with zero.
        let zero = BinaryField128b::new(0);
        // 1010 XOR 0000 = 1010
        // Adding zero should return the same value.
        assert_eq!(a + zero, a);

        // Test addition with self.
        // 1010 XOR 1010 = 0000 (any number XOR itself is 0)
        assert_eq!(a + a, zero);

        // Test addition with maximum value (all bits set to 1).
        let max_field = BinaryField128b::new(u128::MAX);
        // 1010 XOR MAX = !1010
        // Result should be bitwise NOT of a.
        assert_eq!(a + max_field, BinaryField128b::new(!a.0));

        // Test addition with default value (should behave like zero).
        let default_field = BinaryField128b::default();
        // 1010 XOR 0000 = 1010
        // Adding the default should return the same value.
        assert_eq!(a + default_field, a);
    }

    #[test]
    #[allow(clippy::op_ref)]
    fn test_binary_field_addition_with_reference() {
        // Initialize two fields with distinct values.
        // Binary: 1010 (decimal: 10)
        let a = BinaryField128b::new(0b1010);
        // Binary: 0101 (decimal: 5)
        let b = BinaryField128b::new(0b0101);

        // Test the addition of two fields using XOR with a reference.
        // Expected result: 1010 XOR 0101 = 1111 (decimal: 15)
        assert_eq!(a + &b, BinaryField128b::new(0b1111));

        // Test addition with zero.
        let zero = BinaryField128b::new(0);
        // 1010 XOR 0000 = 1010
        // Adding zero (by reference) should return the same value.
        assert_eq!(a + &zero, a);

        // Test addition with self by reference.
        // 1010 XOR 1010 = 0000 (any number XOR itself is 0)
        assert_eq!(a + &a, zero);

        // Test addition with maximum value (all bits set to 1).
        let max_field = BinaryField128b::new(u128::MAX);
        // 1010 XOR MAX = !1010
        // Result should be bitwise NOT of a.
        assert_eq!(a + &max_field, BinaryField128b::new(!a.0));

        // Test addition with default value (should behave like zero).
        let default_field = BinaryField128b::default();
        // 1010 XOR 0000 = 1010
        // Adding the default (by reference) should return the same value.
        assert_eq!(a + &default_field, a);
    }

    #[test]
    fn test_binary_field_negation() {
        // Test negation in a binary field of characteristic 2
        // Negation is a no-op in this field.

        // Case 1: Negating zero
        let zero = BinaryField128b::new(0);
        // Negation of zero should remain zero.
        assert_eq!(-zero, zero);

        // Case 2: Negating a non-zero value
        let value = BinaryField128b::new(42);
        // Negation of any value in characteristic 2 fields should remain unchanged.
        assert_eq!(-value, value);

        // Case 3: Negating the maximum value
        let max_value = BinaryField128b::new(u128::MAX);
        // Negation of the maximum value should remain the maximum value.
        assert_eq!(-max_value, max_value);
    }

    #[test]
    fn test_binary_field_subtraction() {
        // Initialize two fields with distinct values.
        // Binary: 1010 (decimal: 10)
        let a = BinaryField128b::new(0b1010);
        // Binary: 0101 (decimal: 5)
        let b = BinaryField128b::new(0b0101);

        // Test the subtraction of two fields using XOR.
        // Subtraction in binary fields of characteristic 2 is equivalent to XOR.
        // Expected result: 1010 XOR 0101 = 1111 (decimal: 15)
        assert_eq!(a - b, BinaryField128b::new(0b1111));

        // Test subtraction with zero.
        let zero = BinaryField128b::new(0);
        // 1010 XOR 0000 = 1010
        // Subtracting zero should return the same value.
        assert_eq!(a - zero, a);

        // Test subtraction with self.
        // 1010 XOR 1010 = 0000 (any number XOR itself is 0)
        assert_eq!(a - a, zero);

        // Test subtraction with maximum value (all bits set to 1).
        let max_field = BinaryField128b::new(u128::MAX);
        // 1010 XOR MAX = !1010
        // Result should be bitwise NOT of a.
        assert_eq!(a - max_field, BinaryField128b::new(!a.0));
    }

    #[test]
    #[allow(clippy::op_ref)]
    fn test_binary_field_subtraction_with_reference() {
        // Initialize two fields with distinct values.
        // Binary: 1010 (decimal: 10)
        let a = BinaryField128b::new(0b1010);
        // Binary: 0101 (decimal: 5)
        let b = BinaryField128b::new(0b0101);

        // Test the subtraction of two fields using XOR with a reference.
        // Subtraction in binary fields of characteristic 2 is equivalent to XOR.
        // Expected result: 1010 XOR 0101 = 1111 (decimal: 15)
        assert_eq!(a - &b, BinaryField128b::new(0b1111));

        // Test subtraction with zero.
        let zero = BinaryField128b::new(0);
        // 1010 XOR 0000 = 1010
        // Subtracting zero (by reference) should return the same value.
        assert_eq!(a - &zero, a);

        // Test subtraction with self by reference.
        // 1010 XOR 1010 = 0000 (any number XOR itself is 0)
        assert_eq!(a - &a, zero);

        // Test subtraction with maximum value (all bits set to 1).
        let max_field = BinaryField128b::new(u128::MAX);
        // 1010 XOR MAX = !1010
        // Result should be bitwise NOT of a.
        assert_eq!(a - &max_field, BinaryField128b::new(!a.0));
    }

    #[test]
    fn test_binary_field_zero() {
        // Create a BinaryField128b instance using the `zero` method
        let zero = BinaryField128b::zero();

        // Assert that the value is indeed zero
        assert_eq!(zero.0, 0, "BinaryField128b::zero did not return a zero value");

        // Zero should be the default value for BinaryField128b
        assert_eq!(
            zero,
            BinaryField128b::default(),
            "BinaryField128b::zero did not match the default value"
        );

        // Verify that the `is_zero` method returns true for the zero instance
        assert!(zero.is_zero(), "BinaryField128b::is_zero returned false for a zero value");

        // Create a non-zero BinaryField128b instance
        let non_zero = BinaryField128b::new(42);

        // Verify that the `is_zero` method returns false for a non-zero instance
        assert!(!non_zero.is_zero(), "BinaryField128b::is_zero returned true for a non-zero value");
    }

    #[test]
    fn test_mul_zero() {
        // Case: Multiplying zero with any number should return zero.
        let zero = BinaryField128b::new(0);
        let value = BinaryField128b::new(0x123456789ABCDEF0);

        // Multiplying zero with zero
        assert_eq!(zero.mul(&zero), zero);

        // Multiplying zero with a non-zero value
        assert_eq!(zero.mul(&value), zero);

        // Multiplying a non-zero value with zero
        assert_eq!(value.mul(&zero), zero);
    }

    #[test]
    fn test_mul_one() {
        unsafe {
            // Define the "one" value as a 128-bit number.
            let one = BinaryField128b::new(1);

            // Define a test value as a 128-bit number.
            let value = BinaryField128b::new(0x123456789ABCDEF0);

            // Perform the multiplication using the `mul` method of `BinaryField128b`.
            // - `a` computes `one * value`, where `one` is treated as the multiplier.
            // - `b` computes `value * one`, where `value` is treated as the multiplier.
            // Both cases should yield the same result because multiplication in binary fields is
            // commutative.
            let a = one * &value;
            let b = value * &one;

            // Convert the `one` value to a `uint8x16_t` SIMD vector.
            // This is required to align with the helper functions (`expected_pmull_result` and
            // `mont_reduce`).
            // - The lower 64 bits hold the value `1`, and the upper 64 bits are set to `0`.
            let one_simd: uint8x16_t =
                vreinterpretq_u8_u64(vcombine_u64(vcreate_u64(1), vcreate_u64(0)));

            // Convert the `value` to a `uint8x16_t` SIMD vector.
            // - The lower 64 bits hold the actual value, and the upper 64 bits are set to `0`.
            let value_simd: uint8x16_t =
                vreinterpretq_u8_u64(vcombine_u64(vcreate_u64(0x123456789ABCDEF0), vcreate_u64(0)));

            // Compute the expected result using the carry-less polynomial multiplication
            // helper.
            // - `expected_pmull_result` computes the intermediate product of `one_simd` and
            //   `value_simd`.
            // - It returns two 128-bit parts: `x01` (lower 128 bits) and `x23` (upper 128 bits).
            let (x01, x23) = expected_pmull_result(
                std::mem::transmute(one_simd),
                std::mem::transmute(value_simd),
            );

            // Perform Montgomery reduction on the intermediate result.
            // - Montgomery reduction reduces the 256-bit product (`x23:x01`) modulo the defining
            //   polynomial.
            // - The output is the reduced 128-bit result, which matches the expected product.
            let expected_result: u128 = std::mem::transmute(mont_reduce(
                std::mem::transmute(x23),
                std::mem::transmute(x01),
            ));

            // Assert that the result of `one * value` matches the expected result.
            assert_eq!(
                a.0, expected_result,
                "Multiplying one with value failed: got {:#x}, expected {:#x}",
                a.0, expected_result
            );

            // Assert that the result of `value * one` matches the expected result.
            // This ensures the commutativity of multiplication in binary fields.
            assert_eq!(
                b.0, expected_result,
                "Multiplying value with one failed: got {:#x}, expected {:#x}",
                b.0, expected_result
            );
        }
    }

    #[test]
    fn test_mul_random() {
        unsafe {
            // Define two random 128-bit numbers as test values.
            let value1 = BinaryField128b::new(0x123456789ABCDEF0FEDCBA9876543210);
            let value2 = BinaryField128b::new(0x0FEDCBA987654321123456789ABCDEF0);

            // Perform the multiplication using the `mul` method of `BinaryField128b`.
            // - `a` computes `value1 * value2`.
            // - `b` computes `value2 * value1`.
            // Both cases should yield the same result because multiplication in binary fields is
            // commutative.
            let a = value1.mul(&value2);
            let b = value2.mul(&value1);

            // Convert `value1` and `value2` to `uint8x16_t` SIMD vectors.
            // These are required to align with the helper functions (`expected_pmull_result` and
            // `mont_reduce`).
            let value1_simd: uint8x16_t = vreinterpretq_u8_u64(vcombine_u64(
                vcreate_u64(0xFEDCBA9876543210),
                vcreate_u64(0x123456789ABCDEF0),
            ));
            let value2_simd: uint8x16_t = vreinterpretq_u8_u64(vcombine_u64(
                vcreate_u64(0x123456789ABCDEF0),
                vcreate_u64(0x0FEDCBA987654321),
            ));

            // Compute the expected result using the carry-less polynomial multiplication helper.
            // - `expected_pmull_result` computes the intermediate product of `value1_simd` and
            //   `value2_simd`.
            // - It returns two 128-bit parts: `x01` (lower 128 bits) and `x23` (upper 128 bits).
            let (x01, x23) = expected_pmull_result(
                std::mem::transmute(value1_simd),
                std::mem::transmute(value2_simd),
            );

            // Perform Montgomery reduction on the intermediate result.
            // - Montgomery reduction reduces the 256-bit product (`x23:x01`) modulo the defining
            //   polynomial.
            // - The output is the reduced 128-bit result, which matches the expected product.
            let expected_result: u128 = std::mem::transmute(mont_reduce(
                std::mem::transmute(x23),
                std::mem::transmute(x01),
            ));

            // Assert that the result of `value1 * value2` matches the expected result.
            assert_eq!(
                a.0, expected_result,
                "Multiplying value1 with value2 failed: got {:#x}, expected {:#x}",
                a.0, expected_result
            );

            // Assert that the result of `value2 * value1` matches the expected result.
            // This ensures the commutativity of multiplication in binary fields.
            assert_eq!(
                b.0, expected_result,
                "Multiplying value2 with value1 failed: got {:#x}, expected {:#x}",
                b.0, expected_result
            );
        }
    }

    #[test]
    fn test_mul_identity() {
        // Define the multiplicative identity (one) as a 128-bit number
        let one = BinaryField128b::one();

        // Define a few test values
        let values = [
            BinaryField128b::new(0),                  // Zero value
            BinaryField128b::new(0x123456789ABCDEF0), // Arbitrary non-zero value
            BinaryField128b::new(u128::MAX),          // Maximum possible value
            BinaryField128b::new(0xFEDCBA9876543210), // Another arbitrary value
        ];

        for value in values.iter() {
            // Compute value * one
            let result_a = *value * &one;
            // Compute one * value
            let result_b = one * value;

            // Assert that both are equal to the original value
            assert_eq!(
                *value, result_a,
                "Multiplication with one (value * one) failed: got {:#x}, expected {:#x}",
                result_a.0, value.0
            );

            assert_eq!(
                *value, result_b,
                "Multiplication with one (one * value) failed: got {:#x}, expected {:#x}",
                result_b.0, value.0
            );
        }
    }
}
