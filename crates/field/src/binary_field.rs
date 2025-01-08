use crate::{
    backend::karatsuba::{karatsuba1, karatsuba2, mont_reduce},
    bit_iterator::BitIterator,
    frobenius::FROBENIUS,
};
use bytemuck::{Pod, Zeroable};
use num_traits::{MulAdd, MulAddAssign, One, Zero};
use rand::Rng;
use std::{
    arch::aarch64::uint8x16_t,
    iter::Sum,
    ops::{Add, AddAssign, BitAnd, Deref, Mul, MulAssign, Neg, Sub},
};

#[repr(transparent)]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Pod, Zeroable)]
pub struct BinaryField128b(u128);

impl BinaryField128b {
    /// Constructs a new [`BinaryField128b`] instance from a given `u128` value.
    pub const fn new(val: u128) -> Self {
        Self(val)
    }

    /// Returns the inner `u128` value of the [`BinaryField128b`] instance.
    pub const fn into_inner(self) -> u128 {
        self.0
    }

    /// Performs the core multiplication logic using Karatsuba and Montgomery reduction.
    unsafe fn mul_impl(lhs: u128, rhs: u128) -> u128 {
        // Perform Karatsuba decomposition on the operands (self and other).
        // This breaks each 128-bit input into high and low 64-bit parts and computes:
        // - `h`: High product (`self.hi * other.hi`)
        // - `m`: Middle product (`(self.hi ^ self.lo) * (other.hi ^ other.lo)`)
        // - `l`: Low product (`self.lo * other.lo`)
        let (h, m, l) = karatsuba1(
            std::mem::transmute::<u128, uint8x16_t>(lhs),
            std::mem::transmute::<u128, uint8x16_t>(rhs),
        );

        // Combine partial products (`h`, `m`, `l`) into a 256-bit result.
        // - `h`: Combined upper 128 bits.
        // - `l`: Combined lower 128 bits.
        let (h, l) = karatsuba2(h, m, l);

        // Perform Montgomery reduction on the 256-bit result (`h`, `l`) to
        // produce a reduced 128-bit result modulo the field polynomial.
        std::mem::transmute(mont_reduce(h, l))
    }

    /// Constructs a basis element of the binary field \( \mathbb{F}_{2^{128}} \).
    ///
    /// # Description
    /// - The `basis` function generates a basis element corresponding to \( x^i \), where \( i \)
    ///   is the degree of the monomial.
    /// - In the binary field, each power of \( x \) corresponds to a bit position in the 128-bit
    ///   representation.
    ///
    /// # Parameters
    /// - `i`: The degree of the basis element (0 ≤ `i` < 128).
    ///
    /// # Returns
    /// - A `BinaryField128b` instance representing the basis element \( x^i \).
    ///
    /// # Panics
    /// - Panics if the input `i` is greater than or equal to 128, as the field is limited to 128
    ///   bits.
    ///
    /// # Example
    /// ```rust,ignore
    /// let b0 = BinaryField128b::basis(0); // Represents x^0, or 1
    /// let b1 = BinaryField128b::basis(1); // Represents x^1, or 2
    /// let b127 = BinaryField128b::basis(127); // Represents x^127
    /// ```
    pub const fn basis(i: usize) -> Self {
        // Ensure the input degree `i` is within the valid range for the field.
        assert!(i < 128);

        // Compute the basis element by left-shifting 1 by `i` positions.
        // This sets the `i`-th bit in the binary representation.
        Self(1 << i)
    }

    /// Generates a random [`BinaryField128b`] element.
    ///
    /// # Returns
    /// - A random `BinaryField128b` instance, with the internal `u128` value generated uniformly
    ///   across the entire range of 128 bits.
    pub fn random() -> Self {
        // Use the `rand` crate to generate a random `u128` value.
        Self::new(rand::thread_rng().gen())
    }

    /// **Performs the Frobenius map computation in GF(2^128).**
    ///
    /// ### Description
    /// - The Frobenius map is a field automorphism in finite fields of characteristic 2.
    /// - It maps an element `x` to `x^(2^k)`, where `k` is the specified parameter.
    /// - This implementation leverages precomputed Frobenius tables (`FROBENIUS`) for high
    ///   efficiency.
    ///
    /// ### Parameters
    /// - `self`: The field element to which the Frobenius map will be applied.
    /// - `k`: The number of iterations of the Frobenius map. Can be positive or negative.
    ///
    /// ### Returns
    /// - A new `BinaryField128b` instance representing the result of \( x^{2^k} \), where \( x \)
    ///   is the input.
    #[must_use]
    pub fn frobenius(&self, k: i32) -> Self {
        // Normalize `k` to the range [0, 127].
        // Ensures the Frobenius map behaves cyclically with a period of 128.
        let k = k.rem_euclid(128) as usize;

        // Retrieve the precomputed Frobenius matrix for `k`.
        let matrix = &FROBENIUS[k];

        // Initialize the result accumulator and extract the raw value of `self`.
        let mut ret = 0u128;
        // Extract the internal `u128` value of `self`.
        let raw = self.into_inner();

        // Process the lower 64 bits of `raw`.
        // Iterate over each set bit in the lower half of the input (64 bits).
        for tz in BitIterator::new(raw as u64) {
            // XOR the transformation result into the accumulator.
            ret ^= matrix[tz];
        }

        // Process the upper 64 bits of `raw`.
        // Similar to the lower bits but offsets the matrix index by 64.
        for tz in BitIterator::new((raw >> 64) as u64) {
            // XOR the transformation result for upper bits.
            ret ^= matrix[tz + 64];
        }

        // Return the computed result as a new `BinaryField128b` instance.
        Self::new(ret)
    }

    /// Computes a sequence of gamma values for folding operations in a binary field.
    ///
    /// # Description
    /// This function generates a sequence of `M` gamma values, where each gamma value
    /// represents a successive power of the input `gamma`. The sequence starts with `1`
    /// (the multiplicative identity) and progresses as `[1, gamma, gamma^2, gamma^3, ...,
    /// gamma^(M-1)]`.
    ///
    /// # Parameters
    /// - `gamma`: A [`BinaryField128b`] element used as the base for the sequence.
    ///   - This value is successively multiplied to compute the next power in the sequence.
    /// - `M`: The number of gamma values to compute in the sequence.
    ///   - Must be specified as a compile-time constant.
    ///
    /// # Returns
    /// An array of size `M` containing the computed sequence of gamma values.
    pub fn compute_gammas_folding<const M: usize>(gamma: Self) -> [Self; M] {
        // Initialize the array with the identity element
        let mut gammas = [Self::one(); M];
        // Start with the multiplicative identity
        let mut current = Self::one();

        for g in gammas.iter_mut().take(M) {
            // Set the current power in the array
            *g = current;
            // Compute the next power
            current *= gamma;
        }

        gammas
    }
}

impl Sum for BinaryField128b {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, val| acc + val)
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
        Self(257_870_231_182_273_679_343_338_569_694_386_847_745)
    }
}

impl Neg for BinaryField128b {
    type Output = Self;

    fn neg(self) -> Self::Output {
        // Negation in binary fields of characteristic 2 is a no-op.
        self
    }
}

impl Add<Self> for BinaryField128b {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn add(self, other: Self) -> Self::Output {
        // Addition in binary fields of characteristic 2 is equivalent to XOR.
        Self(self.0 ^ other.0)
    }
}

impl Add<&Self> for BinaryField128b {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn add(self, other: &Self) -> Self::Output {
        // Addition in binary fields of characteristic 2 is equivalent to XOR.
        Self(self.0 ^ other.0)
    }
}

impl AddAssign<Self> for BinaryField128b {
    #[allow(clippy::suspicious_op_assign_impl)]
    fn add_assign(&mut self, other: Self) {
        self.0 ^= other.0;
    }
}

impl Sub<Self> for BinaryField128b {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn sub(self, other: Self) -> Self::Output {
        // Subtraction in binary fields of characteristic 2 is equivalent to addition (XOR).
        self + other
    }
}

impl Sub<&Self> for BinaryField128b {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn sub(self, other: &Self) -> Self::Output {
        // Subtraction in binary fields of characteristic 2 is equivalent to addition (XOR).
        self + other
    }
}

impl Mul for BinaryField128b {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, other: Self) -> Self::Output {
        unsafe { Self(Self::mul_impl(self.0, other.0)) }
    }
}

impl Mul<&Self> for BinaryField128b {
    type Output = Self;

    fn mul(self, other: &Self) -> Self::Output {
        unsafe { Self(Self::mul_impl(self.0, other.0)) }
    }
}

impl MulAssign for BinaryField128b {
    fn mul_assign(&mut self, other: Self) {
        *self = unsafe { Self(Self::mul_impl(self.0, other.0)) };
    }
}

impl MulAssign<&Self> for BinaryField128b {
    fn mul_assign(&mut self, other: &Self) {
        *self = unsafe { Self(Self::mul_impl(self.0, other.0)) };
    }
}

impl MulAdd<Self> for BinaryField128b {
    type Output = Self;

    fn mul_add(self, a: Self, b: Self) -> Self::Output {
        (self * a) + b
    }
}

impl MulAdd<&Self> for BinaryField128b {
    type Output = Self;

    fn mul_add(self, a: &Self, b: Self) -> Self::Output {
        (self * a) + b
    }
}

impl MulAddAssign<Self> for BinaryField128b {
    fn mul_add_assign(&mut self, a: Self, b: Self) {
        *self = (*self * a) + b;
    }
}

impl MulAddAssign<&Self> for BinaryField128b {
    fn mul_add_assign(&mut self, a: &Self, b: Self) {
        *self = (*self * *a) + b;
    }
}

impl Deref for BinaryField128b {
    type Target = u128;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl BitAnd<Self> for BinaryField128b {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}

impl BitAnd<&Self> for BinaryField128b {
    type Output = Self;

    fn bitand(self, rhs: &Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}

impl From<u128> for BinaryField128b {
    fn from(val: u128) -> Self {
        Self(val)
    }
}

impl From<bool> for BinaryField128b {
    fn from(val: bool) -> Self {
        if val {
            Self::one()
        } else {
            Self::zero()
        }
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
    #[allow(clippy::unreadable_literal)]
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
    #[allow(clippy::op_ref, clippy::unreadable_literal)]
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
                std::mem::transmute::<uint8x16_t, u128>(one_simd),
                std::mem::transmute::<uint8x16_t, u128>(value_simd),
            );

            // Perform Montgomery reduction on the intermediate result.
            // - Montgomery reduction reduces the 256-bit product (`x23:x01`) modulo the defining
            //   polynomial.
            // - The output is the reduced 128-bit result, which matches the expected product.
            let expected_result: u128 = std::mem::transmute(mont_reduce(
                std::mem::transmute::<u128, uint8x16_t>(x23),
                std::mem::transmute::<u128, uint8x16_t>(x01),
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
    #[allow(clippy::unreadable_literal)]
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
                std::mem::transmute::<uint8x16_t, u128>(value1_simd),
                std::mem::transmute::<uint8x16_t, u128>(value2_simd),
            );

            // Perform Montgomery reduction on the intermediate result.
            // - Montgomery reduction reduces the 256-bit product (`x23:x01`) modulo the defining
            //   polynomial.
            // - The output is the reduced 128-bit result, which matches the expected product.
            let expected_result: u128 = std::mem::transmute(mont_reduce(
                std::mem::transmute::<u128, uint8x16_t>(x23),
                std::mem::transmute::<u128, uint8x16_t>(x01),
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
    #[allow(clippy::op_ref, clippy::unreadable_literal)]
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

        for value in &values {
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

    #[test]
    fn test_basis_valid_input() {
        // Test case for a valid basis element at degree 0.
        // `basis(0)` represents x^0, which corresponds to 1.
        let b0 = BinaryField128b::basis(0);
        // Assert that the underlying value is 1 (binary: 0b1).
        assert_eq!(b0.0, 1, "basis(0) did not return the expected result: {:#b}", b0.0);

        // Test case for a valid basis element at degree 1.
        // `basis(1)` represents x^1, which corresponds to 2 (binary: 0b10).
        let b1 = BinaryField128b::basis(1);
        // Assert that the underlying value is 2.
        assert_eq!(b1.0, 2, "basis(1) did not return the expected result: {:#b}", b1.0);

        // Test case for a valid basis element at degree 127 (maximum).
        // `basis(127)` represents x^127, which corresponds to a 1 followed by 127 zeros.
        let b127 = BinaryField128b::basis(127);
        // Assert that the underlying value matches the expected binary representation.
        assert_eq!(
            b127.0,
            1 << 127,
            "basis(127) did not return the expected result: {:#b}",
            b127.0
        );
    }

    #[test]
    #[should_panic(expected = "assertion failed")]
    const fn test_basis_invalid_input() {
        // Test case for an invalid input exceeding the maximum degree (127).
        // `basis(128)` should panic because 128 is out of range.
        BinaryField128b::basis(128);
    }

    #[test]
    fn test_basis_combination() {
        // Construct a polynomial using multiple basis elements.
        // This represents x^0 + x^1 + x^3 (binary: 0b1011).
        let poly =
            BinaryField128b::basis(0) + BinaryField128b::basis(1) + BinaryField128b::basis(3);
        // Assert that the underlying value matches the expected result.
        assert_eq!(
            poly.0, 0b1011,
            "Combination of basis elements did not produce the expected result: {:#b}",
            poly.0
        );
    }

    #[test]
    fn test_frobenius() {
        // Generate a random field element in GF(2^128).
        let a = BinaryField128b::random();

        // Initialize a variable `apow` with the same value as `a`.
        // This variable will be used to compute successive powers of `a` using repeated squaring.
        let mut apow = a;

        // Iterate over the range 0 to 127.
        // The Frobenius map is periodic with a cycle length of 128, so we test all powers in this
        // range.
        for i in 0..128 {
            // Assert that the result of applying the Frobenius map to `a` matches `apow`.
            // - `a.frobenius(i)` computes \( a^{2^i} \) using the precomputed Frobenius tables.
            // - `apow` is expected to represent \( a^{2^i} \), calculated using repeated squaring.
            assert_eq!(a.frobenius(i), apow);

            // Update `apow` by squaring its current value for the next iteration.
            // This step computes \( apow = apow^2 \), which is equivalent to \( a^{2^{i+1}} \).
            apow *= apow;
        }
    }

    #[test]
    fn test_compute_gammas_folding_identity() {
        // Case: Gamma = 1
        let gamma = BinaryField128b::one();
        let result = BinaryField128b::compute_gammas_folding::<5>(gamma);

        // Expected result is a sequence of ones
        let expected = [BinaryField128b::one(); 5];
        assert_eq!(result, expected, "Gamma folding with identity failed");
    }

    #[test]
    fn test_compute_gammas_folding_simple_case() {
        // Case: Gamma = 2
        let gamma = BinaryField128b::new(2);
        let result = BinaryField128b::compute_gammas_folding::<4>(gamma);

        // Expected result
        let expected = [BinaryField128b::one(), gamma, gamma * gamma, gamma * gamma * gamma];
        assert_eq!(result, expected, "Gamma folding with simple case failed");
    }

    #[test]
    fn test_compute_gammas_folding_large_gamma() {
        // Case: Gamma = large value
        let gamma = BinaryField128b::new(123_456_789);
        let result = BinaryField128b::compute_gammas_folding::<3>(gamma);

        // Expected result calculated manually: [1, gamma, gamma^2]
        let expected = [BinaryField128b::one(), gamma, gamma * gamma];
        assert_eq!(result, expected, "Gamma folding with large gamma failed");
    }

    #[test]
    fn test_compute_gammas_folding_zero_gamma() {
        // Case: Gamma = 0
        let gamma = BinaryField128b::zero();
        let result = BinaryField128b::compute_gammas_folding::<3>(gamma);

        // Expected result: [1, 0, 0]
        let expected = [BinaryField128b::one(), BinaryField128b::zero(), BinaryField128b::zero()];
        assert_eq!(result, expected, "Gamma folding with zero gamma failed");
    }

    #[test]
    fn test_muladd_basic() {
        let a = BinaryField128b::random();
        let b = BinaryField128b::random();
        let c = BinaryField128b::random();

        // Expected result: (a * b) + c
        let expected = (a * b) + c;

        // Using MulAdd trait
        let result = a.mul_add(b, c);

        assert_eq!(
            result,
            expected,
            "MulAdd failed: got {:#x}, expected {:#x}",
            result.into_inner(),
            expected.into_inner()
        );
    }

    #[test]
    fn test_muladd_with_zero() {
        let zero = BinaryField128b::zero();
        let a = BinaryField128b::new(7);
        let b = BinaryField128b::new(11);

        // Multiplying by zero
        assert_eq!(zero.mul_add(a, b), b, "MulAdd with zero failed");
        assert_eq!(a.mul_add(zero, b), b, "MulAdd with zero failed");
    }

    #[test]
    fn test_muladd_with_one() {
        let one = BinaryField128b::one();
        let a = BinaryField128b::new(13);
        let b = BinaryField128b::new(17);

        // Multiplying by one
        let result = one.mul_add(a, b);
        let expected = a + b;

        assert_eq!(
            result,
            expected,
            "MulAdd with one failed: got {:#x}, expected {:#x}",
            result.into_inner(),
            expected.into_inner()
        );
    }

    #[test]
    fn test_muladdassign_basic() {
        let mut result = BinaryField128b::new(10);
        let a = BinaryField128b::new(2);
        let b = BinaryField128b::new(3);

        // Manually calculate expected result
        let expected = (result * a) + b;

        // Using MulAddAssign
        result.mul_add_assign(a, b);

        assert_eq!(
            result,
            expected,
            "MulAddAssign failed: got {:#x}, expected {:#x}",
            result.into_inner(),
            expected.into_inner()
        );
    }

    #[test]
    fn test_muladdassign_with_zero() {
        let mut result = BinaryField128b::new(8);
        let zero = BinaryField128b::zero();

        // Multiplying by zero should leave the accumulator unchanged
        result.mul_add_assign(zero, zero);
        assert_eq!(result, BinaryField128b::new(0), "MulAddAssign with zero failed");

        // Adding zero should leave the result unchanged
        result.mul_add_assign(BinaryField128b::one(), zero);
        assert_eq!(result, BinaryField128b::new(0), "MulAddAssign with zero addition failed");
    }

    #[test]
    fn test_muladdassign_with_one() {
        let mut result = BinaryField128b::new(9);
        let one = BinaryField128b::one();
        let a = BinaryField128b::new(4);

        // Using MulAddAssign
        result.mul_add_assign(one, a);

        // Expected: result + a
        let expected = BinaryField128b::new(9) + a;

        assert_eq!(
            result,
            expected,
            "MulAddAssign with one failed: got {:#x}, expected {:#x}",
            result.into_inner(),
            expected.into_inner()
        );
    }

    #[test]
    fn test_muladdassign_random() {
        let mut result = BinaryField128b::new(0xAAAA_BBBB_CCCC_DDDD);
        let a = BinaryField128b::new(0x1111_2222_3333_4444);
        let b = BinaryField128b::new(0x5555_6666_7777_8888);

        // Manually calculate expected result
        let expected = (result * a) + b;

        // Using MulAddAssign
        result.mul_add_assign(a, b);

        assert_eq!(
            result,
            expected,
            "MulAddAssign failed for random inputs: got {:#x}, expected {:#x}",
            result.into_inner(),
            expected.into_inner()
        );
    }
}
