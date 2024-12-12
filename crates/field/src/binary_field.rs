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

// impl One for BinaryField128b {
//     fn one() -> Self {
//         Self(1)
//     }
// }

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
    fn mul(self, other: Self) -> Self::Output {
        // Multiplication in binary fields is AND.
        Self(self.0 & other.0)
    }
}

// impl Mul<&Self> for BinaryField128b {
//     type Output = Self;

//     #[inline(always)]
//     fn mul(self, other: &Self) -> Self::Output {
//         // // Multiplication in binary fields is AND.
//         // Self(self.0 & other.0)
//     }
// }

impl Deref for BinaryField128b {
    type Target = u128;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
