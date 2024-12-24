/// Removes the most significant bit (MSB) from a given number and returns the modified value along
/// with the MSB's position.
///
/// # Description
/// This function takes an input `x` and performs the following operations:
/// - Finds the position of the most significant bit (MSB).
/// - Returns the number `x` with the MSB cleared and the position of the cleared bit.
///
/// # Parameters
/// - `x`: A `usize` integer for which the MSB will be removed.
///
/// # Returns
/// A tuple `(usize, usize)` containing:
/// - The value of `x` with the MSB cleared.
/// - The zero-based position of the MSB in the binary representation of `x`.
pub(crate) fn drop_top_bit(x: usize) -> (usize, usize) {
    // Compute the index of the most significant bit.
    let s = x.leading_zeros() as usize ^ (usize::BITS as usize - 1);
    // Return x with the top bit cleared and its position.
    (x & !(1 << s), s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_drop_top_bit_standard_cases() {
        // 10 in binary: 1010
        let result = drop_top_bit(0b1010);
        // 1010 -> 0010, MSB position = 3
        assert_eq!(result, (0b0010, 3));

        // 5 in binary: 0101
        let result = drop_top_bit(0b0101);
        // 0101 -> 0001, MSB position = 2
        assert_eq!(result, (0b0001, 2));
    }

    #[test]
    fn test_drop_top_bit_edge_cases() {
        // Smallest non-zero case: 1
        let result = drop_top_bit(1);
        // 0001 -> 0000, MSB position = 0
        assert_eq!(result, (0, 0));
    }

    #[test]
    fn test_drop_top_bit_large_numbers() {
        // Large number example: 2^31
        let result = drop_top_bit(1 << 31);
        // Only one bit set at position 31
        assert_eq!(result, (0, 31));

        // Large number example: 2^63 (maximum bit for 64-bit usize)
        let result = drop_top_bit(1 << 63);
        // Only one bit set at position 63
        assert_eq!(result, (0, 63));
    }

    #[test]
    fn test_drop_top_bit_all_bits_set() {
        // All bits set for a 4-bit number: 1111 (15 in decimal)
        let result = drop_top_bit(0b1111);
        // 1111 -> 0111, MSB position = 3
        assert_eq!(result, (0b0111, 3));
    }
}
