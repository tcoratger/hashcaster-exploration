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

/// Extracts the most significant bit (MSB) from each byte in a 16-element array of 8-bit integers
/// and constructs a 16-bit integer mask from the extracted bits.
///
/// # Description
/// This function is designed to emulate the behavior of the `_mm_movemask_epi8` intrinsic,
/// commonly used in SIMD (Single Instruction, Multiple Data) operations. It processes a
/// 16-byte array, extracts the MSB of each byte, and combines these bits into a single 16-bit
/// integer. The function iterates through the array in reverse order to align with specific
/// hardware conventions.
///
/// # Theory and Objective
/// The purpose of this function is to efficiently pack MSBs from a 16-byte array into a single
/// integer. This operation is commonly used in low-level algorithms for:
/// - Bitwise manipulations.
/// - Vectorized comparisons.
/// - Accelerating data-dependent computations.
///
/// Given an input array `x` of size 16:
/// - `x[0]` contributes its MSB to the least significant bit (LSB) of the output.
/// - `x[15]` contributes its MSB to the most significant bit (MSB) of the output.
///
/// ## Example
/// For an input array:
/// ```text
/// x = [0b10000000, 0b00000000, 0b10000000, 0b00000000, ...]
/// ```
/// The MSBs are:
/// ```text
/// [1, 0, 1, 0, ...]
/// ```
/// The result is the integer `0b1010...`
///
/// # Parameters
/// - `x`: An array of 16 `u8` integers. Each element is an 8-bit value.
///
/// # Returns
/// A 32-bit signed integer (`i32`) that represents the combined MSB mask.
///
/// # Complexity
/// - **Time Complexity**: O(16), as it iterates through all 16 bytes.
/// - **Space Complexity**: O(1), as it only uses a single integer for the result.
///
/// # Usage
/// This function can be used in scenarios where hardware SIMD instructions are unavailable,
/// or in testing environments to verify the behavior of such instructions.
#[unroll::unroll_for_loops]
pub fn cpu_v_movemask_epi8(x: [u8; 16]) -> i32 {
    // Initialize the result variable to store the combined MSB mask.
    let mut ret = 0;
    // Iterate through the 16-byte array in reverse order (from the last byte to the first).
    for i in 0..16 {
        // Shift the result one bit to the left to make space for the next MSB.
        ret <<= 1;
        // Extract the MSB of the current byte by right-shifting by 7 bits.
        // Convert the resulting bit into a 32-bit integer and add it to the result.
        ret += (x[15 - i] >> 7) as i32;
    }
    // Return the final combined mask as a 32-bit integer.
    ret
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

    #[test]
    fn test_cpu_v_movemask_epi8_standard_cases() {
        // Standard test case 1: Alternate high bits
        let input = [
            0b10000000, 0b00000000, 0b10000000, 0b00000000, 0b10000000, 0b00000000, 0b10000000,
            0b00000000, 0b10000000, 0b00000000, 0b10000000, 0b00000000, 0b10000000, 0b00000000,
            0b10000000, 0b00000000,
        ];
        // Updated Expected Result:
        // 0101010101010101 in binary = 0b0101010101010101 = 21845 in decimal
        let result = cpu_v_movemask_epi8(input);
        assert_eq!(result, 0b0101010101010101);
    }

    #[test]
    fn test_cpu_v_movemask_epi8_all_ones() {
        // All high bits set
        let input = [
            0b10000000, 0b10000000, 0b10000000, 0b10000000, 0b10000000, 0b10000000, 0b10000000,
            0b10000000, 0b10000000, 0b10000000, 0b10000000, 0b10000000, 0b10000000, 0b10000000,
            0b10000000, 0b10000000,
        ];
        // Expected result: 1111111111111111 in binary = 0b1111111111111111 = 65535 in decimal
        let result = cpu_v_movemask_epi8(input);
        assert_eq!(result, 0b1111111111111111);
    }

    #[test]
    fn test_cpu_v_movemask_epi8_all_zeros() {
        // No high bits set
        let input = [0b00000000; 16];
        // Expected result: 0
        let result = cpu_v_movemask_epi8(input);
        assert_eq!(result, 0);
    }
}
