use std::arch::aarch64::{int64x2_t, vld1q_s64, vshlq_n_s64};

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

/// Shifts each 64-bit element of a 16-byte input array left by a constant number of bits.
///
/// # Description
/// The `v_slli_epi64` function performs a constant left shift operation on each 64-bit integer
/// element within a 128-bit SIMD vector represented as a 16-byte array. The function is highly
/// optimized for ARM architectures and uses NEON SIMD intrinsics to achieve efficient vectorized
/// operations.
///
/// # Theory
/// Left shifting a binary number by `K` bits is equivalent to multiplying the number by \(2^K\),
/// provided no bits are shifted out of range. This function operates on two 64-bit integers
/// packed into a single SIMD register (`int64x2_t`), allowing simultaneous processing of both
/// integers. The operation ensures high throughput by leveraging hardware support for SIMD
/// operations.
///
/// # Parameters
/// - `K`: A compile-time constant (`const`) representing the number of bits to shift. Must be in
///   the range `[0, 63]` to ensure valid shifts for 64-bit integers.
/// - `x`: A 16-byte array representing two packed 64-bit integers to be shifted.
///
/// # Returns
/// A 16-byte array containing the two shifted 64-bit integers.
///
/// # Safety
/// - The function uses `unsafe` because it relies on raw pointer casts and platform-specific NEON
///   intrinsics.
/// - Ensure the input is valid for a left shift operation, and the target hardware supports NEON.
///
/// # Example
/// ```rust
/// let input: [u8; 16] = [
///     0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // 1
///     0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // 2
/// ];
/// let result = unsafe { v_slli_epi64::<1>(input) };
/// assert_eq!(
///     result,
///     [
///         0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // 1 << 1 = 2
///         0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // 2 << 1 = 4
///     ]
/// );
/// ```
pub unsafe fn v_slli_epi64<const K: i32>(x: [u8; 16]) -> [u8; 16] {
    // Load the 16-byte input array into a 128-bit SIMD register as two 64-bit integers.
    let data = vld1q_s64(x.as_ptr() as *const i64);

    // Perform a left shift operation on each 64-bit integer in the SIMD register.
    //    - `vshlq_n_s64` shifts all 64-bit integers in the SIMD register by `K` bits to the left.
    //    - The operation is performed on both integers simultaneously.
    let result = vshlq_n_s64(data, K);

    // Transmute the result back into a 16-byte array.
    std::mem::transmute::<int64x2_t, [u8; 16]>(result)
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

    #[test]
    fn test_v_slli_epi64_basic_shift() {
        // Example input: 64-bit integers represented in 16 bytes.
        let input: [u8; 16] = [
            0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // 1
            0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // 2
        ];
        let expected: [u8; 16] = [
            0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // 1 << 1 = 2
            0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // 2 << 1 = 4
        ];

        // Call the function with a shift constant of 1.
        let result = unsafe { v_slli_epi64::<1>(input) };
        assert_eq!(result, expected);
    }

    #[test]
    fn test_v_slli_epi64_zero_shift() {
        // No shifting should result in the same values.
        let input: [u8; 16] = [
            0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // 16
            0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // 32
        ];
        let expected = input;

        // Call the function with a shift constant of 0.
        let result = unsafe { v_slli_epi64::<0>(input) };
        assert_eq!(result, expected);
    }

    #[test]
    fn test_v_slli_epi64_edge_cases() {
        // Test with maximum and minimum values.
        let input: [u8; 16] = [
            0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, // -1 (all bits set)
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, // Smallest i64 (-2^63)
        ];
        let expected: [u8; 16] = [
            0xFE, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, // -1 << 1
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // -2^63 << 1 (overflows to 0)
        ];

        // Call the function with a shift constant of 1.
        let result = unsafe { v_slli_epi64::<1>(input) };
        assert_eq!(result, expected);
    }

    #[test]
    fn test_v_slli_epi64_no_overflow() {
        // Check that values do not overflow when within safe bounds.
        let input: [u8; 16] = [
            0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // 1
            0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // 64
        ];
        let expected: [u8; 16] = [
            0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // 1 << 3 = 8
            0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // 64 << 3 = 512
        ];

        // Call the function with a shift constant of 3.
        let result = unsafe { v_slli_epi64::<3>(input) };
        assert_eq!(result, expected);
    }
}
