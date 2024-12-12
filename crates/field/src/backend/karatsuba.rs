use std::arch::aarch64::{
    uint8x16_t, veorq_u8, vextq_u8, vgetq_lane_u64, vmull_p64, vreinterpretq_u64_u8,
};

/// Karatsuba decomposition for `x*y`.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn karatsuba1(x: uint8x16_t, y: uint8x16_t) -> (uint8x16_t, uint8x16_t, uint8x16_t) {
    // First Karatsuba step: decompose x and y.
    //
    // (x1*y0 + x0*y1) = (x1+x0) * (y1+x0) + (x1*y1) + (x0*y0)
    //        M                                 H         L
    //
    // m = x.hi^x.lo * y.hi^y.lo
    let m = pmull(
        veorq_u8(x, vextq_u8(x, x, 8)), // x.hi^x.lo
        veorq_u8(y, vextq_u8(y, y, 8)), // y.hi^y.lo
    );
    let h = pmull2(x, y); // h = x.hi * y.hi
    let l = pmull(x, y); // l = x.lo * y.lo
    (h, m, l)
}

/// Perform a polynomial multiplication of the low 64 bits of two 128-bit vectors.
///
/// This function uses the ARM NEON intrinsic `vmull_p64` to multiply the lower 64-bit lanes
/// of two input vectors, `a` and `b`, interpreted as polynomial coefficients over GF(2).
/// The result is a 128-bit vector containing the full product of the two 64-bit inputs.
///
/// # Parameters
///
/// - `a`: A 128-bit SIMD vector (`uint8x16_t`) containing the first polynomial operand.
/// - `b`: A 128-bit SIMD vector (`uint8x16_t`) containing the second polynomial operand.
///
/// # Returns
///
/// - A 128-bit SIMD vector (`uint8x16_t`) representing the product of the lower 64-bit lanes of `a`
///   and `b` as a polynomial over GF(2).
///
/// # Explanation of Polynomial Multiplication in GF(2)
///
/// In polynomial multiplication over GF(2), the coefficients are either 0 or 1. Arithmetic in GF(2)
/// is carry-less, meaning addition is equivalent to XOR, and multiplication uses bitwise AND.
///
/// For instance:
/// Let `A(x) = x^3 + x + 1` (binary: `0b1011`) and `B(x) = x^2 + x` (binary: `0b0110`).
///
/// To compute the product `C(x) = A(x) * B(x)`:
/// - Perform bit-shifting of `A(x)` for each set bit in `B(x)`.
/// - Accumulate the results using XOR to simulate addition in GF(2).
/// - Result: `C(x) = x^5 + x^4 + x^3 + 2x^2 + x = x^5 + x^4 + x^3 + x mod 2` (binary: `0b111010`).
///
/// This process is implemented in the test function using bitwise operations.
///
/// # Safety
///
/// - This function is marked as `unsafe` because it relies on the `neon` target feature.
/// - The caller must ensure that the CPU supports the `neon` feature before invoking this function.
///
/// # Example
///
/// ```rust
/// use std::arch::aarch64::*;
///
/// unsafe {
///     let a: uint8x16_t =
///         vreinterpretq_u8_u64(vcombine_u64(vcreate_u64(0x12345678), vcreate_u64(0)));
///     let b: uint8x16_t =
///         vreinterpretq_u8_u64(vcombine_u64(vcreate_u64(0x9abcdef0), vcreate_u64(0)));
///     let result = pmull(a, b);
/// }
/// ```
///
/// # References
///
/// - ARM NEON documentation: <https://developer.arm.com/documentation>
/// - `vmull_p64` intrinsic: Performs a polynomial multiplication of two 64-bit inputs.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn pmull(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    std::mem::transmute(vmull_p64(
        vgetq_lane_u64(vreinterpretq_u64_u8(a), 0),
        vgetq_lane_u64(vreinterpretq_u64_u8(b), 0),
    ))
}

/// Perform a polynomial multiplication of the high 64 bits of two 128-bit vectors.
///
/// This function uses the ARM NEON intrinsic `vmull_p64` to multiply the upper 64-bit lanes
/// of two input vectors, `a` and `b`, interpreted as polynomial coefficients over GF(2).
/// The result is a 128-bit vector containing the full product of the two 64-bit inputs.
///
/// # Parameters
///
/// - `a`: A 128-bit SIMD vector (`uint8x16_t`) containing the first polynomial operand.
/// - `b`: A 128-bit SIMD vector (`uint8x16_t`) containing the second polynomial operand.
///
/// # Returns
///
/// - A 128-bit SIMD vector (`uint8x16_t`) representing the product of the upper 64-bit lanes of `a`
///   and `b` as a polynomial over GF(2).
///
/// # Explanation of Polynomial Multiplication in GF(2)
///
/// In polynomial multiplication over GF(2), the coefficients are either 0 or 1. Arithmetic in GF(2)
/// is carry-less, meaning addition is equivalent to XOR, and multiplication uses bitwise AND.
///
/// For instance:
/// Let the upper halves of the inputs be:
/// - `A(x) = x^3 + x + 1` (binary: `0b1011`)
/// - `B(x) = x^2 + x` (binary: `0b0110`)
///
/// To compute the product `C(x) = A(x) * B(x)`:
/// - Perform bit-shifting of `A(x)` for each set bit in `B(x)`.
/// - Accumulate the results using XOR to simulate addition in GF(2).
/// - Result: `C(x) = x^5 + x^4 + x^3 + 2x^2 + x = x^5 + x^4 + x^3 + x mod 2` (binary: `0b111010`).
///
/// This process is implemented in the test function using bitwise operations.
///
/// # Safety
///
/// - This function is marked as `unsafe` because it relies on the `neon` target feature.
/// - The caller must ensure that the CPU supports the `neon` feature before invoking this function.
///
/// # Example
///
/// ```rust
/// use std::arch::aarch64::*;
///
/// unsafe {
///     // Define input polynomials
///     let a: uint8x16_t = vreinterpretq_u8_u64(vcombine_u64(
///         vcreate_u64(0x0000000000000000),
///         vcreate_u64(0b1011), // Upper 64 bits: A(x) = x^3 + x + 1
///     ));
///     let b: uint8x16_t = vreinterpretq_u8_u64(vcombine_u64(
///         vcreate_u64(0x0000000000000000),
///         vcreate_u64(0b0110), // Upper 64 bits: B(x) = x^2 + x
///     ));
///
///     // Perform polynomial multiplication on the upper halves
///     let result = pmull2(a, b);
///
///     // Expected result: C(x) = x^5 + x^4 + x^3 + x
///     let expected: u128 = 0b111010;
///
///     // Convert result for comparison
///     let result_u128: u128 = std::mem::transmute::<uint8x16_t, u128>(result);
///
///     // Assert that the result matches the expected value
///     assert_eq!(result_u128, expected, "pmull2 result did not match expected value");
/// }
/// ```
///
/// # Notes
///
/// - This function focuses exclusively on the upper 64-bit lanes of the input vectors.
/// - The lower 64 bits of the input vectors are ignored during the operation.
///
/// # References
///
/// - ARM NEON documentation: <https://developer.arm.com/documentation>
/// - `vmull_p64` intrinsic: Performs a polynomial multiplication of two 64-bit inputs from the high
///   halves.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn pmull2(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    std::mem::transmute(vmull_p64(
        vgetq_lane_u64(vreinterpretq_u64_u8(a), 1),
        vgetq_lane_u64(vreinterpretq_u64_u8(b), 1),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::arch::aarch64::*;

    #[test]
    #[allow(unsafe_code)]
    fn test_pmull_simple() {
        unsafe {
            // Define the input polynomials
            // Polynomial A(x) = x^3 + x + 1 (binary: 0b1011)
            let v1: u64 = 0b1011;
            // Polynomial B(x) = x^2 + x (binary: 0b0110)
            let v2: u64 = 0b0110;

            // Prepare the input polynomials as `uint8x16_t` vectors
            // Represent the 64-bit polynomial in the lower half of the vector
            let a: uint8x16_t = vreinterpretq_u8_u64(vcombine_u64(
                vcreate_u64(v1), // Lower 64 bits of the polynomial
                vcreate_u64(0),  // Upper 64 bits unused
            ));

            let b: uint8x16_t = vreinterpretq_u8_u64(vcombine_u64(
                vcreate_u64(v2), // Lower 64 bits of the polynomial
                vcreate_u64(0),  // Upper 64 bits unused
            ));

            // Perform polynomial multiplication using `pmull`
            // `pmull` performs carry-less multiplication of the lower 64 bits of `a` and `b`
            let result = pmull(a, b);

            // Compute the expected result directly
            // Explanation of the result:
            // Polynomial A(x) = x^3 + x + 1
            // Polynomial B(x) = x^2 + x
            //
            // Multiplication of A(x) and B(x) involves:
            // - (x^3) * (x^2 + x) = x^5 + x^4
            // - (x) * (x^2 + x) = x^3 + x^2
            // - (1) * (x^2 + x) = x^2 + x
            //
            // Combine all terms: x^5 + x^4 + x^3 + 2x^2 + x
            // Simplify using modulo 2 arithmetic: x^5 + x^4 + x^3 + x
            // Binary representation: 0b111010
            let expected: u128 = 0b111010;

            // Convert the result from `pmull` into a comparable form
            let result_u128: u128 = std::mem::transmute::<uint8x16_t, u128>(result);

            // Assert equality between the computed and expected results
            assert_eq!(
                result_u128, expected,
                "pmull result did not match the expected polynomial multiplication value"
            );
        }
    }

    #[test]
    #[allow(unsafe_code)]
    fn test_pmull() {
        unsafe {
            // Define the input polynomials
            let v1: u64 = u64::MAX;
            let v2: u64 = 0xFEDCBA9876543210;

            // Prepare the input polynomials as `uint8x16_t` vectors
            // Represent the 64-bit polynomial as the lower half of the `uint8x16_t` vector
            let a: uint8x16_t = vreinterpretq_u8_u64(vcombine_u64(
                vcreate_u64(v1), // Lower 64 bits of the polynomial
                vcreate_u64(0),  // Upper 64 bits unused
            ));

            let b: uint8x16_t = vreinterpretq_u8_u64(vcombine_u64(
                vcreate_u64(v2), // Lower 64 bits of the polynomial
                vcreate_u64(0),  // Upper 64 bits unused
            ));

            // Perform polynomial multiplication using `pmull`
            // This uses the carry-less multiplication intrinsic
            let result = pmull(a, b);

            // Compute the expected result manually
            // Use bitwise operations to simulate polynomial multiplication in GF(2^n)
            // Initialize the expected result to zero
            let mut expected: u128 = 0;
            // Interpret the lower half of `v1` as a 128-bit integer
            let operand_a = v1 as u128;
            // Interpret the lower half of `v2` as a 128-bit integer
            let operand_b = v2 as u128;
            for i in 0..64 {
                // Iterate through each bit of the second operand
                if (operand_b & (1 << i)) != 0 {
                    // Check if the i-th bit of `b` is set
                    // If set, XOR the shifted `a` (i.e., multiply by x^i) into the result
                    expected ^= operand_a << i;
                }
            }

            // Verify that the result matches the expected value
            // Convert the result from `pmull` into a comparable form
            let result_u128: u128 = std::mem::transmute::<uint8x16_t, u128>(result);

            // Assert equality between the computed and expected results
            assert_eq!(
                result_u128, expected,
                "pmull result did not match the expected polynomial multiplication value"
            );
        }
    }

    #[test]
    #[allow(unsafe_code)]
    fn test_pmull2_simple() {
        unsafe {
            // Define the input polynomials for the upper 64 bits
            // Polynomial A(x) = x^3 + x^2 + 1 (binary: 0b1101)
            let v1_low: u64 = 0; // Lower 64 bits unused
            let v1_high: u64 = 0b1101; // Upper 64 bits of A(x)

            // Polynomial B(x) = x^2 + x (binary: 0b0110)
            let v2_low: u64 = 0; // Lower 64 bits unused
            let v2_high: u64 = 0b0110; // Upper 64 bits of B(x)

            // Prepare the input polynomials as `uint8x16_t` vectors
            // Represent the 128-bit polynomial across both halves of the vector
            let a: uint8x16_t = vreinterpretq_u8_u64(vcombine_u64(
                vcreate_u64(v1_low),  // Lower 64 bits
                vcreate_u64(v1_high), // Upper 64 bits
            ));

            let b: uint8x16_t = vreinterpretq_u8_u64(vcombine_u64(
                vcreate_u64(v2_low),  // Lower 64 bits
                vcreate_u64(v2_high), // Upper 64 bits
            ));

            // Perform polynomial multiplication on the upper 64 bits using `pmull2`
            let result = pmull2(a, b);

            // Compute the expected result directly
            // Explanation of the result:
            // Polynomial A(x) = x^3 + x^2 + 1
            // Polynomial B(x) = x^2 + x
            //
            // Multiplication of A(x) and B(x) involves:
            // - (x^3) * (x^2 + x) = x^5 + x^4
            // - (x^2) * (x^2 + x) = x^4 + x^3
            // - (1) * (x^2 + x) = x^2 + x
            //
            // Combine all terms: x^5 + 2x^4 + x^3 + x^2 + x
            // Simplify using modulo 2 arithmetic: x^5 + x^3 + x^2 + x
            // Binary representation: 0b101110
            let expected: u128 = 0b101110;

            // Convert the result from `pmull2` into a comparable form
            let result_u128: u128 = std::mem::transmute::<uint8x16_t, u128>(result);

            // Assert equality between the computed and expected results
            assert_eq!(
                result_u128, expected,
                "pmull2 result did not match the expected polynomial multiplication value"
            );
        }
    }

    #[test]
    #[allow(unsafe_code)]
    fn test_pmull2() {
        unsafe {
            // Define the input polynomials
            // Lower 64 bits of the first polynomial
            let v1_low: u64 = 0x123456789ABCDEF0;
            // Upper 64 bits of the first polynomial
            let v1_high: u64 = 0xFEDCBA9876543210;

            // Lower 64 bits of the second polynomial
            let v2_low: u64 = 0x0FEDCBA987654321;
            // Upper 64 bits of the second polynomial
            let v2_high: u64 = 0x123456789ABCDEF0;

            // Prepare the input polynomials as `uint8x16_t` vectors
            // Represent the 128-bit polynomial across both halves of the vector
            let a: uint8x16_t = vreinterpretq_u8_u64(vcombine_u64(
                vcreate_u64(v1_low),  // Lower 64 bits
                vcreate_u64(v1_high), // Upper 64 bits
            ));

            let b: uint8x16_t = vreinterpretq_u8_u64(vcombine_u64(
                vcreate_u64(v2_low),  // Lower 64 bits
                vcreate_u64(v2_high), // Upper 64 bits
            ));

            // Perform polynomial multiplication on the upper 64 bits using `pmull2`
            let result = pmull2(a, b);

            // Compute the expected result manually
            // Use bitwise operations to simulate polynomial multiplication in GF(2^n)
            // Initialize the expected result to zero
            let mut expected: u128 = 0;
            // Interpret the upper half of `v1` as a 128-bit integer
            let operand_a = v1_high as u128;
            // Interpret the upper half of `v2` as a 128-bit integer
            let operand_b = v2_high as u128;
            for i in 0..64 {
                // Iterate through each bit of the second operand
                if (operand_b & (1 << i)) != 0 {
                    // Check if the i-th bit of `b` is set
                    // If set, XOR the shifted `a` (i.e., multiply by x^i) into the result
                    expected ^= operand_a << i;
                }
            }

            // Verify that the result matches the expected value
            // Convert the result from `pmull2` into a comparable form
            let result_u128: u128 = std::mem::transmute::<uint8x16_t, u128>(result);

            // Assert equality between the computed and expected results
            assert_eq!(
                result_u128, expected,
                "pmull2 result did not match the expected polynomial multiplication value"
            );
        }
    }
}
