use std::arch::aarch64::{
    uint8x16_t, veorq_u8, vextq_u8, vgetq_lane_u64, vmull_p64, vreinterpretq_u64_u8,
    vreinterpretq_u8_p128,
};

/// Montgomery reduction polynomial `p(x)` defined as:
/// \[
///     p(x) = x^{127} + x^{126} + x^{121} + x^{63} + x^{62} + x^{57}
/// \]
const POLY: u128 = (1 << 127) | (1 << 126) | (1 << 121) | (1 << 63) | (1 << 62) | (1 << 57);

/// Performs a Karatsuba decomposition for the polynomial multiplication `x * y`.
///
/// The Karatsuba algorithm splits each input (128-bit vectors) into high (`x.hi`, `y.hi`)
/// and low (`x.lo`, `y.lo`) 64-bit halves and computes three partial products:
/// - `H = x.hi * y.hi`: High product
/// - `L = x.lo * y.lo`: Low product
/// - `M = (x.hi ^ x.lo) * (y.hi ^ y.lo)`: Middle product
///
/// The decomposition utilizes the following formula:
/// `x \times y = x_1 \times y_1 \times 10^{2k} + ((x_1 + x_2) \times (y_1 + y_2) - x_1 \times y_1 -
/// x_2 \times y_2) \times 10^k + x_2 \times y_2`
/// - `H = x_1 × y_1 × 10^{2k}` (high product, computed in this function as `h`)
/// - `L = x_2 × y_2` (low product, computed as `l`)
/// - `M = (x_1 + x_2) × (y_1 + y_2) - H - L` (middle product, partially computed here as `m`)
///
/// This implementation focuses on efficiently calculating `H`, `M`, and `L`
/// using NEON intrinsics for polynomial multiplication over GF(2).
///
/// # Parameters
/// - `x`: A 128-bit SIMD vector (`uint8x16_t`) representing the first polynomial operand.
/// - `y`: A 128-bit SIMD vector (`uint8x16_t`) representing the second polynomial operand.
///
/// # Returns
/// A tuple `(H, M, L)` where:
/// - `H` is the high product: `x.hi * y.hi`
/// - `M` is the middle product: `(x.hi ^ x.lo) * (y.hi ^ y.lo)`
/// - `L` is the low product: `x.lo * y.lo`
///
/// # Safety
/// This function is `unsafe` because it requires the `neon` target feature.
/// Ensure the target CPU supports NEON before calling this function.
///
/// # Example
/// ```rust,ignore
/// use std::arch::aarch64::*;
///
/// unsafe {
///     let x: uint8x16_t = vreinterpretq_u8_u64(vcombine_u64(
///         vcreate_u64(0x123456789ABCDEF0),
///         vcreate_u64(0xFEDCBA9876543210),
///     ));
///     let y: uint8x16_t = vreinterpretq_u8_u64(vcombine_u64(
///         vcreate_u64(0x0FEDCBA987654321),
///         vcreate_u64(0x123456789ABCDEF0),
///     ));
///     let (h, m, l) = karatsuba1(x, y);
/// }
/// ```
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn karatsuba1(
    x: uint8x16_t,
    y: uint8x16_t,
) -> (uint8x16_t, uint8x16_t, uint8x16_t) {
    // M = (x.hi ^ x.lo) * (y.hi ^ y.lo)
    let mid = pmull(veorq_u8(x, vextq_u8(x, x, 8)), veorq_u8(y, vextq_u8(y, y, 8)));
    // H = x.hi * y.hi
    let hi = pmull2(x, y);
    // L = x.lo * y.lo
    let lo = pmull(x, y);
    (hi, mid, lo)
}

/// Performs the second step of the Karatsuba algorithm to combine partial products
/// into a full 2n-bit result.
///
/// This function takes the high product (`h`), middle product (`m`), and low product (`l`)
/// computed from the first Karatsuba step and combines them into a complete result.
///
/// The combination leverages the Karatsuba formula in GF(2) arithmetic, where addition
/// is equivalent to XOR and multiplication is carry-less:
///
/// \[ z = H \cdot 2^{128} + (M \oplus H \oplus L) \cdot 2^{64} + L \]
///
/// # Parameters
/// - `h`: The high product (`H`), computed as `x_hi * y_hi`.
/// - `m`: The middle product (`M`), computed as `(x_hi ^ x_lo) * (y_hi ^ y_lo)`.
/// - `l`: The low product (`L`), computed as `x_lo * y_lo`.
///
/// # Returns
/// A tuple `(high, low)` where:
/// - `high`: Upper 128 bits of the product (`x23`).
/// - `low`: Lower 128 bits of the product (`x01`).
///
/// # Safety
/// This function is `unsafe` because it requires the `neon` target feature. Ensure that
/// the target CPU supports NEON before calling this function.
///
/// # Inline Steps
/// 1. Compute temporary XOR results to form intermediate terms.
/// 2. Reconstruct the higher and lower halves of the product.
/// 3. Return the combined 2n-bit product as two 128-bit SIMD vectors.
///
/// # Example
/// ```rust,ignore
/// use std::arch::aarch64::*;
/// unsafe {
///     let h = /* High product */;
///     let m = /* Middle product */;
///     let l = /* Low product */;
///     let (x23, x01) = karatsuba2(h, m, l);
/// }
/// ```
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn karatsuba2(
    h: uint8x16_t,
    m: uint8x16_t,
    l: uint8x16_t,
) -> (uint8x16_t, uint8x16_t) {
    let t = {
        // Compute intermediate XOR results.
        // Temporary term `t0`: XOR middle (`m`) with rotated low (`l1`) and high (`h0`).
        // This represents: {m0, m1} ^ {l1, h0} = {m0 ^ l1, m1 ^ h0}
        let t0 = veorq_u8(m, vextq_u8(l, h, 8));

        // Temporary term `t1`: XOR high (`h`) with low (`l`).
        // This represents: {h0, h1} ^ {l0, l1} = {h0 ^ l0, h1 ^ l1}
        let t1 = veorq_u8(h, l);

        // Combine `t0` and `t1` to get the XOR of all intermediate terms.
        // This represents:
        // {m0^l1, m1^h0} ^ {h0^l0, h1^l1} = {m0 ^ l1 ^ h0 ^ l0, m1 ^ h0 ^ h1 ^ l1}.
        veorq_u8(t0, t1)
    };

    // Construct the lower 128 bits of the result (`x01`).
    // Combine low product (`l`) with intermediate XOR results.
    // {m0 ^ l1 ^ h0 ^ l0, l0}
    let x01 = vextq_u8(
        vextq_u8(l, l, 8), // Rotate low (`l`) to isolate {l1, l0}.
        t,                 // XOR with intermediate term.
        8,
    );

    // Construct the upper 128 bits of the result (`x23`).
    // Combine high product (`h`) with intermediate XOR results.
    // {h1, m1 ^ h0 ^ h1 ^ l1}
    let x23 = vextq_u8(
        t,                 // XOR with intermediate term.
        vextq_u8(h, h, 8), // Rotate high (`h`) to isolate {h1, h0}.
        8,
    );

    // Return the combined result as `(high, low)`:
    // - `x23` is the upper 128 bits.
    // - `x01` is the lower 128 bits.
    (x23, x01)
}

/// Performs Montgomery reduction on a 256-bit value using POLYVAL-based arithmetic.
///
/// # Overview
/// Montgomery reduction is used to efficiently compute modular reduction in finite fields.
/// This function operates on two 128-bit vectors (`x23` and `x01`) representing a 256-bit input:
/// `[X3 : X2 : X1 : X0]`. It reduces the value modulo a polynomial `p(x)` defined as:
/// \[
///     p(x) = x^{127} + x^{124} + x^{121} + x^{114} + 1
/// \]
///
/// The steps of the Montgomery reduction are as follows:
/// 1. Compute `[A1:A0]` as the product of the lower half of the input (`X0`) and the polynomial
///    (`poly`).
/// 2. Compute `[B1:B0]` using XOR operations on `[A1:A0]` and the original input.
/// 3. Compute `[C1:C0]` as the product of `[B1:B0]` and the polynomial.
/// 4. Compute `[D1:D0]` using XOR operations on `[C1:C0]` and `[B1:B0]`.
/// 5. Compute the final result as `[D1 ⊕ X3 : D0 ⊕ X2]`.
///
/// # Parameters
/// - `x23`: High 128 bits of the input (representing `[X3 : X2]`).
/// - `x01`: Low 128 bits of the input (representing `[X1 : X0]`).
///
/// # Returns
/// - A 128-bit vector representing the reduced value.
///
/// # Steps
/// The function leverages NEON intrinsic functions for parallel bitwise and polynomial operations.
#[inline]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn mont_reduce(x23: uint8x16_t, x01: uint8x16_t) -> uint8x16_t {
    // Define the polynomial used for reduction.
    // POLY = x^{127} + x^{126} + x^{121} + x^{63} + x^{62} + x^{57}
    let poly = vreinterpretq_u8_p128(POLY);

    // Compute [A1:A0] = X0 • poly
    // This multiplies the lower 64 bits of x01 (X0) by the polynomial (poly).
    let a = pmull(x01, poly);

    // Compute [B1:B0] = [X0 ⊕ A1 : X1 ⊕ A0]
    // - `vextq_u8(a, a, 8)` extracts the high 64 bits of `a` (A1).
    // - XOR operations are used to compute:
    //   - B0 = X0 ⊕ A1
    //   - B1 = X1 ⊕ A0
    let b = veorq_u8(x01, vextq_u8(a, a, 8));

    // Compute [C1:C0] = B0 • poly
    // This multiplies the lower 64 bits of `b` (B0) by the polynomial (poly).
    let c = pmull2(b, poly);

    // Compute [D1:D0] = [B0 ⊕ C1 : B1 ⊕ C0]
    // XOR operations are used to combine `[B1:B0]` and `[C1:C0]`.
    let d = veorq_u8(c, b);

    // Compute the final output [D1 ⊕ X3 : D0 ⊕ X2]
    // XOR the intermediate result `[D1:D0]` with the upper half of the input `[X3:X2]`.
    veorq_u8(x23, d)
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
/// ```rust,ignore
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
/// ```rust,ignore
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
    use crate::backend::test_utils::{expected_mont_reduce, expected_pmull_result};
    use std::arch::aarch64::*;

    #[test]
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
            let expected: u128 = 0b11_1010;

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
    fn test_pmull() {
        unsafe {
            // Define the input polynomials
            let v1: u64 = u64::MAX;
            let v2: u64 = 0xFEDC_BA98_7654_3210;

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

            // Verify that the result matches the expected value
            // Convert the result from `pmull` into a comparable form
            let result_u128: u128 = std::mem::transmute::<uint8x16_t, u128>(result);

            // Assert equality between the computed and expected results
            assert_eq!(
                result_u128,
                expected_pmull_result(u128::from(v1), u128::from(v2)).0,
                "pmull result did not match the expected polynomial multiplication value"
            );
        }
    }

    #[test]
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
            let expected: u128 = 0b10_1110;

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
    #[allow(clippy::unreadable_literal)]
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

            // Verify that the result matches the expected value
            // Convert the result from `pmull2` into a comparable form
            let result_u128: u128 = std::mem::transmute::<uint8x16_t, u128>(result);

            // Assert equality between the computed and expected results
            assert_eq!(
                result_u128,
                expected_pmull_result(u128::from(v1_high), u128::from(v2_high)).0,
                "pmull2 result did not match the expected polynomial multiplication value"
            );
        }
    }

    #[test]
    #[allow(clippy::unreadable_literal)]
    fn test_karatsuba1() {
        unsafe {
            // Define input polynomials (128-bit numbers)
            let x: uint8x16_t = vreinterpretq_u8_u64(vcombine_u64(
                vcreate_u64(0x123456789ABCDEF0), // Lower 64 bits
                vcreate_u64(0xFEDCBA9876543210), // Upper 64 bits
            ));
            let y: uint8x16_t = vreinterpretq_u8_u64(vcombine_u64(
                vcreate_u64(0x0FEDCBA987654321), // Lower 64 bits
                vcreate_u64(0x123456789ABCDEF0), // Upper 64 bits
            ));

            // Perform Karatsuba decomposition
            let (hi, mid, lo) = karatsuba1(x, y);

            // Define individual components of `x` and `y`
            let x_hi: u64 = 0xFEDCBA9876543210; // Upper 64 bits of x
            let x_lo: u64 = 0x123456789ABCDEF0; // Lower 64 bits of x
            let y_hi: u64 = 0x123456789ABCDEF0; // Upper 64 bits of y
            let y_lo: u64 = 0x0FEDCBA987654321; // Lower 64 bits of y

            // Compute expected values

            // High product: x_hi * y_hi
            let expected_h: u128 = expected_pmull_result(u128::from(x_hi), u128::from(y_hi)).0;

            // Low product: x_lo * y_lo
            let expected_l: u128 = expected_pmull_result(u128::from(x_lo), u128::from(y_lo)).0;

            // Middle product: (x_hi ^ x_lo) * (y_hi ^ y_lo)
            let xor_x = x_hi ^ x_lo;
            let xor_y = y_hi ^ y_lo;
            let expected_m: u128 = expected_pmull_result(u128::from(xor_x), u128::from(xor_y)).0;

            // Convert results for comparison
            let computed_h: u128 = std::mem::transmute::<uint8x16_t, u128>(hi);
            let computed_m: u128 = std::mem::transmute::<uint8x16_t, u128>(mid);
            let computed_l: u128 = std::mem::transmute::<uint8x16_t, u128>(lo);

            // Assert high product
            assert_eq!(
                computed_h, expected_h,
                "High product mismatch: got {computed_h:#x}, expected {expected_h:#x}"
            );

            // Assert middle product
            assert_eq!(
                computed_m, expected_m,
                "Middle product mismatch: got {computed_m:#x}, expected {expected_m:#x}"
            );

            // Assert low product
            assert_eq!(
                computed_l, expected_l,
                "Low product mismatch: got {computed_l:#x}, expected {expected_l:#x}"
            );
        }
    }

    #[test]
    fn test_karatsuba1_simple_case_manual() {
        unsafe {
            // Define input polynomials (128-bit numbers)
            let x: uint8x16_t = vreinterpretq_u8_u64(vcombine_u64(
                vcreate_u64(0b1011), // Lower 64 bits: A(x) = x^3 + x + 1
                vcreate_u64(0),      // Upper 64 bits unused
            ));
            let y: uint8x16_t = vreinterpretq_u8_u64(vcombine_u64(
                vcreate_u64(0b0110), // Lower 64 bits: B(x) = x^2 + x
                vcreate_u64(0),      // Upper 64 bits unused
            ));

            // Perform Karatsuba decomposition
            let (hi, mid, lo) = karatsuba1(x, y);

            // Expected results
            // High product (upper 64 bits): 0 * 0 = 0
            let expected_h: u128 = 0;

            // Low product (lower 64 bits): 0b1011 * 0b0110
            // A(x) = x^3 + x + 1, B(x) = x^2 + x
            // Result: x^5 + x^4 + x^3 + x = 0b111010
            let expected_l: u128 = 0b11_1010;

            // Middle product: (0 ^ 0b1011) * (0 ^ 0b0110) = 0b1011 * 0b0110
            // Same as low product
            let expected_m: u128 = expected_l;

            // Convert results for comparison
            let computed_h: u128 = std::mem::transmute::<uint8x16_t, u128>(hi);
            let computed_m: u128 = std::mem::transmute::<uint8x16_t, u128>(mid);
            let computed_l: u128 = std::mem::transmute::<uint8x16_t, u128>(lo);

            // Assert high product
            assert_eq!(
                computed_h, expected_h,
                "High product mismatch: got {computed_h:#x}, expected {expected_h:#x}"
            );

            // Assert middle product
            assert_eq!(
                computed_m, expected_m,
                "Middle product mismatch: got {computed_m:#x}, expected {expected_m:#x}"
            );

            // Assert low product
            assert_eq!(
                computed_l, expected_l,
                "Low product mismatch: got {computed_l:#x}, expected {expected_l:#x}"
            );
        }
    }

    #[test]
    #[allow(clippy::unreadable_literal)]
    fn test_karatsuba() {
        unsafe {
            // Define input polynomials (128-bit numbers)

            // Input polynomial `x` represented as a 128-bit number
            let x: uint8x16_t = vreinterpretq_u8_u64(vcombine_u64(
                vcreate_u64(0x123456789ABCDEF0), // Lower 64 bits
                vcreate_u64(0xFEDCBA9876543210), // Upper 64 bits
            ));

            // Input polynomial `y` represented as a 128-bit number
            let y: uint8x16_t = vreinterpretq_u8_u64(vcombine_u64(
                vcreate_u64(0x0FEDCBA987654321), // Lower 64 bits
                vcreate_u64(0x123456789ABCDEF0), // Upper 64 bits
            ));

            // Perform the first step of the Karatsuba algorithm
            // Decomposes `x` and `y` into three partial products:
            // - `hi` (high product): x.hi * y.hi
            // - `mid` (middle product): (x.hi ^ x.lo) * (y.hi ^ y.lo)
            // - `lo` (low product): x.lo * y.lo
            let (hi, mid, lo) = karatsuba1(x, y);

            // Perform the second step of the Karatsuba algorithm
            // Combines the partial products `hi`, `mid`, and `lo` into a 256-bit result:
            // - `result_high`: Upper 128 bits of the result
            // - `result_low`: Lower 128 bits of the result
            let (result_high, result_low) = karatsuba2(hi, mid, lo);

            // Compute the expected result using the full polynomial multiplication
            // This function returns both lower and upper 128-bit parts of the result
            let (expected_low, expected_high) = expected_pmull_result(
                0xFEDCBA9876543210123456789ABCDEF0, // Full 128-bit value of `x`
                0x123456789ABCDEF00FEDCBA987654321, // Full 128-bit value of `y`
            );

            // Assert that the computed lower 128 bits match the expected result
            assert_eq!(
                std::mem::transmute::<uint8x16_t, u128>(result_low),
                expected_low,
                "Lower 128 bits mismatch"
            );

            // Assert that the computed upper 128 bits match the expected result
            assert_eq!(
                std::mem::transmute::<uint8x16_t, u128>(result_high),
                expected_high,
                "Upper 128 bits mismatch"
            );
        }
    }

    #[test]
    #[allow(clippy::unreadable_literal)]
    fn test_mont_reduce_simple_case() {
        unsafe {
            // Define inputs for x23 (high) and x01 (low)
            let x23: uint8x16_t = vreinterpretq_u8_u64(vcombine_u64(
                vcreate_u64(0x123456789ABCDEF0),
                vcreate_u64(0xFEDCBA9876543210),
            ));
            let x01: uint8x16_t = vreinterpretq_u8_u64(vcombine_u64(
                vcreate_u64(0x0FEDCBA987654321),
                vcreate_u64(0x123456789ABCDEF0),
            ));

            // Perform Montgomery reduction
            let result = mont_reduce(x23, x01);

            // Convert result to u128 for validation
            let result_u128 = std::mem::transmute::<uint8x16_t, u128>(result);

            // Define the polynomial (matches the POLYVAL specification)
            let poly: u128 = 1 << 127 | 1 << 126 | 1 << 121 | 1 << 63 | 1 << 62 | 1 << 57;

            // Expected result
            let expected = expected_mont_reduce(
                std::mem::transmute::<uint8x16_t, u128>(x23),
                std::mem::transmute::<uint8x16_t, u128>(x01),
                poly,
            );

            // Validate the output
            assert_eq!(
                result_u128, expected,
                "Montgomery reduction failed: got {result_u128:#x}, expected {expected:#x}"
            );
        }
    }

    #[test]
    fn test_mont_reduce_zero_case() {
        unsafe {
            // Zero input case
            let x23: uint8x16_t =
                vreinterpretq_u8_u64(vcombine_u64(vcreate_u64(0), vcreate_u64(0)));
            let x01: uint8x16_t =
                vreinterpretq_u8_u64(vcombine_u64(vcreate_u64(0), vcreate_u64(0)));

            // Perform Montgomery reduction
            let result = mont_reduce(x23, x01);

            // Convert result to u128 for validation
            let result_u128 = std::mem::transmute::<uint8x16_t, u128>(result);

            // Expected result is zero
            let expected: u128 = 0;

            // Validate the output
            assert_eq!(
                result_u128,
                expected,
                "Montgomery reduction failed for zero case: got {result_u128:#x}, expected {expected:#x}"
            );
        }
    }
}
