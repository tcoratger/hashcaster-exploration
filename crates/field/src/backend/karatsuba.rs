use std::arch::aarch64::{
    uint8x16_t, veorq_u8, vextq_u8, vgetq_lane_u64, vmull_p64, vreinterpretq_u64_u8,
};

/// Performs a Karatsuba decomposition for the polynomial multiplication `x * y`.
///
/// The Karatsuba algorithm splits each input (128-bit vectors) into high (`x.hi`, `y.hi`)
/// and low (`x.lo`, `y.lo`) 64-bit halves and computes three partial products:
/// - `H = x.hi * y.hi`: High product
/// - `L = x.lo * y.lo`: Low product
/// - `M = (x.hi ^ x.lo) * (y.hi ^ y.lo)`: Middle product
///
/// The decomposition utilizes the following formula:
/// \[
/// x \times y = x_1 \times y_1 \times 10^{2k} + ((x_1 + x_2) \times (y_1 + y_2) - x_1 \times y_1 -
/// x_2 \times y_2) \times 10^k + x_2 \times y_2 \]
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
/// ```rust
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
unsafe fn karatsuba1(x: uint8x16_t, y: uint8x16_t) -> (uint8x16_t, uint8x16_t, uint8x16_t) {
    // M = (x.hi ^ x.lo) * (y.hi ^ y.lo)
    let m = pmull(veorq_u8(x, vextq_u8(x, x, 8)), veorq_u8(y, vextq_u8(y, y, 8)));
    // H = x.hi * y.hi
    let h = pmull2(x, y);
    // L = x.lo * y.lo
    let l = pmull(x, y);
    (h, m, l)
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
/// - `high`: Upper 128 bits of the product (`z_hi`).
/// - `low`: Lower 128 bits of the product (`z_lo`).
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
/// ```rust
/// use std::arch::aarch64::*;
/// unsafe {
///     let h = /* High product */;
///     let m = /* Middle product */;
///     let l = /* Low product */;
///     let (z_hi, z_lo) = karatsuba2(h, m, l);
/// }
/// ```
#[inline]
#[target_feature(enable = "neon")]
unsafe fn karatsuba2(h: uint8x16_t, m: uint8x16_t, l: uint8x16_t) -> (uint8x16_t, uint8x16_t) {
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

    // Construct the lower 128 bits of the result (`z_lo`).
    // Combine low product (`l`) with intermediate XOR results.
    // {m0 ^ l1 ^ h0 ^ l0, l0}
    let z_lo = vextq_u8(
        vextq_u8(l, l, 8), // Rotate low (`l`) to isolate {l1, l0}.
        t,                 // XOR with intermediate term.
        8,
    );

    // Construct the upper 128 bits of the result (`z_hi`).
    // Combine high product (`h`) with intermediate XOR results.
    // {h1, m1 ^ h0 ^ h1 ^ l1}
    let z_hi = vextq_u8(
        t,                 // XOR with intermediate term.
        vextq_u8(h, h, 8), // Rotate high (`h`) to isolate {h1, h0}.
        8,
    );

    // Return the combined result as `(high, low)`:
    // - `z_hi` is the upper 128 bits.
    // - `z_lo` is the lower 128 bits.
    (z_hi, z_lo)
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
    use ruint::aliases::U256;
    use std::arch::aarch64::*;

    /// Computes the carry-less polynomial multiplication over GF(2).
    ///
    /// This function simulates the polynomial multiplication of two 128-bit integers `a` and `b`
    /// using carry-less arithmetic in the finite field GF(2). In GF(2), addition is performed with
    /// XOR, and multiplication is performed with AND and shifts.
    ///
    /// # Parameters
    /// - `a`: First 128-bit polynomial operand.
    /// - `b`: Second 128-bit polynomial operand.
    ///
    /// # Returns
    /// - A tuple `(low, high)` where:
    ///   - `low` is the lower 128 bits of the product.
    ///   - `high` is the upper 128 bits of the product.
    ///
    /// # Example
    /// ```
    /// let a = 0b1011; // Polynomial: x^3 + x + 1
    /// let b = 0b0110; // Polynomial: x^2 + x
    /// assert_eq!(expected_pmull_result(a, b), (0b111010, 0)); // x^5 + x^4 + x^3 + x
    /// ```
    fn expected_pmull_result(a: u128, b: u128) -> (u128, u128) {
        // Use a `fold` over 256 bits to compute the result iteratively.
        // `U256::ZERO` initializes the accumulator for XOR results.
        // Each iteration computes the contribution of the `i`-th bit of `b`.
        let result = (0..256).fold(U256::ZERO, |acc, i| {
            // Extract the `i`-th bit of `b` using a right shift and AND.
            // If this bit is set, shift `a` left by `i` positions and XOR into the accumulator.
            acc ^ ((U256::from(a) << i) * ((U256::from(b) >> i) & U256::from(1)))
        });

        // Convert the 256-bit result into a big-endian byte array.
        let bytes: [u8; U256::BYTES] = result.to_be_bytes();

        // Split the byte array into two 128-bit segments.
        // The higher 128 bits correspond to the first half of the array.
        let high = u128::from_be_bytes(bytes[..16].try_into().unwrap());

        // The lower 128 bits correspond to the second half of the array.
        let low = u128::from_be_bytes(bytes[16..].try_into().unwrap());

        // Return the low and high parts of the result as a tuple.
        (low, high)
    }

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

            // Verify that the result matches the expected value
            // Convert the result from `pmull` into a comparable form
            let result_u128: u128 = std::mem::transmute::<uint8x16_t, u128>(result);

            // Assert equality between the computed and expected results
            assert_eq!(
                result_u128,
                expected_pmull_result(v1 as u128, v2 as u128).0,
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
                expected_pmull_result(v1_high as u128, v2_high as u128).0,
                "pmull2 result did not match the expected polynomial multiplication value"
            );
        }
    }

    #[test]
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
            let (h, m, l) = karatsuba1(x, y);

            // Define individual components of `x` and `y`
            let x_hi: u64 = 0xFEDCBA9876543210; // Upper 64 bits of x
            let x_lo: u64 = 0x123456789ABCDEF0; // Lower 64 bits of x
            let y_hi: u64 = 0x123456789ABCDEF0; // Upper 64 bits of y
            let y_lo: u64 = 0x0FEDCBA987654321; // Lower 64 bits of y

            // Compute expected values

            // High product: x_hi * y_hi
            let expected_h: u128 = expected_pmull_result(x_hi as u128, y_hi as u128).0;

            // Low product: x_lo * y_lo
            let expected_l: u128 = expected_pmull_result(x_lo as u128, y_lo as u128).0;

            // Middle product: (x_hi ^ x_lo) * (y_hi ^ y_lo)
            let xor_x = x_hi ^ x_lo;
            let xor_y = y_hi ^ y_lo;
            let expected_m: u128 = expected_pmull_result(xor_x as u128, xor_y as u128).0;

            // Convert results for comparison
            let computed_h: u128 = std::mem::transmute::<uint8x16_t, u128>(h);
            let computed_m: u128 = std::mem::transmute::<uint8x16_t, u128>(m);
            let computed_l: u128 = std::mem::transmute::<uint8x16_t, u128>(l);

            // Assert high product
            assert_eq!(
                computed_h, expected_h,
                "High product mismatch: got {:#x}, expected {:#x}",
                computed_h, expected_h
            );

            // Assert middle product
            assert_eq!(
                computed_m, expected_m,
                "Middle product mismatch: got {:#x}, expected {:#x}",
                computed_m, expected_m
            );

            // Assert low product
            assert_eq!(
                computed_l, expected_l,
                "Low product mismatch: got {:#x}, expected {:#x}",
                computed_l, expected_l
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
            let (h, m, l) = karatsuba1(x, y);

            // Expected results
            // High product (upper 64 bits): 0 * 0 = 0
            let expected_h: u128 = 0;

            // Low product (lower 64 bits): 0b1011 * 0b0110
            // A(x) = x^3 + x + 1, B(x) = x^2 + x
            // Result: x^5 + x^4 + x^3 + x = 0b111010
            let expected_l: u128 = 0b111010;

            // Middle product: (0 ^ 0b1011) * (0 ^ 0b0110) = 0b1011 * 0b0110
            // Same as low product
            let expected_m: u128 = expected_l;

            // Convert results for comparison
            let computed_h: u128 = std::mem::transmute::<uint8x16_t, u128>(h);
            let computed_m: u128 = std::mem::transmute::<uint8x16_t, u128>(m);
            let computed_l: u128 = std::mem::transmute::<uint8x16_t, u128>(l);

            // Assert high product
            assert_eq!(
                computed_h, expected_h,
                "High product mismatch: got {:#x}, expected {:#x}",
                computed_h, expected_h
            );

            // Assert middle product
            assert_eq!(
                computed_m, expected_m,
                "Middle product mismatch: got {:#x}, expected {:#x}",
                computed_m, expected_m
            );

            // Assert low product
            assert_eq!(
                computed_l, expected_l,
                "Low product mismatch: got {:#x}, expected {:#x}",
                computed_l, expected_l
            );
        }
    }

    #[test]
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
            // - `h` (high product): x.hi * y.hi
            // - `m` (middle product): (x.hi ^ x.lo) * (y.hi ^ y.lo)
            // - `l` (low product): x.lo * y.lo
            let (h, m, l) = karatsuba1(x, y);

            // Perform the second step of the Karatsuba algorithm
            // Combines the partial products `h`, `m`, and `l` into a 256-bit result:
            // - `result_high`: Upper 128 bits of the result
            // - `result_low`: Lower 128 bits of the result
            let (result_high, result_low) = karatsuba2(h, m, l);

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
}
