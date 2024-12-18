use ruint::aliases::U256;

/// Helper function to precompute the expected result of Montgomery reduction.
///
/// This function simulates the Montgomery reduction process in software, step-by-step.
/// The inputs are split into 64-bit chunks, and the reduction is performed based on
/// the theoretical model of POLYVAL Montgomery reduction.
///
/// # Parameters
/// - `x23`: The high 128 bits of the input (representing `[X3 : X2]`).
/// - `x01`: The low 128 bits of the input (representing `[X1 : X0]`).
/// - `poly`: The polynomial used for reduction.
///
/// # Returns
/// - A 128-bit result representing the reduced value.
///
/// # Steps
/// The Montgomery reduction process involves:
/// 1. Multiplying `X0` with the polynomial (`poly`).
/// 2. `XORing` intermediate results to compute `[B1 : B0]`.
/// 3. Multiplying `B0` with the polynomial.
/// 4. Computing `[D1 : D0]` by `XORing` results.
/// 5. Combining the final result using `[D1 ⊕ X3 : D0 ⊕ X2]`.
pub(crate) fn expected_mont_reduce(x23: u128, x01: u128, poly: u128) -> u128 {
    // Extract the lower and upper 64-bit halves of `x01` (low 128 bits of input)
    let x0 = x01 & ((1 << 64) - 1); // Lower 64 bits of x01 (X0)
    let x1 = x01 >> 64; // Upper 64 bits of x01 (X1)

    // Extract the lower and upper 64-bit halves of `x23` (high 128 bits of input)
    let x2 = x23 & ((1 << 64) - 1); // Lower 64 bits of x23 (X2)
    let x3 = x23 >> 64; // Upper 64 bits of x23 (X3)

    // Extract the lower and upper 64 bits of the polynomial
    let poly_low = poly & ((1 << 64) - 1); // Lower 64 bits of poly
    let poly_high = poly >> 64; // Upper 64 bits of poly

    // Compute A1:A0 = X0 • poly
    // Multiply X0 with the polynomial to get A (128-bit product)
    let a = expected_pmull_result(x0, poly_low).0; // Polynomial multiplication result
    let a0 = a & ((1 << 64) - 1); // Lower 64 bits of A (A0)
    let a1 = a >> 64; // Upper 64 bits of A (A1)

    // Compute B1:B0 = [X0 ⊕ A1 : X1 ⊕ A0]
    // XOR the components to compute intermediate results B0 and B1
    let b0 = x0 ^ a1; // Lower 64 bits (B0 = X0 ⊕ A1)
    let b1 = x1 ^ a0; // Upper 64 bits (B1 = X1 ⊕ A0)

    // Compute C1:C0 = B0 • poly
    // Multiply B0 with the polynomial to get C (128-bit product)
    let c = expected_pmull_result(b1, poly_high).0; // Polynomial multiplication result
    let c0 = c & ((1 << 64) - 1); // Lower 64 bits of C (C0)
    let c1 = c >> 64; // Upper 64 bits of C (C1)

    // Compute D1:D0 = [B0 ⊕ C1 : B1 ⊕ C0]
    // XOR the components to compute intermediate results D0 and D1
    let d0 = b0 ^ c0; // Lower 64 bits (D0 = B0 ⊕ C0)
    let d1 = b1 ^ c1; // Upper 64 bits (D1 = B1 ⊕ C1)

    // Compute output [D1 ⊕ X3 : D0 ⊕ X2]
    // XOR the intermediate results with X2 and X3 to get the final output
    let result_high = d1 ^ x3; // Upper 64 bits of the result
    let result_low = d0 ^ x2; // Lower 64 bits of the result

    // Combine the high and low 64-bit parts into a single 128-bit result
    (result_high << 64) | result_low
}

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
/// ```rust,ignore
/// let a = 0b1011; // Polynomial: x^3 + x + 1
/// let b = 0b0110; // Polynomial: x^2 + x
/// assert_eq!(expected_pmull_result(a, b), (0b111010, 0)); // x^5 + x^4 + x^3 + x
/// ```
pub(crate) fn expected_pmull_result(a: u128, b: u128) -> (u128, u128) {
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
