use hashcaster_primitives::{binary_field::BinaryField128b, linear_trait::LinearOperations};

/// Implements the θ (Theta) step for the Keccak permutation.
///
/// The Theta step ensures diffusion across the state array by adjusting columns
/// based on neighboring column values and their rotations. Specifically:
///
/// \( D[x] = C[x-1] + \text{rot}(C[x+1], 1) \)
///
/// ## Overview
/// - **Input:** `C[x]` is the XOR of all lanes in column `x` of the state matrix.
/// - **Output:** `D[x]` is computed for each column based on the values of adjacent columns.
///
/// The `ThetaCD` struct models this operation as a linear operator, splitting the input state
/// into 5 columns of 64-bit lanes each, and performing the Theta step.
///
/// ## Context in Keccak
/// In the Keccak round function:
/// 1. \( C[x] \) is computed as the XOR of all rows in column `x`.
/// 2. \( D[x] \) is derived from \( C[x] \) as an adjustment to ensure diffusion.
/// 3. \( D[x] \) is XORed back into the state matrix to complete the Theta step.
///
/// ## Functions
/// - `apply`: Computes the \( D[x] \) values.
/// - `apply_transposed`: Distributes \( D[x] \) values across the input matrix in a transpose-like
///   operation.
///
/// This implementation adheres to the Keccak specifications as described in the
/// [Keccak Team Documentation](https://keccak.team/keccak_specs_summary.html).
#[derive(Debug)]
pub struct ThetaCD;

impl LinearOperations for ThetaCD {
    /// Returns the number of input elements required (320 lanes).
    ///
    /// ## Details
    /// - The input represents a flattened 5x64 matrix, where each column contains 64-bit lanes.
    fn n_in(&self) -> usize {
        320
    }

    /// Returns the number of output elements produced (320 lanes).
    ///
    /// ## Details
    /// - The output is also a flattened 5x64 matrix, where each column contains 64-bit lanes.
    fn n_out(&self) -> usize {
        320
    }

    /// Computes the \( D[x] \) values based on the input state \( C[x] \).
    ///
    /// ## Parameters
    /// - `input`: A reference to a 320-element array representing the column parity vector \( C[x]
    ///   \).
    /// - `output`: A mutable reference to a 320-element array to store the resulting vector \( D[x]
    ///   \).
    ///
    /// ## Algorithm
    /// For each column `x`:
    /// - Compute the previous column as \( C[x-1] \).
    /// - Compute the rotated next column as \( \text{rot}(C[x+1], 1) \).
    /// - Add these values to form \( D[x] = C[x-1] + \text{rot}(C[x+1], 1) \).
    ///
    /// ## Implementation Details
    /// - Wrap-around for indices is handled using modulo operations.
    /// - Bit rotation shifts values within the next column.
    fn apply(&self, input: &[BinaryField128b], output: &mut [BinaryField128b]) {
        for x in 0..5 {
            // Compute the index for the previous column with wrap-around.
            let t1 = ((x + 4) % 5) * 64;
            // Compute the index for the next column with wrap-around.
            let t2 = ((x + 1) % 5) * 64;
            for z in 0..64 {
                // Add the value from the previous column and the rotated next column.
                output[x * 64 + z] += input[t1 + z] + input[t2 + (z + 63) % 64];
            }
        }
    }

    /// Distributes \( D[x] \) values across the matrix in a transpose-like operation.
    ///
    /// ## Parameters
    /// - `input`: A reference to a 320-element array representing \( D[x] \), the output of the
    ///   Theta step.
    /// - `output`: A mutable reference to a 320-element array to store the distributed \( C[x] \).
    ///
    /// ## Algorithm
    /// For each column `x`:
    /// - Compute the next column as \( C[x+1] \).
    /// - Compute the rotated previous column as \( \text{rot}(C[x-1], -1) \).
    /// - Add these values to form the distributed \( D[x] \) values.
    ///
    /// ## Implementation Details
    /// - Wrap-around for indices is handled using modulo operations.
    /// - Bit rotation shifts values within the previous column.
    fn apply_transposed(&self, input: &[BinaryField128b], output: &mut [BinaryField128b]) {
        for x in 0..5 {
            // Compute the index for the previous column with wrap-around.
            let t1 = ((x + 4) % 5) * 64;
            // Compute the index for the next column with wrap-around.
            let t2 = ((x + 1) % 5) * 64;
            for z in 0..64 {
                // Add the value from the next column and the rotated previous column.
                output[x * 64 + z] += input[t2 + z] + input[t1 + (z + 1) % 64];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hashcaster_primitives::binary_field::BinaryField128b;

    #[test]
    fn test_theta_cd_apply() {
        let theta_cd = ThetaCD;

        let mut input = [BinaryField128b::from(0); 320];
        for i in 0..5 {
            for j in 0..64 {
                input[i * 64 + j] = BinaryField128b::from(1 << j);
            }
        }

        // Initialize the output array to zeros.
        let mut output = [BinaryField128b::ZERO; 320];

        // Apply the ThetaCD transformation.
        theta_cd.apply(&input, &mut output);

        // Validate the output.
        for x in 0..5 {
            for z in 0..64 {
                assert_eq!(
                    output[x * 64 + z],
                    BinaryField128b::from(1 << (z % 64)) +
                        BinaryField128b::from(1 << ((z + 63) % 64))
                );
            }
        }
    }

    #[test]
    #[allow(clippy::identity_op, clippy::erasing_op)]
    fn test_theta_cd_apply_simple() {
        let theta_cd = ThetaCD;

        // Define some constants for the test.
        let c0 = BinaryField128b::from(1);
        let c1 = BinaryField128b::from(2);
        let c2 = BinaryField128b::from(4);
        let c3 = BinaryField128b::from(8);
        let c4 = BinaryField128b::from(16);

        // Define a simple input where each column has a single distinct value.
        // Each 64-bit segment is initialized with the same value for simplicity.
        let mut input = [BinaryField128b::from(0); 320];
        input[0 * 64..1 * 64].fill(c0); // C[0]
        input[1 * 64..2 * 64].fill(c1); // C[1]
        input[2 * 64..3 * 64].fill(c2); // C[2]
        input[3 * 64..4 * 64].fill(c3); // C[3]
        input[4 * 64..5 * 64].fill(c4); // C[4]

        // Initialize the output array to zeros.
        let mut output = [BinaryField128b::ZERO; 320];

        // Apply the ThetaCD transformation.
        theta_cd.apply(&input, &mut output);

        // Validate the output by manually checking values for a small number of lanes.
        // Each D[x] should match: D[x] = C[x-1] + rot(C[x+1], 1)
        assert_eq!(output[0 * 64 + 0], c4 + c1); // D[0][0]
        assert_eq!(output[1 * 64 + 0], c0 + c2); // D[1][0]
        assert_eq!(output[2 * 64 + 0], c1 + c3); // D[2][0]
        assert_eq!(output[3 * 64 + 0], c2 + c4); // D[3][0]
        assert_eq!(output[4 * 64 + 0], c3 + c0); // D[4][0]
    }

    #[test]
    #[allow(clippy::identity_op, clippy::erasing_op)]
    fn test_theta_cd_apply_transposed_simple() {
        let theta_cd = ThetaCD;

        // Define constants for the test.
        let c0 = BinaryField128b::from(1);
        let c1 = BinaryField128b::from(2);
        let c2 = BinaryField128b::from(4);
        let c3 = BinaryField128b::from(8);
        let c4 = BinaryField128b::from(16);

        // Define a simple input where each column has a single distinct value.
        // Each 64-bit segment is initialized with the same value for simplicity.
        let mut input = [BinaryField128b::ZERO; 320];
        input[0 * 64..1 * 64].fill(c0); // C[0]
        input[1 * 64..2 * 64].fill(c1); // C[1]
        input[2 * 64..3 * 64].fill(c2); // C[2]
        input[3 * 64..4 * 64].fill(c3); // C[3]
        input[4 * 64..5 * 64].fill(c4); // C[4]

        // Initialize the output array to zeros.
        let mut output = [BinaryField128b::ZERO; 320];

        // Apply the transposed transformation.
        theta_cd.apply_transposed(&input, &mut output);

        // Validate the output by manually computing expected values.
        // The transposed operation reverses the logic: D[x] = rot(C[x-1], -1) + C[x+1]

        // Expected values:
        assert_eq!(output[0 * 64 + 0], c1 + c4); // D[0][0]
        assert_eq!(output[1 * 64 + 0], c2 + c0); // D[1][0]
        assert_eq!(output[2 * 64 + 0], c3 + c1); // D[2][0]
        assert_eq!(output[3 * 64 + 0], c4 + c2); // D[3][0]
        assert_eq!(output[4 * 64 + 0], c0 + c3); // D[4][0]
    }

    #[test]
    fn test_theta_cd_apply_and_transposed() {
        let theta_cd = ThetaCD;

        // **Generate random input for `apply`**
        // - This represents the column parity vector `C[x]` with 320 random elements.
        let input_apply = BinaryField128b::random_vec(320);

        // **Prepare output storage for the result of `apply`**
        // - This will store the computed `D[x]` values (320 elements).
        let mut output_apply = vec![BinaryField128b::ZERO; 320];

        // **Apply the ThetaCD transformation**
        // - Compute `D[x]` values from the random input `C[x]`.
        theta_cd.apply(&input_apply, &mut output_apply);

        // **Generate random input for `apply_transposed`**
        // - This represents the `D[x]` vector for testing the transposed operation.
        let input_transposed = BinaryField128b::random_vec(320);

        // **Prepare output storage for the result of `apply_transposed`**
        // - This will store the distributed column parity values `C[x]`.
        let mut output_transposed = vec![BinaryField128b::ZERO; 320];

        // **Apply the transposed ThetaCD transformation**
        // - Compute `C[x]` values from the random input `D[x]`.
        theta_cd.apply_transposed(&input_transposed, &mut output_transposed);

        // **Compute dot product of `apply` output and `apply_transposed` input**
        // - This computes `lhs = sum(D[x] * D_transpose_input[x])`.
        let lhs = output_apply
            .iter()
            .zip(input_transposed.iter())
            .fold(BinaryField128b::ZERO, |acc, (a, b)| acc + (*a * b));

        // **Compute dot product of `apply_transposed` output and `apply` input**
        // - This computes `rhs = sum(C[x] * C_transpose_input[x])`.
        let rhs = output_transposed
            .iter()
            .zip(input_apply.iter())
            .fold(BinaryField128b::ZERO, |acc, (a, b)| acc + (*a * b));

        // **Validate the equality of `lhs` and `rhs`**
        // - **Mathematical justification**:
        //   - For a valid linear transformation `T` and its transpose `T^T`, the following must
        //     hold: `<T(x), y> == <x, T^T(y)>`
        //   - Here, `<., .>` represents the dot product.
        // - This equality ensures that the `apply` and `apply_transposed` methods are implemented
        //   correctly.
        assert_eq!(lhs, rhs, "Dot product property violated for apply and apply_transposed");
    }
}
