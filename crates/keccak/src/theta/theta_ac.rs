use super::idx;
use hashcaster_primitives::{binary_field::BinaryField128b, linear_trait::LinearOperations};

/// Implements the θ (Theta) linear transformation step for Keccak.
///
/// This transformation computes column parity values \( C[x] \) based on the input matrix \( A[x,
/// y] \).
///
/// ## Overview
/// - **Input**: A Keccak state matrix \( A[x, y] \), represented as a flattened array of 1600
///   elements.
/// - **Output**: A column parity vector \( C[x] \), represented as a flattened array of 320
///   elements.
///
/// The \( \theta \) step is the first step in each Keccak round and plays a key role in ensuring
/// diffusion across the state. The operation \( C[x] = A[x, 0] + A[x, 1] + A[x, 2] + A[x, 3] + A[x,
/// 4] \) aggregates the XOR of all rows in column \( x \) for each bit position \( z \) in the
/// 64-bit lanes.
///
/// ## Context in Keccak
/// In the Keccak round function, \( C[x] \) is used to compute the intermediate state adjustments
/// \( D[x] \): \[ D[x] = C[x-1] \oplus \text{rot}(C[x+1], 1) \]
/// The intermediate \( D[x] \) is then XORed back into the state matrix \( A[x, y] \).
///
/// ## Functions
/// - `apply`: Computes the \( C[x] \) values.
/// - `apply_transposed`: Distributes \( C[x] \) values back into \( A[x, y] \) without reversing
///   the transformation.
///
/// This implementation adheres to the Keccak specifications as described in the
/// [Keccak Team Documentation](https://keccak.team/keccak_specs_summary.html).
#[derive(Debug)]
pub struct ThetaAC;

impl LinearOperations for ThetaAC {
    /// Returns the number of input elements expected (1600 elements).
    ///
    /// ## Details
    /// - The input represents the flattened 5x5 Keccak state matrix \( A[x, y] \).
    /// - Each lane of the matrix consists of 64 bits, resulting in 1600 elements.
    fn n_in(&self) -> usize {
        1600
    }

    /// Returns the number of output elements produced (320 elements).
    ///
    /// ## Details
    /// - The output represents the flattened column parity vector \( C[x] \).
    /// - Each column consists of 64-bit lanes, and there are 5 columns, resulting in 320 elements.
    fn n_out(&self) -> usize {
        320
    }

    /// Computes the column parity vector \( C[x] \) from the input matrix \( A[x, y] \).
    ///
    /// ## Parameters
    /// - `input`: A reference to an array of 1600 `BinaryField128b` elements representing the
    ///   flattened Keccak state matrix \( A[x, y] \).
    /// - `output`: A mutable reference to an array of 320 `BinaryField128b` elements to store the
    ///   resulting column parity vector \( C[x] \).
    ///
    /// ## Algorithm
    /// - Iterate over all 5 columns (`x`) of the state.
    /// - For each column, iterate over all 5 rows (`y`).
    /// - Compute the XOR of all rows in the column for each bit position (`z`) in the 64-bit lane.
    ///
    /// ## Implementation Details
    /// - The `idx` helper function computes the flattened index for \( A[x, y, z] \) in the input
    ///   array.
    fn apply(&self, input: &[BinaryField128b], output: &mut [BinaryField128b]) {
        for x in 0..5 {
            for y in 0..5 {
                for z in 0..64 {
                    // Add the value of A[x, y, z] to the column parity C[x, z].
                    output[x * 64 + z] += input[idx(x, y, z)];
                }
            }
        }
    }

    /// Distributes the column parity vector \( C[x] \) back into the state matrix \( A[x, y] \).
    ///
    /// This operation does not reverse the \( \theta \) step but instead spreads the column parity
    /// values \( C[x] \) uniformly across the rows of the state matrix \( A[x, y] \).
    ///
    /// ## Parameters
    /// - `input`: A reference to an array of 320 `BinaryField128b` elements representing the column
    ///   parity vector \( C[x] \).
    /// - `output`: A mutable reference to an array of 1600 `BinaryField128b` elements to store the
    ///   distributed state matrix \( A[x, y] \).
    ///
    /// ## Algorithm
    /// - Iterate over all 5 columns (`x`) of the state.
    /// - For each column, iterate over all 5 rows (`y`).
    /// - Distribute the \( C[x, z] \) value to all corresponding positions in \( A[x, y, z] \).
    ///
    /// ## Implementation Details
    /// - The `idx` helper function computes the flattened index for \( A[x, y, z] \) in the output
    ///   array.
    fn apply_transposed(&self, input: &[BinaryField128b], output: &mut [BinaryField128b]) {
        for x in 0..5 {
            for y in 0..5 {
                for z in 0..64 {
                    // Add the value of C[x, z] back to the corresponding position in A[x, y, z].
                    output[idx(x, y, z)] += input[x * 64 + z];
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hashcaster_primitives::binary_field::BinaryField128b;
    use rand::rngs::OsRng;

    #[test]
    fn test_thetaac_apply() {
        let rng = &mut OsRng;

        let theta_ac = ThetaAC;

        // Initialize input with a random value repeated 1600 times.
        let i1 = BinaryField128b::random(rng);
        let input = [i1; 1600];

        // Initialize output with zeros.
        let mut output = [BinaryField128b::from(0); 320];

        // Apply the transformation.
        theta_ac.apply(&input, &mut output);

        // Validate that output[x * 64 + z] = (i1 + i1 + i1 + i1 + i1) for all x, z
        //
        // Each output element should be the sum of 5 input elements from the same row across
        // columns.
        for x in 0..5 {
            for z in 0..64 {
                assert_eq!(output[x * 64 + z], i1 + i1 + i1 + i1 + i1);
            }
        }
    }

    #[test]
    fn test_thetaac_apply_transposed() {
        let rng = &mut OsRng;

        let theta_ac = ThetaAC;

        // Initialize input with a random value repeated 320 times.
        let i1 = BinaryField128b::random(rng);
        let input = [i1; 320];

        // Initialize output with zeros.
        let mut output = [BinaryField128b::from(0); 1600];

        // Apply the transposed transformation.
        theta_ac.apply_transposed(&input, &mut output);

        // Validate that output[idx(x, y, z)] = i1 for all x, y, z
        //
        // Each output element should match the input value for all corresponding indices.
        for x in 0..5 {
            for y in 0..5 {
                for z in 0..64 {
                    assert_eq!(output[idx(x, y, z)], i1);
                }
            }
        }
    }

    #[test]
    #[allow(clippy::identity_op, clippy::erasing_op)]
    fn test_thetaac_apply_first_index() {
        let theta_ac = ThetaAC;

        // Define simple input values for manual computation.
        let mut input = [BinaryField128b::from(0); 1600];
        input[idx(0, 0, 0)] = BinaryField128b::from(1);
        input[idx(0, 1, 0)] = BinaryField128b::from(2);
        input[idx(0, 2, 0)] = BinaryField128b::from(3);
        input[idx(0, 3, 0)] = BinaryField128b::from(4);
        input[idx(0, 4, 0)] = BinaryField128b::from(5);

        // Initialize the output array with zeros.
        let mut output = [BinaryField128b::from(0); 320];

        // Apply the transformation.
        theta_ac.apply(&input, &mut output);

        // Manually compute the expected value for C[0, 0].
        let expected = BinaryField128b::from(1) +
            BinaryField128b::from(2) +
            BinaryField128b::from(3) +
            BinaryField128b::from(4) +
            BinaryField128b::from(5);

        // Validate the result for the manually computed value.
        assert_eq!(output[0 * 64 + 0], expected);

        // Additional checks for unused indices should remain zero.
        for z in 1..64 {
            assert_eq!(output[0 * 64 + z], BinaryField128b::from(0));
        }
    }

    #[test]
    #[allow(clippy::identity_op, clippy::erasing_op)]
    fn test_thetaac_apply_transposed_first_index() {
        let theta_ac = ThetaAC;

        // Define simple input values for manual computation.
        let mut input = [BinaryField128b::from(0); 320];
        input[0 * 64 + 0] = BinaryField128b::from(10);

        // Initialize the output array with zeros.
        let mut output = [BinaryField128b::from(0); 1600];

        // Apply the transposed transformation.
        theta_ac.apply_transposed(&input, &mut output);

        // Manually compute the expected values for A[0, 0, 0] to A[0, 4, 0].
        let expected = BinaryField128b::from(10);

        // Validate the result for all lanes in the corresponding column.
        for y in 0..5 {
            assert_eq!(output[idx(0, y, 0)], expected);
        }

        // Additional checks for unused indices should remain zero.
        for x in 1..5 {
            for y in 0..5 {
                for z in 0..64 {
                    assert_eq!(output[idx(x, y, z)], BinaryField128b::from(0));
                }
            }
        }
    }

    #[test]
    fn test_thetaac_apply_and_transpose() {
        let theta_ac = ThetaAC;

        // Generate a random input for `apply`.
        let input = BinaryField128b::random_vec(1600);

        // Prepare output storage for the result of `apply`.
        let mut output_apply = vec![BinaryField128b::from(0); 320];

        // Apply the `apply` transformation.
        theta_ac.apply(&input, &mut output_apply);

        // Generate another random input for `apply_transposed`.
        let transposed_input = BinaryField128b::random_vec(320);

        // Prepare output storage for the result of `apply_transposed`.
        let mut output_transposed = vec![BinaryField128b::from(0); 1600];

        // Apply the `apply_transposed` transformation.
        theta_ac.apply_transposed(&transposed_input, &mut output_transposed);

        // Compute the dot product of the `apply` output and the input to `apply_transposed`.
        let lhs = output_apply
            .iter()
            .zip(transposed_input.iter())
            .fold(BinaryField128b::from(0), |acc, (a, b)| acc + (*a * b));

        // Compute the dot product of the `apply_transposed` output and the input to `apply`.
        let rhs = output_transposed
            .iter()
            .zip(input.iter())
            .fold(BinaryField128b::from(0), |acc, (a, b)| acc + (*a * b));

        // Validate the mathematical equivalence of the transformations.
        //
        // ### Explanation:
        // For a valid linear transformation `T` and its transpose `T^T`, the following equality
        // holds:
        // - `<T(x), y> == <x, T^T(y)>`
        // where `<., .>` represents the dot product.
        // This test ensures that the `apply` and `apply_transposed` methods correctly implement
        // this property.
        assert_eq!(lhs, rhs, "Dot product property violated for apply and apply_transposed");
    }
}
