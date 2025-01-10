use super::idx;
use hashcaster_primitives::{binary_field::BinaryField128b, linear_trait::LinearOperations};

/// Implements the θ (Theta) linear transformation for Keccak.
///
/// ## Overview
/// - **Input for `apply`:** A flattened 5x64 matrix representing the state columns, with each
///   column containing 64-bit lanes.
/// - **Output for `apply`:** A flattened 5x5x64 matrix representing the expanded state, where each
///   column is distributed across 5 rows.
/// - **Input for `apply_transposed`:** A flattened 5x5x64 matrix representing the expanded state.
/// - **Output for `apply_transposed`:** A flattened 5x64 matrix representing the reduced state
///   columns.
///
/// ## Functions
/// - `apply`: Distributes the input column vector \( D[x] \) into the expanded matrix \( E[x, y]
///   \), replicating each column \( D[x] \) across rows.
/// - `apply_transposed`: Reduces the expanded matrix \( E[x, y] \) back into the column vector \(
///   D[x] \), summing up contributions.
///
/// ## Context in Keccak
/// - This step is not a core part of the Keccak permutation but provides a linear operator for
///   state transformations in custom applications.
///
/// ## Adherence to Specifications
/// - This implementation adheres to the principles described in the [Keccak Team Documentation](https://keccak.team/keccak_specs_summary.html).
#[derive(Debug)]
pub struct ThetaDE;

impl LinearOperations for ThetaDE {
    /// Returns the number of input elements expected for the `apply` method (320 elements).
    ///
    /// ## Details
    /// - Represents the flattened 5x64 matrix \( D[x] \).
    /// - Each column consists of 64-bit lanes, and there are 5 columns, resulting in 320 elements.
    fn n_in(&self) -> usize {
        320
    }

    /// Returns the number of output elements produced by the `apply` method (1600 elements).
    ///
    /// ## Details
    /// - Represents the flattened 5x5x64 matrix \( E[x, y] \).
    /// - Each column contains 64-bit lanes, distributed across 5 rows, resulting in 1600 elements.
    fn n_out(&self) -> usize {
        1600
    }

    /// Distributes the column vector \( D[x] \) into the expanded matrix \( E[x, y] \).
    ///
    /// ## Parameters
    /// - `input`: A reference to a 320-element array representing the flattened column vector \(
    ///   D[x] \).
    /// - `output`: A mutable reference to a 1600-element array to store the expanded state matrix
    ///   \( E[x, y] \).
    ///
    /// ## Algorithm
    /// - For each column \( x \), distribute the 64-bit lanes across all rows \( y \).
    /// - \( E[x, y, z] = D[x, z] \)
    ///
    /// ## Implementation Details
    /// - The `idx` helper function computes the index for \( E[x, y, z] \) in the flattened array.
    fn apply(&self, input: &[BinaryField128b], output: &mut [BinaryField128b]) {
        // Iterate over all columns (x).
        for x in 0..5 {
            // Iterate over all rows (y).
            for y in 0..5 {
                // Iterate over all bit positions (z) in the 64-bit lane.
                for z in 0..64 {
                    // Distribute the value of D[x, z] to E[x, y, z].
                    output[idx(x, y, z)] += input[x * 64 + z];
                }
            }
        }
    }

    /// Reduces the expanded matrix \( E[x, y] \) back into the column vector \( D[x] \).
    ///
    /// ## Parameters
    /// - `input`: A reference to a 1600-element array representing the flattened expanded state
    ///   matrix \( E[x, y] \).
    /// - `output`: A mutable reference to a 320-element array to store the reduced column vector \(
    ///   D[x] \).
    ///
    /// ## Algorithm
    /// - For each column \( x \), sum up the contributions from all rows \( y \).
    /// - \( D[x, z] = \sum_y E[x, y, z] \)
    ///
    /// ## Implementation Details
    /// - The `idx` helper function computes the index for \( E[x, y, z] \) in the flattened array.
    fn apply_transposed(&self, input: &[BinaryField128b], output: &mut [BinaryField128b]) {
        // Iterate over all columns (x).
        for x in 0..5 {
            // Iterate over all rows (y).
            for y in 0..5 {
                // Iterate over all bit positions (z) in the 64-bit lane.
                for z in 0..64 {
                    // Sum the value of E[x, y, z] into D[x, z].
                    output[x * 64 + z] += input[idx(x, y, z)];
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hashcaster_primitives::binary_field::BinaryField128b;

    #[test]
    #[allow(clippy::identity_op, clippy::erasing_op)]
    fn test_theta_de_apply() {
        let theta_de = ThetaDE;

        // Define simple input values for manual computation.
        let mut input = [BinaryField128b::from(0); 320];
        input[0 * 64 + 0] = BinaryField128b::from(10);
        input[1 * 64 + 1] = BinaryField128b::from(20);
        input[2 * 64 + 2] = BinaryField128b::from(30);
        input[3 * 64 + 3] = BinaryField128b::from(40);
        input[4 * 64 + 4] = BinaryField128b::from(50);

        // Initialize the output array with zeros.
        let mut output = [BinaryField128b::from(0); 1600];

        // Apply the transformation.
        theta_de.apply(&input, &mut output);

        // Check the output values.
        for x in 0..5 {
            for y in 0..5 {
                for z in 0..64 {
                    if x == 0 && z == 0 {
                        assert_eq!(output[idx(x, y, z)], BinaryField128b::from(10));
                    } else if x == 1 && z == 1 {
                        assert_eq!(output[idx(x, y, z)], BinaryField128b::from(20));
                    } else if x == 2 && z == 2 {
                        assert_eq!(output[idx(x, y, z)], BinaryField128b::from(30));
                    } else if x == 3 && z == 3 {
                        assert_eq!(output[idx(x, y, z)], BinaryField128b::from(40));
                    } else if x == 4 && z == 4 {
                        assert_eq!(output[idx(x, y, z)], BinaryField128b::from(50));
                    } else {
                        assert_eq!(output[idx(x, y, z)], BinaryField128b::from(0));
                    }
                }
            }
        }
    }

    #[test]
    fn test_theta_de_apply_transposed() {
        let theta_de = ThetaDE;

        // Define simple input values for manual computation.
        let mut input = [BinaryField128b::from(0); 1600];
        input[idx(0, 0, 0)] = BinaryField128b::from(10);
        input[idx(1, 1, 1)] = BinaryField128b::from(20);
        input[idx(2, 2, 2)] = BinaryField128b::from(30);
        input[idx(3, 3, 3)] = BinaryField128b::from(40);
        input[idx(4, 4, 4)] = BinaryField128b::from(50);

        // Initialize the output array with zeros.
        let mut output = [BinaryField128b::from(0); 320];

        // Apply the transposed transformation.
        theta_de.apply_transposed(&input, &mut output);

        // Check the output values.
        for x in 0..5 {
            for z in 0..64 {
                if x == 0 && z == 0 {
                    assert_eq!(output[x * 64 + z], BinaryField128b::from(10));
                } else if x == 1 && z == 1 {
                    assert_eq!(output[x * 64 + z], BinaryField128b::from(20));
                } else if x == 2 && z == 2 {
                    assert_eq!(output[x * 64 + z], BinaryField128b::from(30));
                } else if x == 3 && z == 3 {
                    assert_eq!(output[x * 64 + z], BinaryField128b::from(40));
                } else if x == 4 && z == 4 {
                    assert_eq!(output[x * 64 + z], BinaryField128b::from(50));
                } else {
                    assert_eq!(output[x * 64 + z], BinaryField128b::from(0));
                }
            }
        }
    }

    #[test]
    fn test_theta_de_apply_and_transposed() {
        let theta_de = ThetaDE;

        // **Generate random input for `apply`**
        // - This represents the flattened 5x64 matrix `D[x]` with 320 random elements.
        let input_apply: Vec<_> = (0..320).map(|_| BinaryField128b::random()).collect();

        // **Prepare output storage for the result of `apply`**
        // - This will store the expanded matrix `E[x, y, z]` (1600 elements).
        let mut output_apply = vec![BinaryField128b::from(0); 1600];

        // **Apply the ThetaDE transformation**
        // - Compute `E[x, y, z]` values from the random input `D[x]`.
        theta_de.apply(&input_apply, &mut output_apply);

        // **Generate random input for `apply_transposed`**
        // - This represents the expanded matrix `E[x, y, z]` with 1600 random elements.
        let input_transposed: Vec<_> = (0..1600).map(|_| BinaryField128b::random()).collect();

        // **Prepare output storage for the result of `apply_transposed`**
        // - This will store the reduced matrix `D[x, z]` (320 elements).
        let mut output_transposed = vec![BinaryField128b::from(0); 320];

        // **Apply the transposed ThetaDE transformation**
        // - Compute `D[x, z]` values from the random input `E[x, y, z]`.
        theta_de.apply_transposed(&input_transposed, &mut output_transposed);

        // **Compute dot product of `apply` output and `apply_transposed` input**
        // - This computes `lhs = sum(E[x, y, z] * E_transposed_input[x, y, z])`.
        let lhs = output_apply
            .iter()
            .zip(input_transposed.iter())
            .fold(BinaryField128b::from(0), |acc, (a, b)| acc + (*a * *b));

        // **Compute dot product of `apply_transposed` output and `apply` input**
        // - This computes `rhs = sum(D[x, z] * D_transpose_input[x, z])`.
        let rhs = output_transposed
            .iter()
            .zip(input_apply.iter())
            .fold(BinaryField128b::from(0), |acc, (a, b)| acc + (*a * *b));

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
