use crate::{matrix::composition::CombinedMatrix, rho_pi::RhoPi, theta::Theta};
use hashcaster_primitives::{binary_field::BinaryField128b, linear_trait::LinearOperations};
use num_traits::Zero;

/// A combined linear operation for Keccak, integrating RhoPi and Theta transformations.
///
/// ## Overview
/// - `KeccakLinear` encapsulates the RhoPi and Theta transformations as a single linear operator.
/// - These transformations are part of the Keccak permutation, ensuring diffusion and mixing.
#[derive(Debug)]
pub struct KeccakLinear(CombinedMatrix<RhoPi, Theta>);

impl Default for KeccakLinear {
    fn default() -> Self {
        Self::new()
    }
}

impl KeccakLinear {
    /// Creates a new instance of `KeccakLinear` by combining RhoPi and Theta transformations.
    ///
    /// ## Returns
    /// - A new `KeccakLinear` instance.
    pub fn new() -> Self {
        // Combine RhoPi and Theta transformations using `CombinedMatrix`.
        Self(CombinedMatrix::new(RhoPi {}, Theta::new()))
    }
}

impl LinearOperations for KeccakLinear {
    /// Returns the number of input elements required by the transformation.
    ///
    /// ## Details
    /// - The input represents the flattened Keccak state (5 blocks of 1024 elements each).
    fn n_in(&self) -> usize {
        5 * 1024
    }

    /// Returns the number of output elements produced by the transformation.
    ///
    /// ## Details
    /// - The output has the same structure as the input.
    fn n_out(&self) -> usize {
        5 * 1024
    }

    /// Applies the combined RhoPi and Theta transformations to the input state.
    ///
    /// ## Parameters
    /// - `input`: A slice of size `n_in()` representing the input state.
    /// - `output`: A mutable slice of size `n_out()` to store the transformed state.
    fn apply(&self, input: &[BinaryField128b], output: &mut [BinaryField128b]) {
        // Create a 3x1600 intermediate state representation.
        let mut state = [[BinaryField128b::zero(); 1600]; 3];

        // Split the input into blocks and map them to the intermediate state.
        for i in 0..5 {
            let input_ptr = &input[i * 1024..];
            for j in 0..3 {
                // Copy 320 elements for each sub-block from the input to the intermediate state.
                state[j][i * 320..(i + 1) * 320]
                    .copy_from_slice(&input_ptr[j * 320..(j + 1) * 320]);
            }
        }

        // Create a 3x1600 intermediate output state.
        let mut output_state = [[BinaryField128b::zero(); 1600]; 3];

        // Apply the combined RhoPi and Theta transformations to each sub-state.
        for j in 0..3 {
            self.0.apply(&state[j], &mut output_state[j]);
        }

        // Map the transformed intermediate output back to the flattened output vector.
        for i in 0..5 {
            let output_ptr = &mut output[i * 1024..];
            for j in 0..3 {
                // Copy 320 elements from each sub-block of the output state back to the flattened
                // output.
                output_ptr[j * 320..(j + 1) * 320]
                    .copy_from_slice(&output_state[j][i * 320..(i + 1) * 320]);
            }
        }
    }

    /// Applies the inverse of the combined RhoPi and Theta transformations to the input state.
    ///
    /// ## Parameters
    /// - `input`: A slice of size `n_in()` representing the input state.
    /// - `output`: A mutable slice of size `n_out()` to store the transformed state.
    fn apply_transposed(&self, input: &[BinaryField128b], output: &mut [BinaryField128b]) {
        // Create a 3x1600 intermediate state representation.
        let mut state = [[BinaryField128b::zero(); 1600]; 3];

        // Split the input into blocks and map them to the intermediate state.
        for i in 0..5 {
            for j in 0..3 {
                // Copy 320 elements for each sub-block from the input to the intermediate state.
                state[j][i * 320..(i + 1) * 320]
                    .copy_from_slice(&input[i * 1024 + j * 320..i * 1024 + (j + 1) * 320]);
            }
        }

        // Create a 3x1600 intermediate output state.
        let mut output_state = [[BinaryField128b::zero(); 1600]; 3];

        // Apply the inverse transformations to each sub-state.
        for j in 0..3 {
            self.0.apply_transposed(&state[j], &mut output_state[j]);
        }

        // Map the transformed intermediate output back to the flattened output vector.
        for i in 0..5 {
            for j in 0..3 {
                // Copy 320 elements from each sub-block of the output state back to the flattened
                // output.
                output[i * 1024 + j * 320..i * 1024 + (j + 1) * 320]
                    .copy_from_slice(&output_state[j][i * 320..(i + 1) * 320]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::MulAdd;

    #[test]
    fn test_keccak_linear() {
        // Create a new instance of the KeccakLinear operator.
        let keccak_linear = KeccakLinear::new();

        // **Validate dimensions**
        // - The number of input elements (`n_in`) and output elements (`n_out`) must be consistent.
        // - These represent the flattened Keccak state with 5 blocks of 1024 elements each.
        assert_eq!(keccak_linear.n_in(), 5 * 1024);
        assert_eq!(keccak_linear.n_out(), 5 * 1024);

        // **Generate random input vector `a`**
        // - This simulates a realistic input for the KeccakLinear transformation.
        let a: Vec<_> = (0..1024 * 5).map(|_| BinaryField128b::random()).collect();

        // **Apply the KeccakLinear transformation**
        // - `m_a` will store the result of applying the transformation to `a`.
        let mut m_a = vec![BinaryField128b::zero(); 1024 * 5];
        keccak_linear.apply(&a, &mut m_a);

        // **Generate another random input vector `b`**
        // - This will be used to test the transpose operation.
        let b: Vec<_> = (0..1024 * 5).map(|_| BinaryField128b::random()).collect();

        // **Apply the transposed transformation**
        // - `m_b` will store the result of applying the transpose operation to `b`.
        let mut m_b = vec![BinaryField128b::zero(); 1024 * 5];
        keccak_linear.apply_transposed(&b, &mut m_b);

        // **Compute the dot product of the forward-transformed `m_a` with `b`**
        // - This computes `lhs = sum(m_a[i] * b[i])`, where `m_a` is the result of applying the
        //   forward transformation to `a`.
        let lhs = m_a
            .iter()
            .zip(b.iter())
            .fold(BinaryField128b::zero(), |acc, (a, b)| a.mul_add(*b, acc));

        // **Compute the dot product of the transpose-transformed `m_b` with `a`**
        // - This computes `rhs = sum(m_b[i] * a[i])`, where `m_b` is the result of applying the
        //   transposed transformation to `b`.
        let rhs = m_b
            .iter()
            .zip(a.iter())
            .fold(BinaryField128b::zero(), |acc, (a, b)| a.mul_add(*b, acc));

        // **Validate the equality of `lhs` and `rhs`**
        // - **Mathematical justification**:
        //   - For a valid linear operator `A` and its transpose `A^T`:
        //     - `<A(x), y> = <x, A^T(y)>`
        //   - Here, `<., .>` represents the dot product.
        // - This test verifies that the transpose operation is correctly implemented and maintains
        //   the mathematical properties of a linear operator.
        assert_eq!(lhs, rhs);
    }
}
