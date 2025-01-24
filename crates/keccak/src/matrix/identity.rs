use hashcaster_primitives::{binary_field::BinaryField128b, linear_trait::LinearOperations};

/// Represents an identity matrix of size `size`.
///
/// The identity matrix has the property that when applied to an input vector,
/// it outputs the same vector. This struct is a minimalistic representation
/// that does not explicitly store matrix elements but works through size-based
/// constraints and operations.
///
/// # Fields
/// - `0`: The size of the matrix, representing both its input and output dimensions.
#[derive(Debug)]
pub struct IdentityMatrix(usize);

impl IdentityMatrix {
    /// Creates a new identity matrix of the given size.
    ///
    /// # Parameters
    /// - `size`: The size of the matrix, determining the number of input/output elements.
    ///
    /// # Returns
    /// A new `IdentityMatrix` instance.
    pub const fn new(size: usize) -> Self {
        Self(size)
    }
}

impl LinearOperations for IdentityMatrix {
    fn n_in(&self) -> usize {
        self.0
    }

    fn n_out(&self) -> usize {
        self.0
    }

    fn apply(&self, input: &[BinaryField128b], output: &mut [BinaryField128b]) {
        // Ensure input has the correct size.
        assert_eq!(input.len(), self.0, "Input size mismatch");
        // Ensure output has the correct size.
        assert_eq!(output.len(), self.0, "Output size mismatch");

        // Iterate over corresponding elements of `input` and `output`.
        output.iter_mut().zip(input.iter()).for_each(|(o, i)| *o += *i);
    }

    fn apply_transposed(&self, input: &[BinaryField128b], output: &mut [BinaryField128b]) {
        self.apply(input, output);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hashcaster_primitives::binary_field::BinaryField128b;

    #[test]
    fn test_identity_matrix_new() {
        let matrix = IdentityMatrix::new(4);
        assert_eq!(matrix.n_in(), 4);
        assert_eq!(matrix.n_out(), 4);
    }

    #[test]
    fn test_identity_matrix_apply() {
        let matrix = IdentityMatrix::new(4);
        let input = [
            BinaryField128b::from(1),
            BinaryField128b::from(2),
            BinaryField128b::from(3),
            BinaryField128b::from(4),
        ];
        let mut output = [
            BinaryField128b::from(0),
            BinaryField128b::from(0),
            BinaryField128b::from(0),
            BinaryField128b::from(0),
        ];

        matrix.apply(&input, &mut output);

        assert_eq!(output, input);
    }

    #[test]
    fn test_identity_matrix_apply_transposed() {
        let matrix = IdentityMatrix::new(4);
        let input = [
            BinaryField128b::from(1),
            BinaryField128b::from(2),
            BinaryField128b::from(3),
            BinaryField128b::from(4),
        ];
        let mut output = [
            BinaryField128b::from(0),
            BinaryField128b::from(0),
            BinaryField128b::from(0),
            BinaryField128b::from(0),
        ];

        matrix.apply_transposed(&input, &mut output);

        assert_eq!(output, input);
    }

    #[test]
    #[should_panic(expected = "Input size mismatch")]
    fn test_identity_matrix_apply_invalid_size() {
        let matrix = IdentityMatrix::new(4);
        // Incorrect size
        let input = [BinaryField128b::from(1), BinaryField128b::from(2), BinaryField128b::from(3)];
        let mut output = [
            BinaryField128b::from(0),
            BinaryField128b::from(0),
            BinaryField128b::from(0),
            BinaryField128b::from(0),
        ];

        matrix.apply(&input, &mut output);
    }

    #[test]
    fn test_identity_matrix_apply_and_transposed() {
        // **Create an IdentityMatrix instance**
        // - This matrix has a size of 5, meaning both its input and output vectors have 5 elements.
        let matrix = IdentityMatrix::new(5);

        // **Generate random input for `apply`**
        // - This represents the input to the identity operation.
        let input_apply = BinaryField128b::random_vec(5);

        // **Prepare output storage for the result of `apply`**
        // - This will store the result of applying the identity operation.
        let mut output_apply = vec![BinaryField128b::from(0); 5];

        // **Apply the IdentityMatrix transformation**
        // - Compute the output by applying the identity operation to the random input.
        matrix.apply(&input_apply, &mut output_apply);

        // **Generate random input for `apply_transposed`**
        // - For the identity matrix, `apply_transposed` should behave identically to `apply`.
        let input_transposed = BinaryField128b::random_vec(5);

        // **Prepare output storage for the result of `apply_transposed`**
        // - This will store the result of applying the transposed operation.
        let mut output_transposed = vec![BinaryField128b::from(0); 5];

        // **Apply the transposed IdentityMatrix transformation**
        // - Compute the output by applying the transposed operation to the random input.
        matrix.apply_transposed(&input_transposed, &mut output_transposed);

        // **Compute dot product of `apply` output and `apply_transposed` input**
        // - This computes `lhs = sum(output_apply[i] * input_transposed[i])`.
        let lhs = output_apply
            .iter()
            .zip(input_transposed.iter())
            .fold(BinaryField128b::from(0), |acc, (a, b)| acc + (*a * *b));

        // **Compute dot product of `apply_transposed` output and `apply` input**
        // - This computes `rhs = sum(output_transposed[i] * input_apply[i])`.
        let rhs = output_transposed
            .iter()
            .zip(input_apply.iter())
            .fold(BinaryField128b::from(0), |acc, (a, b)| acc + (*a * *b));

        // **Validate the equality of `lhs` and `rhs`**
        // - **Mathematical justification**:
        //   - For an identity matrix `I`, the transpose is identical: `I^T = I`.
        //   - Thus, `<I(x), y> == <x, I^T(y)>` must hold true.
        // - This ensures the `apply` and `apply_transposed` methods behave correctly.
        assert_eq!(lhs, rhs, "Dot product property violated for apply and apply_transposed");
    }
}
