use crate::binary_field::BinaryField128b;
use num_traits::Zero;

/// A struct representing a linear transformation using a matrix.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MatrixLinear {
    /// The number of input dimensions (columns of the matrix).
    n_in: usize,
    /// The number of output dimensions (rows of the matrix).
    n_out: usize,
    /// The flattened matrix entries in row-major order.
    entries: Vec<BinaryField128b>,
}

impl MatrixLinear {
    /// Creates a new [`MatrixLinear`] from a flattened vector of entries.
    ///
    /// # Arguments
    /// * `n_in` - Number of input dimensions (columns).
    /// * `n_out` - Number of output dimensions (rows).
    /// * `entries` - A flattened vector of matrix entries in row-major order. The length must be
    ///   `n_in * n_out`.
    ///
    /// # Panics
    /// This function will panic if the length of `entries` is not equal to `n_in * n_out`.
    pub fn new(n_in: usize, n_out: usize, entries: Vec<BinaryField128b>) -> Self {
        assert_eq!(entries.len(), n_in * n_out, "Invalid matrix dimensions");
        Self { n_in, n_out, entries }
    }

    /// Returns the number of input dimensions (columns).
    pub const fn n_in(&self) -> usize {
        self.n_in
    }

    /// Returns the number of output dimensions (rows).
    pub const fn n_out(&self) -> usize {
        self.n_out
    }

    /// Applies the matrix transformation to the input vector.
    ///
    /// # Arguments
    /// * `input` - A slice representing the input vector.
    /// * `output` - A mutable slice where the result of the transformation will be stored.
    ///
    /// # Panics
    /// This function will panic if the length of `input` is not equal to `n_in` or if the length of
    /// `output` is not equal to `n_out`.
    pub fn apply(&self, input: &[BinaryField128b], output: &mut [BinaryField128b]) {
        // Check the dimensions of the input and output vectors.
        assert_eq!(input.len(), self.n_in, "Input vector size mismatch");
        assert_eq!(output.len(), self.n_out, "Output vector size mismatch");

        // Initialize the output vector to zero.
        output.iter_mut().for_each(|o| *o = BinaryField128b::zero());

        // Perform matrix-vector multiplication.
        self.entries.chunks_exact(self.n_in).zip(output.iter_mut()).for_each(|(row, out_elem)| {
            row.iter()
                .zip(input.iter())
                .for_each(|(&col, &input_elem)| *out_elem += col * input_elem);
        });
    }

    /// Applies the transposed matrix transformation to the input vector.
    ///
    /// # Arguments
    /// * `input` - A slice representing the input vector for the transpose.
    /// * `output` - A mutable slice where the result of the transformation will be stored.
    ///
    /// # Panics
    /// This function will panic if the length of `input` is not equal to `n_out` or if the length
    /// of `output` is not equal to `n_in`.
    pub fn apply_transposed(&self, input: &[BinaryField128b], output: &mut [BinaryField128b]) {
        // Check the dimensions of the input and output vectors.
        assert_eq!(input.len(), self.n_out, "Input vector size mismatch");
        assert_eq!(output.len(), self.n_in, "Output vector size mismatch");

        // Initialize the output vector to zero.
        output.iter_mut().for_each(|o| *o = BinaryField128b::zero());

        // Perform matrix-vector multiplication with the transposed matrix.
        input.iter().enumerate().for_each(|(i, &input_elem)| {
            self.entries[i * self.n_in..(i + 1) * self.n_in]
                .iter()
                .zip(output.iter_mut())
                .for_each(|(&entry, out)| *out += entry * input_elem);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binary_field::BinaryField128b;
    use num_traits::One;

    #[test]
    fn test_apply_identity_matrix() {
        // Identity matrix: 2x2
        let matrix_rows = vec![
            vec![BinaryField128b::one(), BinaryField128b::from(0)], // Row 1
            vec![BinaryField128b::from(0), BinaryField128b::one()], // Row 2
        ];

        // Flatten the 2D vector into a single vector
        let flattened_matrix: Vec<BinaryField128b> = matrix_rows.into_iter().flatten().collect();

        // Create the identity matrix with 2 inputs and 2 outputs
        let matrix = MatrixLinear::new(2, 2, flattened_matrix);

        // Define an input vector for the matrix multiplication
        let input = vec![BinaryField128b::from(5), BinaryField128b::from(7)];
        // Create an output vector to store the multiplication result
        let mut output = vec![BinaryField128b::zero(); 2];

        // Apply the identity matrix to the input vector
        matrix.apply(&input, &mut output);

        // Verify the output is identical to the input for an identity matrix
        assert_eq!(output, input);
    }

    #[test]
    fn test_apply_transposed_identity_matrix() {
        // Identity matrix: 2x2
        let matrix_rows = vec![
            vec![BinaryField128b::one(), BinaryField128b::from(0)], // Row 1
            vec![BinaryField128b::from(0), BinaryField128b::one()], // Row 2
        ];

        // Flatten the 2D vector into a single vector
        let flattened_matrix: Vec<BinaryField128b> = matrix_rows.into_iter().flatten().collect();

        // Create the identity matrix with 2 inputs and 2 outputs
        let matrix = MatrixLinear::new(2, 2, flattened_matrix);

        // Define an input vector for the transposed application
        let input = vec![BinaryField128b::from(3), BinaryField128b::from(4)];
        // Create an output vector to store the transposed multiplication result
        let mut output = vec![BinaryField128b::zero(); 2];

        // Apply the transposed identity matrix to the input vector
        matrix.apply_transposed(&input, &mut output);

        // Verify the output is identical to the input for an identity matrix
        assert_eq!(output, input);
    }

    #[test]
    fn test_apply_arbitrary_matrix() {
        // Arbitrary 3x2 matrix
        let a1 = BinaryField128b::from(2);
        let a2 = BinaryField128b::from(3);
        let b1 = BinaryField128b::from(1);
        let b2 = BinaryField128b::from(4);
        let c1 = BinaryField128b::from(0);
        let c2 = BinaryField128b::from(5);
        let matrix_rows = vec![
            vec![a1, a2], // Row 1
            vec![b1, b2], // Row 2
            vec![c1, c2], // Row 3
        ];

        // Flatten the 2D vector into a single vector
        let flattened_matrix: Vec<BinaryField128b> = matrix_rows.into_iter().flatten().collect();

        // Create the matrix with 2 inputs and 3 outputs
        let matrix = MatrixLinear::new(2, 3, flattened_matrix);

        // Define an input vector for the matrix multiplication
        let d1 = BinaryField128b::from(1);
        let d2 = BinaryField128b::from(2);
        let input = vec![d1, d2];
        // Create an output vector to store the multiplication result
        let mut output = vec![BinaryField128b::zero(); 3];

        // Apply the arbitrary matrix to the input vector
        matrix.apply(&input, &mut output);

        // Verify the output matches the expected result:
        // [a1*d1 + a2*d2, b1*d1 + b2*d2, c1*d1 + c2*d2] = [8, 11, 10]
        assert_eq!(output, vec![a1 * d1 + a2 * d2, b1 * d1 + b2 * d2, c1 * d1 + c2 * d2]);
    }

    #[test]
    fn test_apply_transposed_arbitrary_matrix() {
        // Arbitrary 3x2 matrix
        let a1 = BinaryField128b::from(2);
        let a2 = BinaryField128b::from(3);
        let b1 = BinaryField128b::from(1);
        let b2 = BinaryField128b::from(4);
        let c1 = BinaryField128b::from(0);
        let c2 = BinaryField128b::from(5);
        let matrix_rows = vec![
            vec![a1, a2], // Row 1
            vec![b1, b2], // Row 2
            vec![c1, c2], // Row 3
        ];

        // Flatten the 2D vector into a single vector
        let flattened_matrix: Vec<BinaryField128b> = matrix_rows.into_iter().flatten().collect();

        // Create the matrix with 2 inputs and 3 outputs
        let matrix = MatrixLinear::new(2, 3, flattened_matrix);

        // Define an input vector for the transposed application
        let d1 = BinaryField128b::from(1);
        let d2 = BinaryField128b::from(2);
        let d3 = BinaryField128b::from(3);
        let input = vec![d1, d2, d3];
        // Create an output vector to store the transposed multiplication result
        let mut output = vec![BinaryField128b::zero(); 2];

        // Apply the transposed arbitrary matrix to the input vector
        matrix.apply_transposed(&input, &mut output);

        // Verify the output matches the expected result:
        // [a1*d1 + b1*d2 + c1*d3, a2*d1 + b2*d2 + c2*d3] = [4, 26]
        assert_eq!(output, vec![a1 * d1 + b1 * d2 + c1 * d3, a2 * d1 + b2 * d2 + c2 * d3]);
    }

    #[test]
    #[should_panic(expected = "Input vector size mismatch")]
    fn test_apply_invalid_input_size() {
        // Matrix: 2x2
        let matrix_rows = vec![
            vec![BinaryField128b::from(1), BinaryField128b::from(0)], // Row 1
            vec![BinaryField128b::from(0), BinaryField128b::from(1)], // Row 2
        ];

        // Flatten the 2D vector into a single vector
        let flattened_matrix: Vec<BinaryField128b> = matrix_rows.into_iter().flatten().collect();

        // Create the identity matrix with 2 inputs and 2 outputs
        let matrix = MatrixLinear::new(2, 2, flattened_matrix);

        // Define an invalid input vector with size mismatch
        let input = vec![BinaryField128b::from(1)];
        // Create an output vector to store the multiplication result
        let mut output = vec![BinaryField128b::zero(); 2];

        // Attempt to apply the matrix with an invalid input size, which should panic
        matrix.apply(&input, &mut output);
    }

    #[test]
    #[should_panic(expected = "Output vector size mismatch")]
    fn test_apply_invalid_output_size() {
        // Matrix: 2x2
        let matrix_rows = vec![
            vec![BinaryField128b::from(1), BinaryField128b::from(0)], // Row 1
            vec![BinaryField128b::from(0), BinaryField128b::from(1)], // Row 2
        ];

        // Flatten the 2D vector into a single vector
        let flattened_matrix: Vec<BinaryField128b> = matrix_rows.into_iter().flatten().collect();

        // Create the identity matrix with 2 inputs and 2 outputs
        let matrix = MatrixLinear::new(2, 2, flattened_matrix);

        // Define a valid input vector for the matrix multiplication
        let input = vec![BinaryField128b::from(1), BinaryField128b::from(2)];
        // Create an invalid output vector with size mismatch
        let mut output = vec![BinaryField128b::zero(); 1];

        // Attempt to apply the matrix with an invalid output size, which should panic
        matrix.apply(&input, &mut output);
    }
}
