use hashcaster_primitives::{binary_field::BinaryField128b, linear_trait::LinearOperations};

/// A struct that combines two linear operations, `A` and `B`, to act as a single operation.
#[derive(Debug)]
pub struct SumMatrix<A: LinearOperations, B: LinearOperations> {
    /// The first linear operator.
    a: A,
    /// The second linear operator.
    b: B,
}

impl<A: LinearOperations, B: LinearOperations> SumMatrix<A, B> {
    /// Creates a new `SumMatrix` by combining two linear operators, `A` and `B`.
    ///
    /// # Parameters
    /// - `a`: The first linear operator.
    /// - `b`: The second linear operator.
    ///
    /// # Panics
    /// - Panics if the input or output dimensions of `A` and `B` do not match.
    pub fn new(a: A, b: B) -> Self {
        assert_eq!(b.n_in(), a.n_in(), "Input dimensions do not match");
        assert_eq!(b.n_out(), a.n_out(), "Output dimensions do not match");
        Self { a, b }
    }
}

impl<A: LinearOperations, B: LinearOperations> LinearOperations for SumMatrix<A, B> {
    fn n_in(&self) -> usize {
        self.a.n_in()
    }

    fn n_out(&self) -> usize {
        self.a.n_out()
    }

    /// Applies the combined forward operation.
    ///
    /// - Applies the forward transformation of `A` and adds its result to `output`.
    /// - Applies the forward transformation of `B` and adds its result to `output`.
    ///
    /// # Parameters
    /// - `input`: The input vector, with size `n_in()`.
    /// - `output`: The output vector, with size `n_out()`.
    ///
    /// # Panics
    /// - Panics if `input.len() != n_in()` or `output.len() != n_out()`.
    fn apply(&self, input: &[BinaryField128b], output: &mut [BinaryField128b]) {
        self.a.apply(input, output);
        self.b.apply(input, output);
    }

    /// Applies the combined transposed operation.
    ///
    /// - Applies the transposed transformation of `A` and adds its result to `output`.
    /// - Applies the transposed transformation of `B` and adds its result to `output`.
    ///
    /// # Parameters
    /// - `input`: The input vector, with size `n_out()`.
    /// - `output`: The output vector, with size `n_in()`.
    ///
    /// # Panics
    /// - Panics if `input.len() != n_out()` or `output.len() != n_in()`.
    fn apply_transposed(&self, input: &[BinaryField128b], output: &mut [BinaryField128b]) {
        self.a.apply_transposed(input, output);
        self.b.apply_transposed(input, output);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hashcaster_primitives::{binary_field::BinaryField128b, linear_trait::LinearOperations};

    struct DummyLinear {
        size: usize,
    }

    impl LinearOperations for DummyLinear {
        fn n_in(&self) -> usize {
            self.size
        }

        fn n_out(&self) -> usize {
            self.size
        }

        fn apply(&self, input: &[BinaryField128b], output: &mut [BinaryField128b]) {
            output.iter_mut().zip(input.iter()).for_each(|(o, i)| *o += *i);
        }

        fn apply_transposed(&self, input: &[BinaryField128b], output: &mut [BinaryField128b]) {
            output.iter_mut().zip(input.iter()).for_each(|(o, i)| *o += *i);
        }
    }

    #[test]
    fn test_combined_matrix_apply() {
        let a = DummyLinear { size: 3 };
        let b = DummyLinear { size: 3 };
        let combined = SumMatrix::new(a, b);

        let input = vec![BinaryField128b::from(1); 3];
        let mut output = vec![BinaryField128b::from(0); 3];

        combined.apply(&input, &mut output);

        // Each operation adds the input.
        assert_eq!(output, vec![BinaryField128b::from(1) + BinaryField128b::from(1); 3]);
    }

    #[test]
    fn test_combined_matrix_apply_transposed() {
        let a = DummyLinear { size: 3 };
        let b = DummyLinear { size: 3 };
        let combined = SumMatrix::new(a, b);

        let input = vec![BinaryField128b::from(1); 3];
        let mut output = vec![BinaryField128b::from(0); 3];

        combined.apply_transposed(&input, &mut output);

        // Each operation adds the input.
        assert_eq!(output, vec![BinaryField128b::from(1) + BinaryField128b::from(1); 3]);
    }

    #[test]
    #[should_panic(expected = "Input dimensions do not match")]
    fn test_combined_matrix_invalid_dimensions() {
        let a = DummyLinear { size: 3 };
        let b = DummyLinear { size: 4 };
        SumMatrix::new(a, b); // This should panic due to mismatched dimensions.
    }

    #[test]
    fn test_sum_matrix_apply_and_transposed() {
        // **Create two dummy linear transformations**
        // - Both `a` and `b` operate on inputs and outputs of size 4.
        let a = DummyLinear { size: 4 };
        let b = DummyLinear { size: 4 };

        // **Combine the transformations into a `SumMatrix`**
        // - `SumMatrix` models the operation `A(x) + B(x)`.
        let sum_matrix = SumMatrix::new(a, b);

        // **Generate random input for `apply`**
        // - This represents the input to the combined operation.
        let input_apply: Vec<_> = (0..4).map(|_| BinaryField128b::random()).collect();

        // **Prepare output storage for the result of `apply`**
        // - This will store the result of `A(x) + B(x)`.
        let mut output_apply = vec![BinaryField128b::from(0); 4];

        // **Apply the SumMatrix transformation**
        // - Compute `A(x) + B(x)` for the random input `x`.
        sum_matrix.apply(&input_apply, &mut output_apply);

        // **Generate random input for `apply_transposed`**
        // - This represents the input to the transposed operation.
        let input_transposed: Vec<_> = (0..4).map(|_| BinaryField128b::random()).collect();

        // **Prepare output storage for the result of `apply_transposed`**
        // - This will store the result of the transposed operation.
        let mut output_transposed = vec![BinaryField128b::from(0); 4];

        // **Apply the transposed SumMatrix transformation**
        // - Compute `A^T(y) + B^T(y)` for the random input `y`.
        sum_matrix.apply_transposed(&input_transposed, &mut output_transposed);

        // **Compute dot product of `apply` output and `apply_transposed` input**
        // - This computes `lhs = sum((A(x) + B(x)) * y)`.
        let lhs = output_apply
            .iter()
            .zip(input_transposed.iter())
            .fold(BinaryField128b::from(0), |acc, (a, b)| acc + (*a * *b));

        // **Compute dot product of `apply_transposed` output and `apply` input**
        // - This computes `rhs = sum(x * (A^T(y) + B^T(y)))`.
        let rhs = output_transposed
            .iter()
            .zip(input_apply.iter())
            .fold(BinaryField128b::from(0), |acc, (a, b)| acc + (*a * *b));

        // **Validate the equality of `lhs` and `rhs`**
        // - **Mathematical justification**:
        //   - For a valid linear operator `T` and its transpose `T^T`, the following must hold:
        //     `<T(x), y> == <x, T^T(y)>`
        //   - Here, `T` is the combined operation `A(x) + B(x)`, and `T^T` is `A^T(y) + B^T(y)`.
        // - This equality ensures that the combined `apply` and `apply_transposed` methods maintain
        //   the transpose property.
        assert_eq!(lhs, rhs, "Dot product property violated for apply and apply_transposed");
    }
}
