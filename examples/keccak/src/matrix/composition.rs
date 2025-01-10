use hashcaster_primitives::{binary_field::BinaryField128b, linear_trait::LinearOperations};
use num_traits::Zero;

/// A struct representing the composition of two linear operations, `A` and `B`.
///
/// The `CombinedMatrix` struct models the operation `A(B(x))`, where the output
/// of `B` serves as the input to `A`.
#[derive(Debug)]
pub struct CombinedMatrix<A: LinearOperations, B: LinearOperations> {
    /// The first linear operator (applied last in the composition).
    a: A,
    /// The second linear operator (applied first in the composition).
    b: B,
}

impl<A: LinearOperations, B: LinearOperations> CombinedMatrix<A, B> {
    /// Creates a new `CombinedMatrix` by combining two linear operators, `A` and `B`.
    ///
    /// # Parameters
    /// - `a`: The first linear operator.
    /// - `b`: The second linear operator.
    ///
    /// # Panics
    /// - Panics if the output dimensions of `B` do not match the input dimensions of `A`.
    pub fn new(a: A, b: B) -> Self {
        assert_eq!(b.n_out(), a.n_in(), "Matrix dimensions do not match");
        Self { a, b }
    }
}

impl<A: LinearOperations, B: LinearOperations> LinearOperations for CombinedMatrix<A, B> {
    fn n_in(&self) -> usize {
        self.b.n_in()
    }

    fn n_out(&self) -> usize {
        self.a.n_out()
    }

    /// Applies the combined operation `A(B(input))`.
    ///
    /// # Parameters
    /// - `input`: The input vector, with size `b.n_in()`.
    /// - `output`: The output vector, with size `a.n_out()`.
    ///
    /// # Panics
    /// - Panics if `input.len() != b.n_in()` or `output.len() != a.n_out()`.
    fn apply(&self, input: &[BinaryField128b], output: &mut [BinaryField128b]) {
        let mut tmp = vec![BinaryField128b::zero(); self.b.n_out()];
        self.b.apply(input, &mut tmp);
        self.a.apply(&tmp, output);
    }

    /// Applies the transposed combined operation `B^T(A^T(input))`.
    ///
    /// # Parameters
    /// - `input`: The input vector, with size `a.n_out()`.
    /// - `output`: The output vector, with size `b.n_in()`.
    ///
    /// # Panics
    /// - Panics if `input.len() != a.n_out()` or `output.len() != b.n_in()`.
    fn apply_transposed(&self, input: &[BinaryField128b], output: &mut [BinaryField128b]) {
        let mut tmp = vec![BinaryField128b::zero(); self.b.n_out()];
        self.a.apply_transposed(input, &mut tmp);
        self.b.apply_transposed(&tmp, output);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyLinear {
        size_in: usize,
        size_out: usize,
    }

    impl LinearOperations for DummyLinear {
        fn n_in(&self) -> usize {
            self.size_in
        }

        fn n_out(&self) -> usize {
            self.size_out
        }

        fn apply(&self, input: &[BinaryField128b], output: &mut [BinaryField128b]) {
            assert_eq!(input.len(), self.size_in);
            assert_eq!(output.len(), self.size_out);
            output.iter_mut().zip(input.iter()).for_each(|(o, i)| *o += *i);
        }

        fn apply_transposed(&self, input: &[BinaryField128b], output: &mut [BinaryField128b]) {
            assert_eq!(input.len(), self.size_out);
            assert_eq!(output.len(), self.size_in);
            output.iter_mut().zip(input.iter()).for_each(|(o, i)| *o += *i);
        }
    }

    #[test]
    fn test_combined_matrix_apply() {
        let a = DummyLinear { size_in: 3, size_out: 2 };
        let b = DummyLinear { size_in: 4, size_out: 3 };
        let combined = CombinedMatrix::new(a, b);

        let input = vec![BinaryField128b::from(11); 4];
        let mut output = vec![BinaryField128b::from(22); 2];

        combined.apply(&input, &mut output);

        // Manually compute expected output
        assert_eq!(output, vec![BinaryField128b::from(11) + BinaryField128b::from(22); 2]);
    }

    #[test]
    fn test_combined_matrix_apply_transposed() {
        let a = DummyLinear { size_in: 3, size_out: 2 };
        let b = DummyLinear { size_in: 4, size_out: 3 };
        let combined = CombinedMatrix::new(a, b);

        let input = vec![BinaryField128b::from(33); 2];
        let mut output = vec![BinaryField128b::from(44); 4];

        combined.apply_transposed(&input, &mut output);

        assert_eq!(
            output,
            vec![
                BinaryField128b::from(33) + BinaryField128b::from(44),
                BinaryField128b::from(33) + BinaryField128b::from(44),
                BinaryField128b::from(44),
                BinaryField128b::from(44)
            ]
        );
    }

    #[test]
    #[should_panic(expected = "Matrix dimensions do not match")]
    fn test_combined_matrix_invalid_dimensions() {
        let a = DummyLinear { size_in: 3, size_out: 2 };
        let b = DummyLinear { size_in: 5, size_out: 4 };
        CombinedMatrix::new(a, b);
    }

    #[test]
    fn test_combined_matrix_apply_and_transposed() {
        // **Create dummy linear transformations**
        // - `a` transforms an input of size 3 to an output of size 2.
        // - `b` transforms an input of size 4 to an output of size 3.
        let a = DummyLinear { size_in: 3, size_out: 2 };
        let b = DummyLinear { size_in: 4, size_out: 3 };

        // **Combine the transformations into a `CombinedMatrix`**
        // - `CombinedMatrix` models the operation `A(B(x))`.
        let combined = CombinedMatrix::new(a, b);

        // **Generate random input for `apply`**
        // - This represents the input to the combined operation `B(x)`.
        let input_apply: Vec<_> = (0..4).map(|_| BinaryField128b::random()).collect();

        // **Prepare output storage for the result of `apply`**
        // - This will store the output of `A(B(x))` (size 2).
        let mut output_apply = vec![BinaryField128b::from(0); 2];

        // **Apply the combined transformation**
        // - Compute `A(B(x))` from the random input `x`.
        combined.apply(&input_apply, &mut output_apply);

        // **Generate random input for `apply_transposed`**
        // - This represents the input to the transposed operation.
        let input_transposed: Vec<_> = (0..2).map(|_| BinaryField128b::random()).collect();

        // **Prepare output storage for the result of `apply_transposed`**
        // - This will store the output of `B^T(A^T(y))` (size 4).
        let mut output_transposed = vec![BinaryField128b::from(0); 4];

        // **Apply the transposed combined transformation**
        // - Compute `B^T(A^T(y))` from the random input `y`.
        combined.apply_transposed(&input_transposed, &mut output_transposed);

        // **Compute dot product of `apply` output and `apply_transposed` input**
        // - This computes `lhs = sum(A(B(x)) * y)`, where `A(B(x))` is the output of `apply`.
        let lhs = output_apply
            .iter()
            .zip(input_transposed.iter())
            .fold(BinaryField128b::from(0), |acc, (a, b)| acc + (*a * *b));

        // **Compute dot product of `apply_transposed` output and `apply` input**
        // - This computes `rhs = sum(x * B^T(A^T(y)))`.
        let rhs = output_transposed
            .iter()
            .zip(input_apply.iter())
            .fold(BinaryField128b::from(0), |acc, (a, b)| acc + (*a * *b));

        // **Validate the equality of `lhs` and `rhs`**
        // - **Mathematical justification**:
        //   - For a valid linear operator `T` and its transpose `T^T`, the following must hold:
        //     `<T(x), y> == <x, T^T(y)>`
        //   - Here, `T` is the combined operation `A(B(x))`, and `T^T` is `B^T(A^T(y))`.
        // - This equality ensures that the combined `apply` and `apply_transposed` methods maintain
        //   the transpose property.
        assert_eq!(lhs, rhs, "Dot product property violated for apply and apply_transposed");
    }
}
