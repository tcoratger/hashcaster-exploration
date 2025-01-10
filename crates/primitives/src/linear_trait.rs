use crate::binary_field::BinaryField128b;

/// A trait defining generic linear operations for transformations between input and output spaces.
pub trait LinearOperations {
    /// Returns the size of the input space for the transformation.
    ///
    /// # Details
    /// - This method provides the expected length of the `input` slice in the `apply` method and
    ///   the `output` slice in the `apply_transposed` method.
    fn n_in(&self) -> usize;

    /// Returns the size of the output space for the transformation.
    ///
    /// # Details
    /// - This method provides the expected length of the `output` slice in the `apply` method and
    ///   the `input` slice in the `apply_transposed` method.
    fn n_out(&self) -> usize;

    /// Applies the linear transformation from the input space to the output space.
    ///
    /// # Parameters
    /// - `input`: A slice of `BinaryField128b` elements of size `n_in`.
    /// - `output`: A mutable slice of `BinaryField128b` elements of size `n_out`.
    ///
    /// # Behavior
    /// - Adds the transformation result to the existing `output` slice using `+=`.
    /// - Supports cumulative updates to the output vector.
    ///
    /// # Panics
    /// - Panics if the length of `input` does not match `n_in`.
    /// - Panics if the length of `output` does not match `n_out`.
    fn apply(&self, input: &[BinaryField128b], output: &mut [BinaryField128b]);

    /// Applies the transposed linear transformation from the output space to the input space.
    ///
    /// # Parameters
    /// - `input`: A slice of `BinaryField128b` elements of size `n_out`.
    /// - `output`: A mutable slice of `BinaryField128b` elements of size `n_in`.
    ///
    /// # Behavior
    /// - Adds the transposed transformation result to the existing `output` slice using `+=`.
    /// - Supports cumulative updates to the output vector.
    ///
    /// # Panics
    /// - Panics if the length of `input` does not match `n_out`.
    /// - Panics if the length of `output` does not match `n_in`.
    fn apply_transposed(&self, input: &[BinaryField128b], output: &mut [BinaryField128b]);
}
