use hashcaster_field::binary_field::BinaryField128b;

/// The `CompressedFoldedOps` trait defines an interface for executing compressed linear, quadratic,
/// and algebraic operations over input variables and data.
///
/// # Overview
/// This trait abstracts polynomial-based boolean operations, providing methods to:
/// - Compress the results of linear boolean formulas.
/// - Compress the results of quadratic boolean formulas.
/// - Apply algebraic transformations to data and compress the results.
///
/// # Constants
/// - `N`: The number of input variables for the operations.
///
/// # Safety and Concurrency
/// The trait is constrained by `Send` and `Sync` for safe usage in concurrent environments.
pub trait CompressedFoldedOps: Send + Sync {
    /// Computes the compressed result of a linear operation on the input variables.
    ///
    /// # Parameters
    /// - `inputs`: A slice of [`BinaryField128b`] values, representing the input variables.
    ///
    /// # Returns
    /// A single [`BinaryField128b`] value as the compressed result of the linear operation.
    fn compress_linear(&self, inputs: &[BinaryField128b]) -> BinaryField128b;

    /// Computes the compressed result of a quadratic operation on the input variables.
    ///
    /// # Parameters
    /// - `inputs`: A slice of [`BinaryField128b`] values, representing the input variables.
    ///
    /// # Returns
    /// A single [`BinaryField128b`] value as the compressed result of the quadratic operation.
    fn compress_quadratic(&self, inputs: &[BinaryField128b]) -> BinaryField128b;

    /// Applies an algebraic transformation to a subset of data and computes the compressed results.
    ///
    /// # Parameters
    /// - `data`: A slice of [`BinaryField128b`] values representing the input data.
    /// - `start`: The starting index for accessing data within the slice.
    /// - `stride`: The step size for selecting subsequent data elements.
    ///
    /// # Returns
    /// An array of three [`BinaryField128b`] values as the compressed results of the algebraic
    /// operation.
    fn compress_algebraic(
        &self,
        data: &[BinaryField128b],
        start: usize,
        stride: usize,
    ) -> [BinaryField128b; 3];
}
