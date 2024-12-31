use std::ops::Index;

use hashcaster_field::binary_field::BinaryField128b;

/// The `CompressedFoldedOps` trait defines an interface for executing compressed linear,
/// quadratic, and algebraic operations over input variables and data.
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
/// # Explanation
/// The "compressed" form used in this trait computes results by iteratively summing pairs of
/// elements from the input slices. Specifically:
/// - For linear compression, elements at indices `2i` and `2i+1` are summed.
/// - For quadratic compression, products of elements at indices `2i` and `2i+1` are summed.
///
/// These compressed results represent the linear and quadratic parts of the same formula.
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
    ///
    /// # Details
    /// This function computes the sum of elements at indices `2i` and `2i+1` in the input slice,
    /// effectively collapsing the slice into a compressed representation.
    fn compress_linear(&self, inputs: &[BinaryField128b]) -> BinaryField128b;

    /// Computes the compressed result of a quadratic operation on the input variables.
    ///
    /// # Parameters
    /// - `inputs`: A slice of [`BinaryField128b`] values, representing the input variables.
    ///
    /// # Returns
    /// A single [`BinaryField128b`] value as the compressed result of the quadratic operation.
    ///
    /// # Details
    /// This function computes the sum of products of elements at indices `2i` and `2i+1` in the
    /// input slice, representing the quadratic part of the formula.
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
    ///
    /// # Details
    /// This method evaluates algebraic transformations on subsets of the input data using a defined
    /// stride, compressing the results into three components. It supports transformations for
    /// advanced boolean algebra scenarios.
    fn exec_alg(
        &self,
        data: [impl Index<usize, Output = BinaryField128b>; 4],
    ) -> [BinaryField128b; 3];
}
