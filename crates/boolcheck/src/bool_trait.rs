use hashcaster_primitives::binary_field::BinaryField128b;

/// The `CompressedFoldedOps` trait defines an interface for executing compressed linear,
/// quadratic, and algebraic operations over input variables and data.
///
/// # Overview
/// This trait abstracts polynomial-based boolean operations, providing methods to:
/// - Execute linear part of boolean formulas and compress the results.
/// - Execute quadratic part of boolean formulas and compress the results.
/// - Apply algebraic transformations to data and compress the results.
pub trait CompressedFoldedOps<const I: usize>: Send + Sync {
    /// Executes the linear part of a boolean formula and compresses the results.
    fn linear_compressed(&self, inputs: &[BinaryField128b; I]) -> BinaryField128b;

    /// Executes the quadratic part of a boolean formula and compresses the results.
    fn quadratic_compressed(&self, inputs: &[BinaryField128b; I]) -> BinaryField128b;

    /// Applies an algebraic transformation to a subset of data and computes the compressed results.
    fn algebraic_compressed(
        &self,
        data: &[BinaryField128b],
        idx_a: usize,
        offset: usize,
    ) -> [BinaryField128b; 3];
}
