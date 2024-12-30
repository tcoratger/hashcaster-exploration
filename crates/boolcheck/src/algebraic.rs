use hashcaster_field::binary_field::BinaryField128b;

/// A trait defining a generic algebraic evaluation behavior.
///
/// This trait abstracts the evaluation of bitwise operations or similar computations across
/// data slices. The specifics of the evaluation are defined by the implementor.
pub trait AlgebraicOps {
    /// The type of output produced by the algebraic evaluation.
    type AlgebraicOutput;

    /// The type of output produced by the linear compressed evaluation.
    type LinearCompressedOutput;

    /// The type of output produced by the quadratic compressed evaluation.
    type QuadraticCompressedOutput;

    /// Performs an algebraic evaluation based on the provided data and indices.
    ///
    /// Implementations define how the evaluation is conducted, which may involve operations
    /// across slices of binary field elements or other forms of computation. The exact
    /// behavior and return structure depend on the concrete implementor.
    ///
    /// ### Parameters
    /// - `data`: A slice of `BinaryField128b` elements serving as input data.
    /// - `idx_a`: The starting index for processing in the data slice.
    /// - `offset`: The step size used for computing additional indices during evaluation.
    ///
    /// ### Returns
    /// An output of type `Self::AlgebraicOutput`, representing the result of the algebraic
    /// evaluation.
    ///
    /// ### Notes
    /// - The specific computation performed is defined by the implementing type.
    /// - This trait allows for extensible and customizable evaluation strategies.
    fn algebraic(
        &self,
        data: &[BinaryField128b],
        idx_a: usize,
        offset: usize,
    ) -> Self::AlgebraicOutput;

    /// Performs a linear compressed evaluation on the provided data.
    ///
    /// This function applies a linear compression algorithm to the input slice of binary field
    /// elements. The specific compression method and output structure are defined by the
    /// implementing type.
    ///
    /// ### Parameters
    /// - `data`: A slice of `BinaryField128b` elements to be processed.
    ///
    /// ### Returns
    /// An output of type `Self::LinearCompressedOutput`, representing the result of the linear
    /// compression.
    ///
    /// ### Notes
    /// - The compression logic and the expected output format depend on the concrete
    ///   implementation.
    fn linear(&self, data: &[BinaryField128b]) -> Self::LinearCompressedOutput;

    /// Performs a quadratic compressed evaluation on the provided data.
    ///
    /// This function applies a quadratic compression algorithm to the input slice of binary field
    /// elements. The specific compression method and output structure are defined by the
    /// implementing type.
    ///
    /// ### Parameters
    /// - `data`: A slice of `BinaryField128b` elements to be processed.
    ///
    /// ### Returns
    /// An output of type `Self::QuadraticCompressedOutput`, representing the result of the
    /// quadratic compression.
    ///
    /// ### Notes
    /// - The compression logic and the expected output format depend on the concrete
    ///   implementation.
    fn quadratic(&self, data: &[BinaryField128b]) -> Self::QuadraticCompressedOutput;
}
